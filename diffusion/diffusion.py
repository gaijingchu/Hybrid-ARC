"""VANILLA DIFFUSION baseline: masked discrete (MaskGIT-style) diffusion over the output grid.

Contrast with the hybrid (diffusion in a LATENT rule-code z, executed by a frozen filler) and with the
AR baseline (one cell at a time, left-to-right). Here diffusion happens directly in OUTPUT (pixel)
space: the 144 output cells start fully masked and are revealed over T denoising steps, most-confident
first. No latent z, no filler, no search -- conditioning is purely in-context (read demos+qin).

  * forward (training) corruption = absorbing-state masking: each valid cell masked w.p. r~U(0,1).
  * model = bidirectional trunk over [demos+qin context ; 144 output-cell slots], each slot embeds its
    CURRENT (masked/colored) token + a learned 2D position; head -> per-cell color logits.
  * inference = T-step confidence-based unmasking (MaskGIT). T is the test-time compute knob.

Trained on TRAIN tasks; evaluated on fresh TRAIN-split episodes, matched params (~11.4M to the hybrid),
matched C=32 candidate budget (greedy over T steps + self-consistency vote@32).

Run (train+eval):  python diffusion/diffusion.py --steps 3000 --save diffusion/ckpt.pt
     (eval only):   python diffusion/diffusion.py --load diffusion/ckpt.pt
"""
import argparse, sys, os, time, math, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared import (grids, Trunk, build_cohort, sample_episode, eval_episodes, n_params, load_filler,
                    MAXD, NCOL, VOCAB, NCELL, AMP, C as C_DEFAULT)

def ctx_max_for(K): return (K * 2 + 1) * (grids.grid_len(MAXD, MAXD) + 2) + 4


class ARCFlatDLM(nn.Module):
    """Bidirectional trunk over [context ; 144 cell slots]; slot = tok(current cell state) + cell_pos."""
    def __init__(self, d, heads, layers, ctx_max):
        super().__init__()
        self.tok = nn.Embedding(VOCAB, d)                          # shared vocab incl. MASKG
        self.cell_pos = nn.Parameter(torch.randn(NCELL, d) * 0.02)
        self.trunk = Trunk(d, heads, layers, ctx_max + NCELL)
        self.head = nn.Linear(d, NCOL)

    def forward(self, ctx_ids, ctx_pad, cell_ids):
        B = ctx_ids.shape[0]
        e = self.tok(ctx_ids)
        cells = self.tok(cell_ids) + self.cell_pos[None]
        x = torch.cat([e, cells], 1)
        kpm = torch.cat([ctx_pad, torch.zeros(B, NCELL, dtype=torch.bool, device=ctx_ids.device)], 1)
        h = self.trunk(x, causal=False, key_padding_mask=kpm)[:, -NCELL:]
        return self.head(h)


def ctx_tensors(episodes, ctx_max, dev):
    ctx, tgt, valid = [], [], []
    for demos, qin, qout in episodes:
        ctx.append(grids.build_context(demos, qin))
        H, W = qout.shape
        t = np.zeros(NCELL, dtype=np.int64); m = np.zeros(NCELL, dtype=bool)
        for r in range(H):
            for c in range(W):
                t[r * MAXD + c] = qout[r, c]; m[r * MAXD + c] = True
        tgt.append(t); valid.append(m)
    L = min(max(len(s) for s in ctx), ctx_max)                     # dynamic pad to batch-max
    ids = torch.tensor([s[:L] + [grids.PAD] * (L - len(s)) for s in ctx], dtype=torch.long, device=dev)
    return ids, ids == grids.PAD, torch.tensor(np.stack(tgt), device=dev), torch.tensor(np.stack(valid), device=dev)


def corrupt(tgt, valid, dev):
    """absorbing-state forward: per-row mask ratio r~U(0,1); >=1 masked valid cell guaranteed."""
    B = tgt.shape[0]
    r = torch.rand(B, 1, device=dev); u = torch.rand(B, NCELL, device=dev)
    maskpos = (u < r) & valid
    none = (maskpos.sum(1) == 0) & (valid.sum(1) > 0)
    if none.any():
        for b in torch.nonzero(none, as_tuple=False).flatten():
            j = torch.nonzero(valid[b], as_tuple=False).flatten()[0]; maskpos[b, j] = True
    cell_ids = torch.full((B, NCELL), grids.PAD, dtype=torch.long, device=dev)
    cell_ids[valid] = tgt[valid] + grids.N_SPECIAL
    cell_ids[maskpos] = grids.MASKG
    return cell_ids, maskpos


@torch.no_grad()
def decode(model, ids, cpad, valid, T, dev, temp=0.0):
    """MaskGIT: start all valid cells = MASKG, reveal most-confident on a cosine schedule over T steps."""
    B = ids.shape[0]
    cell_ids = torch.full((B, NCELL), grids.PAD, dtype=torch.long, device=dev)
    cell_ids[valid] = grids.MASKG
    committed = torch.zeros(B, NCELL, dtype=torch.bool, device=dev)
    final = torch.zeros(B, NCELL, dtype=torch.long, device=dev)
    nvalid = valid.sum(1); pred = torch.zeros(B, NCELL, dtype=torch.long, device=dev)
    for step in range(T):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=AMP):
            logits = model(ids, cpad, cell_ids).float()
        if temp > 0:
            probs = F.softmax(logits / temp, -1)
            pred = torch.distributions.Categorical(probs=probs).sample()
            conf = probs.gather(-1, pred[..., None]).squeeze(-1)
            g = -torch.log(-torch.log(torch.rand_like(conf).clamp_min(1e-9)).clamp_min(1e-9))
            sc = conf + temp * g
        else:
            probs = F.softmax(logits, -1); conf, pred = probs.max(-1); sc = conf.clone()
        still = valid & ~committed
        sc = torch.where(still, sc, torch.full_like(sc, -1e9))
        keep_frac = math.cos(math.pi / 2 * (step + 1) / T)
        for b in range(B):
            nv = int(nvalid[b].item()); n_keep = int(math.floor(keep_frac * nv))
            n_new = min((nv - n_keep) - int(committed[b].sum().item()), int(still[b].sum().item()))
            if n_new <= 0:
                continue
            idx = torch.topk(sc[b], n_new).indices
            cell_ids[b, idx] = pred[b, idx] + grids.N_SPECIAL; committed[b, idx] = True; final[b, idx] = pred[b, idx]
    rem = valid & ~committed; final[rem] = pred[rem]
    return final


def vote(preds):
    return F.one_hot(preds, NCOL).sum(0).argmax(-1)                # (B,N) modal color over C decodes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=256); ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--layers", type=int, default=14)              # -> ~11.4M to match hybrid
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--steps", type=int, default=3000); ap.add_argument("--bs", type=int, default=192)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--T", type=int, default=8)                    # MaskGIT decode steps
    ap.add_argument("--C", type=int, default=C_DEFAULT); ap.add_argument("--cand_chunk", type=int, default=8)
    ap.add_argument("--eval_n", type=int, default=10); ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--heldout", action="store_true")
    ap.add_argument("--save", default=""); ap.add_argument("--load", default="")
    cfg = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = True; torch.manual_seed(0)

    _, train, heldout, _, _ = load_filler(dev)                     # same train/heldout split
    cohort = build_cohort()
    ctx_max = ctx_max_for(cfg.K)
    model = ARCFlatDLM(cfg.d, cfg.heads, cfg.layers, ctx_max).to(dev)
    print(f"[dlm] params={n_params(model)/1e6:.2f}M  (d={cfg.d} heads={cfg.heads} layers={cfg.layers})", flush=True)

    if cfg.load:
        model.load_state_dict(torch.load(cfg.load, map_location=dev)["model"]); model.eval()
        print(f"[load] {cfg.load}", flush=True)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg.steps)
        rng = np.random.default_rng(0); t0 = time.time()
        for step in range(cfg.steps):
            model.train()
            eps = [sample_episode(rng, cohort[train[rng.integers(len(train))]], cfg.K) for _ in range(cfg.bs)]
            ids, cpad, tgt, valid = ctx_tensors(eps, ctx_max, dev)
            cell_ids, maskpos = corrupt(tgt, valid, dev)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=AMP):
                logits = model(ids, cpad, cell_ids)
                loss = F.cross_entropy(logits[maskpos], tgt[maskpos])
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            if step % max(cfg.steps // 12, 1) == 0:
                print(f"[dlm {step}] loss={loss.item():.3f} ({time.time()-t0:.0f}s)", flush=True)
        model.eval()
        if cfg.save:
            torch.save({"model": model.state_dict(),
                        "arch": {"d": cfg.d, "heads": cfg.heads, "layers": cfg.layers, "ctx_max": ctx_max}}, cfg.save)
            print(f"[save] {cfg.save}", flush=True)

    evalset = heldout if cfg.heldout else train
    split = "HELDOUT" if cfg.heldout else "TRAIN"

    def solver_greedy(pack):
        ids, cpad, _, valid = ctx_tensors(pack["episodes"], ctx_max, dev)
        return decode(model, ids, cpad, valid, cfg.T, dev, temp=0.0)

    def solver_vote(pack):
        ids, cpad, _, valid = ctx_tensors(pack["episodes"], ctx_max, dev); B = ids.shape[0]
        preds = []
        for c0 in range(0, cfg.C, cfg.cand_chunk):
            nc = min(cfg.cand_chunk, cfg.C - c0)
            ii = ids.repeat_interleave(nc, 0); pp = cpad.repeat_interleave(nc, 0); vv = valid.repeat_interleave(nc, 0)
            d = decode(model, ii, pp, vv, cfg.T, dev, temp=cfg.temp).view(B, nc, NCELL)
            preds.append(d.permute(1, 0, 2))
        return vote(torch.cat(preds, 0))

    rg = eval_episodes(solver_greedy, cohort, evalset, dev, eval_n=cfg.eval_n)
    print(f"\n===== VANILLA DIFFUSION ({split}) =====")
    print(f"  greedy T={cfg.T}     exact={rg['exact']:.3f}  cell={rg['cell']:.3f}  (n={rg['n']})", flush=True)
    rv = eval_episodes(solver_vote, cohort, evalset, dev, eval_n=cfg.eval_n)
    print(f"  vote@{cfg.C} T={cfg.T} (temp={cfg.temp})  exact={rv['exact']:.3f}  cell={rv['cell']:.3f}", flush=True)


if __name__ == "__main__":
    main()
