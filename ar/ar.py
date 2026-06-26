"""VANILLA AR baseline: a genuinely autoregressive decoder over the output grid.

Contrast with the hybrid (latent diffusion + guidance) and the diffusion baseline (parallel iterative
denoising). Here the output grid is generated CELL BY CELL in row-major order: to predict cell i the
model attends, causally, to the full demos+query-input context plus every already-decoded cell < i.

  * sequence = [ demos+qin context tokens ; 144 output-cell slots ]
  * the whole sequence is CAUSAL (decoder-only LM); cell slot i carries (embedding of cell i-1's color
    + a learned absolute position for cell i) and predicts cell i.
  * one shot of teacher forcing trains it; inference is a 144-step left-to-right decode.

No latent z, no search, no filler. Conditioning is purely in-context (read the demos, emit the answer).
Trained on the TRAIN tasks; evaluated on fresh TRAIN-split episodes (rule-in-training), matched params,
matched C=32 test-time candidate budget (greedy + self-consistency vote@32).

Run (train+eval):  python ar/ar.py --steps 3000 --save ar/ckpt.pt
     (eval only):   python ar/ar.py --load ar/ckpt.pt
"""
import argparse, sys, os, time, math, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared import (grids, Trunk, build_cohort, sample_episode, eval_episodes, n_params, load_filler,
                    MAXD, NCOL, VOCAB, NCELL, AMP, C as C_DEFAULT)

START = grids.MASKG                                                  # cell-input token for position 0 / unknown
def ctx_max_for(K): return (K * 2 + 1) * (grids.grid_len(MAXD, MAXD) + 2) + 4


class ARCausal(nn.Module):
    """Decoder-only causal LM over [context tokens ; 144 cell slots] -> per-cell color logits.
    Cell slot i = tok(prev cell color) + learned cell_pos[i]; causal mask over the whole sequence."""
    def __init__(self, d, heads, layers, ctx_max):
        super().__init__()
        self.tok = nn.Embedding(VOCAB, d)
        self.cell_pos = nn.Parameter(torch.randn(NCELL, d) * 0.02)
        self.trunk = Trunk(d, heads, layers, ctx_max + NCELL)
        self.head = nn.Linear(d, NCOL)
        self.ctx_max = ctx_max

    def forward(self, ctx_ids, ctx_pad, cell_in):
        B = ctx_ids.shape[0]
        e = self.tok(ctx_ids)                                       # (B, Lc, d)
        cells = self.tok(cell_in) + self.cell_pos[None]             # (B, 144, d)
        x = torch.cat([e, cells], 1)
        kpm = torch.cat([ctx_pad, torch.zeros(B, NCELL, dtype=torch.bool, device=ctx_ids.device)], 1)
        h = self.trunk(x, causal=True, key_padding_mask=kpm)[:, -NCELL:]
        return self.head(h)                                         # (B, 144, NCOL)


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
    L = min(max(len(s) for s in ctx), ctx_max)                     # dynamic pad to batch-max (bounds B*L^2 memory)
    ids = torch.tensor([s[:L] + [grids.PAD] * (L - len(s)) for s in ctx], dtype=torch.long, device=dev)
    return ids, ids == grids.PAD, torch.tensor(np.stack(tgt), device=dev), torch.tensor(np.stack(valid), device=dev)


def teacher_in(tgt):
    """cell_in[i] = color-id of cell i-1 (teacher forcing); cell_in[0] = START."""
    prev = torch.full_like(tgt, START)
    prev[:, 1:] = tgt[:, :-1] + grids.N_SPECIAL
    return prev


@torch.no_grad()
def decode(model, ids, cpad, dev, temp=0.0):
    """144-step left-to-right AR decode. Returns (B, NCELL) predicted colors."""
    B = ids.shape[0]
    cell_in = torch.full((B, NCELL), START, dtype=torch.long, device=dev)
    pred = torch.zeros(B, NCELL, dtype=torch.long, device=dev)
    for i in range(NCELL):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=AMP):
            logits = model(ids, cpad, cell_in).float()[:, i]        # (B, NCOL)
        if temp > 0:
            ci = torch.distributions.Categorical(logits=logits / temp).sample()
        else:
            ci = logits.argmax(-1)
        pred[:, i] = ci
        if i + 1 < NCELL:
            cell_in[:, i + 1] = ci + grids.N_SPECIAL
    return pred


def vote(preds):
    """self-consistency: per cell, majority color across the C candidate decodes. preds: (C,B,NCELL)."""
    oh = F.one_hot(preds, NCOL).sum(0)                              # (B,N,NCOL) per-color counts
    return oh.argmax(-1)                                            # (B,N) modal color


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=256); ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--layers", type=int, default=14)               # -> ~11.4M to match hybrid
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--steps", type=int, default=3000); ap.add_argument("--bs", type=int, default=48)
    ap.add_argument("--accum", type=int, default=4)                 # bs*accum=192 effective (matches diffusion)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--C", type=int, default=C_DEFAULT); ap.add_argument("--cand_chunk", type=int, default=4)
    ap.add_argument("--eval_n", type=int, default=10); ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--heldout", action="store_true")
    ap.add_argument("--save", default=""); ap.add_argument("--load", default="")
    cfg = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = True; torch.manual_seed(0)

    _, train, heldout, _, _ = load_filler(dev)                     # reuse the SAME train/heldout split
    cohort = build_cohort()
    ctx_max = ctx_max_for(cfg.K)
    model = ARCausal(cfg.d, cfg.heads, cfg.layers, ctx_max).to(dev)
    print(f"[ar] params={n_params(model)/1e6:.2f}M  (d={cfg.d} heads={cfg.heads} layers={cfg.layers})", flush=True)

    if cfg.load:
        model.load_state_dict(torch.load(cfg.load, map_location=dev)["model"]); model.eval()
        print(f"[load] {cfg.load}", flush=True)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg.steps)
        rng = np.random.default_rng(0); t0 = time.time()
        for step in range(cfg.steps):
            model.train(); opt.zero_grad(); tot = 0.0
            for _ in range(cfg.accum):                              # grad-accum -> effective bs = bs*accum
                eps = [sample_episode(rng, cohort[train[rng.integers(len(train))]], cfg.K) for _ in range(cfg.bs)]
                ids, cpad, tgt, valid = ctx_tensors(eps, ctx_max, dev)
                cell_in = teacher_in(tgt)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=AMP):
                    logits = model(ids, cpad, cell_in)
                    loss = F.cross_entropy(logits.reshape(-1, NCOL), tgt.reshape(-1)) / cfg.accum  # all 144 cells
                loss.backward(); tot += loss.item()
            opt.step(); sched.step()
            if step % max(cfg.steps // 12, 1) == 0:
                print(f"[ar {step}] loss={tot:.3f} ({time.time()-t0:.0f}s)", flush=True)
        model.eval()
        if cfg.save:
            torch.save({"model": model.state_dict(),
                        "arch": {"d": cfg.d, "heads": cfg.heads, "layers": cfg.layers, "ctx_max": ctx_max}}, cfg.save)
            print(f"[save] {cfg.save}", flush=True)

    evalset = heldout if cfg.heldout else train
    split = "HELDOUT" if cfg.heldout else "TRAIN"

    def build_ctx(pack):
        ids, cpad, _, _ = ctx_tensors(pack["episodes"], ctx_max, dev)
        return ids, cpad

    def solver_greedy(pack):
        ids, cpad = build_ctx(pack)
        return decode(model, ids, cpad, dev, temp=0.0)

    def solver_vote(pack):
        ids, cpad = build_ctx(pack); B = ids.shape[0]
        preds = []
        for c0 in range(0, cfg.C, cfg.cand_chunk):
            nc = min(cfg.cand_chunk, cfg.C - c0)
            ii = ids.repeat_interleave(nc, 0); pp = cpad.repeat_interleave(nc, 0)
            d = decode(model, ii, pp, dev, temp=cfg.temp).view(B, nc, NCELL)
            preds.append(d.permute(1, 0, 2))                         # (nc,B,NCELL)
        return vote(torch.cat(preds, 0))                            # (B,NCELL)

    rg = eval_episodes(solver_greedy, cohort, evalset, dev, eval_n=cfg.eval_n)
    print(f"\n===== VANILLA AR ({split}) =====")
    print(f"  greedy        exact={rg['exact']:.3f}  cell={rg['cell']:.3f}  (n={rg['n']})", flush=True)
    rv = eval_episodes(solver_vote, cohort, evalset, dev, eval_n=cfg.eval_n)
    print(f"  vote@{cfg.C} (temp={cfg.temp})  exact={rv['exact']:.3f}  cell={rv['cell']:.3f}", flush=True)


if __name__ == "__main__":
    main()
