"""HYBRID (best design): unconditional latent prior p(z) + in-loop reconstruction guidance.

This is §12 of COND_DIFFUSION_GUIDED.md with the vestigial encoder *removed*. The realisation that
made it the best design: in the uncond variant the conditioning vector is hard-zeroed, so the
DemosEncoder never receives gradient and its output is discarded — it is dead weight. Dropping it,
the deployed model is exactly:

    frozen FiLM filler  (7.36M, EM-trained executor/verifier, read-only)
  + unconditional denoiser  (4.04M, the ONLY trained diffusion part)
  ----------------------------------------------------------------------
  = 11.39M total, and NO demo-conditioning network at all.

The demos enter the sampler through ONE channel only: the in-loop guidance gradient (DPS) of the
frozen filler's demo-reconstruction loss, folded into every low-noise reverse step. The prior just
paints "what a legal rule code looks like"; guidance steers it to *this* episode's rule.

We reuse the trained denoiser weights from cd_uncond.pt (the encoder weights in that ckpt are
ignored). Eval is the matched protocol in shared.py: C=32 candidates, demo-fit selection, TRAIN split.

Run:  python hybrid/hybrid.py            # deploy + report exact/cell on TRAIN
"""
import argparse, sys, os, numpy as np, torch, torch.nn as nn
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared import (grids, load_filler, build_cohort, search_z, cell_loss, eval_episodes,
                    n_params, AMP, UNCOND_CKPT, MAXD, NCELL, C as C_DEFAULT)


def make_schedule(T, dev):
    betas = torch.linspace(1e-4, 0.02, T, device=dev)
    alphas = 1 - betas
    abar = torch.cumprod(alphas, 0)
    return betas, alphas, abar


class UncondDenoiser(nn.Module):
    """eps_theta(z_t, t): MLP over [z_t ; time_emb ; ZERO-cond]. Architecturally identical to the
    CondDenoiser trained in cd_uncond.pt (so its weights load verbatim), but the conditioning slot is
    hard-wired to a zero vector here -- no encoder, no demos. The dim+h+hcond input width and the
    weights are kept so the trained ckpt loads; the hcond columns just multiply zeros."""
    def __init__(self, dim, h, T, hcond):
        super().__init__()
        self.hcond = hcond
        self.temb = nn.Embedding(T, h)
        self.net = nn.Sequential(nn.Linear(dim + h + hcond, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU(),
                                 nn.Linear(h, h), nn.SiLU(), nn.Linear(h, dim))

    def forward(self, z, t):
        zero = torch.zeros(z.shape[0], self.hcond, device=z.device, dtype=z.dtype)
        return self.net(torch.cat([z, self.temb(t), zero], -1))


def demo_guide_grad(filler, z_std, mu, sd, di, dpad, dt, dm, B, C, K, Lz, nd):
    """grad of demo cell_loss wrt STANDARDIZED z (B*C, DIM), looped per-candidate to bound memory.
    Same filler gradient search_z optimises; chain through zo=z*sd+mu lands the grad back in std space."""
    DIM = Lz * nd
    zv = z_std.view(B, C, DIM)
    gout = torch.zeros(B, C, DIM, device=z_std.device)
    for c in range(C):
        zc_std = zv[:, c, :].detach().clone().requires_grad_(True)
        zo = zc_std * sd + mu
        zc = zo.view(B, 1, Lz, nd).expand(B, K, Lz, nd).reshape(B * K, Lz, nd)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=AMP):
            logits = filler(zc, di, dpad, dt)
            loss = cell_loss(logits, dt, dm)
        gout[:, c, :] = torch.autograd.grad(loss, zc_std)[0]
    return gout.view(B * C, DIM)


def sample_uncond_guided(model, sched, filler, mu, sd, di, dpad, dt, dm, B, C, K, DIM, T, dev, Lz, nd,
                         gscale, gsteps, gtmax, gnorm):
    """Reverse DDPM from the unconditional prior, with in-loop demo-reconstruction guidance.
    Denoiser is called WITHOUT any conditioning (cond=0 inside UncondDenoiser). demos enter ONLY via
    the guidance gradient. Returns ORIGINAL-scale z (B*C, Lz, nd)."""
    betas, alphas, abar = sched
    z = torch.randn(B * C, DIM, device=dev)
    for t in reversed(range(T)):
        tt = torch.full((B * C,), t, dtype=torch.long, device=dev)
        with torch.no_grad():
            eps = model(z, tt)
        z0 = ((z - torch.sqrt(1 - abar[t]) * eps) / torch.sqrt(abar[t])).clamp(-4, 4)
        if gscale > 0 and t <= gtmax:                                  # fold demo-fit into low-noise steps
            for _ in range(gsteps):
                g = demo_guide_grad(filler, z0, mu, sd, di, dpad, dt, dm, B, C, K, Lz, nd)
                step = gscale * g / (g.norm(dim=-1, keepdim=True) + 1e-8) if gnorm else gscale * g
                z0 = (z0 - step).clamp(-4, 4)
        if t > 0:
            c1 = torch.sqrt(abar[t - 1]) * betas[t] / (1 - abar[t])
            c2 = torch.sqrt(alphas[t]) * (1 - abar[t - 1]) / (1 - abar[t])
            z = c1 * z0 + c2 * z + torch.sqrt(betas[t] * (1 - abar[t - 1]) / (1 - abar[t])) * torch.randn_like(z)
        else:
            z = z0
    return (z * sd + mu).view(B * C, Lz, nd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--C", type=int, default=C_DEFAULT)
    ap.add_argument("--T", type=int, default=100)
    ap.add_argument("--guide_scale", type=float, default=0.8)
    ap.add_argument("--guide_steps", type=int, default=4)
    ap.add_argument("--guide_tmax", type=int, default=50)
    ap.add_argument("--guide_norm", type=int, default=1)
    ap.add_argument("--eval_n", type=int, default=10)
    ap.add_argument("--heldout", action="store_true")
    cfg = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)

    filler, train, heldout, Lz, nd = load_filler(dev)
    DIM = Lz * nd
    cohort = build_cohort()
    sched = make_schedule(cfg.T, dev)

    cd = torch.load(UNCOND_CKPT, map_location=dev); A = cd["arch"]
    assert A.get("uncond"), "this ckpt is not the unconditional prior"
    model = UncondDenoiser(DIM, A["h"], A["T"], A["hcond"]).to(dev)
    model.load_state_dict(cd["model"]); model.eval()                   # encoder weights in cd are ignored
    mu = cd["mu"].to(dev); sd = cd["sd"].to(dev)

    print(f"[hybrid] filler(frozen)={n_params(filler)/1e6:.2f}M  denoiser={n_params(model)/1e6:.2f}M  "
          f"TOTAL={(n_params(filler)+n_params(model))/1e6:.2f}M  (encoder removed)", flush=True)

    evalset = heldout if cfg.heldout else train
    split = "HELDOUT" if cfg.heldout else "TRAIN"

    def solver(pack):
        (di, dpad, dt, dm), (qi, qpad, qt, qm), (vi, vpad, vt, vm), shapes = pack["filler"]
        B = len(shapes)
        zo = sample_uncond_guided(model, sched, filler, mu, sd, di, dpad, dt, dm, B, cfg.C, len(di) // B,
                                  DIM, cfg.T, dev, Lz, nd, cfg.guide_scale, cfg.guide_steps,
                                  cfg.guide_tmax, cfg.guide_norm)
        cand = zo.view(B, cfg.C, Lz, nd)
        K = len(di) // B
        dmb = dm.bool()
        best_z = torch.zeros(B, Lz, nd, device=dev); best_v = -np.ones(B)
        for c in range(cfg.C):                                          # demo-fit selection (matched criterion)
            zc = cand[:, c]
            zrep = zc.repeat_interleave(K, dim=0)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=AMP):
                pd = filler(zrep, di, dpad).argmax(-1)
            fit = (((pd == dt) & dmb).float().sum(1) / dmb.float().sum(1).clamp_min(1)).view(B, K).mean(1)
            vc = fit.cpu().numpy(); upd = vc > best_v
            mt = torch.tensor(upd, device=dev); best_z[mt] = zc[mt]; best_v[upd] = vc[upd]
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=AMP):
            pred = filler(best_z, qi, qpad).argmax(-1)                  # (B, NCELL)
        return pred

    res = eval_episodes(solver, cohort, evalset, dev, eval_n=cfg.eval_n)
    print(f"\n===== HYBRID (uncond p(z) + guidance s={cfg.guide_scale} steps={cfg.guide_steps} "
          f"tmax={cfg.guide_tmax}, C={cfg.C}, {split}) =====")
    print(f"  exact={res['exact']:.3f}  cell={res['cell']:.3f}  (n={res['n']})", flush=True)


if __name__ == "__main__":
    main()
