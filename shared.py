"""Shared data pipeline + matched evaluation harness for the clean hybrid-vs-baselines comparison.

The ARC data plumbing (RE-ARC cohort, episode sampling, grid tokenization) and the frozen FiLM
filler live in the original project at HYBRID_ARC. Rather than fork them (and risk drift), we bridge
to that single source of truth via sys.path and re-export exactly what the three solvers need.

Everything downstream (hybrid / ar / diffusion) imports from THIS module so all three see the
identical cohort, the identical train/heldout split, and -- crucially -- the IDENTICAL eval episodes.
`eval_episodes` samples one list of raw episodes per batch (fixed default_rng(999)) and hands every
solver BOTH representations of that same list:
    * filler format  (di/dpad/dt/dm, qi.., vi..)  -- what the hybrid filler/search consume
    * raw episodes   [(demos, qin, qout), ...]     -- what the AR / diffusion baselines tokenize
That is what makes the exact-accuracy numbers in README.md an apples-to-apples comparison.
"""
import os, sys, numpy as np, torch

# Original project (RE-ARC cohort + frozen filler checkpoint). Override with $HYBRID_ARC if relocated.
HYBRID_ARC = os.environ.get("HYBRID_ARC", "/project/flame/jgai/hybrid-ARC")
sys.path.insert(0, HYBRID_ARC)

from common import grids                                                          # noqa: E402
from common.model import Trunk, Block                                             # noqa: E402
from arc_bringup import build_cohort, sample_episode, MAXD, NCOL, VOCAB           # noqa: E402
from arc_hybrid import (FiLMARCFiller, search_z, cell_loss,                       # noqa: E402
                        in_ctx, cells_of, pad_ids, LIN, NCELL, AMP, BATCH)

PAD = grids.PAD
FILLER_CKPT = os.path.join(HYBRID_ARC, "ckpts", "arc_filler_film_bigz.pt")        # frozen EM-trained filler
UNCOND_CKPT = os.path.join(HYBRID_ARC, "ckpts", "cd_uncond.pt")                   # trained uncond denoiser (§12)

# Eval protocol shared by all three methods (matches arc_cond_diffusion.deploy):
EVAL_SEED = 999
EVAL_N    = 10        # number of eval batches
EVAL_BS   = 32        # episodes per batch  -> 320 query episodes total
K         = 4         # demos per episode
C         = 32        # test-time candidate budget (matched across methods)


def load_filler(dev):
    """The frozen executor/verifier. Trained earlier by EM (search z* / update filler); here read-only.
    Also returns the train/heldout task-id split baked into its checkpoint so every method uses it."""
    ck = torch.load(FILLER_CKPT, map_location=dev); A = ck["arch"]
    Lz, nd = A["Lz"], A["nd"]
    filler = FiLMARCFiller(A["d"], A["heads"], A["layers"], Lz, nd, LIN + NCELL + 4).to(dev)
    filler.load_state_dict(ck["filler"]); filler.eval()
    for p in filler.parameters():
        p.requires_grad_(False)
    return filler, ck["train"], ck["heldout"], Lz, nd


def n_params(module):
    return sum(p.numel() for p in module.parameters())


def _filler_tensors(episodes, k, dev):
    """Build the filler-format batch from a list of raw episodes (replicates arc_hybrid.build_batch)."""
    dctx, dt, dm, qctx, qt, qm, vctx, vt, vm, shapes = [], [], [], [], [], [], [], [], [], []
    for demos, qin, qout in episodes:
        for din, dout in demos:
            dctx.append(in_ctx(din)); t, m = cells_of(dout); dt.append(t); dm.append(m)
        qc = in_ctx(qin); t, m = cells_of(qout); qctx.append(qc); qt.append(t); qm.append(m)
        shapes.append(qout.shape)
        vd_in, vd_out = demos[-1]
        vctx.append(in_ctx(vd_in)); t, m = cells_of(vd_out); vt.append(t); vm.append(m)
    di = pad_ids(dctx, LIN, dev); qi = pad_ids(qctx, LIN, dev); vi = pad_ids(vctx, LIN, dev)
    tt = lambda a: torch.tensor(np.stack(a), device=dev)
    return ((di, di == PAD, tt(dt), tt(dm)), (qi, qi == PAD, tt(qt), tt(qm)),
            (vi, vi == PAD, tt(vt), tt(vm)), shapes)


def make_eval_batches(cohort, evalset, dev, eval_n=EVAL_N, eval_bs=EVAL_BS, k=K, seed=EVAL_SEED):
    """Deterministically sample eval_n batches of eval_bs episodes each. Returns a list of `pack` dicts.
    Each pack carries the SAME episodes in both representations:
        pack['episodes'] : [(demos, qin, qout), ...]            (raw, for AR/diffusion)
        pack['filler']   : ((di,dpad,dt,dm),(qi..),(vi..),shapes)  (for hybrid)
    """
    rng = np.random.default_rng(seed)
    packs = []
    for _ in range(eval_n):
        episodes = []
        for _ in range(eval_bs):
            tid = evalset[rng.integers(len(evalset))]
            episodes.append(sample_episode(rng, cohort[tid], k))   # (demos, qin, qout)
        packs.append({"episodes": episodes, "filler": _filler_tensors(episodes, k, dev)})
    return packs


def score(pred, shapes, qt, dev=None):
    """exact-match + cell accuracy of pred (B,NCELL) colors vs filler-format qt (B,NCELL), per shape."""
    B = len(shapes); ex = cl = 0.0
    for b in range(B):
        H, W = shapes[b]
        gt = qt[b].view(MAXD, MAXD)[:H, :W].cpu().numpy()
        gp = pred[b].view(MAXD, MAXD)[:H, :W].cpu().numpy()
        m = (gp == gt); cl += m.mean(); ex += float(m.all())
    return ex, cl, B


def eval_episodes(solver, cohort, evalset, dev, eval_n=EVAL_N, eval_bs=EVAL_BS, k=K, seed=EVAL_SEED):
    """Run `solver(pack) -> pred (B,NCELL)` over the shared fixed episode set; return exact/cell."""
    packs = make_eval_batches(cohort, evalset, dev, eval_n, eval_bs, k, seed)
    ex = cl = 0.0; tot = 0
    for pack in packs:
        (_, _, _, _), (_, _, qt, _), (_, _, _, _), shapes = pack["filler"]
        pred = solver(pack)
        e, c, n = score(pred, shapes, qt)
        ex += e; cl += c; tot += n
    return dict(exact=ex / tot, cell=cl / tot, n=tot)
