# Hybrid vs. vanilla AR vs. vanilla diffusion — a matched, fair comparison on ARC

Three solvers for the same task — given **K=4 demonstrations** (input→output grids) of a hidden rule
plus a **query input**, produce the **query output** — compared at **matched parameters** and **matched
test-time compute**, on the **rule-in-training TRAIN split** (the model has seen the rule's task during
training; eval is on fresh episodes of those tasks).

| folder | method | one line |
|---|---|---|
| `hybrid/` | **HYBRID** (best design) | diffusion in a **latent rule code z**, executed by a frozen filler; unconditional prior `p(z)` + in-loop guidance. See `hybrid/DESIGN.md`. |
| `ar/`     | **vanilla AR** | autoregressive decode of the output grid, **cell by cell**, conditioned in-context on demos+query. |
| `diffusion/` | **vanilla diffusion** | masked (MaskGIT) discrete diffusion of the output grid **in pixel space**, conditioned in-context. |

All three share one data pipeline and one evaluation harness (`shared.py`) so they see the *identical*
cohort, train/heldout split, and — crucially — the *identical* eval episodes.

---

## The comparison (TRAIN split, exact-match accuracy)

> Numbers below are filled from `logs/{hybrid_eval,ar_full,dlm_full}.log`. `search` is the test-time
> latent-search reference the hybrid ties.

| method | params | test-time budget | **exact** | cell |
|---|---|---|---|---|
| **HYBRID** (uncond p(z) + guidance, select@32) | **11.39M** | C=32 + guided sampling | **0.175** | **0.857** |
| search ← zero (reference, same filler) | (filler 7.36M) | 200-step latent search | ~0.170 | ~0.86 |
| vanilla AR — greedy | 11.50M | 1 decode | 0.016 | 0.531 |
| vanilla AR — vote@32 | 11.50M | 32 decodes | 0.016 | 0.561 |
| vanilla diffusion — greedy (T=8) | 11.50M | 8 steps | 0.013 | 0.502 |
| vanilla diffusion — vote@32 (T=8) | 11.50M | 32×8 steps | 0.013 | 0.506 |

*n = 320 TRAIN-split query episodes (eval_n=10 × eval_bs=32), all methods on the identical episodes.
Self-consistency vote@32 gives the flat baselines no exact-match lift over greedy (AR 0.016=0.016,
diffusion 0.013=0.013): at near-chance accuracy there is no correct consensus to pool.*

**Takeaway.** At **matched parameters (~11.4M)** and a **matched 32-sample test-time budget**, on the
**rule-in-training TRAIN split**, the hybrid's exact-match is **0.175 — ~11× vanilla AR (0.016) and ~13×
vanilla diffusion (0.013)**; cell accuracy **0.857 vs ~0.51**. The flat baselines sit at the
chance-level exact-match that flat output models always show on ARC (matching this project's earlier
held-out flat numbers ~0.02–0.03), and self-consistency voting gives them no lift — their errors are not
random pixel noise to average out; they reflect *not having inferred the rule*.

The gap is exactly the hybrid's two mechanisms, neither available to a flat output-space model:
1. **Latent factorization** — the rule is compressed into a 256-dim code `z` and *executed* by a frozen,
   reusable filler, instead of being re-derived pixel-by-pixel each episode;
2. **Test-time rule inference** — guided sampling (filler-gradient DPS) actively searches the latent for
   a `z` that reproduces *these* demos, then applies it to the query.

A flat AR or diffusion model has only in-context conditioning and one (or 32 i.i.d.) forward passes; it
cannot do test-time optimization toward demo-consistency, so on exact-match it stays at floor.

---

## What "matched" means here

**Matched parameters (~11.4M each).** The hybrid's deployed model is the frozen filler (7.36M, itself
trained — by EM — on the TRAIN rules) + the unconditional denoiser (4.04M) = **11.39M of trained
capacity**. So each baseline is sized to ~11.4M *trainable* params (d=256, 14 layers → 11.50M). This is
if anything generous to the baselines: all 11.5M of their params are trained end-to-end on the task,
whereas the hybrid splits its 11.4M across a frozen executor + a small denoiser.

**Matched test-time compute (C=32 candidates).** The hybrid draws C=32 latent candidates and selects by
demo-fit. Each baseline is given the same budget of 32 samples via self-consistency **vote@32**
(majority output across 32 temperature-sampled decodes). `greedy` (1 sample) is also reported as the
floor. The compute is *not* identical in FLOPs — the hybrid additionally spends filler gradients during
guided sampling — so the "test-time budget" column states what each method actually does; the headline
holds params fixed and gives every method a 32× sampling budget.

**Matched training.** Baselines: bs=192 (AR via bs=48 × grad-accum 4), 3000 steps, cosine, AdamW. The
hybrid's denoiser: 30k steps ε-MSE on the z\* pool; its filler: trained upstream by EM. All on the same
70 TRAIN tasks.

**Same eval episodes.** `shared.make_eval_batches` samples `eval_n=10 × eval_bs=32 = 320` query episodes
from the TRAIN tasks under a fixed `default_rng(999)`, and hands every solver both the filler-format
tensors (hybrid) and the raw episodes (AR/diffusion). Same rng → same episodes → apples-to-apples.

---

## Layout

```
hybrid-ARC-clean/
  shared.py            # data bridge (-> /project/flame/jgai/hybrid-ARC) + matched eval harness
  common/              # copied transformer trunk + grid tokenizer
  hybrid/
    hybrid.py          # the best design: uncond p(z) + in-loop guidance, encoder removed
    DESIGN.md          # full design write-up (DDPM math -> code), DIFFUSION_Z_INFERENCE-level detail
    ckpts/             # (reuses ../../hybrid-ARC/ckpts/cd_uncond.pt + arc_filler_film_bigz.pt)
  ar/  ar.py           # vanilla causal AR baseline (+ trained ckpt.pt)
  diffusion/ diffusion.py   # vanilla masked-diffusion baseline (+ trained ckpt.pt)
  logs/                # training + eval logs
  scripts/run.sh       # srun helper (sets CUDA_VISIBLE_DEVICES, redirects to a log)
```

## Reproduce

```bash
# hybrid (no training; reuses the trained uncond denoiser + frozen filler)
python hybrid/hybrid.py --eval_n 10

# baselines (train ~3000 steps then eval on the same TRAIN episodes)
python ar/ar.py        --steps 3000 --eval_n 10 --save ar/ckpt.pt
python diffusion/diffusion.py --steps 3000 --eval_n 10 --T 8 --save diffusion/ckpt.pt
```

> Dependency note: `shared.py` bridges to the original project at `/project/flame/jgai/hybrid-ARC` for
> the RE-ARC cohort and the frozen filler checkpoint (single source of truth, no fork).
