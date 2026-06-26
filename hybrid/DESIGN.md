# HYBRID design: unconditional latent prior `p(z)` + in-loop guidance — from DDPM to code

> This is the "best design" the comparison settles on (§12 of the original `COND_DIFFUSION_GUIDED.md`,
> with the vestigial encoder removed). It is the same algorithm as the older `DIFFUSION_Z_INFERENCE.md`
> — unconditional prior, demos enter only via guidance — but in the 256-dim bigz regime with the
> upgraded (multi-step / low-t / strong) guidance, evaluated on the rule-in-training TRAIN split.
> This document is self-contained and at the same level of detail: every reverse-sampling line is
> traced from the DDPM math down to `hybrid.py`.
>
> Prereqs: **z is a 256-dim continuous latent rule code** (`Lz=16 × nd=16`); the **filler** is a frozen
> executor `filler(z, input_grid) → per-cell color logits`; the unconditional denoiser
> `model(z_t, t) → ε` was trained on standardized z (it is `CondDenoiser` with the conditioning slot
> hard-zeroed — see §4). All checkpoints are produced upstream; here they are read-only.

---

## 0. What inference has in hand

```
frozen denoiser  model(z_t, t)   ← learned UNCONDITIONAL prior p(z); never reads demos
frozen filler    filler(z, x)    ← executor + verifier (EM-trained, then frozen)
mu, sd                            ← per-dim mean/std of the z* pool (standardization dictionary)
one TRAIN episode: K=4 demos (in→out) + a query input
goal: produce z such that filler(z, query_input) emits the right answer grid
```

**Core convention (identical to `DIFFUSION_Z_INFERENCE.md`): demos are NOT fed to the denoiser.**
`model` only ever sees `(z_t, t)`. The demos enter sampling through ONE side channel — the **guidance
gradient** (§5). That is the whole point: no demo-conditioning network, yet the sampler still infers
*this* episode's rule.

---

## 1. The deployed model (encoder removed)

```
frozen FiLM filler          7.36M   (EM-trained executor/verifier, read-only)
unconditional denoiser      4.04M   (the ONLY trained diffusion part)
--------------------------------------------------------------------------
HYBRID TOTAL               11.39M    and NO demo-conditioning network at all
```

The conditional variants of the original project paid 2.6M for a `DemosEncoder` (cross-attention
conditioning). **In the unconditional variant that encoder is dead weight** (§4 proves it: its output
is multiplied by zero and it never receives gradient), so it is deleted here. What remains is the
leanest design that still *ties* test-time search.

---

## 2. DDPM in one screen (the engine both prior and guidance ride on)

Forward (add noise), closed form from clean `z_0` to level `t`:

```
z_t = √ᾱ_t · z_0 + √(1-ᾱ_t) · ε,    ε ~ N(0,I)        … (1)
```
`betas = linspace(1e-4, 0.02, T=100)`, `alphas = 1-betas`, `ᾱ_t = ∏ alphas` (`hybrid.py:make_schedule`).

Denoiser training target: predict the noise, `loss = ‖model(z_t,t) − ε‖²` — so `model(z_t,t)` learns
"given a noised vector and its noise level, what noise is in it".

Reverse (one step) uses the posterior `q(z_{t-1}|z_t,z_0)`, a closed-form Gaussian whose mean is a
weighted average of the predicted clean `ẑ0` and the current `z_t`:

```
ẑ0   = (z_t − √(1-ᾱ_t)·ε) / √ᾱ_t                       … (4) predicted clean z
c1   = √ᾱ_{t-1}·β_t /(1-ᾱ_t)                            (z0 weight)
c2   = √α_t·(1-ᾱ_{t-1})/(1-ᾱ_t)                         (zt weight)
z_{t-1} = c1·ẑ0 + c2·z_t + √(β̃_t)·ξ,  ξ~N(0,I)         … (2,3) posterior sample
```

(Full Bayesian derivation of `c1,c2,β̃_t`: see `DIFFUSION_Z_INFERENCE.md` §3. They are pure
functions of the β/α/ᾱ schedule.) Unconditional sampling = iterate ①predict ε → ②get ẑ0 → ③posterior
sample, from `t=99` down to `0`. That alone produces "a vector that looks like a legal rule code",
**not** this episode's rule. §5 fixes that.

---

## 3. Training Stage A — build the z\* pool (frozen filler + search)

For each TRAIN episode, `search_z` runs Adam on the latent to minimise the demo-reconstruction
`cell_loss` (the filler stays frozen; only z moves), yielding a z\* that reproduces the K demos.
Store `(demos, z*)`. This is the SAME pool the conditional variants use; **`--uncond` changes nothing
here.** Pool stats (logged): `N≈19200`, `|mu|≈7.39`, `sd.mean≈0.778`. Standardize:

```
mu = Z.mean(0);  sd = Z.std(0)         # the only bridge between filler-scale and DDPM-scale
Zs = (Z - mu) / sd                     # DDPM lives in this N(0,1) space
```

The filler itself was produced earlier by an **EM / auto-decoder** loop (inner: search z\* on demos;
outer: update filler so that z\* generalizes to the held-out query). By the time we are here, that EM
is finished and the filler is frozen — Stage A just re-searches z\* against the final filler.

---

## 4. Training Stage B — standard ε-MSE, but the condition is hard-zeroed

```python
# the ONLY change from a conditional denoiser: cond ≡ 0
z0  = Zs[idx]                                   # clean standardized z*
t   = randint(0, T); eps = randn_like(z0)
zt  = √ᾱ_t·z0 + √(1-ᾱ_t)·eps                    # (1)
eps_hat = model(zt, t)                          # UncondDenoiser: cond slot is a zero vector
loss = mse(eps_hat, eps)                         # plain ε-MSE, no tricks
```

`UncondDenoiser` (`hybrid.py`) is architecturally `CondDenoiser(dim, h, T, hcond)` — an MLP over
`[z_t ; time_emb ; cond]` — but `cond` is hard-wired to `torch.zeros(B, hcond)`. The trained weights
from `cd_uncond.pt` load verbatim (the `hcond` input columns just multiply zeros).

**Why the encoder is provably dead weight.** In the upstream uncond training, `build_cond` computed
`c = encoder(demos)` and then returned `torch.zeros_like(c)`. Consequences:
1. the denoiser input is a zero vector → the encoder's output is discarded;
2. `∂loss/∂c = 0` → the encoder receives **no gradient**, ever; its 2.6M params never move.
So the encoder is a pure code artifact (it only existed to give `zeros_like` a shape). Removing it —
as here — leaves the model behaviourally identical and 2.6M lighter. The denoiser learns the **marginal**
`p(z)` over the whole pool: "what a legal rule code looks like", with zero episode-specific information.
(Its ε-MSE floor, ~0.38, is higher than a conditional denoiser's — expected, the marginal has more
variance — but this does not hurt sampling, because guidance supplies the episode information.)

---

## 5. Inference — guided reverse sampling (`hybrid.py:sample_uncond_guided`)

Demos enter ONLY here. One reverse step:

```
① eps = model(z_t, t)                                   prior denoise — NO demos (uncond)
② ẑ0  = (z_t − √(1-ᾱ_t)·eps)/√ᾱ_t                       predicted clean z (clamp ±4)
   guidance (only when t ≤ tmax, repeated `gsteps` times):
       g   = ∇_{ẑ0} cell_loss( filler(ẑ0⊙sd+mu, demos), demo_targets )    ← the ONLY demo channel
       ẑ0 ← ẑ0 − s·g/‖g‖                                                   ← strong: s=0.8
③ z_{t-1} = c1·ẑ0 + c2·z_t + √β̃_t·ξ                     posterior sample (2,3)
```

### 5.1 Why this is principled (score decomposition)

Unguided DDPM samples the prior `p(z_0)`; we want the posterior `p(z_0 | demos)`. The sampling engine
is the score `∇ log p_t`. For the noised conditional `p_t(z_t|y) ∝ p_t(z_t)·p(y|z_t)`:

```
∇_{z_t} log p_t(z_t|y) = ∇_{z_t} log p_t(z_t)  +  ∇_{z_t} log p_t(y|z_t)
                          └ prior score (= -ε_θ/√(1-ᾱ_t), Tweedie) ┘   └ likelihood (we add) ┘
```

The **prior score** is exactly what `model` gives (Tweedie: `∇log p_t(z_t) = -ε_θ/√(1-ᾱ_t)`). The
**likelihood score** is the demo channel: treat the filler's softmax as `p(y|z)`, so
`log p(y|z) = -N·cell_loss(filler(z,·), y)` and the guidance gradient `g = ∇ cell_loss` is (a scaled)
`-∇ log p(y|z)`. DPS (Chung et al. 2023) evaluates it at the predicted clean `ẑ0` rather than the
intractable `z_t`. Folding `g` into every low-noise step turns the chain from sampling `p(z)` into
sampling `p(z|demos)`. (Full Tweedie/DPS derivation: `DIFFUSION_Z_INFERENCE.md` §6.)

### 5.2 The guidance gradient `demo_guide_grad` (`hybrid.py`)

`g = ∇_{ẑ0} cell_loss(filler(ẑ0, K demos), demo_targets)` — the **same** gradient `search_z` descends,
only amortized across the 100 reverse steps instead of run as a standalone 200-step search. Looped
per-candidate (each candidate applied to its episode's K demos) to bound memory to `B·K`; the chain
through `zo = ẑ0⊙sd + mu` lands the gradient back in standardized space automatically.

### 5.3 The four knobs (what makes this guidance "strong", vs the old weak optimizer)

| knob | value | role |
|---|---|---|
| `guide_steps` | 4 | gradient sub-steps per noise level (old design: 1 → under-converged) |
| `guide_tmax`  | 50 | guide ONLY at low t (high-t ẑ0 is garbage; guiding it is wasted) |
| `guide_norm`  | 1  | L2-normalize g → step size = `s` exactly, decoupled from ‖g‖ (which is ~3000× too small raw) |
| `guide_scale` | 0.8 | per-step distance moved along the likelihood-ascent direction (standardized space) |

These four are the upgrade over `DIFFUSION_Z_INFERENCE.md`'s single-step/all-t/`s=0.3` guidance — the
change that turns a "weak optimizer" into one that ties test-time search.

### 5.4 The standardization sandwich

DDPM works in standardized space (N(0,1)); the filler only understands original scale (`|mu|≈7.4`).
`mu/sd` translate. The whole reverse loop is standardized; only inside the guidance gradient do we go
`ẑ0⊙sd+mu` → filler → and the chain rule carries `⊙sd` back, so the returned `g` is already standardized.
The output is un-standardized once at the end: `return (z*sd + mu)`. (`clamp(-4,4)` is a harmless 4σ
guard in standardized space — not the fatal original-scale `clamp(-1,1)` bug of the toy era.)

---

## 6. Selection and applying to the query (`hybrid.py:solver`)

Sample **C=32** candidate z's (32 noise seeds, guided). Pick the one whose filler best reproduces the
K demos (demo-fit, the same criterion search optimises):

```
fit(z) = mean over K demos of cellwise[ filler(z, demo_in).argmax == demo_out ]
best_z = argmax_c fit(z_c)
answer = filler(best_z, query_input).argmax        # (B, 144) predicted query-output colors
```

This selection is gradient-free; it just verifies which prior+guidance sample actually fits, then
executes it on the query. (Search-from-zero `z_s` is also run per batch as a same-budget reference.)

---

## 7. Why this is the best design

Two facts, both on the rule-in-training TRAIN split, frozen weights:

1. **Fewest parameters.** 11.39M total (filler 7.36M + denoiser 4.04M), encoder removed — lighter than
   the cross-attn conditional design (14.2M) and even the in-context "param-saver" (10.6M is close but
   only reaches 0.128). The intelligence is offloaded to test-time guidance, not a big conditioning net.
2. **Highest accuracy among the diffusion variants.** Uncond prior + strong guidance = **0.170 exact**,
   exactly tying test-time search (0.170). The "guidance ties, conditioning exceeds" decomposition: a
   prior that *never sees the demos* still hits search once strong guidance is on — adding a conditional
   (cross-attn) prior only buys the last +0.006 (0.176). For a lean, search-tying design you do not need
   the conditioning network at all.

Trade-off (stated plainly): this design **requires test-time filler gradients** (guidance). It is the
"minimal-parameter, compute-at-test-time" extreme. A cross-attn prior would sample clean (no test-time
gradient) at the cost of +2.8M params. See `../README.md` for the matched comparison vs vanilla AR /
vanilla diffusion, including the test-time compute accounting.

---

## 8. Code map (`hybrid.py`)

| symbol | role |
|---|---|
| `make_schedule` | β/α/ᾱ linear DDPM schedule (T=100) |
| `UncondDenoiser` | `CondDenoiser` with the conditioning slot hard-zeroed (no encoder); loads `cd_uncond.pt` |
| `demo_guide_grad` | per-candidate ∇ demo `cell_loss` wrt standardized z (the search/DPS gradient) |
| `sample_uncond_guided` | reverse DDPM from `p(z)` + in-loop guidance; demos only via `demo_guide_grad` |
| `solver` | sample C=32 → demo-fit select → execute best_z on query |
| `shared.load_filler` | frozen filler + the train/heldout split |
| `shared.eval_episodes` | the shared fixed TRAIN-split episode set + exact/cell scoring |

## 9. Reproduce

```bash
python hybrid/hybrid.py --C 32 --guide_scale 0.8 --guide_steps 4 --guide_tmax 50 --eval_n 10
# -> [hybrid] filler 7.36M + denoiser 4.04M = 11.39M (encoder removed)
# -> HYBRID (TRAIN): exact≈0.170  cell≈0.86
```
