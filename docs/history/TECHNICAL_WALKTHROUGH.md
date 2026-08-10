# A Technical Walkthrough of the Physics-Informed Agulhas DeepONet

*Audience: an ML/AI researcher who knows operator learning, PINNs, and
autoregressive forecasting, and wants both the intuition and the exact mechanics
of **this** implementation. All the code lives in a single standalone file,
`train_agulhas_deeponet_prototype.py`; function names below refer to it.*

---

## 0. One-paragraph summary

We learn a **one-day evolution operator** for the surface ocean state over the
Agulhas Current from the GLORYS12V1 reanalysis. A DeepONet (six per-variable
branch nets + one shared trunk) maps the current field to the next day's field,
with a **persistence residual** so the network only learns the day-to-day
*tendency*. Two **soft physics penalties** — a divergence-free constraint on the
predicted velocity and a geostrophic-balance constraint linking SSH gradients to
velocity — are added to the data loss. We evaluate single-step skill (RMSE, skill
vs persistence, ACC) and, by feeding predictions back, **multi-step
autoregressive skill** vs lead time. The central hypothesis is that the physics
penalties slow error growth during rollout relative to a pure data-driven
baseline.

---

## 1. The problem as operator learning

### Intuition

Classical ocean forecasting integrates PDEs on a grid; it is accurate but hits a
computational wall at eddy-resolving scales. We instead want a *learned* map

$$G:\ \mathbf{s}(\cdot,t)\ \longmapsto\ \mathbf{s}(\cdot,t+\Delta t),\qquad \Delta t = 1\ \text{day}$$

that takes the whole current field and returns the whole next field in one cheap
forward pass. This is an operator between function spaces, not a fixed-size
vector regression — which is exactly what **DeepONet** (Lu et al. 2021) is built
for.

### Why DeepONet rather than a CNN/U-Net

A DeepONet factorizes the operator into

$$G(\mathbf{s})(\mathbf{y}) \approx \sum_{k=1}^{p} b_k(\mathbf{s})\, t_k(\mathbf{y}) + b_0,$$

- **branch** $b(\mathbf{s})$: encodes the *input function* (the current field sampled at $m$ sensor locations) into $p$ coefficients;
- **trunk** $t(\mathbf{y})$: encodes the *query coordinate* $\mathbf{y}=(\text{lon},\text{lat})$ into $p$ basis values.

Read it as a **learned spectral method**: the trunk learns a global spatial basis
$\{t_k(\mathbf{y})\}$; the branch predicts the coefficients $\{b_k\}$ for the
current state. Because the trunk is queried at *continuous* coordinates, the model
is in principle discretization-invariant — you can evaluate the forecast at points
not on the training grid. That is the property a fixed-grid CNN does not have, and
it is why the trunk takes coordinates as input, using **tanh** activations (smooth
derivatives, which matters for the physics residuals below).

---

## 2. Data pipeline — from GLORYS to tensors

Handled by `load_nc()` and `build_dataset()`.

**Source.** GLORYS12V1 reanalysis, Agulhas domain (20°–50°S, 0°–50°E), surface
level, six variables:

| var | meaning | units | role |
|---|---|---|---|
| `zos` | sea surface height | m | primary eddy signature; drives geostrophy |
| `uo`,`vo` | eastward/northward surface velocity | m/s | constrained by both physics terms |
| `thetao` | potential temperature | °C | auxiliary thermodynamic state |
| `so` | salinity | PSU | auxiliary |
| `mlotst` | mixed-layer depth | m | auxiliary |

**From cube to samples.**

1. Load to a cube `states : [T, nlat, nlon, n_vars]` (spatial dims kept separate —
   the physics losses need the 2-D grid for finite differences).
2. **Subsample** every `r`-th grid point in lon and lat (`--subsample-r`, default
   6). This is the knob that trades eddy resolution for input dimensionality; the
   full 1/12° grid is ~217k points × 6 vars ≈ 1.3M inputs, far too wide.
3. Land/mask cells are NaN → replaced with **0** (`nan_to_num`). Land is handled
   specially everywhere downstream (see §4.1, §6).
4. Form pairs $(\mathbf{s}_t,\mathbf{s}_{t+1})$, giving $N=T-1$ samples.

**Layout convention (important for reading the code).** Everything is flattened
**variable-major**: columns $[v\cdot n_{\text{sensors}} : (v{+}1)\cdot
n_{\text{sensors}}]$ hold all sensors of variable $v$. `_var_major_flat()` is the
single source of truth for this; the model output is permuted back to it before
the loss. Get this wrong and variables silently mix.

**Split.** *Chronological*, not random: the timeline is partitioned in order into
**train | validation | test**, with val and test taken from the most recent end
(`--val-fraction`, `--test-fraction`, both 0.15 by default). Consecutive daily
fields are almost identical, so a random split would leak near-duplicates. All
normalization statistics, the ocean mask, and the climatology are fit on the
**train slice only**, so val and test are unseen during preprocessing. Validation
drives early stopping and best-checkpoint selection (§6); test is touched only for
the final metrics and rollout.

**Normalization** (all statistics fit on the training slice only):

| stream | scheme | why |
|---|---|---|
| trunk (coords) | min–max → $[-1,1]$ | bounded input for the coordinate MLP |
| output / target | **per-variable scalar** z-score, ocean cells only | one $(\mu_v,\sigma_v)$ per variable; land excluded so $\sigma$ isn't inflated (e.g. salinity's 35→0 land jump would blow up its std) |
| branch (input) | **same per-variable scalars as the output**, land set to 0 | see §4.1 — this equality is what makes the persistence residual exact |

**Climatology.** `build_dataset()` also computes a per-sensor, per-variable
training-mean field, `climatology : [n_sensors, n_vars]`. This is the reference
field for ACC (§7).

---

## 3. Architecture

`MultivarDeepONet`. One **shared trunk** and **six branches** (one per variable).
Each branch reads the *entire* flattened state (all six variables at all
sensors), so cross-variable information is available to every output — e.g. SSH
can inform the temperature prediction.

```
current state  s_t                      query coords y=(lon,lat)
[N, n_vars*n_sensors]                    [n_sensors, 2]
        │                                        │
   split into 6 blocks                     shared trunk MLP
        │                                   [2]→[64]*2→[latent]
 6 × branch MLP (tanh)                            │
 [D]→[64]*2→[latent]                       trunk_feats [n_sensors, latent]
        │                                        │
   branch_feats_v [N, latent] ───dot(einsum "np,sp->ns")──► [N, n_sensors]
                                                 │
                            + bias_v   + persistence residual  (§4)
                                                 ▼
                              stack over v → prediction [N, n_sensors, n_vars]
```

Defaults: `latent_dim=32`, branch/trunk width 64, depth 2, tanh, Glorot init.
The merge is a per-variable dot product over the latent axis plus a scalar bias
$b_0$ per variable (following Lu et al.). This is an **unstacked** DeepONet: all
$p=\text{latent\_dim}$ trunk outputs come from one network, not $p$ parallel nets.

**Design tension worth noting:** the trunk is *shared* across variables, which
assumes SSH, velocity, temperature, and MLD share a spatial basis. Cheap and
regularizing, but possibly too restrictive (MLD and SSH have very different
correlation lengths). Per-variable trunks are the obvious next lever at ~6× trunk
params.

---

## 4. The persistence residual — the key inductive bias

### 4.1 Intuition

At 1-day lead, the single best trivial forecast is **persistence**: tomorrow ≈
today. The day-to-day change is small relative to the field itself. So instead of
asking the network to regenerate the whole field (most of whose "signal" is just
"the same as yesterday"), we make persistence the *default* and let the network
learn only the **tendency** $\Delta$:

$$\hat{\mathbf{s}}_{t+1} = \underbrace{\mathbf{s}_t}_{\text{persistence}} + \underbrace{\mathrm{DeepONet}(\mathbf{s}_t)}_{\text{learned increment}}.$$

This is the same trick as residual connections, weather models predicting
tendencies (GraphCast/Pangu), and diffusion models predicting noise rather than
the image: **put the easy part in the architecture, learn the hard part.**

Mechanically: the last linear layer of every branch is **zero-initialized**
(`residual=True`), so at step 0 the DeepONet term is exactly 0 and the output
equals the current state. The optimizer starts from persistence and only has to
move $\Delta$ away from zero.

### 4.2 The normalization subtlety (and why it matters)

The residual is added in **normalized** space:
`out += branch_input[block_v]`. For "output = current state" to actually equal
persistence *as the loss sees it*, the branch input and the target must live in
the **same** normalized space. Originally they did not:

- branch was **per-feature** (per-sensor) z-scored;
- target was **per-variable** (scalar) z-scored.

So at init the model output sat in a *different* normalization than persistence.
Measured at init (ocean-masked, in target units):

| quantity | broken (per-feature branch) | fixed (per-variable branch) |
|---|---|---|
| init `L_data` (where training starts) | **1.05** | **0.06** |
| residual vs *true* persistence (MSE) | **0.98** | **0.0000** |

1.05 is roughly the loss of predicting climatology — i.e. the "persistence skip"
was contributing almost nothing, and the network was effectively **rebuilding the
field from scratch** while also fighting the normalization offset. After aligning
the branch normalization to the per-variable output scalars (and zeroing land
inputs), the residual is an *exact* identity: step-1 loss drops to 0.06 (the true
1-day signal), and the network learns only $\Delta$.

Downstream effect at identical (undertrained, coarse) smoke settings:

| | broken | fixed |
|---|---|---|
| single-step ACC (`zos`/`thetao`) | 0.69 / 0.60 | 0.98 / 0.99 |
| rollout ACC `zos` @ 1/5/10 d | 0.69 / 0.06 / 0.01 (collapse) | 0.98 / 0.76 / 0.45 (graceful) |

**Transferable lesson:** a residual/skip connection is only an identity if the
thing you add and the thing you supervise are in the *same* space. Mismatched
normalization silently turns "learn the correction" into "learn everything." If
your residual model won't beat a naive baseline, check this first.

Land sensors are set to 0 in the (normalized) branch input — the per-variable
mean, i.e. a neutral input — because land carries no dynamical signal, is masked
from the loss, and is held fixed during rollout. Zeroing also avoids feeding large
constants (salinity land ≈ $(0-35)/0.78 \approx -45$) into the branch MLPs.

---

## 5. Physics-informed losses

Both are **soft constraints** (PINN-style, Raissi et al. 2019): extra loss terms,
not hard projections. Computed on **denormalized** (physical-unit) predictions via
finite differences on the lon/lat grid (`_fd_grad_lon`, `_fd_grad_lat`: central in
the interior, one-sided at boundaries), and averaged over **ocean cells only**
(`physics_losses(..., ocean_mask_grid)`), because land zeros create spurious
gradients at the coast.

### 5.1 Divergence-free (mass conservation, 2-D)

**Intuition:** at mesoscale, horizontal flow is nearly non-divergent — water
rotating in an eddy neither piles up nor drains. In spherical coordinates:

$$\frac{1}{a\cos\phi}\frac{\partial u}{\partial\lambda} + \frac{1}{a\cos\phi}\frac{\partial(v\cos\phi)}{\partial\phi} \approx 0,$$

$$L_{\text{div}} = \big\langle\, \text{(LHS)}^2 \,\big\rangle_{\text{ocean}}.$$

### 5.2 Geostrophic consistency

**Intuition:** away from the equator, the pressure-gradient force balances the
Coriolis force, so SSH slope *predicts* the surface current. This ties the two
independently-predicted quantities (SSH and velocity) together:

$$u_g = -\frac{g}{f\,a}\frac{\partial\hat\eta}{\partial\phi},\qquad v_g = \frac{g}{f\,a\cos\phi}\frac{\partial\hat\eta}{\partial\lambda},\qquad f = 2\Omega_E\sin\phi,$$

$$L_{\text{geo}} = \big\langle\, (\hat u - u_g)^2 + (\hat v - v_g)^2 \,\big\rangle_{\text{ocean}}.$$

Constants: $a=6.371\times10^6$ m, $\Omega_E=7.29\times10^{-5}$ rad/s, $g=9.81$.
$f$ is clamped away from 0 (safe throughout 20°–50°S; the equatorial breakdown of
geostrophy is never in-domain).

### 5.3 The honest caveat

On reanalysis data these constraints are **weak**:

- $L_{\text{div}}$ is near machine epsilon — the reanalysis velocity field is
  *already* nearly non-divergent, so the penalty is trivially satisfied and gives
  almost no gradient.
- $L_{\text{geo}}$ is roughly constant under the persistence skip (persistence has
  a fixed geostrophic imbalance).

This is the project's main open question (§9). It is *expected*, not a bug, but it
means the physics terms may need reformulation (e.g. constraining the *tendency*,
or targeting ageostrophic diagnostics) to actually move the needle.

---

## 6. Training objective and loop (`run_training`)

$$L_{\text{total}} = L_{\text{data}} + \text{ramp}\cdot\lambda_1\frac{L_{\text{div}}}{L_{\text{div}}^{\text{ref}}} + \text{ramp}\cdot\lambda_2\frac{L_{\text{geo}}}{L_{\text{geo}}^{\text{ref}}}.$$

- $L_{\text{data}}$: MSE on **ocean sensors only** (`ocean_mask_vm`), in
  per-variable-normalized units. With `--loss-weight variability` each
  (sensor, variable) squared error is weighted by a capped inverse
  increment-variance (mean-1 normalized), turning it into a skill-aligned
  objective (§8) so the model is scored on the dynamics, not the static field.
- **Physics weighting is raw:** $L = L_{\text{data}} + \text{ramp}\cdot\lambda_1
  L_{\text{div}} + \text{ramp}\cdot\lambda_2 L_{\text{geo}}$, weighted directly by
  $\lambda$. (An earlier reference-normalization that rescaled each physics term to
  $\mathcal{O}(1)$ was **removed** — it inflated the trivially-satisfied
  $L_{\text{div}}\!\sim\!10^{-12}$ to unit scale and fed pure numerical noise into
  the gradient.) `--lambda-div` now **defaults to 0** for that reason.
- **Warmup ramp:** $\text{ramp}=\min(1,\text{step}/\text{warmup\_steps})$ delays
  the physics terms so the data loss can establish a reasonable field first —
  otherwise physics penalties dominate a randomly-initialized prediction.
- $\lambda_1=\lambda_2=0.1$ by default. **Setting both to 0 is the data-driven
  baseline** — the ablation at the heart of the study.
- Optimizer: Adam, lr $10^{-3}$, full-batch by default (the prototype fits in
  memory; `--batch-size` enables minibatching — necessary once $N$ reaches ~10k).

**Validation & early stopping.** Every `--display-every` steps the model is
evaluated on the validation set (normalized ocean-masked MSE). The best-val
`state_dict` is cached; training stops early after `--patience` evaluations with
no improvement, and the **best-val checkpoint is restored** before final
evaluation — so reported test metrics come from the weights that generalized best,
not the (possibly overfit) last step. Set `--patience 0` to disable.

---

## 7. Evaluation

### 7.1 Single-step metrics (per variable, ocean-masked)

Computed in `main()`; ACC via `anomaly_correlation()`.

- **RMSE** — error in physical units.
- **Persistence RMSE** — the tomorrow=today baseline. The bar to clear.
- **Skill** $= 1 - (\text{RMSE}/\text{RMSE}_{\text{persist}})^2$. $>0$ beats
  persistence; $=1$ is perfect; can go very negative.
- **NRMSE** $= \text{RMSE}/\sigma_{\text{truth}}$. $<1$ means error below natural
  variability.
- **ACC** — anomaly correlation vs climatology:
  $$\text{ACC} = \frac{\sum \hat{\mathbf s}'\cdot\mathbf s'}{\sqrt{\sum \hat{\mathbf s}'^2 \sum \mathbf s'^2}},\quad \mathbf s' = \mathbf s - \text{clim}.$$
  The standard weather/ocean skill score; $>0.6$ = useful. It measures *pattern
  correlation of anomalies*, so it is forgiving of amplitude bias.
- **Bias** — mean signed error.

### 7.2 ACC vs skill — read them together

This pairing is the most common misreading, so state it plainly: **high ACC and
negative skill co-occur and are not contradictory.** ACC near 1 says "the
predicted anomaly pattern matches the truth." Negative skill says "you still did
not beat tomorrow=today." Both are true when the model reproduces the field's
structure but is marginally noisier than persistence — which is the *default*
situation at 1-day lead, because persistence is a brutally strong baseline. Skill
is the honest, hard metric here; ACC confirms the model is learning structure
rather than mush.

### 7.3 Autoregressive rollout (`rollout_evaluate`)

The Aim-2 payload. Roll the model forward, feeding its own output back as input,
at horizons `--rollout-horizons` (default 1/5/10/20 days), over every valid test
start, reporting per-variable RMSE/NRMSE/ACC vs lead time.

The **normalization round-trip** each step is the part to get right:

```
cur_raw ──(−b_mean)/b_std──► branch input ──[zero land]──► model
   ▲                                                         │
   │                                              pred (normalized)
   │                                                         │
   └── hold land fixed ◄── reflatten var-major ◄── denorm (out_mean/out_std)
```

Two correctness points:
- **Denorm/renorm must use the matching scalars** (`out_mean/out_std` to go to raw,
  `b_mean/b_std` to come back). Because of §4.2 these are now the same per-variable
  values, so the round-trip is consistent. Rollout works entirely in raw units for
  the truth comparison, so it is robust regardless.
- **Land is held fixed** at its initial value every step. Land is unlearned
  (masked from the loss), so if you let it evolve autoregressively it blows up and
  contaminates the branch input on the next step. Freezing it is essential for
  rollout stability.

Expect ACC to **decay** with lead time and RMSE to **grow**; the hypothesis is
that the physics-constrained model decays *slower* than the baseline.

### 7.4 Spatial maps and the baseline comparison

- `spatial_rmse_{var}.svg` / `spatial_rmse.npz`: per-sensor, time-averaged
  single-step RMSE reshaped to the grid — a heatmap of *where* the model errs.
- `--compare-to <other_run>`: overlays this run vs another (mean ACC/NRMSE vs lead
  time) and writes a **difference map** `spatial_rmse_diff_{var}.svg`. The intended
  use is physics-run vs baseline-run: red where physics helps, blue where it hurts.
  The proposal's geographic hypothesis is that physics helps in the open-ocean
  geostrophic eddy field and *hurts* in the ageostrophic current core /
  retroflection.

---

## 8. What "good" looks like (and traps)

- **The persistence trap.** At 1-day lead, matching persistence is already hard;
  beating it is the real test. Do not celebrate low RMSE without checking skill —
  persistence gets low RMSE for free.
- **Undertrained smoke runs** (`--subsample-r 12 --iterations 300`) exist only to
  prove the pipeline runs; their skill is meaningless. Real assessment needs
  `--subsample-r 6 --iterations 5000`.
- **Physics diagnostics:** $L_{\text{div}}\to$ machine epsilon and
  $L_{\text{geo}}\approx$ const are expected (§5.3), not failure.
- **Overfitting watch:** train/test loss gap; the field is smooth and the model is
  large, so regularization (physics, capacity) matters.

---

## 9. Known limitations & open questions

1. **Weak physics.** The two constraints are near-trivially satisfied on
   reanalysis (§5.3). Reformulating them (constrain the *increment*; penalize
   ageostrophic imbalance where it shouldn't exist) is the highest-value research
   direction.
2. **Where, not just whether.** The spatial diff map is the mechanism to test the
   geostrophic-vs-ageostrophic hypothesis; interpreting it against published
   Agulhas imbalance diagnostics is future work.
3. **HP tuning is now enabled** (3-way split + early stopping), but no automated
   search (grid/Bayesian) is wired in — you drive it by hand or with an external
   sweeper, selecting on `best_val_loss` in `metrics.json`.
4. **Shared trunk** may under-serve variables with different spatial scales (§3).
5. **Subsampling** (`r=6`) discards eddy-scale structure; a patch-based scheme
   preserving local resolution is the fallback if eddies are under-resolved.
6. **Land in the branch input** is zeroed but still occupies input dimensions; a
   dedicated ocean-only sensor set would be leaner.

---

## 10. How to run

**Get the data (login/transfer node — compute nodes usually lack internet):**

```bash
copernicusmarine login                              # once
# Full 1993–2024 record = two GLORYS streams (no single request spans the ~2021-06
# seam). Both land in data/; the trainer stitches them via the glob.
python download_agulhas_prototype.py --dataset my       # 1993 → 2021-06
python download_agulhas_prototype.py --dataset myint    # 2021-07 → 2024
```

**Train (physics run, then baseline + comparison):**

```bash
python train_agulhas_deeponet_prototype.py \
    --nc 'data/agulhas_*.nc' --out-dir results/agulhas_pinn \
    --subsample-r 6 --iterations 20000 --batch-size 256 \
    --val-fraction 0.15 --test-fraction 0.15 --patience 20 \
    --warmup-steps 1000 --lambda-div 0.1 --lambda-geo 0.1 \
    --rollout-horizons 1 5 10 20

python train_agulhas_deeponet_prototype.py \
    --nc 'data/agulhas_*.nc' --out-dir results/agulhas_baseline \
    --subsample-r 6 --iterations 20000 --batch-size 256 \
    --val-fraction 0.15 --test-fraction 0.15 --patience 20 \
    --lambda-div 0.0 --lambda-geo 0.0 \
    --rollout-horizons 1 5 10 20 --compare-to results/agulhas_pinn
```
(Quote the glob so your shell doesn't expand it — the trainer stitches the matched
files itself.)

**Caching + sweeps.** `--cache data/cache_r6.npz` saves/loads the subsampled cube
as `.npz` (seconds vs ~10 min re-decompressing the raw file); `--prepare-cache`
builds it and exits. `sweep.slurm` uses this to run a val-based grid
(LR × `--loss-weight` × `λ_geo`, resumable) and prints a leaderboard ranked by
`val_mse_unweighted` — the one loss metric comparable across configs (`best_val`
is on the weighted scale). Submit with `sbatch sweep.slurm`.

On Yale YCRC (Bouchet), submit both via `train_agulhas.slurm` (`sbatch
train_agulhas.slurm`). Its header documents the one-time setup: build a conda env
(`module load miniconda`; the macOS `.venv` won't run on the Linux cluster),
download on a login node (compute nodes have no internet) with the big `.nc` on
scratch (`~/scratch_pi_<pi_netid>`), then set `NC`. The script targets the `gpu`
partition (RTX 5000 Ada) with `--gpus=rtx_5000_ada:1`; GPU is auto-detected in
code. `load_nc` reads a **strided hyperslab**
straight from disk, so peak RAM scales with the *subsampled* grid (not the
full-resolution file) — multi-year files load without OOM. The one scaling knob
that still matters: full-batch does not scale to ~10k samples, so pass
`--batch-size`.

Outputs per run: `metrics.json` (all scalar metrics incl. `acc_*`, `rollout_*`,
`best_val_loss`), `predictions.npz`, `rollout.npz`, `spatial_rmse.npz`,
`model.pt`, and SVGs (`loss_history`, `parity`, `forecast_examples`,
`rollout_acc`, `rollout_nrmse`, `spatial_rmse_*`, plus comparison plots with
`--compare-to`).

---

## 11. Code map

| function | responsibility |
|---|---|
| `load_nc` | read one or many NetCDF files (file/dir/glob) via a strided hyperslab (memory-safe), stitch along time (sort + dedup), NaN→0, return the `states` cube |
| `load_states` | `load_nc` wrapped in an optional `.npz` cache of the subsampled cube (fast reuse across runs / sweeps) |
| `build_dataset` | pairs, chronological split, all normalization, ocean mask, climatology |
| `_var_major_flat` | the variable-major flatten convention (used everywhere) |
| `MultivarDeepONet` | shared trunk + 6 branches, dot-product merge, persistence residual |
| `_fd_grad_lon/lat` | finite-difference gradients on the lon/lat grid |
| `physics_losses` | ocean-masked $L_{\text{div}}$, $L_{\text{geo}}$ |
| `run_training` | full loss (data + scaled/warmed physics), Adam loop, logging |
| `evaluate` | single-step test-set inference, denormalized |
| `anomaly_correlation` | ACC vs climatology |
| `rollout_evaluate` | autoregressive multi-step skill vs lead time |
| `_write_*_svg` / `write_*_visualizations` | inlined SVG plotting (no deps) |
| `main` | orchestration, metric assembly, artifact writing |

*This file is standalone: no local module imports, only `torch`, `numpy`,
`netCDF4`.*
