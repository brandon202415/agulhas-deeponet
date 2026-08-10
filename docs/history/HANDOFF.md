# Agulhas DeepONet — Handoff Notes

**Date:** 2026-07-05  
**Project:** Physics-informed DeepONet for Agulhas eddy forecasting (single- and multi-step)  
**Working directory:** `/Users/brandonzhang/Downloads/eddy/`

---

## Update — 2026-07-05 (session 2): standalone + Aim 2 + cleanup

The trainer is now a **single standalone file** implementing both proposal aims.

- **Standalone.** `deeponet_dataset_utils.py` was inlined into
  `train_agulhas_deeponet_prototype.py` (metrics, JSON writer, SVG helpers) and
  deleted. The trainer imports no local module.
- **ACC metric (Aim 1).** Anomaly correlation coefficient vs a training
  climatology is now reported per variable (`acc_{var}` in `metrics.json`, and in
  the summary table). 0.6 = useful-skill threshold.
- **Autoregressive rollout (Aim 2).** New `rollout_evaluate()` rolls the model
  forward feeding predictions back, at horizons `--rollout-horizons` (default
  1/5/10/20 days), reporting per-variable ocean-masked RMSE/NRMSE/ACC vs lead
  time. Land sensors are held fixed at their initial values during rollout.
  Outputs: `rollout.npz`, `rollout_acc.svg`, `rollout_nrmse.svg`.
- **Spatial RMSE maps.** Per-sensor time-averaged single-step RMSE for
  `--spatial-var` (default `zos`) → `spatial_rmse.npz` + `spatial_rmse_zos.svg`.
- **PINN-vs-baseline comparison.** `--compare-to <other_out_dir>` overlays this
  run vs another (mean ACC/NRMSE vs lead time) and writes a spatial RMSE
  difference map. Run once with physics, once with `--lambda-div 0 --lambda-geo 0
  --compare-to <pinn_dir>`.
- **Physics loss land fix.** `physics_losses` now averages the divergence and
  geostrophic residuals over **ocean cells only** (land zeros excluded).
- **Residual normalisation fix (important).** The branch input is now normalised
  with the **same per-variable scalars as the output** (was a per-feature
  z-score), and land sensors are zeroed. This makes the persistence skip an
  *exact* identity at init: step-1 `L_data` dropped from ~1.05 to ~0.06 (the true
  1-day signal), so the network now learns only the day-to-day increment instead
  of also fighting a normalisation mismatch. Single-step ACC jumped to ~0.94–0.99
  and rollout ACC now decays gracefully (zos 0.98→0.76→0.45 at 1/5/10 d) instead
  of collapsing. `b_mean`/`b_std` are still saved and used by rollout for the
  feedback re-normalisation.
- **Folder cleanup.** Deleted `deeponet_dataset_utils.py`, `audit_pipeline.py`,
  `verify_pipeline.py`, the stale `results/{pinn,baseline,agulhas_prototype}`
  dirs, `__pycache__/`, and `.DS_Store` files. Kept the trainer, the download
  script, `data/agulhas_prototype.nc`, and this file.

> The results currently in `results/agulhas_pinn/` and `results/agulhas_baseline/`
> are **smoke-test outputs** (`--subsample-r 12 --iterations 300`) that only
> demonstrate the pipeline; regenerate them with a real run (see Next step).

---

## Update — 2026-07-05 (session 3): validation split, early stopping, HPC

For running the real study on HPC.

- **3-way chronological split.** `build_dataset` now partitions the timeline
  train | val | test (`--val-fraction`, `--test-fraction`, both 0.15). All
  normalisation / ocean-mask / climatology stats are still fit on **train only**.
- **Early stopping + best checkpoint.** `run_training` evaluates on the validation
  set every `--display-every` steps, caches the best-val `state_dict`, stops after
  `--patience` evals without improvement, and **restores the best checkpoint**
  before final test evaluation. So `model.pt` and all reported test/rollout
  metrics come from the best-generalising weights, not the last step. `--patience
  0` disables. `metrics.json` gains `best_val_loss`, `steps_run`, `n_val`.
- **Full-record download (two streams).** `download_agulhas_prototype.py` takes
  `--dataset {my,myint}` (plus optional `--start-date`/`--end-date`); each stream
  has sensible default dates and auto-names its output `data/agulhas_<stream>_<y0>_<y1>.nc`.
  GLORYS is split at ~2021-06 (`_my_` reanalysis vs `_myint_` interim) and no single
  request spans it, so the full ~1993–2024 record = run it twice (`--dataset my`,
  then `--dataset myint`). ~2–3 GB/yr → ~60–90 GB total; put it on scratch.
- **Multi-file loader.** `load_nc` now accepts a file, a directory, or a **glob**
  (`--nc 'data/agulhas_*.nc'`). Multiple files are read (strided), concatenated
  along time, then sorted with duplicate timestamps dropped — so the my + myint
  streams stitch into one clean daily record (verified on a synthetic overlap
  case). Grids must match across files or it errors.
- **SLURM (Yale YCRC — Bouchet).** `train_agulhas.slurm` is tailored for Bouchet:
  `--partition=gpu` with `--gpus=rtx_5000_ada:1` (Bouchet splits GPU partitions by
  type — `gpu`=RTX 5000 Ada, plus `gpu_h200`/`gpu_b200`/`gpu_devel`; the `--gpus`
  flag needs the type), `module load miniconda` + a conda env (the macOS `.venv`
  won't run on the Linux cluster), mail to the Yale address. In-file reminders:
  build the env once on a login node, download on a login node (compute nodes have
  no internet), and keep the large `.nc` on scratch (`~/scratch_pi_<pi_netid>` →
  `/nfs/roberts/scratch`, 10 TiB/group), not `$HOME` (125 GiB).

- **Memory-efficient loading.** `load_nc` now reads a **strided hyperslab**
  (`::r`) directly from disk instead of pulling each full-resolution variable into
  RAM and then subsampling. Peak memory scales with the *subsampled* grid, not the
  full grid — verified bit-identical to the old fancy-index result. This removes
  the multi-year OOM risk; `--mem` can be modest (sized to the subsampled cube,
  not the file).

Tuning is now honest: sweep hyperparameters, select on `best_val_loss`, report on
test. GPU is auto-detected (no code change needed). Use `--batch-size` for large N
since full-batch won't scale to ~10k samples.

---

## Update — 2026-07-06 (session 4): first real result + skill-focused iteration

**Key finding from the first full 1993–2021 run (r=6):** both the physics-informed
(λ=0.1) and data-driven (λ=0) models **collapse to persistence** — single-step
skill ≈ 0 for every variable, `best_val ≈ 0.0569` for both, and rollout mean-ACC
is nearly identical (baseline marginally *better*). So the Aim-2 hypothesis is
**not supported at these settings**, and the deeper issue is that the model can't
beat persistence at all: MSE over the whole field is dominated by the static part
persistence already nails, so the optimiser sits at Δ≈0. High ACC (0.9+) is
*persistence's* ACC and not evidence of skill. This motivated the changes below.

- **Data caching.** `--cache PATH` loads/saves the subsampled cube as `.npz`
  (seconds vs ~10 min re-decompressing the 27 GB NetCDF). `--prepare-cache` builds
  it and exits. `load_states()` wraps `load_nc`.
- **Physics weighting fixed.** Removed the reference-normalisation (it rescaled the
  trivially-satisfied `L_div` to unit scale and injected noise). Physics losses are
  now weighted **raw** by `λ`, and **`--lambda-div` defaults to 0** (it's
  machine-epsilon on reanalysis).
- **Skill-aligned loss.** `--loss-weight variability` weights each (sensor,var)
  squared error by a capped inverse increment-variance, so the objective rewards
  explaining the *change* (≈ per-column skill) instead of the static field.
- **Fair sweep metric.** `metrics.json` now has `val_mse_unweighted` — a plain
  ocean-masked val MSE computed identically for every config (unlike `best_val`,
  which is on the weighted scale). Rank sweeps on this.
- **Sweep harness.** `sweep.slurm` builds the cache once, runs a grid
  (LR × loss-weight × λ_geo, resumable — skips finished configs), and prints a
  leaderboard ranked by `val_mse_unweighted`.

**Sweep result (2026-07-06):** LR was the bottleneck — `lr3e-4` beats persistence
(mean skill +0.043); `lr1e-3` (the original) ~0. Physics and variability weighting
did **not** help. Winner: plain data-driven `lr3e-4`. See `RESEARCH_LOG.md`.

- **Multi-day forecast step.** `--step-days k` makes the model map state(t)→
  state(t+k); rollout advances k days per application (horizons must be multiples
  of k; non-multiples are auto-dropped). Threaded through `build_dataset`,
  `rollout_evaluate`, metrics. The cache is Δt-independent, so one cache serves all
  steps. `sweep.slurm` now sweeps `--step-days ∈ {1,5,10}` at the winning
  LR/loss-weight and **ranks the leaderboard by mean skill** (persistence-relative,
  the only quantity comparable across steps — val_MSE grows with Δt).

Next: `sbatch sweep.slurm` (now the step-days experiment) → read the
skill-ranked leaderboard: does a longer step raise skill, and does physics finally
help there?

Not done (documented for later): automated HP search beyond the fixed grid;
`r=3` finer-resolution memory tuning; interim (`_myint_`) stream for 2021–2024.

---

## What this project does

A `MultivarDeepONet` learns to forecast 6 ocean variables one day forward from the current ocean state over the Agulhas region. The model is trained on NEMO model output with physics-informed regularisation (divergence-free and geostrophic-consistency losses).

**Variables:** `zos` (SSH, m), `uo` (m/s), `vo` (m/s), `thetao` (°C), `so` (PSU), `mlotst` (MLD, m)  
**Data:** 1001 daily snapshots → 850 train / 151 test (chronological split)  
**Sensors:** 6161 ocean+land grid points subsampled at stride 6 from a ~78×79 lat/lon grid

---

## Architecture

```
Input: current ocean state [N, n_vars * n_sensors]
         ↓ split into 6 variable blocks
  6 × Branch MLP (depth-4, width-128, tanh)  →  6 latent vectors [N, latent_dim]
  1 × Trunk MLP (lat/lon → latent) [n_sensors, latent_dim]
         ↓ dot product per variable
  output = Σ branch_v · trunk  +  bias_v  +  skip  →  [N, n_sensors, n_vars]
```

**Persistence skip connection:** At init, each branch's last linear layer is zero-initialised so the network output is exactly 0, and the residual `out += current_state[variable_block]` makes the model start at perfect persistence. This keeps step-1 loss near 0 and gives the optimiser a warm start.

---

## Files changed

### `train_agulhas_deeponet_prototype.py`

Main training file. All significant edits are in this file.

#### 1. `build_dataset` — ocean-only output normalisation

**Problem:** `nan_to_num(nan=0.0)` zeros land points before normalisation. Salinity (35 PSU ocean, 0 PSU land, ~17% land) gave `out_std[so] ≈ 13 PSU` instead of the true `0.78 PSU`, making skill scores for `so` and `thetao` blow up to ~−100 000.

**Fix:** For each variable block, compute mean/std using only sensors with real temporal variability (ocean mask: `std(axis=0) > 1e-4`), then apply those scalars to the full block.

```python
for vi in range(n_vars):
    col_start = vi * n_sensors
    col_end   = (vi + 1) * n_sensors
    block = next_flat[train_idx, col_start:col_end]
    ocean = block.std(axis=0) > 1e-4
    vals  = block[:, ocean] if ocean.any() else block
    m = vals.mean()
    s = vals.std() or 1.0
    out_mean[vi] = m
    out_std[vi]  = s
    next_flat_norm[:, col_start:col_end] = (next_flat[:, col_start:col_end] - m) / s
```

**Side-effect introduced:** After this fix, land sensors have a normalised target of `(0 - 35) / 0.81 ≈ −43`, while the skip connection contributes 0. This caused step-1 loss ≈ 54. Fixed by masking land out of the loss (see next section).

#### 2. Ocean mask — exclude land from loss and metrics

`build_dataset` now computes and returns:
```python
ocean_mask = next_flat[train_idx, :n_sensors].std(axis=0) > 1e-4  # [n_sensors] bool
x_test_raw = _var_major_flat(states[test_idx].astype(np.float64))  # raw current state for persistence baseline
```

In `run_training`, a variable-major version of the mask is created:
```python
ocean_mask_vm = torch.tensor(np.tile(ds["ocean_mask"], n_vars), dtype=torch.bool, device=device)
```

**Data loss** (most recent edit — see below) applies the mask:
```python
l_data = nn.functional.mse_loss(pred_var_major[:, ocean_mask_vm], y_tgt[:, ocean_mask_vm])
```

**Test loss** for logging also masked:
```python
test_loss = nn.functional.mse_loss(pred_test_vm[:, ocean_mask_vm], y_test_norm[:, ocean_mask_vm]).item()
```

**Per-variable evaluation metrics** in `main()` apply `ds["ocean_mask"]` before computing RMSE/skill/NRMSE/bias:
```python
ocean_mask = ds["ocean_mask"]
true_vi  = true_3d[:, ocean_mask, vi]
pred_vi  = pred_3d[:, ocean_mask, vi]
pers_vi  = persist_flat[:, col_s:col_e][:, ocean_mask]
```

#### 3. Persistence skip connection (identity initialisation)

`MultivarDeepONet.__init__` zeros the last linear layer of each branch when `residual=True`:
```python
if residual:
    for branch in self.branches:
        last_linear = [l for l in branch.modules() if isinstance(l, nn.Linear)][-1]
        nn.init.zeros_(last_linear.weight)
        nn.init.zeros_(last_linear.bias)
```

`forward` adds the raw (branch-normalised) current state as a residual:
```python
if self.residual:
    out = out + branch_input[:, vi * self.n_sensors : (vi + 1) * self.n_sensors]
```

#### 4. Physics loss stabilisation

A 10-step running average over the first steps sets a reference scale for each physics loss, preventing the physics terms from dominating before the data loss has converged:
```python
_REF_STEPS = min(10, args.iterations)
# ... accumulate l_div, l_geo for steps 1-10 ...
l_div_scaled = l_div / l_div_ref   # normalised to ~1 at step 1
```

A linear warmup ramp (`min(1, step / warmup_steps)`) delays physics contributions from the very start.

#### 5. Evaluation metrics added

Per-variable physical-unit metrics now saved to `metrics.json`:
- `rmse_{var}` — RMSE in physical units (m, m/s, °C, PSU, m)
- `rmse_persist_{var}` — persistence baseline RMSE (sanity check: should be ~1-day ocean variability)
- `skill_{var}` — skill score `1 − (RMSE_model / RMSE_persist)²`; > 0 beats persistence
- `nrmse_{var}` — RMSE / std(truth); < 1 means error < natural variability
- `bias_{var}` — mean signed error

`predictions.npz` also saves `y_persist` (raw current state) for offline analysis.

#### 6. Visualisations

`write_visualizations` (materials-science) replaced with `write_ocean_visualizations` from `deeponet_dataset_utils.py`:
- `loss_history.svg` — log10 train/test MSE vs step
- `parity.svg` — scatter of predicted vs true values (handles negative SSH/velocity)
- `forecast_examples.svg` — first variable (zos) predicted vs truth for up to 4 test cases

#### 7. Minor fixes

- `--no-residual` CLI flag added to `parse_args()` (default: residual enabled)
- `metrics["rmse"]` / `metrics["mae"]` renamed from the materials-science `rmse_GPa`/`mae_GPa` keys
- `seed` parameter removed from `build_dataset` signature (split is deterministic/chronological)

### `deeponet_dataset_utils.py`

Added three ocean-specific visualisation functions:
- `write_ocean_visualizations(out_dir, y_true, y_pred, history, case_ids, variables)`
- `_write_ocean_parity_svg` — parity plot using symmetric lo/hi bounds (handles negatives)
- `_write_ocean_forecast_svg` — time-series plot per case, variable-major layout aware

---

## Current state

All code changes have been applied. **The model has not been re-run** since the ocean mask was added to the data loss. The results in `results/agulhas_pinn/metrics.json` are from the previous run (before the land-mask fix) and should be discarded — skill scores there are nonsense (e.g. `skill_so ≈ −93 000`).

---

## Next step: re-run the model

```bash
cd /Users/brandonzhang/Downloads/eddy

# Physics-informed run
python train_agulhas_deeponet_prototype.py \
    --nc data/agulhas_prototype.nc \
    --out-dir results/agulhas_pinn \
    --subsample-r 6 --iterations 5000 --display-every 100 \
    --lambda-div 0.1 --lambda-geo 0.1 --warmup-steps 500 \
    --rollout-horizons 1 5 10 20

# Data-driven baseline + PINN-vs-baseline comparison plots
python train_agulhas_deeponet_prototype.py \
    --nc data/agulhas_prototype.nc \
    --out-dir results/agulhas_baseline \
    --subsample-r 6 --iterations 5000 --display-every 100 \
    --lambda-div 0.0 --lambda-geo 0.0 \
    --rollout-horizons 1 5 10 20 \
    --compare-to results/agulhas_pinn
```

(Flags use hyphens, e.g. `--out-dir` / `--lambda-div`; the underscore forms from
the previous note are not valid.)

### What to expect after the fix

| Metric | Before fix | After fix (target) |
|---|---|---|
| Step-1 loss | ~54 | ~0 (persistence is exact at init) |
| `skill_so` | −93 000 | Should be meaningful (−5 to +0.5 range) |
| `skill_thetao` | −795 | Should be meaningful |
| `nrmse_so` | 1.14 (contaminated) | Should reflect true ocean-only error |
| Train/test gap | Moderate overfitting observed | Similar; may improve slightly |

### Interpreting results

- **Good:** `skill > 0` for any variable (model beats persistence)
- **Acceptable:** `nrmse < 1` (error smaller than 1-day natural variability)
- **Watch:** train vs test loss ratio; >2× gap indicates overfitting
- **Physics losses:** `L_div` will be near machine epsilon (ocean already divergence-free); `L_geo` stays roughly constant (persistence has fixed geostrophic imbalance) — this is expected, not a bug

---

## Known limitations / future work

1. ~~**Single-step only**~~ — *done in session 2:* autoregressive rollout at
   1/5/10/20 days is now implemented (`rollout_evaluate`)
2. **Physics losses may be too weak** — L_div is trivially satisfied; L_geo is constant with persistence skip; consider removing or replacing with a different constraint
3. **Overfitting** — try increasing `branch_width`/`latent_dim` or adding dropout; or reduce capacity and rely on physics regularisation more
4. **Land sensors in branch input** — branch still *reads* land zeros as input features; only the loss is masked. Consider masking branch input too (set land to mean, or use a separate ocean-only sensor set)
5. **Subsample stride** — currently stride=6 (6161 sensors from a full ~500×600 grid); increasing resolution will improve physics loss quality at the cost of memory
