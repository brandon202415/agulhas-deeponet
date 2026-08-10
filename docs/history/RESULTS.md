# RESULTS — Physics-Informed DeepONet for Agulhas Eddy Forecasting

Consolidated results and interpretation for the manuscript. All numbers below are
from the finalised runs; the raw arrays behind every figure are in the `results/`
`.npz`/`.json` files (see **Figures** and **Reproducibility**).

---

## 0. How to write the manuscript from these docs

This file plus the others in the repo contain everything needed. Map of
manuscript section → source:

| Manuscript section | Primary source(s) |
|---|---|
| Title / Abstract | **RESULTS.md** §1 (findings) + `Research Proposal.md` (framing) |
| Introduction / Background / Significance | `Research Proposal.md` "Background and Significance" (+ its references list) |
| Hypothesis & Aims | `Research Proposal.md` (Hypothesis, Specific Aims 1–2) |
| Methods — study domain & data | `Research Proposal.md` Aim 1 + `TECHNICAL_WALKTHROUGH.md` §2 |
| Methods — model architecture | `TECHNICAL_WALKTHROUGH.md` §3–4 + `Research Proposal.md` "Model Architecture" |
| Methods — physics-informed losses | `TECHNICAL_WALKTHROUGH.md` §5 + `Research Proposal.md` "Physics-Informed Training" |
| Methods — training & evaluation metrics | `TECHNICAL_WALKTHROUGH.md` §6–7 + `Research Proposal.md` "Training and Evaluation" |
| Methods — reproducibility / config | **RESULTS.md** §7 + `HANDOFF.md` |
| Results | **RESULTS.md** §2–6 (all tables) |
| Figures | **RESULTS.md** §8 (each figure + its data file); generate with `make_figures.py` |
| Discussion / what did & didn't work | **RESULTS.md** §2–6 interpretation + `RESEARCH_LOG.md` (chronology, dead ends) |
| Limitations & Future Work | **RESULTS.md** §9 + `TECHNICAL_WALKTHROUGH.md` §9 |
| References | `Research Proposal.md` reference list |

**Note on metric definitions** (used throughout): *skill* = 1 − (RMSE_model /
RMSE_persist)² (>0 beats persistence); *ACC* = anomaly correlation vs the
training-climatology (>0.6 = useful; standard ocean/weather skill score); *NRMSE*
= RMSE / std(truth); *persistence (naive)* = "tomorrow = today," the forecast that
holds the initial field fixed. Full definitions in `TECHNICAL_WALKTHROUGH.md` §7.

---

## 1. Findings at a glance (abstract-level)

We trained a Deep Operator Network (DeepONet) to forecast the daily surface ocean
state (SSH, u/v velocity, temperature, salinity, mixed-layer depth) over the
Agulhas Current from the GLORYS12V1 reanalysis (1993–2021), and tested whether
divergence-free and geostrophic **physics-informed** losses improve on a
data-driven baseline.

1. **Single-step (1-day) skill:** a well-tuned data-driven DeepONet **beats the
   persistence baseline** by a small but consistent margin (mean skill **+0.043**),
   concentrated in the velocity fields (northward velocity **+0.122**, eastward
   **+0.066**).
2. **The physics constraints provide no benefit** — confirmed across learning
   rates, loss weightings, forecast steps (1/5/10 d), and resolutions (r=6, r=3).
   Diagnostics show why: on reanalysis the fields already satisfy the constraints
   (divergence loss ≈ 10⁻¹², geostrophic loss ≈ constant), so a soft penalty has
   nothing to correct.
3. **Multi-step skill collapses:** under autoregressive rollout the model beats
   persistence **only at 1 day**; at ≥5 days its errors compound and it becomes
   **worse than the frozen "no-change" forecast** (skill goes negative). Useful
   horizon ≈ 1 day. This is the "unphysical drift in multi-step forecasting" the
   proposal hypothesised — present even in the data-driven model, and **not
   prevented by the physics constraints.**
4. **Tuning finding:** the learning rate, not physics, was the decisive factor —
   the original lr=10⁻³ barely beat persistence; lr=3×10⁻⁴ reached the +0.043 above.

---

## 2. Setup (brief; full methods in the walkthrough/proposal)

- **Data:** GLORYS12V1 reanalysis, Agulhas box 20–50°S, 0–50°E, surface level,
  6 variables. Reanalysis stream 1993-01-01 → 2021-06-30 = **10,408 daily steps**.
- **Grid / sensors:** subsample factor r; **r=6 → 6,161 sensors** (101×61 grid),
  the main configuration. r=3 → 24,321 sensors (tested separately).
- **Split:** chronological **train/val/test = 70/15/15** (7,285 / 1,561 / 1,561
  pairs), val/test most recent; all stats fit on train only.
- **Model:** joint DeepONet, one shared trunk + 6 per-variable branches, persistence
  residual. **~14.2 M params at r=6** (56 M at r=3; corrected 2026-07-27 — an
  earlier draft of this doc and the manuscript stated ~3.7 M for r=6, which was
  wrong; verified directly from the saved `model.pt` checkpoint). Adam, seed 2026
  (deterministic).
- **Best configuration:** `--subsample-r 6 --learning-rate 3e-4 --loss-weight none
  --lambda-div 0 --lambda-geo 0 --step-days 1 --batch-size 256`, early stopping.

---

## 3. Result 1 — Single-step forecast skill (Aim 1)

Best model, test set, physical units. RMSE = model error; Persist = persistence
error; Skill = model vs persistence; ACC vs climatology.

| Variable | RMSE | Persist | **Skill** | NRMSE | ACC | Bias | units |
|---|---|---|---|---|---|---|---|
| zos (SSH) | 0.0219 | 0.0221 | +0.015 | 0.042 | 0.991 | +0.0004 | m |
| uo (E vel) | 0.0759 | 0.0786 | +0.066 | 0.283 | 0.942 | −0.0019 | m/s |
| vo (N vel) | 0.0783 | 0.0836 | **+0.122** | 0.314 | 0.941 | +0.0007 | m/s |
| thetao (temp) | 0.3061 | 0.3078 | +0.011 | 0.044 | 0.987 | +0.0179 | °C |
| so (salinity) | 0.0637 | 0.0638 | +0.004 | 0.079 | 0.970 | +0.0000 | PSU |
| mlotst (MLD) | 16.62 | 16.97 | +0.041 | 0.383 | 0.910 | −0.394 | m |
| **mean** | | | **+0.043** | | | | |

**Interpretation.** Every variable has positive skill: the model beats persistence.
The gain is largest for the **velocity/current fields** (the eddies' kinematics) —
exactly where a dynamical model should add value over "no change" — and negligible
for the near-static fields (SSH, temperature, salinity) that persistence already
nails (ACC ≈ 0.99). Data: `results/best/metrics.json`.

---

## 4. Result 2 — Learning rate drives skill; physics & loss-weighting do not

Validation-selected hyperparameter sweep (r=6). Ranked by `val_mse_unweighted`
(the config-comparable selection metric); mean skill and 20-day rollout ACC shown
as test-set readouts.

| config (LR, loss-weight, λ_geo) | val_MSE | mean skill | rollACC@20d | steps |
|---|---|---|---|---|
| **lr3e-4, none, 0.0**  | 0.05297 | **+0.0432** | 0.441 | 3800 |
| lr1e-4, none, 0.0      | 0.05326 | +0.0144 | 0.386 | 4000 |
| lr3e-4, none, 0.05     | 0.05330 | +0.0393 | 0.435 | 3800 |
| lr1e-4, none, 0.05     | 0.05331 | +0.0141 | 0.394 | 4000 |
| lr3e-4, variability, 0.0  | 0.05425 | +0.0240 | 0.428 | 4600 |
| lr3e-4, variability, 0.05 | 0.05452 | +0.0222 | 0.413 | 4600 |
| lr1e-4, variability, 0.0  | 0.05483 | +0.0222 | 0.428 | 3200 |
| lr1e-4, variability, 0.05 | 0.05490 | +0.0210 | 0.429 | 3200 |
| lr1e-3, variability, 0.0  | 0.05617 | +0.0068 | 0.455 | 7600 |
| lr1e-3, none, 0.05        | 0.05637 | +0.0052 | 0.452 | 8000 |
| lr1e-3, variability, 0.05 | 0.05644 | +0.0038 | 0.445 | 6400 |
| lr1e-3, none, 0.0 *(original)* | 0.05688 | +0.0007 | 0.451 | 8000 |

**Interpretation.** (a) **Learning rate is decisive:** every `lr1e-3` config sits
at the bottom (~0 skill) — the original run used lr=10⁻³ and barely beat
persistence; `lr3e-4` moves to the top (+0.043). (b) **Physics doesn't help:** at
matched LR, λ_geo=0 ≈ λ_geo=0.05 (0 marginally better). (c) **Variability
weighting doesn't help:** plain MSE beats it at every LR. The winner is the
**simplest data-driven model**. Data: `results/sweep/*/metrics.json`.

---

## 5. Result 3 — The physics constraints are inert on reanalysis

Directly tested (Aim 1/2 core hypothesis). The physics-informed model (λ>0) never
outperforms the data-driven baseline (λ=0) — in the sweep above, in the original
r=6 run (physics best_val 0.05687 vs baseline 0.05688; both skill ≈ 0 at lr1e-3),
at longer forecast steps (§6), and at r=3 (§6).

**Mechanism (why):** the training targets come from a physics-based reanalysis
that already respects the constraints, so the residuals are near-trivial —
measured **L_div ≈ 10⁻¹²** (divergence-free essentially satisfied) and **L_geo ≈
constant** (fixed geostrophic imbalance the model can't reduce). A soft penalty
toward already-satisfied constraints supplies no useful gradient (and, in the
original reference-normalised form, injected numerical noise — see
`RESEARCH_LOG.md`). This is a clean, mechanistically-explained **negative result**
for the physics-informed hypothesis in this setting.

---

## 6. Result 4 — Multi-step rollout, forecast step, and resolution

### 6a. Autoregressive rollout: naive vs model over lead time (best model)
Mean over variables. Skill = 1 − (model_error / naive_error)² at each lead.

| lead | naive ACC | model ACC | **model skill vs naive** |
|---|---|---|---|
| 1 day  | 0.954 | 0.957 | **+0.043** |
| 5 day  | 0.772 | 0.767 | −0.016 |
| 10 day | 0.635 | 0.627 | −0.034 |
| 20 day | 0.456 | 0.441 | −0.069 |

Per-variable rollout ACC (model):

| var | 1d | 5d | 10d | 20d |
|---|---|---|---|---|
| zos | 0.991 | 0.889 | 0.736 | 0.477 |
| uo | 0.942 | 0.675 | 0.476 | 0.247 |
| vo | 0.941 | 0.660 | 0.446 | 0.186 |
| thetao | 0.987 | 0.902 | 0.828 | 0.714 |
| so | 0.970 | 0.792 | 0.663 | 0.505 |
| mlotst | 0.910 | 0.682 | 0.611 | 0.519 |

**Interpretation (the Aim-2 headline).** The model beats persistence **only at
1 day**. From 5 days on, rolling it forward autoregressively makes it **worse than
the frozen field** (negative skill): the 1-day errors feed back and compound. Its
useful horizon is ~1 day. Data: `results/best/rollout.npz`
(`skill`, `acc`, `acc_persist`, `rmse`, `rmse_persist`, `horizons`), 1,542 rollout
starts.

### 6b. Forecast step Δt (r=6, lr3e-4, plain MSE)

| Δt (days) | mean skill | rollACC@20d |
|---|---|---|
| 1 | +0.043 | 0.441 |
| 5 | +0.006 | 0.450 |
| 10 | +0.014 | 0.456 |

**Interpretation.** A longer forecast step **did not help** — per-step skill is
highest at Δt=1. (Longer steps do drift slightly less in rollout — fewer
autoregressive applications, hence marginally higher ACC@20d — a minor trade-off,
not a skill gain.)

### 6c. Resolution r=3 (finer grid; lr3e-4, plain MSE, Δt=1)

| config | mean skill | rollACC@20d | steps |
|---|---|---|---|
| r=3, λ_geo 0.0  | +0.0011 | 0.461 | 5000 |
| r=3, λ_geo 0.05 | +0.0015 | 0.456 | 5000 |

**Interpretation.** Finer resolution **did not help** — skill fell to ~0 (vs +0.043
at r=6), and physics remained inert. **Caveat / confound:** at r=3 the branch reads
all 24,321 sensors × 6 vars = 145,926 inputs → a 56 M-param model dominated by one
giant input projection, trained on only ~7,285 samples → underfit. The honest
reading is "**this DeepONet formulation does not scale to fine grids**" (the
all-sensors branch input is the bottleneck), matching the high-resolution scaling
limit anticipated in the proposal's pitfalls (remedy: a **patch-based** branch).

---

## 7. Reproducibility

- **Environment:** Python 3.11, PyTorch (CUDA build), NumPy, netCDF4; seed 2026
  (deterministic). Cluster: Yale YCRC Bouchet (see `HANDOFF.md`, `train_agulhas.slurm`).
- **Data provenance:** GLORYS12V1, product `GLOBAL_MULTIYEAR_PHY_001_030`, dataset
  `cmems_mod_glo_phy_my_0.083deg_P1D-m` (reanalysis, 1993→2021-06), DOI
  **10.48670/moi-00021**. Downloaded via `download_agulhas_prototype.py`. Raw file
  is public/re-downloadable — not archived; cite the DOI.
- **Best-model command:**
  ```
  python train_agulhas_deeponet_prototype.py --cache data/cache_r6.npz --subsample-r 6 \
      --learning-rate 3e-4 --loss-weight none --lambda-div 0 --lambda-geo 0 \
      --iterations 8000 --batch-size 256 --patience 15 --rollout-horizons 1 5 10 20 \
      --out-dir results/best
  ```
- **Methodological note (belongs in Methods/repro):** the persistence residual is
  only an identity if the branch input and the target share a normalisation space;
  an initial per-feature-vs-per-variable mismatch was fixed (init loss 1.05→0.06)
  before any results were collected. See `RESEARCH_LOG.md`.

---

## 8. Figures (generate with `make_figures.py`)

`make_figures.py` reads the saved `.npz`/`.json` and writes publication PNGs — runs
on a laptop, no cluster/data needed.

| Figure | Shows | Data file |
|---|---|---|
| `fig_skill_singlestep.png` | Per-variable 1-day skill vs persistence (bar) | `results/best/metrics.json` |
| `fig_skill_vs_lead.png` | Model skill vs naive over lead time (crosses 0 → useful horizon) | `results/best/rollout.npz` |
| `fig_acc_vs_lead.png` | Model vs naive ACC decay over lead time | `results/best/rollout.npz` |
| `fig_rollout_acc_byvar.png` | Per-variable rollout ACC decay | `results/best/rollout.npz` |
| `fig_spatial_rmse.png` | Map of where the model errs (SSH) | `results/best/spatial_rmse.npz` |
| `fig_parity.png` | Predicted vs true scatter (calibration) | `results/best/predictions.npz` |
| `fig_sweep_skill.png` | Skill by LR × physics — LR matters, physics doesn't | `results/sweep/*/metrics.json` |

---

## 9. Limitations & future work

1. **Physics inert on reanalysis.** The constraints are already satisfied by the
   training data, so soft penalties add nothing. They may matter on sparser/noisier
   observational data, or if reformulated to constrain the *tendency*.
2. **Rollout error accumulation.** The model's 1-day edge does not survive
   autoregressive rollout. Remedies not tested: training *for* rollout (unroll k
   steps during training and penalise drift), or noise injection.
3. **Architecture doesn't scale to fine grids.** The all-sensors branch input
   blows up at r=3. A **patch-based** DeepONet (local sub-regions) is the documented
   remedy.
4. **Scope.** Single region (Agulhas), single-level (surface), reanalysis
   1993–2021 only (the 2021–2024 interim stream and the proposal's exact test years
   were not used). Shared trunk across variables (may under-serve differing
   spatial scales).

---

## 10. One-paragraph conclusion (draft)

We show that a data-driven Deep Operator Network produces a small but consistent
improvement over persistence for one-day surface-eddy forecasting in the Agulhas
Current (mean skill +0.043, concentrated in the velocity fields), but that this
skill does not survive multi-step autoregressive rollout: beyond one day the
model's errors compound and it underperforms a frozen-field baseline. Divergence-
free and geostrophic physics-informed constraints yield no improvement at any
learning rate, loss weighting, forecast step, or resolution — a negative result we
attribute to the reanalysis training data already satisfying those constraints, so
that soft penalties supply no corrective signal. These results delimit where
operator-learning and physics-informed approaches add value for mesoscale ocean
forecasting: a modest single-step gain in the current field, but neither a physics
benefit nor multi-day stability without changes to the training objective or
architecture.
