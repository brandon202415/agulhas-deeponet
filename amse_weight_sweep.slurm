#!/bin/bash
#SBATCH --job-name=agulhas-amse-weightsweep
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.out
#SBATCH --partition=gpu
#SBATCH --gpus=rtx_5000_ada:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G                       # r=3 cube needs headroom (see build_cache.slurm); r=6 fit in 32G but this covers both
#SBATCH --time=1-00:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=brandon.zhang@yale.edu

# ═══════════════════════════════════════════════════════════════════════════════
# Full-scale weight sweep for the AMSE spectral loss (train_cnn_baseline_amse.py,
# adapting Subich et al. 2025's double-penalty fix -- see RESEARCH_LOG.md items
# 27-28 for the mechanism and the local-scale result this sweep is meant to
# confirm at full scale). Local screen (700 training days, 800 iterations,
# 3 seeds) found weight=0.02 gave a consistent, mostly-significant eddy-tracking
# position-error improvement (RESEARCH_LOG.md item 28) while weight=1.0 badly
# damaged overall grid skill (item 27) -- this sweep brackets the local winner
# at full scale before committing to a 5-seed confirmation.
#
# Mirrors cnn_lr_sweep.slurm's pattern exactly: same LR (3e-4, the CNN baseline's
# own winner per cnn_lr_sweep.slurm's leaderboard), single seed (2026), full-scale
# r=6 reanalysis, testing weight instead of LR. IMPORTANT: run this BEFORE
# amse_seed_sweep.slurm and check the leaderboard below -- do not assume the
# local-scale winning weight (0.02) transfers unchanged to 10x the training data,
# any more than this study assumes a CNN's best LR equals a DeepONet's.
#
#   sbatch amse_weight_sweep.slurm            (from ~/scratch_pi_<pi_netid>/eddy)
#
# Requires the r=6 cache already built (R=6 sbatch build_cache.slurm). Resumable:
# configs with metrics.json already present are skipped.
#
# NOTE: this sweep only reports grid-point skill (Table-1-equivalent columns),
# same as cnn_lr_sweep.slurm -- it does NOT run eddy-tracking (py-eddy-tracker
# requires the dedicated local `eddytrack` conda env, see eddy_tracking/README.md,
# and does not run on this cluster environment). After this sweep picks a
# weight, download that config's predictions.npz and run
# eddy_tracking/eddy_tracking_analysis.py + eddy_stat_test.py locally, exactly
# as items 27/28 did, before trusting any full-scale eddy-tracking claim.
# ═══════════════════════════════════════════════════════════════════════════════

set -eo pipefail
mkdir -p logs results

module purge
module load miniconda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate agulhas
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

NC="data/agulhas_*.nc"
R="${R:-6}"
CACHE="data/cache_r${R}.npz"
ITERS="${ITERS:-8000}"
LR="${CNN_LR:-1e-3}"
OUTROOT="results/amse_weight_sweep_r${R}_lr${LR}"
COMMON="--nc ${NC} --cache ${CACHE} --subsample-r ${R} --iterations ${ITERS} \
        --batch-size 64 --base-width 24 --patience 15 \
        --learning-rate ${LR} --rollout-horizons 1 5 10 20 --seed 2026"

WEIGHTS="${WEIGHTS:-0.0 0.01 0.02 0.05}"

if [ ! -f "${CACHE}" ]; then
  echo "ERROR: ${CACHE} not found. Build it first: R=${R} sbatch build_cache.slurm"
  exit 1
fi

mkdir -p "${OUTROOT}"

for W in ${WEIGHTS}; do
  TAG="amse_w${W}"
  OUT="${OUTROOT}/${TAG}"
  if [ -f "${OUT}/metrics.json" ]; then echo "skip ${TAG} (done)"; continue; fi
  echo "=== ${TAG} ==="
  python train_cnn_baseline_amse.py ${COMMON} --amse-weight "${W}" --out-dir "${OUT}" \
      || echo "FAILED ${TAG}"
done

echo
echo "=== AMSE WEIGHT-SWEEP LEADERBOARD (ranked by best_val_loss) ==="
python - "${OUTROOT}" <<'PY'
import json, glob, os, sys
import numpy as np
root = sys.argv[1]
rows = []
for mj in glob.glob(os.path.join(root, "*", "metrics.json")):
    m = json.load(open(mj))
    tag = os.path.basename(os.path.dirname(mj))
    mean_skill = float(np.mean([m[k] for k in m if k.startswith("skill_")]))
    rows.append((m.get("best_val_loss", float("nan")), mean_skill, m.get("skill_zos"),
                 m.get("amse_weight"), tag))
rows.sort(key=lambda r: r[0] if r[0] == r[0] else 1e9)
print(f"{'best_val_loss':>14} {'mean_skill':>11} {'zos_skill':>10} {'amse_weight':>12}  config")
for v, s, z, w, tag in rows:
    print(f"{v:>14.5f} {s:>+11.4f} {z:>+10.4f} {str(w):>12}  {tag}")
if rows:
    print(f"\nBest (val loss): {rows[0][4]}")
print("\nDon't pick the weight on grid skill alone -- item 27 showed weight=1.0 keeps")
print("zos skill high while grid skill and rollout both collapse. Cross-check against")
print("the local screen's eddy-tracking result (item 28) before choosing a weight for")
print("amse_seed_sweep.slurm.")
PY
echo "=== amse weight sweep done ==="
