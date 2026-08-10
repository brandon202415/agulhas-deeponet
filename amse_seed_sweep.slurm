#!/bin/bash
#SBATCH --job-name=agulhas-amse-seedsweep
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.out
#SBATCH --partition=gpu
#SBATCH --gpus=rtx_5000_ada:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G                       # r=3 cube needs headroom (see build_cache.slurm); r=6 fit in 32G but this covers both
#SBATCH --time=1-12:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=brandon.zhang@yale.edu

# ═══════════════════════════════════════════════════════════════════════════════
# Multi-seed, full-scale confirmation of the AMSE spectral loss (Subich et al.
# 2025 adaptation, RESEARCH_LOG.md items 27-28) at its winning weight, to the
# SAME rigor standard as the CNN baseline's own 5-seed confirmation
# (cnn_seed_sweep.slurm) and this study's other central claims (5 seeds:
# 2026, 7, 42, 123, 2027).
#
# IMPORTANT: run amse_weight_sweep.slurm FIRST and pick a weight using BOTH its
# leaderboard AND the local-scale eddy-tracking result (item 28) -- do not pick
# on grid skill alone (item 27 showed a badly-scaled weight can preserve or
# even inflate zos skill while damaging everything else and collapsing rollout).
# Default below is 0.02 (the local-scale winner) as a reasonable prior; override
# with AMSE_WEIGHT=<value> if the full-scale weight sweep picks a different one.
#
#   sbatch amse_weight_sweep.slurm                     # first
#   AMSE_WEIGHT=0.02 sbatch amse_seed_sweep.slurm       # then this, at the winning weight
#
# Requires the r=6 cache already built (R=6 sbatch build_cache.slurm). Resumable:
# seeds with metrics.json already present are skipped. Reuses
# aggregate_seed_sweep.py / extract_rollout_rmse.py / compare_cnn_vs_deeponet.py
# unmodified -- metrics.json and rollout.npz are written in the exact schema
# those scripts already expect, and predictions.npz matches the schema
# eddy_tracking_analysis.py expects.
#
# THIS SWEEP DOES NOT RUN EDDY-TRACKING (py-eddy-tracker needs the local
# `eddytrack` conda env, not available on this cluster -- see
# eddy_tracking/README.md). After this completes, download each seed's
# predictions.npz and run eddy_tracking_analysis.py + eddy_stat_test.py locally
# against the matched CNN control (results/cnn_seed_sweep, once it exists at
# full scale) exactly as items 23/24/25/27/28 did at local scale -- that
# eddy-tracking confirmation, not this script's grid-skill numbers alone, is
# what would actually substantiate the double-penalty-fix claim at full scale.
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
W="${AMSE_WEIGHT:-0.02}"
OUTROOT="results/amse_seed_sweep_r${R}_lr${LR}"
COMMON="--nc ${NC} --cache ${CACHE} --subsample-r ${R} --iterations ${ITERS} \
        --batch-size 64 --base-width 24 --patience 15 \
        --learning-rate ${LR} --amse-weight ${W} --rollout-horizons 1 5 10 20"

SEEDS="${SEEDS:-2026 7 42 123 2027}"

if [ ! -f "${CACHE}" ]; then
  echo "ERROR: ${CACHE} not found. Build it first: R=${R} sbatch build_cache.slurm"
  exit 1
fi

echo "Using AMSE weight: ${W}  (check amse_weight_sweep.slurm's leaderboard + item 28 first!)"
mkdir -p "${OUTROOT}"

for SEED in ${SEEDS}; do
  RUNTAG="amse_r${R}_w${W}_seed${SEED}"
  OUT="${OUTROOT}/${RUNTAG}"
  if [ -f "${OUT}/metrics.json" ]; then echo "skip ${RUNTAG} (done)"; continue; fi
  echo "=== ${RUNTAG} ==="
  python train_cnn_baseline_amse.py ${COMMON} --seed "${SEED}" --out-dir "${OUT}" \
      || echo "FAILED ${RUNTAG}"
done

echo
echo "=== mean +/- std across seeds (single-step skill) ==="
python aggregate_seed_sweep.py "${OUTROOT}"
echo
echo "=== per-variable rollout skill/RMSE/NRMSE across seeds ==="
python extract_rollout_rmse.py "${OUTROOT}"
echo "=== amse seed sweep done ==="
echo
echo "NEXT STEP (not automated by this script): download results/amse_seed_sweep/*/predictions.npz"
echo "and run the eddy-tracking + significance-test pipeline locally against the matched"
echo "CNN control, mirroring items 23/24/25/27/28 exactly. Grid skill alone does not"
echo "substantiate the double-penalty-fix claim this loss is meant to test."
