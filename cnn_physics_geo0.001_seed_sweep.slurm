#!/bin/bash
#SBATCH --job-name=agulhas-cnn-physics-geo0.001-seedsweep
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.out
#SBATCH --partition=gpu
#SBATCH --gpus=rtx_5000_ada:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=1-12:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=brandon.zhang@yale.edu

# ═══════════════════════════════════════════════════════════════════════════════
# 5-seed confirmation of the single-seed result from cnn_physics_weight_sweep.slurm
# (2026-08-06 real cluster output): at lambda_geo=0.1 (the DeepONet's own weight,
# Table 8) the CNN loses -0.0602 mean skill and 20-day rollout collapses to -4.467;
# at lambda_geo=0.001 (two orders of magnitude smaller), single-seed skill diff is
# +0.0029 (statistically negligible) and 20-day rollout is -0.034, close to the
# plain control's -0.019. This is the dose-response finding manuscript Sec. 4.4/
# Table 9 now reports -- but only as single-seed evidence. This script is the
# natural next step this project's own convention requires before that finding
# is promoted to a headline table: the exact same single-seed-screen-then-5-seed-
# confirmation workflow used for the CNN's own learning-rate sweep
# (cnn_lr_sweep.slurm -> cnn_seed_sweep.slurm) and for the original w=0.1 physics
# result (which WAS already run at 5 seeds, cnn_physics_seed_sweep.slurm).
#
# Fixed at lambda_geo=0.001 (not re-swept) because that was the single-seed
# winner on BOTH metrics (best single-step skill AND best rollout stability among
# the tested weights) -- do not change this without re-reading the weight-sweep
# leaderboard first. lambda_div stays off (0.0) for the same reason every other
# physics-informed run in this project leaves it off: divergence is already
# trivially satisfied on reanalysis and only adds noise.
#
#   sbatch cnn_physics_geo0.001_seed_sweep.slurm
#
# Requires the r=6 cache already built (same one cnn_seed_sweep.slurm uses).
# Resumable: seeds with metrics.json already present are skipped. Output schema
# matches cnn_seed_sweep.slurm's, so aggregate_seed_sweep.py and
# extract_rollout_rmse.py work unmodified.
#
# After this completes, compare against the existing r=6 CNN control
# (results/cnn_seed_sweep/cnn_r6_lr1e-3_seed*/metrics.json) with a seed-matched
# paired comparison on mean_skill and 20-day rollout skill, mirroring exactly how
# Table 8's original w=0.1 comparison was built -- that comparison is not run
# automatically by this script since it needs both result sets to exist first.
# Report whichever way the real numbers point, not only if they confirm the
# single-seed pattern already in the manuscript.
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
GEO_WEIGHT="${GEO_WEIGHT:-0.001}"
DIV_WEIGHT="${DIV_WEIGHT:-0.0}"
OUTROOT="results/cnn_physics_geo${GEO_WEIGHT}_div${DIV_WEIGHT}_r${R}"
COMMON="--nc ${NC} --cache ${CACHE} --subsample-r ${R} --iterations ${ITERS} \
        --batch-size 64 --base-width 24 --patience 15 --loss-weight none \
        --learning-rate ${LR} --rollout-horizons 1 5 10 20 \
        --lambda-geo ${GEO_WEIGHT} --lambda-div ${DIV_WEIGHT} --warmup-steps 500"

SEEDS="${SEEDS:-2026 7 42 123 2027}"

if [ ! -f "${CACHE}" ]; then
  echo "ERROR: ${CACHE} not found. Build it first: R=${R} sbatch build_cache.slurm"
  exit 1
fi

echo "CNN physics-informed confirmation: lr=${LR} lambda_geo=${GEO_WEIGHT} lambda_div=${DIV_WEIGHT}"
mkdir -p "${OUTROOT}"

for SEED in ${SEEDS}; do
  RUNTAG="cnn_physics_geo${GEO_WEIGHT}_r${R}_lr${LR}_seed${SEED}"
  OUT="${OUTROOT}/${RUNTAG}"
  if [ -f "${OUT}/metrics.json" ]; then echo "skip ${RUNTAG} (done)"; continue; fi
  echo "=== ${RUNTAG} ==="
  python train_cnn_baseline.py ${COMMON} --seed "${SEED}" --out-dir "${OUT}" \
      || echo "FAILED ${RUNTAG}"
done

echo
echo "=== mean +/- std across seeds (single-step skill) ==="
python aggregate_seed_sweep.py "${OUTROOT}"
echo
echo "=== per-variable rollout skill/RMSE/NRMSE across seeds ==="
python extract_rollout_rmse.py "${OUTROOT}"
echo "=== cnn physics geo0.001 confirmation sweep done ==="
