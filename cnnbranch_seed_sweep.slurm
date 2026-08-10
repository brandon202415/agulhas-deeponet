#!/bin/bash
#SBATCH --job-name=agulhas-cnnbranch-seedsweep
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.out
#SBATCH --partition=gpu
#SBATCH --gpus=rtx_5000_ada:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1-12:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=brandon.zhang@yale.edu

# ═══════════════════════════════════════════════════════════════════════════════
# Multi-seed, full-scale run of the CNN-branch DeepONet at its winning learning
# rate, to the same rigor standard as every other central result in this study
# (5 seeds: 2026, 7, 42, 123, 2027). Reports single-step (Table-1-equivalent)
# and rollout (Table-3-equivalent) skill.
#
# IMPORTANT: run cnnbranch_lr_sweep.slurm FIRST and check its leaderboard.
# Default below is lr=3e-4 (the original DeepONet's own winner) as a prior,
# but override with CNNBRANCH_LR=<value> if the sweep picks something else --
# do not skip checking, same reasoning as cnn_seed_sweep.slurm.
#
#   sbatch cnnbranch_lr_sweep.slurm                        # first
#   CNNBRANCH_LR=3e-4 sbatch cnnbranch_seed_sweep.slurm     # then this
#
# Requires the r=6 cache already built. Resumable. Reuses aggregate_seed_sweep.py
# and extract_rollout_rmse.py unmodified (same rollout.npz/metrics.json schema
# as the rest of this study, since this model plugs into the existing
# run_training/evaluate/rollout_evaluate functions unchanged).
# ═══════════════════════════════════════════════════════════════════════════════

set -eo pipefail
mkdir -p logs results

module purge
module load miniconda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate agulhas
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

NC="data/agulhas_*.nc"
R=6
CACHE="data/cache_r${R}.npz"
ITERS="${ITERS:-8000}"
LR="${CNNBRANCH_LR:-3e-4}"
OUTROOT="results/cnnbranch_seed_sweep"
COMMON="--nc ${NC} --cache ${CACHE} --subsample-r ${R} --iterations ${ITERS} \
        --batch-size 256 --patience 15 --loss-weight none --learning-rate ${LR} \
        --rollout-horizons 1 5 10 20"

SEEDS="${SEEDS:-2026 7 42 123 2027}"

if [ ! -f "${CACHE}" ]; then
  echo "ERROR: ${CACHE} not found. Build it first: R=${R} sbatch build_cache.slurm"
  exit 1
fi

echo "Using CNN-branch DeepONet learning rate: ${LR}  (check cnnbranch_lr_sweep.slurm's leaderboard first!)"
mkdir -p "${OUTROOT}"

for SEED in ${SEEDS}; do
  RUNTAG="cnnbranch_r6_lr${LR}_seed${SEED}"
  OUT="${OUTROOT}/${RUNTAG}"
  if [ -f "${OUT}/metrics.json" ]; then echo "skip ${RUNTAG} (done)"; continue; fi
  echo "=== ${RUNTAG} ==="
  python train_agulhas_deeponet_cnnbranch.py ${COMMON} --seed "${SEED}" --out-dir "${OUT}" \
      || echo "FAILED ${RUNTAG}"
done

echo
echo "=== mean +/- std across seeds (single-step skill) ==="
python aggregate_seed_sweep.py "${OUTROOT}"
echo
echo "=== per-variable rollout skill/RMSE/NRMSE across seeds ==="
python extract_rollout_rmse.py "${OUTROOT}"
echo "=== cnn-branch deeponet seed sweep done ==="
