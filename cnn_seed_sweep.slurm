#!/bin/bash
#SBATCH --job-name=agulhas-cnn-seedsweep
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
# Multi-seed, full-scale run of the CNN (U-Net) baseline at its winning learning
# rate, to the SAME rigor standard as the paper's central DeepONet claims (5
# seeds: 2026, 7, 42, 123, 2027; see seed_sweep.slurm / patch_seed_sweep.slurm).
# Reports single-step skill (Table-1-equivalent columns) AND autoregressive
# rollout at 1/5/10/20 days (Table-3-equivalent), so the CNN can be compared
# directly against both the whole-domain DeepONet's headline numbers.
#
# Default below is lr=1e-3, the CNN's OWN established-correct rate (confirmed
# via cnn_lr_sweep.slurm and used for every official CNN number in this study,
# including the +0.3797+/-0.0089 headline -- see RESEARCH_LOG.md item 33 and
# manuscript Sec. 2.3/4.4). This default was WRONG until 2026-08-05 (silently
# 3e-4, the DeepONet's own rate, copied from the wrong template -- exactly the
# class of bug Sec. 4.4 discloses, and it was still live in this file after
# that disclosure was written until caught by test_lr_config_regression.py,
# Issue 32b). Do not revert this default without re-running cnn_lr_sweep.slurm
# and checking its leaderboard first -- see manuscript Sec. 3.2/4.4 for why LR
# was the single decisive factor separating a real DeepONet from a
# near-zero-skill one, and there is no reason to assume a CNN's optimal LR is
# the same as a DeepONet's a priori.
#
#   sbatch cnn_lr_sweep.slurm                    # first, find the CNN's own best LR
#   sbatch cnn_seed_sweep.slurm                  # then this, at the established-correct rate (default)
#
# Requires the r=6 cache already built (R=6 sbatch build_cache.slurm). Resumable:
# seeds with metrics.json already present are skipped. Reuses
# aggregate_seed_sweep.py (skill) and extract_rollout_rmse.py (rollout skill/
# RMSE/NRMSE by horizon and variable) unmodified -- rollout.npz and metrics.json
# are written in the exact schema those scripts already expect.
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
OUTROOT="results/cnn_seed_sweep_r${R}"
COMMON="--nc ${NC} --cache ${CACHE} --subsample-r ${R} --iterations ${ITERS} \
        --batch-size 64 --base-width 24 --patience 15 --loss-weight none \
        --learning-rate ${LR} --rollout-horizons 1 5 10 20"

SEEDS="${SEEDS:-2026 7 42 123 2027}"

if [ ! -f "${CACHE}" ]; then
  echo "ERROR: ${CACHE} not found. Build it first: R=${R} sbatch build_cache.slurm"
  exit 1
fi

echo "Using CNN learning rate: ${LR}  (check cnn_lr_sweep.slurm's leaderboard first!)"
mkdir -p "${OUTROOT}"

for SEED in ${SEEDS}; do
  RUNTAG="cnn_r${R}_lr${LR}_seed${SEED}"
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
echo "=== cnn seed sweep done ==="
