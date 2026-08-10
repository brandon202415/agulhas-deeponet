#!/bin/bash
#SBATCH --job-name=agulhas-weekly-rolling
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.out
#SBATCH --partition=gpu
#SBATCH --gpus=rtx_5000_ada:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=brandon.zhang@yale.edu

# ═══════════════════════════════════════════════════════════════════════════════
# Full-scale test of the rolling-weekly-mean, genuine 7-day-ahead forecast idea
# (local-prototype version showed near-zero single-step skill but a rollout skill
# that GROWS with lead time -- 1wk +0.004 -> 4wk +0.030 -- the opposite of the
# daily model's decay-to-negative pattern in Table 3. Testing whether that
# rollout-stability pattern holds at full scale, on the real reanalysis.)
#
# Builds the rolling-mean cache from the already-built daily cache (fast --
# an in-memory cumulative-sum operation, not a NetCDF re-read, so this is done
# in the same GPU job rather than a separate CPU pre-step), then trains the
# EXISTING, unmodified whole-domain trainer on it with --step-days 7 (so entries
# 7 apart in the rolling series are back-to-back non-overlapping weeks -- a
# genuine week-ahead forecast, not a trivially-easy 1-day-shifted average).
#
# Requires the r=6 cache already built (R=6 sbatch build_cache.slurm).
#   sbatch weekly_rolling_fullscale.slurm
# ═══════════════════════════════════════════════════════════════════════════════

set -eo pipefail
mkdir -p logs results data

module purge
module load miniconda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate agulhas
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

NC="data/agulhas_*.nc"
DAILY_CACHE="data/cache_r6.npz"
WEEKLY_CACHE="data/cache_r6_weekly_rolling_fullscale.npz"

if [ ! -f "${DAILY_CACHE}" ]; then
  echo "ERROR: ${DAILY_CACHE} not found. Build it first: R=6 sbatch build_cache.slurm"
  exit 1
fi

if [ ! -f "${WEEKLY_CACHE}" ]; then
  echo "Building rolling weekly-mean cache from ${DAILY_CACHE} ..."
  python prepare_weekly_cache.py --nc "${NC}" --daily-cache "${DAILY_CACHE}" \
      --subsample-r 6 --window 7 --out "${WEEKLY_CACHE}"
else
  echo "Reusing existing ${WEEKLY_CACHE}"
fi

python train_agulhas_deeponet_prototype.py \
    --nc "${NC}" --cache "${WEEKLY_CACHE}" --subsample-r 6 \
    --step-days 7 --learning-rate 3e-4 --lambda-geo 0.0 --lambda-div 0.0 \
    --iterations 8000 --batch-size 256 --patience 15 --seed 2026 \
    --rollout-horizons 7 14 21 28 \
    --out-dir results/weekly_rolling_r6_fullscale \
    2>&1 | tee results/weekly_rolling_r6_fullscale.log

echo "=== done ==="
