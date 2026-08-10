#!/bin/bash
#SBATCH --job-name=agulhas-cnn-baseline
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.out
#SBATCH --partition=gpu
#SBATCH --gpus=rtx_5000_ada:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=brandon.zhang@yale.edu

# ═══════════════════════════════════════════════════════════════════════════════
# Non-persistence learned baseline (second review round, reviewer item): a small
# U-Net on the same gridded state, same split, same persistence-residual trick as
# the whole-domain DeepONet, to test whether the patch DeepONet's grid-skill
# advantage is architecture-specific or "any reasonably-sized network helps."
#
# SUPERSEDED for official numbers: this was the first, single-seed exploratory
# CNN check, predating the LR investigation that established 1e-3 (not 3e-4,
# the DeepONet's rate) as the CNN's own correct rate (RESEARCH_LOG.md item 33;
# manuscript Sec. 2.3/4.4). Use cnn_seed_sweep.slurm for anything cited in the
# paper. LR below fixed to 1e-3 for consistency (2026-08-05, Issue 32b) even
# though this script's own output isn't the official source.
#
# Requires the r=6 cache already built (R=6 sbatch build_cache.slurm).
#   sbatch cnn_baseline.slurm         (from ~/scratch_pi_<pi_netid>/eddy)
# ═══════════════════════════════════════════════════════════════════════════════

set -eo pipefail
mkdir -p logs results

module purge
module load miniconda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate agulhas
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

R=6
CACHE="data/cache_r${R}.npz"
NC="data/agulhas_*.nc"
SEED="${SEED:-2026}"
OUT="results/cnn_baseline_r6_seed${SEED}"

if [ ! -f "${CACHE}" ]; then
  echo "ERROR: ${CACHE} not found. Build it first: R=${R} sbatch build_cache.slurm"
  exit 1
fi

if [ -f "${OUT}/metrics.json" ]; then
  echo "skip (done): ${OUT}"
else
  python train_cnn_baseline.py --nc "${NC}" --cache "${CACHE}" --subsample-r "${R}" \
      --iterations 5000 --learning-rate 1e-3 --batch-size 64 --base-width 24 \
      --patience 15 --val-fraction 0.15 --test-fraction 0.15 \
      --seed "${SEED}" --out-dir "${OUT}" || echo "FAILED cnn_baseline seed${SEED}"
fi

echo "=== done: ${OUT} ==="
