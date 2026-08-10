#!/bin/bash
#SBATCH --job-name=agulhas-sparse-attn
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
# Full-scale run of the geometric-attention sparse-to-dense reconstruction
# prototype (same architecture as the local run, just ~10x more training days:
# 7285 vs 701) -- testing whether the flat local-prototype results across three
# architectures (dense masked vector, set encoder, geometric attention) were a
# data-scale problem rather than an architecture problem, before taking stock.
#
# Requires the r=6 cache already built (R=6 sbatch build_cache.slurm).
#   sbatch sparse_reconstruction_attention.slurm
# ═══════════════════════════════════════════════════════════════════════════════

set -eo pipefail
mkdir -p logs results

module purge
module load miniconda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate agulhas
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

NC="data/agulhas_*.nc"
CACHE="data/cache_r6.npz"

if [ ! -f "${CACHE}" ]; then
  echo "ERROR: ${CACHE} not found. Build it first: R=6 sbatch build_cache.slurm"
  exit 1
fi

python sparse_reconstruction_attention.py --nc "${NC}" --cache "${CACHE}" \
    --coverage 0.10 --iterations 8000 --learning-rate 3e-4 --batch-size 32 \
    --score-hidden 64 --score-depth 3 --query-subsample 400 --patience 15 \
    --seed 2026 2>&1 | tee results/sparse_reconstruction_attention_fullscale.log

echo "=== done ==="
