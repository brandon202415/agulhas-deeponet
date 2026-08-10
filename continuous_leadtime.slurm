#!/bin/bash
#SBATCH --job-name=agulhas-continuous-leadtime
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
# Continuous lead-time DeepONet: trunk queried at (lon, lat, k) instead of just
# (lon, lat), trained on step_days in {1,5,10}, evaluated zero-shot at unseen
# intermediate horizons k=3,7 against persistence, a naive curve-fit through the
# model's own known-k performance, and dedicated single-horizon models -- the
# temporal analogue of Sec. 3.7's spatial discretization-invariance test.
#
# Uses full-scale data (not the local-prototype file, which only exists on the
# laptop) -- also gives a more convincing test than local-prototype scale, and
# this run moved to the cluster specifically to escape repeated local session
# interruptions that killed it twice on the laptop.
#
# Requires the r=6 cache already built (R=6 sbatch build_cache.slurm).
#   sbatch continuous_leadtime.slurm
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

python train_deeponet_continuous_leadtime.py --nc "${NC}" --cache "${CACHE}" \
    --iterations 8000 --dedicated-iterations 5000 --learning-rate 3e-4 \
    --batch-size 256 --patience 15 --seed 2026 \
    --out-dir results/continuous_leadtime_fullscale \
    2>&1 | tee results/continuous_leadtime_fullscale.log

echo "=== done ==="
