#!/bin/bash
#SBATCH --job-name=agulhas-satellite-val-cpu
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.out
#SBATCH --partition=day                  # CPU only -- this mode is pure inference (checkpoint
                                         #   reuse), no backward pass, no GPU needed.
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G                       # loads r=6 AND r=3 full-scale cubes
#SBATCH --time=04:00:00                  # inference + interpolation over ~1561 test days;
                                         #   should finish in well under an hour, margin only
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=brandon.zhang@yale.edu

# ═══════════════════════════════════════════════════════════════════════════════
# CPU-only variant of satellite_validation.slurm -- use this ONLY when reusing an
# existing checkpoint (CHECKPOINT=... required below). No training happens here,
# so no GPU is needed; this just schedules faster on the CPU partition instead of
# waiting in the GPU queue.
#
#   CHECKPOINT=results/agulhas_baseline/model.pt sbatch satellite_validation_cpu.slurm
#
# If you don't have an existing checkpoint and need to train one fresh, use
# satellite_validation.slurm (GPU) instead -- this script will refuse to run
# without CHECKPOINT set.
# ═══════════════════════════════════════════════════════════════════════════════

set -eo pipefail
mkdir -p logs results

module purge
module load miniconda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate agulhas
python -c "import torch; print('torch', torch.__version__, 'cuda available (expected False here):', torch.cuda.is_available())"

NC="data/agulhas_*.nc"
CACHE_R6="data/cache_r6.npz"
CACHE_R3="data/cache_r3.npz"
SAT="data/satellite_sla_agulhas_c2_2019_2021.nc"
OUT="results/satellite_validation_fullscale"

if [ -z "${CHECKPOINT}" ]; then
  echo "ERROR: this CPU-only script requires an existing checkpoint (no training here)."
  echo "  CHECKPOINT=results/agulhas_baseline/model.pt sbatch satellite_validation_cpu.slurm"
  echo "Or use satellite_validation.slurm (GPU) to train a fresh one."
  exit 1
fi
if [ ! -f "${CHECKPOINT}" ]; then
  echo "ERROR: CHECKPOINT=${CHECKPOINT} not found."
  exit 1
fi

for f in "${CACHE_R6}" "${SAT}"; do
  if [ ! -f "${f}" ]; then
    echo "ERROR: ${f} not found."
    exit 1
  fi
done
if ! ls ${CACHE_R3} >/dev/null 2>&1; then
  echo "ERROR: ${CACHE_R3} not found. Build it first: R=3 sbatch build_cache.slurm"
  exit 1
fi

echo "Reusing existing checkpoint (CPU inference only): ${CHECKPOINT}"
python validate_satellite_fullscale.py --nc "${NC}" \
    --cache-r6 "${CACHE_R6}" --cache-r3 "${CACHE_R3}" \
    --satellite "${SAT}" --checkpoint "${CHECKPOINT}" \
    --seed 2026 --out-dir "${OUT}"

echo "=== done: ${OUT} ==="
