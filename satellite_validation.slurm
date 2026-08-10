#!/bin/bash
#SBATCH --job-name=agulhas-satellite-val
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.out
#SBATCH --partition=gpu
#SBATCH --gpus=rtx_5000_ada:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G                       # loads r=6 AND r=3 full-scale cubes
#SBATCH --time=1-00:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=brandon.zhang@yale.edu

# ═══════════════════════════════════════════════════════════════════════════════
# Real-satellite validation, full scale: does DeepONet's direct continuous query
# beat "predict on the r=6 grid, then interpolate" against REAL CryoSat-2
# along-track sea-level observations, at REAL satellite pass locations?
#
# Unlike the local-prototype version, this needs NO custom split: the paper's
# standard chronological test window (last 15% of 1993-2021, ~2017-03-23 to
# 2021-06-30) already fully contains the satellite data's coverage
# (2019-01-01 to 2020-07-31), so this reuses the REAL, already-established
# whole-domain r=6 checkpoint if you point CHECKPOINT at it -- set the env var
# below, or leave unset to train a fresh one with the standard recipe.
#
# Requires:
#   - r=6 and r=3 caches already built (R=6 / R=3 sbatch build_cache.slurm)
#   - the satellite NetCDF uploaded to data/satellite_sla_agulhas_c2_2019_2021.nc
#     (download it locally first via `copernicusmarine subset`, then scp/rsync
#     it up -- same file already validated locally)
#
#   CHECKPOINT=results/agulhas_baseline/model.pt sbatch satellite_validation.slurm
#   sbatch satellite_validation.slurm                      # trains fresh if unset
# ═══════════════════════════════════════════════════════════════════════════════

set -eo pipefail
mkdir -p logs results

module purge
module load miniconda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate agulhas
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

NC="data/agulhas_*.nc"
CACHE_R6="data/cache_r6.npz"
CACHE_R3="data/cache_r3.npz"
SAT="data/satellite_sla_agulhas_c2_2019_2021.nc"
OUT="results/satellite_validation_fullscale"

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

CKPT_ARG=""
if [ -n "${CHECKPOINT}" ]; then
  if [ ! -f "${CHECKPOINT}" ]; then
    echo "ERROR: CHECKPOINT=${CHECKPOINT} not found."
    exit 1
  fi
  CKPT_ARG="--checkpoint ${CHECKPOINT}"
  echo "Reusing existing checkpoint: ${CHECKPOINT}"
else
  echo "No CHECKPOINT set -- training a fresh whole-domain r=6 model with the standard recipe."
fi

python validate_satellite_fullscale.py --nc "${NC}" \
    --cache-r6 "${CACHE_R6}" --cache-r3 "${CACHE_R3}" \
    --satellite "${SAT}" ${CKPT_ARG} \
    --iterations 8000 --learning-rate 3e-4 --batch-size 256 --patience 15 \
    --seed 2026 --out-dir "${OUT}"

echo "=== done: ${OUT} ==="
