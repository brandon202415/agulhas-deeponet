#!/bin/bash
#SBATCH --job-name=agulhas-cache
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.out
#SBATCH --partition=day                   # CPU partition — NO GPU, so the GPU-idle
                                          #   watchdog does not apply to this job.
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G                        # decompress + float32 cube; r=3 needs headroom
#SBATCH --time=08:00:00                    # bulk reads should finish in minutes; margin only
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=brandon.zhang@yale.edu

# ═══════════════════════════════════════════════════════════════════════════════
# Build the subsampled .npz cache on a CPU node — decompressing the 27 GB NetCDF is
# CPU-only and takes ~10-20 min, during which a GPU would sit idle and get the job
# killed by the >1h-idle watchdog. So do it here (no GPU), then the GPU sweeps just
# load the cache in seconds.
#   R=3 sbatch build_cache.slurm      # build data/cache_r3.npz  (default R=3)
#   R=6 sbatch build_cache.slurm      # build data/cache_r6.npz
# ═══════════════════════════════════════════════════════════════════════════════

set -eo pipefail
mkdir -p logs data

module purge
module load miniconda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate agulhas

NC="${NC:-data/agulhas_*.nc}"
R="${R:-3}"
CACHE="data/cache_r${R}.npz"

echo "Building ${CACHE} from ${NC} (r=${R}) on $(hostname) …"
python train_agulhas_deeponet_prototype.py --nc "${NC}" --subsample-r "${R}" \
    --cache "${CACHE}" --prepare-cache
ls -lh "${CACHE}"
echo "=== cache built: ${CACHE} ==="
