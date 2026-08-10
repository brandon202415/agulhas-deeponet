#!/bin/bash
#SBATCH --job-name=agulhas-resaugment
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.out
#SBATCH --partition=gpu
#SBATCH --gpus=rtx_5000_ada:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G                       # loads BOTH r=6 and r=3 cubes (r=3 alone needed
                                         #   the 128G bump in patch_sweep.slurm; same here).
#SBATCH --time=1-00:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=brandon.zhang@yale.edu

# ═══════════════════════════════════════════════════════════════════════════════
# Resolution-augmented DeepONet training + architecturally-fixed zero-shot r=3
# transfer test (Sec. 3.7 follow-up, "positive DeepONet result" track, second
# review round).
#
# Sec. 3.7's local-prototype result (+0.022 zero-shot skill at r=3 query points)
# used a manual forward-pass workaround because the persistence-residual skip was
# hard-locked to the branch's own sensor count. train_agulhas_deeponet_prototype.py's
# MultivarDeepONet.forward() now accepts an explicit persist_at_query tensor, making
# the residual (and so the whole architecture) genuinely resolution-independent.
# This script trains WITH that fix, mixing standard r=6-query steps with steps that
# query a random subset of r=3 points (using r=3's own persistence field) -- testing
# whether the fix + augmentation gives a stronger, more genuine positive result than
# the local-prototype workaround.
#
# Requires BOTH r=6 and r=3 caches already built:
#   R=6 sbatch build_cache.slurm
#   R=3 sbatch build_cache.slurm
#   sbatch deeponet_resaugment.slurm         (from ~/scratch_pi_<pi_netid>/eddy)
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
SEED="${SEED:-2026}"
OUT="results/resaugment_r6r3_seed${SEED}"

for f in "${CACHE_R6}"; do
  if [ ! -f "${f}" ]; then
    echo "ERROR: ${f} not found. Build it first: R=6 sbatch build_cache.slurm"
    exit 1
  fi
done
# cache_r3.npz is a glob match check via ls since load_states/build_cache write it directly
if ! ls ${CACHE_R3} >/dev/null 2>&1; then
  echo "ERROR: ${CACHE_R3} not found. Build it first: R=3 sbatch build_cache.slurm"
  exit 1
fi

if [ -f "${OUT}/metrics.json" ]; then
  echo "skip (done): ${OUT}"
else
  python train_deeponet_resaugment.py --nc "${NC}" \
      --cache-r6 "${CACHE_R6}" --cache-r3 "${CACHE_R3}" \
      --iterations 8000 --learning-rate 3e-4 --batch-size 32 \
      --aug-prob 0.5 --aug-n-query 6161 --patience 15 \
      --val-fraction 0.15 --test-fraction 0.15 \
      --seed "${SEED}" --out-dir "${OUT}" || echo "FAILED resaugment seed${SEED}"
fi

echo "=== done: ${OUT} ==="
