#!/bin/bash
#SBATCH --job-name=agulhas-patch-seedsweep
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.out
#SBATCH --partition=gpu
#SBATCH --gpus=rtx_5000_ada:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G                        # r=6 only in this script; r=6 has never needed the
                                          #   128G bump r=3 needs (see patch_sweep.slurm).
#SBATCH --time=2-00:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=brandon.zhang@yale.edu

# ═══════════════════════════════════════════════════════════════════════════════
# Multi-seed reruns for the patch-architecture claims (PI review, 2026-07-27):
# "confirmed at full scale" for patch and patch+history was single-seed, exactly
# the same shape of claim this study's own Sec 4.3 lesson warns against (a single
# run can hide a confound). This reruns, at 5 seeds each:
#   - whole-domain r=6 (the current paper headline) -- ALREADY covered by
#     seed_sweep.slurm's "geo0.0" config; not repeated here.
#   - patch r=6, 1-day input (tile=20) -- the plain patch result (+0.0748, one seed)
#   - patch r=6, 2-day input history (tile=20) -- the "raise skill" result (+0.0793, one seed)
# so we can report mean +/- std and judge whether +0.075 vs +0.079 (patch vs
# patch+history) and vs +0.043 (whole-domain) are real separations, not seed noise.
#
#   sbatch patch_seed_sweep.slurm         (from ~/scratch_pi_<pi_netid>/eddy)
#   Also run (if not already):  sbatch seed_sweep.slurm   (covers the whole-domain r=6 headline)
#
# Resumable: configs with metrics.json already present are skipped. Requires the
# r=6 cache (R=6 sbatch build_cache.slurm).
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
ITERS="${ITERS:-8000}"
OUTROOT="results/patch_seed_sweep"
COMMON="--cache ${CACHE} --subsample-r ${R} --patch-h 20 --patch-w 20 --patch-stride 10 \
        --learning-rate 3e-4 --iterations ${ITERS} --batch-size 256 --patience 15 \
        --val-fraction 0.15 --test-fraction 0.15"

SEEDS="${SEEDS:-2026 7 42 123 2027}"   # same 5 seeds as seed_sweep.slurm, for consistency

CONFIGS=(
  "patch_r6_hist1:--n-history-days 1"
  "patch_r6_hist2:--n-history-days 2"
)

if [ ! -f "${CACHE}" ]; then
  echo "ERROR: ${CACHE} not found. Build it first on a CPU node:"
  echo "    R=${R} sbatch build_cache.slurm"
  exit 1
fi

mkdir -p "${OUTROOT}"

for CFG in "${CONFIGS[@]}"; do
  TAG="${CFG%%:*}"
  ARGS="${CFG#*:}"
  for SEED in ${SEEDS}; do
    RUNTAG="${TAG}_seed${SEED}"
    OUT="${OUTROOT}/${RUNTAG}"
    if [ -f "${OUT}/metrics.json" ]; then echo "skip ${RUNTAG} (done)"; continue; fi
    echo "=== ${RUNTAG} ==="
    python train_agulhas_deeponet_patch.py ${COMMON} ${ARGS} \
        --seed "${SEED}" --out-dir "${OUT}" || echo "FAILED ${RUNTAG}"
  done
done

echo
echo "=== per-config mean +/- std across seeds (patch_r6_hist1 vs patch_r6_hist2) ==="
python aggregate_seed_sweep.py "${OUTROOT}"
echo "=== patch seed sweep done ==="
