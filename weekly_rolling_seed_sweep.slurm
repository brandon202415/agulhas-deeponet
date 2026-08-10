#!/bin/bash
#SBATCH --job-name=agulhas-weekly-seedsweep
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.out
#SBATCH --partition=gpu
#SBATCH --gpus=rtx_5000_ada:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=brandon.zhang@yale.edu

# ═══════════════════════════════════════════════════════════════════════════════
# Multi-seed reruns of the rolling-weekly-mean forecast (weekly_rolling_fullscale.slurm),
# same rigor standard as seed_sweep.slurm / patch_seed_sweep.slurm. The single-seed
# full-scale run (seed 2026) found single-step (7d) skill ~0 across most variables
# but a rollout skill that GROWS with lead time instead of decaying:
#   naive-vs-model skill  7d +0.009 -> 14d +0.018 -> 21d +0.033 -> 28d +0.048
# -- the opposite of the daily model's Table 3 decay pattern, and the clearest
# candidate positive finding from the whole 2026-07-27/28 search. Before reporting
# it as confirmed, rerun at 5 seeds to check the rollout-skill trend (not just the
# endpoint numbers) is real and not seed noise, exactly as done for the patch
# architecture claim.
#
# Also runs a --no-residual ablation at the same 5 seeds, anticipating the
# reviewer objection "of course you beat persistence, the model is initialised
# AS persistence." The residual is a trainability aid (removing it elsewhere in
# this study caused outright training collapse, e.g. skill -0.499 on sparse
# reconstruction), not a scoring shortcut -- skill is still measured against an
# independently-computed persistence baseline on held-out data, and several
# other configs in this study demonstrably learn residuals that land BELOW
# persistence despite the same zero-init trick (daily model's Table 3 decay to
# negative; multi-fidelity negative transfer). This ablation makes that
# argument concrete for this specific claim: if --no-residual still trains to a
# meaningful, comparable rollout-skill trend, that's direct evidence the growth
# pattern isn't an artifact of the initialisation.
#
#   sbatch weekly_rolling_seed_sweep.slurm     (from ~/scratch_pi_<pi_netid>/eddy)
#
# Resumable: configs with metrics.json already present are skipped. Requires the
# r=6 daily cache (R=6 sbatch build_cache.slurm); builds the rolling weekly cache
# once (fast, in-memory cumsum) and reuses it across all 5 seeds x 2 configs.
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
ITERS="${ITERS:-8000}"
OUTROOT="results/weekly_rolling_seed_sweep"
COMMON="--cache ${WEEKLY_CACHE} --subsample-r 6 --step-days 7 --learning-rate 3e-4 \
        --lambda-geo 0.0 --lambda-div 0.0 --iterations ${ITERS} --batch-size 256 \
        --patience 15 --rollout-horizons 7 14 21 28"

SEEDS="${SEEDS:-2026 7 42 123 2027}"   # same 5 seeds as seed_sweep.slurm / patch_seed_sweep.slurm

# Named configs to test across seeds. Format "tag:extra_args".
CONFIGS=(
  "weekly_r6:"
  "weekly_r6_noresidual:--no-residual"
)

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

mkdir -p "${OUTROOT}"

for CFG in "${CONFIGS[@]}"; do
  TAG="${CFG%%:*}"
  ARGS="${CFG#*:}"
  for SEED in ${SEEDS}; do
    RUNTAG="${TAG}_seed${SEED}"
    OUT="${OUTROOT}/${RUNTAG}"
    if [ -f "${OUT}/metrics.json" ]; then echo "skip ${RUNTAG} (done)"; continue; fi
    echo "=== ${RUNTAG} ==="
    python train_agulhas_deeponet_prototype.py ${COMMON} ${ARGS} \
        --seed "${SEED}" --out-dir "${OUT}" || echo "FAILED ${RUNTAG}"
  done
done

echo
echo "=== mean +/- std across seeds (single-step skill + rollout skill by horizon) ==="
python aggregate_seed_sweep.py "${OUTROOT}"
echo "=== weekly rolling seed sweep done ==="
