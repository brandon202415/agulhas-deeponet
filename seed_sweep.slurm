#!/bin/bash
#SBATCH --job-name=agulhas-seedsweep
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.out
#SBATCH --partition=gpu
#SBATCH --gpus=rtx_5000_ada:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=brandon.zhang@yale.edu

# ═══════════════════════════════════════════════════════════════════════════════
# Multi-seed reruns for statistical rigor (reviewer request): every comparison in
# the manuscript's Tables 2/3/5 (sweep leaderboard, rollout, forecast-step) is a
# single seed, single run — this reruns the configs that matter most to the
# central claims across several seeds so we can report mean +/- std instead of a
# lone number, and check whether small deltas (e.g. lambda_geo=0.0 vs 0.05: 0.0432
# vs 0.0393 mean skill) are distinguishable from seed noise.
#
#   sbatch seed_sweep.slurm         (from ~/scratch_pi_<pi_netid>/eddy)
#
# Resumable like sweep.slurm: configs with metrics.json already present are
# skipped. Requires the r=6 cache to already exist (R=6 sbatch build_cache.slurm).
# Edit CONFIGS / SEEDS below to add/remove comparisons.
# ═══════════════════════════════════════════════════════════════════════════════

set -eo pipefail
mkdir -p logs results

module purge
module load miniconda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate agulhas
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

NC="${NC:-data/agulhas_*.nc}"
R="${R:-6}"
CACHE="data/cache_r${R}.npz"
ITERS="${ITERS:-8000}"
OUTROOT="results/seed_sweep"
COMMON="--cache ${CACHE} --subsample-r ${R} --iterations ${ITERS} --batch-size 256 \
        --display-every 200 --patience 15 --val-fraction 0.15 --test-fraction 0.15 \
        --lambda-div 0.0 --rollout-horizons 1 5 10 20 --learning-rate 3e-4 --loss-weight none"

SEEDS="${SEEDS:-2026 7 42 123 2027}"   # 5 seeds by default (2026 reproduces the original best run)

# Named configs to test across seeds. Format "tag:extra_args". Defaults cover the
# central physics ablation (the paper's headline comparison) plus the forecast-step
# sweep (Table 5); extend this list for other comparisons (e.g. r=3, add
# "--subsample-r 3" and a separate CACHE=data/cache_r3.npz block).
CONFIGS=(
  "geo0.0:--step-days 1 --lambda-geo 0.0"
  "geo0.05:--step-days 1 --lambda-geo 0.05"
  "step5_geo0.0:--step-days 5 --lambda-geo 0.0"
  "step10_geo0.0:--step-days 10 --lambda-geo 0.0"
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
    python train_agulhas_deeponet_prototype.py ${COMMON} ${ARGS} \
        --seed "${SEED}" --out-dir "${OUT}" || echo "FAILED ${RUNTAG}"
  done
done

echo
echo "=== per-config mean +/- std across seeds ==="
python aggregate_seed_sweep.py "${OUTROOT}"
echo "=== seed sweep done ==="
