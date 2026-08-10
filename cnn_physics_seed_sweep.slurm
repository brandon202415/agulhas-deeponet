#!/bin/bash
#SBATCH --job-name=agulhas-cnn-physics-seedsweep
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.out
#SBATCH --partition=gpu
#SBATCH --gpus=rtx_5000_ada:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=1-12:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=brandon.zhang@yale.edu

# ═══════════════════════════════════════════════════════════════════════════════
# Issue 27 (referee report, 2026-08-05): Sec. 2.7/4.7's physics-informed
# (divergence-free/geostrophic) null result was only ever tested on the
# whole-domain DeepONet -- the architecture this paper's own Sec. 1/3.1
# argument establishes as the wrong choice for this task, not the CNN used
# everywhere else in the paper. This re-runs the identical experiment on the
# CNN, at the SAME 5-seed rigor as every other full-scale claim in this study.
#
# Uses lr=1e-3, the CNN's own established-correct rate (Sec. 2.3/4.4) --
# NOT cnn_seed_sweep.slurm's default of 3e-4, which is the DeepONet's rate
# and predates that script's own LR-bug fix. Do not copy that default here.
#
# Uses --lambda-geo 0.1 to replicate the DeepONet's own winning physics
# config (Sec. 2.7); --lambda-div stays at 0 (default off) for the same
# reason the DeepONet trainer defaults it off: divergence is trivially
# satisfied on reanalysis (~1e-12, confirmed again in the local smoke test
# for this exact CNN integration, RESEARCH_LOG.md item 56) and only adds
# noise. Override via GEO_WEIGHT=<value> if you want a different weight.
#
#   sbatch cnn_physics_seed_sweep.slurm
#
# Requires the r=6 cache already built (same one cnn_seed_sweep.slurm uses).
# Resumable: seeds with metrics.json already present are skipped. Output
# schema is identical to cnn_seed_sweep.slurm's (same metrics.json/
# rollout.npz/predictions.npz layout), so aggregate_seed_sweep.py and
# extract_rollout_rmse.py work unmodified, and predictions.npz can be fed
# straight into eddy_tracking_analysis.py exactly like every other config
# in this study if an eddy-tracking comparison is wanted later.
#
# After this completes, compare against the existing r=6 CNN control
# (results/cnn_seed_sweep/cnn_r6_lr1e-3_seed*/metrics.json) with a
# seed-matched paired t-test on mean_skill, mirroring how Sec. 3.3 compares
# AMSE against the CNN control -- that comparison is not run automatically
# by this script since it needs both result sets to exist first.
# ═══════════════════════════════════════════════════════════════════════════════

set -eo pipefail
mkdir -p logs results

module purge
module load miniconda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate agulhas
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

NC="data/agulhas_*.nc"
R="${R:-6}"
CACHE="data/cache_r${R}.npz"
ITERS="${ITERS:-8000}"
LR="${CNN_LR:-1e-3}"
GEO_WEIGHT="${GEO_WEIGHT:-0.1}"
DIV_WEIGHT="${DIV_WEIGHT:-0.0}"
OUTROOT="results/cnn_physics_geo${GEO_WEIGHT}_div${DIV_WEIGHT}_r${R}"
COMMON="--nc ${NC} --cache ${CACHE} --subsample-r ${R} --iterations ${ITERS} \
        --batch-size 64 --base-width 24 --patience 15 --loss-weight none \
        --learning-rate ${LR} --rollout-horizons 1 5 10 20 \
        --lambda-geo ${GEO_WEIGHT} --lambda-div ${DIV_WEIGHT} --warmup-steps 500"

SEEDS="${SEEDS:-2026 7 42 123 2027}"

if [ ! -f "${CACHE}" ]; then
  echo "ERROR: ${CACHE} not found. Build it first: R=${R} sbatch build_cache.slurm"
  exit 1
fi

echo "CNN physics-informed sweep: lr=${LR} lambda_geo=${GEO_WEIGHT} lambda_div=${DIV_WEIGHT}"
mkdir -p "${OUTROOT}"

for SEED in ${SEEDS}; do
  RUNTAG="cnn_physics_geo${GEO_WEIGHT}_r${R}_lr${LR}_seed${SEED}"
  OUT="${OUTROOT}/${RUNTAG}"
  if [ -f "${OUT}/metrics.json" ]; then echo "skip ${RUNTAG} (done)"; continue; fi
  echo "=== ${RUNTAG} ==="
  python train_cnn_baseline.py ${COMMON} --seed "${SEED}" --out-dir "${OUT}" \
      || echo "FAILED ${RUNTAG}"
done

echo
echo "=== mean +/- std across seeds (single-step skill) ==="
python aggregate_seed_sweep.py "${OUTROOT}"
echo
echo "=== per-variable rollout skill/RMSE/NRMSE across seeds ==="
python extract_rollout_rmse.py "${OUTROOT}"
echo "=== cnn physics seed sweep done ==="
