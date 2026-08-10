#!/bin/bash
#SBATCH --job-name=agulhas-weekly-embargo
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.out
#SBATCH --partition=gpu
#SBATCH --gpus=rtx_5000_ada:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-12:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=brandon.zhang@yale.edu

# ═══════════════════════════════════════════════════════════════════════════════
# Split-boundary embargo check for the weekly-rolling rollout-skill finding
# (review comment): entry i of the rolling-7-day-mean series covers raw days
# [i, i+6], so two samples within 6 index positions of each other share raw
# underlying days. build_dataset()'s plain chronological split (train | val |
# test, taken in index order) does not account for this the way it does for
# daily data -- the last several training samples and first several val/test
# samples can be near-duplicates in raw data, a leakage risk the daily model
# doesn't have at anywhere near this severity.
#
# Fix: a new --embargo N flag (added to build_dataset() in
# train_agulhas_deeponet_prototype.py, default 0 so every other result in this
# study is unaffected) drops N samples from the end of train and the end of
# val before assigning indices, guaranteeing an index gap > N at both split
# boundaries. window=7 -> embargo=6 is sufficient (verified locally: produces
# a 7-index gap at both boundaries, confirmed >6).
#
# Reruns ONLY the with-residual weekly_r6 config (the config whose rollout-
# skill-growth claim is actually at risk from this leakage) at the same 5
# seeds used throughout this study, with --embargo 6. Does not rerun the
# no-residual ablation (that ablation answers a different question --
# initialization artifact, not split leakage -- and embargo doesn't bear on
# it). Also narrows evaluation focus to thetao/mlotst per the climatology-
# control finding (evaluate_weekly_climatology_baseline.py,
# extract_rollout_rmse.py): those are the only two variables where the
# rollout-skill-growth result survived a climatology baseline check, so
# they're the only ones worth re-validating against split leakage. All 6
# variables are still evaluated (the trainer doesn't support per-variable
# subsetting) -- just read the thetao/mlotst rows as the ones that matter for
# this finding.
#
#   sbatch weekly_rolling_embargo_sweep.slurm     (from ~/scratch_pi_<pi_netid>/eddy)
#
# Resumable: configs with metrics.json already present are skipped. Requires
# the weekly rolling cache already built by weekly_rolling_seed_sweep.slurm /
# weekly_rolling_fullscale.slurm (data/cache_r6_weekly_rolling_fullscale.npz).
# ═══════════════════════════════════════════════════════════════════════════════

set -eo pipefail
mkdir -p logs results data

module purge
module load miniconda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate agulhas
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

WEEKLY_CACHE="data/cache_r6_weekly_rolling_fullscale.npz"
ITERS="${ITERS:-8000}"
OUTROOT="results/weekly_rolling_embargo_sweep"
COMMON="--cache ${WEEKLY_CACHE} --subsample-r 6 --step-days 7 --learning-rate 3e-4 \
        --lambda-geo 0.0 --lambda-div 0.0 --iterations ${ITERS} --batch-size 256 \
        --patience 15 --rollout-horizons 7 14 21 28 --embargo 6"

SEEDS="${SEEDS:-2026 7 42 123 2027}"

if [ ! -f "${WEEKLY_CACHE}" ]; then
  echo "ERROR: ${WEEKLY_CACHE} not found. Build it first:"
  echo "    sbatch weekly_rolling_fullscale.slurm   (or weekly_rolling_seed_sweep.slurm)"
  exit 1
fi

mkdir -p "${OUTROOT}"

for SEED in ${SEEDS}; do
  RUNTAG="weekly_r6_embargo6_seed${SEED}"
  OUT="${OUTROOT}/${RUNTAG}"
  if [ -f "${OUT}/metrics.json" ]; then echo "skip ${RUNTAG} (done)"; continue; fi
  echo "=== ${RUNTAG} ==="
  python train_agulhas_deeponet_prototype.py ${COMMON} \
      --seed "${SEED}" --out-dir "${OUT}" || echo "FAILED ${RUNTAG}"
done

echo
echo "=== mean +/- std across seeds (embargoed split) ==="
python aggregate_seed_sweep.py "${OUTROOT}"
echo "=== weekly rolling embargo sweep done ==="
