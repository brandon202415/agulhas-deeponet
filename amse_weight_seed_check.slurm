#!/bin/bash
#SBATCH --job-name=agulhas-amse-weightseedcheck
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.out
#SBATCH --partition=gpu
#SBATCH --gpus=rtx_5000_ada:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=brandon.zhang@yale.edu

# ═══════════════════════════════════════════════════════════════════════════════
# Full-scale version of the weight-neighborhood x seed check RESEARCH_LOG.md item
# 29 records as aborted incomplete at local scale (killed at the user's request --
# the 6 background local runs made the machine unusable; none reached a final
# metrics.json). This answers the same question -- is weight=0.02 (item 28's
# local-scale winner) robust across seeds, or was it a lucky single point in
# weight-space -- but at full scale on GPU instead of the 700-day local cache,
# which is a strictly better version of the check, not just a faster one.
#
# Covers weight in {0.01, 0.05} x seed in {7, 42} (4 configs). Seed 2026 at these
# weights is already covered by amse_weight_sweep.slurm's single-seed sweep --
# not duplicated here. Combined with amse_weight_sweep.slurm (weight sweep x 1
# seed) and amse_seed_sweep.slurm (5 seeds x 1 weight, default 0.02), this fills
# in the weight x seed grid at full scale without re-running configs that already
# exist elsewhere.
#
#   sbatch amse_weight_seed_check.slurm        (from ~/scratch_pi_<pi_netid>/eddy)
#
# Requires the r=6 cache already built (R=6 sbatch build_cache.slurm). Resumable:
# configs with metrics.json already present are skipped. Does NOT run
# eddy-tracking (needs the local `eddytrack` conda env) -- download predictions.npz
# and run eddy_tracking_analysis.py + eddy_stat_test.py locally afterward, same as
# every other AMSE result so far (items 27/28).
# ═══════════════════════════════════════════════════════════════════════════════

set -eo pipefail
mkdir -p logs results

module purge
module load miniconda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate agulhas
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

NC="data/agulhas_*.nc"
R=6
CACHE="data/cache_r${R}.npz"
ITERS="${ITERS:-8000}"
LR="${CNN_LR:-1e-3}"
OUTROOT="results/amse_weight_seed_check_lr${LR}"
COMMON="--nc ${NC} --cache ${CACHE} --subsample-r ${R} --iterations ${ITERS} \
        --batch-size 64 --base-width 24 --patience 15 \
        --learning-rate ${LR} --rollout-horizons 1 5 10 20"

WEIGHTS="${WEIGHTS:-0.01 0.05}"
SEEDS="${SEEDS:-7 42}"

if [ ! -f "${CACHE}" ]; then
  echo "ERROR: ${CACHE} not found. Build it first: R=${R} sbatch build_cache.slurm"
  exit 1
fi

mkdir -p "${OUTROOT}"

for W in ${WEIGHTS}; do
  for SEED in ${SEEDS}; do
    TAG="amse_w${W}_seed${SEED}"
    OUT="${OUTROOT}/${TAG}"
    if [ -f "${OUT}/metrics.json" ]; then echo "skip ${TAG} (done)"; continue; fi
    echo "=== ${TAG} ==="
    python train_cnn_baseline_amse.py ${COMMON} --amse-weight "${W}" --seed "${SEED}" \
        --out-dir "${OUT}" || echo "FAILED ${TAG}"
  done
done

echo
echo "=== AMSE WEIGHT x SEED CHECK LEADERBOARD ==="
python - "${OUTROOT}" <<'PY'
import json, glob, os, sys
import numpy as np
root = sys.argv[1]
rows = []
for mj in glob.glob(os.path.join(root, "*", "metrics.json")):
    m = json.load(open(mj))
    tag = os.path.basename(os.path.dirname(mj))
    mean_skill = float(np.mean([m[k] for k in m if k.startswith("skill_")]))
    rows.append((m.get("amse_weight"), m.get("seed"), mean_skill, m.get("skill_zos"),
                 m.get("best_val_loss"), tag))
rows.sort(key=lambda r: (r[0] if r[0] is not None else -1, r[1] if r[1] is not None else -1))
print(f"{'amse_weight':>12} {'seed':>6} {'mean_skill':>11} {'zos_skill':>10} {'best_val_loss':>14}  config")
for w, s, ms, z, v, tag in rows:
    print(f"{str(w):>12} {str(s):>6} {ms:>+11.4f} {z:>+10.4f} {v:>14.5f}  {tag}")
print("\nCompare against results/amse_weight_sweep/ (seed 2026, weights 0/0.01/0.02/0.05)")
print("and results/amse_seed_sweep/ (weight 0.02, 5 seeds) for the full weight x seed picture.")
print("Grid skill alone does not settle this -- download predictions.npz and run the local")
print("eddy-tracking pipeline (eddy_tracking_analysis.py + eddy_stat_test.py) before")
print("concluding anything, exactly as items 27/28 did.")
PY
echo "=== amse weight x seed check done ==="
