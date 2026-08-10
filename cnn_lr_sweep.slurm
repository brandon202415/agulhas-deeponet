#!/bin/bash
#SBATCH --job-name=agulhas-cnn-lrsweep
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.out
#SBATCH --partition=gpu
#SBATCH --gpus=rtx_5000_ada:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G                       # r=3 cube needs headroom (see build_cache.slurm); r=6 fit in 32G but this covers both
#SBATCH --time=1-00:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=brandon.zhang@yale.edu

# ═══════════════════════════════════════════════════════════════════════════════
# Learning-rate sweep for the non-persistence CNN (U-Net) baseline (review item:
# no non-persistence learned baseline anywhere in the paper). Tests the SAME
# three learning rates as Table 2's DeepONet sweep (1e-3, 3e-4, 1e-4), at plain
# (unweighted) loss, single seed, full-scale r=6 reanalysis -- so the CNN gets
# exactly the same chance the DeepONet got to find its own best LR before any
# "DeepONet vs CNN" comparison is made. Skipping this step and testing the CNN
# at only one LR would be a strawman: this study's own Sec. 3.2/4.3 found LR was
# THE decisive factor separating a near-zero-skill DeepONet from a real one, so
# the same risk applies here.
#
#   sbatch cnn_lr_sweep.slurm         (from ~/scratch_pi_<pi_netid>/eddy)
#
# Requires the r=6 cache already built (R=6 sbatch build_cache.slurm). Resumable:
# configs with metrics.json already present are skipped.
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
OUTROOT="results/cnn_lr_sweep_r${R}"
COMMON="--nc ${NC} --cache ${CACHE} --subsample-r ${R} --iterations ${ITERS} \
        --batch-size 64 --base-width 24 --patience 15 --loss-weight none \
        --rollout-horizons 1 5 10 20 --seed 2026"

LRS="1e-3 3e-4 1e-4"

if [ ! -f "${CACHE}" ]; then
  echo "ERROR: ${CACHE} not found. Build it first: R=${R} sbatch build_cache.slurm"
  exit 1
fi

mkdir -p "${OUTROOT}"

for LR in ${LRS}; do
  TAG="lr${LR}"
  OUT="${OUTROOT}/${TAG}"
  if [ -f "${OUT}/metrics.json" ]; then echo "skip ${TAG} (done)"; continue; fi
  echo "=== ${TAG} ==="
  python train_cnn_baseline.py ${COMMON} --learning-rate "${LR}" --out-dir "${OUT}" \
      || echo "FAILED ${TAG}"
done

echo
echo "=== CNN LR-SWEEP LEADERBOARD (ranked by best_val_loss) ==="
python - "${OUTROOT}" <<'PY'
import json, glob, os, sys
import numpy as np
root = sys.argv[1]
rows = []
for mj in glob.glob(os.path.join(root, "*", "metrics.json")):
    m = json.load(open(mj))
    tag = os.path.basename(os.path.dirname(mj))
    mean_skill = float(np.mean([m[k] for k in m if k.startswith("skill_")]))
    rows.append((m.get("best_val_loss", float("nan")), mean_skill, m.get("learning_rate"), tag))
rows.sort(key=lambda r: r[0] if r[0] == r[0] else 1e9)
print(f"{'best_val_loss':>14} {'mean_skill':>11} {'lr':>8}  config")
for v, s, lr, tag in rows:
    print(f"{v:>14.5f} {s:>+11.4f} {str(lr):>8}  {tag}")
if rows:
    print(f"\nBest (val loss): {rows[0][3]}")
PY
echo "=== cnn lr sweep done ==="
