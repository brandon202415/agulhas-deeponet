#!/bin/bash
#SBATCH --job-name=agulhas-cnnbranch-lrsweep
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
# Learning-rate sweep for the CNN-branch DeepONet (replaces the whole-domain
# DeepONet's dense-MLP branch with a small conv encoder; trunk, persistence
# residual, and combination mechanism are otherwise IDENTICAL to
# MultivarDeepONet -- see train_agulhas_deeponet_cnnbranch.py's docstring).
# Tests whether fixing only the diagnosed weakness (branch has no spatial
# locality/translation-equivariance) closes most of the gap to the plain CNN
# baseline (which beat the original whole-domain DeepONet 7-9x, single seed,
# MANUSCRIPT_ISSUES.md Issue 3), while keeping the trunk's genuine
# discretization-flexibility property that motivated using DeepONet at all.
#
# Same 3 learning rates as Table 2 / cnn_lr_sweep.slurm, single seed, plain
# loss -- same reasoning as always in this study: LR was THE decisive factor
# for the original DeepONet (Sec. 3.2/4.3), don't assume this variant's
# optimum matches either the original DeepONet's or the CNN's.
#
#   sbatch cnnbranch_lr_sweep.slurm         (from ~/scratch_pi_<pi_netid>/eddy)
#
# Requires the r=6 cache already built (R=6 sbatch build_cache.slurm). Resumable.
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
OUTROOT="results/cnnbranch_lr_sweep"
COMMON="--nc ${NC} --cache ${CACHE} --subsample-r ${R} --iterations ${ITERS} \
        --batch-size 256 --patience 15 --loss-weight none \
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
  python train_agulhas_deeponet_cnnbranch.py ${COMMON} --learning-rate "${LR}" --out-dir "${OUT}" \
      || echo "FAILED ${TAG}"
done

echo
echo "=== CNN-BRANCH DEEPONET LR-SWEEP LEADERBOARD (ranked by best_val_loss) ==="
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
    print(f"cf. whole-domain DeepONet (Table 1): +0.043  |  CNN baseline best: +0.379")
PY
echo "=== cnn-branch deeponet lr sweep done ==="
