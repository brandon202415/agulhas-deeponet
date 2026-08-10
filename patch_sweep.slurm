#!/bin/bash
#SBATCH --job-name=agulhas-patch
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.out
#SBATCH --partition=gpu
#SBATCH --gpus=rtx_5000_ada:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G                        # r=3 ≈ 4x the sensors of r=6; build_dataset
                                          #   holds several float64 copies -> needs headroom
                                          #   (same reason sweep_r3.slurm uses 128G).
#SBATCH --time=2-00:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=brandon.zhang@yale.edu

# ═══════════════════════════════════════════════════════════════════════════════
# Full-scale validation of the patch-based DeepONet prototype (RESEARCH_LOG.md,
# 2026-07-27 entry). Local testing on the small 1001-day prototype file found:
#   - patch parameter count flat across r (966K at r=6 AND r=3, vs. whole-domain's
#     14.2M -> 56.1M blowup)
#   - patch skill does NOT collapse at r=3 (+0.054 at both r=6 and r=3, vs.
#     whole-domain's +0.022 -> +0.006 collapse)
#   - patch even beat whole-domain at r=6 on that small dataset (+0.054 vs +0.022)
# All of that was on 1001 days / one seed / one untuned tile size. This script
# reruns the same comparison on the REAL 27 GB reanalysis (10,408 days, the
# paper's actual train/val/test split) to see if it holds up at full scale —
# the honest next step before treating the prototype result as a real finding.
#
#   sbatch patch_sweep.slurm         (from ~/scratch_pi_<pi_netid>/eddy)
#
# Requires the r=6 AND r=3 caches to already exist:
#   R=6 sbatch build_cache.slurm
#   R=3 sbatch build_cache.slurm
# Resumable: configs with metrics.json already present are skipped.
# ═══════════════════════════════════════════════════════════════════════════════

set -eo pipefail
mkdir -p logs results

module purge
module load miniconda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate agulhas
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

ITERS="${ITERS:-8000}"          # match the main study's best-run iteration budget
OUTROOT="results/patch_sweep"
mkdir -p "${OUTROOT}"

# Grid: r in {6, 3} (the exact pair that showed the collapse/no-collapse contrast
# locally) x tile size in {20, 30} sensors (20x20 was the untuned local choice;
# 30x30 tests whether a bigger local receptive field helps). Stride is always
# half the tile size (50% overlap). Edit freely.
RS="6 3"
TILES="20 30"

for R in ${RS}; do
  CACHE="data/cache_r${R}.npz"
  if [ ! -f "${CACHE}" ]; then
    echo "ERROR: ${CACHE} not found. Build it first (no GPU-idle limit):"
    echo "    R=${R} sbatch build_cache.slurm"
    exit 1
  fi
  for TILE in ${TILES}; do
    STRIDE=$((TILE / 2))
    TAG="patch_r${R}_tile${TILE}"
    OUT="${OUTROOT}/${TAG}"
    if [ -f "${OUT}/metrics.json" ]; then echo "skip ${TAG} (done)"; continue; fi
    echo "=== ${TAG} ==="
    python train_agulhas_deeponet_patch.py \
        --cache "${CACHE}" --subsample-r "${R}" \
        --patch-h "${TILE}" --patch-w "${TILE}" --patch-stride "${STRIDE}" \
        --learning-rate 3e-4 --iterations "${ITERS}" --batch-size 256 --patience 15 \
        --val-fraction 0.15 --test-fraction 0.15 \
        --out-dir "${OUT}" || echo "FAILED ${TAG}"
  done
done

# Also rerun the whole-domain baseline at r=3 with the SAME cache/iterations/seed
# for a matched comparison point (results/best already has the r=6 one; RESULTS.md
# section 6c has the original r=3 whole-domain numbers, but not at this exact
# iteration budget/cache — rerun for an apples-to-apples match if not already done).
WHOLE_R3="results/whole_r3_matched"
if [ ! -f "${WHOLE_R3}/metrics.json" ]; then
  echo "=== whole-domain r=3 (matched) ==="
  python train_agulhas_deeponet_prototype.py \
      --cache data/cache_r3.npz --subsample-r 3 \
      --learning-rate 3e-4 --loss-weight none --lambda-div 0 --lambda-geo 0 \
      --iterations "${ITERS}" --batch-size 128 --patience 15 \
      --val-fraction 0.15 --test-fraction 0.15 --rollout-horizons 1 5 10 20 \
      --out-dir "${WHOLE_R3}" || echo "FAILED whole-domain r=3"
fi

echo
echo "=== LEADERBOARD (mean skill, patch vs whole-domain, by r and tile size) ==="
python - "${OUTROOT}" "${WHOLE_R3}" <<'PY'
import json, glob, os, sys
import numpy as np
root, whole_r3 = sys.argv[1], sys.argv[2]
rows = []
for mj in glob.glob(os.path.join(root, "*", "metrics.json")):
    m = json.load(open(mj))
    tag = os.path.basename(os.path.dirname(mj))
    rows.append((tag, m.get("mean_skill", float("nan")),
                 m.get("n_params_patch"), m.get("n_params_whole_domain_equiv")))
wj = os.path.join(whole_r3, "metrics.json")
if os.path.exists(wj):
    m = json.load(open(wj))
    skill = float(np.mean([m[k] for k in m if k.startswith("skill_")]))
    rows.append(("whole_domain_r3_matched", skill, None, None))
rows.sort(key=lambda r: -(r[1] if r[1] == r[1] else -1e9))
print(f"{'config':<28} {'mean_skill':>11} {'params(patch)':>14} {'params(whole-eq)':>18}")
for tag, skill, pp, pw in rows:
    print(f"{tag:<28} {skill:>+11.4f} {str(pp) if pp else '-':>14} {str(pw) if pw else '-':>18}")
PY
echo "=== patch sweep done ==="
