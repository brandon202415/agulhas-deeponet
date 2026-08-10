#!/bin/bash
#SBATCH --job-name=agulhas-patch-hist
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.out
#SBATCH --partition=gpu
#SBATCH --gpus=rtx_5000_ada:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G                        # r=3 needs headroom (same reason as patch_sweep.slurm /
                                          #   sweep_r3.slurm); history days don't change this, they
                                          #   reuse the already-loaded raw states cube.
#SBATCH --time=2-00:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=brandon.zhang@yale.edu

# ═══════════════════════════════════════════════════════════════════════════════
# Full-scale validation of multi-day input history (RESEARCH_LOG.md, 2026-07-27
# "Trying to raise skill further" entry). Local testing on the small 1001-day
# prototype file found 2-day history the only lever (of: longer training + LR
# decay, bigger model, 2/3-day history, cross-tile attention) that actually
# raised skill -- +0.0616 vs the plain patch baseline's +0.0544 at r=6, and
# +0.0628 vs +0.0541 at r=3 -- while 3-day history was WORSE than 2-day. This
# reruns exactly that comparison on the REAL 27 GB reanalysis (10,408 days, the
# paper's actual split), the same way patch_sweep.slurm validated the base patch
# architecture, to see if it holds at full scale before trusting it further.
#
#   sbatch patch_history_sweep.slurm         (from ~/scratch_pi_<pi_netid>/eddy)
#
# Requires the r=6 and r=3 caches (same as patch_sweep.slurm):
#   R=6 sbatch build_cache.slurm
#   R=3 sbatch build_cache.slurm
# Resumable: configs with metrics.json already present are skipped. Also prints
# the baseline (history=1, no-history) numbers from results/patch_sweep/ for
# comparison if that directory exists (it should, from the earlier patch run).
# ═══════════════════════════════════════════════════════════════════════════════

set -eo pipefail
mkdir -p logs results

module purge
module load miniconda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate agulhas
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

ITERS="${ITERS:-8000}"
OUTROOT="results/patch_history_sweep"
mkdir -p "${OUTROOT}"

# Grid: r in {6, 3} x history-days in {2, 3} at tile=20 (matching the local test's
# tile size for direct comparability). history=1 (no history) is already in
# results/patch_sweep/patch_r{6,3}_tile20/ from the earlier patch_sweep.slurm run.
RS="6 3"
HISTS="2 3"
TILE=20
STRIDE=10

for R in ${RS}; do
  CACHE="data/cache_r${R}.npz"
  if [ ! -f "${CACHE}" ]; then
    echo "ERROR: ${CACHE} not found. Build it first (no GPU-idle limit):"
    echo "    R=${R} sbatch build_cache.slurm"
    exit 1
  fi
  for HIST in ${HISTS}; do
    TAG="patch_r${R}_tile${TILE}_hist${HIST}"
    OUT="${OUTROOT}/${TAG}"
    if [ -f "${OUT}/metrics.json" ]; then echo "skip ${TAG} (done)"; continue; fi
    echo "=== ${TAG} ==="
    python train_agulhas_deeponet_patch.py \
        --cache "${CACHE}" --subsample-r "${R}" \
        --patch-h "${TILE}" --patch-w "${TILE}" --patch-stride "${STRIDE}" \
        --n-history-days "${HIST}" \
        --learning-rate 3e-4 --iterations "${ITERS}" --batch-size 256 --patience 15 \
        --val-fraction 0.15 --test-fraction 0.15 \
        --out-dir "${OUT}" || echo "FAILED ${TAG}"
  done
done

echo
echo "=== LEADERBOARD (mean skill, by r and history length; tile=${TILE}) ==="
python - "${OUTROOT}" "results/patch_sweep" "${TILE}" <<'PY'
import json, glob, os, sys
root, base_root, tile = sys.argv[1], sys.argv[2], sys.argv[3]
rows = []
for mj in glob.glob(os.path.join(root, "*", "metrics.json")):
    m = json.load(open(mj))
    tag = os.path.basename(os.path.dirname(mj))
    rows.append((tag, m.get("subsample_r"), m.get("n_history_days", 1),
                 m.get("mean_skill"), m.get("n_params_patch"), m.get("steps_run")))
# Pull in the history=1 baselines from the earlier patch_sweep.slurm run, if present.
for r in (6, 3):
    bj = os.path.join(base_root, f"patch_r{r}_tile{tile}", "metrics.json")
    if os.path.exists(bj):
        m = json.load(open(bj))
        rows.append((f"patch_r{r}_tile{tile}_hist1 (baseline)", r, 1,
                     m.get("mean_skill"), m.get("n_params_patch"), m.get("steps_run")))
rows.sort(key=lambda x: (x[1], x[2]))
print(f"{'config':<40} {'r':>3} {'hist':>5} {'mean_skill':>11} {'params':>10} {'steps':>7}")
for tag, r, h, skill, params, steps in rows:
    skill_s = f"{skill:+.4f}" if skill is not None else "-"
    print(f"{tag:<40} {str(r):>3} {str(h):>5} {skill_s:>11} {str(params):>10} {str(steps):>7}")
PY
echo "=== patch history sweep done ==="
