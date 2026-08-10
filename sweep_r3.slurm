#!/bin/bash
#SBATCH --job-name=agulhas-r3
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.out
#SBATCH --partition=gpu
#SBATCH --gpus=rtx_5000_ada:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G                        # r=3 ≈ 4× the sensors of r=6; build_dataset
                                          #   holds several float64 copies → needs headroom.
#SBATCH --time=2-00:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=brandon.zhang@yale.edu

# ═══════════════════════════════════════════════════════════════════════════════
# Focused r=3 experiment: does the physics constraint help at (near) eddy-resolving
# resolution?  Runs the winning config (Δt=1, lr3e-4, plain MSE) with physics OFF
# vs ON — the single comparison that could still change the conclusion.
#   sbatch sweep_r3.slurm            (from ~/scratch_pi_<pi_netid>/eddy)
#
# Notes:
#  - Builds a NEW data/cache_r3.npz (one-time ~10-min decompress of the 27 GB file).
#  - r=3 → ~24k sensors, a ~57M-param model, bigger matmuls → slower than r=6.
#  - If the GPU OOMs, drop --batch-size to 64. If the node OOMs, raise --mem.
# ═══════════════════════════════════════════════════════════════════════════════

set -eo pipefail                          # NOT -u: conda activation needs unbound vars
mkdir -p logs results

module purge
module load miniconda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate agulhas
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

# ── Config ──────────────────────────────────────────────────────────────────
NC="${NC:-data/agulhas_*.nc}"
R=3
CACHE="data/cache_r${R}.npz"
ITERS="${ITERS:-8000}"
OUTROOT="results/sweep_r3"
COMMON="--cache ${CACHE} --subsample-r ${R} --iterations ${ITERS} --batch-size 128 \
        --display-every 200 --patience 15 --val-fraction 0.15 --test-fraction 0.15 \
        --step-days 1 --learning-rate 3e-4 --loss-weight none --lambda-div 0.0 \
        --rollout-horizons 1 5 10 20"

mkdir -p "${OUTROOT}"

# ── 1. Require a pre-built cache (build it on a CPU node, NOT here) ──────────
# Decompressing the 27 GB file is CPU-only; doing it in this GPU job would leave
# the GPU idle long enough for the >1h-idle watchdog to kill the job.
if [ ! -f "${CACHE}" ]; then
  echo "ERROR: ${CACHE} not found. Build it first on a CPU node (no GPU-idle limit):"
  echo "    R=${R} sbatch build_cache.slurm"
  echo "then re-submit this job."
  exit 1
fi

# ── 2. Physics OFF vs ON (resumable: skips finished configs) ────────────────
for LGEO in 0.0 0.05; do
  TAG="r3_step1_lr3e-4_none_geo${LGEO}"
  OUT="${OUTROOT}/${TAG}"
  if [ -f "${OUT}/metrics.json" ]; then echo "skip ${TAG} (done)"; continue; fi
  echo "=== ${TAG} ==="
  python train_agulhas_deeponet_prototype.py ${COMMON} \
      --lambda-geo "${LGEO}" --out-dir "${OUT}" || echo "FAILED ${TAG}"
done

# ── 3. Leaderboard (ranked by mean skill) ───────────────────────────────────
echo
echo "=== r=3 LEADERBOARD (ranked by mean SKILL) ==="
python - "${OUTROOT}" <<'PY'
import json, glob, os, sys
import numpy as np
root = sys.argv[1]
rows = []
for mj in glob.glob(os.path.join(root, "*", "metrics.json")):
    m = json.load(open(mj))
    tag = os.path.basename(os.path.dirname(mj))
    mean_skill = float(np.mean([m[k] for k in m if k.startswith("skill_")]))
    acc20 = float("nan")
    rz = os.path.join(os.path.dirname(mj), "rollout.npz")
    if os.path.exists(rz):
        d = np.load(rz); hz = list(d["horizons"]); macc = np.nanmean(d["acc"], axis=1)
        acc20 = float(macc[hz.index(20)]) if 20 in hz else float(macc[-1])
    rows.append((mean_skill, m.get("val_mse_unweighted", float("nan")),
                 acc20, m.get("lambda_geo"), m.get("steps_run"), tag))
rows.sort(key=lambda r: -(r[0] if r[0] == r[0] else -1e9))
print(f"{'mean_skill':>11} {'val_MSE':>10} {'rollACC@20d':>11} {'geo':>5} {'steps':>7}  config")
for s, v, a20, g, st, tag in rows:
    print(f"{s:>+11.4f} {v:>10.5f} {a20:>11.3f} {str(g):>5} {str(st):>7}  {tag}")
if len(rows) == 2:
    dphys = rows[0][0] - rows[1][0]
    print(f"\nΔ(skill) best−worst = {dphys:+.4f}  "
          f"→ compare geo0.0 vs geo0.05 to judge whether physics helps at r=3.")
PY
echo "=== r=3 sweep done ==="
