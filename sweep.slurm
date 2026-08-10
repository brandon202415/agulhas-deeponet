#!/bin/bash
#SBATCH --job-name=agulhas-sweep
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.out
#SBATCH --partition=gpu
#SBATCH --gpus=rtx_5000_ada:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00                # many small runs; gpu partition max is 2 days
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=brandon.zhang@yale.edu

# ═══════════════════════════════════════════════════════════════════════════════
# Val-based hyperparameter sweep for the Agulhas DeepONet (Bouchet).
#   sbatch sweep.slurm         (from ~/scratch_pi_<pi_netid>/eddy)
#
# Builds the subsampled cache ONCE, then runs a grid of configs — each loads the
# cache in seconds instead of re-decompressing the 27 GB NetCDF — and prints a
# leaderboard ranked by validation loss. Resumable: configs with a metrics.json
# already present are skipped, so you can re-run after a timeout/preemption.
# Edit the grid (LRS / LWS / LGEOS) and ITERS below to taste.
# ═══════════════════════════════════════════════════════════════════════════════

set -eo pipefail                          # NOT -u: conda activation needs unbound vars
mkdir -p logs results

module purge
module load miniconda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate agulhas
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

# ── Config ──────────────────────────────────────────────────────────────────
NC="${NC:-data/agulhas_*.nc}"             # raw data (only read once, to build cache)
R="${R:-6}"
CACHE="data/cache_r${R}.npz"
ITERS="${ITERS:-8000}"
OUTROOT="results/sweep"
COMMON="--cache ${CACHE} --subsample-r ${R} --iterations ${ITERS} --batch-size 256 \
        --display-every 200 --patience 15 --val-fraction 0.15 --test-fraction 0.15 \
        --lambda-div 0.0 --rollout-horizons 1 5 10 20"

# Grid to sweep (edit freely). Defaults target the current open question — does a
# longer forecast step help, and does physics help there? — at the LR/loss-weight
# that won the earlier tuning sweep (lr3e-4, plain MSE). Widen any list to explore
# more (e.g. LRS="1e-3 3e-4 1e-4", LWS="none variability").
LRS="3e-4"
LWS="none"
LGEOS="0.0 0.05"
SDS="1 5 10"                              # --step-days values (forecast step Δt)

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

# ── 2. Grid (resumable: skips configs already done) ─────────────────────────
for SD in ${SDS}; do
  for LR in ${LRS}; do
    for LW in ${LWS}; do
      for LGEO in ${LGEOS}; do
        TAG="step${SD}_lr${LR}_${LW}_geo${LGEO}"
        OUT="${OUTROOT}/${TAG}"
        if [ -f "${OUT}/metrics.json" ]; then echo "skip ${TAG} (done)"; continue; fi
        echo "=== ${TAG} ==="
        python train_agulhas_deeponet_prototype.py ${COMMON} \
            --step-days "${SD}" --learning-rate "${LR}" --loss-weight "${LW}" \
            --lambda-geo "${LGEO}" --out-dir "${OUT}" || echo "FAILED ${TAG}"
      done
    done
  done
done

# ── 3. Leaderboard ──────────────────────────────────────────────────────────
echo
echo "=== LEADERBOARD (ranked by mean SKILL — persistence-relative, comparable across steps) ==="
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
        d = np.load(rz)
        # ACC at 20-day lead if present (20 is a multiple of steps 1/5/10 → comparable)
        hz = list(d["horizons"]); macc = np.nanmean(d["acc"], axis=1)
        acc20 = float(macc[hz.index(20)]) if 20 in hz else float(macc[-1])
    rows.append((mean_skill, m.get("val_mse_unweighted", float("nan")),
                 acc20, m.get("step_days"), m.get("steps_run"), tag))
# Skill is persistence-relative (scale-free), so it IS comparable across step_days
# and loss-weight, unlike val_MSE (which grows with the forecast step). Higher = better.
rows.sort(key=lambda r: -(r[0] if r[0] == r[0] else -1e9))
print(f"{'mean_skill':>11} {'val_MSE':>10} {'rollACC@20d':>11} {'Δt':>4} {'steps':>7}  config")
for s, v, a20, sd, st, tag in rows:
    print(f"{s:>+11.4f} {v:>10.5f} {a20:>11.3f} {str(sd):>4} {str(st):>7}  {tag}")
if rows:
    print(f"\nBest (mean skill): {rows[0][5]}   | skill>0 beats persistence. "
          f"val_MSE is comparable only WITHIN a fixed Δt.")
PY
echo "=== sweep done ==="
