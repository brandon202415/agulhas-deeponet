#!/bin/bash
#SBATCH --job-name=agulhas-cnn-physics-weightsweep
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.out
#SBATCH --partition=gpu
#SBATCH --gpus=rtx_5000_ada:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=brandon.zhang@yale.edu

# ═══════════════════════════════════════════════════════════════════════════════
# Issue 44 (decision-letter-style referee round, 2026-08-05): the CNN
# physics-informed run (cnn_physics_seed_sweep.slurm, Table 8) used
# lambda_geo=0.1 -- the DeepONet's OWN winning weight, chosen for a model
# with 33x more parameters and an order of magnitude worse baseline skill.
# This paper already caught and disclosed (Sec. 4.2) the exact same mistake
# for learning rate -- assuming a hyperparameter tuned on one architecture
# transfers to another -- and flagged it as a real error. Reporting the CNN's
# harm at the DeepONet's weight as an "architecture-dependent" finding without
# first checking whether a CNN-appropriate weight behaves differently repeats
# that mistake rather than correcting for it.
#
# This sweeps lambda_geo across four values spanning two orders of magnitude
# below the DeepONet's winning weight, SINGLE SEED (2026, this project's
# standard first-look seed) at the CNN's own established-correct lr=1e-3
# (Sec. 2.3/4.2 -- NOT cnn_seed_sweep.slurm's historical 3e-4 default),
# mirroring cnn_lr_sweep.slurm's own single-seed-first, sweep-before-committing
# design. lambda_div stays off (default 0) for the same reason
# cnn_physics_seed_sweep.slurm leaves it off: divergence is already trivially
# satisfied on reanalysis (~1e-12) and only adds noise to the comparison.
#
#   sbatch cnn_physics_weight_sweep.slurm
#
# Requires the r=6 cache already built (same one cnn_seed_sweep.slurm uses).
# Resumable: configs with metrics.json already present are skipped. Output
# schema matches cnn_seed_sweep.slurm's, so aggregate_seed_sweep.py works
# unmodified.
#
# Interpreting the result: if EVERY weight tested still costs mean skill
# relative to the plain CNN control (results/cnn_seed_sweep/cnn_r6_lr1e-3_seed2026/
# metrics.json), that is real evidence the harm is not just an untuned-weight
# artifact, and Table 8's architecture-dependence framing is confirmed --
# strengthen it with this sweep's leaderboard. If a smaller weight is neutral
# or helpful, the "architecture-dependent harm" framing needs to be walked
# back to "harmful only at the DeepONet's weight, untested at a CNN-appropriate
# one" and Sec. 4.4/Table 8 need a real rewrite, not a caveat -- whichever the
# real numbers say, report that, not the answer either framing would prefer.
# A promising weight found here should still be confirmed at 5-seed rigor
# (mirroring cnn_physics_seed_sweep.slurm) before being reported as a table,
# exactly as the LR sweep's own winner was before Table 1 was built.
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
SEED="${SEED:-2026}"
OUTROOT="results/cnn_physics_weight_sweep_r${R}"
COMMON="--nc ${NC} --cache ${CACHE} --subsample-r ${R} --iterations ${ITERS} \
        --batch-size 64 --base-width 24 --patience 15 --loss-weight none \
        --learning-rate ${LR} --rollout-horizons 1 5 10 20 \
        --lambda-div 0.0 --warmup-steps 500 --seed ${SEED}"

GEO_WEIGHTS="${GEO_WEIGHTS:-0.1 0.01 0.001 0.0001}"

if [ ! -f "${CACHE}" ]; then
  echo "ERROR: ${CACHE} not found. Build it first: R=${R} sbatch build_cache.slurm"
  exit 1
fi

echo "CNN physics-weight sweep: lr=${LR} seed=${SEED} lambda_geo in [${GEO_WEIGHTS}]"
mkdir -p "${OUTROOT}"

for GW in ${GEO_WEIGHTS}; do
  TAG="geo${GW}"
  OUT="${OUTROOT}/${TAG}"
  if [ -f "${OUT}/metrics.json" ]; then echo "skip ${TAG} (done)"; continue; fi
  echo "=== ${TAG} ==="
  python train_cnn_baseline.py ${COMMON} --lambda-geo "${GW}" --out-dir "${OUT}" \
      || echo "FAILED ${TAG}"
done

echo
echo "=== CNN PHYSICS-WEIGHT SWEEP LEADERBOARD (ranked by mean_skill, vs. control below) ==="
python - "${OUTROOT}" <<'PY'
import json, glob, os, sys
import numpy as np
root = sys.argv[1]
rows = []
for mj in glob.glob(os.path.join(root, "*", "metrics.json")):
    m = json.load(open(mj))
    tag = os.path.basename(os.path.dirname(mj))
    mean_skill = float(np.mean([m[k] for k in m if k.startswith("skill_")]))
    rows.append((mean_skill, m.get("lambda_geo"), tag))
rows.sort(key=lambda r: -r[0])
print(f"{'mean_skill':>11} {'lambda_geo':>11}  config")
for s, gw, tag in rows:
    print(f"{s:>+11.4f} {str(gw):>11}  {tag}")
control_path = f"results/cnn_seed_sweep/cnn_r6_lr1e-3_seed2026/metrics.json"
if os.path.exists(control_path):
    cm = json.load(open(control_path))
    c_skill = float(np.mean([cm[k] for k in cm if k.startswith("skill_")]))
    print(f"\nPlain CNN control (seed 2026): mean_skill={c_skill:+.4f}")
    for s, gw, tag in rows:
        print(f"  {tag}: diff vs. control = {s - c_skill:+.4f}")
else:
    print(f"\n(control file not found at {control_path} -- compare manually)")
PY
echo "=== cnn physics weight sweep done ==="
