#!/bin/bash
#SBATCH --job-name=agulhas-deeponet
#SBATCH --output=logs/%x_%j.out         # stdout+stderr → logs/agulhas-deeponet_<jobid>.out
#SBATCH --error=logs/%x_%j.out
#SBATCH --partition=gpu                  # Bouchet: gpu = RTX 5000 Ada (4/node, 2-day cap).
                                         #   gpu_devel → quick tests (≤6 h, faster to start)
                                         #   gpu_h200 / gpu_b200 → only if you need big GPUs
#SBATCH --gpus=rtx_5000_ada:1            # Bouchet requires the GPU TYPE. One is plenty here.
                                         #   (gpu_h200 → --gpus=h200:1, gpu_b200 → --gpus=b200:1)
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G                        # load_nc uses a strided read → RAM tracks the SUBSAMPLED
                                         #   grid, not the raw file; 32G is ample at r=6.
#SBATCH --time=1-00:00:00                # gpu partition allows up to 2-00:00:00.
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=brandon.zhang@yale.edu

# ═══════════════════════════════════════════════════════════════════════════════
# Agulhas physics-informed DeepONet on Yale YCRC — BOUCHET cluster
#
# This setup keeps EVERYTHING in one scratch folder (code + data + results), so all
# paths below are RELATIVE. sbatch runs the job from the directory you submit from,
# so submit from that folder:
#     cd ~/scratch_pi_<pi_netid>/eddy
#     sbatch train_agulhas.slurm
# (Do keep the code in git and copy the small final results/ off scratch — scratch
#  is fast and roomy but unbacked and eventually purged.)
# ═══════════════════════════════════════════════════════════════════════════════
#
# Bouchet storage (quotas):  ~/         → /nfs/roberts/home    (125 GiB / user)
#                            project    → /nfs/roberts/project (4 TiB / group)
#                            scratch    → /nfs/roberts/scratch (10 TiB / group)
#   Group dirs are reached via the shortcuts  ~/project_pi_<pi_netid>
#   and  ~/scratch_pi_<pi_netid>.  Work inside ~/scratch_pi_<pi_netid>/eddy.
#
# ── ONE-TIME SETUP (run once on a LOGIN node, not in this job) ──────────────────
#   module load miniconda
#   conda create -y -n agulhas python=3.11 pytorch pytorch-cuda=12.1 \
#         numpy netcdf4 -c pytorch -c nvidia -c conda-forge
#   conda activate agulhas
#   pip install copernicusmarine            # needed only for the download step
#   # (Your macOS .venv/ does NOT run on the Linux cluster — rebuild here.
#   #  pytorch-cuda=12.1 is fine for the RTX 5000 Ada gpu partition. Only the
#   #  Blackwell partitions (gpu_b200) would need a newer CUDA / nightly torch.)
#
# ── DOWNLOAD THE DATA (LOGIN node — compute nodes have no internet) ─────────────
#   cd ~/scratch_pi_<pi_netid>/eddy         # same folder you submit the job from
#   module load miniconda && conda activate agulhas
#   copernicusmarine login                  # caches credentials once
#   # Full record = two GLORYS streams (no single request spans the ~2021-06 seam);
#   # both land in data/ and the trainer stitches them via the glob below.
#   python download_agulhas_prototype.py --dataset my       # 1993 → 2021-06 (~60 GB)
#   python download_agulhas_prototype.py --dataset myint    # 2021-07 → 2024
# ═══════════════════════════════════════════════════════════════════════════════

# NB: intentionally NOT using `set -u` — conda's activation scripts reference
#     unbound vars (e.g. PS1) and would abort the job under nounset.
set -eo pipefail
mkdir -p logs results

# ── Environment ─────────────────────────────────────────────────────────────
module purge
module load miniconda
conda activate agulhas                   # <-- the env you created above

echo "Host: $(hostname)   Cluster: ${SLURM_CLUSTER_NAME:-?}   GPU: ${CUDA_VISIBLE_DEVICES:-none}"
nvidia-smi -L || true
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

# ── Config ──────────────────────────────────────────────────────────────────
NC="data/agulhas_*.nc"                   # glob — stitches the my + myint streams
R=6                                      # subsample stride (eddy resolution vs cost)
ITERS=20000                              # upper bound; early stopping usually ends sooner
BATCH=256                                # minibatch; drop/raise to fit GPU memory
# NB: --nc is passed QUOTED below so the shell does not glob-expand it — the
#     trainer expands 'data/agulhas_*.nc' itself and stitches the files.
COMMON="--subsample-r ${R} --iterations ${ITERS} --batch-size ${BATCH} \
        --display-every 200 --patience 20 --val-fraction 0.15 --test-fraction 0.15 \
        --rollout-horizons 1 5 10 20"

# ── 1. Physics-informed run ─────────────────────────────────────────────────
echo "=== Physics-informed run ==="
python train_agulhas_deeponet_prototype.py --nc "${NC}" ${COMMON} \
    --lambda-div 0.1 --lambda-geo 0.1 --warmup-steps 1000 \
    --out-dir results/agulhas_pinn

# ── 2. Data-driven baseline + PINN-vs-baseline comparison ───────────────────
echo "=== Data-driven baseline + comparison ==="
python train_agulhas_deeponet_prototype.py --nc "${NC}" ${COMMON} \
    --lambda-div 0.0 --lambda-geo 0.0 \
    --out-dir results/agulhas_baseline \
    --compare-to results/agulhas_pinn

echo "=== Done. Metrics in results/*/metrics.json, figures in results/*/*.svg ==="
