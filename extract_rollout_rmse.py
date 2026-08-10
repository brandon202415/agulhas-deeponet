#!/usr/bin/env python3
"""Extract absolute (non-ratio) rollout error trajectories from already-saved
rollout.npz files -- no retraining needed, these arrays were already computed
and saved by every run in weekly_rolling_seed_sweep.slurm.

Motivation (review comment): Table 13 reports skill = 1-(rmse_model/rmse_persist)^2
only. That ratio can't distinguish "model error shrinking" from "persistence
error exploding while model error stays flat" -- two very different findings,
only the first of which supports a "the model learned multi-week dynamics"
reading. Because units differ across variables (zos ~0.08 m, mlotst ~20 m,
thetao ~1 degC), a cross-variable mean of raw RMSE is not meaningful the way
skill's dimensionless ratio is; this script instead reports (a) mean NRMSE
across variables (dimensionless, valid to average) as the aggregate absolute-
error trend, and (b) per-variable raw RMSE for both model and persistence at
every horizon, mirroring Table 4's per-variable style.

Usage: python extract_rollout_rmse.py results/weekly_rolling_seed_sweep
"""
import glob
import os
import re
import sys
from collections import defaultdict

import numpy as np


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "results/weekly_rolling_seed_sweep"
    runs = defaultdict(dict)
    for npz_path in glob.glob(os.path.join(root, "*", "rollout.npz")):
        run_dir = os.path.dirname(npz_path)
        run_name = os.path.basename(run_dir)
        m = re.match(r"^(.*)_seed(-?\d+)$", run_name)
        if not m:
            continue
        tag, seed = m.group(1), int(m.group(2))
        d = np.load(npz_path)
        runs[tag][seed] = {
            "horizons": d["horizons"],
            "variables": [str(v) for v in d["variables"]],
            "rmse": d["rmse"],                    # [n_horizons, n_vars]
            "rmse_persist": d["rmse_persist"],
            "nrmse": d["nrmse"],
            "skill": d["skill"],                  # [n_horizons, n_vars], 1-(rmse/rmse_persist)^2
        }

    if not runs:
        print(f"No rollout.npz files found under {root}/*/rollout.npz")
        return

    for tag in sorted(runs):
        seeds = runs[tag]
        any_run = next(iter(seeds.values()))
        horizons = any_run["horizons"]
        variables = any_run["variables"]

        nrmse_mat = np.stack([np.nanmean(v["nrmse"], axis=1) for v in seeds.values()])
        print(f"\n=== {tag} (n={len(seeds)} seeds) — mean NRMSE across variables "
              f"(dimensionless, valid to average) ===")
        print(f"{'horizon':>8} {'NRMSE_model (mean+/-std)':>28}")
        for hi, h in enumerate(horizons):
            vals = nrmse_mat[:, hi]
            m = vals.mean()
            s = vals.std(ddof=1) if len(vals) > 1 else 0.0
            print(f"{h:>8} {m:>16.4f} +/- {s:<10.4f}")

        rmse_mat = np.stack([v["rmse"] for v in seeds.values()])          # [n_seeds, n_h, n_vars]
        rmse_p_mat = np.stack([v["rmse_persist"] for v in seeds.values()])
        print(f"\n=== {tag} — per-variable raw RMSE, mean +/- std across seeds "
              f"(physical units) ===")
        for hi, h in enumerate(horizons):
            print(f"\n-- horizon {h}d --")
            print(f"{'var':>8} {'rmse_model':>18} {'rmse_persist':>18}")
            for vi, vname in enumerate(variables):
                rm = rmse_mat[:, hi, vi]
                rp = rmse_p_mat[:, hi, vi]
                rm_m, rm_s = rm.mean(), rm.std(ddof=1) if len(rm) > 1 else 0.0
                rp_m, rp_s = rp.mean(), rp.std(ddof=1) if len(rp) > 1 else 0.0
                print(f"{vname:>8} {rm_m:>10.4f}+/-{rm_s:<6.4f} "
                      f"{rp_m:>10.4f}+/-{rp_s:<6.4f}")

        skill_mat = np.stack([v["skill"] for v in seeds.values()])        # [n_seeds, n_h, n_vars]
        print(f"\n=== {tag} — per-variable skill vs. persistence, mean +/- std "
              f"across {skill_mat.shape[0]} seeds (dimensionless: 1-(rmse_model/rmse_persist)^2) ===")
        print(f"{'var':>8} " + " ".join(f"{h:>8}d" for h in horizons))
        for vi, vname in enumerate(variables):
            row = []
            for hi in range(len(horizons)):
                vals = skill_mat[:, hi, vi]
                m = vals.mean()
                s = vals.std(ddof=1) if len(vals) > 1 else 0.0
                row.append(f"{m:+.4f}+/-{s:.4f}")
            print(f"{vname:>8} " + "  ".join(row))


if __name__ == "__main__":
    main()
