#!/usr/bin/env python3
"""Aggregate a seed_sweep.slurm run: mean +/- std across seeds per config, plus a
paired comparison between the two physics configs (geo0.0 vs geo0.05) since they
share the same seed list -- this is the statistical-rigor check the manuscript
currently lacks (reviewer item 1): is the lambda_geo=0.0 vs 0.05 skill gap real or
seed noise?

Usage: python aggregate_seed_sweep.py results/seed_sweep
Reads every results/seed_sweep/<tag>_seed<N>/metrics.json and results/seed_sweep/<tag>_seed<N>/rollout.npz.
No torch dependency -- runs anywhere with numpy.
"""
import json
import glob
import os
import re
import sys
from collections import defaultdict

import numpy as np


def load_runs(root):
    runs = defaultdict(dict)  # tag -> seed -> metrics dict (+ rollout arrays)
    for mj in glob.glob(os.path.join(root, "*", "metrics.json")):
        run_dir = os.path.dirname(mj)
        run_name = os.path.basename(run_dir)
        m = re.match(r"^(.*)_seed(-?\d+)$", run_name)
        if not m:
            continue
        tag, seed = m.group(1), int(m.group(2))
        metrics = json.load(open(mj))
        entry = {"metrics": metrics}
        rz = os.path.join(run_dir, "rollout.npz")
        if os.path.exists(rz):
            d = np.load(rz)
            entry["rollout"] = {
                "horizons": list(d["horizons"]),
                "skill": np.nanmean(d["skill"], axis=1),  # mean over variables per horizon
                "acc": np.nanmean(d["acc"], axis=1),
            }
        runs[tag][seed] = entry
    return runs


def mean_skill(metrics):
    ks = [metrics[k] for k in metrics if k.startswith("skill_")]
    return float(np.mean(ks)) if ks else float("nan")


def summarize(root):
    runs = load_runs(root)
    if not runs:
        print(f"No completed runs found under {root}/ yet (expected <tag>_seed<N>/metrics.json).")
        print("Run `sbatch seed_sweep.slurm` on Bouchet first, then re-run this script.")
        return

    print(f"{'config':<16} {'n_seeds':>7} {'mean_skill (mean+/-std)':>26} {'val_loss (mean+/-std)':>24}")
    tag_mean_skills = {}
    for tag in sorted(runs):
        seeds = runs[tag]
        skills = np.array([mean_skill(v["metrics"]) for v in seeds.values()])
        # val_mse_unweighted (whole-domain trainer) or best_val_loss (patch trainer) --
        # whichever key this config's metrics.json actually has.
        val_mse = np.array([
            v["metrics"].get("val_mse_unweighted", v["metrics"].get("best_val_loss", np.nan))
            for v in seeds.values()
        ])
        tag_mean_skills[tag] = dict(zip(seeds.keys(), skills))
        print(f"{tag:<16} {len(seeds):>7} "
              f"{skills.mean():>+10.4f} +/- {skills.std(ddof=1) if len(skills) > 1 else 0.0:<10.4f} "
              f"{val_mse.mean():>10.5f} +/- {val_mse.std(ddof=1) if len(val_mse) > 1 else 0.0:<10.5f}")

    # Paired comparison: every pair of configs that share seeds (e.g. geo0.0 vs
    # geo0.05, or patch_r6_hist1 vs patch_r6_hist2), matched seed-by-seed.
    tags_sorted = sorted(tag_mean_skills)
    for i, tag_a in enumerate(tags_sorted):
        for tag_b in tags_sorted[i + 1:]:
            a, b = tag_mean_skills[tag_a], tag_mean_skills[tag_b]
            common_seeds = sorted(set(a) & set(b))
            if len(common_seeds) < 2:
                continue
            diffs = np.array([a[s] - b[s] for s in common_seeds])  # tag_a - tag_b
            n = len(diffs)
            mean_d, std_d = diffs.mean(), diffs.std(ddof=1)
            se = std_d / np.sqrt(n)
            t = mean_d / se if se > 0 else float("nan")
            print(f"\nPaired comparison ({tag_a} - {tag_b} mean skill), n={n} matched seeds: "
                  f"{common_seeds}")
            print(f"  mean diff = {mean_d:+.4f}, std = {std_d:.4f}, SE = {se:.4f}, t = {t:.2f}")
            print(f"  Individual per-seed diffs: {[f'{d:+.4f}' for d in diffs]}")
            print("  (Rule of thumb: |t| > ~2-3 with n>=5 suggests the difference is unlikely to be")
            print("   pure seed noise; |t| < 1 means the two configs are not distinguishable at this n.)")

    # Rollout skill mean +/- std by horizon, for whichever tags have rollout data.
    print("\n=== rollout skill (mean over vars), mean +/- std across seeds, by horizon ===")
    for tag in sorted(runs):
        seeds = runs[tag]
        any_rollout = next((v for v in seeds.values() if "rollout" in v), None)
        if any_rollout is None:
            continue
        horizons = any_rollout["rollout"]["horizons"]
        mat = np.stack([v["rollout"]["skill"] for v in seeds.values() if "rollout" in v])  # [n_seeds, n_horizons]
        means = mat.mean(axis=0)
        stds = mat.std(axis=0, ddof=1) if mat.shape[0] > 1 else np.zeros_like(means)
        row = "  ".join(f"{h}d: {m:+.4f}+/-{s:.4f}" for h, m, s in zip(horizons, means, stds))
        print(f"{tag:<16} {row}")


if __name__ == "__main__":
    root = sys.argv[1] if len(sys.argv) > 1 else "results/seed_sweep"
    summarize(root)
