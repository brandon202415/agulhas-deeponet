#!/usr/bin/env python3
"""Direct comparison of the CNN (U-Net) baseline against the whole-domain
DeepONet's own headline result, both at 5 seeds, same split, same metrics.
Addresses the review item: "no non-persistence learned baseline anywhere in
the paper" -- this is the number that actually answers "does DeepONet's
branch-trunk formulation do anything distinctive?"

Reads two separate result roots (DeepONet results live under seed_sweep.slurm's
output, CNN results under cnn_seed_sweep.slurm's), matches by seed, and reports:
  - single-step mean skill, mean +/- std, both models
  - per-variable single-step skill, both models
  - rollout skill by horizon (mean over vars), both models
  - paired, seed-matched t-test (DeepONet - CNN) at single-step and every
    rollout horizon, using the exact t = mean_diff/(std(diff)/sqrt(n))
    convention this study uses everywhere else (Sec. 2.7)

Usage:
    python compare_cnn_vs_deeponet.py \\
        --deeponet-dir results/seed_sweep --deeponet-tag geo0.0 \\
        --cnn-dir results/cnn_seed_sweep --cnn-tag-glob "cnn_r6_lr*"
"""
import argparse
import fnmatch
import glob
import json
import os
import re

import numpy as np


def load_runs(root, tag_glob):
    """tag -> seed -> {"metrics": ..., "rollout": ...}, restricted to tags
    matching tag_glob (fnmatch-style via glob semantics on the directory name).
    """
    runs = {}
    for mj in glob.glob(os.path.join(root, "*", "metrics.json")):
        run_dir = os.path.dirname(mj)
        run_name = os.path.basename(run_dir)
        m = re.match(r"^(.*)_seed(-?\d+)$", run_name)
        if not m:
            continue
        tag, seed = m.group(1), int(m.group(2))
        if not fnmatch.fnmatch(tag, tag_glob):
            continue
        metrics = json.load(open(mj))
        entry = {"metrics": metrics}
        rz = os.path.join(run_dir, "rollout.npz")
        if os.path.exists(rz):
            d = np.load(rz)
            entry["rollout"] = {
                "horizons": list(d["horizons"]),
                "skill": np.nanmean(d["skill"], axis=1),  # mean over vars per horizon
                "variables": [str(v) for v in d["variables"]],
                "skill_per_var": d["skill"],  # [n_horizons, n_vars]
            }
        runs.setdefault(tag, {})[seed] = entry
    return runs


def mean_skill(metrics):
    ks = [metrics[k] for k in metrics if k.startswith("skill_") and not k.startswith("skill_persist")]
    return float(np.mean(ks)) if ks else float("nan")


def paired_t(a_vals, b_vals):
    diffs = np.array(a_vals) - np.array(b_vals)
    n = len(diffs)
    mean_d, std_d = diffs.mean(), diffs.std(ddof=1) if n > 1 else 0.0
    se = std_d / np.sqrt(n) if n > 1 else float("nan")
    t = mean_d / se if se > 0 else float("nan")
    return mean_d, std_d, se, t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deeponet-dir", default="results/seed_sweep")
    ap.add_argument("--deeponet-tag", default="geo0.0")
    ap.add_argument("--cnn-dir", default="results/cnn_seed_sweep")
    ap.add_argument("--cnn-tag-glob", default="cnn_r6_lr*")
    args = ap.parse_args()

    dn_runs = load_runs(args.deeponet_dir, args.deeponet_tag)
    cnn_runs_all = load_runs(args.cnn_dir, args.cnn_tag_glob)

    if args.deeponet_tag not in dn_runs:
        print(f"No DeepONet runs found: {args.deeponet_dir}/{args.deeponet_tag}_seed*/metrics.json")
        return
    if not cnn_runs_all:
        print(f"No CNN runs found: {args.cnn_dir}/{args.cnn_tag_glob}_seed*/metrics.json")
        return
    if len(cnn_runs_all) > 1:
        print(f"Multiple CNN tags matched {args.cnn_tag_glob}: {list(cnn_runs_all)}. "
              f"Pass a more specific --cnn-tag-glob. Using the first one found.")
    cnn_tag = list(cnn_runs_all)[0]

    dn = dn_runs[args.deeponet_tag]
    cnn = cnn_runs_all[cnn_tag]
    common_seeds = sorted(set(dn) & set(cnn))
    if len(common_seeds) < 2:
        print(f"Only {len(common_seeds)} matched seeds between DeepONet ({sorted(dn)}) "
              f"and CNN ({sorted(cnn)}) -- need matched seeds for a paired comparison.")
        return

    print(f"DeepONet: {args.deeponet_dir}/{args.deeponet_tag}_seed*  (seeds {sorted(dn)})")
    print(f"CNN:      {args.cnn_dir}/{cnn_tag}_seed*  (seeds {sorted(cnn)})")
    print(f"Matched seeds: {common_seeds}\n")

    # ── Single-step mean skill ──────────────────────────────────────────────
    dn_skills = [mean_skill(dn[s]["metrics"]) for s in common_seeds]
    cnn_skills = [mean_skill(cnn[s]["metrics"]) for s in common_seeds]
    print("=== Single-step mean skill (6-variable mean, matched seeds) ===")
    print(f"  DeepONet: {np.mean(dn_skills):+.4f} +/- {np.std(dn_skills, ddof=1):.4f}   "
          f"(per-seed: {[round(x,4) for x in dn_skills]})")
    print(f"  CNN:      {np.mean(cnn_skills):+.4f} +/- {np.std(cnn_skills, ddof=1):.4f}   "
          f"(per-seed: {[round(x,4) for x in cnn_skills]})")
    mean_d, std_d, se, t = paired_t(dn_skills, cnn_skills)
    print(f"  Paired (DeepONet - CNN): mean diff = {mean_d:+.4f}, SE = {se:.4f}, t = {t:.2f}")
    print(f"  (|t| >~ 2-3 with n={len(common_seeds)} suggests the difference is unlikely to be pure "
          f"seed noise; |t| < 1 means not distinguishable at this n -- same convention as Sec. 2.7)")

    # ── Per-variable single-step skill ──────────────────────────────────────
    print("\n=== Per-variable single-step skill, mean +/- std across matched seeds ===")
    variables = ["zos", "uo", "vo", "thetao", "so", "mlotst"]
    print(f"{'var':>8} {'DeepONet':>20} {'CNN':>20} {'paired t':>10}")
    for v in variables:
        dn_v = [dn[s]["metrics"].get(f"skill_{v}", float("nan")) for s in common_seeds]
        cnn_v = [cnn[s]["metrics"].get(f"skill_{v}", float("nan")) for s in common_seeds]
        _, _, _, tv = paired_t(dn_v, cnn_v)
        print(f"{v:>8} {np.mean(dn_v):>+10.4f}+/-{np.std(dn_v, ddof=1):<7.4f} "
              f"{np.mean(cnn_v):>+10.4f}+/-{np.std(cnn_v, ddof=1):<7.4f} {tv:>10.2f}")

    # ── Rollout skill by horizon ─────────────────────────────────────────────
    dn_has_rollout = all("rollout" in dn[s] for s in common_seeds)
    cnn_has_rollout = all("rollout" in cnn[s] for s in common_seeds)
    if dn_has_rollout and cnn_has_rollout:
        horizons = dn[common_seeds[0]]["rollout"]["horizons"]
        print(f"\n=== Rollout skill (mean over vars), mean +/- std across matched seeds, by horizon ===")
        print(f"{'horizon':>8} {'DeepONet':>20} {'CNN':>20} {'paired t':>10}")
        for hi, h in enumerate(horizons):
            dn_h = [dn[s]["rollout"]["skill"][hi] for s in common_seeds]
            cnn_h = [cnn[s]["rollout"]["skill"][hi] for s in common_seeds]
            _, _, _, th = paired_t(dn_h, cnn_h)
            print(f"{h:>7}d {np.mean(dn_h):>+10.4f}+/-{np.std(dn_h, ddof=1):<7.4f} "
                  f"{np.mean(cnn_h):>+10.4f}+/-{np.std(cnn_h, ddof=1):<7.4f} {th:>10.2f}")
    else:
        print("\n(Rollout comparison skipped -- rollout.npz missing for some matched seeds.)")

    print("\nInterpretation: positive paired t (DeepONet - CNN) means DeepONet beats the CNN "
          "at that comparison; negative means the CNN beats DeepONet. |t| >~ 2-3 is the "
          "threshold this study uses elsewhere to call a difference real rather than seed noise.")


if __name__ == "__main__":
    main()
