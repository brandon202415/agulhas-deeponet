#!/usr/bin/env python3
"""Paired day-level bootstrap CI and permutation test for eddy-tracking comparisons
(reviewer item: "recall and position-error comparisons are called 'statistically
indistinguishable' on the strength of point estimates alone -- I'd want a bootstrap
CI or permutation test before accepting 'no difference' as a finding").

Loads two eddy_tracking_analysis.py output JSONs (each has a "per_day" list of
records with day-aligned match counts/distances for a "model" field). Both files
must have been run on the identical day list (verified below) so that day-level
resampling is a valid paired bootstrap -- it resamples days with replacement,
preserving whatever correlation exists between the two series on a given day
(they share the same true-eddy field that day).

Two comparisons are supported:
  - vs-persist: compares each file's own "model" field to its own "persist" field
    (both already saved per_day per file) -- e.g. does patch beat persistence?
  - a-vs-b: compares file A's "model" field to file B's "model" field directly
    (e.g. whole-domain model vs patch model, or r=6 patch vs r=3 patch)

For each, reports:
  - observed difference in recall (ratio of pooled matched/true) and mean
    position error (pooled over matched distances)
  - 95% bootstrap CI on that difference (10,000 resamples of days, with
    replacement)
  - two-sided permutation p-value (10,000 random day-level label swaps between
    the two series being compared)

Usage:
    python3 eddy_tracking/eddy_stat_test.py --a FILE_A.json --b FILE_B.json --mode vs-persist
    python3 eddy_tracking/eddy_stat_test.py --a FILE_A.json --b FILE_B.json --mode a-vs-b
"""
import argparse
import json
import re

import numpy as np

RNG_SEED = 0
N_BOOT = 10000


def load(path):
    return json.load(open(path))["per_day"]


def series(per_day, label):
    """Per-day (n_matched, n_true, dists) for a given label ('model' or 'persist')."""
    n_true = np.array([r["n_true_anti"] + r["n_true_cyc"] for r in per_day])
    n_matched = np.array([r[f"{label}_n_matched"] for r in per_day])
    dists = [r[f"{label}_dists"] for r in per_day]
    return n_true, n_matched, dists


def recall_and_poserr(idx, n_true, n_matched, dists):
    """Given a (possibly resampled-with-replacement) array of day indices, compute
    pooled recall and pooled mean position error."""
    nt = n_true[idx].sum()
    nm = n_matched[idx].sum()
    recall = nm / nt if nt else float("nan")
    all_d = [d for i in idx for d in dists[i]]
    pos_err = float(np.mean(all_d)) if all_d else float("nan")
    return recall, pos_err


def bootstrap_ci(n_true_a, n_matched_a, dists_a, n_true_b, n_matched_b, dists_b, rng):
    n_days = len(n_true_a)
    recall_diffs, pos_diffs = [], []
    for _ in range(N_BOOT):
        idx = rng.integers(0, n_days, size=n_days)
        ra, pa = recall_and_poserr(idx, n_true_a, n_matched_a, dists_a)
        rb, pb = recall_and_poserr(idx, n_true_b, n_matched_b, dists_b)
        recall_diffs.append(ra - rb)
        pos_diffs.append(pa - pb)
    recall_diffs = np.array([d for d in recall_diffs if not np.isnan(d)])
    pos_diffs = np.array([d for d in pos_diffs if not np.isnan(d)])
    return recall_diffs, pos_diffs


def permutation_test(n_true_a, n_matched_a, dists_a, n_true_b, n_matched_b, dists_b, rng):
    """Day-level label-swap permutation test: for each day, independently swap
    which series is 'a' and which is 'b' with 50% probability, recompute the
    pooled difference, and see how often the null exceeds the observed |diff|."""
    n_days = len(n_true_a)
    obs_idx = np.arange(n_days)
    ra_obs, pa_obs = recall_and_poserr(obs_idx, n_true_a, n_matched_a, dists_a)
    rb_obs, pb_obs = recall_and_poserr(obs_idx, n_true_b, n_matched_b, dists_b)
    obs_recall_diff = ra_obs - rb_obs
    obs_pos_diff = pa_obs - pb_obs

    null_recall, null_pos = [], []
    for _ in range(N_BOOT):
        swap = rng.random(n_days) < 0.5
        nt_a = np.where(swap, n_true_b, n_true_a)
        nm_a = np.where(swap, n_matched_b, n_matched_a)
        d_a = [dists_b[i] if swap[i] else dists_a[i] for i in range(n_days)]
        nt_b = np.where(swap, n_true_a, n_true_b)
        nm_b = np.where(swap, n_matched_a, n_matched_b)
        d_b = [dists_a[i] if swap[i] else dists_b[i] for i in range(n_days)]
        ra, pa = recall_and_poserr(obs_idx, nt_a, nm_a, d_a)
        rb, pb = recall_and_poserr(obs_idx, nt_b, nm_b, d_b)
        null_recall.append(ra - rb)
        null_pos.append(pa - pb)
    null_recall = np.array(null_recall)
    null_pos = np.array([d for d in null_pos if not np.isnan(d)])

    p_recall = float(np.mean(np.abs(null_recall) >= abs(obs_recall_diff)))
    p_pos = float(np.mean(np.abs(null_pos) >= abs(obs_pos_diff))) if len(null_pos) else float("nan")
    return obs_recall_diff, obs_pos_diff, p_recall, p_pos


def report(name_a, name_b, n_true_a, n_matched_a, dists_a, n_true_b, n_matched_b, dists_b):
    rng = np.random.default_rng(RNG_SEED)
    recall_diffs, pos_diffs = bootstrap_ci(
        n_true_a, n_matched_a, dists_a, n_true_b, n_matched_b, dists_b, rng)
    obs_recall_diff, obs_pos_diff, p_recall, p_pos = permutation_test(
        n_true_a, n_matched_a, dists_a, n_true_b, n_matched_b, dists_b,
        np.random.default_rng(RNG_SEED + 1))

    ra_obs, pa_obs = recall_and_poserr(np.arange(len(n_true_a)), n_true_a, n_matched_a, dists_a)
    rb_obs, pb_obs = recall_and_poserr(np.arange(len(n_true_b)), n_true_b, n_matched_b, dists_b)

    print(f"\n=== {name_a} vs {name_b} ===")
    print(f"  recall:    {ra_obs:.4f} vs {rb_obs:.4f}  (diff {obs_recall_diff:+.4f})")
    print(f"    95% bootstrap CI on diff: [{np.percentile(recall_diffs, 2.5):+.4f}, {np.percentile(recall_diffs, 97.5):+.4f}]")
    print(f"    permutation p-value:      {p_recall:.4f}")
    print(f"  pos. err.: {pa_obs:.3f} vs {pb_obs:.3f} km  (diff {obs_pos_diff:+.3f} km)")
    print(f"    95% bootstrap CI on diff: [{np.percentile(pos_diffs, 2.5):+.3f}, {np.percentile(pos_diffs, 97.5):+.3f}] km")
    print(f"    permutation p-value:      {p_pos:.4f}")
    return {
        "recall_a": ra_obs, "recall_b": rb_obs, "recall_diff": obs_recall_diff,
        "recall_ci95": [float(np.percentile(recall_diffs, 2.5)), float(np.percentile(recall_diffs, 97.5))],
        "recall_perm_p": p_recall,
        "pos_err_a": pa_obs, "pos_err_b": pb_obs, "pos_err_diff": obs_pos_diff,
        "pos_err_ci95": [float(np.percentile(pos_diffs, 2.5)), float(np.percentile(pos_diffs, 97.5))],
        "pos_err_perm_p": p_pos,
    }


def extract_seed(path):
    """Best-effort seed extraction from a result filename (e.g.
    'amse_w0.01_seed2026.json' -> 2026). Returns None if the filename
    doesn't follow this project's `..._seed<N>.json` convention -- callers
    should skip the seed-match check in that case rather than fail on
    legitimately differently-named files."""
    m = re.search(r"_seed(-?\d+)(?:\.json)?$", str(path).rsplit("/", 1)[-1].replace(".json", ""))
    return int(m.group(1)) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--label-a", default=None)
    ap.add_argument("--label-b", default=None)
    ap.add_argument("--mode", choices=["vs-persist", "a-vs-b"], required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--allow-seed-mismatch", action="store_true",
                     help="Suppress the seed-matched-control guard below. Only pass this if "
                          "you deliberately want an unmatched-seed comparison (rare) -- the "
                          "default behavior exists specifically because an earlier version of "
                          "this project's AMSE-vs-control comparison silently used a single, "
                          "arbitrarily-chosen control seed for every AMSE seed (RESEARCH_LOG.md "
                          "item 39, manuscript Sec. 4.2), which this flag exists to prevent from "
                          "happening silently again.")
    args = ap.parse_args()

    if args.mode == "a-vs-b" and not args.allow_seed_mismatch:
        seed_a, seed_b = extract_seed(args.a), extract_seed(args.b)
        if seed_a is not None and seed_b is not None and seed_a != seed_b:
            raise SystemExit(
                f"REFUSING TO RUN: --a and --b appear to be different seeds "
                f"(seed {seed_a} vs seed {seed_b}), inferred from their filenames. "
                f"This project's own convention (Sec. 2.6/4.2) is that every "
                f"'AMSE vs. control' comparison pairs each seed against a control "
                f"trained at the SAME seed -- an earlier version of this comparison "
                f"got this wrong (Sec. 4.2's honesty note) and this check exists so "
                f"that mistake can't happen silently again. If you deliberately want "
                f"a cross-seed comparison, pass --allow-seed-mismatch explicitly."
            )

    per_day_a = load(args.a)
    per_day_b = load(args.b)
    days_a = [r["day"] for r in per_day_a]
    days_b = [r["day"] for r in per_day_b]
    assert days_a == days_b, "day lists differ between the two files -- not a valid paired design"

    label_a = args.label_a or args.a
    label_b = args.label_b or args.b

    results = {}
    if args.mode == "vs-persist":
        nt_a, nm_a, d_a = series(per_day_a, "model")
        nt_ap, nm_ap, d_ap = series(per_day_a, "persist")
        results[f"{label_a}: model vs persist"] = report(
            f"{label_a} model", f"{label_a} persist", nt_a, nm_a, d_a, nt_ap, nm_ap, d_ap)

        nt_b, nm_b, d_b = series(per_day_b, "model")
        nt_bp, nm_bp, d_bp = series(per_day_b, "persist")
        results[f"{label_b}: model vs persist"] = report(
            f"{label_b} model", f"{label_b} persist", nt_b, nm_b, d_b, nt_bp, nm_bp, d_bp)
    else:
        nt_a, nm_a, d_a = series(per_day_a, "model")
        nt_b, nm_b, d_b = series(per_day_b, "model")
        results[f"{label_a} vs {label_b}"] = report(label_a, label_b, nt_a, nm_a, d_a, nt_b, nm_b, d_b)

    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
