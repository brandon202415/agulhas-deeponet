#!/usr/bin/env python3
"""Post-hoc climatology baseline for the weekly-rolling rollout comparison
(Sec. 3.8 of the manuscript). No trained model needed: climatology (the
training-period per-sensor/per-variable mean field) and persistence (the
frozen initial state) are both model-independent forecasts. Computed here by
importing load_states()/build_dataset() unmodified from
train_agulhas_deeponet_prototype.py, so the split, ocean mask, and
climatology field are guaranteed identical to what every actual training run
used -- this is not a reimplementation, it is the same pipeline evaluated
without a model.

Motivation (review comment): Table 13 reports model skill relative to
persistence only. A model that has converged to predicting climatology (i.e.
learned nothing about multi-week dynamics) would show the same qualitative
"skill grows with lead time" pattern purely because persistence's own error
grows relative to a fixed climatology field -- this script establishes
whether persistence itself is already losing to climatology at the horizons
in question, independent of any trained model, which is the missing control.

Usage:
    python evaluate_weekly_climatology_baseline.py \
        --cache data/cache_r6_weekly_rolling_fullscale.npz \
        --subsample-r 6 --step-days 7 --rollout-horizons 7 14 21 28
"""
import argparse

import numpy as np

from train_agulhas_deeponet_prototype import (
    VARIABLES,
    _var_major_flat,
    build_dataset,
    load_states,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--subsample-r", type=int, default=6)
    ap.add_argument("--step-days", type=int, default=7)
    ap.add_argument("--val-fraction", type=float, default=0.15)
    ap.add_argument("--test-fraction", type=float, default=0.15)
    ap.add_argument("--rollout-horizons", type=int, nargs="+", default=[7, 14, 21, 28])
    args = ap.parse_args()

    lon, lat, states = load_states(None, subsample_r=args.subsample_r, cache=args.cache)
    ds = build_dataset(
        states, lon, lat,
        test_fraction=args.test_fraction, val_fraction=args.val_fraction,
        step_days=args.step_days,
    )

    n_sensors = ds["n_sensors"]
    n_vars = ds["n_vars"]
    ocean = ds["ocean_mask"]
    clim = ds["climatology"]  # [n_sensors, n_vars], training-period mean only
    k = max(1, int(args.step_days))
    T = states.shape[0]

    horizons = sorted(h for h in args.rollout_horizons if h > 0 and h % k == 0)
    H = max(horizons)
    starts = np.array([t for t in ds["test_idx"] if t + H <= T - 1], dtype=np.int64)
    print(f"n_starts={starts.size}, horizons={horizons}, n_vars={n_vars}")

    persist_vm = _var_major_flat(states[starts].astype(np.float64))  # frozen t0

    print(f"\n=== Mean over variables (dimensionless-appropriate: skill only; "
          f"raw RMSE below is per-variable since units differ) ===")
    print(f"{'horizon':>8} {'skill(persist vs clim)':>24}")
    per_h_rows = []
    for h in horizons:
        true_vm = _var_major_flat(states[starts + h].astype(np.float64))
        rmse_p_list, rmse_c_list = [], []
        for vi in range(n_vars):
            cs, ce = vi * n_sensors, (vi + 1) * n_sensors
            t = true_vm[:, cs:ce][:, ocean]
            pr = persist_vm[:, cs:ce][:, ocean]
            c = np.broadcast_to(clim[ocean, vi], t.shape)
            rmse_p_list.append(float(np.sqrt(np.mean((pr - t) ** 2))))
            rmse_c_list.append(float(np.sqrt(np.mean((c - t) ** 2))))
        per_h_rows.append((h, rmse_p_list, rmse_c_list))
        skills = [1.0 - (rp / rc) ** 2 if rc > 1e-12 else float("nan")
                  for rp, rc in zip(rmse_p_list, rmse_c_list)]
        print(f"{h:>8} {np.mean(skills):>24.4f}")

    print(f"\n=== Per-variable RMSE: persistence vs. climatology (physical units) ===")
    for h, rmse_p_list, rmse_c_list in per_h_rows:
        print(f"\n-- horizon {h}d --")
        print(f"{'var':>8} {'rmse_persist':>14} {'rmse_clim':>12} "
              f"{'skill(persist vs clim)':>24}")
        for vi, vname in enumerate(VARIABLES):
            rp, rc = rmse_p_list[vi], rmse_c_list[vi]
            sk = 1.0 - (rp / rc) ** 2 if rc > 1e-12 else float("nan")
            print(f"{vname:>8} {rp:>14.4f} {rc:>12.4f} {sk:>24.4f}")


if __name__ == "__main__":
    main()
