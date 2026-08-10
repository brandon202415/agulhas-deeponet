#!/usr/bin/env python3
"""uo/vo degradation confound check (decision-letter-style review, Issue 5 /
manuscript Limitations "Eighth"): py-eddy-tracker's eddy_identification() uses
uo/vo alongside SSH (for the eddy's speed-profile/contour-selection step, not
just SSH contour search -- confirmed by reading eddy_tracking_analysis.py's
identify()). AMSE measurably degrades uo/vo grid-point skill relative to the
CNN control (Sec. 3.6, t=-12.6 to -18.3 at r=6). This paper's own eddy-tracking
benefit for AMSE (Table 3) has never been tested for whether it is partly an
artifact of degraded velocity fields shifting which contours get accepted as
eddies, rather than a genuine improvement driven by AMSE's actual target (SSH).

Design: for each of 5 seeds at r=3 (this paper's primary resolution, weight
0.01), reuse this project's own identify()/match_with_detail() unmodified
(imported directly from eddy_tracking_analysis.py, not reimplemented) but
swap which field's uo/vo goes with which field's zos:
  - "amse_own_uv"    : AMSE's zos + AMSE's own uo/vo           (= Table 3's AMSE condition, recomputed fresh here for a clean seed-matched comparison, not reused from the existing result files)
  - "amse_hybrid_uv" : AMSE's zos + CNN CONTROL's uo/vo        (the confound-isolating condition: same SSH improvement, undegraded velocity)
  - "control_own_uv" : CNN control's zos + CNN control's uo/vo (= Table 2's control condition, recomputed fresh)
"true" is identified once per day from the shared true field (AMSE's and the
control's y_true are asserted identical) and reused for all three matches.

If AMSE's recall/position-error advantage over the control is materially
unchanged between "amse_own_uv" and "amse_hybrid_uv", the advantage is not an
artifact of degraded velocity fields. If it shrinks substantially toward the
control's own numbers under "amse_hybrid_uv", the confound is real and AMSE's
credited benefit is partly a velocity-degradation artifact.

Usage (per seed; run for all 5 and aggregate):
    python3 eddy_tracking/eddy_uv_confound_check.py \
        --amse results/amse_seed_sweep_r3_lr1e-3/amse_r3_w0.01_seed2026/predictions.npz \
        --control results/cnn_seed_sweep_r3/cnn_r3_lr1e-3_seed2026/predictions.npz \
        --pixel-min 4 --pixel-max 2000 --stride 1 \
        --out eddy_tracking/uv_confound_seed2026.json
"""
import argparse
import json
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
import eddy_tracking_analysis as eta  # reuse identify(), match_with_detail(), block(), great_circle_km unmodified

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--amse", required=True)
    ap.add_argument("--control", required=True)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--pixel-min", type=int, default=4)
    ap.add_argument("--pixel-max", type=int, default=2000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    eta.PIXEL_LIMIT = (args.pixel_min, args.pixel_max)

    da = np.load(args.amse)
    dc = np.load(args.control)
    lon = da["lon"].astype("float64")
    lat = da["lat"].astype("float64")
    assert np.array_equal(lon, dc["lon"]) and np.array_equal(lat, dc["lat"]), "grid mismatch between --amse and --control"
    eta.NLAT, eta.NLON = len(lat), len(lon)
    eta.NSENS = eta.NLAT * eta.NLON

    y_true_a, y_pred_a = da["y_true"], da["y_pred"]
    y_true_c, y_pred_c = dc["y_true"], dc["y_pred"]
    n_days = y_true_a.shape[0]
    assert n_days == y_true_c.shape[0]

    zos_check = np.stack([eta.block(y_true_a, t, "zos") for t in range(0, n_days, 200)])
    zos_check_c = np.stack([eta.block(y_true_c, t, "zos") for t in range(0, n_days, 200)])
    max_diff = np.nanmax(np.abs(zos_check - zos_check_c))
    print(f"Sanity check: max |y_true(AMSE run) - y_true(control run)| over sampled days = {max_diff:.6g} "
          f"(should be ~0, both runs share the same held-out true field)")

    zos_all_true = np.stack([eta.block(y_true_a, t, "zos") for t in range(n_days)])
    ocean_mask = zos_all_true.std(axis=0) > 1e-4
    print(f"Grid: {eta.NLAT} x {eta.NLON} = {eta.NSENS} sensors, n_days={n_days}, "
          f"ocean fraction={ocean_mask.mean():.3f}, pixel_limit={eta.PIXEL_LIMIT}")

    day_indices = list(range(0, n_days, args.stride))
    print(f"Running eddy identification on {len(day_indices)} days x 4 fields "
          f"(true, amse_own_uv, amse_hybrid_uv, control_own_uv)...")

    per_day = []
    for k, t in enumerate(day_indices):
        zos_t = eta.block(y_true_a, t, "zos")
        uo_t = eta.block(y_true_a, t, "uo")
        vo_t = eta.block(y_true_a, t, "vo")
        anti_t, cyc_t = eta.identify(zos_t, uo_t, vo_t, lon, lat, ocean_mask, t)

        zos_amse = eta.block(y_pred_a, t, "zos")
        uo_amse = eta.block(y_pred_a, t, "uo")
        vo_amse = eta.block(y_pred_a, t, "vo")
        zos_ctrl = eta.block(y_pred_c, t, "zos")
        uo_ctrl = eta.block(y_pred_c, t, "uo")
        vo_ctrl = eta.block(y_pred_c, t, "vo")

        anti_own, cyc_own = eta.identify(zos_amse, uo_amse, vo_amse, lon, lat, ocean_mask, t)
        anti_hyb, cyc_hyb = eta.identify(zos_amse, uo_ctrl, vo_ctrl, lon, lat, ocean_mask, t)
        anti_ctl, cyc_ctl = eta.identify(zos_ctrl, uo_ctrl, vo_ctrl, lon, lat, ocean_mask, t)

        rec = {"day": int(t), "n_true_anti": len(anti_t), "n_true_cyc": len(cyc_t)}
        for label, other_a, other_c in (
            ("amse_own_uv", anti_own, cyc_own),
            ("amse_hybrid_uv", anti_hyb, cyc_hyb),
            ("control_own_uv", anti_ctl, cyc_ctl),
        ):
            d_a, n_a, _ = eta.match_with_detail(anti_t, other_a, "A")
            d_c, n_c, _ = eta.match_with_detail(cyc_t, other_c, "C")
            rec[f"{label}_dists"] = d_a + d_c
            rec[f"{label}_n_matched"] = len(d_a) + len(d_c)
        per_day.append(rec)
        if (k + 1) % 100 == 0:
            print(f"  ... {k+1}/{len(day_indices)} days done")

    n_true_total = sum(r["n_true_anti"] + r["n_true_cyc"] for r in per_day)
    summary = {"n_days": len(per_day), "stride": args.stride, "n_true_eddies_total": n_true_total,
               "pixel_limit": list(eta.PIXEL_LIMIT)}
    for label in ("amse_own_uv", "amse_hybrid_uv", "control_own_uv"):
        all_dists = [dd for r in per_day for dd in r[f"{label}_dists"]]
        n_matched = sum(r[f"{label}_n_matched"] for r in per_day)
        summary[label] = {
            "recall": n_matched / n_true_total if n_true_total else float("nan"),
            "mean_position_error_km": float(np.mean(all_dists)) if all_dists else float("nan"),
            "n_matched": n_matched,
        }
    print(json.dumps(summary, indent=2))
    with open(args.out, "w") as f:
        json.dump({"summary": summary, "per_day": per_day}, f, indent=2)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
