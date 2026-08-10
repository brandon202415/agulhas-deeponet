#!/usr/bin/env python3
"""Scan a subsample of r=3 primary-resolution test days (seed 2026, standard
4-pixel filter -- Table 2/3's exact configuration) for a day where the CNN
control misses a true eddy that AMSE catches, to use as a qualitative
illustration (Sec. 3.4's mechanism: recall failure concentrates near the
detection threshold). Reuses identify()/match_with_detail() from
eddy_tracking_analysis.py unmodified, same pattern as eddy_uv_confound_check.py.

Writes candidate days (ranked by how illustrative the miss/catch is) to a
JSON so the figure-drawing step can pick one without re-running detection.

    /opt/anaconda3/envs/eddytrack/bin/python3 eddy_tracking/find_qualitative_example.py \
        --amse results/amse_seed_sweep_r3_lr1e-3/amse_r3_w0.01_seed2026/predictions.npz \
        --control results/cnn_seed_sweep_r3/cnn_r3_lr1e-3_seed2026/predictions.npz \
        --stride 20 --out eddy_tracking/qualitative_candidates.json
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import eddy_tracking_analysis as eta

import numpy as np


def obs_to_arrays(obs):
    return {"lon": obs["lon"], "lat": obs["lat"], "amplitude": obs["amplitude"], "radius_e": obs["radius_e"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--amse", required=True)
    ap.add_argument("--control", required=True)
    ap.add_argument("--stride", type=int, default=20)
    ap.add_argument("--pixel-min", type=int, default=4)
    ap.add_argument("--pixel-max", type=int, default=2000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    eta.PIXEL_LIMIT = (args.pixel_min, args.pixel_max)

    da = np.load(args.amse)
    dc = np.load(args.control)
    lon = da["lon"].astype("float64")
    lat = da["lat"].astype("float64")
    eta.NLAT, eta.NLON = len(lat), len(lon)
    eta.NSENS = eta.NLAT * eta.NLON

    y_true, y_pred_a = da["y_true"], da["y_pred"]
    y_pred_c = dc["y_pred"]
    n_days = y_true.shape[0]

    zos_all_true = np.stack([eta.block(y_true, t, "zos") for t in range(n_days)])
    ocean_mask = zos_all_true.std(axis=0) > 1e-4

    candidates = []
    days = list(range(0, n_days, args.stride))
    print(f"Scanning {len(days)} days (stride {args.stride}) for a CNN-misses/AMSE-catches example...")
    for t in days:
        zos_t, uo_t, vo_t = eta.block(y_true, t, "zos"), eta.block(y_true, t, "uo"), eta.block(y_true, t, "vo")
        zos_c, uo_c, vo_c = eta.block(y_pred_c, t, "zos"), eta.block(y_pred_c, t, "uo"), eta.block(y_pred_c, t, "vo")
        zos_a, uo_a, vo_a = eta.block(y_pred_a, t, "zos"), eta.block(y_pred_a, t, "uo"), eta.block(y_pred_a, t, "vo")

        anti_t, cyc_t = eta.identify(zos_t, uo_t, vo_t, lon, lat, ocean_mask, t)
        anti_c, cyc_c = eta.identify(zos_c, uo_c, vo_c, lon, lat, ocean_mask, t)
        anti_a, cyc_a = eta.identify(zos_a, uo_a, vo_a, lon, lat, ocean_mask, t)

        for polarity, true_obs, ctrl_obs, amse_obs in [("A", anti_t, anti_c, anti_a), ("C", cyc_t, cyc_c, cyc_a)]:
            if len(true_obs) == 0:
                continue
            _, _, detail_c = eta.match_with_detail(obs_to_arrays(true_obs), obs_to_arrays(ctrl_obs), polarity)
            _, _, detail_a = eta.match_with_detail(obs_to_arrays(true_obs), obs_to_arrays(amse_obs), polarity)
            for i, (dc_, da_) in enumerate(zip(detail_c, detail_a)):
                if (not dc_["matched"]) and da_["matched"]:
                    candidates.append({
                        "day": t, "polarity": polarity, "true_eddy_index": i,
                        "true_lon": dc_["lon"], "true_lat": dc_["lat"],
                        "true_amplitude": dc_["amplitude"],
                        "amse_distance_km": da_["distance_km"],
                    })
        if len(candidates) >= 8:
            break

    print(f"Found {len(candidates)} candidate CNN-miss/AMSE-catch cases")
    with open(args.out, "w") as f:
        json.dump(candidates, f, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
