#!/usr/bin/env python3
"""Qualitative eddy-comparison figure: true vs. CNN-predicted vs. AMSE-predicted
SSH for one representative test day (r=3 primary resolution, seed 2026),
zoomed to the region around one true eddy that the CNN control misses and
AMSE catches -- chosen from eddy_tracking/qualitative_candidates.json (found
by find_qualitative_example.py, which reuses identify()/match_with_detail()
from eddy_tracking_analysis.py unmodified). This is one illustrative day, not
a claim about typical-case performance -- Tables 2/3 report the pooled,
statistically tested numbers this figure only illustrates.

    /opt/anaconda3/envs/eddytrack/bin/python3 eddy_tracking/make_fig_eddy_qualitative.py \
        --amse results/amse_seed_sweep_r3_lr1e-3/amse_r3_w0.01_seed2026/predictions.npz \
        --control results/cnn_seed_sweep_r3/cnn_r3_lr1e-3_seed2026/predictions.npz \
        --day 180 --polarity A \
        --out ../manuscript_figures/fig_eddy_qualitative.png
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import eddy_tracking_analysis as eta

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--amse", required=True)
    ap.add_argument("--control", required=True)
    ap.add_argument("--day", type=int, required=True)
    ap.add_argument("--polarity", choices=["A", "C"], required=True)
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

    t = args.day
    zos_t, uo_t, vo_t = eta.block(y_true, t, "zos"), eta.block(y_true, t, "uo"), eta.block(y_true, t, "vo")
    zos_c, uo_c, vo_c = eta.block(y_pred_c, t, "zos"), eta.block(y_pred_c, t, "uo"), eta.block(y_pred_c, t, "vo")
    zos_a, uo_a, vo_a = eta.block(y_pred_a, t, "zos"), eta.block(y_pred_a, t, "uo"), eta.block(y_pred_a, t, "vo")

    anti_t, cyc_t = eta.identify(zos_t, uo_t, vo_t, lon, lat, ocean_mask, t)
    anti_c, cyc_c = eta.identify(zos_c, uo_c, vo_c, lon, lat, ocean_mask, t)
    anti_a, cyc_a = eta.identify(zos_a, uo_a, vo_a, lon, lat, ocean_mask, t)

    true_obs = anti_t if args.polarity == "A" else cyc_t
    ctrl_obs = anti_c if args.polarity == "A" else cyc_c
    amse_obs = anti_a if args.polarity == "A" else cyc_a

    def arr(o):
        return {"lon": o["lon"], "lat": o["lat"], "amplitude": o["amplitude"], "radius_e": o["radius_e"]}

    _, _, detail_c = eta.match_with_detail(arr(true_obs), arr(ctrl_obs), args.polarity)
    _, _, detail_a = eta.match_with_detail(arr(true_obs), arr(amse_obs), args.polarity)

    # pick the true eddy this script's caller flagged as a miss/catch case (largest
    # true amplitude among unmatched-by-CNN, matched-by-AMSE eddies that day, as a
    # tie-break since day/polarity alone may contain more than one such case)
    target_i = None
    for i, (dc_, da_) in enumerate(zip(detail_c, detail_a)):
        if (not dc_["matched"]) and da_["matched"]:
            if target_i is None or dc_["amplitude"] > detail_c[target_i]["amplitude"]:
                pass
            target_i = i
            break
    if target_i is None:
        raise SystemExit("No CNN-miss/AMSE-catch eddy found on this day/polarity")

    tgt = detail_c[target_i]
    print(f"Target true eddy: lon={tgt['lon']:.2f} lat={tgt['lat']:.2f} amplitude={tgt['amplitude']:.4f} m")
    print(f"  CNN control: matched={detail_c[target_i]['matched']}")
    print(f"  AMSE:        matched={detail_a[target_i]['matched']}, distance={detail_a[target_i]['distance_km']:.2f} km")

    # zoom window around the target eddy
    pad = 3.5
    lon_lo, lon_hi = tgt["lon"] - pad, tgt["lon"] + pad
    lat_lo, lat_hi = tgt["lat"] - pad, tgt["lat"] + pad

    # local zoom-window color/contour scaling (not full-domain) so the
    # closed-contour structure py-eddy-tracker actually searches on is visible
    lon_idx = np.where((lon >= lon_lo) & (lon <= lon_hi))[0]
    lat_idx = np.where((lat >= lat_lo) & (lat <= lat_hi))[0]
    local_vals = zos_t[np.ix_(lat_idx, lon_idx)]
    vmax = np.nanmax(np.abs(local_vals[ocean_mask[np.ix_(lat_idx, lon_idx)]]))
    # visual contour spacing (coarser than py-eddy-tracker's actual 0.005 m detection
    # step, Sec. 2.5) chosen only for human readability at this figure's scale
    vis_step = max(vmax / 12, 0.01)
    levels = np.arange(-vmax, vmax + 1e-6, vis_step)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6), sharex=True, sharey=True)
    fields = [("True SSH", zos_t, true_obs, None), ("CNN control (misses this eddy)", zos_c, ctrl_obs, "#d62728"),
              ("AMSE (catches it)", zos_a, amse_obs, "#1f77b4")]
    for ax, (title, zos, obs, edgecolor) in zip(axes, fields):
        im = ax.pcolormesh(lon, lat, np.where(ocean_mask, zos, np.nan), cmap="RdBu_r",
                            vmin=-vmax, vmax=vmax, shading="auto")
        ax.contour(lon, lat, np.where(ocean_mask, zos, np.nan), levels=levels,
                   colors="k", linewidths=0.3, alpha=0.6)
        if len(obs) > 0:
            ax.scatter(obs["lon"], obs["lat"], s=18, facecolors="none", edgecolors="k", linewidths=0.8)
        ax.scatter([tgt["lon"]], [tgt["lat"]], marker="*", s=260, facecolors="none",
                   edgecolors=edgecolor or "k", linewidths=2.2, zorder=5)
        ax.set_xlim(lon_lo, lon_hi)
        ax.set_ylim(lat_lo, lat_hi)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Longitude (°E)")
    axes[0].set_ylabel("Latitude (°N)")
    fig.colorbar(im, ax=axes, shrink=0.85, label="SSH (m)", pad=0.02)

    pol_name = "anticyclonic" if args.polarity == "A" else "cyclonic"
    fig.suptitle(
        f"Day {t}, one {pol_name} true eddy (★, amplitude {tgt['amplitude']:.3f} m): "
        f"CNN control misses it, AMSE catches it at {detail_a[target_i]['distance_km']:.1f} km. "
        f"One illustrative case, not the pooled result (Tables 2–3).",
        fontsize=9.5, y=1.03,
    )

    out = args.out
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
