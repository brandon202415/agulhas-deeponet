#!/usr/bin/env python3
"""Eddy-center position error / detection recall at 1-day lead (reviewer item 3).

RMSE/ACC/skill (the manuscript's existing metrics) are grid-point statistics; they
never ask whether a specific ring was detected, or how far off its center was.
This script runs py-eddy-tracker's closed-contour SSH identification (the field
standard) on the model's 1-day prediction, the persistence baseline, and the truth,
for a sample of test days, and reports:
  - detection recall: fraction of "true" eddies that have a same-polarity eddy
    center within a matching radius in the model's / persistence's field
  - position error: great-circle distance between matched centers (km)
separately for model and persistence, so the comparison directly answers "does
the model located rings better than doing nothing."

CAVEAT (stated in the manuscript): the study grid is r=6, ~0.5 deg (~50-55 km at
these latitudes) spacing -- coarse relative to a single Agulhas ring (~100-300 km
diameter), so this measures large/coherent rings only and undercounts anything
close to the grid scale; py-eddy-tracker's own pixel_limit had to be loosened from
its oceanographic-altimetry default (4 px min) to (1, 2000) to detect anything at
all at this resolution, itself informative about the resolution question in Sec 5.

Run with the dedicated `eddytrack` conda env (see eddy_tracking/README.md for why):
    /opt/anaconda3/envs/eddytrack/bin/python3 eddy_tracking/eddy_tracking_analysis.py \
        --predictions /Users/brandonzhang/Downloads/best/predictions.npz \
        --stride 10 --out eddy_tracking/eddy_tracking_results.json
"""
import argparse
import datetime
import json
import logging
import os
import sys
import tempfile
import warnings

import netCDF4
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "aviso"))

warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("pet").setLevel(logging.ERROR)
logging.getLogger().setLevel(logging.ERROR)

NLAT, NLON, NSENS = None, None, None  # set at runtime in main() from the predictions.npz lon/lat
                                       # (was hardcoded to the r=6 whole-domain grid; patch
                                       # predictions can be r=6 or r=3, with different grid shapes)
VARIABLES = ["zos", "uo", "vo", "thetao", "so", "mlotst"]
EARTH_RADIUS_KM = 6371.0
BESSEL_WAVELENGTH_KM = 400  # high-pass filter cutoff before contour search
STEP = 0.005                # SSH contour spacing (m), py-eddy-tracker default
SHAPE_ERROR = 70             # % max contour-shape error allowed (loosened from default 55)
PIXEL_LIMIT = (1, 2000)     # (min, max) pixels inside outer contour; loosened for r=6 grid
                             # overridable via --pixel-min/--pixel-max for sensitivity checks
MATCH_RADIUS_KM = 250        # max distance to call two eddies "the same" ring


def block(arr, day, vname):
    vi = VARIABLES.index(vname)
    col = arr[day, vi * NSENS:(vi + 1) * NSENS]
    return col.reshape(NLAT, NLON)


def write_frame_nc(path, zos, uo, vo, lon, lat):
    with netCDF4.Dataset(path, "w") as h:
        h.createDimension("lat", len(lat))
        h.createDimension("lon", len(lon))
        vlon = h.createVariable("lon", "f8", ("lon",)); vlon[:] = lon; vlon.units = "degrees_east"
        vlat = h.createVariable("lat", "f8", ("lat",)); vlat[:] = lat; vlat.units = "degrees_north"
        vadt = h.createVariable("adt", "f4", ("lat", "lon"), fill_value=np.float32(np.nan))
        vadt[:] = zos; vadt.units = "m"
        vu = h.createVariable("u", "f4", ("lat", "lon"), fill_value=np.float32(np.nan))
        vu[:] = uo; vu.units = "m/s"
        vv = h.createVariable("v", "f4", ("lat", "lon"), fill_value=np.float32(np.nan))
        vv[:] = vo; vv.units = "m/s"


def identify(zos, uo, vo, lon, lat, ocean_mask, day_idx):
    from py_eddy_tracker.dataset.grid import RegularGridDataset

    zos_m = np.where(ocean_mask, zos, np.nan).astype("float32")
    uo_m = np.where(ocean_mask, uo, np.nan).astype("float32")
    vo_m = np.where(ocean_mask, vo, np.nan).astype("float32")
    tmp = tempfile.mktemp(suffix=".nc")
    try:
        write_frame_nc(tmp, zos_m, uo_m, vo_m, lon, lat)
        h = RegularGridDataset(tmp, "lon", "lat", nan_masking=True)
        h.bessel_high_filter("adt", BESSEL_WAVELENGTH_KM)
        date = datetime.datetime(2000, 1, 1) + datetime.timedelta(days=int(day_idx))
        anti, cyclo = h.eddy_identification(
            "adt", "u", "v", date,
            step=STEP, shape_error=SHAPE_ERROR, pixel_limit=PIXEL_LIMIT,
        )
        return anti, cyclo
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def great_circle_km(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))
    dlon, dlat = lon2 - lon1, lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def match(true_obs, other_obs, max_km=MATCH_RADIUS_KM):
    """Greedy nearest-neighbor matching of eddy centers. Returns list of distances
    (km) for matched pairs, and n_true (for recall)."""
    n_true = len(true_obs)
    if n_true == 0 or len(other_obs) == 0:
        return [], n_true
    used = set()
    dists = []
    for i in range(n_true):
        best_j, best_d = None, None
        for j in range(len(other_obs)):
            if j in used:
                continue
            d = great_circle_km(true_obs["lon"][i], true_obs["lat"][i],
                                 other_obs["lon"][j], other_obs["lat"][j])
            if best_d is None or d < best_d:
                best_d, best_j = d, j
        if best_d is not None and best_d <= max_km:
            used.add(best_j)
            dists.append(best_d)
    return dists, n_true


def match_with_detail(true_obs, other_obs, polarity, max_km=MATCH_RADIUS_KM):
    """Identical greedy matching logic to match() (same pooled dists/n_true
    return values, for exact backward compatibility with eddy_stat_test.py),
    plus a per-true-eddy detail record (position, amplitude, radius, match
    outcome, matched prediction's amplitude) for subgroup analyses (domain
    review: retroflection-vs-downstream, polarity, recall-vs-amplitude,
    amplitude error) that the pooled dists list alone cannot support."""
    n_true = len(true_obs)
    detail = []
    if n_true == 0:
        return [], n_true, detail
    used = set()
    dists = []
    matches = [None] * n_true
    if len(other_obs) > 0:
        for i in range(n_true):
            best_j, best_d = None, None
            for j in range(len(other_obs)):
                if j in used:
                    continue
                d = great_circle_km(true_obs["lon"][i], true_obs["lat"][i],
                                     other_obs["lon"][j], other_obs["lat"][j])
                if best_d is None or d < best_d:
                    best_d, best_j = d, j
            if best_d is not None and best_d <= max_km:
                used.add(best_j)
                dists.append(best_d)
                matches[i] = (best_j, best_d)
    for i in range(n_true):
        m = matches[i]
        detail.append({
            "polarity": polarity,
            "lon": float(true_obs["lon"][i]),
            "lat": float(true_obs["lat"][i]),
            "amplitude": float(true_obs["amplitude"][i]),
            "radius_e": float(true_obs["radius_e"][i]),
            "matched": m is not None,
            "distance_km": float(m[1]) if m is not None else None,
            "pred_amplitude": float(other_obs["amplitude"][m[0]]) if m is not None else None,
        })
    return dists, n_true, detail


def main():
    global NLAT, NLON, NSENS, PIXEL_LIMIT
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--stride", type=int, default=10, help="use every Nth test day")
    ap.add_argument("--out", default="eddy_tracking_results.json")
    ap.add_argument("--pixel-min", type=int, default=PIXEL_LIMIT[0],
                     help="min pixels inside outer contour (sensitivity check; default matches paper)")
    ap.add_argument("--pixel-max", type=int, default=PIXEL_LIMIT[1],
                     help="max pixels inside outer contour (sensitivity check; default matches paper)")
    ap.add_argument("--aviso-anti", default=None,
                     help="AVISO META4.0 DT anticyclonic .nc path (Issue 12: use independent "
                          "altimetry-derived truth instead of GLORYS-identify()'d truth; "
                          "requires --aviso-cyclo too). Model/persistence still scored via identify().")
    ap.add_argument("--aviso-cyclo", default=None)
    args = ap.parse_args()

    use_aviso_truth = args.aviso_anti is not None
    if use_aviso_truth != (args.aviso_cyclo is not None):
        raise ValueError("--aviso-anti and --aviso-cyclo must be given together")

    PIXEL_LIMIT = (args.pixel_min, args.pixel_max)
    d = np.load(args.predictions)
    lon = d["lon"].astype("float64")
    lat = d["lat"].astype("float64")
    NLAT, NLON = len(lat), len(lon)
    NSENS = NLAT * NLON
    y_true, y_pred, y_persist = d["y_true"], d["y_pred"], d["y_persist"]
    n_days = y_true.shape[0]
    print(f"Grid: {NLAT} x {NLON} = {NSENS} sensors (from predictions.npz lon/lat)")
    assert y_true.shape[1] == len(VARIABLES) * NSENS, (
        f"predictions array width {y_true.shape[1]} != n_vars*n_sensors "
        f"({len(VARIABLES)}*{NSENS}={len(VARIABLES)*NSENS}) -- lon/lat don't match the data"
    )

    zos_all_true = np.stack([block(y_true, t, "zos") for t in range(n_days)])
    ocean_mask = zos_all_true.std(axis=0) > 1e-4
    print(f"n_days={n_days}, ocean fraction={ocean_mask.mean():.3f}")

    day_indices = list(range(0, n_days, args.stride))
    print(f"Running eddy identification on {len(day_indices)} days "
          f"(stride={args.stride}) x 3 fields (true/pred/persist)...")

    aviso_anti_by_date, aviso_cyclo_by_date, row_to_date = None, None, None
    if use_aviso_truth:
        from aviso_true_obs import load_aviso_truth, target_dates_for_predictions, TRUE_OBS_DTYPE
        row_to_date = target_dates_for_predictions(args.predictions)
        wanted_dates = {row_to_date[t] for t in day_indices}
        print(f"Loading AVISO truth (Issue 12: independent altimetry-derived ground truth) "
              f"for {len(wanted_dates)} dates, domain lon={lon.min():.1f}-{lon.max():.1f} "
              f"lat={lat.min():.1f}-{lat.max():.1f}...")
        aviso_anti_by_date, aviso_cyclo_by_date = load_aviso_truth(
            args.aviso_anti, args.aviso_cyclo,
            (float(lon.min()), float(lon.max())), (float(lat.min()), float(lat.max())),
            wanted_dates,
        )
        empty_obs = np.empty(0, dtype=TRUE_OBS_DTYPE)
        n_with_obs = sum(1 for dd in wanted_dates if dd in aviso_anti_by_date or dd in aviso_cyclo_by_date)
        print(f"  {n_with_obs}/{len(wanted_dates)} dates have >=1 AVISO obs (either polarity) in-domain")

    per_day = []
    for k, t in enumerate(day_indices):
        zos_t, uo_t, vo_t = block(y_true, t, "zos"), block(y_true, t, "uo"), block(y_true, t, "vo")
        zos_p, uo_p, vo_p = block(y_pred, t, "zos"), block(y_pred, t, "uo"), block(y_pred, t, "vo")
        zos_s, uo_s, vo_s = block(y_persist, t, "zos"), block(y_persist, t, "uo"), block(y_persist, t, "vo")

        if use_aviso_truth:
            dd = row_to_date[t]
            anti_t = aviso_anti_by_date.get(dd, empty_obs)
            cyc_t = aviso_cyclo_by_date.get(dd, empty_obs)
        else:
            anti_t, cyc_t = identify(zos_t, uo_t, vo_t, lon, lat, ocean_mask, t)
        anti_p, cyc_p = identify(zos_p, uo_p, vo_p, lon, lat, ocean_mask, t)
        anti_s, cyc_s = identify(zos_s, uo_s, vo_s, lon, lat, ocean_mask, t)

        rec = {"day": int(t), "n_true_anti": len(anti_t), "n_true_cyc": len(cyc_t)}
        for label, other_a, other_c in (("model", anti_p, cyc_p), ("persist", anti_s, cyc_s)):
            d_a, n_a, det_a = match_with_detail(anti_t, other_a, "A")
            d_c, n_c, det_c = match_with_detail(cyc_t, other_c, "C")
            rec[f"{label}_dists"] = d_a + d_c
            rec[f"{label}_n_matched"] = len(d_a) + len(d_c)
            rec[f"{label}_n_other"] = len(other_a) + len(other_c)
            rec[f"{label}_detail"] = det_a + det_c
        per_day.append(rec)
        if (k + 1) % 20 == 0:
            print(f"  ... {k+1}/{len(day_indices)} days done")

    n_true_total = sum(r["n_true_anti"] + r["n_true_cyc"] for r in per_day)
    summary = {
        "n_days": len(per_day), "stride": args.stride, "n_true_eddies_total": n_true_total,
        "truth_source": "aviso" if use_aviso_truth else "glorys_identify",
    }
    for label in ("model", "persist"):
        all_dists = [dd for r in per_day for dd in r[f"{label}_dists"]]
        n_matched = sum(r[f"{label}_n_matched"] for r in per_day)
        n_other_total = sum(r[f"{label}_n_other"] for r in per_day)
        summary[label] = {
            "recall": n_matched / n_true_total if n_true_total else float("nan"),
            "mean_position_error_km": float(np.mean(all_dists)) if all_dists else float("nan"),
            "median_position_error_km": float(np.median(all_dists)) if all_dists else float("nan"),
            "n_matched": n_matched,
            "n_detected_total": n_other_total,
            "n_position_errors": len(all_dists),
        }

    print(json.dumps(summary, indent=2))
    with open(args.out, "w") as f:
        json.dump({"summary": summary, "per_day": per_day}, f, indent=2)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
