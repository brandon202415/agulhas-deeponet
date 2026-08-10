"""Load AVISO META4.0 DT Mesoscale Eddy Trajectory Atlas observations as
true_obs-compatible structured arrays (Issue 12: circularity validation --
this study's "true eddies" have so far always come from running py-eddy-tracker
on GLORYS's own SSH field; this swaps in an independent, altimetry-derived
eddy source for the *ground truth* side only, so recall/position-error can be
recomputed against something that isn't circular with the reanalysis itself).

No retraining and no changes to eddy_tracking_analysis.py's match()/
match_with_detail() -- both already operate generically on any object exposing
obs["lon"], obs["lat"], obs["amplitude"], obs["radius_e"], and len(obs), which
is exactly what py_eddy_tracker's own EddiesObservations exposes for the
GLORYS-derived truth today.

Field mapping (confirmed from this study's actual downloaded files, not
assumed): AVISO longitude/latitude/amplitude/effective_radius map directly
onto lon/lat/amplitude/radius_e; amplitude is meters in both AVISO and this
study's py-eddy-tracker output, so no unit conversion needed for either.

Date handling: AVISO's `time` units are "days since 1950-01-01" (confirmed
from the file's own metadata, not assumed -- see RESEARCH_LOG.md). This
study's predictions.npz files store `test_indices`, which are *input*-day
indices (day t) into the full reanalysis stream (epoch 1993-01-01); the
*target*/true day for a 1-day-lead prediction is t+1 (confirmed against
validate_satellite_fullscale.py's identical convention, and cross-checked:
test_indices[0]+1 / test_indices[-1]+1 land exactly on this study's known
standard test window, 2017-03-23 to 2021-06-30).
"""
import datetime

import netCDF4 as nc
import numpy as np

AVISO_EPOCH = datetime.date(1950, 1, 1)
REANALYSIS_EPOCH = datetime.date(1993, 1, 1)  # day-index 0 of the full reanalysis stream

TRUE_OBS_DTYPE = np.dtype([
    ("lon", "f8"), ("lat", "f8"), ("amplitude", "f8"), ("radius_e", "f8"),
])


def target_dates_for_predictions(predictions_npz_path):
    """Returns {row_index_in_predictions_npz: datetime.date} for every row,
    using this study's own test_indices + the +1 lead-time convention."""
    d = np.load(predictions_npz_path)
    test_indices = d["test_indices"]
    return {
        int(row): REANALYSIS_EPOCH + datetime.timedelta(days=int(idx) + 1)
        for row, idx in enumerate(test_indices)
    }


def _load_polarity_by_day(path, lon_bounds, lat_bounds, wanted_dates):
    """Single pass over one AVISO file's scalar fields only (never touches the
    30-point contour/profile arrays). wanted_dates: set of datetime.date.
    Returns {datetime.date: structured array (TRUE_OBS_DTYPE)}."""
    ds = nc.Dataset(path)
    lon = ds.variables["longitude"][:].astype(np.float64)
    lat = ds.variables["latitude"][:].astype(np.float64)
    t_days = ds.variables["time"][:].astype(np.float64)  # scale_factor already applied by netCDF4
    amp = ds.variables["amplitude"][:].astype(np.float64)
    rad = ds.variables["effective_radius"][:].astype(np.float64)
    ds.close()

    domain_mask = (
        (lon >= lon_bounds[0]) & (lon <= lon_bounds[1])
        & (lat >= lat_bounds[0]) & (lat <= lat_bounds[1])
    )
    date_min = min(wanted_dates)
    date_max = max(wanted_dates)
    day_off = np.floor(t_days).astype(np.int64)  # 1 obs/track/day; floor guards fractional noise
    off_min = (date_min - AVISO_EPOCH).days
    off_max = (date_max - AVISO_EPOCH).days
    date_mask = (day_off >= off_min) & (day_off <= off_max)

    keep = domain_mask & date_mask
    lon, lat, amp, rad, day_off = lon[keep], lat[keep], amp[keep], rad[keep], day_off[keep]

    wanted_offsets = {(dd - AVISO_EPOCH).days: dd for dd in wanted_dates}
    by_day = {}
    order = np.argsort(day_off, kind="stable")
    lon, lat, amp, rad, day_off = lon[order], lat[order], amp[order], rad[order], day_off[order]
    uniq_offsets, starts = np.unique(day_off, return_index=True)
    starts = list(starts) + [len(day_off)]
    for i, off in enumerate(uniq_offsets):
        off = int(off)
        if off not in wanted_offsets:
            continue
        s, e = starts[i], starts[i + 1]
        n = e - s
        arr = np.empty(n, dtype=TRUE_OBS_DTYPE)
        arr["lon"] = lon[s:e]
        arr["lat"] = lat[s:e]
        arr["amplitude"] = amp[s:e]
        arr["radius_e"] = rad[s:e]
        by_day[wanted_offsets[off]] = arr
    return by_day


def load_aviso_truth(anti_path, cyclo_path, lon_bounds, lat_bounds, wanted_dates):
    """Returns (anti_by_date, cyclo_by_date), each {datetime.date: structured
    array}. Dates in wanted_dates with zero matching AVISO obs are simply
    absent from the returned dicts (empty, not zero-filled) -- callers should
    treat a missing date as n_true=0 for that polarity, matching identify()'s
    own behavior when a day has no eddies."""
    anti_by_date = _load_polarity_by_day(anti_path, lon_bounds, lat_bounds, wanted_dates)
    cyclo_by_date = _load_polarity_by_day(cyclo_path, lon_bounds, lat_bounds, wanted_dates)
    return anti_by_date, cyclo_by_date


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--anti", required=True)
    ap.add_argument("--cyclo", required=True)
    ap.add_argument("--predictions", required=True,
                     help="any r=3 predictions.npz, just to read test_indices/lon/lat bounds")
    args = ap.parse_args()

    d = np.load(args.predictions)
    lon_bounds = (float(d["lon"].min()), float(d["lon"].max()))
    lat_bounds = (float(d["lat"].min()), float(d["lat"].max()))
    date_map = target_dates_for_predictions(args.predictions)
    wanted = set(date_map.values())
    print(f"domain: lon {lon_bounds}, lat {lat_bounds}")
    print(f"wanted dates: {min(wanted)} to {max(wanted)} ({len(wanted)} unique days)")

    anti_by_date, cyclo_by_date = load_aviso_truth(args.anti, args.cyclo, lon_bounds, lat_bounds, wanted)
    n_days_anti = len(anti_by_date)
    n_days_cyclo = len(cyclo_by_date)
    n_obs_anti = sum(len(v) for v in anti_by_date.values())
    n_obs_cyclo = sum(len(v) for v in cyclo_by_date.values())
    print(f"anticyclonic: {n_days_anti}/{len(wanted)} days have >=1 obs, {n_obs_anti} obs total")
    print(f"cyclonic:     {n_days_cyclo}/{len(wanted)} days have >=1 obs, {n_obs_cyclo} obs total")
    if anti_by_date:
        sample_date = next(iter(anti_by_date))
        sample = anti_by_date[sample_date]
        print(f"sample day {sample_date}: n={len(sample)}, "
              f"lon range {sample['lon'].min():.2f}-{sample['lon'].max():.2f}, "
              f"amplitude range {sample['amplitude'].min():.4f}-{sample['amplitude'].max():.4f} m")
