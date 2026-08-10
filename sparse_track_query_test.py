#!/usr/bin/env python3
"""Direct query at irregular, off-grid points -- the capability a grid-locked
architecture (CNN, U-Net, Swin-Transformer/WenHai-style) genuinely does not
have without an extra interpolation step.

Sec. 3.7 tested zero-shot query at a finer REGULAR grid (r=3) -- a CNN with an
upsampling head could, in principle, also hit a finer regular grid. What a CNN
cannot do is predict directly at arbitrary, irregularly-scattered coordinates,
such as a satellite altimetry ground track (real nadir altimeters sample densely
along-track but sparsely across-track, not on any grid). This script builds a
small set of synthetic straight-line "tracks" crossing the domain at points that
land off both the r=6 and r=3 grids, and compares three ways of getting a
forecast at those exact points:

  1. DeepONet DIRECT query: the trunk queried exactly at the track coordinates
     (using the persist_at_query fix so the residual is resolution/point-correct).
  2. GRID-THEN-INTERPOLATE: run the model on its native r=6 grid as usual (the
     only thing a grid-locked architecture could do), then bilinear-interpolate
     that prediction to the track points. This is the fair baseline representing
     what any non-operator-learning architecture would have to do.
  3. Persistence at the track points (interpolated), as the "do nothing" floor.

"True" values and "persistence" at the off-grid track points are obtained by
bilinear-interpolating the r=3 (0.25 deg) reanalysis grid -- an honest proxy,
not independent finer observations (GLORYS itself has no truth finer than its
native 1/12 deg grid, and r=3 is 3x coarser than that). This is a local-
prototype-scale, single-checkpoint proof of concept.
"""
import sys
import numpy as np
import torch
from scipy.interpolate import RegularGridInterpolator

sys.path.insert(0, ".")
from train_agulhas_deeponet_prototype import (
    load_states, build_dataset, MultivarDeepONet, to_tensor, VARIABLES,
)

CKPT = "results/whole_r6_local/model.pt"
NC_PATH = "data/agulhas_prototype.nc"
CACHE_R3 = "data/cache_r3_local.npz"


def make_tracks(lon_min, lon_max, lat_min, lat_max, n_tracks=6, angle_deg=25,
                 along_spacing=0.12, track_spacing=None, seed=0):
    """Synthetic satellite ground tracks: straight lines at a fixed inclination,
    evenly spaced across the domain, points sampled at along_spacing intervals.
    Chosen spacing/angle deliberately does not align with either the r=6 (0.5 deg)
    or r=3 (0.25 deg) grid, so points land off-grid."""
    rng = np.random.default_rng(seed)
    theta = np.radians(angle_deg)
    lon_span, lat_span = lon_max - lon_min, lat_max - lat_min
    if track_spacing is None:
        track_spacing = lon_span / (n_tracks + 1)
    pts = []
    for i in range(n_tracks):
        # start point along the bottom edge, offset per track, small random jitter
        x0 = lon_min + (i + 1) * track_spacing + rng.uniform(-0.05, 0.05)
        y0 = lat_min - 2.0  # start below domain so the track fully crosses it
        # walk along the track direction until leaving the domain
        length = (lat_span + 4.0) / np.cos(theta)
        n_steps = int(length / along_spacing)
        for s in range(n_steps):
            d = s * along_spacing
            x = x0 + d * np.sin(theta)
            y = y0 + d * np.cos(theta)
            if lon_min < x < lon_max and lat_min < y < lat_max:
                pts.append((x, y))
    pts = np.array(pts)
    return pts[:, 0], pts[:, 1]  # lon, lat


def main():
    device = "cpu"

    # --- r=6 branch/trunk (same as Sec. 3.7) ---
    lon6, lat6, states6 = load_states(NC_PATH, subsample_r=6, cache="data/cache_r6_local.npz")
    ds6 = build_dataset(states6, lon6, lat6, test_fraction=0.15, val_fraction=0.15, step_days=1)
    n_sensors6, n_vars = ds6["n_sensors"], ds6["n_vars"]
    out_mean, out_std = ds6["out_mean"], ds6["out_std"]

    model = MultivarDeepONet(
        d_branch=n_sensors6 * n_vars, n_sensors=n_sensors6, n_vars=n_vars,
        branch_width=64, branch_depth=2, trunk_width=64, trunk_depth=2, latent_dim=32,
    ).to(device)
    model.load_state_dict(torch.load(CKPT, map_location=device, weights_only=False))
    model.eval()

    def T(a):
        return torch.tensor(a, dtype=torch.float32, device=device)

    branch_test = T(ds6["branch_test"])
    trunk6 = T(ds6["trunk"])

    LAT6, LON6 = np.meshgrid(lat6, lon6, indexing="ij")
    trunk6_raw = np.stack([LON6.ravel(), LAT6.ravel()], axis=-1).astype(np.float64)
    t_min = trunk6_raw.min(axis=0, keepdims=True)
    t_span = trunk6_raw.max(axis=0, keepdims=True) - t_min

    # --- r=3 grid: source of "true"/"persistence" via interpolation ---
    lon3, lat3, states3 = load_states(NC_PATH, subsample_r=3, cache=CACHE_R3)
    k = 1
    next3 = states3[k:].astype(np.float64)   # [N, nlat3, nlon3, n_vars] raw, "true"
    curr3 = states3[:-k].astype(np.float64)  # [N, nlat3, nlon3, n_vars] raw, persistence
    test_idx = ds6["test_idx"]

    ocean6_grid = ds6["ocean_mask"].reshape(len(lat6), len(lon6))
    # r=3 ocean mask, same convention as build_dataset's own (temporal std on the
    # training slice) -- land is exactly 0 always, so this reliably separates them.
    train_idx6 = ds6["train_idx"]
    ocean3_grid = (next3[train_idx6, :, :, 0].std(axis=0) > 1e-4)  # zos-based, [nlat3, nlon3]

    # --- synthetic track points, off both grids ---
    track_lon, track_lat = make_tracks(
        lon_min=float(lon6.min()), lon_max=float(lon6.max()),
        lat_min=float(lat6.min()), lat_max=float(lat6.max()),
        n_tracks=6, angle_deg=25, along_spacing=0.12,
    )
    print(f"Synthetic tracks: {len(track_lon)} points across 6 lines, off-grid "
          f"(r=6 spacing 0.5deg, r=3 spacing 0.25deg, along-track spacing 0.12deg)")

    def all_corners_ocean(qlon, qlat, lon_grid, lat_grid, ocean_grid):
        """True where all 4 grid cells bilinear interpolation would blend for this
        query point are ocean -- excludes points whose interpolated value would be
        contaminated by the land=0 sentinel (found: land-adjacent points were
        interpolating to near-zero salinity/etc., a real bug in the raw approach,
        not a genuine model failure)."""
        lon_i = np.clip(np.searchsorted(lon_grid, qlon) - 1, 0, len(lon_grid) - 2)
        lat_i = np.clip(np.searchsorted(lat_grid, qlat) - 1, 0, len(lat_grid) - 2)
        c00 = ocean_grid[lat_i, lon_i]
        c01 = ocean_grid[lat_i, lon_i + 1]
        c10 = ocean_grid[lat_i + 1, lon_i]
        c11 = ocean_grid[lat_i + 1, lon_i + 1]
        return c00 & c01 & c10 & c11

    safe3 = all_corners_ocean(track_lon, track_lat, lon3, lat3, ocean3_grid)
    safe6 = all_corners_ocean(track_lon, track_lat, lon6, lat6, ocean6_grid)
    keep = safe3 & safe6
    n_dropped = (~keep).sum()
    track_lon, track_lat = track_lon[keep], track_lat[keep]
    n_track = len(track_lon)
    print(f"  dropped {n_dropped} points within one grid cell of land (interpolation "
          f"would blend in the land=0 sentinel); {n_track} clean open-ocean points remain")
    off_r6 = np.min(np.abs(track_lon[:, None] - lon6[None, :]), axis=1).mean()
    off_r3 = np.min(np.abs(track_lon[:, None] - lon3[None, :]), axis=1).mean()
    print(f"  mean min-distance to nearest r=6 lon node: {off_r6:.4f} deg  "
          f"(nonzero confirms points are off-grid)")
    print(f"  mean min-distance to nearest r=3 lon node: {off_r3:.4f} deg")

    trunk_track_raw = np.stack([track_lon, track_lat], axis=-1).astype(np.float64)
    trunk_track_norm = (2.0 * (trunk_track_raw - t_min) / t_span - 1.0).astype(np.float32)
    trunk_track = T(trunk_track_norm)

    # --- interpolators built once per (day, variable) would be slow; instead
    #     interpolate all test days at once using a vectorized RegularGridInterpolator
    #     over (day, lat, lon) with nearest-neighbor in day (exact match) ---
    def interp_r3_to_track(field_days):
        """field_days: [N_test, nlat3, nlon3, n_vars] raw physical values at the
        r=3 grid, for the test-set days. Returns [N_test, n_track, n_vars]."""
        out = np.zeros((field_days.shape[0], n_track, n_vars), dtype=np.float64)
        for vi in range(n_vars):
            for i in range(field_days.shape[0]):
                interp = RegularGridInterpolator(
                    (lat3, lon3), field_days[i, :, :, vi],
                    method="linear", bounds_error=False, fill_value=None,
                )
                out[i, :, vi] = interp(np.stack([track_lat, track_lon], axis=-1))
        return out

    true_track = interp_r3_to_track(next3[test_idx])      # [N_test, n_track, n_vars]
    persist_track = interp_r3_to_track(curr3[test_idx])   # [N_test, n_track, n_vars]

    # --- 1. DeepONet DIRECT query at the track points ---
    persist_track_norm = (persist_track - out_mean[None, None, :]) / out_std[None, None, :]
    persist_track_t = T(persist_track_norm.transpose(0, 2, 1))  # [N_test, n_vars, n_track]
    with torch.no_grad():
        pred_direct = model(branch_test, trunk_track, persist_at_query=persist_track_t)
    pred_direct = pred_direct.cpu().numpy() * out_std[None, None, :] + out_mean[None, None, :]
    # -> [N_test, n_track, n_vars]

    # --- 2. GRID-THEN-INTERPOLATE baseline: native r=6 prediction, then interpolate ---
    with torch.no_grad():
        pred_grid_norm = model(branch_test, trunk6).cpu().numpy()  # [N_test, n_sensors6, n_vars]
    pred_grid_phys = pred_grid_norm * out_std[None, None, :] + out_mean[None, None, :]
    pred_grid_reshaped = pred_grid_phys.reshape(pred_grid_phys.shape[0], len(lat6), len(lon6), n_vars)

    pred_grid_interp = np.zeros((pred_grid_reshaped.shape[0], n_track, n_vars), dtype=np.float64)
    for vi in range(n_vars):
        for i in range(pred_grid_reshaped.shape[0]):
            interp = RegularGridInterpolator(
                (lat6, lon6), pred_grid_reshaped[i, :, :, vi],
                method="linear", bounds_error=False, fill_value=None,
            )
            pred_grid_interp[i, :, vi] = interp(np.stack([track_lat, track_lon], axis=-1))

    # --- Score all three against interpolated "truth" at the track points ---
    print("\n=== Skill at off-grid synthetic satellite-track points ===")
    print(f"{'Variable':10s} {'DeepONet direct':>16s} {'Grid+interpolate':>18s} {'Persistence(floor)':>20s}")
    direct_skills, grid_skills = [], []
    for vi, vname in enumerate(VARIABLES):
        yt = true_track[:, :, vi]
        yp_direct = pred_direct[:, :, vi]
        yp_grid = pred_grid_interp[:, :, vi]
        ys = persist_track[:, :, vi]
        std = yt.std()
        if std < 1e-6:
            continue
        rmse_direct = np.sqrt(np.mean((yp_direct - yt) ** 2))
        rmse_grid = np.sqrt(np.mean((yp_grid - yt) ** 2))
        rmse_persist = np.sqrt(np.mean((ys - yt) ** 2))
        sk_direct = 1.0 - (rmse_direct / rmse_persist) ** 2 if rmse_persist > 0 else float("nan")
        sk_grid = 1.0 - (rmse_grid / rmse_persist) ** 2 if rmse_persist > 0 else float("nan")
        direct_skills.append(sk_direct)
        grid_skills.append(sk_grid)
        print(f"{vname:10s} {sk_direct:+16.4f} {sk_grid:+18.4f} {'(0.0000 by definition)':>20s}")
    print(f"{'Mean':10s} {np.mean(direct_skills):+16.4f} {np.mean(grid_skills):+18.4f}")
    print("\n(DeepONet direct query beating grid-then-interpolate would mean continuous")
    print(" querying adds real value beyond what any grid-based method could get by")
    print(" simply interpolating its own native-grid prediction.)")


if __name__ == "__main__":
    main()
