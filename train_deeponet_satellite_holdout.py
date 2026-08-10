#!/usr/bin/env python3
"""Train a whole-domain r=6 DeepONet with a custom (non-chronological) train/test
split that holds out Jan-Jul 2020 as test -- the exact window covered by a real
downloaded CryoSat-2 along-track sea-level dataset -- then validates DeepONet's
direct continuous-coordinate query against REAL satellite observations at the
REAL satellite pass locations, compared against the "predict on the native grid,
then interpolate" baseline any grid-locked architecture would be forced to use.

This is a deliberate exception to the paper's main chronological-split
methodology (train is prior to test everywhere else in this study): the local
prototype file only starts 2020-01-01, so the one real satellite-overlap window
available sits at the very start of the record. Held out here so this specific
check can use real observations; clearly flagged, not the paper's main split.

Caveats (report honestly, do not paper over):
  - "Truth" is the satellite's own sla_filtered + mdt (an estimate of absolute
    dynamic topography), compared against GLORYS zos -- different products,
    potentially different reference conventions. We remove a constant sample-
    mean bias between the two before scoring (standard practice), which fixes a
    constant offset but not any spatially- or temporally-varying bias.
  - GLORYS is a DAILY MEAN field; satellite obs are INSTANTANEOUS point
    measurements. Some disagreement is expected from this alone
    ("representativeness error"), not a model deficiency.
  - Land-adjacent points are dropped (interpolation would blend in the land=0
    sentinel), same standard as everywhere else in this study.
"""
import argparse
import datetime
from pathlib import Path

import numpy as np
import torch
from scipy.interpolate import RegularGridInterpolator

from train_agulhas_deeponet_prototype import (
    load_states, MultivarDeepONet, VARIABLES, save_json, _var_major_flat,
)

torch.set_num_threads(4)

NC_PATH = "data/agulhas_prototype.nc"
CACHE_R6 = "data/cache_r6_local.npz"
CACHE_R3 = "data/cache_r3_local.npz"
SAT_FILE = "/Users/brandonzhang/Downloads/data/satellite_sla_agulhas_c2_2019_2021/Cryosat-2.nc"
LOCAL_EPOCH = datetime.datetime(2020, 1, 1)  # day-index 0 of agulhas_prototype.nc


def build_dataset_explicit_split(states, lon_sub, lat_sub, train_idx, val_idx, test_idx, step_days=1):
    """Same normalisation/branch/trunk logic as build_dataset(), but with
    explicit index arrays instead of a chronological fraction-based split."""
    T, nlat_s, nlon_s, n_vars = states.shape
    n_sensors = nlat_s * nlon_s
    k = max(1, int(step_days))

    flat = _var_major_flat(states.astype(np.float64))
    branch_inputs = flat[:-k]
    next_grid = states[k:].astype(np.float64)
    next_flat = _var_major_flat(next_grid)

    LAT, LON = np.meshgrid(lat_sub, lon_sub, indexing="ij")
    trunk_raw = np.stack([LON.ravel(), LAT.ravel()], axis=-1).astype(np.float64)
    t_min = trunk_raw.min(axis=0, keepdims=True)
    t_span = trunk_raw.max(axis=0, keepdims=True) - t_min
    t_span = np.where(t_span < 1e-12, 1.0, t_span)
    trunk_norm = (2.0 * (trunk_raw - t_min) / t_span - 1.0).astype(np.float32)

    out_mean = np.zeros(n_vars, dtype=np.float64)
    out_std = np.ones(n_vars, dtype=np.float64)
    next_flat_norm = next_flat.copy()
    for vi in range(n_vars):
        c0, c1 = vi * n_sensors, (vi + 1) * n_sensors
        block = next_flat[train_idx, c0:c1]
        ocean = block.std(axis=0) > 1e-4
        vals = block[:, ocean] if ocean.any() else block
        m, s = vals.mean(), vals.std()
        if s < 1e-12:
            s = 1.0
        out_mean[vi], out_std[vi] = m, s
        next_flat_norm[:, c0:c1] = (next_flat[:, c0:c1] - m) / s

    y_train_norm = next_flat_norm[train_idx].astype(np.float32)
    y_val_norm = next_flat_norm[val_idx].astype(np.float32)
    y_test_norm = next_flat_norm[test_idx].astype(np.float32)

    ocean_mask = next_flat[train_idx, :n_sensors].std(axis=0) > 1e-4

    b_mean = np.repeat(out_mean, n_sensors)[None, :]
    b_std = np.repeat(out_std, n_sensors)[None, :]
    land_vm = np.tile(~ocean_mask, n_vars)
    branch_train = ((branch_inputs[train_idx] - b_mean) / b_std)
    branch_val = ((branch_inputs[val_idx] - b_mean) / b_std)
    branch_test = ((branch_inputs[test_idx] - b_mean) / b_std)
    branch_train[:, land_vm] = 0.0
    branch_val[:, land_vm] = 0.0
    branch_test[:, land_vm] = 0.0

    incr = (y_train_norm.astype(np.float64) - branch_train.astype(np.float64))
    incr_var = incr.var(axis=0)
    om_vm = np.tile(ocean_mask, n_vars)
    floor = 0.1 * np.median(incr_var[om_vm]) if om_vm.any() else 1.0
    loss_weight = 1.0 / np.maximum(incr_var, max(floor, 1e-12))
    loss_weight[~om_vm] = 0.0
    if om_vm.any():
        loss_weight *= om_vm.sum() / loss_weight[om_vm].sum()

    x_test_raw = _var_major_flat(states[test_idx].astype(np.float64))
    x_train_raw = _var_major_flat(states[train_idx].astype(np.float64))

    return dict(
        branch_train=branch_train.astype(np.float32), branch_val=branch_val.astype(np.float32),
        branch_test=branch_test.astype(np.float32), trunk=trunk_norm,
        y_train_norm=y_train_norm, y_val_norm=y_val_norm, y_test_norm=y_test_norm,
        y_test_raw=next_flat[test_idx], x_test_raw=x_test_raw, x_train_raw=x_train_raw,
        ocean_mask=ocean_mask, loss_weight=loss_weight.astype(np.float32),
        out_mean=out_mean, out_std=out_std,
        train_idx=train_idx, val_idx=val_idx, test_idx=test_idx,
        nlat_s=nlat_s, nlon_s=nlon_s, n_sensors=n_sensors, n_vars=n_vars,
        t_min=t_min, t_span=t_span,
    )


def load_satellite_obs(test_target_days, lon_bounds, lat_bounds):
    """Load CryoSat-2 along-track points whose calendar date falls on one of the
    given target days (local-file day indices, day D's forecast target),
    returning per-day arrays of (lon, lat, absolute_ssh_proxy)."""
    import netCDF4 as nc
    d = nc.Dataset(SAT_FILE)
    lon = d.variables["longitude"][:].astype(np.float64)
    lat = d.variables["latitude"][:].astype(np.float64)
    t = d.variables["time"][:]
    sla = np.ma.filled(d.variables["sla_filtered"][:], np.nan).astype(np.float64)
    mdt = np.ma.filled(d.variables["mdt"][:], np.nan).astype(np.float64)
    ssh_proxy = sla + mdt

    valid = np.isfinite(ssh_proxy) & (lon >= lon_bounds[0]) & (lon <= lon_bounds[1]) \
        & (lat >= lat_bounds[0]) & (lat <= lat_bounds[1])
    lon, lat, t, ssh_proxy = lon[valid], lat[valid], t[valid], ssh_proxy[valid]

    dates = np.array([(datetime.datetime(1970, 1, 1) + datetime.timedelta(seconds=float(s))).date()
                       for s in t])
    day_idx = np.array([(datetime.datetime.combine(dd, datetime.time()) - LOCAL_EPOCH).days for dd in dates])

    by_day = {}
    for D in test_target_days:
        m = day_idx == D
        if m.sum() > 0:
            by_day[D] = (lon[m], lat[m], ssh_proxy[m])
    return by_day


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iterations", type=int, default=5000)
    ap.add_argument("--learning-rate", type=float, default=3e-4)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out-dir", type=Path, default=Path("results/satellite_holdout_r6_local"))
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = "cpu"

    lon6, lat6, states6 = load_states(NC_PATH, subsample_r=6, cache=CACHE_R6)
    T = states6.shape[0]
    N = T - 1

    # Custom split: test = pairs [0, 211] (target days 1..212, Jan2-Jul31 2020,
    # the satellite window); small val slice right after; train = the rest.
    test_idx = np.arange(0, 212, dtype=np.int64)
    val_idx = np.arange(212, 312, dtype=np.int64)
    train_idx = np.arange(312, N, dtype=np.int64)
    print(f"Custom split: train={len(train_idx)} days, val={len(val_idx)} days, "
          f"test={len(test_idx)} days (target days 1-212 = Jan2-Jul31 2020, satellite window)")

    ds = build_dataset_explicit_split(states6, lon6, lat6, train_idx, val_idx, test_idx, step_days=1)
    n_sensors6, n_vars = ds["n_sensors"], ds["n_vars"]
    out_mean, out_std = ds["out_mean"], ds["out_std"]

    model = MultivarDeepONet(
        d_branch=n_sensors6 * n_vars, n_sensors=n_sensors6, n_vars=n_vars,
        branch_width=64, branch_depth=2, trunk_width=64, trunk_depth=2, latent_dim=32,
    ).to(device)

    def T_(a):
        return torch.tensor(a, dtype=torch.float32, device=device)

    branch_train, branch_val = T_(ds["branch_train"]), T_(ds["branch_val"])
    trunk6 = T_(ds["trunk"])
    y_train, y_val = T_(ds["y_train_norm"]), T_(ds["y_val_norm"])
    ocean_vm = T_(np.tile(ds["ocean_mask"], n_vars).astype(np.float32)).bool()
    w = T_(ds["loss_weight"])

    opt = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    n_train = branch_train.shape[0]
    bs = n_train if args.batch_size is None else min(args.batch_size, n_train)
    best_val, best_state, bad = float("inf"), None, 0

    for step in range(1, args.iterations + 1):
        model.train()
        idx = torch.randint(0, n_train, (bs,))
        pred = model(branch_train[idx], trunk6)
        pred_vm = pred.permute(0, 2, 1).reshape(pred.shape[0], -1)
        e2 = (pred_vm[:, ocean_vm] - y_train[idx][:, ocean_vm]) ** 2
        loss = (e2 * w[ocean_vm]).mean()
        opt.zero_grad(); loss.backward(); opt.step()

        if step % 100 == 0 or step == args.iterations:
            model.eval()
            with torch.no_grad():
                pv = model(branch_val, trunk6)
                pv_vm = pv.permute(0, 2, 1).reshape(pv.shape[0], -1)
                vloss = ((pv_vm[:, ocean_vm] - y_val[:, ocean_vm]) ** 2).mean().item()
            print(f"step {step:6d}  train_loss {loss.item():.5f}  val_loss {vloss:.5f}")
            if vloss < best_val - 1e-6:
                best_val, best_state, bad = vloss, {k: v.clone() for k, v in model.state_dict().items()}, 0
            else:
                bad += 1
                if bad >= args.patience:
                    print(f"Early stop at step {step} (best val_loss={best_val:.5f})")
                    break

    model.load_state_dict(best_state)
    model.eval()
    torch.save(model.state_dict(), args.out_dir / "model.pt")

    # --- standard r=6 test skill on the held-out Jan-Jul 2020 window (sanity check) ---
    branch_test = T_(ds["branch_test"])
    with torch.no_grad():
        pred6 = model(branch_test, trunk6).numpy()
    pred6_phys = pred6 * out_std[None, None, :] + out_mean[None, None, :]
    ocean6 = ds["ocean_mask"]
    print("\n=== Standard r=6 test skill (Jan-Jul 2020 holdout) ===")
    skills = []
    for vi, vname in enumerate(VARIABLES):
        c0, c1 = vi * n_sensors6, (vi + 1) * n_sensors6
        yt = ds["y_test_raw"][:, c0:c1][:, ocean6]
        yp = pred6_phys[:, :, vi][:, ocean6]
        ys = ds["x_test_raw"][:, c0:c1][:, ocean6]
        rmse_m = np.sqrt(np.mean((yp - yt) ** 2))
        rmse_p = np.sqrt(np.mean((ys - yt) ** 2))
        sk = 1.0 - (rmse_m / rmse_p) ** 2 if rmse_p > 0 else float("nan")
        skills.append(sk)
        print(f"  {vname:8s} skill={sk:+.4f}")
    print(f"  mean skill = {np.mean(skills):+.4f}  (cf. chronological-split whole_r6_local: +0.043ish)")

    # --- REAL satellite validation ---
    print("\n=== Loading real CryoSat-2 satellite observations ===")
    target_days = list(range(1, 213))  # day D's actual state, verified against obs on day D
    by_day = load_satellite_obs(target_days, (float(lon6.min()), float(lon6.max())),
                                 (float(lat6.min()), float(lat6.max())))
    n_days_with_obs = len(by_day)
    n_obs_total = sum(len(v[0]) for v in by_day.values())
    print(f"  {n_days_with_obs} test days have satellite passes; {n_obs_total} total observations")

    lon3, lat3, states3 = load_states(NC_PATH, subsample_r=3, cache=CACHE_R3)
    ocean3_grid = (states3[:, :, :, 0].std(axis=0) > 1e-4)  # zos-based, land is land regardless of which days
    ocean6_grid = ds["ocean_mask"].reshape(len(lat6), len(lon6))

    def corners_ocean(qlon, qlat, lon_grid, lat_grid, ocean_grid):
        lon_i = np.clip(np.searchsorted(lon_grid, qlon) - 1, 0, len(lon_grid) - 2)
        lat_i = np.clip(np.searchsorted(lat_grid, qlat) - 1, 0, len(lat_grid) - 2)
        return (ocean_grid[lat_i, lon_i] & ocean_grid[lat_i, lon_i + 1]
                & ocean_grid[lat_i + 1, lon_i] & ocean_grid[lat_i + 1, lon_i + 1])

    t_min, t_span = ds["t_min"], ds["t_span"]
    zos_idx = VARIABLES.index("zos")

    direct_preds, grid_preds, persist_preds, truths = [], [], [], []
    for D, (qlon, qlat, obs) in by_day.items():
        i = D - 1  # pair index (branch reads day i, target day D=i+1)
        pos = np.where(test_idx == i)[0]
        if len(pos) == 0:
            continue
        pos = pos[0]

        keep = corners_ocean(qlon, qlat, lon3, lat3, ocean3_grid) & corners_ocean(qlon, qlat, lon6, lat6, ocean6_grid)
        if keep.sum() == 0:
            continue
        qlon, qlat, obs = qlon[keep], qlat[keep], obs[keep]

        # r=3 persistence (day i's actual state) interpolated to exact points -- best-available residual input
        r3_persist_zos_grid = states3[i, :, :, zos_idx]
        interp_persist = RegularGridInterpolator((lat3, lon3), r3_persist_zos_grid, method="linear", bounds_error=False, fill_value=None)
        persist_here = interp_persist(np.stack([qlat, qlon], axis=-1))

        # direct query
        trunk_pts_raw = np.stack([qlon, qlat], axis=-1).astype(np.float64)
        trunk_pts_norm = (2.0 * (trunk_pts_raw - t_min) / t_span - 1.0).astype(np.float32)
        trunk_pts = T_(trunk_pts_norm)
        persist_norm = (persist_here - out_mean[zos_idx]) / out_std[zos_idx]
        persist_query = T_(np.zeros((1, n_vars, len(qlon)), dtype=np.float32))
        persist_query[0, zos_idx, :] = T_(persist_norm.astype(np.float32))
        branch_i = branch_test[pos:pos + 1]
        with torch.no_grad():
            pred_direct = model(branch_i, trunk_pts, persist_at_query=persist_query)
        pred_direct_zos = (pred_direct[0, :, zos_idx].numpy() * out_std[zos_idx] + out_mean[zos_idx])

        # grid-then-interpolate: model's native r=6 grid prediction for this day, interpolated
        grid_zos = pred6_phys[pos, :, zos_idx].reshape(len(lat6), len(lon6))
        interp_grid = RegularGridInterpolator((lat6, lon6), grid_zos, method="linear", bounds_error=False, fill_value=None)
        pred_grid_here = interp_grid(np.stack([qlat, qlon], axis=-1))

        direct_preds.append(pred_direct_zos)
        grid_preds.append(pred_grid_here)
        persist_preds.append(persist_here)
        truths.append(obs)

    direct_preds = np.concatenate(direct_preds)
    grid_preds = np.concatenate(grid_preds)
    persist_preds = np.concatenate(persist_preds)
    truths = np.concatenate(truths)
    n_final = len(truths)
    print(f"\n  {n_final} clean (open-ocean, test-day) satellite obs used for scoring")

    # constant bias correction between GLORYS zos and satellite absolute SSH proxy
    bias_direct = np.mean(truths - direct_preds)
    bias_grid = np.mean(truths - grid_preds)
    bias_persist = np.mean(truths - persist_preds)
    print(f"  mean bias (truth - pred), removed before scoring: direct={bias_direct:.3f}m "
          f"grid={bias_grid:.3f}m persist={bias_persist:.3f}m")

    rmse_direct = np.sqrt(np.mean((direct_preds + bias_direct - truths) ** 2))
    rmse_grid = np.sqrt(np.mean((grid_preds + bias_grid - truths) ** 2))
    rmse_persist = np.sqrt(np.mean((persist_preds + bias_persist - truths) ** 2))
    sk_direct = 1.0 - (rmse_direct / rmse_persist) ** 2
    sk_grid = 1.0 - (rmse_grid / rmse_persist) ** 2

    print("\n=== Skill vs. REAL satellite-observed SSH (zos, bias-corrected) ===")
    print(f"  DeepONet direct query:   rmse={rmse_direct:.4f} m  skill={sk_direct:+.4f}")
    print(f"  Grid-then-interpolate:   rmse={rmse_grid:.4f} m  skill={sk_grid:+.4f}")
    print(f"  Persistence (floor):     rmse={rmse_persist:.4f} m  skill=0.0000 by definition")

    metrics = {
        "n_test_days_with_obs": n_days_with_obs, "n_obs_total_raw": n_obs_total,
        "n_obs_scored": n_final,
        "bias_direct": bias_direct, "bias_grid": bias_grid, "bias_persist": bias_persist,
        "rmse_direct": rmse_direct, "rmse_grid": rmse_grid, "rmse_persist": rmse_persist,
        "skill_direct": sk_direct, "skill_grid": sk_grid,
        "standard_r6_mean_skill_holdout_split": float(np.mean(skills)),
    }
    save_json(args.out_dir / "satellite_validation_metrics.json", metrics)
    print(f"\nSaved to {args.out_dir}/")


if __name__ == "__main__":
    main()
