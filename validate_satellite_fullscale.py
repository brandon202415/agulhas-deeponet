#!/usr/bin/env python3
"""Full-scale version of the real-satellite validation check.

Unlike the local-prototype version (train_deeponet_satellite_holdout.py, which
needed a custom non-chronological split because the local file only starts
2020-01-01), the FULL reanalysis (1993-01-01 to 2021-06-30) has its standard
chronological test window (2017-03-23 to 2021-06-30, from the paper's actual
7285/1561/1561 split) already fully containing the CryoSat-2 satellite data's
entire coverage (2019-01-01 to 2020-07-31). So this uses the paper's REAL,
already-established whole-domain checkpoint and standard train/val/test split --
no compromise split needed.

Reuses an existing checkpoint if --checkpoint is given (recommended: point it at
whatever produced the paper's Table 1 headline whole-domain r=6 result); trains a
fresh one with the standard recipe otherwise.

Usage:
    python3 validate_satellite_fullscale.py \\
        --nc "data/agulhas_*.nc" --cache-r6 data/cache_r6.npz --cache-r3 data/cache_r3.npz \\
        --satellite data/satellite_sla_agulhas_c2_2019_2021.nc \\
        --checkpoint results/agulhas_baseline/model.pt \\
        --out-dir results/satellite_validation_fullscale
"""
import argparse
import datetime
from pathlib import Path

import numpy as np
import torch
from scipy.interpolate import RegularGridInterpolator

from train_agulhas_deeponet_prototype import (
    load_states, build_dataset, MultivarDeepONet, VARIABLES, save_json,
)

torch.set_num_threads(8)

REANALYSIS_EPOCH = datetime.date(1993, 1, 1)  # day-index 0 of the FULL reanalysis stream


def load_satellite_obs(sat_path, test_target_dates_set, lon_bounds, lat_bounds):
    import netCDF4 as nc
    d = nc.Dataset(sat_path)
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
    day_idx = np.array([(dd - REANALYSIS_EPOCH).days for dd in dates])

    by_day = {}
    for D in np.unique(day_idx):
        if D not in test_target_dates_set:
            continue
        m = day_idx == D
        by_day[int(D)] = (lon[m], lat[m], ssh_proxy[m])
    return by_day


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nc", default="data/agulhas_*.nc")
    ap.add_argument("--cache-r6", default="data/cache_r6.npz")
    ap.add_argument("--cache-r3", default="data/cache_r3.npz")
    ap.add_argument("--satellite", required=True)
    ap.add_argument("--checkpoint", default=None,
                     help="existing whole-domain r=6 model.pt to reuse; trains fresh if omitted")
    ap.add_argument("--iterations", type=int, default=8000)
    ap.add_argument("--learning-rate", type=float, default=3e-4)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")

    lon6, lat6, states6 = load_states(args.nc, subsample_r=6, cache=args.cache_r6)
    ds = build_dataset(states6, lon6, lat6, test_fraction=0.15, val_fraction=0.15, step_days=1)
    n_sensors6, n_vars = ds["n_sensors"], ds["n_vars"]
    out_mean, out_std = ds["out_mean"], ds["out_std"]
    print(f"Standard chronological split: train={len(ds['train_idx'])} val={len(ds['val_idx'])} "
          f"test={len(ds['test_idx'])} days")
    test_target_start = REANALYSIS_EPOCH + datetime.timedelta(days=int(ds["test_idx"][0]) + 1)
    test_target_end = REANALYSIS_EPOCH + datetime.timedelta(days=int(ds["test_idx"][-1]) + 1)
    print(f"Test target-day date range: {test_target_start} to {test_target_end}")

    model = MultivarDeepONet(
        d_branch=n_sensors6 * n_vars, n_sensors=n_sensors6, n_vars=n_vars,
        branch_width=64, branch_depth=2, trunk_width=64, trunk_depth=2, latent_dim=32,
    ).to(device)

    def T_(a):
        return torch.tensor(a, dtype=torch.float32, device=device)

    if args.checkpoint:
        print(f"Loading existing checkpoint: {args.checkpoint}")
        model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=False))
    else:
        print("No checkpoint given -- training fresh with the standard recipe.")
        branch_train, branch_val = T_(ds["branch_train"]), T_(ds["branch_val"])
        trunk6 = T_(ds["trunk"])
        y_train, y_val = T_(ds["y_train_norm"]), T_(ds["y_val_norm"])
        ocean_vm = T_(np.tile(ds["ocean_mask"], n_vars).astype(np.float32)).bool()
        w = T_(ds["loss_weight"])
        opt = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        n_train = branch_train.shape[0]
        best_val, best_state, bad = float("inf"), None, 0
        for step in range(1, args.iterations + 1):
            model.train()
            idx = torch.randint(0, n_train, (min(args.batch_size, n_train),), device=device)
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
        torch.save(model.state_dict(), args.out_dir / "model.pt")

    model.eval()
    branch_test = T_(ds["branch_test"])
    trunk6 = T_(ds["trunk"])
    with torch.no_grad():
        pred6 = model(branch_test, trunk6).cpu().numpy()
    pred6_phys = pred6 * out_std[None, None, :] + out_mean[None, None, :]
    ocean6 = ds["ocean_mask"]

    print("\n=== Standard r=6 test skill (full-scale, standard split) ===")
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
    print(f"  mean skill = {np.mean(skills):+.4f}  (should match paper's Table 1 headline, ~+0.043, if --checkpoint reused it)")

    print("\n=== Loading real CryoSat-2 satellite observations ===")
    test_target_days = set((ds["test_idx"] + 1).tolist())
    by_day = load_satellite_obs(args.satellite, test_target_days,
                                 (float(lon6.min()), float(lon6.max())),
                                 (float(lat6.min()), float(lat6.max())))
    n_days_with_obs = len(by_day)
    n_obs_total = sum(len(v[0]) for v in by_day.values())
    print(f"  {n_days_with_obs} test days have satellite passes; {n_obs_total} total observations")

    lon3, lat3, states3 = load_states(args.nc, subsample_r=3, cache=args.cache_r3)
    ocean3_grid = (states3[:, :, :, 0].std(axis=0) > 1e-4)
    ocean6_grid = ds["ocean_mask"].reshape(len(lat6), len(lon6))

    def corners_ocean(qlon, qlat, lon_grid, lat_grid, ocean_grid):
        lon_i = np.clip(np.searchsorted(lon_grid, qlon) - 1, 0, len(lon_grid) - 2)
        lat_i = np.clip(np.searchsorted(lat_grid, qlat) - 1, 0, len(lat_grid) - 2)
        return (ocean_grid[lat_i, lon_i] & ocean_grid[lat_i, lon_i + 1]
                & ocean_grid[lat_i + 1, lon_i] & ocean_grid[lat_i + 1, lon_i + 1])

    LAT6, LON6 = np.meshgrid(lat6, lon6, indexing="ij")
    trunk6_raw = np.stack([LON6.ravel(), LAT6.ravel()], axis=-1).astype(np.float64)
    t_min = trunk6_raw.min(axis=0, keepdims=True)
    t_span = trunk6_raw.max(axis=0, keepdims=True) - t_min
    zos_idx = VARIABLES.index("zos")
    test_idx = ds["test_idx"]

    direct_preds, grid_preds, persist_preds, truths = [], [], [], []
    for D, (qlon, qlat, obs) in by_day.items():
        i = D - 1
        pos_arr = np.where(test_idx == i)[0]
        if len(pos_arr) == 0:
            continue
        pos = pos_arr[0]

        keep = corners_ocean(qlon, qlat, lon3, lat3, ocean3_grid) & corners_ocean(qlon, qlat, lon6, lat6, ocean6_grid)
        if keep.sum() == 0:
            continue
        qlon, qlat, obs = qlon[keep], qlat[keep], obs[keep]

        r3_persist_zos_grid = states3[i, :, :, zos_idx]
        interp_persist = RegularGridInterpolator((lat3, lon3), r3_persist_zos_grid, method="linear", bounds_error=False, fill_value=None)
        persist_here = interp_persist(np.stack([qlat, qlon], axis=-1))

        trunk_pts_raw = np.stack([qlon, qlat], axis=-1).astype(np.float64)
        trunk_pts_norm = (2.0 * (trunk_pts_raw - t_min) / t_span - 1.0).astype(np.float32)
        trunk_pts = T_(trunk_pts_norm)
        persist_norm = (persist_here - out_mean[zos_idx]) / out_std[zos_idx]
        persist_query = T_(np.zeros((1, n_vars, len(qlon)), dtype=np.float32))
        persist_query[0, zos_idx, :] = T_(persist_norm.astype(np.float32))
        branch_i = branch_test[pos:pos + 1]
        with torch.no_grad():
            pred_direct = model(branch_i, trunk_pts, persist_at_query=persist_query)
        pred_direct_zos = (pred_direct[0, :, zos_idx].cpu().numpy() * out_std[zos_idx] + out_mean[zos_idx])

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

    print("\n=== Skill vs. REAL satellite-observed SSH (zos, bias-corrected, FULL SCALE) ===")
    print(f"  DeepONet direct query:   rmse={rmse_direct:.4f} m  skill={sk_direct:+.4f}")
    print(f"  Grid-then-interpolate:   rmse={rmse_grid:.4f} m  skill={sk_grid:+.4f}")
    print(f"  Persistence (floor):     rmse={rmse_persist:.4f} m  skill=0.0000 by definition")

    metrics = {
        "n_test_days_with_obs": n_days_with_obs, "n_obs_total_raw": n_obs_total,
        "n_obs_scored": n_final,
        "bias_direct": bias_direct, "bias_grid": bias_grid, "bias_persist": bias_persist,
        "rmse_direct": rmse_direct, "rmse_grid": rmse_grid, "rmse_persist": rmse_persist,
        "skill_direct": sk_direct, "skill_grid": sk_grid,
        "standard_r6_mean_skill": float(np.mean(skills)),
    }
    save_json(args.out_dir / "satellite_validation_metrics.json", metrics)
    print(f"\nSaved to {args.out_dir}/")


if __name__ == "__main__":
    main()
