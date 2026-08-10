#!/usr/bin/env python3
"""Multi-fidelity sparse reconstruction: pretrain on abundant, cheap
low-fidelity (LF) data -- synthetic random masks over GLORYS, where we control
the mask and so always know the true full field -- then fine-tune and evaluate
on a small amount of genuinely real high-fidelity (HF) data: actual CryoSat-2
satellite passes.

The key design fix versus every earlier attempt today: real satellite data has
no "true full field" to supervise against (that's the whole reason
reconstruction is needed). So the HF task is NOT "predict the full field from
real points" -- it's self-supervised, real-to-real: for each real satellite
pass day, split ITS OWN points into an observed subset (input) and a held-out
subset (target), both genuinely real. No GLORYS truth enters the HF loss or
evaluation at all, so this is not vulnerable to the proxy-truth home-field-
advantage problem found earlier today.

Compares four things on held-out real satellite points from held-out real days:
  1. Pretrained-only  (LF pretraining, zero-shot transfer to the HF task)
  2. Fine-tuned        (LF pretraining + HF fine-tuning)
  3. HF-only           (fresh model, trained only on the small real HF set)
  4. Naive scattered interpolation of the observed real points

The residual/prior used during HF fine-tuning and evaluation is the MEAN of
real HF-train observed values (a scalar), not GLORYS climatology -- avoids
baking in a GLORYS-vs-satellite reference-frame offset into the "sensible
starting point" the model begins from.
"""
import argparse
import datetime
import copy

import numpy as np
import torch
from scipy.interpolate import griddata

from train_agulhas_deeponet_prototype import load_states, to_tensor
from sparse_reconstruction_attention import GeometricAttentionReconstructor

torch.set_num_threads(4)

SAT_FILE = "/Users/brandonzhang/Downloads/data/satellite_sla_agulhas_c2_2019_2021/Cryosat-2.nc"


def load_hf_days(sat_path, lon_bounds, lat_bounds, min_points=20):
    import netCDF4 as nc
    d = nc.Dataset(sat_path)
    lon = d.variables["longitude"][:].astype(np.float64)
    lat = d.variables["latitude"][:].astype(np.float64)
    t = d.variables["time"][:]
    sla = np.ma.filled(d.variables["sla_filtered"][:], np.nan).astype(np.float64)
    mdt = np.ma.filled(d.variables["mdt"][:], np.nan).astype(np.float64)
    val = sla + mdt

    valid = np.isfinite(val) & (lon >= lon_bounds[0]) & (lon <= lon_bounds[1]) \
        & (lat >= lat_bounds[0]) & (lat <= lat_bounds[1])
    lon, lat, t, val = lon[valid], lat[valid], t[valid], val[valid]
    dates = np.array([(datetime.datetime(1970, 1, 1) + datetime.timedelta(seconds=float(s))).date()
                       for s in t])

    by_day = {}
    for dd in np.unique(dates):
        m = dates == dd
        if m.sum() >= min_points:
            by_day[str(dd)] = (lon[m], lat[m], val[m])
    return by_day


def normalise_coords(lon, lat, t_min, t_span):
    raw = np.stack([lon, lat], axis=-1).astype(np.float64)
    return (2.0 * (raw - t_min) / t_span - 1.0).astype(np.float32)


def split_obs_target(n_points, obs_frac, rng):
    idx = rng.permutation(n_points)
    n_obs = max(1, int(n_points * obs_frac))
    return idx[:n_obs], idx[n_obs:]


def evaluate(model, days, day_names, t_min, t_span, clim_scalar, obs_frac, rng, device):
    """Returns (rmse_model, rmse_naive, n_points) pooled over all given days."""
    model.eval()
    model_errs, naive_errs = [], []
    for name in day_names:
        lon, lat, val = days[name]
        obs_i, tgt_i = split_obs_target(len(lon), obs_frac, rng)
        if len(tgt_i) == 0 or len(obs_i) < 3:
            continue
        obs_coords_norm = normalise_coords(lon[obs_i], lat[obs_i], t_min, t_span)
        tgt_coords_norm = normalise_coords(lon[tgt_i], lat[tgt_i], t_min, t_span)
        obs_vals = val[obs_i] - clim_scalar
        tgt_vals = val[tgt_i] - clim_scalar

        with torch.no_grad():
            pred = model(
                to_tensor(tgt_coords_norm, device),
                to_tensor(obs_coords_norm, device),
                to_tensor(obs_vals[None, :], device),
                torch.zeros(1, len(tgt_i), device=device),
            ).cpu().numpy()[0]
        model_errs.append(pred - tgt_vals)

        naive = griddata(np.stack([lon[obs_i], lat[obs_i]], axis=-1), val[obs_i] - clim_scalar,
                          np.stack([lon[tgt_i], lat[tgt_i]], axis=-1), method="linear")
        nanmask = np.isnan(naive)
        naive[nanmask] = 0.0  # already de-meaned; 0 = "predict the mean" fallback
        naive_errs.append(naive - tgt_vals)

    model_errs = np.concatenate(model_errs)
    naive_errs = np.concatenate(naive_errs)
    return (np.sqrt(np.mean(model_errs ** 2)), np.sqrt(np.mean(naive_errs ** 2)), len(model_errs))


def fine_tune(model, days, train_names, val_names, t_min, t_span, clim_scalar,
              obs_frac, iterations, lr, patience, device, seed):
    rng = np.random.default_rng(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best_val, best_state, bad = float("inf"), copy.deepcopy(model.state_dict()), 0

    for step in range(1, iterations + 1):
        model.train()
        name = train_names[rng.integers(0, len(train_names))]
        lon, lat, val = days[name]
        obs_i, tgt_i = split_obs_target(len(lon), obs_frac, rng)
        if len(tgt_i) == 0 or len(obs_i) < 3:
            continue
        obs_coords_norm = normalise_coords(lon[obs_i], lat[obs_i], t_min, t_span)
        tgt_coords_norm = normalise_coords(lon[tgt_i], lat[tgt_i], t_min, t_span)
        obs_vals = to_tensor((val[obs_i] - clim_scalar)[None, :], device)
        tgt_vals = to_tensor((val[tgt_i] - clim_scalar)[None, :], device)
        clim_at_query = torch.zeros(1, len(tgt_i), device=device)

        pred = model(to_tensor(tgt_coords_norm, device), to_tensor(obs_coords_norm, device),
                     obs_vals, clim_at_query)
        loss = ((pred - tgt_vals) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()

        if step % 50 == 0 or step == iterations:
            rmse_val, _, _ = evaluate(model, days, val_names, t_min, t_span, clim_scalar,
                                       obs_frac, np.random.default_rng(12345), device)
            print(f"  ft step {step:5d}  train_loss {loss.item():.5f}  val_rmse {rmse_val:.5f}")
            if rmse_val < best_val - 1e-6:
                best_val, best_state, bad = rmse_val, copy.deepcopy(model.state_dict()), 0
            else:
                bad += 1
                if bad >= patience:
                    print(f"  early stop at ft step {step} (best val_rmse={best_val:.5f})")
                    break

    model.load_state_dict(best_state)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nc", default="data/agulhas_prototype.nc")
    ap.add_argument("--cache", default="data/cache_r6_local.npz")
    ap.add_argument("--lf-coverage", type=float, default=0.10)
    ap.add_argument("--lf-iterations", type=int, default=3000)
    ap.add_argument("--lf-lr", type=float, default=3e-4)
    ap.add_argument("--lf-batch-size", type=int, default=32)
    ap.add_argument("--lf-query-subsample", type=int, default=400)
    ap.add_argument("--hf-obs-frac", type=float, default=0.7)
    ap.add_argument("--hf-ft-iterations", type=int, default=800)
    ap.add_argument("--hf-ft-lr", type=float, default=1e-4)
    ap.add_argument("--score-hidden", type=int, default=64)
    ap.add_argument("--score-depth", type=int, default=3)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")

    # ══════════════════ Stage 1: LF pretraining (GLORYS masking) ══════════════════
    lon6, lat6, states6 = load_states(args.nc, subsample_r=6, cache=args.cache)
    zos = states6[:, :, :, 0].astype(np.float64)
    T, nlat, nlon = zos.shape
    n_sensors = nlat * nlon
    zos_flat = zos.reshape(T, n_sensors)

    n_test = int(T * 0.15)
    n_val = int(T * 0.15)
    n_train = T - n_val - n_test
    train_idx = np.arange(0, n_train)

    ocean_mask = zos_flat[train_idx].std(axis=0) > 1e-4
    n_ocean = ocean_mask.sum()
    ocean_idx = np.where(ocean_mask)[0]

    m = zos_flat[train_idx][:, ocean_mask].mean()
    s = zos_flat[train_idx][:, ocean_mask].std()
    zos_norm = (zos_flat - m) / s
    zos_norm[:, ~ocean_mask] = 0.0
    zos_train = zos_norm[train_idx]
    clim_norm = zos_train.mean(axis=0).astype(np.float32)

    LAT, LON = np.meshgrid(lat6, lon6, indexing="ij")
    trunk_raw = np.stack([LON.ravel(), LAT.ravel()], axis=-1).astype(np.float64)
    t_min = trunk_raw.min(axis=0, keepdims=True)
    t_span = trunk_raw.max(axis=0, keepdims=True) - t_min
    trunk_norm = (2.0 * (trunk_raw - t_min) / t_span - 1.0).astype(np.float32)
    query_t = to_tensor(trunk_norm, device)
    clim_t = to_tensor(clim_norm, device)

    model = GeometricAttentionReconstructor(hidden=args.score_hidden, depth=args.score_depth).to(device)
    print(f"model params: {sum(p.numel() for p in model.parameters()):,}")

    opt = torch.optim.Adam(model.parameters(), lr=args.lf_lr)
    ocean_query_idx = np.where(ocean_mask)[0]
    lon_min6, lon_max6 = float(lon6.min()), float(lon6.max())
    lat_min6, lat_max6 = float(lat6.min()), float(lat6.max())
    nlon6, nlat6 = len(lon6), len(lat6)

    # Precompute width-offset grid once (vectorised track-mask generation).
    _wc = 2
    _doff, _loff = np.meshgrid(np.arange(-_wc, _wc + 1), np.arange(-_wc, _wc + 1), indexing="ij")
    _doff, _loff = _doff.ravel(), _loff.ravel()  # (dlat, dlon) offset pairs

    def sample_track_mask(n_tracks=2, width_cells=2):
        """Ocean grid cells near a few random straight-line tracks crossing the
        full domain -- narrow, elongated swaths, matching the real CryoSat-2
        along-track geometry (angles roughly -45 to +45 deg off north-south,
        each pass spanning the full latitude range) -- rather than uniform 2D
        scatter, which was diagnosed as causing negative transfer to the real
        HF task (the model learns a distance-relevance kernel tuned to the
        wrong sampling distribution). Vectorised (numpy, no Python loops over
        points) for speed -- the original pure-Python version was too slow to
        call once per training step."""
        all_idx = []
        for _ in range(n_tracks):
            angle = np.random.uniform(-45, 45)
            theta = np.radians(angle)
            x0 = np.random.uniform(lon_min6, lon_max6)
            y0 = lat_min6 - 3.0
            length = (lat_max6 - lat_min6 + 6.0) / max(np.cos(theta), 1e-3)
            d = np.arange(0, length, 0.15)
            x = x0 + d * np.sin(theta)
            y = y0 + d * np.cos(theta)
            valid = (x >= lon_min6) & (x <= lon_max6) & (y >= lat_min6) & (y <= lat_max6)
            x, y = x[valid], y[valid]
            if len(x) == 0:
                continue
            lon_i = np.clip(np.round((x - lon_min6) / (lon_max6 - lon_min6) * (nlon6 - 1)).astype(int), 0, nlon6 - 1)
            lat_i = np.clip(np.round((y - lat_min6) / (lat_max6 - lat_min6) * (nlat6 - 1)).astype(int), 0, nlat6 - 1)
            # broadcast every track point against every width offset at once
            li = lat_i[:, None] + _loff[None, :]       # [n_points, n_offsets]
            lj = lon_i[:, None] + _doff[None, :]
            in_bounds = (li >= 0) & (li < nlat6) & (lj >= 0) & (lj < nlon6)
            li_c, lj_c = np.clip(li, 0, nlat6 - 1), np.clip(lj, 0, nlon6 - 1)
            idx = (li_c * nlon6 + lj_c)[in_bounds]
            all_idx.append(idx)
        if not all_idx:
            return np.array([ocean_idx[0]])
        idx = np.unique(np.concatenate(all_idx))
        idx = idx[ocean_mask[idx]]
        return idx if len(idx) > 0 else np.array([ocean_idx[0]])

    def sample_mask(coverage):
        # `coverage` kept as an argument for interface compatibility but no
        # longer used directly -- track shape/width now determines point count.
        obs_idx = sample_track_mask(n_tracks=2, width_cells=2)
        return obs_idx, to_tensor(trunk_norm[obs_idx], device)

    def subsample_query(n):
        qi = np.random.choice(ocean_query_idx, size=min(n, len(ocean_query_idx)), replace=False)
        return qi, query_t[qi]

    print("=== Stage 1: LF pretraining on GLORYS-masking ===")
    best_val, best_state, bad = float("inf"), None, 0
    for step in range(1, args.lf_iterations + 1):
        model.train()
        idx = np.random.randint(0, len(train_idx), size=args.lf_batch_size)
        rows = zos_train[idx]
        obs_idx, obs_coords_t = sample_mask(args.lf_coverage)
        obs_vals = to_tensor(rows[:, obs_idx], device)
        qi, query_sub = subsample_query(args.lf_query_subsample)
        target_sub = to_tensor(rows[:, qi], device)
        clim_sub = clim_t[qi].unsqueeze(0)
        pred = model(query_sub, obs_coords_t, obs_vals, clim_sub)
        loss = ((pred - target_sub) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()

        if step % 200 == 0 or step == args.lf_iterations:
            model.eval()
            with torch.no_grad():
                vobs_idx, vobs_coords_t = sample_mask(args.lf_coverage)
                vobs_vals = to_tensor(zos_norm[n_train:n_train + n_val][:, vobs_idx], device)
                vqi, vquery_sub = subsample_query(args.lf_query_subsample)
                vtarget = to_tensor(zos_norm[n_train:n_train + n_val][:, vqi], device)
                vclim_sub = clim_t[vqi].unsqueeze(0)
                vpred = model(vquery_sub, vobs_coords_t, vobs_vals, vclim_sub)
                vloss = ((vpred - vtarget) ** 2).mean().item()
            print(f"  lf step {step:5d}  train_loss {loss.item():.5f}  val_loss {vloss:.5f}")
            if vloss < best_val - 1e-6:
                best_val, best_state, bad = vloss, copy.deepcopy(model.state_dict()), 0
            else:
                bad += 1
                if bad >= args.patience:
                    print(f"  LF early stop at step {step} (best val_loss={best_val:.5f})")
                    break
    model.load_state_dict(best_state)
    pretrained_state = copy.deepcopy(model.state_dict())

    # ══════════════════ Stage 2: HF data (real CryoSat-2) ══════════════════
    print("\n=== Stage 2: loading real HF (satellite) days ===")
    days = load_hf_days(SAT_FILE, (float(lon6.min()), float(lon6.max())),
                         (float(lat6.min()), float(lat6.max())), min_points=20)
    day_names = sorted(days.keys())
    print(f"  {len(day_names)} real satellite days with >=20 points in domain")

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(day_names))
    n_hf_train = int(0.70 * len(day_names))
    n_hf_val = int(0.15 * len(day_names))
    hf_train_names = [day_names[i] for i in perm[:n_hf_train]]
    hf_val_names = [day_names[i] for i in perm[n_hf_train:n_hf_train + n_hf_val]]
    hf_test_names = [day_names[i] for i in perm[n_hf_train + n_hf_val:]]
    print(f"  HF split: train={len(hf_train_names)} val={len(hf_val_names)} test={len(hf_test_names)} days")

    clim_scalar = float(np.mean([days[n][2].mean() for n in hf_train_names]))
    print(f"  HF residual base (mean of real HF-train values): {clim_scalar:.4f} m")

    eval_rng = np.random.default_rng(777)

    # --- 1. Pretrained-only (zero-shot LF -> HF transfer) ---
    model.load_state_dict(pretrained_state)
    rmse_pt, rmse_naive_pt, n_pt = evaluate(model, days, hf_test_names, t_min, t_span,
                                             clim_scalar, args.hf_obs_frac, eval_rng, device)

    # --- 2. Fine-tuned (LF pretrain + HF fine-tune) ---
    print("\n=== Fine-tuning pretrained model on real HF data ===")
    model.load_state_dict(pretrained_state)
    model = fine_tune(model, days, hf_train_names, hf_val_names, t_min, t_span, clim_scalar,
                       args.hf_obs_frac, args.hf_ft_iterations, args.hf_ft_lr, args.patience, device, args.seed)
    eval_rng = np.random.default_rng(777)
    rmse_ft, rmse_naive_ft, n_ft = evaluate(model, days, hf_test_names, t_min, t_span,
                                             clim_scalar, args.hf_obs_frac, eval_rng, device)

    # --- 3. HF-only (fresh model, no LF pretraining) ---
    print("\n=== Training HF-only model (no LF pretraining) from scratch ===")
    hf_only_model = GeometricAttentionReconstructor(hidden=args.score_hidden, depth=args.score_depth).to(device)
    hf_only_model = fine_tune(hf_only_model, days, hf_train_names, hf_val_names, t_min, t_span, clim_scalar,
                              args.hf_obs_frac, args.hf_ft_iterations, args.hf_ft_lr, args.patience, device, args.seed)
    eval_rng = np.random.default_rng(777)
    rmse_hfonly, rmse_naive_hfonly, n_hfonly = evaluate(hf_only_model, days, hf_test_names, t_min, t_span,
                                                          clim_scalar, args.hf_obs_frac, eval_rng, device)

    print(f"\n=== Results on held-out REAL satellite points, held-out REAL days ({len(hf_test_names)} test days) ===")
    print(f"  Pretrained-only (zero-shot):  model_rmse={rmse_pt:.4f} m  naive_rmse={rmse_naive_pt:.4f} m  n={n_pt}")
    print(f"  Fine-tuned (LF+HF):           model_rmse={rmse_ft:.4f} m  naive_rmse={rmse_naive_ft:.4f} m  n={n_ft}")
    print(f"  HF-only (no LF pretraining):  model_rmse={rmse_hfonly:.4f} m  naive_rmse={rmse_naive_hfonly:.4f} m  n={n_hfonly}")
    print("\n(lower rmse is better; naive = scattered linear interpolation of the observed")
    print(" real points to the held-out real target points, same de-meaned reference frame)")


if __name__ == "__main__":
    main()
