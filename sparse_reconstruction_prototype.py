#!/usr/bin/env python3
"""Quick, local-only prototype for the sparse-to-dense reconstruction idea:
given only a random, sparse, masked subset of a day's SSH (zos) field --
mimicking sparse satellite altimetry coverage -- can a DeepONet-style
branch/trunk network reconstruct the FULL field better than naive scattered-
point interpolation of the same sparse data?

This is a deliberately different task from everything else in this study: same-
day spatial reconstruction/gap-filling, not next-day forecasting. No real
satellite data needed yet -- this uses synthetic random masks over the existing
local GLORYS-like cube to de-risk the architecture (does sparse-masked branch
input + full-grid trunk query even work at all) before spending effort on real
multi-satellite data engineering.

Branch input: [masked_value; mask_indicator] concatenated (2 x n_sensors),
land AND unmasked-out points all zero. No persistence residual here (there is
no "current state" to persist from -- this is single-day reconstruction).
"""
import argparse
import numpy as np
import torch
from scipy.interpolate import griddata

from train_agulhas_deeponet_prototype import load_states, MultivarDeepONet, to_tensor

torch.set_num_threads(4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nc", default="data/agulhas_prototype.nc")
    ap.add_argument("--cache", default="data/cache_r6_local.npz")
    ap.add_argument("--coverage", type=float, default=0.10, help="fraction of ocean cells 'observed' per sample")
    ap.add_argument("--iterations", type=int, default=3000)
    ap.add_argument("--learning-rate", type=float, default=3e-4)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cpu"

    lon6, lat6, states6 = load_states(args.nc, subsample_r=6, cache=args.cache)
    zos = states6[:, :, :, 0].astype(np.float64)  # [T, nlat, nlon]
    T, nlat, nlon = zos.shape
    n_sensors = nlat * nlon
    zos_flat = zos.reshape(T, n_sensors)

    n_test = int(T * 0.15)
    n_val = int(T * 0.15)
    n_train = T - n_val - n_test
    train_idx = np.arange(0, n_train)
    val_idx = np.arange(n_train, n_train + n_val)
    test_idx = np.arange(n_train + n_val, T)
    print(f"days: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")

    ocean_mask = zos_flat[train_idx].std(axis=0) > 1e-4
    n_ocean = ocean_mask.sum()
    print(f"ocean cells: {n_ocean} / {n_sensors}")

    m = zos_flat[train_idx][:, ocean_mask].mean()
    s = zos_flat[train_idx][:, ocean_mask].std()
    zos_norm = (zos_flat - m) / s
    zos_norm[:, ~ocean_mask] = 0.0

    LAT, LON = np.meshgrid(lat6, lon6, indexing="ij")
    trunk_raw = np.stack([LON.ravel(), LAT.ravel()], axis=-1).astype(np.float64)
    t_min = trunk_raw.min(axis=0, keepdims=True)
    t_span = trunk_raw.max(axis=0, keepdims=True) - t_min
    trunk_norm = (2.0 * (trunk_raw - t_min) / t_span - 1.0).astype(np.float32)

    ocean_idx = np.where(ocean_mask)[0]

    def make_sparse_branch(day_rows, coverage):
        """day_rows: [B, n_sensors] normalised full fields. Returns branch input
        [B, 2*n_sensors] = concat(masked_value, mask_indicator), and the boolean
        mask used (same for all rows in this call, for simplicity/speed)."""
        n_obs = max(1, int(n_ocean * coverage))
        obs_idx = np.random.choice(ocean_idx, size=n_obs, replace=False)
        mask = np.zeros(n_sensors, dtype=np.float32)
        mask[obs_idx] = 1.0
        masked_val = day_rows * mask[None, :]
        branch = np.concatenate([masked_val, np.tile(mask[None, :], (day_rows.shape[0], 1))], axis=1)
        return branch.astype(np.float32), obs_idx

    model = MultivarDeepONet(
        d_branch=2 * n_sensors, n_sensors=n_sensors, n_vars=1,
        branch_width=64, branch_depth=2, trunk_width=64, trunk_depth=2,
        latent_dim=32, residual=True,
    ).to(device)
    print(f"model params: {sum(p.numel() for p in model.parameters()):,}")

    trunk_t = to_tensor(trunk_norm, device)
    ocean_t = torch.tensor(ocean_mask, dtype=torch.bool, device=device)
    zos_train = zos_norm[train_idx]
    zos_val = zos_norm[val_idx]

    # Per-sensor climatology (long-term mean field), in the SAME normalised space as
    # the target -- used as the residual base ("start from the mean field, learn only
    # the sparse-informed deviation") instead of persistence, since there is no
    # previous timestep in a same-day reconstruction task. Reuses the exact
    # persist_at_query mechanism built for the forecasting models (Sec. 3.7) --
    # climatology just fills the same "sensible starting point" slot persistence did.
    clim_norm = zos_train.mean(axis=0).astype(np.float32)  # [n_sensors]
    clim_t = to_tensor(clim_norm, device)

    def clim_query(batch_size):
        return clim_t.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, n_sensors)  # [B, 1, n_sensors]

    opt = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    best_val, best_state, bad = float("inf"), None, 0

    for step in range(1, args.iterations + 1):
        model.train()
        idx = np.random.randint(0, len(train_idx), size=args.batch_size)
        rows = zos_train[idx]
        branch_np, _ = make_sparse_branch(rows, args.coverage)
        branch = to_tensor(branch_np, device)
        target = to_tensor(rows, device)
        pred = model(branch, trunk_t, persist_at_query=clim_query(branch.shape[0]))[:, :, 0]  # [B, n_sensors]
        loss = ((pred[:, ocean_t] - target[:, ocean_t]) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()

        if step % 100 == 0 or step == args.iterations:
            model.eval()
            with torch.no_grad():
                vbranch_np, _ = make_sparse_branch(zos_val, args.coverage)
                vbranch = to_tensor(vbranch_np, device)
                vtarget = to_tensor(zos_val, device)
                vpred = model(vbranch, trunk_t, persist_at_query=clim_query(vbranch.shape[0]))[:, :, 0]
                vloss = ((vpred[:, ocean_t] - vtarget[:, ocean_t]) ** 2).mean().item()
            print(f"step {step:5d}  train_loss {loss.item():.5f}  val_loss {vloss:.5f}")
            if vloss < best_val - 1e-6:
                best_val, best_state, bad = vloss, {k: v.clone() for k, v in model.state_dict().items()}, 0
            else:
                bad += 1
                if bad >= args.patience:
                    print(f"Early stop at step {step} (best val_loss={best_val:.5f})")
                    break

    model.load_state_dict(best_state)
    model.eval()

    # --- Evaluate on TEST days: model reconstruction vs naive scattered interpolation vs climatology ---
    zos_test = zos_norm[test_idx]
    np.random.seed(999)  # fixed test-time masks, independent of training-time mask stream
    branch_np, obs_idx_last = make_sparse_branch(zos_test, args.coverage)
    # (re-derive per-row masks isn't needed: make_sparse_branch uses ONE shared mask
    #  per call, applied identically to all rows -- fine for this prototype, since we
    #  just want a representative single coverage pattern to score against)
    branch = to_tensor(branch_np, device)
    with torch.no_grad():
        pred_test = model(branch, trunk_t, persist_at_query=clim_query(branch.shape[0]))[:, :, 0].numpy()

    obs_idx = obs_idx_last
    qlon, qlat = LON.ravel()[obs_idx], LAT.ravel()[obs_idx]
    all_lon, all_lat = LON.ravel(), LAT.ravel()

    naive_preds = np.zeros_like(zos_test)
    clim_field = zos_train.mean(axis=0)
    for i in range(zos_test.shape[0]):
        vals = zos_test[i, obs_idx]
        interp = griddata(np.stack([qlon, qlat], axis=-1), vals,
                           np.stack([all_lon, all_lat], axis=-1), method="linear")
        nanmask = np.isnan(interp)
        interp[nanmask] = clim_field[nanmask]  # fall back to climatology where linear interp can't extrapolate
        naive_preds[i] = interp

    def skill(pred, true, ref):
        yt, yp, yr = true[:, ocean_mask], pred[:, ocean_mask], ref[:, ocean_mask]
        rmse_p = np.sqrt(np.mean((yp - yt) ** 2))
        rmse_r = np.sqrt(np.mean((yr - yt) ** 2))
        return 1.0 - (rmse_p / rmse_r) ** 2 if rmse_r > 0 else float("nan"), rmse_p

    clim_tiled = np.tile(clim_field[None, :], (zos_test.shape[0], 1))
    sk_model, rmse_model = skill(pred_test, zos_test, clim_tiled)
    sk_naive, rmse_naive = skill(naive_preds, zos_test, clim_tiled)

    print(f"\n=== Sparse reconstruction test (coverage={args.coverage:.0%} of ocean cells observed) ===")
    print(f"  n_obs points used: {len(obs_idx)} / {n_ocean} ocean cells")
    print(f"  DeepONet reconstruction:      rmse(norm)={rmse_model:.4f}  skill vs climatology={sk_model:+.4f}")
    print(f"  Naive scattered interpolation: rmse(norm)={rmse_naive:.4f}  skill vs climatology={sk_naive:+.4f}")
    print("\n(skill here is relative to CLIMATOLOGY, not persistence -- there is no previous")
    print(" timestep in a same-day reconstruction task. Positive = better than the mean field;")
    print(" DeepONet beating naive interpolation would mean the learned reconstruction uses")
    print(" more than just spatial smoothness of the sparse points themselves.)")


if __name__ == "__main__":
    main()
