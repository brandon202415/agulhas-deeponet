#!/usr/bin/env python3
"""Set-encoder version of the sparse-to-dense reconstruction prototype.

The dense masked-vector branch (sparse_reconstruction_prototype.py) collapsed
to predicting climatology and ignoring the sparse input -- likely because a
plain Linear(2*n_sensors -> hidden) branch gives every one of the 6161 possible
sensor positions its own dedicated weight column, and since a different random
~500 positions are "active" each step, the network never gets enough repetition
on any specific position to learn much from it.

This replaces that branch with a set encoder (DeepSets/PointNet-style): each
observed point is represented as (lon, lat, value) and passed through the SAME
small shared per-point MLP, then pooled (mean) into one fixed-size latent
vector -- so the network learns one shared rule ("what does an observation of
value V at some location imply"), which applies no matter which specific
points happen to be lit up this time, rather than 6161 position-specific rules
it can't get enough signal on individually.

Does NOT modify MultivarDeepONet (used throughout the rest of this study) --
this is a standalone model for this experiment only.
"""
import argparse
import numpy as np
import torch
import torch.nn as nn
from scipy.interpolate import griddata

from train_agulhas_deeponet_prototype import load_states, _make_mlp, to_tensor

torch.set_num_threads(4)


class SetEncoder(nn.Module):
    """Per-point MLP (shared weights) + mean pooling -> fixed-size latent vector,
    invariant to point count and order."""

    def __init__(self, in_dim, hidden, latent_dim, depth):
        super().__init__()
        sizes = [in_dim] + [hidden] * (depth - 1) + [latent_dim]
        self.point_mlp = _make_mlp(sizes, "gelu")
        # Zero-init the last linear layer so every point's embedding starts at
        # exactly 0 regardless of input -- the branch output starts at 0, so the
        # whole model starts at exactly climatology (same start-from-a-sensible-
        # baseline philosophy as the persistence residual used elsewhere).
        last_linear = None
        for layer in self.point_mlp.modules():
            if isinstance(layer, nn.Linear):
                last_linear = layer
        if last_linear is not None:
            nn.init.zeros_(last_linear.weight)
            nn.init.zeros_(last_linear.bias)

    def forward(self, points):
        # points: [B, n_obs, in_dim] -> [B, n_obs, latent_dim] -> mean pool -> [B, latent_dim]
        feats = self.point_mlp(points)
        return feats.mean(dim=1)


class SetEncoderDeepONet(nn.Module):
    def __init__(self, branch_hidden, branch_depth, trunk_width, trunk_depth, latent_dim):
        super().__init__()
        self.set_encoder = SetEncoder(in_dim=3, hidden=branch_hidden, latent_dim=latent_dim, depth=branch_depth)
        trunk_sizes = [2] + [trunk_width] * trunk_depth + [latent_dim]
        self.trunk = _make_mlp(trunk_sizes, "tanh")
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, points, trunk_input, clim_at_query):
        branch_feats = self.set_encoder(points)          # [B, latent_dim]
        trunk_feats = self.trunk(trunk_input)             # [n_query, latent_dim]
        out = torch.einsum("bp,qp->bq", branch_feats, trunk_feats) + self.bias
        return out + clim_at_query                        # [B, n_query]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nc", default="data/agulhas_prototype.nc")
    ap.add_argument("--cache", default="data/cache_r6_local.npz")
    ap.add_argument("--coverage", type=float, default=0.10)
    ap.add_argument("--iterations", type=int, default=3000)
    ap.add_argument("--learning-rate", type=float, default=3e-4)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--branch-hidden", type=int, default=64)
    ap.add_argument("--branch-depth", type=int, default=3)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cpu"

    lon6, lat6, states6 = load_states(args.nc, subsample_r=6, cache=args.cache)
    zos = states6[:, :, :, 0].astype(np.float64)
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
    ocean_idx = np.where(ocean_mask)[0]
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
    lon_norm_all, lat_norm_all = trunk_norm[:, 0], trunk_norm[:, 1]

    zos_train = zos_norm[train_idx]
    zos_val = zos_norm[val_idx]
    clim_norm = zos_train.mean(axis=0).astype(np.float32)  # [n_sensors]

    def make_points(day_rows, coverage):
        """day_rows: [B, n_sensors] normalised full fields (one shared mask per
        call, same simplification as the dense-vector prototype). Returns
        points [B, n_obs, 3] = (lon_norm, lat_norm, value_norm) at the SAME
        observed locations for every row in this batch."""
        n_obs = max(1, int(n_ocean * coverage))
        obs_idx = np.random.choice(ocean_idx, size=n_obs, replace=False)
        lon_o = np.tile(lon_norm_all[obs_idx][None, :], (day_rows.shape[0], 1))
        lat_o = np.tile(lat_norm_all[obs_idx][None, :], (day_rows.shape[0], 1))
        val_o = day_rows[:, obs_idx]
        points = np.stack([lon_o, lat_o, val_o], axis=-1).astype(np.float32)  # [B, n_obs, 3]
        return points, obs_idx

    model = SetEncoderDeepONet(
        branch_hidden=args.branch_hidden, branch_depth=args.branch_depth,
        trunk_width=64, trunk_depth=2, latent_dim=32,
    ).to(device)
    print(f"model params: {sum(p.numel() for p in model.parameters()):,}")

    trunk_t = to_tensor(trunk_norm, device)
    ocean_t = torch.tensor(ocean_mask, dtype=torch.bool, device=device)
    clim_t = to_tensor(clim_norm, device)

    opt = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    best_val, best_state, bad = float("inf"), None, 0

    for step in range(1, args.iterations + 1):
        model.train()
        idx = np.random.randint(0, len(train_idx), size=args.batch_size)
        rows = zos_train[idx]
        points_np, _ = make_points(rows, args.coverage)
        points = to_tensor(points_np, device)
        target = to_tensor(rows, device)
        clim_batch = clim_t.unsqueeze(0).expand(points.shape[0], -1)
        pred = model(points, trunk_t, clim_batch)
        loss = ((pred[:, ocean_t] - target[:, ocean_t]) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()

        if step % 100 == 0 or step == args.iterations:
            model.eval()
            with torch.no_grad():
                vpoints_np, _ = make_points(zos_val, args.coverage)
                vpoints = to_tensor(vpoints_np, device)
                vtarget = to_tensor(zos_val, device)
                vclim = clim_t.unsqueeze(0).expand(vpoints.shape[0], -1)
                vpred = model(vpoints, trunk_t, vclim)
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

    zos_test = zos_norm[test_idx]
    np.random.seed(999)
    points_np, obs_idx = make_points(zos_test, args.coverage)
    points = to_tensor(points_np, device)
    clim_batch = clim_t.unsqueeze(0).expand(points.shape[0], -1)
    with torch.no_grad():
        pred_test = model(points, trunk_t, clim_batch).numpy()

    qlon, qlat = LON.ravel()[obs_idx], LAT.ravel()[obs_idx]
    all_lon, all_lat = LON.ravel(), LAT.ravel()

    naive_preds = np.zeros_like(zos_test)
    clim_field = zos_train.mean(axis=0)
    for i in range(zos_test.shape[0]):
        vals = zos_test[i, obs_idx]
        interp = griddata(np.stack([qlon, qlat], axis=-1), vals,
                           np.stack([all_lon, all_lat], axis=-1), method="linear")
        nanmask = np.isnan(interp)
        interp[nanmask] = clim_field[nanmask]
        naive_preds[i] = interp

    def skill(pred, true, ref):
        yt, yp, yr = true[:, ocean_mask], pred[:, ocean_mask], ref[:, ocean_mask]
        rmse_p = np.sqrt(np.mean((yp - yt) ** 2))
        rmse_r = np.sqrt(np.mean((yr - yt) ** 2))
        return (1.0 - (rmse_p / rmse_r) ** 2 if rmse_r > 0 else float("nan")), rmse_p

    clim_tiled = np.tile(clim_field[None, :], (zos_test.shape[0], 1))
    sk_model, rmse_model = skill(pred_test, zos_test, clim_tiled)
    sk_naive, rmse_naive = skill(naive_preds, zos_test, clim_tiled)

    print(f"\n=== Set-encoder sparse reconstruction (coverage={args.coverage:.0%}) ===")
    print(f"  n_obs points used: {len(obs_idx)} / {n_ocean} ocean cells")
    print(f"  DeepONet (set encoder):        rmse(norm)={rmse_model:.4f}  skill vs climatology={sk_model:+.4f}")
    print(f"  Naive scattered interpolation: rmse(norm)={rmse_naive:.4f}  skill vs climatology={sk_naive:+.4f}")


if __name__ == "__main__":
    main()
