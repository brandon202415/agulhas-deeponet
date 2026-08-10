#!/usr/bin/env python3
"""Attention-based sparse-to-dense reconstruction: fixes the diagnosed problem
with both prior prototypes (dense masked vector, mean-pooled set encoder) --
neither had any mechanism connecting a specific query location to the specific
nearby observations that should matter most for it, since both produced one
global, query-independent summary vector used identically for every output
point.

This computes attention weights directly from RELATIVE position (query coord
minus observation coord) via a small learned kernel -- a direct, learnable
generalization of classical inverse-distance-weighting / kriging-style
interpolation, rather than raw query/key dot-product attention (which would
need to implicitly rediscover "nearby = more relevant" from absolute
coordinates, a much harder thing to learn from scratch with this little data).

Because observation locations are shared across a training batch (one random
mask per step, matching the earlier prototypes' simplification), the
(query, observation) relative-position grid and its attention weights are
computed ONCE per step, shared across the batch -- only the actual observed
VALUES vary per sample. This keeps it cheap: O(n_query * n_obs) for the
weights (via a tiny MLP), then a plain batched matmul per sample.
"""
import argparse
import numpy as np
import torch
import torch.nn as nn
from scipy.interpolate import griddata

from train_agulhas_deeponet_prototype import load_states, _make_mlp, to_tensor

torch.set_num_threads(4)


class GeometricAttentionReconstructor(nn.Module):
    def __init__(self, hidden=64, depth=3):
        super().__init__()
        sizes = [2] + [hidden] * (depth - 1) + [1]  # input: (dlon, dlat) -> score
        self.score_mlp = _make_mlp(sizes, "gelu")
        # Zero-init the last layer -> all raw scores start at 0 -> softmax gives
        # uniform weights at init (equivalent to mean pooling, i.e. climatology-
        # like behaviour at init, same start-from-a-sensible-baseline philosophy).
        last_linear = None
        for layer in self.score_mlp.modules():
            if isinstance(layer, nn.Linear):
                last_linear = layer
        if last_linear is not None:
            nn.init.zeros_(last_linear.weight)
            nn.init.zeros_(last_linear.bias)

    def forward(self, query_coords, obs_coords, obs_vals, clim_at_query):
        # query_coords: [n_query, 2]   obs_coords: [n_obs, 2]  (both shared across batch)
        # obs_vals: [B, n_obs]         clim_at_query: [n_query] or [B, n_query]
        rel = query_coords[:, None, :] - obs_coords[None, :, :]   # [n_query, n_obs, 2]
        raw_scores = self.score_mlp(rel).squeeze(-1)              # [n_query, n_obs]
        weights = torch.softmax(raw_scores, dim=-1)                # [n_query, n_obs]
        recon = torch.einsum("qn,bn->bq", weights, obs_vals)       # [B, n_query]
        return recon + clim_at_query


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nc", default="data/agulhas_prototype.nc")
    ap.add_argument("--cache", default="data/cache_r6_local.npz")
    ap.add_argument("--coverage", type=float, default=0.10)
    ap.add_argument("--iterations", type=int, default=3000)
    ap.add_argument("--learning-rate", type=float, default=3e-4)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--score-hidden", type=int, default=64)
    ap.add_argument("--score-depth", type=int, default=3)
    ap.add_argument("--query-subsample", type=int, default=400,
                     help="random query points scored per TRAINING step (full grid used only at final eval)")
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")

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

    zos_train = zos_norm[train_idx]
    zos_val = zos_norm[val_idx]
    clim_norm = zos_train.mean(axis=0).astype(np.float32)

    model = GeometricAttentionReconstructor(hidden=args.score_hidden, depth=args.score_depth).to(device)
    print(f"model params: {sum(p.numel() for p in model.parameters()):,}")

    query_t = to_tensor(trunk_norm, device)     # [n_sensors, 2] -- query = all sensor positions
    ocean_t = torch.tensor(ocean_mask, dtype=torch.bool, device=device)
    clim_t = to_tensor(clim_norm, device)
    opt = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    best_val, best_state, bad = float("inf"), None, 0

    def sample_mask(coverage):
        n_obs = max(1, int(n_ocean * coverage))
        obs_idx = np.random.choice(ocean_idx, size=n_obs, replace=False)
        obs_coords = trunk_norm[obs_idx]  # [n_obs, 2]
        return obs_idx, to_tensor(obs_coords, device)

    ocean_query_idx = np.where(ocean_mask)[0]

    def subsample_query(n):
        qi = np.random.choice(ocean_query_idx, size=min(n, len(ocean_query_idx)), replace=False)
        return qi, query_t[qi]

    for step in range(1, args.iterations + 1):
        model.train()
        idx = np.random.randint(0, len(train_idx), size=args.batch_size)
        rows = zos_train[idx]
        obs_idx, obs_coords_t = sample_mask(args.coverage)
        obs_vals = to_tensor(rows[:, obs_idx], device)
        qi, query_sub = subsample_query(args.query_subsample)
        target_sub = to_tensor(rows[:, qi], device)
        clim_sub = clim_t[qi].unsqueeze(0)
        pred = model(query_sub, obs_coords_t, obs_vals, clim_sub)
        loss = ((pred - target_sub) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()

        if step % 100 == 0 or step == args.iterations:
            model.eval()
            with torch.no_grad():
                vobs_idx, vobs_coords_t = sample_mask(args.coverage)
                vobs_vals = to_tensor(zos_val[:, vobs_idx], device)
                vqi, vquery_sub = subsample_query(args.query_subsample)
                vtarget = to_tensor(zos_val[:, vqi], device)
                vclim_sub = clim_t[vqi].unsqueeze(0)
                vpred = model(vquery_sub, vobs_coords_t, vobs_vals, vclim_sub)
                vloss = ((vpred - vtarget) ** 2).mean().item()
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
    obs_idx, obs_coords_t = sample_mask(args.coverage)
    obs_vals = to_tensor(zos_test[:, obs_idx], device)
    with torch.no_grad():
        pred_test = model(query_t, obs_coords_t, obs_vals, clim_t.unsqueeze(0)).cpu().numpy()

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

    print(f"\n=== Geometric-attention sparse reconstruction (coverage={args.coverage:.0%}) ===")
    print(f"  n_obs points used: {len(obs_idx)} / {n_ocean} ocean cells")
    print(f"  DeepONet (geometric attention): rmse(norm)={rmse_model:.4f}  skill vs climatology={sk_model:+.4f}")
    print(f"  Naive scattered interpolation:  rmse(norm)={rmse_naive:.4f}  skill vs climatology={sk_naive:+.4f}")


if __name__ == "__main__":
    main()
