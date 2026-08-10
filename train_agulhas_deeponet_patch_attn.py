#!/usr/bin/env python3
"""Patch-based DeepONet + cross-tile self-attention — exploratory extension.

Motivation: the plain patch model (train_agulhas_deeponet_patch.py) fixed the r=3
parameter blowup by applying one small shared DeepONet independently to every
overlapping tile, but each tile is processed in total isolation — a tile near the
Agulhas retroflection has no way to "see" what a neighboring tile is doing except
through weights baked in at training time. This is the same limitation Vision
Transformers solve for image patches: chop the input into patches, then add
self-attention BETWEEN patch embeddings so each patch's representation can be
informed by every other patch, before decoding back to pixels.

Architecture (built on the same MultivarDeepONet building blocks):
  1. Per tile, per variable: the SAME small branch MLP (shared across tiles, as
     before) maps the tile's (possibly multi-day-history) local state to a latent
     vector -- this is exactly train_agulhas_deeponet_patch.py's branch step.
  2. A learned tile-position embedding (from the tile's center lon/lat) is added
     to each tile's latent vector, so attention can tell tiles apart by location
     (the same role positional embeddings play in a ViT).
  3. ONE shared multi-head self-attention layer (shared across variables, like the
     shared trunk) lets every tile's latent vector attend to every other tile's,
     for that day -- this is the new cross-tile communication channel.
  4. The trunk (shared, same as before) still maps each tile's own query
     coordinates to basis functions; the post-attention latent is dot-producted
     with the tile's trunk features, plus the persistence residual, exactly as in
     the plain patch model.
  Attention's output projection is zero-initialized, so at step 0 attention
  contributes nothing extra and the model starts at the same persistence-residual
  init as every other model in this study.

Training-loop consequence: because attention needs multiple tiles' tokens at
once, each training step now processes ALL tiles for a batch of days together
(not one random tile per step, as the plain patch script does).

This is a LOCAL, SMALL-DATA exploratory script (see train_agulhas_deeponet_patch.py's
header for the same caveat) -- a first test of whether cross-tile attention adds
anything on top of the (already-validated) plain patch architecture.

Example:
    python3 train_agulhas_deeponet_patch_attn.py --nc data/agulhas_prototype.nc \
        --subsample-r 6 --patch-h 20 --patch-w 20 --patch-stride 10 \
        --n-history-days 2 --iterations 3000 --batch-size 8 --out-dir results/patch_attn
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from train_agulhas_deeponet_prototype import (
    VARIABLES, load_states, build_dataset, anomaly_correlation, to_tensor,
    save_json, _make_mlp,
)
from train_agulhas_deeponet_patch import enumerate_tiles, tile_indices, history_block


class AttentivePatchDeepONet(nn.Module):
    """Shared per-tile branch/trunk (as in MultivarDeepONet), plus one shared
    cross-tile self-attention layer operating on the branch's per-tile latent
    vectors before the trunk dot-product.

    forward(branch_bt, trunk_per_tile, tile_pos):
      branch_bt       : [B, T, D_tile]        branch input, per (day, tile)
      trunk_per_tile  : [T, n_sensors, 2]     per-tile query coords (normalised)
      tile_pos        : [T, 2]                tile-center coords (normalised)
      returns         : [B, T, n_sensors, n_vars]
    """

    def __init__(self, d_branch_tile, n_sensors_tile, n_vars,
                 branch_width, branch_depth, trunk_width, trunk_depth,
                 latent_dim, attn_heads=4, activation="tanh", residual=True):
        super().__init__()
        self.n_vars = n_vars
        self.n_sensors = n_sensors_tile
        self.latent_dim = latent_dim
        self.residual = residual

        branch_sizes = [d_branch_tile] + [branch_width] * branch_depth + [latent_dim]
        trunk_sizes = [2] + [trunk_width] * trunk_depth + [latent_dim]
        self.trunk = _make_mlp(trunk_sizes, activation)
        self.branches = nn.ModuleList([_make_mlp(branch_sizes, activation) for _ in range(n_vars)])
        self.biases = nn.Parameter(torch.zeros(n_vars))

        self.pos_mlp = _make_mlp([2, 32, latent_dim], activation)
        self.attn = nn.MultiheadAttention(embed_dim=latent_dim, num_heads=attn_heads, batch_first=True)
        self.attn_norm = nn.LayerNorm(latent_dim)
        # Zero the attention output projection AND the position-embedding MLP's
        # last layer: at init, attention contributes exactly 0 AND pos_emb is
        # exactly 0, so branch_feats (already 0 from the branches' own zero-init
        # below) stays exactly 0 into the trunk dot-product -- the model starts
        # at the same exact-persistence state as every other model in this study
        # (see MultivarDeepONet's own comment). Without this, pos_emb alone
        # injects nonzero noise at step 0 even though everything else is zeroed.
        nn.init.zeros_(self.attn.out_proj.weight)
        nn.init.zeros_(self.attn.out_proj.bias)
        pos_last_linear = [m for m in self.pos_mlp.modules() if isinstance(m, nn.Linear)][-1]
        nn.init.zeros_(pos_last_linear.weight)
        nn.init.zeros_(pos_last_linear.bias)

        if residual:
            for branch in self.branches:
                last_linear = None
                for layer in branch.modules():
                    if isinstance(layer, nn.Linear):
                        last_linear = layer
                if last_linear is not None:
                    nn.init.zeros_(last_linear.weight)
                    nn.init.zeros_(last_linear.bias)

    def forward(self, branch_bt, trunk_per_tile, tile_pos, attn_mask=None):
        B, T, D = branch_bt.shape
        pos_emb = self.pos_mlp(tile_pos)  # [T, latent_dim]
        trunk_feats = self.trunk(trunk_per_tile.reshape(-1, 2)).reshape(T, self.n_sensors, self.latent_dim)

        outputs = []
        for vi in range(self.n_vars):
            feats = self.branches[vi](branch_bt.reshape(B * T, D)).reshape(B, T, self.latent_dim)
            feats = feats + pos_emb.unsqueeze(0)
            # self-attention across the T (tile) dimension; attn_mask (if given) restricts
            # each tile to only attend to its K geographically-nearest neighbors (Swin-style
            # windowed attention) instead of every other tile globally.
            attn_out, _ = self.attn(feats, feats, feats, attn_mask=attn_mask)
            feats_ctx = feats + self.attn_norm(attn_out)  # residual around attention

            out = torch.einsum("btp,tsp->bts", feats_ctx, trunk_feats) + self.biases[vi]
            if self.residual:
                cur_block = branch_bt[:, :, vi * self.n_sensors:(vi + 1) * self.n_sensors]
                out = out + cur_block
            outputs.append(out)
        return torch.stack(outputs, dim=-1)  # [B, T, n_sensors, n_vars]


def parse_args():
    p = argparse.ArgumentParser(description="Patch DeepONet + cross-tile attention (exploratory).")
    p.add_argument("--nc", type=Path, default=Path("data/agulhas_prototype.nc"))
    p.add_argument("--cache", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, default=Path("results/patch_attn"))
    p.add_argument("--subsample-r", type=int, default=6)
    p.add_argument("--val-fraction", type=float, default=0.15)
    p.add_argument("--test-fraction", type=float, default=0.15)
    p.add_argument("--patch-h", type=int, default=20)
    p.add_argument("--patch-w", type=int, default=20)
    p.add_argument("--patch-stride", type=int, default=10)
    p.add_argument("--latent-dim", type=int, default=32)
    p.add_argument("--branch-width", type=int, default=64)
    p.add_argument("--trunk-width", type=int, default=64)
    p.add_argument("--branch-depth", type=int, default=2)
    p.add_argument("--trunk-depth", type=int, default=2)
    p.add_argument("--attn-heads", type=int, default=4)
    p.add_argument("--attn-k-neighbors", type=int, default=0,
                   help="If >0, restrict attention to each tile's K geographically-"
                        "nearest neighbors (Swin-style windowed attention) instead of "
                        "attending globally to all tiles (0 = global attention, the "
                        "original design).")
    p.add_argument("--n-history-days", type=int, default=1)
    p.add_argument("--iterations", type=int, default=3000)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--batch-size", type=int, default=8, help="Days per step (all tiles processed together).")
    p.add_argument("--display-every", type=int, default=200)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--seed", type=int, default=2026)
    return p.parse_args()


def masked_mse(pred, target, ocean_mask_vm_bt):
    return torch.mean((pred[ocean_mask_vm_bt] - target[ocean_mask_vm_bt]) ** 2)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device: {device}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    lon_sub, lat_sub, states = load_states(args.nc, args.subsample_r, args.cache)
    ds = build_dataset(states, lon_sub, lat_sub, args.test_fraction, args.val_fraction)
    nlat_s, nlon_s = ds["nlat_s"], ds["nlon_s"]
    n_sensors, n_vars = ds["n_sensors"], ds["n_vars"]
    print(f"Full grid: {nlat_s} x {nlon_s} = {n_sensors} sensors, r={args.subsample_r}")
    print(f"Train/val/test days: {len(ds['train_idx'])}/{len(ds['val_idx'])}/{len(ds['test_idx'])}")

    ocean_mask = ds["ocean_mask"]

    all_tiles = enumerate_tiles(nlat_s, nlon_s, args.patch_h, args.patch_w,
                                args.patch_stride, args.patch_stride)
    th, tw = min(args.patch_h, nlat_s), min(args.patch_w, nlon_s)
    all_tile_meta = [tile_indices(t, nlat_s, nlon_s, n_vars, n_sensors, args.n_history_days)
                      for t in all_tiles]
    keep = [i for i, (sidx, _, _) in enumerate(all_tile_meta) if ocean_mask[sidx].any()]
    tile_meta = [all_tile_meta[i] for i in keep]
    n_dropped = len(all_tiles) - len(tile_meta)
    if n_dropped:
        print(f"  Dropped {n_dropped}/{len(all_tiles)} all-land tiles")
    T = len(tile_meta)
    print(f"Tiling: {T} tiles of {th}x{tw} sensors (stride {args.patch_stride}), "
          f"covering {nlat_s}x{nlon_s} -- ALL {T} tiles processed together each step (attention)")

    n_sensors_tile = th * tw
    d_branch_tile = args.n_history_days * n_vars * n_sensors_tile

    model = AttentivePatchDeepONet(
        d_branch_tile=d_branch_tile, n_sensors_tile=n_sensors_tile, n_vars=n_vars,
        branch_width=args.branch_width, branch_depth=args.branch_depth,
        trunk_width=args.trunk_width, trunk_depth=args.trunk_depth,
        latent_dim=args.latent_dim, attn_heads=args.attn_heads,
        activation="tanh", residual=True,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters (patch + cross-tile attention): {n_params:,}")

    # Precompute per-tile index tensors, shared across all steps.
    sensor_idx_list = [tm[0] for tm in tile_meta]
    target_cols_all = np.stack([tm[1] for tm in tile_meta])   # [T, n_vars*n_sensors_tile]
    branch_cols_all = np.stack([tm[2] for tm in tile_meta])   # [T, D_tile]
    target_cols_t = torch.as_tensor(target_cols_all, dtype=torch.long, device=device)
    branch_cols_t = torch.as_tensor(branch_cols_all, dtype=torch.long, device=device)
    ocean_mask_bt = torch.as_tensor(
        np.stack([ocean_mask[si] for si in sensor_idx_list]), dtype=torch.bool, device=device
    )  # [T, n_sensors_tile]
    ocean_mask_bt_vm = ocean_mask_bt.repeat(1, n_vars)  # [T, n_vars*n_sensors_tile] matches target_cols layout...

    branch_train_full = np.concatenate(
        [ds["branch_train"], history_block(states, ds["train_idx"], ds, args.n_history_days)], axis=1)
    branch_val_full = np.concatenate(
        [ds["branch_val"], history_block(states, ds["val_idx"], ds, args.n_history_days)], axis=1)
    branch_test_full = np.concatenate(
        [ds["branch_test"], history_block(states, ds["test_idx"], ds, args.n_history_days)], axis=1)

    branch_train = to_tensor(branch_train_full, device)
    y_train_norm = to_tensor(ds["y_train_norm"], device)
    branch_val   = to_tensor(branch_val_full, device)
    y_val_norm   = to_tensor(ds["y_val_norm"], device)
    trunk_full   = to_tensor(ds["trunk"], device)  # [n_sensors, 2]
    n_train = branch_train.shape[0]

    # Per-tile trunk coords [T, n_sensors_tile, 2], and tile-center position [T, 2]
    trunk_per_tile = torch.stack([trunk_full[torch.as_tensor(si, dtype=torch.long, device=device)]
                                   for si in sensor_idx_list])  # [T, n_sensors_tile, 2]
    tile_pos = trunk_per_tile.mean(dim=1)  # [T, 2] tile-center in normalised coords

    attn_mask = None
    if args.attn_k_neighbors > 0:
        k = min(args.attn_k_neighbors, T - 1)
        with torch.no_grad():
            dists = torch.cdist(tile_pos, tile_pos)  # [T, T] tile-center distances
            # allowed = self + K nearest neighbors; block (True) everything else
            nearest = torch.topk(dists, k=k + 1, largest=False).indices  # [T, k+1], incl. self (dist 0)
            attn_mask = torch.ones(T, T, dtype=torch.bool, device=device)
            attn_mask.scatter_(1, nearest, False)
        print(f"Windowed attention: each tile attends to its {k} nearest neighbors + itself "
              f"(of {T} total tiles)")
    else:
        print("Global attention: each tile attends to all tiles")

    def gather_tile_batch(branch_all, target_all, day_idx_t):
        # branch_all/target_all: [N, D_full] tensors; day_idx_t: [B] long tensor
        b = branch_all[day_idx_t][:, branch_cols_t]   # [B, T, D_tile]  (2D fancy index broadcasts)
        y = target_all[day_idx_t][:, target_cols_t]   # [B, T, n_vars*n_sensors_tile]
        return b, y

    opt = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    best_val, best_state, since_best = float("inf"), None, 0
    t0 = time.time()
    for step in range(1, args.iterations + 1):
        model.train()
        day_idx = np.random.randint(n_train, size=min(args.batch_size, n_train))
        day_idx_t = torch.as_tensor(day_idx, dtype=torch.long, device=device)
        b, y = gather_tile_batch(branch_train, y_train_norm, day_idx_t)  # [B,T,D_tile], [B,T,n_vars*n_sensors_tile]

        pred = model(b, trunk_per_tile, tile_pos, attn_mask)  # [B, T, n_sensors_tile, n_vars]
        pred_vm = pred.permute(0, 1, 3, 2).reshape(pred.shape[0], T, -1)  # [B, T, n_vars*n_sensors_tile] var-major
        mask_bt = ocean_mask_bt_vm.unsqueeze(0).expand(pred_vm.shape[0], -1, -1)
        loss = masked_mse(pred_vm, y, mask_bt)

        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % args.display_every == 0 or step == 1:
            model.eval()
            with torch.no_grad():
                val_day_idx_t = torch.arange(branch_val.shape[0], device=device)
                bv, yv = gather_tile_batch(branch_val, y_val_norm, val_day_idx_t)
                pv = model(bv, trunk_per_tile, tile_pos, attn_mask).permute(0, 1, 3, 2).reshape(bv.shape[0], T, -1)
                mv = ocean_mask_bt_vm.unsqueeze(0).expand(pv.shape[0], -1, -1)
                val_loss = masked_mse(pv, yv, mv).item()
            elapsed = time.time() - t0
            print(f"step {step:6d}  train_loss {loss.item():.5f}  val_loss {val_loss:.5f}  ({elapsed:.0f}s)")
            if val_loss < best_val - 1e-6:
                best_val, since_best = val_loss, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                since_best += 1
                if args.patience > 0 and since_best >= args.patience:
                    print(f"Early stopping at step {step} (best val_loss={best_val:.5f})")
                    break

    if best_state is not None:
        model.load_state_dict(best_state)
    steps_run = step

    # ── Test evaluation ──────────────────────────────────────────────────────
    branch_test = to_tensor(branch_test_full, device)
    N_test = branch_test.shape[0]
    out_mean, out_std = ds["out_mean"], ds["out_std"]

    model.eval()
    accum = np.zeros((N_test, n_sensors, n_vars), dtype=np.float64)
    counts = np.zeros((n_sensors,), dtype=np.float64)
    with torch.no_grad():
        test_day_idx_t = torch.arange(N_test, device=device)
        bt = branch_test[test_day_idx_t][:, branch_cols_t]  # [N_test, T, D_tile]
        pred_norm = model(bt, trunk_per_tile, tile_pos, attn_mask).cpu().numpy()  # [N_test, T, n_sensors_tile, n_vars]
    for ti, si in enumerate(sensor_idx_list):
        pred_raw = pred_norm[:, ti].copy()  # [N_test, n_sensors_tile, n_vars]
        for vi in range(n_vars):
            pred_raw[:, :, vi] = pred_norm[:, ti, :, vi] * out_std[vi] + out_mean[vi]
        accum[:, si, :] += pred_raw
        counts[si] += 1
    assert (counts[ocean_mask] > 0).all(), "tiling left uncovered OCEAN sensors"
    safe_counts = np.where(counts > 0, counts, 1.0)
    pred_3d = accum / safe_counts[None, :, None]

    true_3d = ds["y_test_raw"].reshape(N_test, n_vars, n_sensors).transpose(0, 2, 1)
    persist_flat = ds["x_test_raw"]
    climatology = ds["climatology"]

    metrics = {
        "n_params": int(n_params), "n_tiles": T,
        "patch_h": th, "patch_w": tw, "patch_stride": args.patch_stride,
        "subsample_r": args.subsample_r, "n_history_days": args.n_history_days,
        "attn_heads": args.attn_heads,
        "steps_run": int(steps_run), "best_val_loss": float(best_val),
        "n_train": int(len(ds["train_idx"])), "n_val": int(len(ds["val_idx"])),
        "n_test": int(len(ds["test_idx"])),
    }
    for vi, vname in enumerate(VARIABLES):
        col_s, col_e = vi * n_sensors, (vi + 1) * n_sensors
        true_vi = true_3d[:, ocean_mask, vi]
        pred_vi = pred_3d[:, ocean_mask, vi]
        pers_vi = persist_flat[:, col_s:col_e][:, ocean_mask]
        clim_vi = climatology[ocean_mask, vi]

        rmse_m = float(np.sqrt(np.mean((true_vi - pred_vi) ** 2)))
        rmse_p = float(np.sqrt(np.mean((true_vi - pers_vi) ** 2)))
        std_true = float(np.std(true_vi))
        skill = float(1.0 - (rmse_m / rmse_p) ** 2) if rmse_p > 1e-12 else float("nan")
        nrmse = float(rmse_m / std_true) if std_true > 1e-12 else float("nan")
        bias = float(np.mean(pred_vi - true_vi))
        acc = anomaly_correlation(pred_vi, true_vi, clim_vi)

        metrics[f"rmse_{vname}"] = rmse_m
        metrics[f"rmse_persist_{vname}"] = rmse_p
        metrics[f"skill_{vname}"] = skill
        metrics[f"nrmse_{vname}"] = nrmse
        metrics[f"bias_{vname}"] = bias
        metrics[f"acc_{vname}"] = acc

    mean_skill = float(np.mean([metrics[f"skill_{v}"] for v in VARIABLES]))
    metrics["mean_skill"] = mean_skill

    print("\n=== Patch + cross-tile attention — test-set summary ===")
    print(f"tiles={T} ({th}x{tw})  params={n_params:,}  history_days={args.n_history_days}")
    print(f"mean skill = {mean_skill:+.4f}")
    for vname in VARIABLES:
        print(f"  {vname:8s} skill={metrics[f'skill_{vname}']:+.4f}  "
              f"acc={metrics[f'acc_{vname}']:.3f}  rmse={metrics[f'rmse_{vname}']:.4f}")

    save_json(args.out_dir / "metrics.json", metrics)
    torch.save(model.state_dict(), args.out_dir / "model.pt")
    print(f"\nSaved to {args.out_dir}/")


if __name__ == "__main__":
    main()
