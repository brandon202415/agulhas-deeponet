#!/usr/bin/env python3
"""Patch DeepONet + CNN branch: combines the two locality fixes tested
separately so far, rather than either alone.

Motivation: the CNN-branch whole-domain DeepONet (train_agulhas_deeponet_
cnnbranch.py) gave a real, if partial, local improvement by adding conv
structure to the branch -- but its branch still ends in a full global-average
pool over the ENTIRE domain (61x101), collapsing all spatial specificity to
one vector before the trunk ever sees it. That is a structural mismatch for
dense, spatially-varying prediction (a U-Net's skip connections preserve
per-location detail all the way to the output; a global-pooled DeepONet
branch structurally cannot). Patch-DeepONet (train_agulhas_deeponet_patch.py)
already independently shows tiling helps (+0.075 vs. +0.043 whole-domain,
Table 7) even with a plain dense-MLP branch and no conv structure at all --
plausibly because limiting the branch's summarization region to a small tile
is itself a crude way of preserving locality, quite apart from the tiling-as-
data-augmentation confound already flagged (MANUSCRIPT_ISSUES.md Issue 2).

This script tests both mechanisms together: CNN encoder for LOCAL spatial
structure (small conv kernels) + tiling for LOCAL pooling region (global-
average-pooling within a 20x20 tile discards far less than pooling over the
whole 61x101 domain). Neither fix alone directly addresses the other's
mechanism, so this is not redundant with either prior experiment.

Design: `CNNBranchDeepONet` (imported unchanged from
train_agulhas_deeponet_cnnbranch.py) is reused AS THE SHARED PER-TILE MODEL --
its constructor already takes nlat/nlon/n_sensors as parameters rather than
hardcoding whole-domain size, so instantiating it at tile size (e.g. 20x20)
makes it directly usable as the patch trainer's shared model with zero
changes to that class. Tiling/sampling infra (enumerate_tiles, tile_indices,
masked_mse) reused unchanged from train_agulhas_deeponet_patch.py. Training
loop, single-tile-per-step sampling, and overlap-averaged test-time
reconstruction mirror the base patch trainer exactly -- the only thing that
differs from train_agulhas_deeponet_patch.py is which model class is shared
across tiles (CNNBranchDeepONet instead of the base MultivarDeepONet).

No multi-day input history in this variant (out of scope for a first test of
whether this combination helps at all; the base patch trainer's history
option is a separate, already-validated enhancement that could be added
later if this shows promise).

Usage:
    python3 train_agulhas_deeponet_patch_cnnbranch.py --nc data/agulhas_prototype.nc \\
        --cache data/cache_r6_local.npz --subsample-r 6 --patch-h 20 --patch-w 20 \\
        --patch-stride 10 --iterations 3000 --out-dir results/patch_cnnbranch_r6_local
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch

from train_agulhas_deeponet_prototype import (
    VARIABLES, load_states, build_dataset, anomaly_correlation, to_tensor, save_json,
)
from train_agulhas_deeponet_patch import enumerate_tiles, tile_indices, masked_mse
from train_agulhas_deeponet_cnnbranch import CNNBranchDeepONet


def parse_args():
    p = argparse.ArgumentParser(description="Patch DeepONet with a CNN-branch shared model per tile.")
    p.add_argument("--nc", type=Path, default=Path("data/agulhas_prototype.nc"))
    p.add_argument("--cache", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, default=Path("results/patch_cnnbranch_prototype"))
    p.add_argument("--subsample-r", type=int, default=6)
    p.add_argument("--val-fraction", type=float, default=0.15)
    p.add_argument("--test-fraction", type=float, default=0.15)
    p.add_argument("--patch-h", type=int, default=20)
    p.add_argument("--patch-w", type=int, default=20)
    p.add_argument("--patch-stride", type=int, default=10)
    # CNN encoder (per tile)
    p.add_argument("--base-width", type=int, default=24)
    p.add_argument("--hidden-dim", type=int, default=128,
                    help="Smaller default than the whole-domain CNN-branch script (256) "
                         "since a tile has far less to summarize than the whole domain.")
    # Trunk
    p.add_argument("--trunk-width", type=int, default=64)
    p.add_argument("--trunk-depth", type=int, default=2)
    p.add_argument("--latent-dim", type=int, default=32)
    # Training
    p.add_argument("--iterations", type=int, default=3000)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--batch-size", type=int, default=64, help="Days per step.")
    p.add_argument("--display-every", type=int, default=200)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--seed", type=int, default=2026)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    # No MPS branch -- see train_agulhas_deeponet_cnnbranch.py's note; this
    # model shares the same conv-encoder op combination that stalls on Metal
    # locally, and the real cluster run always has CUDA regardless.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    all_tile_meta = [tile_indices(t, nlat_s, nlon_s, n_vars, n_sensors, n_history_days=1)
                      for t in all_tiles]
    keep = [i for i, (sidx, _, _) in enumerate(all_tile_meta) if ocean_mask[sidx].any()]
    tiles = [all_tiles[i] for i in keep]
    tile_meta = [all_tile_meta[i] for i in keep]
    if len(tiles) < len(all_tiles):
        print(f"  Dropped {len(all_tiles) - len(tiles)}/{len(all_tiles)} all-land tiles")
    print(f"Tiling: {len(tiles)} tiles of {th}x{tw} sensors (stride {args.patch_stride})")

    # ── Model: ONE shared CNNBranchDeepONet, sized for a single tile ────────
    n_sensors_tile = th * tw
    model = CNNBranchDeepONet(
        n_vars=n_vars, nlat=th, nlon=tw, n_sensors=n_sensors_tile,
        base_width=args.base_width, hidden_dim=args.hidden_dim,
        trunk_width=args.trunk_width, trunk_depth=args.trunk_depth,
        latent_dim=args.latent_dim, activation="tanh",
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters (shared, per-tile, CNN branch): {n_params:,}  "
          f"(cf. plain patch DeepONet 965,862; whole-domain CNN-branch 243,622)")

    branch_train = to_tensor(ds["branch_train"], device)
    y_train_norm = to_tensor(ds["y_train_norm"], device)
    branch_val   = to_tensor(ds["branch_val"], device)
    y_val_norm   = to_tensor(ds["y_val_norm"], device)
    trunk_full   = to_tensor(ds["trunk"], device)
    n_train = branch_train.shape[0]

    opt = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    best_val, best_state, since_best = float("inf"), None, 0
    t0 = time.time()
    step = 0
    for step in range(1, args.iterations + 1):
        model.train()
        tile_i = np.random.randint(len(tiles))
        sensor_idx, target_cols, branch_cols = tile_meta[tile_i]
        sensor_idx_t = torch.as_tensor(sensor_idx, dtype=torch.long, device=device)
        target_cols_t = torch.as_tensor(target_cols, dtype=torch.long, device=device)
        branch_cols_t = torch.as_tensor(branch_cols, dtype=torch.long, device=device)
        day_idx = np.random.randint(n_train, size=min(args.batch_size, n_train))
        day_idx_t = torch.as_tensor(day_idx, dtype=torch.long, device=device)

        b = branch_train[day_idx_t][:, branch_cols_t]
        y = y_train_norm[day_idx_t][:, target_cols_t]
        trunk_tile = trunk_full[sensor_idx_t]
        tile_ocean_vm = torch.as_tensor(
            np.tile(ocean_mask[sensor_idx], n_vars), dtype=torch.bool, device=device
        )

        pred = model(b, trunk_tile)  # [B, th*tw, n_vars]
        pred_vm = pred.permute(0, 2, 1).reshape(pred.shape[0], -1)
        loss = masked_mse(pred_vm, y, tile_ocean_vm)

        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % args.display_every == 0 or step == 1:
            model.eval()
            with torch.no_grad():
                val_losses = []
                for sensor_idx_v, target_cols_v, branch_cols_v in tile_meta:
                    sidx = torch.as_tensor(sensor_idx_v, dtype=torch.long, device=device)
                    bcidx = torch.as_tensor(branch_cols_v, dtype=torch.long, device=device)
                    tcidx = torch.as_tensor(target_cols_v, dtype=torch.long, device=device)
                    bv = branch_val[:, bcidx]
                    yv = y_val_norm[:, tcidx]
                    tv = trunk_full[sidx]
                    pv = model(bv, tv).permute(0, 2, 1).reshape(bv.shape[0], -1)
                    tvm = torch.as_tensor(
                        np.tile(ocean_mask[sensor_idx_v], n_vars), dtype=torch.bool, device=device
                    )
                    val_losses.append(masked_mse(pv, yv, tvm).item())
                val_loss = float(np.mean(val_losses))
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

    # ── Test evaluation: predict every tile, overlap-average onto full grid ──
    branch_test = to_tensor(ds["branch_test"], device)
    N_test = branch_test.shape[0]
    out_mean, out_std = ds["out_mean"], ds["out_std"]

    accum = np.zeros((N_test, n_sensors, n_vars), dtype=np.float64)
    counts = np.zeros((n_sensors,), dtype=np.float64)
    model.eval()
    with torch.no_grad():
        for sensor_idx_v, target_cols_v, branch_cols_v in tile_meta:
            sidx = torch.as_tensor(sensor_idx_v, dtype=torch.long, device=device)
            bcidx = torch.as_tensor(branch_cols_v, dtype=torch.long, device=device)
            b = branch_test[:, bcidx]
            tv = trunk_full[sidx]
            pred_norm = model(b, tv).cpu().numpy()
            pred_raw = pred_norm.copy()
            for vi in range(n_vars):
                pred_raw[:, :, vi] = pred_norm[:, :, vi] * out_std[vi] + out_mean[vi]
            accum[:, sensor_idx_v, :] += pred_raw
            counts[sensor_idx_v] += 1
    assert (counts[ocean_mask] > 0).all(), "tiling left uncovered OCEAN sensors"
    safe_counts = np.where(counts > 0, counts, 1.0)
    pred_3d = accum / safe_counts[None, :, None]

    true_3d = ds["y_test_raw"].reshape(N_test, n_vars, n_sensors).transpose(0, 2, 1)
    persist_flat = ds["x_test_raw"]
    climatology = ds["climatology"]

    pred_flat = pred_3d.transpose(0, 2, 1).reshape(N_test, -1)
    true_flat_vm = true_3d.transpose(0, 2, 1).reshape(N_test, -1)
    np.savez_compressed(
        args.out_dir / "predictions.npz",
        y_true=true_flat_vm, y_pred=pred_flat, y_persist=persist_flat,
        lon=lon_sub, lat=lat_sub, test_indices=ds["test_idx"],
    )

    metrics = {
        "n_params": int(n_params), "n_tiles": len(tiles),
        "patch_h": th, "patch_w": tw, "patch_stride": args.patch_stride,
        "subsample_r": args.subsample_r, "steps_run": int(steps_run),
        "best_val_loss": float(best_val),
        "n_train": int(len(ds["train_idx"])), "n_val": int(len(ds["val_idx"])),
        "n_test": int(len(ds["test_idx"])), "seed": args.seed,
        "learning_rate": args.learning_rate,
        "base_width": args.base_width, "hidden_dim": args.hidden_dim,
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

    print("\n=== Patch DeepONet + CNN branch -- test-set summary ===")
    print(f"tiles={len(tiles)} ({th}x{tw}, stride {args.patch_stride})  params={n_params:,}")
    print(f"mean skill = {mean_skill:+.4f}")
    for vname in VARIABLES:
        print(f"  {vname:8s} skill={metrics[f'skill_{vname}']:+.4f}  "
              f"acc={metrics[f'acc_{vname}']:.3f}  rmse={metrics[f'rmse_{vname}']:.4f}")

    save_json(args.out_dir / "metrics.json", metrics)
    torch.save(model.state_dict(), args.out_dir / "model.pt")
    print(f"\nSaved to {args.out_dir}/")


if __name__ == "__main__":
    main()
