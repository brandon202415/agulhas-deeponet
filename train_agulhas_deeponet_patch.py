#!/usr/bin/env python3
"""Patch-based DeepONet — exploratory prototype for the r=3 resolution scaling fix.

Motivation (see RESULTS.md / manuscript Sec. 3.4, 3.5, 4.8, 5): the main trainer's
`MultivarDeepONet` branch reads the ENTIRE flattened sensor grid, so finer
resolution multiplies both branch input size and parameter count while the
training-set size stays fixed — at r=3 this produced a ~56M-parameter, underfit
model (mean skill collapsed to ~0). Both the original proposal's "Potential
Pitfalls" section and this study's own Limitations identify a **patch-based**
architecture — divide the domain into overlapping spatial tiles, apply ONE
shared small DeepONet independently to each tile, and reassemble the full grid
by averaging predictions in overlapping regions — as the documented remedy. This
script is a first, local, exploratory implementation and test of that idea.

Design (deliberately minimal, reusing the main trainer's tested building blocks):
  - Reuses `load_states`/`build_dataset`/`MultivarDeepONet`/`anomaly_correlation`
    UNCHANGED from train_agulhas_deeponet_prototype.py. `build_dataset` still
    normalises and splits over the WHOLE domain (so stats match the main study
    exactly); the only new thing is *what the branch/trunk see* per training/eval
    step: a fixed-size tile's sensors, not the whole grid.
  - One `MultivarDeepONet` instance, sized for a SINGLE tile (n_sensors = tile_h *
    tile_w), is shared across every tile and the whole domain — parameter count is
    therefore independent of the subsampling factor r, which is exactly the
    scaling failure this is meant to fix.
  - Training samples a random tile + a random minibatch of days at every step.
  - Test-time reconstruction predicts every tile and averages overlapping
    predictions (uniform weight) back onto the full grid, then computes the SAME
    per-variable RMSE/skill/NRMSE/ACC/bias metrics as the main trainer, so results
    are directly comparable to Table 1 / RESULTS.md.
  - Physics losses are NOT included here (kept out deliberately — the main study
    already found them inert, and FD gradients across arbitrary tile boundaries
    would need extra care that isn't central to the resolution question this
    script targets).

This is a LOCAL, SMALL-DATA exploratory script, not a replacement for a real
cluster run: see the printed summary and eddy_tracking-style README-in-comments
below for what it does and does not establish.

Example (local prototype data, r=3, comparable tile size to the r=6 whole-domain
model):
    python3 train_agulhas_deeponet_patch.py --nc data/agulhas_prototype.nc \
        --subsample-r 3 --patch-h 20 --patch-w 20 --patch-stride 10 \
        --iterations 3000 --batch-size 64 --out-dir results/patch_r3
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch

from train_agulhas_deeponet_prototype import (
    VARIABLES, load_states, build_dataset, MultivarDeepONet,
    anomaly_correlation, to_tensor, save_json, _var_major_flat,
)


def history_block(states, day_idx, ds, n_history_days):
    """Extra branch-input blocks for days t-1, t-2, ... (t-n_history_days+1),
    normalised/land-zeroed the SAME way as the current-day branch (same
    b_mean/b_std/ocean_mask), so they live in the same space and can be
    concatenated directly. day_idx are the states-cube row indices for the
    "current" day of each sample (train_idx/val_idx/test_idx double as this,
    since build_dataset's branch_inputs = flat[:-k] is states[0:N] row-aligned).
    Clipped at the start of the record (no earlier day => duplicate the earliest
    available one; still causal, just a repeated frame).
    """
    if n_history_days <= 1:
        return np.zeros((len(day_idx), 0), dtype=np.float32)
    T = states.shape[0]
    b_mean, b_std = ds["b_mean"], ds["b_std"]  # [1, D]
    land_vm = np.tile(~ds["ocean_mask"], ds["n_vars"])
    blocks = []
    for h in range(1, n_history_days):
        prev_idx = np.clip(day_idx - h, 0, T - 1)
        prev_flat = _var_major_flat(states[prev_idx].astype(np.float64))
        prev_norm = ((prev_flat - b_mean) / b_std).astype(np.float32)
        prev_norm[:, land_vm] = 0.0
        blocks.append(prev_norm)
    return np.concatenate(blocks, axis=1)


# ── Tiling ────────────────────────────────────────────────────────────────────

def enumerate_tiles(nlat, nlon, th, tw, stride_h, stride_w):
    """Overlapping tile top-left corners covering the full (nlat, nlon) grid.

    Guarantees full coverage (the last tile in each axis is shifted to align
    with the bottom/right edge exactly), standard sliding-window tiling.
    """
    th, tw = min(th, nlat), min(tw, nlon)

    def starts(total, size, stride):
        s = list(range(0, total - size + 1, max(1, stride)))
        if not s or s[-1] != total - size:
            s.append(total - size)
        return sorted(set(s))

    r_starts = starts(nlat, th, stride_h)
    c_starts = starts(nlon, tw, stride_w)
    return [(r0, r0 + th, c0, c0 + tw) for r0 in r_starts for c0 in c_starts]


def tile_indices(tile, nlat, nlon, n_vars, n_sensors, n_history_days=1):
    """For one (r0,r1,c0,c1) tile, return:
      sensor_idx  : [th*tw]             local grid-flat indices (row-major, matches
                                         the main trainer's reshape(nlat,nlon) convention)
      target_cols : [n_vars*th*tw]      columns into the (single-day) target/ocean-mask
                                         arrays (y_*_norm) — these are NEVER history-extended.
      branch_cols : [n_history_days*n_vars*th*tw]  columns into the (possibly multi-day-
                    concatenated) BRANCH INPUT array. Block 0 (first n_vars*th*tw entries)
                    is always the CURRENT day == target_cols, so MultivarDeepONet's
                    persistence residual — which only reads columns [0, n_vars*n_sensors) —
                    still works unmodified.
    """
    r0, r1, c0, c1 = tile
    grid_idx = np.arange(nlat * nlon).reshape(nlat, nlon)
    sensor_idx = grid_idx[r0:r1, c0:c1].ravel()
    target_cols = np.concatenate([vi * n_sensors + sensor_idx for vi in range(n_vars)])
    D_full = n_vars * n_sensors
    branch_cols = np.concatenate([target_cols + h * D_full for h in range(n_history_days)])
    return sensor_idx, target_cols, branch_cols


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Patch-based Agulhas DeepONet (exploratory).")
    p.add_argument("--nc", type=Path, default=Path("data/agulhas_prototype.nc"))
    p.add_argument("--cache", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, default=Path("results/patch_prototype"))
    p.add_argument("--subsample-r", type=int, default=6)
    p.add_argument("--val-fraction", type=float, default=0.15)
    p.add_argument("--test-fraction", type=float, default=0.15)
    # Tiling
    p.add_argument("--patch-h", type=int, default=20, help="Tile height in sensors.")
    p.add_argument("--patch-w", type=int, default=20, help="Tile width in sensors.")
    p.add_argument("--patch-stride", type=int, default=10,
                   help="Stride in sensors (< patch size => overlap).")
    # Network (defaults match the main trainer for a fair parameter-count comparison)
    p.add_argument("--latent-dim", type=int, default=32)
    p.add_argument("--branch-width", type=int, default=64)
    p.add_argument("--trunk-width", type=int, default=64)
    p.add_argument("--branch-depth", type=int, default=2)
    p.add_argument("--trunk-depth", type=int, default=2)
    # Training
    p.add_argument("--iterations", type=int, default=3000)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--batch-size", type=int, default=64, help="Days per step.")
    p.add_argument("--display-every", type=int, default=200)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--seed", type=int, default=2026)
    # Extra: multi-day input history and an LR scheduler
    p.add_argument("--n-history-days", type=int, default=1,
                   help="Number of consecutive days (t, t-1, ..., t-n+1) concatenated "
                        "into the branch input. 1 = current behaviour (single day).")
    p.add_argument("--lr-decay-factor", type=float, default=1.0,
                   help="ReduceLROnPlateau factor (1.0 = no scheduler / constant LR).")
    p.add_argument("--lr-decay-patience", type=int, default=8,
                   help="Evals (each --display-every steps) with no val improvement "
                        "before the LR is decayed.")
    # Gradient-aware loss: penalize wrong spatial gradients of a chosen variable
    # (default zos/SSH, the eddy-identification variable), not just point values.
    # Motivated by the eddy-tracking finding that RMSE-based skill gains don't
    # show up in eddy-center detection -- MSE alone doesn't reward the sharp
    # local structure (eddy cores/edges) a contour tracker depends on.
    p.add_argument("--grad-loss-weight", type=float, default=0.0,
                   help="Weight for a finite-difference gradient-matching loss on "
                        "--grad-loss-var (0 = off, the default/original behaviour).")
    p.add_argument("--grad-loss-var", type=str, default="zos",
                   help="Variable to apply the gradient loss to (default: zos/SSH, "
                        "the eddy-identification variable).")
    # Adversarial (LSGAN) loss: a small discriminator judges real vs. predicted
    # SSH tiles, trained alongside the usual masked-MSE objective. Motivated by
    # the same failure mode as --grad-loss-weight (MSE-trained regressors blur
    # sharp structure) but via a different, more standard mechanism from the
    # video-prediction/super-resolution literature: reward the generator for
    # producing tiles the discriminator cannot tell from real ocean fields,
    # rather than penalizing a specific hand-picked statistic (gradients).
    p.add_argument("--adv-weight", type=float, default=0.0,
                   help="Weight for the generator's adversarial loss term (0 = off, "
                        "the default/original behaviour; no discriminator is built).")
    p.add_argument("--adv-var", type=str, default="zos",
                   help="Variable the discriminator judges (default: zos/SSH).")
    p.add_argument("--disc-lr", type=float, default=1e-4,
                   help="Discriminator learning rate (LSGAN, Adam).")
    p.add_argument("--disc-width", type=int, default=128,
                   help="Discriminator MLP hidden width.")
    return p.parse_args()


class TileDiscriminator(torch.nn.Module):
    """Minimal LSGAN discriminator: judges one variable's tile (flattened,
    normalised units) as real (true ocean field) or fake (model prediction).
    A plain MLP, not a CNN, matching this codebase's existing preference for
    simple, small components over convolutional architectures -- a first,
    deliberately minimal test of whether adversarial training helps at all
    before investing in a more sophisticated discriminator.
    """

    def __init__(self, n_sensors_tile, width=128):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(n_sensors_tile, width), torch.nn.LeakyReLU(0.2),
            torch.nn.Linear(width, width), torch.nn.LeakyReLU(0.2),
            torch.nn.Linear(width, 1),
        )

    def forward(self, x):
        return self.net(x)


def grad_loss(pred_tile, true_tile, ocean_mask_tile):
    """Finite-difference gradient-matching loss between predicted and true tile
    fields (both [B, th, tw], normalised units). Penalizes error in local spatial
    gradients (forward differences along each axis), not point values -- this is
    what a plain MSE loss does not reward, and what a closed-contour eddy tracker
    (which follows SSH contour shape, not point accuracy) is sensitive to.
    Boundary rows/cols of the tile are excluded (no valid forward difference).
    """
    def d_lon(x):
        return x[:, :, 1:] - x[:, :, :-1]

    def d_lat(x):
        return x[:, 1:, :] - x[:, :-1, :]

    mask = ocean_mask_tile.unsqueeze(0).to(pred_tile.dtype)  # [1, th, tw]
    mask_lon = mask[:, :, 1:] * mask[:, :, :-1]
    mask_lat = mask[:, 1:, :] * mask[:, :-1, :]

    dlon_err = (d_lon(pred_tile) - d_lon(true_tile)) ** 2 * mask_lon
    dlat_err = (d_lat(pred_tile) - d_lat(true_tile)) ** 2 * mask_lat
    denom = mask_lon.sum() + mask_lat.sum()
    if denom < 1:
        return torch.zeros((), dtype=pred_tile.dtype, device=pred_tile.device)
    return (dlon_err.sum() + dlat_err.sum()) / denom.clamp(min=1)


def masked_mse(pred, target, ocean_mask_vm):
    return torch.mean((pred[:, ocean_mask_vm] - target[:, ocean_mask_vm]) ** 2)


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

    ocean_mask = ds["ocean_mask"]  # [n_sensors] bool, full domain

    all_tiles = enumerate_tiles(nlat_s, nlon_s, args.patch_h, args.patch_w,
                                args.patch_stride, args.patch_stride)
    th, tw = min(args.patch_h, nlat_s), min(args.patch_w, nlon_s)
    all_tile_meta = [tile_indices(t, nlat_s, nlon_s, n_vars, n_sensors, args.n_history_days)
                      for t in all_tiles]
    # Drop all-land tiles (zero ocean cells): they contribute nothing to the loss
    # and their masked mean is undefined (0/0 = NaN), so exclude them up front
    # rather than special-casing NaNs during training/eval.
    keep = [i for i, (sidx, _, _) in enumerate(all_tile_meta) if ocean_mask[sidx].any()]
    tiles = [all_tiles[i] for i in keep]
    tile_meta = [all_tile_meta[i] for i in keep]
    if len(tiles) < len(all_tiles):
        print(f"  Dropped {len(all_tiles) - len(tiles)}/{len(all_tiles)} all-land tiles")

    grad_var_idx = VARIABLES.index(args.grad_loss_var)
    n_sensors_tile = th * tw
    if args.grad_loss_weight > 0:
        print(f"Gradient-aware loss ON: weight={args.grad_loss_weight}, "
              f"var={args.grad_loss_var} (index {grad_var_idx})")
    print(f"Tiling: {len(tiles)} tiles of {th}x{tw} sensors "
          f"(stride {args.patch_stride}), covering {nlat_s}x{nlon_s}")

    # ── Model: ONE shared DeepONet sized for a single tile ──────────────────
    d_branch_tile = args.n_history_days * n_vars * th * tw
    model = MultivarDeepONet(
        d_branch=d_branch_tile, n_sensors=th * tw, n_vars=n_vars,
        branch_width=args.branch_width, branch_depth=args.branch_depth,
        trunk_width=args.trunk_width, trunk_depth=args.trunk_depth,
        latent_dim=args.latent_dim, activation="tanh", residual=True,
    ).to(device)
    n_params_patch = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Cheap side-by-side: parameter count of the equivalent WHOLE-DOMAIN model at
    # this same r (and history length), for direct comparison (construction only).
    whole_model = MultivarDeepONet(
        d_branch=args.n_history_days * n_vars * n_sensors, n_sensors=n_sensors, n_vars=n_vars,
        branch_width=args.branch_width, branch_depth=args.branch_depth,
        trunk_width=args.trunk_width, trunk_depth=args.trunk_depth,
        latent_dim=args.latent_dim, activation="tanh", residual=True,
    )
    n_params_whole = sum(p.numel() for p in whole_model.parameters() if p.requires_grad)
    del whole_model
    print(f"Trainable parameters — patch model (shared, per-tile): {n_params_patch:,}")
    print(f"Trainable parameters — equivalent whole-domain model : {n_params_whole:,}")
    if args.n_history_days > 1:
        print(f"Input history: {args.n_history_days} days concatenated per sample")

    # Multi-day history blocks (raw states -> normalised same as branch) are built
    # once per split and concatenated after the current-day block; tile_meta's
    # var_major_cols already account for the offset (see tile_indices).
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

    opt = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = None
    if args.lr_decay_factor < 1.0:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", factor=args.lr_decay_factor, patience=args.lr_decay_patience
        )

    adv_var_idx = VARIABLES.index(args.adv_var)
    discriminator, disc_opt = None, None
    if args.adv_weight > 0:
        discriminator = TileDiscriminator(n_sensors_tile, width=args.disc_width).to(device)
        disc_opt = torch.optim.Adam(discriminator.parameters(), lr=args.disc_lr)
        print(f"Adversarial (LSGAN) loss ON: weight={args.adv_weight}, var={args.adv_var} "
              f"(index {adv_var_idx}), disc_lr={args.disc_lr}, disc_width={args.disc_width}")

    best_val, best_state, since_best = float("inf"), None, 0
    t0 = time.time()
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
        pred_vm = pred.permute(0, 2, 1).reshape(pred.shape[0], -1)  # var-major, matches y
        loss = masked_mse(pred_vm, y, tile_ocean_vm)

        if args.grad_loss_weight > 0:
            gcs, gce = grad_var_idx * n_sensors_tile, (grad_var_idx + 1) * n_sensors_tile
            pred_grid = pred[:, :, grad_var_idx].reshape(-1, th, tw)
            true_grid = y[:, gcs:gce].reshape(-1, th, tw)
            ocean_tile = torch.as_tensor(
                ocean_mask[sensor_idx].reshape(th, tw), dtype=torch.bool, device=device
            )
            loss = loss + args.grad_loss_weight * grad_loss(pred_grid, true_grid, ocean_tile)

        d_loss_val = None
        if args.adv_weight > 0:
            acs, ace = adv_var_idx * n_sensors_tile, (adv_var_idx + 1) * n_sensors_tile
            real_var = y[:, acs:ace]              # [B, n_sensors_tile] true (normalised)
            fake_var = pred[:, :, adv_var_idx]     # [B, n_sensors_tile] predicted

            # --- Discriminator step (LSGAN: MSE toward 1=real, 0=fake) ---
            disc_opt.zero_grad()
            d_real = discriminator(real_var)
            d_fake = discriminator(fake_var.detach())
            d_loss = 0.5 * ((d_real - 1) ** 2).mean() + 0.5 * (d_fake ** 2).mean()
            d_loss.backward()
            disc_opt.step()
            d_loss_val = d_loss.item()

            # --- Generator adversarial term (reward fooling the discriminator) ---
            g_adv = ((discriminator(fake_var) - 1) ** 2).mean()
            loss = loss + args.adv_weight * g_adv

        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % args.display_every == 0 or step == 1:
            model.eval()
            with torch.no_grad():
                val_losses = []
                for ti, (sensor_idx_v, target_cols_v, branch_cols_v) in enumerate(tile_meta):
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
            cur_lr = opt.param_groups[0]["lr"]
            d_str = f"  d_loss {d_loss_val:.4f}" if d_loss_val is not None else ""
            print(f"step {step:6d}  train_loss {loss.item():.5f}  val_loss {val_loss:.5f}  "
                  f"lr {cur_lr:.2e}{d_str}  ({elapsed:.0f}s)")
            if scheduler is not None:
                scheduler.step(val_loss)
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
    branch_test = to_tensor(branch_test_full, device)
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
            pred_norm = model(b, tv).cpu().numpy()  # [N_test, th*tw, n_vars]
            pred_raw = pred_norm.copy()
            for vi in range(n_vars):
                pred_raw[:, :, vi] = pred_norm[:, :, vi] * out_std[vi] + out_mean[vi]
            accum[:, sensor_idx_v, :] += pred_raw
            counts[sensor_idx_v] += 1
    assert (counts[ocean_mask] > 0).all(), (
        "tiling left uncovered OCEAN sensors — check patch/stride settings"
    )  # land sensors may be uncovered (their tiles were all-land and dropped); irrelevant, masked out below
    safe_counts = np.where(counts > 0, counts, 1.0)
    pred_3d = accum / safe_counts[None, :, None]  # [N_test, n_sensors, n_vars] full-grid reconstruction

    true_3d = ds["y_test_raw"].reshape(N_test, n_vars, n_sensors).transpose(0, 2, 1)
    persist_flat = ds["x_test_raw"]
    climatology = ds["climatology"]

    # Save true/pred/persist in the SAME variable-major-flat format as the main
    # trainer's predictions.npz, so downstream tools (e.g. the eddy-tracking
    # analysis) can be run on patch predictions without any format changes.
    pred_flat = pred_3d.transpose(0, 2, 1).reshape(N_test, -1)   # [N_test, n_vars*n_sensors] var-major
    true_flat_vm = true_3d.transpose(0, 2, 1).reshape(N_test, -1)
    np.savez_compressed(
        args.out_dir / "predictions.npz",
        y_true=true_flat_vm, y_pred=pred_flat, y_persist=persist_flat,
        lon=lon_sub, lat=lat_sub, test_indices=ds["test_idx"],
    )

    metrics = {
        "n_params_patch": int(n_params_patch),
        "n_params_whole_domain_equiv": int(n_params_whole),
        "n_tiles": len(tiles),
        "patch_h": th, "patch_w": tw, "patch_stride": args.patch_stride,
        "subsample_r": args.subsample_r,
        "steps_run": int(steps_run),
        "best_val_loss": float(best_val),
        "n_train": int(len(ds["train_idx"])), "n_val": int(len(ds["val_idx"])),
        "n_test": int(len(ds["test_idx"])),
        "n_history_days": args.n_history_days,
        "branch_width": args.branch_width, "branch_depth": args.branch_depth,
        "trunk_width": args.trunk_width, "trunk_depth": args.trunk_depth,
        "latent_dim": args.latent_dim,
        "lr_decay_factor": args.lr_decay_factor, "lr_decay_patience": args.lr_decay_patience,
        "grad_loss_weight": args.grad_loss_weight, "grad_loss_var": args.grad_loss_var,
        "adv_weight": args.adv_weight, "adv_var": args.adv_var, "disc_lr": args.disc_lr,
        "final_lr": float(opt.param_groups[0]["lr"]),
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

    print("\n=== Patch-based DeepONet — test-set summary ===")
    print(f"tiles={len(tiles)} ({th}x{tw}, stride {args.patch_stride})  "
          f"params(patch)={n_params_patch:,}  params(whole-domain equiv)={n_params_whole:,}")
    print(f"mean skill = {mean_skill:+.4f}")
    for vname in VARIABLES:
        print(f"  {vname:8s} skill={metrics[f'skill_{vname}']:+.4f}  "
              f"acc={metrics[f'acc_{vname}']:.3f}  rmse={metrics[f'rmse_{vname}']:.4f}")

    save_json(args.out_dir / "metrics.json", metrics)
    torch.save(model.state_dict(), args.out_dir / "model.pt")
    print(f"\nSaved to {args.out_dir}/")


if __name__ == "__main__":
    main()
