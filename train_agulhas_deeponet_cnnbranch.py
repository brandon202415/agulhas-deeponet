#!/usr/bin/env python3
"""CNN-branch DeepONet: replaces the whole-domain DeepONet's branch (currently
a flatten-then-dense-MLP over the entire sensor grid, with no spatial locality
or translation-equivariance) with a small convolutional encoder, while keeping
the trunk, the per-variable dot-product combination, and the persistence
residual exactly as in MultivarDeepONet.

Motivation (MANUSCRIPT_ISSUES.md Issues 3 & 4): a U-Net baseline beats the
whole-domain DeepONet's own best config by 7-9x on mean skill (single seed,
full scale), and recent literature attributes DeepONet/FNO's known weak
zero-shot resolution transfer specifically to a lack of translation
equivariance (Raonic et al. 2023) -- both point at the SAME diagnosis: the
branch has no spatial inductive bias, not that the trunk-branch operator
formulation itself is unworkable here. This script tests that diagnosis
directly and narrowly: fix only the branch's encoder, change nothing else.

Design:
  - Trunk: UNCHANGED, the exact _make_mlp-based dense trunk from
    MultivarDeepONet (query coordinates -> latent_dim). This is the part of
    the architecture the literature says genuinely IS resolution-flexible;
    it is not the diagnosed problem and is not touched.
  - Branch: the input grid [n_vars, nlat, nlon] goes through a small shared
    CNN encoder (same ConvBlock design as the U-Net baseline's encoder half,
    reused directly -- ONLY the encoder, no decoder/upsampling, since the
    trunk supplies the spatial expansion via the dot product, exactly as in
    the original DeepONet formulation), global-average-pooled to one feature
    vector, then mapped by a small per-variable linear head to latent_dim
    branch coefficients -- structurally identical to MultivarDeepONet's
    per-variable branch heads, just fed a CNN-derived summary instead of the
    raw flattened pixel vector. Every branch head is zero-initialized, so the
    persistence residual behaves exactly as in the rest of this study.
  - Combination, persistence residual, and everything downstream (training
    loop, evaluation, rollout) are UNCHANGED: this script imports and reuses
    run_training/evaluate/rollout_evaluate/build_dataset/load_states from
    train_agulhas_deeponet_prototype.py verbatim. Because those functions
    only ever call `model(branch_input, trunk_input)` generically, this new
    model is a drop-in replacement for MultivarDeepONet -- there is no
    separate, differently-implemented eval/rollout path to introduce a
    methodology confound; the comparison against Table 1/Table 3 is as
    apples-to-apples as this codebase can make it.

Usage:
    python3 train_agulhas_deeponet_cnnbranch.py --nc data/agulhas_prototype.nc \\
        --cache data/cache_r6_local.npz --subsample-r 6 --iterations 5000 \\
        --loss-weight none --out-dir results/cnnbranch_r6_local
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from train_agulhas_deeponet_prototype import (
    VARIABLES, load_states, build_dataset, run_training, evaluate,
    rollout_evaluate, anomaly_correlation, save_json, _make_mlp,
)
from train_cnn_baseline import ConvBlock


class CNNBranchEncoder(nn.Module):
    """Shared conv encoder: [N, n_vars, nlat, nlon] -> [N, hidden_dim].
    Encoder-only (no decoder/upsampling) -- the trunk supplies spatial
    expansion via the dot product, so the branch only needs to produce a
    single global summary per sample, exactly as the original dense-MLP
    branch did, just computed with local/translation-equivariant convolutions
    instead of one big matrix multiply over raw flattened pixels.
    """

    def __init__(self, n_vars, base_width=24, hidden_dim=256):
        super().__init__()
        w = base_width
        self.enc1 = ConvBlock(n_vars, w)
        self.enc2 = ConvBlock(w, 2 * w)
        self.enc3 = ConvBlock(2 * w, 4 * w)
        self.pool = nn.AvgPool2d(2, ceil_mode=True)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Sequential(nn.Linear(4 * w, hidden_dim), nn.GELU())

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        feat = self.gap(e3).flatten(1)  # [N, 4w]
        return self.proj(feat)          # [N, hidden_dim]


class CNNBranchDeepONet(nn.Module):
    """Drop-in replacement for MultivarDeepONet: same (branch_input [N,D]
    var-major-flat, trunk_input [n_query,2]) -> [N, n_query, n_vars] contract,
    same persistence-residual convention (reads branch_input's own column
    slice, so trunk MUST query the branch's own native grid -- this variant
    does not attempt zero-shot resolution transfer, matching MultivarDeepONet's
    default/original behaviour, Sec. 2.2/3.1).
    """

    def __init__(self, n_vars, nlat, nlon, n_sensors, base_width=24, hidden_dim=256,
                 trunk_width=64, trunk_depth=2, latent_dim=32, activation="tanh"):
        super().__init__()
        self.n_vars = n_vars
        self.nlat = nlat
        self.nlon = nlon
        self.n_sensors = n_sensors

        self.encoder = CNNBranchEncoder(n_vars, base_width, hidden_dim)
        self.branch_heads = nn.ModuleList([
            nn.Linear(hidden_dim, latent_dim) for _ in range(n_vars)
        ])
        for head in self.branch_heads:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)  # persistence residual: branch term starts at 0

        trunk_sizes = [2] + [trunk_width] * trunk_depth + [latent_dim]
        self.trunk = _make_mlp(trunk_sizes, activation)
        self.biases = nn.Parameter(torch.zeros(n_vars))

    def forward(self, branch_input, trunk_input):
        N = branch_input.shape[0]
        # [N, n_vars*n_sensors] var-major-flat -> [N, n_vars, nlat, nlon]
        grid = branch_input.reshape(N, self.n_vars, self.n_sensors) \
            .reshape(N, self.n_vars, self.nlat, self.nlon)

        feat = self.encoder(grid)                 # [N, hidden_dim]
        trunk_feats = self.trunk(trunk_input)      # [n_query, latent_dim]

        outputs = []
        for vi in range(self.n_vars):
            branch_feats = self.branch_heads[vi](feat)              # [N, latent_dim]
            out = torch.einsum("np,sp->ns", branch_feats, trunk_feats) + self.biases[vi]
            # Persistence residual: branch_input's own column slice for variable
            # vi (identical convention to MultivarDeepONet's default path) --
            # only valid because trunk_input here always queries this model's
            # own native sensor grid, same restriction as the base architecture.
            out = out + branch_input[:, vi * self.n_sensors:(vi + 1) * self.n_sensors]
            outputs.append(out)
        return torch.stack(outputs, dim=-1)  # [N, n_query, n_vars]


def parse_args():
    p = argparse.ArgumentParser(description="CNN-branch DeepONet (fixes the branch's lack of spatial locality).")
    p.add_argument("--nc", type=Path, default=Path("data/agulhas_prototype.nc"))
    p.add_argument("--cache", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--subsample-r", type=int, default=6)
    p.add_argument("--val-fraction", type=float, default=0.15)
    p.add_argument("--test-fraction", type=float, default=0.15)
    # CNN encoder
    p.add_argument("--base-width", type=int, default=24)
    p.add_argument("--hidden-dim", type=int, default=256)
    # Trunk (same defaults as the main trainer, for comparability)
    p.add_argument("--trunk-width", type=int, default=64)
    p.add_argument("--trunk-depth", type=int, default=2)
    p.add_argument("--latent-dim", type=int, default=32)
    # Training (mirrors train_agulhas_deeponet_prototype.py's relevant flags,
    # so run_training() works unmodified)
    p.add_argument("--iterations", type=int, default=8000)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--display-every", type=int, default=100)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--loss-weight", choices=["none", "variability"], default="none")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--rollout-horizons", type=int, nargs="+", default=[1, 5, 10, 20])
    args = p.parse_args()
    # Physics losses are established inert (Sec. 3.2-3.4) and out of scope for
    # this architecture test; fixed off rather than exposed as CLI options.
    args.lambda_div = 0.0
    args.lambda_geo = 0.0
    args.warmup_steps = 500
    args.step_days = 1
    return args


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    # No MPS branch: on Apple Silicon, this model's ops (AdaptiveAvgPool2d /
    # GroupNorm combination) hit a slow or stalling MPS path in local testing;
    # CPU is reliably fast for local smoke tests, and the real cluster run
    # always has CUDA regardless, so this only affects local dev convenience.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    lon_sub, lat_sub, states = load_states(args.nc, args.subsample_r, args.cache)
    ds = build_dataset(states, lon_sub, lat_sub, args.test_fraction, args.val_fraction,
                        step_days=args.step_days)
    nlat_s, nlon_s = ds["nlat_s"], ds["nlon_s"]
    n_sensors, n_vars = ds["n_sensors"], ds["n_vars"]
    print(f"Full grid: {nlat_s} x {nlon_s} = {n_sensors} sensors, r={args.subsample_r}")
    print(f"  Train / Val / Test    : {len(ds['train_idx'])} / "
          f"{len(ds['val_idx'])} / {len(ds['test_idx'])}  (chronological)")

    model = CNNBranchDeepONet(
        n_vars=n_vars, nlat=nlat_s, nlon=nlon_s, n_sensors=n_sensors,
        base_width=args.base_width, hidden_dim=args.hidden_dim,
        trunk_width=args.trunk_width, trunk_depth=args.trunk_depth,
        latent_dim=args.latent_dim, activation="tanh",
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,} "
          f"(cf. whole-domain dense-branch DeepONet 14,239,206; CNN baseline 433,734)")

    history = run_training(model, ds, lon_sub, lat_sub, args, device)

    pred_flat, true_flat, pred_3d = evaluate(model, ds, device)
    N_test = pred_3d.shape[0]
    true_3d = ds["y_test_raw"].reshape(N_test, n_vars, n_sensors).transpose(0, 2, 1)
    persist_flat = ds["x_test_raw"]
    climatology = ds["climatology"]
    ocean_mask = ds["ocean_mask"]

    print("\n=== CNN-branch DeepONet -- single-step test-set summary (Table 1 columns) ===")
    metrics = {
        "n_params": int(n_params), "subsample_r": args.subsample_r,
        "loss_weight": args.loss_weight, "learning_rate": args.learning_rate,
        "seed": args.seed, "best_val_loss": history["val_mse_unweighted"],
        "steps_run": len(history["steps"]) and history["steps"][-1],
        "base_width": args.base_width, "hidden_dim": args.hidden_dim,
    }
    skills = []
    print(f"  {'var':8s} {'rmse':>9s} {'persist':>9s} {'skill':>8s} {'nrmse':>7s} "
          f"{'acc':>6s} {'bias':>9s}")
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
        skills.append(skill)

        metrics[f"rmse_{vname}"] = rmse_m
        metrics[f"rmse_persist_{vname}"] = rmse_p
        metrics[f"skill_{vname}"] = skill
        metrics[f"nrmse_{vname}"] = nrmse
        metrics[f"bias_{vname}"] = bias
        metrics[f"acc_{vname}"] = acc
        print(f"  {vname:8s} {rmse_m:9.4f} {rmse_p:9.4f} {skill:+8.3f} "
              f"{nrmse:7.3f} {acc:6.3f} {bias:+9.4f}")
    mean_skill = float(np.mean(skills))
    metrics["mean_skill"] = mean_skill
    print(f"  mean skill = {mean_skill:+.4f}")

    print("\nRunning autoregressive rollout …")
    rollout = rollout_evaluate(model, ds, states, args, device)
    if rollout is not None:
        np.savez_compressed(
            args.out_dir / "rollout.npz",
            horizons=rollout["horizons"], rmse=rollout["rmse"], nrmse=rollout["nrmse"],
            acc=rollout["acc"], rmse_persist=rollout["rmse_persist"],
            acc_persist=rollout["acc_persist"], skill=rollout["skill"],
            n_starts=rollout["n_starts"], variables=np.array(VARIABLES),
        )
        for hi, h in enumerate(rollout["horizons"]):
            for vi, vname in enumerate(VARIABLES):
                metrics[f"rollout_rmse_{vname}_{h}d"] = float(rollout["rmse"][hi, vi])
                metrics[f"rollout_rmse_persist_{vname}_{h}d"] = float(rollout["rmse_persist"][hi, vi])
                metrics[f"rollout_acc_{vname}_{h}d"] = float(rollout["acc"][hi, vi])
                metrics[f"rollout_skill_{vname}_{h}d"] = float(rollout["skill"][hi, vi])
            mean_h_skill = float(np.nanmean(rollout["skill"][hi]))
            print(f"    {h:3d}d: mean skill = {mean_h_skill:+.4f}")

    save_json(args.out_dir / "metrics.json", metrics)
    torch.save(model.state_dict(), args.out_dir / "model.pt")
    print(f"\nSaved to {args.out_dir}/")


if __name__ == "__main__":
    main()
