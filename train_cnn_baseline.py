#!/usr/bin/env python3
"""Non-persistence learned baseline: a small U-Net operating on the gridded
surface state, trained identically to the whole-domain DeepONet (same data
pipeline, split, persistence-residual trick, loss, and skill/NRMSE/ACC
metrics), to tell whether the patch DeepONet's advantage is architecture-
specific or "any reasonably-sized network helps" (review item: no
non-persistence learned baseline anywhere in the paper).

Reuses load_states/build_dataset/anomaly_correlation/_var_major_flat/
VARIABLES/save_json from train_agulhas_deeponet_prototype.py so the
comparison is on identical data, split, and normalization -- only the model
class differs. Single-step evaluation reports the same columns as Table 1
(RMSE, Persist RMSE, Skill, NRMSE, ACC, Bias); rollout evaluation reuses the
exact same test-starts selection, land-holding, and skill formula as
rollout_evaluate() in the DeepONet trainer, and saves rollout.npz in the
identical schema -- so aggregate_seed_sweep.py and extract_rollout_rmse.py
work unmodified on this script's output directories.

The branch-normalized, land-zeroed, var-major-flat tensors from build_dataset
are reshaped back to [N, n_vars, nlat, nlon] grids for convolution; the same
persistence-residual design used throughout this study is kept (zero-init the
final conv so the network starts at exact persistence), since removing it
would not be a fair test of "architecture" -- it would also be testing away
the one trick already shown (Sec. 4.1) to be responsible for most of the
data-driven gain in this study. Branch and output normalization use the same
per-variable scalars (out_mean/out_std), so denormalizing a prediction and
renormalizing it for the next rollout step is the identity operation except
for re-zeroing land -- mirrored exactly from the DeepONet's rollout_evaluate.

Usage:
    python3 train_cnn_baseline.py --nc data/agulhas_prototype.nc --subsample-r 6 \\
        --iterations 5000 --loss-weight none --out-dir results/cnn_baseline_r6_local

Physics-informed option (Issue 27, referee report 2026-08-05): Sec. 2.7/4.7's
divergence-free/geostrophic null result was previously tested only on the
whole-domain DeepONet, not the CNN used everywhere else in the paper. Pass
--lambda-div/--lambda-geo (reusing physics_losses() from the DeepONet trainer
unmodified -- it already takes a model-agnostic [N,nlat,nlon,n_vars] grid of
physical-unit predictions) to run the identical experiment on this
architecture, e.g. to replicate the DeepONet's own winning physics config:
    python3 train_cnn_baseline.py --nc data/agulhas_prototype.nc --subsample-r 6 \\
        --iterations 8000 --lambda-geo 0.1 --seed 2026 \\
        --out-dir results/cnn_physics_geo0.1_r6_seed2026
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(4)  # small grids -- PyTorch's default all-core intra-op
                          # parallelism thrashes on tensors this tiny (CPU only)

from train_agulhas_deeponet_prototype import (
    load_states, build_dataset, anomaly_correlation,
    _var_major_flat, VARIABLES, save_json,
    physics_losses, DEG2RAD,
)


class ConvBlock(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, padding=1), nn.GroupNorm(1, c_out), nn.GELU(),
            nn.Conv2d(c_out, c_out, 3, padding=1), nn.GroupNorm(1, c_out), nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class UNetForecast(nn.Module):
    """Small U-Net, 2 downsampling levels, persistence-residual output.

    Input:  [N, n_vars, nlat, nlon]  normalized, land-zeroed current state
    Output: [N, n_vars, nlat, nlon]  normalized predicted next state
    """

    def __init__(self, n_vars, base_width=24):
        super().__init__()
        w = base_width
        self.enc1 = ConvBlock(n_vars, w)
        self.enc2 = ConvBlock(w, 2 * w)
        self.enc3 = ConvBlock(2 * w, 4 * w)
        self.pool = nn.AvgPool2d(2, ceil_mode=True)
        self.bottleneck = ConvBlock(4 * w, 4 * w)
        self.dec2 = ConvBlock(4 * w + 2 * w, 2 * w)
        self.dec1 = ConvBlock(2 * w + w, w)
        self.out_conv = nn.Conv2d(w, n_vars, 1)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(self, x):
        e1 = self.enc1(x)                                   # [N, w,   nlat,   nlon]
        e2 = self.enc2(self.pool(e1))                        # [N, 2w,  nlat/2, nlon/2]
        e3 = self.enc3(self.pool(e2))                        # [N, 4w,  nlat/4, nlon/4]
        b = self.bottleneck(e3)
        d2_up = F.interpolate(b, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([d2_up, e2], dim=1))
        d1_up = F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([d1_up, e1], dim=1))
        increment = self.out_conv(d1)
        return x + increment                                 # persistence-residual


def to_grid(flat_vm, n_vars, nlat, nlon):
    """[N, n_vars*n_sensors] var-major-flat -> [N, n_vars, nlat, nlon]."""
    n = flat_vm.shape[0]
    n_sensors = nlat * nlon
    return flat_vm.reshape(n, n_vars, n_sensors).reshape(n, n_vars, nlat, nlon)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def chunked_forward(model, x, chunk_size=256):
    """Run model(x) in batch-dim chunks and concatenate. Exactly equivalent to
    model(x) in one shot -- GroupNorm(1, ...) normalizes per-sample, not across
    the batch -- but bounds peak VRAM. Needed because validation/test/rollout
    push the *entire* val/test set through in one forward pass, which at r=6
    fit in a 32GB GPU but at r=3 (4x the grid cells) does not."""
    outs = []
    for i in range(0, x.shape[0], chunk_size):
        outs.append(model(x[i:i + chunk_size]))
    return torch.cat(outs, dim=0)


def rollout_evaluate_cnn(model, ds, states, horizons, nlat_s, nlon_s, device):
    """Autoregressive multi-step rollout, mirroring rollout_evaluate() in
    train_agulhas_deeponet_prototype.py exactly: same test-starts selection
    (test_idx filtered by t+H <= T-1), same land-holding (frozen at the raw
    t0 value every step), same skill formula, same output schema. Runs
    entirely in raw physical units between steps, normalizing fresh at each
    step's input and denormalizing each prediction -- identical convention
    to the DeepONet rollout so the numbers are directly comparable.
    """
    n_sensors = ds["n_sensors"]
    n_vars = ds["n_vars"]
    ocean = ds["ocean_mask"]
    clim = ds["climatology"]  # [n_sensors, n_vars]
    out_mean, out_std = ds["out_mean"], ds["out_std"]  # [n_vars], same scalars as branch norm

    k = 1  # this baseline is trained at step_days=1, matching Table 1/3
    horizons = sorted(set(int(h) for h in horizons if h > 0))
    H = max(horizons)
    T = states.shape[0]
    starts = np.array([t for t in ds["test_idx"] if t + H <= T - 1], dtype=np.int64)
    if starts.size == 0:
        print("  [rollout] not enough test lead time; skipping.")
        return None
    S = starts.size

    mean_t = torch.tensor(out_mean, dtype=torch.float32, device=device).view(1, n_vars, 1, 1)
    std_t = torch.tensor(out_std, dtype=torch.float32, device=device).view(1, n_vars, 1, 1)
    land_mask = torch.tensor(~ocean.reshape(nlat_s, nlon_s), dtype=torch.bool, device=device)
    land_mask = land_mask.view(1, 1, nlat_s, nlon_s).expand(S, n_vars, -1, -1)

    # Initial raw state grids for every start: [S, n_vars, nlat, nlon]
    init_flat = _var_major_flat(states[starts].astype(np.float64))
    init_grid = torch.tensor(
        init_flat.reshape(S, n_vars, n_sensors).reshape(S, n_vars, nlat_s, nlon_s),
        dtype=torch.float32, device=device,
    )
    cur_grid = init_grid.clone()

    preds_at = {}
    model.eval()
    with torch.no_grad():
        for app in range(1, H // k + 1):
            lead = app * k
            norm_in = (cur_grid - mean_t) / std_t
            norm_in = norm_in.masked_fill(land_mask, 0.0)
            pred_norm = chunked_forward(model, norm_in)
            pred_raw = pred_norm * std_t + mean_t
            pred_raw = torch.where(land_mask, init_grid, pred_raw)  # hold land at raw t0 value
            cur_grid = pred_raw
            if lead in horizons:
                preds_at[lead] = cur_grid.detach().cpu().numpy().reshape(S, n_vars, n_sensors) \
                    .reshape(S, n_vars * n_sensors)

    persist_vm = init_flat  # frozen t0, raw var-major flat -- naive forecast at every horizon
    rmse = np.full((len(horizons), n_vars), np.nan)
    nrmse = np.full((len(horizons), n_vars), np.nan)
    acc = np.full((len(horizons), n_vars), np.nan)
    rmse_persist = np.full((len(horizons), n_vars), np.nan)
    acc_persist = np.full((len(horizons), n_vars), np.nan)
    skill = np.full((len(horizons), n_vars), np.nan)

    for hi, h in enumerate(horizons):
        pred_vm = preds_at[h]
        true_vm = _var_major_flat(states[starts + h].astype(np.float64))
        for vi in range(n_vars):
            cs, ce = vi * n_sensors, (vi + 1) * n_sensors
            p = pred_vm[:, cs:ce][:, ocean]
            t = true_vm[:, cs:ce][:, ocean]
            pr = persist_vm[:, cs:ce][:, ocean]
            c = clim[ocean, vi]
            rmse[hi, vi] = float(np.sqrt(np.mean((p - t) ** 2)))
            rmse_persist[hi, vi] = float(np.sqrt(np.mean((pr - t) ** 2)))
            std_true = float(np.std(t))
            nrmse[hi, vi] = rmse[hi, vi] / std_true if std_true > 1e-12 else np.nan
            acc[hi, vi] = anomaly_correlation(p, t, c)
            acc_persist[hi, vi] = anomaly_correlation(pr, t, c)
            skill[hi, vi] = (1.0 - (rmse[hi, vi] / rmse_persist[hi, vi]) ** 2
                             if rmse_persist[hi, vi] > 1e-12 else np.nan)

    return {
        "horizons": np.array(horizons, dtype=np.int64),
        "rmse": rmse, "nrmse": nrmse, "acc": acc,
        "rmse_persist": rmse_persist, "acc_persist": acc_persist,
        "skill": skill, "n_starts": int(S), "variables": VARIABLES,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nc", default="data/agulhas_prototype.nc")
    ap.add_argument("--cache", default=None)
    ap.add_argument("--subsample-r", type=int, default=6)
    ap.add_argument("--iterations", type=int, default=5000)
    ap.add_argument("--learning-rate", type=float, default=3e-4)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--base-width", type=int, default=24)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--val-fraction", type=float, default=0.15)
    ap.add_argument("--test-fraction", type=float, default=0.15)
    ap.add_argument("--loss-weight", choices=["none", "variability"], default="none",
                     help="none = plain ocean-masked MSE, matching the DeepONet's own "
                          "winning whole-domain config (Table 2: lr3e-4, loss-weight none). "
                          "variability = skill-aligned inverse-increment-variance weighting.")
    ap.add_argument("--rollout-horizons", type=int, nargs="+", default=[1, 5, 10, 20],
                     help="Matches Table 3's daily-model rollout horizons exactly.")
    ap.add_argument("--lambda-div", type=float, default=0.0,
                     help="Weight for divergence-free physics loss on the RAW residual "
                          "(0 = off; default off since it's trivially satisfied on reanalysis, "
                          "same rationale and same default as the DeepONet's own --lambda-div). "
                          "Issue 27 (referee report, 2026-08-05): re-tests Sec. 2.7/4.7's "
                          "physics-informed null result on the CNN, the architecture actually "
                          "used everywhere else in the paper -- that experiment was previously "
                          "only run on the whole-domain DeepONet.")
    ap.add_argument("--lambda-geo", type=float, default=0.0,
                     help="Weight for geostrophic-consistency physics loss on the RAW residual "
                          "(the DeepONet's own winning config used 0.1; pass --lambda-geo 0.1 "
                          "to replicate that config on the CNN for Issue 27).")
    ap.add_argument("--warmup-steps", type=int, default=500,
                     help="Linear ramp for physics loss weights over the first N steps "
                          "(matches the DeepONet's own default).")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    lon, lat, states = load_states(args.nc, subsample_r=args.subsample_r, cache=args.cache)
    ds = build_dataset(states, lon, lat, test_fraction=args.test_fraction,
                        val_fraction=args.val_fraction, step_days=1)
    nlat_s, nlon_s = ds["nlat_s"], ds["nlon_s"]
    n_vars, n_sensors = ds["n_vars"], ds["n_sensors"]

    model = UNetForecast(n_vars=n_vars, base_width=args.base_width).to(device)
    n_params = count_params(model)
    print(f"U-Net baseline: {n_params:,} trainable parameters "
          f"(cf. whole-domain DeepONet 14,239,206 at r=6; patch DeepONet 965,862-2,117,862)")
    print(f"  Train / Val / Test    : {len(ds['train_idx'])} / "
          f"{len(ds['val_idx'])} / {len(ds['test_idx'])}  (chronological)")
    print(f"  loss-weight           : {args.loss_weight}")

    def T(a):
        return torch.tensor(a, dtype=torch.float32, device=device)

    x_train = to_grid(T(ds["branch_train"]), n_vars, nlat_s, nlon_s)
    y_train = to_grid(T(ds["y_train_norm"]), n_vars, nlat_s, nlon_s)
    x_val   = to_grid(T(ds["branch_val"]),   n_vars, nlat_s, nlon_s)
    y_val   = to_grid(T(ds["y_val_norm"]),   n_vars, nlat_s, nlon_s)
    x_test  = to_grid(T(ds["branch_test"]),  n_vars, nlat_s, nlon_s)

    ocean_mask = T(ds["ocean_mask"].astype(np.float32)).reshape(1, 1, nlat_s, nlon_s)
    if args.loss_weight == "variability":
        loss_weight = to_grid(T(ds["loss_weight"]).unsqueeze(0), n_vars, nlat_s, nlon_s)[0]  # [n_vars,nlat,nlon]
    else:
        loss_weight = torch.ones(n_vars, nlat_s, nlon_s, device=device)

    # Physics-loss geometry (Issue 27): identical construction to the DeepONet
    # trainer's run_training(), reusing its own physics_losses() unmodified --
    # that function is already model-agnostic (takes a [N,nlat,nlon,n_vars]
    # grid of physical-unit predictions, not anything DeepONet-specific).
    use_physics = args.lambda_div > 0.0 or args.lambda_geo > 0.0
    if use_physics:
        lon_rad_1d = T(lon * DEG2RAD)
        lat_rad_1d = T(lat * DEG2RAD)
        LAT_rad, _ = torch.meshgrid(lat_rad_1d, lon_rad_1d, indexing="ij")
        ocean_mask_grid = T(ds["ocean_mask"].reshape(nlat_s, nlon_s).astype(np.float32)).bool()
        out_mean_t = T(ds["out_mean"])
        out_std_t = T(ds["out_std"])
        print(f"  physics losses        : lambda_div={args.lambda_div} lambda_geo={args.lambda_geo} "
              f"warmup_steps={args.warmup_steps}")

    opt = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    n_train = x_train.shape[0]
    best_val, best_state, bad_evals = float("inf"), None, 0
    eval_every = 100
    step = 0

    for step in range(1, args.iterations + 1):
        model.train()
        idx = torch.randint(0, n_train, (min(args.batch_size, n_train),), device=device)
        xb, yb = x_train[idx], y_train[idx]
        pred = model(xb)
        err2 = (pred - yb) ** 2 * loss_weight[None] * ocean_mask
        l_data = err2.sum() / (ocean_mask.sum() * n_vars * xb.shape[0])

        l_div = torch.tensor(0.0, device=device)
        l_geo = torch.tensor(0.0, device=device)
        if use_physics:
            # pred: [N, n_vars, nlat, nlon] normalized -> denormalize, then
            # permute to [N, nlat, nlon, n_vars] to match physics_losses()'s
            # variable-last convention (unchanged from the DeepONet trainer).
            pred_phys_grid = pred * out_std_t[None, :, None, None] + out_mean_t[None, :, None, None]
            pred_phys_grid = pred_phys_grid.permute(0, 2, 3, 1)
            l_div, l_geo = physics_losses(pred_phys_grid, LAT_rad, lon_rad_1d, lat_rad_1d, ocean_mask_grid)
        ramp = min(1.0, step / max(args.warmup_steps, 1))
        loss = l_data + ramp * args.lambda_div * l_div + ramp * args.lambda_geo * l_geo

        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % eval_every == 0 or step == args.iterations:
            model.eval()
            with torch.no_grad():
                pv = chunked_forward(model, x_val)
                verr2 = (pv - y_val) ** 2 * ocean_mask
                vloss = (verr2.sum() / (ocean_mask.sum() * n_vars * x_val.shape[0])).item()
            phys_str = f"  l_div {l_div.item():.3e}  l_geo {l_geo.item():.3e}" if use_physics else ""
            print(f"step {step:6d}  train_loss {loss.item():.5f}  val_loss {vloss:.5f}{phys_str}")
            if vloss < best_val - 1e-6:
                best_val, best_state, bad_evals = vloss, {k: v.clone() for k, v in model.state_dict().items()}, 0
            else:
                bad_evals += 1
                if bad_evals >= args.patience:
                    print(f"Early stop at step {step} (best val_loss={best_val:.5f})")
                    break

    model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        pred_test_norm = chunked_forward(model, x_test).cpu().numpy()  # [N, n_vars, nlat, nlon]
    pred_test_flat = pred_test_norm.reshape(pred_test_norm.shape[0], n_vars, n_sensors) \
        .reshape(pred_test_norm.shape[0], n_vars * n_sensors)
    out_mean, out_std = ds["out_mean"], ds["out_std"]
    pred_phys = np.zeros_like(pred_test_flat)
    for vi in range(n_vars):
        c0, c1 = vi * n_sensors, (vi + 1) * n_sensors
        pred_phys[:, c0:c1] = pred_test_flat[:, c0:c1] * out_std[vi] + out_mean[vi]

    y_true = ds["y_test_raw"]
    y_persist = ds["x_test_raw"]
    clim_vm = ds["climatology"].T.reshape(-1)  # [n_vars, n_sensors] -> flat matching var-major

    print("\n=== U-Net baseline -- single-step test-set summary (Table 1 columns) ===")
    print(f"params={n_params:,}")
    metrics = {"n_params": n_params, "n_train": int(ds["train_idx"].shape[0]),
               "n_test": int(ds["test_idx"].shape[0]), "subsample_r": args.subsample_r,
               "loss_weight": args.loss_weight, "learning_rate": args.learning_rate,
               "seed": args.seed, "iterations": step, "best_val_loss": best_val,
               "lambda_div": args.lambda_div, "lambda_geo": args.lambda_geo,
               "warmup_steps": args.warmup_steps}
    skills = []
    ocean = ds["ocean_mask"]
    print(f"  {'var':8s} {'rmse':>9s} {'persist':>9s} {'skill':>8s} {'nrmse':>7s} "
          f"{'acc':>6s} {'bias':>9s}")
    for vi, vname in enumerate(VARIABLES):
        c0, c1 = vi * n_sensors, (vi + 1) * n_sensors
        yt, yp, ys = y_true[:, c0:c1][:, ocean], pred_phys[:, c0:c1][:, ocean], y_persist[:, c0:c1][:, ocean]
        rmse_model = np.sqrt(np.mean((yp - yt) ** 2))
        rmse_persist = np.sqrt(np.mean((ys - yt) ** 2))
        skill = 1.0 - (rmse_model / rmse_persist) ** 2 if rmse_persist > 0 else float("nan")
        std_true = np.std(yt)
        nrmse = rmse_model / std_true if std_true > 1e-12 else float("nan")
        bias = float(np.mean(yp - yt))
        clim_v = clim_vm[c0:c1][ocean]
        acc = anomaly_correlation(yp, yt, clim_v)
        skills.append(skill)
        metrics[f"skill_{vname}"] = skill
        metrics[f"rmse_{vname}"] = rmse_model
        metrics[f"rmse_persist_{vname}"] = rmse_persist
        metrics[f"nrmse_{vname}"] = nrmse
        metrics[f"acc_{vname}"] = acc
        metrics[f"bias_{vname}"] = bias
        print(f"  {vname:8s} {rmse_model:9.4f} {rmse_persist:9.4f} {skill:+8.3f} "
              f"{nrmse:7.3f} {acc:6.3f} {bias:+9.4f}")
    mean_skill = float(np.mean(skills))
    metrics["mean_skill"] = mean_skill
    print(f"  mean skill = {mean_skill:+.4f}")

    print("\nRunning autoregressive rollout …")
    rollout = rollout_evaluate_cnn(model, ds, states, args.rollout_horizons, nlat_s, nlon_s, device)
    if rollout is not None:
        np.savez_compressed(
            args.out_dir / "rollout.npz",
            horizons=rollout["horizons"], rmse=rollout["rmse"], nrmse=rollout["nrmse"],
            acc=rollout["acc"], rmse_persist=rollout["rmse_persist"],
            acc_persist=rollout["acc_persist"], skill=rollout["skill"],
            n_starts=rollout["n_starts"], variables=np.array(VARIABLES),
        )
        print(f"  Rollout — mean skill vs. persistence by horizon (n_starts={rollout['n_starts']}):")
        for hi, h in enumerate(rollout["horizons"]):
            for vi, vname in enumerate(VARIABLES):
                metrics[f"rollout_rmse_{vname}_{h}d"] = float(rollout["rmse"][hi, vi])
                metrics[f"rollout_rmse_persist_{vname}_{h}d"] = float(rollout["rmse_persist"][hi, vi])
                metrics[f"rollout_acc_{vname}_{h}d"] = float(rollout["acc"][hi, vi])
                metrics[f"rollout_skill_{vname}_{h}d"] = float(rollout["skill"][hi, vi])
            mean_h_skill = float(np.nanmean(rollout["skill"][hi]))
            print(f"    {h:3d}d: mean skill = {mean_h_skill:+.4f}")

    lat_grid = lat if lat.ndim == 1 else lat
    np.savez_compressed(
        args.out_dir / "predictions.npz",
        y_true=y_true, y_pred=pred_phys, y_persist=y_persist,
        lon=lon, lat=lat, test_indices=ds["test_idx"],
    )
    save_json(args.out_dir / "metrics.json", metrics)
    torch.save(model.state_dict(), args.out_dir / "model.pt")
    print(f"\nSaved to {args.out_dir}/")


if __name__ == "__main__":
    main()
