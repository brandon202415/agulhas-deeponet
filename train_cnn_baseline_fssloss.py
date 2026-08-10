#!/usr/bin/env python3
"""CNN baseline (train_cnn_baseline.py's UNetForecast, unchanged) trained with
an added differentiable, neighborhood-based auxiliary loss on the SSH (zos)
field -- a soft, trainable analogue of the Fractions Skill Score (Roberts &
Lean 2008) used as a *training objective* rather than only an evaluation
metric.

Motivation: item 23 in RESEARCH_LOG.md found the CNN baseline's massive
grid-point RMSE win over persistence (mean skill +0.244 local / +0.38
full-scale) still does NOT translate into a statistically significant
eddy-tracking improvement (py-eddy-tracker recall/position-error,
`eddy_tracking/eddy_stat_test.py`) -- the double-penalty problem (Ebert 2008)
survives a change of architecture. Pointwise MSE is exactly the loss that
double-penalizes a correctly-shaped, slightly-displaced eddy (once as a
missed observation, once as a false detection); an architecture fix cannot
address a training-objective problem. This script tests the fix at the
objective level instead: does rewarding "the eddy-scale structure was in
roughly the right neighborhood" over "every pixel matched exactly" change
whether the model's real skill converts into eddy-tracking skill?

Soft-FSS construction (all differentiable, computed only on the zos channel,
since that's the field py-eddy-tracker's contour identification operates on):
  1. High-pass filter: local anomaly = x - box_local_mean(x, k=fss-highpass-k).
     k=9 grid cells at r=6 (~50-55 km spacing) is ~450-500 km, matching the
     400 km Bessel high-pass cutoff eddy_tracking_analysis.py already uses,
     so the loss is looking at roughly the same spatial scale the evaluation
     pipeline does.
  2. Soft threshold: sigmoid((anomaly - theta)/s) for positive (anticyclonic)
     and sigmoid((-anomaly - theta)/s) for negative (cyclonic) structure,
     theta/s set as multiples of the *training set's own* high-pass std
     (data-derived, not a hardcoded magic number).
  3. Neighborhood fraction: box-average the soft indicator over a window
     (fss-window=5 cells, ~250-275 km, matching MATCH_RADIUS_KM=250 in the
     eddy-tracking matcher) -- this is literally FSS's own construction
     (fraction of neighborhood exceeding threshold), made differentiable by
     replacing the hard indicator with a sigmoid.
  4. Loss term: MSE between the prediction's and truth's neighborhood-
     fraction fields, ocean-masked, added to the existing pointwise loss
     with weight --fss-weight.

`--fss-weight 0` must reproduce train_cnn_baseline.py's training exactly --
verified below (no new parameters are introduced at all, only an additional
loss term multiplied by a weight that zeroes it out).

Reuses UNetForecast/to_grid/count_params/rollout_evaluate_cnn from
train_cnn_baseline.py completely unchanged -- same reason as every other
"isolate one variable" trainer in this study: no separate model or eval path
to introduce a confound between this and the baseline it's compared against.

Usage:
    python3 train_cnn_baseline_fssloss.py --nc data/agulhas_prototype.nc \\
        --subsample-r 6 --iterations 4100 --fss-weight 1.0 \\
        --out-dir results/cnn_fssloss_r6_local
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

torch.set_num_threads(4)

from train_agulhas_deeponet_prototype import (
    load_states, build_dataset, anomaly_correlation,
    _var_major_flat, VARIABLES, save_json,
)
from train_cnn_baseline import UNetForecast, to_grid, count_params, rollout_evaluate_cnn

ZOS_IDX = VARIABLES.index("zos")


def box_local_mean(x, k):
    """[N,1,nlat,nlon] -> local mean over a kxk window, reflect-padded so
    edge cells aren't biased toward zero by pooling in land/boundary zeros."""
    pad = k // 2
    xp = F.pad(x, [pad, pad, pad, pad], mode="reflect")
    return F.avg_pool2d(xp, kernel_size=k, stride=1, padding=0)


def soft_fraction_fields(zos, highpass_k, window, theta, softness):
    """zos: [N,1,nlat,nlon] -> (frac_pos, frac_neg), each [N,1,nlat,nlon],
    the soft-thresholded, neighborhood-averaged anomaly fraction fields."""
    anomaly = zos - box_local_mean(zos, highpass_k)
    pos = torch.sigmoid((anomaly - theta) / softness)
    neg = torch.sigmoid((-anomaly - theta) / softness)
    frac_pos = box_local_mean(pos, window)
    frac_neg = box_local_mean(neg, window)
    return frac_pos, frac_neg


def fss_loss_term(pred_zos, true_zos, ocean_mask, highpass_k, window, theta, softness):
    fp_pred, fn_pred = soft_fraction_fields(pred_zos, highpass_k, window, theta, softness)
    fp_true, fn_true = soft_fraction_fields(true_zos, highpass_k, window, theta, softness)
    err = (fp_pred - fp_true) ** 2 + (fn_pred - fn_true) ** 2
    return (err * ocean_mask).sum() / ocean_mask.sum().clamp_min(1.0)


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
    ap.add_argument("--rollout-horizons", type=int, nargs="+", default=[1, 5, 10, 20])
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--fss-weight", type=float, default=0.0,
                     help="weight on the soft-FSS auxiliary loss (zos channel only). "
                          "0.0 reproduces train_cnn_baseline.py's plain pointwise-MSE training exactly.")
    ap.add_argument("--fss-highpass-k", type=int, default=9,
                     help="box high-pass kernel, grid cells (~450-500km at r=6, matches the "
                          "400km Bessel cutoff eddy_tracking_analysis.py uses).")
    ap.add_argument("--fss-window", type=int, default=5,
                     help="neighborhood-fraction pooling window, grid cells "
                          "(~250-275km at r=6, matches MATCH_RADIUS_KM=250).")
    ap.add_argument("--fss-threshold-std", type=float, default=0.75,
                     help="soft-threshold theta, in units of the training set's own "
                          "high-passed zos std (data-derived, not a fixed physical value).")
    ap.add_argument("--fss-softness", type=float, default=0.25,
                     help="sigmoid transition width, in units of the same std.")
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
    print(f"U-Net + soft-FSS loss: {n_params:,} trainable parameters (identical to plain baseline -- "
          f"no new parameters, only an added loss term)")
    print(f"  Train / Val / Test    : {len(ds['train_idx'])} / "
          f"{len(ds['val_idx'])} / {len(ds['test_idx'])}  (chronological)")
    print(f"  fss-weight={args.fss_weight}  highpass-k={args.fss_highpass_k}  "
          f"window={args.fss_window}  threshold-std={args.fss_threshold_std}  softness={args.fss_softness}")

    def T(a):
        return torch.tensor(a, dtype=torch.float32, device=device)

    x_train = to_grid(T(ds["branch_train"]), n_vars, nlat_s, nlon_s)
    y_train = to_grid(T(ds["y_train_norm"]), n_vars, nlat_s, nlon_s)
    x_val   = to_grid(T(ds["branch_val"]),   n_vars, nlat_s, nlon_s)
    y_val   = to_grid(T(ds["y_val_norm"]),   n_vars, nlat_s, nlon_s)
    x_test  = to_grid(T(ds["branch_test"]),  n_vars, nlat_s, nlon_s)

    ocean_mask = T(ds["ocean_mask"].astype(np.float32)).reshape(1, 1, nlat_s, nlon_s)
    loss_weight = torch.ones(n_vars, nlat_s, nlon_s, device=device)  # plain MSE, matches
                                                                       # the DeepONet's winning config

    with torch.no_grad():
        zos_train_true = y_train[:, ZOS_IDX:ZOS_IDX + 1]
        hp_train = zos_train_true - box_local_mean(zos_train_true, args.fss_highpass_k)
        hp_std = float(hp_train[:, :, ds["ocean_mask"].reshape(nlat_s, nlon_s)].std().item())
    theta = args.fss_threshold_std * hp_std
    softness = args.fss_softness * hp_std
    print(f"  data-derived: high-pass zos std={hp_std:.5f}  theta={theta:.5f}  softness={softness:.5f}")

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
        base_loss = err2.sum() / (ocean_mask.sum() * n_vars * xb.shape[0])
        if args.fss_weight != 0.0:
            fss = fss_loss_term(pred[:, ZOS_IDX:ZOS_IDX + 1], yb[:, ZOS_IDX:ZOS_IDX + 1],
                                 ocean_mask, args.fss_highpass_k, args.fss_window, theta, softness)
        else:
            fss = torch.zeros((), device=device)
        loss = base_loss + args.fss_weight * fss
        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % eval_every == 0 or step == args.iterations:
            model.eval()
            with torch.no_grad():
                pv = model(x_val)
                verr2 = (pv - y_val) ** 2 * ocean_mask
                vloss = (verr2.sum() / (ocean_mask.sum() * n_vars * x_val.shape[0])).item()
            print(f"step {step:6d}  train_loss {loss.item():.5f}  "
                  f"(base {base_loss.item():.5f}  fss {fss.item():.5f})  val_loss {vloss:.5f}")
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
        pred_test_norm = model(x_test).cpu().numpy()
    pred_test_flat = pred_test_norm.reshape(pred_test_norm.shape[0], n_vars, n_sensors) \
        .reshape(pred_test_norm.shape[0], n_vars * n_sensors)
    out_mean, out_std = ds["out_mean"], ds["out_std"]
    pred_phys = np.zeros_like(pred_test_flat)
    for vi in range(n_vars):
        c0, c1 = vi * n_sensors, (vi + 1) * n_sensors
        pred_phys[:, c0:c1] = pred_test_flat[:, c0:c1] * out_std[vi] + out_mean[vi]

    y_true = ds["y_test_raw"]
    y_persist = ds["x_test_raw"]
    clim_vm = ds["climatology"].T.reshape(-1)

    print("\n=== U-Net + soft-FSS loss -- single-step test-set summary (Table 1 columns) ===")
    print(f"params={n_params:,}")
    metrics = {"n_params": n_params, "n_train": int(ds["train_idx"].shape[0]),
               "n_test": int(ds["test_idx"].shape[0]), "subsample_r": args.subsample_r,
               "fss_weight": args.fss_weight, "fss_highpass_k": args.fss_highpass_k,
               "fss_window": args.fss_window, "fss_threshold_std": args.fss_threshold_std,
               "fss_softness": args.fss_softness, "learning_rate": args.learning_rate,
               "seed": args.seed, "iterations": step, "best_val_loss": best_val}
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
