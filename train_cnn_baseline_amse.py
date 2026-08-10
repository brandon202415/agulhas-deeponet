#!/usr/bin/env python3
"""CNN baseline (train_cnn_baseline.py's UNetForecast, unchanged) trained with
an added spectrally-adjusted MSE (AMSE) auxiliary loss on the SSH (zos)
field, adapting Subich et al. 2025 (ICML; "Fixing the Double Penalty in
Data-Driven Weather Forecasting Through a Modified Spherical Harmonic Loss
Function", arXiv 2501.19374) to this project's regional Cartesian grid.

Background: item 24/25 in RESEARCH_LOG.md tested a differentiable
neighborhood-based (soft-FSS) auxiliary loss as a training-objective fix for
the double-penalty problem (item 23: the CNN's massive grid-skill win over
persistence still doesn't produce a statistically significant eddy-tracking
improvement). That result did not replicate across seeds (item 25). This
script is the second training-objective attempt (item 26's decision "C"),
using a different, published mechanism.

The paper's method (fine-tuning GraphCast, a global lat/lon model): decompose
latitude-weighted MSE via a spherical-harmonic transform. By Parseval's
theorem this is an EXACT decomposition of pixelwise MSE into, per total
wavenumber k:
    MSE = sum_k [ (sqrt(PSD_k(x)) - sqrt(PSD_k(y)))^2                    (amplitude term)
                + 2*sqrt(PSD_k(x)*PSD_k(y))*(1 - Coh_k(x,y)) ]            (decorrelation term)
where PSD_k is power spectral density at wavenumber k and Coh_k is spectral
coherence (normalized cross-spectrum). A model can cheat on the
decorrelation term by suppressing its own spectral amplitude toward zero
(blurring) -- as PSD_pred(k) -> 0 the geometric-mean prefactor
sqrt(PSD_pred*PSD_true) -> 0 too, so the decorrelation penalty vanishes even
though the model produced no genuine structure at that scale. This is the
"double penalty" smoothing mechanism in spectral form. Their fix,
Spectrally Adjusted MSE (AMSE), replaces the geometric-mean prefactor with
max(PSD_pred(k), PSD_true(k)): now suppressing amplitude cannot shrink the
decorrelation penalty, because the *true* field's power is still there
providing the floor.

Adaptation to this project: the spherical-harmonic transform is a
consequence of GraphCast being a *global* lat/lon model, not the
load-bearing part of the mechanism -- the underlying idea is a spectral
decomposition into per-wavenumber amplitude and coherence, then reweighting
to remove the incentive to suppress amplitude. For this project's regional
Cartesian SSH grid, the natural substitute basis is a 2D FFT (ortho-
normalized, so Parseval's theorem holds exactly, mirroring the spherical
harmonics' orthonormality) with power/cross-spectra binned into radial
wavenumber shells |k| = sqrt(kx^2+ky^2), in place of grouping by total
spherical-harmonic wavenumber. Applied to the zos channel only (the field
py-eddy-tracker's contour identification operates on), matching item 24's
scope for direct comparability.

Verified before use (see `_selftest()` below, run automatically at import):
the spectral amplitude+decorrelation decomposition exactly reconstructs the
literal pixelwise MSE on a random synthetic field (Parseval's theorem check)
-- the same "verify before trusting" discipline as this project's
`test_physics_losses_synthetic.py` (Issue 9's audit theme). `--amse-weight 0`
reproduces train_cnn_baseline.py's plain training exactly (no new
parameters, only an added loss term multiplied by zero).

Departure from the paper, stated plainly: Subich et al. use AMSE as the
*entire* training loss when fine-tuning GraphCast; this script instead adds
it as a weighted auxiliary term on top of the existing pointwise MSE,
matching this project's own convention (physics losses, soft-FSS) of testing
an addition against a `--*-weight 0` control for a directly comparable,
controlled ablation. A boundary/non-periodicity caveat also applies: this
regional Cartesian window is not periodic the way GraphCast's global sphere
is, so the FFT implicitly assumes periodic boundary conditions the true
field does not have; land is zeroed (as elsewhere in this study) rather than
windowed/tapered, so some spectral leakage at land-ocean and domain-edge
boundaries is expected. This leakage affects prediction and truth
identically (same geometry/mask), so it should not bias the coherence
comparison, but it does mean this is a spectrally-motivated adaptation of
the mechanism, not a byte-exact port of the paper's exact-Parseval-on-a-
sphere setting.

Usage:
    python3 train_cnn_baseline_amse.py --nc data/agulhas_prototype.nc \\
        --subsample-r 6 --iterations 800 --amse-weight 1.0 \\
        --out-dir results/cnn_amse_r6_local
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
from train_cnn_baseline import UNetForecast, to_grid, count_params, rollout_evaluate_cnn, chunked_forward

ZOS_IDX = VARIABLES.index("zos")


def build_radial_bins(nlat, nlon, n_bins, device):
    """Precompute a [n_bins, nlat, nlon] one-hot membership tensor grouping
    each 2D FFT frequency cell into a radial-wavenumber shell, the Cartesian
    analog of grouping spherical-harmonic coefficients by total wavenumber."""
    ky = torch.fft.fftfreq(nlat, device=device).view(nlat, 1)
    kx = torch.fft.fftfreq(nlon, device=device).view(1, nlon)
    kmag = torch.sqrt(kx ** 2 + ky ** 2)  # [nlat, nlon]
    kmax = kmag.max()
    edges = torch.linspace(0.0, float(kmax), n_bins + 1, device=device)
    edges[-1] = edges[-1] + 1e-8  # include the Nyquist corner in the last bin
    bin_idx = torch.bucketize(kmag, edges[1:-1], right=False)  # [nlat, nlon], in [0, n_bins-1]
    onehot = F.one_hot(bin_idx, num_classes=n_bins).permute(2, 0, 1).to(torch.float32)  # [n_bins, nlat, nlon]
    return onehot


def radial_power_and_cross(pred, true, bin_onehot):
    """pred, true: [N,1,nlat,nlon] real. Returns PSD_pred, PSD_true, cross
    (real part of pred . conj(true) spectrum), each [N,1,n_bins], via an
    ortho-normalized 2D FFT (Parseval-exact) binned into radial shells."""
    Xp = torch.fft.fft2(pred, norm="ortho")
    Xt = torch.fft.fft2(true, norm="ortho")
    power_p = Xp.real ** 2 + Xp.imag ** 2           # [N,1,nlat,nlon]
    power_t = Xt.real ** 2 + Xt.imag ** 2
    cross = Xp.real * Xt.real + Xp.imag * Xt.imag    # Re(Xp * conj(Xt))
    psd_p = torch.einsum("bhw,nchw->ncb", bin_onehot, power_p)
    psd_t = torch.einsum("bhw,nchw->ncb", bin_onehot, power_t)
    cr = torch.einsum("bhw,nchw->ncb", bin_onehot, cross)
    return psd_p, psd_t, cr


def amse_loss_term(pred_zos, true_zos, bin_onehot, eps=1e-12):
    psd_p, psd_t, cr = radial_power_and_cross(pred_zos, true_zos, bin_onehot)
    coh = cr / torch.sqrt(psd_p * psd_t + eps)
    amp_term = (torch.sqrt(psd_p + eps) - torch.sqrt(psd_t + eps)) ** 2
    decorr_term = 2.0 * torch.maximum(psd_p, psd_t) * (1.0 - coh)
    return (amp_term + decorr_term).sum(dim=-1).mean()


def _selftest():
    """Parseval check: the spectral amplitude+decorrelation decomposition
    (Sec. 2.2 of Subich et al. 2025) must exactly reconstruct the literal
    pixelwise MSE on an arbitrary field pair, before AMSE (Sec. 2.3's
    reweighted variant) is trusted for anything. Run at import time."""
    torch.manual_seed(0)
    nlat, nlon, n_bins = 17, 23, 9
    x = torch.randn(2, 1, nlat, nlon, dtype=torch.float64)
    y = torch.randn(2, 1, nlat, nlon, dtype=torch.float64)
    onehot = build_radial_bins(nlat, nlon, n_bins, device="cpu").to(torch.float64)

    pixel_mse = ((x - y) ** 2).sum(dim=(-2, -1)).mean()

    psd_x, psd_y, cr = radial_power_and_cross(x, y, onehot)
    coh = cr / torch.sqrt(psd_x * psd_y + 1e-30)
    spectral_mse = ((torch.sqrt(psd_x) - torch.sqrt(psd_y)) ** 2
                     + 2 * torch.sqrt(psd_x * psd_y) * (1 - coh)).sum(dim=-1).mean()

    err = abs(float(pixel_mse) - float(spectral_mse))
    assert err < 1e-8, (
        f"Parseval check FAILED: pixel MSE={float(pixel_mse):.10f} vs. "
        f"spectral decomposition={float(spectral_mse):.10f} (diff {err:.2e}) "
        f"-- the ortho-FFT radial-binning implementation does not exactly "
        f"decompose MSE; do not trust amse_loss_term until this is fixed."
    )


_selftest()


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
    ap.add_argument("--amse-weight", type=float, default=0.0,
                     help="weight on the spectrally-adjusted MSE auxiliary loss (zos channel "
                          "only, Subich et al. 2025 adapted via 2D FFT). 0.0 reproduces "
                          "train_cnn_baseline.py's plain pointwise-MSE training exactly.")
    ap.add_argument("--amse-bins", type=int, default=16,
                     help="number of radial-wavenumber shells the 2D FFT spectrum is binned into "
                          "(Cartesian analog of grouping by total spherical-harmonic wavenumber).")
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
    print(f"U-Net + AMSE spectral loss: {n_params:,} trainable parameters (identical to plain "
          f"baseline -- no new parameters, only an added loss term)")
    print(f"  Train / Val / Test    : {len(ds['train_idx'])} / "
          f"{len(ds['val_idx'])} / {len(ds['test_idx'])}  (chronological)")
    print(f"  amse-weight={args.amse_weight}  amse-bins={args.amse_bins}")

    bin_onehot = build_radial_bins(nlat_s, nlon_s, args.amse_bins, device=device)

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
        if args.amse_weight != 0.0:
            pred_zos = pred[:, ZOS_IDX:ZOS_IDX + 1] * ocean_mask
            true_zos = yb[:, ZOS_IDX:ZOS_IDX + 1] * ocean_mask
            amse = amse_loss_term(pred_zos, true_zos, bin_onehot)
        else:
            amse = torch.zeros((), device=device)
        loss = base_loss + args.amse_weight * amse
        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % eval_every == 0 or step == args.iterations:
            model.eval()
            with torch.no_grad():
                pv = chunked_forward(model, x_val)
                verr2 = (pv - y_val) ** 2 * ocean_mask
                vloss = (verr2.sum() / (ocean_mask.sum() * n_vars * x_val.shape[0])).item()
            print(f"step {step:6d}  train_loss {loss.item():.5f}  "
                  f"(base {base_loss.item():.5f}  amse {amse.item():.5f})  val_loss {vloss:.5f}")
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
        pred_test_norm = chunked_forward(model, x_test).cpu().numpy()
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

    print("\n=== U-Net + AMSE spectral loss -- single-step test-set summary (Table 1 columns) ===")
    print(f"params={n_params:,}")
    metrics = {"n_params": n_params, "n_train": int(ds["train_idx"].shape[0]),
               "n_test": int(ds["test_idx"].shape[0]), "subsample_r": args.subsample_r,
               "amse_weight": args.amse_weight, "amse_bins": args.amse_bins,
               "learning_rate": args.learning_rate,
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
