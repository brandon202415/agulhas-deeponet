#!/usr/bin/env python3
"""Zero-shot discretization-invariance test (reviewer item: DeepONet's headline
architectural claim -- that the trunk can be queried at arbitrary continuous
coordinates -- is never actually exercised in this study; the patch branch is
tied to a fixed tile size just like a CNN kernel would be).

This loads the whole-domain r=6-trained local-prototype checkpoint, rebuilds its
EXACT branch input (r=6 sensor readings, same normalization as training) and
trunk (verified by reproducing the saved predictions.npz numerically to <1e-3),
then queries the SAME trained trunk at r=3 output coordinates -- a 2x finer
query grid than the model ever saw during training, no retraining performed.

IMPORTANT ARCHITECTURAL FINDING, discovered while building this test: the
model's persistence-residual skip (out = dot(branch,trunk) + bias + branch_input)
slices branch_input using the model's OWN fixed n_sensors (r=6's 6161), so
calling model.forward() directly with an r=3-sized trunk raises a shape
mismatch -- the residual skip as implemented is resolution-locked, not just the
trunk. This is itself informative: the mechanism that makes this architecture
beat persistence (Sec. 4.1 of the manuscript) is not, as built, discretization-
invariant, only the raw trunk-branch dot product is. To test the part that
COULD be invariant, we bypass model.forward() and manually recompute:
    learned_increment_norm = dot(branch_feats, trunk_feats(r=3 coords)) + bias
    prediction = learned_increment_norm * out_std + out_mean + r3_persistence
i.e. the learned correction queried at r=3 points, added to r=3's OWN
persistence field (using the per-variable out_mean/out_std scalars, which are
resolution-independent) rather than to r=6 sensor values. This is the
conceptually correct zero-shot formulation -- it tests whether the learned
trunk-branch increment generalizes to unseen finer coordinates, which is the
actual claim under test.

This is a local-prototype-scale proof of concept only (n_train=700), not a
full-scale validation.
"""
import sys
import numpy as np
import torch

sys.path.insert(0, ".")
from train_agulhas_deeponet_prototype import (
    load_states, build_dataset, MultivarDeepONet, to_tensor, VARIABLES,
)

CKPT = "results/whole_r6_local/model.pt"
NC_PATH = "data/agulhas_prototype.nc"
R6_PRED = "results/whole_r6_local/predictions.npz"
R3_PRED = "results/whole_r3_local/predictions.npz"


def main():
    device = "cpu"

    # --- Rebuild the exact r=6 dataset used for training/eval ---
    lon6, lat6, states6 = load_states(NC_PATH, subsample_r=6, cache="data/cache_r6_local.npz")
    ds6 = build_dataset(states6, lon6, lat6, test_fraction=0.15, val_fraction=0.15, step_days=1)
    n_sensors6 = len(lon6) * len(lat6)
    n_vars = len(VARIABLES)

    print(f"r=6 grid: {len(lon6)} x {len(lat6)} = {n_sensors6} sensors")
    print(f"test_idx range: {ds6['test_idx'][0]}..{ds6['test_idx'][-1]} "
          f"(should match metrics.json test_cases 850..999)")

    # --- Load model ---
    model = MultivarDeepONet(
        d_branch=n_sensors6 * n_vars, n_sensors=n_sensors6, n_vars=n_vars,
        branch_width=64, branch_depth=2, trunk_width=64, trunk_depth=2,
        latent_dim=32,
    ).to(device)
    state_dict = torch.load(CKPT, map_location=device, weights_only=False)
    model.load_state_dict(state_dict)
    model.eval()

    branch_test = to_tensor(ds6["branch_test"], device)
    trunk6 = to_tensor(ds6["trunk"], device)

    # --- Sanity check: reproduce the saved r=6 predictions exactly ---
    with torch.no_grad():
        pred_norm = model(branch_test, trunk6).cpu().numpy()  # [N_test, n_sensors, n_vars]
    out_mean, out_std = ds6["out_mean"], ds6["out_std"]
    pred_phys = pred_norm * out_std[None, None, :] + out_mean[None, None, :]
    pred_flat_vm = pred_phys.transpose(0, 2, 1).reshape(pred_phys.shape[0], -1)

    saved = np.load(R6_PRED)
    saved_pred = saved["y_pred"]
    max_abs_diff = np.max(np.abs(pred_flat_vm - saved_pred))
    print(f"Sanity check -- max|reconstructed r=6 pred - saved r=6 pred| = {max_abs_diff:.6g} "
          f"(should be ~0; confirms exact reproduction before the novel step)")
    assert max_abs_diff < 1e-3, "reconstruction does not match saved predictions -- do not trust r=3 result"

    # --- Confirm the naive model.forward() call breaks on a differently-sized trunk ---
    r3 = np.load(R3_PRED)
    lon3, lat3 = r3["lon"].astype("float64"), r3["lat"].astype("float64")
    n_sensors3 = len(lon3) * len(lat3)
    LAT3, LON3 = np.meshgrid(lat3, lon3, indexing="ij")
    trunk3_raw = np.stack([LON3.ravel(), LAT3.ravel()], axis=-1).astype(np.float64)

    LAT6, LON6 = np.meshgrid(lat6, lon6, indexing="ij")
    trunk6_raw = np.stack([LON6.ravel(), LAT6.ravel()], axis=-1).astype(np.float64)
    t_min = trunk6_raw.min(axis=0, keepdims=True)
    t_span = trunk6_raw.max(axis=0, keepdims=True) - t_min
    t_min3 = trunk3_raw.min(axis=0, keepdims=True)
    t_span3 = trunk3_raw.max(axis=0, keepdims=True) - t_min3
    print(f"\nr=6 trunk bounds: min={t_min.ravel()} span={t_span.ravel()}")
    print(f"r=3 trunk bounds: min={t_min3.ravel()} span={t_span3.ravel()} (matches r=6, as expected -- same domain)")

    trunk3_norm = (2.0 * (trunk3_raw - t_min) / t_span - 1.0).astype(np.float32)
    trunk3 = to_tensor(trunk3_norm, device)
    print(f"r=3 grid: {len(lon3)} x {len(lat3)} = {n_sensors3} query points")

    try:
        with torch.no_grad():
            _ = model(branch_test, trunk3)
        print("model.forward() with r=3 trunk did NOT raise (unexpected) -- inspect residual logic")
    except RuntimeError as e:
        print(f"\nConfirmed: naive model.forward(branch_r6, trunk_r3) raises a shape error:\n  {e}")
        print("  -> the persistence-residual skip is hard-locked to the model's own n_sensors,")
        print("     so the architecture as trained/implemented is NOT drop-in discretization-invariant.")

    # --- Manually recompute the part that COULD be invariant: the learned increment ---
    with torch.no_grad():
        trunk_feats3 = model.trunk(trunk3)  # [n_sensors3, latent_dim]
        incs = []
        for vi in range(n_vars):
            branch_feats = model.branches[vi](branch_test)          # [N_test, latent_dim]
            raw = torch.einsum("np,sp->ns", branch_feats, trunk_feats3)  # [N_test, n_sensors3]
            raw = raw + model.biases[vi]
            incs.append(raw)
        increment_norm = torch.stack(incs, dim=-1).cpu().numpy()  # [N_test, n_sensors3, n_vars]

    increment_phys = increment_norm * out_std[None, None, :]  # per-variable scalar, resolution-independent
    persist3_grid = r3["y_persist"].reshape(r3["y_persist"].shape[0], n_vars, n_sensors3).transpose(0, 2, 1)
    pred3_phys = persist3_grid + increment_phys
    pred3_flat_vm = pred3_phys.transpose(0, 2, 1).reshape(pred3_phys.shape[0], -1)

    true3 = r3["y_true"]
    persist3 = r3["y_persist"]
    print(f"\nzero-shot r=3 prediction (persistence + zero-shot learned increment) finite: "
          f"{np.isfinite(pred3_flat_vm).all()}")

    print("\n=== Zero-shot r=3-query skill (r=6-trained model's learned increment, "
          "queried at r=3 coords, added to r=3 persistence, NO retraining) ===")
    skills = []
    for vi, vname in enumerate(VARIABLES):
        c0, c1 = vi * n_sensors3, (vi + 1) * n_sensors3
        yt, yp, ys = true3[:, c0:c1], pred3_flat_vm[:, c0:c1], persist3[:, c0:c1]
        ocean = yt.std(axis=0) > 1e-4
        rmse_model = np.sqrt(np.mean((yp[:, ocean] - yt[:, ocean]) ** 2))
        rmse_persist = np.sqrt(np.mean((ys[:, ocean] - yt[:, ocean]) ** 2))
        skill = 1.0 - (rmse_model / rmse_persist) ** 2 if rmse_persist > 0 else float("nan")
        skills.append(skill)
        print(f"  {vname:8s} skill={skill:+.4f}  rmse_model={rmse_model:.4f}  rmse_persist={rmse_persist:.4f}")
    print(f"  mean skill = {np.mean(skills):+.4f}")
    print("\n(compare to the r=6-trained model's own r=6 mean skill of +0.043, Table 1;"
          " a positive number here -- even if smaller -- would mean the learned increment")
    print(" partially transfers to unseen query resolution; ~0 or negative means it does not.)")


if __name__ == "__main__":
    main()
