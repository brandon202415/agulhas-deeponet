#!/usr/bin/env python3
"""Resolution-augmented DeepONet training + a properly-fixed zero-shot
resolution-transfer test (Sec. 3.7 follow-up, "positive DeepONet result" track).

Sec. 3.7 found a real but narrow zero-shot skill (+0.022) when querying an
r=6-trained model's trunk at r=3 coordinates -- but only via a manual forward-
pass workaround, because the model's persistence-residual skip was hard-locked
to the branch's own sensor count (train_agulhas_deeponet_prototype.py's
MultivarDeepONet.forward, now fixed to accept an explicit persist_at_query
tensor so the residual can be supplied at the query resolution).

This script tests whether (a) that architectural fix alone, and (b) training
with resolution augmentation -- mixing standard r=6-query steps with steps that
query a random subset of r=3 points (finer, partially unseen during training)
using the r=3 field's own persistence as the residual -- produces a stronger,
more genuine discretization-invariance result than the original single-
resolution-trained workaround.

Branch input is ALWAYS r=6 (fixed "sensor" resolution, representing observed
state); only the TRUNK query set and its matching residual/target vary between
r=6 and r=3 during training. Both resolutions come from the same source file
(load_states with different subsample_r), so time indices align exactly and
the r=6/r=3 train/val/test day split is shared.

Local-prototype scale (n_train=700), exploratory. Not yet run at full scale.
"""
import argparse
from pathlib import Path

import numpy as np
import torch

from train_agulhas_deeponet_prototype import (
    load_states, build_dataset, MultivarDeepONet, to_tensor, VARIABLES, save_json,
)

torch.set_num_threads(4)


def build_r3_aux(nc_path, cache_r3, subsample_r3, out_mean, out_std, t_min, t_span,
                  train_idx, val_idx, test_idx, step_days=1):
    """Build r=3 trunk (normalised with r=6's t_min/t_span) and per-day
    normalised target/persistence grids [N, n_vars, n_sensors3], using r=6's
    per-variable out_mean/out_std (resolution-independent scalars)."""
    lon3, lat3, states3 = load_states(nc_path, subsample_r=subsample_r3, cache=cache_r3)
    T, nlat3, nlon3, n_vars = states3.shape
    n_sensors3 = nlat3 * nlon3
    k = step_days

    next3 = states3[k:].astype(np.float64)     # [N, nlat3, nlon3, n_vars] raw, target
    curr3 = states3[:-k].astype(np.float64)     # [N, nlat3, nlon3, n_vars] raw, persistence

    next3_flat = next3.reshape(next3.shape[0], n_sensors3, n_vars)   # [N, n_sensors3, n_vars]
    curr3_flat = curr3.reshape(curr3.shape[0], n_sensors3, n_vars)

    next3_norm = (next3_flat - out_mean[None, None, :]) / out_std[None, None, :]
    curr3_norm = (curr3_flat - out_mean[None, None, :]) / out_std[None, None, :]
    # -> [N, n_vars, n_sensors3]
    next3_norm = next3_norm.transpose(0, 2, 1).astype(np.float32)
    curr3_norm = curr3_norm.transpose(0, 2, 1).astype(np.float32)

    ocean3 = next3_flat[train_idx, :, 0].std(axis=0) > 1e-4  # [n_sensors3] bool, zos-based

    LAT3, LON3 = np.meshgrid(lat3, lon3, indexing="ij")
    trunk3_raw = np.stack([LON3.ravel(), LAT3.ravel()], axis=-1).astype(np.float64)
    trunk3_norm = (2.0 * (trunk3_raw - t_min) / t_span - 1.0).astype(np.float32)

    return {
        "lon3": lon3, "lat3": lat3, "n_sensors3": n_sensors3,
        "trunk3_norm": trunk3_norm, "ocean3": ocean3,
        "next3_norm": next3_norm, "curr3_norm": curr3_norm,
        "train_idx": train_idx, "val_idx": val_idx, "test_idx": test_idx,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nc", default="data/agulhas_prototype.nc")
    ap.add_argument("--cache-r6", default="data/cache_r6_local.npz")
    ap.add_argument("--cache-r3", default="data/cache_r3_local.npz")
    ap.add_argument("--iterations", type=int, default=5000)
    ap.add_argument("--learning-rate", type=float, default=3e-4)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--aug-prob", type=float, default=0.5,
                     help="fraction of steps that query r=3 points instead of r=6")
    ap.add_argument("--aug-n-query", type=int, default=6161,
                     help="number of r=3 points sampled per augmented step")
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--val-fraction", type=float, default=0.15)
    ap.add_argument("--test-fraction", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = "cpu"

    lon6, lat6, states6 = load_states(args.nc, subsample_r=6, cache=args.cache_r6)
    ds6 = build_dataset(states6, lon6, lat6, test_fraction=args.test_fraction,
                         val_fraction=args.val_fraction, step_days=1)
    n_sensors6, nlat6, nlon6, n_vars = ds6["n_sensors"], ds6["nlat_s"], ds6["nlon_s"], ds6["n_vars"]
    out_mean, out_std = ds6["out_mean"], ds6["out_std"]

    LAT6, LON6 = np.meshgrid(lat6, lon6, indexing="ij")
    trunk6_raw = np.stack([LON6.ravel(), LAT6.ravel()], axis=-1).astype(np.float64)
    t_min = trunk6_raw.min(axis=0, keepdims=True)
    t_span = trunk6_raw.max(axis=0, keepdims=True) - t_min

    aux3 = build_r3_aux(args.nc, args.cache_r3, 3, out_mean, out_std, t_min, t_span,
                         ds6["train_idx"], ds6["val_idx"], ds6["test_idx"])
    n_sensors3 = aux3["n_sensors3"]
    print(f"r=6: {nlat6}x{nlon6}={n_sensors6} sensors (branch, always)  |  "
          f"r=3: {len(aux3['lat3'])}x{len(aux3['lon3'])}={n_sensors3} points (aug query pool)")

    model = MultivarDeepONet(
        d_branch=n_sensors6 * n_vars, n_sensors=n_sensors6, n_vars=n_vars,
        branch_width=64, branch_depth=2, trunk_width=64, trunk_depth=2, latent_dim=32,
    ).to(device)

    def T(a):
        return torch.tensor(a, dtype=torch.float32, device=device)

    branch_train = T(ds6["branch_train"])
    branch_val = T(ds6["branch_val"])
    trunk6 = T(ds6["trunk"])
    y_train6 = T(ds6["y_train_norm"])
    y_val6 = T(ds6["y_val_norm"])
    ocean6_vm = T(np.tile(ds6["ocean_mask"], n_vars).astype(np.float32))  # [n_vars*n_sensors6]
    w6 = T(ds6["loss_weight"])  # [n_vars*n_sensors6]

    trunk3_full = T(aux3["trunk3_norm"])
    ocean3 = torch.tensor(aux3["ocean3"], dtype=torch.bool, device=device)
    next3_norm = T(aux3["next3_norm"])   # [N_total, n_vars, n_sensors3]
    curr3_norm = T(aux3["curr3_norm"])   # [N_total, n_vars, n_sensors3]
    train_idx = ds6["train_idx"]

    opt = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    n_train = branch_train.shape[0]
    best_val, best_state, bad_evals = float("inf"), None, 0
    eval_every = 100
    n_r3_steps = n_r6_steps = 0

    for step in range(1, args.iterations + 1):
        model.train()
        idx = np.random.randint(0, n_train, size=min(args.batch_size, n_train))
        idx_t = torch.tensor(idx, dtype=torch.long, device=device)
        bt = branch_train[idx_t]

        if np.random.rand() < args.aug_prob:
            n_r3_steps += 1
            qidx = np.random.choice(n_sensors3, size=args.aug_n_query, replace=False)
            qidx_t = torch.tensor(qidx, dtype=torch.long, device=device)
            trunk_q = trunk3_full[qidx_t]                                  # [Q, 2]
            # global day indices for this batch, mapped into train_idx's absolute day space
            day_idx_t = torch.tensor(train_idx[idx], dtype=torch.long, device=device)
            persist_q = curr3_norm[day_idx_t][:, :, qidx_t]                # [B, n_vars, Q]
            target_q = next3_norm[day_idx_t][:, :, qidx_t]                 # [B, n_vars, Q]
            pred = model(bt, trunk_q, persist_at_query=persist_q)          # [B, Q, n_vars]
            pred = pred.permute(0, 2, 1)                                   # [B, n_vars, Q]
            ocean_q = ocean3[qidx_t]                                       # [Q] bool
            err2 = (pred - target_q) ** 2
            loss = err2[:, :, ocean_q].mean() if ocean_q.any() else err2.mean()
        else:
            n_r6_steps += 1
            pred = model(bt, trunk6)                                       # [B, n_sensors6, n_vars]
            pred_vm = pred.permute(0, 2, 1).reshape(pred.shape[0], -1)     # [B, n_vars*n_sensors6]
            e2 = (pred_vm[:, ocean6_vm.bool()] - y_train6[idx_t][:, ocean6_vm.bool()]) ** 2
            loss = (e2 * w6[ocean6_vm.bool()]).mean()

        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % eval_every == 0 or step == args.iterations:
            model.eval()
            with torch.no_grad():
                pv = model(branch_val, trunk6)
                pv_vm = pv.permute(0, 2, 1).reshape(pv.shape[0], -1)
                verr2 = (pv_vm[:, ocean6_vm.bool()] - y_val6[:, ocean6_vm.bool()]) ** 2
                vloss = verr2.mean().item()
            print(f"step {step:6d}  train_loss {loss.item():.5f}  val_loss(r6) {vloss:.5f}  "
                  f"[r6 steps {n_r6_steps}, r3-aug steps {n_r3_steps}]")
            if vloss < best_val - 1e-6:
                best_val, best_state, bad_evals = vloss, {k: v.clone() for k, v in model.state_dict().items()}, 0
            else:
                bad_evals += 1
                if bad_evals >= args.patience:
                    print(f"Early stop at step {step} (best val_loss={best_val:.5f})")
                    break

    model.load_state_dict(best_state)
    model.eval()
    torch.save(model.state_dict(), args.out_dir / "model.pt")

    # --- Eval 1: standard r=6 test skill (sanity / comparison to original single-res training) ---
    branch_test = T(ds6["branch_test"])
    with torch.no_grad():
        pred6 = model(branch_test, trunk6).cpu().numpy()
    pred6_phys = pred6 * out_std[None, None, :] + out_mean[None, None, :]
    x_test_raw = ds6["x_test_raw"]

    print("\n=== Standard r=6 test skill (resolution-augmented training) ===")
    ocean6 = ds6["ocean_mask"]
    r6_skills = {}
    for vi, vname in enumerate(VARIABLES):
        c0, c1 = vi * n_sensors6, (vi + 1) * n_sensors6
        yt = ds6["y_test_raw"][:, c0:c1][:, ocean6]
        yp = pred6_phys[:, :, vi][:, ocean6]
        ys = x_test_raw[:, c0:c1][:, ocean6]
        rmse_m = np.sqrt(np.mean((yp - yt) ** 2))
        rmse_p = np.sqrt(np.mean((ys - yt) ** 2))
        skill = 1.0 - (rmse_m / rmse_p) ** 2 if rmse_p > 0 else float("nan")
        r6_skills[vname] = skill
        print(f"  {vname:8s} skill={skill:+.4f}")
    mean_r6_skill = float(np.mean(list(r6_skills.values())))
    print(f"  mean skill = {mean_r6_skill:+.4f}  (cf. original single-resolution r=6 training: +0.043 full-scale / see local baseline)")

    # --- Eval 2: PROPER zero-shot r=3 test skill, fixed architecture, no manual workaround ---
    day_idx_test_t = torch.tensor(ds6["test_idx"], dtype=torch.long, device=device)
    # ds6["test_idx"] indexes into the SAME absolute day space as train_idx/val_idx (0..N-1),
    # and next3_norm/curr3_norm were built over that same absolute day space, so index directly.
    persist_test3 = curr3_norm[day_idx_test_t]   # [N_test, n_vars, n_sensors3]
    target_test3 = next3_norm[day_idx_test_t]    # [N_test, n_vars, n_sensors3]

    with torch.no_grad():
        pred3 = model(branch_test, trunk3_full, persist_at_query=persist_test3)  # [N_test, n_sensors3, n_vars]
    pred3 = pred3.permute(0, 2, 1)  # [N_test, n_vars, n_sensors3]
    pred3_phys = pred3.cpu().numpy() * out_std[None, :, None] + out_mean[None, :, None]
    target3_phys = target_test3.cpu().numpy() * out_std[None, :, None] + out_mean[None, :, None]
    persist3_phys = persist_test3.cpu().numpy() * out_std[None, :, None] + out_mean[None, :, None]

    print("\n=== PROPER zero-shot r=3-query skill (fixed residual, resolution-augmented training) ===")
    ocean3_np = aux3["ocean3"]
    r3_skills = {}
    for vi, vname in enumerate(VARIABLES):
        yt = target3_phys[:, vi, :][:, ocean3_np]
        yp = pred3_phys[:, vi, :][:, ocean3_np]
        ys = persist3_phys[:, vi, :][:, ocean3_np]
        rmse_m = np.sqrt(np.mean((yp - yt) ** 2))
        rmse_p = np.sqrt(np.mean((ys - yt) ** 2))
        skill = 1.0 - (rmse_m / rmse_p) ** 2 if rmse_p > 0 else float("nan")
        r3_skills[vname] = skill
        print(f"  {vname:8s} skill={skill:+.4f}  rmse_model={rmse_m:.4f}  rmse_persist={rmse_p:.4f}")
    mean_r3_skill = float(np.mean(list(r3_skills.values())))
    print(f"  mean skill = {mean_r3_skill:+.4f}")
    print("\n(compare to Sec. 3.7's manual-workaround, single-resolution-trained result: +0.0219 mean skill)")

    metrics = {
        "aug_prob": args.aug_prob, "aug_n_query": args.aug_n_query,
        "n_r6_steps": n_r6_steps, "n_r3_steps": n_r3_steps, "best_val_loss": best_val,
        "mean_skill_r6": mean_r6_skill, "skills_r6": r6_skills,
        "mean_skill_r3_zeroshot": mean_r3_skill, "skills_r3_zeroshot": r3_skills,
    }
    save_json(args.out_dir / "metrics.json", metrics)
    print(f"\nSaved to {args.out_dir}/")


if __name__ == "__main__":
    main()
