#!/usr/bin/env python3
"""Patch-based continuous lead-time DeepONet: combines the patch tiling
architecture (one small shared DeepONet applied to overlapping local tiles,
Sec. 3.6) with the (lon, lat, k) trunk from the whole-domain continuous
lead-time experiment -- trained on step_days in {1,5,10}, evaluated zero-shot
at unseen intermediate horizons k=3,7.

Reuses ContinuousLeadTimeDeepONet directly, sized for a single tile instead of
the whole domain -- exactly mirroring how train_agulhas_deeponet_patch.py
resizes MultivarDeepONet for a tile. Reuses enumerate_tiles/tile_indices from
that same script unchanged.

Deliberately matches the patch trainer's OWN proven training recipe first
(plain masked MSE, no variability/skill-aligned loss weighting) rather than
assuming the fix that mattered for the whole-domain continuous lead-time
experiment applies unchanged here -- the existing, validated patch trainer
gets its +0.075-0.079 headline result with plain MSE, so that is the more
faithful starting point for this combination. If training shows the same
flat/noisy signature diagnosed yesterday, that would be evidence weighting is
needed here too, not assumed in advance.
"""
import argparse
import copy

import numpy as np
import torch

from train_agulhas_deeponet_prototype import load_states, _var_major_flat, to_tensor, VARIABLES, save_json
from train_agulhas_deeponet_patch import enumerate_tiles, tile_indices
from train_deeponet_continuous_leadtime import ContinuousLeadTimeDeepONet, build_shared_split, make_pair, k_to_norm

torch.set_num_threads(4)


def masked_mse(pred, target, ocean_vm):
    return torch.mean((pred[:, ocean_vm] - target[:, ocean_vm]) ** 2)


def train_patch_model(states, train_idx, val_idx, ks, tiles, tile_meta, ocean_mask,
                       out_mean, out_std, b_mean, b_std, land_vm, n_sensors, n_vars,
                       trunk6_raw_norm, k_min, k_max, th, tw, iterations, lr, batch_size,
                       patience, device, seed, label, display_every=200):
    rng = np.random.default_rng(seed)
    model = ContinuousLeadTimeDeepONet(
        d_branch=n_vars * th * tw, n_sensors=th * tw, n_vars=n_vars,
        branch_width=64, branch_depth=2, trunk_width=64, trunk_depth=2, latent_dim=32,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best_val, best_state, bad = float("inf"), copy.deepcopy(model.state_dict()), 0

    for step in range(1, iterations + 1):
        model.train()
        k = ks[rng.integers(0, len(ks))]
        tile_i = rng.integers(0, len(tiles))
        sensor_idx, target_cols, branch_cols = tile_meta[tile_i]  # n_history_days=1 => branch_cols==target_cols
        idx = train_idx[rng.integers(0, len(train_idx), size=min(batch_size, len(train_idx)))]

        branch_full, target_full, _ = make_pair(states, idx, k, out_mean, out_std, ocean_mask,
                                                  b_mean, b_std, land_vm, n_sensors, n_vars)
        b = to_tensor(branch_full[:, branch_cols], device)
        y = to_tensor(target_full[:, target_cols], device)
        kn = k_to_norm(k, k_min, k_max)
        trunk_tile = np.concatenate([trunk6_raw_norm[sensor_idx],
                                      np.full((len(sensor_idx), 1), kn, dtype=np.float32)], axis=1)
        trunk_tile_t = to_tensor(trunk_tile, device)
        tile_ocean_vm = to_tensor(np.tile(ocean_mask[sensor_idx], n_vars), device).bool()

        pred = model(b, trunk_tile_t)
        pred_vm = pred.permute(0, 2, 1).reshape(pred.shape[0], -1)
        loss = masked_mse(pred_vm, y, tile_ocean_vm)
        opt.zero_grad(); loss.backward(); opt.step()

        if step % display_every == 0 or step == iterations:
            model.eval()
            vlosses = []
            with torch.no_grad():
                for vk in ks:
                    vbranch_full, vtarget_full, _ = make_pair(states, val_idx, vk, out_mean, out_std, ocean_mask,
                                                               b_mean, b_std, land_vm, n_sensors, n_vars)
                    vkn = k_to_norm(vk, k_min, k_max)
                    for sensor_idx_v, target_cols_v, branch_cols_v in tile_meta:
                        bv = to_tensor(vbranch_full[:, branch_cols_v], device)
                        yv = to_tensor(vtarget_full[:, target_cols_v], device)
                        tv = np.concatenate([trunk6_raw_norm[sensor_idx_v],
                                              np.full((len(sensor_idx_v), 1), vkn, dtype=np.float32)], axis=1)
                        tv_t = to_tensor(tv, device)
                        tile_ocean_vm_v = to_tensor(np.tile(ocean_mask[sensor_idx_v], n_vars), device).bool()
                        pv = model(bv, tv_t)
                        pv_vm = pv.permute(0, 2, 1).reshape(pv.shape[0], -1)
                        vlosses.append(masked_mse(pv_vm, yv, tile_ocean_vm_v).item())
            vloss = float(np.mean(vlosses))
            print(f"  [{label}] step {step:5d}  train_loss {loss.item():.5f}  val_loss(avg over {ks}, all tiles) {vloss:.5f}")
            if vloss < best_val - 1e-6:
                best_val, best_state, bad = vloss, copy.deepcopy(model.state_dict()), 0
            else:
                bad += 1
                if bad >= patience:
                    print(f"  [{label}] early stop at step {step} (best val_loss={best_val:.5f})")
                    break
    model.load_state_dict(best_state)
    return model


def evaluate_patch_skill(model, states, idx, k, tiles, tile_meta, ocean_mask, out_mean, out_std,
                          b_mean, b_std, land_vm, n_sensors, n_vars, trunk6_raw_norm, k_min, k_max,
                          th, tw, device):
    """Reconstruct the full grid by predicting every tile and overlap-averaging
    (same procedure as train_agulhas_deeponet_patch.py's test-time evaluation),
    then compute per-variable skill vs. persistence."""
    branch_full, target_full, cur_full = make_pair(states, idx, k, out_mean, out_std, ocean_mask,
                                                     b_mean, b_std, land_vm, n_sensors, n_vars)
    N = len(idx)
    kn = k_to_norm(k, k_min, k_max)
    accum = np.zeros((N, n_sensors, n_vars), dtype=np.float64)
    counts = np.zeros((n_sensors,), dtype=np.float64)
    model.eval()
    with torch.no_grad():
        for sensor_idx_v, target_cols_v, branch_cols_v in tile_meta:
            b = to_tensor(branch_full[:, branch_cols_v], device)
            tv = np.concatenate([trunk6_raw_norm[sensor_idx_v],
                                  np.full((len(sensor_idx_v), 1), kn, dtype=np.float32)], axis=1)
            tv_t = to_tensor(tv, device)
            pred_norm = model(b, tv_t).cpu().numpy()  # [N, th*tw, n_vars]
            pred_raw = pred_norm.copy()
            for vi in range(n_vars):
                pred_raw[:, :, vi] = pred_norm[:, :, vi] * out_std[vi] + out_mean[vi]
            accum[:, sensor_idx_v, :] += pred_raw
            counts[sensor_idx_v] += 1
    safe_counts = np.where(counts > 0, counts, 1.0)
    pred_3d = accum / safe_counts[None, :, None]

    nxt_raw = _var_major_flat(states[idx + k].astype(np.float64))
    skills, rmses_model, rmses_persist = {}, {}, {}
    for vi, vname in enumerate(VARIABLES):
        c0, c1 = vi * n_sensors, (vi + 1) * n_sensors
        yt = nxt_raw[:, c0:c1][:, ocean_mask]
        yp = pred_3d[:, :, vi][:, ocean_mask]
        ys = cur_full[:, c0:c1][:, ocean_mask]
        rmse_m = np.sqrt(np.mean((yp - yt) ** 2))
        rmse_p = np.sqrt(np.mean((ys - yt) ** 2))
        skills[vname] = 1.0 - (rmse_m / rmse_p) ** 2 if rmse_p > 0 else float("nan")
        rmses_model[vname] = float(rmse_m)
        rmses_persist[vname] = float(rmse_p)
    return float(np.mean(list(skills.values()))), skills, rmses_model, rmses_persist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nc", default="data/agulhas_prototype.nc")
    ap.add_argument("--cache", default="data/cache_r6_local.npz")
    ap.add_argument("--patch-h", type=int, default=20)
    ap.add_argument("--patch-w", type=int, default=20)
    ap.add_argument("--patch-stride", type=int, default=10)
    ap.add_argument("--iterations", type=int, default=8000)
    ap.add_argument("--dedicated-iterations", type=int, default=8000)
    ap.add_argument("--learning-rate", type=float, default=3e-4)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out-dir", default="results/patch_continuous_leadtime")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")

    lon6, lat6, states6 = load_states(args.nc, subsample_r=6, cache=args.cache)
    T, nlat, nlon, n_vars = states6.shape
    n_sensors = nlat * nlon
    TRAIN_KS = [1, 5, 10]
    EVAL_KS = [3, 7]
    K_MIN, K_MAX = 1, 10

    train_idx, val_idx, test_idx = build_shared_split(states6, max_k=max(TRAIN_KS + EVAL_KS))
    print(f"shared split: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)} start-days")

    next1 = _var_major_flat(states6[train_idx + 1].astype(np.float64))
    ocean_mask = next1[:, :n_sensors].std(axis=0) > 1e-4
    out_mean = np.zeros(n_vars); out_std = np.ones(n_vars)
    for vi in range(n_vars):
        c0, c1 = vi * n_sensors, (vi + 1) * n_sensors
        block = next1[:, c0:c1]
        ocean = block.std(axis=0) > 1e-4
        vals = block[:, ocean] if ocean.any() else block
        out_mean[vi], out_std[vi] = vals.mean(), max(vals.std(), 1e-12)
    b_mean = np.repeat(out_mean, n_sensors)[None, :].astype(np.float32)
    b_std = np.repeat(out_std, n_sensors)[None, :].astype(np.float32)
    land_vm = np.tile(~ocean_mask, n_vars)

    LAT, LON = np.meshgrid(lat6, lon6, indexing="ij")
    trunk_raw = np.stack([LON.ravel(), LAT.ravel()], axis=-1).astype(np.float64)
    t_min = trunk_raw.min(axis=0, keepdims=True)
    t_span = trunk_raw.max(axis=0, keepdims=True) - t_min
    trunk6_raw_norm = (2.0 * (trunk_raw - t_min) / t_span - 1.0).astype(np.float32)

    all_tiles = enumerate_tiles(nlat, nlon, args.patch_h, args.patch_w, args.patch_stride, args.patch_stride)
    th, tw = min(args.patch_h, nlat), min(args.patch_w, nlon)
    all_tile_meta = [tile_indices(t, nlat, nlon, n_vars, n_sensors, n_history_days=1) for t in all_tiles]
    keep = [i for i, (sidx, _, _) in enumerate(all_tile_meta) if ocean_mask[sidx].any()]
    tiles = [all_tiles[i] for i in keep]
    tile_meta = [all_tile_meta[i] for i in keep]
    print(f"Tiling: {len(tiles)} tiles of {th}x{tw} (dropped {len(all_tiles)-len(tiles)} all-land)")

    def report(mean_sk, skills, rmses_model, rmses_persist, k, tag):
        print(f"  k={k:2d}  {tag}  mean skill = {mean_sk:+.4f}")
        print(f"        {'var':>8s} {'rmse_model':>11s} {'rmse_persist':>13s}")
        for vname in VARIABLES:
            print(f"        {vname:>8s} {rmses_model[vname]:11.4f} {rmses_persist[vname]:13.4f}")

    print(f"\n=== Training PATCH continuous lead-time model on k in {TRAIN_KS} ===")
    cont_model = train_patch_model(states6, train_idx, val_idx, TRAIN_KS, tiles, tile_meta, ocean_mask,
                                    out_mean, out_std, b_mean, b_std, land_vm, n_sensors, n_vars,
                                    trunk6_raw_norm, K_MIN, K_MAX, th, tw, args.iterations,
                                    args.learning_rate, args.batch_size, args.patience, device,
                                    args.seed, "patch-continuous")
    torch.save(cont_model.state_dict(), args.out_dir + "_continuous_model.pt")

    print("\n=== Patch continuous model: skill at TRAINED horizons (test set) ===")
    trained_skills, trained_rmse = {}, {}
    for k in TRAIN_KS:
        mean_sk, skills, rmses_model, rmses_persist = evaluate_patch_skill(
            cont_model, states6, test_idx, k, tiles, tile_meta, ocean_mask,
            out_mean, out_std, b_mean, b_std, land_vm, n_sensors, n_vars,
            trunk6_raw_norm, K_MIN, K_MAX, th, tw, device)
        trained_skills[k] = mean_sk
        trained_rmse[k] = {"model": rmses_model, "persist": rmses_persist}
        report(mean_sk, skills, rmses_model, rmses_persist, k, "(trained horizon)")

    print("\n=== Patch continuous model: ZERO-SHOT skill at UNSEEN horizons (test set) ===")
    zeroshot_skills, zeroshot_rmse = {}, {}
    for k in EVAL_KS:
        mean_sk, skills, rmses_model, rmses_persist = evaluate_patch_skill(
            cont_model, states6, test_idx, k, tiles, tile_meta, ocean_mask,
            out_mean, out_std, b_mean, b_std, land_vm, n_sensors, n_vars,
            trunk6_raw_norm, K_MIN, K_MAX, th, tw, device)
        zeroshot_skills[k] = mean_sk
        zeroshot_rmse[k] = {"model": rmses_model, "persist": rmses_persist}
        report(mean_sk, skills, rmses_model, rmses_persist, k, "(zero-shot, never trained)")

    print("\n=== Naive baseline: linear interpolation of TRAINED-horizon skill ===")
    ks_sorted = sorted(trained_skills.keys())
    sk_sorted = [trained_skills[k] for k in ks_sorted]
    curvefit_skills = {}
    for k in EVAL_KS:
        curvefit_skills[k] = float(np.interp(k, ks_sorted, sk_sorted))
        print(f"  k={k:2d}  curve-fit skill = {curvefit_skills[k]:+.4f}")

    print("\n=== Training DEDICATED single-horizon PATCH models for comparison ===")
    dedicated_skills, dedicated_rmse = {}, {}
    for k in EVAL_KS:
        ded_model = train_patch_model(states6, train_idx, val_idx, [k], tiles, tile_meta, ocean_mask,
                                       out_mean, out_std, b_mean, b_std, land_vm, n_sensors, n_vars,
                                       trunk6_raw_norm, K_MIN, K_MAX, th, tw, args.dedicated_iterations,
                                       args.learning_rate, args.batch_size, args.patience, device,
                                       args.seed, f"patch-dedicated-k{k}")
        torch.save(ded_model.state_dict(), f"{args.out_dir}_dedicated_k{k}_model.pt")
        mean_sk, skills, rmses_model, rmses_persist = evaluate_patch_skill(
            ded_model, states6, test_idx, k, tiles, tile_meta, ocean_mask,
            out_mean, out_std, b_mean, b_std, land_vm, n_sensors, n_vars,
            trunk6_raw_norm, K_MIN, K_MAX, th, tw, device)
        dedicated_skills[k] = mean_sk
        dedicated_rmse[k] = {"model": rmses_model, "persist": rmses_persist}
        report(mean_sk, skills, rmses_model, rmses_persist, k, "(dedicated model)")

    print("\n=== SUMMARY ===")
    print(f"{'k':>4} {'continuous(zero-shot)':>22} {'curve-fit':>12} {'dedicated':>12}")
    for k in EVAL_KS:
        print(f"{k:4d} {zeroshot_skills[k]:+22.4f} {curvefit_skills[k]:+12.4f} {dedicated_skills[k]:+12.4f}")

    save_json(args.out_dir + "_metrics.json", {
        "trained_skills": trained_skills, "zeroshot_skills": zeroshot_skills,
        "curvefit_skills": curvefit_skills, "dedicated_skills": dedicated_skills,
        "trained_rmse": trained_rmse, "zeroshot_rmse": zeroshot_rmse, "dedicated_rmse": dedicated_rmse,
    })
    print(f"\nSaved to {args.out_dir}_metrics.json (+ model checkpoints alongside)")


if __name__ == "__main__":
    main()
