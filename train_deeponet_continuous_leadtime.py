#!/usr/bin/env python3
"""Continuous lead-time DeepONet: trunk queried at (lon, lat, k) instead of just
(lon, lat), where k is the forecast horizon in days. Trained on a mix of
step_days in {1, 5, 10} (the same horizons already validated in Sec. 3.4), then
queried zero-shot at UNSEEN intermediate horizons (k=3, k=7) -- the temporal
analogue of Sec. 3.7's spatial discretization-invariance test.

Unlike everything tried the previous day, there is no strong, hard-to-beat
classical baseline for "one model, arbitrary continuous lead time" -- the real
alternative is training a separate dedicated model per lead time, which is
exactly the expensive thing this is meant to approximate. So the key
comparison is not against a closed-form procedure, but against:
  1. Persistence at that lead time (the basic floor, used everywhere else)
  2. A naive curve-fit through the model's OWN measured skill at the trained
     k=1,5,10 horizons, extrapolated to k=3/7 -- tests whether directly
     querying the model at unseen k gives more than a trend line would
  3. A DEDICATED model trained only at that one held-out k (the "gold
     standard" this approach is trying to cheaply approximate)

Persistence residual is unchanged from every other model in this study
(predict today's state) -- it doesn't depend on k, which is physically
sensible (persistence means "no change," for any horizon), so the network only
has to learn a k-and-location-dependent correction on top.

Because different training examples can have different k, and a shared
GeometricAttention-style per-sample trunk would be needed to mix k within a
batch, this instead uses ONE k per training STEP (varying across steps, not
within a batch) -- simpler, avoids restructuring the batched einsum, and still
lets the model see all training horizons over the course of training.

Local-prototype scale, single seed, exploratory.
"""
import argparse
import copy

import numpy as np
import torch
import torch.nn as nn

from train_agulhas_deeponet_prototype import load_states, _var_major_flat, _make_mlp, to_tensor, VARIABLES, save_json

torch.set_num_threads(4)


class ContinuousLeadTimeDeepONet(nn.Module):
    """Same structure as MultivarDeepONet, but trunk takes (lon, lat, k) -- a
    standalone class so MultivarDeepONet (used throughout the rest of this
    study) is untouched."""

    def __init__(self, d_branch, n_sensors, n_vars, branch_width, branch_depth,
                 trunk_width, trunk_depth, latent_dim):
        super().__init__()
        self.n_vars = n_vars
        self.n_sensors = n_sensors
        branch_sizes = [d_branch] + [branch_width] * branch_depth + [latent_dim]
        trunk_sizes = [3] + [trunk_width] * trunk_depth + [latent_dim]  # (lon, lat, k_norm)
        self.trunk = _make_mlp(trunk_sizes, "tanh")
        self.branches = nn.ModuleList([_make_mlp(branch_sizes, "tanh") for _ in range(n_vars)])
        self.biases = nn.Parameter(torch.zeros(n_vars))
        for branch in self.branches:
            last_linear = None
            for layer in branch.modules():
                if isinstance(layer, nn.Linear):
                    last_linear = layer
            if last_linear is not None:
                nn.init.zeros_(last_linear.weight)
                nn.init.zeros_(last_linear.bias)

    def forward(self, branch_input, trunk_input):
        trunk_feats = self.trunk(trunk_input)  # [n_query, latent_dim]
        outputs = []
        for vi in range(self.n_vars):
            branch_feats = self.branches[vi](branch_input)
            out = torch.einsum("np,sp->ns", branch_feats, trunk_feats) + self.biases[vi]
            # persistence residual: today's state, independent of k (physically
            # sensible -- "no change" doesn't depend on how far out you ask)
            out = out + branch_input[:, vi * self.n_sensors:(vi + 1) * self.n_sensors]
            outputs.append(out)
        return torch.stack(outputs, dim=-1)


def build_shared_split(states, max_k, test_fraction=0.15, val_fraction=0.15):
    """Starting-day indices valid for ANY k up to max_k (i+k always < T)."""
    T = states.shape[0]
    N_common = T - max_k
    n_test = int(N_common * test_fraction)
    n_val = int(N_common * val_fraction)
    n_train = N_common - n_val - n_test
    return (np.arange(0, n_train), np.arange(n_train, n_train + n_val),
            np.arange(n_train + n_val, N_common))


def make_pair(states, i_idx, k, out_mean, out_std, ocean_mask, b_mean, b_std, land_vm, n_sensors, n_vars):
    """(branch_input, target_norm) for start-day indices i_idx at lead time k."""
    cur = _var_major_flat(states[i_idx].astype(np.float64))
    nxt = _var_major_flat(states[i_idx + k].astype(np.float64))
    branch = ((cur - b_mean) / b_std).astype(np.float32)
    branch[:, land_vm] = 0.0
    target_norm = np.zeros_like(nxt, dtype=np.float32)
    for vi in range(n_vars):
        c0, c1 = vi * n_sensors, (vi + 1) * n_sensors
        target_norm[:, c0:c1] = (nxt[:, c0:c1] - out_mean[vi]) / out_std[vi]
    return branch, target_norm, cur


def k_to_norm(k, k_min, k_max):
    return 2.0 * (k - k_min) / (k_max - k_min) - 1.0


def compute_loss_weight(states, train_idx, k, out_mean, out_std, ocean_mask, b_mean, b_std,
                         land_vm, n_sensors, n_vars):
    """Skill-aligned inverse-increment-variance weighting (Sec. 2.4/4.2 of the
    manuscript) -- the fix that makes every other working model in this study
    learn real dynamics instead of settling near "barely move from input."
    Computed separately per k, since typical increment size/variance changes
    with lead time (skill degrades with k, Table 3), so one fixed weighting
    across all k would be wrong for at least some of them."""
    branch, target_norm, _ = make_pair(states, train_idx, k, out_mean, out_std, ocean_mask,
                                        b_mean, b_std, land_vm, n_sensors, n_vars)
    incr = target_norm.astype(np.float64) - branch.astype(np.float64)
    incr_var = incr.var(axis=0)
    om_vm = np.tile(ocean_mask, n_vars)
    floor = 0.1 * np.median(incr_var[om_vm]) if om_vm.any() else 1.0
    loss_weight = 1.0 / np.maximum(incr_var, max(floor, 1e-12))
    loss_weight[~om_vm] = 0.0
    if om_vm.any():
        loss_weight *= om_vm.sum() / loss_weight[om_vm].sum()
    return loss_weight.astype(np.float32)


def evaluate_skill(model, states, idx, k, trunk6_raw_norm, out_mean, out_std, ocean_mask,
                    b_mean, b_std, land_vm, n_sensors, n_vars, k_min, k_max, device):
    branch, target_norm, cur = make_pair(states, idx, k, out_mean, out_std, ocean_mask,
                                          b_mean, b_std, land_vm, n_sensors, n_vars)
    kn = k_to_norm(k, k_min, k_max)
    trunk_k = np.concatenate([trunk6_raw_norm, np.full((trunk6_raw_norm.shape[0], 1), kn, dtype=np.float32)], axis=1)
    model.eval()
    with torch.no_grad():
        pred = model(to_tensor(branch, device), to_tensor(trunk_k, device)).cpu().numpy()
    pred_phys = pred * out_std[None, None, :] + out_mean[None, None, :]
    nxt_raw = _var_major_flat(states[idx + k].astype(np.float64))
    skills = {}
    for vi, vname in enumerate(VARIABLES):
        c0, c1 = vi * n_sensors, (vi + 1) * n_sensors
        yt = nxt_raw[:, c0:c1][:, ocean_mask]
        yp = pred_phys[:, :, vi][:, ocean_mask]
        ys = cur[:, c0:c1][:, ocean_mask]
        rmse_m = np.sqrt(np.mean((yp - yt) ** 2))
        rmse_p = np.sqrt(np.mean((ys - yt) ** 2))
        skills[vname] = 1.0 - (rmse_m / rmse_p) ** 2 if rmse_p > 0 else float("nan")
    return float(np.mean(list(skills.values()))), skills


def train_model(states, train_idx, val_idx, ks, out_mean, out_std, ocean_mask, b_mean, b_std,
                 land_vm, n_sensors, n_vars, trunk6_raw_norm, k_min, k_max, iterations, lr,
                 batch_size, patience, device, seed, label, loss_weights):
    rng = np.random.default_rng(seed)
    model = ContinuousLeadTimeDeepONet(
        d_branch=n_sensors * n_vars, n_sensors=n_sensors, n_vars=n_vars,
        branch_width=64, branch_depth=2, trunk_width=64, trunk_depth=2, latent_dim=32,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    ocean_t = torch.tensor(np.tile(ocean_mask, n_vars), dtype=torch.bool, device=device)
    w_t = {k: to_tensor(loss_weights[k], device) for k in ks}
    best_val, best_state, bad = float("inf"), copy.deepcopy(model.state_dict()), 0

    for step in range(1, iterations + 1):
        model.train()
        k = ks[rng.integers(0, len(ks))]
        idx = train_idx[rng.integers(0, len(train_idx), size=min(batch_size, len(train_idx)))]
        branch, target_norm, _ = make_pair(states, idx, k, out_mean, out_std, ocean_mask,
                                            b_mean, b_std, land_vm, n_sensors, n_vars)
        kn = k_to_norm(k, k_min, k_max)
        trunk_k = np.concatenate([trunk6_raw_norm, np.full((trunk6_raw_norm.shape[0], 1), kn, dtype=np.float32)], axis=1)
        pred = model(to_tensor(branch, device), to_tensor(trunk_k, device))
        pred_vm = pred.permute(0, 2, 1).reshape(pred.shape[0], -1)
        e2 = (pred_vm[:, ocean_t] - to_tensor(target_norm, device)[:, ocean_t]) ** 2
        loss = (e2 * w_t[k][ocean_t]).mean()
        opt.zero_grad(); loss.backward(); opt.step()

        if step % 200 == 0 or step == iterations:
            model.eval()
            vlosses = []
            with torch.no_grad():
                for vk in ks:
                    vbranch, vtarget, _ = make_pair(states, val_idx, vk, out_mean, out_std, ocean_mask,
                                                     b_mean, b_std, land_vm, n_sensors, n_vars)
                    vkn = k_to_norm(vk, k_min, k_max)
                    vtrunk = np.concatenate([trunk6_raw_norm, np.full((trunk6_raw_norm.shape[0], 1), vkn, dtype=np.float32)], axis=1)
                    vpred = model(to_tensor(vbranch, device), to_tensor(vtrunk, device))
                    vpred_vm = vpred.permute(0, 2, 1).reshape(vpred.shape[0], -1)
                    ve2 = (vpred_vm[:, ocean_t] - to_tensor(vtarget, device)[:, ocean_t]) ** 2
                    vlosses.append((ve2 * w_t[vk][ocean_t]).mean().item())
            vloss = float(np.mean(vlosses))
            print(f"  [{label}] step {step:5d}  train_loss {loss.item():.5f}  val_loss(avg over {ks}) {vloss:.5f}")
            if vloss < best_val - 1e-6:
                best_val, best_state, bad = vloss, copy.deepcopy(model.state_dict()), 0
            else:
                bad += 1
                if bad >= patience:
                    print(f"  [{label}] early stop at step {step} (best val_loss={best_val:.5f})")
                    break
    model.load_state_dict(best_state)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nc", default="data/agulhas_prototype.nc")
    ap.add_argument("--cache", default="data/cache_r6_local.npz")
    ap.add_argument("--iterations", type=int, default=5000)
    ap.add_argument("--dedicated-iterations", type=int, default=3000)
    ap.add_argument("--learning-rate", type=float, default=3e-4)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out-dir", default="results/continuous_leadtime_local")
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
    print(f"shared split: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)} start-days "
          f"(valid for any k up to {max(TRAIN_KS + EVAL_KS)})")

    # Normalisation stats: fit once (k=1 next-state convention) on the training slice.
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

    print("\n=== Computing skill-aligned loss weights per k (the fix) ===")
    all_ks = sorted(set(TRAIN_KS + EVAL_KS))
    loss_weights = {}
    for k in all_ks:
        loss_weights[k] = compute_loss_weight(states6, train_idx, k, out_mean, out_std, ocean_mask,
                                               b_mean, b_std, land_vm, n_sensors, n_vars)
        print(f"  k={k:2d}  loss_weight computed (mean={loss_weights[k][np.tile(ocean_mask, n_vars)].mean():.3f}, "
              f"should be ~1.0 over ocean)")

    print(f"\n=== Training CONTINUOUS lead-time model on k in {TRAIN_KS} ===")
    cont_model = train_model(states6, train_idx, val_idx, TRAIN_KS, out_mean, out_std, ocean_mask,
                              b_mean, b_std, land_vm, n_sensors, n_vars, trunk6_raw_norm, K_MIN, K_MAX,
                              args.iterations, args.learning_rate, args.batch_size, args.patience,
                              device, args.seed, "continuous", loss_weights)

    print("\n=== Continuous model: skill at TRAINED horizons (test set) ===")
    trained_skills = {}
    for k in TRAIN_KS:
        mean_sk, per_var = evaluate_skill(cont_model, states6, test_idx, k, trunk6_raw_norm, out_mean,
                                           out_std, ocean_mask, b_mean, b_std, land_vm, n_sensors, n_vars,
                                           K_MIN, K_MAX, device)
        trained_skills[k] = mean_sk
        print(f"  k={k:2d}  mean skill = {mean_sk:+.4f}")

    print("\n=== Continuous model: ZERO-SHOT skill at UNSEEN horizons (test set) ===")
    zeroshot_skills = {}
    for k in EVAL_KS:
        mean_sk, per_var = evaluate_skill(cont_model, states6, test_idx, k, trunk6_raw_norm, out_mean,
                                           out_std, ocean_mask, b_mean, b_std, land_vm, n_sensors, n_vars,
                                           K_MIN, K_MAX, device)
        zeroshot_skills[k] = mean_sk
        print(f"  k={k:2d}  mean skill = {mean_sk:+.4f}  (never trained on this k)")

    # Naive curve-fit baseline: linear interpolation in k through the model's
    # OWN measured skill at the trained horizons, evaluated at the unseen k.
    print("\n=== Naive baseline: linear interpolation of TRAINED-horizon skill ===")
    ks_sorted = sorted(trained_skills.keys())
    sk_sorted = [trained_skills[k] for k in ks_sorted]
    curvefit_skills = {}
    for k in EVAL_KS:
        curvefit_skills[k] = float(np.interp(k, ks_sorted, sk_sorted))
        print(f"  k={k:2d}  curve-fit skill = {curvefit_skills[k]:+.4f}  "
              f"vs. actual zero-shot query = {zeroshot_skills[k]:+.4f}")

    # Dedicated single-purpose models at the held-out horizons (gold standard).
    print("\n=== Training DEDICATED single-horizon models for comparison ===")
    dedicated_skills = {}
    for k in EVAL_KS:
        ded_model = train_model(states6, train_idx, val_idx, [k], out_mean, out_std, ocean_mask,
                                 b_mean, b_std, land_vm, n_sensors, n_vars, trunk6_raw_norm, K_MIN, K_MAX,
                                 args.dedicated_iterations, args.learning_rate, args.batch_size,
                                 args.patience, device, args.seed, f"dedicated-k{k}", loss_weights)
        mean_sk, _ = evaluate_skill(ded_model, states6, test_idx, k, trunk6_raw_norm, out_mean, out_std,
                                     ocean_mask, b_mean, b_std, land_vm, n_sensors, n_vars, K_MIN, K_MAX, device)
        dedicated_skills[k] = mean_sk
        print(f"  k={k:2d}  dedicated model mean skill = {mean_sk:+.4f}")

    print("\n=== SUMMARY ===")
    print(f"{'k':>4} {'continuous(zero-shot)':>22} {'curve-fit':>12} {'dedicated':>12}")
    for k in EVAL_KS:
        print(f"{k:4d} {zeroshot_skills[k]:+22.4f} {curvefit_skills[k]:+12.4f} {dedicated_skills[k]:+12.4f}")

    save_json(args.out_dir + "_metrics.json", {
        "trained_skills": trained_skills, "zeroshot_skills": zeroshot_skills,
        "curvefit_skills": curvefit_skills, "dedicated_skills": dedicated_skills,
    })
    print(f"\nSaved to {args.out_dir}_metrics.json")


if __name__ == "__main__":
    main()
