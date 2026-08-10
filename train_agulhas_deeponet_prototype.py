#!/usr/bin/env python3
"""Physics-informed DeepONet for single-step Agulhas eddy forecasting.

Architecture
------------
A single joint model with one shared trunk network and six branch networks
(one per variable). All six output fields share the same spatial query points,
so the full forward pass produces [N, n_sensors, 6] in one shot.

    MultivarDeepONet(branch_inputs)  →  [N, n_sensors, 6]
        shared trunk  : [2]       → [64]*depth → [latent_dim]
        branch_zos    : [D_branch] → [64]*depth → [latent_dim]
        branch_uo     : [D_branch] → [128]*depth → [latent_dim]
        ...
        output_v(y)   = dot(branch_v(s), trunk(y)) + bias_v

Physics loss (soft constraints, finite differences on the lon/lat grid)
-----------------------------------------------------------------------
L_div  : divergence-free condition on predicted (uo, vo)
           (1/a·cosφ) ∂û/∂λ  +  (1/a·cosφ) ∂(v̂·cosφ)/∂φ  ≈ 0

L_geo  : geostrophic consistency between predicted SSH and velocities
           u_g = -(g/f·a) ∂η̂/∂φ      v_g = (g/f·a·cosφ) ∂η̂/∂λ
           penalise ||û - u_g||² + ||v̂ - v_g||²

Total loss (physics terms scaled by step-1 reference, linear warmup):
    L = L_data  +  ramp·λ₁·(L_div/L_div_ref)  +  ramp·λ₂·(L_geo/L_geo_ref)
    (defaults: λ₁=λ₂=0.1, warmup=500; set either λ to 0 to ablate)

Training
--------
Pure PyTorch training loop. The forward pass, physics loss, and optimiser
are all written explicitly, giving full control over the multi-output +
physics loss. No external framework is required.

Evaluation
----------
Single-step: per-variable RMSE, skill vs persistence, NRMSE, bias, and ACC
             (anomaly correlation coefficient vs training climatology).
Multi-step : autoregressive rollout at horizons 1/5/10/20 days (RMSE + ACC
             vs lead time), plus spatial time-averaged RMSE maps. Pass
             --compare-to <other_run> to overlay a baseline for the Aim-2
             physics-vs-data-driven comparison.

Dependencies
------------
    pip install torch netCDF4 numpy
    # This file is standalone — no local module imports.

Usage
-----
    # Smoke-test:
    python train_agulhas_deeponet_prototype.py --nc data/agulhas_prototype.nc \
        --subsample-r 10 --iterations 500 --latent-dim 32

    # Prototype with physics constraints:
    python train_agulhas_deeponet_prototype.py --nc data/agulhas_prototype.nc \
        --subsample-r 6 --iterations 5000 --latent-dim 64

    # Ablation — data-driven only:
    python train_agulhas_deeponet_prototype.py --nc data/agulhas_prototype.nc \
        --lambda-div 0.0 --lambda-geo 0.0
"""

import argparse
import glob
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


# ── Physical constants ────────────────────────────────────────────────────────

EARTH_RADIUS  = 6.371e6        # metres
OMEGA_EARTH   = 7.29e-5        # rad s⁻¹
GRAVITY       = 9.81           # m s⁻²
DEG2RAD       = math.pi / 180.0

VARIABLES = ["zos", "uo", "vo", "thetao", "so", "mlotst"]
I_ZOS, I_UO, I_VO = 0, 1, 2   # indices in VARIABLES

VARIABLE_UNITS = {
    "zos":    "m",
    "uo":     "m/s",
    "vo":     "m/s",
    "thetao": "°C",
    "so":     "PSU",
    "mlotst": "m",
}


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Physics-informed Agulhas DeepONet (joint 6-variable model)."
    )
    p.add_argument("--nc", type=Path, default=Path("data/agulhas_prototype.nc"),
                   help="NetCDF file, a directory of them, or a glob (e.g. "
                        "'data/agulhas_*.nc'). Multiple files are stitched along "
                        "time — use this to combine the GLORYS my + myint streams.")
    p.add_argument("--cache", type=Path, default=None,
                   help="Path to a .npz cache of the subsampled cube. If it exists, "
                        "load from it (fast — skips decompressing the raw NetCDF); "
                        "otherwise build it from --nc and save it here for reuse.")
    p.add_argument("--prepare-cache", action="store_true",
                   help="Build the --cache from --nc and exit (no training). Use once "
                        "before a sweep so every run then loads the cache in seconds.")
    p.add_argument("--out-dir", type=Path, default=Path("results/agulhas_pinn"))
    p.add_argument("--subsample-r", type=int, default=6,
                   help="Retain every r-th grid point in lon and lat.")
    p.add_argument("--step-days", type=int, default=1,
                   help="Forecast step Δt in days: the model maps state(t) → "
                        "state(t+k). Larger k = bigger, more-predictable increment "
                        "(persistence weaker). Rollout horizons must be multiples of k.")
    # Network
    p.add_argument("--latent-dim",    type=int,   default=32)
    p.add_argument("--branch-width",  type=int,   default=64)
    p.add_argument("--trunk-width",   type=int,   default=64)
    p.add_argument("--branch-depth",  type=int,   default=2)
    p.add_argument("--trunk-depth",   type=int,   default=2)
    # Training
    p.add_argument("--iterations",    type=int,   default=5000)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--batch-size",    type=int,   default=None,
                   help="None = full-batch.")
    p.add_argument("--val-fraction",  type=float, default=0.15,
                   help="Fraction of the (chronological) timeline held out for "
                        "validation / early stopping.")
    p.add_argument("--test-fraction", type=float, default=0.15,
                   help="Fraction held out (most recent) for final testing.")
    p.add_argument("--embargo",       type=int,   default=0,
                   help="Sample-index gap dropped at the train/val and val/test split "
                        "boundaries (see build_dataset() docstring). 0 (default) matches "
                        "every result reported elsewhere in this study; only needed when "
                        "training on a rolling-window aggregate where adjacent samples "
                        "share raw underlying days.")
    p.add_argument("--patience",      type=int,   default=20,
                   help="Early-stop after this many evaluations (each --display-every "
                        "steps) with no val-loss improvement. 0 disables early stopping.")
    p.add_argument("--seed",          type=int,   default=2026)
    p.add_argument("--display-every", type=int,   default=100,
                   help="Print loss table every N iterations (always prints step 1).")
    # Data loss weighting
    p.add_argument("--loss-weight", choices=["none", "variability"], default="none",
                   help="'variability' weights each (sensor,variable) squared error by "
                        "a capped inverse increment-variance, turning the objective into "
                        "a skill-aligned (fraction-of-variance-explained) loss so the "
                        "model is rewarded for the dynamics, not the static field.")
    # Physics weights — applied to the RAW losses (no reference-normalisation; that
    # scheme inflated the trivially-satisfied L_div to ~unit scale and injected noise).
    # L_div defaults to 0: on reanalysis it is ~machine-epsilon, so it only adds noise.
    p.add_argument("--lambda-div",    type=float, default=0.0,
                   help="Weight for divergence-free loss on the RAW residual (0 = off; "
                        "recommended, it is trivially satisfied on reanalysis).")
    p.add_argument("--lambda-geo",    type=float, default=0.1,
                   help="Weight for geostrophic consistency loss on the RAW residual.")
    p.add_argument("--warmup-steps",  type=int,   default=500,
                   help="Linear ramp for physics loss weights over the first N steps.")
    p.add_argument("--no-residual", action="store_true",
                   help="Disable the persistence skip connection (default: on).")
    # Multi-step rollout / comparison (Aim 2)
    p.add_argument("--rollout-horizons", type=int, nargs="+", default=[1, 5, 10, 20],
                   help="Lead times (days) for autoregressive rollout evaluation.")
    p.add_argument("--spatial-var", type=str, default="zos",
                   help="Variable for the spatial time-averaged RMSE map.")
    p.add_argument("--dump-rollout-fields", type=int, default=0,
                   help="If >0, save the raw (model/truth/persistence) SSH+velocity "
                        "fields for this many rollout starts at every requested "
                        "horizon to rollout_fields.npz, for offline eddy-tracking "
                        "analysis (e.g. py-eddy-tracker) beyond the scalar metrics.")
    p.add_argument("--compare-to", type=Path, default=None,
                   help="Another run's out-dir; overlay its rollout/spatial RMSE "
                        "against this run (e.g. baseline vs physics-informed).")
    return p.parse_args()


# ── Data loading ──────────────────────────────────────────────────────────────

def _var_major_flat(grid):
    """Flatten [N, nlat, nlon, n_vars] → [N, n_vars*n_sensors] (variable-major).

    Columns [vi*n_sensors : (vi+1)*n_sensors] hold all grid values for variable vi,
    with sensors in lat-major order (matches pred.permute(0, 2, 1).reshape).
    """
    n = grid.shape[0]
    nlat_s, nlon_s, n_vars = grid.shape[1:]
    n_sensors = nlat_s * nlon_s
    return grid.reshape(n, n_sensors, n_vars).transpose(0, 2, 1).reshape(n, n_vars * n_sensors)


def _resolve_nc_files(nc_path):
    """Expand nc_path to a sorted list of files.

    Accepts a single file, a directory (→ every *.nc inside), or a glob pattern
    (e.g. 'data/agulhas_*.nc').  Multiple files let us stitch the GLORYS `_my_`
    reanalysis and `_myint_` interim streams — which no single Copernicus request
    can span — into one continuous record.
    """
    p = str(nc_path)
    if os.path.isdir(p):
        files = sorted(glob.glob(os.path.join(p, "*.nc")))
    elif any(c in p for c in "*?["):
        files = sorted(glob.glob(p))
    else:
        files = [p]
    if not files:
        raise SystemExit(f"No NetCDF files matched: {nc_path}")
    return files


def _read_one_nc(nc4, path, subsample_r, time_block=1000):
    """Read one file's subsampled fields + time axis.

    We read **contiguous time-blocks at full spatial resolution and subsample in
    memory** rather than issuing a single strided (`::r`) hyperslab.  A strided
    read touches every time-chunk of a chunked/compressed file with per-chunk
    latency — on a multi-year file that's tens of thousands of tiny reads and can
    take *hours* on a network filesystem.  Bulk contiguous reads move the same
    bytes as a few large sequential I/Os instead; memory stays bounded by one
    block (time_block × full grid).
    """
    ds = nc4.Dataset(path, "r")
    lat_sl = slice(0, None, subsample_r)
    lon_sl = slice(0, None, subsample_r)
    lon_sub = np.asarray(ds.variables["longitude"][lon_sl], dtype=np.float32)
    lat_sub = np.asarray(ds.variables["latitude"][lat_sl],  dtype=np.float32)
    nlon_s, nlat_s = len(lon_sub), len(lat_sub)

    time = np.asarray(ds.variables["time"][:], dtype=np.float64)
    T = len(time)
    states = np.zeros((T, nlat_s, nlon_s, len(VARIABLES)), dtype=np.float32)
    for vi, vname in enumerate(VARIABLES):
        v = ds.variables[vname]
        if v.ndim not in (3, 4):
            raise ValueError(
                f"Variable '{vname}' in {path} has unexpected ndim={v.ndim}; "
                "expected 3 (time,lat,lon) or 4 (time,depth,lat,lon)."
            )
        for t0 in range(0, T, time_block):
            t1 = min(t0 + time_block, T)
            blk = v[t0:t1, 0, :, :] if v.ndim == 4 else v[t0:t1, :, :]  # contiguous read
            blk = blk[:, lat_sl, lon_sl]                               # subsample in memory
            if np.ma.is_masked(blk):
                blk = np.ma.filled(blk, np.nan)
            states[t0:t1, :, :, vi] = np.asarray(blk, dtype=np.float32)
        print(f"    read {vname} ({T} steps)", flush=True)
    ds.close()
    return lon_sub, lat_sub, time, states


def load_nc(nc_path: Path, subsample_r: int):
    """Load one or more NetCDF files, subsample, and return (lon_sub, lat_sub, states).

    nc_path may be a file, a directory, or a glob (see _resolve_nc_files).  Multiple
    files are concatenated along time, then sorted with duplicate timestamps dropped
    — so overlapping `_my_`/`_myint_` GLORYS streams merge into one clean daily
    record.  states : float32 [T, nlat_s, nlon_s, n_vars].
    """
    try:
        import netCDF4 as nc4
    except ModuleNotFoundError:
        raise SystemExit("netCDF4 not installed. Run: pip install netCDF4")

    files = _resolve_nc_files(nc_path)
    print(f"Loading {len(files)} file(s) matching {nc_path} …")

    lon_ref = lat_ref = None
    times, blocks = [], []
    for f in files:
        lon_sub, lat_sub, time, states_i = _read_one_nc(nc4, f, subsample_r)
        if lon_ref is None:
            lon_ref, lat_ref = lon_sub, lat_sub
        elif lon_sub.shape != lon_ref.shape or lat_sub.shape != lat_ref.shape:
            raise SystemExit(
                f"Grid mismatch in {f} ({len(lon_sub)}×{len(lat_sub)}) vs "
                f"{len(lon_ref)}×{len(lat_ref)}; files must share the same domain/grid."
            )
        times.append(time)
        blocks.append(states_i)
        print(f"  {os.path.basename(f)}: {states_i.shape[0]} steps, "
              f"{len(lon_sub)}×{len(lat_sub)} subsampled pts (r={subsample_r})")

    if len(blocks) == 1:
        states = blocks[0]
    else:
        all_time   = np.concatenate(times)
        all_states = np.concatenate(blocks, axis=0)
        # Sort by time and drop duplicate timestamps (handles a my/myint seam overlap).
        uniq, idx = np.unique(all_time, return_index=True)
        states = all_states[idx]
        print(f"  Combined: {states.shape[0]} unique daily steps "
              f"({all_time.size - uniq.size} duplicate/overlap steps dropped)")
        d = np.diff(uniq)
        if d.size and d.max() > 1.5 * np.median(d):
            print(f"  NOTE: non-uniform time spacing (max gap {d.max()/np.median(d):.1f}× "
                  f"median) — one t→t+1 pair may straddle a gap between files.")

    lon_sub, lat_sub = lon_ref, lat_ref
    print(f"  Subsampled grid : {len(lon_sub)} × {len(lat_sub)} = "
          f"{len(lon_sub)*len(lat_sub):,} pts   Total time steps : {states.shape[0]}")

    nan_frac = np.isnan(states).mean()
    if nan_frac > 0:
        print(f"  NaN fraction (land/mask): {nan_frac:.1%} → replaced with 0")
        states = np.nan_to_num(states, nan=0.0)

    return lon_sub, lat_sub, states   # states: [T, nlat_s, nlon_s, n_vars]


def load_states(nc_path, subsample_r, cache=None):
    """load_nc with an optional .npz cache of the subsampled cube.

    Decompressing the raw multi-GB NetCDF with strided reads takes minutes; the
    subsampled cube is small.  With --cache set, the first run builds the cube and
    saves it, and every later run (e.g. a hyperparameter sweep) loads it in seconds.
    """
    if cache is not None and Path(cache).exists():
        print(f"Loading cached subsampled cube from {cache} …")
        z = np.load(cache)
        if int(z["subsample_r"]) != subsample_r:
            raise SystemExit(
                f"Cache {cache} was built with r={int(z['subsample_r'])} but "
                f"--subsample-r={subsample_r}. Delete it or point --cache elsewhere."
            )
        lon_sub, lat_sub, states = z["lon"], z["lat"], z["states"]
        print(f"  Subsampled grid : {len(lon_sub)} × {len(lat_sub)}   "
              f"Time steps : {states.shape[0]}  (from cache)")
        return lon_sub, lat_sub, states

    lon_sub, lat_sub, states = load_nc(nc_path, subsample_r)
    if cache is not None:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        print(f"Saving subsampled cube to cache {cache} …")
        np.savez_compressed(
            cache, states=states.astype(np.float32),
            lon=lon_sub, lat=lat_sub, subsample_r=subsample_r,
        )
    return lon_sub, lat_sub, states


def build_dataset(states, lon_sub, lat_sub, test_fraction, val_fraction=0.0, step_days=1,
                   embargo=0):
    """Build normalised train/val/test arrays.

    Returns a dict with everything the training loop needs.

    Branch inputs are the *flattened* current state [N, nlat_s*nlon_s*n_vars].
    Outputs are kept as [N, nlat_s, nlon_s, n_vars] for physics loss convenience,
    then also provided flat for the data loss.

    step_days (k): each sample is a pair (state(t), state(t+k)), so the model
    learns a k-day evolution.  Pair index i uses current time i and target time
    i+k, so persistence (state(t)) and climatology carry over unchanged.

    The split is chronological (not randomised) — consecutive daily fields are
    nearly identical, so a random split would leak near-duplicates.  The timeline
    is partitioned in order: train | validation | test, with validation and test
    taken from the most recent end.  All normalisation statistics (branch/output
    mean-std, ocean mask, climatology) are fit on the TRAIN slice only, so val and
    test are never seen during preprocessing.

    embargo: number of sample-index positions dropped from the END of train and
    the END of val (immediately before the val and test boundaries respectively)
    before assigning train_idx/val_idx/test_idx. For plain daily data (step_days
    small relative to any within-sample aggregation) 0 is correct and matches
    every result reported elsewhere in this study. It matters when `states` is
    itself a ROLLING aggregate (e.g. prepare_weekly_cache.py's rolling 7-day
    mean): there, sample i and sample i+1 share up to (window-1) raw underlying
    days, so a plain chronological split at the sample-index level still lets
    the last training sample and the first validation/test sample overlap
    almost entirely in raw data. Setting embargo >= window-1 guarantees a
    sample-index gap large enough that no two samples on opposite sides of a
    split boundary share a raw day.
    """
    T, nlat_s, nlon_s, n_vars = states.shape
    n_sensors = nlat_s * nlon_s
    k = max(1, int(step_days))
    N = T - k

    # Variable-major flatten: columns [vi*n_sensors:(vi+1)*n_sensors] = variable vi
    flat = _var_major_flat(states.astype(np.float64))
    branch_inputs = flat[:-k]                                 # [N, D]  current = state(t)
    # Keep next-state (t+k) as grid for physics, flat for data loss
    next_grid = states[k:].astype(np.float64)                 # [N, nlat_s, nlon_s, n_vars]
    next_flat = _var_major_flat(next_grid)                    # [N, D]

    # Trunk: (lon, lat) pairs, lat-major order to match grid layout
    LAT, LON = np.meshgrid(lat_sub, lon_sub, indexing="ij")   # (nlat_s, nlon_s)
    trunk_raw = np.stack([LON.ravel(), LAT.ravel()], axis=-1).astype(np.float64)

    # Chronological split: train | val | test  (val/test taken from the recent end)
    n_test  = max(1, int(N * test_fraction))
    n_val   = max(1, int(N * val_fraction)) if val_fraction > 0 else 0
    n_train = N - n_val - n_test
    if n_train <= 0:
        raise ValueError(
            f"val_fraction + test_fraction too large: n_train={n_train} "
            f"(N={N}, n_val={n_val}, n_test={n_test})"
        )
    train_idx = np.arange(0, n_train, dtype=np.int64)
    val_idx   = np.arange(n_train, n_train + n_val, dtype=np.int64)
    test_idx  = np.arange(n_train + n_val, N, dtype=np.int64)

    if embargo > 0:
        # Drop `embargo` samples from the end of train (border with val) and
        # from the end of val (border with test), so the kept train/val
        # boundary and val/test boundary each have an index gap > embargo —
        # sufficient to guarantee zero shared raw days for a rolling-window
        # aggregate with window <= embargo + 1 (see docstring).
        if len(train_idx) > embargo:
            train_idx = train_idx[:-embargo]
        else:
            train_idx = train_idx[:0]
        if n_val > 0:
            if len(val_idx) > embargo:
                val_idx = val_idx[:-embargo]
            else:
                val_idx = val_idx[:0]

    # Normalise trunk (min-max to [-1, 1])
    t_min  = trunk_raw.min(axis=0, keepdims=True)
    t_span = trunk_raw.max(axis=0, keepdims=True) - t_min
    t_span = np.where(t_span < 1e-12, 1.0, t_span)
    trunk_norm = (2.0 * (trunk_raw - t_min) / t_span - 1.0).astype(np.float32)

    # Per-variable output normalisation (scalar mean/std, fit on train, ocean only).
    # Land points are zeroed by nan_to_num; including them inflates std by ~10×
    # for variables like thetao and so, making physical-unit metrics meaningless.
    # We identify ocean sensors as those with non-trivial temporal variance in the
    # training next-states, then fit mean/std on those sensors only.
    out_mean = np.zeros(n_vars, dtype=np.float64)
    out_std  = np.ones(n_vars,  dtype=np.float64)
    next_flat_norm = next_flat.copy()
    for vi in range(n_vars):
        col_start = vi * n_sensors
        col_end   = (vi + 1) * n_sensors
        block = next_flat[train_idx, col_start:col_end]   # [N_train, n_sensors]
        ocean = block.std(axis=0) > 1e-4                  # sensors with real variability
        vals  = block[:, ocean] if ocean.any() else block
        m = vals.mean()
        s = vals.std()
        if s < 1e-12:
            s = 1.0
        out_mean[vi] = m
        out_std[vi]  = s
        next_flat_norm[:, col_start:col_end] = (next_flat[:, col_start:col_end] - m) / s

    y_train_norm = next_flat_norm[train_idx].astype(np.float32)  # [N_train, D]
    y_val_norm   = next_flat_norm[val_idx].astype(np.float32)    # [N_val,   D]
    y_test_norm  = next_flat_norm[test_idx].astype(np.float32)   # [N_test,  D]

    # Ocean mask: sensors with real temporal variability in training data.
    # Land sensors are zeroed (nan_to_num) so their std ≈ 0.  Using any one
    # variable is sufficient — land is the same in every variable.
    ocean_mask = next_flat[train_idx, :n_sensors].std(axis=0) > 1e-4  # [n_sensors] bool

    # Normalise branch with the SAME per-variable scalars as the output, so the
    # persistence skip is exact: at init the DeepONet term is 0 and the residual
    # branch_input[block_v] == (raw_current - out_mean[v]) / out_std[v], which is
    # persistence expressed in the target's normalisation.  (A per-feature branch
    # z-score would live in a different space and break the identity, leaving
    # step-1 loss ~1 instead of the true 1-day signal.)  Land sensors are set to 0
    # (the per-variable mean → neutral input); they are masked from the loss and
    # held fixed during rollout, so their branch value is irrelevant, and zeroing
    # avoids feeding large constants (e.g. salinity land ≈ (0-35)/0.78 ≈ -45) into
    # the branch MLPs.
    b_mean = np.repeat(out_mean, n_sensors)[None, :]   # [1, D] per-variable, tiled
    b_std  = np.repeat(out_std,  n_sensors)[None, :]   # [1, D]
    land_vm = np.tile(~ocean_mask, n_vars)             # [D] bool
    branch_train = ((branch_inputs[train_idx] - b_mean) / b_std)
    branch_val   = ((branch_inputs[val_idx]   - b_mean) / b_std)
    branch_test  = ((branch_inputs[test_idx]  - b_mean) / b_std)
    branch_train[:, land_vm] = 0.0
    branch_val[:,   land_vm] = 0.0
    branch_test[:,  land_vm] = 0.0
    branch_train = branch_train.astype(np.float32)
    branch_val   = branch_val.astype(np.float32)
    branch_test  = branch_test.astype(np.float32)

    # Variability weights for a skill-aligned data loss.
    # Because of the persistence residual, the data error equals (Δ_pred − Δ_true)
    # in normalised units, where Δ = next − current.  Weighting each (sensor,var)
    # column by 1/Var_train(Δ) turns the objective into "fraction of increment
    # variance explained" (≈ per-column skill), so the model is rewarded for the
    # dynamics rather than the static field persistence already nails.  A floor
    # caps the weight on near-static columns (where Var(Δ)→0) to avoid blow-up.
    incr     = (y_train_norm.astype(np.float64) - branch_train.astype(np.float64))
    incr_var = incr.var(axis=0)                          # [D] per (sensor,var)
    om_vm    = np.tile(ocean_mask, n_vars)               # [D] bool
    floor    = 0.1 * np.median(incr_var[om_vm]) if om_vm.any() else 1.0
    loss_weight = 1.0 / np.maximum(incr_var, max(floor, 1e-12))
    loss_weight[~om_vm] = 0.0
    if om_vm.any():                                      # normalise: mean weight = 1 over ocean
        loss_weight *= om_vm.sum() / loss_weight[om_vm].sum()
    loss_weight = loss_weight.astype(np.float32)         # [D]

    # Current state for each test pair in raw units, variable-major flat.
    # x_test_raw[i] = u_t  for pair i;  y_test_raw[i] = u_{t+1}.
    # Subtracting gives the persistence forecast error without running the model.
    x_test_raw = _var_major_flat(states[test_idx].astype(np.float64))

    # Climatology: per-sensor, per-variable mean of the training next-states,
    # used as the reference field for the anomaly correlation coefficient (ACC).
    # next_flat is variable-major, so reshape (n_vars, n_sensors) then transpose
    # to [n_sensors, n_vars] to match the pred/true 3-D layout used in main().
    clim_vm = next_flat[train_idx].mean(axis=0)                       # [n_vars*n_sensors]
    climatology = clim_vm.reshape(n_vars, n_sensors).T               # [n_sensors, n_vars]

    return dict(
        branch_train=branch_train,      # [N_train, D_branch]
        branch_val=branch_val,          # [N_val,   D_branch]
        branch_test=branch_test,        # [N_test,  D_branch]
        trunk=trunk_norm,               # [n_sensors, 2]
        y_train_norm=y_train_norm,      # [N_train, n_sensors*n_vars]
        y_val_norm=y_val_norm,          # [N_val,   n_sensors*n_vars]
        y_test_norm=y_test_norm,        # [N_test,  n_sensors*n_vars]
        y_train_raw=next_flat[train_idx],  # [N_train, n_sensors*n_vars] raw
        y_test_raw=next_flat[test_idx],    # [N_test,  n_sensors*n_vars] raw
        x_test_raw=x_test_raw,          # [N_test,  n_sensors*n_vars] raw — persistence forecast
        climatology=climatology,        # [n_sensors, n_vars] — training-mean reference for ACC
        ocean_mask=ocean_mask,          # [n_sensors] bool — excludes land from loss & metrics
        loss_weight=loss_weight,        # [n_vars*n_sensors] — skill-aligned data-loss weights
        # Grid-shaped next-state (raw) for physics loss on train set
        next_grid_train=next_grid[train_idx],  # [N_train, nlat_s, nlon_s, n_vars]
        out_mean=out_mean,              # [n_vars]
        out_std=out_std,               # [n_vars]
        b_mean=b_mean.astype(np.float32),
        b_std=b_std.astype(np.float32),
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        nlat_s=nlat_s,
        nlon_s=nlon_s,
        n_sensors=n_sensors,
        n_vars=n_vars,
        cases=np.arange(N, dtype=np.int64),
    )


# ── Model ─────────────────────────────────────────────────────────────────────

def _make_mlp(layer_sizes, activation):
    """Build a fully-connected MLP with the given layer sizes."""
    act_fn = {"tanh": nn.Tanh, "relu": nn.ReLU, "gelu": nn.GELU}[activation]
    layers = []
    for i in range(len(layer_sizes) - 1):
        layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
        if i < len(layer_sizes) - 2:
            layers.append(act_fn())
    net = nn.Sequential(*layers)
    # Glorot (Xavier) uniform initialisation
    for m in net.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)
    return net


class MultivarDeepONet(nn.Module):
    """Joint DeepONet with one shared trunk and one branch per variable.

    Design note — shared trunk: the trunk output is reused for all six variables,
    which implicitly assumes they share the same latent spatial structure. This
    keeps parameter count low and encourages shared spatial representations, but
    may be overly restrictive (e.g. SSH and MLD have very different spatial
    correlation lengths than velocity). An alternative is per-variable trunk
    networks at the cost of ~6× trunk parameters.

    Forward pass
    ------------
    branch_input : [N, D_branch]   normalised flattened current state
    trunk_input  : [n_sensors, 2]  normalised (lon, lat) query coords
    returns      : [N, n_sensors, n_vars]  predicted normalised next state
    """

    def __init__(self, d_branch, n_sensors, n_vars,
                 branch_width, branch_depth,
                 trunk_width,  trunk_depth,
                 latent_dim, activation="tanh", residual=True):
        super().__init__()
        self.n_vars     = n_vars
        self.n_sensors  = n_sensors
        self.latent_dim = latent_dim
        self.residual   = residual

        branch_sizes = [d_branch] + [branch_width] * branch_depth + [latent_dim]
        trunk_sizes  = [2]        + [trunk_width]  * trunk_depth  + [latent_dim]

        self.trunk = _make_mlp(trunk_sizes, activation)
        self.branches = nn.ModuleList([
            _make_mlp(branch_sizes, activation) for _ in range(n_vars)
        ])
        # One scalar bias per variable (following Lu et al. 2021)
        self.biases = nn.Parameter(torch.zeros(n_vars))

        if residual:
            # Zero the last linear layer of every branch so the DeepONet
            # contribution starts at exactly zero, making the full output
            # equal to the current state (persistence) at step 0.
            for branch in self.branches:
                last_linear = None
                for layer in branch.modules():
                    if isinstance(layer, nn.Linear):
                        last_linear = layer
                if last_linear is not None:
                    nn.init.zeros_(last_linear.weight)
                    nn.init.zeros_(last_linear.bias)

    def forward(self, branch_input, trunk_input, persist_at_query=None):
        """persist_at_query : optional [N, n_vars, n_query] normalised persistence
        field evaluated AT the trunk's query coordinates, in the same per-variable
        scalar normalisation as the output (out_mean/out_std).  When None (the
        default, and the only path used anywhere in this study prior to the
        discretization-invariance work), the residual is read from branch_input
        itself, which is only correct when trunk_input queries the branch's own
        sensor grid at the branch's own resolution -- the original, resolution-
        locked behaviour, preserved exactly for backward compatibility with every
        existing checkpoint and result in this study.

        When trunk_input queries a DIFFERENT resolution or set of coordinates
        than the branch's own sensors (e.g. zero-shot resolution transfer,
        Sec. 3.7), branch_input has no entry corresponding to those query points,
        so the residual must instead be supplied explicitly at the query
        resolution via persist_at_query. This is what makes the persistence skip
        -- and therefore the whole architecture -- genuinely resolution-
        independent, rather than only the trunk MLP being so.
        """
        # trunk_feats : [n_query, latent_dim]
        trunk_feats = self.trunk(trunk_input)

        outputs = []
        for vi in range(self.n_vars):
            # branch_feats : [N, latent_dim]
            branch_feats = self.branches[vi](branch_input)
            # dot product → [N, n_query]
            out = torch.einsum("np,sp->ns", branch_feats, trunk_feats)
            out = out + self.biases[vi]
            if self.residual:
                if persist_at_query is not None:
                    # Resolution-independent residual: persistence supplied at
                    # the trunk's own query coordinates, whatever they are.
                    out = out + persist_at_query[:, vi, :]
                else:
                    # Original behaviour: persistence skip: branch_input is
                    # variable-major, so columns [vi*n_sensors : (vi+1)*n_sensors]
                    # are the normalised current state for variable vi.  At init
                    # (DeepONet ≈ 0) the model predicts the current state, i.e.
                    # starts at the persistence baseline. Only valid when
                    # trunk_input queries the branch's own sensor grid.
                    out = out + branch_input[:, vi * self.n_sensors : (vi + 1) * self.n_sensors]
            outputs.append(out)

        # Stack to [N, n_query, n_vars]
        return torch.stack(outputs, dim=-1)


# ── Physics losses (finite differences) ───────────────────────────────────────

def _fd_grad_lon(field, dlon_rad):
    """Central finite differences along longitude axis (axis=-1).

    field   : [..., nlat_s, nlon_s]
    dlon_rad: scalar spacing in radians
    returns : [..., nlat_s, nlon_s]  ∂field/∂λ  (boundary: one-sided)
    """
    g = torch.zeros_like(field)
    g[..., 1:-1] = (field[..., 2:] - field[..., :-2]) / (2.0 * dlon_rad)
    g[..., 0]    = (field[..., 1]  - field[..., 0])   / dlon_rad
    g[..., -1]   = (field[..., -1] - field[..., -2])  / dlon_rad
    return g


def _fd_grad_lat(field, dlat_rad):
    """Central finite differences along latitude axis (axis=-2).

    field   : [..., nlat_s, nlon_s]
    dlat_rad: scalar spacing in radians
    returns : [..., nlat_s, nlon_s]  ∂field/∂φ  (boundary: one-sided)
    """
    g = torch.zeros_like(field)
    g[..., 1:-1, :] = (field[..., 2:, :] - field[..., :-2, :]) / (2.0 * dlat_rad)
    g[..., 0, :]    = (field[..., 1, :]  - field[..., 0, :])   / dlat_rad
    g[..., -1, :]   = (field[..., -1, :] - field[..., -2, :])  / dlat_rad
    return g


def physics_losses(pred_grid, lat_rad_grid, lon_rad_1d, lat_rad_1d, ocean_mask_grid=None):
    """Compute divergence-free and geostrophic consistency losses.

    Parameters
    ----------
    pred_grid   : [N, nlat_s, nlon_s, n_vars]  predicted fields in physical units
                  (denormalised by the caller via denorm_grid before this call).
                  pred_grid[..., vi] corresponds to variable vi in VARIABLES; this
                  matches out_mean[vi]/out_std[vi] because both index variables last.
    lat_rad_grid: [nlat_s, nlon_s]  latitude in radians for each grid point
    lon_rad_1d  : [nlon_s]  longitude in radians
    lat_rad_1d  : [nlat_s]  latitude  in radians
    ocean_mask_grid : [nlat_s, nlon_s] bool tensor or None.  When given, the loss
                  is averaged over ocean cells only.  Land/mask points are zeroed
                  (nan_to_num) rather than masked, so FD gradients at coastal cells
                  are corrupted by the 0→ocean discontinuity; restricting the mean
                  to interior ocean cells keeps that contamination out of the loss.

    Returns
    -------
    l_div, l_geo : scalar tensors
    """
    a = EARTH_RADIUS

    dlon_rad = float(lon_rad_1d[1] - lon_rad_1d[0]) if len(lon_rad_1d) > 1 else 1.0
    dlat_rad = float(lat_rad_1d[1] - lat_rad_1d[0]) if len(lat_rad_1d) > 1 else 1.0

    u_hat = pred_grid[..., I_UO]    # [N, nlat_s, nlon_s]
    v_hat = pred_grid[..., I_VO]
    eta   = pred_grid[..., I_ZOS]

    cos_phi = torch.cos(lat_rad_grid)   # [nlat_s, nlon_s]

    # Masked mean over ocean cells (falls back to a plain mean when no mask given).
    # sq is [N, nlat_s, nlon_s]; the mask [nlat_s, nlon_s] broadcasts over N, so
    # the normaliser is (n_ocean_cells × N).
    def _masked_mean(sq):
        if ocean_mask_grid is None:
            return sq.mean()
        m = ocean_mask_grid.to(sq.dtype)                # [nlat_s, nlon_s]
        denom = m.sum().clamp(min=1.0) * sq.shape[0]
        return (sq * m).sum() / denom

    # ── Divergence-free loss ──────────────────────────────────────────────────
    # (1/(a·cosφ)) ∂u/∂λ + (1/(a·cosφ)) ∂(v·cosφ)/∂φ = 0
    du_dlon  = _fd_grad_lon(u_hat, dlon_rad)           # ∂u/∂λ
    vcos     = v_hat * cos_phi
    dvcos_dlat = _fd_grad_lat(vcos, dlat_rad)           # ∂(v·cosφ)/∂φ

    div = (du_dlon + dvcos_dlat) / (a * cos_phi.clamp(min=1e-6))
    l_div = _masked_mean(div ** 2)

    # ── Geostrophic consistency loss ──────────────────────────────────────────
    # u_g = -(g/(f·a)) ∂η/∂φ        f = 2·Ω·sinφ
    # v_g =  (g/(f·a·cosφ)) ∂η/∂λ
    sin_phi = torch.sin(lat_rad_grid)
    f = 2.0 * OMEGA_EARTH * sin_phi                    # Coriolis  [nlat_s, nlon_s]
    # Clamp |f| to avoid division by zero near equator (not in our domain, but safe)
    f_safe = torch.where(f.abs() < 1e-6, torch.full_like(f, 1e-6), f)

    deta_dlat = _fd_grad_lat(eta, dlat_rad)
    deta_dlon = _fd_grad_lon(eta, dlon_rad)

    u_g = -(GRAVITY / (f_safe * a)) * deta_dlat
    v_g =  (GRAVITY / (f_safe * a * cos_phi.clamp(min=1e-6))) * deta_dlon

    l_geo = _masked_mean((u_hat - u_g) ** 2 + (v_hat - v_g) ** 2)

    return l_div, l_geo


# ── Training loop ─────────────────────────────────────────────────────────────

def to_tensor(arr, device):
    return torch.tensor(arr, dtype=torch.float32, device=device)


def denorm_grid(pred_norm_grid, out_mean, out_std, device):
    """Denormalise predicted grid [N, nlat_s, nlon_s, n_vars] back to physical units.

    out_mean/out_std are [n_vars] vectors; broadcasting over [..., n_vars] is safe
    because pred_norm_grid[..., vi] and out_mean[vi] both address variable vi last.
    """
    mean_t = torch.tensor(out_mean, dtype=torch.float32, device=device)
    std_t  = torch.tensor(out_std,  dtype=torch.float32, device=device)
    return pred_norm_grid * std_t + mean_t


def run_training(model, ds, lon_sub, lat_sub, args, device):
    """Full training loop with data + physics losses.

    Returns lists of (step, train_loss, test_loss, l_data, l_div, l_geo).
    """
    branch_train = to_tensor(ds["branch_train"], device)
    branch_val   = to_tensor(ds["branch_val"],   device)
    trunk        = to_tensor(ds["trunk"],        device)
    y_train_norm = to_tensor(ds["y_train_norm"], device)
    y_val_norm   = to_tensor(ds["y_val_norm"],   device)
    has_val      = branch_val.shape[0] > 0

    n_sensors = ds["n_sensors"]
    nlat_s    = ds["nlat_s"]
    nlon_s    = ds["nlon_s"]
    n_vars    = ds["n_vars"]
    out_mean  = ds["out_mean"]
    out_std   = ds["out_std"]

    # Variable-major ocean mask: tile [n_sensors] bool → [n_vars*n_sensors] then to tensor.
    # Used to exclude land sensors (always zero) from the data loss.
    ocean_mask_vm = torch.tensor(
        np.tile(ds["ocean_mask"], n_vars), dtype=torch.bool, device=device
    )
    # Grid-shaped ocean mask [nlat_s, nlon_s] for the physics losses, so FD
    # residuals are averaged over ocean cells only (land zeros excluded).
    ocean_mask_grid = torch.tensor(
        ds["ocean_mask"].reshape(nlat_s, nlon_s), dtype=torch.bool, device=device
    )

    # Data-loss weighting.  w_ocean holds the per-ocean-column weights; with mean 1
    # the weighted mean stays on the same scale as the plain mean.  --loss-weight
    # none → uniform; variability → skill-aligned inverse-increment-variance weights.
    use_wloss = args.loss_weight == "variability"
    w_ocean = to_tensor(ds["loss_weight"], device)[ocean_mask_vm]   # [n_ocean_cols]

    def data_loss(pred_vm, tgt_vm):
        e2 = (pred_vm[:, ocean_mask_vm] - tgt_vm[:, ocean_mask_vm]) ** 2
        return (e2 * w_ocean).mean() if use_wloss else e2.mean()

    # Pre-compute lat/lon grids in radians (fixed, not trainable)
    lon_rad_1d = torch.tensor(lon_sub * DEG2RAD, dtype=torch.float32, device=device)
    lat_rad_1d = torch.tensor(lat_sub * DEG2RAD, dtype=torch.float32, device=device)
    LAT_rad, _ = torch.meshgrid(lat_rad_1d, lon_rad_1d, indexing="ij")  # [nlat_s, nlon_s]

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # Batch indices (full-batch if batch_size is None)
    N_train = branch_train.shape[0]
    batch_size = N_train if args.batch_size is None else args.batch_size

    history = {"steps": [], "train_loss": [], "val_loss": [],
               "l_data": [], "l_div": [], "l_geo": []}

    # Early stopping on validation loss (best-checkpoint selection).
    best_val   = float("inf")
    best_state = None
    best_step  = 0
    evals_no_improve = 0

    es = f"patience={args.patience}" if (args.patience > 0 and has_val) else "off"
    print(f"\nTraining for up to {args.iterations} iterations …")
    print(f"  λ_div={args.lambda_div}  λ_geo={args.lambda_geo}  warmup={args.warmup_steps}"
          f"  loss-weight={args.loss_weight}")
    print(f"  batch_size={'full' if args.batch_size is None else args.batch_size}"
          f"  early-stop={es}\n")
    val_col = "Val" if has_val else "Train*"
    header = f"{'Step':>6}  {'Train':>10}  {val_col:>10}  {'L_data':>10}  {'L_div':>10}  {'L_geo':>10}"
    print(header)
    print("─" * len(header))

    for step in range(1, args.iterations + 1):
        model.train()

        # Mini-batch sampling
        if batch_size < N_train:
            idx = torch.randperm(N_train, device=device)[:batch_size]
            b_in = branch_train[idx]
            y_tgt = y_train_norm[idx]
        else:
            b_in  = branch_train
            y_tgt = y_train_norm

        optimizer.zero_grad()

        # Forward pass → [N_batch, n_sensors, n_vars]
        pred = model(b_in, trunk)

        # ── Data loss (ocean-masked; optionally variability-weighted) ─────────
        # pred: [N, n_sensors, n_vars] → variable-major flat to match y_tgt
        pred_var_major = pred.permute(0, 2, 1).reshape(pred.shape[0], -1)
        l_data = data_loss(pred_var_major, y_tgt)

        # ── Physics losses (RAW, weighted directly by λ — no ref-normalisation) ─
        l_div = torch.tensor(0.0, device=device)
        l_geo = torch.tensor(0.0, device=device)

        if args.lambda_div > 0.0 or args.lambda_geo > 0.0:
            # Denormalise predictions for physical-unit gradients, reshape to grid
            pred_grid = denorm_grid(pred, out_mean, out_std, device)
            pred_grid = pred_grid.reshape(pred.shape[0], nlat_s, nlon_s, n_vars)
            l_div, l_geo = physics_losses(
                pred_grid, LAT_rad, lon_rad_1d, lat_rad_1d, ocean_mask_grid
            )

        ramp = min(1.0, step / max(args.warmup_steps, 1))
        loss = l_data + ramp * args.lambda_div * l_div + ramp * args.lambda_geo * l_geo
        loss.backward()
        optimizer.step()

        # ── Logging + validation / early stopping ─────────────────────────────
        if step % args.display_every == 0 or step == 1:
            model.eval()
            if has_val:
                with torch.no_grad():
                    pred_val = model(branch_val, trunk)
                    pred_val_vm = pred_val.permute(0, 2, 1).reshape(pred_val.shape[0], -1)
                    val_loss = data_loss(pred_val_vm, y_val_norm).item()
            else:
                # No validation set: report data loss so the column is meaningful.
                val_loss = l_data.item()

            train_loss_val = loss.item()
            history["steps"].append(step)
            history["train_loss"].append(train_loss_val)
            history["val_loss"].append(val_loss)
            history["l_data"].append(l_data.item())
            history["l_div"].append(l_div.item() if isinstance(l_div, torch.Tensor) else l_div)
            history["l_geo"].append(l_geo.item() if isinstance(l_geo, torch.Tensor) else l_geo)

            marker = ""
            if has_val and val_loss < best_val - 1e-9:
                best_val   = val_loss
                best_step  = step
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                evals_no_improve = 0
                marker = "  *best"
            elif has_val:
                evals_no_improve += 1

            print(f"{step:>6}  {train_loss_val:>10.3e}  {val_loss:>10.3e}  "
                  f"{l_data.item():>10.3e}  "
                  f"{history['l_div'][-1]:>10.3e}  "
                  f"{history['l_geo'][-1]:>10.3e}{marker}")

            # Early stop: patience is counted in evaluations, not raw steps.
            if has_val and args.patience > 0 and evals_no_improve >= args.patience:
                print(f"\nEarly stopping at step {step}: no val improvement for "
                      f"{args.patience} evals (best {best_val:.3e} @ step {best_step}).")
                break

    # Restore the best-validation checkpoint so evaluation uses the model that
    # generalised best, not the (possibly overfit) final-step weights.
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        print(f"Restored best-val checkpoint from step {best_step} "
              f"(val {best_val:.3e}).")

    # Unweighted validation MSE of the restored (best) model. Unlike best_val —
    # which is on the weighted scale and NOT comparable across --loss-weight
    # settings — this plain ocean-masked MSE is computed identically for every
    # config, so it's the fair, leak-free metric for ranking a sweep.
    val_mse_unweighted = float("nan")
    if has_val:
        model.eval()
        with torch.no_grad():
            pv = model(branch_val, trunk).permute(0, 2, 1).reshape(branch_val.shape[0], -1)
            val_mse_unweighted = nn.functional.mse_loss(
                pv[:, ocean_mask_vm], y_val_norm[:, ocean_mask_vm]
            ).item()
    history["val_mse_unweighted"] = val_mse_unweighted

    return history


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(model, ds, device):
    """Run inference on test set and return raw-unit predictions and truths."""
    branch_test = to_tensor(ds["branch_test"], device)
    trunk       = to_tensor(ds["trunk"],       device)
    out_mean    = ds["out_mean"]
    out_std     = ds["out_std"]
    nlat_s      = ds["nlat_s"]
    nlon_s      = ds["nlon_s"]
    n_vars      = ds["n_vars"]
    n_sensors   = ds["n_sensors"]

    model.eval()
    with torch.no_grad():
        pred_norm = model(branch_test, trunk).cpu().numpy()  # [N_test, n_sensors, n_vars]

    # Denormalise per variable
    pred_raw = pred_norm.copy()
    for vi in range(n_vars):
        pred_raw[:, :, vi] = pred_norm[:, :, vi] * out_std[vi] + out_mean[vi]

    # Variable-major flat layout [N_test, n_vars*n_sensors] to match utils expectations
    N_test = pred_raw.shape[0]
    pred_flat = pred_raw.transpose(0, 2, 1).reshape(N_test, -1)   # var-major
    true_flat = ds["y_test_raw"]                                    # [N_test, n_vars*n_sensors]

    return pred_flat, true_flat, pred_raw   # pred_raw: [N_test, n_sensors, n_vars]


# ── Metrics & I/O (inlined; this file is standalone) ──────────────────────────

def regression_metrics(y_true, y_pred):
    """Aggregate error metrics over all variables/sensors (mixed-unit reference)."""
    diff = y_pred - y_true
    rmse = float(np.sqrt(np.mean(diff * diff)))
    mae  = float(np.mean(np.abs(diff)))
    rel_l2 = np.sqrt(np.sum(diff * diff, axis=1)) / np.maximum(
        np.sqrt(np.sum(y_true * y_true, axis=1)), 1.0e-12
    )
    return {
        "rmse":              rmse,
        "mae":               mae,
        "mean_relative_l2":  float(np.mean(rel_l2)),
        "median_relative_l2": float(np.median(rel_l2)),
        "max_relative_l2":   float(np.max(rel_l2)),
    }


def anomaly_correlation(pred, true, clim):
    """Anomaly correlation coefficient (ACC) vs a climatological reference field.

    pred, true : [N, n_points]  predicted / true values (already ocean-masked)
    clim       : [n_points]     climatological mean for those points

    ACC = Σ p'·t' / √(Σ p'² · Σ t'²),  p' = pred - clim,  t' = true - clim,
    summed over every sample and point.  Returns NaN if either anomaly field is
    flat.  An ACC > 0.6 is the conventional threshold of useful forecast skill.
    """
    p = pred - clim[None, :]
    t = true - clim[None, :]
    num = float(np.sum(p * t))
    den = float(np.sqrt(np.sum(p * p) * np.sum(t * t)))
    return num / den if den > 1e-12 else float("nan")


def save_json(path, payload):
    def convert(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return float(value)
        return value

    clean = {k: convert(v) for k, v in payload.items()}
    Path(path).write_text(json.dumps(clean, indent=2, sort_keys=True) + "\n")


# ── Visualisations (inlined SVG helpers) ──────────────────────────────────────

def _svg_header(width, height):
    return [
        '<svg xmlns="http://www.w3.org/2000/svg" width="{}" height="{}" viewBox="0 0 {} {}">'.format(
            width, height, width, height
        ),
        '<rect width="100%" height="100%" fill="white"/>',
    ]


def _scale(values, lo, hi, out_lo, out_hi):
    values = np.asarray(values, dtype=np.float64)
    if abs(hi - lo) < 1.0e-12:
        return np.full_like(values, 0.5 * (out_lo + out_hi))
    return out_lo + (values - lo) * (out_hi - out_lo) / (hi - lo)


def _polyline(x, y, color, width=2.0, dash=None):
    points = " ".join("{:.2f},{:.2f}".format(float(a), float(b)) for a, b in zip(x, y))
    dash_attr = ' stroke-dasharray="{}"'.format(dash) if dash else ""
    return '<polyline points="{}" fill="none" stroke="{}" stroke-width="{:.1f}"{} />'.format(
        points, color, width, dash_attr
    )


def _axes(elements, width, height, title, xlabel, ylabel):
    left, right, top, bottom = 70, width - 25, 35, height - 55
    elements.append('<line x1="{0}" y1="{1}" x2="{2}" y2="{1}" stroke="#222"/>'.format(left, bottom, right))
    elements.append('<line x1="{0}" y1="{1}" x2="{0}" y2="{2}" stroke="#222"/>'.format(left, top, bottom))
    elements.append('<text x="{}" y="22" font-size="17" font-family="sans-serif">{}</text>'.format(left, title))
    elements.append('<text x="{}" y="{}" font-size="13" font-family="sans-serif">{}</text>'.format((left + right) / 2 - 30, height - 15, xlabel))
    elements.append('<text x="18" y="{}" font-size="13" font-family="sans-serif" transform="rotate(-90 18,{})">{}</text>'.format((top + bottom) / 2 + 35, (top + bottom) / 2 + 35, ylabel))
    return left, right, top, bottom


def _write_loss_svg(path, history):
    width, height = 720, 420
    elements = _svg_header(width, height)
    left, right, top, bottom = _axes(elements, width, height, "DeepONet training history", "epoch", "log10 MSE")
    train = np.log10(np.maximum(np.asarray(history["train_mse"]), 1.0e-14))
    test = np.log10(np.maximum(np.asarray(history["test_mse"]), 1.0e-14))
    x = np.arange(1, len(train) + 1)
    ymin = float(min(train.min(), test.min()))
    ymax = float(max(train.max(), test.max()))
    px = _scale(x, x.min(), x.max(), left, right)
    elements.append(_polyline(px, _scale(train, ymin, ymax, bottom, top), "#1f77b4", 2.4))
    elements.append(_polyline(px, _scale(test, ymin, ymax, bottom, top), "#d62728", 2.4))
    elements.append('<text x="530" y="55" font-size="13" font-family="sans-serif" fill="#1f77b4">train</text>')
    elements.append('<text x="530" y="75" font-size="13" font-family="sans-serif" fill="#d62728">val</text>')
    elements.append("</svg>")
    path.write_text("\n".join(elements) + "\n")


def _write_ocean_parity_svg(path, y_true, y_pred):
    width, height = 500, 500
    elements = _svg_header(width, height)
    left, right, top, bottom = _axes(
        elements, width, height, "Parity plot", "true value", "predicted value"
    )
    truth = y_true.reshape(-1)
    pred  = y_pred.reshape(-1)
    step  = max(1, int(math.ceil(float(truth.size) / 900.0)))
    truth = truth[::step]
    pred  = pred[::step]
    lo = float(min(truth.min(), pred.min()))
    hi = float(max(truth.max(), pred.max()))
    if abs(hi - lo) < 1e-12:
        hi = lo + 1.0
    px = _scale(truth, lo, hi, left, right)
    py = _scale(pred,  lo, hi, bottom, top)
    elements.append(_polyline([left, right], [bottom, top], "#333", 1.2, dash="4,4"))
    for x, y in zip(px, py):
        elements.append(
            '<circle cx="{:.2f}" cy="{:.2f}" r="1.8" fill="#1f77b4" fill-opacity="0.45"/>'.format(
                float(x), float(y)
            )
        )
    elements.append("</svg>")
    path.write_text("\n".join(elements) + "\n")


def _write_ocean_forecast_svg(path, y_true, y_pred, case_ids, variables=None):
    """Plot first-variable predictions vs. truth for up to 4 test cases.

    y_true/y_pred are [N_test, n_vars*n_sensors] in variable-major layout, so
    columns [0 : n_sensors] correspond to the first variable.
    """
    width, height = 760, 460
    elements = _svg_header(width, height)
    var_name = variables[0] if variables else "var0"
    left, right, top, bottom = _axes(
        elements, width, height,
        "Forecast vs. truth — {}".format(var_name), "sensor index", var_name
    )
    n_vars = len(variables) if variables else 1
    n_pts  = y_true.shape[1] // n_vars
    count  = min(4, y_true.shape[0])
    ylo = float(min(y_true[:count, :n_pts].min(), y_pred[:count, :n_pts].min()))
    yhi = float(max(y_true[:count, :n_pts].max(), y_pred[:count, :n_pts].max()))
    if abs(yhi - ylo) < 1e-12:
        yhi = ylo + 1.0
    x  = np.arange(n_pts, dtype=np.float64)
    px = _scale(x, x.min(), x.max(), left, right)
    colors = ["#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e"]
    for i in range(count):
        true_y = _scale(y_true[i, :n_pts], ylo, yhi, bottom, top)
        pred_y = _scale(y_pred[i, :n_pts], ylo, yhi, bottom, top)
        color  = colors[i % len(colors)]
        elements.append(_polyline(px, true_y, color, 2.2))
        elements.append(_polyline(px, pred_y, color, 2.0, dash="5,4"))
        elements.append(
            '<text x="{}" y="{}" font-size="12" font-family="sans-serif" fill="{}">t={} true/--pred</text>'.format(
                500, 55 + 18 * i, color, int(case_ids[i])
            )
        )
    elements.append("</svg>")
    path.write_text("\n".join(elements) + "\n")


def write_ocean_visualizations(out_dir, y_true, y_pred, history, case_ids, variables=None):
    """Loss-history, parity, and forecast-example SVGs for the ocean pipeline."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "loss":     out_dir / "loss_history.svg",
        "forecast": out_dir / "forecast_examples.svg",
        "parity":   out_dir / "parity.svg",
    }
    _write_loss_svg(paths["loss"], history)
    _write_ocean_parity_svg(paths["parity"], y_true, y_pred)
    _write_ocean_forecast_svg(paths["forecast"], y_true, y_pred, case_ids, variables)
    return paths


_LINE_COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf",
                "#8c564b", "#e377c2"]


def _write_lines_svg(path, x, series, title, xlabel, ylabel, hline=None,
                     ymin=None, ymax=None):
    """Generic multi-line plot.  series: list of (label, y, color, dash)."""
    width, height = 760, 460
    elements = _svg_header(width, height)
    left, right, top, bottom = _axes(elements, width, height, title, xlabel, ylabel)
    x = np.asarray(x, dtype=np.float64)
    all_y = np.concatenate([np.asarray(y, dtype=np.float64) for _, y, _, _ in series])
    lo = float(all_y.min()) if ymin is None else ymin
    hi = float(all_y.max()) if ymax is None else ymax
    if hline is not None:
        lo, hi = min(lo, hline), max(hi, hline)
    if abs(hi - lo) < 1e-12:
        hi = lo + 1.0
    px = _scale(x, x.min(), x.max(), left, right)
    if hline is not None:
        hy = float(_scale([hline], lo, hi, bottom, top)[0])
        elements.append('<line x1="{:.1f}" y1="{:.1f}" x2="{:.1f}" y2="{:.1f}" stroke="#999" stroke-dasharray="4,4"/>'.format(left, hy, right, hy))
        elements.append('<text x="{:.1f}" y="{:.1f}" font-size="11" font-family="sans-serif" fill="#999">{:g}</text>'.format(right - 30, hy - 4, hline))
    # x tick labels
    for xv, pxv in zip(x, px):
        elements.append('<text x="{:.1f}" y="{:.1f}" font-size="11" font-family="sans-serif" text-anchor="middle">{:g}</text>'.format(float(pxv), bottom + 16, float(xv)))
    for i, (label, y, color, dash) in enumerate(series):
        py = _scale(y, lo, hi, bottom, top)
        elements.append(_polyline(px, py, color, 2.2, dash=dash))
        for xv, yv in zip(px, py):
            elements.append('<circle cx="{:.2f}" cy="{:.2f}" r="2.5" fill="{}"/>'.format(float(xv), float(yv), color))
        elements.append('<text x="{}" y="{}" font-size="12" font-family="sans-serif" fill="{}">{}</text>'.format(
            right - 130, 55 + 18 * i, color, label))
    elements.append("</svg>")
    Path(path).write_text("\n".join(elements) + "\n")


def _heat_color(t):
    """Sequential blue→yellow→red ramp for t in [0,1]; returns 'rgb(r,g,b)'."""
    t = 0.0 if t < 0 else (1.0 if t > 1 else t)
    # three-stop ramp: (30,60,150) → (240,230,80) → (200,30,30)
    if t < 0.5:
        f = t / 0.5
        r, g, b = 30 + f * (240 - 30), 60 + f * (230 - 60), 150 + f * (80 - 150)
    else:
        f = (t - 0.5) / 0.5
        r, g, b = 240 + f * (200 - 240), 230 + f * (30 - 230), 80 + f * (30 - 80)
    return "rgb({},{},{})".format(int(r), int(g), int(b))


def _diverging_color(t):
    """Diverging blue(−)→white(0)→red(+) ramp for t in [-1,1]."""
    t = -1.0 if t < -1 else (1.0 if t > 1 else t)
    if t >= 0:
        r, g, b = 255, 255 - t * 215, 255 - t * 215
    else:
        s = -t
        r, g, b = 255 - s * 215, 255 - s * 215, 255
    return "rgb({},{},{})".format(int(r), int(g), int(b))


def _write_heatmap_svg(path, grid, lon, lat, title, diverging=False, vlim=None):
    """Render a [nlat, nlon] field as a lon/lat grid of colored cells.

    NaN cells (land) are drawn light gray.  For diverging maps the scale is
    symmetric about zero; otherwise it spans [min, max] (or [0, vlim]).
    """
    nlat, nlon = grid.shape
    plot_w = 560
    cell = max(3, int(plot_w / max(nlon, 1)))
    left, top = 70, 50
    width  = left + nlon * cell + 120
    height = top + nlat * cell + 60
    elements = _svg_header(width, height)
    elements.append('<text x="{}" y="26" font-size="16" font-family="sans-serif">{}</text>'.format(left, title))

    finite = grid[np.isfinite(grid)]
    if finite.size == 0:
        finite = np.array([0.0])
    if diverging:
        vmax = vlim if vlim is not None else float(np.max(np.abs(finite)))
        vmax = vmax if vmax > 1e-12 else 1.0
        vmin = -vmax
    else:
        vmin = 0.0
        vmax = vlim if vlim is not None else float(np.max(finite))
        vmax = vmax if vmax > 1e-12 else 1.0

    # Latitude decreases downward in image space; row 0 = first lat entry.
    for i in range(nlat):
        for j in range(nlon):
            v = grid[i, j]
            x = left + j * cell
            y = top + i * cell
            if not np.isfinite(v):
                color = "#e8e8e8"
            elif diverging:
                color = _diverging_color(v / vmax)
            else:
                color = _heat_color((v - vmin) / (vmax - vmin))
            elements.append('<rect x="{}" y="{}" width="{}" height="{}" fill="{}"/>'.format(
                x, y, cell, cell, color))

    # Colorbar
    bar_x = left + nlon * cell + 25
    bar_top, bar_h, bar_w = top, nlat * cell, 18
    n_stops = 40
    for k in range(n_stops):
        frac = k / (n_stops - 1)
        val_t = 1.0 - frac  # top = high
        color = _diverging_color(2 * val_t - 1) if diverging else _heat_color(val_t)
        yk = bar_top + frac * bar_h
        elements.append('<rect x="{}" y="{:.1f}" width="{}" height="{:.1f}" fill="{}"/>'.format(
            bar_x, yk, bar_w, bar_h / n_stops + 1, color))
    top_lbl = "{:+.3g}".format(vmax) if diverging else "{:.3g}".format(vmax)
    bot_lbl = "{:+.3g}".format(vmin) if diverging else "{:.3g}".format(vmin)
    elements.append('<text x="{}" y="{}" font-size="11" font-family="sans-serif">{}</text>'.format(bar_x + bar_w + 4, bar_top + 10, top_lbl))
    elements.append('<text x="{}" y="{}" font-size="11" font-family="sans-serif">{}</text>'.format(bar_x + bar_w + 4, bar_top + bar_h, bot_lbl))
    # axis extent labels
    elements.append('<text x="{}" y="{}" font-size="11" font-family="sans-serif">lon {:.1f}–{:.1f}°  lat {:.1f}–{:.1f}°</text>'.format(
        left, height - 12, float(lon.min()), float(lon.max()), float(lat.min()), float(lat.max())))
    elements.append("</svg>")
    Path(path).write_text("\n".join(elements) + "\n")


# ── Autoregressive rollout (Aim 2) ─────────────────────────────────────────────

def rollout_evaluate(model, ds, states, args, device):
    """Autoregressive multi-step forecast skill vs lead time.

    Starting from each valid test-set state, the model is rolled forward by
    feeding its own predictions back as inputs.  At each requested horizon we
    compute per-variable ocean-masked RMSE, NRMSE and ACC, averaged over all
    starts.  Land sensors (unlearned) are held fixed at their initial values so
    they cannot blow up under feedback.

    Returns a dict with horizons and [n_horizons, n_vars] metric matrices.
    """
    n_sensors = ds["n_sensors"]
    n_vars    = ds["n_vars"]
    out_mean  = ds["out_mean"]
    out_std   = ds["out_std"]
    ocean     = ds["ocean_mask"]                      # [n_sensors] bool
    clim      = ds["climatology"]                     # [n_sensors, n_vars]

    # Each model application advances k = step_days days, so a requested lead-time
    # horizon (in days) must be a positive multiple of k; reach it in h/k steps.
    k = max(1, int(args.step_days))
    requested = sorted(set(int(h) for h in args.rollout_horizons))
    horizons = [h for h in requested if h > 0 and h % k == 0]
    dropped  = [h for h in requested if h not in horizons]
    if dropped:
        print(f"  [rollout] step_days={k}: horizons {dropped} are not multiples of "
              f"{k} → skipped.")
    if not horizons:
        horizons = [k * i for i in (1, 2, 3, 4)]
        print(f"  [rollout] no requested horizon is a multiple of k={k}; "
              f"using {horizons}.")
    H = max(horizons)
    T = states.shape[0]

    # Valid starts: test current-state times t0 with t0 + H available as truth.
    starts = np.array([t for t in ds["test_idx"] if t + H <= T - 1], dtype=np.int64)
    if starts.size == 0:
        print("  [rollout] not enough test lead time for horizon "
              f"{H}; skipping rollout.")
        return None
    S = starts.size

    # Tensors on device
    b_mean = to_tensor(ds["b_mean"], device)          # [1, D]
    b_std  = to_tensor(ds["b_std"],  device)          # [1, D]
    trunk  = to_tensor(ds["trunk"],  device)
    out_mean_t = to_tensor(out_mean, device)          # [n_vars]
    out_std_t  = to_tensor(out_std,  device)
    land_mask_vm = torch.tensor(
        np.tile(~ocean, n_vars), dtype=torch.bool, device=device
    )                                                 # [n_vars*n_sensors]

    # Initial raw var-major states for every start:  [S, D]
    cur_vm = to_tensor(_var_major_flat(states[starts].astype(np.float64)), device)
    init_vm = cur_vm.clone()                          # land held at these values

    preds_at = {}  # lead-time (days) → [S, D] raw var-major
    model.eval()
    with torch.no_grad():
        for app in range(1, H // k + 1):
            lead = app * k                            # days elapsed after this application
            branch_norm = (cur_vm - b_mean) / b_std   # [S, D]
            branch_norm[:, land_mask_vm] = 0.0        # match training: neutral land input
            pred = model(branch_norm, trunk)          # [S, n_sensors, n_vars] normalised
            pred_raw = pred * out_std_t + out_mean_t  # denorm (broadcast over vars)
            pred_vm = pred_raw.permute(0, 2, 1).reshape(S, -1)   # [S, D] var-major
            # Hold land fixed at initial values
            pred_vm[:, land_mask_vm] = init_vm[:, land_mask_vm]
            cur_vm = pred_vm
            if lead in horizons:
                preds_at[lead] = cur_vm.detach().cpu().numpy()

    # Metrics per horizon per variable (ocean-masked). At each lead we score both
    # OUR MODEL and the NAIVE persistence forecast (the frozen initial state, held
    # fixed for the whole horizon) against the truth, then form the skill of model
    # over naive.  persist_vm is state(t0) for every start — the naive forecast at
    # every horizon.
    persist_vm = _var_major_flat(states[starts].astype(np.float64))     # [S, D] frozen
    rmse          = np.full((len(horizons), n_vars), np.nan)  # model error
    nrmse         = np.full((len(horizons), n_vars), np.nan)  # model error / variability
    acc           = np.full((len(horizons), n_vars), np.nan)  # model ACC vs climatology
    rmse_persist  = np.full((len(horizons), n_vars), np.nan)  # naive (persistence) error
    acc_persist   = np.full((len(horizons), n_vars), np.nan)  # naive ACC vs climatology
    skill         = np.full((len(horizons), n_vars), np.nan)  # 1-(model/naive)²; >0 beats naive
    for hi, h in enumerate(horizons):
        pred_vm = preds_at[h]                                       # [S, D]
        true_vm = _var_major_flat(states[starts + h].astype(np.float64))  # [S, D]
        for vi in range(n_vars):
            cs, ce = vi * n_sensors, (vi + 1) * n_sensors
            p  = pred_vm[:,    cs:ce][:, ocean]                     # [S, n_ocean] model
            t  = true_vm[:,    cs:ce][:, ocean]                     # truth at t0+h
            pr = persist_vm[:, cs:ce][:, ocean]                     # naive (frozen t0)
            c  = clim[ocean, vi]                                    # [n_ocean]
            rmse[hi, vi]         = float(np.sqrt(np.mean((p - t) ** 2)))
            rmse_persist[hi, vi] = float(np.sqrt(np.mean((pr - t) ** 2)))
            std_t = float(np.std(t))
            nrmse[hi, vi] = rmse[hi, vi] / std_t if std_t > 1e-12 else np.nan
            acc[hi, vi]         = anomaly_correlation(p,  t, c)
            acc_persist[hi, vi] = anomaly_correlation(pr, t, c)
            skill[hi, vi] = (1.0 - (rmse[hi, vi] / rmse_persist[hi, vi]) ** 2
                             if rmse_persist[hi, vi] > 1e-12 else np.nan)

    out = {
        "horizons":     np.array(horizons, dtype=np.int64),
        "rmse":         rmse,
        "nrmse":        nrmse,
        "acc":          acc,
        "rmse_persist": rmse_persist,
        "acc_persist":  acc_persist,
        "skill":        skill,
        "n_starts":     int(S),
        "variables":    VARIABLES,
    }

    # Optional: dump raw (model/truth/persistence) fields for a subsample of starts
    # at every horizon, for offline eddy-tracking analysis (py-eddy-tracker etc.),
    # which the scalar metrics above cannot support. Off by default (adds no cost
    # to the normal path); gated by --dump-rollout-fields N.
    dump_n = int(getattr(args, "dump_rollout_fields", 0) or 0)
    if dump_n > 0:
        n = min(dump_n, S)
        out["dump_starts"] = starts[:n]
        out["dump_persist"] = persist_vm[:n]           # frozen t0 state, same at every horizon
        out["dump_pred"] = {h: preds_at[h][:n] for h in horizons}
        out["dump_true"] = {
            h: _var_major_flat(states[starts[:n] + h].astype(np.float64)) for h in horizons
        }

    return out


def write_rollout_visualizations(out_dir, rollout, variables):
    """ACC/NRMSE-vs-lead plots, plus a naive-vs-model comparison over lead time."""
    horizons = rollout["horizons"]
    acc_series = [
        (variables[vi], rollout["acc"][:, vi], _LINE_COLORS[vi % len(_LINE_COLORS)], None)
        for vi in range(len(variables))
    ]
    _write_lines_svg(
        out_dir / "rollout_acc.svg", horizons, acc_series,
        "Autoregressive skill (ACC) vs lead time", "lead time (days)", "ACC",
        hline=0.6, ymin=-0.1, ymax=1.0,
    )
    nrmse_series = [
        (variables[vi], rollout["nrmse"][:, vi], _LINE_COLORS[vi % len(_LINE_COLORS)], None)
        for vi in range(len(variables))
    ]
    _write_lines_svg(
        out_dir / "rollout_nrmse.svg", horizons, nrmse_series,
        "Autoregressive error (NRMSE) vs lead time", "lead time (days)", "NRMSE",
        hline=1.0, ymin=0.0,
    )
    # Naive-vs-model over lead time, averaged over variables: (1) mean skill of the
    # model over the naive persistence forecast (>0 = model wins), (2) mean ACC of
    # model (solid) vs naive (dashed) so you can see both patterns decay.
    if "skill" in rollout:
        _write_lines_svg(
            out_dir / "rollout_skill_vs_naive.svg", horizons,
            [("model skill vs naive", np.nanmean(rollout["skill"], axis=1), "#1f77b4", None)],
            "Skill of model over naive (persistence) vs lead time",
            "lead time (days)", "skill  (>0 beats naive)", hline=0.0,
        )
        _write_lines_svg(
            out_dir / "rollout_acc_vs_naive.svg", horizons,
            [("model", np.nanmean(rollout["acc"], axis=1), "#1f77b4", None),
             ("naive", np.nanmean(rollout["acc_persist"], axis=1), "#d62728", "5,4")],
            "Pattern accuracy (ACC): model vs naive over lead time",
            "lead time (days)", "mean ACC", hline=0.6, ymin=-0.1, ymax=1.0,
        )


def write_comparison_visualizations(out_dir, this_run, other_dir, variables):
    """Overlay this run's rollout skill against another run's (e.g. baseline).

    Compares mean-over-variables ACC and NRMSE vs lead time, and a spatial RMSE
    difference map (other − this) for the primary variable.  Returns True on
    success, False if the other run's artefacts are missing/mismatched.
    """
    other_npz = Path(other_dir) / "rollout.npz"
    if not other_npz.exists():
        print(f"  [compare] {other_npz} not found; skipping comparison plots.")
        return False
    other = np.load(other_npz, allow_pickle=True)
    if not np.array_equal(other["horizons"], this_run["horizons"]):
        print("  [compare] horizon mismatch between runs; skipping comparison.")
        return False

    horizons = this_run["horizons"]
    this_label  = Path(out_dir).name
    other_label = Path(other_dir).name

    # Mean ACC / NRMSE across variables
    this_acc  = np.nanmean(this_run["acc"],  axis=1)
    other_acc = np.nanmean(other["acc"],     axis=1)
    _write_lines_svg(
        out_dir / "rollout_compare_acc.svg", horizons,
        [(this_label, this_acc, "#1f77b4", None),
         (other_label, other_acc, "#d62728", "5,4")],
        "Mean ACC vs lead time — {} vs {}".format(this_label, other_label),
        "lead time (days)", "mean ACC", hline=0.6, ymin=-0.1, ymax=1.0,
    )
    this_nrmse  = np.nanmean(this_run["nrmse"], axis=1)
    other_nrmse = np.nanmean(other["nrmse"],    axis=1)
    _write_lines_svg(
        out_dir / "rollout_compare_nrmse.svg", horizons,
        [(this_label, this_nrmse, "#1f77b4", None),
         (other_label, other_nrmse, "#d62728", "5,4")],
        "Mean NRMSE vs lead time — {} vs {}".format(this_label, other_label),
        "lead time (days)", "mean NRMSE", hline=1.0, ymin=0.0,
    )

    # Spatial RMSE difference (other − this): positive → this run is better.
    other_sp = Path(other_dir) / "spatial_rmse.npz"
    this_sp  = Path(out_dir)  / "spatial_rmse.npz"
    if other_sp.exists() and this_sp.exists():
        a = np.load(this_sp);  b = np.load(other_sp)
        if a["grid"].shape == b["grid"].shape and str(a["variable"]) == str(b["variable"]):
            diff = b["grid"] - a["grid"]
            _write_heatmap_svg(
                Path(out_dir) / "spatial_rmse_diff_{}.svg".format(str(a["variable"])),
                diff, a["lon"], a["lat"],
                "RMSE diff ({} − {}), {}: red = {} better".format(
                    other_label, this_label, str(a["variable"]), this_label),
                diverging=True,
            )
    return True


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # 1. Load data (optionally from / to a subsampled cache)
    lon_sub, lat_sub, states = load_states(args.nc, args.subsample_r, args.cache)
    if args.prepare_cache:
        print("Cache prepared; exiting (--prepare-cache).")
        return
    T, nlat_s, nlon_s, n_vars = states.shape
    n_sensors = nlat_s * nlon_s

    ds = build_dataset(states, lon_sub, lat_sub, args.test_fraction, args.val_fraction,
                       step_days=args.step_days, embargo=args.embargo)
    print(f"\nDataset summary:")
    print(f"  Forecast step Δt      : {args.step_days} day(s)")
    print(f"  Samples (t→t+{args.step_days} pairs): {T - args.step_days}")
    print(f"  Branch input dim      : {ds['branch_train'].shape[1]:,}")
    print(f"  Trunk points          : {n_sensors:,}")
    print(f"  Train / Val / Test    : {len(ds['train_idx'])} / "
          f"{len(ds['val_idx'])} / {len(ds['test_idx'])}  (chronological)"
          + (f"  [embargo={args.embargo}]" if args.embargo > 0 else ""))

    # 2. Build joint model
    model = MultivarDeepONet(
        d_branch     = ds["branch_train"].shape[1],
        n_sensors    = n_sensors,
        n_vars       = n_vars,
        branch_width = args.branch_width,
        branch_depth = args.branch_depth,
        trunk_width  = args.trunk_width,
        trunk_depth  = args.trunk_depth,
        latent_dim   = args.latent_dim,
        activation   = "tanh",
        residual     = not args.no_residual,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters  : {n_params:,}")

    # 3. Train
    history = run_training(model, ds, lon_sub, lat_sub, args, device)

    # 4. Evaluate
    pred_flat, true_flat, pred_3d = evaluate(model, ds, device)

    # pred_3d shape: [N_test, n_sensors, n_vars]
    # true_flat and pred_flat: [N_test, n_vars*n_sensors]  (variable-major)
    metrics = regression_metrics(true_flat, pred_flat)

    # Reconstruct variable-indexed 3-D arrays in physical units
    N_test    = pred_3d.shape[0]
    true_3d   = ds["y_test_raw"].reshape(N_test, n_vars, n_sensors).transpose(0, 2, 1)
    persist_flat = ds["x_test_raw"]  # [N_test, n_vars*n_sensors], current state = persistence forecast
    climatology  = ds["climatology"]  # [n_sensors, n_vars] training-mean reference
    # true_3d, pred_3d: [N_test, n_sensors, n_vars]

    ocean_mask = ds["ocean_mask"]   # [n_sensors] bool
    for vi, vname in enumerate(VARIABLES):
        col_s, col_e = vi * n_sensors, (vi + 1) * n_sensors
        true_vi  = true_3d[:, ocean_mask, vi]   # [N_test, n_ocean]
        pred_vi  = pred_3d[:, ocean_mask, vi]
        pers_vi  = persist_flat[:, col_s:col_e][:, ocean_mask]
        clim_vi  = climatology[ocean_mask, vi]  # [n_ocean]

        rmse_m   = float(np.sqrt(np.mean((true_vi - pred_vi) ** 2)))
        rmse_p   = float(np.sqrt(np.mean((true_vi - pers_vi) ** 2)))
        std_true = float(np.std(true_vi))
        # Skill score: SS > 0 means model beats persistence; SS = 1 is perfect.
        skill    = float(1.0 - (rmse_m / rmse_p) ** 2) if rmse_p > 1e-12 else float("nan")
        # NRMSE: error as a fraction of the natural variability (< 1 is baseline target).
        nrmse    = float(rmse_m / std_true) if std_true > 1e-12 else float("nan")
        bias     = float(np.mean(pred_vi - true_vi))
        # ACC: anomaly correlation vs climatology (> 0.6 → useful forecast skill).
        acc      = anomaly_correlation(pred_vi, true_vi, clim_vi)

        metrics[f"rmse_{vname}"]         = rmse_m
        metrics[f"rmse_persist_{vname}"] = rmse_p
        metrics[f"skill_{vname}"]        = skill
        metrics[f"nrmse_{vname}"]        = nrmse
        metrics[f"bias_{vname}"]         = bias
        metrics[f"acc_{vname}"]          = acc

    metrics.update({
        "iterations":   args.iterations,
        "latent_dim":   args.latent_dim,
        "lambda_div":   args.lambda_div,
        "lambda_geo":   args.lambda_geo,
        "warmup_steps": args.warmup_steps,
        "learning_rate": args.learning_rate,
        "loss_weight":  args.loss_weight,
        "step_days":    args.step_days,
        "subsample_r":  args.subsample_r,
        "rollout_horizons": list(args.rollout_horizons),
        "spatial_var":  args.spatial_var,
        "val_fraction": args.val_fraction,
        "test_fraction": args.test_fraction,
        "embargo":      args.embargo,
        "patience":     args.patience,
        "best_val_loss": float(min(history["val_loss"])) if history["val_loss"] else float("nan"),
        "val_mse_unweighted": float(history.get("val_mse_unweighted", float("nan"))),
        "steps_run":    int(history["steps"][-1]) if history["steps"] else 0,
        "n_sensors":    int(n_sensors),
        "n_vars":       n_vars,
        "n_train":      int(len(ds["train_idx"])),
        "n_val":        int(len(ds["val_idx"])),
        "n_test":       int(len(ds["test_idx"])),
        "train_cases":  ds["cases"][ds["train_idx"]],
        "test_cases":   ds["cases"][ds["test_idx"]],
        "device":       str(device),
    })

    # 5. Save
    steps        = np.array(history["steps"],      dtype=np.int64)
    train_losses = np.array(history["train_loss"], dtype=np.float64)
    val_losses   = np.array(history["val_loss"],   dtype=np.float64)

    np.savez_compressed(
        args.out_dir / "predictions.npz",
        test_indices = ds["test_idx"],
        test_cases   = ds["cases"][ds["test_idx"]],
        y_true       = true_flat,
        y_pred       = pred_flat,
        y_persist    = ds["x_test_raw"].astype(np.float32),  # persistence forecast
        lon          = lon_sub,
        lat          = lat_sub,
        train_loss   = train_losses,
        val_loss     = val_losses,
        l_div        = np.array(history["l_div"]),
        l_geo        = np.array(history["l_geo"]),
        steps        = steps,
    )

    write_ocean_visualizations(
        out_dir   = args.out_dir,
        y_true    = true_flat,
        y_pred    = pred_flat,
        history   = {"train_mse": train_losses, "test_mse": val_losses},
        case_ids  = ds["cases"][ds["test_idx"]],
        variables = VARIABLES,
    )

    # 5b. Spatial time-averaged single-step RMSE map for the chosen variable.
    spatial_var = args.spatial_var if args.spatial_var in VARIABLES else VARIABLES[0]
    svi = VARIABLES.index(spatial_var)
    per_sensor_rmse = np.sqrt(np.mean((pred_3d[:, :, svi] - true_3d[:, :, svi]) ** 2, axis=0))
    per_sensor_rmse[~ocean_mask] = np.nan            # land → gray in the map
    rmse_grid = per_sensor_rmse.reshape(nlat_s, nlon_s)
    np.savez_compressed(
        args.out_dir / "spatial_rmse.npz",
        grid=rmse_grid, lon=lon_sub, lat=lat_sub, variable=spatial_var,
    )
    _write_heatmap_svg(
        args.out_dir / f"spatial_rmse_{spatial_var}.svg",
        rmse_grid, lon_sub, lat_sub,
        f"Single-step RMSE ({spatial_var}, {VARIABLE_UNITS.get(spatial_var, '?')})",
    )

    # 5c. Autoregressive multi-step rollout (Aim 2).
    print("\nRunning autoregressive rollout …")
    rollout = rollout_evaluate(model, ds, states, args, device)
    if rollout is not None:
        np.savez_compressed(
            args.out_dir / "rollout.npz",
            horizons=rollout["horizons"], rmse=rollout["rmse"],
            nrmse=rollout["nrmse"], acc=rollout["acc"],
            rmse_persist=rollout["rmse_persist"], acc_persist=rollout["acc_persist"],
            skill=rollout["skill"],
            n_starts=rollout["n_starts"], variables=np.array(VARIABLES),
        )
        write_rollout_visualizations(args.out_dir, rollout, VARIABLES)
        if "dump_pred" in rollout:
            field_kwargs = {f"pred_h{h}": arr for h, arr in rollout["dump_pred"].items()}
            field_kwargs.update({f"true_h{h}": arr for h, arr in rollout["dump_true"].items()})
            np.savez_compressed(
                args.out_dir / "rollout_fields.npz",
                horizons=rollout["horizons"], starts=rollout["dump_starts"],
                persist=rollout["dump_persist"], lon=lon_sub, lat=lat_sub,
                variables=np.array(VARIABLES), **field_kwargs,
            )
            print(f"  Dumped raw rollout fields for {len(rollout['dump_starts'])} starts "
                  f"to {args.out_dir / 'rollout_fields.npz'}")
        for hi, h in enumerate(rollout["horizons"]):
            for vi, vname in enumerate(VARIABLES):
                metrics[f"rollout_rmse_{vname}_{h}d"]         = float(rollout["rmse"][hi, vi])
                metrics[f"rollout_rmse_persist_{vname}_{h}d"] = float(rollout["rmse_persist"][hi, vi])
                metrics[f"rollout_acc_{vname}_{h}d"]          = float(rollout["acc"][hi, vi])
                metrics[f"rollout_skill_{vname}_{h}d"]        = float(rollout["skill"][hi, vi])

    # metrics.json is written after rollout so it captures the rollout skill too.
    save_json(args.out_dir / "metrics.json", metrics)

    # 5d. Optional PINN-vs-baseline comparison plots.
    if args.compare_to is not None and rollout is not None:
        write_comparison_visualizations(args.out_dir, rollout, args.compare_to, VARIABLES)

    torch.save(model.state_dict(), args.out_dir / "model.pt")
    print(f"\nModel saved to {args.out_dir / 'model.pt'}")

    # 6. Summary
    print(f"\n{'═'*75}")
    print("  FINAL SUMMARY")
    print(f"{'═'*75}")
    print()
    print(f"  Single-step (Δt={args.step_days}d) evaluation vs. persistence "
          f"(physical units, test set)")
    print()
    hdr = (f"  {'var':<10}  {'RMSE':>9}  {'Persist':>9}  "
           f"{'Skill':>7}  {'NRMSE':>7}  {'ACC':>7}  {'Bias':>9}  units")
    print(hdr)
    print("  " + "─" * (len(hdr) - 2))
    for vname in VARIABLES:
        units = VARIABLE_UNITS.get(vname, "?")
        sk = metrics[f"skill_{vname}"]
        print(f"  {vname:<10}  "
              f"{metrics[f'rmse_{vname}']:>9.4f}  "
              f"{metrics[f'rmse_persist_{vname}']:>9.4f}  "
              f"{sk:>+7.3f}  "
              f"{metrics[f'nrmse_{vname}']:>7.3f}  "
              f"{metrics[f'acc_{vname}']:>7.3f}  "
              f"{metrics[f'bias_{vname}']:>+9.4f}  {units}")
    print()
    print(f"  Skill > 0 → beats persistence.  NRMSE < 1 → error < natural variability.")
    print(f"  ACC > 0.6 → useful forecast skill.")

    if rollout is not None:
        hzs = rollout["horizons"]
        print()
        print("  Autoregressive rollout — ACC by lead time "
              f"(averaged over {rollout['n_starts']} starts)")
        print()
        head = "  " + f"{'var':<10}" + "".join(f"{str(h)+'d':>9}" for h in hzs)
        print(head)
        print("  " + "─" * (len(head) - 2))
        for vi, vname in enumerate(VARIABLES):
            print("  " + f"{vname:<10}" + "".join(
                f"{rollout['acc'][hi, vi]:>9.3f}" for hi in range(len(hzs))))

        # Naive vs model over lead time, averaged over variables: does the model
        # beat the frozen "no-change" forecast, and by how much, as lead grows?
        print()
        print("  Naive (persistence) vs model over lead time  (mean over variables)")
        print()
        print("  " + f"{'metric':<16}" + "".join(f"{str(h)+'d':>9}" for h in hzs))
        print("  " + "─" * (16 + 9 * len(hzs)))
        print("  " + f"{'naive ACC':<16}" + "".join(
            f"{np.nanmean(rollout['acc_persist'], axis=1)[hi]:>9.3f}" for hi in range(len(hzs))))
        print("  " + f"{'model ACC':<16}" + "".join(
            f"{np.nanmean(rollout['acc'], axis=1)[hi]:>9.3f}" for hi in range(len(hzs))))
        print("  " + f"{'model skill':<16}" + "".join(
            f"{np.nanmean(rollout['skill'], axis=1)[hi]:>+9.3f}" for hi in range(len(hzs))))
        print("  (skill > 0 → model beats the naive no-change forecast at that lead)")

    print(f"\nOutputs written to {args.out_dir}/")


if __name__ == "__main__":
    main()
