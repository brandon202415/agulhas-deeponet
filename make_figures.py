#!/usr/bin/env python3
"""Generate publication figures for the Agulhas DeepONet manuscript.

Reads the saved artefacts from a run directory (metrics.json, rollout.npz,
spatial_rmse.npz, predictions.npz) and an optional sweep directory, and writes
PNGs.  Runs entirely offline on a laptop — no cluster, model, or raw data needed.

    pip install matplotlib numpy
    python make_figures.py --run-dir results/best --sweep-dir results/sweep --out-dir figures

Each figure is written only if its source file exists; missing inputs are skipped
with a note rather than failing.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")            # headless
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    raise SystemExit("matplotlib not installed. Run:  pip install matplotlib")

VARIABLES = ["zos", "uo", "vo", "thetao", "so", "mlotst"]
LABELS = {
    "zos": "SSH", "uo": "U vel", "vo": "V vel",
    "thetao": "temp", "so": "salinity", "mlotst": "MLD",
}
UNITS = {"zos": "m", "uo": "m/s", "vo": "m/s", "thetao": "°C", "so": "PSU", "mlotst": "m"}
DPI = 200


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


def fig_skill_singlestep(run, out):
    mj = run / "metrics.json"
    if not mj.exists():
        print(f"  skip single-step skill: {mj} missing"); return
    m = json.load(open(mj))
    vals = [m.get(f"skill_{v}", np.nan) for v in VARIABLES]
    fig, ax = plt.subplots(figsize=(6, 3.6))
    colors = ["#1f77b4" if v >= 0 else "#d62728" for v in vals]
    ax.bar([LABELS[v] for v in VARIABLES], vals, color=colors)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("skill vs persistence")
    ax.set_title("Single-step (1-day) forecast skill")
    for i, v in enumerate(vals):
        ax.annotate(f"{v:+.3f}", (i, v), ha="center",
                    va="bottom" if v >= 0 else "top", fontsize=8)
    _save(fig, out / "fig_skill_singlestep.png")


def fig_rollout(run, out):
    rz = run / "rollout.npz"
    if not rz.exists():
        print(f"  skip rollout figures: {rz} missing"); return
    d = np.load(rz, allow_pickle=True)
    hz = np.asarray(d["horizons"], dtype=float)

    # (a) model skill vs naive over lead time (mean over variables)
    if "skill" in d.files:
        sk = np.nanmean(d["skill"], axis=1)
        fig, ax = plt.subplots(figsize=(6, 3.8))
        ax.plot(hz, sk, "-o", color="#1f77b4", lw=2)
        ax.axhline(0, color="#d62728", ls="--", lw=1.2, label="naive (persistence)")
        ax.fill_between(hz, sk, 0, where=(sk >= 0), color="#1f77b4", alpha=0.15)
        ax.fill_between(hz, sk, 0, where=(sk < 0), color="#d62728", alpha=0.12)
        ax.set_xlabel("lead time (days)"); ax.set_ylabel("model skill vs naive")
        ax.set_title("Skill collapses under rollout: model beats naive only at 1 day")
        ax.legend(frameon=False, fontsize=9)
        _save(fig, out / "fig_skill_vs_lead.png")

    # (b) model vs naive ACC decay (mean over variables)
    if "acc" in d.files and "acc_persist" in d.files:
        fig, ax = plt.subplots(figsize=(6, 3.8))
        ax.plot(hz, np.nanmean(d["acc"], axis=1), "-o", color="#1f77b4", lw=2, label="model")
        ax.plot(hz, np.nanmean(d["acc_persist"], axis=1), "--s", color="#d62728", lw=2, label="naive")
        ax.axhline(0.6, color="gray", ls=":", lw=1, label="useful (0.6)")
        ax.set_xlabel("lead time (days)"); ax.set_ylabel("ACC (mean over variables)")
        ax.set_ylim(0, 1); ax.set_title("Pattern accuracy vs lead time")
        ax.legend(frameon=False, fontsize=9)
        _save(fig, out / "fig_acc_vs_lead.png")

    # (c) per-variable rollout ACC decay
    if "acc" in d.files:
        fig, ax = plt.subplots(figsize=(6, 3.8))
        for vi, v in enumerate(VARIABLES):
            ax.plot(hz, d["acc"][:, vi], "-o", lw=1.6, label=LABELS[v])
        ax.axhline(0.6, color="gray", ls=":", lw=1)
        ax.set_xlabel("lead time (days)"); ax.set_ylabel("ACC")
        ax.set_ylim(0, 1); ax.set_title("Rollout skill (ACC) by variable")
        ax.legend(frameon=False, fontsize=8, ncol=2)
        _save(fig, out / "fig_rollout_acc_byvar.png")


def fig_spatial(run, out):
    sz = run / "spatial_rmse.npz"
    if not sz.exists():
        print(f"  skip spatial map: {sz} missing"); return
    d = np.load(sz, allow_pickle=True)
    grid = np.asarray(d["grid"], dtype=float)
    lon, lat = np.asarray(d["lon"], float), np.asarray(d["lat"], float)
    var = str(d["variable"]) if "variable" in d.files else "zos"
    fig, ax = plt.subplots(figsize=(6, 4.2))
    cmap = plt.cm.viridis.copy(); cmap.set_bad("#dddddd")
    mesh = ax.pcolormesh(lon, lat, np.ma.masked_invalid(grid), cmap=cmap, shading="auto")
    ax.set_xlabel("longitude (°E)"); ax.set_ylabel("latitude (°N)")
    ax.set_title(f"Time-averaged single-step RMSE — {LABELS.get(var, var)}")
    cb = fig.colorbar(mesh, ax=ax); cb.set_label(f"RMSE ({UNITS.get(var, '')})")
    _save(fig, out / "fig_spatial_rmse.png")


def fig_parity(run, out):
    pz = run / "predictions.npz"
    if not pz.exists():
        print(f"  skip parity: {pz} missing"); return
    d = np.load(pz, allow_pickle=True)
    yt, yp = np.asarray(d["y_true"], float), np.asarray(d["y_pred"], float)
    n_sensors = len(np.asarray(d["lon"])) * len(np.asarray(d["lat"]))
    fig, axes = plt.subplots(2, 3, figsize=(9, 6))
    for vi, (v, ax) in enumerate(zip(VARIABLES, axes.ravel())):
        cs, ce = vi * n_sensors, (vi + 1) * n_sensors
        t = yt[:, cs:ce].ravel(); p = yp[:, cs:ce].ravel()
        step = max(1, t.size // 4000)                     # subsample for scatter
        t, p = t[::step], p[::step]
        ax.scatter(t, p, s=2, alpha=0.2, color="#1f77b4", edgecolors="none")
        lo, hi = np.nanpercentile(np.concatenate([t, p]), [0.5, 99.5])
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)
        ax.set_title(f"{LABELS[v]} ({UNITS[v]})", fontsize=9)
        ax.set_xlabel("true", fontsize=8); ax.set_ylabel("predicted", fontsize=8)
        ax.tick_params(labelsize=7)
    fig.suptitle("Predicted vs true (test set)")
    _save(fig, out / "fig_parity.png")


def fig_sweep(sweep, out):
    if sweep is None or not Path(sweep).is_dir():
        print("  skip sweep figure: no --sweep-dir"); return
    rows = []
    for mj in sorted(Path(sweep).glob("*/metrics.json")):
        m = json.load(open(mj))
        skills = [m[k] for k in m if k.startswith("skill_")]
        if not skills:
            continue
        rows.append((float(m.get("learning_rate", np.nan)),
                     str(m.get("loss_weight", "?")),
                     float(m.get("lambda_geo", np.nan)),
                     float(np.mean(skills))))
    if not rows:
        print(f"  skip sweep figure: no metrics.json in {sweep}"); return
    fig, ax = plt.subplots(figsize=(6.2, 4))
    styles = {("none", 0.0): ("#1f77b4", "o", "plain, no physics"),
              ("none", 0.05): ("#7fbfff", "s", "plain, physics"),
              ("variability", 0.0): ("#2ca02c", "^", "var-weighted, no physics"),
              ("variability", 0.05): ("#98df8a", "v", "var-weighted, physics")}
    seen = set()
    for lr, lw, lg, sk in rows:
        key = (lw, round(lg, 3))
        color, marker, lab = styles.get(key, ("#888", "x", f"{lw},{lg}"))
        ax.scatter(lr, sk, c=color, marker=marker, s=60,
                   label=(lab if key not in seen else None))
        seen.add(key)
    ax.set_xscale("log"); ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("learning rate"); ax.set_ylabel("mean skill (test)")
    ax.set_title("Learning rate drives skill; physics & weighting do not")
    ax.legend(frameon=False, fontsize=8)
    _save(fig, out / "fig_sweep_skill.png")


def main():
    ap = argparse.ArgumentParser(description="Manuscript figures from saved results.")
    ap.add_argument("--run-dir", type=Path, default=Path("results/best"),
                    help="Directory with metrics.json, rollout.npz, spatial_rmse.npz, predictions.npz")
    ap.add_argument("--sweep-dir", type=Path, default=Path("results/sweep"),
                    help="Directory of per-config subdirs each with metrics.json (optional)")
    ap.add_argument("--out-dir", type=Path, default=Path("figures"))
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Writing figures to {args.out_dir}/ from run {args.run_dir}")
    fig_skill_singlestep(args.run_dir, args.out_dir)
    fig_rollout(args.run_dir, args.out_dir)
    fig_spatial(args.run_dir, args.out_dir)
    fig_parity(args.run_dir, args.out_dir)
    fig_sweep(args.sweep_dir, args.out_dir)
    print("Done.")


if __name__ == "__main__":
    main()
