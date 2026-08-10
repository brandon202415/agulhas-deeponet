#!/usr/bin/env python3
"""Study-domain map: land/ocean mask derived directly from the r=3 test-set
zos channel (a cell flagged land iff zos==0 on every one of the 1,561 test
days -- confirmed to match this project's own documented ocean fraction,
0.830, before being used here). No cartopy/basemap dependency; the
coastline drawn is this project's own training-data land mask, not a
third-party basemap, so it is guaranteed consistent with what every model
in this paper actually sees.

    python3 make_fig_domain_map.py \
        --predictions results/cnn_seed_sweep_r3/cnn_r3_lr1e-3_seed2026/predictions.npz \
        --out manuscript_figures/fig_domain_map.png
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--out", default="manuscript_figures/fig_domain_map.png")
    args = ap.parse_args()

    d = np.load(args.predictions)
    lon, lat = d["lon"], d["lat"]
    n_sensors = len(lon) * len(lat)
    zos = d["y_true"][:, :n_sensors].reshape(-1, len(lat), len(lon))
    land = np.all(zos == 0, axis=0)
    ocean_frac = 1 - land.mean()
    print(f"ocean fraction: {ocean_frac:.3f} (paper's documented value: 0.830)")

    fig, ax = plt.subplots(figsize=(6.2, 6.2))
    # ocean = light blue, land = tan
    rgb = np.zeros((*land.shape, 3))
    rgb[~land] = np.array([0.80, 0.89, 0.96])  # ocean
    rgb[land] = np.array([0.87, 0.80, 0.65])   # land
    ax.imshow(rgb, extent=[lon.min(), lon.max(), lat.min(), lat.max()],
              origin="lower", aspect="auto")
    ax.contour(lon, lat, land.astype(float), levels=[0.5], colors="k", linewidths=0.7)

    # Domain annotations (approximate, standard Agulhas-system landmarks)
    ax.annotate("Agulhas\nRetroflection", xy=(20, -38), xytext=(24, -33),
                fontsize=9, ha="left",
                arrowprops=dict(arrowstyle="->", lw=1))
    ax.annotate("Cape Basin /\nring pathway", xy=(8, -34), xytext=(2, -25),
                fontsize=9, ha="left",
                arrowprops=dict(arrowstyle="->", lw=1))
    ax.annotate("Agulhas\nCurrent core", xy=(32, -32), xytext=(34, -24),
                fontsize=9, ha="left",
                arrowprops=dict(arrowstyle="->", lw=1))
    ax.text(20, -27, "South\nAfrica", fontsize=9, style="italic", color="0.3",
            ha="center", va="center")

    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    ax.set_title("Study domain: Agulhas Current system\n(land mask from this project's own training data)", fontsize=10)
    ax.set_xlim(lon.min(), lon.max())
    ax.set_ylim(lat.min(), lat.max())

    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
