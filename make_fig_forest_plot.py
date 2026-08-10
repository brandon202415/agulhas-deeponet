#!/usr/bin/env python3
"""Effect-size forest plot: recall-diff and position-error-diff across this
paper's four main eddy-tracking comparisons (Tables 2, 3, 4, 6). Numbers are
taken directly from the manuscript's already-verified tables, not
reprocessed from raw result files -- this figure visualizes reported
numbers, it does not compute new ones.

    python3 make_fig_forest_plot.py --out manuscript_figures/fig_forest_plot.png
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# (label, recall_diff_pct_points, recall_se, pos_diff_km, pos_se, resolution)
# Recall SEs derived from Table 2/4's per-seed std (n=5) as std/sqrt(5); Table 3/6
# report weight-0.01 mean +/- std across 5 seeds directly.
ROWS = [
    ("Architecture (CNN vs. persist.), $r$=3",  0.02, 0.26/np.sqrt(5), -2.605, 0.440/np.sqrt(5), "primary"),
    ("AMSE vs. control, $r$=3 ($w$=0.01)",       0.91, 0.26/np.sqrt(5), -2.505, 0.499/np.sqrt(5), "primary"),
    ("Architecture (CNN vs. persist.), $r$=6",   0.13, 0.14/np.sqrt(5), -2.104, 0.332/np.sqrt(5), "secondary"),
    ("AMSE vs. control, $r$=6 ($w$=0.01)",       0.83, 0.09/np.sqrt(5), -3.322, 0.197/np.sqrt(5), "secondary"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="manuscript_figures/fig_forest_plot.png")
    args = ap.parse_args()

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    y = np.arange(len(ROWS))[::-1]

    colors = ["#1f77b4" if r[5] == "primary" else "#888888" for r in ROWS]

    ax = axes[0]
    for yi, r, c in zip(y, ROWS, colors):
        ax.errorbar([r[1]], [yi], xerr=[1.96 * r[2]], fmt="o", color=c, ecolor=c,
                    elinewidth=2.5, capsize=4, markersize=5)
    ax.axvline(0, color="k", lw=0.8, ls="--")
    ax.set_xlabel("Recall diff (percentage points)")
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in ROWS], fontsize=8.5)
    ax.set_title("Detection recall", fontsize=10)

    ax = axes[1]
    for yi, r, c in zip(y, ROWS, colors):
        ax.errorbar([r[3]], [yi], xerr=[1.96 * r[4]], fmt="o", color=c, ecolor=c,
                    elinewidth=2.5, capsize=4, markersize=5)
    ax.axvline(0, color="k", lw=0.8, ls="--")
    ax.set_xlabel("Position-error diff (km)")
    ax.set_title("Position error", fontsize=10)

    # legend
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], color="#1f77b4", lw=2.5, label="$r$=3 (primary)"),
        Line2D([0], [0], color="#888888", lw=2.5, label="$r$=6 (extended, Sec. 3.6)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.04))

    fig.suptitle("Effect sizes vs. their comparison baseline (95% CI), Tables 2/3/4/6", fontsize=10, y=1.02)
    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
