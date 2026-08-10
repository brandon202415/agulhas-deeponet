#!/usr/bin/env python3
"""AMSE mechanism schematic (Sec. 2.4): plots the exact decorrelation-term
formulas from the manuscript's Eq. 1 (MSE) and Eq. 2 (AMSE),

    MSE decorrelation term  = 2*sqrt(PSD_x * PSD_y) * (1 - Coh)
    AMSE decorrelation term = 2*max(PSD_x, PSD_y) * (1 - Coh)

at a fixed illustrative truth power (PSD_y=1) and fixed misalignment
(1-Coh=0.5), varying only the model's own spectral amplitude PSD_x. This is
a plot of the paper's own stated formulas, not a fitted or empirical
result -- captioned as illustrative/schematic, not a measurement.

    python3 make_fig_amse_schematic.py --out manuscript_figures/fig_amse_schematic.png
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="manuscript_figures/fig_amse_schematic.png")
    args = ap.parse_args()

    psd_y = 1.0          # truth's spectral power at this wavenumber (normalized)
    one_minus_coh = 0.5  # fixed illustrative phase misalignment
    psd_x = np.linspace(1e-6, 2.0, 400)

    mse_term = 2 * np.sqrt(psd_x * psd_y) * one_minus_coh
    amse_term = 2 * np.maximum(psd_x, psd_y) * one_minus_coh

    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.plot(psd_x, mse_term, color="#d62728", lw=2.2, label="MSE: $2\\sqrt{\\mathrm{PSD}_k(x)\\,\\mathrm{PSD}_k(y)}\\,(1-\\mathrm{Coh}_k)$")
    ax.plot(psd_x, amse_term, color="#1f77b4", lw=2.2, label="AMSE: $2\\max(\\mathrm{PSD}_k(x),\\mathrm{PSD}_k(y))\\,(1-\\mathrm{Coh}_k)$")
    ax.axvline(psd_y, color="0.5", lw=1, ls=":")
    ax.annotate("model matches\ntruth's amplitude", xy=(psd_y, 0.05), fontsize=8,
                ha="center", color="0.4")

    ax.annotate("blurring toward zero amplitude\nshrinks MSE's penalty to zero\n(the double-penalty loophole)",
                xy=(0.05, mse_term[2]), xytext=(0.15, 0.55),
                fontsize=8.5, color="#d62728",
                arrowprops=dict(arrowstyle="->", color="#d62728", lw=1))
    ax.annotate("AMSE's penalty stays floored\nby the truth's own power —\nno reward for blurring",
                xy=(0.05, amse_term[2]), xytext=(0.85, 0.45),
                fontsize=8.5, color="#1f77b4",
                arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=1))

    ax.set_xlabel("Model's own spectral amplitude at wavenumber $k$, $\\mathrm{PSD}_k(x)$\n(truth fixed at $\\mathrm{PSD}_k(y)=1$)")
    ax.set_ylabel("Decorrelation-term penalty\n(fixed misalignment, $1-\\mathrm{Coh}_k=0.5$)")
    ax.set_title("Why AMSE removes MSE's blurring incentive\n(illustrative — the paper's Eq. 1/2 formulas, not a fitted result)", fontsize=10)
    ax.legend(fontsize=8, loc="upper left", frameon=False)
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 1.6)

    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
