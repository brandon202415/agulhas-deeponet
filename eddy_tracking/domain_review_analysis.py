#!/usr/bin/env python3
"""Domain-science review follow-up (MANUSCRIPT_ISSUES.md Issues 14-17,
2026-07-30): retroflection-vs-downstream, cyclonic-vs-anticyclonic, and
recall-vs-amplitude/radius breakdowns, plus matched-eddy amplitude error --
none of which the original pooled recall/position-error summary can answer.

Consumes the enriched `{label}_detail` per-eddy records added to
eddy_tracking_analysis.py's output (lon, lat, amplitude, radius_e, polarity,
matched, distance_km, pred_amplitude for every true eddy, every day) rather
than the pooled dists/n_matched fields eddy_stat_test.py uses -- this script
is additive, not a replacement, and does not touch the existing confirmed
comparisons.

Usage (single file):
    python3 eddy_tracking/domain_review_analysis.py --files eddy_tracking/detailed_amse_w0.01_seed2026.json
Usage (aggregate across seeds, e.g. an architecture/weight's full 5-seed set):
    python3 eddy_tracking/domain_review_analysis.py \\
        --files eddy_tracking/detailed_amse_w0.01_seed{2026,7,42,123,2027}.json \\
        --label "AMSE w=0.01"
"""
import argparse
import glob
import json

import numpy as np

RETROFLECTION_LON = 20.0  # deg E; matches the manuscript's own stated
                           # retroflection longitude (Sec. 2.1/1)


def load_all_details(files, label):
    """Flatten every day's per-eddy detail records for one label ('model' or
    'persist') across one or more result files into a single long-format list."""
    rows = []
    for f in files:
        d = json.load(open(f))
        for rec in d["per_day"]:
            for row in rec[f"{label}_detail"]:
                rows.append(row)
    return rows


def region(lon):
    return "retroflection+upstream (>=20E)" if lon >= RETROFLECTION_LON else "downstream/Cape Basin (<20E)"


def recall_by(rows, keyfn, bins=None, bin_labels=None):
    """rows: list of detail dicts. keyfn(row) -> bin label (categorical) or
    a numeric value (if bins given, digitized). Returns dict bin -> (n_true, n_matched, recall)."""
    buckets = {}
    for r in rows:
        if bins is not None:
            v = keyfn(r)
            idx = np.digitize([v], bins)[0]
            key = bin_labels[idx] if idx < len(bin_labels) else bin_labels[-1]
        else:
            key = keyfn(r)
        n_true, n_matched = buckets.get(key, (0, 0))
        buckets[key] = (n_true + 1, n_matched + (1 if r["matched"] else 0))
    return {k: (nt, nm, nm / nt if nt else float("nan")) for k, (nt, nm) in buckets.items()}


def position_error_by(rows, keyfn):
    """Mean position error (matched eddies only) grouped by a categorical key."""
    sums = {}
    for r in rows:
        if not r["matched"]:
            continue
        key = keyfn(r)
        sums.setdefault(key, []).append(r["distance_km"])
    return {k: (len(v), float(np.mean(v))) for k, v in sums.items()}


def amplitude_error(rows):
    """Mean/median absolute amplitude error (m) for matched eddies only."""
    errs = [abs(r["pred_amplitude"] - r["amplitude"]) for r in rows if r["matched"]]
    if not errs:
        return None
    return {
        "n_matched": len(errs),
        "mean_abs_amplitude_error_m": float(np.mean(errs)),
        "median_abs_amplitude_error_m": float(np.median(errs)),
        "mean_true_amplitude_m": float(np.mean([r["amplitude"] for r in rows if r["matched"]])),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="+", required=True)
    ap.add_argument("--label", default=None, help="display label, e.g. 'AMSE w=0.01'")
    args = ap.parse_args()

    files = []
    for pat in args.files:
        files.extend(sorted(glob.glob(pat)) if any(c in pat for c in "*?[") else [pat])
    label = args.label or ", ".join(files)
    print(f"=== {label} ({len(files)} file(s)) ===\n")

    for series in ("model", "persist"):
        rows = load_all_details(files, series)
        n_true = len(rows)
        n_matched = sum(1 for r in rows if r["matched"])
        print(f"--- {series}: n_true={n_true}, recall={n_matched/n_true:.4f} ---")

        # Issue 14: retroflection vs. downstream
        by_region = recall_by(rows, lambda r: region(r["lon"]))
        pe_region = position_error_by(rows, lambda r: region(r["lon"]))
        print("  By region (retroflection+upstream >=20E vs. downstream/Cape Basin <20E):")
        for k in sorted(by_region):
            nt, nm, rec = by_region[k]
            pe = pe_region.get(k, (0, float("nan")))
            print(f"    {k:32s} n_true={nt:5d}  recall={rec:.4f}  "
                  f"pos_err(n={pe[0]:5d})={pe[1]:7.3f} km")

        # Issue 15: polarity
        by_pol = recall_by(rows, lambda r: "anticyclonic" if r["polarity"] == "A" else "cyclonic")
        pe_pol = position_error_by(rows, lambda r: "anticyclonic" if r["polarity"] == "A" else "cyclonic")
        print("  By polarity:")
        for k in sorted(by_pol):
            nt, nm, rec = by_pol[k]
            pe = pe_pol.get(k, (0, float("nan")))
            print(f"    {k:32s} n_true={nt:5d}  recall={rec:.4f}  "
                  f"pos_err(n={pe[0]:5d})={pe[1]:7.3f} km")

        # Issue 16: recall vs. amplitude (quartile bins) and radius (quartile bins)
        amps = np.array([r["amplitude"] for r in rows])
        amp_q = np.quantile(amps, [0.25, 0.5, 0.75])
        amp_labels = ["Q1 (smallest amp)", "Q2", "Q3", "Q4 (largest amp)"]
        by_amp = recall_by(rows, lambda r: r["amplitude"], bins=amp_q, bin_labels=amp_labels)
        print(f"  By true-eddy amplitude quartile (m): {[f'{q:.4f}' for q in amp_q]}")
        for k in amp_labels:
            if k in by_amp:
                nt, nm, rec = by_amp[k]
                print(f"    {k:20s} n_true={nt:5d}  recall={rec:.4f}")

        radii = np.array([r["radius_e"] for r in rows])
        rad_q = np.quantile(radii, [0.25, 0.5, 0.75])
        rad_labels = ["Q1 (smallest radius)", "Q2", "Q3", "Q4 (largest radius)"]
        by_rad = recall_by(rows, lambda r: r["radius_e"], bins=rad_q, bin_labels=rad_labels)
        print(f"  By true-eddy radius_e quartile (m): {[f'{q:.1f}' for q in rad_q]}")
        for k in rad_labels:
            if k in by_rad:
                nt, nm, rec = by_rad[k]
                print(f"    {k:22s} n_true={nt:5d}  recall={rec:.4f}")

        # Issue 17 (amplitude-error half): matched-eddy amplitude error
        ae = amplitude_error(rows)
        if ae:
            print(f"  Matched-eddy amplitude error: mean_abs={ae['mean_abs_amplitude_error_m']:.5f} m, "
                  f"median_abs={ae['median_abs_amplitude_error_m']:.5f} m "
                  f"(mean true amplitude={ae['mean_true_amplitude_m']:.5f} m, n={ae['n_matched']})")
        print()


if __name__ == "__main__":
    main()
