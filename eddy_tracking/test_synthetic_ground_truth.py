#!/usr/bin/env python3
"""Synthetic-ground-truth regression test for the eddy-tracking pipeline
(MANUSCRIPT_ISSUES.md Issue 20: a formal review asked what systematic
verification happens *before* trusting a pipeline component, not just what
gets patched after a bug is caught -- this is that test, added after the
fact but meant to run before every future full-scale eddy-tracking result).

Builds a synthetic SSH field with N well-separated, hand-placed Gaussian
"eddies" at known positions, amplitudes, and polarities -- ground truth that
does not depend on py-eddy-tracker, matplotlib, or any part of the pipeline
under test. Runs it through this project's actual identify() (imported
directly from eddy_tracking_analysis.py, not reimplemented) and asserts the
detected count and positions match what was planted.

This is exactly the kind of test that would have caught RESEARCH_LOG.md item
49's matplotlib/py-eddy-tracker version bug immediately: under that bug,
every contour level collapsed to at most one path regardless of how many
eddies were actually present, so this test would have failed outright
(detecting ~1 eddy total instead of N) the first time it was run, rather
than the bug going unnoticed for most of this project's history.

Run with the dedicated `eddytrack` conda env (matplotlib pinned <3.8, see
RESEARCH_LOG.md item 49):
    /opt/anaconda3/envs/eddytrack/bin/python3 eddy_tracking/test_synthetic_ground_truth.py
"""
import sys
import os
import datetime

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from eddy_tracking_analysis import identify, great_circle_km  # noqa: E402

# Domain matching this study's actual grid (Sec. 2.1): 20-50S, 0-50E
LON = np.linspace(0, 50, 201)   # r=3-like density
LAT = np.linspace(-50, -20, 121)
LONG, LATG = np.meshgrid(LON, LAT)

# Known ground truth: (lon, lat, amplitude_m, radius_deg, polarity)
# polarity: +1 = anticyclonic (SSH high), -1 = cyclonic (SSH low)
# Well-separated (>=4 deg apart) so no planted eddy's contour can merge with another's.
PLANTED = [
    (8.0,  -44.0,  0.20, 1.2, +1),
    (16.0, -44.0,  0.18, 1.0, -1),
    (24.0, -44.0,  0.22, 1.3, +1),
    (32.0, -44.0,  0.19, 1.1, -1),
    (40.0, -44.0,  0.21, 1.2, +1),
    (8.0,  -30.0,  0.17, 1.0, -1),
    (16.0, -30.0,  0.23, 1.3, +1),
    (24.0, -30.0,  0.16, 0.9, -1),
    (32.0, -30.0,  0.24, 1.4, +1),
    (40.0, -30.0,  0.18, 1.0, -1),
]


def build_synthetic_field():
    zos = np.zeros_like(LONG)
    for lon0, lat0, amp, radius_deg, polarity in PLANTED:
        r2 = (LONG - lon0) ** 2 + (LATG - lat0) ** 2
        zos += polarity * amp * np.exp(-r2 / (2 * radius_deg ** 2))
    # Geostrophic-like circulation around each bump: u=-k*dh/dy, v=+k*dh/dx
    # (produces clockwise flow around highs, matching Southern Hemisphere
    # anticyclonic convention used elsewhere in this pipeline; k is an
    # arbitrary but consistent scale, not a physically calibrated value --
    # this test only needs a non-degenerate, correctly-signed flow field for
    # py-eddy-tracker's speed-contour step, not a realistic geostrophic Rossby number).
    dlat = LAT[1] - LAT[0]
    dlon = LON[1] - LON[0]
    dzdy, dzdx = np.gradient(zos, dlat, dlon)
    k = 50.0
    uo = -k * dzdy
    vo = k * dzdx
    # A small land patch in a corner, far from every planted eddy: without at least
    # one masked cell, numpy/py_eddy_tracker's internal mask can collapse to a scalar
    # `False` rather than a real boolean array, which breaks py_eddy_tracker's numba
    # interpolation code -- an artifact of an all-ocean domain, not present in this
    # study's real data (which always has genuine land), so not a pipeline bug.
    ocean_mask = np.ones_like(zos, dtype=bool)
    ocean_mask[:5, :5] = False
    return zos, uo, vo, ocean_mask


def main():
    zos, uo, vo, ocean_mask = build_synthetic_field()
    anti, cyclo = identify(zos, uo, vo, LON, LAT, ocean_mask, day_idx=0)

    n_planted_anti = sum(1 for p in PLANTED if p[4] == +1)
    n_planted_cyclo = sum(1 for p in PLANTED if p[4] == -1)

    print(f"Planted: {n_planted_anti} anticyclonic, {n_planted_cyclo} cyclonic ({len(PLANTED)} total)")
    print(f"Detected: {len(anti)} anticyclonic, {len(cyclo)} cyclonic ({len(anti) + len(cyclo)} total)")

    # Match each planted eddy to the nearest detection of the correct polarity
    def check_matches(planted_subset, detected, label):
        n_matched = 0
        for lon0, lat0, amp, radius_deg, polarity in planted_subset:
            if len(detected) == 0:
                continue
            dists = [great_circle_km(lon0, lat0, dlon, dlat)
                     for dlon, dlat in zip(detected["lon"], detected["lat"])]
            best = min(dists)
            if best < 300:  # generous tolerance in km given planted radius ~1-1.4 deg (~100-150km)
                n_matched += 1
        print(f"  {label}: {n_matched}/{len(planted_subset)} planted eddies matched within 300 km")
        return n_matched

    planted_anti = [p for p in PLANTED if p[4] == +1]
    planted_cyclo = [p for p in PLANTED if p[4] == -1]
    n_matched_anti = check_matches(planted_anti, anti, "anticyclonic")
    n_matched_cyclo = check_matches(planted_cyclo, cyclo, "cyclonic")

    ok = True
    if len(anti) + len(cyclo) < len(PLANTED) * 0.7:
        print(f"FAIL: detected total ({len(anti)+len(cyclo)}) is far below planted total "
              f"({len(PLANTED)}) -- exactly the failure signature of RESEARCH_LOG.md item 49's "
              f"matplotlib bug (contour collapse to ~1 path/level).")
        ok = False
    if n_matched_anti < len(planted_anti) * 0.7:
        print(f"FAIL: too few anticyclonic matches ({n_matched_anti}/{len(planted_anti)})")
        ok = False
    if n_matched_cyclo < len(planted_cyclo) * 0.7:
        print(f"FAIL: too few cyclonic matches ({n_matched_cyclo}/{len(planted_cyclo)})")
        ok = False

    if ok:
        print("PASS: synthetic ground-truth eddies recovered as expected.")
        sys.exit(0)
    else:
        print("FAIL: see above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
