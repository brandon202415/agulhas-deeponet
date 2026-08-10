#!/usr/bin/env python3
"""Fetch only the essential per-observation scalar fields (no 30-point contour/
profile arrays) from AVISO's META4.0 DT global eddy trajectory atlas via
OPeNDAP, for one polarity file, saving to a local .npz for later domain/date
filtering against this project's model predictions.

Run this LOCALLY with your own AVISO+ credentials configured in ~/.netrc --
do not embed your password in this script, a URL, or anything shared back.
Add an entry like:

    machine tds-odatis.aviso.altimetry.fr
    login YOUR_AVISO_EMAIL
    password YOUR_AVISO_PASSWORD

to ~/.netrc, then `chmod 600 ~/.netrc` so only you can read it. netCDF4 (via
libcurl) picks this up automatically for OPeNDAP requests -- no code change
needed to authenticate.

The full file is 29,353,140 observations; contour/profile arrays (30 points
each) are what make these files multi-GB, and we don't need them for
center-position matching -- only the scalar fields below. Fetching just
those should be ~470 MB for this file, not 7+ GB.

Usage:
    python3 fetch_essential_fields.py --url "<the OPeNDAP dataset URL>" --out aviso_meta4_anticyclonic_essential.npz
"""
import argparse

import netCDF4
import numpy as np

FIELDS = [
    "longitude", "latitude", "time", "amplitude", "effective_radius",
    "track", "observation_number",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True, help="OPeNDAP dataset URL (the .nc URL, no .dds/.das suffix)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    print(f"Opening {args.url} ...")
    ds = netCDF4.Dataset(args.url)

    print("\n--- Verifying variable attributes before trusting any epoch/units assumption ---")
    for f in FIELDS:
        v = ds.variables[f]
        attrs = {a: v.getncattr(a) for a in v.ncattrs()}
        print(f"  {f}: shape={v.shape} dtype={v.dtype} attrs={attrs}")

    n_obs = ds.variables["longitude"].shape[0]
    print(f"\nTotal observations in file: {n_obs:,}")

    out = {}
    for f in FIELDS:
        print(f"Fetching {f} ({n_obs:,} values) ...")
        out[f] = ds.variables[f][:].filled(np.nan) if hasattr(ds.variables[f][:], "filled") else np.asarray(ds.variables[f][:])
    ds.close()

    np.savez(args.out, **out)
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
