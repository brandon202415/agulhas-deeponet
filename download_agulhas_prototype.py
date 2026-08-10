#!/usr/bin/env python3
"""Download a GLORYS12V1 surface time series for the Agulhas Current domain.

Dataset : GLOBAL_MULTIYEAR_PHY_001_030 (GLORYS12V1, 1/12°, ~8 km)
Domain  : 20°S – 50°S, 0°E – 50°E  (Agulhas Current system)
Depth   : surface level only (0.494 m)
Variables: zos, uo, vo, thetao, so, mlotst

Two GLORYS streams
------------------
GLORYS12V1 (GLOBAL_MULTIYEAR_PHY_001_030) is split into two daily datasets that no
single Copernicus request can span:
    my    = cmems_mod_glo_phy_my_0.083deg_P1D-m      (~1993-01 → ~2021-06)  reanalysis
    myint = cmems_mod_glo_phy_myint_0.083deg_P1D-m   (~2021-07 → near present) interim
`--dataset` selects which; each stream has sensible default dates. To cover the
FULL record, run this twice into the same folder, then point the trainer at the
glob so it stitches them:

    python download_agulhas_prototype.py --dataset my                 # 1993→2021-06
    python download_agulhas_prototype.py --dataset myint              # 2021-07→2024
    python train_agulhas_deeponet_prototype.py --nc 'data/agulhas_*.nc'

Size note
---------
Downloaded at FULL 1/12° resolution (subsampling is a training-time knob, not a
download one): about **2–3 GB per year**, so the full ~32-year record is ~60–90 GB.
Put it on HPC scratch, and download on a login/transfer node — compute nodes
usually have no internet. Exact stream cutover dates drift; check the product page
if a request errors or truncates.

Usage
-----
    pip install copernicusmarine
    copernicusmarine login                      # cache credentials once
    python download_agulhas_prototype.py --dataset my      # or myint, or a custom range
"""

import argparse
from pathlib import Path

DATASET_IDS = {
    "my":    "cmems_mod_glo_phy_my_0.083deg_P1D-m",
    "myint": "cmems_mod_glo_phy_myint_0.083deg_P1D-m",
}
# Default full span of each stream (edit if Copernicus shifts the cutover).
STREAM_DEFAULTS = {
    "my":    ("1993-01-01", "2021-06-30"),
    "myint": ("2021-07-01", "2024-12-31"),
}


def parse_args():
    p = argparse.ArgumentParser(
        description="Download a GLORYS12V1 Agulhas surface time series."
    )
    p.add_argument(
        "--dataset", choices=list(DATASET_IDS), default="my",
        help="GLORYS stream: 'my' (1993→2021-06) or 'myint' (2021-07→present).",
    )
    p.add_argument(
        "--out", type=Path, default=None,
        help="Output NetCDF path. Default: data/agulhas_<dataset>_<y0>_<y1>.nc",
    )
    p.add_argument(
        "--start-date", type=str, default=None,
        help="First day YYYY-MM-DD (default: start of the chosen stream).",
    )
    p.add_argument(
        "--end-date", type=str, default=None,
        help="Last day YYYY-MM-DD (default: end of the chosen stream).",
    )
    args = p.parse_args()
    d0, d1 = STREAM_DEFAULTS[args.dataset]
    if args.start_date is None:
        args.start_date = d0
    if args.end_date is None:
        args.end_date = d1
    if args.out is None:
        y0, y1 = args.start_date[:4], args.end_date[:4]
        args.out = Path(f"data/agulhas_{args.dataset}_{y0}_{y1}.nc")
    return args


def main():
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    try:
        import copernicusmarine
    except ModuleNotFoundError:
        raise SystemExit(
            "copernicusmarine is not installed.\n"
            "Install it with:  pip install copernicusmarine"
        )

    n_years = max(1, int(args.end_date[:4]) - int(args.start_date[:4]) + 1)
    dataset_id = DATASET_IDS[args.dataset]
    print("Downloading GLORYS12V1 Agulhas surface time series …")
    print(f"  Stream  : {args.dataset}  ({dataset_id})")
    print(f"  Domain  : 20°S–50°S, 0°E–50°E")
    print(f"  Period  : {args.start_date} → {args.end_date}  (~{n_years} years)")
    print(f"  Depth   : surface (0.494 m)")
    print(f"  Output  : {args.out}")
    print(f"  NOTE    : full-resolution download ≈ {2*n_years}–{3*n_years} GB; "
          f"run this on a login/transfer node, not a compute node.")

    copernicusmarine.subset(
        # ── Dataset ────────────────────────────────────────────────────────────
        dataset_id=dataset_id,                              # GLORYS12V1 daily
        # ── Variables ──────────────────────────────────────────────────────────
        variables=["zos", "uo", "vo", "thetao", "so", "mlotst"],
        # ── Spatial domain: Agulhas Current system ─────────────────────────────
        minimum_longitude=0.0,
        maximum_longitude=50.0,
        minimum_latitude=-50.0,
        maximum_latitude=-20.0,
        # ── Time window ────────────────────────────────────────────────────────
        start_datetime=f"{args.start_date}T00:00:00",
        end_datetime=f"{args.end_date}T00:00:00",
        # ── Surface level only ─────────────────────────────────────────────────
        minimum_depth=0.49402499198913574,
        maximum_depth=0.49402499198913574,
        # ── Output ─────────────────────────────────────────────────────────────
        output_filename=str(args.out.name),
        output_directory=str(args.out.parent),
    )

    print(f"\nDone. File written to: {args.out}")
    print("Next: download the other stream too (--dataset "
          f"{'myint' if args.dataset == 'my' else 'my'}), then train on both:")
    print("  python train_agulhas_deeponet_prototype.py --nc 'data/agulhas_*.nc'")


if __name__ == "__main__":
    main()