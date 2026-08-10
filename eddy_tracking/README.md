# Eddy-tracking evaluation (reviewer item 3)

Runs py-eddy-tracker (Mason et al., 2014) on the saved single-step prediction
arrays (`results/best/predictions.npz` equivalent) to answer a question RMSE/ACC
cannot: does the model detect/locate individual rings any better than
persistence? See manuscript Sec. 2.4/3.5/4.8.

## Why a dedicated conda env

`pyeddytracker` on PyPI pins very old dependencies (`numpy<1.23`, `netCDF4<1.6`,
`numba<0.56`, `polygon3`) that conflict with a modern numpy 2.x / Python 3.13
stack. `polygon3` in particular only ships as a source tarball (no prebuilt
wheels) but does compile fine with Xcode command line tools. The working recipe:

```bash
conda create -n eddytrack -c conda-forge python=3.10 "numpy<1.23" scipy netcdf4 matplotlib pyyaml requests -y
/path/to/envs/eddytrack/bin/pip install pyeddytracker
conda install -n eddytrack -c conda-forge "netcdf4<1.6" -y   # fix HDF5 linkage vs the pip wheel
/path/to/envs/eddytrack/bin/pip install "setuptools<81"      # numba 0.55 needs pkg_resources
```

This env is local-only (this Mac's Anaconda), separate from Bouchet's `agulhas`
conda env — no cluster changes needed, since this analysis runs entirely on the
saved prediction arrays, not on the training data or model.

## Running it

```bash
/opt/anaconda3/envs/eddytrack/bin/python3 eddy_tracking_analysis.py \
    --predictions /Users/brandonzhang/Downloads/best/predictions.npz \
    --stride 10 --out eddy_tracking_results.json
```

`--stride 10` samples 157 of the 1,561 test days (~2 minutes runtime); `--stride 1`
runs all days (~20 minutes) for a less noisy estimate if wanted later.

## Key parameters (see script docstring for full rationale)

- `BESSEL_WAVELENGTH_KM = 400` — high-pass filter cutoff before contour search.
- `STEP = 0.005` (m) — SSH contour spacing, py-eddy-tracker's own default.
- `SHAPE_ERROR = 70` (%) — loosened from the default 55 for the coarse grid.
- `PIXEL_LIMIT = (1, 2000)` — loosened from the altimetry default (4, ...); at
  the study's r=6 (~50-55 km) grid, the default rejected every detection. This is
  itself a data point for the manuscript's resolution discussion (Sec. 4.8/5).
- `MATCH_RADIUS_KM = 250` — greedy nearest-neighbor matching radius between
  same-polarity eddy centers.

## Result (stride=10, n=197 true eddies over 157 days)

| | Recall | Mean pos. error (km) | Median pos. error (km) |
|---|---|---|---|
| Model | 0.756 | 13.96 | 10.61 |
| Persistence | 0.756 | 13.95 | 10.61 |

Model and persistence are statistically indistinguishable — consistent with the
paper's central finding that the 1-day skill gain (+0.043 mean) is real in
RMSE terms but too small to move eddy-center estimates.

## Not done (scope, see manuscript Sec. 5)

- Only 1-day lead was evaluated (available from `predictions.npz` directly). A
  5/10/20-day version needs raw rollout fields, which `rollout_evaluate()` did
  not save; `train_agulhas_deeponet_prototype.py` now supports
  `--dump-rollout-fields N` for exactly this, but it has not been run.
- Greedy nearest-neighbor matching, not a full multi-day track-linking algorithm
  (py-eddy-tracker also ships `EddyStateTracking`/network-tracking tools for that,
  unused here since we only compare single frames at 1-day lead).
