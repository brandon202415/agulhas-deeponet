#!/usr/bin/env python3
"""Rolling (overlapping) 7-day mean of daily states, packaged as a cache file
in the EXACT format load_states() expects (states, lon, lat, subsample_r) --
so the existing, already-validated whole-domain trainer
(train_agulhas_deeponet_prototype.py) can run on it completely UNMODIFIED,
just pointed at this cache and with --step-days 7 instead of the daily
default of 1.

First attempt used NON-overlapping weekly blocks (1001 days -> 143 samples)
and collapsed the dataset too far to test anything -- a 14M-parameter model
had ~100 training examples and never learned past initialization. This
version uses a ROLLING 7-day mean instead: entry i = mean(days[i:i+7]), giving
T-6 = 995 samples, nearly as many as daily. Critically, entries i and i+7 in
this rolling series correspond to two back-to-back, NON-overlapping 7-day
windows (days [i,i+6] and [i+7,i+13]) -- so training the existing trainer with
--step-days 7 on this cache genuinely asks "predict a full week ahead," not a
trivially-easy 1-day-shifted rolling average, while still getting a
sample-rich training set from the overlap ACROSS different starting days i.

Directly tests the timescale-mismatch hypothesis: if daily persistence is a
strong baseline because daily change is genuinely small (mesoscale eddies
evolve over weeks-to-months, not days), then persistence at a WEEKLY lag
should be a much weaker baseline, and real learnable dynamics should show up
more clearly against it -- without also being confounded by a collapsed
training set the way the non-overlapping version was.
"""
import argparse
import numpy as np

from train_agulhas_deeponet_prototype import load_states


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nc", default="data/agulhas_prototype.nc")
    ap.add_argument("--daily-cache", default="data/cache_r6_local.npz")
    ap.add_argument("--subsample-r", type=int, default=6)
    ap.add_argument("--window", type=int, default=7)
    ap.add_argument("--out", default="data/cache_r6_weekly_rolling_local.npz")
    args = ap.parse_args()

    lon, lat, states = load_states(args.nc, subsample_r=args.subsample_r, cache=args.daily_cache)
    T = states.shape[0]
    w = args.window
    n_out = T - w + 1
    # Rolling mean via cumulative sum (vectorised, no Python loop over T).
    cumsum = np.cumsum(states.astype(np.float64), axis=0)
    cumsum = np.concatenate([np.zeros_like(cumsum[:1]), cumsum], axis=0)  # prepend zero row
    rolling = (cumsum[w:] - cumsum[:-w]) / w  # [n_out, nlat, nlon, n_vars]
    assert rolling.shape[0] == n_out

    print(f"Rolling {w}-day mean: {T} daily steps -> {n_out} overlapping weekly samples "
          f"(grid unchanged: {states.shape[1]}x{states.shape[2]})")
    print(f"Use with --step-days {w} so entries i and i+{w} are back-to-back, "
          f"non-overlapping {w}-day windows (a genuine week-ahead forecast).")

    np.savez_compressed(
        args.out,
        states=rolling.astype(np.float32),
        lon=lon, lat=lat,
        subsample_r=np.array(args.subsample_r),
    )
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
