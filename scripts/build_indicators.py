#!/usr/bin/env python3
"""Build `data/spine/indicator_panel.parquet` — causal indicators per ticker-day.

    python3 scripts/build_indicators.py

Joins the OHLCV cache's high/low onto the spine panel's adjusted close and
volume, rebases high/low with the panel's own adjustment factor (so the three
repaired names inherit the repair), and computes EMA / ATR / stochastic / RSI /
turnover-z per ticker in date order.

Written once and cached because the exit study rescores the same bars under
dozens of rules; recomputing per cohort was the difference between a two-minute
job and an hour-long one.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

from idxbot.spine import signals as S                            # noqa: E402

PANEL = os.path.join("data", "spine", "price_panel.parquet")
OUT = os.path.join("data", "spine", "indicator_panel.parquet")
CACHES = ("data/cache/ohlcv/{t}.JK.csv.gz",
          "data/cache/delisted/{t}.JK.csv.gz",
          "data/cache/delisted/{t}.csv.gz")


def _ohlc(t: str):
    for pat in CACHES:
        p = pat.format(t=t)
        if os.path.exists(p):
            d = pd.read_csv(p, parse_dates=["date"])
            if {"high", "low", "close"} <= set(d.columns):
                return d[["date", "high", "low", "close"]]
    return None


def main() -> int:
    P = pd.read_parquet(PANEL, columns=["date", "ticker", "close",
                                        "adj_close", "volume"])
    P["date"] = pd.to_datetime(P["date"])
    names = sorted(P["ticker"].unique())
    print(f"{len(names)} names, {len(P):,} panel rows")

    out, missing, mismatch = [], [], []
    for i, t in enumerate(names):
        g = P[P["ticker"] == t]
        d = _ohlc(t)
        if d is None:
            missing.append(t)
            continue
        m = g.merge(d, on="date", how="left", suffixes=("", "_c"))
        # the join must not silently substitute a different vendor's bar
        rel = (m["close"] - m["close_c"]).abs() / m["close_c"].replace(0, np.nan)
        bad = float((rel > 1e-6).mean())
        if bad > 0.001:
            mismatch.append((t, round(bad, 4)))
        m.loc[rel > 1e-6, ["high", "low"]] = np.nan
        m["ticker"] = t
        out.append(S.build(m))
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(names)}")

    R = pd.concat(out, ignore_index=True)
    R.to_parquet(OUT, index=False)
    print(f"\nwrote {OUT}: {len(R):,} rows, {R['ticker'].nunique()} names")
    if missing:
        print(f"no OHLCV file for {len(missing)}: {missing[:12]}")
    if mismatch:
        print(f"close mismatch >0.1% of bars on {len(mismatch)}: {mismatch[:12]}")
    cov = R[S.COLUMNS].notna().mean().round(4)
    print("\nnon-null coverage per column:")
    print(cov.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
