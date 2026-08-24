#!/usr/bin/env python3
"""Build the DAILY price/TA panel H13 tests, over the whole spine.

WHY THIS PANEL IS SO MUCH BIGGER THAN THE FLOW ONE
----------------------------------------------------
The flow branch was capped at 176 names and fortnightly windows because broker
summary costs a network request per (ticker, window). Price features cost
nothing but arithmetic on bars that are already on disk, so this panel is
**~1,000 names at daily resolution over twenty years** — roughly two orders of
magnitude more observations than H9 had. If §8's families carry anything, this
is the sample that finds it.

SURVIVORSHIP
-------------
Both stores are read: `data/cache/ohlcv` (live) and `data/cache/delisted`. §5
is blunt that a universe of currently-listed tickers is survivorship-biased and
every backtest on it is inflated. The delisted store is partial — its snapshot
ends 2019-04-07, so names that died after that are missing their final months —
and that residual bias is stated in the memo rather than papered over.

WHAT IS EXCLUDED, AND WHY EACH EXCLUSION IS NOT OPTIONAL
---------------------------------------------------------
  LOCKED BARS. §5: "A day where a stock is locked at ARA is a day you could not
  buy. Any backtest filling at close on an ARA day is fiction." Entry bars that
  `quality.locked_bars` flags are dropped, using the point-in-time ARA/ARB
  schedule rather than today's.

  STALE BARS. A forward-filled suspension prints an unchanged price, which
  reads as a zero return and as a compressed range. Both would be features.

  SHORT HISTORY. A name needs MIN_HISTORY bars before it enters the
  cross-section, because the longest lookback is 252 bars and a feature
  computed on half its window is a different feature.

THE HOLDOUT IS NOT IN THIS FILE
--------------------------------
§11 reserves the most recent 24 months, untouched, to be spent once. The panel
carries a boolean `holdout` column and every statistic in `price_ic.py` filters
it out. The split is by date, computed here, so it cannot drift between runs.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from typing import List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

from idxbot.features.price import (CONTROLS, FEATURES,           # noqa: E402
                                   MIN_HISTORY, compute,
                                   forward_return)
from idxbot.spine.quality import locked_bars, stale_bars         # noqa: E402
from idxbot.spine.repairs import apply_repairs                   # noqa: E402

LIVE = os.path.join("data", "cache", "ohlcv")
DEAD = os.path.join("data", "cache", "delisted")
OUT = os.path.join("data", "spine", "price_panel.parquet")

HORIZONS = (1, 5, 10, 20)

#: §11's true holdout: the most recent 24 months, reserved and untouched.
HOLDOUT_MONTHS = 24


def load_one(path: str, ticker: str, src: str) -> Optional[pd.DataFrame]:
    try:
        d = pd.read_csv(path, usecols=["date", "open", "high", "low", "close",
                                       "adj_close", "volume"])
    except Exception:                                           # noqa: BLE001
        try:
            d = pd.read_csv(path, usecols=["date", "high", "low", "close",
                                           "volume"])
            d["adj_close"] = d["close"]
            d["open"] = d["close"]
        except Exception:                                       # noqa: BLE001
            return None
    if d.empty or "close" not in d:
        return None
    d["date"] = pd.to_datetime(d["date"])
    d = d[(d["close"] > 0) & d["close"].notna()].sort_values("date")
    if len(d) < MIN_HISTORY:
        return None
    d = apply_repairs(d.reset_index(drop=True), ticker)
    if "adj_close" not in d:
        d["adj_close"] = d["close"]
    d["adj_close"] = pd.to_numeric(d["adj_close"],
                                   errors="coerce").fillna(d["close"])
    d["ticker"] = ticker
    d["src"] = src
    return d


def build(limit: Optional[int] = None) -> pd.DataFrame:
    files = ([(p, "live") for p in sorted(glob.glob(os.path.join(LIVE, "*.JK.csv.gz")))]
             + [(p, "delisted") for p in sorted(glob.glob(os.path.join(DEAD, "*.JK.csv.gz")))])
    if limit:
        files = files[:limit]
    # dict.fromkeys, not a set: mom12_1 is BOTH a control and a tested feature,
    # so the naive concatenation duplicates it and parquet refuses the frame.
    keep = list(dict.fromkeys(
        ["date", "ticker", "src", "close", "adj_close", "volume", "tradeable"]
        + list(CONTROLS) + list(FEATURES)
        + [f"fwd{k}" for k in HORIZONS]))
    out: List[pd.DataFrame] = []
    for i, (p, src) in enumerate(files):
        ticker = os.path.basename(p).replace(".JK.csv.gz", "")
        d = load_one(p, ticker, src)
        if d is None:
            continue
        try:
            d = compute(d)
        except Exception:                                       # noqa: BLE001
            continue
        # A bar you could not have traded is not a bar you may label.
        try:
            bad = locked_bars(d).to_numpy() | stale_bars(d).to_numpy()
        except Exception:                                       # noqa: BLE001
            bad = stale_bars(d).to_numpy()
        d["tradeable"] = ~bad
        for k in HORIZONS:
            d[f"fwd{k}"] = forward_return(d["adj_close"], k)
        out.append(d[[c for c in keep if c in d.columns]])
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(files)} files, {sum(len(x) for x in out):,} rows")
    if not out:
        return pd.DataFrame()
    D = pd.concat(out, ignore_index=True)
    cut = D["date"].max() - pd.DateOffset(months=HOLDOUT_MONTHS)
    D["holdout"] = D["date"] > cut
    return D


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()

    D = build(a.limit)
    if D.empty:
        print("nothing built")
        return 1
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    D.to_parquet(a.out, index=False)
    cut = D.loc[~D["holdout"], "date"].max()
    print(f"\n  rows      {len(D):,}")
    print(f"  tickers   {D.ticker.nunique():,} "
          f"({D[D.src=='delisted'].ticker.nunique()} delisted)")
    print(f"  dates     {D.date.min().date()} … {D.date.max().date()}")
    print(f"  tradeable {D.tradeable.mean():.1%} of bars")
    print(f"  HOLDOUT   reserved after {cut.date()} — "
          f"{D.holdout.mean():.1%} of rows, untouched")
    n = D[~D.holdout].groupby("date")["ticker"].nunique()
    print(f"  names per day (in-sample): median {n.median():.0f}, "
          f"min {n.min()}, max {n.max()}")
    print(f"  written to {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
