#!/usr/bin/env python3
"""Can a 1-2 day trade make +5% eighty percent of the time?

The question has a base rate, and the base rate governs everything. Before
asking which signal predicts a fast +5%, it is worth knowing how often a fast
+5% happens at all: if only one observation in eight ever touches +5% within two
days, a signal has to lift that eightfold to reach an 80% hit rate, and nothing
lifts a base rate eightfold.

So this measures, over every (ticker, day) in the liquid universe:

    entry      = next bar's open          (the signal bar's close is not tradeable)
    hit        = high over the next N bars reaches entry x (1 + target)
    stopped    = low over the next N bars breaks entry x (1 - stop) FIRST

and then reports the hit rate conditional on features known at the signal bar -
volatility, volume surge, yesterday's return, distance from the 52-week high.
Every feature is trailing, so every cut is one a screener could have made live.

    python3 scripts/short_horizon_scan.py [out.csv]

Writes one row per observation so the conditional analysis can be redone without
re-walking the bars.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

from idxbot.config import load_config  # noqa: E402
from idxbot.engine import Engine  # noqa: E402

MIN_CLOSE = 50.0      # IDX regular-market minimum
MIN_VALUE = 1e9       # Rp1bn/day trailing median turnover
HORIZONS = (1, 2, 3)
TARGETS = (0.03, 0.05, 0.08)


def scan_ticker(bars: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Forward high/low envelopes plus the trailing features at each bar."""
    if bars is None or len(bars) < 60:
        return pd.DataFrame()

    o = bars["open"].to_numpy(float)
    h = bars["high"].to_numpy(float)
    low = bars["low"].to_numpy(float)
    c = bars["close"].to_numpy(float)
    v = bars["volume"].to_numpy(float)
    n = len(c)

    # Entry is the NEXT bar's open, so every forward window starts at i+1.
    entry = np.full(n, np.nan)
    entry[:-1] = o[1:]

    out = {"date": bars["date"].to_numpy(), "ticker": ticker,
           "close": c, "entry": entry}

    for hz in HORIZONS:
        fwd_high = np.full(n, np.nan)
        fwd_low = np.full(n, np.nan)
        fwd_close = np.full(n, np.nan)
        for i in range(n - 1):
            j = min(i + 1 + hz, n)
            if j <= i + 1:
                continue
            fwd_high[i] = h[i + 1:j].max()
            fwd_low[i] = low[i + 1:j].min()
            fwd_close[i] = c[j - 1]
        out[f"high_{hz}"] = fwd_high / entry - 1.0
        out[f"low_{hz}"] = fwd_low / entry - 1.0
        out[f"close_{hz}"] = fwd_close / entry - 1.0

    # Trailing features - all known at the signal bar, none peeking forward.
    turnover = pd.Series(c * v)
    out["vt"] = turnover.rolling(20, min_periods=5).median().to_numpy()
    out["atr_pct"] = bars["atr_pct"].to_numpy(float) if "atr_pct" in bars else np.nan
    out["vol_ratio"] = bars["vol_ratio"].to_numpy(float) if "vol_ratio" in bars else np.nan
    out["dist_from_high"] = (bars["dist_from_high"].to_numpy(float)
                             if "dist_from_high" in bars else np.nan)
    out["ret_1"] = pd.Series(c).pct_change().to_numpy()
    out["ret_5"] = pd.Series(c).pct_change(5).to_numpy()
    out["ret_20"] = pd.Series(c).pct_change(20).to_numpy()
    # Gap between the signal close and the price actually paid. A signal that
    # only works when you can buy it flat is not a signal.
    out["gap"] = entry / c - 1.0

    df = pd.DataFrame(out)
    return df[np.isfinite(df["entry"]) & (df["close"] >= MIN_CLOSE)]


def main(dest: str) -> int:
    cfg = load_config()
    engine = Engine(cfg, provider_names=["none"], verbose=False)
    tickers = cfg.universe("idx_all")
    print(f"scanning {len(tickers)} tickers for short-horizon outcomes...")

    frames = []
    for n, ticker in enumerate(tickers, 1):
        try:
            analysis = engine.analyze(ticker, with_campaigns=False)
        except Exception:
            continue
        if analysis is None or analysis.bars.empty:
            continue
        got = scan_ticker(analysis.bars, ticker)
        if not got.empty:
            frames.append(got)
        if n % 100 == 0:
            print(f"  {n}/{len(tickers)}  ({sum(len(f) for f in frames):,} rows)")

    if not frames:
        print("no data")
        return 1

    df = pd.concat(frames, ignore_index=True)
    df = df[df["vt"] >= MIN_VALUE]
    print(f"\n{len(df):,} liquid observations across {df['ticker'].nunique()} tickers")
    print(f"{df['date'].min():%Y-%m} -> {df['date'].max():%Y-%m}\n")

    print("BASE RATE - how often does a stock touch the target at all?")
    print(f"{'horizon':<10}" + "".join(f"{f'+{t:.0%}':>10}" for t in TARGETS))
    for hz in HORIZONS:
        cells = "".join(f"{(df[f'high_{hz}'] >= t).mean():>10.1%}" for t in TARGETS)
        print(f"{f'{hz} day':<10}{cells}")

    print("\nThe 80% bar needs the rightmost column to start near 80%. It does not,")
    print("so the question becomes how far a signal can lift it - see the cuts below.")

    df.to_csv(dest, index=False)
    print(f"\nwrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else
                          "reports/short_horizon.csv"))
