#!/usr/bin/env python3
"""Resolve the target-vs-stop ordering that daily bars cannot answer.

Daily bars record that both a target and a stop were touched, never which came
first. On the burst setup that ambiguity covers ~13.5% of trades and is enough
to flip expectancy from -0.56% to +0.52%. This walks real 5-minute bars to
settle it.

The sample is small by construction: Yahoo serves only ~60 days of 5-minute
history, so only recent signals can be resolved. Treat the output as a
directional check on the assumption, not as a backtest.

    PYTHONPATH=src python3 scripts/daytrade_path_study.py
"""

from __future__ import annotations

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                os.pardir, "src"))

from idxbot import daytrade as dt                     # noqa: E402
from idxbot.config import load_config                 # noqa: E402
from idxbot.data import YahooOHLCV                    # noqa: E402
from idxbot.data.intraday import (                    # noqa: E402
    YahooIntraday,
    resolve_path,
    session_frame,
)

TARGET = 0.05
STOP = 0.03
COST = 0.004


def main() -> int:
    cfg = load_config()
    daily = YahooOHLCV(cfg)
    tickers = cfg.universe("all")

    print("Loading daily bars...")
    bars = {}
    for ticker in tickers:
        df = daily.get(ticker)
        if df is not None and len(df) > 100:
            bars[ticker] = df
    print(f"  {len(bars)} tickers\n")

    any_bars = next(iter(bars.values()))
    recent_dates = list(any_bars["date"].tail(60))
    print(f"Scanning {len(recent_dates)} recent sessions for setups...")

    candidates = []
    for date in recent_dates:
        found = dt.scan(bars, cfg, as_of=date,
                        min_value_traded=float(cfg.get(
                            "daytrade.min_value_traded_idr", 5e9)))
        # scan() reports the latest bar <= date; keep only signals ON that date.
        candidates.extend([c for c in found
                           if pd.Timestamp(c.date) == pd.Timestamp(date)])

    if not candidates:
        print("  No setups fired inside the intraday-data window.")
        print("  Expected: the burst setup averages ~17 signals a year on this")
        print("  universe, and this window is roughly three months.")
        return 0

    by_setup = pd.Series([c.setup for c in candidates]).value_counts()
    print(f"  {len(candidates)} signals: {dict(by_setup)}\n")

    print("Fetching 5-minute bars for the signalled names...")
    intraday_loader = YahooIntraday(cfg)
    intraday = {}
    for ticker in sorted({c.ticker for c in candidates}):
        df = intraday_loader.get(ticker, interval="5m")
        if df is not None and not df.empty:
            intraday[ticker] = df
            print(f"  {ticker:<6} {len(df):>5} bars")

    rows = []
    for candidate in candidates:
        frame = intraday.get(candidate.ticker)
        if frame is None or frame.empty:
            continue
        later = frame[frame["date"] > pd.Timestamp(candidate.date)]
        if later.empty:
            continue
        session = session_frame(later, later["date"].min())
        if session.empty or len(session) < 5:
            continue

        entry = float(session["open"].iloc[0])
        outcome = resolve_path(session, entry, entry * (1 + TARGET),
                               entry * (1 - STOP))
        rows.append({
            "ticker": candidate.ticker,
            "signal_date": pd.Timestamp(candidate.date).date(),
            "trade_date": pd.Timestamp(session["date"].iloc[0]).date(),
            "setup": candidate.setup,
            "rvol": round(candidate.rvol, 1),
            "entry": entry,
            "outcome": outcome.get("outcome"),
            "return": outcome.get("return"),
        })

    if not rows:
        print("\n  No signal had a following session inside the 5-minute window.")
        return 0

    result = pd.DataFrame(rows)
    print("\n" + "=" * 74)
    print(" PATH RESOLUTION  (entry at next open, target +5%, stop -3%)")
    print("=" * 74)
    print(result.to_string(index=False))

    counts = result["outcome"].value_counts()
    print(f"\n outcomes: {dict(counts)}")

    resolved = result[result["outcome"].isin(["target", "stop", "close"])]
    if len(resolved):
        net = resolved["return"].fillna(0.0) - COST
        print(f"\n resolved trades   : {len(resolved)}")
        print(f" mean net return   : {net.mean():+.2%}")
        print(f" win rate          : {(net > 0).mean():.0%}")
        hit_target = int((result["outcome"] == "target").sum())
        hit_stop = int((result["outcome"] == "stop").sum())
        if hit_target + hit_stop:
            share = hit_target / (hit_target + hit_stop)
            print(f" target first      : {hit_target} of {hit_target + hit_stop} "
                  f"target/stop races ({share:.0%})")
            print()
            print(" The daily-bar analysis assumed the STOP always won that race,")
            print(" giving -0.56%/trade. The optimistic assumption gave +0.52%.")
            print(" This is the measured answer - on this sample.")

    ambiguous = int((result["outcome"] == "ambiguous").sum())
    if ambiguous:
        print(f"\n {ambiguous} trade(s) still ambiguous: both levels fell inside one")
        print(" 5-minute bar. Finer bars would be needed to resolve those.")

    print("\n SAMPLE IS SMALL. This checks the assumption; it does not replace a")
    print(" backtest, which would need years of intraday data nobody serves free.")

    os.makedirs("reports", exist_ok=True)
    result.to_csv("reports/daytrade_path_study.csv", index=False)
    print("\n wrote reports/daytrade_path_study.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
