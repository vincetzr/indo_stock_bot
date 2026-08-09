#!/usr/bin/env python3
"""The hard ceiling on any intraday strategy, and where a real one lands.

A same-session trade cannot return more than the session's own range. That makes
the ceiling computable directly from daily bars, without any strategy at all:

    buying at the open      -> best possible gain is high/open - 1
    buying at the day's LOW -> best possible gain is high/low - 1

The second is perfect foresight. Nobody can buy the exact low of the day; it is
only knowable after the close. So the fraction of sessions where ``high/low - 1``
reaches +5% is an absolute upper bound on the hit rate of *any* intraday system
targeting +5%, no matter how good. Skill can approach it and can never exceed it.

That bound is worth computing before building anything, because if it sits at
30%, then an 80% intraday hit rate is not difficult - it is arithmetically
unavailable, and no amount of signal engineering changes that.

The realistic bound is the open-based one, and it is the relevant number for a
strategy that reads the tape and buys during the session.

    python3 scripts/intraday_ceiling.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

from idxbot.config import load_config  # noqa: E402
from idxbot.engine import Engine  # noqa: E402

MIN_CLOSE = 50.0
MIN_VALUE = 1e9
TARGETS = (0.02, 0.03, 0.05, 0.08)


def collect(engine: Engine, tickers) -> pd.DataFrame:
    frames = []
    for n, ticker in enumerate(tickers, 1):
        try:
            analysis = engine.analyze(ticker, with_campaigns=False)
        except Exception:
            continue
        if analysis is None or analysis.bars.empty:
            continue
        b = analysis.bars
        if not {"open", "high", "low", "close", "volume"} <= set(b.columns):
            continue
        vt = (b["close"] * b["volume"]).rolling(20, min_periods=5).median()
        frames.append(pd.DataFrame({
            "date": b["date"], "ticker": ticker, "close": b["close"], "vt": vt,
            # Reachable from the open - the realistic intraday entry.
            "from_open": b["high"] / b["open"] - 1.0,
            # Reachable from the day's low - perfect foresight, the hard ceiling.
            "from_low": b["high"] / b["low"] - 1.0,
            # Downside from the open, for the stop side of the question.
            "down_open": b["low"] / b["open"] - 1.0,
            "open_to_close": b["close"] / b["open"] - 1.0,
            "atr_pct": b["atr_pct"] if "atr_pct" in b else np.nan,
            "vol_ratio": b["vol_ratio"] if "vol_ratio" in b else np.nan,
        }))
        if n % 150 == 0:
            print(f"  {n}/{len(tickers)}")
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    return df[(df["close"] >= MIN_CLOSE) & (df["vt"] >= MIN_VALUE)]


def main() -> int:
    cfg = load_config()
    engine = Engine(cfg, provider_names=["none"], verbose=False)
    tickers = cfg.universe("idx_all")
    print(f"collecting intraday ranges for {len(tickers)} tickers...")
    df = collect(engine, tickers)
    if df.empty:
        print("no data")
        return 1

    print(f"\n{len(df):,} liquid sessions, {df['ticker'].nunique()} tickers, "
          f"{df['date'].min():%Y-%m} -> {df['date'].max():%Y-%m}")

    print("\n" + "=" * 74)
    print(" CEILING ON ANY INTRADAY STRATEGY")
    print("=" * 74)
    print(f" {'target':>8}{'from the open':>16}{'from the LOW':>16}   interpretation")
    for t in TARGETS:
        a = (df["from_open"] >= t).mean()
        b = (df["from_low"] >= t).mean()
        note = "perfect foresight" if t == TARGETS[0] else ""
        print(f" {t:>7.0%}{a:>16.1%}{b:>16.1%}   {note}")

    print("\n The right column is what a trader who buys the exact low of every")
    print(" session would achieve - unattainable by construction, since the low is")
    print(" only identifiable after the close. It bounds every intraday system.")

    print("\n" + "=" * 74)
    print(" CONDITIONAL - does filtering lift the realistic (open) column?")
    print("=" * 74)
    cuts = {
        "all sessions": df,
        "ATR% top decile": df[df["atr_pct"] > df["atr_pct"].quantile(0.90)],
        "ATR% top 1%": df[df["atr_pct"] > df["atr_pct"].quantile(0.99)],
        "volume ratio > 3": df[df["vol_ratio"] > 3],
        "ATR top 1% & vol > 3": df[(df["atr_pct"] > df["atr_pct"].quantile(0.99))
                                   & (df["vol_ratio"] > 3)],
    }
    print(f" {'cut':<24}{'n':>10}{'+5% from open':>16}{'+5% from low':>15}")
    for name, sub in cuts.items():
        if len(sub) < 100:
            continue
        print(f" {name:<24}{len(sub):>10,}{(sub['from_open'] >= 0.05).mean():>16.1%}"
              f"{(sub['from_low'] >= 0.05).mean():>15.1%}")

    print("\n" + "=" * 74)
    print(" AND THE SIDE NOBODY QUOTES: the same sessions go down too")
    print("=" * 74)
    both = ((df["from_open"] >= 0.05) & (df["down_open"] <= -0.05)).mean()
    up_only = ((df["from_open"] >= 0.05) & (df["down_open"] > -0.05)).mean()
    print(f" sessions reaching +5% from the open        : {(df['from_open'] >= 0.05).mean():.1%}")
    print(f"   ...that ALSO broke -5% the same session  : {both:.1%}")
    print(f"   ...that reached +5% cleanly              : {up_only:.1%}")
    print("\n A session that touches both barriers is a coin flip on daily data -")
    print(" the tape order is unknowable - so the clean figure is the honest one.")
    print(f"\n open-to-close drift: {df['open_to_close'].mean():+.3%} "
          f"vs a 0.40% round trip")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
