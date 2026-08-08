#!/usr/bin/env python3
"""Backfill point-in-time turnover onto an observations CSV.

`idxbot backtest` now writes a ``vt`` column itself, so this only exists for
observation files produced before that change — notably ``reports/obs_full.csv``,
the 460k-row full-exchange run, which is expensive enough to regenerate that
patching it is the sane option.

Why it matters: without a liquidity column the evaluation cannot tell a real
opportunity from an untradeable one, and IDX has a great many of the latter. The
full-exchange run initially reported a 5-day baseline of +10.91%, driven by 3,242
rows priced below Rp10 whose mean 5-day forward return was +1485%. IDX's regular
market cannot print below Rp50; those were split-adjustment artifacts.

``vt`` is the trailing 20-bar median of close x volume. Trailing, so filtering on
it is a decision the screener could have made live. Median rather than mean, so a
single block crossing does not make a dead stock look tradeable.

    python3 scripts/backfill_liquidity.py reports/obs_full.csv reports/obs_full_clean.csv
"""

from __future__ import annotations

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

from idxbot.config import load_config  # noqa: E402
from idxbot.engine import Engine  # noqa: E402

MIN_CLOSE = 50.0     # IDX regular-market minimum price
MIN_VALUE = 1e9      # Rp1bn/day - below this a Rp10jt order moves the book
WINDOW = 20


def main(src: str, dest: str) -> int:
    if not os.path.exists(src):
        print(f"no observations at {src}")
        return 2

    df = pd.read_csv(src, parse_dates=["date"])
    print(f"read {len(df):,} observations, {df['ticker'].nunique()} tickers")

    engine = Engine(load_config(), provider_names=["none"], verbose=False)

    frames = []
    tickers = sorted(df["ticker"].unique())
    for n, ticker in enumerate(tickers, 1):
        analysis = engine.analyze(ticker, with_campaigns=False)
        if analysis is None or analysis.bars.empty:
            continue
        bars = analysis.bars
        if "volume" not in bars.columns:
            continue
        vt = (bars["close"] * bars["volume"]).rolling(WINDOW, min_periods=5).median()
        frames.append(pd.DataFrame({"date": bars["date"], "ticker": ticker, "vt": vt}))
        if n % 100 == 0:
            print(f"  {n}/{len(tickers)}")

    if not frames:
        print("no bars available - nothing to backfill")
        return 1

    liquidity = pd.concat(frames, ignore_index=True)
    merged = df.merge(liquidity, on=["date", "ticker"], how="left")

    before = len(merged)
    clean = merged[(merged["close"] >= MIN_CLOSE) & (merged["vt"] >= MIN_VALUE)]
    print(f"after close>=Rp{MIN_CLOSE:.0f} and vt>=Rp{MIN_VALUE/1e9:.0f}bn/day: "
          f"{len(clean):,} ({before - len(clean):,} dropped), "
          f"{clean['ticker'].nunique()} tickers")

    clean.to_csv(dest, index=False)
    print(f"wrote {dest}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        raise SystemExit(2)
    raise SystemExit(main(sys.argv[1], sys.argv[2]))
