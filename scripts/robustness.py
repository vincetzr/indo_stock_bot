#!/usr/bin/env python3
"""Split-sample robustness check on backtest observations.

A single full-sample number can be produced by one regime. This splits the
observations by time and re-runs the score-bucket comparison on each half, plus
an ex-crisis subsample, so a result that only exists in 2008 is visible as such.

    python3 scripts/backtest_real.py            # produce the observations first
    python3 scripts/robustness.py reports/backtest_observations.csv
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

EDGES = [0, 40, 55, 65, 101]
LABELS = ["0-39", "40-54", "55-64", "65+"]


def welch_t(a: pd.Series, b: pd.Series) -> float:
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    denom = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return float((a.mean() - b.mean()) / denom) if denom else float("nan")


def report(sub: pd.DataFrame, name: str, horizons=(20, 60)) -> None:
    if sub.empty:
        return
    print(f"\n=== {name}  (n={len(sub):,}, "
          f"{sub['date'].min():%Y-%m-%d} -> {sub['date'].max():%Y-%m-%d}) ===")
    grouped = sub.groupby("bucket", observed=True)
    for bucket, g in grouped:
        cells = "  ".join(
            f"{h}d {g[f'fwd_{h}'].mean() * 100:+7.2f}%" for h in horizons
            if f"fwd_{h}" in g
        )
        print(f"  {str(bucket):<7} n={len(g):>6}  {cells}")

    low = sub.loc[sub["score"] < 40, "fwd_20"].dropna()
    high = sub.loc[sub["score"] >= 65, "fwd_20"].dropna()
    if len(low) > 30 and len(high) > 30:
        print(f"  high-minus-low 20d: {(high.mean() - low.mean()) * 100:+.2f}%"
              f"  t={welch_t(high, low):.2f}")


def main(path: str) -> int:
    df = pd.read_csv(path, parse_dates=["date"])
    df["bucket"] = pd.cut(df["score"], bins=EDGES, labels=LABELS, right=False)

    midpoint = df["date"].quantile(0.5)
    report(df, "FULL SAMPLE")
    report(df[df["date"] < midpoint], "FIRST HALF")
    report(df[df["date"] >= midpoint], "SECOND HALF")
    report(df[~df["date"].dt.year.isin([2008, 2009, 2020])],
           "EX-CRISIS (excludes 2008, 2009, 2020)")

    print("\nIf the sign of 'high-minus-low' flips between subsamples, the "
          "full-sample result is regime-specific and should not be trusted.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1
                          else "reports/backtest_observations.csv"))
