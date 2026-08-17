#!/usr/bin/env python3
"""Split capital between the momentum book and the multibagger sleeve, and price it.

Two engines that share nothing:

**Momentum book** - rank every liquid IDX name on trailing momentum, hold the
best few, rebalance on a schedule, and sell any holding that falls a set
distance from its own high since entry. That last rule is what lets a scheduled
book exit between rebalances instead of riding a position down for twenty
sessions, and it is the portfolio form of "sell the peak, buy back lower".

**Multibagger sleeve** - cheap, small, far below its old high, state-owned. Held
three years, laddered a third at a time so capital is not committed on one date.

They are combined on an annual grid because that is the only cadence both can
express: the momentum book compounds through the year, the multibagger sleeve
matures one tranche a year. Constant-mix rebalancing resets the split every
year, which mechanically trims whichever sleeve ran.

The comparison that matters is not which sleeve wins. It is what the *mix* does
to the years you would actually have to live through, so every allocation is
reported with its worst year and its time underwater beside its CAGR.

    python3 scripts/fifty_fifty.py [--lookback 120 --top 10 --rebalance 20 --trail 0.20]
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

from idxbot import twosleeve as ts                      # noqa: E402
from optimize_consistent import load_wide, simulate     # noqa: E402


def annual_from_equity(eq: pd.Series) -> pd.Series:
    """Calendar-year returns from a daily equity curve."""
    yearly = eq.resample("YE").last()
    first = pd.Series([eq.iloc[0]], index=[eq.index[0]])
    joined = pd.concat([first, yearly])
    out = joined.pct_change().dropna()
    out.index = [d.year for d in out.index]
    return out[~out.index.duplicated(keep="last")]


def summarise(annual: pd.Series, label: str) -> Dict[str, float]:
    curve = np.cumprod(1.0 + annual.to_numpy())
    peak = np.maximum.accumulate(curve)
    dd = curve / peak - 1.0
    n = len(annual)
    return {
        "label": label, "years": float(n),
        "growth": float(curve[-1]),
        "cagr": float(curve[-1] ** (1 / n) - 1) if n else np.nan,
        "median_year": float(annual.median()),
        "worst_year": float(annual.min()),
        "best_year": float(annual.max()),
        "pct_positive": float((annual > 0).mean()),
        "max_dd": float(dd.min()),
        "ulcer": float(np.sqrt((dd ** 2).mean())),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lookback", type=int, default=120)
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--rebalance", type=int, default=20)
    ap.add_argument("--trail", type=float, default=0.20)
    ap.add_argument("--bagger-top", type=int, default=10)
    args = ap.parse_args()
    os.makedirs("reports", exist_ok=True)

    print("building the momentum book ...")
    W = load_wide(verbose=False)
    eq, trades = simulate(W, args.lookback, args.top, args.rebalance,
                          args.trail if args.trail > 0 else None)
    mom_annual = annual_from_equity(eq)
    print(f"  {args.lookback}d momentum, top {args.top}, rebalance {args.rebalance}, "
          f"trail {args.trail:.0%}: {trades:,} trades")

    print("building the multibagger sleeve ...")
    panel = pd.read_csv("reports/multibagger_panel.csv", parse_dates=["date"])
    if "bumn" not in panel.columns:
        panel = ts.attach_state_ownership(panel)
    sleeve = ts.run_bagger_sleeve(panel, top_n=args.bagger_top)
    bag_annual = ts.ladder_bagger(sleeve.periods)
    print(f"  {int(sleeve.stats['rebalances'])} three-year windows, laddered to "
          f"{len(bag_annual)} years")

    years = sorted(set(mom_annual.index) & set(bag_annual.index))
    m = mom_annual.reindex(years)
    b = bag_annual.reindex(years)
    print(f"\noverlapping years: {len(years)} ({years[0]}-{years[-1]})")

    rows = [summarise(m, "100% momentum book"),
            summarise(b, "100% multibagger")]
    for w in (0.7, 0.6, 0.5, 0.4, 0.3):
        mix = w * m + (1 - w) * b
        rows.append(summarise(mix, f"{w:.0%} momentum / {1-w:.0%} multibagger"))
    # drift: never rebalanced, the winner takes over
    mc, bc = np.cumprod(1 + m.to_numpy()), np.cumprod(1 + b.to_numpy())
    drift = pd.Series(np.diff(np.concatenate([[1.0], 0.5 * mc + 0.5 * bc]))
                      / np.concatenate([[1.0], (0.5 * mc + 0.5 * bc)[:-1]]), index=years)
    rows.append(summarise(drift, "50/50, never rebalanced"))

    R = pd.DataFrame(rows)
    print("\n" + "=" * 100)
    print(" ALLOCATION COMPARISON — same years, same engines, only the split moves")
    print("=" * 100)
    print(f" {'allocation':<36}{'CAGR':>8}{'growth':>9}{'medYr':>8}"
          f"{'worstYr':>9}{'+yrs':>6}{'maxDD':>8}{'ulcer':>7}")
    for _, r in R.iterrows():
        print(f" {r['label']:<36}{r['cagr']:>+8.1%}{r['growth']:>8.1f}x"
              f"{r['median_year']:>+8.1%}{r['worst_year']:>+9.1%}"
              f"{r['pct_positive']:>6.0%}{r['max_dd']:>8.0%}{r['ulcer']:>7.2f}")

    print(f"\n year-by-year, 50/50 rebalanced annually:")
    mix = 0.5 * m + 0.5 * b
    print("  " + "  ".join(f"{y}:{v:+.0%}" for y, v in mix.items()))
    print(f"\n correlation between the two sleeves: {m.corr(b):+.3f}")
    print("  (low or negative is what makes the mix worth holding)")
    R.to_csv("reports/fifty_fifty.csv", index=False)
    print("\n -> reports/fifty_fifty.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
