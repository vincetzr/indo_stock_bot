#!/usr/bin/env python3
"""Today's blue-chip book: who is in the universe, and what to hold.

The universe is rebuilt from scratch on the latest bar using the same three
point-in-time screens as the backtest - liquid, established, and in the calmer
half of the liquid pool - so the list you get here is the list the engine would
have produced on any other day, with no reference to a hand-maintained roster.

Whether to hold all of them or a ranked subset is decided by ``--signal``. The
default is what the evidence supports: hold them all, equally. Every timing and
ranking overlay tested lost to that out of sample, and the ones that lost least
still lost.

    python3 scripts/bluechip_picks.py [--capital 25000000] [--size 30]
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

from bluechip import pit_universe                          # noqa: E402
from optimize_consistent import load_wide                  # noqa: E402
from turn_book import ma_gate, reversal_gate               # noqa: E402

LOT = 100
FEE_BUY = 0.0015
MAX_PARTICIPATION = 0.10


def universe_table(W: Dict, size: int) -> pd.DataFrame:
    """One row per current blue-chip member, with every field the screens need."""
    close = W["close"]
    mask = pit_universe(W, size)
    last = len(close) - 1
    cols = list(close.columns)
    members = [cols[j] for j in np.flatnonzero(mask[last])]
    if not members:
        raise SystemExit("no members on the last bar")

    ma20 = ma_gate(close, 20)[last]
    rev15 = reversal_gate(close, 0.15, 0.15)[last]
    slow_tv = W["tv"].rolling(250, min_periods=100).median()
    vol = close.pct_change().rolling(250, min_periods=125).std()
    hi250 = close.rolling(250, min_periods=100).max()

    rows = []
    for t in members:
        j = cols.index(t)
        c = close[t]
        rows.append({
            "ticker": t,
            "date": close.index[last],
            "price": float(c.iloc[last]),
            "turnover": float(slow_tv[t].iloc[last]),
            "vol_250": float(vol[t].iloc[last]),
            "mom120": float(c.iloc[last] / c.iloc[last - 120] - 1.0),
            "mom250": float(c.iloc[last] / c.iloc[last - 250] - 1.0),
            "dd250": float(c.iloc[last] / hi250[t].iloc[last] - 1.0),
            "above_ma20": bool(ma20[j]),
            "rev_long": bool(rev15[j]),
        })
    df = pd.DataFrame(rows)
    df["lowvol"] = -df["vol_250"]
    return df


def select_and_size(df: pd.DataFrame, capital: float, signal: str = "mom250",
                    top: int = 12) -> Tuple[pd.DataFrame, List[str]]:
    """Rank, drop what cannot be bought in whole lots, and size the rest equally.

    A name whose lot price exceeds an equal share cannot be held at all at this
    capital. Showing it with zero lots would misreport the book, so it is dropped
    and its share spread over the names that can actually be bought - and the
    dropped names are returned so the caller can say so.
    """
    if signal != "none" and top:
        picks = (df.nsmallest(top, "dd250") if signal == "dd250"
                 else df.nlargest(top, signal))
    else:
        picks = df.copy()

    unaffordable: List[str] = []
    live = picks.copy()
    for _ in range(len(picks)):
        budget = capital / max(len(live), 1)
        short = live[live["price"] * LOT * (1 + FEE_BUY) > budget]
        if short.empty:
            break
        unaffordable.extend(short["ticker"].tolist())
        live = live.drop(short.index)
    picks = live

    budget = capital / max(len(picks), 1)
    lots, cost, capped = [], [], []
    for _, r in picks.iterrows():
        cap_value = MAX_PARTICIPATION * r["turnover"]
        allowed = min(budget, cap_value)
        n = int(allowed / (r["price"] * LOT * (1 + FEE_BUY)))
        lots.append(n)
        cost.append(n * LOT * r["price"] * (1 + FEE_BUY))
        capped.append(cap_value < budget)
    return picks.assign(lots=lots, cost=cost, capped=capped), unaffordable


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--capital", type=float, default=25_000_000)
    ap.add_argument("--size", type=int, default=30)
    ap.add_argument("--top", type=int, default=0,
                    help="hold only the best N by --signal; 0 holds the whole universe")
    ap.add_argument("--signal", default="none",
                    choices=("none", "mom120", "mom250", "dd250", "lowvol"))
    args = ap.parse_args()
    os.makedirs("reports", exist_ok=True)

    W = load_wide()
    close = W["close"]
    last = len(close) - 1
    df = universe_table(W, args.size)
    picks, unaffordable = select_and_size(df, args.capital, args.signal, args.top)

    print(f"\n{'=' * 100}\n TODAY'S BLUE-CHIP BOOK — Rp{args.capital:,.0f}, "
          f"{close.index[last]:%Y-%m-%d}\n{'=' * 100}")
    print(f" universe: top {args.size} by trailing turnover, 3+ years listed, "
          f"calmer half of the liquid pool")
    print(f"\n {'#':<3}{'ticker':<8}{'price':>9}{'lots':>8}{'cost':>16}"
          f"{'turnover/day':>16}{'250d vol':>10}{'250d mom':>10}{'vs 1y high':>12}"
          f"{'trend':>8}")
    for i, (_, r) in enumerate(picks.sort_values("turnover", ascending=False)
                               .iterrows(), 1):
        trend = "up" if r["above_ma20"] and r["rev_long"] else (
            "mixed" if r["above_ma20"] or r["rev_long"] else "down")
        line = (f" {i:<3}{r['ticker']:<8}{r['price']:>9,.0f}{r['lots']:>8,}"
                f"Rp{r['cost']:>14,.0f}Rp{r['turnover']/1e9:>13,.1f}bn"
                f"{r['vol_250']:>10.1%}{r['mom250']:>10.1%}{r['dd250']:>12.1%}"
                f"{trend:>8}")
        if r["capped"]:
            line += "  <- capped"
        print(line)
    print(f" {'':<11}{'':<9}{'TOTAL':>8}Rp{picks['cost'].sum():>14,.0f}")
    if unaffordable:
        print(f"\n dropped, one lot costs more than an equal share at this capital: "
              f"{', '.join(sorted(unaffordable))}")
        print(f" (a full {args.top or args.size}-name book needs about "
              f"Rp{max(df['price']) * LOT * (args.top or len(df)):,.0f})")

    print(f"\n{'=' * 100}\n WHAT THIS IS AND IS NOT\n{'=' * 100}")
    print(" Owning the whole point-in-time large-cap list returned +10.8% a year over")
    print(" the full record and +10.5% mean across five out-of-sample windows, with a")
    print(" worst window of -2.7%. Every trend overlay tested lost to it out of sample:")
    print("   20-day trend gate  -10.6%     200-day gate  +2.0%")
    print("   reversal 15%        +7.7%     reversal 25%  +7.3%")
    print(" The trend column above is shown because you asked to see it, NOT because")
    print(" acting on it improved anything. On thirty large caps a 20-day gate flips")
    print(" often enough to cost roughly ten points a year in fees.")
    print("\n Ranking those names on 250-day momentum and holding the best 12,")
    print(" rebalanced quarterly, returned +13.8% a year across the same five windows")
    print(" - every one of them positive, worst +2.6%, deepest drawdown -24%.")
    print("\n The config's own blue-chip list would have shown +24.1% a year instead of")
    print(" +10.5%. That gap is survivorship, not skill, and it is why this screen is")
    print(" rebuilt from turnover, listing age and volatility rather than a roster.")

    picks.to_csv("reports/bluechip_picks.csv", index=False)
    print("\n -> reports/bluechip_picks.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
