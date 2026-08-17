#!/usr/bin/env python3
"""Today's picks for the 50/50 book, with lot-level sizing.

Applies the two validated screens to the latest bar in the data:

**Momentum sleeve (50%)** - rank every liquid IDX name on 120-day momentum, take
the top 8, equal weight, rebalance every 10 sessions, **and only hold a name
while it trades above its own 20-day average**. That last rule is checked daily:
a holding is sold the session after it closes below the line, and nothing is
bought while it is below. Walk-forward it cut the mean drawdown from -47% to
-31% - improving in five folds of five - and turned the most recent fold from
-6.3% into +34.6%.

**Multibagger sleeve (50%)** - cheap, small, far below its three-year high, and
state-owned. Held three years and laddered a third of the sleeve per year, so a
fresh start buys one tranche now and the next two on the following anniversaries.

Both screens are computed exactly as they were backtested: same liquidity floor,
same Rp50 price floor, same within-date ranking. Sizing is in whole lots after
the 0.15% buy fee, and positions are capped at 10% of the name's 20-day median
turnover, so the list refuses to hand you a size you could not fill.

    python3 scripts/current_picks.py [--capital 50000000]
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

from idxbot import twosleeve as ts             # noqa: E402
from idxbot.config import load_config          # noqa: E402
from idxbot.data.cache import Cache            # noqa: E402
from idxbot.data.ohlcv import YahooOHLCV       # noqa: E402

LOT = 100
FEE_BUY = 0.0015
MOM_MIN_TURNOVER = 1e10      # Rp10bn/day for the momentum sleeve
BAG_MIN_TURNOVER = 1e9       # Rp1bn/day for the multibagger sleeve
MAX_PARTICIPATION = 0.10
STALE_DAYS = 10


def build(verbose: bool = True) -> pd.DataFrame:
    """One row per name, carrying every field both screens need, as of its last bar."""
    cfg = load_config()
    loader = YahooOHLCV(cfg, Cache(cfg.path("data.cache_dir", "data/cache")))
    names = sorted(set(cfg.universe("idx_all")) | set(cfg.universe("bluechip"))
                   | set(cfg.universe("lq45")) | set(cfg.universe("conglomerate")))
    raw = loader.get_many(names, max_age=86400, verbose=False)
    latest = max((d["date"].max() for d in raw.values() if len(d)), default=None)
    if latest is None:
        raise SystemExit("no price data")

    rows = []
    for t, d in raw.items():
        if len(d) < 300:
            continue
        x = d.sort_values("date").reset_index(drop=True)
        # A name that stopped printing is not tradeable, whatever its momentum.
        if (latest - x["date"].iloc[-1]).days > STALE_DAYS:
            continue
        c = x["close"].astype(float)
        turnover = float((c * x["volume"]).tail(20).median())
        rows.append({
            "ticker": t,
            "date": x["date"].iloc[-1],
            "price": float(c.iloc[-1]),
            "turnover": turnover,
            "mom_120": float(c.iloc[-1] / c.iloc[-121] - 1.0) if len(c) > 121 else np.nan,
            "hi_750": float(c.iloc[-1] / c.tail(750).max() - 1.0),
            "ma20": float(c.tail(20).mean()),
            "above_ma20": bool(c.iloc[-1] > c.tail(20).mean()),
            "off_1y": float(c.iloc[-1] / c.tail(250).max() - 1.0),
        })
    df = pd.DataFrame(rows)
    df["bumn"] = df["ticker"].isin(ts.BUMN).astype(float)
    if verbose:
        print(f"screened {len(df)} names, data to {latest:%Y-%m-%d}")
    return df


def size(picks: pd.DataFrame, sleeve_capital: float) -> pd.DataFrame:
    """Whole-lot sizing, equal weight, capped at a tenth of daily turnover."""
    out = picks.copy()
    budget = sleeve_capital / max(len(out), 1)
    lots, values, capped = [], [], []
    for _, r in out.iterrows():
        cap_value = MAX_PARTICIPATION * r["turnover"]
        allowed = min(budget, cap_value)
        n = int(allowed / (r["price"] * LOT * (1 + FEE_BUY)))
        lots.append(n)
        values.append(n * LOT * r["price"] * (1 + FEE_BUY))
        capped.append(cap_value < budget)
    out["lots"] = lots
    out["cost"] = values
    out["capped"] = capped
    return out


def render(title: str, picks: pd.DataFrame, cols: List[str]) -> None:
    print("\n" + "=" * 92)
    print(f" {title}")
    print("=" * 92)
    head = f" {'#':<3}{'ticker':<8}{'price':>9}{'lots':>9}{'cost':>18}{'turnover/day':>16}"
    extra = {"mom_120": f"{'120d mom':>10}", "hi_750": f"{'vs 3y high':>12}",
             "bumn": f"{'state':>7}", "ma20": f"{'20d MA (exit)':>15}"}
    for c in cols:
        head += extra.get(c, "")
    print(head)
    for i, (_, r) in enumerate(picks.iterrows(), 1):
        line = (f" {i:<3}{r['ticker']:<8}{r['price']:>9,.0f}{r['lots']:>9,}"
                f"Rp{r['cost']:>16,.0f}Rp{r['turnover']/1e9:>13,.1f}bn")
        for c in cols:
            if c == "mom_120":
                line += f"{r['mom_120']:>10.1%}"
            elif c == "hi_750":
                line += f"{r['hi_750']:>12.1%}"
            elif c == "bumn":
                line += f"{'yes' if r['bumn'] > 0 else '-':>7}"
            elif c == "ma20":
                line += f"{r['ma20']:>15,.0f}"
        if r["capped"]:
            line += "  <- capped by turnover"
        print(line)
    print(f" {'':<11}{'':<9}{'':<9}{'TOTAL':>9}Rp{picks['cost'].sum():>16,.0f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--capital", type=float, default=50_000_000)
    ap.add_argument("--mom-top", type=int, default=8)
    ap.add_argument("--bagger-top", type=int, default=10)
    args = ap.parse_args()
    os.makedirs("reports", exist_ok=True)

    df = build()
    half = args.capital / 2.0

    # Only buy what is pointing up: the daily direction filter, applied here at
    # the moment of selection exactly as it is applied every day in the book.
    mom = df[(df["turnover"] >= MOM_MIN_TURNOVER) & (df["price"] >= 50)
             & df["mom_120"].notna() & df["above_ma20"]]
    mom_picks = size(mom.nlargest(args.mom_top, "mom_120"), half)

    bag_pool = df[(df["turnover"] >= BAG_MIN_TURNOVER) & (df["price"] >= 50)].copy()
    bag_pool["_score"] = (
        bag_pool["price"].rank(pct=True) * -1.0
        + bag_pool["turnover"].rank(pct=True) * -1.0
        + bag_pool["hi_750"].rank(pct=True) * -1.0
        + bag_pool["bumn"].rank(pct=True) * 1.0) / 4.0
    # One tranche of three: the sleeve is laddered, so a fresh start commits a
    # third now and the rest on the next two anniversaries.
    bag_picks = size(bag_pool.nlargest(args.bagger_top, "_score"), half / 3.0)

    print(f"\n{'=' * 92}\n CURRENT PICKS — 50/50 BOOK, Rp{args.capital:,.0f}\n{'=' * 92}")
    render(f"MOMENTUM SLEEVE (50% = Rp{half:,.0f}) — rebalance every 10 sessions, "
           f"sell any name that closes below its 20-day average",
           mom_picks, ["mom_120", "ma20"])
    render(f"MULTIBAGGER SLEEVE — tranche 1 of 3 (Rp{half/3:,.0f}), hold 3 years",
           bag_picks, ["hi_750", "bumn"])

    deployed = mom_picks["cost"].sum() + bag_picks["cost"].sum()
    print(f"\n deployed Rp{deployed:,.0f} of Rp{args.capital:,.0f} "
          f"({deployed/args.capital:.0%}); the rest is the two multibagger "
          f"tranches held back for the next two years, plus lot rounding.")

    print(f"\n{'=' * 92}\n WHAT THESE NUMBERS ARE\n{'=' * 92}")
    print(" The momentum sleeve's configuration was chosen by all five walk-forward")
    print(" folds. The 20-day exit cut mean drawdown from -47% to -31% and improved")
    print(" it in five folds of five - but it UNDERPERFORMED the unfiltered book in")
    print(" three of those five, because a trend filter gives up ground in a strong")
    print(" uptrend. It buys smoother, not strictly more.")
    print("\n The multibagger sleeve rests on SIX independent three-year windows and")
    print(" is the most survivorship-exposed thing here: 'cheap, small, beaten down'")
    print(" is also the profile of a company about to delist, and the delisted ones")
    print(" are missing from the data entirely.")
    print("\n This is a screen output, not advice. Verify every name is still")
    print(" trading and still liquid before acting on any line of it.")

    mom_picks.assign(sleeve="momentum").to_csv("reports/current_picks_momentum.csv", index=False)
    bag_picks.assign(sleeve="multibagger").to_csv("reports/current_picks_multibagger.csv", index=False)
    print("\n -> reports/current_picks_momentum.csv, reports/current_picks_multibagger.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
