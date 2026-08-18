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
from bluechip_picks import select_and_size, universe_table   # noqa: E402
from optimize_consistent import load_wide                    # noqa: E402

def reversal_long(close: np.ndarray, entry: float, exit_: float) -> bool:
    """Is the bounded-lag reversal state currently long? (Part XVII's filter.)

    Sell when the close falls ``exit_`` below its high since entry, buy when it
    rises ``entry`` above its low since exit. Only the final state is needed
    here, but it is computed from the whole series because the state is path
    dependent - there is no shortcut that gives the same answer.
    """
    long, ext = False, close[0]
    for p in close:
        if not np.isfinite(p):
            continue
        if long:
            if p > ext:
                ext = p
            elif 1.0 - p / ext >= exit_ - 1e-12:
                long, ext = False, p
        else:
            if p < ext:
                ext = p
            elif p / ext - 1.0 >= entry - 1e-12:
                long, ext = True, p
    return long


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
        rev = reversal_long(c.to_numpy(float), 0.15, 0.15)
        rows.append({
            "ticker": t,
            "date": x["date"].iloc[-1],
            "price": float(c.iloc[-1]),
            "turnover": turnover,
            "mom_120": float(c.iloc[-1] / c.iloc[-121] - 1.0) if len(c) > 121 else np.nan,
            "mom_250": float(c.iloc[-1] / c.iloc[-251] - 1.0) if len(c) > 251 else np.nan,
            "hi_750": float(c.iloc[-1] / c.tail(750).max() - 1.0),
            "ma20": float(c.tail(20).mean()),
            "above_ma20": bool(c.iloc[-1] > c.tail(20).mean()),
            "rev_long": bool(rev),
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
    extra = {"mom_120": f"{'120d mom':>10}", "mom_250": f"{'250d mom':>10}",
             "hi_750": f"{'vs 3y high':>12}",
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
            elif c == "mom_250":
                line += f"{r['mom_250']:>10.1%}"
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
    ap.add_argument("--conc-top", type=int, default=3)
    ap.add_argument("--bluechip-top", type=int, default=12)
    ap.add_argument("--bluechip-size", type=int, default=30)
    args = ap.parse_args()
    os.makedirs("reports", exist_ok=True)

    df = build()
    half = args.capital / 2.0

    # The blue-chip half, built the way Part XIX validated it: a universe
    # defined as of today by turnover, listing age and volatility, ranked on
    # 250-day momentum, top 12, held quarterly, no gate.
    W = load_wide(verbose=False)
    bc_all = universe_table(W, args.bluechip_size)
    bc_picks, bc_dropped = select_and_size(bc_all, half, "mom250", args.bluechip_top)

    # Only buy what is pointing up: the daily direction filter, applied here at
    # the moment of selection exactly as it is applied every day in the book.
    mom = df[(df["turnover"] >= MOM_MIN_TURNOVER) & (df["price"] >= 50)
             & df["mom_120"].notna() & df["above_ma20"]]
    mom_picks = size(mom.nlargest(args.mom_top, "mom_120"), half)

    # The concentrated variant (Part XVIII): 250-day momentum, top 3, and BOTH
    # gates on. Across the grid, concentration raises out-of-sample return
    # monotonically - and only survives when a gate is on: ungated top 3 has a
    # -39.9% fold and a -61% drawdown, gated top 3 has +24.4% worst and -35%.
    conc_pool = df[(df["turnover"] >= MOM_MIN_TURNOVER) & (df["price"] >= 50)
                   & df["mom_250"].notna() & df["above_ma20"] & df["rev_long"]]
    conc_picks = size(conc_pool.nlargest(args.conc_top, "mom_250"), half)

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
    print("\n" + "=" * 92)
    print(f" BLUE-CHIP SLEEVE (50% = Rp{half:,.0f}) — 250-day momentum, top "
          f"{args.bluechip_top} of the point-in-time large caps, rebalance quarterly, "
          f"no stop")
    print("=" * 92)
    print(f" {'#':<3}{'ticker':<8}{'price':>9}{'lots':>9}{'cost':>18}"
          f"{'turnover/day':>16}{'250d mom':>10}{'250d vol':>10}")
    for i, (_, r) in enumerate(bc_picks.sort_values("mom250", ascending=False)
                               .iterrows(), 1):
        line = (f" {i:<3}{r['ticker']:<8}{r['price']:>9,.0f}{r['lots']:>9,}"
                f"Rp{r['cost']:>16,.0f}Rp{r['turnover']/1e9:>13,.1f}bn"
                f"{r['mom250']:>10.1%}{r['vol_250']:>10.1%}")
        if r["capped"]:
            line += "  <- capped by turnover"
        print(line)
    print(f" {'':<11}{'':<9}{'':<9}{'TOTAL':>9}Rp{bc_picks['cost'].sum():>16,.0f}")
    if bc_dropped:
        print(f" dropped, one lot exceeds an equal share here: "
              f"{', '.join(sorted(bc_dropped))}")

    render(f"AGGRESSIVE ALTERNATIVE TO THE BLUE-CHIP HALF — whole-exchange "
           f"momentum (50% = Rp{half:,.0f}), rebalance every 10 sessions, "
           f"sell any name that closes below its 20-day average",
           mom_picks, ["mom_120", "ma20"])
    render(f"MOMENTUM SLEEVE, CONCENTRATED ALTERNATIVE (50% = Rp{half:,.0f}) — "
           f"250-day momentum, top {args.conc_top}, above the 20-day average AND "
           f"reversal-long",
           conc_picks, ["mom_250", "ma20"])
    render(f"MULTIBAGGER SLEEVE — tranche 1 of 3 (Rp{half/3:,.0f}), hold 3 years",
           bag_picks, ["hi_750", "bumn"])

    deployed = mom_picks["cost"].sum() + bag_picks["cost"].sum()
    print(f"\n deployed Rp{deployed:,.0f} of Rp{args.capital:,.0f} "
          f"({deployed/args.capital:.0%}); the rest is the two multibagger "
          f"tranches held back for the next two years, plus lot rounding.")

    print(f"\n{'=' * 92}\n WHICH HALF TO PUT THE FIRST 50% IN\n{'=' * 92}")
    print(" Across five out-of-sample windows, chained:")
    print("   blue chip, 250d momentum top 12, quarterly   +13.8%/yr, worst window")
    print("     +2.6%, deepest drawdown -24%, all five windows positive")
    print("   whole exchange, 120d momentum top 8, 20d gate  +37.5% mean/yr, worst")
    print("     window +9.2%, deepest drawdown -28%")
    print(" The exchange-wide book earns far more and holds names like JGLE and KOTA")
    print(" to do it. The blue-chip book earns less and holds ASII and BBRI. That is")
    print(" the trade, and no measurement here decides it for you.")

    print(f"\n{'=' * 92}\n CHOOSING BETWEEN THE TWO EXCHANGE-WIDE SLEEVES\n{'=' * 92}")
    print(" Across all 315 configurations searched in Part XVIII, holding fewer names")
    print(" raised out-of-sample return monotonically and raised drawdown with it:")
    print("   top 3  +40.4% median OOS mean, -47% mean drawdown")
    print("   top 8  +30.7%                  -37%")
    print("   top 20 +18.4%                  -30%")
    print(" At the deployed 120d/reb10 setting with the 20-day gate, top 8 returned")
    print(" +37.5% mean OOS with -28% drawdown; top 3 returned +57.2% with -38%.")
    print(" The concentrated book above is the higher-return, rougher-ride choice.")
    print(" It is NOT a free upgrade: three names means one blow-up is a third of")
    print(" the sleeve, and the turnover cap binds sooner as capital grows.")

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

    bc_picks.assign(sleeve="bluechip").to_csv("reports/current_picks_bluechip.csv", index=False)
    mom_picks.assign(sleeve="momentum").to_csv("reports/current_picks_momentum.csv", index=False)
    conc_picks.assign(sleeve="concentrated").to_csv("reports/current_picks_concentrated.csv", index=False)
    bag_picks.assign(sleeve="multibagger").to_csv("reports/current_picks_multibagger.csv", index=False)
    print("\n -> reports/current_picks_momentum.csv, reports/current_picks_multibagger.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
