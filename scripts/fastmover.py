#!/usr/bin/env python3
"""The fast-multiplier screen: 2 doubles per 10 names a year, and its price.

    python3 scripts/fastmover.py

WHAT THIS IS FOR. The decade basket (`scripts/decade.py`) answers "7 of 10
multi-baggers" at a ten-year horizon and is the wrong side of the trade below
about three years. This is the opposite instrument: it maximises the chance a
name doubles WITHIN A YEAR, which is what an emerging-market position is
usually bought for.

THE SCREEN. Of names trading at least Rp1bn a day: the most volatile 5% by
60-day realised vol, AND the thinnest-traded 20%. Measured on 443 name-years,
234 names, pre-holdout.

    P(touch 2x within one year)   21.2%   against a 12.05% base rate
    lift                          1.76x
    clustered permutation null    z = +5.66, p = 0.00020

**It clears this project's Bonferroni bar (0.00064 after 78 trials) — the only
result here that ever has.** And it is positive in both halves (1.51x early,
2.06x late).

AND IT IS NOT SKILL, IT IS VARIANCE, WHICH IS WHY IT CLEARS SO EASILY.
Every top feature in the sweep is the same axis: `lowvol` correlates -1.00
with vol60 because it IS vol60 negated, `amihud60` +0.34, `squeeze` +0.31.
And H13's PREDICTED-NULL control, `squeeze`, ranked THIRD at 18.53%. When the
negative control places third, the thing being measured is not signal. A
volatile name is mechanically more likely to touch ANY level: this screen
touches 2x 21.2% of the time and ends below HALF 18.7% of the time.

THE PRICE, AND IT IS THE WHOLE STORY. Per name the screen's mean LOG return is
-0.1927, i.e. it compounds at -17.5% a year. Diversifying into ten names
recovers most of that -- an equal-weighted basket rebalanced annually is paid
close to the arithmetic mean -- but a 100% allocation still compounds at about
+5.1% against the index's +14.6% over the same span: 3.1x against 22.8x.

THE RESOLUTION IS POSITION SIZE, NOT SELECTION. The number of names that
double does not depend on how much money is in them. Hold ten screen names and
about two double every year whatever the sleeve weighs; the sleeve weight
decides only what that does to the account. At 20-30% you keep 94-96% of the
index's compounding and still watch two names double a year.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

from idxbot.report import brief as B                            # noqa: E402
from idxbot.spine import multiplier as MU                       # noqa: E402

PANEL = os.path.join("data", "spine", "price_panel.parquet")

#: THREE TIERS, EACH WITH ITS OWN MEASURED ODDS. The tight screen intersects
#: two filters and on a given day can yield only a handful of names; handing
#: back four when ten were asked for, or quoting the tight screen's odds for a
#: loosened one, are both wrong. Each tier carries the numbers actually
#: measured for it.
#: (label, vol percentile, turnover percentile, P(touch 2x), median, mean,
#:  P(end<=0.5), n)
TIERS = [
    ("tight  (vol 95% & thin 20%)", 0.95, 0.20,
     0.2122, -0.176, 0.185, 0.187, 443),
    ("wide   (vol 90% & thin 30%)", 0.90, 0.30,
     0.1951, -0.167, 0.136, 0.198, 1230),
    ("widest (vol 90%, any liquidity)", 0.90, 1.00,
     0.1989, -0.167, 0.121, 0.220, 3303),
]

#: The tight screen, kept as the headline because it is the one with the
#: permutation null and the half-split behind it.
VOL_PCT, TURN_PCT = 0.95, 0.20

#: Measured on 443 name-years pre-holdout. See reports/fastmover.md.
EVIDENCE = {
    "touch2x": 0.212, "base": 0.1205, "lift": 1.76, "z": 5.66, "p": 0.00020,
    "bar": 0.05 / 78, "n": 443, "n_names": 234,
    "median": -0.191, "mean": 0.169, "mean_log": -0.1927,
    "p_half": 0.187, "p_fifth": 0.027,
    "early_lift": 1.51, "late_lift": 2.06, "cost_median": 0.0138,
}

#: Blend curve: weight in the screen sleeve -> (median CAGR, terminal x,
#: P(ending below 1.0x)). 23 years, 2001-2024, 500 draws, index on a
#: total-return basis at the measured 1.77% yield.
BLEND = [
    (0.00, 0.146, 22.8, 0.000),
    (0.10, 0.145, 22.4, 0.000),
    (0.20, 0.140, 20.2, 0.000),
    (0.30, 0.137, 19.3, 0.000),
    (0.50, 0.122, 14.2, 0.000),
    (0.75, 0.095, 8.1, 0.004),
    (1.00, 0.051, 3.1, 0.148),
]


def eligible(P: pd.DataFrame, day: pd.Timestamp) -> pd.DataFrame:
    d = P[P["date"] == day].copy()
    d = d[d["tradeable"].astype(bool)].dropna(subset=["log_turnover", "vol60"])
    d = d[np.exp(d["log_turnover"]) >= MU.MIN_VALUE]
    return d if len(d) >= 40 else d.iloc[0:0]


def screen(P: pd.DataFrame, day: pd.Timestamp,
           vol_pct: float = VOL_PCT,
           turn_pct: float = TURN_PCT) -> pd.DataFrame:
    """Today's cross-section through the same filters the test measured."""
    d = eligible(P, day)
    if d.empty:
        return d
    vq = d["vol60"].quantile(vol_pct)
    out = d[d["vol60"] >= vq]
    if turn_pct < 1.0:
        out = out[out["log_turnover"] <= d["log_turnover"].quantile(turn_pct)]
    return out.sort_values("vol60", ascending=False)


def pick_tier(P: pd.DataFrame, day: pd.Timestamp, size: int):
    """The tightest tier that can actually fill the basket.

    Returns the tier record alongside its names, so the odds printed are the
    odds of the screen that was USED and never of a tighter one that came up
    short. Falls back to the widest tier and says so.
    """
    for t in TIERS:
        S = screen(P, day, t[1], t[2])
        if len(S) >= size:
            return t, S
    t = TIERS[-1]
    return t, screen(P, day, t[1], t[2])


def resolve_day(P: pd.DataFrame) -> pd.Timestamp:
    cnt = P.groupby("date")["ticker"].count()
    return cnt[cnt >= 0.8 * cnt.tail(250).max()].index.max()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default=PANEL)
    ap.add_argument("--size", type=int, default=10)
    a = ap.parse_args()

    P = pd.read_parquet(a.panel, columns=["date", "ticker", "close",
                                          "tradeable", "log_turnover",
                                          "vol60"])
    P["date"] = pd.to_datetime(P["date"])
    day = resolve_day(P)
    tier, S = pick_tier(P, day, a.size)
    e = dict(EVIDENCE)
    e.update({"touch2x": tier[3], "median": tier[4], "mean": tier[5],
              "p_half": tier[6], "n": tier[7],
              "lift": tier[3] / EVIDENCE["base"]})
    W = 80

    print("=" * W)
    print(" THE FAST-MULTIPLIER SCREEN".center(W))
    print("=" * W)
    if S.empty:
        print(" no representative cross-section — refresh the panel")
        return 1
    print(f" as of {day.date()}   tier: {tier[0]}")
    print(f" {len(S)} names pass; showing {min(a.size, len(S))}")
    tight = screen(P, day)
    if tier[1] != VOL_PCT or tier[2] != TURN_PCT:
        print(f" NOTE: the tight screen yields only {len(tight)} names today,"
              f" so a WIDER tier")
        print(f"       is used and the odds below are that tier's, not the"
              f" tight one's.")
    if len(S) < a.size:
        print(f" NOTE: even the widest tier yields {len(S)} — fewer than the"
              f" {a.size} requested.")

    print(f"\n {'#':<3}{'ticker':<8}{'close':>9}{'Rp/day':>11}"
          f"{'ann vol':>10}{'round trip':>12}")
    for i, (_, r) in enumerate(S.head(a.size).iterrows(), 1):
        cb = B.cost_bar(float(r["close"]), day)
        print(f" {i:<3}{r['ticker']:<8}{r['close']:>9,.0f}"
              f"{np.exp(r['log_turnover']) / 1e9:>8,.0f} bn"
              f"{r['vol60'] * np.sqrt(252):>9.0%}{cb['total']:>11.2%}")

    print("\n" + "=" * W)
    print(" THE ODDS, MEASURED")
    print("=" * W)
    print(f"   P(a name touches 2x within a year)   {e['touch2x']:>8.1%}"
          f"   base {e['base']:.1%}, lift {e['lift']:.2f}x")
    print(f"   P(a name ends below HALF)            {e['p_half']:>8.1%}")
    print(f"   P(a name ends below a FIFTH)         {e['p_fifth']:>8.1%}")
    print(f"\n   per 10 names per year: {10 * e['touch2x']:.1f} double, "
          f"{10 * e['p_half']:.1f} halve.")
    print(f"   measured on {e['n']:,} name-years for this tier")
    print(f"\n   The TIGHT tier carries the significance test: z = "
          f"{e['z']:+.2f}, p = {e['p']:.5f}")
    print(f"   against a Bonferroni bar of {e['bar']:.5f} -> it CLEARS, the"
          f" only result")
    print(f"   in this project that ever has. Positive in both halves "
          f"({e['early_lift']:.2f}x / {e['late_lift']:.2f}x).")
    print(f"   The wider tiers are NOT separately null-tested; their odds are"
          f" measured")
    print(f"   but their significance is inherited, which is weaker.")

    print("\n" + "=" * W)
    print(" AND IT IS VARIANCE, NOT SKILL — WHICH IS WHY IT CLEARS SO EASILY")
    print("=" * W)
    print("   Every top feature in the sweep is the same axis: `lowvol`")
    print("   correlates -1.00 with vol60 because it IS vol60 negated,")
    print("   `amihud60` +0.34, `squeeze` +0.31. And H13's PREDICTED-NULL")
    print("   control, `squeeze`, ranked THIRD at 18.5%.")
    print("   When the negative control places third, what is being measured")
    print("   is not signal. A volatile name is mechanically likelier to")
    print(f"   touch ANY level: 2x {e['touch2x']:.1%} of the time, below half"
          f" {e['p_half']:.1%}.")

    print("\n" + "=" * W)
    print(" THE PRICE: IT DOES NOT COMPOUND")
    print("=" * W)
    print(f"   arithmetic mean, net of cost   {e['mean']:>+8.1%}"
          f"   <- what a rebalanced basket is paid")
    print(f"   median                         {e['median']:>+8.1%}"
          f"   <- what most single names do")
    print(f"   MEAN LOG                       {e['mean_log']:>+8.4f}"
          f"   -> {np.exp(e['mean_log']) - 1:+.1%} a year compounded")
    print(f"\n   Round trip on THESE names is {e['cost_median']:.2%} median,"
          f" against 0.90% for")
    print("   the liquid decile — thin names pay a wider spread as well.")

    print("\n" + "=" * W)
    print(" THE RESOLUTION IS POSITION SIZE, NOT SELECTION")
    print("=" * W)
    print(" The NUMBER of names that double does not depend on how much money")
    print(" is in them. Ten screen names double about twice a year whatever")
    print(" the sleeve weighs. 23 years, 2001-2024, index on a total-return")
    print(" basis.\n")
    print(f"   {'sleeve':<9}{'median CAGR':>13}{'terminal':>11}"
          f"{'P(<1.0x)':>10}{'doubles seen per year':>23}")
    for w, cagr, term, ruin in BLEND:
        note = "  <- all index" if w == 0 else (
            "  <- all screen" if w == 1 else "")
        dbl = "—" if w == 0 else f"{10 * e['touch2x']:.1f} of 10"
        print(f"   {w:<9.0%}{cagr:>+13.1%}{term:>10.1f}x{ruin:>10.1%}"
              f"{dbl:>23}{note}")
    print("\n   Each 10% of the account moved into the sleeve costs roughly")
    print("   1 point of CAGR. At 20-30% you keep 94-96% of the index's")
    print("   compounding and still watch two names double a year.")
    print("   At 100% you turn 22.8x into 3.1x over 23 years and take a")
    print("   14.8% chance of ending below where you started.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
