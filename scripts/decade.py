#!/usr/bin/env python3
"""The decade basket: the one construction in this repo that beat the index.

    python3 scripts/decade.py

WHAT THIS IS. H23 varied the horizon that twelve earlier studies had inherited
without choosing, and found that over TEN-YEAR holds the most liquid IDX names
beat the IHSG on a total-return basis by a paired median of +51.8%, positive in
both halves of the sample. This script applies exactly that rule to today's
cross-section and prints the basket with its measured expectations attached.

THE RULE, and it is deliberately dull:

    1. every name trading at least Rp1bn a day (MU.MIN_VALUE)
    2. keep the top half by traded value
    3. of those, take the top decile
    4. equal weight, hold ten years, do not touch it

Steps 2 and 3 compose to the top ~5% of eligible names, which is what the
historical test measured; applying "top decile of everything" instead would be
a different rule from the one with the evidence behind it.

WHAT THE EVIDENCE DOES AND DOES NOT SUPPORT. It is a candidate, not a finding.
The permutation p resolves to 0.0014 against a Bonferroni bar of 0.00071 after
70 trials, so it does NOT clear this project's own threshold. Effective n is
about 56 for the sample and ~6 for the decile — a ten-year window over a
twenty-four-year panel is roughly two independent observations per name. Only
35 distinct names were ever in the historical decile. And the 24-month holdout
was spent at H16, so none of it is out of sample.

WHY IT IS STILL WORTH PRINTING. Every other construction in this project lost
to the index outright, and this one wins in both halves against a benchmark
put on a like-for-like total-return basis. The mechanism is not mysterious: a
one-year hold pays ~1.3% round trip every year and a ten-year hold pays it
once. That is arithmetic, not a discovered edge.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

from idxbot.report import brief as B                            # noqa: E402
from idxbot.spine import multiplier as MU                       # noqa: E402

PANEL = os.path.join("data", "spine", "price_panel.parquet")

#: Measured on 6,332 matched ten-year windows, pre-holdout. See
#: reports/horizon.md §4. These are the numbers the basket is bought on and
#: they are quoted with the basket so it can never be shown without them.
EVIDENCE = {
    "touch2x": 0.691, "median": 1.747, "mean": 4.325, "p_half": 0.137,
    "index_median": 1.085, "paired_median": 0.518, "win_rate": 0.577,
    "z": 2.87, "p": 0.0014, "bar": 0.05 / 70, "eff_n": 56,
    "base_touch": 0.555, "base_half": 0.286,
}

#: H24. Names already in the decile for all three prior years touched 2x
#: 82.8% of the time — 8 of 10 — against the 69.1% for the decile as a whole,
#: and the rate barely moved between halves (83.6% / 82.4%) while the BASE
#: rate collapsed from 70.6% to 40.3%. It is the strongest cell in the project
#: and the thinnest: 221 windows, EIGHT distinct names, effective n 1.8.
CORE = {"touch2x": 0.828, "median": 1.972, "p_half": 0.072, "n_names": 8,
        "eff_n": 1.8, "z": 3.04, "p": 0.0010, "bar": 0.05 / 72,
        "early": 0.836, "late": 0.824,
        "hist": ["ASII", "BBCA", "BBNI", "BBRI", "BMRI", "BUMI", "PGAS",
                 "TLKM"]}

#: A name whose entire listed history is shorter than the hold cannot be the
#: object the historical cell measured, whatever its tenure score says.
MIN_LISTED_YEARS = 10.0


def universe(P: pd.DataFrame, day: pd.Timestamp) -> pd.DataFrame:
    """The three filter steps, in the order the historical test applied them."""
    d = P[(P["date"] == day)].copy()
    d = d[d["tradeable"].astype(bool)].dropna(subset=["log_turnover"])
    d = d[np.exp(d["log_turnover"]) >= MU.MIN_VALUE]        # 1. Rp1bn/day
    if len(d) < 40:
        return d.iloc[0:0]
    d = d[d["log_turnover"] >= d["log_turnover"].median()]  # 2. top half
    cut = d["log_turnover"].quantile(0.90)                  # 3. top decile
    return d[d["log_turnover"] >= cut].sort_values("log_turnover",
                                                   ascending=False)


def tenure(P: pd.DataFrame, day: pd.Timestamp, names: set) -> Dict[str, int]:
    """How many of the three prior years each name was ALSO in the decile.

    Knowable entirely at the cohort date — it looks backwards only — so it
    passes A5. Each anniversary snaps to the last trading day at or before it,
    because an exact anniversary is usually a weekend or an Idul Fitri closure
    and a missing date would silently score every name zero.
    """
    days = pd.DatetimeIndex(sorted(P["date"].unique()))
    out = {t: 0 for t in names}
    for j in (1, 2, 3):
        i = days.searchsorted(day - pd.Timedelta(days=int(365.25 * j)),
                              "right") - 1
        if i < 0:
            continue
        past = set(universe(P, days[i])["ticker"])
        for t in names:
            out[t] += int(t in past)
    return out


def listed_years(P: pd.DataFrame, day: pd.Timestamp) -> pd.Series:
    return (day - P.groupby("ticker")["date"].min()).dt.days / 365.25


def resolve_day(P: pd.DataFrame) -> pd.Timestamp:
    """The last session with a REPRESENTATIVE cross-section, not just the last.

    A19's brief hit this: the panel can carry a handful of names for several
    days after a partial refresh, and ranking a forty-name cross-section by
    liquidity returns the forty largest names trivially. A11 records the same
    defect printing "71.7% of names above the 20-day" off forty large caps.
    """
    cnt = P.groupby("date")["ticker"].count()
    full = cnt[cnt >= 0.8 * cnt.tail(250).max()]
    return full.index.max()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default=PANEL)
    ap.add_argument("--size", type=int, default=10,
                    help="names to print; the rule itself has no cut")
    a = ap.parse_args()

    P = pd.read_parquet(a.panel, columns=["date", "ticker", "close",
                                          "tradeable", "log_turnover"])
    P["date"] = pd.to_datetime(P["date"])
    day = resolve_day(P)
    U = universe(P, day)
    if U.empty:
        print(" no representative cross-section available — refresh the panel")
        return 1

    W = 78
    print("=" * W)
    print(" THE DECADE BASKET".center(W))
    print("=" * W)
    print(f" as of {day.date()}   ({len(U)} names clear the rule)")
    if day < P["date"].max():
        print(f" NOTE: the panel runs to {P['date'].max().date()} but the last"
              f" {(P['date'].max() - day).days} days")
        print("       carry too few names to rank; using the last full session.")

    ten = tenure(P, day, set(U["ticker"]))
    age = listed_years(P, day)
    print(f"\n {'#':<3}{'ticker':<8}{'close':>10}{'Rp/day':>12}"
          f"{'round trip':>12}{'yrs in decile':>15}{'listed':>9}")
    tot = 0.0
    for i, (_, r) in enumerate(U.head(a.size).iterrows(), 1):
        t = r["ticker"]
        cb = B.cost_bar(float(r["close"]), day)
        tot += cb["total"]
        yrs = float(age.get(t, 0.0))
        flag = ""
        if ten[t] == 3 and yrs >= MIN_LISTED_YEARS:
            flag = "  CORE"
        elif ten[t] == 3:
            flag = "  core*"
        print(f" {i:<3}{t:<8}{r['close']:>10,.0f}"
              f"{np.exp(r['log_turnover']) / 1e9:>9,.0f} bn"
              f"{cb['total']:>11.2%}{ten[t]:>12} of 3{yrs:>8.1f}y{flag}")
    n = min(a.size, len(U))
    core = [t for t in U["ticker"] if ten[t] == 3
            and float(age.get(t, 0.0)) >= MIN_LISTED_YEARS]
    short = [t for t in U["ticker"] if ten[t] == 3
             and float(age.get(t, 0.0)) < MIN_LISTED_YEARS]
    print(f"\n mean round trip across the basket: {tot / n:.2%}")
    print(f" over ten years that is {tot / n / 10:.3%} a year — which is the")
    print(" whole mechanism. The same rule rebalanced annually pays it ten")
    print(" times and every one-year study in this repo shows that losing.")

    e = EVIDENCE
    print("\n" + "=" * W)
    print(" WHAT WAS MEASURED, ON 6,332 MATCHED TEN-YEAR WINDOWS")
    print("=" * W)
    print(f"   {'':<26}{'this basket':>14}{'all liquid':>12}{'IHSG TR':>11}")
    print(f"   {'name doubles in 10y':<26}{e['touch2x']:>14.1%}"
          f"{e['base_touch']:>12.1%}{'—':>11}")
    print(f"   {'median return':<26}{e['median']:>+14.1%}{'+12.8%':>12}"
          f"{e['index_median']:>+11.1%}")
    print(f"   {'P(-50%)':<26}{e['p_half']:>14.1%}{e['base_half']:>12.1%}"
          f"{'—':>11}")
    print(f"\n   paired against the index, per window: median "
          f"{e['paired_median']:+.1%}, won {e['win_rate']:.1%}")
    print(f"   positive in BOTH halves (+32.9% early, +73.9% late)")
    print(f"\n   So: about SEVEN of ten names double over the decade. That is")
    print(f"   the answer to 'can I pick 7 of 10 multi-baggers' — yes, at")
    print(f"   this horizon, and it is settled at ENTRY.")

    print("\n" + "=" * W)
    print(" WHAT IT IS NOT")
    print("=" * W)
    print(f" * NOT established. Permutation p = {e['p']:.4f} against this")
    print(f"   project's Bonferroni bar of {e['bar']:.5f} after 70 trials.")
    print(f" * Effective n is {e['eff_n']} for the sample and ~6 for the")
    print("   decile. Two independent decades, not 6,332 observations.")
    print(" * Only 35 distinct names were ever in the historical decile, so")
    print("   the cross-section is a list rather than a population.")
    print(" * The base rate does more work than the selection: 70.6% of")
    print("   decades starting early, 40.3% starting late.")
    print(" * The holdout was spent at H16. None of this is out of sample.")
    print(" * TEN YEARS IS THE MECHANISM. Selling at year three puts you back")
    print("   in the regime where costs eat every effect this data can find.")
    print("\n One reassurance: the historical decile contains BUMI, which fell")
    print(" ~99% from its peak. This is not a survivor list.")

    c = CORE
    print("\n" + "=" * W)
    print(" THE TIGHTER CELL — 8 OF 10, AND WHY IT IS NOT A FREE UPGRADE")
    print("=" * W)
    print(f" Names ALREADY in the decile for all three prior years touched 2x")
    print(f" {c['touch2x']:.1%} of the time — eight of ten — against "
          f"{EVIDENCE['touch2x']:.1%} for the decile")
    print(f" as a whole, median {c['median']:+.1%}, P(-50%) {c['p_half']:.1%}.")
    print(f" It barely moved between halves ({c['early']:.1%} early, "
          f"{c['late']:.1%} late)")
    print(f" while the BASE rate collapsed from 70.6% to 40.3%.")
    print(f"\n THREE REASONS NOT TO TREAT THAT AS THE ANSWER:")
    print(f"  1. It rests on {c['n_names']} distinct names and an effective n"
          f" of {c['eff_n']}. That")
    print(f"     is about ONE independent observation. p = {c['p']:.4f}"
          f" against a bar of")
    print(f"     {c['bar']:.5f} — closer than the decile, still short.")
    print(f"  2. It was found by LOOKING, after the decile result was in hand.")
    print(f"     Every earlier post-hoc cell in this repo that looked this")
    print(f"     good failed its replication.")
    if core:
        print(f"  3. It yields only {len(core)} names today "
              f"({', '.join(core)}) — not ten.")
        banks = [t for t in core if t.startswith("BB") or t == "BMRI"]
        if len(banks) >= 2:
            print(f"     And {len(banks)} of {len(core)} are banks "
                  f"({', '.join(banks)}), so it is one sector bet,")
            print(f"     not a diversified basket.")
    if short:
        print(f"\n  Excluded from CORE despite a 3-of-3 score: "
              f"{', '.join(short)} — listed")
        print(f"  less than {MIN_LISTED_YEARS:.0f} years, so the tenure score"
              f" is its whole life and it")
        print(f"  is not the object the historical cell measured (those eight"
              f" were all")
        print(f"  20-year names). Marked core* above.")
    print(f"\n So the choice is explicit: {len(core) if core else 0} names at"
          f" ~{c['touch2x']:.0%}, or {n} names at ~{EVIDENCE['touch2x']:.0%}.")
    print(" Ten names at eight-of-ten is not available from this evidence.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
