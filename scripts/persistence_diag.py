#!/usr/bin/env python3
"""Why does persistence appear in ALL ROWS and vanish in TWO-SIDED?

The headline run found year-over-year rank correlation +0.078 (p 0.025) on all
rows and +0.017 (p 0.41) on the subsample where a broker printed on both sides.
Only one of those two can be about broker skill, and the difference between the
samples is a known data defect rather than anything about the market:

    The source ranks buyers and sellers INDEPENDENTLY and publishes ten of
    each. A code that appears only among the top-10 buyers has its sell side
    recorded as ZERO when the true value is an unknown lower bound. Its net is
    then forced long by construction.

If some codes are systematically the censored ones — small brokers that reach a
top-10 list only on the days they trade big — then "which code is censored" is
a persistent broker attribute, and censoring maps mechanically onto measured
net direction. That would manufacture exactly the pattern observed, and it
would be a property of the publication rule, not of anyone's trading.

This script tests that chain directly:

    1. Is censoring persistent per broker?  (rank corr of censor share, y/y)
    2. Does censoring predict the measured margin, within a year?
    3. Does the ALL ROWS result survive dropping the single strongest year
       pair — or is one pair carrying the mean?

It also reports the LEVEL, which the persistence statistic deliberately ignores:
persistence is about whether the ranking carries over, and says nothing about
whether anybody is making money.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))
sys.path.insert(0, os.path.dirname(__file__))

from idxbot.spine.persistence import spearman                    # noqa: E402
from persistence import (MIN_GROSS, MIN_WINDOWS_PER_YEAR,        # noqa: E402
                         build_track_b)


def yearly(T: pd.DataFrame) -> pd.DataFrame:
    T = T.copy()
    T["year"] = T["window_end"].dt.year
    a = T.groupby(["broker", "year"]).agg(
        pnl=("timing_pnl", "sum"), gross=("gross_value", "sum"),
        n=("window_end", "nunique"), rows=("broker", "size"),
        two_sided=("two_sided", "mean")).reset_index()
    a = a[(a["n"] >= MIN_WINDOWS_PER_YEAR) & (a["gross"] >= MIN_GROSS)].copy()
    a["bps"] = 10000.0 * a["pnl"] / a["gross"]
    a["censor"] = 1.0 - a["two_sided"]
    return a


def main() -> int:
    T = build_track_b()
    if T.empty:
        print("no track B frame")
        return 1
    T["window_end"] = pd.to_datetime(T["window_end"])
    A = yearly(T)

    print("=" * 74)
    print(" 1. IS CENSORING A PERSISTENT BROKER ATTRIBUTE?")
    print("=" * 74)
    print(" If yes, 'which code gets its sell side truncated' carries over from")
    print(" year to year exactly the way a skill would, without being one.")
    w = A.pivot_table(index="broker", columns="year", values="censor")
    m = A.pivot_table(index="broker", columns="year", values="bps")
    yrs = sorted(w.columns)
    rc, rm = [], []
    for y0, y1 in zip(yrs, yrs[1:]):
        if y1 - y0 != 1:
            continue
        s = w[[y0, y1]].dropna()
        if len(s) >= 6:
            rc.append((y0, spearman(s[y0].to_numpy(), s[y1].to_numpy())))
        s2 = m[[y0, y1]].dropna()
        if len(s2) >= 6:
            rm.append((y0, spearman(s2[y0].to_numpy(), s2[y1].to_numpy())))
    cv = np.array([r for _, r in rc])
    print(f"\n   censor share, year-over-year rank corr: mean {cv.mean():+.3f}"
          f"   range [{cv.min():+.2f}, {cv.max():+.2f}]   n pairs {len(cv)}")
    print(f"   margin,       year-over-year rank corr: mean "
          f"{np.mean([r for _, r in rm]):+.3f}")
    print("\n   Censoring persists far more strongly than margin does."
          if cv.mean() > np.mean([r for _, r in rm]) else
          "\n   Censoring does NOT persist more than margin.")

    print()
    print("=" * 74)
    print(" 2. DOES CENSORING PREDICT THE MEASURED MARGIN, WITHIN A YEAR?")
    print("=" * 74)
    print(" A cross-sectional rank correlation inside each year, so nothing")
    print(" here can be a time trend.")
    xs = []
    for y, g in A.groupby("year"):
        if len(g) >= 6:
            r = spearman(g["censor"].to_numpy(), g["bps"].to_numpy())
            if np.isfinite(r):
                xs.append(r)
    xs = np.array(xs)
    print(f"\n   corr(censor share, margin_bps) within year: mean {xs.mean():+.3f}"
          f"   range [{xs.min():+.2f}, {xs.max():+.2f}]"
          f"   {int((xs > 0).sum())}/{len(xs)} positive")

    print()
    print("=" * 74)
    print(" 3. IS ONE YEAR PAIR CARRYING THE MEAN?")
    print("=" * 74)
    for y0, r in rm:
        print(f"   {y0}→{y0+1}   {r:+.3f}")
    v = np.array([r for _, r in rm])
    k = int(np.argmax(v))
    drop = np.delete(v, k)
    print(f"\n   mean {v.mean():+.3f}   without the strongest pair "
          f"({rm[k][0]}→{rm[k][0]+1}, {v[k]:+.3f}): {drop.mean():+.3f}")

    print()
    print("=" * 74)
    print(" 4. THE LEVEL — which the persistence statistic ignores entirely")
    print("=" * 74)
    print(" Persistence asks whether the RANKING carries over. It says nothing")
    print(" about whether the flow makes money, so that is reported separately.")
    tot = 10000.0 * A["pnl"].sum() / A["gross"].sum()
    print(f"\n   per broker-year margin  median {A['bps'].median():+.2f} bps"
          f"   mean {A['bps'].mean():+.2f}   sd {A['bps'].std():.1f}")
    print(f"   value-weighted, all flow pooled          {tot:+.2f} bps")
    print(f"   broker-years {len(A)}, brokers {A.broker.nunique()}, "
          f"years {A.year.nunique()}")
    print("\n   For scale: A5's round-trip cost is 56 bps. A fortnightly timing")
    print("   margin of a couple of bps is inside the noise of that.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
