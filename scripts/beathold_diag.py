#!/usr/bin/env python3
"""H54 diagnostics: D3, D4, and today's basket. All on ONE fixed window.

WHY THIS IS A SEPARATE FILE. `beathold.py`'s table equalises the six PHASES of
each strategy, so a "4 of 6" is a replication test. It does NOT equalise across
STRATEGIES: each row starts at its own first tradeable mark, so the index column
in that table runs +6.21% to +11.31% over the same panel. Reading D3 or D4 off
two rows of it would compare two different decades -- A19's error class, which
this repo has now committed at three separate levels. Every comparison here
therefore runs on one explicitly fixed window.

  D3  PREDICTED NULL. `own everything` quarterly vs annual differ only in toll.
      If their excess over the index differs by much more than the toll gap, the
      harness is charging or crediting something it should not and no number in
      the main table is readable.
  D4  Is it NAME CHURN or WEIGHT REBALANCING that destroys the equal-weight
      arm? Four arms separate them: re-select+reset, freeze+reset, freeze+drift,
      re-select+drift.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from beathold import Frozen, Sticky, ew, s_all, strcalm               # noqa: E402
from bhbench import Bench, load                                       # noqa: E402

#  The narrowest window every arm below can actually occupy, taken from the
#  main run's scout output rather than chosen.
LO, HI = np.datetime64("2008-02-04"), np.datetime64("2025-07-17")


class Drifter:
    """Re-select names every mark, but never reset weights on survivors.

    The missing cell of D4's 2x2: `Frozen(drift)` freezes names AND drifts
    weights, so on its own it cannot say which of the two is doing the work.
    """

    def __init__(self, base):
        self.base, self.w, self.px = base, {}, {}

    def reset(self):
        self.w, self.px = {}, {}

    def __call__(self, day):
        picks = self.base(day)
        if not picks:
            return []
        live = dict(zip(day["ticker"], day["adj_close"]))
        new = {t: w for t, w in picks}
        out = {}
        for t in new:
            if t in self.w and t in self.px and self.px[t] > 0:
                out[t] = self.w[t] * (live[t] / self.px[t])   # let it ride
            else:
                out[t] = new[t]                               # fund the new one
        tot = sum(out.values())
        if tot <= 0:
            return picks
        self.w = {t: v / tot for t, v in out.items()}
        self.px = {t: live[t] for t in out if t in live}
        return list(self.w.items())


def main() -> None:
    P = load()
    B = Bench(P)
    Q, A = 63, 252

    print(f"ONE FIXED WINDOW for every arm below: "
          f"{pd.Timestamp(LO).date()} -> {pd.Timestamp(HI).date()}\n")

    arms = [
        ("D3  own everything, quarterly", s_all, Q),
        ("D3  own everything, annual", s_all, A),
        ("D4  re-select names + reset weights", s_all, A),
        ("D4  freeze names + reset weights", Frozen(s_all), A),
        ("D4  freeze names + drift weights",
         Frozen(s_all, reset_weights=False), A),
        ("D4  re-select names + drift weights", Drifter(s_all), A),
        ("    strength+calm, hard screen", strcalm(10), Q),
        ("    strength+calm, tight buffer",
         Sticky(10, keep_hi_q=0.80, keep_vol_q=0.60), Q),
    ]
    print(f"{'arm':<38}{'CAGR':>9}{'index':>9}{'vs idx':>9}"
          f"{'cost/yr':>9}{'turn':>7}{'>idx':>6}")
    print("-" * 87)
    for label, sel, freq in arms:
        v = B.evaluate(sel, label, freq=freq, draws=4, phases=6, lo=LO, hi=HI)
        if not v.get("ok"):
            print(f"{label:<38}  FAILED TO RUN")
            continue
        print(f"{label:<38}{v['cagr_med']:>9.2%}{v['bh_index']:>9.2%}"
              f"{v['excess_med']:>+9.2%}{v['cost_yr']:>9.2%}"
              f"{v['turnover']:>7.0%}"
              f"{v['beats_index_n']:>4}/{v['phases']}")

    # ------------------------------------------------------- today's basket
    print("\nTODAY'S BASKET from the leading rule (sticky, tight buffer).")
    print("IN-SAMPLE. The 24-month holdout was spent at H16, so this rule has")
    print("never been tested on data it did not see. It is a research output,")
    print("not advice, and A23's execution costs -- impact, suspension, auto-")
    print("rejection -- are in NO number in this project.\n")
    last = P[P["elig"]]["date"].max()
    day = P[(P["date"] == last) & P["elig"]].copy()
    sticky = Sticky(10, keep_hi_q=0.80, keep_vol_q=0.60)
    picks = dict(sticky(day))
    d = day[day["ticker"].isin(picks)].copy()
    d["Rp bn/day"] = d["tv60"] / 1e9
    d = d.sort_values("hi52", ascending=False)
    print(f"as of {pd.Timestamp(last).date()}   "
          f"({len(day)} eligible names, {len(d)} picked)")
    print(f"{'ticker':<9}{'close':>10}{'% off 52w high':>16}"
          f"{'60d vol':>10}{'Rp bn/day':>12}")
    for _, r in d.iterrows():
        print(f"{r['ticker']:<9}{r['close']:>10,.0f}{r['hi52'] * 100:>15.1f}%"
              f"{r['vol60'] * 100:>9.1f}%{r['Rp bn/day']:>12,.1f}")


if __name__ == "__main__":
    main()
