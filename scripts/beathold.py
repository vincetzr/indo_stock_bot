#!/usr/bin/env python3
"""H54 — beat buy-and-hold. Every family this repo has built, one fixed bar.

THE GOAL, in the user's words: "beat buy and hold. if need be forget everything
and start over." So the target is not a win rate, not a per-trade mean, not an
IC. It is three benchmarks a holder could actually take instead, and a strategy
must beat ALL THREE net of cost and be ahead in BOTH HALVES:

    BH_INDEX      the IHSG, total-return, over the strategy's own window
    BH_UNIVERSE   equal-weighted basket of everything eligible on day one,
                  never touched again
    BH_PICKS      the strategy's own first basket, bought once and held

plus a random control drawn from the same universe on the same calendar.

WHY THIS FILE EXISTS AND THE EARLIER RUN DOES NOT COUNT
A fleet of strategy designs was run against the first version of this harness
and three of them "passed". All three passes were artefacts of MY harness, not
findings about the market:

  1. CASH DRAG. A mark the strategy could not act on earned it ZERO while all
     three benchmarks compounded through the window. At an annual calendar that
     was 9 of 27 marks, including windows the IHSG ran +90.6% and +95.5% -- a
     handicap of 1.8 to 6.7 points a year applied to the strategy alone.
  2. THE REBALANCE PHASE. Offset 0 is the panel's first bar for no reason but
     that it is first. All three passes were properties of that start date and
     vanished at every other offset tested.
  3. THE CONTROL'S CALENDAR. The random arm ran at offset 0 whatever the
     strategy did.

All three are fixed in `bhbench.py`. This is the re-run, and everything below
is measured on the corrected harness with a PASS requiring a MAJORITY of
rebalance phases rather than one lucky calendar.

PRE-REGISTRATION, written before this file was run.
  D1  No selection rule beats all three benchmarks in a majority of phases.
      This is the prediction. 54 hypotheses of prior art say so.
  D2  Removing the cash drag lifts every strategy's CAGR, most at the lowest
      rebalance frequency (where skipped marks are longest), and does NOT
      change the ORDERING of strategies against each other -- because it is a
      common-mode handicap. If the ordering does change, the drag was
      interacting with turnover and that is a finding.
  D3  PREDICTED NULL: `own everything, quarterly` and `own everything, annual`
      differ only in toll, so their gross CAGRs should be within noise of each
      other. If they are not, the harness is still charging or crediting
      something it should not, and no result in the table is readable.
  D4  Name churn, not weight rebalancing, is what destroys the equal-weight
      arm. Freezing the NAMES and resetting their WEIGHTS to equal should stay
      close to the never-touched basket; re-selecting names each period should
      not. (The fleet measured +9.09% against +1.15%; this is the check.)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Callable, Dict, List, Sequence

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from bhbench import (MIN_BASKET, Bench, load,                      # noqa: E402
                     report)

OUT = os.path.join("reports", "beathold.txt")
JOUT = os.path.join("reports", "beathold.json")


# ------------------------------------------------------------------ selectors
def ew(tks: Sequence[str]) -> List:
    tks = list(tks)
    return [(t, 1.0 / len(tks)) for t in tks] if tks else []


def s_all(day: pd.DataFrame) -> List:
    """Own every eligible name, equal-weighted. The reference arm."""
    return ew(day["ticker"])


def top(col: str, k: int, largest: bool = True) -> Callable:
    def f(day: pd.DataFrame) -> List:
        d = day.dropna(subset=[col])
        if len(d) < k:
            return []
        s = d.nlargest(k, col) if largest else d.nsmallest(k, col)
        return ew(s["ticker"])
    return f


def strcalm(k: int = 10, hi_q: float = 0.90, vol_q: float = 0.50) -> Callable:
    """A24/H26: within ~2% of the 52-week high AND below-median 60-day vol.

    The best selection signal in this project -- +12.9%/yr gross over a random
    basket from its own universe (A38) -- and it has tied the index every time
    it has been measured, because its edge is almost exactly the size of the
    equal-weight handicap it runs under.
    """
    def f(day: pd.DataFrame) -> List:
        d = day.dropna(subset=["hi52", "vol60"])
        if len(d) < 3 * k:
            return []
        d = d[(d["hi52"] >= d["hi52"].quantile(hi_q))
              & (d["vol60"] <= d["vol60"].quantile(vol_q))]
        if len(d) < k:
            return ew(d["ticker"])
        return ew(d.nlargest(k, "hi52")["ticker"])
    return f


class Sticky:
    """Hold what you own while it stays merely GOOD; replace only what fails.

    THE BUFFER IS THE WHOLE IDEA. A hard screen sells a name the day it slips
    out of the top decile and buys it back when it returns, which is churn with
    no informational content. A name is kept while it stays inside the top
    `keep_q` of the screen and is only dropped when it falls out of that wider
    band -- so turnover is driven by real deterioration rather than by rank
    noise around a cut.
    """

    def __init__(self, k: int = 10, hi_q: float = 0.90, vol_q: float = 0.50,
                 keep_hi_q: float = 0.60, keep_vol_q: float = 0.75,
                 reset_weights: bool = True):
        self.k, self.hi_q, self.vol_q = k, hi_q, vol_q
        self.keep_hi_q, self.keep_vol_q = keep_hi_q, keep_vol_q
        self.reset_weights = reset_weights
        self.held: List[str] = []

    def reset(self) -> None:
        self.held = []

    def __call__(self, day: pd.DataFrame) -> List:
        d = day.dropna(subset=["hi52", "vol60"])
        if len(d) < 3 * self.k:
            return []
        keep_ok = set(d[(d["hi52"] >= d["hi52"].quantile(self.keep_hi_q))
                        & (d["vol60"] <= d["vol60"].quantile(self.keep_vol_q))]
                      ["ticker"])
        live = set(d["ticker"])
        held = [t for t in self.held if t in keep_ok]
        if len(held) < self.k:
            cand = d[(d["hi52"] >= d["hi52"].quantile(self.hi_q))
                     & (d["vol60"] <= d["vol60"].quantile(self.vol_q))]
            cand = cand[~cand["ticker"].isin(held)].nlargest(
                self.k - len(held), "hi52")
            held = held + list(cand["ticker"])
        held = [t for t in held if t in live][:self.k]
        if len(held) < MIN_BASKET:
            return []
        self.held = held
        return ew(held)


class Frozen:
    """Pick ONCE, then never change the names. D4's instrument.

    `reset_weights=True` resets the surviving names to equal weight at every
    mark -- a real trade, and the harness charges for it. `False` returns the
    drifted weights, which is a pick-once-and-sleep book. The pair separates
    NAME CHURN from WEIGHT REBALANCING, which the fleet found to be the whole
    difference between +1.15% and +9.09% a year.
    """

    def __init__(self, base: Callable, reset_weights: bool = True):
        self.base, self.reset_weights = base, reset_weights
        self.held: List[str] = []
        self.w: Dict[str, float] = {}
        self.px: Dict[str, float] = {}

    def reset(self) -> None:
        self.held, self.w, self.px = [], {}, {}

    def __call__(self, day: pd.DataFrame) -> List:
        if not self.held:
            picks = self.base(day)
            #  DO NOT LATCH A BASKET THE HARNESS WILL REFUSE. A first version
            #  latched whatever the screen returned, so on the very first
            #  tradeable mark -- a 40-name universe, from which the joint screen
            #  yields TWO names -- it froze a two-name book the floor rejected
            #  and then never re-picked, for nineteen years. The whole arm read
            #  "no valid equity path" and would have been reported as a family
            #  that cannot be run rather than a bug in one line.
            if len(picks) < MIN_BASKET:
                return []
            self.held = [t for t, _ in picks]
            self.w = {t: w for t, w in picks}
            self.px = dict(zip(day["ticker"], day["adj_close"]))
            return picks
        live = dict(zip(day["ticker"], day["adj_close"]))
        held = [t for t in self.held if t in live]
        if len(held) < MIN_BASKET:
            return []
        if self.reset_weights:
            return ew(held)
        w = {}
        for t in held:
            p0 = self.px.get(t)
            w[t] = self.w.get(t, 0.0) * (live[t] / p0 if p0 else 1.0)
        tot = sum(w.values())
        if tot <= 0:
            return ew(held)
        self.w = {t: v / tot for t, v in w.items()}
        self.px = {t: live[t] for t in held}
        return list(self.w.items())


# ------------------------------------------------------------------- the grid
def grid() -> List:
    """(label, selector, freq). Every family, one bar."""
    Q, A, S = 63, 252, 126
    return [
        ("own everything, quarterly", s_all, Q),
        ("own everything, annual", s_all, A),
        ("liquidity top 10, annual", top("log_turnover", 10), A),
        ("liquidity top 20, annual", top("log_turnover", 20), A),
        ("momentum top 10, quarterly", top("mom12_1", 10), Q),
        ("low vol top 10, quarterly", top("lowvol", 10), Q),
        ("strength+calm 10, quarterly", strcalm(10), Q),
        ("strength+calm 10, semiannual", strcalm(10), S),
        ("strength+calm 10, annual", strcalm(10), A),
        ("strength+calm 20, annual", strcalm(20), A),
        #  THE STICKY FAMILY, swept over its one free parameter. The buffer is
        #  registered as a SWEEP and every cell is printed, because the point of
        #  the family is that turnover should be driven by deterioration rather
        #  than rank noise, and that claim is only readable if the whole
        #  buffer-width curve is shown rather than its maximum.
        ("strength+calm sticky, quarterly", Sticky(10), Q),
        ("sticky wide buffer, quarterly", Sticky(10, keep_hi_q=0.40,
                                                 keep_vol_q=0.90), Q),
        ("sticky tight buffer, quarterly", Sticky(10, keep_hi_q=0.80,
                                                  keep_vol_q=0.60), Q),
        ("sticky no buffer (=hard screen)", Sticky(10, keep_hi_q=0.90,
                                                   keep_vol_q=0.50), Q),
        ("sticky 20 names, quarterly", Sticky(20), Q),
        ("strength+calm sticky, semiannual", Sticky(10), S),
        ("strength+calm sticky, annual", Sticky(10), A),
        ("frozen strength+calm, reset wts", Frozen(strcalm(10)), A),
        ("frozen strength+calm, drift wts",
         Frozen(strcalm(10), reset_weights=False), A),
        ("frozen everything, reset wts", Frozen(s_all), A),
        ("frozen everything, drift wts", Frozen(s_all, reset_weights=False), A),
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phases", type=int, default=6)
    ap.add_argument("--draws", type=int, default=6)
    ap.add_argument("--nocarry", action="store_true",
                    help="reinstate the cash-drag bug, to measure what it cost")
    ap.add_argument("--out", default=None)
    ap.add_argument("--only", default=None,
                    help="substring filter on the label")
    args = ap.parse_args()

    P = load()
    B = Bench(P, carry=not args.nocarry)
    rows, txt = [], []
    for label, sel, freq in grid():
        if args.only and args.only.lower() not in label.lower():
            continue
        v = B.evaluate(sel, label, freq=freq, draws=args.draws,
                       phases=args.phases)
        rows.append({k: x for k, x in v.items() if k != "curve"})
        txt.append(report(v))
        print(report(v), flush=True)
        print("", flush=True)

    os.makedirs("reports", exist_ok=True)
    ok = [r for r in rows if r.get("ok")]
    #  SORTED ON EXCESS OVER EACH ARM'S OWN INDEX, not on CAGR. Every row has
    #  its own window -- the index column runs +7.21% to +11.31% across this
    #  table -- so ranking on raw CAGR would rank the decades, not the rules.
    hdr = (f"{'strategy':<34}{'vs idx':>9}{'lo..hi':>17}{'CAGR':>9}"
           f"{'index':>9}{'picks':>9}{'rand':>9}{'>idx':>6}"
           f"{'strict':>8}{'divers':>8}")
    tab = [hdr, "-" * len(hdr)]
    for r in sorted(ok, key=lambda x: -x.get("excess_med", -9)):
        tab.append(
            f"{r['label']:<34}{r.get('excess_med', np.nan):>8.2%} "
            f"{r.get('excess_lo', np.nan):>7.2%}.."
            f"{r.get('excess_hi', np.nan):<8.2%}"
            f"{r.get('cagr_med', r['cagr']):>9.2%}"
            f"{r['bh_index']:>9.2%}"
            f"{r['bh_picks']:>9.2%}{r['random']:>9.2%}"
            f"{r.get('beats_index_n', 0):>3}/{r.get('phases', 0)}"
            f"{r.get('phases_pass', 0):>5}/{r.get('phases', 0)}"
            f"{r.get('phases_pass_div', 0):>5}/{r.get('phases', 0)}")
    body = "\n".join(txt) + "\n\n" + "\n".join(tab) + "\n"
    out = args.out or OUT
    with open(out, "w") as f:
        f.write(body)
    with open(out.replace(".txt", ".json"), "w") as f:
        json.dump(rows, f, indent=1, default=str)
    print("\n".join(tab))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
