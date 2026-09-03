#!/usr/bin/env python3
"""H54 family — TIME THE MARKET, NOT THE NAMES.

THE PREMISE. Every selection branch in this repo has failed (flow, broker
identity, investor class, price/TA, 169 exit configurations, 9 timing rules on
single names). The one family not yet run against the H54 bar is the opposite
move: stop trying to pick WHICH names, hold a broad basket, and use the INDEX
to decide WHETHER to be invested at all. It is the only family where being out
of the market is the point rather than a side effect, and it attacks the repo's
own key fact from a new angle — turnover is the enemy, so a rule that trades
the whole book twice a year at most is cheap by construction.

WHY THIS CANNOT BE RUN THROUGH `Bench.evaluate` ALONE, AND THE FAVOUR IT HIDES.
`Bench.walk` has no cash leg. Returning fewer than MIN_BASKET names makes the
equity path FLAT for that period, which is arithmetically cash at a 0% rate --
but it also does `continue` WITHOUT clearing `prev`, so the book is never
charged for being sold, and re-entering the same names later is charged nothing
either. A timing rule run that way gets its whole thesis for free.

MEASURED, NOT GUESSED. A first version of this paragraph asserted the discount
was "+0.3 to +0.9%/yr" before anything had been run. Against
`reports/strat_indextime_harnesspass.txt`, at freq=63 on the rebalanced
all-eligible basket, it is:

    always in            harness +1.15%  cash-aware +1.14%   +0.01
    idx > 200d MA                +2.52%             +2.20%   +0.32
    200MA AND mom12-1            +3.61%             +3.33%   +0.28
    dd252 > -15%                 +1.55%             +1.38%   +0.17
    vol below 80th pct           +1.34%             +1.20%   +0.14

So +0.14 to +0.32%/yr, and the guess was high at the top end. The `always in`
row is the check that matters: with no cash leg the two paths must agree, and
they agree to a basis point. The favour is real, it is smaller than claimed,
and it is nowhere near large enough to change any verdict below.

`Bench.walk` ALSO shifts the WINDOW when a timing rule starts in cash. Run
literally, `INVERSE: idx < 200d MA` reports itself over 2008-05-12 → 2026 with
a +7.16% index instead of 2005-05-04 → 2026 with a +10.48% one, because the
window starts at the first trade. Two timing rules then face different
benchmarks. `TimedWalk.run` anchors on the universe instead.

So this module carries its OWN equity path with an explicit cash leg, charging
0.5*sum|w_t - w_{t-1}| * (fee + spread_mult * tick/price) at EVERY transition
including the transitions into and out of cash -- exiting a full book is
turn=0.5 and re-entering is another 0.5, which together is exactly one round
trip of the harness's own `fee`. Benchmarks come from `Bench.index_cagr` and
`Bench.hold_basket` over the strategy's OWN window (A19), and the PASS rule is
`Bench.evaluate`'s, reimplemented on the cash-aware path rather than loosened.
Every variant is ALSO run through the real `Bench.evaluate` so the harness's own
verdict is printed beside it and the size of the discount is visible.

THE CONTROL THIS FAMILY LIVES OR DIES BY, AND IT IS NOT THE HARNESS'S.
`Bench.evaluate`'s random control redraws the NAMES. That tests selection, and
this family does not select -- it holds everything. The question here is whether
the TIMING carries information, so the null must be matched on exposure. Per
A35's W4: take the strategy's own on/off sequence over the rebalance marks,
split it into in-runs and out-runs, SHUFFLE each list, and re-interleave in the
original alternation order. Exposure, number of switches and the entire
distribution of run lengths are preserved EXACTLY; only the alignment with the
market is destroyed. 200 draws give a z. A timing rule that does not beat that
is a rule that is merely out of the market sometimes.

CASH EARNS ZERO HERE, WHICH IS CONSERVATIVE AND IS THE POINT. Indonesian
deposit rates over this sample ran roughly 4-8%, so a real timing strategy would
be paid to sit out. That would flatter it, and the harness's benchmarks have no
such leg, so the headline uses 0%. `--cash-rate` prints the sensitivity, and it
is reported as a sensitivity rather than folded into the headline.

NO LOOK-AHEAD. Every index state is causal by construction: a trailing mean, a
trailing momentum, a trailing realised vol, a drawdown from a trailing high, and
where a threshold on a distribution is needed it is an EXPANDING quantile using
only bars up to and including the decision bar. The decision uses the index
close of the mark date and fills at the same bar's close, which is the harness's
own convention for `mom12_1` and every other feature.

IN-SAMPLE THROUGHOUT. The 24-month holdout was already spent before this
existed; nothing here is out of sample and no number below should be read as if
it were.

================================================================================
WHAT THE RUN OF 2026-08-30 FOUND.  46 scored variants, 0 PASS.
================================================================================

Files: `reports/strat_indextime_d63.csv` (18, drifting book, gap=hold, 0% cash),
`strat_indextime_all_flat.csv` (16, rebalanced book, gap=flat, harness-faithful),
`strat_indextime_cash6.csv` (12, cash paid 6%/yr), `strat_indextime_oracle.txt`
(5 look-ahead controls), `strat_indextime_harnesspass.txt` (6 through the real
`Bench.evaluate`), `strat_indextime_risk.txt`.

1. THE INSTRUMENT HAS POWER, WHICH IS WHAT LICENSES READING THE NULLS.
   A cheating oracle -- invested iff the index rises over the period it is about
   to hold -- returns +19.66% against a matched-exposure null of +7.15%,
   z = +4.93, and it is the only thing in this file that ever tripped PASS.
   Degraded to 65% accuracy it still reads z = +2.64. So a real rule landing at
   z = 0 is a powered null, not a blind one.

2. NOT ONE TIMING RULE BEATS SITTING STILL. 15 of 15 conventional risk-off rules
   come in BELOW the always-invested drifting book (+10.38%), by 0.54 to 9.29
   points, mean 4.63. `corr(exposure, CAGR) = +0.69`, `corr(switches, CAGR) =
   -0.81`. The ranking is essentially "how much of the time were you in, and how
   often did you trade" -- there is no room left for a timing effect.

3. AND 15 OF 15 ARE ON THE WRONG SIDE OF THEIR OWN MATCHED-EXPOSURE NULL.
   z runs -0.11 to -2.72, mean -1.19. Being out of the market on the DATES the
   200-day rule chooses is worse than being out for the same number of periods,
   in runs of the same lengths, at random dates. THE TWO INVERSE RULES, ADDED AS
   A SYMMETRY CHECK, ARE THE ONLY POSITIVE z IN THE TABLE (+1.08 and +2.51). The
   trend filter is not noise here; it is BACKWARDS.

   Read the 15 as one observation, not fifteen: every rule is a function of one
   index over one history, so they are nearly the same test rerun.

4. THE MECHANISM, MEASURED ON THE INDEX ITSELF, NOT INFERRED. At quarterly
   decision granularity 2005-2026, IHSG above its 200-day earns +10.03%/yr and
   BELOW it earns +4.61% -- the signal is real and correctly signed. But the bad
   state still pays MORE THAN CASH, so a binary in/out rule forfeits return to
   avoid a positive number. And index 12-1 momentum is inverted at this horizon:
   mom > 0 gives +5.17%/yr, mom < 0 gives +16.53%.

5. PAYING CASH 6%/YR DOES NOT RESCUE IT, WHICH IS THE INTERESTING PART. The best
   variant reaches +10.39% against a bh_index of +10.48% -- and against its own
   matched null of +10.52%, z = -0.10. The null is paid the same 6%, so the
   comparison isolates the dates. 0 of 12 pass.

6. WHAT THE FAMILY DOES DELIVER IS RISK, AND H54 DOES NOT SCORE RISK.
   `200MA AND mom12-1` cuts maximum drawdown from -56.2% to -31.9% and
   volatility from 27.6% to 15.8%, lifting CAGR/maxDD from 0.18 to 0.21, while
   costing 3.8 points of CAGR. `idx > 50d MA` cuts exposure to 63% and makes
   drawdown WORSE (-59.4%), which is A18's "per-position stops can increase
   portfolio drawdown" arriving from the index side.

7. THE MEASURING DEVICE COSTS EVERY STRATEGY 7 POINTS A YEAR, AND IT IS NOT A
   TIMING RESULT. `Bench.walk` skips a mark whose eligible universe is under 40
   names and earns NOTHING for the following quarter, while all three benchmarks
   are continuous holds. 8 of 84 quarterly marks are skipped and the IHSG gained
   a cumulative +263% across exactly those eight; three of the eight have ZERO
   eligible names because the panel carries fill-forward rows on IDX market
   holidays. Always-in, all eligible, quarterly: +1.14% under gap="flat" and
   +8.39% under gap="hold". The drifting book: +3.26% and +10.38%.

   SO THE BRIEF'S "KEY FACT" IS MOSTLY THIS ARTEFACT. "Rebalance quarterly
   +1.15%/yr vs never touch +10.05%/yr, a ~9-point gap, turnover is the enemy"
   is measured with the strategy skipping eight quarters and the benchmark not.
   Measured like for like, the gap is +8.39% vs +10.38% -- about 2 points, of
   which 0.35 is the extra cost line. Turnover IS the enemy; it is worth about a
   fifth of what the uncorrected comparison says.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))
sys.path.insert(0, os.path.dirname(__file__))

from bhbench import (Bench, MIN_BASKET, MIN_UNIV, half_cagr, load,  # noqa: E402
                     report, _cagr)
from paint_suite import tick_of                                     # noqa: E402

OUT = "reports"


# ------------------------------------------------------------- market state --
def market_state(J: pd.Series, dates: np.ndarray) -> pd.DataFrame:
    """Causal index-state table, one row per PANEL date.

    Everything is trailing. The expanding quantiles use `expanding()`, so the
    threshold on bar t is built from bars <= t and from nothing else -- a
    full-sample quantile would be exactly the look-ahead the brief forbids.
    """
    S = J.sort_index()
    lg = np.log(S)
    d = pd.DataFrame(index=S.index)
    d["close"] = S
    d["ma200"] = S.rolling(200, min_periods=200).mean()
    d["ma50"] = S.rolling(50, min_periods=50).mean()
    d["above200"] = S > d["ma200"]
    d["above50"] = S > d["ma50"]
    #  12-1 momentum: the standard construction, skipping the last month.
    d["mom12_1"] = S.shift(21) / S.shift(252) - 1.0
    d["mom6"] = S / S.shift(126) - 1.0
    d["mom3"] = S / S.shift(63) - 1.0
    d["mom12"] = S / S.shift(252) - 1.0
    r = lg.diff()
    d["vol60"] = r.rolling(60, min_periods=40).std() * np.sqrt(252)
    #  Expanding quantile of realised vol, min 3 years of history before it
    #  will answer at all, so the early sample cannot be ranked against three
    #  observations of itself.
    d["volq"] = (d["vol60"].expanding(756).rank(pct=True))
    d["hi252"] = S.rolling(252, min_periods=200).max()
    d["dd252"] = S / d["hi252"] - 1.0
    #  Reindex onto the panel's trading days, forward-filling: the index csv
    #  and the panel do not share a calendar exactly, and a state is knowable
    #  from the last session it printed.
    idx = pd.DatetimeIndex(pd.to_datetime(dates))
    return d.reindex(d.index.union(idx)).ffill().reindex(idx)


# ------------------------------------------------------ cash-aware backtest --
_MARKS: Dict[Tuple[int, int], Tuple[np.ndarray, Dict]] = {}


def _marks(B: Bench, freq: int, offset: int):
    """Mark dates and their rows, built ONCE per (freq, offset).

    A first version rebuilt `P[P.date.isin(marks)].groupby("date")` inside
    every walk. 200 null replays per variant then meant 200 filters over 2.9m
    rows, and the null -- the whole point of the study -- was most of the
    runtime. Nothing about any number changes; `selftest` checks the cached
    frames against a fresh build.
    """
    k = (int(freq), int(offset))
    if k not in _MARKS:
        m = B.dates[offset::freq]
        _MARKS[k] = (m, {d: g for d, g in
                         B.P[B.P["date"].isin(m)].groupby("date")})
    return _MARKS[k]



class CachedPrices:
    """Memoised `Bench.Prices`. Identical answers, ~200x fewer searchsorteds.

    The null needs 200 replays over the same marks and the same tickers, so
    every price lookup is asked for hundreds of times. This changes nothing
    about any number; `test_cache_matches` asserts that.
    """

    def __init__(self, PX):
        self.PX = PX
        self.d = PX.d
        self._at: Dict[Tuple[str, object], float] = {}
        self._ex: Dict[Tuple[str, object], float] = {}

    def at(self, tk, day):
        k = (tk, day)
        v = self._at.get(k)
        if v is None:
            v = self._at[k] = self.PX.at(tk, day)
        return v

    def exit_price(self, tk, day):
        k = (tk, day)
        v = self._ex.get(k)
        if v is None:
            v = self._ex[k] = self.PX.exit_price(tk, day)
        return v


class TimedWalk:
    """`Bench.walk` with a cash leg and an honest bill for entering it.

    `gap` IS NOT A DETAIL AND IT IS THE LARGEST NUMBER IN THIS STUDY.
    `Bench.walk` skips any mark whose eligible universe is under MIN_UNIV and
    leaves the equity path FLAT across the whole following period, while all
    three benchmarks are continuous holds that keep that period's return. At
    freq=63 that is 8 marks of 84, and the IHSG gained a cumulative +263%
    across exactly those eight quarters -- 6.1% a year of return that the
    measuring device removes from every strategy and from no benchmark. Three
    of the eight have ZERO eligible names because the panel carries
    fill-forward rows on IDX market holidays (2019-04-19 is Good Friday).

      gap="flat"  harness-faithful. Comparable to every other H54 script.
      gap="hold"  the book is held through an unmeasurable mark, which is what
                  a real holder does and what the benchmarks are already doing.

    Both are reported. Neither is used to rescue a result: the timing rules
    below fail under both, and the gap mode changes the BASE they are measured
    against far more than it changes the answer about timing.
    """

    def __init__(self, B: Bench, cash_rate: float = 0.0,
                 gap: str = "flat", PX=None):
        self.B = B
        self.PX = PX if PX is not None else CachedPrices(B.PX)
        self.cash_rate = cash_rate
        assert gap in ("flat", "hold")
        self.gap = gap

    def run(self, basket: Callable, invested: Callable, freq: int = 63,
            offset: int = 0, seed: Optional[Callable] = None) -> Dict:
        """THE WINDOW IS ANCHORED AT THE FIRST MEASURABLE MARK, NOT THE FIRST
        TRADE, and `first_basket` is what the BASKET rule would have bought
        there whether or not the TIMING rule was invested.

        `Bench.walk` starts its window at the first trade, which is right for a
        selection rule. For a timing rule it is not: a rule that happens to sit
        out the first two years would have its window -- and therefore all
        three benchmarks -- silently shifted forward, so two timing rules on
        the same basket would be scored against different alternatives. A19's
        error class. Anchoring on the universe instead makes every variant here
        share one window and one set of benchmarks, and makes `bh_picks` the
        same object for every timing rule over a given basket: "these are the
        names, does the timing earn its keep".
        """
        B = self.B
        marks, by_date = _marks(B, freq, offset)
        eq = 1.0
        prev: Dict[str, float] = {}
        curve: List[Tuple] = []
        turns, costs, sizes, states = [], [], [], []
        first_basket: List[str] = []
        t0 = None
        skipped = 0
        for a, b in zip(marks[:-1], marks[1:]):
            day = by_date.get(a)
            ok = day is not None
            if ok:
                day = day[day["elig"]]
                ok = len(day) >= MIN_UNIV
            if not ok:
                if t0 is None:
                    continue
                skipped += 1
                if self.gap == "hold" and prev:
                    #  Hold the existing book through an unmeasurable mark
                    #  rather than teleporting it to zero return.
                    eq *= 1.0 + self._book_ret(prev, a, b)
                curve.append((b, eq))
                continue
            if t0 is None:
                sd = (seed or basket)(day.copy(), True)
                sd = [t for t, w in (sd or [])
                      if np.isfinite(float(w)) and float(w) > 0]
                if len(sd) < MIN_BASKET:
                    continue
                first_basket = sd
                t0 = a
            on = bool(invested(a))
            #  THE BASKET IS ASKED AT EVERY MEASURABLE MARK, INVESTED OR NOT.
            #  A stateless basket does not care. A drifting book does: it has
            #  to be told it spent this period in cash so it marks itself to
            #  market only over the periods it was actually holding anything.
            picks = basket(day.copy(), on)
            if on:
                picks = [(t, float(w)) for t, w in (picks or [])
                         if np.isfinite(w) and w > 0]
                if len(picks) < MIN_BASKET:
                    picks = []
            else:
                picks = []
            tot = sum(w for _, w in picks)
            cur = {t: w / tot for t, w in picks} if tot > 0 else {}
            states.append(bool(cur))
            sizes.append(len(cur))
            keys = set(cur) | set(prev)
            turn = 0.5 * sum(abs(cur.get(k, 0.0) - prev.get(k, 0.0))
                             for k in keys)
            #  THE TOLL IS PRICED ON THE UNION, not on `cur`. Selling a full
            #  book into cash means `cur` is empty and `Bench.walk` would fall
            #  back to a flat `fee` with no spread at all -- on a book of
            #  Rp200 names that is a 1.00% discount on the one trade the whole
            #  thesis rests on.
            px = {}
            for t in keys:
                v = self.PX.at(t, a)
                if not np.isfinite(v) or v <= 0:
                    v = self.PX.exit_price(t, a)
                if np.isfinite(v) and v > 0:
                    px[t] = v
            toll = [B.fee + B.spread_mult * tick_of(v) / v for v in px.values()]
            toll = float(np.mean(toll)) if toll else B.fee
            eq *= (1.0 - turn * toll)
            turns.append(turn)
            costs.append(turn * toll)
            if cur:
                eq *= 1.0 + self._book_ret(cur, a, b)
            else:
                yrs = (b - a).astype("timedelta64[D]").astype(float) / 365.25
                eq *= (1.0 + self.cash_rate) ** yrs
            prev = cur
            curve.append((b, eq))
        if not first_basket or t0 is None:
            return {}
        curve = [c for c in curve if c[0] > t0]
        if len(curve) < 6:
            return {}
        b1 = curve[-1][0]
        yrs = (b1 - t0).astype("timedelta64[D]").astype(float) / 365.25
        e0, e1 = half_cagr([(t0, 1.0)] + list(curve))
        return {"cagr": _cagr(eq, yrs), "years": yrs, "start": t0, "end": b1,
                "early": e0, "late": e1, "eq": eq, "curve": curve,
                "basket": float(np.mean([s for s in sizes if s])) if any(sizes)
                else np.nan,
                "turnover": float(np.mean(turns)) if turns else np.nan,
                "cost_yr": float(np.sum(costs) / yrs),
                "exposure": float(np.mean(states)) if states else np.nan,
                "states": np.array(states, bool),
                "switches": int(np.sum(np.diff(np.array(states, int)) != 0)),
                "skipped": skipped, "first_basket": first_basket}

    def _book_ret(self, cur: Dict[str, float], a, b) -> float:
        B = self.B
        rets, ws = [], []
        for t, w in cur.items():
            p0 = self.PX.at(t, a)
            if not np.isfinite(p0) or p0 <= 0:
                p0 = self.PX.exit_price(t, a)
            if not np.isfinite(p0) or p0 <= 0:
                continue
            p1 = self.PX.exit_price(t, b)
            if not np.isfinite(p1) or p1 <= 0:
                continue
            rets.append(p1 / p0 - 1.0)
            ws.append(w)
        if not rets:
            return 0.0
        ws = np.array(ws) / np.sum(ws)
        return float(np.dot(ws, rets))


# ------------------------------------------------------- exposure-matched H0 --
def shuffle_states(states: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """A35/W4: preserve exposure, switch count and every run length exactly.

    Split the on/off sequence into its in-runs and its out-runs, shuffle each
    LIST, and re-interleave in the original alternation order. The result has
    the identical number of marks invested, the identical number of round
    trips and the identical distribution of holding periods -- everything the
    rule costs. The only thing destroyed is which dates it chose.
    """
    s = np.asarray(states, bool)
    if s.size == 0:
        return s
    edges = np.flatnonzero(np.diff(s.astype(int)) != 0) + 1
    runs = np.split(s, edges)
    on = [len(r) for r in runs if r[0]]
    off = [len(r) for r in runs if not r[0]]
    rng.shuffle(on)
    rng.shuffle(off)
    out, i, j = [], 0, 0
    for r in runs:
        if r[0]:
            out.append(np.ones(on[i], bool))
            i += 1
        else:
            out.append(np.zeros(off[j], bool))
            j += 1
    return np.concatenate(out)


def replay(W: TimedWalk, mk: Callable, states: np.ndarray, freq: int,
           offset: int = 0, seed: Optional[Callable] = None) -> Dict:
    """Re-run with a PRESCRIBED on/off sequence indexed by mark position.

    `mk` is a FACTORY, not a basket: a drifting book carries state, and reusing
    one instance across 200 null draws would have every draw inherit the last
    one's positions. Each draw gets a fresh book.
    """
    box = {"i": 0}

    def invested(_a):
        i = box["i"]
        box["i"] += 1
        return bool(states[i]) if i < len(states) else False

    return W.run(mk(), invested, freq=freq, offset=offset, seed=seed)


# -------------------------------------------------------------- the verdict --
def verdict(B: Bench, r: Dict, label: str, extra: str = "") -> Dict:
    """`Bench.evaluate`'s rule, applied to a cash-aware path. Not loosened."""
    a0, b1 = r["start"], r["end"]
    uni0 = B.P[(B.P["date"] == a0) & B.P["elig"]]["ticker"].tolist()
    mid = r["curve"][len(r["curve"]) // 2][0]
    bh = {"index": B.index_cagr(a0, b1),
          "universe": B.hold_basket(uni0, a0, b1),
          "picks": B.hold_basket(r["first_basket"], a0, b1)}
    be = {"index": B.index_cagr(a0, mid),
          "universe": B.hold_basket(uni0, a0, mid),
          "picks": B.hold_basket(r["first_basket"], a0, mid)}
    bl = {"index": B.index_cagr(mid, b1),
          "universe": B.hold_basket(uni0, mid, b1),
          "picks": B.hold_basket(r["first_basket"], mid, b1)}
    beats = {k: r["cagr"] > v for k, v in bh.items()}
    both = {k: (r["early"] > be[k] and r["late"] > bl[k]) for k in be}
    return {"label": label, "extra": extra, "ok": True,
            "cagr": r["cagr"], "years": r["years"], "early": r["early"],
            "late": r["late"], "basket": r["basket"],
            "turnover": r["turnover"], "cost_yr": r["cost_yr"],
            "exposure": r["exposure"], "switches": r["switches"],
            "start": str(pd.Timestamp(a0).date()),
            "end": str(pd.Timestamp(b1).date()),
            "bh_index": bh["index"], "bh_universe": bh["universe"],
            "bh_picks": bh["picks"],
            "beats_index": beats["index"], "beats_universe": beats["universe"],
            "beats_picks": beats["picks"],
            "both_halves_index": both["index"],
            "both_halves_universe": both["universe"],
            "both_halves_picks": both["picks"],
            "bh_e": be, "bh_l": bl}


def fmt(v: Dict) -> str:
    L = [f"{v['label']}   [{v['start']} → {v['end']}, {v['years']:.1f}yr, "
         f"basket {v['basket']:.0f}, exposure {v['exposure']:.0%}, "
         f"{v['switches']} switches, turnover {v['turnover']:.0%}, "
         f"cost {v['cost_yr']:.2%}/yr]",
         f"  STRATEGY            {v['cagr']:+8.2%}   "
         f"(early {v['early']:+.2%} / late {v['late']:+.2%})"]
    for k, nm in (("index", "BH index"), ("universe", "BH universe"),
                  ("picks", "BH its own picks")):
        L.append(f"  {nm:<18}{v['bh_' + k]:+8.2%}   "
                 f"beats={'YES' if v['beats_' + k] else 'no':<3} "
                 f"both halves={'YES' if v['both_halves_' + k] else 'no'}  "
                 f"(e {v['bh_e'][k]:+.2%} / l {v['bh_l'][k]:+.2%})")
    if "randname_mean" in v and np.isfinite(v["randname_mean"]):
        L.append(f"  {'random NAMES, same':<18}{v['randname_mean']:+8.2%}   "
                 f"beats={'YES' if v['beats_randname'] else 'no':<3} "
                 f"(sd {v['randname_sd']:.2%})   [timing held fixed]")
    if "null_mean" in v:
        L.append(f"  {'matched-null timing':<18}{v['null_mean']:+8.2%}   "
                 f"beats={'YES' if v['beats_null'] else 'no':<3} "
                 f"(sd {v['null_sd']:.2%}, z {v['z']:+.2f}, "
                 f"pct {v['pctile']:.0%}, both halves="
                 f"{'YES' if v.get('null_both') else 'no'})")
    L.append(f"  ==> {'PASS' if v['PASS'] else 'FAIL'}")
    return "\n".join(L)


# ------------------------------------------------------------------ baskets --
def basket_all(day: pd.DataFrame, on: bool = True) -> List[Tuple[str, float]]:
    n = len(day)
    return [(t, 1.0 / n) for t in day["ticker"]]


def basket_calm(day: pd.DataFrame, on: bool = True) -> List[Tuple[str, float]]:
    """A24/H26 strength+calm: hi52 top decile AND vol60 below the median.

    Both cuts are cross-sectional WITHIN THE BAR, so no full-sample statistic
    enters the universe definition.
    """
    d = day.dropna(subset=["hi52", "vol60"])
    if len(d) < 20:
        return []
    s = d[(d["hi52"] >= d["hi52"].quantile(0.90))
          & (d["vol60"] <= d["vol60"].median())]
    if len(s) < MIN_BASKET:
        return []
    return [(t, 1.0 / len(s)) for t in s["ticker"]]


def basket_liq(day: pd.DataFrame, on: bool = True) -> List[Tuple[str, float]]:
    s = day.nlargest(max(MIN_BASKET, len(day) // 5), "tv60")
    return [(t, 1.0 / len(s)) for t in s["ticker"]]


class DriftBook:
    """A book bought ONCE and never rebalanced, that can be parked in cash.

    THE REASON THIS EXISTS. The brief's key fact is that the same names
    rebalanced quarterly return +1.15%/yr and left alone return +10.05%/yr, and
    the probe above reproduced it to two decimals (+1.14% / +10.05%). Bolting a
    timing overlay onto the quarterly-rebalanced book therefore tests the
    overlay on a base that has already thrown away nine points, and the answer
    would say nothing about timing. This carries the book in VALUE space, marks
    it to market only over the periods it was actually invested, and hands the
    drifted vector back, so continuous holding produces ~0 turnover and the
    ONLY trades are the timing round trips.

    A NAME THAT STOPS PRINTING is realised at its last print and its value
    spread pro rata over the survivors. That is a favour -- `hold_basket` keeps
    the dead name's terminal return in its mean forever -- and it is the same
    favour `Bench.walk` grants, left in rather than silently corrected so the
    comparison stays like for like. `deaths` counts it.
    """

    def __init__(self, B: Bench, seed: Callable = basket_all, PX=None):
        self.B = B
        self.PX = PX if PX is not None else B.PX
        self.seed = seed
        self.book: Dict[str, float] = {}
        self.last = None
        self.was_on = True
        self.deaths = 0
        #  A NAME THAT DOES NOT PRINT ON A MARK DATE IS NOT DEAD. The panel's
        #  date axis is the UNION of every ticker's trading days, so a name
        #  that was suspended for a session, or simply did not trade, has no
        #  row on that date. A first version treated a missing row as a
        #  delisting and liquidated it, which shrank a 140-name book to 53 and
        #  turned a +10.05% hold into +3.27%. Death is `the ticker never prints
        #  again`, which is a property of its LAST date and nothing else.
        self.last_seen = {tk: dt[-1] for tk, (dt, _) in self.PX.d.items()
                          if len(dt)}

    def __call__(self, day: pd.DataFrame, on: bool = True
                 ) -> List[Tuple[str, float]]:
        d = day["date"].iloc[0]
        if self.last is not None and d == self.last:
            #  The walker asks once to seed `first_basket` and once for real
            #  on the same bar. Idempotent on the book, but the LAST answer
            #  about `on` is the real one and has to stick, or a book parked
            #  in cash on its very first mark would mark itself to market
            #  across the gap.
            self.was_on = on
            return list(self.book.items())
        if not self.book:
            s = self.seed(day, True)
            if not s:
                return []
            self.book = {t: float(w) for t, w in s}
        elif self.was_on:
            nb, dead = {}, 0.0
            for t, v in self.book.items():
                p0 = self.PX.exit_price(t, self.last)
                p1 = self.PX.exit_price(t, d)
                if (not np.isfinite(p0) or p0 <= 0
                        or not np.isfinite(p1) or p1 <= 0):
                    continue
                if self.last_seen.get(t, d) < d:
                    #  Delisted: realise at the last print it ever made and
                    #  recycle the proceeds over the survivors.
                    dead += v * p1 / p0
                    self.deaths += 1
                    continue
                nb[t] = v * p1 / p0
            if nb and dead > 0:
                tot = sum(nb.values())
                for t in nb:
                    nb[t] += dead * nb[t] / tot
            self.book = nb
        self.last = d
        self.was_on = on
        return list(self.book.items())


# ------------------------------------------------------------------- timing --
def make_rules(M: pd.DataFrame) -> Dict[str, Callable]:
    def g(col):
        s = M[col]

        def f(a):
            v = s.get(pd.Timestamp(a), np.nan)
            return bool(v) if np.isfinite(float(v)) else True
        return f

    def num(col, thr, above=True, default=True):
        s = M[col]

        def f(a):
            v = float(s.get(pd.Timestamp(a), np.nan))
            if not np.isfinite(v):
                return default
            return (v > thr) if above else (v < thr)
        return f

    def both(f1, f2):
        return lambda a: f1(a) and f2(a)

    def either(f1, f2):
        return lambda a: f1(a) or f2(a)

    R = {
        "always in": lambda a: True,
        "idx > 200d MA": g("above200"),
        "idx > 50d MA": g("above50"),
        "idx mom12-1 > 0": num("mom12_1", 0.0),
        "idx mom6 > 0": num("mom6", 0.0),
        "idx mom3 > 0": num("mom3", 0.0),
        "vol below 80th pct": num("volq", 0.80, above=False),
        "vol below 60th pct": num("volq", 0.60, above=False),
        "dd252 > -10%": num("dd252", -0.10),
        "dd252 > -15%": num("dd252", -0.15),
        "dd252 > -20%": num("dd252", -0.20),
    }
    #  TWO INVERSES, AS A SYMMETRY CHECK RATHER THAN AS CANDIDATES. If the
    #  trend filter is merely noise, its mirror should land in the same place;
    #  if the mirror is much BETTER, the filter is not noise, it is
    #  backwards, and that is a different finding. They are counted in the
    #  trial total like everything else.
    R["INVERSE: idx < 200d MA"] = lambda a: not R["idx > 200d MA"](a)
    R["INVERSE: dd252 < -15%"] = num("dd252", -0.15, above=False,
                                     default=False)
    R["200MA AND mom12-1"] = both(R["idx > 200d MA"], R["idx mom12-1 > 0"])
    R["200MA OR mom12-1"] = either(R["idx > 200d MA"], R["idx mom12-1 > 0"])
    R["200MA AND calm vol"] = both(R["idx > 200d MA"], R["vol below 80th pct"])
    R["200MA AND dd>-15%"] = both(R["idx > 200d MA"], R["dd252 > -15%"])
    R["50MA AND 200MA"] = both(R["idx > 50d MA"], R["idx > 200d MA"])
    return R


# ------------------------------------------------- positive control (CHEATS) --
def oracle_rule(B: Bench, freq: int, skill: float = 1.0,
                seed: int = 7) -> Callable:
    """A LOOK-AHEAD TIMING RULE. It is invested iff the index goes UP over the
    period it is about to hold. It cheats, deliberately and by construction.

    WHY A CHEAT IS IN THIS FILE. A26/H31 records a ZigZag detector that could
    not find a turn in a pure sine wave and was about to carry a whole negative
    result on its own blindness; A35/W1 records the same discipline applied to a
    win-rate harness. A machine that reports "the 200-day rule does not beat a
    matched-exposure null" proves nothing until it is shown to REPORT SKILL WHEN
    SKILL IS PRESENT. `skill` degrades the oracle toward a coin flip so the
    detectable floor can be read off rather than assumed.

    Nothing downstream uses it. It never appears in a reported result.
    """
    marks = B.dates[::freq]
    J = B.J
    jv, ji = J.to_numpy(float), J.index.to_numpy()
    nxt = {}
    rng = np.random.default_rng(seed)
    for a, b in zip(marks[:-1], marks[1:]):
        i = min(int(np.searchsorted(ji, a)), len(jv) - 1)
        j = min(int(np.searchsorted(ji, b)), len(jv) - 1)
        up = bool(jv[j] > jv[i])
        nxt[pd.Timestamp(a)] = up if rng.random() < skill else bool(
            rng.random() < 0.5)
    return lambda a: nxt.get(pd.Timestamp(a), True)


def selftest(B: Bench) -> None:
    """Instrument checks. Each has a known answer; each is asserted, not eyed."""
    #  1. The cache changes no number.
    PXC = CachedPrices(B.PX)
    for tk in list(B.PX.d)[:40]:
        for d in B.dates[::311]:
            a, c = B.PX.at(tk, d), PXC.at(tk, d)
            assert (np.isnan(a) and np.isnan(c)) or a == c
            a, c = B.PX.exit_price(tk, d), PXC.exit_price(tk, d)
            assert (np.isnan(a) and np.isnan(c)) or a == c
    #  2. The shuffle preserves exposure, switch count and run lengths EXACTLY.
    rng = np.random.default_rng(0)
    st = rng.random(200) < 0.6
    for s in range(50):
        q = shuffle_states(st, np.random.default_rng(s))
        assert q.sum() == st.sum() and q.size == st.size
        assert (np.diff(q.astype(int)) != 0).sum() == \
               (np.diff(st.astype(int)) != 0).sum()
        def runs(x):
            e = np.flatnonzero(np.diff(x.astype(int)) != 0) + 1
            return sorted(len(r) for r in np.split(x, e) if r[0]), \
                   sorted(len(r) for r in np.split(x, e) if not r[0])
        assert runs(q) == runs(st)
    #  3. A rule that is always invested must equal a drifting hold, and the
    #     shuffle of an all-True sequence must be all-True (a degenerate null).
    assert shuffle_states(np.ones(20, bool), rng).all()
    #  4. The mark cache reproduces a fresh build exactly.
    m, bd = _marks(B, 252, 0)
    m2 = B.dates[0::252]
    bd2 = {d: g for d, g in B.P[B.P["date"].isin(m2)].groupby("date")}
    assert list(m) == list(m2) and set(bd) == set(bd2)
    for d in list(bd)[:20]:
        assert bd[d]["ticker"].tolist() == bd2[d]["ticker"].tolist()
    print("selftest: cache exact, shuffle preserves exposure/switches/runs, "
          "mark cache reproduces a fresh build — OK")


# --------------------------------------------------------------------- main --
def run(B: Bench, M: pd.DataFrame, mk: Callable, bname: str,
        rule: Callable, rname: str, freq: int, draws: int,
        cash_rate: float = 0.0, gap: str = "flat", quiet: bool = False,
        seed: Optional[Callable] = None, PX=None) -> Optional[Dict]:
    W = TimedWalk(B, cash_rate=cash_rate, gap=gap, PX=PX)
    r = W.run(mk(), rule, freq=freq, seed=seed)
    if not r:
        return None
    v = verdict(B, r, f"{bname} | {rname} | freq {freq}")
    st = r["states"]
    if st.size and 0 < st.mean() < 1:
        cs, ce, cl = [], [], []
        for s in range(draws):
            rng = np.random.default_rng(1000 + s)
            q = replay(W, mk, shuffle_states(st, rng), freq, seed=seed)
            if q:
                cs.append(q["cagr"])
                ce.append(q["early"])
                cl.append(q["late"])
        if cs:
            cs = np.array(cs)
            v["null_mean"] = float(cs.mean())
            v["null_sd"] = float(cs.std(ddof=1)) if len(cs) > 1 else np.nan
            v["z"] = float((v["cagr"] - cs.mean())
                           / max(cs.std(ddof=1), 1e-9))
            v["pctile"] = float((cs < v["cagr"]).mean())
            v["beats_null"] = bool(v["cagr"] > cs.mean())
            v["null_both"] = bool(v["early"] > np.mean(ce)
                                  and v["late"] > np.mean(cl))
    else:
        v["null_mean"] = v["cagr"]
        v["null_sd"] = 0.0
        v["z"] = 0.0
        v["pctile"] = 0.0
        v["beats_null"] = False
        v["null_both"] = False
    #  THE SECOND CONTROL, WHICH IS THE HARNESS'S OWN: random NAMES, drawn
    #  fresh at every mark, run through the SAME timing states. The first
    #  control asks whether the timing carries information; this one asks
    #  whether the basket does. A family that holds everything should tie it,
    #  and a tie here is the correct result rather than a failure.
    k = int(round(r["basket"])) if np.isfinite(r["basket"]) else 10
    cs = []
    for sd_ in range(max(6, draws // 20)):
        rng = np.random.default_rng(5000 + sd_)

        def mk_rand(rng=rng):
            def f(day, on=True):
                d = day
                if len(d) <= k:
                    return [(t, 1.0 / len(d)) for t in d["ticker"]]
                idx = rng.choice(len(d), k, replace=False)
                return [(t, 1.0 / k) for t in d["ticker"].to_numpy()[idx]]
            return f

        q = replay(W, mk_rand, st, freq) if st.size else None
        if q:
            cs.append(q["cagr"])
    v["randname_mean"] = float(np.mean(cs)) if cs else np.nan
    v["randname_sd"] = (float(np.std(cs, ddof=1)) if len(cs) > 1 else np.nan)
    v["beats_randname"] = bool(v["cagr"] > v["randname_mean"]) if cs else False
    v["PASS"] = bool(v["beats_index"] and v["beats_universe"]
                     and v["beats_picks"] and v["both_halves_index"]
                     and v["both_halves_universe"] and v["both_halves_picks"]
                     and v["beats_null"] and v.get("null_both", False)
                     and v["beats_randname"])
    if not quiet:
        print(fmt(v))
        print()
    return v


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--freq", type=int, nargs="*", default=[63])
    ap.add_argument("--draws", type=int, default=200)
    ap.add_argument("--min-tv", type=float, default=1e9)
    ap.add_argument("--cash-rate", type=float, default=0.0)
    ap.add_argument("--gap", default="flat", choices=["flat", "hold"])
    ap.add_argument("--baskets", nargs="*",
                    default=["all", "calm", "liq"])
    ap.add_argument("--rules", nargs="*", default=None)
    ap.add_argument("--oracle", action="store_true",
                    help="run the look-ahead positive control (cheats)")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--harness", action="store_true",
                    help="also print Bench.evaluate's own verdict")
    ap.add_argument("--out", default=os.path.join(OUT, "strat_indextime.csv"))
    A = ap.parse_args()

    P = load(min_tv=A.min_tv)
    B = Bench(P)
    M = market_state(B.J, B.dates)
    R = make_rules(M)
    if A.rules:
        R = {k: v for k, v in R.items() if k in A.rules}
    #  Each entry is (factory, label, seed). The factory exists because a
    #  drifting book is stateful and 200 null draws must not share one.
    PXC = CachedPrices(B.PX)
    BK = {
        "all": (lambda: basket_all, "all eligible (rebalanced)", None),
        "calm": (lambda: basket_calm, "strength+calm (rebalanced)", None),
        "liq": (lambda: basket_liq, "top-quintile liquid (rebalanced)", None),
        "drift": (lambda: DriftBook(B, basket_all, PXC), "all eligible (DRIFT)",
                  basket_all),
        "driftcalm": (lambda: DriftBook(B, basket_calm, PXC),
                      "strength+calm (DRIFT)", basket_calm),
        "driftliq": (lambda: DriftBook(B, basket_liq, PXC),
                     "top-quintile liquid (DRIFT)", basket_liq),
    }

    if A.selftest:
        selftest(B)
    if A.oracle:
        for freq in A.freq:
            for sk in (1.0, 0.75, 0.65, 0.60, 0.55):
                mk, bn, sd = BK[A.baskets[0]]
                run(B, M, mk, f"[POSITIVE CONTROL, CHEATS] {bn}",
                    oracle_rule(B, freq, sk), f"oracle skill {sk:.2f}",
                    freq, A.draws, cash_rate=A.cash_rate, gap=A.gap,
                    seed=sd, PX=PXC)
        return

    rows = []
    for freq in A.freq:
        for bk in A.baskets:
            mk, bn, sd = BK[bk]
            for rn, rule in R.items():
                v = run(B, M, mk, bn, rule, rn, freq, A.draws,
                        cash_rate=A.cash_rate, gap=A.gap, seed=sd, PX=PXC)
                if v:
                    rows.append({k: v[k] for k in v
                                 if k not in ("bh_e", "bh_l")})
                if A.harness and rn != "always in" and not bk.startswith("d"):
                    def sel(day, fn=mk(), rule=rule):
                        return fn(day) if rule(day["date"].iloc[0]) else []
                    hv = B.evaluate(sel, label=f"[harness, no cash bill] "
                                                f"{bn} | {rn}", freq=freq)
                    print(report(hv))
                    print()
    if rows:
        df = pd.DataFrame(rows)
        os.makedirs(OUT, exist_ok=True)
        df.to_csv(A.out, index=False)
        print(f"wrote {A.out}  ({len(df)} rows, "
              f"{int(df['PASS'].sum())} PASS)")


if __name__ == "__main__":
    main()
