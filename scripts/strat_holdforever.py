#!/usr/bin/env python3
"""H54 family — SELECT ONCE, HOLD FOREVER. Take turnover to zero, let weights
drift, and ask whether ANY selection rule survives the three buy-and-hold bars.

THE PREMISE is the repo's own: owning every eligible name and rebalancing
quarterly returns ~+1%/yr while owning the same names and never touching them
returns ~+10%/yr. Rebalancing sells the compounders. So: pick on a rare
schedule, never rebalance back to equal, let a winner grow into whatever share
of the book it earns.

HOW DRIFT IS EXPRESSED INSIDE A HARNESS THAT REBALANCES.
`Bench.walk` rebalances to whatever weights `select` returns, so drift is
produced by RETURNING THE DRIFTED WEIGHTS: the book is carried in rupiah-value
space, marked to market at each rebalance mark with the harness's own price
accessor, and handed back un-normalised. The harness normalises, so the period
return it computes is exactly the buy-and-hold return of the drifting book.

TWO PLACES THAT IS NOT EXACT. BOTH ARE MEASURED BELOW RATHER THAN ASSERTED.

  1. SPURIOUS TURNOVER — AGAINST the strategy. `walk` charges
     0.5*sum|w_t - w_{t-1}| * toll at every mark. Drift moves weights without
     any trade, so a pure hold-forever book is billed for dispersion it never
     traded. It shows up in `cost_yr` and is left in.

  2. COSTLESS REDEPLOYMENT OF DELISTING PROCEEDS — FOR the strategy, and it is
     the larger one. When a name stops printing, `walk` drops it and
     renormalises the remaining weights, which is arithmetically identical to
     recovering its last price in cash and spreading it over the survivors for
     free. The BUY-AND-HOLD benchmarks do NOT get that: `hold_basket` keeps the
     dead name's terminal return in the equal-weighted mean forever. That
     asymmetry runs one way, so `truth_walk` below re-runs the identical book
     with dead capital FROZEN — the pessimistic bound, no recovery at all — and
     both numbers are printed. The truth for a real holder sits between them,
     nearer the harness, because delisting proceeds usually are recoverable and
     redeploying a 5% slice costs one round trip on 5% of the book.

A STRUCTURAL RESULT WORTH READING BEFORE THE NUMBERS.
`BH_PICKS` is "buy the strategy's own first basket and hold it to the end". A
strategy that selects once at the first mark and never re-selects IS that
benchmark. It cannot beat it; it can only lose to it by the spurious turnover
above. So the literal family member is unfalsifiable-by-construction against
this harness, and the only testable version is "re-select rarely" — which turns
the question into "what is the LOWEST turnover that earns its keep".

AND PICK-ONCE IS n = 1 IN THE MOST LITERAL SENSE: one basket drawn on one date.
`offsets()` re-runs it from 12 different start dates to show the spread, because
a single-draw result quoted without its dispersion is the smallest-cell trap
this repo has already recorded three times.

WHAT THE RUN OF 2026-08-30 FOUND. In-sample throughout; the 24-month holdout
was already spent before this existed.

  * NOTHING IN THE FAMILY PASSES. Across 12 start dates, every near-zero-
    turnover configuration beats all three benchmarks 0/12 times at the
    harness's default liquidity floor and at most 1/12 at a loosened one.
    24 "never sell, only add" variants (`strat_holdforever_accum.py`) pass
    0 of 24 at offset 0.

  * PICK-ONCE IS `BH_PICKS`, VERIFIED NUMERICALLY. The drifting book's
    terminal multiple equals the equal-weighted hold of its own first basket
    to four decimals (5.9892 vs 5.9892 for liquid k=15 from 2007-01-24). So
    the benchmark and the strategy are the same object and the comparison can
    only ever be lost.

  * IT IS LOST TO A PROPERTY OF THE MEASURING DEVICE. `walk` skips any mark
    with fewer than MIN_UNIV eligible names and the book earns NOTHING across
    that whole period, while the benchmarks are continuous holds. 9 of 27
    marks at freq=252, including 2003-02-21 (IHSG +90.6% to the next mark),
    2006-02-02 (+45.2%), 2009-02-23 (+95.5%) and 2019-04-19 (-27.9%). The
    measured penalty is -1.78% to -6.73%/yr across 12 offsets. 2019-04-19 is
    GOOD FRIDAY: the panel carries 506 fill-forward rows on a market holiday
    with zero eligible names.

  * TWO CONFIGURATIONS DID TRIP THE PASS FLAG AT OFFSET 0 and neither is in
    this family: `mom12_1 re-pick 1yr` (+18.59%, 89% turnover) and
    `strength+calm re-pick 1yr` (+12.34%, 74%). Both are the MAXIMUM of their
    own 12-offset distribution (means +9.53% and +9.55%) and pass 2/12 and
    1/12 start dates.

  * THE FAMILY'S PREMISE IS FALSE HERE, as a controlled pair. Frozen name set,
    same marks, same costs, only the weighting policy varying: RESET-TO-EQUAL
    beat DRIFT in 18 of 18 cells, by 0.60 to 5.90 points a year.

  * AND THE "~9-POINT GAP" DECOMPOSES. Quarterly all-eligible re-selection
    plus reset-to-equal reproduces the handed-down +1.15%/yr exactly. The
    SAME rebalancing policy on a FROZEN name set returns +9.09%. The damage
    is NAME CHURN driven by the point-in-time eligibility filter, not the act
    of rebalancing weights. Those are two different things and only one of
    them is "turnover".

  * A COST-ACCOUNTING INVERSION WORTH KNOWING. `walk` charges turnover on the
    change in TARGET weights. A drifting book's target moves 3-15% a mark
    though it trades nothing; a constant-equal-weight target never moves
    though it trades the most. The harness therefore bills the zero-trade arm
    and gives the full-trade arm a free ride. Worth ~0.1-0.5%/yr here, which
    does not close gaps of 2-6 points but runs in the direction that flatters
    the reset arm.

Usage:  python3 scripts/strat_holdforever.py
"""

from __future__ import annotations

import os
import sys
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))
sys.path.insert(0, os.path.dirname(__file__))

from bhbench import MIN_BASKET, MIN_UNIV, Bench, load, report    # noqa: E402
from paint_suite import tick_of                                  # noqa: E402


# --------------------------------------------------------------- selection --
#  Every rule sees ONE bar and only that bar's columns. No full-sample
#  statistic defines any universe: the `median`/`quantile` calls below are
#  taken across the names present on that single day, which is information a
#  holder standing on that day has.

def rule_liquid(day: pd.DataFrame, k: int) -> pd.DataFrame:
    """Largest trailing turnover — the liquidity decile, a mega-cap proxy."""
    return day.nlargest(k, "tv60")


def rule_strength_calm(day: pd.DataFrame, k: int) -> pd.DataFrame:
    """H26's screen: near the 52-week high AND calm. Ranked inside the cell by
    liquidity so the basket is dealable rather than the thinnest survivors."""
    s = day[(day["hi52"] >= day["hi52"].quantile(0.90))
            & (day["vol60"] <= day["vol60"].median())]
    if len(s) < k:
        s = day[(day["hi52"] >= day["hi52"].quantile(0.70))
                & (day["vol60"] <= day["vol60"].median())]
    return s.nlargest(k, "tv60")


def rule_lowvol(day: pd.DataFrame, k: int) -> pd.DataFrame:
    return day.nsmallest(k, "vol60")


def rule_mom(day: pd.DataFrame, k: int) -> pd.DataFrame:
    return day.nlargest(k, "mom12_1")


def rule_all(day: pd.DataFrame, k: int) -> pd.DataFrame:
    """Everything eligible. The BH_UNIVERSE arm, re-selected on schedule."""
    return day


RULES: Dict[str, Callable] = {"liquid": rule_liquid,
                              "strength+calm": rule_strength_calm,
                              "lowvol": rule_lowvol,
                              "mom12_1": rule_mom,
                              "all-eligible": rule_all}


# ------------------------------------------------------------- the strategy --
class HoldForever:
    """Pick on a rare schedule, then carry the book in VALUE space and drift.

    `reselect_every` counts REBALANCE MARKS, so at freq=252 a value of 1
    re-picks yearly, 5 re-picks every five years, and anything larger than the
    number of marks never re-picks at all.

    A MISSING BAR IS NOT A DELISTING. The book marks to the last print on or
    before the date, and a name is declared dead only when its final print in
    the whole panel is behind that date. A first version froze a name on any
    missing bar and reported 12 temporary absences as deaths.
    """

    def __init__(self, bench: Bench, rule: Callable, k: int = 15,
                 reselect_every: int = 1, weight: str = "equal"):
        self.B = bench
        self.rule = rule
        self.k = k
        self.every = reselect_every
        self.weight = weight
        self.last_print = {t: v[0][-1] for t, v in bench.PX.d.items()}
        self.reset()

    def reset(self) -> None:
        self.hold: Dict[str, float] = {}     # ticker -> rupiah value, live
        self.px: Dict[str, float] = {}       # ticker -> price last marked at
        self.dead: float = 0.0               # value frozen in delisted names
        self.n_mark = 0
        self.n_pick = 0
        self.dead_share: List[float] = []

    def _pick(self, day: pd.DataFrame) -> Dict[str, float]:
        s = self.rule(day, self.k)
        s = s[np.isfinite(s["adj_close"]) & (s["adj_close"] > 0)]
        if len(s) < MIN_BASKET:
            return {}
        if self.weight == "turnover":
            w = s["tv60"].to_numpy(float)
            w = np.where(np.isfinite(w) & (w > 0), w, 0.0)
            if w.sum() <= 0:
                w = np.ones(len(s))
        else:
            w = np.ones(len(s))
        return dict(zip(s["ticker"], w / w.sum()))

    def mark(self, d) -> None:
        """Mark the book to market at date `d`; freeze anything delisted."""
        live, died = {}, 0.0
        for t, v in self.hold.items():
            p1 = self.B.PX.exit_price(t, d)      # last print on or before d
            p0 = self.px.get(t, np.nan)
            if not (np.isfinite(p1) and p1 > 0 and np.isfinite(p0) and p0 > 0):
                died += v
                continue
            v = v * p1 / p0
            if self.last_print.get(t) is not None and self.last_print[t] < d:
                died += v                        # its final print is behind us
            else:
                live[t] = v
                self.px[t] = p1
        self.hold, self.dead = live, self.dead + died
        tot = sum(live.values()) + self.dead
        self.dead_share.append(self.dead / tot if tot > 0 else 0.0)

    def __call__(self, day: pd.DataFrame) -> List[Tuple[str, float]]:
        d = day["date"].iloc[0]
        if self.hold:
            self.mark(d)
        take = (not self.hold) or (self.n_mark > 0
                                   and self.n_mark % self.every == 0)
        if take:
            new = self._pick(day)
            if new:
                cap = sum(self.hold.values()) or 1.0
                self.hold = {t: cap * w for t, w in new.items()}
                self.px = {t: float(self.B.PX.at(t, d)) for t in self.hold}
                self.n_pick += 1
        self.n_mark += 1
        #  Only names with a real print today can be traded/valued by the
        #  harness; a suspended name is carried in `self.hold` regardless.
        return [(t, v) for t, v in self.hold.items()
                if v > 0 and np.isfinite(self.B.PX.at(t, d))]


# --------------------------------------------------- the leak-free simulator --
def truth_walk(B: Bench, make: Callable[[], HoldForever],
               freq: int = 252, offset: int = 0) -> Dict:
    """`Bench.walk` with delisting proceeds FROZEN instead of redeployed.

    Same marks, same eligibility gate, same cost model, same weights. The only
    difference is that dead capital stays in the denominator earning nothing,
    which is the pessimistic bound on the one bias that favours the strategy.
    Returns both the frozen path and the redeployed path so the gap is visible.
    """
    marks = B.dates[offset::freq]
    by_date = {d: g for d, g in B.P[B.P["date"].isin(marks)].groupby("date")}
    S = make()
    eq_frozen, eq_redeploy = 1.0, 1.0
    prev: Dict[str, float] = {}
    t0, last = None, None
    for a, b in zip(marks[:-1], marks[1:]):
        day = by_date.get(a)
        if day is None:
            continue
        day = day[day["elig"]]
        if len(day) < MIN_UNIV:
            continue
        picks = S(day.copy())
        if len(picks) < MIN_BASKET:
            continue
        tot = sum(w for _, w in picks)
        cur = {t: w / tot for t, w in picks}
        live_share = tot / (tot + S.dead) if (tot + S.dead) > 0 else 1.0
        if t0 is None:
            t0 = a
        last = b
        keys = set(cur) | set(prev)
        turn = 0.5 * sum(abs(cur.get(k, 0.0) - prev.get(k, 0.0)) for k in keys)
        px_a = {t: B.PX.at(t, a) for t in cur}
        toll = [B.fee + B.spread_mult * tick_of(v) / v
                for v in px_a.values() if np.isfinite(v) and v > 0]
        toll = float(np.mean(toll)) if toll else B.fee
        eq_frozen *= (1.0 - turn * toll)
        eq_redeploy *= (1.0 - turn * toll)
        rets, ws = [], []
        for t, w in cur.items():
            p0, p1 = px_a[t], B.PX.exit_price(t, b)
            if np.isfinite(p0) and p0 > 0 and np.isfinite(p1) and p1 > 0:
                rets.append(p1 / p0 - 1.0)
                ws.append(w)
        if rets:
            ws = np.array(ws) / np.sum(ws)
            r = float(np.dot(ws, rets))
            eq_redeploy *= (1.0 + r)
            eq_frozen *= (1.0 + r * live_share)   # dead slice earns nothing
        prev = cur
    if t0 is None or last is None:
        return {}
    yrs = (last - t0).astype("timedelta64[D]").astype(float) / 365.25
    f = lambda e: float(max(e, 1e-9) ** (1 / max(yrs, 1e-9)) - 1)   # noqa: E731
    return {"frozen": f(eq_frozen), "redeploy": f(eq_redeploy),
            "dead_peak": max(S.dead_share) if S.dead_share else 0.0,
            "years": yrs}


# ----------------------------------------------- start-date sensitivity --
def verdict_at(B: Bench, make: Callable[[], HoldForever], freq: int,
               offset: int, draws: int = 6) -> Dict:
    """`Bench.evaluate`'s verdict, computed at an arbitrary start offset.

    `evaluate` hard-codes offset 0. This reuses its own methods -- `walk`,
    `index_cagr`, `hold_basket`, `half_cagr` -- so the PASS rule is identical;
    only the mark grid moves. Nothing about the criterion is relaxed.
    """
    r = B.walk(make(), freq=freq, offset=offset)
    if not r:
        return {}
    a0, b1 = r["start"], r["end"]
    mid = r["curve"][len(r["curve"]) // 2][0]
    uni0 = B.P[(B.P["date"] == a0) & B.P["elig"]]["ticker"].tolist()
    fb = r["first_basket"]
    full = {"index": B.index_cagr(a0, b1),
            "universe": B.hold_basket(uni0, a0, b1),
            "picks": B.hold_basket(fb, a0, b1)}
    early = {"index": B.index_cagr(a0, mid),
             "universe": B.hold_basket(uni0, a0, mid),
             "picks": B.hold_basket(fb, a0, mid)}
    late = {"index": B.index_cagr(mid, b1),
            "universe": B.hold_basket(uni0, mid, b1),
            "picks": B.hold_basket(fb, mid, b1)}
    ctl = [B.walk(make(), freq=freq, offset=offset,
                  rng=np.random.default_rng(s)) for s in range(draws)]
    ctl = [c for c in ctl if c]
    rand = float(np.mean([c["cagr"] for c in ctl])) if ctl else np.nan
    ok = all(r["cagr"] > full[kk] for kk in full) and \
        all(r["early"] > early[kk] and r["late"] > late[kk] for kk in early) \
        and r["cagr"] > rand
    return {"cagr": r["cagr"], "rand": rand, "PASS": bool(ok),
            "start": a0, **{f"bh_{kk}": full[kk] for kk in full},
            "beats_all": all(r["cagr"] > full[kk] for kk in full)}


def offsets(B: Bench, rule: Callable, k: int, freq: int, every: int,
            n: int = 12, weight: str = "equal") -> List[Dict]:
    """The same rule started on `n` different dates. Pick-once is ONE draw of
    ONE basket; without this its headline has no dispersion attached to it,
    and a single-cell headline is the trap this repo has recorded three
    times."""
    out = []
    for off in np.linspace(0, freq - 1, n).astype(int):
        v = verdict_at(B, lambda: HoldForever(B, rule, k=k,
                                              reselect_every=every,
                                              weight=weight),
                       freq, int(off))
        if v:
            out.append(v)
    return out


class Accumulate(HoldForever):
    """NEVER SELL, ONLY ADD — the one member of this family that can differ
    from `BH_PICKS` at all.

    Pick-once IS `BH_PICKS` by definition, so it cannot beat it; the only way
    to stay in the family and still diverge is to keep the existing names
    forever and fund a small new position out of a pro-rata trim of the book.
    Adding one name at weight `add_w` costs `add_w` of measured turnover, so
    at add_w=3% and yearly marks the toll is ~0.03%/yr -- an order of magnitude
    below anything else tested here.
    """

    def __init__(self, *a, add_w: float = 0.03, n_add: int = 1, **kw):
        self.add_w = add_w
        self.n_add = n_add
        super().__init__(*a, **kw)

    def __call__(self, day: pd.DataFrame) -> List[Tuple[str, float]]:
        d = day["date"].iloc[0]
        if self.hold:
            self.mark(d)
        if not self.hold:
            new = self._pick(day)
            if new:
                self.hold = {t: w for t, w in new.items()}
                self.px = {t: float(self.B.PX.at(t, d)) for t in self.hold}
                self.n_pick += 1
        else:
            ranked = self.rule(day, self.k + len(self.hold) + self.n_add)
            cand = [t for t in ranked["ticker"] if t not in self.hold][
                :self.n_add]
            cand = [t for t in cand if np.isfinite(self.B.PX.at(t, d))
                    and self.B.PX.at(t, d) > 0]
            if cand:
                cap = sum(self.hold.values())
                take = self.add_w * len(cand)
                self.hold = {t: v * (1.0 - take)
                             for t, v in self.hold.items()}
                for t in cand:
                    self.hold[t] = cap * self.add_w
                    self.px[t] = float(self.B.PX.at(t, d))
                self.n_pick += 1
        self.n_mark += 1
        return [(t, v) for t, v in self.hold.items()
                if v > 0 and np.isfinite(self.B.PX.at(t, d))]


class RebalanceEW(HoldForever):
    """Identical selection and schedule, but weights are reset to equal at
    EVERY mark. The only difference from `HoldForever` is the weighting
    policy, which is what isolates 'does letting winners run earn its keep'
    from 'is the selection any good'. Everything else -- names, dates, cost
    model, delisting treatment -- is held fixed."""

    def __call__(self, day: pd.DataFrame) -> List[Tuple[str, float]]:
        picks = super().__call__(day)
        if not picks:
            return picks
        d = day["date"].iloc[0]
        #  Reset to equal, and put the book back on the drifted total so the
        #  two arms are compounding the same capital. A name held but not
        #  printing today keeps its value untouched rather than being deleted
        #  from the book -- a suspension is not a sale.
        live = {t for t, _ in picks}
        cap = sum(v for _, v in picks)
        w = cap / len(picks)
        for t in live:
            self.hold[t] = w
            self.px[t] = float(self.B.PX.at(t, d))
        return [(t, self.hold[t]) for t in live]


def drift_vs_rebalance(B: Bench, freqs: Sequence[int] = (21, 63, 252)) -> None:
    """THE FAMILY'S PREMISE, tested as a controlled pair rather than quoted.

    Same names, same marks, same costs; only DRIFT vs RESET-TO-EQUAL differs.
    The turnover column is the mechanism: reset-to-equal trades every period
    to undo the drift, and pays for it.

    THE NAME SET MUST BE FROZEN (`reselect_every` = never) FOR THIS TO BE A
    CONTROLLED PAIR. A first version passed `reselect_every=1`, which makes
    `HoldForever` re-pick and reset to equal at every mark -- i.e. it IS the
    rebalanced arm -- and the table dutifully printed a gap of exactly +0.00%
    in all nine cells. A difference of exactly zero between two arms that are
    supposed to differ is a bug, not a finding.
    """
    print(f"  {'selection':<16}{'freq':>5}{'drift':>10}{'reset-EW':>10}"
          f"{'gap':>9}{'turn drift':>12}{'turn EW':>9}{'cost EW':>9}")
    for nm, rl in (("all-eligible", rule_all), ("liquid k=15", rule_liquid),
                   ("lowvol k=15", rule_lowvol)):
        for f in freqs:
            a = B.walk(HoldForever(B, rl, k=15, reselect_every=10 ** 9),
                       freq=f)
            b = B.walk(RebalanceEW(B, rl, k=15, reselect_every=10 ** 9),
                       freq=f)
            if not a or not b:
                continue
            print(f"  {nm:<16}{f:>5}{a['cagr']:>+10.2%}{b['cagr']:>+10.2%}"
                  f"{a['cagr'] - b['cagr']:>+9.2%}{a['turnover']:>12.0%}"
                  f"{b['turnover']:>9.0%}{b['cost_yr']:>9.2%}")


def skip_report(B: Bench, freq: int) -> None:
    """THE HARNESS ARTEFACT THAT DOMINATES THIS FAMILY, measured not asserted.

    `walk` skips any mark where the eligible universe is under MIN_UNIV and
    the book earns NOTHING over that whole period, while every buy-and-hold
    benchmark is one continuous hold that earns it. At freq=252 a single skip
    costs a YEAR of return. That is not a property of any strategy; it is a
    property of the measuring device, and it falls only on the walked arms.
    """
    e = B.P[B.P["elig"]].groupby("date")["ticker"].size()
    marks = B.dates[0::freq]
    bad = [a for a in marks if int(e.get(pd.Timestamp(a), 0)) < MIN_UNIV]
    print(f"harness skips at freq={freq}: {len(bad)} of {len(marks)} marks "
          f"have under {MIN_UNIV} eligible names.")
    for a in bad[-4:]:
        j = list(marks).index(a)
        if j + 1 >= len(marks):
            continue
        b = marks[j + 1]
        s = B.J[(B.J.index >= pd.Timestamp(a)) & (B.J.index <= pd.Timestamp(b))]
        if len(s) < 5:
            continue
        print(f"    skipped {str(a)[:10]} -> {str(b)[:10]}: the strategy sits "
              f"out an IHSG move of {float(s.iloc[-1] / s.iloc[0]) - 1:+.1%} "
              f"(eligible that day: {int(e.get(pd.Timestamp(a), 0))})")


# ---------------------------------------------------------------------- run --
def main() -> None:
    P = load()
    B = Bench(P)
    print(f"panel {P['ticker'].nunique()} names, {len(B.dates)} sessions, "
          f"{str(B.dates[0])[:10]} -> {str(B.dates[-1])[:10]}")
    print(f"cost: fee {B.fee:.2%} round trip + {B.spread_mult} x fraksi tick")
    print("THE 24-MONTH HOLDOUT IS ALREADY SPENT. EVERYTHING BELOW IS "
          "IN-SAMPLE.\n")
    skip_report(B, 252)
    print()

    NEVER = 999
    variants: List[Tuple] = []
    #  A. the literal family member: pick once at the first mark, never again.
    for nm, rl in RULES.items():
        variants.append((f"{nm} k=15 pick-once", rl, 15, 252, NEVER, "equal"))
    #  B. re-select on a rare schedule, drift in between.
    for nm, rl in RULES.items():
        for every, tag in ((5, "5yr"), (2, "2yr"), (1, "1yr")):
            variants.append((f"{nm} k=15 re-pick {tag}", rl, 15, 252, every,
                             "equal"))
    #  C. basket size, and a turnover-weighted START (the cap-weight proxy:
    #     A19 measured an equal-weighted IDX basket structurally trailing the
    #     cap-weighted index, so starting nearer cap weight is the obvious fix).
    for kk in (10, 30, 60):
        variants.append((f"liquid k={kk} pick-once", rule_liquid, kk, 252,
                         NEVER, "equal"))
    variants.append(("liquid k=15 pick-once tv-weighted", rule_liquid, 15, 252,
                     NEVER, "turnover"))
    variants.append(("liquid k=30 pick-once tv-weighted", rule_liquid, 30, 252,
                     NEVER, "turnover"))
    variants.append(("all-eligible pick-once tv-weighted", rule_all, 15, 252,
                     NEVER, "turnover"))
    variants.append(("strength+calm k=30 pick-once", rule_strength_calm, 30,
                     252, NEVER, "equal"))
    #  D. the coarser clock: 504-session marks (~2yr). 1260 is unavailable —
    #     it yields 6 marks and `walk` requires 8.
    for nm, rl in (("liquid", rule_liquid),
                   ("strength+calm", rule_strength_calm),
                   ("lowvol", rule_lowvol)):
        variants.append((f"{nm} k=15 pick-once freq=504", rl, 15, 504, NEVER,
                         "equal"))
        variants.append((f"{nm} k=15 re-pick 2yr freq=504", rl, 15, 504, 1,
                         "equal"))

    rows = []
    for label, rl, k, freq, every, wt in variants:
        S = HoldForever(B, rl, k=k, reselect_every=every, weight=wt)
        v = B.evaluate(S, label=label, freq=freq)
        print(report(v))
        if v.get("ok"):
            t = truth_walk(B, lambda: HoldForever(B, rl, k=k,
                                                  reselect_every=every,
                                                  weight=wt), freq=freq)
            v.update({"picks_taken": S.n_pick, "rule": label,
                      "frozen": t.get("frozen", np.nan),
                      "dead_peak": t.get("dead_peak", np.nan)})
            print(f"  [selections {S.n_pick} | delisting proceeds FROZEN "
                  f"instead of redeployed: {t.get('frozen', float('nan')):+.2%}"
                  f"/yr vs {t.get('redeploy', float('nan')):+.2%} — peak dead "
                  f"share {t.get('dead_peak', float('nan')):.1%}]")
            rows.append(v)
        print()

    print("=" * 78)
    npass = sum(1 for r in rows if r["PASS"])
    print(f"{len(variants)} variants run, {npass} passed.")
    if rows:
        best = max(rows, key=lambda r: r["cagr"])
        print(f"best by CAGR: {best['label']}  {best['cagr']:+.2%}")
        pd.DataFrame([{kk: r.get(kk) for kk in
                       ("label", "cagr", "frozen", "bh_index", "bh_universe",
                        "bh_picks", "random", "early", "late", "turnover",
                        "cost_yr", "dead_peak", "picks_taken", "PASS")}
                      for r in rows]).to_csv(
            "reports/strat_holdforever.csv", index=False)
        print("wrote reports/strat_holdforever.csv")

    # ------------------------------------------------- the liquidity floor --
    print("\n" + "=" * 78)
    print("LIQUIDITY-FLOOR SENSITIVITY — min_tv is a knob and it moves "
          "everything.")
    print("A lower floor enlarges the universe, so fewer marks are skipped "
          "AND the\nwindow starts earlier. Both benchmarks are recomputed on "
          "the same universe.\n")
    P2 = load(min_tv=3e8)
    B2 = Bench(P2)
    skip_report(B2, 252)
    print()
    rows2 = []
    for nm, rl in RULES.items():
        for every, tag in ((NEVER, "pick-once"), (1, "re-pick 1yr")):
            S = HoldForever(B2, rl, k=15, reselect_every=every)
            v = B2.evaluate(S, label=f"{nm} k=15 {tag} [min_tv=3e8]",
                            freq=252)
            print(report(v))
            print()
            if v.get("ok"):
                rows2.append(v)
    print(f"{len(rows2)} variants at min_tv=3e8, "
          f"{sum(1 for r in rows2 if r['PASS'])} passed.")

    # ---------------------------------------------------------- the arbiter --
    print("\n" + "=" * 78)
    print("THE ARBITER — the SAME verdict rule at 12 different start dates.")
    print("Offset 0 is one draw. A configuration that passes at offset 0 and")
    print("nowhere else is the maximum of a sweep, not a result.\n")
    print(f"  {'configuration':<40} {'CAGR mean':>10} {'sd':>7} {'min':>8} "
          f"{'max':>8}  {'beats all 3':>11} {'PASS':>7}")
    arb = []
    for lab, Bx, mtv in (("min_tv=1e9", B, 1e9), ("min_tv=3e8", B2, 3e8)):
        for nm, rl in RULES.items():
            for every, tag in ((NEVER, "pick-once"), (5, "re-pick 5yr"),
                               (1, "re-pick 1yr")):
                o = offsets(Bx, rl, 15, 252, every)
                if not o:
                    continue
                c = np.array([x["cagr"] for x in o])
                nb = sum(1 for x in o if x["beats_all"])
                npz = sum(1 for x in o if x["PASS"])
                name = f"{nm} {tag} [{lab}]"
                print(f"  {name:<40} {c.mean():>+10.2%} "
                      f"{c.std(ddof=1):>7.2%} {c.min():>+8.2%} "
                      f"{c.max():>+8.2%}  {nb:>6}/{len(o):<4} {npz:>4}/{len(o)}")
                arb.append({"config": name, "mean": c.mean(),
                            "sd": c.std(ddof=1), "min": c.min(), "max": c.max(),
                            "beats_all_n": nb, "pass_n": npz, "n": len(o)})
    # ---------------------------------------------------- the premise itself --
    print("\n" + "=" * 78)
    print("THE FAMILY'S PREMISE, as a controlled pair: same names, same "
          "marks, same\ncosts; only DRIFT vs RESET-TO-EQUAL-WEIGHT differs.\n")
    print(" min_tv=1e9")
    drift_vs_rebalance(B)
    print("\n min_tv=3e8")
    drift_vs_rebalance(B2)

    pd.DataFrame(arb).to_csv("reports/strat_holdforever_offsets.csv",
                             index=False)
    print("\nwrote reports/strat_holdforever_offsets.csv")
    tot = len(variants) + len(rows2)
    print(f"\nVARIANTS EVALUATED: {tot} at offset 0, plus "
          f"{sum(a['n'] for a in arb)} start-date re-runs = "
          f"{tot + sum(a['n'] for a in arb)} configuration-runs in total.")


if __name__ == "__main__":
    main()
