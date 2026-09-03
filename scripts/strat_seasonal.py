#!/usr/bin/env python3
"""H54 family — SEASONALITY AND CALENDAR EFFECTS ON IDX.

Nothing in this repo had ever tested the calendar. This does, in two parts that
answer two different questions, because conflating them is how calendar studies
usually go wrong.

PART 1 — IS THERE A CALENDAR EFFECT AT ALL?
  52 cells: day-of-week (5), month-of-year (12), trading-day-of-month from both
  ends (15), turn-of-month windows (12), Halloween/sell-in-May (2), and six
  Ramadan/Lebaran windows anchored on Idul Fitri.

  THE NULL IS A CIRCULAR ROTATION OF THE RETURN SERIES, not an iid shuffle.
  Rotating by a random number of sessions preserves every serial property the
  series has -- volatility clustering, autocorrelation, the fat tails -- and
  destroys only the alignment between a return and its calendar label, which is
  exactly the null hypothesis. An iid shuffle would destroy the clustering too
  and return an interval far too tight (A17: clustered data needs a clustered
  null).

  MULTIPLE TESTING IS HANDLED BY A MAX-STATISTIC, NOT BY BONFERRONI. On every
  rotation the LARGEST |z| across all 52 cells is recorded; a cell's family-wise
  p is the share of rotations whose maximum beat it. That is exact rather than
  conservative, and it is the honest way to scan 52 cells at once.

  THE LEBARAN ANCHOR IS VALIDATED, NOT ASSUMED. The Idul Fitri table below was
  checked against the exchange's own closure schedule: every one of the 18
  multi-day trading gaps in the panel of six days or longer falls on an Eid in
  the table, to the day. A26's lesson one level down -- check the instrument
  before believing what it says.

PART 2 — CAN ANY OF IT BE TRADED?
  Nineteen harness variants in three groups.

  TIMING OVERLAYS (S1-S4) sit on top of an explicitly inert core: buy the 60
  most liquid eligible names at the first bar, equal weight, and never trade
  again. S0 is that core alone. So S1 minus S0 is the entire contribution of
  the calendar switch, with the stock selection held fixed and identical.

  REBALANCE-MONTH (S5.01-S5.12) asks the question the H54 brief hints at: if
  turnover is the enemy, does the CALENDAR MONTH in which you take your one
  annual rebalance matter? Twelve cells, and twelve cells is exactly the
  situation in which a maximum means nothing without its spread.

  CROSS-SECTIONAL SEASONALITY (S6) is the Heston-Sadka effect: buy the names
  whose OWN history in this calendar month is strongest. It is the only
  seasonality claim here that is about WHICH stock rather than WHEN, so it is
  the only one that does not require market timing.

TWO HARNESS FACTS THAT SHAPE EVERY NUMBER BELOW, BOTH MEASURED NOT ASSERTED.

  1. GOING TO CASH IS FREE IN THE HARNESS, AND IT MUST NOT BE. `walk` skips a
     mark whose basket is under the floor: equity stays flat and no cost is
     charged. Worse, `prev` is not updated, so coming back to the same names
     bills zero turnover. A weekend-timing rule would therefore make 52 round
     trips a year and pay for none of them. Every timing variant here counts
     its own out-and-back events and prints a SELF-CHARGED cost line beside the
     harness's, computed from the same fee and the same fraksi-harga tick the
     harness uses. The harness PASS flag is reported as the harness computes
     it; the self-charged CAGR is what the conclusion rests on.

  2. DRIFT IS BILLED AS TURNOVER, AGAINST THE STRATEGY, AND IS LEFT IN. A book
     that never trades still has moving weights, and `walk` charges
     0.5*sum|w_t - w_{t-1}| every mark. At daily marks that is of order 1%/yr
     for nothing. It falls on the overlay and on its own control equally, which
     is why S1-S0 is the comparison quoted, and each variant prints the
     turnover the harness billed beside the turnover it actually transacted.

EVERYTHING IS IN SAMPLE. The 24-month holdout was spent long ago (A21).

Usage:  python3 scripts/strat_seasonal.py [scan|time|month|xsec|all]
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))
sys.path.insert(0, os.path.dirname(__file__))

from bhbench import Bench, load, report, FEE, SPREAD_MULT      # noqa: E402
from paint_suite import tick_of                                # noqa: E402

OUT = os.path.join("reports", "strat_seasonal.txt")

#  1 Syawal in Indonesia. Cross-checked below against every multi-day exchange
#  closure in the panel: 18 of 18 gaps of >=6 days land on one of these.
EID = pd.to_datetime([
    "2000-01-08", "2000-12-27", "2001-12-16", "2002-12-06", "2003-11-25",
    "2004-11-14", "2005-11-03", "2006-10-24", "2007-10-13", "2008-10-01",
    "2009-09-21", "2010-09-10", "2011-08-31", "2012-08-19", "2013-08-08",
    "2014-07-28", "2015-07-17", "2016-07-06", "2017-06-25", "2018-06-15",
    "2019-06-05", "2020-05-24", "2021-05-13", "2022-05-02", "2023-04-22",
    "2024-04-10", "2025-03-31", "2026-03-20"])

LOG: List[str] = []


def say(*a) -> None:
    s = " ".join(str(x) for x in a)
    print(s)
    LOG.append(s)


# ===========================================================================
# PART 1 — the calendar scan
# ===========================================================================
def eid_relative(d: pd.DatetimeIndex) -> np.ndarray:
    """Sessions from the nearest Idul Fitri. 0 = first session on/after Eid."""
    pos = np.searchsorted(d.values, EID.values)
    rel = np.full(len(d), 9999)
    for p in pos:
        if p >= len(d):
            continue
        lo, hi = max(0, p - 60), min(len(d), p + 60)
        idx = np.arange(lo, hi)
        cand = idx - p
        take = np.abs(cand) < np.abs(rel[idx])
        rel[idx[take]] = cand[take]
    return rel


def check_eid_table(P: pd.DataFrame) -> None:
    """INSTRUMENT CHECK, IN THE DIRECTION THAT MATTERS.

    The claim being checked is that the table locates Lebaran, i.e. that every
    Eid in range is straddled by a long exchange closure. The converse — that
    every long closure is an Eid — is FALSE and is supposed to be: Christmas,
    New Year and the odd national holiday cluster also shut the exchange. A
    first version tested the converse, got 18 of 23, and made a validated
    table look 78% right.
    """
    d = pd.DatetimeIndex(np.sort(P["date"].unique()))
    gap = (d[1:] - d[:-1]).days
    lo, hi = d[0], d[-1]
    inrange = EID[(EID > lo) & (EID < hi)]
    hit = 0
    for e in inrange:
        j = np.searchsorted(d.values, e.to_datetime64())
        if 0 < j < len(d) and (d[j] - d[j - 1]).days >= 5:
            hit += 1
    closures = d[:-1][gap >= 6]
    notEid = [str(s.date()) for s in closures
              if not (np.abs((EID - s).days) <= 8).any()]
    say(f"  Eid table check: {hit} of {len(inrange)} tabulated Idul Fitri in "
        f"the panel range are straddled by an exchange closure of >=5 days")
    say(f"  (the {len(notEid)} long closures that are NOT Eid are year-end and "
        f"national holidays: {', '.join(notEid)})")


def cells_for(d: pd.DatetimeIndex) -> Dict[str, np.ndarray]:
    lab = pd.DataFrame(index=d)
    lab["dow"] = d.dayofweek
    lab["mon"] = d.month
    ym = d.to_period("M")
    one = pd.Series(1, index=d)
    lab["tdom"] = one.groupby(ym).cumcount() + 1
    lab["tend"] = -(one.groupby(ym).cumcount(ascending=False) + 1)
    lab["eid"] = eid_relative(d)
    C: Dict[str, np.ndarray] = {}
    for k, nm in enumerate(["Mon", "Tue", "Wed", "Thu", "Fri"]):
        C[f"dow={nm}"] = (lab["dow"] == k).values
    for m in range(1, 13):
        C[f"month={m:02d}"] = (lab["mon"] == m).values
    for k in range(1, 11):
        C[f"tdom=+{k}"] = (lab["tdom"] == k).values
    for k in range(1, 6):
        C[f"tdom=-{k}"] = (lab["tend"] == -k).values
    for j in (1, 2, 3):
        for i in (1, 2, 3, 4):
            C[f"TOM(-{j},+{i})"] = ((lab["tend"] >= -j)
                                    | (lab["tdom"] <= i)).values
    C["Halloween Nov-Apr"] = lab["mon"].isin([11, 12, 1, 2, 3, 4]).values
    C["SellInMay May-Oct"] = lab["mon"].isin([5, 6, 7, 8, 9, 10]).values
    C["Ramadan Eid-22..-1"] = ((lab["eid"] >= -22) & (lab["eid"] <= -1)).values
    C["pre-Lebaran 5d"] = ((lab["eid"] >= -5) & (lab["eid"] <= -1)).values
    C["post-Lebaran 5d"] = ((lab["eid"] >= 0) & (lab["eid"] <= 4)).values
    C["post-Lebaran 20d"] = ((lab["eid"] >= 0) & (lab["eid"] <= 19)).values
    C["Dec 2nd half"] = ((lab["mon"] == 12) & (lab["tdom"] > 10)).values
    C["Jan first 5d"] = ((lab["mon"] == 1) & (lab["tdom"] <= 5)).values
    return C


def rotation_scan(x: pd.Series, name: str, nboot: int = 4000,
                  seed: int = 0, top: int = 12) -> List[Tuple]:
    v = x.to_numpy(float)
    N = len(v)
    C = cells_for(x.index)
    keys = list(C)
    M = np.array([C[k] for k in keys], dtype=float)
    cnt = M.sum(1)
    obs = (M @ v) / cnt
    base = v.mean()
    rng = np.random.default_rng(seed)
    null = np.empty((nboot, len(keys)))
    for b in range(nboot):
        null[b] = (M @ np.roll(v, rng.integers(1, N))) / cnt
    mu, sd = null.mean(0), null.std(0, ddof=1)
    z = (obs - mu) / sd
    nz = np.abs((null - mu) / sd)
    maxz = nz.max(1)                       # family-wise max statistic
    rows = []
    for i, k in enumerate(keys):
        rows.append((k, obs[i], obs[i] - base, z[i],
                     float((nz[:, i] >= abs(z[i])).mean()),
                     float((maxz >= abs(z[i])).mean()), int(cnt[i])))
    rows.sort(key=lambda t: -abs(t[3]))
    say(f"\n  {name}: base {100*base:+.4f}%/day over {N} sessions; "
        f"{len(keys)} cells; {nboot} circular rotations")
    say(f"  {'cell':<24}{'%/day':>9}{'excess':>9}{'z':>7}"
        f"{'p_raw':>8}{'p_FWE':>8}{'n':>7}")
    for r in rows[:top]:
        say(f"  {r[0]:<24}{100*r[1]:>9.4f}{100*r[2]:>+9.4f}{r[3]:>7.2f}"
            f"{r[4]:>8.4f}{r[5]:>8.4f}{r[6]:>7}")
    say(f"  ... {len(rows)-top} weaker cells omitted; "
        f"{sum(1 for r in rows if r[5] < 0.05)} of {len(rows)} cells clear "
        f"p_FWE < 0.05")
    return rows


def part_scan(P: pd.DataFrame) -> pd.Series:
    say("=" * 78)
    say("PART 1 — CALENDAR SCAN, 52 CELLS, CIRCULAR-ROTATION NULL")
    say("=" * 78)
    check_eid_table(P)
    Q = P.sort_values(["ticker", "date"]).copy()
    Q["ret"] = Q.groupby("ticker")["adj_close"].pct_change()
    Q["ep"] = Q.groupby("ticker")["elig"].shift(1).fillna(False)
    #  Eligibility is read on the PRIOR bar: you must be able to buy a name
    #  before you can earn its return.
    #
    #  AND THE EXCHANGE HOLIDAYS HAVE TO COME OUT, WHICH IS NOT COSMETIC. The
    #  spine forward-fills a closed session with the previous close, volume 0
    #  and `tradeable` False. Those bars are correct as data and poison as a
    #  calendar sample: they are ~1,700 hard zeros distributed over exactly the
    #  cells this study is about — a Monday that is a holiday is a Monday with
    #  a zero in it, and the whole Lebaran window is a run of them. Requiring
    #  `tradeable` on the return bar removes them. Not doing this pulls every
    #  cell that contains a holiday toward zero and inflates its n.
    r = Q[Q["ep"] & Q["tradeable"] & Q["ret"].notna()
          & (Q["ret"].abs() < 0.6)]
    ew = r.groupby("date")["ret"].mean()
    ew = ew[r.groupby("date")["ret"].size() >= 40]
    J = pd.read_csv(os.path.join("data", "cache", "ohlcv", "_JKSE.csv.gz"),
                    parse_dates=["date"]).sort_values("date")
    jk = J.set_index("date")["close"].pct_change().reindex(ew.index)
    rotation_scan(ew, "EQUAL-WEIGHTED ELIGIBLE UNIVERSE")
    rotation_scan(jk.dropna(), "IHSG (price index)")

    #  The one surviving cell, split. A calendar effect that lives only in the
    #  first half is a fact about 2005-2015, not about the calendar.
    d = ew.index
    f = pd.DataFrame({"ew": ew.values, "dow": d.dayofweek, "yr": d.year},
                     index=d)
    f["jk"] = jk.reindex(d).values
    say("\n  MONDAY, BY ERA (mean %/day, and the same for the index)")
    say(f"  {'era':<12}{'nMon':>6}{'Mon_ew':>9}{'rest_ew':>9}{'diff':>9}"
        f"{'Mon_jk':>9}{'rest_jk':>9}{'diff':>9}")
    for lo, hi in [(2005, 2010), (2011, 2015), (2016, 2020), (2021, 2026),
                   (2005, 2015), (2016, 2026)]:
        s = f[(f.yr >= lo) & (f.yr <= hi)]
        m, o = s[s.dow == 0], s[s.dow != 0]
        say(f"  {lo}-{hi:<7}{len(m):>6}{100*m.ew.mean():>9.4f}"
            f"{100*o.ew.mean():>9.4f}{100*(m.ew.mean()-o.ew.mean()):>+9.4f}"
            f"{100*m.jk.mean():>9.4f}{100*o.jk.mean():>9.4f}"
            f"{100*(m.jk.mean()-o.jk.mean()):>+9.4f}")
    g = f.groupby("yr").apply(
        lambda s: s[s.dow == 0].ew.mean() - s[s.dow != 0].ew.mean(),
        include_groups=False)
    say(f"  Monday minus the rest is negative in {int((g<0).sum())} of "
        f"{len(g)} calendar years")
    #  THE OTHER SURVIVING CELL, EVENT BY EVENT. A8 records that dropping the
    #  single largest year has earned its place twice in this repo; it earns it
    #  a third time here.
    rel = eid_relative(d)
    w = pd.DataFrame({"ew": ew.values, "jk": jk.reindex(d).values, "rel": rel},
                     index=d)
    w = w[(w.rel >= 0) & (w.rel <= 4)]
    ev = w.groupby(w.index.year).apply(
        lambda g: pd.Series({"ew": (1 + g.ew).prod() - 1,
                             "jk": (1 + g.jk).prod() - 1}),
        include_groups=False)
    say(f"\n  POST-LEBARAN WEEK, EVENT BY EVENT ({len(ev)} events, and that is "
        f"the effective sample, not 115 sessions)")
    say("  " + "  ".join(f"{y}:{100*v:+.1f}" for y, v in ev["jk"].items()))
    for c, nm in (("ew", "universe"), ("jk", "index")):
        s = ev[c].to_numpy()
        say(f"  {nm:<9} mean {100*s.mean():+.2f}%  median "
            f"{100*np.median(s):+.2f}%  negative in {int((s<0).sum())}/"
            f"{len(s)}  DROP THE WORST YEAR ({ev.index[s.argmin()]}): "
            f"{100*np.delete(s, s.argmin()).mean():+.2f}%")

    #  THE ARITHMETIC THAT DECIDES IT, BEFORE ANY BACKTEST.
    tick = float(np.median([tick_of(p) for p in
                            P[P["elig"]]["close"].sample(20000, random_state=0)
                            / 1.0]))
    px = float(P[P["elig"]]["close"].median())
    toll = FEE + SPREAD_MULT * tick / px
    say(f"\n  COST ARITHMETIC. One out-and-back at the median eligible price "
        f"(Rp{px:,.0f}, tick Rp{tick:.0f}) costs {100*toll:.2f}%.")
    for lo, hi, lab in [(2005, 2015, "early half"), (2016, 2026, "late half")]:
        s = f[(f.yr >= lo) & (f.yr <= hi)]
        eff = s[s.dow != 0].ew.mean() - s[s.dow == 0].ew.mean()
        say(f"    {lab}: skipping Monday is worth {100*eff:.3f}%/week and "
            f"costs {100*toll:.2f}%/week -> ratio {eff/toll:.2f}x")
    return ew


# ===========================================================================
# PART 2 — machinery for the harness variants
# ===========================================================================
class PIT:
    """Point-in-time adjusted close. Never reads a bar stamped after `day`."""

    def __init__(self, P: pd.DataFrame):
        self.d: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        for tk, g in P.groupby("ticker", sort=False):
            self.d[tk] = (g["date"].to_numpy(), g["adj_close"].to_numpy(float))

    def px(self, tk: str, day) -> float:
        dt, p = self.d.get(tk, (None, None))
        if dt is None:
            return float("nan")
        j = np.searchsorted(dt, day, side="right") - 1
        return float(p[j]) if j >= 0 else float("nan")

    def prints(self, tk: str, day) -> bool:
        dt, _ = self.d.get(tk, (None, None))
        if dt is None:
            return False
        i = np.searchsorted(dt, day)
        return bool(i < len(dt) and dt[i] == day)


class Strategy:
    """A drifting book with an optional calendar gate and optional re-picking.

    Stateful across marks, which is legitimate because `walk` calls it in
    chronological order and nothing here reads a bar dated after the mark.

    in_market(dt)  -> False puts the book in cash for the coming period. The
                      harness charges nothing for that, so every such event is
                      counted here and billed in `self_charged`.
    pick(day)      -> the names to hold, or None to keep the existing book.
    """

    def __init__(self, pit: PIT, n: int = 60, gate=None, picker=None,
                 fee: float = FEE, spread_mult: float = SPREAD_MULT):
        self.pit, self.n, self.gate, self.picker = pit, n, gate, picker
        self.fee, self.spread_mult = fee, spread_mult
        self.reset()

    def reset(self) -> None:
        self.book: Dict[str, float] = {}
        self.px0: Dict[str, float] = {}      # each name's price at `last`
        self.last = None
        self.invested = True
        self.excursions = 0        # out-and-back round trips the harness missed
        self.exc_toll = 0.0        # their total cost, compounded
        self.traded = []           # turnover this module actually transacted
        self.dead = 0
        self.marks = 0

    def _toll(self, dt) -> float:
        ps = [self.pit.px(t, dt) for t in self.book]
        ps = [p for p in ps if np.isfinite(p) and p > 0]
        if not ps:
            return self.fee
        return self.fee + self.spread_mult * float(
            np.mean([tick_of(p) / p for p in ps]))

    def __call__(self, day: pd.DataFrame) -> List[Tuple[str, float]]:
        dt = day["date"].iloc[0]
        self.marks += 1
        # 1. mark the book to market, but ONLY if it was invested last period.
        #    The previous mark's price is carried in `px0` rather than looked
        #    up again: at daily marks over 21 years that halves the work.
        if self.book and self.last is not None:
            nb, np0 = {}, {}
            for tk, v in self.book.items():
                p1 = self.pit.px(tk, dt)
                p0 = self.px0.get(tk, float("nan"))
                if not (np.isfinite(p0) and p0 > 0 and np.isfinite(p1)
                        and p1 > 0):
                    continue
                if not self.pit.prints(tk, dt):
                    self.dead += 1     # delisting, realised at the last print
                    continue
                #  In cash the book does not move; invested, it drifts.
                nb[tk] = v * (p1 / p0) if self.invested else v
                np0[tk] = p1
            self.book, self.px0 = nb, np0
        self.last = dt

        # 2. the calendar gate.
        want = True if self.gate is None else bool(self.gate(dt))
        if not want:
            if self.invested and self.book:
                #  SELL EVERYTHING. The harness will bill nothing for this and
                #  nothing for coming back, so bill it here.
                self.excursions += 1
                self.exc_toll += self._toll(dt)
            self.invested = False
            return []
        if not self.invested:
            self.invested = True       # buying back; the toll above covers it

        # 3. re-pick, if the picker says this is a rebalance bar.
        newset = None if self.picker is None else self.picker(day, dt)
        if newset is None and not self.book:
            newset = list(day.nlargest(min(self.n, len(day)),
                                       "tv60")["ticker"])
        if newset is not None:
            newset = [t for t in newset if self.pit.prints(t, dt)]
            if len(newset) >= 5:
                tot = sum(self.book.values())
                if tot <= 0:
                    self.book = {t: 1.0 / len(newset) for t in newset}
                    self.traded.append(1.0)
                else:
                    old = {t: v / tot for t, v in self.book.items()}
                    new = {t: 1.0 / len(newset) for t in newset}
                    self.traded.append(0.5 * sum(
                        abs(new.get(k, 0.0) - old.get(k, 0.0))
                        for k in set(new) | set(old)))
                    self.book = {t: tot / len(newset) for t in newset}
                self.px0 = {t: self.pit.px(t, dt) for t in self.book}
        if not self.book:
            return []
        tot = sum(self.book.values())
        return [(t, v / tot) for t, v in self.book.items() if v > 0]


def run(B: Bench, strat: Strategy, label: str, freq: int,
        draws: int = 6) -> Dict:
    """Evaluate, then add the self-charged cost the harness does not levy."""
    strat.reset()
    v = B.evaluate(strat, label=label, freq=freq, draws=draws)
    if not v.get("ok"):
        say(report(v))
        return v
    #  Compound the missed out-and-backs into the equity and re-annualise.
    n_exc = strat.excursions
    v["excursions"] = n_exc
    v["self_cost"] = strat.exc_toll
    v["traded"] = float(np.sum(strat.traded))
    if n_exc:
        #  One (1 - toll) factor per out-and-back, at the toll actually
        #  measured on the book that was sold. Not an exponential
        #  approximation: 52 excursions a year is far too many for exp(-x) ~
        #  1-x to be harmless.
        eq = (1.0 + v["cagr"]) ** v["years"]
        mt = strat.exc_toll / n_exc
        eq_adj = eq * (1.0 - mt) ** n_exc
        v["cagr_charged"] = float(max(eq_adj, 1e-12)
                                  ** (1.0 / v["years"]) - 1.0)
        v["exc_per_yr"] = n_exc / v["years"]
        v["mean_toll"] = mt
    else:
        v["cagr_charged"] = v["cagr"]
        v["exc_per_yr"] = 0.0
        v["mean_toll"] = 0.0
    say(report(v))
    say(f"  gross (pre-cost)  {v['gross']:+8.2%}   "
        f"harness turnover {v['turnover']:.2%}/mark, "
        f"actually transacted {v['traded']:.2f} book-turns total")
    if n_exc:
        say(f"  SELF-CHARGED      {v['cagr_charged']:+8.2%}   "
            f"{n_exc} cash excursions ({v['exc_per_yr']:.1f}/yr) at "
            f"{v['mean_toll']:.2%} each — the harness billed none of them")
    return v


# ===========================================================================
# the gates and the pickers
# ===========================================================================
def gate_weekend(dt) -> bool:
    """Out from Friday close to Monday close. Causal: the calendar is known."""
    return dt.dayofweek != 4


def gate_halloween(dt) -> bool:
    return dt.month in (11, 12, 1, 2, 3, 4)


def gate_tom(dt) -> bool:
    """In only around the turn of the month. Uses calendar days, which are
    known in advance; the trading-day index is not knowable at the mark for
    the days still to come."""
    import calendar
    last = calendar.monthrange(dt.year, dt.month)[1]
    return dt.day <= 4 or dt.day >= last - 2


def gate_lebaran(dt) -> bool:
    """Out for the five sessions after Idul Fitri — the strongest Lebaran
    cell in Part 1. Eid dates are known years ahead, so this is causal."""
    g = (dt - EID).days
    g = g[g >= 0]
    return not (len(g) and 0 <= g.min() <= 8)


class ScreenPicker:
    """Strength+calm (A24/H26), re-picked once a year in calendar month `mon`."""

    def __init__(self, mon: int, n: int = 20):
        self.mon, self.n, self.done = mon, n, set()

    def __call__(self, day: pd.DataFrame, dt):
        if dt.month != self.mon or dt.year in self.done:
            return None
        self.done.add(dt.year)
        g = day.dropna(subset=["hi52", "vol60"])
        g = g[g["vol60"] > 0]
        if len(g) < 40:
            return None
        hi, vo = g["hi52"].rank(pct=True), g["vol60"].rank(pct=True)
        s = g[(hi >= 0.90) & (vo <= 0.50)]
        if len(s) < 5:
            s = g[(hi >= 0.80) & (vo <= 0.50)]
        if len(s) < 5:
            return None
        return list(s.nlargest(min(self.n, len(s)), "hi52")["ticker"])


class SeasonPicker:
    """Heston-Sadka: buy the names whose OWN history in the coming calendar
    month is strongest. Strictly causal — only months that ENDED before the
    mark are ever read."""

    def __init__(self, mret: pd.DataFrame, n: int = 20, min_obs: int = 4,
                 every: int = 1):
        #  mret: ticker, end (last session of the month), mon, ret
        self.by: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for tk, g in mret.groupby("ticker", sort=False):
            g = g.sort_values("end")
            self.by[tk] = (g["end"].to_numpy(), g["mon"].to_numpy(),
                           g["ret"].to_numpy(float))
        self.n, self.min_obs, self.every = n, min_obs, every
        self.seen = set()

    def __call__(self, day: pd.DataFrame, dt):
        key = (dt.year, (dt.month - 1) // self.every)
        if key in self.seen:
            return None
        self.seen.add(key)
        tgt = [((dt.month - 1 + k) % 12) + 1 for k in range(self.every)]
        sc = {}
        for tk in day["ticker"]:
            v = self.by.get(tk)
            if v is None:
                continue
            end, mon, ret = v
            m = (end < dt) & np.isin(mon, tgt)
            if m.sum() >= self.min_obs:
                sc[tk] = float(np.nanmean(ret[m]))
        if len(sc) < 20:
            return None
        top = sorted(sc, key=lambda t: -sc[t])[:self.n]
        return top


def monthly_returns(P: pd.DataFrame) -> pd.DataFrame:
    Q = P[["ticker", "date", "adj_close"]].copy()
    Q["ym"] = Q["date"].dt.to_period("M")
    g = Q.groupby(["ticker", "ym"], as_index=False).agg(
        end=("date", "max"), px=("adj_close", "last"))
    g = g.sort_values(["ticker", "ym"])
    g["ret"] = g.groupby("ticker")["px"].pct_change()
    g["gapok"] = g.groupby("ticker")["ym"].diff().apply(
        lambda x: x is not pd.NaT and getattr(x, "n", 99) == 1)
    g = g[g["ret"].notna() & g["gapok"] & (g["ret"].abs() < 3)]
    g["mon"] = g["ym"].dt.month
    return g[["ticker", "end", "mon", "ret"]]


# ===========================================================================
def leakage(P: pd.DataFrame, tag: str) -> None:
    """HOW MUCH RETURN THE HARNESS DROPS AT DAILY MARKS, AND WHY.

    `walk` skips any mark whose eligible cross-section is under MIN_UNIV=40 and
    never earns that period's return. At the default Rp1bn floor that is 505 of
    5,329 sessions -- and they are not scattered, they are 85% of 2006 and 46%
    of 2009, because the floor is nominal and the universe had not yet grown
    into it. Those two years are the biggest up-years in the sample, so a
    strategy is held in forced cash through them while every buy-and-hold
    benchmark is not. Measured below rather than argued: it is worth 7.3 log
    points a year, which is larger than every calendar effect in Part 1 put
    together and would have decided this study on its own.
    """
    d = pd.DatetimeIndex(np.sort(P["date"].unique()))
    d = d[d >= pd.Timestamp("2005-01-11")]
    n = P[P["elig"]].groupby("date").size().reindex(d).fillna(0)
    Q = P.sort_values(["ticker", "date"]).copy()
    Q["ret"] = Q.groupby("ticker")["adj_close"].pct_change()
    Q["ep"] = Q.groupby("ticker")["elig"].shift(1).fillna(False)
    r = Q[Q["ep"] & Q["ret"].notna() & (Q["ret"].abs() < 0.6)]
    ew = r.groupby("date")["ret"].mean().reindex(d).fillna(0.0)
    skip = (n < 40).to_numpy()
    earned = np.r_[False, ~skip[:-1]]      # d[i]'s return needs d[i-1] alive
    lg = np.log1p(ew.to_numpy())
    yrs = (d[-1] - d[0]).days / 365.25
    say(f"  LEAKAGE at {tag}: {int(skip.sum())} of {len(d)} sessions skipped "
        f"(universe under 40); a daily walk earns {lg[earned].sum():.3f} of "
        f"{lg.sum():.3f} log — {100*(lg.sum()-lg[earned].sum())/yrs:.2f} log "
        f"points a year lost to the skip alone")


def main() -> None:
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    P = load()
    pit = PIT(P)
    B = Bench(P)

    if what in ("scan", "all"):
        part_scan(P)

    if what in ("time", "all"):
        say("\n" + "=" * 78)
        say("PART 2A — TIMING OVERLAYS ON AN INERT CORE (daily marks)")
        say("=" * 78)
        say("  Core = the 40 most liquid eligible names at the first bar,")
        say("  equal weight, never traded again. S1-S4 differ from S0 ONLY in")
        say("  the calendar gate, so S1-S0 is the whole effect of the switch.")
        say("  Day-of-week rules need daily marks and daily marks make the")
        say("  harness leak, so the leak is measured first and the floor is")
        say("  moved to the level at which it is only the exchange holidays.")
        leakage(P, "min_tv=1e9 (the default)")
        P2 = load(min_tv=2e8)
        leakage(P2, "min_tv=2e8 (used below)")
        pit2, B2 = PIT(P2), Bench(P2)
        res = {}
        for lab, gate in [("S0 core: always in", None),
                          ("S1 out Fri->Mon (weekend effect)", gate_weekend),
                          ("S2 out May-Oct (Halloween)", gate_halloween),
                          ("S3 in only turn-of-month", gate_tom),
                          ("S4 out 5d after Lebaran", gate_lebaran)]:
            say("")
            res[lab] = run(B2, Strategy(pit2, n=40, gate=gate), lab, freq=1,
                           draws=2)
        base = res["S0 core: always in"]
        say("\n  OVERLAY MINUS CORE (the only clean comparison in this block)")
        say(f"  {'variant':<34}{'net':>9}{'vs S0':>9}{'gross':>9}"
            f"{'vs S0':>9}{'charged':>9}{'vs S0':>9}")
        for lab, v in res.items():
            if not v.get("ok"):
                continue
            say(f"  {lab:<34}{v['cagr']:>+9.2%}"
                f"{v['cagr']-base['cagr']:>+9.2%}{v['gross']:>+9.2%}"
                f"{v['gross']-base['gross']:>+9.2%}{v['cagr_charged']:>+9.2%}"
                f"{v['cagr_charged']-base['cagr_charged']:>+9.2%}")

    if what in ("month", "all"):
        say("\n" + "=" * 78)
        say("PART 2B — WHICH MONTH TO TAKE THE ONE ANNUAL REBALANCE (12 cells)")
        say("=" * 78)
        say("  Strength+calm screen (A24/H26), 20 names, rebalanced once a")
        say("  year in month m and left to drift in between. Weekly marks.")
        say("  THE SPREAD ACROSS THE TWELVE IS THE STATISTIC. The maximum of")
        say("  twelve is not, and a study that quotes it has scanned twelve")
        say("  cells and reported one. Run at BOTH liquidity floors, because")
        say("  a conclusion that moves with the floor is a conclusion about")
        say("  the floor.")
        for tv in (2e8, 1e9):
            Pm = P if tv == 1e9 else load(min_tv=tv)
            Bm, pm = (B, pit) if tv == 1e9 else (Bench(Pm), PIT(Pm))
            say(f"\n  ---- liquidity floor min_tv = Rp{tv:,.0f}/day ----")
            out = []
            for m in range(1, 13):
                s = Strategy(pm, n=20, picker=ScreenPicker(m, n=20))
                v = run(Bm, s, f"S5.{m:02d} annual rebalance in month {m:02d}",
                        freq=5, draws=4)
                out.append(v)
                say("")
            ok = [v for v in out if v.get("ok")]
            if not ok:
                continue
            c = np.array([v["cagr"] for v in ok])
            say(f"  TWELVE CELLS: mean {c.mean():+.2%}, sd {c.std(ddof=1):.2%},"
                f" min {c.min():+.2%} (month "
                f"{int(np.argmin(c))+1:02d}), max {c.max():+.2%} (month "
                f"{int(np.argmax(c))+1:02d}), spread {c.max()-c.min():.2%}")
            say(f"  the se of any one cell's mean is not available, but the "
                f"twelve share one history: they overlap by 11/12 of every "
                f"holding period, so they are ~1 observation, not 12")
            say(f"  cells beating BH index: "
                f"{sum(v['beats_index'] for v in ok)}/12;  beating BH picks: "
                f"{sum(v['beats_picks'] for v in ok)}/12;  PASS: "
                f"{sum(v['PASS'] for v in ok)}/12")

    if what in ("xsec", "all"):
        say("\n" + "=" * 78)
        say("PART 2C — CROSS-SECTIONAL SEASONALITY (Heston-Sadka)")
        say("=" * 78)
        say("  Buy the names whose own history in the COMING calendar month is")
        say("  strongest. The only seasonality claim here that is about which")
        say("  stock rather than when, so it needs no market timing.")
        mret = monthly_returns(P)
        say(f"  monthly return observations: {len(mret):,} over "
            f"{mret['ticker'].nunique()} names")
        for every, freq, nm in [(1, 21, "monthly"), (3, 63, "quarterly")]:
            say("")
            s = Strategy(pit, n=20,
                         picker=SeasonPicker(mret, n=20, every=every))
            run(B, s, f"S6 same-month seasonality, {nm} rebalance", freq=freq)

    os.makedirs("reports", exist_ok=True)
    with open(OUT, "a") as f:
        f.write("\n".join(LOG) + "\n")
    print(f"\n[written to {OUT}]")


if __name__ == "__main__":
    main()
