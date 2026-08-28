#!/usr/bin/env python3
"""H43 — how often should the surviving screen be rebalanced?

    python3 scripts/rebalance.py

THE PARAMETER NOBODY CHOSE.

H26's strength+calm screen is the one result in this project that cleared the
Bonferroni bar, replicated across both halves, and was independently confirmed
out of sample by H27's purged walk-forward model. Its basket was rebalanced
ANNUALLY — not because anyone tested that, but because H16 picked a 252-session
horizon for an unrelated reason and twelve studies inherited it.

A20 is explicit about what that costs: "a parameter fixed once by convenience
and then inherited by twelve studies is not a constant; it is an untested
assumption with a project-wide blast radius. Vary the thing every experiment
holds fixed." When the horizon was finally varied it INVERTED the answer.

So this varies rebalance frequency from monthly to triennial, on the same
screen, the same universe, the same cost model, against the same two controls
that have decided every result here: a random basket drawn from the identical
liquid universe on the identical schedule, and the IHSG on a total-return basis
over each arm's OWN window.

WHY THE ANSWER IS NOT OBVIOUS IN EITHER DIRECTION.
Rebalancing less often pays the toll less often — H23 measured that the cost
structure "stops binding when you stop paying it annually". But a screen is a
statement about the CURRENT cross-section, and a three-year-old ranking is a
statement about a market that no longer exists. Cost falls monotonically with
holding period; signal decays monotonically with it. The optimum is interior
and nothing in this repo has ever located it.

COST IS CHARGED ON TURNOVER, NOT ON FREQUENCY. Two names changing out of ten
costs a fifth of a round trip, not a whole one. Every earlier study in this repo
that priced "a rebalance" priced the whole book, which overstates the toll on a
sticky screen and understates the advantage of rebalancing less. Turnover is
measured here rather than assumed.

PRE-REGISTERED, WRITTEN BEFORE ANY CELL WAS SCORED
--------------------------------------------------
P1  Net mean-log return is HUMPED in the holding period: monthly is clearly
    worst (turnover eats it), and the optimum sits somewhere between one
    quarter and one year. If it turns out monotone-increasing, the honest
    reading is that the screen carries no decay this data can see and the only
    thing frequency does is cost.
P2  The picks-minus-random EDGE is roughly INVARIANT to rebalance frequency.
    Signal quality is a property of the cross-section, not of how often you
    act on it, so the whole frequency effect should be cost and drift rather
    than selection. If the edge itself moves strongly with frequency, suspect
    the harness.
P3  PREDICTED NULL — the rebalance PHASE does not matter. Starting the same
    schedule in January or in July is an arbitrary choice, and if the spread
    across phases is comparable to the spread across FREQUENCIES then the
    study is measuring a calendar accident and not a holding period. This is
    the A15 tie-break lesson in a new place: check the arbitrary choice before
    believing the deliberate one.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))
sys.path.insert(0, os.path.dirname(__file__))

from asymmetry import SCREEN                                     # noqa: E402
from hull_trade import COST                                      # noqa: E402
from paint_suite import tick_of                                  # noqa: E402

PANEL = os.path.join("data", "spine", "price_panel.parquet")
INDEX = os.path.join("data", "cache", "ohlcv", "_JKSE.csv.gz")
OUT = "reports"
MIN_TV = 1e9
#: A19 measured IDX dividend yield rising monotonically with liquidity, 0.65%
#: in decile 1 to 2.01% in decile 10; the cap-weighted index earns the
#: large-cap 1.77%. `^JKSE` is a PRICE index and the names run on `adj_close`,
#: which is already total return — so comparing them raw understates the index.
IDX_YIELD = 0.0177
NAMES = 10
#: monthly, quarterly, half-yearly, yearly, two-yearly, three-yearly
FREQS = (21, 63, 126, 252, 504, 756)


def load() -> pd.DataFrame:
    P = pd.read_parquet(PANEL)
    P = P[P["adj_close"] > 0].sort_values(["ticker", "date"])
    P["elig"] = (P["tradeable"].astype(bool)
                 & (np.exp(P["log_turnover"].fillna(-np.inf)) >= MIN_TV))
    return P


class Prices:
    """Per-ticker arrays, so a price lookup never touches a pivot.

    A11's defect: a rolling window on a date x ticker pivot is indexed by the
    UNION of trading days, so one suspended name inserts rows it never had.
    Nothing here rolls on a pivot; every lookup is inside one name's own bars,
    which also makes delisting detectable rather than silently forward-filled.
    """

    def __init__(self, P: pd.DataFrame):
        self.d: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        for tk, g in P.groupby("ticker", sort=False):
            self.d[tk] = (g["date"].to_numpy(),
                          g["adj_close"].to_numpy(float))

    def at(self, tk: str, day) -> float:
        """Close on `day`, or nan if the name did not trade that day."""
        dt, px = self.d[tk]
        i = np.searchsorted(dt, day)
        return float(px[i]) if i < len(dt) and dt[i] == day else np.nan

    def exit_price(self, tk: str, buy_day, sell_day) -> Tuple[float, bool]:
        """Price on `sell_day`, or the last price the name ever printed.

        A NAME THAT STOPS PRINTING IS A DELISTING, NOT A HOLD FOREVER. H41 had
        exactly this bug: a position whose ticker vanished was carried at its
        last price for the rest of the backtest, blocking a slot and inflating
        the book. Here it is realised at the last real bar and flagged.
        """
        dt, px = self.d[tk]
        j = np.searchsorted(dt, sell_day, side="right") - 1
        if j < 0:
            return np.nan, False
        return float(px[j]), bool(dt[j] < sell_day)


def screen(d: pd.DataFrame, n: int, rng=None) -> List[str]:
    """The H26 cell: strength AND calm, cross-sectionally, among liquid names.

    `rng` switches to the control — a random draw from the IDENTICAL universe.
    Same size, same schedule, same costs; only the selection differs.
    """
    d = d[d["elig"]].dropna(subset=["hi52", "vol60"])
    if len(d) < 40:
        return []
    s = d[(d["hi52"] >= d["hi52"].quantile(SCREEN["hi52_pct"]))
          & (d["vol60"] <= d["vol60"].quantile(SCREEN["vol_pct"]))]
    if rng is not None:
        #  THE CONTROL IS SIZE-MATCHED TO THE CELL, not to a fixed N. A random
        #  basket of a different width is a different portfolio, and breadth
        #  alone changes the variance and therefore the compounding.
        k = min(max(len(s), 1), len(d))
        return list(d["ticker"].to_numpy()[rng.choice(len(d), k,
                                                      replace=False)])
    #  HOLD THE WHOLE CELL, NOT ITS TOP N BY hi52.
    #  A first version sorted by hi52 and took the top ten, which is a SECOND
    #  selection stacked on the screen -- and H26's own frontier says it is a
    #  worse one: "strength, very strong" scores skew 1.95 against 2.15 for
    #  plain "strength". It returned +11.1% a year where H26 measured +18.5%,
    #  and two studies of one screen disagreeing is the signal that one of them
    #  is wrong. A15's lesson in a new place: a top-N cut imposed on a cell is
    #  an arbitrary choice that nobody registered, and it decided the answer.
    return list(s["ticker"])


def run(P: pd.DataFrame, PX: Prices, dates: np.ndarray, freq: int,
        offset: int, n: int, seed: int | None = None) -> Dict:
    """One equity path: rebalance every `freq` sessions starting at `offset`."""
    rng = np.random.default_rng(seed) if seed is not None else None
    marks = dates[offset::freq]
    if len(marks) < 3:
        return {}
    by_date = {d: g for d, g in P[P["date"].isin(marks)].groupby("date")}
    book: List[str] = []
    #  GROSS IS TRACKED ALONGSIDE NET BECAUSE THE CONTROL DOES NOT CHURN LIKE
    #  THE TREATMENT. A random basket redrawn every month rotates ~100% of its
    #  book; the screen rotates 62%. So a net-of-cost difference between them
    #  confounds SELECTION with TURNOVER, and at monthly frequency the cost gap
    #  alone is several points a year. The selection effect is the GROSS
    #  difference; the rest is the toll, and the two are reported separately.
    eq, gross, turns, costs, ndel = 1.0, 1.0, [], [], 0
    curve: List[Tuple] = []
    sizes: List[int] = []
    for a, b in zip(marks[:-1], marks[1:]):
        want = screen(by_date.get(a, P.iloc[0:0]), n, rng)
        sizes.append(len(want))
        if not want:
            curve.append((b, eq))
            continue
        #  COST ON TURNOVER, NOT ON FREQUENCY. Names retained across a
        #  rebalance pay nothing; only the rotated fraction pays a round trip.
        keep = len(set(book) & set(want))
        turn = 1.0 - keep / max(len(want), 1) if book else 1.0
        px_a = {tk: PX.at(tk, a) for tk in want}
        toll = np.nanmean([COST + tick_of(v) / v
                           for v in px_a.values() if np.isfinite(v) and v > 0])
        toll = float(toll) if np.isfinite(toll) else COST
        eq *= (1.0 - turn * toll)
        turns.append(turn)
        costs.append(turn * toll)
        rets = []
        for tk in want:
            p0 = px_a[tk]
            if not np.isfinite(p0) or p0 <= 0:
                continue
            p1, dead = PX.exit_price(tk, a, b)
            if not np.isfinite(p1) or p1 <= 0:
                continue
            ndel += int(dead)
            rets.append(p1 / p0 - 1.0)
        if rets:
            eq *= (1.0 + float(np.mean(rets)))
            gross *= (1.0 + float(np.mean(rets)))
        book = want
        curve.append((b, eq))
    if len(curve) < 3:
        return {}
    yrs = (curve[-1][0] - marks[0]).astype("timedelta64[D]").astype(float) / 365.25
    return {"freq": freq, "offset": offset, "start": marks[0],
            "end": curve[-1][0], "years": yrs, "eq": eq,
            "cagr": eq ** (1.0 / yrs) - 1.0,
            "gross_cagr": gross ** (1.0 / yrs) - 1.0,
            "mean_log": float(np.log(max(eq, 1e-9)) / yrs),
            "basket": float(np.mean([s for s in sizes if s])) if any(sizes)
            else np.nan,
            "turnover": float(np.mean(turns)) if turns else np.nan,
            "cost_yr": float(np.sum(costs) / yrs),
            "rebalances": len(turns), "delistings": ndel,
            "curve": curve}


def half_cagr(curve: List[Tuple]) -> Tuple[float, float]:
    """CAGR in the early and late half of one equity path.

    THE HALF-SPLIT HAS TO BE ON THE RETURN PERIOD, NOT THE START DATE. Every
    arm here starts within a few years of the panel's beginning, so splitting
    on `start` would put all of them in the early bucket and silently report a
    replication test that never ran.
    """
    if len(curve) < 4:
        return np.nan, np.nan
    mid = len(curve) // 2
    out = []
    for lo, hi in ((0, mid), (mid, len(curve) - 1)):
        d0, e0 = curve[lo]
        d1, e1 = curve[hi]
        yrs = (d1 - d0).astype("timedelta64[D]").astype(float) / 365.25
        out.append((max(e1, 1e-9) / max(e0, 1e-9)) ** (1.0 / max(yrs, 1e-9))
                   - 1.0 if yrs > 0 else np.nan)
    return out[0], out[1]


def index_tr(a, b) -> float:
    """IHSG CAGR between two dates on a TOTAL-return basis, over the arm's own
    window. A19: comparing quantities measured over different windows was the
    single error that ran through every version of that table."""
    d = pd.read_csv(INDEX)
    d["date"] = pd.to_datetime(d["date"])
    d = d[(d["adj_close"] > 0)].set_index("date")["adj_close"]
    s = d.reindex(pd.DatetimeIndex([a, b]), method="ffill")
    yrs = (b - a).astype("timedelta64[D]").astype(float) / 365.25
    return float((s.iloc[-1] / s.iloc[0]) ** (1.0 / yrs) - 1.0) + IDX_YIELD


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", type=int, default=NAMES)
    ap.add_argument("--phases", type=int, default=8)
    a = ap.parse_args()

    P = load()
    PX = Prices(P)
    dates = np.sort(P["date"].unique())
    print(f"{len(P):,} rows, {P['ticker'].nunique()} names, "
          f"{pd.Timestamp(dates[0]):%Y-%m-%d} -> "
          f"{pd.Timestamp(dates[-1]):%Y-%m-%d}")
    print(f"screen = hi52 >= p{SCREEN['hi52_pct']:.0%} AND "
          f"vol60 <= p{SCREEN['vol_pct']:.0%}, {a.names} names, "
          f"equal weight, cost on TURNOVER at {COST:.2%} + fraksi harga\n")

    rows: List[Dict] = []
    for f in FREQS:
        #  MANY PHASES, BECAUSE THE START MONTH IS ARBITRARY (P3). If the
        #  spread across phases rivals the spread across frequencies, the
        #  study is measuring a calendar accident.
        offs = np.linspace(0, f - 1, min(a.phases, f)).astype(int)
        for o in offs:
            r = run(P, PX, dates, f, int(o), a.names)
            if not r:
                continue
            c = run(P, PX, dates, f, int(o), a.names, seed=9000 + f + int(o))
            r["rand_cagr"] = c.get("cagr", np.nan)
            r["rand_gross"] = c.get("gross_cagr", np.nan)
            r["rand_turn"] = c.get("turnover", np.nan)
            r["rand_cost"] = c.get("cost_yr", np.nan)
            r["idx_cagr"] = index_tr(r["start"], r["end"])
            pe, pl = half_cagr(r["curve"])
            ce, cl = half_cagr(c.get("curve", []))
            r["early"], r["late"] = pe, pl
            r["rand_early"], r["rand_late"] = ce, cl
            rows.append(r)
    R = pd.DataFrame(rows)
    R.drop(columns=["curve"]).to_csv(os.path.join(OUT, "rebalance.csv"),
                                     index=False)

    print("=== P1/P2 — by rebalance frequency, averaged over start phases")
    print(f"{'hold':>10}{'n':>4}{'held':>6}{'turn':>7}{'rnd trn':>9}{'cost/yr':>9}"
          f"{'CAGR':>9}{'random':>9}{'index':>9}{'vs idx':>9}"
          f"{'GROSS edge':>12}{'net edge':>10}{'phase sd':>10}")
    for f in FREQS:
        g = R[R["freq"] == f]
        if not len(g):
            continue
        lbl = {21: "1 month", 63: "1 quarter", 126: "6 months",
               252: "1 year", 504: "2 years", 756: "3 years"}[f]
        print(f"{lbl:>10}{len(g):>4}{g['basket'].mean():>6.0f}"
              f"{g['turnover'].mean():>7.0%}"
              f"{g['rand_turn'].mean():>9.0%}{g['cost_yr'].mean():>9.2%}"
              f"{g['cagr'].mean():>+9.2%}"
              f"{g['rand_cagr'].mean():>+9.2%}{g['idx_cagr'].mean():>+9.2%}"
              f"{g['cagr'].mean() - g['idx_cagr'].mean():>+9.2%}"
              f"{g['gross_cagr'].mean() - g['rand_gross'].mean():>+12.2%}"
              f"{g['cagr'].mean() - g['rand_cagr'].mean():>+10.2%}"
              f"{(g['cagr'] - g['idx_cagr']).std(ddof=1):>10.2%}")

    print("\n=== P3 — the predicted null: does the START PHASE decide it?")
    #  MEASURED ON THE EXCESS OVER THE INDEX, NOT ON RAW CAGR. For a 3-year
    #  frequency the phase offset moves the START DATE by up to three years, so
    #  different phases cover different markets and a raw-CAGR spread would
    #  confound phase with window. `cagr - idx_cagr` is computed against the
    #  index over each arm's OWN span, which is what makes the phases
    #  comparable — A19's lesson, that comparing quantities measured over
    #  different windows was the error running through every draft of it.
    R["exc"] = R["cagr"] - R["idx_cagr"]
    across_freq = R.groupby("freq")["exc"].mean().std(ddof=1)
    within = R.groupby("freq")["exc"].std(ddof=1).mean()
    print(f"  spread across FREQUENCIES (sd of the six means)  {across_freq:.2%}")
    print(f"  spread across PHASES within a frequency (mean sd) {within:.2%}")
    ratio = across_freq / max(within, 1e-9)
    print(f"  ratio {ratio:.2f}x")
    #  A BINARY VERDICT HERE WOULD BE TOO GENEROUS. A first version printed
    #  "frequency dominates, the result is readable" for a ratio of 1.15, which
    #  is very nearly a tie. What the two numbers actually license is a
    #  statement about WHICH comparisons survive: with 8 phases the standard
    #  error of a frequency's mean is within/sqrt(8), and only differences
    #  larger than about two of those are readable at all.
    se = within / np.sqrt(R.groupby("freq").size().mean())
    print(f"  se of each frequency's mean ~{se:.2%}; "
          f"differences under ~{2 * se:.2%} are NOT readable")
    if ratio < 1.5:
        print("  THE PHASE NOISE IS COMPARABLE TO THE FREQUENCY EFFECT. The "
              "SHAPE of the curve is")
        print("  readable; the precise optimum among the middle frequencies "
              "is not, because an")
        print("  arbitrary choice of start month moves any single arm by more "
              "than they differ.")
    M = R.groupby("freq")["exc"].mean()
    print(f"  readable pairs (|difference| > 2se): ", end="")
    pairs = [f"{a}v{b}" for i, a in enumerate(FREQS) for b in FREQS[i + 1:]
             if abs(M.get(a, np.nan) - M.get(b, np.nan)) > 2 * se]
    print(", ".join(pairs) if pairs else "NONE")

    print("\n=== the half-split, picks minus random GROSS, inside each path")
    for f in FREQS:
        g = R[R["freq"] == f]
        if not len(g):
            continue
        de = (g["early"] - g["rand_early"]).mean()
        dl = (g["late"] - g["rand_late"]).mean()
        print(f"  {f:>4} sessions: early {de:+.2%}  late {dl:+.2%}   "
              f"picks {g['early'].mean():+.2%}/{g['late'].mean():+.2%}   "
              f"{'BOTH' if de > 0 and dl > 0 else 'not both'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
