#!/usr/bin/env python3
"""The card backtested year by year against the IHSG, as printed and corrected.

WHAT THIS IS
    The shipped quarterly card (scripts/rules.py: hi52/vol60 entry and keep
    bands, K=10, -20% stop, sell half at +100%) walked through
    `stoptest.walk` on the same six rebalance calendars H56/H63 use, and
    reported over ONE COMMON WINDOW, so that every calendar and the index are
    measured over exactly the same dates. A19 records comparing quantities
    measured over different windows as this repo's most repeated error; here
    the window is shared by construction and a test pins it.

    It registers no hypothesis and adds no trial. It re-measures a rule that
    H56/H63 already measured and shows it the way a holder reads it: year by
    year, next to the index, with the drawdown.

    "Common window" means the dates on which every calendar HAS AN EQUITY
    CURVE, not the dates on which every calendar is invested: on its first
    day, 2008-03-25, three of the six calendars held no stock at all.

THE COMPARISON IN BETWEEN. Beside the cap-weighted IHSG, every eligible name is
also held in EQUAL weight, rebalanced on the card's own calendars at the same
costs (`ew_universe_curve`). An equal-weighted IDX basket and the cap-weighted
index are different animals (A19). The card-versus-index gap mixes selection
with weighting; the card-versus-universe gap isolates the selection.

TWO HARNESS CORRECTIONS, found by the 2026-11-16 feature inventory and printed
BESIDE the uncorrected numbers rather than replacing them:

  1. ONE ROUND TRIP PER POSITION. `stoptest.walk` charges every sale the full
     round-trip cost on top of the half every purchase pays at the rebalance,
     so each position pays 1.5 round trips. `exit_cost_mult=0.5` charges one.
  2. THE POINT-IN-TIME TICK LADDER. `stoptest.prep` prices the spread with
     `paint_suite.tick_of`, today's ladder, on every bar. The point-in-time
     ladder (src/idxbot/spine/reference.py) was coarser before 2014-01-06 and
     FINER in two bands from 2014-01-06 to 2016-05-02, so the error runs both
     ways; the net is a small understatement. The lookahead is REDUCED, not
     removed: about a fifth of bars close off the point-in-time tick grid
     because they are vendor-adjusted (A2), and their band lookup still
     depends on later corporate actions.

TWO MORE HARNESS DEFECTS, found by the independent check of this file
(2026-11-17) and NOT corrected here. Both are pre-existing in `stoptest.walk`:

  * A review that lands on a panel row where nothing trades (an exchange
    holiday or a vendor zero-volume day) sees no eligible names and skips the
    whole quarter. Deferring such reviews to the next trading session moves the
    corrected card's median by -0.04 points and its weakest calendar by -0.8.
    It also roughly halves the measured CAGR cost of the stop and scale-out
    (1.47 -> 0.70 points on stoptest.json's own-window convention).
  * A name added unfunded at a review that could not form a basket keeps that
    review's price as the reference for its stop and target, and fires
    zero-value "stops". Resetting the reference at funding moves the median by
    +0.09 points.

WHAT IT DOES NOT FIX, stated here so no number below is read past it:

  * SURVIVOR-ONLY. The panel holds no delisted names; the delisted store waits
    on the user's licence decision (A51). H54's partial correction moved this
    family by -2.64 to +0.90 points a year, so the sign of the bias on the
    corrected figure is not known.
  * SAME-CLOSE FILLS. Every trade fills at the close of the bar that signalled
    it, which a manual trader acting after 15:50 cannot do.
  * HALF A TICK of spread against OJK's measured ~1.6 ticks (A51), and no
    impact, suspension or auto-rejection term (A23). `--spread-mult` prices the
    spread at another multiple of a tick.
  * THE THIN YEARS ARE A GUARD, NOT A RULE. In 2008-09 most reviews could not
    form a 5-name basket (MIN_BASKET; the entry band passed 2-6 names), and
    from late 2008 to mid-2009 the universe also fell under the 40-name floor.
    Stops and band exits then sat in cash: about 82% of the book from
    2008-03-25 and 71% in 2009. The 2008 "protection" is that cash, not a
    decision a holder of rules.py would have taken.
  * IN-SAMPLE. The holdout was spent at H16. Six heavily overlapping calendars
    are a robustness check, not a significance test.

USAGE
    python scripts/card_backtest.py [--chart reports/card_backtest.png]
    python scripts/card_backtest.py --chart-only --chart reports/x.png
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))
sys.path.insert(0, os.path.dirname(__file__))

import stoptest as st                                              # noqa: E402
from bhbench import IDX_YIELD, INDEX, load                         # noqa: E402
from rules import STOP, TP, TP_FRAC                                # noqa: E402

from idxbot.spine import reference as ref                          # noqa: E402

PHASES = 6
OUT = os.path.join("reports", "card_backtest.json")
CURVES = os.path.join("reports", "card_backtest_curves.parquet")

#  The levels are IMPORTED from rules.py, never retyped: A47 records two copies
#  of one constant drifting apart, and this file must describe the card that
#  ships, whatever it is when it runs.
ARMS: Dict[str, Dict] = {
    "card": {"stop": STOP, "tp": TP, "tp_frac": TP_FRAC},
    "no_levels": {},
}
VARIANTS: Dict[str, Dict] = {
    "as_printed": {"pit_ticks": False, "exit_cost_mult": 1.0,
                   "spread_mult": st.SPREAD_MULT},
    "corrected": {"pit_ticks": True, "exit_cost_mult": 0.5,
                  "spread_mult": st.SPREAD_MULT},
}
#  Spread sensitivity, card arm only: the corrected harness with the spread
#  priced at this many ticks per round trip instead of half of one. OJK's
#  measured market-wide quoted spread is about 1.6 ticks (A51).
SPREAD_SENSITIVITY = (1.0, 1.6)


# ------------------------------------------------------------------ costs --
def pit_tick(close: np.ndarray, dates) -> np.ndarray:
    """`reference.tick_size` vectorised over a panel.

    Bars before the schedule's first regime take that first (coarsest) ladder.
    They carry no trade in any calendar here -- the earliest phase opens its
    first basket in 2005 -- and the test compares this to `tick_size` on every
    regime the schedule covers.
    """
    close = np.asarray(close, float)
    d = pd.DatetimeIndex(pd.to_datetime(dates))
    out = np.full(len(close), np.nan)
    for i, r in enumerate(ref._TICK):
        lo = r.start if i > 0 else pd.Timestamp("1900-01-01")
        hi = r.end if r.end is not None else pd.Timestamp("2200-01-01")
        m = np.asarray((d >= lo) & (d < hi))
        if not m.any():
            continue
        p = close[m]
        val = np.full(p.shape, float(r.bands[-1][1]))
        done = np.zeros(p.shape, bool)
        for ceiling, v in r.bands:
            sel = ~done if ceiling is None else (~done & (p < ceiling))
            val[sel] = float(v)
            done |= sel
        out[np.where(m)[0]] = val
    return out


def with_costs(P: pd.DataFrame, pit_ticks: bool,
               spread_mult: float = st.SPREAD_MULT) -> pd.DataFrame:
    """The prepared panel with its per-name round-trip cost column.

    Unchanged when the variant is the shipped one, so the positive control
    runs on exactly the panel stoptest.py builds.
    """
    if not pit_ticks and spread_mult == st.SPREAD_MULT:
        return P
    Q = P.copy()
    close = Q["close"].to_numpy(float)
    if pit_ticks:
        tick = pit_tick(close, Q["date"].to_numpy())
    else:
        tick = np.array([st.tick_of(x) for x in close])
    Q["cost"] = st.FEE + spread_mult * tick / close
    return Q


def ew_universe_curve(P: pd.DataFrame, offset: int
                      ) -> Tuple[np.ndarray, np.ndarray]:
    """Own EVERY eligible name in equal weight, on the card's own calendar.

    The comparison between the card and the cap-weighted index (A19). It uses
    the same marks (every FREQ sessions from `offset`), the same 40-name floor,
    the same panel and the same per-name cost, charged as one round trip per
    position: half on each side of the rebalance turnover. Between marks the
    weights drift with prices. Returns are taken within ticker, never across
    a pivot's union of dates (A11).
    """
    Q = P.sort_values(["ticker", "date"])
    r = Q.groupby("ticker")["adj_close"].pct_change()
    dates = np.sort(Q["date"].unique())
    R = (Q.assign(r=r).pivot(index="date", columns="ticker", values="r")
         .reindex(dates).fillna(0.0).to_numpy())
    E = (Q.pivot(index="date", columns="ticker", values="elig")
         .reindex(dates).astype("boolean").fillna(False)
         .to_numpy(dtype=bool))
    C = (Q.pivot(index="date", columns="ticker", values="cost")
         .reindex(dates).to_numpy(dtype=float))
    marks = set(dates[offset::st.FREQ].tolist())
    val = np.zeros(R.shape[1])
    cash, t0, out = 1.0, None, []
    for j, d in enumerate(dates):
        val = val * (1.0 + R[j])
        if d in marks and E[j].sum() >= st.MIN_UNIV:
            el = E[j]
            total = cash + val.sum()
            tgt = np.where(el, total / el.sum(), 0.0)
            c = np.where(np.isfinite(C[j]), C[j], st.FEE)
            total -= 0.5 * float((np.abs(tgt - val) * c).sum())
            val = np.where(el, total / el.sum(), 0.0)
            cash = 0.0
            if t0 is None:
                t0 = d
        if t0 is not None:
            out.append((d, cash + val.sum()))
    dts = np.array([x[0] for x in out])
    eqs = np.array([x[1] for x in out], float)
    return dts, eqs / eqs[0]


# ------------------------------------------------------------ the windows --
def common_window(curves: List[Tuple[np.ndarray, np.ndarray]]
                  ) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """The dates on which EVERY series has an equity curve: latest start,
    first end. Not the dates on which every calendar is invested."""
    starts = [pd.Timestamp(d[0]) for d, _ in curves]
    ends = [pd.Timestamp(d[-1]) for d, _ in curves]
    return max(starts), min(ends)


def clip(dts, eqs, a: pd.Timestamp, b: pd.Timestamp) -> pd.Series:
    """A curve cut to [a, b] and rebased to 1.0 on its first point."""
    s = pd.Series(np.asarray(eqs, float), index=pd.DatetimeIndex(dts))
    s = s[(s.index >= a) & (s.index <= b)]
    return s / s.iloc[0]


def index_tr(a: pd.Timestamp, b: pd.Timestamp, on: pd.DatetimeIndex,
             path: str = INDEX, yld: float = IDX_YIELD) -> pd.Series:
    """IHSG total return on the CARD'S OWN DATES, rebased to 1.0 at `a`.

    ^JKSE is a price index; the dividend is accrued daily at A19's measured
    top-decile yield, compounded in calendar time. stoptest.index_cagr ADDS the
    yield to the price CAGR instead, which reads about 0.1 point lower a year;
    the difference is stated, not hidden. The index is read AS OF each card
    date, so a session one series has and the other lacks cannot shift the
    window.
    """
    J = pd.read_csv(path, parse_dates=["date"]).sort_values("date")
    px = J.set_index("date")["close"].astype(float)
    px = px[px > 0]
    on = pd.DatetimeIndex(on)
    p = px.reindex(px.index.union(on)).ffill().reindex(on)
    days = (on - pd.Timestamp(a)).days.to_numpy(float)
    tr = p.to_numpy(float) * (1.0 + yld) ** (days / 365.25)
    s = pd.Series(tr, index=on)
    return s / s.iloc[0]


# ------------------------------------------------------------- statistics --
def max_dd(s: pd.Series) -> float:
    v = s.to_numpy(float)
    return float(np.min(v / np.maximum.accumulate(v)) - 1.0)


def cagr(s: pd.Series) -> float:
    yrs = (s.index[-1] - s.index[0]).days / 365.25
    return float((s.iloc[-1] / s.iloc[0]) ** (1.0 / yrs) - 1.0)


def halves(s: pd.Series) -> Tuple[float, float]:
    """CAGR before and after the window's DATE midpoint."""
    mid = s.index[0] + (s.index[-1] - s.index[0]) / 2
    a, b = s[s.index <= mid], s[s.index >= mid]
    return cagr(a), cagr(b)


def year_returns(s: pd.Series) -> pd.Series:
    """Calendar-year returns of a curve, the first and last years partial.

    Each year runs from the last value on or before the previous year-end (the
    curve's first point for the first year) to the last value on or before its
    own year-end. Chained, they reproduce the whole-window return exactly; the
    test checks that identity.
    """
    out = {}
    years = range(s.index[0].year, s.index[-1].year + 1)
    for y in years:
        end = s[s.index <= pd.Timestamp(f"{y}-12-31")]
        start = s[s.index <= pd.Timestamp(f"{y - 1}-12-31")]
        v0 = start.iloc[-1] if len(start) else s.iloc[0]
        out[y] = float(end.iloc[-1] / v0 - 1.0)
    return pd.Series(out)


# -------------------------------------------------------------- the walks --
_P: Dict[str, pd.DataFrame] = {}


_V: Dict[str, Dict] = {}


def _one(job: Tuple[str, str, int]) -> Tuple[str, str, int, Dict]:
    variant, arm, offset = job
    if arm == "ew_universe":
        dts, eqs = ew_universe_curve(_P[variant], offset)
        return variant, arm, offset, {"curve": (dts, eqs)}
    r = st.walk(_P[variant], offset=offset,
                exit_cost_mult=_V[variant]["exit_cost_mult"], **ARMS[arm])
    return variant, arm, offset, r


def offsets() -> List[int]:
    return [int(round(i * st.FREQ / PHASES)) for i in range(PHASES)]


def run_walks(workers: int = 3, spreads=SPREAD_SENSITIVITY
              ) -> Dict[Tuple[str, str], List[Dict]]:
    P = st.prep(load())
    _V.clear()
    _V.update(VARIANTS)
    for m in spreads:
        _V[f"corrected_spread{m:g}"] = {"pit_ticks": True,
                                        "exit_cost_mult": 0.5,
                                        "spread_mult": float(m)}
    for name, v in _V.items():
        _P[name] = with_costs(P, v["pit_ticks"], v["spread_mult"])
    jobs = [(v, a, o) for v in VARIANTS for a in ARMS for o in offsets()]
    jobs += [(v, "card", o) for v in _V if v not in VARIANTS
             for o in offsets()]
    jobs += [("corrected", "ew_universe", o) for o in offsets()]
    out: Dict[Tuple[str, str], List[Dict]] = {}
    #  fork shares the prepared panels copy-on-write; each worker only reads.
    with ProcessPoolExecutor(max_workers=workers,
                             mp_context=multiprocessing.get_context(
                                 "fork")) as ex:
        for variant, arm, offset, r in ex.map(_one, jobs):
            out.setdefault((variant, arm), []).append(dict(r, offset=offset))
    for k in out:
        out[k].sort(key=lambda r: r["offset"])
    return out


def _stats(s: pd.Series, idx: pd.Series) -> Dict:
    yr, iy = year_returns(s), year_returns(idx)
    return {"cagr": cagr(s), "max_dd": max_dd(s), "halves": list(halves(s)),
            "excess": cagr(s) - cagr(idx),
            "years_ahead": int((yr > iy).sum()), "n_years": int(len(yr)),
            "worst_year": float(yr.min()),
            "by_year": {int(y): v for y, v in yr.items()}}


def sub(s: pd.Series, start: Optional[str], end: Optional[str]) -> pd.Series:
    """A sub-window, opening on the last value on or before the day before
    `start` -- the convention `year_returns` uses -- and rebased to 1.0."""
    if start is not None:
        k = s[s.index < pd.Timestamp(start)]
        s = s[s.index >= (k.index[-1] if len(k) else s.index[0])]
    if end is not None:
        s = s[s.index <= pd.Timestamp(end)]
    return s / s.iloc[0]


#  Sub-windows reported beside the full one. The first two drop the thin
#  2008-09 years (and 2010-11, when the book was still part cash); the last two
#  end BEFORE the 2026 fall, because about half of the full-window lead over
#  the index is that one partial year.
SUB_WINDOWS = (("2010-01-01", None), ("2012-01-01", None),
               (None, "2025-12-31"), (None, "2024-12-31"))


def summarise(runs: Dict[Tuple[str, str], List[Dict]]) -> Dict:
    every = [r["curve"] for rs in runs.values() for r in rs
             if "curve" in r]
    a, b = common_window(every)
    on = clip(*every[0], a, b).index
    idx = index_tr(a, b, on)
    res: Dict = {"window": [str(a.date()), str(b.date())],
                 "years": round((b - a).days / 365.25, 2),
                 "index": {"cagr": cagr(idx), "max_dd": max_dd(idx),
                           "halves": list(halves(idx)),
                           "by_year": {int(y): r for y, r in
                                       year_returns(idx).items()}},
                 "variants": {}, "sub_windows": {}}
    curves: Dict[Tuple[str, str], Dict[int, pd.Series]] = {}
    for (variant, arm), rs in runs.items():
        per = []
        for r in rs:
            if "curve" not in r:
                continue
            s = clip(*r["curve"], a, b)
            #  Every series shares the window, so every curve must share its
            #  dates; a mismatch here is a window error, not a rounding one.
            assert s.index.equals(on), "series disagree on the dates"
            curves.setdefault((variant, arm), {})[r["offset"]] = s
            row = dict(_stats(s, idx), offset=r["offset"])
            for k in ("n_stop", "n_tp", "n_band"):
                if k in r:
                    row[k] = int(r[k])
            per.append(row)
        res["variants"].setdefault(variant, {})[arm] = per
        if arm != "ew_universe":
            #  POSITIVE CONTROL. Over each calendar's OWN window, the shipped
            #  variant must reproduce reports/stoptest.json's medians exactly;
            #  if not, this is not the harness the card quotes.
            res.setdefault("own_window_median", {})[f"{variant}/{arm}"] = \
                float(np.median([r["cagr"] for r in rs if "cagr" in r]))
    ew = curves.get(("corrected", "ew_universe"), {})
    for (variant, arm), per in [((v, a_), p) for v, d in
                                res["variants"].items()
                                for a_, p in d.items()]:
        if arm == "ew_universe" or not ew:
            continue
        for row in per:
            e = ew[row["offset"]]
            row["excess_ew"] = row["cagr"] - cagr(e)
            row["halves_ew"] = [x - y for x, y in
                                zip(row["halves"], halves(e))]
    for start, end in SUB_WINDOWS:
        key = f"{start or res['window'][0]}..{end or res['window'][1]}"
        ix = sub(idx, start, end)
        row = {"ihsg": cagr(ix), "ihsg_max_dd": max_dd(ix)}
        for (variant, arm) in (("corrected", "card"),
                               ("corrected", "ew_universe")):
            cs = curves.get((variant, arm), {})
            if not cs:
                continue
            g = [cagr(sub(c, start, end)) for c in cs.values()]
            d = [max_dd(sub(c, start, end)) for c in cs.values()]
            row[arm] = float(np.median(g))
            row[arm + "_max_dd"] = float(np.median(d))
            row[arm + "_ahead"] = int(sum(x > row["ihsg"] for x in g))
        res["sub_windows"][key] = row
    return res


def table(res: Dict, variant: str, arm: str) -> pd.DataFrame:
    per = res["variants"][variant][arm]
    yrs = sorted(per[0]["by_year"])
    rows = []
    for y in yrs:
        v = np.array([p["by_year"][y] for p in per])
        rows.append({"year": y, "median": float(np.median(v)),
                     "lo": float(v.min()), "hi": float(v.max()),
                     "ahead_n": int((v > res["index"]["by_year"][y]).sum()),
                     "ihsg_tr": res["index"]["by_year"][y]})
    return pd.DataFrame(rows).set_index("year")


def headline(res: Dict, variant: str, arm: str) -> Dict:
    per = res["variants"][variant][arm]

    def med(k, i=None):
        return float(np.median([p[k] if i is None else p[k][i] for p in per]))

    h = {"cagr": med("cagr"),
         "cagr_lo": min(p["cagr"] for p in per),
         "cagr_hi": max(p["cagr"] for p in per),
         "max_dd": med("max_dd"),
         "max_dd_worst": min(p["max_dd"] for p in per),
         "early": med("halves", 0), "late": med("halves", 1),
         "excess": med("excess"),
         "calendars_ahead": sum(p["excess"] > 0 for p in per),
         "years_ahead": med("years_ahead"), "n_years": per[0]["n_years"],
         "worst_year": med("worst_year")}
    if all("excess_ew" in p for p in per):
        h["excess_ew"] = med("excess_ew")
        h["calendars_ahead_ew"] = sum(p["excess_ew"] > 0 for p in per)
        h["halves_ew"] = [med("halves_ew", 0), med("halves_ew", 1)]
        h["first_half_ahead_ew"] = sum(p["halves_ew"][0] > 0 for p in per)
    return h


# ------------------------------------------------------------------ chart --
def chart_frames(runs, res: Dict) -> Dict[str, pd.DataFrame]:
    """The corrected card's and the equal-weight universe's six curves on the
    common window, one column per calendar. Saved beside the JSON so the
    chart can be redrawn without re-running every walk."""
    a, b = (pd.Timestamp(x) for x in res["window"])
    out = {}
    for name, key in (("card", ("corrected", "card")),
                      ("ew", ("corrected", "ew_universe"))):
        out[name] = pd.concat({r["offset"]: clip(*r["curve"], a, b)
                               for r in runs[key] if "curve" in r}, axis=1)
    return out


def chart(frames: Dict[str, pd.DataFrame], res: Dict, path: str) -> None:
    """Growth of Rp1 and drawdown: the corrected card, the IHSG, and every
    eligible stock in equal weight.

    Two panels sharing the date axis, never two y-scales on one panel.
    Colours are the dataviz reference palette's slots 1-3, in order. The top
    panel's lines are the per-date MEDIAN across the six calendars, with the
    card's min-max band. The drawdown panel does NOT plot the drawdown of that
    median line: no calendar followed it, and in the first version its
    drawdown understated the median calendar's by about 8 points and the
    worst calendar's by about 18. It shades the range of the six calendars'
    OWN drawdowns instead.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    a, b = (pd.Timestamp(x) for x in res["window"])
    blue, orange, aqua = "#2a78d6", "#eb6834", "#1baf7a"
    ink, ink2, grid, surface = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

    card, ew = frames["card"], frames["ew"]
    idx = index_tr(a, b, card.index)
    card_dd = card / card.cummax() - 1.0

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10, 7.4), sharex=True,
        gridspec_kw={"height_ratios": [2.3, 1.0], "hspace": 0.10})
    fig.patch.set_facecolor(surface)
    for ax in (ax1, ax2):
        ax.set_facecolor(surface)
        ax.grid(True, color=grid, linewidth=0.8)
        ax.tick_params(colors=ink2, labelsize=9)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(grid)

    ax1.fill_between(card.index, card.min(axis=1), card.max(axis=1),
                     color=blue, alpha=0.14, linewidth=0)
    lines = ((card.median(axis=1), blue, 2.0,
              "the card with SL/TP (median of 6 calendars; band = range)"),
             (idx, orange, 2.0, "IHSG total return"),
             (ew.median(axis=1), aqua, 1.6,
              "every eligible stock, equal weight, same costs"))
    for ser, col, lw, lab in lines:
        ax1.plot(ser.index, ser, color=col, linewidth=lw, label=lab)
        ax1.annotate(f"{ser.iloc[-1]:.1f}x", (ser.index[-1], ser.iloc[-1]),
                     xytext=(6, 0), textcoords="offset points",
                     va="center", fontsize=9, color=ink)
    ax1.set_yscale("log")
    ax1.set_ylabel("growth of Rp1 (log scale)", color=ink2, fontsize=9)
    ticks = [0.5, 1, 2, 4, 8, 16]
    ax1.set_yticks(ticks)
    ax1.set_yticklabels([f"{t:g}x" for t in ticks])
    ax1.legend(loc="upper left", frameon=False, fontsize=9, labelcolor=ink)
    ax1.set_title(
        f"The card vs the IHSG, {a:%d %b %Y} to {b:%d %b %Y} "
        f"(corrected costs)", loc="left", fontsize=11, color=ink)

    ax2.fill_between(card_dd.index, card_dd.min(axis=1), card_dd.max(axis=1),
                     color=blue, alpha=0.30, linewidth=0,
                     label="the card: range of the 6 calendars' drawdowns")
    ax2.plot(idx.index, idx / idx.cummax() - 1.0, color=orange,
             linewidth=1.6, label="IHSG total return")
    ax2.set_ylabel("drawdown", color=ink2, fontsize=9)
    ax2.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, _: f"{v:.0%}"))
    #  Anchored right of 2008-09 so the legend never sits on the IHSG's
    #  deepest drawdown.
    ax2.legend(loc="lower left", bbox_to_anchor=(0.13, 0.0), frameon=False,
               fontsize=8, labelcolor=ink)
    fig.text(0.01, 0.005,
             "In-sample (holdout spent at H16). Survivor-only panel: no "
             "delisted names. Fills at the signal close. Half-tick spread, no "
             "impact. 2008-09 is mostly cash (a backtest guard, not a rule).",
             fontsize=7.5, color=ink2)
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=surface)
    plt.close(fig)


# ------------------------------------------------------------------- main --
def _control(res: Dict) -> str:
    """Does the shipped variant reproduce reports/stoptest.json?"""
    try:
        with open(os.path.join("reports", "stoptest.json")) as fh:
            arms = {x["arm"]: x["cagr_med"] for x in json.load(fh)["arms"]}
    except (OSError, ValueError, KeyError, TypeError):
        return "reports/stoptest.json unreadable; control not checked"
    ow = res["own_window_median"]
    want = (arms.get("BASE: quarterly keep-band only"),
            arms.get("SHIPPED: stop 20% + sell HALF at +100%"))
    got = (ow["as_printed/no_levels"], ow["as_printed/card"])
    ok = all(w is not None and abs(w - g) < 1e-12 for w, g in zip(want, got))
    return ("reproduces reports/stoptest.json exactly" if ok else
            f"DIFFERS from reports/stoptest.json {want} -- the panel or the "
            f"rule has moved since it was written, or this is not its harness")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chart", default="")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--chart-only", action="store_true",
                    help="redraw --chart from the saved JSON and curves")
    args = ap.parse_args(argv)

    if args.chart_only:
        with open(OUT) as fh:
            res = json.load(fh)
        D = pd.read_parquet(CURVES)
        frames = {k: D[D["series"] == k].pivot(index="date", columns="offset",
                                               values="equity")
                  for k in ("card", "ew")}
        chart(frames, res, args.chart)
        print(f"chart -> {args.chart}")
        return 0

    runs = run_walks(args.workers)
    res = summarise(runs)
    res["rule"] = {"stop": STOP, "tp": TP, "tp_frac": TP_FRAC,
                   "freq": st.FREQ, "phases": PHASES,
                   "keep": [st.KEEP_HI, st.KEEP_VOL],
                   "entry": [st.ENTRY_HI, st.ENTRY_VOL], "k": st.K}
    res["headline"] = {f"{v}/{a}": headline(res, v, a)
                       for v, d in res["variants"].items() for a in d}
    res["control"] = _control(res)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(res, fh, indent=1)

    ow = res["own_window_median"]
    print(f"control, each calendar's own window: no levels "
          f"{ow['as_printed/no_levels']:.4%}, card "
          f"{ow['as_printed/card']:.4%} -- {res['control']}")
    print(f"window {res['window'][0]} -> {res['window'][1]} "
          f"({res['years']} years), every series on the same dates")
    ix = res["index"]
    print(f"IHSG total return: CAGR {ix['cagr']:+.2%}  maxDD "
          f"{ix['max_dd']:.1%}  halves {ix['halves'][0]:+.2%} / "
          f"{ix['halves'][1]:+.2%}\n")
    for key, h in res["headline"].items():
        extra = ""
        if "excess_ew" in h:
            extra = (f"  vs equal-weight universe {h['excess_ew']:+.2%} "
                     f"({h['calendars_ahead_ew']}/{PHASES}; halves "
                     f"{h['halves_ew'][0]:+.2%} / {h['halves_ew'][1]:+.2%})")
        print(f"{key:<28} CAGR {h['cagr']:+.2%} "
              f"[{h['cagr_lo']:+.2%}..{h['cagr_hi']:+.2%}]  vs IHSG "
              f"{h['excess']:+.2%} ({h['calendars_ahead']}/{PHASES})  maxDD "
              f"{h['max_dd']:.1%} (worst {h['max_dd_worst']:.1%})  halves "
              f"{h['early']:+.2%} / {h['late']:+.2%}{extra}")
    print("\nSUB-WINDOWS, corrected card vs the IHSG and the equal-weight "
          "universe (medians of 6 calendars):")
    for k, r in res["sub_windows"].items():
        print(f"  {k:<24} card {r['card']:+.2%} (maxDD "
              f"{r['card_max_dd']:.1%}, {r['card_ahead']}/{PHASES} ahead)  "
              f"IHSG {r['ihsg']:+.2%} (maxDD {r['ihsg_max_dd']:.1%})  "
              f"equal-weight {r['ew_universe']:+.2%}")
    print()
    t = table(res, "corrected", "card")
    print("THE CARD WITH SL/TP, corrected costs, year by year "
          "(median of 6 calendars, min..max, calendars ahead of the IHSG):")
    for y, r in t.iterrows():
        print(f"  {y}  {r['median']:+7.1%}  [{r['lo']:+7.1%} .. "
              f"{r['hi']:+7.1%}]   IHSG {r['ihsg_tr']:+7.1%}  "
              f"{int(r['ahead_n'])}/{PHASES}")
    frames = chart_frames(runs, res)
    pd.concat([f.stack().rename("equity").rename_axis(["date", "offset"])
               .reset_index().assign(series=k) for k, f in frames.items()]
              ).to_parquet(CURVES, index=False)
    if args.chart:
        chart(frames, res, args.chart)
        print(f"\nchart -> {args.chart}")
    print(f"written {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
