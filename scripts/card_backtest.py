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

TWO HARNESS CORRECTIONS, found by the 2026-11-16 feature inventory and printed
BESIDE the uncorrected numbers rather than replacing them:

  1. ONE ROUND TRIP PER POSITION. `stoptest.walk` charges every sale the full
     round-trip cost on top of the half every purchase pays at the rebalance,
     so each position pays 1.5 round trips. `exit_cost_mult=0.5` charges one.
  2. THE POINT-IN-TIME TICK LADDER. `stoptest.prep` prices the spread with
     `paint_suite.tick_of`, today's ladder, on every bar. Before 2014-01-06 the
     ladder was coarser (src/idxbot/spine/reference.py), so the spread there is
     understated: the lookahead CLAUDE.md section 5 names.

WHAT IT DOES NOT FIX, stated here so no number below is read past it:

  * SURVIVOR-ONLY. The panel holds no delisted names; the delisted store waits
    on the user's licence decision (A51). H54's partial correction moved this
    family by -2.64 to +0.90 points a year, so the sign of the bias on the
    corrected figure is not known.
  * SAME-CLOSE FILLS. Every trade fills at the close of the bar that signalled
    it, which a manual trader acting after 15:50 cannot do.
  * HALF A TICK of spread against OJK's measured ~1.6 ticks (A51), and no
    impact, suspension or auto-rejection term (A23).
  * IN-SAMPLE. The holdout was spent at H16. Six heavily overlapping calendars
    are a robustness check, not a significance test.

USAGE
    python scripts/card_backtest.py [--chart reports/card_backtest.png]
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

#  The levels are IMPORTED from rules.py, never retyped: A47 records two copies
#  of one constant drifting apart, and this file must describe the card that
#  ships, whatever it is when it runs.
ARMS: Dict[str, Dict] = {
    "card": {"stop": STOP, "tp": TP, "tp_frac": TP_FRAC},
    "no_levels": {},
}
VARIANTS: Dict[str, Dict] = {
    "as_printed": {"pit_ticks": False, "exit_cost_mult": 1.0},
    "corrected": {"pit_ticks": True, "exit_cost_mult": 0.5},
}


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


def with_costs(P: pd.DataFrame, pit_ticks: bool) -> pd.DataFrame:
    """The prepared panel with its per-name round-trip cost column."""
    if not pit_ticks:
        return P
    Q = P.copy()
    tick = pit_tick(Q["close"].to_numpy(float), Q["date"].to_numpy())
    Q["cost"] = st.FEE + st.SPREAD_MULT * tick / Q["close"].to_numpy(float)
    return Q


# ------------------------------------------------------------ the windows --
def common_window(curves: List[Tuple[np.ndarray, np.ndarray]]
                  ) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """The dates EVERY calendar was invested over: latest start, first end."""
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


def _one(job: Tuple[str, str, int]) -> Tuple[str, str, int, Dict]:
    variant, arm, offset = job
    v = VARIANTS[variant]
    r = st.walk(_P[variant], offset=offset,
                exit_cost_mult=v["exit_cost_mult"], **ARMS[arm])
    return variant, arm, offset, r


def run_walks(workers: int = 3) -> Dict[Tuple[str, str], List[Dict]]:
    P = st.prep(load())
    for name, v in VARIANTS.items():
        _P[name] = with_costs(P, v["pit_ticks"])
    offs = [int(round(i * st.FREQ / PHASES)) for i in range(PHASES)]
    jobs = [(v, a, o) for v in VARIANTS for a in ARMS for o in offs]
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
                 "variants": {}}
    for (variant, arm), rs in runs.items():
        per = []
        for r in rs:
            if "curve" not in r:
                continue
            s = clip(*r["curve"], a, b)
            #  Every calendar shares the window, so every curve must share its
            #  dates; a mismatch here is a window error, not a rounding one.
            assert s.index.equals(on), "calendars disagree on the dates"
            yr = year_returns(s)
            iy = year_returns(idx)
            h = halves(s)
            per.append({"offset": r["offset"], "cagr": cagr(s),
                        "max_dd": max_dd(s), "halves": list(h),
                        "excess": cagr(s) - res["index"]["cagr"],
                        "years_ahead": int((yr > iy).sum()),
                        "n_years": int(len(yr)),
                        "worst_year": float(yr.min()),
                        "by_year": {int(y): v for y, v in yr.items()},
                        "n_stop": int(r["n_stop"]), "n_tp": int(r["n_tp"])})
        res["variants"].setdefault(variant, {})[arm] = per
        #  POSITIVE CONTROL. Over each calendar's OWN window, the uncorrected
        #  variant must reproduce reports/stoptest.json's medians exactly;
        #  if not, this is not the harness the card quotes.
        res.setdefault("own_window_median", {})[f"{variant}/{arm}"] = float(
            np.median([r["cagr"] for r in rs if "cagr" in r]))
    return res


def table(res: Dict, variant: str, arm: str) -> pd.DataFrame:
    per = res["variants"][variant][arm]
    yrs = sorted(per[0]["by_year"])
    rows = []
    for y in yrs:
        v = np.array([p["by_year"][y] for p in per])
        rows.append({"year": y, "median": float(np.median(v)),
                     "lo": float(v.min()), "hi": float(v.max()),
                     "ihsg_tr": res["index"]["by_year"][y]})
    return pd.DataFrame(rows).set_index("year")


def headline(res: Dict, variant: str, arm: str) -> Dict:
    per = res["variants"][variant][arm]

    def med(k, i=None):
        return float(np.median([p[k] if i is None else p[k][i] for p in per]))

    return {"cagr": med("cagr"),
            "cagr_lo": min(p["cagr"] for p in per),
            "cagr_hi": max(p["cagr"] for p in per),
            "max_dd": med("max_dd"), "early": med("halves", 0),
            "late": med("halves", 1), "excess": med("excess"),
            "calendars_ahead": sum(p["excess"] > 0 for p in per),
            "years_ahead": med("years_ahead"), "n_years": per[0]["n_years"],
            "worst_year": med("worst_year")}


# ------------------------------------------------------------------ chart --
def chart(runs, res: Dict, path: str) -> None:
    """Growth of Rp1 and drawdown, the corrected card against the index.

    Two panels sharing the date axis, never two y-scales on one panel. The
    card is the median calendar with the min-max band of all six; colours are
    the dataviz reference palette's slots 1-3, in order.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    a, b = (pd.Timestamp(x) for x in res["window"])
    blue, orange, aqua = "#2a78d6", "#eb6834", "#1baf7a"
    ink, ink2, grid, surface = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

    def band(arm):
        C = pd.concat([clip(*r["curve"], a, b) for r in
                       runs[("corrected", arm)]], axis=1)
        return C.median(axis=1), C.min(axis=1), C.max(axis=1)

    card_m, card_lo, card_hi = band("card")
    base_m, _, _ = band("no_levels")
    idx = index_tr(a, b, card_m.index)

    def dd(s):
        return s / s.cummax() - 1.0

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10, 7.2), sharex=True,
        gridspec_kw={"height_ratios": [2.4, 1.0], "hspace": 0.08})
    fig.patch.set_facecolor(surface)
    for ax in (ax1, ax2):
        ax.set_facecolor(surface)
        ax.grid(True, color=grid, linewidth=0.8)
        ax.tick_params(colors=ink2, labelsize=9)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(grid)

    ax1.fill_between(card_m.index, card_lo, card_hi, color=blue, alpha=0.14,
                     linewidth=0)
    ax1.plot(base_m.index, base_m, color=aqua, linewidth=1.6,
             label="card without SL/TP")
    ax1.plot(card_m.index, card_m, color=blue, linewidth=2.0,
             label="card with SL/TP (shipped)")
    ax1.plot(idx.index, idx, color=orange, linewidth=2.0,
             label="IHSG total return")
    ax1.set_yscale("log")
    ax1.set_ylabel("growth of Rp1 (log scale)", color=ink2, fontsize=9)
    ticks = [0.5, 1, 2, 4, 8, 16]
    ax1.set_yticks(ticks)
    ax1.set_yticklabels([f"{t:g}x" for t in ticks])
    for s, c in ((card_m, blue), (base_m, aqua), (idx, orange)):
        ax1.annotate(f"{s.iloc[-1]:.1f}x", (s.index[-1], s.iloc[-1]),
                     xytext=(6, 0), textcoords="offset points",
                     va="center", fontsize=9, color=ink)
    ax1.legend(loc="upper left", frameon=False, fontsize=9,
               labelcolor=ink)
    ax1.set_title(
        f"The card vs the IHSG, {a:%d %b %Y} to {b:%d %b %Y} "
        f"(corrected costs; band = 6 rebalance calendars)",
        loc="left", fontsize=11, color=ink)

    ax2.plot(card_m.index, dd(card_m), color=blue, linewidth=1.6)
    ax2.plot(idx.index, dd(idx), color=orange, linewidth=1.6)
    ax2.set_ylabel("drawdown", color=ink2, fontsize=9)
    ax2.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, _: f"{v:.0%}"))
    fig.text(0.01, 0.005,
             "In-sample (holdout spent at H16). Survivor-only panel: no "
             "delisted names. Fills at the signal close. Half-tick spread, "
             "no impact.", fontsize=8, color=ink2)
    fig.savefig(path, dpi=200, bbox_inches="tight",
                facecolor=surface)
    plt.close(fig)


# ------------------------------------------------------------------- main --
def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chart", default="")
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args(argv)

    runs = run_walks(args.workers)
    res = summarise(runs)
    res["rule"] = {"stop": STOP, "tp": TP, "tp_frac": TP_FRAC,
                   "freq": st.FREQ, "phases": PHASES,
                   "keep": [st.KEEP_HI, st.KEEP_VOL],
                   "entry": [st.ENTRY_HI, st.ENTRY_VOL], "k": st.K}
    res["headline"] = {f"{v}/{a}": headline(res, v, a)
                       for v in VARIANTS for a in ARMS}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(res, fh, indent=1)

    ow = res["own_window_median"]
    print(f"control, each calendar's own window (stoptest.json convention): "
          f"no levels {ow['as_printed/no_levels']:.4%}, card "
          f"{ow['as_printed/card']:.4%}")
    print(f"window {res['window'][0]} -> {res['window'][1]} "
          f"({res['years']} years), every calendar and the index on the "
          f"same dates")
    ix = res["index"]
    print(f"IHSG total return: CAGR {ix['cagr']:+.2%}  maxDD "
          f"{ix['max_dd']:.1%}  halves {ix['halves'][0]:+.2%} / "
          f"{ix['halves'][1]:+.2%}\n")
    for v in VARIANTS:
        for a in ARMS:
            h = res["headline"][f"{v}/{a}"]
            print(f"{v:<11} {a:<10} CAGR {h['cagr']:+.2%} "
                  f"[{h['cagr_lo']:+.2%}..{h['cagr_hi']:+.2%}]  "
                  f"vs IHSG {h['excess']:+.2%}  ahead in "
                  f"{h['calendars_ahead']}/{PHASES} calendars  maxDD "
                  f"{h['max_dd']:.1%}  halves {h['early']:+.2%} / "
                  f"{h['late']:+.2%}")
    print()
    t = table(res, "corrected", "card")
    t["ahead"] = np.where(t["median"] > t["ihsg_tr"], "yes", "")
    print("THE CARD WITH SL/TP, corrected costs, year by year "
          "(median of 6 calendars, min..max):")
    for y, r in t.iterrows():
        print(f"  {y}  {r['median']:+7.1%}  [{r['lo']:+7.1%} .. "
              f"{r['hi']:+7.1%}]   IHSG {r['ihsg_tr']:+7.1%}  {r['ahead']}")
    if args.chart:
        chart(runs, res, args.chart)
        print(f"\nchart -> {args.chart}")
    print(f"written {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
