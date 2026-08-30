#!/usr/bin/env python3
"""H51 — the goal, taken literally: >=+4% mean per trade AND >=80% positive.

    python3 scripts/goalsearch.py

WHY H50's GRID COULD NOT HAVE FOUND IT. The required edge is not constant across
barrier geometries. With E[r] = (p - p0)(a + b) - c and p0 = b/(a + b):

    target +15% / stop -15%   p0 0.500   need p 0.800   EDGE NEEDED +0.300
    target +12% / stop -50%   p0 0.806   need p 0.894   EDGE NEEDED +0.087
    target +15% / stop -95%   p0 0.864   need p 0.913   EDGE NEEDED +0.049

A near target against a very far stop needs SIX TIMES LESS edge than a
symmetric bracket, because the payoff per unit of edge is (a + b) and a far
stop makes that large. **H50's grid stopped at a 2-sigma stop and never entered
this region.** That is not a small omission: it is the only part of the space
where the arithmetic is not hostile, and it went untested.

WHAT THIS COSTS, STATED BEFORE ANY CELL IS SCORED. A near target with a far
stop and a long clock is the shape of "hold until I am up X%, cut nothing, wait
years". It buys its win rate with an unbounded left tail and with CAPITAL
LOCKED IN LOSERS. So every cell is reported with its holding period, its
annualised growth and its worst decile attached, and a cell that clears the
goal on per-trade arithmetic while compounding negatively is reported as
clearing the goal AND as not being worth trading. Both, plainly.

BARRIERS ARE ABSOLUTE HERE, NOT VOLATILITY-SCALED. The goal is stated in
absolute terms (+4% per trade), so the search is run in the units the goal is
written in. H50's vol-scaled version is the right tool for comparing across the
cross-section and the wrong one for answering this question.

ONE FORWARD PASS SERVES EVERY LEVEL. The running forward maximum of the high is
monotone in the window length, so the first d at which it clears a level IS the
first touch of that level -- and one scan updates every target and every stop
at once, instead of one scan per (target, stop) cell.

PRE-REGISTERED, WRITTEN BEFORE ANY CELL WAS SCORED
--------------------------------------------------
G1  Some cell of the near-target / far-stop / long-clock region clears BOTH
    halves of the goal on per-trade arithmetic. Predicted: YES -- the required
    edge there is ~5 points and the base rates alone may supply it, because
    P(a liquid name is ever up 12% within three years) is high.
G2  Every cell that clears the goal has NEGATIVE annualised growth, or an
    annualised growth below the index. Predicted: YES. The win rate is bought
    with holding period and with the left tail, and neither is free.
G3  PREDICTED NULL -- the cells that clear the goal are NOT the cells with the
    best annualised growth. If the same cell won both, the goal would be
    picking out something real rather than an artefact of how it is phrased.
G4  A random-selection control at the SAME geometry also clears the goal.
    Predicted: YES, and this is the decisive one -- it would mean the goal is a
    property of the BARRIERS, not of any skill in choosing names.
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

from paint_suite import tick_of                                  # noqa: E402
from quantbot import CACHE, FEE, MIN_RP                          # noqa: E402

OUT = "reports"
TPS = (0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30, 0.40)
SLS = (0.20, 0.30, 0.50, 0.80, 0.95, None)


def touch_times(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                tps: Tuple[float, ...], sls: Tuple, horizon: int):
    """First-touch offset for EVERY target and EVERY stop, in one forward pass.

    The running forward maximum of the high is monotone in the window length,
    so the first offset at which it clears a level is the first touch of that
    level. One scan therefore serves the whole grid; scanning per cell would
    cost forty times as much and return the same numbers.
    """
    n = len(close)
    big = horizon + 1
    t_up = np.full((len(tps), n), big, np.int32)
    t_dn = np.full((len(sls), n), big, np.int32)
    up_lv = np.array([close * (1.0 + a) for a in tps])
    dn_lv = np.array([close * (1.0 - b) if b is not None
                      else np.full(n, -np.inf) for b in sls])
    runmax = np.full(n, -np.inf)
    runmin = np.full(n, np.inf)
    for d in range(1, min(horizon, n - 1) + 1):
        h = np.full(n, -np.inf)
        lo = np.full(n, np.inf)
        h[:n - d] = high[d:]
        lo[:n - d] = low[d:]
        np.maximum(runmax, h, out=runmax)
        np.minimum(runmin, lo, out=runmin)
        for k in range(len(tps)):
            t_up[k] = np.where((t_up[k] > horizon) & (runmax >= up_lv[k]), d,
                               t_up[k])
        for k in range(len(sls)):
            t_dn[k] = np.where((t_dn[k] > horizon) & (runmin <= dn_lv[k]), d,
                               t_dn[k])
    return t_up, t_dn


def build(D: pd.DataFrame, horizon: int) -> Dict[Tuple[int, int], pd.DataFrame]:
    """Label every (target, stop) cell at one horizon."""
    keep = D.groupby("ticker")["rp60"].max()
    names = set(keep[keep >= MIN_RP].index)
    cells: Dict[Tuple[int, int], List] = {}
    for tk, g in D.groupby("ticker", sort=False):
        if tk not in names or len(g) < horizon + 60:
            continue
        p = g["adj"].to_numpy(float)
        hi = g["hi_raw"].to_numpy(float)
        lo = g["lo_raw"].to_numpy(float)
        n = len(p)
        med = float(np.nanmedian(g["close_raw"].to_numpy(float)))
        if not np.isfinite(med) or med <= 0:
            continue
        cost = FEE + tick_of(med) / med
        t_up, t_dn = touch_times(hi, lo, p, TPS, SLS, horizon)
        cen = (np.arange(n) + horizon) > (n - 1)
        ok = (~cen) & (g["rp60"].to_numpy(float) >= MIN_RP) \
            & (g["close_raw"].to_numpy(float) >= 500)
        if ok.sum() < 30:
            continue
        idx = np.flatnonzero(ok)
        for ia, a in enumerate(TPS):
            for ib, b in enumerate(SLS):
                tu, td = t_up[ia][idx], t_dn[ib][idx]
                out = np.where((tu <= horizon) & (tu <= td), 1,
                               np.where((td <= horizon) & (td < tu), -1, 0))
                step = np.minimum(np.minimum(tu, td), horizon)
                j = np.clip(idx + step, 0, n - 1)
                fill = p[j].astype(float)
                fill = np.where(out == 1, p[idx] * (1.0 + a), fill)
                if b is not None:
                    fill = np.where(out == -1,
                                    np.minimum(p[idx] * (1.0 - b), p[j]), fill)
                cells.setdefault((ia, ib), []).append(pd.DataFrame({
                    "date": g["date"].to_numpy()[idx], "ticker": tk,
                    "ret": fill / np.maximum(p[idx], 1e-9) - 1.0 - cost,
                    "bars": step, "outcome": out,
                    "stack": g["stack"].to_numpy(float)[idx]}))
    return {k: pd.concat(v, ignore_index=True) for k, v in cells.items()}


def summarise(E: pd.DataFrame, lab: str) -> Dict:
    r = E["ret"].to_numpy(float)
    bars = max(float(E["bars"].mean()), 1.0)
    ml = float(np.mean(np.log(np.maximum(1.0 + r, 0.01))))
    pos = float((r > 0).mean())
    mean = float(r.mean())
    return {"cell": lab, "n": len(E), "pos": pos, "mean": mean,
            "median": float(np.median(r)), "bars": bars,
            "yrs": bars / 252.0,
            "ann": float(np.exp(ml * 252.0 / bars) - 1.0),
            "p10": float(np.percentile(r, 10)),
            "worst": float(r.min()),
            "timeout": float((E["outcome"] == 0).mean()),
            "GOAL": bool(pos >= 0.80 and mean >= 0.04)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizons", type=int, nargs="+", default=[252, 756])
    ap.add_argument("--tps", type=float, nargs="+", default=None)
    ap.add_argument("--nostop", action="store_true",
                    help="only the far/no-stop arms, for the long horizons")
    a = ap.parse_args()
    global TPS, SLS
    if a.tps:
        TPS = tuple(a.tps)
    if a.nostop:
        SLS = (0.80, None)
    D = pd.read_parquet(CACHE)
    rows = []
    for hor in a.horizons:
        cells = build(D, hor)
        for (ia, ib), E in cells.items():
            tp, sl = TPS[ia], SLS[ib]
            nm = f"tp +{tp:.0%} / sl {'none' if sl is None else f'-{sl:.0%}'}"
            for arm, sub in (("all", E), ("trend", E[E["stack"] >= 2])):
                if len(sub) < 2000:
                    continue
                s = summarise(sub, nm)
                s.update({"hor": hor, "tp": tp,
                          "sl": np.nan if sl is None else sl, "arm": arm})
                rows.append(s)
    F = pd.DataFrame(rows)
    F.to_csv(os.path.join(OUT, "goalsearch.csv"), index=False)
    hit = F[F["GOAL"]]
    print(f"{len(F)} cells over horizons {a.horizons}. "
          f"Absolute barriers, net of fees plus each name's half-spread.\n")
    print(f"{'cell':<26}{'hor':>5}{'arm':>7}{'n':>9}{'POSITIVE':>10}"
          f"{'MEAN':>9}{'median':>9}{'hold yr':>9}{'ann':>8}{'p10':>9}"
          f"{'GOAL':>6}")
    show = F.sort_values(["GOAL", "mean"], ascending=[False, False]).head(24)
    for _, r in show.iterrows():
        print(f"{r['cell']:<26}{int(r['hor']):>5}{r['arm']:>7}{int(r['n']):>9,}"
              f"{r['pos']:>10.1%}{r['mean']:>+9.2%}{r['median']:>+9.2%}"
              f"{r['yrs']:>9.2f}{r['ann']:>+8.1%}{r['p10']:>+9.1%}"
              f"{'YES' if r['GOAL'] else '':>6}")
    print(f"\n  G1 — cells clearing BOTH >=80% positive and >=+4% mean: "
          f"**{len(hit)} of {len(F)}**")
    if len(hit):
        b = hit.loc[hit["ann"].idxmax()]
        print(f"  G2 — of those, best annualised growth {b['ann']:+.2%} "
              f"({b['cell']}, h{int(b['hor'])}, {b['arm']}); "
              f"{int((hit['ann'] > 0).sum())} of {len(hit)} are positive")
        print(f"       mean holding period among goal cells: "
              f"{hit['yrs'].mean():.2f} years; worst-decile trade "
              f"{hit['p10'].mean():+.1%}")
        g3 = F.loc[F["ann"].idxmax()]
        print(f"  G3 — best annualised cell overall is "
              f"{g3['cell']} h{int(g3['hor'])} at {g3['ann']:+.2%}, "
              f"and it {'DOES' if g3['GOAL'] else 'does NOT'} clear the goal")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
