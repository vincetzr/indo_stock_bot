#!/usr/bin/env python3
"""H34/H35 — are support, resistance and Fibonacci real, and what bracket wins?

    python3 scripts/levels.py fib       # H34a  are the Fibonacci ratios special?
    python3 scripts/levels.py sr        # H34b  is a prior swing extreme a barrier?
    python3 scripts/levels.py bracket   # H35   the (take-profit, stop) grid

WHY THE PLACEBO IS THE WHOLE DESIGN.

"Price bounced off the 61.8% retracement" is not evidence, because price bounces
off lots of things and a shallow retracement is reached by every pullback while a
deep one is reached only by big ones. Any test that compares 0.618 to 0.90 is
comparing depths, not ratios, and will always find the shallow level "stronger".

So the ratio grid here is FINE and CONTINUOUS — 0.15 to 0.95 in steps of 0.025 —
and the question is not "does 0.618 work" but **"does 0.618 stand out from 0.60
and 0.65"**. Those neighbours are matched on depth almost exactly, and they are
not Fibonacci numbers. If the ratios carry information the curve has bumps at
them; if they do not, the curve is smooth and the folklore is a smooth function
being read at five arbitrary points.

The same logic runs the support/resistance test: the placebo is the SAME
construction against the WRONG level — a prior pivot high displaced 7% — so both
events are "price first closed above a level referenced to a past extreme" and
they differ only in whether the level is the real one.

PRE-REGISTERED PREDICTIONS, WRITTEN BEFORE ANY CELL WAS SCORED
--------------------------------------------------------------
L1  (fib) THE FIBONACCI RATIOS ARE NOT SPECIAL. The bounce-rate curve will be
    smooth in the retracement depth and the residual at the five Fibonacci
    ratios will sit inside the distribution of residuals at five randomly drawn
    non-Fibonacci ratios. There is no mechanism: the ratios come from a number
    sequence with no connection to order flow, and unlike a round number or a
    prior high they are not a level anyone can see without first agreeing which
    swing to measure from.
L2  (sr) A PRIOR SWING HIGH IS MILDLY SPECIAL, unlike Fibonacci. Prior extremes
    are visible to everyone without a construction choice, are where resting
    orders accumulate, and the microstructure literature finds real effects at
    them. I expect a small effect — a few percentage points on the false-break
    rate against the displaced placebo — not a large one.
L3  (bracket) NO (take-profit, stop) PAIR BEATS HOLDING to the same horizon on
    mean log return net of cost. H20 already retracted two exit rules that
    looked good on cohort medians, and A15 measured the frontier: a hard stop
    takes P(-50%) to near zero and costs 6-8 points of median return.
L4  (bracket) A SYMMETRIC BRACKET LOSES MORE OFTEN THAN IT WINS. The median
    60-session IDX return is -1.02%, so with equal distances the stop is hit
    first more than half the time. Symmetry is not neutrality on a series whose
    median drifts down.

WHAT WOULD FALSIFY EACH: L1 fails if the Fibonacci residual clears its
permutation null; L2 fails if the true level is indistinguishable from the
displaced one; L3 and L4 fail if any grid cell beats hold on mean log in BOTH
halves, which is the only replication test this repo trusts.
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))
sys.path.insert(0, os.path.dirname(__file__))

from time_price import (MIN_BARS, eligible, first_passage,       # noqa: E402
                        load)

OUT = "reports"
SEED = 20260828
ZZ = 0.10                 # ZigZag threshold defining a swing
FWD = 20                  # bars over which an "it held" claim is settled
REGAIN = 60               # bars allowed to regain the leg high
DRAWS = 2000

RATIOS = np.round(np.arange(0.150, 0.951, 0.025), 3)
FIB = (0.236, 0.382, 0.500, 0.618, 0.786)
#  The grid step is 0.025, so a Fibonacci ratio is "hit" by the nearest node.
FIB_NODES = tuple(float(RATIOS[np.argmin(np.abs(RATIOS - f))]) for f in FIB)

TPS = (0.05, 0.10, 0.15, 0.20, 0.30, 0.50)
SLS = (0.05, 0.10, 0.15, 0.20, 0.30)
COST = 0.0056
HORIZON = 252


# ================================================= pivots, with confirmation ==
def pivots_confirmed(p: np.ndarray, k: float) -> Tuple[np.ndarray, np.ndarray]:
    """(pivot bar, bar at which it became knowable).

    THE LOOK-AHEAD TRAP IN EVERY ZIGZAG STUDY. A ZigZag high at bar i is only a
    high once price has fallen k% from it, which happens at some later bar j.
    Drawing the pivot at i and then using it to make a decision at i+1 uses
    information that did not exist until j. Every level below is gated on the
    CONFIRMATION bar, never the pivot bar.
    """
    n = len(p)
    if n < 3:
        return np.empty(0, np.int64), np.empty(0, np.int64)
    piv: List[int] = []
    conf: List[int] = []
    hi_i = lo_i = 0
    hi = lo = p[0]
    up: object = None
    for i in range(1, n):
        v = p[i]
        if up is None:
            if v > hi:
                hi_i, hi = i, v
            if v < lo:
                lo_i, lo = i, v
            if v <= hi * (1.0 - k):
                piv.append(hi_i)
                conf.append(i)
                up, lo_i, lo = False, i, v
            elif v >= lo * (1.0 + k):
                piv.append(lo_i)
                conf.append(i)
                up, hi_i, hi = True, i, v
            continue
        if up:
            if v > hi:
                hi_i, hi = i, v
            elif v <= hi * (1.0 - k):
                piv.append(hi_i)
                conf.append(i)
                up, lo_i, lo = False, i, v
        else:
            if v < lo:
                lo_i, lo = i, v
            elif v >= lo * (1.0 + k):
                piv.append(lo_i)
                conf.append(i)
                up, hi_i, hi = True, i, v
    return np.asarray(piv, np.int64), np.asarray(conf, np.int64)


# ============================================== H34a — the Fibonacci ratios ===
def fib_events(p: np.ndarray, elig: np.ndarray) -> List[Dict]:
    """One row per (completed up-leg, ratio) the pullback actually reached.

    The leg is low -> high, both confirmed. The retracement level for ratio r is
    high - r*(high - low). The event is the FIRST bar after confirmation at
    which close touches it; anything later in the same pullback is the same
    event seen twice.
    """
    piv, conf = pivots_confirmed(p, ZZ)
    out: List[Dict] = []
    for j in range(1, len(piv)):
        i0, i1, c1 = piv[j - 1], piv[j], conf[j]
        if p[i1] <= p[i0]:
            continue                              # want an UP leg to retrace
        lo, hi = float(p[i0]), float(p[i1])
        span = hi - lo
        if span <= 0 or lo <= 0:
            continue
        #  The window runs from confirmation to the next confirmed pivot: after
        #  that a new leg exists and this one is no longer the reference.
        end = int(conf[j + 1]) if j + 1 < len(conf) else len(p) - 1
        if end - c1 < 5 or not elig[c1]:
            continue
        seg = p[c1:end + 1]
        for r in RATIOS:
            lvl = hi - r * span
            hit = np.flatnonzero(seg <= lvl)
            if not len(hit):
                continue
            t = c1 + int(hit[0])
            if t + FWD >= len(p):
                continue
            #  Did it hold? Two readings: the plain forward return, and whether
            #  the leg high was regained, which is what a buyer at the level
            #  is actually hoping for.
            fwd = p[t + FWD] / p[t] - 1.0
            w = p[t + 1:min(len(p), t + 1 + REGAIN)]
            out.append({"r": float(r), "fwd": float(fwd),
                        "regain": float(len(w) and w.max() >= hi),
                        "deeper": float(len(w) and w.min() <= lvl * 0.90),
                        "depth": float((hi - lvl) / hi)})
    return out


def fib_curve(E: pd.DataFrame) -> pd.DataFrame:
    g = E.groupby("r").agg(n=("fwd", "size"), fwd=("fwd", "mean"),
                           med=("fwd", "median"), regain=("regain", "mean"),
                           deeper=("deeper", "mean"))
    return g.reset_index()


def fib_test(C: pd.DataFrame, col: str, rng) -> Dict:
    """Do the Fibonacci nodes stand out from the smooth trend in depth?

    A quadratic in r absorbs the fact that a deeper retracement is reached less
    often and bounces less reliably. What is left is whether five particular
    ratios sit above their neighbours. The null draws five nodes at random from
    the same grid, so it is matched on everything except being Fibonacci.
    """
    C = C.dropna(subset=[col])
    x, y = C["r"].to_numpy(float), C[col].to_numpy(float)
    X = np.column_stack([np.ones(len(x)), x, x ** 2])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    idx = np.array([int(np.flatnonzero(np.isclose(x, f))[0]) for f in FIB_NODES
                    if np.any(np.isclose(x, f))])
    obs = float(resid[idx].mean())
    k = len(idx)
    null = np.array([resid[rng.choice(len(resid), k, replace=False)].mean()
                     for _ in range(DRAWS)])
    sd = null.std(ddof=1)
    return {"stat": col, "obs": obs, "null_mean": float(null.mean()),
            "null_sd": float(sd), "z": float((obs - null.mean()) / sd) if sd else np.nan,
            "p": float((np.sum(np.abs(null - null.mean()) >= abs(obs - null.mean()))
                        + 1) / (DRAWS + 1))}


# ========================================= H34b — prior swing high as a level =
def sr_events(p: np.ndarray, elig: np.ndarray, tk: str) -> List[Dict]:
    """First close above a confirmed prior swing high, and the same event
    against that high displaced +/-7% — the same construction, the wrong level."""
    piv, conf = pivots_confirmed(p, ZZ)
    out: List[Dict] = []
    for j in range(len(piv)):
        if j > 0 and p[piv[j]] <= p[piv[j - 1]]:
            continue                              # want a swing HIGH
        c = int(conf[j])
        base = float(p[piv[j]])
        if base <= 0 or not elig[c]:
            continue
        #  A year to break the level. The first version ran only to the next
        #  confirmed pivot, which is the down-leg plus a 10% bounce -- price
        #  almost never regains an old high inside that, and the test had 1,038
        #  events instead of tens of thousands. A window too short to contain
        #  the event is not a null result, it is no test.
        end = min(len(p) - 1, c + HORIZON)
        for tag, mult in (("true", 1.00), ("placebo -7%", 0.93),
                          ("placebo +7%", 1.07)):
            lvl = base * mult
            seg = p[c:end + 1]
            hit = np.flatnonzero(seg > lvl)
            if not len(hit):
                continue
            t = c + int(hit[0])
            if t + FWD >= len(p):
                continue
            w = p[t + 1:t + 1 + FWD]
            out.append({"kind": tag, "ticker": tk, "bar": int(t),
                        #  How far the level sat above the price on the day it
                        #  became knowable. THE CONFOUND THIS TEST LIVES OR
                        #  DIES ON: a displaced level is crossed at a different
                        #  point in the rally, so true and placebo are only
                        #  comparable within a distance band.
                        "dist": float(lvl / p[c] - 1.0),
                        "fwd": float(p[t + FWD] / p[t] - 1.0),
                        #  A "false break" is closing back under the level it
                        #  just cleared, which is the thing a breakout trader
                        #  is exposed to.
                        "false_break": float(w.min() < lvl),
                        "follow5": float(w.max() >= lvl * 1.05)})
    return out


def sr_adjusted(S: pd.DataFrame, col: str, rng, draws: int = 500) -> Dict:
    """The true-level effect with the distance confound regressed out.

    THE POOLED ROW CANNOT BE READ. A level displaced downward is crossed early
    in a rally and one displaced upward late, so true and placebo events sit at
    different points in the move. A linear probability model in the distance
    (and its square) absorbs that, and the coefficient on `is_true` is what is
    left. Reported in percentage points because that is the unit of the claim.

    The interval RESAMPLES TICKERS, not rows. One name contributes many events
    from overlapping legs, so an iid row bootstrap would understate the width —
    this repo has made that mistake twice and both times the interval came out
    several times too narrow.
    """
    S = S.dropna(subset=[col, "dist"])
    names = S["ticker"].unique()

    def fit(d: pd.DataFrame) -> float:
        X = np.column_stack([np.ones(len(d)), (d["kind"] == "true").astype(float),
                             d["dist"], d["dist"] ** 2])
        b, *_ = np.linalg.lstsq(X, d[col].to_numpy(float), rcond=None)
        return float(b[1])

    obs = fit(S)
    idx = {n: g for n, g in S.groupby("ticker")}
    boot = []
    for _ in range(draws):
        pick = rng.choice(names, size=len(names), replace=True)
        boot.append(fit(pd.concat([idx[n] for n in pick], ignore_index=True)))
    boot = np.asarray(boot)
    return {"stat": col, "effect_pp": 100 * obs,
            "lo_pp": 100 * float(np.percentile(boot, 2.5)),
            "hi_pp": 100 * float(np.percentile(boot, 97.5)),
            "n": len(S)}


# ================================================= H35 — the bracket grid =====
def bracket(P: pd.DataFrame) -> pd.DataFrame:
    """Which barrier is hit first, and what a bracket order would have earned.

    Costs are charged on every path, including the timeout, because every path
    is a round trip. A bracket that "avoids" a trade does not exist: the entry
    already happened.
    """
    P = P.copy()
    P["elig"] = eligible(P)
    rows: List[pd.DataFrame] = []
    for tk, g in P.groupby("ticker", sort=False):
        if len(g) < MIN_BARS:
            continue
        p = g["adj_close"].to_numpy(float)
        n = len(p)
        el = g["elig"].to_numpy()
        yr = pd.DatetimeIndex(g["date"]).year.to_numpy()
        vol = (g["adj_close"].pct_change().rolling(60, min_periods=60)
               .std().to_numpy())
        j = np.arange(n) + HORIZON
        term = np.full(n, np.nan)
        ok = j < n
        term[ok] = p[j[ok]] / p[ok] - 1.0
        base = el & np.isfinite(term)
        if not base.any():
            continue
        up = {t: first_passage(p, 1.0 + t, HORIZON) for t in TPS}
        dn = {s: first_passage(p, 1.0 - s, HORIZON) for s in SLS}
        for t in TPS:
            a = up[t]
            for s in SLS:
                b = dn[s]
                #  -1 means "never touched inside the horizon"; make it +inf so
                #  the comparison below is a plain race between two arrival
                #  times rather than a special case at every branch.
                ta = np.where(a > 0, a, np.inf)
                tb = np.where(b > 0, b, np.inf)
                #  FILL AT THE ACTUAL CLOSE, NOT AT THE LEVEL. A stop is not
                #  filled at the number you wrote on it: the bar that breaches
                #  -5% often closes at -8%, and on IDX a name can gap to ARB
                #  and be untradeable all day. Crediting the bracket with its
                #  nominal level flatters exactly the tight stops that win this
                #  grid, so both barriers exit at the close of the bar that
                #  breached them and the overstatement is removed rather than
                #  argued about.
                first = np.minimum(ta, tb)
                fin = np.isfinite(first)
                exit_i = np.where(fin, np.arange(n) + np.where(fin, first, 0),
                                  0).astype(np.int64)
                exit_i = np.clip(exit_i, 0, n - 1)
                real = p[exit_i] / p - 1.0
                res = np.where(fin, real, term)
                #  EXPOSURE IS NOT FREE AND MUST BE REPORTED. A bracket that
                #  stops out on day 30 sits in cash for the remaining 222, so
                #  comparing its return to a full-horizon hold compares two
                #  different amounts of time invested.
                held = np.minimum(np.minimum(ta, tb), float(HORIZON))
                sel = base & np.isfinite(res)
                rows.append(pd.DataFrame({
                    "tp": t, "sl": s, "year": yr[sel], "ret": res[sel],
                    "bars": held[sel], "vol": vol[sel],
                    "hit": np.where(ta[sel] < tb[sel], "tp",
                                    np.where(tb[sel] < ta[sel], "sl", "time"))}))
        rows.append(pd.DataFrame({"tp": np.nan, "sl": np.nan, "year": yr[base],
                                  "ret": term[base], "bars": float(HORIZON),
                                  "vol": vol[base], "hit": "hold"}))
    return pd.concat(rows, ignore_index=True)


def fit_race(B: pd.DataFrame) -> Dict[str, float]:
    """P(the target is reached before the stop | one of them is).

    THE QUESTION A BRACKET ACTUALLY ASKS, and it is not answered by the two
    touch probabilities separately: both barriers are usually reachable inside a
    year, so what decides the trade is which arrives first. Fitted on the cells
    of the grid crossed with volatility deciles, in the two log distances and
    log sigma.
    """
    R = B[B["hit"].isin(("tp", "sl")) & B["vol"].notna()].copy()
    R["vd"] = pd.qcut(R["vol"], 10, labels=False, duplicates="drop")
    g = R.groupby(["tp", "sl", "vd"]).agg(
        p=("hit", lambda x: float((x == "tp").mean())), n=("hit", "size"),
        sig=("vol", "median")).reset_index()
    g = g[(g["n"] >= 500) & g["p"].between(1e-3, 1 - 1e-3)]
    X = np.column_stack([np.ones(len(g)), np.log(g["tp"]), np.log(g["sl"]),
                         np.log(g["sig"])])
    y = np.log(g["p"] / (1 - g["p"])).to_numpy()
    w = np.sqrt(g["n"].to_numpy(float))
    b, *_ = np.linalg.lstsq(X * w[:, None], y * w, rcond=None)
    pred = 1.0 / (1.0 + np.exp(-(X @ b)))
    err = np.abs(pred - g["p"].to_numpy())
    return {"a": b[0], "b_up": b[1], "c_dn": b[2], "e_vol": b[3],
            "cells": len(g), "med_err_pp": 100 * float(np.median(err)),
            "p90_err_pp": 100 * float(np.quantile(err, 0.9))}


def hold_curve(P: pd.DataFrame, horizons: Tuple[int, ...]) -> pd.DataFrame:
    """Plain buy-and-hold at a range of fixed horizons.

    THE CONTROL THE BRACKET GRID CANNOT BE READ WITHOUT. The best cell on mean
    log is invested 52 sessions out of 252 and holds cash for the other 200. On
    a market whose per-name yearly log return is negative, being out of it is
    most of what a stop does — so a bracket must be compared to a hold of the
    SAME DURATION, not to a hold of four times the duration.
    """
    P = P.copy()
    P["elig"] = eligible(P)
    rows: List[pd.DataFrame] = []
    for tk, g in P.groupby("ticker", sort=False):
        if len(g) < MIN_BARS:
            continue
        p = g["adj_close"].to_numpy(float)
        n = len(p)
        el = g["elig"].to_numpy()
        yr = pd.DatetimeIndex(g["date"]).year.to_numpy()
        for h in horizons:
            j = np.arange(n) + h
            ok = (j < n) & el
            if not ok.any():
                continue
            r = p[j[ok]] / p[ok] - 1.0 - COST
            rows.append(pd.DataFrame({"h": h, "year": yr[ok], "ret": r}))
    H = pd.concat(rows, ignore_index=True)
    H["lg"] = np.log1p(np.clip(H["ret"], -0.99, None))
    return H.groupby("h").agg(n=("ret", "size"), mean=("ret", "mean"),
                              median=("ret", "median"),
                              meanlog=("lg", "mean")).reset_index()


def bracket_summary(B: pd.DataFrame) -> pd.DataFrame:
    B = B.copy()
    B["net"] = B["ret"] - COST
    B["lg"] = np.log1p(np.clip(B["net"], -0.99, None))
    cut = int(B["year"].median())
    out: List[Dict] = []
    for (t, s), g in B.groupby(["tp", "sl"], dropna=False):
        e, l = g[g["year"] < cut], g[g["year"] >= cut]
        out.append({
            "tp": t, "sl": s, "n": len(g),
            "p_tp": float((g["hit"] == "tp").mean()),
            "p_sl": float((g["hit"] == "sl").mean()),
            "bars": float(g["bars"].mean()),
            "mean": float(g["net"].mean()), "median": float(g["net"].median()),
            "meanlog": float(g["lg"].mean()),
            "early": float(e["lg"].mean()), "late": float(l["lg"].mean())})
    R = pd.DataFrame(out)
    #  A PER-TRADE FIGURE IS NOT A YEARLY ONE. A21 records this repo quoting a
    #  ten-year doubling rate as if it were annual; the same mistake is
    #  available here, because a bracket that exits in 52 sessions is redeployed
    #  about five times a year. Annualising by the cell's own holding period is
    #  the only comparable column, and it is what the reader will act on.
    R["ann"] = R["meanlog"] * (252.0 / R["bars"])
    R["ann_early"] = R["early"] * (252.0 / R["bars"])
    R["ann_late"] = R["late"] * (252.0 / R["bars"])
    return R.sort_values("ann", ascending=False)


# ==================================================================== main ====
def main(argv: List[str]) -> int:
    modes = argv or ["fib", "sr", "bracket"]
    P = load()
    P["elig"] = eligible(P)
    rng = np.random.default_rng(SEED)
    print(f"panel {len(P):,} rows  {P['ticker'].nunique()} names\n")

    if "fib" in modes:
        ev: List[Dict] = []
        for tk, g in P.groupby("ticker", sort=False):
            if len(g) < MIN_BARS:
                continue
            ev.extend(fib_events(g["adj_close"].to_numpy(float),
                                 g["elig"].to_numpy()))
        E = pd.DataFrame(ev)
        C = fib_curve(E)
        C.to_csv(os.path.join(OUT, "levels_fib_curve.csv"), index=False)
        print(f"=== H34a Fibonacci: {len(E):,} (leg, ratio) touches, "
              f"{len(C)} ratios from {RATIOS[0]:.3f} to {RATIOS[-1]:.3f}")
        print("ratio  n        P(regain leg high)  fwd20 mean   fwd20 med   "
              "P(-10% more)   fib?")
        for _, r in C.iterrows():
            mark = "  <-- FIB" if np.any(np.isclose(r["r"], FIB_NODES)) else ""
            print(f"{r['r']:.3f}  {int(r['n']):>7,}      {r['regain']:>8.4f}"
                  f"     {r['fwd']:>8.4f}    {r['med']:>8.4f}"
                  f"      {r['deeper']:>7.4f}{mark}")
        print("\n  do the Fibonacci nodes stand out from their neighbours?")
        T = pd.DataFrame([fib_test(C, c, rng)
                          for c in ("regain", "fwd", "med", "deeper")])
        T.to_csv(os.path.join(OUT, "levels_fib_test.csv"), index=False)
        print(T.to_string(index=False, float_format=lambda v: f"{v:,.4f}"))
        print()

    if "sr" in modes:
        ev = []
        for tk, g in P.groupby("ticker", sort=False):
            if len(g) < MIN_BARS:
                continue
            ev.extend(sr_events(g["adj_close"].to_numpy(float),
                                g["elig"].to_numpy(), str(tk)))
        S = pd.DataFrame(ev)
        S = S.merge(P[["ticker", "date"]].assign(bar=P.groupby("ticker").cumcount()),
                    on=["ticker", "bar"], how="left")
        S["year"] = pd.DatetimeIndex(S["date"]).year
        agg = dict(n=("fwd", "size"), dist=("dist", "median"),
                   fwd=("fwd", "mean"), med=("fwd", "median"),
                   false_break=("false_break", "mean"),
                   follow5=("follow5", "mean"))
        G = S.groupby("kind").agg(**agg).reset_index()
        G.to_csv(os.path.join(OUT, "levels_sr.csv"), index=False)
        print("=== H34b prior swing high as a barrier: first close above it,")
        print("    against the same construction on a level displaced 7%")
        print(G.to_string(index=False, float_format=lambda v: f"{v:,.4f}"))
        #  The pooled row is not readable on its own, because a displaced level
        #  is crossed at a different point in the rally. Within a distance band
        #  the two are comparable.
        S["band"] = pd.qcut(S["dist"], 5, labels=False, duplicates="drop")
        Q = S.groupby(["band", "kind"]).agg(**agg).reset_index()
        Q.to_csv(os.path.join(OUT, "levels_sr_bands.csv"), index=False)
        print("\n  stratified by how far the level sat above the price")
        print(Q.to_string(index=False, float_format=lambda v: f"{v:,.4f}"))
        #  THE ONLY REPLICATION TEST THIS REPO TRUSTS. A19/A18 record three
        #  occasions where a within-sample consistency statistic read as
        #  overwhelming and the half-split said nothing replicated.
        yr = S["year"].to_numpy()
        cut = int(np.median(yr))
        rowsA = []
        for c in ("false_break", "follow5", "fwd"):
            r = sr_adjusted(S, c, rng)
            r["early"] = 100 * sr_adjusted(S[S["year"] < cut], c, rng,
                                           draws=1)["effect_pp"] / 100
            r["late"] = 100 * sr_adjusted(S[S["year"] >= cut], c, rng,
                                          draws=1)["effect_pp"] / 100
            rowsA.append(r)
        A = pd.DataFrame(rowsA)
        A.to_csv(os.path.join(OUT, "levels_sr_adjusted.csv"), index=False)
        print("\n  true-level effect, distance-adjusted, ticker-clustered 95% CI")
        print(A.to_string(index=False, float_format=lambda v: f"{v:,.3f}"))
        print()

    if "bracket" in modes:
        B = bracket(P)
        R = bracket_summary(B)
        R.to_csv(os.path.join(OUT, "levels_bracket.csv"), index=False)
        hold = R[R["tp"].isna()].iloc[0]
        print(f"=== H35 the (take-profit, stop) grid, {HORIZON}-session horizon,")
        print(f"    net of {COST:.2%}, {len(B):,} entries. HOLD: "
              f"mean {hold['mean']:+.4f}  median {hold['median']:+.4f}  "
              f"mean log {hold['meanlog']:+.4f}  "
              f"(early {hold['early']:+.4f} late {hold['late']:+.4f})")
        print("    'bars' = mean sessions invested of 252. ANN annualises the "
              "per-trade mean log by 252/bars,")
        print("    which is the only column comparable across cells and to the "
              "index's ~+12.7%/yr.")
        print(f"{'tp':>5}{'sl':>6}{'P(tp1st)':>10}{'P(sl1st)':>10}{'bars':>6}"
              f"{'mean':>9}{'meanlog':>9}{'ANN':>9}{'ann.e':>9}{'ann.l':>9}"
              f"{'both+?':>8}")
        for _, r in R.iterrows():
            if pd.isna(r["tp"]):
                continue
            better = "YES" if (r["ann_early"] > 0 and r["ann_late"] > 0) else ""
            print(f"{r['tp']:>5.2f}{r['sl']:>6.2f}"
                  f"{r['p_tp']:>10.4f}{r['p_sl']:>10.4f}{r['bars']:>6.0f}"
                  f"{r['mean']:>9.4f}{r['meanlog']:>9.4f}{r['ann']:>9.4f}"
                  f"{r['ann_early']:>9.4f}{r['ann_late']:>9.4f}{better:>8}")
        FR = fit_race(B)
        pd.DataFrame([FR]).to_csv(os.path.join(OUT, "levels_race.csv"),
                                  index=False)
        print("\n    P(target first | one of them is reached) = logistic of")
        print("    [1, log tp, log sl, log sigma]  fitted on "
              f"{FR['cells']} (tp, sl, vol-decile) cells")
        print(f"      {FR['a']:+.4f}  {FR['b_up']:+.4f}*log(tp)  "
              f"{FR['c_dn']:+.4f}*log(sl)  {FR['e_vol']:+.4f}*log(sigma)")
        print(f"      median error {FR['med_err_pp']:.2f}pp, "
              f"p90 {FR['p90_err_pp']:.2f}pp")
        HC = hold_curve(P, (10, 20, 30, 40, 50, 60, 80, 100, 130, 160, 200, 252))
        HC.to_csv(os.path.join(OUT, "levels_hold_curve.csv"), index=False)
        print("    the duration-matched control: plain hold at a fixed horizon")
        print(f"{'bars':>6}{'n':>12}{'mean':>9}{'median':>9}{'meanlog':>9}")
        for _, r in HC.iterrows():
            print(f"{int(r['h']):>6}{int(r['n']):>12,}{r['mean']:>9.4f}"
                  f"{r['median']:>9.4f}{r['meanlog']:>9.4f}")
        print("\n    each bracket against a hold of ITS OWN average duration")
        print(f"{'tp':>5}{'sl':>6}{'bars':>6}{'meanlog':>9}"
              f"{'hold same bars':>16}{'edge':>9}")
        for _, r in R.iterrows():
            if pd.isna(r["tp"]):
                continue
            m = float(np.interp(r["bars"], HC["h"], HC["meanlog"]))
            print(f"{r['tp']:>5.2f}{r['sl']:>6.2f}{r['bars']:>6.0f}"
                  f"{r['meanlog']:>9.4f}{m:>16.4f}{r['meanlog'] - m:>9.4f}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
