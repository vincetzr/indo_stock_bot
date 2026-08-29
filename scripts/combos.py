#!/usr/bin/env python3
"""H45 — which method, or COMBINATION of methods, is best and most reliable?

    python3 scripts/combos.py --null 200

H44 scored the simple methods one at a time. This scores every pair and the
full stack, in the two ways a trader actually combines them:

  AVG  average of the cross-sectional percentile ranks — a blended score, so a
       name can be mediocre on one leg and carried by the other.
  AND  the MINIMUM of the percentile ranks — "confluence": the name has to be
       strong on EVERY leg. This is what people mean by "EMA stack AND
       stochastic strong AND breaking resistance", and it is a different animal
       from a blend, because it concentrates the book and cuts turnover.

WHAT "BEST" AND "RELIABLE" MEAN HERE, FIXED BEFORE ANYTHING WAS SCORED.
A rate is not an objective (H25 -> H26) and neither is a win rate (H40, H42),
so:
  BEST      net CAGR at 56 bp (the user's real toll) MINUS the mean of its own
            clustered permutation null. The null is not zero — a decile
            portfolio of IDX names returns ~+14%/yr with the signal destroyed —
            so a raw CAGR is meaningless on its own.
  RELIABLE  a conjunction, and all four parts must hold:
              (1) z above the Bonferroni bar for the whole trial count
              (2) positive against its own null in BOTH halves
              (3) still positive at the 56 bp retail toll, not just at 0
              (4) not carried by one era or a handful of names
            Anything failing any part is reported as failing it, by name.

THE MULTIPLE-TESTING PROBLEM IS THE WHOLE DANGER AND IT IS COUNTED.
Eight singles plus 28 pairs in two modes plus stacks is ~70 cells in one sweep.
The best of 70 cells is a MAXIMUM, not a measurement — A11 logged exactly this
trap when a 54-cell sweep produced a 1.67% excess that turned out to be the
maximum of the sweep and not a finding. Every number here is reported with the
trial count attached, and the headline cell is explicitly labelled as the
argmax of a sweep.

PRE-REGISTERED, WRITTEN BEFORE ANY CELL WAS SCORED
--------------------------------------------------
K1  CONFLUENCE (AND) BEATS BLENDING (AVG) on the null-adjusted number, because
    the legs are correlated momentum measures and a blend of correlated signals
    adds little while an intersection concentrates on agreement.
K2  The best combination beats the best single by less than its own null sd —
    i.e. combining helps, but not by enough to establish a new winner. The legs
    measure nearly the same thing (all four are trend/strength), so the extra
    information should be small.
K3  PREDICTED NULL — `ema_stack + rand` scores strictly BETWEEN `ema_stack`
    alone and `rand` alone, and `fib_618 + rand` is flat. If pairing a real
    signal with noise does not hurt it, the combination machinery is averaging
    something other than what it claims.
K4  Confluence cuts TURNOVER relative to its legs, so its break-even toll is
    higher even where its gross return is lower. That is the practical reason
    to prefer it and it is separately checkable.
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))
sys.path.insert(0, os.path.dirname(__file__))

from cost_ladder import RULES, breakeven, build, walk         # noqa: E402

OUT = "reports"
#: the legs worth combining: every rule that is not itself a control
LEGS = ("ema_stack", "ema_cross", "stoch_strong", "sr_break", "mom12_1",
        "strength_calm", "fib_618", "stoch_oversold")
#: the four that cleared their own null as singles in H44
CLEARED = ("ema_stack", "ema_cross", "stoch_strong", "sr_break")
MODES = ("avg", "and")


def rank_frame(D: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional percentile rank of every rule, per rebalance date.

    Ranking WITHIN the date is what makes legs on different scales (a 0-3
    ordinal, a log ratio, a 0-100 oscillator) combinable at all. Doing it on
    raw values would let whichever leg has the widest units dominate.
    """
    R = D[["date", "ticker", "fwd"]].copy()
    for r in RULES:
        R[r] = D.groupby("date")[r].rank(pct=True)
    return R


def combo(R: pd.DataFrame, legs: Tuple[str, ...], mode: str) -> pd.Series:
    """AVG = mean of the ranks (a blend). AND = min of the ranks (confluence).

    `min` is the honest continuous form of an intersection: a name scores high
    only if it is high on EVERY leg, and unlike a hard AND it never returns an
    empty book, so the portfolio is comparable across dates.
    """
    X = R[list(legs)]
    return X.mean(axis=1) if mode == "avg" else X.min(axis=1)


def score(R: pd.DataFrame, legs: Tuple[str, ...], mode: str,
          tolls=(0, 25, 56)) -> Dict:
    S = R.copy()
    S["_c"] = combo(R, legs, mode)
    out = {"name": ("+".join(legs) if len(legs) > 1 else legs[0]),
           "mode": mode if len(legs) > 1 else "-", "legs": len(legs)}
    rows = []
    for t in tolls:
        w = walk(S, "_c", t)
        if not w:
            return {}
        out[f"cagr{t}"] = w["cagr"]
        rows.append({"toll": t, "cagr": w["cagr"]})
        if t == 56:
            out["turnover"] = w["turnover"]
            out["early"], out["late"] = w["early"], w["late"]
    out["breakeven"] = breakeven(rows)
    return out


def null_dist(R: pd.DataFrame, legs, mode, draws: int, seed: int,
              toll: int = 56) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Clustered permutation null: whole (ticker, year) blocks of RETURNS are
    reassigned to other blocks' features.

    A25's lesson, learned the hard way: permuting INSIDE a block is nearly a
    no-op because one ticker-year's rows carry near-identical labels, so the
    null preserves the mapping it exists to destroy. And the fill is CYCLIC —
    a truncating version left 29.2% of rows holding their own return, which
    retained signal in the null and made every z conservative.
    """
    S = R.copy()
    S["_c"] = combo(R, legs, mode)
    blk = (S["ticker"].astype(str) + "|"
           + pd.to_datetime(S["date"]).dt.year.astype(str)).to_numpy()
    keys = np.unique(blk)
    idx_of = {k: np.flatnonzero(blk == k) for k in keys}
    fwd = S["fwd"].to_numpy()
    rng = np.random.default_rng(seed)
    full, early, late = [], [], []
    for _ in range(draws):
        perm = rng.permutation(len(keys))
        f2 = np.empty_like(fwd)
        for a, b in zip(keys, keys[perm]):
            ia, ib = idx_of[a], idx_of[b]
            f2[ia] = fwd[ib[np.arange(len(ia)) % len(ib)]]
        T = S.copy()
        T["fwd"] = f2
        w = walk(T, "_c", toll)
        if w:
            full.append(w["cagr"])
            early.append(w["early"])
            late.append(w["late"])
    return np.array(full), np.array(early), np.array(late)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--null", type=int, default=200)
    ap.add_argument("--top", type=int, default=8,
                    help="how many cells get a full null (the expensive part)")
    a = ap.parse_args()

    D = build()
    #  ONE COMMON UNIVERSE FOR EVERY CELL. Different NaN patterns per rule meant
    #  each was scanning a different market, so the arms were not comparable —
    #  found by chasing a "random" control that sat 10 points below the pool it
    #  was drawn from.
    D = D.dropna(subset=list(RULES)).reset_index(drop=True)
    D = D[D.groupby("date")["ticker"].transform("size") >= 40]
    D = D.reset_index(drop=True)
    R = rank_frame(D)
    uni = D.groupby("date")["fwd"].mean()
    print(f"{len(D):,} (period, name) rows, {D['ticker'].nunique()} names, "
          f"{D['date'].nunique()} monthly rebalances, "
          f"{pd.Timestamp(D['date'].min()):%Y-%m} → "
          f"{pd.Timestamp(D['date'].max()):%Y-%m}")
    print(f"equal-weighted universe over the same periods: "
          f"{(np.exp(np.log1p(uni).mean() * 12) - 1) * 100:+.2f}%/yr\n")

    cells: List[Dict] = []
    for leg in LEGS + ("rand",):
        s = score(R, (leg,), "-")
        if s:
            cells.append(s)
    for x, y in itertools.combinations(LEGS, 2):
        for m in MODES:
            s = score(R, (x, y), m)
            if s:
                cells.append(s)
    for m in MODES:
        s = score(R, CLEARED, m)
        if s:
            cells.append(s)
    #  K3's predicted-null pairs: a real signal blended with noise must land
    #  BETWEEN the two, and noise with noise must be flat.
    for pair in (("ema_stack", "rand"), ("fib_618", "rand")):
        for m in MODES:
            s = score(R, pair, m)
            if s:
                cells.append(s)
    C = pd.DataFrame(cells)
    C.to_csv(os.path.join(OUT, "combos.csv"), index=False)
    print(f"{len(C)} cells scored. THE BEST OF {len(C)} CELLS IS A MAXIMUM, "
          f"not a measurement.")
    print(f"Bonferroni bar for this sweep alone: "
          f"{0.05 / len(C):.5f}\n")

    print("=== every cell, ranked by net CAGR at the 56bp RETAIL toll")
    print(f"{'cell':<42}{'mode':>5}{'turn':>6}{'0bp':>9}{'25bp':>9}"
          f"{'56bp':>9}{'early':>9}{'late':>9}{'b/e':>8}")
    for _, r in C.sort_values("cagr56", ascending=False).iterrows():
        be = r["breakeven"]
        bes = ("never" if be == 0 else ">56" if be == float("inf")
               else f"{be:.0f}bp")
        print(f"{r['name']:<42}{r['mode']:>5}{r['turnover']:>6.0%}"
              f"{r['cagr0']:>+9.1%}{r['cagr25']:>+9.1%}{r['cagr56']:>+9.1%}"
              f"{r['early']:>+9.1%}{r['late']:>+9.1%}{bes:>8}")

    #  THE NULL IS THE EXPENSIVE PART, so it runs only on the leaders plus the
    #  controls — and the controls are NOT optional: without them a leader has
    #  nothing to be a leader against.
    want = list(C.sort_values("cagr56", ascending=False).head(a.top)["name"])
    for must in ("rand", "fib_618", "ema_stack+rand", "fib_618+rand"):
        want += [n for n in C["name"] if n == must and n not in want]
    print(f"\n=== clustered (ticker, year) null, {a.null} draws, at 56bp")
    print(f"{'cell':<42}{'mode':>5}{'real':>9}{'null':>9}{'sd':>7}"
          f"{'z':>7}{'p':>8}{'halves vs null':>16}")
    res = []
    for _, r in C.iterrows():
        if r["name"] not in want:
            continue
        legs = tuple(r["name"].split("+")) if r["legs"] > 1 else (r["name"],)
        mode = r["mode"] if r["mode"] != "-" else "avg"
        f, e, l = null_dist(R, legs, mode, a.null, seed=4242)
        if not len(f):
            continue
        z = (r["cagr56"] - f.mean()) / max(f.std(ddof=1), 1e-9)
        p = (np.sum(f >= r["cagr56"]) + 1) / (len(f) + 1)
        de, dl = r["early"] - e.mean(), r["late"] - l.mean()
        res.append({**r.to_dict(), "null": f.mean(), "null_sd": f.std(ddof=1),
                    "z": z, "p": p, "d_early": de, "d_late": dl,
                    "both": bool(de > 0 and dl > 0)})
        print(f"{r['name']:<42}{r['mode']:>5}{r['cagr56']:>+9.1%}"
              f"{f.mean():>+9.1%}{f.std(ddof=1):>7.1%}{z:>+7.2f}{p:>8.4f}"
              f"   {de:>+6.1%}/{dl:>+6.1%} {'BOTH' if de > 0 and dl > 0 else 'no'}")
    if res:
        pd.DataFrame(res).to_csv(os.path.join(OUT, "combos_null.csv"),
                                 index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
