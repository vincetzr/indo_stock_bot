#!/usr/bin/env python3
"""H52 — beat the index by a mile. The size penalty is the thing in the way.

    python3 scripts/beatindex.py

THE DIAGNOSIS THIS SCRIPT ACTS ON, WHICH THE REPO ALREADY CONTAINS BUT NEVER
JOINED UP.

H43 measured the strength+calm screen beating a RANDOM basket from its own
universe by **+12.78%/yr gross** at a monthly hold — the largest selection
margin anywhere in this project — and landing at +11.97% against the index's
+11.64%. A tie.

A19 measured why: an **equal-weighted** basket of IDX names structurally trails
the **cap-weighted** index, because a handful of mega-caps carry the index and
an equal-weighted basket inherits none of that while keeping the penalty. H43's
own table shows it directly — a random basket from the eligible universe returns
−5.21% to +5.90% while the index returns +11.5% to +14.5%.

Put the two together: **selection alpha ~+12.8%, universe penalty ~−12.5%, and
they cancel.** Every study in this repo has spent the screen's entire edge
paying a structural headwind that has nothing to do with stock picking.

**SO THE UNTESTED MOVE IS TO STOP PAYING IT.** Two levers, neither tried here:

  UNIVERSE   run the screen inside the large-cap end only, where the
             equal-weight-vs-cap-weight gap is small because those names ARE
             most of the index.
  WEIGHTING  weight the basket by size rather than equally, so the portfolio
             is built the way the benchmark is built.

SIZE IS PROXIED BY TRAILING TURNOVER, AND THAT IS A REAL LIMITATION. This repo
has no point-in-time shares outstanding: A25 established that the only share
count available is frozen at 2024-07-10, so using it on a 2010 bar is
look-ahead, and Indonesian rights issues are exactly what makes that wrong.
Trailing 60-day median turnover is causal, is what actually constrains a
tradeable position, and correlates strongly with size — but it is not size, and
a turnover-weighted portfolio is not a cap-weighted one. Stated, not hidden.

PRE-REGISTERED, WRITTEN BEFORE ANY ARM WAS RUN
----------------------------------------------
B1  Restricting the universe to the large-cap end RAISES the screen's return
    relative to the index, because it removes the penalty rather than adding
    skill. Predicted: YES, and the random control inside the same tier should
    rise by a SIMILAR amount — which would prove the gain is the penalty
    disappearing, not the screen improving.
B2  Size-weighting the basket beats equal-weighting it against the index, for
    the same reason. Predicted: YES.
B3  PREDICTED NULL — the screen's edge over its OWN random control, measured
    gross and inside the same tier, is roughly INVARIANT to the tier and the
    weighting. If the edge over random grows when the universe narrows, the
    screen is picking up something about size rather than about strength+calm,
    and the earlier attribution was wrong.
B4  Even at the best (tier, weighting), the screen does NOT beat the index "by
    a mile". Predicted: it lands within a few points, because a +12.8% gross
    edge over random cannot survive both the toll and a benchmark that is
    itself a momentum-weighted portfolio of the same names.
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
from rebalance import (IDX_YIELD, INDEX, MIN_TV, PANEL,          # noqa: E402
                       Prices, half_cagr)

OUT = "reports"
#: how many of the largest-turnover names the universe is cut to. None = all,
#: which is what every earlier study used.
TIERS = (None, 150, 100, 60, 40)
WEIGHTS = ("equal", "sqrt_tv", "tv")


def load() -> pd.DataFrame:
    P = pd.read_parquet(PANEL)
    P = P[P["adj_close"] > 0].sort_values(["ticker", "date"])
    P["tv"] = np.exp(P["log_turnover"].fillna(-np.inf))
    P["elig"] = P["tradeable"].astype(bool) & (P["tv"] >= MIN_TV)
    return P


def universe(d: pd.DataFrame, tier) -> pd.DataFrame:
    """The eligible names, optionally cut to the largest `tier` by turnover.

    The cut is made on the REBALANCE BAR's own trailing turnover, so it is
    point-in-time: a name's later liquidity never decides whether it was in the
    universe today. A full-sample size filter is look-ahead and this repo has
    shipped one before (H44).
    """
    d = d[d["elig"]].dropna(subset=["hi52", "vol60", "tv"])
    if tier is not None and len(d) > tier:
        d = d.nlargest(tier, "tv")
    return d


#: a universe thinner than this is not a cross-section to rank, and a basket
#: narrower than this is not a portfolio. BOTH GUARDS ARE LOAD-BEARING.
#: A first version used MIN_UNIV = 20 and no basket floor, and it returned
#: +17.63% against the index's +11.15% -- a headline that came ENTIRELY from
#: NINE quarters in 2003-04 where the eligible universe was 20-40 names and the
#: "portfolio" was one to three stocks. The selection was identical to
#: `rebalance.py` on the other 78 bars. Nine degenerate baskets at the start of
#: IDX's biggest bull run compounded through a 26-year path and moved the CAGR
#: by six points; the same nine bars lifted the RANDOM control by the same
#: amount, which is what gave it away. A19 records this trap twice: the
#: smallest cell producing the largest effect.
MIN_UNIV = 40
MIN_BASKET = 5


def pick(d: pd.DataFrame, rng=None) -> pd.DataFrame:
    """H26's cell — strength AND calm — inside whatever universe it is given.

    The control is size-matched to the cell and drawn from the SAME tier, so
    tier and selection never move together.
    """
    if len(d) < MIN_UNIV:
        return d.iloc[0:0]
    s = d[(d["hi52"] >= d["hi52"].quantile(SCREEN["hi52_pct"]))
          & (d["vol60"] <= d["vol60"].quantile(SCREEN["vol_pct"]))]
    if len(s) < MIN_BASKET:
        return d.iloc[0:0]
    if rng is not None:
        k = min(max(len(s), 1), len(d))
        return d.iloc[rng.choice(len(d), k, replace=False)]
    return s


def weights(sub: pd.DataFrame, mode: str) -> np.ndarray:
    """Portfolio weights. `equal` is what every earlier study used.

    A cap-weighted benchmark cannot be beaten by an equal-weighted portfolio of
    the same names unless the small ones outperform, and A19 measured that they
    do not. Weighting by turnover builds the portfolio the way the benchmark is
    built; the square root is the usual moderation, keeping the tilt without
    handing the whole book to one name.
    """
    if mode == "equal" or len(sub) == 0:
        return np.full(len(sub), 1.0 / max(len(sub), 1))
    w = sub["tv"].to_numpy(float)
    w = np.sqrt(np.maximum(w, 0.0)) if mode == "sqrt_tv" else np.maximum(w, 0.0)
    return w / w.sum() if w.sum() > 0 else np.full(len(sub), 1.0 / len(sub))


def run(P: pd.DataFrame, PX: Prices, dates: np.ndarray, freq: int, offset: int,
        tier, wmode: str, seed: int | None = None, fee: float = COST,
        spread_mult: float = 1.0) -> Dict:
    """One equity path. Cost is charged on TURNOVER, in weight space.

    THE COST HAS TWO PARTS AND THEY ARE NOT THE SAME KIND OF THING.
    `fee` is commission plus the sell tax — a published schedule, 0.56% round
    trip on the user's broker, and not negotiable. `spread_mult` scales the
    fraksi-harga tick: 1.0 means buy at the ask and sell at the bid, i.e. TAKE
    liquidity on both sides; 0.0 means every fill is passive and earns the
    spread rather than paying it. The truth is in between and depends on how the
    order is worked, so it is a PARAMETER and not a constant. Quoting the two
    blended as "1.4%" — as I did — hides an execution assumption inside what
    looks like a fee.
    """
    rng = np.random.default_rng(seed) if seed is not None else None
    marks = dates[offset::freq]
    if len(marks) < 6:
        return {}
    by_date = {d: g for d, g in P[P["date"].isin(marks)].groupby("date")}
    eq, gross = 1.0, 1.0
    prev: Dict[str, float] = {}
    curve: List[Tuple] = []
    turns, costs, sizes = [], [], []
    for a, b in zip(marks[:-1], marks[1:]):
        sub = pick(universe(by_date.get(a, P.iloc[0:0]), tier), rng)
        if not len(sub):
            curve.append((b, eq))
            continue
        w = weights(sub, wmode)
        tks = list(sub["ticker"])
        cur = dict(zip(tks, w))
        sizes.append(len(tks))
        #  TURNOVER IN WEIGHT SPACE: a name whose weight is unchanged pays
        #  nothing, a name half sold pays half a round trip. Charging the whole
        #  book every rebalance is what made every earlier study of this screen
        #  look worse than it is at high frequency.
        keys = set(cur) | set(prev)
        turn = 0.5 * sum(abs(cur.get(k, 0.0) - prev.get(k, 0.0)) for k in keys)
        px_a = {tk: PX.at(tk, a) for tk in tks}
        toll = [fee + spread_mult * tick_of(v) / v for v in px_a.values()
                if np.isfinite(v) and v > 0]
        toll = float(np.mean(toll)) if toll else fee
        eq *= (1.0 - turn * toll)
        turns.append(turn)
        costs.append(turn * toll)
        rets, ws = [], []
        for tk, wi in zip(tks, w):
            p0 = px_a[tk]
            if not np.isfinite(p0) or p0 <= 0:
                continue
            p1, _dead = PX.exit_price(tk, a, b)
            if not np.isfinite(p1) or p1 <= 0:
                continue
            rets.append(p1 / p0 - 1.0)
            ws.append(wi)
        if rets:
            ws = np.array(ws) / np.sum(ws)
            r = float(np.dot(ws, rets))
            eq *= (1.0 + r)
            gross *= (1.0 + r)
        prev = cur
        curve.append((b, eq))
    if len(curve) < 5:
        return {}
    yrs = (curve[-1][0] - marks[0]).astype(
        "timedelta64[D]").astype(float) / 365.25
    return {"freq": freq, "tier": tier, "w": wmode, "start": marks[0],
            "end": curve[-1][0], "years": yrs, "eq": eq,
            "cagr": eq ** (1.0 / yrs) - 1.0,
            "gross": gross ** (1.0 / yrs) - 1.0,
            "basket": float(np.mean(sizes)) if sizes else np.nan,
            "turnover": float(np.mean(turns)) if turns else np.nan,
            "cost_yr": float(np.sum(costs) / yrs), "curve": curve}


def index_over(a, b) -> float:
    J = pd.read_csv(INDEX, parse_dates=["date"]).sort_values("date")
    J = J[(J["date"] >= pd.Timestamp(a)) & (J["date"] <= pd.Timestamp(b))]
    if len(J) < 50:
        return float("nan")
    yrs = (J["date"].iloc[-1] - J["date"].iloc[0]).days / 365.25
    #  TOTAL-RETURN BASIS. The names run on adj_close and are already total
    #  return; ^JKSE is a PRICE index. A19 records comparing them raw as the
    #  error that manufactured a result.
    return (float(J["close"].iloc[-1] / J["close"].iloc[0])
            ** (1.0 / yrs) - 1.0) + IDX_YIELD


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--freq", type=int, default=63)
    ap.add_argument("--draws", type=int, default=8)
    a = ap.parse_args()
    P = load()
    PX = Prices(P)
    dates = np.sort(P["date"].unique())

    rows = []
    for tier in TIERS:
        for wm in WEIGHTS:
            r = run(P, PX, dates, a.freq, 0, tier, wm)
            if not r:
                continue
            ctl = [run(P, PX, dates, a.freq, 0, tier, wm, seed=s)
                   for s in range(a.draws)]
            ctl = [c for c in ctl if c]
            ic = index_over(r["start"], r["end"])
            e0, e1 = half_cagr(r["curve"])
            r.update({"index": ic, "vs_index": r["cagr"] - ic,
                      "rand": float(np.mean([c["cagr"] for c in ctl])),
                      "rand_sd": float(np.std([c["cagr"] for c in ctl],
                                              ddof=1)) if len(ctl) > 1 else
                      np.nan,
                      "rand_gross": float(np.mean([c["gross"] for c in ctl])),
                      "early": e0, "late": e1})
            rows.append(r)
    F = pd.DataFrame([{k: v for k, v in r.items() if k != "curve"}
                      for r in rows])
    F.to_csv(os.path.join(OUT, "beatindex.csv"), index=False)

    print(f"H52 — the screen inside a size-restricted universe, "
          f"{a.freq}-session rebalance,\ncost on weight-space turnover, "
          f"index on a total-return basis over each arm's own window.\n")
    print(f"{'universe':>10}{'weight':>9}{'names':>7}{'turn':>7}{'cost/yr':>9}"
          f"{'CAGR':>9}{'INDEX':>9}{'vs index':>10}{'random':>9}"
          f"{'GROSS edge':>12}{'early':>9}{'late':>9}")
    for _, r in F.iterrows():
        t = "all" if pd.isna(r["tier"]) else f"top {int(r['tier'])}"
        print(f"{t:>10}{r['w']:>9}{r['basket']:>7.0f}{r['turnover']:>7.0%}"
              f"{r['cost_yr']:>9.2%}{r['cagr']:>+9.2%}{r['index']:>+9.2%}"
              f"{r['vs_index']:>+10.2%}{r['rand']:>+9.2%}"
              f"{r['gross'] - r['rand_gross']:>+12.2%}"
              f"{r['early']:>+9.2%}{r['late']:>+9.2%}")

    best = F.loc[F["vs_index"].idxmax()]
    print(f"\n  B1 — narrowing the universe: vs-index runs "
          + ", ".join(f"{('all' if pd.isna(t) else f'top {int(t)}')} "
                      f"{v:+.2%}"
                      for t, v in F[F['w'] == 'equal'][
                          ['tier', 'vs_index']].to_numpy()))
    print(f"  B2 — best weighting at the best tier: {best['w']}, "
          f"vs index {best['vs_index']:+.2%}")
    print(f"  B3 — gross edge over random by tier (equal weight): "
          + ", ".join(f"{('all' if pd.isna(t) else int(t))} {v:+.1%}"
                      for t, v in F[F['w'] == 'equal'][
                          ['tier', 'gross']].assign(
                          g=lambda x: x['gross']).to_numpy()[:, :2]))
    print(f"  B4 — best arm overall: "
          f"{'all' if pd.isna(best['tier']) else f'top {int(best.tier)}'} / "
          f"{best['w']}, CAGR {best['cagr']:+.2%} against the index at "
          f"{best['index']:+.2%} = {best['vs_index']:+.2%}")
    print(f"       and its half-split: early {best['early']:+.2%}, "
          f"late {best['late']:+.2%}")
    print(f"  arms beating the index at all: "
          f"{int((F['vs_index'] > 0).sum())} of {len(F)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
