#!/usr/bin/env python3
"""H46 — the best allocation this research supports, measured against the index.

    python3 scripts/allocator.py

WHAT THIS IS, AND WHAT IT DELIBERATELY IS NOT.

It is not a trading algorithm. Forty-five hypotheses in, the measured position
is that no rule which TRADES beats owning the asset: 0 of 40 Hull
configurations, 0 of 13 stops, 0 of 55 brackets, 158 exit rules in total, and
a bracket costs -13.06%/yr against simply holding. So this builds the only
thing left standing — a SELECTION and an ALLOCATION, with no exits at all —
and prices it honestly against the alternative of buying the index and going
away.

EVERY COMPONENT IS HERE BECAUSE A SPECIFIC MEASUREMENT PUT IT HERE.

  universe    close >= Rp500 AND 60-bar median turnover >= Rp10bn.
              H44: below this line the rank IC of an indicator against the
              forward return is +0.072 and rises with price STALENESS to
              +0.099; above it, -0.0021 (t -0.15). Everything that looked like
              an edge in the thin tail was a stale price predicting its own
              catch-up. This gate is not a liquidity preference, it is the
              boundary of where any of this was ever real.

  score       equal-weight blend of three cross-sectional percentile ranks:
                strength   hi52   (distance from the 52-week high)
                calm      -vol60  (NEGATED: low volatility scores high)
                momentum   mom12_1
              H26 measured strength+calm at skew 2.60 against a null of
              1.20+/-0.15, z = +9.44, p = 0.00033 — the only cell in this
              project that ever cleared its Bonferroni bar — and H27 then
              confirmed it OUT OF SAMPLE by an unrelated method (gradient
              boosting, purged walk-forward, 15 folds, top-decile skew 2.31).
              Two methods landing within 0.3 of each other is the strongest
              evidence here. `mom12_1` is added because it was the ONLY
              survivor of H44's cost ladder on a tradeable universe (+10.4%
              gross, +8.1% net against a +4.1% pool).

  no exits    158 configurations tested, none beat holding. A take-profit
              raises the win rate to 46.6% and collapses the CAGR from -0.15%
              to -1.98%: you buy the feeling of winning and pay with the right
              tail. There is no stop and no target in this file, on purpose.

  quarterly   H43 measured the selection edge decaying from +12.78% at a
              one-month hold to +2.01% at two years, so this is a
              short-horizon signal — but monthly, quarterly and six-monthly
              were mutually INDISTINGUISHABLE there (differences under ~2.08%
              are inside the phase noise). Quarterly is chosen for the lowest
              turnover among the indistinguishable options, not because it won.

  sizing      the free variable is HOW MUCH, not WHICH. A22 measured that each
              10% of the account moved into a concentrated sleeve costs about
              a point of CAGR; A18 measured that per-POSITION stops can make
              portfolio drawdown WORSE (a -25% stop produced the deepest
              drawdown in its table, -68.7%, by realising losses and
              redeploying into the same regime). So risk is controlled by the
              sleeve weight and nothing else.

THE BENCHMARK IS THE POINT. A19 recorded the missing index comparison as the
error that MANUFACTURED a result, and A33 caught the same omission a second
time. So the index is priced on a TOTAL-return basis over each arm's own
window, and the index leg of the blend is charged a real 50 bp/yr — an
Indonesian retail investor buys the index through an ETF, not for free.

PRE-REGISTERED, WRITTEN BEFORE ANY CELL WAS SCORED
--------------------------------------------------
G1  The 100% sleeve does NOT beat the index. H43 measured a closely related
    screen at +10.98% against the index's +13.20% over a matched window.
G2  Therefore the best blend sits at or near 0% sleeve, and the honest output
    of this file is a recommendation to buy the index.
G3  PREDICTED NULL — the same machine selecting at RANDOM from the identical
    universe must be strictly worse than the real score at every sleeve
    weight. If it is not, the selection is worthless and the blend curve is
    measuring the universe rather than the signal.
G4  The sleeve beats the random control by a positive but SMALLER margin than
    the gap to the index — i.e. the selection is real and still not enough.
    This is the shape every surviving result in this project has had.

EVERYTHING HERE IS IN-SAMPLE. The reserved holdout was spent at H16, and the
three score components were chosen because they worked in this sample. That is
selection on the dependent variable and it inflates the sleeve, not the index.
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

PANEL = os.path.join("data", "spine", "price_panel.parquet")
INDEX = os.path.join("data", "cache", "ohlcv", "_JKSE.csv.gz")
OUT = "reports"

MIN_PRICE = 500.0              # H44's gate
MIN_TV = 1e10                  # H44's gate: Rp 10bn/day, 60-bar median
STEP = 63                      # quarterly
DECILE = 0.10
MIN_BOOK, MAX_BOOK = 8, 40
FEE = 0.0056                   # A5: the user's actual Mandiri round trip
IDX_FEE = 0.0050               # ETF expense ratio on the index leg, per year
IDX_YIELD = 0.0177             # A19: measured large-cap IDX dividend yield
WEIGHTS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
#: the three survivors. `calm` is NEGATED so that low volatility scores high.
LEGS = (("strength", "hi52", +1.0),
        ("calm", "vol60", -1.0),
        ("momentum", "mom12_1", +1.0))


def build(step: int = STEP) -> pd.DataFrame:
    """One row per (rebalance date, ticker): the three legs, and the forward
    return to the next rebalance realised at the name's last print if it dies.
    """
    P = pd.read_parquet(PANEL)
    P = P[P["adj_close"] > 0].sort_values(["ticker", "date"])
    dates = np.sort(P["date"].unique())
    marks = dates[::step]
    mset = set(marks.tolist())

    frames: List[pd.DataFrame] = []
    for tk, g in P.groupby("ticker", sort=False):
        if len(g) < 300:
            continue
        dt = g["date"].to_numpy()
        px = g["adj_close"].to_numpy(float)
        sel = np.flatnonzero(np.isin(dt, list(mset)))
        if len(sel) < 3:
            continue
        nxt = np.searchsorted(marks, dt[sel], side="right")
        fwd = np.full(len(sel), np.nan)
        for j, (i, m) in enumerate(zip(sel, nxt)):
            if m >= len(marks):
                continue
            #  A NAME THAT STOPS PRINTING IS A DELISTING, NOT A HOLD FOREVER.
            #  Realised at its last real close (H41's bug, not repeated).
            e = np.searchsorted(dt, marks[m], side="right") - 1
            if e > i:
                fwd[j] = px[e] / px[i] - 1.0
        d = g.iloc[sel].copy()
        d["fwd"] = fwd
        frames.append(d)
    D = pd.concat(frames, ignore_index=True)

    #  ENTRY eligibility only (A19: never a filter along the forward path).
    D["rp60"] = np.exp(D["log_turnover"].fillna(-np.inf))
    D = D[D["tradeable"].astype(bool)]
    D = D[(D["close"] >= MIN_PRICE) & (D["rp60"] >= MIN_TV)]
    D = D.dropna(subset=["fwd", "hi52", "vol60", "mom12_1"])
    D["tick_bps"] = [10000.0 * tick_of(c) / c
                     for c in D["close"].to_numpy(float)]
    #  A19: `adj_close` is already a TOTAL-return series, so the names carry
    #  their dividends and only the price-index benchmark needs correcting.
    return D.sort_values(["date", "ticker"]).reset_index(drop=True)


def score(D: pd.DataFrame, random_seed: int | None = None) -> pd.Series:
    """Equal-weight blend of the three legs' cross-sectional percentile ranks.

    Ranking WITHIN the date is what makes legs on different scales comparable;
    on raw values whichever leg has the widest units would dominate.
    """
    if random_seed is not None:
        #  THE CONTROL: same machine, same universe, same book size, random
        #  selection. Deterministic from a hash so it is reproducible.
        h = pd.util.hash_pandas_object(
            D["ticker"] + D["date"].astype(str) + str(random_seed),
            index=False).to_numpy()
        return pd.Series((h % 1_000_003) / 1_000_003.0, index=D.index)
    parts = []
    for _, col, sign in LEGS:
        parts.append(sign * D.groupby("date")[col].rank(pct=True))
    return sum(parts) / len(parts)


def sleeve(D: pd.DataFrame, random_seed: int | None = None) -> pd.DataFrame:
    """Per-period sleeve return, net of the toll charged on TURNOVER."""
    S = D.copy()
    S["_s"] = score(D, random_seed)
    prev: set = set()
    rows = []
    for day, g in S.groupby("date", sort=True):
        if len(g) < 40:
            continue
        n = int(np.clip(round(len(g) * DECILE), MIN_BOOK, MAX_BOOK))
        L = g.nlargest(n, "_s")
        held = set(L["ticker"])
        turn = 1.0 - len(held & prev) / max(len(held), 1) if prev else 1.0
        #  the toll is FEES PLUS the point-in-time fraksi-harga half-spread of
        #  the names actually held (A23; omitted once in H44 and it halved the
        #  measured cost)
        toll = FEE + float(L["tick_bps"].mean()) / 10000.0
        rows.append({"date": day, "n": n, "turn": turn,
                     "ret": float(L["fwd"].mean()) - turn * toll,
                     "gross": float(L["fwd"].mean())})
        prev = held
    return pd.DataFrame(rows)


def index_series(dates) -> pd.Series:
    """IHSG on a TOTAL-return basis, net of a real ETF expense ratio."""
    d = pd.read_csv(INDEX)
    d["date"] = pd.to_datetime(d["date"])
    d = d[d["adj_close"] > 0].set_index("date")["adj_close"]
    s = d.reindex(pd.DatetimeIndex(dates), method="ffill")
    r = s.pct_change().dropna()
    per_yr = 252.0 / STEP
    return r + (IDX_YIELD - IDX_FEE) / per_yr


def cagr(r: np.ndarray) -> float:
    lg = np.log(np.clip(1.0 + np.asarray(r, float), 1e-6, None))
    return float(np.exp(lg.mean() * 252.0 / STEP) - 1.0)


def stats(r: np.ndarray) -> Dict:
    r = np.asarray(r, float)
    mid = len(r) // 2
    eq = np.cumprod(1.0 + r)
    dd = float((eq / np.maximum.accumulate(eq) - 1.0).min())
    return {"cagr": cagr(r), "sd": float(r.std(ddof=1) * np.sqrt(252 / STEP)),
            "early": cagr(r[:mid]), "late": cagr(r[mid:]), "maxdd": dd,
            "worst": float(r.min())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=20,
                    help="random-control draws (G3)")
    a = ap.parse_args()

    D = build()
    S = sleeve(D)
    idx = index_series(np.sort(D["date"].unique()))
    #  ALIGN THE WINDOWS. A19: comparing quantities measured over different
    #  windows was the error running through every draft of that section.
    common = S["date"].isin(idx.index)
    S = S[common].reset_index(drop=True)
    I = idx.reindex(S["date"]).to_numpy()
    ok = np.isfinite(I)
    S, I = S[ok].reset_index(drop=True), I[ok]

    print(f"{len(D):,} (period, name) rows, {D['ticker'].nunique()} names, "
          f"{S['date'].nunique()} quarterly rebalances, "
          f"{pd.Timestamp(S['date'].min()):%Y-%m} → "
          f"{pd.Timestamp(S['date'].max()):%Y-%m}")
    print(f"universe: close ≥ Rp{MIN_PRICE:,.0f} and 60-bar turnover ≥ "
          f"Rp{MIN_TV/1e9:,.0f}bn — median {D.groupby('date').size().median():.0f}"
          f" names a quarter, book of {S['n'].median():.0f}")
    print(f"sleeve turnover {S['turn'].mean():.0%} a quarter; toll = "
          f"{FEE:.2%} + fraksi harga; index leg charged {IDX_FEE:.2%}/yr\n")

    R = np.asarray(S["ret"])
    ctrl = [np.asarray(sleeve(D, random_seed=s)
                       .set_index("date").reindex(S["date"])["ret"])
            for s in range(a.draws)]
    ctrl = [c[np.isfinite(c)] for c in ctrl]

    print("=== THE BLEND — sleeve mixed with the index, quarterly")
    print(f"{'sleeve':>8}{'CAGR':>9}{'vol':>8}{'maxDD':>8}{'early':>9}"
          f"{'late':>9}{'vs index':>10}{'random':>9}")
    best, bestw = -9.0, None
    for w in WEIGHTS:
        b = w * R + (1.0 - w) * I
        st = stats(b)
        rc = np.mean([cagr(w * c[:len(I)] + (1.0 - w) * I[:len(c)])
                      for c in ctrl])
        if st["cagr"] > best:
            best, bestw = st["cagr"], w
        print(f"{w:>8.0%}{st['cagr']:>+9.2%}{st['sd']:>8.1%}"
              f"{st['maxdd']:>8.1%}{st['early']:>+9.2%}{st['late']:>+9.2%}"
              f"{st['cagr'] - cagr(I):>+10.2%}{rc:>+9.2%}")

    print(f"\n=== the verdict")
    print(f"  best blend: {bestw:.0%} sleeve at {best:+.2%} a year")
    print(f"  100% index: {cagr(I):+.2%}   100% sleeve: {cagr(R):+.2%}")
    print(f"  sleeve vs random control: "
          f"{cagr(R) - np.mean([cagr(c) for c in ctrl]):+.2%} a year "
          f"(G3/G4)")
    e = np.mean([cagr(c) for c in ctrl])
    print(f"  random sleeve alone: {e:+.2%}")
    sm, im = stats(R), stats(I)
    print(f"  half-split, sleeve minus index: "
          f"early {sm['early'] - im['early']:+.2%}, "
          f"late {sm['late'] - im['late']:+.2%}   "
          f"{'BOTH' if sm['early'] > im['early'] and sm['late'] > im['late'] else 'NOT BOTH'}")
    pd.DataFrame({"date": S["date"], "sleeve": R, "index": I}).to_csv(
        os.path.join(OUT, "allocator.csv"), index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
