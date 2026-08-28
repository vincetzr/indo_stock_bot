#!/usr/bin/env python3
"""H41 — Rp 50 juta into the Hull rule: the MONTHLY return distribution.

    python3 scripts/monthly.py --capital 50000000 --slots 5

WHY A SINGLE "EXPECTED RETURN PER MONTH" IS THE WRONG ANSWER AND THIS SCRIPT
GIVES A DISTRIBUTION INSTEAD.

H39 measured the rule's win rate at 32.5% and found the best 1% of trades
carrying 71% of the total return. A distribution that lopsided has a mean that
almost never occurs: the typical month is a small loss, and the average is
dragged up by a handful of months nobody can schedule. Quoting the mean alone
would be true and useless.

So this runs an actual account. Rp X of capital, at most N positions at a time,
equal-weighted, entering on the measured rule and exiting on the measured stop,
paying 0.56% a round trip and the fraksi-harga spread, with cash earning nothing
while it waits. Then it reports the month-by-month distribution and the same for
simply owning the index over the identical window.

WHAT THIS CANNOT DO, STATED BEFORE THE NUMBERS.
  * Every figure is IN-SAMPLE. The reserved holdout was spent at H16.
  * The rule has no live track record. Backtested edge is not live edge, and
    slippage, capacity and the fact that this sample is one realisation of
    history all cut the same way.
  * A19 measured that these picks trail the index. This script does not repair
    that; it prices it.

PRE-REGISTERED, WRITTEN BEFORE ANY CELL WAS SCORED
--------------------------------------------------
M1  The MEDIAN month will be near zero or negative and the MEAN will be
    positive, because the trade distribution is right-skewed. If the median is
    negative the honest headline is "most months lose a little".
M2  The account will underperform the index over the same window, following
    H39 (median CAGR +1.13% against +9.88% for owning the same names) and A19
    (the picks trail the index by ~2.2%/yr).
M3  Monthly volatility will be large relative to the mean — a standard
    deviation several times the mean — so a year is far too short to tell skill
    from noise at this signal-to-noise ratio.
"""

#  A18: "a within-sample consistency statistic over correlated units reads as
#  overwhelming and says almost nothing about whether an effect replicates.
#  Only the half-split does. It is cheap and it should run before any rule is
#  reported, not after." So it runs here, on the one comparison that decides
#  this study: the Hull filter against the SAME machine picking at random.

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))
sys.path.insert(0, os.path.dirname(__file__))

from hull_stop import flip_price                                 # noqa: E402
from hull_trade import COST, states                              # noqa: E402
from time_price import MIN_BARS, eligible, load                  # noqa: E402

OUT = "reports"
INDEX = os.path.join("data", "cache", "ohlcv", "_JKSE.csv.gz")
MIN_TV = 1e9
TRAIL = 0.25          # H40's best stop: a plain -25% trail from the peak
MAX_BARS = 252


def signals(P: pd.DataFrame) -> pd.DataFrame:
    """Per (ticker, date): can I enter, and where is the stop."""
    frames: List[pd.DataFrame] = []
    for tk, g in P.groupby("ticker", sort=False):
        if len(g) < MIN_BARS:
            continue
        px = g["px"].reset_index(drop=True)
        up, on = states(px, 55, "EMA stack")
        #  SHIFT THE ENTRY BY ONE BAR. Both conditions are computed from the
        #  CLOSE of bar t, so filling at that same close is a look-ahead: it
        #  buys at a price only known once the bar is finished. H39 filled at
        #  t+1 throughout, and the first run of this script did not -- which is
        #  why it reported the account BEATING the index while H39 measured the
        #  same rule compounding at a tenth of the index's rate. Two studies of
        #  one rule disagreeing is the signal that one of them is wrong.
        sig = np.concatenate([[False], (up & on)[:-1]])
        frames.append(pd.DataFrame({
            "date": g["date"].to_numpy(), "ticker": tk,
            "px": px.to_numpy(float), "enter": sig,
            "elig": g["elig"].to_numpy(),
            "flip": flip_price(px, 55)}))
    return pd.concat(frames, ignore_index=True)


def account(S: pd.DataFrame, capital: float, slots: int,
            seed: int = 20260828, random_entry: bool = False) -> pd.DataFrame:
    """Walk a real book: at most `slots` positions, equal-weighted, cash idle.

    NEW ENTRIES ARE PICKED AT RANDOM FROM THE ELIGIBLE SET on days when more
    names signal than there are free slots. Ranking them by anything would be a
    second, untested selection rule smuggled in beside the one being measured —
    and this repo has watched that turn a null into a headline four times.
    """
    rng = np.random.default_rng(seed)
    dates = np.sort(S["date"].unique())
    by_date = {d: g for d, g in S.groupby("date")}
    cash = capital
    book: Dict[str, Dict] = {}
    rows: List[Dict] = []
    for d in dates:
        g = by_date[d]
        px = dict(zip(g["ticker"], g["px"]))
        flip = dict(zip(g["ticker"], g["flip"]))
        #  mark to market first, so the equity curve is dated correctly
        for tk, pos in book.items():
            if tk in px:
                pos["last"] = px[tk]
                pos["peak"] = max(pos["peak"], px[tk])
                pos["bars"] += 1
                pos["gone"] = 0
            else:
                pos["gone"] += 1
        #  A NAME THAT STOPS PRINTING IS A DELISTING, NOT A HOLD FOREVER. The
        #  panel deliberately contains delisted names (a survivorship-free
        #  universe is the whole point of the spine), so a position whose
        #  ticker disappears must be realised at its last price rather than
        #  carried at that price for the rest of the backtest, blocking a slot
        #  and quietly inflating the book.
        for tk in [t for t, q in book.items() if q["gone"] >= 5]:
            cash += book[tk]["shares"] * book[tk]["last"] * (1.0 - COST / 2.0)
            del book[tk]
        #  exits: the H40 winner, a -25% trail from the running peak, plus the
        #  252-session cap every rule in H40 carried
        for tk in list(book):
            pos = book[tk]
            if tk not in px:
                continue
            stop = pos["peak"] * (1.0 - TRAIL)
            if px[tk] < stop or pos["bars"] >= MAX_BARS:
                cash += pos["shares"] * px[tk] * (1.0 - COST / 2.0)
                del book[tk]
        free = slots - len(book)
        if free > 0:
            #  THE CONTROL THAT DECIDES THE WHOLE STUDY. A book of five IDX
            #  names with a 25% trailing stop, rotating constantly, is an
            #  equal-weighted portfolio with an exit rule -- and A19 measured
            #  that an equal-weighted IDX basket behaves nothing like the
            #  cap-weighted index. So the Hull filter has to be compared to the
            #  SAME MACHINE picking names at random, not to the index. Without
            #  this arm the study cannot tell a selection effect from the
            #  mechanics of equal weighting.
            pool = g["elig"] if random_entry else (g["enter"] & g["elig"])
            cand = g[pool & ~g["ticker"].isin(book)]
            if len(cand):
                pick = cand.sample(n=min(free, len(cand)), random_state=
                                   int(rng.integers(0, 2 ** 31)))
                for _, r in pick.iterrows():
                    size = (cash + sum(p["shares"] * p["last"]
                                       for p in book.values())) / slots
                    size = min(size, cash)
                    if size < 1e6:               # below one lot's worth, skip
                        continue
                    sh = size / (r["px"] * (1.0 + COST / 2.0))
                    cash -= sh * r["px"] * (1.0 + COST / 2.0)
                    book[r["ticker"]] = {"shares": sh, "last": r["px"],
                                         "peak": r["px"], "bars": 0,
                                         "gone": 0}
        eq = cash + sum(p["shares"] * p["last"] for p in book.values())
        rows.append({"date": d, "equity": eq, "cash": cash,
                     "held": len(book), "invested": 1.0 - cash / eq})
    return pd.DataFrame(rows)


def index_curve(dates: np.ndarray) -> pd.Series:
    d = pd.read_csv(INDEX)
    d["date"] = pd.to_datetime(d["date"])
    d = d[d["adj_close"] > 0].set_index("date")["adj_close"]
    return d.reindex(pd.DatetimeIndex(dates), method="ffill")


def monthly(eq: pd.Series) -> pd.Series:
    m = eq.resample("ME").last()
    return m.pct_change().dropna()


def half_cagr(eq: pd.Series) -> tuple:
    """CAGR in the early half and the late half of the SAME window.

    Split by DATE, not by row count, so both arms are scored on identical
    calendar spans. A rule that only works in one half is a regime coincidence,
    which is what A18 found for every exit rule it tested and what killed them.
    """
    mid = eq.index[len(eq) // 2]
    out = []
    for a, b in ((eq.index[0], mid), (mid, eq.index[-1])):
        s = eq.loc[a:b]
        yrs = (b - a).days / 365.25
        out.append(float((s.iloc[-1] / s.iloc[0]) ** (1.0 / yrs) - 1.0))
    return out[0], out[1]


def describe(r: pd.Series, name: str, capital: float) -> Dict:
    return {"series": name, "months": len(r), "mean": float(r.mean()),
            "median": float(r.median()), "sd": float(r.std(ddof=1)),
            "pos": float((r > 0).mean()), "worst": float(r.min()),
            "best": float(r.max()),
            "mean_rp": float(r.mean() * capital),
            "median_rp": float(r.median() * capital),
            "cagr": float((1.0 + r).prod() ** (12.0 / len(r)) - 1.0)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--capital", type=float, default=50e6)
    ap.add_argument("--slots", type=int, default=5)
    ap.add_argument("--draws", type=int, default=12)
    a = ap.parse_args()

    P = load()
    P["elig"] = eligible(P)
    P["px"] = P["adj_close"]
    P = P.sort_values(["ticker", "date"])
    S = signals(P)
    print(f"{len(S):,} name-days, {S['ticker'].nunique()} names, "
          f"{pd.Timestamp(S['date'].min()):%Y-%m-%d} -> "
          f"{pd.Timestamp(S['date'].max()):%Y-%m-%d}")
    print(f"Rp {a.capital:,.0f} across at most {a.slots} positions, "
          f"exit = {TRAIL:.0%} trail from peak, {COST:.2%} a round trip\n")

    #  MANY DRAWS, BECAUSE ONE ACCOUNT IS ONE SAMPLE. Which names a small book
    #  happens to hold is luck, and a single run would report that luck as the
    #  strategy's return.
    curves, rand, inv = [], [], []
    for i in range(a.draws):
        A = account(S, a.capital, a.slots, seed=20260828 + i)
        curves.append(A.set_index("date")["equity"])
        inv.append(float(A["invested"].mean()))
        B = account(S, a.capital, a.slots, seed=77000 + i, random_entry=True)
        rand.append(B.set_index("date")["equity"])
    dates = curves[0].index
    idx = index_curve(np.asarray(dates))

    rows = [describe(monthly(c), f"account draw {i + 1}", a.capital)
            for i, c in enumerate(curves)]
    R = pd.DataFrame(rows)
    RD = pd.DataFrame([describe(monthly(c), f"random draw {i + 1}", a.capital)
                       for i, c in enumerate(rand)])
    ir = describe(monthly(idx), "IHSG buy and hold", a.capital)

    print("=== monthly returns, one row per independent account draw")
    print(f"{'draw':>5}{'months':>8}{'mean':>9}{'median':>9}{'sd':>8}"
          f"{'% up':>7}{'worst':>9}{'best':>9}{'CAGR':>9}")
    for i, r in R.iterrows():
        print(f"{i + 1:>5}{int(r['months']):>8}{r['mean']:>9.2%}"
              f"{r['median']:>9.2%}{r['sd']:>8.2%}{r['pos']:>7.0%}"
              f"{r['worst']:>9.1%}{r['best']:>9.1%}{r['cagr']:>9.2%}")
    print(f"{'RAND':>5}{int(RD['months'].mean()):>8}{RD['mean'].mean():>9.2%}"
          f"{RD['median'].mean():>9.2%}{RD['sd'].mean():>8.2%}"
          f"{RD['pos'].mean():>7.0%}{RD['worst'].min():>9.1%}"
          f"{RD['best'].max():>9.1%}{RD['cagr'].mean():>9.2%}")
    print(f"{'INDEX':>5}{int(ir['months']):>8}{ir['mean']:>9.2%}"
          f"{ir['median']:>9.2%}{ir['sd']:>8.2%}{ir['pos']:>7.0%}"
          f"{ir['worst']:>9.1%}{ir['best']:>9.1%}{ir['cagr']:>9.2%}")

    print(f"\n=== THE ANSWER, on Rp {a.capital:,.0f}")
    print(f"  mean month      {R['mean'].mean():+.2%}   "
          f"= Rp {R['mean_rp'].mean():+,.0f}")
    print(f"  MEDIAN month    {R['median'].mean():+.2%}   "
          f"= Rp {R['median_rp'].mean():+,.0f}    <- the typical month")
    print(f"  months positive {R['pos'].mean():.0%}")
    print(f"  month-to-month sd {R['sd'].mean():.2%}, i.e. "
          f"+/-Rp {R['sd'].mean() * a.capital:,.0f} of ordinary swing")
    print(f"  worst month     {R['worst'].min():.1%} "
          f"= Rp {R['worst'].min() * a.capital:+,.0f}")
    print(f"  invested {np.mean(inv):.0%} of the time on average")
    print(f"\n  THE TWO CONTROLS")
    print(f"  same machine, RANDOM entries: mean {RD['mean'].mean():+.2%} a "
          f"month, CAGR {RD['cagr'].mean():+.2%}")
    print(f"  IHSG buy and hold:            mean {ir['mean']:+.2%} a month, "
          f"CAGR {ir['cagr']:+.2%}")
    edge = R["cagr"].mean() - RD["cagr"].mean()
    print(f"  the Hull filter is worth {edge:+.2%} a year against picking at "
          f"random from the same universe with the same exit,")
    print(f"  and the strategy's own draw-to-draw spread is "
          f"{R['cagr'].max() - R['cagr'].min():.2%}.")
    print(f"\n  spread across draws: CAGR {R['cagr'].min():+.2%} to "
          f"{R['cagr'].max():+.2%} — that range is LUCK, not strategy, and it "
          f"is what a single backtest would have reported as the answer.")

    #  THE HALF-SPLIT, WHICH IS THE ONLY REPLICATION TEST THIS REPO TRUSTS.
    #  A single full-sample edge over 293 correlated months is exactly the
    #  statistic A18 warned reads as overwhelming and means nothing.
    ha = np.array([half_cagr(c) for c in curves])
    hr = np.array([half_cagr(c) for c in rand])
    print("\n=== THE HALF-SPLIT (A18: the only replication test this repo "
          "trusts)")
    print(f"{'':>18}{'early':>10}{'late':>10}")
    print(f"{'Hull-filtered':>18}{ha[:, 0].mean():>10.2%}"
          f"{ha[:, 1].mean():>10.2%}")
    print(f"{'random, same exit':>18}{hr[:, 0].mean():>10.2%}"
          f"{hr[:, 1].mean():>10.2%}")
    for j, nm in ((0, "early"), (1, "late")):
        d = ha[:, j].mean() - hr[:, j].mean()
        se = np.sqrt(ha[:, j].var(ddof=1) / len(ha)
                     + hr[:, j].var(ddof=1) / len(hr))
        print(f"  edge {nm:<6} {d:+.2%}  +/- {2 * se:.2%} (2se across draws)"
              f"   {'POSITIVE' if d > 0 else 'NEGATIVE'}")
    both = ((ha[:, 0] > hr[:, 0].mean()) & (ha[:, 1] > hr[:, 1].mean())).sum()
    print(f"  draws beating the random mean in BOTH halves: {both} of "
          f"{len(ha)}  (chance alone gives ~{len(ha) / 4:.1f})")

    #  AND THE POWER STATEMENT, because A19 recorded writing a power claim as
    #  an effect claim as its own error. How long must this account run before
    #  its own mean month is distinguishable from zero at t = 2?
    mu, sd = R["mean"].mean(), R["sd"].mean()
    n_zero = (2.0 * sd / mu) ** 2
    n_rand = (2.0 * sd / max(1e-9, edge / 12.0)) ** 2
    print(f"\n  months needed to tell this account's mean from ZERO at t=2: "
          f"{n_zero:.0f}  ({n_zero / 12:.1f} years)")
    print(f"  months needed to tell it from RANDOM PICKING: {n_rand:,.0f}  "
          f"({n_rand / 12:,.0f} years) — i.e. never.")
    R.to_csv(os.path.join(OUT, "monthly.csv"), index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
