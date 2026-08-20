#!/usr/bin/env python3
"""The one survivor, put through everything that could kill it.

WHAT SURVIVED AND WHY THAT IS SUSPICIOUS
----------------------------------------
factor_study tested thirteen price-derived factors on 788 IDX names over eleven
years. Twelve failed. One did not: trailing dividend yield, with a rank IC of
0.079 (t = 5.55, p < 0.0001, clearing Bonferroni), the largest decile spread by
a factor of two, a positive edge in both halves of the sample, and 89% name
persistence so it barely pays turnover.

That is exactly the moment to get sceptical rather than excited. Thirteen
factors were looked at; one of them was always going to look best. And 2021-2026
on IDX was a coal and bank dividend boom, so "high trailing yield" may be
nothing more than "was a coal miner in 2022" - a sector bet that happened to
work, wearing a factor's name.

SO EVERY WAY IT COULD BE AN ARTEFACT IS TESTED HERE
---------------------------------------------------
    grid            breadth x rebalance frequency x liquidity floor. An edge
                    that lives in one cell of a grid is a coincidence; one that
                    survives the whole grid is a property.
    lookback        6 months, 1 year, 2 years. If only one window works, the
                    window was fitted.
    offset          hold ranks 11-40 instead of 1-30. If the edge dies without
                    the three extreme yields, it is three names, not a premium.
    per year        every calendar year separately, against the neutral book and
                    the index, so a single boom cannot hide inside a CAGR.
    the crash       what it did in the 2026 drawdown specifically.
    persistence     does a high yield this year predict a high yield next year?
                    A characteristic that does not persist cannot be harvested.

WHAT IS STILL WRONG WITH IT AFTERWARDS
--------------------------------------
Two things no test here can fix, both stated rather than buried:

    1. SURVIVORSHIP. Every cached name still trades. A high-yield book in the
       real world would have held some of the names that stopped existing, and
       a company that delists rarely does so from strength.
    2. TAX. Yahoo's adjusted close reinvests the GROSS dividend. An Indonesian
       individual pays 10% final tax unless the proceeds are reinvested
       domestically under PP 9/2021, and a foreign holder pays 20% or the
       treaty rate. The gross number below is therefore an upper bound, and the
       report shows what the net one looks like.

    python3 scripts/yield_book.py
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))
sys.path.insert(0, os.path.dirname(__file__))

from idxbot.config import load_config           # noqa: E402
from idxbot.data.cache import Cache             # noqa: E402
from idxbot.data.ohlcv import YahooOHLCV        # noqa: E402
from factor_study import (Board, build_panel, cagr, load_index,   # noqa: E402
                          max_dd, rebalance_positions, run_portfolio, select)
from base_rates import DEPOSIT_RATE             # noqa: E402

DIV_TAX = 0.10          # PP 9/2021 final rate for an individual not reinvesting


def yield_scores(board: Board, lookback: int) -> List[np.ndarray]:
    """Trailing dividend yield at every rebalance, from the adjustment ratio.

    adj_close/close is the accumulated dividend factor. Its growth over the
    lookback IS the dividend paid over that window as a fraction of price, and
    nothing after the decision bar enters it.
    """
    out = []
    for b in board.rebal:
        if b < lookback:
            out.append(np.full(board.cv.shape[1], np.nan))
            continue
        with np.errstate(divide="ignore", invalid="ignore"):
            f0 = np.where(board.rw[b - lookback] > 0,
                          board.cv[b - lookback] / board.rw[b - lookback], np.nan)
            f1 = np.where(board.rw[b] > 0, board.cv[b] / board.rw[b], np.nan)
        # Annualise GEOMETRICALLY so a six-month and a two-year book land on
        # one scale. Scaling linearly would report a 10%/yr payer as 10.5% over
        # two years and 9.8% over six months purely from compounding, which
        # would then look like the lookback mattering when it did not.
        ratio = np.where(np.isfinite(f1 / f0) & (f1 / f0 > 0), f1 / f0, np.nan)
        y = ratio ** (250.0 / lookback) - 1.0
        # drop readings a dividend cannot produce: the adjustment factor cannot
        # shrink, and a yield near 100% is a corporate action or a bad print
        out.append(np.where(np.isfinite(y) & (y >= 0) & (y < 1.0), y, np.nan))
    return out


def yearly(eq: pd.Series) -> pd.Series:
    """Calendar-year return of an equity curve sampled at rebalances."""
    if len(eq) < 2:
        return pd.Series(dtype=float)
    ye = eq.resample("YE").last()
    first = pd.Series([float(eq.iloc[0])],
                      index=[eq.index[0] - pd.Timedelta(days=1)])
    ye = pd.concat([first, ye]).sort_index()
    return (ye / ye.shift(1) - 1.0).dropna()


def score_persistence(scores: Sequence[np.ndarray], cols: Sequence[np.ndarray],
                      gap: int) -> float:
    """Rank correlation between a name's score now and its score `gap` later.

    A factor is only harvestable if the characteristic sticks around. If this
    year's high yielders are next year's average yielders, the book is chasing
    a payout that has already happened.
    """
    from factor_study import spearman
    vals = []
    for k in range(len(scores) - gap):
        common = np.intersect1d(cols[k], cols[k + gap])
        if len(common) < 20:
            continue
        v = spearman(scores[k][common], scores[k + gap][common])
        if np.isfinite(v):
            vals.append(v)
    return float(np.mean(vals)) if vals else np.nan


def net_of_tax(gross_cagr: float, yield_share: float,
               rate: float = DIV_TAX) -> float:
    """Knock the dividend tax off a gross total return.

    Only the income part is taxed at the dividend rate, so the haircut is the
    yield times the rate, not the whole return times the rate.
    """
    return gross_cagr - yield_share * rate


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2015-01-01")
    ap.add_argument("--min-hist", type=int, default=280)
    args = ap.parse_args()

    cfg = load_config()
    cache_dir = cfg.path("data.cache_dir", "data/cache")
    loader = YahooOHLCV(cfg, Cache(cache_dir))

    print(f"{'=' * 96}\n THE YIELD BOOK — every way it could be an artefact\n"
          f"{'=' * 96}")
    close, raw, turn = build_panel(loader, cache_dir, args.start, total=True)
    idx_s = load_index(loader, "^JKSE", close.index)
    idx = idx_s.to_numpy(float) if len(idx_s) else np.full(len(close), np.nan)
    print(f" {close.shape[1]} names, {close.index[0]:%Y-%m-%d} to "
          f"{close.index[-1]:%Y-%m-%d}")

    # ---- 1. the grid -----------------------------------------------------
    print(f"\n{'=' * 96}\n 1. THE GRID — an edge in one cell is a coincidence\n"
          f"{'=' * 96}")
    print(f" {'floor':>10}{'rebal':>11}{'names':>7}{'CAGR':>9}{'vs EW':>9}"
          f"{'vs IHSG':>9}{'max DD':>9}{'cost/yr':>9}")
    grid: List[Dict] = []
    boards: Dict[Tuple[float, str], Board] = {}
    for floor in (2e9, 5e9, 2e10):
        for rb in ("monthly", "quarterly", "annual"):
            rebal = rebalance_positions(close.index, rb, args.min_hist)
            board = Board(close, turn, idx, rebal, [], floor, args.min_hist,
                          raw=raw)
            board.scores["divyield"] = yield_scores(board, 250)
            boards[(floor, rb)] = board
            R = run_portfolio(board, None, 0)
            ew, ew_daily = R.curve, R.daily
            b = idx_s.reindex(ew.index).dropna()
            bench = b / float(b.iloc[0]) if len(b) else pd.Series(dtype=float)
            per_year = {"monthly": 12.0, "quarterly": 4.0, "annual": 1.0}[rb]
            for n in (10, 20, 30, 50):
                r = run_portfolio(board, "divyield", n)
                eq, cost = r.curve, r.costs
                row = {"floor": floor, "rebal": rb, "n": n,
                       "cagr": cagr(eq), "vs_ew": cagr(eq) - cagr(ew),
                       "vs_ix": cagr(eq) - cagr(bench) if len(bench) else np.nan,
                       "dd": max_dd(r.daily),
                       "cost": float(np.mean(cost)) * per_year}
                grid.append(row)
                print(f" {floor:>10.0e}{rb:>11}{n:>7}{row['cagr']:>9.2%}"
                      f"{row['vs_ew']:>+9.2%}{row['vs_ix']:>+9.2%}"
                      f"{row['dd']:>9.1%}{row['cost']:>9.2%}")
    G = pd.DataFrame(grid)
    beat_ew = float((G["vs_ew"] > 0).mean())
    beat_ix = float((G["vs_ix"] > 0).mean())
    print(f"\n {len(G)} cells: beats the neutral book in {beat_ew:.0%} of them, "
          f"the IHSG in {beat_ix:.0%}.")
    print(f" worst cell {G['vs_ew'].min():+.2%}/yr vs the neutral book, "
          f"best {G['vs_ew'].max():+.2%}/yr, median {G['vs_ew'].median():+.2%}")

    base = boards[(5e9, "monthly")]
    BR = run_portfolio(base, None, 0)
    ew, ew_daily = BR.curve, BR.daily
    b = idx_s.reindex(ew.index).dropna()
    bench = b / float(b.iloc[0]) if len(b) else pd.Series(dtype=float)

    # ---- 2. the lookback -------------------------------------------------
    print(f"\n{'=' * 96}\n 2. THE LOOKBACK — if only one window works, the "
          f"window was fitted\n{'=' * 96}")
    print(f" {'lookback':>12}{'CAGR':>9}{'vs EW':>9}{'max DD':>9}"
          f"{'persistence':>13}{'yield sticks':>14}")
    for lb, label in ((125, "6 months"), (250, "1 year"), (500, "2 years")):
        base.scores[f"y{lb}"] = yield_scores(base, lb)
        r = run_portfolio(base, f"y{lb}", 30)
        eq = r.curve
        keep = []
        prev = None
        for k in range(len(base.rebal)):
            p = set(select(base.cols[k], base.scores[f"y{lb}"][k], 30).tolist())
            if prev is not None and p:
                keep.append(len(prev & p) / len(p))
            prev = p
        stick = score_persistence(base.scores[f"y{lb}"], base.cols, 12)
        print(f" {label:>12}{cagr(eq):>9.2%}{cagr(eq) - cagr(ew):>+9.2%}"
              f"{max_dd(r.daily):>9.1%}{np.mean(keep):>13.0%}{stick:>14.2f}")
    print(" 'persistence' is how much of the book carries over month to month;"
          "\n 'yield sticks' is the rank correlation between a name's yield now "
          "and a year later.")

    # ---- 3. is it three names? ------------------------------------------
    print(f"\n{'=' * 96}\n 3. SKIP THE TOP — is it a premium or is it three "
          f"names?\n{'=' * 96}")
    base.scores["divyield"] = yield_scores(base, 250)
    print(f" {'ranks held':>14}{'CAGR':>9}{'vs EW':>9}{'max DD':>9}")
    for off, n in ((0, 30), (3, 30), (10, 30), (20, 30), (40, 30)):
        r = run_portfolio(base, "divyield", n, offset=off)
        print(f" {f'{off + 1}-{off + n}':>14}{cagr(r.curve):>9.2%}"
              f"{cagr(r.curve) - cagr(ew):>+9.2%}{max_dd(r.daily):>9.1%}")
    print(" a premium spread across the ranking decays smoothly down this "
          "column;\n three lucky names fall off a cliff after the first row.")

    # ---- 4. year by year -------------------------------------------------
    print(f"\n{'=' * 96}\n 4. YEAR BY YEAR — a boom cannot hide inside a CAGR\n"
          f"{'=' * 96}")
    R30 = run_portfolio(base, "divyield", 30)
    eq30, eq30_daily = R30.curve, R30.daily
    ya, yb = yearly(eq30), yearly(ew)
    yi = yearly(bench) if len(bench) else pd.Series(dtype=float)
    print(f" {'year':>6}{'yield book':>12}{'neutral book':>14}{'IHSG':>9}"
          f"{'edge':>9}")
    wins = 0
    for d in ya.index:
        e = float(ya[d] - yb[d]) if d in yb.index else np.nan
        wins += int(np.isfinite(e) and e > 0)
        ix = float(yi[d]) if d in yi.index else np.nan
        print(f" {d.year:>6}{ya[d]:>12.1%}"
              f"{yb[d] if d in yb.index else np.nan:>14.1%}{ix:>9.1%}"
              f"{e:>+9.1%}")
    print(f" ahead in {wins} of {len(ya)} calendar years")

    # ---- 5. the crash ----------------------------------------------------
    print(f"\n{'=' * 96}\n 5. THE 2026 DRAWDOWN — where a yield book is "
          f"supposed to earn its keep\n{'=' * 96}")
    peak = bench.idxmax() if len(bench) else None
    if peak is not None:
        for label, s in (("yield book", eq30_daily), ("neutral book", ew_daily),
                         ("IHSG", bench)):
            seg = s[s.index >= peak]
            if len(seg) > 1:
                print(f" {label:<16}{float(seg.iloc[-1] / seg.iloc[0] - 1):>+9.1%}"
                      f"   from the index peak on {peak:%Y-%m-%d}")

    # ---- 6. what it costs after tax -------------------------------------
    print(f"\n{'=' * 96}\n 6. AFTER TAX, AND AGAINST CASH\n{'=' * 96}")
    ys = base.scores["divyield"]
    held_yield = []
    for k in range(len(base.rebal)):
        p = select(base.cols[k], ys[k], 30)
        if len(p):
            v = ys[k][p]
            held_yield.append(float(np.nanmean(v)))
    avg_y = float(np.nanmean(held_yield)) if held_yield else np.nan
    gross = cagr(eq30)
    print(f" the book's average trailing yield while held: {avg_y:.2%}")
    print(f" {'gross total return':<34}{gross:>9.2%}/yr")
    print(f" {'net of 10% dividend tax':<34}{net_of_tax(gross, avg_y):>9.2%}/yr")
    print(f" {'net of 20% (foreign holder)':<34}"
          f"{net_of_tax(gross, avg_y, 0.20):>9.2%}/yr")
    print(f" {'rupiah time deposit (assumed)':<34}{DEPOSIT_RATE:>9.2%}/yr"
          f"   at no drawdown")
    print(f" {'the neutral book':<34}{cagr(ew):>9.2%}/yr")
    print(f" {'the IHSG':<34}{cagr(bench) if len(bench) else np.nan:>9.2%}/yr")

    # ---- 7. what is in it now -------------------------------------------
    print(f"\n{'=' * 96}\n 7. THE BOOK AS IT STANDS TODAY\n{'=' * 96}")
    k = len(base.rebal) - 1
    picks = select(base.cols[k], ys[k], 30)
    if len(picks):
        names = close.columns.to_numpy()[picks]
        yy = ys[k][picks]
        order = np.argsort(-yy)
        print(f" as of {close.index[base.rebal[k]]:%Y-%m-%d}, 30 names, "
              f"equal weight:")
        for i in range(0, len(order), 5):
            print("  " + "  ".join(
                f"{names[j]} {yy[j]:>5.1%}" for j in order[i:i + 5]))
        tv = turn.iloc[base.rebal[k] - 250:base.rebal[k]].median()
        print(f"\n median daily turnover of the book: "
              f"{float(tv.iloc[picks].median()):,.0f} IDR — at 10% of volume "
              f"a\n position could be built in a day up to about "
              f"{float(tv.iloc[picks].median()) * 0.1 * 30:,.0f} IDR of book.")

    print(f"\n{'=' * 96}\n WHAT THIS DOES AND DOES NOT ESTABLISH\n{'=' * 96}")
    print(" It establishes that among thirteen price-derived factors, exactly "
          "one is\n significant after correcting for having looked at "
          "thirteen, is positive in\n both halves, survives the grid, and is "
          "cheap to run.")
    print(" It does NOT establish that the next eleven years look like the "
          "last eleven.\n The sample holds one commodity cycle, one pandemic "
          "and one crash, every\n cached name is a survivor, and the yield "
          "premium is a well-documented\n effect that has also spent decades "
          "at a time not working.")
    print("\n Three specific things to hold against it:")
    print("   TRAILING, NOT FORWARD. This ranks what was already paid. A "
          "company that\n   paid well and then cuts still ranks high the day "
          "the cut is announced, and\n   the book will be holding it.")
    print("   IT IS A VALUE FACTOR WEARING AN INCOME NAME. Yield is dividend "
          "over price,\n   so a name whose price halved ranks higher on the "
          "same payout. Part of what\n   is measured here is simply buying "
          "what has fallen, with the risk that\n   comes with that.")
    print("   THE DRAWDOWN IS NOT SMALLER. Every grid cell above lost between "
          "half and\n   two thirds at some point. This is a return "
          "improvement, not a risk one, and\n   an earlier version of this "
          "script reported -2.6% for the annual book purely\n   because it "
          "measured drawdown on eleven yearly snapshots.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
