#!/usr/bin/env python3
"""The annotated chart, scored against everything this repository can actually do.

One table, on the exact series the circles were drawn on (ADRO weekly), plus the
deployed book. Every row is the same money over the same years, so the columns
are comparable and the gap between the top row and the rest is the honest answer
to "can you do this".

    CIRCLES         every turn called correctly, in hindsight. Not attainable.
    80% ACCURACY    the standard the chart asked for. Not attainable either, but
                    it is the number that made the request reasonable.
    CAUSAL FILTER   the best honest single-name approximation: buy once price is
                    x% off the low, sell once it is x% off the high.
    REGIME          the deployed overlay: fully invested while IHSG is above its
                    30-week average, 25% below it, idle cash in deposits.
    BUY AND HOLD    the thing all of it has to beat.

The deposit rate matters and is therefore an input, not an assumption buried in
the code: the overlay spends a quarter of its life out of the market, and whether
that cash earns 0% or 6% is the difference between the overlay costing 1.8% a
year and paying for itself.

    python3 scripts/compare_to_chart.py [--ticker ADRO] [--cash 0.05]
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

from bluechip import pit_universe                        # noqa: E402
from bluechip_final import market_state                  # noqa: E402
from exposure import apply_exposure                      # noqa: E402
from idxbot.config import load_config                    # noqa: E402
from idxbot.data.cache import Cache                      # noqa: E402
from idxbot.data.ohlcv import YahooOHLCV                 # noqa: E402
from optimize_consistent import load_wide, score_curve   # noqa: E402
from swing_accuracy import legs, simulate as mc, zigzag  # noqa: E402
from turn_book import simulate                           # noqa: E402
from turn_trader import ROUND_TRIP, clean_weekly, reversal_state, run  # noqa: E402


def stats(growth: float, years: float, dd: float = np.nan) -> Dict[str, float]:
    return {"growth": growth, "cagr": growth ** (1 / years) - 1, "max_dd": dd}


def curve_dd(eq: np.ndarray) -> float:
    return float((eq / np.maximum.accumulate(eq) - 1).min())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default="ADRO")
    ap.add_argument("--threshold", type=float, default=0.20)
    ap.add_argument("--cash", type=float, default=0.05)
    ap.add_argument("--trials", type=int, default=4000)
    args = ap.parse_args()
    os.makedirs("reports", exist_ok=True)

    cfg = load_config()
    loader = YahooOHLCV(cfg, Cache(cfg.path("data.cache_dir", "data/cache")))
    d = loader.get(args.ticker, max_age=86400 * 30)
    w = clean_weekly(d)
    if w is None:
        raise SystemExit(f"{args.ticker}: not enough history")
    px = w.to_numpy(float)
    years = (w.index[-1] - w.index[0]).days / 365.25
    piv = zigzag(px, args.threshold)
    lg = legs(px, piv)

    print("=" * 100)
    print(f" {args.ticker} WEEKLY — {w.index[0]:%Y-%m-%d} to {w.index[-1]:%Y-%m-%d} "
          f"({years:.1f} years)")
    print(f" {len(lg)} swings of {args.threshold:.0%}+, one turn every "
          f"{years * 12 / max(len(lg), 1):.1f} months")
    print("=" * 100)

    rows: List[Dict] = []

    bh = px[-1] / px[0]
    rows.append({"what": "buy and hold", **stats(bh, years, curve_dd(px)),
                 "note": "own it, do nothing"})

    perfect = 1.0
    for _a, _b, r in lg:
        if r > 0:
            perfect *= (1.0 + r) * (1.0 - ROUND_TRIP)
    rows.append({"what": "the circles (hindsight)", **stats(perfect, years, np.nan),
                 "note": "every turn right — NOT ATTAINABLE"})

    rng = np.random.default_rng(20260818)
    g80 = float(np.median([mc(lg, 0.80, "inverse", rng) for _ in range(args.trials)]))
    rows.append({"what": "80% of turns right", **stats(g80, years, np.nan),
                 "note": "what the chart asked for — not attainable"})
    g62 = float(np.median([mc(lg, 0.62, "inverse", rng) for _ in range(args.trials)]))
    rows.append({"what": "62% of turns right", **stats(g62, years, np.nan),
                 "note": "break-even against holding"})

    best = None
    for thr in (0.10, 0.15, 0.20, 0.25, 0.30):
        st = reversal_state(px, thr, thr)
        eq, _ = run(px, st)
        if best is None or eq[-1] > best[1][-1]:
            best = (thr, eq)
    rows.append({"what": f"causal filter ({best[0]:.0%})",
                 **stats(float(best[1][-1]), years, curve_dd(best[1])),
                 "note": "best honest single-name rule, tuned WITH hindsight"})

    # the deployed overlay, applied to this one name
    state = market_state(w.index, 30)
    e = pd.Series(np.where(state.to_numpy() > 0.5, 1.0, 0.25), index=w.index)
    eqs = pd.Series(px / px[0], index=w.index)
    timed, _ = apply_exposure(eqs, e, cash=args.cash)
    rows.append({"what": "regime overlay on this name",
                 **stats(float(timed.iloc[-1]), years, curve_dd(timed.to_numpy())),
                 "note": f"deployed rule, cash at {args.cash:.0%}"})

    R = pd.DataFrame(rows)
    print(f"\n {'':<30}{'growth':>10}{'CAGR':>9}{'maxDD':>8}   note")
    for _, r in R.iterrows():
        dd = "     —" if not np.isfinite(r["max_dd"]) else f"{r['max_dd']:>8.0%}"
        print(f" {r['what']:<30}{r['growth']:>9,.1f}x{r['cagr']:>+9.1%}{dd}   {r['note']}")
    R.to_csv(f"reports/compare_{args.ticker}.csv", index=False)

    # ------------------------- the deployed book ------------------------- #
    print(f"\n{'=' * 100}\n WHAT IS ACTUALLY DEPLOYED — the blue-chip book, "
          f"not one stock\n{'=' * 100}")
    W = load_wide(verbose=False)
    pit = pit_universe(W, 30)
    eqb, trades = simulate(W, pit, lookback=250, top_n=12, rebalance=60)
    bstate = market_state(eqb.index, 30)
    flat = score_curve(eqb, trades)
    out = []
    for name, lvl in (("book, always on", None), ("book + regime overlay", 0.25)):
        if lvl is None:
            cur = eqb
        else:
            ee = pd.Series(np.where(bstate.to_numpy() > 0.5, 1.0, lvl), index=eqb.index)
            cur, _ = apply_exposure(eqb, ee, cash=args.cash)
        s = score_curve(cur, trades)
        out.append({"what": name, **s})
    print(f" {'':<26}{'growth':>10}{'CAGR':>9}{'worst yr':>10}{'maxDD':>8}"
          f"{'+yrs':>7}{'ulcer':>7}")
    for r in out:
        print(f" {r['what']:<26}{r['growth']:>9,.1f}x{r['cagr']:>+9.1%}"
              f"{r['worst_year']:>+10.1%}{r['max_dd']:>8.0%}"
              f"{r['pct_positive']:>7.0%}{r['ulcer']:>7.2f}")
    print(f"\n cash held out of the market earns {args.cash:.0%}; at 0% the overlay "
          f"costs about 1.8%/yr instead.")
    pd.DataFrame(out).to_csv("reports/compare_book.csv", index=False)
    print("\n -> reports/compare_*.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
