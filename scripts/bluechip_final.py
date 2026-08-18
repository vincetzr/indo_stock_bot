#!/usr/bin/env python3
"""The blue-chip algorithm, assembled - and how much insurance to buy.

Part XXI left a binary choice: run the book fully invested at +7.8% with a -38%
drawdown, or fully out whenever IHSG is below its 30-week average at +5.1% with
-16%. Nobody has to choose between those two corners. The overlay can scale
exposure instead of switching it, and the middle of that range has never been
measured.

    exposure when the market filter says OUT = 0%, 25%, 50%, 75%, 100%

0% is Part XXI's filter, 100% is always-on, and everything between is a partial
hedge. The question is whether the return given up and the drawdown removed
trade off linearly - if they do, the middle is uninteresting and the choice
really is binary; if the drawdown falls faster than the return does, there is a
better point than either corner.

The book itself is fixed and comes from Part XIX, chosen by lever agreement
rather than by search: point-in-time large caps, 250-day momentum, top 12,
rebalanced quarterly, no per-name gate.

    python3 scripts/bluechip_final.py [--folds 5]
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

from idxbot.config import load_config          # noqa: E402
from idxbot.data.cache import Cache            # noqa: E402
from idxbot.data.ohlcv import YahooOHLCV       # noqa: E402
from bluechip import pit_universe              # noqa: E402
from optimize_consistent import load_wide, score_curve   # noqa: E402
from turn_book import simulate                 # noqa: E402
from turn_trader import clean_weekly           # noqa: E402

INDEX = "^JKSE"
OVERLAY_COST = 0.004        # scaling the whole book in or out, per unit changed


def market_state(daily: pd.DatetimeIndex, weeks: int = 30) -> pd.Series:
    """1 while IHSG is above its own N-week average, as known that week.

    Computed on weekly closes and then held flat across the following days, so a
    daily bar never sees a weekly close that had not printed yet.
    """
    cfg = load_config()
    loader = YahooOHLCV(cfg, Cache(cfg.path("data.cache_dir", "data/cache")))
    idx = loader.get(INDEX, max_age=86400 * 30)
    w = clean_weekly(idx)
    if w is None:
        raise SystemExit("index history too short")
    st = (w > w.rolling(weeks, min_periods=weeks).mean()).astype(float)
    # shift one week: the signal from Friday's close governs the week after
    return st.shift(1).reindex(daily, method="ffill").fillna(1.0)


def overlay(eq: pd.Series, state: pd.Series, out_exposure: float,
            cost: float = OVERLAY_COST) -> pd.Series:
    """Scale the book by exposure, charging a fee on every change in exposure."""
    r = eq.pct_change().fillna(0.0).to_numpy()
    e = np.where(state.to_numpy() > 0.5, 1.0, out_exposure)
    out = np.ones(len(r))
    prev = e[0]
    for i in range(1, len(r)):
        out[i] = out[i - 1] * (1.0 + r[i] * e[i - 1])
        if e[i] != prev:
            out[i] *= (1.0 - cost * abs(e[i] - prev))
            prev = e[i]
    return pd.Series(out, index=eq.index)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--universe-size", type=int, default=30)
    ap.add_argument("--weeks", type=int, default=30)
    args = ap.parse_args()
    os.makedirs("reports", exist_ok=True)

    W = load_wide()
    pit = pit_universe(W, args.universe_size)
    print("building the blue-chip book (250d momentum, top 12, quarterly) ...")
    eq, trades = simulate(W, pit, lookback=250, top_n=12, rebalance=60)
    state = market_state(eq.index, args.weeks)
    print(f"  {trades:,} trades; the market filter is on "
          f"{state.mean():.0%} of days")

    n = len(eq)
    edges = np.linspace(int(n * 0.35), n, args.folds + 1).astype(int)
    windows = [(edges[k], edges[k + 1]) for k in range(args.folds)]
    labels = [f"{eq.index[a]:%Y-%m}" for a, _ in windows]

    exposures = (0.0, 0.25, 0.50, 0.75, 1.0)
    print(f"\n{'=' * 104}\n HOW MUCH TO HOLD WHEN THE MARKET FILTER SAYS OUT"
          f"\n{'=' * 104}")
    print(f" {'out-exposure':<14}" + "".join(f"{f'from {l}':>12}" for l in labels)
          + f"{'mean':>9}{'worst':>9}{'maxDD':>8}{'ulcer':>7}")
    rows = []
    curves = {}
    for x in exposures:
        cur = overlay(eq, state, x)
        curves[x] = cur
        cs, dds, uls = [], [], []
        for a, b in windows:
            seg = cur.iloc[a:b] / cur.iloc[a]
            s = score_curve(seg, 0)
            cs.append(s["cagr"])
            dds.append(s["max_dd"])
            uls.append(s["ulcer"])
        full = score_curve(cur, trades)
        rows.append({"out_exposure": x,
                     **{f"fold{k+1}": cs[k] for k in range(args.folds)},
                     "mean": float(np.mean(cs)), "worst": float(np.min(cs)),
                     "mean_dd": float(np.mean(dds)), "mean_ulcer": float(np.mean(uls)),
                     "full_cagr": full["cagr"], "full_growth": full["growth"],
                     "full_dd": full["max_dd"], "full_worst_year": full["worst_year"],
                     "full_ulcer": full["ulcer"],
                     "pct_positive": full["pct_positive"]})
        print(f" {x:<14.0%}" + "".join(f"{c:>+12.1%}" for c in cs)
              + f"{np.mean(cs):>+9.1%}{np.min(cs):>+9.1%}"
              f"{np.mean(dds):>8.0%}{np.mean(uls):>7.2f}")
    R = pd.DataFrame(rows)
    R.to_csv("reports/bluechip_final.csv", index=False)

    print(f"\n{'=' * 104}\n THE FULL RECORD, AND WHAT EACH POINT OF SAFETY COSTS"
          f"\n{'=' * 104}")
    print(f" {'out-exposure':<14}{'growth':>10}{'CAGR':>9}{'worst yr':>11}"
          f"{'maxDD':>8}{'+yrs':>7}{'ulcer':>7}{'return given up':>17}"
          f"{'drawdown removed':>19}")
    base = R[R["out_exposure"] == 1.0].iloc[0]
    for _, r in R.iterrows():
        give = r["full_cagr"] - base["full_cagr"]
        save = (r["full_dd"] - base["full_dd"]) * 100
        print(f" {r['out_exposure']:<14.0%}{r['full_growth']:>9,.1f}x"
              f"{r['full_cagr']:>+9.1%}{r['full_worst_year']:>+11.1%}"
              f"{r['full_dd']:>8.0%}{r['pct_positive']:>7.0%}{r['full_ulcer']:>7.2f}"
              f"{give:>+17.2%}{save:>+18.0f}pt")

    # Is the trade-off linear? If drawdown falls faster than return, the middle
    # is worth holding; if it is a straight line, the choice really is binary.
    print(f"\n{'=' * 104}\n IS THERE A BETTER POINT THAN EITHER CORNER?\n{'=' * 104}")
    print(f" {'out-exposure':<14}{'return kept':>14}{'drawdown kept':>16}"
          f"{'points of DD removed per point of return':>44}")
    for _, r in R.iterrows():
        if r["out_exposure"] == 1.0:
            continue
        give = base["full_cagr"] - r["full_cagr"]
        save = (r["full_dd"] - base["full_dd"]) * 100
        ratio = save / (give * 100) if give > 0 else np.inf
        print(f" {r['out_exposure']:<14.0%}"
              f"{r['full_cagr'] / base['full_cagr']:>13.0%}"
              f"{r['full_dd'] / base['full_dd']:>16.0%}{ratio:>44.2f}")
    print("\n a ratio that RISES as exposure falls means the first units of")
    print(" insurance are the cheap ones and a partial hedge beats both corners;")
    print(" a flat ratio means the choice is genuinely binary.")

    pd.DataFrame({f"out_{int(x*100)}": c for x, c in curves.items()}).to_csv(
        "reports/bluechip_final_curves.csv")
    print("\n -> reports/bluechip_final.csv, reports/bluechip_final_curves.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
