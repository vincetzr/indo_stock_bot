#!/usr/bin/env python3
"""Find a rule that calls 80%+ of the weekly turns right on EVERY name.

The target is stated plainly: direction accuracy of at least 80% on every blue
chip, not on average and not on the best one. This searches for it and reports
whether it exists.

Direction accuracy here means: over each completed hindsight leg, was the rule
positioned on the correct side for most of it? That is the natural reading of
"calling the turns right", and it is the metric the 80% figure came from.

The search covers the whole speed range, because accuracy is monotone in speed:
a rule that re-evaluates every week tracks the price closely and is almost always
on the right side; a slow one is late and is wrong across the turn. So the
question is not really whether 80% is reachable - it is what a rule that
reaches it is worth, which is why every candidate is reported with its return
beside its accuracy.

    python3 scripts/accuracy_target.py [--target 0.80]
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

from idxbot.config import load_config          # noqa: E402
from idxbot.data.cache import Cache            # noqa: E402
from idxbot.data.ohlcv import YahooOHLCV       # noqa: E402
from swing_accuracy import legs, zigzag        # noqa: E402
from turn_trader import (MIN_TURNOVER, clean_weekly,   # noqa: E402
                         reversal_state, run)

MIN_WEEKS = 200


def ma_state(px: np.ndarray, n: int) -> np.ndarray:
    s = pd.Series(px)
    return (s > s.rolling(n, min_periods=n).mean()).fillna(False).to_numpy().astype(np.int8)


def ema_state(px: np.ndarray, n: int) -> np.ndarray:
    s = pd.Series(px)
    return (s > s.ewm(span=n, adjust=False).mean()).fillna(False).to_numpy().astype(np.int8)


def slope_state(px: np.ndarray, n: int) -> np.ndarray:
    """Is the n-week average itself rising? Slower to flip than price-vs-average."""
    m = pd.Series(px).rolling(n, min_periods=n).mean()
    return (m > m.shift(1)).fillna(False).to_numpy().astype(np.int8)


def build_rules(px: np.ndarray) -> Dict[str, np.ndarray]:
    r: Dict[str, np.ndarray] = {}
    for n in (2, 3, 4, 6, 8, 10, 13, 20, 26, 30, 40):
        r[f"MA {n}w"] = ma_state(px, n)
    for n in (3, 5, 8, 13, 21):
        r[f"EMA {n}w"] = ema_state(px, n)
    for n in (4, 8, 13):
        r[f"slope MA {n}w"] = slope_state(px, n)
    for t in (0.08, 0.12, 0.15, 0.20, 0.25, 0.30):
        r[f"reversal {t:.0%}"] = reversal_state(px, t, t)
    return r


def score_name(px: np.ndarray, st: np.ndarray, lg, years: float) -> Dict[str, float]:
    eq, trades = run(px, st)
    right = 0
    for a, b, rr in lg:
        seg = st[a:b]
        if len(seg) == 0:
            continue
        if (rr > 0) == (seg.mean() > 0.5):
            right += 1
    g = float(eq[-1])
    bh = px[-1] / px[0]
    return {"direction": right / max(len(lg), 1),
            "cagr": g ** (1 / years) - 1 if years > 0 else np.nan,
            "bh_cagr": bh ** (1 / years) - 1 if years > 0 else np.nan,
            "trades": trades}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=float, default=0.80)
    ap.add_argument("--threshold", type=float, default=0.20)
    ap.add_argument("--universe", default="bluechip")
    args = ap.parse_args()
    os.makedirs("reports", exist_ok=True)

    cfg = load_config()
    loader = YahooOHLCV(cfg, Cache(cfg.path("data.cache_dir", "data/cache")))
    names = sorted(set(cfg.universe(args.universe)) | set(cfg.universe("lq45")))
    raw = loader.get_many(names, max_age=86400 * 30, verbose=False)

    series: Dict[str, pd.Series] = {}
    for t, d in raw.items():
        if len(d) < 500:
            continue
        c = d["close"].astype(float)
        if float((c * d["volume"]).median()) < MIN_TURNOVER:
            continue
        w = clean_weekly(d)
        if w is not None and len(w) >= MIN_WEEKS:
            series[t] = w
    print(f"{len(series)} names with enough weekly history and liquidity")

    rows: List[Dict] = []
    for t, w in series.items():
        px = w.to_numpy(float)
        years = (w.index[-1] - w.index[0]).days / 365.25
        lg = legs(px, zigzag(px, args.threshold))
        if len(lg) < 4:
            continue
        for rule, st in build_rules(px).items():
            rows.append({"ticker": t, "rule": rule, "legs": len(lg),
                         **score_name(px, st, lg, years)})
    R = pd.DataFrame(rows)
    R.to_csv("reports/accuracy_target.csv", index=False)

    g = R.groupby("rule").agg(
        min_acc=("direction", "min"), median_acc=("direction", "median"),
        pct_names_at_target=("direction", lambda s: float((s >= args.target).mean())),
        median_cagr=("cagr", "median"), median_bh=("bh_cagr", "median"),
        median_trades=("trades", "median")).reset_index()
    g["excess"] = g["median_cagr"] - g["median_bh"]
    g = g.sort_values("min_acc", ascending=False)

    print(f"\n{'=' * 104}\n CAN A RULE CALL {args.target:.0%}+ OF THE TURNS RIGHT ON "
          f"EVERY NAME?\n{'=' * 104}")
    print(f" {'rule':<16}{'WORST name':>12}{'median':>9}{'names at target':>17}"
          f"{'median CAGR':>13}{'vs hold':>10}{'trades':>8}")
    for _, r in g.iterrows():
        flag = "  <-- MEETS IT ON EVERY NAME" if r["min_acc"] >= args.target else ""
        print(f" {r['rule']:<16}{r['min_acc']:>12.0%}{r['median_acc']:>9.0%}"
              f"{r['pct_names_at_target']:>17.0%}{r['median_cagr']:>+13.1%}"
              f"{r['excess']:>+10.1%}{r['median_trades']:>8.0f}{flag}")

    winners = g[g["min_acc"] >= args.target]
    print(f"\n{'=' * 104}")
    if winners.empty:
        best = g.iloc[0]
        print(f" No rule reaches {args.target:.0%} on every name. Closest: "
              f"{best['rule']} at {best['min_acc']:.0%} on its worst name, "
              f"{best['pct_names_at_target']:.0%} of names at target.")
    else:
        print(f" {len(winners)} rule(s) call {args.target:.0%}+ of the turns right on "
              f"EVERY ONE of {R['ticker'].nunique()} names:\n")
        for _, r in winners.iterrows():
            print(f"   {r['rule']:<16} worst name {r['min_acc']:.0%}, "
                  f"median {r['median_acc']:.0%}, "
                  f"returns {r['median_cagr']:+.1%}/yr against "
                  f"{r['median_bh']:+.1%} for holding "
                  f"({r['excess']:+.1%}), {r['median_trades']:.0f} trades")
        prof = winners[winners["excess"] > 0]
        print(f"\n of those, rules that also BEAT buy-and-hold: {len(prof)}")
        if prof.empty:
            print("   none — every rule that hits the accuracy target loses money,")
            print("   because hitting it requires re-evaluating fast enough that the")
            print("   fees and the late entries eat more than the direction is worth.")

    # the other direction: what does the most PROFITABLE rule score on accuracy?
    best_ret = g.sort_values("excess", ascending=False).iloc[0]
    print(f"\n Best rule by RETURN: {best_ret['rule']} at {best_ret['excess']:+.1%} "
          f"over holding — its worst-name accuracy is {best_ret['min_acc']:.0%} "
          f"and its median is {best_ret['median_acc']:.0%}.")
    print("\n -> reports/accuracy_target.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
