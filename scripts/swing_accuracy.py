#!/usr/bin/env python3
"""What is a weekly swing trader's CAGR worth at a given accuracy?

The question this answers: *if you can mark the major turns on a weekly chart -
ignoring minor noise - and you get them right X% of the time, what do you earn?*

Method
------
1. Resample to weekly bars and build a **zigzag**: alternating peaks and troughs
   where each leg moves at least ``--threshold``. That is the formal version of
   "circle the big swings and omit the noise". Legs smaller than the threshold
   are deliberately invisible, exactly as they are to the eye.
2. Score a perfect trader who is long every up-leg and flat (or short) every
   down-leg. This is the ceiling and it is not attainable - the turns are only
   identifiable afterwards.
3. Score an **imperfect** trader by Monte Carlo. At each turn, with probability
   ``p`` the call is right; otherwise it is wrong, under two models:

   ``miss``    a wrong call means you do nothing - you sit out an up-leg, or you
               stay long through a down-leg.
   ``inverse`` a wrong call means you do the opposite - you buy the top and sell
               the bottom of that leg.

   Reality sits between them: a wrong call usually means being late rather than
   perfectly inverted, so ``miss`` flatters and ``inverse`` punishes.
4. Costs are charged on every leg traded.

The number that matters is not the ceiling. It is how fast the ceiling decays as
accuracy falls, because that tells you what your edge has to be worth.

    python3 scripts/swing_accuracy.py --ticker ADRO --threshold 0.20
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

from idxbot.config import load_config          # noqa: E402
from idxbot.data.cache import Cache            # noqa: E402
from idxbot.data.ohlcv import YahooOHLCV       # noqa: E402

ROUND_TRIP = 0.006          # 0.15% buy + 0.25% sell + slippage


def weekly(df: pd.DataFrame) -> pd.DataFrame:
    x = df.set_index("date").sort_index()
    return pd.DataFrame({
        "close": x["close"].resample("W-FRI").last(),
        "adj_close": x["adj_close"].resample("W-FRI").last(),
    }).dropna()


def zigzag(prices: np.ndarray, threshold: float) -> List[int]:
    """Alternating turning points, each leg at least ``threshold`` in size.

    This is what "the big swings, ignoring noise" means when written down. It is
    computed on the whole series at once and is therefore **pure hindsight** -
    which is the point: it defines the ceiling, not a strategy.
    """
    if len(prices) < 3:
        return []
    pivots = [0]
    direction = 0            # +1 seeking a peak, -1 seeking a trough, 0 unknown
    # The running high and low are tracked SEPARATELY and reset only when a turn
    # is confirmed. A single running extreme that follows the price in whichever
    # direction it happens to move collapses onto the previous bar, and then
    # confirming a turn needs a one-bar move of the full threshold - which is why
    # an earlier version of this returned two pivots for a clean saw-tooth.
    hi_i = lo_i = 0
    for i in range(1, len(prices)):
        p = prices[i]
        if p > prices[hi_i]:
            hi_i = i
        if p < prices[lo_i]:
            lo_i = i
        if direction >= 0 and p <= prices[hi_i] * (1 - threshold):
            pivots.append(hi_i)          # the high is confirmed as a peak
            direction = -1
            hi_i = lo_i = i
        elif direction <= 0 and p >= prices[lo_i] * (1 + threshold):
            pivots.append(lo_i)          # the low is confirmed as a trough
            direction = 1
            hi_i = lo_i = i
    if pivots[-1] != len(prices) - 1:
        pivots.append(len(prices) - 1)
    return sorted(set(pivots))


def legs(prices: np.ndarray, pivots: List[int]) -> List[Tuple[int, int, float]]:
    out = []
    for a, b in zip(pivots[:-1], pivots[1:]):
        out.append((a, b, prices[b] / prices[a] - 1.0))
    return out


def simulate(leg_list, accuracy: float, mode: str, rng, cost: float = ROUND_TRIP,
             allow_short: bool = False) -> float:
    """Total growth for a trader who calls each leg right with probability ``accuracy``."""
    equity = 1.0
    for _a, _b, r in leg_list:
        correct = rng.random() < accuracy
        up = r > 0
        if correct:
            if up:
                equity *= (1.0 + r) * (1.0 - cost)
            elif allow_short:
                equity *= (1.0 - r) * (1.0 - cost)
            # correct on a down leg without shorting = sit out, no cost
        else:
            if mode == "miss":
                # wrong on an up leg = missed it; wrong on a down leg = rode it
                if not up:
                    equity *= (1.0 + r)
            else:                       # inverse: you did the opposite
                if up:
                    equity *= (1.0 - r) * (1.0 - cost) if allow_short else 1.0
                else:
                    equity *= (1.0 + r) * (1.0 - cost)
    return equity


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default="ADRO")
    ap.add_argument("--threshold", type=float, default=0.20)
    ap.add_argument("--trials", type=int, default=4000)
    ap.add_argument("--short", action="store_true")
    args = ap.parse_args()
    os.makedirs("reports", exist_ok=True)

    cfg = load_config()
    loader = YahooOHLCV(cfg, Cache(cfg.path("data.cache_dir", "data/cache")))
    d = loader.get(args.ticker, max_age=86400)
    if d is None or d.empty:
        raise SystemExit(f"no data for {args.ticker}")
    w = weekly(d)
    px = w["adj_close"].to_numpy(float)
    years = (w.index[-1] - w.index[0]).days / 365.25

    piv = zigzag(px, args.threshold)
    lg = legs(px, piv)
    ups = [l for l in lg if l[2] > 0]
    downs = [l for l in lg if l[2] <= 0]

    print("=" * 78)
    print(f" {args.ticker} WEEKLY — swings of at least {args.threshold:.0%}")
    print("=" * 78)
    print(f" {len(w)} weekly bars, {w.index[0]:%Y-%m-%d} -> {w.index[-1]:%Y-%m-%d} "
          f"({years:.1f} years)")
    print(f" buy & hold: {px[-1]/px[0]:.2f}x  ({(px[-1]/px[0])**(1/years)-1:+.1%} a year)")
    print(f"\n turning points found: {len(piv)}  ->  {len(lg)} legs "
          f"({len(ups)} up, {len(downs)} down)")
    print(f" that is one decision every {years*12/max(len(lg),1):.1f} months\n")
    print(f" {'from':<12}{'to':<12}{'weeks':>7}{'move':>9}")
    for a, b, r in lg:
        print(f" {w.index[a]:%Y-%m-%d}  {w.index[b]:%Y-%m-%d}{b-a:>7}{r:>9.1%}")

    perfect = 1.0
    for _a, _b, r in lg:
        if r > 0:
            perfect *= (1.0 + r) * (1.0 - ROUND_TRIP)
        elif args.short:
            perfect *= (1.0 - r) * (1.0 - ROUND_TRIP)
    print(f"\n PERFECT capture (hindsight, {'long/short' if args.short else 'long/flat'}):"
          f" {perfect:,.1f}x = {perfect**(1/years)-1:+.1%} a year")

    print("\n" + "=" * 78)
    print(" WHAT ACCURACY IS WORTH")
    print("=" * 78)
    rng = np.random.default_rng(20260817)
    rows = []
    bh = px[-1] / px[0]
    print(f" {'accuracy':>9}{'':4}{'miss model':>26}{'':4}{'inverse model':>26}")
    print(f" {'':9}{'':4}{'median':>10}{'CAGR':>8}{'>B&H':>8}"
          f"{'':4}{'median':>10}{'CAGR':>8}{'>B&H':>8}")
    for acc in (1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.6, 0.5):
        res = {}
        for mode in ("miss", "inverse"):
            g = np.array([simulate(lg, acc, mode, rng, allow_short=args.short)
                          for _ in range(args.trials)])
            res[mode] = (float(np.median(g)),
                         float(np.median(g)) ** (1 / years) - 1,
                         float((g > bh).mean()))
        rows.append({"accuracy": acc,
                     "miss_growth": res["miss"][0], "miss_cagr": res["miss"][1],
                     "miss_beat": res["miss"][2],
                     "inv_growth": res["inverse"][0], "inv_cagr": res["inverse"][1],
                     "inv_beat": res["inverse"][2]})
        print(f" {acc:>9.0%}{'':4}{res['miss'][0]:>10,.1f}x{res['miss'][1]:>8.1%}"
              f"{res['miss'][2]:>8.0%}{'':4}{res['inverse'][0]:>10,.1f}x"
              f"{res['inverse'][1]:>8.1%}{res['inverse'][2]:>8.0%}")
    pd.DataFrame(rows).to_csv(
        f"reports/swing_accuracy_{args.ticker}.csv", index=False)
    print(f"\n -> reports/swing_accuracy_{args.ticker}.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
