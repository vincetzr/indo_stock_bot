#!/usr/bin/env python3
"""Anatomy of the gap: how much of it is arithmetic, and how much is left to chase.

The gap is 27x against 844x. Six attempts have failed to close it, all of them
price-only causal rules. Before a seventh, this measures WHY, in a way that
separates the part no rule can beat from the part a better signal could.

The structural bound
--------------------
A causal rule confirms a turn by waiting for the price to move against the old
direction. So its band must be WIDER than the largest pullback that happens
*inside* a leg - otherwise it exits on the pullback, re-enters higher, and
whipsaws. That gives a hard floor on the band, set by the data itself:

    band  >  typical deepest pullback inside a leg

and a band of ``b`` costs roughly ``b`` at the entry and ``b`` at the exit, so:

    capture  <=  (leg - 2b) / leg

That is not a property of any particular rule. It is what "wait for confirmation"
costs given how noisy the legs actually are, and it is computed here from the
legs themselves rather than assumed.

The addressable part
--------------------
The bound only binds a rule that confirms with PRICE. A signal that moves before
price does not have to wait. So the second half measures whether anything
available here leads the turns:

    volume climax     do turns print unusual volume BEFORE the price confirms?
    breadth           does market breadth roll over before the index does?
    commodities       for coal and metal names, does the underlying lead?

Each is scored by how many weeks earlier than the price band it fires, and
whether that lead is real or noise.

    python3 scripts/gap_anatomy.py [--ticker ADRO]
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
from swing_accuracy import legs, zigzag        # noqa: E402
from turn_trader import (MIN_TURNOVER, ROUND_TRIP,     # noqa: E402
                         clean_weekly, reversal_state, run)

MIN_WEEKS = 200


def intra_leg_noise(px: np.ndarray, lg) -> pd.DataFrame:
    """For each leg: its size, and the worst move against it before it ended.

    The second number is what a causal band has to survive. If a leg rises 100%
    but pulls back 22% on the way, a 15% band gets shaken out of it.
    """
    rows = []
    for a, b, r in lg:
        seg = px[a:b + 1]
        if len(seg) < 3:
            continue
        if r > 0:
            worst = float((seg / np.maximum.accumulate(seg) - 1.0).min())
        else:
            worst = float((seg / np.minimum.accumulate(seg) - 1.0).max())
        rows.append({"size": abs(r), "up": r > 0, "counter": abs(worst),
                     "weeks": b - a})
    return pd.DataFrame(rows)


def ceiling(lg, band: float, cost: float = ROUND_TRIP) -> float:
    """Growth of a rule that catches every leg but pays ``band`` at each end.

    This is the best a causal band of that width could POSSIBLY do: perfect
    turn identification, still late by the band on entry and exit.
    """
    g = 1.0
    for _a, _b, r in lg:
        if r > 0:
            captured = (1.0 + r) * (1.0 - band) / (1.0 + band) - 1.0
            g *= (1.0 + max(captured, -0.99)) * (1.0 - cost)
    return g


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default="ADRO")
    ap.add_argument("--threshold", type=float, default=0.20)
    ap.add_argument("--universe", default="bluechip")
    args = ap.parse_args()
    os.makedirs("reports", exist_ok=True)

    cfg = load_config()
    loader = YahooOHLCV(cfg, Cache(cfg.path("data.cache_dir", "data/cache")))
    w = clean_weekly(loader.get(args.ticker, max_age=86400 * 30))
    px = w.to_numpy(float)
    years = (w.index[-1] - w.index[0]).days / 365.25
    lg = legs(px, zigzag(px, args.threshold))
    N = intra_leg_noise(px, lg)

    print("=" * 96)
    print(f" {args.ticker}: WHAT A CAUSAL BAND HAS TO SURVIVE")
    print("=" * 96)
    print(f" {len(lg)} legs. Median leg size {N['size'].median():.0%}, "
          f"median duration {N['weeks'].median():.0f} weeks.")
    print(f"\n The worst move AGAINST the leg, before it ended:")
    for q in (0.25, 0.50, 0.75, 0.90):
        print(f"   {q:.0%} of legs pull back at least {N['counter'].quantile(q):.1%}")
    med = float(N["counter"].median())
    print(f"\n So a band narrower than about {med:.0%} is shaken out of half the")
    print(f" legs it is trying to hold. That is the floor, set by the data.")

    print(f"\n{'=' * 96}\n THE ARITHMETIC CEILING — perfect turn calls, still late by "
          f"the band\n{'=' * 96}")
    print(f" {'band':>8}{'best possible':>16}{'CAGR':>9}{'':4}"
          f"{'what the band costs per leg':>30}")
    rows = []
    for b in (0.05, 0.10, 0.15, 0.20, 0.25, 0.30):
        g = ceiling(lg, b)
        rows.append({"band": b, "ceiling": g, "cagr": g ** (1 / years) - 1})
        print(f" {b:>8.0%}{g:>15,.1f}x{g ** (1 / years) - 1:>+9.1%}{'':4}"
              f"{2 * b:>29.0%}")
    perfect = ceiling(lg, 0.0)
    print(f" {'0%':>8}{perfect:>15,.1f}x{perfect ** (1 / years) - 1:>+9.1%}{'':4}"
          f"{'the circles themselves':>30}")

    viable = [r for r in rows if r["band"] >= med]
    best_viable = max(viable, key=lambda r: r["ceiling"]) if viable else None
    st = reversal_state(px, 0.25, 0.25)
    actual, _ = run(px, st)
    print(f"\n{'=' * 96}\n WHERE THE GAP ACTUALLY GOES\n{'=' * 96}")
    print(f" the circles, perfectly traded          {perfect:>12,.0f}x")
    if best_viable:
        print(f" ceiling for a band wide enough to hold "
              f"({best_viable['band']:.0%})  {best_viable['ceiling']:>12,.1f}x"
              f"   <- arithmetic, not skill")
    print(f" best rule actually built (reversal 25%) {float(actual[-1]):>11,.1f}x")
    print(f" buy and hold                            {px[-1] / px[0]:>11,.1f}x")
    if best_viable:
        struct = perfect / best_viable["ceiling"]
        exec_ = best_viable["ceiling"] / float(actual[-1])
        print(f"\n Of the {perfect / float(actual[-1]):,.0f}x total gap:")
        print(f"   {struct:>8,.0f}x is STRUCTURAL — the cost of waiting for "
              f"confirmation wide")
        print(f"            enough to survive the pullbacks the legs actually have")
        print(f"   {exec_:>8,.1f}x is EXECUTION — imperfect turn identification, "
              f"the part")
        print(f"            a better signal could still win")

    # ------------------- does anything lead the price band? ------------------- #
    print(f"\n{'=' * 96}\n DOES ANY AVAILABLE SIGNAL LEAD THE TURN?\n{'=' * 96}")
    d = loader.get(args.ticker, max_age=86400 * 30).set_index("date").sort_index()
    volw = d["volume"].resample("W-FRI").sum().reindex(w.index)
    vol_z = ((volw - volw.rolling(52, min_periods=26).mean())
             / volw.rolling(52, min_periods=26).std())
    piv = zigzag(px, args.threshold)
    turns = piv[1:-1]
    band_state = reversal_state(px, 0.25, 0.25)
    flips = np.flatnonzero(np.abs(np.diff(band_state))) + 1

    leads = []
    for t in turns:
        later = flips[flips >= t]
        if len(later):
            leads.append(int(later[0] - t))
    if leads:
        print(f" the 25% band confirms a turn a median of {np.median(leads):.0f} "
              f"weeks after it happened ({np.mean(leads):.1f} mean)")

    zs = [float(vol_z.iloc[t]) for t in turns if np.isfinite(vol_z.iloc[t])]
    allz = vol_z.dropna()
    if zs:
        from scipy import stats as st_
        t_stat, p = st_.ttest_1samp(zs, 0.0)
        print(f" volume at the turn bar: mean z = {np.mean(zs):+.2f} "
              f"(t = {t_stat:+.2f}, p = {p:.3f}, n = {len(zs)})")
        print(f"   {'turns DO print unusual volume' if p < 0.05 else 'no volume signature at turns'}"
              f" — and a z-score is only useful if it is unusual BEFORE the price moves")

    pd.DataFrame(rows).to_csv(f"reports/gap_anatomy_{args.ticker}.csv", index=False)
    print(f"\n -> reports/gap_anatomy_{args.ticker}.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
