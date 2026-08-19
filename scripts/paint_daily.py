#!/usr/bin/env python3
"""The leg painter on DAILY bars, and what it costs to keep the same accuracy.

The weekly painter reproduces the hand-drawn segmentation above 90% for any bar
older than six weeks, and its closed legs never repaint. Moving to daily changes
one thing and one thing only: how many bars a swing of a given size takes to
resolve. The zigzag rule is scale-free - a 12% swing is a 12% swing - so what has
to be re-measured is the SETTLING TIME, in days, and whether a band that gives a
comparable number of legs still settles fast enough to be useful.

Two numbers are reported for every band:

    prefix identity   painting the first half with only the first half must
                      reproduce the full-series painting of it exactly. This is
                      the no-cheating check and it should be every name.
    settle curve      for a bar k days old, how often its colour already matches
                      the colour the finished series gives it - including the
                      leg still in progress, which is the honest version.

    python3 scripts/paint_daily.py --min-mcap 1e13
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
from leg_signals import market_caps            # noqa: E402
from legpaint import zigzag_labels             # noqa: E402
from paint_live import band_state              # noqa: E402
from swing_accuracy import legs, zigzag        # noqa: E402

DAILY_CAP = 0.35


def unadjusted_daily(loader: YahooOHLCV, ticker: str,
                     start: str = "2015-01-01") -> Optional[pd.Series]:
    """Daily close as charted: raw, not dividend-adjusted, impossible prints capped."""
    d = loader.get(ticker, max_age=86400 * 30)
    if d is None or len(d) < 500:
        return None
    d = d.set_index("date").sort_index()
    c = d["close"].astype(float).dropna()
    r = c.pct_change().clip(-DAILY_CAP, DAILY_CAP).fillna(0.0)
    c = c.iloc[0] * (1.0 + r).cumprod()
    return c[c.index >= pd.Timestamp(start)]


def settle(px: np.ndarray, band: float, ages: Tuple[int, ...],
           step: int = 5) -> Dict[int, float]:
    """Share of bars whose colour is already final at each age, running leg included."""
    final = zigzag_labels(px, band, drop_last=False)
    hit = {k: [0, 0] for k in ages}
    lo = max(120, max(ages) + 10)
    for now in range(lo, len(px), step):
        live = zigzag_labels(px[:now + 1], band, drop_last=False)
        for k in ages:
            i = now - k
            if i < 0:
                continue
            if np.isfinite(live[i]) and np.isfinite(final[i]):
                hit[k][1] += 1
                hit[k][0] += int(live[i] == final[i])
    return {k: (h / t if t else np.nan) for k, (h, t) in hit.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-mcap", type=float, default=1e13)
    ap.add_argument("--names", type=int, default=20,
                    help="cap the sample; the settle curve is expensive on daily bars")
    args = ap.parse_args()
    os.makedirs("reports", exist_ok=True)

    cfg = load_config()
    loader = YahooOHLCV(cfg, Cache(cfg.path("data.cache_dir", "data/cache")))
    mc = market_caps()
    big = sorted(mc[mc >= args.min_mcap].index)
    series: Dict[str, pd.Series] = {}
    for t in big:
        s = unadjusted_daily(loader, t)
        if s is not None and len(s) >= 900:
            series[t] = s
        if len(series) >= args.names:
            break
    print(f"{len(series)} big caps, daily bars, median "
          f"{int(np.median([len(s) for s in series.values()])):,} sessions each")

    AGES = (1, 5, 10, 20, 30, 45, 65, 90)
    bands = (0.08, 0.10, 0.12, 0.15, 0.20)
    rows = []
    for band in bands:
        prefix_ok = 0
        legs_n, curves = [], []
        for t, s in series.items():
            px = s.to_numpy(float)
            cut = len(px) // 2
            f, _ = band_state(px, band)
            p, _ = band_state(px[:cut], band)
            prefix_ok += int(np.array_equal(f[:cut], p))
            legs_n.append(len(legs(px, zigzag(px, band))))
            curves.append(settle(px, band, AGES))
        row = {"band": band, "prefix_identical": prefix_ok,
               "names": len(series), "median_legs": float(np.median(legs_n))}
        for k in AGES:
            row[f"d{k}"] = float(np.median([c[k] for c in curves]))
        rows.append(row)
        print(f"  band {band:.0%} done")

    R = pd.DataFrame(rows)
    R.to_csv("reports/paint_daily.csv", index=False)

    print(f"\n{'=' * 96}\n DAILY PAINTER — colour already final, by bar age (median "
          f"across names)\n{'=' * 96}")
    print(f" {'band':>6}{'legs':>7}{'no-cheat':>10}"
          + "".join(f"{str(k) + 'd':>8}" for k in AGES))
    for _, r in R.iterrows():
        print(f" {r['band']:>6.0%}{r['median_legs']:>7.0f}"
              f"{int(r['prefix_identical'])}/{int(r['names']):<7}"
              + "".join(f"{r[f'd{k}']:>8.0%}" for k in AGES))

    print(f"\n{'=' * 96}\n WHERE EACH BAND CROSSES 90%\n{'=' * 96}")
    for _, r in R.iterrows():
        first = next((k for k in AGES if r[f"d{k}"] >= 0.90), None)
        wk = f"{first / 5:.0f} weeks" if first else "never in 90 days"
        print(f" band {r['band']:>4.0%}: 90% reached at "
              f"{str(first) + ' days' if first else 'not within 90 days':<16}"
              f"({wk})   {r['median_legs']:.0f} legs")
    print("\n weekly reference: 12% band reached 92.4% at 6 weeks = 30 trading days")
    print("\n -> reports/paint_daily.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
