#!/usr/bin/env python3
"""Extract the pullback events whose outcome nothing in price could predict.

Result 93 measured the wall: on a 15% band, **71% of exits are false** - the
price is higher within 13 weeks - and that rate is 70-75% under every veto
tried, whether the veto looked at price, volume, the stock's trend or the
market's trend. Whatever separates a shakeout from a real turn is not in any of
those.

This builds the dataset to test the one candidate left. For every pullback of at
least ``--band`` from a running high inside a rising leg, it records:

    the window          peak date -> the date the band was breached
    the outcome         did the price exceed the old high within 13 weeks?
                        yes = FALSE exit (a shakeout), no = TRUE exit (a turn)

The window is what makes the broker-summary query cheap. IndoPremier's module
answers a whole date range in ONE request, so a pullback of any length costs one
call, not one per session - and the figures come back abbreviated, which matters
not at all because the question is directional: over this pullback, was foreign
money net buying or net selling?

Nothing here fetches. This writes the event list so the size of the fetch is
known before a single request is made.

    python3 scripts/pullback_events.py [--band 0.15]
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

from idxbot.config import load_config          # noqa: E402
from idxbot.data.cache import Cache            # noqa: E402
from idxbot.data.ohlcv import YahooOHLCV       # noqa: E402
from turn_trader import MIN_TURNOVER, clean_weekly   # noqa: E402

MIN_WEEKS = 100
HORIZON = 13            # weeks allowed for the old high to be exceeded
IPOT_START = "2010-01-01"


def events(w: pd.Series, band: float, horizon: int = HORIZON) -> List[Dict]:
    """Every ``band`` pullback from a running high, with what happened next.

    Walks forward the way a trader would: track the high since the last event,
    and when price closes ``band`` below it, that is an exit signal. Whether it
    was a mistake is decided by what the price did over the following
    ``horizon`` weeks - which is the future, and is used ONLY as the label.
    """
    px = w.to_numpy(float)
    idx = w.index
    out: List[Dict] = []
    hi = px[0]
    hi_i = 0
    for i in range(1, len(px)):
        if px[i] > hi:
            hi = px[i]
            hi_i = i
            continue
        if 1.0 - px[i] / hi >= band:
            end = min(i + horizon, len(px) - 1)
            future_max = px[i + 1:end + 1].max() if end > i else px[i]
            out.append({
                "peak_date": idx[hi_i], "peak_px": float(hi),
                "signal_date": idx[i], "signal_px": float(px[i]),
                "drawdown": float(px[i] / hi - 1.0),
                "weeks_from_peak": int(i - hi_i),
                "future_max": float(future_max),
                "recovered": bool(future_max > hi),          # exceeded the OLD high
                "bounced_5pct": bool(future_max > px[i] * 1.05),
            })
            hi = px[i]      # restart the search from here
            hi_i = i
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--band", type=float, default=0.15)
    ap.add_argument("--universe", default="bluechip")
    ap.add_argument("--start", default=IPOT_START)
    args = ap.parse_args()
    os.makedirs("reports", exist_ok=True)

    cfg = load_config()
    loader = YahooOHLCV(cfg, Cache(cfg.path("data.cache_dir", "data/cache")))
    names = sorted(set(cfg.universe(args.universe)) | set(cfg.universe("lq45")))
    raw = loader.get_many(names, max_age=86400 * 30, verbose=False)

    rows: List[Dict] = []
    for t, d in raw.items():
        if len(d) < 500:
            continue
        c = d["close"].astype(float)
        if float((c * d["volume"]).median()) < MIN_TURNOVER:
            continue
        w = clean_weekly(d)
        if w is None:
            continue
        w = w[w.index >= pd.Timestamp(args.start)]
        if len(w) < MIN_WEEKS:
            continue
        for e in events(w, args.band):
            rows.append({"ticker": t, **e})

    E = pd.DataFrame(rows).sort_values(["ticker", "signal_date"])
    E.to_csv("reports/pullback_events.csv", index=False)

    print("=" * 92)
    print(f" PULLBACK EVENTS — {args.band:.0%} band, {args.universe}+lq45, "
          f"from {args.start}")
    print("=" * 92)
    print(f" {len(E):,} events across {E['ticker'].nunique()} names")
    print(f" median drawdown at the signal   {E['drawdown'].median():.1%}")
    print(f" median weeks from peak to signal {E['weeks_from_peak'].median():.0f}")
    print(f"\n OUTCOMES (this is what has to be predicted)")
    print(f"   bounced 5%+ within {HORIZON}w        "
          f"{E['bounced_5pct'].mean():.0%}   <- the 'false exit' rate")
    print(f"   exceeded the OLD high within {HORIZON}w  {E['recovered'].mean():.0%}")
    print(f"\n a coin flip is 50%. A veto that predicts 'recovered' correctly")
    print(f" {'more' if E['recovered'].mean() != 0.5 else ''} than "
          f"{max(E['recovered'].mean(), 1 - E['recovered'].mean()):.0%} of the time "
          f"beats always guessing the majority class.")

    print(f"\n{'=' * 92}\n THE FETCH THIS IMPLIES\n{'=' * 92}")
    print(f" one range request per event = {len(E):,} requests")
    print(f" at 1.2s between calls that is {len(E) * 1.2 / 60:.0f} minutes")
    print(f" per name: median {E.groupby('ticker').size().median():.0f} events, "
          f"max {E.groupby('ticker').size().max()}")
    print(f"\n For comparison the day-at-a-time route would need "
          f"{int((E['weeks_from_peak'].sum()) * 5):,} requests for the same windows.")
    print(f" Range queries are {int((E['weeks_from_peak'].sum() * 5) / max(len(E), 1))}x "
          f"fewer, which is what makes this defensible at all.")
    print("\n -> reports/pullback_events.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
