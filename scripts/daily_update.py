#!/usr/bin/env python3
"""The twice-daily refresh: pull what is new, then say where every name stands.

Scheduled for 07:00 WIB (two hours before the 09:00 open) and 18:00 WIB (two
hours after the 15:50 close, by which time the broker summary for the session is
published).

WHAT THIS IS, AND WHAT IT IS NOT
--------------------------------
It is a POSITION REPORT. For every large cap it prints the leg colour on each
timeframe, the exact price that flips it, and how far that is from here. Those
numbers are arithmetic and they are exactly right.

It is NOT a buy list, and the header of every run says so, because the evidence
in this repo does not support one:

  * Result 100  the band signals lose to buy-and-hold by 1.5-2.6% a year across
                40 large caps, costs charged.
  * Result 110  the leg-size distribution of IDX large caps is indistinguishable
                from a driftless random walk at 15m, 1h, 4h and daily - median
                gap in leg/M* of -0.001. There is nothing in the price alone for
                a band to exploit.
  * Result 111  every band rule and every multi-timeframe combination tested
                loses to random timing at the same exposure.

What survives all of that is narrow and worth stating plainly: the rule cannot
MISS a move bigger than its band (Result 109), it repaints never, and its trigger
price is known in advance. So it is a good instrument for knowing where you are.
It is not evidence about where price goes next.

REFRESH POLICY
--------------
Daily bars come from Yahoo. Broker summary comes from IndoPremier, which is a
public page on a licensed member's site: requests are paced, everything is
cached permanently, and only names on the watchlist are touched. This is not a
bulk harvester and must not become one.

    python3 scripts/daily_update.py --session pre
    python3 scripts/daily_update.py --session post
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

from idxbot.config import load_config          # noqa: E402
from idxbot.data.cache import Cache            # noqa: E402
from idxbot.data.ohlcv import YahooOHLCV       # noqa: E402
from capture_toll import breakeven_move        # noqa: E402
from leg_signals import market_caps            # noqa: E402
from paint_daily import unadjusted_daily       # noqa: E402
from paint_live import band_state              # noqa: E402

WIB = dt.timezone(dt.timedelta(hours=7))
BANDS = {"weekly": 0.12, "daily": 0.08}
STALE_OK_HOURS = 6


def now_wib() -> dt.datetime:
    return dt.datetime.now(tz=WIB)


def refresh_prices(loader: YahooOHLCV, tickers: List[str],
                   max_age: float) -> Dict[str, int]:
    """Pull fresh daily bars, pacing the requests. Returns bars per ticker."""
    got = {}
    for i, t in enumerate(tickers, 1):
        try:
            d = loader.get(t, max_age=max_age)
            got[t] = 0 if d is None or d.empty else len(d)
        except Exception as exc:                      # keep going on one failure
            print(f"   ! {t}: {type(exc).__name__}: {exc}")
            got[t] = 0
        if i % 25 == 0:
            print(f"   ... {i}/{len(tickers)}")
    return got


def weekly_from_daily(s: pd.Series) -> pd.Series:
    return s.resample("W-FRI").last().dropna()


def status(loader: YahooOHLCV, ticker: str) -> Optional[Dict]:
    """Where this name stands on every timeframe, and what flips it."""
    s = unadjusted_daily(loader, ticker, start="2015-01-01")
    if s is None or len(s) < 260:
        return None
    out = {"ticker": ticker, "close": float(s.iloc[-1]),
           "asof": s.index[-1].date().isoformat()}
    for name, band in BANDS.items():
        px = (weekly_from_daily(s) if name == "weekly" else s).to_numpy(float)
        if len(px) < 60:
            continue
        st, trig = band_state(px, band)
        out[f"{name}_state"] = "GREEN" if st[-1] else "RED"
        out[f"{name}_trigger"] = float(trig[-1])
        out[f"{name}_gap"] = float(trig[-1] / px[-1] - 1.0)
        flips = np.flatnonzero(np.diff(st.astype(int)) != 0)
        out[f"{name}_bars_in_leg"] = int(len(st) - 1 - flips[-1]) if len(flips) else len(st)
    # context that does not depend on the band at all
    out["above_200d"] = float(s.iloc[-1] / s.rolling(200).mean().iloc[-1] - 1.0)
    out["ret_1m"] = float(s.iloc[-1] / s.iloc[-21] - 1.0) if len(s) > 21 else np.nan
    out["ret_ytd"] = float(
        s.iloc[-1] / s[s.index.year == s.index[-1].year].iloc[0] - 1.0)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", choices=["pre", "post"], default="post",
                    help="pre = 2h before the open; post = 2h after the close")
    ap.add_argument("--min-mcap", type=float, default=1e13)
    ap.add_argument("--names", type=int, default=60)
    ap.add_argument("--no-refresh", action="store_true",
                    help="report from cache without touching the network")
    args = ap.parse_args()
    os.makedirs("reports", exist_ok=True)

    t0 = now_wib()
    print(f"{'=' * 92}\n IDX DAILY UPDATE — {args.session.upper()} session, "
          f"{t0:%Y-%m-%d %H:%M} WIB\n{'=' * 92}")
    print(" This is a POSITION REPORT, not a buy list. The band rule loses to "
          "buy-and-hold\n (Result 100) and its leg structure matched a random "
          "walk at every timeframe\n tested (Result 110). What it does exactly "
          "is tell you where you are and what\n price changes that.\n")

    cfg = load_config()
    loader = YahooOHLCV(cfg, Cache(cfg.path("data.cache_dir", "data/cache")))
    mc = market_caps()
    universe = sorted(mc[mc >= args.min_mcap].index)[:args.names]
    print(f" universe: {len(universe)} names at Rp{args.min_mcap/1e12:,.0f}T+ "
          f"market cap")

    if not args.no_refresh:
        # pre-open wants yesterday's close, which is already final; post-close
        # wants today's, so it must not be served from a stale cache
        max_age = 3600.0 * (12 if args.session == "pre" else 1)
        print(f"\n refreshing daily bars (cache max age {max_age/3600:.0f}h)...")
        refresh_prices(loader, universe, max_age)

    rows = [r for r in (status(loader, t) for t in universe) if r]
    if not rows:
        raise SystemExit("no usable data")
    R = pd.DataFrame(rows).sort_values("daily_gap", key=abs)
    stamp = t0.strftime("%Y%m%d_%H%M")
    R.to_csv(f"reports/daily_update_{stamp}.csv", index=False)
    R.to_csv("reports/daily_update_latest.csv", index=False)

    asof = pd.to_datetime(R["asof"]).max()
    lag = (t0.date() - asof.date()).days
    print(f"\n data as of {asof:%Y-%m-%d} ({lag} calendar days behind "
          f"{t0:%Y-%m-%d})")
    if lag > 4:
        print(" ! the price cache is stale; the colours below are from an old "
              "close and the\n ! trigger distances are wrong by however much "
              "price has moved since.")

    green_d = int((R["daily_state"] == "GREEN").sum())
    green_w = int((R["weekly_state"] == "GREEN").sum())
    print(f"\n{'=' * 92}\n MARKET-WIDE — how much of the board is green\n"
          f"{'=' * 92}")
    print(f" weekly {BANDS['weekly']:.0%} band: {green_w}/{len(R)} green "
          f"({green_w/len(R):.0%})")
    print(f" daily  {BANDS['daily']:.0%} band: {green_d}/{len(R)} green "
          f"({green_d/len(R):.0%})")
    print(f" above the 200-day average: {int((R['above_200d'] > 0).sum())}"
          f"/{len(R)}")
    print(f" break-even leg at the daily band: "
          f"{breakeven_move(BANDS['daily']):.1%} — a leg smaller than that "
          f"loses by construction")

    print(f"\n{'=' * 92}\n CLOSEST TO FLIPPING — where a small move changes the "
          f"picture\n{'=' * 92}")
    print(f" {'ticker':<8}{'close':>10}{'weekly':>9}{'daily':>8}"
          f"{'flips at':>11}{'move':>9}{'in leg':>8}{'vs 200d':>10}{'1m':>8}")
    for _, r in R.head(20).iterrows():
        print(f" {r['ticker']:<8}{r['close']:>10,.0f}"
              f"{r.get('weekly_state', '-'):>9}{r.get('daily_state', '-'):>8}"
              f"{r.get('daily_trigger', float('nan')):>11,.0f}"
              f"{r.get('daily_gap', float('nan')):>+9.1%}"
              f"{int(r.get('daily_bars_in_leg', 0)):>7}d"
              f"{r.get('above_200d', float('nan')):>+10.1%}"
              f"{r.get('ret_1m', float('nan')):>+8.1%}")

    both = R[(R["weekly_state"] == "GREEN") & (R["daily_state"] == "GREEN")]
    print(f"\n both timeframes green: {len(both)} names"
          + (f" — {', '.join(both['ticker'].head(12))}" if len(both) else ""))

    payload = {"generated": t0.isoformat(), "session": args.session,
               "asof": asof.date().isoformat(), "names": len(R),
               "green_daily": green_d, "green_weekly": green_w,
               "stale_days": lag}
    with open("reports/daily_update_latest.json", "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\n -> reports/daily_update_{stamp}.csv, "
          f"reports/daily_update_latest.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
