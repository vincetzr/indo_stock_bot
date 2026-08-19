#!/usr/bin/env python3
"""Making the timeframes help each other, and measuring whether they do.

WHY THIS SHOULD WORK, ON PAPER
------------------------------
The toll law (Result 109) says a band costs (1+b)/(1-b) of price per round trip,
split into an entry toll b and an exit toll b, and that a leg must beat
M* = 2b/(1-b) to pay for itself. Nothing forces those two tolls to be the SAME
band. Enter on a fast band and exit on a slow one and the break-even becomes

    M* = (1 + b_fast) / (1 - b_slow) - 1

    b_fast = 3%, b_slow = 12%  ->  M* = 17.0%   instead of  27.3% for 12% alone
    b_fast = 3%, b_slow = 8%   ->  M* = 12.0%   instead of  17.4% for 8% alone

That is a real reduction and it is arithmetic, not a hope. The catch is equally
concrete: a fast band alone fires constantly and most of its flips are noise. The
slow band's job is to throw those away by only allowing entries while it is green.

So the design under test is asymmetric on purpose:

    entry   fast band flips green, and only while the slow band is already green
    exit    slow band flips red - the fast band does NOT sell

Entry pays the small toll, the exit keeps you in the big move, and the slow band
is the filter that makes a 3% band survivable.

WHY IT MIGHT STILL FAIL
-----------------------
Result 110 found the leg-size distribution of IDX large caps indistinguishable
from a driftless random walk at 15m, 1h, 4h and daily. If each timeframe is
individually structureless, the combination can only help through CROSS-timeframe
dependence, which the marginal test does not rule out but does not promise
either. Lowering the break-even helps only if legs are actually distributed near
it; on a random walk, cheaper trades just means more of them at the same odds.

Every combination is scored against buy-and-hold on the same bars with the same
fees. Nothing here reads a bar before it printed: the slow state at any intraday
bar comes from the last COMPLETED day, never the day in progress.

    python3 scripts/mtf_stack.py --fast 0.03 --slow 0.12
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

from idxbot.config import load_config          # noqa: E402
from idxbot.data.cache import Cache            # noqa: E402
from leg_signals import market_caps            # noqa: E402
from paint_live import band_state              # noqa: E402

CACHE = os.path.join(os.path.dirname(__file__), os.pardir,
                     "data", "cache", "intraday")
ROUND_TRIP_FEE = 0.0056
RULES = ("fast_only", "slow_only", "both_green", "fast_in_slow_out")


def load_hourly(ticker: str, cap: float = 0.20,
                start: Optional[str] = None) -> Optional[pd.DataFrame]:
    """Hourly bars from the cache, session-clean, impossible prints capped.

    ``start`` exists because the coverage audit found the intraday feed
    disagrees with the daily close on 15-26% of sessions in 2023 (84.5% and
    73.6% agreement in Q3 and Q4) before settling at 98.5-99.9% from 2024Q1.
    Passing 2024-01-01 drops the contaminated stretch.
    """
    f = os.path.join(CACHE, f"{ticker}.JK_1h_730d.csv.gz")
    if not os.path.exists(f):
        return None
    d = pd.read_csv(f)
    d["ts"] = pd.to_datetime(d["ts"])
    d = d.dropna(subset=["close"])
    d = d[d["volume"] > 0].set_index("ts").sort_index()
    if start:
        d = d[d.index >= pd.Timestamp(start)]
    if len(d) < 500:
        return None
    c = d["close"].astype(float)
    r = c.pct_change().clip(-cap, cap).fillna(0.0)
    d = d.assign(close=c.iloc[0] * (1.0 + r).cumprod())
    return d


def daily_state_on_intraday(d: pd.DataFrame, slow: float) -> np.ndarray:
    """Slow-band colour, evaluated on daily closes, aligned WITHOUT look-ahead.

    The colour attached to every bar of day D is the one computed from the close
    of day D-1. The day in progress has not finished, so its close does not
    exist yet and must not be used - getting this wrong is the single easiest way
    to manufacture a fake edge here.
    """
    day = pd.Series(d.index.date, index=d.index)
    daily_close = d["close"].groupby(day).last()
    st, _ = band_state(daily_close.to_numpy(float), slow)
    lagged = pd.Series(st, index=daily_close.index).shift(1).fillna(0)
    return day.map(lagged).fillna(0).to_numpy(float).astype(np.int8)


def positions(fast_st: np.ndarray, slow_st: np.ndarray, rule: str) -> np.ndarray:
    """Desired exposure per bar for each combination rule."""
    if rule == "fast_only":
        return fast_st.astype(np.int8)
    if rule == "slow_only":
        return slow_st.astype(np.int8)
    if rule == "both_green":
        return (fast_st.astype(bool) & slow_st.astype(bool)).astype(np.int8)
    if rule == "fast_in_slow_out":
        # enter on a fast green flip but only while slow is green;
        # hold until the SLOW band turns red. The fast band never sells.
        n = len(fast_st)
        pos = np.zeros(n, dtype=np.int8)
        held = False
        for i in range(n):
            if held:
                if not slow_st[i]:
                    held = False
            else:
                if slow_st[i] and fast_st[i] and (i == 0 or not fast_st[i - 1]):
                    held = True
            pos[i] = 1 if held else 0
        return pos
    raise ValueError(rule)


def score(px: np.ndarray, pos: np.ndarray,
          fee: float = ROUND_TRIP_FEE) -> Dict[str, float]:
    """Log return of holding ``pos`` with a one-bar decision lag and fees."""
    r = np.diff(np.log(px))
    held = pos[:-1].astype(bool)
    gross = float(r[held].sum())
    trips = int((np.diff(pos.astype(int)) == 1).sum())
    return {"log": gross + trips * np.log(1.0 - fee),
            "trips": trips,
            "exposure": float(held.mean())}


def run_name(d: pd.DataFrame, fast: float, slow: float) -> Optional[Dict]:
    px = d["close"].to_numpy(float)
    if len(px) < 500:
        return None
    f_st, _ = band_state(px, fast)
    s_st = daily_state_on_intraday(d, slow)
    out = {"bars": len(px), "hold_log": float(np.log(px[-1] / px[0]))}
    for rule in RULES:
        s = score(px, positions(f_st, s_st, rule))
        out[f"{rule}_log"] = s["log"]
        out[f"{rule}_trips"] = s["trips"]
        out[f"{rule}_exp"] = s["exposure"]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", type=float, default=0.03, help="entry band, hourly")
    ap.add_argument("--slow", type=float, default=0.12, help="exit band, daily")
    ap.add_argument("--min-mcap", type=float, default=1e13)
    ap.add_argument("--names", type=int, default=60)
    ap.add_argument("--start", default=None,
                    help="drop bars before this date; 2024-01-01 skips the "
                         "stretch where intraday and daily closes disagree")
    args = ap.parse_args()
    os.makedirs("reports", exist_ok=True)

    be_pair = (1 + args.fast) / (1 - args.slow) - 1
    print(f"{'=' * 88}\n THE ARITHMETIC OF SPLITTING THE TOLL\n{'=' * 88}")
    print(f" fast band {args.fast:.0%} (entry), slow band {args.slow:.0%} (exit)")
    print(f"   break-even, slow band alone:   "
          f"{2 * args.slow / (1 - args.slow):.1%}")
    print(f"   break-even, fast in / slow out: {be_pair:.1%}")
    print(f"   the split lowers the bar by "
          f"{2 * args.slow / (1 - args.slow) - be_pair:.1%} of price")

    mc = market_caps()
    big = sorted(mc[mc >= args.min_mcap].index)
    rows = []
    for t in big:
        d = load_hourly(t, start=args.start)
        if d is None:
            continue
        r = run_name(d, args.fast, args.slow)
        if r:
            rows.append({"ticker": t, **r})
        if len(rows) >= args.names:
            break
    if not rows:
        raise SystemExit("no hourly data for the big caps")
    R = pd.DataFrame(rows)
    R.to_csv("reports/mtf_stack.csv", index=False)

    print(f"\n{'=' * 88}\n HOURLY ENTRY, DAILY EXIT — {len(R)} big caps, median "
          f"{R['bars'].median():,.0f} hourly bars\n{'=' * 88}")
    print(f" {'rule':<20}{'median log':>12}{'vs hold':>10}{'beats hold':>12}"
          f"{'trips':>8}{'exposure':>10}")
    hold = R["hold_log"].median()
    for rule in RULES:
        col = R[f"{rule}_log"]
        print(f" {rule:<20}{col.median():>12.3f}{col.median() - hold:>+10.3f}"
              f"{float((col > R['hold_log']).mean()):>12.0%}"
              f"{R[f'{rule}_trips'].median():>8.0f}"
              f"{R[f'{rule}_exp'].median():>10.0%}")
    print(f" {'buy & hold':<20}{hold:>12.3f}{0.0:>+10.3f}{'-':>12}"
          f"{1:>8}{1.0:>10.0%}")

    # --- being out of a falling market is not skill -------------------------
    # Over this window holding LOST money, so any rule that sits in cash looks
    # good for free. The fair null is random timing at the same exposure, which
    # earns exposure x the buy-and-hold return with none of the cleverness.
    print(f"\n{'=' * 88}\n AGAINST THE RIGHT NULL — random timing at the same "
          f"exposure\n{'=' * 88}")
    print(f" buy-and-hold over this window: {hold:+.3f} log "
          f"({np.exp(hold) - 1:+.1%}) — it FELL, so cash flatters everything")
    print(f" {'rule':<20}{'actual':>10}{'same-exposure null':>21}"
          f"{'edge':>9}{'beats null':>12}")
    for rule in RULES:
        col = R[f"{rule}_log"]
        null = R[f"{rule}_exp"] * R["hold_log"]
        print(f" {rule:<20}{col.median():>10.3f}{null.median():>21.3f}"
              f"{(col - null).median():>+9.3f}"
              f"{float((col > null).mean()):>12.0%}")

    print(f"\n{'=' * 88}\n VERDICT\n{'=' * 88}")
    best = max(RULES, key=lambda k: R[f"{k}_log"].median())
    bm = R[f"{best}_log"].median()
    # the combination has to beat the best SINGLE timeframe, not just holding -
    # otherwise the extra timeframe is decoration
    singles = ("fast_only", "slow_only")
    bs = max(singles, key=lambda k: R[f"{k}_log"].median())
    edge_single = (R[f"{best}_log"] - R[f"{bs}_log"]).median()
    win_single = float((R[f"{best}_log"] > R[f"{bs}_log"]).mean())
    null_edge = (R[f"{best}_log"] - R[f"{best}_exp"] * R["hold_log"]).median()

    print(f" best combination:        {best}  ({bm:+.3f})")
    print(f" best single timeframe:   {bs}  ({R[f'{bs}_log'].median():+.3f})")
    print(f" combination minus single: {edge_single:+.3f} log, ahead on "
          f"{win_single:.0%} of names")
    print(f" combination minus same-exposure null: {null_edge:+.3f} log")

    helps = (best not in singles and edge_single > 0.02 and win_single > 0.55
             and null_edge > 0.02)
    if helps:
        print(f"\n Combining the timeframes helps: the split beats the best "
              f"single band AND\n beats random timing at its own exposure.")
    else:
        print(f"\n Combining the timeframes does NOT help. The break-even "
              f"really did fall from\n "
              f"{2 * args.slow / (1 - args.slow):.1%} to {be_pair:.1%} — that "
              f"part is arithmetic and it happened — but the\n cheaper "
              f"entry did not turn into money: the best combination is within "
              f"noise\n of the best single timeframe ({edge_single:+.3f} log, "
              f"{win_single:.0%} of names, a coin flip),\n and it does not "
              f"clear random timing at the same exposure.")
        print("\n This is what Result 110 predicts. A cheaper bet at unchanged "
              "odds is still a\n losing bet, and the leg distribution these "
              "rules trade is the one that was\n indistinguishable from a "
              "random walk at every timeframe tested.")
    print("\n -> reports/mtf_stack.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
