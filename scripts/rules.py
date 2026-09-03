#!/usr/bin/env python3
"""The entry and exit rules for the H54 basket, as levels you can act on.

THE RULE IS RELATIVE, NOT ABSOLUTE, AND THAT IS THE FIRST THING TO UNDERSTAND
ABOUT THE LEVELS BELOW. Both conditions are CROSS-SECTIONAL percentiles of the
eligible universe on the review day, so the sell level moves as the rest of the
board moves. A name can be sold without falling a rupiah if the other 240
eligible names rally past it. The printed price is therefore "where the line
sits TODAY", not a stop you leave resting with your broker — and the difference
matters, because a resting stop would fire on days the rule would not sell and
would sit idle on days it would.

WHAT IS MEASURED (H54, `reports/beathold.md`):
  * this rule beat the IHSG in 6 of 6 rebalance calendars, median +6.50%/yr;
  * it beat a random basket from its own universe in essentially every
    calendar, which is the strongest selection result in this project;
  * it did NOT clear the full H54 bar, and the four arms that cleared the
    weaker bar did so at 4 of 6 calendars over heavily overlapping windows,
    which is a robustness check and not a significance statement;
  * the 24-month holdout was spent at H16, so every number is in-sample.

WHY THERE IS NO STOP AND NO TAKE-PROFIT IN THIS FILE. Not an oversight, and not
conservatism — it is the single most-tested question in this repo and the answer
has never changed across 169 exit configurations:
  * H17's 32 exit rules and H18's 58 indicator rules: their headline wins were
    WITHDRAWN by H20, which redid them on portfolio accounting. `trail 15%
    armed +50%` turned a 6.4x buy-and-hold into 1.6x, at +2.4%/yr against
    hold's +10.5%. They won a MEDIAN by cutting the right tail, and the right
    tail is where the return is.
  * H35 and H38: 55 (take-profit, stop) and placement combinations, and NOT ONE
    is positive in both halves.
  * H40: 13 stop rules, 0 beat buy-and-hold. The best was the dumbest — a plain
    wide percentage trail — and the EMA stops had the lowest win rate in the
    table (22.5%), because price crosses an EMA on noise.
  * H20: `stop 25%` caps every name at −25% and produced the WORST PORTFOLIO
    DRAWDOWN in its table, −68.7%, because it realises losses 29 times against
    hold's 18 and redeploys straight back into the same bad regime.
    Position-level risk control is not portfolio-level risk control.
  * H47: 11 sell-off detectors, every one gives back MORE of the peak than a
    coin flip of the same speed, because a detector fires BECAUSE price fell.

So the exit here is the screen ceasing to hold, evaluated quarterly. Adding a
stop to it is a change this repo has measured and rejected six ways.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))
sys.path.insert(0, os.path.dirname(__file__))

from beathold import Sticky                                       # noqa: E402
from bhbench import MIN_TV, load                                  # noqa: E402
from paint_suite import tick_of                                   # noqa: E402

#  H54's leading arm: `sticky tight buffer, quarterly`.
ENTRY_HI, ENTRY_VOL = 0.90, 0.50      # to BUY: top 10% on hi52, calmest half
KEEP_HI, KEEP_VOL = 0.80, 0.60        # to KEEP: top 20% on hi52, calmest 60%
K = 10
FEE = 0.0056


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--held", default="", help="comma-separated current book")
    args = ap.parse_args()

    P = load()
    last = P[P["elig"]]["date"].max()
    day = P[(P["date"] == last) & P["elig"]].copy()
    day = day.dropna(subset=["hi52", "vol60"])

    hi_entry = day["hi52"].quantile(ENTRY_HI)
    hi_keep = day["hi52"].quantile(KEEP_HI)
    vol_entry = day["vol60"].quantile(ENTRY_VOL)
    vol_keep = day["vol60"].quantile(KEEP_VOL)

    sticky = Sticky(K, keep_hi_q=KEEP_HI, keep_vol_q=KEEP_VOL)
    if args.held:
        sticky.held = [t.strip().upper() for t in args.held.split(",")]
    picks = [t for t, _ in sticky(day)]

    print(f"REVIEW DATE {pd.Timestamp(last).date()}   "
          f"{len(day)} eligible names\n")
    print("THE TWO LINES, as they sit today (both are CROSS-SECTIONAL "
          "percentiles and move each quarter):")
    print(f"  to BUY  : hi52 >= {hi_entry:.4f} (top 10% of the board)   AND  "
          f"vol60 <= {vol_entry:.4f} (calmest half)")
    print(f"  to KEEP : hi52 >= {hi_keep:.4f} (top 20%)              AND  "
          f"vol60 <= {vol_keep:.4f} (calmest 60%)")
    print()

    d = day[day["ticker"].isin(picks)].copy()
    #  hi52 = close / 252-day max, so a hi52 threshold IS a price.
    d["hi52w"] = d["close"] / d["hi52"]
    d["sell_px"] = d["hi52w"] * hi_keep
    d["room"] = d["sell_px"] / d["close"] - 1.0
    d["vol_head"] = vol_keep / d["vol60"] - 1.0
    d["rt_cost"] = FEE + 0.5 * np.array(
        [tick_of(p) for p in d["close"].to_numpy(float)]) / d["close"]
    d["tvbn"] = d["tv60"] / 1e9
    d = d.sort_values("hi52", ascending=False)

    print(f"{'ticker':<8}{'close':>9}{'52w high':>10}{'sell below':>12}"
          f"{'room':>8}{'vol60':>8}{'vol room':>10}{'cost r/t':>10}"
          f"{'Rp bn/d':>9}")
    print("-" * 84)
    for _, r in d.iterrows():
        print(f"{r['ticker']:<8}{r['close']:>9,.0f}{r['hi52w']:>10,.0f}"
              f"{r['sell_px']:>12,.0f}{r['room']:>8.1%}{r['vol60']:>8.2%}"
              f"{r['vol_head']:>10.0%}{r['rt_cost']:>10.2%}{r['tvbn']:>9,.1f}")
    print()
    print(f"equal weight = {1 / max(len(d), 1):.1%} each; "
          f"basket round-trip cost {d['rt_cost'].mean():.2%}")
    print()
    print("READ 'sell below' AS: at the NEXT QUARTERLY REVIEW, if the close is")
    print("under that price the name has left the keep band and is sold. It is")
    print("NOT a resting stop -- the threshold is a percentile of the board and")
    print("moves. A name can also be sold with no price fall at all, if its")
    print("60-day volatility rises past the calmest-60% line (the 'vol room'")
    print("column is how much room it has) or if it drops out of the eligible")
    print("universe (turnover under Rp1bn/day, or suspended).")


if __name__ == "__main__":
    main()
