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

THERE IS A HARD STOP, AND IT IS HERE BECAUSE I WAS WRONG TO LEAVE IT OUT.
H56 bolted stops onto THIS rule, with portfolio accounting and daily exit
checks, and its pre-registered S1 — "a hard stop will not cut the PORTFOLIO's
maximum drawdown" — FAILED. Every stop level tested cut it, in both halves:

    arm              maxDD    DD early   DD late    CAGR early / late
    no stop          -42.0%    -37.2%    -39.9%       10.83% / 11.43%
    stop -10%        -30.7%    -27.1%    -23.4%        8.92% / 11.42%
    stop -15%        -34.1%    -25.7%    -25.6%       10.35% / 13.03%
    stop -20%        -36.3%    -26.9%    -29.4%        9.87% / 11.82%
    stop -30%        -40.0%    -31.2%    -34.9%        9.47% / 11.24%

Worst single-name loss falls from -91% to -30%. And on RETURN no stop beats the
base in both halves — every one is worse early and better late, which A18 records
as the signature of regime noise rather than skill. So the stop is a RISK
decision that costs approximately nothing, not a return edge, and that is the
only basis on which it is shipped.

WHY H20 SAID THE OPPOSITE AND WHY BOTH ARE RIGHT. H20 measured `stop 25%`
producing the WORST portfolio drawdown in its table, -68.7%. Its entry rule was
the H16 multiplier screen — a lottery-ticket selector whose whole premise was a
fat left tail bought in exchange for a fat right one (A15: "for a rule selected
ON P(2x), cutting the left tail cuts the premise"). This rule is selected on
strength-plus-calm, where P(a name halves) is 4.1%. The left tail is not the
premise here, so cutting it is cheap. A result about exits is a result about the
ENTRY it was measured on, and I cited six studies of other entries at a question
about this one — which is an argument, not a measurement.

THE TAKE-PROFIT IS A WIDE ONE, AND THE COST CURVE IS WHY.
S3's predicted null was CONFIRMED — a target costs money — but the first sweep
stopped at +50% and so could not say WHICH target costs least. Extended (H56b):

    target        CAGR     cost vs no target
    +20%         6.51%          -3.81
    +30%         7.90%          -2.42
    +50%         8.86%          -1.46
    +75%         9.79%          -0.53
    +100%       10.31%          -0.01   <- the cost curve flattens to zero here
    +150%       10.21%          -0.11
    sell HALF at +100%    10.33%   +0.01
    sell THIRD at +100%   10.30%   -0.02

So the level is NOT an argmax: it is where a monotone cost curve reaches zero.
A tighter target is available on request and its price is in the table above —
+30% costs 2.4 points of CAGR a year, which is most of the edge.

WHAT SHIPS, and each leg was chosen on its own curve rather than as a
combination: stop -20% (middle of a flat family) plus SELL HALF AT +100%
(where the target's cost curve flattens). Measured together:

    arm                              CAGR    maxDD   early / late   worst 1
    BASE, band only                 10.32%  -42.0%  10.87 / 11.40    -91%
    SHIPPED, stop + half at +100%   11.08%  -35.2%   9.96 / 12.40    -41%

Better CAGR, 7 points less drawdown, worst single name less than half as bad.
BUT the early half is WORSE (9.96 against 10.87) and the late half better, which
A18 records as the signature of regime noise, so the CAGR gain is not claimed —
only the drawdown, which improves in both halves.

Every prior measurement of TIGHT targets still stands and is why the level is
wide rather than conventional:
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

AND WHY THE KEEP BAND IS NOT CHECKED DAILY. S2 was the honest form of the
objection — "quarterly leaves a hole" — and it fails badly: checking the band
every session takes CAGR from 10.24% to 2.37%, because the band is a
CROSS-SECTIONAL PERCENTILE that moves every day, so a daily check sells on the
board's noise rather than on the name's deterioration (24.1 exits a year against
16.3). The band is a quarterly instrument. The HARD STOP is what covers the gap
between reviews, which is exactly the hole it was pointed at.
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
#  H56. The whole family -10% to -30% cuts portfolio drawdown in BOTH halves at
#  no measurable cost in return, so the level is not a tuned parameter. -20% is
#  the MIDDLE of that family and deliberately not its argmax: -15% measured the
#  best CAGR (11.77%) and the best drawdown-by-half, but it is the peak of a
#  five-point sweep and this repo has been burned by argmaxes (A11's O1, A21's
#  8-of-10 cell). Tighter cuts more drawdown and costs a little more in the
#  early half; the frontier is printed so the choice is visible.
STOP = 0.20
#  H56b. The target's cost curve is monotone in tightness and reaches zero at
#  +100%, so this is a curve-shape choice, not an argmax. Selling only HALF
#  there is free (+0.01) because it banks a double while leaving the rest to
#  run -- the only form of profit-taking that does not cap the winner.
TP = 1.00
TP_FRAC = 0.5


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

    #  The hard stop is from the ENTRY price, so for a book opened today it is
    #  today's close; for names already held, pass --held and use your own fill.
    d["stop_px"] = d["close"] * (1.0 - STOP)
    d["tp_px"] = d["close"] * (1.0 + TP)
    print(f"{'ticker':<7}{'ENTRY':>9}{'SL -20%':>10}{'TP +100%':>11}"
          f"{'band exit':>11}{'room':>7}{'vol60':>8}{'vol room':>10}"
          f"{'cost r/t':>10}{'Rp bn/d':>9}")
    print("-" * 92)
    for _, r in d.iterrows():
        print(f"{r['ticker']:<7}{r['close']:>9,.0f}{r['stop_px']:>10,.0f}"
              f"{r['tp_px']:>11,.0f}{r['sell_px']:>11,.0f}{r['room']:>7.1%}"
              f"{r['vol60']:>8.2%}"
              f"{r['vol_head']:>10.0%}{r['rt_cost']:>10.2%}{r['tvbn']:>9,.1f}")
    print()
    print(f"equal weight = {1 / max(len(d), 1):.1%} each; "
          f"basket round-trip cost {d['rt_cost'].mean():.2%}")
    print()
    print("THREE LEVELS, AND WHAT EACH ONE MEASURED. Sell HALF at TP; the")
    print("rest keeps running to the band exit or the stop.")
    print()
    print(f"  ENTRY      today's close. Buy line: hi52 >= {hi_entry:.4f} AND "
          f"vol60 <= {vol_entry:.4f}.")
    print("             Equal weight, %.0f%% each." % (100.0 / max(len(d), 1)))
    print()
    print("  SL  -20%   RESTING ORDER from your own fill, live every session.")
    print("             MEASURED: portfolio drawdown -42.0% -> -36.3%, and in")
    print("             BOTH halves (-37/-40 -> -27/-29). Worst single name")
    print("             -91% -> -41%. Costs nothing in return.")
    print()
    print("  TP +100%   SELL HALF, let the rest run. RESTING ORDER.")
    print("             MEASURED: a target's cost is monotone in how tight it")
    print("             is -- +20% costs 3.81 points of CAGR a year, +30%")
    print("             costs 2.42, +50% costs 1.46, +75% costs 0.53, and")
    print("             +100% costs 0.01. This is where the curve reaches")
    print("             zero, not where a sweep peaked. Selling only half")
    print("             there measures +0.01, i.e. free.")
    print("             A TIGHTER TARGET IS YOURS TO SET -- the price of each")
    print("             is in the line above.")
    print()
    print("  BAND EXIT  NOT a resting order: checked at the QUARTERLY review")
    print("             only. The threshold is a percentile of the board, so")
    print("             it moves -- a name can be sold without falling a")
    print("             rupiah if the board rallies past it, if its 60-day")
    print("             vol rises past the calmest-60% line ('vol room' is the")
    print("             headroom), or if it leaves the universe.")
    print("             MEASURED: checking this line DAILY is a disaster --")
    print("             CAGR 10.32% -> 2.41%, because it then sells on the")
    print("             board's noise rather than the name's decline.")
    print()
    print("  TOGETHER   11.08%/yr against 10.32% with none, drawdown -35.2%")
    print("             against -42.0%, over 2000-2026 on 6 rebalance")
    print("             calendars. The CAGR gain is NOT claimed: it is worse")
    print("             in the early half and better in the late one, which is")
    print("             regime noise. The drawdown gain holds in both halves.")
    print("             IN-SAMPLE: the holdout was spent at H16.")


if __name__ == "__main__":
    main()
