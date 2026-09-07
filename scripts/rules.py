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

WHAT IS MEASURED (H54, `reports/beathold.md`; re-measured on the H62 buffer):
  * this rule beat the IHSG in 6 of 6 rebalance calendars;
  * it beat a random basket from its own universe in essentially every
    calendar, which is the strongest selection result in this project;
  * it did NOT clear the full H54 bar, and the four arms that cleared the
    weaker bar did so at 4 of 6 calendars over heavily overlapping windows,
    which is a robustness check and not a significance statement;
  * the 24-month holdout was spent at H16, so every number is in-sample.

THERE IS A HARD STOP, AND IT IS HERE BECAUSE I WAS WRONG TO LEAVE IT OUT.
H56 bolted stops onto THIS rule, with portfolio accounting and daily exit
checks, and its pre-registered S1 — "a hard stop will not cut the PORTFOLIO's
maximum drawdown" — FAILED. Every stop level tested cut it.

RE-MEASURED ON THE H62 BUFFER (0.70/0.60), because the numbers below were taken
on the 0.80 one and `stoptest.py` held its own copy of that constant which did
not move when the shipped one did:

    arm              CAGR     maxDD    early / late    worst 1
    no stop         13.46%    -40.2%   12.71 / 11.60     -84%
    stop -10%       11.00%    -29.2%    9.64 / 11.74     -30%
    stop -15%       12.01%    -31.8%    9.04 / 13.86     -30%
    stop -20%       12.76%    -33.8%   10.51 / 12.30     -41%
    stop -25%       12.79%    -36.0%   11.07 / 12.07     -42%
    stop -30%       12.72%    -36.3%   12.80 / 12.32     -44%

AND THE SIGN OF THE RETURN EFFECT FLIPPED WHEN THE BUFFER MOVED. On the old
buffer the stop looked approximately free; here it costs 0.70 points of CAGR
(13.46 -> 12.76) and buys 6.4 points of drawdown and a worst single name of
-41% against -84%. That is a clearer version of what H56 always claimed — a
RISK decision, not a return edge — and it is the only basis on which it ships.
The family is still flat: every level from -10% to -30% lands between 11.00%
and 12.79%, so -20% remains the MIDDLE and not the argmax.

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
stopped at +50% and so could not say WHICH target costs least. Extended (H56b)
and RE-MEASURED on the H62 buffer:

    target                 CAGR     cost vs no target (13.46%)
    +20%                  6.62%          -6.84
    +30%                  8.29%          -5.17
    +50%                  9.68%          -3.78
    +75%                 10.73%          -2.73
    +100%                11.97%          -1.49
    +150%                12.18%          -1.28
    sell HALF at +50%    12.49%          -0.97
    sell THIRD at +100%  13.28%          -0.18
    sell HALF at +100%   13.18%          -0.28   <- what ships

THE COST CURVE NO LONGER REACHES ZERO, and that matters: on the old buffer the
+100% scale-out measured +0.01, i.e. free, and that was H56b's whole
justification for the level. On the shipped buffer it costs 0.28 points. The
curve is still monotone in tightness and the scale-out is still by far the
cheapest form of target, so the level stands — but it is now a small PRICE
rather than a free option, and quoting the old "+0.01" would have been quoting
a rule nobody ships. Selling a THIRD is marginally cheaper still (-0.18) and
banks less; both are printed rather than one chosen silently.

WHAT SHIPS, and each leg was chosen on its own curve rather than as a
combination: stop -20% (middle of a flat family) plus SELL HALF AT +100%
(the cheapest target on a monotone cost curve). Measured together on the H62
buffer:

    arm                              CAGR    maxDD   early / late   worst 1
    BASE, band only                 13.46%  -40.2%  12.71 / 11.60    -84%
    SHIPPED, stop + half at +100%   12.97%  -31.9%  11.04 / 12.42    -41%
    IHSG, total return, same span    6.81%

The CAGR now COSTS 0.49 points where the old buffer's table showed a 0.76-point
gain — the sign of that effect flipped when the buffer moved, which is why no
CAGR claim was ever made for it. What is bought is 8.3 points of portfolio
drawdown and a worst single name of -41% against -84%. The early half is worse
(11.04 against 12.71) and the late half better, A18's signature of regime
noise, so the return effect stays unclaimed in either direction and the
drawdown is the whole case.

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

from idxbot import measured                                       # noqa: E402

ENTRY_HI, ENTRY_VOL = 0.90, 0.50      # to BUY: top 10% on hi52, calmest half
#  H62. WAS `KEEP_HI, KEEP_VOL = 0.80, 0.60` -- H54's "tight buffer" arm, which
#  was the ARGMAX of a three-point sweep at ONE rebalance phase. A39 already
#  flagged it as unresolved ("the buffer sweep is not monotone ... the family's
#  spread is about as wide as its effect"), and the full 8x6 grid across six
#  phases says worse than unresolved:
#
#    * the spread ACROSS phases within one cell is +6.57%, more than DOUBLE
#      the +3.01% spread across the whole buffer grid -- so a buffer chosen at
#      one calendar is chosen on one draw (B1);
#    * five distinct cells win across six phases (B3);
#    * and keep_hi 0.80 is the WORST ROW of the grid, +4.01% to +5.09%, below
#      every other row. The argmax of three points at one phase turned out to
#      be the weakest setting once the grid and the phases were filled in.
#
#  So the constant moves to the MIDDLE of the family, by the rule registered
#  before the run and for the same reason H56 chose STOP = 0.20: not because
#  the middle measured better, but because a maximum this unstable is not a
#  measurement. The +1.92pp of median excess that comes with it is a SIDE
#  EFFECT and is not claimed -- B1 says the phase spread swamps it.
KEEP_HI, KEEP_VOL = 0.70, 0.60        # to KEEP: top 30% on hi52, calmest 60%
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
#  there costs 0.28 points on the shipped buffer -- it measured +0.01, i.e.
#  free, on the pre-H62 one, and that number is withdrawn -- because it banks a
#  double while leaving the rest to run, the only form of profit-taking that
#  does not cap the winner.
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
    #  THE LABELS ARE DERIVED FROM THE CONSTANTS, NOT TYPED NEXT TO THEM.
    #  They read "top 20%" for a full day after H62 moved KEEP_HI from 0.80 to
    #  0.70 -- the same drift this repo has now recorded five times, and the
    #  only fix that survives the next edit is to compute them.
    print(f"  to BUY  : hi52 >= {hi_entry:.4f} "
          f"(top {1 - ENTRY_HI:.0%} of the board)   AND  "
          f"vol60 <= {vol_entry:.4f} (calmest {ENTRY_VOL:.0%})")
    print(f"  to KEEP : hi52 >= {hi_keep:.4f} (top {1 - KEEP_HI:.0%})"
          f"           AND  "
          f"vol60 <= {vol_keep:.4f} (calmest {KEEP_VOL:.0%})")
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
    print(f"{'ticker':<7}{'ENTRY':>9}{f'SL {-STOP:.0%}':>10}{f'TP {TP:+.0%}':>11}"
          f"{'band exit':>11}{'room':>7}{'vol60':>8}{'vol room':>10}"
          f"{'cost r/t':>10}{'Rp bn/d':>9}")
    print("-" * 92)
    for _, r in d.iterrows():
        print(f"{r['ticker']:<7}{r['close']:>9,.0f}{r['stop_px']:>10,.0f}"
              f"{r['tp_px']:>11,.0f}{r['sell_px']:>11,.0f}{r['room']:>7.1%}"
              f"{r['vol60']:>8.2%}"
              f"{r['vol_head']:>10.0%}{r['rt_cost']:>10.2%}{r['tvbn']:>9,.1f}")
    #  ------------------------------------------------ SECTOR CONCENTRATION
    #  A DISCLOSURE, NOT A TILT. H59 measured sector momentum as worth about
    #  +5.05%/yr against a random sector choice at MATCHED concentration -- and
    #  measured that restricting the universe to three sectors costs MORE than
    #  that, so no sector rule enters the selection here. But this screen has
    #  no diversification constraint of any kind, and the IDX-IC map covers
    #  96.5% of the panel, so what the basket actually holds is knowable and is
    #  printed. A reader who sees four of ten names in two commodity sectors
    #  can size accordingly; one who is never told cannot.
    try:
        S = pd.read_parquet(os.path.join("data", "reference",
                                         "idx_classification.parquet"))
        smap = dict(zip(S["ticker"], S["sector"]))
    except Exception:                                       # noqa: BLE001
        smap = {}
    if smap:
        vc = pd.Series([smap.get(t, "unmapped")
                        for t in d["ticker"]]).value_counts()
        top = vc.iloc[0] / max(len(d), 1)
        print()
        print("SECTOR MIX — a DISCLOSURE, not a tilt. No sector rule enters "
              "the selection:")
        print("H59 measured that restricting to a few sectors costs more than "
              "choosing them")
        print("well is worth. This screen has NO diversification constraint, "
              "so what it")
        print("happens to hold is worth knowing.")
        print("  " + ",  ".join(f"{k} {int(v)}" for k, v in vc.items()))
        if top >= 0.30:
            print(f"  *** {top:.0%} of the basket sits in ONE sector "
                  f"({vc.index[0]}). A shock there hits")
            print("      that share of the book at once, and nothing in the "
                  "rule prevents it.")

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
    #  --------------------------------------- THE FOURTH COLUMN, READ NOT TYPED
    #  The standing instruction's fourth column is "the measured cost of each
    #  level", and until now every figure below was a LITERAL in a format
    #  string. H62 moved KEEP_HI and `stoptest.py` held its own copy that did
    #  not move, so the card went on describing a rule nobody ships and nothing
    #  failed -- a number copied out of a study has no link back to the study.
    #  `measured.load()` is that link: it refuses the file when its rule stamp
    #  disagrees with the constants above and says which one moved.
    m = measured.load()
    base = m.get(measured.BASE_ARM)
    base_dd = m.get(measured.BASE_ARM, measured.F_DD)
    base_worst = m.get(measured.BASE_ARM, measured.F_WORST)
    stop_arm = measured.STOP_ARMS[STOP]
    stops = measured.family(m, measured.STOP_ARMS)
    tp_costs = measured.family_costs(m, measured.TP_ARMS)
    half_cost = -m.cost(measured.HALF_ARM)
    ship_gap = m.cost(measured.SHIPPED_ARM)

    print()
    print(f"  SL {-STOP:>5.0%}   RESTING ORDER from your own fill, live every "
          f"session.")
    print("             MEASURED on the SHIPPED buffer: portfolio drawdown")
    print(f"             {base_dd:.1%} -> {m.get(stop_arm, measured.F_DD):.1%}, "
          f"worst single name {base_worst:.0%} -> "
          f"{m.get(stop_arm, measured.F_WORST):.0%},")
    print(f"             at a {measured.noun(m.cost(stop_arm))} of "
          f"{abs(m.cost(stop_arm)) * 100:.2f} points of CAGR "
          f"({base:.2%} -> {m.get(stop_arm):.2%}).")
    if stops:
        #  Levels are stored positive and quoted negative, so the SMALLEST
        #  level is the tightest stop and belongs first.
        print(f"             The whole family {-min(stops):.0%} to "
              f"{-max(stops):.0%} lands between {min(stops.values()):.2%}")
        print(f"             and {max(stops.values()):.2%}, so "
              f"{-measured.middle(stops):.0%} is the MIDDLE, not the argmax.")
    print()
    print(f"  TP {TP:>+5.0%}   SELL {TP_FRAC:.0%}, let the rest run. RESTING "
          f"ORDER.")
    if tp_costs:
        #  S3's predicted null: the cost RISES as the target TIGHTENS. Levels
        #  ascend here, so that claim is "costs are non-increasing" -- and it
        #  is CHECKED, because a claim about a shape can be.
        shape = ("monotone" if measured.is_monotone(tp_costs.values())
                 else "NOT monotone")
        print(f"             MEASURED: a target's cost is {shape} in how tight")
        print("             it is --")
        for lvl, c in tp_costs.items():
            print(f"                 {lvl:>+5.0%} costs {c * 100:>5.2f} points "
                  f"of CAGR a year")
        print(f"             Selling only {TP_FRAC:.0%} at {TP:+.0%} costs "
              f"{half_cost * 100:.2f} -- the cheapest")
        print("             target on that curve. It measured +0.01 (free) on")
        print("             the PRE-H62 buffer; that is withdrawn.")
    print("             A TIGHTER TARGET IS YOURS TO SET -- the price of each")
    print("             is in the line above.")
    print()
    print("  BAND EXIT  NOT a resting order: checked at the QUARTERLY review")
    print("             only. The threshold is a percentile of the board, so")
    print("             it moves -- a name can be sold without falling a")
    print("             rupiah if the board rallies past it, if its 60-day")
    print(f"             vol rises past the calmest-{KEEP_VOL:.0%} line "
          f"('vol room' is the")
    print("             headroom), or if it leaves the universe.")
    print("             MEASURED: checking this line DAILY is a disaster --")
    print(f"             CAGR {base:.2%} -> {m.get(measured.DAILY_ARM):.2%}, "
          f"because it then sells on the")
    print("             board's noise rather than the name's decline.")
    print()
    #  THE NOUN COMES FROM THE SIGN. The pre-H62 table showed these levels
    #  GAINING 0.76 points; prose that says "edge" where the number says "cost"
    #  is the drift this whole block exists to end.
    print(f"  TOGETHER   {m.get(measured.SHIPPED_ARM):.2%}/yr against "
          f"{base:.2%} with none -- the levels now")
    print(f"             {measured.noun(ship_gap).upper()} "
          f"{abs(ship_gap) * 100:.2f} points -- drawdown "
          f"{m.get(measured.SHIPPED_ARM, measured.F_DD):.1%} against "
          f"{base_dd:.1%},")
    print(f"             worst single name "
          f"{m.get(measured.SHIPPED_ARM, measured.F_WORST):.0%} against "
          f"{base_worst:.0%}, on 6 rebalance")
    if m.index_cagr is not None:
        print(f"             calendars. IHSG total return over the same span: "
              f"{m.index_cagr:.2%}.")
    else:
        print("             calendars. The IHSG over that exact span is not in")
        print("             the result file, so it is not quoted here -- see")
        print("             `python scripts/stoptest.py` for the paired figure.")
    print("             THE SIGN OF THE RETURN EFFECT FLIPPED when H62 moved")
    print("             the buffer: the pre-H62 table showed a 0.76-point")
    print("             GAIN. It was never claimed then and is not claimed")
    print("             now -- worse early, better late, A18's regime noise.")
    print("             The DRAWDOWN is the whole case for these levels.")
    print("             IN-SAMPLE: the holdout was spent at H16.")
    print(f"             SOURCE: {m.provenance}.")
    print()
    print("  THE SEARCH  H58 computed the deflated Sharpe this repo's §11 has")
    print("             demanded since day one, and it is the caveat that")
    print("             belongs on the line above. This family's Sharpe is")
    print("             +0.73 and its PSR against zero is 0.999 -- but charged")
    print("             for having searched 350 times, the DSR is 0.974 on the")
    print("             most flattering dispersion assumption available and")
    print("             0.947 on the very next one, which FAILS. It survives 1")
    print("             of 7 assumptions. A permutation null asks whether the")
    print("             label carries information; it cannot ask whether this")
    print("             is the best of 350 attempts, and that is the question")
    print("             a reader of the CAGR above should be asking.")


if __name__ == "__main__":
    main()
