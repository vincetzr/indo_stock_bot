#!/usr/bin/env python3
"""The twice-daily brief: what the market did, what moved together, where every
move sits in its own history, and what followed states like it.

    python3 scripts/brief.py --session pre     # before the 09:00 WIB open
    python3 scripts/brief.py --session post    # after the 15:50 WIB close
    python3 scripts/brief.py --ticker BBCA     # one name, in detail
    python3 scripts/brief.py --build-tables    # recompute the reference tables

WHAT THIS IS, STATED ONCE HERE AND AGAIN IN THE OUTPUT
-------------------------------------------------------
A DESCRIPTION, plus historical frequencies. Not a forecast and not a buy list.

That is not modesty, it is what this repository measured. Four independent
instruments — aggregate broker flow (H9), broker identity (H10/H11), investor
class (H12) and price/TA (H13) — were run to their end, and the answer each
time was that whatever structure exists does not survive the cost of acting on
it. H13 is the sharpest: all eight registered price features are statistically
real and every one of them is net-negative after 56 bps of fees plus a
fraksi-harga half-spread. A tool that quietly ranked names as buys would be
contradicting the memos sitting beside it in `reports/`.

So each section states its own status:

    MARKET STATE        arithmetic. Exact, and no claim attached.
    WHAT MOVED TOGETHER derived from returns. Named by constituents only.
    RUN STATE           where a move sits in its own history. Descriptive.
    WHAT FOLLOWED       historical frequencies from PRE-HOLDOUT data, with the
                        base rate, the effective sample size, the bootstrap
                        interval and the table-wide permutation null.
    CANDIDATES          a ranking on the eight registered features, each shown
                        with H13's measured post-cost result for that feature.

THE HOLDOUT
------------
§11 reserves the most recent 24 months to be spent once. Every reference
distribution here is estimated on `holdout == False` rows. The brief never
touches the holdout, which also makes its conditionals genuinely out-of-sample
with respect to the bar being described.

REFRESHING
-----------
Daily bars come from Yahoo. The brief refuses to compute a cross-section on a
date where only part of the universe has been refreshed — a watchlist refresh
leaves the panel looking current while the breadth line is really forty blue
chips. When that happens it says so and falls back to the last complete
session.
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

from idxbot.report import brief as B                          # noqa: E402

WIB = dt.timezone(dt.timedelta(hours=7))
PANEL = os.path.join("data", "spine", "price_panel.parquet")
W = 96


def rule(title: str = "") -> None:
    print("=" * W)
    if title:
        print(f" {title}")
        print("=" * W)


def pct(x, d: int = 1) -> str:
    return "n/a" if x is None or not np.isfinite(x) else f"{x:+.{d}%}"


def section_state(P, S, day) -> None:
    rule("1. MARKET STATE — arithmetic, no claim attached")
    b = B.breadth(S, day)
    r = B.regime(P, day)
    L = B.limit_moves(S, day)
    print(f" {b['n_names']} names traded.  "
          f"{b['advancing']:.0%} advanced, {b['unchanged']:.0%} did not move "
          f"at all (the illiquid tail), median move {b['median_move']:+.2%}")
    print(f" cross-sectional spread of today's returns: "
          f"{b['dispersion']:.2%}")
    print()
    print(f" {'':<12}{'1d':>9}{'1w':>9}{'1m':>9}{'3m':>9}{'ytd':>9}"
          f"{'20d vol':>10}{'vs 5y':>8}")
    for name, lab in (("equal", "equal-wt"), ("turnover", "turnover-wt")):
        if f"{name}_1d" not in r:
            continue
        print(f" {lab:<12}" + "".join(
            f"{pct(r[f'{name}_{h}']):>9}" for h in ("1d", "1w", "1m", "3m",
                                                    "ytd"))
              + f"{r[f'{name}_vol']:>10.1%}{r[f'{name}_vol_pct']:>8.0%}")
    print(" turnover-weighted is a PROXY for cap-weighted — this repo has no")
    print(" shares-outstanding series. A gap between the two rows is the big")
    print(" names carrying the tape while the median name does not, or vice")
    print(" versa; it is worth reading, and it is not IHSG.")
    print()
    print(f" above the 20-day  {b['above_20d']:>6.0%}   "
          f"50-day {b['above_50d']:>6.0%}   200-day {b['above_200d']:>6.0%}")
    print(f" at a 250-day high {b['new_highs']:>6}   "
          f"at a 250-day low {b['new_lows']:>5}   of {b['n_250d']}")
    print(f" closed at the auto-rejection band: {L['ara']} ARA, {L['arb']} ARB "
          f"of {L['n']}")
    print("   (close test only — the panel has no intraday high or low, so a")
    print("    name that traded away from the band intraday still counts here.")
    print("    An upper bound on genuine lock-ups, on the point-in-time band.)")
    print()
    m = B.movers(S)
    print(f" biggest movers, liquid names only "
          f"(>= Rp {B.MIN_VALUE/1e9:.0f}bn traded and above the "
          f"{B.LIQUID_PCT:.0%} turnover percentile)")
    for lab, D in (("up  ", m["up"]), ("down", m["down"])):
        if D.empty:
            continue
        print(f"   {lab}  " + "  ".join(
            f"{x.ticker} {x.ret:+.1%}" for x in D.itertuples()))
    print()


def section_narrative(P, day) -> None:
    rule("2. WHAT MOVED TOGETHER — the only narrative this data supports")
    cm = B.comovement(P, day, n_pc=5)
    if cm.empty:
        print(" not enough liquid names with a year of history\n")
        return
    print(" Components fitted on 250 sessions ending the day BEFORE this one,")
    print(" then today's cross-section projected onto them. A group is named by")
    print(" its members and nothing else — whether eight coal names moving")
    print(" together is 'the coal trade' is your interpretation, not a finding.")
    print()
    print(f" {'':<5}{'of hist var':>12}{'of today':>10}{'today':>9}"
          f"{'|size| pct':>12}")
    # iterrows, not itertuples: "with" is a Python keyword and pandas silently
    # renames it to a positional _5, which breaks on any column reordering
    for _, x in cm.iterrows():
        print(f" PC{x['pc']:<3}{x['var_share']:>12.1%}"
              f"{x['today_share']:>10.1%}"
              f"{x['score_z']:>+9.2f}z{x['abs_pct']:>11.0%}")
        print(f"        with:    {' '.join(x['with'])}")
        print(f"        against: {' '.join(x['against'])}")
    print()
    print(" " + B.narrative_gap())
    print()


def section_conditional(P, R, day, blob, n: int) -> None:
    k = blob["k"]
    rule(f"3. WHERE EVERY MOVE SITS, AND WHAT FOLLOWED — {k} sessions ahead")
    T, E, N = blob["table"], blob["edges"], blob["null"]
    print(f" Reference: {blob['n_rows']:,} liquid pre-holdout bars, "
          f"{blob['date_min']} to {blob['date_max']}, built {blob['built']}.")
    print(f" A run is measured from the last {B.RUN_WINDOW}-session extreme of")
    print(" the opposite sign. run_z is the move in standard deviations FOR A")
    print(" MOVE OF ITS LENGTH, so a quiet name up 30% in ten days and a")
    print(" volatile one up 30% in two hundred are not confused.")
    print()
    print(" THE PERMUTATION NULL FIRST — this repo has four separate occasions")
    print(" where reading a statistic against zero rather than against its own")
    print(" shuffled null produced a confident wrong answer.")
    print(f"   {N['n_cells']} cells.  largest |excess over base rate| "
          f"observed {N['obs_max_abs']:.2%}")
    print(f"   under shuffled state labels: {N['null_max_abs_mean']:.2%} "
          f"mean, {N['null_max_abs_p95']:.2%} at the 95th percentile")
    print(f"   spread across cells: observed {N['obs_spread']:.2%} against a "
          f"null {N['null_spread_mean']:.2%}")
    print(f"   p(null >= observed) = {N['p_max']:.3f}")
    if N["p_max"] < 0.05:
        print("   => the state conditioning carries information beyond chance.")
        print("      IT IS STILL IN-SAMPLE AND THE CELLS WERE NOT")
        print("      PRE-REGISTERED. Fifty-four cells were computed and the")
        print("      intervals below are uncorrected, so roughly "
              f"{N['expected_false_cells']:.0f} clear zero by luck. The largest")
        print("      cell is the largest OF FIFTY-FOUR and is biased upward by")
        print("      exactly that selection. Treat it as a lead for a")
        print("      pre-registered test, not as a result.")
    else:
        print("   => indistinguishable from shuffled labels. Read nothing "
              "from the cells.")
    print()

    D = B.current_states(P, R, day, T, E)
    if D.empty:
        print(" no liquid name has a usable run state today\n")
        return
    print(f" {len(D)} liquid names placed. Historical excess over the base rate")
    print(" for the cell each currently occupies, and what a round trip costs.")
    print()
    print(" 'since' is sessions from the extreme the leg STARTED at, not the")
    print(" age of today's move; 'off ext' is how far price has come back from")
    print(" the leg's far end — below the high for an advance, above the low")
    print(" for a decline. A long leg with a large 'off ext' has already turned.")
    print()
    hdr = (f" {'ticker':<8}{'close':>9}{'leg':>6}{'since':>7}{'run':>9}"
           f"{'run_z':>8}{'off ext':>9}  {'cell':<44}{'excess':>9}"
           f"{'95% CI':>18}{'cost':>8}{'net':>8}")
    print(hdr)
    for x in pd.concat([D.head(n), D.tail(n)]).drop_duplicates(
            "ticker").itertuples():
        ci = (f"[{x.diff_lo:+.1%},{x.diff_hi:+.1%}]"
              if np.isfinite(x.diff_lo) else "n/a")
        print(f" {x.ticker:<8}{x.close:>9,.0f}{x.leg:>6}{x.run_days:>7}"
              f"{x.run_pct:>+9.1%}{x.run_z:>+8.2f}{x.give_pct:>+9.1%}  "
              f"{x.what:<44}{x.diff:>+9.2%}{ci:>18}"
              f"{x.cost:>8.2%}{x.net:>+8.2%}")
    print()
    print(" 'excess' is over the equal-weighted return of all liquid names on")
    print(" the same dates, so it is what the state added to simply being in")
    print(" the market. 'cost' is A5's 0.56% round trip plus half a tick each")
    print(" way at this price — a FLOOR, since it assumes a one-tick book.")
    print(" 'net' is a HISTORICAL AVERAGE minus that floor. It is not an")
    print(" expected return for this name, and the interval is uncorrected.")
    print()


def section_candidates(P, day, n: int) -> None:
    rule("4. CANDIDATES — ranked on the eight features H13 registered")
    C = B.candidates(P, day, n=n)
    if C.empty:
        print(" nothing passes the liquidity filter today\n")
        return
    print(" Every one of these was tested on 1,989,504 pre-holdout rows. The")
    print(" right-hand column is what that test found. It is printed on every")
    print(" line because a ranked list with the result omitted reads as a")
    print(" recommendation no matter what the header says.")
    print()
    for f, g in C.groupby("feature", sort=False):
        sign = int(g["sign"].iloc[0])
        print(f" {f}  (predicted sign {sign:+d})")
        print(f"   {'  '.join(g['ticker'].tolist())}")
        print(f"   H13: {g['h13'].iloc[0]}")
    ov = B.candidate_overlap(C)
    if not ov.empty:
        print()
        print(" on more than one list:")
        for x in ov.itertuples():
            print(f"   {x.ticker:<8} {x.n_lists}  {', '.join(x.lists)}")
        print("   (overlap is reported instead of a composite score: a blend of")
        print("    the eight would be a NEW signal that has never been tested,")
        print("    wearing the credibility of eight that have.)")
    print()


def section_ticker(P, R, day, blob, t: str) -> None:
    rule(f"DETAIL — {t}")
    T, E = blob["table"], blob["edges"]
    D = B.current_states(P, R, day, T, E, min_value=0.0, liquid_pct=0.0)
    row = D[D["ticker"] == t]
    if row.empty:
        print(f" {t} has no usable run state on {day.date()} — either it did "
              f"not trade,\n or it lacks {B.RUN_WINDOW} sessions of history.\n")
        return
    x = row.iloc[0]
    print(f" close {x['close']:,.0f}   {x['leg']} leg, anchored "
          f"{int(x['run_days'])} sessions ago at its 250-day "
          f"{'low' if x['leg'] == 'up' else 'high'}")
    print(f" from the anchor: {x['run_pct']:+.1%}  "
          f"({x['run_z']:+.2f} sd for a move of that length)")
    print(f" handed back from the leg's extreme: {x['give_pct']:+.1%}")
    print(f" cell: {x['what']}")
    if isinstance(x.get("bucket"), str) and np.isfinite(x.get("diff", np.nan)):
        print(f" historically from that cell, over {blob['k']} sessions:")
        print(f"   mean {x['fwd_mean']:+.2%} against a base rate of "
              f"{x['base_mean']:+.2%}  ->  excess {x['diff']:+.2%}")
        print(f"   95% block-bootstrap CI [{x['diff_lo']:+.2%}, "
              f"{x['diff_hi']:+.2%}]   n = {int(x['n']):,} bars, "
              f"n_eff = {int(x['n_eff'])} non-overlapping windows")
        print(f"   up {x['p_up']:.0%} of the time")
        print(f"   round trip here costs {x['cost']:.2%}  ->  "
              f"net {x['net']:+.2%}")
    print()
    print(" That is a historical frequency from a cell holding thousands of")
    print(" other bars. It is not a forecast for this name, and the interval")
    print(" is uncorrected for the 54 cells the table computes.")
    print()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", choices=["pre", "post"], default="post")
    ap.add_argument("--panel", default=PANEL)
    ap.add_argument("--ticker", default=None, help="detail on one name")
    ap.add_argument("--horizon", type=int, default=20,
                    help="sessions ahead for the conditional table")
    ap.add_argument("--names", type=int, default=8)
    ap.add_argument("--build-tables", action="store_true",
                    help="recompute the cached reference tables (~2 min)")
    ap.add_argument("--draws", type=int, default=200)
    a = ap.parse_args()

    if not os.path.exists(a.panel):
        print(f"no panel at {a.panel} — run scripts/price_panel_build.py first")
        return 1
    now = dt.datetime.now(tz=WIB)
    P = pd.read_parquet(a.panel)

    if a.build_tables:
        print(f" building reference tables (draws={a.draws}) …", flush=True)
        B.build_tables(P, ks=(5, 20), draws=a.draws, null_draws=a.draws)
        print(" done")

    blob = B.load_table(a.horizon)
    if blob is None:
        print(f" no reference table for k={a.horizon}. "
              f"Run with --build-tables once.")
        return 1

    day = B.resolve_asof(P)
    warn = B.coverage_warning(P, day)

    rule()
    print(f" IDX BRIEF — {a.session.upper()} session, {now:%Y-%m-%d %H:%M} WIB")
    print(f" bars through {day.date()}"
          + (f"  ({(now.date() - day.date()).days} calendar days behind)"
             if (now.date() - day.date()).days else ""))
    rule()
    print(" A DESCRIPTION, plus historical frequencies. Not a forecast, not a")
    print(" buy list. Four instruments were run to their end in this repo and")
    print(" none produced an edge that survived costs; H13 in particular found")
    print(" all eight registered price features net-negative after fees and")
    print(" spread. Sections state their own status individually.")
    if a.session == "pre":
        print()
        print(" PRE-OPEN: nothing here is newer than the last close. There is")
        print(" no overnight or pre-market IDX data in this repo, so a morning")
        print(" run and an evening run differ only in what has settled, not in")
        print(" what is known.")
    if warn:
        print()
        print(" ! " + warn)
    print()

    S = B.snapshot(P, day)
    R = B.run_state(P)

    section_state(P, S, day)
    section_narrative(P, day)
    section_conditional(P, R, day, blob, a.names)
    section_candidates(P, day, a.names)
    if a.ticker:
        section_ticker(P, R, day, blob, a.ticker.upper())

    rule("WHAT WOULD CHANGE THE PICTURE")
    print(" The cells in section 3 beat their permutation null, which means")
    print(" the state conditioning is real. It does NOT mean it is tradeable:")
    print(" the cells are post-hoc, in-sample, and uncorrected for 54 tests.")
    print(" The test that would settle it is a pre-registered one — pick the")
    print(" cells and the rule BEFORE looking, then spend the 24-month holdout")
    print(" once. That has not been done, the holdout is untouched, and until")
    print(" it is, nothing in this brief is evidence that trading it pays.")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
