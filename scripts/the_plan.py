#!/usr/bin/env python3
"""The plan, sized in rupiah and in whole lots, with what it cannot promise.

THE BRIEF, AND THE ONE WORD IN IT THAT HAS TO GO
------------------------------------------------
"Dominate IDX with client money so profit is close to guarantee." Everything in
this repo can be delivered against that except the last three words, and it is
worth being exact about why rather than waving at "markets are risky":

    - 43% of IDX names lost money over eleven years, dividends included, in a
      universe that contains no delistings at all.
    - 37% of five-year holding periods in that same universe were negative and
      83% came in under a bank deposit.
    - Every configuration of the best portfolio found here lost between half
      and two thirds of its value at some point.

A guarantee is not on the menu. What IS on the menu is a set of things that are
arithmetic rather than forecast, and one premium that has been measured about as
carefully as this data allows.

WHAT IS CERTAIN, AND THEREFORE WHERE THE WORK GOES FIRST
--------------------------------------------------------
    COST          0.56% a round trip, paid whether or not the trade works. The
                  only line in this whole project guaranteed to be exactly what
                  it says. Trading less is the one return improvement that
                  cannot fail.
    DRAG          A one-name book gives up 0.96% a month to variance drag and a
                  hundred-name book 0.20%. Randomly drawn books, so this is not
                  a selection effect - it is arithmetic. Breadth turns -10.5%/yr
                  into +4.3%/yr without forecasting anything.
    INCOME        87% of liquid IDX names pay a dividend. It arrives whether or
                  not the price cooperates, and every study in this repo before
                  this session simply left it out.
    TAX           Dividends reinvested in Indonesia are exempt under PP 9/2021;
                  otherwise 10% final for a domestic individual. That exemption
                  is worth more than most signals.

WHAT IS MEASURED BUT NOT PROMISED
---------------------------------
The yield premium: +4.90%/yr median across 36 grid cells, all 36 positive,
positive in both halves of the sample, ahead in 8 of 11 calendar years. Real as
far as this data can show. Not a guarantee, and sized below its measurement
below for that reason.

WHAT IS MEASURED NOT TO WORK, AND IS THEREFORE ABSENT
-----------------------------------------------------
Timing at any timeframe (110, 111, 113, 116). Fitting the band (115) or picking
the best factor (this session - the first-half leader placed 11th of 13 out of
sample). Concentration. The other twelve factors. None of it appears in the plan
because it was tested and it lost.

    python3 scripts/the_plan.py --capital 1000000000
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))
sys.path.insert(0, os.path.dirname(__file__))

from idxbot.config import load_config           # noqa: E402
from idxbot.data.cache import Cache             # noqa: E402
from idxbot.data.ohlcv import YahooOHLCV        # noqa: E402
from factor_study import (Board, build_panel, cagr, load_index,   # noqa: E402
                          max_dd, rebalance_positions, run_portfolio, select)
from base_rates import DEPOSIT_RATE, INFLATION, real_return   # noqa: E402
from yield_book import DIV_TAX, net_of_tax, yield_scores      # noqa: E402

LOT = 100                    # shares in one IDX lot
FEE_BUY = 0.0028
FEE_SELL = 0.0028            # 0.18% commission + 0.1% sales tax
HAIRCUT = 0.5                # the share of the measured premium actually planned on


def lot_plan(prices: pd.Series, capital: float,
             names: Sequence[str]) -> pd.DataFrame:
    """Turn an equal-weight target into whole lots, and report the error.

    IDX trades in lots of 100 shares, so an equal-weight book is only equal to
    the nearest lot. On a small account that rounding is not a rounding - a
    Rp 10,000 name costs Rp 1,000,000 a lot, and thirty of those cannot be held
    equally with fifty million rupiah however the weights are written down.
    """
    target = capital / max(len(names), 1)
    rows = []
    for t in names:
        px = float(prices.get(t, np.nan))
        if not np.isfinite(px) or px <= 0:
            continue
        lots = int(np.floor(target / (px * LOT)))
        value = lots * px * LOT
        rows.append({"ticker": t, "price": px, "lots": lots, "value": value,
                     "target": target, "drift": value / target - 1.0
                     if target > 0 else np.nan})
    df = pd.DataFrame(rows)
    return df.sort_values("value", ascending=False) if not df.empty else df


def minimum_capital(prices: pd.Series, names: Sequence[str],
                    tolerance: float = 0.10) -> float:
    """Smallest book on which whole lots still land within tolerance of equal.

    Affording one lot of everything is NOT the bar, and setting it there was the
    first version of this function. Buying k lots when the target was worth
    k + something leaves the position short by up to one whole lot, so the worst
    weight error is about 1/(k+1) - which means holding one lot of the dearest
    name is a 50% error, not a 0% one.

    Turning that around: the target must buy at least 1/tolerance - 1 lots of
    the dearest name, so at 10% the dearest position needs nine lots' worth of
    room and the book needs that many times the number of names. Below this
    number the account is not running the strategy that was tested, it is
    running a lumpier cousin, and the difference is not small.
    """
    px = np.array([float(prices.get(t, np.nan)) for t in names])
    px = px[np.isfinite(px) & (px > 0)]
    if not len(px) or not 0 < tolerance < 1:
        return np.nan
    dearest_lot = float(px.max()) * LOT
    return dearest_lot * (1.0 / tolerance - 1.0) * len(px)


def income_forecast(book: pd.DataFrame, yields: Dict[str, float],
                    tax: float = DIV_TAX) -> Tuple[float, float]:
    """Rupiah of dividend the current book would throw off in a year.

    Trailing, so it is what was paid rather than what will be. Reported as a
    band, not a point, for that reason.
    """
    gross = float(sum(r.value * yields.get(r.ticker, 0.0)
                      for r in book.itertuples()))
    return gross, gross * (1.0 - tax)


def plan_return(measured_edge: float, neutral: float,
                haircut: float = HAIRCUT) -> float:
    """What to actually plan on, which is less than what was measured.

    An estimated premium is the true premium plus whatever the estimate got
    wrong, and the error is not symmetric here: the universe holds no
    delistings, thirteen factors were looked at before this one was chosen, and
    published factor premia routinely halve out of sample. Half is a convention,
    not a calculation, and it is stated as one.
    """
    return neutral + haircut * measured_edge


def drawdown_budget(capital: float, worst: float) -> Dict[str, float]:
    """What the worst case in the sample looks like in rupiah, before it happens.

    A percentage is easy to agree to in a meeting. The same number in rupiah,
    written down before the money is committed, is what the client is actually
    being asked to sign for.
    """
    return {"peak": capital, "trough": capital * (1.0 + worst),
            "lost": capital * abs(worst)}


def book_status(loader: YahooOHLCV, cache_dir: str, names: int = 30,
                min_turnover: float = 5e9, start: str = "2015-01-01") -> Dict:
    """The plan's live state: what it holds, what drifted, what is due.

    Split out so the twice-daily job reports the thing that has evidence behind
    it, not only the band colours. The band report is a position report by its
    own admission (Result 100, 110); this is the part with a measured premium
    attached, and it belongs in the same run.
    """
    close, raw, turn = build_panel(loader, cache_dir, start, total=True)
    idx_s = load_index(loader, "^JKSE", close.index)
    idx = idx_s.to_numpy(float) if len(idx_s) else np.full(len(close), np.nan)
    rebal = rebalance_positions(close.index, "annual", 280)
    board = Board(close, turn, idx, rebal, [], min_turnover, 280, raw=raw)
    board.scores["divyield"] = yield_scores(board, 250)

    k = len(board.rebal) - 1
    cur = list(close.columns.to_numpy()[
        select(board.cols[k], board.scores["divyield"][k], names)])
    ys = {t: float(board.scores["divyield"][k][p]) for t, p in
          zip(cur, select(board.cols[k], board.scores["divyield"][k], names))}
    held, set_on = cur, close.index[board.rebal[k]]
    if k >= 1:
        held = list(close.columns.to_numpy()[
            select(board.cols[k - 1], board.scores["divyield"][k - 1], names)])
        set_on = close.index[board.rebal[k - 1]]
    when, days = next_rebalance(close.index[-1])
    avg = float(np.mean(list(ys.values()))) if ys else np.nan
    return {"asof": close.index[-1], "held": held, "set_on": set_on,
            "current": cur, "yields": ys, "avg_yield": avg,
            "eligible": int(len(board.cols[k])), "next": when, "days": days,
            "drift": book_drift(held, cur),
            "kills": kill_check(avg, int(len(board.cols[k])), names),
            "prices": raw.iloc[board.rebal[k]]}


def print_book_status(st: Dict, names: int = 30) -> None:
    """One compact block, for the twice-daily run."""
    d = st["drift"]
    print(f" the plan holds the top {names} by trailing dividend yield, set on "
          f"{st['set_on']:%Y-%m-%d}.")
    print(f" {len(d['unchanged'])} of {len(st['held'])} are still in the top "
          f"{names} as of {st['asof']:%Y-%m-%d}; average trailing yield "
          f"{st['avg_yield']:.1%}")
    if d["dropped_out"]:
        print(f"   drifted out : {', '.join(d['dropped_out'])}")
    if d["moved_in"]:
        print(f"   drifted in  : {', '.join(d['moved_in'])}")
    print(f" NEXT REBALANCE {st['next']:%Y-%m-%d}, {st['days']} days away. "
          f"Nothing above is an instruction —\n the annual discipline is what "
          f"makes this cost 0.19%/yr instead of 0.73%.")
    if st["kills"]:
        for w in st["kills"]:
            print(f" ! ABANDON CONDITION LIVE: {w}")
    else:
        print(f" abandon conditions: none live "
              f"({st['eligible']} names clear the liquidity floor).")


def next_rebalance(today: pd.Timestamp) -> Tuple[pd.Timestamp, int]:
    """The next annual rebalance date, and how many days away it is.

    The tested rule rebalances on the last business day of December and executes
    the next session, so that is the date reported - not a rolling "twelve
    months from whenever you started". A rule whose date drifts with the start
    date is a different rule from the one that was measured.

    Weekends are handled by rolling BACK into December, never forward into
    January - rolling forward would put the rebalance in the wrong year and
    quietly skip one. The IDX holiday calendar is not modelled and the exchange
    is usually shut for part of the year end, so read the date as "the last
    session at or before this", which is what the backtest did.
    """
    def last_weekday_of_december(year: int) -> pd.Timestamp:
        d = pd.Timestamp(f"{year}-12-31")
        while d.weekday() >= 5:
            d -= pd.Timedelta(days=1)
        return d

    today = pd.Timestamp(today).normalize()
    date = last_weekday_of_december(today.year)
    if date < today:
        date = last_weekday_of_december(today.year + 1)
    return date, int((date - today).days)


def book_drift(held: Sequence[str], current: Sequence[str]) -> Dict[str, List[str]]:
    """What the book would change to if it rebalanced today.

    Reported as INFORMATION, never as an instruction. The whole reason annual
    rebalancing beats monthly here is that it does not act on this; a report
    that showed the drift next to a "sell" column would quietly turn a 0.19%/yr
    strategy into a 0.73%/yr one.
    """
    h, c = list(dict.fromkeys(held)), list(dict.fromkeys(current))
    hs, cs = set(h), set(c)
    return {"dropped_out": [t for t in h if t not in cs],
            "moved_in": [t for t in c if t not in hs],
            "unchanged": [t for t in h if t in cs]}


def kill_check(avg_yield: float, eligible: int, names: int,
               deposit: float = DEPOSIT_RATE) -> List[str]:
    """Which of the pre-committed abandon conditions are live right now.

    The three-consecutive-years condition needs the annual review and is not
    checkable here; the other two are, and are checked every run so nobody has
    to remember them.
    """
    out = []
    if np.isfinite(avg_yield) and avg_yield < deposit:
        out.append(f"the book's average trailing yield {avg_yield:.1%} is "
                   f"below the {deposit:.1%} deposit rate — there is no income "
                   f"premium left to harvest")
    if eligible < names:
        out.append(f"only {eligible} names clear the liquidity floor, fewer "
                   f"than the {names} the book holds")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--capital", type=float, default=1e9,
                    help="rupiah to deploy")
    ap.add_argument("--names", type=int, default=30)
    ap.add_argument("--cash", type=float, default=0.15,
                    help="share held in deposit rather than equities")
    ap.add_argument("--min-turnover", type=float, default=5e9)
    ap.add_argument("--start", default="2015-01-01")
    args = ap.parse_args()

    cfg = load_config()
    cache_dir = cfg.path("data.cache_dir", "data/cache")
    loader = YahooOHLCV(cfg, Cache(cache_dir))

    print(f"{'=' * 96}\n THE PLAN — what the evidence supports, sized in "
          f"rupiah\n{'=' * 96}")
    close, raw, turn = build_panel(loader, cache_dir, args.start, total=True)
    idx_s = load_index(loader, "^JKSE", close.index)
    idx = idx_s.to_numpy(float) if len(idx_s) else np.full(len(close), np.nan)
    rebal = rebalance_positions(close.index, "annual", 280)
    board = Board(close, turn, idx, rebal, [], args.min_turnover, 280, raw=raw)
    board.scores["divyield"] = yield_scores(board, 250)

    neutral = run_portfolio(board, None, 0)
    book = run_portfolio(board, "divyield", args.names)
    b = idx_s.reindex(neutral.curve.index).dropna()
    bench = b / float(b.iloc[0]) if len(b) else pd.Series(dtype=float)

    measured = cagr(book.curve)
    edge = measured - cagr(neutral.curve)
    worst = max_dd(book.daily)

    # ---- 1. what is being bought ----------------------------------------
    print(f"\n{'=' * 96}\n 1. THE RULE, IN FULL, SO IT CAN BE FOLLOWED WITHOUT "
          f"ME\n{'=' * 96}")
    print(f"   universe    IDX equities with median daily turnover >= "
          f"{args.min_turnover:,.0f} IDR over the")
    print(f"               trailing year and at least 250 sessions of history")
    print(f"   rank        by dividends paid over the trailing 12 months, "
          f"divided by price")
    print(f"   hold        the top {args.names}, equal weight")
    print(f"   rebalance   ONCE A YEAR. Quarterly and monthly also work; annual "
          f"costs 0.19%/yr")
    print(f"               against 0.73% monthly, and the grid showed no return "
          f"given up for it")
    print(f"   never       no timing overlay, no leverage, no shorting, no "
          f"concentration.")
    print(f"               Each of those was tested here and each of them lost.")

    # ---- 2. what to plan on ---------------------------------------------
    print(f"\n{'=' * 96}\n 2. WHAT TO PLAN ON — less than what was measured\n"
          f"{'=' * 96}")
    planned = plan_return(edge, cagr(neutral.curve))
    print(f" {'measured in the sample':<40}{measured:>9.2%}/yr")
    print(f" {'  of which the neutral book':<40}"
          f"{cagr(neutral.curve):>9.2%}/yr")
    print(f" {'  of which the yield premium':<40}{edge:>+9.2%}/yr")
    print(f" {'planned on (half the premium)':<40}{planned:>9.2%}/yr")
    print(f"\n the same number under each tax position, because the gap "
          f"between them is\n larger than most of the signals in this repo:")
    keeps = [("domestic, dividends reinvested in Indonesia", 0.0),
             ("domestic, dividends taken as cash", DIV_TAX),
             ("foreign holder", 0.20)]
    for label, rate in keeps:
        net = net_of_tax(planned, 0.0679, rate)
        print(f"   {label:<44}{net:>8.2%}/yr   real "
              f"{real_return(net, INFLATION):>6.2%}   vs cash "
              f"{net - DEPOSIT_RATE:>+6.2%}")
    print(f" {'a rupiah deposit, for comparison':<40}{DEPOSIT_RATE:>9.2%}/yr"
          f"   at no drawdown")
    best = net_of_tax(planned, 0.0679, 0.0)
    if best < DEPOSIT_RATE:
        print("\n ! Even at the exempt rate the deposit wins on these numbers. "
              "Recommend the deposit.")
    else:
        print(f"\n Read the top row, because it is the one that applies to an "
              f"Indonesian\n individual reinvesting at home: the book is "
              f"planned to beat cash by\n {best - DEPOSIT_RATE:+.2%}/yr. That "
              f"margin is what the drawdown below is being accepted\n FOR. If "
              f"the client would not accept a halving for it, the deposit is "
              f"the\n better answer and saying so is the job.")

    # ---- 3. the drawdown, in money --------------------------------------
    print(f"\n{'=' * 96}\n 3. WHAT IT COSTS TO EARN THAT — signed for in "
          f"advance\n{'=' * 96}")
    eq_capital = args.capital * (1.0 - args.cash)
    d = drawdown_budget(eq_capital, worst)
    print(f" worst drawdown of this exact book in the sample: {worst:.1%}")
    print(f" on {eq_capital:,.0f} IDR of equity that is a fall to "
          f"{d['trough']:,.0f}, a loss of {d['lost']:,.0f} IDR")
    print(f" 37% of five-year holding periods in this universe were negative, "
          f"so money that\n might be needed inside five years does not belong "
          f"in this book at all.")
    print(f" the {args.cash:.0%} cash sleeve ({args.capital * args.cash:,.0f} "
          f"IDR at {DEPOSIT_RATE:.1%}) is not a view on the market.\n It is "
          f"what lets the annual rebalance happen without selling into a fall.")

    # ---- 4. the book today ----------------------------------------------
    print(f"\n{'=' * 96}\n 4. THE BOOK TODAY, IN WHOLE LOTS\n{'=' * 96}")
    k = len(board.rebal) - 1
    picks = select(board.cols[k], board.scores["divyield"][k], args.names)
    names = list(close.columns.to_numpy()[picks])
    ys = {n: float(board.scores["divyield"][k][p])
          for n, p in zip(names, picks)}
    last = raw.iloc[board.rebal[k]]
    plan = lot_plan(last, eq_capital, names)
    if plan.empty:
        print(" no book could be formed.")
        return 1
    print(f" {'ticker':<8}{'price':>9}{'yield':>8}{'lots':>7}{'value IDR':>16}"
          f"{'weight':>9}{'off target':>12}")
    for r in plan.itertuples():
        print(f" {r.ticker:<8}{r.price:>9,.0f}{ys.get(r.ticker, 0):>8.1%}"
              f"{r.lots:>7}{r.value:>16,.0f}"
              f"{r.value / eq_capital:>9.2%}{r.drift:>+12.1%}")
    invested = float(plan["value"].sum())
    print(f" {'':<8}{'':>9}{'':>8}{'':>7}{invested:>16,.0f}"
          f"{invested / eq_capital:>9.2%}   invested")
    print(f" {'':<8}{'':>9}{'':>8}{'':>7}{eq_capital - invested:>16,.0f}"
          f"{1 - invested / eq_capital:>9.2%}   left as odd cash")
    entry_fee = invested * FEE_BUY
    print(f"\n entry commission at {FEE_BUY:.2%}: {entry_fee:,.0f} IDR")
    gross, net = income_forecast(plan, ys)
    print(f" trailing income of this book: {gross:,.0f} IDR/yr gross, "
          f"{net:,.0f} after 10% tax")
    print(f" ({gross / invested:.2%} of the book — trailing, so it is what was "
          f"paid, not a forecast)")
    print(f" reinvest the dividends in Indonesia and PP 9/2021 exempts them "
          f"entirely:\n that exemption alone is worth "
          f"{gross * DIV_TAX:,.0f} IDR a year here.")

    mincap = minimum_capital(last, names)
    print(f"\n smallest capital that still holds {len(names)} names within 10% "
          f"of equal weight:\n {mincap:,.0f} IDR. Below that, hold fewer names "
          f"rather than uneven ones —\n the breadth table says 20 names costs "
          f"little and 5 costs a great deal.")
    if eq_capital < mincap:
        print(f" ! this book at {eq_capital:,.0f} IDR is below that line; the "
              f"'off target' column above\n ! shows what the rounding is "
              f"actually doing to the weights.")

    # ---- 4b. drift since the last rebalance ------------------------------
    prior = len(board.rebal) - 2
    if prior >= 0:
        held = list(close.columns.to_numpy()[
            select(board.cols[prior], board.scores["divyield"][prior],
                   args.names)])
        drift = book_drift(held, names)
        when, days = next_rebalance(close.index[-1])
        print(f"\n{'=' * 96}\n 4b. DRIFT SINCE THE LAST REBALANCE — "
              f"information, not an instruction\n{'=' * 96}")
        print(f" the book set on {close.index[board.rebal[prior]]:%Y-%m-%d} "
              f"still holds {len(drift['unchanged'])} of {len(held)} names.")
        if drift["dropped_out"]:
            print(f" out of the top {args.names} today: "
                  f"{', '.join(drift['dropped_out'])}")
        if drift["moved_in"]:
            print(f" into it today:                "
                  f"{', '.join(drift['moved_in'])}")
        print(f" NEXT REBALANCE {when:%Y-%m-%d}, {days} days away. Do nothing "
              f"until then.")
        print(f" acting on this list monthly instead would cost 0.73%/yr "
              f"against 0.19%, and the\n grid showed no return bought with "
              f"that difference.")

    # ---- 5. when to stop -------------------------------------------------
    print(f"\n{'=' * 96}\n 5. WHEN TO ABANDON IT — decided now, not later\n"
          f"{'=' * 96}")
    live = kill_check(float(np.mean(list(ys.values()))), len(board.cols[k]),
                      args.names)
    if live:
        for w in live:
            print(f" ! LIVE NOW: {w}")
    else:
        print(" none of the checkable conditions is live today.")
    print(" A rule with no exit is a rule that gets rationalised. These are "
          "committed to\n before any money moves, and each one is checkable "
          "from the annual review:")
    print(f"   1. The book underperforms the equal-weight liquid universe for "
          f"THREE consecutive\n      calendar years. In the sample it was "
          f"behind in 3 of 11 and never twice running.")
    print(f"   2. The average trailing yield of the top {args.names} falls "
          f"below the deposit rate.\n      There would then be no income "
          f"premium left to harvest. It is "
          f"{np.mean(list(ys.values())):.1%} today.")
    print(f"   3. The liquidity floor stops clearing {args.names} names, which "
          f"would mean the book\n      can no longer be built at the size it "
          f"is being run.")
    print("   Nothing else. Not a bad quarter, not a drawdown, not a headline.")

    # ---- 6. what could still change the plan ----------------------------
    print(f"\n{'=' * 96}\n 6. THE PART THAT IS STILL OPEN\n{'=' * 96}")
    print(" Layer 2 - broker flow - has never been tested with enough data to "
          "detect an\n effect worth having, which is different from having "
          "been tested and failed.\n The collection runs daily and the "
          "hypotheses are frozen and hashed so they\n cannot be edited once "
          "the answer is visible. When 250 complete days exist on a\n name, "
          "the protocol runs and this plan gets revisited on the result.")
    print(" Until then the plan does not assume anything about it, which is "
          "the whole\n reason it was frozen in advance.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
