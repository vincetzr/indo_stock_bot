#!/usr/bin/env python3
"""Rp X into the card from a given date: the picks, every trade, the end value.

    python3 scripts/card_account.py --capital 50000000 --start 2026-01-01

WHAT THIS IS
    The shipped quarterly card (scripts/rules.py) run as one real account
    instead of a backtest statistic. The account holds whole 100-share lots and
    pays the user's Mandiri fees (A5: 0.28% to buy, 0.18% plus 0.10% tax to
    sell) plus a quarter tick of spread per side. Every decision uses only data
    available at the close it is made on:

    * REVIEW. On the last session before `--start`, and every 63 sessions
      after it, run the card's selection on that day's eligible board: keep a
      held name while it is in the top 30% on hi52 and the calmest 60% on
      vol60; fill empty slots from the top 10% / calmest half, best hi52
      first; hold at most 10. Orders fill at the NEXT session's close, because
      you see the card in the evening and trade the next day (the backtests
      fill at the same close; `--lag 0` reproduces that). Each review
      re-equalises the book, as the measured rule does. A held name that has
      stopped printing is realised at its last value, as stoptest.walk does.
    * STOP. Sold at the close on which the position's TOTAL return since entry
      (dividends included, on the adjusted close) reaches -20%. That is the
      trigger stoptest.walk measures, not the order the card tells you to
      leave at your broker: a resting order on the raw price, which can fire
      intraday and earlier. An independent check measured the difference on
      the 2026 path at about +0.2 jt (raw close) to +1.6 jt (intraday).
    * TAKE-PROFIT. At +100% from entry, half the position is sold, once.
    * CASH from stops and take-profits waits for the next review and earns
      nothing. A name stopped out is not bought back before the next decision,
      even if the review it was picked at fills on the day it stopped.
    * LOCKED OR STALE DAYS. Nothing fills on a bar the panel marks
      untradeable (auto-rejection lock or no trades). The order waits for the
      next tradeable session; selling into an ARB lock is the fiction CLAUDE.md
      section 5 names. That guard is only as good as the auto-rejection
      schedule behind the flag, which missed the 7% limit of 2026-10-19 until
      an independent check of this file found it (A54).
    * TIES. When the 10th slot falls among names tied on hi52 (several names
      sitting exactly at their 52-week high), rules.py's Sticky breaks the tie
      by ticker order. That is A15's under-determined top-N; the log says when
      it happens, and `--tie-seed` breaks ties at random instead so the spread
      it hides can be measured.

    Positions are valued on the adjusted close, so a dividend is reinvested in
    the same name, net of the 10% dividend tax (`--div-tax`). On the 2026 path
    the dividends came to about 5.6% of capital, so the tax is not negligible.

    CONTROLS, run through the identical machine with `--selector`:
    `random_fresh` (ten random eligible names at every review),
    `random_sticky` (keep held names that are still eligible, fill at random)
    and `hold_first` (buy the first basket and never trade again: no stops, no
    reviews). The equal-weighted eligible universe is reported beside every
    run as the comparison between the card and the cap-weighted IHSG (A19).

WHAT IT IS NOT
    * Not a test of the rule. The card was chosen in August-September 2026
      (H54, H56, H62) on data that already contained most of 2026, so a 2026
      start replays a period the rule was fitted on. The only forward
      evidence is the forward record from 2026-09-04, and this replay is NOT
      it: its review calendar differs, so its September basket shares only 3
      of 10 names with the recorded one.
    * One path. The result moves by more than 20 points with the start date
      alone, and by several more with the tie-break.
    * Survivor-only panel, no market impact, close-based fills, and some
      closes are vendor back-adjusted (A2), so a printed fill price can differ
      from the price that traded by a tick or two.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))
sys.path.insert(0, os.path.dirname(__file__))

import stoptest as st                                              # noqa: E402
from bhbench import IDX_YIELD, INDEX, MIN_BASKET, load             # noqa: E402
from paint_suite import tick_of                                    # noqa: E402
from rules import K, STOP, TP, TP_FRAC                             # noqa: E402

LOT = 100
BUY_FEE = 0.0028                  # A5: Mandiri buy commission
SELL_FEE = 0.0018 + 0.0010        # A5: sell commission + 0.1% final tax
DIV_TAX = 0.10                    # final tax on dividends to individuals
STICKY_MIN_ROWS = 3 * K           # beathold.Sticky refuses below this
SELECTORS = ("card", "random_fresh", "random_sticky", "hold_first")


def side_cost(price: float, fee: float, spread_ticks: float) -> float:
    """One side's cost as a fraction of price: the fee plus half the assumed
    round-trip spread. The card's own cost model is half a tick per round trip
    (spread_ticks=0.5); OJK measures the market's average quoted spread at
    about 1.6 ticks (A51)."""
    return fee + 0.5 * spread_ticks * tick_of(price) / price


def lots_affordable(cash: float, price: float, fee: float) -> int:
    return int(np.floor(cash / (LOT * price * (1.0 + fee))))


class Account:
    """Cash plus positions valued on the adjusted close."""

    def __init__(self, capital: float, spread_ticks: float,
                 div_tax: float = DIV_TAX):
        self.cash = float(capital)
        self.pos: Dict[str, Dict] = {}
        self.spread = spread_ticks
        self.div_tax = div_tax
        self.dividends = 0.0
        self.log: List[Dict] = []

    def value(self) -> float:
        return self.cash + sum(p["val"] for p in self.pos.values())

    def note(self, d, kind, t, lots, px, rupiah, why=""):
        self.log.append({"date": str(pd.Timestamp(d).date()), "action": kind,
                         "ticker": t, "lots": int(lots), "price": float(px),
                         "rupiah": float(rupiah), "why": why})

    def buy(self, d, t, lots, close, adj):
        if lots <= 0:
            return
        c = side_cost(close, BUY_FEE, self.spread)
        gross = lots * LOT * close
        self.cash -= gross * (1.0 + c)
        p = self.pos.setdefault(t, {"val": 0.0, "lots": 0, "entry_adj": adj,
                                    "entry_px": close, "tp_done": False,
                                    "adj": adj, "close": close,
                                    "factor": adj / close})
        p["val"] += gross
        p["lots"] += int(lots)
        self.note(d, "BUY", t, lots, close, gross * (1.0 + c))

    def sell_value(self, d, t, val, close, why, lots=None):
        """Sell `val` rupiah of a position at this close; `lots` is what the
        log reports, defaulting to the share of the lots actually held."""
        p = self.pos[t]
        val = min(val, p["val"])
        share = val / p["val"] if p["val"] > 0 else 1.0
        if lots is None:
            lots = (p["lots"] if share > 0.999
                    else int(round(p["lots"] * share)))
        c = side_cost(close, SELL_FEE, self.spread)
        self.cash += val * (1.0 - c)
        p["val"] -= val
        p["lots"] = max(p["lots"] - int(lots), 0)
        self.note(d, "SELL", t, lots, close, val * (1.0 - c), why)
        if p["val"] <= 1.0:
            del self.pos[t]

    def mark(self, adj_today: Dict[str, float], close_today: Dict[str, float]):
        """Carry every position to today's adjusted close.

        A dividend shows up as a step in adj/close (Yahoo's close is already
        split-adjusted, so only distributions move that ratio). The step's
        size is the dividend yield on the ex-date, and the tax is withheld from
        the reinvested amount.
        """
        for t, p in self.pos.items():
            a, c = adj_today.get(t), close_today.get(t)
            if a is None or not np.isfinite(a) or a <= 0:
                continue
            grown = p["val"] * a / p["adj"]
            if c is not None and np.isfinite(c) and c > 0:
                f = a / c
                step = f / p["factor"] - 1.0
                if step > 1e-6:
                    paid = p["val"] * (c / p["close"]) * step
                    self.dividends += paid
                    grown -= self.div_tax * paid
                p["factor"], p["close"] = f, c
            p["val"], p["adj"] = grown, a


def select(day: pd.DataFrame, held: List[str], how: str = "card",
           rng: Optional[np.random.Generator] = None,
           tie_rng: Optional[np.random.Generator] = None):
    """One review's basket, and a note when its last slot fell in a tie.

    "card" is beathold.Sticky on stoptest.prep's in_entry/in_keep flags, so
    the thresholds are the percentiles rules.py prints. The random selectors
    draw from the same eligible board, so the only thing they lack is the
    selection itself.
    """
    e = day[day["elig"]]
    if len(e) < STICKY_MIN_ROWS:
        return [], ""
    live = set(e["ticker"])
    tie = ""
    if how == "card":
        keep_ok = set(e.loc[e["in_keep"], "ticker"])
        kept = [t for t in held if t in keep_ok]
        need = K - len(kept)
        if need > 0:
            cand = e[e["in_entry"] & ~e["ticker"].isin(kept)]
            if tie_rng is not None:
                cand = cand.iloc[tie_rng.permutation(len(cand))]
            take = cand.nlargest(need, "hi52")
            if len(cand) > need:
                cut = take["hi52"].min()
                n_tied = int((cand["hi52"] == cut).sum())
                n_in = int((take["hi52"] == cut).sum())
                if n_tied > n_in:
                    by = ("random draw" if tie_rng is not None
                          else "ticker order")
                    tie = (f"{n_tied} names tied at hi52={cut:.4f} for "
                           f"{n_in} slot(s); broken by {by}")
            kept = kept + list(take["ticker"])
    elif how in ("random_fresh", "random_sticky"):
        kept = [t for t in held if t in live] if how == "random_sticky" else []
        pool = sorted(live - set(kept))
        kept = kept + list(rng.choice(pool, size=min(K - len(kept),
                                                     len(pool)),
                                      replace=False))
    else:
        raise ValueError(how)
    basket = [t for t in kept if t in live][:K]
    return (basket if len(basket) >= MIN_BASKET else []), tie


def run(capital: float, start: str, end: Optional[str] = None, lag: int = 1,
        spread_ticks: float = 0.5, raw: Optional[pd.DataFrame] = None,
        index_path: str = INDEX, selector: str = "card", seed: int = 0,
        tie_seed: Optional[int] = None, div_tax: float = DIV_TAX,
        prepared: Optional[pd.DataFrame] = None) -> Dict:
    """One account. `prepared` is stoptest.prep(raw), passed in when many
    runs share one panel so the percentile pass is done once."""
    if selector not in SELECTORS:
        raise ValueError(selector)
    raw = load() if raw is None else raw
    P = st.prep(raw) if prepared is None else prepared
    raw = raw.sort_values(["date", "ticker"])
    dates = pd.DatetimeIndex(np.sort(raw["date"].unique()))
    if end is not None:
        dates = dates[dates <= pd.Timestamp(end)]
    #  the first session on or after `start`, and the review the evening
    #  before it
    first = int(dates.searchsorted(pd.Timestamp(start)))
    if first < 1 or first >= len(dates):
        raise SystemExit(f"no session before and after {start} in the panel")
    d0 = first - 1
    decisions = set(range(d0, len(dates), st.FREQ))
    by_raw = {d: g.set_index("ticker") for d, g in raw.groupby("date")}
    by_sel = {d: g for d, g in P.groupby("date")}
    rng = np.random.default_rng(seed)
    tie_rng = None if tie_seed is None else np.random.default_rng(tie_seed)
    frozen = selector == "hold_first"
    how = "card" if frozen else selector

    acct = Account(capital, spread_ticks, div_tax)
    pending: Optional[List[str]] = None       # basket waiting to be filled
    pending_at = None
    basket: List[str] = []
    to_sell: set = set()                      # names out of the basket
    to_buy: Dict[str, bool] = {}              # basket names not yet bought
    blocked: set = set()                      # stopped since the last review
    curve, ew_curve = [], []
    ew: Optional[Dict[str, float]] = None     # equal-weight universe, Rp
    ew_adj: Dict[str, float] = {}
    ew_names: List[str] = []
    stops = tps = 0

    def fill(d, close, adj, tradeable, first_pass):
        """Execute what the current basket still needs on this bar.

        On the review's fill day this sells what left, trims what is over
        weight and buys up to equal weight. Afterwards it retries only the
        orders a locked or stale bar blocked, never a fresh rebalance."""
        for t in [t for t in list(to_sell) if t in acct.pos]:
            if tradeable.get(t, False):
                acct.sell_value(d, t, acct.pos[t]["val"], close[t],
                                "review: left the keep band")
                to_sell.discard(t)
        to_sell.intersection_update(acct.pos)
        if not basket:
            return
        target = acct.value() / len(basket)
        ok = [t for t in basket if tradeable.get(t, False)
              and np.isfinite(close.get(t, np.nan)) and t not in blocked]
        if first_pass:
            for t in ok:                       # trims fund the buys
                if t in acct.pos and acct.pos[t]["val"] > target:
                    lots = int((acct.pos[t]["val"] - target)
                               // (LOT * close[t]))
                    if lots > 0:
                        acct.sell_value(d, t, lots * LOT * close[t],
                                        close[t], "review: trim to equal",
                                        lots=lots)
        for t in ok:
            if not (first_pass or to_buy.get(t)):
                continue
            have = acct.pos[t]["val"] if t in acct.pos else 0.0
            want = max(target - have, 0.0)
            cst = side_cost(close[t], BUY_FEE, spread_ticks)
            #  size the order so its COST fits the equal share; sizing on
            #  price alone lets the first names eat the last ones' fees
            lots = min(int(want // (LOT * close[t] * (1.0 + cst))),
                       lots_affordable(acct.cash, close[t], cst))
            acct.buy(d, t, lots, close[t], adj[t])
            to_buy.pop(t, None)
        if first_pass:
            for t in basket:
                if t not in ok and t not in blocked:
                    to_buy[t] = True

    for i in range(d0, len(dates)):
        d = dates[i]
        g = by_raw.get(d)
        if g is None:
            continue
        adj = g["adj_close"].to_dict()
        close = g["close"].to_dict()
        tradeable = g["tradeable"].fillna(False).astype(bool).to_dict()
        acct.mark(adj, close)

        # ---- stop and take-profit, every session after the fill ---------
        for t in ([] if frozen else list(acct.pos)):
            p = acct.pos[t]
            if not tradeable.get(t, False):
                continue                       # locked or stale: cannot sell
            r = p["adj"] / p["entry_adj"] - 1.0
            if r <= -STOP:
                acct.sell_value(d, t, p["val"], close[t],
                                f"stop: {r:+.1%} from entry")
                to_buy.pop(t, None)
                blocked.add(t)
                stops += 1
            elif r >= TP and not p["tp_done"]:
                p["tp_done"] = True
                acct.sell_value(d, t, p["val"] * TP_FRAC, close[t],
                                f"take-profit: half sold at {r:+.1%}")
                tps += 1

        # ---- fill a pending review, or retry what a locked bar blocked ---
        if pending is not None and i >= pending_at:
            basket = pending
            to_sell.clear()
            to_sell.update(t for t in acct.pos if t not in basket)
            to_buy.clear()
            fill(d, close, adj, tradeable, first_pass=True)
            pending = None
            if ew is None and ew_names:
                #  every eligible name of the decision day, equal weight,
                #  bought at this close and never touched
                e = [t for t in ew_names if t in close and
                     tradeable.get(t, False)]
                per = capital / len(e)
                ew = {t: per * (1.0 - side_cost(close[t], BUY_FEE,
                                                spread_ticks)) for t in e}
                ew_adj = {t: adj[t] for t in e}
        elif to_sell or to_buy:
            fill(d, close, adj, tradeable, first_pass=False)
        if ew is not None:
            for t in ew:
                a = adj.get(t)
                if a is not None and np.isfinite(a) and a > 0:
                    ew[t] *= a / ew_adj[t]
                    ew_adj[t] = a

        # ---- a review decision at this close ----------------------------
        if i in decisions and not (frozen and i > d0):
            day = by_sel.get(d)
            if day is not None:
                #  a held name that has stopped printing is a delisting,
                #  realised at its last value (stoptest.walk; H41's bug)
                for t in [t for t in acct.pos if t not in g.index]:
                    acct.sell_value(d, t, acct.pos[t]["val"],
                                    acct.pos[t]["close"],
                                    "stopped printing: realised at last "
                                    "value")
                pick, tie = select(day, list(acct.pos), how, rng, tie_rng)
                if i == d0:
                    ew_names = list(day.loc[day["elig"], "ticker"])
                if pick:
                    pending, pending_at = pick, i + lag
                    blocked.clear()
                    acct.note(d, "REVIEW", ",".join(pick), 0, 0.0, 0.0,
                              f"{int(day['elig'].sum())} eligible"
                              + (f"; TIE: {tie}" if tie else ""))
                    if lag == 0:
                        basket = pending
                        to_sell.clear()
                        to_sell.update(t for t in acct.pos
                                       if t not in basket)
                        to_buy.clear()
                        fill(d, close, adj, tradeable, first_pass=True)
                        pending = None
        if i >= first or (lag == 0 and i == d0):
            curve.append((d, acct.value()))
            ew_curve.append((d, sum(ew.values()) if ew else capital))

    s = pd.Series([v for _, v in curve], index=[d for d, _ in curve])
    es = pd.Series([v for _, v in ew_curve], index=s.index)
    J = pd.read_csv(index_path, parse_dates=["date"]).set_index("date")[
        "close"]
    J = J[J > 0].reindex(J.index.union(s.index)).ffill().reindex(s.index)
    days = (s.index - s.index[0]).days.to_numpy(float)
    ihsg = capital * (J / J.iloc[0]).to_numpy() * (1 + IDX_YIELD) ** (
        days / 365.25)
    roll = s / s.cummax() - 1.0
    hold = []
    last = by_raw[s.index[-1]]
    for t, p in sorted(acct.pos.items()):
        px = float(last.loc[t, "close"]) if t in last.index else float("nan")
        hold.append({"ticker": t, "value": p["val"], "lots": p["lots"],
                     "total_return": p["adj"] / p["entry_adj"] - 1.0,
                     "price_return": px / p["entry_px"] - 1.0,
                     "close": px, "entry_px": p["entry_px"]})
    trough = roll.idxmin()
    return {"capital": capital, "start": str(s.index[0].date()),
            "end": str(s.index[-1].date()), "lag": lag,
            "spread_ticks": spread_ticks, "selector": selector,
            "seed": seed, "tie_seed": tie_seed, "div_tax": div_tax,
            "end_value": float(s.iloc[-1]), "cash": acct.cash,
            "dividends": acct.dividends,
            "peak": float(s.max()), "peak_date": str(s.idxmax().date()),
            "max_dd": float(roll.min()),
            "max_dd_window": [str(s.loc[:trough].idxmax().date()),
                              str(trough.date())],
            "low": float(s.min()), "low_date": str(s.idxmin().date()),
            "stops": stops, "take_profits": tps,
            "ties": [x["why"] for x in acct.log if "TIE" in x["why"]],
            "ihsg_end": float(ihsg[-1]),
            "ihsg_max_dd": float((pd.Series(ihsg) /
                                  pd.Series(ihsg).cummax() - 1).min()),
            "ew_end": float(es.iloc[-1]),
            "holdings": hold, "log": acct.log,
            "curve": {str(k.date()): float(v) for k, v in s.items()},
            "ihsg_curve": {str(k.date()): float(v)
                           for k, v in zip(s.index, ihsg)},
            "ew_curve": {str(k.date()): float(v) for k, v in es.items()}}


def rp(x: float) -> str:
    return f"Rp {x / 1e6:,.2f} jt"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--capital", type=float, default=50e6)
    ap.add_argument("--start", default="2026-01-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--lag", type=int, default=1)
    ap.add_argument("--spread-ticks", type=float, default=0.5)
    ap.add_argument("--div-tax", type=float, default=DIV_TAX)
    ap.add_argument("--selector", choices=SELECTORS, default="card")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tie-seed", type=int, default=None)
    ap.add_argument("--out", default=os.path.join("reports",
                                                  "card_account.json"))
    a = ap.parse_args(argv)
    r = run(a.capital, a.start, a.end, a.lag, a.spread_ticks,
            selector=a.selector, seed=a.seed, tie_seed=a.tie_seed,
            div_tax=a.div_tax)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(r, fh, indent=1)
    print(f"{rp(r['capital'])} from {r['start']} to {r['end']}, "
          f"selector {a.selector} (orders fill {a.lag} session(s) after the "
          f"decision; spread {a.spread_ticks:g} tick per round trip; dividend "
          f"tax {a.div_tax:.0%})\n")
    for x in r["log"]:
        if x["action"] == "REVIEW":
            print(f"{x['date']}  REVIEW  -> {x['ticker']}  ({x['why']})")
        else:
            print(f"{x['date']}  {x['action']:<4} {x['ticker']:<6} "
                  f"{x['lots']:>5} lots @ {x['price']:>9,.0f}  "
                  f"{rp(x['rupiah']):>16}  {x['why']}")
    print(f"\nEND {r['end']}: {rp(r['end_value'])} "
          f"({r['end_value'] / r['capital'] - 1:+.1%}), of which cash "
          f"{rp(r['cash'])}; dividends received {rp(r['dividends'])} gross")
    print(f"high {rp(r['peak'])} on {r['peak_date']}, low {rp(r['low'])} on "
          f"{r['low_date']}; deepest fall {r['max_dd']:.1%} "
          f"({r['max_dd_window'][0]} -> {r['max_dd_window'][1]}); stops "
          f"{r['stops']}, take-profits {r['take_profits']}")
    print(f"IHSG with dividends, same dates: {rp(r['ihsg_end'])} "
          f"({r['ihsg_end'] / r['capital'] - 1:+.1%}), deepest fall "
          f"{r['ihsg_max_dd']:.1%}")
    print(f"every eligible stock, equal weight, held: {rp(r['ew_end'])} "
          f"({r['ew_end'] / r['capital'] - 1:+.1%})")
    print("\nHELD NOW (total return includes dividends; price return is what "
          "a broker app shows):")
    for h in r["holdings"]:
        print(f"  {h['ticker']:<6} {h['lots']:>4} lots  {rp(h['value']):>14}"
              f"  total {h['total_return']:+.1%}  price "
              f"{h['price_return']:+.1%}  (entry {h['entry_px']:,.0f}, last "
              f"{h['close']:,.0f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
