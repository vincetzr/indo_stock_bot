#!/usr/bin/env python3
"""Rp X into the card from a given date: the picks, every trade, the end value.

    python3 scripts/card_account.py --capital 50000000 --start 2026-01-01

WHAT THIS IS
    The shipped quarterly card (scripts/rules.py) run as one real account
    instead of a backtest statistic. The account holds whole 100-share lots and
    pays the user's Mandiri fees (A5: 0.28% to buy, 0.18% plus 0.10% tax to
    sell). Every decision uses only data available at the close it is made on:

    * REVIEW. On the last session before `--start`, and every 63 sessions
      after it, run the card's selection on that day's eligible board: keep a
      held name while it is in the top 30% on hi52 and the calmest 60% on
      vol60; fill empty slots from the top 10% / calmest half, best hi52
      first; hold at most 10. The orders fill at the NEXT session's close,
      because you see the card in the evening and trade the next day (the
      backtests fill at the same close; `--lag 0` reproduces that). Each
      review re-equalises the book, as the measured rule does.
    * STOP. A position whose close is 20% or more below its entry is sold at
      that close (a resting order, measured on the close as in H56).
    * TAKE-PROFIT. At +100% from entry, half the position is sold, once.
    * CASH from stops and take-profits waits for the next review and earns
      nothing.
    * LOCKED OR STALE DAYS. Nothing fills on a bar the panel marks
      untradeable (auto-rejection lock or no trades). The order waits for the
      next tradeable session; selling into an ARB lock is the fiction CLAUDE.md
      section 5 names.

    Positions are valued on the adjusted close, so a dividend is reinvested in
    the same name, untaxed. The 10% dividend tax is ignored, at about 0.2% a
    year.

WHAT IT IS NOT
    * Not a test of the rule. The card was chosen in August-September 2026
      (H54, H56, H62) on data that already contained most of 2026, so a 2026
      start replays a period the rule was fitted on. Only the stretch after
      2026-09-04 is genuinely forward, and the forward record holds it.
    * One path. The rule's own backtest shows the year's result moving by
      more than 20 points with the review calendar alone.
    * Survivor-only panel, no market impact, close-based fills.
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
STICKY_MIN_ROWS = 3 * K           # beathold.Sticky refuses below this


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

    def __init__(self, capital: float, spread_ticks: float):
        self.cash = float(capital)
        self.pos: Dict[str, Dict] = {}
        self.spread = spread_ticks
        self.log: List[Dict] = []

    def value(self) -> float:
        return self.cash + sum(p["val"] for p in self.pos.values())

    def _note(self, d, kind, t, lots, px, rupiah, why=""):
        self.log.append({"date": str(pd.Timestamp(d).date()), "action": kind,
                         "ticker": t, "lots": int(lots), "price": float(px),
                         "rupiah": float(rupiah), "why": why})

    def buy(self, d, t, lots, close, adj):
        if lots <= 0:
            return
        c = side_cost(close, BUY_FEE, self.spread)
        gross = lots * LOT * close
        self.cash -= gross * (1.0 + c)
        p = self.pos.setdefault(t, {"val": 0.0, "entry_adj": adj,
                                    "entry_px": close, "tp_done": False,
                                    "adj": adj})
        p["val"] += gross
        self._note(d, "BUY", t, lots, close, gross * (1.0 + c))

    def sell_value(self, d, t, val, close, why):
        """Sell `val` rupiah of a position at this close."""
        p = self.pos[t]
        val = min(val, p["val"])
        c = side_cost(close, SELL_FEE, self.spread)
        self.cash += val * (1.0 - c)
        p["val"] -= val
        self._note(d, "SELL", t, round(val / (LOT * close)), close,
                   val * (1.0 - c), why)
        if p["val"] <= 1.0:
            del self.pos[t]

    def mark(self, adj_today: Dict[str, float]):
        for t, p in self.pos.items():
            a = adj_today.get(t)
            if a is not None and np.isfinite(a) and a > 0:
                p["val"] *= a / p["adj"]
                p["adj"] = a


def select(day: pd.DataFrame, held: List[str]) -> List[str]:
    """The card's Sticky selection on one day's prepared rows.

    `day` carries stoptest.prep's in_entry/in_keep flags for every eligible
    name, so the thresholds are the same percentiles rules.py prints.
    """
    e = day[day["elig"]]
    if len(e) < STICKY_MIN_ROWS:
        return []
    keep_ok = set(e.loc[e["in_keep"], "ticker"])
    live = set(e["ticker"])
    kept = [t for t in held if t in keep_ok]
    need = K - len(kept)
    if need > 0:
        cand = e[e["in_entry"] & ~e["ticker"].isin(kept)]
        kept = kept + list(cand.nlargest(need, "hi52")["ticker"])
    basket = [t for t in kept if t in live][:K]
    return basket if len(basket) >= MIN_BASKET else []


def run(capital: float, start: str, end: Optional[str] = None, lag: int = 1,
        spread_ticks: float = 0.5, raw: Optional[pd.DataFrame] = None,
        index_path: str = INDEX) -> Dict:
    raw = load() if raw is None else raw
    P = st.prep(raw)
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

    acct = Account(capital, spread_ticks)
    pending: Optional[List[str]] = None       # basket waiting to be filled
    pending_at = None
    basket: List[str] = []
    to_sell: set = set()                      # names out of the basket
    to_buy: Dict[str, bool] = {}              # basket names not yet bought
    curve = []
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
              and np.isfinite(close.get(t, np.nan))]
        if first_pass:
            for t in ok:                       # trims fund the buys
                if t in acct.pos and acct.pos[t]["val"] > target:
                    lots = int((acct.pos[t]["val"] - target)
                               // (LOT * close[t]))
                    if lots > 0:
                        acct.sell_value(d, t, lots * LOT * close[t],
                                        close[t], "review: trim to equal")
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
                if t not in ok:
                    to_buy[t] = True

    for i in range(d0, len(dates)):
        d = dates[i]
        g = by_raw.get(d)
        if g is None:
            continue
        adj = g["adj_close"].to_dict()
        close = g["close"].to_dict()
        tradeable = g["tradeable"].fillna(False).astype(bool).to_dict()
        acct.mark(adj)

        # ---- stop and take-profit, every session after the fill ---------
        for t in list(acct.pos):
            p = acct.pos[t]
            if not tradeable.get(t, False):
                continue                       # locked or stale: cannot sell
            r = p["adj"] / p["entry_adj"] - 1.0
            if r <= -STOP:
                acct.sell_value(d, t, p["val"], close[t],
                                f"stop: {r:+.1%} from entry")
                to_buy.pop(t, None)
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
        elif to_sell or to_buy:
            fill(d, close, adj, tradeable, first_pass=False)

        # ---- a review decision at this close ----------------------------
        if i in decisions:
            day = by_sel.get(d)
            if day is not None:
                pick = select(day, list(acct.pos))
                if pick:
                    pending, pending_at = pick, i + lag
                    acct._note(d, "REVIEW", ",".join(pick), 0, 0.0, 0.0,
                               f"{int(day['elig'].sum())} eligible")
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

    s = pd.Series([v for _, v in curve], index=[d for d, _ in curve])
    J = pd.read_csv(index_path, parse_dates=["date"]).set_index("date")[
        "close"]
    J = J[J > 0].reindex(J.index.union(s.index)).ffill().reindex(s.index)
    days = (s.index - s.index[0]).days.to_numpy(float)
    ihsg = capital * (J / J.iloc[0]).to_numpy() * (1 + IDX_YIELD) ** (
        days / 365.25)
    dd = float((s / s.cummax() - 1.0).min())
    hold = []
    last = by_raw[s.index[-1]]
    for t, p in sorted(acct.pos.items()):
        px = float(last.loc[t, "close"]) if t in last.index else float("nan")
        hold.append({"ticker": t, "value": p["val"],
                     "since_entry": p["adj"] / p["entry_adj"] - 1.0,
                     "close": px, "entry_px": p["entry_px"]})
    return {"capital": capital, "start": str(s.index[0].date()),
            "end": str(s.index[-1].date()), "lag": lag,
            "spread_ticks": spread_ticks,
            "end_value": float(s.iloc[-1]), "cash": acct.cash,
            "peak": float(s.max()), "peak_date": str(s.idxmax().date()),
            "max_dd": dd, "stops": stops, "take_profits": tps,
            "ihsg_end": float(ihsg[-1]),
            "ihsg_max_dd": float((pd.Series(ihsg) /
                                  pd.Series(ihsg).cummax() - 1).min()),
            "holdings": hold, "log": acct.log,
            "curve": {str(k.date()): float(v) for k, v in s.items()},
            "ihsg_curve": {str(k.date()): float(v)
                           for k, v in zip(s.index, ihsg)}}


def rp(x: float) -> str:
    return f"Rp {x / 1e6:,.2f} jt"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--capital", type=float, default=50e6)
    ap.add_argument("--start", default="2026-01-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--lag", type=int, default=1)
    ap.add_argument("--spread-ticks", type=float, default=0.5)
    ap.add_argument("--out", default=os.path.join("reports",
                                                  "card_account.json"))
    a = ap.parse_args(argv)
    r = run(a.capital, a.start, a.end, a.lag, a.spread_ticks)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(r, fh, indent=1)
    print(f"{rp(r['capital'])} from {r['start']} to {r['end']}  "
          f"(orders fill {a.lag} session(s) after the decision; spread "
          f"{a.spread_ticks:g} tick per round trip)\n")
    for x in r["log"]:
        if x["action"] == "REVIEW":
            print(f"{x['date']}  REVIEW  -> {x['ticker']}  ({x['why']})")
        else:
            print(f"{x['date']}  {x['action']:<4} {x['ticker']:<6} "
                  f"{x['lots']:>5} lots @ {x['price']:>9,.0f}  "
                  f"{rp(x['rupiah']):>16}  {x['why']}")
    print(f"\nEND {r['end']}: {rp(r['end_value'])} "
          f"({r['end_value'] / r['capital'] - 1:+.1%}), of which cash "
          f"{rp(r['cash'])}")
    print(f"peak {rp(r['peak'])} on {r['peak_date']}; deepest fall from a "
          f"peak {r['max_dd']:.1%}; stops {r['stops']}, take-profits "
          f"{r['take_profits']}")
    print(f"IHSG with dividends, same dates: {rp(r['ihsg_end'])} "
          f"({r['ihsg_end'] / r['capital'] - 1:+.1%}), deepest fall "
          f"{r['ihsg_max_dd']:.1%}")
    print("\nHELD NOW:")
    for h in r["holdings"]:
        print(f"  {h['ticker']:<6} {rp(h['value']):>14}  "
              f"{h['since_entry']:+.1%} since entry (entry "
              f"{h['entry_px']:,.0f}, last {h['close']:,.0f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
