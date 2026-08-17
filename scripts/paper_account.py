#!/usr/bin/env python3
"""Rp50,000,000 at the open of the data, run to today, every transaction logged.

This is the validated engine from Part VIII - rank liquid IDX names on momentum
within each date, own the strongest few, rebalance, always invested - run as a
real account rather than as an index of returns. That means:

* **whole lots only** (1 lot = 100 shares), so small accounts cannot hold five
  names in expensive stocks and the simulation refuses rather than pretending;
* **IDX retail fees**: 0.15% to buy, 0.25% to sell (the extra 0.1% is the sale
  tax), charged on every fill;
* **next-bar execution** - the ranking is computed on the rebalance date and
  filled at the following session's open;
* **daily returns capped at +/-35%**, the auto-rejection band, because 0.044% of
  IDX daily prints are physically impossible and uncapped they compound into
  fantasy (Result 41);
* **cash is real** - leftover rupiah that cannot buy a whole lot sits idle.

Survivorship still applies and cannot be removed: the panel is today's listing,
so companies that delisted are absent. The final figure is an upper bound, and
the comparison against buy-and-hold benchmarks computed on the identical panel
is the part that survives the bias.

    python3 scripts/paper_account.py [--capital 50000000] [--top 5] [--rebalance 20]
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

from idxbot.config import load_config          # noqa: E402
from idxbot.data.cache import Cache            # noqa: E402
from idxbot.data.ohlcv import YahooOHLCV       # noqa: E402

LOT = 100
FEE_BUY, FEE_SELL = 0.0015, 0.0025
MIN_TURNOVER = 1e10          # Rp10bn/day: a real account must be able to fill
#: Largest share of a name's daily turnover one position may represent. Above
#: this the fill is imaginary: you are the market, and the slippage exceeds the
#: edge. This is what makes a compounding account's returns decay with size.
MAX_PARTICIPATION = 0.10
CAP = 0.35


def load(verbose: bool = True) -> Dict[str, pd.DataFrame]:
    cfg = load_config()
    loader = YahooOHLCV(cfg, Cache(cfg.path("data.cache_dir", "data/cache")))
    names = sorted(set(cfg.universe("bluechip")) | set(cfg.universe("lq45"))
                   | set(cfg.universe("conglomerate")) | set(cfg.universe("idx_all")))
    raw = loader.get_many(names, max_age=86400 * 30, verbose=False)
    panel = {}
    for t, d in raw.items():
        if len(d) < 400:
            continue
        x = d.sort_values("date").reset_index(drop=True).copy()
        c = x["close"].astype(float)
        x["mom"] = c / c.shift(120) - 1.0
        x["tv"] = (c * x["volume"]).rolling(20, min_periods=10).median()
        x["adjret"] = x["adj_close"].pct_change().clip(-CAP, CAP)
        panel[t] = x
    if verbose:
        print(f"universe: {len(panel)} names with tradeable history")
    return panel


def run(panel: Dict[str, pd.DataFrame], capital: float, top_n: int,
        rebalance: int, start: Optional[str] = None):
    dates = pd.DatetimeIndex(sorted({d for x in panel.values() for d in x["date"]}))
    if start:
        dates = dates[dates >= pd.Timestamp(start)]
    # wide frames, aligned once
    px = pd.DataFrame(index=dates, columns=list(panel), dtype=float)
    op = pd.DataFrame(index=dates, columns=list(panel), dtype=float)
    mom = pd.DataFrame(index=dates, columns=list(panel), dtype=float)
    tv = pd.DataFrame(index=dates, columns=list(panel), dtype=float)
    fac = pd.DataFrame(index=dates, columns=list(panel), dtype=float)
    for t, x in panel.items():
        i = pd.DatetimeIndex(x["date"])
        k = i.isin(dates)
        px.loc[i[k], t] = x["close"].to_numpy()[k]
        op.loc[i[k], t] = x["open"].to_numpy()[k]
        mom.loc[i[k], t] = x["mom"].to_numpy()[k]
        tv.loc[i[k], t] = x["tv"].to_numpy()[k]
        # dividend factor: total return divided by price return
        pr = x["close"].pct_change()
        tr = x["adj_close"].pct_change()
        f = ((1 + tr) / (1 + pr)).replace([np.inf, -np.inf], np.nan)
        f = f.where((f > 0.5) & (f < 1.5), 1.0).fillna(1.0)
        fac.loc[i[k], t] = f.to_numpy()[k]
    fac = fac.fillna(1.0)
    # A name that did not trade on a given day has no price on that row. Left
    # as NaN it silently valued the position at ZERO - with five concentrated
    # holdings that produced -99.9% single-day "losses" while the underlying
    # stocks moved -12%. A non-trading day means the position keeps its last
    # mark, so prices are carried forward for VALUATION.
    px_mark = px.ffill()
    # Fills are different: an order can only be filled on a day the stock
    # actually traded, so the raw (un-filled) open is what execution uses.
    tradable = op.notna()

    cash = float(capital)
    holdings: Dict[str, int] = {}          # ticker -> lots
    log: List[Dict[str, object]] = []
    equity = []

    for n, day in enumerate(dates):
        # dividends on what is held
        for t, lots in holdings.items():
            f = fac.at[day, t]
            mark = px_mark.at[day, t] if day in px_mark.index else np.nan
            if np.isfinite(f) and f > 1.0 and np.isfinite(mark):
                cash += lots * LOT * mark * (f - 1.0)

        if n % rebalance == 0 and n + 1 < len(dates):
            fill_day = dates[n + 1]
            scores = mom.loc[day].where(
                (tv.loc[day] >= MIN_TURNOVER) & (px_mark.loc[day] >= 50)
                & tradable.loc[fill_day])
            want = list(scores.nlargest(top_n).index) if scores.notna().any() else []

            for t in list(holdings):                       # sell what is dropped
                if t in want:
                    continue
                p = op.at[fill_day, t]
                if not np.isfinite(p):
                    continue
                lots = holdings.pop(t)
                gross = lots * LOT * p
                cash += gross * (1 - FEE_SELL)
                log.append({"date": fill_day, "action": "SELL", "ticker": t,
                            "lots": lots, "price": p, "value": gross,
                            "cash_after": cash})
            if want:
                budget = (cash + sum(holdings.get(t, 0) * LOT * px_mark.at[fill_day, t]
                                     for t in holdings
                                     if np.isfinite(px_mark.at[fill_day, t]))) / len(want)
                for t in want:                             # size each to target
                    p = op.at[fill_day, t]
                    if not np.isfinite(p) or p <= 0:
                        continue
                    # Capacity: a real account cannot take a position larger
                    # than a slice of the name's daily turnover. Without this
                    # the book compounded into Rp7.8bn positions in stocks
                    # trading Rp10bn a day - fills that would move the market
                    # by more than the edge being harvested.
                    cap_value = MAX_PARTICIPATION * tv.at[day, t]
                    allowed = min(budget, cap_value) if np.isfinite(cap_value) else budget
                    target = int(allowed / (p * LOT * (1 + FEE_BUY)))
                    have = holdings.get(t, 0)
                    if target > have:
                        buy = target - have
                        cost = buy * LOT * p * (1 + FEE_BUY)
                        while buy > 0 and cost > cash:
                            buy -= 1
                            cost = buy * LOT * p * (1 + FEE_BUY)
                        if buy > 0:
                            cash -= cost
                            holdings[t] = have + buy
                            log.append({"date": fill_day, "action": "BUY", "ticker": t,
                                        "lots": buy, "price": p, "value": cost,
                                        "cash_after": cash})
                    elif target < have:
                        sell = have - target
                        gross = sell * LOT * p
                        cash += gross * (1 - FEE_SELL)
                        holdings[t] = have - sell
                        if holdings[t] == 0:
                            holdings.pop(t)
                        log.append({"date": fill_day, "action": "TRIM", "ticker": t,
                                    "lots": sell, "price": p, "value": gross,
                                    "cash_after": cash})
        mv = sum(l * LOT * px_mark.at[day, t] for t, l in holdings.items()
                 if np.isfinite(px_mark.at[day, t]))
        equity.append({"date": day, "equity": cash + mv, "cash": cash,
                       "positions": len(holdings)})
    return pd.DataFrame(log), pd.DataFrame(equity).set_index("date")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--capital", type=float, default=50_000_000)
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--rebalance", type=int, default=20)
    ap.add_argument("--start")
    args = ap.parse_args()

    os.makedirs("reports", exist_ok=True)
    panel = load()
    log, eq = run(panel, args.capital, args.top, args.rebalance, args.start)
    log.to_csv("reports/paper_account_trades.csv", index=False)
    eq.to_csv("reports/paper_account_equity.csv")

    years = (eq.index[-1] - eq.index[0]).days / 365.25
    final = float(eq["equity"].iloc[-1])
    growth = final / args.capital
    peak = eq["equity"].cummax()
    print(f"\n{'='*74}\n PAPER ACCOUNT: Rp{args.capital:,.0f} "
          f"on {eq.index[0]:%Y-%m-%d}\n{'='*74}")
    print(f" final value {eq.index[-1]:%Y-%m-%d} : Rp{final:,.0f}")
    print(f" growth                    : {growth:,.1f}x over {years:.1f} years")
    print(f" CAGR                      : {growth**(1/years)-1:+.2%}")
    print(f" max drawdown              : {float((eq['equity']/peak-1).min()):.1%}")
    print(f" transactions              : {len(log):,}")
    print(f" fees paid (approx)        : Rp{(log['value']*0.002).sum():,.0f}")
    print(f" -> reports/paper_account_trades.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
