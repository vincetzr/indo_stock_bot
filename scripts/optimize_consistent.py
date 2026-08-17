#!/usr/bin/env python3
"""Optimise for the highest return that is CONSISTENT across IDX history.

The book in Part XIV compounds at +25% a year and is a bad thing to own: eight
losing years in twenty-six, a -66% worst drawdown, and -53.8% as of today. Most
of its compounding came from three manias. Maximising CAGR alone selects for
exactly that shape, because one 140% year pays for four bad ones in the mean and
tells you nothing about what you will experience.

So this optimises a different objective and reports both, and it adds the one
mechanism the CAGR-only search never had: **a path-dependent exit**. A momentum
book that only rebalances on a schedule rides every position all the way down
between rebalances. Selling a holding when it falls from its own peak - and
letting the next rebalance buy back in - is the portfolio version of "sell the
peak, buy the low", and whether it helps is an empirical question this engine
can finally answer.

What is searched
----------------
    lookback     momentum window used for ranking
    top_n        how many names held
    rebalance    sessions between reshuffles
    trail        sell a holding if it falls this far from its own high
                 since entry; None means ride it to the next rebalance

Objectives reported for every configuration
-------------------------------------------
    cagr             what the CAGR-only search maximises
    median_year      the typical year, not the mean one
    worst_year       what the bad ones cost
    pct_positive     share of years that made money
    worst_3y         worst rolling three-year outcome
    max_dd           deepest peak-to-trough
    ulcer            RMS drawdown - punishes long time spent underwater,
                     which a single max-drawdown number hides

Execution is the same as Part XIV: whole lots, 0.15%/0.25% fees, next-bar fills,
positions capped at 10% of the name's daily turnover, daily returns capped at the
+/-35% auto-rejection band, dividends collected.

    python3 scripts/optimize_consistent.py [--quick]
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

from idxbot.config import load_config          # noqa: E402
from idxbot.data.cache import Cache            # noqa: E402
from idxbot.data.ohlcv import YahooOHLCV       # noqa: E402

LOT = 100
FEE_BUY, FEE_SELL = 0.0015, 0.0025
MIN_TURNOVER = 1e10
MAX_PARTICIPATION = 0.10
CAP = 0.35
CAPITAL = 50_000_000.0


def load_wide(verbose: bool = True) -> Dict[str, pd.DataFrame]:
    """Aligned wide frames: open, mark, momentum, turnover, dividend factor."""
    cfg = load_config()
    loader = YahooOHLCV(cfg, Cache(cfg.path("data.cache_dir", "data/cache")))
    names = sorted(set(cfg.universe("idx_all")) | set(cfg.universe("bluechip"))
                   | set(cfg.universe("lq45")) | set(cfg.universe("conglomerate")))
    raw = loader.get_many(names, max_age=86400 * 30, verbose=False)
    keep = {t: d.sort_values("date").reset_index(drop=True)
            for t, d in raw.items() if len(d) >= 400}
    dates = pd.DatetimeIndex(sorted({d for x in keep.values() for d in x["date"]}))
    cols = list(keep)
    out = {k: pd.DataFrame(index=dates, columns=cols, dtype=float)
           for k in ("open", "close", "tv", "fac")}
    mom: Dict[int, pd.DataFrame] = {}
    for lb in (60, 120, 250):
        mom[lb] = pd.DataFrame(index=dates, columns=cols, dtype=float)

    for t, x in keep.items():
        i = pd.DatetimeIndex(x["date"])
        m = i.isin(dates)
        c = x["close"].astype(float)
        out["open"].loc[i[m], t] = x["open"].to_numpy()[m]
        out["close"].loc[i[m], t] = c.to_numpy()[m]
        out["tv"].loc[i[m], t] = (c * x["volume"]).rolling(
            20, min_periods=10).median().to_numpy()[m]
        pr = c.pct_change()
        tr = x["adj_close"].pct_change()
        f = ((1 + tr) / (1 + pr)).replace([np.inf, -np.inf], np.nan)
        out["fac"].loc[i[m], t] = f.where((f > 0.5) & (f < 1.5), 1.0).fillna(1.0).to_numpy()[m]
        for lb in mom:
            mom[lb].loc[i[m], t] = (c / c.shift(lb) - 1.0).to_numpy()[m]

    out["fac"] = out["fac"].fillna(1.0)
    # Carry marks forward for VALUATION only. A name that did not trade has no
    # price on that row; left as NaN it values the holding at zero and invents
    # a -99% day (Result 52). Fills still require a day the stock traded.
    out["mark"] = out["close"].ffill()
    out["mom"] = mom
    if verbose:
        print(f"universe: {len(cols)} names, {dates[0]:%Y-%m-%d} -> {dates[-1]:%Y-%m-%d}")
    return out


def simulate(W: Dict, lookback: int, top_n: int, rebalance: int,
             trail: Optional[float] = None) -> Tuple[pd.Series, int]:
    """Run one configuration, returning the daily equity curve and trade count.

    ``trail`` is checked every day against each holding's high since entry, so a
    position can be exited between rebalances. That is the only way a scheduled
    book can 'sell the peak' rather than ride a name down for twenty sessions.
    """
    dates = W["mark"].index
    op, mark, tv, fac = W["open"], W["mark"], W["tv"], W["fac"]
    mom = W["mom"][lookback]
    op_v, mark_v, tv_v, fac_v, mom_v = (f.to_numpy() for f in (op, mark, tv, fac, mom))
    cols = list(mark.columns)
    idx = {t: j for j, t in enumerate(cols)}
    tradable = ~np.isnan(op_v)

    cash = CAPITAL
    lots: Dict[str, int] = {}
    peak: Dict[str, float] = {}          # each holding's high since entry
    blocked: Dict[str, int] = {}         # stopped-out names, until next rebalance
    equity = np.empty(len(dates))
    trades = 0

    for n in range(len(dates)):
        for t, l in list(lots.items()):
            j = idx[t]
            f = fac_v[n, j]
            m = mark_v[n, j]
            if np.isfinite(f) and f > 1.0 and np.isfinite(m):
                cash += l * LOT * m * (f - 1.0)
            if np.isfinite(m):
                peak[t] = max(peak.get(t, m), m)

        # --- path-dependent exit, checked daily ---
        if trail is not None and n + 1 < len(dates):
            for t, l in list(lots.items()):
                j = idx[t]
                m = mark_v[n, j]
                if not np.isfinite(m) or peak.get(t, 0) <= 0:
                    continue
                if m / peak[t] - 1.0 <= -trail and tradable[n + 1, j]:
                    p = op_v[n + 1, j]
                    cash += l * LOT * p * (1 - FEE_SELL)
                    lots.pop(t); peak.pop(t, None); blocked[t] = n + 1
                    trades += 1

        if n % rebalance == 0 and n + 1 < len(dates):
            score = np.where((tv_v[n] >= MIN_TURNOVER) & (mark_v[n] >= 50)
                             & tradable[n + 1], mom_v[n], np.nan)
            blocked.clear()      # a rebalance re-opens every name
            order = np.argsort(np.where(np.isnan(score), -np.inf, score))[::-1]
            want = [cols[j] for j in order[:top_n] if np.isfinite(score[j])]

            for t in list(lots):
                if t in want:
                    continue
                j = idx[t]
                if not tradable[n + 1, j]:
                    continue
                cash += lots.pop(t) * LOT * op_v[n + 1, j] * (1 - FEE_SELL)
                peak.pop(t, None); trades += 1

            if want:
                held = sum(l * LOT * mark_v[n + 1, idx[t]] for t, l in lots.items()
                           if np.isfinite(mark_v[n + 1, idx[t]]))
                budget = (cash + held) / len(want)
                for t in want:
                    j = idx[t]
                    p = op_v[n + 1, j]
                    if not np.isfinite(p) or p <= 0:
                        continue
                    capv = MAX_PARTICIPATION * tv_v[n, j]
                    allowed = min(budget, capv) if np.isfinite(capv) else budget
                    target = int(allowed / (p * LOT * (1 + FEE_BUY)))
                    have = lots.get(t, 0)
                    if target > have:
                        buy = target - have
                        cost = buy * LOT * p * (1 + FEE_BUY)
                        while buy > 0 and cost > cash:
                            buy -= 1
                            cost = buy * LOT * p * (1 + FEE_BUY)
                        if buy > 0:
                            cash -= cost
                            lots[t] = have + buy
                            peak[t] = max(peak.get(t, p), p)
                            trades += 1
                    elif target < have:
                        sell = have - target
                        cash += sell * LOT * p * (1 - FEE_SELL)
                        lots[t] = have - sell
                        if not lots[t]:
                            lots.pop(t); peak.pop(t, None)
                        trades += 1

        mv = 0.0
        for t, l in lots.items():
            m = mark_v[n, idx[t]]
            if np.isfinite(m):
                mv += l * LOT * m
        equity[n] = cash + mv

    return pd.Series(equity, index=dates), trades


def score_curve(eq: pd.Series, trades: int) -> Dict[str, float]:
    """Return AND consistency, because the two disagree and only one is quoted."""
    yrs = max((eq.index[-1] - eq.index[0]).days / 365.25, 1e-9)
    total = eq.iloc[-1] / eq.iloc[0]
    annual = eq.resample("YE").last().pct_change().dropna()
    peak = eq.cummax()
    dd = eq / peak - 1.0
    roll3 = eq.pct_change(756).dropna()
    return {
        "final": float(eq.iloc[-1]), "growth": float(total),
        "cagr": float(total ** (1 / yrs) - 1),
        "median_year": float(annual.median()) if len(annual) else np.nan,
        "worst_year": float(annual.min()) if len(annual) else np.nan,
        "pct_positive": float((annual > 0).mean()) if len(annual) else np.nan,
        "worst_3y": float(roll3.min()) if len(roll3) else np.nan,
        "max_dd": float(dd.min()),
        # RMS drawdown: a book that spends years underwater scores badly here
        # even when its single worst day looks the same as a book that recovers.
        "ulcer": float(np.sqrt((dd ** 2).mean())),
        "trades": float(trades),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    os.makedirs("reports", exist_ok=True)
    W = load_wide()

    lookbacks = (120,) if args.quick else (60, 120, 250)
    tops = (5, 10, 20) if args.quick else (3, 5, 8, 12, 20, 30)
    rebals = (20,) if args.quick else (5, 10, 20, 60)
    trails = (None, 0.20) if args.quick else (None, 0.12, 0.20, 0.30)

    rows = []
    combos = list(itertools.product(lookbacks, tops, rebals, trails))
    print(f"searching {len(combos)} configurations\n")
    for k, (lb, tn, rb, tr) in enumerate(combos, 1):
        eq, nt = simulate(W, lb, tn, rb, tr)
        rows.append({"lookback": lb, "top_n": tn, "rebalance": rb,
                     "trail": tr if tr else 0.0, **score_curve(eq, nt)})
        if k % 10 == 0:
            print(f"  {k}/{len(combos)}")
    R = pd.DataFrame(rows)
    R.to_csv("reports/consistency_sweep.csv", index=False)

    def show(df, title, cols=12):
        print(f"\n{title}")
        print(f" {'lb':>4}{'top':>5}{'reb':>5}{'trail':>7}{'CAGR':>8}{'medYr':>8}"
              f"{'worstYr':>9}{'+yrs':>6}{'worst3y':>9}{'maxDD':>8}{'ulcer':>7}{'trades':>8}")
        for _, r in df.head(cols).iterrows():
            print(f" {r['lookback']:>4.0f}{r['top_n']:>5.0f}{r['rebalance']:>5.0f}"
                  f"{r['trail']:>7.0%}{r['cagr']:>+8.1%}{r['median_year']:>+8.1%}"
                  f"{r['worst_year']:>+9.1%}{r['pct_positive']:>6.0%}"
                  f"{r['worst_3y']:>+9.1%}{r['max_dd']:>8.0%}{r['ulcer']:>7.2f}"
                  f"{r['trades']:>8.0f}")

    show(R.sort_values("cagr", ascending=False), "RANKED BY CAGR (what maximising return alone picks)")
    show(R.sort_values("median_year", ascending=False), "RANKED BY MEDIAN YEAR (the typical year)")
    R["consistency"] = R["cagr"] / R["ulcer"].replace(0, np.nan)
    show(R.sort_values("consistency", ascending=False),
         "RANKED BY CAGR PER UNIT OF TIME-UNDERWATER (return you can hold)")
    print("\n DOES THE TRAILING EXIT HELP? median across all configurations")
    print(R.groupby("trail")[["cagr", "median_year", "max_dd", "ulcer", "trades"]]
          .median().to_string(formatters={
              "cagr": "{:+.1%}".format, "median_year": "{:+.1%}".format,
              "max_dd": "{:.0%}".format, "ulcer": "{:.2f}".format,
              "trades": "{:,.0f}".format}))
    print("\n -> reports/consistency_sweep.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
