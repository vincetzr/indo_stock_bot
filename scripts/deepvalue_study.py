#!/usr/bin/env python3
"""Buy stocks far below their high, sell when they make a new one. All of IDX.

Where this came from. Solving for the *perfect* set of trades on ADRO (dynamic
programming over the whole price history, 18 round trips) and then asking what
those points looked like beforehand gave a very clear answer:

    optimal BUYS   RSI 26.6 (3rd pct), z-score -2.04 (5th pct), 49% below high
    optimal SELLS  RSI 66.3 (90th pct), z-score +2.14 (92nd pct), at the high
    holding period 220-550 days for every large winner

So the shape is: enter at a deep drawdown, and **hold until a new high**. Earlier
reversion tests in this repo failed because they exited at z=0 - selling the
first bounce and never seeing the move that mattered.

This applies that rule to every liquid IDX name from 2001, as a portfolio:
equal-weight whatever is currently in a position, cash when nothing qualifies.

Honesty notes that decide whether any of this means anything:

* **Survivorship is the central risk here and it cuts one way.** A stock 50%
  below its high either recovers or delists, and the panel only contains names
  that still exist. The measured recovery rate is therefore an upper bound. This
  is the same bias that flatters every deep-value backtest ever run.
* Entry and exit are charged, and the position is taken the day *after* the
  condition is met.
* The equal-weight buy-and-hold of the same universe is the benchmark, because
  a rule that only buys after crashes will look brilliant against cash.
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

from idxbot.config import load_config            # noqa: E402
from idxbot.data.cache import Cache              # noqa: E402
from idxbot.data.ohlcv import YahooOHLCV         # noqa: E402

COST = 0.006
MIN_BARS = 1000
MIN_TURNOVER = 1e9


def load_panel(verbose: bool = True) -> Dict[str, pd.DataFrame]:
    cfg = load_config()
    loader = YahooOHLCV(cfg, Cache(cfg.path("data.cache_dir", "data/cache")))
    raw = loader.get_many(cfg.universe("idx_all"), max_age=86400 * 30, verbose=False)
    panel = {}
    for t, d in raw.items():
        if len(d) < MIN_BARS:
            continue
        x = d.sort_values("date").reset_index(drop=True).copy()
        c = x["close"].astype(float)
        x["dd"] = c / c.rolling(250, min_periods=120).max() - 1.0
        x["turnover"] = (c * x["volume"]).rolling(20, min_periods=10).median()
        # IDX auto-rejection caps a regular-market session at roughly +-35%.
        # Anything beyond that is a data error or an unadjusted corporate
        # action, and left uncapped it compounds into a fantasy: an
        # equal-weight book over 654 names produced 34,000,000x before this
        # line existed, entirely from a handful of impossible daily moves.
        x["ret"] = x["adj_close"].pct_change().clip(-0.35, 0.35).fillna(0.0)
        x["liquid"] = x["turnover"] >= MIN_TURNOVER
        panel[t] = x
    if verbose:
        print(f"panel: {len(panel)} names with >= {MIN_BARS} bars")
    return panel


def signals(panel, depth, exit_at, min_turnover=MIN_TURNOVER):
    """Per-name 0/1 holding state: enter below ``depth``, exit at ``exit_at``."""
    out = {}
    for t, x in panel.items():
        dd = x["dd"].to_numpy(); tv = x["turnover"].to_numpy()
        pos = np.zeros(len(dd)); state = 0.0
        for i in range(len(dd)):
            if np.isfinite(dd[i]):
                if state == 0.0 and dd[i] <= depth and np.isfinite(tv[i]) \
                        and tv[i] >= min_turnover:
                    state = 1.0
                elif state > 0.0 and dd[i] >= exit_at:
                    state = 0.0
            pos[i] = state
        out[t] = pd.Series(pos, index=pd.DatetimeIndex(x["date"]))
    return out


def _frames(panel, holds):
    index = pd.DatetimeIndex(sorted({d for x in panel.values() for d in x["date"]}))
    held = pd.DataFrame(0.0, index=index, columns=list(panel))
    rets = pd.DataFrame(0.0, index=index, columns=list(panel))
    for t, x in panel.items():
        idx = pd.DatetimeIndex(x["date"])
        held.loc[idx, t] = holds[t].to_numpy()
        rets.loc[idx, t] = x["ret"].to_numpy()
    return index, held.shift(1).fillna(0.0), rets


def simulate(panel, holds, cost=COST):
    """Equal-weight whatever is held that day; cash when nothing qualifies."""
    index, held, rets = _frames(panel, holds)
    weights = held.div(held.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    gross = (weights * rets).sum(axis=1)
    turnover = weights.diff().abs().sum(axis=1).fillna(0.0)
    equity = (1.0 + gross - turnover * (cost / 2.0)).cumprod()

    trades = []
    for t in panel:
        h = held[t]
        ch = h.diff().fillna(0.0)
        entries = list(index[ch > 0]); exits = list(index[ch < 0])
        for e in entries:
            later = [x for x in exits if x > e]
            x = later[0] if later else index[-1]
            sub = rets.loc[e:x, t]
            trades.append({"ticker": t, "entry": e, "exit": x,
                           "days": int((x - e).days),
                           "ret": float(np.prod(1.0 + sub.to_numpy()) - 1.0 - cost)})
    return equity, pd.DataFrame(trades), held


def benchmark(panel):
    """Equal-weight the same LIQUID universe the strategy may trade.

    Must carry the identical turnover filter, or the comparison is against a
    portfolio of untradeable microcaps rather than against not choosing.
    """
    index = pd.DatetimeIndex(sorted({d for x in panel.values() for d in x["date"]}))
    rets = pd.DataFrame(np.nan, index=index, columns=list(panel))
    mask = pd.DataFrame(False, index=index, columns=list(panel))
    for t, x in panel.items():
        idx = pd.DatetimeIndex(x["date"])
        rets.loc[idx, t] = x["ret"].to_numpy()
        mask.loc[idx, t] = x["liquid"].to_numpy()
    eligible = rets.where(mask.shift(1).fillna(False))
    return (1.0 + eligible.mean(axis=1).fillna(0.0)).cumprod()


def stats(equity):
    e = equity.to_numpy(float)
    years = max((equity.index[-1] - equity.index[0]).days / 365.25, 1e-9)
    peak = np.maximum.accumulate(e)
    return {"growth": float(e[-1]), "cagr": float(e[-1] ** (1 / years) - 1.0),
            "max_drawdown": float(np.min(e / peak - 1.0)), "years": years}


def main() -> int:
    os.makedirs("reports", exist_ok=True)
    panel = load_panel()
    b = stats(benchmark(panel))
    print(f"equal-weight universe, always invested: {b['growth']:,.1f}x "
          f"({b['cagr']:+.2%} CAGR, maxDD {b['max_drawdown']:.0%}) "
          f"over {b['years']:.1f} years\n")

    print("PARAMETER SWEEP - depth to enter, level to exit\n")
    print(f" {'enter below':>12}{'exit at':>9}{'growth':>11}{'CAGR':>9}"
          f"{'maxDD':>8}{'trades':>8}{'win%':>7}{'medhold':>9}")
    rows = []
    for depth in (-0.30, -0.40, -0.50, -0.60, -0.70):
        for exit_at in (-0.10, -0.05, 0.0):
            holds = signals(panel, depth, exit_at)
            eq, tr, _ = simulate(panel, holds)
            s = stats(eq)
            if tr.empty:
                continue
            rows.append({"depth": depth, "exit": exit_at, **s, "trades": len(tr),
                         "win": float((tr["ret"] > 0).mean()),
                         "med_hold": float(tr["days"].median()),
                         "med_ret": float(tr["ret"].median()),
                         "mean_ret": float(tr["ret"].mean())})
            print(f" {depth:>12.0%}{exit_at:>9.0%}{s['growth']:>10,.1f}x"
                  f"{s['cagr']:>9.2%}{s['max_drawdown']:>8.0%}{len(tr):>8}"
                  f"{float((tr['ret']>0).mean()):>7.0%}"
                  f"{float(tr['days'].median()):>9.0f}")
    table = pd.DataFrame(rows).sort_values("cagr", ascending=False)
    table.to_csv("reports/deepvalue_sweep.csv", index=False)
    best = table.iloc[0]
    print(f"\n best: enter below {best['depth']:.0%}, exit at {best['exit']:.0%}"
          f"  ->  {best['growth']:,.1f}x  ({best['cagr']:+.2%} CAGR)")
    print(f" equal-weight universe: {b['growth']:,.1f}x ({b['cagr']:+.2%})")
    print(f" ratio: {best['growth']/b['growth']:.1f}x better")

    holds = signals(panel, float(best["depth"]), float(best["exit"]))
    eq, tr, held = simulate(panel, holds)
    eq.to_csv("reports/deepvalue_equity.csv")
    tr.sort_values("entry").to_csv("reports/deepvalue_trades.csv", index=False)
    print(f"\n -> reports/deepvalue_trades.csv ({len(tr)} transactions)")
    print(f" names held over time: median {held.sum(axis=1).median():.0f}, "
          f"max {held.sum(axis=1).max():.0f}, "
          f"share of days fully in cash {float((held.sum(axis=1)==0).mean()):.0%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
