#!/usr/bin/env python3
"""Generate a plan blind at a past date, then score it against what happened.

The only way a backtest earns trust is if the plan is written before the answer
is visible, so this does exactly that and nothing cleverer. For each cut-off
date the engine is constructed with ``as_of`` set, which truncates every price
series at that date inside the data layer - the screener cannot see one bar past
it. The plan is recorded. Only then are the real forward bars loaded, from a
separate unblinded engine, and the outcome scored.

Two things are deliberately *not* done, because both would flatter the result:

  * No re-picking. Whatever the screener said on the cut-off date is what gets
    scored, including the names that look obviously wrong in hindsight.
  * No survivorship repair. A pick that stopped trading is a loss, not a
    dropped row.

    python3 scripts/forward_test.py
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

from idxbot.config import load_config  # noqa: E402
from idxbot.engine import Engine  # noqa: E402

# The shipped 20-day exit: sell a quarter at +2%, then the stop goes to entry.
TARGET_PCT = 0.02
SCALE_OUT = 0.25
STOP_PCT = 0.15
MAX_DAYS = 20
COST = 0.004
TOP_N = 5
MIN_TURNOVER = 1e9


def liquid_universe(cfg, coverage_path: str = "reports/idx_all_coverage.csv") -> List[str]:
    """Names liquid enough to trade a real order in."""
    if os.path.exists(coverage_path):
        cov = pd.read_csv(coverage_path)
        return cov[cov["vt"] >= 5e9]["ticker"].tolist()
    return cfg.universe("lq45")


def pick(as_of: pd.Timestamp, tickers: List[str], profile: str = "momentum") -> pd.DataFrame:
    """Rank blind: the engine is truncated at ``as_of`` before anything is read."""
    cfg = load_config()
    engine = Engine(cfg, provider_names=["none"], verbose=False,
                    profile=profile, as_of=as_of)
    rows = []
    for ticker in tickers:
        try:
            analysis = engine.analyze(ticker, with_campaigns=False)
        except Exception:
            continue
        if analysis is None or analysis.bars.empty:
            continue
        bars = analysis.bars
        if bars["date"].max() < as_of - pd.Timedelta(days=10):
            continue  # stale/suspended as of the cut-off
        turnover = (bars["close"] * bars["volume"]).tail(20).median()
        if not np.isfinite(turnover) or turnover < MIN_TURNOVER:
            continue
        rows.append({"ticker": ticker, "score": analysis.signal.score,
                     "close": float(bars["close"].iloc[-1]),
                     "turnover": float(turnover)})
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("score", ascending=False).head(TOP_N)


def outcome(ticker: str, as_of: pd.Timestamp) -> Optional[Dict[str, object]]:
    """Score one pick on real bars after the cut-off, using the shipped exit."""
    cfg = load_config()
    engine = Engine(cfg, provider_names=["none"], verbose=False)  # unblinded
    analysis = engine.analyze(ticker, with_campaigns=False)
    if analysis is None or analysis.bars.empty:
        return None
    bars = analysis.bars
    fwd = bars[bars["date"] > as_of].head(MAX_DAYS + 1)
    if len(fwd) < 2:
        return None

    entry = float(fwd["open"].iloc[0])          # next session's open
    if not np.isfinite(entry) or entry <= 0:
        return None
    target, stop = entry * (1 + TARGET_PCT), entry * (1 - STOP_PCT)
    realised, remaining, took = 0.0, 1.0, False

    path = fwd.iloc[1:]
    for _, bar in path.iterrows():
        if bar["low"] <= stop:
            leg = stop / entry - 1.0
            return {"ticker": ticker, "entry": entry, "exit_reason":
                    "stop" if not took else "breakeven after target",
                    "ret": realised + remaining * leg,
                    "days": int((bar["date"] - fwd["date"].iloc[0]).days)}
        if not took and bar["high"] >= target:
            realised += SCALE_OUT * TARGET_PCT
            remaining -= SCALE_OUT
            took = True
            stop = max(stop, entry)             # breakeven floor
    last = float(path["close"].iloc[-1])
    return {"ticker": ticker, "entry": entry,
            "exit_reason": "time stop" + (" (target hit)" if took else ""),
            "ret": realised + remaining * (last / entry - 1.0),
            "days": MAX_DAYS}


def run(cut_offs: List[str]) -> pd.DataFrame:
    cfg = load_config()
    tickers = liquid_universe(cfg)
    print(f"universe: {len(tickers)} names with >= Rp5bn/day turnover\n")

    records = []
    for cut in cut_offs:
        as_of = pd.Timestamp(cut)
        picks = pick(as_of, tickers)
        if picks.empty:
            print(f"{cut}: no picks")
            continue
        print(f"=== PLAN WRITTEN BLIND AS OF {cut} ===")
        print(f"  {'ticker':<8}{'score':>7}{'close':>10}   -> outcome over the next 20 sessions")
        for _, p in picks.iterrows():
            out = outcome(p["ticker"], as_of)
            if out is None:
                print(f"  {p['ticker']:<8}{p['score']:>7.1f}{p['close']:>10,.0f}"
                      f"   no forward data")
                continue
            net = out["ret"] - COST
            records.append({"cut": cut, "ticker": p["ticker"], "score": p["score"],
                            "net": net, "reason": out["exit_reason"]})
            print(f"  {p['ticker']:<8}{p['score']:>7.1f}{p['close']:>10,.0f}"
                  f"   {net:+7.2%}  {out['exit_reason']}")
        done = [r for r in records if r["cut"] == cut]
        if done:
            m = np.mean([r["net"] for r in done])
            print(f"  {'basket':<8}{'':>7}{'':>10}   {m:+7.2%}   "
                  f"{sum(1 for r in done if r['net'] > 0)}/{len(done)} positive")
        print()
    return pd.DataFrame(records)


if __name__ == "__main__":
    cuts = sys.argv[1:] or ["2026-03-06", "2026-04-07", "2026-05-07",
                            "2026-06-08", "2026-07-07"]
    df = run(cuts)
    if df.empty:
        raise SystemExit(1)
    print("=" * 64)
    print(" ALL BLIND PLANS POOLED")
    print("=" * 64)
    print(f"  trades          : {len(df)}")
    print(f"  mean net        : {df['net'].mean():+.2%}")
    print(f"  median net      : {df['net'].median():+.2%}")
    print(f"  positive        : {(df['net'] > 0).mean():.0%}")
    print(f"  made >= +5%     : {(df['net'] >= 0.05).mean():.0%}")
    print(f"  worst / best    : {df['net'].min():+.2%} / {df['net'].max():+.2%}")
    by = df.groupby("cut")["net"].agg(["mean", "size"])
    print("\n  by cut-off:")
    for cut, r in by.iterrows():
        print(f"    {cut}  {r['mean']:+7.2%}  (n={int(r['size'])})")
