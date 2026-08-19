#!/usr/bin/env python3
"""Buy and sell signals from the painted legs, backtested across the big caps.

The painter draws a leg green once the price has risen `band` off the running
low and red once it has fallen `band` off the running high. Those confirmations
ARE the signals - buy when a green leg opens, sell when a red one does - and
because the painter never uses a bar it has not seen, the signals are causal by
construction.

What is compared
----------------
    12% band          the setting that reproduces the annotated chart
    20% / 25% band    wider confirmations, fewer whipsaws
    12% + model gate  the 12% band, but a long is only taken when the pooled
                      live-leg model also says the leg is rising. This is the
                      first use of the model as a veto rather than as a painter.
    buy and hold      what all of it has to beat

Two rules kept from earlier parts of this project, because breaking either one
produces numbers that are not real:

* **Point-in-time universe.** Market capitalisation exists here only as a
  current snapshot. Picking the historical universe with it is the look-ahead
  measured at +13.5%/yr in Result 76, so the backtest selects on trailing
  turnover, listing age and volatility, and the market-cap screen only names
  what is tradeable TODAY.
* **Costs on every fill**, 0.15% buy and 0.25% sell plus slippage, and the
  signal from week t is filled at week t+1.

    python3 scripts/leg_signals.py --min-mcap 1e13
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

from idxbot.config import load_config          # noqa: E402
from idxbot.data.cache import Cache            # noqa: E402
from idxbot.data.ohlcv import YahooOHLCV       # noqa: E402
from legpaint import (build_panel, smooth_state, unadjusted_weekly,   # noqa: E402
                      walk_forward_pooled, zigzag_labels)
from turn_trader import ROUND_TRIP, reversal_state, run   # noqa: E402


def market_caps() -> pd.Series:
    """Current snapshot. Used to name today's universe, never to select history."""
    rows = []
    for p in sorted(glob.glob("data/cache/fundamentals/*.csv.gz")):
        try:
            d = pd.read_csv(p)
        except Exception:
            continue
        if "market_cap" in d.columns and len(d):
            rows.append((str(d["ticker"].iloc[0]).upper(),
                         float(d["market_cap"].iloc[0])))
    return pd.Series(dict(rows)).dropna()


def signal_stats(px: np.ndarray, state: np.ndarray, years: float,
                 cost: float = ROUND_TRIP) -> Dict[str, float]:
    eq, trades = run(px, state, cost)
    bh = px[-1] / px[0]
    peak = np.maximum.accumulate(eq)
    return {
        "growth": float(eq[-1]),
        "cagr": float(eq[-1]) ** (1 / years) - 1 if years > 0 else np.nan,
        "bh_cagr": bh ** (1 / years) - 1 if years > 0 else np.nan,
        "trades": int(trades), "time_in": float(state.mean()),
        "max_dd": float((eq / peak - 1).min()),
        "bh_dd": float((px / np.maximum.accumulate(px) - 1).min()),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-mcap", type=float, default=1e13,
                    help="Rp; 1e13 = Rp10 trillion, the real big-cap line")
    ap.add_argument("--threshold", type=float, default=0.12)
    ap.add_argument("--folds", type=int, default=6)
    ap.add_argument("--gate-hi", type=float, default=0.55)
    ap.add_argument("--gate-lo", type=float, default=0.45)
    args = ap.parse_args()
    os.makedirs("reports", exist_ok=True)

    cfg = load_config()
    loader = YahooOHLCV(cfg, Cache(cfg.path("data.cache_dir", "data/cache")))
    mcap = market_caps()
    big = sorted(mcap[mcap >= args.min_mcap].index)
    names = sorted(set(cfg.universe("bluechip")) | set(cfg.universe("lq45"))
                   | set(big))
    print(f"market-cap screen: {len(big)} names at Rp{args.min_mcap/1e12:,.0f}T+ "
          f"(snapshot; used for today's list only)")

    panel = build_panel(names, args.threshold)
    preds = walk_forward_pooled(panel, args.threshold, args.folds)

    rows: List[Dict] = []
    for t, (w, F) in panel.items():
        px = w.to_numpy(float)
        if len(px) < 150:
            continue
        years = (w.index[-1] - w.index[0]).days / 365.25
        prob = preds[t]
        # score everything on the SAME bars the model can speak for
        m = np.isfinite(prob)
        if m.sum() < 100:
            continue
        first = int(np.flatnonzero(m)[0])
        sub = px[first:]
        yrs = (w.index[-1] - w.index[first]).days / 365.25
        base = {"ticker": t, "mcap": float(mcap.get(t, np.nan)),
                "bars": int(len(sub)), "years": yrs}

        variants = {
            "band 12%": reversal_state(px, 0.12, 0.12)[first:],
            "band 20%": reversal_state(px, 0.20, 0.20)[first:],
            "band 25%": reversal_state(px, 0.25, 0.25)[first:],
        }
        gate = smooth_state(prob, args.gate_hi, args.gate_lo)[first:]
        variants["12% + model gate"] = (variants["band 12%"] & gate).astype(np.int8)
        for k, st in variants.items():
            rows.append({**base, "signal": k, **signal_stats(sub, st, yrs)})

    R = pd.DataFrame(rows)
    R["excess"] = R["cagr"] - R["bh_cagr"]
    R.to_csv("reports/leg_signals.csv", index=False)

    tradeable = R[R["mcap"] >= args.min_mcap]
    print(f"\n{'=' * 96}\n SIGNALS FROM THE PAINTED LEGS — "
          f"{tradeable['ticker'].nunique()} big caps, out-of-sample window"
          f"\n{'=' * 96}")
    print(f" {'signal':<20}{'median CAGR':>13}{'vs hold':>10}{'beats hold':>12}"
          f"{'median DD':>11}{'hold DD':>10}{'trades':>8}")
    for k, g in tradeable.groupby("signal"):
        print(f" {k:<20}{g['cagr'].median():>+13.1%}{g['excess'].median():>+10.2%}"
              f"{(g['excess'] > 0).mean():>12.0%}{g['max_dd'].median():>11.0%}"
              f"{g['bh_dd'].median():>10.0%}{g['trades'].median():>8.0f}")
    bh = tradeable[tradeable["signal"] == "band 12%"]
    print(f" {'buy and hold':<20}{bh['bh_cagr'].median():>+13.1%}{0.0:>+10.2%}"
          f"{'':>12}{bh['bh_dd'].median():>11.0%}{'':>10}{1:>8}")

    print(f"\n{'=' * 96}\n DOES IT WORK ON EVERY BIG CAP, OR JUST ON AVERAGE?"
          f"\n{'=' * 96}")
    for k, g in tradeable.groupby("signal"):
        q = g["excess"].quantile([0.1, 0.25, 0.5, 0.75, 0.9])
        print(f" {k:<20}p10 {q[0.1]:>+7.1%}  p25 {q[0.25]:>+7.1%}  "
              f"p50 {q[0.5]:>+7.1%}  p75 {q[0.75]:>+7.1%}  p90 {q[0.9]:>+7.1%}")

    best = tradeable.groupby("signal")["excess"].median().idxmax()
    bg = tradeable[tradeable["signal"] == best]
    print(f"\n best by median excess: {best} at {bg['excess'].median():+.2%}/yr, "
          f"beating hold on {(bg['excess'] > 0).mean():.0%} of names")
    print(f" its drawdown {bg['max_dd'].median():.0%} against "
          f"{bg['bh_dd'].median():.0%} for holding")
    print("\n -> reports/leg_signals.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
