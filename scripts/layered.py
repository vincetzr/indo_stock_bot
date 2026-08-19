#!/usr/bin/env python3
"""All three layers together, scored against the null that has killed everything else.

The user's method has three layers: fundamentals and news, then broker-summary
flow, then technical analysis informed by both. Measured separately in this repo:

    layer 3, technical   Result 110: the leg-size distribution of IDX large caps
                         is indistinguishable from a driftless random walk at
                         15m, 1h, 4h and daily. Result 111: every band rule and
                         every multi-timeframe combination loses to random timing
                         at its own exposure.
    layer 2, broker flow structurally unavailable intraday - the source is
                         end-of-day - and measured at d = 0.002 (p = 0.49) and
                         d = -0.005 (p = 0.52) on two pre-registered replications.
    layer 1, regime      not yet tested as a predictor here. This script tests it.

So this is the last place an edge can be hiding, and the question is narrow:
does gating the band rule on a REGIME condition - the sort of thing layer 1
produces - beat the band rule alone, and more importantly beat random timing at
the same exposure?

THE NULL, AND WHY IT IS THE ONLY ONE THAT COUNTS
------------------------------------------------
Any filter that keeps you out of the market during declines looks brilliant in a
falling market, for free. IHSG fell 41.5% between January and June 2026. A rule
that is in cash half the time will "beat buy-and-hold" on that tape while having
no skill whatsoever.

The honest benchmark is therefore what the SAME exposure would have earned with
no timing at all: exposure x the buy-and-hold log return. A filter earns the
right to be called a filter only by beating that. Everything here is scored
against it, and against holding, and against the ungated rule.

Filters tested, all computed causally on data available at the decision:

    none          the daily band alone
    trend         only hold while price is above its own 200-day average
    rs            only hold while the name is in the top third of 120-day
                  relative strength across the universe, ranked on lagged data
    trend_rs      both
    market        only hold while the equal-weight universe index is above its
                  own 200-day average - a market-regime gate rather than a
                  single-name one

    python3 scripts/layered.py
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

from idxbot.config import load_config          # noqa: E402
from idxbot.data.cache import Cache            # noqa: E402
from idxbot.data.ohlcv import YahooOHLCV       # noqa: E402
from leg_signals import market_caps            # noqa: E402
from paint_daily import unadjusted_daily       # noqa: E402
from paint_live import band_state              # noqa: E402

ROUND_TRIP_FEE = 0.0056
FILTERS = ("none", "trend", "rs", "trend_rs", "market")


def build_panel(loader: YahooOHLCV, tickers: List[str],
                min_len: int = 800) -> pd.DataFrame:
    cols = {}
    for t in tickers:
        s = unadjusted_daily(loader, t)
        if s is not None and len(s) >= min_len:
            cols[t] = s
    if not cols:
        return pd.DataFrame()
    return pd.DataFrame(cols).sort_index()


def gates(px: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Every filter as a boolean frame, each one lagged so it is knowable.

    The shift(1) is the whole ballgame: a 200-day average that includes today's
    close is not a filter, it is a peek. Same for the cross-sectional rank.
    """
    ma200 = px.rolling(200, min_periods=200).mean()
    trend = (px > ma200).shift(1).fillna(False)

    mom = px.pct_change(120)
    rank = mom.rank(axis=1, pct=True)
    rs = (rank > 2.0 / 3.0).shift(1).fillna(False)

    # equal-weight index of the same names, rebuilt from returns so a name
    # joining late cannot retroactively change earlier index values
    idx = (1.0 + px.pct_change().mean(axis=1)).cumprod()
    imk = (idx > idx.rolling(200, min_periods=200).mean()).shift(1).fillna(False)
    market = pd.DataFrame({c: imk for c in px.columns}, index=px.index)

    return {"none": pd.DataFrame(True, index=px.index, columns=px.columns),
            "trend": trend, "rs": rs,
            "trend_rs": trend & rs, "market": market}


def run(px: pd.Series, gate: pd.Series, band: float,
        fee: float = ROUND_TRIP_FEE) -> Optional[Dict[str, float]]:
    v = px.dropna()
    if len(v) < 400:
        return None
    st, _ = band_state(v.to_numpy(float), band)
    g = gate.reindex(v.index).fillna(False).to_numpy(bool)
    pos = (st.astype(bool) & g).astype(np.int8)
    r = np.diff(np.log(v.to_numpy(float)))
    held = pos[:-1].astype(bool)
    trips = int((np.diff(pos.astype(int)) == 1).sum())
    hold = float(np.log(v.iloc[-1] / v.iloc[0]))
    exposure = float(held.mean())
    return {"log": float(r[held].sum()) + trips * np.log(1.0 - fee),
            "hold": hold, "exposure": exposure,
            "null": exposure * hold, "trips": trips}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--band", type=float, default=0.08)
    ap.add_argument("--min-mcap", type=float, default=1e13)
    ap.add_argument("--names", type=int, default=60)
    args = ap.parse_args()
    os.makedirs("reports", exist_ok=True)

    cfg = load_config()
    loader = YahooOHLCV(cfg, Cache(cfg.path("data.cache_dir", "data/cache")))
    mc = market_caps()
    big = sorted(mc[mc >= args.min_mcap].index)[:args.names]
    px = build_panel(loader, big)
    if px.empty:
        raise SystemExit("no panel")
    print(f"panel: {px.shape[1]} names x {px.shape[0]:,} sessions, "
          f"{px.index[0]:%Y-%m-%d} to {px.index[-1]:%Y-%m-%d}")

    G = gates(px)
    rows = []
    for name, gate in G.items():
        for t in px.columns:
            r = run(px[t], gate[t], args.band)
            if r:
                rows.append({"filter": name, "ticker": t, **r})
    R = pd.DataFrame(rows)
    R.to_csv("reports/layered.csv", index=False)

    print(f"\n{'=' * 94}\n LAYER 1 GATES ON THE LAYER 3 RULE — daily "
          f"{args.band:.0%} band, fees charged\n{'=' * 94}")
    print(f" {'filter':<12}{'median log':>12}{'vs hold':>10}{'exposure':>10}"
          f"{'same-exp null':>15}{'edge vs null':>14}{'beats null':>12}")
    hold = R[R["filter"] == "none"]["hold"].median()
    for name in FILTERS:
        s = R[R["filter"] == name]
        if not len(s):
            continue
        edge = (s["log"] - s["null"]).median()
        print(f" {name:<12}{s['log'].median():>12.3f}"
              f"{s['log'].median() - hold:>+10.3f}{s['exposure'].median():>10.0%}"
              f"{s['null'].median():>15.3f}{edge:>+14.3f}"
              f"{float((s['log'] > s['null']).mean()):>12.0%}")
    print(f" {'buy & hold':<12}{hold:>12.3f}{0.0:>+10.3f}{1.0:>10.0%}"
          f"{hold:>15.3f}{0.0:>+14.3f}{'-':>12}")

    print(f"\n{'=' * 94}\n VERDICT\n{'=' * 94}")
    best, best_edge, best_win = None, -np.inf, 0.0
    for name in FILTERS:
        s = R[R["filter"] == name]
        if not len(s):
            continue
        e = (s["log"] - s["null"]).median()
        if e > best_edge:
            best, best_edge, best_win = name, e, float((s["log"] > s["null"]).mean())
    base = R[R["filter"] == "none"]
    base_edge = (base["log"] - base["null"]).median()
    print(f" best filter: '{best}', edge over its own same-exposure null "
          f"{best_edge:+.3f} log,\n ahead on {best_win:.0%} of names. "
          f"Ungated, the same rule scores {base_edge:+.3f}.")
    if best_edge > 0 and best_win > 0.55:
        print(f"\n A regime gate DOES add something the band rule cannot get by "
              f"itself.\n This is the first thing tested in this repo that "
              f"clears the same-exposure null.")
    else:
        print(f"\n No gate clears the null. Filtering changes how much you are "
              f"exposed, and\n therefore changes the return, but it does not "
              f"beat spending that same\n exposure at random. On this "
              f"evidence layer 1 does not rescue layer 3.")
        print("\n That is a real result, not a failure to try: all three layers "
              "have now been\n measured against the same benchmark, and none "
              "of them clears it.")
    print("\n -> reports/layered.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
