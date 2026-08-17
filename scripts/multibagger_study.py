#!/usr/bin/env python3
"""What did IDX multibaggers look like *before* they ran?

A multibagger study is the single easiest place in quantitative finance to fool
yourself, so the traps are handled explicitly rather than mentioned:

* **Survivorship.** The panel is built from today's listing. Names that went to
  zero and delisted are absent, so the *base rate* of multibaggers computed here
  is too high. Every probability below is an upper bound. What survives the bias
  better is the *relative* lift of one bucket over another, measured inside the
  same biased panel, which is why nothing is reported as a bare probability.
* **Look-ahead.** Every feature is trailing-only as of its own bar. The label is
  a forward return and never an input. Features and labels are computed in one
  pass per ticker and then split, so no feature can accidentally see its label.
* **Overlap.** Adjacent dates share almost the same three-year future. Reported
  cross-sections are thinned to non-overlapping dates before any rate is
  computed, otherwise one lucky run counts hundreds of times.
* **Price, not total return, for the label.** A 3x is a 3x in price terms; using
  the dividend-adjusted series would let a high yielder qualify without moving.
  Features use the same series so nothing is mixed.

    python3 scripts/multibagger_study.py
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

from idxbot.config import load_config           # noqa: E402
from idxbot.data.cache import Cache             # noqa: E402
from idxbot.data.ohlcv import YahooOHLCV        # noqa: E402

HORIZON = 750          # ~3 years of trading days
BAGGER = 2.0           # +200% == a 3-bagger
MIN_BARS = 1200        # need history before the window even opens
MIN_TURNOVER = 1e9     # Rp1bn/day: tradeable at retail size
W = 84


def banner(t: str) -> None:
    print("\n" + "=" * W + f"\n {t}\n" + "=" * W)


def build_panel(verbose: bool = True) -> pd.DataFrame:
    cfg = load_config()
    loader = YahooOHLCV(cfg, Cache(cfg.path("data.cache_dir", "data/cache")))
    tickers = cfg.universe("idx_all")
    raw = loader.get_many(tickers, max_age=86400 * 30, verbose=False)

    frames: List[pd.DataFrame] = []
    for ticker, bars in raw.items():
        if len(bars) < MIN_BARS:
            continue
        d = bars.sort_values("date").reset_index(drop=True).copy()
        close = d["close"].astype(float)

        # ---- label: forward 3-year price return (never an input) ----
        d["fwd_3y"] = close.shift(-HORIZON) / close - 1.0

        # ---- features: trailing only, as of each bar ----
        d["turnover"] = (close * d["volume"]).rolling(20, min_periods=10).median()
        d["mom_250"] = close / close.shift(250) - 1.0
        d["mom_60"] = close / close.shift(60) - 1.0
        d["hi_250"] = close / close.rolling(250, min_periods=100).max() - 1.0
        d["hi_750"] = close / close.rolling(750, min_periods=250).max() - 1.0
        # Drawdown from the highest price seen SO FAR - expanding, not full-sample.
        d["dd_all"] = close / close.expanding(min_periods=100).max() - 1.0
        d["vol_60"] = close.pct_change().rolling(60, min_periods=30).std() * np.sqrt(252)
        d["vol_trend"] = (d["volume"].rolling(20, min_periods=10).median()
                          / d["volume"].rolling(250, min_periods=100).median())
        d["price"] = close
        d["ticker"] = ticker
        frames.append(d[["date", "ticker", "price", "turnover", "mom_250", "mom_60",
                         "hi_250", "hi_750", "dd_all", "vol_60", "vol_trend",
                         "fwd_3y"]])

    panel = pd.concat(frames, ignore_index=True)
    panel = panel.dropna(subset=["fwd_3y", "turnover", "mom_250", "hi_750",
                                 "vol_60", "dd_all"])
    panel = panel[(panel["turnover"] >= MIN_TURNOVER) & (panel["price"] >= 50)]
    if verbose:
        print(f"panel: {len(panel):,} ticker-days, {panel['ticker'].nunique()} names, "
              f"{panel['date'].min():%Y-%m} -> {panel['date'].max():%Y-%m}")
    return panel


def thin(panel: pd.DataFrame, every_days: int = 250) -> pd.DataFrame:
    """Keep dates spaced so adjacent observations do not share a future."""
    dates = pd.DatetimeIndex(sorted(panel["date"].unique()))
    kept, last = [], None
    for d in dates:
        if last is None or (d - last).days >= every_days:
            kept.append(d)
            last = d
    return panel[panel["date"].isin(kept)]


def base_rate(panel: pd.DataFrame) -> float:
    return float((panel["fwd_3y"] >= BAGGER).mean())


def feature_lift(panel: pd.DataFrame, feature: str, bins: int = 5) -> pd.DataFrame:
    """Multibagger rate by quintile of one feature, ranked *within each date*.

    Ranking within the date is what makes this a usable screen rather than a
    description of history: an absolute threshold on momentum means something
    different in 2008 and 2021, a within-date quintile does not.
    """
    rows = []
    df = panel.dropna(subset=[feature]).copy()
    df["q"] = df.groupby("date")[feature].transform(
        lambda s: pd.qcut(s, bins, labels=False, duplicates="drop")
        if s.nunique() > bins else np.nan)
    for q, g in df.dropna(subset=["q"]).groupby("q"):
        rows.append({"quintile": int(q) + 1, "n": len(g),
                     "bagger_rate": float((g["fwd_3y"] >= BAGGER).mean()),
                     "median_fwd_3y": float(g["fwd_3y"].median()),
                     "mean_fwd_3y": float(g["fwd_3y"].mean())})
    return pd.DataFrame(rows)


def main() -> int:
    os.makedirs("reports", exist_ok=True)
    panel = build_panel()
    thinned = thin(panel)
    print(f"non-overlapping cross-sections: {thinned['date'].nunique()} dates, "
          f"{len(thinned):,} observations")

    banner("BASE RATE — how often does a liquid IDX stock 3x in three years?")
    rate = base_rate(thinned)
    print(f" {rate:.1%} of ticker-observations reached +200% over 3 years")
    print(f" median 3-year return: {thinned['fwd_3y'].median():+.1%}")
    print(f" mean 3-year return  : {thinned['fwd_3y'].mean():+.1%}")
    print("\n Survivorship inflates this. Read it as a ceiling, and compare")
    print(" buckets against each other rather than against your expectations.")

    banner("WHICH TRAILING FEATURES PRECEDE A 3x?")
    features = [("mom_250", "12-month momentum"),
                ("mom_60", "3-month momentum"),
                ("hi_250", "distance below 1-year high"),
                ("hi_750", "distance below 3-year high"),
                ("dd_all", "drawdown from all-time high"),
                ("vol_60", "realised volatility"),
                ("vol_trend", "volume now vs 1-year normal"),
                ("turnover", "turnover (size proxy)"),
                ("price", "share price")]
    summary = []
    for col, label in features:
        table = feature_lift(thinned, col)
        if table.empty or len(table) < 3:
            continue
        lo = table.iloc[0]
        hi = table.iloc[-1]
        lift = hi["bagger_rate"] / lo["bagger_rate"] if lo["bagger_rate"] > 0 else np.inf
        summary.append({"feature": label, "col": col,
                        "q1_rate": lo["bagger_rate"], "q5_rate": hi["bagger_rate"],
                        "lift_q5_over_q1": lift,
                        "q1_median": lo["median_fwd_3y"],
                        "q5_median": hi["median_fwd_3y"]})
        print(f"\n {label}  (quintile 1 = lowest)")
        print(f"   {'q':>3}{'n':>8}{'3x rate':>10}{'median 3y':>12}{'mean 3y':>10}")
        for _, r in table.iterrows():
            bar = "#" * int(round(r["bagger_rate"] * 60))
            print(f"   {int(r['quintile']):>3}{int(r['n']):>8}{r['bagger_rate']:>10.1%}"
                  f"{r['median_fwd_3y']:>12.1%}{r['mean_fwd_3y']:>10.1%}  {bar}")

    s = pd.DataFrame(summary).sort_values("lift_q5_over_q1", ascending=False)
    banner("RANKED BY HOW MUCH THE TOP QUINTILE BEATS THE BOTTOM")
    print(s[["feature", "q1_rate", "q5_rate", "lift_q5_over_q1"]].to_string(
        index=False, formatters={"q1_rate": "{:.1%}".format,
                                 "q5_rate": "{:.1%}".format,
                                 "lift_q5_over_q1": "{:.2f}x".format}))
    s.to_csv("reports/multibagger_features.csv", index=False)
    thinned.to_csv("reports/multibagger_panel.csv", index=False)
    print("\n -> reports/multibagger_features.csv, reports/multibagger_panel.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
