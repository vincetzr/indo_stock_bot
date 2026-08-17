#!/usr/bin/env python3
"""Can a learned model time ADRO better than the hand-built rules?

Part IX showed no hand-built rule beat holding ADRO. That is evidence about
those rules, not about the stock, so this searches the space properly: ~50
causal features, gradient boosting, and a strictly forward-rolling protocol.

The protocol is the whole point, because ML on price series is the easiest way
in finance to produce a beautiful lie:

* **No shuffled cross-validation.** Rows are ordered in time and adjacent rows
  share overlapping futures; k-fold would train on tomorrow to predict today.
  The model is retrained on an expanding window and only ever predicts the block
  immediately after it.
* **No feature computed across the split.** Every feature is a trailing window
  as of its own bar, and scaling parameters are fitted on train and applied to
  test rather than fitted on everything.
* **The label is embargoed.** A forward ``h``-day return overlaps the next ``h``
  rows, so the last ``h`` rows before each test block are dropped from training.
  Without that the model sees its own test period through the label.
* **Predictions are traded with a one-bar lag** and charged costs.

    python3 scripts/adro_ml.py [TICKER]
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

from idxbot import timing as T                    # noqa: E402
from idxbot.config import load_config             # noqa: E402
from idxbot.data.cache import Cache               # noqa: E402
from idxbot.data.ohlcv import YahooOHLCV          # noqa: E402

HORIZON = 20           # predict the next 20 sessions
EMBARGO = HORIZON      # rows dropped between train and test
MIN_TRAIN = 750        # ~3 years before the first prediction
STEP = 120             # retrain every ~6 months
EXOG = ["^JKSE", "EEM", "HG=F", "BZ=F", "USDIDR=X", "PTBA", "ITMG", "UNTR"]


def features(bars: pd.DataFrame, exog: Dict[str, pd.Series]) -> pd.DataFrame:
    """~50 trailing features. Every one is a function of bars up to its own date."""
    d = bars.sort_values("date").reset_index(drop=True).copy()
    close, high, low, vol = d["close"], d["high"], d["low"], d["volume"]
    adj = d["adj_close"]
    f = pd.DataFrame(index=d.index)

    for n in (1, 2, 3, 5, 10, 20, 60, 120, 250):
        f[f"ret_{n}"] = close.pct_change(n)
    for n in (10, 20, 60, 120):
        f[f"vol_{n}"] = close.pct_change().rolling(n).std()
        f[f"hi_{n}"] = close / close.rolling(n).max() - 1.0
        f[f"lo_{n}"] = close / close.rolling(n).min() - 1.0
        f[f"ma_{n}"] = close / close.rolling(n).mean() - 1.0
    for n in (7, 14, 28):
        f[f"rsi_{n}"] = T.rsi(close, n)
    for n in (20, 60, 120):
        f[f"z_{n}"] = T.zscore(close, n)
    f["atr_20"] = ((high - low).rolling(20).mean() / close)
    f["range_pos"] = (close - low.rolling(20).min()) / (
        high.rolling(20).max() - low.rolling(20).min()).replace(0, np.nan)
    for n in (5, 20, 60):
        f[f"volratio_{n}"] = vol.rolling(n).mean() / vol.rolling(250).mean()
    f["dd_all"] = close / close.expanding(min_periods=100).max() - 1.0
    f["vol_ratio"] = f["vol_20"] / f["vol_120"]
    f["month"] = d["date"].dt.month
    f["dow"] = d["date"].dt.dayofweek

    for name, series in exog.items():
        aligned = series.reindex(d["date"]).to_numpy()
        s = pd.Series(aligned, index=d.index).ffill()
        key = name.replace("^", "").replace("=", "").replace(".", "")
        for n in (5, 20, 60):
            f[f"x_{key}_{n}"] = s.pct_change(n)

    f["date"] = d["date"]
    f["fwd"] = adj.shift(-HORIZON) / adj - 1.0
    return f


def walk_forward_ml(f: pd.DataFrame, bars: pd.DataFrame, model_kind: str = "gbr",
                    long_only: bool = True, threshold: float = 0.0
                    ) -> Tuple[pd.Series, Dict[str, float]]:
    """Expanding-window retrain, predict the next block, never look back."""
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    cols = [c for c in f.columns if c not in ("date", "fwd")]
    X_all = f[cols].to_numpy(float)
    y_all = f["fwd"].to_numpy(float)
    n = len(f)
    pred = np.full(n, np.nan)

    start = MIN_TRAIN
    while start < n:
        stop = min(start + STEP, n)
        tr_end = start - EMBARGO                    # embargo the overlapping label
        if tr_end < 250:
            start = stop
            continue
        tr = slice(0, tr_end)
        ok = np.isfinite(X_all[tr]).all(axis=1) & np.isfinite(y_all[tr])
        if ok.sum() < 200:
            start = stop
            continue
        Xtr, ytr = X_all[tr][ok], y_all[tr][ok]

        scaler = StandardScaler().fit(Xtr)          # fitted on TRAIN only
        if model_kind == "gbr":
            model = GradientBoostingRegressor(n_estimators=200, max_depth=3,
                                              learning_rate=0.03, subsample=0.8,
                                              random_state=0)
        elif model_kind == "rf":
            model = RandomForestRegressor(n_estimators=300, max_depth=6,
                                          min_samples_leaf=20, random_state=0,
                                          n_jobs=-1)
        else:
            model = Ridge(alpha=10.0)
        model.fit(scaler.transform(Xtr), ytr)

        te = slice(start, stop)
        Xte = X_all[te]
        good = np.isfinite(Xte).all(axis=1)
        if good.any():
            out = np.full(stop - start, np.nan)
            out[good] = model.predict(scaler.transform(Xte[good]))
            pred[te] = out
        start = stop

    signal = pd.Series(pred, index=f.index)
    if long_only:
        position = (signal > threshold).astype(float)
    else:
        position = np.sign(signal - threshold)
    position = pd.Series(position, index=f.index).fillna(0.0)

    live = f["date"].notna() & signal.notna()
    first = int(np.argmax(live.to_numpy())) if live.any() else 0
    sub = bars.iloc[first:].reset_index(drop=True)
    pos = position.iloc[first:].reset_index(drop=True)
    result = T.backtest(sub, pos, label=f"ML {model_kind}")
    hold = T.buy_and_hold(sub)
    stats = dict(result.stats)
    stats["buy_hold_cagr"] = hold.stats["cagr"]
    stats["excess"] = stats["cagr"] - hold.stats["cagr"]
    stats["ic"] = float(pd.Series(signal[live]).corr(
        pd.Series(f["fwd"][live]), method="spearman"))
    stats["coverage"] = float(live.mean())
    return position, stats


def main() -> int:
    ticker = (sys.argv[1] if len(sys.argv) > 1 else "ADRO").upper()
    cfg = load_config()
    loader = YahooOHLCV(cfg, Cache(cfg.path("data.cache_dir", "data/cache")))
    bars = loader.get(ticker, max_age=86400 * 7).sort_values("date").reset_index(drop=True)

    exog: Dict[str, pd.Series] = {}
    for sym in EXOG:
        try:
            e = loader.get(sym, max_age=86400 * 30)
            if e is not None and len(e) > 500:
                exog[sym] = e.set_index("date")["adj_close"]
        except Exception:
            continue

    f = features(bars, exog)
    print(f"{ticker}: {len(f):,} bars, {len([c for c in f.columns if c not in ('date','fwd')])} "
          f"features, {len(exog)} exogenous series")
    print(f"protocol: expanding retrain every {STEP} bars, {EMBARGO}-bar embargo, "
          f"{HORIZON}-day target\n")

    print(f" {'model':<24}{'CAGR':>9}{'buy&hold':>11}{'excess':>9}{'maxDD':>8}"
          f"{'in mkt':>8}{'IC':>8}")
    best = None
    for kind in ("ridge", "rf", "gbr"):
        for long_only in (True, False):
            pos, s = walk_forward_ml(f, bars, kind, long_only=long_only)
            label = f"{kind}{' long-only' if long_only else ' long/short'}"
            print(f" {label:<24}{s['cagr']:>+9.1%}{s['buy_hold_cagr']:>+11.1%}"
                  f"{s['excess']:>+9.1%}{s['max_drawdown']:>8.0%}"
                  f"{s['time_in_market']:>8.0%}{s['ic']:>+8.3f}")
            if best is None or s["excess"] > best[0]:
                best = (s["excess"], label, s)
    print(f"\n best: {best[1]} with {best[0]:+.1%} excess over buy-and-hold")
    print(f" out-of-sample rank IC of the model's forecast: {best[2]['ic']:+.3f}")
    print("\n An IC near zero means the model found nothing, whatever the CAGR")
    print(" column happens to say on one particular path.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
