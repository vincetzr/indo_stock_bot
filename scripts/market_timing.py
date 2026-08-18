#!/usr/bin/env python3
"""Time the market, not the stock.

Every turn-detection attempt so far failed for a reason that is now specific
rather than mysterious. Result 77: a 20-day gate applied to thirty large caps
independently flips each name about twenty times a year and costs roughly ten
points a year in fees. The gate was not wrong about direction - it was applied
thirty times over when the names it gated are one highly-correlated basket.

A market-level call fixes exactly that:

* **One decision, not thirty.** The whole book moves together, so the same
  number of correct calls costs a thirtieth of the fees.
* **Better signal to noise.** An index averages away the single-name noise that
  made per-name turns unlearnable (Result 81: 60% accuracy against a 62%
  break-even).
* **Information that does not exist for one stock.** Breadth - how much of the
  market is above its own trend - is a property of the panel, not of a name, and
  it is the classic regime indicator. Plus the macro panel: VIX, US 10-year,
  dollar, oil, copper, the rupiah.

And the circles on the annotated chart are mostly market events anyway - 2008,
2011, 2015, 2020, 2025. If they are callable at all, they are callable here.

What is tested
--------------
    always on          own it, no timing - the benchmark
    above 30w / 40w MA the classic index trend filter, nearly free to run
    breadth            in when most of the market is above its own trend
    learned            gradient boosting on IHSG state + breadth + macro,
                       labels rebuilt inside each training window

Each is scored twice: on the index itself, and as an overlay on the blue-chip
book from Part XIX. The second is what matters, because the book is the thing
that would actually be traded.

A warning that belongs in the output, not a footnote: the index makes roughly
one major turn every two years, so the whole sample contains a few dozen
decisions. That is a small-sample problem no amount of walk-forward fixes, and
it is why the simple rules are reported beside the learned one.

    python3 scripts/market_timing.py [--threshold 0.15] [--folds 4]
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

from sklearn.ensemble import HistGradientBoostingClassifier   # noqa: E402

from idxbot.config import load_config          # noqa: E402
from idxbot.data.cache import Cache            # noqa: E402
from idxbot.data.ohlcv import YahooOHLCV       # noqa: E402
from bluechip import pit_universe              # noqa: E402
from optimize_consistent import load_wide, score_curve        # noqa: E402
from swing_accuracy import zigzag, legs        # noqa: E402
from turn_book import simulate                 # noqa: E402
from turn_ml import load_macro, name_features, zigzag_state   # noqa: E402
from turn_trader import ROUND_TRIP, capture, clean_weekly, run  # noqa: E402

INDEX = "^JKSE"


def breadth(W: Dict) -> pd.DataFrame:
    """Regime features that only exist for a panel, never for one name.

    All trailing: the share of names above their own 200-day average, the share
    near their own 250-day high, the median drawdown, and the cross-sectional
    dispersion of returns. Breadth rolling over while the index still rises is
    the classic tell that a top is forming, and it is untestable on a single
    series by construction.
    """
    close = W["close"]
    ma200 = close.rolling(200, min_periods=120).mean()
    hi250 = close.rolling(250, min_periods=150).max()
    r60 = close / close.shift(60) - 1.0
    live = close.notna()
    n = live.sum(axis=1).replace(0, np.nan)
    out = pd.DataFrame(index=close.index)
    out["br_ma200"] = (close > ma200).sum(axis=1) / n
    out["br_nearhi"] = ((close / hi250 - 1.0) > -0.05).sum(axis=1) / n
    out["br_dd"] = (close / hi250 - 1.0).median(axis=1)
    out["br_disp"] = r60.std(axis=1)
    out["br_up60"] = (r60 > 0).sum(axis=1) / n
    return out


def build(threshold: float, verbose: bool = True):
    cfg = load_config()
    loader = YahooOHLCV(cfg, Cache(cfg.path("data.cache_dir", "data/cache")))
    idx = loader.get(INDEX, max_age=86400 * 30)
    if idx is None or idx.empty:
        raise SystemExit(f"no data for {INDEX}")
    w = clean_weekly(idx)
    if w is None:
        raise SystemExit("index history too short")

    W = load_wide(verbose=False)
    br = breadth(W).resample("W-FRI").last()

    f = name_features(w)
    f = f.join(br.reindex(f.index, method="ffill"))
    macro = load_macro(f.index)
    f = f.join(macro)
    f["px"] = w.to_numpy(float)
    f["date"] = f.index
    if verbose:
        print(f"index: {len(w)} weekly bars, {w.index[0]:%Y-%m-%d} -> "
              f"{w.index[-1]:%Y-%m-%d}; {f.shape[1] - 2} features")
        piv = zigzag(w.to_numpy(float), threshold)
        lg = legs(w.to_numpy(float), piv)
        yrs = (w.index[-1] - w.index[0]).days / 365.25
        print(f"IHSG makes {len(lg)} swings of {threshold:.0%}+ in {yrs:.0f} years "
              f"- one decision every {yrs * 12 / max(len(lg), 1):.0f} months")
    return w, f, W


# --------------------------------------------------------------------------- #
# the timers
# --------------------------------------------------------------------------- #
def timer_ma(w: pd.Series, weeks: int) -> np.ndarray:
    ma = w.rolling(weeks, min_periods=weeks).mean()
    return (w > ma).fillna(False).to_numpy().astype(np.int8)


def timer_breadth(f: pd.DataFrame, col: str, level: float) -> np.ndarray:
    return (f[col] > level).fillna(False).to_numpy().astype(np.int8)


def timer_model(f: pd.DataFrame, w: pd.Series, threshold: float, folds: int,
                p_hi: float, p_lo: float) -> Tuple[np.ndarray, List[Dict]]:
    """Walk-forward learned timer. Labels rebuilt inside each training window."""
    from turn_ml import positions
    px = w.to_numpy(float)
    n = len(f)
    edges = np.linspace(int(n * 0.45), n, folds + 1).astype(int)
    prob = np.full(n, np.nan)
    info = []
    feats = [c for c in f.columns if c not in ("px", "date")]
    for k in range(folds):
        cut, end = edges[k], edges[k + 1]
        y = zigzag_state(px[:cut], threshold, drop_last_leg=True)
        tr = f.iloc[:cut][feats].to_numpy(float)
        ok = np.isfinite(y)
        if ok.sum() < 200 or len(np.unique(y[ok])) < 2:
            continue
        # A feature that is entirely missing inside this training window - a
        # macro series that had not started yet, say - carries no information
        # and breaks the binner. Selecting on the TRAINING slice only, and
        # applying the same selection to the test slice, keeps the two aligned.
        usable = np.isfinite(tr[ok]).any(axis=0)
        if not usable.any():
            continue
        cols = [c for c, u in zip(feats, usable) if u]
        m = HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.05, max_depth=3,
            min_samples_leaf=40, l2_regularization=2.0, random_state=0)
        m.fit(tr[ok][:, usable], y[ok].astype(int))
        prob[cut:end] = m.predict_proba(
            f.iloc[cut:end][cols].to_numpy(float))[:, 1]
        info.append({"fold": k + 1, "start": f.index[cut], "end": f.index[end - 1],
                     "train_rows": int(ok.sum())})
    st = positions(prob, p_hi, p_lo)
    st[:edges[0]] = 1                       # before the first fold: no opinion, hold
    return st, info


def score_timer(w: pd.Series, state: np.ndarray, start: int,
                threshold: float) -> Dict[str, float]:
    px = w.to_numpy(float)[start:]
    st = state[start:]
    eq, trades = run(px, st)
    yrs = (w.index[-1] - w.index[start]).days / 365.25
    bh = px[-1] / px[0]
    lg = legs(px, zigzag(px, threshold))
    cap = capture(px, st, lg) if lg else {}
    peak = np.maximum.accumulate(eq)
    return {"growth": float(eq[-1]), "cagr": float(eq[-1]) ** (1 / yrs) - 1,
            "bh_growth": float(bh), "bh_cagr": bh ** (1 / yrs) - 1,
            "trades": trades, "time_in": float(st.mean()),
            "max_dd": float((eq / peak - 1).min()),
            "bh_dd": float((px / np.maximum.accumulate(px) - 1).min()),
            "acc": cap.get("direction_acc", np.nan),
            "capture": cap.get("up_fraction", np.nan)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=0.15)
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--p-hi", type=float, default=0.60)
    ap.add_argument("--p-lo", type=float, default=0.40)
    ap.add_argument("--universe-size", type=int, default=30)
    args = ap.parse_args()
    os.makedirs("reports", exist_ok=True)

    w, f, W = build(args.threshold)
    n = len(w)
    start = int(n * 0.45)          # everything is scored on the same window

    timers: Dict[str, np.ndarray] = {
        "always on": np.ones(n, dtype=np.int8),
        "above 30w MA": timer_ma(w, 30),
        "above 40w MA": timer_ma(w, 40),
        "breadth > 50%": timer_breadth(f, "br_ma200", 0.50),
        "breadth > 40%": timer_breadth(f, "br_ma200", 0.40),
    }
    model_state, info = timer_model(f, w, args.threshold, args.folds,
                                    args.p_hi, args.p_lo)
    timers["learned"] = model_state
    timers["learned AND 40w"] = (model_state & timers["above 40w MA"]).astype(np.int8)
    timers["learned OR 40w"] = (model_state | timers["above 40w MA"]).astype(np.int8)

    print(f"\n{'=' * 104}\n TIMING THE INDEX ITSELF — scored from "
          f"{w.index[start]:%Y-%m}\n{'=' * 104}")
    print(f" {'timer':<20}{'growth':>10}{'CAGR':>9}{'vs hold':>10}{'trades':>8}"
          f"{'in mkt':>8}{'maxDD':>8}{'turn acc':>10}{'capture':>9}")
    rows = []
    for name, st in timers.items():
        s = score_timer(w, st, start, args.threshold)
        rows.append({"timer": name, **s})
        print(f" {name:<20}{s['growth']:>9,.2f}x{s['cagr']:>+9.1%}"
              f"{s['cagr'] - s['bh_cagr']:>+10.1%}{s['trades']:>8.0f}"
              f"{s['time_in']:>8.0%}{s['max_dd']:>8.0%}"
              f"{s['acc']:>10.0%}{s['capture']:>9.0%}")
    R = pd.DataFrame(rows)
    R.to_csv("reports/market_timing_index.csv", index=False)
    print(f"\n buy and hold: {R['bh_growth'].iloc[0]:,.2f}x "
          f"({R['bh_cagr'].iloc[0]:+.1%}/yr), deepest drawdown "
          f"{R['bh_dd'].iloc[0]:.0%}")

    # ---------------- as an overlay on the blue-chip book ---------------- #
    print(f"\n{'=' * 104}\n THE SAME CALL, APPLIED TO THE BLUE-CHIP BOOK\n{'=' * 104}")
    pit = pit_universe(W, args.universe_size)
    daily = W["mark"].index
    book_rows = []
    for name, st in timers.items():
        # weekly state -> daily gate, held until the next weekly bar
        s = pd.Series(st, index=w.index).reindex(daily, method="ffill").fillna(1)
        gate = np.repeat(s.to_numpy().astype(np.int8)[:, None],
                         pit.shape[1], axis=1)
        eq, trades = simulate(W, (pit & gate).astype(np.int8),
                              lookback=250, top_n=12, rebalance=60)
        lo = int(len(daily) * 0.45)
        sc = score_curve(eq.iloc[lo:] / eq.iloc[lo], trades)
        book_rows.append({"timer": name, **sc})
        print(f" {name:<20}{sc['growth']:>9,.1f}x{sc['cagr']:>+9.1%}"
              f"{sc['median_year']:>+10.1%}{sc['worst_year']:>+10.1%}"
              f"{sc['pct_positive']:>8.0%}{sc['max_dd']:>8.0%}{sc['ulcer']:>7.2f}")
    B = pd.DataFrame(book_rows)
    B.to_csv("reports/market_timing_book.csv", index=False)

    base = B[B["timer"] == "always on"].iloc[0]
    best = B.sort_values("cagr", ascending=False).iloc[0]
    print(f"\n header: growth / CAGR / median yr / worst yr / +yrs / maxDD / ulcer")
    print(f"\n best timer on the book: {best['timer']} at {best['cagr']:+.1%} "
          f"against {base['cagr']:+.1%} always on "
          f"({best['cagr'] - base['cagr']:+.1%}), drawdown "
          f"{best['max_dd']:.0%} against {base['max_dd']:.0%}")
    print(f"\n {len(info)} learned folds trained on "
          f"{[i['train_rows'] for i in info]} labelled weeks")
    print("\n -> reports/market_timing_index.csv, reports/market_timing_book.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
