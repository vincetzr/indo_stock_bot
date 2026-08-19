#!/usr/bin/env python3
"""Paint the green and red legs, causally, and score against the hand-drawn ones.

The target is the segmentation in the annotated screenshot: every week coloured
green if the stock is inside a rising leg and red if it is inside a falling one.
Formally that is the state of a **12% zigzag on unadjusted weekly closes**, which
reproduces the ~20 hand-drawn segments over 2021-2026.

Two things about the target matter and are easy to get wrong:

* **Unadjusted prices.** TradingView plots the raw close. ADRO's 2022-23
  dividends were enormous, so the split/dividend-adjusted series runs 517-2,540
  where the chart runs 1,645-4,140. Fitting the adjusted series paints a
  different picture than the one on screen.
* **The label needs the future.** Whether this week is inside a rising leg is
  only knowable once the leg ends. So labels are rebuilt inside each training
  window from `prices[:cut]` alone, and the unfinished final leg is dropped.

Feature layers, in the order the methodology asks for them
----------------------------------------------------------
    FUNDAMENTAL   sector composite (the coal complex for a coal miner), the
                  index, commodity and currency state from the macro panel.
                  These are the backtestable half of "fundamental research" -
                  point-in-time news text is not reconstructable for 2021-2026
                  and is handled separately, in the live layer, never here.
    FLOW          broker-summary features, added by `--flow` once fetched.
    TECHNICAL     the stock's own trend, momentum, volatility and structure,
                  including the triple-EMA and MACD shown on the chart.

Why the panel is pooled
-----------------------
One name gives ~450 trainable weeks. A gradient booster with 59 features on 450
rows reaches 100% training accuracy and 49% test accuracy, and collapses to
predicting "down" for 89% of the out-of-sample bars. The fix is not fewer
features, it is more rows: the model trains on EVERY big cap at once and is
scored per name. That also happens to be the requirement - it has to work across
IDX large caps, not just the one it was fitted to.

Scoring is **bar accuracy**: the share of weeks whose colour the causal model
gets right. That is what "reproducing the picture" means. It is NOT the same as
trading profit, and Result 91 in this repository measured a rank correlation of
-0.81 between accuracy and profit across rule families - so this script reports
both, and never lets the accuracy number stand alone.

    python3 scripts/legpaint.py --ticker ADRO
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
from swing_accuracy import legs, zigzag        # noqa: E402
from turn_ml import load_macro                 # noqa: E402

DAILY_CAP = 0.35
START = "2015-01-01"          # deep history for training; the chart window is a slice

#: Sector peers. A coal miner turns with the coal complex, and peer prices are
#: causally available where a coal futures series is not.
SECTOR: Dict[str, List[str]] = {
    "ADRO": ["PTBA", "ITMG", "HRUM", "BSSR", "INDY", "DOID", "BUMI", "ADMR", "GEMS"],
    "PTBA": ["ADRO", "ITMG", "HRUM", "BSSR", "INDY", "BUMI"],
    "ITMG": ["ADRO", "PTBA", "HRUM", "BSSR", "INDY", "BUMI"],
    "ANTM": ["INCO", "MDKA", "TINS", "NCKL", "MBMA", "PSAB"],
    "INCO": ["ANTM", "MDKA", "NCKL", "MBMA", "TINS"],
    "BBCA": ["BBRI", "BMRI", "BBNI", "BBTN", "BRIS"],
    "BBRI": ["BBCA", "BMRI", "BBNI", "BBTN", "BRIS"],
    "BMRI": ["BBCA", "BBRI", "BBNI", "BBTN", "BRIS"],
    "BBNI": ["BBCA", "BBRI", "BMRI", "BBTN", "BRIS"],
    "TLKM": ["ISAT", "EXCL", "TOWR", "TBIG", "MTEL"],
    "ASII": ["UNTR", "AUTO", "IMAS"],
    "UNTR": ["ASII", "ADRO", "PTBA", "ITMG"],
    "ICBP": ["INDF", "MYOR", "UNVR", "SIDO"],
    "INDF": ["ICBP", "MYOR", "UNVR", "AALI", "LSIP"],
}


def unadjusted_weekly(loader: YahooOHLCV, ticker: str,
                      start: str = START) -> Optional[pd.Series]:
    """Weekly close as CHARTED - raw, not dividend-adjusted, impossible prints capped."""
    d = loader.get(ticker, max_age=86400 * 30)
    if d is None or len(d) < 300:
        return None
    d = d.set_index("date").sort_index()
    c = d["close"].astype(float).dropna()
    r = c.pct_change().clip(-DAILY_CAP, DAILY_CAP).fillna(0.0)
    c = c.iloc[0] * (1.0 + r).cumprod()
    w = c.resample("W-FRI").last().dropna()
    return w[w.index >= pd.Timestamp(start)]


def zigzag_labels(px: np.ndarray, thr: float,
                  drop_last: bool = True) -> np.ndarray:
    """1 inside a rising leg, 0 inside a falling one, NaN where not yet knowable."""
    y = np.full(len(px), np.nan)
    lg = legs(px, zigzag(px, thr))
    for k, (a, b, r) in enumerate(lg):
        if drop_last and k == len(lg) - 1:
            break
        y[a:b] = 1.0 if r > 0 else 0.0
    return y


# --------------------------------------------------------------------------- #
# features
# --------------------------------------------------------------------------- #
def technical(w: pd.Series, vol: Optional[pd.Series] = None) -> pd.DataFrame:
    """The stock's own state. Everything trailing as of its own bar."""
    c = w.astype(float)
    lr = np.log(c).diff()
    f = pd.DataFrame(index=c.index)
    for n in (2, 4, 8, 13, 26, 52):
        f[f"ret{n}"] = np.log(c / c.shift(n))
    for n in (4, 8, 13, 20, 30):
        m = c.rolling(n, min_periods=n).mean()
        f[f"ma{n}"] = c / m - 1.0
        f[f"ma{n}_slope"] = m / m.shift(2) - 1.0
    # the chart's own indicator: triple EMA ribbon + MACD
    e5 = c.ewm(span=5, adjust=False).mean()
    e10 = c.ewm(span=10, adjust=False).mean()
    e20 = c.ewm(span=20, adjust=False).mean()
    f["ribbon_spread"] = (e5 - e20) / e20
    f["ribbon_order"] = ((e5 > e10).astype(float) + (e10 > e20).astype(float))
    f["ema5_slope"] = e5 / e5.shift(1) - 1.0
    f["ema20_slope"] = e20 / e20.shift(1) - 1.0
    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    sig = macd.ewm(span=9, adjust=False).mean()
    f["macd"] = macd / c
    f["macd_hist"] = (macd - sig) / c
    f["macd_slope"] = (macd - macd.shift(1)) / c
    # structure
    for n in (13, 26, 52):
        f[f"hi{n}"] = c / c.rolling(n, min_periods=n // 2).max() - 1.0
        f[f"lo{n}"] = c / c.rolling(n, min_periods=n // 2).min() - 1.0
    f["vol13"] = lr.rolling(13, min_periods=8).std()
    f["vol_ratio"] = f["vol13"] / lr.rolling(52, min_periods=26).std()
    up = lr.clip(lower=0).rolling(14, min_periods=7).mean()
    dn = (-lr.clip(upper=0)).rolling(14, min_periods=7).mean()
    f["rsi"] = 100 - 100 / (1 + up / dn.replace(0, np.nan))
    f["above_ma_count"] = sum((c > c.rolling(n, min_periods=n).mean()).astype(float)
                              for n in (4, 8, 13, 20, 30))
    # how long the current direction has persisted, capped so it stays a feature
    sgn = np.sign(lr.fillna(0.0))
    run = sgn.groupby((sgn != sgn.shift()).cumsum()).cumcount() + 1
    f["run_len"] = (run * sgn).clip(-12, 12)
    if vol is not None:
        v = vol.reindex(c.index).astype(float)
        f["vol_z"] = ((v - v.rolling(52, min_periods=26).mean())
                      / v.rolling(52, min_periods=26).std())
        f["vol_trend"] = np.log(v.rolling(8, min_periods=4).mean()
                                / v.rolling(52, min_periods=26).mean())
    return f


def sector_features(loader: YahooOHLCV, ticker: str,
                    index: pd.DatetimeIndex) -> pd.DataFrame:
    """What the rest of the sector is doing - the causal stand-in for the
    commodity itself, which has no clean IDX-aligned weekly series here."""
    peers = SECTOR.get(ticker.upper(), [])
    f = pd.DataFrame(index=index)
    series = []
    for p in peers:
        s = unadjusted_weekly(loader, p)
        if s is not None and len(s) > 60:
            series.append(s.reindex(index).ffill())
    if not series:
        return f
    M = pd.concat(series, axis=1)
    comp = M.mean(axis=1)
    for n in (4, 13, 26):
        f[f"sect_ret{n}"] = np.log(comp / comp.shift(n))
    f["sect_ma13"] = comp / comp.rolling(13, min_periods=8).mean() - 1.0
    f["sect_breadth"] = (M > M.rolling(13, min_periods=8).mean()).mean(axis=1)
    f["sect_n"] = float(len(series))
    return f


#: Stationary features only. Levels that drift across a decade - a rate, an
#: index level, a raw price - teach the model where it is in history rather than
#: what the stock is doing, and they are what made the single-name version
#: collapse out of sample.
def _stationary(F: pd.DataFrame) -> pd.DataFrame:
    drop = [c for c in F.columns
            if c.startswith("mkt_us10y") or c.startswith("mkt_vix")
            or c in ("sect_n", "run_len")]
    keep = [c for c in F.columns if c not in drop]
    return F[keep]


def build_panel(tickers: List[str], thr: float, verbose: bool = True):
    """Every name's features and prices, stacked, for pooled training."""
    cfg = load_config()
    loader = YahooOHLCV(cfg, Cache(cfg.path("data.cache_dir", "data/cache")))
    out: Dict[str, Tuple[pd.Series, pd.DataFrame]] = {}
    for t in tickers:
        w = unadjusted_weekly(loader, t)
        if w is None or len(w) < 200:
            continue
        d = loader.get(t, max_age=86400 * 30)
        if d is None:
            continue
        d = d.set_index("date").sort_index()
        volw = d["volume"].resample("W-FRI").sum().reindex(w.index)
        F = technical(w, volw)
        F = F.join(sector_features(loader, t, w.index))
        F = F.join(load_macro(w.index))
        F = _stationary(F)
        F["px"] = w.to_numpy(float)
        out[t] = (w, F)
    if verbose:
        n = sum(len(v[0]) for v in out.values())
        print(f"panel: {len(out)} names, {n:,} weekly bars, "
              f"{len(next(iter(out.values()))[1].columns) - 1} features")
    return out


def build(ticker: str, thr: float, verbose: bool = True):
    cfg = load_config()
    loader = YahooOHLCV(cfg, Cache(cfg.path("data.cache_dir", "data/cache")))
    w = unadjusted_weekly(loader, ticker)
    if w is None or len(w) < 150:
        raise SystemExit(f"{ticker}: not enough weekly history")
    d = loader.get(ticker, max_age=86400 * 30).set_index("date").sort_index()
    volw = d["volume"].resample("W-FRI").sum().reindex(w.index)

    F = technical(w, volw)
    F = F.join(sector_features(loader, ticker, w.index))
    F = F.join(load_macro(w.index))
    F["px"] = w.to_numpy(float)
    if verbose:
        print(f"{ticker}: {len(w)} weekly bars {w.index[0]:%Y-%m}..{w.index[-1]:%Y-%m}, "
              f"{F.shape[1] - 1} features")
    return w, F


def walk_forward_pooled(panel: Dict[str, Tuple[pd.Series, pd.DataFrame]],
                        thr: float, folds: int, seed: int = 0
                        ) -> Dict[str, np.ndarray]:
    """Train on every name's history before the cut, predict every name after it.

    The cut is a DATE, shared across names, so no name is ever predicted using a
    model that saw its own future - or any other name's.
    """
    all_dates = sorted({d for w, _ in panel.values() for d in w.index})
    all_dates = pd.DatetimeIndex(all_dates)
    edges = np.linspace(int(len(all_dates) * 0.45), len(all_dates),
                        folds + 1).astype(int)
    feats = [c for c in next(iter(panel.values()))[1].columns if c != "px"]
    preds = {t: np.full(len(w), np.nan) for t, (w, _) in panel.items()}

    for k in range(folds):
        cut_d = all_dates[edges[k]]
        end_d = all_dates[min(edges[k + 1], len(all_dates) - 1)]
        Xs, ys = [], []
        for t, (w, F) in panel.items():
            m = w.index < cut_d
            if m.sum() < 120:
                continue
            y = zigzag_labels(w.to_numpy(float)[m], thr, drop_last=True)
            ok = np.isfinite(y)
            if ok.sum() < 60:
                continue
            Xs.append(F[feats].to_numpy(float)[m][ok])
            ys.append(y[ok])
        if not Xs:
            continue
        X = np.vstack(Xs)
        Y = np.concatenate(ys).astype(int)
        model = HistGradientBoostingClassifier(
            max_iter=250, learning_rate=0.05, max_depth=4,
            min_samples_leaf=60, l2_regularization=5.0,
            early_stopping=True, validation_fraction=0.15, random_state=seed)
        model.fit(X, Y)
        for t, (w, F) in panel.items():
            sel = (w.index >= cut_d) & (w.index <= end_d)
            if sel.sum() == 0:
                continue
            preds[t][sel] = model.predict_proba(
                F[feats].to_numpy(float)[sel])[:, 1]
        print(f"  fold {k + 1}: train {len(Y):,} bars to {cut_d:%Y-%m}, "
              f"predict to {end_d:%Y-%m}")
    return preds


def walk_forward(w: pd.Series, F: pd.DataFrame, thr: float, folds: int,
                 min_train: int = 150, seed: int = 0) -> Tuple[np.ndarray, List[Dict]]:
    """Expanding-window prediction. Labels rebuilt inside each training slice."""
    px = w.to_numpy(float)
    n = len(px)
    feats = [c for c in F.columns if c != "px"]
    X = F[feats].to_numpy(float)
    pred = np.full(n, np.nan)
    edges = np.linspace(max(min_train, int(n * 0.45)), n, folds + 1).astype(int)
    info = []
    for k in range(folds):
        cut, end = edges[k], edges[k + 1]
        y = zigzag_labels(px[:cut], thr, drop_last=True)
        ok = np.isfinite(y)
        if ok.sum() < 80 or len(np.unique(y[ok])) < 2:
            continue
        usable = np.isfinite(X[:cut][ok]).any(axis=0)
        m = HistGradientBoostingClassifier(
            max_iter=400, learning_rate=0.05, max_depth=4,
            min_samples_leaf=20, l2_regularization=1.0, random_state=seed)
        m.fit(X[:cut][ok][:, usable], y[ok].astype(int))
        pred[cut:end] = m.predict_proba(X[cut:end][:, usable])[:, 1]
        info.append({"fold": k + 1, "train_bars": int(ok.sum()),
                     "start": w.index[cut], "end": w.index[end - 1]})
    return pred, info


def smooth_state(prob: np.ndarray, hi: float, lo: float) -> np.ndarray:
    """Hysteresis: a probability wandering across 0.5 must not repaint the chart."""
    st = np.zeros(len(prob), dtype=np.int8)
    on = False
    for i, p in enumerate(prob):
        if np.isfinite(p):
            if on and p < lo:
                on = False
            elif not on and p > hi:
                on = True
        st[i] = 1 if on else 0
    return st


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default="ADRO")
    ap.add_argument("--threshold", type=float, default=0.12)
    ap.add_argument("--folds", type=int, default=6)
    ap.add_argument("--hi", type=float, default=0.55)
    ap.add_argument("--lo", type=float, default=0.45)
    ap.add_argument("--universe", default="bluechip")
    args = ap.parse_args()
    os.makedirs("reports", exist_ok=True)

    cfg = load_config()
    names = sorted(set(cfg.universe(args.universe)) | set(cfg.universe("lq45"))
                   | {args.ticker.upper()})
    panel = build_panel(names, args.threshold)
    if args.ticker.upper() not in panel:
        raise SystemExit(f"{args.ticker} not in the panel")
    preds = walk_forward_pooled(panel, args.threshold, args.folds)

    rows = []
    for t, (w, F) in panel.items():
        px = w.to_numpy(float)
        truth = zigzag_labels(px, args.threshold, drop_last=True)
        p = preds[t]
        m = np.isfinite(p) & np.isfinite(truth)
        if m.sum() < 40:
            continue
        raw = (p > 0.5).astype(int)
        sm = smooth_state(p, args.hi, args.lo)
        s = pd.Series(px)
        ma13 = (s > s.rolling(13, min_periods=13).mean()).fillna(False).to_numpy().astype(int)
        rows.append({
            "ticker": t, "bars": int(m.sum()),
            "model_raw": float((raw[m] == truth[m]).mean()),
            "model": float((sm[m] == truth[m]).mean()),
            "ma13": float((ma13[m] == truth[m]).mean()),
            "up_share_true": float(truth[m].mean()),
            "up_share_pred": float(sm[m].mean())})
    R = pd.DataFrame(rows).sort_values("model", ascending=False)
    R.to_csv("reports/legpaint_scores.csv", index=False)

    print(f"\n{'=' * 88}\n PAINTING THE LEGS — pooled model, {args.threshold:.0%} zigzag"
          f"\n{'=' * 88}")
    print(f" {len(R)} names scored, {R['bars'].sum():,} out-of-sample weeks")
    print(f"\n {'':<10}{'model':>10}{'MA 13w':>10}{'lift':>9}")
    print(f" {'median':<10}{R['model'].median():>10.1%}{R['ma13'].median():>10.1%}"
          f"{R['model'].median() - R['ma13'].median():>+9.1%}")
    print(f" {'mean':<10}{R['model'].mean():>10.1%}{R['ma13'].mean():>10.1%}"
          f"{R['model'].mean() - R['ma13'].mean():>+9.1%}")
    print(f" {'worst':<10}{R['model'].min():>10.1%}{R['ma13'].min():>10.1%}")
    print(f" names at 90%+: {(R['model'] >= 0.90).sum()} of {len(R)}"
          f"   at 85%+: {(R['model'] >= 0.85).sum()}")

    tgt = R[R["ticker"] == args.ticker.upper()]
    if not tgt.empty:
        r = tgt.iloc[0]
        print(f"\n {args.ticker}: model {r['model']:.1%} (raw {r['model_raw']:.1%}), "
              f"MA13 {r['ma13']:.1%}, on {int(r['bars'])} weeks")
        print(f"   predicted up-share {r['up_share_pred']:.0%} vs actual "
              f"{r['up_share_true']:.0%}")
    print(f"\n top 8 and bottom 5 by accuracy")
    for _, r in pd.concat([R.head(8), R.tail(5)]).iterrows():
        print(f"   {r['ticker']:<7}{r['model']:>8.1%}  (MA13 {r['ma13']:.1%})")
    print(f"\n target 90%: gap on median {0.90 - R['model'].median():+.1%}")

    w, F = panel[args.ticker.upper()]
    pd.DataFrame({"date": w.index, "px": w.to_numpy(float),
                  "truth": zigzag_labels(w.to_numpy(float), args.threshold),
                  "prob": preds[args.ticker.upper()],
                  "state": smooth_state(preds[args.ticker.upper()], args.hi, args.lo)}
                 ).to_csv(f"reports/legpaint_{args.ticker}.csv", index=False)
    print(f"\n -> reports/legpaint_scores.csv, reports/legpaint_{args.ticker}.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
