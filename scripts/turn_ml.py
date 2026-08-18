#!/usr/bin/env python3
"""Learn the turn instead of asserting it.

Every attempt so far to trade the circled turns shared two limitations: it used
only the name's own price, and it used a rule *I* chose - a moving average, a
percentage reversal band, a volatility band. Result 70 and Result 73 killed the
rule family from both directions, with the hindsight-best setting still at
-0.12%/yr and -0.33%/yr. That is evidence about those rules, not about whether
the turns are learnable.

This asks the question properly.

The target
----------
The label is the **hindsight zigzag state**: 1 while the market is in an up-leg
between a trough and the next peak, 0 while it is in a down-leg. That is exactly
what the circles on the chart are - not a forward return, but "which side of the
turn are we on". A model that predicts it well *is* the thing that was asked for.

Why this is not cheating
------------------------
The label needs the future, so the protocol has to be airtight or it will
produce a beautiful lie:

* **Labels are rebuilt inside each training window.** The zigzag is recomputed on
  ``prices[:train_end]`` alone, never on the full series. A pivot that is only
  identifiable using post-cut data does not exist as far as training is concerned.
* **The final incomplete leg is dropped.** At the cut the last leg has not
  finished, so its label is unknown at the time and those rows are embargoed.
* **Features are trailing windows as of their own bar**, per name, plus market
  state that is also as-of-date.
* **No shuffled cross-validation.** Training is strictly before testing, always.
* **Predictions are traded with a one-bar lag** and charged 0.6% per round trip.

What is measured
----------------
Not accuracy alone - Result 69 showed a rule can be 94% directionally right and
lose money. The report gives **capture fraction** (what share of each up leg was
banked), flips, and net CAGR against buy-and-hold, which is the only number that
settles it. Break-even for a turn caller on this data is about 62%.

Hysteresis is available because it addresses the specific failure that killed the
moving average: enter above ``--p-hi``, exit below ``--p-lo``, so a probability
hovering near the boundary does not churn the position.

    python3 scripts/turn_ml.py [--folds 4] [--p-hi 0.6 --p-lo 0.4]
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
from swing_accuracy import zigzag, legs        # noqa: E402
from turn_trader import (DAILY_CAP, MIN_TURNOVER, ROUND_TRIP,  # noqa: E402
                         capture, clean_weekly, run)

MIN_WEEKS = 200
MACRO_PATH = "reports/macro_features.csv"


# --------------------------------------------------------------------------- #
# labels
# --------------------------------------------------------------------------- #
def zigzag_state(prices: np.ndarray, threshold: float,
                 drop_last_leg: bool = True) -> np.ndarray:
    """1 while in an up-leg, 0 while in a down-leg, NaN where unknowable.

    The final leg is unfinished at the right-hand edge of whatever series it is
    given, so its direction is not yet determined. Marking it NaN rather than
    guessing is what keeps a training window from learning the answer to its own
    last few months.
    """
    out = np.full(len(prices), np.nan)
    piv = zigzag(prices, threshold)
    if len(piv) < 2:
        return out
    lg = legs(prices, piv)
    for k, (a, b, r) in enumerate(lg):
        if drop_last_leg and k == len(lg) - 1:
            break
        out[a:b] = 1.0 if r > 0 else 0.0
    return out


# --------------------------------------------------------------------------- #
# features
# --------------------------------------------------------------------------- #
def rsi(x: pd.Series, n: int = 14) -> pd.Series:
    d = x.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100.0 - 100.0 / (1.0 + up / dn.replace(0, np.nan))


def name_features(w: pd.Series, tv: Optional[pd.Series] = None) -> pd.DataFrame:
    """Everything the model sees about one name, all trailing as of each bar."""
    c = w.astype(float)
    lr = np.log(c).diff()
    f = pd.DataFrame(index=c.index)
    for n in (4, 13, 26, 52):
        f[f"r{n}"] = np.log(c / c.shift(n))
    f["accel"] = f["r4"] - f["r13"] / 3.0
    for n in (52, 156):
        f[f"hi{n}"] = c / c.rolling(n, min_periods=n // 2).max() - 1.0
        f[f"lo{n}"] = c / c.rolling(n, min_periods=n // 2).min() - 1.0
    f["vol13"] = lr.rolling(13, min_periods=8).std()
    f["vol52"] = lr.rolling(52, min_periods=26).std()
    f["vol_ratio"] = f["vol13"] / f["vol52"]
    f["rsi14"] = rsi(c, 14)
    for n in (10, 30, 40):
        f[f"ma{n}"] = c / c.rolling(n, min_periods=n // 2).mean() - 1.0
    runmax = c.rolling(156, min_periods=52).max()
    f["dd"] = c / runmax - 1.0
    # weeks since the running high was set: a long time below it is a different
    # state from a fresh pullback, and the zigzag treats them differently
    f["weeks_since_high"] = (c.rolling(156, min_periods=52)
                             .apply(lambda v: len(v) - 1 - int(np.argmax(v)), raw=True))
    if tv is not None:
        t = tv.astype(float)
        f["tv_trend"] = np.log(t.rolling(13, min_periods=8).median()
                               / t.rolling(52, min_periods=26).median())
    return f


def load_macro(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Market state, aligned to the weekly grid with no forward peeking."""
    if not os.path.exists(MACRO_PATH):
        return pd.DataFrame(index=index)
    m = pd.read_csv(MACRO_PATH, parse_dates=["date"]).set_index("date").sort_index()
    m = m[~m.index.duplicated(keep="last")]
    # reindex with ffill: each weekly bar sees the most recent macro print at or
    # before it, never a later one
    return m.reindex(index.union(m.index)).ffill().reindex(index).add_prefix("mkt_")


def build_panel(verbose: bool = True) -> Tuple[pd.DataFrame, Dict[str, pd.Series]]:
    cfg = load_config()
    loader = YahooOHLCV(cfg, Cache(cfg.path("data.cache_dir", "data/cache")))
    names = sorted(set(cfg.universe("idx_all")) | set(cfg.universe("bluechip"))
                   | set(cfg.universe("lq45")) | set(cfg.universe("conglomerate")))
    raw = loader.get_many(names, max_age=86400 * 30, verbose=False)

    weeks: Dict[str, pd.Series] = {}
    frames: List[pd.DataFrame] = []
    for t, d in raw.items():
        if len(d) < 500:
            continue
        c = d["close"].astype(float)
        if float((c * d["volume"]).median()) < MIN_TURNOVER:
            continue
        w = clean_weekly(d)
        if w is None or len(w) < MIN_WEEKS:
            continue
        x = d.set_index("date").sort_index()
        tvw = (x["close"].astype(float) * x["volume"]).resample("W-FRI").sum()
        f = name_features(w, tvw.reindex(w.index))
        f["ticker"] = t
        f["date"] = w.index
        f["px"] = w.to_numpy(float)
        weeks[t] = w
        frames.append(f)

    panel = pd.concat(frames, ignore_index=True)
    dates = pd.DatetimeIndex(sorted(panel["date"].unique()))
    macro = load_macro(dates)
    panel = panel.merge(macro.reset_index().rename(columns={"index": "date"}),
                        on="date", how="left")
    # cross-sectional position within the same week: a name down 20% in a week
    # everything fell is not the same signal as one that fell alone
    for col in ("r13", "hi52"):
        panel[f"x_{col}"] = panel.groupby("date")[col].rank(pct=True)
    if verbose:
        print(f"panel: {len(panel):,} weekly rows, {panel['ticker'].nunique()} names, "
              f"{panel['date'].min():%Y-%m} -> {panel['date'].max():%Y-%m}")
    return panel, weeks


FEATURES_EXCLUDE = {"ticker", "date", "px", "y"}


def fit_predict(train: pd.DataFrame, test: pd.DataFrame,
                seed: int = 0) -> np.ndarray:
    feats = [c for c in train.columns if c not in FEATURES_EXCLUDE]
    tr = train.dropna(subset=["y"])
    if len(tr) < 500 or tr["y"].nunique() < 2:
        return np.full(len(test), np.nan)
    model = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.06, max_depth=5,
        min_samples_leaf=100, l2_regularization=1.0, random_state=seed)
    model.fit(tr[feats].to_numpy(float), tr["y"].to_numpy(int))
    return model.predict_proba(test[feats].to_numpy(float))[:, 1]


def positions(prob: np.ndarray, p_hi: float, p_lo: float) -> np.ndarray:
    """Hysteresis: enter above ``p_hi``, exit below ``p_lo``, otherwise hold.

    This is the direct fix for the failure in Result 69 - a signal hovering at
    the boundary flipped 569 times and paid 170% of capital in fees.
    """
    st = np.zeros(len(prob), dtype=np.int8)
    long = False
    for i, p in enumerate(prob):
        if np.isfinite(p):
            if long and p < p_lo:
                long = False
            elif not long and p > p_hi:
                long = True
        st[i] = 1 if long else 0
    return st


def pick_hysteresis(val: pd.DataFrame, grid) -> Tuple[float, float]:
    """Choose the entry/exit probabilities on a validation slice of the TRAINING
    window, so the pair applied out of sample was never scored on it."""
    best, best_v = (0.6, 0.4), -np.inf
    for hi, lo in grid:
        tot = []
        for _t, g in val.groupby("ticker"):
            g = g.sort_values("date")
            px = g["px"].to_numpy(float)
            if len(px) < 30 or not np.isfinite(g["prob"]).all():
                continue
            st = positions(g["prob"].to_numpy(float), hi, lo)
            eq, _ = run(px, st)
            tot.append(float(eq[-1]) / max(px[-1] / px[0], 1e-9))
        if tot and float(np.median(tot)) > best_v:
            best, best_v = (hi, lo), float(np.median(tot))
    return best


def portfolio(test: pd.DataFrame, top_n: int, every: int,
              cost: float = ROUND_TRIP, col: str = "prob") -> Tuple[float, float, int]:
    """Hold the ``top_n`` names with the highest P(up-leg), refreshed every
    ``every`` weeks. Returns (growth, years, rebalances).

    This is the cross-sectional use of the same model. A turn caller that is
    only 67% right on any single name can still be useful if the names it is
    most confident about are, on average, the ones in an up-leg - which is a
    weaker claim than calling every turn correctly.
    """
    dates = pd.DatetimeIndex(sorted(test["date"].unique()))
    wide_p = test.pivot_table(index="date", columns="ticker", values=col)
    wide_x = test.pivot_table(index="date", columns="ticker", values="px")
    # No extra clip here. ``clean_weekly`` already capped DAILY moves at the
    # +/-35% auto-rejection band before resampling, and five capped days compound
    # to +348%, so a second clip on the weekly series truncates legitimate moves
    # rather than corrupt ones.
    ret = wide_x.pct_change()
    equity, held, rebals = 1.0, [], 0
    for i in range(1, len(dates)):
        if (i - 1) % every == 0:
            p = wide_p.iloc[i - 1].dropna()
            new = list(p.nlargest(top_n).index) if len(p) else []
            if set(new) != set(held):
                churn = len(set(new) ^ set(held)) / max(2 * max(len(new), 1), 1)
                equity *= (1.0 - cost * churn)
                rebals += 1
            held = new
        if held:
            r = ret.iloc[i].reindex(held).dropna()
            if len(r):
                equity *= (1.0 + float(r.mean()))
    yrs = (dates[-1] - dates[0]).days / 365.25
    return equity, yrs, rebals


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=0.20)
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--p-hi", type=float, default=0.60)
    ap.add_argument("--p-lo", type=float, default=0.40)
    args = ap.parse_args()
    os.makedirs("reports", exist_ok=True)

    panel, weeks = build_panel()
    dates = pd.DatetimeIndex(sorted(panel["date"].unique()))
    edges = np.linspace(int(len(dates) * 0.45), len(dates), args.folds + 1).astype(int)

    print(f"\n{'=' * 104}\n LEARNING THE TURN — {args.folds} folds, labels rebuilt "
          f"inside each training window\n{'=' * 104}")
    rows, pf, bench = [], [], []
    for k in range(args.folds):
        cut, end = dates[edges[k]], dates[min(edges[k + 1], len(dates) - 1)]
        # --- labels: zigzag recomputed on the training slice ONLY ---
        lab = {}
        for t, w in weeks.items():
            tr_px = w[w.index < cut].to_numpy(float)
            if len(tr_px) < 60:
                continue
            s = zigzag_state(tr_px, args.threshold, drop_last_leg=True)
            lab[t] = pd.Series(s, index=w.index[w.index < cut])
        ylab = pd.concat([v.rename("y").to_frame().assign(ticker=t).reset_index()
                          .rename(columns={"index": "date"})
                          for t, v in lab.items()], ignore_index=True)
        train = (panel[panel["date"] < cut]
                 .merge(ylab, on=["ticker", "date"], how="left"))
        test = panel[(panel["date"] >= cut) & (panel["date"] <= end)].copy()

        # A validation slice at the END of the training window: the model is
        # refit on the part before it, so the hysteresis pair is chosen on data
        # the final model has seen but the test window has not.
        vcut = train["date"].quantile(0.80)
        inner = train[train["date"] < vcut]
        val = train[train["date"] >= vcut].copy()
        val["prob"] = fit_predict(inner, val)
        if np.isfinite(val["prob"]).any():
            p_hi, p_lo = pick_hysteresis(
                val, [(0.55, 0.45), (0.60, 0.40), (0.65, 0.35),
                      (0.70, 0.30), (0.50, 0.50), (0.75, 0.25)])
        else:
            p_hi, p_lo = args.p_hi, args.p_lo

        prob = fit_predict(train, test)
        test["prob"] = prob

        per = []
        for t, g in test.groupby("ticker"):
            g = g.sort_values("date")
            px = g["px"].to_numpy(float)
            if len(px) < 30 or not np.isfinite(g["prob"]).all():
                continue
            st = positions(g["prob"].to_numpy(float), p_hi, p_lo)
            eq, trades = run(px, st)
            yrs = (g["date"].iloc[-1] - g["date"].iloc[0]).days / 365.25
            if yrs <= 0.5:
                continue
            bh = px[-1] / px[0]
            lg = legs(px, zigzag(px, args.threshold))
            cap = capture(px, st, lg) if lg else {}
            per.append({
                "fold": k + 1, "ticker": t,
                "cagr": float(eq[-1]) ** (1 / yrs) - 1,
                "bh_cagr": bh ** (1 / yrs) - 1,
                "trades": trades, "time_in": float(st.mean()),
                "acc": cap.get("direction_acc", np.nan),
                "up_frac": cap.get("up_fraction", np.nan),
                "flips": cap.get("flips", np.nan)})
        P = pd.DataFrame(per)
        if P.empty:
            print(f" fold {k+1}: no testable names")
            continue
        P["excess"] = P["cagr"] - P["bh_cagr"]
        P["p_hi"], P["p_lo"] = p_hi, p_lo
        rows.append(P)
        print(f" fold {k+1}  {cut:%Y-%m}..{end:%Y-%m}  {len(P)} names   "
              f"median excess {P['excess'].median():+.2%}/yr   "
              f"beats B&H {(P['excess'] > 0).mean():.0%}   "
              f"turn accuracy {P['acc'].median():.0%}   "
              f"capture {P['up_frac'].median():.0%}   "
              f"(enter>{p_hi:.2f}, exit<{p_lo:.2f})")

        # the same probability used cross-sectionally instead of as a switch
        for tn in (5, 10, 20):
            for ev in (1, 4, 13):
                g, yrs, nb = portfolio(test, tn, ev)
                pf.append({"fold": k + 1, "signal": "model", "top_n": tn, "every": ev,
                           "cagr": g ** (1 / yrs) - 1 if yrs > 0 else np.nan,
                           "rebalances": nb})
        # CONTROLS. The model's features include momentum, so beating equal
        # weight proves nothing unless it also beats the momentum it was fed.
        for ctrl in ("r13", "r26", "r52"):
            for tn in (10, 20):
                for ev in (4, 13):
                    g, yrs, nb = portfolio(test, tn, ev, col=ctrl)
                    pf.append({"fold": k + 1, "signal": ctrl, "top_n": tn,
                               "every": ev,
                               "cagr": g ** (1 / yrs) - 1 if yrs > 0 else np.nan,
                               "rebalances": nb})
        ew = test.pivot_table(index="date", columns="ticker", values="px")
        ewr = ew.pct_change().mean(axis=1)
        yrs = (test["date"].max() - test["date"].min()).days / 365.25
        bench.append({"fold": k + 1,
                      "cagr": float((1 + ewr.fillna(0)).prod()) ** (1 / yrs) - 1})

    if not rows:
        raise SystemExit("nothing to report")
    A = pd.concat(rows, ignore_index=True)
    A.to_csv("reports/turn_ml.csv", index=False)

    print(f"\n{'=' * 104}\n ALL FOLDS POOLED\n{'=' * 104}")
    print(f" names scored            {len(A):,}")
    print(f" median excess over B&H  {A['excess'].median():+.2%}/yr")
    print(f" mean excess over B&H    {A['excess'].mean():+.2%}/yr")
    print(f" beats buy & hold        {(A['excess'] > 0).mean():.0%} of names")
    print(f" median turn accuracy    {A['acc'].median():.0%}   "
          f"(break-even against holding is about 62%)")
    print(f" median capture of an up leg  {A['up_frac'].median():.0%}   "
          f"(the moving average managed 31%)")
    print(f" median time in market   {A['time_in'].median():.0%}, "
          f"{A['trades'].median():.0f} trades, {A['flips'].median():.0f} flips")

    print(f"\n{'=' * 104}\n THE COMPARISON THAT MATTERS\n{'=' * 104}")
    print(f" {'method':<44}{'median excess/yr':>20}{'beats B&H':>12}{'capture':>10}")
    print(f" {'20-day moving average (Result 69)':<44}{'-5.5%':>20}{'':>12}{'31%':>10}")
    print(f" {'reversal band, tuned honestly (Result 70)':<44}{'-1.92%':>20}"
          f"{'35%':>12}{'68%':>10}")
    print(f" {'reversal band, best with hindsight':<44}{'-0.12%':>20}{'':>12}{'':>10}")
    print(f" {'volatility-scaled band (Result 73)':<44}{'-0.80%':>20}{'41%':>12}{'':>10}")
    print(f" {'learned turn model (this)':<44}{A['excess'].median():>+20.2%}"
          f"{(A['excess'] > 0).mean():>12.0%}{A['up_frac'].median():>10.0%}")
    PF = pd.DataFrame(pf)
    BM = pd.DataFrame(bench)
    if not PF.empty:
        PF.to_csv("reports/turn_ml_portfolio.csv", index=False)
        print(f"\n{'=' * 104}\n THE SAME MODEL USED TO RANK, NOT TO SWITCH\n{'=' * 104}")
        print(" hold the N names with the highest P(up-leg), refreshed every E weeks")
        print(f"\n {'signal':<8}{'topN':>5}{'every':>8}"
              + "".join(f"{f'fold {k+1}':>11}" for k in range(args.folds))
              + f"{'mean':>10}{'worst':>10}")
        for (sig, tn, ev), g in PF.groupby(["signal", "top_n", "every"]):
            cs = [float(g[g["fold"] == k + 1]["cagr"].iloc[0])
                  if (g["fold"] == k + 1).any() else np.nan
                  for k in range(args.folds)]
            print(f" {sig:<8}{tn:>5}{ev:>8}" + "".join(f"{c:>+11.1%}" for c in cs)
                  + f"{np.nanmean(cs):>+10.1%}{np.nanmin(cs):>+10.1%}")
        bs = [float(BM[BM["fold"] == k + 1]["cagr"].iloc[0])
              if (BM["fold"] == k + 1).any() else np.nan for k in range(args.folds)]
        print(f" {'equal weight, all names':<21}" + "".join(f"{c:>+11.1%}" for c in bs)
              + f"{np.nanmean(bs):>+10.1%}{np.nanmin(bs):>+10.1%}")
    print("\n -> reports/turn_ml.csv, reports/turn_ml_portfolio.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
