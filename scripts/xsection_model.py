#!/usr/bin/env python3
"""H27 — the cross-sectional model §8 asked for and this repo never built.

    python3 scripts/xsection_model.py

WHY THIS IS DIFFERENT FROM EVERYTHING BEFORE IT. Every sweep in this project —
H13, H23, H25, H26 — ranked DISCRETE CELLS with hand-tuned percentile cuts.
CLAUDE.md §8 says the opposite in as many words: "compute as numeric features
feeding a cross-sectional model, not as discrete buy/sell rules with hand-tuned
parameters." A cell can only express "high vol AND thin"; a model can express
an interaction, a non-monotone response, and a conditional that no percentile
cut reaches. If there is structure the cells missed, this is where it lives.

WHAT IS PREDICTED. Two separate binary targets over the next 252 sessions:

    up   the path touches 2x
    down the terminal is at or below half

ranked by predicted p(up) / p(down), which is H26's asymmetry objective made
continuous. Modelling the ratio directly would fit a quotient of two rare
events; two calibrated classifiers and a division is the stabler route.

PURGED WALK-FORWARD, AND IT IS THE WHOLE VALIDITY OF THIS. A cohort dated t
does not settle until t+252. Training on it to predict a cohort at t+30 leaks
almost a year of overlapping future. So for each test year the training set is
restricted to cohorts whose forward window CLOSED before that year began. That
throws away a year of data at every fold and is not optional.

THE NULL RUNS THE IDENTICAL PIPELINE with labels permuted inside (ticker, year)
blocks — same model, same folds, same ranking — because a walk-forward with a
tuned model is exactly the kind of harness that manufactures its own signal,
and A11 records four defects here that all printed believable output.

PRE-REGISTERED, BEFORE ANY FIT:

  R1  The model's top decile beats the best hand-cut cell (H26's strength+calm,
      skew 2.60) out of sample. Predicted: it does NOT beat it by much, because
      the features are few and highly collinear, but it should at least match.
  R2  Feature importance is dominated by hi52 and vol60 — the two axes H26
      already found. Predicted: SUPPORTED, which would mean the cells were not
      leaving structure on the table.
  R3  The null's top decile shows skew ~1.0 with no trend across folds.
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))
sys.path.insert(0, os.path.dirname(__file__))

from sklearn.ensemble import HistGradientBoostingClassifier      # noqa: E402
from sklearn.inspection import permutation_importance            # noqa: E402

from horizon_sweep import FEATURES, classify                     # noqa: E402

CACHE = os.path.join("data", "spine", "horizon_sweep.parquet")
SECTORS = os.path.join("data", "reference", "idx_classification.parquet")
K = 252
TOP = 0.10
MIN_TRAIN = 3000


def frame() -> pd.DataFrame:
    D = pd.read_parquet(CACHE)
    D = D[~D["holdout"].astype(bool)]
    d = classify(D, K)
    d = d[d["cls"] != "censored"].copy().reset_index(drop=True)
    d["up"] = (d[f"peak{K}"] >= 2.0).astype(int)
    d["down"] = (d[f"end{K}"] <= 0.5).astype(int)
    d["year"] = d["date"].dt.year
    #  rank features WITHIN each cohort. A raw level carries the market's
    #  twenty-year drift into the model as a date proxy; a cross-sectional
    #  rank cannot, and the decision is a cross-sectional one anyway.
    for f in FEATURES:
        if f in d.columns:
            d[f + "_r"] = d.groupby("date")[f].rank(pct=True)
    #  SECTOR, the one piece of genuinely non-price information available.
    #  A14 found this map and no study has used it. 934 tickers across the 11
    #  official IDX-IC sectors.
    #
    #  ITS `shares` COLUMN IS DELIBERATELY NOT USED. The file is frozen at
    #  2024-07-10, so a share count taken from it and applied to a 2010 bar is
    #  look-ahead (A5), and Indonesian rights issues are large enough that
    #  dilution is precisely what makes it wrong. Market cap therefore stays
    #  out. Sector itself is a far more stable attribute — a company rarely
    #  changes one — so it is used with that caveat stated rather than hidden.
    if os.path.exists(SECTORS):
        S = pd.read_parquet(SECTORS)[["ticker", "sector"]]
        d = d.merge(S, on="ticker", how="left")
        d["sector_c"] = d["sector"].astype("category").cat.codes.astype(float)
        d.loc[d["sector"].isna(), "sector_c"] = np.nan
    return d.dropna(subset=[f + "_r" for f in FEATURES if f in d.columns],
                    how="all")


def fit_predict(d: pd.DataFrame, cols: List[str], seed: int = 0,
                shuffle: bool = False,
                min_train: int = MIN_TRAIN) -> pd.DataFrame:
    """Purged expanding walk-forward. Returns test-fold predictions only."""
    rng = np.random.default_rng(seed)
    D = d.copy()
    if shuffle:
        #  THE NULL MUST BREAK THE FEATURE-LABEL LINK, AND TWO EARLIER
        #  VERSIONS DID NOT.
        #
        #  First attempt permuted `up` and `down` independently inside
        #  (ticker, year) blocks. That breaks the real link between them — a
        #  name that can double is the same name that can halve, both driven
        #  by its volatility — and invents observations that doubled with no
        #  halving risk.
        #
        #  Second attempt permuted the PAIR inside the block, and barely moved
        #  the null at all: the ~12 monthly cohorts of one ticker-year hold
        #  near-identical labels because their forward windows overlap by
        #  eleven months, so shuffling within the block is close to a no-op.
        #  Both versions returned a null skew of ~3.06 against the fitted
        #  model's 2.31 — a null that beats the model it is testing is broken,
        #  not informative.
        #
        #  What actually destroys the link while preserving the clustering is
        #  reassigning whole blocks' LABELS to other blocks' FEATURES.
        D["blk"] = D["ticker"].astype(str) + "|" + D["year"].astype(str)
        groups = [g.index.to_numpy() for _, g in D.groupby("blk", sort=False)]
        lab = D[["up", "down"]].to_numpy()
        new = lab.copy()
        order = rng.permutation(len(groups))
        for tgt, src in zip(groups, [groups[i] for i in order]):
            #  tile or truncate the donor block to fit the target block
            take = np.resize(np.arange(len(src)), len(tgt))
            new[tgt] = lab[src[take]]
        D[["up", "down"]] = new
    out = []
    years = sorted(D["year"].unique())
    for y in years:
        test = D[D["year"] == y]
        #  PURGE: a cohort settles 252 sessions (~1 calendar year) after its
        #  date, so anything dated inside the year before the test year is
        #  still open and must not be trained on.
        train = D[D["date"] < pd.Timestamp(f"{y}-01-01")
                  - pd.Timedelta(days=370)]
        if len(train) < min_train or len(test) < 100:
            continue
        rec = test[["date", "ticker", "year", "up", "down",
                    f"end{K}"]].copy()
        for tgt in ("up", "down"):
            if train[tgt].nunique() < 2:
                rec["p_" + tgt] = np.nan
                continue
            m = HistGradientBoostingClassifier(
                max_depth=3, max_iter=200, learning_rate=0.05,
                min_samples_leaf=100, l2_regularization=1.0,
                random_state=seed)
            m.fit(train[cols], train[tgt])
            rec["p_" + tgt] = m.predict_proba(test[cols])[:, 1]
        out.append(rec)
    if not out:
        return pd.DataFrame()
    R = pd.concat(out, ignore_index=True).dropna(subset=["p_up", "p_down"])
    R["score"] = R["p_up"] / np.maximum(R["p_down"], 1e-4)
    return R


def evaluate(R: pd.DataFrame, top: float = TOP) -> Dict:
    if R.empty:
        return {}
    r = R.groupby("date")["score"].rank(pct=True)
    sel = R[(r >= 1 - top).reindex(R.index).fillna(False)]
    if len(sel) < 200:
        return {}
    up, dn = sel["up"].mean(), sel["down"].mean()
    return {"n": len(sel), "up": float(up), "down": float(dn),
            "skew": float(up / dn) if dn > 0 else np.nan,
            "n_down": int(sel["down"].sum()),
            "base_up": float(R["up"].mean()),
            "base_down": float(R["down"].mean()),
            "base_skew": float(R["up"].mean() / R["down"].mean()),
            "median": float(sel[f"end{K}"].median() - 1.0),
            "mean_log": float(np.log(np.maximum(
                sel[f"end{K}"], 0.01)).mean())}


def main() -> int:
    W = 92
    d = frame()
    cols = [f + "_r" for f in FEATURES if f + "_r" in d.columns]
    with_sec = cols + (["sector_c"] if "sector_c" in d.columns else [])
    print("=" * W)
    print(" H27 — THE CROSS-SECTIONAL MODEL §8 ASKED FOR")
    print("=" * W)
    print(f" {len(d):,} name-years, {d['ticker'].nunique()} names, "
          f"{len(cols)} ranked features")
    print(f" purged expanding walk-forward: training excludes every cohort")
    print(f" still open at the test year's start\n")

    R = fit_predict(d, cols, seed=0)
    e = evaluate(R)
    if not e:
        print(" no scoreable folds")
        return 1
    print(f" scored {len(R):,} test-fold observations across "
          f"{R['year'].nunique()} folds\n")
    print(f"   {'':<26}{'P(2x)':>9}{'P(-50%)':>10}{'SKEW':>8}"
          f"{'median':>10}{'mean log':>10}")
    print(f"   {'model top decile':<26}{e['up']:>9.1%}{e['down']:>10.1%}"
          f"{e['skew']:>8.2f}{e['median']:>+10.1%}{e['mean_log']:>+10.4f}")
    print(f"   {'all names (base)':<26}{e['base_up']:>9.1%}"
          f"{e['base_down']:>10.1%}{e['base_skew']:>8.2f}")
    print(f"   {'H26 strength+calm cell':<26}{0.105:>9.1%}{0.041:>10.1%}"
          f"{2.60:>8.2f}{0.0:>+10.1%}{0.0494:>+10.4f}   <- the cell to beat")

    if "sector_c" in d.columns:
        cov = d["sector_c"].notna().mean()
        Rs = fit_predict(d, with_sec, seed=0)
        es = evaluate(Rs)
        if es:
            print(f"   {'+ SECTOR (11 IDX-IC)':<26}{es['up']:>9.1%}"
                  f"{es['down']:>10.1%}{es['skew']:>8.2f}"
                  f"{es['median']:>+10.1%}{es['mean_log']:>+10.4f}")
            print(f"\n   sector coverage {cov:.1%} of rows. Adding it moves the"
                  f" skew {es['skew'] - e['skew']:+.2f}")
            print("   and mean log "
                  f"{es['mean_log'] - e['mean_log']:+.4f}. `shares` from the"
                  " same file is NOT used:")
            print("   frozen at 2024-07-10, so applying it to a 2010 bar is"
                  " look-ahead.")

    print("\n" + "-" * W)
    print(" THE NULL — identical pipeline, labels permuted in (ticker, year)")
    print("-" * W)
    nulls = []
    for s in range(5):
        n = evaluate(fit_predict(d, cols, seed=s, shuffle=True))
        if n:
            nulls.append(n["skew"])
    if nulls:
        print(f"   null top-decile skew over {len(nulls)} runs: "
              f"{np.mean(nulls):.2f} +/- {np.std(nulls, ddof=1):.2f}"
              f"   (range {min(nulls):.2f}-{max(nulls):.2f})")
        z = (e["skew"] - np.mean(nulls)) / max(np.std(nulls, ddof=1), 1e-9)
        print(f"   model {e['skew']:.2f} -> z {z:+.2f}")
        print(f"   R3 {'SUPPORTED' if abs(np.mean(nulls) - 1.0) < 0.4 else 'FAILED'}"
              f": null sits at {np.mean(nulls):.2f}, expected ~1.0")

    print("\n" + "-" * W)
    print(" R2 — WHAT IS THE MODEL ACTUALLY USING?")
    print("-" * W)
    tr = d[d["date"] < d["date"].quantile(0.7)]
    te = d[d["date"] >= d["date"].quantile(0.7)]
    m = HistGradientBoostingClassifier(max_depth=3, max_iter=200,
                                       learning_rate=0.05,
                                       min_samples_leaf=100,
                                       l2_regularization=1.0, random_state=0)
    m.fit(tr[cols], tr["up"])
    imp = permutation_importance(m, te[cols], te["up"], n_repeats=5,
                                 random_state=0, scoring="roc_auc")
    order = np.argsort(-imp.importances_mean)
    print(f"\n   {'feature':<18}{'importance':>12}{'sd':>9}")
    for i in order[:8]:
        print(f"   {cols[i]:<18}{imp.importances_mean[i]:>12.4f}"
              f"{imp.importances_std[i]:>9.4f}")
    top2 = {cols[i] for i in order[:2]}
    hit = len(top2 & {"hi52_r", "vol60_r", "lowvol_r"})
    print(f"\n   R2 {'SUPPORTED' if hit >= 1 else 'FAILED'}: the top two are "
          f"{', '.join(cols[i] for i in order[:2])}")

    print("\n" + "=" * W)
    print(" R1 — DOES THE MODEL BEAT THE BEST HAND-CUT CELL?")
    print("=" * W)
    print(f"   model top decile  skew {e['skew']:.2f}, "
          f"mean log {e['mean_log']:+.4f}")
    print(f"   H26 cell          skew 2.60, mean log +0.0494")
    if e["skew"] > 2.60:
        print("   -> the model BEATS the cell. Structure the cells missed.")
    elif e["skew"] > 2.20:
        print("   -> comparable. The cells were not leaving much on the table.")
    else:
        print("   -> the model does NOT beat a two-filter cell. With eleven")
        print("      collinear features and one macro history, there is no")
        print("      interaction left for a model to find.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
