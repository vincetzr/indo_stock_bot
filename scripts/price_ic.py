#!/usr/bin/env python3
"""H13: do §8's price/TA features predict IDX returns, post-cost?

THE SAME TEST AS GATE 1, ON PURPOSE
-------------------------------------
Every statistic here is the one `flow_ic.py` ran for H9 — cross-sectional
Spearman IC per period, control-neutralised by per-period OLS, HAC t on the IC
series, a permutation null through the identical pipeline, IC by liquidity
decile, and costs charged to the quintile spread. Reusing the machinery
verbatim is the point: it makes the price result directly comparable with the
flow result, and it means the pipeline was debugged on a question whose answer
is already known.

WHAT IS DIFFERENT, AND WHY IT MATTERS
---------------------------------------
Resolution and size. H9 had 176 names on a fortnightly grid, ~21,700 rows.
This has ~1,000 names daily over twenty years — two orders of magnitude more
observations, and it can speak to k = 1 and k = 3, which the flow panel could
never reach. If §8's families carry anything, the failure to find it here would
not be for want of sample.

THE SELF-CONTROL PROBLEM, STATED RATHER THAN FUDGED
-----------------------------------------------------
Two of the eight tested features ARE controls: `mom12_1` is itself a control,
and `lowvol` is minus `vol60`. Regressing a feature on itself leaves a residual
of zero and an IC of exactly nothing. So a feature that is a control is dropped
from its OWN control set and neutralised on the remainder. This is noted in the
output for every affected row, because it means those two are held to a
slightly weaker standard than the other six.

COSTS
------
An IC is unit-free, so cost cannot be subtracted from it. What cost applies to
is the QUINTILE SPREAD, which is a return. A5's schedule is 0.56% a round trip;
on top of that goes the point-in-time fraksi-harga half-spread, twice, and on a
small cap that is the larger of the two. Both are charged.

THE HOLDOUT IS NOT TOUCHED
---------------------------
Every frame here is filtered to `~holdout` before anything is computed. §11
reserves the most recent 24 months to be spent once, at the end, and this run
is not that.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))
sys.path.insert(0, os.path.dirname(__file__))

from idxbot.features.price import (CONTROLS, PREDICTED,          # noqa: E402
                                   SELF_CONTROL)
from idxbot.spine.reference import half_spread                   # noqa: E402
from flow_ic import neutralise, newey_west_t, spearman           # noqa: E402

PANEL = os.path.join("data", "spine", "price_panel.parquet")
FEE_ROUND_TRIP = 0.0056
HORIZONS = (1, 5, 10, 20)
CONFIRM_K = 5              # pre-specified in H13, not chosen after the fact
N_QUINTILE = 5
MIN_NAMES = 30             # a cross-section smaller than this is not one


def ic_by_day(D: pd.DataFrame, feature: str, label: str,
              controls: Sequence[str]) -> pd.DataFrame:
    """One IC per day, on the control-neutralised feature."""
    cols = [c for c in controls if c in D.columns]
    out = []
    for day, g in D.groupby("date", sort=True):
        if len(g) < MIN_NAMES:
            continue
        y = g[feature].to_numpy(dtype=float)
        r = g[label].to_numpy(dtype=float)
        raw = spearman(y, r)
        if cols:
            y = neutralise(y, g[cols].to_numpy(dtype=float))
        out.append({"date": day, "n": int(np.isfinite(y).sum()),
                    "ic_raw": raw, "ic": spearman(y, r)})
    return pd.DataFrame(out)


def controls_for(feature: str) -> List[str]:
    """A feature that IS a control is dropped from its own control set."""
    drop = SELF_CONTROL.get(feature)
    return [c for c in CONTROLS if c != drop]


def quintile_spread(D: pd.DataFrame, feature: str, label: str,
                    k: int) -> Dict[str, float]:
    """Top-minus-bottom quintile return per rebalance, gross and net.

    Costs: a long-short quintile portfolio turns over both legs every k days,
    so one round trip per leg per holding period. The fee is A5's 0.56% and the
    spread is the point-in-time half-spread at each name's own price, charged
    twice (in and out) on each leg.
    """
    rows = []
    for day, g in D.groupby("date", sort=True):
        g = g[np.isfinite(g[feature]) & np.isfinite(g[label])]
        if len(g) < MIN_NAMES:
            continue
        q = pd.qcut(g[feature].rank(method="first"), N_QUINTILE,
                    labels=False, duplicates="drop")
        if q.isna().all() or q.nunique() < N_QUINTILE:
            continue
        top = g.loc[q == N_QUINTILE - 1, label].mean()
        bot = g.loc[q == 0, label].mean()
        hs = g.apply(lambda r: half_spread_safe(r["close"], r["date"]), axis=1)
        rows.append({"date": day, "gross": float(top - bot),
                     "half_spread": float(hs.mean())})
    if not rows:
        return {}
    S = pd.DataFrame(rows)
    # per rebalance: both legs pay fee + 2x half-spread
    cost = FEE_ROUND_TRIP + 2.0 * S["half_spread"].mean()
    mu, se, t = newey_west_t(S["gross"].to_numpy(), lags=max(k, 1))
    ann = 252.0 / k
    return {"n_rebalances": float(len(S)),
            "gross_per_period": mu, "t": t,
            "cost_per_period": float(cost),
            "net_per_period": float(mu - cost),
            "gross_annual": float(mu * ann),
            "net_annual": float((mu - cost) * ann)}


def half_spread_safe(price: float, day) -> float:
    try:
        return float(half_spread(float(price), day)) / float(price)
    except Exception:                                           # noqa: BLE001
        return np.nan


def permutation_null(D: pd.DataFrame, feature: str, label: str,
                     controls: Sequence[str], draws: int,
                     seed: int) -> np.ndarray:
    """Shuffle the feature WITHIN each day, then rerun the whole pipeline.

    Preserves every day's cross-section of the feature and of the return, and
    destroys only which name's feature met which name's forward return. The
    guards and the neutralisation run inside the loop, not once outside it —
    H11 records why that distinction is not cosmetic.
    """
    rng = np.random.default_rng(seed)
    out = np.full(draws, np.nan)
    for i in range(draws):
        N = D.copy()
        N[feature] = N.groupby("date")[feature].transform(
            lambda s: rng.permutation(s.to_numpy()))
        s = ic_by_day(N, feature, label, controls)
        if len(s):
            out[i] = float(s["ic"].mean())
    return out


def liquidity_deciles(D: pd.DataFrame, feature: str, label: str,
                      controls: Sequence[str], n: int = 5) -> pd.Series:
    """IC within each liquidity stratum. §7: an effect that lives only in the
    bottom deciles is likely untradeable and must be said to be."""
    d = D.copy()
    d["_q"] = d.groupby("date")["log_turnover"].transform(
        lambda s: pd.qcut(s.rank(method="first"), n, labels=False,
                          duplicates="drop"))
    out = {}
    for q, g in d.groupby("_q"):
        s = ic_by_day(g, feature, label, controls)
        if len(s) > 20:
            out[int(q)] = float(s["ic"].mean())
    return pd.Series(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default=PANEL)
    ap.add_argument("--draws", type=int, default=100)
    ap.add_argument("--seed", type=int, default=20260823)
    ap.add_argument("--features", default=",".join(PREDICTED))
    a = ap.parse_args()

    D = pd.read_parquet(a.panel)
    D = D[~D["holdout"] & D["tradeable"]].copy()
    print("=" * 78)
    print(" H13 — §8 PRICE/TA FEATURES, cross-sectional IC (Gate 1's test)")
    print("=" * 78)
    print(f"   {D.ticker.nunique():,} names, {D.date.nunique():,} days, "
          f"{len(D):,} tradeable in-sample rows")
    print(f"   {D.date.min().date()} … {D.date.max().date()}   "
          f"(holdout after this date is untouched)")
    print(f"   confirmatory horizon pre-specified at k={CONFIRM_K}; "
          f"the full decay curve is printed for every feature\n")

    feats = [f for f in a.features.split(",") if f.strip()]
    summary = []
    for f in feats:
        if f not in D.columns:
            print(f" {f}: not in panel\n")
            continue
        ctrl = controls_for(f)
        note = ""
        if f in SELF_CONTROL:
            note = f"  [dropped {SELF_CONTROL[f]} from its own controls]"
        pred = PREDICTED[f]
        pname = {1: "+", -1: "-", 0: "0 (negative control)"}[pred]
        print(f" {f}   predicted {pname}{note}")
        curve = []
        for k in HORIZONS:
            lab = f"fwd{k}"
            s = ic_by_day(D, f, lab, ctrl)
            if not len(s):
                continue
            mu, se, t = newey_west_t(s["ic"].to_numpy(), lags=k)
            curve.append((k, mu, t, len(s)))
        print("     decay curve   " + "   ".join(
            f"k={k}: IC {mu:+.4f} (t {t:+.2f})" for k, mu, t, _ in curve))

        # confirmatory horizon
        lab = f"fwd{CONFIRM_K}"
        s = ic_by_day(D, f, lab, ctrl)
        mu, se, t = newey_west_t(s["ic"].to_numpy(), lags=CONFIRM_K)
        nulls = permutation_null(D, f, lab, ctrl, a.draws, a.seed)
        v = nulls[np.isfinite(nulls)]
        p = (float((np.abs(v - np.mean(v)) >= abs(mu - np.mean(v))).sum() + 1)
             / (len(v) + 1)) if len(v) else np.nan
        lo, hi = (np.percentile(v, [2.5, 97.5]) if len(v) > 20
                  else (np.nan, np.nan))
        dec = liquidity_deciles(D, f, lab, ctrl)
        sign_ok = (len(dec) > 1 and
                   (np.sign(dec).nunique() == 1))
        qs = quintile_spread(D, f, lab, CONFIRM_K)
        print(f"     k={CONFIRM_K}   IC {mu:+.4f}  HAC t {t:+.2f}  "
              f"days {len(s):,}")
        print(f"            null {np.mean(v):+.4f} [{lo:+.4f}, {hi:+.4f}] "
              f"over {len(v)} draws   p {p:.3f}")
        print(f"            IC by liquidity quintile "
              + " ".join(f"Q{q+1} {x:+.3f}" for q, x in dec.items())
              + ("   SIGN STABLE" if sign_ok else "   SIGN FLIPS"))
        if qs:
            print(f"            quintile spread  gross "
                  f"{qs['gross_per_period']*100:+.3f}%/period "
                  f"(t {qs['t']:+.2f})   cost {qs['cost_per_period']*100:.3f}%"
                  f"   NET {qs['net_per_period']*100:+.3f}%"
                  f"   net annual {qs['net_annual']*100:+.1f}%")
        summary.append({"feature": f, "predicted": pred, "ic": mu, "t": t,
                        "p": p, "sign_stable": sign_ok,
                        "net_annual": qs.get("net_annual", np.nan)})
        print()

    S = pd.DataFrame(summary)
    print("=" * 78)
    print(" SUMMARY — Gate 1's conjunction applied to each feature")
    print("=" * 78)
    print(f" Bonferroni bar with 38 trials: p < 0.0013\n")
    for _, r in S.iterrows():
        signed = (np.sign(r["ic"]) == r["predicted"]) if r["predicted"] else None
        marks = [
            "sig" if r["p"] < 0.0013 else "NOT sig",
            "sign stable" if r["sign_stable"] else "sign FLIPS",
            "net>0" if r["net_annual"] > 0 else "net<=0",
        ]
        d = ("as predicted" if signed else
             "AGAINST prediction" if signed is False else "n/a (null control)")
        print(f"  {r['feature']:<11} IC {r['ic']:+.4f}  p {r['p']:.3f}  "
              f"net/yr {r['net_annual']*100:+6.1f}%   "
              f"{' | '.join(marks)}   {d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
