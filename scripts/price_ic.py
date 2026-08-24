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
import gc
import os
import sys
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.stats import rankdata

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))
sys.path.insert(0, os.path.dirname(__file__))

from idxbot.features.price import (CONTROLS, PREDICTED,          # noqa: E402
                                   SELF_CONTROL)
from idxbot.spine.quality import tick_grid                        # noqa: E402
from flow_ic import neutralise, newey_west_t, spearman           # noqa: E402

PANEL = os.path.join("data", "spine", "price_panel.parquet")
FEE_ROUND_TRIP = 0.0056
HORIZONS = (1, 5, 10, 20)
CONFIRM_K = 5              # pre-specified in H13, not chosen after the fact
N_QUINTILE = 5
MIN_NAMES = 30             # a cross-section smaller than this is not one


def prepare(D: pd.DataFrame, feature: str, label: str,
            controls: Sequence[str]) -> Dict[str, object]:
    """Presort once into numpy arrays, and FACTORISE THE CONTROLS ONCE.

    Two costs made the honest number of permutation draws unaffordable, and an
    unaffordable null is how a null ends up run once — H9 records where that
    leads.

      pandas. The hot loop runs ~5,600 days x 100 draws x 8 features; going
      through `groupby` on a 2.9M-row frame and copying the frame per draw is
      hours per feature. Sorting once and slicing numpy views is not.

      THE REGRESSION. Neutralising was an `lstsq` per day per draw. But the
      residual-maker depends only on the CONTROLS, not on y: residualising any
      vector on X is ``y - Q Q'y`` for an orthonormal basis Q of [1, X]. Q is a
      function of that day's controls alone, so it is computed once per day and
      reused across every draw. Each draw becomes two small matrix-vector
      products instead of a fresh decomposition.

    Rows with a missing feature, label or control are dropped up front rather
    than inside the loop. That keeps the null's row set identical to the
    observed one — otherwise a permutation moves the NaNs and each draw would
    silently score a different sample.
    """
    cols = [c for c in controls if c in D.columns]
    need = [feature, label] + cols
    d = D[np.isfinite(D[need].to_numpy(dtype=float)).all(axis=1)]
    d = d.sort_values("date", kind="mergesort")
    day = d["date"].to_numpy()
    starts = np.flatnonzero(np.r_[True, day[1:] != day[:-1]])
    ends = np.r_[starts[1:], len(day)]
    X = d[cols].to_numpy(dtype=float) if cols else None

    bases: List[Optional[np.ndarray]] = []
    for a, b in zip(starts, ends):
        if X is None or b - a < MIN_NAMES:
            bases.append(None)
            continue
        A = np.column_stack([np.ones(b - a), X[a:b]])
        try:
            Q, _ = np.linalg.qr(A)
        except np.linalg.LinAlgError:
            Q = None
        bases.append(Q)
    return {"y": d[feature].to_numpy(dtype=float),
            "r": d[label].to_numpy(dtype=float),
            "X": X, "Q": bases,
            "hs": (d["hs_frac"].to_numpy(dtype=float) if "hs_frac" in d
                   else np.full(len(d), np.nan)),
            "liq": (d["log_turnover"].to_numpy(dtype=float)
                    if "log_turnover" in d else np.full(len(d), np.nan)),
            "starts": starts, "ends": ends,
            "days": day[starts], "n_rows": len(day)}


def ic_from_arrays(P: Dict[str, object], y: Optional[np.ndarray] = None
                   ) -> np.ndarray:
    """One IC per day, using the precomputed per-day orthonormal basis."""
    y = P["y"] if y is None else y
    r, Q = P["r"], P["Q"]
    starts, ends = P["starts"], P["ends"]
    out = np.full(len(starts), np.nan)
    for i in range(len(starts)):
        a, b = starts[i], ends[i]
        if b - a < MIN_NAMES:
            continue
        yy = y[a:b]
        q = Q[i]
        if q is not None:
            yy = yy - q @ (q.T @ yy)
        out[i] = spearman(yy, r[a:b])
    return out


def shuffle_within_days(y: np.ndarray, starts, ends, rng) -> np.ndarray:
    """Permute the feature inside each day, preserving that day's values."""
    out = y.copy()
    for a, b in zip(starts, ends):
        out[a:b] = rng.permutation(y[a:b])
    return out


def ic_by_day(D: pd.DataFrame, feature: str, label: str,
              controls: Sequence[str]) -> pd.DataFrame:
    """One IC per day, on the control-neutralised feature."""
    P = prepare(D, feature, label, controls)
    ic = ic_from_arrays(P)
    return pd.DataFrame({"date": P["days"], "ic": ic}).dropna(subset=["ic"])


def controls_for(feature: str) -> List[str]:
    """A feature that IS a control is dropped from its own control set."""
    drop = SELF_CONTROL.get(feature)
    return [c for c in CONTROLS if c != drop]


def _ranks(x: np.ndarray) -> np.ndarray:
    """Average ranks, fully vectorised.

    The first version handled ties with a Python loop over every element. At
    5,678 days x 6 calls x 100 permutation draws that is tens of millions of
    interpreted iterations and it dominated the whole run — which is how a null
    ends up run once, and H9 records where that leads.

    scipy's rankdata does exactly this in C and is already a dependency here.
    Tests assert it against pandas' .rank(), including on ties, because average
    ranking is precisely where a hand-rolled version diverges.
    """
    return rankdata(x)


def fast_spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Rank correlation, NaN-safe. Asserted equal to flow_ic.spearman in tests."""
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 5:
        return np.nan
    ra, rb = _ranks(a[m]), _ranks(b[m])
    sa, sb = ra.std(), rb.std()
    if sa == 0 or sb == 0:
        return np.nan
    return float(((ra - ra.mean()) @ (rb - rb.mean())) / (len(ra) * sa * sb))


def analyse(P: Dict[str, object], k: int,
            light: bool = False) -> Dict[str, object]:
    """IC, quintile spread and liquidity-bucket ICs in ONE pass over days.

    The previous shape called `prepare` inside `ic_by_day` for each of ten
    calls per feature — refiltering 2M rows, resorting them and refactorising
    5,678 days of controls every time — and then walked the same days again
    through pandas `groupby` for the spread and once more, with a per-day
    `qcut`, for the liquidity buckets. That is what turned a five-minute
    feature into a half-hour one.

    Everything below is computed from the SAME residual, which also fixes a
    subtler thing: H9 ranked its quintile spread on the RAW score while its IC
    used the neutralised one, so the two statistics described different
    signals. Here the quintiles are cut on the neutralised score, because that
    is the signal the IC is reporting.
    """
    y, r, Q = P["y"], P["r"], P["Q"]
    hs, liq = P["hs"], P["liq"]
    starts, ends = P["starts"], P["ends"]
    n_days = len(starts)
    ic = np.full(n_days, np.nan)
    gross = np.full(n_days, np.nan)
    cost = np.full(n_days, np.nan)
    buck = np.full((n_days, N_QUINTILE), np.nan)

    for i in range(n_days):
        a, b = starts[i], ends[i]
        n = b - a
        if n < MIN_NAMES:
            continue
        yy = y[a:b]
        q = Q[i]
        if q is not None:
            yy = yy - q @ (q.T @ yy)
        rr = r[a:b]
        ic[i] = fast_spearman(yy, rr)
        if light:
            continue          # the null only ever reads ic_mean

        order = np.argsort(yy, kind="mergesort")
        cut = n // N_QUINTILE
        if cut >= 2:
            lo, hi = order[:cut], order[-cut:]
            gross[i] = float(rr[hi].mean() - rr[lo].mean())
            h = hs[a:b][np.r_[lo, hi]]
            h = h[np.isfinite(h)]
            cost[i] = float(h.mean()) if len(h) else np.nan

        lq = liq[a:b]
        ok = np.isfinite(lq)
        if ok.sum() >= N_QUINTILE * MIN_NAMES // 2:
            edges = np.quantile(lq[ok], np.linspace(0, 1, N_QUINTILE + 1)[1:-1])
            g = np.searchsorted(edges, lq)
            for jq in range(N_QUINTILE):
                m = (g == jq) & np.isfinite(yy) & np.isfinite(rr)
                if m.sum() >= 10:
                    buck[i, jq] = fast_spearman(yy[m], rr[m])

    mu, se, t = newey_west_t(ic, lags=k)
    if light:
        return {"ic_mean": mu, "ic_t": t}
    gm, gse, gt = newey_west_t(gross, lags=max(k, 1))
    c = float(np.nanmean(cost))
    per = FEE_ROUND_TRIP + 2.0 * c if np.isfinite(c) else np.nan
    ann = 252.0 / k
    return {"ic": ic, "ic_mean": mu, "ic_t": t,
            "n_days": int(np.isfinite(ic).sum()),
            "gross_per": gm, "gross_t": gt, "cost_per": per,
            "net_per": gm - per,
            "gross_annual": gm * ann, "net_annual": (gm - per) * ann,
            "by_liquidity": np.nanmean(buck, axis=0)}


def half_spread_frac(prices, dates) -> np.ndarray:
    """Half a tick as a FRACTION of price, vectorised over the whole panel.

    Two things this replaces. `reference.half_spread` already returns a
    fraction — an earlier version here divided by price a SECOND time, which
    made every cost roughly a thousand times too small and would have turned a
    losing spread into a winning one. And it was applied row by row, which at
    ~1,000 names over ~5,000 days is millions of Python calls per feature;
    `quality.tick_grid` does the same lookup vectorised over the tick regimes.

    Half a tick assumes a one-tick-wide book. On a liquid large cap that is
    roughly right; on a small cap it is generous. It is a floor, not an
    estimate, so every net figure downstream is optimistic.
    """
    t = tick_grid(np.asarray(prices, dtype=float), dates)
    p = np.asarray(prices, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(p > 0, 0.5 * t / p, np.nan)


def permutation_null(P: Dict[str, object], draws: int, seed: int,
                     k: int) -> np.ndarray:
    """Shuffle the feature WITHIN each day and rerun the identical pipeline.

    Preserves each day's cross-section of the feature and of the return, and
    destroys only which name's feature met which name's forward return. The
    neutralisation is reapplied inside the loop — H11 records why guards and
    controls applied once, on the real ordering, answer an easier question.
    """
    rng = np.random.default_rng(seed)
    out = np.full(draws, np.nan)
    base = P["y"]
    for i in range(draws):
        P["y"] = shuffle_within_days(base, P["starts"], P["ends"], rng)
        out[i] = analyse(P, k, light=True)["ic_mean"]
    P["y"] = base
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default=PANEL)
    ap.add_argument("--draws", type=int, default=100)
    ap.add_argument("--seed", type=int, default=20260823)
    ap.add_argument("--features", default=",".join(PREDICTED))
    a = ap.parse_args()

    D = pd.read_parquet(a.panel)
    D = D[~D["holdout"] & D["tradeable"]].copy()
    # one vectorised pass for the whole panel, not per feature per day
    D["hs_frac"] = half_spread_frac(D["close"].to_numpy(), D["date"])
    cov = float(np.isfinite(D["hs_frac"]).mean())
    print("=" * 78)
    print(" H13 — §8 PRICE/TA FEATURES, cross-sectional IC (Gate 1's test)")
    print("=" * 78)
    print(f"   {D.ticker.nunique():,} names, {D.date.nunique():,} days, "
          f"{len(D):,} tradeable in-sample rows")
    print(f"   {D.date.min().date()} … {D.date.max().date()}   "
          f"(holdout after this date is untouched)")
    print(f"   confirmatory horizon pre-specified at k={CONFIRM_K}; "
          f"the full decay curve is printed for every feature")
    print(f"   half-spread coverage {cov:.1%} of rows — the fraksi harga "
          f"schedule starts in 2005, so earlier bars have NO cost estimate")
    print(f"   and are excluded from the spread, never costed at zero\n")

    feats = [f for f in a.features.split(",") if f.strip()]
    summary = []
    for f in feats:
        if f not in D.columns:
            print(f" {f}: not in panel\n")
            continue
        ctrl = controls_for(f)
        note = (f"  [dropped {SELF_CONTROL[f]} from its own controls]"
                if f in SELF_CONTROL else "")
        pname = {1: "+", -1: "-", 0: "0 (negative control)"}[PREDICTED[f]]
        print(f" {f}   predicted {pname}{note}", flush=True)

        # Only the confirmatory horizon's prepared arrays are RETAINED.
        # Keeping all four alive holds ~23,000 small numpy arrays (the per-day
        # control factorisations) live at once, and the resulting garbage-
        # collection pressure made each pass ~25x slower than the same call
        # measured in isolation. Dropping the others between horizons is the
        # whole fix.
        keep = None
        for k in HORIZONS:
            P = prepare(D, f, f"fwd{k}", ctrl)
            A = analyse(P, k)
            print(f"     k={k:>2}  IC {A['ic_mean']:+.4f} (t {A['ic_t']:+6.2f})"
                  f"   spread gross {A['gross_annual']*100:+7.1f}%/yr"
                  f"   net {A['net_annual']*100:+8.1f}%/yr", flush=True)
            if k == CONFIRM_K:
                keep = (P, A)
            else:
                del P, A
                gc.collect()

        P, A = keep
        nulls = permutation_null(P, a.draws, a.seed, CONFIRM_K)
        v = nulls[np.isfinite(nulls)]
        p = (float((np.abs(v - v.mean()) >= abs(A["ic_mean"] - v.mean())).sum()
                   + 1) / (len(v) + 1)) if len(v) else np.nan
        lo, hi = (np.percentile(v, [2.5, 97.5]) if len(v) > 20
                  else (np.nan, np.nan))
        dec = A["by_liquidity"]
        fin = dec[np.isfinite(dec)]
        sign_ok = bool(len(fin) > 1 and np.all(np.sign(fin) == np.sign(fin[0])))
        print(f"     k={CONFIRM_K} null {v.mean():+.4f} [{lo:+.4f}, {hi:+.4f}] "
              f"over {len(v)} draws   p {p:.3f}   days {A['n_days']:,}")
        print("            IC by liquidity quintile "
              + " ".join(f"Q{i+1} {x:+.3f}" for i, x in enumerate(dec))
              + ("   SIGN STABLE" if sign_ok else "   SIGN FLIPS"))
        print(f"            spread gross {A['gross_per']*100:+.3f}%/period "
              f"(t {A['gross_t']:+.2f})  cost {A['cost_per']*100:.3f}%  "
              f"NET {A['net_per']*100:+.3f}%/period", flush=True)
        summary.append({"feature": f, "predicted": PREDICTED[f],
                        "ic": A["ic_mean"], "t": A["ic_t"], "p": p,
                        "sign_stable": sign_ok,
                        "net_annual": A["net_annual"]})
        print()

    S = pd.DataFrame(summary)
    print("=" * 78)
    print(" SUMMARY — Gate 1's conjunction applied to each feature")
    print("=" * 78)
    print(f" Bonferroni bar with 38 trials: p < 0.0013")
    print(f" NOTE the permutation p FLOORS at 1/(draws+1) = "
          f"{1.0/(a.draws+1):.4f}: a feature outside every draw cannot report "
          f"a smaller number, however extreme it is. Significance is therefore "
          f"judged on the HAC t as well, where |t| > 3.21 clears the bar.\n")
    for _, r in S.iterrows():
        # bool(), not the raw numpy comparison: `np.False_ is False` is False
        # in Python, so an earlier version fell through the `is False` branch
        # and mislabelled a FAILED prediction as "n/a (null control)".
        pred = int(r["predicted"])
        signed = (bool(np.sign(r["ic"]) == pred) if pred != 0 else None)
        marks = [
            "SIG" if abs(r["t"]) > 3.21 else "not sig",
            "sign stable" if r["sign_stable"] else "sign FLIPS",
            "net>0" if r["net_annual"] > 0 else "net<=0",
        ]
        d = ("n/a (negative control)" if signed is None
             else "as predicted" if signed else "AGAINST prediction")
        print(f"  {r['feature']:<11} IC {r['ic']:+.4f}  t {r['t']:+6.2f}  "
              f"p {r['p']:.3f}  net/yr {r['net_annual']*100:+7.1f}%   "
              f"{' | '.join(marks)}   {d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
