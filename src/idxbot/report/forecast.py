"""Forecast verification — is the brief a good forecaster, separately from
whether the forecast is worth trading.

WHY THESE ARE TWO QUESTIONS AND NOT ONE
-----------------------------------------
"Can you forecast?" and "can you make money?" get conflated constantly, and in
this market they have opposite answers. A weather service that says "70% chance
of rain" and is right 70% of the time is an excellent forecaster; whether you
can profit from it depends entirely on what a raincoat costs. This repository
has measured the raincoat: 56 bps of fees plus a fraksi-harga half-spread, and
four independent instruments all died against it.

None of that says the forecasts are *wrong*. It says they are *small*. So this
module asks the forecasting question on its own terms, using the standard
verification apparatus rather than a P&L:

    CALIBRATION   when the brief says 45%, does it rain 45% of the time?
    RESOLUTION    does it say different things on different days, or is it
                  just quoting the base rate back?
    SKILL         does it beat climatology — the unconditional base rate —
                  by the Brier score?

A forecast can be perfectly calibrated and completely useless: always predict
the base rate and you are calibrated by construction, with zero resolution.
Both numbers are reported for exactly that reason.

EVERYTHING HERE IS PRE-HOLDOUT
--------------------------------
§11 reserves the last 24 months. `scripts/power.py` established that those 24
months hold 24 non-overlapping 20-session periods and are powered to detect
+1.97% per period against a real effect of +0.59% — so they cannot answer the
economic question and are not spent on it. Calibration is measured in-sample
with a walk-forward split, which is the honest thing available.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

#: Probability bins for the reliability curve. Ten is conventional and gives
#: enough per bin at this sample size to be worth plotting.
BINS = 10


def brier(p: np.ndarray, y: np.ndarray) -> float:
    """Mean squared error of a probabilistic forecast. Lower is better."""
    p, y = np.asarray(p, float), np.asarray(y, float)
    m = np.isfinite(p) & np.isfinite(y)
    return float(np.mean((p[m] - y[m]) ** 2)) if m.any() else np.nan


def brier_skill(p: np.ndarray, y: np.ndarray,
                ref: Optional[np.ndarray] = None) -> float:
    """Brier skill against climatology. 0 = no better than the base rate.

    The reference is the unconditional frequency, which is the forecast you
    get for free by saying the same thing every day. Anything that cannot beat
    it is decorative, however well calibrated.
    """
    p, y = np.asarray(p, float), np.asarray(y, float)
    m = np.isfinite(p) & np.isfinite(y)
    if not m.any():
        return np.nan
    r = np.full(m.sum(), float(y[m].mean())) if ref is None \
        else np.asarray(ref, float)[m]
    bs, bs_ref = brier(p[m], y[m]), brier(r, y[m])
    return float(1.0 - bs / bs_ref) if bs_ref > 0 else np.nan


def reliability(p: np.ndarray, y: np.ndarray,
                bins: int = BINS) -> pd.DataFrame:
    """The reliability curve: predicted probability against realised frequency.

    Bins are on the PREDICTED value, so a perfectly calibrated forecaster puts
    every row on the diagonal. Bin counts travel with it — a bin holding
    eleven observations tells you nothing and should not be read as a wobble
    in calibration.
    """
    p, y = np.asarray(p, float), np.asarray(y, float)
    m = np.isfinite(p) & np.isfinite(y)
    p, y = p[m], y[m]
    if not len(p):
        return pd.DataFrame(columns=["bin", "predicted", "observed", "n"])
    edges = np.linspace(p.min(), p.max() + 1e-12, bins + 1)
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, bins - 1)
    rows = []
    for b in range(bins):
        s = idx == b
        if not s.any():
            continue
        rows.append({"bin": b, "predicted": float(p[s].mean()),
                     "observed": float(y[s].mean()), "n": int(s.sum())})
    return pd.DataFrame(rows)


def decompose(p: np.ndarray, y: np.ndarray,
              bins: int = BINS) -> Dict[str, float]:
    """Murphy's decomposition: Brier = reliability - resolution + uncertainty.

    RELIABILITY (lower better) is calibration error — how far the reliability
    curve sits off the diagonal.

    RESOLUTION (higher better) is how much the forecast actually varies away
    from the base rate. **This is the number that matters here.** A forecaster
    that always says "47%" is perfectly reliable and has zero resolution, and
    is worth nothing. The brief's conditional cells will be judged on this.

    UNCERTAINTY is the base rate's own variance — a property of the market,
    not of the forecast, and not improvable.
    """
    p, y = np.asarray(p, float), np.asarray(y, float)
    m = np.isfinite(p) & np.isfinite(y)
    p, y = p[m], y[m]
    n = len(p)
    if n < bins * 5:
        return {}
    base = float(y.mean())
    R = reliability(p, y, bins)
    rel = float((R["n"] * (R["predicted"] - R["observed"]) ** 2).sum() / n)
    res = float((R["n"] * (R["observed"] - base) ** 2).sum() / n)
    unc = float(base * (1 - base))
    return {"brier": brier(p, y), "reliability": rel, "resolution": res,
            "uncertainty": unc, "base_rate": base, "n": n,
            "skill": float(1.0 - brier(p, y) / unc) if unc > 0 else np.nan}


# ==========================================================================
#: Prior strength for shrinking a cell's frequency toward the base rate, in
#: pseudo-observations. Chosen ON TRAINING FOLDS ONLY by :func:`pick_prior`;
#: this is the fallback when that cannot run.
PRIOR = 2000.0


def shrink(counts: pd.Series, ups: pd.Series, base: float,
           prior: float = PRIOR) -> pd.Series:
    """Empirical-Bayes shrinkage of each cell's up-frequency toward the base.

    THE RAW FREQUENCIES ARE OVERCONFIDENT AND THE VERIFICATION PROVED IT. The
    first pass scored a Brier skill of **-0.0093** — worse than saying "45%"
    every day — with a reliability curve that predicted 99.4% where reality was
    58.9%. Resolution was genuinely non-zero (p = 0.000 against a shuffle), so
    the conditioning carries information; it was calibration that destroyed it,
    because a cell holding a few hundred training bars was being quoted with
    the same confidence as one holding forty thousand.

    Shrinkage is the standard correction and it is not a fudge: a cell's
    posterior mean under a Beta prior centred on the base rate is exactly

        (ups + prior * base) / (n + prior)

    so a large cell barely moves and a small one collapses onto climatology,
    which is the correct amount of confidence to have in each.
    """
    n = counts.astype(float)
    return (ups.astype(float) + prior * base) / (n + prior)


def pick_prior(tr: pd.DataFrame, fwd: str,
               grid: Sequence[float] = (0, 100, 300, 1000, 3000, 10000, 30000),
               inner: int = 3) -> float:
    """Choose the prior strength on TRAINING data only, by inner walk-forward.

    Tuning it on the scored fold would be the leak the whole embargo exists to
    prevent — the prior would be fitted to the outcomes it is then graded
    against. So the training window is split again, forward in time, and the
    value that scores best on the inner holdouts is carried outward.
    """
    tr = tr.sort_values("date")
    days = np.array(sorted(tr["date"].unique()))
    if len(days) < inner * 60:
        return PRIOR
    cuts = np.array_split(days, inner + 1)
    best, best_bs = PRIOR, np.inf
    for pr in grid:
        errs = []
        for i in range(1, len(cuts)):
            a = tr[tr["date"] <= cuts[i - 1][-1]]
            b = tr[tr["date"].isin(cuts[i])]
            if a.empty or b.empty:
                continue
            up = (a[fwd] > 0).astype(float)
            base = float(up.mean())
            g = a.assign(_u=up).groupby("bucket")["_u"]
            p = shrink(g.count(), g.sum(), base, pr)
            q = b["bucket"].map(p).fillna(base).to_numpy()
            errs.append(brier(q, (b[fwd] > 0).astype(float).to_numpy()))
        if errs and np.mean(errs) < best_bs:
            best, best_bs = pr, float(np.mean(errs))
    return best


def walk_forward(D: pd.DataFrame, k: int = 20, n_folds: int = 5,
                 embargo: int = 20, prior: Optional[float] = None,
                 min_train: int = 200) -> pd.DataFrame:
    """Fit the cell probabilities on past folds, score them on the next.

    §11 requires purged, embargoed walk-forward and is explicit that plain
    k-fold leaks on a financial panel. Here the leak would be flagrant: the
    forward return of a bar near a fold boundary overlaps bars on the other
    side, so a cell's "historical" frequency would partly be built from the
    very outcomes it is scored against.

    So folds run forward in time only, and an ``embargo`` of ``k`` sessions is
    dropped after each training window — the exact span over which a training
    label can still reach into the test set.

    Returns one row per scored bar with the probability the cell predicted and
    the outcome that followed.
    """
    fwd = f"fwd{k}"
    D = D[["date", "ticker", "bucket", fwd]].dropna().copy()
    D["date"] = pd.to_datetime(D["date"])
    D = D.sort_values("date")
    days = np.array(sorted(D["date"].unique()))
    if len(days) < n_folds * (embargo + 30):
        return pd.DataFrame()
    cuts = np.array_split(days, n_folds + 1)
    out = []
    for i in range(1, len(cuts)):
        train_end = cuts[i - 1][-1]
        # the embargo: nothing within k sessions of the boundary trains
        emb = days[days <= train_end]
        # `emb[:-0]` is EMPTY, not "all of it" — so an embargo of zero silently
        # trained on nothing and returned no forecasts at all. Guard the zero
        # case explicitly rather than relying on negative-index arithmetic.
        if embargo > 0:
            emb = emb[:-embargo] if len(emb) > embargo else emb[:0]
        if not len(emb):
            continue
        tr = D[D["date"] <= emb[-1]]
        te = D[D["date"].isin(cuts[i])]
        if tr.empty or te.empty:
            continue
        base = float((tr[fwd] > 0).mean())
        pr = pick_prior(tr, fwd) if prior is None else float(prior)
        up = (tr[fwd] > 0).astype(float)
        grp = tr.assign(_u=up).groupby("bucket")["_u"]
        cnt, sm = grp.count(), grp.sum()
        p_up = shrink(cnt, sm, base, pr)
        # a cell too thin to have learned anything falls back to climatology
        # rather than quoting a frequency built from a handful of bars
        p_up = p_up.where(cnt >= min_train, base)
        g = te.copy()
        g["p"] = g["bucket"].map(p_up).fillna(base)
        g["base"] = base
        g["prior"] = pr
        g["y"] = (g[fwd] > 0).astype(float)
        g["fold"] = i
        out.append(g)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def verify(W: pd.DataFrame, bins: int = BINS) -> Dict[str, object]:
    """Score a walk-forward frame, against climatology and against a shuffle.

    THE SHUFFLE NULL IS NOT OPTIONAL. Resolution is a sum of squared deviations
    and is therefore positive by construction on any finite sample, including
    one where the forecast is pure noise. Reading it against zero would report
    skill that is arithmetic rather than evidence — the same error this repo
    has made four times with other statistics. The null shuffles the predicted
    probability within each date, preserving the cross-section of forecasts and
    of outcomes and destroying only which name got which.
    """
    if W.empty:
        return {}
    d = decompose(W["p"].to_numpy(), W["y"].to_numpy(), bins)
    if not d:
        return {}
    d["skill_vs_base"] = brier_skill(W["p"].to_numpy(), W["y"].to_numpy(),
                                     W["base"].to_numpy())
    rng = np.random.default_rng(20260825)
    codes = pd.factorize(W["date"])[0]
    p = W["p"].to_numpy()
    y = W["y"].to_numpy()
    nulls = np.empty(200)
    for i in range(200):
        order = np.lexsort((rng.random(len(codes)), codes))
        perm = np.empty_like(p)
        perm[np.argsort(codes, kind="mergesort")] = p[order]
        nulls[i] = decompose(perm, y, bins).get("resolution", np.nan)
    d["resolution_null_mean"] = float(np.nanmean(nulls))
    d["resolution_null_p95"] = float(np.nanpercentile(nulls, 95))
    d["resolution_p"] = float(np.nanmean(nulls >= d["resolution"]))
    return d


def summarise(d: Dict[str, object]) -> Sequence[str]:
    """Plain sentences for the numbers, with the honest reading attached."""
    if not d:
        return ["not enough data to verify"]
    out = [
        f"Scored {d['n']:,} forecasts. Base rate {d['base_rate']:.1%} of bars "
        f"up over the horizon.",
        f"Brier {d['brier']:.4f} against an uncertainty floor of "
        f"{d['uncertainty']:.4f} — skill {d['skill']:+.4f}, and against the "
        f"walk-forward base rate {d['skill_vs_base']:+.4f}.",
        f"Calibration error {d['reliability']:.5f} (lower is better); "
        f"resolution {d['resolution']:.5f} against a shuffled null of "
        f"{d['resolution_null_mean']:.5f} "
        f"(p95 {d['resolution_null_p95']:.5f}, p = {d['resolution_p']:.3f}).",
    ]
    if d["resolution_p"] < 0.05:
        out.append(
            "The forecasts carry real resolution — they say different things "
            "on different days and the differences are informative. That is a "
            "genuine forecasting result and it is NOT a trading result: "
            "scripts/power.py measured the same conditioning at +0.59% per 20 "
            "sessions gross and -0.31% after costs.")
    else:
        out.append(
            "Resolution is indistinguishable from a shuffle — the forecasts "
            "are quoting the base rate back with noise on top.")
    return out
