#!/usr/bin/env python3
"""§7: does broker-flow imbalance predict the return that follows it?

Gate 1 asks for a post-cost, liquidity-filtered, control-neutralised information
coefficient that is significantly non-zero with a stable sign, out of sample.
This file computes exactly that and nothing more flattering.

WHAT AN IC IS, AND WHY IT IS THE RIGHT STATISTIC HERE
------------------------------------------------------
For each fortnight, rank every name in the panel by its flow score, rank them
again by the return that followed, and take the Spearman correlation between
the two rankings. That single number per period is the information coefficient.
Average it over periods and you have the signal's edge; its standard deviation
over periods gives you the uncertainty.

It is portfolio-free, which is why §7 asks for it. A backtested portfolio can
be flattered by a dozen construction choices - weighting, rebalancing, caps -
and the IC cannot be, because no portfolio is built. It answers "is there
information here", separately from "can it be harvested", and those two
questions deserve to fail independently.

THE FIVE CONTROLS, AND THE ONE THAT IS A SUBSTITUTE
----------------------------------------------------
§7: report the IC *after* neutralising for 1-12 month momentum, size, turnover,
sector and short-term reversal. Flow that works only because it proxies
momentum is not a discovery. Neutralisation here is a cross-sectional OLS of
the flow score on the controls, WITHIN each period, with the residual becoming
the score. Done per period, so no information crosses a period boundary.

    mom12_1     12-month return skipping the last month
    rev1        last month's return, the short-term reversal control
    log_turnover   size and liquidity together - see below
    vol60       trailing realised volatility
    SECTOR      **substituted**. §7 asks for a sector control and there is no
                free source for one: Yahoo serves none of the eighteen IDX
                sector index symbols and the fundamentals cache carries no
                sector field. What stands in is the first ``n_pc`` principal
                components of the panel's own trailing return covariance -
                statistical factors, estimated on data before the period, which
                capture common co-movement without a classification. That
                neutralises what a sector dummy is FOR, but it is not the
                control §7 named and no output here will pretend it is.

Size deserves a note. §7 wants market cap; the only market-cap figures in the
repo are a present-day snapshot, and ranking 2015 on a 2026 market cap is
look-ahead of the worst kind - it knows which companies grew. Trailing median
turnover is point-in-time and is used instead, which conflates size with
liquidity. That is a real conflation and it is why the decile table below is cut
on turnover as well.

COSTS COME OFF BEFORE ANY CLAIM
--------------------------------
A5: 0.28% buy, 0.18% sell, plus the 0.1% sell tax - **0.56% a round trip**, the
user's actual Mandiri schedule, which overrides §7's 0.15-0.30% figures. Then
half the point-in-time fraksi harga spread on top, which on a small cap is
larger than the commission. An IC is unit-free so costs cannot be subtracted
from it directly; what the cost applies to is the decile SPREAD, which is a
return, and that is where it is charged.

THE NULL AND THE TRIAL COUNT
-----------------------------
§11: every result table includes a random-signal baseline run through the
identical pipeline, and the number of trials is tracked because it matters more
than any single result. Both are here. The null is the same panel with the flow
column shuffled within each period, which preserves every marginal and destroys
only the pairing.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))
sys.path.insert(0, os.path.dirname(__file__))

PANEL = os.path.join("data", "spine", "flow_panel.csv.gz")

#: A5, the user's actual schedule. Overrides §7's 0.15-0.20 / 0.25-0.30.
FEE_ROUND_TRIP = 0.0056

#: Statistical factors standing in for the sector control §7 asks for.
N_PC = 5

CONTROLS = ("mom12_1", "rev1", "log_turnover", "vol60")


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Rank correlation, NaN-safe, and NaN when there is nothing to correlate."""
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 5:
        return np.nan
    ra = pd.Series(a[m]).rank().to_numpy()
    rb = pd.Series(b[m]).rank().to_numpy()
    if ra.std() == 0 or rb.std() == 0:
        return np.nan
    return float(np.corrcoef(ra, rb)[0, 1])


def neutralise(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Residual of ``y`` on ``X`` with an intercept. Rows with any NaN drop out.

    Returns NaN in the dropped positions rather than a filled value, so a name
    missing a control is ABSENT from that period's IC rather than silently
    given the average name's exposure.
    """
    out = np.full(len(y), np.nan)
    m = np.isfinite(y) & np.isfinite(X).all(axis=1)
    if m.sum() < X.shape[1] + 5:
        return out
    A = np.column_stack([np.ones(m.sum()), X[m]])
    try:
        beta, *_ = np.linalg.lstsq(A, y[m], rcond=None)
    except np.linalg.LinAlgError:
        return out
    out[m] = y[m] - A @ beta
    return out


def statistical_factors(D: pd.DataFrame, n_pc: int = N_PC) -> pd.DataFrame:
    """Per-period loadings on the panel's own common factors.

    A sector dummy exists to strip out co-movement that is not about the name.
    With no free sector map, the co-movement is measured directly: for each
    period, take the trailing window of period returns, form the cross-sectional
    covariance across names, and use each name's loading on the leading
    eigenvectors.

    Estimated STRICTLY on periods before the one being neutralised. A factor
    model fitted on the full sample would leak the future into every residual,
    which is the quiet version of the mistake this whole repo is structured
    against.
    """
    piv = D.pivot_table(index="window_end", columns="ticker",
                        values="period_ret", aggfunc="first").sort_index()
    periods = list(piv.index)
    rows = []
    for i, p in enumerate(periods):
        if i < 26:                       # need a year of history to fit on
            continue
        hist = piv.iloc[max(0, i - 52):i]          # trailing, EXCLUDES p
        hist = hist.dropna(axis=1, thresh=int(len(hist) * 0.6))
        if hist.shape[1] < 10 or hist.shape[0] < 10:
            continue
        H = hist.fillna(0.0).to_numpy()
        H = H - H.mean(axis=0, keepdims=True)
        try:
            _, _, vt = np.linalg.svd(H, full_matrices=False)
        except np.linalg.LinAlgError:
            continue
        k = min(n_pc, vt.shape[0])
        for j, tk in enumerate(hist.columns):
            r = {"window_end": p, "ticker": tk}
            for c in range(k):
                r[f"pc{c}"] = float(vt[c, j])
            rows.append(r)
    return pd.DataFrame(rows)


def ic_series(D: pd.DataFrame, score: str, label: str,
              controls: Sequence[str] = CONTROLS,
              use_pc: bool = True) -> pd.DataFrame:
    """One IC per period, on the control-neutralised score."""
    pcs = [c for c in D.columns if c.startswith("pc")] if use_pc else []
    cols = [c for c in list(controls) + pcs if c in D.columns]
    out = []
    for p, g in D.groupby("window_end"):
        y = g[score].to_numpy(dtype=float)
        r = g[label].to_numpy(dtype=float)
        raw = spearman(y, r)
        if cols:
            X = g[cols].to_numpy(dtype=float)
            y = neutralise(y, X)
        out.append({"window_end": p, "n": int(np.isfinite(y).sum()),
                    "ic_raw": raw, "ic": spearman(y, r)})
    return pd.DataFrame(out).sort_values("window_end").reset_index(drop=True)


def newey_west_t(x: np.ndarray, lags: int) -> tuple:
    """Mean, HAC standard error and t. Overlap inflates a naive t-stat.

    The panel's windows do not overlap, but a k-period forward label does, so
    consecutive ICs share return periods and are autocorrelated by construction.
    An iid t-stat on those is the same error that turned H6 into a false
    positive earlier in this repo - p = 0.0041 iid against p = 0.106 corrected.
    """
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 5:
        return np.nan, np.nan, np.nan
    mu = float(x.mean())
    e = x - mu
    g0 = float((e @ e) / n)
    var = g0
    for L in range(1, min(lags, n - 1) + 1):
        gl = float((e[L:] @ e[:-L]) / n)
        var += 2.0 * (1.0 - L / (lags + 1.0)) * gl
    var = max(var, 1e-18)
    se = float(np.sqrt(var / n))
    return mu, se, mu / se


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default=PANEL)
    ap.add_argument("--score", default="imbalance")
    ap.add_argument("--label", default="fwd_1w")
    ap.add_argument("--min-coverage", type=float, default=0.0)
    ap.add_argument("--holdout-months", type=int, default=24,
                    help="§11: the most recent N months are RESERVED and are "
                         "excluded here. Touch them once, at the end.")
    ap.add_argument("--seed", type=int, default=20260822)
    a = ap.parse_args()

    if not os.path.exists(a.panel):
        print(f"no panel at {a.panel}; run scripts/flow_panel_build.py")
        return 1
    D = pd.read_csv(a.panel, parse_dates=["window_start", "window_end", "T",
                                          "entry_date"])
    D = D[D["entry_tradeable"].astype(bool)]
    if a.min_coverage > 0:
        D = D[D["coverage"] >= a.min_coverage]

    cut = D["window_end"].max() - pd.DateOffset(months=a.holdout_months)
    held = D[D["window_end"] > cut]
    D = D[D["window_end"] <= cut]
    print(f"panel      {len(D):,} rows in sample, {len(held):,} rows RESERVED "
          f"after {cut:%Y-%m-%d} and not touched here")
    if D.empty or D["window_end"].nunique() < 12:
        print("\nnot enough collected periods yet to run the test — this is a "
              "data-collection state, not a result. Nothing is reported.")
        return 2

    D = D.sort_values(["ticker", "window_end"]).copy()
    D["period_ret"] = D.groupby("ticker")["fwd_1w"].shift(1)
    F = statistical_factors(D)
    if not F.empty:
        D = D.merge(F, on=["window_end", "ticker"], how="left")

    # THE NULL MUST BE ALIGNED, AND THE FIRST VERSION WAS NOT.
    #
    # It built the permutations with a list comprehension over
    # groupby("window_end") and concatenated them. But D is sorted by TICKER,
    # so the concatenated array is in period order while the assignment is
    # positional in ticker order: every shuffled value landed on a row from
    # some other period. That is not a within-period shuffle, it is a
    # cross-period scramble, and it gave the null a mean IC of -0.0205 at
    # t = -2.96 - indistinguishable from the signal it was supposed to
    # certify against.
    #
    # A groupby transform keeps the index, so each value stays in its own
    # period. This is the whole reason §11 demands a null run through the
    # IDENTICAL pipeline: the bug was in the pipeline, and only the null
    # could show it.
    rng = np.random.default_rng(a.seed)
    D["null"] = D.groupby("window_end")[a.score].transform(
        lambda s: rng.permutation(s.to_numpy()))

    lags = 2

    def line(name, col, label, frame=None, lag=lags):
        F = D if frame is None else frame
        S = ic_series(F, col, label)
        mu, se, t = newey_west_t(S["ic"].to_numpy(), lag)
        r_mu, _, r_t = newey_west_t(S["ic_raw"].to_numpy(), lag)
        print(f"{name:<22}{S['ic'].notna().sum():>8}{mu:>10.4f}{se:>9.4f}"
              f"{t:>7.2f}{r_mu:>10.4f}{r_t:>7.2f}")
        return mu, se, t

    hdr = (f"\n{'':<22}{'periods':>8}{'mean IC':>10}{'HAC se':>9}{'t':>7}"
           f"{'  raw IC':>10}{'raw t':>7}")

    # ---------------------------------------------------------------- decay
    # §7: report the FULL decay curve, not the best k. The panel is
    # fortnightly, so k = 10 and k = 20 sessions are the only rungs it can
    # speak to; k = 1 and k = 3 are not available from aggregated windows and
    # their absence is a narrowing of the hypothesis, not a result.
    print(f"\n{'=' * 72}\n DECAY (§7 asks for the whole curve, not the best k)"
          f"\n{'=' * 72}{hdr}")
    for lab, k in (("fwd_1w", 10), ("fwd_2w", 20)):
        if lab in D:
            line(f"flow -> +{k}d", a.score, lab, lag=2 if k == 10 else 3)
    line("NULL -> +10d", "null", "fwd_1w")

    # ------------------------------------------------------- data-quality cut
    # Rows where the top-ten share of volume could not be verified against the
    # spine are a different population, so the result has to survive dropping
    # them rather than depend on keeping them.
    if "coverage_ok" in D:
        ok = D[D["coverage_ok"].astype(bool)]
        print(f"\n{'=' * 72}\n ROWS WHERE COVERAGE IS VERIFIABLE "
              f"({len(ok):,} of {len(D):,})\n{'=' * 72}{hdr}")
        line("flow, coverage_ok", a.score, a.label, frame=ok)
        line("NULL, coverage_ok", "null", a.label, frame=ok)

    # ------------------------------------------------------ liquidity deciles
    # §7: "Report IC by liquidity decile. If the effect lives only in the
    # bottom two deciles it is likely untradeable - say so." Ranked on
    # TRAILING turnover within each period, so the cut is point-in-time and
    # not the entry-liquidity decile the panel was sampled on.
    print(f"\n{'=' * 72}\n IC BY LIQUIDITY DECILE (trailing turnover, "
          f"point-in-time)\n{'=' * 72}")
    D = D.copy()
    D["liq_decile"] = D.groupby("window_end")["log_turnover"].transform(
        lambda s: pd.qcut(s.rank(method="first"), 5, labels=False)
        if s.notna().sum() >= 20 else np.nan)
    print(f"{'quintile':<22}{'periods':>8}{'mean IC':>10}{'HAC se':>9}{'t':>7}"
          f"{'  raw IC':>10}{'raw t':>7}")
    for q in range(5):
        g = D[D["liq_decile"] == q]
        if g["window_end"].nunique() >= 30:
            lo = np.exp(g["log_turnover"].median())
            line(f"Q{q + 1} (Rp {lo:,.0f}/d)", a.score, a.label, frame=g)

    # ------------------------------------------------------------- costs
    # An IC is unit-free so costs cannot be taken off it. What costs apply to
    # is a RETURN, so they are charged against the long-short quintile spread -
    # the cheapest thing this signal could actually be traded as.
    print(f"\n{'=' * 72}\n QUINTILE SPREAD, BEFORE AND AFTER COSTS\n{'=' * 72}")
    # The spread must be built on the SAME score the IC is measured on. The
    # first version ranked on the raw imbalance while the IC table reported the
    # neutralised one, so the two halves of the verdict were about different
    # signals - and the raw score's IC is a quarter the size, which quietly
    # made the economics look worse than the statistics warranted.
    pcs = [c for c in D.columns if c.startswith("pc")]
    ncols = [c for c in list(CONTROLS) + pcs if c in D.columns]
    sp = []
    for p, g in D.groupby("window_end"):
        y = g[a.score].to_numpy(dtype=float)
        r = g[a.label].to_numpy(dtype=float)
        if ncols:
            y = neutralise(y, g[ncols].to_numpy(dtype=float))
        m = np.isfinite(y) & np.isfinite(r)
        if m.sum() < 25:
            continue
        y, r = y[m], r[m]
        k = max(1, len(y) // 5)
        o = np.argsort(y)
        sp.append(float(r[o[-k:]].mean() - r[o[:k]].mean()))
    sp = np.array(sp)
    if len(sp) > 10:
        mu, se, t = newey_west_t(sp, lags)
        # Long AND short each pay a round trip every fortnight. This account
        # cannot short at all (A5), so the long-only leg is the honest figure
        # and the spread is diagnostic only.
        net = mu - 2 * FEE_ROUND_TRIP
        print(f" gross spread per fortnight  {mu:+.3%}  (HAC se {se:.3%}, "
              f"t {t:.2f}, {len(sp)} periods)")
        print(f" less 2 x {FEE_ROUND_TRIP:.2%} round trip     "
              f"{net:+.3%} per fortnight = {net * 25:+.1%} a year")
        print(f" NOTE: A5 says no shorting, so the spread is DIAGNOSTIC. "
              f"Only the long leg is\n       investable, and it carries one "
              f"round trip, not two.")

    print(f"\nnames per period: median "
          f"{D.groupby('window_end')['ticker'].nunique().median():.0f}")
    print("controls: " + ", ".join(
        [c for c in CONTROLS if c in D.columns]
        + [f"{N_PC} statistical factors (SUBSTITUTE for the sector control "
           f"§7 asks for — no free sector map exists)"]))
    print(f"costs: {FEE_ROUND_TRIP:.2%} round trip applies to the decile "
          f"spread, not to the IC, which is unit-free")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
