#!/usr/bin/env python3
"""H19 — at what give-back does the edge actually die, and do indicators know?

    python3 scripts/recovery.py

THE QUESTION H17 AND H18 BOTH DODGED
--------------------------------------
Both studies picked a trailing-stop distance by GRID SEARCH: score 15%, 20%,
25%, 30%, 40%, keep whichever won the cohort median. That answers "which of
these five did best on this sample" and NOT the question a holder actually
asks, which is:

    given that I am already X% below the peak, what are the odds this comes
    back, and what do I make on average if I sit through it?

A trail at 15% is only correct if 15% is roughly where the forward edge turns
negative. Nobody measured that. If the edge survives to 30% then a 15% trail is
throwing away positions that were going to recover, and the grid never had a
chance to say so because it only ever compared whole rules against each other.

So this measures the conditional directly.

PRE-REGISTERED, BEFORE ANY OF IT WAS RUN
------------------------------------------
  H19a  P(new high) falls monotonically with give-back depth. Near-trivial,
        included as a sanity check on the construction — if this fails the
        panel is wrong, not the market.

  H19b  There is a depth at which the mean forward return, net of the round
        trip, turns negative. THAT depth is where a trail belongs, and the
        prediction is that it is DEEPER than the 15% H17 selected — because
        the entry selects for high-volatility names whose ordinary noise is
        larger than 15%.

  H19c  At MATCHED give-back depth, indicator state separates recoverers from
        non-recoverers by more than a label-shuffled null. This is the sharp
        version of H18b, which asked a whole-rule question and failed. A
        conditional probability can carry information that a whole rule
        cannot monetise, so the two are not the same test.

WHAT IS MEASURED
-----------------
Every liquid pre-holdout bar whose trailing 252-session window contains a run
of at least +50% into its peak — the panel-wide equivalent of an ARMED
position, which is the only state where a trailing stop can fire at all. For
each such bar: how far below the running peak it sits, whether it makes a new
high within k sessions, and what it returns over k.

Dependence is handled by resampling CALENDAR MONTHS in blocks, not rows: bars
overlap by construction and the whole cross-section moves together, so a row
bootstrap would report intervals an order of magnitude too narrow.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

PANEL = os.path.join("data", "spine", "price_panel.parquet")
IND = os.path.join("data", "spine", "indicator_panel.parquet")

#: Trailing window the peak is measured over, in sessions.
LOOK = 252

#: How far the name must have run INTO that peak to count as armed.
ARM = 0.50

#: Give-back buckets, as fractions below the running peak.
EDGES = [-1.0, -0.60, -0.50, -0.40, -0.35, -0.30, -0.25, -0.20,
         -0.15, -0.10, -0.05, 0.0]

#: A5's round trip, before the per-name spread term.
FEE = 0.0056

#: Minimum 20-day median traded value, in rupiah.
MIN_VALUE = 1e9


def build(P: pd.DataFrame, I: pd.DataFrame, k: int) -> pd.DataFrame:
    """One row per armed liquid bar: give-back now, what happened over k."""
    P = P.sort_values(["ticker", "date"])
    out: List[pd.DataFrame] = []
    ind = {t: g for t, g in I.groupby("ticker", sort=False)}
    for t, g in P.groupby("ticker", sort=False):
        a = g["adj_close"].to_numpy(dtype=float)
        n = len(a)
        if n < LOOK + k + 2:
            continue
        s = pd.Series(a)
        peak = s.rolling(LOOK, min_periods=LOOK).max().to_numpy()
        base = s.shift(LOOK - 1).to_numpy()          # price entering the window
        with np.errstate(invalid="ignore", divide="ignore"):
            dd = a / peak - 1.0
            runup = peak / base - 1.0
            fwd = np.concatenate([a[k:] / a[:-k] - 1.0, np.full(k, np.nan)])
        # a new high means exceeding the running peak at any point in k
        fmax = pd.Series(a).shift(-1).rolling(k, min_periods=1).max()
        fmax = fmax.shift(-(k - 1)).to_numpy()
        newhigh = fmax > peak
        # deepest further fall over the next k, from here
        fmin = pd.Series(a).shift(-1).rolling(k, min_periods=1).min()
        fmin = fmin.shift(-(k - 1)).to_numpy()
        with np.errstate(invalid="ignore", divide="ignore"):
            mae = fmin / a - 1.0

        d = pd.DataFrame({
            "date": g["date"].to_numpy(), "ticker": t,
            "close": g["close"].to_numpy(dtype=float),
            "dd": dd, "runup": runup, "fwd": fwd,
            "newhigh": newhigh, "mae": mae,
            "log_turnover": g["log_turnover"].to_numpy(dtype=float),
            "tradeable": g["tradeable"].to_numpy(),
            "holdout": g["holdout"].to_numpy()})
        f = ind.get(t)
        if f is not None:
            m = d.merge(f[["date", "ema20", "ema50", "atr22", "stoch_k",
                           "stoch_d", "tvz20", "close"]].rename(
                columns={"close": "adj"}), on="date", how="left")
        else:
            m = d.assign(ema20=np.nan, ema50=np.nan, atr22=np.nan,
                         stoch_k=np.nan, stoch_d=np.nan, tvz20=np.nan,
                         adj=np.nan)
        out.append(m)
    D = pd.concat(out, ignore_index=True)
    D = D[(~D["holdout"].astype(bool)) & D["tradeable"].astype(bool)
          & np.isfinite(D["fwd"]) & np.isfinite(D["dd"])
          & np.isfinite(D["runup"]) & (D["runup"] >= ARM)
          & np.isfinite(D["log_turnover"])
          & (np.exp(D["log_turnover"]) >= MIN_VALUE)]
    # An interval categorical cannot be written to parquet, and lexicographic
    # ordering of the string form scrambles the curve. Keep an integer bin for
    # ordering and a plain label for display.
    D["bin"] = pd.cut(D["dd"], EDGES, labels=False).astype("float")
    D = D[np.isfinite(D["bin"])]
    D["bin"] = D["bin"].astype(int)
    lab = np.array([f"{EDGES[i]:.0%} to {EDGES[i + 1]:.0%}"
                    for i in range(len(EDGES) - 1)])
    D["bucket"] = lab[D["bin"].to_numpy()]
    D["month"] = pd.PeriodIndex(pd.DatetimeIndex(D["date"]), freq="M")
    # give-back in the name's own volatility units, the chandelier's premise
    with np.errstate(invalid="ignore", divide="ignore"):
        D["dd_atr"] = (D["adj"] * D["dd"] / -D["atr22"])
    return D


def block_ci(D: pd.DataFrame, col: str, draws: int = 1000,
             seed: int = 20260825, block: int = 12) -> tuple:
    """CI on a mean, resampling CALENDAR MONTHS in blocks.

    Rows overlap and the cross-section co-moves, so a row bootstrap is not an
    option — it would treat one bull quarter as thousands of observations.
    """
    if D.empty:
        return (np.nan, np.nan)
    g = D.groupby("month")[col].agg(["sum", "count"])
    g = g[g["count"] > 0]
    if len(g) < 3 * block:
        return (np.nan, np.nan)
    s, c = g["sum"].to_numpy(), g["count"].to_numpy()
    n = len(s)
    rng = np.random.default_rng(seed)
    kb = int(np.ceil(n / block))
    st = rng.integers(0, n - block + 1, size=(draws, kb))
    take = (st[:, :, None] + np.arange(block)[None, None, :]
            ).reshape(draws, kb * block)[:, :n]
    m = s[take].sum(axis=1) / np.maximum(c[take].sum(axis=1), 1)
    return (float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)))


def curve(D: pd.DataFrame, k: int) -> pd.DataFrame:
    rows = []
    for bi, g in D.groupby("bin", observed=True, sort=True):
        if g.empty:
            continue
        b = g["bucket"].iloc[0]
        lo, hi = block_ci(g, "fwd")
        plo, phi = block_ci(g.assign(nh=g["newhigh"].astype(float)), "nh")
        rows.append({
            "give_back": str(b), "n": len(g), "names": g["ticker"].nunique(),
            "months": g["month"].nunique(),
            "p_newhigh": float(g["newhigh"].mean()),
            "p_lo": plo, "p_hi": phi,
            "fwd": float(g["fwd"].mean()),
            "fwd_lo": lo, "fwd_hi": hi,
            "fwd_med": float(g["fwd"].median()),
            "mae": float(g["mae"].mean()),
            "dd_atr": float(np.nanmedian(g["dd_atr"]))})
    R = pd.DataFrame(rows)
    return R.iloc[::-1].reset_index(drop=True)


def conditional(D: pd.DataFrame, draws: int = 200, seed: int = 20260825,
                min_side: int = 300, min_share: float = 0.05) -> pd.DataFrame:
    """H19c — at MATCHED depth, does indicator state separate outcomes?

    The split is computed WITHIN each give-back bucket so depth cannot leak in.

    THE NULL PERMUTES NAMES, NOT ROWS, AND THE FIRST VERSION GOT THAT WRONG.
    A name contributes roughly twenty near-identical bars a month — same
    indicator state, overlapping forward windows — so shuffling rows destroys
    the label while leaving the null far too tight, and every z came back
    inflated (one read -8.7). Permuting whole (ticker, month) blocks respects
    the unit the information actually varies at.

    A cell also needs BOTH SIDES populated. The first version guarded only
    against a share of exactly 0 or 1, so a flag true for 0.2% of rows sailed
    through and produced the largest effect in the table (-46.1%) off a
    handful of observations. ``min_side`` and ``min_share`` are the fix.
    """
    tests = {
        "above EMA20": lambda g: g["adj"] > g["ema20"],
        "above EMA50": lambda g: g["adj"] > g["ema50"],
        "stoch %K > %D": lambda g: g["stoch_k"] > g["stoch_d"],
        "stoch %K > 50": lambda g: g["stoch_k"] > 50.0,
        "turnover z > 0": lambda g: g["tvz20"] > 0.0,
        "give-back < 2 ATR": lambda g: g["dd_atr"] < 2.0,
    }
    rng = np.random.default_rng(seed)
    rows = []
    for name, fn in tests.items():
        for bi, g in D.groupby("bin", observed=True, sort=True):
            b = g["bucket"].iloc[0]
            m = fn(g)
            ok = m.notna() & np.isfinite(g["fwd"])
            g2 = g[ok].assign(flag=m[ok].astype(bool))
            if g2.empty:
                continue
            # collapse to the unit the label actually varies at
            blk = (g2.groupby(["ticker", "month"], observed=True)
                   .agg(fwd=("fwd", "mean"), nh=("newhigh", "mean"),
                        flag=("flag", "mean"), w=("fwd", "size"))
                   .reset_index())
            blk = blk[(blk["flag"] <= 0.2) | (blk["flag"] >= 0.8)]
            blk["flag"] = blk["flag"] >= 0.8
            t_side = int(blk.loc[blk["flag"], "w"].sum())
            f_side = int(blk.loc[~blk["flag"], "w"].sum())
            share = t_side / max(t_side + f_side, 1)
            if (min(t_side, f_side) < min_side
                    or not (min_share <= share <= 1 - min_share)
                    or blk["flag"].nunique() < 2 or len(blk) < 60):
                continue

            w = blk["w"].to_numpy(dtype=float)
            fw = blk["fwd"].to_numpy()
            nh = blk["nh"].to_numpy()
            lab = blk["flag"].to_numpy()
            mo = blk.groupby("month", observed=True).ngroup().to_numpy()

            def gap(x, s):
                a, bq = w[s].sum(), w[~s].sum()
                if a <= 0 or bq <= 0:
                    return np.nan
                return float((x[s] * w[s]).sum() / a
                             - (x[~s] * w[~s]).sum() / bq)

            d_fwd, d_nh = gap(fw, lab), gap(nh, lab)
            nf, nn = [], []
            for _ in range(draws):
                sh = np.empty_like(lab)
                for u in np.unique(mo):                # permute WITHIN month
                    q = mo == u
                    sh[q] = rng.permutation(lab[q])
                a, bq = gap(fw, sh), gap(nh, sh)
                if np.isfinite(a) and np.isfinite(bq):
                    nf.append(a)
                    nn.append(bq)
            if len(nf) < 20 or not np.isfinite(d_fwd):
                continue
            nf, nn = np.array(nf), np.array(nn)
            rows.append({
                "test": name, "give_back": str(b),
                "n": t_side + f_side, "blocks": len(blk),
                "share_true": share,
                "d_fwd": d_fwd, "null_sd_fwd": float(nf.std(ddof=1)),
                "z_fwd": float((d_fwd - nf.mean()) / max(nf.std(ddof=1), 1e-12)),
                "p_fwd": float(np.mean(np.abs(nf - nf.mean())
                                       >= abs(d_fwd - nf.mean()))),
                "d_nh": d_nh,
                "z_nh": float((d_nh - nn.mean()) / max(nn.std(ddof=1), 1e-12)),
                "p_nh": float(np.mean(np.abs(nn - nn.mean())
                                      >= abs(d_nh - nn.mean())))})
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default=PANEL)
    ap.add_argument("--indicators", default=IND)
    ap.add_argument("--k", type=int, default=60,
                    help="forward horizon in sessions")
    ap.add_argument("--cache", default="data/spine/recovery_state.parquet")
    ap.add_argument("--rebuild", action="store_true")
    a = ap.parse_args()

    P = pd.read_parquet(a.panel, columns=[
        "date", "ticker", "close", "adj_close", "log_turnover",
        "tradeable", "holdout"])
    P["date"] = pd.to_datetime(P["date"])
    I = pd.read_parquet(a.indicators)
    I["date"] = pd.to_datetime(I["date"])

    print("=" * 84)
    print(" H19 — THE RECOVERY CURVE: where does the edge actually die?")
    print("=" * 84)
    print(f" armed liquid pre-holdout bars: a {LOOK}-session window whose peak")
    print(f" sits at least +{ARM:.0%} above the price entering it.")
    print(f" forward horizon k = {a.k} sessions; round trip {FEE:.2%} + spread.\n")

    cache = a.cache.replace(".parquet", f"_k{a.k}.parquet")
    if os.path.exists(cache) and not a.rebuild:
        D = pd.read_parquet(cache)
        D["month"] = pd.PeriodIndex(pd.DatetimeIndex(D["date"]), freq="M")
        print(f" [state panel loaded from {cache}]")
    else:
        D = build(P, I, a.k)
        D.drop(columns=["month"]).to_parquet(cache, index=False)
        print(f" [state panel cached to {cache}]")
    print(f" {len(D):,} bars, {D['ticker'].nunique()} names, "
          f"{D['month'].nunique()} months, "
          f"{D['date'].min().date()} -> {D['date'].max().date()}\n")

    R = curve(D, a.k)
    print(" H19a/H19b — outcome by give-back depth")
    print(f"   {'give-back':<18}{'n':>9}{'P(new high)':>13}{'95% CI':>18}"
          f"{'mean fwd':>10}{'95% CI':>18}{'median':>9}{'ATRs':>6}")
    for _, r in R.iterrows():
        print(f"   {r['give_back']:<18}{r['n']:>9,}{r['p_newhigh']:>13.1%}"
              f"  [{r['p_lo']:>5.1%},{r['p_hi']:>6.1%}]"
              f"{r['fwd']:>10.1%}  [{r['fwd_lo']:>6.1%},{r['fwd_hi']:>7.1%}]"
              f"{r['fwd_med']:>9.1%}{r['dd_atr']:>6.1f}")

    R.to_csv("reports/recovery_curve.csv", index=False)
    mono = bool((R["p_newhigh"].diff().dropna() <= 1e-9).all())
    print(f"\n   H19a monotone decline in P(new high): "
          f"{'YES' if mono else 'NO'}")

    # where does the net edge die?
    R["net"] = R["fwd"] - FEE
    neg = R[R["fwd_hi"] < FEE]
    print("   H19b the depth where the mean forward return, net of cost,")
    if neg.empty:
        print("        is significantly negative: NONE — the upper bound of the")
        print("        interval stays above the round trip at every depth.")
    else:
        print(f"        is significantly negative: {neg.iloc[0]['give_back']}"
              f" and deeper")

    C = conditional(D)
    print("\n" + "=" * 84)
    print(" H19c — at MATCHED depth, do the indicators separate outcomes?")
    print("=" * 84)
    print(" null shuffles the flag inside (give-back bucket, month), so depth")
    print(" and regime are held fixed and only the indicator is destroyed.\n")
    if C.empty:
        print(" no cell had enough data")
    else:
        C = C.sort_values("p_fwd")
        print(f"   {'test':<20}{'give-back':<18}{'n':>8}{'blk':>6}{'share':>7}"
              f"{'d fwd':>9}{'z':>7}{'p':>7}{'d P(nh)':>9}{'p':>7}")
        for _, r in C.iterrows():
            print(f"   {r['test']:<20}{r['give_back']:<18}{r['n']:>8,}"
                  f"{r['blocks']:>6,}{r['share_true']:>7.0%}"
                  f"{r['d_fwd']:>+9.1%}"
                  f"{r['z_fwd']:>+7.1f}{r['p_fwd']:>7.3f}"
                  f"{r['d_nh']:>+9.1%}{r['p_nh']:>7.3f}")
        C.to_csv("reports/recovery_conditional.csv", index=False)
        bar = 0.05 / max(len(C), 1)
        hit = C[C["p_fwd"] < bar]
        print(f"\n   {len(C)} cells tested -> Bonferroni bar {bar:.4f}; "
              f"{len(hit)} clear it on forward return")
        if len(hit):
            best = hit.iloc[0]
            print(f"   strongest: {best['test']} at {best['give_back']}, "
                  f"{best['d_fwd']:+.1%} forward, z {best['z_fwd']:+.1f}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
