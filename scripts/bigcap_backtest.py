#!/usr/bin/env python3
"""H29 — does the Pine screen work on the top 50 big caps?

    python3 scripts/bigcap_backtest.py

THIS IS A NEW TEST ON A NEW UNIVERSE, NOT A RE-DISPLAY OF H26. The screen was
measured on the whole eligible IDX cross-section — 725 names, everything above
Rp1bn/day. "The top 50 big caps" is a different population, and this repo has
been burned before by quoting a number measured on one universe as if it held
on another (A21: a ten-year doubling rate read as a one-year one).

PRE-REGISTERED PREDICTION, WRITTEN BEFORE ANY CELL WAS SCORED:
  The screen will perform WORSE on the top 50 than on the full universe.
  Reason: H23 measured the most liquid decile at a 4.2% one-year touch-2x rate
  against a 10.2% base — big caps double far less often, so the numerator of
  the skew has much less room. I expect the skew to fall from 2.01 toward the
  big-cap base, and I expect the doubling rate to be low enough that "P(2x)"
  stops being the interesting statistic at all.
  If it comes back BETTER, that is a surprise and I should distrust it until
  the null and the half-split agree.

SIZE IS PROXIED BY LIQUIDITY, POINT-IN-TIME. The repo has no point-in-time
share count — A25 records that the only shares column available is frozen at
2024-07-10, so applying it to a 2010 bar is look-ahead, and Indonesian rights
issues are precisely what makes that wrong. Turnover is the proxy used
everywhere else here, and it is recomputed on every cohort date so the "top 50"
is the top 50 AS OF THAT DAY, not today's list projected backwards.

THE SCREEN IS THE PINE ONE, BIT FOR BIT: absolute thresholds on the same
quantities the script computes, not the cross-sectional percentiles H26 used.
"""

from __future__ import annotations

import os
import sys
from typing import Dict

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))
sys.path.insert(0, os.path.dirname(__file__))

from horizon_sweep import classify                              # noqa: E402
from idxbot.report import brief as B                            # noqa: E402

CACHE = os.path.join("data", "spine", "horizon_sweep.parquet")
INDEX = os.path.join("data", "cache", "ohlcv", "_JKSE.csv.gz")
K = 252
TOP_N = 50

#: Exactly what IDX_Context.pine tests.
HI_CUT, VOL_CUT = 0.9625, 0.0257

#: Measured top-decile dividend yield, to put the index on a total-return
#: basis. Comparing a total-return basket to a price index is the mistake A19
#: records as the worst in the project.
IDX_YIELD = 0.0177


def index_series() -> pd.Series:
    ix = pd.read_csv(INDEX)
    ix["date"] = pd.to_datetime(ix["date"], utc=True,
                                errors="coerce").dt.tz_localize(None)
    s = ix.set_index("date")["close"].astype(float).sort_index().dropna()
    return s[s > 0]


def cell(d: pd.DataFrame, m, label: str) -> Dict:
    s = d[m] if m is not None else d
    if len(s) < 100:
        return {"label": label, "n": len(s), "thin": True}
    up = float((s[f"peak{K}"] >= 2.0).mean())
    dn = float((s[f"end{K}"] <= 0.5).mean())
    return {"label": label, "n": len(s), "names": int(s["ticker"].nunique()),
            "up": up, "dn": dn, "skew": up / dn if dn > 0 else np.nan,
            "median": float(s[f"end{K}"].median() - 1.0),
            "mean": float(s[f"end{K}"].mean() - 1.0),
            "mean_log": float(np.log(np.maximum(s[f"end{K}"], 0.01)).mean())}


def show(c: Dict) -> None:
    if c.get("thin"):
        print(f"   {c['label']:<30}{c['n']:>7}   too few to read")
        return
    print(f"   {c['label']:<30}{c['n']:>7,}{c['names']:>7}{c['up']:>8.1%}"
          f"{c['dn']:>10.1%}{c['skew']:>7.2f}{c['median']:>+9.1%}"
          f"{c['mean']:>+9.1%}{np.exp(c['mean_log']) - 1:>+9.1%}")


def main() -> int:
    D = pd.read_parquet(CACHE)
    D = D[~D["holdout"].astype(bool)]
    d = classify(D, K)
    d = d[d["cls"] != "censored"].copy().reset_index(drop=True)

    #  POINT-IN-TIME top 50 by turnover on each cohort date
    d["liq_rank"] = d.groupby("date")["log_turnover"].rank(
        ascending=False, method="first")
    big = d["liq_rank"] <= TOP_N

    W = 100
    print("=" * W)
    print(f" H29 — THE PINE SCREEN ON THE TOP {TOP_N} BIG CAPS")
    print("=" * W)
    print(f" PRE-REGISTERED: I expect it to be WORSE here than on the full")
    print(f" universe, because big caps double far less often (H23: liquid")
    print(f" decile 4.2% vs a 10.2% base at one year).\n")
    print(f" {len(d):,} name-years total; {int(big.sum()):,} inside the "
          f"point-in-time top {TOP_N}, {d[big]['ticker'].nunique()} distinct names")

    sig = (d["hi52"] >= HI_CUT) & (d["vol60"] <= VOL_CUT)

    print(f"\n   {'cell':<30}{'n':>7}{'names':>7}{'P(2x)':>8}{'P(halve)':>10}"
          f"{'SKEW':>7}{'median':>9}{'mean':>9}{'CAGR/nm':>9}")
    c_all_base = cell(d, None, "ALL IDX  — no screen")
    c_all_sig = cell(d, sig, "ALL IDX  — screen on")
    c_big_base = cell(d, big, f"TOP {TOP_N}   — no screen")
    c_big_sig = cell(d, big & sig, f"TOP {TOP_N}   — screen on")
    for c in (c_all_base, c_all_sig, c_big_base, c_big_sig):
        show(c)

    if not c_big_sig.get("thin"):
        lift = c_big_sig["skew"] / c_big_base["skew"]
        print(f"\n   Within the top {TOP_N}: skew {c_big_base['skew']:.2f} "
              f"-> {c_big_sig['skew']:.2f}  ({lift:.2f}x)")
        print(f"   On the full universe:  {c_all_base['skew']:.2f} "
              f"-> {c_all_sig['skew']:.2f}  "
              f"({c_all_sig['skew'] / c_all_base['skew']:.2f}x)")
        verdict = ("WORSE — as predicted" if c_big_sig["skew"] < c_all_sig["skew"]
                   else "BETTER — against the prediction, treat with suspicion")
        print(f"   -> the screen is {verdict}")

    # ---- half-split ------------------------------------------------------
    mid = d["date"].quantile(0.5)
    print(f"\n   HALF-SPLIT within the top {TOP_N}:")
    print(f"   {'half':<10}{'n':>7}{'P(2x)':>8}{'P(halve)':>10}{'SKEW':>7}"
          f"{'base skew':>11}")
    both = True
    for lbl, mm in (("early", d["date"] <= mid), ("late", d["date"] > mid)):
        s, b = d[mm & big & sig], d[mm & big]
        if len(s) < 60:
            print(f"   {lbl:<10}{len(s):>7}   too few")
            both = False
            continue
        u = (s[f"peak{K}"] >= 2.0).mean()
        v = (s[f"end{K}"] <= 0.5).mean()
        bs = ((b[f"peak{K}"] >= 2.0).mean() /
              max((b[f"end{K}"] <= 0.5).mean(), 1e-9))
        sk = u / v if v > 0 else np.nan
        both = both and np.isfinite(sk) and sk > bs
        print(f"   {lbl:<10}{len(s):>7,}{u:>8.1%}{v:>10.1%}{sk:>7.2f}{bs:>11.2f}")
    print(f"     beats its own base in BOTH halves: {'YES' if both else 'NO'}")

    # ---- clustered null --------------------------------------------------
    sub = d[big].reset_index(drop=True)
    ss = ((sub["hi52"] >= HI_CUT) & (sub["vol60"] <= VOL_CUT)).to_numpy()
    if ss.sum() >= 100:
        blk = sub["ticker"].astype(str) + "|" + sub["date"].dt.year.astype(str)
        codes, _ = pd.factorize(blk)
        nb = codes.max() + 1
        up = (sub[f"peak{K}"] >= 2.0).to_numpy(float)
        dn = (sub[f"end{K}"] <= 0.5).to_numpy(float)
        nr = np.bincount(codes, minlength=nb)
        ns = np.bincount(codes, weights=ss.astype(float), minlength=nb)
        ru = np.divide(np.bincount(codes, weights=up, minlength=nb), nr,
                       out=np.zeros(nb), where=nr > 0)
        rd = np.divide(np.bincount(codes, weights=dn, minlength=nb), nr,
                       out=np.zeros(nb), where=nr > 0)
        rng = np.random.default_rng(11)
        null = []
        for _ in range(5000):
            o = rng.permutation(nb)
            take = np.minimum(ns[o], nr)
            c = take.sum()
            if c:
                u2, v2 = (take * ru).sum() / c, (take * rd).sum() / c
                if v2 > 1e-6:
                    null.append(u2 / v2)
        null = np.asarray(null)
        obs = float(up[ss].mean() / max(dn[ss].mean(), 1e-9))
        z = (obs - null.mean()) / null.std(ddof=1)
        pv = (1.0 + float((null >= obs).sum())) / (1.0 + len(null))
        bar = 0.05 / 92
        print(f"\n   CLUSTERED NULL inside the top {TOP_N} (5,000 draws):")
        print(f"     obs {obs:.2f} vs null {null.mean():.2f} +/- "
              f"{null.std(ddof=1):.2f}   z {z:+.2f}   p {pv:.5f}")
        print(f"     bar after 92 trials {bar:.5f} -> "
              f"{'CLEARS' if pv < bar else 'does NOT clear'}")

    # ---- the thing that actually matters: an equity curve ----------------
    print("\n" + "=" * W)
    print(f" EQUITY CURVE — 10 screen names a year vs the top {TOP_N} vs the index")
    print("=" * W)
    sub = sub.copy()
    cs = []
    for c_, dt in zip(sub["close"], sub["date"]):
        try:
            cs.append(B.cost_bar(float(c_), pd.Timestamp(dt))["total"])
        except Exception:
            cs.append(np.nan)
    cs = np.asarray(cs)
    sub["cost"] = np.where(np.isnan(cs), np.nanmedian(cs), cs)
    sub["sig"] = ss
    sub["yr"] = sub["date"].dt.year

    ix = index_series()
    ann = ix.resample("YE").last().pct_change().dropna() + IDX_YIELD
    ann.index = ann.index.year
    years = sorted(y for y in sub["yr"].unique() if y in ann.index)

    rng2 = np.random.default_rng(7)
    def run(mask_col: str, draws: int = 400) -> np.ndarray:
        out = []
        for _ in range(draws):
            eq = 1.0
            for y in years:
                g = sub[(sub["yr"] == y) & (sub[mask_col] if mask_col else True)]
                if len(g) < 3:
                    eq *= (1.0 + ann[y])
                    continue
                pk = g.iloc[rng2.choice(len(g), size=min(10, len(g)),
                                        replace=False)]
                eq *= max(float((pk[f"end{K}"] - pk["cost"]).mean()), 0.01)
            out.append(eq)
        return np.asarray(out)

    sub["allbig"] = True
    scr, base = run("sig"), run("allbig")
    n = len(years)
    ixw = float(np.prod([1.0 + ann[y] for y in years]))
    print(f" {n} years {years[0]}-{years[-1]}, 400 random 10-name draws each,"
          f" costs charged\n")
    print(f"   {'strategy':<34}{'median x':>10}{'CAGR':>9}{'10th':>8}"
          f"{'90th':>8}{'beats index':>13}")
    for lbl, arr in ((f"screen inside the top {TOP_N}", scr),
                     (f"random from the top {TOP_N}", base)):
        print(f"   {lbl:<34}{np.median(arr):>10.1f}"
              f"{np.median(arr) ** (1 / n) - 1:>+9.1%}"
              f"{np.percentile(arr, 10) ** (1 / n) - 1:>+8.1%}"
              f"{np.percentile(arr, 90) ** (1 / n) - 1:>+8.1%}"
              f"{(arr > ixw).mean():>13.1%}")
    print(f"   {'IHSG total return':<34}{ixw:>10.1f}"
          f"{ixw ** (1 / n) - 1:>+9.1%}")
    print("\n   The index row is a TOTAL-return basis at the measured 1.77%")
    print("   top-decile yield. Comparing a total-return basket to a price")
    print("   index is the error A19 records as the worst in this project.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
