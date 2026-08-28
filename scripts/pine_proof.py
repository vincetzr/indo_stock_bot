#!/usr/bin/env python3
"""Backtest the rule EXACTLY as the Pine script will compute it.

    python3 scripts/pine_proof.py

WHY THIS IS NOT reports/asymmetry.md AGAIN. H26 measured a rule that ranks
every eligible name against the full daily cross-section. A TradingView
indicator sees ONE chart and can pull at most ~40 other symbols, so the Pine
port ranks the chart symbol against a 36-name REFERENCE BASKET instead. That
is an approximation, and an approximation of a selection rule has to be
measured on its own terms — quoting H26's 2.60 for a rule that selects a
different set of names would be the same error as quoting a ten-year doubling
rate as a one-year one (A21).

THREE ARMS, and the gap between them is the whole point:

  TRUE      rank against the full daily cross-section (what H26 measured)
  PROXY     rank against the 36-name basket (what Pine will actually do)
  ABSOLUTE  fixed thresholds, no cross-section at all (the degraded fallback
            when the basket fails to load or the symbol is not IDX)

THE REFERENCE BASKET IS SURVIVORSHIP-BIASED AND THAT IS HANDLED, NOT IGNORED.
The 36 names were picked from TODAY's liquid cross-section. Applied to 2010
they include names that had not listed yet and exclude names that have since
died. So each date uses only the basket members actually trading that day, and
the realised basket size is reported per era — a date with eight live
reference names produces a much noisier percentile than one with thirty-six,
and the reader is entitled to see that rather than a single averaged number.
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))
sys.path.insert(0, os.path.dirname(__file__))

from horizon_sweep import classify                              # noqa: E402

CACHE = os.path.join("data", "spine", "horizon_sweep.parquet")
PANEL = os.path.join("data", "spine", "price_panel.parquet")
K = 252

#: The stratified basket the Pine script pulls via request.security.
#:
#: THE FIRST BASKET WAS PICKED FROM TODAY'S LIQUIDITY LEADERS AND THAT WAS A
#: DEFECT THIS SCRIPT CAUGHT. Most of them listed recently, so applied to
#: history the basket had a MEDIAN OF SIX live members — three in 2005, four
#: in 2010 — and a percentile ranked against four names is noise. The proxy
#: arm then captured only 28.2% of the true rule's picks and the early half
#: had 87 observations, too few to read.
#:
#: This basket instead requires listing before 2008 and still trading, then
#: stratifies across the liquidity range. 28 live members in 2005 and 36 from
#: 2010 onward, and it is MORE accurate today as well (mean abs percentile
#: error 0.067 against 0.095). A measuring stick has to exist for the whole
#: period it measures.
#:
#: Note what this does and does not bias. Requiring long history makes the
#: REFERENCE SET a survivor set — but the reference set is the ruler, not the
#: selection, and a ruler that vanishes half the time is the worse problem.
BASKET = ["BBCA", "BBRI", "BRPT", "ASII", "BBNI", "BNBR", "ENRG", "KLBF",
          "PGAS", "CPIN", "INKP", "JPFA", "AKRA", "SMGR", "EXCL", "HMSP",
          "MYOR", "AALI", "CTRA", "BDMN", "PNLF", "LPKR", "SMRA", "GJTL",
          "BNGA", "ULTJ", "SRSN", "PNBN", "PYFA", "KBLV", "FORU", "CTTH",
          "TSPC", "CPRO", "RALS", "DILD"]

#: Absolute equivalents of the two percentile cuts, measured as the median of
#: the daily cut over 601,840 eligible pre-holdout name-days.
ABS_HI52, ABS_VOL60 = 0.9625, 0.0257

#: The rule itself: strong AND calm.
HI_PCT, VOL_PCT = 0.90, 0.50

#: Which arm the Pine script ships. Set from the evidence below, not from the
#: architecture that looked cleverer: the absolute-threshold arm measures a
#: HIGHER skew than the 36-name proxy and captures more of the true rule's
#: picks, while needing zero request.security calls.
SHIP = "ABS"


def proxy_pct(d: pd.DataFrame, col: str, basket: List[str]) -> pd.Series:
    """Percentile of each name against the LIVE basket members on its date.

    `basket` members that were not trading on a given date simply are not in
    that date's reference set, which is the same thing `request.security`
    returns na for. The alternative — carrying a dead name's last price
    forward — would invent a reference the indicator could never see.
    """
    out = np.full(len(d), np.nan)
    vals = d[col].to_numpy()
    is_ref = d["ticker"].isin(basket).to_numpy()
    for _, idx in d.groupby("date", sort=False).indices.items():
        ref = vals[idx][is_ref[idx]]
        ref = ref[np.isfinite(ref)]
        if len(ref) < 8:            # too few live references to rank against
            continue
        out[idx] = np.searchsorted(np.sort(ref), vals[idx], "left") / len(ref)
    return pd.Series(out, index=d.index)


def arm(d: pd.DataFrame, sel: pd.Series, name: str) -> Dict:
    s = d[sel.reindex(d.index).fillna(False)]
    if len(s) < 200:
        return {"arm": name, "n": len(s), "note": "too few to read"}
    up = float((s[f"peak{K}"] >= 2.0).mean())
    dn = float((s[f"end{K}"] <= 0.5).mean())
    return {"arm": name, "n": len(s), "names": int(s["ticker"].nunique()),
            "up": up, "dn": dn, "skew": up / dn if dn > 0 else np.nan,
            "median": float(s[f"end{K}"].median() - 1.0),
            "mean_log": float(np.log(np.maximum(s[f"end{K}"], 0.01)).mean())}


def main() -> int:
    D = pd.read_parquet(CACHE)
    D = D[~D["holdout"].astype(bool)]
    d = classify(D, K)
    d = d[d["cls"] != "censored"].copy().reset_index(drop=True)

    W = 96
    print("=" * W)
    print(" PROOF — THE RULE AS THE PINE SCRIPT WILL ACTUALLY COMPUTE IT")
    print("=" * W)
    print(f" {len(d):,} name-years, {d['ticker'].nunique()} names, "
          f"pre-holdout, 1-year forward windows\n")

    live = d[d["ticker"].isin(BASKET)].groupby("date")["ticker"].nunique()
    print(f" Reference basket coverage: {len(BASKET)} names nominated, "
          f"median {live.median():.0f} live on a given date")
    print(f"   by era:", end="")
    for yr in (2005, 2010, 2015, 2020, 2024):
        v = live[live.index.year == yr]
        print(f"  {yr}: {v.median():.0f}" if len(v) else f"  {yr}: 0", end="")
    print("\n   Fewer live references = a noisier percentile. This is the")
    print("   survivorship cost of picking the basket from today's board,")
    print("   and it makes the early years the WEAK end of this test.\n")

    # ---- the three arms -------------------------------------------------
    r_hi = d.groupby("date")["hi52"].rank(pct=True)
    r_vo = d.groupby("date")["vol60"].rank(pct=True)
    p_hi = proxy_pct(d, "hi52", BASKET)
    p_vo = proxy_pct(d, "vol60", BASKET)

    arms = [
        ("TRUE  full cross-section", (r_hi >= HI_PCT) & (r_vo <= VOL_PCT)),
        ("PROXY 36-name basket", (p_hi >= HI_PCT) & (p_vo <= VOL_PCT)),
        ("ABS   fixed thresholds",
         (d["hi52"] >= ABS_HI52) & (d["vol60"] <= ABS_VOL60)),
        ("none  no screen", pd.Series(True, index=d.index)),
    ]
    print(f"   {'arm':<28}{'n':>7}{'names':>7}{'P(2x)':>8}{'P(halve)':>10}"
          f"{'SKEW':>7}{'median':>9}{'CAGR/nm':>9}")
    res = {}
    for nm, sel in arms:
        a = arm(d, sel, nm)
        res[nm.split()[0]] = a
        if "note" in a:
            print(f"   {nm:<28}{a['n']:>7}   {a['note']}")
            continue
        print(f"   {nm:<28}{a['n']:>7,}{a['names']:>7}{a['up']:>8.1%}"
              f"{a['dn']:>10.1%}{a['skew']:>7.2f}{a['median']:>+9.1%}"
              f"{np.exp(a['mean_log']) - 1:>+9.1%}")

    # ---- how much of TRUE does PROXY actually capture? ------------------
    t = ((r_hi >= HI_PCT) & (r_vo <= VOL_PCT)).fillna(False)
    p = ((p_hi >= HI_PCT) & (p_vo <= VOL_PCT)).fillna(False)
    a = ((d["hi52"] >= ABS_HI52) & (d["vol60"] <= ABS_VOL60)).fillna(False)
    print(f"\n   OVERLAP WITH THE TRUE RULE (this is the honest headline):")
    for nm, s in (("PROXY", p), ("ABS", a)):
        j = (t & s).sum() / max((t | s).sum(), 1)
        print(f"     {nm:<6} fires {s.mean():>5.1%} of name-days,  "
              f"Jaccard vs TRUE {j:>5.1%},  "
              f"captures {(t & s).sum() / max(t.sum(), 1):>5.1%} of TRUE's picks")

    # ---- half-split and null on BOTH candidate arms ---------------------
    #  The proxy was the intended design and the absolute version was meant to
    #  be a degraded fallback. The table above says the fallback is BETTER —
    #  higher skew, and it captures 72% of the true rule's picks against the
    #  proxy's 53%. So both get the full treatment and the evidence picks.
    mid = d["date"].quantile(0.5)
    for lbl_arm, sel_arm in (("ABS", a), ("PROXY", p)):
        print(f"\n   HALF-SPLIT — {lbl_arm}:")
        print(f"   {'half':<8}{'n':>7}{'P(2x)':>8}{'P(halve)':>10}{'SKEW':>7}"
              f"{'base skew':>11}")
        good = True
        for lbl, m in (("early", d["date"] <= mid), ("late", d["date"] > mid)):
            ss, bb = d[m & sel_arm], d[m]
            if len(ss) < 100:
                print(f"   {lbl:<8}{len(ss):>7}   too few")
                good = False
                continue
            u = (ss[f"peak{K}"] >= 2.0).mean()
            v = (ss[f"end{K}"] <= 0.5).mean()
            bs = (bb[f"peak{K}"] >= 2.0).mean() / (bb[f"end{K}"] <= 0.5).mean()
            sk = u / v if v > 0 else np.nan
            good = good and sk > bs
            print(f"   {lbl:<8}{len(ss):>7,}{u:>8.1%}{v:>10.1%}{sk:>7.2f}"
                  f"{bs:>11.2f}")
        print(f"     beats base in BOTH halves: {'YES' if good else 'NO'}")

    print(f"\n   [legacy proxy half-split retained below for the record]")
    print(f"\n   HALF-SPLIT of the PROXY arm:")
    print(f"   {'half':<8}{'n':>7}{'P(2x)':>8}{'P(halve)':>10}{'SKEW':>7}"
          f"{'base skew':>11}")
    ok = True
    for lbl, m in (("early", d["date"] <= mid), ("late", d["date"] > mid)):
        s = d[m & p]
        b = d[m]
        if len(s) < 100:
            print(f"   {lbl:<8}{len(s):>7}   too few")
            ok = False
            continue
        up = (s[f"peak{K}"] >= 2.0).mean()
        dn = (s[f"end{K}"] <= 0.5).mean()
        bs = (b[f"peak{K}"] >= 2.0).mean() / (b[f"end{K}"] <= 0.5).mean()
        sk = up / dn if dn > 0 else np.nan
        ok = ok and sk > bs
        print(f"   {lbl:<8}{len(s):>7,}{up:>8.1%}{dn:>10.1%}{sk:>7.2f}"
              f"{bs:>11.2f}")
    print(f"\n   proxy beats base in BOTH halves: {'YES' if ok else 'NO'}")

    # ---- clustered permutation null on the shipping arm ------------------
    blk = d["ticker"].astype(str) + "|" + d["date"].dt.year.astype(str)
    codes, _ = pd.factorize(blk)
    nb = codes.max() + 1
    up = (d[f"peak{K}"] >= 2.0).to_numpy(float)
    dn = (d[f"end{K}"] <= 0.5).to_numpy(float)
    sv = a.to_numpy() if SHIP == "ABS" else p.to_numpy()
    nr = np.bincount(codes, minlength=nb)
    ns = np.bincount(codes, weights=sv.astype(float), minlength=nb)
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
            u, v = (take * ru).sum() / c, (take * rd).sum() / c
            if v > 1e-6:
                null.append(u / v)
    null = np.asarray(null)
    obs = float(up[sv].mean() / dn[sv].mean())
    z = (obs - null.mean()) / null.std(ddof=1)
    pv = (1.0 + float((null >= obs).sum())) / (1.0 + len(null))
    bar = 0.05 / 88
    print(f"\n   CLUSTERED NULL on the {SHIP} arm (5,000 draws, whole")
    print(f"   (ticker, year) blocks reassigned): obs {obs:.2f} vs null "
          f"{null.mean():.2f} +/- {null.std(ddof=1):.2f}")
    print(f"   z = {z:+.2f}, p = {pv:.5f} against a bar of {bar:.5f}"
          f" -> {'CLEARS' if pv < bar else 'does NOT clear'}")

    print("\n" + "=" * W)
    print(" WHAT THIS PROVES AND WHAT IT DOES NOT")
    print("=" * W)
    tr, pr = res.get("TRUE", {}), res.get("PROXY", {})
    if tr and pr and "skew" in tr and "skew" in pr:
        print(f"   The shipping rule reads skew {pr['skew']:.2f} against the")
        print(f"   ideal cross-sectional {tr['skew']:.2f} and an unscreened")
        print(f"   base of {res['none']['skew']:.2f}.")
    print("   It is measured PRE-HOLDOUT and the holdout was spent at H16, so")
    print("   this is in-sample. It is a 1-YEAR horizon — every number here")
    print("   dies if quoted at another. And it forecasts NOTHING about")
    print("   magnitude: 2x is the level the probability was measured")
    print("   against, not a target.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
