#!/usr/bin/env python3
"""H18 — do indicator-conditioned exits beat the fixed price-path ones?

    python3 scripts/exit_indicators.py [--illustrate]

WHAT H17 LEFT OPEN
-------------------
H17 found that a trailing stop armed at +50% adds **+4.13% [+1.87%, +6.33%]**
per cohort over buy-and-hold, winning 74 of the 94 cohorts where it does
anything. It also found the thing it could NOT do: `P(-50%)` was 15.0% against
16.3%, essentially unchanged, because a name that falls straight from entry
never arms a trail. The frontier said a hard stop takes `P(-50%)` to ~0% but
costs 6-8 points of median return, and for a rule selected ON P(2x) that means
giving up the premise.

THE PRE-REGISTERED PREDICTION, WRITTEN BEFORE ANY OF THIS WAS SCORED
----------------------------------------------------------------------
Two claims, both falsifiable, both about mechanism rather than about a
parameter:

  H18a  A volatility-normalised trail (chandelier, k x ATR) beats the fixed
        percentage trail, because the multiplier-cell entry selects for high
        realised vol and a fixed 15% band is therefore a DIFFERENT rule for
        every name it picks — noise-tight on one, half-the-move-loose on the
        next. Predicted: chandelier >= trail on cohort median.

  H18b  An indicator-conditioned stop cuts P(-50%) at a smaller cost in P(2x)
        than a fixed hard stop does, because it can tell trend failure (below
        EMA, stochastic rolled over, distribution volume) from ordinary noise,
        whereas a percentage stop cannot. Predicted: at matched P(-50%), an
        indicator rule keeps more P(2x) than `stop X%`.

  NULL  `NULL random exit` exits at a bar drawn with no reference to the data.
        Predicted: no better than `hold n` at the same holding period. A9
        records `squeeze` being registered as a null and coming back at
        t = +3.55 on two million rows; a catalogue without a null cannot tell
        you the pipeline is manufacturing its own signal.

NEWS IS ABSENT FROM THIS CATALOGUE AND CANNOT BE ADDED
-------------------------------------------------------
There is no point-in-time news archive. `tests/test_news.py` walks the AST of
`spine/` and `features/` and FAILS if either imports the news module, exactly
so that a headline visible today cannot be attached to a 2015 bar. Any exit
rule conditioned on news that appeared in this table would be look-ahead by
construction, and its backtested edge would be an artefact of knowing the
future. The news layer stays where it is: a live read, never a validated
signal. See `reports/exit_indicators.md`.

EVERYTHING ELSE IS AS H17
--------------------------
Pre-holdout only, cohorts monthly, entry one bar after the signal, tie='all'
baskets, costs once per name at 0.56% plus the point-in-time fraksi-harga
half-spread, purged walk-forward, cohort-level moving-block bootstrap.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

from idxbot.spine import exits as X                              # noqa: E402
from idxbot.spine import multiplier as MU                        # noqa: E402

PANEL = os.path.join("data", "spine", "price_panel.parquet")
IND = os.path.join("data", "spine", "indicator_panel.parquet")


def cohort(P, I, as_of, cells, tab, tie="all") -> Dict[str, object]:
    """One date's basket: paths, costs, and the aligned indicator frames."""
    day, M = MU.rank_live(P, as_of, cells, tab)
    if day is None or len(M) < MU.TOP_N:
        return {}
    picks = list(MU.select(M, MU.TOP_N, tie)["ticker"])
    pm = MU.path_map(P, day, picks)
    fm = MU.feature_map(I, day, picks)
    got = [t for t in picks if t in pm]
    if len(got) < MU.TOP_N // 2:
        return {}
    return {"day": day,
            "paths": [pm[t][0] for t in got],
            "costs": [pm[t][1] for t in got],
            "feats": [fm.get(t) for t in got],
            "tickers": got,
            "settles": MU.settle_date(P["date"], day)}


def _table(cohorts, rules, title):
    rows = []
    for name, fn in rules.items():
        S = X.score_cohorts(cohorts, fn)
        if S.empty:
            continue
        rows.append({"rule": name, "median": S["median"].mean(),
                     "mean": S["mean"].mean(), "p2": S["p2"].mean(),
                     "pdn": S["pdn"].mean(), "held": S["held"].mean(),
                     "n": S["n"].mean()})
    F = pd.DataFrame(rows).sort_values("median", ascending=False)
    print(f"\n {title}")
    print(f"   {'rule':<36}{'cohort med':>11}{'mean':>9}{'P(2x)':>8}"
          f"{'P(-50%)':>9}{'days':>6}{'names':>7}")
    for _, r in F.iterrows():
        print(f"   {r['rule']:<36}{r['median']:>+11.1%}{r['mean']:>+9.1%}"
              f"{r['p2']:>8.1%}{r['pdn']:>9.1%}{r['held']:>6.0f}"
              f"{r['n']:>7.1f}")
    return F


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default=PANEL)
    ap.add_argument("--indicators", default=IND)
    ap.add_argument("--start", default="2002-01-01")
    ap.add_argument("--illustrate", action="store_true")
    a = ap.parse_args()

    P = pd.read_parquet(a.panel)
    P["date"] = pd.to_datetime(P["date"])
    P = P.sort_values(["ticker", "date"])
    I = pd.read_parquet(a.indicators)
    I["date"] = pd.to_datetime(I["date"])
    cells, tab = MU.build_cells(P)
    pre_end = P.loc[~P["holdout"].astype(bool), "date"].max()

    print("=" * 86)
    print(" H18 — INDICATOR-CONDITIONED EXITS, PRE-HOLDOUT ONLY")
    print("=" * 86)
    print(f" cells and cohorts end {pre_end.date()}; holdout untouched.")
    print(" news is NOT in this catalogue and cannot be: no point-in-time"
          " archive exists.\n")

    last = pre_end - pd.Timedelta(days=int(X.HORIZON * 1.5))
    cohorts = {}
    for d in pd.date_range(a.start, last, freq="MS"):
        c = cohort(P, I, d, cells, tab)
        if c:
            cohorts[c["day"]] = c
    if len(cohorts) < 40:
        print(" too few cohorts")
        return 1
    nf = np.mean([np.mean([f is not None for f in c["feats"]])
                  for c in cohorts.values()])
    print(f" {len(cohorts)} monthly cohorts, {min(cohorts).date()} -> "
          f"{max(cohorts).date()}; indicators present for {nf:.1%} of names")

    price = X.catalogue()
    ind = X.indicator_catalogue()
    print(f" {len(price)} price-path rules (H17) + {len(ind)} indicator rules")

    F1 = _table(cohorts, price, "FULL-SAMPLE, price-path rules (H17 baseline)")
    F2 = _table(cohorts, ind, "FULL-SAMPLE, indicator rules — OVERFIT VIEW")

    # ---- the three pre-registered questions -------------------------------
    print("\n" + "=" * 86)
    print(" THE PRE-REGISTERED COMPARISONS")
    print("=" * 86)
    best_trail = F1[F1["rule"].str.startswith("trail")].iloc[0]
    chan = F2[F2["rule"].str.startswith("chandelier")].iloc[0]
    null = F2[F2["rule"] == "NULL random exit"]
    print(f" H18a  best fixed trail  {best_trail['rule']:<28}"
          f"{best_trail['median']:>+8.1%} median, P(2x) {best_trail['p2']:.1%}")
    print(f"       best chandelier   {chan['rule']:<28}"
          f"{chan['median']:>+8.1%} median, P(2x) {chan['p2']:.1%}")
    print(f"       -> {'SUPPORTED' if chan['median'] >= best_trail['median'] else 'FAILED'}"
          f" on the full-sample median (walk-forward below decides)")
    if len(null):
        n0 = null.iloc[0]
        near = F1.iloc[(F1['held'] - n0['held']).abs().argsort()[:1]].iloc[0]
        print(f"\n NULL  random exit      {n0['median']:>+8.1%} median, "
              f"{n0['held']:.0f}d, P(2x) {n0['p2']:.1%}")
        print(f"       matched hold      {near['rule']:<28}"
              f"{near['median']:>+8.1%} median, {near['held']:.0f}d, "
              f"P(2x) {near['p2']:.1%}")
        beat = int((F2["median"] > n0["median"]).sum())
        print(f"       {beat}/{len(F2)} indicator rules beat the null on median")

    print("\n H18b  P(-50%) frontier — indicator rules vs hard stops")
    fr = pd.concat([F1.assign(fam="price"), F2.assign(fam="ind")])
    fr = fr.sort_values("pdn")
    best = -np.inf
    print(f"   {'rule':<36}{'fam':>6}{'P(-50%)':>9}{'P(2x)':>8}"
          f"{'cohort med':>12}{'days':>6}")
    for _, r in fr.iterrows():
        if r["p2"] > best:
            best = r["p2"]
            print(f"   {r['rule']:<36}{r['fam']:>6}{r['pdn']:>9.1%}"
                  f"{r['p2']:>8.1%}{r['median']:>+12.1%}{r['held']:>6.0f}")

    # ---- walk-forward over the COMBINED catalogue --------------------------
    print("\n" + "=" * 86)
    print(" WALK-FORWARD over price + indicator rules together")
    print("=" * 86)
    both = dict(price)
    both.update(ind)

    # THE OBJECTIVE DECIDES THE ANSWER, so all three are run and printed.
    # H17 and this study both default to the cohort MEDIAN, which rewards
    # being right often. On an entry rule selected FOR P(2x) that is arguably
    # the wrong target, and the disagreement is the finding.
    print("\n objective sensitivity — the SAME walk-forward, three targets")
    print(f"   {'objective':<10}{'median':>9}{'mean':>9}{'P(2x)':>8}"
          f"{'P(-50%)':>9}{'days':>6}  most-chosen rule")
    alt = {}
    for obj in ("median", "mean", "p2"):
        Wo = X.walk_forward_select(cohorts, both, objective=obj)
        if Wo.empty:
            continue
        alt[obj] = Wo
        print(f"   {obj:<10}{Wo['median'].mean():>+9.2%}"
              f"{Wo['mean'].mean():>+9.2%}{Wo['p2'].mean():>8.1%}"
              f"{Wo['pdn'].mean():>9.1%}{Wo['held'].mean():>6.0f}"
              f"  {Wo['rule'].value_counts().idxmax()}")
    bh = X.score_cohorts(cohorts, price["hold 252"])
    print(f"   {'buy&hold':<10}{bh['median'].mean():>+9.2%}"
          f"{bh['mean'].mean():>+9.2%}{bh['p2'].mean():>8.1%}"
          f"{bh['pdn'].mean():>9.1%}{bh['held'].mean():>6.0f}")

    W = alt.get("median")
    if W is None or W.empty:
        print(" not enough cohorts")
        return 1
    lo, hi = X.bootstrap_cohorts(W, "median")
    blo, bhi = X.bootstrap_cohorts(W.rename(columns={"bh_median": "m"}), "m")
    print(f" {len(W)} scored cohorts, {W.as_of.min().date()} -> "
          f"{W.as_of.max().date()}")
    print(f"   selected rule   cohort median {W['median'].mean():+.2%}  "
          f"95% CI [{lo:+.2%}, {hi:+.2%}]")
    print(f"   buy and hold    cohort median {W['bh_median'].mean():+.2%}  "
          f"95% CI [{blo:+.2%}, {bhi:+.2%}]")
    d = W["median"] - W["bh_median"]
    dlo, dhi = X.bootstrap_cohorts(pd.DataFrame({"d": d}), "d")
    wins, n = int((d > 0).sum()), int((d != 0).sum())
    print(f"   DIFFERENCE                    {d.mean():+.2%}  "
          f"95% CI [{dlo:+.2%}, {dhi:+.2%}]")
    print(f"     better in {wins}/{len(d)} ({wins / max(n, 1):.0%} of the "
          f"{n} that differ), sign test p "
          f"{stats.binomtest(wins, n, 0.5).pvalue:.2e}")
    print(f"   P(-50%)  rule {W['pdn'].mean():.1%}  vs buy-and-hold "
          f"{X.score_cohorts(cohorts, lambda p, F=None: X.hold(p))['pdn'].mean():.1%}")
    print(f"   mean holding period {W['held'].mean():.0f} sessions")
    print("\n   rules the walk-forward chose:")
    for r, k in W["rule"].value_counts().head(8).items():
        print(f"     {k:>4}x  {r}")

    # H17's incumbent, scored on the identical cohorts, as the thing to beat
    inc = X.score_cohorts(cohorts, price["trail 15% armed +50%"])
    W2 = W.merge(inc[["as_of", "median"]].rename(
        columns={"median": "inc"}), on="as_of", how="left")
    di = W2["median"] - W2["inc"]
    ilo, ihi = X.bootstrap_cohorts(pd.DataFrame({"d": di}), "d")
    iw, ino = int((di > 0).sum()), int((di != 0).sum())
    print(f"\n   VS H17's INCUMBENT (trail 15% armed +50%, same cohorts):")
    print(f"     difference {di.mean():+.2%}  95% CI [{ilo:+.2%}, {ihi:+.2%}]"
          f"   better in {iw}/{ino} that differ"
          + (f", sign test p {stats.binomtest(iw, ino, 0.5).pvalue:.3f}"
             if ino else ""))

    if a.illustrate:
        print("\n" + "=" * 86)
        print(" ILLUSTRATION — 2025-08-25, ALREADY-SPENT holdout (H16). "
              "Certifies nothing.")
        print("=" * 86)
        c = cohort(P, I, pd.Timestamp("2025-08-25"), cells, tab)
        if c:
            pick = W["rule"].value_counts().idxmax()
            for lab, fn in (("buy and hold 252", price["hold 252"]),
                            ("H17: trail 15% armed +50%",
                             price["trail 15% armed +50%"]),
                            (f"H18 walk-forward pick: {pick}", both[pick])):
                D = X.apply_rule(c["paths"], c["costs"], fn, c["feats"])
                D = D[np.isfinite(D["net"])]
                print(f"   {lab:<44}mean {D['net'].mean():>+8.1%}  "
                      f"median {D['net'].median():>+8.1%}  "
                      f"2x {int((D['gross'] >= 1).sum())}  "
                      f"-50% {int((D['gross'] <= -.5).sum())}  "
                      f"held {D['held'].mean():.0f}d")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
