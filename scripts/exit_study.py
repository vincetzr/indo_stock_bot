#!/usr/bin/env python3
"""Fit and score exit rules for the multiplier-cell entry, PRE-HOLDOUT ONLY.

    python3 scripts/exit_study.py            # the full walk-forward
    python3 scripts/exit_study.py --illustrate  # ... plus the 2025 cohort

WHAT H16 ESTABLISHED, AND WHAT THIS ANSWERS
---------------------------------------------
The entry rule picked ten names on 2025-08-25 that reached a mean PEAK of
+102.2% and realised +15.1%. The names worked; there was no exit rule.

Choosing one on those ten names would be worthless — the same cohort makes a
six-month hold look like +41.4% and a nine-month hold like -14.6%. So every
number below comes from cohorts ending 2023-08, inside the pre-holdout window,
selected walk-forward: the rule used at cohort *t* is chosen only on cohorts
before *t*.

THE HOLDOUT IS ALREADY SPENT (H16) and cannot certify anything now. The 2025
cohort appears under --illustrate, after the fact, labelled as illustration.

WHICH TEN NAMES — read `spine/multiplier.py` before trusting any number here
------------------------------------------------------------------------------
The first version of this script re-implemented the entry rule instead of
sharing it, and on 2025-08-25 it drew a different basket from H16's (IMPC where
H16 had MERI) and therefore a different buy-and-hold return, +26.3% against
+15.1%. The cause was not a bug in either: the rule's top ten is drawn from a
much larger group of names carrying the identical cell score, so "top ten" was
being decided by frame order. The selection now lives in one module, the
default holds the whole tied group so nothing arbitrary decides membership,
and `scripts/tie_sensitivity.py` measures what the arbitrariness was worth.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

from idxbot.spine import exits as X                            # noqa: E402
from idxbot.spine import multiplier as MU                      # noqa: E402

PANEL = os.path.join("data", "spine", "price_panel.parquet")


def cohort(P: pd.DataFrame, as_of: pd.Timestamp, cells, tab,
           tie: object = "all") -> Dict[str, object]:
    """One date's basket, with forward normalised price paths.

    Entry is the NEXT bar's close — see ``multiplier.paths``.
    """
    day, M = MU.rank_live(P, as_of, cells, tab)
    if day is None or len(M) < MU.TOP_N:
        return {}
    picks = MU.select(M, MU.TOP_N, tie)
    paths, costs = MU.paths(P, day, list(picks["ticker"]))
    if len(paths) < MU.TOP_N // 2:
        return {}
    # the same-window universe, for a benchmark
    uni, fut = [], P[P["date"] > day]
    for t in M["ticker"].head(400):
        g = fut[fut["ticker"] == t].sort_values("date")
        if len(g) < 2:
            continue
        px = g["adj_close"].astype(float).to_numpy()
        if px[0] > 0:
            uni.append(px[min(X.HORIZON, len(px) - 1)] / px[0] - 1.0)
    return {"day": day, "paths": paths, "costs": costs,
            "n_tied": MU.tie_report(M).get("n_tied_at_cut", np.nan),
            "settles": MU.settle_date(P["date"], day),
            "uni_med": float(np.median(uni)) if uni else np.nan}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default=PANEL)
    ap.add_argument("--start", default="2002-01-01")
    ap.add_argument("--illustrate", action="store_true",
                    help="also show the already-spent 2025 cohort")
    ap.add_argument("--tie", default="all", choices=["all", "first"],
                    help="'all' holds the whole tied group (deterministic); "
                         "'first' reproduces the old arbitrary top-10")
    a = ap.parse_args()

    P = pd.read_parquet(a.panel)
    P["date"] = pd.to_datetime(P["date"])
    P = P.sort_values(["ticker", "date"])
    cells, tab = MU.build_cells(P)
    pre_end = P.loc[~P["holdout"].astype(bool), "date"].max()
    print("=" * 84)
    print(" EXIT RULES for the multiplier-cell entry — PRE-HOLDOUT ONLY")
    print("=" * 84)
    print(f" cells and cohorts both end {pre_end.date()}; the holdout is not "
          f"touched here.\n")

    # cohorts need a full horizon inside the pre-holdout window
    last = pre_end - pd.Timedelta(days=int(X.HORIZON * 1.5))
    days = pd.date_range(a.start, last, freq="MS")
    cohorts = {}
    for d in days:
        c = cohort(P, d, cells, tab, a.tie)
        if c:
            cohorts[c["day"]] = c
    nb = np.mean([len(c["paths"]) for c in cohorts.values()]) if cohorts else 0
    nt = np.nanmedian([c["n_tied"] for c in cohorts.values()]) if cohorts else np.nan
    print(f" {len(cohorts)} monthly cohorts, "
          f"{min(cohorts).date()} -> {max(cohorts).date()}")
    print(f" tie handling {a.tie!r}: mean basket {nb:.1f} names; "
          f"median {nt:.0f} names share the 10th-place score\n")
    if len(cohorts) < 40:
        print(" too few cohorts to select a rule")
        return 1

    rules = X.catalogue()
    print(f" FULL-SAMPLE view of {len(rules)} candidate rules — "
          f"THIS IS THE OVERFIT VIEW, shown so the")
    print(" walk-forward below can be compared against it.\n")
    rows = []
    for name, fn in rules.items():
        S = X.score_cohorts(cohorts, fn)
        if S.empty:
            continue
        rows.append({"rule": name, "median": S["median"].mean(),
                     "mean": S["mean"].mean(), "p2": S["p2"].mean(),
                     "pdn": S["pdn"].mean(), "held": S["held"].mean()})
    F = pd.DataFrame(rows).sort_values("median", ascending=False)
    print(f"   {'rule':<38}{'cohort med':>11}{'mean':>9}{'P(2x)':>8}"
          f"{'P(-50%)':>9}{'days':>7}")
    for _, r in F.head(10).iterrows():
        print(f"   {r['rule']:<38}{r['median']:>+11.1%}{r['mean']:>+9.1%}"
              f"{r['p2']:>8.1%}{r['pdn']:>9.1%}{r['held']:>7.0f}")
    bh = F[F["rule"] == "hold 252"].iloc[0]
    print(f"   {'-- buy and hold 252 --':<38}{bh['median']:>+11.1%}"
          f"{bh['mean']:>+9.1%}{bh['p2']:>8.1%}{bh['pdn']:>9.1%}"
          f"{bh['held']:>7.0f}")

    # THERE IS NO SINGLE BEST RULE — there is a frontier, and cutting the
    # left tail costs upside every time. Printing only the median-maximising
    # rule hides that an exit which halves P(-50%) also roughly halves P(2x),
    # which for a rule selected ON P(2x) is close to giving up the premise.
    print("\n   the trade-off, not a winner: rules on the P(-50%) frontier")
    Fr = F.sort_values("pdn")
    best = -np.inf
    print(f"   {'rule':<38}{'P(-50%)':>9}{'P(2x)':>8}{'cohort med':>12}"
          f"{'days':>7}")
    for _, r in Fr.iterrows():
        if r["p2"] > best:                     # undominated on (P(-50%), P(2x))
            best = r["p2"]
            print(f"   {r['rule']:<38}{r['pdn']:>9.1%}{r['p2']:>8.1%}"
                  f"{r['median']:>+12.1%}{r['held']:>7.0f}")

    print("\n" + "=" * 84)
    print(" WALK-FORWARD — the rule at each cohort chosen only on earlier ones")
    print("=" * 84)
    W = X.walk_forward_select(cohorts, rules)
    if W.empty:
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
    print(f"   DIFFERENCE                    {d.mean():+.2%}  "
          f"95% CI [{dlo:+.2%}, {dhi:+.2%}]")
    # THE MEAN AND THE SIGN DISAGREE AND BOTH MUST BE PRINTED. A stop can only
    # help in the left tail, so it wins rarely and by a lot while losing often
    # and by a little. Quoting the mean alone would sell it as a rule that
    # usually helps, which it is not.
    wins = int((d > 0).sum())
    n = int((d != 0).sum())
    from scipy import stats as _st
    sgn = _st.binomtest(wins, n, 0.5).pvalue if n else np.nan
    print(f"     better in {wins}/{len(d)} cohorts "
          f"({wins / max(n, 1):.0%} of the {n} that differ), "
          f"sign test p {sgn:.3f}")
    print(f"     median difference {d.median():+.2%}   "
          f"mean when it wins {d[d > 0].mean():+.2%}   "
          f"mean when it loses {d[d < 0].mean():+.2%}")
    print(f"   P(-50%)  rule {W['pdn'].mean():.1%}  vs buy-and-hold "
          f"{X.score_cohorts(cohorts, lambda p: X.hold(p, X.HORIZON))['pdn'].mean():.1%}")
    print(f"   mean holding period {W['held'].mean():.0f} sessions")
    print("\n   rules the walk-forward actually chose:")
    for r, n in W["rule"].value_counts().head(6).items():
        print(f"     {n:>4}x  {r}")

    if a.illustrate:
        print("\n" + "=" * 84)
        print(" ILLUSTRATION ONLY — the 2025-08-25 cohort, on ALREADY-SPENT")
        print(" holdout data (H16). This certifies nothing; it is shown "
              "because it is\n the cohort the question was asked about.")
        print("=" * 84)
        c = cohort(P, pd.Timestamp("2025-08-25"), cells, tab, a.tie)
        if c:
            pick = W["rule"].value_counts().idxmax()
            for lab, fn in (("buy and hold 252", lambda p: X.hold(p, 252)),
                            (f"walk-forward's most-chosen: {pick}",
                             rules[pick])):
                D = X.apply_rule(c["paths"], c["costs"], fn)
                D = D[np.isfinite(D["net"])]
                print(f"   {lab:<46}mean {D['net'].mean():>+8.1%}  "
                      f"median {D['net'].median():>+8.1%}  "
                      f"2x {int((D['gross'] >= 1).sum())}  "
                      f"-50% {int((D['gross'] <= -.5).sum())}  "
                      f"held {D['held'].mean():.0f}d")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
