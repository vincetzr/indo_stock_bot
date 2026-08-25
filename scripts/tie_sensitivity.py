#!/usr/bin/env python3
"""How much of the multiplier-cell entry's result is the TIE-BREAK?

    python3 scripts/tie_sensitivity.py             # pre-holdout cohorts
    python3 scripts/tie_sensitivity.py --illustrate  # ... plus 2025-08-25

WHY THIS EXISTS
----------------
The entry rule scores each liquid name by the historical P(2x) of the
(price band, liquidity quintile, 60-day-vol quintile) cell it occupies, sorts,
and takes the top ten. There are at most 125 cells and roughly 800 live names,
so the scores are massively tied: on 2025-08-25 only four distinct (p2, p5)
pairs existed in the top thirty and **seventeen names shared the tenth-place
value.** Which ten of those seventeen you get was decided by frame order.

That is how the same rule produced +15.1% in H16 and +26.3% in
``exit_study.py`` for the identical buy-and-hold exit: H16 drew MERI (−54.8%),
exit_study drew IMPC. Neither implementation had a bug.

So this measures the size of that arbitrariness rather than arguing about it:

* draw R independent random tie-breaks per cohort and report the spread of the
  outcome. This is variance that is NOT in any interval quoted so far.
* score the deterministic alternative — hold the whole tied group — which is
  not a better tie-break but the absence of one, and is the only version two
  implementations are obliged to agree on.

The null being tested is not "is the rule good". It is **"is the rule a rule"**.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

from idxbot.spine import exits as X                                # noqa: E402
from idxbot.spine import multiplier as MU                          # noqa: E402

PANEL = os.path.join("data", "spine", "price_panel.parquet")


def name_nets(P, day, tickers, rule) -> pd.Series:
    """Net return per name under one exit rule — computed ONCE per cohort.

    The whole study rescores the same names under hundreds of different
    baskets, so the path and the rule are evaluated per name and every basket
    is then a subset selection over this series. Recomputing per basket made
    the job an hour long for no different answer.
    """
    m = MU.path_map(P, day, list(tickers))
    out = {}
    for t, (p, c) in m.items():
        r, _ = rule(np.asarray(p, dtype=float))
        if np.isfinite(r):
            out[t] = r - c
    return pd.Series(out, dtype=float)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default=PANEL)
    ap.add_argument("--start", default="2005-01-01")
    ap.add_argument("--draws", type=int, default=40)
    ap.add_argument("--illustrate", action="store_true")
    a = ap.parse_args()

    P = pd.read_parquet(a.panel)
    P["date"] = pd.to_datetime(P["date"])
    P = P.sort_values(["ticker", "date"])
    cells, tab = MU.build_cells(P)
    pre_end = P.loc[~P["holdout"].astype(bool), "date"].max()
    hold252 = (lambda p: X.hold(p, 252))

    print("=" * 84)
    print(" IS THE MULTIPLIER-CELL ENTRY A RULE? — tie-break sensitivity")
    print("=" * 84)
    print(f" cells built on pre-holdout rows only, ending {pre_end.date()}\n")

    last = pre_end - pd.Timedelta(days=int(X.HORIZON * 1.5))
    rows, ties = [], []
    for d in pd.date_range(a.start, last, freq="MS"):
        day, M = MU.rank_live(P, d, cells, tab)
        if day is None or len(M) < MU.TOP_N:
            continue
        tr = MU.tie_report(M)
        if tr:
            ties.append(tr)
        allp = list(MU.select(M, tie="all")["ticker"])
        net = name_nets(P, day, allp, hold252)
        if len(net) < MU.TOP_N // 2:
            continue

        def med(names):
            s = net.reindex([n for n in names if n in net.index]).dropna()
            return float(s.median()) if len(s) >= 3 else np.nan

        draws = [med(MU.select(M, tie=s)["ticker"]) for s in range(a.draws)]
        draws = [d for d in draws if np.isfinite(d)]
        if len(draws) < 5:
            continue
        rows.append({"as_of": day, "n_all": len(net), "n_tied": tr.get(
            "n_tied_at_cut", np.nan),
            "all": med(allp),
            "first": med(MU.select(M, tie="first")["ticker"]),
            "draw_mean": float(np.mean(draws)),
            "draw_sd": float(np.std(draws, ddof=1)),
            "draw_lo": float(np.percentile(draws, 5)),
            "draw_hi": float(np.percentile(draws, 95))})
    R = pd.DataFrame(rows)
    if R.empty:
        print(" no cohorts")
        return 1

    T = pd.DataFrame(ties)
    print(f" {len(R)} monthly cohorts, {R.as_of.min().date()} -> "
          f"{R.as_of.max().date()}, {a.draws} tie-breaks each\n")
    print(" THE TIE STRUCTURE ITSELF")
    print(f"   cohorts where the top-10 cut falls inside a tie   "
          f"{T['arbitrary'].mean():.0%}")
    print(f"   names sharing the 10th-place score, median        "
          f"{T['n_tied_at_cut'].median():.0f}   "
          f"(90th pct {T['n_tied_at_cut'].quantile(.9):.0f}, "
          f"max {T['n_tied_at_cut'].max():.0f})")
    print(f"   distinct scores in the top 30, median             "
          f"{T['distinct_scores_top30'].median():.0f} of 30")

    print("\n THE COST OF THAT, IN RETURN")
    print(f"   within-cohort sd across tie-breaks, mean          "
          f"{R['draw_sd'].mean():>7.2%}")
    print(f"   90% of tie-breaks span, mean                      "
          f"{(R['draw_hi'] - R['draw_lo']).mean():>7.2%}")
    lo, hi = X.bootstrap_cohorts(R.rename(columns={"draw_mean": "m"}), "m")
    print(f"   cohort median, averaged over tie-breaks           "
          f"{R['draw_mean'].mean():>+7.2%}  95% CI "
          f"[{lo:+.2%}, {hi:+.2%}]")
    # State the two uncertainties side by side rather than asserting a ratio.
    # They are not the same kind of thing: the sampling interval is the error
    # on the 211-cohort AVERAGE, the tie-break sd is the spread WITHIN one
    # cohort. What matters is that the second exists at all and appears in no
    # interval anyone has quoted from this rule.
    print(f"   for scale: sampling half-width on the average is "
          f"±{(hi - lo) / 2:.2%};")
    print(f"   the tie-break sd of {R['draw_sd'].mean():.2%} is per COHORT and "
          f"is additional to it.")

    print("\n THE DETERMINISTIC ALTERNATIVE — hold the whole tied group")
    alo, ahi = X.bootstrap_cohorts(R.rename(columns={"all": "m"}), "m")
    flo, fhi = X.bootstrap_cohorts(R.rename(columns={"first": "m"}), "m")
    print(f"   tie='all'    {R['all'].mean():>+8.2%}  95% CI "
          f"[{alo:+.2%}, {ahi:+.2%}]   basket "
          f"{R['n_all'].mean():.0f} names")
    print(f"   tie='first'  {R['first'].mean():>+8.2%}  95% CI "
          f"[{flo:+.2%}, {fhi:+.2%}]   basket 10 names (arbitrary)")
    d = R["all"] - R["draw_mean"]
    dlo, dhi = X.bootstrap_cohorts(pd.DataFrame({"d": d}), "d")
    print(f"   difference   {d.mean():>+8.2%}  95% CI "
          f"[{dlo:+.2%}, {dhi:+.2%}]")

    if a.illustrate:
        print("\n" + "=" * 84)
        print(" ILLUSTRATION — 2025-08-25, on ALREADY-SPENT holdout data (H16).")
        print(" This certifies nothing. It is the cohort the question was "
              "asked about.")
        print("=" * 84)
        day, M = MU.rank_live(P, pd.Timestamp("2025-08-25"), cells, tab)
        tr = MU.tie_report(M)
        print(f"   {tr['n_tied_at_cut']} names share the 10th-place score "
              f"{tr['cut_value']}; the top 30 holds "
              f"{tr['distinct_scores_top30']} distinct scores")
        allp = list(MU.select(M, tie="all")["ticker"])
        net = name_nets(P, day, allp, hold252)
        draws = [float(net.reindex(MU.select(M, tie=s)["ticker"])
                       .dropna().mean()) for s in range(500)]
        q = np.percentile(draws, [5, 25, 50, 75, 95])
        print("   500 random tie-breaks, cohort MEAN net over a year:")
        print(f"     p5 {q[0]:+.1%}   p25 {q[1]:+.1%}   median {q[2]:+.1%}   "
              f"p75 {q[3]:+.1%}   p95 {q[4]:+.1%}")
        print(f"     sd {np.std(draws, ddof=1):.1%}   "
              f"min {min(draws):+.1%}   max {max(draws):+.1%}")
        print(f"   tie='all' ({len(net)} names)  mean "
              f"{net.mean():+.1%}  median {net.median():+.1%}")
        pct = float(np.mean(np.array(draws) <= net.mean()))
        print(f"   the deterministic basket sits at the {pct:.0%} percentile "
              f"of the arbitrary ones")
        for lab, names in (("H16 drew", ["ARCI", "BLOG", "DKFT", "ENRG",
                                         "FORE", "INET", "KRAS", "KRYA",
                                         "MBMA", "MERI"]),
                           ("exit_study drew", ["ARCI", "BLOG", "DKFT", "ENRG",
                                                "FORE", "IMPC", "INET", "KRAS",
                                                "KRYA", "MBMA"])):
            s = net.reindex(names).dropna()
            if len(s):
                print(f"   {lab:<18}mean {s.mean():+.1%}  "
                      f"({len(s)} of {len(names)} priced)")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
