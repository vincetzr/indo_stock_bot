#!/usr/bin/env python3
"""Push the verified mechanism to its maximum, and report what survives selection.

Result 71 established the one thing on IDX that beats holding out of sample: rank
every liquid name on momentum, hold the strongest few, and gate each holding on a
trend rule. It was run at ONE setting - 120-day momentum, top 8, rebalance 10 -
because that setting was chosen for consistency in Part XV, not for return.

This searches the whole grid for the maximum, and it searches the gate too,
including gates that Result 71 could not test because they combine two rules:

    both   trade a name only while it is above its 20-day average AND its
           bounded-lag reversal state is long - the strict version
    either above the 20-day average OR reversal-long - the permissive version

Why two selectors are reported
------------------------------
Picking the best of 200-odd configurations on the same windows you then quote is
how backtests lie. So the honest number here is produced by **nested** selection:
inside each fold the configuration is chosen using only data before that fold
begins, and the return quoted is what that choice then earned. The best-by-
hindsight row is printed beside it, labelled as a ceiling, so the gap between
"what you could have picked" and "what you would have picked" is visible instead
of hidden.

Two selectors are run because Result 72 showed the choice matters more than the
grid does:

    expanding   score on everything before the fold - what Part XIII did, and
                what Result 72 showed is dominated by the pre-2009 survivor slice
    trailing    score on the last five years before the fold only

Neither peeks. The difference between them is a real design decision, and it is
worth more than any parameter in the grid.

    python3 scripts/maximize_book.py [--quick]
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

from optimize_consistent import load_wide, score_curve      # noqa: E402
from turn_book import ma_gate, reversal_gate, simulate      # noqa: E402

TRAIL_YEARS = 5


def build_gates(close: pd.DataFrame, quick: bool) -> Dict[str, Optional[np.ndarray]]:
    ma20 = ma_gate(close, 20)
    rev15 = reversal_gate(close, 0.15, 0.15)
    g: Dict[str, Optional[np.ndarray]] = {
        "none": None,
        "ma20": ma20,
        "rev15": rev15,
        "both": (ma20 & rev15).astype(np.int8),
        "either": (ma20 | rev15).astype(np.int8),
    }
    if not quick:
        g["ma50"] = ma_gate(close, 50)
        g["rev20/12"] = reversal_gate(close, 0.20, 0.12)
    return g


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()
    os.makedirs("reports", exist_ok=True)

    W = load_wide()
    gates = build_gates(W["close"], args.quick)
    lookbacks = (120,) if args.quick else (60, 120, 250)
    tops = (5, 12) if args.quick else (3, 5, 8, 12, 20)
    rebals = (10,) if args.quick else (5, 10, 20)
    combos = [(g, lb, tn, rb) for g in gates
              for lb, tn, rb in itertools.product(lookbacks, tops, rebals)]
    print(f"grid: {len(gates)} gates x {len(lookbacks)*len(tops)*len(rebals)} "
          f"parameter sets = {len(combos)} configurations")

    n = len(W["mark"])
    edges = np.linspace(int(n * 0.35), n, args.folds + 1).astype(int)
    windows = [(edges[k], edges[k + 1]) for k in range(args.folds)]
    per_year = 250

    # ---- out-of-sample record of every configuration on every fold ---- #
    oos: Dict[Tuple, List[Dict[str, float]]] = {}
    for i, (g, lb, tn, rb) in enumerate(combos, 1):
        oos[(g, lb, tn, rb)] = [
            score_curve(*simulate(W, gates[g], lb, tn, rb, lo=a, hi=b))
            for a, b in windows]
        if i % 20 == 0:
            print(f"  out of sample {i}/{len(combos)}")

    rows = []
    for key, sc in oos.items():
        c = [s["cagr"] for s in sc]
        rows.append({"gate": key[0], "lookback": key[1], "top_n": key[2],
                     "rebalance": key[3],
                     **{f"fold{k+1}": c[k] for k in range(args.folds)},
                     "mean": float(np.mean(c)), "worst": float(np.min(c)),
                     "mean_dd": float(np.mean([s["max_dd"] for s in sc]))})
    R = pd.DataFrame(rows)
    R.to_csv("reports/maximize_book_grid.csv", index=False)

    def show(df: pd.DataFrame, title: str, k: int = 10) -> None:
        print(f"\n{title}")
        print(f" {'gate':<10}{'lb':>5}{'top':>5}{'reb':>5}"
              + "".join(f"{f'f{i+1}':>9}" for i in range(args.folds))
              + f"{'mean':>9}{'worst':>9}{'meanDD':>9}")
        for _, r in df.head(k).iterrows():
            print(f" {r['gate']:<10}{r['lookback']:>5.0f}{r['top_n']:>5.0f}"
                  f"{r['rebalance']:>5.0f}"
                  + "".join(f"{r[f'fold{i+1}']:>+9.1%}" for i in range(args.folds))
                  + f"{r['mean']:>+9.1%}{r['worst']:>+9.1%}{r['mean_dd']:>9.0%}")

    print(f"\n{'=' * 112}\n THE CEILING — best by out-of-sample mean, which is "
          f"chosen with hindsight and NOT attainable\n{'=' * 112}")
    show(R.sort_values("mean", ascending=False), "")
    print(f"\n{'=' * 112}\n THE CEILING — best by worst fold\n{'=' * 112}")
    show(R.sort_values("worst", ascending=False), "")

    # ---- nested selection: the honest number ---- #
    print(f"\n{'=' * 112}\n NESTED SELECTION — configuration chosen inside each fold, "
          f"using only earlier data\n{'=' * 112}")
    sel_rows = []
    for mode in ("expanding", "trailing"):
        chain, picks = 1.0, []
        for k, (tr_hi, te_hi) in enumerate(windows):
            lo = 0 if mode == "expanding" else max(0, tr_hi - TRAIL_YEARS * per_year)
            best, best_v = None, -np.inf
            for key in oos:
                g, lb, tn, rb = key
                s = score_curve(*simulate(W, gates[g], lb, tn, rb, lo=lo, hi=tr_hi))
                v = s["cagr"] / s["ulcer"] if s["ulcer"] > 0 else -np.inf
                if v > best_v:
                    best, best_v = key, v
            got = oos[best][k]
            yrs = (W["mark"].index[te_hi - 1] - W["mark"].index[tr_hi]).days / 365.25
            chain *= (1.0 + got["cagr"]) ** yrs
            picks.append((best, got, yrs))
            print(f" {mode:<10} fold {k+1} from "
                  f"{W['mark'].index[tr_hi]:%Y-%m}: chose "
                  f"{best[0]}/{best[1]}d/top{best[2]}/reb{best[3]:<3} -> "
                  f"{got['cagr']:>+7.1%} OOS, maxDD {got['max_dd']:>5.0%}")
        total_years = sum(p[2] for p in picks)
        cagr = chain ** (1 / total_years) - 1
        print(f" {mode:<10} chained: {chain:,.1f}x over {total_years:.1f} years "
              f"= {cagr:+.1%} a year\n")
        sel_rows.append({"selector": mode, "growth": chain, "years": total_years,
                         "cagr": cagr,
                         "mean_fold": float(np.mean([p[1]["cagr"] for p in picks])),
                         "worst_fold": float(np.min([p[1]["cagr"] for p in picks])),
                         "mean_dd": float(np.mean([p[1]["max_dd"] for p in picks]))})

    # the deployed setting, chained the same way, as the thing to beat
    for label, key in (("deployed (ma20/120d/top8/reb10)", ("ma20", 120, 8, 10)),
                       ("ungated (none/120d/top8/reb10)", ("none", 120, 8, 10))):
        if key not in oos:
            continue
        chain, ty = 1.0, 0.0
        for k, (tr_hi, te_hi) in enumerate(windows):
            yrs = (W["mark"].index[te_hi - 1] - W["mark"].index[tr_hi]).days / 365.25
            chain *= (1.0 + oos[key][k]["cagr"]) ** yrs
            ty += yrs
        sel_rows.append({"selector": label, "growth": chain, "years": ty,
                         "cagr": chain ** (1 / ty) - 1,
                         "mean_fold": float(np.mean([s["cagr"] for s in oos[key]])),
                         "worst_fold": float(np.min([s["cagr"] for s in oos[key]])),
                         "mean_dd": float(np.mean([s["max_dd"] for s in oos[key]]))})

    S = pd.DataFrame(sel_rows)
    print(f"{'=' * 112}\n WHAT YOU WOULD ACTUALLY HAVE EARNED, {windows[0][0] and ''}"
          f"April 2009 onward\n{'=' * 112}")
    print(f" {'':36}{'growth':>11}{'CAGR':>9}{'mean fold':>12}"
          f"{'worst fold':>12}{'mean maxDD':>12}")
    for _, r in S.iterrows():
        print(f" {r['selector']:<36}{r['growth']:>10,.1f}x{r['cagr']:>+9.1%}"
              f"{r['mean_fold']:>+12.1%}{r['worst_fold']:>+12.1%}{r['mean_dd']:>12.0%}")
    S.to_csv("reports/maximize_book_selection.csv", index=False)
    print("\n -> reports/maximize_book_grid.csv, reports/maximize_book_selection.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
