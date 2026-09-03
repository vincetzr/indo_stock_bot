#!/usr/bin/env python3
"""H54, hold-forever family, addendum: NEVER SELL, ONLY ADD.

Pick-once IS `BH_PICKS` by construction, so it cannot beat that benchmark --
it can only lose to it by the turnover the harness charges on drift. The one
way to stay inside "turnover is the enemy" and still produce a book that
DIFFERS from the first basket is to never sell anything and fund each new
position out of a small pro-rata trim. Adding one name a year at 3% costs
~0.03%/yr, which is an order of magnitude below every other variant tested.

Run against the same fixed harness, at 12 start dates, with the same PASS rule.

    python3 scripts/strat_holdforever_accum.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))
sys.path.insert(0, os.path.dirname(__file__))

from bhbench import Bench, load, report                          # noqa: E402
from strat_holdforever import (RULES, Accumulate, verdict_at)    # noqa: E402


def main() -> None:
    print("THE 24-MONTH HOLDOUT IS ALREADY SPENT. EVERYTHING IS IN-SAMPLE.\n")
    grid = []
    for mtv, lab in ((1e9, "min_tv=1e9"), (3e8, "min_tv=3e8")):
        P = load(min_tv=mtv)
        B = Bench(P)
        for nm, rl in RULES.items():
            if nm == "all-eligible":
                continue          # adding to "everything" is meaningless
            for k0, aw, na in ((10, 0.03, 1), (10, 0.05, 2), (15, 0.02, 1)):
                lbl = (f"{nm} accumulate k0={k0} add {aw:.0%}x{na}/yr [{lab}]")
                S = Accumulate(B, rl, k=k0, reselect_every=1,
                               add_w=aw, n_add=na)
                v = B.evaluate(S, label=lbl, freq=252)
                print(report(v))
                print()
                grid.append((lbl, B, rl, k0, aw, na, v))

    print("=" * 78)
    print("ARBITER — same verdict rule, 12 start dates.\n")
    print(f"  {'configuration':<46}{'mean':>9}{'sd':>7}{'min':>8}{'max':>8}"
          f"{'beats all 3':>13}{'PASS':>8}")
    rows = []
    for lbl, B, rl, k0, aw, na, v in grid:
        out = []
        for off in np.linspace(0, 251, 12).astype(int):
            r = verdict_at(B, lambda: Accumulate(B, rl, k=k0,
                                                 reselect_every=1,
                                                 add_w=aw, n_add=na),
                           252, int(off))
            if r:
                out.append(r)
        if not out:
            continue
        c = np.array([x["cagr"] for x in out])
        nb = sum(1 for x in out if x["beats_all"])
        npz = sum(1 for x in out if x["PASS"])
        print(f"  {lbl:<46}{c.mean():>+9.2%}{c.std(ddof=1):>7.2%}"
              f"{c.min():>+8.2%}{c.max():>+8.2%}{nb:>8}/{len(out):<4}"
              f"{npz:>4}/{len(out)}")
        rows.append({"config": lbl, "mean": c.mean(), "sd": c.std(ddof=1),
                     "min": c.min(), "max": c.max(), "beats_all_n": nb,
                     "pass_n": npz, "n": len(out),
                     "offset0_cagr": v.get("cagr"),
                     "offset0_PASS": v.get("PASS")})
    pd.DataFrame(rows).to_csv("reports/strat_holdforever_accum.csv",
                              index=False)
    print(f"\n{len(grid)} accumulate variants at offset 0, "
          f"{sum(1 for g in grid if g[-1].get('PASS'))} passed.")
    print("wrote reports/strat_holdforever_accum.csv")


if __name__ == "__main__":
    main()
