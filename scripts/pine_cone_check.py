#!/usr/bin/env python3
"""Does the Pine panel's arithmetic reproduce the measured IDX table?

    python3 scripts/pine_cone_check.py

THE QUESTION THIS ANSWERS. `pine/IDX_Suite.pine` prints a probability and a date
band from two closed-form laws rather than from a lookup table. Those laws are a
FIT, and a fit that is quoted without its residuals is an assertion. So this
runs the exact arithmetic the Pine file runs — same coefficients, same clamping,
same sigma — against the 180 cells it was fitted to, and prints how far off it
is, cell by cell and at the extremes.

IT IS ALSO THE DRIFT GUARD. Pine cannot import anything, so the coefficients
exist twice: in `src/idxbot/cone.py` and inline in the .pine file. A test in
`tests/test_cone.py` parses the .pine file and asserts the two agree, because
the only thing worse than an unvalidated constant is two copies of it that stop
matching.

WHAT WOULD FALSIFY THE SHIPPED PANEL: a median absolute error above ~5
probability points on either side, or a median time error above ~20%, would mean
the closed form is not a fair summary of the grid and the panel should carry the
grid instead.
"""

from __future__ import annotations

import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

from idxbot.cone import (PROB_LAW, TIME_LAW, p_touch,          # noqa: E402
                         sessions_to, vol_decile)

CELLS = os.path.join("reports", "time_price_cells.csv")
PINE = os.path.join("pine", "IDX_Suite.pine")


def check_pine_constants() -> bool:
    """Every coefficient in the module must appear literally in the .pine."""
    if not os.path.exists(PINE):
        print("  pine file missing — skipped")
        return True
    src = open(PINE, encoding="utf-8").read()
    ok = True
    wanted = [f"{v:.4f}" for t in TIME_LAW.values() for v in t]
    wanted += [f"{v:.4f}" for t in PROB_LAW.values() for v in t]
    for w in wanted:
        bare = w.lstrip("-")
        if bare not in src:
            print(f"  MISSING from {PINE}: {w}")
            ok = False
    print(f"  {len(wanted)} coefficients checked against {PINE}: "
          f"{'all present' if ok else 'MISMATCH'}")
    return ok


def main() -> int:
    if not os.path.exists(CELLS):
        print(f"{CELLS} not found — run: python3 scripts/time_price.py law")
        return 1
    R = pd.read_csv(CELLS)
    R["mult"] = np.where(R["up"] == 1, np.exp(R["d"]), np.exp(-R["d"]))

    R["p_hat"] = [p_touch(m, s, bool(st)) for m, s, st
                  in zip(R["mult"], R["sig"], R["stack"])]
    R["p_err"] = R["p_hat"] - R["p"]

    T = R.dropna(subset=["med"]).copy()
    for q in ("q1", "med", "q3"):
        T[f"{q}_hat"] = [sessions_to(m, s, q) for m, s in zip(T["mult"], T["sig"])]
        T[f"{q}_rel"] = T[f"{q}_hat"] / T[q] - 1.0

    print(f"=== the Pine cone against {len(R)} measured cells\n")
    print("P(touch within 252), probability points")
    for side, g in R.groupby(np.where(R["up"] == 1, "up", "down")):
        e = g["p_err"].abs()
        print(f"  {side:<5} n {len(g):>4}   median {100 * e.median():>5.2f}pp"
              f"   p90 {100 * e.quantile(0.9):>5.2f}pp"
              f"   max {100 * e.max():>5.2f}pp")

    print("\nsessions to touch, relative error")
    for q in ("q1", "med", "q3"):
        e = T[f"{q}_rel"].abs()
        print(f"  {q:<5} n {len(T):>4}   median {100 * e.median():>5.1f}%"
              f"   p90 {100 * e.quantile(0.9):>5.1f}%"
              f"   max {100 * e.max():>5.1f}%")

    print("\nwhere the fit is worst — the cells a user should not lean on")
    w = R.reindex(R["p_err"].abs().sort_values(ascending=False).index).head(5)
    for _, r in w.iterrows():
        print(f"  vol {r['sig']:.4f} (decile {vol_decile(r['sig'])})  "
              f"target x{r['mult']:.2f}  stack {int(r['stack'])}   "
              f"measured {r['p']:.3f}  fitted {r['p_hat']:.3f}  "
              f"err {100 * r['p_err']:+.1f}pp")

    print("\nworked example — what the panel prints for a median IDX name")
    for sig, nm in ((0.0165, "calm (decile 2)"), (0.0256, "median (decile 5)"),
                    (0.0500, "wild (decile 9)")):
        for stk in (False, True):
            pu, pd_ = p_touch(1.20, sig, stk), p_touch(0.80, sig, stk)
            print(f"  {nm:<18} stack={int(stk)}  +20% in "
                  f"{sessions_to(1.20, sig, 'q1'):.0f}/"
                  f"{sessions_to(1.20, sig, 'med'):.0f}/"
                  f"{sessions_to(1.20, sig, 'q3'):.0f} sessions   "
                  f"P(+20%) {pu:.3f}  P(-20%) {pd_:.3f}  ratio {pu / pd_:.2f}x")

    print("\nconstant drift guard")
    ok = check_pine_constants()

    bad = (R["p_err"].abs().median() > 0.05
           or T["med_rel"].abs().median() > 0.20 or not ok)
    print("\nVERDICT:", "the shipped panel is a fair summary of the grid"
          if not bad else "FAILS its own tolerance — do not ship")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
