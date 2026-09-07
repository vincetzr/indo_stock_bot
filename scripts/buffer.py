#!/usr/bin/env python3
"""H62 — the shipped buffer is the argmax of a three-point sweep, and A39 said so.

WHAT IS BEING RE-OPENED, AND WHY IT IS THE SHIPPED RULE'S LAST LOOSE PARAMETER.
`scripts/rules.py` ships `KEEP_HI = 0.80, KEEP_VOL = 0.60` — the "tight buffer"
arm of H54's grid. That arm was the ARGMAX of a three-point sweep whose own
memo records it as unresolved:

    tight  +6.50%   wide  +4.76%   none  +4.14%   (excess over the index)

and A39 states the problem plainly: *"the buffer sweep is NOT monotone
(+6.50% tight / +4.76% wide / +4.14% none) so the parameter is unresolved and
the family's spread is about as wide as its effect."* The tight arm's own
per-phase range in that table is **+1.97 to +6.61** — 4.6 points — against a
1.74-point lead over the next arm. **A lead smaller than its own sampling
spread is not a lead.**

H56 already faced exactly this for the stop and answered it correctly: `STOP =
0.20` is "the middle of the family and deliberately not its argmax". The buffer
never got the same treatment, and it is the one shipped constant still standing
on a maximum.

--------------------------------------------------------------------- REGISTERED
Written before any cell was scored.

  B1  The sweep is not monotone in buffer width, and the spread ACROSS buffer
      settings is smaller than the spread ACROSS REBALANCE PHASES within one
      setting. PREDICTION: confirmed. A39 already reports the tight arm's phase
      range as wider than its lead; this measures it on a full grid instead of
      three points.

  B2  PREDICTED NULL. Moving the shipped constant from the argmax to the MIDDLE
      of the family costs nothing measurable — the median-across-phases excess
      of the middle cell sits inside the argmax cell's own phase spread.
      PREDICTION: confirmed, and if it FAILS the buffer is a real parameter and
      the argmax should stay.

  B3  The RANKING of buffer settings is unstable across phases: the best cell
      at one rebalance calendar is not the best at another. PREDICTION:
      confirmed. A39 records three strategies passing this harness at offset 0
      and at no other phase, which is the same failure one level up.

The decision rule is fixed HERE, before the numbers: **if B1 and B3 both hold,
the shipped constant moves to the middle of the family**, exactly as H56 did
for the stop. It does not move to a new argmax under any outcome.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))
sys.path.insert(0, os.path.dirname(__file__))

from bhbench import Bench, load                                  # noqa: E402
from beathold import Sticky                                      # noqa: E402

#: The entry lines are FIXED. Only the KEEP band moves, which is what makes
#: this a sweep of the buffer rather than of the screen.
ENTRY_HI, ENTRY_VOL = 0.90, 0.50

#: `keep_hi_q` at 0.90 and `keep_vol_q` at 0.50 IS the entry line, i.e. no
#: buffer at all; the grid runs from there to a very wide band.
HI_GRID = (0.90, 0.85, 0.80, 0.75, 0.70, 0.60, 0.50, 0.40)
VOL_GRID = (0.50, 0.55, 0.60, 0.70, 0.80, 0.90)
SHIPPED = (0.80, 0.60)
FREQ = 63
PHASES = 6


def middle_cell() -> Tuple[float, float]:
    """The MIDDLE of the family, defined on the grid AXES and nowhere else.

    Picking the middle by outcome is an argmax wearing a different word, so the
    definition may not touch the results. An even-length axis has no median
    that is also a grid point, so it is the nearest grid point to the axis
    median — and `min` breaks a tie toward the SMALLER value deterministically,
    which matters only because it must not be re-decided later.

    One definition, used by the script and by its test, so the shipped constant
    and the rule that chose it cannot drift apart.
    """
    hi = min(HI_GRID, key=lambda x: (abs(x - float(np.median(HI_GRID))), x))
    vol = min(VOL_GRID, key=lambda x: (abs(x - float(np.median(VOL_GRID))), x))
    return float(hi), float(vol)


def sweep(B: Bench, freq: int = FREQ, phases: int = PHASES) -> pd.DataFrame:
    """Excess over the index, per (buffer cell, rebalance phase).

    THE PHASE IS THE POINT. A39 records the rebalance phase as a free parameter
    nobody chose — `marks` starts at `dates[offset]` and offset 0 is the
    panel's first bar for no reason but that it is first — and three strategies
    passed that harness at offset 0 and at no other phase. A buffer chosen at
    one phase is chosen on one draw.
    """
    dates = np.sort(B.P.loc[B.P["elig"], "date"].unique())
    step = max(len(dates) // (phases * 40), 1)
    offsets = [i * step for i in range(phases)]
    rows = []
    for hi in HI_GRID:
        for vol in VOL_GRID:
            sel = Sticky(10, hi_q=ENTRY_HI, vol_q=ENTRY_VOL,
                         keep_hi_q=hi, keep_vol_q=vol)
            for off in offsets:
                sel.reset()
                r = B.walk(sel, freq=freq, offset=off)
                if not r:
                    continue
                idx = B.index_cagr(r["start"], r["end"])
                if not np.isfinite(idx):
                    continue
                rows.append({"hi": hi, "vol": vol, "offset": off,
                             "cagr": r["cagr"], "index": idx,
                             "excess": r["cagr"] - idx,
                             "turnover": r["turnover"],
                             "early": r["early"], "late": r["late"]})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--freq", type=int, default=FREQ)
    ap.add_argument("--phases", type=int, default=PHASES)
    a = ap.parse_args()

    P = load()
    B = Bench(P)
    S = sweep(B, a.freq, a.phases)
    if S.empty:
        print("nothing ran")
        return
    print("H62 — the buffer, swept properly")
    print(f"{len(HI_GRID)}x{len(VOL_GRID)} buffer cells x {a.phases} rebalance "
          f"phases = {len(S):,} walks, freq {a.freq}")
    print(f"entry line FIXED at hi52 >= q{ENTRY_HI:.2f} and "
          f"vol60 <= q{ENTRY_VOL:.2f}; only the KEEP band moves")

    G = (S.groupby(["hi", "vol"])
         .agg(med=("excess", "median"), lo=("excess", "min"),
              hi_=("excess", "max"), n=("excess", "size"),
              turn=("turnover", "median"))
         .reset_index())
    G["spread"] = G["hi_"] - G["lo"]

    print(f"\n{'keep_hi':>8}" + "".join(f"{v:>9.2f}" for v in VOL_GRID)
          + "     <- keep_vol")
    print("-" * (8 + 9 * len(VOL_GRID) + 15))
    for hi in HI_GRID:
        line = f"{hi:>8.2f}"
        for vol in VOL_GRID:
            c = G[(G["hi"] == hi) & (G["vol"] == vol)]
            line += (f"{c['med'].iloc[0]:>+9.2%}" if len(c) else f"{'--':>9}")
        mark = "  <- SHIPPED row" if hi == SHIPPED[0] else ""
        print(line + mark)
    print("     (median excess over the index across "
          f"{a.phases} rebalance phases)")

    # ------------------------------------------------------------------ B1
    best = G.loc[G["med"].idxmax()]
    ship = G[(G["hi"] == SHIPPED[0]) & (G["vol"] == SHIPPED[1])].iloc[0]
    across = float(G["med"].max() - G["med"].min())
    within = float(G["spread"].median())
    print(f"\nB1  spread ACROSS buffer settings   {across:+.2%}")
    print(f"    spread ACROSS PHASES, median cell {within:+.2%}")
    b1 = within > across
    print(f"    {'CONFIRMED' if b1 else 'FAILED'}: the phase spread is "
          f"{'WIDER' if b1 else 'narrower'} than the whole buffer effect, so a "
          f"buffer\n    chosen at one calendar is chosen on one draw.")

    # ------------------------------------------------------------------ B3
    win = (S.loc[S.groupby("offset")["excess"].idxmax()]
           [["offset", "hi", "vol", "excess"]])
    n_distinct = win.groupby(["hi", "vol"]).ngroups
    print(f"\nB3  best cell by phase:")
    for _i, r in win.iterrows():
        print(f"      offset {int(r['offset']):>4}: "
              f"keep_hi {r['hi']:.2f} keep_vol {r['vol']:.2f}  "
              f"{r['excess']:+.2%}")
    b3 = n_distinct > 1
    print(f"    {n_distinct} distinct winners across {len(win)} phases — "
          f"{'CONFIRMED' if b3 else 'FAILED'}")

    # ------------------------------------------------------------------ B2
    #  The MIDDLE of the family, defined before the numbers: the median of each
    #  grid axis, not the median of the RESULTS. Picking the middle by outcome
    #  would be an argmax wearing a different word.
    mid_hi, mid_vol = middle_cell()
    mid = G[(G["hi"] == mid_hi) & (G["vol"] == mid_vol)].iloc[0]
    print(f"\nB2  argmax cell   keep_hi {best['hi']:.2f} "
          f"keep_vol {best['vol']:.2f}   median excess {best['med']:+.2%} "
          f"[{best['lo']:+.2%}, {best['hi_']:+.2%}]")
    print(f"    shipped cell  keep_hi {ship['hi']:.2f} "
          f"keep_vol {ship['vol']:.2f}   median excess {ship['med']:+.2%} "
          f"[{ship['lo']:+.2%}, {ship['hi_']:+.2%}]")
    print(f"    MIDDLE cell   keep_hi {mid['hi']:.2f} "
          f"keep_vol {mid['vol']:.2f}   median excess {mid['med']:+.2%} "
          f"[{mid['lo']:+.2%}, {mid['hi_']:+.2%}]")
    inside = bool(best["lo"] <= mid["med"] <= best["hi_"])
    print(f"    the middle cell sits {'INSIDE' if inside else 'OUTSIDE'} the "
          f"argmax cell's own phase spread — B2 "
          f"{'CONFIRMED' if inside else 'FAILED'}")

    # -------------------------------------------------------------- verdict
    print("\nVERDICT")
    if b1 and b3:
        print("    B1 and B3 both hold, so the decision rule registered before")
        print("    the run applies: THE SHIPPED CONSTANT MOVES TO THE MIDDLE OF")
        print(f"    THE FAMILY — keep_hi {mid['hi']:.2f}, "
              f"keep_vol {mid['vol']:.2f} — exactly as H56 did for the stop.")
        print(f"    It does NOT move to the argmax (keep_hi {best['hi']:.2f}, "
              f"keep_vol {best['vol']:.2f}) under any outcome.")
        if not inside:
            print("    B2 FAILED, so the move is not free: the middle cell is")
            print("    outside the argmax's phase spread. It is still the right")
            print("    move — a maximum this unstable is not a measurement —")
            print("    and the cost is printed rather than hidden.")
    else:
        print("    B1 or B3 failed, so the buffer behaves like a real")
        print("    parameter and the shipped constant STAYS. Re-read the grid")
        print("    before changing it.")
    print("\n  Every number here is IN-SAMPLE: the holdout was spent at H16.")
    print("  Turnover at the shipped cell "
          f"{ship['turn']:.0%} against {best['turn']:.0%} at the argmax and "
          f"{mid['turn']:.0%} at the middle.")

    out = os.path.join("reports", "buffer_sweep.csv")
    os.makedirs("reports", exist_ok=True)
    S.to_csv(out, index=False)
    print(f"\n  wrote {out}")


if __name__ == "__main__":
    main()
