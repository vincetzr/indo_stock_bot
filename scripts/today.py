#!/usr/bin/env python3
"""ONE COMMAND: what this system says today, and which part of it to believe.

WHY THIS EXISTS. The standing instruction is that every signal carries ENTRY,
SL, TP and the measured cost of each level. Three surfaces implement it and
they DISAGREE — `rules.py` names a quarterly basket, `daily_signal.py` names a
daily bracket, and `positions.py` monitors what is already open. A reader who
runs all three gets three different lists and nothing telling them which one
the evidence supports. That is not a presentation problem; it is the difference
between a research repo and a system someone can act on.

WHAT THIS IS NOT, AND THE DISTINCTION IS THE WHOLE DESIGN.
It does NOT blend the two rules. A13 records the reason: a composite of
separately-tested components is a new signal wearing their credibility, and it
has never been tested. The two lists are printed side by side, each with what
IT was measured to do, and the precedence below is DERIVED FROM THOSE MEASURED
NUMBERS IN CODE rather than asserted in prose — so it cannot drift from them.

THE PRECEDENCE, and every term in it is a measurement this repo made:

  * The quarterly card (H54/H56) beats the index in 6 of 6 rebalance calendars
    and its stop cuts portfolio drawdown in BOTH halves. Its CAGR edge is NOT
    claimed — worse early, better late, which is regime noise (A18) — and its
    deflated Sharpe survives only the most flattering of seven dispersion
    assumptions (H58).

  * The daily bracket (H42) is measured at −13.06% a year [−16.25%, −10.22%]
    against simply holding the same name, negative in ALL TEN expectancy
    deciles. `MIN_RR` deletes the worst 76% of rows; the survivors still lose
    to holding by 7 to 10 points a year, and beating a random name is not
    established (95% CI contains zero).

  * So the bracket list is a WATCHLIST and the card is the book. That sentence
    is the output of the comparison below, not an opinion placed above it.

AND THE THING BOTH LOSE TO IS PRINTED FIRST. A19 records the missing benchmark
as the error class that manufactures results, and it is the same one here: the
comparison a reader would actually take is buying the index, and it belongs at
the top rather than in a footnote.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from typing import List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))
sys.path.insert(0, os.path.dirname(__file__))

from idxbot import signal_store as ss                            # noqa: E402
from idxbot.cone import BRACKET_VS_HOLD, MIN_RR                  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

#: Measured, each with the hypothesis that produced it. These are the terms the
#: precedence is computed from; changing one changes the verdict, which is the
#: point of computing it rather than writing it down.
CARD = {
    "label": "quarterly card (H54/H56)",
    "cagr": 0.1108,          # with stop + half-scale-out
    "cagr_none": 0.1032,     # band exit only
    "maxdd": -0.352,
    "maxdd_none": -0.420,
    "beats_index_n": 6, "phases": 6,
    "dd_both_halves": True,
    "cagr_both_halves": False,     # worse early, better late — regime noise
    "dsr_best": 0.974, "dsr_next": 0.947, "dsr_survives": 1, "dsr_of": 7,
}
BRACKET = {
    "label": "daily bracket scan (H42)",
    "vs_hold": BRACKET_VS_HOLD[0],
    "vs_hold_lo": BRACKET_VS_HOLD[1], "vs_hold_hi": BRACKET_VS_HOLD[2],
    "deciles_negative": 10, "deciles": 10,
    "gate_removes": 0.76,
    "vs_random_lo": -0.0044, "vs_random_hi": +0.0277,   # CI contains zero
}
#: THE BENCHMARK, AND BOTH FIGURES ARE QUOTED BECAUSE THEY ARE OVER DIFFERENT
#: WINDOWS. A19 measured the index at +12.7% on a TOTAL-RETURN basis over its
#: own span; A38 measured +11.15% over the span its arms occupied. Picking one
#: and calling it "the index" would be A19's own error class -- comparing
#: quantities measured over different windows -- so both are printed with
#: their source, and the precedence below uses neither: it turns on the
#: 6-of-6 calendar count, which is already window-matched per arm.
INDEX = {
    "label": "IHSG, bought and held",
    "a19": 0.127,
    "a38": 0.1115,
    "note": ("A19 +12.7% total-return over its span; A38 +11.15% over the "
             "arms' span. The names run on adj_close and are TOTAL returns "
             "while ^JKSE is a PRICE index, so the index is corrected UP by "
             "the measured 1.77% top-decile yield before either comparison"),
}


def _run(script: str, args: Optional[List[str]] = None) -> str:
    """Run a sibling script and return its stdout, or an explanation."""
    cmd = [sys.executable, os.path.join(HERE, script)] + list(args or [])
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    except Exception as exc:                                # noqa: BLE001
        return f"  [{script} could not run: {exc}]"
    if p.returncode != 0:
        tail = (p.stderr or "").strip().splitlines()[-3:]
        return (f"  [{script} exited {p.returncode}] "
                + " | ".join(tail))
    return p.stdout


def precedence() -> List[str]:
    """Which list to act on, COMPUTED from the measured numbers above.

    A sentence written by hand drifts from the table it describes; this repo
    has recorded that four times (a hardcoded reconciliation figure that
    outlived its source, a docstring asserting a claim its own function had
    retracted, a Pine constant that had to be pinned by test). So the verdict
    is derived, and if a constant changes the verdict changes with it.
    """
    L = []
    card_ok = (CARD["beats_index_n"] == CARD["phases"]
               and CARD["dd_both_halves"])
    brk_bad = (BRACKET["vs_hold"] < 0
               and BRACKET["deciles_negative"] == BRACKET["deciles"])
    if card_ok and brk_bad:
        L.append("  ACT ON THE CARD. The bracket list is a WATCHLIST.")
        L.append(f"    the card beats the index in "
                 f"{CARD['beats_index_n']}/{CARD['phases']} rebalance "
                 f"calendars and its stop cuts drawdown in BOTH halves;")
        L.append(f"    the bracket returns {BRACKET['vs_hold']:+.2%} a year "
                 f"[{BRACKET['vs_hold_lo']:+.2%}, "
                 f"{BRACKET['vs_hold_hi']:+.2%}] against simply HOLDING the "
                 f"same name,")
        L.append(f"    negative in all {BRACKET['deciles_negative']} of "
                 f"{BRACKET['deciles']} expectancy deciles.")
    elif card_ok:
        L.append("  ACT ON THE CARD; the bracket's own measurement no longer "
                 "disqualifies it, so re-read H42 before using either.")
    else:
        L.append("  NEITHER LIST IS SUPPORTED by its own measurement today. "
                 "Re-read the memos before acting.")
    #  The qualifications are printed whatever the verdict, because a verdict
    #  without them is the thing this file exists to prevent.
    L.append("")
    L.append("  AND WHAT THE CARD IS NOT:")
    if not CARD["cagr_both_halves"]:
        L.append(f"    its CAGR edge ({CARD['cagr']:.2%} against "
                 f"{CARD['cagr_none']:.2%} with no stop or target) is NOT "
                 f"claimed —")
        L.append("    worse in the early half, better in the late one, which "
                 "is regime noise (A18).")
        L.append(f"    What IS claimed is the drawdown: "
                 f"{CARD['maxdd']:.1%} against {CARD['maxdd_none']:.1%}, "
                 f"and it holds in both halves.")
    L.append(f"    Charged for having searched 350 times, its deflated Sharpe "
             f"is {CARD['dsr_best']:.3f} on the")
    L.append(f"    most flattering dispersion assumption available and "
             f"{CARD['dsr_next']:.3f} on the next, which FAILS —")
    L.append(f"    it survives {CARD['dsr_survives']} of "
             f"{CARD['dsr_of']} assumptions (H58).")
    L.append("    Every number in this repo is IN-SAMPLE: the holdout was "
             "spent at H16.")
    return L


def open_book() -> pd.DataFrame:
    """Signals in the append-only store that have not settled."""
    em = ss.load_emitted()
    if em.empty:
        return em
    oc = ss.load_outcomes()
    if len(oc) and "settled" in oc.columns:
        done = set(oc.loc[oc["settled"].astype(bool), "signal_id"])
        em = em[~em["signal_id"].isin(done)]
    return em


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-scan", action="store_true",
                    help="skip the daily bracket scan (it is the slow part)")
    ap.add_argument("--no-monitor", action="store_true")
    a = ap.parse_args()

    print("=" * 78)
    print(" WHAT THIS SYSTEM SAYS TODAY")
    print("=" * 78)

    #  ------------------------------------------------ the benchmark, FIRST
    print(f"\n THE ALTERNATIVE YOU ALREADY HAVE: {INDEX['label']}, "
          f"{INDEX['a38']:.2%} to {INDEX['a19']:.2%}/yr")
    print(f"   {INDEX['note']}.")
    print("   Everything below is measured against this,")
    print("   because the comparison a reader would actually take belongs at "
          "the top")
    print("   and not in a footnote (A19).")

    #  ------------------------------------------------------- the precedence
    print("\n" + "-" * 78)
    print(" WHICH LIST TO ACT ON")
    print("-" * 78)
    for line in precedence():
        print(line)

    #  ------------------------------------------------------- the open book
    print("\n" + "-" * 78)
    print(" THE OPEN BOOK — what the store already says you hold")
    print("-" * 78)
    ob = open_book()
    if ob.empty:
        print("  nothing open. Run scripts/refresh.py --signals to record "
              "today's.")
    else:
        print(f"  {'ticker':<8}{'rule':<20}{'asof':<12}{'entry':>10}"
              f"{'SL':>10}{'TP':>11}{'part':>6}")
        for _i, r in ob.iterrows():
            frac = ss._tp_frac(r)
            print(f"  {r['ticker']:<8}{str(r['rule']):<20}"
                  f"{str(pd.Timestamp(r['asof']).date()):<12}"
                  f"{float(r['entry']):>10,.0f}{float(r['sl']):>10,.0f}"
                  f"{float(r['tp']):>11,.0f}{frac:>6.0%}")
        print(f"\n  {len(ob)} open across "
              f"{ob['rule'].nunique()} rule(s). None of these is out-of-sample")
        print("  evidence until it settles; scripts/signal_log.py --score "
              "walks them forward.")

    #  ------------------------------------------------------------- the card
    print("\n" + "-" * 78)
    print(" THE BOOK — quarterly card (act on this one)")
    print("-" * 78)
    print(_run("rules.py"))

    #  ---------------------------------------------------------- the scanner
    if not a.no_scan:
        print("-" * 78)
        print(" THE WATCHLIST — daily bracket scan (do NOT act on this one)")
        print("-" * 78)
        print(_run("daily_signal.py"))

    #  ---------------------------------------------------------- the monitor
    if not a.no_monitor and not ob.empty:
        print("-" * 78)
        print(" THE MONITOR — live levels on the open book")
        print("-" * 78)
        print(_run("positions.py", ["--from-store"]))

    print("=" * 78)
    print(" Nothing here is validated. The holdout was spent at H16, so every")
    print(" number above is in-sample, and A23 applies in full to third-party")
    print(" money: no live track record exists, and a suitability judgement")
    print(" about a specific client is not something this repo can supply.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
