#!/usr/bin/env python3
"""§38 — log today's signals, score the settled ones, print what is known.

    python scripts/signal_log.py --emit     # record today's basket
    python scripts/signal_log.py --score    # walk every past signal forward
    python scripts/signal_log.py            # both, then the summary

THIS IS THE ONLY SOURCE OF OUT-OF-SAMPLE EVIDENCE THIS PROJECT HAS LEFT.
The 24-month holdout was spent at H16, so all 347 registered trials are
in-sample. Recording what the system says before the outcome exists is the one
thing that changes that, and it only ever gets later.

It is wired into the daily job rather than run by hand for the same reason:
a log that depends on someone remembering is a log with gaps exactly where the
interesting days were.
"""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))
sys.path.insert(0, os.path.dirname(__file__))

from idxbot import signal_store as ss                             # noqa: E402
from beathold import Sticky                                       # noqa: E402
from bhbench import load                                          # noqa: E402
import rules                                                       # noqa: E402

#  The rule's identity. CHANGING ANY OF THESE CHANGES THE SIGNAL ID, which is
#  correct: a different parameter set is a different prediction and must not
#  overwrite the old one's record.
#  A NAME MUST NOT CARRY A PARAMETER VALUE. This was `h54_sticky_tight`, after
#  H54's arm name for keep_hi = 0.80 -- and H62 moved it to 0.70, at which
#  point the store's own label said "tight" for a rule that is not. A name
#  carrying a parameter drifts every time the parameter moves, which is the
#  same failure as the card's hardcoded "top 20%" label. The FAMILY is the
#  name; the parameters live in `rule_version`, which is where they can change
#  without lying.
RULE = "h54_sticky"
RULE_VERSION = (f"hi{rules.ENTRY_HI}/{rules.KEEP_HI}"
                f"_vol{rules.ENTRY_VOL}/{rules.KEEP_VOL}"
                f"_k{rules.K}_sl{rules.STOP}_tp{rules.TP}x{rules.TP_FRAC}")
#  Fixed AT EMISSION, per A20: choosing the horizon after seeing the outcome is
#  the purest form of the error that appendix records.
HORIZON_DAYS = 252


def todays_rows(P: pd.DataFrame):
    last = P[P["elig"]]["date"].max()
    day = P[(P["date"] == last) & P["elig"]].dropna(subset=["hi52", "vol60"])
    sticky = Sticky(rules.K, keep_hi_q=rules.KEEP_HI,
                    keep_vol_q=rules.KEEP_VOL)
    picks = [t for t, _ in sticky(day.copy())]
    d = day[day["ticker"].isin(picks)]
    rows = []
    for _, r in d.iterrows():
        rows.append({
            "ticker": r["ticker"],
            "entry": float(r["close"]),
            "sl": float(r["close"]) * (1.0 - rules.STOP),
            "tp": float(r["close"]) * (1.0 + rules.TP),
            #  the shipped rule sells HALF at the target (H56b)
            "tp_frac": float(rules.TP_FRAC),
            #  Everything below is the knowable state at `asof`, so an outcome
            #  can be attributed rather than merely counted.
            "hi52": float(r["hi52"]), "vol60": float(r["vol60"]),
            "tv60": float(r["tv60"]), "close": float(r["close"]),
            "n_eligible": int(len(day)),
        })
    return last, rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--emit", action="store_true")
    ap.add_argument("--score", action="store_true")
    args = ap.parse_args()
    do_all = not (args.emit or args.score)

    P = load()
    if args.emit or do_all:
        asof, rows = todays_rows(P)
        res = ss.emit(rows, RULE, RULE_VERSION, asof, HORIZON_DAYS)
        print(f"EMIT  asof {pd.Timestamp(asof).date()}  rule {RULE}")
        print(f"      version {RULE_VERSION}")
        print(f"      written {res['written']}, skipped {res.get('skipped', 0)}"
              f", total {res.get('total', '?')}")
    if args.score or do_all:
        o = ss.score(P)
        print(f"\nSCORE {len(o):,} signals walked forward")
    print()
    print(ss.summary())


if __name__ == "__main__":
    main()
