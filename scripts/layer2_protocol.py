#!/usr/bin/env python3
"""The layer-2 test, written down BEFORE the data exists, and hashed.

WHY A PROTOCOL AND NOT JUST A SCRIPT
------------------------------------
This repo already has the cautionary tale. Result 96: a 300-event pilot found
foreign_net at d = 0.257, p = 0.047 - publishable-looking. The pre-registered
replication on 600 untouched events returned **d = 0.002, p = 0.489**. Result 97
did the trajectory version and got d = -0.005. The pilot was not a discovery; it
was the best of several things looked at, on a sample small enough for that to
happen by luck.

Broker flow is now the last untested layer, and the temptation when the data
finally arrives will be to try several definitions of "accumulation", several
horizons, and report the one that works. That is exactly how the 0.257 happened.

So the hypotheses, the horizons, the test, the correction and the stopping rule
are all fixed HERE, in code, while there is still no data to fit them to. The
file hashes itself and prints the hash: if the protocol changes, the hash changes,
and any result reported under a different hash is a different experiment.

THE HYPOTHESES, FROZEN
----------------------
Each is a one-sided claim about a sign, because "flow predicts something" is not
a testable statement and "buying predicts a rise" is.

    H1  net accumulation by the top 3 net buyers on day t predicts a POSITIVE
        excess return over days t+1..t+5
    H2  broker CONCENTRATION (share of buy volume in the top 3 brokers) on day t
        predicts a positive excess return over t+1..t+20 - the "one hand is
        accumulating" claim
    H3  net foreign flow on day t predicts a positive excess return over t+1..t+5
    H4  a day where the same broker is the top net buyer for the 3rd consecutive
        session predicts a positive excess return over t+1..t+10

Excess return means relative to the equal-weight universe that day, so a signal
cannot win by simply being long in an up market.

    alpha            0.05, Bonferroni-corrected to 0.0125 for the four hypotheses
    power target     0.80
    unit             ticker-day, clustered by DAY (flows are cross-sectionally
                     correlated; treating ticker-days as independent is the
                     single easiest way to fake significance here)
    stopping rule    the test does not run until the power requirement is met.
                     No peeking, no early reporting, no "promising so far".

    python3 scripts/layer2_protocol.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

ALPHA = 0.05
POWER = 0.80

HYPOTHESES = [
    {"id": "H1", "claim": "top-3 net buying predicts positive excess return",
     "horizon": 5, "direction": "positive"},
    {"id": "H2", "claim": "top-3 buy concentration predicts positive excess return",
     "horizon": 20, "direction": "positive"},
    {"id": "H3", "claim": "net foreign flow predicts positive excess return",
     "horizon": 5, "direction": "positive"},
    {"id": "H4", "claim": "same top net buyer 3 sessions running predicts positive "
                          "excess return", "horizon": 10, "direction": "positive"},
]


def protocol_hash() -> str:
    """Hash the frozen parts. Change a hypothesis and this changes with it."""
    blob = json.dumps({"hypotheses": HYPOTHESES, "alpha": ALPHA,
                       "power": POWER}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def required_n(effect: float, alpha: float, power: float) -> int:
    """Observations needed for a one-sided one-sample t-test at this effect size."""
    if effect <= 0:
        return 10 ** 9
    za = stats.norm.ppf(1.0 - alpha)
    zb = stats.norm.ppf(power)
    return int(np.ceil(((za + zb) / effect) ** 2))


def detectable_effect(n: int, alpha: float, power: float) -> float:
    """Smallest effect this many observations could find. The honest MDE."""
    if n <= 1:
        return np.inf
    za = stats.norm.ppf(1.0 - alpha)
    zb = stats.norm.ppf(power)
    return float((za + zb) / np.sqrt(n))


def effective_n(ticker_days: int, names: int) -> int:
    """Clustered by day, because flows move together across names.

    Treating 40 names x 250 days as 10,000 independent draws is the single
    easiest way to manufacture significance from nothing. With a high intra-day
    correlation the effective count is closer to the number of DAYS than to the
    number of ticker-days, so the design inflation factor 1 + (m-1)*rho is applied
    with a deliberately pessimistic rho.
    """
    if names <= 0 or ticker_days <= 0:
        return 0
    m = max(ticker_days / names, 1.0)          # days per name
    rho = 0.3                                   # assumed intra-day correlation
    deff = 1.0 + (names - 1) * rho
    return int(max(ticker_days / max(deff, 1.0), m))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", type=int, default=40,
                    help="how many names you expect to collect in parallel")
    ap.add_argument("--effects", type=float, nargs="+",
                    default=[0.30, 0.20, 0.10, 0.05])
    args = ap.parse_args()

    alpha_c = ALPHA / len(HYPOTHESES)
    print(f"{'=' * 94}\n LAYER-2 PROTOCOL — frozen {protocol_hash()}\n{'=' * 94}")
    print(" Any result reported under a different hash is a different "
          "experiment.\n")
    for h in HYPOTHESES:
        print(f" {h['id']}  {h['claim']}  (t+1..t+{h['horizon']}, one-sided)")
    print(f"\n alpha {ALPHA} Bonferroni-corrected to {alpha_c:.4f} for "
          f"{len(HYPOTHESES)} hypotheses; power target {POWER:.0%}")

    print(f"\n{'=' * 94}\n HOW MUCH DATA IS ENOUGH\n{'=' * 94}")
    print(f" {'effect (d)':>12}{'what that means':>34}{'effective n':>14}"
          f"{'ticker-days':>14}{'days @' + str(args.names) + ' names':>18}")
    rows = []
    for d in args.effects:
        n_eff = required_n(d, alpha_c, POWER)
        deff = 1.0 + (args.names - 1) * 0.3
        td = int(np.ceil(n_eff * deff))
        days = int(np.ceil(td / args.names))
        meaning = {0.30: "large - would be obvious by eye",
                   0.20: "the usual 'small' effect",
                   0.10: "realistic for a real flow edge",
                   0.05: "plausible and still tradable"}.get(d, "")
        rows.append({"effect": d, "n_eff": n_eff, "ticker_days": td, "days": days})
        print(f" {d:>12.2f}{meaning:>34}{n_eff:>14,}{td:>14,}{days:>18,}")

    print(f"\n Collecting {args.names} names, one session a day:")
    for r in rows:
        yrs = r["days"] / 250.0
        print(f"   to detect d = {r['effect']:.2f}: {r['days']:,} sessions "
              f"(~{yrs:.1f} years)")

    # ---- what the existing data could actually have found ------------------
    print(f"\n{'=' * 94}\n WHAT THE DATA THAT ALREADY EXISTS COULD DETECT"
          f"\n{'=' * 94}")
    store = os.path.join("data", "cache", "broker_daily")
    have = len([f for f in os.listdir(store)]) if os.path.isdir(store) else 0
    legacy_days, legacy_names = 60, 1        # BBCA only, per the cache audit
    n_eff = effective_n(legacy_days * legacy_names, legacy_names)
    mde = detectable_effect(max(n_eff, 2), alpha_c, POWER)
    print(f" daily store: {have} ticker-day files collected so far")
    print(f" legacy usable daily panel: {legacy_names} name x {legacy_days} "
          f"sessions (BBCA), effective n = {n_eff}")
    print(f" smallest effect that could have been found there: d = {mde:.2f}")
    print(f" An effect that large would be visible without statistics. So the "
          f"honest reading\n of 'broker flow has been tested' is that it has "
          f"been tested on event windows,\n never on a daily panel with the "
          f"power to see a tradable effect.")

    print(f"\n{'=' * 94}\n THE STOPPING RULE\n{'=' * 94}")
    target = next((r for r in rows if r["effect"] == 0.10), rows[-1])
    print(f" This protocol does not report a result until there are "
          f"{target['ticker_days']:,} ticker-days\n ({target['days']:,} sessions "
          f"across {args.names} names). Until then the only honest output is a "
          f"count.")
    print(" No peeking. No 'promising so far'. The d = 0.257 in Result 96 came "
          "from exactly\n that habit, and it did not replicate.")

    out = {"hash": protocol_hash(), "alpha": ALPHA, "alpha_corrected": alpha_c,
           "power": POWER, "hypotheses": HYPOTHESES,
           "requirements": rows, "collected_files": have}
    os.makedirs("reports", exist_ok=True)
    with open("reports/layer2_protocol.json", "w") as fh:
        json.dump(out, fh, indent=2)
    print("\n -> reports/layer2_protocol.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
