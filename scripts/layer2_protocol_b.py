#!/usr/bin/env python3
"""Protocol B: the reversed hypotheses, frozen BEFORE the confirming data exists.

WHY THERE IS A SECOND PROTOCOL
------------------------------
Protocol A (``layer2_protocol.py``, hash 6b8e0a2c9d1f4e73) asked whether broker
buying predicts a RISE. Run on BBCA it failed - and failed in a specific way
worth recording: all four effects came back NEGATIVE, one of them hard
(H4: d = -1.34, t = -5.37). High broker buying was followed by
UNDERPERFORMANCE, not outperformance.

That is an observation, not a finding. It was produced by looking at data, which
is exactly how Result 96 produced a d = 0.257 that the pre-registered
replication then measured at d = 0.002. A hypothesis discovered in the data
cannot be tested on the same data, and dressing an exploratory look up as a
result is the single most expensive mistake available here.

So the reversed claim gets its own protocol, written down now, hashed now, while
the only name examined is BBCA and the other nine in the panel are still being
collected and have not been looked at.

THE STOPPING RULE THAT MATTERS MOST
-----------------------------------
**BBCA IS EXCLUDED FROM THE CONFIRMATORY SAMPLE.** It generated these
hypotheses; it cannot also test them. The test runs on the untouched names only,
and if that leaves too few days for the power requirement the answer is "not yet
known", not "promising".

THE HYPOTHESES, FROZEN
----------------------
    H5  high net buying by the top 3 net buyers on day t predicts a NEGATIVE
        excess return over t+1..t+5
    H6  high broker concentration on day t predicts a NEGATIVE excess return
        over t+1..t+20
    H7  high net foreign buying on day t predicts a NEGATIVE excess return over
        t+1..t+5
    H8  a third consecutive session with the same top net buyer predicts a
        NEGATIVE excess return over t+1..t+10

Everything else is inherited from Protocol A deliberately - same alpha, same
Bonferroni correction, same day-level clustering, same executable entry at the
close of t+1, same three censoring levels. Only the DIRECTION changes, because
the direction is the only thing the exploratory look suggested.

WHAT WOULD MAKE THIS REAL, AND WHAT WOULD NOT
---------------------------------------------
Real: the sign holds on the nine untouched names, at every censoring level,
after Bonferroni, with day-level clustering, and it survives controlling for the
same-day return - because flow correlates +0.22 with the day's own move, so
"brokers buy on up days and up days mean-revert" is a live alternative
explanation that must be ruled out rather than waved at.

Not real: a significant result on BBCA. A result at one censoring level. A
result that disappears once the same-day return is controlled for. A result on
fewer days than the power calculation demands.

    python3 scripts/layer2_protocol_b.py
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from typing import Dict, List

sys.path.insert(0, os.path.dirname(__file__))

from layer2_protocol import ALPHA, POWER, detectable_effect, required_n

#: The name that generated these hypotheses, and therefore cannot test them.
EXCLUDED = ("BBCA",)

HYPOTHESES_B = [
    {"id": "H5", "claim": "top-3 net buying predicts NEGATIVE excess return",
     "horizon": 5, "direction": "negative", "signal": "top3_net"},
    {"id": "H6", "claim": "top-3 buy concentration predicts NEGATIVE excess return",
     "horizon": 20, "direction": "negative", "signal": "concentration"},
    {"id": "H7", "claim": "net foreign flow predicts NEGATIVE excess return",
     "horizon": 5, "direction": "negative", "signal": "foreign_net"},
    {"id": "H8", "claim": "same top net buyer 3 sessions running predicts "
                          "NEGATIVE excess return", "horizon": 10,
     "direction": "negative", "signal": "streak3"},
]

#: Any confirmatory result must ALSO survive this control, because flow and the
#: day's own return correlate +0.22 and up days mean-revert.
CONTROLS = ("same_day_return",)


def protocol_b_hash() -> str:
    """Hash the frozen parts, including the exclusion and the required control."""
    blob = json.dumps({"hypotheses": HYPOTHESES_B, "alpha": ALPHA,
                       "power": POWER, "excluded": list(EXCLUDED),
                       "controls": list(CONTROLS)}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def main() -> int:
    a = ALPHA / len(HYPOTHESES_B)
    print(f"{'=' * 92}\n LAYER-2 PROTOCOL B — frozen {protocol_b_hash()}\n{'=' * 92}")
    print(" Written while only BBCA had been examined and the other nine names "
          "were still\n being collected. Any result reported under a different "
          "hash is a different\n experiment.\n")
    for h in HYPOTHESES_B:
        print(f" {h['id']}  {h['claim']}  (t+1..t+{h['horizon']}, one-sided)")
    print(f"\n alpha {ALPHA} Bonferroni-corrected to {a:.4f}; power target "
          f"{POWER:.0%}")
    print(f" EXCLUDED from the confirmatory sample: {', '.join(EXCLUDED)} — it "
          f"generated these\n hypotheses and therefore cannot test them.")
    print(f" REQUIRED control: {', '.join(CONTROLS)} — flow correlates +0.22 "
          f"with the day's own\n move, so 'brokers buy on up days and up days "
          f"mean-revert' must be ruled out.")
    print(f"\n{'=' * 92}\n HOW MUCH UNTOUCHED DATA IS ENOUGH\n{'=' * 92}")
    print(f" {'effect d':>9}{'days needed (9 names, ICC 0.30)':>36}")
    for d in (0.30, 0.20, 0.10):
        n = required_n(d, a, POWER)
        deff = 1 + (9 - 1) * 0.30
        print(f" {d:>9.2f}{int(-(-n * deff // 9)):>36,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
