#!/usr/bin/env python3
"""Pre-registered replication: does foreign buying into a pullback predict a bounce?

The pilot (300 events) found ONE thing, and it must be treated as a hypothesis
rather than a result, because it came out of 12 tests:

    foreign_net is higher on pullbacks that bounced
    difference +0.0206, Cohen d 0.257, t +2.00, p 0.047

p = 0.047 against a Bonferroni threshold of 0.05/12 = 0.0042. It does not
survive correction, so on the pilot data it is not evidence - it is a candidate.

THE PRE-REGISTRATION, fixed before this script fetches anything
---------------------------------------------------------------
    hypothesis   foreign_net is HIGHER for pullbacks that bounce than for
                 those that do not
    test         Welch t-test, ONE-SIDED (the direction is predicted)
    alpha        0.05
    outcome      `bounced` - price more than 5% above the signal within 13
                 weeks, which is the operational false-exit definition
    sample       events NOT used in the pilot, drawn from the same population
    features     ONE. foreign_net. Nothing else is tested, so nothing needs
                 correcting.

Everything else in the pilot output - bumn_net, imbalance, concentration,
local_net, foreign_share, and the `recovered` label - is deliberately excluded.
Testing them again here would recreate the multiple-comparison problem this
script exists to escape.

A replication that fails is the expected outcome for a p=0.047 finding out of
12 tests, and it will be reported as such.

    python3 scripts/pullback_replicate.py --limit 600
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

from idxbot.config import load_config          # noqa: E402
from idxbot.data.cache import Cache            # noqa: E402
from pullback_flow import fetch_window, flow_features   # noqa: E402

ALPHA = 0.05


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=600)
    ap.add_argument("--min-weeks", type=int, default=2)
    ap.add_argument("--seed", type=int, default=4242)
    args = ap.parse_args()
    os.makedirs("reports", exist_ok=True)

    E = pd.read_csv("reports/pullback_events.csv",
                    parse_dates=["peak_date", "signal_date"])
    E = E[E["weeks_from_peak"] >= args.min_weeks].reset_index(drop=True)

    pilot = pd.read_csv("reports/pullback_flow.csv", parse_dates=["signal_date"])
    used = set(zip(pilot["ticker"], pilot["signal_date"]))
    fresh = E[~E.apply(lambda r: (r["ticker"], r["signal_date"]) in used, axis=1)]
    print(f"{len(E):,} events total, {len(used)} used in the pilot, "
          f"{len(fresh):,} untouched")

    fresh = fresh.sample(n=min(args.limit, len(fresh)), random_state=args.seed)
    fresh = fresh.sort_values(["ticker", "signal_date"]).reset_index(drop=True)
    n1 = int(fresh["bounced_5pct"].sum())
    n2 = len(fresh) - n1
    mde = (1.645 + 0.84) * np.sqrt(1 / max(n1, 1) + 1 / max(n2, 1))
    print(f"replication sample: {len(fresh)} events, {fresh['ticker'].nunique()} names")
    print(f"  expected split {n1}/{n2}, one-sided MDE at 80% power: d = {mde:.2f}")
    print(f"  the pilot's effect was d = 0.257 — "
          f"{'adequately powered' if mde <= 0.257 else 'UNDERPOWERED for that effect'}")

    cfg = load_config()
    cache = Cache(cfg.path("data.cache_dir", "data/cache"))
    reg = cfg.brokers
    classes = {c.upper(): ("bumn" if getattr(reg.get(c), "state_owned", False)
                           else "foreign" if getattr(reg.get(c), "foreign", False)
                           else "local")
               for c in reg.codes() if reg.get(c) is not None}

    rows: List[Dict] = []
    for i, e in fresh.iterrows():
        df = fetch_window(cache, e["ticker"], e["peak_date"], e["signal_date"])
        f = flow_features(df, classes)
        if f:
            rows.append({"ticker": e["ticker"], "signal_date": e["signal_date"],
                         "bounced": bool(e["bounced_5pct"]),
                         "recovered": bool(e["recovered"]),
                         "drawdown": e["drawdown"], **f})
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(fresh)} ({len(rows)} with data)")

    R = pd.DataFrame(rows)
    R.to_csv("reports/pullback_replicate.csv", index=False)
    if R.empty:
        raise SystemExit("no data returned")

    a = R[R["bounced"]]["foreign_net"].dropna()
    b = R[~R["bounced"]]["foreign_net"].dropna()
    sp = np.sqrt(((len(a) - 1) * a.var() + (len(b) - 1) * b.var())
                 / (len(a) + len(b) - 2))
    d = (a.mean() - b.mean()) / sp if sp > 0 else np.nan
    t, p_two = stats.ttest_ind(a, b, equal_var=False)
    p_one = p_two / 2 if t > 0 else 1 - p_two / 2

    print(f"\n{'=' * 92}\n PRE-REGISTERED REPLICATION — foreign_net, one-sided, "
          f"alpha {ALPHA}\n{'=' * 92}")
    print(f" bounced      n={len(a):>4}  mean foreign_net {a.mean():+.4f}")
    print(f" did not      n={len(b):>4}  mean foreign_net {b.mean():+.4f}")
    print(f" difference   {a.mean() - b.mean():+.4f}   Cohen d {d:.3f}")
    print(f" t = {t:+.3f}   one-sided p = {p_one:.4f}")
    print(f"\n pilot was:   difference +0.0206, d 0.257, two-sided p 0.047")

    print(f"\n{'=' * 92}")
    if p_one < ALPHA and t > 0:
        print(f" REPLICATED. Foreign net buying into a pullback is associated with")
        print(f" the pullback bouncing, at p = {p_one:.4f} on data not used to")
        print(f" generate the hypothesis. Effect size d = {d:.3f}.")
        print(f"\n This is the first signal in this project that is not price, and it")
        print(f" is the one Result 93 said would have to exist. What it is NOT yet is")
        print(f" a trading rule: an association at d~0.25 has to survive being turned")
        print(f" into a decision, with costs, out of sample.")
    else:
        print(f" DID NOT REPLICATE. one-sided p = {p_one:.4f} against alpha {ALPHA}.")
        print(f" The pilot's p = 0.047 came out of 12 tests and did not survive")
        print(f" correction; this was the check, and it failed it.")
        print(f"\n With price, volume, stock trend, market trend AND foreign order")
        print(f" flow all returning nothing, the 72% false-exit rate stands")
        print(f" unexplained by anything measurable here.")
    print("\n -> reports/pullback_replicate.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
