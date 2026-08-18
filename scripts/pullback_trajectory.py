#!/usr/bin/env python3
"""Capitulation is a trajectory, not a level. Test it that way.

Result 96 tested whether foreign flow *during* a pullback predicts a bounce, by
aggregating the whole window into one number. It found nothing: d = 0.002 on 600
events. But that test had a flaw worth naming, because it is the difference
between the right question and a nearby wrong one.

**Averaging the window erases the sequence.** The classic bottom is not "foreign
money was net buying throughout" - it is foreign money selling into the decline
and then turning, late, as the sellers exhaust. A window average of a seller who
turns buyer is roughly zero, which is exactly what the null looked like.

So this splits every pullback in half and asks a different question:

    does foreign flow IMPROVE from the first half of the fall to the second?

THE PRE-REGISTRATION, fixed before any fetch
--------------------------------------------
    hypothesis   delta_foreign = foreign_net(late half) - foreign_net(early half)
                 is HIGHER for pullbacks that bounce
    test         Welch t, ONE-SIDED, alpha 0.05
    primary      delta_foreign. ONE feature. Everything else below is
                 exploratory and is labelled as such, so the primary test needs
                 no correction.
    outcome      `bounced` - price >5% above the signal within 13 weeks
    sample       events with at least 4 weeks from peak to signal, so each half
                 has 2+ weeks of sessions to aggregate

Cost is two range requests per event rather than one. Everything is cached, so
the halves of an event already seen cost nothing on a rerun.

    python3 scripts/pullback_trajectory.py --limit 400
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
    ap.add_argument("--limit", type=int, default=400)
    ap.add_argument("--min-weeks", type=int, default=4)
    ap.add_argument("--seed", type=int, default=99)
    args = ap.parse_args()
    os.makedirs("reports", exist_ok=True)

    E = pd.read_csv("reports/pullback_events.csv",
                    parse_dates=["peak_date", "signal_date"])
    E = E[E["weeks_from_peak"] >= args.min_weeks].reset_index(drop=True)
    print(f"{len(E):,} events with {args.min_weeks}+ weeks from peak to signal")
    E = E.sample(n=min(args.limit, len(E)), random_state=args.seed)
    E = E.sort_values(["ticker", "signal_date"]).reset_index(drop=True)

    n1 = int(E["bounced_5pct"].sum())
    n2 = len(E) - n1
    mde = (1.645 + 0.84) * np.sqrt(1 / max(n1, 1) + 1 / max(n2, 1))
    print(f"sample: {len(E)} events, {E['ticker'].nunique()} names, "
          f"split {n1}/{n2}")
    print(f"one-sided MDE at 80% power: d = {mde:.2f}")

    cfg = load_config()
    cache = Cache(cfg.path("data.cache_dir", "data/cache"))
    reg = cfg.brokers
    classes = {c.upper(): ("bumn" if getattr(reg.get(c), "state_owned", False)
                           else "foreign" if getattr(reg.get(c), "foreign", False)
                           else "local")
               for c in reg.codes() if reg.get(c) is not None}

    rows: List[Dict] = []
    for i, e in E.iterrows():
        peak, sig = e["peak_date"], e["signal_date"]
        mid = peak + (sig - peak) / 2
        early = fetch_window(cache, e["ticker"], peak, mid)
        late = fetch_window(cache, e["ticker"], mid, sig)
        fe = flow_features(early, classes)
        fl = flow_features(late, classes)
        if not fe or not fl:
            continue
        rows.append({
            "ticker": e["ticker"], "signal_date": sig,
            "bounced": bool(e["bounced_5pct"]),
            "drawdown": e["drawdown"], "weeks": e["weeks_from_peak"],
            "foreign_early": fe["foreign_net"], "foreign_late": fl["foreign_net"],
            "delta_foreign": fl["foreign_net"] - fe["foreign_net"],
            "delta_bumn": fl["bumn_net"] - fe["bumn_net"],
            "delta_local": fl["local_net"] - fe["local_net"],
            "delta_imbalance": fl["imbalance"] - fe["imbalance"],
            "delta_conc": fl["concentration"] - fe["concentration"],
        })
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(E)} ({len(rows)} usable)")

    R = pd.DataFrame(rows)
    R.to_csv("reports/pullback_trajectory.csv", index=False)
    if R.empty:
        raise SystemExit("no data")

    a = R[R["bounced"]]["delta_foreign"].dropna()
    b = R[~R["bounced"]]["delta_foreign"].dropna()
    sp = np.sqrt(((len(a) - 1) * a.var() + (len(b) - 1) * b.var())
                 / (len(a) + len(b) - 2))
    d = (a.mean() - b.mean()) / sp if sp > 0 else np.nan
    t, p2 = stats.ttest_ind(a, b, equal_var=False)
    p1 = p2 / 2 if t > 0 else 1 - p2 / 2

    print(f"\n{'=' * 92}\n PRIMARY TEST — delta_foreign, one-sided, alpha {ALPHA}"
          f"\n{'=' * 92}")
    print(f" bounced   n={len(a):>4}   early {R[R['bounced']]['foreign_early'].mean():+.4f}"
          f"  ->  late {R[R['bounced']]['foreign_late'].mean():+.4f}"
          f"   delta {a.mean():+.4f}")
    print(f" did not   n={len(b):>4}   early {R[~R['bounced']]['foreign_early'].mean():+.4f}"
          f"  ->  late {R[~R['bounced']]['foreign_late'].mean():+.4f}"
          f"   delta {b.mean():+.4f}")
    print(f"\n difference {a.mean() - b.mean():+.4f}   Cohen d {d:.3f}   "
          f"t {t:+.3f}   one-sided p {p1:.4f}")
    print(f"\n {'SIGNIFICANT' if (p1 < ALPHA and t > 0) else 'NOT SIGNIFICANT'} "
          f"at alpha {ALPHA}")

    print(f"\n{'-' * 92}\n EXPLORATORY (not corrected, not evidence on their own)"
          f"\n{'-' * 92}")
    print(f" {'feature':<18}{'bounced':>11}{'did not':>11}{'d':>8}{'two-sided p':>14}")
    for c in ("delta_bumn", "delta_local", "delta_imbalance", "delta_conc",
              "foreign_late", "foreign_early"):
        x = R[R["bounced"]][c].dropna()
        y = R[~R["bounced"]][c].dropna()
        if len(x) < 5 or len(y) < 5:
            continue
        s2 = np.sqrt(((len(x) - 1) * x.var() + (len(y) - 1) * y.var())
                     / (len(x) + len(y) - 2))
        dd = (x.mean() - y.mean()) / s2 if s2 > 0 else np.nan
        tt, pp = stats.ttest_ind(x, y, equal_var=False)
        print(f" {c:<18}{x.mean():>+11.4f}{y.mean():>+11.4f}{dd:>8.3f}{pp:>14.3f}")
    print("\n -> reports/pullback_trajectory.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
