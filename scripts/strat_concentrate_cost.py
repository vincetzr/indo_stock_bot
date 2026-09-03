#!/usr/bin/env python3
"""The cost correction the harness cannot make for itself.

`Bench.walk` charges `turn = 0.5*sum|w_target_now - w_target_prev|`. The
previous TARGET is not the portfolio's actual weights when the next rebalance
arrives -- prices moved in between. So:

  * an EQUAL-WEIGHT arm holding a stable basket reads turnover ~0 and pays
    almost nothing, when in reality resetting a 10-name book to equal weight
    after a year of dispersion is a real, sizeable trade;
  * a genuine no-trade DRIFT arm reads turnover > 0 and pays for trades that
    were never made.

Both are small in absolute terms here, but the first flatters exactly the arm
that comes out best, so it has to be measured rather than waved at. This script
recomputes the TRUE turnover -- target weights against the drifted weights the
book actually carries into the rebalance -- and reports what each arm's CAGR
becomes once the difference is charged at the harness's own per-name
fraksi-harga toll.

The drift is computed from `Bench.PX` at the rebalance bar only, so it uses no
information stamped after the decision.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

from bhbench import Bench, load, report  # noqa: E402
from paint_suite import tick_of          # noqa: E402
from strat_concentrate import (VARIANTS, FROZEN, HOLDALL, add_size,  # noqa: E402
                               make_select)

ARMS = ["j* size10 STICKY EW", "l* size5  STICKY EW", "t* FROZEN-HOLDALL EW",
        "k* size10 STICKY DRIFT", "s* FROZEN-HOLDALL DRIFT",
        "o* calm+strong10 of top60", "u* size10 STICKY EW 2-yearly",
        "i* liq10 STICKY EW"]


def audit(B: Bench, inner, freq: int):
    """Run one arm, tracking the weights the book really carries."""
    st = {"prev": None, "prev_date": None, "extra": 0.0, "harness": 0.0,
          "true": 0.0, "n": 0}

    def select(day: pd.DataFrame):
        d0 = day["date"].iloc[0]
        tgt = dict(inner(day))
        if not tgt:
            return []
        tot = sum(tgt.values())
        tgt = {t: w / tot for t, w in tgt.items()}
        prev, pd0 = st["prev"], st["prev_date"]
        if prev:
            # what the previous target has grown into by today
            grown = {}
            for t, w in prev.items():
                p0 = B.PX.at(t, pd0)
                p1 = B.PX.exit_price(t, d0)
                grown[t] = w * (p1 / p0) if (np.isfinite(p0) and p0 > 0
                                             and np.isfinite(p1) and p1 > 0) else w
            g = sum(grown.values())
            grown = {t: w / g for t, w in grown.items()} if g > 0 else prev
            keys = set(tgt) | set(grown)
            true_turn = 0.5 * sum(abs(tgt.get(k, 0.0) - grown.get(k, 0.0))
                                  for k in keys)
            harness_turn = 0.5 * sum(abs(tgt.get(k, 0.0) - prev.get(k, 0.0))
                                     for k in set(tgt) | set(prev))
            tolls = []
            for t in tgt:
                p = B.PX.at(t, d0)
                if np.isfinite(p) and p > 0:
                    tolls.append(B.fee + B.spread_mult * tick_of(p) / p)
            toll = float(np.mean(tolls)) if tolls else B.fee
            st["true"] += true_turn * toll
            st["harness"] += harness_turn * toll
            st["extra"] += (true_turn - harness_turn) * toll
            st["n"] += 1
        st["prev"], st["prev_date"] = tgt, d0
        return list(tgt.items())

    return select, st


def main() -> None:
    P = add_size(load())
    B = Bench(P)
    spec = {v[0]: v for v in VARIANTS}
    rows = []
    for lbl in ARMS:
        _, n, sc, uni, drift, sticky, wm, freq = spec[lbl]
        fz, ha = lbl in FROZEN, lbl in HOLDALL

        def mkinner():
            return make_select(n, sc, uni, drift, sticky, wm,
                               frozen=fz, hold_all=ha)

        sel, st = audit(B, mkinner(), freq)
        v = B.evaluate(sel, label=lbl, freq=freq)
        if not v.get("ok"):
            print(report(v))
            continue
        yrs = v["years"]
        extra_yr = st["extra"] / yrs
        # cost is a drag on the growth factor, so subtract it in log space
        adj = (1.0 + v["cagr"]) * np.exp(-extra_yr) - 1.0
        rows.append({
            "label": lbl, "freq": freq, "cagr": v["cagr"],
            "true_turn_cost_yr": st["true"] / yrs,
            "harness_turn_cost_yr": st["harness"] / yrs,
            "undercharge_yr": extra_yr, "cagr_true_cost": adj,
            "bh_index": v["bh_index"], "bh_universe": v["bh_universe"],
            "bh_picks": v["bh_picks"], "random": v["random"],
            "still_beats_all": bool(adj > max(v["bh_index"], v["bh_universe"],
                                              v["bh_picks"], v["random"])),
        })
        print(f"{lbl:<30} cagr {v['cagr']:+7.2%}  harness cost "
              f"{st['harness'] / yrs:.3%}/yr  TRUE cost {st['true'] / yrs:.3%}/yr"
              f"  -> corrected {adj:+7.2%}   (index {v['bh_index']:+.2%}, "
              f"universe {v['bh_universe']:+.2%}, picks {v['bh_picks']:+.2%})")
    T = pd.DataFrame(rows)
    os.makedirs("reports", exist_ok=True)
    T.to_csv("reports/strat_concentrate_cost.csv", index=False)
    print()
    print(T.to_string(index=False))


if __name__ == "__main__":
    main()
