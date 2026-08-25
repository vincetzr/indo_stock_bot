#!/usr/bin/env python3
"""Where your stops sit tonight, for names you actually hold.

    python3 scripts/positions.py --hold ARCI:2025-08-26 --hold ENRG:2025-08-26
    python3 scripts/positions.py --file positions.csv

`--file` takes a CSV with columns ``ticker,entry_date`` and optionally
``entry_price``. Everything is evaluated at the last session with adequate
cross-sectional coverage, the same as-of rule the daily brief uses.

WHAT THIS IS AND IS NOT
------------------------
It is the exit rules from `spine/exits.py`, the ones H17/H18 validated,
evaluated FORWARD and printed as prices you could type into a broker screen.
Same code, so the monitor cannot drift from the study.

It is NOT a recommendation, and two of its columns are explicitly weaker than
the rest. The stochastic and volume readings are STATE, not validated rules —
read H18's table before acting on them. The event tags come from public RSS and
have **never been backtested and cannot be**: there is no point-in-time news
archive, so a news-conditioned rule would be look-ahead by construction. They
are printed because a suspension is a fact about whether you can trade at all.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

from idxbot.report import brief as B                            # noqa: E402
from idxbot.report import monitor as M                          # noqa: E402
from idxbot.spine import exits as X                             # noqa: E402

#: The rules replayed per position. The H17 incumbent, the H18 challenger,
#: the two moving-average breaks and a hard floor — the ones with a measured
#: number attached, not the whole catalogue.
RULES = {
    "trail 15% armed +50%": X.catalogue()["trail 15% armed +50%"],
    "chandelier 3x ATR armed +50%":
        X.indicator_catalogue()["chandelier 3x ATR armed +50%"],
    "chandelier 3x ATR": X.indicator_catalogue()["chandelier 3x ATR"],
    "ema20 break": X.indicator_catalogue()["ema20 break"],
    "ema50 break": X.indicator_catalogue()["ema50 break"],
    "stop 25%": X.catalogue()["stop 25%"],
    "hold 252": X.catalogue()["hold 252"],
}

PANEL = os.path.join("data", "spine", "price_panel.parquet")
IND = os.path.join("data", "spine", "indicator_panel.parquet")


def _positions(a) -> list:
    out = []
    for spec in (a.hold or []):
        bits = spec.split(":")
        if len(bits) < 2:
            raise SystemExit(f"--hold wants TICKER:YYYY-MM-DD[:price], got {spec}")
        d = {"ticker": bits[0], "entry_date": bits[1]}
        if len(bits) > 2:
            d["entry_price"] = float(bits[2])
        out.append(d)
    if a.file:
        df = pd.read_csv(a.file)
        out += df.to_dict("records")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default=PANEL)
    ap.add_argument("--indicators", default=IND)
    ap.add_argument("--hold", action="append",
                    help="TICKER:YYYY-MM-DD[:entry_price], repeatable")
    ap.add_argument("--file", help="CSV with ticker,entry_date[,entry_price]")
    ap.add_argument("--trail", type=float, default=0.15)
    ap.add_argument("--arm", type=float, default=0.50)
    ap.add_argument("--chandelier", type=float, default=3.0)
    ap.add_argument("--no-news", action="store_true")
    a = ap.parse_args()

    pos = _positions(a)
    if not pos:
        raise SystemExit("nothing to monitor — pass --hold or --file")

    P = pd.read_parquet(a.panel)
    P["date"] = pd.to_datetime(P["date"])
    I = pd.read_parquet(a.indicators)
    I["date"] = pd.to_datetime(I["date"])
    day = B.resolve_asof(P)
    warn = B.coverage_warning(P, day)

    F = M.position_frame(P, I, pos, day)
    tags = {} if a.no_news else M.event_tags(
        [r["ticker"] for r in pos])

    print("=" * 78)
    print(f" POSITION MONITOR — as of {pd.Timestamp(day).date()}")
    print("=" * 78)
    if warn:
        print(f" {warn}")
    print(" levels are where each rule fires on the NEXT close, in quoted"
          " rupiah.\n")

    for _, r in F.iterrows():
        if r.get("status") != "held":
            print(f" {r['ticker']:<6} {r.get('status', 'unknown')}\n")
            continue
        print(f" {r['ticker']:<6} Rp {r['price']:,.0f}   "
              f"{r['gain']:+.1%} from entry ({pd.Timestamp(r['entry_date']).date()}, "
              f"{r['sessions']} sessions)")
        print(f"        peak {r['peak_gain']:+.1%}, now {r['give_back']:+.1%} "
              f"off that peak")

        L = M.levels(r, arm=a.arm, trail=a.trail, chand_k=a.chandelier)
        hit = L[L["active"] & (L["distance"] >= 0)]
        for _, x in L.iterrows():
            if not x["active"]:
                print(f"          {x['rule']:<34} —          {x['note']}")
            elif not np.isfinite(x["level"]):
                print(f"          {x['rule']:<34} —          unavailable")
            else:
                # a level ABOVE today's price is a rule that fired
                # weeks ago, not a fresh signal — the replay below dates it
                flag = ("  <-- already fired, see replay"
                        if x["distance"] >= 0 else "")
                print(f"          {x['rule']:<34} Rp {x['level']:>9,.0f}"
                      f"  {x['distance']:>+7.1%}{flag}")
        n = M.nearest_trigger(L)
        if n is not None and np.isfinite(n["level"]) and n["distance"] < 0:
            print(f"        nearest live stop: {n['rule']} at "
                  f"Rp {n['level']:,.0f} ({n['distance']:+.1%})")

        # A level far ABOVE today's price is a stale trigger, not a fresh one.
        # Replay the rules over the realised path and say when they fired.
        R = M.replay(P, I, r, {k: v for k, v in RULES.items()})
        if not R.empty:
            fired = R[R["fired"]]
            if len(fired):
                print("        rules that ALREADY exited this position:")
                for _, x in fired.iterrows():
                    print(f"          {x['rule']:<34} "
                          f"{pd.Timestamp(x['date']).date()}  "
                          f"Rp {x['price']:>9,.0f}  {x['gross']:>+8.1%} gross"
                          f"  ({x['sessions']}d)")
            still = R[~R["fired"]]
            if len(still):
                print(f"        still holding under: "
                      f"{', '.join(still['rule'].tolist())}")
        osc = M.oscillator_state(r)
        if osc:
            print(f"        state (not a validated rule): {osc}")
        if r["ticker"] in tags:
            print(f"        EVENT TAGS (never backtested): "
                  f"{', '.join(tags[r['ticker']])}")
        print()

    print(" The trail and chandelier levels are the rules H17/H18 measured.")
    print(" A rule shown as 'not armed' CANNOT fire — that is why the")
    print(" measured P(-50%) barely moved: a name that falls from entry")
    print(" never arms an armed trail. If you want the left tail cut, the")
    print(" hard stop is the only line here that does it, and H17 measured")
    print(" the cost at 6-8 points of median return.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
