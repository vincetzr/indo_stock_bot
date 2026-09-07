#!/usr/bin/env python3
"""Where your stops sit tonight, for names you actually hold.

    python3 scripts/positions.py --hold ARCI:2025-08-26 --hold ENRG:2025-08-26
    python3 scripts/positions.py --file positions.csv

`--file` takes a CSV with columns ``ticker,entry_date`` and optionally
``entry_price``. Everything is evaluated at the last session with adequate
cross-sectional coverage, the same as-of rule the daily brief uses.

WHAT THIS IS AND IS NOT
------------------------
It is the exit rules from `spine/exits.py`, the ones H17/H18 MEASURED,
evaluated FORWARD and printed as prices you could type into a broker screen.
Same code, so the monitor cannot drift from the study.

    THIS LINE SAID "the ones H17/H18 VALIDATED" AND THAT WAS A RETRACTED
    CLAIM. A18 withdrew both headlines: `trail 15% armed +50%` and `stoch
    rollover armed +50%` turn a 6.4x buy-and-hold into 1.6x and 1.4x and are
    the two WORST rules of the seven tested. A34 puts the running total at
    169 exit configurations across H17/H18/H35/H38/H40/H47 with NONE beating
    a hold. The sentence is kept, marked, rather than deleted, because A19
    records deleting a refuted claim as its own failure. These are levels to
    look at, not rules that were shown to work.

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
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import rules                                                    # noqa: E402
from idxbot import measured                                     # noqa: E402
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
    if getattr(a, "from_store", False):
        #  THE BOOK THE SYSTEM ITSELF SAID TO HOLD, rather than one retyped by
        #  hand. `signal_store` is append-only and already carries the entry,
        #  the stop and the target for every signal this repo has emitted, so
        #  a monitor that asks the user to re-enter them can drift from the
        #  record that will later be SCORED -- and then the thing being
        #  monitored is not the thing being measured.
        sys.path.insert(0, os.path.join(os.path.dirname(__file__),
                                        os.pardir, "src"))
        from idxbot import signal_store as ss                # noqa: PLC0415
        em = ss.load_emitted()
        if a.rule:
            em = em[em["rule"] == a.rule]
        if len(em):
            oc = ss.load_outcomes()
            #  Only what is still OPEN: a settled signal is history, not a
            #  position, and printing it as one would overstate the book.
            if len(oc) and "settled" in oc.columns:
                done = set(oc.loc[oc["settled"].astype(bool), "signal_id"])
                em = em[~em["signal_id"].isin(done)]
            for _i, r in em.iterrows():
                out.append({"ticker": r["ticker"],
                            "entry_date": str(pd.Timestamp(r["asof"]).date()),
                            "entry_price": float(r["entry"]),
                            "sl": float(r["sl"]), "tp": float(r["tp"]),
                            "rule": r["rule"]})
    #  Same ticker from two sources is one position, and the FIRST wins --
    #  a hand-entered fill is the real one, the store's is the model's close.
    seen, uniq = set(), []
    for d in out:
        if d["ticker"] in seen:
            continue
        seen.add(d["ticker"])
        uniq.append(d)
    return uniq


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default=PANEL)
    ap.add_argument("--indicators", default=IND)
    ap.add_argument("--hold", action="append",
                    help="TICKER:YYYY-MM-DD[:entry_price], repeatable")
    ap.add_argument("--file", help="CSV with ticker,entry_date[,entry_price]")
    ap.add_argument("--from-store", action="store_true",
                    help="monitor every OPEN signal in the append-only store")
    ap.add_argument("--rule", default="",
                    help="with --from-store, restrict to one rule name")
    ap.add_argument("--trail", type=float, default=0.15)
    ap.add_argument("--arm", type=float, default=0.50)
    ap.add_argument("--chandelier", type=float, default=3.0)
    ap.add_argument("--no-news", action="store_true")
    a = ap.parse_args()

    pos = _positions(a)
    if not pos:
        raise SystemExit("nothing to monitor — pass --hold, --file or "
                         "--from-store")

    P = pd.read_parquet(a.panel)
    P["date"] = pd.to_datetime(P["date"])
    I = pd.read_parquet(a.indicators)
    I["date"] = pd.to_datetime(I["date"])
    day = B.resolve_asof(P)
    warn = B.coverage_warning(P, day)

    F = M.position_frame(P, I, pos, day)
    #  `position_frame` returns only what IT computes, so the levels that came
    #  in with the position (from the store, or from a CSV) have to be carried
    #  across by ticker rather than assumed present.
    given = {d["ticker"]: d for d in pos}
    for key in ("sl", "tp", "rule"):
        F[key] = [given.get(t, {}).get(key, np.nan) for t in F["ticker"]]
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

        #  ------------------------------------------------------------
        #  THE SHIPPED LEVELS COME FIRST, AND THEY WERE MISSING.
        #  Everything below is the CATALOGUE of exit rules H17/H18 measured
        #  -- and A34 records 169 exit configurations of which NONE beat
        #  holding. Printing that catalogue while omitting the SL and TP the
        #  reader was actually given hands them a screen full of rules that
        #  lost money and none of the two the standing instruction requires.
        px_now = float(r.get("price", np.nan))
        for lab, key, sign in (("SL  (the level you were given)", "sl", -1),
                               ("TP  (the level you were given)", "tp", +1)):
            lvl = r.get(key)
            try:
                lvl = float(lvl)
            except (TypeError, ValueError):
                lvl = float("nan")
            if not np.isfinite(lvl) or lvl <= 0:
                continue
            dist = (lvl / px_now - 1.0) if np.isfinite(px_now) and px_now > 0 \
                else float("nan")
            fired = (np.isfinite(dist)
                     and ((sign < 0 and dist >= 0) or (sign > 0 and dist <= 0)))
            print(f"          {lab:<34} Rp {lvl:>9,.0f}  {dist:>+7.1%}"
                  + ("  <-- REACHED" if fired else ""))
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

    #  --------------------------------------------- THE FOURTH COLUMN
    #  The standing instruction's contract is ENTRY, SL, TP *and the measured
    #  cost of each level*, and it calls the fourth column the non-negotiable
    #  one. A42 wired the shipped levels into this screen and left their cost
    #  off it, so the monitor printed two levels with no measurement beside
    #  them and 169 catalogue rules with one. It is read from the study, never
    #  typed -- `measured.load()` refuses the file if it measured a different
    #  rule than the one those levels came from.
    mm = measured.load()
    base = mm.get(measured.BASE_ARM)
    ship = mm.get(measured.SHIPPED_ARM)
    gap = mm.cost(measured.SHIPPED_ARM)
    stop_arm = measured.STOP_ARMS.get(rules.STOP)
    print(" WHAT THESE TWO LEVELS COST, measured on the rule that shipped them:")
    if ship == ship and base == base:
        print(f"   together   {ship:.2%}/yr against {base:.2%} with neither -- "
              f"a {measured.noun(gap)} of")
        print(f"              {abs(gap) * 100:.2f} points of CAGR -- for a "
              f"drawdown of "
              f"{mm.get(measured.SHIPPED_ARM, measured.F_DD):.1%} against "
              f"{mm.get(measured.BASE_ARM, measured.F_DD):.1%}")
        print(f"              and a worst single name of "
              f"{mm.get(measured.SHIPPED_ARM, measured.F_WORST):.0%} against "
              f"{mm.get(measured.BASE_ARM, measured.F_WORST):.0%}.")
    if stop_arm and mm.get(stop_arm) == mm.get(stop_arm):
        print(f"   SL {-rules.STOP:.0%} alone  "
              f"{mm.get(stop_arm):.2%}/yr, drawdown "
              f"{mm.get(stop_arm, measured.F_DD):.1%}. The DRAWDOWN is the "
              f"whole case;")
        print("              the return effect is worse early and better late, "
              "which is")
        print("              A18's regime noise, so it is not claimed in "
              "either direction.")
    #  THE SHAPE IS CHECKED, NOT TYPED, for the same reason the figures are.
    #  S3's predicted null is that a target's cost is monotone in how tight it
    #  is; `rules.py` verifies that before printing the word and so does this.
    half = -mm.cost(measured.HALF_ARM)
    tpc = measured.family_costs(mm, measured.TP_ARMS)
    if half == half:
        shape = ("monotone" if tpc and measured.is_monotone(tpc.values())
                 else "NOT monotone")
        print(f"   TP {rules.TP:+.0%}    selling {rules.TP_FRAC:.0%} costs "
              f"{half * 100:.2f} points; a target's cost is {shape}")
        print("              in how tight it is, so a tighter one costs more.")
    else:
        print(f"   TP {rules.TP:+.0%}    no scale-out arm in the result file, "
              f"so its cost is not quoted")
    print(f"   SOURCE: {mm.provenance}. IN-SAMPLE: holdout spent at H16.\n")
    print(" The SL and TP rows are the levels the rule actually shipped and")
    print(" the ones the append-only store will SCORE. Everything under them")
    print(" is a catalogue: A34 records 169 exit configurations tested across")
    print(" H17/H18/H35/H38/H40/H47 and NONE beat simply holding.")
    print(" The trail and chandelier levels are the rules H17/H18 measured.")
    print(" A rule shown as 'not armed' CANNOT fire — that is why the")
    print(" measured P(-50%) barely moved: a name that falls from entry")
    print(" never arms an armed trail. If you want the left tail cut, the")
    print(" hard stop is the only line here that does it, and H17 measured")
    print(" the cost at 6-8 points of median return.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
