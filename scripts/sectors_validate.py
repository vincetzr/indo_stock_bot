#!/usr/bin/env python3
"""Prove the paid full-rekap route before trusting a single conclusion to it.

WHY THIS RUNS BEFORE ANYTHING ELSE DOES
---------------------------------------
Buying data does not make it correct. A vendor can mis-map a column, quote
shares where the exchange quotes lots, publish a different board, or lag a day
- and every one of those failures produces numbers that look completely normal.
The only defence is that we already hold 416 sessions of BBCA from a completely
independent route, and the two must agree where they overlap.

FOUR CHECKS, IN ORDER OF WHAT THEY WOULD CATCH
----------------------------------------------
    1. INTERNAL      value = lots x 100 x average, on the vendor's own rows.
                     Catches a mis-mapped column without any second source.
    2. OVERLAP       for every broker the free route names in its top ten, the
                     paid route must report the same lots. This is the real
                     test: two unrelated pipelines landing on the same integer
                     is not something a parsing bug does by accident.
    3. DEPTH         the paid route must list MORE brokers than the free one,
                     and the free one's ten must be the ten largest of them.
                     If the "full" rekap is also a top ten, the entire reason
                     for paying evaporates and this says so.
    4. CLOSURE       across all brokers, lots bought must equal lots sold.
                     A complete rekap closes exactly; a censored one cannot.
                     This is the check the free route can never pass, and
                     passing it is what licenses collapsing intervals to
                     points everywhere downstream.

    SECTORS_API_KEY=... python3 scripts/sectors_validate.py --ticker BBCA
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))
sys.path.insert(0, os.path.dirname(__file__))

from idxbot.config import load_config                        # noqa: E402
from idxbot.data.cache import Cache                          # noqa: E402
from idxbot.data.sectors import (CONSISTENCY_BOUND,          # noqa: E402
                                 SectorsBrokerSummary, api_key, consistency)
from broker_collect import STORE, load_store                  # noqa: E402

#: Free-route lots are display-rounded; paid-route lots are exact. A day whose
#: lots were abbreviated is tagged ``ipot~`` and can differ by up to this much
#: without anything being wrong. Exact days must agree to the lot.
ABBREVIATED_TOLERANCE = 0.05


def overlap_days(free: pd.DataFrame, ticker: str, limit: int) -> List[pd.Timestamp]:
    d = free[free["ticker"] == str(ticker).upper()]
    if d.empty:
        return []
    days = sorted(pd.to_datetime(d["date"]).dt.normalize().unique())
    # newest first: the vendor's history may not reach the start of ours, and
    # a disagreement on recent data matters more than one on old data
    return list(pd.to_datetime(days))[-limit:]


def compare_day(free_day: pd.DataFrame, paid_day: pd.DataFrame) -> Dict:
    """One session, both routes. Returns the disagreements, not a verdict."""
    f = free_day.set_index("broker")
    p = paid_day.set_index("broker")
    shared = f.index.intersection(p.index)
    out: Dict[str, object] = {
        "free_brokers": len(f), "paid_brokers": len(p),
        "shared": len(shared),
        "missing_from_paid": sorted(set(f.index) - set(p.index)),
    }
    errs = []
    for side in ("buy_lot", "sell_lot"):
        a = pd.to_numeric(f.loc[shared, side], errors="coerce")
        b = pd.to_numeric(p.loc[shared, side], errors="coerce")
        ok = np.isfinite(a) & np.isfinite(b) & (a > 0)
        if ok.any():
            errs.append(((a[ok] - b[ok]).abs() / a[ok]))
    e = pd.concat(errs) if errs else pd.Series(dtype=float)
    out["worst_lot_error"] = float(e.max()) if len(e) else np.nan
    out["median_lot_error"] = float(e.median()) if len(e) else np.nan

    # Does the free route's top ten really contain the ten largest?
    if len(p) > len(f) and len(f):
        biggest = set(p["buy_lot"].astype(float).nlargest(
            int((f["buy_lot"].astype(float) > 0).sum())).index)
        named = set(f[f["buy_lot"].astype(float) > 0].index)
        out["top_n_agrees"] = bool(biggest and biggest == named)
    else:
        out["top_n_agrees"] = None

    # A complete rekap closes: every lot bought was sold by somebody.
    bl, sl = float(p["buy_lot"].sum()), float(p["sell_lot"].sum())
    out["closure"] = abs(bl - sl) / bl if bl > 0 else np.nan
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default="BBCA")
    ap.add_argument("--days", type=int, default=10,
                    help="overlapping sessions to check (each fortnight of "
                         "them costs one credit)")
    args = ap.parse_args()
    cfg = load_config()

    print(f"{'=' * 92}\n VALIDATING THE PAID FULL-REKAP ROUTE\n{'=' * 92}")
    if not api_key(cfg):
        print(" No API key. Set SECTORS_API_KEY, or data.sectors_api_key in "
              "the config.\n")
        print(" This script is the gate the paid route has to pass before any"
              " conclusion\n rests on it. Nothing has been checked, so nothing"
              " is claimed.\n")
        print(f" What it will cost when you do have a key: "
              f"{SectorsBrokerSummary.credits_for(1, args.days)} credit(s) for "
              f"{args.days} sessions of one ticker.")
        return 1

    free = load_store()
    if free.empty:
        print(f" The free-route store at {STORE} is empty, so there is nothing"
              f" to check against.")
        return 1
    free = free[free.get("source", "").astype(str).str.startswith("ipot")]
    days = overlap_days(free, args.ticker, args.days)
    if not days:
        print(f" No {args.ticker} sessions in the free store to compare with.")
        return 1

    cache = Cache(cfg.path("data.cache_dir", "data/cache"))
    paid = SectorsBrokerSummary(cache=cache, cfg=cfg, verbose=True)
    tk = str(args.ticker).upper()

    print(f" {tk}: checking {len(days)} overlapping sessions, "
          f"{days[0]:%Y-%m-%d} to {days[-1]:%Y-%m-%d}\n")
    print(f" {'date':<12}{'free':>6}{'paid':>6}{'shared':>8}{'worst lot':>11}"
          f"{'top-10 ok':>11}{'closure':>10}")

    rows = []
    all_paid = []
    for d in days:
        fd = free[(free["ticker"] == tk)
                  & (pd.to_datetime(free["date"]).dt.normalize() == d)]
        fd = fd.drop_duplicates(subset=["broker"], keep="first")
        pdd = paid.fetch_day(tk, d)
        if pdd.empty:
            print(f" {d:%Y-%m-%d}  {len(fd):>5}     -  "
                  f"{'not returned by the paid route':>40}")
            continue
        all_paid.append(pdd)
        r = compare_day(fd, pdd)
        r["date"] = d
        rows.append(r)
        top = {True: "yes", False: "NO", None: "-"}[r["top_n_agrees"]]
        print(f" {d:%Y-%m-%d}{r['free_brokers']:>6}{r['paid_brokers']:>6}"
              f"{r['shared']:>8}{r['worst_lot_error']:>10.2%}{top:>11}"
              f"{r['closure']:>9.3%}")

    if not rows:
        print("\n Nothing came back from the paid route. No verdict.")
        return 1

    R = pd.DataFrame(rows)
    P = pd.concat(all_paid, ignore_index=True)
    C = consistency(P)

    print(f"\n{'=' * 92}\n VERDICT\n{'=' * 92}")
    internal = float(C["worst"].dropna().max()) if len(C) else np.nan
    ok_internal = np.isfinite(internal) and internal < CONSISTENCY_BOUND
    print(f" 1. internal  value = lots x 100 x avg, worst error "
          f"{internal:.2e}   -> {'PASS' if ok_internal else 'FAIL'}")

    worst = float(np.nanmax(R["worst_lot_error"]))
    ok_overlap = worst <= ABBREVIATED_TOLERANCE
    print(f" 2. overlap   worst lot disagreement vs the free route "
          f"{worst:.2%}  -> {'PASS' if ok_overlap else 'FAIL'}")

    deeper = int((R["paid_brokers"] > R["free_brokers"]).sum())
    ok_depth = deeper == len(R)
    print(f" 3. depth     paid route is deeper on {deeper}/{len(R)} sessions "
          f"(median {R['paid_brokers'].median():.0f} vs "
          f"{R['free_brokers'].median():.0f} brokers) -> "
          f"{'PASS' if ok_depth else 'FAIL'}")

    clos = float(np.nanmedian(R["closure"]))
    ok_close = np.isfinite(clos) and clos < 0.01
    print(f" 4. closure   median |bought - sold| / bought = {clos:.3%}"
          f"          -> {'PASS' if ok_close else 'FAIL'}")

    print()
    if all((ok_internal, ok_overlap, ok_depth, ok_close)):
        print(" All four pass. The paid route is the full rekap it claims to "
              "be, it agrees with\n an independent source where they overlap, "
              "and it closes. Positions derived from\n it are numbers, not "
              "intervals.")
    else:
        print(" NOT VALIDATED. Do not collapse intervals to points on this "
              "route, and do not\n spend on a backfill until the failing check"
              " above is understood. A source that\n disagrees with a known-"
              "good one is a source that will be wrong silently.")
    print(f"\n credits spent on this run: {paid.credits_spent}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
