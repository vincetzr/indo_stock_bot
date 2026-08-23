#!/usr/bin/env python3
"""Collect the FOREIGN / DOMESTIC investor-type views of the rekap.

WHY THIS IS A DIFFERENT QUANTITY FROM THE PANEL'S ``foreign_net``
-------------------------------------------------------------------
`scripts/flow_panel_build.flow_features` computes ``foreign_net`` by summing
the brokers this repo's config FLAGS as foreign-owned. That is a proxy for
investor domicile and it is not a good one. IndoPremier's own endpoint will
filter the rekap by the investor's domicile — the flag IDX attaches to the
TRADE, not to the member — and the two disagree materially:

    BBCA, 28 sessions with all three views cached
        correlation of the two nets   +0.749
        SIGN DISAGREES                29% of sessions
        foreign share of gross        74.6% (domicile) vs 65.0% (broker flag)

A foreign-owned member executes for domestic clients all day, and YP (Mirae) is
the largest RETAIL broker in the country while carrying a foreign flag. So the
broker-flag version answers "how much did foreign-owned MEMBERS trade", and
§12's question is about who the INVESTOR was.

WHAT THE ENDPOINT GIVES, AND WHAT IT COSTS
--------------------------------------------
``fd=F`` and ``fd=D`` compose with ``start``/``end``, so the investor split
costs the same one-request-per-(ticker, window) as the combined panel did —
verified live, not assumed. Three properties were checked against 28 cached
BBCA sessions before any collection was planned:

    F + D = all      footer totals reconcile to 0.023% median
    F view = F.NVal  the F view's net reproduces the published net-foreign
                     figure to 1.1%, while summing foreign-FLAGGED brokers
                     misses it badly
    range = sum of days   every broker present in both reconciles inside
                     abbreviation error; brokers missing from the range view
                     are the ones too small for its top ten, which is
                     censoring and not disagreement

BOTH VIEWS ARE COLLECTED, NOT ONE PLUS A SUBTRACTION
------------------------------------------------------
D could in principle be derived as all − F. It is not, for two reasons. The
combined run threw the footer away (``pullback_flow.fetch_window`` calls
``parse_table`` and never ``attach_totals``), so the totals that would make the
subtraction exact do not exist for those 31,824 windows. And each view is
independently top-ten censored, so a per-broker subtraction across views is
not a difference of like with like. Collecting both makes the identity a CHECK
rather than an assumption — which is the same discipline that caught three
estimator bugs earlier in this repo.

THE FOOTER IS KEPT THIS TIME
-----------------------------
``attach_totals`` is called here. The combined collection dropped it, and that
single omission is why net-foreign value is unavailable for the existing panel
without re-fetching all of it. The footer is uncensored: it is the view's own
total, so it bounds exactly how much of each view the visible top ten covers.

UNIVERSE — WHERE THIS TEST HAS ANY POWER
------------------------------------------
Foreign participation tracks liquidity hard. Measured on one fortnight:

    decile 9  49.4%      decile 7  19.4%      decile 5   3.4%
    decile 8  30.0%

In the bottom half of the panel the foreign side is a rounding error, so a
foreign-versus-domestic comparison there measures noise at some expense to
someone else's server. ``--deciles`` defaults to the top three accordingly, and
the restriction is a statement about power rather than a convenience.

POLITENESS IS A CONSTRAINT, NOT A SETTING
------------------------------------------
A5 again: polite delay, permanent cache, no bulk harvesting, no redistribution.
``--budget`` bounds a run on purpose, the cache is permanent, and the breaker
stops a loop that is firing into a server which has started refusing.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))
sys.path.insert(0, os.path.dirname(__file__))

from idxbot.data.cache import Cache                            # noqa: E402
from idxbot.data.ipot import (BASE_URL, attach_totals,         # noqa: E402
                              parse_table, parse_totals)
from pullback_flow import HEADERS                              # noqa: E402
from flow_panel_collect import (BREAKER_ROUNDS, BREAKER_SLEEP,  # noqa: E402
                                BREAKER_TRIP, EMPTY_LEDGER,
                                HTTP_RETRIES, HTTP_TIMEOUT,
                                listed_span, load_empty, load_panel,
                                not_served, windows)

VIEWS = ("F", "D")


def cache_key(ticker: str, start: pd.Timestamp, end: pd.Timestamp,
              board: str, view: str) -> str:
    """The view MUST be in the key.

    all / F / D are three different tables for the same ticker and window, and
    a key that omitted the view would serve a foreign-only table to a caller
    asking for the whole market. ``ipot.fetch_day`` learned this already; the
    range path needs the same discipline.
    """
    return f"{ticker}_{start:%Y%m%d}_{end:%Y%m%d}_{board}_{view}_range"


def fetch_one(cache: Cache, ticker: str, start: pd.Timestamp,
              end: pd.Timestamp, view: str, delay: float, board: str,
              timeout: float = HTTP_TIMEOUT,
              retries: int = HTTP_RETRIES) -> str:
    """One window, one view. Returns 'ok', 'empty' or 'error'.

    The three are kept distinct because a network failure must trip the
    breaker and an empty table must not: the first means the source is
    refusing, the second means it answered and had nothing.
    """
    import requests

    key = cache_key(ticker, start, end, board, view)
    r = None
    for _ in range(retries + 1):
        time.sleep(delay)
        try:
            r = requests.get(BASE_URL, timeout=timeout, headers=HEADERS,
                             params={"code": ticker, "board": board,
                                     "fd": view,
                                     "start": start.strftime("%Y-%m-%d"),
                                     "end": end.strftime("%Y-%m-%d")})
            r.raise_for_status()
            break
        except Exception:                                       # noqa: BLE001
            r = None
    if r is None:
        return "error"
    df = parse_table(r.text, ticker, end)
    if df is None or df.empty:
        with open(EMPTY_LEDGER, "a") as f:
            f.write(key + "\n")
        return "empty"
    # The footer, which the combined collection discarded. It is the view's own
    # uncensored total, so it is what turns a top-ten table into a censored
    # sample of KNOWN size rather than a biased one of unknown size.
    cache.write("ipot_broker", key, attach_totals(df, parse_totals(r.text)))
    return "ok"


def plan(P: pd.DataFrame, wins, spans: Dict[str, Optional[Tuple]],
         views: Tuple[str, ...]
         ) -> List[Tuple[pd.Timestamp, pd.Timestamp, str, str]]:
    """Every (window, ticker, view) worth a request, newest window first.

    Window-major so that stopping at any point leaves a COMPLETE cross-section
    for every fortnight collected, which is the shape the analysis consumes.
    Both views of one ticker-window are adjacent so a partial run does not
    leave F without its D.
    """
    skip = not_served()
    jobs = []
    for a, b in reversed(wins):
        for t in P["ticker"]:
            if t in skip:
                continue
            sp = spans.get(t)
            if sp is None or b < sp[0] or a > sp[1]:
                continue
            for v in views:
                jobs.append((a, b, t, v))
    return jobs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2014-01-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--step", type=int, default=10)
    ap.add_argument("--deciles", default="7,8,9",
                    help="liquidity strata to collect; foreign participation "
                         "below these is a rounding error")
    ap.add_argument("--views", default="F,D")
    ap.add_argument("--budget", type=int, default=4000,
                    help="hard cap on NETWORK requests for this run")
    ap.add_argument("--max-seconds", type=float, default=None,
                    help="stop cleanly after this long. Background processes "
                         "are frozen at the end of a turn here, so collection "
                         "only advances while a command runs and every run is "
                         "a bounded slice.")
    ap.add_argument("--delay", type=float, default=1.3)
    ap.add_argument("--latency", type=float, default=3.2,
                    help="MEASURED seconds per request including network")
    ap.add_argument("--board", default="RG")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    views = tuple(v.strip().upper() for v in a.views.split(",") if v.strip())
    for v in views:
        if v not in VIEWS:
            print(f"view must be one of {VIEWS}, got {v!r}")
            return 1
    keep = {int(d) for d in a.deciles.split(",") if d.strip()}

    end = a.end or (pd.Timestamp.today().normalize()
                    - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    P = load_panel()
    if P.empty:
        print("no panel; run scripts/flow_panel_universe.py first")
        return 1
    P = P[P["decile"].isin(keep)].reset_index(drop=True)
    wins = windows(a.start, end, a.step)
    spans = {t: listed_span(t, s) for t, s in zip(P["ticker"], P["src"])}
    jobs = plan(P, wins, spans, views)

    cache = Cache(os.path.join("data", "cache"))
    known_empty = load_empty()
    todo, skipped_empty = [], 0
    for s, e, t, v in jobs:
        key = cache_key(t, s, e, a.board, v)
        if key in known_empty:
            skipped_empty += 1
            continue
        if cache.read("ipot_broker", key) is None:
            todo.append((s, e, t, v))

    print(f"universe   {len(P)} names, deciles {sorted(keep)} "
          f"({int((P.src=='delisted').sum())} delisted)")
    print(f"views      {', '.join(views)}")
    print(f"windows    {len(wins)} x {a.step} business days, "
          f"{a.start} .. {end}")
    print(f"requests   {len(jobs):,} listed (ticker, window, view) triples")
    print(f"cached     {len(jobs) - len(todo) - skipped_empty:,}")
    print(f"known empty {skipped_empty:,} (asked once, answered nothing)")
    print(f"REMAINING  {len(todo):,}  "
          f"~{len(todo) * a.latency / 3600:.1f}h at {a.latency}s/request")
    if a.dry_run:
        return 0

    t0 = time.time()
    done = {"ok": 0, "empty": 0, "error": 0}
    fails = rounds = 0
    for s, e, tk, vw in todo:
        if sum(done.values()) >= a.budget:
            print(f"\n  budget of {a.budget} requests reached")
            break
        if a.max_seconds and time.time() - t0 > a.max_seconds:
            print(f"\n  stopping cleanly at {a.max_seconds}s")
            break
        r = fetch_one(cache, tk, s, e, vw, a.delay, a.board)
        done[r] += 1
        if r == "error":
            fails += 1
            if fails >= BREAKER_TRIP:
                rounds += 1
                if rounds > BREAKER_ROUNDS:
                    print("\n  breaker: the source is refusing; stopping")
                    break
                print(f"\n  breaker round {rounds}: backing off "
                      f"{BREAKER_SLEEP}s")
                time.sleep(BREAKER_SLEEP)
                fails = 0
        else:
            fails = 0
        n = sum(done.values())
        if n % 100 == 0:
            el = time.time() - t0
            print(f"  {n:>6}/{len(todo):,}  ok {done['ok']:>6}  "
                  f"empty {done['empty']:>5}  err {done['error']:>4}  "
                  f"{el/n:.2f}s/req  {el/60:.1f}m")

    el = time.time() - t0
    n = max(sum(done.values()), 1)
    print(f"\n  ok {done['ok']:,}  empty {done['empty']:,}  "
          f"error {done['error']:,}  in {el/60:.1f} minutes "
          f"({el/n:.2f}s per request)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
