#!/usr/bin/env python3
"""Collect the broker-flow panel Phase 1 actually needs, one window at a time.

THE ARITHMETIC THAT MAKES THIS POSSIBLE
----------------------------------------
CLAUDE.md A1 costs the §4 panel at "1 request per ticker-day, ~2,000,000
requests, 27 days of continuous fetching" and concludes Phase 1 has to run on a
10-name panel. That costing is for IndoPremier's DAILY mode. The same endpoint
takes ``start`` and ``end`` and returns the top-10 rekap AGGREGATED over the
whole window for one request - which ``scripts/pullback_flow.fetch_window`` has
been using all along.

At a fortnightly window the panel costs one request per ticker-fortnight, not
per ticker-day. That is roughly a twentieth of A1's figure, and it is the
difference between a 10-name panel and a real cross-section.

WHAT IS GIVEN UP, STATED PLAINLY
---------------------------------
Fortnightly flow, not daily flow. §7's hypothesis is stated on daily net-buy
imbalance and its decay curve runs k ∈ {1,3,5,10,20}; this panel can speak to
k = 10 and 20 and cannot speak to k = 1 or 3. That is a real narrowing of the
hypothesis and every result off this panel has to say so.

It is a narrowing rather than a substitution for two reasons. §6 point 2 warns
that the raw daily net-buy column is public and simultaneous, so whatever edge
exists is likelier in transformations than in the raw daily print - a fortnight
of accumulated imbalance IS such a transformation. And aggregation only runs
one way: two fortnights make a month, but no amount of arithmetic recovers a
day from a fortnight. So the fortnight is the finest unit worth paying for, and
the monthly study is free once this exists.

WINDOW-MAJOR, NEWEST FIRST, AND WHY THAT ORDER
-----------------------------------------------
This will be interrupted - it is hours of work against someone else's server
and the container is not permanent. So the loop is ordered so that whatever it
has finished is USABLE rather than ragged:

    for each fortnight, newest first:
        for each ticker in the panel:
            fetch unless the cache already has it

Stopping at any point leaves a COMPLETE cross-section for every fortnight
collected, which is exactly the shape a cross-sectional IC test consumes. The
ticker-major alternative leaves a full history for some names and nothing for
others, which is not a panel at all.

Resumption is free and needs no state file: a cached window is a cache hit and
costs no request, so re-running simply continues.

POLITENESS IS A CONSTRAINT, NOT A SETTING
------------------------------------------
A5: "IndoPremier is a public page on a licensed member's site: polite delay,
permanent cache, no bulk harvesting, no redistribution." The delay is not a
knob to minimise, the cache is permanent so nothing is ever fetched twice, and
``--budget`` exists so the total volume of a run is a decision someone made on
purpose rather than a consequence of how long a loop was left running.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))
sys.path.insert(0, os.path.dirname(__file__))

from idxbot.data.cache import Cache              # noqa: E402
from pullback_flow import fetch_window           # noqa: E402

PANEL_FILE = os.path.join("config", "flow_panel.yaml")
LIVE = os.path.join("data", "cache", "ohlcv")
DEAD = os.path.join("data", "cache", "delisted")


def load_panel(path: str = PANEL_FILE) -> pd.DataFrame:
    """Read the universe file. Deliberately not a YAML dependency."""
    rows = []
    seen_header = False
    for line in open(path):
        if line.startswith("tickers:"):
            seen_header = True
            continue
        if not seen_header:
            continue
        m = re.match(r"\s+([A-Z]{4}):\s*\[(\d+),\s*(\w+),\s*([\d.]+)\]", line)
        if m:
            rows.append({"ticker": m.group(1), "decile": int(m.group(2)),
                         "src": m.group(3), "entry_turnover": float(m.group(4))})
    return pd.DataFrame(rows)


#: Delisted names IndoPremier does not serve at all. A code that has left the
#: exchange sometimes leaves their master too, and then every window for it
#: returns an empty table - about half of the recovered delisted names, probed
#: once each mid-life rather than discovered 150 times over. Listed here rather
#: than dropped from the universe on purpose: which names the FLOW panel loses
#: is itself the measurement of how survivorship-biased it still is, and a
#: universe file that quietly omitted them would hide that.
NOT_SERVED_FILE = os.path.join("config", "flow_panel_unserved.txt")


def not_served() -> set:
    if not os.path.exists(NOT_SERVED_FILE):
        return set()
    return {ln.strip() for ln in open(NOT_SERVED_FILE)
            if ln.strip() and not ln.startswith("#")}


def listed_span(ticker: str, src: str) -> Optional[Tuple[pd.Timestamp,
                                                         pd.Timestamp]]:
    """First and last bar with a real print. Windows outside cost nothing."""
    fp = os.path.join(DEAD if src == "delisted" else LIVE, f"{ticker}.JK.csv.gz")
    if not os.path.exists(fp):
        return None
    try:
        x = pd.read_csv(fp, usecols=["date", "close", "volume"])
    except Exception:                                           # noqa: BLE001
        return None
    x["date"] = pd.to_datetime(x["date"])
    x = x[(x["close"] > 0) & (x["volume"] > 0)]
    if x.empty:
        return None
    return x["date"].min(), x["date"].max()


def windows(start: str, end: str, step_days: int) -> List[Tuple[pd.Timestamp,
                                                                pd.Timestamp]]:
    """Non-overlapping [start, end] business-day windows, oldest first.

    Non-overlapping matters. Overlapping windows would double-count flow and
    make the forward-return labels overlap twice over - once from the window
    and once from the horizon - and §11 already has to purge and embargo the
    horizon overlap without help.
    """
    bd = pd.bdate_range(start, end)
    out = []
    for i in range(0, len(bd) - 1, step_days):
        chunk = bd[i:i + step_days]
        if len(chunk) >= 2:
            out.append((chunk[0], chunk[-1]))
    return out


def plan(P: pd.DataFrame, wins, spans: Dict[str, Optional[Tuple]]
         ) -> List[Tuple[pd.Timestamp, pd.Timestamp, str]]:
    """Every (window, ticker) pair worth a request, newest window first."""
    skip = not_served()
    jobs = []
    for a, b in reversed(wins):
        for t in P["ticker"]:
            if t in skip:
                continue           # code gone from the source's master
            sp = spans.get(t)
            if sp is None or b < sp[0] or a > sp[1]:
                continue           # not listed then - never spend a request
            jobs.append((a, b, t))
    return jobs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2014-01-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--step", type=int, default=10,
                    help="business days per window; 10 = fortnightly")
    ap.add_argument("--budget", type=int, default=5000,
                    help="hard cap on NETWORK requests for this run")
    ap.add_argument("--delay", type=float, default=1.3)
    ap.add_argument("--latency", type=float, default=3.2,
                    help="MEASURED seconds per request, delay + network. The "
                         "delay alone understates the run by 2.5x and the "
                         "first version of this script quoted 11.5h for a job "
                         "that takes 28.")
    ap.add_argument("--board", default="RG")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    end = a.end or (pd.Timestamp.today().normalize()
                    - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    P = load_panel()
    if P.empty:
        print("no panel; run scripts/flow_panel_universe.py first")
        return 1
    wins = windows(a.start, end, a.step)
    spans = {t: listed_span(t, s) for t, s in zip(P["ticker"], P["src"])}
    jobs = plan(P, wins, spans)

    cache = Cache(os.path.join("data", "cache"))
    todo = []
    for s, e, t in jobs:
        key = f"{t}_{s:%Y%m%d}_{e:%Y%m%d}_{a.board}_range"
        if cache.read("ipot_broker", key) is None:
            todo.append((s, e, t))

    print(f"panel      {len(P)} names ({int((P.src=='delisted').sum())} delisted)")
    print(f"windows    {len(wins)} x {a.step} business days, "
          f"{a.start} .. {end}")
    print(f"pairs      {len(jobs):,} listed (ticker, window) pairs")
    print(f"cached     {len(jobs) - len(todo):,}")
    print(f"to fetch   {len(todo):,}  ~{len(todo) * a.latency / 3600:.1f} h "
          f"at a measured {a.latency}s/request")
    print(f"budget     {a.budget:,} requests this run "
          f"(~{min(len(todo), a.budget) * a.latency / 3600:.1f} h)\n")
    if a.dry_run:
        return 0

    done = ok = empty = fail = 0
    t0 = time.time()
    last_win = None
    for s, e, t in todo:
        if done >= a.budget:
            print(f"\nbudget of {a.budget:,} requests reached — stopping.")
            break
        if (s, e) != last_win:
            last_win = (s, e)
            el = time.time() - t0
            print(f"[{done:>6,}/{min(len(todo), a.budget):,}  {el/3600:>4.1f}h] "
                  f"window {s:%Y-%m-%d}..{e:%Y-%m-%d}", flush=True)
        try:
            df = fetch_window(cache, t, s, e, delay=a.delay, board=a.board)
        except Exception as exc:                                # noqa: BLE001
            print(f"   ! {t}: {type(exc).__name__} {exc}", flush=True)
            fail += 1
            done += 1
            continue
        done += 1
        if df is None:
            fail += 1
        elif df.empty:
            empty += 1
        else:
            ok += 1

    el = time.time() - t0
    print(f"\n{done:,} requests in {el/3600:.2f} h — "
          f"{ok:,} with data, {empty:,} empty, {fail:,} failed")
    print(f"remaining after this run: {max(0, len(todo) - done):,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
