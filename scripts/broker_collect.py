#!/usr/bin/env python3
"""Accumulate broker data from whatever legitimate source exists, forever.

THE PIVOTAL FAILURE, STATED PLAINLY
-----------------------------------
Layer 2 is the only place left where an edge could live. Layer 3 (price) matched
a random walk at every timeframe (110) and lost to a same-exposure null
everywhere (111, 113, 116). Layer 1 has never been testable here for want of a
point-in-time news set. That leaves broker flow - and the reason it has never
been settled is not that it was tested and failed, it is that THERE HAS NEVER
BEEN ENOUGH DATA.

What exists today: 1,703 range-aggregate files (one row per window, not per day),
92 true single-day files covering 5 tickers, and a daily net-flow series
buildable for exactly one name. The two tests that were run returned d = 0.002
and d = -0.005 - but both were on pullback-event windows, an outcome-conditioned
sample, not on a clean daily panel.

So the job is not to test harder. It is to COLLECT, every day, from today, until
there is enough - and to know in advance how much "enough" is.

WHAT THIS DOES
--------------
Normalises anything into one canonical daily store and never loses a row:

    data/cache/broker_daily/TICKER_YYYYMMDD.csv.gz
    date, ticker, broker, buy_lot, buy_val, buy_avg, sell_lot, sell_val,
    sell_avg, source, complete

``complete`` is the field that matters. A full rekap BALANCES - total buy lots
equal total sell lots, because every share bought was sold. A top-N table does
not. That single check separates data that can answer the question from data that
cannot, and it is recorded per row rather than assumed.

SOURCES, IN ORDER OF WHAT THEY ARE WORTH
----------------------------------------
    running trade   every print carries a buyer AND a seller code, so the FULL
                    rekap for all ~90 members is reconstructable at any
                    resolution. This is the one worth having.
    broker summary  whatever rows the platform chose to print, end of day.
    nothing         the honest state most days, and the script says so.

ON THIRD-PARTY APIs
-------------------
Checked 2026-08-20 against the providers circulating on r/JudiSaham: none
verifiably serves a buyer+seller broker code per trade, none appears on IDX's
authorised redistributor list, and at least one popular repo is an open scraper
of idx.co.id. This module therefore refuses any host not explicitly allowlisted
in config under ``data.broker_allowed_hosts``, and ships with that list EMPTY.
Add a host only if you have checked its licensing yourself.

    python3 scripts/broker_collect.py --report
    python3 scripts/broker_collect.py --ingest data/inbox
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

from idxbot.config import load_config                     # noqa: E402
from idxbot.data.broker_summary import SCHEMA             # noqa: E402

STORE = os.path.join("data", "cache", "broker_daily")
BALANCE_TOL = 1e-6


def is_complete(g: pd.DataFrame) -> bool:
    """A full rekap balances exactly; a truncated top-N table does not."""
    b, s = float(g["buy_lot"].sum()), float(g["sell_lot"].sum())
    if max(b, s) <= 0:
        return False
    return abs(b - s) / max(b, s) < BALANCE_TOL


def store_path(ticker: str, date: pd.Timestamp) -> str:
    return os.path.join(STORE, f"{ticker}_{date:%Y%m%d}.csv.gz")


def save(df: pd.DataFrame) -> Tuple[int, int]:
    """Write one row-group per (ticker, date). Never overwrites a complete file
    with a truncated one - a full rekap outranks a top-N table for the same day."""
    os.makedirs(STORE, exist_ok=True)
    written = skipped = 0
    for (tk, dt), g in df.groupby(["ticker", "date"]):
        path = store_path(str(tk), pd.Timestamp(dt))
        g = g.copy()
        g["complete"] = is_complete(g)
        if os.path.exists(path):
            try:
                old = pd.read_csv(path)
                if bool(old.get("complete", pd.Series([False])).iloc[0]) and \
                        not bool(g["complete"].iloc[0]):
                    skipped += 1
                    continue
            except Exception:
                pass
        g.to_csv(path, index=False, compression="gzip")
        written += 1
    return written, skipped


def load_store() -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(STORE, "*.csv.gz")))
    if not files:
        return pd.DataFrame(columns=SCHEMA + ["complete"])
    out = []
    for f in files:
        try:
            out.append(pd.read_csv(f))
        except Exception:
            continue
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(
        columns=SCHEMA + ["complete"])


def coverage(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    g = df.groupby("ticker").agg(
        days=("date", "nunique"),
        first=("date", "min"),
        last=("date", "max"),
        brokers=("broker", "nunique"))
    comp = df.groupby(["ticker", "date"])["complete"].first().groupby("ticker")
    g["complete_days"] = comp.sum().astype(int)
    return g.sort_values("days", ascending=False)


def allowed_hosts(cfg) -> List[str]:
    """Explicit allowlist, empty by default. Nothing is fetched without one."""
    try:
        v = cfg.get("data.broker_allowed_hosts", [])
    except Exception:
        v = []
    return list(v) if isinstance(v, (list, tuple)) else []


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ingest", default=None,
                    help="folder of exports to fold into the daily store")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--need-days", type=int, default=250,
                    help="days per name the protocol needs before it will run")
    args = ap.parse_args()
    cfg = load_config()
    os.makedirs(STORE, exist_ok=True)

    print(f"{'=' * 92}\n BROKER DATA COLLECTION — the pivotal gap\n{'=' * 92}")

    if args.ingest:
        sys.path.insert(0, os.path.dirname(__file__))
        from import_broker_data import (hint_from_name, import_summary,
                                        import_ticks, looks_like_ticks, read_any)
        files = [f for f in sorted(glob.glob(os.path.join(args.ingest, "*")))
                 if os.path.isfile(f) and not f.endswith(".md")]
        got = []
        for path in files:
            d = read_any(path)
            if d is None or d.empty:
                continue
            tk, dt = hint_from_name(path)
            ticks = looks_like_ticks(d)
            out = import_ticks(d, tk, dt) if ticks else import_summary(d, tk, dt)
            if out is not None and not out.empty:
                got.append(out)
                print(f" + {os.path.basename(path):<40} "
                      f"{'FULL REKAP' if ticks else 'top-N only'}")
        if got:
            w, s = save(pd.concat(got, ignore_index=True))
            print(f"\n stored {w} ticker-days"
                  + (f", kept {s} existing complete files over truncated ones"
                     if s else ""))
        else:
            print(" nothing ingestible found.")

    hosts = allowed_hosts(cfg)
    df = load_store()
    C = coverage(df)

    print(f"\n{'=' * 92}\n WHAT IS IN THE DAILY STORE\n{'=' * 92}")
    if C.empty:
        print(" empty. Nothing has been collected yet.\n")
    else:
        print(f" {len(C)} tickers, {int(C['days'].sum())} ticker-days, "
              f"{int(C['complete_days'].sum())} of them a complete rekap\n")
        print(f" {'ticker':<8}{'days':>7}{'complete':>10}{'brokers':>9}"
              f"{'first':>12}{'last':>12}")
        for t, r in C.head(15).iterrows():
            print(f" {t:<8}{int(r['days']):>7}{int(r['complete_days']):>10}"
                  f"{int(r['brokers']):>9}{str(r['first']):>12}"
                  f"{str(r['last']):>12}")

    print(f"\n{'=' * 92}\n HOW FAR OFF IS AN ANSWER?\n{'=' * 92}")
    need = args.need_days
    have = int(C["complete_days"].max()) if not C.empty else 0
    names_ready = int((C["complete_days"] >= need).sum()) if not C.empty else 0
    print(f" the protocol needs {need} complete days on a name before it will "
          f"report anything.")
    print(f" best-covered name has {have}; names at or past the bar: {names_ready}")
    if have < need:
        left = need - have
        print(f" at one session a day that is about {left} trading days "
              f"(~{left / 21:.0f} months) of collection.")
    print(f"\n fetching is restricted to hosts in config "
          f"data.broker_allowed_hosts: "
          f"{hosts if hosts else 'EMPTY — nothing will be fetched'}")
    print(" Checked 2026-08-20: no third-party IDX API verifiably serves a "
          "buyer+seller\n broker code per trade, none is on IDX's authorised "
          "redistributor list, and a\n popular open repo is a scraper of "
          "idx.co.id. Add a host only if YOU have\n verified its licensing.")

    print(f"\n{'=' * 92}\n WHAT TO FEED IT\n{'=' * 92}")
    print(" Best  : a RUNNING TRADE export (every print carries a buyer and a")
    print("         seller code) — rebuilds the complete rekap for all members.")
    print(" Useful: a broker summary export — top-N only, and flagged as such.")
    print(" Drop either into data/inbox/ and this folds it in permanently.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
