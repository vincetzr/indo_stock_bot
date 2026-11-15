#!/usr/bin/env python3
"""Collect KSEI's monthly holding-composition archive and build the panel.

    python3 scripts/ksei_collect.py              # fetch what is missing, rebuild
    python3 scripts/ksei_collect.py --offline    # rebuild from the cached zips only
    python3 scripts/ksei_collect.py --start 2024-01 --end 2024-12

What it writes
--------------
``data/cache/ksei/raw/*.zip``      the files exactly as served, never rewritten
``data/cache/ksei/ownership.parquet``  EQUITY rows of every month, tidy
``data/reference/ksei_manifest.csv``   one row per file: state date, creation
                                       stamp, bytes, sha256, row counts

The first two are gitignored because this repository is public and they are
KSEI's data. The manifest is tracked: it carries no holdings, only what is
needed to prove a later run used the same files — which matters more than
usual here, because every file before November 2023 is served but no longer
linked from any page, and can vanish without notice.

Cost: one request per month when the calendar is right, a few more when a
month ends on a holiday. ~106 months at a 2-second courtesy delay is about
five minutes cold and no network at all warm.
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

from idxbot.config import load_config                           # noqa: E402
from idxbot.data import ksei                                    # noqa: E402

PANEL = os.path.join("data", "spine", "price_panel.parquet")


def calendar() -> list:
    """IDX sessions from the spine, if it is on disk; else weekdays only."""
    if not os.path.exists(PANEL):
        return []
    d = pd.read_parquet(PANEL, columns=["date"])["date"].drop_duplicates()
    return [x.date() for x in pd.to_datetime(d)]


def last_complete_month(today: dt.date) -> tuple:
    first = today.replace(day=1)
    prev = first - dt.timedelta(days=1)
    return prev.year, prev.month


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="%d-%02d" % ksei.FIRST_MONTH)
    ap.add_argument("--end", default=None)
    ap.add_argument("--offline", action="store_true")
    a = ap.parse_args()

    cfg = load_config()
    hosts = ksei.allowed_hosts(cfg)
    start = tuple(int(x) for x in a.start.split("-"))
    end = (tuple(int(x) for x in a.end.split("-")) if a.end
           else last_complete_month(dt.date.today()))

    print("=" * 78)
    print(" KSEI holding composition — collect and build")
    print("=" * 78)
    if not a.offline:
        if ksei.HOST not in {h.lower() for h in hosts}:
            print(f" {ksei.HOST} is not in data.ksei_allowed_hosts; nothing "
                  f"fetched. Rebuilding from cache only.")
        else:
            cal = calendar()
            print(f" calendar: {len(cal):,} spine sessions"
                  if cal else " calendar: none on disk, weekdays only")
            got = missing = 0
            for (y, m) in ksei.months(start, end):
                try:
                    r = ksei.fetch_month(y, m, cal or None, hosts=hosts)
                except ConnectionError as e:
                    print(f"   {y}-{m:02d}  NO ANSWER — {e}")
                    continue
                if r is None:
                    missing += 1
                    print(f"   {y}-{m:02d}  absent (every candidate date "
                          f"answered non-200)")
                    continue
                day, f = r
                got += 1
                if not f.from_cache:
                    print(f"   {y}-{m:02d}  {f.name}  {len(f.blob):,} bytes",
                          flush=True)
            print(f" {got} months on disk, {missing} absent\n")

    P, man = ksei.build_panel()
    if P.empty:
        print(" no files cached — nothing to build")
        return 1
    os.makedirs(os.path.dirname(ksei.PANEL_PATH), exist_ok=True)
    P.to_parquet(ksei.PANEL_PATH, index=False)
    os.makedirs(os.path.dirname(ksei.MANIFEST_PATH), exist_ok=True)
    man.to_csv(ksei.MANIFEST_PATH, index=False)

    lag = (man["created"].dt.normalize() - man["month_end"]).dt.days
    print(f" {len(man)} files, {man.month_end.min().date()} … "
          f"{man.month_end.max().date()}")
    print(f" {len(P):,} EQUITY rows, {P.ticker.nunique():,} distinct codes, "
          f"{P.groupby('month_end').size().median():.0f} a month at the median")
    print(" both identities held on every row of every file (parse refuses "
          "otherwise)")
    print(f" creation stamp minus state date, calendar days: "
          f"median {lag.median():.0f}, max {lag.max():.0f}, "
          f"min {lag.min():.0f}")
    gaps = sorted(set(pd.period_range(man.month_end.min(), man.month_end.max(),
                                      freq="M"))
                  - set(man.month_end.dt.to_period("M")))
    print(f" months missing inside the span: "
          f"{', '.join(str(g) for g in gaps) if gaps else 'none'}")
    cov = (P["local_total"] + P["foreign_total"]) / P["sec_num"].replace(0, np.nan)
    print(f" coverage (held / sec_num): median {cov.median():.3f}, "
          f"share of rows below 0.9 {float((cov < 0.9).mean()):.1%}")
    print(f"\n wrote {ksei.PANEL_PATH}  (gitignored — KSEI's data)")
    print(f" wrote {ksei.MANIFEST_PATH}  (tracked — hashes and stamps only)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
