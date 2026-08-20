#!/usr/bin/env python3
"""Drop a file in, get broker data out. The only sanctioned way to feed this repo.

WHY THIS EXISTS INSTEAD OF A SCRAPER
------------------------------------
Research on 2026-08-19 established three things about IDX broker data:

  1. The "top 10 per side" limit is a property of the free IndoPremier PAGE, not
     of the data. Measured against that page for BBCA on 2026-08-18: it rendered
     10 rows per side summing to 943,424 buy lots, while the page's own footer
     reported T.Lot 1.1M for the same stock, board and day. Roughly 12-14% of
     regular-board volume sat in members the page never printed.
  2. The complete rekap therefore exists upstream, and institutions read it
     directly - LSEG/Refinitiv publishes one RIC per broker per instrument
     (<OD-BBCA.JK> = Danareksa in BBCA, buy volume on FID 731, sell on FID 736),
     with no top-N anywhere in that structure.
  3. No Indonesian retail platform publishes a documented API for this, and
     Stockbit's terms explicitly forbid "data mining, robots, spiders, or similar"
     without written consent. IDX prohibits scraping outright.

So there is no legitimate automated route from a retail account, and this repo
will not build an illegitimate one. What IS legitimate is a person exporting or
saving what their own platform shows them, and a program reading that file. That
is what this script does.

BROKER SUMMARY IS AN AGGREGATION, NOT A PRODUCT
-----------------------------------------------
Every print on IDX carries a buyer member code and a seller member code. Sum the
prints by code and you have reconstructed the complete broker summary - all ~90
members, not ten - for any window you like, including intraday.
``src/idxbot/data/running_trade.py`` already does this. So a running-trade export
is strictly more valuable than a broker-summary export, and this importer prefers
it when both are available:

    running trade  ->  full rekap at any resolution        BEST
    broker summary ->  whatever rows the platform printed  useful, truncated

WHAT TO DROP IN
---------------
Put files in ``data/inbox/``. Any CSV/TSV/JSON/JSONL/XLSX. Column names are
matched against the alias tables in ``broker_summary.py`` and
``running_trade.py``, which already cover English and Indonesian headers from
Stockbit, RTI, IPOT, Mirae HOTS, MOST and IDX exports (kode, harga, lot, pembeli,
penjual, tanggal, volume beli, ...). Filenames are used only as a fallback hint
for ticker and date.

Nothing is deleted from the inbox. Everything imported is written to the
permanent cache and is never re-fetched.

    python3 scripts/import_broker_data.py
    python3 scripts/import_broker_data.py --inbox data/inbox --report
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

from idxbot.config import load_config                    # noqa: E402
from idxbot.data.broker_summary import (COLUMN_ALIASES,  # noqa: E402
                                        SCHEMA, _resolve_columns, normalise)
from idxbot.data.running_trade import (TICK_COLUMNS,     # noqa: E402
                                       RunningTradeAggregator, parse_tick)

INBOX = os.path.join("data", "inbox")
STORE = os.path.join("data", "cache", "imported")
# NOT \b([A-Z]{4})\b - underscore is a word character, so \b never fires between
# "BBCA" and "_" and every BBCA_20260819_*.csv silently lost its ticker.
TICKER_RE = re.compile(r"(?<![A-Z])([A-Z]{4})(?![A-Z])")
DATE_RE = re.compile(r"(20\d{2})[-_]?(\d{2})[-_]?(\d{2})")


def read_any(path: str) -> Optional[pd.DataFrame]:
    """Read whatever the platform exported, without guessing too hard."""
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext in (".jsonl", ".ndjson"):
            rows = [json.loads(x) for x in open(path) if x.strip()]
            return pd.DataFrame(rows)
        if ext == ".json":
            obj = json.load(open(path))
            if isinstance(obj, dict):
                for key in ("data", "results", "rows", "items"):
                    if isinstance(obj.get(key), list):
                        obj = obj[key]
                        break
            return pd.DataFrame(obj if isinstance(obj, list) else [obj])
        if ext in (".xlsx", ".xls"):
            return pd.read_excel(path)
        if ext in (".html", ".htm", ".xhtml"):
            # "Save page as" or a copied rendered table. Take the widest table on
            # the page: broker summaries are the biggest grid on those screens,
            # and the small ones are navigation and summary boxes.
            tables = pd.read_html(path, converters={i: str for i in range(64)})
            return max(tables, key=lambda t: t.shape[0] * t.shape[1]) if tables else None
        for sep in (None, ",", ";", "\t", "|"):
            try:
                d = pd.read_csv(path, sep=sep, engine="python")
                if d.shape[1] > 1:
                    return d
            except Exception:
                continue
    except Exception as exc:
        print(f"   ! {os.path.basename(path)}: cannot read ({type(exc).__name__}: {exc})")
    return None


def hint_from_name(path: str) -> Tuple[Optional[str], Optional[str]]:
    """Ticker and date guessed from the filename - a fallback, never an override."""
    base = os.path.basename(path).upper()
    t = TICKER_RE.search(base)
    m = DATE_RE.search(base)
    return (t.group(1) if t else None,
            f"{m.group(1)}-{m.group(2)}-{m.group(3)}" if m else None)


def looks_like_ticks(df: pd.DataFrame) -> bool:
    """Ticks carry a buyer AND a seller on the same row; a summary carries one broker."""
    cols = {re.sub(r"[^a-z0-9]", "", c.lower()) for c in df.columns}
    def has(field: str) -> bool:
        from idxbot.data.running_trade import TICK_ALIASES
        return any(re.sub(r"[^a-z0-9]", "", a) in cols for a in TICK_ALIASES[field])
    return has("buyer") and has("seller")


def import_ticks(df: pd.DataFrame, ticker: Optional[str],
                 date: Optional[str] = None) -> Optional[pd.DataFrame]:
    """Turn a running-trade export into the COMPLETE broker summary.

    ``date`` matters more than it looks. A running-trade export usually carries
    only a TIME per print ("09:00:12"), which pandas dates to today - so an
    export of last Friday's tape would be stored under this morning's date and
    then joined to the wrong price bar. When the filename carries a date it wins.
    """
    agg = RunningTradeAggregator()
    n = agg.ingest(df.to_dict("records"))
    if not n:
        return None
    out = agg.snapshot(source="imported_ticks")
    if out.empty:
        return None
    if ticker and "ticker" in out.columns:
        out["ticker"] = out["ticker"].fillna(ticker)
    if date:
        out["date"] = pd.Timestamp(date)
    return out


def import_summary(df: pd.DataFrame, ticker: Optional[str],
                   date: Optional[str]) -> Optional[pd.DataFrame]:
    """Normalise a broker-summary table into the repo's schema."""
    # normalise() takes no date argument and REQUIRES a date column, but most
    # exports carry the date once in a page header or only in the filename. So
    # it is injected here, before the call, rather than repaired afterwards.
    have_date = "date" in _resolve_columns(df.columns)
    if not have_date:
        if not date:
            print("   ! no date column and none in the filename — skipping, "
                  "because an undated\n     broker row cannot be joined to a "
                  "price bar. Rename it TICKER_YYYYMMDD_*.csv")
            return None
        df = df.assign(date=pd.Timestamp(date))
    try:
        out = normalise(df, ticker=ticker, source="imported_summary")
    except Exception as exc:
        print(f"   ! could not normalise: {type(exc).__name__}: {exc}")
        return None
    return None if out is None or out.empty else out


def completeness(out: pd.DataFrame) -> Dict[str, float]:
    """How much of the tape this file actually accounts for."""
    if out.empty:
        return {}
    buy = float(out["buy_lot"].sum())
    sell = float(out["sell_lot"].sum())
    # a complete rekap has buy == sell exactly; the gap measures truncation
    gap = abs(buy - sell) / max(buy, sell, 1.0)
    return {"brokers": int(out["broker"].nunique()),
            "buy_lot": buy, "sell_lot": sell, "imbalance": gap}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inbox", default=INBOX)
    ap.add_argument("--store", default=STORE)
    ap.add_argument("--report", action="store_true",
                    help="also summarise what is already in the store")
    args = ap.parse_args()
    os.makedirs(args.inbox, exist_ok=True)
    os.makedirs(args.store, exist_ok=True)

    files = sorted(f for f in glob.glob(os.path.join(args.inbox, "*"))
                   if os.path.isfile(f) and not f.endswith(".md"))
    print(f"{'=' * 88}\n BROKER DATA IMPORT — inbox {args.inbox}\n{'=' * 88}")
    if not files:
        print(" nothing to import.\n")
        print(" Drop any CSV/TSV/JSON/JSONL/XLSX your platform gives you into "
              "this folder.\n Headers are matched against English and Indonesian "
              "aliases already known\n to the repo (kode, harga, lot, pembeli, "
              "penjual, tanggal, volume beli, ...).\n")
        print(" A RUNNING TRADE export is worth far more than a broker summary:")
        print("   running trade -> every print carries a buyer and a seller code,")
        print("                    so the FULL rekap for all members is "
              "reconstructable,")
        print("                    at any resolution including intraday.")
        print("   broker summary -> only the rows your platform chose to print, "
              "end of day.")
        print("\n Nothing here scrapes anything. Export manually; this reads the "
              "file.")
        return 0

    kept: List[pd.DataFrame] = []
    for path in files:
        name = os.path.basename(path)
        df = read_any(path)
        if df is None or df.empty:
            print(f" - {name:<44} unreadable or empty")
            continue
        tk, dt = hint_from_name(path)
        ticks = looks_like_ticks(df)
        out = (import_ticks(df, tk, dt) if ticks else import_summary(df, tk, dt))
        if out is None or out.empty:
            print(f" - {name:<44} {len(df):>6,} rows -> nothing usable "
                  f"({'ticks' if ticks else 'summary'})")
            continue
        c = completeness(out)
        kept.append(out)
        flag = "FULL REKAP" if ticks else "top-N only"
        print(f" + {name:<44} {len(df):>6,} rows -> {c['brokers']:>3} brokers "
              f"[{flag}]")
        if not ticks and c.get("imbalance", 0) > 0.02:
            print(f"     buy/sell lots differ by {c['imbalance']:.1%} — a "
                  f"complete rekap must balance exactly,\n     so this table is "
                  f"truncated. A running-trade export would not be.")

    if not kept:
        print("\n nothing imported.")
        return 1

    allrows = pd.concat(kept, ignore_index=True)
    for (tk, dt), g in allrows.groupby(["ticker", "date"]):
        dest = os.path.join(args.store, f"{tk}_{pd.Timestamp(dt):%Y%m%d}.csv.gz")
        g.to_csv(dest, index=False, compression="gzip")
    print(f"\n imported {len(allrows):,} broker-rows covering "
          f"{allrows['ticker'].nunique()} tickers and "
          f"{allrows['date'].nunique()} dates -> {args.store}/")

    if args.report:
        have = sorted(glob.glob(os.path.join(args.store, "*.csv.gz")))
        print(f"\n{'=' * 88}\n STORE\n{'=' * 88}")
        print(f" {len(have)} files")
        if have:
            idx = pd.DataFrame([os.path.basename(h).replace(".csv.gz", "").split("_")
                                for h in have], columns=["ticker", "date"])
            for tk, g in idx.groupby("ticker"):
                print(f"   {tk:<8} {len(g):>4} days  "
                      f"{g['date'].min()} .. {g['date'].max()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
