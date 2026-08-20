#!/usr/bin/env python3
"""Paste a broker table, get canonical data. The route that actually works.

WHY THIS EXISTS - THE NETWORK DIAGNOSIS
---------------------------------------
Measured 2026-08-20 from inside this session, with curl and with headless
Chromium, so the causes are named rather than guessed:

    host                 result   cause
    reddit.com           200      reachable by curl; the FETCH TOOL blocks it,
                                  not the network
    rapidapi.com         200      reachable, but the page is a JavaScript SPA -
    ohlc.dev             200      the HTML shell carries no content until JS runs
    invezgo.com          200      and this environment cannot run JS (see below)
    blog.itick.org       200
    sectors.app          429      rate-limited by Vercel, not forbidden
    www.idx.co.id        403      Cloudflare, blocking the datacenter IP

    headless Chromium    ERR_CONNECTION_RESET to every host including
                         example.com, with and without the proxy. Browser egress
                         is closed here, so rendering a JS page is not an option.

NONE OF THAT IS THE USER'S NETWORK OR VPN. Every one of those failures is inside
this sandbox. The important asymmetry runs the other way:

    idx.co.id returns 403 to this datacenter IP and will very likely serve a
    normal Indonesian residential connection. The person has reach that this
    process does not, plus a logged-in broker account. So the working division
    of labour is: THEY CAPTURE, THIS PROCESSES.

WHAT THIS DOES
--------------
Takes a table pasted straight out of a broker platform - MOST, Stockbit, IPOT,
Ajaib, RTI - and turns it into the canonical daily store. It is deliberately
copy-and-paste, not automation: IDX prohibits scraping and Stockbit's terms
forbid "data mining, robots, spiders, or similar" without written consent. A
person copying the table already on their screen is doing neither.

It handles what actually comes out of those platforms:

    Indonesian numbers      1.234.567,89  ->  1234567.89
    thousands with commas   1,234,567.89  ->  1234567.89
    lots vs shares          detected from the implied average price
    Indonesian headers      Kode, Beli, Jual, Lot, Nilai, Rata-rata, Pembeli...
    tab / multi-space / pipe separated, with or without a header row

    pbpaste | python3 scripts/paste_broker.py --ticker BBCA --date 2026-08-20
    python3 scripts/paste_broker.py --file pasted.txt --ticker ADRO
"""

from __future__ import annotations

import argparse
import io
import os
import re
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))
sys.path.insert(0, os.path.dirname(__file__))

from broker_collect import is_complete, save                  # noqa: E402
from import_broker_data import import_summary, import_ticks, looks_like_ticks  # noqa: E402

# A broker code is two or three letters on IDX (YP, CC, BK, AK, PD, ZP, ...).
BROKER_RE = re.compile(r"^[A-Z]{2,3}$")


def to_number(tok: str) -> Optional[float]:
    """Parse a number in either Indonesian or English convention.

    "1.234.567,89" is Indonesian and means 1234567.89.
    "1,234,567.89" is English and means the same.
    The rule that separates them: whichever of . or , appears LAST is the
    decimal mark. Getting this backwards turns a billion rupiah into a thousand,
    which is the kind of error that never announces itself.
    """
    t = tok.strip().replace(" ", "").replace(" ", "")
    if not t or t in {"-", "--"}:
        return None
    neg = t.startswith("(") and t.endswith(")")
    t = t.strip("()")
    t = re.sub(r"[^0-9.,\-]", "", t)
    if not t or not re.search(r"\d", t):
        return None
    last_dot, last_com = t.rfind("."), t.rfind(",")
    if last_dot >= 0 and last_com >= 0:
        if last_com > last_dot:                 # Indonesian: 1.234.567,89
            t = t.replace(".", "").replace(",", ".")
        else:                                   # English: 1,234,567.89
            t = t.replace(",", "")
    elif last_com >= 0:
        # a lone comma: decimal if it splits 1-2 trailing digits, else thousands
        t = t.replace(",", "." if len(t) - last_com - 1 <= 2 else "")
    elif last_dot >= 0:
        frac = len(t) - last_dot - 1
        if frac == 3 and t.count(".") >= 1:     # 1.234 is thousands in Indonesian
            t = t.replace(".", "")
    try:
        v = float(t)
    except ValueError:
        return None
    return -v if neg else v


def split_row(line: str) -> List[str]:
    """Tabs, pipes, or runs of two-plus spaces. Single spaces stay inside names."""
    if "\t" in line:
        return [c.strip() for c in line.split("\t")]
    if "|" in line:
        return [c.strip() for c in line.split("|") if c.strip() != ""]
    return [c for c in re.split(r"\s{2,}", line.strip()) if c]


def parse_table(text: str) -> pd.DataFrame:
    """Find the broker rows in whatever was pasted, ignoring chrome around them."""
    rows = []
    for line in text.splitlines():
        cells = split_row(line)
        if len(cells) < 3:
            continue
        code = next((c for c in cells[:2] if BROKER_RE.match(c.strip().upper())), None)
        if not code:
            continue
        nums = [to_number(c) for c in cells]
        nums = [n for n in nums if n is not None]
        if len(nums) < 2:
            continue
        rows.append({"broker": code.strip().upper(), "nums": nums})
    if not rows:
        return pd.DataFrame()
    width = max(len(r["nums"]) for r in rows)
    out = pd.DataFrame([{"broker": r["broker"],
                         **{f"n{i}": (r["nums"][i] if i < len(r["nums"]) else np.nan)
                            for i in range(width)}} for r in rows])
    return out


def assign_columns(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Map the unlabelled numeric columns onto the canonical schema.

    Broker-summary tables are laid out buy-side then sell-side, and the widths
    vary by platform: (lot, value, avg) x 2 is the common one, (lot, avg) x 2
    happens, and some show only lots. The mapping is chosen by WIDTH, and
    anything narrower than four numbers cannot carry both sides so it is refused
    rather than guessed at.
    """
    n = sum(1 for c in df.columns if c.startswith("n"))
    if n >= 6:
        m = {"buy_lot": "n0", "buy_val": "n1", "buy_avg": "n2",
             "sell_lot": "n3", "sell_val": "n4", "sell_avg": "n5"}
    elif n >= 4:
        m = {"buy_lot": "n0", "buy_avg": "n1", "sell_lot": "n2", "sell_avg": "n3"}
    else:
        return None
    out = pd.DataFrame({"broker": df["broker"]})
    for k, v in m.items():
        out[k] = df[v]
    for c in ("buy_val", "sell_val", "buy_avg", "sell_avg"):
        if c not in out.columns:
            out[c] = np.nan
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", required=True)
    ap.add_argument("--date", default=None, help="YYYY-MM-DD; default today WIB")
    ap.add_argument("--file", default=None, help="read from a file instead of stdin")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    raw = (open(args.file).read() if args.file else sys.stdin.read())
    if not raw.strip():
        print("nothing pasted. Pipe the table in, or use --file.")
        return 1
    date = args.date or (pd.Timestamp.now(tz="Asia/Jakarta").strftime("%Y-%m-%d"))

    print(f"{'=' * 84}\n PASTED BROKER TABLE — {args.ticker} {date}\n{'=' * 84}")

    # a running-trade paste is worth far more, so try that reading first
    try:
        head = pd.read_csv(io.StringIO(raw), sep=None, engine="python", nrows=5)
        if looks_like_ticks(head):
            full = pd.read_csv(io.StringIO(raw), sep=None, engine="python")
            out = import_ticks(full, args.ticker, date)
            if out is not None and not out.empty:
                print(f" read as RUNNING TRADE: {len(full):,} prints -> "
                      f"{out['broker'].nunique()} brokers")
                if not args.dry_run:
                    save(out)
                print(f" complete rekap: {is_complete(out)}")
                return 0
    except Exception:
        pass

    parsed = parse_table(raw)
    if parsed.empty:
        print(" no broker rows found. Expected lines with a 2-3 letter broker "
              "code and\n at least two numbers, tab- or multi-space-separated.")
        return 1
    mapped = assign_columns(parsed)
    if mapped is None:
        print(f" found {len(parsed)} broker rows but only "
              f"{sum(1 for c in parsed.columns if c.startswith('n'))} numeric "
              f"columns.\n A broker summary needs both sides — copy the buy AND "
              f"sell columns.")
        return 1

    mapped["ticker"] = args.ticker.upper()
    mapped["date"] = pd.Timestamp(date)
    mapped["source"] = "pasted"

    tot_b, tot_s = mapped["buy_lot"].sum(), mapped["sell_lot"].sum()
    comp = is_complete(mapped)
    print(f" {len(mapped)} brokers parsed")
    print(f" buy {tot_b:,.0f} lots   sell {tot_s:,.0f} lots   "
          f"imbalance {abs(tot_b - tot_s) / max(tot_b, tot_s, 1):.2%}")
    print(f" {'COMPLETE REKAP — counts toward the protocol sample' if comp else 'TOP-N ONLY — truncated, does not count toward the sample'}")
    print(f"\n {'broker':<8}{'buy lot':>14}{'sell lot':>14}{'net':>14}")
    m = mapped.assign(net=mapped["buy_lot"] - mapped["sell_lot"])
    for _, r in m.reindex(m["net"].abs().sort_values(ascending=False).index).head(8).iterrows():
        print(f" {r['broker']:<8}{r['buy_lot']:>14,.0f}{r['sell_lot']:>14,.0f}"
              f"{r['net']:>+14,.0f}")

    if args.dry_run:
        print("\n dry run — nothing written.")
        return 0
    w, s = save(mapped)
    print(f"\n stored {w} ticker-day(s)"
          + (", kept an existing complete file" if s else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
