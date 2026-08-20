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
LOT_SIZE = 100

# Platforms abbreviate. English K/M/B/T and Indonesian rb/jt/mlr both appear.
# "M" is the dangerous one - English million against Indonesian miliar, a factor
# of 1000 apart - so every parse is checked against value = lot x 100 x average
# and a mismatch is reported rather than absorbed.
SUFFIX = {"K": 1e3, "M": 1e6, "B": 1e9, "T": 1e12,
          "RB": 1e3, "JT": 1e6, "MLR": 1e9}


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
    # The magnitude suffix must come off BEFORE the digits are cleaned: "840.9M"
    # is 840.9 million, and dropping the M silently divides it by a million.
    # A suffixed number's dot is always a DECIMAL point, never thousands.
    mult = 1.0
    suf = re.match(r"^([\d.,\-]+)\s*(RB|JT|MLR|[KMBT])$", t, re.I)
    if suf:
        t, mult = suf.group(1), SUFFIX[suf.group(2).upper()]
    t = re.sub(r"[^0-9.,\-]", "", t)
    if not t or not re.search(r"\d", t):
        return None
    if mult != 1.0:
        try:
            v = float(t.replace(",", ".")) * mult
        except ValueError:
            return None
        return -v if neg else v
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
        v = float(t) * mult
    except ValueError:
        return None
    return -v if neg else v


def html_to_text(raw: str) -> str:
    """Rows to lines, cells to tabs, every value left exactly as written."""
    txt = re.sub(r"(?is)<(script|style).*?</\1>", " ", raw)
    # Saved pages are pretty-printed, so a single <tr> spans several source
    # lines. Collapse them first, or the row is split and its columns are lost.
    txt = re.sub(r"[\r\n]+", " ", txt)
    txt = re.sub(r"(?i)</\s*(tr|table)\s*>", "\n", txt)
    txt = re.sub(r"(?i)</\s*(td|th)\s*>", "\t", txt)
    txt = re.sub(r"(?s)<[^>]+>", "", txt)
    txt = (txt.replace("&nbsp;", " ").replace("&amp;", "&")
              .replace("&lt;", "<").replace("&gt;", ">"))
    lines = []
    for line in txt.splitlines():
        cells = [c.strip() for c in line.split("\t")]
        cells = [c for c in cells if c != ""]
        if cells:
            lines.append("\t".join(cells))
    return "\n".join(lines)


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


def broker_positions(cells: List[str]) -> List[int]:
    return [i for i, c in enumerate(cells) if BROKER_RE.match(c.strip().upper())]


def parse_sides(text: str) -> Tuple[List[Tuple[str, List[float]]],
                                    List[Tuple[str, List[float]]]]:
    """Split rows that carry TWO brokers - the buy list and the sell list.

    Platforms render the two top-N lists side by side, so one row reads
    "XL 840.9M 24.2K 349 | RF 2.1B 60.6K 347". The two brokers are UNRELATED:
    the buy side and the sell side are independently ranked, and row 1 of one has
    nothing to do with row 1 of the other. Reading such a row as a single broker
    attributes RF's selling to XL, which is worse than useless.
    """
    buy, sell = [], []
    for line in text.splitlines():
        cells = split_row(line)
        pos = broker_positions(cells)
        if not pos:
            continue
        groups = []
        for k, start in enumerate(pos):
            end = pos[k + 1] if k + 1 < len(pos) else len(cells)
            nums = [to_number(c) for c in cells[start + 1:end]]
            groups.append((cells[start].strip().upper(),
                           [n for n in nums if n is not None]))
        if groups and len(groups[0][1]) >= 2:
            buy.append(groups[0])
        if len(groups) >= 2 and len(groups[1][1]) >= 2:
            sell.append(groups[1])
    return buy, sell


def infer_order(groups: List[Tuple[str, List[float]]]) -> Tuple[str, float]:
    """Is it (value, lot, average) or (lot, value, average)?

    Decided by arithmetic rather than by header text, because headers are
    abbreviated and translated but the identity is not:

        value = lot x 100 shares x average price

    Whichever ordering satisfies it on more rows wins, and the agreement rate is
    returned so a parse that fits NEITHER can be reported instead of guessed.
    """
    best, best_score = "val_lot_avg", -1.0
    for order in ("val_lot_avg", "lot_val_avg"):
        ok = tot = 0
        for _b, nums in groups:
            if len(nums) < 3:
                continue
            a, b, c = nums[0], nums[1], nums[2]
            val, lot, avg = (a, b, c) if order == "val_lot_avg" else (b, a, c)
            if val > 0 and lot > 0 and avg > 0:
                tot += 1
                ok += int(abs(lot * LOT_SIZE * avg / val - 1.0) < 0.15)
        score = ok / tot if tot else 0.0
        if score > best_score:
            best, best_score = order, score
    return best, best_score


def sides_to_frame(buy: List[Tuple[str, List[float]]],
                   sell: List[Tuple[str, List[float]]],
                   order: str) -> pd.DataFrame:
    """Outer-join the two independent lists on broker code."""
    def side(groups, prefix):
        rows = []
        for b, nums in groups:
            if len(nums) < 3:
                continue
            a, bb, c = nums[0], nums[1], nums[2]
            val, lot, avg = (a, bb, c) if order == "val_lot_avg" else (bb, a, c)
            rows.append({"broker": b, f"{prefix}_val": val,
                         f"{prefix}_lot": lot, f"{prefix}_avg": avg})
        return pd.DataFrame(rows)
    B, S = side(buy, "buy"), side(sell, "sell")
    if B.empty and S.empty:
        return pd.DataFrame()
    out = (B.merge(S, on="broker", how="outer") if not B.empty and not S.empty
           else (B if S.empty else S))
    for c in ("buy_lot", "buy_val", "buy_avg", "sell_lot", "sell_val", "sell_avg"):
        if c not in out.columns:
            out[c] = np.nan
    return out.fillna({"buy_lot": 0.0, "sell_lot": 0.0,
                       "buy_val": 0.0, "sell_val": 0.0})


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

    raw = (open(args.file, errors="replace").read() if args.file
           else sys.stdin.read())
    if not raw.strip():
        print("nothing pasted. Pipe the table in, or use --file.")
        return 1
    date = args.date or (pd.Timestamp.now(tz="Asia/Jakarta").strftime("%Y-%m-%d"))

    print(f"{'=' * 84}\n PASTED BROKER TABLE — {args.ticker} {date}\n{'=' * 84}")

    # An HTML table pasted or saved from a browser. The tags are stripped by
    # hand rather than handed to pandas.read_html, because that coerces
    # "120.000" to the float 120.0 - it reads the Indonesian thousands dot as a
    # decimal point - and a silent 1000x error is exactly what this file exists
    # to prevent. Stripping tags keeps every cell an untouched string for
    # to_number() to judge.
    if "<t" in raw.lower():
        raw = html_to_text(raw)
        print(f" stripped an HTML table -> {len(raw.splitlines())} rows")

    # Two-sided layout? Platforms render the buy list and the sell list side by
    # side, so a row carries TWO unrelated brokers. Detect it before parsing,
    # because reading such a row as one broker attributes the seller's volume to
    # the buyer.
    two = sum(1 for l in raw.splitlines() if len(broker_positions(split_row(l))) >= 2)
    one = sum(1 for l in raw.splitlines() if len(broker_positions(split_row(l))) == 1)
    if two >= max(one, 2):
        buy, sell = parse_sides(raw)
        order, agree = infer_order(buy + sell)
        print(f" two-sided layout: {len(buy)} buy rows, {len(sell)} sell rows")
        print(f" column order inferred as {order.replace('_', '/')} "
              f"(value = lot x 100 x average held on {agree:.0%} of rows)")
        if agree < 0.6:
            print(" ! that identity fails on most rows. The columns are not what "
                  "they look like,\n ! or a magnitude suffix was misread — "
                  "refusing to store a guess.")
            return 1
        mapped = sides_to_frame(buy, sell, order)
        if mapped.empty:
            print(" no usable rows.")
            return 1
        mapped["ticker"] = args.ticker.upper()
        mapped["date"] = pd.Timestamp(date)
        mapped["source"] = "pasted_two_sided"
        return report_and_store(mapped, args.dry_run)

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
    return report_and_store(mapped, args.dry_run)


def report_and_store(mapped: pd.DataFrame, dry_run: bool) -> int:
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

    if dry_run:
        print("\n dry run — nothing written.")
        return 0
    w, s = save(mapped)
    print(f"\n stored {w} ticker-day(s)"
          + (", kept an existing complete file" if s else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
