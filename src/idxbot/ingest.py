"""Paste-in ingestion for broker summary.

The engine's whole broker-flow half is blocked on one thing: getting the daily
broker summary out of a trading platform and onto disk. No free API serves it,
but every Indonesian platform *displays* it - so the practical path is to select
the table, copy it, and paste it here.

That sounds trivial and is not, because pasted tables are hostile:

  * columns separated by tabs, multiple spaces, pipes or commas, inconsistently
  * Indonesian number formats (``1.234.567,89``) mixed with English ones
  * thousands suffixes (``1,2 M``, ``450 rb``, ``2.5B``)
  * a leading rank column, a trailing percentage column, or both
  * two tables side by side - buyers on the left, sellers on the right, which is
    how most platforms lay it out
  * header rows repeated mid-table, totals rows, blank lines

:func:`parse_pasted` handles all of the above and tells you what it inferred, so
a silent mis-parse becomes a visible one. Everything it produces goes through
the same :func:`~idxbot.data.broker_summary.normalise` path as file imports, so
downstream code cannot tell the difference.
"""

from __future__ import annotations

import os
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import Config
from .data.broker_summary import LOT_SIZE, SCHEMA, empty_frame, normalise

# Broker codes are 2-3 letters on IDX. Anchor row detection on that.
BROKER_RE = re.compile(r"^[A-Z]{2,3}$")

# "1,2 M" / "450rb" / "2.5B" - platforms abbreviate value columns aggressively.
SUFFIX = {
    "RB": 1e3, "K": 1e3,
    "JT": 1e6, "M": 1e6, "MN": 1e6,      # "M" in Indonesian listings = juta
    "B": 1e9, "MLR": 1e9, "BN": 1e9,
    "T": 1e12, "TR": 1e12,
}


def _to_number(token: str) -> float:
    """Parse one numeric cell, tolerating Indonesian formats and suffixes."""
    if token is None:
        return 0.0
    s = str(token).strip().upper().replace("(", "-").replace(")", "")
    if not s or s in {"-", "--", "N/A"}:
        return 0.0

    multiplier = 1.0
    for suffix, scale in sorted(SUFFIX.items(), key=lambda kv: -len(kv[0])):
        if s.endswith(suffix):
            candidate = s[: -len(suffix)].strip()
            # Only treat it as a suffix if what remains is actually numeric.
            if candidate and re.fullmatch(r"[-\d.,]+", candidate):
                s, multiplier = candidate, scale
                break

    s = re.sub(r"[^\d,.\-]", "", s)
    if not s:
        return 0.0

    last_dot, last_comma = s.rfind("."), s.rfind(",")
    if last_dot >= 0 and last_comma >= 0:
        if last_comma > last_dot:
            s = s.replace(".", "").replace(",", ".")
        else:
            s = s.replace(",", "")
    else:
        sep = "." if last_dot >= 0 else ("," if last_comma >= 0 else "")
        if sep:
            head, _, tail = s.rpartition(sep)
            if (len(tail) == 3 and tail.isdigit() and head not in ("", "-")) or s.count(sep) > 1:
                s = s.replace(sep, "")
            elif sep == ",":
                s = s.replace(",", ".")
    try:
        return float(s) * multiplier
    except ValueError:
        return 0.0


def _split_row(line: str) -> List[str]:
    """Split a pasted row on tabs, pipes, or runs of two or more spaces."""
    line = line.rstrip()
    if "\t" in line:
        return [c.strip() for c in line.split("\t")]
    if "|" in line:
        return [c.strip() for c in line.split("|")]
    if re.search(r"\s{2,}", line):
        return [c.strip() for c in re.split(r"\s{2,}", line)]
    return [c.strip() for c in line.split(",")]


def _find_broker_positions(cells: List[str]) -> List[int]:
    return [i for i, c in enumerate(cells) if BROKER_RE.match(c.strip().upper())]


def parse_pasted(
    text: str,
    ticker: str,
    date,
    volume_unit: str = "auto",
    source: str = "pasted",
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Parse a copied broker-summary table.

    Returns ``(frame, report)``. The report records what was inferred and what
    was skipped, because the failure mode that matters here is a *quiet* one -
    a table that parses into plausible-looking nonsense.
    """
    report: Dict[str, object] = {"rows_seen": 0, "rows_used": 0, "skipped": [],
                                 "layout": "unknown"}
    if not text or not text.strip():
        return empty_frame(), report

    lines = [ln for ln in text.splitlines() if ln.strip()]
    report["rows_seen"] = len(lines)

    records: List[dict] = []
    side_by_side = 0

    for line in lines:
        cells = _split_row(line)
        if len(cells) < 3:
            continue
        positions = _find_broker_positions(cells)
        if not positions:
            report["skipped"].append(line[:60])
            continue

        # Two broker codes on one line = buyers and sellers side by side, which
        # is the default layout on most Indonesian platforms.
        if len(positions) >= 2:
            side_by_side += 1
            left, right = positions[0], positions[1]
            buy_nums = [_to_number(c) for c in cells[left + 1:right]]
            sell_nums = [_to_number(c) for c in cells[right + 1:]]
            records.append(_record(cells[left], buy_nums, "buy"))
            records.append(_record(cells[right], sell_nums, "sell"))
        else:
            idx = positions[0]
            nums = [_to_number(c) for c in cells[idx + 1:]]
            records.append(_record(cells[idx], nums, "both"))

    if not records:
        return empty_frame(), report

    report["layout"] = "side-by-side (buyers | sellers)" if side_by_side > len(records) / 4 \
        else "single table"

    merged: Dict[str, dict] = {}
    for rec in records:
        code = rec.pop("broker")
        slot = merged.setdefault(code, {"buy_lot": 0.0, "buy_val": 0.0,
                                        "sell_lot": 0.0, "sell_val": 0.0})
        for key, value in rec.items():
            slot[key] = slot.get(key, 0.0) + value

    frame = pd.DataFrame([
        {"date": pd.Timestamp(date).normalize(), "ticker": str(ticker).upper(),
         "broker": code, **vals}
        for code, vals in merged.items()
    ])
    report["rows_used"] = len(frame)

    frame["buy_avg"] = np.where(frame["buy_lot"] > 0,
                                frame["buy_val"] / (frame["buy_lot"] * LOT_SIZE), 0.0)
    frame["sell_avg"] = np.where(frame["sell_lot"] > 0,
                                 frame["sell_val"] / (frame["sell_lot"] * LOT_SIZE), 0.0)
    frame["source"] = source

    out = normalise(frame, ticker=ticker, source=source, volume_unit=volume_unit)
    return out[SCHEMA], report


def _record(code: str, nums: List[float], side: str) -> dict:
    """Map the numbers following a broker code onto lot/value fields.

    Column order varies by platform, so infer by magnitude rather than position:
    a rupiah value is orders of magnitude larger than a lot count, and an
    average price sits in between at a plausible share price.
    """
    code = code.strip().upper()
    nums = [n for n in nums if n != 0.0]
    rec = {"broker": code, "buy_lot": 0.0, "buy_val": 0.0,
           "sell_lot": 0.0, "sell_val": 0.0}
    if not nums:
        return rec

    lot = nums[0]
    value = max(nums) if len(nums) > 1 else 0.0
    # If the largest number is not much bigger than the first, there is no
    # separate value column - treat the second as an average price.
    if len(nums) > 1 and value < lot * 50:
        avg = nums[1]
        value = lot * LOT_SIZE * avg if avg > 0 else 0.0

    if side in ("buy", "both"):
        rec["buy_lot"], rec["buy_val"] = lot, value
    if side == "sell":
        rec["sell_lot"], rec["sell_val"] = lot, value
    return rec


def save(frame: pd.DataFrame, cfg: Config, ticker: str,
         directory: Optional[str] = None) -> str:
    """Append to the per-ticker CSV the CSV provider already reads."""
    directory = directory or cfg.path("data.csv_dir", "data/broker_summary")
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, f"{str(ticker).upper()}.csv")

    if os.path.exists(path):
        existing = pd.read_csv(path, parse_dates=["date"])
        combined = pd.concat([existing, frame], ignore_index=True)
        # Last write wins for a given day, so re-pasting a corrected table works.
        combined = combined.drop_duplicates(subset=["date", "ticker", "broker"],
                                            keep="last")
    else:
        combined = frame

    combined.sort_values(["date", "broker"]).to_csv(path, index=False)
    return path


def describe(frame: pd.DataFrame, cfg: Config) -> str:
    """Human-readable check of what was just ingested."""
    if frame is None or frame.empty:
        return "  nothing parsed"

    registry = cfg.brokers
    out = []
    df = frame.copy()
    df["net_lot"] = df["buy_lot"] - df["sell_lot"]
    df["tier"] = df["broker"].map(lambda c: registry.get(c).tier)

    total_buy = df["buy_lot"].sum()
    total_sell = df["sell_lot"].sum()
    out.append(f"  brokers parsed : {len(df)}")
    out.append(f"  total buy      : {total_buy:,.0f} lots")
    out.append(f"  total sell     : {total_sell:,.0f} lots")

    # Every lot bought was sold. A large mismatch means the parse is wrong.
    if total_buy > 0 and total_sell > 0:
        imbalance = abs(total_buy - total_sell) / max(total_buy, total_sell)
        verdict = "OK" if imbalance < 0.02 else "CHECK THE PARSE"
        out.append(f"  buy/sell match : {1 - imbalance:.1%}  [{verdict}]")
        if imbalance >= 0.02:
            out.append("    Every lot bought is a lot sold. A mismatch this large")
            out.append("    means columns were misread - check the pasted table.")

    unknown = df[df["tier"] == "unknown"]["broker"].tolist()
    if unknown:
        out.append(f"  unknown codes  : {', '.join(unknown[:10])}"
                   f"{' ...' if len(unknown) > 10 else ''}")
        out.append("    (still ingested; add them to config/brokers.yaml to tier them)")

    top = df.nlargest(5, "net_lot")[["broker", "tier", "net_lot", "buy_avg"]]
    out.append("  top net buyers :")
    for _, r in top.iterrows():
        out.append(f"    {r['broker']:<4} {r['tier']:<11} {r['net_lot']:>12,.0f} lots"
                   f"  @ {r['buy_avg']:,.0f}")
    return "\n".join(out)
