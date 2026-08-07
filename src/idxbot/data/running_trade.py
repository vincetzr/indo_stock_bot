"""Reconstruct broker summary from running trade, in real time.

The key structural fact about IDX data: **broker summary is not a separate
product, it is an aggregation of running trade**. Every print on the exchange
carries a buyer broker code and a seller broker code. Sum those prints by broker
and you have reconstructed the broker summary yourself - the same table your
platform shows you two hours after the close, except you have it now.

This matters because running trade *with broker codes* is displayed live,
during the session, on essentially every Indonesian trading platform (Mirae
HOTS/Neo, IPOT, RTI, Stockbit, BIONS, MOST). The end-of-day delay applies to
the aggregated summary view, not to the underlying tick stream. If you can get
the prints out of whatever platform you already have, this module turns them
into live broker summary.

Tick schema consumed here
-------------------------
    ts       datetime  print timestamp
    ticker   str       bare IDX code
    price    float     trade price in rupiah
    lot      float     size in lots
    buyer    str       buying member code, e.g. "BK"
    seller   str       selling member code, e.g. "YP"

Feed it from any of:
  * a JSONL/CSV file your platform (or a userscript) appends to - use
    :meth:`RunningTradeAggregator.tail_file` for a live follow
  * a websocket or broker API - call :meth:`ingest` with parsed ticks
  * ``idxbot live --stdin`` piping one JSON object per line

See ``docs/LIVE_DATA.md`` for how to obtain the stream, including the licensing
and terms-of-service considerations, which are real and worth reading.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, Iterator, List, Optional

import pandas as pd

from .broker_summary import LOT_SIZE, SCHEMA, empty_frame

TICK_COLUMNS = ["ts", "ticker", "price", "lot", "buyer", "seller"]

# Aliases for running-trade exports, English and Indonesian.
TICK_ALIASES: Dict[str, tuple] = {
    "ts": ("ts", "time", "timestamp", "waktu", "jam", "datetime", "trade_time"),
    "ticker": ("ticker", "symbol", "stock", "kode", "code", "emiten"),
    "price": ("price", "harga", "last", "prc", "trade_price"),
    "lot": ("lot", "volume", "vol", "qty", "quantity", "size", "lots"),
    "buyer": ("buyer", "buy", "bbroker", "buybroker", "buyer_code", "pembeli",
              "brokerbeli", "b"),
    "seller": ("seller", "sell", "sbroker", "sellbroker", "seller_code", "penjual",
               "brokerjual", "s"),
}


def _pick(record: dict, field: str) -> Optional[object]:
    lowered = {str(k).lower().replace(" ", "_"): v for k, v in record.items()}
    for alias in TICK_ALIASES[field]:
        if alias in lowered:
            return lowered[alias]
    return None


def parse_tick(record: dict, default_ticker: Optional[str] = None,
               default_date: Optional[pd.Timestamp] = None) -> Optional[dict]:
    """Normalise one running-trade record; return None if unusable.

    Platforms often render only a clock time ("09:41:07") because the date is
    implicit in the session. ``default_date`` supplies it.
    """
    try:
        raw_ts = _pick(record, "ts")
        if raw_ts is None:
            ts = pd.Timestamp.now()
        else:
            ts = pd.to_datetime(raw_ts, errors="coerce")
            if pd.isna(ts):
                return None
            # A bare clock time parses to today's date; re-anchor if told to.
            if default_date is not None and ts.normalize() != pd.Timestamp(default_date).normalize():
                if len(str(raw_ts)) <= 8:
                    base = pd.Timestamp(default_date).normalize()
                    ts = base + (ts - ts.normalize())

        ticker = _pick(record, "ticker") or default_ticker
        if not ticker:
            return None

        price = float(_pick(record, "price") or 0)
        lot = float(_pick(record, "lot") or 0)
        buyer = _pick(record, "buyer")
        seller = _pick(record, "seller")
        if price <= 0 or lot <= 0 or not buyer or not seller:
            return None

        return {
            "ts": ts,
            "ticker": str(ticker).upper().replace(".JK", ""),
            "price": price,
            "lot": lot,
            "buyer": str(buyer).upper().strip(),
            "seller": str(seller).upper().strip(),
        }
    except (TypeError, ValueError):
        return None


@dataclass
class _BrokerBook:
    """Running totals for one broker in one ticker on one day."""

    buy_lot: float = 0.0
    buy_val: float = 0.0
    sell_lot: float = 0.0
    sell_val: float = 0.0

    def add_buy(self, lot: float, price: float) -> None:
        self.buy_lot += lot
        self.buy_val += lot * LOT_SIZE * price

    def add_sell(self, lot: float, price: float) -> None:
        self.sell_lot += lot
        self.sell_val += lot * LOT_SIZE * price


@dataclass
class RunningTradeAggregator:
    """Incrementally folds running-trade prints into live broker summary.

    Aggregation is O(1) per tick, so a full-session stream (hundreds of
    thousands of prints) costs nothing meaningful.
    """

    session_date: Optional[pd.Timestamp] = None
    #: (ticker, date) -> broker -> book
    books: Dict[tuple, Dict[str, _BrokerBook]] = field(default_factory=dict)
    tick_count: int = 0
    last_price: Dict[str, float] = field(default_factory=dict)
    last_ts: Optional[pd.Timestamp] = None
    _seen: set = field(default_factory=set)

    def ingest(self, ticks: Iterable[dict], dedupe: bool = True) -> int:
        """Fold ticks into the running books. Returns the number accepted.

        ``dedupe`` guards against the common case of re-reading a running-trade
        window that still contains prints already counted.
        """
        accepted = 0
        for raw in ticks:
            # Only skip parsing when the record is already normalised - which
            # means `ts` is a real Timestamp. Having the right *keys* is not
            # enough: JSON hands back a string timestamp with the same key.
            already_parsed = isinstance(raw.get("ts"), pd.Timestamp) and "buyer" in raw
            tick = raw if already_parsed else parse_tick(raw, default_date=self.session_date)
            if not tick:
                continue

            if dedupe:
                key = (tick["ts"].value, tick["ticker"], tick["price"],
                       tick["lot"], tick["buyer"], tick["seller"])
                if key in self._seen:
                    continue
                self._seen.add(key)

            day = pd.Timestamp(tick["ts"]).normalize()
            book_key = (tick["ticker"], day)
            books = self.books.setdefault(book_key, {})

            books.setdefault(tick["buyer"], _BrokerBook()).add_buy(tick["lot"], tick["price"])
            books.setdefault(tick["seller"], _BrokerBook()).add_sell(tick["lot"], tick["price"])

            self.last_price[tick["ticker"]] = tick["price"]
            self.last_ts = tick["ts"]
            self.tick_count += 1
            accepted += 1
        return accepted

    def snapshot(self, source: str = "running_trade") -> pd.DataFrame:
        """Current broker summary, in the canonical schema."""
        rows: List[dict] = []
        for (ticker, day), books in self.books.items():
            for broker, book in books.items():
                buy_shares = book.buy_lot * LOT_SIZE
                sell_shares = book.sell_lot * LOT_SIZE
                rows.append({
                    "date": day,
                    "ticker": ticker,
                    "broker": broker,
                    "buy_lot": book.buy_lot,
                    "buy_val": book.buy_val,
                    "buy_avg": book.buy_val / buy_shares if buy_shares else 0.0,
                    "sell_lot": book.sell_lot,
                    "sell_val": book.sell_val,
                    "sell_avg": book.sell_val / sell_shares if sell_shares else 0.0,
                    "source": source,
                })
        if not rows:
            return empty_frame()
        return pd.DataFrame(rows)[SCHEMA].sort_values(
            ["date", "ticker", "broker"]
        ).reset_index(drop=True)

    # -- live input helpers -------------------------------------------------
    def tail_file(
        self,
        path: str,
        poll_seconds: float = 1.0,
        from_start: bool = True,
        on_update: Optional[Callable[["RunningTradeAggregator"], None]] = None,
        max_seconds: Optional[float] = None,
    ) -> None:
        """Follow an append-only running-trade file, ``tail -f`` style.

        This is the practical live path: point your platform's export (or a
        browser userscript that mirrors the running-trade window) at a JSONL or
        CSV file, then run this against it. Broker summary stays current to
        within ``poll_seconds`` of the exchange.
        """
        started = time.time()
        position = 0
        header: Optional[List[str]] = None

        while True:
            if max_seconds is not None and time.time() - started > max_seconds:
                return
            if not os.path.exists(path):
                time.sleep(poll_seconds)
                continue

            size = os.path.getsize(path)
            if size < position:      # file truncated / rotated at session start
                position = 0
                header = None
            if size == position:
                time.sleep(poll_seconds)
                continue

            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                if position == 0 and not from_start:
                    fh.seek(size)
                    position = size
                else:
                    fh.seek(position)
                    chunk = fh.read()
                    position = fh.tell()
                    ticks, header = _parse_chunk(chunk, header, self.session_date)
                    if ticks:
                        self.ingest(ticks)
                        if on_update:
                            on_update(self)
            time.sleep(poll_seconds)

    def ingest_stream(self, lines: Iterable[str]) -> int:
        """Ingest newline-delimited JSON (e.g. piped on stdin)."""
        count = 0
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, list):
                count += self.ingest(record)
            elif isinstance(record, dict):
                count += self.ingest([record])
        return count


def _parse_chunk(chunk: str, header: Optional[List[str]],
                 session_date) -> tuple:
    """Parse a chunk that may be JSONL or CSV; returns (ticks, header)."""
    ticks: List[dict] = []
    for line in chunk.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("{") or line.startswith("["):
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            records = record if isinstance(record, list) else [record]
            for r in records:
                tick = parse_tick(r, default_date=session_date)
                if tick:
                    ticks.append(tick)
            continue

        parts = [p.strip() for p in line.split(",")]
        if header is None:
            # First non-JSON line is treated as the CSV header.
            header = parts
            continue
        if len(parts) != len(header):
            continue
        tick = parse_tick(dict(zip(header, parts)), default_date=session_date)
        if tick:
            ticks.append(tick)
    return ticks, header


def from_ticks_file(path: str, ticker: Optional[str] = None) -> pd.DataFrame:
    """One-shot: read a completed running-trade file into broker summary.

    Useful for backfilling history from archived tick files, and for the
    end-of-day path where you export the full session's prints at once.
    """
    agg = RunningTradeAggregator()
    if path.lower().endswith((".json", ".jsonl", ".ndjson")):
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            agg.ingest_stream(fh)
    else:
        df = pd.read_csv(path, sep=None, engine="python")
        agg.ingest(df.to_dict("records"))
    snap = agg.snapshot(source=f"running_trade:{os.path.basename(path)}")
    if ticker and not snap.empty:
        snap = snap[snap["ticker"] == str(ticker).upper()].reset_index(drop=True)
    return snap


def intraday_pace(agg: RunningTradeAggregator, ticker: str,
                  reference: Optional[pd.DataFrame] = None) -> Dict[str, float]:
    """Compare today's in-progress broker flow against a historical baseline.

    During the session the raw numbers are always "small" simply because the day
    is incomplete. What carries information is the *pace*: an institutional desk
    that has already absorbed 80% of its typical full-day volume by 10:30 is
    behaving differently from one tracking its normal rate.
    """
    snap = agg.snapshot()
    if snap.empty:
        return {}
    today = snap[snap["ticker"] == str(ticker).upper()]
    if today.empty:
        return {}

    out = {
        "ticks": float(agg.tick_count),
        "brokers_active": float(today["broker"].nunique()),
        "net_lot": float((today["buy_lot"] - today["sell_lot"]).sum()),
        "gross_lot": float((today["buy_lot"] + today["sell_lot"]).sum() / 2.0),
    }

    if reference is not None and not reference.empty:
        ref = reference[reference["ticker"] == str(ticker).upper()]
        if not ref.empty:
            per_day = ref.groupby("date")[["buy_lot", "sell_lot"]].sum()
            typical = float((per_day["buy_lot"] + per_day["sell_lot"]).median() / 2.0)
            if typical > 0:
                out["pace_vs_typical"] = out["gross_lot"] / typical

    # Session progress: IDX trades 09:00-15:50 WIB with a midday break, roughly
    # six hours of continuous trading.
    if agg.last_ts is not None:
        minutes = (agg.last_ts.hour - 9) * 60 + agg.last_ts.minute
        out["session_pct"] = max(0.0, min(1.0, minutes / 360.0))
        if out.get("pace_vs_typical") and out["session_pct"] > 0.05:
            # >1 means running hot relative to how much of the day has elapsed.
            out["pace_ratio"] = out["pace_vs_typical"] / out["session_pct"]
    return out
