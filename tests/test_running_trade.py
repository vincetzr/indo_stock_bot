"""Running trade -> broker summary reconstruction.

This is the path that makes live broker flow possible, so the invariants matter:
every lot bought is a lot sold, VWAPs are volume-weighted, and re-reading a
window that still contains counted prints must not double-count them.
"""

import json
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

from idxbot.data.running_trade import (  # noqa: E402
    RunningTradeAggregator,
    from_ticks_file,
    intraday_pace,
    parse_tick,
)

TICKS = [
    {"ts": "2026-08-07 09:41:07", "ticker": "BBCA", "price": 6350, "lot": 150,
     "buyer": "BK", "seller": "YP"},
    {"ts": "2026-08-07 09:41:11", "ticker": "BBCA", "price": 6375, "lot": 200,
     "buyer": "BK", "seller": "CC"},
    {"ts": "2026-08-07 09:43:20", "ticker": "BBCA", "price": 6400, "lot": 60,
     "buyer": "BK", "seller": "PD"},
]


def test_parse_tick_from_json_strings():
    tick = parse_tick(TICKS[0])
    assert isinstance(tick["ts"], pd.Timestamp)
    assert tick["ticker"] == "BBCA"
    assert tick["buyer"] == "BK"
    assert tick["lot"] == 150


def test_parse_tick_rejects_incomplete_records():
    assert parse_tick({"ts": "2026-08-07", "ticker": "BBCA"}) is None
    assert parse_tick({"price": 100, "lot": 1, "buyer": "BK", "seller": "YP"}) is None
    assert parse_tick({"ts": "x", "ticker": "T", "price": 0, "lot": 5,
                       "buyer": "BK", "seller": "YP"}) is None


def test_parse_tick_indonesian_aliases():
    tick = parse_tick({"waktu": "2026-08-07 09:41:07", "kode": "BBCA",
                       "harga": 6350, "volume": 10,
                       "pembeli": "BK", "penjual": "YP"})
    assert tick is not None
    assert tick["price"] == 6350
    assert tick["seller"] == "YP"


def test_ingest_parses_raw_json_records():
    """Regression: records with the right KEYS but a string ts must be parsed.

    An earlier version treated any dict containing 'ts' and 'buyer' as already
    normalised, then crashed reaching for .value on a string.
    """
    agg = RunningTradeAggregator()
    assert agg.ingest(TICKS) == 3
    assert agg.tick_count == 3


def test_snapshot_conserves_volume():
    """Every lot bought is a lot sold - the defining invariant of the tape."""
    agg = RunningTradeAggregator()
    agg.ingest(TICKS)
    snap = agg.snapshot()
    assert snap["buy_lot"].sum() == pytest.approx(snap["sell_lot"].sum())
    assert snap["buy_lot"].sum() == pytest.approx(410)
    assert snap["buy_val"].sum() == pytest.approx(snap["sell_val"].sum())


def test_snapshot_computes_volume_weighted_average():
    agg = RunningTradeAggregator()
    agg.ingest(TICKS)
    snap = agg.snapshot()
    bk = snap[snap["broker"] == "BK"].iloc[0]
    expected = (150 * 6350 + 200 * 6375 + 60 * 6400) / 410
    assert bk["buy_lot"] == pytest.approx(410)
    assert bk["buy_avg"] == pytest.approx(expected)
    assert bk["sell_lot"] == 0


def test_both_sides_of_a_print_are_recorded():
    agg = RunningTradeAggregator()
    agg.ingest([TICKS[0]])
    snap = agg.snapshot()
    assert set(snap["broker"]) == {"BK", "YP"}
    assert snap[snap["broker"] == "YP"].iloc[0]["sell_lot"] == 150


def test_dedupe_prevents_double_counting():
    """Re-reading a running-trade window must not inflate the totals."""
    agg = RunningTradeAggregator()
    agg.ingest(TICKS)
    agg.ingest(TICKS)                       # same window read again
    assert agg.tick_count == 3
    assert agg.snapshot()["buy_lot"].sum() == pytest.approx(410)


def test_dedupe_can_be_disabled():
    agg = RunningTradeAggregator()
    agg.ingest(TICKS, dedupe=False)
    agg.ingest(TICKS, dedupe=False)
    assert agg.tick_count == 6


def test_ingest_stream_reads_jsonl():
    agg = RunningTradeAggregator()
    lines = [json.dumps(t) for t in TICKS] + ["", "not json"]
    assert agg.ingest_stream(lines) == 3


def test_from_ticks_file_jsonl(tmp_path):
    path = tmp_path / "ticks.jsonl"
    path.write_text("\n".join(json.dumps(t) for t in TICKS))
    snap = from_ticks_file(str(path))
    assert not snap.empty
    assert snap["buy_lot"].sum() == pytest.approx(snap["sell_lot"].sum())


def test_from_ticks_file_csv(tmp_path):
    path = tmp_path / "ticks.csv"
    pd.DataFrame(TICKS).to_csv(path, index=False)
    snap = from_ticks_file(str(path), ticker="BBCA")
    assert not snap.empty
    assert set(snap["ticker"]) == {"BBCA"}


def test_snapshot_output_matches_broker_summary_schema():
    """The reconstruction must be ingestible by the rest of the pipeline."""
    from idxbot.data.broker_summary import SCHEMA, normalise

    agg = RunningTradeAggregator()
    agg.ingest(TICKS)
    snap = agg.snapshot()
    assert list(snap.columns) == SCHEMA

    round_trip = normalise(snap, source="roundtrip", volume_unit="lot")
    assert round_trip["buy_lot"].sum() == pytest.approx(410)


def test_intraday_pace_reports_activity():
    agg = RunningTradeAggregator()
    agg.ingest(TICKS)
    pace = intraday_pace(agg, "BBCA")
    assert pace["ticks"] == 3
    assert pace["brokers_active"] == 4
    assert pace["gross_lot"] == pytest.approx(410)


def test_empty_snapshot():
    assert RunningTradeAggregator().snapshot().empty
