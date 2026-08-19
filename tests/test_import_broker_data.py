"""Tests for the broker-data importer.

The property that matters: a rekap reconstructed from running trade must BALANCE
- total buy lots equal total sell lots - because every print has a buyer and a
seller. A top-N summary does not balance, and that gap is exactly how the
importer detects truncation. If these two ever behave the same way, the
importer has stopped being able to tell a complete rekap from a partial one.
"""

from __future__ import annotations

import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "scripts"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

from import_broker_data import (completeness, hint_from_name,      # noqa: E402
                                import_summary, import_ticks, looks_like_ticks,
                                read_any)

TICKS = pd.DataFrame({
    "Jam": ["09:00:12", "09:01:03", "09:02:44", "10:15:02"],
    "Kode": ["ADRO"] * 4,
    "Harga": [2450, 2455, 2460, 2470],
    "Lot": [150, 80, 220, 410],
    "Pembeli": ["YP", "CC", "BK", "YP"],
    "Penjual": ["BK", "YP", "PD", "ZP"],
})

SUMMARY = pd.DataFrame({
    "Broker": ["BK", "YP", "CC"],
    "BLot": [120000, 95000, 80000],
    "BVal": [7.56e9, 5.985e9, 5.04e9],
    "SLot": [90000, 110000, 60000],
    "SVal": [5.67e9, 6.93e9, 3.78e9],
})


# --------------------------------------------------------------------------- #
# telling the two shapes apart
# --------------------------------------------------------------------------- #
def test_indonesian_running_trade_headers_are_recognised_as_ticks():
    assert looks_like_ticks(TICKS)


def test_a_broker_summary_is_not_mistaken_for_ticks():
    assert not looks_like_ticks(SUMMARY)


def test_english_tick_headers_are_recognised_too():
    df = TICKS.rename(columns={"Jam": "time", "Kode": "symbol", "Harga": "price",
                               "Lot": "volume", "Pembeli": "buyer",
                               "Penjual": "seller"})
    assert looks_like_ticks(df)


# --------------------------------------------------------------------------- #
# the balance property
# --------------------------------------------------------------------------- #
def test_a_rekap_rebuilt_from_ticks_balances_exactly():
    out = import_ticks(TICKS, "ADRO")
    assert out is not None and not out.empty
    assert out["buy_lot"].sum() == out["sell_lot"].sum()


def test_the_rebuilt_rekap_totals_the_source_prints():
    out = import_ticks(TICKS, "ADRO")
    assert out["buy_lot"].sum() == TICKS["Lot"].sum()


def test_every_broker_in_the_prints_appears_in_the_rekap():
    out = import_ticks(TICKS, "ADRO")
    seen = set(TICKS["Pembeli"]) | set(TICKS["Penjual"])
    assert set(out["broker"]) == seen


def test_completeness_flags_a_truncated_summary():
    out = import_summary(SUMMARY.copy(), "BBCA", "2026-08-19")
    assert out is not None
    assert completeness(out)["imbalance"] > 0.02


def test_completeness_passes_a_balanced_rekap():
    out = import_ticks(TICKS, "ADRO")
    assert completeness(out)["imbalance"] == pytest.approx(0.0, abs=1e-12)


# --------------------------------------------------------------------------- #
# filename hints
# --------------------------------------------------------------------------- #
def test_ticker_is_read_across_an_underscore():
    """Regression: \\b never fires between BBCA and _, so this used to return None."""
    assert hint_from_name("BBCA_20260819_brokersummary.csv")[0] == "BBCA"


def test_date_is_read_from_the_filename():
    assert hint_from_name("ADRO_20260819_rt.csv")[1] == "2026-08-19"


def test_dashed_dates_are_read_too():
    assert hint_from_name("ADRO_2026-08-19.csv")[1] == "2026-08-19"


def test_a_nameless_file_yields_no_hints():
    assert hint_from_name("export (3).csv") == (None, None)


def test_a_summary_without_any_date_is_refused():
    """An undated broker row cannot be joined to a price bar, so it is dropped."""
    assert import_summary(SUMMARY.copy(), "BBCA", None) is None


# --------------------------------------------------------------------------- #
# reading real files off disk
# --------------------------------------------------------------------------- #
def test_reads_csv_semicolon_and_tab(tmp_path):
    for sep, name in ((",", "a.csv"), (";", "b.csv"), ("\t", "c.tsv")):
        p = tmp_path / name
        TICKS.to_csv(p, sep=sep, index=False)
        got = read_any(str(p))
        assert got is not None and len(got) == len(TICKS)


def test_reads_jsonl(tmp_path):
    p = tmp_path / "t.jsonl"
    p.write_text("\n".join(TICKS.to_json(orient="records", lines=True).splitlines()))
    got = read_any(str(p))
    assert got is not None and len(got) == len(TICKS)


def test_unreadable_file_returns_none(tmp_path):
    p = tmp_path / "x.csv"
    p.write_bytes(b"\x00\x01\x02")
    assert read_any(str(p)) is None or read_any(str(p)).empty
