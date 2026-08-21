"""Tests for the licensed full-rekap route.

This module is different from every other data source in the package in two
ways that decide what has to be tested:

  IT SPENDS MONEY. One call is one credit and returns up to fourteen days, so
  a per-day fetcher would cost fourteen times as much for identical data. The
  windowing is not an optimisation, it is the difference between a 29-credit
  backfill and a 400-credit one, and it is only correct if two callers asking
  for two different days in one fortnight compute the SAME window.

  IT HOLDS A CREDENTIAL. A key must never reach the cache, a log line or an
  exception message, and a config placeholder must never be sent as if it were
  a real key.

The parsing is checked against the vendor's own worked example, because the
identity value = lots x 100 x average is the only thing that would catch a
silently mis-mapped column - and reading `navg_per_share` where `bavg_per_share`
was meant is a mistake that produces plausible numbers and wrong conclusions.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

from idxbot.data.cache import Cache                          # noqa: E402
from idxbot.data.sectors import (CONSISTENCY_BOUND,          # noqa: E402
                                 MAX_WINDOW_DAYS,
                                 SectorsBrokerSummary, api_key, consistency,
                                 parse_payload, window_for, windows_covering)

#: Verbatim from the vendor's API specification, including the 55-lot broker
#: that proves the endpoint is not a top-ten table.
VENDOR_EXAMPLE = {
    "symbol": "BBCA.JK", "start": "2025-05-01", "end": "2025-05-14",
    "data": [{"date": "2025-05-02", "summary": [
        {"broker_code": "AF", "bfreq": 1, "blot": 55, "bval": 48950000,
         "bavg_per_share": 8900, "sfreq": 1, "slot": 50, "sval": 44875000,
         "savg_per_share": 8975, "nlot": 5, "nval": 4075000,
         "navg_per_share": 8900}]}],
}


# --------------------------------------------------------------------------
# the windowing, which is what the bill depends on
# --------------------------------------------------------------------------
def test_two_days_in_one_fortnight_resolve_to_one_window():
    """The whole credit saving rests on this."""
    s, e = window_for(pd.Timestamp("2025-05-08"))
    assert window_for(s) == (s, e)
    assert window_for(e) == (s, e)
    assert window_for(s + pd.Timedelta(days=6)) == (s, e)


def test_the_day_after_a_window_ends_is_a_different_window():
    """Alignment is to a fixed epoch, so boundaries fall where they fall - and
    a range that straddles one costs two credits, not one. Stating it here so
    the cost model is not quietly assumed to be ceil(days / 14)."""
    s, e = window_for(pd.Timestamp("2025-05-08"))
    assert window_for(e + pd.Timedelta(days=1))[0] == e + pd.Timedelta(days=1)
    assert len(windows_covering("2025-05-01", "2025-05-14")) == 2


def test_a_window_is_exactly_the_vendors_maximum():
    s, e = window_for(pd.Timestamp("2025-05-02"))
    assert (e - s).days + 1 == MAX_WINDOW_DAYS


def test_windows_are_aligned_to_a_fixed_epoch_not_to_the_request():
    """Aligning to the requested day would make overlapping windows and pay
    twice for the days in the overlap."""
    starts = {window_for(pd.Timestamp("2025-05-01") + pd.Timedelta(days=i))[0]
              for i in range(60)}
    # 60 days spans 5 fortnights at most, and every one of them is a multiple
    # of 14 days from the epoch
    assert len(starts) <= 5
    for s in starts:
        assert (s - pd.Timestamp("2000-01-03")).days % MAX_WINDOW_DAYS == 0


def test_windows_covering_a_range_do_not_overlap_and_leave_no_gap():
    w = windows_covering("2025-01-01", "2025-06-30")
    assert w == sorted(w)
    for (s1, e1), (s2, _) in zip(w, w[1:]):
        assert s2 == e1 + pd.Timedelta(days=1)
    assert w[0][0] <= pd.Timestamp("2025-01-01")
    assert w[-1][1] >= pd.Timestamp("2025-06-30")


def test_a_single_day_needs_exactly_one_window():
    assert len(windows_covering("2025-05-02", "2025-05-02")) == 1


def test_a_backwards_range_asks_for_nothing():
    assert windows_covering("2025-06-30", "2025-01-01") == []


def test_the_quoted_cost_counts_calendar_days_not_trading_days():
    """A fortnight holds ~10 sessions, so costing it in sessions understates
    the bill by about 40% - which is the direction that matters."""
    # 400 sessions is ~560 calendar days, ~40 windows, times 10 names
    assert SectorsBrokerSummary.credits_for(10, 400) == 400
    assert SectorsBrokerSummary.credits_for(1, 14) > 1     # 14 sessions > 14 days
    assert SectorsBrokerSummary.credits_for(0, 400) == 0


# --------------------------------------------------------------------------
# parsing, checked against the identity that catches a mis-mapped column
# --------------------------------------------------------------------------
def test_the_vendors_own_example_parses_to_the_canonical_schema():
    df = parse_payload(VENDOR_EXAMPLE, "BBCA")
    assert len(df) == 1
    r = df.iloc[0]
    assert r["ticker"] == "BBCA" and r["broker"] == "AF"
    assert r["date"] == pd.Timestamp("2025-05-02")
    assert r["buy_lot"] == 55 and r["buy_val"] == 48_950_000
    assert r["buy_avg"] == 8900 and r["sell_avg"] == 8975
    assert r["source"] == "sectors"


def test_value_reconciles_to_lots_times_a_hundred_times_average():
    c = consistency(parse_payload(VENDOR_EXAMPLE, "BBCA"))
    assert (c["worst"].dropna() < CONSISTENCY_BOUND).all()


def test_reading_the_net_average_instead_of_the_buy_average_is_caught():
    """The mistake this test exists for produces entirely plausible numbers."""
    bad = {"data": [{"date": "2025-05-02", "summary": [
        {"broker_code": "AF", "blot": 55, "bval": 48950000,
         # navg, not bavg - a one-word slip in the field map
         "bavg_per_share": 8900 * 1.5,
         "slot": 50, "sval": 44875000, "savg_per_share": 8975}]}]}
    c = consistency(parse_payload(bad, "BBCA"))
    assert float(c.loc[c["side"] == "buy", "worst"].iloc[0]) > CONSISTENCY_BOUND


def test_trade_counts_survive_the_parse():
    """bfreq/sfreq are the one field no free route provides."""
    df = parse_payload(VENDOR_EXAMPLE, "BBCA")
    assert df.iloc[0]["buy_freq"] == 1 and df.iloc[0]["sell_freq"] == 1


def test_a_broker_with_no_code_is_dropped_rather_than_named_blank():
    bad = {"data": [{"date": "2025-05-02", "summary": [
        {"broker_code": "", "blot": 1, "bval": 1, "bavg_per_share": 1},
        {"broker_code": "AF", "blot": 1, "bval": 100, "bavg_per_share": 1}]}]}
    df = parse_payload(bad, "BBCA")
    assert list(df["broker"]) == ["AF"]


def test_an_unparseable_date_drops_that_block_not_the_whole_payload():
    mixed = {"data": [{"date": "not-a-date", "summary": [
        {"broker_code": "ZZ", "blot": 1, "bval": 1, "bavg_per_share": 1}]},
        VENDOR_EXAMPLE["data"][0]]}
    df = parse_payload(mixed, "BBCA")
    assert list(df["broker"]) == ["AF"]


def test_an_empty_payload_is_an_empty_frame_not_a_crash():
    for p in ({}, {"data": []}, {"data": [{"date": "2025-05-02", "summary": []}]}):
        assert parse_payload(p, "BBCA").empty
    # a frame with the right columns and no rows, never a None or an exception
    c = consistency(pd.DataFrame())
    assert list(c.columns) == ["side", "rows", "worst", "median"]
    assert len(c) == 0


# --------------------------------------------------------------------------
# the credential
# --------------------------------------------------------------------------
def test_the_environment_key_is_used(monkeypatch):
    monkeypatch.setenv("SECTORS_API_KEY", "sk-real")
    assert api_key(None) == "sk-real"


def test_a_config_placeholder_is_never_sent_as_a_credential(monkeypatch):
    monkeypatch.delenv("SECTORS_API_KEY", raising=False)

    class Cfg:
        def __init__(self, v):
            self.v = v

        def get(self, k, d=None):
            return self.v

    for placeholder in ("YOUR_API_KEY_HERE", "<your key>", "changeme", "TODO",
                        "", "   "):
        assert api_key(Cfg(placeholder)) is None
    assert api_key(Cfg("sk-actual")) == "sk-actual"


def test_no_key_means_unavailable_not_a_crash(monkeypatch):
    monkeypatch.delenv("SECTORS_API_KEY", raising=False)
    p = SectorsBrokerSummary(cache=None, key=None)
    assert p.available() is False
    assert p.fetch_day("BBCA", pd.Timestamp("2025-05-02")).empty
    assert p.credits_spent == 0


def test_a_rejected_key_is_reported_without_echoing_it(tmp_path,
                                                       monkeypatch):
    import idxbot.data.sectors as S

    class Resp:
        status_code = 403
        text = "forbidden"

    class FakeRequests:
        @staticmethod
        def get(url, **kw):
            return Resp()

    monkeypatch.setitem(sys.modules, "requests", FakeRequests)
    p = S.SectorsBrokerSummary(cache=Cache(str(tmp_path)), key="sk-SECRET",
                               delay=0.0, retries=1, verbose=True)
    out = p.fetch_day("BBCA", pd.Timestamp("2025-05-02"))
    assert out.empty
    # nothing anywhere should carry the key
    for f in tmp_path.rglob("*"):
        if f.is_file():
            assert b"sk-SECRET" not in f.read_bytes()


def test_a_failed_call_is_not_billed(tmp_path, monkeypatch):
    import idxbot.data.sectors as S

    class Resp:
        status_code = 429

    monkeypatch.setitem(sys.modules, "requests",
                        type("R", (), {"get": staticmethod(lambda u, **k: Resp())}))
    p = S.SectorsBrokerSummary(cache=Cache(str(tmp_path)), key="k",
                               delay=0.0, retries=2)
    p.fetch_day("BBCA", pd.Timestamp("2025-05-02"))
    assert p.credits_spent == 0


# --------------------------------------------------------------------------
# the cache, which is what stops a paid day being paid for twice
# --------------------------------------------------------------------------
def _fake_provider(tmp_path, calls):
    import idxbot.data.sectors as S

    class Resp:
        status_code = 200

        @staticmethod
        def json():
            return VENDOR_EXAMPLE

        @staticmethod
        def raise_for_status():
            return None

    def get(url, **kw):
        calls.append(kw.get("params"))
        return Resp()

    return S.SectorsBrokerSummary(cache=Cache(str(tmp_path)), key="k",
                                  delay=0.0), get


def test_a_paid_fortnight_is_never_paid_for_twice(tmp_path, monkeypatch):
    calls = []
    p, get = _fake_provider(tmp_path, calls)
    monkeypatch.setitem(sys.modules, "requests",
                        type("R", (), {"get": staticmethod(get)}))
    p.fetch_day("BBCA", pd.Timestamp("2025-05-02"))
    p.fetch_day("BBCA", pd.Timestamp("2025-05-02"))
    assert len(calls) == 1 and p.credits_spent == 1


def test_asking_for_a_different_day_in_the_same_fortnight_costs_nothing_more(
        tmp_path, monkeypatch):
    """The single most valuable behaviour in this module."""
    calls = []
    p, get = _fake_provider(tmp_path, calls)
    monkeypatch.setitem(sys.modules, "requests",
                        type("R", (), {"get": staticmethod(get)}))
    p.fetch_day("BBCA", pd.Timestamp("2025-05-02"))
    for i in range(MAX_WINDOW_DAYS):
        p.fetch_day("BBCA", window_for(pd.Timestamp("2025-05-02"))[0]
                    + pd.Timedelta(days=i))
    assert len(calls) == 1


def test_fetch_day_returns_only_the_day_asked_for(tmp_path, monkeypatch):
    calls = []
    p, get = _fake_provider(tmp_path, calls)
    monkeypatch.setitem(sys.modules, "requests",
                        type("R", (), {"get": staticmethod(get)}))
    same = p.fetch_day("BBCA", pd.Timestamp("2025-05-02"))
    other = p.fetch_day("BBCA", pd.Timestamp("2025-05-05"))
    assert len(same) == 1 and other.empty


def test_an_empty_fortnight_is_not_cached_as_permanent_emptiness(
        tmp_path, monkeypatch):
    """A holiday fortnight and an unreachable one look identical from here."""
    import idxbot.data.sectors as S
    calls = []

    class Resp:
        status_code = 200

        @staticmethod
        def json():
            return {"data": []}

        @staticmethod
        def raise_for_status():
            return None

    def get(url, **kw):
        calls.append(1)
        return Resp()

    monkeypatch.setitem(sys.modules, "requests",
                        type("R", (), {"get": staticmethod(get)}))
    p = S.SectorsBrokerSummary(cache=Cache(str(tmp_path)), key="k", delay=0.0)
    p.fetch_day("BBCA", pd.Timestamp("2025-05-02"))
    p.fetch_day("BBCA", pd.Timestamp("2025-05-02"))
    assert len(calls) == 2


def test_this_route_declares_itself_complete():
    """Downstream bounds collapse to points on it, so the flag must be right."""
    assert SectorsBrokerSummary.complete is True
    from idxbot.data.ipot import IpotBrokerSummary
    assert getattr(IpotBrokerSummary, "complete", False) is False
