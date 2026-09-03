"""The settled close that arrives in `meta` rather than in the array.

THE BUG: Yahoo backfills the daily `close` array hours after an exchange
closes, but `meta.regularMarketPrice` carries the settled price within minutes.
`dropna(subset=["close"])` deleted the row, so every EOD signal this repo
produced was ONE FULL SESSION STALE on the evening it was meant to be acted on.
Measured 2026-09-03 at 17:43 UTC, nine hours after the 08:50 UTC Jakarta close:
the array held None while meta had ADRO at 2,740 against the prior 2,650.

THE RISK THE FIX INTRODUCES is the opposite one — an INTRADAY quote leaking
into a daily bar — so the settled-close test is what these tests are really
about.
"""

from __future__ import annotations

import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

from idxbot.data.ohlcv import YahooOHLCV                          # noqa: E402

DAY = 24 * 3600
T_CLOSE = 1788426611          # 2026-09-03 09:10:11 UTC = 16:10 WIB
T_NEXT_OPEN = 1788487200      # 2026-09-04 02:00:00 UTC = 09:00 WIB


def _payload(close_last=None, mt=T_CLOSE, start=T_NEXT_OPEN, px=2350.0):
    ts = [T_CLOSE - 2 * DAY, T_CLOSE - DAY, T_CLOSE]
    return {"chart": {"result": [{
        "timestamp": ts,
        "indicators": {"quote": [{"open": [2300.0, 2310.0, 2340.0],
                                  "high": [2320.0, 2350.0, 2350.0],
                                  "low": [2280.0, 2250.0, 2210.0],
                                  "close": [2310.0, 2330.0, close_last],
                                  "volume": [1e6, 2e6, 3e6]}]},
        "meta": {"regularMarketPrice": px, "regularMarketTime": mt,
                 "regularMarketDayHigh": 2350.0, "regularMarketDayLow": 2210.0,
                 "regularMarketVolume": 52292700.0,
                 "currentTradingPeriod": {"regular": {"start": start,
                                                      "end": start + 26100}}},
    }]}}


def test_the_settled_close_is_recovered_from_meta():
    df = YahooOHLCV._parse(_payload(close_last=None))
    assert len(df) == 3, "today's row must survive"
    assert df["close"].iloc[-1] == pytest.approx(2350.0)
    #  adj_close == close on the newest bar: no later corporate action applied.
    assert df["adj_close"].iloc[-1] == pytest.approx(2350.0)
    assert df["volume"].iloc[-1] == pytest.approx(52292700.0)


def test_an_INTRADAY_quote_is_never_written_into_a_daily_bar():
    """THE WHOLE SAFETY OF THE FIX. `currentTradingPeriod.regular` only rolls
    to the NEXT session once the current one has ended, so a start that is
    still BEFORE regularMarketTime means the market is open right now and the
    price is a live quote, not a close."""
    mid_session = T_CLOSE - 3 * 3600
    p = _payload(close_last=None, mt=mid_session, start=mid_session - 7200)
    df = YahooOHLCV._parse(p)
    assert len(df) == 2, "an open session must NOT produce a daily bar"


def test_a_stale_meta_price_from_a_different_day_is_ignored():
    """If regularMarketTime is yesterday, meta describes a session the last row
    is not, and copying it in would stamp the wrong day's price."""
    p = _payload(close_last=None, mt=T_CLOSE - DAY, start=T_NEXT_OPEN)
    df = YahooOHLCV._parse(p)
    assert len(df) == 2


def test_a_real_array_close_is_never_overwritten_by_meta():
    """Once Yahoo backfills the array, the array wins. meta can lag or carry a
    consolidated price that differs; the bar must not silently change after the
    fact or a backtest stops being reproducible."""
    df = YahooOHLCV._parse(_payload(close_last=2295.0, px=2350.0))
    assert df["close"].iloc[-1] == pytest.approx(2295.0)


def test_no_meta_means_the_old_behaviour():
    p = _payload(close_last=None)
    del p["chart"]["result"][0]["meta"]
    assert len(YahooOHLCV._parse(p)) == 2
