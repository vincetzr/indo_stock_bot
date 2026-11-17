"""scripts/card_account.py: Rp X into the card as one account.

The replay is only worth showing if it trades the way a holder could have:
whole lots, the user's fees, orders the evening after the decision, and never
a fill on a bar nobody could trade. Each of those has a test.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir,
                                "scripts"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

import card_account as ca                                          # noqa: E402
import rules                                                       # noqa: E402
from paint_suite import tick_of                                    # noqa: E402


def test_costs_are_the_users_schedule_plus_the_spread():
    assert ca.BUY_FEE == pytest.approx(0.0028)
    assert ca.SELL_FEE == pytest.approx(0.0028)       # 0.18% + 0.10% tax
    px = 1000.0
    assert ca.side_cost(px, ca.BUY_FEE, 0.5) == pytest.approx(
        0.0028 + 0.25 * tick_of(px) / px)


def test_whole_lots_only():
    assert ca.lots_affordable(5_000_000, 9_900, 0.003) == 5
    assert ca.lots_affordable(900_000, 9_900, 0.003) == 0


def test_levels_are_the_shipped_card():
    assert (ca.STOP, ca.TP, ca.TP_FRAC, ca.K) == (
        rules.STOP, rules.TP, rules.TP_FRAC, rules.K)


def _panel(n_days=80, crash_day=None, locked=()):
    """40 eligible names; W0..W9 sit at their 52-week high and are calm, so
    they are the basket. W0 can fall 30% on `crash_day` and be untradeable on
    the days in `locked`."""
    dates = pd.bdate_range("2025-12-01", periods=n_days)
    rows = []
    for j, d in enumerate(dates):
        for n in range(40):
            t = f"W{n}" if n < 10 else f"Z{n:02d}"
            px = 1000.0
            if t == "W0" and crash_day is not None and j >= crash_day:
                px = 700.0
            rows.append({"date": d, "ticker": t, "close": px,
                         "adj_close": px, "tv60": 5e9,
                         "tradeable": not (t == "W0" and j in locked),
                         "hi52": 1.0 if n < 10 else 0.5,
                         "vol60": 0.01 if n < 10 else 0.03})
    P = pd.DataFrame(rows)
    P["elig"] = P["tradeable"]
    return P, dates


def _index(tmp_path, dates):
    p = tmp_path / "jkse.csv"
    pd.DataFrame({"date": dates, "close": 100.0}).to_csv(p, index=False)
    return str(p)


def test_the_first_buy_is_the_session_after_the_decision(tmp_path):
    P, dates = _panel()
    start = str(dates[20].date())
    r = ca.run(50e6, start, raw=P, index_path=_index(tmp_path, dates))
    review = [x for x in r["log"] if x["action"] == "REVIEW"][0]
    buys = [x for x in r["log"] if x["action"] == "BUY"]
    assert review["date"] == str(dates[19].date())
    assert {b["date"] for b in buys} == {str(dates[20].date())}
    assert len(buys) == 10
    #  Rp 5 jt a name at Rp 1,000 is 49 lots once costs are paid
    assert {b["lots"] for b in buys} == {49}


def test_the_stop_sells_at_the_breach_close(tmp_path):
    P, dates = _panel(crash_day=30)
    r = ca.run(50e6, str(dates[20].date()), raw=P,
               index_path=_index(tmp_path, dates))
    sells = [x for x in r["log"] if x["action"] == "SELL"]
    assert [(x["ticker"], x["date"]) for x in sells] == [
        ("W0", str(dates[30].date()))]
    assert r["stops"] == 1


def test_nothing_fills_on_a_locked_bar(tmp_path):
    """W0 falls 30% into an auto-rejection lock for two sessions: the stop
    can only sell on the first bar anyone could trade."""
    P, dates = _panel(crash_day=30, locked=(30, 31))
    r = ca.run(50e6, str(dates[20].date()), raw=P,
               index_path=_index(tmp_path, dates))
    sells = [x for x in r["log"] if x["action"] == "SELL"]
    assert [x["date"] for x in sells] == [str(dates[32].date())]


def test_cash_is_conserved_through_a_round_trip(tmp_path):
    """Flat prices: the only change in value is what the fees and the spread
    took."""
    P, dates = _panel(crash_day=None)
    r = ca.run(50e6, str(dates[20].date()), raw=P,
               index_path=_index(tmp_path, dates))
    bought = sum(x["rupiah"] for x in r["log"] if x["action"] == "BUY")
    fees = bought - sum(x["lots"] * 100 * x["price"] for x in r["log"]
                        if x["action"] == "BUY")
    assert r["end_value"] == pytest.approx(50e6 - fees, rel=1e-9)
