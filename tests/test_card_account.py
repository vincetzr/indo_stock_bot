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


def _cfg_panel(n_days=100, **kw):
    """_panel with knobs: `crash` {ticker: (day, factor)}, `dividend`
    {ticker: (day, yield)}, `gone` {ticker: day it stops printing},
    `n_top` names at hi52 = 1.0."""
    n_top = kw.get("n_top", 10)
    dates = pd.bdate_range("2025-12-01", periods=n_days)
    rows = []
    for j, d in enumerate(dates):
        for n in range(40):
            t = f"W{n}" if n < n_top else f"Z{n:02d}"
            if t in kw.get("gone", {}) and j >= kw["gone"][t]:
                continue
            px = adj = 1000.0
            day, f = kw.get("crash", {}).get(t, (None, 1.0))
            if day is not None and j >= day:
                px = adj = 1000.0 * f
            day, y = kw.get("dividend", {}).get(t, (None, 0.0))
            if day is not None and j >= day:
                adj = adj * (1.0 + y)
            rows.append({"date": d, "ticker": t, "close": px,
                         "adj_close": adj, "tv60": 5e9, "tradeable": True,
                         "hi52": 1.0 if n < n_top else 0.5,
                         "vol60": 0.01 if n < n_top else 0.03})
    P = pd.DataFrame(rows)
    P["elig"] = P["tradeable"]
    return P, dates


def test_a_dividend_is_reinvested_net_of_tax(tmp_path):
    P, dates = _cfg_panel(dividend={"W1": (40, 0.05)})
    base, _ = _cfg_panel()
    ix = _index(tmp_path, dates)
    r = ca.run(50e6, str(dates[20].date()), raw=P, index_path=ix)
    r0 = ca.run(50e6, str(dates[20].date()), raw=base, index_path=ix)
    w1 = 49 * 100 * 1000.0                    # W1's value when it went ex
    assert r["dividends"] == pytest.approx(w1 * 0.05)
    #  the review at session 83 re-equalises, so compare just before it
    d = str(dates[82].date())
    assert r["curve"][d] - r0["curve"][d] == pytest.approx(
        w1 * 0.05 * (1 - ca.DIV_TAX))


def test_a_name_stopped_on_the_fill_day_is_not_bought_back(tmp_path):
    """W0 is kept at the review decided on session 82 and falls 30% on 83,
    the fill day. The stop sells it; the fill must not buy it straight back
    at the same close."""
    P, dates = _cfg_panel(crash={"W0": (83, 0.70)})
    r = ca.run(50e6, str(dates[20].date()), raw=P,
               index_path=_index(tmp_path, dates))
    day = str(dates[83].date())
    acts = [(x["action"], x["ticker"]) for x in r["log"]
            if x["date"] == day]
    assert ("SELL", "W0") in acts
    assert ("BUY", "W0") not in acts


def test_a_tie_at_the_cut_is_reported(tmp_path):
    P, dates = _cfg_panel(n_top=12)
    r = ca.run(50e6, str(dates[20].date()), raw=P,
               index_path=_index(tmp_path, dates))
    assert r["ties"] and "12 names tied" in r["ties"][0]
    a = ca.run(50e6, str(dates[20].date()), raw=P,
               index_path=_index(tmp_path, dates), tie_seed=1)
    assert "random draw" in a["ties"][0]


def test_a_name_that_stops_printing_is_realised_at_the_review(tmp_path):
    P, dates = _cfg_panel(gone={"W5": 50})
    r = ca.run(50e6, str(dates[20].date()), raw=P,
               index_path=_index(tmp_path, dates))
    gone = [x for x in r["log"] if x["ticker"] == "W5"
            and x["action"] == "SELL"]
    assert len(gone) == 1 and "stopped printing" in gone[0]["why"]
    assert gone[0]["date"] == str(dates[82].date())


def test_hold_first_never_trades_after_the_first_fill(tmp_path):
    P, dates = _cfg_panel(crash={"W0": (30, 0.50)})
    r = ca.run(50e6, str(dates[20].date()), raw=P,
               index_path=_index(tmp_path, dates), selector="hold_first")
    assert {x["action"] for x in r["log"]} <= {"REVIEW", "BUY"}
    assert r["stops"] == 0


def test_random_controls_draw_from_the_eligible_board(tmp_path):
    P, dates = _cfg_panel()
    for how in ("random_fresh", "random_sticky"):
        r = ca.run(50e6, str(dates[20].date()), raw=P,
                   index_path=_index(tmp_path, dates), selector=how, seed=3)
        buys = {x["ticker"] for x in r["log"] if x["action"] == "BUY"}
        assert buys and buys <= set(P["ticker"])


def test_the_equal_weight_universe_pays_only_its_entry(tmp_path):
    P, dates = _cfg_panel()
    r = ca.run(50e6, str(dates[20].date()), raw=P,
               index_path=_index(tmp_path, dates))
    assert r["ew_end"] == pytest.approx(
        50e6 * (1 - ca.side_cost(1000.0, ca.BUY_FEE, 0.5)))
