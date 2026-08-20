"""Tests for the sizing and framing of the plan.

Two kinds of thing are pinned here, and the second matters more:

  ARITHMETIC. Whole lots, weight drift, minimum viable capital, income, the
  drawdown in rupiah. A plan that misstates any of these is worse than no plan,
  because it will be acted on.

  HONESTY. plan_return must return LESS than what was measured, and it must be
  impossible for it to quietly return more. An estimated premium is the true one
  plus whatever the estimate got wrong, and in this sample the error has a known
  direction: the universe holds no delistings and thirteen factors were looked
  at before one was chosen. A haircut that could be set above 1.0 would let the
  whole plan be talked upward later, so the test forbids it.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "scripts"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

from the_plan import (HAIRCUT, LOT, drawdown_budget,          # noqa: E402
                      income_forecast, lot_plan, minimum_capital, plan_return)


PRICES = pd.Series({"AAAA": 1000.0, "BBBB": 5000.0, "CCCC": 250.0,
                    "DDDD": 25000.0})


# --------------------------------------------------------------------------
# whole lots
# --------------------------------------------------------------------------
def test_lots_are_whole_and_never_overshoot_the_target():
    plan = lot_plan(PRICES, 4_000_000.0, list(PRICES.index))
    assert len(plan) == 4
    assert (plan["lots"] == plan["lots"].astype(int)).all()
    assert (plan["value"] <= plan["target"] + 1e-9).all()


def test_a_cheap_name_lands_almost_exactly_on_target():
    plan = lot_plan(PRICES, 4_000_000.0, ["CCCC"])
    r = plan.iloc[0]
    assert r["lots"] == int(4_000_000 // 25_000)
    assert abs(r["drift"]) < 0.01


def test_a_dear_name_on_a_small_book_drifts_badly_and_says_so():
    # DDDD costs 2.5m a lot; a 3m target buys one lot and 0.5m sits idle
    plan = lot_plan(PRICES, 12_000_000.0, list(PRICES.index))
    d = plan[plan["ticker"] == "DDDD"].iloc[0]
    assert d["lots"] == 1
    assert d["drift"] < -0.15


def test_a_name_too_dear_to_afford_gets_no_lots_rather_than_a_fraction():
    plan = lot_plan(PRICES, 1_000_000.0, ["DDDD"])
    assert plan.iloc[0]["lots"] == 0
    assert plan.iloc[0]["value"] == 0.0


def test_a_missing_price_drops_the_name_instead_of_guessing_one():
    plan = lot_plan(PRICES, 10_000_000.0, ["AAAA", "ZZZZ"])
    assert list(plan["ticker"]) == ["AAAA"]


def test_a_zero_price_is_refused():
    p = PRICES.copy()
    p["EEEE"] = 0.0
    plan = lot_plan(p, 10_000_000.0, ["EEEE", "AAAA"])
    assert "EEEE" not in list(plan["ticker"])


def test_an_empty_book_produces_an_empty_plan():
    assert lot_plan(PRICES, 1e9, []).empty


def test_a_bigger_book_makes_the_rounding_matter_less():
    small = lot_plan(PRICES, 12_000_000.0, list(PRICES.index))
    big = lot_plan(PRICES, 12_000_000_000.0, list(PRICES.index))
    assert big["drift"].abs().max() < small["drift"].abs().max()


def test_lot_size_is_the_idx_hundred():
    plan = lot_plan(pd.Series({"AAAA": 1000.0}), 1_000_000.0, ["AAAA"])
    assert plan.iloc[0]["value"] == plan.iloc[0]["lots"] * 1000.0 * LOT
    assert LOT == 100


# --------------------------------------------------------------------------
# how small is too small
# --------------------------------------------------------------------------
def test_minimum_capital_is_set_by_the_dearest_name():
    # at 10% tolerance the dearest position must buy nine lots, not one
    m = minimum_capital(PRICES, list(PRICES.index), tolerance=0.10)
    assert m == pytest.approx(25000.0 * LOT * 9 * 4)


def test_affording_one_lot_is_not_the_bar():
    """One lot of the dearest name is a 50% weight error, not a 0% one."""
    names = list(PRICES.index)
    one_lot_each = 25000.0 * LOT * len(names)
    assert minimum_capital(PRICES, names) > 5 * one_lot_each


def test_a_book_at_the_minimum_keeps_every_weight_inside_tolerance():
    names = list(PRICES.index)
    plan = lot_plan(PRICES, minimum_capital(PRICES, names, 0.10), names)
    assert (plan["lots"] >= 1).all()
    assert plan["drift"].abs().max() <= 0.10 + 1e-9


def test_a_book_well_under_the_minimum_breaks_the_tolerance():
    names = list(PRICES.index)
    plan = lot_plan(PRICES, minimum_capital(PRICES, names) / 4, names)
    assert plan["drift"].abs().max() > 0.10


def test_a_tighter_tolerance_demands_more_capital():
    names = list(PRICES.index)
    assert minimum_capital(PRICES, names, 0.05) \
        > minimum_capital(PRICES, names, 0.20)


def test_a_nonsensical_tolerance_is_refused():
    assert np.isnan(minimum_capital(PRICES, list(PRICES.index), 0.0))
    assert np.isnan(minimum_capital(PRICES, list(PRICES.index), 1.5))


def test_dropping_the_dearest_name_lowers_the_bar():
    names = list(PRICES.index)
    assert minimum_capital(PRICES, [n for n in names if n != "DDDD"]) \
        < minimum_capital(PRICES, names)


def test_minimum_capital_of_nothing_is_undefined():
    assert np.isnan(minimum_capital(PRICES, []))


# --------------------------------------------------------------------------
# the haircut — the plan must not be talkable upward
# --------------------------------------------------------------------------
def test_the_plan_is_below_what_was_measured():
    measured_edge, neutral = 0.05, 0.04
    assert plan_return(measured_edge, neutral) < neutral + measured_edge


def test_the_plan_is_above_the_neutral_book_when_the_edge_is_positive():
    assert plan_return(0.05, 0.04) > 0.04


def test_half_the_premium_is_exactly_half():
    assert plan_return(0.06, 0.04) == pytest.approx(0.04 + 0.03)


def test_the_default_haircut_cannot_flatter_the_measurement():
    assert 0.0 < HAIRCUT <= 1.0


def test_a_measured_edge_of_zero_plans_on_the_neutral_book_alone():
    assert plan_return(0.0, 0.043) == pytest.approx(0.043)


def test_a_negative_edge_is_carried_through_rather_than_floored_at_zero():
    """If the factor lost, the plan says so instead of rounding it up."""
    assert plan_return(-0.02, 0.04) < 0.04


# --------------------------------------------------------------------------
# income
# --------------------------------------------------------------------------
def test_income_is_the_yield_on_what_is_actually_held():
    book = pd.DataFrame({"ticker": ["AAAA", "BBBB"],
                         "value": [100e6, 200e6]})
    gross, net = income_forecast(book, {"AAAA": 0.10, "BBBB": 0.05}, tax=0.10)
    assert gross == pytest.approx(100e6 * 0.10 + 200e6 * 0.05)
    assert net == pytest.approx(gross * 0.9)


def test_a_name_with_no_known_yield_contributes_nothing():
    book = pd.DataFrame({"ticker": ["AAAA", "ZZZZ"], "value": [100e6, 100e6]})
    gross, _ = income_forecast(book, {"AAAA": 0.08})
    assert gross == pytest.approx(8e6)


def test_the_exemption_is_worth_the_whole_tax():
    book = pd.DataFrame({"ticker": ["AAAA"], "value": [1e9]})
    gross, net = income_forecast(book, {"AAAA": 0.07}, tax=0.10)
    exempt_gross, exempt_net = income_forecast(book, {"AAAA": 0.07}, tax=0.0)
    assert exempt_net - net == pytest.approx(gross * 0.10)
    assert exempt_gross == gross


# --------------------------------------------------------------------------
# the drawdown, in money
# --------------------------------------------------------------------------
def test_the_drawdown_is_reported_in_rupiah_not_only_percent():
    d = drawdown_budget(1e9, -0.56)
    assert d["trough"] == pytest.approx(440e6)
    assert d["lost"] == pytest.approx(560e6)
    assert d["peak"] == pytest.approx(1e9)


def test_a_book_that_never_fell_still_reports_a_peak():
    d = drawdown_budget(1e9, 0.0)
    assert d["lost"] == pytest.approx(0.0)
    assert d["trough"] == pytest.approx(d["peak"])


def test_a_total_loss_leaves_nothing():
    assert drawdown_budget(1e9, -1.0)["trough"] == pytest.approx(0.0)
