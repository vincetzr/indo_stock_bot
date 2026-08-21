"""Tests for corporate-action adjustment, built around the rights-issue trap.

CLAUDE.md §5 calls rights issues the trap and asks for extreme-dilution cases
as fixtures: "if the adjustment handles those, it handles most things." So the
fixtures here go to 1-for-1 at a 90% discount and 10-for-1 at a 95% discount -
events that take 45% and 86% off the quote while costing a participating holder
nothing at all.

THE ACCEPTANCE TEST, and the reason this module exists:

    a correctly adjusted series shows NO return on the ex-date for a holder
    whose wealth did not change, and STILL SHOWS a real move when there was
    one.

Both halves matter. An adjustment that flattens everything would pass the first
and destroy the data. ``test_a_real_fall_on_the_ex_date_survives_adjustment``
is the one that stops that.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

from idxbot.spine.corporate_actions import (Action,                # noqa: E402
                                            MissingTerms, adjust,
                                            adjustment_factor, describe,
                                            dilution, ex_date_return,
                                            from_records, terp)

EX = pd.Timestamp("2024-06-03")


def frame(pre_close, post_close, n=6):
    """A flat series that steps to ``post_close`` on the ex-date."""
    days = pd.bdate_range("2024-05-20", periods=n * 2)
    closes = [pre_close] * n + [post_close] * n
    return pd.DataFrame({
        "date": days, "open": closes, "high": closes, "low": closes,
        "close": closes, "volume": [1000.0] * (n * 2)})


def realistic(pre, post):
    """A frame whose ex-date is EX exactly."""
    pre_days = pd.bdate_range(end=EX - pd.Timedelta(days=1), periods=5)
    post_days = pd.bdate_range(start=EX, periods=5)
    days = list(pre_days) + list(post_days)
    closes = [pre] * 5 + [post] * 5
    return pd.DataFrame({
        "date": days, "open": closes, "high": closes, "low": closes,
        "close": closes, "volume": [1000.0] * 10})


# --------------------------------------------------------------------------
# TERP arithmetic
# --------------------------------------------------------------------------
def test_terp_of_a_one_for_one_at_half_price_is_three_quarters():
    assert terp(1000.0, new=1, held=1, subscription=500.0) == pytest.approx(750.0)


def test_terp_at_the_market_price_changes_nothing():
    """Subscribing at the market price is not dilutive."""
    assert terp(1000.0, new=1, held=1, subscription=1000.0) == pytest.approx(1000.0)


def test_terp_falls_as_the_discount_deepens():
    prev = 1e9
    for s in (900.0, 500.0, 100.0, 50.0):
        t = terp(1000.0, new=1, held=1, subscription=s)
        assert t < prev
        prev = t


def test_terp_falls_as_the_issue_gets_bigger():
    prev = 1e9
    for n in (1, 2, 5, 10):
        t = terp(1000.0, new=n, held=1, subscription=50.0)
        assert t < prev
        prev = t


def test_terp_rejects_nonsense_terms():
    with pytest.raises(ValueError):
        terp(1000.0, new=1, held=0, subscription=500.0)
    with pytest.raises(ValueError):
        terp(0.0, new=1, held=1, subscription=500.0)


# --------------------------------------------------------------------------
# the extreme-dilution fixtures §5 asks for
# --------------------------------------------------------------------------
def test_a_one_for_one_at_a_ninety_percent_discount_removes_45pc_of_the_quote():
    a = Action("TEST", EX, "rights", new=1, held=1, subscription=100.0)
    assert dilution(a, cum_price=1000.0) == pytest.approx(0.45)


def test_a_ten_for_one_at_a_ninety_five_percent_discount_removes_86pc():
    a = Action("TEST", EX, "rights", new=10, held=1, subscription=50.0)
    assert dilution(a, cum_price=1000.0) == pytest.approx(0.8636, abs=1e-3)


def test_extreme_dilution_leaves_the_participating_holder_flat():
    """THE ACCEPTANCE TEST. A crash of 86% that cost nobody anything."""
    a = Action("TEST", EX, "rights", new=10, held=1, subscription=50.0)
    cum, ex_price = 1000.0, terp(1000.0, 10, 1, 50.0)
    raw = realistic(cum, ex_price)
    assert ex_date_return(raw, EX) < -0.85          # the fake crash
    fixed = adjust(raw, [a])
    assert abs(ex_date_return(fixed, EX)) < 1e-9    # and it is gone


def test_a_real_fall_on_the_ex_date_survives_adjustment():
    """The other half. An adjustment that flattens everything is useless."""
    a = Action("TEST", EX, "rights", new=1, held=1, subscription=100.0)
    cum = 1000.0
    fair = terp(cum, 1, 1, 100.0)
    raw = realistic(cum, fair * 0.90)               # 10% worse than fair
    fixed = adjust(raw, [a])
    assert ex_date_return(fixed, EX) == pytest.approx(-0.10, abs=1e-9)


# --------------------------------------------------------------------------
# splits, bonuses, dividends
# --------------------------------------------------------------------------
def test_a_two_for_one_split_halves_prior_prices():
    a = Action("TEST", EX, "split", ratio=2.0)
    assert adjustment_factor(a) == pytest.approx(0.5)


def test_a_reverse_split_raises_prior_prices():
    a = Action("TEST", EX, "reverse_split", ratio=4.0)
    assert adjustment_factor(a) == pytest.approx(4.0)


def test_sccos_one_for_four_split_removes_the_fake_crash():
    """SCCO's real 2024-02-01 event: 9,975 to 2,494 prints as -75%."""
    ex = pd.Timestamp("2024-02-01")
    a = Action("SCCO", ex, "split", ratio=4.0)
    pre = pd.bdate_range(end=ex - pd.Timedelta(days=1), periods=5)
    post = pd.bdate_range(start=ex, periods=5)
    raw = pd.DataFrame({
        "date": list(pre) + list(post),
        "open": [9975.0] * 5 + [2493.75] * 5,
        "high": [9975.0] * 5 + [2493.75] * 5,
        "low": [9975.0] * 5 + [2493.75] * 5,
        "close": [9975.0] * 5 + [2493.75] * 5,
        "volume": [1000.0] * 10})
    assert ex_date_return(raw, ex) == pytest.approx(-0.75)
    assert abs(ex_date_return(adjust(raw, [a]), ex)) < 1e-9


def test_a_bonus_issue_dilutes_without_any_cash():
    a = Action("TEST", EX, "bonus", new=1, held=2)
    assert adjustment_factor(a) == pytest.approx(2 / 3)


def test_a_dividend_removes_exactly_its_own_amount():
    a = Action("TEST", EX, "dividend", amount=50.0)
    assert adjustment_factor(a, cum_price=1000.0) == pytest.approx(0.95)


def test_a_zero_dividend_changes_nothing():
    a = Action("TEST", EX, "dividend", amount=0.0)
    assert adjustment_factor(a, cum_price=1000.0) == pytest.approx(1.0)


# --------------------------------------------------------------------------
# refusal to guess
# --------------------------------------------------------------------------
def test_a_rights_adjustment_without_the_cum_price_refuses():
    a = Action("TEST", EX, "rights", new=1, held=1, subscription=100.0)
    with pytest.raises(MissingTerms):
        adjustment_factor(a, cum_price=None)


def test_a_rights_adjustment_without_the_terms_refuses():
    """The terms come from an announcement, never from the tape."""
    a = Action("TEST", EX, "rights", new=0, held=1, subscription=0.0)
    with pytest.raises(MissingTerms, match="announcement"):
        adjustment_factor(a, cum_price=1000.0)


def test_the_refusal_explains_why_deriving_it_would_be_circular():
    a = Action("TEST", EX, "rights", new=1, held=1, subscription=100.0)
    with pytest.raises(MissingTerms, match="explain the drop with itself"):
        adjustment_factor(a, cum_price=None)


def test_an_unknown_kind_is_rejected_at_construction():
    with pytest.raises(ValueError):
        Action("TEST", EX, "spin_off")


def test_a_split_with_no_ratio_refuses():
    with pytest.raises(MissingTerms):
        adjustment_factor(Action("TEST", EX, "split", ratio=0.0))


# --------------------------------------------------------------------------
# applying to a series
# --------------------------------------------------------------------------
def test_only_prices_before_the_ex_date_move():
    a = Action("TEST", EX, "split", ratio=2.0)
    raw = realistic(1000.0, 500.0)
    fixed = adjust(raw, [a])
    assert fixed[fixed["date"] >= EX]["close"].iloc[0] == pytest.approx(500.0)
    assert fixed[fixed["date"] < EX]["close"].iloc[0] == pytest.approx(500.0)


def test_traded_value_is_preserved_so_liquidity_is_not_corrupted():
    a = Action("TEST", EX, "split", ratio=2.0)
    raw = realistic(1000.0, 500.0)
    fixed = adjust(raw, [a])
    before_raw = raw[raw["date"] < EX]
    before_fix = fixed[fixed["date"] < EX]
    assert ((before_fix["close"] * before_fix["volume"]).sum()
            == pytest.approx((before_raw["close"] * before_raw["volume"]).sum()))


def test_two_actions_compound_on_the_oldest_bars_only():
    """A split then a rights issue: both hit the oldest bars, one hits between."""
    ex1 = pd.Timestamp("2024-03-01")
    ex2 = pd.Timestamp("2024-06-03")
    days = list(pd.bdate_range(end=ex1 - pd.Timedelta(days=1), periods=3)) \
        + list(pd.bdate_range(start=ex1, periods=3)) \
        + list(pd.bdate_range(start=ex2, periods=3))
    closes = [1000.0] * 3 + [500.0] * 3 + [250.0] * 3
    raw = pd.DataFrame({"date": days, "open": closes, "high": closes,
                        "low": closes, "close": closes,
                        "volume": [1000.0] * 9})
    acts = [Action("T", ex1, "split", ratio=2.0),
            Action("T", ex2, "split", ratio=2.0)]
    fixed = adjust(raw, acts)
    c = fixed["close"].to_numpy()
    assert c[0] == pytest.approx(250.0)     # both applied
    assert c[3] == pytest.approx(250.0)     # one applied
    assert c[6] == pytest.approx(250.0)     # none


def test_an_action_before_the_series_starts_is_a_no_op():
    a = Action("TEST", pd.Timestamp("2020-01-01"), "split", ratio=2.0)
    raw = realistic(1000.0, 1000.0)
    assert adjust(raw, [a])["close"].equals(raw["close"])


def test_adjusting_nothing_returns_nothing():
    assert adjust(pd.DataFrame(), [Action("T", EX, "split", ratio=2.0)]).empty


def test_no_actions_leaves_the_frame_alone():
    raw = realistic(1000.0, 500.0)
    assert adjust(raw, [])["close"].equals(raw["close"])


# --------------------------------------------------------------------------
# plumbing
# --------------------------------------------------------------------------
def test_records_round_trip_into_actions():
    acts = from_records([{"ticker": "scco", "ex_date": "2024-02-01",
                          "kind": "split", "ratio": 4}])
    assert acts[0].ticker == "SCCO" and acts[0].ratio == 4.0


def test_describe_is_checkable_against_an_announcement():
    a = Action("TEST", EX, "rights", new=10, held=1, subscription=50.0)
    text = describe(a, cum_price=1000.0)
    assert "10 new per 1 at 50" in text and "86" in text


def test_describe_of_an_unpriceable_action_says_so_rather_than_crashing():
    a = Action("TEST", EX, "rights", new=1, held=1, subscription=100.0)
    assert "needs the cum-rights price" in describe(a, cum_price=None)
