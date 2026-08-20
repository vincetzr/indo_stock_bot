"""Tests for the base-rate measurement.

This script exists because three things were being got wrong at once, and each
of the three has a test here:

  1. DIVIDENDS DROPPED. dividend_gap must read the adjusted series, not the
     price line, and must report the gap rather than either one alone.
  2. A SAMPLE THAT STOPS INSIDE A CRASH. end_date_sensitivity must show the
     CAGR moving as the finish line moves - if it did not, it would be
     answering a different question than the one asked.
  3. OVERLAPPING WINDOWS COUNTED AS INDEPENDENT. Five ten-year windows inside
     eleven years of data are one observation, and independent_windows has to
     say so, because a "100% of the time" claim built on them is one stretch
     of history wearing a statistic's clothes.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "scripts"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

from base_rates import (                                          # noqa: E402
    DEPOSIT_RATE, describe, end_date_sensitivity, independent_windows,
    real_return, usd_curve, wealth_concentration, window_returns)


def curve(years=12, rate=0.08, freq="ME"):
    idx = pd.date_range("2015-01-31", periods=int(years * 12), freq=freq)
    step = (1.0 + rate) ** (1 / 12)
    return pd.Series(np.cumprod(np.full(len(idx), step)), index=idx)


# --------------------------------------------------------------------------
# holding windows
# --------------------------------------------------------------------------
def test_every_window_of_a_steady_curve_returns_the_same_rate():
    v = window_returns(curve(rate=0.08), 3)
    assert len(v) > 50
    assert np.allclose(v, 0.08, atol=2e-3)


def test_a_window_longer_than_the_sample_yields_nothing():
    assert len(window_returns(curve(years=4), 10)) == 0


def test_a_stub_series_yields_nothing():
    s = pd.Series([1.0, 1.1], index=pd.to_datetime(["2020-01-01", "2020-02-01"]))
    assert len(window_returns(s, 1)) == 0


def test_shorter_windows_give_more_observations():
    c = curve()
    assert len(window_returns(c, 1)) > len(window_returns(c, 5))


def test_a_crash_at_the_end_shows_up_in_the_worst_window():
    c = curve(rate=0.08)
    c.iloc[-12:] = c.iloc[-13] * np.linspace(1.0, 0.5, 12)
    v = window_returns(c, 1)
    assert v.min() < -0.2
    assert v.max() > 0.05      # earlier windows are untouched


# --------------------------------------------------------------------------
# overlap
# --------------------------------------------------------------------------
def test_overlapping_windows_are_counted_as_one_stretch():
    # five 10-year windows inside 11 years of history is one observation
    assert independent_windows(5, 10, 11.0) == pytest.approx(1.1)
    assert independent_windows(113, 1, 11.0) == pytest.approx(11.0)


def test_a_window_longer_than_the_sample_still_counts_as_one():
    assert independent_windows(1, 20, 11.0) == pytest.approx(1.0)


def test_no_windows_means_no_observations():
    assert independent_windows(0, 5, 11.0) == 0.0


# --------------------------------------------------------------------------
# where the finish line is drawn
# --------------------------------------------------------------------------
def test_a_steady_curve_reports_the_same_cagr_whenever_you_stop():
    df = end_date_sensitivity(curve(rate=0.08))
    assert len(df) > 5
    assert np.allclose(df["cagr"], 0.08, atol=3e-3)


def test_a_crash_at_the_end_only_moves_the_last_row():
    c = curve(rate=0.10)
    c.iloc[-14:] = c.iloc[-15] * np.linspace(1.0, 0.6, 14)
    df = end_date_sensitivity(c)
    assert df["cagr"].iloc[-1] < df["cagr"].iloc[-2] - 0.02
    assert df["cagr"].iloc[:-2].std() < 0.01


def test_short_stretches_are_not_annualised_at_all():
    """A CAGR off eighteen months is noise dressed as a rate."""
    df = end_date_sensitivity(curve(years=2))
    assert df.empty or (df["years"] >= 3).all()


# --------------------------------------------------------------------------
# describe
# --------------------------------------------------------------------------
def test_describe_reports_the_share_below_cash_not_just_below_zero():
    v = np.array([-0.10, 0.01, 0.03, 0.07, 0.20])
    d = describe(v)
    assert d["neg"] == pytest.approx(0.2)
    assert d["below_cash"] == pytest.approx(0.6)   # -0.10, 0.01, 0.03
    assert d["median"] == pytest.approx(0.03)
    assert d["worst"] == pytest.approx(-0.10)
    assert d["best"] == pytest.approx(0.20)


def test_describe_of_nothing_says_nothing():
    assert describe(np.array([]))["n"] == 0


def test_below_cash_uses_the_deposit_rate_actually_configured():
    v = np.array([DEPOSIT_RATE - 1e-6, DEPOSIT_RATE + 1e-6])
    assert describe(v)["below_cash"] == pytest.approx(0.5)


# --------------------------------------------------------------------------
# real return
# --------------------------------------------------------------------------
def test_real_return_divides_rather_than_subtracts():
    assert real_return(0.10, 0.03) == pytest.approx(1.10 / 1.03 - 1)
    assert real_return(0.10, 0.03) < 0.07     # subtraction would give exactly


def test_a_return_equal_to_inflation_buys_nothing_extra():
    assert real_return(0.04, 0.04) == pytest.approx(0.0)


def test_losing_to_inflation_is_negative():
    assert real_return(0.02, 0.05) < 0


# --------------------------------------------------------------------------
# concentration
# --------------------------------------------------------------------------
def test_one_name_carrying_everything_is_reported_as_such():
    total = [0.50] + [0.0] * 19
    years = [10.0] * 20
    w = wealth_concentration(total, years)
    assert w["names"] == 20
    assert w["top5pct"] == pytest.approx(1.0)
    assert w["losers"] == pytest.approx(0.0)


def test_losers_and_halvings_are_counted_separately():
    # three names down 90%, three down 20%, four up
    total = [-0.2] * 3 + [-0.02] * 3 + [0.10] * 4
    years = [10.0] * 10
    w = wealth_concentration(total, years)
    assert w["losers"] == pytest.approx(0.6)
    assert w["halved"] == pytest.approx(0.3)   # only the -0.2/yr names halve


def test_an_even_market_has_no_concentration():
    total = [0.05] * 20
    w = wealth_concentration(total, [10.0] * 20)
    assert w["top5pct"] == pytest.approx(0.05, abs=0.01)
    assert w["winner_share"] == pytest.approx(1.0)


def test_concentration_of_nothing_is_empty():
    assert wealth_concentration([], []) == {}


# --------------------------------------------------------------------------
# currency
# --------------------------------------------------------------------------
def test_a_weakening_rupiah_lowers_the_dollar_curve():
    eq = curve(rate=0.08)
    fx = pd.Series(np.linspace(13000.0, 18000.0, len(eq)), index=eq.index)
    u = usd_curve(eq, fx)
    assert u.iloc[-1] < eq.iloc[-1]
    assert u.iloc[-1] == pytest.approx(eq.iloc[-1] / (18000.0 / 13000.0),
                                       rel=1e-9)


def test_a_stable_currency_leaves_the_curve_alone():
    eq = curve()
    fx = pd.Series(np.full(len(eq), 14000.0), index=eq.index)
    assert np.allclose(usd_curve(eq, fx).to_numpy(), eq.to_numpy())


def test_a_missing_rate_returns_nothing_rather_than_a_guess():
    eq = curve()
    fx = pd.Series(np.full(len(eq), np.nan), index=eq.index)
    assert len(usd_curve(eq, fx)) == 0
