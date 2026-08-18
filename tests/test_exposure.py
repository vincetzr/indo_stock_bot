"""Tests for the exposure overlay.

The overlay is the one thing in this project that beats always-on on both return
and drawdown, and it does so only because idle cash earns interest. That makes
the financing arithmetic load-bearing, so it is what gets tested:

  * full exposure must reproduce the underlying book exactly;
  * cash held out of the market must earn the deposit rate;
  * borrowed exposure must pay margin interest - a leveraged backtest that
    ignores financing is a fiction;
  * changing exposure must cost something;
  * the volatility estimate must be causal, or the rule is reading its own future.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "scripts"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

from exposure import apply_exposure, vol_exposure   # noqa: E402


def _curve(n=300, step=1.001):
    idx = pd.bdate_range("2015-01-01", periods=n)
    return pd.Series(np.cumprod(np.full(n, step)), index=idx)


def test_full_exposure_reproduces_the_book():
    eq = _curve()
    e = pd.Series(1.0, index=eq.index)
    out, turn = apply_exposure(eq, e, cost=0.01, cash=0.05, margin=0.09)
    # the overlay curve is indexed to 1.0 at the first bar, so it reproduces the
    # book's GROWTH rather than its level
    assert out.iloc[-1] == pytest.approx(eq.iloc[-1] / eq.iloc[0], rel=1e-9)
    assert turn == pytest.approx(0.0)


def test_cash_earns_the_deposit_rate():
    """Sitting entirely in cash must compound at the deposit rate, not zero.

    This is the whole reason the overlay is nearly free in Indonesia.
    """
    eq = _curve(n=253)
    e = pd.Series(0.0, index=eq.index)
    out, _ = apply_exposure(eq, e, cost=0.0, cash=0.05, margin=0.09)
    # 252 trading days of 5%/252 per day
    assert out.iloc[-1] == pytest.approx((1 + 0.05 / 252) ** 252, rel=1e-6)


def test_zero_cash_rate_earns_nothing():
    eq = _curve(n=253)
    e = pd.Series(0.0, index=eq.index)
    out, _ = apply_exposure(eq, e, cost=0.0, cash=0.0)
    assert out.iloc[-1] == pytest.approx(1.0)


def test_leverage_pays_margin_interest():
    """A flat book held at 2x must LOSE the financing cost, not stay flat."""
    idx = pd.bdate_range("2015-01-01", periods=253)
    eq = pd.Series(1.0, index=idx)          # book goes nowhere
    e = pd.Series(2.0, index=idx)
    out, _ = apply_exposure(eq, e, cost=0.0, margin=0.09, cash=0.0)
    assert out.iloc[-1] < 1.0
    assert out.iloc[-1] == pytest.approx((1 - 0.09 / 252) ** 252, rel=1e-6)


def test_changing_exposure_costs_money():
    idx = pd.bdate_range("2015-01-01", periods=10)
    eq = pd.Series(1.0, index=idx)
    steady = pd.Series(1.0, index=idx)
    flips = pd.Series([1.0, 0.0] * 5, index=idx)
    a, ta = apply_exposure(eq, steady, cost=0.01, cash=0.0)
    b, tb = apply_exposure(eq, flips, cost=0.01, cash=0.0)
    assert tb > ta
    assert b.iloc[-1] < a.iloc[-1]


def test_turnover_counts_absolute_exposure_change():
    idx = pd.bdate_range("2015-01-01", periods=4)
    eq = pd.Series(1.0, index=idx)
    e = pd.Series([1.0, 0.5, 0.5, 1.0], index=idx)
    _, turn = apply_exposure(eq, e, cost=0.0)
    assert turn == pytest.approx(1.0)       # 0.5 down then 0.5 back up


def test_vol_exposure_is_capped_and_non_negative():
    eq = _curve(n=400, step=1.0005)         # almost no volatility
    e = vol_exposure(eq, target=0.20, lookback=60, max_lev=1.5)
    assert e.max() <= 1.5 + 1e-12
    assert e.min() >= 0.0


def test_vol_exposure_falls_when_volatility_rises():
    rng = np.random.default_rng(5)
    idx = pd.bdate_range("2015-01-01", periods=600)
    calm = rng.normal(0.0004, 0.004, 300)
    wild = rng.normal(0.0004, 0.040, 300)
    eq = pd.Series(np.cumprod(1 + np.concatenate([calm, wild])), index=idx)
    e = vol_exposure(eq, target=0.20, lookback=60, max_lev=1.0)
    assert e.iloc[250] > e.iloc[-1]         # calm period held more than the wild one


@pytest.mark.parametrize("cut", [200, 350])
def test_vol_exposure_has_no_look_ahead(cut):
    rng = np.random.default_rng(9)
    idx = pd.bdate_range("2015-01-01", periods=500)
    eq = pd.Series(np.cumprod(1 + rng.normal(0.0005, 0.02, 500)), index=idx)
    full = vol_exposure(eq, 0.20, 60, 1.0)
    part = vol_exposure(eq.iloc[:cut], 0.20, 60, 1.0)
    assert np.allclose(full.iloc[:cut].to_numpy(), part.to_numpy(), equal_nan=True)
