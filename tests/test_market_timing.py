"""Tests for the market-level timer.

The market timer is the one thing in this line of work that survived a
robustness sweep, so the claims it rests on need to hold:

  * breadth is a trailing statistic - truncating the panel must not change any
    earlier value, or the regime call is reading its own future;
  * breadth counts what it says it counts;
  * the moving-average timer is in exactly when price is above the average;
  * a timer that is always on reproduces buy-and-hold, which is what makes the
    'vs hold' column meaningful.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "scripts"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

from market_timing import breadth, score_timer, timer_breadth, timer_ma  # noqa: E402


def _panel(series: dict) -> dict:
    idx = pd.bdate_range("2010-01-01", periods=len(next(iter(series.values()))))
    close = pd.DataFrame(series, index=idx)
    return {"close": close}


def test_breadth_counts_names_above_their_own_average():
    n = 400
    rising = 1000.0 * np.cumprod(np.full(n, 1.002))
    falling = 1000.0 * np.cumprod(np.full(n, 0.998))
    b = breadth(_panel({"UP": rising, "DOWN": falling}))
    assert b["br_ma200"].iloc[-1] == pytest.approx(0.5)


def test_breadth_is_one_when_everything_trends_up():
    n = 400
    up = 1000.0 * np.cumprod(np.full(n, 1.002))
    b = breadth(_panel({"A": up, "B": up * 2, "C": up * 3}))
    assert b["br_ma200"].iloc[-1] == pytest.approx(1.0)
    assert b["br_nearhi"].iloc[-1] == pytest.approx(1.0)
    assert b["br_dd"].iloc[-1] == pytest.approx(0.0, abs=1e-9)


def test_breadth_drawdown_is_negative_after_a_fall():
    n = 400
    px = np.concatenate([np.linspace(1000, 2000, 300), np.linspace(2000, 1000, 100)])
    b = breadth(_panel({"A": px}))
    assert b["br_dd"].iloc[-1] < -0.4
    assert b["br_nearhi"].iloc[-1] == pytest.approx(0.0)


@pytest.mark.parametrize("cut", [250, 320])
def test_breadth_has_no_look_ahead(cut):
    rng = np.random.default_rng(21)
    n = 400
    series = {f"N{i}": 1000.0 * np.cumprod(1.0 + rng.normal(0.0004, 0.02, n))
              for i in range(6)}
    full = breadth(_panel(series))
    part = breadth({"close": _panel(series)["close"].iloc[:cut]})
    assert np.allclose(full.iloc[:cut].to_numpy(float),
                       part.to_numpy(float), equal_nan=True)


def test_ma_timer_is_in_exactly_when_above():
    idx = pd.date_range("2010-01-01", periods=60, freq="W-FRI")
    # flat, then a steady climb: a flat series sits exactly ON its own average
    # and is therefore NOT above it, which is the boundary worth pinning down
    w = pd.Series(np.concatenate([np.full(30, 100.0),
                                  100.0 * np.cumprod(np.full(30, 1.03))]), index=idx)
    st = timer_ma(w, 10)
    assert st[:10].sum() == 0          # warmup: no average yet
    assert st[20] == 0                 # flat: equal to its average, not above
    assert st[-1] == 1                 # climbing: above it


def test_breadth_timer_thresholds():
    f = pd.DataFrame({"br_ma200": [0.2, 0.45, 0.55, 0.8, np.nan]})
    assert timer_breadth(f, "br_ma200", 0.50).tolist() == [0, 0, 1, 1, 0]


def test_always_on_reproduces_buy_and_hold():
    """The 'vs hold' column is only meaningful if this holds exactly."""
    idx = pd.date_range("2010-01-01", periods=200, freq="W-FRI")
    rng = np.random.default_rng(3)
    w = pd.Series(1000.0 * np.cumprod(1.0 + rng.normal(0.001, 0.02, 200)), index=idx)
    s = score_timer(w, np.ones(200, dtype=np.int8), 0, 0.20)
    # NOT exactly buy-and-hold: the signal is acted on with a two-bar delay, so
    # the first bar's return is never earned. Pinning the exact relationship
    # here is what stops that delay from being quietly dropped later.
    px = w.to_numpy(float)
    assert s["growth"] == pytest.approx(px[-1] / px[1] * (1 - 0.003), rel=1e-9)
    assert s["time_in"] == 1.0
    assert s["trades"] == 1


def test_sitting_out_entirely_earns_nothing():
    idx = pd.date_range("2010-01-01", periods=100, freq="W-FRI")
    w = pd.Series(1000.0 * np.cumprod(np.full(100, 1.01)), index=idx)
    s = score_timer(w, np.zeros(100, dtype=np.int8), 0, 0.20)
    assert s["growth"] == pytest.approx(1.0)
    assert s["cagr"] == pytest.approx(0.0)
    assert s["bh_cagr"] > 0
