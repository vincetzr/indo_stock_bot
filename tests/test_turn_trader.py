"""Tests for the bounded-lag reversal filter.

The filter's entire claim is that its lag is *bounded by its own threshold*,
unlike a moving average whose lag depends on the window and on the path. If that
claim is wrong the whole result built on it is wrong, so it is what gets tested:

  * the state at bar i depends only on bars <= i (no look-ahead, checked by
    truncation - the classic test, since a rule that peeks changes its past
    answers when the future is removed);
  * a flip happens exactly when the threshold is crossed, never earlier and
    never later;
  * the equity curve acts on the signal with a delay, so no bar is traded at a
    price used to generate its own signal;
  * costs only ever reduce the result;
  * a flat series never trades, and a monotone series trades once.

The capture diagnostics are tested too, because Result 69 turned on them: if
``up_fraction`` were computed wrongly the conclusion would invert.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "scripts"))

from swing_accuracy import zigzag, legs                                # noqa: E402
from turn_trader import (capture, clean_weekly, reversal_state, run,    # noqa: E402
                         score, vol_reversal_state, DAILY_CAP)


# --------------------------------------------------------------------------- #
# the state machine
# --------------------------------------------------------------------------- #
def test_flat_series_never_trades():
    px = np.full(200, 1000.0)
    st = reversal_state(px, 0.10, 0.10)
    assert st.sum() == 0                      # never buys: nothing ever rises 10%
    _, trades = run(px, st)
    assert trades == 0


def test_monotone_rise_buys_once_and_holds():
    px = 1000.0 * np.cumprod(np.full(200, 1.01))
    st = reversal_state(px, 0.10, 0.10)
    assert st[-1] == 1
    # exactly one transition from flat to long, and no transition back
    assert int(np.abs(np.diff(st)).sum()) == 1


def test_entry_fires_exactly_at_the_threshold():
    # up 9.9% then over 10%: the flip must land on the bar that crosses, not before
    px = np.array([100.0, 105.0, 109.9, 110.0, 111.0])
    st = reversal_state(px, 0.10, 0.10)
    assert st.tolist() == [0, 0, 0, 1, 1]


def test_exit_fires_exactly_at_the_threshold():
    px = np.array([100.0, 120.0, 150.0, 136.0, 135.0, 130.0])
    st = reversal_state(px, 0.10, 0.10)
    # long from the bar that is 10% above the low (120 >= 100*1.1)
    assert st[1] == 1
    # high is 150; 10% below is 135. 136 is not enough, 135 is.
    assert st[3] == 1
    assert st[4] == 0


def test_extreme_resets_on_flip_so_lag_cannot_compound():
    """After a flip the running extreme restarts at the flip price.

    Without this the filter would measure the next leg from a stale extreme and
    its lag would no longer be bounded by the threshold - the exact failure it
    exists to avoid.
    """
    px = np.array([100.0, 90.0, 99.0, 100.0, 89.0])
    st = reversal_state(px, 0.10, 0.10)
    # low 90 -> buy at 99 (90*1.1). Then the high since entry is 100, and 89 is
    # more than 10% below it, so it must exit on the last bar.
    assert st.tolist() == [0, 0, 1, 1, 0]


@pytest.mark.parametrize("cut", [50, 100, 175])
def test_no_look_ahead_under_truncation(cut):
    """Removing the future must not change any past state. This is the test."""
    rng = np.random.default_rng(11)
    px = 1000.0 * np.cumprod(1.0 + rng.normal(0.001, 0.03, 250))
    full = reversal_state(px, 0.12, 0.18)
    part = reversal_state(px[:cut], 0.12, 0.18)
    assert np.array_equal(full[:cut], part)


def test_thresholds_are_independent():
    """A wider exit must never exit sooner than a narrower one."""
    rng = np.random.default_rng(3)
    px = 1000.0 * np.cumprod(1.0 + rng.normal(0.0, 0.04, 400))
    tight = reversal_state(px, 0.10, 0.10)
    loose = reversal_state(px, 0.10, 0.25)
    # a looser exit spends at least as much time long
    assert loose.mean() >= tight.mean()


# --------------------------------------------------------------------------- #
# execution
# --------------------------------------------------------------------------- #
def test_signal_is_acted_on_with_a_delay():
    """The bar that generates a signal must not be traded at its own price."""
    px = np.array([100.0, 100.0, 100.0, 130.0, 130.0, 130.0])
    st = reversal_state(px, 0.10, 0.10)
    assert st[3] == 1                      # the jump bar is where the signal fires
    eq, _ = run(px, st, cost=0.0)
    # it must NOT have captured the jump from bar 2 to bar 3
    assert eq[3] == pytest.approx(1.0)


def test_costs_only_reduce():
    rng = np.random.default_rng(7)
    px = 1000.0 * np.cumprod(1.0 + rng.normal(0.001, 0.03, 500))
    st = reversal_state(px, 0.12, 0.12)
    free, n_free = run(px, st, cost=0.0)
    paid, n_paid = run(px, st, cost=0.02)
    assert n_free == n_paid
    assert paid[-1] < free[-1]


def test_holding_forever_reproduces_buy_and_hold():
    rng = np.random.default_rng(5)
    px = 1000.0 * np.cumprod(1.0 + rng.normal(0.0, 0.02, 300))
    always = np.ones(len(px), dtype=np.int8)
    eq, trades = run(px, always, cost=0.0)
    # one entry, then held: from bar 2 onward the curve tracks the price exactly
    assert trades == 1
    assert eq[-1] == pytest.approx(px[-1] / px[1])


def test_score_reports_buy_and_hold_consistently():
    idx = pd.date_range("2010-01-01", periods=300, freq="W-FRI")
    px = 1000.0 * np.cumprod(1.0 + np.full(300, 0.002))
    s = score(px, idx, 0.10, 0.10)
    assert s["bh_growth"] == pytest.approx(px[-1] / px[0])
    assert s["excess"] == pytest.approx(s["cagr"] - s["bh_cagr"], abs=1e-12)


# --------------------------------------------------------------------------- #
# capture diagnostics — Result 69 rests on these
# --------------------------------------------------------------------------- #
def test_perfect_capture_is_one():
    """A filter that is long for exactly the up leg captures all of it."""
    px = np.array([100.0, 100.0, 200.0, 200.0])
    st = np.array([1, 1, 1, 1], dtype=np.int8)
    lg = [(1, 2, 1.0)]
    c = capture(px, st, lg, cost=0.0)
    assert c["up_fraction"] == pytest.approx(1.0)


def test_sitting_out_an_up_leg_captures_nothing():
    px = np.array([100.0, 100.0, 200.0, 200.0])
    st = np.zeros(4, dtype=np.int8)
    c = capture(px, st, [(1, 2, 1.0)], cost=0.0)
    assert c["up_captured"] == pytest.approx(0.0)
    assert c["dn_absorbed"] != c["dn_absorbed"]        # NaN: there is no down leg


def test_direction_accuracy_and_capture_can_disagree():
    """The whole point of Result 69: right about direction, wrong about size.

    The rule is long for four of the five bars of the leg - 'accurate' by any
    directional scoring - but the two-bar execution delay means it only ever
    holds the tail of the move, and banks a quarter of it.
    """
    px = np.array([100.0, 100.0, 100.0, 120.0, 160.0,
                   200.0, 200.0, 200.0, 200.0, 200.0])
    st = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1], dtype=np.int8)
    lg = [(1, 8, 1.0)]                                 # the leg doubles
    c = capture(px, st, lg, cost=0.0)
    assert c["direction_acc"] == 1.0                   # long for 5 of the 7 leg bars
    assert c["up_fraction"] == pytest.approx(0.25)     # banked a quarter of it


def test_flips_are_counted_within_the_leg():
    px = np.linspace(100, 200, 11)
    st = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0], dtype=np.int8)
    c = capture(px, st, [(0, 10, 1.0)], cost=0.0)
    assert c["flips"] == 10


# --------------------------------------------------------------------------- #
# data hygiene
# --------------------------------------------------------------------------- #
def test_clean_weekly_caps_impossible_prints():
    """A 900% print inside a week must not survive into the weekly series."""
    dates = pd.bdate_range("2015-01-01", periods=1200)
    close = pd.Series(1000.0, index=dates)
    close.iloc[600] = 10000.0                       # a corrupt print
    close.iloc[601] = 1000.0
    df = pd.DataFrame({"date": dates, "adj_close": close.to_numpy(),
                       "close": close.to_numpy()})
    w = clean_weekly(df)
    assert w is not None
    r = w.pct_change().dropna()
    # weekly moves are built from capped daily moves, so nothing near 900% remains
    assert r.abs().max() < (1 + DAILY_CAP) ** 5 - 1


def test_clean_weekly_rejects_short_history():
    dates = pd.bdate_range("2024-01-01", periods=120)
    df = pd.DataFrame({"date": dates, "adj_close": np.linspace(100, 200, 120),
                       "close": np.linspace(100, 200, 120)})
    assert clean_weekly(df) is None


# --------------------------------------------------------------------------- #
# the causal filter against the hindsight zigzag
# --------------------------------------------------------------------------- #
def test_lag_is_bounded_by_the_threshold():
    """The claim the whole result rests on, stated as an assertion.

    On a clean V the filter must turn long on the *first* bar that is ``entry``
    above the trough and flat on the *first* bar that is ``exit`` below the peak
    - never earlier (that would be look-ahead) and never later (that would make
    the lag unbounded, which is the moving average's failure).
    """
    px = np.array([100.0, 80.0, 60.0, 50.0, 55.0, 70.0,
                   100.0, 150.0, 200.0, 190.0, 180.0, 120.0])
    st = reversal_state(px, 0.10, 0.10)
    #                    trough 50 -> long at 55 = 50*1.10
    #                    peak  200 -> flat at 180 = 200*0.90
    assert st.tolist() == [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0]


def test_capture_can_exceed_the_leg_when_the_filter_trades_inside_it():
    """``up_fraction`` is not capped at 1, and the tests must not assume it is.

    A zigzag leg is defined by its endpoints. A filter that sells a dip inside
    the leg and buys back lower earns more than the endpoints imply. That is a
    real outcome, not a bug, and Result 70's numbers have to be read knowing it.
    """
    px = np.array([100.0, 100.0, 100.0, 200.0, 100.0, 200.0, 200.0])
    st = np.ones(len(px), dtype=np.int8)
    lg = [(1, 6, 1.0)]
    c = capture(px, st, lg, cost=0.0)
    assert c["up_fraction"] == pytest.approx(1.0)      # always long: exactly the leg
    # now sit out the dip in the middle - the signal fires two bars ahead of the
    # bar it governs - and the same leg pays four times its endpoint move
    st2 = np.array([1, 1, 0, 1, 1, 1, 1], dtype=np.int8)
    assert capture(px, st2, lg, cost=0.0)["up_captured"] == pytest.approx(3.0)


def test_filter_scores_a_real_series_without_error():
    """End to end on a trending random walk: the diagnostics must be finite."""
    rng = np.random.default_rng(19)
    drift = np.repeat(rng.choice([0.006, -0.006], 12), 60)
    px = 1000.0 * np.cumprod(1.0 + drift + rng.normal(0, 0.02, len(drift)))
    lg = legs(px, zigzag(px, 0.20))
    c = capture(px, reversal_state(px, 0.12, 0.12), lg)
    assert c["legs"] == len(lg)
    assert 0.0 < c["direction_acc"] <= 1.0
    assert np.isfinite(c["up_fraction"])


# --------------------------------------------------------------------------- #
# volatility-scaled thresholds
# --------------------------------------------------------------------------- #
def test_vol_filter_is_flat_before_it_can_measure_volatility():
    """No sigma, no trade. Guessing during warmup would be a free parameter."""
    px = 1000.0 * np.cumprod(np.full(60, 1.02))
    st = vol_reversal_state(px, 2.0, 2.0, window=52)
    assert st[:52].sum() == 0


@pytest.mark.parametrize("cut", [80, 150, 260])
def test_vol_filter_has_no_look_ahead(cut):
    """The sigma used at bar i must come from returns strictly before it."""
    rng = np.random.default_rng(23)
    px = 1000.0 * np.cumprod(1.0 + rng.normal(0.001, 0.03, 300))
    full = vol_reversal_state(px, 2.0, 3.0, window=52)
    part = vol_reversal_state(px[:cut], 2.0, 3.0, window=52)
    assert np.array_equal(full[:cut], part)


def test_vol_thresholds_are_clamped():
    """A dead name must not produce a threshold so small it trades every bar."""
    # 52 weeks of nearly zero movement, then a small wiggle
    px = np.concatenate([1000.0 + np.arange(60) * 1e-9,
                         np.array([1000.0, 1000.5, 1000.0, 1000.5] * 10)])
    st = vol_reversal_state(px, 2.0, 2.0, window=52)
    # the clamp floor is 3%, and nothing here moves 3%, so it never buys
    assert st.sum() == 0


def test_vol_filter_scales_with_volatility():
    """The same k must produce a wider band on a wilder series."""
    rng = np.random.default_rng(31)
    quiet = 1000.0 * np.cumprod(1.0 + rng.normal(0.0, 0.01, 400))
    wild = 1000.0 * np.cumprod(1.0 + rng.normal(0.0, 0.06, 400))
    q = int(np.abs(np.diff(vol_reversal_state(quiet, 2.0, 2.0))).sum())
    w = int(np.abs(np.diff(vol_reversal_state(wild, 2.0, 2.0))).sum())
    # a wider band on the wild series means it does NOT flip proportionally more
    assert w <= q * 3
