"""The arithmetic bound on any intraday strategy.

A same-session trade cannot return more than the session's own range. These
tests pin that reasoning down on constructed sessions, because it is the basis
for a strong claim - that a +5% intraday trade at an 80% win rate is above the
ceiling rather than merely hard - and a strong claim needs its machinery checked.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

from idxbot.data.intraday import opening_range, resolve_path  # noqa: E402


def _session(prices, start="2026-08-07 09:00"):
    """One session of 5-minute bars from a close path."""
    n = len(prices)
    p = np.asarray(prices, dtype=float)
    ts = pd.date_range(start, periods=n, freq="5min")
    return pd.DataFrame({
        "ts": ts, "open": p, "high": p, "low": p, "close": p,
        "volume": np.full(n, 1000.0),
        "minutes_since_open": np.arange(n) * 5,
    })


def test_reachable_from_open_never_exceeds_reachable_from_low():
    """The ordering that makes the ceiling a ceiling.

    high/low - 1 is always at least high/open - 1, because low <= open. So the
    perfect-foresight figure bounds the realistic one for every session.
    """
    rng = np.random.default_rng(0)
    for _ in range(200):
        o, h, low = 1000.0, 1000.0 * (1 + rng.uniform(0, 0.3)), 1000.0 * (1 - rng.uniform(0, 0.3))
        h = max(h, o, low)
        low = min(low, o, h)
        assert (h / low - 1) >= (h / o - 1) - 1e-12


def test_a_session_cannot_pay_more_than_its_range():
    """No entry/exit pair inside a session beats buying the low and selling the high."""
    prices = [100, 104, 98, 107, 101]
    best = max(prices) / min(prices) - 1
    rng = np.random.default_rng(1)
    for _ in range(200):
        i, j = sorted(rng.choice(len(prices), 2, replace=False))
        assert prices[j] / prices[i] - 1 <= best + 1e-12


# --------------------------------------------------------------------------
# opening range
# --------------------------------------------------------------------------

def test_opening_range_uses_only_the_opening_window():
    # A spike after the window must not contaminate the range it defines.
    session = _session([100, 101, 102, 100, 500, 500])
    orb = opening_range(session, minutes=10)
    assert orb["or_high"] == pytest.approx(102.0)


def test_opening_range_on_a_flat_open():
    session = _session([100, 100, 100, 105])
    orb = opening_range(session, minutes=10)
    assert orb["or_high"] == pytest.approx(100.0)
    assert orb["or_low"] == pytest.approx(100.0)


# --------------------------------------------------------------------------
# path resolution
# --------------------------------------------------------------------------

def test_target_reached_is_reported_as_target():
    session = _session([100, 102, 106])
    out = resolve_path(session, entry=100.0, target=105.0, stop=95.0)
    assert out["outcome"] == "target"


def test_stop_reached_is_reported_as_stop():
    session = _session([100, 98, 94])
    out = resolve_path(session, entry=100.0, target=105.0, stop=95.0)
    assert out["outcome"] == "stop"


def test_neither_reached_is_neither():
    session = _session([100, 101, 99, 100])
    out = resolve_path(session, entry=100.0, target=105.0, stop=95.0)
    assert out["outcome"] not in ("target", "stop")


def test_a_target_below_the_round_trip_cost_is_a_losing_win():
    """Why the intraday win rate collapses to zero under a ~0.4% target.

    A 'win' that pays less than it costs to trade is a loss. This is the whole
    reason shrinking the target cannot buy an 80% win rate.
    """
    cost = 0.004
    for target in (0.003, 0.002, 0.001):
        assert target - cost < 0
    for target in (0.005, 0.01, 0.02):
        assert target - cost > 0
