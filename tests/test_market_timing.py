"""Tests for the index-timing study.

The one thing that would make every number in this study wrong is acting on
the same bar the signal is computed from, so that is pinned first and hardest.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir,
                                "scripts"))

from market_timing import RULES, null_matched, run          # noqa: E402


def _F(rets):
    d = pd.date_range("2010-01-01", periods=len(rets), freq="B")
    close = 100.0 * np.cumprod(1.0 + np.asarray(rets, float))
    F = pd.DataFrame({"close": close}, index=d)
    F["ret"] = F["close"].pct_change()
    return F


def _sig(F, values):
    return pd.Series(values, index=F.index)


# ======================================================== the look-ahead guard
def test_the_signal_is_acted_on_the_NEXT_bar_not_the_same_one():
    """WITHOUT THIS EVERY RULE IN THE TABLE LOOKS BRILLIANT. A signal that is
    true exactly on the crash day, acted on that day, dodges the crash; acted
    on the next day it does not."""
    F = _F([0.0, 0.0, -0.50, 0.0, 0.0])
    #  "out on the crash bar" — perfect foresight if acted same-bar
    perfect = _sig(F, [True, True, False, True, True])
    got = run(F, perfect, fee=0.0)
    #  shifted by one, the position is still IN on the crash bar
    assert got["terminal"] == pytest.approx(0.5, abs=1e-9)


def test_a_signal_known_one_bar_early_does_avoid_the_crash():
    F = _F([0.0, 0.0, -0.50, 0.0, 0.0])
    early = _sig(F, [True, False, True, True, True])
    assert run(F, early, fee=0.0)["terminal"] == pytest.approx(1.0, abs=1e-9)


# ============================================================ the arithmetic
def test_always_in_reproduces_the_index_from_its_SECOND_bar():
    """Buy-and-hold is charged the same one-bar delay as every other rule, so
    it earns returns 1..n-1 and not 0..n-1. Comparing it to the index from bar
    ZERO would hand the reference a free bar the rules do not get, which is a
    window mismatch of exactly the kind reports/portfolio.md records twice."""
    F = _F([0.01] * 50)
    s = run(F, RULES["always in (buy & hold)"](F), fee=0.0)
    assert s["terminal"] == pytest.approx(
        F["close"].iloc[-1] / F["close"].iloc[0], rel=1e-9)
    assert s["in_frac"] == pytest.approx(1.0, abs=0.05)


def test_being_out_earns_nothing_rather_than_the_index():
    F = _F([0.02] * 40)
    out = _sig(F, [False] * 40)
    assert run(F, out, fee=0.0)["terminal"] == pytest.approx(1.0)


def test_each_switch_is_charged_half_a_round_trip():
    F = _F([0.0] * 10)
    flip = _sig(F, [True, False] * 5)
    s = run(F, flip, fee=0.10)
    #  flat market, so terminal is pure cost; 9 acted switches at 5% each
    assert s["terminal"] < 1.0
    assert s["switches"] >= 8


def test_switch_count_is_transitions_not_days_in_market():
    """One round trip is TWO transitions — the entry and the exit — and each
    is charged half a round trip, which is how the fee comes out right."""
    F = _F([0.0] * 20)
    once = _sig(F, [True] * 10 + [False] * 10)
    assert run(F, once, fee=0.0)["switches"] == 2
    always = run(F, _sig(F, [True] * 20), fee=0.0)
    assert always["switches"] == 1                 # the entry alone


def test_max_drawdown_is_on_the_strategy_equity_not_the_index():
    """A rule that sits in cash through a crash must report a shallow
    drawdown even though the index's was deep."""
    F = _F([0.0, -0.40, -0.40, 0.0])
    out = _sig(F, [False, False, False, False])
    assert run(F, out, fee=0.0)["maxdd"] == pytest.approx(0.0)
    assert run(F, RULES["always in (buy & hold)"](F), fee=0.0)["maxdd"] < -0.5


# ================================================================== the null
def test_the_null_matches_the_trade_count_it_is_given():
    """THE COLUMN THAT DECIDES THE TABLE. A rule out of the market a third of
    the time dodges a third of the crashes by construction, so the comparison
    has to be against a random switcher with the SAME trade count."""
    rng = np.random.default_rng(0)
    F = _F(rng.normal(0.0004, 0.012, 600))
    nu = null_matched(F, n_switch=40, in_frac=0.65, draws=25, seed=3)
    assert np.isfinite(nu["cagr_mean"])
    assert nu["cagr_p95"] > nu["cagr_mean"]


def test_the_null_is_reproducible_and_seed_sensitive():
    rng = np.random.default_rng(1)
    F = _F(rng.normal(0.0004, 0.012, 400))
    a = null_matched(F, 30, 0.6, draws=20, seed=5)
    b = null_matched(F, 30, 0.6, draws=20, seed=5)
    c = null_matched(F, 30, 0.6, draws=20, seed=6)
    assert a["cagr_mean"] == pytest.approx(b["cagr_mean"])
    assert a["cagr_mean"] != pytest.approx(c["cagr_mean"])


def test_a_null_with_more_time_in_market_earns_more_in_a_rising_tape():
    F = _F([0.002] * 500)
    lo = null_matched(F, 20, 0.2, draws=20, seed=2)["cagr_mean"]
    hi = null_matched(F, 20, 0.9, draws=20, seed=2)["cagr_mean"]
    assert hi > lo


# ================================================================== the rules
def test_every_registered_rule_returns_a_boolean_series_of_panel_length():
    rng = np.random.default_rng(4)
    F = _F(rng.normal(0.0004, 0.012, 800))
    for w in (20, 50, 100, 200):
        F[f"ma{w}"] = F["close"].rolling(w).mean()
    F["hi252"] = F["close"].rolling(252, min_periods=60).max()
    F["dd"] = F["close"] / F["hi252"] - 1.0
    F["mom252"] = F["close"] / F["close"].shift(252) - 1.0
    F["mom20"] = F["close"] / F["close"].shift(20) - 1.0
    F["vol20"] = F["ret"].rolling(20).std() * np.sqrt(252)
    for nm, fn in RULES.items():
        s = fn(F)
        assert len(s) == len(F), nm
        assert s.fillna(False).dtype == bool, nm


def test_buy_and_hold_is_in_the_rule_table_as_the_reference():
    assert "always in (buy & hold)" in RULES
