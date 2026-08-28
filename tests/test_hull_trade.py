"""Tests for H39 — the Hull-green/BUY to Hull-red/SELL round trip.

Two defects would each turn this study into an advertisement rather than a
measurement: filling on the signal bar instead of the next one, and comparing
the rule to a "buy-and-hold" that is secretly the same trade. Both happened
here, and both are pinned.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir,
                                "scripts"))

from hull_trade import COST, EXITS, HULLS, SIGNALS, states, trades  # noqa: E402


def _t(p, up, on, exit_mode="both", el=None, yr=None):
    p = np.asarray(p, float)
    up = np.asarray(up, bool)
    on = np.asarray(on, bool)
    el = np.ones(len(p), bool) if el is None else np.asarray(el, bool)
    yr = np.full(len(p), 2015) if yr is None else np.asarray(yr)
    return trades(p, up, on, el, yr, exit_mode)


# ============================================== the look-ahead guard, first ===
def test_the_fill_is_the_bar_AFTER_the_condition():
    """THE SINGLE MOST COMMON WAY A BACKTEST INVENTS MONEY. If the entry filled
    on the bar whose close produced the signal, the rule would buy at a price it
    could only know after the close."""
    p = [10.0, 10.0, 20.0, 20.0, 20.0, 5.0, 5.0]
    up = [False, True, True, True, False, False, False]
    on = [False, True, True, True, False, False, False]
    out = _t(p, up, on)
    assert len(out) == 1
    #  condition first true at bar 1, so the fill is bar 2 at 20.0 — not bar 1
    #  at 10.0, which would have handed the rule the whole gap for free.
    assert out[0]["i"] == 2
    assert p[out[0]["i"]] == 20.0


def test_the_exit_is_also_delayed_by_one_bar():
    p = [10.0, 10.0, 10.0, 10.0, 8.0, 4.0]
    up = [False, True, True, False, False, False]
    on = [False, True, True, False, False, False]
    out = _t(p, up, on)
    assert len(out) == 1
    #  exit condition first true at bar 3, so the sale is bar 4 at 8.0
    assert out[0]["j"] == 4 and p[out[0]["j"]] == 8.0


def test_a_trade_still_open_at_the_end_is_dropped_not_marked_to_market():
    """An unclosed position has no realised return, and giving it the last
    price would let a rising sample end on a free winner."""
    p = [10.0] * 4 + [50.0]
    up = [False, True, True, True, True]
    on = [False, True, True, True, True]
    assert _t(p, up, on) == []


# ==================================================== the rule's own logic ====
def test_entry_needs_BOTH_conditions():
    p = [10.0] * 8
    assert _t(p, [False, True, True, True, False, False, False, False],
              [False, False, False, False, False, False, False, False]) == []


def test_exit_both_holds_through_a_single_condition_turning_off():
    """The distinction the user asked about: waiting for both means sitting
    through the first leg down."""
    p = [10.0, 10.0, 10.0, 9.0, 8.0, 7.0, 7.0]
    up = [False, True, True, False, False, False, False]
    on = [False, True, True, True, True, False, False]
    both = _t(p, up, on, "both")
    hull = _t(p, up, on, "hull only")
    assert both[0]["j"] > hull[0]["j"], "exiting on the Hull alone must be sooner"
    assert both[0]["ret"] < hull[0]["ret"], "and in a fall, must lose more"


def test_exit_hull_only_ignores_the_signal_entirely():
    p = [10.0, 10.0, 10.0, 12.0, 12.0]
    up = [False, True, True, False, False]
    on = [False, True, True, True, True]
    out = _t(p, up, on, "hull only")
    assert len(out) == 1 and out[0]["j"] == 4


def test_every_exit_mode_is_a_distinct_rule():
    assert set(EXITS) == {"both", "hull only", "signal only"}


# ============================================================ the accounting ==
def test_the_round_trip_toll_is_charged_once_per_trade():
    p = [10.0, 10.0, 10.0, 10.0, 10.0]
    up = [False, True, True, False, False]
    on = [False, True, True, False, False]
    out = _t(p, up, on)
    assert out[0]["ret"] == pytest.approx(-COST)


def test_a_flat_round_trip_loses_exactly_the_toll_and_not_twice():
    """A19: the toll is 0.28 buy + 0.18 sell + 0.10 tax = 0.56% per ROUND TRIP,
    not per side. Charging it twice would quietly halve every result."""
    assert COST == pytest.approx(0.0056)


def test_an_ineligible_entry_bar_is_skipped():
    """A bar you could not have bought is not a trade. Counting it would let
    the rule 'trade' suspended and untradeable sessions."""
    p = [10.0, 10.0, 10.0, 20.0, 20.0]
    up = [False, True, True, False, False]
    on = [False, True, True, False, False]
    el = [True, True, False, True, True]
    assert _t(p, up, on, el=el) == []


def test_the_bars_held_is_the_gap_between_the_two_fills():
    p = [10.0] * 10
    up = [False, True, True, True, True, True, False, False, False, False]
    on = up
    out = _t(p, up, on)
    assert out[0]["bars"] == out[0]["j"] - out[0]["i"]


# ================================================================ the states ==
def test_the_hull_slope_is_undefined_before_the_window_fills():
    """A rising slope computed off a NaN would fire an entry on bar one of
    every name, which is 891 free trades at the start of the sample."""
    px = pd.Series(np.linspace(100.0, 200.0, 120))
    up, _ = states(px, 55, "hull only")
    assert not up[:56].any()
    assert up[-1]


def test_the_signal_sources_are_the_ones_on_the_chart():
    assert set(SIGNALS) == {"EMA34", "EMA50", "EMA stack", "hull only"}
    assert 55 in HULLS


def test_hull_only_leaves_the_signal_permanently_on():
    px = pd.Series(np.linspace(100.0, 200.0, 300))
    _, on = states(px, 55, "hull only")
    assert on.all()


def test_the_ema_stack_signal_needs_all_three_to_be_ordered():
    px = pd.Series(np.linspace(100.0, 400.0, 400))
    _, on = states(px, 55, "EMA stack")
    assert on[-1], "a clean uptrend must satisfy price>50>100>200"
    down = pd.Series(np.linspace(400.0, 100.0, 400))
    _, off = states(down, 55, "EMA stack")
    assert not off[-1]
