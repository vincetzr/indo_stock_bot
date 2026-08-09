"""The intraday dip-reversal rule.

The index filter is the part that matters most: without it the measured win rate
falls from 69% to 61%, because it is what separates panic in one name from the
start of a market-wide decline. So the tests concentrate on the conditions being
enforced identically everywhere, and on the ordering rule that the daily-bar
version of this analysis got wrong.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

from idxbot.dipreversal import (  # noqa: E402
    DIP_PCT,
    MAX_ENTRY_BAR,
    STOP_PCT,
    TARGET_PCT,
    levels,
    qualifies,
    render_plan,
    simulate_session,
    summarise,
)

GOOD_INDEX = np.full(7, 0.01)      # index up 1% all session
BAD_INDEX = np.full(7, -0.01)      # index down: a market-wide selloff


def test_levels_are_derived_from_the_open_alone():
    lv = levels(1000.0)
    assert lv["limit"] == pytest.approx(900.0)
    assert lv["target"] == pytest.approx(900.0 * 1.05)
    assert lv["stop"] == pytest.approx(900.0 * 0.80)


def test_no_fill_means_no_trade():
    # Never trades down to the -10% limit.
    highs = np.full(7, 1010.0); lows = np.full(7, 950.0); closes = np.full(7, 1000.0)
    assert simulate_session(highs, lows, closes, 1000.0, GOOD_INDEX, 0.01) is None


def test_a_filled_dip_that_recovers_takes_the_target():
    lows = np.array([1000, 890, 895, 900, 950, 960, 970.0])
    highs = np.array([1005, 995, 960, 970, 990, 1000, 1010.0])
    closes = highs
    r = simulate_session(highs, lows, closes, 1000.0, GOOD_INDEX, 0.01)
    assert r is not None and r["outcome"] == "target"
    assert r["ret"] == pytest.approx(TARGET_PCT)


def test_a_market_wide_selloff_is_refused():
    """The filter doing its job: same price path, index down instead of up."""
    lows = np.array([1000, 890, 895, 900, 950, 960, 970.0])
    highs = np.array([1005, 995, 960, 970, 990, 1000, 1010.0])
    assert simulate_session(highs, lows, highs, 1000.0, BAD_INDEX, 0.01) is None


def test_a_downtrending_name_is_refused():
    lows = np.array([1000, 890, 895, 900, 950, 960, 970.0])
    highs = np.array([1005, 995, 960, 970, 990, 1000, 1010.0])
    assert simulate_session(highs, lows, highs, 1000.0, GOOD_INDEX,
                            prior_trend=-0.01) is None


def test_a_late_dip_is_refused():
    # The limit is only touched on the sixth bar, past the three-hour window.
    lows = np.array([1000, 995, 990, 985, 980, 890, 895.0])
    highs = np.array([1005, 1000, 995, 990, 985, 950, 960.0])
    assert simulate_session(highs, lows, highs, 1000.0, GOOD_INDEX, 0.01) is None


def test_the_stop_wins_a_bar_that_spans_both_levels():
    """Intraday order is unknowable within a bar, so the bad outcome is assumed."""
    lows = np.array([1000, 890, 700.0])        # bar 3 breaks the 720 stop
    highs = np.array([1005, 995, 950.0])       # and also clears the 945 target
    r = simulate_session(highs, lows, highs, 1000.0, GOOD_INDEX, 0.01)
    assert r["outcome"] == "stop"
    assert r["ret"] == pytest.approx(-STOP_PCT)


def test_only_bars_after_the_fill_can_resolve_the_trade():
    """The failure the daily-bar version made: a pre-entry rally counting as a win.

    Bar 1 prints a high far above the target, but the fill happens on bar 2, so
    that high must be invisible to the outcome.
    """
    highs = np.array([1200.0, 995.0, 900.0, 880.0])   # the 1200 precedes the fill
    lows = np.array([1000.0, 890.0, 880.0, 870.0])
    r = simulate_session(highs, lows, highs, 1000.0, GOOD_INDEX, 0.01)
    assert r is not None
    assert r["outcome"] != "target"


def test_unknown_index_return_refuses_rather_than_assumes():
    lows = np.array([1000, 890, 895, 950.0])
    highs = np.array([1005, 995, 960, 990.0])
    assert simulate_session(highs, lows, highs, 1000.0, None, 0.01) is None


# --------------------------------------------------------------------------
# qualifies / summarise / render
# --------------------------------------------------------------------------

def test_qualifies_lists_every_failing_condition():
    ok, reasons = qualifies(index_return=-0.01, prior_trend=-0.02, entry_bar=5)
    assert not ok
    assert len(reasons) == 3


def test_qualifies_passes_when_all_conditions_hold():
    ok, reasons = qualifies(index_return=0.01, prior_trend=0.01, entry_bar=0)
    assert ok and reasons == []


def test_summarise_charges_costs_and_reports_profit_factor():
    s = summarise([TARGET_PCT] * 7 + [-STOP_PCT] * 3, cost_pct=0.004)
    assert s["trades"] == 10
    assert s["win_rate"] == pytest.approx(0.7)
    assert s["expectancy"] == pytest.approx(0.7 * 0.046 + 0.3 * -0.204)
    assert s["profit_factor"] < 1        # 7 wins at 5% lose to 3 losses at 20%


def test_summarise_handles_nothing():
    assert summarise([]) == {}


def test_render_refuses_and_says_why_when_conditions_fail():
    text = render_plan("BBCA", 1000.0, index_return=-0.01, prior_trend=0.01)
    assert "do NOT take the fill" in text


def test_render_states_the_sample_and_that_80_percent_is_not_reached():
    text = render_plan("BBCA", 1000.0, index_return=0.01, prior_trend=0.01)
    assert "all conditions met" in text
    assert "138 trades" in text
    assert "63.0% reach +5%" in text
