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


# --------------------------------------------------------------------------
# gap-down variant
# --------------------------------------------------------------------------

from idxbot.dipreversal import (  # noqa: E402
    GAP_MIN,
    GAP_STOP,
    GAP_TARGET,
    gap_levels,
    gap_qualifies,
    render_gap_plan,
    simulate_gap_session,
)


def test_gap_levels_hang_off_the_open_because_that_is_the_entry():
    lv = gap_levels(1000.0)
    assert lv["entry"] == pytest.approx(1000.0)
    assert lv["target"] == pytest.approx(1000.0 * (1 + GAP_TARGET))
    assert lv["stop"] == pytest.approx(1000.0 * (1 - GAP_STOP))


def test_a_shallow_gap_does_not_qualify():
    ok, reasons = gap_qualifies(gap=-0.05, index_return=0.01, prior_trend=0.01)
    assert not ok and any("not below" in r for r in reasons)


def test_a_gap_up_never_qualifies():
    ok, _ = gap_qualifies(gap=+0.20, index_return=0.01, prior_trend=0.01)
    assert not ok


def test_a_deep_gap_into_a_rising_market_qualifies():
    ok, reasons = gap_qualifies(gap=-0.18, index_return=0.008, prior_trend=0.01)
    assert ok and reasons == []


def test_a_deep_gap_into_a_falling_market_is_refused():
    """The distinction the whole rule rests on: idiosyncratic vs market-wide."""
    ok, reasons = gap_qualifies(gap=-0.18, index_return=-0.01, prior_trend=0.01)
    assert not ok and any("index" in r for r in reasons)


def test_gap_session_takes_the_target_when_it_recovers():
    highs = np.array([1000.0, 1020.0, 1060.0])
    lows = np.array([980.0, 990.0, 1010.0])
    r = simulate_gap_session(highs, lows, highs, 1000.0, -0.18, 0.008, 0.01)
    assert r["outcome"] == "target"
    assert r["ret"] == pytest.approx(GAP_TARGET)


def test_gap_session_exits_at_the_close_when_it_stalls():
    highs = np.array([1000.0, 1020.0, 1030.0])
    lows = np.array([980.0, 990.0, 1000.0])
    closes = np.array([1000.0, 1010.0, 1020.0])
    r = simulate_gap_session(highs, lows, closes, 1000.0, -0.18, 0.008, 0.01)
    assert r["outcome"] == "close"
    assert r["ret"] == pytest.approx(0.02)


def test_the_entry_bar_itself_cannot_resolve_the_trade():
    """Entry is the open, so bar 0's own high is part of the entry bar.

    Counting it would let a trade win on a print that may have happened before
    the fill - the same ordering error the daily-bar analysis made.
    """
    highs = np.array([2000.0, 1010.0, 1020.0])   # bar 0 spikes above the target
    lows = np.array([1000.0, 1000.0, 1005.0])
    r = simulate_gap_session(highs, lows, highs, 1000.0, -0.18, 0.008, 0.01)
    assert r["outcome"] != "target"


def test_gap_render_states_the_confidence_interval_not_just_the_hit_rate():
    text = render_gap_plan("BREN", 1000.0, -0.18, 0.008, 0.01)
    assert "84.0% made +5%" in text
    assert "[70%, 98%]" in text          # the error bar must travel with it
    assert "n=25" in text


def test_gap_render_refuses_and_explains_when_conditions_fail():
    text = render_gap_plan("BREN", 1000.0, -0.05, 0.008, 0.01)
    assert "no trade" in text


# --------------------------------------------------------------------------
# capitulation variant — the validated rule
# --------------------------------------------------------------------------

from idxbot.dipreversal import (  # noqa: E402
    CAP_GAP_MIN,
    CAP_STOP,
    CAP_TARGET,
    capitulation_levels,
    capitulation_qualifies,
    render_capitulation_plan,
    simulate_capitulation,
)


def test_capitulation_needs_all_three_conditions():
    ok, reasons = capitulation_qualifies(gap=-0.12, index_gap=-0.01,
                                         prior_20d_return=-0.15)
    assert ok and reasons == []


@pytest.mark.parametrize("gap,igap,r20,missing", [
    (-0.05, -0.01, -0.15, "gap"),        # gap too shallow
    (-0.12, -0.01, +0.15, "falling"),    # stock was rising
    (-0.12, +0.01, -0.15, "market-wide"),  # index gapped UP
])
def test_each_missing_condition_refuses_with_a_reason(gap, igap, r20, missing):
    ok, reasons = capitulation_qualifies(gap, igap, r20)
    assert not ok
    assert any(missing in r for r in reasons)


def test_the_rising_stock_case_is_the_one_the_small_sample_got_backwards():
    """On 25 hourly trades a RISING prior trend looked right; on 761k daily
    sessions it measured 68.2% against 82.2% for a falling one. The rule
    encodes the large-sample direction."""
    rising = capitulation_qualifies(-0.12, -0.01, +0.10)[0]
    falling = capitulation_qualifies(-0.12, -0.01, -0.10)[0]
    assert falling and not rising


def test_a_qualifying_session_that_bounces_makes_the_target():
    r = simulate_capitulation(session_high=1060.0, session_low=980.0,
                              session_close=1050.0, session_open=1000.0,
                              gap=-0.12, index_gap=-0.01, prior_20d_return=-0.15)
    assert r["outcome"] == "target" and r["made_target"] is True
    assert r["ret"] == pytest.approx(CAP_TARGET)


def test_a_qualifying_session_that_stalls_exits_at_the_close():
    r = simulate_capitulation(1020.0, 980.0, 1010.0, 1000.0,
                              -0.12, -0.01, -0.15)
    assert r["outcome"] == "close" and r["made_target"] is False
    assert r["ret"] == pytest.approx(0.01)


def test_the_stop_takes_precedence_over_the_target():
    # Touched both; daily bars cannot order them, so the loss is assumed.
    r = simulate_capitulation(session_high=1060.0, session_low=750.0,
                              session_close=800.0, session_open=1000.0,
                              gap=-0.12, index_gap=-0.01, prior_20d_return=-0.15)
    assert r["outcome"] == "stop"
    assert r["ret"] == pytest.approx(-CAP_STOP)


def test_a_non_qualifying_session_is_not_traded_at_all():
    assert simulate_capitulation(1060.0, 980.0, 1050.0, 1000.0,
                                 gap=-0.02, index_gap=-0.01,
                                 prior_20d_return=-0.15) is None


def test_levels_are_exactly_the_open_and_its_multiples():
    lv = capitulation_levels(1000.0)
    assert lv["entry"] == pytest.approx(1000.0)
    assert lv["target"] == pytest.approx(1050.0)
    assert lv["stop"] == pytest.approx(800.0)


def test_render_carries_the_walk_forward_number_and_the_weak_year():
    text = render_capitulation_plan("BBRI", 1000.0, -0.12, -0.01, -0.15)
    assert "all three conditions met" in text
    assert "84.6%" in text and "[81.5%, 87.6%]" in text
    assert "86.2%" in text          # the walk-forward figure
    assert "2025 at 65%" in text    # the weak year travels with it


def test_render_refuses_and_explains():
    text = render_capitulation_plan("BBRI", 1000.0, -0.02, -0.01, -0.15)
    assert "no trade" in text


def test_the_gap_threshold_matches_the_auto_rejection_floor():
    """The 10% threshold is mechanical, and the config knows the ARB rules.

    A rule whose threshold coincides with an exchange limit is a different kind
    of claim from one whose threshold was searched. This pins the coincidence so
    that changing CAP_GAP_MIN to some fitted value fails loudly.
    """
    from idxbot.config import load_config
    cfg = load_config()
    arb = cfg.get("market.auto_rejection.arb_symmetric_pct", None)
    assert arb is not None, "ARB must stay configured; the rule leans on it"
    # The entry threshold must sit at or inside a real rejection band, never at
    # some intermediate value that only a backtest would choose.
    assert CAP_GAP_MIN in (0.10, 0.15, 0.20, 0.25), (
        f"CAP_GAP_MIN={CAP_GAP_MIN} is not an IDX auto-rejection level")


def test_a_gap_just_short_of_the_floor_is_refused():
    """-9.5% and -10.5% are half a percent apart and measured 56.9% vs 86.3%.

    The rule must fall on the right side of that step.
    """
    assert capitulation_qualifies(-0.095, -0.01, -0.15)[0] is False
    assert capitulation_qualifies(-0.105, -0.01, -0.15)[0] is True
