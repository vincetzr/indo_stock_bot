"""Path-dependent exits.

Hand-built price paths where the correct answer is obvious by inspection, so a
wrong result is a bug rather than a judgement call. The ambiguity rule and the
entry-timing rule get the most attention: both are places where a small
optimistic choice silently manufactures a win rate that does not exist.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

from idxbot.barrier import (  # noqa: E402
    STOP,
    TARGET,
    TIMEOUT,
    BarrierConfig,
    render_grid,
    simulate_one,
    simulate_ticker,
    summarise,
)


def _bars(closes, highs=None, lows=None, opens=None, start="2020-01-01"):
    n = len(closes)
    closes = np.asarray(closes, dtype=float)
    return pd.DataFrame({
        "date": pd.bdate_range(start, periods=n),
        "open": np.asarray(opens, dtype=float) if opens is not None else closes,
        "high": np.asarray(highs, dtype=float) if highs is not None else closes,
        "low": np.asarray(lows, dtype=float) if lows is not None else closes,
        "close": closes,
    })


CFG = BarrierConfig(target_pct=0.05, stop_pct=0.08, max_days=10)


# --------------------------------------------------------------------------
# simulate_one
# --------------------------------------------------------------------------

def test_target_touched_first_is_a_win_at_the_target():
    highs = np.array([1010, 1060, 1100.0])
    lows = np.array([995, 1000, 1050.0])
    r = simulate_one(highs, lows, highs, 1000.0, CFG)
    assert r["outcome"] == TARGET
    assert r["ret"] == pytest.approx(0.05)
    assert r["days"] == 2


def test_stop_touched_first_is_a_loss_at_the_stop():
    highs = np.array([1010, 1005, 1000.0])
    lows = np.array([995, 915, 900.0])
    r = simulate_one(highs, lows, highs, 1000.0, CFG)
    assert r["outcome"] == STOP
    assert r["ret"] == pytest.approx(-0.08)
    assert r["days"] == 2


def test_neither_barrier_touched_exits_at_the_last_close():
    closes = np.full(10, 1020.0)
    r = simulate_one(np.full(10, 1030.0), np.full(10, 990.0), closes, 1000.0, CFG)
    assert r["outcome"] == TIMEOUT
    assert r["ret"] == pytest.approx(0.02)
    assert r["days"] == 10


def test_one_bar_spanning_both_barriers_is_scored_as_a_loss():
    """The rule that stops an 80% win rate from being conjured out of nothing.

    Daily data cannot say whether the high or the low came first, so the
    pessimistic reading is the only defensible one.
    """
    highs = np.array([1060.0])   # clears the +5% target
    lows = np.array([915.0])     # and breaks the -8% stop
    r = simulate_one(highs, lows, highs, 1000.0, CFG)
    assert r["outcome"] == STOP


def test_position_is_not_held_past_max_days():
    highs = np.concatenate([np.full(10, 1010.0), np.full(10, 2000.0)])
    lows = np.full(20, 990.0)
    closes = np.full(20, 1000.0)
    r = simulate_one(highs, lows, closes, 1000.0, CFG)
    assert r["outcome"] == TIMEOUT       # the +100% bar arrives after the exit
    assert r["days"] == 10


def test_invalid_entry_price_yields_no_trade():
    for bad in (0.0, -5.0, np.nan):
        r = simulate_one(np.array([1.0]), np.array([1.0]), np.array([1.0]), bad, CFG)
        assert np.isnan(r["ret"])


# --------------------------------------------------------------------------
# entry timing
# --------------------------------------------------------------------------

def test_entry_is_the_next_bar_open_not_the_signal_close():
    """Buying at the close that generated the signal is not a tradeable fill."""
    bars = _bars(closes=[1000, 1200, 1200, 1200],
                 opens=[1000, 1100, 1200, 1200],
                 highs=[1000, 1200, 1200, 1200],
                 lows=[1000, 1100, 1200, 1200])
    out = simulate_ticker(bars, [bars["date"].iloc[0]], CFG)
    assert len(out) == 1
    # entry is 1100 (next open), not 1000 (signal close)
    assert out["entry"].iloc[0] == pytest.approx(1100.0)


def test_signal_on_the_final_bar_is_dropped_not_filled():
    # There is no next open, so there is no trade - rather than inventing one.
    bars = _bars(closes=[1000, 1010, 1020])
    out = simulate_ticker(bars, [bars["date"].iloc[-1]], CFG)
    assert out.empty


def test_unknown_signal_date_is_ignored():
    bars = _bars(closes=[1000, 1010, 1020])
    out = simulate_ticker(bars, [pd.Timestamp("1999-01-01")], CFG)
    assert out.empty


def test_simulate_ticker_handles_empty_input():
    assert simulate_ticker(pd.DataFrame(), [], CFG).empty
    assert simulate_ticker(_bars([1, 2, 3]), [], CFG).empty


# --------------------------------------------------------------------------
# summarise
# --------------------------------------------------------------------------

def test_costs_are_charged_on_every_trade():
    out = pd.DataFrame({"outcome": [TARGET] * 10, "ret": [0.05] * 10,
                        "days": [5] * 10})
    s = summarise(out, cost_pct=0.004)
    assert s["hit_rate"] == pytest.approx(1.0)
    assert s["expectancy"] == pytest.approx(0.046)   # 5% gross less the round trip


def test_a_high_hit_rate_can_still_have_negative_expectancy():
    """The trap this module exists to expose.

    Nine wins at +3% against one loss at -30% is an 90% hit rate and a losing
    strategy. Any report that quotes the first without the second is misleading.
    """
    out = pd.DataFrame({
        "outcome": [TARGET] * 9 + [STOP],
        "ret": [0.03] * 9 + [-0.30],
        "days": [10] * 10,
    })
    s = summarise(out, cost_pct=0.004)
    assert s["hit_rate"] == pytest.approx(0.9)
    assert s["expectancy"] < 0
    assert s["profit_factor"] < 1


def test_annualised_return_accounts_for_capital_being_tied_up():
    fast = summarise(pd.DataFrame({"outcome": [TARGET] * 10, "ret": [0.05] * 10,
                                   "days": [5] * 10}), cost_pct=0.0)
    slow = summarise(pd.DataFrame({"outcome": [TARGET] * 10, "ret": [0.05] * 10,
                                   "days": [60] * 10}), cost_pct=0.0)
    assert fast["expectancy"] == pytest.approx(slow["expectancy"])
    assert fast["ann_return"] > slow["ann_return"]   # same trade, less waiting


def test_summarise_handles_empty_and_all_nan():
    assert summarise(pd.DataFrame()) == {}
    assert summarise(pd.DataFrame({"outcome": [TARGET], "ret": [np.nan],
                                   "days": [1]})) == {}


# --------------------------------------------------------------------------
# render
# --------------------------------------------------------------------------

def test_render_flags_when_every_qualifying_setup_loses_money():
    rows = [{
        "setup": "+3%/-30%/60d", "trades": 100, "hit_rate": 0.90, "stop_rate": 0.10,
        "timeout_rate": 0.0, "expectancy": -0.005, "avg_win": 0.03, "avg_loss": -0.30,
        "profit_factor": 0.9, "avg_days": 10, "ann_return": -0.12, "win_rate": 0.90,
    }]
    text = render_grid(rows, target_hit_rate=0.80)
    assert "Every setup that clears the win-rate bar loses money" in text


def test_render_reports_counts_for_a_profitable_qualifier():
    rows = [{
        "setup": "+5%/-8%/60d", "trades": 100, "hit_rate": 0.85, "stop_rate": 0.10,
        "timeout_rate": 0.05, "expectancy": 0.02, "avg_win": 0.05, "avg_loss": -0.08,
        "profit_factor": 2.1, "avg_days": 12, "ann_return": 0.42, "win_rate": 0.85,
    }]
    text = render_grid(rows, target_hit_rate=0.80)
    assert "...of which are profitable after costs   : 1" in text
    assert "Every setup that clears" not in text


def test_render_survives_no_rows():
    assert "(no results)" in render_grid([])


# --------------------------------------------------------------------------
# scale-out with a trailing remainder
# --------------------------------------------------------------------------

SCALE = BarrierConfig(target_pct=0.05, stop_pct=0.08, max_days=10,
                      scale_out=0.5, trail_pct=0.10)


def test_scaling_out_banks_half_and_leaves_half_running():
    # Touches +5% on bar 1, then closes at +20% on the last bar.
    highs = np.array([1060, 1100, 1200.0])
    lows = np.array([1000, 1050, 1150.0])
    closes = np.array([1050, 1090, 1200.0])
    cfg = BarrierConfig(0.05, 0.08, 3, scale_out=0.5, trail_pct=0.0)
    r = simulate_one(highs, lows, closes, 1000.0, cfg)
    # half banked at +5%, half exited at +20% on the close
    assert r["ret"] == pytest.approx(0.5 * 0.05 + 0.5 * 0.20)
    assert r["outcome"] == TARGET


def test_the_banked_slice_survives_a_full_reversal():
    """The reason to scale out: a round trip is no longer a scratch.

    Without scaling, price touching +5% then collapsing to the stop is a full
    -8%. With half banked, the trade is barely negative.
    """
    highs = np.array([1060, 1000, 950.0])
    lows = np.array([1000, 950, 915.0])
    closes = np.array([1050, 960, 920.0])
    r = simulate_one(highs, lows, closes, 1000.0, SCALE)
    # Half banked at +5%. The trail ratchets off bar 1's *high* of 1060, not the
    # 1050 target, so the remainder's stop sits at 954 (-4.6%) and is taken on
    # bar 2. Without scaling this same path is a flat -8%.
    assert r["ret"] > -0.08
    assert r["ret"] == pytest.approx(0.5 * 0.05 + 0.5 * (954 / 1000 - 1))


def test_trailing_stop_ratchets_up_and_never_down():
    # Runs to +40%, then falls back. The trail must capture most of the move.
    highs = np.array([1060, 1200, 1400, 1300, 1000.0])
    lows = np.array([1000, 1100, 1300, 1200, 900.0])
    closes = np.array([1050, 1180, 1390, 1250, 950.0])
    r = simulate_one(highs, lows, closes, 1000.0, SCALE)
    # remainder exits on the 10% trail below the 1400 peak, i.e. 1260
    assert r["ret"] == pytest.approx(0.5 * 0.05 + 0.5 * 0.26)


def test_a_trade_that_never_reaches_the_target_behaves_exactly_as_before():
    highs = np.array([1010, 1020, 1030.0])
    lows = np.array([990, 985, 980.0])
    closes = np.array([1000, 1010, 1020.0])
    plain = simulate_one(highs, lows, closes, 1000.0,
                         BarrierConfig(0.05, 0.08, 3))
    scaled = simulate_one(highs, lows, closes, 1000.0, SCALE)
    assert plain["ret"] == pytest.approx(scaled["ret"])
    assert plain["outcome"] == scaled["outcome"] == TIMEOUT


def test_scale_out_of_one_is_identical_to_a_plain_target():
    highs = np.array([1060, 1200.0])
    lows = np.array([1000, 1100.0])
    closes = np.array([1050, 1180.0])
    plain = simulate_one(highs, lows, closes, 1000.0, BarrierConfig(0.05, 0.08, 2))
    full = simulate_one(highs, lows, closes, 1000.0,
                        BarrierConfig(0.05, 0.08, 2, scale_out=1.0, trail_pct=0.10))
    assert plain["ret"] == pytest.approx(full["ret"])


def test_scaling_still_scores_a_pre_target_stop_as_a_loss():
    highs = np.array([1010, 1020.0])
    lows = np.array([995, 900.0])
    closes = np.array([1000, 910.0])
    r = simulate_one(highs, lows, closes, 1000.0, SCALE)
    assert r["outcome"] == STOP
    assert r["ret"] == pytest.approx(-0.08)


def test_label_distinguishes_a_scale_out_from_a_plain_target():
    assert BarrierConfig(0.05, 0.08, 60).label == "+5%/-8%/60d"
    assert "x50%" in BarrierConfig(0.05, 0.08, 60, scale_out=0.5).label
    assert "trail10%" in BarrierConfig(0.05, 0.08, 60, scale_out=0.5,
                                       trail_pct=0.10).label


# --------------------------------------------------------------------------
# breakeven stop after the target
# --------------------------------------------------------------------------

def test_breakeven_stop_prevents_a_hit_trade_from_turning_negative():
    """The structure that actually lifts the win rate.

    Scaling out alone does not: a 25% slice banked at +3% cannot rescue the
    other 75% falling to the stop. Moving the stop to entry can.
    """
    highs = np.array([1060, 1020, 1000.0])
    lows = np.array([1010, 995, 900.0])     # dives well below entry on bar 3
    closes = np.array([1050, 1000, 910.0])
    cfg = BarrierConfig(0.05, 0.08, 3, scale_out=0.5, breakeven=True)
    r = simulate_one(highs, lows, closes, 1000.0, cfg)
    # half banked at +5%, half exited at entry - not at -8%
    assert r["ret"] == pytest.approx(0.5 * 0.05 + 0.5 * 0.0)
    assert r["ret"] > 0


def test_without_breakeven_the_same_path_is_a_loss_on_the_remainder():
    highs = np.array([1060, 1020, 1000.0])
    lows = np.array([1010, 995, 900.0])
    closes = np.array([1050, 1000, 910.0])
    cfg = BarrierConfig(0.05, 0.08, 3, scale_out=0.5, breakeven=False)
    r = simulate_one(highs, lows, closes, 1000.0, cfg)
    assert r["ret"] == pytest.approx(0.5 * 0.05 + 0.5 * -0.08)


def test_breakeven_does_nothing_before_the_target_is_reached():
    # A trade that never prints the target keeps its original stop.
    highs = np.array([1010, 1020.0])
    lows = np.array([995, 900.0])
    closes = np.array([1000, 910.0])
    cfg = BarrierConfig(0.05, 0.08, 2, scale_out=0.5, breakeven=True)
    r = simulate_one(highs, lows, closes, 1000.0, cfg)
    assert r["outcome"] == STOP
    assert r["ret"] == pytest.approx(-0.08)


def test_breakeven_costs_the_tail_when_a_winner_dips_and_recovers():
    """The tradeoff, asserted so it cannot be quietly forgotten.

    Same path, same signal: breakeven exits at entry on the dip and misses the
    recovery to +50% that the plain stop rides all the way.
    """
    highs = np.array([1060, 1020, 1200, 1500.0])
    lows = np.array([1010, 990, 1100, 1400.0])   # bar 2 dips just under entry
    closes = np.array([1050, 1000, 1180, 1500.0])
    shaken = simulate_one(highs, lows, closes, 1000.0,
                          BarrierConfig(0.05, 0.08, 4, scale_out=0.5, breakeven=True))
    rides = simulate_one(highs, lows, closes, 1000.0,
                         BarrierConfig(0.05, 0.08, 4, scale_out=0.5, breakeven=False))
    assert shaken["ret"] == pytest.approx(0.025)          # banked half, rest at entry
    assert rides["ret"] == pytest.approx(0.5 * 0.05 + 0.5 * 0.50)
    assert rides["ret"] > shaken["ret"]


def test_breakeven_and_trail_take_whichever_stop_is_higher():
    highs = np.array([1060, 1400, 1300.0])
    lows = np.array([1010, 1100, 1150.0])
    closes = np.array([1050, 1390, 1200.0])
    cfg = BarrierConfig(0.05, 0.08, 3, scale_out=0.5, breakeven=True, trail_pct=0.10)
    r = simulate_one(highs, lows, closes, 1000.0, cfg)
    # trail from the 1400 peak is 1260, well above breakeven, so it governs
    assert r["ret"] == pytest.approx(0.5 * 0.05 + 0.5 * 0.26)


def test_label_marks_the_breakeven_variant():
    assert "BE" in BarrierConfig(0.03, 0.10, 60, scale_out=0.25, breakeven=True).label
    assert "BE" not in BarrierConfig(0.03, 0.10, 60, scale_out=0.25).label
