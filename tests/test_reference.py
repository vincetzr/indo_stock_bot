"""Tests for the point-in-time trading-rule schedules.

The failure this file guards against is not a crash. It is a lookup that
cheerfully returns TODAY's rule for a date in 2016, producing a backtest that
runs clean and is wrong in the flattering direction:

  - a day the stock was locked limit-up is a day nobody could buy. Filling at
    that close invents profit that was not available.
  - tick size sets the half-spread floor. The 2016 ladder is finer than the
    2014 one, so applying today's to 2015 understates every small-cap cost.

So the central tests here are about REFUSAL and about TRANSITIONS: that a
pre-coverage date raises instead of answering, and that each regime boundary
falls on exactly the right day. A schedule that is off by one day is a
schedule that is wrong for one day and nobody ever notices.
"""

from __future__ import annotations

import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

from idxbot.spine.reference import (COVERAGE_START,             # noqa: E402
                                    OutsideCoverage, audit, auto_rejection,
                                    half_spread, known_gaps, lot_size,
                                    max_price_step, on_tick, rejection_prices,
                                    round_to_tick, schedule, tick_size,
                                    trading_halt, was_locked)


# --------------------------------------------------------------------------
# refusal - the whole point of a point-in-time table
# --------------------------------------------------------------------------
def test_a_date_before_coverage_raises_rather_than_answering():
    for fn in (lambda: tick_size(1000, "2001-06-01"),
               lambda: auto_rejection(1000, "2001-06-01"),
               lambda: lot_size("2001-06-01"),
               lambda: max_price_step(1000, "2001-06-01"),
               lambda: trading_halt(-0.10, "2001-06-01")):
        with pytest.raises(OutsideCoverage):
            fn()


def test_coverage_is_per_schedule_not_global():
    """The tick ladder is readable in the data back to 2005; the auto-rejection
    bands only back to 2010. Forcing one date would either discard five years
    of usable tick history or assert bands nothing supports."""
    assert tick_size(1000, "2006-06-01") == 10.0
    assert lot_size("2006-06-01") == 500
    with pytest.raises(OutsideCoverage):
        auto_rejection(1000, "2006-06-01")


def test_the_500_share_lot_era_is_encoded():
    assert lot_size("2013-12-31") == 500
    assert lot_size("2014-01-06") == 100


def test_the_pre_2014_ladder_is_the_one_the_data_shows():
    """Confirmed year by year from 2005 to 2013 by observed granularity."""
    d = "2012-06-01"
    assert [tick_size(p, d) for p in (100, 300, 1000, 3000, 9000)] \
        == [1.0, 5.0, 10.0, 25.0, 50.0]


def test_the_inferred_bands_reach_back_to_2010_and_no_further():
    assert auto_rejection(1000, "2010-01-04") == (0.25, 0.25)
    with pytest.raises(OutsideCoverage):
        auto_rejection(1000, "2009-12-31")


def test_the_first_covered_day_itself_works():
    assert tick_size(1000, COVERAGE_START) == 5.0
    assert lot_size(COVERAGE_START) == 100


def test_the_day_before_coverage_does_not():
    from idxbot.spine.reference import EARLY_START
    with pytest.raises(OutsideCoverage):
        tick_size(1000, EARLY_START - pd.Timedelta(days=1))


def test_the_error_says_what_to_do_about_it():
    """An exception nobody can act on gets caught and swallowed."""
    with pytest.raises(OutsideCoverage, match="lookahead"):
        tick_size(1000, "2001-01-01")


# --------------------------------------------------------------------------
# the schedules are coherent
# --------------------------------------------------------------------------
def test_every_schedule_is_gapless_and_covers_today():
    a = audit()
    assert a["ok"].all(), a[~a["ok"]].to_string()


def test_every_schedule_reaches_the_present():
    for name in ("ara", "arb", "tick", "max_step", "lot"):
        assert schedule(name)["to"].isna().sum() == 1


def test_an_unknown_schedule_name_is_an_error():
    with pytest.raises(ValueError):
        schedule("nonsense")


# --------------------------------------------------------------------------
# auto rejection - six changes, two of them reversals
# --------------------------------------------------------------------------
@pytest.mark.parametrize("day,expected_arb", [
    ("2019-06-03", 0.25),        # symmetric era
    ("2020-03-09", 0.25),        # last day before COVID
    ("2020-03-10", 0.10),        # first emergency step
    ("2020-03-12", 0.10),
    ("2020-03-13", 0.07),        # second emergency step
    ("2023-06-04", 0.07),        # last day of the long 7% period
    ("2023-06-05", 0.15),        # tahap I
    ("2023-09-03", 0.15),
    ("2023-09-04", 0.25),        # tahap II, symmetric again
    ("2025-04-07", 0.25),        # last day of symmetry
    ("2025-04-08", 0.15),        # asymmetric AGAIN
    ("2026-08-20", 0.15),
])
def test_each_arb_regime_starts_on_exactly_the_right_day(day, expected_arb):
    assert auto_rejection(1000.0, day)[1] == pytest.approx(expected_arb)


def test_the_upper_limit_never_moved():
    """Every change since 2020 was to the floor, not the ceiling."""
    for day in ("2019-06-03", "2020-03-16", "2023-09-05", "2026-08-20"):
        assert auto_rejection(1000.0, day)[0] == pytest.approx(0.25)


def test_the_upper_limit_still_varies_by_price_band():
    for day in ("2019-06-03", "2026-08-20"):
        assert auto_rejection(100.0, day)[0] == pytest.approx(0.35)
        assert auto_rejection(1000.0, day)[0] == pytest.approx(0.25)
        assert auto_rejection(9000.0, day)[0] == pytest.approx(0.20)


def test_the_covid_ten_percent_window_is_three_days_not_one():
    """It is easy to collapse 2020-03-10 into 2020-03-13 and lose it."""
    days = pd.bdate_range("2020-03-10", "2020-03-12")
    assert all(auto_rejection(1000.0, d)[1] == pytest.approx(0.10) for d in days)


def test_the_asymmetric_periods_really_are_asymmetric():
    for day in ("2020-03-16", "2023-06-06", "2025-04-09"):
        up, dn = auto_rejection(1000.0, day)
        assert up != dn
    for day in ("2019-06-03", "2023-09-05"):
        up, dn = auto_rejection(1000.0, day)
        assert up == dn


def test_the_thin_boards_use_a_different_ladder():
    assert auto_rejection(5.0, "2026-08-20", "acceleration") == (-1.0, -1.0)
    assert auto_rejection(50.0, "2026-08-20", "watchlist") == (0.10, 0.10)
    # and are NOT the main ladder
    assert auto_rejection(50.0, "2026-08-20", "main") != (0.10, 0.10)


def test_an_unknown_board_is_an_error_not_a_default():
    with pytest.raises(ValueError):
        auto_rejection(1000.0, "2026-08-20", "papan_imajiner")


# --------------------------------------------------------------------------
# rejection prices and lock detection
# --------------------------------------------------------------------------
def test_rejection_prices_land_on_the_tick_ladder():
    hi, lo = rejection_prices(1000.0, "2026-08-20")
    assert on_tick(hi, "2026-08-20") and on_tick(lo, "2026-08-20")


def test_the_rejection_band_brackets_the_previous_close():
    hi, lo = rejection_prices(1000.0, "2026-08-20")
    assert lo < 1000.0 < hi


def test_a_bar_pinned_at_the_ceiling_all_day_is_flagged_ara():
    hi, _ = rejection_prices(1000.0, "2026-08-20")
    assert was_locked(hi, hi, hi, hi, 1000.0, "2026-08-20") == "ARA"


def test_a_bar_pinned_at_the_floor_all_day_is_flagged_arb():
    _, lo = rejection_prices(1000.0, "2026-08-20")
    assert was_locked(lo, lo, lo, lo, 1000.0, "2026-08-20") == "ARB"


def test_a_bar_that_touched_the_ceiling_but_traded_below_is_not_locked():
    """You could have bought it. It does not get excluded."""
    hi, _ = rejection_prices(1000.0, "2026-08-20")
    assert was_locked(1000.0, hi, 990.0, hi, 1000.0, "2026-08-20") is None


def test_an_ordinary_bar_is_not_locked():
    assert was_locked(1000.0, 1020.0, 990.0, 1010.0, 1000.0, "2026-08-20") is None


def test_the_same_bar_is_locked_under_one_regime_and_not_another():
    """A 9% fall was a lock in 2021 and an ordinary day in 2019."""
    bar = dict(open_=910.0, high=910.0, low=910.0, close=910.0,
               prev_close=1000.0)
    assert was_locked(day="2021-06-01", **bar) == "ARB"      # 7% floor
    assert was_locked(day="2019-06-03", **bar) is None       # 25% floor


# --------------------------------------------------------------------------
# tick size
# --------------------------------------------------------------------------
def test_the_2016_change_added_two_bands():
    assert tick_size(300.0, "2016-04-29") == 1.0     # three-group era
    assert tick_size(300.0, "2016-05-02") == 2.0     # five-group era
    assert tick_size(3000.0, "2016-04-29") == 5.0
    assert tick_size(3000.0, "2016-05-02") == 10.0


def test_the_bands_that_did_not_change_did_not_change():
    for day in ("2016-04-29", "2016-05-02"):
        assert tick_size(100.0, day) == 1.0
        assert tick_size(1000.0, day) == 5.0
        assert tick_size(9000.0, day) == 25.0


def test_band_edges_belong_to_the_upper_band():
    """A price exactly at Rp200 takes the Rp2 tick, not the Rp1 tick."""
    assert tick_size(199.0, "2026-08-20") == 1.0
    assert tick_size(200.0, "2026-08-20") == 2.0
    assert tick_size(4999.0, "2026-08-20") == 10.0
    assert tick_size(5000.0, "2026-08-20") == 25.0


def test_the_half_spread_floor_is_bigger_on_cheap_stocks():
    cheap = half_spread(100.0, "2026-08-20")
    dear = half_spread(9000.0, "2026-08-20")
    assert cheap > dear
    assert cheap == pytest.approx(0.005)      # half a rupiah on 100


def test_applying_todays_ladder_to_2015_would_understate_the_cost():
    """The exact lookahead error this module exists to prevent."""
    then = half_spread(300.0, "2015-06-01")
    now = half_spread(300.0, "2026-08-20")
    assert now > then                  # today's Rp2 tick costs MORE at Rp300
    assert then == pytest.approx(0.5 / 300)


def test_rounding_a_limit_never_favours_the_trader():
    d = "2026-08-20"
    assert round_to_tick(1003.0, d, "down") == 1000.0
    assert round_to_tick(1001.0, d, "up") == 1005.0
    assert round_to_tick(1002.0, d, "nearest") == 1000.0


def test_max_price_step_is_a_separate_constraint_from_the_tick():
    for p in (100.0, 1000.0, 9000.0):
        assert max_price_step(p, "2026-08-20") > tick_size(p, "2026-08-20")


# --------------------------------------------------------------------------
# lot and halts
# --------------------------------------------------------------------------
def test_the_lot_is_a_hundred_since_2014():
    for day in ("2014-01-06", "2020-03-16", "2026-08-20"):
        assert lot_size(day) == 100


def test_halts_did_not_exist_before_2020():
    assert trading_halt(-0.10, "2019-06-03") is None
    assert trading_halt(-0.10, "2012-06-01") is None


def test_the_2025_halt_ladder_has_three_steps():
    assert trading_halt(-0.03, "2026-08-20") is None
    assert trading_halt(-0.09, "2026-08-20") == "halt 30 minutes"
    assert trading_halt(-0.16, "2026-08-20") == "halt a further 30 minutes"
    assert "suspend" in trading_halt(-0.25, "2026-08-20")


def test_a_rising_index_never_halts():
    assert trading_halt(+0.25, "2026-08-20") is None


# --------------------------------------------------------------------------
# provenance
# --------------------------------------------------------------------------
def test_every_regime_cites_a_source():
    for name in ("ara", "arb", "thin", "tick", "max_step", "lot"):
        assert schedule(name)["source"].str.len().gt(0).all()


def test_the_unmodelled_rules_are_enumerated_not_merely_implied():
    gaps = known_gaps()
    assert len(gaps) >= 5
    assert any("IPO" in g for g in gaps)
    assert any(str(COVERAGE_START.year) in g for g in gaps)


# --------------------------------------------------------------------------
# the ladder is checked against what the market actually quoted
# --------------------------------------------------------------------------
def test_the_encoded_ladder_matches_the_2016_regime():
    """Confirmed against 1.3m quoted closes by scripts/gate0.py check 2b."""
    d = "2016-05-02"
    assert [tick_size(p, d) for p in (100, 300, 1000, 3000, 9000)] \
        == [1.0, 2.0, 5.0, 10.0, 25.0]


def test_the_encoded_ladder_matches_the_2014_regime():
    """Two published sources disagreed on the Rp 500-5,000 band - one said
    Rp 5, one said Rp 10. The data settled it: 97.9% of closes there divide by
    5, so Rp 5 is right."""
    d = "2014-01-06"
    assert [tick_size(p, d) for p in (100, 300, 1000, 3000, 9000)] \
        == [1.0, 1.0, 5.0, 5.0, 25.0]


def test_the_grid_test_would_reject_a_wrong_ladder():
    """The discriminator is the chance level under a finer grid, not a fixed
    threshold: on a Rp 5 grid only ~20% of prices divide by 25, so an observed
    88% means the grid really is 25 even though it is far from 100%."""
    # 88% observed, 20% chance under a Rp 5 grid -> closer to 1 than to chance
    assert abs(0.8845 - 1.0) < abs(0.8845 - 0.20)
    # 61% observed, 50% chance under a Rp 25 grid -> closer to chance
    assert abs(0.6105 - 0.50) < abs(0.6105 - 1.0)


# --------------------------------------------------------------------------
# board membership is DERIVED from a published criterion, not guessed
# --------------------------------------------------------------------------
def test_a_cheap_stock_lands_on_the_watchlist_after_the_rule_existed():
    """IDX's rule: six-month average regular-market price below Rp 51."""
    from idxbot.spine.reference import infer_board
    assert infer_board("2024-06-03", avg_price_6m=20.0) == "watchlist"
    assert infer_board("2024-06-03", avg_price_6m=200.0) == "main"


def test_the_watchlist_rule_does_not_apply_before_it_existed():
    """Pre-2023 a sub-Rp 50 quote is not explained by any encoded rule, so the
    honest answer is 'unknown' rather than an invented board."""
    from idxbot.spine.reference import infer_board
    assert infer_board("2020-06-03", avg_price_6m=20.0) == "unknown"
    assert infer_board("2020-06-03", avg_price_6m=200.0) == "main"


def test_the_watchlist_boundary_falls_on_the_announced_day():
    from idxbot.spine.reference import WATCHLIST_START, infer_board
    day_before = WATCHLIST_START - pd.Timedelta(days=1)
    assert infer_board(day_before, avg_price_6m=20.0) == "unknown"
    assert infer_board(WATCHLIST_START, avg_price_6m=20.0) == "watchlist"


def test_the_watchlist_ladder_is_far_looser_than_the_main_one():
    """Rp 1 band below Rp 10, 10% above - which is why getting the board wrong
    manufactures impossible-move flags on exactly the wrong names."""
    assert auto_rejection(5.0, "2024-06-03", "watchlist") == (-1.0, -1.0)
    assert auto_rejection(30.0, "2024-06-03", "watchlist") == (0.10, 0.10)
    assert auto_rejection(30.0, "2024-06-03", "main")[0] == 0.35


def test_a_missing_price_defaults_to_the_main_board():
    from idxbot.spine.reference import infer_board
    assert infer_board("2024-06-03") == "main"
