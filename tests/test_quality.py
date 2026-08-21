"""Tests for the price-spine quality gates.

Each detector here was written because a real defect was found in 843 tickers
of real IDX history, and each test encodes the specific thing that made the
first version of that detector wrong.

  STALE BARS are the important one and the least dramatic: 16.2% of the spine
  records no trading at all. A backtest that fills on one has bought from
  nobody; a return series that keeps one reports a real zero where there was no
  observation.

  DECIMAL SPIKES had to require BOTH a clean factor of ten AND a reversion,
  because a genuine 10x move exists and must not be deleted.

  LEVEL SHIFTS is the one that was actually wrong first time. A ratio alone
  flagged 79 "unadjusted splits" and most were penny stocks going from Rp 3 to
  Rp 2 - a ratio of 1.5 and ONE TICK. Requiring the move to be large in ticks
  as well cut it to 11 real candidates. The regression test for that is
  ``test_a_penny_stock_moving_one_tick_is_not_a_split``, and it is the most
  valuable test in this file.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

from idxbot.spine.quality import (MIN_SPLIT_TICKS, clean,          # noqa: E402
                                  decimal_spikes, level_shifts, locked_bars,
                                  report, stale_bars, tradeable)


def series(closes, volumes=None, start="2024-01-01", flat=True):
    n = len(closes)
    v = [1000.0] * n if volumes is None else volumes
    d = pd.bdate_range(start, periods=n)
    c = np.asarray(closes, dtype=float)
    return pd.DataFrame({
        "date": d, "open": c, "close": c,
        "high": c if flat else c * 1.01,
        "low": c if flat else c * 0.99, "volume": v})


# --------------------------------------------------------------------------
# stale bars
# --------------------------------------------------------------------------
def test_a_zero_volume_bar_is_stale():
    d = series([100.0] * 5, volumes=[10, 0, 10, 0, 10])
    assert list(stale_bars(d)) == [False, True, False, True, False]


def test_a_flat_bar_on_real_volume_is_not_stale():
    """Zero volume is the definition, not the flat shape."""
    d = series([100.0] * 3, volumes=[10, 10, 10])
    assert not stale_bars(d).any()


def test_a_bar_with_a_range_on_no_volume_is_still_stale():
    d = series([100.0, 101.0, 102.0], volumes=[10, 0, 10], flat=False)
    assert bool(stale_bars(d).iloc[1])


def test_missing_volume_counts_as_stale_not_as_traded():
    d = series([100.0] * 3, volumes=[10, np.nan, 10])
    assert bool(stale_bars(d).iloc[1])


# --------------------------------------------------------------------------
# decimal spikes
# --------------------------------------------------------------------------
def test_an_isolated_tenth_of_a_bar_is_a_spike():
    """The MAPI May-2018 shape."""
    d = series([820.0, 82.0, 815.0, 810.0])
    assert bool(decimal_spikes(d).iloc[1])


def test_an_isolated_ten_times_bar_is_a_spike():
    """The ELTY shape, same defect in the other direction."""
    d = series([500.0, 5000.0, 500.0, 500.0])
    assert bool(decimal_spikes(d).iloc[1])


def test_a_real_ten_fold_move_that_does_not_revert_is_kept():
    """A genuine 10x exists. Deleting it would be worse than the defect."""
    d = series([100.0, 1000.0, 1000.0, 1010.0])
    assert not decimal_spikes(d).any()


def test_an_ordinary_move_is_not_a_spike():
    d = series([100.0, 105.0, 103.0, 108.0])
    assert not decimal_spikes(d).any()


def test_a_v_shape_the_exchange_could_have_permitted_is_not_a_spike():
    """Reverting is not enough on its own.

    A -20% day followed by +25% is entirely legal inside the 35% band for a
    Rp 100 stock, so it is a real pair of days however symmetric it looks.
    """
    d = series([100.0, 80.0, 100.0, 100.0])
    assert not decimal_spikes(d).any()


def test_a_v_shape_the_exchange_could_NOT_have_permitted_is_a_spike():
    """-50% on a Rp 100 stock is outside every band. It cannot have happened."""
    d = series([100.0, 50.0, 100.0, 100.0])
    assert bool(decimal_spikes(d).iloc[1])


def test_a_penny_stock_moving_one_tick_and_back_is_not_a_spike():
    """The same false positive the split detector had, in the other detector.

    A stock at Rp 3 going to Rp 2 and back is one tick each way. It is a 1.5x
    ratio that reverts exactly, and it is an ordinary week on the acceleration
    board - whose ladder applies because the main board has a Rp 50 floor.
    """
    d = series([3.0, 2.0, 3.0, 3.0])
    assert not decimal_spikes(d).any()


def test_a_pre_coverage_spike_is_not_asserted():
    """Before 2014 the bands are unknown, so impossibility cannot be judged."""
    d = series([100.0, 50.0, 100.0, 100.0], start="2005-01-03")
    assert not decimal_spikes(d).any()


def test_the_first_and_last_bars_can_never_be_spikes():
    """They have no neighbour to revert against, so nothing can be claimed."""
    d = series([82.0, 820.0, 815.0, 82.0])
    assert not bool(decimal_spikes(d).iloc[0])
    assert not bool(decimal_spikes(d).iloc[-1])


# --------------------------------------------------------------------------
# level shifts - the detector that was wrong first time
# --------------------------------------------------------------------------
def test_a_penny_stock_moving_one_tick_is_not_a_split():
    """THE REGRESSION TEST. Rp 3 to Rp 2 is a ratio of 1.5 and one tick.

    Ratio alone reported 79 unadjusted corporate actions across the universe,
    of which most were this. The tick requirement cut it to 11.
    """
    d = series([3.0] * 6 + [2.0] * 6)
    assert level_shifts(d).empty


def test_a_real_split_on_a_dear_stock_is_found():
    """SCCO's actual 1:4 on 2024-02-01: Rp 9,975 to Rp 2,494, 744 ticks."""
    d = series([9975.0] * 6 + [2493.75] * 6)
    s = level_shifts(d)
    assert len(s) == 1
    assert s.iloc[0]["ratio"] == pytest.approx(4.0, rel=0.01)
    assert s.iloc[0]["ticks"] >= MIN_SPLIT_TICKS


def test_a_shift_that_reverts_is_not_a_split():
    """A split does not undo itself the following week."""
    d = series([9975.0] * 6 + [2493.0] * 3 + [9975.0] * 6)
    assert level_shifts(d).empty


def test_a_ratio_that_is_not_a_clean_split_is_ignored():
    d = series([9975.0] * 6 + [6500.0] * 6)      # ratio 1.53, not a split ratio
    s = level_shifts(d)
    assert s.empty or all(abs(r - 1.5) > 0.05 for r in s["ratio"])


def test_stale_bars_do_not_create_phantom_shifts():
    """A gap of non-trading days around a move must not become a split."""
    d = series([100.0] * 5 + [100.0] * 4 + [100.0] * 5,
               volumes=[10] * 5 + [0] * 4 + [10] * 5)
    assert level_shifts(d).empty


def test_a_pre_coverage_shift_is_skipped_rather_than_guessed():
    """Before 2014 the tick ladder is unknown, so the tick test cannot run.

    ELTY produced seven false 'splits' in 2003 when this returned infinity for
    an unknown tick and therefore always passed the size test.
    """
    d = series([500.0] * 6 + [50.0] * 6, start="2003-01-06")
    assert level_shifts(d).empty


# --------------------------------------------------------------------------
# locked bars and the combined mask
# --------------------------------------------------------------------------
def test_a_limit_down_bar_is_locked():
    # 15% ARB regime: 1000 -> 850, pinned all day
    d = pd.DataFrame({
        "date": pd.to_datetime(["2026-08-19", "2026-08-20"]),
        "open": [1000.0, 850.0], "high": [1000.0, 850.0],
        "low": [1000.0, 850.0], "close": [1000.0, 850.0],
        "volume": [1000.0, 1000.0]})
    assert bool(locked_bars(d).iloc[1])


def test_the_first_bar_is_never_locked_because_there_is_no_reference():
    d = series([1000.0, 1000.0, 1000.0])
    assert not bool(locked_bars(d).iloc[0])


def test_tradeable_excludes_stale_and_spikes_together():
    d = series([820.0, 82.0, 815.0, 810.0], volumes=[10, 10, 0, 10])
    t = tradeable(d)
    assert not bool(t.iloc[1])          # spike
    assert not bool(t.iloc[2])          # stale
    assert bool(t.iloc[3])


def test_clean_drops_stale_by_default_and_can_keep_them():
    d = series([100.0] * 5, volumes=[10, 0, 10, 0, 10])
    assert len(clean(d)) == 3
    assert len(clean(d, drop_stale=False)) == 5


def test_clean_flags_survive_when_nothing_is_dropped():
    d = series([100.0] * 5, volumes=[10, 0, 10, 0, 10])
    c = clean(d, drop_stale=False)
    assert c["stale"].sum() == 2
    assert set(["stale", "spike", "locked", "tradeable"]).issubset(c.columns)


def test_clean_returns_a_sorted_frame():
    d = series([100.0, 101.0, 102.0]).iloc[::-1]
    assert clean(d)["date"].is_monotonic_increasing


# --------------------------------------------------------------------------
# the summary
# --------------------------------------------------------------------------
def test_the_report_counts_what_it_found():
    d = series([820.0, 82.0, 815.0, 810.0, 810.0],
               volumes=[10, 10, 0, 10, 10])
    r = report(d, "TEST")
    assert r["ticker"] == "TEST" and r["bars"] == 5
    assert r["stale"] == 1 and r["spikes"] == 1
    assert 0.0 < r["tradeable_pct"] < 1.0


def test_the_report_of_nothing_does_not_crash():
    assert report(pd.DataFrame(), "EMPTY")["bars"] == 0


# --------------------------------------------------------------------------
# the tick grid
#
# Every price IDX ever printed is an exact multiple of that day's fraksi
# harga. A price that is not was never traded - it is arithmetic a vendor did.
# The tests below encode the two things that made the first two versions of
# this wrong, both of them the same mistake: treating ON-grid as evidence of
# anything. It is not. The test is one-sided.
# --------------------------------------------------------------------------
def test_the_tick_grid_matches_the_scalar_lookup_everywhere():
    """The vectorised grid is an optimisation, so it must not be a second
    implementation with its own opinions."""
    from idxbot.spine.quality import tick_grid
    from idxbot.spine.reference import tick_size
    rng = np.random.default_rng(0)
    px = rng.uniform(1.0, 20000.0, 500)
    dt = (pd.Timestamp("2005-01-03")
          + pd.to_timedelta(rng.integers(0, 7800, 500), "D"))
    assert np.allclose(tick_grid(px, dt),
                       [tick_size(a, b) for a, b in zip(px, dt)])


def test_a_traded_price_is_on_the_grid_and_a_quarter_of_it_is_not():
    """The SCCO shape in miniature: Rp 10,175 is a legal Rp 25 tick, and a
    vendor's quarter of it, Rp 2,543.75, is not a legal Rp 10 one."""
    from idxbot.spine.quality import off_tick
    d = series([10175.0, 2543.75])
    assert list(off_tick(d)) == [False, True]


def test_a_date_outside_the_tick_schedule_is_not_called_a_defect():
    """No rule encoded is an absence, not evidence. Before 2005 the ladder is
    unknown and anything else would manufacture defects across two decades."""
    from idxbot.spine.quality import off_tick
    d = series([1234.567], start="1998-01-05")
    assert not off_tick(d).any()


def test_the_off_grid_rate_is_a_lower_bound_not_an_estimate():
    """PTBA's whole pre-2017 history is divided by 5 and a large part of it
    lands on the grid regardless - every session the real price was a multiple
    of Rp 50. So a rate below 1.0 on a fully adjusted series is correct
    behaviour, and reading it as "the rest is clean" is the error."""
    from idxbot.spine.quality import off_grid_rate
    # 12,050/5 = 2,410 lands on the Rp 10 grid; 12,075/5 = 2,415 does not.
    d = series([2410.0, 2415.0, 2410.0, 2415.0])
    assert off_grid_rate(d) == pytest.approx(0.5)


def test_a_whole_number_back_adjustment_is_invisible_to_this_test():
    """Stated as a test because it is a real blind spot, not a bug. MAPI's 10
    and ELTY's proposed 10 leave every price on the grid."""
    from idxbot.spine.quality import off_grid_rate
    d = series([10000.0, 10250.0, 9750.0])
    assert off_grid_rate(d) == 0.0
    assert off_grid_rate(series([1000.0, 1025.0, 975.0])) == 0.0


# --------------------------------------------------------------------------
# islands
# --------------------------------------------------------------------------
def _island_frame():
    """120 clean bars, then 8 quartered ones, then 120 clean bars.

    The quartered prices step by Rp 50 so that every quarter lands on .25 or
    .75 and is off the Rp 10 grid. Stepping by 25 would put Rp 2,500 back ON
    it, which is the one-sidedness this whole module is about - and it silently
    made the island seven bars long instead of eight.
    """
    good = [10000.0 + 25 * (i % 8) for i in range(120)]
    bad = [(10025.0 + 50 * i) / 4 for i in range(8)]
    return series(good + bad + good, start="2015-01-01")


def test_an_island_is_found_with_its_boundaries_exact():
    from idxbot.spine.quality import adjustment_islands
    isl = adjustment_islands(_island_frame())
    assert len(isl) == 1
    assert isl.iloc[0]["bars"] == 8


def test_a_stale_bar_cannot_make_an_island():
    """WIKA was suspended for six months and its stale quote re-marked once
    for a rights issue. That is one forward-filled number, not a ten-session
    vendor adjustment, and counting it made exactly that claim."""
    from idxbot.spine.quality import adjustment_islands
    d = _island_frame()
    d.loc[120:127, "volume"] = 0.0
    assert adjustment_islands(d).empty


def test_a_defect_in_the_last_weeks_of_the_series_is_still_visible():
    """SINI's island is eight weeks from the end, and it is the one that would
    be traded on. Demanding a full clean flank on both sides hid it."""
    from idxbot.spine.quality import adjustment_islands
    good = [10000.0 + 25 * (i % 8) for i in range(120)]
    bad = [(10025.0 + 50 * i) / 4 for i in range(8)]
    d = series(good + bad + good[:10], start="2015-01-01")
    assert len(adjustment_islands(d)) == 1


def test_an_island_alone_is_not_reported_as_a_defect():
    """The conjunction with level_shifts is what took the sweep from 2,760
    findings to 3. Here the level barely moves, so there is no break."""
    from idxbot.spine.quality import adjustment_islands, suspect_islands
    good = [10000.0] * 120
    bad = [10002.5] * 8            # off the Rp 25 grid, but no shift at all
    d = series(good + bad + good, start="2015-01-01")
    assert len(adjustment_islands(d)) == 1
    assert suspect_islands(d).empty


# --------------------------------------------------------------------------
# proving a factor
# --------------------------------------------------------------------------
def test_the_announced_factor_lands_every_price_on_the_grid():
    from idxbot.spine.quality import factor_fits
    d = _island_frame()
    f = factor_fits(d, d["date"].iloc[120], d["date"].iloc[127], 0.25)
    assert f["price_exact"] == 1.0
    assert f["max_grid_error"] == pytest.approx(0.0, abs=1e-9)


def test_a_factor_two_per_cent_wrong_lands_essentially_none_of_them():
    """This is what makes the grid a TEST rather than a description. The
    factor comes from an announcement; the grid then gets an independent vote,
    and a wrong factor scores at chance."""
    from idxbot.spine.quality import factor_fits
    d = _island_frame()
    f = factor_fits(d, d["date"].iloc[120], d["date"].iloc[127], 0.25 * 1.02)
    assert f["price_exact"] < 0.2


def test_the_volume_test_tolerates_the_vendor_rounding_but_not_a_wrong_factor():
    """The vendor stores adjusted volume as a whole share, so recovering the
    true figure through a factor of 0.67 is only good to about one share.
    Demanding exactness failed SINI on 0.22 shares in 3.4 million."""
    from idxbot.spine.quality import factor_fits
    d = _island_frame()
    d.loc[120:127, "volume"] = 501652.0     # 338,100 shares / 0.673973, rounded
    ok = factor_fits(d, d["date"].iloc[120], d["date"].iloc[127], 246 / 365)
    assert ok["volume_exact"] == 1.0
    bad = factor_fits(d, d["date"].iloc[120], d["date"].iloc[127], 0.9)
    assert bad["volume_exact"] == 0.0
