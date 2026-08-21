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


def test_a_v_shaped_but_not_decade_move_is_not_a_spike():
    """Reverting is not enough on its own - halving and doubling is a real day."""
    d = series([100.0, 50.0, 100.0, 100.0])
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
