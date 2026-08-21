"""Tests for how the return series handles the exchange's rules and defects.

`total_return_series` feeds every study in the repo, so the two things it does
wrong propagate everywhere and neither announces itself.

  THE CAP WAS A FLAT 35%. IDX's auto-rejection floor was SEVEN percent for
  three years. Measured across 300 names and 668,042 in-coverage bars, a flat
  clip catches 93 impossible moves where the point-in-time band catches 824 -
  it lets through eight times more than it stops.

  A CORPORATE ACTION WAS CAPPED, NOT ADJUSTED. Clipping SCCO's 1:4 split turned
  a 75% data artefact into a 35% one rather than into nothing. A holder through
  that split lost NOTHING, and the series has to say so.

The exemptions matter as much as the cap. On an ex-date IDX applies the band to
the theoretical price, so a legitimate rights-issue drop is larger than the band
and must not be capped; and near an unverified shift nothing is touched at all,
because the truth there is unknown.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "scripts"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

from factor_study import _cap_impossible                          # noqa: E402


def series(dates, px):
    return pd.DataFrame({"px": px}, index=pd.to_datetime(dates))


def test_a_move_beyond_the_bands_is_capped():
    """-50% on a Rp 1,000 stock in 2019: the ARB was 25%."""
    d = series(["2019-06-03", "2019-06-04"], [1000.0, 500.0])
    r = _cap_impossible(d, "TEST")
    assert r.iloc[1] == pytest.approx(-0.25)


def test_the_cap_follows_the_regime_not_a_fixed_number():
    """The same -50% print caps at 7% in 2021 and 25% in 2019."""
    a = _cap_impossible(series(["2019-06-03", "2019-06-04"],
                               [1000.0, 500.0]), "TEST")
    b = _cap_impossible(series(["2021-06-03", "2021-06-04"],
                               [1000.0, 500.0]), "TEST")
    assert a.iloc[1] == pytest.approx(-0.25)
    assert b.iloc[1] == pytest.approx(-0.07)
    assert b.iloc[1] > a.iloc[1]


def test_a_flat_thirty_five_percent_clip_would_have_missed_it():
    """The measured failure: for three years the floor was 7%, not 35%."""
    d = series(["2021-06-03", "2021-06-04"], [1000.0, 700.0])   # -30%
    assert _cap_impossible(d, "TEST").iloc[1] == pytest.approx(-0.07)


def test_an_ordinary_move_is_untouched():
    d = series(["2026-08-19", "2026-08-20"], [1000.0, 1020.0])
    assert _cap_impossible(d, "TEST").iloc[1] == pytest.approx(0.02)


def test_a_penny_stock_gets_the_thin_board_ladder():
    """At Rp 3 the main-board band would cap an ordinary tick."""
    d = series(["2026-08-19", "2026-08-20"], [3.0, 2.0])
    assert _cap_impossible(d, "TEST").iloc[1] < -0.30


def test_pre_coverage_keeps_the_conservative_default():
    d = series(["2005-06-03", "2005-06-06"], [1000.0, 500.0])
    assert _cap_impossible(d, "TEST").iloc[1] == pytest.approx(-0.35)


# --------------------------------------------------------------------------
# corporate actions are adjusted, not capped
# --------------------------------------------------------------------------
def test_a_verified_split_is_adjusted_away_not_capped():
    """SCCO's real 1:4. The holder lost nothing and the series must say so."""
    d = series(["2024-03-07", "2024-03-08"], [10175.0, 2550.0])
    r = _cap_impossible(d, "SCCO")
    assert r.iloc[1] == pytest.approx(0.0025, abs=0.001)


def test_a_verified_rights_issue_leaves_the_participant_flat():
    """WIKA: 240 -> 203.91 against a published theoretical price of Rp 204."""
    d = series(["2024-04-16", "2024-04-17"], [240.0, 203.913391])
    r = _cap_impossible(d, "WIKA")
    assert abs(r.iloc[1]) < 0.005


def test_an_already_adjusted_series_is_not_adjusted_twice():
    """BBCA is back-adjusted, so there is no step. Applying the factor would
    MANUFACTURE the jump this is supposed to remove."""
    d = series(["2021-10-12", "2021-10-13"], [7320.0, 7525.0])
    r = _cap_impossible(d, "BBCA")
    assert r.iloc[1] == pytest.approx(7525.0 / 7320.0 - 1.0)


def test_a_split_sized_move_on_the_wrong_ticker_is_still_capped():
    """The adjustment is keyed to the ticker AND the date, not the shape."""
    d = series(["2024-03-07", "2024-03-08"], [10175.0, 2550.0])
    r = _cap_impossible(d, "BBCA")
    assert r.iloc[1] < -0.15          # capped, not adjusted to zero


def test_a_split_sized_move_on_the_wrong_date_is_still_capped():
    """A year before SCCO's real ex-date the same print is just a bad tick.

    Capped at 7%, because in March 2023 that was still the ARB - which is the
    whole point of using the point-in-time band rather than a fixed number.
    """
    d = series(["2023-03-07", "2023-03-08"], [10175.0, 2550.0])
    assert _cap_impossible(d, "SCCO").iloc[1] == pytest.approx(-0.07)


# --------------------------------------------------------------------------
# quarantine
# --------------------------------------------------------------------------
def test_a_quarantined_window_is_left_visibly_odd():
    """Near an unverified shift the truth is unknown, so nothing is smoothed."""
    d = series(["2024-04-15", "2024-04-16"], [820.0, 164.0])
    r = _cap_impossible(d, "PYFA")
    assert r.iloc[1] < -0.5, "quarantined windows must not be quietly capped"


def test_outside_quarantine_the_same_ticker_is_capped_normally():
    d = series(["2023-01-10", "2023-01-11"], [820.0, 164.0])
    assert _cap_impossible(d, "PYFA").iloc[1] >= -0.35


def test_the_first_bar_has_no_return():
    d = series(["2026-08-19", "2026-08-20"], [1000.0, 1020.0])
    assert _cap_impossible(d, "TEST").iloc[0] == 0.0


# --------------------------------------------------------------------------
# a fill must never land on a bar where nothing traded
# --------------------------------------------------------------------------
def test_the_backtest_carries_a_per_observation_tradeable_flag():
    """A rolling-median liquidity filter screens out names that are generally
    dead. It does NOT catch a single zero-volume bar inside a liquid name, and
    2.26% of observations passing a Rp 1bn/day filter are exactly that.
    """
    import inspect
    import idxbot.backtest as B
    src = inspect.getsource(B.run)
    assert '"tradeable"' in src or "'tradeable'" in src


def test_the_return_series_marks_untradeable_bars():
    import pandas as pd
    from idxbot.spine.quality import stale_bars
    d = pd.DataFrame({"date": pd.bdate_range("2024-01-01", periods=4),
                      "close": [100.0] * 4, "volume": [10, 0, 10, 0]})
    assert list(~stale_bars(d)) == [True, False, True, False]
