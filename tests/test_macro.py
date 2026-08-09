"""Macro alignment, and the two ways it could leak the future.

Macro series publish on their own calendars, which creates exactly the gaps
where look-ahead creeps in: a back-fill imports tomorrow's number into today,
and a full-sample percentile ranks today against years that have not happened.
Both are silent. Both are tested here.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

from idxbot.macro import (  # noqa: E402
    MAX_STALE_DAYS,
    MacroPanel,
    _trailing_rank,
    align_to,
    features,
    render,
)


def _panel(values, start="2020-01-01", col="IHSG"):
    idx = pd.date_range(start, periods=len(values), freq="D")
    return MacroPanel(frame=pd.DataFrame({col: values}, index=idx),
                      available=[col])


# --------------------------------------------------------------------------
# alignment must never reach forward
# --------------------------------------------------------------------------

def test_a_decision_date_inherits_the_previous_macro_value_not_the_next():
    """The core safety property of the whole module."""
    f = pd.DataFrame({"x": [1.0, 99.0]},
                     index=pd.to_datetime(["2020-01-01", "2020-01-10"]))
    out = align_to(f, [pd.Timestamp("2020-01-05")])
    assert out["x"].iloc[0] == pytest.approx(1.0)   # not 99.0


def test_a_date_before_any_publication_gets_nothing_rather_than_a_guess():
    f = pd.DataFrame({"x": [5.0]}, index=pd.to_datetime(["2020-06-01"]))
    out = align_to(f, [pd.Timestamp("2020-01-01")])
    assert np.isnan(out["x"].iloc[0])


def test_alignment_preserves_the_requested_dates_in_order():
    f = pd.DataFrame({"x": [1.0, 2.0, 3.0]},
                     index=pd.date_range("2020-01-01", periods=3, freq="D"))
    dates = [pd.Timestamp("2020-01-03"), pd.Timestamp("2020-01-02")]
    out = align_to(f, dates)
    assert list(out.index) == sorted(pd.to_datetime(dates))


def test_align_to_handles_empty_features():
    assert align_to(pd.DataFrame(), [pd.Timestamp("2020-01-01")]).empty


# --------------------------------------------------------------------------
# trailing rank must not see the future
# --------------------------------------------------------------------------

def test_trailing_rank_scores_against_the_past_only():
    """A value that is the highest so far ranks top, even if later values exceed it."""
    s = pd.Series(np.arange(300, dtype=float))
    r = _trailing_rank(s, window=100, min_periods=50)
    # A monotonically rising series is always its own trailing maximum.
    assert r.dropna().max() == pytest.approx(1.0)
    assert r.iloc[-1] == pytest.approx(1.0)


def test_trailing_rank_is_undefined_before_enough_history():
    s = pd.Series(np.arange(100, dtype=float))
    r = _trailing_rank(s, window=500, min_periods=200)
    assert r.isna().all()


def test_trailing_rank_differs_from_a_full_sample_rank():
    """If these matched, the look-ahead fix would be doing nothing."""
    rng = np.random.default_rng(0)
    s = pd.Series(rng.normal(size=800))
    trailing = _trailing_rank(s, window=200, min_periods=100).dropna()
    full = s.rank(pct=True).loc[trailing.index]
    assert not np.allclose(trailing.values, full.values)


# --------------------------------------------------------------------------
# features
# --------------------------------------------------------------------------

def test_features_are_empty_for_an_empty_panel():
    assert features(MacroPanel()).empty


def test_ihsg_trend_flags_are_computed_from_a_trailing_average():
    rising = _panel(np.linspace(100, 300, 400))
    f = features(rising)
    assert "ihsg_above_200d" in f
    # A steadily rising series sits above its own trailing mean.
    assert f["ihsg_above_200d"].dropna().iloc[-1] == 1.0


def test_missing_series_are_skipped_rather_than_faked():
    # Only IHSG present: rupiah/oil features must be absent, not zero-filled.
    f = features(_panel(np.linspace(100, 200, 300)))
    assert "idr_60d" not in f
    assert "oil_60d" not in f


def test_stale_limit_is_a_week_not_forever():
    """A series that stops reporting must go NaN, not repeat indefinitely."""
    assert MAX_STALE_DAYS == 7


# --------------------------------------------------------------------------
# render
# --------------------------------------------------------------------------

def test_render_labels_the_foreign_proxy_as_a_proxy():
    """It must never read as a measurement of foreign broker flow."""
    idx = pd.date_range("2020-01-01", periods=3, freq="D")
    feats = pd.DataFrame({"foreign_appetite": [0.5, 0.6, 0.7]}, index=idx)
    text = render(_panel([100.0, 101.0, 102.0]), feats)
    assert "PROXY" in text.upper()
    assert "not a measurement of foreign flow" in text


def test_render_survives_an_empty_panel():
    assert "No macro series retrieved" in render(MacroPanel(), pd.DataFrame())
