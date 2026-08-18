"""Tests for the learned turn model.

The label needs the future, which makes this the easiest script in the
repository to make lie. So the tests are almost entirely about leakage:

  * the unfinished leg at the right-hand edge must be unlabelled, because its
    direction is not knowable at the cut;
  * labels must be derivable from the training slice alone;
  * hysteresis must not flip inside its own band, which is the whole reason it
    exists;
  * the portfolio must select on the PREVIOUS bar's signal and pay costs.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "scripts"))

from turn_ml import name_features, portfolio, positions, zigzag_state  # noqa: E402


# --------------------------------------------------------------------------- #
# labels
# --------------------------------------------------------------------------- #
def test_last_leg_is_unlabelled():
    """At the right-hand edge the current leg has not finished, so it is NaN."""
    px = np.array([100.0, 80.0, 60.0, 90.0, 120.0, 150.0])
    y = zigzag_state(px, 0.20, drop_last_leg=True)
    assert np.isnan(y[-1])
    # and the finished down-leg before it is labelled 0
    assert y[0] == 0.0


def test_keeping_the_last_leg_labels_it():
    px = np.array([100.0, 80.0, 60.0, 90.0, 120.0, 150.0])
    kept = zigzag_state(px, 0.20, drop_last_leg=False)
    dropped = zigzag_state(px, 0.20, drop_last_leg=True)
    assert np.isfinite(kept).sum() > np.isfinite(dropped).sum()


def test_up_and_down_legs_get_opposite_labels():
    px = np.array([100.0, 150.0, 200.0, 150.0, 100.0, 150.0, 200.0])
    y = zigzag_state(px, 0.20, drop_last_leg=False)
    assert y[0] == 1.0            # rising into the first peak
    assert y[2] == 0.0            # falling away from it


def test_labels_come_from_the_training_slice_only():
    """A pivot only visible with post-cut data must not label a pre-cut row.

    This is the protocol the whole result depends on: the training window is
    labelled by recomputing the zigzag on ``prices[:cut]``, so nothing after the
    cut can reach back into a training label.
    """
    # a series that actually oscillates, so several legs complete before the cut
    rng = np.random.default_rng(4)
    drift = np.tile(np.repeat([0.02, -0.02], 25), 8)
    px = 1000.0 * np.cumprod(1.0 + drift + rng.normal(0, 0.01, len(drift)))
    cut = 250
    train_only = zigzag_state(px[:cut], 0.20, drop_last_leg=True)
    assert len(train_only) == cut
    # the labelled region must end strictly before the cut, leaving the
    # unfinished leg unlabelled
    labelled = np.flatnonzero(np.isfinite(train_only))
    assert len(labelled) > 0
    assert labelled[-1] < cut - 1


def test_no_labels_when_nothing_moves():
    px = np.full(100, 1000.0)
    assert not np.isfinite(zigzag_state(px, 0.20)).any()


# --------------------------------------------------------------------------- #
# hysteresis
# --------------------------------------------------------------------------- #
def test_hysteresis_does_not_flip_inside_the_band():
    """A probability wandering between the two thresholds must hold position."""
    p = np.array([0.9, 0.55, 0.45, 0.55, 0.45, 0.55])
    st = positions(p, 0.6, 0.4)
    assert st.tolist() == [1, 1, 1, 1, 1, 1]
    assert int(np.abs(np.diff(st)).sum()) == 0


def test_hysteresis_enters_and_exits_at_its_thresholds():
    p = np.array([0.5, 0.61, 0.5, 0.39, 0.5, 0.61])
    st = positions(p, 0.6, 0.4)
    assert st.tolist() == [0, 1, 1, 0, 0, 1]


def test_hysteresis_starts_flat():
    assert positions(np.array([0.5, 0.5, 0.5]), 0.6, 0.4).sum() == 0


def test_nan_probability_holds_the_previous_state():
    p = np.array([0.9, np.nan, np.nan, 0.1])
    st = positions(p, 0.6, 0.4)
    assert st.tolist() == [1, 1, 1, 0]


# --------------------------------------------------------------------------- #
# cross-sectional use
# --------------------------------------------------------------------------- #
def _panel(probs, prices):
    rows = []
    dates = pd.bdate_range("2020-01-03", periods=len(prices["A"]), freq="W-FRI")
    for t in prices:
        for i, d in enumerate(dates):
            rows.append({"date": d, "ticker": t, "px": prices[t][i],
                         "prob": probs[t][i]})
    return pd.DataFrame(rows)


def test_portfolio_selects_on_the_previous_bar():
    """A signal that spikes on the bar it would profit from must not be tradeable."""
    # B doubles on the last bar; its probability only rises ON that bar
    prices = {"A": [100.0, 100.0, 100.0], "B": [100.0, 100.0, 200.0]}
    probs = {"A": [0.9, 0.9, 0.0], "B": [0.0, 0.0, 1.0]}
    g, _, _ = portfolio(_panel(probs, prices), top_n=1, every=1, cost=0.0)
    assert g == pytest.approx(1.0)      # it held A, not B


def test_portfolio_charges_costs_on_churn():
    prices = {"A": [100.0, 110.0, 120.0], "B": [100.0, 110.0, 120.0]}
    probs = {"A": [1.0, 0.0, 1.0], "B": [0.0, 1.0, 0.0]}
    free, _, _ = portfolio(_panel(probs, prices), top_n=1, every=1, cost=0.0)
    paid, _, _ = portfolio(_panel(probs, prices), top_n=1, every=1, cost=0.05)
    assert paid < free


def test_portfolio_holding_everything_matches_the_average():
    prices = {"A": [100.0, 200.0], "B": [100.0, 100.0]}
    probs = {"A": [1.0, 1.0], "B": [1.0, 1.0]}
    g, _, _ = portfolio(_panel(probs, prices), top_n=2, every=1, cost=0.0)
    assert g == pytest.approx(1.5)      # mean of +100% and 0%


# --------------------------------------------------------------------------- #
# features
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("cut", [80, 150])
def test_features_are_trailing_only(cut):
    """Truncating the series must not change any earlier feature value."""
    rng = np.random.default_rng(8)
    idx = pd.date_range("2010-01-01", periods=250, freq="W-FRI")
    w = pd.Series(1000.0 * np.cumprod(1.0 + rng.normal(0.001, 0.03, 250)), index=idx)
    full = name_features(w)
    part = name_features(w.iloc[:cut])
    common = [c for c in full.columns if c in part.columns]
    a = full[common].iloc[:cut].to_numpy(float)
    b = part[common].to_numpy(float)
    assert np.allclose(a, b, equal_nan=True)


# --------------------------------------------------------------------------- #
# the zigzag itself — it defines the labels, so a bug here poisons everything
# --------------------------------------------------------------------------- #
def test_zigzag_finds_every_turn_of_a_saw_tooth():
    """The regression that exposed the original bug.

    100 -> 200 -> 100 -> 200 -> 100 has four turns. An earlier version returned
    two pivots, because its single running extreme chased the price in both
    directions and no threshold was ever exceeded relative to a fixed extreme.
    """
    from swing_accuracy import zigzag
    px = np.concatenate([np.linspace(100, 200, 20), np.linspace(200, 100, 20),
                         np.linspace(100, 200, 20), np.linspace(200, 100, 20)])
    piv = zigzag(px, 0.20)
    assert len(piv) >= 5                       # start, 3 interior turns, end
    # the interior pivots must sit at the actual extremes
    assert px[piv[1]] == pytest.approx(200, abs=6)
    assert px[piv[2]] == pytest.approx(100, abs=6)
    assert px[piv[3]] == pytest.approx(200, abs=6)


def test_zigzag_alternates_between_highs_and_lows():
    from swing_accuracy import zigzag, legs
    px = np.concatenate([np.linspace(100, 200, 15), np.linspace(200, 120, 15),
                         np.linspace(120, 260, 15), np.linspace(260, 150, 15)])
    lg = legs(px, zigzag(px, 0.20))
    signs = [r > 0 for _a, _b, r in lg]
    assert all(a != b for a, b in zip(signs, signs[1:]))   # strictly alternating


def test_zigzag_ignores_moves_below_the_threshold():
    from swing_accuracy import zigzag
    rng = np.random.default_rng(2)
    px = 1000.0 * (1.0 + 0.02 * np.sin(np.linspace(0, 40, 300))
                   + 0.002 * rng.normal(size=300))
    # nothing moves 20%, so the only pivots are the endpoints
    assert zigzag(px, 0.20) == [0, 299]


def test_zigzag_threshold_is_monotone():
    """A bigger threshold can never find more turns."""
    from swing_accuracy import zigzag
    rng = np.random.default_rng(6)
    px = 1000.0 * np.cumprod(1.0 + rng.normal(0.0005, 0.035, 600))
    counts = [len(zigzag(px, t)) for t in (0.10, 0.20, 0.30, 0.40)]
    assert counts == sorted(counts, reverse=True)
