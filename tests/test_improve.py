"""Tests for the improvement attempt.

This script exists to resist a specific temptation: a grid always has a best
cell, and reporting that cell is how a backtest lies. The tests below pin the
machinery that stops it - train/test separation, and a random-band control that
the fitted choice has to beat before "improvement" can be claimed.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "scripts"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

from improve import score, split_test                            # noqa: E402


def walk(n=1200, seed=1, drift=0.0003, vol=0.02):
    rng = np.random.default_rng(seed)
    return pd.Series(100 * np.cumprod(1 + rng.normal(drift, vol, n)),
                     index=pd.bdate_range("2015-01-01", periods=n))


GRID = [0.10, 0.15, 0.22, 0.30]


def test_score_returns_the_edge_over_the_null_not_over_hold():
    from method_review import measure, resample
    s = walk()
    got = score(s, 0.12, "weekly")
    want = measure(resample(s, "weekly"), 0.12)
    assert np.isclose(got, want["edge"])
    assert not np.isclose(got, want["log"] - want["hold"])


def test_score_is_none_on_a_series_too_short_to_measure():
    assert score(walk(n=40), 0.12, "weekly") is None


def test_split_test_chooses_from_the_grid():
    rng = np.random.default_rng(0)
    out = split_test(walk(), GRID, "weekly", rng)
    assert out is not None
    assert out["fitted_band"] in GRID
    assert out["coin_band"] in GRID


def test_the_fitted_band_is_chosen_on_the_train_half_only():
    """Rewriting the TEST half must not change which band was selected."""
    s = walk(seed=4)
    a = split_test(s, GRID, "weekly", np.random.default_rng(0))
    tampered = s.copy()
    cut = len(s) // 2
    tampered.iloc[cut:] = tampered.iloc[cut - 1] * np.cumprod(
        1 + np.random.default_rng(99).normal(0.01, 0.05, len(s) - cut))
    b = split_test(tampered, GRID, "weekly", np.random.default_rng(0))
    assert a is not None and b is not None
    assert a["fitted_band"] == b["fitted_band"]


def test_the_out_of_sample_score_does_change_with_the_test_half():
    """Sanity check on the test above: the SCORE must react even though the
    CHOICE does not, otherwise the split is not being applied at all."""
    s = walk(seed=6)
    a = split_test(s, GRID, "weekly", np.random.default_rng(0))
    tampered = s.copy()
    cut = len(s) // 2
    tampered.iloc[cut:] = tampered.iloc[cut - 1] * np.cumprod(
        1 + np.random.default_rng(7).normal(-0.005, 0.05, len(s) - cut))
    b = split_test(tampered, GRID, "weekly", np.random.default_rng(0))
    assert not np.isclose(a["fitted_oos"], b["fitted_oos"])


def test_a_random_control_is_always_produced():
    out = split_test(walk(), GRID, "weekly", np.random.default_rng(3))
    assert "coin_oos" in out and np.isfinite(out["coin_oos"])


def test_the_incumbent_is_scored_on_the_same_test_half():
    out = split_test(walk(), GRID, "weekly", np.random.default_rng(3))
    assert "base_oos" in out and np.isfinite(out["base_oos"])


def test_split_test_refuses_a_series_too_short_to_halve():
    assert split_test(walk(n=400), GRID, "weekly", np.random.default_rng(0)) is None


def test_the_coin_band_actually_varies_with_the_seed():
    outs = {split_test(walk(), GRID, "weekly",
                       np.random.default_rng(k))["coin_band"] for k in range(12)}
    assert len(outs) > 1, "the random control is not random"
