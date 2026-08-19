"""Tests for the leg painter and the signals derived from it.

One of these is a regression test for a bug that produced +87% CAGR out of thin
air: the holding state was lagged a week but the RANKING was not, so the book
selected on the same bar's price and then earned that bar's return. Every input
to a selection has to be lagged, not just the obvious one.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "scripts"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

from legpaint import smooth_state, technical, zigzag_labels   # noqa: E402
from paint_chart import live_segments, settle_curve           # noqa: E402
from turn_trader import reversal_state                        # noqa: E402


# --------------------------------------------------------------------------- #
# labels
# --------------------------------------------------------------------------- #
def test_labels_mark_up_and_down_legs():
    px = np.array([100.0, 130.0, 160.0, 120.0, 90.0, 130.0, 170.0])
    y = zigzag_labels(px, 0.15, drop_last=False)
    assert y[0] == 1.0            # rising into the first peak
    assert y[2] == 0.0            # falling away from it


def test_last_leg_is_unlabelled_by_default():
    px = np.array([100.0, 130.0, 160.0, 120.0, 90.0, 130.0, 170.0])
    assert np.isnan(zigzag_labels(px, 0.15)[-1])


def test_labels_never_use_bars_after_the_cut():
    """Truncation: the labelled region of a prefix must match the full series."""
    rng = np.random.default_rng(3)
    drift = np.tile(np.repeat([0.03, -0.03], 20), 8)
    px = 1000.0 * np.cumprod(1.0 + drift + rng.normal(0, 0.01, len(drift)))
    cut = 200
    part = zigzag_labels(px[:cut], 0.12, drop_last=True)
    full = zigzag_labels(px, 0.12, drop_last=False)
    ok = np.isfinite(part)
    assert ok.sum() > 20
    assert np.array_equal(part[ok], full[:cut][ok])


# --------------------------------------------------------------------------- #
# the painter
# --------------------------------------------------------------------------- #
def test_closed_legs_exclude_the_running_one():
    px = np.concatenate([np.linspace(100, 200, 25), np.linspace(200, 120, 25),
                         np.linspace(120, 190, 20)])
    closed, live_start = live_segments(px, 0.15)
    assert len(closed) >= 2
    assert live_start >= closed[-1][1]        # the live leg starts where closed end
    assert live_start < len(px) - 1


def test_settle_curve_improves_with_age():
    rng = np.random.default_rng(8)
    drift = np.tile(np.repeat([0.025, -0.025], 18), 9)
    px = 1000.0 * np.cumprod(1.0 + drift + rng.normal(0, 0.012, len(drift)))
    s = settle_curve(px, 0.12, (1, 4, 13), step=3)
    assert s[13] >= s[4] >= s[1]
    assert s[13] > 0.9                        # old bars stop repainting


def test_technical_features_are_trailing_only():
    rng = np.random.default_rng(11)
    idx = pd.date_range("2015-01-02", periods=300, freq="W-FRI")
    w = pd.Series(1000.0 * np.cumprod(1.0 + rng.normal(0.001, 0.03, 300)), index=idx)
    full = technical(w)
    part = technical(w.iloc[:180])
    common = [c for c in full.columns if c in part.columns]
    assert np.allclose(full[common].iloc[:180].to_numpy(float),
                       part[common].to_numpy(float), equal_nan=True)


# --------------------------------------------------------------------------- #
# signals
# --------------------------------------------------------------------------- #
def test_hysteresis_holds_inside_the_band():
    p = np.array([0.9, 0.52, 0.48, 0.52, 0.1])
    assert smooth_state(p, 0.55, 0.45).tolist() == [1, 1, 1, 1, 0]


def test_reversal_state_is_causal_under_truncation():
    rng = np.random.default_rng(5)
    px = 1000.0 * np.cumprod(1.0 + rng.normal(0.001, 0.03, 400))
    assert np.array_equal(reversal_state(px, 0.12, 0.12)[:250],
                          reversal_state(px[:250], 0.12, 0.12))


def test_selection_must_lag_every_input():
    """Regression: ranking on the CURRENT bar and earning that bar's return.

    Built so that the 'winner' each week is whichever name is about to rise.
    Ranking on the unlagged score must look impossibly good; ranking on the
    lagged score must not. If this test ever passes with an unlagged score, the
    +87% bug is back.
    """
    n = 120
    idx = pd.date_range("2015-01-02", periods=n, freq="W-FRI")
    rng = np.random.default_rng(2)
    R = pd.DataFrame(rng.normal(0.0, 0.05, (n, 6)),
                     index=idx, columns=list("ABCDEF"))
    W = 1000.0 * (1 + R).cumprod()

    def book(score: pd.DataFrame) -> float:
        pick = score.rank(axis=1, ascending=False) <= 2
        got = R.where(pick).sum(axis=1) / pick.sum(axis=1).replace(0, np.nan)
        return float((1 + got.fillna(0)).prod())

    cheating = book(R)                 # rank on this week's own return
    honest = book(R.shift(1))          # rank on last week's
    assert cheating > honest * 5, "the cheating book must be obviously inflated"
    assert honest < 5.0, "the lagged book must not be inflated"
