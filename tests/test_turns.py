"""Tests for H37/H38 — turn detection and level placement.

The two ways this study could report a good number for a bad reason are a
detector scored without a matched null, and a per-bar level that leaks. Both
are pinned.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir,
                                "scripts"))

from turns import (OFFSETS, WINDOW, detectors, first_passage_level,  # noqa: E402
                   random_flips, swing_levels, turn_stats)


def _zig(n_legs=6, up=60, dn=40, amp=0.6):
    """A clean alternating path with known peaks."""
    p = [100.0]
    for i in range(n_legs):
        tgt = p[-1] * (1 + amp) if i % 2 == 0 else p[-1] * (1 - amp / 2)
        p.extend(np.linspace(p[-1], tgt, up if i % 2 == 0 else dn)[1:])
    return np.asarray(p, float)


# ================================================== the detectors themselves ==
def test_every_detector_fires_on_the_bar_the_state_turns_off():
    p = _zig()
    det = detectors(p)
    assert set(det) == {"hull55 rising", "hma21 over hma55", "close over EMA34",
                        "close over EMA50", "price>50>100>200"}
    for nm, fl in det.items():
        assert fl.dtype == bool and len(fl) == len(p), nm


def test_a_detector_never_fires_on_the_first_bar():
    """A flip is a CHANGE, so bar zero has nothing to change from. Firing there
    would credit the detector with a turn that has no prior state."""
    for fl in detectors(_zig()).values():
        assert not fl[0]


def test_a_monotone_rise_produces_no_flip_down():
    p = np.linspace(100.0, 400.0, 600)
    for nm, fl in detectors(p).items():
        assert not fl.any(), nm


def test_a_faster_line_flips_more_often_than_a_slower_one():
    """The recall/precision trade-off in its rawest form: EMA34 fires roughly
    twice as often as the EMA stack, which is why the null has to match the
    flip COUNT and not just exist."""
    rng = np.random.default_rng(0)
    p = 100.0 * np.exp(np.cumsum(rng.normal(0.0004, 0.02, 4000)))
    det = detectors(p)
    assert det["close over EMA34"].sum() > det["price>50>100>200"].sum()


# =========================================================== turn scoring =====
def test_a_flip_before_the_peak_does_not_count_as_catching_it():
    """THE LOOK-AHEAD GUARD OF THIS STUDY. Only flips strictly AFTER the peak
    can have been caused by it; crediting an earlier one would let a detector
    'catch' tops it fired before."""
    p = np.concatenate([np.linspace(100, 200, 50), np.linspace(200, 120, 50)])
    peaks = np.array([49])
    early = np.zeros(len(p), bool)
    early[40] = True
    assert turn_stats(p, peaks, early)["caught"] == 0
    late = np.zeros(len(p), bool)
    late[55] = True
    assert turn_stats(p, peaks, late)["caught"] == 1


def test_a_flip_outside_the_window_does_not_count():
    p = np.concatenate([np.linspace(100, 200, 50), np.linspace(200, 120, 200)])
    peaks = np.array([49])
    far = np.zeros(len(p), bool)
    far[49 + WINDOW + 5] = True
    assert turn_stats(p, peaks, far)["caught"] == 0


def test_give_back_is_measured_from_the_peak_not_from_the_entry():
    """The number that decides whether to act on a flip: how much of the top is
    already gone when it fires."""
    p = np.concatenate([np.linspace(100, 200, 50), np.linspace(200, 100, 50)])
    peaks = np.array([49])
    fl = np.zeros(len(p), bool)
    fl[74] = True                       # roughly halfway down the second leg
    st = turn_stats(p, peaks, fl)
    assert st["give"][0] == pytest.approx(1.0 - p[74] / p[49])
    assert 0.2 < st["give"][0] < 0.7


def test_precision_counts_a_flip_as_on_time_only_near_a_real_peak():
    p = np.concatenate([np.linspace(100, 200, 50), np.linspace(200, 120, 200)])
    peaks = np.array([49])
    fl = np.zeros(len(p), bool)
    fl[55] = True                       # on time
    fl[200] = True                      # nowhere near a peak
    st = turn_stats(p, peaks, fl)
    assert st["n_flips"] == 2 and st["on_time"] == 1


def test_the_null_matches_the_flip_count_exactly():
    """A DETECTOR THAT FLIPS OFTEN CATCHES EVERY TOP BY CONSTRUCTION. The null
    has to spend the same number of flips or the comparison is free."""
    rng = np.random.default_rng(1)
    for k in (5, 50, 500):
        assert random_flips(2000, k, rng).sum() == k


def test_the_null_is_reproducible_from_its_seed():
    a = random_flips(500, 40, np.random.default_rng(3))
    b = random_flips(500, 40, np.random.default_rng(3))
    assert np.array_equal(a, b)


# ================================================= per-bar level passage ======
def test_the_per_bar_level_is_strictly_forward():
    """first_passage_level takes a DIFFERENT level on every row, so the
    off-by-one is easier to get wrong than the fixed-multiple version."""
    p = np.array([100.0, 100.0, 130.0])
    lvl = np.full(3, 120.0)
    assert first_passage_level(p, lvl, 5, True)[0] == 2
    #  bar 2 is already above its own level; it must not count for itself
    assert first_passage_level(np.array([130.0, 100.0]), np.full(2, 120.0),
                               5, True)[0] == -1


def test_the_level_travels_with_the_row():
    """The whole reason this function exists: a support line is a different
    number on every bar, and using one bar's level for another would silently
    test a strategy nobody could have run."""
    p = np.array([100.0, 105.0, 110.0, 115.0])
    near = first_passage_level(p, np.array([104.0, 999.0, 999.0, 999.0]), 3, True)
    far = first_passage_level(p, np.array([114.0, 999.0, 999.0, 999.0]), 3, True)
    assert near[0] == 1 and far[0] == 3


def test_the_downside_direction_looks_down():
    """The same path answers both questions and they must not agree: a stop at
    80 is hit on bar 2, while a target at 130 is never reached."""
    p = np.array([100.0, 95.0, 79.0])
    assert first_passage_level(p, np.full(3, 80.0), 5, False)[0] == 2
    assert first_passage_level(p, np.full(3, 130.0), 5, True)[0] == -1


def test_a_window_running_past_the_data_is_censored():
    out = first_passage_level(np.full(10, 100.0), np.full(10, 120.0), 5, True)
    assert out[-1] == -2 and out[0] == -1


# ================================================== the levels themselves =====
def test_swing_levels_are_only_known_after_confirmation():
    """A21/A26's rule in a new place: a swing high is not a level until price
    has fallen from it, so the array must be NaN until the confirmation bar."""
    p = _zig()
    hi, lo = swing_levels(p)
    first = int(np.flatnonzero(np.isfinite(hi))[0])
    peak = int(np.argmax(p[:first + 1]))
    assert first > peak, "the level appeared before the peak was confirmed"


def test_swing_levels_step_and_never_look_ahead():
    p = _zig()
    hi, _ = swing_levels(p)
    fin = np.isfinite(hi)
    #  at every bar the recorded high must already have occurred
    for i in np.flatnonzero(fin)[::37]:
        assert hi[i] <= p[:i + 1].max() + 1e-9


def test_the_offsets_bracket_the_level_symmetrically():
    """0.00 has to be in the grid or 'at the level' is not one of the answers,
    and the grid has to be symmetric or the comparison is loaded."""
    assert 0.0 in OFFSETS
    assert sorted(OFFSETS) == list(OFFSETS)
    assert OFFSETS[0] == pytest.approx(-OFFSETS[-1])
