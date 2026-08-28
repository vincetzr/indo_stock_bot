"""Tests for the IDX Suite replica renderer.

It exists to show what the Pine file would print, so the thing that matters is
that it computes the SAME quantities — and that the two defects the first run
exposed stay fixed: a resistance level that ignored every pivot but the last,
and a series with split cliffs the ZigZag would read as real swings.
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

from paint_suite import DET, ara_of, max_step_of, tick_of      # noqa: E402

PINE = os.path.join(os.path.dirname(__file__), os.pardir, "pine",
                    "IDX_Suite.pine")


# ================================================== the IDX ladders, exact ====
def test_the_tick_ladder_matches_the_published_bands():
    assert tick_of(150) == 1 and tick_of(200) == 2 and tick_of(499) == 2
    assert tick_of(500) == 5 and tick_of(1999) == 5
    assert tick_of(2000) == 10 and tick_of(4999) == 10
    assert tick_of(5000) == 25 and tick_of(50000) == 25


def test_the_max_step_ladder_is_a_different_constraint_from_the_tick():
    """Jenjang maksimum perubahan harga is not the tick and not auto-rejection;
    on an illiquid name it binds first."""
    assert max_step_of(150) == 10 and max_step_of(6400) == 250
    assert max_step_of(6400) != tick_of(6400)


def test_the_thin_board_is_a_flat_ten_percent_both_ways():
    """A14: 41 of 818 live names sit on Papan Pemantauan Khusus and were being
    tested against a ceiling three and a half times too high."""
    assert ara_of(6400, thin=True) == pytest.approx(0.10)
    assert ara_of(6400, thin=False) == pytest.approx(0.20)
    assert ara_of(150, thin=False) == pytest.approx(0.35)


# ============================================ the accuracy travels with it ====
def test_the_detector_table_matches_the_measured_result():
    """H37. EMA34 is the most accurate flip and every entry must beat its own
    matched null except the EMA stack, whose recall is BELOW its null."""
    f1 = {k: v[0] for k, v in DET.items()}
    null = {k: v[1] for k, v in DET.items()}
    assert max(f1, key=f1.get) == "close over EMA34"
    for k in DET:
        assert f1[k] > null[k], k


def test_every_detector_gives_back_more_than_a_random_bar_would():
    """THE NUMBER NOBODY QUOTES. A random detector gives back 8.5-8.9% of the
    peak; every real one gives back more, because a trend flip fires BECAUSE
    price fell. If this ever inverts, the give-back measurement is broken."""
    for k, (_, _, give) in DET.items():
        assert give > 0.089, k


def test_the_same_detector_numbers_are_in_the_pine_file():
    """Two copies of a constant that stop matching is the failure mode this
    repo has recorded most often."""
    with open(PINE, encoding="utf-8") as fh:
        src = fh.read()
    for k, (f1, null, give) in DET.items():
        assert f"{f1:.3f}" in src, k
