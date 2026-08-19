"""Tests for the paired "remove the best k" test.

The point of the script is to be fair to BOTH sides, so the tests are mostly
about symmetry and about the decomposition being exact. If the segment
multiples do not multiply back to the total, every removal number built on them
is meaningless.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "scripts"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

from concentration import (breadth, segment_multiples, survives,     # noqa: E402
                           top_share, top_share_gross)


def curve(vals, start="2020-01-01"):
    return pd.Series(np.asarray(vals, float),
                     index=pd.date_range(start, periods=len(vals), freq="D"))


# --------------------------------------------------------------------------- #
# the decomposition has to be exact
# --------------------------------------------------------------------------- #
def test_segment_multiples_multiply_back_to_total_growth():
    eq = curve([100, 110, 90, 140, 200, 180, 260])
    cuts = [eq.index[2], eq.index[4]]
    m = segment_multiples(eq, cuts)
    assert np.isclose(np.prod(m), eq.iloc[-1] / eq.iloc[0])


def test_segment_count_is_one_more_than_interior_cuts():
    eq = curve(np.linspace(100, 200, 10))
    cuts = [eq.index[3], eq.index[6]]
    assert len(segment_multiples(eq, cuts)) == 3


def test_cuts_outside_the_span_are_ignored():
    eq = curve(np.linspace(100, 200, 10))
    outside = [pd.Timestamp("1999-01-01"), pd.Timestamp("2030-01-01")]
    m = segment_multiples(eq, outside)
    assert len(m) == 1
    assert np.isclose(m[0], 2.0)


def test_duplicate_cuts_do_not_create_zero_length_segments():
    eq = curve([100, 120, 150, 200])
    c = eq.index[2]
    assert len(segment_multiples(eq, [c, c, c])) == 2


def test_two_curves_cut_on_the_same_dates_get_the_same_segment_count():
    a = curve([100, 130, 90, 200, 260])
    b = curve([100, 101, 102, 103, 104])
    cuts = [a.index[1], a.index[3]]
    assert len(segment_multiples(a, cuts)) == len(segment_multiples(b, cuts))


# --------------------------------------------------------------------------- #
# removal
# --------------------------------------------------------------------------- #
def test_removing_none_is_the_full_product():
    m = np.array([1.5, 0.8, 2.0])
    assert np.isclose(survives(m, 0), np.prod(m))


def test_removal_drops_the_largest_not_the_last():
    m = np.array([1.1, 5.0, 1.2])
    assert np.isclose(survives(m, 1), 1.1 * 1.2)


def test_removal_is_monotone_downward():
    m = np.array([1.4, 2.2, 0.9, 3.1, 1.05, 1.8])
    got = [survives(m, k) for k in range(5)]
    assert all(got[i] >= got[i + 1] for i in range(len(got) - 1))


def test_removing_everything_is_not_negative_or_exploding():
    m = np.array([2.0, 3.0])
    assert survives(m, 5) in (1.0,) or np.isnan(survives(m, 5))


# --------------------------------------------------------------------------- #
# concentration share
# --------------------------------------------------------------------------- #
def test_equal_segments_give_share_proportional_to_count():
    m = np.full(10, 1.2)
    assert np.isclose(top_share(m, 5), 0.5)


def test_one_segment_carrying_everything_gives_share_of_one():
    m = np.array([1.0, 1.0, 1.0, 4.0])
    assert np.isclose(top_share(m, 1), 1.0)


def test_share_exceeds_one_when_the_rest_lose_money():
    # the honest reading of >100%: everything outside the best few is a net loss
    m = np.array([0.7, 0.8, 4.0, 0.9])
    assert top_share(m, 1) > 1.0


def test_share_is_nan_when_the_total_is_flat():
    m = np.array([1.25, 0.8])          # product is exactly 1
    assert np.isnan(top_share(m, 1))


# --------------------------------------------------------------------------- #
# the bounded version, which is the one compared across names
# --------------------------------------------------------------------------- #
def test_gross_share_is_bounded_even_when_the_net_total_is_flat():
    m = np.array([1.25, 0.8])          # product is exactly 1, net share is nan
    assert np.isnan(top_share(m, 1))
    assert 0.0 <= top_share_gross(m, 1) <= 1.0


def test_gross_share_never_exceeds_one_when_losers_dominate():
    m = np.array([0.4, 0.3, 1.9, 0.5])
    assert top_share_gross(m, 1) <= 1.0


def test_gross_share_of_all_winners_is_exactly_one():
    m = np.array([1.2, 0.9, 1.5, 0.7])
    assert np.isclose(top_share_gross(m, 2), 1.0)   # only two segments rose


def test_gross_share_is_even_when_gains_are_even():
    m = np.concatenate([np.full(10, 1.3), np.full(10, 0.9)])
    assert np.isclose(top_share_gross(m, 5), 0.5)


def test_gross_share_is_nan_when_nothing_rose():
    assert np.isnan(top_share_gross(np.array([0.9, 0.5, 0.8]), 2))


def test_gross_share_is_monotone_in_k():
    m = np.array([1.1, 2.0, 1.05, 0.8, 1.7, 1.3])
    got = [top_share_gross(m, k) for k in (1, 2, 3, 4)]
    assert all(got[i] <= got[i + 1] + 1e-12 for i in range(len(got) - 1))


# --------------------------------------------------------------------------- #
# breadth
# --------------------------------------------------------------------------- #
def test_breadth_is_none_on_a_falling_name():
    px = curve(np.linspace(200, 100, 400))
    assert breadth(px) is None


def test_breadth_is_none_when_history_is_too_short():
    px = curve(np.linspace(100, 200, 40))
    assert breadth(px) is None


def test_steady_compounding_needs_most_of_its_weeks():
    px = curve(100 * (1.001 ** np.arange(700)))
    needed, top5 = breadth(px)
    assert needed > 0.9          # nothing is concentrated, so nearly all weeks
    assert top5 < 0.15           # and the best 5% carry about 5%


def test_a_single_jump_makes_breadth_tiny():
    vals = np.full(700, 100.0)
    vals[400:] = 300.0
    needed, top5 = breadth(curve(vals))
    assert needed < 0.05
    assert top5 > 0.9


def test_breadth_shares_are_finite_and_ordered():
    rng = np.random.default_rng(7)
    px = curve(100 * np.cumprod(1 + rng.normal(0.0008, 0.02, 800)))
    out = breadth(px)
    if out is not None:
        needed, top5 = out
        assert 0 < needed <= 1.0
        assert np.isfinite(top5)
