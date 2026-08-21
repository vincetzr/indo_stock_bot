"""Tests for the overlapping-window correction.

WHY THIS FILE EXISTS
--------------------
A k-day forward return computed on consecutive days shares k-1 days of its
window with its neighbour. The day-level series is therefore autocorrelated BY
CONSTRUCTION, even when the signal predicts nothing at all, and an iid standard
error divides by sqrt(n) as though those were n independent observations.

This is not a subtlety. It decided the only pre-registered hypothesis in this
project that ever survived:

    H6, iid           t = -2.67   p = 0.0041   <- passed Bonferroni
    H6, Newey-West    t = -1.25   p = 0.106    <- did not

The tests below are built around the one property that matters: on data with NO
signal but WITH overlap, the corrected test must fire at roughly its stated
rate, and the uncorrected one must fire far more often. If that gap ever closes,
the correction has stopped working and every horizon > 1 result in the repo
becomes unsafe.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "scripts"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

from layer2_test import (block_bootstrap_ci, newey_west_se,      # noqa: E402
                         one_sided)


def overlapping_noise(n_days=300, horizon=20, seed=0):
    """Pure noise, then windowed - so there is no signal but plenty of overlap.

    This is the null the test has to survive: daily returns are iid, and the
    only structure in the k-day series is the overlap itself.
    """
    rng = np.random.default_rng(seed)
    daily = rng.normal(0, 0.01, n_days + horizon)
    return np.array([daily[i:i + horizon].sum() for i in range(n_days)])


# --------------------------------------------------------------------------
# the property the whole correction exists for
# --------------------------------------------------------------------------
def test_the_iid_test_fires_far_too_often_on_overlapping_noise():
    """Establishes the problem is real before testing the fix."""
    fires = sum(one_sided(overlapping_noise(seed=s), "negative", 20)["p_iid"]
                < 0.05 for s in range(60))
    assert fires / 60 > 0.15, ("the iid test should be badly miscalibrated "
                               "here; if it is not, this fixture is wrong")


def test_the_corrected_test_is_close_to_its_stated_rate():
    p = [one_sided(overlapping_noise(seed=s), "negative", 20)["p"]
         for s in range(60)]
    p = np.array([v for v in p if np.isfinite(v)])
    assert (p < 0.05).mean() <= 0.20, (
        f"false-positive rate {(p < 0.05).mean():.0%}, expected about 5%")


def test_the_correction_always_widens_never_narrows():
    """A HAC error smaller than the iid one would be the fix running backwards."""
    for s in range(20):
        r = one_sided(overlapping_noise(seed=s), "negative", 20)
        if np.isfinite(r["t"]) and np.isfinite(r["t_iid"]):
            assert abs(r["t"]) <= abs(r["t_iid"]) * 1.05


# --------------------------------------------------------------------------
# the Newey-West estimator itself
# --------------------------------------------------------------------------
def test_on_independent_data_the_hac_error_matches_the_iid_one():
    rng = np.random.default_rng(3)
    x = rng.normal(0, 1, 4000)
    iid = x.std(ddof=1) / np.sqrt(len(x))
    assert newey_west_se(x, 5) == pytest.approx(iid, rel=0.15)


def test_on_positively_autocorrelated_data_the_hac_error_is_larger():
    rng = np.random.default_rng(4)
    e = rng.normal(0, 1, 4000)
    x = np.array([e[0]] + [0.0] * 3999)
    for i in range(1, 4000):
        x[i] = 0.8 * x[i - 1] + e[i]
    assert newey_west_se(x, 20) > 2 * (x.std(ddof=1) / np.sqrt(len(x)))


def test_a_negative_bartlett_sum_refuses_rather_than_returning_infinity():
    """Clipping to zero would report an infinite t-statistic."""
    x = np.array([1.0, -1.0] * 10)     # strong alternation
    se = newey_west_se(x, 9)
    assert np.isnan(se) or se > 0


def test_too_short_a_series_is_nan_not_a_crash():
    assert np.isnan(newey_west_se(np.array([1.0]), 5))


# --------------------------------------------------------------------------
# horizon plumbing - the failure was that the horizon never arrived
# --------------------------------------------------------------------------
def test_a_horizon_of_one_needs_no_correction():
    rng = np.random.default_rng(5)
    x = rng.normal(0, 1, 200)
    r = one_sided(x, "positive", 1)
    assert r["t"] == pytest.approx(r["t_iid"])
    assert r["n_eff"] == pytest.approx(len(x))


def test_the_effective_sample_shrinks_by_the_horizon():
    """266 overlapping 20-day windows are about 13 independent ones."""
    r = one_sided(overlapping_noise(n_days=266), "negative", 20)
    assert r["n"] == 266
    assert r["n_eff"] == pytest.approx(266 / 20)


def test_the_uncorrected_statistic_is_kept_so_the_gap_stays_visible():
    r = one_sided(overlapping_noise(seed=1), "negative", 20)
    for k in ("t", "p", "t_iid", "p_iid", "ci_lo", "ci_hi", "n_eff", "se"):
        assert k in r


# --------------------------------------------------------------------------
# the block bootstrap
# --------------------------------------------------------------------------
def test_the_bootstrap_interval_covers_the_truth_on_null_data():
    covered = 0
    for s in range(40):
        lo, hi = block_bootstrap_ci(overlapping_noise(seed=s), 20, n_boot=800,
                                    seed=s)
        covered += bool(lo <= 0.0 <= hi)
    assert covered >= 32           # ~95% nominal, allow for 40 draws


def test_blocks_shorter_than_the_series_are_required():
    assert all(np.isnan(v) for v in block_bootstrap_ci(np.arange(5.0), 20))


def test_the_bootstrap_is_reproducible_from_its_seed():
    x = overlapping_noise(seed=2)
    assert block_bootstrap_ci(x, 20, n_boot=500, seed=7) == \
        block_bootstrap_ci(x, 20, n_boot=500, seed=7)
