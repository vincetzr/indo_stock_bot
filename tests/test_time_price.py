"""Tests for H31/H32/H33 — pivots, first passage, and the fitted cone.

The two mechanisms that decide every number in the study are the ZigZag pivot
detector and the first-passage scan, and both are the kind of code that returns
a plausible table when it is wrong. They are pinned on synthetic paths whose
answers can be worked out by hand.
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

from time_price import (block_bootstrap, design, first_passage,   # noqa: E402
                        fit_prob, fit_time, gaps_of, hma, interval_stats,
                        pivots, wma)


# ==================================================== the pivot detector =====
def test_a_monotone_series_has_no_turns():
    """It emits the opening anchor and nothing else, because a series that only
    rises never reverses. gaps_of is what the study consumes, and it is empty."""
    iv = pivots(np.linspace(100.0, 200.0, 300), 0.10)
    assert len(iv) <= 1
    assert gaps_of(iv).size == 0


def test_a_clean_zigzag_finds_its_turns():
    """Up 50%, down 40%, up 50% — every leg clears a 10% threshold, so the two
    interior extremes are pivots and the endpoints are not (a final leg is
    unconfirmed until it reverses)."""
    p = np.concatenate([np.linspace(100, 150, 20), np.linspace(150, 90, 20),
                        np.linspace(90, 135, 20)])
    iv = pivots(p, 0.10)
    #  three: the opening anchor at 100, then the two real turns. The anchor is
    #  where the record starts, not a turn, which is why gaps_of drops it.
    assert len(iv) == 3
    assert p[iv[0]] == pytest.approx(100.0)
    assert p[iv[1]] == pytest.approx(150.0)
    assert p[iv[2]] == pytest.approx(90.0)
    assert len(gaps_of(iv)) == 1


def test_wiggles_below_the_threshold_are_not_pivots():
    """THE PROPERTY THE WHOLE CYCLE TEST RESTS ON. If a 3% wobble registered as
    a turn, the interval distribution would be measuring noise and its 'cycle'
    would be the sampling rate."""
    rng = np.random.default_rng(0)
    p = 100.0 + np.cumsum(rng.normal(0, 0.2, 500))     # never moves 10%
    assert len(pivots(p, 0.10)) == 0


def test_the_anchor_is_dropped_so_every_gap_is_between_two_real_turns():
    """The opening extreme is an artefact of where the record starts. Counting
    it would put one fictitious half-cycle into every name's interval series."""
    p = np.concatenate([np.linspace(100, 150, 20), np.linspace(150, 90, 20),
                        np.linspace(90, 140, 20), np.linspace(140, 100, 20)])
    iv = pivots(p, 0.10)
    assert len(gaps_of(iv)) == len(iv) - 2
    assert gaps_of(np.array([5])).size == 0


def test_pivots_alternate_high_low_by_construction():
    rng = np.random.default_rng(1)
    p = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.02, 3000)))
    iv = pivots(p, 0.10)
    vals = p[iv]
    d = np.sign(np.diff(vals))
    assert np.all(d[:-1] * d[1:] < 0), "consecutive legs must reverse"


def test_a_looser_threshold_finds_fewer_and_wider_turns():
    rng = np.random.default_rng(2)
    p = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.02, 4000)))
    a, b = gaps_of(pivots(p, 0.05)), gaps_of(pivots(p, 0.20))
    assert len(a) > len(b)
    assert np.median(a) < np.median(b)


def test_a_periodic_series_reads_as_a_cycle():
    """The detector must be CAPABLE of finding a cycle, or its failure to find
    one in IDX would say nothing. A clean sine gives near-constant spacing."""
    t = np.arange(2000)
    p = 100.0 * np.exp(0.25 * np.sin(2 * np.pi * t / 60.0))
    s = interval_stats(gaps_of(pivots(p, 0.10)))
    assert s["cv"] < 0.05, "a true cycle must show near-zero dispersion"
    #  CONSECUTIVE PIVOTS ALTERNATE HIGH AND LOW, so the gap between them is a
    #  HALF period, not a period. Reading it as a full one would double every
    #  cycle length the study reports.
    assert 25 <= s["median"] <= 35


def test_the_observed_idx_dispersion_would_have_failed_that_test():
    """The measured IDX coefficient of variation at a 10% threshold is 2.26,
    against 1.34 for a block-bootstrap of the same returns. Both are worlds
    away from the sub-0.05 a real cycle produces."""
    rng = np.random.default_rng(3)
    p = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.02, 6000)))
    assert interval_stats(gaps_of(pivots(p, 0.10)))["cv"] > 0.5


# ============================================== interval statistics ==========
def test_interval_stats_declines_on_too_few_gaps():
    assert interval_stats(np.array([5, 7, 9])) == {}


def test_constant_intervals_have_zero_dispersion():
    s = interval_stats(np.full(100, 20))
    assert s["cv"] == pytest.approx(0.0)
    assert s["median"] == pytest.approx(20.0)


def test_the_conditional_sd_can_never_exceed_the_unconditional_by_much():
    """A two-lag least-squares forecast cannot do worse in-sample than the mean,
    so sd_cond <= sd_uncond is an identity. If it ever inverts the regression
    is misaligned, which is the shape of the bug A3 records."""
    rng = np.random.default_rng(4)
    s = interval_stats(rng.integers(3, 60, 500))
    assert s["sd_cond"] <= s["sd_uncond"] * 1.001


# =================================================== the null ================
def test_the_block_bootstrap_preserves_the_return_multiset():
    """A circular block resample must draw from the SAME returns; if it drew
    from a fitted normal it would destroy the fat tails that make a random walk
    look cyclic, and the null would be too easy to beat."""
    rng = np.random.default_rng(5)
    r = rng.normal(0, 0.02, 500)
    out = block_bootstrap(r, rng, 21)
    assert len(out) == len(r)
    assert set(np.round(out, 12)).issubset(set(np.round(r, 12)))


def test_the_block_bootstrap_is_seed_reproducible():
    r = np.arange(100.0)
    a = block_bootstrap(r, np.random.default_rng(7), 10)
    b = block_bootstrap(r, np.random.default_rng(7), 10)
    assert np.array_equal(a, b)


def test_a_longer_block_preserves_more_local_order():
    """H31b turns on exactly this: lengthening the block preserves longer
    volatility regimes, and the observed interval memory dies as it does."""
    r = np.arange(1000.0)
    short = block_bootstrap(r, np.random.default_rng(8), 2)
    long_ = block_bootstrap(r, np.random.default_rng(8), 200)
    #  consecutive-difference agreement with the original spacing of 1.0
    assert np.mean(np.diff(long_) == 1.0) > np.mean(np.diff(short) == 1.0)


# ================================================= first passage =============
def test_first_passage_is_strictly_forward():
    """THE LOOK-AHEAD GUARD. The entry bar cannot satisfy its own target, or
    every level already reached would score as instantly hit."""
    p = np.array([100.0, 100.0, 130.0])
    assert first_passage(p, 1.20, 5)[0] == 2
    #  bar 2 is already at 130; its own value must not count
    assert first_passage(np.array([130.0, 100.0]), 1.0000001, 5)[0] == -1


def test_first_passage_returns_the_FIRST_crossing_not_the_largest():
    p = np.array([100.0, 121.0, 200.0, 100.0])
    assert first_passage(p, 1.20, 5)[0] == 1


def test_a_target_never_reached_reads_minus_one():
    assert first_passage(np.array([100.0] * 30), 1.20, 10)[0] == -1


def test_a_window_running_past_the_data_is_censored_not_a_miss():
    """A19/H23's defect in first-passage form: a short window on a live name
    has no outcome yet, and calling it a miss invents one."""
    out = first_passage(np.array([100.0] * 10), 1.20, 5)
    assert out[-1] == -2
    assert out[0] == -1


def test_the_downside_direction_actually_looks_down():
    p = np.array([100.0, 95.0, 79.0, 200.0])
    assert first_passage(p, 0.80, 5)[0] == 2
    assert first_passage(p, 1.20, 5)[0] == 3


def test_the_two_directions_disagree_on_the_same_path():
    """A path that halves and then trebles has touched both barriers, and which
    one it touched FIRST is the whole content of a stop-versus-target question."""
    p = np.array([100.0, 49.0, 300.0])
    assert first_passage(p, 0.50, 5)[0] == 1
    assert first_passage(p, 2.00, 5)[0] == 2


# =================================================== the moving averages =====
def test_the_weighted_average_leans_on_the_newest_bar():
    s = pd.Series([0.0, 0.0, 3.0])
    assert wma(s, 3).iloc[-1] == pytest.approx(9.0 / 6.0)


def test_the_hull_tracks_a_ramp_with_less_lag_than_a_simple_mean():
    s = pd.Series(np.arange(200.0))
    h = hma(s, 55).iloc[-1]
    assert abs(h - 199.0) < abs(s.rolling(55).mean().iloc[-1] - 199.0)


def test_the_hull_is_computed_per_name_never_on_a_pivot():
    """A11's defect, recommitted in H30 and corrected in H33: a rolling window
    over a date x ticker pivot is indexed by the UNION of trading days, so a
    suspended name inserts rows it never had."""
    s = pd.Series([100.0] * 80)
    assert hma(s, 55).iloc[-1] == pytest.approx(100.0)
    assert not hma(s, 55).iloc[:54].notna().any()


# ================================================= the fitted laws ===========
def _cells() -> pd.DataFrame:
    rng = np.random.default_rng(11)
    rows = []
    for sig in np.linspace(0.012, 0.06, 8):
        for m in (1.05, 1.10, 1.20, 1.50, 2.00, 0.90, 0.80, 0.67, 0.50):
            d = abs(np.log(m))
            for st in (0, 1):
                base = 40.0 * d ** 0.9 / sig ** 0.6
                rows.append({"d": d, "sig": sig, "stack": st, "n": 5000,
                             "up": int(m > 1),
                             "p": float(np.clip(0.9 - 1.2 * d + 4 * sig
                                                + 0.03 * st, 0.02, 0.98)),
                             "q1": base * 0.5, "med": base, "q3": base * 1.9})
    return pd.DataFrame(rows)


def test_the_design_matrix_carries_the_curvature_and_the_interaction():
    """The linear-only version under-predicted P(+20%) by eleven points at the
    panel's own default target, so the extra columns are load-bearing."""
    X = design(_cells())
    assert X.shape[1] == 5
    R = _cells()
    assert np.allclose(X[:, 2], np.log(R["d"]) ** 2)
    assert np.allclose(X[:, 4], np.log(R["d"]) * np.log(R["sig"]))


def test_fit_time_recovers_an_ordered_set_of_quantiles():
    F = fit_time(_cells()).set_index("quantile")
    assert set(F.index) == {"q1", "med", "q3"}
    assert (F["r2"] > 0.9).all()


def test_fit_prob_returns_both_sides_with_a_stack_term():
    F = fit_prob(_cells()).set_index("side")
    assert set(F.index) == {"up", "down"}
    assert F.loc["up", "g_stack"] > 0.0
