"""Tests for H45 — combinations of the simple methods.

The danger in a combination sweep is not that the arithmetic is wrong, it is
that ~70 cells get scored and the best one is reported as a measurement rather
than as the maximum of a sweep. These tests pin the machinery; the trial count
and the null are what defend the conclusion.

The tie-break tests are load-bearing. `ema_stack` is a 0-3 ordinal and
`sr_break` is exactly 0.0 for every name with no confirmed resistance overhead,
so both have enormous ties, and a stable sort resolves them by the frame's
existing order — which is alphabetical by ticker. That is not a tie-break, it
is a fixed selection of names wearing a signal's clothes (A15).
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

from combos import CLEARED, LEGS, MODES, combo, rank_frame     # noqa: E402
from cost_ladder import RULES                                   # noqa: E402


def _frame(n=40, names=80, seed=1):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2015-01-01", periods=n, freq="21D")
    out = []
    for i in range(names):
        out.append(pd.DataFrame({
            "date": dates, "ticker": f"T{i:03d}",
            "fwd": rng.normal(0.01, 0.10, n),
            **{r: rng.normal(size=n) for r in RULES}}))
    return pd.concat(out, ignore_index=True).sort_values(["date", "ticker"])


# ============================================================ the rank frame ==
def test_ranks_are_percentiles_within_each_date_not_across_the_panel():
    """Ranking across the panel would let a rule's level, rather than its
    cross-sectional position, decide the book — and a market-wide drift would
    then masquerade as stock selection."""
    R = rank_frame(_frame())
    for _, g in R.groupby("date"):
        for r in RULES:
            assert g[r].min() > 0.0 and g[r].max() <= 1.0
            assert abs(g[r].mean() - 0.5) < 0.05


def test_ranking_makes_legs_on_wildly_different_scales_comparable():
    """A 0-3 ordinal, a log ratio and a 0-100 oscillator cannot be averaged
    raw: whichever has the widest units would dominate the blend."""
    D = _frame()
    D["ema_stack"] = D["ema_stack"] * 1000.0     # blow up one leg's scale
    R = rank_frame(D)
    assert R["ema_stack"].max() <= 1.0


# ================================================== the two combination modes =
def test_avg_is_the_mean_and_and_is_the_minimum():
    R = rank_frame(_frame())
    legs = ("ema_cross", "mom12_1")
    assert np.allclose(combo(R, legs, "avg"), R[list(legs)].mean(axis=1))
    assert np.allclose(combo(R, legs, "and"), R[list(legs)].min(axis=1))


def test_confluence_is_never_above_either_leg_and_a_blend_is_between_them():
    """The whole point of AND: a name scores high only if it is high on EVERY
    leg. If confluence could exceed a leg it would not be confluence."""
    R = rank_frame(_frame())
    legs = ("ema_cross", "stoch_strong")
    a, b = R[legs[0]], R[legs[1]]
    andv, avgv = combo(R, legs, "and"), combo(R, legs, "avg")
    assert (andv <= a + 1e-12).all() and (andv <= b + 1e-12).all()
    assert (avgv >= np.minimum(a, b) - 1e-12).all()
    assert (avgv <= np.maximum(a, b) + 1e-12).all()


def test_combining_a_leg_with_itself_returns_that_leg():
    R = rank_frame(_frame())
    for m in MODES:
        assert np.allclose(combo(R, ("mom12_1", "mom12_1"), m), R["mom12_1"])


def test_a_single_leg_is_the_same_under_both_modes():
    R = rank_frame(_frame())
    assert np.allclose(combo(R, ("sr_break",), "avg"),
                       combo(R, ("sr_break",), "and"))


def test_confluence_concentrates_more_than_a_blend():
    """K4's mechanism: AND pushes most names down, so the surviving book is a
    narrower and stickier set — which is why its turnover should be lower."""
    R = rank_frame(_frame(names=200))
    legs = ("ema_cross", "stoch_strong", "mom12_1")
    assert combo(R, legs, "and").mean() < combo(R, legs, "avg").mean()


# ================================================ the sweep's own discipline ==
def test_the_predicted_null_leg_is_available_to_pair_with():
    """K3 needs `rand` in the rank frame so a real signal can be paired with
    noise. If pairing with noise does not hurt, the machinery is averaging
    something other than what it claims."""
    R = rank_frame(_frame())
    assert "rand" in R.columns


def test_the_cleared_set_is_a_subset_of_the_legs():
    assert set(CLEARED) <= set(LEGS)


def test_both_stochastic_readings_and_fibonacci_stay_in_the_sweep():
    """Dropping the losers after seeing the answer is how a sweep becomes a
    mined result. fib_618 and stoch_oversold are the honest negative cells."""
    assert "fib_618" in LEGS and "stoch_oversold" in LEGS


# ======================================================== THE TIE-BREAK TRAP ==
def test_a_heavily_tied_score_is_resolved_by_ticker_order_not_by_signal():
    """DEMONSTRATED, NOT ASSERTED. With an ordinal score, a stable sort hands
    the whole book to the alphabetically first names inside the top tier. This
    test shows the defect exists so the study cannot quietly rely on it."""
    d = pd.DataFrame({"ticker": [f"T{i:03d}" for i in range(100)],
                      "score": [3.0] * 50 + [0.0] * 50})
    top = d.sort_values("score", ascending=False, kind="stable").head(10)
    assert list(top["ticker"]) == [f"T{i:03d}" for i in range(10)]


def test_percentile_ranking_does_not_remove_the_tie():
    """Ranking converts ties to a shared mid-rank, so the ordering problem
    survives into the combined score. Combining does not launder it."""
    D = _frame(names=100)
    D["ema_stack"] = np.floor(D["ema_stack"] * 0 + 3.0)   # everything tied
    R = rank_frame(D)
    g = R[R["date"] == R["date"].iloc[0]]
    assert g["ema_stack"].nunique() == 1


def test_a_tie_free_score_is_unaffected_by_row_order():
    """The control for the two tests above: with a continuous score the book is
    a property of the signal, and shuffling the frame changes nothing."""
    D = _frame(names=60, seed=5)
    day = D["date"].iloc[0]
    g = D[D["date"] == day]
    a = set(g.sort_values("ema_cross", ascending=False).head(6)["ticker"])
    b = set(g.sample(frac=1.0, random_state=3)
            .sort_values("ema_cross", ascending=False).head(6)["ticker"])
    assert a == b
