"""Tests for H52 — the screen inside a size-restricted universe.

The load-bearing tests here are the two GUARDS. The first version of this study
returned +17.63% against the index's +11.15% and it was an artefact of nine
quarters holding one-to-three-name "portfolios" in a thin early universe. The
selection was identical to `rebalance.py` on the other 78 of 105 bars. So the
tests that matter are the ones making a degenerate basket impossible, plus the
equivalence test against the older implementation that caught it.
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

from beatindex import (MIN_BASKET, MIN_UNIV, TIERS, WEIGHTS,     # noqa: E402
                       pick, universe, weights)


def _bar(n=120, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "ticker": [f"T{i:03d}" for i in range(n)],
        "elig": True,
        "hi52": rng.normal(-0.15, 0.12, n),
        "vol60": rng.uniform(0.01, 0.06, n),
        "tv": np.exp(rng.normal(23, 1.5, n))})


# ====================================================== THE DEGENERATE GUARD ==
def test_a_thin_universe_returns_no_basket_at_all():
    """THE BUG THAT NEARLY BECAME THE HEADLINE. Nine quarters with a 20-40 name
    universe produced one-to-three-stock baskets whose 2003-04 returns moved a
    26-year CAGR by six points. Below the floor the correct action is to hold
    nothing, not to hold whatever is left."""
    assert len(pick(_bar(MIN_UNIV - 1))) == 0
    assert len(pick(_bar(MIN_UNIV - 1), rng=np.random.default_rng(0))) == 0


def test_a_basket_narrower_than_the_floor_is_refused():
    """The cuts are QUANTILES, so the cell is always ~5% of the universe and
    `MIN_BASKET` binds through universe SIZE rather than through names failing
    the filter. Just above the universe floor the cell is two or three names —
    which is exactly the shape that produced the +17.63% artefact — and it must
    be refused.

    (A first version of this test tried to shrink the cell by setting most
    `vol60` values huge. That does nothing: the median is computed over the same
    column, so half still pass.)"""
    for n in (40, 50, 60):
        d = _bar(n, seed=3)
        got = pick(d)
        assert len(got) == 0 or len(got) >= MIN_BASKET, n
    assert len(pick(_bar(50, seed=3))) == 0


def test_a_healthy_universe_does_produce_a_basket():
    """The control for the two above: guards that always refuse are not guards,
    they are an off switch."""
    got = pick(_bar(300, seed=5))
    assert len(got) >= MIN_BASKET


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_the_random_control_is_size_matched_to_the_cell(seed):
    """Breadth alone changes variance and therefore compounding, so a control of
    a different width is a different portfolio, not a null."""
    d = _bar(300, seed=seed)
    a = pick(d)
    b = pick(d, rng=np.random.default_rng(seed))
    assert len(a) == len(b)


def test_the_control_is_drawn_from_the_same_tier_as_the_cell():
    """Otherwise tier and selection move together and neither can be read."""
    d = _bar(300, seed=7)
    u = universe(d, 100)
    b = pick(u, rng=np.random.default_rng(1))
    assert set(b["ticker"]) <= set(u["ticker"])


# ============================================================== the universe ==
def test_the_tier_cut_takes_the_largest_by_turnover():
    d = _bar(200, seed=11)
    u = universe(d, 50)
    assert len(u) == 50
    assert u["tv"].min() >= d["tv"].nlargest(50).min() - 1e-9


def test_the_tier_cut_is_a_no_op_when_the_universe_is_already_smaller():
    d = _bar(30, seed=2)
    assert len(universe(d, 100)) == 30


def test_ineligible_names_never_enter_the_universe():
    d = _bar(100, seed=4)
    d.loc[d.index[:60], "elig"] = False
    assert len(universe(d, None)) == 40


# ================================================================= weighting ==
def test_equal_weights_sum_to_one_and_are_equal():
    w = weights(_bar(10), "equal")
    assert w.sum() == pytest.approx(1.0)
    assert w.std() == pytest.approx(0.0)


@pytest.mark.parametrize("mode", ["sqrt_tv", "tv"])
def test_size_weights_sum_to_one_and_favour_the_larger_name(mode):
    d = _bar(20, seed=9).sort_values("tv").reset_index(drop=True)
    w = weights(d, mode)
    assert w.sum() == pytest.approx(1.0)
    assert w[-1] > w[0]


def test_sqrt_weighting_is_less_concentrated_than_raw_turnover_weighting():
    """The whole point of the square root: keep the tilt, refuse to hand the
    book to one name."""
    d = _bar(40, seed=13)
    assert weights(d, "sqrt_tv").max() < weights(d, "tv").max()


def test_a_degenerate_turnover_column_falls_back_to_equal_weight():
    d = _bar(8)
    d["tv"] = 0.0
    w = weights(d, "tv")
    assert w.sum() == pytest.approx(1.0) and w.std() == pytest.approx(0.0)


# ================================================ the registered arm listing ==
def test_the_full_universe_arm_is_kept_in_the_sweep():
    """`None` is the arm every earlier study used and the one the others must be
    compared against. Dropping it would remove the baseline."""
    assert None in TIERS
    assert "equal" in WEIGHTS


def test_the_tiers_never_cut_below_the_universe_floor():
    """A tier tighter than MIN_UNIV would silently produce an all-cash arm and
    report it as a strategy."""
    assert min(t for t in TIERS if t is not None) >= MIN_UNIV
