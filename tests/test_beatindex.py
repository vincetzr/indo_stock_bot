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


# ============================================ THE COST MODEL IS TWO THINGS ====
def _tiny_panel(n_names=200, n_dates=40, seed=0):
    """200 names, not 60. The cell is ~5% of the universe (two quantile cuts),
    so a 60-name panel yields a 3-name basket and `MIN_BASKET` correctly refuses
    to trade it — leaving every cost column NaN. The guard working is not a
    fixture that can test cost."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2010-01-01", periods=n_dates, freq="21D")
    rows = []
    for i in range(n_names):
        p = np.exp(np.cumsum(rng.normal(0.002, 0.05, n_dates))) * 1500
        rows.append(pd.DataFrame({
            "date": dates, "ticker": f"T{i:03d}", "adj_close": p, "close": p,
            "elig": True, "tv": np.exp(rng.normal(24, 1.0)),
            "hi52": rng.normal(-0.1, 0.1, n_dates),
            "vol60": rng.uniform(0.01, 0.05, n_dates)}))
    return pd.concat(rows, ignore_index=True)


def test_fee_and_spread_are_separate_knobs_not_one_blended_constant():
    """The user pays 0.56% in commission and tax. The fraksi-harga tick is an
    EXECUTION assumption on top — a full tick means taking liquidity on both
    sides, zero means every fill is passive. Quoting the two blended as "1.4%"
    hides an assumption inside what looks like a broker's schedule, which is
    exactly the confusion this parameterisation exists to prevent."""
    from beatindex import run
    from rebalance import Prices
    P = _tiny_panel()
    PX = Prices(P)
    dates = np.sort(P["date"].unique())
    cheap = run(P, PX, dates, 2, 0, None, "equal", fee=0.0056, spread_mult=0.0)
    dear = run(P, PX, dates, 2, 0, None, "equal", fee=0.0056, spread_mult=1.0)
    assert cheap and dear
    assert cheap["cost_yr"] < dear["cost_yr"]
    assert cheap["cagr"] > dear["cagr"]


def test_zero_cost_is_reachable_and_equals_the_gross_path():
    """With no fee and no spread the net path must coincide with the gross one;
    if it does not, cost is leaking in somewhere unaccounted."""
    from beatindex import run
    from rebalance import Prices
    P = _tiny_panel(seed=2)
    r = run(P, Prices(P), np.sort(P["date"].unique()), 2, 0, None, "equal",
            fee=0.0, spread_mult=0.0)
    assert r["cost_yr"] == pytest.approx(0.0, abs=1e-12)
    assert r["cagr"] == pytest.approx(r["gross"], rel=1e-9)


def test_the_cost_model_does_not_change_the_picks():
    """Cost is charged AFTER selection, so a cheaper assumption must not quietly
    alter which names are held — otherwise a cost sensitivity would be
    confounded with a selection change."""
    from beatindex import run
    from rebalance import Prices
    P = _tiny_panel(seed=5)
    PX = Prices(P)
    dates = np.sort(P["date"].unique())
    a = run(P, PX, dates, 2, 0, None, "equal", fee=0.0, spread_mult=0.0)
    b = run(P, PX, dates, 2, 0, None, "equal", fee=0.02, spread_mult=1.0)
    assert a["gross"] == pytest.approx(b["gross"], rel=1e-9)
    assert a["turnover"] == pytest.approx(b["turnover"], rel=1e-9)
