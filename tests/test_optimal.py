"""Tests for the portfolio optimality sweep.

A portfolio backtest has three classic ways to lie, and each has a test here:

  1. Survivorship - selecting today's liquid names and running them from 2015.
     `eligible` must only see bars strictly before the decision.
  2. Look-ahead in the ranking - ranking on a window that includes the bar being
     traded. `rank_by` must be blind to it.
  3. Free rebalancing - turnover with no cost, which makes frequent rebalancing
     look free when it is the main expense.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "scripts"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

from optimal import REBALANCE, SELECTIONS, eligible, rank_by, run   # noqa: E402


def panel(n=700, cols=("AAAA", "BBBB", "CCCC", "DDDD"), seed=2):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2015-01-01", periods=n)
    px = pd.DataFrame(
        {c: 100 * np.cumprod(1 + rng.normal(0.0004, 0.02, n)) for c in cols},
        index=idx)
    tv = pd.DataFrame({c: np.full(n, 1e10) for c in cols}, index=idx)
    return px, tv


# --------------------------------------------------------------------------- #
# survivorship and look-ahead
# --------------------------------------------------------------------------- #
def test_a_name_that_has_not_listed_yet_is_not_eligible():
    px, tv = panel()
    px.loc[px.index[:400], "DDDD"] = np.nan     # lists late
    got = eligible(px, tv, 300, min_hist=250, min_turnover=1e9)
    assert "DDDD" not in set(got)


def test_eligibility_uses_only_bars_before_the_decision():
    px, tv = panel()
    a = eligible(px, tv, 300, 250, 1e9)
    px2 = px.copy()
    px2.iloc[300:] = np.nan                      # obliterate the future
    b = eligible(px2, tv, 300, 250, 1e9)
    assert set(a) == set(b)


def test_an_illiquid_name_is_screened_out():
    px, tv = panel()
    tv["CCCC"] = 1e6
    assert "CCCC" not in set(eligible(px, tv, 400, 250, 1e9))


def test_ranking_cannot_see_the_bar_it_will_trade():
    px, tv = panel()
    a = rank_by(px, tv, 400, list(px.columns), "momentum")
    px2 = px.copy()
    px2.iloc[400:] *= 10.0                       # make the future spectacular
    b = rank_by(px2, tv, 400, list(px.columns), "momentum")
    assert a == b


@pytest.mark.parametrize("how", SELECTIONS)
def test_every_selection_returns_a_full_ordering(how):
    px, tv = panel()
    got = rank_by(px, tv, 400, list(px.columns), how)
    assert set(got) == set(px.columns)


def test_lowvol_ranks_the_calmest_name_first():
    px, tv = panel()
    idx = px.index
    px["CCCC"] = 100.0 * np.exp(np.linspace(0, 0.1, len(idx)))   # nearly flat
    assert rank_by(px, tv, 500, list(px.columns), "lowvol")[0] == "CCCC"


def test_an_unknown_selection_is_refused():
    px, tv = panel()
    with pytest.raises(ValueError):
        rank_by(px, tv, 400, list(px.columns), "vibes")


# --------------------------------------------------------------------------- #
# the backtest itself
# --------------------------------------------------------------------------- #
def test_run_produces_a_curve_and_a_drawdown():
    px, tv = panel()
    r = run(px, tv, 2, "liquidity", "annual", False, min_turnover=1e9)
    assert r is not None
    assert r["final"] > 0
    assert r["max_dd"] <= 0


def test_more_frequent_rebalancing_costs_more_turnover():
    px, tv = panel(n=900, seed=5)
    never = run(px, tv, 2, "momentum", "never", False, min_turnover=1e9)
    quart = run(px, tv, 2, "momentum", "quarterly", False, min_turnover=1e9)
    assert quart["turnover"] >= never["turnover"]


def test_a_flat_market_with_no_rebalancing_keeps_its_capital():
    idx = pd.bdate_range("2015-01-01", periods=600)
    px = pd.DataFrame({c: np.full(600, 100.0) for c in ("AAAA", "BBBB")}, index=idx)
    tv = pd.DataFrame({c: np.full(600, 1e10) for c in ("AAAA", "BBBB")}, index=idx)
    r = run(px, tv, 2, "liquidity", "never", False, min_turnover=1e9)
    assert r["final"] == pytest.approx(1.0, rel=0.01)


def test_the_overlay_reduces_time_invested():
    px, tv = panel(n=900, seed=8)
    plain = run(px, tv, 3, "liquidity", "annual", False, min_turnover=1e9)
    over = run(px, tv, 3, "liquidity", "annual", True, min_turnover=1e9)
    assert over["exposure"] <= plain["exposure"] + 1e-9


def test_exposure_without_an_overlay_is_fully_invested():
    px, tv = panel()
    r = run(px, tv, 2, "liquidity", "annual", False, min_turnover=1e9)
    assert r["exposure"] == pytest.approx(1.0, abs=1e-9)


def test_run_is_none_when_there_is_not_enough_history():
    px, tv = panel(n=200)
    assert run(px, tv, 2, "liquidity", "annual", False, min_turnover=1e9) is None


def test_every_rebalance_key_is_understood():
    px, tv = panel()
    for key in REBALANCE:
        assert run(px, tv, 2, "liquidity", key, False, min_turnover=1e9) is not None
