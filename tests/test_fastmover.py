"""Tests for the fast-multiplier screen.

The screen is two composed percentile filters and a tier fallback. The tier
fallback is what these tests are mostly for: handing back four names when ten
were asked for, or quoting the tight screen's odds for a loosened one, are both
ways of being quietly wrong.
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

from fastmover import (BLEND, EVIDENCE, TIERS, eligible,        # noqa: E402
                       pick_tier, screen)


def _P(n=200, day="2026-08-24", vol=None, turnover=None):
    """A cross-section where volatility and turnover run OPPOSITE ways.

    They must be anti-correlated or the tight screen — high vol AND thin — has
    an empty intersection by construction, and every tiering test then passes
    or fails for the wrong reason. Real IDX looks like this too: the thin end
    is the volatile end.
    """
    d = pd.Timestamp(day)
    return pd.DataFrame({
        "date": d, "ticker": [f"T{i:03d}" for i in range(n)],
        "close": np.linspace(100.0, 5000.0, n), "tradeable": True,
        "log_turnover": (turnover if turnover is not None
                         else np.linspace(30.0, 21.0, n)),
        "vol60": vol if vol is not None else np.linspace(0.01, 0.20, n)})


# ================================================================ the filters
def test_the_screen_takes_the_most_volatile_end():
    P = _P()
    S = screen(P, P["date"].max(), vol_pct=0.90, turn_pct=1.0)
    assert "T199" in set(S["ticker"])            # highest vol
    assert "T000" not in set(S["ticker"])


def test_the_turnover_filter_takes_the_THIN_end_not_the_liquid_end():
    """The decade basket takes the liquid end and this takes the thin one;
    getting the direction wrong would silently invert the whole instrument."""
    P = _P(n=200, vol=np.full(200, 0.15),
           turnover=np.linspace(30.0, 21.0, 200))
    S = screen(P, P["date"].max(), vol_pct=0.0, turn_pct=0.20)
    assert S["log_turnover"].max() <= P["log_turnover"].quantile(0.20) + 1e-9


def test_the_traded_value_floor_still_applies():
    P = _P(n=200, turnover=np.linspace(15.0, 30.0, 200))
    assert (np.exp(eligible(P, P["date"].max())["log_turnover"])
            >= 1e9 - 1).all()


def test_a_thin_cross_section_returns_empty_rather_than_ranking_noise():
    assert eligible(_P(n=20), _P(n=20)["date"].max()).empty


# ============================================================== the tiering
def test_it_widens_when_the_tight_screen_cannot_fill_the_basket():
    """THE DEFECT THIS EXISTS FOR. Today the tight screen yields four names;
    returning four when ten were asked for is a silent failure."""
    P = _P(n=200)
    tight = screen(P, P["date"].max(), TIERS[0][1], TIERS[0][2])
    tier, S = pick_tier(P, P["date"].max(), size=len(tight) + 5)
    assert len(S) > len(tight)
    assert tier != TIERS[0]


def test_it_keeps_the_tight_tier_when_that_tier_suffices():
    P = _P(n=400)
    tight = screen(P, P["date"].max(), TIERS[0][1], TIERS[0][2])
    tier, S = pick_tier(P, P["date"].max(), size=max(len(tight) - 1, 1))
    assert tier == TIERS[0]


def test_each_tier_carries_its_own_odds_not_the_tight_ones():
    """Quoting 21.2% for a loosened screen is the same class of error as
    quoting a ten-year doubling rate as a one-year one."""
    touch = [t[3] for t in TIERS]
    assert len(set(touch)) == len(touch), "tiers must not share odds"
    assert TIERS[0][3] > TIERS[1][3], "the tight tier should be the best"


def test_the_tiers_widen_monotonically():
    assert [t[1] for t in TIERS] == sorted([t[1] for t in TIERS],
                                           reverse=True)
    assert [t[2] for t in TIERS] == sorted([t[2] for t in TIERS])


# ============================================================== the evidence
def test_the_downside_is_reported_alongside_the_upside():
    """A volatile name is likelier to touch ANY level. Every tier must carry
    its halving rate next to its doubling rate, because they are the same
    fact and reporting one alone is a lie of omission."""
    for label, _, _, touch, _, _, half, _ in TIERS:
        assert 0.10 < half < 0.30, label
        assert abs(touch - half) < 0.06, (
            f"{label}: doubling and halving must stay comparable — this is "
            "variance, not skill")


def test_the_screen_does_not_compound():
    """The number that decides it, and the one H17/H18 were withdrawn over."""
    assert EVIDENCE["mean_log"] < 0
    assert EVIDENCE["mean"] > 0, (
        "arithmetic mean positive while mean log is negative is the whole "
        "trap: a rebalanced basket is paid the first and a holder the second")


def test_the_blend_curve_is_monotone_in_both_directions():
    cagr = [b[1] for b in BLEND]
    ruin = [b[3] for b in BLEND]
    assert cagr == sorted(cagr, reverse=True), "more sleeve, less compounding"
    assert ruin == sorted(ruin), "more sleeve, more chance of ending below 1x"


def test_full_allocation_is_recorded_as_much_worse_than_none():
    none_, full = BLEND[0], BLEND[-1]
    assert full[1] < none_[1] - 0.05
    assert full[2] < none_[2] / 3
    assert full[3] > 0.10


def test_the_tight_tier_is_the_one_with_the_significance_test():
    assert EVIDENCE["p"] < EVIDENCE["bar"], (
        "the tight screen is the only result in this project that clears the "
        "Bonferroni bar and the constants must keep saying so")
