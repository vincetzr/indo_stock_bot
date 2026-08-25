"""Tests for the decade basket and the tenure refinement (H23/H24).

The selection is three composed filters and a backward-looking tenure count.
Each step is pinned, because the live basket has to be the SAME object the
historical test measured or its numbers do not transfer.
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

from decade import (BY_HORIZON, CORE, EVIDENCE,               # noqa: E402
                    MIN_LISTED_YEARS, listed_years, resolve_day, tenure,
                    universe)


def _P(n=200, day="2026-08-24", years=6, turnover=None):
    """A panel with a known liquidity ordering repeated on every session."""
    days = pd.date_range(pd.Timestamp(day) - pd.Timedelta(days=365 * years),
                         pd.Timestamp(day), freq="B")
    tv = turnover if turnover is not None else np.linspace(21.0, 30.0, n)
    rows = []
    for d in days:
        rows.append(pd.DataFrame({
            "date": d, "ticker": [f"T{i:03d}" for i in range(n)],
            "close": np.linspace(100.0, 5000.0, n),
            "tradeable": True, "log_turnover": tv}))
    return pd.concat(rows, ignore_index=True)


# ============================================================ the three filters
def test_the_traded_value_floor_is_applied_first():
    """Rp1bn/day is log 20.72; everything below it is out before ranking."""
    P = _P(n=200, turnover=np.linspace(15.0, 30.0, 200))
    U = universe(P, P["date"].max())
    assert (np.exp(U["log_turnover"]) >= 1e9).all()


def test_top_half_then_top_decile_compose_to_about_the_top_five_percent():
    """THE STEP THAT IS EASY TO GET WRONG. The historical test ranked within
    the above-median subset, so 'top decile of everything' is a DIFFERENT rule
    from the one with the evidence behind it."""
    P = _P(n=200)
    U = universe(P, P["date"].max())
    assert 8 <= len(U) <= 12                     # ~5% of 200


def test_the_basket_is_the_liquid_end_not_the_thin_end():
    P = _P(n=200)
    U = universe(P, P["date"].max())
    assert "T199" in set(U["ticker"])            # most liquid
    assert "T000" not in set(U["ticker"])


def test_a_cross_section_too_small_to_rank_returns_empty():
    """Ranking thirty names by liquidity returns the thirty largest trivially;
    A11 records that printing a believable number off forty large caps."""
    assert universe(_P(n=30), _P(n=30)["date"].max()).empty


def test_untradeable_names_are_excluded():
    P = _P(n=200)
    P.loc[P["ticker"] == "T199", "tradeable"] = False
    assert "T199" not in set(universe(P, P["date"].max())["ticker"])


# ==================================================================== the day
def test_resolve_day_skips_a_ragged_tail():
    """A19's defect: a partial refresh leaves a handful of names for several
    sessions, and ranking those by liquidity is meaningless."""
    P = _P(n=200)
    last = P["date"].max()
    P = P[(P["date"] < last) | (P["ticker"].isin([f"T{i:03d}"
                                                  for i in range(20)]))]
    assert resolve_day(P) < last


def test_resolve_day_returns_the_last_session_when_the_panel_is_clean():
    P = _P(n=200)
    assert resolve_day(P) == P["date"].max()


# ================================================================== tenure
def test_tenure_counts_only_backwards():
    """A5. The score must be computable on the cohort date, so a name's FUTURE
    membership cannot contribute."""
    P = _P(n=200, years=6)
    day = P["date"].max()
    t = tenure(P, day, {"T199", "T000"})
    assert t["T199"] == 3                        # liquid on every session
    assert t["T000"] == 0


def test_tenure_snaps_each_anniversary_to_a_trading_day():
    """An exact anniversary is often a weekend or an Idul Fitri closure, and a
    missing date would silently score every name zero."""
    P = _P(n=200, years=6)
    P = P[P["date"].dt.dayofweek < 4]            # drop Fridays too
    t = tenure(P, P["date"].max(), {"T199"})
    assert t["T199"] == 3


def test_tenure_is_zero_when_the_history_is_too_short():
    P = _P(n=200, years=1)
    assert tenure(P, P["date"].max(), {"T199"})["T199"] <= 1


def test_listed_years_measures_from_the_first_bar():
    P = _P(n=200, years=6)
    y = listed_years(P, P["date"].max())
    assert y["T199"] == pytest.approx(6.0, abs=0.1)


# ============================================== the evidence travels with it
def test_the_quoted_evidence_matches_the_memo():
    """The basket must never be printable without its measured numbers, and
    those numbers must be the ones reports/horizon.md actually reports."""
    assert EVIDENCE["touch2x"] == pytest.approx(0.691)
    assert EVIDENCE["index_median"] == pytest.approx(1.085)
    assert EVIDENCE["p"] > EVIDENCE["bar"], (
        "the decile result does NOT clear this project's Bonferroni bar and "
        "the constants must keep saying so")


def test_the_core_cell_is_recorded_as_not_clearing_the_bar_either():
    assert CORE["p"] > CORE["bar"]
    assert CORE["n_names"] == 8
    assert CORE["eff_n"] < 3


def test_the_core_cell_is_stable_across_halves_which_is_its_only_strength():
    assert abs(CORE["early"] - CORE["late"]) < 0.05


def test_a_name_listed_for_less_than_the_hold_cannot_be_core():
    assert MIN_LISTED_YEARS >= 10.0, (
        "the hold is ten years; a name whose entire history is shorter than "
        "that is not the object the historical cell measured")


# ================================================ the horizon cannot be dropped
def test_the_tilt_inverts_below_the_crossover():
    """THE MISREADING THIS TABLE EXISTS TO PREVENT. The headline 69% was
    quoted without "over ten years" beside it and was read as a one-year
    number. At one year the decile touches 2x LESS often than the liquid names
    it excludes — it is the wrong side of the trade, not a weaker version of
    the right one."""
    one = [r for r in BY_HORIZON if r[0] == 1.0][0]
    ten = [r for r in BY_HORIZON if r[0] == 10.0][0]
    assert one[2] < one[1], "at one year the decile must be BELOW the base"
    assert ten[2] > ten[1], "at ten years it must be above"


def test_the_touch_rate_rises_monotonically_with_the_horizon():
    for col in (1, 2, 3):
        vals = [r[col] for r in BY_HORIZON]
        assert vals == sorted(vals), f"column {col} is not monotone"


def test_the_ten_year_row_matches_the_headline_evidence():
    ten = [r for r in BY_HORIZON if r[0] == 10.0][0]
    assert ten[1] == pytest.approx(EVIDENCE["base_touch"], abs=0.01)
    assert ten[3] == pytest.approx(CORE["touch2x"], abs=0.01)


def test_a_ten_name_basket_doubles_almost_nothing_in_a_year():
    one = [r for r in BY_HORIZON if r[0] == 1.0][0]
    assert one[2] * 10 < 1.0, (
        "fewer than one name in ten doubles inside a year; any summary that "
        "implies otherwise is wrong")
