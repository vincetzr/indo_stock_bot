"""Tests for §12 at investor-class resolution.

The measure is small and the ways to get it silently wrong are not. Three
classes of test:

  KNOWN ANSWERS. Injected timing skill must be found and must beat its null;
  flow that is independent of the next window's move must not.

  THE NULLS' CONSTRUCTION. There are two and they answer different questions:
  shuffling returns across tickers within a window asks whether the class
  picked the right NAMES, and shuffling across windows within a ticker asks
  whether it was long at the right TIMES. Each must preserve exactly what it
  claims to and destroy exactly what it claims to, and the pair must separate
  a market-timing effect from a stock-selection one — which is tested directly
  on synthetic flow built to have one and not the other.

  The need for the second null was found by running the pipeline on partial
  data: the within-window null came back at +25.1 bps against an observed
  +23.85, sitting on top of the signal, because a market move common to every
  ticker in a window survives a shuffle across tickers.

  THE ARITHMETIC. margin_bps is value-weighted, not a mean of ratios; the
  mirror residual is zero when nothing is censored; and sign_persistence must
  actually notice when one year carries a pooled average, which is the failure
  mode H11 hit.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

from idxbot.spine.investor_split import (ROUND_TRIP_BPS,          # noqa: E402
                                         class_margin, margin_bps,
                                         mirror_residual,
                                         permutation_margin,
                                         shuffle_forward,
                                         sign_persistence)


def panel(n_win, n_tick, rng, skill=0.0, view="F", start="2015-01-01"):
    """Synthetic class-flow panel. ``skill`` couples net flow to the NEXT
    window's return; at 0 the two are independent."""
    wins = pd.bdate_range(start, periods=n_win, freq="10B")
    rows = []
    for w in wins:
        for t in range(n_tick):
            gross = float(rng.uniform(1e9, 1e10))
            net = float(rng.normal(0, 0.3)) * gross
            noise = float(rng.normal(0, 0.05))
            ret = skill * np.sign(net) * 0.02 + noise
            rows.append({"ticker": f"T{t}", "window_end": w, "view": view,
                         "net_value": net, "gross_value": gross,
                         "fwd_ret": ret, "timing_pnl": net * ret,
                         "year": w.year})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# the arithmetic
# --------------------------------------------------------------------------
def test_margin_is_value_weighted_not_a_mean_of_ratios():
    """A big losing window and a small winning one must not average to neutral."""
    pnl = [-1e9, +1e8]
    gross = [1e11, 1e9]
    assert margin_bps(pnl, gross) == pytest.approx(10000 * (-9e8) / 1.01e11)


def test_margin_is_scale_free():
    assert margin_bps([100.0], [10_000.0]) == pytest.approx(
        margin_bps([1e6], [1e8]))


def test_margin_is_nan_when_nothing_traded():
    assert np.isnan(margin_bps([0.0], [0.0]))


def test_the_round_trip_cost_is_a5s_number():
    """0.28% buy + 0.18% sell + 0.1% sell tax. If this drifts, every SIZE
    verdict in the memo silently changes."""
    assert ROUND_TRIP_BPS == pytest.approx(56.0)


# --------------------------------------------------------------------------
# the null's construction
# --------------------------------------------------------------------------
def test_the_null_preserves_each_windows_returns_exactly():
    rng = np.random.default_rng(1)
    P = panel(6, 8, rng)
    S = shuffle_forward(P, np.random.default_rng(2))
    for w, g in S.groupby("window_end"):
        o = P[P.window_end == w]
        assert sorted(np.round(g["fwd_ret"], 12)) == sorted(
            np.round(o["fwd_ret"], 12))


def test_the_null_never_moves_a_return_between_windows():
    """Leaking across windows would destroy the market's time structure too,
    making the null far weaker than the question deserves."""
    rng = np.random.default_rng(3)
    P = panel(5, 6, rng)
    P["fwd_ret"] = P.groupby("window_end").ngroup().astype(float)
    S = shuffle_forward(P, np.random.default_rng(4))
    assert (S["fwd_ret"] == S.groupby("window_end").ngroup()).all()


def test_the_null_leaves_flow_untouched():
    rng = np.random.default_rng(5)
    P = panel(5, 6, rng)
    S = shuffle_forward(P, np.random.default_rng(6))
    assert np.allclose(sorted(S["net_value"]), sorted(P["net_value"]))
    assert np.allclose(sorted(S["gross_value"]), sorted(P["gross_value"]))


def test_the_null_recomputes_pnl_rather_than_carrying_the_old_one():
    """If timing_pnl were carried over unchanged the null would be identical
    to the observation and every p-value would be 1.0."""
    rng = np.random.default_rng(7)
    P = panel(5, 8, rng)
    S = shuffle_forward(P, np.random.default_rng(8))
    assert np.allclose(S["timing_pnl"], S["net_value"] * S["fwd_ret"])
    assert not np.allclose(sorted(S["timing_pnl"]), sorted(P["timing_pnl"]))


# --------------------------------------------------------------------------
# known answers
# --------------------------------------------------------------------------
def test_injected_timing_skill_is_found_and_beats_its_null():
    rng = np.random.default_rng(9)
    P = panel(40, 10, rng, skill=1.0)
    obs, nulls, p = permutation_margin(P, "F", draws=80, seed=11)
    assert obs > 0, "flow that anticipates the move must read positive"
    assert p < 0.05
    assert obs > np.nanmax(nulls)


def test_injected_negative_skill_reads_negative():
    rng = np.random.default_rng(10)
    P = panel(40, 10, rng, skill=-1.0)
    obs, _, p = permutation_margin(P, "F", draws=80, seed=12)
    assert obs < 0 and p < 0.05


def test_independent_flow_does_not_beat_its_null():
    rng = np.random.default_rng(11)
    P = panel(40, 10, rng, skill=0.0)
    obs, nulls, p = permutation_margin(P, "F", draws=100, seed=13)
    assert p > 0.05, f"independent flow read as significant (p={p}, obs={obs})"


def test_the_null_is_centred_near_zero_on_independent_flow():
    """A null centred away from zero would make every p-value meaningless —
    the failure H11 documented on its Track A."""
    rng = np.random.default_rng(12)
    P = panel(40, 10, rng, skill=0.0)
    _, nulls, _ = permutation_margin(P, "F", draws=120, seed=14)
    v = nulls[np.isfinite(nulls)]
    assert abs(np.mean(v)) < 3.0, np.mean(v)


# --------------------------------------------------------------------------
# persistence of the sign
# --------------------------------------------------------------------------
def test_sign_persistence_notices_one_year_carrying_the_average():
    """THE H11 failure mode, ported. A pooled figure driven by a single year
    must be visible as such."""
    a = pd.Series({2015: 1.0, 2016: -2.0, 2017: 1.0, 2018: -1.0,
                   2019: 2.0, 2020: 300.0})
    sp = sign_persistence(a)
    assert sp["mean"] > 45
    assert sp["mean_drop_largest"] < 1.0, \
        "dropping the largest year must expose that it carried the mean"
    assert sp["largest_year"] == 2020


def test_sign_persistence_reports_a_consistent_series_as_consistent():
    a = pd.Series({y: -20.0 - i for i, y in enumerate(range(2015, 2023))})
    sp = sign_persistence(a)
    assert sp["share_same_sign"] == pytest.approx(1.0)
    assert sp["mean_drop_largest"] < 0


def test_sign_persistence_refuses_too_few_years():
    assert sign_persistence(pd.Series({2020: 1.0}))["n_years"] == 1.0
    assert "share_same_sign" not in sign_persistence(pd.Series({2020: 1.0}))


# --------------------------------------------------------------------------
# the censoring bound
# --------------------------------------------------------------------------
def test_mirror_residual_is_zero_when_the_views_are_exact_mirrors():
    """Structurally F_net = -D_net, because every rupiah bought is sold."""
    rows = []
    for t in "AB":
        for w in pd.to_datetime(["2020-01-14", "2020-01-28"]):
            rows += [{"ticker": t, "window_end": w, "view": "F",
                      "net_value": 5e9, "gross_value": 2e10},
                     {"ticker": t, "window_end": w, "view": "D",
                      "net_value": -5e9, "gross_value": 1e10}]
    assert mirror_residual(pd.DataFrame(rows)).abs().max() == pytest.approx(0.0)


def test_mirror_residual_measures_the_gap_when_they_are_not():
    rows = [{"ticker": "A", "window_end": pd.Timestamp("2020-01-14"),
             "view": "F", "net_value": 5e9, "gross_value": 5e10},
            {"ticker": "A", "window_end": pd.Timestamp("2020-01-14"),
             "view": "D", "net_value": -4e9, "gross_value": 5e10}]
    r = mirror_residual(pd.DataFrame(rows))
    assert r.iloc[0] == pytest.approx(1e9 / 1e11)


def test_class_margin_splits_by_year():
    rng = np.random.default_rng(13)
    P = panel(60, 5, rng, skill=0.0)
    a = class_margin(P, "F", by="year")
    assert len(a) == P["year"].nunique()
    assert np.isfinite(a).all()


# --------------------------------------------------------------------------
# the two nulls answer DIFFERENT questions
# --------------------------------------------------------------------------
def market_panel(n_win, n_tick, rng, market_beta=0.0, selection=0.0):
    """Flow that is either long into up MARKETS (beta) or long the right NAMES
    within a window (selection). The two nulls must separate these."""
    wins = pd.bdate_range("2015-01-01", periods=n_win, freq="10B")
    rows = []
    for w in wins:
        mkt = float(rng.normal(0, 0.04))            # common to every ticker
        for t in range(n_tick):
            idio = float(rng.normal(0, 0.04))
            ret = mkt + idio
            gross = float(rng.uniform(1e9, 1e10))
            net = (market_beta * np.sign(mkt) + selection * np.sign(idio)
                   + float(rng.normal(0, 0.2))) * gross
            rows.append({"ticker": f"T{t}", "window_end": w, "view": "F",
                         "net_value": net, "gross_value": gross,
                         "fwd_ret": ret, "timing_pnl": net * ret,
                         "year": w.year})
    return pd.DataFrame(rows)


def test_within_ticker_null_catches_market_timing_that_within_window_misses():
    """THE reason there are two nulls. Flow that is simply long into rising
    markets scores the same under a within-window shuffle, because the market
    move is common to every ticker and survives the permutation."""
    rng = np.random.default_rng(21)
    P = market_panel(60, 12, rng, market_beta=1.0)
    _, _, p_dir = permutation_margin(P, "F", kind="within_ticker",
                                     draws=80, seed=22)
    _, _, p_sel = permutation_margin(P, "F", kind="within_window",
                                     draws=80, seed=22)
    assert p_dir < 0.05, f"market timing must be caught by the direction null (p={p_dir})"
    assert p_sel > 0.05, f"and must NOT read as stock selection (p={p_sel})"


def test_within_window_null_catches_selection():
    rng = np.random.default_rng(23)
    P = market_panel(60, 12, rng, selection=1.0)
    _, _, p_sel = permutation_margin(P, "F", kind="within_window",
                                     draws=80, seed=24)
    assert p_sel < 0.05, f"name-picking must be caught by the selection null (p={p_sel})"


def test_within_ticker_null_preserves_each_tickers_returns():
    rng = np.random.default_rng(25)
    P = market_panel(8, 6, rng)
    S = shuffle_forward(P, np.random.default_rng(26), kind="within_ticker")
    for t, g in S.groupby("ticker"):
        o = P[P.ticker == t]
        assert sorted(np.round(g["fwd_ret"], 12)) == sorted(
            np.round(o["fwd_ret"], 12))


def test_within_ticker_null_never_moves_a_return_between_tickers():
    rng = np.random.default_rng(27)
    P = market_panel(6, 5, rng)
    P["fwd_ret"] = P.groupby("ticker").ngroup().astype(float)
    S = shuffle_forward(P, np.random.default_rng(28), kind="within_ticker")
    assert (S["fwd_ret"] == S.groupby("ticker").ngroup()).all()


def test_an_unknown_null_kind_is_refused_rather_than_silently_defaulted():
    rng = np.random.default_rng(29)
    P = market_panel(4, 4, rng)
    with pytest.raises(ValueError):
        shuffle_forward(P, np.random.default_rng(30), kind="whatever")
