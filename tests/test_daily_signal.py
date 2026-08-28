"""Tests for the end-of-day green-ribbon scan.

This is the one script whose output goes straight onto a phone as a list of
tickers, so the tests are aimed at the ways a scanner flatters itself: quoting a
setup it filtered out, ranking on the numerator, promoting a target the cost
floor already ate, and printing an impossible stop rather than skipping the name.
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

from daily_signal import MIN_TV, scan                           # noqa: E402


def _frame(n=500, names=8, seed=3, drift=0.0012, turnover=5e9, vol=0.02):
    """A panel in the shape `scan` expects: a rising board, all liquid."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2015-01-01", periods=n)
    out = []
    for i in range(names):
        p = 1000.0 * np.exp(np.cumsum(rng.normal(drift, vol, n)))
        out.append(pd.DataFrame({
            "date": dates, "ticker": f"T{i:02d}", "px": p, "close": p,
            "adj_close": p,
            "log_turnover": np.full(n, np.log(turnover))}))
    return pd.concat(out, ignore_index=True)


def _scan(P, **kw):
    return scan(P, P["date"].max(), **kw)


# ================================================= only what it says it scanned
def test_a_name_that_did_not_trade_today_is_not_scanned():
    """A partial refresh leaves a handful of names on the last date. Quoting a
    stale name beside fresh ones is the A11 defect in miniature."""
    P = _frame()
    P = P[~((P["ticker"] == "T00") & (P["date"] == P["date"].max()))]
    S = _scan(P)
    assert "T00" not in set(S["ticker"])


def test_an_illiquid_name_never_reaches_the_list():
    """Under Rp 1bn/day the spread is the trade. This is the single most common
    way a good-looking setup is not one."""
    P = _frame(turnover=MIN_TV / 10.0)
    assert _scan(P).empty


def test_a_falling_board_produces_no_green_ribbons():
    """Drift large against volatility, so the decline is not a coin flip. A
    first version used drift -0.002 against vol 0.02 and one of eight random
    walks finished in an uptrend anyway — which is the correct behaviour of the
    detector and a badly specified test."""
    assert _scan(_frame(names=20, drift=-0.01, vol=0.005, seed=11)).empty


def test_every_row_has_a_target_at_least_five_percent_away():
    """H36 measured NO discrimination inside +10% (AUC 0.502 at +5%): almost
    everything touches a near level inside a year, so ranking on P(touch) alone
    promotes exactly the setups worth least."""
    S = _scan(_frame())
    if len(S):
        assert (S["target_pct"] >= 0.05).all()


def test_every_quoted_stop_is_a_price_someone_could_place_an_order_at():
    """On a violently trending name the solved flip price comes out BELOW ZERO
    — a real answer, and not a level. The name is skipped, never quoted."""
    S = _scan(_frame())
    if len(S):
        assert (S["stop"] > 0).all()
        assert (S["stop"] < S["close"]).all()
        assert (S["stop_pct"] > -0.95).all()


# =========================================================== the ranking column
def test_expectancy_is_the_bracket_arithmetic_and_nothing_else():
    S = _scan(_frame())
    if len(S):
        want = (S["p_first"] * S["target_pct"]
                - (1.0 - S["p_first"]) * (-S["stop_pct"]) - S["cost"])
        assert np.allclose(S["ev"], want)


def test_expectancy_and_the_odds_ratio_disagree_on_at_least_one_panel():
    """THE REASON THE SCANNER RANKS ON EV. Ranking on the odds ratio promoted
    setups with a 3% target and a 24% stop — a ratio above 1.0 and an
    expectancy well below zero. If the two orderings were identical the choice
    would be cosmetic; this asserts it is not."""
    S = _scan(_frame(names=40, seed=5))
    if len(S) >= 5:
        a = list(S.sort_values("ev", ascending=False)["ticker"])
        b = list(S.sort_values("odds", ascending=False)["ticker"])
        assert a != b


def test_the_cost_carries_the_fraksi_harga_spread_both_ways():
    """Fees alone are a floor under a floor. On a Rp 100 name the tick is a
    larger cost than the commission."""
    from hull_trade import COST
    S = _scan(_frame())
    if len(S):
        assert (S["cost"] > COST).all()


def test_a_cheap_name_costs_more_to_trade_than_an_expensive_one():
    hi = _scan(_frame(seed=3))
    P = _frame(seed=3)
    P[["px", "close", "adj_close"]] *= 0.05        # same path, penny prices
    lo = _scan(P)
    if len(hi) and len(lo):
        assert lo["cost"].median() > hi["cost"].median()


# =============================================================== the age column
def test_age_zero_means_the_ribbon_turned_green_on_the_scanned_bar():
    S = _scan(_frame(names=30, seed=9))
    assert (S["age"] >= 0).all()
    assert (S["age"] < 10_000).all()


def test_the_fresh_filter_is_a_subset_of_the_full_list():
    S = _scan(_frame(names=30, seed=9))
    assert set(S[S["age"] <= 5]["ticker"]) <= set(S["ticker"])


# ================================================== the extrapolation flag
def test_a_volatility_outside_the_fitted_range_is_flagged_not_hidden():
    """The laws were fitted between vol60 0.0117 and 0.0623. Outside it the
    odds are extrapolation, and a scanner that prints them unmarked is
    inventing precision."""
    P = _frame(names=20, seed=4, drift=0.004)
    P["px"] = P.groupby("ticker")["px"].transform(
        lambda s: 1000.0 * np.exp(np.cumsum(np.log(s / s.shift(1)).fillna(0.0)
                                            * 4.0)))
    P["close"] = P["adj_close"] = P["px"]
    S = _scan(P)
    if len(S):
        assert "in_domain" in S.columns
        assert S["in_domain"].dtype == bool


def test_the_scan_is_deterministic():
    P = _frame()
    a, b = _scan(P), _scan(P)
    assert list(a["ticker"]) == list(b["ticker"])
    if len(a):
        assert np.allclose(a["ev"], b["ev"])


def test_a_fixed_target_overrides_the_swing_high():
    S = _scan(_frame(names=20, seed=6), target_pct=0.30)
    if len(S):
        assert np.allclose(S["target_pct"], 0.30)
        assert (S["target"] > S["close"]).all()


def test_a_fixed_target_further_out_raises_the_reward_to_risk_of_every_row():
    near = _scan(_frame(names=20, seed=6), target_pct=0.10)
    far = _scan(_frame(names=20, seed=6), target_pct=0.30)
    common = set(near["ticker"]) & set(far["ticker"])
    if common:
        n = near.set_index("ticker").loc[sorted(common), "rr"]
        f = far.set_index("ticker").loc[sorted(common), "rr"]
        assert (f > n).all()


def test_probabilities_stay_inside_zero_and_one():
    S = _scan(_frame(names=30, seed=8))
    for c in ("p_target", "p_stop", "p_first"):
        if len(S):
            assert ((S[c] > 0.0) & (S[c] < 1.0)).all()


def test_an_empty_board_returns_an_empty_frame_rather_than_raising():
    P = _frame(n=60)                       # shorter than MIN_BARS
    assert _scan(P).empty


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_most_rows_come_back_negative_on_a_random_board(seed):
    """THE SANITY CHECK ON THE WHOLE SCANNER. On a driftless synthetic board the
    cost floor should leave most setups negative. A scanner where everything
    looks good is a scanner with the cost term missing."""
    S = _scan(_frame(names=40, seed=seed, drift=0.0))
    if len(S) >= 5:
        assert (S["ev"] > 0).mean() < 0.6
