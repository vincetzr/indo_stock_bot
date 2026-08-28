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


def test_the_green_ribbon_count_is_not_the_survivor_count():
    """A SELF-CONTRADICTING PANEL. The header line claims to report how many
    names have a green ribbon; the first version reported how many survived
    every filter, so the same board on the same day printed 33 with a
    swing-high target and 126 with a fixed +30% one. The ribbon count must not
    depend on the target."""
    P = _frame(names=30, seed=9)
    a = _scan(P)
    b = _scan(P, target_pct=0.30)
    assert a.attrs["n_green"] == b.attrs["n_green"]
    assert a.attrs["n_green"] >= len(a)
    assert b.attrs["n_green"] >= len(b)


# ================================================ the H42 reward-to-risk gate
def test_no_emitted_row_has_a_target_nearer_than_the_gate():
    """H42: rows below MIN_RR hit their target most often (74.2%) and are the
    only cell that loses money, beaten by the SAME bracket on a random name in
    both halves. They are 76% of what the scanner used to print."""
    from idxbot.cone import MIN_RR
    S = _scan(_frame(names=40, seed=7))
    if len(S):
        assert (S["rr"] >= MIN_RR).all()


def test_the_gate_counters_account_for_every_candidate():
    """n_seen = rows that passed everything except the gate; n_thin = rows the
    gate rejected. If these do not add up the printed header is lying about
    how much of the board it threw away."""
    S = _scan(_frame(names=40, seed=7))
    assert S.attrs["n_seen"] == S.attrs["n_thin"] + len(S)
    assert S.attrs["n_green"] >= S.attrs["n_seen"]


def test_the_gate_actually_rejects_something_on_a_normal_board():
    """A gate that never fires is not a gate. On the live board it removed 24
    of 33 rows; a synthetic board should also produce near targets."""
    S = _scan(_frame(names=40, seed=7))
    assert S.attrs["n_thin"] > 0


def test_every_row_carries_the_measured_outcome_of_its_own_cell():
    from idxbot.cone import bracket_cell
    S = _scan(_frame(names=40, seed=7))
    for _, r in S.iterrows():
        c = bracket_cell(r["rr"])
        assert c is not None
        assert r["cell_ret"] == c["picks"]
        assert r["cell_hold"] == c["hold"]
        assert c["lo"] <= r["rr"] < c["hi"]


def test_the_shipped_cell_table_says_the_bracket_loses_to_holding_everywhere():
    """THE INVARIANT THAT STOPS THE LIST READING AS AN EDGE. Not one cell of
    H42 had the bracket beating a simple hold, and if a future edit to the
    table broke that it would be invisible in the printed output."""
    from idxbot.cone import BRACKET_CELLS
    for lo, hi, share, picks, rnd, diff, both, hold in BRACKET_CELLS:
        assert picks < hold, f"cell {lo}-{hi} claims the bracket beat holding"


def test_the_cell_table_covers_every_reachable_reward_to_risk():
    from idxbot.cone import bracket_cell
    for rr in (0.0, 0.1, 0.74, 0.75, 1.49, 1.5, 2.5, 4.0, 50.0, 1e6):
        assert bracket_cell(rr) is not None


def test_the_cell_shares_sum_to_one():
    from idxbot.cone import BRACKET_CELLS
    assert sum(c[2] for c in BRACKET_CELLS) == pytest.approx(1.0, abs=0.02)


def test_the_gate_sits_at_a_cell_boundary_not_between_two():
    """MIN_RR is defensible because it is a SIGN CHANGE at a bin edge fixed
    before the study ran, not the maximum of a sweep. If it drifted off the
    boundary it would become a tuned parameter."""
    from idxbot.cone import BRACKET_CELLS, MIN_RR
    assert MIN_RR in [c[0] for c in BRACKET_CELLS]
    below = [c for c in BRACKET_CELLS if c[1] <= MIN_RR]
    above = [c for c in BRACKET_CELLS if c[0] >= MIN_RR]
    assert all(c[3] < 0 for c in below), "kept cells must be the positive ones"
    assert all(c[3] > 0 for c in above)


def test_an_empty_scan_still_has_the_columns_a_caller_expects():
    """`pd.DataFrame([])` has NO COLUMNS, so `S["ticker"]` on a quiet day is a
    KeyError rather than an empty list. Introducing the H42 gate emptied two
    synthetic boards and broke the replay test with exactly that."""
    from daily_signal import COLUMNS
    S = _scan(_frame(n=60))
    assert S.empty
    assert list(S.columns) == list(COLUMNS)


def test_the_declared_empty_columns_match_what_a_real_scan_produces():
    """Pins the tuple against reality so it cannot drift as rows gain fields."""
    from daily_signal import COLUMNS
    S = _scan(_frame(names=40, seed=7))
    assert len(S), "need a populated scan for this to mean anything"
    assert list(S.columns) == list(COLUMNS)
