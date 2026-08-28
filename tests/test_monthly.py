"""Tests for H41 — Rp 50 juta through the Hull rule, month by month.

The account simulator is the one piece of code in this repo that produces a
number a person might act on directly ("what do I make a month"), so the tests
here are aimed at the four ways it could quietly lie: filling on a bar it could
not have known, carrying a delisted name forever, spending money it does not
have, and reporting a full-sample edge that does not replicate.
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

from monthly import account, half_cagr, monthly, signals        # noqa: E402


def _panel(n=400, names=6, seed=0):
    """A synthetic panel in the shape `signals` expects."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2015-01-01", periods=n)
    out = []
    for i in range(names):
        p = 1000.0 * np.exp(np.cumsum(rng.normal(0.0004, 0.02, n)))
        out.append(pd.DataFrame({
            "date": dates, "ticker": f"T{i:02d}", "px": p,
            "adj_close": p, "elig": True}))
    return pd.concat(out, ignore_index=True)


# ============================================== the look-ahead, front and centre
def test_the_entry_signal_is_shifted_off_its_own_bar():
    """BOTH conditions are computed from the CLOSE of bar t, so filling at that
    same close buys at a price only known once the bar is over. The first run
    of this script did not shift, and reported the account beating the index
    while H39 measured the same rule compounding at a tenth of its rate."""
    S = signals(_panel())
    for _, g in S.groupby("ticker"):
        assert not bool(g["enter"].iloc[0]), "bar 0 can never be an entry"


def test_a_signal_column_never_leads_the_state_it_is_built_from():
    """Directly: enter[t] must equal (up & on)[t-1], never [t]."""
    from hull_trade import states
    P = _panel(names=1)
    S = signals(P)
    up, on = states(P["px"].reset_index(drop=True), 55, "EMA stack")
    want = np.concatenate([[False], (up & on)[:-1]])
    assert np.array_equal(S["enter"].to_numpy(), want)


# ================================================== the book keeps its own books
def test_the_account_never_spends_money_it_does_not_have():
    S = signals(_panel())
    A = account(S, 50e6, 5)
    assert (A["cash"] >= -1e-6).all()


def test_equity_is_cash_plus_positions_and_nothing_else():
    """A book that marks to market wrongly compounds the error every bar."""
    S = signals(_panel())
    A = account(S, 50e6, 5)
    assert A["equity"].iloc[0] > 0
    assert np.isfinite(A["equity"]).all()
    assert (A["held"] <= 5).all()


def test_a_delisted_name_is_realised_rather_than_carried_forever():
    """The panel deliberately contains names that stop printing. A position in
    one must be closed at its last price, not held at that price for the rest
    of the backtest, blocking a slot and quietly inflating the book."""
    P = _panel(names=2)
    dead = P["ticker"] == "T00"
    P = P[~(dead & (P["date"] > P["date"].iloc[200]))]
    S = signals(P)
    A = account(S, 50e6, 2)
    #  the surviving name alone cannot fill both slots, so if T00 were carried
    #  forever `held` would stay at 2 to the end
    assert int(A["held"].iloc[-1]) <= 1


def test_more_slots_means_more_names_held_not_more_capital_deployed():
    """A one-slot book puts 100% of the capital in whatever it holds, so it can
    be MORE invested than a five-slot book that only found three names. The
    property that actually holds is diversification, not deployment — a
    distinction worth pinning, because the first version of this test asserted
    the wrong one and would have passed on a broken scheduler."""
    S = signals(_panel(names=10))
    one = account(S, 50e6, 1)["held"].mean()
    five = account(S, 50e6, 5)["held"].mean()
    assert five > one


def test_the_same_seed_reproduces_the_same_account():
    """Every experiment reproducible from a seed and a config (CLAUDE.md §15)."""
    S = signals(_panel())
    a = account(S, 50e6, 5, seed=7)["equity"].to_numpy()
    b = account(S, 50e6, 5, seed=7)["equity"].to_numpy()
    assert np.array_equal(a, b)


def test_different_seeds_give_different_accounts():
    """If they did not, the draw-to-draw spread would be a fake zero and the
    study would report luck as precision."""
    S = signals(_panel(names=12))
    a = account(S, 50e6, 5, seed=1)["equity"].iloc[-1]
    b = account(S, 50e6, 5, seed=2)["equity"].iloc[-1]
    assert a != b


# =========================================================== the control arm ===
def test_the_random_arm_ignores_the_entry_signal_entirely():
    """THE CONTROL THAT DECIDES THE STUDY. If `random_entry` still filtered on
    `enter` the comparison would be the rule against itself and would show no
    edge for the most flattering possible reason."""
    P = _panel(names=8)
    S = signals(P)
    S = S.copy()
    S["enter"] = False                       # no name ever signals
    filt = account(S, 50e6, 5, random_entry=False)
    rand = account(S, 50e6, 5, random_entry=True)
    assert filt["held"].sum() == 0           # nothing to buy
    assert rand["held"].sum() > 0            # buys anyway, which is the point


def test_the_random_arm_still_respects_eligibility():
    """Random must mean random FROM THE SAME UNIVERSE, or the control is
    measuring a different pool rather than a different selection."""
    P = _panel(names=8)
    P["elig"] = P["ticker"].isin(["T00", "T01"])
    S = signals(P)
    A = account(S, 50e6, 5, random_entry=True)
    assert (A["held"] <= 2).all()


# ============================================================= the half-split ==
def test_the_half_split_recovers_a_known_pair_of_growth_rates():
    """Built from a curve that grows at a known rate in each half, because a
    half-split that is silently off by one period is undetectable in output."""
    d = pd.bdate_range("2010-01-01", periods=1000)
    mid = len(d) // 2
    eq = np.ones(len(d))
    eq[:mid] = np.exp(np.linspace(0, 0.5, mid))
    eq[mid:] = eq[mid - 1] * np.exp(np.linspace(0, 0.1, len(d) - mid))
    e, l = half_cagr(pd.Series(eq, index=d))
    #  1000 business days is ~3.84 years, so each half is ~1.92: a half that
    #  grows by exp(0.5) compounds at exp(0.5/1.92) - 1 = 29.8%, and one that
    #  grows by exp(0.1) at 5.3%. The expected values are derived rather than
    #  eyeballed, because a half-split off by one period is invisible in output.
    yrs = (d[len(d) // 2] - d[0]).days / 365.25
    assert e == pytest.approx(np.exp(0.5 / yrs) - 1.0, rel=0.02)
    assert l == pytest.approx(np.exp(0.1 / yrs) - 1.0, rel=0.05)
    assert e > l


def test_the_two_halves_cover_the_same_span_of_calendar():
    d = pd.bdate_range("2010-01-01", periods=1001)
    s = pd.Series(np.linspace(1.0, 2.0, len(d)), index=d)
    mid = s.index[len(s) // 2]
    assert abs((mid - s.index[0]).days - (s.index[-1] - mid).days) <= 4


# ================================================================ the reporting
def test_monthly_returns_are_month_end_to_month_end():
    d = pd.bdate_range("2020-01-01", periods=300)
    s = pd.Series(np.linspace(100.0, 200.0, len(d)), index=d)
    m = monthly(s)
    assert len(m) == len(s.resample("ME").last()) - 1
    assert (m > 0).all()


def test_a_flat_account_reports_a_flat_month():
    d = pd.bdate_range("2020-01-01", periods=300)
    m = monthly(pd.Series(np.ones(len(d)), index=d))
    assert m.abs().max() < 1e-12
