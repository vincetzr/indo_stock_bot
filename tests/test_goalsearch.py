"""Tests for H51 — the goal search.

`touch_times` is the load-bearing piece and it is a genuinely non-obvious
optimisation: it computes the first-touch offset for EVERY target and EVERY stop
in ONE forward pass, by exploiting the fact that the running forward maximum of
the high is monotone in the window length. If that reasoning is wrong anywhere,
the whole search returns an any-touch or a last-touch label and prints entirely
believable numbers. So it is checked against the brute-force reference that
H50's own labeller is checked against.

The second group pins the economics the study exists to demonstrate: that a win
rate is purchasable with barrier placement alone and that buying it is worth
nothing. If those stopped holding on synthetic data the memo's conclusion would
be unsupported.
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

from goalsearch import TPS, SLS, summarise, touch_times          # noqa: E402


def _first_touch_ref(high, low, close, a, b, horizon):
    """Obviously-correct: walk forward one bar at a time, per entry."""
    n = len(close)
    tu = np.full(n, horizon + 1, int)
    td = np.full(n, horizon + 1, int)
    for i in range(n):
        for d in range(1, horizon + 1):
            k = i + d
            if k > n - 1:
                break
            if tu[i] > horizon and high[k] >= close[i] * (1 + a):
                tu[i] = d
            if b is not None and td[i] > horizon and low[k] <= close[i] * (1 - b):
                td[i] = d
            if tu[i] <= horizon and (b is None or td[i] <= horizon):
                break
    return tu, td


def _series(seed=0, n=200):
    rng = np.random.default_rng(seed)
    c = np.exp(np.cumsum(rng.normal(0, 0.025, n))) * 1000
    h = c * (1 + np.abs(rng.normal(0, 0.012, n)))
    lo = c * (1 - np.abs(rng.normal(0, 0.012, n)))
    return h, lo, c


# ===================================== THE ONE-PASS SCAN MUST BE FIRST-TOUCH ==
@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_one_pass_scan_matches_brute_force_for_every_level(seed):
    """THE TEST THE SEARCH RESTS ON. One scan serves the whole grid only because
    the running forward maximum is monotone in the window length; if that is
    wrong the label becomes any-touch, which is a different and easier question.
    """
    h, lo, c = _series(seed)
    tps, sls = (0.05, 0.15, 0.30), (0.20, 0.50)
    t_up, t_dn = touch_times(h, lo, c, tps, sls, 60)
    for k, a in enumerate(tps):
        ru, _ = _first_touch_ref(h, lo, c, a, None, 60)
        assert (t_up[k] == ru).all(), a
    for k, b in enumerate(sls):
        _, rd = _first_touch_ref(h, lo, c, 9.9, b, 60)
        assert (t_dn[k] == rd).all(), b


def test_a_none_stop_never_triggers():
    h, lo, c = _series(5)
    _, t_dn = touch_times(h, lo, c, (0.10,), (None,), 40)
    assert (t_dn[0] == 41).all()


def test_touch_offsets_are_monotone_in_the_target_level():
    """A nearer target can never be reached later than a further one."""
    h, lo, c = _series(7)
    t_up, _ = touch_times(h, lo, c, (0.05, 0.10, 0.20, 0.40), (None,), 80)
    for k in range(3):
        assert (t_up[k] <= t_up[k + 1]).all()


def test_a_level_never_reached_returns_the_sentinel_not_zero():
    """Returning 0 would read as 'touched immediately' and invert the label."""
    c = np.full(50, 100.0)
    t_up, _ = touch_times(c, c, c, (0.50,), (None,), 20)
    assert (t_up[0] == 21).all()


def test_the_scan_never_reads_a_bar_before_the_entry():
    """Only forward bars may set a touch. Changing bar 0 must not change the
    first-touch time of any later entry."""
    h, lo, c = _series(9)
    h2, lo2, c2 = h.copy(), lo.copy(), c.copy()
    h2[0] *= 5.0
    a1, _ = touch_times(h, lo, c, (0.10,), (None,), 50)
    a2, _ = touch_times(h2, lo2, c2, (0.10,), (None,), 50)
    assert (a1[0][1:] == a2[0][1:]).all()


# ================== THE ECONOMICS THE STUDY EXISTS TO DEMONSTRATE ============
def test_a_near_target_and_a_far_stop_buys_a_high_win_rate_on_pure_noise():
    """The mechanism behind the whole result: 83% is a property of the two lines
    you draw, not of the stock. On a driftless walk with no cost it still
    appears."""
    rng = np.random.default_rng(2)
    n, sig = 6000, 0.02
    c = np.exp(np.cumsum(rng.normal(-0.5 * sig * sig, sig, n))) * 1000
    t_up, t_dn = touch_times(c, c, c, (0.10,), (0.80,), 2000)
    hit = (t_up[0] <= 2000) | (t_dn[0] <= 2000)
    won = (t_up[0] <= 2000) & (t_up[0] <= t_dn[0])
    assert float(won[hit].mean()) > 0.80


def test_buying_that_win_rate_does_not_buy_expectation():
    """Same walk: the win rate is high AND the expectation is ~zero, which is
    what makes the goal satisfiable and worthless at the same time."""
    rng = np.random.default_rng(2)
    n, sig, a, b = 6000, 0.02, 0.10, 0.80
    c = np.exp(np.cumsum(rng.normal(-0.5 * sig * sig, sig, n))) * 1000
    t_up, t_dn = touch_times(c, c, c, (a,), (b,), 2000)
    ok = np.arange(n) + 2000 <= n - 1
    out = np.where((t_up[0] <= 2000) & (t_up[0] <= t_dn[0]), 1,
                   np.where(t_dn[0] <= 2000, -1, 0))
    ret = np.where(out == 1, a, np.where(out == -1, -b, 0.0))[ok]
    assert abs(float(ret.mean())) < 0.03


def test_the_goal_flag_requires_both_halves():
    base = {"date": pd.to_datetime(["2015-01-01"] * 200), "bars": 250,
            "outcome": 1}
    only_win = summarise(pd.DataFrame(
        {**base, "ret": [0.01] * 170 + [-0.2] * 30}), "x")
    only_mean = summarise(pd.DataFrame(
        {**base, "ret": [0.40] * 80 + [-0.1] * 120}), "x")
    both = summarise(pd.DataFrame(
        {**base, "ret": [0.15] * 170 + [-0.5] * 30}), "x")
    assert only_win["pos"] >= 0.80 and not only_win["GOAL"]
    assert only_mean["mean"] >= 0.04 and not only_mean["GOAL"]
    assert both["GOAL"]


def test_the_summary_carries_the_columns_that_qualify_the_goal():
    """A cell that clears the goal must never be reportable without its holding
    period, its compounding and its left tail beside it."""
    s = summarise(pd.DataFrame({"date": pd.to_datetime(["2015-01-01"] * 50),
                                "ret": [0.1] * 50, "bars": 500,
                                "outcome": 1}), "x")
    for k in ("yrs", "ann", "p10", "worst", "timeout"):
        assert k in s
    assert s["yrs"] == pytest.approx(500 / 252.0)


def test_a_positive_mean_can_coexist_with_negative_compounding():
    """The wedge that decides the whole memo, built to MIRROR THE MEASURED
    DISTRIBUTION rather than a convenient one: 83.5% win at the +29% median,
    and among the 16.5% of losers a heavy tail with 7.8% losing 90%.

    A first version used a flat -70% for every loser and the account still GREW
    (+0.39%/yr), which is the honest reason the real cell compounds negatively:
    it is not the average loss that does it, it is the near-total ones."""
    E = pd.DataFrame({"date": pd.to_datetime(["2015-01-01"] * 1000),
                      "ret": [0.29] * 835 + [-0.60] * 87 + [-0.90] * 78,
                      "bars": 740, "outcome": 1})
    s = summarise(E, "x")
    assert s["mean"] > 0.04 and s["pos"] > 0.80 and s["GOAL"]
    assert s["ann"] < 0


def test_the_registered_grid_still_spans_the_low_edge_region():
    """H50 failed by never testing a stop past 2 sigma. The grid must keep a
    no-stop arm and near targets, or the same omission returns."""
    assert None in SLS and max(x for x in SLS if x) >= 0.80
    assert min(TPS) <= 0.10
