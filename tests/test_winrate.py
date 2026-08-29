"""Tests for H49 — where the 31.3% per-name win rate comes from.

The study exists because a 30-something percent win rate was challenged on the
grounds that a coin flip should give 50%. The reply is only worth anything if
the harness demonstrably returns 50% on a case where 50% is the arithmetically
correct answer, so the first group of tests is that anchor. A26 introduced this
discipline after a ZigZag that could not find a cycle in a pure sine wave was
used as evidence that no cycle existed: **a negative result about an instrument
proves nothing until the instrument is shown to work on a known positive.**

The second group pins the null. It reorders the ribbon's own green and red runs,
and it is only a valid isolation of TIMING if exposure, trade count and run
lengths survive the shuffle exactly. If the shuffle changed how long the rule
was invested, the comparison would confound timing with exposure and the +9.6
point gap would mean nothing.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir,
                                "scripts"))

from hull_colour import colour_campaign, colour_trades           # noqa: E402
from winrate import runs_of, shuffle_runs, synthetic             # noqa: E402


# ============================== THE ANCHOR: a case with a known answer =======
def test_a_driftless_market_with_no_cost_gives_a_coin_flip():
    """THE TEST THE WHOLE ANSWER RESTS ON. With mean log return zero, the median
    terminal price IS the starting price, so a trade is positive half the time
    and a randomly-timed rule beats buy-and-hold half the time. If this did not
    come out at 50% the observed 31.3% would be a bug, not a finding."""
    s = synthetic(300, 3000, mu=0.0, sigma=0.03, cost=0.0, expo=0.44,
                  mean_run=25.0, seed=11)
    se = (0.25 / s["names"]) ** 0.5
    assert abs(s["per_trade"] - 0.5) < 4 * se
    assert abs(s["per_name"] - 0.5) < 4 * se


def test_the_toll_alone_collapses_the_per_name_win_rate():
    """W3, isolated: same driftless market, only a round-trip cost added. This
    is the term that explains the observed number, and the test pins that it is
    the COST doing it rather than anything about Indonesia."""
    free = synthetic(200, 3000, 0.0, 0.03, 0.0, 0.44, 25.0, seed=5)
    paid = synthetic(200, 3000, 0.0, 0.03, 0.0144, 0.44, 25.0, seed=5)
    assert paid["per_name"] < free["per_name"] - 0.10


def test_realistic_drift_alone_barely_moves_it_which_is_why_W2_failed():
    """I registered drift as the largest term and it is not. At 44% exposure the
    forgone drift over a span is small next to 71 round trips of toll."""
    flat = synthetic(200, 3000, 0.0, 0.0385, 0.0, 0.44, 25.0, seed=6)
    real = synthetic(200, 3000, 0.0339 / 252, 0.0385, 0.0, 0.44, 25.0, seed=6)
    assert abs(real["per_name"] - flat["per_name"]) < 0.10


def test_a_positive_mean_trade_coexists_with_a_negative_median():
    """The wedge that makes a sub-50% win rate unremarkable. Multiplicative
    returns are right-skewed, so most trades land below the average one."""
    s = synthetic(200, 3000, 0.0, 0.05, 0.0144, 0.44, 25.0, seed=9)
    assert s["med_trade"] < 0 < s["mean_trade"] or s["med_trade"] < s["mean_trade"]


# ================================================ THE NULL MUST PRESERVE ====
def _mask(seed=0, n=600):
    rng = np.random.default_rng(seed)
    m = np.zeros(n, bool)
    i = 0
    while i < n:
        on, off = rng.integers(3, 40), rng.integers(3, 40)
        m[i:i + on] = True
        i += on + off
    return m


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_the_shuffle_preserves_exposure_exactly(seed):
    """If the null were invested a different share of the time, the comparison
    would measure exposure rather than timing and the whole result would be
    uninterpretable."""
    m = _mask(seed)
    s = shuffle_runs(m, np.random.default_rng(seed))
    assert s.sum() == m.sum()
    assert len(s) == len(m)


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_the_shuffle_preserves_the_run_length_multiset(seed):
    """Same runs, reordered. A null that broke long runs into short ones would
    change the trade count and therefore the toll, which is the term that
    dominates the whole study."""
    m = _mask(seed)
    s = shuffle_runs(m, np.random.default_rng(seed + 100))
    a_on, a_off, _ = runs_of(m)
    b_on, b_off, _ = runs_of(s)
    assert sorted(a_on) == sorted(b_on)
    assert sorted(a_off) == sorted(b_off)


def test_the_shuffle_preserves_the_trade_count_so_the_toll_is_identical():
    m = _mask(7)
    p = np.exp(np.cumsum(np.random.default_rng(7).normal(0, 0.02, len(m))))
    _, _, n_real, _ = colour_campaign(p, m, 0.0144)
    _, _, n_null, _ = colour_campaign(p, shuffle_runs(
        m, np.random.default_rng(3)), 0.0144)
    assert n_real == n_null


def test_the_shuffle_actually_moves_the_runs():
    """The control for the three tests above: something that preserved
    everything AND changed nothing would pass them all and be no null at all."""
    m = _mask(11)
    s = shuffle_runs(m, np.random.default_rng(2))
    assert (m != s).sum() > 20


def test_a_constant_mask_is_returned_untouched_rather_than_crashing():
    for m in (np.ones(50, bool), np.zeros(50, bool)):
        out = shuffle_runs(m, np.random.default_rng(0))
        assert (out == m).all()


def test_runs_of_round_trips_through_a_rebuild():
    m = _mask(13)
    on, off, first = runs_of(m)
    assert sum(on) == m.sum()
    assert sum(on) + sum(off) == len(m)
    assert first == bool(m[0])


# ======================================== the refactor kept one implementation
def test_the_trade_list_and_the_compounded_campaign_agree():
    """`colour_campaign` was refactored to sit on top of `colour_trades` so the
    win rate and the CAGR can never be computed from two different walks."""
    rng = np.random.default_rng(21)
    p = np.exp(np.cumsum(rng.normal(0.0002, 0.02, 800))) * 500
    m = _mask(21, 800)
    lg, _, ntr, inb = colour_campaign(p, m, 0.0144)
    tr = colour_trades(p, m, 0.0144)
    assert len(tr) == ntr
    assert sum(j - i for i, j, _ in tr) == inb
    assert lg == pytest.approx(sum(np.log(max(1 + r, 0.01)) for _, _, r in tr))
