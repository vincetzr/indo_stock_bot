"""Tests for H42 — replaying the daily scan through history.

The study's whole claim to legitimacy is that a full-series computation at bar
t equals what the live scanner would have printed on day t. That is asserted
nowhere in the code and is easy to break — one non-causal helper anywhere in
the chain turns the entire backtest into a look-ahead. So the first test here
truncates the panel to a past date, runs the LIVE scanner on it, and demands
the replay produce the same row.
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

from daily_signal import scan                                   # noqa: E402
from idxbot.cone import MIN_RR                                  # noqa: E402
from signal_backtest import (HORIZON, _walk, block_boot,        # noqa: E402
                             signal_rows)


def _panel(n=900, names=6, seed=1, drift=0.0012, vol=0.02, turnover=5e9):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2010-01-01", periods=n)
    out = []
    for i in range(names):
        p = 1000.0 * np.exp(np.cumsum(rng.normal(drift, vol, n)))
        out.append(pd.DataFrame({
            "date": dates, "ticker": f"T{i:02d}", "px": p, "close": p,
            "adj_close": p, "log_turnover": np.full(n, np.log(turnover))}))
    return pd.concat(out, ignore_index=True)


# ================================== THE TEST THE WHOLE STUDY DEPENDS ON =======
@pytest.mark.parametrize("back", [0, 40, 150])
def test_the_replay_reproduces_the_live_scanner_at_a_past_date(back):
    """Truncate the panel to a past date, run the LIVE scanner on what was
    knowable then, and demand the replay's row for that date matches. If any
    input were non-causal — a ZigZag drawn at the pivot instead of the
    confirmation bar, a centred rolling window — these would diverge."""
    P = _panel()
    #  THE GATE MUST BE PASSED EXPLICITLY. `signal_rows` defaults to no gate so
    #  the study can measure the cells the live scanner throws away; comparing
    #  an ungated replay to a gated scanner is comparing two different screens,
    #  which is what this test caught when MIN_RR was introduced.
    S = signal_rows(P, min_rr=MIN_RR)
    asof = pd.Timestamp(np.sort(P["date"].unique())[-1 - back])
    live = scan(P[P["date"] <= asof], asof)
    back_rows = S[S["date"] == asof]
    assert set(live["ticker"]) == set(back_rows["ticker"]), \
        "the two disagree about WHICH names signal"
    if len(live):
        a = live.set_index("ticker").sort_index()
        b = back_rows.set_index("ticker").sort_index()
        assert np.allclose(a["target_pct"], b["d_up"], atol=1e-9)
        assert np.allclose(-a["stop_pct"], b["d_dn"], atol=1e-9)
        assert np.allclose(a["p_first"], b["p_first"], atol=1e-9)
        assert np.allclose(a["ev"], b["ev"], atol=1e-9)


def test_a_future_bar_cannot_change_a_past_signal():
    """The same date scanned on a short panel and a long one must agree — the
    direct statement of no look-ahead, independent of the scanner."""
    P = _panel(n=900)
    cut = pd.Timestamp(np.sort(P["date"].unique())[600])
    a = signal_rows(P[P["date"] <= cut], min_rr=MIN_RR)
    b = signal_rows(P, min_rr=MIN_RR)
    a = a[a["date"] == cut].set_index("ticker").sort_index()
    b = b[b["date"] == cut].set_index("ticker").sort_index()
    assert list(a.index) == list(b.index)
    if len(a):
        for c in ("d_up", "d_dn", "p_first", "ev", "age", "vol"):
            assert np.allclose(a[c], b[c], atol=1e-9)


# ================================================== the forward walk ==========
def _walk1(path, d_up, d_dn, cost=0.0):
    p = np.asarray(path, float)
    return _walk(p, np.array([0]), np.array([d_up]), np.array([d_dn]),
                 np.array([cost]))


def test_the_target_is_recorded_when_the_path_reaches_it():
    o = _walk1([100.0] + [101.0] * 5 + [120.0] + [100.0] * 300, 0.10, 0.10)
    assert o["first"][0] == "target"
    assert o["t_target"][0] == 6
    assert bool(o["hit_target"][0])


def test_the_stop_is_recorded_when_the_path_reaches_it_first():
    o = _walk1([100.0, 99.0, 85.0] + [200.0] * 300, 0.10, 0.10)
    assert o["first"][0] == "stop"
    assert o["t_stop"][0] == 2


def test_a_path_that_touches_neither_is_neither():
    o = _walk1([100.0] + [100.5] * 300, 0.10, 0.10)
    assert o["first"][0] == "none"
    assert not bool(o["hit_target"][0])
    assert not bool(o["hit_stop"][0])


def test_the_entry_bar_cannot_satisfy_its_own_target():
    """Strictly forward. A window starting at the entry bar would let a name
    that is already at its target register an instant win."""
    o = _walk1([100.0, 90.0] + [90.0] * 300, 0.0001, 0.50)
    assert o["t_target"][0] != 0


# =============================================== the fill, which is the point
def test_the_exit_fills_at_the_actual_close_not_the_nominal_level():
    """A27's control. A bar that breaches a -10% stop often closes at -20%, and
    filling at the level flatters exactly the tight stops that win a grid."""
    o = _walk1([100.0, 80.0] + [80.0] * 300, 0.10, 0.10)
    assert o["ret"][0] == pytest.approx(-0.20, abs=1e-9)   # not -0.10


def test_the_cost_is_subtracted_from_the_realised_return():
    a = _walk1([100.0, 120.0] + [120.0] * 300, 0.10, 0.10, cost=0.0)
    b = _walk1([100.0, 120.0] + [120.0] * 300, 0.10, 0.10, cost=0.01)
    assert a["ret"][0] - b["ret"][0] == pytest.approx(0.01, abs=1e-12)


def test_a_same_duration_hold_would_be_degenerate_and_is_not_the_benchmark():
    """THE TRAP THIS STUDY WAS BUILT TO AVOID. The bracket exits at the CLOSE
    of its exit bar, so a hold of that many bars returns the identical number
    and the 'edge' is exactly zero by construction — H39's '0.0% of trades'
    shape. `hold` must therefore be the FULL-horizon hold, which differs
    whenever the bracket exited early."""
    o = _walk1([100.0, 120.0] + [50.0] * 300, 0.10, 0.10)
    assert o["first"][0] == "target"
    assert o["ret"][0] == pytest.approx(0.20, abs=1e-9)
    assert o["hold"][0] < 0.0            # the name collapsed afterwards
    assert o["ret"][0] != o["hold"][0]


def test_a_window_running_off_the_end_of_a_life_is_censored_not_a_miss():
    """A delisted name would otherwise count as 'the target was never
    reached', which is a survivorship error with the sign reversed."""
    o = _walk1([100.0] + [100.5] * 30, 0.10, 0.10)
    assert bool(o["censored"][0])


def test_a_full_window_that_touches_nothing_is_a_miss_not_censored():
    o = _walk1([100.0] + [100.5] * (HORIZON + 5), 0.10, 0.10)
    assert not bool(o["censored"][0])


# ======================================================= the clustered null ===
def test_the_block_bootstrap_is_wider_than_an_iid_one_on_clustered_data():
    """A17: one name in a green ribbon contributes a near-identical signal
    every session for months. A row-level resample destroys nothing and returns
    an interval several times too narrow. Demonstrated on synthetic data with a
    known cluster structure rather than asserted."""
    rng = np.random.default_rng(0)
    blocks = np.repeat(np.arange(40), 50)
    v = np.repeat(rng.normal(0, 1, 40), 50) + rng.normal(0, 0.01, 2000)
    _, lo_b, hi_b = block_boot(v, blocks, draws=400)
    _, lo_i, hi_i = block_boot(v, np.arange(len(v)), draws=400)
    assert (hi_b - lo_b) > 3.0 * (hi_i - lo_i)


def test_the_block_bootstrap_brackets_the_sample_mean():
    rng = np.random.default_rng(2)
    v = rng.normal(0.05, 1.0, 1500)
    blocks = np.repeat(np.arange(150), 10)
    m, lo, hi = block_boot(v, blocks, draws=400)
    assert lo < m < hi


# ============================================================ the population ==
def test_every_emitted_signal_obeys_the_scanners_own_filters():
    S = signal_rows(_panel(names=10, seed=4), min_rr=MIN_RR)
    if len(S):
        assert (S["d_up"] / S["d_dn"] >= MIN_RR).all()
        assert (S["d_up"] >= 0.05).all()
        assert ((S["d_dn"] > 0.02) & (S["d_dn"] < 0.95)).all()
        assert (S["cost"] > 0).all()


def test_expectancy_matches_the_live_scanners_definition():
    S = signal_rows(_panel(names=10, seed=4))
    if len(S):
        want = (S["p_first"] * S["d_up"]
                - (1.0 - S["p_first"]) * S["d_dn"] - S["cost"])
        assert np.allclose(S["ev"], want)


def test_a_falling_board_returns_an_empty_frame_rather_than_raising():
    """`pd.concat([])` raises, so a board on which nothing qualifies would come
    back as a traceback instead of "no signals today"."""
    S = signal_rows(_panel(names=10, drift=-0.01, vol=0.005, seed=5))
    assert isinstance(S, pd.DataFrame)
    assert S.empty
