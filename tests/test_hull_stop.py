"""Tests for H40 — the Hull's own flip price as a dynamic stop.

The whole study rests on one algebraic claim: there is a single close at which
tomorrow's HMA equals today's, and it is solvable. That claim is checked against
a brute-force recomputation rather than asserted, because a closed form that is
subtly wrong produces a stop level that looks plausible on every chart.
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

from hull_stop import campaign, flip_price, swing_res            # noqa: E402
from time_price import hma                                       # noqa: E402


def _p(n=400, seed=0, vol=0.02):
    rng = np.random.default_rng(seed)
    return 100.0 * np.exp(np.cumsum(rng.normal(0.0, vol, n)))


# ======================================= the algebra, against brute force =====
@pytest.mark.parametrize("n", [21, 34, 55, 89])
def test_the_solved_price_reproduces_todays_hull_exactly(n):
    """THE CLAIM THE WHOLE STUDY RESTS ON. Append the solved close and the HMA
    must not move — if it does, the stop is at the wrong price on every bar."""
    p = _p(seed=n)
    fp = flip_price(pd.Series(p), n)
    for t in (250, 300, 350, 399):
        h_t = float(hma(pd.Series(p[:t + 1]), n).iloc[-1])
        h_next = float(hma(pd.Series(np.append(p[:t + 1], fp[t])), n).iloc[-1])
        assert h_next == pytest.approx(h_t, rel=1e-9)


@pytest.mark.parametrize("n", [21, 55])
def test_a_close_above_the_level_keeps_the_ribbon_green(n):
    """The sign matters as much as the level: if b were negative the stop would
    be an entry trigger and every trade would be backwards."""
    p = _p(seed=n + 1)
    fp = flip_price(pd.Series(p), n)
    t = 350
    h_t = float(hma(pd.Series(p[:t + 1]), n).iloc[-1])
    hi = float(hma(pd.Series(np.append(p[:t + 1], fp[t] * 1.02)), n).iloc[-1])
    lo = float(hma(pd.Series(np.append(p[:t + 1], fp[t] * 0.98)), n).iloc[-1])
    assert hi > h_t and lo < h_t


def test_the_level_is_undefined_before_the_window_fills():
    """A stop computed off a partial window would fire on bar one of every
    name, which is 891 free exits at the start of the sample."""
    fp = flip_price(pd.Series(_p(200)), 55)
    assert not np.isfinite(fp[:60]).any()
    assert np.isfinite(fp[-1])


def test_a_slower_hull_has_a_wider_stop_in_a_trend():
    """S1 in miniature, and the reason the flip price is not a tight stop: a
    slow average only turns after a large move."""
    p = 100.0 * np.exp(np.cumsum(np.full(400, 0.004)))   # a steady uptrend
    f21 = flip_price(pd.Series(p), 21)[-1]
    f55 = flip_price(pd.Series(p), 55)[-1]
    assert f55 < f21 < p[-1]


# =========================================== the position walk, look-ahead ====
def _camp(p, enter, stop, tp=None, lag=False, max_bars=252):
    p = np.asarray(p, float)
    n = len(p)
    tp = np.full(n, np.nan) if tp is None else np.asarray(tp, float)
    return campaign(p, np.asarray(enter, bool), np.asarray(stop, float), tp,
                    np.ones(n, bool), np.full(n, 2015), lag, 0.0, 0.0, max_bars)


def test_the_entry_fills_on_the_bar_after_the_signal():
    p = [10.0, 10.0, 20.0, 20.0, 5.0]
    out = _camp(p, [True, False, False, False, False], np.full(5, 9.0))
    assert out and out[0]["i"] == 1


def test_the_stop_level_is_the_one_computed_the_bar_BEFORE_the_breach():
    """stop[t] is the level for bar t+1. Comparing it to p[t] instead would use
    a level derived from the very bar it is judging."""
    p = [10.0, 10.0, 10.0, 8.0, 8.0]
    stop = np.array([np.nan, 9.0, 9.0, 9.0, 9.0])
    out = _camp(p, [True, False, False, False, False], stop)
    assert len(out) == 1
    assert out[0]["j"] == 3 and out[0]["why"] == "stop"


def test_the_extra_lag_bar_exits_strictly_later_and_at_a_worse_price():
    """The comparison H39 makes: waiting one more bar to confirm costs whatever
    that bar did, and in a fall it is negative."""
    p = [10.0, 10.0, 10.0, 8.0, 6.0, 6.0]
    stop = np.full(6, 9.0)
    stop[0] = np.nan
    fast = _camp(p, [True] + [False] * 5, stop, lag=False)
    slow = _camp(p, [True] + [False] * 5, stop, lag=True)
    assert slow[0]["j"] > fast[0]["j"]
    assert slow[0]["ret"] < fast[0]["ret"]


def test_a_target_exit_is_labelled_as_one():
    p = [10.0, 10.0, 10.0, 15.0, 15.0]
    out = _camp(p, [True, False, False, False, False], np.full(5, 5.0),
                tp=np.full(5, 12.0))
    assert out and out[0]["why"] == "tp"


def test_the_stop_wins_a_tie_because_risk_is_settled_first():
    """A bar that breaches both levels is ambiguous on daily data. Calling it a
    target would flatter every result; calling it a stop is the conservative
    reading and the one that cannot overstate the rule."""
    p = [10.0, 10.0, 10.0, 20.0]
    out = _camp(p, [True, False, False, False], np.full(4, 30.0),
                tp=np.full(4, 12.0))
    assert out and out[0]["why"] == "stop"


def test_a_position_unresolved_at_the_end_is_dropped():
    """Still open when the name's history ends means no realised return.
    Marking it to the last price would hand a rising sample a free winner on
    every name — the cap below closes trades, the end of the data does not."""
    p = [10.0] * 4 + [99.0]
    out = _camp(p, [True] + [False] * 4, np.full(5, 1.0))
    assert out == []


def test_the_time_cap_closes_a_trade_the_stop_would_never_reach():
    """THE BUG THIS FIXES. A fixed stop below the entry only ever closes
    LOSERS, so without a cap every completed trade is a loss and the win rate
    reads a definitionally impossible 0.0% — which is what the first run
    printed. The cap is what makes a fixed stop comparable to a trailing one."""
    p = [10.0] * 3 + list(np.linspace(10.0, 40.0, 40))
    out = _camp(p, [True] + [False] * 42, np.full(43, 1.0), max_bars=20)
    assert len(out) == 1
    assert out[0]["why"] == "time" and out[0]["bars"] == 20
    assert out[0]["ret"] > 0, "a winner must be allowed to close as a winner"


def test_positions_never_overlap():
    """Re-entering before the previous exit would double-count the same bars
    and inflate the trade count."""
    p = list(np.linspace(10.0, 30.0, 60)) + list(np.linspace(30.0, 5.0, 60))
    stop = np.full(120, np.nan)
    stop[1:] = np.asarray(p[:-1]) * 0.97
    out = _camp(p, [True] * 120, stop)
    for a, b in zip(out, out[1:]):
        assert b["i"] >= a["j"]


# ================================================================= the target =
def test_the_swing_target_is_the_nearest_confirmed_high_above_price():
    p = np.concatenate([np.linspace(100, 150, 30), np.linspace(150, 110, 20),
                        np.linspace(110, 135, 25)])
    res = swing_res(p)
    fin = np.isfinite(res)
    assert fin.any()
    assert (res[fin] > p[fin]).all(), "a target below price is not a target"
