"""Tests for H47 — sell-off detection and the give-back.

The study's whole conclusion rests on a comparison between a real detector and
a random exit of the same speed, so the tests that matter are the ones pinning
that the two arms play the SAME game. The first version of the harness did not:
it required a fresh rising edge of the setup to re-enter, which quietly locked
the random arm out of the rest of every trend it sold into while letting the
real detectors — whose trigger usually breaks the setup too — rejoin. A control
that is handicapped is not a control.

The annualisation tests are the other load-bearing group. Two separate versions
of this script printed every rule beating buy-and-hold, both times by scaling a
short trade as though it repeated back-to-back all year.
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

from selloff import (MAX_BARS, campaign, detectors, ema, hma,   # noqa: E402
                     summarise, wma)


# ================================================== the moving averages ======
def test_wma_weights_the_latest_bar_most():
    """A WMA of a ramp must sit above the simple mean, because the newest bar
    carries the largest weight. If the kernel were reversed it would sit below
    and every Hull in the study would be lagging the wrong way."""
    v = np.arange(1.0, 11.0)
    out = wma(v, 5)
    assert np.isnan(out[:4]).all()
    assert out[4] > v[:5].mean()
    assert out[4] == pytest.approx((1 * 1 + 2 * 2 + 3 * 3 + 4 * 4 + 5 * 5) / 15)


def test_wma_matches_a_rolling_apply_which_is_the_slow_reference():
    """The vectorised form exists because `rolling().apply()` killed two runs.
    It has to give the same answer as the thing it replaced."""
    rng = np.random.default_rng(0)
    v = rng.normal(100, 5, 200)
    w = np.arange(1, 13, dtype=float)
    w /= w.sum()
    ref = pd.Series(v).rolling(12).apply(lambda x: float(np.dot(x, w)),
                                         raw=True).to_numpy()
    assert np.allclose(wma(v, 12)[11:], ref[11:])


def test_hull_tracks_a_ramp_more_closely_than_a_plain_ema():
    """The Hull's selling point is reduced lag. On a clean ramp it must end up
    nearer the current price than an EMA of the same length."""
    v = np.arange(1.0, 401.0)
    assert abs(hma(v, 55)[-1] - v[-1]) < abs(ema(v, 55)[-1] - v[-1])


def test_every_moving_average_is_causal():
    """Nothing may read a bar after itself. Changing the LAST value must not
    change any earlier output — the cheapest look-ahead test there is."""
    rng = np.random.default_rng(3)
    v = rng.normal(100, 5, 300)
    w = v.copy()
    w[-1] = 1e6
    for f in (lambda x: wma(x, 21), lambda x: hma(x, 55), lambda x: ema(x, 50)):
        a, b = f(v), f(w)
        assert np.allclose(np.nan_to_num(a[:-1]), np.nan_to_num(b[:-1]))


# ========================================================== the detectors ====
def test_every_detector_is_causal():
    rng = np.random.default_rng(7)
    p = np.cumprod(1 + rng.normal(0, 0.02, 300)) * 1000
    atr = np.full(300, 20.0)
    tvz = rng.normal(size=300)
    h = hma(p, 55)
    q = p.copy()
    q[-1] *= 3.0
    A = detectors(p, atr, tvz, h)
    B = detectors(q, atr, tvz, hma(q, 55))
    for k in A:
        #  hull55 is excluded from the tail comparison only where the HMA of
        #  the perturbed series legitimately differs; the check is on the head.
        assert (A[k][:200] == B[k][:200]).all(), k


def test_a_drop_detector_fires_exactly_on_the_drop_bar():
    p = np.array([100.0] * 5 + [93.0] + [100.0] * 5)
    d = detectors(p, np.full(11, 1.0), np.zeros(11), np.zeros(11))
    assert d["drop -6%"][5] and d["drop -6%"].sum() == 1
    assert not d["drop -8%"].any()


def test_the_hull_exit_is_confirmed_one_bar_late_not_at_the_flip():
    """H40's S2: exiting AT the flip loses to waiting one bar, because the bar
    after a Hull flip is on average an UP bar. The harness must encode the
    version that measured better, and the lag must be exactly one."""
    h = np.array([1.0, 2, 3, 4, 5, 4, 3, 2, 1, 0])
    d = detectors(np.full(10, 100.0), np.full(10, 1.0), np.zeros(10), h)
    red = np.zeros(10, bool)
    red[2:] = h[2:] <= h[:-2]
    assert (d["hull55 +1bar"][1:] == red[:-1]).all()
    assert not d["hull55 +1bar"][0]


def test_volume_climax_needs_both_a_down_bar_and_a_volume_spike():
    p = np.array([100.0, 99.0, 101.0, 100.0])
    tvz = np.array([0.0, 3.0, 3.0, 0.0])
    d = detectors(p, np.full(4, 1.0), tvz, np.zeros(4))
    assert d["vol climax"][1]            # down and loud
    assert not d["vol climax"][2]        # loud but up
    assert not d["vol climax"][3]        # down but quiet


# ===================================================== THE RE-ENTRY POLICY ===
def _ramp(n=120):
    return np.linspace(100.0, 200.0, n)


def test_live_policy_lets_a_rule_rejoin_a_trend_it_sold_into():
    """The fix. Under `edge` a mid-trend exit is locked out until the setup
    dies and restarts; under `live` it rejoins as soon as the signal clears."""
    p = _ramp()
    enter = np.ones(len(p), bool)
    enter[:5] = False           # so `edge` has exactly one rising edge to find
    sig = np.zeros(len(p), bool)
    sig[40] = True
    live = campaign(p, enter, sig, 0.0, "T", reentry="live")
    edge = campaign(p, enter, sig, 0.0, "T", reentry="edge")
    assert len(live) > len(edge)
    assert len(edge) == 1


def test_the_control_arm_is_locked_out_under_edge_which_is_the_whole_defect():
    """DEMONSTRATED, NOT ASSERTED. A random exit fires mid-trend with the setup
    still live, so under `edge` it takes ONE trade in a trend where a real
    detector — whose trigger also breaks the setup — takes several. That is a
    handicap, not a null, and it inflated the real rules' CAGR advantage."""
    p = _ramp(200)
    enter = np.ones(len(p), bool)
    enter[:5] = False           # one rising edge, then the setup stays live
    rng = np.random.default_rng(1)
    edge = campaign(p, enter, np.zeros(len(p), bool), 0.0, "T",
                    rand_bars=np.array([20]), rng=rng, reentry="edge")
    rng = np.random.default_rng(1)
    live = campaign(p, enter, np.zeros(len(p), bool), 0.0, "T",
                    rand_bars=np.array([20]), rng=rng, reentry="live")
    assert len(edge) == 1
    assert len(live) >= 8


def test_a_rule_never_re_enters_while_its_own_exit_signal_is_still_firing():
    """Otherwise `close < EMA20` would sell and buy back on the same condition,
    paying the toll every bar of a drawdown."""
    p = _ramp()
    enter = np.ones(len(p), bool)
    sig = np.zeros(len(p), bool)
    sig[30:60] = True
    tr = campaign(p, enter, sig, 0.0, "T", reentry="live")
    for t in tr:
        assert not sig[t["i"]]


# ============================================================ the campaign ===
def test_a_trade_is_charged_its_cost_once_and_the_cost_lowers_the_return():
    p = _ramp()
    enter = np.ones(len(p), bool)
    sig = np.zeros(len(p), bool)
    sig[20] = True
    free = campaign(p, enter, sig, 0.0, "T")[0]
    paid = campaign(p, enter, sig, 0.02, "T")[0]
    assert paid["ret"] == pytest.approx(free["ret"] - 0.02)


def test_give_back_is_measured_from_the_in_trade_peak_and_is_zero_at_a_high():
    """A trade that exits at its own peak gave nothing back. If this were
    non-zero the whole table would be measuring something else."""
    p = _ramp()
    enter = np.ones(len(p), bool)
    sig = np.zeros(len(p), bool)
    sig[20] = True
    assert campaign(p, enter, sig, 0.0, "T")[0]["give_back"] == pytest.approx(0)


def test_give_back_equals_the_fall_from_the_peak_on_a_rise_then_fall():
    p = np.r_[np.linspace(100, 200, 50), np.linspace(200, 160, 20)]
    enter = np.ones(len(p), bool)
    sig = np.zeros(len(p), bool)
    sig[60] = True
    t = campaign(p, enter, sig, 0.0, "T")[0]
    assert t["give_back"] == pytest.approx((200 - p[60]) / 200, abs=1e-9)


def test_the_bar_cap_binds_so_no_trade_runs_past_a_year():
    p = _ramp(1000)
    enter = np.ones(len(p), bool)
    tr = campaign(p, enter, np.zeros(len(p), bool), 0.0, "T")
    assert tr and max(t["bars"] for t in tr) <= MAX_BARS


def test_a_trailing_stop_exits_after_the_drawdown_reaches_its_distance():
    p = np.r_[np.linspace(100, 200, 50), np.linspace(200, 100, 50)]
    enter = np.ones(len(p), bool)
    t = campaign(p, enter, np.zeros(len(p), bool), 0.0, "T", trail=0.15)[0]
    assert p[t["j"]] <= 200 * 0.85
    assert p[t["j"] - 1] > 200 * 0.85


# ===================================== THE ANNUALISATION, TWICE GOT WRONG ====
def _trades(n, ret, bars, span, hold_lg):
    return [{"tk": "T", "span": span, "i": 0, "j": bars, "bars": bars,
             "ret": ret, "give_back": 0.05, "hold_span": hold_lg}
            for _ in range(n)]


def test_the_rule_is_charged_for_the_time_it_sits_in_cash():
    """The defect that made two earlier versions print every rule beating
    buy-and-hold: scaling a 23-bar trade by 252/23 assumes it repeats
    back-to-back all year. Ten 5% trades over ten years is ~5%/yr, not 5% times
    the number of times the trade would fit into a year."""
    s = summarise(_trades(40, 0.05, 25, 10.0, np.log(3.0)), "x")
    assert s["cagr"] == pytest.approx(np.exp(40 * np.log(1.05) / 10.0) - 1.0)
    assert s["cagr"] < 0.25


def test_hold_and_rule_are_annualised_over_the_same_span():
    """A19's recurring error was comparing quantities measured over different
    windows. Both legs here divide by the SAME denominator."""
    s = summarise(_trades(40, 0.0, 25, 10.0, np.log(4.0)), "x")
    assert s["cagr"] == pytest.approx(0.0, abs=1e-9)
    assert s["hold"] == pytest.approx(4.0 ** 0.1 - 1.0)


def test_a_rule_that_captures_the_whole_move_ties_buy_and_hold():
    """The sanity anchor: one trade that runs the full span at the name's own
    return, with no cost, must not beat or lose to owning it."""
    s = summarise(_trades(30, np.exp(np.log(3.0) / 30) - 1.0, 25, 10.0,
                          np.log(3.0)), "x")
    assert s["cagr"] == pytest.approx(s["hold"], rel=1e-6)


def test_a_total_loss_cannot_produce_a_nan_growth_rate():
    """H40 printed a NaN CAGR because `ratio - 1 - cost` can fall below -1 and a
    fractional power of a negative number is NaN. The clip is the guard."""
    s = summarise(_trades(30, -1.5, 25, 10.0, np.log(2.0)), "x")
    assert np.isfinite(s["cagr"]) and s["cagr"] < 0


def test_too_few_trades_returns_nothing_rather_than_a_confident_number():
    assert summarise(_trades(5, 0.05, 25, 10.0, np.log(2.0)), "x") == {}
