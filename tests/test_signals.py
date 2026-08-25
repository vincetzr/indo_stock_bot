"""Tests for the causal indicator layer and the indicator-conditioned exits.

THE CENTRAL TEST IN THIS FILE IS `test_*_is_causal`. Every other property here
is a convenience; causality is the one that decides whether any number the exit
study prints means anything. It is checked the only way that actually proves it
— recompute each indicator on a truncated prefix of the series and require the
value at bar i to be bit-identical to the value at bar i of the full series.
Reading the code and believing it is not the same test, and a rolling window
with the wrong alignment passes inspection easily.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from idxbot.spine import exits as X
from idxbot.spine import signals as S


def _series(n=400, seed=11):
    rng = np.random.default_rng(seed)
    c = 100.0 * np.cumprod(1.0 + rng.normal(0.0005, 0.03, n))
    spread = np.abs(rng.normal(0, 0.015, n)) * c
    return (c + spread, c - spread, c,
            np.abs(rng.lognormal(15, 1.2, n)))


# ==========================================================================
# causality — the property everything else depends on
# ==========================================================================
@pytest.mark.parametrize("name", ["ema10", "ema20", "ema50", "atr22",
                                  "stoch_k", "stoch_d", "rsi14", "tvz20"])
def test_indicator_is_causal(name):
    """Bar i must not change when bars after i are removed."""
    h, l, c, v = _series()
    full = S.build(pd.DataFrame({"date": pd.bdate_range("2015-01-01",
                                                        periods=len(c)),
                                 "high": h, "low": l, "close": c,
                                 "adj_close": c, "volume": v}))
    for cut in (120, 200, 333):
        pre = S.build(pd.DataFrame({
            "date": pd.bdate_range("2015-01-01", periods=cut),
            "high": h[:cut], "low": l[:cut], "close": c[:cut],
            "adj_close": c[:cut], "volume": v[:cut]}))
        a = full[name].to_numpy()[:cut]
        b = pre[name].to_numpy()
        ok = np.isfinite(a) & np.isfinite(b)
        assert np.isfinite(a).sum() == np.isfinite(b).sum(), name
        assert np.allclose(a[ok], b[ok], rtol=1e-9, atol=1e-12), name


def test_the_causality_check_can_actually_fail():
    """A test that cannot fail proves nothing. Run the SAME harness against an
    indicator that deliberately peeks one bar ahead and require it to trip."""
    _, _, c, _ = _series(200)

    def peeking(x):                       # tomorrow's close, known today
        return pd.Series(x).shift(-1).to_numpy()

    cut = 150
    full = peeking(c)[:cut]
    pre = peeking(c[:cut])
    ok = np.isfinite(full) & np.isfinite(pre)
    #  identical everywhere except the last bar of the prefix, where the full
    #  series knows the next close and the truncated one cannot
    assert np.isfinite(full).sum() != np.isfinite(pre).sum()
    assert np.allclose(full[ok], pre[ok])       # the leak is only at the edge
    assert np.isfinite(full[cut - 1]) and not np.isfinite(pre[cut - 1])


# ==========================================================================
# the indicators themselves
# ==========================================================================
def test_ema_is_the_recursive_form_not_the_expanding_one():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    a = 2.0 / (3 + 1)
    want, e = [], x[0]
    for v in x:
        e = v if not want else a * v + (1 - a) * e
        want.append(e)
    assert np.allclose(S.ema(x, 3), want)


def test_true_range_uses_the_previous_close():
    h = np.array([10.0, 12.0]); l = np.array([9.0, 11.5]); c = np.array([9.5, 12.0])
    #  bar 1: h-l = 0.5, |h - c_prev| = 2.5, |l - c_prev| = 2.0  -> 2.5
    assert S.true_range(h, l, c)[1] == pytest.approx(2.5)


def test_true_range_first_bar_has_no_prior_close():
    tr = S.true_range(np.array([10.0]), np.array([9.0]), np.array([9.5]))
    assert tr[0] == pytest.approx(1.0)


def test_atr_is_wilder_smoothed_not_simple_mean():
    h, l, c, _ = _series(120)
    w = S.atr(h, l, c, 14)
    simple = pd.Series(S.true_range(h, l, c)).rolling(14).mean().to_numpy()
    assert not np.allclose(w[50:], simple[50:])


def test_stochastic_is_100_at_the_top_of_its_range_and_0_at_the_bottom():
    n = 40
    c = np.concatenate([np.full(n - 3, 10.0), [20.0, 20.0, 20.0]])
    k, d = S.stochastic(c + 0.001, c - 0.001, c, k=14, d=3, smooth=3)
    assert k[-1] == pytest.approx(100.0, abs=0.5)
    c2 = np.concatenate([np.full(n - 3, 20.0), [10.0, 10.0, 10.0]])
    k2, _ = S.stochastic(c2 + 0.001, c2 - 0.001, c2, k=14, d=3, smooth=3)
    assert k2[-1] == pytest.approx(0.0, abs=0.5)


def test_stochastic_is_nan_on_a_flat_window_not_an_arbitrary_fifty():
    """A suspended or limit-locked name has no position in its own range."""
    c = np.full(40, 5.0)
    k, _ = S.stochastic(c, c, c)
    assert np.isnan(k[-1])


def test_rsi_saturates_high_on_an_unbroken_advance():
    c = np.cumprod(np.full(60, 1.01)) * 100
    assert S.rsi(c, 14)[-1] == pytest.approx(100.0, abs=1e-6)


def test_rsi_saturates_low_on_an_unbroken_decline():
    c = np.cumprod(np.full(60, 0.99)) * 100
    assert S.rsi(c, 14)[-1] == pytest.approx(0.0, abs=1e-6)


def test_turnover_z_flags_a_spike_and_is_scale_invariant():
    v = np.full(60, 1e6); v[-1] = 1e8
    c = np.full(60, 100.0)
    z = S.turnover_z(v, c, 20)
    assert z[-1] > 3
    assert S.turnover_z(v * 7.0, c, 20)[-1] == pytest.approx(z[-1])


def test_build_rebases_high_and_low_onto_the_adjusted_close():
    """Raw high with adjusted close is a split-shaped error waiting to happen."""
    n = 30
    d = pd.DataFrame({"date": pd.bdate_range("2020-01-01", periods=n),
                      "high": np.full(n, 110.0), "low": np.full(n, 90.0),
                      "close": np.full(n, 100.0),
                      "adj_close": np.full(n, 50.0),      # factor 0.5
                      "volume": np.full(n, 1e6)})
    out = S.build(d)
    assert out["adj_high"].iloc[0] == pytest.approx(55.0)
    assert out["adj_low"].iloc[0] == pytest.approx(45.0)
    assert out["close"].iloc[0] == pytest.approx(50.0)


def test_build_returns_nan_rather_than_guessing_a_missing_high():
    n = 30
    d = pd.DataFrame({"date": pd.bdate_range("2020-01-01", periods=n),
                      "high": np.full(n, np.nan), "low": np.full(n, np.nan),
                      "close": np.full(n, 100.0), "adj_close": np.full(n, 100.0),
                      "volume": np.full(n, 1e6)})
    out = S.build(d)
    assert out["adj_high"].isna().all()
    assert out["stoch_k"].isna().all()


# ==========================================================================
# the indicator exit rules
# ==========================================================================
def _F(path, **over):
    """Feature frame for a normalised path, defaults benign (nothing fires)."""
    n = len(path)
    F = {"close": np.asarray(path, float) * 100.0,
         "high": np.asarray(path, float) * 100.0,
         "ema10": np.zeros(n), "ema20": np.zeros(n), "ema30": np.zeros(n),
         "ema50": np.zeros(n), "atr22": np.full(n, 1e9),
         "stoch_k": np.full(n, np.nan), "stoch_d": np.full(n, np.nan),
         "tvz20": np.zeros(n)}
    F.update({k: np.asarray(v, float) for k, v in over.items()})
    return F


def test_ema_break_fires_on_the_first_close_below_the_line_after_warm_up():
    p = np.array([1.0] * 10 + [0.9] * 5)
    F = _F(p, ema20=np.array([50.0] * 10 + [200.0] * 5))
    r, held = X.ema_break(p, F, 20, warm=5)
    #   fires at bar 10, fills at bar 11 (one-bar delay)
    assert held == 12 and r == pytest.approx(-0.1)


def test_ema_break_holds_when_the_line_is_never_broken():
    p = np.linspace(1.0, 2.0, 40)
    assert X.ema_break(p, _F(p), 20) == X.hold(p)


def test_warm_up_stops_a_rule_firing_on_the_entry_bar():
    p = np.array([0.9] * 20)
    F = _F(p, ema20=np.full(20, 1e9))
    assert X.ema_break(p, F, 20, warm=0)[1] == 2      # fires bar 0, fills bar 1
    assert X.ema_break(p, F, 20, warm=10)[1] == 12


def test_the_arm_suppresses_an_indicator_rule_until_the_position_has_run():
    p = np.array([1.0] * 5 + [0.8] * 20)
    F = _F(p, ema20=np.full(25, 1e9))
    assert X.ema_break(p, F, 20, arm=0.50, warm=0) == X.hold(p)
    assert X.ema_break(p, F, 20, arm=0.0, warm=0)[1] == 2


def test_chandelier_measures_give_back_in_atr_not_percent():
    """The whole reason to prefer it: same rule for a quiet and a wild name."""
    p = np.array([1.0] * 6 + [2.0, 1.9, 1.8, 1.7])
    hi = p * 100.0
    quiet = _F(p, high=hi, atr22=np.full(len(p), 2.0))    # 3 ATR = 6 points
    wild = _F(p, high=hi, atr22=np.full(len(p), 20.0))    # 3 ATR = 60 points
    assert X.chandelier(p, quiet, 3.0, warm=0)[1] < len(p)
    assert X.chandelier(p, wild, 3.0, warm=0) == X.hold(p)


def test_chandelier_trails_the_highest_high_since_entry_not_the_close():
    """The discriminating case: the intraday high ran further than the close,
    so trailing the high arms a tighter stop and fires a bar earlier."""
    p = np.array([1.0, 3.0, 1.0, 1.0, 1.0])
    close = p * 100.0                       # 100, 300, 100, 100, 100
    atr = np.full(5, 50.0)                  # 3 ATR = 150

    #  high peaks at 500 -> threshold 350 -> the 300 close at bar 1 fires
    hi = X.chandelier(p, _F(p, high=np.array([100., 500., 100., 100., 100.]),
                            atr22=atr), 3.0, warm=0)
    #  trailing the close instead peaks at 300 -> threshold 150 -> bar 2 fires
    lo = X.chandelier(p, _F(p, high=close, atr22=atr), 3.0, warm=0)
    assert hi[1] == 3 and lo[1] == 4


def test_stoch_rollover_needs_both_overbought_and_the_cross():
    n = 20
    k = np.full(n, 90.0); d = np.full(n, 50.0)
    p = np.ones(n)
    assert X.stoch_rollover(p, _F(p, stoch_k=k, stoch_d=d), warm=0) == X.hold(p)
    k2 = k.copy(); k2[10:] = 40.0                       # crosses below D
    assert X.stoch_rollover(p, _F(p, stoch_k=k2, stoch_d=d), warm=0)[1] == 12


def test_stoch_cool_requires_having_been_hot_first():
    n = 20
    p = np.ones(n)
    never = np.full(n, 30.0)                            # cool but never hot
    assert X.stoch_cool(p, _F(p, stoch_k=never), warm=0) == X.hold(p)
    was = np.concatenate([np.full(5, 90.0), np.full(15, 30.0)])
    assert X.stoch_cool(p, _F(p, stoch_k=was), warm=0)[1] == 7


def test_volume_climax_needs_a_DOWN_bar_not_just_a_big_one():
    n = 20
    up = np.linspace(1.0, 2.0, n)
    z = np.zeros(n); z[10] = 4.0
    assert X.volume_climax(up, _F(up, tvz20=z), arm=0.0, warm=0) == X.hold(up)
    dn = up.copy(); dn[10] = up[9] - 0.01
    assert X.volume_climax(dn, _F(dn, tvz20=z), arm=0.0, warm=0)[1] == 12


def test_every_indicator_rule_returns_a_finite_pair_on_a_real_shaped_path():
    rng = np.random.default_rng(5)
    p = np.cumprod(1.0 + rng.normal(0.001, 0.04, 252))
    h, l, c, v = _series(252, seed=6)
    b = S.build(pd.DataFrame({"date": pd.bdate_range("2015-01-01", periods=252),
                              "high": h, "low": l, "close": c,
                              "adj_close": c, "volume": v}))
    F = {"close": b["close"].to_numpy(), "high": b["adj_high"].to_numpy(),
         "ema10": b["ema10"].to_numpy(), "ema20": b["ema20"].to_numpy(),
         "ema30": b["ema30"].to_numpy(), "ema50": b["ema50"].to_numpy(),
         "atr22": b["atr22"].to_numpy(), "stoch_k": b["stoch_k"].to_numpy(),
         "stoch_d": b["stoch_d"].to_numpy(), "tvz20": b["tvz20"].to_numpy()}
    for name, fn in X.indicator_catalogue().items():
        r, held = fn(p, F)
        assert np.isfinite(r), name
        assert 1 <= held <= X.HORIZON, name


def test_no_indicator_rule_beats_the_paths_own_maximum():
    """Exiting above the highest price the path ever reached is look-ahead."""
    rng = np.random.default_rng(9)
    for s in range(6):
        p = np.cumprod(1.0 + rng.normal(0.001, 0.05, 252))
        h, l, c, v = _series(252, seed=20 + s)
        b = S.build(pd.DataFrame({
            "date": pd.bdate_range("2015-01-01", periods=252),
            "high": h, "low": l, "close": c, "adj_close": c, "volume": v}))
        F = {"close": b["close"].to_numpy(), "high": b["adj_high"].to_numpy(),
             "ema10": b["ema10"].to_numpy(), "ema20": b["ema20"].to_numpy(),
             "ema30": b["ema30"].to_numpy(), "ema50": b["ema50"].to_numpy(),
             "atr22": b["atr22"].to_numpy(), "stoch_k": b["stoch_k"].to_numpy(),
             "stoch_d": b["stoch_d"].to_numpy(), "tvz20": b["tvz20"].to_numpy()}
        best = float(np.max(p)) - 1.0
        for name, fn in X.indicator_catalogue().items():
            assert fn(p, F)[0] <= best + 1e-9, name


def test_the_catalogue_carries_a_registered_null():
    assert "NULL random exit" in X.indicator_catalogue()


def test_the_null_exit_is_deterministic_and_ignores_the_indicators():
    p = np.cumprod(1.0 + np.random.default_rng(3).normal(0, 0.03, 252))
    a = X._random_exit(p, None)
    assert a == X._random_exit(p, None)
    assert a == X._random_exit(p, _F(p, ema20=np.full(252, 1e9)))


def test_apply_rule_drops_a_name_whose_indicators_are_missing():
    """Silently holding to the horizon would average two different rules."""
    p = np.array([1.0, 1.1, 0.5])
    D = X.apply_rule([p, p], [0.01, 0.01],
                     X.indicator_catalogue()["ema20 break"],
                     feats=[_F(p, ema20=np.full(3, 1e9)), None])
    assert np.isfinite(D["net"].iloc[0])
    assert not np.isfinite(D["net"].iloc[1])


def test_price_rules_still_work_when_a_feature_frame_is_supplied():
    p = np.array([1.0, 2.0, 1.5])
    D = X.apply_rule([p], [0.0], X.catalogue()["trail 20%"], feats=[_F(p)])
    assert np.isfinite(D["net"].iloc[0])


def test_first_takes_whichever_rule_fired_earliest():
    assert X._first((0.5, 10), (0.2, 3), (0.9, 40)) == (0.2, 3)


def test_first_ignores_a_rule_that_returned_nan():
    assert X._first((np.nan, 0), (0.2, 3)) == (0.2, 3)
