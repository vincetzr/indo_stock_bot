"""Tests for the Hull and UT Bot indicator ports.

Two kinds of assertion here, and only the second kind is worth much.

The first checks each function against an independent naive implementation, so
a fast convolution cannot quietly disagree with the definition it claims to
implement.

The second is the one that matters: **no value may change when the future is
removed.** An indicator that peeks is not detectable by reading it, only by
truncating the input and recomputing. This repo has shipped a look-ahead bug
before (``docs/FINDINGS.md``, the 96%-win-rate dip simulation), so that check
is here by default rather than as an afterthought.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from idxbot.indicators import (
    HULL_MODES, atr, ehma, ema, hma, hull, hull_is_green, rma, thma,
    true_range, ut_bot, ut_bot_stop, warmup_bars, wma,
)


@pytest.fixture(scope="module")
def series() -> pd.Series:
    rng = np.random.default_rng(20260817)
    return pd.Series(100.0 + np.cumsum(rng.normal(0, 1.0, 400)))


@pytest.fixture(scope="module")
def bars(series) -> pd.DataFrame:
    rng = np.random.default_rng(11)
    span = np.abs(rng.normal(1.0, 0.4, len(series)))
    return pd.DataFrame({
        "high": series + span,
        "low": series - span,
        "close": series,
    })


def naive_wma(s: pd.Series, n: int) -> pd.Series:
    w = np.arange(1, n + 1)
    return s.rolling(n).apply(lambda v: float(np.dot(v, w) / w.sum()), raw=True)


# ---------------------------------------------------------------------------
# moving averages against independent implementations
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("n", [2, 3, 5, 9, 21, 55])
def test_wma_matches_the_definition(series, n):
    fast, slow = wma(series, n), naive_wma(series, n)
    assert np.nanmax(np.abs(fast - slow)) < 1e-9
    assert (fast.isna() == slow.isna()).all()


def test_wma_weights_the_latest_bar_heaviest():
    """A rising ramp must sit above its own simple mean."""
    ramp = pd.Series(np.arange(1.0, 21.0))
    assert wma(ramp, 10).iloc[-1] > ramp.rolling(10).mean().iloc[-1]


def test_wma_does_not_smear_nan(series):
    dirty = series.copy()
    dirty.iloc[100] = np.nan
    out = wma(dirty, 10)
    assert out.iloc[100:110].isna().all()
    assert out.iloc[111:120].notna().all()


@pytest.mark.parametrize("n", [10, 14, 21])
def test_rma_matches_wilder(series, n):
    values = series.to_numpy(float)
    expected = np.full(len(values), np.nan)
    expected[n - 1] = values[:n].mean()
    for i in range(n, len(values)):
        expected[i] = (expected[i - 1] * (n - 1) + values[i]) / n
    assert np.nanmax(np.abs(rma(series, n) - expected)) < 1e-9


def test_rma_is_not_an_sma_and_not_a_same_length_ema(series):
    """The most common way a Pine port silently drifts.

    Wilder's alpha of 1/n is the same recursion as an EMA of span 2n-1, so the
    two converge to identical values - but only asymptotically. They seed
    differently (RMA from an SMA of 14 at bar 13, the EMA from an SMA of 27 at
    bar 26), so they agree in the tail and not at the start. Asserting equality
    over the whole series would be wrong; asserting it in the tail is the real
    statement.
    """
    tail = slice(-50, None)
    assert np.nanmax(np.abs((rma(series, 14) - ema(series, 27)).iloc[tail])) < 1e-9
    # ... and it is emphatically not the same-length EMA or a simple mean.
    assert np.nanmedian(np.abs(rma(series, 14) - ema(series, 14))) > 0.1
    assert np.nanmedian(np.abs(rma(series, 14) - series.rolling(14).mean())) > 0.1


def test_ema_of_length_one_is_the_series_itself(series):
    """``ema(src, 1) == src`` - the identity that collapses UT Bot's buy rule."""
    pd.testing.assert_series_equal(ema(series, 1), series.astype(float),
                                   check_names=False)


# ---------------------------------------------------------------------------
# Hull family
# ---------------------------------------------------------------------------
def test_hma_matches_nested_definition(series):
    n = 55
    expected = naive_wma(2 * naive_wma(series, round(n / 2))
                         - naive_wma(series, n), round(math.sqrt(n)))
    assert np.nanmax(np.abs(hma(series, n) - expected)) < 1e-9


def test_hull_dispatches_all_three_modes(series):
    for mode in HULL_MODES:
        line = hull(series, 55, mode)
        assert line.notna().sum() > 100, mode


def test_thma_is_invoked_at_half_length(series):
    """The published script's ``Mode()`` passes ``length/2`` to THMA only.

    Miss it and the effective lookback doubles, which is a different indicator
    wearing the same parameter.
    """
    pd.testing.assert_series_equal(hull(series, 55, "thma"),
                                   thma(series, round(55 / 2)),
                                   check_names=False)
    # Calling THMA at the full length is a materially different line, which is
    # precisely why the half-length dispatch has to be preserved.
    half, full = hull(series, 55, "thma"), thma(series, 55)
    both = half.notna() & full.notna()
    assert both.sum() > 50
    assert np.nanmedian(np.abs(half[both] - full[both])) > 1e-6


def test_hma_uses_rounding_not_truncation():
    """``round(n/2)`` and ``round(sqrt(n))``, as Pine does.

    At n=50 truncation gives sqrt->7 where rounding gives 7, but at n=55 they
    differ (7.416 -> 7 both ways) while n/2 at odd lengths does not. n=45 is the
    case that separates them: sqrt(45)=6.7 rounds to 7, truncates to 6.
    """
    assert round(math.sqrt(45)) == 7 and int(math.sqrt(45)) == 6
    ramp = pd.Series(np.arange(1.0, 200.0))
    expected = naive_wma(2 * naive_wma(ramp, round(45 / 2)) - naive_wma(ramp, 45), 7)
    assert np.nanmax(np.abs(hma(ramp, 45) - expected)) < 1e-9


def test_ehma_uses_ema_throughout(series):
    n = 55
    expected = ema(2 * ema(series, round(n / 2)) - ema(series, n),
                   round(math.sqrt(n)))
    assert np.nanmax(np.abs(ehma(series, n) - expected)) < 1e-9


def test_hull_colour_compares_against_two_bars_back(series):
    """Green is ``HULL > HULL[2]``, not ``HULL[1]``.

    The two-bar comparison is a slope filter with one bar of hysteresis. Using
    one bar produces a visibly different, noisier signal - asserted here so the
    difference cannot be introduced silently.
    """
    line = hull(series, 55)
    green = hull_is_green(line)
    manual = line > line.shift(2)
    valid = line.notna() & line.shift(2).notna()
    assert (green[valid].astype(bool) == manual[valid]).all()

    one_bar = (line > line.shift(1))[valid]
    assert (one_bar != manual[valid]).sum() > 10, "1-bar and 2-bar must differ"


def test_hull_colour_is_unknown_during_warmup(series):
    green = hull_is_green(hull(series, 55))
    assert green.iloc[:50].isna().all()


def test_unknown_colour_is_never_silently_red(series):
    """NaN must not collapse to False anywhere in the indicator layer.

    If it did, the warm-up would read as a bearish regime and a strategy that
    exits on red would start every series flat for a spurious reason.
    """
    green = hull_is_green(hull(series, 55))
    assert green.dtype == "boolean"
    assert green.isna().any()


def test_hull_rejects_unknown_mode(series):
    with pytest.raises(ValueError):
        hull(series, 55, "kaufman")


# ---------------------------------------------------------------------------
# UT Bot
# ---------------------------------------------------------------------------
def test_true_range_first_bar_is_high_minus_low(bars):
    tr = true_range(bars["high"], bars["low"], bars["close"])
    assert tr.iloc[0] == pytest.approx(bars["high"].iloc[0] - bars["low"].iloc[0])
    assert tr.notna().all()


def test_true_range_spans_the_gap(bars):
    """A gap up must make TR exceed the bar's own range."""
    high = pd.Series([10.0, 20.0])
    low = pd.Series([9.0, 19.0])
    close = pd.Series([9.5, 19.5])
    assert true_range(high, low, close).iloc[1] == pytest.approx(20.0 - 9.5)


def test_atr_is_wilder_smoothed(bars):
    expected = rma(true_range(bars["high"], bars["low"], bars["close"]), 10)
    assert np.nanmax(np.abs(atr(bars["high"], bars["low"], bars["close"], 10)
                            - expected)) < 1e-9


def test_stop_ratchets_up_in_an_uptrend():
    """The defining property: a trailing stop never loosens while trend holds."""
    ramp = pd.Series(np.arange(100.0, 200.0))
    stop = ut_bot_stop(ramp, pd.Series(np.full(len(ramp), 2.0)), key=1.0)
    live = stop.dropna()
    assert (live.diff().dropna() >= -1e-9).all()


def test_stop_flips_side_on_a_reversal():
    path = pd.Series(list(np.arange(100.0, 140.0)) + list(np.arange(140.0, 100.0, -1.0)))
    stop = ut_bot_stop(path, pd.Series(np.full(len(path), 2.0)), key=1.0)
    assert (path.iloc[:40] > stop.iloc[:40]).sum() > 30      # below price on the way up
    assert (path.iloc[-20:] < stop.iloc[-20:]).sum() > 10    # above price on the way down


def test_buy_reduces_to_a_plain_crossover(bars):
    """The published rule is ``src > stop and crossover(ema(src,1), stop)``.

    ``ema(x, 1)`` is ``x`` and a crossover already implies the inequality, so
    the whole expression is one crossover. Asserted rather than assumed.
    """
    out = ut_bot(bars["high"], bars["low"], bars["close"], key=1.0, atr_length=10)
    stop, src = out["stop"], bars["close"]
    manual = ((src > stop) & (src.shift(1) <= stop.shift(1))
              & stop.notna() & stop.shift(1).notna()).fillna(False)
    assert (out["buy"] == manual).all()
    assert out["buy"].sum() > 3


def test_buy_and_sell_never_fire_on_the_same_bar(bars):
    out = ut_bot(bars["high"], bars["low"], bars["close"])
    assert not (out["buy"] & out["sell"]).any()


def test_larger_key_means_a_wider_stop_and_fewer_signals(bars):
    tight = ut_bot(bars["high"], bars["low"], bars["close"], key=0.5)
    wide = ut_bot(bars["high"], bars["low"], bars["close"], key=4.0)
    assert wide["buy"].sum() < tight["buy"].sum()


def test_stop_is_nan_until_atr_exists(bars):
    """A zero-seeded stop would read as "price is far above it" and fake a buy."""
    out = ut_bot(bars["high"], bars["low"], bars["close"], atr_length=10)
    assert out["stop"].iloc[:9].isna().all()
    assert out["buy"].iloc[:9].sum() == 0


# ---------------------------------------------------------------------------
# the check that actually matters
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("cut", [150, 220, 300, 399])
def test_no_indicator_changes_when_the_future_is_removed(bars, cut):
    """Truncate the input at ``cut`` and recompute; every value must be identical.

    This is the only test in the file that could catch a look-ahead bug, and it
    is deliberately parameterised across several cut points rather than run
    once at the end of the series.
    """
    full_hull = hull(bars["close"], 55)
    full_green = hull_is_green(full_hull)
    full_ut = ut_bot(bars["high"], bars["low"], bars["close"], 1.0, 10)

    part = bars.iloc[:cut + 1]
    part_hull = hull(part["close"], 55)
    part_green = hull_is_green(part_hull)
    part_ut = ut_bot(part["high"], part["low"], part["close"], 1.0, 10)

    assert part_hull.iloc[cut] == pytest.approx(full_hull.iloc[cut], abs=1e-9)
    assert part_ut["stop"].iloc[cut] == pytest.approx(full_ut["stop"].iloc[cut], abs=1e-9)
    assert bool(part_green.iloc[cut]) == bool(full_green.iloc[cut])
    assert bool(part_ut["buy"].iloc[cut]) == bool(full_ut["buy"].iloc[cut])
    assert bool(part_ut["sell"].iloc[cut]) == bool(full_ut["sell"].iloc[cut])


def test_the_whole_signal_history_is_stable_under_truncation(bars):
    """Stronger than the spot checks: every bar of the shorter run must agree."""
    cut = 300
    full = ut_bot(bars["high"], bars["low"], bars["close"], 1.0, 10)["stop"]
    part = ut_bot(bars["high"].iloc[:cut], bars["low"].iloc[:cut],
                  bars["close"].iloc[:cut], 1.0, 10)["stop"]
    assert np.nanmax(np.abs(part - full.iloc[:cut])) < 1e-9


def test_warmup_covers_the_longest_dependency_chain():
    assert warmup_bars(55, "hma", 10) >= 55 + round(math.sqrt(55)) + 2
    assert warmup_bars(21, "hma", 50) >= 150       # ATR can be the binding one
    assert warmup_bars(144, "hma", 21) > warmup_bars(55, "hma", 10)


def test_indicators_survive_a_short_series():
    tiny = pd.Series([1.0, 2.0, 3.0])
    assert hull(tiny, 55).isna().all()
    assert ut_bot(tiny + 1, tiny - 1, tiny, atr_length=10)["stop"].isna().all()
