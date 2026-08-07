"""Scoring: weight renormalisation, price-only mode, Wyckoff, no look-ahead."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

from idxbot.analytics import indicators, wyckoff  # noqa: E402
from idxbot.analytics.accumulation import (  # noqa: E402
    BROKER_COMPONENTS,
    _normalise_weights,
    score,
    sigmoid,
)
from idxbot.config import load_config  # noqa: E402

cfg = load_config()


def _bars(n=400, seed=7):
    """A plausible OHLCV series: a decline, a base, then a markup."""
    rng = np.random.default_rng(seed)
    decline = np.linspace(1000, 700, n // 3)
    base = 700 + rng.normal(0, 8, n // 3)
    markup = np.linspace(700, 950, n - 2 * (n // 3))
    close = np.concatenate([decline, base, markup])
    noise = rng.normal(0, 4, len(close))
    close = np.maximum(close + noise, 50)

    df = pd.DataFrame({
        "date": pd.bdate_range("2022-01-03", periods=len(close)),
        "open": close * (1 + rng.normal(0, 0.003, len(close))),
        "close": close,
        "volume": rng.lognormal(15, 0.4, len(close)),
    })
    df["high"] = np.maximum(df["open"], df["close"]) * 1.01
    df["low"] = np.minimum(df["open"], df["close"]) * 0.99
    df["adj_close"] = df["close"]
    return df[["date", "open", "high", "low", "close", "adj_close", "volume"]]


# ----------------------------------------------------------------- sigmoid ---
def test_sigmoid_bounds_and_centre():
    assert sigmoid(0.0) == pytest.approx(0.5)
    assert 0.0 < sigmoid(-5.0) < 0.5 < sigmoid(5.0) < 1.0


def test_sigmoid_survives_extreme_inputs():
    """Huge inputs must saturate, not overflow."""
    with np.errstate(over="raise"):
        assert sigmoid(1e9, scale=1e-6) == pytest.approx(1.0)
        assert sigmoid(-1e9, scale=1e-6) == pytest.approx(0.0)
    assert sigmoid(float("nan")) == 0.5
    assert sigmoid(float("inf")) == 0.5


# ----------------------------------------------------------------- weights ---
def test_weights_renormalise_to_one():
    weights = {"a": 0.2, "b": 0.3, "c": 0.5}
    out = _normalise_weights(weights, {"a", "b", "c"})
    assert sum(out.values()) == pytest.approx(1.0)


def test_weights_renormalise_when_components_missing():
    """Dropping broker components must not shrink the score's scale."""
    weights = {"a": 0.2, "b": 0.3, "c": 0.5}
    out = _normalise_weights(weights, {"a", "b"})
    assert sum(out.values()) == pytest.approx(1.0)
    assert set(out) == {"a", "b"}
    assert out["a"] == pytest.approx(0.4)


def test_weights_empty_available():
    assert _normalise_weights({"a": 1.0}, set()) == {}


# ------------------------------------------------------------------- score ---
def test_score_price_only_mode_without_broker_data():
    bars = indicators.enrich(_bars(), cfg=cfg)
    signal = score(bars, cfg, flow=None, ticker="TEST")

    assert signal.data_mode == "price-only"
    assert 0.0 <= signal.score <= 100.0
    assert not (set(signal.components) & BROKER_COMPONENTS)
    assert sum(signal.weights_used.values()) == pytest.approx(1.0)


def test_score_is_bounded_across_the_whole_series():
    bars = indicators.enrich(_bars(), cfg=cfg)
    for i in range(300, len(bars), 17):
        signal = score(bars, cfg, index=i, ticker="TEST")
        assert 0.0 <= signal.score <= 100.0


def test_score_has_no_look_ahead():
    """Scoring at bar i must not change when future bars are appended.

    This is the property the backtester depends on. If it fails, every
    backtested number is contaminated.
    """
    bars = indicators.enrich(_bars(), cfg=cfg)
    cut = 340

    truncated = indicators.enrich(_bars().iloc[:cut + 1].copy(), cfg=cfg)
    full_signal = score(bars, cfg, index=cut, ticker="TEST")
    truncated_signal = score(truncated, cfg, index=cut, ticker="TEST")

    assert full_signal.score == pytest.approx(truncated_signal.score, abs=1e-6)
    assert full_signal.wyckoff_state.phase == truncated_signal.wyckoff_state.phase


def test_score_empty_bars():
    signal = score(pd.DataFrame(), cfg, ticker="TEST")
    assert signal.score == 0.0


def test_level_thresholds():
    bars = indicators.enrich(_bars(), cfg=cfg)
    signal = score(bars, cfg, ticker="TEST")
    signal.score = 80.0
    assert signal.level == "STRONG"
    signal.score = 70.0
    assert signal.level == "SIGNAL"
    signal.score = 55.0
    assert signal.level == "WATCH"
    signal.score = 10.0
    assert signal.level == "NONE"


# ----------------------------------------------------------------- wyckoff ---
def test_wyckoff_returns_a_valid_phase():
    bars = indicators.enrich(_bars(), cfg=cfg)
    state = wyckoff.classify(bars)
    assert state.phase in wyckoff.PHASES
    assert 0.0 <= state.confidence <= 1.0
    assert state.meaning


def test_wyckoff_insufficient_history():
    state = wyckoff.classify(_bars(n=20))
    assert state.phase == "none"


def test_phase_score_ordering():
    """The spring must outrank an extended markup."""
    def make(phase, conf=1.0):
        s = wyckoff.WyckoffState(phase=phase, confidence=conf)
        return wyckoff.phase_score(s)

    assert make("C") > make("B") > make("A")
    assert make("C") > make("E")
    assert make("none") == 0.0


# -------------------------------------------------------------- indicators ---
def test_indicators_do_not_look_ahead():
    bars = _bars()
    full = indicators.enrich(bars, cfg=cfg)
    cut = 300
    partial = indicators.enrich(bars.iloc[:cut + 1].copy(), cfg=cfg)

    for col in ("atr", "obv", "cmf", "vol_ratio", "range_compression", "range_pos_60"):
        a, b = full[col].iloc[cut], partial[col].iloc[cut]
        if np.isfinite(a) and np.isfinite(b):
            assert a == pytest.approx(b, rel=1e-6), col


def test_enrich_adds_expected_columns():
    out = indicators.enrich(_bars(), cfg=cfg)
    for col in ("atr", "obv", "ad", "cmf", "mfi", "vol_ratio",
                "range_compression", "obv_divergence", "drawdown"):
        assert col in out.columns


def test_forward_return_shifts_backwards():
    close = pd.Series([100.0, 110.0, 121.0])
    fwd = indicators.forward_return(close, 1)
    assert fwd.iloc[0] == pytest.approx(0.10)
    assert np.isnan(fwd.iloc[-1])
