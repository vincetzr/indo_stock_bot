"""Coordination, lead-lag, and campaign-stage analysis.

These use constructed flows with a known relationship, so each metric is checked
against an answer known in advance rather than one that merely looks plausible.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

from idxbot.analytics.coordination import (  # noqa: E402
    campaign_stage_returns,
    coordination_matrix,
    herding_index,
    lead_lag,
    render_plan,
    summarise_stage_returns,
)
from idxbot.config import load_config  # noqa: E402

cfg = load_config()


def _summary(flows: dict, ticker="TEST", start="2024-01-01"):
    """Build a broker-summary frame from {broker: [daily net values]}."""
    n = len(next(iter(flows.values())))
    dates = pd.bdate_range(start, periods=n)
    rows = []
    for broker, series in flows.items():
        for date, net in zip(dates, series):
            rows.append({
                "date": date, "ticker": ticker, "broker": broker,
                "buy_lot": max(net, 0) / 100, "sell_lot": max(-net, 0) / 100,
                "buy_val": max(net, 0.0), "sell_val": max(-net, 0.0),
                "buy_avg": 1000.0, "sell_avg": 1000.0, "source": "test",
            })
    return pd.DataFrame(rows)


# ------------------------------------------------------------ coordination ---
def test_coordination_detects_desks_moving_together():
    rng = np.random.default_rng(1)
    base = rng.normal(0, 1e9, 200)
    # BK and AK share a driver; KZ is independent.
    flows = {
        "BK": base + rng.normal(0, 2e8, 200),
        "AK": base + rng.normal(0, 2e8, 200),
        "KZ": rng.normal(0, 1e9, 200),
    }
    matrix = coordination_matrix(_summary(flows), cfg.brokers, min_days=60)
    assert not matrix.empty
    assert matrix.loc["BK", "AK"] > 0.7
    assert abs(matrix.loc["BK", "KZ"]) < 0.4


def test_coordination_detects_opposite_sides():
    rng = np.random.default_rng(2)
    base = rng.normal(0, 1e9, 200)
    matrix = coordination_matrix(_summary({"BK": base, "AK": -base}),
                                 cfg.brokers, min_days=60)
    assert matrix.loc["BK", "AK"] < -0.9


def test_coordination_needs_enough_history():
    rng = np.random.default_rng(3)
    flows = {"BK": rng.normal(0, 1e9, 20), "AK": rng.normal(0, 1e9, 20)}
    assert coordination_matrix(_summary(flows), cfg.brokers, min_days=60).empty


def test_coordination_empty_input():
    assert coordination_matrix(pd.DataFrame(), cfg.brokers).empty


# ---------------------------------------------------------------- lead-lag ---
def test_lead_lag_identifies_the_leader():
    """AK is BK shifted forward 3 days, so BK must be reported as the leader."""
    rng = np.random.default_rng(4)
    n = 400
    bk = rng.normal(0, 1e9, n)
    ak = np.concatenate([np.zeros(3), bk[:-3]]) + rng.normal(0, 1e8, n)
    out = lead_lag(_summary({"BK": bk, "AK": ak}), cfg.brokers, min_days=90)
    assert not out.empty
    top = out.iloc[0]
    assert top["leader"] == "BK"
    assert top["follower"] == "AK"
    assert top["lag_days"] == 3
    assert top["corr"] > 0.8


def test_lead_lag_needs_history():
    rng = np.random.default_rng(5)
    flows = {"BK": rng.normal(0, 1e9, 30), "AK": rng.normal(0, 1e9, 30)}
    assert lead_lag(_summary(flows), cfg.brokers, min_days=90).empty


# ----------------------------------------------------------------- herding ---
def test_herding_index_when_all_desks_buy():
    flows = {"BK": [1e9] * 10, "AK": [1e9] * 10, "KZ": [1e9] * 10}
    out = herding_index(_summary(flows), cfg.brokers)
    assert not out.empty
    assert (out["buy_share"] == 1.0).all()
    assert (out["desks_active"] == 3).all()


def test_herding_index_when_desks_disagree():
    flows = {"BK": [1e9] * 10, "AK": [-1e9] * 10}
    out = herding_index(_summary(flows), cfg.brokers)
    assert (out["buy_share"] == 0.5).all()


def test_herding_ignores_non_bulge_tiers():
    """YP and PD are retail; herding is a statement about institutions."""
    flows = {"YP": [1e9] * 10, "PD": [1e9] * 10}
    assert herding_index(_summary(flows), cfg.brokers, tier="bulge").empty


# ---------------------------------------------------------- campaign stage ---
def _prices(n=200, start=100.0, drift=0.004):
    dates = pd.bdate_range("2024-01-01", periods=n)
    close = start * (1 + drift) ** np.arange(n)
    return pd.DataFrame({
        "date": dates, "open": close, "high": close * 1.01,
        "low": close * 0.99, "close": close, "volume": 1e7,
    })


def test_campaign_stage_returns_samples_through_the_leg():
    bars = _prices()
    campaigns = pd.DataFrame([{
        "ticker": "TEST", "broker": "BK",
        "acc_start": bars["date"].iloc[10], "acc_end": bars["date"].iloc[60],
    }])
    out = campaign_stage_returns(campaigns, {"TEST": bars},
                                 stages=(0.1, 0.5, 0.9), horizons=(10, 20))
    assert len(out) == 3
    assert set(out["stage"]) == {0.1, 0.5, 0.9}
    # Price rises monotonically, so every forward return is positive.
    assert (out["fwd_10"] > 0).all()


def test_campaign_stage_returns_handles_missing_prices():
    campaigns = pd.DataFrame([{
        "ticker": "NOPE", "broker": "BK",
        "acc_start": pd.Timestamp("2024-01-05"), "acc_end": pd.Timestamp("2024-03-05"),
    }])
    assert campaign_stage_returns(campaigns, {}).empty


def test_campaign_stage_returns_empty():
    assert campaign_stage_returns(pd.DataFrame(), {}).empty


def test_summarise_stage_returns_aggregates():
    frame = pd.DataFrame({
        "stage": [0.1] * 10 + [0.9] * 10,
        "fwd_20": [0.05] * 10 + [-0.02] * 10,
    })
    out = summarise_stage_returns(frame, horizons=(20,))
    assert len(out) == 2
    early = out[out["stage"] == 0.1].iloc[0]
    late = out[out["stage"] == 0.9].iloc[0]
    assert early["mean_20d"] > late["mean_20d"]
    assert early["win_20d"] == pytest.approx(1.0)


# ------------------------------------------------------------------ render ---
def test_render_plan_warns_loudly_on_simulated_data():
    text = render_plan(pd.DataFrame(), pd.DataFrame(), pd.DataFrame(),
                       pd.DataFrame(), data_is_real=False, provenance="synthetic")
    assert "SIMULATED" in text
    # The disclaimer wraps across lines, so assert on a phrase within one line.
    assert "demonstrates that the analysis runs" in text
    assert "Connect real data first" in text


def test_render_plan_omits_warning_on_real_data():
    text = render_plan(pd.DataFrame(), pd.DataFrame(), pd.DataFrame(),
                       pd.DataFrame(), data_is_real=True, provenance="csv")
    assert "SIMULATED" not in text


def test_render_plan_survives_empty_inputs():
    text = render_plan(pd.DataFrame(), pd.DataFrame(), pd.DataFrame(),
                       pd.DataFrame(), data_is_real=True)
    assert "REVERSE-ENGINEERED" in text
