"""Day-trade scanner, intraday path resolution, and the ORB entry rule."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

from idxbot import daytrade as dt  # noqa: E402
from idxbot.config import load_config  # noqa: E402
from idxbot.data.intraday import (  # noqa: E402
    enrich_session,
    opening_range,
    resolve_path,
    volume_pace,
)

cfg = load_config()


def _daily(n=120, last_rvol=10.0, last_return=0.10, at_high=True, price=1000.0):
    """Daily bars ending in a controllable burst day."""
    dates = pd.bdate_range("2025-01-01", periods=n)
    close = np.full(n, price)
    close[:-1] = price * 0.9        # prior days sit below, so the last is a high
    if not at_high:
        close[-20:-1] = price * 1.5  # put a higher high in the recent window
    volume = np.full(n, 1_000_000.0)
    volume[-1] = 1_000_000.0 * last_rvol
    close[-1] = close[-2] * (1 + last_return)

    df = pd.DataFrame({
        "date": dates,
        "open": close * 0.99,
        "high": close * 1.005,
        "low": close * 0.97,
        "close": close,
        "volume": volume,
    })
    df["adj_close"] = df["close"]
    return df


def _session(opens, highs, lows, closes, volumes=None, start="2025-06-02 09:00"):
    """A 5-minute session frame."""
    n = len(opens)
    ts = pd.date_range(start, periods=n, freq="5min")
    df = pd.DataFrame({
        "ts": ts, "open": opens, "high": highs, "low": lows, "close": closes,
        "volume": volumes if volumes is not None else np.full(n, 100_000.0),
    })
    return enrich_session(df)


# ------------------------------------------------------------------- scan ---
def test_scan_detects_a_burst():
    bars = {"TEST": _daily(last_rvol=10, last_return=0.10, at_high=True)}
    found = dt.scan(bars, cfg, min_value_traded=0)
    assert found, "an obvious burst must be detected"
    assert found[0].setup == "burst"
    assert found[0].rvol > 8


def test_scan_classifies_weaker_setup_as_surge():
    bars = {"TEST": _daily(last_rvol=6, last_return=0.06, at_high=True)}
    found = dt.scan(bars, cfg, min_value_traded=0)
    assert found and found[0].setup == "surge"


def test_scan_ignores_quiet_days():
    bars = {"TEST": _daily(last_rvol=1.0, last_return=0.005)}
    assert dt.scan(bars, cfg, min_value_traded=0) == []


def test_scan_rejects_illiquid_names():
    """A burst in a name nobody trades is not actionable - drop it entirely."""
    bars = {"TEST": _daily(last_rvol=10, last_return=0.10)}
    assert dt.scan(bars, cfg, min_value_traded=1e15) == []


def test_scan_rejects_suspended_names():
    """A suspended stock reopening posts a huge rvol; that is not a burst.

    Momentum measures rate a frozen price highly - it sits above its moving
    average and its close equals its high - so this must be filtered before
    ranking, not after.
    """
    bars = _daily(last_rvol=10, last_return=0.10)
    bars.loc[bars.index[-25:-1], "volume"] = 0.0
    for col in ("open", "high", "low", "close"):
        bars.loc[bars.index[-25:-1], col] = 500.0
    assert dt.scan({"TEST": bars}, cfg, min_value_traded=0) == []


def test_scan_requires_enough_history():
    assert dt.scan({"TEST": _daily(n=30)}, cfg, min_value_traded=0) == []


# ---------------------------------------------------------- opening range ---
def test_opening_range_uses_only_the_first_bars():
    session = _session([100] * 12, [105, 110] + [102] * 10, [98] * 12, [100] * 12)
    orange = opening_range(session, minutes=30)
    # 30 minutes = the first 7 bars at 5-minute resolution (09:00..09:30).
    assert orange["or_high"] == 110
    assert orange["or_low"] == 98


def test_volume_pace_scales_with_session_progress():
    session = _session([100] * 12, [101] * 12, [99] * 12, [100] * 12,
                       volumes=np.full(12, 100_000.0))
    pace = volume_pace(session, reference_daily_volume=1_200_000.0)
    assert np.isfinite(pace) and pace > 0


# ------------------------------------------------------------ path resolve ---
def test_resolve_path_target_first():
    session = _session([100] * 6, [100, 101, 106, 106, 106, 106],
                       [99, 99, 99, 99, 95, 95], [100] * 6)
    out = resolve_path(session, entry=100, target=105, stop=97)
    assert out["outcome"] == "target"


def test_resolve_path_stop_first():
    session = _session([100] * 6, [100, 101, 101, 106, 106, 106],
                       [99, 96, 96, 96, 96, 96], [100] * 6)
    out = resolve_path(session, entry=100, target=105, stop=97)
    assert out["outcome"] == "stop"


def test_resolve_path_reports_ambiguity_rather_than_guessing():
    """Both levels inside one bar is genuinely unknowable - say so."""
    session = _session([100] * 3, [106, 106, 106], [96, 96, 96], [100] * 3)
    out = resolve_path(session, entry=100, target=105, stop=97)
    assert out["outcome"] == "ambiguous"


def test_resolve_path_falls_through_to_close():
    session = _session([100] * 6, [101] * 6, [99] * 6, [100, 100, 100, 100, 100, 102])
    out = resolve_path(session, entry=100, target=105, stop=97)
    assert out["outcome"] == "close"
    assert out["return"] == pytest.approx(0.02)


# --------------------------------------------------------------- ORB rule ---
def _orb_session(breakout=True, then="up"):
    """First 7 bars form the range; later bars break out or do not."""
    opens = [100] * 20
    highs = [102] * 7 + ([106] * 13 if breakout else [101] * 13)
    # Post-breakout lows must stay above the stop, or the fixture tests the
    # stop path rather than the target path.
    lows = [98] * 7 + ([102] * 13 if breakout else [99] * 13)
    closes = [100] * 7 + ([103] * 13 if breakout else [100] * 13)
    if breakout and then == "down":
        highs = [102] * 7 + [104] + [101] * 12
        closes = [100] * 7 + [103] + [95] * 12
        lows = [98] * 7 + [99] + [94] * 12
    return _session(opens, highs, lows, closes,
                    volumes=np.full(20, 500_000.0))


def test_orb_does_not_trade_without_a_breakout():
    """The filter's whole value is refusing most days."""
    out = dt.simulate_orb(_orb_session(breakout=False), reference_daily_volume=1e6)
    assert out["traded"] is False
    assert out["outcome"] == "no_trigger"


def test_orb_enters_on_breakout_and_takes_target():
    session = _orb_session(breakout=True)
    out = dt.simulate_orb(session, reference_daily_volume=1e6,
                          target_pct=0.02, stop_pct=0.03)
    assert out["traded"] is True
    assert out["outcome"] in ("target", "close")


def test_orb_stops_out_when_the_breakout_fails():
    out = dt.simulate_orb(_orb_session(breakout=True, then="down"),
                          reference_daily_volume=1e6,
                          target_pct=0.05, stop_pct=0.03)
    assert out["traded"] is True
    assert out["outcome"] == "stop"
    assert out["return"] < 0


def test_orb_volume_filter_blocks_low_pace_breakouts():
    """A breakout on thin volume is not a burst."""
    session = _orb_session(breakout=True)
    # Reference volume so high that pace can never reach 1.5x.
    out = dt.simulate_orb(session, reference_daily_volume=1e12, min_pace=1.5)
    assert out["traded"] is False


def test_orb_handles_degenerate_input():
    assert dt.simulate_orb(pd.DataFrame(), 1e6)["traded"] is False


# ------------------------------------------------------------------- plan ---
def test_day_plan_levels_are_ordered_and_on_the_tick_grid():
    from idxbot.market import tick_size

    bars = {"TEST": _daily(last_rvol=10, last_return=0.10, price=2000.0)}
    candidate = dt.scan(bars, cfg, min_value_traded=0)[0]
    plan = dt.build_day_plan(candidate, cfg, equity=100_000_000)

    assert plan.stop < plan.entry_trigger < plan.targets[0] < plan.targets[-1]
    for price in [plan.entry_trigger, plan.stop] + plan.targets:
        assert price % tick_size(price, cfg) == 0, price


def test_day_plan_target_is_five_percent():
    bars = {"TEST": _daily(last_rvol=10, last_return=0.10, price=1000.0)}
    candidate = dt.scan(bars, cfg, min_value_traded=0)[0]
    plan = dt.build_day_plan(candidate, cfg, equity=100_000_000)
    assert plan.target_pcts[-1] == pytest.approx(0.05)


def test_day_plan_always_warns_about_expectancy():
    """The plan must never present these odds as settled."""
    bars = {"TEST": _daily(last_rvol=10, last_return=0.10)}
    candidate = dt.scan(bars, cfg, min_value_traded=0)[0]
    plan = dt.build_day_plan(candidate, cfg, equity=100_000_000)
    assert any("straddles zero" in w for w in plan.warnings)
    assert "CLOSE EVERYTHING" in plan.render()


def test_day_plan_risk_stays_within_budget():
    bars = {"TEST": _daily(last_rvol=10, last_return=0.10, price=1000.0)}
    candidate = dt.scan(bars, cfg, min_value_traded=0)[0]
    equity = 100_000_000
    plan = dt.build_day_plan(candidate, cfg, equity=equity)
    budget = equity * float(cfg.get("daytrade.risk_per_trade_pct", 0.005))
    assert plan.risk_idr <= budget + 1e-6


# --------------------------------------------------------- broker trigger ---
def test_broker_trigger_fires_when_bulge_desks_dominate():
    summary = pd.DataFrame({
        "date": pd.to_datetime(["2026-08-06"] * 3),
        "ticker": ["DSSA"] * 3,
        "broker": ["BK", "AK", "YP"],
        "buy_lot": [400, 300, 100],
        "sell_lot": [0, 0, 700],
        "buy_val": [4e9, 3e9, 1e9], "sell_val": [0, 0, 7e9],
        "buy_avg": [1000] * 3, "sell_avg": [1000] * 3,
        "source": ["test"] * 3,
    })
    out = dt.broker_trigger(summary, cfg, "DSSA")
    assert out["triggered"] is True
    assert set(out["bulge_leaders"]) <= {"BK", "AK"}


def test_broker_trigger_quiet_when_retail_dominates():
    summary = pd.DataFrame({
        "date": pd.to_datetime(["2026-08-06"] * 2),
        "ticker": ["DSSA"] * 2, "broker": ["YP", "PD"],
        "buy_lot": [700, 300], "sell_lot": [100, 100],
        "buy_val": [7e9, 3e9], "sell_val": [1e9, 1e9],
        "buy_avg": [1000] * 2, "sell_avg": [1000] * 2, "source": ["t"] * 2,
    })
    assert dt.broker_trigger(summary, cfg, "DSSA")["triggered"] is False


def test_broker_trigger_without_data():
    assert dt.broker_trigger(pd.DataFrame(), cfg, "DSSA")["triggered"] is False
