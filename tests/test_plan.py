"""How a plan describes the provenance of its own numbers.

This is a correctness concern, not a cosmetic one. "No broker data" and
"fabricated broker data" are different claims, and a plan that confuses them
either tells the user real levels are invented or - far worse in the other
direction - lets synthetic levels pass as real.
"""

import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

from idxbot.plan import TradingPlan  # noqa: E402


def _plan(**kwargs) -> TradingPlan:
    base = dict(ticker="GGRM", as_of=pd.Timestamp("2026-08-07"), close=21725.0,
                score=100.0, level="STRONG", verdict="TAKE")
    base.update(kwargs)
    return TradingPlan(**base)


def test_price_only_is_not_described_as_simulated():
    """The bug: --providers none produced 'BROKER FLOW IS SIMULATED'.

    Price-only mode feeds the engine genuine exchange OHLCV and simply omits
    the flow components. Nothing about it is simulated.
    """
    text = _plan(data_mode="price-only", data_source="none",
                 data_is_real=False).render()
    assert "SIMULATED" not in text
    assert "PRICE-ONLY" in text
    assert "Prices and levels are real" in text


def test_synthetic_broker_data_is_still_flagged_as_simulated():
    text = _plan(data_mode="broker+price", data_source="synthetic",
                 data_is_real=False).render()
    assert "SIMULATED" in text
    assert "synthetic" in text


def test_real_broker_data_carries_no_provenance_warning():
    text = _plan(data_mode="broker+price", data_source="goapi",
                 data_is_real=True).render()
    assert "SIMULATED" not in text
    assert "PRICE-ONLY" not in text


@pytest.mark.parametrize("mode,source,is_real,expected", [
    ("broker+price", "goapi", True, "TAKE"),
    ("price-only", "none", False, "TAKE (PRICE-ONLY)"),
    ("broker+price", "synthetic", False, "TAKE (SIMULATED DATA)"),
])
def test_verdict_distinguishes_all_three_data_modes(mode, source, is_real, expected):
    plan = _plan(data_mode=mode, data_source=source, data_is_real=is_real,
                 verdict=expected)
    assert plan.verdict == expected
    # and the rendered header carries it verbatim
    assert expected in plan.render()


# --------------------------------------------------------------------------
# the exit rule quotes the configuration it is actually running
# --------------------------------------------------------------------------

def _cfg(**overrides):
    from idxbot.config import Config
    base = {"plan": {"exit": {
        "target_pct": 0.02, "scale_out": 0.25, "stop_pct": 0.15,
        "breakeven_after_target": True, "max_days": 20,
        "measured": {"win_rate": 0.85, "profit_factor": 1.82,
                     "trades": 1050, "avg_days": 9.8},
    }}, "market": {"lot_size": 100, "tick_bands": [[200, 1], [500, 2],
                                                   [2000, 5], [5000, 10],
                                                   [None, 25]]}}
    base["plan"]["exit"].update(overrides)
    return Config(base)


def test_exit_basis_quotes_the_configured_cap_not_a_remembered_one():
    """The 20d and 60d caps are different strategies with different numbers.

    Hard-coding either means the plan eventually advertises one variant's
    result while instructing the reader to trade the other.
    """
    from idxbot.plan import _exit_basis
    text = _exit_basis(_cfg())
    assert "20-day cap" in text
    assert "85%" in text and "1.82" in text

    other = _exit_basis(_cfg(max_days=60,
                             measured={"win_rate": 0.86, "profit_factor": 2.23,
                                       "trades": 360, "avg_days": 17.9}))
    assert "60-day cap" in other
    assert "86%" in other and "2.23" in other


def test_exit_basis_says_so_when_nothing_has_been_measured():
    from idxbot.plan import _exit_basis
    assert "not yet measured" in _exit_basis(_cfg(measured={}))


def test_exit_rule_steps_do_not_hardcode_a_win_rate():
    """Any percentage baked into the prose is a number that can go stale."""
    from idxbot.plan import _exit_rule
    steps = " ".join(_exit_rule(_cfg(), entry=2540.0, atr_stop=2410.0))
    assert "88%" not in steps
    assert "85%" not in steps
    assert "15%" in steps and "25%" in steps   # the actual configured levels


def test_exit_rule_reflects_configured_levels():
    from idxbot.plan import _exit_rule
    steps = _exit_rule(_cfg(target_pct=0.05, scale_out=0.5, stop_pct=0.10,
                            max_days=40), entry=1000.0, atr_stop=950.0)
    joined = " ".join(steps)
    assert "50%" in joined            # scale-out share
    assert "10% below entry" in joined
    assert "40 trading days" in joined


def test_exit_rule_is_empty_without_configuration():
    from idxbot.config import Config
    from idxbot.plan import _exit_rule
    assert _exit_rule(Config({}), entry=1000.0, atr_stop=950.0) == []
