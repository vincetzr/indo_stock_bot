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
