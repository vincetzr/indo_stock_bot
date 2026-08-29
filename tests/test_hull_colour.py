"""Tests for the board-wide pure Hull-55 colour rule.

This script exists because I asserted the Hull suite loses to buy-and-hold and,
when challenged, found I had measured EMA34 breaks and labelled them the ribbon.
So the tests that matter here are the ones pinning that what is measured IS the
ribbon the chart draws — the two-bar slope of HMA(55), not any other average —
and that the benchmark is owning the name over its whole span rather than a
duration-matched hold, which is degenerate (H39 printed "beat buy-and-hold on
0.0% of trades" from exactly that mistake).
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir,
                                "scripts"))

from hull_colour import FEE, colour_campaign                     # noqa: E402
from selloff import hma                                          # noqa: E402


def _green(p, n=55):
    h = hma(p, n)
    g = np.zeros(len(p), bool)
    g[2:] = h[2:] > h[:-2]
    return g


# ============================================ what the rule actually reads ===
def test_the_colour_is_the_two_bar_slope_of_the_hull_and_nothing_else():
    """The chart paints green when HMA(55) is above its value two bars back.
    Any other definition is a different indicator wearing its name."""
    p = np.linspace(100.0, 300.0, 400)
    g = _green(p)
    assert g[300:].all()
    p2 = np.r_[np.linspace(100, 300, 300), np.linspace(300, 100, 300)]
    assert not _green(p2)[-50:].any()


def test_a_hull_of_a_different_length_is_not_the_ribbon():
    """Guard against the exact substitution that produced the wrong claim: a
    faster average flips at different bars, so it is a different rule."""
    rng = np.random.default_rng(11)
    p = np.cumprod(1 + rng.normal(0.0003, 0.02, 800)) * 1000
    assert (_green(p, 55) != _green(p, 21)).sum() > 20


# ==================================================== the campaign accounts ==
def test_a_permanently_green_name_is_one_trade_and_ties_hold_minus_one_toll():
    """The anchor. If the ribbon never turns red the rule IS buy-and-hold, and
    the only difference must be a single round trip of cost."""
    p = np.linspace(100.0, 400.0, 400)
    g = np.ones(len(p), bool)
    g[0] = False                # the rule needs a rising edge to buy on
    lg, hl, ntr, inb = colour_campaign(p, g, 0.01)
    assert ntr == 1
    assert lg == pytest.approx(np.log(p[-1] / p[1] - 0.01))
    assert hl == pytest.approx(np.log(p[-1] / p[0]))
    assert lg < hl


def test_a_permanently_red_name_never_trades_and_the_hold_is_still_measured():
    """Being flat all span must show as zero rule return against whatever the
    name did — that asymmetry is the point of the comparison."""
    p = np.linspace(400.0, 100.0, 400)
    lg, hl, ntr, inb = colour_campaign(p, np.zeros(len(p), bool), 0.01)
    assert (ntr, inb, lg) == (0, 0, 0.0)
    assert hl < 0


def test_the_cost_is_charged_once_per_round_trip_not_once_per_bar():
    p = np.linspace(100.0, 400.0, 400)
    g = np.ones(len(p), bool)
    g[0] = False
    a, _, _, _ = colour_campaign(p, g, 0.0)
    b, _, _, _ = colour_campaign(p, g, 0.02)
    assert a - b == pytest.approx(np.log(p[-1] / p[1])
                                  - np.log(p[-1] / p[1] - 0.02))


def test_two_green_stretches_are_two_round_trips_and_two_tolls():
    p = np.full(100, 100.0)
    g = np.zeros(100, bool)
    g[10:30] = True
    g[50:70] = True
    _, _, ntr, inb = colour_campaign(p, g, 0.01)
    assert ntr == 2
    #  The exit bar is the FIRST red bar — the ribbon changes colour at 30 and
    #  you sell at that close — so each stretch holds 20 bars, not 19.
    assert inb == (30 - 10) + (70 - 50)


def test_a_collapsing_name_cannot_produce_a_nan_or_an_infinite_log():
    """H40's NaN: `ratio - cost` can go non-positive, and log of that is -inf.
    The clip keeps a total loss finite and very negative rather than unusable."""
    p = np.r_[np.full(50, 1000.0), np.full(50, 0.001)]
    g = np.ones(100, bool)
    g[0] = False
    lg, hl, _, _ = colour_campaign(p, g, 0.0056)
    assert np.isfinite(lg) and np.isfinite(hl) and lg < 0


def test_the_benchmark_spans_the_whole_series_not_the_trade():
    """A duration-matched hold is degenerate here — it differs from the trade
    only by the toll, so the rule can never win. The hold leg must be measured
    from the FIRST bar to the LAST, whatever the rule did in between."""
    p = np.r_[np.linspace(100, 200, 100), np.linspace(200, 150, 100)]
    g = np.zeros(200, bool)
    g[10:90] = True
    _, hl, _, _ = colour_campaign(p, g, 0.0)
    assert hl == pytest.approx(np.log(p[-1] / p[0]))


def test_being_out_of_a_rising_name_costs_return_which_is_the_whole_mechanism():
    """The board result is that the rule is in the market ~44% of the time and
    loses to holding. This pins the mechanism: on a name that only rises,
    every bar spent flat is return forgone."""
    p = np.linspace(100.0, 400.0, 400)
    g = np.zeros(400, bool)
    g[100:200] = True
    lg, hl, _, _ = colour_campaign(p, g, 0.0)
    assert lg < hl


def test_the_standing_fee_is_the_users_actual_schedule():
    """0.28% buy + 0.18% sell + 0.1% sell tax. A5 fixes this and it overrides
    the brief's 0.15-0.30% range; the half-spread is added per name on top."""
    assert FEE == pytest.approx(0.0056)
