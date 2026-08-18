"""Structural checks on the Pine scripts.

Pine cannot be executed here, so these tests cover the failure modes that are
checkable from the text and that actually bite when a script is pasted into
TradingView:

  * a missing version tag silently compiles as v1 and behaves differently;
  * unbalanced delimiters are the usual result of an edit made by hand;
  * ``barmerge.lookahead_on`` is the single most common way a Pine script reads
    its own future - it must never appear in anything this repository ships;
  * the numbers quoted in a script's header are claims, so the ones that matter
    are pinned here against the findings they came from.
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

from idxbot.tradingview import list_pine, read_pine   # noqa: E402

SCRIPTS = list_pine()


def test_scripts_are_discovered():
    assert {"turn_reality", "bluechip_regime"} <= set(SCRIPTS)


@pytest.mark.parametrize("name", SCRIPTS)
def test_declares_a_pine_version(name):
    src = read_pine(name)
    assert "//@version=" in src, f"{name} has no version tag"


@pytest.mark.parametrize("name", SCRIPTS)
def test_delimiters_balance(name):
    src = read_pine(name)
    for open_ch, close_ch in (("(", ")"), ("[", "]")):
        assert src.count(open_ch) == src.count(close_ch), \
            f"{name}: unbalanced {open_ch}{close_ch}"


@pytest.mark.parametrize("name", SCRIPTS)
def test_never_looks_ahead(name):
    """The one Pine setting that lets a script read data it could not have had."""
    src = read_pine(name)
    assert "lookahead_on" not in src, f"{name} uses barmerge.lookahead_on"


@pytest.mark.parametrize("name", SCRIPTS)
def test_declares_indicator_or_strategy(name):
    src = read_pine(name)
    assert ("indicator(" in src) or ("strategy(" in src)


def test_regime_script_uses_a_closed_weekly_bar():
    """Reading the CURRENT weekly close on a daily bar is look-ahead by another
    name: the week has not finished. The [1] is what makes it honest."""
    src = read_pine("bluechip_regime")
    assert "close[1]" in src
    assert 'request.security(idxSym, "W"' in src
    assert src.count("lookahead=barmerge.lookahead_off") >= 2


def test_regime_script_defaults_match_the_finding():
    """25% out-exposure and a 30-week average are the measured settings.

    If someone changes the defaults, the header prose stops being true, so the
    two are pinned together here.
    """
    src = read_pine("bluechip_regime")
    assert 'input.int(30, "Regime average (weeks)"' in src
    assert 'input.float(25.0, "Exposure when OUT (%)"' in src


def test_turn_script_admits_the_zigzag_repaints():
    """The hindsight layer must be labelled as such or the script is misleading."""
    src = read_pine("turn_reality")
    assert "REPAINT" in src.upper()
    assert "hindsight" in src.lower()


def test_turn_script_tracks_high_and_low_separately():
    """The bug that produced two pivots for a saw-tooth must not be reintroduced."""
    src = read_pine("turn_reality")
    assert "zzHi" in src and "zzLo" in src
    assert "zzHiBar" in src and "zzLoBar" in src
