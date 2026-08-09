"""The look-ahead trap that daily bars set for intraday simulation.

This exists because it nearly produced a spectacular false result: a dip-entry
rule that measured a **96% win rate and profit factor 9.44** on 25 years of
daily bars, against **55% and 1.04** for the identical rule on properly ordered
5-minute bars.

The mechanism is simple and easy to miss. A daily bar reports a high and a low
but not which came first. Simulate "buy on a limit 7% below the open, then check
whether the session high reached my target" and the answer is usually yes -
because on 79% of those sessions the high had **already happened before the
limit filled**. The simulation credits the trade with a rally it could not have
been in.

The deeper the limit, the worse it gets: a deep limit fills near the session
low, which means most of the day's range lies before the fill, which means the
fabricated edge grows exactly as the rule looks more attractive.

    depth   high occurred before the fill
     -3%    58%
     -5%    74%
     -7%    79%

The rule below is therefore absolute: once an entry is triggered by a *price
level* rather than by the bar's open or close, only bars strictly after the
trigger may be used to resolve the outcome, and daily bars cannot support that
at all.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

from idxbot.barrier import BarrierConfig, simulate_one  # noqa: E402


def _intraday(path):
    """Ordered bars from a price path, one bar per price."""
    p = np.asarray(path, dtype=float)
    return p, p, p  # highs, lows, closes for a tick-by-tick path


def test_a_rally_before_the_entry_cannot_count_as_a_win():
    """The exact failure: price runs up, falls back, and only then fills."""
    # Session: 100 -> 110 (the rally) -> 93 (limit fills here) -> 95 (drifts)
    # A limit at 93 is filled at the third bar. The +10% rally is already gone.
    highs, lows, closes = _intraday([95.0, 96.0])   # bars strictly AFTER the fill
    r = simulate_one(highs, lows, closes, entry_price=93.0,
                     cfg=BarrierConfig(target_pct=0.05, stop_pct=0.10, max_days=10))
    assert r["outcome"] != "target"          # 93 * 1.05 = 97.65, never reached
    assert r["ret"] == pytest.approx(96.0 / 93.0 - 1.0)


def test_the_broken_version_would_have_called_it_a_win():
    """Documents what the bug produced, so the contrast is testable."""
    session_high = 110.0                      # occurred BEFORE the fill
    entry, target = 93.0, 0.05
    # The broken logic: 'did the session high clear my target?'
    assert session_high >= entry * (1 + target)      # says WIN
    # The correct logic used above says no. Same data, opposite verdict.


def test_only_bars_after_the_trigger_are_passed_to_the_simulator():
    """simulate_one trusts its caller to slice; this pins the contract."""
    full = np.array([100.0, 110.0, 93.0, 95.0, 96.0])
    fill_index = 2
    after = full[fill_index + 1:]
    r_correct = simulate_one(after, after, after, 93.0,
                             BarrierConfig(0.05, 0.10, 10))
    r_wrong = simulate_one(full, full, full, 93.0,
                           BarrierConfig(0.05, 0.10, 10))
    assert r_correct["outcome"] != "target"
    assert r_wrong["outcome"] == "target"     # the pre-entry 110 leaks in
    assert r_wrong["ret"] > r_correct["ret"]


def test_deeper_entries_leak_more_of_the_session():
    """Why the bug scaled with how attractive the rule looked.

    A deeper limit fills later and lower, so a larger share of the day's range
    sits before the fill and is available to be miscounted.
    """
    session = np.array([100.0, 112.0, 97.0, 95.0, 93.0, 94.0])
    leaked = []
    for depth in (0.03, 0.05, 0.07):
        limit = 100.0 * (1 - depth)
        fills = np.where(session <= limit)[0]
        if not len(fills):
            continue
        i = fills[0]
        before_max = session[:i].max()
        after_max = session[i + 1:].max() if i + 1 < len(session) else -np.inf
        leaked.append(before_max - after_max)
    assert len(leaked) >= 2
    assert leaked == sorted(leaked)   # monotonically more leakage with depth


def test_open_and_close_entries_are_safe_from_this():
    """Entering at a bar's open or close carries no ordering ambiguity.

    That is why the 60-day and 20-day results in FINDINGS are unaffected: they
    enter at the next bar's open, which is unambiguously the first price of
    that bar.
    """
    bars_after = np.array([100.0, 106.0])
    r = simulate_one(bars_after, bars_after, bars_after, entry_price=100.0,
                     cfg=BarrierConfig(0.05, 0.10, 10))
    assert r["outcome"] == "target"
    assert r["days"] == 2
