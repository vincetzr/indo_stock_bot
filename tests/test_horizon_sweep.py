"""Tests for the horizon sweep (H23).

The first version of this study reported a 73.8% doubling rate that was two
bugs stacked — a universe accidentally restricted to sub-Rp50 names, and a
survivorship filter that kept only names trading every bar of a 7.5-year
window. Both are pinned here, because both printed a plausible table.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir,
                                "scripts"))

from horizon_sweep import classify, forward, label, summarise  # noqa: E402


def _g(prices, ticker="X", start="2020-01-01"):
    return pd.DataFrame({
        "date": pd.date_range(start, periods=len(prices), freq="B"),
        "ticker": [ticker] * len(prices),
        "adj_close": np.asarray(prices, float)})


# ================================================== the forward-path mechanics
def test_forward_peak_is_the_maximum_ahead_not_behind():
    """A peak that already happened is not a peak you can sell into."""
    F = forward(_g([10., 12., 25., 8.]), 3)
    assert F["peak"].iloc[0] == pytest.approx(2.5)     # sees the 25
    assert F["peak"].iloc[3] == pytest.approx(1.0)     # nothing ahead


def test_forward_end_is_the_terminal_not_the_best():
    F = forward(_g([10., 12., 25., 8.]), 3)
    assert F["end"].iloc[0] == pytest.approx(0.8)      # ends at 8, not 25


def test_the_touch_and_end_definitions_can_disagree_completely():
    """THE WHOLE POINT OF THE STUDY. A path that trebles then collapses has
    touched 2x and ended below water, and only a take-profit captures it."""
    F = forward(_g([10., 30., 5.]), 2)
    assert F["peak"].iloc[0] >= 2.0
    assert F["end"].iloc[0] < 1.0


def test_bars_counts_what_is_actually_available():
    """The truncation counter is what separates a death from a censored
    window, so it has to be exact and not approximately right."""
    F = forward(_g([1.0] * 5), 3)
    assert list(F["bars"]) == [3, 3, 2, 1, 0]
    F2 = forward(_g([1.0] * 10), 3)
    assert list(F2["bars"]) == [3] * 7 + [2, 1, 0]


# ============================================== survivorship, the real defect
def _D(cls_bars, k=100, last_gap_days=2000):
    """A frame shaped like the real one, with controllable window outcomes."""
    n = len(cls_bars)
    end = pd.Timestamp("2024-01-01")
    return pd.DataFrame({
        "date": pd.date_range("2010-01-01", periods=n, freq="MS"),
        "ticker": [f"T{i}" for i in range(n)],
        f"bars{k}": cls_bars,
        f"peak{k}": np.full(n, 2.5),
        f"end{k}": np.full(n, 1.5),
        "last_bar": [end - pd.Timedelta(days=last_gap_days)] * n,
        "panel_end": [end] * n})


def test_a_name_that_died_is_kept_not_dropped():
    """THE DEFECT THIS FILE EXISTS FOR. Requiring a full window keeps only
    names that lived, which at 7.5 years discarded 91% of cohorts and measured
    the doubling rate of the survivors."""
    D = _D([100, 100, 40, 30])
    c = classify(D, 100)
    assert (c["cls"] == "full").sum() == 2
    assert (c["cls"] == "died").sum() == 2
    assert (c["cls"] == "censored").sum() == 0


def test_a_window_running_past_the_data_is_censored_and_must_be_dropped():
    """A short window on a name still trading at the panel end has no outcome
    yet. Treating it as a death would invent one."""
    D = _D([100, 40], last_gap_days=5)          # still alive at panel end
    c = classify(D, 100)
    assert (c["cls"] == "censored").sum() == 1
    assert (c["cls"] == "died").sum() == 0


def test_deaths_at_zero_are_a_total_loss_but_the_peak_still_counts():
    """A name that trebled and then died DID touch 2x — a take-profit order
    would have filled — so writing the terminal to zero must not erase it."""
    big = _D([100] * 40 + [40] * 40)
    a = summarise(big, 100, deaths="last")
    b = summarise(big, 100, deaths="zero")
    assert a["touch2x"] == pytest.approx(1.0)
    assert a["touch2x"] == b["touch2x"] == pytest.approx(1.0)
    assert b["half"] > a["half"]                # zeroed deaths are -50%+ losses
    assert b["median"] <= a["median"]


def test_dropping_deaths_reproduces_the_survivors_only_figure():
    big = _D([100] * 40 + [40] * 40)
    s = summarise(big, 100, deaths="drop", min_n=10)
    assert s["n"] == 40
    assert s["died"] == 0


def test_effective_n_shrinks_with_the_horizon():
    """Overlapping monthly cohorts over a long window produce rows, not
    information, and the report has to say which it is quoting."""
    big = _D([100] * 200, k=100)
    short = summarise(big, 100)
    assert short["eff_n"] < short["n"]


# ============================================================ small mechanics
def test_label_reads_in_years():
    assert label(252) == "1y"
    assert label(2520) == "10y"
    assert label(1890) == "7.5y"


def test_summarise_declines_on_too_few_windows():
    assert summarise(_D([100] * 10), 100) == {}
