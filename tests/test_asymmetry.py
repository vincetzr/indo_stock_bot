"""Tests for the asymmetry search (H26).

The objective is a RATIO, and a ratio has two ways to be wrong that a single
rate does not: a tiny denominator makes it explode, and a zero denominator
makes it undefined rather than infinite. Both are pinned.
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

from asymmetry import (EVIDENCE, FRONTIER, MIN_CELL, MIN_LEG,   # noqa: E402
                       SCREEN, cell, live)

K = 100


def _d(n, peak, end, start="2010-01-01"):
    return pd.DataFrame({
        "date": pd.date_range(start, periods=n, freq="MS"),
        "ticker": [f"T{i}" for i in range(n)],
        f"peak{K}": np.asarray(peak, float),
        f"end{K}": np.asarray(end, float)})


# ==================================================================== the ratio
def test_skew_is_doubles_over_halvings():
    d = _d(400, [2.5] * 100 + [1.0] * 300, [1.0] * 350 + [0.4] * 50)
    c = cell(d, K, None, "x")
    assert c["up"] == pytest.approx(0.25)
    assert c["dn"] == pytest.approx(0.125)
    assert c["skew"] == pytest.approx(2.0)


def test_an_empty_losing_leg_gives_an_UNDEFINED_ratio_not_an_infinite_one():
    """THE DEFECT A RATIO OBJECTIVE INVITES. A cell with no halvings would
    rank first with a skew of infinity on the strength of having too few
    observations to show one."""
    d = _d(400, [2.5] * 400, [1.5] * 400)        # nothing ever halves
    c = cell(d, K, None, "x")
    assert np.isnan(c["skew"])
    assert c["n_dn"] == 0


def test_a_losing_leg_below_the_floor_is_also_refused():
    n_dn = MIN_LEG - 1
    d = _d(400, [2.5] * 400, [0.4] * n_dn + [1.5] * (400 - n_dn))
    assert np.isnan(cell(d, K, None, "x")["skew"])


def test_a_losing_leg_at_the_floor_is_quoted():
    d = _d(400, [2.5] * 400, [0.4] * MIN_LEG + [1.5] * (400 - MIN_LEG))
    assert np.isfinite(cell(d, K, None, "x")["skew"])


def test_a_cell_below_the_observation_floor_is_refused_entirely():
    d = _d(MIN_CELL - 1, [2.5] * (MIN_CELL - 1), [0.4] * (MIN_CELL - 1))
    assert cell(d, K, None, "x") == {}


def test_the_peak_decides_doubling_and_the_END_decides_halving():
    """A name that trebled and then collapsed both doubled AND halved; the two
    legs are measured on different columns and must not be conflated."""
    d = _d(400, [3.0] * 400, [0.3] * 400)
    c = cell(d, K, None, "x")
    assert c["up"] == pytest.approx(1.0)
    assert c["dn"] == pytest.approx(1.0)


# ================================================================ the frontier
#: The strength family — the cells that share a construction and therefore
#: form a frontier. The unscreened baseline is a REFERENCE POINT, not a point
#: on the curve, and H25's screen has no strength filter at all, so neither
#: belongs in a monotonicity claim. A first draft of the memo called the whole
#: table "perfectly monotone" and these two tests are what caught it.
STRENGTH_FAMILY = ("STRENGTH + CALM", "strength only", "strength, very strong",
                   "strength + some vol")


def _family():
    return [r for r in FRONTIER if r[0].startswith(STRENGTH_FAMILY)]


def test_the_frontier_trades_doubling_rate_against_asymmetry():
    """THE ANSWER TO THE QUESTION, stated only where it holds. Within one
    construction, more doubling means less asymmetry."""
    rows = sorted(_family(), key=lambda r: r[1])
    skews = [r[3] for r in rows]
    assert len(rows) >= 4
    assert skews == sorted(skews, reverse=True), (
        "within the strength family a cell with both a high doubling rate "
        "and high skew would break the trade-off the memo reports")


def test_compounding_falls_with_the_doubling_rate_within_the_family():
    rows = sorted(_family(), key=lambda r: r[1])
    logs = [r[4] for r in rows]
    assert logs == sorted(logs, reverse=True)


def test_adding_strength_improves_the_ratio_at_the_SAME_doubling_rate():
    """The claim that actually matters, and the one the baseline obscured.
    H25's screen and `strength + some vol` both double ~21% of the time; the
    one with the strength filter has a far better ratio and loses far less."""
    h25 = [r for r in FRONTIER if "H25" in r[0]][0]
    both = [r for r in FRONTIER if "strength + some vol" in r[0]][0]
    assert abs(h25[1] - both[1]) < 0.01, "same doubling rate"
    assert both[3] > h25[3] + 0.3, "strictly better asymmetry"
    assert both[2] < h25[2], "and it halves less often"
    assert both[4] > h25[4], "and it compounds better"


def test_the_h25_screen_is_recorded_as_the_worst_cell_on_asymmetry():
    h25 = [r for r in FRONTIER if "H25" in r[0]][0]
    assert h25[3] == min(r[3] for r in FRONTIER)
    assert h25[4] == min(r[4] for r in FRONTIER)


def test_the_winner_beats_the_no_screen_baseline_on_both_axes():
    base = [r for r in FRONTIER if "everything" in r[0]][0]
    win = [r for r in FRONTIER if "STRENGTH + CALM" in r[0]][0]
    assert win[3] > base[3] and win[4] > base[4]
    assert win[2] < base[2], "it must also halve LESS often"


# ================================================================ the evidence
def test_the_winner_clears_the_bar_and_the_constants_say_so():
    assert EVIDENCE["p"] < EVIDENCE["bar"]
    assert EVIDENCE["z"] > 5


def test_the_winner_replicates_in_both_halves():
    assert min(EVIDENCE["early_skew"], EVIDENCE["late_skew"]) > 2.0


def test_the_downside_is_the_headline_not_the_upside():
    assert EVIDENCE["dn"] < 0.05
    assert EVIDENCE["skew"] == pytest.approx(EVIDENCE["up"] / EVIDENCE["dn"],
                                             rel=0.05)


def test_even_the_tenth_percentile_draw_matches_the_index():
    assert EVIDENCE["cagr_p10"] >= EVIDENCE["index_cagr"] - 0.005


# ==================================================================== the live
def _P(n=200, day="2026-08-24"):
    d = pd.Timestamp(day)
    return pd.DataFrame({
        "date": d, "ticker": [f"T{i:03d}" for i in range(n)],
        "close": np.linspace(100.0, 5000.0, n), "tradeable": True,
        "log_turnover": np.full(n, 24.0),
        "hi52": np.linspace(0.30, 1.00, n),
        "vol60": np.linspace(0.60, 0.01, n)})


def test_the_live_screen_takes_strong_AND_calm():
    P = _P()
    S = live(P, P["date"].max())
    assert (S["hi52"] >= P["hi52"].quantile(SCREEN["hi52_pct"])).all()
    assert (S["vol60"] <= P["vol60"].quantile(SCREEN["vol_pct"])).all()


def test_the_live_screen_excludes_a_strong_but_wild_name():
    P = _P()
    P.loc[P["ticker"] == "T199", "vol60"] = 5.0      # strongest, but wildest
    assert "T199" not in set(live(P, P["date"].max())["ticker"])


def test_the_live_screen_declines_on_a_thin_cross_section():
    assert live(_P(n=20), _P(n=20)["date"].max()).empty
