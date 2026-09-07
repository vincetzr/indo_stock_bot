"""H61 — §14 chart patterns and the base-rate harness.

THE TEST THIS FILE EXISTS FOR is `test_every_detector_is_causal`. It truncates
the panel to a past date, runs each detector on what was knowable then, and
demands the same firings appear. A32 records why: one non-causal helper
anywhere in the chain — a ZigZag drawn at the pivot instead of the confirmation
bar, a centred window — turns the whole study into a look-ahead and NOTHING in
the output looks wrong.

The second is `test_a_shifted_bool_series_is_not_negated_with_tilde`. `~` on an
object-dtype Series applies bitwise NOT to the underlying Python bools, and
`~False == -1` is truthy, so the negation silently does nothing. Measured, that
made the golden cross fire 357,925 times on 690,591 eligible bars — 52% of the
panel, for a signal that should fire a dozen times per name in twenty-six
years. The number was impossible, which is the only reason it was caught.

The third is `test_the_net_column_is_the_edge_not_the_raw_return`. A first
version printed the pattern's own return net of the fee, reading "+6.30%" for a
pattern whose matched control returned MORE — A19's error class, and the number
a reader would quote.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "scripts"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

import patterns as pt                                             # noqa: E402
from patterns import (PATTERNS, Pool, _b, fire_table,             # noqa: E402
                      matched_control, p_digit, p_double_bottom,
                      p_golden, p_gap_volume, p_hh_hl)

SRC = os.path.join(os.path.dirname(__file__), os.pardir, "scripts",
                   "patterns.py")


def _one(n=900, seed=0, trend=0.0005):
    rng = np.random.default_rng(seed)
    d = pd.bdate_range("2015-01-05", periods=n)
    px = np.round(np.exp(np.cumsum(rng.normal(trend, 0.02, n))) * 1000)
    return pd.DataFrame({"date": d, "ticker": "AAAA", "close": px,
                         "adj_close": px, "elig": True,
                         "volume": rng.integers(1e5, 1e7, n).astype(float)})


def _panel(k=8, n=900):
    return pd.concat([_one(n=n, seed=i).assign(ticker=f"T{i:03d}")
                      for i in range(k)], ignore_index=True)


# ------------------------------------------------------------- CAUSALITY

@pytest.mark.parametrize("label,fn", list(PATTERNS))
def test_every_detector_is_causal(label, fn):
    """Truncating the future must not change any past firing."""
    g = _one(n=900, seed=3)
    full = fn(g)
    for cut in (500, 650, 800):
        part = fn(g.iloc[:cut].copy())
        assert len(part) == cut
        assert np.array_equal(part, full[:cut]), (
            f"{label} changed its verdict on bars it had already seen once "
            f"the future arrived — it is reading ahead")


def test_a_detector_that_peeked_would_be_caught():
    """POSITIVE CONTROL for the causality test: a deliberately non-causal
    detector must fail it, or passing means nothing."""
    def peeker(d):
        c = d["adj_close"]
        return _b(c.shift(-5) > c)          # tomorrow's price, on purpose
    g = _one(n=400, seed=4)
    full = peeker(g)
    part = peeker(g.iloc[:300].copy())
    assert not np.array_equal(part, full[:300])


# --------------------------------------------------- THE OBJECT-DTYPE TRAP

def test_a_shifted_bool_series_is_not_negated_with_tilde():
    """`~` on object dtype inverts the integer: ~False == -1, which is truthy."""
    up = pd.Series([False, False, True, True])
    naive = ~up.shift(1).fillna(False)
    assert list(naive) == [-1, -1, -1, -2]
    assert all(bool(x) for x in naive), "the naive negation is True everywhere"
    assert list(~pd.Series(_b(up.shift(1)))) == [True, True, True, False]


def test_the_golden_cross_fires_at_a_believable_rate():
    """A dozen crosses in twenty-odd years, not half the panel."""
    g = _one(n=2000, seed=5)
    hits = int(p_golden(g).sum())
    assert 0 <= hits <= 40, hits
    assert hits / len(g) < 0.05


def test_b_returns_a_real_numpy_bool_array():
    got = _b(pd.Series([True, np.nan, False], dtype=object))
    assert got.dtype == bool
    assert list(got) == [True, False, False]


# ---------------------------------------------------------- THE DETECTORS

def test_the_double_bottom_fires_on_a_planted_one():
    """POSITIVE CONTROL. A detector that cannot find the shape it is named
    after proves nothing by not finding it — A26's sine wave."""
    w, h = 40, 20      # each phase is half a window, so both lows land inside
    px = np.r_[np.full(h, 100.0), np.full(h, 80.0), np.full(h, 100.0),
               np.full(h, 80.5), np.full(10, 120.0)]
    d = pd.DataFrame({"date": pd.bdate_range("2020-01-01", periods=len(px)),
                      "adj_close": px, "close": px})
    assert p_double_bottom(d, w=w, tol=0.05).any()


def test_the_double_bottom_does_not_fire_when_the_lows_differ():
    w, h = 40, 20
    px = np.r_[np.full(h, 100.0), np.full(h, 80.0), np.full(h, 100.0),
               np.full(h, 40.0), np.full(10, 120.0)]
    d = pd.DataFrame({"date": pd.bdate_range("2020-01-01", periods=len(px)),
                      "adj_close": px, "close": px})
    assert not p_double_bottom(d, w=w, tol=0.05).any()


def test_the_double_bottom_fires_on_the_confirmation_not_the_low():
    """Marking it at the second low would be drawing it with knowledge that no
    lower low followed — A27's Pine ZigZag defect one level up."""
    w, h = 40, 20
    px = np.r_[np.full(h, 100.0), np.full(h, 80.0), np.full(h, 100.0),
               np.full(h, 80.0), np.full(5, 130.0)]
    d = pd.DataFrame({"date": pd.bdate_range("2020-01-01", periods=len(px)),
                      "adj_close": px, "close": px})
    hit = p_double_bottom(d, w=w, tol=0.05)
    assert hit.any()
    #  the breakout starts at 4*h; nothing may fire before it
    assert int(np.flatnonzero(hit)[0]) >= 4 * h, "fired before the breakout"


def test_the_digit_pattern_reads_the_traded_close_not_the_adjusted_one():
    """The fraksi-harga grid is a property of the PRINTED price; a back-adjusted
    close is off the grid by construction (A2)."""
    d = pd.DataFrame({"date": pd.bdate_range("2020-01-01", periods=4),
                      "close": [100.0, 101.0, 105.0, 107.0],
                      "adj_close": [1.0, 2.0, 3.0, 4.0]})
    assert list(p_digit(d)) == [True, False, True, False]


def test_gap_up_needs_both_the_gap_and_the_volume():
    n = 200
    rng = np.random.default_rng(1)
    px = np.full(n, 1000.0)
    px[150] = 1100.0                       # a 10% gap
    vol = np.full(n, 1e6)
    d = pd.DataFrame({"date": pd.bdate_range("2020-01-01", periods=n),
                      "adj_close": px, "close": px, "volume": vol})
    assert not p_gap_volume(d).any(), "fired on a gap with flat volume"
    d.loc[150, "volume"] = 1e9
    assert p_gap_volume(d).any()


# ------------------------------------------------------------- THE CONTROL

def test_the_control_matches_the_ticker_year_cells_and_the_counts():
    """A34: a control drawn from the whole panel would let a pattern that fires
    on good names in good years beat it without any information."""
    P = _panel(k=6, n=600)
    fires = fire_table(P, p_hh_hl, 63)
    assert len(fires) > 50
    pool = Pool(P, 63)
    ctl = matched_control(pool, fires, seed=0)
    a = fires.assign(y=fires["date"].dt.year).groupby(["ticker", "y"]).size()
    b = ctl.assign(y=ctl["date"].dt.year).groupby(["ticker", "y"]).size()
    common = a.index.intersection(b.index)
    assert len(common) > 0
    assert (b[common] <= a[common]).all(), "the control drew more than matched"
    assert set(ctl["ticker"]) <= set(fires["ticker"])


def test_the_control_is_reproducible_from_its_seed():
    P = _panel(k=4, n=500)
    fires = fire_table(P, p_hh_hl, 63)
    pool = Pool(P, 63)
    a = matched_control(pool, fires, seed=7)
    b = matched_control(pool, fires, seed=7)
    assert a.equals(b)
    c = matched_control(pool, fires, seed=8)
    assert not a.equals(c)


def test_the_pool_holds_only_eligible_bars_with_a_finite_forward():
    P = _panel(k=2, n=400)
    P.loc[P.index[:100], "elig"] = False
    pool = Pool(P, 63)
    total = sum(len(f) for _d, f in pool.cells.values())
    assert total < int(P["elig"].sum())
    assert all(np.isfinite(f).all() for _d, f in pool.cells.values())


def test_fire_table_returns_the_forward_from_the_confirmation_bar():
    d = _one(n=400, seed=9)
    P = d.copy()
    fires = fire_table(P, p_hh_hl, 63)
    if fires.empty:
        pytest.skip("no firings on this synthetic")
    c = d.set_index("date")["adj_close"]
    row = fires.iloc[0]
    i = list(c.index).index(row["date"])
    assert row["fwd"] == pytest.approx(c.iloc[i + 63] / c.iloc[i] - 1.0)


# ------------------------------------------------------------ THE REPORTING

def test_the_net_column_is_the_edge_not_the_raw_return():
    src = open(SRC).read()
    assert 'net = raw - float(ctl["fwd"].mean()) - FEE' in src, (
        "the gated column must subtract the CONTROL, not just the fee — a "
        "number with no benchmark in it is the one a reader quotes")
    assert "net edge" in src and "'raw'" in src


def test_a_predicted_null_is_registered_and_present():
    assert any("PREDICTED NULL" in lbl for lbl, _fn in PATTERNS)
    assert "PREDICTED NULL" in (pt.__doc__ or "")


def test_the_minimum_firing_floor_exists():
    """A19 records the smallest cell producing the largest effect three times."""
    assert pt.MIN_FIRES >= 300


def test_the_registration_survives_the_answer():
    doc = pt.__doc__ or ""
    for k in ("P1", "P2", "P3"):
        assert k in doc
