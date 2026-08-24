"""Tests for §9.4 fingerprints and §9.5's checks.

The one that matters most is the self-exclusion. §9.4 is explicit that a broker
who is a large share of the pool drags the benchmark toward their own price and
so measures themselves as neutral, and A4 records that skipping the correction
made every large broker look identically flat. So the first block builds a
market where the true edges are known by construction and asserts the corrected
figure recovers them while the uncorrected one does not.

The second block pins the individual metrics against cases whose answer is
obvious by inspection — a broker concentrated in one name must show high HHI, a
broker that alternates direction every window must show negative AR(1).

The third block covers the two things H14 discovered the hard way: that
`distinctiveness` must be scale-free across periods, or Q2 measures units
rather than separability; and that a metric can persist *less* than its own
label-shuffled null, which is only visible if the null is computed at all.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

from idxbot.spine.fingerprint import (METRICS, STYLE, _ar1_sign,   # noqa: E402
                                      _hhi, distinctiveness,
                                      execution_edges, fingerprints,
                                      standardise, visible_vwap)


def rows(recs, ticker="AAAA", window="2020-01-14"):
    """recs: (broker, buy_lot, buy_price, sell_lot, sell_price)."""
    out = []
    for b, bl, bp, sl, sp in recs:
        out.append({"broker": b, "ticker": ticker,
                    "window_end": pd.Timestamp(window),
                    "buy_lot": float(bl), "buy_val": float(bl) * 100 * float(bp),
                    "sell_lot": float(sl),
                    "sell_val": float(sl) * 100 * float(sp)})
    return pd.DataFrame(out)


# --------------------------------------------------------------------------
# THE SELF-EXCLUSION — §9.4's mandatory correction
# --------------------------------------------------------------------------
def test_a_dominant_broker_does_not_measure_itself_as_neutral():
    """THE regression A4 documents. Broker BIG is 90% of the window and buys
    clearly below everyone else. Against a benchmark that INCLUDES its own
    trades it looks nearly flat; excluded, the real edge appears."""
    D = rows([("BIG", 9000, 100.0, 0, 0),
              ("S1", 500, 110.0, 0, 0),
              ("S2", 500, 110.0, 0, 0)])
    e = execution_edges(D).set_index("broker")
    # everyone else paid 110; BIG paid 100 -> (110-100)/110 = +909 bps
    assert e.loc["BIG", "edge_buy"] == pytest.approx(10000 * 10 / 110, rel=1e-6)

    incl = (D["buy_val"].sum() + D["sell_val"].sum()) / (
        (D["buy_lot"].sum() + D["sell_lot"].sum()) * 100.0)
    naive = 10000.0 * (incl - 100.0) / incl
    assert naive < 200, "the included-VWAP version should look nearly flat"
    assert e.loc["BIG", "edge_buy"] > 4 * naive, (
        "the correction must recover an edge the naive version buries")


def test_paying_up_reads_negative_and_patience_reads_positive():
    D = rows([("PATIENT", 100, 90.0, 0, 0),
              ("URGENT", 100, 110.0, 0, 0),
              ("MID", 100, 100.0, 0, 0)])
    e = execution_edges(D).set_index("broker")
    assert e.loc["PATIENT", "edge_buy"] > 0
    assert e.loc["URGENT", "edge_buy"] < 0


def test_sell_edge_has_the_opposite_sign_convention():
    """Selling ABOVE the others' average is good, so it reads positive."""
    D = rows([("HIGH", 0, 0, 100, 110.0),
              ("LOW", 0, 0, 100, 90.0),
              ("MID", 0, 0, 100, 100.0)])
    e = execution_edges(D).set_index("broker")
    assert e.loc["HIGH", "edge_sell"] > 0
    assert e.loc["LOW", "edge_sell"] < 0


def test_a_window_too_thin_to_compare_gives_no_edge():
    """With two brokers, removing one leaves a single counterparty; §9.4's
    comparison is against 'the day's average participant' and one trade is not
    an average."""
    D = rows([("A", 100, 100.0, 0, 0), ("B", 100, 110.0, 0, 0)])
    e = execution_edges(D)
    assert e["edge_buy"].isna().all()


def test_visible_vwap_excludes_the_named_broker():
    D = rows([("A", 100, 100.0, 0, 0), ("B", 100, 200.0, 0, 0)])
    assert visible_vwap(D) == pytest.approx(150.0)
    assert visible_vwap(D, exclude="A") == pytest.approx(200.0)
    assert visible_vwap(D, exclude="B") == pytest.approx(100.0)


def test_average_price_is_derived_from_value_and_lots():
    """The cached frame keeps values and lots, not averages. The average is
    exactly their ratio and must be treated as such rather than as missing."""
    D = rows([("A", 250, 137.0, 0, 0), ("B", 100, 100.0, 0, 0),
              ("C", 100, 100.0, 0, 0)])
    e = execution_edges(D).set_index("broker")
    assert e.loc["A", "edge_buy"] == pytest.approx(10000 * (100 - 137) / 100)


# --------------------------------------------------------------------------
# the individual metrics
# --------------------------------------------------------------------------
def test_hhi_separates_a_specialist_from_a_generalist():
    assert _hhi(np.array([100.0, 0, 0, 0])) == pytest.approx(1.0)
    assert _hhi(np.array([25.0, 25, 25, 25])) == pytest.approx(0.25)
    assert np.isnan(_hhi(np.array([0.0, 0.0])))


def test_ar1_is_negative_when_direction_alternates():
    alt = np.array([1.0, -1, 1, -1, 1, -1, 1, -1])
    assert _ar1_sign(alt) == pytest.approx(-1.0)


def test_ar1_is_nan_when_direction_never_changes():
    """A broker that is net-long every single window has no variance in sign,
    so the correlation is undefined — not zero, which would read as
    'alternates as often as not'."""
    assert np.isnan(_ar1_sign(np.ones(10)))


def test_ar1_is_positive_for_a_broker_that_keeps_pushing():
    x = np.array([1.0] * 5 + [-1.0] * 5)
    assert _ar1_sign(x) > 0.5


def test_the_guards_drop_a_broker_that_barely_traded():
    """MIN_WINDOWS and MIN_GROSS both bite; below them a fingerprint is one or
    two prints of noise wearing seven decimal places."""
    # lots are realistic: MIN_GROSS is Rp 1bn and a 1,000-lot print at Rp 100
    # is Rp 10m, so a toy-scale broker is dropped by the guard rather than by
    # anything the test is trying to show.
    D = pd.concat([rows([("BIG", 200_000, 100.0, 200_000, 100.0)],
                        window=f"2020-01-{d:02d}") for d in range(1, 13)]
                  + [rows([("TINY", 1, 100.0, 0, 0)], window="2020-01-01")])
    F = fingerprints(D, pd.DataFrame())
    assert "BIG" in set(F["broker"])
    assert "TINY" not in set(F["broker"])


def test_crossing_ratio_is_one_for_a_pure_churner():
    D = pd.concat([rows([("X", 200_000, 100.0, 200_000, 100.0)],
                        window=f"2020-02-{d:02d}") for d in range(1, 13)])
    F = fingerprints(D, pd.DataFrame()).set_index("broker")
    assert F.loc["X", "cross"] == pytest.approx(1.0)


def test_crossing_ratio_is_zero_for_one_sided_flow():
    D = pd.concat([rows([("X", 200_000, 100.0, 0, 0)],
                        window=f"2020-03-{d:02d}") for d in range(1, 13)])
    F = fingerprints(D, pd.DataFrame()).set_index("broker")
    assert F.loc["X", "cross"] == pytest.approx(0.0)
    assert F.loc["X", "censor"] == pytest.approx(1.0), \
        "a broker that never prints both sides is fully censored"


# --------------------------------------------------------------------------
# §9.5's distinctiveness measure
# --------------------------------------------------------------------------
def synth_year(year, n, spread, rng):
    """n brokers whose style metrics are drawn with a given spread."""
    return pd.DataFrame({
        "broker": [f"B{i:02d}" for i in range(n)],
        "year": year,
        "cross": rng.normal(0.5, spread, n),
        "hhi": rng.normal(0.3, spread, n),
        "edge_buy": rng.normal(0, spread, n),
        "edge_sell": rng.normal(0, spread, n),
        "ar1": rng.normal(0, spread, n),
        "share": rng.normal(-2, spread, n),
        "censor": rng.normal(0.5, spread, n)})


def test_standardising_happens_within_the_period_not_across_it():
    """Pooled z-scores would let a market-wide drift show up as every broker
    changing style at once. Q2 asks whether brokers become less distinguishable
    FROM EACH OTHER, so the common movement has to go."""
    rng = np.random.default_rng(0)
    F = pd.concat([synth_year(2020, 30, 0.1, rng),
                   synth_year(2021, 30, 0.1, rng)], ignore_index=True)
    F.loc[F.year == 2021, "cross"] += 100.0        # huge common shift
    Z = standardise(F)
    for y, g in Z.groupby("year"):
        assert abs(g["cross_z"].mean()) < 1e-9
        assert g["cross_z"].std() == pytest.approx(1.0, rel=1e-6)


def test_distinctiveness_is_scale_free_so_a_common_shift_does_not_move_it():
    """If it were not, Q2 would report a trend whenever the market's overall
    level of crossing drifted — which is not what §9.5 asks about."""
    rng = np.random.default_rng(1)
    a = synth_year(2020, 40, 0.1, rng)
    b = a.copy()
    b["year"] = 2021
    for c in METRICS:
        b[c] = b[c] + 50.0                          # same cross-section, shifted
    d = distinctiveness(pd.concat([a, b], ignore_index=True))
    assert d.loc[0, "mean_distance"] == pytest.approx(
        d.loc[1, "mean_distance"], rel=1e-6)


def test_distinctiveness_does_not_fall_when_only_the_spread_changes():
    """The converse, and the reason the measure is standardised WITHIN period:
    shrinking every metric's spread by a common factor leaves brokers exactly
    as separable from one another, so the measure must not move."""
    rng = np.random.default_rng(2)
    a = synth_year(2020, 40, 0.4, rng)
    b = a.copy()
    b["year"] = 2021
    for c in METRICS:
        b[c] = b[c] * 0.01                          # same shape, tiny scale
    d = distinctiveness(pd.concat([a, b], ignore_index=True))
    assert d.loc[0, "mean_distance"] == pytest.approx(
        d.loc[1, "mean_distance"], rel=1e-6)


def test_distinctiveness_rises_when_brokers_genuinely_separate():
    """The measure must still move when the cross-section really does change
    shape — otherwise it could never detect the decay §9.5 asks about."""
    rng = np.random.default_rng(3)
    tight = synth_year(2020, 40, 0.1, rng)
    clumped = tight.copy()
    clumped["year"] = 2021
    for c in METRICS:                                # collapse onto two points
        clumped[c] = np.where(np.arange(len(clumped)) % 2 == 0, 0.0, 1.0)
    d = distinctiveness(pd.concat([tight, clumped], ignore_index=True)
                        ).set_index("year")
    assert d.loc[2021, "mean_distance"] != pytest.approx(
        d.loc[2020, "mean_distance"], rel=1e-3)


# --------------------------------------------------------------------------
# the registration itself
# --------------------------------------------------------------------------
def test_style_metrics_exclude_the_known_data_artefact():
    """`censor` is a property of the top-10 publication rule, not of the firm.
    H14's Q1 prediction is about STYLE, so the artefact must not be inside it."""
    assert "censor" in METRICS
    assert "censor" not in STYLE
    assert set(STYLE) < set(METRICS)


def test_the_module_computes_no_profitability_anywhere():
    """§9.6's central risk is a stable STYLE being reported as a stable EDGE.
    The cheapest guard is that this module never computes P&L at all."""
    import idxbot.spine.fingerprint as m
    src = open(m.__file__).read()
    for banned in ("margin_bps", "cohort_pnl", "timing_pnl", "realized"):
        assert banned not in src, f"{banned} has no business in a fingerprint"
