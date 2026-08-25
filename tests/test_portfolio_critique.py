"""Tests for the H20 critique — the benchmark, the power check, the dividend.

The three functions here decide whether H20's conclusion stands, so each is
pinned against a case whose answer is known by construction rather than by
re-running the study.
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

from portfolio_critique import (LANDMARKS, agg, bench_slots,  # noqa: E402
                                index_cagr, max_dd, power, validate_index)
from portfolio_sim import slots                                 # noqa: E402


def _ix(rates, start="2005-01-01", n=250):
    """A synthetic index compounding at a known annual rate per year."""
    d, lv, out = pd.Timestamp(start), 100.0, []
    idx = []
    for r in rates:
        step = (1.0 + r) ** (1.0 / n)
        for _ in range(n):
            out.append(lv)
            idx.append(d)
            lv *= step
            d += pd.Timedelta(days=365.25 / n)
    return pd.Series(out, index=pd.DatetimeIndex(idx))


# ======================================================================= CAGR
def test_index_cagr_recovers_a_known_constant_rate():
    s = _ix([0.10] * 5)
    got = index_cagr(s, s.index[0], s.index[-1])
    assert got == pytest.approx(0.10, abs=0.005)


def test_index_cagr_uses_the_nearest_prior_close_to_each_endpoint():
    """The cohort dates are month-starts and the index does not trade every
    one of them; asking for a non-trading day must not return NaN."""
    s = _ix([0.10] * 3)
    a = s.index[0] + pd.Timedelta(days=3)
    assert np.isfinite(index_cagr(s, a, s.index[-1]))


def test_index_cagr_is_nan_when_the_window_is_degenerate():
    s = _ix([0.10] * 2)
    assert np.isnan(index_cagr(s, s.index[-1], s.index[0]))     # reversed
    assert np.isnan(index_cagr(s, s.index[0] - pd.Timedelta(days=900),
                               s.index[0] - pd.Timedelta(days=800)))


def test_index_cagr_signs_a_decline_negative():
    assert index_cagr(_ix([-0.20] * 4), *_ix([-0.20] * 4).index[[0, -1]]) < 0


# =================================================================== drawdown
def test_max_dd_is_measured_peak_to_trough_on_the_path():
    s = pd.Series([100.0, 150.0, 75.0, 200.0],
                  index=pd.date_range("2010-01-01", periods=4))
    #  the drop is 150 -> 75, i.e. -50%, not 200 -> 75
    assert max_dd(s, s.index[0], s.index[-1]) == pytest.approx(-0.5)


def test_max_dd_is_zero_on_a_monotone_rise():
    s = _ix([0.10] * 3)
    assert max_dd(s, s.index[0], s.index[-1]) == pytest.approx(0.0, abs=1e-9)


def test_max_dd_respects_the_window():
    s = pd.Series([100.0, 40.0, 100.0, 110.0, 120.0],
                  index=pd.date_range("2010-01-01", periods=5))
    assert max_dd(s, s.index[0], s.index[-1]) == pytest.approx(-0.6)
    assert max_dd(s, s.index[2], s.index[-1]) == pytest.approx(0.0)


# ====================================================================== power
def test_power_reports_an_interval_that_covers_a_target_inside_it():
    """THE DISTINCTION A8 INSISTS ON. A wide interval around zero is not a
    null result — it is an unpowered one, and the flag has to say which."""
    x = np.array([0.0, 0.10, -0.10, 0.05, -0.05, 0.02, -0.02, 0.08])
    p = power(x, 0.03)
    assert p["covers_target"] is True
    assert p["lo"] < 0.03 < p["hi"]


def test_power_excludes_a_target_outside_a_tight_interval():
    x = np.full(12, 0.001) + np.linspace(-1e-4, 1e-4, 12)
    p = power(x, 0.06)
    assert p["covers_target"] is False


def test_power_mde_shrinks_as_the_sample_grows():
    rng = np.random.default_rng(0)
    small = power(rng.normal(0, 0.05, 8), 0.0)["mde"]
    large = power(rng.normal(0, 0.05, 200), 0.0)["mde"]
    assert large < small


def test_power_mde_is_the_half_width_of_its_own_interval():
    x = np.array([0.01, 0.03, -0.02, 0.05, 0.00, 0.04, -0.01, 0.02])
    p = power(x, 0.0)
    assert p["mde"] == pytest.approx((p["hi"] - p["lo"]) / 2)


def test_power_declines_to_answer_on_too_few_slots():
    assert power(np.array([0.01, 0.02]), 0.0) == {}


# ================================================================ median/mean
def test_agg_returns_median_then_mean_and_they_differ_on_a_skew():
    """The +10.5% vs +9.8% discrepancy between the memo and this script was
    entirely this choice, so both are returned and neither is implicit."""
    S = pd.DataFrame({"cagr": [0.10, 0.10, 0.10, 0.10, -0.40]})
    med, mean = agg(S)
    assert med == pytest.approx(0.10)
    assert mean == pytest.approx(0.0)
    assert med != pytest.approx(mean)


def test_agg_reads_the_column_it_is_asked_for():
    S = pd.DataFrame({"cagr": [0.1, 0.2], "maxdd": [-0.5, -0.3]})
    assert agg(S, "maxdd")[0] == pytest.approx(-0.4)


def test_agg_is_nan_on_an_empty_frame_rather_than_raising():
    assert all(np.isnan(v) for v in agg(pd.DataFrame()))
    assert all(np.isnan(v) for v in agg(None))


# ================================================== the paired benchmark (C3b)
def _B(rets, held, start="2006-01-01"):
    d = pd.date_range(start, periods=len(rets), freq="MS")
    return pd.DataFrame({"as_of": d, "n": 12, "ret": rets, "med": rets,
                         "logret": np.log1p(rets), "p2": 0.0, "pdn": 0.0,
                         "held": held})


def test_a_slot_span_ends_after_its_last_entry_not_on_it():
    """DEFECT ONE. The final position is still open for its holding period,
    so a benchmark measured to the last entry date is short by that year."""
    S = slots(_B(np.zeros(60), np.full(60, 252.0)), n_slots=1)
    r = S.iloc[0]
    assert r["end"] > r["start"]
    assert (r["end"] - r["start"]).days / 365.25 == pytest.approx(
        r["years"], abs=0.01)
    #  the span must exceed the gap between first and last ENTRY
    assert r["end"] > pd.Timestamp("2006-01-01") + pd.Timedelta(days=365)


def test_each_slot_reports_its_own_distinct_span():
    """DEFECT TWO. Twelve slots begin in twelve different months, so one
    global benchmark window compares each to a period it did not occupy."""
    S = slots(_B(np.zeros(120), np.full(120, 252.0)), n_slots=12)
    assert S["start"].nunique() == len(S)


def test_bench_slots_pairs_each_slot_against_its_own_window():
    S = slots(_B(np.zeros(120), np.full(120, 252.0)), n_slots=12)
    ix = _ix([0.10] * 20, start="2005-01-01")
    Bs = bench_slots(ix, S)
    assert len(Bs) == len(S)
    assert Bs["index"].notna().all()
    #  a flat 10%/yr index against zero-return picks -> uniformly negative
    assert (Bs["d"] < 0).all()
    assert Bs["d"].mean() == pytest.approx(-0.10, abs=0.02)


def test_bench_slots_adds_the_yield_to_the_index_side_only():
    S = slots(_B(np.zeros(120), np.full(120, 252.0)), n_slots=4)
    ix = _ix([0.10] * 20, start="2005-01-01")
    a = bench_slots(ix, S, yld=0.0)
    b = bench_slots(ix, S, yld=0.02)
    assert np.allclose(b["index"] - a["index"], 0.02)
    assert np.allclose(a["picks"], b["picks"])
    assert np.allclose(a["d"] - b["d"], 0.02)


def test_bench_slots_signs_an_outperforming_basket_positive():
    S = slots(_B(np.full(120, 0.25), np.full(120, 252.0)), n_slots=4)
    ix = _ix([0.02] * 20, start="2005-01-01")
    assert (bench_slots(ix, S)["d"] > 0).all()


# ============================================ validating the benchmark itself
def _real() -> pd.Series:
    ix = pd.read_csv(os.path.join(os.path.dirname(__file__), os.pardir,
                                  "data", "cache", "ohlcv", "_JKSE.csv.gz"))
    ix["date"] = pd.to_datetime(ix["date"], utc=True,
                                errors="coerce").dt.tz_localize(None)
    s = ix.set_index("date")["close"].astype(float).sort_index().dropna()
    return s[s > 0]


_HAVE = os.path.exists(os.path.join(os.path.dirname(__file__), os.pardir,
                                    "data", "cache", "ohlcv", "_JKSE.csv.gz"))
_skip = pytest.mark.skipif(not _HAVE, reason="IHSG cache not present")


def test_a_decimal_shift_is_caught():
    """A13's EXACT DEFECT, from this same unauthenticated endpoint: `IDR=X`
    printed 888.11 against a true ~8,881 and reversed the next day. An
    endpoint-to-endpoint CAGR reads two bars, so a defect at either end is
    maximally damaging rather than averaged away."""
    s = _ix([0.10] * 6)
    s.iloc[len(s) // 2] /= 10.0
    v = validate_index(s)
    assert v["n_moves_over_20pct"] >= 2          # down and straight back up


def test_a_clean_series_passes_every_internal_check():
    v = validate_index(_ix([0.10] * 6))
    assert v["n_moves_over_20pct"] == 0
    assert v["max_gap_days"] <= 12


def test_a_long_calendar_gap_is_reported():
    s = _ix([0.10] * 4)
    s = pd.concat([s.iloc[:100], s.iloc[160:]])
    assert validate_index(s)["n_gaps_over_12d"] >= 1


@_skip
def test_the_real_ihsg_matches_every_published_landmark():
    v = validate_index(_real())
    assert v["worst_landmark_error"] < 0.001, (
        "the cached IHSG disagrees with published year-end closes — every "
        "benchmark number in reports/portfolio.md rests on this series")


@_skip
def test_the_real_ihsg_has_no_decimal_shift_and_only_holiday_gaps():
    """The half of the check that needs NO external reference: the landmarks
    come from knowledge and can only confirm identity, but a decimal shift and
    a non-holiday gap both announce themselves from inside the series."""
    v = validate_index(_real())
    assert v["n_moves_over_20pct"] == 0
    assert v["n_gaps_over_12d"] == 0             # Idul Fitri is <= 12 days


@_skip
def test_the_landmark_dates_are_all_inside_the_cached_range():
    """A landmark outside the sample silently passes by matching nothing."""
    s = _real()
    for d in LANDMARKS:
        assert s.index.min() <= pd.Timestamp(d) <= s.index.max()
