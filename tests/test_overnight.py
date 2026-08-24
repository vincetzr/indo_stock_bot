"""Tests for the overnight/global block.

THE CLOCK IS THE WHOLE DIFFICULTY AND THE FIRST BLOCK IS ABOUT IT. Jakarta
closes 08:50 UTC; New York closes 20:00 UTC the SAME calendar date. So a bar
dated 2026-08-24 means "already priced by IDX" for Tokyo and "not yet seen by
IDX" for New York, and every number in this module turns on telling those
apart. The first implementation tested `date > idx_day`, found nothing, and
reported a silent NaN for every symbol on the board.

The second block is the statistics. Switching from Pearson to Spearman changed
the headline completely — Pearson said the S&P was uncorrelated with IDX at
r = -0.001 when the rank figure is +0.207 — because these series carry
kurtosis from 10 to 2,800 and Pearson on that is a statistic about four days.
Yahoo's IDR=X additionally carries decimal-shift defects that must be dropped
rather than winsorised into something plausible-looking.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

from idxbot.data import overnight as O                         # noqa: E402


def series(dates, closes) -> pd.DataFrame:
    return pd.DataFrame({"date": pd.to_datetime(dates),
                         "close": np.asarray(closes, dtype=float)})


# --------------------------------------------------------------------------
# THE CLOCK
# --------------------------------------------------------------------------
def test_a_market_closing_after_jakarta_uses_its_same_date_bar():
    """New York's bar dated like today's IDX session landed eleven hours
    later, so it IS overnight news. Requiring a strictly later date finds
    nothing and reports NaN for the entire board — what the first version did.
    """
    d = series(["2026-08-21", "2026-08-24"], [100.0, 102.0])
    r = O._overnight_ret(d, pd.Timestamp("2026-08-24"), after_jakarta=True)
    assert r == pytest.approx(0.02)


def test_a_market_closing_before_jakarta_has_no_overnight_number():
    """Tokyo shuts 06:00 UTC, before Jakarta's 08:50. Its same-date session was
    visible while IDX still traded, so calling it 'overnight' would credit the
    reader with information the market had already priced."""
    d = series(["2026-08-21", "2026-08-24"], [100.0, 102.0])
    assert np.isnan(O._overnight_ret(d, pd.Timestamp("2026-08-24"),
                                     after_jakarta=False))


def test_a_feed_that_has_not_printed_yet_gives_nan_not_zero():
    """'No bar yet' and 'unchanged' are different facts; rendering them the
    same way is lying by formatting."""
    d = series(["2026-08-20", "2026-08-21"], [100.0, 101.0])
    assert np.isnan(O._overnight_ret(d, pd.Timestamp("2026-08-24"),
                                     after_jakarta=True))


def test_the_board_separates_closes_before_from_feed_behind():
    """Two different reasons for a blank, and they call for different
    responses: one is a fact about the clock, the other about the feed."""
    bars = {"^GSPC": series(["2026-08-21", "2026-08-24"], [100, 101]),
            "^N225": series(["2026-08-21", "2026-08-24"], [100, 101]),
            "GC=F": series(["2026-08-20", "2026-08-21"], [100, 101])}
    Bd = O.board(bars, pd.Timestamp("2026-08-24")).set_index("symbol")
    assert Bd.loc["^GSPC", "after_jakarta"] and not Bd.loc["^GSPC", "behind"]
    assert not Bd.loc["^N225", "after_jakarta"]
    assert Bd.loc["GC=F", "after_jakarta"] and Bd.loc["GC=F", "behind"]
    assert np.isnan(Bd.loc["^N225", "overnight"])
    assert np.isnan(Bd.loc["GC=F", "overnight"])


def test_every_after_jakarta_symbol_is_one_the_board_knows():
    assert O.AFTER_JAKARTA <= {s for s, _ in O.SYMBOLS}


def test_the_asian_indices_are_not_marked_as_closing_after_jakarta():
    for s in ("^N225", "^HSI", "000001.SS"):
        assert s not in O.AFTER_JAKARTA


# --------------------------------------------------------------------------
# NO LOOKAHEAD IN THE HISTORICAL ALIGNMENT
# --------------------------------------------------------------------------
def test_each_idx_session_is_matched_to_the_previous_global_move():
    """Pairing by date would regress Jakarta's session on news that had not
    happened when it closed."""
    idx_dates = pd.to_datetime(["2026-08-19", "2026-08-20", "2026-08-21"])
    bars = {"^GSPC": series(["2026-08-18", "2026-08-19", "2026-08-20",
                             "2026-08-21"], [100.0, 110.0, 121.0, 200.0])}
    a = O._align(bars, "^GSPC", idx_dates)
    # session 08-20 gets the move INTO 08-19 (+10%), not 08-20's own
    assert a.loc[pd.Timestamp("2026-08-20")] == pytest.approx(0.10)
    assert a.loc[pd.Timestamp("2026-08-21")] == pytest.approx(0.10, rel=1e-6)
    # the huge 08-21 move must not reach any session in this window
    assert (a.dropna() < 0.5).all()


def test_a_rate_is_differenced_and_a_price_is_not():
    """^TNX fell 0.93 -> 0.50 in March 2020: a real 43 bp move and a spurious
    -46% return. Percentage change hands the sample's weight to the fortnight
    the yield sat near zero."""
    idx = pd.to_datetime(["2026-08-19", "2026-08-20", "2026-08-21"])
    bars = {"^TNX": series(["2026-08-18", "2026-08-19", "2026-08-20",
                            "2026-08-21"], [4.0, 4.2, 4.1, 4.5])}
    a = O._align(bars, "^TNX", idx)
    assert a.loc[pd.Timestamp("2026-08-20")] == pytest.approx(0.2)  # not 5%
    assert "^TNX" in O.DIFFERENCED


def test_a_vendor_decimal_shift_is_dropped_not_kept():
    """IDR=X prints 888.11 against a true ~8,881 and reverses the next day,
    giving +903% then -90%. Those are defects, not market events."""
    idx = pd.to_datetime(["2026-08-18", "2026-08-19", "2026-08-20",
                          "2026-08-21"])
    bars = {"IDR=X": series(["2026-08-17", "2026-08-18", "2026-08-19",
                             "2026-08-20", "2026-08-21"],
                            [8900.0, 890.0, 8905.0, 8910.0, 8920.0])}
    a = O._align(bars, "IDR=X", idx)
    kept = a.dropna()
    assert (kept.abs() <= O.IMPLAUSIBLE).all(), "the shift must not survive"


def test_a_real_large_move_below_the_threshold_survives():
    """The guard must not eat a genuine crash. -20% in a day is 1987, not a
    decimal shift."""
    idx = pd.to_datetime(["2026-08-19", "2026-08-20"])
    bars = {"^GSPC": series(["2026-08-18", "2026-08-19", "2026-08-20"],
                            [100.0, 80.0, 82.0])}
    a = O._align(bars, "^GSPC", idx)
    assert a.loc[pd.Timestamp("2026-08-20")] == pytest.approx(-0.20)


# --------------------------------------------------------------------------
# THE STATISTICS
# --------------------------------------------------------------------------
def test_the_block_bootstrap_recovers_a_known_correlation():
    """The estimator that produced the published intervals, checked against a
    sample whose true correlation is known by construction."""
    rng = np.random.default_rng(0)
    n = 4000
    a = rng.normal(size=n)
    b = 0.10 * a + np.sqrt(1 - 0.01) * rng.normal(size=n)
    r = float(np.corrcoef(a, b)[0, 1])
    lo, hi = O._block_ci(a, b, 400, rng, 21)
    assert lo <= r <= hi
    assert abs((lo + hi) / 2 - r) < 0.02, "the interval must be centred on r"
    assert lo < 0.10 < hi


def test_the_bootstrap_declines_a_sample_too_short_to_block():
    rng = np.random.default_rng(0)
    lo, hi = O._block_ci(np.arange(10.0), np.arange(10.0), 50, rng, 21)
    assert np.isnan(lo) and np.isnan(hi)


def test_rank_correlation_is_not_hijacked_by_four_outliers():
    """THE REGRESSION THAT CHANGED THE HEADLINE. Two series that agree on
    nearly every day but disagree violently on four. Pearson reports the four;
    Spearman reports the many, and IDX's macro series carry kurtosis up to
    2,800.
    """
    rng = np.random.default_rng(1)
    n = 2000
    x = rng.normal(size=n)
    y = 0.5 * x + 0.5 * rng.normal(size=n)
    for i in (10, 500, 1200, 1900):                 # four violent disagreements
        x[i], y[i] = 60.0, -60.0
    from scipy.stats import rankdata
    pear = float(np.corrcoef(x, y)[0, 1])
    spear = float(np.corrcoef(rankdata(x), rankdata(y))[0, 1])
    assert pear < 0.0, "Pearson is dragged negative by four points"
    assert spear > 0.4, "Spearman keeps the relationship the other 1,996 show"


def test_sensitivity_reports_the_stale_share_so_a_ffilled_series_is_visible():
    """Aluminium runs 19% exact zeros — forward fill, not a flat market — and
    its correlation should not be read beside one that prints every day."""
    idx_dates = pd.bdate_range("2015-01-01", periods=900)
    rng = np.random.default_rng(2)
    y = pd.Series(rng.normal(0, 0.01, len(idx_dates)), index=idx_dates)
    px = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(idx_dates) + 5)))
    d = series(pd.bdate_range("2014-12-25", periods=len(px)), px)
    # every other bar forward-filled, exactly the shape a thin future has
    d.loc[d.index % 2 == 1, "close"] = np.nan
    d["close"] = d["close"].ffill()
    S = O.sensitivity({"^GSPC": d.dropna()}, y, draws=30)
    if not S.empty:
        assert "stale" in S
        assert S["stale"].iloc[0] > 0.3, "a half-ffilled series must show it"


def test_sensitivity_skips_a_series_too_short_to_speak():
    idx_dates = pd.bdate_range("2024-01-01", periods=100)
    y = pd.Series(np.random.default_rng(0).normal(0, 0.01, 100),
                  index=idx_dates)
    d = series(pd.bdate_range("2024-01-01", periods=100), np.arange(100.0) + 1)
    assert O.sensitivity({"^GSPC": d}, y, draws=10).empty


# --------------------------------------------------------------------------
# HONESTY OF THE PROXIES
# --------------------------------------------------------------------------
def test_every_proxy_is_labelled_with_what_it_stands_for():
    """IDX's economy is coal, nickel and palm oil and none has a usable free
    series. An unlabelled proxy becomes the thing it proxies."""
    for sym in O.PROXY_NOTE:
        assert sym in {s for s, _ in O.SYMBOLS}
        assert len(O.PROXY_NOTE[sym]) > 10
    Bd = O.board({}, pd.Timestamp("2026-08-24")).set_index("symbol")
    assert "coal" in Bd.loc["GLEN.L", "proxy"]


def test_the_board_still_returns_a_row_for_a_symbol_that_failed():
    """A missing feed must be visibly missing, not silently absent."""
    Bd = O.board({}, pd.Timestamp("2026-08-24"))
    assert len(Bd) == len(O.SYMBOLS)
    assert Bd["behind"].all()
    assert Bd["overnight"].isna().all()
