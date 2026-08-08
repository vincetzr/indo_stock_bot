"""Fundamentals parsing, the currency trap, and the exclusion screen.

No network here: every test feeds the parser a payload shaped exactly like
Yahoo's, so the expected answer is known rather than merely plausible.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

from idxbot.data.fundamentals import (  # noqa: E402
    Fundamentals,
    _from_frame,
    _parse,
    _raw,
    quality_flags,
    render,
    screen,
)


def _payload(currency="IDR", price_to_book=2.9, pe=13.5, roe=0.22,
             margin=0.53, d_e=None, current_ratio=None, periods=4):
    return {
        "financialData": {
            "financialCurrency": currency,
            "returnOnEquity": {"raw": roe},
            "profitMargins": {"raw": margin},
            "revenueGrowth": {"raw": 0.02},
            "earningsGrowth": {"raw": 0.05},
            **({"debtToEquity": {"raw": d_e}} if d_e is not None else {}),
            **({"currentRatio": {"raw": current_ratio}} if current_ratio is not None else {}),
        },
        "defaultKeyStatistics": {"priceToBook": {"raw": price_to_book}},
        "summaryDetail": {"trailingPE": {"raw": pe}, "marketCap": {"raw": 1e14}},
        "incomeStatementHistory": {"incomeStatementHistory": [
            {"totalRevenue": {"raw": 1e12}, "netIncome": {"raw": 1e11}}
        ] * periods},
        "balanceSheetHistory": {"balanceSheetStatements": []},
    }


# --------------------------------------------------------------------------
# _raw
# --------------------------------------------------------------------------

def test_raw_unwraps_yahoo_boxes_and_bare_numbers():
    assert _raw({"raw": 1.5, "fmt": "1.50"}) == 1.5
    assert _raw(2.25) == 2.25


def test_raw_returns_nan_for_missing_or_junk():
    for junk in (None, {}, {"fmt": "N/A"}, "abc", float("inf")):
        assert np.isnan(_raw(junk))


# --------------------------------------------------------------------------
# the currency trap
# --------------------------------------------------------------------------

def test_idr_reporter_is_left_alone():
    f = _parse("BBCA", _payload(currency="IDR", price_to_book=2.9), fx_rate=17885.0)
    assert f.price_to_book == pytest.approx(2.9)
    assert f.fx_corrected is False


def test_usd_reporter_has_price_to_book_repaired():
    """ADRO's real numbers: Yahoo divides an IDR price by a USD book value."""
    f = _parse("ADRO", _payload(currency="USD", price_to_book=14941.177),
               fx_rate=17885.0)
    assert f.price_to_book == pytest.approx(0.8354, abs=1e-3)
    assert f.fx_corrected is True
    assert f.financial_currency == "USD"


def test_pe_is_never_touched_by_the_currency_repair():
    # trailingEps is already served in IDR, so PE is correct as published.
    f = _parse("ADRO", _payload(currency="USD", price_to_book=14941.177, pe=8.28),
               fx_rate=17885.0)
    assert f.trailing_pe == pytest.approx(8.28)


def test_ratios_are_currency_neutral_and_survive_untouched():
    f = _parse("INCO", _payload(currency="USD", roe=0.06, margin=0.14), fx_rate=17885.0)
    assert f.return_on_equity == pytest.approx(0.06)
    assert f.profit_margin == pytest.approx(0.14)


def test_missing_fx_rate_leaves_the_bogus_value_rather_than_inventing_one():
    # Better a visibly absurd number than a silently fabricated plausible one.
    f = _parse("ADRO", _payload(currency="USD", price_to_book=14941.177),
               fx_rate=np.nan)
    assert f.price_to_book == pytest.approx(14941.177)
    assert f.fx_corrected is False


# --------------------------------------------------------------------------
# exclusions
# --------------------------------------------------------------------------

def test_healthy_company_has_no_exclusion_reasons():
    f = _parse("BBCA", _payload(roe=0.22, margin=0.53, d_e=40, current_ratio=1.4))
    assert quality_flags(f) == []


@pytest.mark.parametrize("kwargs,fragment", [
    ({"roe": -0.15}, "negative ROE"),
    ({"margin": -0.08}, "loss-making"),
    ({"d_e": 350}, "debt/equity"),
    ({"pe": 180}, "PE"),
    ({"current_ratio": 0.5, "d_e": 160}, "current ratio"),
])
def test_each_exclusion_fires_on_its_own_condition(kwargs, fragment):
    f = _parse("X", _payload(**kwargs))
    reasons = quality_flags(f)
    assert any(fragment in r for r in reasons), reasons


def test_exclusions_do_not_fire_on_missing_data():
    # An absent field must never be read as a failing one.
    f = Fundamentals(ticker="X")
    assert quality_flags(f) == []


def test_low_current_ratio_alone_is_not_distress():
    """TLKM, MTEL and UNVR all sit below 1.0 and are not in trouble.

    Banks, telcos and toll roads fund long-dated assets with rolling short-term
    debt, so a sub-1 current ratio is their normal state. Flagging it on its own
    deleted the defensive half of LQ45.
    """
    healthy = _parse("TLKM", _payload(current_ratio=0.79, d_e=60, roe=0.18))
    assert quality_flags(healthy) == []


def test_low_current_ratio_with_high_leverage_is_distress():
    levered = _parse("TOWR", _payload(current_ratio=0.20, d_e=157, roe=0.16))
    assert any("current ratio" in r for r in quality_flags(levered))


# --------------------------------------------------------------------------
# screen / render
# --------------------------------------------------------------------------

def test_screen_marks_excluded_rows_and_records_why():
    universe = {
        "GOOD": _parse("GOOD", _payload(roe=0.2, margin=0.3)),
        "BAD": _parse("BAD", _payload(roe=-0.2, margin=-0.1)),
    }
    df = screen(universe)
    assert bool(df.loc[df.ticker == "GOOD", "excluded"].iloc[0]) is False
    bad = df.loc[df.ticker == "BAD"].iloc[0]
    assert bool(bad["excluded"]) is True
    assert "negative ROE" in bad["reasons"]


def test_render_keeps_columns_separated_for_an_extreme_value():
    """The bug this guards: an uncorrected P/B of 14,941 merged two columns."""
    universe = {"ADRO": _parse("ADRO", _payload(currency="USD",
                                                price_to_book=14941.177),
                               fx_rate=np.nan)}
    text = render(screen(universe))
    row = [ln for ln in text.splitlines() if ln.strip().startswith("ADRO")][0]
    header = [ln for ln in text.splitlines() if "ticker" in ln and "P/B" in ln][0]
    assert len(row.split()) >= 6          # fields did not fuse into one token
    assert len(row) <= len(header) + 40   # and the row did not blow the layout


def test_render_survives_an_empty_screen():
    assert "No fundamentals retrieved" in render(pd.DataFrame())


def test_render_states_that_it_cannot_be_backtested():
    text = render(screen({"BBCA": _parse("BBCA", _payload())}))
    assert "NOT backtestable" in text


# --------------------------------------------------------------------------
# cache round-trip
# --------------------------------------------------------------------------

def test_cache_round_trip_preserves_the_currency_repair():
    f = _parse("ADRO", _payload(currency="USD", price_to_book=14941.177),
               fx_rate=17885.0)
    restored = _from_frame("ADRO", pd.DataFrame([f.as_row()]))
    assert restored.price_to_book == pytest.approx(f.price_to_book)
    assert restored.fx_corrected is True
    assert restored.financial_currency == "USD"


def test_cache_round_trip_tolerates_a_file_without_the_new_columns():
    # Entries written before the currency fields existed must still load.
    legacy = pd.DataFrame([{"ticker": "BBCA", "trailing_pe": 13.5,
                            "price_to_book": 2.9, "roe": 0.22}])
    restored = _from_frame("BBCA", legacy)
    assert restored.trailing_pe == pytest.approx(13.5)
    assert restored.fx_corrected is False
    assert restored.financial_currency == ""
