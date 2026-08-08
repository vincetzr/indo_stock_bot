"""Acceptance test for broker-summary data.

The point of verify.py is catching bad data, so most of these construct a
specific defect and assert that the matching check fails. A verifier that only
passes good data is worthless.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

from idxbot.config import load_config  # noqa: E402
from idxbot.verify import FAIL, PASS, WARN, render, verify  # noqa: E402

cfg = load_config()

BULGE = ["BK", "AK", "KZ", "MS", "ML", "CG", "RX", "DB"]
OTHERS = ["YP", "PD", "CC", "NI", "LG", "DH", "AZ", "KI", "GR", "EP", "IF", "OD"]


def _good(n_days=300, n_brokers=20, price=1000.0, ticker="BBCA"):
    """A well-formed frame: balanced sides, values present, many brokers."""
    rng = np.random.default_rng(7)
    dates = pd.bdate_range("2024-01-01", periods=n_days)
    codes = (BULGE + OTHERS)[:n_brokers]
    rows = []
    for date in dates:
        # Split one day's volume so buys and sells total the same.
        weights = rng.dirichlet(np.ones(len(codes)))
        sell_weights = rng.dirichlet(np.ones(len(codes)))
        day_lots = 100_000.0
        for code, wb, ws in zip(codes, weights, sell_weights):
            buy_lot, sell_lot = day_lots * wb, day_lots * ws
            rows.append({
                "date": date, "ticker": ticker, "broker": code,
                "buy_lot": buy_lot, "sell_lot": sell_lot,
                "buy_val": buy_lot * 100 * price,
                "sell_val": sell_lot * 100 * price,
                "buy_avg": price, "sell_avg": price, "source": "test",
            })
    return pd.DataFrame(rows)


def _status(report, name):
    for check in report.checks:
        if check.name == name:
            return check.status
    return None


# --------------------------------------------------------------- good data ---
def test_good_data_passes_everything():
    report = verify(_good(), cfg.brokers)
    assert report.usable
    assert all(c.status == PASS for c in report.checks), \
        [(c.name, c.status, c.detail) for c in report.checks if c.status != PASS]


def test_render_reports_usable():
    assert "USABLE" in render(verify(_good(), cfg.brokers))


# ------------------------------------------------- defect: aggregate feed ----
def test_rejects_foreign_domestic_aggregate():
    """The most likely vendor mis-sell: aggregate flow labelled bandarmology."""
    df = _good(n_brokers=2)
    df["broker"] = np.where(df["broker"] == df["broker"].iloc[0], "FOREIGN", "DOMESTIC")
    report = verify(df, cfg.brokers)
    assert not report.usable
    assert _status(report, "per-broker granularity") == FAIL


def test_rejects_too_few_brokers_per_day():
    report = verify(_good(n_brokers=2), cfg.brokers)
    assert _status(report, "per-broker granularity") == FAIL


def test_warns_on_top_n_view():
    """A top-8 table is usable but biases inventory - warn, do not fail."""
    report = verify(_good(n_brokers=8), cfg.brokers)
    assert _status(report, "per-broker granularity") == WARN


# ------------------------------------------------ defect: missing values -----
def test_rejects_missing_value_columns():
    """Lots without rupiah means no VWAP, so no cost basis."""
    df = _good()
    df["buy_val"] = 0.0
    df["sell_val"] = 0.0
    report = verify(df, cfg.brokers)
    assert not report.usable
    assert _status(report, "value columns") == FAIL


def test_rejects_implausible_implied_price():
    """Value/volume unit mismatch shows up as an absurd implied price."""
    df = _good()
    df["buy_val"] = df["buy_val"] / 1e6
    df["sell_val"] = df["sell_val"] / 1e6
    report = verify(df, cfg.brokers)
    assert _status(report, "value columns") == FAIL


def test_cross_checks_values_against_real_prices():
    """Values that don't match the day's traded range are mislabelled."""
    df = _good(price=1000.0)
    dates = sorted(df["date"].unique())
    # Real bars sit at ~5000 while the feed implies ~1000.
    bars = pd.DataFrame({
        "date": dates, "open": 5000.0, "high": 5100.0,
        "low": 4900.0, "close": 5000.0, "volume": 1e7,
    })
    report = verify(df, cfg.brokers, prices={"BBCA": bars})
    assert _status(report, "value columns") == FAIL


# ----------------------------------------------- defect: broken balance ------
def test_rejects_unbalanced_sides():
    """Every lot bought is a lot sold; a big gap means misread columns."""
    df = _good()
    df["sell_lot"] = df["sell_lot"] * 0.4
    df["sell_val"] = df["sell_val"] * 0.4
    report = verify(df, cfg.brokers)
    assert not report.usable
    assert _status(report, "buy/sell balance") == FAIL


def test_balance_check_passes_when_sides_agree():
    assert _status(verify(_good(), cfg.brokers), "buy/sell balance") == PASS


# ----------------------------------------------- defect: thin history --------
def test_rejects_trivial_history():
    report = verify(_good(n_days=10), cfg.brokers)
    assert not report.usable
    assert _status(report, "history depth") == FAIL


def test_warns_on_short_history():
    assert _status(verify(_good(n_days=60), cfg.brokers), "history depth") == WARN


def test_warns_on_medium_history():
    assert _status(verify(_good(n_days=150), cfg.brokers), "history depth") == WARN


# --------------------------------------------- defect: no bulge desks --------
def test_rejects_data_without_institutional_desks():
    """The entire thesis is about these desks - their absence is fatal."""
    df = _good(n_brokers=20)
    df = df[~df["broker"].isin(BULGE)]
    report = verify(df, cfg.brokers)
    assert not report.usable
    assert _status(report, "institutional desks") == FAIL


def test_unknown_codes_are_reported_not_dropped():
    df = _good()
    df.loc[df["broker"] == "YP", "broker"] = "ZZ9"
    report = verify(df, cfg.brokers)
    detail = next(c.detail for c in report.checks if c.name == "institutional desks")
    assert "ZZ9" in detail


# ------------------------------------------------------------- edge cases ----
def test_empty_input_fails_cleanly():
    report = verify(pd.DataFrame(), cfg.brokers)
    assert not report.usable
    assert "NOT USABLE" in render(report)


def test_render_lists_failures():
    df = _good()
    df["buy_val"] = 0.0
    df["sell_val"] = 0.0
    text = render(verify(df, cfg.brokers))
    assert "NOT USABLE" in text
    assert "value columns" in text
