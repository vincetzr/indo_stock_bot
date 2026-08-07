"""Ledger reconstruction and campaign segmentation.

The ledger is the load-bearing piece: if the cost basis or realised P/L is
wrong, every downstream conclusion about a broker is wrong too. These tests use
hand-computable numbers so the expected values can be verified by inspection.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

from idxbot.analytics.broker_flow import (  # noqa: E402
    _Position,
    build_ledger,
    daily_aggregates,
)
from idxbot.analytics.campaigns import _detrend, extract_campaigns, find_pivots  # noqa: E402
from idxbot.config import load_config  # noqa: E402
from idxbot.data.broker_summary import LOT_SIZE, add_derived, normalise  # noqa: E402

cfg = load_config()


# ---------------------------------------------------------------- position ---
def test_position_weighted_average_cost():
    pos = _Position()
    pos.trade(100, 1000)          # 100 shares @ 1000
    pos.trade(100, 1200)          # 100 shares @ 1200
    assert pos.shares == 200
    assert pos.avg_cost == pytest.approx(1100)


def test_position_realises_pnl_on_sale():
    pos = _Position()
    pos.trade(100, 1000)
    pos.trade(-50, 1500)          # sell half at 1500
    assert pos.shares == 50
    assert pos.avg_cost == pytest.approx(1000)   # unchanged by a partial sale
    assert pos.realized == pytest.approx(50 * 500)


def test_position_flips_through_zero():
    pos = _Position()
    pos.trade(100, 1000)
    pos.trade(-150, 1200)         # sell more than held -> now short 50
    assert pos.shares == -50
    assert pos.realized == pytest.approx(100 * 200)
    assert pos.avg_cost == pytest.approx(1200)   # residual opens at the trade price


def test_position_closing_exactly_resets():
    pos = _Position()
    pos.trade(100, 1000)
    pos.trade(-100, 1100)
    assert pos.shares == 0
    assert pos.avg_cost == 0
    assert pos.realized == pytest.approx(100 * 100)


def test_position_ignores_degenerate_trades():
    pos = _Position()
    pos.trade(0, 1000)
    pos.trade(100, 0)
    assert pos.shares == 0


# ------------------------------------------------------------------ ledger ---
def _frame(rows):
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df


def test_build_ledger_tracks_inventory_and_cost():
    summary = _frame([
        {"date": "2026-01-05", "ticker": "TEST", "broker": "BK",
         "buy_lot": 100, "buy_val": 100 * LOT_SIZE * 1000, "buy_avg": 1000,
         "sell_lot": 0, "sell_val": 0, "sell_avg": 0, "source": "test"},
        {"date": "2026-01-06", "ticker": "TEST", "broker": "BK",
         "buy_lot": 100, "buy_val": 100 * LOT_SIZE * 1200, "buy_avg": 1200,
         "sell_lot": 0, "sell_val": 0, "sell_avg": 0, "source": "test"},
    ])
    prices = _frame([
        {"date": "2026-01-05", "open": 1000, "high": 1010, "low": 990,
         "close": 1000, "volume": 1e6},
        {"date": "2026-01-06", "open": 1200, "high": 1210, "low": 1190,
         "close": 1200, "volume": 1e6},
    ])

    ledger = build_ledger(summary, prices, ticker="TEST")
    assert len(ledger) == 2
    last = ledger.iloc[-1]
    assert last["inventory_lot"] == pytest.approx(200)
    assert last["avg_cost"] == pytest.approx(1100)
    # 200 lots = 20,000 shares marked at 1200 against a cost of 1100.
    assert last["unrealized_pnl"] == pytest.approx(200 * LOT_SIZE * 100)


def test_build_ledger_carries_inventory_across_quiet_days():
    """A broker that does not trade still holds its position."""
    summary = _frame([
        {"date": "2026-01-05", "ticker": "TEST", "broker": "BK",
         "buy_lot": 100, "buy_val": 100 * LOT_SIZE * 1000, "buy_avg": 1000,
         "sell_lot": 0, "sell_val": 0, "sell_avg": 0, "source": "test"},
        {"date": "2026-01-08", "ticker": "TEST", "broker": "BK",
         "buy_lot": 0, "buy_val": 0, "buy_avg": 0,
         "sell_lot": 50, "sell_val": 50 * LOT_SIZE * 1300, "sell_avg": 1300,
         "source": "test"},
    ])
    dates = pd.date_range("2026-01-05", "2026-01-08", freq="D")
    prices = pd.DataFrame({
        "date": dates, "open": 1000.0, "high": 1310.0, "low": 990.0,
        "close": [1000.0, 1100.0, 1200.0, 1300.0], "volume": 1e6,
    })

    ledger = build_ledger(summary, prices, ticker="TEST")
    assert len(ledger) == 4                       # reindexed onto the calendar
    assert ledger["inventory_lot"].tolist() == pytest.approx([100, 100, 100, 50])
    assert ledger.iloc[-1]["realized_pnl"] == pytest.approx(50 * LOT_SIZE * 300)


def test_build_ledger_empty_input():
    assert build_ledger(pd.DataFrame(), pd.DataFrame()).empty


# ------------------------------------------------------------- aggregates ----
def test_daily_aggregates_concentration_and_smart_dumb():
    summary = _frame([
        # BK is bulge (institutional), YP/PD are retail.
        {"date": "2026-01-05", "ticker": "TEST", "broker": "BK",
         "buy_lot": 100, "buy_val": 1000, "buy_avg": 10,
         "sell_lot": 0, "sell_val": 0, "sell_avg": 0, "source": "t"},
        {"date": "2026-01-05", "ticker": "TEST", "broker": "YP",
         "buy_lot": 0, "buy_val": 0, "buy_avg": 0,
         "sell_lot": 60, "sell_val": 600, "sell_avg": 10, "source": "t"},
        {"date": "2026-01-05", "ticker": "TEST", "broker": "PD",
         "buy_lot": 0, "buy_val": 0, "buy_avg": 0,
         "sell_lot": 40, "sell_val": 400, "sell_avg": 10, "source": "t"},
    ])
    agg = daily_aggregates(summary, cfg.brokers)
    row = agg.iloc[0]
    assert row["bulge_net_val"] == pytest.approx(1000)
    assert row["retail_net_val"] == pytest.approx(-1000)
    assert row["buyer_hhi"] == pytest.approx(1.0)      # a single net buyer
    assert row["top_buyer"] == "BK"
    assert row["smart_dumb_spread"] == pytest.approx(2000)


def test_add_derived_columns():
    df = _frame([{"date": "2026-01-05", "ticker": "T", "broker": "BK",
                  "buy_lot": 10, "buy_val": 100, "buy_avg": 1,
                  "sell_lot": 4, "sell_val": 40, "sell_avg": 1, "source": "t"}])
    out = add_derived(df)
    assert out["net_lot"].iloc[0] == 6
    assert out["net_val"].iloc[0] == 60
    assert out["total_lot"].iloc[0] == 14


# ---------------------------------------------------------------- pivots -----
def test_find_pivots_alternates():
    values = np.array([0, 1, 2, 3, 2, 1, 0, 1, 2, 3, 4, 3, 2], dtype=float)
    pivots = find_pivots(values, threshold=1.5)
    kinds = [p[2] for p in pivots]
    assert all(a != b for a, b in zip(kinds, kinds[1:])), kinds


def test_find_pivots_ignores_noise_below_threshold():
    """Sub-threshold wobble must not produce a swing worth trading.

    The seeded opening extreme means a flat series can still yield a pivot or
    two; what matters is that no pair of them spans a meaningful move.
    """
    values = np.array([0, 0.1, 0, 0.1, 0, 0.1], dtype=float)
    pivots = find_pivots(values, threshold=5.0)
    if len(pivots) >= 2:
        span = max(p[1] for p in pivots) - min(p[1] for p in pivots)
        assert span <= 0.1 + 1e-9, f"noise produced a {span} swing"


def test_find_pivots_degenerate():
    assert find_pivots(np.array([1.0, 2.0]), threshold=1.0) == []
    assert find_pivots(np.array([]), threshold=1.0) == []


def test_detrend_removes_linear_drift():
    drift = np.arange(500, dtype=float) * 10.0
    detrended = _detrend(drift, span=120)
    # A pure trend should leave a bounded residual, not a growing one.
    assert abs(detrended[-1]) < abs(drift[-1]) * 0.5


# -------------------------------------------------------------- campaigns ----
def test_extract_campaigns_finds_a_build_and_unwind():
    """A synthetic accumulate-then-distribute cycle must be segmented."""
    n = 300
    dates = pd.bdate_range("2024-01-01", periods=n)
    # Inventory: flat, build for 60 days, hold, then unwind.
    inventory = np.concatenate([
        np.zeros(60),
        np.linspace(0, 1000, 60),
        np.full(60, 1000.0),
        np.linspace(1000, 0, 60),
        np.zeros(60),
    ])
    net = np.diff(inventory, prepend=0.0)
    price = np.concatenate([
        np.full(60, 100.0),
        np.full(60, 100.0),
        np.linspace(100, 150, 60),
        np.linspace(150, 130, 60),
        np.full(60, 130.0),
    ])

    prices = pd.DataFrame({
        "date": dates, "open": price, "high": price * 1.01,
        "low": price * 0.99, "close": price, "volume": 1e7,
    })
    summary = pd.DataFrame({
        "date": dates, "ticker": "TEST", "broker": "BK",
        "buy_lot": np.maximum(net, 0.0),
        "sell_lot": np.maximum(-net, 0.0),
        "buy_avg": price, "sell_avg": price, "source": "test",
    })
    summary["buy_val"] = summary["buy_lot"] * LOT_SIZE * price
    summary["sell_val"] = summary["sell_lot"] * LOT_SIZE * price

    ledger = build_ledger(summary, prices, ticker="TEST")
    campaigns = extract_campaigns(ledger, prices, cfg)

    assert not campaigns.empty, "an obvious build/unwind cycle must be detected"
    camp = campaigns.iloc[0]
    assert camp["broker"] == "BK"
    assert camp["lots_accumulated"] > 0
    # They bought at ~100 and price peaked at ~150.
    assert camp["entry_vwap"] == pytest.approx(100, rel=0.15)
    assert camp["markup_pct"] > 0.2


def test_extract_campaigns_empty_ledger():
    assert extract_campaigns(pd.DataFrame(), pd.DataFrame(), cfg).empty


# -------------------------------------------------------------- normalise ----
def test_normalise_indonesian_headers_and_numbers():
    raw = pd.DataFrame({
        "Tanggal": ["05/01/2026"],
        "Kode Broker": ["bk"],
        "Volume Beli": ["1.234"],
        "Nilai Beli": ["123.400.000"],
        "Volume Jual": ["0"],
        "Nilai Jual": ["0"],
    })
    out = normalise(raw, ticker="BBCA", source="test", volume_unit="lot")
    assert len(out) == 1
    assert out["broker"].iloc[0] == "BK"
    assert out["buy_lot"].iloc[0] == pytest.approx(1234)
    assert out["buy_val"].iloc[0] == pytest.approx(123_400_000)
    assert out["buy_avg"].iloc[0] == pytest.approx(1000)


def test_normalise_detects_share_denominated_volume():
    """Volume in shares must be converted to lots, not taken at face value."""
    raw = pd.DataFrame({
        "date": ["2026-01-05"], "broker": ["BK"],
        "buy_lot": [123_400],                  # shares, not lots
        "buy_val": [123_400_000],              # implies 1000/share
        "sell_lot": [0], "sell_val": [0],
    })
    out = normalise(raw, ticker="BBCA", source="test", volume_unit="auto")
    assert out["buy_lot"].iloc[0] == pytest.approx(1234)
    assert out["buy_avg"].iloc[0] == pytest.approx(1000)


def test_normalise_requires_a_broker_column():
    with pytest.raises(ValueError, match="broker"):
        normalise(pd.DataFrame({"date": ["2026-01-05"], "buy_lot": [1]}),
                  ticker="X", source="test")


def test_normalise_empty():
    assert normalise(pd.DataFrame(), ticker="X").empty
