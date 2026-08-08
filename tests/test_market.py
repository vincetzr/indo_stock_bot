"""IDX microstructure maths: ticks, lots, auto-rejection, costs, sizing."""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

from idxbot.config import load_config  # noqa: E402
from idxbot.market import (  # noqa: E402
    Costs,
    ara_pct,
    auto_rejection_bounds,
    days_to_reach,
    format_idr,
    position_size,
    round_to_tick,
    shares_to_lots,
    tick_size,
)

cfg = load_config()


@pytest.mark.parametrize("price,expected", [
    (50, 1), (199, 1),
    (200, 2), (499, 2),
    (500, 5), (1999, 5),
    (2000, 10), (4999, 10),
    (5000, 25), (12345, 25),
])
def test_tick_bands(price, expected):
    assert tick_size(price, cfg) == expected


def test_round_to_tick_lands_on_grid():
    # Every rounded price must be an exact multiple of its own band's tick.
    for price in (57, 213, 888, 3333, 6377, 21118):
        for mode in ("nearest", "up", "down"):
            snapped = round_to_tick(price, cfg, mode)
            assert snapped % tick_size(snapped, cfg) == 0, (price, mode, snapped)


def test_round_to_tick_direction():
    assert round_to_tick(6377, cfg, "down") <= 6377
    assert round_to_tick(6377, cfg, "up") >= 6377


def test_round_to_tick_across_band_boundary():
    """A price just above 5000 must snap onto the 25 grid, not the 10 grid."""
    assert round_to_tick(5010, cfg, "down") % 25 == 0


def test_tick_size_handles_garbage():
    assert tick_size(0, cfg) == 1
    assert tick_size(-5, cfg) == 1
    assert tick_size(float("nan"), cfg) == 1
    assert round_to_tick(float("nan"), cfg) == 0.0


def test_ara_bands():
    assert ara_pct(150, cfg) == pytest.approx(0.35)
    assert ara_pct(1000, cfg) == pytest.approx(0.25)
    assert ara_pct(9000, cfg) == pytest.approx(0.20)


def test_auto_rejection_bounds_bracket_the_price():
    low, high = auto_rejection_bounds(1000, cfg)
    assert low < 1000 < high
    assert low == pytest.approx(850, abs=5)     # 15% ARB
    assert high == pytest.approx(1250, abs=5)   # 25% ARA


def test_days_to_reach_compounds_at_the_limit():
    assert days_to_reach(1000, 1000, cfg) == 0
    assert days_to_reach(1000, 900, cfg) == 0      # already below
    assert days_to_reach(1000, 1200, cfg) == 1     # inside one ARA
    assert days_to_reach(1000, 2000, cfg) >= 3     # needs several limit-ups


def test_shares_to_lots_rounds_down():
    assert shares_to_lots(999, cfg) == 9
    assert shares_to_lots(100, cfg) == 1
    assert shares_to_lots(99, cfg) == 0


def test_position_size_never_exceeds_the_risk_budget():
    equity, entry, stop = 100_000_000, 1000, 900
    lots, notional, risk = position_size(equity, entry, stop, 0.01, 0.20, cfg)
    assert lots > 0
    assert risk <= equity * 0.01 + 1e-6      # rounding down can only reduce risk
    assert notional <= equity * 0.20 + 1e-6
    assert notional == pytest.approx(lots * 100 * entry)


def test_position_size_respects_the_exposure_cap():
    # A tight stop would otherwise imply an enormous position.
    equity, entry, stop = 100_000_000, 1000, 999
    _lots, notional, _risk = position_size(equity, entry, stop, 0.01, 0.20, cfg)
    assert notional <= equity * 0.20 + 1e-6


def test_position_size_rejects_inverted_stop():
    assert position_size(1e8, 1000, 1000, 0.01, 0.2, cfg) == (0, 0.0, 0.0)
    assert position_size(1e8, 1000, 1100, 0.01, 0.2, cfg) == (0, 0.0, 0.0)


def test_breakeven_is_above_entry():
    costs = Costs.from_config(cfg)
    entry = 1000
    breakeven = costs.breakeven_price(entry, cfg)
    assert breakeven > entry
    assert costs.net_return(entry, breakeven) >= 0


def test_net_return_is_negative_at_entry():
    costs = Costs.from_config(cfg)
    assert costs.net_return(1000, 1000) < 0     # fees make a flat exit a loss


def test_format_idr_scales():
    assert "T" in format_idr(2.5e12)
    assert "M" in format_idr(3.1e9)
    assert "jt" in format_idr(4.2e6)
    assert format_idr(float("nan")) == "-"
    assert format_idr(-1e9).startswith("-")


# ------------------------------------------------------------- tradeability --
def _bars(n=30, volume=1_000_000.0, price=1000.0, flat=False):
    import pandas as pd
    df = pd.DataFrame({
        "date": pd.bdate_range("2026-01-01", periods=n),
        "open": price, "close": price,
        "high": price if flat else price * 1.01,
        "low": price if flat else price * 0.99,
        "volume": volume,
    })
    return df


def test_tradeability_accepts_a_normal_name():
    from idxbot.market import tradeability
    assert tradeability(_bars())["tradeable"] is True


def test_tradeability_rejects_suspended_names():
    """The WIKA case: 20 zero-volume days at a frozen price."""
    from idxbot.market import tradeability
    result = tradeability(_bars(volume=0.0, flat=True))
    assert result["tradeable"] is False
    assert any("suspended" in r for r in result["reasons"])


def test_tradeability_rejects_zero_range_bars():
    from idxbot.market import tradeability
    result = tradeability(_bars(flat=True))
    assert result["tradeable"] is False
    assert any("zero range" in r for r in result["reasons"])


def test_tradeability_rejects_thin_turnover():
    from idxbot.market import tradeability
    result = tradeability(_bars(volume=10.0), min_value_traded=1e9)
    assert result["tradeable"] is False
    assert any("turnover" in r for r in result["reasons"])


def test_tradeability_needs_history():
    from idxbot.market import tradeability
    assert tradeability(_bars(n=3))["tradeable"] is False
    assert tradeability(None)["tradeable"] is False


# --------------------------------------------------------- pasted ingestion --
def test_parse_pasted_side_by_side():
    """The layout every Indonesian platform uses: buyers | sellers."""
    from idxbot.ingest import parse_pasted
    text = (
        "#\tBroker\tB.Lot\tB.Val\tB.Avg\t|\tBroker\tS.Lot\tS.Val\tS.Avg\n"
        "1\tBK\t152.340\t124.518.000.000\t8.174\t|\tYP\t152.340\t124.500.000.000\t8.172\n"
    )
    frame, report = parse_pasted(text, "BBCA", "2026-08-06")
    assert not frame.empty
    assert report["layout"].startswith("side-by-side")
    assert frame["buy_lot"].sum() == pytest.approx(frame["sell_lot"].sum())
    bk = frame[frame["broker"] == "BK"].iloc[0]
    assert bk["buy_lot"] == pytest.approx(152340)
    assert bk["buy_avg"] == pytest.approx(8174, rel=0.01)


def test_parse_pasted_rejects_garbage():
    from idxbot.ingest import parse_pasted
    frame, _ = parse_pasted("no broker codes here at all", "BBCA", "2026-08-06")
    assert frame.empty


def test_parse_pasted_empty():
    from idxbot.ingest import parse_pasted
    frame, _ = parse_pasted("", "BBCA", "2026-08-06")
    assert frame.empty


def test_ingest_number_formats():
    """Indonesian thousands, suffixes, and negatives in parentheses."""
    from idxbot.ingest import _to_number
    assert _to_number("1.234.567") == pytest.approx(1234567)
    assert _to_number("1.234.567,89") == pytest.approx(1234567.89)
    assert _to_number("1,234,567.89") == pytest.approx(1234567.89)
    assert _to_number("450 rb") == pytest.approx(450e3)
    assert _to_number("1,2 M") == pytest.approx(1.2e6)
    assert _to_number("2.5B") == pytest.approx(2.5e9)
    assert _to_number("(1.000)") == pytest.approx(-1000)
    assert _to_number("-") == 0.0
