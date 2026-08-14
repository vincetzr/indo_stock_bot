"""Bandarmology arithmetic.

Every number here is hand-computable from the fixture, because the whole point
of the module is that the conventions are explicit. A test that just checks
"returns a DataFrame" would defeat it.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

from idxbot.bandarmology import (  # noqa: E402
    LOT_SIZE,
    bandar_profiles,
    bandar_score,
    concentration,
    foreign_flow,
    foreign_streak,
    render,
)
from idxbot.config import BrokerRegistry  # noqa: E402

REGISTRY = BrokerRegistry.from_yaml("config/brokers.yaml")


def _summary(rows):
    """rows: (date, ticker, broker, buy_lot, buy_val, sell_lot, sell_val)."""
    return pd.DataFrame(rows, columns=["date", "ticker", "broker",
                                       "buy_lot", "buy_val",
                                       "sell_lot", "sell_val"]).assign(
        date=lambda d: pd.to_datetime(d["date"]))


# --------------------------------------------------------------------------
# net foreign
# --------------------------------------------------------------------------

def test_net_foreign_is_foreign_buy_minus_foreign_sell():
    # BK is bulge/foreign; PD (Indo Premier) is local retail. Only BK counts.
    s = _summary([
        ("2026-01-05", "BBCA", "BK", 100, 1_000_000, 20, 200_000),
        ("2026-01-05", "BBCA", "PD", 500, 5_000_000, 10, 100_000),
    ])
    f = foreign_flow(s, REGISTRY)
    assert f["net_foreign_val"].iloc[0] == pytest.approx(800_000)
    assert f["net_foreign_lot"].iloc[0] == pytest.approx(80)


def test_foreign_flow_is_the_running_total_not_the_daily_figure():
    s = _summary([
        ("2026-01-05", "BBCA", "BK", 100, 1_000_000, 0, 0),
        ("2026-01-06", "BBCA", "BK", 0, 0, 40, 400_000),
        ("2026-01-07", "BBCA", "BK", 50, 500_000, 0, 0),
    ])
    f = foreign_flow(s, REGISTRY).sort_values("date")
    assert f["net_foreign_val"].tolist() == pytest.approx([1e6, -4e5, 5e5])
    assert f["foreign_flow"].tolist() == pytest.approx([1e6, 6e5, 1.1e6])


def test_gross_foreign_buying_can_coexist_with_negative_net():
    """The distinction most screens blur."""
    s = _summary([("2026-01-05", "BBCA", "BK", 100, 1_000_000, 200, 2_000_000)])
    f = foreign_flow(s, REGISTRY)
    assert f["foreign_buy_val"].iloc[0] == pytest.approx(1_000_000)
    assert f["net_foreign_val"].iloc[0] == pytest.approx(-1_000_000)


def test_foreign_flow_is_computed_per_ticker_not_pooled():
    s = _summary([
        ("2026-01-05", "BBCA", "BK", 100, 1_000_000, 0, 0),
        ("2026-01-05", "BBRI", "BK", 0, 0, 100, 900_000),
        ("2026-01-06", "BBCA", "BK", 100, 1_000_000, 0, 0),
    ])
    f = foreign_flow(s, REGISTRY)
    bbca = f[f.ticker == "BBCA"].sort_values("date")
    assert bbca["foreign_flow"].tolist() == pytest.approx([1e6, 2e6])
    assert f[f.ticker == "BBRI"]["foreign_flow"].iloc[0] == pytest.approx(-9e5)


def test_empty_summary_yields_empty_flow():
    assert foreign_flow(pd.DataFrame(), REGISTRY).empty


# --------------------------------------------------------------------------
# streak
# --------------------------------------------------------------------------

def test_streak_counts_consecutive_one_way_sessions():
    s = _summary([
        ("2026-01-05", "BBCA", "BK", 0, 0, 10, 100_000),   # sell
        ("2026-01-06", "BBCA", "BK", 10, 100_000, 0, 0),   # buy
        ("2026-01-07", "BBCA", "BK", 10, 100_000, 0, 0),   # buy
        ("2026-01-08", "BBCA", "BK", 10, 100_000, 0, 0),   # buy
    ])
    st = foreign_streak(foreign_flow(s, REGISTRY), "BBCA")
    assert st["streak_days"] == 3
    assert st["direction"] == "accumulation"


def test_streak_reports_distribution_when_foreigners_are_selling():
    s = _summary([
        ("2026-01-06", "BBCA", "BK", 0, 0, 10, 100_000),
        ("2026-01-07", "BBCA", "BK", 0, 0, 10, 100_000),
    ])
    st = foreign_streak(foreign_flow(s, REGISTRY), "BBCA")
    assert st["direction"] == "distribution"
    assert st["streak_days"] == 2


# --------------------------------------------------------------------------
# bandar profiles
# --------------------------------------------------------------------------

def test_average_buy_price_is_value_over_shares_not_over_lots():
    """buy_val / (buy_lot * 100). Forgetting the lot size is off by 100x."""
    s = _summary([("2026-01-05", "BBCA", "BK", 100, 60_000_000, 0, 0)])
    p = bandar_profiles(s, REGISTRY, "BBCA")[0]
    assert p.avg_buy == pytest.approx(60_000_000 / (100 * LOT_SIZE))
    assert p.avg_buy == pytest.approx(6000.0)


def test_a_broker_that_never_bought_has_no_average_buy():
    s = _summary([("2026-01-05", "BBCA", "BK", 0, 0, 50, 30_000_000)])
    p = bandar_profiles(s, REGISTRY, "BBCA")[0]
    assert np.isnan(p.avg_buy)
    assert p.avg_sell == pytest.approx(6000.0)


def test_profiles_aggregate_across_the_window_and_rank_by_net_value():
    s = _summary([
        ("2026-01-05", "BBCA", "BK", 100, 1_000_000, 0, 0),
        ("2026-01-06", "BBCA", "BK", 100, 1_100_000, 0, 0),
        ("2026-01-05", "BBCA", "AK", 300, 3_000_000, 0, 0),
    ])
    ps = bandar_profiles(s, REGISTRY, "BBCA")
    assert [p.broker for p in ps] == ["AK", "BK"]
    bk = next(p for p in ps if p.broker == "BK")
    assert bk.net_val == pytest.approx(2_100_000)
    assert bk.days_active == 2 and bk.days_net_buy == 2


def test_window_truncates_to_the_most_recent_sessions():
    rows = [(f"2026-01-{d:02d}", "BBCA", "BK", 10, 100_000, 0, 0)
            for d in range(1, 11)]
    ps = bandar_profiles(_summary(rows), REGISTRY, "BBCA", window=3)
    assert ps[0].days_active == 3
    assert ps[0].net_val == pytest.approx(300_000)


def test_profiles_for_an_absent_ticker_are_empty():
    s = _summary([("2026-01-05", "BBCA", "BK", 10, 100_000, 0, 0)])
    assert bandar_profiles(s, REGISTRY, "TLKM") == []


# --------------------------------------------------------------------------
# concentration
# --------------------------------------------------------------------------

def test_a_single_net_buyer_gives_an_hhi_of_one():
    s = _summary([
        ("2026-01-05", "BBCA", "BK", 100, 1_000_000, 0, 0),
        ("2026-01-05", "BBCA", "PD", 0, 0, 100, 1_000_000),
    ])
    c = concentration(bandar_profiles(s, REGISTRY, "BBCA"))
    assert c["hhi"] == pytest.approx(1.0)
    assert c["buyer_count"] == 1
    assert c["lead_broker"] == "BK"


def test_four_equal_buyers_give_an_hhi_of_one_quarter():
    rows = [("2026-01-05", "BBCA", b, 100, 1_000_000, 0, 0)
            for b in ("BK", "AK", "KZ", "MS")]
    c = concentration(bandar_profiles(_summary(rows), REGISTRY, "BBCA"))
    assert c["hhi"] == pytest.approx(0.25)
    assert c["buyer_count"] == 4


def test_concentration_ignores_sellers_so_crossing_does_not_inflate_it():
    """A broker crossing a block with itself nets to zero and must not count."""
    rows = [
        ("2026-01-05", "BBCA", "BK", 100, 1_000_000, 0, 0),
        ("2026-01-05", "BBCA", "AK", 900, 9_000_000, 900, 9_000_000),  # crossed
    ]
    c = concentration(bandar_profiles(_summary(rows), REGISTRY, "BBCA"))
    assert c["buyer_count"] == 1
    assert c["lead_broker"] == "BK"


def test_concentration_with_no_net_buyers_is_zero_not_an_error():
    s = _summary([("2026-01-05", "BBCA", "BK", 0, 0, 100, 1_000_000)])
    c = concentration(bandar_profiles(s, REGISTRY, "BBCA"))
    assert c["hhi"] == 0.0 and c["buyer_count"] == 0


# --------------------------------------------------------------------------
# score and rendering
# --------------------------------------------------------------------------

def test_score_is_higher_when_institutions_dominate_the_buying():
    inst = _summary([("2026-01-05", "BBCA", "BK", 100, 1_000_000, 0, 0)])
    retail = _summary([("2026-01-05", "BBCA", "PD", 100, 1_000_000, 0, 0)])
    a = bandar_score(bandar_profiles(inst, REGISTRY, "BBCA"))
    b = bandar_score(bandar_profiles(retail, REGISTRY, "BBCA"))
    assert a["score"] > b["score"]


def test_score_always_declares_itself_unvalidated():
    s = _summary([("2026-01-05", "BBCA", "BK", 100, 1_000_000, 0, 0)])
    assert bandar_score(bandar_profiles(s, REGISTRY, "BBCA"))["validated"] is False


def test_render_without_data_says_so_rather_than_printing_zeros():
    text = render("BBCA", [])
    assert "No broker summary available" in text
    assert "0" not in text.split("No broker summary")[0].split("\n")[-2]


def test_render_carries_the_not_validated_warning():
    s = _summary([("2026-01-05", "BBCA", "BK", 100, 1_000_000, 0, 0)])
    text = render("BBCA", bandar_profiles(s, REGISTRY, "BBCA"))
    assert "NOT VALIDATED" in text
    assert "refuted" in text


# --------------------------------------------------------------------------
# foreign-owned is not foreign money
# --------------------------------------------------------------------------

from idxbot.bandarmology import (  # noqa: E402
    foreign_basis_comparison,
    is_foreign,
)


def test_yp_is_foreign_owned_but_not_institutional_foreign():
    """Mirae Asset: Korean-owned, and Indonesia's largest RETAIL broker.

    Including it in 'net foreign' does not shade the number, it dominates it -
    YP is routinely the highest-volume member of the day.
    """
    assert REGISTRY.get("YP").foreign is True
    assert REGISTRY.get("YP").tier == "retail"
    assert is_foreign("YP", REGISTRY, "ownership") is True
    assert is_foreign("YP", REGISTRY, "institutional") is False


def test_bulge_desks_count_under_both_conventions():
    for code in ("BK", "AK", "KZ"):
        assert is_foreign(code, REGISTRY, "ownership") is True
        assert is_foreign(code, REGISTRY, "institutional") is True


def test_a_domestic_broker_is_never_foreign():
    for code in ("PD", "CC", "NI"):
        assert is_foreign(code, REGISTRY, "ownership") is False
        assert is_foreign(code, REGISTRY, "institutional") is False


def test_the_two_conventions_disagree_when_retail_routes_through_mirae():
    s = _summary([
        ("2026-01-05", "BBCA", "BK", 100, 1_000_000, 0, 0),        # real foreign
        ("2026-01-05", "BBCA", "YP", 900, 9_000_000, 0, 0),        # retail via YP
    ])
    inst = foreign_flow(s, REGISTRY, "institutional")
    own = foreign_flow(s, REGISTRY, "ownership")
    assert inst["net_foreign_val"].iloc[0] == pytest.approx(1_000_000)
    assert own["net_foreign_val"].iloc[0] == pytest.approx(10_000_000)
    # The "foreign inflow" is 10x larger under the loose convention, and 90% of
    # the difference is domestic retail.
    assert own["net_foreign_val"].iloc[0] == 10 * inst["net_foreign_val"].iloc[0]


def test_comparison_isolates_the_retail_component():
    s = _summary([
        ("2026-01-05", "BBCA", "BK", 100, 1_000_000, 0, 0),
        ("2026-01-05", "BBCA", "YP", 900, 9_000_000, 0, 0),
    ])
    cmp = foreign_basis_comparison(s, REGISTRY)
    assert cmp["retail_via_foreign_broker"].iloc[0] == pytest.approx(9_000_000)


def test_an_unknown_basis_is_rejected_rather_than_silently_defaulted():
    with pytest.raises(ValueError, match="foreign_basis"):
        is_foreign("BK", REGISTRY, "whatever")


def test_the_chosen_basis_is_recorded_on_the_result():
    s = _summary([("2026-01-05", "BBCA", "BK", 100, 1_000_000, 0, 0)])
    assert foreign_flow(s, REGISTRY, "ownership").attrs["foreign_basis"] == "ownership"


# --------------------------------------------------------------------------
# bandar detection: loyalty over time, not size on one day
# --------------------------------------------------------------------------

from idxbot.bandarmology import (  # noqa: E402
    BANDAR_MIN_DAYS,
    detect_bandar,
    render_bandar,
)


def _campaign(broker, days, val=1_000_000, ticker="GOTO", start=1):
    """One broker buying steadily for `days` sessions."""
    return [(f"2026-03-{d:02d}", ticker, broker, val / 1000, val, 0, 0)
            for d in range(start, start + days)]


def _bars(days, first=100.0, last=102.0, start=1):
    px = np.linspace(first, last, days)
    return pd.DataFrame({"date": pd.to_datetime(
        [f"2026-03-{d:02d}" for d in range(start, start + days)]), "close": px})


def test_one_broker_buying_every_day_is_a_bandar_footprint():
    s = _summary(_campaign("CC", 20))
    sig = detect_bandar(s, REGISTRY, "GOTO", bars=_bars(20))
    lead = sig[0]
    assert lead.broker == "CC"
    assert lead.loyalty == pytest.approx(1.0)
    assert lead.presence == pytest.approx(1.0)
    assert lead.qualifies is True


def test_a_single_huge_block_is_NOT_a_bandar_footprint():
    """The distinction the whole detector exists for.

    One enormous day outweighs everyone on net value, but a campaign is a
    campaign because it repeats.
    """
    rows = _campaign("CC", 1, val=500_000_000)          # one giant block
    rows += [r for b in ("PD", "NI", "DH") for r in _campaign(b, 20, val=1_000_000)]
    sig = detect_bandar(_summary(rows), REGISTRY, "GOTO", bars=_bars(20))
    cc = next(s for s in sig if s.broker == "CC")
    assert cc.net_val > sum(s.net_val for s in sig if s.broker != "CC")
    assert cc.days_present == 1
    assert cc.qualifies is False            # biggest buyer, not a bandar


def test_buying_spread_across_many_brokers_matches_nobody():
    rows = [r for b in ("CC", "NI", "DH", "PD", "AT", "YJ")
            for r in _campaign(b, 20, val=1_000_000)]
    sig = detect_bandar(_summary(rows), REGISTRY, "GOTO", bars=_bars(20))
    assert not any(s.qualifies for s in sig)
    assert max(s.loyalty for s in sig) < 0.35


def test_loyalty_measures_share_of_net_buying():
    rows = _campaign("CC", 20, val=3_000_000) + _campaign("NI", 20, val=1_000_000)
    sig = detect_bandar(_summary(rows), REGISTRY, "GOTO", bars=_bars(20))
    cc = next(s for s in sig if s.broker == "CC")
    assert cc.loyalty == pytest.approx(0.75)


def test_a_short_campaign_is_rejected_however_loyal():
    s = _summary(_campaign("CC", BANDAR_MIN_DAYS - 1))
    sig = detect_bandar(s, REGISTRY, "GOTO", bars=_bars(BANDAR_MIN_DAYS - 1))
    assert sig[0].loyalty == pytest.approx(1.0)
    assert sig[0].qualifies is False


def test_stealth_is_high_when_price_barely_moves_while_absorbing():
    quiet = detect_bandar(_summary(_campaign("CC", 20)), REGISTRY, "GOTO",
                          bars=_bars(20, 100.0, 101.0))
    marked = detect_bandar(_summary(_campaign("CC", 20)), REGISTRY, "GOTO",
                           bars=_bars(20, 100.0, 160.0))
    assert quiet[0].stealth > marked[0].stealth


def test_stealth_stays_zero_when_no_bars_are_supplied_rather_than_guessing():
    sig = detect_bandar(_summary(_campaign("CC", 20)), REGISTRY, "GOTO", bars=None)
    assert sig[0].stealth == 0.0


def test_detection_is_empty_without_data():
    assert detect_bandar(pd.DataFrame(), REGISTRY, "GOTO") == []
    assert detect_bandar(_summary(_campaign("CC", 5)), REGISTRY, "NOPE") == []


def test_render_names_the_matching_broker_and_its_average_price():
    sig = detect_bandar(_summary(_campaign("CC", 20)), REGISTRY, "GOTO",
                        bars=_bars(20))
    text = render_bandar("GOTO", sig)
    assert "CC" in text and "took 100% of net buying" in text
    assert "not a forecast" in text


def test_render_says_so_when_nothing_matches():
    rows = [r for b in ("CC", "NI", "DH", "PD") for r in _campaign(b, 20)]
    text = render_bandar("GOTO", detect_bandar(_summary(rows), REGISTRY, "GOTO"))
    assert "No broker matches" in text
