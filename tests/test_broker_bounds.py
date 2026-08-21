"""Tests for censored inference on a top-10 broker summary.

The centrepiece is `test_bounds_contain_the_truth`. A bound that is merely
plausible is worse than no bound, because it will be believed. So a FULL rekap
is generated with every broker's true buy and sell known, truncated to exactly
what the free source publishes - the top ten a side plus the day's totals - and
the recovered brackets are required to contain the truth they were built
without. It runs over 200 random markets so a single lucky draw cannot pass it.

Everything else here defends one specific way this can silently go wrong: a zero
in a lot column means "not in that ranking", never "traded none", and reading
those zeros as observations is precisely the bug that drove BBCA's market-wide
net - which must be zero - to -2.8 million lots.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

from idxbot.broker_bounds import (                                # noqa: E402
    bracket_frame, certain_sign, cumulative_bounds, day_bounds, merge_views,
    foreign_net, foreign_net_agreement, midpoint, naive_error,
    relative_width, settled_flow_share, settled_fraction, visibility,
    zero_sum_residual)


# --------------------------------------------------------------------------
# a market where the truth is known
# --------------------------------------------------------------------------
def full_market(n_brokers=90, total=1_000_000, seed=0, concentration=2.0):
    """A complete, balanced rekap: every lot bought by someone is sold by someone.

    Buys and sells are drawn independently from a skewed distribution and then
    rescaled to the same total, which is the one property real rekaps always
    have and the one the truncated view destroys.
    """
    rng = np.random.default_rng(seed)
    codes = [f"{chr(65 + i // 26)}{chr(65 + i % 26)}" for i in range(n_brokers)]
    buy = rng.pareto(concentration, n_brokers) + 0.01
    sell = rng.pareto(concentration, n_brokers) + 0.01
    buy = buy / buy.sum() * total
    sell = sell / sell.sum() * total
    return pd.DataFrame({"broker": codes, "buy_lot": buy, "sell_lot": sell})


def truncate(truth: pd.DataFrame, top=10, ticker="TEST",
             date="2026-08-18") -> pd.DataFrame:
    """Publish only what the free source publishes: top N a side, plus totals.

    The published total bounds BOTH sides of its own view - every lot inside a
    view was bought by someone in it and sold by someone in it - so the fixture
    uses the larger of the two sums. Setting it to the buy sum alone would make
    the fixture claim something the real footer never claims, and the soundness
    test would then fail on the fixture rather than on the code. Verified
    against real data: across 37 broker-sides the three views partition exactly,
    max difference zero.
    """
    top_buy = truth.nlargest(top, "buy_lot")["broker"]
    top_sell = truth.nlargest(top, "sell_lot")["broker"]
    keep = sorted(set(top_buy) | set(top_sell))
    v = truth[truth["broker"].isin(keep)].copy()
    v.loc[~v["broker"].isin(set(top_buy)), "buy_lot"] = 0.0
    v.loc[~v["broker"].isin(set(top_sell)), "sell_lot"] = 0.0
    v["ticker"] = ticker
    v["date"] = pd.Timestamp(date)
    v["total_lot"] = float(max(truth["buy_lot"].sum(), truth["sell_lot"].sum()))
    return v.reset_index(drop=True)


# --------------------------------------------------------------------------
# 1. soundness — the bracket must contain the truth
# --------------------------------------------------------------------------
@pytest.mark.parametrize("seed", range(40))
def test_bounds_contain_the_truth(seed):
    truth = full_market(seed=seed)
    seen = truncate(truth)
    b = day_bounds(seen).set_index("broker")
    t = truth.set_index("broker")
    for code in b.index:
        assert b.loc[code, "buy_lo"] - 1e-6 <= t.loc[code, "buy_lot"] \
            <= b.loc[code, "buy_hi"] + 1e-6, f"{seed} {code} buy"
        assert b.loc[code, "sell_lo"] - 1e-6 <= t.loc[code, "sell_lot"] \
            <= b.loc[code, "sell_hi"] + 1e-6, f"{seed} {code} sell"
        net = t.loc[code, "buy_lot"] - t.loc[code, "sell_lot"]
        assert b.loc[code, "net_lo"] - 1e-6 <= net \
            <= b.loc[code, "net_hi"] + 1e-6, f"{seed} {code} net"


@pytest.mark.parametrize("seed,conc,top",
                         [(s, c, k) for s in range(10)
                          for c in (0.8, 2.0, 5.0) for k in (5, 10, 20)])
def test_bounds_hold_at_any_concentration_or_depth(seed, conc, top):
    """Thin tables and fat ones, concentrated markets and flat ones."""
    truth = full_market(seed=seed, concentration=conc)
    b = day_bounds(truncate(truth, top=top)).set_index("broker")
    t = truth.set_index("broker")
    for code in b.index:
        net = t.loc[code, "buy_lot"] - t.loc[code, "sell_lot"]
        assert b.loc[code, "net_lo"] - 1e-6 <= net <= b.loc[code, "net_hi"] + 1e-6


def test_the_naive_reading_is_the_thing_that_breaks():
    """The bounds are not decoration: the obvious reading is genuinely wrong."""
    truth = full_market(seed=7)
    seen = truncate(truth)
    b = day_bounds(seen).set_index("broker")
    t = truth.set_index("broker")
    wrong = 0
    for code in b.index:
        net = t.loc[code, "buy_lot"] - t.loc[code, "sell_lot"]
        if abs(b.loc[code, "net_naive"] - net) > 1e-6:
            wrong += 1
    assert wrong > 0, "the truncation must actually distort something"
    assert wrong >= len(b) // 3


def test_cumulative_bounds_contain_the_cumulative_truth():
    days, truths = [], []
    for seed in range(30):
        truth = full_market(seed=100 + seed)
        truths.append(truth.set_index("broker"))
        days.append(day_bounds(truncate(truth, date=f"2026-01-{seed + 1:02d}")))
    cum = cumulative_bounds(days).set_index("broker")
    for code in cum.index:
        real = sum(float(t.loc[code, "buy_lot"] - t.loc[code, "sell_lot"])
                   for t in truths if code in t.index)
        assert cum.loc[code, "net_lo"] - 1e-6 <= real \
            <= cum.loc[code, "net_hi"] + 1e-6, code


def test_certain_sign_is_never_wrong():
    """Every direction this claims to have settled must actually be right."""
    for seed in range(25):
        truth = full_market(seed=500 + seed)
        b = day_bounds(truncate(truth))
        called = certain_sign(cumulative_bounds([b])).set_index("broker")
        t = truth.set_index("broker")
        for code in called.index:
            real = t.loc[code, "buy_lot"] - t.loc[code, "sell_lot"]
            if called.loc[code, "direction"] == "net buyer":
                assert real > -1e-6, f"{seed} {code}"
            else:
                assert real < 1e-6, f"{seed} {code}"


# --------------------------------------------------------------------------
# 2. the zero that is not a zero
# --------------------------------------------------------------------------
def test_an_unlisted_side_is_a_range_not_a_zero():
    rows = pd.DataFrame({"broker": ["AA", "BB"], "buy_lot": [500.0, 300.0],
                         "sell_lot": [0.0, 200.0], "total_lot": [2000.0] * 2})
    b = day_bounds(rows).set_index("broker")
    assert b.loc["AA", "sell_lo"] == 0.0
    assert b.loc["AA", "sell_hi"] > 0.0          # NOT pinned to zero
    assert b.loc["AA", "net_hi"] == 500.0
    assert b.loc["AA", "net_lo"] < 500.0


def test_a_listed_side_is_exact():
    rows = pd.DataFrame({"broker": ["AA"], "buy_lot": [500.0],
                         "sell_lot": [200.0], "total_lot": [2000.0]})
    b = day_bounds(rows).iloc[0]
    assert b["buy_lo"] == b["buy_hi"] == 500.0
    assert b["sell_lo"] == b["sell_hi"] == 200.0
    assert b["net_lo"] == b["net_hi"] == 300.0


def test_the_ceiling_is_the_smallest_listed_not_the_largest():
    rows = pd.DataFrame({"broker": ["AA", "BB", "CC"],
                         "buy_lot": [900.0, 500.0, 0.0],
                         "sell_lot": [0.0, 0.0, 700.0],
                         "total_lot": [5000.0] * 3})
    b = day_bounds(rows).set_index("broker")
    # CC is unlisted on the buy side; the ceiling is BB's 500, not AA's 900
    assert b.loc["CC", "buy_hi"] == 500.0


def test_the_hidden_pool_can_bind_tighter_than_the_ranking():
    """When almost everything is visible, the ceiling is the leftover volume."""
    rows = pd.DataFrame({"broker": ["AA", "BB"], "buy_lot": [900.0, 800.0],
                         "sell_lot": [0.0, 1700.0], "total_lot": [1750.0] * 2})
    b = day_bounds(rows).set_index("broker")
    # only 50 lots of selling are unaccounted for, far below the 1700 ranking
    assert b.loc["AA", "sell_hi"] == pytest.approx(50.0)


def test_visible_volume_above_the_total_is_flagged_not_absorbed():
    """The rows and the footer contradict each other; clamping would hide it."""
    rows = pd.DataFrame({"broker": ["AA"], "buy_lot": [900.0],
                         "sell_lot": [0.0], "total_lot": [100.0]})
    b = day_bounds(rows).iloc[0]
    assert not b["consistent"]
    assert b["net_lo"] <= b["net_hi"]


def test_a_contradictory_day_is_left_out_of_the_cumulative_bound():
    good = day_bounds(pd.DataFrame({"broker": ["AA"], "buy_lot": [100.0],
                                    "sell_lot": [50.0],
                                    "total_lot": [1000.0]}))
    bad = day_bounds(pd.DataFrame({"broker": ["AA"], "buy_lot": [9e9],
                                   "sell_lot": [0.0], "total_lot": [1000.0]}))
    assert bool(good["consistent"].iloc[0]) and not bool(bad["consistent"].iloc[0])
    cum = cumulative_bounds([good, bad])
    assert cum["days"].iloc[0] == 1
    assert cum["net_naive"].iloc[0] == pytest.approx(50.0)


def test_an_absent_day_widens_the_bracket_rather_than_counting_as_zero():
    """The bug a randomised test caught: absent is not the same as flat."""
    d1 = day_bounds(pd.DataFrame({"broker": ["AA", "BB"],
                                  "buy_lot": [100.0, 80.0],
                                  "sell_lot": [0.0, 180.0],
                                  "total_lot": [1000.0] * 2}))
    d2 = day_bounds(pd.DataFrame({"broker": ["BB", "CC"],
                                  "buy_lot": [90.0, 70.0],
                                  "sell_lot": [160.0, 0.0],
                                  "total_lot": [1000.0] * 2}))
    cum = cumulative_bounds([d1, d2]).set_index("broker")
    # AA is missing from day 2 entirely, so its bracket must widen on both sides
    single = cumulative_bounds([d1]).set_index("broker")
    assert cum.loc["AA", "net_hi"] > single.loc["AA", "net_hi"]
    assert cum.loc["AA", "net_lo"] < single.loc["AA", "net_lo"]
    assert cum.loc["AA", "days_seen"] == 1 and cum.loc["AA", "days"] == 2


def test_without_totals_the_ranking_still_bounds():
    rows = pd.DataFrame({"broker": ["AA", "BB"], "buy_lot": [900.0, 500.0],
                         "sell_lot": [0.0, 400.0]})
    b = day_bounds(rows).set_index("broker")
    assert np.isfinite(b.loc["AA", "sell_hi"])
    assert b.loc["AA", "sell_hi"] == 400.0


def test_a_side_nobody_was_listed_on_has_no_ranking_ceiling():
    rows = pd.DataFrame({"broker": ["AA"], "buy_lot": [900.0],
                         "sell_lot": [0.0], "total_lot": [2000.0]})
    b = day_bounds(rows).iloc[0]
    assert b["sell_hi"] == pytest.approx(2000.0)   # the pool alone bounds it


def test_an_empty_day_produces_an_empty_frame_not_a_crash():
    assert day_bounds(pd.DataFrame()).empty
    assert day_bounds(None).empty


# --------------------------------------------------------------------------
# 3. visibility and the arithmetic check
# --------------------------------------------------------------------------
def test_visibility_reports_the_share_each_side_shows():
    rows = pd.DataFrame({"broker": ["AA", "BB"], "buy_lot": [600.0, 200.0],
                         "sell_lot": [500.0, 100.0], "total_lot": [1000.0] * 2})
    v = visibility(rows)
    assert v["cover_buy"] == pytest.approx(0.8)
    assert v["cover_sell"] == pytest.approx(0.6)
    assert v["hidden_buy"] == pytest.approx(200.0)
    assert v["hidden_sell"] == pytest.approx(400.0)


def test_visibility_without_totals_says_nothing_rather_than_guessing():
    rows = pd.DataFrame({"broker": ["AA"], "buy_lot": [600.0],
                         "sell_lot": [500.0]})
    assert visibility(rows) == {}


def test_the_zero_sum_holds_identically_and_is_therefore_only_a_check():
    """It must be zero for ANY table, which is why it adds no information."""
    for seed in range(15):
        rows = truncate(full_market(seed=seed))
        assert zero_sum_residual(rows) == pytest.approx(0.0, abs=1e-6)


def test_the_zero_sum_catches_a_misparsed_column():
    rows = pd.DataFrame({"broker": ["AA"], "buy_lot": [600.0],
                         "sell_lot": [500.0], "total_lot": [1000.0]})
    good = zero_sum_residual(rows)
    bad = rows.copy()
    bad["buy_lot"] = 6e9                      # value read into the lot column
    assert abs(zero_sum_residual(bad)) > abs(good) + 1.0


# --------------------------------------------------------------------------
# 4. how much the bounds actually settle
# --------------------------------------------------------------------------
def test_more_visible_volume_settles_more_brokers():
    """The bracket narrows as the published table covers more of the day."""
    truth = full_market(seed=3, concentration=1.0)
    thin = settled_fraction(cumulative_bounds([day_bounds(truncate(truth, 5))]))
    fat = settled_fraction(cumulative_bounds([day_bounds(truncate(truth, 30))]))
    assert fat >= thin


def test_settled_fraction_of_nothing_is_undefined():
    assert np.isnan(settled_fraction(pd.DataFrame()))


def test_undetermined_brokers_are_dropped_rather_than_guessed():
    b = pd.DataFrame({"broker": ["AA", "BB", "CC"],
                      "net_lo": [10.0, -50.0, -30.0],
                      "net_hi": [90.0, -10.0, 40.0]})
    out = certain_sign(b)
    assert list(out["broker"]) == ["AA", "BB"]
    assert list(out["direction"]) == ["net buyer", "net seller"]


def test_bracket_frame_splits_by_ticker_and_day():
    rows = pd.concat([truncate(full_market(seed=1), date="2026-01-05"),
                      truncate(full_market(seed=2), date="2026-01-06")],
                     ignore_index=True)
    parts = bracket_frame(rows)
    assert len(parts) == 2
    assert all(not p.empty for p in parts)


def test_cumulative_width_grows_with_the_window():
    days = [day_bounds(truncate(full_market(seed=s))) for s in range(20)]
    short = cumulative_bounds(days[:5])["width"].median()
    long = cumulative_bounds(days)["width"].median()
    assert long > short


# --------------------------------------------------------------------------
# 5. counting flow, not names
# --------------------------------------------------------------------------
def test_flow_share_weights_by_size_not_by_headcount():
    """One settled whale outweighs a crowd of undetermined minnows."""
    b = pd.DataFrame({"broker": ["BIG", "a", "b", "c"],
                      "net_naive": [900.0, 10.0, -10.0, 5.0],
                      "net_lo": [800.0, -50.0, -50.0, -50.0],
                      "net_hi": [1000.0, 50.0, 50.0, 50.0]})
    assert settled_fraction(b) == pytest.approx(0.25)      # 1 name of 4
    assert settled_flow_share(b) > 0.95                    # but nearly all flow


def test_flow_share_of_nothing_is_undefined():
    assert np.isnan(settled_flow_share(pd.DataFrame()))
    assert np.isnan(settled_flow_share(
        pd.DataFrame({"net_naive": [0.0], "net_lo": [-1.0], "net_hi": [1.0]})))


def test_a_broker_listed_on_both_sides_every_day_has_no_uncertainty_at_all():
    """The exact case that makes this worth doing: zero censoring, zero width."""
    days = [day_bounds(pd.DataFrame({"broker": ["AA", "BB"],
                                     "buy_lot": [500.0, 300.0],
                                     "sell_lot": [200.0, 600.0],
                                     "total_lot": [1600.0] * 2}))
            for _ in range(20)]
    cum = cumulative_bounds(days).set_index("broker")
    assert cum.loc["AA", "width"] == pytest.approx(0.0)
    assert cum.loc["AA", "net_lo"] == pytest.approx(20 * 300.0)
    assert relative_width(cumulative_bounds(days)).iloc[0] == pytest.approx(0.0)


def test_relative_width_is_undefined_rather_than_infinite_at_zero_net():
    b = pd.DataFrame({"broker": ["AA"], "net_naive": [0.0],
                      "net_lo": [-5.0], "net_hi": [5.0]})
    assert np.isnan(relative_width(b).iloc[0])


def test_a_broker_seen_on_fewer_days_gets_a_wider_bracket():
    dense = [day_bounds(pd.DataFrame({"broker": ["AA", "BB"],
                                      "buy_lot": [500.0, 300.0],
                                      "sell_lot": [200.0, 600.0],
                                      "total_lot": [1600.0] * 2}))
             for _ in range(10)]
    sparse = list(dense)
    sparse[3] = day_bounds(pd.DataFrame({"broker": ["BB"], "buy_lot": [300.0],
                                         "sell_lot": [600.0],
                                         "total_lot": [1600.0]}))
    a = cumulative_bounds(dense).set_index("broker").loc["AA", "width"]
    b = cumulative_bounds(sparse).set_index("broker").loc["AA", "width"]
    assert b > a


# --------------------------------------------------------------------------
# 6. merging the all / foreign / domestic views
# --------------------------------------------------------------------------
def split_market(seed=0, n_brokers=90, total=1_000_000, foreign_share=0.7):
    """A full market where every broker's trade is split into foreign/domestic.

    The truth is known three ways over, so a merge that claims more than it may
    is caught immediately.
    """
    rng = np.random.default_rng(seed)
    truth = full_market(n_brokers=n_brokers, total=total, seed=seed)
    fb = rng.uniform(0.0, 1.0, len(truth))
    fs = rng.uniform(0.0, 1.0, len(truth))
    truth["buy_F"] = truth["buy_lot"] * fb
    truth["buy_D"] = truth["buy_lot"] - truth["buy_F"]
    truth["sell_F"] = truth["sell_lot"] * fs
    truth["sell_D"] = truth["sell_lot"] - truth["sell_F"]
    return truth


def view(truth, buy_col, sell_col, top=10):
    v = truth[["broker", buy_col, sell_col]].rename(
        columns={buy_col: "buy_lot", sell_col: "sell_lot"})
    return truncate(v, top=top)


def test_merging_the_three_views_still_contains_the_truth():
    """The tightening must never tighten past the real number."""
    for seed in range(20):
        t = split_market(seed=seed)
        m = merge_views(view(t, "buy_lot", "sell_lot"),
                        view(t, "buy_F", "sell_F"),
                        view(t, "buy_D", "sell_D")).set_index("broker")
        tt = t.set_index("broker")
        for br in m.index:
            assert m.loc[br, "buy_lo"] - 1e-5 <= tt.loc[br, "buy_lot"] \
                <= m.loc[br, "buy_hi"] + 1e-5, f"{seed} {br} buy"
            assert m.loc[br, "sell_lo"] - 1e-5 <= tt.loc[br, "sell_lot"] \
                <= m.loc[br, "sell_hi"] + 1e-5, f"{seed} {br} sell"
            net = tt.loc[br, "buy_lot"] - tt.loc[br, "sell_lot"]
            assert m.loc[br, "net_lo"] - 1e-5 <= net \
                <= m.loc[br, "net_hi"] + 1e-5, f"{seed} {br} net"


def test_merging_sees_more_brokers_than_any_single_view():
    t = split_market(seed=5)
    m = merge_views(view(t, "buy_lot", "sell_lot"),
                    view(t, "buy_F", "sell_F"), view(t, "buy_D", "sell_D"))
    one = day_bounds(view(t, "buy_lot", "sell_lot"))
    assert len(m) > len(one)


def test_merging_never_widens_a_bracket_it_already_had():
    t = split_market(seed=11)
    combined = view(t, "buy_lot", "sell_lot")
    m = merge_views(combined, view(t, "buy_F", "sell_F"),
                    view(t, "buy_D", "sell_D")).set_index("broker")
    one = day_bounds(combined).set_index("broker")
    for br in one.index:
        assert m.loc[br, "buy_hi"] <= one.loc[br, "buy_hi"] + 1e-6
        assert m.loc[br, "buy_lo"] >= one.loc[br, "buy_lo"] - 1e-6


def test_two_known_sides_pin_the_third_exactly():
    """The identity at work: listed in all and in foreign gives domestic free."""
    rows = lambda b, s: pd.DataFrame({"broker": ["AA", "BB"], "buy_lot": b,
                                      "sell_lot": s, "total_lot": [10_000.0] * 2})
    m = merge_views(rows([600.0, 400.0], [300.0, 500.0]),
                    rows([250.0, 100.0], [120.0, 200.0]),
                    rows([0.0, 400.0], [180.0, 300.0])).set_index("broker")
    # AA is unlisted in the domestic view, but all(600) - foreign(250) = 350,
    # which is under BB's 400 so the ranking permits it
    assert m.loc["AA", "buy_D_lo"] == pytest.approx(350.0)
    assert m.loc["AA", "buy_D_hi"] == pytest.approx(350.0)


def test_the_merge_narrows_the_bracket_in_practice():
    t = split_market(seed=13, foreign_share=0.6)
    combined = view(t, "buy_lot", "sell_lot")
    one = day_bounds(combined)
    m = merge_views(combined, view(t, "buy_F", "sell_F"),
                    view(t, "buy_D", "sell_D"))
    a = (one["net_hi"] - one["net_lo"]).median()
    b = (m.set_index("broker").reindex(one["broker"])["net_hi"]
         - m.set_index("broker").reindex(one["broker"])["net_lo"]).median()
    assert b <= a + 1e-9


def test_contradictory_views_are_flagged_not_reconciled_by_force():
    rows = lambda b, s: pd.DataFrame({"broker": ["AA"], "buy_lot": b,
                                      "sell_lot": s, "total_lot": [10_000.0]})
    # foreign alone claims more than the combined view allows
    m = merge_views(rows([100.0], [50.0]), rows([900.0], [50.0]),
                    rows([900.0], [50.0]))
    assert not bool(m["reconciled"].iloc[0])


def test_merged_days_accumulate_like_any_other_day():
    t = split_market(seed=17)
    m = merge_views(view(t, "buy_lot", "sell_lot"), view(t, "buy_F", "sell_F"),
                    view(t, "buy_D", "sell_D"))
    cum = cumulative_bounds([m, m])
    assert not cum.empty
    assert (cum["net_hi"] >= cum["net_lo"]).all()


def test_a_broker_in_only_one_view_is_still_finitely_bounded():
    """Absent from a view means under that view's ceiling, never unbounded.

    The first run of merge_views on real data returned inf for every broker that
    appeared in one table and not the others, which then poisoned the sort and
    the flow share. Absent is censored, not unknown.
    """
    rows = lambda names, b, s: pd.DataFrame(
        {"broker": names, "buy_lot": b, "sell_lot": s,
         "total_lot": [10_000.0] * len(names)})
    m = merge_views(rows(["AA", "BB"], [600.0, 400.0], [300.0, 500.0]),
                    rows(["AA", "CC"], [250.0, 90.0], [120.0, 70.0]),
                    rows(["AA", "DD"], [350.0, 80.0], [180.0, 60.0]))
    assert np.isfinite(m[["buy_lo", "buy_hi", "sell_lo", "sell_hi"]]).all().all()
    assert np.isfinite(m[["net_lo", "net_hi", "net_naive"]]).all().all()
    for br in ("BB", "CC", "DD"):
        assert br in set(m["broker"])


def test_a_bracket_is_never_inverted():
    """Lower above upper is not a weak claim, it is a nonsensical one.

    Real BBCA data produced a width of -10,571 lots on ZP before this guard,
    because lot counts above a million print to two or three significant figures
    and `all` then misses `foreign + domestic` by a few thousand.
    """
    rows = lambda b, s: pd.DataFrame({"broker": ["AA", "BB"], "buy_lot": b,
                                      "sell_lot": s,
                                      "total_lot": [10_000_000.0] * 2})
    # deliberately inconsistent at the rounding scale: 2.9M vs 2.4M + 0.6M
    m = merge_views(rows([2_900_000.0, 500_000.0], [1_000_000.0, 400_000.0]),
                    rows([2_400_000.0, 300_000.0], [700_000.0, 200_000.0]),
                    rows([600_000.0, 250_000.0], [350_000.0, 220_000.0]))
    assert (m["buy_hi"] >= m["buy_lo"]).all()
    assert (m["sell_hi"] >= m["sell_lo"]).all()
    assert (m["net_hi"] >= m["net_lo"]).all()


def test_wildly_inconsistent_views_are_reported_as_unreconciled():
    rows = lambda b, s: pd.DataFrame({"broker": ["AA"], "buy_lot": b,
                                      "sell_lot": s, "total_lot": [1e7]})
    m = merge_views(rows([100.0], [50.0]), rows([9e6], [50.0]),
                    rows([9e6], [50.0]))
    assert not bool(m["reconciled"].iloc[0])
    assert m["buy_hi"].iloc[0] >= m["buy_lo"].iloc[0]


def test_the_headline_number_is_the_same_merged_or_not():
    """Merging must narrow the bracket, not move the number it brackets."""
    t = split_market(seed=23)
    combined = view(t, "buy_lot", "sell_lot")
    one = day_bounds(combined).set_index("broker")["net_naive"]
    many = merge_views(combined, view(t, "buy_F", "sell_F"),
                       view(t, "buy_D", "sell_D")).set_index("broker")["net_naive"]
    shared = one.index.intersection(many.index)
    assert len(shared) > 5
    assert np.allclose(one[shared].to_numpy(), many[shared].to_numpy())


def test_naive_error_is_zero_when_the_plain_reading_is_admissible():
    b = pd.DataFrame({"net_naive": [50.0], "net_lo": [10.0], "net_hi": [90.0]})
    assert naive_error(b).iloc[0] == pytest.approx(0.0)


def test_naive_error_measures_the_distance_outside_the_bracket():
    b = pd.DataFrame({"net_naive": [5.0, 200.0], "net_lo": [10.0, 10.0],
                      "net_hi": [90.0, 90.0]})
    assert naive_error(b).tolist() == pytest.approx([5.0, 110.0])


def test_the_naive_reading_really_does_fall_outside_on_censored_data():
    """Not a theoretical worry: reading zeros as zeros is ruled out by the data."""
    t = split_market(seed=31)
    m = merge_views(view(t, "buy_lot", "sell_lot"), view(t, "buy_F", "sell_F"),
                    view(t, "buy_D", "sell_D"))
    assert (naive_error(m) > 0).any()


def test_midpoint_sits_inside_its_own_bracket():
    t = split_market(seed=37)
    m = merge_views(view(t, "buy_lot", "sell_lot"), view(t, "buy_F", "sell_F"),
                    view(t, "buy_D", "sell_D"))
    mid = midpoint(m)
    assert (mid >= m["net_lo"] - 1e-9).all() and (mid <= m["net_hi"] + 1e-9).all()


# --------------------------------------------------------------------------
# 7. the independent cross-check
# --------------------------------------------------------------------------
def test_foreign_net_is_buy_value_less_sell_value():
    rows = pd.DataFrame({"broker": ["AA", "BB"], "buy_val": [100.0, 50.0],
                         "sell_val": [30.0, 20.0]})
    assert foreign_net(rows) == pytest.approx(100.0)


def test_agreement_with_the_published_figure_is_reported_as_a_fraction():
    rows = pd.DataFrame({"broker": ["AA"], "buy_val": [1.0e11],
                         "sell_val": [5.5e10], "foreign_net_val": [4.5e10]})
    a = foreign_net_agreement(rows)
    assert a["computed"] == pytest.approx(4.5e10)
    assert a["relative_error"] == pytest.approx(0.0)
    assert a["agrees"]


def test_a_figure_that_is_half_out_does_not_agree():
    """The foreign-BROKER proxy misses the published net by about this much."""
    rows = pd.DataFrame({"broker": ["AA"], "buy_val": [1.0e11],
                         "sell_val": [7.75e10], "foreign_net_val": [4.5e10]})
    a = foreign_net_agreement(rows)
    assert a["relative_error"] == pytest.approx(0.5)
    assert not a["agrees"]


def test_agreement_without_a_published_figure_reports_only_what_it_computed():
    rows = pd.DataFrame({"broker": ["AA"], "buy_val": [10.0], "sell_val": [4.0]})
    a = foreign_net_agreement(rows)
    assert a == {"computed": pytest.approx(6.0)}


def test_agreement_of_nothing_is_empty():
    assert foreign_net_agreement(pd.DataFrame()) == {}
