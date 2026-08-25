"""Tests for the exit rules and for the entry rule's tie handling.

The exit rules are small enough that every branch can be checked against a
hand-constructed path, which is the right way to test them: a rule that
silently never fires looks identical to buy-and-hold in aggregate, and that is
exactly the failure mode that would go unnoticed in a backtest.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from idxbot.spine import exits as X
from idxbot.spine import multiplier as MU


# ==========================================================================
# the rules
# ==========================================================================
def test_hold_takes_the_last_bar_in_window():
    p = np.array([1.1, 1.2, 1.5, 0.9])
    assert X.hold(p, 4) == pytest.approx((-0.1, 4))
    assert X.hold(p, 2) == pytest.approx((0.2, 2))


def test_hold_on_a_short_path_uses_what_exists():
    r, held = X.hold(np.array([1.3]), 252)
    assert (r, held) == pytest.approx((0.3, 1))


def test_hold_on_an_empty_path_is_nan_not_zero():
    r, held = X.hold(np.array([]))
    assert np.isnan(r) and held == 0


def test_trailing_fires_on_the_drop_from_the_peak_not_from_entry():
    # peaks at 2.0, then 1.5 is a 25% give-back but still +50% from entry
    p = np.array([1.2, 2.0, 1.5, 3.0])
    r, held = X.trailing(p, 0.20)
    assert held == 3 and r == pytest.approx(0.5)


def test_trailing_holds_when_the_drop_is_never_reached():
    p = np.array([1.1, 1.2, 1.15, 1.3])
    assert X.trailing(p, 0.50) == X.hold(p)


def test_arm_suppresses_the_trail_until_the_position_is_up():
    #   never reaches +50%, so an armed trail cannot fire at all
    p = np.array([1.2, 1.4, 0.7, 0.8])
    assert X.trailing(p, 0.20, arm=0.50) == X.hold(p)
    assert X.trailing(p, 0.20, arm=0.0)[1] == 3


def test_arm_is_measured_against_the_peak_not_the_current_bar():
    # peak 1.6 arms the trail; 1.2 is then a 25% give-back
    p = np.array([1.6, 1.2, 5.0])
    r, held = X.trailing(p, 0.20, arm=0.50)
    assert held == 2 and r == pytest.approx(0.2)


def test_hard_stop_measures_from_entry():
    p = np.array([1.5, 1.2, 0.8, 2.0])
    r, held = X.hard_stop(p, 0.20)
    assert held == 3 and r == pytest.approx(-0.2)


def test_hard_stop_ignores_a_give_back_that_stays_above_entry():
    p = np.array([3.0, 1.1, 1.2])
    assert X.hard_stop(p, 0.20) == X.hold(p)


def test_time_stop_releases_a_name_that_moved_and_cuts_one_that_did_not():
    moved = np.concatenate([np.full(21, 1.30), np.full(50, 2.0)])
    flat = np.concatenate([np.full(21, 1.02), np.full(50, 0.3)])
    assert X.time_stop(moved, 21, 0.10) == X.hold(moved)
    r, held = X.time_stop(flat, 21, 0.10)
    assert held == 21 and r == pytest.approx(0.02)


def test_time_stop_looks_at_the_running_max_not_the_bar_at_the_deadline():
    # touched +40% on day 2 and came back; it DID move, so it is released
    p = np.concatenate([[1.0, 1.4], np.full(19, 1.01), np.full(30, 3.0)])
    assert X.time_stop(p, 21, 0.10) == X.hold(p)


def test_combined_checks_the_hard_stop_before_the_trail():
    # 0.7 is both -30% from entry and -65% from the 2.0 peak; the hard stop
    # is the binding one and must be the reported reason via its threshold
    p = np.array([2.0, 0.7, 5.0])
    r, held = X.combined(p, stop=0.25, drop=0.20, arm=0.50)
    assert held == 2 and r == pytest.approx(-0.3)


def test_combined_time_stop_fires_only_at_the_deadline_bar():
    p = np.concatenate([np.full(30, 1.01), np.full(30, 4.0)])
    r, held = X.combined(p, stop=0.5, drop=0.5, arm=0.5, by=21, need=0.10)
    assert held == 21


def test_every_catalogue_rule_returns_a_finite_pair():
    p = np.cumprod(np.concatenate([np.full(120, 1.01), np.full(132, 0.99)]))
    for name, fn in X.catalogue().items():
        r, held = fn(p)
        assert np.isfinite(r), name
        assert 1 <= held <= X.HORIZON, name


def test_no_rule_can_beat_the_realised_maximum_of_the_path():
    """A rule that exits above the path's own peak is a look-ahead bug."""
    rng = np.random.default_rng(7)
    for _ in range(20):
        p = np.cumprod(1.0 + rng.normal(0.001, 0.05, 252))
        best = float(np.max(p)) - 1.0
        for name, fn in X.catalogue().items():
            assert fn(p)[0] <= best + 1e-9, name


# ==========================================================================
# scoring
# ==========================================================================
def test_apply_rule_charges_the_cost_once():
    D = X.apply_rule([np.array([1.5])], [0.02], lambda p: X.hold(p, 1))
    assert D["gross"].iloc[0] == pytest.approx(0.5)
    assert D["net"].iloc[0] == pytest.approx(0.48)


def _cohorts(n=60, seed=3):
    rng = np.random.default_rng(seed)
    out = {}
    for i in range(n):
        paths = [np.cumprod(1.0 + rng.normal(0.0008, 0.045, 252))
                 for _ in range(10)]
        out[pd.Timestamp("2010-01-01") + pd.Timedelta(days=30 * i)] = {
            "paths": paths, "costs": [0.02] * 10}
    return out


def test_score_cohorts_is_one_row_per_cohort_not_per_name():
    C = _cohorts(12)
    S = X.score_cohorts(C, lambda p: X.hold(p))
    assert len(S) == 12 and set(S["n"]) == {10}


def test_bootstrap_resamples_cohorts_and_widens_with_fewer_of_them():
    C = _cohorts(80)
    S = X.score_cohorts(C, lambda p: X.hold(p))
    wide = X.bootstrap_cohorts(S.head(15), "median", block=1)
    narrow = X.bootstrap_cohorts(S, "median", block=1)
    assert (wide[1] - wide[0]) > (narrow[1] - narrow[0])


def test_cohort_block_counts_the_cohorts_that_share_a_forward_window():
    """Monthly cohorts holding a year overlap eleven-twelfths of the time."""
    S = pd.DataFrame({"as_of": pd.date_range("2010-01-01", periods=60,
                                             freq="MS"), "m": 0.0})
    assert X.cohort_block(S, horizon=252) == 12     # 365 days / 30.5 per step
    assert X.cohort_block(S, horizon=21) == 1
    q = pd.DataFrame({"as_of": pd.date_range("2010-01-01", periods=40,
                                             freq="QS"), "m": 0.0})
    # four 91-day quarters is 364 days, just short of the 365-day horizon, so
    # the block rounds up to five — cover the overlap, never under-cover it
    assert X.cohort_block(q, horizon=252) == 5


def test_cohort_block_is_unit_safe_across_datetime_resolutions():
    """``.asi8`` is microseconds on a datetime64[us] index and nanoseconds on
    a [ns] one; a hardcoded divisor returned 11,783 for monthly cohorts."""
    base = pd.date_range("2010-01-01", periods=60, freq="MS")
    for unit in ("s", "ms", "us", "ns"):
        S = pd.DataFrame({"as_of": base.as_unit(unit), "m": 0.0})
        assert X.cohort_block(S, horizon=252) == 12, unit


def test_the_block_bootstrap_is_wider_than_the_iid_one_on_serial_data():
    """The whole reason for the block: overlapping cohorts are not 188
    observations, and an iid resample says they are."""
    rng = np.random.default_rng(1)
    x = pd.Series(rng.normal(size=200)).rolling(12).mean().dropna()
    S = pd.DataFrame({"as_of": pd.date_range("2005-01-01", periods=len(x),
                                             freq="MS"), "m": x.to_numpy()})
    iid = X.bootstrap_cohorts(S, "m", block=1)
    blk = X.bootstrap_cohorts(S, "m")
    assert (blk[1] - blk[0]) > 1.8 * (iid[1] - iid[0])


def test_the_block_bootstrap_keeps_duplicate_blocks():
    """A11: filtering duplicates with ``np.isin`` shrank every resample and
    made every interval too narrow. Each draw must be exactly n long."""
    S = pd.DataFrame({"as_of": pd.date_range("2005-01-01", periods=60,
                                             freq="MS"),
                      "m": np.arange(60, dtype=float)})
    lo, hi = X.bootstrap_cohorts(S, "m", draws=500)
    # a mean of n draws from 0..59 must stay inside the range, and a shrunken
    # resample would concentrate far too tightly around the sample mean
    assert 0 <= lo < hi <= 59
    assert (hi - lo) > 3.0


def test_bootstrap_is_deterministic_for_a_given_seed():
    S = pd.DataFrame({"as_of": pd.date_range("2005-01-01", periods=60,
                                             freq="MS"),
                      "m": np.linspace(-1, 1, 60)})
    assert X.bootstrap_cohorts(S, "m") == X.bootstrap_cohorts(S, "m")


def test_bootstrap_declines_to_answer_on_a_tiny_sample():
    S = X.score_cohorts(_cohorts(3), lambda p: X.hold(p))
    assert all(np.isnan(v) for v in X.bootstrap_cohorts(S, "median"))


def test_walk_forward_never_scores_a_cohort_it_selected_on():
    """The property the whole module exists to guarantee."""
    C = _cohorts(60)
    days = sorted(C)
    R = X.catalogue()
    W = X.walk_forward_select(C, R, min_train=24)
    assert not W.empty
    assert W["as_of"].min() >= days[24]
    # and the choice at each date must be reproducible from earlier dates only
    per = {n: X.score_cohorts(C, f).set_index("as_of") for n, f in R.items()}
    row = W.iloc[5]
    past = [d for d in days if d < row["as_of"]]
    best = max(R, key=lambda n: np.nan_to_num(
        per[n].reindex(past)["median"].mean(), nan=-np.inf))
    assert row["rule"] == best


def test_walk_forward_returns_nothing_when_training_is_impossible():
    assert X.walk_forward_select(_cohorts(10), X.catalogue(),
                                 min_train=24).empty


def _settled(n=60):
    """Monthly cohorts, each carrying the date its one-year horizon closes."""
    C = _cohorts(n)
    return {d: dict(c, settles=d + pd.Timedelta(days=366))
            for d, c in C.items()}


def test_the_purge_excludes_cohorts_still_running_at_the_decision_date():
    """The leak the first version had: monthly cohorts holding for a year
    means last month's outcome is eleven months from being known."""
    C = _settled(60)
    R = {"hold 252": lambda p: X.hold(p, 252),
         "trail 20%": lambda p: X.trailing(p, 0.20)}
    W = X.walk_forward_select(C, R, min_train=12, purge=True)
    days = sorted(C)
    assert not W.empty
    for d in W["as_of"]:
        train = [p for p in days if p < d and C[p]["settles"] <= d]
        assert len(train) >= 12
        assert all(C[p]["settles"] <= d for p in train)


def test_the_purge_costs_cohorts_and_starts_later_than_the_leaky_version():
    C = _settled(60)
    R = {"hold 252": lambda p: X.hold(p, 252),
         "trail 20%": lambda p: X.trailing(p, 0.20)}
    clean = X.walk_forward_select(C, R, min_train=12, purge=True)
    leaky = X.walk_forward_select(C, R, min_train=12, purge=False)
    assert len(clean) < len(leaky)
    assert clean["as_of"].min() > leaky["as_of"].min()


def test_a_missing_settles_falls_back_to_a_calendar_horizon():
    """Cohorts without the field must still be purged, not silently trusted."""
    C = _cohorts(60)                       # no 'settles' key anywhere
    R = {"hold 252": lambda p: X.hold(p, 252),
         "trail 20%": lambda p: X.trailing(p, 0.20)}
    assert len(X.walk_forward_select(C, R, min_train=12, purge=True)) \
        < len(X.walk_forward_select(C, R, min_train=12, purge=False))


def test_settle_date_counts_trading_sessions_not_calendar_days():
    d = pd.bdate_range("2024-01-01", periods=300)
    got = MU.settle_date(d, d[0], horizon=10)
    assert got == d[11]                    # +1 for the entry gap, +10 held


def test_settle_date_clamps_to_the_end_of_the_calendar():
    d = pd.bdate_range("2024-01-01", periods=20)
    assert MU.settle_date(d, d[0], horizon=252) == d[-1]


# ==========================================================================
# the entry rule's ties — the reproducibility failure this module documents
# ==========================================================================
def _ranked(scores):
    return pd.DataFrame({"ticker": [f"T{i:02d}" for i in range(len(scores))],
                         "p2": scores, "p5": [0.0] * len(scores)})


def test_tie_report_detects_an_arbitrary_cut():
    M = _ranked([0.9] * 4 + [0.5] * 12)
    tr = MU.tie_report(M, top_n=10)
    assert tr["arbitrary"] is True
    assert tr["n_tied_at_cut"] == 12 and tr["taken_from_tied"] == 6


def test_tie_report_is_clean_when_the_cut_is_not_in_a_tie():
    M = _ranked(list(np.linspace(0.9, 0.1, 16)))
    assert MU.tie_report(M, top_n=10)["arbitrary"] is False


def test_select_all_holds_the_entire_tied_group():
    M = _ranked([0.9] * 4 + [0.5] * 12)
    out = MU.select(M, top_n=10, tie="all")
    assert len(out) == 16
    assert out["weight"].sum() == pytest.approx(1.0)


def test_select_first_reproduces_the_old_arbitrary_cut():
    M = _ranked([0.9] * 4 + [0.5] * 12)
    assert list(MU.select(M, 10, "first")["ticker"]) == list(M["ticker"][:10])


def test_a_seeded_tie_break_always_keeps_the_names_above_the_cut():
    M = _ranked([0.9] * 4 + [0.5] * 12)
    for s in range(20):
        got = set(MU.select(M, 10, s)["ticker"])
        assert len(got) == 10
        assert set(M["ticker"][:4]) <= got


def test_different_seeds_actually_give_different_baskets():
    """If they did not, the sensitivity study would measure nothing."""
    M = _ranked([0.9] * 4 + [0.5] * 12)
    seen = {tuple(MU.select(M, 10, s)["ticker"]) for s in range(30)}
    assert len(seen) > 5


def test_select_is_a_no_op_on_a_frame_with_no_ties():
    M = _ranked(list(np.linspace(0.9, 0.1, 16)))
    for tie in ("all", "first", 0, 1):
        assert list(MU.select(M, 10, tie)["ticker"]) == list(M["ticker"][:10])


def test_select_refuses_a_frame_smaller_than_the_basket():
    assert MU.select(_ranked([0.5] * 4), top_n=10).empty


def test_edges_are_strictly_increasing_even_on_a_degenerate_column():
    e = MU.edges(np.zeros(500))
    assert np.all(np.diff(e) > 0)


# ==========================================================================
# the one-bar entry gap
# ==========================================================================
def _panel(px, ticker="AAAA", start="2024-01-01"):
    d = pd.bdate_range(start, periods=len(px))
    return pd.DataFrame({"ticker": ticker, "date": d,
                         "close": px, "adj_close": px})


def test_path_map_enters_at_the_next_bar_not_the_signal_bar():
    """The signal is computed from the close of ``day``; filling at that same
    close is an execution nobody achieves, and it is worth 100% here."""
    P = _panel([100.0, 200.0, 400.0, 400.0])
    day = P["date"].iloc[0]
    (path, _), = MU.path_map(P, day, ["AAAA"]).values()
    #      entry is bar 1 (200), so the path is 400/200 = 2.0, not 4.0
    assert path[0] == pytest.approx(2.0)


def test_path_map_skips_a_name_with_no_tradeable_future():
    P = _panel([100.0, 200.0])
    assert MU.path_map(P, P["date"].iloc[1], ["AAAA"]) == {}


def test_path_map_drops_a_non_positive_entry_rather_than_dividing_by_it():
    """The `adj_close <= 0` bars that produced 915 impossible advances in A11
    must not become an infinite return here."""
    P = _panel([100.0, 0.0, 50.0, 50.0])
    assert MU.path_map(P, P["date"].iloc[0], ["AAAA"]) == {}


def test_path_map_is_one_scan_and_agrees_with_the_list_form():
    P = pd.concat([_panel([10.0, 11.0, 13.0, 13.0], "AAAA"),
                   _panel([20.0, 22.0, 20.0, 20.0], "BBBB")])
    day = P["date"].iloc[0]
    paths, costs = MU.paths(P, day, ["BBBB", "AAAA"])
    m = MU.path_map(P, day, ["BBBB", "AAAA"])
    assert len(paths) == 2 and len(costs) == 2
    assert paths[0][0] == pytest.approx(m["BBBB"][0][0])   # caller's order
    assert paths[1][0] == pytest.approx(13.0 / 11.0)


def test_costs_are_at_least_the_fee_schedule():
    """A5's 0.56% round trip is a floor; the fraksi-harga spread adds to it."""
    P = _panel([100.0, 200.0, 400.0, 400.0])
    _, cost = MU.path_map(P, P["date"].iloc[0], ["AAAA"])["AAAA"]
    assert cost >= X.FEE - 1e-12
