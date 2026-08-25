"""Tests for the twice-daily brief.

Nearly every block below is a regression for something that was wrong first
time and produced plausible output while wrong — which is the only kind of bug
that matters in a tool whose whole purpose is to be read and believed.

    the ragged edge      breadth computed on a watchlist refresh, reported as
                         the market
    the union index      moving averages computed on a pivot, returning "0 of
                         830 names above the 200-day"
    argmax and NaN       a run anchored to a hole in the data, giving advances
                         with negative returns
    pooled cut points    extension terciles pooled across legs, creating
                         structurally impossible cells
    isin in a bootstrap  duplicate blocks silently dropped, every interval too
                         narrow
    inverted ranks       a +1 momentum sign returning the year's worst losers
    log printed as pct   a -2.61 log return displayed as "-261%"

The last block asserts the two things that are policy rather than arithmetic:
the holdout is never read, and the news-narrative gap is stated rather than
filled.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

from idxbot.report import brief as B                           # noqa: E402


def panel(prices: dict, start="2020-01-01", holdout_from=None,
          turnover=1e11) -> pd.DataFrame:
    """A minimal panel: {ticker: [closes]} on shared business days."""
    rows = []
    n = max(len(v) for v in prices.values())
    dates = pd.bdate_range(start, periods=n)
    for t, px in prices.items():
        d = dates[:len(px)]
        s = pd.Series(px, index=d, dtype=float)
        rows.append(pd.DataFrame({
            "date": d, "ticker": t, "src": "live", "close": s.to_numpy(),
            "adj_close": s.to_numpy(), "volume": 1e6, "tradeable": True,
            "log_turnover": np.log(turnover),
            "vol60": s.pct_change().rolling(20, min_periods=5).std()
                      .fillna(0.02).to_numpy(),
            "fwd20": (s.shift(-21) / s.shift(-1) - 1.0).to_numpy(),
            "fwd5": (s.shift(-6) / s.shift(-1) - 1.0).to_numpy()}))
    P = pd.concat(rows, ignore_index=True)
    P["holdout"] = (P["date"] >= pd.Timestamp(holdout_from)) \
        if holdout_from else False
    return P


# --------------------------------------------------------------------------
# THE RAGGED EDGE — a watchlist refresh must not be reported as the market
# --------------------------------------------------------------------------
def test_asof_falls_back_when_only_a_handful_of_names_refreshed():
    P = panel({f"T{i:02d}": list(range(100, 160)) for i in range(20)})
    full = pd.to_datetime(P["date"]).max()
    # one extra session, for two names only — exactly what a watchlist pull
    # leaves behind, and the panel still looks current
    extra = pd.DataFrame([{"date": full + pd.Timedelta(days=1), "ticker": t,
                           "src": "live", "close": 200.0, "adj_close": 200.0,
                           "volume": 1e6, "tradeable": True,
                           "log_turnover": np.log(1e11), "vol60": 0.02,
                           "fwd20": np.nan, "fwd5": np.nan, "holdout": False}
                          for t in ("T00", "T01")])
    P2 = pd.concat([P, extra], ignore_index=True)
    assert B.resolve_asof(P2) == full, "must not stop on the partial session"
    assert B.resolve_asof(P) == full


def test_the_partial_refresh_is_reported_not_swallowed():
    P = panel({f"T{i:02d}": list(range(100, 160)) for i in range(20)})
    full = pd.to_datetime(P["date"]).max()
    extra = pd.DataFrame([{"date": full + pd.Timedelta(days=1), "ticker": "T00",
                           "src": "live", "close": 200.0, "adj_close": 200.0,
                           "volume": 1e6, "tradeable": True,
                           "log_turnover": np.log(1e11), "vol60": 0.02,
                           "fwd20": np.nan, "fwd5": np.nan, "holdout": False}])
    P2 = pd.concat([P, extra], ignore_index=True)
    msg = B.coverage_warning(P2, full)
    assert msg and "PARTIAL REFRESH" in msg
    assert B.coverage_warning(P, full) is None


# --------------------------------------------------------------------------
# THE UNION INDEX — a gap in one name must not blank every name's average
# --------------------------------------------------------------------------
def test_a_suspended_name_does_not_blank_everyone_elses_moving_average():
    """The pivot version of this returned zero names above the 200-day on a
    panel of 830. One name missing a few sessions inserts NaN rows into the
    union index, and min_periods then fails for every column at once.
    """
    px = list(np.linspace(100, 200, 300))
    P = panel({"AAAA": px, "BBBB": px, "CCCC": px})
    # CCCC misses ten sessions in the middle, as a suspension would
    d = sorted(pd.to_datetime(P["date"]).unique())
    gap = set(d[150:160])
    P = P[~((P["ticker"] == "CCCC") & (pd.to_datetime(P["date"]).isin(gap)))]
    day = B.resolve_asof(P)
    S = B.snapshot(P, day)
    b = B.breadth(S, day)
    assert b["n_200d"] >= 2, "the unaffected names must still have a 200-day"
    assert b["above_200d"] == pytest.approx(1.0)


# --------------------------------------------------------------------------
# THE ARGMAX/NaN TRAP — the BATA regression
# --------------------------------------------------------------------------
def test_a_non_positive_price_cannot_anchor_a_run():
    """np.argmax returns the index of a NaN when one is present, so an unmasked
    scan anchors the run to a hole. The spine has 2,327 bars with a
    non-positive adjusted close and before the mask they produced 915
    'advances' with a NEGATIVE return from their own anchor — an impossibility
    under the definition.
    """
    x = np.linspace(100.0, 200.0, 300)
    x[40] = 0.0                                   # the defect
    x[41] = np.nan
    hi, lo = B._last_argext(x, 250)
    ok = hi >= 0
    assert ok.any()
    assert not np.isin(hi[ok], [40, 41]).any()
    assert not np.isin(lo[ok], [40, 41]).any()


def test_an_advance_never_has_a_negative_return_from_its_anchor():
    rng = np.random.default_rng(0)
    px = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, 400)))
    px[100] = -5.0
    P = panel({"AAAA": list(px)})
    R = B.run_state(P)
    up = R[R["leg"] == "up"]
    assert (up["run_ret"] >= -1e-9).all()
    down = R[R["leg"] == "down"]
    assert (down["run_ret"] <= 1e-9).all()


def test_a_window_that_is_mostly_holes_gets_no_run_state():
    x = np.full(400, np.nan)
    x[::5] = np.linspace(100, 200, len(x[::5]))
    hi, _ = B._last_argext(x, 250)
    assert (hi < 0).all(), "20% coverage is not a 250-session extreme"


def test_zero_volatility_does_not_become_infinite_extension():
    """164,627 panel bars carry vol60 <= 0. Dividing by it turns a motionless
    name into an enormous move."""
    P = panel({"AAAA": list(np.linspace(100, 300, 400))})
    P["vol60"] = 0.0
    R = B.run_state(P)
    assert not np.isfinite(R["run_z"]).any()


def test_a_clean_advance_anchors_at_the_low_and_has_given_nothing_back():
    """The leg runs from the OLDER extreme to the NEWER one. A stock making new
    highs has its high most recent, so the advance is anchored at the low it
    came from and `give_back` sits at zero."""
    P = panel({"AAAA": list(np.linspace(100, 300, 320))})
    last = B.run_state(P).iloc[-1]
    assert last["leg"] == "up"
    assert last["give_pct"] == pytest.approx(0.0, abs=1e-9)
    assert last["run_pct"] > 0.5


def test_a_v_shape_reads_as_a_completed_decline_with_a_large_recovery():
    """AND THAT IS THE INTENDED ANSWER, not a miss. Price is 60% off the low
    but still below the 250-session high, so the last COMPLETED swing is the
    decline and the recovery has not yet taken out the old high. The pair
    (leg, give_back) carries what one alone cannot: 'the decline is over and
    price has come a long way back'. The first version of this test asserted
    'up' and was simply wrong about the definition it was testing.

    The limitation this exposes is real and is stated in the memo rather than
    patched away: `give_back` is NOT one of the four bucket dimensions, so a
    name deep in recovery pools with one still falling.
    """
    P = panel({"AAAA": list(np.linspace(200, 100, 200))
               + list(np.linspace(100, 160, 120))})
    last = B.run_state(P).iloc[-1]
    assert last["leg"] == "down"
    assert last["run_pct"] < 0                      # below the anchor high
    assert last["give_pct"] == pytest.approx(0.60, rel=0.05)


def test_the_leg_flips_to_up_once_the_recovery_takes_out_the_old_high():
    P = panel({"AAAA": list(np.linspace(200, 100, 200))
               + list(np.linspace(100, 260, 120))})
    last = B.run_state(P).iloc[-1]
    assert last["leg"] == "up"
    assert last["run_pct"] == pytest.approx(1.60, rel=0.05)


# --------------------------------------------------------------------------
# LOG VERSUS SIMPLE — a -2.61 log return is -92.7%, not -261%
# --------------------------------------------------------------------------
def test_the_display_return_is_simple_and_the_z_score_return_is_log():
    P = panel({"AAAA": list(np.linspace(100, 1000, 400))})
    R = B.run_state(P)
    last = R.iloc[-1]
    assert last["run_pct"] == pytest.approx(np.expm1(last["run_ret"]))
    assert last["run_pct"] > last["run_ret"], "a 10x move: simple > log"
    assert (R["run_pct"] >= -1.0).all(), "no simple return below -100%"


# --------------------------------------------------------------------------
# THE BUCKETS
# --------------------------------------------------------------------------
def test_extension_cuts_are_per_leg_so_no_cell_is_impossible():
    """run_z is non-negative for advances and non-positive for declines. Pooled
    terciles therefore leave cells that can only fill with data defects, and
    the first version produced an 'advance' cell whose median extension was
    -8.2 standard deviations."""
    edges = {"run_days": np.array([100.0, 200.0]),
             "run_z_up": np.array([1.0, 2.0]),
             "run_z_down": np.array([-2.0, -1.0]),
             "ivol": np.array([1 / 3, 2 / 3])}
    b = B.bucket_of([150, 150], [0.5, -0.5], [0.5, 0.5], edges, ["up", "down"])
    assert b.iloc[0] == "up|1|0|1"
    assert b.iloc[1] == "down|1|2|1"


def test_the_stretched_label_flips_for_declines():
    """A decline's most stretched cell is its most NEGATIVE one. Printing
    'shallow' over the deepest cell would invert the reader's understanding."""
    assert "stretched" in B.describe_bucket("up|0|2|0")
    assert "shallow" in B.describe_bucket("up|0|0|0")
    assert "stretched" in B.describe_bucket("down|0|0|0")
    assert "shallow" in B.describe_bucket("down|0|2|0")


def test_an_unusable_state_gets_no_bucket_rather_than_bucket_zero():
    edges = {"run_days": np.array([100.0, 200.0]),
             "run_z_up": np.array([1.0, 2.0]),
             "run_z_down": np.array([-2.0, -1.0]),
             "ivol": np.array([1 / 3, 2 / 3])}
    b = B.bucket_of([np.nan, 150], [1.5, np.nan], [0.5, 0.5], edges,
                    ["up", "up"])
    assert b.isna().all()


# --------------------------------------------------------------------------
# THE BOOTSTRAP — duplicates must survive the resample
# --------------------------------------------------------------------------
def test_the_block_bootstrap_keeps_repeated_blocks():
    """Selecting the drawn blocks with np.isin is a set-membership test, so a
    block drawn twice contributes once. Every resample is then smaller and less
    variable than the sample, and every interval comes out too narrow — a
    bootstrap that understates uncertainty is worse than none, because it looks
    like rigour.

    The check: a sample whose every row is identical has zero sampling
    variance, so the interval must be a point. Then a sample with real spread
    must give an interval wide enough to contain the truth — which a
    deduplicating version systematically fails to do.
    """
    rng = np.random.default_rng(0)
    dates = np.repeat(np.arange(200), 5)
    flat = np.ones(len(dates))
    lo, hi = B._block_bootstrap(dates, flat, np.zeros(len(dates)), 100, rng,
                                block=21)
    assert lo == pytest.approx(1.0) and hi == pytest.approx(1.0)

    vals = rng.normal(0.01, 0.05, len(dates))
    lo, hi = B._block_bootstrap(dates, vals, np.zeros(len(dates)), 300, rng,
                                block=21)
    assert lo < vals.mean() < hi
    assert hi - lo > 0.002, "an interval this tight on n=200 dates is not real"


def test_the_bootstrap_declines_a_sample_too_short_to_block():
    rng = np.random.default_rng(0)
    d = np.arange(10)
    lo, hi = B._block_bootstrap(d, np.ones(10), np.zeros(10), 50, rng,
                                block=21)
    assert np.isnan(lo) and np.isnan(hi)


# --------------------------------------------------------------------------
# THE CANDIDATE RANKING — a +1 sign must return the high end
# --------------------------------------------------------------------------
def test_a_positive_sign_ranks_the_high_end_of_the_feature():
    """pandas ranks the largest value at pct 1.0 under ascending=True, which
    reads backwards. Got backwards first time: the mom12_1 list came back
    holding the year's biggest LOSERS under a +1 momentum sign."""
    P = panel({"WIN": [100.0] * 300, "LOSE": [100.0] * 300,
               "MID": [100.0] * 300})
    day = B.resolve_asof(P)
    m = pd.to_datetime(P["date"]) == day
    P.loc[m & (P["ticker"] == "WIN"), "mom12_1"] = 2.0
    P.loc[m & (P["ticker"] == "MID"), "mom12_1"] = 0.0
    P.loc[m & (P["ticker"] == "LOSE"), "mom12_1"] = -2.0
    C = B.candidates(P, day, n=1, features=["mom12_1"])
    assert C["ticker"].iloc[0] == "WIN"


def test_a_negative_sign_ranks_the_low_end():
    P = panel({"HI": [100.0] * 300, "LO": [100.0] * 300, "MID": [100.0] * 300})
    day = B.resolve_asof(P)
    m = pd.to_datetime(P["date"]) == day
    for t, v in (("HI", 2.0), ("MID", 0.0), ("LO", -2.0)):
        P.loc[m & (P["ticker"] == t), "volz20"] = v
    C = B.candidates(P, day, n=1, features=["volz20"])
    assert C["ticker"].iloc[0] == "LO"


def test_the_negative_control_is_never_ranked():
    """squeeze has no signed prediction — it is the control. Ranking it would
    manufacture a direction the registration deliberately withheld."""
    P = panel({"A": [100.0] * 300, "B": [100.0] * 300})
    day = B.resolve_asof(P)
    P["squeeze"] = 1.0
    assert B.candidates(P, day, features=["squeeze"]).empty


def test_every_candidate_carries_h13s_measured_result():
    """A ranked list with the result omitted reads as a recommendation no
    matter what the header says."""
    P = panel({"A": [100.0] * 300, "B": [100.0] * 300})
    day = B.resolve_asof(P)
    P["mom12_1"] = 1.0
    C = B.candidates(P, day, n=2, features=["mom12_1"])
    assert (C["h13"].str.len() > 0).all()
    assert "net-negative" in C["h13"].iloc[0]


# --------------------------------------------------------------------------
# COSTS
# --------------------------------------------------------------------------
def test_the_cost_bar_is_a_fraction_and_is_not_divided_by_price_twice():
    """A2 records the same error costing three orders of magnitude elsewhere,
    which turned losing spreads into winners. At Rp 1,000 the tick is Rp 5, so
    half a tick each way is 0.5%, on top of A5's 0.56% round trip."""
    c = B.cost_bar(1000.0, "2026-08-14")
    assert c["fee"] == pytest.approx(0.0056)
    assert c["spread"] == pytest.approx(0.005)
    assert c["total"] == pytest.approx(0.0106)


def test_a_cheap_name_costs_more_to_trade_than_a_dear_one():
    cheap = B.cost_bar(60.0, "2026-08-14")["total"]
    dear = B.cost_bar(10000.0, "2026-08-14")["total"]
    assert cheap > 2 * dear


# --------------------------------------------------------------------------
# NO LOOKAHEAD, AND THE HOLDOUT
# --------------------------------------------------------------------------
def test_upto_strict_excludes_the_decision_bar():
    P = panel({"A": list(range(100, 200))})
    day = B.resolve_asof(P)
    assert (pd.to_datetime(B.upto(P, day)["date"]) <= day).all()
    assert (pd.to_datetime(B.upto(P, day, strict=True)["date"]) < day).all()


def test_the_factorisation_never_sees_the_day_it_describes():
    """A component fitted on today's returns explains today by construction."""
    rng = np.random.default_rng(1)
    px = {f"T{i:02d}": list(100 * np.exp(np.cumsum(
        rng.normal(0, 0.02, 400)))) for i in range(80)}
    P = panel(px)
    day = B.resolve_asof(P)
    a = B.comovement(P, day, n_pc=2, min_names=40)
    # move every price on the LAST bar only; the loadings must not budge
    P2 = P.copy()
    m = pd.to_datetime(P2["date"]) == day
    P2.loc[m, "adj_close"] = P2.loc[m, "adj_close"] * 1.5
    b = B.comovement(P2, day, n_pc=2, min_names=40)
    assert not a.empty and not b.empty
    assert a["var_share"].tolist() == pytest.approx(b["var_share"].tolist())


def test_the_reference_sample_never_reads_the_holdout():
    """§11 reserves the last 24 months to be spent once. The brief runs twice a
    day, so a reference table that read them would spend them immediately."""
    rng = np.random.default_rng(2)
    px = {f"T{i:02d}": list(100 * np.exp(np.cumsum(
        rng.normal(0, 0.02, 600)))) for i in range(12)}
    P = panel(px, holdout_from="2021-06-01")
    R = B.run_state(P)
    D, _ = B.conditional_frame(P, R, k=20, liquid_pct=0.0)
    if not D.empty:
        assert pd.to_datetime(D["date"]).max() < pd.Timestamp("2021-06-01")


def test_the_news_caveat_no_longer_claims_news_is_unavailable():
    """THIS TEST USED TO ASSERT THE OPPOSITE. It pinned a `narrative_gap()`
    that printed "no news source is available", which was a true statement
    about the repo turned into a false one about the world — §3 listed no news
    source because nobody had looked. Eight endpoints were then tested and five
    answered. The assertion is inverted deliberately so the old claim cannot
    quietly come back.
    """
    c = B.news_caveat()
    assert "not available" not in c.lower()
    assert "MAY NOT ENTER ANY STATISTIC" in c
    assert "point-in-time" in c, "the real limit is the archive, not access"


# --------------------------------------------------------------------------
# THE AUTO-REJECTION BAND MUST FOLLOW THE BOARD
# --------------------------------------------------------------------------
def test_a_thin_board_name_is_banded_at_ten_percent_not_thirty_five():
    """Papan Pemantauan Khusus and Akselerasi trade a flat +/-10%; the main
    ladder is +35%/-15%. An earlier version took the default board="main" for
    every ticker, so a name locked limit-up at +10% did not register at all.
    On the live cross-section 41 of 818 names sit on the thin board.
    """
    from idxbot.spine import reference
    day = pd.Timestamp("2026-08-24")
    # a penny stock whose six-month average is under Rp 51 -> thin board
    assert reference.infer_board(day, 30.0, 30.0) in reference.THIN_BOARDS
    assert reference.infer_board(day, 3000.0, 3000.0) in reference.MAIN_BOARDS
    assert day >= reference.WATCHLIST_START, "the rule must exist on this date"
    thin_up, _ = reference.auto_rejection(30.0, day, "watchlist")
    main_up, _ = reference.auto_rejection(30.0, day, "main")
    assert thin_up == pytest.approx(0.10)
    assert main_up > 3 * thin_up, "the two ladders must genuinely differ"


def test_limit_moves_counts_a_thin_board_lock_the_main_ladder_would_miss():
    # THE PANEL MUST BE DATED AFTER 2023-06-12. Papan Pemantauan Khusus did not
    # exist before then, so `infer_board` correctly answers "unknown" for a
    # sub-Rp-51 name in 2020 rather than inventing a board — which is the
    # point-in-time discipline working, and which broke an earlier version of
    # this test that started the panel in 2020.
    px = [30.0] * 300
    P = panel({"THIN": px + [33.0]}, start="2024-01-01")
    day = B.resolve_asof(P)
    S = B.snapshot(P, day)
    L = B.limit_moves(S, day)
    assert L["thin"] >= 1, "the six-month rule must place this name"
    assert L["ara"] >= 1, "+10% on the thin board IS a lock-up"


def test_the_snapshot_carries_the_six_month_average_the_board_rule_needs():
    P = panel({"AAAA": list(np.linspace(100, 200, 300))})
    day = B.resolve_asof(P)
    S = B.snapshot(P, day)
    assert "avg_price_6m" in S
    assert np.isfinite(S["avg_price_6m"].iloc[0])


# --------------------------------------------------------------------------
# THE INDEX MUST HONOUR `tradeable`
# --------------------------------------------------------------------------
def test_an_untradeable_bar_never_enters_the_index():
    """§5: a bar you could not have traded is not a bar you may label, and an
    index is exactly a claim about what you could have held.

    THE REAL CASE. KOPI printed a +10,915,284% one-day return on 2007-04-20 —
    a vendor defect the quality gate had already flagged untradeable. The
    equal-weighted index inherited it anyway and compounded to 354,721,616x
    over 2000-2024 against 1,128.5x once the flag is honoured.
    """
    P = panel({"GOOD": [100.0] * 50, "DEFECT": [100.0] * 50})
    m = (P["ticker"] == "DEFECT") & (pd.to_datetime(P["date"])
                                     == pd.to_datetime(P["date"]).max())
    P.loc[m, ["close", "adj_close"]] = 1e8          # the impossible print
    P["tradeable"] = ~m                              # which the gate caught
    idx = B.index_series(P, "equal")
    assert idx.iloc[-1] == pytest.approx(1.0, rel=1e-6), \
        "a flagged bar must not move the index at all"
    dirty = B.index_series(P, "equal", tradeable_only=False)
    assert dirty.iloc[-1] > 100, "and without the guard it obliterates it"


def test_the_turnover_weighted_index_honours_the_flag_too():
    P = panel({"GOOD": [100.0] * 50, "DEFECT": [100.0] * 50})
    m = (P["ticker"] == "DEFECT") & (pd.to_datetime(P["date"])
                                     == pd.to_datetime(P["date"]).max())
    P.loc[m, ["close", "adj_close"]] = 1e8
    P["tradeable"] = ~m
    idx = B.index_series(P, "turnover")
    assert idx.iloc[-1] == pytest.approx(1.0, rel=1e-6)
