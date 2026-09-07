"""H59 — sector rotation (§8).

THE TEST THIS FILE EXISTS FOR is `test_the_shares_column_never_leaves_the_map`.
The IDX-IC file carries a `shares` column frozen at 2024-07-10, and A25 records
why applying a 2024 share count to a 2010 bar is look-ahead: Indonesian rights
issues change it by up to 41x (H57). `sector_map` drops the column rather than
merely not using it, so no later edit can reach for it, and this test fails if
that guard is removed.

The second is `test_tier_arms_are_compared_on_excess_not_raw_cagr`. A first run
printed the untilted screen at +12.01% against a top-3 tilt at +6.65% — with
their INDEX benchmarks at +8.71% and +4.63%. A benchmark moving four points
between arms is proof the windows differ, so the difference was the calendar
and not the tilt. A19 records that error class three times; this is the fourth.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "scripts"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

import sector                                                     # noqa: E402
from sector import (coverage, date_block_null, ic_series,         # noqa: E402
                    rank_ic, sector_map, sector_panel, sector_tilt)

HERE = os.path.dirname(__file__)
SRC = os.path.join(HERE, os.pardir, "scripts", "sector.py")
MAP = os.path.join(HERE, os.pardir, "data", "reference",
                   "idx_classification.parquet")


def _panel(n_names=40, n_dates=600, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2015-01-05", periods=n_dates)
    out = []
    for i in range(n_names):
        px = np.exp(np.cumsum(rng.normal(0.0004, 0.02, n_dates))) * 1000
        out.append(pd.DataFrame({
            "date": dates, "ticker": f"T{i:03d}", "adj_close": px,
            "close": px, "elig": True,
            "hi52": rng.random(n_dates), "vol60": rng.random(n_dates),
            "mom12_1": rng.random(n_dates)}))
    return pd.concat(out, ignore_index=True)


def _map(n_names=40, n_sectors=8):
    return pd.DataFrame({
        "ticker": [f"T{i:03d}" for i in range(n_names)],
        "sector": [f"S{i % n_sectors}" for i in range(n_names)],
        "board": ["main"] * n_names,
        "listing_date": ["2010-01-01"] * n_names,
    })


# ------------------------------------------------------ the look-ahead guard

@pytest.mark.skipif(not os.path.exists(MAP), reason="sector map absent")
def test_the_shares_column_never_leaves_the_map():
    """A25: a 2024 share count on a 2010 bar is look-ahead. The column is
    dropped at the source so no later edit can reach for it by accident."""
    raw = pd.read_parquet(MAP)
    assert "shares" in raw.columns, "the source file changed shape"
    got = sector_map(MAP)
    assert "shares" not in got.columns


@pytest.mark.skipif(not os.path.exists(MAP), reason="sector map absent")
def test_the_map_has_the_eleven_official_sectors():
    got = sector_map(MAP)
    assert got["sector"].nunique() == 11
    assert len(got) > 900


def test_the_taxonomy_launch_date_is_recorded():
    """IDX-IC launched 2021-01-25, replacing JASICA. A label on a 2010 bar is a
    backward projection, which is a real limit even though it is not
    look-ahead, and it has to be stated with the result."""
    assert sector.IDXIC_LAUNCH == pd.Timestamp("2021-01-25")
    assert "IDX-IC ITSELF LAUNCHED" in (sector.__doc__ or "")


# ------------------------------------------------------------- the coverage

def test_coverage_separates_live_names_from_dead_ones():
    """The freeze's cost is entirely about DEAD names — a map of what was
    listed in 2024 cannot name what delisted in 2012 — so a single overall
    coverage figure would hide the only number that matters."""
    P = _panel(n_names=6, n_dates=50)
    P.loc[P["ticker"] == "T005", "date"] = P.loc[
        P["ticker"] == "T005", "date"] - pd.Timedelta(days=4000)
    c = coverage(P, _map(n_names=5))
    assert c["dead"] >= 1
    assert c["dead_mapped"] == 0.0     # T005 is not in a 5-name map
    assert c["live_mapped"] == c["live"]


# ------------------------------------------------------------ the statistic

def test_a_sector_needs_a_minimum_number_of_names():
    """A19 records the smallest cell producing the largest effect three times.
    A 'sector' of two names is a name wearing a label."""
    P = _panel(n_names=12, n_dates=400)
    M = pd.DataFrame({"ticker": [f"T{i:03d}" for i in range(12)],
                      "sector": ["BIG"] * 10 + ["TINY"] * 2})
    A = sector_panel(P, M, freq=63)
    assert set(A["sector"]) == {"BIG"}


def test_sector_returns_are_computed_within_ticker():
    """A11: never roll on a date x ticker pivot. A suspended name inserts NaN
    rows it never had and the window fails for every column at once."""
    src = open(SRC).read()
    assert 'groupby("ticker")' in src
    assert "pivot" not in src.lower() or "never roll on a pivot" in src


def test_rank_ic_is_zero_on_independent_data():
    rng = np.random.default_rng(3)
    A = pd.DataFrame({
        "date": np.repeat(pd.bdate_range("2015-01-05", periods=60), 8),
        "sector": list("ABCDEFGH") * 60,
        "mom": rng.normal(size=480), "fwd": rng.normal(size=480)})
    assert abs(rank_ic(A)) < 0.12


def test_rank_ic_finds_a_planted_relationship():
    """POSITIVE CONTROL. A statistic that cannot find an effect that IS there
    proves nothing by failing to find one — A26's sine wave, A27's planted
    Fibonacci bump, the same discipline."""
    rng = np.random.default_rng(4)
    n, s = 60, 8
    mom = rng.normal(size=n * s)
    A = pd.DataFrame({
        "date": np.repeat(pd.bdate_range("2015-01-05", periods=n), s),
        "sector": list("ABCDEFGH") * n,
        "mom": mom, "fwd": mom + rng.normal(0, 0.3, n * s)})
    assert rank_ic(A) > 0.7


# ------------------------------------------------------------------ the null

def test_the_null_moves_whole_dates_not_rows():
    """Eleven sectors on one day share the market's move almost entirely, so a
    within-date shuffle leaves most of the dependence intact and the null far
    too tight. This asserts the difference rather than the intent."""
    rng = np.random.default_rng(5)
    n, s = 80, 8
    dates = pd.bdate_range("2015-01-05", periods=n)
    day = rng.normal(0, 0.08, n)                     # a big common component
    rows = []
    for i, d in enumerate(dates):
        m = rng.normal(size=s)
        rows.append(pd.DataFrame({"date": d, "sector": list("ABCDEFGH"),
                                  "mom": m,
                                  "fwd": day[i] + 0.05 * rng.normal(size=s)}))
    A = pd.concat(rows, ignore_index=True)
    _mu, sd_date, nd = date_block_null(A, draws=80, seed=1)
    assert nd > 40 and np.isfinite(sd_date)
    within = []
    for _ in range(80):
        B = A.copy()
        B["fwd"] = B.groupby("date")["fwd"].transform(
            lambda x: x.sample(frac=1.0, random_state=rng.integers(1e6)).values)
        within.append(rank_ic(B))
    assert sd_date > 0.5 * float(np.std(within, ddof=1)), (
        "the date-block null is not materially wider than a within-date "
        "shuffle — every z computed against it would be inflated")


def test_a_constant_vector_is_skipped_not_scored_as_zero():
    """A NaN coerced to zero would pull the null toward the origin and make
    every z too large."""
    src = open(SRC).read()
    assert "np.unique(m)) < 2" in src
    assert "SKIPPED" in src


# ----------------------------------------------------------------- the arms

def test_the_random_sector_control_exists_and_picks_the_same_count():
    """A34: a control denied the treatment's own affordances is a handicap,
    not a null. Restricting to three sectors concentrates the book whatever
    chooses them, so the control must restrict to three too."""
    P = _panel(n_names=40, n_dates=5)
    M = _map(40, 8)
    day = P[P["date"] == P["date"].iloc[0]]
    top = sector_tilt(M, 3, k=5)
    rnd = sector_tilt(M, 3, k=5, random_sectors=True)
    smap = dict(zip(M["ticker"], M["sector"]))
    a = {t for t, _w in top(day)}
    b = {t for t, _w in rnd(day)}
    assert a and b
    assert len({smap[t] for t in a}) <= 3
    assert len({smap[t] for t in b}) <= 3


def test_tier_arms_are_compared_on_excess_not_raw_cagr():
    src = open(SRC).read()
    assert '"excess"' in src
    assert "scouted across all" in src
    assert "MATCHED AT 3 SECTORS" in src, (
        "the matched top-3-vs-random-3 comparison is the only cell that varies "
        "the sector CHOICE while holding concentration fixed")


def test_the_registration_survives_the_answer():
    doc = sector.__doc__ or ""
    for k in ("R1", "R2", "R3"):
        assert k in doc
    assert "PREDICTED NULL" in doc
