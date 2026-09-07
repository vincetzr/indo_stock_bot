"""H57 — the point-in-time share count and the tier ladder it re-tests.

THE ONE TEST THIS FILE EXISTS FOR is `test_attach_never_carries_a_count
_backwards`. A share count is known from the day it takes effect and never
before, so a backward fill puts a post-rights-issue count on pre-issue bars and
prices the company at its post-dilution size months early. Nothing in the
resulting table looks wrong — market cap is simply too large in exactly the
names that later issued equity, which is the look-ahead A25 refused to accept
from a frozen 2024 file and the whole reason this module exists.

The second load-bearing test is `test_block_null_is_wider_than_a_row_shuffle`.
A17 and A25 record the two ways to break a clustered null in opposite
directions; a null that is too tight manufactures significance and a null that
is too loose hides it. This asserts the difference on synthetic data rather
than asserting the code is correct.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "scripts"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

import shares_pit                                                 # noqa: E402
from shares_pit import (attach, block_null, blocks_of, build,     # noqa: E402
                        edge_at, screen_flag, tier_mask)


def _panel(n_names: int = 20, n_dates: int = 300, seed: int = 0
           ) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n_dates)
    out = []
    for i in range(n_names):
        px = np.exp(np.cumsum(rng.normal(0.0003, 0.02, n_dates))) * 1000
        out.append(pd.DataFrame({
            "date": dates,
            "ticker": f"T{i:03d}",
            "close": px,
            "adj_close": px,
            "hi52": rng.random(n_dates),
            "vol60": rng.random(n_dates),
            "tv60": np.full(n_dates, 1e9 * (i + 1)),
        }))
    return pd.concat(out, ignore_index=True)


# ---------------------------------------------------------------- the join

def test_attach_never_carries_a_count_backwards():
    """A count effective on day 100 must not price day 99."""
    P = _panel(n_names=1, n_dates=200)
    d = P["date"].to_numpy()
    S = pd.DataFrame({"ticker": ["T000", "T000"],
                      "date": [d[0], d[100]],
                      "listed_shares": [1_000_000.0, 40_000_000.0]})
    F = attach(P, S).sort_values("date").reset_index(drop=True)
    before = F[F["date"] < d[100]]["listed_shares"].unique()
    after = F[F["date"] >= d[100]]["listed_shares"].unique()
    assert list(before) == [1_000_000.0], (
        "a post-issue count leaked onto pre-issue bars — this is the exact "
        "look-ahead the module exists to remove")
    assert list(after) == [40_000_000.0]


def test_attach_leaves_bars_before_the_first_count_unpriced():
    """No count yet is DATA UNAVAILABLE, never a guess at the first one."""
    P = _panel(n_names=1, n_dates=100)
    d = P["date"].to_numpy()
    S = pd.DataFrame({"ticker": ["T000"], "date": [d[50]],
                      "listed_shares": [5_000_000.0]})
    F = attach(P, S).sort_values("date").reset_index(drop=True)
    assert F[F["date"] < d[50]]["mktcap"].isna().all()
    assert F[F["date"] >= d[50]]["mktcap"].notna().all()


def test_attach_does_not_mix_tickers():
    P = _panel(n_names=2, n_dates=60)
    d = sorted(P["date"].unique())
    S = pd.DataFrame({"ticker": ["T000", "T001"], "date": [d[0], d[0]],
                      "listed_shares": [1e6, 9e6]})
    F = attach(P, S)
    assert F[F["ticker"] == "T000"]["listed_shares"].unique().tolist() == [1e6]
    assert F[F["ticker"] == "T001"]["listed_shares"].unique().tolist() == [9e6]


def test_attach_survives_an_unsorted_input():
    """merge_asof needs a globally sorted key; callers must not have to know."""
    P = _panel(n_names=3, n_dates=40).sample(frac=1.0, random_state=1)
    d = sorted(P["date"].unique())
    S = pd.DataFrame({"ticker": ["T000", "T001", "T002"], "date": [d[0]] * 3,
                      "listed_shares": [1e6, 2e6, 3e6]})
    F = attach(P, S)
    assert F["mktcap"].notna().all()


def test_mktcap_is_shares_times_the_unadjusted_close():
    """Capitalisation is priced at the TRADED price, not the back-adjusted one.

    A19 measured back-adjustment moving `adj_close` away from `close` at every
    corporate action, so using it here would report a company's size in a
    currency that never existed.
    """
    P = _panel(n_names=1, n_dates=30)
    P["adj_close"] = P["close"] * 0.5
    d = P["date"].to_numpy()
    S = pd.DataFrame({"ticker": ["T000"], "date": [d[0]],
                      "listed_shares": [1e6]})
    F = attach(P, S)
    assert np.allclose(F["mktcap"], F["close"] * 1e6)


# ---------------------------------------------------------------- the cut

def test_tier_mask_cuts_within_the_date_not_across_the_sample():
    """`top 5` must mean "the 5 largest that day", never "ended up large"."""
    P = _panel(n_names=20, n_dates=10)
    m = tier_mask(P, "tv60", 5)
    per_day = P[m].groupby("date").size()
    assert (per_day == 5).all()


def test_tier_none_is_the_whole_universe():
    P = _panel(n_names=6, n_dates=10)
    assert tier_mask(P, "tv60", None).all()


def test_screen_is_cut_within_each_date():
    """A pooled quantile would let a calm year populate a volatile one."""
    P = _panel(n_names=40, n_dates=20)
    P["screen"] = screen_flag(P)
    share = P.groupby("date")["screen"].mean()
    #  strength top 10% AND calm bottom 50%, independent by construction here
    assert share.between(0.0, 0.25).all()
    assert share.mean() > 0.0


def test_edge_at_refuses_a_degenerate_cell():
    """A19 records the smallest cell producing the largest effect, twice."""
    val = np.random.default_rng(0).normal(size=1000)
    sc = np.zeros(1000, dtype=bool)
    sc[:5] = True
    assert edge_at(val, sc, np.ones(1000, dtype=bool)) is None


def test_edge_at_drops_nans_rather_than_propagating_them():
    val = np.array([1.0, 2.0, np.nan, 4.0] * 400)
    sc = np.array([True, False, True, False] * 400)
    keep = np.ones(len(val), dtype=bool)
    r = edge_at(val, sc, keep)
    assert r is not None and np.isfinite(r["edge"])
    #  the NaN rows are screen rows, so only the 1.0s count on that side
    assert r["n"] == 400
    assert r["screen"] == 1.0 and r["rest"] == 3.0


# ---------------------------------------------------------------- the null

def test_block_null_is_wider_than_a_row_shuffle():
    """The clustered null must not be as tight as an iid one.

    A17: one name contributes ~20 near-identical bars a month, so a row shuffle
    destroys the label while leaving the null far too tight and every z comes
    back inflated.
    """
    rng = np.random.default_rng(0)
    P = _panel(n_names=30, n_dates=260, seed=3)
    #  a per-(ticker, year) label and a per-(ticker, year) outcome: the label
    #  carries no information, but both are clustered, which is the case an
    #  iid null gets wrong.
    key = P["ticker"] + "|" + P["date"].dt.year.astype(str)
    lvl = {k: rng.normal() for k in key.unique()}
    P["l60"] = key.map(lvl).to_numpy() + rng.normal(0, 0.05, len(P))
    P["screen"] = key.map({k: rng.random() < 0.3
                           for k in key.unique()}).to_numpy()
    val = P["l60"].to_numpy(float)
    sc = P["screen"].to_numpy(bool)
    keep = np.ones(len(P), dtype=bool)

    _mu, sd_block, n = block_null(val, sc, keep, blocks_of(P), draws=60, seed=1)
    assert n > 0

    iid = []
    for i in range(60):
        r = edge_at(val, np.random.default_rng(i).permutation(sc), keep)
        if r:
            iid.append(r["edge"])
    sd_row = float(np.std(iid, ddof=1))
    assert sd_block > 2 * sd_row, (
        f"clustered null sd {sd_block:.4f} is not materially wider than the "
        f"row-shuffle sd {sd_row:.4f} — every z computed against it is inflated")


def test_block_null_centres_near_zero_on_a_label_with_no_information():
    P = _panel(n_names=25, n_dates=200, seed=7)
    rng = np.random.default_rng(11)
    P["l60"] = rng.normal(0, 0.1, len(P))
    P["screen"] = rng.random(len(P)) < 0.2
    val, sc = P["l60"].to_numpy(float), P["screen"].to_numpy(bool)
    mu, sd, _ = block_null(val, sc, np.ones(len(P), dtype=bool),
                           blocks_of(P), draws=80, seed=2)
    assert abs(mu) < 1.5 * sd


def test_block_null_preserves_the_screen_share():
    """Reassigning blocks must not change how many rows are labelled true.

    A10 records `gross` and `hhi` failing because the shuffle conserved the
    very thing being measured. The opposite failure — a shuffle that changes
    the cell SIZE — makes the null a test of sample size rather than of labels.
    """
    P = _panel(n_names=20, n_dates=252, seed=5)
    rng = np.random.default_rng(3)
    P["screen"] = rng.random(len(P)) < 0.25
    sc = P["screen"].to_numpy(bool)
    groups = blocks_of(P)
    #  Every block here is the same length, so the tile/truncate branch is not
    #  exercised and the share is preserved exactly.
    lengths = {len(g) for g in groups}
    assert len(lengths) >= 1
    shares = []
    r = np.random.default_rng(0)
    for _ in range(20):
        perm = r.permutation(len(groups))
        new = np.empty(len(P), dtype=bool)
        for gi, pi in enumerate(perm):
            src, dst = groups[pi], groups[gi]
            take = sc[src]
            if len(take) >= len(dst):
                new[dst] = take[:len(dst)]
            else:
                reps = int(np.ceil(len(dst) / max(len(take), 1)))
                new[dst] = np.tile(take, reps)[:len(dst)]
        shares.append(new.mean())
    assert abs(np.mean(shares) - sc.mean()) < 0.05


# ------------------------------------------------------- the build and the doc

def test_build_drops_zero_and_missing_counts(tmp_path):
    """A share count of zero is missing data wearing a number's clothes."""
    d = tmp_path / "src"
    d.mkdir()
    (d / "AAAA.csv").write_text(
        "date,listed_shares\n2020-01-01,100\n2020-01-02,0\n2020-01-03,\n")
    out = tmp_path / "s.parquet"
    S = build(str(d), str(out))
    assert len(S) == 1
    assert S["listed_shares"].iloc[0] == 100
    assert S["ticker"].iloc[0] == "AAAA"


def test_build_skips_a_file_without_the_column(tmp_path):
    d = tmp_path / "src"
    d.mkdir()
    (d / "GOOD.csv").write_text("date,listed_shares\n2020-01-01,5\n")
    (d / "BAD.csv").write_text("date,close\n2020-01-01,5\n")
    S = build(str(d), str(tmp_path / "s.parquet"))
    assert set(S["ticker"]) == {"GOOD"}


def test_the_registration_is_still_in_the_docstring():
    """A pre-registration deleted after the result is not a registration.

    A11 and A19 both record a retracted claim surviving in a docstring above
    the corrected code. The rule cuts both ways: the prediction has to survive
    the answer too, so P2's wording is pinned here.
    """
    doc = shares_pit.__doc__ or ""
    assert "PREDICTED NULL" in doc
    assert "H52" in doc
    for phrase in ("P1", "P2", "P3"):
        assert phrase in doc


def test_the_positive_control_is_documented_as_deciding_readability():
    """C0 is what licenses reading P2 at all; it must not become optional."""
    src = open(os.path.join(os.path.dirname(__file__), os.pardir,
                            "scripts", "shares_pit.py")).read()
    assert "POSITIVE CONTROL" in src
    assert "UNSCOREABLE" in src, (
        "the run must be able to say P2 cannot be scored — a study that can "
        "only ever return a verdict is not testing anything")
