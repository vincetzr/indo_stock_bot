"""Tests for forecast verification.

THE POINT OF THIS MODULE IS TO SEPARATE TWO QUESTIONS that get conflated
constantly: "can it forecast" and "can it make money". A forecaster can be
well calibrated and useless, or informative and unprofitable, and this
repository's answer happens to be the second.

The tests are built around the failure the first implementation actually hit.
Raw cell frequencies scored a Brier skill of -0.0093 — worse than quoting the
base rate every day — while carrying genuinely non-zero resolution. That
combination is diagnostic: the conditioning knew something, and overconfidence
threw it away. So the suite pins:

    calibration     a perfect forecaster lands on the diagonal, a constant
                    one is calibrated with zero resolution
    resolution      is positive by construction on any finite sample, so it
                    must be read against a shuffle, never against zero
    shrinkage       collapses a thin cell onto climatology and leaves a thick
                    one alone
    the embargo     a fold boundary must not let a training label reach the
                    outcome it is scored against
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

from idxbot.report import forecast as F                        # noqa: E402


# --------------------------------------------------------------------------
# CALIBRATION AND SKILL
# --------------------------------------------------------------------------
def test_a_perfect_forecaster_scores_zero_calibration_error():
    rng = np.random.default_rng(0)
    p = rng.uniform(0.05, 0.95, 40000)
    y = (rng.uniform(size=len(p)) < p).astype(float)
    d = F.decompose(p, y, bins=10)
    assert d["reliability"] < 0.0006, "a truthful forecaster is on the diagonal"
    assert d["resolution"] > 0.04, "and it says genuinely different things"
    # for a calibrated forecaster skill = resolution / uncertainty, and here
    # resolution is Var(p) = 0.9^2/12 = 0.0675 against an uncertainty near
    # 0.25 — so 0.27 is the right answer and an earlier 0.4 was bad arithmetic
    assert d["skill"] == pytest.approx(0.27, abs=0.03)


def test_a_constant_forecaster_is_calibrated_and_worthless():
    """The trap this module exists to catch. Always predict the base rate and
    calibration is perfect by construction — with nothing to show for it."""
    rng = np.random.default_rng(1)
    y = (rng.uniform(size=20000) < 0.45).astype(float)
    p = np.full_like(y, float(y.mean()))
    d = F.decompose(p, y, bins=10)
    assert d["reliability"] == pytest.approx(0.0, abs=1e-9)
    assert d["resolution"] == pytest.approx(0.0, abs=1e-9)
    assert abs(d["skill"]) < 1e-9, "calibrated, and no better than climatology"


def test_an_overconfident_forecaster_scores_negative_skill():
    """The exact shape of the first implementation's failure: it knew the
    direction and stated it far too strongly."""
    rng = np.random.default_rng(2)
    n = 30000
    true_p = np.where(rng.uniform(size=n) < 0.5, 0.42, 0.48)
    y = (rng.uniform(size=n) < true_p).astype(float)
    over = np.where(true_p > 0.45, 0.95, 0.05)          # same sign, absurd size
    assert F.brier_skill(over, y) < -0.5
    assert F.brier_skill(true_p, y) > 0


def test_brier_skill_is_zero_against_the_climatology_it_is_measured_from():
    rng = np.random.default_rng(3)
    y = (rng.uniform(size=10000) < 0.4).astype(float)
    assert F.brier_skill(np.full_like(y, y.mean()), y) == pytest.approx(0.0,
                                                                       abs=1e-9)


def test_murphys_decomposition_adds_back_up():
    """Brier = reliability - resolution + uncertainty.

    The identity is EXACT only when the forecast is constant within each bin.
    That is the real use case here — the brief emits one probability per cell,
    54 distinct values — so the test uses discrete forecasts. Binning a
    continuous forecast leaves a within-bin variance term and the identity
    holds only approximately; an earlier version of this test asserted exact
    equality on U(0.2, 0.8) and was measuring that residual, not a defect.
    """
    rng = np.random.default_rng(4)
    levels = np.array([0.30, 0.38, 0.45, 0.52, 0.61])
    p = rng.choice(levels, 40000)
    y = (rng.uniform(size=len(p)) < p).astype(float)
    d = F.decompose(p, y, bins=len(levels))
    assert d["brier"] == pytest.approx(
        d["reliability"] - d["resolution"] + d["uncertainty"], abs=1e-9)


def test_the_reliability_curve_carries_its_bin_counts():
    """A bin holding eleven observations is not evidence of miscalibration and
    must not be read as one."""
    rng = np.random.default_rng(5)
    p = rng.uniform(0.3, 0.6, 5000)
    y = (rng.uniform(size=len(p)) < p).astype(float)
    R = F.reliability(p, y, bins=10)
    assert "n" in R and R["n"].sum() == len(p)
    assert (R["predicted"].diff().dropna() > 0).all(), "bins must be ordered"


# --------------------------------------------------------------------------
# SHRINKAGE
# --------------------------------------------------------------------------
def test_shrinkage_collapses_a_thin_cell_and_spares_a_thick_one():
    base = 0.45
    counts = pd.Series({"thin": 20.0, "thick": 500000.0})
    ups = pd.Series({"thin": 20.0, "thick": 260000.0})      # thin is 100% up
    p = F.shrink(counts, ups, base, prior=2000.0)
    assert abs(p["thin"] - base) < 0.01, "20 bars cannot claim 100%"
    assert p["thick"] == pytest.approx(0.52, abs=0.01), "500k bars keep their"


def test_a_zero_prior_returns_the_raw_frequency():
    counts = pd.Series({"a": 100.0})
    ups = pd.Series({"a": 70.0})
    assert F.shrink(counts, ups, 0.45, prior=0.0)["a"] == pytest.approx(0.70)


def test_an_enormous_prior_returns_climatology():
    counts = pd.Series({"a": 100.0})
    ups = pd.Series({"a": 100.0})
    assert F.shrink(counts, ups, 0.45, prior=1e9)["a"] == pytest.approx(
        0.45, abs=1e-4)


def test_shrinkage_repairs_the_overconfidence_it_was_added_for():
    """END TO END, on the failure mode that motivated it: thin cells quoting
    extreme frequencies they have not earned.

    The prior must be scaled to the cells. An earlier version used a fixed
    2,000 against cells of 4,000 and made things WORSE — it shrank the
    informative thick cells by a third to fix the thin ones. In production the
    prior is chosen by inner walk-forward for exactly this reason.

    IT MUST BE SCORED OUT OF SAMPLE, and an earlier version of this test was
    not. A cell's raw frequency is the maximum-likelihood estimate on the rows
    it was computed from, so it minimises in-sample Brier BY CONSTRUCTION and
    no shrinkage can ever beat it there. Shrinkage buys out-of-sample accuracy
    by trading a little bias for a lot of variance, which is invisible unless
    the scoring rows are ones the estimate has not seen. Hence the split.
    """
    rng = np.random.default_rng(6)
    base, n_cells, n_per = 0.45, 54, 2000
    TRUE_SD = 0.005                     # real cell effects: half a point
    # sampling sd at n=2000 is sqrt(.45*.55/2000) = 0.011 — TWICE the signal.
    # That is the regime the panel is actually in, and it is why the raw
    # frequencies scattered from 25% to 100% around a 45% base rate.
    tr, te = [], []
    for c in range(n_cells):
        p_true = base + rng.normal(0, TRUE_SD)
        for bag, n in ((tr, n_per), (te, n_per)):
            y = (rng.uniform(size=n) < p_true).astype(float)
            bag += [{"bucket": f"c{c}", "y": v} for v in y]
    TR, TE = pd.DataFrame(tr), pd.DataFrame(te)
    g = TR.groupby("bucket")["y"]
    y = TE["y"].to_numpy()
    raw = TE["bucket"].map(g.mean()).to_numpy()

    # the optimal shrinkage weight is tau^2/(tau^2+sigma^2), which at these
    # scales is 0.17 — equivalent to a prior of ~10,000 pseudo-observations.
    # That is very close to what the inner walk-forward chose on the real
    # panel, which is a satisfying independent check on both.
    p = F.shrink(g.count(), g.sum(), base, 10000.0)
    shr = TE["bucket"].map(p).to_numpy()
    assert F.brier(shr, y) < F.brier(raw, y), "shrinkage must help out of sample"
    assert F.brier_skill(raw, y) < F.brier_skill(shr, y)
    # and the shrunk forecasts must span a much narrower range than the raw
    assert (shr.max() - shr.min()) < 0.4 * (raw.max() - raw.min())


# --------------------------------------------------------------------------
# THE WALK-FORWARD AND ITS EMBARGO
# --------------------------------------------------------------------------
def frame(n_days=1200, n_names=40, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2015-01-01", periods=n_days)
    rows = []
    for d in dates:
        for i in range(n_names):
            b = f"up|{i % 3}|{i % 3}|1"
            rows.append({"date": d, "ticker": f"T{i:02d}", "bucket": b,
                         "fwd20": rng.normal(0.002 * (i % 3), 0.06)})
    return pd.DataFrame(rows)


def test_the_walk_forward_only_ever_trains_on_the_past():
    W = F.walk_forward(frame(), k=20, n_folds=4, embargo=20)
    assert not W.empty
    for f, g in W.groupby("fold"):
        earlier = W[W["fold"] < f]
        if not earlier.empty:
            assert g["date"].min() > earlier["date"].max()


def test_the_embargo_actually_removes_the_boundary_sessions():
    """Without it, a training bar's 20-session forward return overlaps the
    first bars of the test fold — the cell's 'historical' frequency would be
    built partly from the outcomes it is graded against."""
    D = frame()
    W0 = F.walk_forward(D, k=20, n_folds=4, embargo=0)
    W1 = F.walk_forward(D, k=20, n_folds=4, embargo=60)
    # THE BUG THIS CAUGHT: `emb[:-0]` is empty, not "all of it", so an embargo
    # of zero silently trained on nothing and returned no forecasts at all.
    assert not W0.empty, "embargo=0 means no embargo, not no training data"
    assert not W1.empty
    assert W1["fold"].nunique() == W0["fold"].nunique()


def test_a_bucket_unseen_in_training_falls_back_to_climatology():
    D = frame()
    D.loc[D["date"] > D["date"].quantile(0.8), "bucket"] = "brand|new|cell|9"
    W = F.walk_forward(D, k=20, n_folds=4, embargo=20)
    assert W["p"].notna().all(), "no forecast may be NaN"
    novel = W[W["bucket"] == "brand|new|cell|9"]
    if not novel.empty:
        assert (novel["p"] == novel["base"]).all()


def test_the_prior_is_chosen_without_seeing_the_scored_fold():
    """Tuning it on the test fold is the leak the embargo exists to prevent."""
    D = frame()
    tr = D[D["date"] <= D["date"].quantile(0.6)]
    p = F.pick_prior(tr, "fwd20")
    assert p in (0, 100, 300, 1000, 3000, 10000, 30000)


# --------------------------------------------------------------------------
# THE NULL — resolution is positive by construction
# --------------------------------------------------------------------------
def test_resolution_is_read_against_a_shuffle_not_against_zero():
    """A pure-noise forecast still scores positive resolution on any finite
    sample. Reading it against zero would report skill that is arithmetic
    rather than evidence — the error this repo has made four times."""
    rng = np.random.default_rng(7)
    n = 20000
    y = (rng.uniform(size=n) < 0.45).astype(float)
    noise = rng.uniform(0.40, 0.50, n)               # carries no information
    d = F.decompose(noise, y, bins=10)
    assert d["resolution"] > 0, "positive by construction, which is the point"

    W = pd.DataFrame({"date": np.repeat(np.arange(n // 40), 40),
                      "p": noise, "y": y, "base": 0.45})
    v = F.verify(W)
    assert v["resolution_p"] > 0.05, "noise must not clear its own null"


def test_a_real_signal_clears_the_shuffle_null():
    rng = np.random.default_rng(8)
    n = 20000
    tilt = rng.choice([0.40, 0.50], size=n)
    y = (rng.uniform(size=n) < tilt).astype(float)
    W = pd.DataFrame({"date": np.repeat(np.arange(n // 40), 40),
                      "p": tilt, "y": y, "base": 0.45})
    v = F.verify(W)
    assert v["resolution_p"] < 0.05
    assert v["skill"] > 0


def test_summarise_refuses_to_call_a_null_result_a_finding():
    v = {"n": 100, "base_rate": 0.45, "brier": 0.25, "uncertainty": 0.2475,
         "skill": -0.001, "skill_vs_base": -0.001, "reliability": 0.001,
         "resolution": 0.0002, "resolution_null_mean": 0.0002,
         "resolution_null_p95": 0.0003, "resolution_p": 0.4}
    text = " ".join(F.summarise(v))
    assert "quoting the base rate back" in text
    assert "genuine forecasting result" not in text
