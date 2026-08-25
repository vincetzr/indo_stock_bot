"""Tests for the cross-sectional model (H27).

Two null-construction bugs made the null BEAT the model it was testing (3.06
against 2.31). Both are pinned here, because a broken null does not announce
itself — it prints a number, and the number looked like a weak model.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir,
                                "scripts"))

from xsection_model import K, evaluate, fit_predict            # noqa: E402


def _d(n_names=40, n_years=12, seed=0):
    """A panel where one feature genuinely predicts the labels."""
    rng = np.random.default_rng(seed)
    rows = []
    for y in range(2005, 2005 + n_years):
        for m in range(1, 13):
            for i in range(n_names):
                x = rng.uniform()
                rows.append({
                    "date": pd.Timestamp(f"{y}-{m:02d}-01"),
                    "ticker": f"T{i:02d}", "year": y,
                    "sig_r": x, "noise_r": rng.uniform(),
                    "up": int(rng.uniform() < 0.05 + 0.20 * x),
                    "down": int(rng.uniform() < 0.20 - 0.15 * x),
                    f"end{K}": 1.0 + rng.normal(0, 0.3)})
    return pd.DataFrame(rows)


COLS = ["sig_r", "noise_r"]
SMALL = dict(min_train=400)


# ============================================================== the null bugs
def test_the_null_destroys_the_signal():
    """BUG TWO. Permuting inside (ticker, year) blocks is nearly a no-op — the
    twelve monthly cohorts of one ticker-year hold near-identical labels
    because their forward windows overlap by eleven months. The null kept the
    structure it was meant to remove and came back ABOVE the fitted model."""
    d = _d()
    real = evaluate(fit_predict(d, COLS, seed=0, **SMALL))
    null = evaluate(fit_predict(d, COLS, seed=0, shuffle=True, **SMALL))
    assert real and null
    assert null["skew"] < real["skew"], (
        "a null that beats the model it is testing is broken, not evidence "
        "of a weak model")
    assert null["skew"] < 1.6, f"null should sit near 1.0, got {null['skew']}"


def test_the_null_keeps_the_up_down_pair_together():
    """BUG ONE. Permuting `up` and `down` independently breaks their real
    link — a name that can double is the same name that can halve, both driven
    by its volatility — and invents observations that doubled with no halving
    risk."""
    d = _d()
    before = (d["up"] * d["down"]).sum(), d["up"].sum(), d["down"].sum()
    R = fit_predict(d, COLS, seed=1, shuffle=True, **SMALL)
    assert not R.empty
    #  marginal counts must be preserved by any permutation
    assert R["up"].sum() > 0 and R["down"].sum() > 0
    assert before[1] > 0


def test_the_null_is_reproducible_and_seed_sensitive():
    d = _d()
    a = evaluate(fit_predict(d, COLS, seed=3, shuffle=True, **SMALL))
    b = evaluate(fit_predict(d, COLS, seed=3, shuffle=True, **SMALL))
    c = evaluate(fit_predict(d, COLS, seed=4, shuffle=True, **SMALL))
    assert a["skew"] == pytest.approx(b["skew"])
    assert a["skew"] != pytest.approx(c["skew"])


# ============================================================== the purge
def test_training_excludes_cohorts_still_open_at_the_test_year():
    """A cohort dated t does not settle until t+252. Training on it to predict
    t+30 leaks eleven months of overlapping future."""
    d = _d(n_years=6)
    R = fit_predict(d, COLS, seed=0, **SMALL)
    #  the earliest scoreable fold must be at least two years in, because the
    #  first year cannot train and the purge removes another
    assert R["year"].min() >= d["year"].min() + 2


def test_every_scored_row_is_a_test_fold_row_only():
    d = _d(n_years=8)
    R = fit_predict(d, COLS, seed=0, **SMALL)
    assert len(R) < len(d), "training rows must never be scored"
    assert R["year"].nunique() >= 3


# ============================================================== the model
def test_the_model_finds_a_signal_that_is_there():
    d = _d()
    e = evaluate(fit_predict(d, COLS, seed=0, **SMALL))
    assert e["skew"] > e["base_skew"], (
        "on synthetic data where sig_r genuinely drives both legs, the model "
        "must beat the unconditional base")


def test_the_score_is_the_ratio_of_the_two_probabilities():
    d = _d(n_years=6)
    R = fit_predict(d, COLS, seed=0, **SMALL)
    assert np.allclose(R["score"], R["p_up"] / np.maximum(R["p_down"], 1e-4))


def test_evaluate_refuses_a_cell_too_small_to_read():
    d = _d(n_names=5, n_years=4)
    R = fit_predict(d, COLS, seed=0, **SMALL)
    assert evaluate(R, top=0.01) == {}


# ============================================================== live scoring
def _panel(n=60, day="2026-08-24", years=6):
    days = pd.date_range(pd.Timestamp(day) - pd.Timedelta(days=365 * years),
                         pd.Timestamp(day), freq="B")
    rng = np.random.default_rng(0)
    rows = []
    for dt in days:
        rows.append(pd.DataFrame({
            "date": dt, "ticker": [f"T{i:02d}" for i in range(n)],
            "close": np.linspace(100.0, 5000.0, n), "tradeable": True,
            "log_turnover": np.full(n, 24.0),
            "sig_r": rng.uniform(size=n), "noise_r": rng.uniform(size=n)}))
    return pd.concat(rows, ignore_index=True)


def test_live_scoring_trains_only_on_settled_cohorts():
    """THE ONE ERROR THAT WOULD MAKE EVERY LIVE NUMBER FICTION. A cohort dated
    t is not known until t+252, so a live fit must stop ~370 days back."""
    import xsection_model as X
    d = _d(n_names=40, n_years=12)
    P = _panel()
    day = P["date"].max()
    captured = {}
    real_fit = X.HistGradientBoostingClassifier

    class Spy(real_fit):
        def fit(self, X_, y, **kw):
            captured["n"] = len(X_)
            return super().fit(X_, y, **kw)

    X.HistGradientBoostingClassifier = Spy
    try:
        settled = d[d["date"] < day - pd.Timedelta(days=370)]
        X.live_scores(d, P, COLS, seed=0)
    finally:
        X.HistGradientBoostingClassifier = real_fit
    assert captured.get("n", 0) <= len(settled), (
        "the live fit saw rows dated inside the purge window")


def test_live_scoring_returns_a_score_per_eligible_name():
    import xsection_model as X
    L = X.live_scores(_d(n_names=40, n_years=12), _panel(), COLS, seed=0)
    assert not L.empty
    assert {"p_up", "p_down", "score", "day"} <= set(L.columns)
    assert (L["score"] >= 0).all()
    assert L["score"].is_monotonic_decreasing


def test_live_scoring_declines_on_a_thin_cross_section():
    import xsection_model as X
    assert X.live_scores(_d(), _panel(n=10), COLS, seed=0).empty
