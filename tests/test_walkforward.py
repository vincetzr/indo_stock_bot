"""Walk-forward machinery.

The point of this module is to prevent a specific lie: parameters chosen with
knowledge of the test period. So the tests here are mostly about *what the
engine is not allowed to see*, checked with constructed data where the honest
answer is known in advance.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

from idxbot.walkforward import (  # noqa: E402
    WEIGHT_CANDIDATES,
    benchmark,
    blend,
    make_folds,
    nonoverlapping,
    portfolio_evaluator,
    run,
    stability,
)


def _panel(n_dates=200, n_names=30, seed=0, flip_date=None):
    """Synthetic panel where ``c_good`` predicts fwd_60 and ``c_bad`` does not.

    If ``flip_date`` is given, ``c_good`` stops working and ``c_other`` starts,
    which is the regime change a walk-forward is supposed to adapt to.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2000-01-03", periods=n_dates, freq="5B")
    rows = []
    for d in dates:
        for j in range(n_names):
            good = rng.uniform()
            other = rng.uniform()
            bad = rng.uniform()
            if flip_date is not None and d >= flip_date:
                signal = other
            else:
                signal = good
            rows.append({
                "date": d,
                "ticker": f"T{j:02d}",
                "close": 1000.0,
                "vt": 5e9,
                "c_good": good,
                "c_other": other,
                "c_bad": bad,
                # forward return is driven by the active signal plus noise
                "fwd_60": 0.40 * (signal - 0.5) + rng.normal(0, 0.02),
                "fwd_20": 0.20 * (signal - 0.5) + rng.normal(0, 0.02),
            })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# blend
# --------------------------------------------------------------------------

def test_blend_matches_normalised_weighted_sum():
    df = pd.DataFrame({"c_a": [0.0, 1.0, 0.5], "c_b": [1.0, 1.0, 0.5]})
    out = blend(df, {"a": 1.0, "b": 3.0})
    # normalised to 0.25 / 0.75, scaled to 0-100 like the engine's score
    assert out.tolist() == pytest.approx([75.0, 100.0, 50.0])


def test_blend_ignores_label_keys_and_missing_components():
    df = pd.DataFrame({"c_a": [0.25, 1.0]})
    out = blend(df, {"a": 1.0, "profile": "some label", "not_a_column": 5.0})
    assert out.tolist() == pytest.approx([25.0, 100.0])


def test_blend_with_no_usable_component_is_nan_not_zero():
    # Zero would rank as a real (lowest) score and quietly enter a portfolio.
    df = pd.DataFrame({"c_a": [1.0, 2.0]})
    out = blend(df, {"nonexistent": 1.0})
    assert out.isna().all()


# --------------------------------------------------------------------------
# nonoverlapping
# --------------------------------------------------------------------------

def test_nonoverlapping_spaces_dates_by_the_holding_period():
    dates = pd.bdate_range("2020-01-01", periods=60, freq="5B")  # every 5 trading days
    kept = nonoverlapping(dates, horizon_days=60)
    gaps = [(b - a).days for a, b in zip(kept, kept[1:])]
    assert all(g >= 84 for g in gaps)  # 60 trading days ~= 84 calendar days
    assert len(kept) < len(dates) / 5   # 12x overlap actually removed


def test_nonoverlapping_keeps_everything_when_horizon_is_one():
    dates = pd.bdate_range("2020-01-01", periods=10)
    assert len(nonoverlapping(dates, horizon_days=1)) == 10


def test_nonoverlapping_handles_empty():
    assert nonoverlapping([], 60) == []


# --------------------------------------------------------------------------
# folds
# --------------------------------------------------------------------------

def test_folds_never_let_train_reach_into_test():
    dates = pd.bdate_range("2000-01-01", "2020-01-01", freq="5B")
    folds = make_folds(dates, train_years=8, test_years=2)
    assert folds
    for f in folds:
        assert f.train_end <= f.test_start   # the whole point
        assert f.test_start < f.test_end


def test_folds_expand_and_do_not_overlap_in_test():
    dates = pd.bdate_range("2000-01-01", "2020-01-01", freq="5B")
    folds = make_folds(dates, train_years=8, test_years=2)
    for a, b in zip(folds, folds[1:]):
        assert b.train_end > a.train_end       # training window expands
        assert b.test_start >= a.test_end      # test slices tile forward


def test_too_little_history_yields_no_folds():
    dates = pd.bdate_range("2019-01-01", "2019-06-01")
    assert make_folds(dates, train_years=8, test_years=2) == []


# --------------------------------------------------------------------------
# portfolio evaluator
# --------------------------------------------------------------------------

def test_evaluator_picks_the_top_names_not_the_average():
    df = pd.DataFrame({
        "date": [pd.Timestamp("2020-01-01")] * 4,
        "c_a": [0.9, 0.8, 0.1, 0.0],
        "fwd_60": [0.10, 0.20, -0.50, -0.50],
    })
    ev = portfolio_evaluator(horizon=60, top_n=2, min_names=4)
    out = ev(df, {"a": 1.0})
    assert out.iloc[0] == pytest.approx(0.15)  # mean of the two best-scored


def test_evaluator_skips_cross_sections_that_are_too_thin():
    df = pd.DataFrame({
        "date": [pd.Timestamp("2020-01-01")] * 3,
        "c_a": [0.9, 0.5, 0.1],
        "fwd_60": [0.1, 0.0, -0.1],
    })
    ev = portfolio_evaluator(horizon=60, top_n=2, min_names=20)
    assert ev(df, {"a": 1.0}).empty


def test_evaluator_finds_the_predictive_component():
    df = _panel()
    ev = portfolio_evaluator(horizon=60, top_n=5, min_names=10)
    good = ev(df, {"good": 1.0}).mean()
    bad = ev(df, {"bad": 1.0}).mean()
    assert good > bad
    assert good > benchmark(df, 60, 10).mean()


# --------------------------------------------------------------------------
# run
# --------------------------------------------------------------------------

def test_run_selects_the_component_that_worked_in_training():
    df = _panel(n_dates=900)  # ~17 years: enough for an 8y train plus test folds
    ev = portfolio_evaluator(horizon=60, top_n=5, min_names=10)
    grid = [{"good": 1.0, "profile": "good"},
            {"bad": 1.0, "profile": "bad"}]
    res = run(df, grid, ev, train_years=8, test_years=2, verbose=False)
    assert not res.empty
    chosen = [f.chosen.get("profile") for f in res.folds if f.chosen]
    assert chosen and all(c == "good" for c in chosen)


def test_run_follows_whichever_regime_the_training_window_contains():
    """Selection tracks the training data, not the label on the component."""
    ev = portfolio_evaluator(horizon=60, top_n=5, min_names=10)
    grid = [{"good": 1.0, "profile": "good"},
            {"other": 1.0, "profile": "other"}]

    # Flipped from the very first bar: every training window is the new regime.
    flipped = _panel(n_dates=900, flip_date=pd.Timestamp("1999-01-01"))
    res = run(flipped, grid, ev, train_years=8, test_years=2, verbose=False)
    picks = [f.chosen.get("profile") for f in res.folds if f.chosen]
    assert picks and all(p == "other" for p in picks)

    # Never flipped: the same grid must land on the other component.
    steady = _panel(n_dates=900)
    res = run(steady, grid, ev, train_years=8, test_years=2, verbose=False)
    picks = [f.chosen.get("profile") for f in res.folds if f.chosen]
    assert picks and all(p == "good" for p in picks)


def test_expanding_window_is_slow_to_notice_a_regime_change():
    """A documented limitation, asserted so it cannot be forgotten.

    The training window only ever grows, so a regime that worked for a decade
    keeps outvoting a newer one for years after it stops working. That is the
    price of using every scrap of history, and it is why the walk-forward result
    on real IDX data should not be read as "the engine adapts".
    """
    flip = pd.Timestamp("2008-01-01")
    df = _panel(n_dates=900, flip_date=flip)
    ev = portfolio_evaluator(horizon=60, top_n=5, min_names=10)
    grid = [{"good": 1.0, "profile": "good"},
            {"other": 1.0, "profile": "other"}]
    res = run(df, grid, ev, train_years=8, test_years=2, verbose=False)

    picks = {f.test_start: f.chosen.get("profile") for f in res.folds if f.chosen}
    # Four years after the regime died, the stale component is still winning.
    stale = [v for k, v in picks.items()
             if flip + pd.DateOffset(years=2) < k <= flip + pd.DateOffset(years=5)]
    assert stale and all(v == "good" for v in stale)


def test_run_scores_the_test_slice_with_train_chosen_params_only():
    """A parameter that only wins in the test years must not be picked there."""
    df = _panel(n_dates=900)
    calls = []

    def spy(slice_df, params):
        calls.append((slice_df["date"].min(), slice_df["date"].max(), params))
        return portfolio_evaluator(60, 5, 10)(slice_df, params)

    grid = [{"good": 1.0, "profile": "good"}, {"bad": 1.0, "profile": "bad"}]
    res = run(df, grid, spy, train_years=8, test_years=2, verbose=False)

    for fold in res.folds:
        if not fold.chosen:
            continue
        train_calls = [c for c in calls if c[1] < fold.test_start]
        assert train_calls, "selection must have been driven by pre-test data"


def test_run_returns_empty_on_empty_input():
    assert run(pd.DataFrame(), [{"a": 1.0}], lambda d, p: pd.Series(dtype=float)).empty
    assert run(_panel(n_dates=50), [], lambda d, p: pd.Series(dtype=float)).empty


def test_stability_reports_share_of_folds_won():
    df = _panel(n_dates=900)
    ev = portfolio_evaluator(horizon=60, top_n=5, min_names=10)
    grid = [{"good": 1.0, "profile": "good"}, {"bad": 1.0, "profile": "bad"}]
    res = run(df, grid, ev, train_years=8, test_years=2, verbose=False)
    st = stability(res)
    assert not st.empty
    assert st["share"].sum() == pytest.approx(1.0)
    assert st.iloc[0]["params"] == "good"


# --------------------------------------------------------------------------
# shipped candidates
# --------------------------------------------------------------------------

def test_shipped_candidates_are_normalisable_and_named():
    for label, weights in WEIGHT_CANDIDATES.items():
        assert weights, f"{label} has no weights"
        assert all(v > 0 for v in weights.values()), label
        assert sum(weights.values()) == pytest.approx(1.0, abs=1e-6), label
