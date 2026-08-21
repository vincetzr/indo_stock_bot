"""Tests for spine repairs and the Gate 0 corporate-action reconciliation.

Editing market data is the most dangerous thing in this repo. A wrong repair is
worse than the defect it replaces, because the defect is at least discoverable
and a repaired series looks clean by construction. So these tests are built
around three things:

  THE REPAIR IS JUSTIFIED. It must restore the split step the announcement
  implies. SCCO's last cum bar reads 2,543.75; repaired it is 10,175 against a
  first ex bar of 2,550, which is x0.2506 for an announced 1:4. Before the
  repair that boundary reads +0.2% and the split has silently vanished.

  THE REPAIR IS VISIBLE. A repaired bar carries a flag. Nothing may quietly
  become indistinguishable from original data.

  THE RECONCILIATION DISTINGUISHES THREE STATES, not two. A price series may be
  back-adjusted (no step at ex) OR unadjusted-but-consistent (a step equal to
  the theoretical factor), and both are correct. Only a step at the WRONG date
  is a failure. The first version of that check failed WIKA and DSSA for being
  in perfectly legitimate states.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

from idxbot.spine.repairs import (REPAIRS, Repair, apply_repairs,  # noqa: E402
                                  repairs_for, summary, verify)
from idxbot.spine.verified_actions import (VERIFIED,               # noqa: E402
                                           VerifiedAction, classify,
                                           reconcile)
from idxbot.spine.verified_actions import summary as ca_summary    # noqa: E402


def frame(dates, closes, volume=1000.0):
    return pd.DataFrame({
        "date": pd.to_datetime(dates), "open": closes, "high": closes,
        "low": closes, "close": closes, "adj_close": closes,
        "volume": [volume] * len(closes)})


# --------------------------------------------------------------------------
# the registry is deliberately tiny and fully sourced
# --------------------------------------------------------------------------
def test_every_repair_cites_a_reason_and_a_source():
    for r in REPAIRS:
        assert r.reason and r.source, f"{r.ticker} repair is unjustified"


def test_the_registry_holds_exactly_the_three_swept_defects():
    """If this grows, each addition must be a deliberate, sourced decision.

    It grew once, from SCCO alone, and only because the SHAPE was swept for:
    quality.suspect_islands finds off-tick-grid stretches that level_shifts
    also calls a break, and across 937 tickers it returns these three names
    and nothing else.
    """
    assert {r.ticker for r in REPAIRS} == {"SCCO", "PYFA", "SINI"}


def test_the_rights_repairs_touch_volume_and_the_split_repair_does_not():
    """Not a style choice - it is what the 100-share lot grid says happened.

    SCCO's early basis change moved the price only; share count did not change
    until the split took effect. PYFA's and SINI's windows carry volumes that
    are not multiples of 100, which no IDX print ever is, so the vendor scaled
    those too and undoing the price alone would leave traded value wrong.
    """
    by = {r.ticker: r for r in REPAIRS}
    assert by["SCCO"].volume_factor == 1.0
    for t in ("PYFA", "SINI"):
        assert by[t].volume_factor != 1.0
        assert by[t].price_factor * by[t].volume_factor == pytest.approx(1.0)


def test_a_ticker_with_no_repair_is_untouched():
    d = frame(pd.bdate_range("2024-02-01", periods=5), [100.0] * 5)
    out = apply_repairs(d, "BBCA")
    assert out["close"].equals(d["close"])
    assert not out["repaired"].any()


def test_repairs_are_looked_up_case_insensitively_and_without_suffix():
    assert repairs_for("scco.jk") and repairs_for("SCCO")


# --------------------------------------------------------------------------
# the repair does what it claims
# --------------------------------------------------------------------------
def test_only_the_registered_window_moves():
    dates = pd.to_datetime(["2024-01-31", "2024-02-01", "2024-03-07",
                            "2024-03-08"])
    d = frame(dates, [9975.0, 2493.75, 2543.75, 2550.0])
    out = apply_repairs(d, "SCCO")
    assert out["close"].iloc[0] == pytest.approx(9975.0)     # before, untouched
    assert out["close"].iloc[1] == pytest.approx(9975.0)     # in window, x4
    assert out["close"].iloc[2] == pytest.approx(10175.0)    # in window, x4
    assert out["close"].iloc[3] == pytest.approx(2550.0)     # after, untouched


def test_the_repair_restores_the_announced_split_step():
    """THE ACCEPTANCE TEST. 1:4 must reappear at the announced ex-date."""
    dates = pd.to_datetime(["2024-03-07", "2024-03-08"])
    d = frame(dates, [2543.75, 2550.0])
    out = apply_repairs(d, "SCCO")
    step = out["close"].iloc[1] / out["close"].iloc[0]
    assert step == pytest.approx(0.25, abs=0.01)


def test_without_the_repair_the_split_has_vanished_from_the_series():
    """What the defect actually does: the boundary reads as a normal day."""
    dates = pd.to_datetime(["2024-03-07", "2024-03-08"])
    d = frame(dates, [2543.75, 2550.0])
    step = d["close"].iloc[1] / d["close"].iloc[0]
    assert abs(step - 1.0) < 0.01          # +0.2%, indistinguishable from noise


def test_a_repaired_bar_is_flagged():
    dates = pd.to_datetime(["2024-01-31", "2024-02-15", "2024-03-08"])
    d = frame(dates, [9975.0, 2500.0, 2550.0])
    out = apply_repairs(d, "SCCO")
    assert list(out["repaired"]) == [False, True, False]


def test_volume_is_deliberately_not_repaired():
    """Share count did not change until the real ex-date, so February volume
    is already on the right basis. That asymmetry IS the defect."""
    dates = pd.to_datetime(["2024-02-15"])
    d = frame(dates, [2500.0], volume=76000.0)
    assert apply_repairs(d, "SCCO")["volume"].iloc[0] == pytest.approx(76000.0)


def test_all_price_columns_move_together():
    dates = pd.to_datetime(["2024-02-15"])
    d = frame(dates, [2500.0])
    out = apply_repairs(d, "SCCO")
    for c in ("open", "high", "low", "close", "adj_close"):
        assert out[c].iloc[0] == pytest.approx(10000.0)


def test_repairing_nothing_returns_nothing():
    assert apply_repairs(pd.DataFrame(), "SCCO").empty


def test_applying_a_repair_twice_does_not_apply_it_twice():
    """A x4 repair applied twice is x16, and the result still looks like a
    price series. This is the most dangerous thing the module could do, and it
    happened the first time Gate 0 repaired in its loader and again in verify.
    """
    dates = pd.to_datetime(["2024-02-15"])
    d = frame(dates, [2500.0])
    once = apply_repairs(d, "SCCO")
    twice = apply_repairs(once, "SCCO")
    assert twice["close"].iloc[0] == pytest.approx(10000.0)
    assert once["close"].equals(twice["close"])


def test_the_summary_lists_every_repair():
    s = summary()
    assert len(s) == len(REPAIRS)
    assert set(["ticker", "from", "to", "price_factor", "source"]).issubset(
        s.columns)


def test_verify_confirms_the_repair_against_the_announcement():
    def load(tk):
        dates = pd.to_datetime(["2024-01-31", "2024-03-07", "2024-03-08"])
        return frame(dates, [9975.0, 2543.75, 2550.0])
    v = verify(load)
    assert bool(v.iloc[0]["ok"])


def test_verify_rejects_a_repair_that_does_not_restore_the_step():
    """A wrong factor must fail, not pass quietly."""
    def load(tk):
        dates = pd.to_datetime(["2024-01-31", "2024-03-07", "2024-03-08"])
        return frame(dates, [9975.0, 2543.75, 9000.0])   # no split at ex
    assert not bool(verify(load).iloc[0]["ok"])


# --------------------------------------------------------------------------
# the three-state reconciliation
# --------------------------------------------------------------------------
SPLIT = VerifiedAction("T", "split", pd.Timestamp("2024-06-03"), ratio=4.0)


def around(pre, post, ex="2024-06-03"):
    ex = pd.Timestamp(ex)
    pre_d = pd.bdate_range(end=ex - pd.Timedelta(days=1), periods=6)
    post_d = pd.bdate_range(start=ex, periods=6)
    return frame(list(pre_d) + list(post_d), [pre] * 6 + [post] * 6)


def test_a_back_adjusted_series_has_no_step_at_the_ex_date():
    assert classify(around(2500.0, 2500.0), SPLIT)["state"] == "back_adjusted"


def test_an_unadjusted_series_with_the_right_step_is_consistent():
    """The step equals the factor and lands on the ex-date. Correct."""
    assert classify(around(10000.0, 2500.0), SPLIT)["state"] \
        == "unadjusted_consistent"


def test_a_step_at_the_wrong_date_is_misdated():
    """The SCCO shape: an action-sized step five weeks early."""
    ex = pd.Timestamp("2024-06-03")
    early = pd.Timestamp("2024-04-29")
    dates = list(pd.bdate_range(end=early - pd.Timedelta(days=1), periods=4)) \
        + list(pd.bdate_range(start=early, periods=30))
    closes = [10000.0] * 4 + [2500.0] * 30
    r = classify(frame(dates, closes), SPLIT)
    assert r["state"] == "misdated"
    assert "misdated_on" in r


def test_a_step_years_away_is_a_different_action_not_a_misdating():
    """Searching the whole history blamed SCCO's 2024 split on a 2019 step."""
    ex = pd.Timestamp("2024-06-03")
    old = list(pd.bdate_range("2019-01-01", periods=4))
    dates = old + list(pd.bdate_range(end=ex - pd.Timedelta(days=1),
                                      periods=6)) \
        + list(pd.bdate_range(start=ex, periods=6))
    # The ONLY 4x step is in 2019. Everything from then to the ex-date and
    # beyond sits flat, so nothing near the ex-date can be mistaken for it.
    closes = [10000.0] * 2 + [2500.0] * 2 + [2500.0] * 6 + [2500.0] * 6
    assert classify(frame(dates, closes), SPLIT)["state"] != "misdated"


def test_an_ordinary_big_day_inside_the_band_is_not_a_break():
    """DSSA rose 16.4% on its ex-date, inside a 25% band. Not a failure."""
    assert classify(around(2680.0, 3120.0), SPLIT)["state"] == "back_adjusted"


def test_a_suspension_across_the_ex_date_does_not_read_as_a_break():
    """WIKA was suspended either side of its rights issue.

    Comparing traded bars spans the suspension and reports a 32% break that is
    just the stock reopening. Quoted bars, stale included, are the right basis.
    """
    ex = pd.Timestamp("2024-04-17")
    pre = pd.bdate_range(end=ex - pd.Timedelta(days=1), periods=4)
    post = pd.bdate_range(start=ex, periods=4)
    d = frame(list(pre) + list(post), [203.913] * 4 + [203.913] * 4, volume=0.0)
    d.loc[d.index[-1], "volume"] = 1e6
    d.loc[d.index[-1], ["open", "high", "low", "close"]] = 162.0
    act = VerifiedAction("WIKA", "rights", ex, factor=204.0 / 240.0)
    assert classify(d, act)["state"] in ("back_adjusted",
                                         "unadjusted_consistent")


def test_the_gate_fails_on_a_single_misdated_action():
    R = pd.DataFrame({"state": ["back_adjusted"] * 6 + ["misdated"]})
    s = ca_summary(R)
    assert s["gate_passes"] is False
    assert "misdated" in s["verdict"]


def test_the_gate_needs_five_events_even_if_all_pass():
    R = pd.DataFrame({"state": ["back_adjusted"] * 3})
    s = ca_summary(R)
    assert s["gate_passes"] is False
    assert "fewer than 5" in s["verdict"]


def test_the_gate_passes_on_five_consistent_events():
    R = pd.DataFrame({"state": ["back_adjusted"] * 3
                      + ["unadjusted_consistent"] * 2})
    assert ca_summary(R)["gate_passes"] is True


def test_an_unclassifiable_event_fails_the_gate_rather_than_being_ignored():
    R = pd.DataFrame({"state": ["back_adjusted"] * 5 + ["unclear"]})
    assert ca_summary(R)["gate_passes"] is False


def test_the_verified_registry_meets_the_five_event_requirement():
    assert len(VERIFIED) >= 5


def test_every_verified_action_cites_a_source():
    for a in VERIFIED:
        assert a.source, f"{a.ticker} {a.ex_date} has no source"


# --------------------------------------------------------------------------
# quarantining what has NOT been verified
# --------------------------------------------------------------------------
def test_an_unverified_shift_is_quarantined_not_trusted():
    """SCCO proved a detected shift can be confidently wrong about its date."""
    from idxbot.spine.repairs import SUSPECT, suspect_mask
    assert SUSPECT, "the unverified shifts must be recorded somewhere"
    d = frame(pd.to_datetime(["2018-06-07", "2018-10-01"]), [100.0, 100.0])
    m = suspect_mask(d, "ELTY")
    assert bool(m.iloc[0]) and not bool(m.iloc[1])


def test_a_shift_leaves_quarantine_only_when_its_factor_is_confirmed():
    """PYFA and SINI sat in SUSPECT while their cause was known and their
    factor was not. Both left only once the announced ratio was found - never
    by reading the factor off the move it was supposed to explain."""
    from idxbot.spine.repairs import suspects_for
    for t in ("PYFA", "SINI"):
        assert not suspects_for(t)
        assert any(r.ticker == t and r.source for r in REPAIRS)


def test_a_clean_ticker_has_no_quarantine():
    from idxbot.spine.repairs import suspect_mask
    d = frame(pd.to_datetime(["2024-04-16"]), [100.0])
    assert not suspect_mask(d, "BBCA").any()


def test_the_quarantine_is_wider_than_the_scco_error():
    """SCCO's error spanned 36 days; a narrower window would have missed it."""
    from idxbot.spine.repairs import SUSPECT_WINDOW_DAYS
    assert SUSPECT_WINDOW_DAYS > 36


def test_a_verified_action_is_not_also_a_suspect():
    """Once checked, a shift leaves quarantine. SCCO is repaired, not suspect."""
    from idxbot.spine.repairs import suspects_for
    assert not suspects_for("SCCO")


# --------------------------------------------------------------------------
# the loader applies repairs on every path
# --------------------------------------------------------------------------
def test_the_loader_repairs_on_read_and_leaves_the_cache_raw(tmp_path):
    import idxbot.data.ohlcv as O
    from idxbot.data.cache import Cache
    cache = Cache(str(tmp_path))
    raw = frame(pd.to_datetime(["2024-02-15", "2024-03-08"]), [2500.0, 2550.0])
    cache.write("ohlcv", "SCCO.JK", raw)
    loader = O.YahooOHLCV.__new__(O.YahooOHLCV)
    loader.cache = cache
    out = loader.get("SCCO", max_age=1e12)
    assert out["close"].iloc[0] == pytest.approx(10000.0)
    assert out["close"].iloc[1] == pytest.approx(2550.0)
    back = cache.read("ohlcv", "SCCO.JK")
    assert back["close"].iloc[0] == pytest.approx(2500.0), "cache must stay raw"


def test_a_ticker_with_no_repair_passes_through_the_loader_unchanged(tmp_path):
    import idxbot.data.ohlcv as O
    from idxbot.data.cache import Cache
    cache = Cache(str(tmp_path))
    raw = frame(pd.to_datetime(["2024-02-15"]), [2500.0])
    cache.write("ohlcv", "BBCA.JK", raw)
    loader = O.YahooOHLCV.__new__(O.YahooOHLCV)
    loader.cache = cache
    assert loader.get("BBCA", max_age=1e12)["close"].iloc[0] == pytest.approx(2500.0)
