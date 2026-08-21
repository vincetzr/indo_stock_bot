"""Tests for the point-in-time broker code master.

The distinction this file protects is the one that decides whether the master
is useful or destructive:

  A RENAME IS NOT A REASSIGNMENT. Code YP was eTrading, then Daewoo, then
  Mirae Asset - three names, one continuing business, one continuous client
  base. Splitting YP's history at each rename would destroy a real fifteen-year
  record to fix a problem that does not exist.

  A MERGER IS A DISCONTINUITY. When UBS absorbed Credit Suisse the flow behind
  CS did not gradually become UBS flow, it moved. Comparing a fingerprint
  across that date measures the merger.

The other thing tested here is refusal: a code with no recorded history must
say so rather than return a confident blank, because "we know nothing" and
"nothing changed" look identical in a result table and mean opposite things.
"""

from __future__ import annotations

import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

from idxbot.spine.brokers import (activity_audit, coverage,        # noqa: E402
                                  firm, history, same_entity)


# --------------------------------------------------------------------------
# renames preserve the entity
# --------------------------------------------------------------------------
def test_yp_is_named_differently_in_different_eras():
    assert firm("YP", "2010-06-01")["name"] == "eTrading Securities"
    assert firm("YP", "2014-06-01")["name"] == "Daewoo Securities Indonesia"
    assert firm("YP", "2026-06-01")["name"] == "Mirae Asset Sekuritas Indonesia"


def test_yp_is_the_same_continuing_business_throughout():
    """THE KEY TEST. Three names, one client base - comparable end to end."""
    entities = {firm("YP", d)["entity"]
                for d in ("2010-06-01", "2014-06-01", "2026-06-01")}
    assert len(entities) == 1
    assert same_entity("YP", "2010-06-01", "2026-06-01")


def test_other_ownership_changes_are_also_renames_not_breaks():
    for code, a, b in (("OD", "2019-01-01", "2026-01-01"),
                       ("ZP", "2019-01-01", "2026-01-01"),
                       ("YU", "2016-01-01", "2026-01-01"),
                       ("DR", "2016-01-01", "2026-01-01")):
        assert same_entity(code, a, b), f"{code} should be continuous"


# --------------------------------------------------------------------------
# mergers are not
# --------------------------------------------------------------------------
def test_a_merger_is_marked_as_a_discontinuity():
    assert firm("CS", "2020-01-01")["discontinuity"] is True


def test_a_fingerprint_may_not_be_compared_across_a_merger():
    assert not same_entity("CS", "2020-01-01", "2026-01-01")


def test_the_absorbing_broker_carries_a_warning_about_its_own_continuity():
    """AK's client base is not continuous across 2023 either."""
    note = firm("AK", "2026-01-01")["note"]
    assert "Credit Suisse" in note and "not continuous" in note


def test_two_different_codes_are_never_the_same_entity():
    assert same_entity("YP", "2026-01-01", "2026-01-02") is True
    assert firm("YP", "2026-01-01")["entity"] != firm("AK", "2026-01-01")["entity"]


# --------------------------------------------------------------------------
# refusal
# --------------------------------------------------------------------------
def test_an_unrecorded_code_says_it_is_unrecorded():
    f = firm("ZZ", "2026-01-01")
    assert f["confidence"] == "unknown"
    assert f["name"] is None
    assert "no recorded history" in f["note"]


def test_a_recorded_code_outside_its_dates_is_distinguished_from_an_unknown_one():
    """CS after the merger: we know the code, and we know it is not covered."""
    f = firm("CS", "2026-01-01")
    assert f["confidence"] == "unknown"
    assert "has recorded history" in f["note"]


def test_an_unknown_code_is_never_treated_as_continuous():
    assert not same_entity("ZZ", "2020-01-01", "2026-01-01")


def test_a_registry_only_code_is_named_but_marked_history_unknown():
    """CC is Mandiri Sekuritas today; nothing is known about when it changed."""
    f = firm("CC", "2026-01-01", registry={"CC": "Mandiri Sekuritas"})
    assert f["name"] == "Mandiri Sekuritas"
    assert f["confidence"] == "current_only"
    assert f["entity"] is None
    assert "must not be compared" in f["note"]


def test_a_registry_only_code_may_not_be_compared_across_eras():
    """Naming it is safe. Treating it as continuous is not."""
    reg = {"CC": "Mandiri Sekuritas"}
    assert firm("CC", "2015-01-01", registry=reg)["name"] == "Mandiri Sekuritas"
    assert not same_entity("CC", "2015-01-01", "2026-01-01")


def test_a_dated_record_beats_the_registry():
    f = firm("YP", "2010-06-01", registry={"YP": "Mirae Asset Sekuritas"})
    assert f["name"] == "eTrading Securities"
    assert f["confidence"] == "verified"


def test_an_empty_registry_falls_back_to_unknown():
    f = firm("CC", "2026-01-01", registry={})
    assert f["confidence"] == "unknown" and f["name"] is None


def test_every_record_carries_a_confidence_the_caller_can_see():
    h = history()
    assert h["confidence"].isin(("verified", "reported", "unknown")).all()
    assert (h["confidence"] == "verified").any()


def test_the_verified_records_cite_a_source():
    h = history()
    v = h[h["confidence"] == "verified"]
    assert len(v) >= 3
    assert v["source"].str.len().gt(0).all()


def test_case_and_whitespace_do_not_change_the_answer():
    assert firm(" yp ", "2026-01-01")["entity"] == firm("YP", "2026-01-01")["entity"]


# --------------------------------------------------------------------------
# coverage
# --------------------------------------------------------------------------
def test_coverage_reports_what_has_no_history():
    c = coverage(["YP", "CS", "ZZ", "QQ"])
    assert c["observed"] == 4
    assert set(c["missing"]) == {"ZZ", "QQ"}
    assert 0 < c["fraction"] < 1


def test_coverage_of_nothing_does_not_divide_by_zero():
    assert coverage([])["fraction"] == 0


# --------------------------------------------------------------------------
# the empirical audit, and its honesty about censoring
# --------------------------------------------------------------------------
def frame(rows):
    return pd.DataFrame(rows)


def test_a_code_that_stops_appearing_is_flagged_as_a_candidate():
    days = pd.bdate_range("2025-01-01", periods=300)
    rows = [{"broker": "AA", "date": d} for d in days]
    rows += [{"broker": "BB", "date": d} for d in days[:60]]
    a = activity_audit(frame(rows))
    assert list(a["broker"]) == ["BB"]
    assert a.iloc[0]["event"] == "stopped appearing"


def test_a_code_that_starts_appearing_is_flagged():
    days = pd.bdate_range("2025-01-01", periods=300)
    rows = [{"broker": "AA", "date": d} for d in days]
    rows += [{"broker": "BB", "date": d} for d in days[-60:]]
    a = activity_audit(frame(rows))
    assert a.iloc[0]["event"] == "started appearing"


def test_a_censored_source_is_labelled_censored():
    """On a top-ten table absence means smallness, not non-existence."""
    days = pd.bdate_range("2025-01-01", periods=300)
    rows = [{"broker": "AA", "date": d, "complete": False} for d in days]
    rows += [{"broker": "BB", "date": d, "complete": False} for d in days[:60]]
    assert bool(activity_audit(frame(rows)).iloc[0]["censored"]) is True


def test_a_complete_source_is_not_labelled_censored():
    days = pd.bdate_range("2025-01-01", periods=300)
    rows = [{"broker": "AA", "date": d, "complete": True} for d in days]
    rows += [{"broker": "BB", "date": d, "complete": True} for d in days[:60]]
    assert bool(activity_audit(frame(rows)).iloc[0]["censored"]) is False


def test_a_code_with_too_short_a_record_is_not_flagged():
    days = pd.bdate_range("2025-01-01", periods=300)
    rows = [{"broker": "AA", "date": d} for d in days]
    rows += [{"broker": "BB", "date": d} for d in days[:5]]
    assert "BB" not in set(activity_audit(frame(rows))["broker"])


def test_the_audit_of_nothing_is_empty_not_a_crash():
    assert activity_audit(pd.DataFrame()).empty
