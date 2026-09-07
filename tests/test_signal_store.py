"""§38 — the signal log. Tests are about IMMUTABILITY and CAUSALITY.

With the holdout spent at H16, this store is the only mechanism in the repo that
can produce out-of-sample evidence. That makes two properties load-bearing: a
record must never be rewritten after the fact, and an outcome must never be
computed from a bar the decision could not have seen. Everything else is detail.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

from idxbot import signal_store as ss                              # noqa: E402

SCRIPTS = os.path.join(os.path.dirname(__file__), os.pardir, "scripts")


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    monkeypatch.setattr(ss, "STORE_DIR", str(tmp_path))
    monkeypatch.setattr(ss, "EMITTED", str(tmp_path / "emitted.csv.gz"))
    monkeypatch.setattr(ss, "OUTCOMES", str(tmp_path / "outcomes.csv.gz"))


def _panel(tk="AAAA", start="2026-01-05", n=60, path="up"):
    d = pd.bdate_range(start, periods=n)
    if path == "up":
        px = np.linspace(100, 200, n)
    elif path == "down":
        px = np.linspace(100, 40, n)
    else:
        px = np.full(n, 100.0)
    return pd.DataFrame({"ticker": tk, "date": d, "adj_close": px,
                         "close": px})


# ================================================================ IMMUTABILITY
def test_the_same_signal_emitted_twice_is_recorded_once():
    """The daily job re-running is normal and must be a no-op, not a duplicate
    and not a crash."""
    rows = [{"ticker": "AAAA", "entry": 100.0, "sl": 80.0, "tp": 200.0}]
    a = ss.emit(rows, "test", "v1", "2026-01-02", 20)
    b = ss.emit(rows, "test", "v1", "2026-01-02", 20)
    assert a["written"] == 1
    assert b["written"] == 0 and b["skipped"] == 1
    assert len(ss.load_emitted()) == 1


def test_an_existing_record_is_never_overwritten():
    """A record a later run can rewrite is a draft, not evidence. Re-emitting
    the SAME id with different levels must leave the original standing."""
    ss.emit([{"ticker": "AAAA", "entry": 100.0, "sl": 80.0, "tp": 200.0}],
            "test", "v1", "2026-01-02", 20)
    ss.emit([{"ticker": "AAAA", "entry": 999.0, "sl": 1.0, "tp": 2.0}],
            "test", "v1", "2026-01-02", 20)
    e = ss.load_emitted()
    assert len(e) == 1
    assert e["entry"].iloc[0] == 100.0, "the original record was overwritten"


def test_changing_the_rule_version_makes_it_a_DIFFERENT_prediction():
    """H56 changed STOP and H56b added TP. A signal under new parameters is not
    the old signal corrected — it is a new claim and needs its own row."""
    rows = [{"ticker": "AAAA", "entry": 100.0, "sl": 80.0, "tp": 200.0}]
    ss.emit(rows, "test", "v1", "2026-01-02", 20)
    ss.emit(rows, "test", "v2", "2026-01-02", 20)
    assert len(ss.load_emitted()) == 2


def test_every_row_carries_the_code_version_and_the_asof_bar():
    """§51: a signal generated today must be explainable months later. Without
    the git SHA it cannot be re-derived; without `asof` a scan run at 19:00 is
    indistinguishable from one acting on the next day's open."""
    ss.emit([{"ticker": "AAAA", "entry": 100.0, "sl": 80.0, "tp": 200.0,
              "hi52": 0.99}], "test", "v1", "2026-01-02", 20)
    e = ss.load_emitted().iloc[0]
    assert e["code_version"] and str(e["code_version"]) != "nan"
    assert pd.Timestamp(e["asof"]) == pd.Timestamp("2026-01-02")
    assert "hi52" in e["features"], "the knowable state was not recorded"


# =================================================================== CAUSALITY
def test_the_outcome_never_uses_the_decision_bar_or_anything_before_it():
    """THE TEST THIS FILE EXISTS FOR. A signal decided on the 15:50 close cannot
    be filled before the next session. Scoring from its own bar is a
    one-session look-ahead that would make a live record look better than the
    thing it tracks, and nothing in the output would look wrong."""
    P = _panel(n=60, start="2026-01-05")
    #  Decide on a bar in the middle; everything before must be invisible.
    asof = P["date"].iloc[30]
    ss.emit([{"ticker": "AAAA", "entry": float(P["adj_close"].iloc[30]),
              "sl": None, "tp": None}], "test", "v1", asof, 5)
    o = ss.score(P).iloc[0]
    assert pd.Timestamp(o["exit_date"]) > asof
    #  And the bars it saw are exactly the 5 after asof.
    assert o["bars_seen"] == 5
    expected = float(P["adj_close"].iloc[35])
    assert o["exit_px"] == pytest.approx(expected)


def test_an_unsettled_signal_is_reported_and_never_counted():
    """Some of the sample has not finished happening. Dropping it biases the
    record toward whatever resolves fastest; counting it as a win or a loss is
    worse."""
    P = _panel(n=10)
    ss.emit([{"ticker": "AAAA", "entry": 100.0, "sl": None, "tp": None}],
            "test", "v1", P["date"].iloc[0], 250)
    o = ss.score(P).iloc[0]
    assert not o["settled"]
    assert o["bars_seen"] == 9
    txt = ss.summary()
    assert "NOTHING HAS SETTLED" in txt or "open" in txt


def test_a_stop_and_a_target_are_taken_at_whichever_comes_FIRST():
    P = _panel(path="up", n=60)
    ss.emit([{"ticker": "AAAA", "entry": 100.0, "sl": 50.0, "tp": 150.0}],
            "test", "v1", P["date"].iloc[0], 250)
    o = ss.score(P).iloc[0]
    assert o["exit_reason"] == "tp" and o["hit_tp"]
    assert not o["hit_sl"]


def test_a_missing_ticker_is_recorded_as_no_data_not_as_a_zero_return():
    P = _panel(tk="BBBB")
    ss.emit([{"ticker": "AAAA", "entry": 100.0, "sl": None, "tp": None}],
            "test", "v1", "2026-01-05", 20)
    o = ss.score(P).iloc[0]
    assert o["exit_reason"] == "no data"
    assert not o["settled"]
    assert pd.isna(o["ret"]), "a delisted/absent name became a 0% return"


# ==================================================================== HONESTY
def test_the_summary_prints_power_before_it_concludes_anything():
    """A31 recorded conflating a POWER statement with an EFFECT statement as
    its own error, and measured that this kind of edge needs 46,856 months to
    distinguish from random. The store must say how much evidence exists before
    any number derived from it."""
    P = _panel(path="up", n=60)
    for i in range(6):
        ss.emit([{"ticker": "AAAA", "entry": float(P["adj_close"].iloc[i]),
                  "sl": None, "tp": None}], "test", "v1", P["date"].iloc[i], 5)
    ss.score(P)
    txt = ss.summary()
    assert "POWER" in txt
    assert "mean log" in txt, "the mean was printed without the mean log (A36)"


def test_an_empty_store_says_so_rather_than_printing_a_rate():
    assert "EMPTY" in ss.summary()


# ===================================================== THE ADJUSTMENT BASIS ==
#
# THE WORST BUG THIS REPO'S FORWARD RECORD HAS HAD, and the one that would have
# been invisible for months. `entry`, `sl` and `tp` are RAW prices — they have
# to be, they are the numbers you give a broker. The forward path is
# `adj_close`, which is BACK-adjusted and anchors to the newest bar, so every
# dividend or split after emission divides the whole history before it,
# INCLUDING the emission bar. Comparing a forward `adj_close` to a raw recorded
# `entry` therefore drifts further wrong with every corporate action — and on
# the day of emission the two bases agree, so nothing looks wrong at first.

def _panel_with_split(n=120, split_at=60, factor=5.0):
    """A name that splits 1:`factor` at `split_at`, back-adjusted like Yahoo."""
    dates = pd.bdate_range("2020-01-01", periods=n)
    raw = np.full(n, 1000.0)
    raw[split_at:] = 1000.0 / factor        # the traded price halves/fifths
    adj = np.full(n, 1000.0 / factor)       # back-adjusted: anchored at the end
    return pd.DataFrame({"ticker": "AAAA", "date": dates,
                         "close": raw, "adj_close": adj})


def test_a_split_after_emission_does_not_print_a_fake_loss():
    """Without the factor this scored -80% on a name that did not move."""
    P = _panel_with_split()
    asof = P["date"].iloc[10]
    ss.emit([{"ticker": "AAAA", "entry": 1000.0, "sl": 800.0, "tp": 2000.0}],
            "r", "v1", asof, 60)
    out = ss.score(P)
    r = out.iloc[0]
    assert r["adj_factor"] == pytest.approx(0.2)
    assert abs(float(r["ret"])) < 1e-6, (
        f"a flat name across a 1:5 split recorded {float(r['ret']):+.2%} — the "
        f"raw entry is being compared to an adjusted forward price")


def test_the_stop_is_carried_onto_the_adjusted_basis_too():
    """A raw stop tested against adjusted prices fires on the split, not on a
    fall — which would close every position in the book on an ex-date."""
    P = _panel_with_split()
    asof = P["date"].iloc[10]
    ss.emit([{"ticker": "AAAA", "entry": 1000.0, "sl": 800.0, "tp": 2000.0}],
            "r", "v1", asof, 60)
    out = ss.score(P)
    assert not bool(out.iloc[0]["hit_sl"]), (
        "the stop fired on a corporate action rather than on a price fall")


def test_a_real_fall_still_hits_the_stop_on_an_adjusted_series():
    """The control for the two tests above: the stop must still WORK."""
    P = _panel_with_split()
    #  inside the 60-bar horizon that starts at index 10
    P.loc[P.index[40:], ["close", "adj_close"]] = [100.0, 100.0]
    asof = P["date"].iloc[10]
    ss.emit([{"ticker": "AAAA", "entry": 1000.0, "sl": 800.0, "tp": 2000.0}],
            "r", "v1", asof, 60)
    out = ss.score(P)
    assert bool(out.iloc[0]["hit_sl"])
    assert out.iloc[0]["exit_reason"] == "sl"


def test_a_missing_decision_bar_is_refused_rather_than_assumed_unadjusted():
    """Falling back to a factor of 1.0 is the bug, not the safe default."""
    src = open(os.path.join(os.path.dirname(__file__), os.pardir, "src",
                            "idxbot", "signal_store.py")).read()
    assert "no basis bar" in src
    assert "assuming no adjustment is the bug" in src


# ========================================================== THE SCALE-OUT ==

def _flat_then_double(n=120, up_at=40):
    dates = pd.bdate_range("2020-01-01", periods=n)
    px = np.full(n, 1000.0)
    px[up_at:] = 2100.0
    return pd.DataFrame({"ticker": "AAAA", "date": dates,
                         "close": px, "adj_close": px})


def test_a_scale_out_is_not_scored_as_a_full_exit():
    """`rules.py` ships TP_FRAC = 0.5. Treating the target as a full exit caps
    the winner in the record while the live book runs on — the exact asymmetry
    H56b priced at 8.77% CAGR against 10.24%."""
    P = _flat_then_double()
    P.loc[P.index[80:], ["close", "adj_close"]] = [3000.0, 3000.0]
    asof = P["date"].iloc[10]
    ss.emit([{"ticker": "AAAA", "entry": 1000.0, "sl": 800.0, "tp": 2000.0,
              "tp_frac": 0.5}], "r", "v1", asof, 100)
    out = ss.score(P, cost=0.0)
    r = out.iloc[0]
    assert r["exit_reason"] == "tp_then_horizon"
    #  half at +110% (the bar that breached 2000 closed at 2100), half at +200%
    assert float(r["ret"]) == pytest.approx(0.5 * 1.10 + 0.5 * 2.00, abs=0.01)


def test_a_scale_out_whose_remainder_stops_out_is_recorded_as_such():
    P = _flat_then_double()
    P.loc[P.index[80:], ["close", "adj_close"]] = [500.0, 500.0]
    asof = P["date"].iloc[10]
    ss.emit([{"ticker": "AAAA", "entry": 1000.0, "sl": 800.0, "tp": 2000.0,
              "tp_frac": 0.5}], "r", "v1", asof, 100)
    out = ss.score(P, cost=0.0)
    r = out.iloc[0]
    assert r["exit_reason"] == "tp_then_sl"
    assert float(r["ret"]) == pytest.approx(0.5 * 1.10 + 0.5 * -0.50, abs=0.01)


def test_a_signal_without_a_frac_still_scores_as_a_full_exit():
    """Backward compatibility: rows emitted before `tp_frac` existed must not
    change meaning when the column arrives."""
    P = _flat_then_double()
    asof = P["date"].iloc[10]
    ss.emit([{"ticker": "AAAA", "entry": 1000.0, "sl": 800.0, "tp": 2000.0}],
            "r", "v1", asof, 100)
    out = ss.score(P, cost=0.0)
    assert out.iloc[0]["exit_reason"] == "tp"
    assert float(out.iloc[0]["ret"]) == pytest.approx(1.10, abs=0.01)


def test_the_shipped_logger_records_the_scale_out_fraction():
    """`rules.py` and the log must agree, or the record measures a rule nobody
    is trading."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir,
                                    "scripts"))
    import rules                                              # noqa: PLC0415
    src = open(os.path.join(os.path.dirname(__file__), os.pardir, "scripts",
                            "signal_log.py")).read()
    assert '"tp_frac": float(rules.TP_FRAC)' in src
    assert 0.0 < rules.TP_FRAC < 1.0


def test_the_scale_out_fraction_is_recovered_from_the_rule_version():
    """`tp_frac` was added after signals had been logged. Defaulting those rows
    to 1.0 would score them as full exits — a different rule from the one that
    produced them — but the information was never lost: `rule_version` carries
    `...tp1.0x0.5`, which IS the fraction."""
    assert ss._tp_frac({"rule_version": "hi0.9/0.8_vol0.5/0.6_k10_sl0.2_tp1.0x0.5",
                        "tp_frac": np.nan}) == pytest.approx(0.5)
    #  an explicit column wins over the string
    assert ss._tp_frac({"rule_version": "tp1.0x0.5", "tp_frac": 0.25}) == 0.25
    #  and an unparseable version falls back to a full exit, not to a guess
    assert ss._tp_frac({"rule_version": "whatever", "tp_frac": np.nan}) == 1.0
    assert ss._tp_frac({"rule_version": "tp1.0x9.0",
                        "tp_frac": np.nan}) == 1.0


def test_the_existing_log_rows_recover_their_fraction():
    """The rows already on disk must not silently change meaning."""
    em = ss.load_emitted()
    if em.empty:
        pytest.skip("no emitted signals on disk")
    fr = [ss._tp_frac(r) for _i, r in em.iterrows()]
    assert all(0.0 < f <= 1.0 for f in fr)


# ============================================ THE LIVE SURFACES AND THE STORE
#
# THE GAP THESE CLOSE. Every number in this repo is in-sample -- the holdout
# was spent at H16 -- so the append-only store is the ONLY route to
# out-of-sample evidence about ENTRY, SL and TP. The daily scanner fires every
# weekday under a Routine and wrote NOTHING to it: each of those predictions
# evaporated when the terminal scrolled. And the position monitor printed a
# catalogue of exit rules -- 169 of which A34 records as never beating a hold
# -- while omitting the two levels the reader was actually given.

def test_the_daily_scanner_can_log_what_it_showed():
    src = open(os.path.join(SCRIPTS, "daily_signal.py")).read()
    assert "--log" in src
    assert "h42_bracket_scan" in src


def test_the_scanner_logs_the_SHOWN_rows_not_the_whole_scan():
    """A row the reader never saw is not a prediction that was made, and a row
    the H42 gate REJECTED is one the scanner explicitly declined. Logging
    either would score a list nobody was given."""
    src = open(os.path.join(SCRIPTS, "daily_signal.py")).read()
    i = src.index("if a.log:")
    body = src[i:i + 2000]
    assert "fresh.iterrows()" in body
    assert "S.iterrows()" not in body


def test_the_scanner_uses_a_different_rule_name_from_the_quarterly_card():
    """`signal_id` hashes the rule, so pooling a quarterly basket and a daily
    bracket under one name would merge two records that must be read apart."""
    dsrc = open(os.path.join(SCRIPTS, "daily_signal.py")).read()
    lsrc = open(os.path.join(SCRIPTS, "signal_log.py")).read()
    assert "h42_bracket_scan" in dsrc
    assert "h42_bracket_scan" not in lsrc
    assert "h54_sticky_tight" in lsrc
    assert "h54_sticky_tight" not in dsrc


def test_the_scanner_fixes_its_horizon_at_emission():
    """A20: the horizon is the parameter twelve studies inherited without
    choosing, and choosing it after seeing the outcome is the purest form of
    the error."""
    src = open(os.path.join(SCRIPTS, "daily_signal.py")).read()
    i = src.index("h42_bracket_scan")
    assert "252" in src[i - 400:i + 200]


def test_the_monitor_can_take_its_book_from_the_store():
    src = open(os.path.join(SCRIPTS, "positions.py")).read()
    assert "--from-store" in src
    assert "load_emitted" in src


def test_the_monitor_drops_settled_signals_from_the_open_book():
    """A settled signal is history, not a position; printing it as one would
    overstate the book."""
    src = open(os.path.join(SCRIPTS, "positions.py")).read()
    assert 'oc["settled"]' in src
    assert "~em[\"signal_id\"].isin(done)" in src


def test_a_hand_entered_fill_wins_over_the_stores_close():
    """The store records the model's close; a hand-entered position records
    what was actually paid. Two rows for one ticker is one position."""
    src = open(os.path.join(SCRIPTS, "positions.py")).read()
    assert "seen, uniq" in src
    assert "the FIRST wins" in src


def test_the_monitor_prints_the_shipped_levels_before_the_catalogue():
    """The catalogue is 169 configurations of which none beat holding (A34).
    Printing it while omitting the SL and TP the reader was given hands them a
    screen full of rules that lost money and neither of the two that ship."""
    src = open(os.path.join(SCRIPTS, "positions.py")).read()
    assert "the level you were given" in src
    assert src.index("the level you were given") < src.index("M.levels(r,"), (
        "the shipped levels must print BEFORE the catalogue, where a reader "
        "who stops early still meets them")


def test_the_monitor_carries_the_given_levels_across_the_frame():
    """`position_frame` returns only what it computes, so levels arriving with
    the position have to be carried by ticker rather than assumed present --
    which they silently were not, and the rows printed nothing."""
    src = open(os.path.join(SCRIPTS, "positions.py")).read()
    assert "given = {d[\"ticker\"]: d for d in pos}" in src


def test_both_surfaces_are_logged_by_the_scheduled_refresh():
    """A log that depends on someone remembering has gaps exactly where the
    interesting days were. `signal_log.py` was wired in; the daily bracket scan
    -- which fires every weekday under a Routine -- was not."""
    src = open(os.path.join(SCRIPTS, "refresh.py")).read()
    i = src.index("if a.signals:")
    body = src[i:i + 1200]
    assert "signal_log.py" in body
    assert "daily_signal.py" in body and '"--log"' in body


# =========================================== RULE VERSIONS ARE NEVER POOLED ==
#
# A first `summary()` printed ONE mean over every row in the store. That is the
# composite error this repo bans everywhere else (A13: a blend of
# separately-tested components is a new signal wearing their credibility), and
# it was about to matter: H62 moved KEEP_HI from 0.80 to 0.70, so the store
# holds a SUPERSEDED card alongside the live one, plus a daily bracket the same
# repo says explicitly not to act on. A pooled mean over those describes no rule
# anyone is trading.

def test_outcomes_carry_the_rule_version():
    """Without it the scored rows cannot be separated by rule at all."""
    assert "rule_version" in ss.OUTCOME_COLUMNS
    P = _panel(path="up")
    asof = P["date"].iloc[5]
    ss.emit([{"ticker": "AAAA", "entry": 100.0, "sl": 80.0, "tp": 200.0}],
            "r", "v7", asof, 20)
    out = ss.score(P)
    assert out["rule_version"].iloc[0] == "v7"


def test_the_summary_breaks_out_by_rule_version_and_never_pools():
    P = _panel(path="up")
    asof = P["date"].iloc[5]
    ss.emit([{"ticker": "AAAA", "entry": 100.0, "sl": 80.0, "tp": 200.0}],
            "r", "v1", asof, 20)
    ss.emit([{"ticker": "AAAA", "entry": 100.0, "sl": 80.0, "tp": 200.0}],
            "r", "v2", asof, 20)
    txt = ss.summary(ss.score(P))
    assert "BY RULE VERSION (2 distinct)" in txt
    assert "never pooled" in txt
    assert "[v1]" in txt and "[v2]" in txt


def test_a_superseded_version_is_labelled_and_explained():
    """Declared rather than deleted: the store has no delete BY DESIGN, and
    adding one to cover an author's own mistake is exactly the door that design
    closes."""
    assert ss.SUPERSEDED, "nothing declared superseded"
    for (rule, ver), why in ss.SUPERSEDED.items():
        assert isinstance(rule, str) and isinstance(ver, str)
        assert len(why) > 30, (rule, ver, why)
        assert not ss.is_live(rule, ver)


def test_the_live_rule_is_not_marked_superseded():
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir,
                                    "scripts"))
    import signal_log                                          # noqa: PLC0415
    assert ss.is_live(signal_log.RULE, signal_log.RULE_VERSION), (
        "the rule the logger is about to emit is declared superseded")


def test_a_rule_name_does_not_carry_a_parameter_value():
    """`h54_sticky_tight` said 'tight' for a rule that was no longer tight the
    moment H62 moved the buffer. A name carrying a parameter drifts every time
    the parameter moves; the FAMILY is the name and the parameters live in
    `rule_version`."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir,
                                    "scripts"))
    import signal_log                                          # noqa: PLC0415
    for word in ("tight", "wide", "0.8", "0.7", "20", "100"):
        assert word not in signal_log.RULE, signal_log.RULE
    #  and the parameters ARE in the version
    assert "hi" in signal_log.RULE_VERSION and "sl" in signal_log.RULE_VERSION


# ------------------------------------------------------- THE REVIEW CADENCE

def _panel_days(n=200, start="2026-01-05"):
    d = pd.bdate_range(start, periods=n)
    return pd.DataFrame({"date": d, "ticker": "AAAA", "close": 100.0,
                         "adj_close": 100.0})


def test_a_quarterly_rule_is_not_logged_daily(monkeypatch, tmp_path):
    """THE DEFECT THIS GUARD EXISTS FOR.

    `refresh.py --signals` runs every weekday and `signal_log.py` had no
    cadence guard, so a rule that decides once every 63 sessions recorded a
    fresh basket every session. That inflates the store ~63x, which is untidy
    — and then `summary()` pools 63 overlapping near-identical predictions per
    quarter as though they were independent, which is the effective-n error
    this repo has recorded from A15 through A18, committed in the one place
    the numbers are supposed to become out-of-sample.

    Worse, it records the WRONG RULE: daily re-entry is H56's S2 arm, measured
    at 4.29% CAGR against the quarterly 13.46%.
    """
    import signal_log as SL
    monkeypatch.setattr(ss, "STORE_DIR", str(tmp_path))
    monkeypatch.setattr(ss, "EMITTED", str(tmp_path / "e.csv.gz"))
    monkeypatch.setattr(ss, "OUTCOMES", str(tmp_path / "o.csv.gz"))
    P = _panel_days()
    days = sorted(P["date"].unique())

    #  nothing emitted yet -> due
    assert SL.sessions_since_last(P, days[10]) is None

    ss.emit([{"ticker": "AAAA", "entry": 100.0, "sl": 80.0, "tp": 200.0}],
            SL.RULE, SL.RULE_VERSION, days[10], SL.HORIZON_DAYS)
    #  the next session is NOT a review
    assert SL.sessions_since_last(P, days[11]) == 1
    assert SL.sessions_since_last(P, days[11]) < SL.REVIEW_EVERY
    #  ...and neither is anything short of the full cadence
    assert SL.sessions_since_last(P, days[10 + SL.REVIEW_EVERY - 1]) \
        == SL.REVIEW_EVERY - 1
    #  the review session itself is
    assert SL.sessions_since_last(P, days[10 + SL.REVIEW_EVERY]) \
        >= SL.REVIEW_EVERY


def test_the_cadence_is_counted_in_SESSIONS_not_calendar_days(monkeypatch,
                                                              tmp_path):
    """A18 records a scheduler that converted held sessions to calendar days
    and silently skipped whole cohorts, hitting hardest exactly the
    short-holding arms the study existed to compare."""
    import signal_log as SL
    monkeypatch.setattr(ss, "STORE_DIR", str(tmp_path))
    monkeypatch.setattr(ss, "EMITTED", str(tmp_path / "e.csv.gz"))
    monkeypatch.setattr(ss, "OUTCOMES", str(tmp_path / "o.csv.gz"))
    P = _panel_days()
    days = sorted(P["date"].unique())
    ss.emit([{"ticker": "AAAA", "entry": 100.0, "sl": 80.0, "tp": 200.0}],
            SL.RULE, SL.RULE_VERSION, days[0], SL.HORIZON_DAYS)
    #  63 business days is 88 calendar days; the count must be 63, not 88.
    later = days[63]
    assert (later - days[0]).days > SL.REVIEW_EVERY
    assert SL.sessions_since_last(P, later) == SL.REVIEW_EVERY


def test_a_version_change_starts_a_new_review_clock(monkeypatch, tmp_path):
    """A46: a parameter change is a NEW prediction. Inheriting the old rule's
    clock would make the new version wait out the old one's quarter before it
    could ever be recorded."""
    import signal_log as SL
    monkeypatch.setattr(ss, "STORE_DIR", str(tmp_path))
    monkeypatch.setattr(ss, "EMITTED", str(tmp_path / "e.csv.gz"))
    monkeypatch.setattr(ss, "OUTCOMES", str(tmp_path / "o.csv.gz"))
    P = _panel_days()
    days = sorted(P["date"].unique())
    ss.emit([{"ticker": "AAAA", "entry": 100.0, "sl": 80.0, "tp": 200.0}],
            SL.RULE, "OLD_VERSION", days[0], SL.HORIZON_DAYS)
    #  the live version has never been emitted, so it is due immediately
    assert SL.sessions_since_last(P, days[1]) is None


def test_the_guard_explains_itself_rather_than_failing_silently():
    """A log that records nothing looks exactly like a log with nothing to
    record (A42). A skipped review must SAY it was skipped and why."""
    src = open(os.path.join(os.path.dirname(__file__), os.pardir, "scripts",
                            "signal_log.py")).read()
    assert "NOT DUE" in src
    assert "4.29%" in src and "13.46%" in src
    assert "--force" in src


def test_force_exists_so_a_review_can_be_taken_early():
    import signal_log as SL
    src = open(SL.__file__).read()
    assert 'args.force or n is None' in src
