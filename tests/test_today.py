"""The one command that says what to act on, and why.

THE TEST THIS FILE EXISTS FOR is `test_the_verdict_is_derived_not_asserted`.
Three surfaces implement the standing instruction's contract and they DISAGREE
— a quarterly basket, a daily bracket, and a monitor over what is already open.
A reader running all three gets three lists and nothing telling them which the
evidence supports. The resolution must be COMPUTED from the measured constants,
because a sentence written by hand drifts from the table it describes, and this
repo has recorded that four times: a reconciliation figure that outlived its
source, a docstring asserting what its own function had retracted, and Pine
constants that had to be pinned by test.

The second is `test_the_two_rules_are_never_blended`. A13 records why: a
composite of separately-tested components is a new signal wearing their
credibility, and it has never been tested.
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "scripts"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

import today                                                      # noqa: E402
from today import BRACKET, CARD, INDEX, open_book, precedence     # noqa: E402

SRC = os.path.join(os.path.dirname(__file__), os.pardir, "scripts", "today.py")


# ------------------------------------------------------------- THE VERDICT

def test_the_verdict_is_derived_not_asserted():
    out = "\n".join(precedence())
    assert "ACT ON THE CARD" in out
    assert "WATCHLIST" in out
    #  every number in the verdict comes from the constants
    assert f"{BRACKET['vs_hold']:+.2%}" in out
    assert f"{CARD['beats_index_n']}/{CARD['phases']}" in out


def test_the_verdict_flips_when_the_measurement_does(monkeypatch):
    """If the bracket stopped losing to a hold, the sentence must change. A
    verdict that cannot change is a slogan."""
    monkeypatch.setitem(BRACKET, "vs_hold", +0.05)
    out = "\n".join(precedence())
    assert "The bracket list is a WATCHLIST" not in out
    assert "re-read H42" in out


def test_the_verdict_refuses_both_when_the_card_stops_clearing(monkeypatch):
    monkeypatch.setitem(CARD, "beats_index_n", 3)
    out = "\n".join(precedence())
    assert "NEITHER LIST IS SUPPORTED" in out


def test_the_qualifications_print_whatever_the_verdict(monkeypatch):
    """A verdict without them is the thing this file exists to prevent."""
    for patch in ({}, {"beats_index_n": 3}, {"cagr_both_halves": True}):
        for k, v in patch.items():
            monkeypatch.setitem(CARD, k, v)
        out = "\n".join(precedence())
        assert "deflated Sharpe" in out
        assert "IN-SAMPLE" in out
        assert "H16" in out


def test_the_deflated_sharpe_caveat_carries_the_number_that_fails():
    out = "\n".join(precedence())
    assert f"{CARD['dsr_next']:.3f}" in out and "FAILS" in out
    assert f"{CARD['dsr_survives']} of {CARD['dsr_of']}" in out


def test_the_cagr_edge_is_explicitly_not_claimed():
    """A18: worse in the early half, better in the late one, is regime noise.
    The drawdown is what holds in both halves and is what may be claimed."""
    out = "\n".join(precedence())
    assert "NOT claimed" in out
    assert "What IS claimed is the drawdown" in out


# --------------------------------------------------------- THE COMPOSITION

def test_the_two_rules_are_never_blended():
    """A13: a composite of separately-tested components is a new signal
    wearing their credibility."""
    src = open(SRC).read()
    assert "does NOT blend" in src
    for forbidden in ("mean(", "concat([card", "+ bracket[", "blend("):
        assert forbidden not in src.split('"""', 2)[2], forbidden


def test_the_bracket_rows_are_labelled_do_not_act():
    src = open(SRC).read()
    assert "do NOT act on this one" in src
    assert "act on this one" in src


# --------------------------------------------------------- THE BENCHMARK

def test_the_benchmark_prints_before_either_list():
    src = open(SRC).read()
    body = src.split('"""', 2)[2]
    assert body.index("THE ALTERNATIVE YOU ALREADY HAVE") < \
        body.index("WHICH LIST TO ACT ON"), (
        "A19 records the missing benchmark as the error class that "
        "manufactures results; it belongs at the top, not in a footnote")


def test_both_index_figures_are_quoted_with_their_windows():
    """Picking one and calling it 'the index' is A19's own error class —
    comparing quantities measured over different windows."""
    assert "a19" in INDEX and "a38" in INDEX
    assert INDEX["a19"] != INDEX["a38"]
    assert "span" in INDEX["note"]


def test_the_precedence_does_not_depend_on_which_index_figure_is_used():
    """It turns on the 6-of-6 calendar count, which is already window-matched
    per arm — so the ambiguity above cannot leak into the verdict."""
    src = open(SRC).read()
    i = src.index("def precedence(")
    body = src[i:src.index("\ndef ", i + 10)]
    assert "INDEX[" not in body


# --------------------------------------------------------- THE OPEN BOOK

def test_the_open_book_excludes_settled_signals(monkeypatch, tmp_path):
    """A settled signal is history, not a position."""
    import pandas as pd
    from idxbot import signal_store as ss
    monkeypatch.setattr(ss, "STORE_DIR", str(tmp_path))
    monkeypatch.setattr(ss, "EMITTED", str(tmp_path / "e.csv.gz"))
    monkeypatch.setattr(ss, "OUTCOMES", str(tmp_path / "o.csv.gz"))
    ss.emit([{"ticker": "AAAA", "entry": 100.0, "sl": 80.0, "tp": 200.0},
             {"ticker": "BBBB", "entry": 100.0, "sl": 80.0, "tp": 200.0}],
            "r", "v", pd.Timestamp("2026-01-05"), 252)
    em = ss.load_emitted()
    done = em["signal_id"].iloc[0]
    #  `_read` parses `asof`, so an outcomes file without it fails to load and
    #  comes back EMPTY — which silently drops nothing and looks like a pass.
    pd.DataFrame({"signal_id": [done], "asof": [pd.Timestamp("2026-01-05")],
                  "settled": [True]}).to_csv(
        str(tmp_path / "o.csv.gz"), index=False, compression="gzip")
    ob = open_book()
    assert list(ob["ticker"]) == ["BBBB"]


def test_an_empty_store_says_so_rather_than_printing_a_blank_table(capsys,
                                                                   monkeypatch,
                                                                   tmp_path):
    from idxbot import signal_store as ss
    monkeypatch.setattr(ss, "EMITTED", str(tmp_path / "none.csv.gz"))
    monkeypatch.setattr(ss, "OUTCOMES", str(tmp_path / "none2.csv.gz"))
    assert open_book().empty
    src = open(SRC).read()
    assert "nothing open" in src


def test_the_scale_out_fraction_is_shown_per_row():
    """The card sells HALF at the target and the bracket exits fully. A book
    that shows both without saying which is which is two rules under one
    heading."""
    src = open(SRC).read()
    assert "_tp_frac(r)" in src
    assert "'part'" in src or '"part"' in src


# ------------------------------------------------------------- THE CLOSER

def test_the_closing_disclaimer_is_present_and_unconditional():
    src = open(SRC).read()
    body = src.split('"""', 2)[2]
    assert "holdout was spent at H16" in body
    assert "A23" in body
    assert "no live track record" in body


def test_a_failing_subprocess_is_reported_rather_than_swallowed():
    src = open(SRC).read()
    assert "exited" in src and "returncode" in src


# ============================================ THE FIGURES COME FROM THE FILE ==

def test_the_card_figures_are_read_from_the_result_file():
    """H62 moved KEEP_HI from 0.80 to 0.70 and every hardcoded figure went on
    printing as "what the card is measured to do" — describing a rule nobody
    ships. A number copied out of a study has no link back to the study."""
    src = open(SRC).read()
    assert "def card_measured(" in src
    assert "stoptest.json" in src


def test_a_result_file_from_a_different_rule_is_refused(tmp_path, monkeypatch):
    """The file stamps the constants it ran with. If they differ from what
    ships, the numbers describe a different rule and the caller must be told
    rather than shown them."""
    import json
    import rules
    p = tmp_path / "stoptest.json"
    p.write_text(json.dumps({
        "rule": {"ENTRY_HI": rules.ENTRY_HI, "ENTRY_VOL": rules.ENTRY_VOL,
                 "KEEP_HI": rules.KEEP_HI + 0.10, "KEEP_VOL": rules.KEEP_VOL},
        "arms": [{"arm": "BASE: quarterly keep-band only",
                  "cagr_med": 0.99, "dd": -0.10},
                 {"arm": "SHIPPED: stop 20% + sell HALF at +100%",
                  "cagr_med": 0.99, "dd": -0.10}]}))
    monkeypatch.setattr(today, "STOPTEST", str(p))
    m, why = today.card_measured()
    assert "DIFFERENT rule" in why
    assert m["cagr"] != 0.99, "figures from the wrong rule were adopted"


def test_a_matching_result_file_is_adopted(tmp_path, monkeypatch):
    import json
    import rules
    p = tmp_path / "stoptest.json"
    p.write_text(json.dumps({
        "rule": {"ENTRY_HI": rules.ENTRY_HI, "ENTRY_VOL": rules.ENTRY_VOL,
                 "KEEP_HI": rules.KEEP_HI, "KEEP_VOL": rules.KEEP_VOL},
        "arms": [{"arm": "BASE: quarterly keep-band only",
                  "cagr_med": 0.11, "dd": -0.40},
                 {"arm": "SHIPPED: stop 20% + sell HALF at +100%",
                  "cagr_med": 0.13, "dd": -0.38}]}))
    monkeypatch.setattr(today, "STOPTEST", str(p))
    m, why = today.card_measured()
    assert "measured on the live rule" in why
    assert m["cagr"] == 0.13 and m["cagr_none"] == 0.11
    assert m["maxdd"] == -0.38 and m["maxdd_none"] == -0.40


def test_a_missing_result_file_falls_back_and_says_so(tmp_path, monkeypatch):
    monkeypatch.setattr(today, "STOPTEST", str(tmp_path / "nope.json"))
    m, why = today.card_measured()
    assert "unavailable" in why and "stored values" in why
    assert m["cagr"] == today.CARD["cagr"]


def test_the_provenance_line_is_printed_with_the_verdict():
    src = open(SRC).read()
    assert 'print(f"  card figures: {provenance}")' in src


def test_an_unstamped_result_file_is_refused_not_crashed_on(tmp_path,
                                                             monkeypatch):
    """The file used to be a bare LIST of arms with no record of which rule
    produced them — exactly the case this check exists to catch, so it must
    report rather than raise. `except` around the load alone did not cover
    `.get` on a list."""
    import json
    p = tmp_path / "stoptest.json"
    p.write_text(json.dumps([{"arm": "BASE: quarterly keep-band only",
                              "cagr_med": 0.99}]))
    monkeypatch.setattr(today, "STOPTEST", str(p))
    m, why = today.card_measured()
    assert "no rule stamp" in why
    assert m["cagr"] == today.CARD["cagr"]


def test_an_empty_rule_stamp_is_refused(tmp_path, monkeypatch):
    import json
    p = tmp_path / "stoptest.json"
    p.write_text(json.dumps({"rule": {}, "arms": []}))
    monkeypatch.setattr(today, "STOPTEST", str(p))
    _m, why = today.card_measured()
    assert "empty rule stamp" in why


def test_a_schema_mismatch_falls_back_rather_than_raising(tmp_path,
                                                           monkeypatch):
    """A first reader assumed `cagr`/`maxdd`; the writer emits `cagr_med`/`dd`.
    The KeyError was the GOOD outcome -- a silently wrong value would have been
    the bad one -- but a shipped surface must report, not crash."""
    import json
    import rules
    p = tmp_path / "stoptest.json"
    p.write_text(json.dumps({
        "rule": {"ENTRY_HI": rules.ENTRY_HI, "ENTRY_VOL": rules.ENTRY_VOL,
                 "KEEP_HI": rules.KEEP_HI, "KEEP_VOL": rules.KEEP_VOL},
        "arms": [{"arm": "BASE: quarterly keep-band only", "wrong": 1},
                 {"arm": "SHIPPED: stop 20% + sell HALF at +100%",
                  "wrong": 1}]}))
    monkeypatch.setattr(today, "STOPTEST", str(p))
    m, why = today.card_measured()
    assert "expected fields" in why
    assert m["cagr"] == today.CARD["cagr"]


def test_the_cagr_noun_follows_the_sign(monkeypatch):
    """It read "its CAGR edge" after the effect turned negative: H62 moved the
    buffer, the levels went from a +0.76-point gain to a -0.49-point cost, and
    the prose did not follow. Deriving the noun is the only version that
    survives the next re-measurement."""
    monkeypatch.setitem(CARD, "cagr", 0.10)
    monkeypatch.setitem(CARD, "cagr_none", 0.12)
    assert "CAGR cost" in "\n".join(precedence())
    monkeypatch.setitem(CARD, "cagr", 0.14)
    assert "CAGR edge" in "\n".join(precedence())
