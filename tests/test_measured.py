"""The link between a printed figure and the study that measured it.

WHAT THIS FILE IS DEFENDING. The standing instruction's fourth column is "the
measured cost of each level", and until H63 every one of those figures was a
LITERAL in a format string. H62 moved `KEEP_HI` from 0.80 to 0.70,
`scripts/stoptest.py` held its own copy of that constant which did not move,
and the rule card went on describing a rule nobody ships. Nothing failed,
because a number copied out of a study has no link back to the study.

So the tests that matter here are the REFUSALS. `load()` returning good numbers
is easy; what makes it worth having is that it declines the file when the file
measured something else, and says which constant moved. Each refusal below
corresponds to a case that has actually occurred in this repo.

AND THE POSITIVE CONTROL. A26's sine wave, A27's planted Fibonacci bump, A36's
Q0 martingale and A44's linter control are all the same discipline: a guard
that cannot fire proves nothing by not firing. `test_a_drifted_stamp_is_caught`
plants the drift and demands the refusal.
"""

from __future__ import annotations

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "scripts"))

from idxbot import measured as M                                  # noqa: E402

REAL = os.path.join(os.path.dirname(__file__), os.pardir, "reports",
                    "stoptest.json")


def _write(tmp_path, blob) -> str:
    p = os.path.join(str(tmp_path), "stoptest.json")
    with open(p, "w") as f:
        json.dump(blob, f)
    return p


def _stamp():
    import rules                                                # noqa: PLC0415
    return {"ENTRY_HI": rules.ENTRY_HI, "ENTRY_VOL": rules.ENTRY_VOL,
            "KEEP_HI": rules.KEEP_HI, "KEEP_VOL": rules.KEEP_VOL}


def _arms(**over):
    base = {"arm": M.BASE_ARM, M.F_CAGR: 0.10, M.F_DD: -0.40,
            M.F_WORST: -0.80}
    ship = {"arm": M.SHIPPED_ARM, M.F_CAGR: 0.12, M.F_DD: -0.30,
            M.F_WORST: -0.40}
    ship.update(over)
    return [base, ship]


# --------------------------------------------------------------- THE REFUSALS

def test_a_drifted_stamp_is_caught_and_names_the_constant(tmp_path):
    """THE POSITIVE CONTROL. This is the case that actually happened, planted."""
    stamp = _stamp()
    stamp["KEEP_HI"] = stamp["KEEP_HI"] + 0.10
    m = M.load(_write(tmp_path, {"rule": stamp, "arms": _arms()}))
    assert not m.fresh
    assert "DIFFERENT rule" in m.provenance
    assert "KEEP_HI" in m.provenance
    assert m.arms is M.FALLBACK


def test_a_matching_stamp_is_accepted(tmp_path):
    """The other half of the control: it must also NOT fire when it should not,
    or the refusal above would be indistinguishable from a broken reader."""
    m = M.load(_write(tmp_path, {"rule": _stamp(), "arms": _arms()}))
    assert m.fresh
    assert m.get(M.SHIPPED_ARM) == pytest.approx(0.12)


def test_a_missing_file_falls_back_rather_than_raising(tmp_path):
    m = M.load(os.path.join(str(tmp_path), "nope.json"))
    assert not m.fresh and "unavailable" in m.provenance


def test_an_unstamped_file_is_refused_not_crashed_on(tmp_path):
    """The result file used to be a BARE LIST of arms with no record of the
    rule that produced them — the exact case this module exists to catch, so it
    must report it rather than raise. A `try` around the load alone did not
    cover `.get` on a list."""
    m = M.load(_write(tmp_path, _arms()))
    assert not m.fresh and "no rule stamp" in m.provenance


def test_an_empty_stamp_is_refused(tmp_path):
    m = M.load(_write(tmp_path, {"rule": {}, "arms": _arms()}))
    assert not m.fresh and "empty rule stamp" in m.provenance


def test_a_missing_arm_is_refused(tmp_path):
    m = M.load(_write(tmp_path, {"rule": _stamp(),
                                 "arms": [{"arm": M.BASE_ARM, M.F_CAGR: 0.1,
                                           M.F_DD: -0.4}]}))
    assert not m.fresh and M.SHIPPED_ARM in m.provenance


def test_a_schema_change_is_refused_by_field_name(tmp_path):
    """A first reader assumed `cagr`/`maxdd`; the writer emits `cagr_med`/`dd`.
    The KeyError was the GOOD outcome — a silently wrong value would have been
    the bad one — and this keeps that failure loud."""
    bad = [{"arm": M.BASE_ARM, "cagr": 0.1, "maxdd": -0.4},
           {"arm": M.SHIPPED_ARM, "cagr": 0.12, "maxdd": -0.3}]
    m = M.load(_write(tmp_path, {"rule": _stamp(), "arms": bad}))
    assert not m.fresh and M.F_CAGR in m.provenance


def test_every_refusal_says_the_figures_are_not_current(tmp_path):
    """A fallback the reader is TOLD about is a different object from a stale
    number presented as current, and that is the only thing separating
    `FALLBACK` from the literals it replaced."""
    cases = [os.path.join(str(tmp_path), "nope.json"),
             _write(tmp_path, {"rule": {}, "arms": _arms()})]
    for p in cases:
        m = M.load(p)
        assert not m.fresh
        assert "last known-good" in m.provenance


# ----------------------------------------------------------- THE ARITHMETIC

def test_the_sign_convention_is_negative_means_costs(tmp_path):
    """The direction has already flipped once: on the pre-H62 buffer the
    shipped levels showed a CAGR GAIN and on the shipped buffer they cost."""
    m = M.load(_write(tmp_path, {"rule": _stamp(), "arms": _arms()}))
    assert m.cost(M.SHIPPED_ARM) == pytest.approx(0.02)
    assert M.noun(m.cost(M.SHIPPED_ARM)) == "edge"
    worse = _arms(**{M.F_CAGR: 0.08})
    m2 = M.load(_write(tmp_path, {"rule": _stamp(), "arms": worse}))
    assert m2.cost(M.SHIPPED_ARM) == pytest.approx(-0.02)
    assert M.noun(m2.cost(M.SHIPPED_ARM)) == "cost"


def test_the_noun_is_never_typed_next_to_the_number():
    """Prose that says "edge" where the number says "cost" is the drift this
    module exists to end, so the word comes from the sign."""
    assert M.noun(+0.01) == "edge"
    assert M.noun(-0.01) == "cost"
    assert M.noun(float("nan")) == "effect"


def test_a_missing_arm_yields_nan_rather_than_raising():
    m = M.load(REAL) if os.path.exists(REAL) else M._fallback("x")
    assert m.get("no such arm") != m.get("no such arm")
    assert m.cost("no such arm") != m.cost("no such arm")


def test_the_middle_of_a_family_is_defined_on_its_LEVELS():
    """H56 chose STOP = 0.20 as "the middle of the family and deliberately not
    its argmax"; defining it on the outcomes would be an argmax wearing a
    different word, so `middle` may not see a result at all."""
    assert M.middle(M.STOP_ARMS) == pytest.approx(0.20)
    assert M.middle([0.1, 0.2, 0.3]) == pytest.approx(0.2)
    #  Even-length: deterministic, and toward the lower half.
    assert M.middle([0.1, 0.2, 0.3, 0.4]) == pytest.approx(0.2)
    assert M.middle([]) != M.middle([])


def test_monotone_is_checked_not_asserted():
    assert M.is_monotone([3.0, 2.0, 1.0])
    assert M.is_monotone([1.0, 1.0])
    assert not M.is_monotone([1.0, 2.0])


# ------------------------------------------------------- THE SHIPPED SURFACES

@pytest.mark.skipif(not os.path.exists(REAL), reason="no result file here")
def test_the_real_file_measures_the_rule_that_ships():
    """If this fails, `scripts/stoptest.py` needs re-running — that is the
    whole alarm, and it is the one H62 did not have."""
    m = M.load(REAL)
    assert m.fresh, m.provenance


@pytest.mark.skipif(not os.path.exists(REAL), reason="no result file here")
def test_the_families_the_card_quotes_are_all_present():
    """The card prints a stop family and a take-profit curve. A missing arm
    would silently shorten the curve rather than fail, so it is asserted."""
    m = M.load(REAL)
    assert set(M.family(m, M.STOP_ARMS)) == set(M.STOP_ARMS)
    assert set(M.family_costs(m, M.TP_ARMS)) == set(M.TP_ARMS)
    assert m.get(M.HALF_ARM) == m.get(M.HALF_ARM)
    assert m.get(M.DAILY_ARM) == m.get(M.DAILY_ARM)


@pytest.mark.skipif(not os.path.exists(REAL), reason="no result file here")
def test_the_shipped_stop_and_target_are_arms_that_were_measured():
    """A level the card prints with no arm behind it has no measured cost, and
    the fourth column would be empty for it."""
    import rules                                                # noqa: PLC0415
    assert rules.STOP in M.STOP_ARMS
    assert rules.TP in M.TP_ARMS
    m = M.load(REAL)
    assert m.get(M.STOP_ARMS[rules.STOP]) == m.get(M.STOP_ARMS[rules.STOP])


@pytest.mark.skipif(not os.path.exists(REAL), reason="no result file here")
def test_S3s_predicted_null_still_holds_on_the_live_file():
    """S3: a target's cost is monotone in how tight it is. The card prints that
    word; here it is the assertion, so a file where it stopped being true fails
    the build instead of printing a false shape."""
    m = M.load(REAL)
    costs = M.family_costs(m, M.TP_ARMS)
    assert M.is_monotone(costs.values()), costs
    assert min(costs.values()) > 0, "a target that PAYS would overturn S3"


def test_the_card_reads_the_measurement_rather_than_typing_it():
    """The regression that started this: eleven percentage literals in printed
    strings. The level costs must come through `measured`."""
    src = open(os.path.join(os.path.dirname(__file__), os.pardir, "scripts",
                            "rules.py")).read()
    assert "measured.load()" in src
    for gone in ("6.84 points", "13.46% -> 12.76%", "-40.2% -> -33.8%",
                 "12.97%/yr against 13.46%", "CAGR 13.46% -> 4.29%"):
        assert gone not in src, f"still typed: {gone}"


def test_both_surfaces_share_one_reader():
    """Two readers of one file drift exactly the way two copies of one constant
    did, which is the bug that produced this module."""
    src = open(os.path.join(os.path.dirname(__file__), os.pardir, "scripts",
                            "today.py")).read()
    assert "measured.load(STOPTEST)" in src
    assert "blob.get(\"rule\")" not in src, "today.py re-implements the load"


# ------------------------------------------- THE PROSE TABLES ABOVE THE CODE

#: Every row of the three tables in `rules.py`'s module docstring, mapped onto
#: the arm that produced it. A19 and A44 both record the same failure — the
#: code gets corrected and the sentence above it does not — and the docstring
#: here is a narrative copy of exactly the numbers the card now computes. It
#: cannot be computed (it is a docstring), so it is CHECKED.
DOC_ROWS = {
    "no stop": M.BASE_ARM,
    "stop -10%": M.STOP_ARMS[0.10], "stop -15%": M.STOP_ARMS[0.15],
    "stop -20%": M.STOP_ARMS[0.20], "stop -25%": M.STOP_ARMS[0.25],
    "stop -30%": M.STOP_ARMS[0.30],
    "+20%": M.TP_ARMS[0.20], "+30%": M.TP_ARMS[0.30],
    "+50%": M.TP_ARMS[0.50], "+75%": M.TP_ARMS[0.75],
    "+100%": M.TP_ARMS[1.00], "+150%": "+ take-profit 150%",
    "sell HALF at +50%": "+ sell HALF at +50%",
    "sell THIRD at +100%": "+ sell THIRD at +100%",
    "sell HALF at +100%": M.HALF_ARM,
    "BASE, band only": M.BASE_ARM,
    "SHIPPED, stop + half at +100%": M.SHIPPED_ARM,
}


def _doc_cagr():
    """`{row label: CAGR as printed}` from `rules.py`'s docstring tables."""
    import re                                                   # noqa: PLC0415
    import rules                                                # noqa: PLC0415
    out = {}
    for line in (rules.__doc__ or "").splitlines():
        m = re.match(r"^\s{4}(\S.*?)\s{2,}(-?\d+\.\d+)%", line)
        if m and m.group(1).strip() in DOC_ROWS:
            out[m.group(1).strip()] = float(m.group(2)) / 100.0
    return out


@pytest.mark.skipif(not os.path.exists(REAL), reason="no result file here")
def test_the_docstring_tables_match_the_measurement_they_narrate():
    """THE SENTENCE ABOVE THE CODE IS CHECKED TOO.

    A19 records `brief.news_caveat()` being corrected while the module
    docstring above it went on asserting the retracted claim, and A44 found the
    same shape in fourteen more places. The card's figures are computed now;
    this docstring is a hand-written copy of the same run, so a re-run that
    moves them must fail the build rather than leave a stale narrative sitting
    above correct output.
    """
    m = M.load(REAL)
    doc = _doc_cagr()
    assert len(doc) >= 12, f"docstring tables not parsed: {sorted(doc)}"
    bad = []
    for label, got in doc.items():
        want = m.get(DOC_ROWS[label])
        if want != want or abs(want - got) > 5e-4:
            bad.append(f"{label}: doc {got:.4f} vs file {want:.4f}")
    assert not bad, "rules.py docstring has drifted from stoptest.json:\n" + \
                    "\n".join(bad)


@pytest.mark.skipif(not os.path.exists(REAL), reason="no result file here")
def test_the_docstring_check_can_actually_fire():
    """The positive control: a guard that cannot fire proves nothing by not
    firing (A26's sine wave, A27's planted bump, A36's Q0, A44's linter)."""
    m = M.load(REAL)
    planted = {"no stop": m.get(M.BASE_ARM) + 0.05}
    bad = [k for k, v in planted.items()
           if abs(m.get(DOC_ROWS[k]) - v) > 5e-4]
    assert bad == ["no stop"]


def test_a_fresh_file_without_the_index_declines_rather_than_borrowing(tmp_path):
    """The stored IHSG figure belongs to the H63 run's window. Pairing it with
    a NEWER run's arms would price a benchmark over a window it was not
    measured over — A19's error class — so a fresh file that lacks the field
    hands the caller `None` and the caller says so."""
    m = M.load(_write(tmp_path, {"rule": _stamp(), "arms": _arms()}))
    assert m.fresh and m.index_cagr is None
    m2 = M.load(_write(tmp_path, {"rule": _stamp(), "arms": _arms(),
                                  "index_cagr": 0.09}))
    assert m2.index_cagr == pytest.approx(0.09)


def test_an_unreadable_file_still_carries_the_last_known_benchmark(tmp_path):
    """The other side of the same split: when nothing is fresh, everything is
    the labelled fallback together, benchmark included."""
    m = M.load(os.path.join(str(tmp_path), "nope.json"))
    assert not m.fresh and m.index_cagr == pytest.approx(M.FALLBACK_INDEX)


def test_the_card_declines_the_benchmark_it_cannot_verify():
    src = open(os.path.join(os.path.dirname(__file__), os.pardir, "scripts",
                            "rules.py")).read()
    assert "m.index_cagr is not None" in src
    assert "not in" in src and "so it is not quoted here" in src


def test_the_writer_persists_the_span_and_the_benchmark():
    """A benchmark a reader cannot check against the arms' own window is the
    comparison A19 records as manufacturing results."""
    src = open(os.path.join(os.path.dirname(__file__), os.pardir, "scripts",
                            "stoptest.py")).read()
    assert '"index_cagr": float(index_cagr(a, b))' in src
    assert '"span": [str(a), str(b)]' in src


@pytest.mark.skipif(not os.path.exists(REAL), reason="no result file here")
def test_the_docstring_benchmark_row_matches_the_file_when_it_carries_one():
    """The IHSG row of the WHAT SHIPS table maps to no arm, so the check above
    skips it — and an unchecked row beside checked ones is exactly where drift
    hides. It is checked here against the field the writer now persists, and
    skipped only while the file predates that field."""
    import re                                                   # noqa: PLC0415
    import rules                                                # noqa: PLC0415
    m = M.load(REAL)
    if m.index_cagr is None:
        pytest.skip("stoptest.json predates the index_cagr field")
    hit = re.search(r"IHSG, total return, same span\s+(-?\d+\.\d+)%",
                    rules.__doc__ or "")
    assert hit, "the WHAT SHIPS table lost its benchmark row"
    assert abs(float(hit.group(1)) / 100.0 - m.index_cagr) < 5e-4, \
        f"doc {hit.group(1)}% vs file {m.index_cagr:.4%}"
