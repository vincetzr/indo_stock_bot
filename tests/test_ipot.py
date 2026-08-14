"""Tests for the IndoPremier broker-summary parser.

Every fixture is a byte-for-byte capture of a real response, so these tests
fail if IndoPremier changes its markup - which is the point. A silent layout
change that made the parser return zeros would otherwise look exactly like a
quiet day.

The important assertions are not "the parser returns ten rows". They are the
ones that would have caught a real mis-parse:

  * value == lots x 100 x average, checked per row against the source's own
    published average price;
  * regular-board totals reconcile to the exchange tape (BBCA 832,077 lots on
    2026-08-13 against Yahoo's 832,080);
  * a row's buyer and its seller are independent rankings and must never be
    read as one broker's two sides.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from idxbot.data.ipot import (
    BOARDS,
    BROKER_CLASSES,
    DISPLAY_ROUNDING_BOUND,
    broker_classes,
    IpotBrokerSummary,
    consistency,
    foreign_flags,
    is_abbreviated,
    parse_number,
    is_truncated,
    parse_table,
    parse_totals,
    truncation_bias,
)

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")
DATE = pd.Timestamp("2026-08-13")


def _fixture(name: str) -> str:
    with open(os.path.join(FIXTURES, name), encoding="utf-8") as fh:
        return fh.read()


@pytest.fixture(scope="module")
def bbca_html() -> str:
    return _fixture("ipot_bbca_20260813.html")


@pytest.fixture(scope="module")
def unvr_html() -> str:
    return _fixture("ipot_unvr_20260813.html")


@pytest.fixture(scope="module")
def range_html() -> str:
    """BBCA over 2026-07-01..2026-08-13: big enough that lots get abbreviated."""
    return _fixture("ipot_bbca_range.html")


@pytest.fixture(scope="module")
def bbca(bbca_html: str) -> pd.DataFrame:
    return parse_table(bbca_html, "BBCA", DATE)


# ---------------------------------------------------------------------------
# number parsing
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("text,expected", [
    ("771,597", 771_597.0),
    ("3.4 M", 3_400_000.0),
    ("699.9 B", 699_900_000_000.0),
    ("3.2 T", 3_200_000_000_000.0),
    ("-2.5 T", -2_500_000_000_000.0),
    ("802.0 M", 802_000_000.0),
    ("9,179 ", 9_179.0),
    ("0", 0.0),
    ("1.5 K", 1_500.0),
    ("450 rb", 450_000.0),
])
def test_parse_number(text, expected):
    assert parse_number(text) == pytest.approx(expected)


@pytest.mark.parametrize("text", ["", "-", "--", "n/a", "abc", "12 QQ"])
def test_parse_number_returns_nan_rather_than_raising(text):
    assert np.isnan(parse_number(text))


def test_multiplier_scale_is_fixed_by_the_tables_own_arithmetic():
    """M must be 1e6, not "miliar" (1e9).

    The footer of the BBCA range fixture reports 18.4 M lots, avg 9,120 and a
    total value of 16.8 T. Only M=1e6 makes that identity hold, so the scale is
    pinned by the data and not by a guess about Indonesian usage.
    """
    lots, avg, value = parse_number("18.4 M"), parse_number("9,120"), parse_number("16.8 T")
    assert lots * 100 * avg == pytest.approx(value, rel=0.01)


def test_is_abbreviated_marks_only_suffixed_cells():
    assert is_abbreviated("3.4 M")
    assert is_abbreviated("699.9 B")
    assert not is_abbreviated("771,597")
    assert not is_abbreviated("9,179 ")


# ---------------------------------------------------------------------------
# table parsing
# ---------------------------------------------------------------------------
def test_parses_real_table(bbca):
    assert not bbca.empty
    assert set(bbca["ticker"]) == {"BBCA"}
    assert set(bbca["date"]) == {DATE}
    assert bbca["broker"].is_unique
    assert bbca["broker"].str.fullmatch(r"[A-Z]{2}").all()


def test_at_most_ten_brokers_per_side(bbca):
    """The source publishes a top-10, so anything more means rows were duplicated."""
    assert (bbca["buy_lot"] > 0).sum() <= 10
    assert (bbca["sell_lot"] > 0).sum() <= 10


def test_a_row_is_two_independent_rankings_not_one_broker(bbca):
    """The buy and sell columns rank separately.

    If the parser paired them positionally, every broker would carry both a buy
    and a sell leg. In the real fixture some appear on only one side, which is
    the observable signature of correct unstacking.
    """
    one_sided = ((bbca["buy_lot"] == 0) | (bbca["sell_lot"] == 0)).sum()
    assert one_sided > 0


def test_value_equals_lots_times_hundred_times_average(bbca):
    """The source's own redundancy, used as the acceptance test.

    Lots, value and average price over-determine each other. The bound is
    derived, not tuned: an abbreviated value carries one decimal place, so it
    is rounded to +/-0.05 of its mantissa and the worst relative error is
    0.05/mantissa = 5%. Reading the wrong column misses by orders of magnitude,
    so nothing real ever lands between the two.
    """
    checks = consistency(bbca)
    assert not checks.empty
    assert checks["rel_err"].max() < DISPLAY_ROUNDING_BOUND
    assert checks["rel_err"].median() < 0.01


def test_display_rounding_bound_is_the_arithmetic_worst_case():
    """Guards the constant against being quietly relaxed to make a test pass."""
    worst_mantissa = 1.0
    assert DISPLAY_ROUNDING_BOUND == pytest.approx(0.05 / worst_mantissa)


def test_totals_are_recovered_from_the_footer(bbca_html):
    totals = parse_totals(bbca_html)
    assert totals["tlot"] == pytest.approx(832_077, rel=1e-6)
    assert totals["tval"] > 0
    assert np.isfinite(totals["fnval"])          # net foreign, signed
    assert 5_000 < totals["avg"] < 8_000         # BBCA traded ~6,300 that day


def test_regular_board_total_reconciles_to_the_exchange_tape(bbca_html):
    """832,077 lots parsed here; Yahoo reported 83,208,000 shares that session.

    Three lots apart in eight hundred thousand. This is the check that decides
    whether the source is the real rekap broker or a lookalike, and it is
    pinned to a specific date so it keeps meaning something.
    """
    parsed_lots = parse_totals(bbca_html)["tlot"]
    tape_lots = 83_208_000 / 100
    assert abs(parsed_lots - tape_lots) / tape_lots < 1e-4


def test_top_ten_covers_most_but_not_all_of_the_tape(bbca, bbca_html):
    """A top-10 cannot sum to 100%, and if it does the totals were mis-parsed."""
    total = parse_totals(bbca_html)["tlot"]
    for side in ("buy", "sell"):
        share = bbca[f"{side}_lot"].sum() / total
        assert 0.5 < share < 1.0


def test_rounded_lots_are_flagged_in_the_source_tag(range_html):
    """Precision loss is recorded rather than hidden.

    Summed over six weeks the desks move millions of lots, so the lot cells
    come back abbreviated and the frame is tagged ``ipot~``.
    """
    df = parse_table(range_html, "BBCA", DATE)
    assert not df.empty
    assert set(df["source"]) == {"ipot~"}
    assert (df["buy_lot"].max() % 100_000) == 0, "expected a visibly rounded figure"


def test_single_session_lots_are_exact_where_a_range_query_would_round(bbca, range_html):
    """Why the provider fetches one day at a time rather than one range.

    A range query sums first and abbreviates afterwards, so six weeks of BBCA
    comes back as ``6.8 M`` while each individual session is exact to the lot.
    Narrow queries are more precise, not merely more polite.
    """
    ranged = parse_table(range_html, "BBCA", DATE)
    assert set(bbca["source"]) == {"ipot"}
    assert set(ranged["source"]) == {"ipot~"}


def test_the_marker_tracks_lots_only_and_so_can_be_absent(unvr_html):
    """UNVR's lot figures sit below the abbreviation threshold.

    This is the test that keeps the marker meaningful. Value is suffixed for
    almost every stock on the exchange, so flagging on value would tag every
    row ever fetched and carry no information. UNVR proves the marker can
    actually be off.
    """
    df = parse_table(unvr_html, "UNVR", DATE)
    assert not df.empty
    assert set(df["source"]) == {"ipot"}
    assert consistency(df)["rel_err"].max() < DISPLAY_ROUNDING_BOUND


def test_unknown_ticker_yields_empty_not_wrong(bbca):
    """An unknown code renders no table at all, and must not fall back."""
    empty = parse_table(_fixture("ipot_empty.html"), "ZZZZ", DATE)
    assert empty.empty
    assert list(empty.columns) == list(bbca.columns)
    assert parse_totals(_fixture("ipot_empty.html")) == {}


def test_parse_table_tolerates_garbage():
    assert parse_table("", "BBCA", DATE).empty
    assert parse_table("<html><body>nope</body></html>", "BBCA", DATE).empty


# ---------------------------------------------------------------------------
# foreign flags
# ---------------------------------------------------------------------------
def test_foreign_flags_are_recovered(bbca_html):
    flags = foreign_flags(bbca_html)
    assert flags, "no F/D classification found"
    assert all(re_ok(code) for code in flags)
    assert set(flags.values()) == {True, False}, "expected both foreign and domestic"


def re_ok(code: str) -> bool:
    return len(code) == 2 and code.isalpha() and code.isupper()


def test_yp_is_flagged_foreign_by_the_source(bbca_html):
    """Mirae is foreign-owned and retail-serving, and this is why two bases exist.

    The source classifies YP foreign. Treating that as "institutional foreign
    money is buying" is the trap ``foreign_basis`` exists to prevent, so the
    flag is asserted here to keep the distinction visible.
    """
    flags = foreign_flags(bbca_html)
    if "YP" in flags:
        assert flags["YP"] is True


def test_foreign_flags_agree_with_the_registry_where_both_are_confident(bbca_html):
    from idxbot.config import BrokerRegistry

    registry = BrokerRegistry.from_yaml(
        os.path.join(os.path.dirname(__file__), os.pardir, "config", "brokers.yaml"))
    # BQ/DR/TP are the documented, deliberate disagreements; see brokers.yaml.
    disputed = {"BQ", "DR", "TP"}
    for code, is_foreign in foreign_flags(bbca_html).items():
        if code in disputed or code not in registry:
            continue
        assert registry.get(code).foreign == is_foreign, code


# ---------------------------------------------------------------------------
# provider behaviour (no network)
# ---------------------------------------------------------------------------
def test_board_is_validated_at_construction():
    with pytest.raises(ValueError):
        IpotBrokerSummary(board="REGULAR")
    for board in BOARDS:
        assert IpotBrokerSummary(board=board).board == board


def test_regular_board_is_the_default():
    """All-board folds in negotiated crossings: GOTO 25.0M lots vs 196k regular."""
    assert IpotBrokerSummary().board == "RG"


def test_polite_delay_is_on_by_default():
    assert IpotBrokerSummary().delay >= 1.0


def test_fetch_is_bounded_by_max_days(monkeypatch):
    provider = IpotBrokerSummary(max_days=5, delay=0.0)
    seen = []

    def fake_day(ticker, day):
        seen.append(pd.Timestamp(day))
        return pd.DataFrame()

    monkeypatch.setattr(provider, "fetch_day", fake_day)
    provider.fetch("BBCA", "2020-01-01", "2026-08-13")
    assert len(seen) == 5
    assert max(seen) <= pd.Timestamp("2026-08-13")


def test_fetch_skips_weekends(monkeypatch):
    provider = IpotBrokerSummary(delay=0.0)
    seen = []
    monkeypatch.setattr(provider, "fetch_day",
                        lambda t, d: seen.append(pd.Timestamp(d)) or pd.DataFrame())
    provider.fetch("BBCA", "2026-08-08", "2026-08-14")   # Sat -> Fri
    assert all(d.weekday() < 5 for d in seen)


def test_empty_days_are_not_cached(monkeypatch, tmp_path):
    """Holidays and suspensions render an empty table.

    Caching that would write a permanent blank for a day that may simply have
    failed, and every later run would trust it.
    """
    from idxbot.data.cache import Cache

    cache = Cache(str(tmp_path))
    provider = IpotBrokerSummary(cache=cache, delay=0.0)
    monkeypatch.setattr(provider, "_get", lambda t, d: "<html>no table</html>")
    assert provider.fetch_day("BBCA", DATE).empty
    assert cache.read("ipot_broker", f"BBCA_{DATE:%Y%m%d}_RG") is None


def test_network_failure_degrades_to_empty(monkeypatch):
    provider = IpotBrokerSummary(delay=0.0)

    def boom(ticker, day):
        raise RuntimeError("connection reset")

    monkeypatch.setattr(provider, "_get", boom)
    assert provider.fetch_day("BBCA", DATE).empty


def test_cached_day_is_served_without_network(monkeypatch, tmp_path, bbca_html):
    from idxbot.data.cache import Cache

    cache = Cache(str(tmp_path))
    provider = IpotBrokerSummary(cache=cache, delay=0.0)
    monkeypatch.setattr(provider, "_get", lambda t, d: bbca_html)
    first = provider.fetch_day("BBCA", DATE)
    assert not first.empty

    def forbidden(ticker, day):
        raise AssertionError("network hit despite a warm cache")

    monkeypatch.setattr(provider, "_get", forbidden)
    second = provider.fetch_day("BBCA", DATE)
    pd.testing.assert_frame_equal(
        first.reset_index(drop=True), second.reset_index(drop=True), check_dtype=False)


def test_provider_registers_in_the_chain():
    from idxbot.config import load_config
    from idxbot.data.broker_summary import build_provider

    chain = build_provider(load_config(), ["ipot"])
    assert any(p.name == "ipot" for p in chain.providers)
    assert chain.is_real


def test_describe_names_the_board():
    assert "RG" in IpotBrokerSummary().describe()


# ---------------------------------------------------------------------------
# top-10 truncation: the limitation that can actually mislead
# ---------------------------------------------------------------------------
def _two_sided_day(date, rows):
    return pd.DataFrame([
        {"date": date, "ticker": "X", "broker": b, "buy_lot": bl, "buy_val": bl * 100.0,
         "buy_avg": 1.0, "sell_lot": sl, "sell_val": sl * 100.0, "sell_avg": 1.0,
         "source": "ipot"}
        for b, bl, sl in rows])


def test_complete_rekap_shows_no_truncation_bias():
    """The control case. Buys equal sells, so the diagnostic must read zero."""
    df = _two_sided_day(pd.Timestamp("2026-08-13"),
                        [("AA", 100, 0), ("BB", 0, 60), ("CC", 0, 40)])
    bias = truncation_bias(df)
    assert bias["cumulative_net_lot"] == pytest.approx(0.0)
    assert bias["imbalance_mean"] == pytest.approx(0.0)


def test_censored_side_produces_drift():
    """A broker seen only when buying accrues inventory it never actually held."""
    days = [_two_sided_day(pd.Timestamp("2026-08-1%d" % i),
                           [("AA", 100, 0), ("BB", 0, 20)]) for i in (1, 2, 3)]
    bias = truncation_bias(pd.concat(days, ignore_index=True))
    assert bias["cumulative_net_lot"] == pytest.approx(240.0)
    assert bias["imbalance_mean"] > 0.5


def test_real_data_is_measurably_truncated(bbca):
    """The shipped fixture must exhibit the bias, so the caveat stays earned."""
    assert is_truncated(bbca)
    bias = truncation_bias(bbca)
    assert bias["sessions"] == 1
    assert abs(bias["cumulative_net_lot"]) > 0, "a top-10 view cannot balance"


def test_is_truncated_is_false_for_other_sources(bbca):
    other = bbca.assign(source="csv")
    assert not is_truncated(other)
    assert not is_truncated(pd.DataFrame())


def test_truncation_bias_handles_empty():
    assert truncation_bias(pd.DataFrame()) == {}


# ---------------------------------------------------------------------------
# three-way broker classification (the class that was nearly missed)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def bumn_html() -> str:
    """A session whose top-10 includes state-owned desks (DX, CC, OD)."""
    return _fixture("ipot_bbca_bumn.html")


def test_source_classifies_into_three_buckets_not_two(bumn_html):
    """The regression test for the bug that mattered most.

    An earlier parser matched only ``text-foreign|text-local`` and silently
    dropped every ``text-bumn`` broker - which is not a rounding error, it is
    DX, CC, NI and OD, some of the largest desks on the exchange, vanishing
    from the classification entirely while still counting toward totals.
    """
    classes = broker_classes(bumn_html)
    assert set(classes.values()) == {"foreign", "bumn", "local"}
    assert classes["DX"] == "bumn"
    assert classes["CC"] == "bumn"


def test_every_class_is_a_known_bucket(bumn_html):
    assert set(broker_classes(bumn_html)).issubset(set(broker_classes(bumn_html)))
    assert all(k in BROKER_CLASSES for k in broker_classes(bumn_html).values())


def test_bumn_is_domestic_not_foreign(bumn_html):
    """State-owned is a separate axis, and must not leak into the foreign flag."""
    flags = foreign_flags(bumn_html)
    classes = broker_classes(bumn_html)
    for code, kind in classes.items():
        if kind == "bumn":
            assert flags[code] is False, f"{code} is state-owned, not foreign"


def test_registry_marks_exactly_the_observed_state_owned_houses():
    """Observed across 18 stocks x 10 dates with no code ever changing class."""
    from idxbot.config import BrokerRegistry

    registry = BrokerRegistry.from_yaml(
        os.path.join(os.path.dirname(__file__), os.pardir, "config", "brokers.yaml"))
    assert registry.state_owned_codes() == ["CC", "DX", "NI", "OD"]
    for code in registry.state_owned_codes():
        assert not registry.get(code).foreign


def test_dx_is_in_the_registry(bumn_html):
    """DX (Bahana) was absent entirely until real data surfaced it."""
    from idxbot.config import BrokerRegistry

    registry = BrokerRegistry.from_yaml(
        os.path.join(os.path.dirname(__file__), os.pardir, "config", "brokers.yaml"))
    assert "DX" in registry
    assert registry.get("DX").state_owned
    assert broker_classes(bumn_html)["DX"] == "bumn"


def test_every_broker_in_a_real_table_is_classified(bumn_html):
    """No code may appear in the table yet be missing from the classification."""
    df = parse_table(bumn_html, "BBCA", pd.Timestamp("2026-07-15"))
    classes = broker_classes(bumn_html)
    assert set(df["broker"]) == set(classes)
