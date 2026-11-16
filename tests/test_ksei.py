"""KSEI holding composition: the parser refuses what it cannot vouch for."""

from __future__ import annotations

import datetime as dt
import io
import os
import sys
import zipfile

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

from idxbot.data import ksei                                    # noqa: E402

HEAD = "|".join(ksei.HEADER)


def _row(code, local, foreign, sec=None, price=1000, day="30-OCT-2026",
         typ="EQUITY", local_total=None, foreign_total=None):
    lt = sum(local) if local_total is None else local_total
    ft = sum(foreign) if foreign_total is None else foreign_total
    sec = lt + ft if sec is None else sec
    return "|".join([day, code, typ, str(sec), str(price)]
                    + [str(x) for x in local] + [str(lt)]
                    + [str(x) for x in foreign] + [str(ft)])


L1 = [10, 500, 5, 0, 300, 40, 2, 1, 0]     # IS CP PF IB ID MF SC FD OT
F1 = [0, 100, 20, 30, 1, 50, 2, 0, 0]


def _text(*rows):
    return "\n".join([HEAD, *rows]) + "\n"


def _zip(text, name="Balancepos20261030.txt",
         stamp=(2026, 11, 2, 6, 0, 0)) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr(zipfile.ZipInfo(name, date_time=stamp), text)
    return buf.getvalue()


# --------------------------------------------------------------------------- #
# parsing
# --------------------------------------------------------------------------- #

def test_parse_reads_a_well_formed_file():
    df = ksei.parse(_text(_row("AADI", L1, F1), _row("BOND", L1, F1,
                                                      typ="CORPORATE BOND")))
    assert list(df["ticker"]) == ["AADI"]          # equity only by default
    r = df.iloc[0]
    assert r["local_total"] == sum(L1) and r["foreign_total"] == sum(F1)
    assert r["local_ID"] == 300 and r["foreign_CP"] == 100
    assert r["month_end"] == pd.Timestamp("2026-10-30")
    assert len(ksei.parse(_text(_row("AADI", L1, F1),
                                _row("BOND", L1, F1, typ="CORPORATE BOND")),
                          equity_only=False)) == 2


def test_a_broken_identity_refuses_the_whole_file():
    """The positive control: the check must be able to fire."""
    good = _row("AADI", L1, F1)
    bad = _row("BBCA", L1, F1, local_total=sum(L1) + 1)
    with pytest.raises(ksei.ParseError, match="identity"):
        ksei.parse(_text(good, bad))
    badf = _row("BBCA", L1, F1, foreign_total=sum(F1) - 7)
    with pytest.raises(ksei.ParseError, match="identity"):
        ksei.parse(_text(good, badf))


def test_a_shifted_column_is_refused_not_misread():
    shifted = _row("AADI", L1, F1) + "|0"
    with pytest.raises(ksei.ParseError):
        ksei.parse(_text(shifted))


def test_an_unexpected_header_is_refused():
    with pytest.raises(ksei.ParseError, match="header"):
        ksei.parse(_text(_row("AADI", L1, F1)).replace("Local ID", "Local XX"))


def test_two_state_dates_in_one_file_is_refused():
    with pytest.raises(ksei.ParseError, match="state dates"):
        ksei.parse(_text(_row("AADI", L1, F1),
                         _row("BBCA", L1, F1, day="29-SEP-2026")))


def test_read_zip_returns_the_creation_stamp():
    text, member, stamp = ksei.read_zip(_zip(_text(_row("AADI", L1, F1))))
    assert member == "Balancepos20261030.txt"
    assert stamp == dt.datetime(2026, 11, 2, 6, 0, 0)
    assert text.startswith("Date|Code")


# --------------------------------------------------------------------------- #
# point in time
# --------------------------------------------------------------------------- #

SESS = pd.to_datetime(["2026-10-29", "2026-10-30", "2026-11-02",
                       "2026-11-03", "2026-11-04", "2026-11-05"])


def test_available_on_is_strictly_after_the_stamp_day():
    stamp = dt.datetime(2026, 11, 2, 6, 0)
    assert ksei.available_on(stamp, SESS, lag=1) == pd.Timestamp("2026-11-03")
    assert ksei.available_on(stamp, SESS, lag=3) == pd.Timestamp("2026-11-05")
    # never the state date, never the stamp's own day
    for lag in (1, 2, 3):
        assert ksei.available_on(stamp, SESS, lag) > pd.Timestamp(stamp.date())
    assert ksei.available_on(stamp, SESS, lag=9) is None


def test_month_candidates_try_the_calendar_first():
    cal = [dt.date(2019, 12, d) for d in (23, 24, 26, 27, 30)]
    c = ksei.month_candidates(2019, 12, cal)
    assert c[0] == dt.date(2019, 12, 30)
    assert dt.date(2019, 12, 31) in c            # a weekday, tried after
    assert c.index(dt.date(2019, 12, 31)) > c.index(dt.date(2019, 12, 23))
    assert all(d.weekday() < 5 for d in ksei.month_candidates(2026, 2))


# --------------------------------------------------------------------------- #
# derived quantities
# --------------------------------------------------------------------------- #

def _panel(rows):
    return pd.DataFrame(rows, columns=["month_end", "ticker", "sec_num", "price",
                                       "local_ID", "local_total",
                                       "foreign_total"])


def _full(df):
    for c in ksei.LOCAL + ksei.FOREIGN:
        if c not in df:
            df[c] = 0
    df["foreign_IS"] = df["foreign_total"]
    df["local_OT"] = df["local_total"] - df["local_ID"]
    return df


def test_shares_are_against_held_units_not_sec_num():
    P = _full(_panel([[pd.Timestamp("2026-01-30"), "ABCD", 1000, 50,
                       100, 200, 50]]))
    S = ksei.shares(P)
    assert S["held"].iloc[0] == 250
    assert S["retail"].iloc[0] == pytest.approx(100 / 250)
    assert S["foreign"].iloc[0] == pytest.approx(50 / 250)
    assert S["coverage"].iloc[0] == pytest.approx(0.25)
    assert S["mcap"].iloc[0] == pytest.approx(1000 * 50)


def test_changes_are_consecutive_and_within_ticker():
    m = pd.to_datetime(["2026-01-30", "2026-02-27", "2026-03-31", "2026-04-30"])
    rows = [[m[0], "AAAA", 100, 1, 10, 80, 20],
            [m[1], "AAAA", 100, 1, 20, 80, 20],
            # AAAA absent in March: April's change must NOT be a 2-month change
            [m[3], "AAAA", 100, 1, 50, 80, 20],
            [m[0], "BBBB", 100, 1, 40, 80, 20],
            [m[1], "BBBB", 100, 1, 30, 80, 20],
            [m[2], "BBBB", 100, 1, 30, 80, 20]]
    S = ksei.changes(ksei.shares(_full(_panel(rows))))
    a = S[S.ticker == "AAAA"].set_index("month_end")["d_retail"]
    assert np.isnan(a[m[0]])
    assert a[m[1]] == pytest.approx(0.1)
    assert np.isnan(a[m[3]])                     # the gap, refused
    b = S[S.ticker == "BBBB"].set_index("month_end")["d_retail"]
    assert b[m[1]] == pytest.approx(-0.1) and b[m[2]] == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# fetching: the host list, absence versus silence, the permanent cache
# --------------------------------------------------------------------------- #

def test_a_host_off_the_list_is_refused(tmp_path):
    with pytest.raises(ksei.HostNotAllowed):
        ksei.fetch_file("BalanceposEfek20261030.zip", hosts=[],
                        raw_dir=str(tmp_path), getter=lambda u: (200, b"PK"))


def test_a_redirect_is_absence(tmp_path):
    got = ksei.fetch_file("BalanceposEfek20171229.zip",
                          raw_dir=str(tmp_path), getter=lambda u: (302, b""),
                          delay=0)
    assert got is None


def test_silence_is_not_absence(tmp_path):
    def dead(u):
        raise ConnectionResetError("reset by peer")
    with pytest.raises(ConnectionError):
        ksei.fetch_file("BalanceposEfek20261030.zip", raw_dir=str(tmp_path),
                        getter=dead, retries=2, delay=0)


def test_a_404_is_absence_too(tmp_path):
    assert ksei.fetch_file("BalanceposEfek20171229.zip", raw_dir=str(tmp_path),
                           getter=lambda u: (404, b"<html>"), delay=0) is None


@pytest.mark.parametrize("code,body", [(503, b"busy"), (500, b""),
                                       (429, b"slow down"), (403, b"no"),
                                       (200, b"<html>maintenance</html>")])
def test_a_server_that_does_not_answer_is_not_absence(tmp_path, code, body):
    """Second review, 2026-11-16: the first version returned None for ANY
    non-200, so a 503 was recorded exactly like a file that does not exist."""
    with pytest.raises(ConnectionError, match=str(code)):
        ksei.fetch_file("BalanceposEfek20261030.zip", raw_dir=str(tmp_path),
                        getter=lambda u: (code, body), retries=2, delay=0)


def test_a_transient_error_is_retried_into_the_file(tmp_path):
    blob = _zip(_text(_row("AADI", L1, F1)))
    answers = iter([(503, b""), (429, b""), (200, blob)])
    got = ksei.fetch_file("BalanceposEfek20261030.zip", raw_dir=str(tmp_path),
                          getter=lambda u: next(answers), retries=4, delay=0)
    assert got is not None and got.blob == blob


def test_a_missing_archive_month_breaks_the_chain():
    """If the WHOLE archive lacks March, April's change is a two-month change
    and must be NaN — the first version counted positions in the archive, so
    April looked consecutive to February."""
    m = pd.to_datetime(["2026-01-30", "2026-02-27", "2026-04-30"])
    rows = [[m[0], "AAAA", 100, 1, 10, 80, 20],
            [m[1], "AAAA", 100, 1, 20, 80, 20],
            [m[2], "AAAA", 100, 1, 50, 80, 20]]
    P = _full(_panel(rows))
    S = ksei.changes(ksei.shares(P))
    a = S.set_index("month_end")["d_retail"]
    assert a[m[1]] == pytest.approx(0.1) and np.isnan(a[m[2]])
    assert ksei.months_missing(P) == ["2026-03"]
    assert ksei.months_missing(P[P.month_end < m[2]]) == []


def test_the_manifest_identifies_content_not_only_the_zip(tmp_path):
    """The legacy host's 2018-01 copy has a different zip hash and the same
    text; a re-download must be checkable by content."""
    text = _text(_row("AADI", L1, F1))
    a = ksei.manifest_row(dt.date(2026, 10, 30), ksei.Fetched(
        "BalanceposEfek20261030.zip", _zip(text, stamp=(2026, 11, 2, 6, 0, 0)),
        True))
    b = ksei.manifest_row(dt.date(2026, 10, 30), ksei.Fetched(
        "BalanceposEfek20261030.zip", _zip(text, stamp=(2027, 4, 13, 12, 32, 10)),
        True))
    assert a["sha256"] != b["sha256"]
    assert (a["member_crc"], a["member_bytes"], a["text_sha256"]) == (
        b["member_crc"], b["member_bytes"], b["text_sha256"])
    assert a["created"] != b["created"]


def test_the_cache_is_read_without_touching_the_network(tmp_path):
    blob = _zip(_text(_row("AADI", L1, F1)))
    calls = []

    def once(u):
        calls.append(u)
        return 200, blob
    a = ksei.fetch_file("BalanceposEfek20261030.zip", raw_dir=str(tmp_path),
                        getter=once, delay=0)
    b = ksei.fetch_file("BalanceposEfek20261030.zip", raw_dir=str(tmp_path),
                        getter=once, delay=0)
    assert len(calls) == 1 and not a.from_cache and b.from_cache
    assert a.blob == b.blob


def test_build_panel_checks_the_name_against_the_state_date(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "BalanceposEfek20261030.zip").write_bytes(
        _zip(_text(_row("AADI", L1, F1))))
    P, man = ksei.build_panel(str(raw))
    assert len(P) == 1 and man["sha256"].str.len().iloc[0] == 64
    assert man["created"].iloc[0] == pd.Timestamp("2026-11-02 06:00")
    (raw / "BalanceposEfek20261127.zip").write_bytes(
        _zip(_text(_row("AADI", L1, F1))))        # says Oct inside, Nov outside
    with pytest.raises(ksei.ParseError, match="does not match"):
        ksei.build_panel(str(raw))


# --------------------------------------------------------------------------- #
# republication: the repo is public
# --------------------------------------------------------------------------- #

ROOT = os.path.join(os.path.dirname(__file__), os.pardir)


def test_the_holdings_are_gitignored_and_only_the_manifest_is_tracked():
    ign = open(os.path.join(ROOT, ".gitignore"), encoding="utf-8").read()
    assert "data/cache/" in ign.splitlines()
    assert ksei.PANEL_PATH.startswith(os.path.join("data", "cache"))
    assert ksei.RAW_DIR.startswith(os.path.join("data", "cache"))
    assert not ksei.MANIFEST_PATH.startswith(os.path.join("data", "cache"))


def test_the_manifest_carries_no_holdings(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "BalanceposEfek20261030.zip").write_bytes(
        _zip(_text(_row("AADI", L1, F1))))
    _, man = ksei.build_panel(str(raw))
    held = set(ksei.LOCAL + ksei.FOREIGN + ["local_total", "foreign_total",
                                            "ticker", "price", "sec_num"])
    assert not held & set(man.columns)


def test_fractional_fund_units_pass_but_fractional_shares_do_not():
    """MUTUAL FUND units are fractional; their float noise is not a break."""
    fund = ("29-OCT-2021|710902|MUTUAL FUND|100000000|1|52541809.7586|"
            "5179237.4106|0|0|0|0|0|0|0|57721047.1692|0|0|0|0|0|0|0|0|0|0")
    head = "|".join(ksei.HEADER)
    eq = _row("AADI", L1, F1, day="29-OCT-2021")
    df = ksei.parse("\n".join([head, eq, fund]), equity_only=False)
    assert len(df) == 2
    assert ksei.parse("\n".join([head, eq, fund]))["local_total"].dtype == "int64"
    bad = eq.replace("|300|", "|300.5|", 1)     # half a share of local ID
    bad = bad.replace(f"|{sum(L1)}|", f"|{sum(L1) + 0.5}|", 1)
    with pytest.raises(ksei.ParseError, match="fractional"):
        ksei.parse("\n".join([head, bad]))
