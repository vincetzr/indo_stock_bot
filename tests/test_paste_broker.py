"""Tests for the pasted-table route.

`to_number` gets the most tests because it is the only place in this repo where
a bug turns a billion rupiah into a thousand and says nothing. Indonesian and
English number conventions are mirror images - 1.234,56 against 1,234.56 - and
reading one as the other is a 1000x error that no downstream check would catch.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "scripts"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

from paste_broker import (assign_columns, parse_table,           # noqa: E402
                          split_row, to_number)


# --------------------------------------------------------------------------- #
# numbers, in both conventions
# --------------------------------------------------------------------------- #
def test_indonesian_thousands_and_decimal():
    assert to_number("1.234.567,89") == pytest.approx(1234567.89)


def test_english_thousands_and_decimal():
    assert to_number("1,234,567.89") == pytest.approx(1234567.89)


def test_indonesian_thousands_without_decimals():
    assert to_number("120.000") == pytest.approx(120000.0)
    assert to_number("7.560.000.000") == pytest.approx(7.56e9)


def test_a_lone_comma_with_two_digits_is_a_decimal():
    assert to_number("6.300,50") == pytest.approx(6300.50)
    assert to_number("1234,56") == pytest.approx(1234.56)


def test_a_lone_comma_with_three_digits_is_thousands():
    assert to_number("1,234") == pytest.approx(1234.0)


def test_a_plain_integer_survives():
    assert to_number("6300") == pytest.approx(6300.0)


def test_parentheses_mean_negative():
    assert to_number("(1.500)") == pytest.approx(-1500.0)


def test_currency_and_spaces_are_stripped():
    assert to_number("Rp 1.500.000") == pytest.approx(1500000.0)


def test_dashes_and_blanks_are_not_numbers():
    for t in ("", "  ", "-", "--", "n/a", "abc"):
        assert to_number(t) is None


def test_the_two_conventions_never_collide():
    """The decisive rule: whichever separator comes LAST is the decimal mark."""
    assert to_number("1.234,56") == pytest.approx(1234.56)
    assert to_number("1,234.56") == pytest.approx(1234.56)


# --------------------------------------------------------------------------- #
# splitting
# --------------------------------------------------------------------------- #
def test_tabs_split():
    assert split_row("BK\t100\t200") == ["BK", "100", "200"]


def test_runs_of_spaces_split_but_single_spaces_do_not():
    assert split_row("BK   100   200") == ["BK", "100", "200"]
    assert split_row("PT ABC   100") == ["PT ABC", "100"]


def test_pipes_split():
    assert split_row("| BK | 100 | 200 |") == ["BK", "100", "200"]


# --------------------------------------------------------------------------- #
# finding the broker rows in a messy paste
# --------------------------------------------------------------------------- #
TABLE = """Broker Summary BBCA  20 Agustus 2026
Kode   B.Lot     B.Val           B.Avg   S.Lot     S.Val           S.Avg
BK     120.000   7.560.000.000   6.300   90.000    5.670.000.000   6.300
YP     95.000    5.985.000.000   6.300   110.000   6.930.000.000   6.300
"""


def test_header_and_title_lines_are_ignored():
    got = parse_table(TABLE)
    assert list(got["broker"]) == ["BK", "YP"]


def test_the_numbers_survive_the_round_trip():
    got = assign_columns(parse_table(TABLE))
    bk = got[got["broker"] == "BK"].iloc[0]
    assert bk["buy_lot"] == pytest.approx(120000.0)
    assert bk["buy_val"] == pytest.approx(7.56e9)
    assert bk["sell_lot"] == pytest.approx(90000.0)


def test_an_empty_paste_gives_an_empty_frame():
    assert parse_table("").empty
    assert parse_table("just some prose with no table").empty


def test_a_three_letter_broker_code_is_accepted():
    got = parse_table("AKS   100   200   300   400   500   600")
    assert list(got["broker"]) == ["AKS"]


def test_a_too_narrow_table_is_refused_rather_than_guessed():
    """Fewer than four numbers cannot carry both sides, so it must not invent one."""
    assert assign_columns(parse_table("BK   100   200")) is None


def test_a_four_column_layout_maps_lot_and_average():
    got = assign_columns(parse_table("BK   100   6.300   200   6.400"))
    assert got is not None
    assert got.iloc[0]["buy_lot"] == pytest.approx(100.0)
    assert got.iloc[0]["sell_lot"] == pytest.approx(200.0)


def test_every_canonical_column_is_present_even_when_absent_from_the_paste():
    got = assign_columns(parse_table("BK   100   6.300   200   6.400"))
    for c in ("buy_lot", "buy_val", "buy_avg", "sell_lot", "sell_val", "sell_avg"):
        assert c in got.columns


# --------------------------------------------------------------------------- #
# HTML, and the 1000x bug it caused
# --------------------------------------------------------------------------- #
from paste_broker import html_to_text                              # noqa: E402

HTML = """<table>
<tr><th>Kode</th><th>B.Lot</th><th>B.Val</th><th>B.Avg</th>
    <th>S.Lot</th><th>S.Val</th><th>S.Avg</th></tr>
<tr><td>BK</td><td>120.000</td><td>7.560.000.000</td><td>6.300</td>
    <td>90.000</td><td>5.670.000.000</td><td>6.300</td></tr>
</table>"""


def test_html_cells_keep_the_indonesian_thousands_dot():
    """Regression: pandas.read_html turned "120.000" into the float 120.0 by
    reading the thousands dot as a decimal point - a silent 1000x error."""
    got = assign_columns(parse_table(html_to_text(HTML)))
    assert got is not None
    assert got.iloc[0]["buy_lot"] == pytest.approx(120000.0)
    assert got.iloc[0]["buy_val"] == pytest.approx(7.56e9)


def test_html_rows_become_lines_and_cells_become_tabs():
    txt = html_to_text(HTML)
    assert "\t" in txt
    assert len([l for l in txt.splitlines() if l.strip()]) >= 2


def test_script_and_style_blocks_are_dropped():
    noisy = "<style>td{color:red}</style><table><tr><td>BK</td>" \
            "<td>1.000</td><td>2.000</td><td>3.000</td><td>4.000</td></tr></table>"
    txt = html_to_text(noisy)
    assert "color" not in txt
    assert "BK" in txt


def test_html_entities_are_decoded():
    assert "&nbsp;" not in html_to_text("<td>a&nbsp;b</td>")


def test_empty_cells_are_dropped_not_shifted():
    """A blank cell must not slide the buy column into the sell column."""
    h = "<tr><td>BK</td><td></td><td>1.000</td></tr>"
    assert html_to_text(h).split("\t") == ["BK", "1.000"]


# --------------------------------------------------------------------------- #
# the real Stockbit layout: magnitudes, two brokers per row, val/lot/avg
# --------------------------------------------------------------------------- #
from paste_broker import (broker_positions, infer_order,           # noqa: E402
                          parse_sides, sides_to_frame)

ACES = """Buy\tB.Val\tB.Lot\tB.Avg\tSell\tS.Val\tS.Lot\tS.Avg
XL\t840.9M\t24.2K\t349\tRF\t2.1B\t60.6K\t347
XA\t589.7M\t16.9K\t348\tBB\t541.5M\t15.6K\t347
AK\t373.1M\t10.7K\t348\tCP\t164.6M\t4.7K\t352
YP\t320M\t9.2K\t348\tPD\t86.4M\t2.4K\t351"""


def test_magnitude_suffixes_are_not_dropped():
    """Regression: "840.9M" parsed as 840.9 - a 1,000,000x error."""
    assert to_number("840.9M") == pytest.approx(840_900_000.0)
    assert to_number("2.1B") == pytest.approx(2_100_000_000.0)
    assert to_number("24.2K") == pytest.approx(24_200.0)
    assert to_number("1K") == pytest.approx(1_000.0)


def test_indonesian_magnitude_words_are_understood():
    assert to_number("840,9 jt") == pytest.approx(840_900_000.0)
    assert to_number("2,1 mlr") == pytest.approx(2_100_000_000.0)


def test_a_suffixed_dot_is_a_decimal_not_thousands():
    assert to_number("840.9M") != pytest.approx(8_409_000_000.0)


def test_two_brokers_on_one_row_are_found():
    cells = ["XL", "840.9M", "24.2K", "349", "RF", "2.1B", "60.6K", "347"]
    assert broker_positions(cells) == [0, 4]


def test_the_sell_broker_keeps_its_own_volume():
    """Regression: reading a two-sided row as one broker gave RF's 2.1B to XL."""
    buy, sell = parse_sides(ACES)
    order, _ = infer_order(buy + sell)
    f = sides_to_frame(buy, sell, order).set_index("broker")
    assert f.loc["RF", "sell_lot"] == pytest.approx(60_600.0)
    assert f.loc["RF", "buy_lot"] == pytest.approx(0.0)
    assert f.loc["XL", "buy_lot"] == pytest.approx(24_200.0)
    assert f.loc["XL", "sell_lot"] == pytest.approx(0.0)


def test_column_order_is_decided_by_arithmetic_not_by_the_header():
    """value = lot x 100 x average is the identity that settles val/lot/avg."""
    buy, sell = parse_sides(ACES)
    order, agree = infer_order(buy + sell)
    assert order == "val_lot_avg"
    assert agree > 0.9


def test_a_scrambled_table_fails_the_identity_rather_than_parsing_wrong():
    bad = "XL\t1\t2\t3\tRF\t4\t5\t6"
    buy, sell = parse_sides(bad)
    _order, agree = infer_order(buy + sell)
    assert agree < 0.6


def test_both_sides_appear_in_the_joined_frame():
    buy, sell = parse_sides(ACES)
    f = sides_to_frame(buy, sell, "val_lot_avg")
    assert set(f["broker"]) == {"XL", "XA", "AK", "YP", "RF", "BB", "CP", "PD"}


def test_a_broker_missing_from_one_side_gets_zero_not_nan():
    buy, sell = parse_sides(ACES)
    f = sides_to_frame(buy, sell, "val_lot_avg").set_index("broker")
    assert f.loc["XL", "sell_lot"] == 0.0
    assert not np.isnan(f.loc["XL", "sell_lot"])
