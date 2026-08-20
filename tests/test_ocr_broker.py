"""Tests for the screenshot route.

The point of OCR here is not to be clever, it is to be CHECKABLE. Tesseract will
misread digits on a dark trading panel; what makes the route usable is that
value = lot x 100 x average catches the misreads, so a wrong number is refused
rather than stored. These tests pin the cleanup that makes good reads survive and
the check that makes bad ones fail.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "scripts"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

from ocr_broker import clean_ocr, to_rows                          # noqa: E402
from paste_broker import infer_order, parse_sides                  # noqa: E402


# --------------------------------------------------------------------------- #
# the spacing Tesseract sprays into numbers
# --------------------------------------------------------------------------- #
def test_a_space_before_the_decimal_point_is_closed():
    """Regression: "840 .9M" tokenised as two cells and broke the row."""
    assert "840.9M" in clean_ocr("XL 840 .9M 24.2K 349")


def test_a_space_after_the_decimal_point_is_closed():
    assert "16.9K" in clean_ocr("XA 589.7M 16. 9K 348")


def test_a_space_before_the_magnitude_letter_is_closed():
    assert "2.1B" in clean_ocr("RF 2.1 B 60.6K 347")


def test_a_space_between_two_separate_numbers_is_kept():
    """Closing every space would merge adjacent columns into one number."""
    out = clean_ocr("XL 349 347")
    assert out == "XL 349 347"


def test_o_is_read_as_zero_inside_a_number_only():
    assert clean_ocr("XL 1O0K") == "XL 100K"


def test_a_broker_code_containing_o_is_left_alone():
    """AO and IO are real IDX member codes - they must not become A0 and I0."""
    assert "AO" in clean_ocr("AO 69.2M 2K 346")


def test_pipes_from_table_borders_are_dropped():
    assert "|" not in clean_ocr("| XL | 840.9M |")


def test_blank_lines_are_dropped():
    assert clean_ocr("\n\n  \n") == ""


# --------------------------------------------------------------------------- #
# re-tabulating
# --------------------------------------------------------------------------- #
def test_rows_become_tab_separated():
    assert to_rows("XL 840.9M 24.2K 349") == "XL\t840.9M\t24.2K\t349"


def test_lines_with_too_few_tokens_are_dropped():
    assert to_rows("Broker Summary") == ""


# --------------------------------------------------------------------------- #
# the check that makes the whole route trustworthy
# --------------------------------------------------------------------------- #
GOOD = """XL\t840.9M\t24.2K\t349\tRF\t2.1B\t60.6K\t347
XA\t589.7M\t16.9K\t348\tBB\t541.5M\t15.6K\t347
AK\t373.1M\t10.7K\t348\tCP\t164.6M\t4.7K\t352"""


def test_a_clean_read_passes_the_identity():
    buy, sell = parse_sides(GOOD)
    _order, agree = infer_order(buy + sell)
    assert agree > 0.9


def test_a_misread_digit_fails_the_identity():
    """The safety net: change one digit and the table must stop validating."""
    bad = GOOD.replace("24.2K", "24.2M")      # a plausible OCR slip, 1000x wrong
    buy, sell = parse_sides(bad)
    _order, agree = infer_order(buy + sell)
    assert agree < 1.0


def test_the_identity_is_what_distinguishes_the_column_order():
    buy, sell = parse_sides(GOOD)
    order, _ = infer_order(buy + sell)
    assert order == "val_lot_avg"


# --------------------------------------------------------------------------- #
# image preparation
# --------------------------------------------------------------------------- #
def test_a_dark_panel_is_inverted_before_ocr(tmp_path):
    from PIL import Image
    from ocr_broker import prepare
    p = tmp_path / "dark.png"
    a = np.zeros((40, 120), dtype=np.uint8)
    a[10:20, 10:110] = 230                      # light text on a dark ground
    Image.fromarray(a).save(p)
    out = np.asarray(prepare(str(p), scale=2))
    # after inversion and thresholding the background must be the light majority
    assert float(out.mean()) > 127


def test_a_light_panel_is_not_inverted(tmp_path):
    from PIL import Image
    from ocr_broker import prepare
    p = tmp_path / "light.png"
    a = np.full((40, 120), 240, dtype=np.uint8)
    a[10:20, 10:110] = 20
    Image.fromarray(a).save(p)
    out = np.asarray(prepare(str(p), scale=2))
    assert float(out.mean()) > 127
