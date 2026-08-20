#!/usr/bin/env python3
"""Read a broker-summary screenshot. No server is touched at any point.

WHY THIS IS THE ROUTE THAT WORKS
--------------------------------
Pulling live running trade and broker summary from IDX has exactly three
sanctioned shapes, and only one of them is open to a person:

    1. A LICENSED IDX FEED. ITCH Basic or Total View carries BuyParticipantId
       and SellParticipantId on every execution - verified against IDX's own
       published ITCH sample - so it is precisely the data wanted. It also costs
       Rp 17.9-44 million a month, requires a limited company, a signed IDX
       licence and a leased line from an authorised NSP, plus a security deposit
       of 300% of the fee. Correct, and not retail.

    2. A LICENSED REDISTRIBUTOR. LSEG publishes one RIC per broker per
       instrument (<OD-BBCA.JK> = Danareksa in BBCA, buy volume on FID 731).
       Bloomberg and Refinitiv terminals are the usual delivery and cost tens of
       thousands of dollars a year.

    3. YOUR OWN SCREEN. Your broker already shows you this data - you are
       licensed to look at it. Reading the pixels on your own display sends no
       request to anybody, so it is not scraping under IDX's rules and not "data
       mining, robots, spiders or similar" under Stockbit's. It is the one route
       that is both free and clean.

This module is (3). It never opens a socket.

WHAT IT HANDLES
---------------
Broker panels are dark-themed, tightly kerned and full of abbreviated
magnitudes, all of which OCR hates. So the image is inverted when the background
is dark, upscaled, and thresholded before Tesseract sees it, and the text is then
handed to the same parser used for pasted tables - which knows about "840.9M",
about two brokers sharing a row, and about deciding the column order from
value = lot x 100 x average rather than from a header.

That identity is also the ACCURACY CHECK. OCR misreads digits; a misread digit
breaks the identity, and the parser refuses the table rather than storing a
plausible-looking wrong number. An OCR pipeline without that check would be
worse than no pipeline.

    python3 scripts/ocr_broker.py shot.png --ticker ACES --date 2026-07-27
    python3 scripts/ocr_broker.py shot.png --ticker ACES --show-text
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))
sys.path.insert(0, os.path.dirname(__file__))

from paste_broker import (broker_positions, infer_order, parse_sides,  # noqa: E402
                          parse_table, assign_columns, report_and_store,
                          sides_to_frame, split_row)

TESS_CONFIG = "--psm 6 -c preserve_interword_spaces=1"


def prepare(path: str, scale: int = 3) -> "Image.Image":
    """Make a dark trading panel legible to an engine trained on documents."""
    from PIL import Image, ImageOps, ImageFilter
    img = Image.open(path).convert("L")
    # A trading panel is light text on a dark ground; Tesseract expects the
    # opposite. Decide by the median pixel rather than by assumption, because
    # light-theme screenshots exist too.
    if float(np.median(np.asarray(img))) < 128:
        img = ImageOps.invert(img)
    img = img.resize((img.width * scale, img.height * scale), Image.LANCZOS)
    img = img.filter(ImageFilter.SHARPEN)
    arr = np.asarray(img)
    # Otsu-style split, computed rather than hardcoded: panels vary in contrast.
    hist, _ = np.histogram(arr, bins=256, range=(0, 256))
    total = arr.size
    best_t, best_var = 128, -1.0
    w0 = 0.0
    sum0 = 0.0
    total_sum = float((np.arange(256) * hist).sum())
    for t in range(1, 256):
        w0 += hist[t - 1]
        if w0 == 0 or w0 == total:
            continue
        sum0 += (t - 1) * hist[t - 1]
        m0 = sum0 / w0
        m1 = (total_sum - sum0) / (total - w0)
        var = w0 * (total - w0) * (m0 - m1) ** 2
        if var > best_var:
            best_var, best_t = var, t
    return Image.fromarray(((arr > best_t) * 255).astype(np.uint8))


def ocr_text(path: str, scale: int = 3) -> str:
    import pytesseract
    return pytesseract.image_to_string(prepare(path, scale), config=TESS_CONFIG)


def clean_ocr(text: str) -> str:
    """Undo the substitutions Tesseract makes on this kind of panel.

    Only unambiguous ones, and only where a letter sits inside a number - never
    inside a broker code, which is letters by definition.
    """
    out = []
    for line in text.splitlines():
        line = line.replace("|", " ").replace("—", "-")
        # O/l/I are read for 0/1 inside numeric tokens only
        line = re.sub(r"(?<=\d)[OoQ](?=[\d.,KMBT])", "0", line)
        line = re.sub(r"(?<=\d)[lI](?=[\d.,KMBT])", "1", line)
        line = re.sub(r"\s+", " ", line).strip()
        # Tesseract sprays spaces around the decimal point and before the
        # magnitude letter on tightly-kerned panels: "840 .9M", "16. 9K",
        # "2.1 B". Every digit is right; only the spacing is wrong, so the
        # tokens are rejoined rather than the numbers rejected.
        line = re.sub(r"(?<=\d)\s*([.,])\s*(?=\d)", r"\1", line)
        line = re.sub(r"(?<=[\d.,])\s+(?=[KMBT]\b)", "", line)
        if line:
            out.append(line)
    return "\n".join(out)


def to_rows(text: str) -> str:
    """Re-tabulate: OCR gives single spaces, the parser wants column breaks."""
    rows = []
    for line in text.splitlines():
        toks = line.split(" ")
        if len(toks) >= 3:
            rows.append("\t".join(toks))
    return "\n".join(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--ticker", required=True)
    ap.add_argument("--date", default=None)
    ap.add_argument("--scale", type=int, default=3)
    ap.add_argument("--show-text", action="store_true",
                    help="print what the OCR actually read, to debug a bad parse")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not os.path.exists(args.image):
        print(f"no such image: {args.image}")
        return 1
    date = args.date or pd.Timestamp.now(tz="Asia/Jakarta").strftime("%Y-%m-%d")

    print(f"{'=' * 84}\n OCR BROKER SUMMARY — {args.ticker} {date}\n{'=' * 84}")
    print(f" {os.path.basename(args.image)}  (no network request is made)")

    raw = clean_ocr(ocr_text(args.image, args.scale))
    if args.show_text:
        print(f"\n--- what Tesseract read ---\n{raw}\n---------------------------")

    text = to_rows(raw)
    two = sum(1 for l in text.splitlines() if len(broker_positions(split_row(l))) >= 2)
    one = sum(1 for l in text.splitlines() if len(broker_positions(split_row(l))) == 1)
    if not (two or one):
        print("\n no broker rows recognised. Try --scale 4, crop to just the "
              "table, or\n use --show-text to see what came back.")
        return 1

    if two >= max(one, 2):
        buy, sell = parse_sides(text)
        order, agree = infer_order(buy + sell)
        print(f"\n two-sided layout: {len(buy)} buy rows, {len(sell)} sell rows")
        print(f" column order {order.replace('_', '/')}, identity holds on "
              f"{agree:.0%} of rows")
        if agree < 0.6:
            print(" ! value = lot x 100 x average fails on most rows, so a digit "
                  "was misread.\n ! Refusing to store it. Re-shoot at a higher "
                  "zoom, or use --scale 4.")
            return 1
        mapped = sides_to_frame(buy, sell, order)
    else:
        parsed = parse_table(text)
        mapped = assign_columns(parsed) if not parsed.empty else None
        if mapped is None:
            print("\n found broker codes but not enough numeric columns.")
            return 1
        print(f"\n single-sided layout: {len(mapped)} rows")

    if mapped is None or mapped.empty:
        print(" nothing usable.")
        return 1
    mapped["ticker"] = args.ticker.upper()
    mapped["date"] = pd.Timestamp(date)
    mapped["source"] = "ocr"
    return report_and_store(mapped, args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
