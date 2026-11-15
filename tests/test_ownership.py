"""The ownership report: descriptive, flagged where flows are not trades,
and quoting H64 only from H64's own output."""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, os.pardir, "src"))
sys.path.insert(0, os.path.join(HERE, os.pardir, "scripts"))

import ownership as ow                                          # noqa: E402
from idxbot.data import ksei                                    # noqa: E402


def _panel(rows):
    """rows: (month_end, ticker, local dict, foreign_total)."""
    out = []
    for me, t, loc, ft in rows:
        r = {"month_end": pd.Timestamp(me), "ticker": t, "sec_num": 0,
             "price": 1000.0, "foreign_total": ft}
        for c in ksei.LOCAL:
            r[c] = loc.get(c, 0)
        for c in ksei.FOREIGN:
            r[c] = 0
        r["foreign_IS"] = ft
        r["local_total"] = sum(r[c] for c in ksei.LOCAL)
        r["sec_num"] = r["local_total"] + ft
        out.append(r)
    return pd.DataFrame(out)


M = ["2026-08-31", "2026-09-30", "2026-10-30"]


def test_a_founder_moving_a_stake_is_a_transfer_not_retail_selling():
    """BYAN, October 2026, in miniature."""
    P = _panel([(M[0], "BYAN", {"local_ID": 1450, "local_CP": 330}, 1450),
                (M[1], "BYAN", {"local_ID": 1450, "local_CP": 330}, 1450),
                (M[2], "BYAN", {"local_ID": 375, "local_CP": 1405}, 1450)])
    f = ksei.transfer_flags(P)
    assert list(f) == [False, False, True]


def test_the_crowd_selling_to_foreigners_is_not_a_transfer():
    P = _panel([(M[0], "AAAA", {"local_ID": 1000, "local_CP": 500}, 500),
                (M[1], "AAAA", {"local_ID": 900, "local_CP": 500}, 600)])
    assert not ksei.transfer_flags(P).any()


def test_a_small_mirrored_move_is_not_flagged():
    P = _panel([(M[0], "AAAA", {"local_ID": 1000, "local_MF": 500}, 8500),
                (M[1], "AAAA", {"local_ID": 995, "local_MF": 505}, 8500)])
    assert not ksei.transfer_flags(P).any()      # 0.05% of the balance


def test_composition_partitions_the_balance():
    P = _panel([(M[0], "AAAA", {"local_ID": 100, "local_CP": 300,
                                "local_MF": 50, "local_SC": 25}, 525)])
    C = ow.composition(P)
    assert C[list(ow.BUCKETS)].sum(axis=1).iloc[0] == pytest.approx(1.0)
    assert C["retail"].iloc[0] == pytest.approx(0.1)


def test_transfer_months_are_out_of_the_market_sum_and_flagged():
    P = _panel([(M[0], "BYAN", {"local_ID": 1450, "local_CP": 330}, 1450),
                (M[1], "BYAN", {"local_ID": 375, "local_CP": 1405}, 1450),
                (M[0], "AAAA", {"local_ID": 1000}, 1000),
                (M[1], "AAAA", {"local_ID": 1100}, 900)])
    F = ow.flows(P)
    oct_ = F[F["month_end"] == pd.Timestamp(M[1])].set_index("ticker")
    assert bool(oct_.loc["BYAN", "transfer"])
    assert np.isnan(oct_.loc["BYAN", "retail_rp"])
    assert oct_.loc["AAAA", "retail_rp"] == pytest.approx(100 * 1000.0)
    lines = ow.ticker_report("BYAN", ow.composition(P), F,
                             P.groupby("month_end")["month_end"].first()
                             .map(lambda d: d + pd.Timedelta(days=3)), None)
    assert any("re-registered" in ln for ln in lines)


def test_the_caveat_says_so_when_h64_has_not_run(tmp_path):
    lines = ow.h64_caveat(str(tmp_path / "missing.json"))
    assert "has not been run" in lines[0]
    assert not any(ch.isdigit() and "IC" in ln for ln in lines for ch in ln
                   if "~40 s" not in ln)


def test_the_caveat_is_read_from_the_result_file(tmp_path):
    p = tmp_path / "r.json"
    p.write_text(json.dumps({
        "spec": "amendment A", "in_sample": ["2019-05-03", "2024-08-02"],
        "confirm_k": 63, "bonferroni": 0.000136,
        "signals": {
            "f_retail": {"ic": 0.0123, "t": 1.11, "p": 0.2222, "gate": False,
                         "net_annual_predicted_leg": -0.031},
            "f_foreign": {"ic": -0.0456, "t": -2.22, "p": 0.0333,
                          "gate": False}}}))
    text = "\n".join(ow.h64_caveat(str(p)))
    assert "+0.0123" in text and "-0.0456" in text and "-3.1%/yr" in text
    assert "Neither side is a trade" in text


def test_no_h64_figure_is_typed_into_the_report():
    """A47: the figures arrive through the result file or not at all."""
    src = open(os.path.join(HERE, os.pardir, "scripts", "ownership.py"),
               encoding="utf-8").read()
    for typed in ("0.0275", "2.58", "0.0069", "-7.9", "0.0167", "1.75"):
        assert typed not in src, typed
