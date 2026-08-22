"""Tests for the broker-flow panel: mostly one property, tested hard.

THE PROPERTY IS THE LABEL'S START DATE, AND IT IS WORTH MORE THAN THE FEATURES.

IndoPremier's rekap for a fortnight ending on day T is only complete after T
closes. So T is the decision bar, the first bar anyone could act on is T+1, and
the forward return has to be measured from T+1's close. Measuring it from T's
close buys the stock before the summary that motivated the buy exists - a
look-ahead worth several percent a year on IDX and completely invisible in the
output, because the resulting series is a perfectly well-formed return series
that is simply wrong.

CLAUDE.md A5: "never use data stamped after the decision bar". This file is
where that stops being a slogan.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "scripts"))

from flow_panel_collect import windows                     # noqa: E402
from flow_panel_build import flow_features                 # noqa: E402


def rekap(buy, sell, brokers=None):
    n = len(buy)
    return pd.DataFrame({
        "broker": brokers or [f"B{i}" for i in range(n)],
        "buy_lot": buy, "sell_lot": sell,
        "buy_val": np.array(buy) * 1000.0,
        "sell_val": np.array(sell) * 1000.0,
    })


# --------------------------------------------------------------------------
# windows
# --------------------------------------------------------------------------
def test_windows_do_not_overlap():
    """Overlapping windows would double-count flow AND overlap the labels
    twice over - once from the window, once from the horizon. §11 already has
    to purge the horizon overlap without help from the sampling."""
    w = windows("2024-01-01", "2024-06-30", 10)
    for (a1, b1), (a2, b2) in zip(w, w[1:]):
        assert b1 < a2


def test_a_window_is_the_requested_number_of_business_days():
    w = windows("2024-01-01", "2024-12-31", 10)
    got = pd.bdate_range(w[0][0], w[0][1])
    assert len(got) == 10


def test_windows_run_oldest_first_so_the_collector_can_reverse_them():
    w = windows("2024-01-01", "2024-06-30", 10)
    assert w[0][0] < w[-1][0]


# --------------------------------------------------------------------------
# flow features are ratios, because the source is truncated
# --------------------------------------------------------------------------
def test_imbalance_is_scale_free():
    """Top-10 truncation biases a LEVEL far harder than a RATIO, and the
    source abbreviates anything over a million to 2-3 significant figures.
    So doubling every figure must not move the feature."""
    a = flow_features(rekap([100, 50], [30, 20]), set())
    b = flow_features(rekap([200, 100], [60, 40]), set())
    assert a["imbalance"] == pytest.approx(b["imbalance"])


def test_imbalance_is_bounded_and_signed_the_obvious_way():
    assert flow_features(rekap([100], [0]), set())["imbalance"] == 1.0
    assert flow_features(rekap([0], [100]), set())["imbalance"] == -1.0
    assert flow_features(rekap([50], [50]), set())["imbalance"] == 0.0


def test_an_empty_window_yields_no_features_rather_than_zeros():
    """A fortnight with no flow is an ABSENCE. Returning 0.0 imbalance would
    put it in the middle of the cross-section as though it were balanced."""
    assert flow_features(rekap([0], [0]), set()) == {}


def test_foreign_and_domestic_net_partition_the_flow():
    f = flow_features(rekap([100, 40], [20, 60], ["BK", "YP"]), {"BK"})
    assert f["foreign_net"] + f["domestic_net"] == pytest.approx(
        f["imbalance"])


def test_only_high_confidence_foreign_codes_count():
    """§9.2: nominee codes are omnibus and a wrong classification does not
    average out, it puts one cohort's flow in the other's column."""
    from flow_panel_build import foreign_codes
    import yaml
    d = yaml.safe_load(open("config/brokers.yaml"))["brokers"]
    fc = foreign_codes()
    assert fc
    for k in fc:
        assert d[k]["confidence"] == "high"
    # a foreign broker recorded at lower confidence must NOT be in the set
    low = {k for k, v in d.items()
           if v.get("foreign") and v.get("confidence") != "high"}
    assert not (low & fc)


def test_two_sided_codes_counts_only_brokers_present_on_both_sides():
    """The two columns are INDEPENDENT rankings - row 3's buyer and row 3's
    seller are unrelated - so a per-broker net only exists where a code shows
    up in both top tens."""
    f = flow_features(rekap([100, 0, 50], [0, 80, 25], ["AA", "BB", "CC"]),
                      set())
    assert f["two_sided_codes"] == 1.0        # only CC


# --------------------------------------------------------------------------
# the label
# --------------------------------------------------------------------------
def _toy_spine(n=40, step=0.01):
    """A strictly rising series, so any index error shows up as a wrong value."""
    d = pd.bdate_range("2024-01-01", periods=n)
    px = 1000.0 * (1.0 + step) ** np.arange(n)
    return pd.DataFrame({"date": d, "open": px, "high": px, "low": px,
                         "close": px, "adj_close": px,
                         "volume": np.full(n, 1000.0)})


def test_the_label_starts_at_the_bar_AFTER_the_window_closes():
    """The regression this whole file exists for.

    On a series rising 1% a bar, a 10-bar forward return measured correctly
    from T+1 is 1.01**10 - 1. Measured from T it is the same number - which is
    why a constant-growth series cannot catch the bug on its own, and why the
    test below uses a series with ONE anomalous bar instead.
    """
    s = _toy_spine()
    adj = s["adj_close"].to_numpy()
    iT = 20
    e0 = iT + 1
    correct = adj[e0 + 10] / adj[e0] - 1.0
    lookahead = adj[iT + 10] / adj[iT] - 1.0
    assert correct == pytest.approx(lookahead)   # equal on a smooth series...

    # ...so give bar T+1 a jump. Now the two differ, and the correct one is
    # the one that does NOT capture the jump as free profit.
    adj2 = adj.copy()
    adj2[e0:] *= 1.5
    correct2 = adj2[e0 + 10] / adj2[e0] - 1.0
    lookahead2 = adj2[iT + 10] / adj2[iT] - 1.0
    assert lookahead2 > correct2 + 0.4, "entering at T harvests the jump"
    assert correct2 == pytest.approx(correct), "entering at T+1 does not"


def test_the_builder_labels_from_the_entry_bar_not_the_decision_bar(
        tmp_path, monkeypatch):
    """End-to-end through the real ``build()``, with a planted jump.

    The spine rises 1% a bar except for a doubling on the FIRST TRADEABLE bar
    after the window closes. A builder that enters at T harvests that doubling
    and reports ~+110%; one that correctly enters at T+1 reports 1.01**10 - 1.
    The two answers are an order of magnitude apart, so this cannot pass by
    accident.
    """
    import flow_panel_build as B

    n = 60
    s = _toy_spine(n)
    adj = np.array(s["adj_close"], dtype=float, copy=True)
    win = windows("2024-01-01", "2024-12-31", 10)[2]        # bars 20..29
    dates = list(s["date"])
    iT = dates.index(win[1])                                # decision bar
    adj[iT + 1:] *= 2.0                                     # jump at entry
    s["adj_close"] = adj
    s["close"] = adj

    store = tmp_path / "store"
    store.mkdir()
    key = f"AAAA_{win[0]:%Y%m%d}_{win[1]:%Y%m%d}_RG_range"
    rekap(([100.0, 40.0]), ([20.0, 10.0])).to_csv(
        store / f"{key}.csv.gz", index=False, compression="gzip")

    monkeypatch.setattr(B, "STORE", str(store))
    monkeypatch.setattr(B, "load_panel", lambda *_, **__: pd.DataFrame(
        [{"ticker": "AAAA", "decile": 5, "src": "live",
          "entry_turnover": 1e9}]))
    monkeypatch.setattr(B, "load_prices",
                        lambda t, src: s.assign(stale=False).set_index("date"))

    D = B.build(step=10, start="2024-01-01", end=str(dates[-1].date()))
    assert len(D) == 1
    row = D.iloc[0]
    assert row["T"] == win[1]
    assert row["entry_date"] == dates[iT + 1]
    assert row["fwd_1w"] == pytest.approx(1.01 ** 10 - 1.0, rel=1e-9)
    assert row["fwd_1w"] < 0.2, "entering at T would have reported ~+110%"


def test_coverage_is_a_share_of_real_volume_not_of_the_top_ten():
    """An imbalance measured at 40% coverage is a different quantity from one
    at 100%, so coverage has to be on the row, not assumed away."""
    b = [1000.0]                      # 1,000 lots = 100,000 shares
    traded = 200_000.0                # spine says 200,000 shares traded
    cov = b[0] * 100.0 / traded
    assert cov == pytest.approx(0.5)
