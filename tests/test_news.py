"""Tests for the news layer.

The first block is the one that matters most, and it is a policy test rather
than an arithmetic one: **no statistic may read this module.** There is no
point-in-time news archive, so a headline visible today cannot be
reconstructed as it stood on a past bar. Anything under `spine/` or
`features/` importing it would make every downstream backtest look-ahead by
construction, which is the thing A5 forbids outright.

Everything after that is a regression for something the build got wrong while
producing plausible output:

    the doubled suffix   Google emits "... - Bisnis.com - Bisnis.com" and the
                         hyphen-excluding capture backtracks past the first
                         separator, so one pass strips only the last
    ambiguous tickers    GULA is sugar, KOTA is a city, BABY is a nursery
    the flat date cut    Google ranks by relevance not date, so a 14-day
                         window kept a "top gainers" listicle and threw away
                         a rights issue
    missing halt words   "dihentikan sementara" is how a suspension is
                         actually reported; a list of only "suspensi" and
                         "suspend" missed a live one
    "umum" firing UMA    substring matching on a three-letter tag
"""

from __future__ import annotations

import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

from idxbot.data import news                                    # noqa: E402


def rss(items: str) -> bytes:
    return (b'<?xml version="1.0"?><rss version="2.0"><channel>'
            + items.encode() + b"</channel></rss>")


def item(title, src="", pub="Mon, 24 Aug 2026 06:22:00 GMT", link="x") -> str:
    s = f"<source>{src}</source>" if src else ""
    return (f"<item><title>{title}</title><link>{link}</link>"
            f"<pubDate>{pub}</pubDate>{s}</item>")


# --------------------------------------------------------------------------
# THE QUARANTINE — the reason this file leads with a policy test
# --------------------------------------------------------------------------
def test_no_statistical_module_imports_the_news_layer():
    """A headline cannot be reconstructed as it stood on a past bar, so any
    statistic reading this is look-ahead by construction (A5)."""
    import ast
    root = os.path.join(os.path.dirname(__file__), os.pardir, "src", "idxbot")
    offenders = []
    for sub in ("spine", "features"):
        d = os.path.join(root, sub)
        for fn in sorted(os.listdir(d)):
            if not fn.endswith(".py"):
                continue
            tree = ast.parse(open(os.path.join(d, fn)).read())
            for node in ast.walk(tree):
                names = []
                if isinstance(node, ast.Import):
                    names = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom):
                    names = [f"{node.module or ''}.{a.name}"
                             for a in node.names]
                if any(n.split(".")[-1] == "news" for n in names):
                    offenders.append(f"{sub}/{fn}")
    assert not offenders, f"news must not reach a statistic: {offenders}"


def test_the_caveat_states_the_quarantine_rather_than_implying_it():
    from idxbot.report.brief import news_caveat
    c = news_caveat()
    assert "MAY NOT ENTER ANY STATISTIC" in c
    assert "look-ahead" in c
    assert "not available" not in c.lower(), (
        "this replaced a claim that news was unavailable; it must not "
        "re-assert it")


# --------------------------------------------------------------------------
# HOSTS — the fetcher refuses anything not sanctioned
# --------------------------------------------------------------------------
def test_an_unlisted_host_is_refused_rather_than_fetched():
    with pytest.raises(news.HostNotAllowed):
        news.fetch("https://evil.example.com/rss", hosts=["news.google.com"])


def test_idx_co_id_is_not_in_the_default_allowlist():
    """Its endpoints sit behind a Cloudflare check and docs/FULL_REKAP.md §3
    explains why this repo does not go through it. That reasoning does not
    stop applying because a different module is doing the asking."""
    assert not any("idx.co.id" in h for h in news.DEFAULT_HOSTS)


def test_every_default_host_is_actually_used_by_a_feed():
    import urllib.parse
    used = {urllib.parse.urlparse(u).netloc for u in news.MARKET_FEEDS.values()}
    used.add(urllib.parse.urlparse(news.ticker_url("AAAA")).netloc)
    assert used <= set(news.DEFAULT_HOSTS)


# --------------------------------------------------------------------------
# THE DOUBLED PUBLISHER SUFFIX
# --------------------------------------------------------------------------
def test_a_doubled_publisher_suffix_is_fully_stripped():
    """Google emits it twice. Because the capture class excludes hyphens the
    regex backtracks past the first separator and one pass removes only the
    last, leaving a headline still ending in its own publisher — correct on 91
    of 100 items and wrong on 9."""
    D = news.parse_rss(rss(item(
        "Deretan Saham Top Gainers, EKAD-BABY Teratas - Bisnis.com - Bisnis.com",
        src="Bisnis.com - Market")), "google")
    assert D["title"].iloc[0] == "Deretan Saham Top Gainers, EKAD-BABY Teratas"
    assert D["source"].iloc[0] == "Bisnis.com - Market"


def test_a_single_suffix_is_stripped_and_becomes_the_source():
    D = news.parse_rss(rss(item("Saham BBRI Menguat - investor.id")), "google")
    assert D["title"].iloc[0] == "Saham BBRI Menguat"
    assert D["source"].iloc[0] == "investor.id"


def test_a_legitimate_dash_in_a_headline_survives():
    """Not every ' - ' is a publisher tag. Stripping blindly would eat words."""
    D = news.parse_rss(rss(item("Rekomendasi Saham Hari Ini - Simak Dulu",
                                src="Kontan")), "google")
    assert "Simak Dulu" in D["title"].iloc[0]


# --------------------------------------------------------------------------
# TAGGING
# --------------------------------------------------------------------------
def test_the_halt_wording_the_indonesian_press_actually_uses_is_caught():
    """A list of only "suspensi"/"suspend" missed a live ADHI halt sitting at
    the top of the market feed."""
    for t in ("Perdagangan Saham Dihentikan Sementara oleh Bursa, ADHI Buka Suara",
              "Belum Bisa Bayar Bunga Obligasi, BEI Gembok Saham ADHI",
              "BEI Suspensi Saham ADHI"):
        assert "SUSPEND" in news.tag(t), t


def test_right_issue_singular_is_caught_as_well_as_plural():
    assert "RIGHTS" in news.tag("Right Issue Saham GULA: Sebelum Cum Date")
    assert "RIGHTS" in news.tag("BABY Rights Issue Rp140,7 Miliar")


def test_uma_does_not_fire_on_umum():
    """A three-letter tag matched as a substring fires on 'Penawaran Umum'."""
    assert "UMA" not in news.tag("Penawaran Umum Perdana Saham")
    assert "UMA" in news.tag("Pola Transaksi Saham PACK Masuk UMA")


def test_an_untagged_headline_gets_an_empty_list_not_a_false_positive():
    assert news.tag("Wall Street Dibuka Melemah") == []


# --------------------------------------------------------------------------
# RELEVANCE
# --------------------------------------------------------------------------
def test_general_business_copy_is_kept_out_of_the_market_section():
    assert not news.is_market("Belajar Membatik untuk Bekal Kemandirian Ekonomi")
    assert news.is_market("IHSG Ditutup Menguat, Saham Bank Memimpin")
    assert news.is_market("BI Rate Ditahan di Level 5,25%")


def test_dedupe_collapses_syndicated_repeats_keeping_the_earliest():
    D = pd.DataFrame({
        "title": ["BEI Gembok Saham ADHI", "BEI gembok saham ADHI!", "Lainnya"],
        "link": ["a", "b", "c"], "source": ["x", "y", "z"],
        "feed": ["1", "2", "3"],
        "published": pd.to_datetime(["2026-08-24T09:00Z", "2026-08-24T08:00Z",
                                     "2026-08-24T07:00Z"], utc=True)})
    out = news.dedupe(D)
    assert len(out) == 2
    keep = out[out["title"].str.contains("ADHI", case=False)]
    assert str(keep["published"].iloc[0]).startswith("2026-08-24 08")


# --------------------------------------------------------------------------
# FAILURE MODES
# --------------------------------------------------------------------------
def test_malformed_xml_gives_an_empty_frame_rather_than_raising():
    """One broken feed must not take the whole brief down."""
    D = news.parse_rss(b"<rss><channel><item><title>unclosed", "x")
    assert D.empty and list(D.columns)[:2] == ["title", "link"]


def test_an_empty_body_gives_an_empty_frame():
    assert news.parse_rss(b"", "x").empty
    assert news.parse_rss(None, "x").empty


def test_an_item_with_no_title_is_skipped_not_kept_blank():
    D = news.parse_rss(rss("<item><link>a</link></item>" + item("Real")), "x")
    assert len(D) == 1 and D["title"].iloc[0] == "Real"


def test_an_unparseable_date_becomes_nat_rather_than_today():
    """A bad timestamp silently read as 'now' would promote stale news to the
    top of a brief that sorts by recency."""
    D = news.parse_rss(rss(item("Saham X", pub="not a date")), "x")
    assert pd.isna(D["published"].iloc[0])
