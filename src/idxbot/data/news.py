"""Public news feeds — the narrative layer the brief was missing.

WHY THIS EXISTS, AND WHY IT DID NOT
-------------------------------------
`reports/daily_brief.md` shipped saying a news narrative was unreachable
because "there is no news source anywhere in this repo and §3's data table
lists none". Both halves of that were true and the conclusion drawn from them
was wrong: §3's table lists none because nobody had looked, not because none
exists. Eight candidate endpoints were then tested in about a minute and five
of them answered.

This is the same failure CLAUDE.md A1 already records — recording a constraint
before checking the cheapest route past it — and A1's own lesson applies
verbatim: *check the unit price before writing down the arithmetic.*

WHAT IS REACHABLE, MEASURED RATHER THAN ASSUMED
-------------------------------------------------
    Google News RSS, per query   200 OK, 100 items, arbitrary query  <- best
    CNBC Indonesia /market/rss   200 OK, 100 items
    Kontan investasi RSS         200 OK,  25 items
    Detik finance RSS            200 OK, 100 items
    Yahoo per-ticker .JK RSS     200 OK, 0-1 items                   <- useless
    Bisnis.com RSS               403
    idnfinancials RSS            403
    idx.co.id announcements      403 (Cloudflare, as docs/FULL_REKAP.md records)

The per-ticker route is Google News with the ticker quoted. It works, and it
surfaces exactly what price data cannot: on the day it was built it returned
IDX's own UMA (Unusual Market Activity) flag for PACK, a Rp140.7bn rights
issue for BABY, and an acquisition for GULA.

THESE ARE SYNDICATION FEEDS, NOT SCRAPING
-------------------------------------------
RSS exists to be consumed by readers; that is the entire purpose of the
format. Nothing here parses a rendered page, defeats a bot check, or
impersonates a browser fingerprint — the distinction `docs/FULL_REKAP.md` §3
draws for IDX holds here too, and idx.co.id is *not* in the allowlist for
exactly that reason. Only headline, link, timestamp and source are kept; no
article body is stored and nothing is republished.

Hosts live in `data.news_allowed_hosts` and the fetcher REFUSES any host not
on that list, the same discipline `data.broker_allowed_hosts` uses. Unlike
that one this ships populated, because these are public syndication feeds
rather than a licensed member's data — but the list is in config so it stays
the user's decision.

QUARANTINE — READ THIS BEFORE USING ANY OF IT
-----------------------------------------------
**No statistic in this repo may read this module.** There is no point-in-time
news archive, so a headline available today cannot be reconstructed as it
stood on a past bar, which makes every backtest built on it look-ahead by
construction — the exact thing A5 forbids. `tests/test_news.py` asserts that
nothing under `spine/` or `features/` imports it.

It is for reading. That is a real and useful job, and it is the only one.
"""

from __future__ import annotations

import datetime as dt
import html
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from typing import Dict, Iterable, List, Optional, Sequence

import pandas as pd

#: Hosts this module will fetch from unless config overrides. Every one was
#: verified to answer 200 with parseable items before being listed.
DEFAULT_HOSTS = (
    "news.google.com",
    "www.cnbcindonesia.com",
    "investasi.kontan.co.id",
    "finance.detik.com",
)

#: Market-wide feeds, fetched once per run.
MARKET_FEEDS = {
    "google": ("https://news.google.com/rss/search?"
               + urllib.parse.urlencode({"q": "IDX saham bursa efek indonesia",
                                         "hl": "id", "gl": "ID",
                                         "ceid": "ID:id"})),
    "cnbc": "https://www.cnbcindonesia.com/market/rss",
    "kontan": "https://investasi.kontan.co.id/rss",
    "detik": "https://finance.detik.com/rss",
}

#: Seconds between requests. These are small XML files on large sites and this
#: fetches a few dozen a day; the delay is courtesy, not rate-limit avoidance.
DELAY = 0.7

CACHE_DIR = os.path.join("data", "cache", "news")
UA = "idxbot/1.0 (personal research; contact via repository)"

#: Event words worth pulling to the front of a brief. Chosen because each maps
#: to something the price series either cannot show or shows misleadingly.
#: `rights` in particular is §5's named trap: a dilutive rights issue without a
#: theoretical ex-rights adjustment prints as a crash that never happened.
TAGS: Dict[str, Sequence[str]] = {
    "UMA": ("uma", "unusual market activity"),
    # "dihentikan sementara" is how the Indonesian press actually reports a
    # trading halt — an earlier list had only "suspensi"/"suspend" and missed
    # a live ADHI halt sitting at the top of the market feed.
    "SUSPEND": ("suspensi", "suspend", "digembok", "dibekukan",
                "dihentikan sementara", "penghentian sementara",
                "gembok", "disuspensi"),
    # "right issue" without the s is at least as common in Indonesian
    # financial copy as the correct plural.
    "RIGHTS": ("rights issue", "right issue", "hmetd", "penambahan modal",
               "cum date", "ex date"),
    "SPLIT": ("stock split", "reverse stock", "pemecahan saham",
              "penggabungan saham"),
    "DIVIDEND": ("dividen", "dividend"),
    "M&A": ("akuisisi", "merger", "tender offer", "caplok", "diakuisisi",
            "mengakuisisi", "divestasi"),
    "IPO": ("ipo", "penawaran umum perdana"),
    "DELIST": ("delisting", "dihapus dari bursa", "papan pemantauan",
               "pailit", "pkpu"),
    "BUYBACK": ("buyback", "pembelian kembali"),
    "EARNINGS": ("laba", "rugi", "kinerja keuangan", "laporan keuangan",
                 "pendapatan"),
}

#: Tags that stay relevant long after the headline. A rights issue or a split
#: changes what the price series MEANS (§5 calls the rights adjustment the
#: trap), so it belongs in a brief for months; a "top gainers" listicle does
#: not survive the week.
STANDING_TAGS = ("UMA", "SUSPEND", "RIGHTS", "SPLIT", "M&A", "DELIST", "IPO")

#: A headline must touch one of these to count as market news. Detik and
#: Kontan carry general business coverage on the same feed, and without this
#: the section fills with batik workshops and export tariffs.
MARKET_WORDS = ("saham", "bursa", "ihsg", "bei", "emiten", "idx", "indeks",
                "investor", "dividen", "ipo", "obligasi", "rupiah",
                "wall street", "bi rate", "suku bunga", "asing")


class HostNotAllowed(RuntimeError):
    """Raised rather than fetching a host the config has not sanctioned."""


def allowed_hosts(cfg=None) -> List[str]:
    if cfg is None:
        return list(DEFAULT_HOSTS)
    v = cfg.get("data.news_allowed_hosts", None)
    return list(DEFAULT_HOSTS) if v is None else [str(h) for h in v]


def _check(url: str, hosts: Sequence[str]) -> None:
    host = urllib.parse.urlparse(url).netloc.lower()
    if host not in {h.lower() for h in hosts}:
        raise HostNotAllowed(
            f"{host} is not in data.news_allowed_hosts. Add it there after "
            f"checking its terms, or leave it out.")


def _cache_path(url: str) -> str:
    import hashlib
    key = hashlib.sha1(url.encode()).hexdigest()[:20]
    return os.path.join(CACHE_DIR, f"{key}.xml")


def fetch(url: str, ttl: float = 6 * 3600.0, hosts: Optional[Sequence[str]] = None,
          timeout: float = 25.0) -> Optional[bytes]:
    """One feed, cached on disk, refetched only once ``ttl`` has elapsed.

    Returns None rather than raising on a network or HTTP failure: one dead
    feed must not take the whole brief down, and the caller reports which
    sources answered. A host outside the allowlist DOES raise, because that is
    a configuration decision rather than a transient failure.
    """
    hosts = allowed_hosts() if hosts is None else hosts
    _check(url, hosts)
    p = _cache_path(url)
    if os.path.exists(p) and (time.time() - os.path.getmtime(p)) < ttl:
        with open(p, "rb") as fh:
            return fh.read()
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = r.read()
    except (urllib.error.URLError, urllib.error.HTTPError, OSError, TimeoutError):
        if os.path.exists(p):                    # stale beats nothing
            with open(p, "rb") as fh:
                return fh.read()
        return None
    os.makedirs(CACHE_DIR, exist_ok=True)
    tmp = p + ".tmp"
    with open(tmp, "wb") as fh:
        fh.write(body)
    os.replace(tmp, p)
    time.sleep(DELAY)
    return body


_GOOGLE_SUFFIX = re.compile(r"\s+-\s+([^-]{2,60})$")


def _strip_source(title: str, src: str) -> tuple:
    """Peel Google News's " - Publisher" tail off a headline.

    IT CAN APPEAR TWICE. Google sometimes emits
    ``"... Teratas - Bisnis.com - Bisnis.com"``, and because the capture class
    excludes hyphens the regex backtracks past the first separator and strips
    only the last one — leaving a headline that still ends in its own
    publisher. One pass looked correct on 91 of 100 items and wrong on 9.

    A headline may also contain a legitimate " - ", so a blind repeat would
    eat real words. The loop therefore only strips a tail it can match against
    the feed's declared ``<source>``; with no source it strips exactly once and
    adopts what it removed.
    """
    t, s = str(title).strip(), str(src or "").strip()
    for _ in range(3):
        m = _GOOGLE_SUFFIX.search(t)
        if not m:
            break
        cand = m.group(1).strip()
        if not s:
            return t[:m.start()].strip(), cand
        head = s.split(" - ")[0].strip().lower()
        if cand.lower() != head and cand.lower() not in s.lower():
            break                       # a real dash in the headline; leave it
        t = t[:m.start()].strip()
    return t, s


def parse_rss(body: bytes, feed: str = "") -> pd.DataFrame:
    """RSS items to a frame: title, link, published, source, feed.

    Google News appends " - Publisher" to every title, which is the real
    source; it is split out so a headline is attributable rather than credited
    to the aggregator.
    """
    cols = ["title", "link", "published", "source", "feed"]
    if not body:
        return pd.DataFrame(columns=cols)
    try:
        root = ET.fromstring(body)
    except ET.ParseError:
        return pd.DataFrame(columns=cols)
    rows = []
    for it in root.findall(".//item"):
        t = (it.findtext("title") or "").strip()
        if not t:
            continue
        t = html.unescape(t)
        src = (it.findtext("{http://news.google.com/rss}source")
               or it.findtext("source") or "")
        t, src = _strip_source(t, src)
        pub = it.findtext("pubDate") or ""
        rows.append({"title": t, "link": (it.findtext("link") or "").strip(),
                     "published": pd.to_datetime(pub, errors="coerce",
                                                 utc=True),
                     "source": html.unescape(src) or feed, "feed": feed})
    return pd.DataFrame(rows, columns=cols)


def tag(title: str) -> List[str]:
    """Event labels for a headline, from :data:`TAGS`."""
    low = " " + re.sub(r"[^\w\s]", " ", str(title).lower()) + " "
    out = []
    for label, words in TAGS.items():
        for w in words:
            # word-boundary match, so "uma" does not fire on "umum"
            if re.search(rf"(?<!\w){re.escape(w)}(?!\w)", low):
                out.append(label)
                break
    return out


def _norm(t: str) -> str:
    return re.sub(r"\W+", " ", str(t).lower()).strip()


def dedupe(df: pd.DataFrame) -> pd.DataFrame:
    """Drop syndicated repeats, keeping the earliest timestamp.

    Indonesian financial media republish each other heavily, so without this a
    single announcement fills the section and crowds out everything else.
    """
    if df.empty:
        return df
    d = df.copy()
    d["_k"] = d["title"].map(_norm)
    d = d.sort_values("published", na_position="last")
    return (d.drop_duplicates("_k", keep="first")
             .drop(columns="_k").reset_index(drop=True))


def is_market(title: str) -> bool:
    low = " " + re.sub(r"[^\w\s]", " ", str(title).lower()) + " "
    return any(re.search(rf"(?<!\w){re.escape(w)}(?!\w)", low)
               for w in MARKET_WORDS)


def market_news(limit: int = 25, ttl: float = 6 * 3600.0,
                cfg=None, days: int = 3, market_only: bool = True
                ) -> pd.DataFrame:
    """Market-wide headlines across every feed that answers.

    ``market_only`` drops general business coverage. Detik and Kontan run
    equities and everything else down one pipe, so without it the section fills
    with export-tariff talks and batik workshops while a live trading halt sits
    below the fold.
    """
    hosts = allowed_hosts(cfg)
    frames = []
    for name, url in MARKET_FEEDS.items():
        try:
            body = fetch(url, ttl, hosts)
        except HostNotAllowed:
            continue
        frames.append(parse_rss(body, name))
    if not frames:
        return pd.DataFrame(columns=["title", "link", "published", "source",
                                     "feed", "tags"])
    D = dedupe(pd.concat(frames, ignore_index=True))
    if days and D["published"].notna().any():
        cut = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days)
        D = D[D["published"].isna() | (D["published"] >= cut)]
    if market_only:
        D = D[D["title"].map(is_market)]
    D["tags"] = D["title"].map(tag)
    # a tagged event outranks recency: a halt from this morning matters more
    # than a market wrap filed twenty minutes ago
    D["_ev"] = D["tags"].map(lambda x: any(t in STANDING_TAGS for t in x))
    D = D.sort_values(["_ev", "published"], ascending=[False, False])
    return D.drop(columns="_ev").head(limit).reset_index(drop=True)


def ticker_url(ticker: str) -> str:
    return ("https://news.google.com/rss/search?"
            + urllib.parse.urlencode({"q": f'"{ticker}" saham', "hl": "id",
                                      "gl": "ID", "ceid": "ID:id"}))


def ticker_news(tickers: Iterable[str], per: int = 3,
                ttl: float = 12 * 3600.0, cfg=None,
                days: int = 14, standing_days: int = 270) -> pd.DataFrame:
    """Headlines per ticker, filtered to items that actually name it.

    THE RELEVANCE FILTER IS NOT OPTIONAL. Many IDX tickers are ordinary
    Indonesian or English words — GULA (sugar), KOTA (city), RAJA (king), CASH,
    BABY, COAL — so an unfiltered query returns the commodity, the municipality
    and the nursery. Requiring the uppercase ticker as a standalone token in
    the headline cuts GULA from 100 items to 73 and KOTA from 100 to 40, and
    what survives is on-topic: the Indonesian financial press writes
    "Saham PACK" or "(PACK)" as a matter of house style.

    TWO WINDOWS, AND THE SECOND IS THE POINT. Google News returns 100 items
    ranked by relevance, not date, spanning years — for BABY the span was
    2018 to 2026 with only 3 items inside a fortnight. A flat 14-day cut
    therefore threw away BABY's April rights issue while keeping a "top
    gainers" listicle from the 13th, which is precisely backwards. Corporate
    actions change what the price series MEANS (§5 calls the rights adjustment
    the trap) and stay relevant for months, so :data:`STANDING_TAGS` items are
    kept for ``standing_days`` and everything else for ``days``. The ``recent``
    column marks which window an item came through.
    """
    hosts = allowed_hosts(cfg)
    cols = ["ticker", "title", "link", "published", "source", "feed", "tags",
            "recent"]
    out = []
    now = pd.Timestamp.now(tz="UTC")
    for t in tickers:
        t = str(t).upper()
        try:
            body = fetch(ticker_url(t), ttl, hosts)
        except HostNotAllowed:
            continue
        D = parse_rss(body, "google")
        if D.empty:
            continue
        pat = re.compile(rf"(?<![A-Z0-9]){re.escape(t)}(?![A-Z0-9])")
        D = D[D["title"].map(lambda s: bool(pat.search(str(s))))].copy()
        if D.empty:
            continue
        D["tags"] = D["title"].map(tag)
        D["standing"] = D["tags"].map(
            lambda x: any(g in STANDING_TAGS for g in x))
        age = (now - D["published"]).dt.days
        D["recent"] = D["published"].isna() | (age <= days)
        keep = D["recent"] | (D["standing"] & (age <= standing_days))
        D = D[keep]
        if D.empty:
            continue
        D = dedupe(D)
        D.insert(0, "ticker", t)
        # standing events first, then recency — a rights issue outranks a
        # market wrap however fresh the wrap is
        D = D.sort_values(["standing", "published"], ascending=[False, False])
        out.append(D.drop(columns="standing").head(per))
    if not out:
        return pd.DataFrame(columns=cols)
    return pd.concat(out, ignore_index=True)[cols]


def source_report(cfg=None) -> pd.DataFrame:
    """Which feeds answered, so a silent outage is visible in the brief.

    A narrative section that quietly shrank because one feed 403'd would read
    as a quiet day. This makes the difference between "nothing happened" and
    "nothing was fetched" explicit.
    """
    hosts = allowed_hosts(cfg)
    rows = []
    for name, url in MARKET_FEEDS.items():
        try:
            body = fetch(url, 6 * 3600.0, hosts)
            ok, n = body is not None, len(parse_rss(body, name))
        except HostNotAllowed:
            ok, n = False, 0
            name += " (not allowed)"
        rows.append({"feed": name, "ok": ok, "items": n,
                     "host": urllib.parse.urlparse(url).netloc})
    return pd.DataFrame(rows)
