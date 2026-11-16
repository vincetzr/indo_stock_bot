"""KSEI's monthly holding composition: every listed security, by investor type.

WHAT IT IS
----------
PT Kustodian Sentral Efek Indonesia is the central securities depository. Each
month it publishes ``BalanceposEfekYYYYMMDD.zip`` on www.ksei.co.id: one
pipe-delimited row per security giving the number of units held in KSEI's
scripless system, split into nine investor TYPES on each side of the
local/foreign line::

    IS insurance            CP corporate            PF pension fund
    IB financial institution ID individual          MF mutual fund
    SC securities company   FD foundation           OT other

plus the security's ``Sec. Num`` (units on issue) and month-end ``Price``.

WHY IT MATTERS TO THIS REPO
---------------------------
A6 and A8 closed CLAUDE.md §12 by saying the Taiwan and Finland results it
cites rest on ACCOUNT TYPES, "which those exchanges publish and IDX does not".
The first half is right. The second is a fact about IDX, not about Indonesia:
the depository publishes exactly that split, for every security, every month.
It is a POSITION, not a flow, so the starting-inventory problem that dominated
§9.3 (inventory negative 46.4% of the time) does not exist here — and the
month-on-month change in a type's holding IS that type's net flow, exactly,
with no top-10 censoring.

What it is NOT: an account. ``ID`` is every individual — the retail crowd and
also a founder holding 40% in their own name. ``CP`` is every corporate, which
on IDX is mostly controlling shareholders and their vehicles. CLAUDE.md §6.1's
warning applies with the labels changed: a type is a class of holder, not an
actor.

THE TWO IDENTITIES EVERY FILE MUST PASS
---------------------------------------
``Total`` (local) equals the sum of the nine local types, and the second
``Total`` equals the sum of the nine foreign types. A file that fails either is
refused rather than half-parsed: a misaligned column produces numbers that look
perfectly plausible. EQUITY rows are whole shares and must also be integral;
MUTUAL FUND units are fractional (``52541809.7586``), so the identity carries a
tolerance of a hundredth of a unit. The first version cast everything to
integers, truncated those fund units, and refused the 2018-01 file over five
"breaks" that were its own rounding. With the tolerance, all 106 files pass on
every row, equity and otherwise.

``Total + Total.1`` does NOT equal ``Sec. Num``. The gap is units outside the
scripless system (scrip certificates, unlisted tranches); the median EQUITY row
is 99.1% covered but the lower quartile only 43%. Shares of a type are
therefore computed against what KSEI actually holds, and ``coverage`` is
carried so a caller can refuse the thin rows.

POINT IN TIME
-------------
The month-end in the file name is the STATE date, not the date anyone could
read it. The zip member carries its creation stamp — 2026-11-02 06:00 for the
30 October file — and nothing earlier than that stamp can have been public.
:func:`available_on` turns the stamp into the first session a decision could
use, and never anything earlier than the stamp's own day plus a configurable
number of sessions. The listing page also carries a CMS date ("29 Okt" for the
30 October file) that PRECEDES the state date it describes, so it is not used
for anything.

HISTORY, AND THE PART THAT IS NOT LISTED
----------------------------------------
The page lists 36 files (November 2023 onward). Older files are still served at
the same path although no page links them: probed on 2026-11-15, January 2018
onward answers 200 and every 2017 candidate answers 302. Unlisted files can
disappear without notice, so the raw zips are cached permanently and a manifest
of their sizes, hashes and creation stamps is kept under ``data/reference/``.

THAT IS THE HISTORY OF ONE PATH, NOT OF THE ARCHIVE (review, 2026-11-16).
KSEI's legacy server ``web.ksei.co.id/Download/`` answered HEAD 200 for the
2009-12, 2012-12 and 2017-12 files (2007-12 404, 2008-12 500), and its copy of
2018-01 carries the SAME member text — CRC 0x0010c988, 263,916 bytes, identical
sha256 — stamped 2018-02-01 06:00 where this path's copy is stamped 2020-04-13.
So the 2020 stamps are a re-zip, at least for that month, and the archive is
probably older than 2018. Neither host fact is used: ``web.ksei.co.id`` is not
in ``data.ksei_allowed_hosts`` (A5 — the user's call), no pre-2018 file has been
opened, and a re-dated month is adopted only on an exact CRC match. The
manifest records the member CRC so a copy from either host can be checked by
CONTENT; its zip sha256 differs between hosts for the same text.

TERMS
-----
KSEI's "Penafian Umum" and "Ketentuan dan Kebijakan" pages (read 2026-11-15)
disclaim accuracy and govern K-CASH / AKSes accounts; neither restricts the
public statistics, and there is no counterpart to IDX's explicit
anti-crawling clause. That is a reading, not a licence. The host still lives in
``data.ksei_allowed_hosts`` so the user can withdraw it, the fetch is polite
and cached forever, and nothing here republishes the files — this repository is
PUBLIC, so the derived panel is gitignored and only the manifest is tracked.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import io
import os
import time
import urllib.parse
import zipfile
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

HOST = "www.ksei.co.id"
DEFAULT_HOSTS = (HOST,)
BASE_URL = "https://www.ksei.co.id/storage/Download/"
UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
      "Chrome/124.0 Safari/537.36")

#: First month on THIS path (2017 candidates answered 302). Not the first
#: month KSEI published: see "HISTORY" above for the legacy host.
FIRST_MONTH = (2018, 1)
CACHE_DIR = os.path.join("data", "cache", "ksei")
RAW_DIR = os.path.join(CACHE_DIR, "raw")
PANEL_PATH = os.path.join(CACHE_DIR, "ownership.parquet")
MANIFEST_PATH = os.path.join("data", "reference", "ksei_manifest.csv")
#: Someone else's public server. A courtesy, not a latency to tune. Was 2 s;
#: the host reset every TLS handshake for a while some hours after the
#: 106-file collection, and KSEI's legacy server reset the 7th request in
#: under three minutes at 2.5 s but none of 14 at >= 30 s (second review,
#: 2026-11-16). The cache is permanent, so this only costs a cold rebuild.
DELAY_SECONDS = 30.0
#: The only answers that mean "no such file". Anything else is "no answer".
ABSENT = frozenset({301, 302, 303, 307, 308, 404, 410})

CATEGORIES = ("IS", "CP", "PF", "IB", "ID", "MF", "SC", "FD", "OT")
CATEGORY_NAMES = {
    "IS": "insurance", "CP": "corporate", "PF": "pension fund",
    "IB": "financial institution", "ID": "individual", "MF": "mutual fund",
    "SC": "securities company", "FD": "foundation", "OT": "other",
}
HEADER = (["Date", "Code", "Type", "Sec. Num", "Price"]
          + [f"Local {c}" for c in CATEGORIES] + ["Total"]
          + [f"Foreign {c}" for c in CATEGORIES] + ["Total"])
LOCAL = [f"local_{c}" for c in CATEGORIES]
FOREIGN = [f"foreign_{c}" for c in CATEGORIES]
TIDY = (["month_end", "ticker", "type", "sec_num", "price"] + LOCAL
        + ["local_total"] + FOREIGN + ["foreign_total"])
#: Identity tolerance, in units. See :func:`parse` for why it is not zero.
UNIT_TOL = 0.01


class HostNotAllowed(RuntimeError):
    """Raised rather than fetching a host the config has not sanctioned."""


class ParseError(ValueError):
    """The file does not have the shape this parser was written for."""


# --------------------------------------------------------------------------- #
# hosts
# --------------------------------------------------------------------------- #

def allowed_hosts(cfg=None) -> List[str]:
    if cfg is None:
        return list(DEFAULT_HOSTS)
    v = cfg.get("data.ksei_allowed_hosts", None)
    return list(DEFAULT_HOSTS) if v is None else [str(h) for h in v]


def check_host(url: str, hosts: Sequence[str]) -> None:
    host = urllib.parse.urlparse(url).netloc.lower()
    if host not in {h.lower() for h in hosts}:
        raise HostNotAllowed(
            f"{host} is not in data.ksei_allowed_hosts. Add it there after "
            f"reading KSEI's terms, or leave it out.")


# --------------------------------------------------------------------------- #
# names and dates
# --------------------------------------------------------------------------- #

def file_name(day: dt.date) -> str:
    return f"BalanceposEfek{day:%Y%m%d}.zip"


def month_candidates(year: int, month: int,
                     calendar: Optional[Sequence[dt.date]] = None,
                     n: int = 7) -> List[dt.date]:
    """Dates to try for a month's file, latest first.

    The file is stamped with the month's last KSEI business day, which is
    normally the last IDX session. With a trading calendar the sessions of that
    month are tried first; weekdays fill in behind them, because a depository
    holiday and an exchange holiday are not guaranteed to coincide.
    """
    last = dt.date(year + (month == 12), month % 12 + 1, 1) - dt.timedelta(days=1)
    out: List[dt.date] = []
    if calendar is not None:
        out.extend(sorted({d for d in calendar
                           if d.year == year and d.month == month},
                          reverse=True)[:n])
    d = last
    weekdays: List[dt.date] = []
    while len(weekdays) < n:
        if d.weekday() < 5:
            weekdays.append(d)
        d -= dt.timedelta(days=1)
    out.extend(w for w in weekdays if w not in out)
    return out


def months(start: Tuple[int, int], end: Tuple[int, int]) -> List[Tuple[int, int]]:
    y, m = start
    out = []
    while (y, m) <= end:
        out.append((y, m))
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


# --------------------------------------------------------------------------- #
# fetching
# --------------------------------------------------------------------------- #

@dataclass
class Fetched:
    name: str
    blob: bytes
    from_cache: bool


def _raw_path(name: str, raw_dir: str) -> str:
    return os.path.join(raw_dir, name)


def fetch_file(name: str, hosts: Optional[Sequence[str]] = None,
               raw_dir: str = RAW_DIR, getter: Optional[Callable] = None,
               retries: int = 4, delay: float = DELAY_SECONDS
               ) -> Optional[Fetched]:
    """One zip, from the permanent cache or the server. None if absent.

    The server answers a missing file with a 302 to its home page rather than a
    404, so a REDIRECT, a 404 or a 410 is absence (``ABSENT``). Everything else
    that is not a zip — a 5xx, a 429, a 403, a 200 carrying an HTML page — is
    the server not answering, and is retried like a connection reset. Resets
    are frequent and transient — four probes in a row reset on 2026-11-15 and
    the fifth succeeded — so only an exhausted retry budget is reported, as an
    exception, never as "absent": "the server would not answer" and "the file
    does not exist" are different facts and the second one ends a backfill.

    The first version returned None for ANY non-200, which put a 503 on the
    same footing as a missing file (review, 2026-11-16); a month recorded as
    absent that way then made the NEXT month look consecutive downstream.
    """
    url = BASE_URL + name
    check_host(url, allowed_hosts() if hosts is None else hosts)
    p = _raw_path(name, raw_dir)
    if os.path.exists(p) and os.path.getsize(p) > 0:
        with open(p, "rb") as fh:
            return Fetched(name, fh.read(), True)
    if getter is None:
        import requests
        sess = requests.Session()
        sess.headers["User-Agent"] = UA

        def getter(u):                                   # noqa: E306
            r = sess.get(u, timeout=60, allow_redirects=False)
            return r.status_code, r.content

    last_err: Optional[Exception] = None
    for k in range(retries):
        try:
            code, body = getter(url)
        except Exception as e:                           # noqa: BLE001
            last_err = e
            time.sleep(delay * (k + 1))
            continue
        time.sleep(delay)
        if code in ABSENT:
            return None
        if code != 200 or not body or not body.startswith(b"PK"):
            last_err = RuntimeError(f"HTTP {code}, {len(body or b'')} bytes, "
                                    f"not a zip")
            time.sleep(delay * (k + 1))
            continue
        os.makedirs(raw_dir, exist_ok=True)
        tmp = p + ".tmp"
        with open(tmp, "wb") as fh:
            fh.write(body)
        os.replace(tmp, p)
        return Fetched(name, body, False)
    raise ConnectionError(f"{name}: no answer after {retries} tries "
                          f"({last_err!r})")


def _member(blob: bytes) -> Tuple[zipfile.ZipInfo, bytes]:
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        infos = [i for i in z.infolist() if not i.is_dir()]
        if len(infos) != 1:
            raise ParseError(f"expected one member, found {len(infos)}")
        return infos[0], z.read(infos[0])


def read_zip(blob: bytes) -> Tuple[str, str, dt.datetime]:
    """(text, member name, member creation stamp). Exactly one member."""
    i, raw = _member(blob)
    return (raw.decode("utf-8", errors="replace"), i.filename,
            dt.datetime(*i.date_time))


# --------------------------------------------------------------------------- #
# parsing
# --------------------------------------------------------------------------- #

def parse(text: str, equity_only: bool = True) -> pd.DataFrame:
    """Tidy frame from one file's text, after both identities are checked.

    Raises :class:`ParseError` on a header that is not the one this was written
    for, on a file carrying more than one state date, and on ANY row where a
    side's nine types do not sum to that side's total. Refusing the whole file
    on one bad row is deliberate: the identity can only fail through a shifted
    column or a mangled number, and either corrupts rows that still pass.
    """
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        raise ParseError("empty file")
    head = [h.strip() for h in lines[0].split("|")]
    if head != HEADER:
        raise ParseError(f"unexpected header: {head!r}")
    rows = [ln.split("|") for ln in lines[1:]]
    bad = [i for i, r in enumerate(rows) if len(r) != len(HEADER)]
    if bad:
        raise ParseError(f"{len(bad)} rows with {len(rows[bad[0]])} fields, "
                         f"expected {len(HEADER)}")
    df = pd.DataFrame(rows, columns=TIDY)
    for c in ("ticker", "type"):
        df[c] = df[c].str.strip()
    dates = pd.to_datetime(df["month_end"].str.strip(), format="%d-%b-%Y",
                           errors="coerce")
    if dates.isna().any():
        raise ParseError(f"{int(dates.isna().sum())} unparseable dates")
    if dates.nunique() != 1:
        raise ParseError(f"{dates.nunique()} state dates in one file")
    df["month_end"] = dates
    num = ["sec_num", "price"] + LOCAL + ["local_total"] + FOREIGN + ["foreign_total"]
    for c in num:
        df[c] = pd.to_numeric(df[c].str.strip().replace("", "0"),
                              errors="coerce")
    if df[num].isna().any().any():
        raise ParseError("non-numeric holdings")
    hold = LOCAL + ["local_total"] + FOREIGN + ["foreign_total"]
    # MUTUAL FUND units are fractional (52541809.7586), so their sums carry
    # float noise of ~1e-8; the first version cast everything to int64, which
    # truncated those units and then "found" an identity break in 5 rows of
    # the 2018-01 file that did not exist. The tolerance is a hundredth of a
    # unit: float noise passes it, a shifted column or a mangled digit cannot.
    dl = (df[LOCAL].sum(axis=1) - df["local_total"]).abs()
    dfo = (df[FOREIGN].sum(axis=1) - df["foreign_total"]).abs()
    if (dl > UNIT_TOL).any() or (dfo > UNIT_TOL).any():
        raise ParseError(f"identity broken: {int((dl > UNIT_TOL).sum())} local "
                         f"rows, {int((dfo > UNIT_TOL).sum())} foreign rows")
    eq = df["type"] == "EQUITY"
    frac = (df.loc[eq, hold] % 1 != 0).any(axis=1)
    if frac.any():
        raise ParseError(f"{int(frac.sum())} EQUITY rows carry fractional "
                         f"shares — shares are whole units, so the columns "
                         f"are not what this parser thinks they are")
    if equity_only:
        df = df[eq].copy()
        # Whole shares, so from here on the identities are exact integers.
        df[hold] = df[hold].astype("int64")
    return df.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# point in time
# --------------------------------------------------------------------------- #

def available_on(stamp: dt.datetime, sessions: Sequence[pd.Timestamp],
                 lag: int = 1) -> Optional[pd.Timestamp]:
    """First session a decision could use a file created at ``stamp``.

    ``lag=1`` means the first session STRICTLY AFTER the stamp's own day: a file
    written at 06:00 is public at best that morning, and acting at that day's
    close on it would assume someone downloaded and processed it inside the
    session. Larger lags are a sensitivity, never smaller ones.
    """
    s = pd.DatetimeIndex(sorted(pd.to_datetime(list(sessions))))
    i = s.searchsorted(pd.Timestamp(stamp.date()), side="right")
    j = i + lag - 1
    return s[j] if j < len(s) else None


# --------------------------------------------------------------------------- #
# the whole archive
# --------------------------------------------------------------------------- #

def fetch_month(year: int, month: int,
                calendar: Optional[Sequence[dt.date]] = None,
                **kw) -> Optional[Tuple[dt.date, Fetched]]:
    for day in month_candidates(year, month, calendar):
        got = fetch_file(file_name(day), **kw)
        if got is not None:
            return day, got
    return None


def manifest_row(day: dt.date, got: Fetched) -> Dict[str, object]:
    """One manifest line. ``sha256`` identifies the ZIP; ``member_crc`` and
    ``text_sha256`` identify the CONTENT, which is what a copy fetched from
    another host or re-zipped by KSEI must match (the legacy copy of 2018-01
    has a different zip hash and an identical text hash)."""
    i, raw = _member(got.blob)
    df = parse(raw.decode("utf-8", errors="replace"), equity_only=False)
    return {"month_end": pd.Timestamp(day), "file": got.name,
            "member": i.filename, "created": pd.Timestamp(dt.datetime(*i.date_time)),
            "bytes": len(got.blob),
            "sha256": hashlib.sha256(got.blob).hexdigest(),
            "member_crc": f"{i.CRC:08x}", "member_bytes": i.file_size,
            "text_sha256": hashlib.sha256(raw).hexdigest(),
            "rows": len(df), "equity_rows": int((df["type"] == "EQUITY").sum())}


def build_panel(raw_dir: str = RAW_DIR) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """(equity panel, manifest) from every cached zip. No network."""
    frames, man = [], []
    for name in sorted(os.listdir(raw_dir)):
        if not (name.startswith("BalanceposEfek") and name.endswith(".zip")):
            continue
        with open(os.path.join(raw_dir, name), "rb") as fh:
            blob = fh.read()
        day = dt.datetime.strptime(name[14:22], "%Y%m%d").date()
        text, member, stamp = read_zip(blob)
        df = parse(text)
        if df["month_end"].iloc[0].date() != day:
            raise ParseError(f"{name}: state date {df['month_end'].iloc[0]} "
                             f"does not match the file name")
        df["created"] = pd.Timestamp(stamp)
        frames.append(df)
        man.append(manifest_row(day, Fetched(name, blob, True)))
    if not frames:
        return pd.DataFrame(columns=TIDY + ["created"]), pd.DataFrame()
    P = pd.concat(frames, ignore_index=True)
    return P, pd.DataFrame(man).sort_values("month_end").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# derived quantities
# --------------------------------------------------------------------------- #

def month_index(month_end) -> pd.Series:
    """Calendar month as an integer, so "consecutive" means the CALENDAR.

    Every consecutiveness test in the first version was relative to the months
    present in the archive, so a whole missing file made the next month look
    consecutive and a two-month change entered as a one-month flow (review,
    2026-11-16). It fired on nothing — all 106 months are present — and it is
    the kind of defect that would print a believable number the day it did.
    """
    m = pd.to_datetime(pd.Series(month_end))
    return m.dt.year * 12 + m.dt.month


def months_missing(P: pd.DataFrame) -> List[str]:
    """Calendar months inside the panel's span with no file, as 'YYYY-MM'."""
    if P.empty:
        return []
    have = set(pd.to_datetime(P["month_end"]).dt.to_period("M"))
    span = pd.period_range(min(have), max(have), freq="M")
    return [str(p) for p in span if p not in have]


#: Investor-class groupings, defined from the type labels alone.
GROUPS = {
    "retail": ["local_ID"],
    "foreign": FOREIGN,
    "strategic": ["local_CP", "foreign_CP"],
    "local_inst": ["local_IS", "local_PF", "local_MF", "local_IB"],
    "local_mf": ["local_MF"],
}


def shares(P: pd.DataFrame) -> pd.DataFrame:
    """Each group's share of what KSEI holds, plus coverage and cap.

    Shares are taken against KSEI's held units, not ``sec_num``, because units
    outside the scripless system cannot trade on the exchange without first
    being converted — a share of ``sec_num`` would read a scrip-heavy name as a
    name nobody holds.
    """
    out = P[["month_end", "ticker", "sec_num", "price"]].copy()
    held = (P["local_total"] + P["foreign_total"]).astype(float)
    out["held"] = held
    with np.errstate(divide="ignore", invalid="ignore"):
        out["coverage"] = np.where(P["sec_num"] > 0, held / P["sec_num"], np.nan)
        for g, cols in GROUPS.items():
            out[g] = np.where(held > 0, P[cols].sum(axis=1) / held, np.nan)
    out["mcap"] = P["sec_num"].astype(float) * P["price"].astype(float)
    return out


def changes(S: pd.DataFrame, cols: Sequence[str] = tuple(GROUPS)) -> pd.DataFrame:
    """Month-on-month change in each share, WITHIN ticker, consecutive only.

    A change is defined only between a ticker's file and the IMMEDIATELY
    preceding CALENDAR month. If a name — or the whole archive — is missing
    for a month, the next change is NaN rather than a two-month change
    mislabelled as one: A11's "never roll on a pivot" in its monthly form.
    """
    S = S.sort_values(["ticker", "month_end"]).copy()
    S["_k"] = month_index(S["month_end"]).to_numpy()
    prev_k = S.groupby("ticker")["_k"].shift(1)
    ok = (S["_k"] - prev_k) == 1
    for c in cols:
        d = S[c] - S.groupby("ticker")[c].shift(1)
        S[f"d_{c}"] = d.where(ok)
    return S.drop(columns="_k")


# --------------------------------------------------------------------------- #
# transfers between types, which are not trades
# --------------------------------------------------------------------------- #

#: A month is a probable TRANSFER when a type moves more than this share of
#: the previous balance ...
TRANSFER_MIN = 0.01
#: ... and some single type ON THE OTHER SIDE moves the opposite way by an
#: amount within this relative distance of it.
TRANSFER_MATCH = 0.10
#: The near-exact variant, printed as a sensitivity: at 1% the shifted-month
#: placebo finds 18-29 chance mirrors against 466 real ones (foreign partners)
#: and 23-29 against 1,191 (local partners); at 10% it is 168-261 against
#: 1,272 and 2,511.
TRANSFER_EXACT = 0.01


def _mirrored(P: pd.DataFrame, movers: Sequence[str], partners: Sequence[str],
              match: float) -> pd.Series:
    """True where a column in ``movers`` moves > TRANSFER_MIN of the previous
    balance and one column in ``partners`` (other than itself) moves the
    opposite way within ``match`` of it. Calendar-consecutive months only."""
    P = P.sort_values(["ticker", "month_end"])
    held = (P["local_total"] + P["foreign_total"]).astype(float)
    prev_held = held.groupby(P["ticker"]).shift(1)
    k = month_index(P["month_end"]).to_numpy()
    cons = pd.Series(k, index=P.index).groupby(P["ticker"]).diff() == 1
    d = P[list(dict.fromkeys(list(movers) + list(partners)))].astype(float) \
        .groupby(P["ticker"]).diff()
    out = pd.Series(False, index=P.index)
    for a in movers:
        da = d[a]
        B = d[[c for c in partners if c != a]]
        opp = np.sign(B).mul(-np.sign(da), axis=0) > 0
        near = (B.abs().sub(da.abs(), axis=0)).abs().le(match * da.abs(), axis=0)
        out |= (da.abs() > TRANSFER_MIN * prev_held) & (opp & near).any(axis=1)
    return (out & cons).reindex(P.index).fillna(False).astype(bool)


def transfer_flags(P: pd.DataFrame, partners: str = "all",
                   match: float = TRANSFER_MATCH) -> pd.Series:
    """True where local_ID's change is mirrored by one other type's.

    WHY THIS EXISTS. In October 2026 BYAN's individual holding fell by
    10,751,166,170 shares while its corporate holding rose by 10,751,301,170.
    The balance barely moved, so the share-count and balance filters let it
    through, and read as flow it was retail "selling" Rp 98 trillion in one
    month: 88% of that month's absolute retail flow across the whole market
    came from three names. A type-to-type move of matching size is consistent
    with one holder re-registering a block — KSEI carries no holder identity,
    so WHO moved it is not observable — and it is not the crowd.

    THE MIRROR IS SEARCHED ON BOTH SIDES OF THE DOMICILE LINE (review,
    2026-11-16). The first version searched only the other LOCAL types, on the
    premise that an individual-to-foreign mirror is "the crowd selling to
    foreigners". The data says otherwise: 122 local_ID moves are mirrored by a
    single foreign type to within a millionth (0-1 by chance), and the same
    blocks travel back — EMTK's foreign_ID moved +4,990,264,500, −4,989,570,900
    and +4,994,405,900 in 2023-01, -02 and -07; PNGO's whole individual
    holding (98.3% of the balance) went to foreign_CP in 2026-05. Those passed
    as crowd flow. ``partners="local"`` reproduces the first version.

    Calendar-consecutive months only, within ticker; a name's first month is
    False.
    """
    if partners not in ("all", "local"):
        raise ValueError(partners)
    others = [c for c in LOCAL if c != "local_ID"]
    if partners == "all":
        others = others + FOREIGN
    return _mirrored(P, ["local_ID"], others, match)


def foreign_transfer_flags(P: pd.DataFrame,
                           match: float = TRANSFER_MATCH) -> pd.Series:
    """True where a foreign type's change is mirrored by one LOCAL type's.

    The foreign-side counterpart, for anything that reads foreign_total. A
    local-to-local transfer leaves foreign_total unchanged and a
    foreign-to-foreign one cancels inside it, so only a cross-domicile mirror
    can contaminate it — which is exactly the case the first version could not
    see.
    """
    return _mirrored(P, FOREIGN, LOCAL, match)
