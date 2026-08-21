"""Broker summary from IndoPremier's public stock module.

Background: why this file exists
--------------------------------
For most of this project's life the answer to "where does broker summary come
from" was "nowhere you can reach". ``idx.co.id`` answers 403 to every path -
including its own homepage, which is the tell: the block is on the requesting
network, not on the endpoint. Commercial platforms are not defeating that WAF,
they are licensed IDX data-feed subscribers or they run from Indonesian
networks. The asymmetry was never technical skill.

IndoPremier (PT Indo Premier Sekuritas, an IDX member) renders the same table
into a public, unauthenticated page. One GET returns the full top-10 rekap
broker for a stock over any date range::

    /module/saham/include/data-brokersummary.php?code=BBCA&start=&end=&board=RG

What this source is, precisely
------------------------------
* **Top 10 buyers and top 10 sellers**, ranked independently. The two columns
  of a row are unrelated - row 3's buyer and row 3's seller are simply the
  third-largest of each side. Never read a row as one broker's two sides.
* **Board matters more than anything else here.** The default (all boards)
  folds in the negotiated market, where block crossings print at arbitrary
  prices. Measured on 2026-08-13: GOTO all-board reports 25.0M lots against a
  regular-market 196k - a factor of 127. This module therefore defaults to
  ``board="RG"`` and you should have a specific reason to change it.
* **Regular-board totals reconcile to the tape.** Same date, against Yahoo's
  daily volume: BBCA 832,077 lots vs 832,080, ANTM 860,776 vs 860,777, ASII
  484,063 vs 484,073, UNVR 99,947 vs 99,958. Agreement to ~1e-5.
* **Foreign/domestic is flagged by the source itself** via a CSS class, so
  :func:`foreign_flags` recovers IndoPremier's own F/D classification. Over 30
  stocks x 3 dates no code ever changed its flag, and it agrees with
  ``config/brokers.yaml`` on every confident entry - including YP (Mirae Asset)
  as foreign, despite YP being Indonesia's largest *retail* broker. That is
  exactly the trap ``foreign_basis`` exists to handle in
  :mod:`idxbot.bandarmology`.

  It is *not* a pure ownership basis, though, and assuming so will burn you:
  BQ, DR and TP have Korean, Malaysian and Singaporean parents yet come back
  domestic here, consistently. The flag appears to follow the member's own
  registration rather than its shareholders. Those three are left disagreeing
  on purpose - see the note at the foot of ``config/brokers.yaml``.
* **History runs to roughly 2008.** 2005 returns zeroes, 2008-06-16 returns a
  populated table. Prices are as-traded and unadjusted, which is correct for
  broker summary and means they will not line up with split-adjusted closes.

Precision limit, stated plainly
-------------------------------
Figures at or above one million are abbreviated for display - ``3.4 M``,
``699.9 B`` - so they carry 2-3 significant figures, while anything smaller is
exact. Average prices are always exact.

Only the lot column is worth worrying about. Value is suffixed for practically
every stock and is also the redundant column, recoverable as lots x 100 x
average; rounded lots are recoverable from nothing. So a day whose *lots* were
abbreviated is tagged ``ipot~`` in the ``source`` field and travels that way
through every downstream report. Ratio-shaped metrics (loyalty, presence,
dominance, net direction) are unaffected at that resolution; a rupiah-exact
cost basis is not available from this route. :func:`consistency` turns the same
redundancy into an acceptance test - value *should* equal lots x 100 x average,
so a parse that reads the wrong column fails by orders of magnitude while
display rounding costs a fraction of a percent.

Terms of use
------------
This is a public page on a licensed member's site, read the way a browser reads
it. It is still someone else's server and IDX still restricts redistribution of
its market data, so this module fetches one day at a time, sleeps between
requests by default, and caches everything it has already seen. Do not turn it
into a bulk harvester, and do not redistribute what it returns.
"""

from __future__ import annotations

import re
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .broker_summary import LOT_SIZE, BrokerSummaryProvider, empty_frame

BASE_URL = "https://www.indopremier.com/module/saham/include/data-brokersummary.php"

#: Suffix multipliers. English and Indonesian forms both appear in the wild.
#: Verified against the table's own arithmetic: 18.4 M lots x 100 x 9,120 =
#: 16.8 T, which fixes M=1e6 / B=1e9 / T=1e12 and rules out M="miliar".
_MULTIPLIER: Dict[str, float] = {
    "": 1.0, "K": 1e3, "RB": 1e3, "JT": 1e6,
    "M": 1e6, "B": 1e9, "T": 1e12,
}

#: Largest ``|reported - implied| / reported`` attributable to display
#: rounding alone. Abbreviated values carry one decimal, so the error is
#: 0.05/mantissa, maximised at 5% for a mantissa just above 1.0. Anything
#: beyond this is a parse fault, not rounding.
DISPLAY_ROUNDING_BOUND = 0.05

_ROW_RE = re.compile(r"<tr[^>]*>(.*?)</tr>", re.S | re.I)
_CELL_RE = re.compile(r"<td[^>]*>(.*?)</td>", re.S | re.I)
_TAG_RE = re.compile(r"<[^>]+>")
_FOOT_RE = re.compile(r'fsize12[^>]*>([^<]+)<', re.S | re.I)
#: The source classifies every member into one of THREE buckets, not two.
#: Missing the third is easy and expensive: an early version of this parser
#: matched only foreign|local and silently dropped every state-owned broker -
#: including DX, CC and OD, which are among the largest desks on the exchange.
_CLASS_RE = re.compile(
    r'class="text-(foreign|local|bumn)"[^>]*>\s*([A-Z]{2})\s*<', re.I)

#: ``bumn`` = Badan Usaha Milik Negara, an Indonesian state-owned enterprise.
BROKER_CLASSES = ("foreign", "bumn", "local")

BOARDS = ("RG", "TN", "NG", "all")


def parse_number(text: str) -> float:
    """Parse ``3.4 M``, ``771,597``, ``-2.5 T``, ``802.0 M`` or ``0``.

    Returns NaN rather than raising on anything unrecognised, because a single
    odd cell must not take down a whole day's fetch - :func:`consistency` is
    what catches a systematically wrong parse.
    """
    s = _TAG_RE.sub("", str(text)).strip().replace(",", "").replace("\xa0", " ")
    if not s or s in {"-", "--"}:
        return float("nan")
    m = re.match(r"^(-?\d*\.?\d+)\s*([A-Za-z]*)$", s)
    if not m:
        return float("nan")
    unit = m.group(2).upper()
    if unit not in _MULTIPLIER:
        return float("nan")
    return float(m.group(1)) * _MULTIPLIER[unit]


def is_abbreviated(text: str) -> bool:
    """True when the cell was rendered with a magnitude suffix, so it is rounded."""
    s = _TAG_RE.sub("", str(text)).strip()
    return bool(re.search(r"\d\s*[A-Za-z]+$", s))


def parse_table(html: str, ticker: str, date: pd.Timestamp,
                source: str = "ipot") -> pd.DataFrame:
    """Turn one rendered rekap-broker table into the canonical schema.

    The two halves of each row are separate rankings, so they are unstacked
    into buy-side and sell-side records and only then joined on broker code. A
    broker that appears on both sides ends up as one row carrying both.
    """
    if "<tbody>" not in html:
        return empty_frame()
    body = html.split("<tbody>", 1)[1].split("</tbody>", 1)[0]

    buys: Dict[str, Tuple[float, float, float]] = {}
    sells: Dict[str, Tuple[float, float, float]] = {}
    rounded = False
    for chunk in _ROW_RE.findall(body):
        cells = _CELL_RE.findall(chunk)
        if len(cells) != 9:
            continue
        for offset, sink in ((0, buys), (5, sells)):
            code = _TAG_RE.sub("", cells[offset]).strip().upper()
            if not re.fullmatch(r"[A-Z]{2}", code):
                continue
            lot, val, avg = (parse_number(cells[offset + i]) for i in (1, 2, 3))
            if not np.isfinite(lot) or lot <= 0:
                continue
            # Only the LOT cell decides the marker. Value is suffixed for
            # essentially every stock - a hundred million rupiah already reads
            # "100.0 M" - so flagging on value would mark every row ever
            # fetched and mean nothing. Value is also the redundant column:
            # given exact lots and an exact average, it can be recomputed.
            # Rounded lots cannot be recovered from anything.
            rounded = rounded or is_abbreviated(cells[offset + 1])
            sink[code] = (lot, val, avg)

    if not buys and not sells:
        return empty_frame()

    rows: List[Dict[str, object]] = []
    for code in sorted(set(buys) | set(sells)):
        b_lot, b_val, b_avg = buys.get(code, (0.0, 0.0, np.nan))
        s_lot, s_val, s_avg = sells.get(code, (0.0, 0.0, np.nan))
        rows.append({
            "date": pd.Timestamp(date).normalize(), "ticker": str(ticker).upper(),
            "broker": code,
            "buy_lot": b_lot, "buy_val": b_val, "buy_avg": b_avg,
            "sell_lot": s_lot, "sell_val": s_val, "sell_avg": s_avg,
            "source": f"{source}~" if rounded else source,
        })
    out = pd.DataFrame(rows)
    out["date"] = pd.to_datetime(out["date"])
    return out


def parse_totals(html: str) -> Dict[str, float]:
    """Read the table footer: total value, net foreign value, total lots, VWAP.

    ``F. NVal`` is IndoPremier's own net-foreign figure for the stock and day.
    It is computed on the F-flagged *members*, so it belongs to the F-broker
    definition, not to IDX's per-trade foreign-investor flag.
    """
    if "<tfoot>" not in html:
        return {}
    foot = html.split("<tfoot>", 1)[1]
    out: Dict[str, float] = {}
    for item in _FOOT_RE.findall(foot):
        key, _, value = item.partition(":")
        key = re.sub(r"[^a-z]", "", key.lower())
        if key:
            out[key] = parse_number(value)
    # tval / fnval / tlot / avg
    return out


def exact_total_lots(totals: Dict[str, float]) -> float:
    """Recover total lots from value and VWAP instead of reading the rounded cell.

    The footer prints ``T.Lot`` through the same abbreviator as everything else,
    so BBCA on a busy day reads ``1.1 M`` - two significant figures on the one
    number the whole censored-inference below is measured against. But the same
    footer also prints ``T.Val`` to four figures and ``Avg`` exactly, and
    value = lots x 100 x average. Dividing recovers the lot count to about 0.03%
    where the printed cell is out by up to 5%.

    Falls back to the printed cell when value or average is missing.
    """
    tval, avg = totals.get("tval"), totals.get("avg")
    if tval and avg and np.isfinite(tval) and np.isfinite(avg) and avg > 0:
        return float(tval) / (LOT_SIZE * float(avg))
    v = totals.get("tlot", float("nan"))
    return float(v) if v is not None else float("nan")


#: Columns carrying the day's market-wide totals, repeated on every row of that
#: day so a row-group survives a concat, a groupby or a round-trip to CSV
#: without a second table to join back to.
TOTAL_COLUMNS = ("total_lot", "total_val", "foreign_net_val", "vwap")


def attach_totals(df: pd.DataFrame, totals: Dict[str, float]) -> pd.DataFrame:
    """Carry the day's market-wide totals alongside the top-10 rows.

    THE POINT OF THIS FUNCTION. Without the totals a top-10 table is a biased
    sample of unknown size and nothing quantitative survives it - which is why
    every earlier reading of this source stopped at "direction and ranking
    survive, absolute positions do not". With them it is a CENSORED sample of
    known size: the visible brokers are named and measured, the rest are unnamed
    but their aggregate is exactly ``total - visible`` and each of them is
    individually smaller than the tenth-ranked broker. That is enough to bracket
    every quantity the ledger wants, instead of abandoning it.

    The footer was already being parsed by :func:`parse_totals` and then thrown
    away, because ``fetch_day`` never called it.
    """
    if df is None or df.empty:
        return df if df is not None else empty_frame()
    out = df.copy()
    out["total_lot"] = exact_total_lots(totals) if totals else np.nan
    out["total_val"] = totals.get("tval", np.nan) if totals else np.nan
    out["foreign_net_val"] = totals.get("fnval", np.nan) if totals else np.nan
    out["vwap"] = totals.get("avg", np.nan) if totals else np.nan
    return out


def broker_classes(html: str) -> Dict[str, str]:
    """The source's own three-way classification, keyed by broker code.

    Returns one of ``"foreign"``, ``"bumn"`` or ``"local"``. The middle one is
    the interesting one and the easy one to lose: ``bumn`` marks Indonesian
    state-owned houses - Bahana, Mandiri Sekuritas, BRI Danareksa - which the
    repo's own registry lumps into a generic ``local_inst`` tier. State-linked
    buying is a distinct actor from ordinary domestic institutional buying, and
    this is a free label for it straight from an exchange member.
    """
    return {code.upper(): kind.lower() for kind, code in _CLASS_RE.findall(html)}


def foreign_flags(html: str) -> Dict[str, bool]:
    """Collapse :func:`broker_classes` to the foreign / not-foreign question.

    Useful as an independent check on ``config/brokers.yaml``. Note this is not
    a pure ownership basis: YP (Mirae) is foreign-owned and retail-serving and
    comes back foreign, while BQ, DR and TP have foreign parents and come back
    domestic. See the note at the foot of ``config/brokers.yaml``.
    """
    return {code: kind == "foreign" for code, kind in broker_classes(html).items()}


def consistency(df: pd.DataFrame) -> pd.DataFrame:
    """Check each row against ``value == lots x 100 x average``.

    The table publishes lots, value and average price, and any two of those
    determine the third. That redundancy is the only self-contained way to
    prove the parser read the right columns in the right units, so it is worth
    more than any amount of eyeballing.

    The tolerance is not a fitted number. Abbreviated values carry one decimal
    place, so a figure rendered ``x.y U`` is rounded to +/-0.05 of its mantissa
    and the worst relative error is ``0.05 / mantissa`` - 5% when the mantissa
    is just above 1.0, falling to 0.5% at 9.9. Measured over 160 rows across
    eight stocks the pooled median is 0.2% and the maximum 4.5%, sitting under
    that bound as it must. So :data:`DISPLAY_ROUNDING_BOUND` is the line: below
    it is display rounding, and a swapped or mis-scaled column lands orders of
    magnitude above it, never in between.
    """
    if df.empty:
        return pd.DataFrame(columns=["broker", "side", "reported", "implied", "rel_err"])
    rows = []
    for side in ("buy", "sell"):
        sub = df[df[f"{side}_lot"] > 0]
        implied = sub[f"{side}_lot"] * LOT_SIZE * sub[f"{side}_avg"]
        reported = sub[f"{side}_val"]
        with np.errstate(divide="ignore", invalid="ignore"):
            rel = (reported - implied).abs() / reported.replace(0, np.nan)
        rows.append(pd.DataFrame({"broker": sub["broker"], "side": side,
                                  "reported": reported, "implied": implied,
                                  "rel_err": rel}))
    return pd.concat(rows, ignore_index=True)


def truncation_bias(summary: pd.DataFrame) -> Dict[str, float]:
    """Measure how badly a top-10 view distorts a cumulative inventory ledger.

    This is the one limitation of this source that can actually mislead, and it
    is invisible unless you look for it.

    In a *complete* rekap broker every lot bought is a lot sold, so summing all
    members gives exactly zero net on every session. A top-10 view breaks that
    identity, and not randomly: a broker only appears on the side where it was
    large that day. A desk that is a steady accumulator shows up among the top
    buyers constantly and among the top sellers rarely, so its *unobserved*
    selling is censored away and the reconstructed inventory marches upward
    whether or not it actually bought anything on net.

    Measured on BBCA over 52 sessions to 2026-08-13: DX appears as a top-10
    buyer on 21 days and a top-10 seller on 5, and the market-wide cumulative
    net - which must be zero - comes to -2.8 million lots.

    So *direction over a window* and *relative size between brokers* survive
    this. An absolute position, and any cost basis or open P/L derived from
    one, does not. Returns ``imbalance_*`` (per-session, signed) and
    ``cumulative_net_lot`` (the drift the ledger integrates).
    """
    if summary is None or summary.empty:
        return {}
    per_day = summary.groupby("date")[["buy_lot", "sell_lot"]].sum()
    scale = (per_day["buy_lot"] + per_day["sell_lot"]) / 2.0
    imbalance = (per_day["buy_lot"] - per_day["sell_lot"]) / scale.replace(0, np.nan)
    net = float((per_day["buy_lot"] - per_day["sell_lot"]).sum())
    observed = float(per_day["buy_lot"].sum() + per_day["sell_lot"].sum())
    return {
        "sessions": float(len(per_day)),
        "imbalance_mean": float(imbalance.mean()),
        "imbalance_abs_median": float(imbalance.abs().median()),
        "imbalance_abs_max": float(imbalance.abs().max()),
        "cumulative_net_lot": net,
        "cumulative_net_share": abs(net) / observed if observed else float("nan"),
    }


def is_truncated(summary: pd.DataFrame) -> bool:
    """True when the frame came from a top-N source, so the ledger is censored."""
    if summary is None or summary.empty or "source" not in summary:
        return False
    return summary["source"].astype(str).str.startswith("ipot").any()


class IpotBrokerSummary(BrokerSummaryProvider):
    """Fetches rekap broker one trading day at a time, with caching.

    ``delay`` is not a tuning knob to minimise. This reads someone else's
    public page; the polite default stays.
    """

    name = "ipot"
    is_real = True

    def __init__(self, cache=None, board: str = "RG", delay: float = 1.2,
                 timeout: int = 30, session_type: str = "all",
                 max_days: int = 400, verbose: bool = False,
                 retries: int = 3):
        if board not in BOARDS:
            raise ValueError(f"board must be one of {BOARDS}, got {board!r}")
        self.cache = cache
        self.board = board
        self.delay = float(delay)
        self.timeout = timeout
        self.session_type = session_type
        self.max_days = int(max_days)
        self.verbose = verbose
        self.retries = max(1, int(retries))
        self._last_call = 0.0

    def available(self) -> bool:
        try:
            import requests  # noqa: F401
        except ImportError:
            return False
        return True

    # -- network ----------------------------------------------------------
    def _get(self, ticker: str, day: pd.Timestamp) -> str:
        import requests

        gap = self.delay - (time.time() - self._last_call)
        if gap > 0:
            time.sleep(gap)
        params = {"code": str(ticker).upper(),
                  "start": day.strftime("%Y-%m-%d"),
                  "end": day.strftime("%Y-%m-%d")}
        if self.board != "all":
            params["board"] = self.board
        if self.session_type != "all":
            params["fd"] = self.session_type
        headers = {
            "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                           "AppleWebKit/537.36 (KHTML, like Gecko) "
                           "Chrome/126.0 Safari/537.36"),
            "Referer": "https://www.indopremier.com/",
            "Accept": "text/html,application/xhtml+xml,*/*;q=0.8",
        }
        # About one request in five comes back as a reset connection rather than
        # an HTTP error. Retrying that is not impoliteness - it is one page that
        # did not arrive - but the backoff lengthens rather than tightens, so a
        # server having a bad minute is left alone rather than hammered.
        last: Optional[Exception] = None
        for attempt in range(self.retries):
            if attempt:
                time.sleep(self.delay * (2 ** attempt))
            try:
                resp = requests.get(BASE_URL, params=params,
                                    timeout=self.timeout, headers=headers)
                self._last_call = time.time()
                resp.raise_for_status()
                return resp.text
            except Exception as exc:                       # noqa: BLE001
                last = exc
                self._last_call = time.time()
        raise last if last is not None else RuntimeError("no response")

    def fetch_day(self, ticker: str, day: pd.Timestamp) -> pd.DataFrame:
        """One ticker, one session. Cached forever once seen - it never changes.

        Fetching a day at a time rather than asking for the whole range in one
        call is not only about politeness: a range query sums the lots first
        and only then abbreviates, so a month of BBCA comes back as ``6.8 M``
        where the individual sessions are exact to the lot. Narrow queries are
        simply more precise.
        """
        day = pd.Timestamp(day).normalize()
        # The view MUST be part of the key. all / F / D are three different
        # tables for the same ticker-day, and a key that omits the view would
        # serve a foreign-only table to a caller asking for the whole market.
        view = "" if self.session_type == "all" else f"_{self.session_type}"
        key = f"{str(ticker).upper()}_{day:%Y%m%d}_{self.board}{view}"
        if self.cache is not None:
            hit = self.cache.read("ipot_broker", key)
            if hit is not None:
                return hit
        try:
            html = self._get(ticker, day)
        except Exception as exc:
            if self.verbose:
                print(f"  ! ipot {ticker} {day:%Y-%m-%d}: {exc}")
            return empty_frame()
        df = attach_totals(parse_table(html, ticker, day), parse_totals(html))
        # A holiday, a suspension and an unknown ticker all render an empty
        # table. Caching that would poison the store with permanent blanks, so
        # only real rows are written.
        if not df.empty and self.cache is not None:
            self.cache.write("ipot_broker", key, df)
        return df

    def fetch(self, ticker: str, start=None, end=None) -> pd.DataFrame:
        """Walk business days between ``start`` and ``end``, inclusive.

        Weekends are skipped locally; IDX holidays are not knowable ahead of
        time, so they cost one request that returns an empty table.
        """
        if end is None:
            end = pd.Timestamp.today().normalize() - pd.Timedelta(days=1)
        if start is None:
            start = pd.Timestamp(end) - pd.Timedelta(days=90)
        days = pd.bdate_range(pd.Timestamp(start).normalize(),
                              pd.Timestamp(end).normalize())
        if len(days) > self.max_days:
            days = days[-self.max_days:]  # newest wins; bounded by construction
        frames = [self.fetch_day(ticker, d) for d in days]
        frames = [f for f in frames if f is not None and not f.empty]
        if not frames:
            return empty_frame()
        out = pd.concat(frames, ignore_index=True)
        return out.sort_values(["date", "broker"]).reset_index(drop=True)

    def describe(self) -> str:
        return f"ipot (indopremier public module, board={self.board}, top-10)"
