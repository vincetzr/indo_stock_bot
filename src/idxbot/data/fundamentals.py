"""Company fundamentals for IDX tickers — and a hard limit on what they can prove.

**Read this before using any number this module returns.**

Everything here is a *current snapshot*. Yahoo serves the trailing PE, price/book,
ROE and margins as they stand today, and at most four annual statements of
history. There is no point-in-time archive: no way to ask "what was BBCA's
reported book value as known to the market in March 2011".

That single gap decides what is and is not possible:

  * **A fundamental screen for today: possible.** Rank the current universe on
    current numbers, and use it to decide what to buy now.
  * **A fundamental backtest over 25 years: impossible.** Not slow, not
    expensive — impossible from any source reachable here. Joining today's
    balance sheet to a 2009 price is the purest form of look-ahead there is:
    it "knows" which companies would still be solvent and profitable in 2026,
    which is exactly the question a 2009 investor was being paid to answer.

So this module deliberately offers no ``score_series`` and no historical
sampling, unlike ``accumulation``. There is nothing to sample. The technical
engine is validated across 25 years because price history is genuine and deep;
the fundamental layer is a present-tense filter, and is labelled as such
everywhere it surfaces.

The second-order caveat: reported financials for Indonesian issuers are
restated, delayed, and occasionally fictional. Treat a fundamental screen as a
way to *exclude* obvious wreckage — negative equity, no earnings, crushing
leverage — rather than as a way to rank quality finely.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import requests

from ..config import Config
from .cache import Cache
from .ohlcv import to_yahoo_symbol

CRUMB_URL = "https://query2.finance.yahoo.com/v1/test/getcrumb"
COOKIE_URL = "https://fc.yahoo.com"
SUMMARY_URL = "https://query2.finance.yahoo.com/v10/finance/quoteSummary/{symbol}"

MODULES = ",".join([
    "defaultKeyStatistics",
    "financialData",
    "summaryDetail",
    "incomeStatementHistory",
    "balanceSheetHistory",
])

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept": "application/json,text/plain,*/*",
}

# Fundamentals move quarterly at best, so the cache is deliberately long-lived.
# It also matters for a duller reason: this endpoint rate-limits aggressively,
# and a cached universe is the difference between a screen that runs and one
# that spends ten minutes collecting 429s.
DEFAULT_MAX_AGE = 7 * 24 * 3600


FX_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{pair}=X"


@dataclass
class Fundamentals:
    ticker: str
    financial_currency: str = ""
    fx_corrected: bool = False
    trailing_pe: float = np.nan
    forward_pe: float = np.nan
    price_to_book: float = np.nan
    return_on_equity: float = np.nan
    profit_margin: float = np.nan
    revenue_growth: float = np.nan
    earnings_growth: float = np.nan
    debt_to_equity: float = np.nan
    current_ratio: float = np.nan
    market_cap: float = np.nan
    annual_periods: int = 0
    revenue_history: List[float] = field(default_factory=list)
    income_history: List[float] = field(default_factory=list)

    def as_row(self) -> Dict[str, object]:
        return {
            "ticker": self.ticker,
            "financial_currency": self.financial_currency,
            "fx_corrected": self.fx_corrected,
            "trailing_pe": self.trailing_pe,
            "price_to_book": self.price_to_book,
            "roe": self.return_on_equity,
            "profit_margin": self.profit_margin,
            "revenue_growth": self.revenue_growth,
            "earnings_growth": self.earnings_growth,
            "debt_to_equity": self.debt_to_equity,
            "current_ratio": self.current_ratio,
            "market_cap": self.market_cap,
            "annual_periods": self.annual_periods,
        }


def _raw(node: object) -> float:
    """Yahoo wraps most numbers as {"raw": x, "fmt": "..."}; some are bare."""
    if isinstance(node, dict):
        node = node.get("raw")
    try:
        value = float(node)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return np.nan
    return value if np.isfinite(value) else np.nan


class YahooFundamentals:
    """Snapshot fundamentals, cached, with the crumb/cookie handshake Yahoo wants."""

    def __init__(self, cfg: Config, cache: Optional[Cache] = None,
                 session: Optional[requests.Session] = None):
        self.cfg = cfg
        self.cache = cache or Cache(cfg.path("data.cache_dir", "data/cache"))
        self.timeout = int(cfg.get("data.request_timeout", 30))
        self.retries = int(cfg.get("data.request_retries", 4))
        self.session = session or requests.Session()
        self.session.headers.update(HEADERS)
        self._crumb: Optional[str] = None
        self._fx: Dict[str, float] = {}

    # -- auth ---------------------------------------------------------------
    def _get_crumb(self, force: bool = False) -> Optional[str]:
        if self._crumb and not force:
            return self._crumb
        try:
            self.session.get(COOKIE_URL, timeout=self.timeout)
        except requests.RequestException:
            pass  # the cookie is best-effort; the crumb call may still work
        for attempt in range(self.retries):
            try:
                r = self.session.get(CRUMB_URL, timeout=self.timeout)
            except requests.RequestException:
                r = None
            if r is not None and r.status_code == 200:
                text = (r.text or "").strip()
                # A 200 carrying "Too Many Requests" as its *body* is Yahoo's
                # habit; treating it as a crumb produces confusing 401s later.
                if text and "Too Many" not in text and len(text) < 32:
                    self._crumb = text
                    return self._crumb
            time.sleep(2 ** attempt)
        return None

    # -- fx -----------------------------------------------------------------
    def fx_rate(self, base: str = "USD", quote: str = "IDR") -> float:
        """Spot rate, used only to repair price/book for non-IDR reporters.

        A *current* rate against a *current* price and a *current* book value is
        internally consistent, which is all this needs to be. It would be wrong
        for anything historical - one more reason this module refuses to
        produce a time series.
        """
        if base == quote:
            return 1.0
        pair = f"{base}{quote}"
        if pair in self._fx:
            return self._fx[pair]
        try:
            r = self.session.get(FX_URL.format(pair=pair),
                                 params={"range": "5d", "interval": "1d"},
                                 timeout=self.timeout)
            meta = r.json()["chart"]["result"][0]["meta"]
            rate = float(meta["regularMarketPrice"])
        except Exception:
            return np.nan
        self._fx[pair] = rate
        return rate

    # -- fetch --------------------------------------------------------------
    def fetch(self, ticker: str, max_age: Optional[float] = DEFAULT_MAX_AGE,
              force_refresh: bool = False) -> Optional[Fundamentals]:
        key = ticker.upper()
        if not force_refresh:
            cached = self.cache.read("fundamentals", key, max_age=max_age,
                                     parse_dates=[])
            if cached is not None and not cached.empty:
                return _from_frame(key, cached)

        crumb = self._get_crumb()
        if not crumb:
            return None

        symbol = to_yahoo_symbol(ticker)
        payload = None
        for attempt in range(self.retries):
            try:
                r = self.session.get(SUMMARY_URL.format(symbol=symbol),
                                     params={"modules": MODULES, "crumb": crumb},
                                     timeout=self.timeout)
            except requests.RequestException:
                time.sleep(2 ** attempt)
                continue
            if r.status_code == 401:
                crumb = self._get_crumb(force=True) or crumb
                continue
            if r.status_code == 429:
                time.sleep(2 ** attempt * 5)  # throttled: back off harder
                continue
            if r.status_code != 200:
                return None
            try:
                payload = r.json()
            except ValueError:
                return None
            break

        if not payload:
            return None
        results = (payload.get("quoteSummary") or {}).get("result") or []
        if not results:
            return None

        node = results[0]
        currency = str(((node.get("financialData") or {}).get("financialCurrency")) or "").upper()
        rate = self.fx_rate(currency, "IDR") if currency and currency != "IDR" else np.nan
        fundamentals = _parse(key, node, fx_rate=rate)
        self.cache.write("fundamentals", key, pd.DataFrame([fundamentals.as_row()]))
        return fundamentals

    def fetch_many(self, tickers: List[str], verbose: bool = False,
                   pause: float = 0.4) -> Dict[str, Fundamentals]:
        out: Dict[str, Fundamentals] = {}
        for n, ticker in enumerate(tickers, 1):
            data = self.fetch(ticker)
            if data is not None:
                out[ticker.upper()] = data
            if verbose and n % 25 == 0:
                print(f"  fundamentals {n}/{len(tickers)}  ({len(out)} retrieved)")
            time.sleep(pause)  # deliberate: this endpoint throttles quickly
        return out


def _parse(ticker: str, node: Dict, fx_rate: float = np.nan) -> Fundamentals:
    fd = node.get("financialData") or {}
    ks = node.get("defaultKeyStatistics") or {}
    sd = node.get("summaryDetail") or {}
    inc = (node.get("incomeStatementHistory") or {}).get("incomeStatementHistory") or []
    bal = (node.get("balanceSheetHistory") or {}).get("balanceSheetStatements") or []

    # A trap that silently wrecks the entire mining and energy cohort: many IDX
    # issuers report in USD while trading in IDX, and Yahoo divides the IDR price
    # by the USD book value per share. ADRO comes back at P/B 14,941 and INCO at
    # 20,074 - both actually under 1.3x. Left uncorrected, every USD reporter
    # looks catastrophically overvalued and gets excluded, which would quietly
    # delete the sector that produces most of IDX's big momentum winners.
    #
    # Only price/book is affected. `trailingEps` is already served in IDR (ADRO
    # 306.83, giving the correct PE of 8.3), and ROE, margins, growth and
    # debt/equity are ratios, so they are currency-neutral.
    currency = str(fd.get("financialCurrency") or "").upper()
    price_to_book = _raw(ks.get("priceToBook"))
    corrected = False
    if currency and currency != "IDR" and np.isfinite(price_to_book) and np.isfinite(fx_rate):
        price_to_book = price_to_book / fx_rate
        corrected = True

    return Fundamentals(
        ticker=ticker,
        financial_currency=currency,
        fx_corrected=corrected,
        trailing_pe=_raw(sd.get("trailingPE")),
        forward_pe=_raw(ks.get("forwardPE")) if "forwardPE" in ks else _raw(sd.get("forwardPE")),
        price_to_book=price_to_book,
        return_on_equity=_raw(fd.get("returnOnEquity")),
        profit_margin=_raw(fd.get("profitMargins")),
        revenue_growth=_raw(fd.get("revenueGrowth")),
        earnings_growth=_raw(fd.get("earningsGrowth")),
        debt_to_equity=_raw(fd.get("debtToEquity")),
        current_ratio=_raw(fd.get("currentRatio")),
        market_cap=_raw(sd.get("marketCap")),
        annual_periods=max(len(inc), len(bal)),
        revenue_history=[_raw(x.get("totalRevenue")) for x in inc],
        income_history=[_raw(x.get("netIncome")) for x in inc],
    )


def _from_frame(ticker: str, df: pd.DataFrame) -> Fundamentals:
    row = df.iloc[0]

    def val(name: str) -> float:
        return float(row[name]) if name in row and pd.notna(row[name]) else np.nan

    return Fundamentals(
        ticker=ticker,
        financial_currency=str(row["financial_currency"])
        if "financial_currency" in row and pd.notna(row["financial_currency"]) else "",
        fx_corrected=bool(row["fx_corrected"]) if "fx_corrected" in row else False,
        trailing_pe=val("trailing_pe"),
        price_to_book=val("price_to_book"),
        return_on_equity=val("roe"),
        profit_margin=val("profit_margin"),
        revenue_growth=val("revenue_growth"),
        earnings_growth=val("earnings_growth"),
        debt_to_equity=val("debt_to_equity"),
        current_ratio=val("current_ratio"),
        market_cap=val("market_cap"),
        annual_periods=int(val("annual_periods")) if pd.notna(row.get("annual_periods")) else 0,
    )


# ---------------------------------------------------------------------------
# Screening
# ---------------------------------------------------------------------------

def quality_flags(f: Fundamentals, cfg: Optional[Config] = None) -> List[str]:
    """Reasons to exclude a name outright.

    Framed as exclusions rather than a ranking on purpose. Indonesian reported
    financials are restated and delayed often enough that fine-grained quality
    scoring reads more precision into them than they carry; what they *are*
    reliable for is spotting a company that is losing money or drowning in debt.
    """
    reasons: List[str] = []
    if np.isfinite(f.return_on_equity) and f.return_on_equity < 0:
        reasons.append(f"negative ROE ({f.return_on_equity:.1%})")
    if np.isfinite(f.profit_margin) and f.profit_margin < 0:
        reasons.append(f"loss-making (margin {f.profit_margin:.1%})")
    if np.isfinite(f.debt_to_equity) and f.debt_to_equity > 200:
        reasons.append(f"debt/equity {f.debt_to_equity:.0f}%")
    if np.isfinite(f.trailing_pe) and f.trailing_pe > 100:
        reasons.append(f"PE {f.trailing_pe:.0f}")
    # A current ratio below 1 is normal — not alarming — for banks, telcos and
    # toll-road operators, which fund long-dated assets with rolling short-term
    # debt. Testing it alone flagged TLKM, MTEL and UNVR as distressed, which is
    # simply wrong, and would have deleted the defensive half of LQ45. Tight
    # liquidity only signals danger when the balance sheet is also levered, so
    # the two conditions are required together. Yahoo exposes no reliable sector
    # field for IDX, so this stands in for the sector carve-out.
    if (np.isfinite(f.current_ratio) and f.current_ratio < 0.8
            and np.isfinite(f.debt_to_equity) and f.debt_to_equity > 100):
        reasons.append(f"current ratio {f.current_ratio:.2f} with debt/equity "
                       f"{f.debt_to_equity:.0f}%")
    return reasons


def screen(universe: Dict[str, Fundamentals], cfg: Optional[Config] = None) -> pd.DataFrame:
    """One row per ticker with the exclusion reasons attached."""
    rows = []
    for ticker, f in universe.items():
        reasons = quality_flags(f, cfg)
        row = f.as_row()
        row["excluded"] = bool(reasons)
        row["reasons"] = "; ".join(reasons)
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("ticker").reset_index(drop=True)


def render(df: pd.DataFrame, width: int = 78) -> str:
    line = "=" * width
    out = [line, " FUNDAMENTAL SCREEN (current snapshot - NOT backtestable)", line]
    if df.empty:
        out.append(" No fundamentals retrieved.")
        return "\n".join(out + [line])

    kept = df[~df["excluded"]]
    out.append(f" retrieved : {len(df)} tickers")
    out.append(f" excluded  : {int(df['excluded'].sum())}   passing: {len(kept)}")
    out.append("")
    out.append(f" {'ticker':<8}{'PE':>8}{'P/B':>8}{'ROE':>7}{'margin':>8}"
               f"{'rev g':>8}{'D/E':>7} ccy  status")

    def fmt(v, pct=False, dp=1, width=7):
        """Never let an outlier run two columns together - clamp and mark it."""
        if v is None or not np.isfinite(v):
            return " " * (width - 1) + "-"
        text = f"{v:.{dp}%}" if pct else f"{v:,.{dp}f}"
        if len(text) > width:
            text = ">999" if v > 0 else "<-999"
        return text.rjust(width)

    for _, r in df.iterrows():
        status = r["reasons"] if r["excluded"] else "ok"
        ccy = str(r.get("financial_currency") or "")[:3]
        if r.get("fx_corrected"):
            ccy = ccy.lower()  # lower-case marks a P/B repaired via spot FX
        out.append(f" {r['ticker']:<8}{fmt(r['trailing_pe'], width=8)}"
                   f"{fmt(r['price_to_book'], dp=2, width=8)}"
                   f"{fmt(r['roe'], pct=True, dp=0, width=7)}"
                   f"{fmt(r['profit_margin'], pct=True, dp=0, width=8)}"
                   f"{fmt(r['revenue_growth'], pct=True, dp=0, width=8)}"
                   f"{fmt(r['debt_to_equity'], dp=0, width=7)} {ccy:<3}  {status}")

    corrected = int(df["fx_corrected"].sum()) if "fx_corrected" in df else 0
    if corrected:
        out.append("")
        out.append(f" {corrected} name(s) report in a non-IDR currency; their price/book is")
        out.append(" served against an IDR price and is nonsense as published (ADRO comes")
        out.append(" back at 14,941x, INCO at 20,074x). Repaired here with the spot rate")
        out.append(" and shown with a lower-case currency tag. PE needs no repair - Yahoo")
        out.append(" already serves trailing EPS in IDR.")

    out.append("")
    out.append(" These are today's numbers. They cannot be backtested: no")
    out.append(" point-in-time fundamental archive exists for IDX from any source")
    out.append(" reachable here, and joining today's balance sheet to a past price")
    out.append(" would 'know' which companies survived. Use this to exclude")
    out.append(" wreckage from a technically-ranked list, not to rank quality.")
    return "\n".join(out + [line])
