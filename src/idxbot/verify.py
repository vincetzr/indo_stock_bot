"""Acceptance test for broker-summary data.

Run this the moment any broker data arrives — a vendor trial, a pasted table, a
platform export — and before trusting a single number computed from it. Bad
broker data does not announce itself: it produces ledgers, campaigns and
playbooks that look entirely plausible and are wrong.

The four failure modes this exists to catch, in rough order of how often they
bite:

  1. **Aggregate sold as per-broker.** A feed that returns "foreign" and
     "domestic" rather than BK, AK, KZ cannot drive any of the analysis. Vendors
     describe both as "bandarmology". Detected by counting distinct brokers per
     (ticker, day).
  2. **Missing value columns.** Lots without rupiah means no VWAP, so no cost
     basis, so no way to know whether a desk is underwater — roughly half of
     what the ledger computes. Detected by checking value columns are populated
     and imply a sane price.
  3. **Broken buy/sell balance.** On any given day every lot bought was sold, so
     the two sides must total the same. They will not if columns were misread,
     if the table was truncated to a top-N view, or if buy and sell rows were
     merged wrongly. This is the single most powerful check available, because
     it needs no external reference.
  4. **Too little history.** Campaign segmentation needs months; lead-lag needs
     ~90 overlapping days per pair. A real-time-only feed cannot answer the
     question the engine was built for.

Each check returns a verdict and, where it fails, what it means for which
analysis — so the output is actionable rather than a bare pass/fail.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .config import BrokerRegistry
from .market import format_idr

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"


@dataclass
class Check:
    name: str
    status: str
    detail: str
    impact: str = ""

    @property
    def ok(self) -> bool:
        return self.status != FAIL


@dataclass
class Report:
    checks: List[Check] = field(default_factory=list)
    stats: Dict[str, object] = field(default_factory=dict)

    @property
    def usable(self) -> bool:
        return all(c.ok for c in self.checks)

    def add(self, name: str, status: str, detail: str, impact: str = "") -> None:
        self.checks.append(Check(name, status, detail, impact))


def verify(summary: pd.DataFrame, registry: BrokerRegistry,
           prices: Optional[Dict[str, pd.DataFrame]] = None) -> Report:
    """Run every acceptance check against a broker-summary frame."""
    report = Report()

    if summary is None or summary.empty:
        report.add("data present", FAIL, "no rows at all",
                   "nothing can be computed")
        return report

    df = summary.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df["net_lot"] = df["buy_lot"] - df["sell_lot"]

    n_days = df["date"].nunique()
    n_tickers = df["ticker"].nunique()
    n_brokers = df["broker"].nunique()
    span_days = (df["date"].max() - df["date"].min()).days

    report.stats = {
        "rows": len(df),
        "days": n_days,
        "tickers": n_tickers,
        "brokers": n_brokers,
        "first": df["date"].min(),
        "last": df["date"].max(),
        "span_days": span_days,
        "sources": sorted(df["source"].astype(str).unique())[:4],
    }

    _check_granularity(df, report, registry)
    _check_values(df, report, prices)
    _check_balance(df, report)
    _check_history(df, report, n_days, span_days)
    _check_bulge_presence(df, report, registry)
    _check_continuity(df, report)

    return report


def _check_granularity(df: pd.DataFrame, report: Report,
                       registry: BrokerRegistry) -> None:
    """Is this really per-broker, or an aggregate wearing the same name?"""
    per_day = df.groupby(["ticker", "date"])["broker"].nunique()
    median_brokers = float(per_day.median()) if len(per_day) else 0.0

    codes = set(df["broker"].astype(str).str.upper())
    aggregate_markers = {"FOREIGN", "DOMESTIC", "ASING", "LOKAL", "ALL", "TOTAL",
                         "F", "D", "NET"}
    looks_aggregate = bool(codes & aggregate_markers)

    if looks_aggregate:
        report.add(
            "per-broker granularity", FAIL,
            f"found aggregate labels in the broker column: "
            f"{sorted(codes & aggregate_markers)}",
            "this is foreign/domestic flow, not per-member data. The ledger, "
            "campaigns, lead-lag and coordination analysis all need member codes.",
        )
    elif median_brokers < 3:
        report.add(
            "per-broker granularity", FAIL,
            f"median of only {median_brokers:.0f} brokers per ticker-day",
            "too coarse for member-level analysis - likely an aggregate or a "
            "top-1/top-2 view.",
        )
    elif median_brokers < 10:
        report.add(
            "per-broker granularity", WARN,
            f"median {median_brokers:.0f} brokers per ticker-day",
            "looks like a top-N view rather than the full table. Ledgers will "
            "be biased: a desk's quiet days are missing, so inventory drifts.",
        )
    else:
        report.add("per-broker granularity", PASS,
                   f"median {median_brokers:.0f} brokers per ticker-day")


def _check_values(df: pd.DataFrame, report: Report,
                  prices: Optional[Dict[str, pd.DataFrame]]) -> None:
    """Are rupiah value columns present, and do they imply a sane price?"""
    traded = df[(df["buy_lot"] > 0) | (df["sell_lot"] > 0)]
    if traded.empty:
        report.add("value columns", FAIL, "no rows with any volume",
                   "nothing to compute")
        return

    has_value = ((traded["buy_val"] > 0) | (traded["sell_val"] > 0)).mean()
    if has_value < 0.5:
        report.add(
            "value columns", FAIL,
            f"only {has_value:.0%} of traded rows carry a rupiah value",
            "without value there is no VWAP, so no cost basis and no way to "
            "tell whether a desk is underwater. Roughly half the ledger's "
            "output depends on this.",
        )
        return

    # Implied price sanity: value / (lots * 100) should look like a share price.
    buy = traded[(traded["buy_lot"] > 0) & (traded["buy_val"] > 0)]
    if len(buy):
        implied = buy["buy_val"] / (buy["buy_lot"] * 100)
        median_price = float(implied.median())
        if not (30 <= median_price <= 200_000):
            report.add(
                "value columns", FAIL,
                f"implied price of {median_price:,.0f} is outside any plausible "
                f"IDX range",
                "value and volume are probably in mismatched units - check "
                "whether volume is lots or shares.",
            )
            return

        # Cross-check against real prices when available.
        if prices:
            checked, agreed = 0, 0
            for ticker, group in buy.groupby("ticker"):
                bars = prices.get(str(ticker).upper())
                if bars is None or bars.empty:
                    continue
                bar_index = bars.set_index(pd.to_datetime(bars["date"]).dt.normalize())
                for _, row in group.head(40).iterrows():
                    if row["date"] not in bar_index.index:
                        continue
                    bar = bar_index.loc[row["date"]]
                    if isinstance(bar, pd.DataFrame):
                        bar = bar.iloc[0]
                    price = row["buy_val"] / (row["buy_lot"] * 100)
                    checked += 1
                    if float(bar["low"]) * 0.9 <= price <= float(bar["high"]) * 1.1:
                        agreed += 1
            if checked >= 20:
                rate = agreed / checked
                if rate < 0.7:
                    report.add(
                        "value columns", FAIL,
                        f"only {rate:.0%} of implied VWAPs fall inside that day's "
                        f"traded range ({checked} sampled)",
                        "the values do not correspond to the prices actually "
                        "traded - the feed is mislabelled or misaligned by date.",
                    )
                    return
                report.add("value columns", PASS,
                           f"median implied price {median_price:,.0f}; "
                           f"{rate:.0%} of VWAPs sit inside the day's range")
                return

        report.add("value columns", PASS,
                   f"present on {has_value:.0%} of traded rows, "
                   f"median implied price {median_price:,.0f}")


def _check_balance(df: pd.DataFrame, report: Report) -> None:
    """Every lot bought is a lot sold. The strongest check available."""
    per_day = df.groupby(["ticker", "date"])[["buy_lot", "sell_lot"]].sum()
    per_day = per_day[(per_day["buy_lot"] > 0) & (per_day["sell_lot"] > 0)]
    if per_day.empty:
        report.add("buy/sell balance", WARN, "no day has both sides populated",
                   "cannot validate - a one-sided feed hides truncation.")
        return

    mismatch = (per_day["buy_lot"] - per_day["sell_lot"]).abs() / \
        per_day[["buy_lot", "sell_lot"]].max(axis=1)
    median_mismatch = float(mismatch.median())
    balanced = float((mismatch < 0.02).mean())

    if median_mismatch > 0.15:
        report.add(
            "buy/sell balance", FAIL,
            f"median mismatch {median_mismatch:.1%} across "
            f"{len(per_day):,} ticker-days",
            "on any real day both sides total the same. This large a gap means "
            "columns were misread, or the table is a truncated top-N view.",
        )
    elif median_mismatch > 0.02:
        report.add(
            "buy/sell balance", WARN,
            f"median mismatch {median_mismatch:.1%}; "
            f"{balanced:.0%} of days balance within 2%",
            "probably a top-N table rather than the full member list. Usable, "
            "but treat inventory levels as approximate.",
        )
    else:
        report.add("buy/sell balance", PASS,
                   f"{balanced:.0%} of ticker-days balance within 2%")


def _check_history(df: pd.DataFrame, report: Report, n_days: int,
                   span_days: int) -> None:
    """Enough depth for campaigns and lead-lag?"""
    if n_days < 20:
        report.add(
            "history depth", FAIL, f"{n_days} trading days",
            "campaign segmentation needs months. This is enough to sanity-check "
            "the parser, not to draw a conclusion.",
        )
    elif n_days < 90:
        report.add(
            "history depth", WARN, f"{n_days} trading days (~{span_days} calendar)",
            "enough for ledgers and current positioning, not for lead-lag "
            "(needs ~90 overlapping days) or a stable campaign profile.",
        )
    elif n_days < 250:
        report.add(
            "history depth", WARN, f"{n_days} trading days (~{span_days} calendar)",
            "campaigns will be detectable but samples per broker stay small. "
            "Pool across many tickers before believing any profile.",
        )
    else:
        report.add("history depth", PASS,
                   f"{n_days} trading days (~{span_days / 365.25:.1f} years)")


def _check_bulge_presence(df: pd.DataFrame, report: Report,
                          registry: BrokerRegistry) -> None:
    """Are the desks the whole thesis is about actually in the data?"""
    tiers = df["broker"].map(lambda c: registry.get(c).tier)
    bulge = sorted(set(df.loc[tiers == "bulge", "broker"]))
    unknown = sorted(set(df.loc[tiers == "unknown", "broker"]))

    if not bulge:
        report.add(
            "institutional desks", FAIL,
            "no bulge-bracket codes (BK, AK, KZ, MS, ML, CG, RX) appear",
            "the thesis is about these desks. Either they are genuinely absent "
            "from these names, or the codes differ - check config/brokers.yaml "
            "against your feed's coding.",
        )
    else:
        detail = f"{len(bulge)} present: {', '.join(bulge)}"
        if unknown:
            detail += f"  |  {len(unknown)} unrecognised code(s): " \
                      f"{', '.join(unknown[:8])}"
        report.add("institutional desks",
                   WARN if len(unknown) > len(bulge) else PASS, detail,
                   "unrecognised codes still count toward totals but are not "
                   "tiered - add them to config/brokers.yaml." if unknown else "")


def _check_continuity(df: pd.DataFrame, report: Report) -> None:
    """Gaps break the ledger: a missing day is silently read as no trading."""
    dates = pd.Series(sorted(df["date"].unique()))
    if len(dates) < 10:
        return
    gaps = dates.diff().dt.days.dropna()
    # More than 4 calendar days between consecutive observations implies a
    # missing session (weekends alone give 3).
    big_gaps = int((gaps > 4).sum())
    if big_gaps > len(dates) * 0.1:
        report.add(
            "continuity", WARN, f"{big_gaps} gaps longer than a weekend",
            "the ledger treats a missing day as a day with no trading, so "
            "inventory carries flat across holes. Gappy history biases campaign "
            "boundaries.",
        )
    else:
        report.add("continuity", PASS, f"{big_gaps} gaps beyond weekends")


def render(report: Report, width: int = 78) -> str:
    line = "=" * width
    out = [line, " BROKER SUMMARY - ACCEPTANCE TEST", line]

    s = report.stats
    if s:
        out.append(f" rows      : {s['rows']:,}")
        out.append(f" coverage  : {s['tickers']} ticker(s), {s['days']:,} day(s), "
                   f"{s['brokers']} broker code(s)")
        out.append(f" period    : {s['first']:%Y-%m-%d} -> {s['last']:%Y-%m-%d}")
        out.append(f" source    : {', '.join(str(x) for x in s['sources'])}")
    out.append("")

    for check in report.checks:
        marker = {PASS: "  ok  ", WARN: " warn ", FAIL: " FAIL "}[check.status]
        out.append(f" [{marker}] {check.name}")
        out.append(f"           {check.detail}")
        if check.impact:
            for wrapped in _wrap(check.impact, width - 13):
                out.append(f"           {wrapped}")
        out.append("")

    out.append(line)
    if report.usable:
        warns = sum(1 for c in report.checks if c.status == WARN)
        if warns:
            out.append(f" USABLE, with {warns} caveat(s). Read the warnings above -")
            out.append(" they change how much weight the output deserves.")
        else:
            out.append(" USABLE. Run:  idxbot reverse --universe <name>")
    else:
        failed = [c.name for c in report.checks if c.status == FAIL]
        out.append(f" NOT USABLE. Failed: {', '.join(failed)}")
        out.append(" Fix the data before running any analysis on it - the engine")
        out.append(" will happily produce confident, wrong output from bad input.")
    out.append(line)
    return "\n".join(out)


def _wrap(text: str, width: int) -> List[str]:
    words, lines, current = text.split(), [], ""
    for word in words:
        if len(current) + len(word) + 1 > width:
            lines.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        lines.append(current)
    return lines
