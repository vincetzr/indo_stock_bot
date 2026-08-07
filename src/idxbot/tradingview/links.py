"""TradingView helpers: chart links, watchlist export, Pine input generation.

TradingView is the charting surface, not a data source: it has no broker
summary, so nothing here tries to pull data from it. What it does is push
idxbot's conclusions *into* TradingView - as importable watchlists, deep links
and ready-to-paste Pine inputs.
"""

from __future__ import annotations

import os
from typing import Iterable, List, Optional, Sequence
from urllib.parse import quote

import pandas as pd

PINE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pine")


def tv_symbol(ticker: str, prefix: str = "IDX") -> str:
    """``BBCA`` -> ``IDX:BBCA``."""
    ticker = str(ticker).upper().replace(".JK", "").strip()
    if ":" in ticker:
        return ticker
    return f"{prefix}:{ticker}"


def chart_url(ticker: str, prefix: str = "IDX", interval: str = "D") -> str:
    """Deep link that opens the ticker on a TradingView chart."""
    return (
        f"https://www.tradingview.com/chart/?symbol={quote(tv_symbol(ticker, prefix))}"
        f"&interval={interval}"
    )


def symbol_url(ticker: str, prefix: str = "IDX") -> str:
    """Link to the symbol overview page."""
    return f"https://www.tradingview.com/symbols/{prefix}-{str(ticker).upper()}/"


def watchlist_text(tickers: Iterable[str], prefix: str = "IDX") -> str:
    """Body of a TradingView-importable watchlist file.

    Import via the watchlist menu -> "Import list...". One symbol per line.
    """
    return "\n".join(tv_symbol(t, prefix) for t in tickers)


def export_watchlist(tickers: Sequence[str], path: str, prefix: str = "IDX",
                     sections: Optional[dict] = None) -> str:
    """Write a watchlist file, optionally grouped into TradingView sections.

    ``sections`` maps a heading to a list of tickers; TradingView renders
    ``###HEADING`` lines as group separators.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    lines: List[str] = []
    if sections:
        for heading, group in sections.items():
            if not group:
                continue
            lines.append(f"###{heading.upper()}")
            lines.extend(tv_symbol(t, prefix) for t in group)
    else:
        lines.extend(tv_symbol(t, prefix) for t in tickers)

    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    return path


def watchlist_from_screener(results: pd.DataFrame, path: str,
                            prefix: str = "IDX") -> str:
    """Export screener output grouped by signal level."""
    if results is None or results.empty:
        return export_watchlist([], path, prefix)
    sections = {}
    for level in ("STRONG", "SIGNAL", "WATCH"):
        group = results[results["level"] == level]["ticker"].tolist()
        if group:
            sections[f"IDXBOT {level}"] = group
    return export_watchlist([], path, prefix, sections=sections)


def pine_inputs(plan) -> str:
    """Render a plan as a paste-ready Pine input block.

    Pine's ``input.*`` defaults must be literals, so this emits the whole block
    with the plan's numbers baked in - paste it over the inputs section of
    ``broker_campaign.pine``.
    """
    targets = list(plan.targets) + [0.0, 0.0, 0.0]
    start = pd.Timestamp(plan.as_of) - pd.Timedelta(days=int(max(plan.time_stop_days, 30)))

    return "\n".join([
        f"// idxbot plan for {plan.ticker} generated {pd.Timestamp(plan.as_of):%Y-%m-%d}",
        f"// verdict {plan.verdict} | score {plan.score:.1f} | data {plan.data_source}"
        + ("" if plan.data_is_real else "  <-- SIMULATED BROKER FLOW"),
        f'brokerCode  = input.string("{plan.anchor_broker or "NA"}", "Lead broker code", group="Plan")',
        f'brokerCost  = input.float({_num(plan.anchor_cost)}, "Broker average cost", group="Plan", step=1)',
        f'entryLow    = input.float({_num(plan.entry_low)}, "Entry zone low", group="Plan", step=1)',
        f'entryHigh   = input.float({_num(plan.entry_high)}, "Entry zone high", group="Plan", step=1)',
        f'stopPrice   = input.float({_num(plan.stop)}, "Stop", group="Plan", step=1)',
        f'target1     = input.float({_num(targets[0])}, "Target 1", group="Plan", step=1)',
        f'target2     = input.float({_num(targets[1])}, "Target 2", group="Plan", step=1)',
        f'target3     = input.float({_num(targets[2])}, "Target 3", group="Plan", step=1)',
        f'campaignStart = input.time(timestamp("{start:%Y-%m-%d}T00:00:00"), '
        f'"Campaign start", group="Plan")',
    ])


def _num(value) -> str:
    try:
        if value is None or not pd.notna(value):
            return "0.0"
        return f"{float(value):.1f}"
    except (TypeError, ValueError):
        return "0.0"


def read_pine(name: str) -> str:
    """Load a bundled Pine script by filename stem."""
    if not name.endswith(".pine"):
        name = f"{name}.pine"
    path = os.path.join(PINE_DIR, name)
    if not os.path.exists(path):
        available = ", ".join(sorted(
            f[:-5] for f in os.listdir(PINE_DIR) if f.endswith(".pine")
        ))
        raise FileNotFoundError(f"No Pine script {name!r}. Available: {available}")
    with open(path, "r", encoding="utf-8") as fh:
        return fh.read()


def list_pine() -> List[str]:
    if not os.path.isdir(PINE_DIR):
        return []
    return sorted(f[:-5] for f in os.listdir(PINE_DIR) if f.endswith(".pine"))
