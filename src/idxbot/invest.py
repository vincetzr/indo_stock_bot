"""Long-horizon portfolio construction.

This is the horizon with actual out-of-sample evidence behind it. The same
momentum score the day-trade scanner leans on informally was validated properly
at 60 days on a holdout period never used for selection:

    holdout 2016-12 to 2026-08, cross-sectional
      rank IC            +0.046  (t = 4.92)
      quintile spread    +5.16% per 60 days (t = 5.90), positive on 62% of dates
      long-only top 10   +4.67% per period vs the equal-weight universe (t = 2.32)
      max drawdown       -16.9% vs -32.2% for IHSG

That is the strongest result in this repository, and it is worth contrasting
with the other two horizons:

    horizon   evidence
    day       expectancy straddles zero; ORB filter looks helpful on n=3
    swing     holdout IC +0.031 (t = 3.08) at 20 days - real but thinner
    long      holdout IC +0.046 (t = 4.92) at 60 days - the validated one

The honest ordering is the reverse of most people's instinct: the *slowest*
horizon has the best evidence, and the fastest has none.

Sizing here is deliberately plain - equal weight across the top N, with a
volatility cap so one wild name cannot dominate. Fancier weighting schemes fit
noise at this sample size.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .config import Config
from .engine import Engine
from .market import format_idr, round_to_tick, shares_to_lots, tradeability

# Measured on the untouched holdout; see docs/FINDINGS.md.
HOLDOUT_STATS = {
    "ic_60d": 0.0462,
    "t_60d": 4.92,
    "spread_60d": 0.0516,
    "excess_per_period": 0.0467,
    "excess_t": 2.32,
    "max_drawdown": -0.169,
    "index_max_drawdown": -0.3223,
    "positive_years": "6 of 10",
    "worst_year_spread": -0.0665,
}


@dataclass
class Holding:
    ticker: str
    score: float
    close: float
    lots: int
    shares: float
    notional: float
    weight: float
    atr_pct: float = np.nan
    momentum: float = np.nan
    entry_note: str = ""

    def to_row(self) -> dict:
        return {
            "ticker": self.ticker, "score": round(self.score, 1),
            "close": self.close, "lots": self.lots, "notional": self.notional,
            "weight": round(self.weight, 4), "atr_pct": self.atr_pct,
            "entry_note": self.entry_note,
        }


@dataclass
class InvestmentPlan:
    as_of: pd.Timestamp
    horizon_days: int
    profile: str
    equity: float
    holdings: List[Holding] = field(default_factory=list)
    cash: float = 0.0
    universe_size: int = 0
    rebalance_on: Optional[pd.Timestamp] = None
    warnings: List[str] = field(default_factory=list)

    @property
    def invested(self) -> float:
        return sum(h.notional for h in self.holdings)

    def render(self, width: int = 78) -> str:
        line = "=" * width
        out = [line,
               f" LONG-HORIZON PORTFOLIO   {self.as_of:%Y-%m-%d}   "
               f"profile={self.profile}",
               line]
        out.append(f" hold for      : {self.horizon_days} trading days "
                   f"(~{self.horizon_days / 21:.0f} months)")
        if self.rebalance_on is not None:
            out.append(f" rebalance on  : {self.rebalance_on:%Y-%m-%d} (approx)")
        out.append(f" equity        : {format_idr(self.equity)}")
        out.append(f" from universe : {self.universe_size} names")
        out.append("")

        if not self.holdings:
            out.append(" No holdings selected.")
            return "\n".join(out + [line])

        out.append(f" {'ticker':<8}{'score':>7}{'close':>10}{'lots':>8}"
                   f"{'value':>14}{'weight':>9}{'ATR%':>7}")
        out.append(" " + "-" * (width - 2))
        for h in self.holdings:
            out.append(f" {h.ticker:<8}{h.score:>7.1f}{h.close:>10,.0f}{h.lots:>8,}"
                       f"{format_idr(h.notional):>14}{h.weight:>8.1%}"
                       f"{h.atr_pct * 100 if np.isfinite(h.atr_pct) else 0:>6.1f}%")
        out.append(" " + "-" * (width - 2))
        out.append(f" {'invested':<8}{'':>7}{'':>10}{'':>8}"
                   f"{format_idr(self.invested):>14}"
                   f"{self.invested / self.equity if self.equity else 0:>8.1%}")
        out.append(f" {'cash':<8}{'':>7}{'':>10}{'':>8}{format_idr(self.cash):>14}"
                   f"{self.cash / self.equity if self.equity else 0:>8.1%}")

        out.append("")
        out.append(" HOW TO EXECUTE")
        out.append("   Buy at market over the next 1-3 sessions; do not chase a gap.")
        out.append("   This is a cross-sectional signal - the edge is in holding the")
        out.append("   basket, not in timing any single name. Partial fills are fine.")
        out.append("")
        out.append(" RULES WHILE HOLDING")
        out.append(f"   - Hold the full {self.horizon_days} days. The measured edge is at")
        out.append("     this horizon; exiting early forfeits it.")
        out.append("   - No stop losses on individual names. The drawdown statistic")
        out.append("     below already reflects riding positions through the period,")
        out.append("     and stops would cut winners that dipped.")
        out.append("   - Rebalance on schedule, not on feelings. Re-run this command,")
        out.append("     sell what dropped out, buy what came in.")

        out.append("")
        out.append(" WHAT THE HOLDOUT SAID (2016-12 to 2026-08, never used to fit)")
        s = HOLDOUT_STATS
        out.append(f"   rank IC 60d          {s['ic_60d']:+.4f}  (t={s['t_60d']:.2f})")
        out.append(f"   excess vs universe   {s['excess_per_period']:+.2%} per period "
                   f"(t={s['excess_t']:.2f})")
        out.append(f"   max drawdown         {s['max_drawdown']:.1%}  "
                   f"vs {s['index_max_drawdown']:.1%} for IHSG")
        out.append(f"   winning years        {s['positive_years']}  "
                   f"(worst {s['worst_year_spread']:.1%})")
        out.append("")
        out.append("   Read that last line twice. Four years in ten lost money, and")
        out.append("   roughly 8 points of the headline CAGR is survivorship bias in")
        out.append("   the universe, not skill. See docs/FINDINGS.md section 4.")

        if self.warnings:
            out.append("")
            out.append(" WARNINGS")
            for warning in self.warnings:
                out.append(f"   ! {warning}")
        out.append(line)
        return "\n".join(out)


def build(
    engine: Engine,
    tickers: List[str],
    equity: Optional[float] = None,
    top_n: int = 10,
    horizon_days: int = 60,
    max_weight: float = 0.20,
    min_score: float = 0.0,
    max_atr_pct: float = 0.10,
) -> InvestmentPlan:
    """Build a long-horizon portfolio from the momentum screen."""
    cfg: Config = engine.cfg
    equity = float(equity if equity is not None else cfg.get("plan.account_equity_idr", 1e8))
    profile = engine.profile or str(cfg.get("accumulation.default_profile", "momentum"))

    results = engine.screen(tickers, with_campaigns=False)
    plan = InvestmentPlan(
        as_of=pd.Timestamp.now().normalize(),
        horizon_days=horizon_days,
        profile=profile,
        equity=equity,
        universe_size=len(results),
    )
    if results.empty:
        plan.warnings.append("screen produced no results")
        return plan

    if profile != "momentum":
        plan.warnings.append(
            f"profile '{profile}' is not the validated one; the holdout statistics "
            f"below describe 'momentum' and do not apply"
        )

    ranked = results[results["score"] >= min_score].copy()

    # Drop anything that cannot actually be traded. Suspended names score
    # *highly* on momentum measures - a frozen price sits above its moving
    # average forever - so this filter has to run before selection, not after.
    untradeable = []
    for ticker in list(ranked["ticker"]):
        bars = engine.prices(ticker)
        check = tradeability(
            bars, min_value_traded=float(cfg.get("daytrade.min_value_traded_idr", 5e9)) / 5
        )
        if not check["tradeable"]:
            untradeable.append((ticker, check["reasons"][0]))
    if untradeable:
        ranked = ranked[~ranked["ticker"].isin([t for t, _ in untradeable])]
        for ticker, reason in untradeable[:5]:
            plan.warnings.append(f"excluded {ticker}: {reason}")
        if len(untradeable) > 5:
            plan.warnings.append(f"...and {len(untradeable) - 5} more untradeable names")

    # Exclude names whose daily range makes a 60-day hold a different animal.
    if "atr_pct" in ranked.columns:
        wild = ranked["atr_pct"] > max_atr_pct
        if wild.any():
            dropped = ", ".join(ranked.loc[wild, "ticker"].head(6))
            plan.warnings.append(
                f"excluded {int(wild.sum())} name(s) with daily range above "
                f"{max_atr_pct:.0%}: {dropped}"
            )
            ranked = ranked[~wild]

    selected = ranked.head(top_n)
    if selected.empty:
        plan.warnings.append("nothing passed the filters")
        return plan

    # Equal weight, rounded down to whole lots.
    #
    # The max-weight cap exists to stop one name dominating a diversified book.
    # It must not silently strand cash when the caller deliberately asked for a
    # concentrated one: requesting 3 names IS a request for ~33% each, and
    # capping at 20% would leave 40% uninvested with no explanation.
    equal_weight = 1.0 / max(len(selected), 1)
    if equal_weight > max_weight:
        target_weight = equal_weight
        plan.warnings.append(
            f"{len(selected)} positions means {equal_weight:.0%} in each, above the "
            f"{max_weight:.0%} cap - concentration is deliberate here, so the cap "
            f"was not applied. One bad name moves the whole account."
        )
    else:
        target_weight = max_weight
    for _, row in selected.iterrows():
        close = float(row["close"])
        if close <= 0:
            continue
        budget = equity * target_weight
        lots = shares_to_lots(budget / close, cfg)
        if lots <= 0:
            plan.warnings.append(
                f"{row['ticker']} at {close:,.0f} needs more than "
                f"{format_idr(budget)} for one lot - skipped"
            )
            continue
        shares = lots * int(cfg.get("market.lot_size", 100))
        notional = shares * close
        plan.holdings.append(Holding(
            ticker=str(row["ticker"]),
            score=float(row["score"]),
            close=close,
            lots=lots,
            shares=shares,
            notional=notional,
            weight=notional / equity if equity else 0.0,
            atr_pct=float(row.get("atr_pct", np.nan)),
            entry_note=f"buy up to {round_to_tick(close * 1.02, cfg, 'up'):,.0f}",
        ))

    plan.cash = equity - plan.invested
    plan.rebalance_on = plan.as_of + pd.tseries.offsets.BDay(horizon_days)

    if len(plan.holdings) < top_n:
        plan.warnings.append(
            f"only {len(plan.holdings)} of {top_n} slots filled - concentration is "
            f"higher than intended"
        )
    return plan


def compare_horizons(cfg: Config) -> pd.DataFrame:
    """Side-by-side of what each horizon's evidence actually is."""
    rows = [
        {"horizon": "day", "hold": "hours", "profile": "momentum burst",
         "measured_edge": "straddles zero",
         "evidence": "-0.62%/trade naive entry (n=11); +0.15% with ORB filter (n=3)",
         "verdict": "not established"},
        {"horizon": "swing", "hold": "20 days", "profile": "momentum",
         "measured_edge": "IC +0.031 (t=3.08)",
         "evidence": "holdout 2016-2026, cross-sectional",
         "verdict": "real but thin after costs"},
        {"horizon": "long", "hold": "60 days", "profile": "momentum",
         "measured_edge": "IC +0.046 (t=4.92)",
         "evidence": "holdout; +4.67%/period vs universe (t=2.32)",
         "verdict": "validated - the one to use"},
    ]
    return pd.DataFrame(rows)
