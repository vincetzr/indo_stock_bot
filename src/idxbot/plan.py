"""Trading plan generation.

The plan answers five questions, in the order they matter:

  1. **Where do I buy?**   An entry band anchored on the lead institutional
     desk's reconstructed cost basis. Buying materially above where a large
     holder accumulated means paying for their position; buying inside their
     band means your risk is aligned with theirs.
  2. **Where am I wrong?** A stop below the structural low of the accumulation
     base, floored by an ATR multiple so it is not inside the noise. Plus a
     *thesis* invalidation that has nothing to do with price: the lead broker
     turning net seller.
  3. **Where do I take profit?** Targets learned from that broker's own realised
     markup distribution, not from round numbers.
  4. **How much?**         Risk-based sizing, rounded down to whole lots.
  5. **When do I give up?** A time stop from the broker's median holding period,
     because capital sitting in a stalled campaign has an opportunity cost.

Every price is snapped to the IDX tick grid, and targets are checked against
auto-rejection limits so the plan is executable as written.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .analytics import playbook as playbook_mod
from .config import Config
from .engine import Engine, TickerAnalysis
from .market import (
    Costs,
    auto_rejection_bounds,
    days_to_reach,
    format_idr,
    position_size,
    round_to_tick,
)


@dataclass
class TradingPlan:
    ticker: str
    as_of: pd.Timestamp
    close: float
    score: float
    level: str
    verdict: str                       # TAKE / WATCH / SKIP
    reasons: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    # entry
    entry_low: float = np.nan
    entry_high: float = np.nan
    entry_trigger: str = ""
    anchor_broker: str = ""
    anchor_cost: float = np.nan
    anchor_inventory_lot: float = np.nan

    # risk
    stop: float = np.nan
    stop_basis: str = ""
    invalidation: str = ""
    risk_per_share: float = np.nan
    risk_pct: float = np.nan

    # targets
    targets: List[float] = field(default_factory=list)
    target_pcts: List[float] = field(default_factory=list)
    target_basis: str = ""
    reward_risk: float = np.nan

    # sizing
    lots: int = 0
    shares: float = 0.0
    notional: float = 0.0
    risk_idr: float = 0.0
    equity: float = 0.0

    # timing / context
    time_stop_days: int = 0
    breakeven: float = np.nan
    ara_today: float = np.nan
    arb_today: float = np.nan
    data_mode: str = "price-only"
    data_source: str = "none"
    data_is_real: bool = False
    wyckoff_phase: str = "none"

    def to_dict(self) -> dict:
        out = {k: v for k, v in self.__dict__.items()}
        out["reasons"] = " | ".join(self.reasons)
        out["warnings"] = " | ".join(self.warnings)
        out["targets"] = ",".join(f"{t:.0f}" for t in self.targets)
        out["target_pcts"] = ",".join(f"{p:.3f}" for p in self.target_pcts)
        return out

    def render(self, width: int = 78) -> str:
        """Human-readable plan for the terminal."""
        line = "=" * width
        thin = "-" * width
        out: List[str] = []

        out.append(line)
        out.append(f" {self.ticker}  -  {self.verdict}   (score {self.score:.1f} / {self.level})")
        out.append(f" as of {self.as_of:%Y-%m-%d}   last {self.close:,.0f}   "
                   f"Wyckoff phase {self.wyckoff_phase}")
        out.append(line)

        # "No broker data" and "fabricated broker data" are different claims and
        # must not share a warning. Price-only mode feeds the engine genuine
        # exchange OHLCV and simply omits the flow components; calling that
        # SIMULATED tells the user their real levels are invented.
        if self.data_mode == "price-only":
            out.append(" ii PRICE-ONLY: no broker summary connected, so there is no")
            out.append(" ii institutional confirmation. Prices and levels are real.")
            out.append(thin)
        elif not self.data_is_real:
            out.append(" !! BROKER FLOW IS SIMULATED (source: %s)." % self.data_source)
            out.append(" !! Levels below are a demonstration of the format, not a trade.")
            out.append(thin)

        out.append(" WHY")
        for reason in self.reasons or ["(no supporting evidence)"]:
            out.append(f"   - {reason}")

        out.append("")
        out.append(" ENTRY")
        if np.isfinite(self.entry_low) and np.isfinite(self.entry_high):
            out.append(f"   buy zone      {self.entry_low:,.0f} - {self.entry_high:,.0f}")
        out.append(f"   trigger       {self.entry_trigger}")
        if self.anchor_broker:
            out.append(f"   anchored on   {self.anchor_broker} cost basis "
                       f"{self.anchor_cost:,.0f} "
                       f"({self.anchor_inventory_lot:,.0f} lots accumulated)")

        out.append("")
        out.append(" RISK")
        out.append(f"   stop          {self.stop:,.0f}   ({self.risk_pct:.1%} below entry)  "
                   f"[{self.stop_basis}]")
        out.append(f"   invalidation  {self.invalidation}")
        out.append(f"   time stop     {self.time_stop_days} trading days")

        out.append("")
        out.append(" TARGETS   [%s]" % self.target_basis)
        for i, (target, pct) in enumerate(zip(self.targets, self.target_pcts), 1):
            days = days_to_reach(self.close, target)
            note = f"   (>= {days} limit-up days away)" if days > 3 else ""
            out.append(f"   T{i}            {target:,.0f}   ({pct:+.1%}){note}")
        out.append(f"   reward:risk   {self.reward_risk:.2f} : 1")
        out.append(f"   breakeven     {self.breakeven:,.0f}  (after fees)")

        out.append("")
        out.append(" SIZE")
        out.append(f"   equity        {format_idr(self.equity)}")
        out.append(f"   position      {self.lots:,} lots ({self.shares:,.0f} shares) "
                   f"= {format_idr(self.notional)}")
        out.append(f"   risk          {format_idr(self.risk_idr)} "
                   f"({self.risk_idr / self.equity:.2%} of equity)" if self.equity else "")
        out.append(f"   today's band  ARB {self.arb_today:,.0f} / ARA {self.ara_today:,.0f}")

        if self.warnings:
            out.append("")
            out.append(" WARNINGS")
            for warning in self.warnings:
                out.append(f"   ! {warning}")

        out.append(line)
        return "\n".join(o for o in out if o is not None)


def build_plan(
    analysis: TickerAnalysis,
    engine: Engine,
    equity: Optional[float] = None,
    risk_pct: Optional[float] = None,
    pooled_playbook: Optional[pd.DataFrame] = None,
) -> TradingPlan:
    """Turn an analysis into an executable plan."""
    cfg: Config = engine.cfg
    equity = float(equity if equity is not None else cfg.get("plan.account_equity_idr", 1e8))
    risk_pct = float(risk_pct if risk_pct is not None else cfg.get("plan.risk_per_trade_pct", 0.01))
    max_position_pct = float(cfg.get("plan.max_position_pct", 0.20))
    atr_multiple = float(cfg.get("plan.atr_stop_multiple", 2.0))
    min_rr = float(cfg.get("plan.min_reward_risk", 1.8))
    signal_threshold = float(cfg.get("accumulation.signal_threshold", 65))

    bars = analysis.bars
    signal = analysis.signal
    close = float(bars["close"].iloc[-1])
    atr = float(bars["atr"].iloc[-1]) if "atr" in bars and np.isfinite(bars["atr"].iloc[-1]) else close * 0.02

    plan = TradingPlan(
        ticker=analysis.ticker,
        as_of=analysis.last_date,
        close=close,
        score=signal.score,
        level=signal.level,
        verdict="SKIP",
        data_mode=signal.data_mode,
        data_source=signal.data_source,
        data_is_real=signal.data_is_real,
        wyckoff_phase=signal.wyckoff_state.phase if signal.wyckoff_state else "none",
        equity=equity,
    )
    plan.reasons = list(signal.flags)

    # ---- anchor on the lead institutional desk ----------------------------
    anchor = engine.lead_institutional_broker(analysis)
    anchor_cost = np.nan
    if anchor and not analysis.positions.empty:
        row = analysis.positions[analysis.positions["broker"] == anchor]
        if not row.empty:
            anchor_cost = float(row["avg_cost"].iloc[0])
            plan.anchor_broker = anchor
            plan.anchor_cost = anchor_cost
            plan.anchor_inventory_lot = float(row["inventory_lot"].iloc[0])
            meta = cfg.brokers.get(anchor)
            plan.reasons.append(
                f"{anchor} ({meta.name}) holds {plan.anchor_inventory_lot:,.0f} lots "
                f"at an average cost of {anchor_cost:,.0f}"
            )

    # ---- entry band --------------------------------------------------------
    base_low = float(bars["low"].tail(60).min())
    base_high = float(bars["high"].tail(60).max())

    if np.isfinite(anchor_cost) and anchor_cost > 0:
        # Buy between the desk's cost basis and a half-ATR above it. Above that
        # you are paying a premium to their basis for the same thesis.
        entry_low = min(anchor_cost, close) - 0.25 * atr
        entry_high = max(anchor_cost + 0.5 * atr, close)
    else:
        entry_low = close - 0.5 * atr
        entry_high = close + 0.5 * atr

    plan.entry_low = round_to_tick(max(entry_low, base_low * 0.97), cfg, "down")
    plan.entry_high = round_to_tick(entry_high, cfg, "up")

    phase = plan.wyckoff_phase
    if phase == "C":
        plan.entry_trigger = ("buy the reclaim: enter once price closes back above "
                              f"{round_to_tick(base_low * 1.02, cfg, 'up'):,.0f} after the spring")
    elif phase == "D":
        plan.entry_trigger = (f"buy pullbacks into the zone while price holds above "
                              f"{round_to_tick(base_high * 0.97, cfg, 'down'):,.0f}")
    elif phase == "B":
        plan.entry_trigger = ("scale in inside the zone; add on a close above "
                              f"{round_to_tick(base_high, cfg, 'up'):,.0f} on above-average volume")
    else:
        plan.entry_trigger = (f"no confirmed structure - wait for a close above "
                              f"{round_to_tick(base_high, cfg, 'up'):,.0f} on volume")

    # Risk and sizing are measured from the MIDPOINT of the entry band, which is
    # roughly the average fill of a scale-in. Measuring from the top of the band
    # overstates risk and understates reward:risk on every plan.
    reference_entry = (plan.entry_low + plan.entry_high) / 2.0

    # ---- stop --------------------------------------------------------------
    # Prefer the structural low - that is where the thesis actually breaks. Fall
    # back to an ATR stop when the structure sits so close that normal noise
    # would take the trade out, and floor the whole thing at 15% risk.
    # Take the TIGHTEST candidate that still leaves at least one ATR of room.
    # Taking the widest guarantees an unworkable reward:risk, because the
    # targets are learned from realised markups that are often only 5-15%.
    structural_stop = base_low - 0.5 * atr
    atr_stop = reference_entry - atr_multiple * atr
    min_room = reference_entry - 1.0 * atr

    candidates = {
        f"base low {base_low:,.0f} less 0.5 ATR": structural_stop,
        f"{atr_multiple:g} x ATR ({atr:,.0f})": atr_stop,
    }
    workable = {basis: s for basis, s in candidates.items() if s <= min_room}
    if workable:
        plan.stop_basis, raw_stop = max(workable.items(), key=lambda kv: kv[1])
    else:
        plan.stop_basis, raw_stop = "1 x ATR minimum room", min_room

    floor_stop = reference_entry * 0.85
    if raw_stop < floor_stop:
        raw_stop = floor_stop
        plan.stop_basis = "15% max-risk floor (structure sits further away)"

    plan.stop = round_to_tick(raw_stop, cfg, "down")
    plan.risk_per_share = reference_entry - plan.stop
    plan.risk_pct = plan.risk_per_share / reference_entry if reference_entry > 0 else np.nan

    if anchor:
        plan.invalidation = (
            f"{anchor} turns net seller for 3 consecutive sessions, or inventory falls "
            f"20% from its peak - exit regardless of price"
        )
    else:
        plan.invalidation = "close below the stop, or volume expansion on down days"

    # ---- targets -----------------------------------------------------------
    book = pooled_playbook if pooled_playbook is not None and not pooled_playbook.empty \
        else analysis.playbook
    target_pcts = playbook_mod.playbook_targets(book, anchor or "", cfg)

    # Label the targets by where they actually came from. The anchor having a
    # name is not the same as that broker having a usable campaign record.
    learned = (
        anchor
        and book is not None
        and not book.empty
        and anchor in set(book["broker"])
    )
    plan.target_basis = (
        f"{anchor} realised markup distribution ("
        f"{int(book.loc[book['broker'] == anchor, 'campaigns'].iloc[0])} campaigns)"
        if learned else
        "configured fallback - no campaign history for the anchor broker"
    )
    plan.target_pcts = list(target_pcts)
    plan.targets = [round_to_tick(reference_entry * (1 + p), cfg, "down") for p in target_pcts]

    costs = Costs.from_config(cfg)
    plan.breakeven = costs.breakeven_price(reference_entry, cfg)

    if plan.targets and plan.risk_per_share > 0:
        # Measure R:R on the middle target - the one actually expected to fill.
        mid = plan.targets[len(plan.targets) // 2]
        plan.reward_risk = (mid - reference_entry) / plan.risk_per_share
    else:
        plan.reward_risk = np.nan

    # ---- sizing ------------------------------------------------------------
    lots, notional, risk_idr = position_size(
        equity, reference_entry, plan.stop, risk_pct, max_position_pct, cfg
    )
    plan.lots = lots
    plan.shares = lots * int(cfg.get("market.lot_size", 100))
    plan.notional = notional
    plan.risk_idr = risk_idr

    # ---- timing ------------------------------------------------------------
    holding = np.nan
    if book is not None and not book.empty and anchor:
        row = book[book["broker"] == anchor]
        if not row.empty:
            holding = float(row["median_holding_days"].iloc[0])
    if not np.isfinite(holding):
        holding = 45.0
    # Cap it: a desk's median hold can run past a year, but a retail position
    # sitting that long on a stalled thesis is dead capital, not a trade.
    plan.time_stop_days = int(min(
        holding * float(cfg.get("plan.time_stop_multiple", 1.5)),
        float(cfg.get("plan.max_time_stop_days", 120)),
    ))

    plan.arb_today, plan.ara_today = auto_rejection_bounds(close, cfg)

    # ---- verdict -----------------------------------------------------------
    _decide(plan, signal, cfg, min_rr, signal_threshold, atr, close)
    return plan


def _decide(plan: TradingPlan, signal, cfg: Config, min_rr: float,
            signal_threshold: float, atr: float, close: float) -> None:
    """Set the verdict and collect anything that should stop a trade."""
    if not plan.data_is_real and plan.data_mode == "broker+price":
        plan.warnings.append(
            "broker flow is SIMULATED - this plan demonstrates the format only"
        )

    if plan.data_mode == "price-only":
        plan.warnings.append(
            "no broker summary available: price/volume evidence only, no institutional "
            "confirmation"
        )

    if plan.lots <= 0:
        plan.warnings.append(
            "position rounds to zero lots - the risk budget is too small for this price"
        )

    if np.isfinite(plan.reward_risk) and plan.reward_risk < min_rr:
        plan.warnings.append(
            f"reward:risk {plan.reward_risk:.2f} is below the {min_rr:g} minimum"
        )

    if plan.wyckoff_phase == "E":
        plan.warnings.append("markup already extended - this is a chase, not an accumulation entry")

    if np.isfinite(plan.risk_pct) and plan.risk_pct > 0.14:
        plan.warnings.append(f"stop is {plan.risk_pct:.0%} away - unusually wide risk")

    blocking = [
        plan.lots <= 0,
        np.isfinite(plan.reward_risk) and plan.reward_risk < min_rr,
        plan.wyckoff_phase == "E",
    ]

    if signal.score >= signal_threshold and not any(blocking):
        if plan.data_is_real:
            plan.verdict = "TAKE"
        elif plan.data_mode == "price-only":
            plan.verdict = "TAKE (PRICE-ONLY)"
        else:
            plan.verdict = "TAKE (SIMULATED DATA)"
    elif signal.score >= 50 or (signal.score >= signal_threshold and any(blocking)):
        plan.verdict = "WATCH"
    else:
        plan.verdict = "SKIP"


def build_plans(
    engine: Engine,
    tickers: List[str],
    equity: Optional[float] = None,
    risk_pct: Optional[float] = None,
    pooled_playbook: Optional[pd.DataFrame] = None,
) -> List[TradingPlan]:
    plans = []
    for ticker in tickers:
        analysis = engine.analyze(ticker, with_campaigns=True)
        if analysis is None:
            continue
        plans.append(build_plan(analysis, engine, equity=equity, risk_pct=risk_pct,
                                pooled_playbook=pooled_playbook))
    return plans


def portfolio_heat(plans: List[TradingPlan], equity: float) -> Dict[str, float]:
    """Aggregate exposure if every TAKE plan were filled at once."""
    takes = [p for p in plans if p.verdict.startswith("TAKE")]
    total_risk = sum(p.risk_idr for p in takes)
    total_notional = sum(p.notional for p in takes)
    return {
        "positions": float(len(takes)),
        "total_risk_idr": float(total_risk),
        "total_risk_pct": float(total_risk / equity) if equity else 0.0,
        "total_notional_idr": float(total_notional),
        "gross_exposure_pct": float(total_notional / equity) if equity else 0.0,
    }
