"""Intraday momentum day-trading engine.

The target is a 5-10% move captured and closed within one session. What the data
says about that goal, measured on 296,344 real IDX stock-days (2001-2026):

  * On a random stock-day, the chance price touches +5% from the open is
    **6.9%**; +10% is **1.4%**. These are not everyday events.
  * The burst setup this module hunts - a huge relative-volume day that closes
    strongly at a 20-day high - lifts that to **38.7% / 18.5%**, roughly 5.6x
    the base rate. It is a real, large conditioning effect.
  * But it fired only **416 times in 25 years across 66 names** (~17/year). The
    lever for more trades is a **wider universe**, not a looser filter: every
    loosening tested degraded expectancy.
  * Net expectancy on daily bars is genuinely undetermined. With a +5% target
    and -3% stop, 13.5% of qualifying days touch *both* levels, and the true
    result sits between **-0.56%** (stop always first) and **+0.52%** (target
    always first) per trade. Only intraday data resolves it - see
    ``data/intraday.resolve_path``.

So this engine is built to be run with intraday data and with the path
resolution measured, not assumed. It reports the ambiguity rather than picking
the flattering side, and every plan carries the honest expectancy band.

The broker-flow trigger is the other half. A burst accompanied by a bulge desk
(AK, BK, ...) dominating the buy side of the running trade is the setup the user
of this tool is actually hunting - DSSA on 2026-08-06 went from 870 to 975
(+12.1%) on 6.5x normal volume. Whether broker confirmation improves the burst's
expectancy is **untested**, because it needs live broker data. It is wired in and
ready to measure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .config import Config
from .data.intraday import opening_range, resolve_path, session_frame, volume_pace
from .market import (
    Costs,
    ara_pct,
    auto_rejection_bounds,
    format_idr,
    position_size,
    round_to_tick,
    tradeability,
)

# Measured on 2001-2026 IDX data; see the module docstring and docs/DAYTRADE.md.
SETUP_STATS = {
    "burst": {"p_touch_5": 0.387, "p_touch_10": 0.185, "n": 416,
              "exp_low": -0.0056, "exp_high": 0.0052, "ambiguous": 0.135},
    "surge": {"p_touch_5": 0.363, "p_touch_10": 0.157, "n": 1792,
              "exp_low": -0.0079, "exp_high": -0.0002, "ambiguous": 0.109},
    "base": {"p_touch_5": 0.069, "p_touch_10": 0.014, "n": 296344,
             "exp_low": -0.0056, "exp_high": -0.0050, "ambiguous": 0.008},
}


@dataclass
class DayCandidate:
    """A name flagged after the close as a candidate for the next session."""

    ticker: str
    date: pd.Timestamp
    close: float
    setup: str
    score: float

    rvol: float = np.nan            # volume vs 20-day median
    day_return: float = np.nan      # today's close-to-close
    near_high: float = np.nan       # distance to the 20-day high
    atr_pct: float = np.nan
    value_traded: float = np.nan    # rupiah turnover, the liquidity gate
    close_position: float = np.nan  # where the close sat inside today's range

    reasons: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def stats(self) -> Dict[str, float]:
        return SETUP_STATS.get(self.setup, SETUP_STATS["base"])

    def to_row(self) -> dict:
        return {
            "ticker": self.ticker, "date": self.date, "close": self.close,
            "setup": self.setup, "score": round(self.score, 1),
            "rvol": round(self.rvol, 1), "day_return": self.day_return,
            "near_high": self.near_high, "atr_pct": self.atr_pct,
            "value_traded": self.value_traded,
            "close_position": round(self.close_position, 2),
            "p_touch_5": self.stats["p_touch_5"],
            "reasons": " | ".join(self.reasons),
            "warnings": " | ".join(self.warnings),
        }


@dataclass
class DayPlan:
    """An intraday plan: enter after the open, flat by the close."""

    ticker: str
    date: pd.Timestamp
    setup: str
    reference_price: float

    entry_trigger: float = np.nan
    entry_note: str = ""
    stop: float = np.nan
    targets: List[float] = field(default_factory=list)
    target_pcts: List[float] = field(default_factory=list)

    target_shares: List[float] = field(default_factory=list)
    breakeven: float = np.nan
    gap_skip_pct: float = 0.03

    lots: int = 0
    notional: float = 0.0
    risk_idr: float = 0.0
    equity: float = 0.0

    ara_price: float = np.nan
    arb_price: float = np.nan
    headroom_to_ara: float = np.nan

    p_touch_target: float = np.nan
    expectancy_low: float = np.nan
    expectancy_high: float = np.nan
    reasons: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def render(self, width: int = 74) -> str:
        line = "=" * width
        out = [line,
               f" {self.ticker}  DAY TRADE  [{self.setup}]   {self.date:%Y-%m-%d}",
               line]

        out.append(" WHY")
        for reason in self.reasons or ["(none)"]:
            out.append(f"   - {reason}")

        out.append("")
        out.append(" EXECUTION PLAN  (times are WIB)")
        out.append(" " + "-" * (width - 2))
        out.append(" 09:00-09:30   DO NOTHING. Let the opening range form.")
        out.append(f"               Mark the high and low of the first 30 minutes.")
        out.append("               Skip the day entirely if it gaps up more than")
        out.append(f"               {self.gap_skip_pct:.0%} - the stop would sit under the open.")
        out.append("")
        out.append(" 09:30-11:30   ENTRY WINDOW. Buy only when ALL of these hold:")
        out.append(f"                 1. a 5-min bar CLOSES above the opening-range high")
        out.append(f"                    (indicative: {self.entry_trigger:,.0f})")
        out.append("                 2. price is above the session VWAP")
        out.append("                 3. volume pace >= 1.5x normal for the time of day")
        out.append("               No trigger by 11:30 -> no trade. Walk away.")
        out.append("               In testing this filter skipped 9 of 12 signals,")
        out.append("               and those skips are where the losses were.")
        out.append("")
        out.append(" AFTER ENTRY   Stop is the HIGHER of:")
        out.append(f"                 - the opening-range low")
        out.append(f"                 - {self.stop:,.0f}  (-{abs(self.stop / self.entry_trigger - 1):.1%} from entry)")
        out.append("               Set it as a real order. Do not widen it.")
        out.append("")
        for i, (t, p) in enumerate(zip(self.targets, self.target_pcts), 1):
            share = self.target_shares[i - 1] if i <= len(self.target_shares) else 0.0
            out.append(f" TAKE PROFIT   T{i}  {t:,.0f}  ({p:+.1%})  -> sell {share:.0%} "
                       f"of the position")
        if self.targets:
            out.append(f"               After T1 fills, move the stop to breakeven")
            out.append(f"               ({self.breakeven:,.0f}, fee-adjusted). The trade")
            out.append("               can no longer lose money from there.")
        out.append("")
        out.append(" 15:00         If T1 has not filled and price is below VWAP, exit.")
        out.append("               The burst has failed; holding is hope, not a plan.")
        out.append(" 15:45         CLOSE EVERYTHING. This is a day trade. Overnight")
        out.append("               gap risk is not priced into any number here.")

        out.append("")
        out.append(" SIZE")
        out.append(f"   position     {self.lots:,} lots = {format_idr(self.notional)}")
        out.append(f"   risk         {format_idr(self.risk_idr)}")
        out.append(f"   ARA today    {self.ara_price:,.0f}  "
                   f"(headroom {self.headroom_to_ara:.1%})")
        out.append(f"   ARB today    {self.arb_price:,.0f}")

        out.append("")
        out.append(" ODDS  (be honest with yourself about these)")
        out.append(f"   P(touch +5% next day, this setup) {self.p_touch_target:.0%}"
                   f"   vs 6.9% for a random day")
        out.append(f"   naive 'buy at the open' entry     {self.expectancy_low:+.2%} to "
                   f"{self.expectancy_high:+.2%}/trade")
        out.append("   measured on real 5-min bars       -0.62%/trade, 36% win rate")
        out.append("   same signals, ORB+pace filter     +0.15%/trade, but only 3")
        out.append("                                     trades survived the filter")
        out.append("   The filter is the edge, if there is one. n=3 is not evidence -")
        out.append("   it is consistent with the design and nothing more. Size small.")

        if self.warnings:
            out.append("")
            out.append(" WARNINGS")
            for warning in self.warnings:
                out.append(f"   ! {warning}")
        out.append(line)
        return "\n".join(out)


# ---------------------------------------------------------------------------
# End-of-day scan
# ---------------------------------------------------------------------------
def scan(
    bars_by_ticker: Dict[str, pd.DataFrame],
    cfg: Config,
    as_of: Optional[pd.Timestamp] = None,
    min_value_traded: float = 5e9,
) -> List[DayCandidate]:
    """Rank names after the close as candidates for the next session.

    ``min_value_traded`` (default Rp5bn/day) is a liquidity gate, and it is not
    optional: a burst in a name that trades Rp200m a day cannot be entered or
    exited in size, and the historical statistics for such names are dominated
    by prints you could never have participated in.
    """
    rvol_burst = float(cfg.get("daytrade.rvol_burst", 8.0))
    rvol_surge = float(cfg.get("daytrade.rvol_surge", 5.0))
    ret_burst = float(cfg.get("daytrade.return_burst", 0.07))
    ret_surge = float(cfg.get("daytrade.return_surge", 0.05))
    near_high_tol = float(cfg.get("daytrade.near_high_tolerance", 0.005))

    candidates: List[DayCandidate] = []
    for ticker, bars in bars_by_ticker.items():
        if bars is None or len(bars) < 60:
            continue
        df = bars if as_of is None else bars[bars["date"] <= pd.Timestamp(as_of)]
        if len(df) < 60:
            continue

        row = df.iloc[-1]
        vol_median = float(df["volume"].tail(20).median())
        if vol_median <= 0:
            continue

        rvol = float(row["volume"]) / vol_median
        day_return = float(row["close"] / df["close"].iloc[-2] - 1.0)
        high_20 = float(df["high"].tail(20).max())
        near_high = float(row["close"] / high_20 - 1.0) if high_20 > 0 else np.nan
        bar_range = float(row["high"] - row["low"])
        close_position = float((row["close"] - row["low"]) / bar_range) if bar_range > 0 else 0.5
        atr_pct = float(((df["high"] - df["low"]) / df["close"]).tail(14).mean())
        value_traded = float(row["close"] * row["volume"])

        # A suspended or barely-traded name can post a huge rvol on the one day
        # it reopens; that is not a burst worth trading.
        check = tradeability(df, min_value_traded=min_value_traded / 5)
        if not check["tradeable"]:
            continue

        # Setup classification, strongest first.
        if rvol >= rvol_burst and day_return >= ret_burst and near_high >= -near_high_tol:
            setup = "burst"
        elif rvol >= rvol_surge and day_return >= ret_surge:
            setup = "surge"
        else:
            continue

        reasons = [
            f"volume {rvol:.1f}x its 20-day median",
            f"closed {day_return:+.1%} on the day",
        ]
        if near_high >= -near_high_tol:
            reasons.append("closed at or above the 20-day high")
        if close_position >= 0.75:
            reasons.append(f"closed in the top {(1 - close_position):.0%} of the day's range")

        warnings: List[str] = []
        if value_traded < min_value_traded:
            warnings.append(
                f"only {format_idr(value_traded)} traded - too thin to enter in size"
            )
        if atr_pct > 0.09:
            warnings.append(f"average range {atr_pct:.0%}/day - stops will be hit by noise")
        if day_return >= ara_pct(float(row["close"]), cfg) * 0.95:
            warnings.append("closed at or near the ARA limit - the next open often gaps")

        # Score: relative volume dominates, then strength of the close.
        score = float(np.clip(
            35 * min(rvol / 10.0, 1.5)
            + 30 * min(day_return / 0.10, 1.5)
            + 20 * close_position
            + 15 * (1.0 if near_high >= -near_high_tol else 0.0),
            0, 100,
        ))

        candidates.append(DayCandidate(
            ticker=ticker, date=pd.Timestamp(row["date"]), close=float(row["close"]),
            setup=setup, score=score, rvol=rvol, day_return=day_return,
            near_high=near_high, atr_pct=atr_pct, value_traded=value_traded,
            close_position=close_position, reasons=reasons, warnings=warnings,
        ))

    candidates.sort(key=lambda c: (-c.score, c.ticker))
    return candidates


# ---------------------------------------------------------------------------
# Live intraday state
# ---------------------------------------------------------------------------
@dataclass
class IntradayState:
    ticker: str
    ts: pd.Timestamp
    last: float
    open_price: float
    vwap: float
    or_high: float
    or_low: float
    session_pct: float
    pace: float
    from_open: float
    above_vwap: bool
    broke_or: bool
    ara_price: float = np.nan
    headroom: float = np.nan
    broker_note: str = ""

    def summary(self) -> str:
        return (
            f"{self.ticker}  {self.last:,.0f}  ({self.from_open:+.1%} from open)  "
            f"VWAP {self.vwap:,.0f} {'ABOVE' if self.above_vwap else 'below'}  "
            f"OR-high {self.or_high:,.0f} {'BROKEN' if self.broke_or else 'intact'}  "
            f"pace {self.pace:.1f}x  session {self.session_pct:.0%}"
        )


def intraday_state(
    ticker: str,
    intraday: pd.DataFrame,
    reference_daily_volume: float,
    cfg: Config,
    prev_close: Optional[float] = None,
    opening_minutes: int = 30,
) -> Optional[IntradayState]:
    """Current live state of a name from its intraday bars."""
    session = session_frame(intraday)
    if session is None or session.empty:
        return None

    orange = opening_range(session, opening_minutes)
    if not orange:
        return None

    last = float(session["close"].iloc[-1])
    open_price = float(orange["open"])
    vwap = float(session["vwap"].iloc[-1])
    pace = volume_pace(session, reference_daily_volume)

    ara_price = np.nan
    headroom = np.nan
    if prev_close:
        _arb, ara_price = auto_rejection_bounds(prev_close, cfg)
        headroom = (ara_price / last - 1.0) if last > 0 else np.nan

    return IntradayState(
        ticker=ticker,
        ts=pd.Timestamp(session["ts"].iloc[-1]),
        last=last,
        open_price=open_price,
        vwap=vwap,
        or_high=float(orange["or_high"]),
        or_low=float(orange["or_low"]),
        session_pct=float(session["session_pct"].iloc[-1]),
        pace=float(pace) if np.isfinite(pace) else np.nan,
        from_open=(last / open_price - 1.0) if open_price > 0 else np.nan,
        above_vwap=last > vwap,
        broke_or=last > float(orange["or_high"]),
        ara_price=ara_price,
        headroom=headroom,
    )


def broker_trigger(summary: pd.DataFrame, cfg: Config,
                   ticker: str, min_share: float = 0.25) -> Dict[str, object]:
    """Is a bulge desk dominating the buy side right now?

    Consumes a broker-summary snapshot - live, this comes from
    ``RunningTradeAggregator.snapshot()``. This is the confirmation the user of
    this tool is hunting: a volume burst *with* AK or BK on the bid, as happened
    in DSSA on 2026-08-06.

    Whether it improves the burst setup's expectancy is UNMEASURED - it needs
    live broker data to test. Treat a hit as a reason to prefer one candidate
    over another, not as an independent edge.
    """
    if summary is None or summary.empty:
        return {"triggered": False, "note": "no broker data"}

    day = summary[summary["ticker"] == str(ticker).upper()]
    if day.empty:
        return {"triggered": False, "note": "no broker data for this ticker"}

    day = day.copy()
    day["net_lot"] = day["buy_lot"] - day["sell_lot"]
    total_buy = float(day["buy_lot"].sum())
    if total_buy <= 0:
        return {"triggered": False, "note": "no buying yet"}

    registry = cfg.brokers
    day["tier"] = day["broker"].map(lambda c: registry.get(c).tier)
    bulge = day[day["tier"] == "bulge"]
    bulge_share = float(bulge["buy_lot"].sum() / total_buy)

    net_buyers = day[day["net_lot"] > 0].sort_values("net_lot", ascending=False)
    leaders = net_buyers.head(3)["broker"].tolist()
    bulge_leaders = [b for b in leaders if registry.get(b).tier == "bulge"]

    triggered = bulge_share >= min_share and bool(bulge_leaders)
    note = (
        f"bulge desks are {bulge_share:.0%} of buy volume; top net buyers "
        f"{', '.join(leaders) if leaders else 'none'}"
    )
    return {
        "triggered": triggered,
        "bulge_share": bulge_share,
        "leaders": leaders,
        "bulge_leaders": bulge_leaders,
        "note": note,
    }


# ---------------------------------------------------------------------------
# Plan
# ---------------------------------------------------------------------------
def build_day_plan(
    candidate: DayCandidate,
    cfg: Config,
    state: Optional[IntradayState] = None,
    equity: Optional[float] = None,
    risk_pct: Optional[float] = None,
) -> DayPlan:
    """Turn a candidate (plus optional live state) into an intraday plan."""
    equity = float(equity if equity is not None else cfg.get("plan.account_equity_idr", 1e8))
    risk_pct = float(risk_pct if risk_pct is not None
                     else cfg.get("daytrade.risk_per_trade_pct", 0.005))
    target_pcts = list(cfg.get("daytrade.targets", [0.05, 0.08]))
    stop_pct = float(cfg.get("daytrade.stop_pct", 0.03))
    max_position_pct = float(cfg.get("plan.max_position_pct", 0.20))

    stats = candidate.stats
    reference = candidate.close

    plan = DayPlan(
        ticker=candidate.ticker,
        date=candidate.date,
        setup=candidate.setup,
        reference_price=reference,
        reasons=list(candidate.reasons),
        warnings=list(candidate.warnings),
        equity=equity,
        p_touch_target=stats["p_touch_5"],
        expectancy_low=stats["exp_low"],
        expectancy_high=stats["exp_high"],
    )

    if state is not None:
        # Live: trigger on a break of the opening range, held above VWAP.
        trigger = round_to_tick(max(state.or_high, state.vwap) * 1.001, cfg, "up")
        plan.entry_note = (
            f"buy only on a 5-minute close above {trigger:,.0f} while price holds "
            f"above VWAP ({state.vwap:,.0f}); skip if volume pace falls below 1.5x"
        )
        reference = state.last
        if np.isfinite(state.headroom):
            plan.headroom_to_ara = state.headroom
            if state.headroom < max(target_pcts):
                plan.warnings.append(
                    f"only {state.headroom:.1%} of headroom to ARA - the top target "
                    f"cannot print today"
                )
        if state.pace and np.isfinite(state.pace) and state.pace < 1.5:
            plan.warnings.append(f"volume pace only {state.pace:.1f}x - burst is fading")
    else:
        # Pre-open: trigger above yesterday's high.
        trigger = round_to_tick(candidate.close * 1.005, cfg, "up")
        plan.entry_note = (
            "buy only on a break above the first 30 minutes' high with volume pace "
            "above 1.5x; if it opens with a gap larger than the stop, stand aside"
        )

    plan.entry_trigger = trigger
    plan.stop = round_to_tick(trigger * (1 - stop_pct), cfg, "down")
    plan.targets = [round_to_tick(trigger * (1 + p), cfg, "down") for p in target_pcts]
    plan.target_pcts = target_pcts
    plan.target_shares = list(cfg.get("daytrade.target_shares", [0.5, 0.5]))[:len(target_pcts)]
    plan.gap_skip_pct = stop_pct

    costs = Costs.from_config(cfg)
    plan.breakeven = costs.breakeven_price(trigger, cfg)

    arb, ara = auto_rejection_bounds(candidate.close, cfg)
    plan.arb_price, plan.ara_price = arb, ara
    if not np.isfinite(plan.headroom_to_ara):
        plan.headroom_to_ara = (ara / trigger - 1.0) if trigger > 0 else np.nan

    lots, notional, risk = position_size(equity, trigger, plan.stop, risk_pct,
                                         max_position_pct, cfg)
    plan.lots, plan.notional, plan.risk_idr = lots, notional, risk

    if lots <= 0:
        plan.warnings.append("position rounds to zero lots at this risk budget")
    plan.warnings.append(
        f"measured expectancy for this setup spans {stats['exp_low']:+.2%} to "
        f"{stats['exp_high']:+.2%} - it straddles zero, so treat size accordingly"
    )
    return plan


def simulate_orb(
    session: pd.DataFrame,
    reference_daily_volume: float,
    target_pct: float = 0.05,
    stop_pct: float = 0.03,
    opening_minutes: int = 30,
    min_pace: float = 1.5,
    use_or_low_stop: bool = True,
    exit_minute: int = 945,
    cost: float = 0.004,
) -> Dict[str, object]:
    """Replay one session under the opening-range-breakout rule.

    This is the rule the plans actually state, and it differs from "buy at the
    open" in three ways that all matter:

      * **No trigger, no trade.** If price never closes a bar above the opening
        range high, the day is skipped entirely. Most losing days never trigger.
      * **The volume filter.** A breakout on fading volume is not a burst; the
        pace test rejects it.
      * **A structural stop.** The opening-range low is a real level, often
        tighter than a flat -3%, which changes the risk per trade.

    Returns a dict with ``traded``, ``outcome``, ``entry``, ``exit``, ``return``.
    """
    if session is None or session.empty or len(session) < 4:
        return {"traded": False, "outcome": "no_data"}

    orange = opening_range(session, opening_minutes)
    if not orange or orange["or_high"] <= 0:
        return {"traded": False, "outcome": "no_range"}

    or_high = float(orange["or_high"])
    or_low = float(orange["or_low"])
    after = session[session["minutes_since_open"] > opening_minutes]
    if after.empty:
        return {"traded": False, "outcome": "no_bars_after_range"}

    entry = None
    entry_ts = None
    entry_index = None
    for i, (_, bar) in enumerate(after.iterrows()):
        if bar["close"] <= or_high:
            continue
        # Volume pace check at the moment of the breakout.
        so_far = float(session[session["ts"] <= bar["ts"]]["volume"].sum())
        elapsed = float(bar["session_pct"])
        if reference_daily_volume > 0 and elapsed > 0.02:
            pace = so_far / (reference_daily_volume * elapsed)
            if pace < min_pace:
                continue
        entry = float(bar["close"])
        entry_ts = bar["ts"]
        entry_index = i
        break

    if entry is None:
        return {"traded": False, "outcome": "no_trigger", "or_high": or_high}

    target = entry * (1.0 + target_pct)
    flat_stop = entry * (1.0 - stop_pct)
    stop = max(or_low, flat_stop) if use_or_low_stop else flat_stop
    if stop >= entry:
        stop = flat_stop

    held = after.iloc[entry_index + 1:]
    for _, bar in held.iterrows():
        if bar["minute_of_day"] >= exit_minute:
            break
        hit_target = bar["high"] >= target
        hit_stop = bar["low"] <= stop
        if hit_target and hit_stop:
            return {"traded": True, "outcome": "ambiguous", "entry": entry,
                    "entry_ts": entry_ts, "return": np.nan}
        if hit_target:
            return {"traded": True, "outcome": "target", "entry": entry,
                    "entry_ts": entry_ts, "exit": target,
                    "return": target / entry - 1.0 - cost}
        if hit_stop:
            return {"traded": True, "outcome": "stop", "entry": entry,
                    "entry_ts": entry_ts, "exit": stop,
                    "return": stop / entry - 1.0 - cost}

    exit_price = float(held["close"].iloc[-1]) if len(held) else entry
    return {"traded": True, "outcome": "close", "entry": entry,
            "entry_ts": entry_ts, "exit": exit_price,
            "return": exit_price / entry - 1.0 - cost}


def study_paths(
    candidates: List[DayCandidate],
    intraday_by_ticker: Dict[str, pd.DataFrame],
    cfg: Config,
    target_pct: float = 0.05,
    stop_pct: float = 0.03,
) -> pd.DataFrame:
    """Resolve target-vs-stop ordering on real intraday bars.

    This is the measurement that daily bars cannot provide. Each candidate is
    replayed against the following session's 5-minute bars, entering at that
    session's open.
    """
    rows = []
    for candidate in candidates:
        intraday = intraday_by_ticker.get(candidate.ticker)
        if intraday is None or intraday.empty:
            continue

        # The session AFTER the signal date.
        later = intraday[intraday["date"] > pd.Timestamp(candidate.date)]
        if later.empty:
            continue
        session = session_frame(later, later["date"].min())
        if session.empty:
            continue

        entry = float(session["open"].iloc[0])
        outcome = resolve_path(session, entry, entry * (1 + target_pct),
                               entry * (1 - stop_pct))
        rows.append({
            "ticker": candidate.ticker,
            "signal_date": candidate.date,
            "trade_date": session["date"].iloc[0],
            "setup": candidate.setup,
            "entry": entry,
            "outcome": outcome.get("outcome"),
            "return": outcome.get("return", np.nan),
            "hit_ts": outcome.get("ts"),
        })
    return pd.DataFrame(rows)
