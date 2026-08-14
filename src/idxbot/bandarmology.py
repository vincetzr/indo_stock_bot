"""Bandarmology metrics: net foreign accumulation and big-player footprints.

This is the arithmetic Indonesian platforms present as "Bandar Detector",
"Foreign Flow" or "Akumulasi/Distribusi". Every formula here is stated
explicitly, because the interesting part of bandarmology is not the idea - it
is that small differences in how the numbers are computed change the answer
completely, and most tools do not say which convention they use.

**This module computes F-BROKER, not F-FLAG.** Two different quantities are
called "foreign flow" on IDX and they do not reconcile:

    F-flag    IDX's per-trade foreign-investor flag, per stock/day, in SHARES.
              Obtainable free for 2019-2025; see scripts/foreign_flow_study.py.
    F-broker  the sum over members flagged foreign in config/brokers.yaml, in
              lots and IDR. That is what everything below computes.

A foreign investor can trade through a domestic member, and a foreign-owned
member mostly serves domestic retail, so the two never sum or difference.

**And F-flag has now been tested.** On 322,827 liquid rows it produced a 60-day
rank IC of -0.0254 (t=-10.88) that survived a chronological holdout, every year,
and every size tercile - and a long-short spread of +0.15% (t=0.76), because the
relationship is U-shaped rather than monotone. Net foreign, measured that way,
is not tradeable in either direction. See docs/FINDINGS.md Part V. Whether
F-broker behaves the same is still unmeasured.

**The inputs.** Everything below needs *broker summary*: one row per
(date, ticker, broker) carrying buy lot, buy value, sell lot and sell value.
That is an aggregation of the running trade, published by IDX after the close.
This repo has never obtained it from a free source - see docs/LIVE_DATA.md - so
these functions are exercised against pasted or CSV data supplied by the user,
and the module refuses to invent anything when the data is absent.

**The conventions that matter**, and which this module fixes explicitly:

*Value or lot?* Net value (rupiah) and net lot (shares/100) disagree whenever a
broker buys low and sells high within the window. Value is what moves a
position's cost basis and is used for flow; lots are used for inventory. Both
are reported, never blended.

*Gross or net?* "Foreign buy" on most screens means gross buy value by
foreign-flagged members. Net foreign is buy minus sell. A stock can show huge
foreign buying and negative net foreign on the same day.

*Which brokers are foreign?* There is no official flag, and this is where the
headline number most often goes wrong. `config/brokers.yaml` records ownership,
but **foreign-owned is not the same as foreign money**. The clearest case is YP,
Mirae Asset Sekuritas: Korean-owned, and simultaneously the largest *retail*
brokerage in Indonesia. Its tape is overwhelmingly domestic retail orders, and
it is routinely the highest-volume member of the day. Counting YP in "net
foreign" does not shade the answer, it dominates it.

So ``foreign_basis`` selects the convention:

    "institutional"  foreign-owned AND not a retail-tier member   (default)
    "ownership"      every member flagged foreign, YP included

The default is the one that means "foreign institutional money". The
alternative exists because it is what several platforms publish, and comparing
the two is the fastest way to see how much of a "foreign inflow" was actually
local retail routed through a foreign-owned broker. A member that serves both
a foreign parent and local clients is still one bucket in this data and cannot
be split any further.

*Crossing.* A broker that appears on both sides in size is often crossing a
block between its own clients, or moving stock between accounts. That inflates
gross volume and leaves net near zero, which is why concentration is measured
on *net* buyers only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .config import BrokerRegistry

LOT_SIZE = 100

# See the module docstring: foreign-OWNED is not foreign MONEY.
FOREIGN_BASES = ("institutional", "ownership")


def is_foreign(broker: str, registry: BrokerRegistry,
               basis: str = "institutional") -> bool:
    """Whether a member's flow counts as foreign under the chosen convention."""
    if basis not in FOREIGN_BASES:
        raise ValueError(f"foreign_basis must be one of {FOREIGN_BASES}, got {basis!r}")
    meta = registry.get(broker)
    if not meta.foreign:
        return False
    if basis == "ownership":
        return True
    # "institutional": drop foreign-owned members whose book is retail. YP
    # (Mirae) alone would otherwise swamp the number.
    return meta.tier != "retail"


# ---------------------------------------------------------------------------
# Net foreign flow
# ---------------------------------------------------------------------------

def foreign_flow(summary: pd.DataFrame, registry: BrokerRegistry,
                 foreign_basis: str = "institutional") -> pd.DataFrame:
    """Daily net foreign buy/sell and its running total, per ticker.

    The two headline numbers on every Indonesian platform::

        net_foreign_val = SUM(buy_val) - SUM(sell_val)   over foreign members
        foreign_flow    = cumulative sum of net_foreign_val through time

    The cumulative series is what gets charted as "Foreign Flow". Its *level*
    is arbitrary - it depends entirely on where the window starts - so only its
    slope and its turning points carry information. A rising line means foreign
    members have been net buyers since the window opened, nothing more.
    """
    if summary is None or summary.empty:
        return pd.DataFrame()

    df = summary.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df["is_foreign"] = df["broker"].map(
        lambda c: is_foreign(c, registry, foreign_basis))

    rows: List[dict] = []
    for (ticker, date), g in df.groupby(["ticker", "date"], sort=True):
        fo = g[g["is_foreign"]]
        buy_val = float(fo["buy_val"].sum())
        sell_val = float(fo["sell_val"].sum())
        buy_lot = float(fo["buy_lot"].sum())
        sell_lot = float(fo["sell_lot"].sum())
        gross = float(g["buy_val"].sum())
        rows.append({
            "date": date, "ticker": ticker,
            "foreign_buy_val": buy_val, "foreign_sell_val": sell_val,
            "net_foreign_val": buy_val - sell_val,
            "net_foreign_lot": buy_lot - sell_lot,
            # Share of the whole tape that foreign members were on. A large net
            # figure in a stock nobody else traded is a different event from the
            # same figure inside heavy two-way volume.
            "foreign_share": (buy_val + sell_val) / (2 * gross) if gross > 0 else 0.0,
            "market_val": gross,
        })

    out = pd.DataFrame(rows).sort_values(["ticker", "date"])
    out["foreign_flow"] = out.groupby("ticker")["net_foreign_val"].cumsum()
    out["foreign_flow_lot"] = out.groupby("ticker")["net_foreign_lot"].cumsum()
    out.attrs["foreign_basis"] = foreign_basis
    return out.reset_index(drop=True)


def foreign_basis_comparison(summary: pd.DataFrame,
                             registry: BrokerRegistry) -> pd.DataFrame:
    """Net foreign under both conventions side by side.

    The gap between the columns is how much of the "foreign" figure is really
    domestic retail arriving through a foreign-owned member. On IDX that gap is
    frequently larger than the number itself.
    """
    inst = foreign_flow(summary, registry, "institutional")
    own = foreign_flow(summary, registry, "ownership")
    if inst.empty or own.empty:
        return pd.DataFrame()
    merged = inst[["date", "ticker", "net_foreign_val"]].merge(
        own[["date", "ticker", "net_foreign_val"]],
        on=["date", "ticker"], suffixes=("_institutional", "_ownership"))
    merged["retail_via_foreign_broker"] = (
        merged["net_foreign_val_ownership"] - merged["net_foreign_val_institutional"])
    return merged


def foreign_streak(flow: pd.DataFrame, ticker: str) -> Dict[str, object]:
    """Consecutive sessions of one-way foreign flow, and the window totals.

    Practitioners watch the streak rather than any single day, because one
    day's net foreign is dominated by whichever block happened to cross.
    """
    g = flow[flow["ticker"] == ticker].sort_values("date")
    if g.empty:
        return {}
    sign = np.sign(g["net_foreign_val"].to_numpy())
    streak, direction = 0, 0
    for s in sign[::-1]:
        if s == 0:
            break
        if direction == 0:
            direction = int(s)
        if s != direction:
            break
        streak += 1
    return {
        "ticker": ticker,
        "streak_days": streak,
        "direction": "accumulation" if direction > 0 else
                     ("distribution" if direction < 0 else "flat"),
        "net_5d": float(g["net_foreign_val"].tail(5).sum()),
        "net_20d": float(g["net_foreign_val"].tail(20).sum()),
        "flow_level": float(g["foreign_flow"].iloc[-1]),
    }


# ---------------------------------------------------------------------------
# Big-player (bandar) footprints
# ---------------------------------------------------------------------------

@dataclass
class BandarProfile:
    """One broker's footprint in one ticker over a window."""
    broker: str
    name: str = ""
    tier: str = ""
    foreign: bool = False
    buy_lot: float = 0.0
    sell_lot: float = 0.0
    buy_val: float = 0.0
    sell_val: float = 0.0
    net_lot: float = 0.0
    net_val: float = 0.0
    avg_buy: float = np.nan
    avg_sell: float = np.nan
    days_active: int = 0
    days_net_buy: int = 0

    @property
    def is_accumulating(self) -> bool:
        return self.net_val > 0

    def as_row(self) -> Dict[str, object]:
        return {
            "broker": self.broker, "name": self.name, "tier": self.tier,
            "foreign": self.foreign, "net_lot": self.net_lot,
            "net_val": self.net_val, "avg_buy": self.avg_buy,
            "avg_sell": self.avg_sell, "days_active": self.days_active,
            "days_net_buy": self.days_net_buy,
        }


def bandar_profiles(summary: pd.DataFrame, registry: BrokerRegistry,
                    ticker: str, window: int = 20) -> List[BandarProfile]:
    """Per-broker accumulation over the last ``window`` sessions of one ticker.

    ``avg_buy`` is the value-weighted average price a broker paid::

        avg_buy = SUM(buy_val) / (SUM(buy_lot) * 100)

    This is the number retail traders call "harga bandar", and it is worth being
    precise about what it is *not*. It is the average over this window only, not
    a cost basis: a broker holding stock from before the window has a true basis
    this cannot see. It also says nothing about position - a broker can have a
    high average buy price and be flat, having sold everything back. For a real
    cost basis use ``analytics.broker_flow.build_ledger``, which tracks signed
    inventory across the whole history.
    """
    if summary is None or summary.empty:
        return []

    df = summary[summary["ticker"] == ticker].copy()
    if df.empty:
        return []
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    recent = sorted(df["date"].unique())[-window:]
    df = df[df["date"].isin(recent)]

    profiles: List[BandarProfile] = []
    for broker, g in df.groupby("broker"):
        meta = registry.get(broker)
        buy_lot = float(g["buy_lot"].sum())
        sell_lot = float(g["sell_lot"].sum())
        buy_val = float(g["buy_val"].sum())
        sell_val = float(g["sell_val"].sum())
        net_daily = g["buy_val"] - g["sell_val"]
        profiles.append(BandarProfile(
            broker=broker, name=meta.name, tier=str(meta.tier),
            foreign=bool(meta.foreign),
            buy_lot=buy_lot, sell_lot=sell_lot,
            buy_val=buy_val, sell_val=sell_val,
            net_lot=buy_lot - sell_lot, net_val=buy_val - sell_val,
            avg_buy=buy_val / (buy_lot * LOT_SIZE) if buy_lot > 0 else np.nan,
            avg_sell=sell_val / (sell_lot * LOT_SIZE) if sell_lot > 0 else np.nan,
            days_active=int(g["date"].nunique()),
            days_net_buy=int((net_daily > 0).sum()),
        ))
    return sorted(profiles, key=lambda p: -p.net_val)


def concentration(profiles: List[BandarProfile], top: int = 5) -> Dict[str, float]:
    """How concentrated the *net* buying was.

    Measured on net buyers only. Including sellers would let a broker that
    crossed a block with itself register as heavy participation while having
    accumulated nothing.

        share_i = net_val_i / SUM(net_val over net buyers)
        hhi     = SUM(share_i ^ 2)          1.0 = one broker took everything
        top_n   = SUM(largest n shares)
    """
    buyers = [p for p in profiles if p.net_val > 0]
    total = sum(p.net_val for p in buyers)
    if total <= 0:
        return {"hhi": 0.0, f"top{top}_share": 0.0, "buyer_count": 0,
                "lead_broker": "", "lead_share": 0.0}
    shares = sorted((p.net_val / total for p in buyers), reverse=True)
    lead = max(buyers, key=lambda p: p.net_val)
    return {
        "hhi": float(sum(s * s for s in shares)),
        f"top{top}_share": float(sum(shares[:top])),
        "buyer_count": len(buyers),
        "lead_broker": lead.broker,
        "lead_share": float(shares[0]),
    }


def bandar_score(profiles: List[BandarProfile], flow_row: Optional[Dict] = None,
                 top: int = 5) -> Dict[str, object]:
    """A 0-100 composite of the four things bandarmology actually looks at.

    **This score is not validated.** Cross-sectional testing in this repo
    refuted the price-only contrarian score outright (docs/FINDINGS.md Result 1),
    and the broker-flow components have never been tested against forward
    returns because the data was never obtained. It is assembled here because
    it is what the question asks for, and it is labelled unvalidated everywhere
    it surfaces. Treat it as a description of today's tape, not a forecast.

    Components, each mapped to 0-1:
      institutional_share - net value taken by bulge/foreign/local-institution
      concentration       - top-N share of net buying
      persistence         - fraction of sessions the lead broker was a net buyer
      foreign_tilt        - net foreign value as a share of gross
    """
    if not profiles:
        return {}

    inst = sum(p.net_val for p in profiles
               if p.net_val > 0 and p.tier in ("bulge", "foreign", "local_inst"))
    total_buy = sum(p.net_val for p in profiles if p.net_val > 0)
    conc = concentration(profiles, top=top)

    lead = next((p for p in profiles if p.broker == conc.get("lead_broker")), None)
    persistence = (lead.days_net_buy / lead.days_active
                   if lead and lead.days_active else 0.0)

    components = {
        "institutional_share": (inst / total_buy) if total_buy > 0 else 0.0,
        "concentration": float(conc.get(f"top{top}_share", 0.0)),
        "persistence": float(persistence),
        "foreign_tilt": 0.0,
    }
    if flow_row:
        gross = float(flow_row.get("market_val") or 0.0)
        if gross > 0:
            raw = float(flow_row.get("net_foreign_val", 0.0)) / gross
            components["foreign_tilt"] = float(np.clip(raw * 2 + 0.5, 0.0, 1.0))

    weights = {"institutional_share": 0.35, "concentration": 0.25,
               "persistence": 0.20, "foreign_tilt": 0.20}
    score = sum(weights[k] * components[k] for k in weights) * 100.0
    return {"score": float(np.clip(score, 0, 100)), "components": components,
            **conc, "validated": False}


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render(ticker: str, profiles: List[BandarProfile],
           flow: Optional[pd.DataFrame] = None, window: int = 20,
           top: int = 10, width: int = 82) -> str:
    line = "=" * width
    out = [line, f" BANDARMOLOGY — {ticker}   (last {window} sessions)", line]

    if not profiles:
        out.append(" No broker summary available for this ticker.")
        out.append(" This module computes nothing without it - see docs/LIVE_DATA.md")
        out.append(" for how to connect a source, or use `idxbot paste` to supply it.")
        return "\n".join(out + [line])

    if flow is not None and not flow.empty:
        streak = foreign_streak(flow, ticker)
        if streak:
            out.append(" NET FOREIGN")
            out.append(f"   5-day net    {streak['net_5d']:>18,.0f}")
            out.append(f"   20-day net   {streak['net_20d']:>18,.0f}")
            out.append(f"   streak       {streak['streak_days']} sessions of "
                       f"{streak['direction']}")
            out.append("")

    out.append(f" {'broker':<7}{'name':<26}{'tier':<11}{'net lot':>12}"
               f"{'net value':>16}{'avg buy':>10}")
    for p in profiles[:top]:
        avg = f"{p.avg_buy:,.0f}" if np.isfinite(p.avg_buy) else "-"
        out.append(f" {p.broker:<7}{p.name[:25]:<26}{p.tier:<11}"
                   f"{p.net_lot:>12,.0f}{p.net_val:>16,.0f}{avg:>10}")

    sellers = [p for p in profiles if p.net_val < 0]
    if sellers:
        out.append("")
        out.append(" heaviest net sellers")
        for p in sorted(sellers, key=lambda x: x.net_val)[:3]:
            avg = f"{p.avg_sell:,.0f}" if np.isfinite(p.avg_sell) else "-"
            out.append(f" {p.broker:<7}{p.name[:25]:<26}{p.tier:<11}"
                       f"{p.net_lot:>12,.0f}{p.net_val:>16,.0f}{avg:>10}")

    conc = concentration(profiles, top=5)
    out.append("")
    out.append(" CONCENTRATION OF NET BUYING")
    out.append(f"   net buyers   {conc['buyer_count']}")
    out.append(f"   top-5 share  {conc['top5_share']:.0%}")
    out.append(f"   HHI          {conc['hhi']:.3f}  "
               f"(1.00 = a single broker took everything)")
    if conc["lead_broker"]:
        out.append(f"   lead         {conc['lead_broker']} "
                   f"at {conc['lead_share']:.0%} of net buying")

    scored = bandar_score(profiles, top=5)
    if scored:
        out.append("")
        out.append(f" BANDAR SCORE {scored['score']:.0f}/100   ** NOT VALIDATED **")
        for k, v in scored["components"].items():
            out.append(f"   {k:<22}{v:>6.2f}")
        out.append("   No broker-flow signal in this repo has ever been tested")
        out.append("   against forward returns - the data was never obtained. The")
        out.append("   price-only contrarian score WAS tested, and was refuted.")
    return "\n".join(out + [line])


# ---------------------------------------------------------------------------
# Bandar detection
# ---------------------------------------------------------------------------
#
# A bandar is not "whoever bought the most today". The word describes a
# controlling party - typically the issuer's owner or an affiliate - accumulating
# their own stock through one or two chosen brokerage houses, in size, over
# weeks. That definition has a specific footprint in broker summary, and it is
# quite different from the footprint of ordinary institutional demand:
#
#   loyalty      the SAME broker does the buying, week after week. Genuine
#                institutional demand arrives through whichever desk has the
#                axe that day; an owner works through their house.
#   persistence  present on a large fraction of sessions, not one block trade.
#   dominance    that broker's share of the stock's whole tape is large. An
#                owner accumulating a small-cap can be most of the volume.
#   stealth      price barely moves while inventory builds. Someone marking up
#                is not accumulating; someone accumulating does not want to pay
#                more than they must.
#
# The last one is the reason a naive "top buyer" screen finds nothing useful:
# the loudest buyer on a +15% day is chasing, not accumulating.
#
# NOTE ON EVIDENCE. Every number below is a description of a footprint, not a
# prediction. The one broker-flow-shaped hypothesis this repo HAS tested - the
# contrarian accumulation score - was refuted on price-only data (FINDINGS
# Result 1). Whether a real bandar footprint predicts returns is unmeasured,
# because the data to measure it has never been obtained.

BANDAR_MIN_DAYS = 10          # a block trade is not a campaign
BANDAR_MIN_LOYALTY = 0.35     # share of the window's net buying
BANDAR_MIN_PRESENCE = 0.40    # fraction of sessions as a net buyer


@dataclass
class BandarSignature:
    """One broker's claim to be working an accumulation campaign in a name."""
    broker: str
    name: str = ""
    tier: str = ""
    loyalty: float = 0.0        # share of all net buying in the window
    presence: float = 0.0       # fraction of sessions as a net buyer
    dominance: float = 0.0      # share of the stock's gross traded value
    stealth: float = 0.0        # 0-1; 1 = inventory built with no price move
    net_val: float = 0.0
    net_lot: float = 0.0
    avg_buy: float = np.nan
    days_present: int = 0
    sessions: int = 0

    @property
    def qualifies(self) -> bool:
        return (self.days_present >= BANDAR_MIN_DAYS
                and self.loyalty >= BANDAR_MIN_LOYALTY
                and self.presence >= BANDAR_MIN_PRESENCE)

    def as_row(self) -> Dict[str, object]:
        return {"broker": self.broker, "name": self.name, "tier": self.tier,
                "loyalty": self.loyalty, "presence": self.presence,
                "dominance": self.dominance, "stealth": self.stealth,
                "net_val": self.net_val, "avg_buy": self.avg_buy,
                "days_present": self.days_present, "qualifies": self.qualifies}


def detect_bandar(summary: pd.DataFrame, registry: BrokerRegistry, ticker: str,
                  bars: Optional[pd.DataFrame] = None,
                  window: int = 60) -> List[BandarSignature]:
    """Find brokers whose footprint matches owner-style accumulation.

    ``bars`` is optional daily OHLCV for the same ticker; without it ``stealth``
    cannot be computed and is left at zero rather than guessed.

    Formulas over the trailing ``window`` sessions, per broker::

        loyalty   = SUM(net_val_i for i where net_val_i > 0)
                    / SUM(all positive net_val across brokers)
        presence  = (sessions where this broker was a net buyer) / sessions
        dominance = (buy_val + sell_val) / SUM(buy_val + sell_val, all brokers)
        stealth   = 1 - clip(|price change over window| / accumulation share, 0, 1)

    Stealth deserves a word. It compares how far the price travelled against how
    much of the float this broker absorbed. Absorbing 30% of the tape while the
    price moves 2% is a very different act from absorbing 30% while it moves
    40%, and only the first is accumulation in the sense the word implies.
    """
    if summary is None or summary.empty:
        return []
    df = summary[summary["ticker"] == ticker].copy()
    if df.empty:
        return []
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    sessions = sorted(df["date"].unique())[-window:]
    df = df[df["date"].isin(sessions)]
    n_sessions = len(sessions)
    if n_sessions == 0:
        return []

    df["net_val"] = df["buy_val"] - df["sell_val"]
    total_net_buying = float(df.loc[df["net_val"] > 0, "net_val"].sum())
    total_gross = float((df["buy_val"] + df["sell_val"]).sum())

    price_move = np.nan
    if bars is not None and not bars.empty:
        b = bars[bars["date"].isin(sessions)].sort_values("date")
        if len(b) >= 2:
            first, last = float(b["close"].iloc[0]), float(b["close"].iloc[-1])
            if first > 0:
                price_move = abs(last / first - 1.0)

    out: List[BandarSignature] = []
    for broker, g in df.groupby("broker"):
        meta = registry.get(broker)
        net = float(g["net_val"].sum())
        buy_lot = float(g["buy_lot"].sum())
        buy_val = float(g["buy_val"].sum())
        gross = float((g["buy_val"] + g["sell_val"]).sum())
        pos_days = int((g.groupby("date")["net_val"].sum() > 0).sum())

        loyalty = (float(g.loc[g["net_val"] > 0, "net_val"].sum()) / total_net_buying
                   if total_net_buying > 0 else 0.0)
        dominance = gross / total_gross if total_gross > 0 else 0.0

        stealth = 0.0
        if np.isfinite(price_move) and dominance > 0:
            # Price travelled per unit of tape absorbed. Low ratio = quiet.
            stealth = float(np.clip(1.0 - (price_move / max(dominance, 1e-9)) / 2.0,
                                    0.0, 1.0))

        out.append(BandarSignature(
            broker=broker, name=meta.name, tier=str(meta.tier),
            loyalty=loyalty, presence=pos_days / n_sessions, dominance=dominance,
            stealth=stealth, net_val=net,
            net_lot=float(g["buy_lot"].sum() - g["sell_lot"].sum()),
            avg_buy=buy_val / (buy_lot * LOT_SIZE) if buy_lot > 0 else np.nan,
            days_present=pos_days, sessions=n_sessions))

    return sorted(out, key=lambda s: (-s.qualifies, -s.loyalty))


def render_bandar(ticker: str, signatures: List[BandarSignature],
                  width: int = 82) -> str:
    line = "=" * width
    out = [line, f" BANDAR DETECTION — {ticker}", line]
    if not signatures:
        out.append(" No broker summary for this ticker. Nothing is computed without it.")
        return "\n".join(out + [line])

    out.append(f" window: {signatures[0].sessions} sessions")
    out.append(f" a bandar footprint needs loyalty >= {BANDAR_MIN_LOYALTY:.0%}, "
               f"presence >= {BANDAR_MIN_PRESENCE:.0%}, >= {BANDAR_MIN_DAYS} days")
    out.append("")
    out.append(f" {'broker':<7}{'name':<24}{'loyalty':>9}{'presence':>10}"
               f"{'dominance':>11}{'stealth':>9}{'avg buy':>10}")
    for s in signatures[:8]:
        avg = f"{s.avg_buy:,.0f}" if np.isfinite(s.avg_buy) else "-"
        mark = " <<" if s.qualifies else ""
        out.append(f" {s.broker:<7}{s.name[:23]:<24}{s.loyalty:>9.0%}"
                   f"{s.presence:>10.0%}{s.dominance:>11.0%}{s.stealth:>9.2f}"
                   f"{avg:>10}{mark}")

    hits = [s for s in signatures if s.qualifies]
    out.append("")
    if hits:
        out.append(f" {len(hits)} broker(s) match the footprint:")
        for s in hits:
            out.append(f"   {s.broker} ({s.name}) took {s.loyalty:.0%} of net buying "
                       f"across {s.days_present}/{s.sessions} sessions,")
            out.append(f"     {s.dominance:.0%} of the whole tape, "
                       f"average price {s.avg_buy:,.0f}")
    else:
        out.append(" No broker matches. The buying was spread across the market,")
        out.append(" which is what ordinary demand looks like.")
    out.append("")
    out.append(" A footprint is a description, not a forecast. No broker-flow")
    out.append(" signal in this repo has been tested against forward returns.")
    return "\n".join(out + [line])
