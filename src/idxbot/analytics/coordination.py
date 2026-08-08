"""Reverse-engineering institutional intent: coordination, lead-lag, and timing.

``playbook.py`` profiles each desk in isolation — how it enters, how long it
holds, how much of a move it keeps. That is half the picture. The other half is
relational, and it is where the tradeable questions live:

  * **Do they act together?** If AK and BK buy the same name in the same week
    far more often than chance, that is either genuine herding or a shared
    signal. Either way, two bulge desks accumulating is a different event from
    one, and should not be scored the same.
  * **Who moves first?** If BK's net buying today predicts AK's net buying three
    days from now, BK is the leader and AK is confirmation that arrives too
    late. You want to act on the leader.
  * **When is it safe to join?** A campaign 10% complete and a campaign 90%
    complete look identical in a snapshot of "BK is accumulating". Their forward
    returns are not the same. Stage matters more than the fact of accumulation.

Every function here needs real broker summary. On simulated data they measure
the simulator's assumptions and nothing else, which is why each result carries
the provenance of the data it was computed from.

A caution that applies to all of it: correlated buying is not proof of
coordination. Desks share research, react to the same news, and run similar
factor models. Observing that two members bought together says they behaved
alike, not that they agreed to.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from ..config import BrokerRegistry


def _net_flow_matrix(summary: pd.DataFrame, brokers: Optional[Sequence[str]] = None,
                     value: bool = True) -> pd.DataFrame:
    """Wide date x broker matrix of daily net flow, summed across tickers."""
    if summary is None or summary.empty:
        return pd.DataFrame()
    df = summary.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df["net"] = (df["buy_val"] - df["sell_val"]) if value else (df["buy_lot"] - df["sell_lot"])
    if brokers:
        df = df[df["broker"].isin([b.upper() for b in brokers])]
    if df.empty:
        return pd.DataFrame()
    return df.pivot_table(index="date", columns="broker", values="net",
                          aggfunc="sum").fillna(0.0)


def coordination_matrix(
    summary: pd.DataFrame,
    registry: BrokerRegistry,
    tiers: Sequence[str] = ("bulge",),
    min_days: int = 60,
) -> pd.DataFrame:
    """Pairwise correlation of daily net flow between desks.

    Correlation is computed on flow *normalised per broker*, so a desk that
    simply trades larger does not dominate. High positive values mean two
    members tend to be on the same side on the same day.
    """
    if summary is None or summary.empty:
        return pd.DataFrame()

    codes = [c for c in summary["broker"].unique()
             if registry.get(c).tier in tiers]
    matrix = _net_flow_matrix(summary, codes)
    if matrix.empty or len(matrix) < min_days:
        return pd.DataFrame()

    # Scale each column so correlation reflects direction, not size.
    scaled = matrix / matrix.abs().mean().replace(0, np.nan)
    return scaled.corr(min_periods=min_days // 2)


def lead_lag(
    summary: pd.DataFrame,
    registry: BrokerRegistry,
    tiers: Sequence[str] = ("bulge",),
    max_lag: int = 5,
    min_days: int = 90,
) -> pd.DataFrame:
    """Who moves first: does broker A's flow today predict broker B's later?

    For every ordered pair, correlate A's net flow at *t* with B's at *t+lag*
    for lag 1..max_lag, and keep the strongest. A pair where A leads B with a
    materially higher correlation than B leads A identifies the desk worth
    watching.
    """
    if summary is None or summary.empty:
        return pd.DataFrame()

    codes = [c for c in summary["broker"].unique() if registry.get(c).tier in tiers]
    matrix = _net_flow_matrix(summary, codes)
    if matrix.empty or len(matrix) < min_days:
        return pd.DataFrame()

    scaled = matrix / matrix.abs().mean().replace(0, np.nan)
    rows: List[dict] = []
    for leader in scaled.columns:
        for follower in scaled.columns:
            if leader == follower:
                continue
            best_lag, best_corr = 0, 0.0
            for lag in range(1, max_lag + 1):
                a = scaled[leader].iloc[:-lag]
                b = scaled[follower].iloc[lag:]
                if len(a) < min_days // 2:
                    continue
                corr = float(np.corrcoef(a.to_numpy(), b.to_numpy())[0, 1])
                if np.isfinite(corr) and abs(corr) > abs(best_corr):
                    best_lag, best_corr = lag, corr
            if best_lag:
                rows.append({
                    "leader": leader, "follower": follower,
                    "lag_days": best_lag, "corr": best_corr,
                    "leader_name": registry.get(leader).name,
                })

    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)

    # Net leadership: A leads B more than B leads A.
    pair_key = out.apply(lambda r: tuple(sorted([r["leader"], r["follower"]])), axis=1)
    out["pair"] = pair_key
    best = []
    for _pair, group in out.groupby("pair"):
        top = group.loc[group["corr"].abs().idxmax()]
        reverse = group[group["leader"] == top["follower"]]
        edge = abs(top["corr"]) - (abs(reverse["corr"].iloc[0]) if len(reverse) else 0.0)
        best.append({**top.to_dict(), "leadership_edge": edge})
    return pd.DataFrame(best).sort_values("leadership_edge", ascending=False).reset_index(
        drop=True
    )


def herding_index(
    summary: pd.DataFrame,
    registry: BrokerRegistry,
    tier: str = "bulge",
) -> pd.DataFrame:
    """Daily measure of how aligned the bulge desks are.

    Returns, per (ticker, date), the share of active bulge desks on the buy
    side. Values near 1 mean every institutional desk present was buying — a
    far stronger statement than any single desk's net flow.
    """
    if summary is None or summary.empty:
        return pd.DataFrame()

    df = summary.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df["tier"] = df["broker"].map(lambda c: registry.get(c).tier)
    df = df[df["tier"] == tier]
    if df.empty:
        return pd.DataFrame()

    df["net"] = df["buy_val"] - df["sell_val"]
    df["active"] = (df["buy_val"] + df["sell_val"]) > 0

    rows = []
    for (ticker, date), g in df.groupby(["ticker", "date"]):
        active = g[g["active"]]
        if len(active) < 2:
            continue
        buyers = int((active["net"] > 0).sum())
        rows.append({
            "ticker": ticker, "date": date,
            "desks_active": len(active),
            "desks_buying": buyers,
            "buy_share": buyers / len(active),
            "net_val": float(active["net"].sum()),
        })
    return pd.DataFrame(rows)


def campaign_stage_returns(
    campaigns: pd.DataFrame,
    prices_by_ticker: Dict[str, pd.DataFrame],
    stages: Sequence[float] = (0.1, 0.25, 0.5, 0.75, 0.9),
    horizons: Sequence[int] = (10, 20, 60),
) -> pd.DataFrame:
    """The question that decides whether following a desk is tradeable.

    For each detected campaign, sample points at fractions of the way through
    the accumulation leg and measure the forward return from there. If joining
    at 25% pays and joining at 90% does not, the signal is not "this desk is
    accumulating" but "this desk *started* accumulating recently" — a much
    narrower and more perishable edge.
    """
    if campaigns is None or campaigns.empty:
        return pd.DataFrame()

    rows: List[dict] = []
    for _, camp in campaigns.iterrows():
        bars = prices_by_ticker.get(str(camp["ticker"]).upper())
        if bars is None or bars.empty:
            continue
        bars = bars.reset_index(drop=True)
        start = pd.Timestamp(camp["acc_start"])
        end = pd.Timestamp(camp["acc_end"])

        window = bars[(bars["date"] >= start) & (bars["date"] <= end)]
        if len(window) < 5:
            continue
        first_pos = int(window.index[0])
        length = len(window)

        for stage in stages:
            offset = int(round(stage * (length - 1)))
            pos = first_pos + offset
            if pos >= len(bars):
                continue
            entry = float(bars["close"].iloc[pos])
            if entry <= 0:
                continue
            record = {
                "ticker": camp["ticker"], "broker": camp["broker"],
                "stage": stage, "entry_date": bars["date"].iloc[pos], "entry": entry,
            }
            for h in horizons:
                target = pos + h
                record[f"fwd_{h}"] = (
                    float(bars["close"].iloc[target]) / entry - 1.0
                    if target < len(bars) else np.nan
                )
            rows.append(record)

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def summarise_stage_returns(stage_returns: pd.DataFrame,
                            horizons: Sequence[int] = (10, 20, 60)) -> pd.DataFrame:
    """Aggregate stage-entry returns into a join-timing table."""
    if stage_returns is None or stage_returns.empty:
        return pd.DataFrame()
    rows = []
    for stage, g in stage_returns.groupby("stage"):
        record = {"stage": stage, "n": int(len(g))}
        for h in horizons:
            col = f"fwd_{h}"
            if col not in g:
                continue
            s = g[col].dropna()
            if len(s) < 5:
                record[f"mean_{h}d"] = np.nan
                continue
            record[f"mean_{h}d"] = float(s.mean())
            record[f"median_{h}d"] = float(s.median())
            record[f"win_{h}d"] = float((s > 0).mean())
        rows.append(record)
    return pd.DataFrame(rows).sort_values("stage").reset_index(drop=True)


def render_plan(
    playbook: pd.DataFrame,
    coordination: pd.DataFrame,
    leadlag: pd.DataFrame,
    stage_summary: pd.DataFrame,
    data_is_real: bool,
    provenance: str = "",
    width: int = 78,
) -> str:
    """The reverse-engineered operating plan, as a report."""
    line = "=" * width
    out = [line, " REVERSE-ENGINEERED INSTITUTIONAL PLAN", line]

    if not data_is_real:
        out.append(" !! BROKER FLOW IS SIMULATED (%s)." % (provenance or "synthetic"))
        out.append(" !! Everything below describes the simulator's built-in assumptions,")
        out.append(" !! not the market. It demonstrates that the analysis runs; it is not")
        out.append(" !! evidence about how any desk behaves. Connect real data first.")
        out.append("-" * width)

    # ---- 1. individual playbooks
    out.append("")
    out.append(" 1. HOW EACH DESK OPERATES")
    if playbook is None or playbook.empty:
        out.append("    (no completed campaigns to profile)")
    else:
        out.append(f"    {'desk':<6}{'style':<38}{'enter':>7}{'markup':>9}{'keep':>7}{'hold':>7}")
        for _, r in playbook.iterrows():
            def pct(v, d=0):
                return "n/a" if not np.isfinite(v) else f"{v * 100:.{d}f}%"
            out.append(
                f"    {r['broker']:<6}{str(r.get('style', ''))[:36]:<38}"
                f"{pct(r.get('median_entry_percentile', np.nan)):>7}"
                f"{pct(r.get('median_markup_pct', np.nan)):>9}"
                f"{pct(r.get('median_exit_capture', np.nan)):>7}"
                f"{r.get('median_holding_days', np.nan):>6.0f}d"
            )
        out.append("")
        out.append("    enter  = where in the trailing range their buy VWAP sat")
        out.append("    markup = how far price ran above that before they unwound")
        out.append("    keep   = share of the available move they actually captured")

    # ---- 2. do they move together
    out.append("")
    out.append(" 2. DO THEY ACT TOGETHER?")
    if coordination is None or coordination.empty:
        out.append("    (insufficient overlapping history)")
    else:
        pairs = []
        cols = list(coordination.columns)
        for i, a in enumerate(cols):
            for b in cols[i + 1:]:
                v = coordination.loc[a, b]
                if np.isfinite(v):
                    pairs.append((abs(v), a, b, v))
        pairs.sort(reverse=True)
        for _, a, b, v in pairs[:6]:
            verdict = ("move together" if v > 0.3 else
                       "opposite sides" if v < -0.3 else "largely independent")
            out.append(f"    {a}-{b}   corr {v:+.2f}   {verdict}")
        out.append("")
        out.append("    Correlated buying is not proof of coordination - desks share")
        out.append("    research and react to the same news. It does mean two desks")
        out.append("    accumulating is a different event from one.")

    # ---- 3. who leads
    out.append("")
    out.append(" 3. WHO MOVES FIRST")
    if leadlag is None or leadlag.empty:
        out.append("    (insufficient history to establish lead-lag)")
    else:
        for _, r in leadlag.head(5).iterrows():
            out.append(
                f"    {r['leader']} leads {r['follower']} by {int(r['lag_days'])}d"
                f"   corr {r['corr']:+.2f}   edge {r['leadership_edge']:+.2f}"
            )
        out.append("")
        out.append("    Act on the leader. The follower is confirmation that arrives")
        out.append("    after the price has already moved.")

    # ---- 4. when to join
    out.append("")
    out.append(" 4. WHEN IS IT SAFE TO JOIN?")
    if stage_summary is None or stage_summary.empty:
        out.append("    (no campaigns with forward data)")
    else:
        out.append(f"    {'stage':>7}{'n':>7}{'10d':>9}{'20d':>9}{'60d':>9}{'win 60d':>9}")
        for _, r in stage_summary.iterrows():
            def cell(key):
                v = r.get(key, np.nan)
                return f"{v * 100:>8.2f}%" if np.isfinite(v) else f"{'-':>9}"
            win = r.get("win_60d", np.nan)
            out.append(
                f"    {r['stage'] * 100:>6.0f}%{int(r['n']):>7}"
                f"{cell('mean_10d')}{cell('mean_20d')}{cell('mean_60d')}"
                + (f"{win * 100:>8.0f}%" if np.isfinite(win) else f"{'-':>9}")
            )
        out.append("")
        out.append("    stage = how far through their accumulation leg you entered.")
        out.append("    If early stages pay and late ones do not, the edge is in")
        out.append("    catching the START of a build, not in the fact of one.")

    out.append("")
    out.append(line)
    return "\n".join(out)
