#!/usr/bin/env python3
"""Who holds it: KSEI's investor-type composition and monthly flows.

    python3 scripts/ownership.py BBCA ADRO PANI   # named tickers
    python3 scripts/ownership.py                  # the open book's names
    python3 scripts/ownership.py --market         # all equities, by class

Needs ``scripts/ksei_collect.py`` first (five minutes cold, no network warm).

DESCRIPTIVE, AND MEASURED AS SUCH. H64 tested whether a class's monthly flow
predicts the next quarter and found nothing tradeable — and the retail result
came back with the OPPOSITE sign to CLAUDE.md §12's premise. Those numbers are
read from ``reports/ksei_flow.json`` and printed under every report, never
typed here (A47: a number copied out of a study has no link back to it). If
the file is missing the report says so instead of borrowing a figure.

THREE THINGS EVERY READER OF THIS NEEDS
---------------------------------------
1. A type is a class of holder, not an actor (§6.1). "Retail bought 2% of the
   balance" can be ten thousand accounts or one founder buying in their own
   name. KSEI does not say which.
2. "In the depository balance" is not the shareholder base. For BBCA, TLKM,
   BBRI and UNVR the balance is 15–48% of issued shares; the rest is a block
   KSEI's file does not itemise, which matches the controlling stake to within
   a point. Shares of the balance are shares of what actually trades.
3. Two kinds of month carry "flows" that are not trades. One where the
   balance itself moved by more than 0.5% — a block entering or leaving, an
   issuance, a split (flagged; H64 excluded them). And one where the
   individual type's change is mirrored by another local type's: BYAN in
   October 2026 moved 10.75bn shares from "individual" to "corporate", which
   read as flow is retail selling Rp 98 trillion in a month (flagged;
   ``ksei.transfer_flags``; H64 prints its result with and without them).

Named holders above 1% are a different publication: IDX posts "Pemegang Saham
di atas 1% (KSEI)" monthly since 2026-03-03, behind the same Cloudflare check
that blocks every idx.co.id request from here. Download it in a browser if you
want it; this file does not touch it.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

from idxbot.data import ksei                                    # noqa: E402

RESULT = os.path.join("reports", "ksei_flow.json")
DLOG_FLAG = 0.005
#: A partition of the balance: every unit lands in exactly one bucket.
BUCKETS = {
    "retail": ["local_ID"],
    "local corp": ["local_CP"],
    "local inst": ["local_IS", "local_PF", "local_MF", "local_IB"],
    "local other": ["local_SC", "local_FD", "local_OT"],
    "foreign": ksei.FOREIGN,
}


def composition(P: pd.DataFrame) -> pd.DataFrame:
    """Each bucket's share of the balance, plus coverage. Rows sum to 1."""
    held = (P["local_total"] + P["foreign_total"]).astype(float)
    out = P[["month_end", "ticker", "sec_num", "price"]].copy()
    out["held"] = held
    out["coverage"] = np.where(P["sec_num"] > 0, held / P["sec_num"], np.nan)
    for b, cols in BUCKETS.items():
        out[b] = np.where(held > 0, P[cols].sum(axis=1) / held, np.nan)
    return out


def flows(P: pd.DataFrame) -> pd.DataFrame:
    """Retail and foreign net flow as a share of the previous balance.

    Consecutive months only, within ticker. ``moved`` flags a month whose
    balance or share count changed by more than 0.5%, where a "flow" is a
    block or an issuance rather than trading.
    """
    P = P.sort_values(["ticker", "month_end"]).copy()
    grid = pd.Index(sorted(P["month_end"].unique()))
    k = P["month_end"].map(pd.Series(np.arange(len(grid)), index=grid))
    held = (P["local_total"] + P["foreign_total"]).astype(float)
    g = P.groupby("ticker")
    prev_held = held.groupby(P["ticker"]).shift(1)
    ok = (k - k.groupby(P["ticker"]).shift(1)) == 1
    out = P[["month_end", "ticker"]].copy()
    with np.errstate(divide="ignore", invalid="ignore"):
        out["retail_flow"] = ((P["local_ID"] - g["local_ID"].shift(1))
                              / prev_held).where(ok)
        out["foreign_flow"] = ((P["foreign_total"] - g["foreign_total"].shift(1))
                               / prev_held).where(ok)
        moved = ((np.log(held / prev_held)).abs() > DLOG_FLAG) | (
            (np.log(P["sec_num"] / g["sec_num"].shift(1))).abs() > DLOG_FLAG)
    out["moved"] = moved.where(ok, False).astype(bool)
    out["transfer"] = ksei.transfer_flags(P).reindex(P.index).astype(bool)
    clean = ok & ~out["moved"] & ~out["transfer"]
    out["retail_rp"] = ((P["local_ID"] - g["local_ID"].shift(1))
                        * P["price"]).where(clean)
    out["foreign_rp"] = ((P["foreign_total"] - g["foreign_total"].shift(1))
                         * P["price"]).where(clean)
    return out


def h64_caveat(path: str = RESULT) -> List[str]:
    """The measured result, read from the study's own output."""
    if not os.path.exists(path):
        return ["H64 (retail/foreign flow -> next quarter) has not been run on "
                "this machine: `python3 scripts/ksei_flow.py` (~40 s). No "
                "figure is quoted in its place."]
    with open(path) as fh:
        r = json.load(fh)
    s = r["signals"]
    rt, fo = s["f_retail"], s["f_foreign"]
    lines = [
        f"H64, {r['spec']}, in-sample {r['in_sample'][0]} … "
        f"{r['in_sample'][1]}, k={r['confirm_k']} sessions:",
        f"  retail flow  IC {rt['ic']:+.4f} (HAC t {rt['t']:+.2f}, "
        f"permutation p {rt['p']:.4f}) — predicted NEGATIVE, came back "
        f"{'positive' if rt['ic'] > 0 else 'negative'}; gate "
        f"{'PASSED' if rt['gate'] else 'FAILED'} (bar p < {r['bonferroni']:.6f})",
        f"  foreign flow IC {fo['ic']:+.4f} (HAC t {fo['t']:+.2f}, "
        f"permutation p {fo['p']:.4f}); gate "
        f"{'PASSED' if fo['gate'] else 'FAILED'}",
        f"  holding what retail SOLD, net of cost: "
        f"{rt['net_annual_predicted_leg']*100:+.1f}%/yr long-only",
    ]
    ex = rt.get(f"exploratory_hold_retail_bought_net_annual_k{r['confirm_k']}")
    if ex is not None:
        lines.append(f"  holding what retail BOUGHT (exploratory, post-result): "
                     f"{ex*100:+.1f}%/yr")
    lines.append("  Neither side is a trade. What follows is who holds the "
                 "stock, not what it will do.")
    return lines


def open_book_tickers() -> List[str]:
    from idxbot import signal_store as ss
    em = ss.load_emitted()
    if em.empty:
        return []
    oc = ss.load_outcomes()
    if len(oc) and "settled" in oc.columns:
        em = em[~em["signal_id"].isin(
            set(oc.loc[oc["settled"].astype(bool), "signal_id"]))]
    em = em[[ss.is_live(r, v) for r, v in zip(em["rule"], em["rule_version"])]]
    return sorted(em["ticker"].unique())


def ticker_report(t: str, C: pd.DataFrame, F: pd.DataFrame,
                  created: pd.Series, sessions: Optional[pd.DatetimeIndex]
                  ) -> List[str]:
    c = C[C["ticker"] == t].sort_values("month_end")
    if c.empty:
        return [f"{t}: not in the KSEI file — unlisted, or a code change"]
    last = c.iloc[-1]
    me = last["month_end"]
    stamp = created.get(me)
    when = ""
    if stamp is not None and sessions is not None and len(sessions):
        a = ksei.available_on(stamp.to_pydatetime(), sessions, 1)
        when = f"; actionable from {a.date()}" if a is not None else ""
    out = [f"{t}   KSEI state {me.date()}"
           + (f" (file created {stamp:%Y-%m-%d %H:%M}{when})" if stamp is not None else "")]
    out.append(f"  in the depository balance  {last['coverage']:.1%} of issued "
               f"shares" + ("" if last["coverage"] > 0.98 else
                            "  — the rest is a block KSEI does not itemise, "
                            "usually the controlling stake"))
    out.append("  of the balance:  " + "   ".join(
        f"{b} {last[b]:.1%}" for b in BUCKETS))
    f = F[F["ticker"] == t].sort_values("month_end").tail(3)
    if len(f):
        head = "".join(f"{d:%b %Y}".rjust(11) for d in f["month_end"])
        out.append(f"  net flow, % of balance  {head}")
        for col, lab in (("retail_flow", "retail"), ("foreign_flow", "foreign")):
            out.append(f"    {lab:<21}" + "".join(
                ("n/a" if not np.isfinite(v) else f"{v*100:+.2f}%").rjust(11)
                for v in f[col]))
        for d, m, x in zip(f["month_end"], f["moved"], f["transfer"]):
            if m:
                out.append(f"  ! {d:%b %Y}: the balance or share count moved "
                           f">0.5% — a block, an issuance or a split. That "
                           f"month's flows are not trades.")
            if x:
                out.append(f"  ! {d:%b %Y}: the individual type's change is "
                           f"mirrored by another local type's — a stake "
                           f"re-registered (e.g. a founder moving shares into a "
                           f"company), not the crowd trading.")
    return out


def market_report(F: pd.DataFrame, C: pd.DataFrame, n: int = 6) -> List[str]:
    m = (F.groupby("month_end")[["retail_rp", "foreign_rp"]].sum() / 1e12).tail(n)
    held = C.assign(rv=C["retail"] * C["held"], fv=C["foreign"] * C["held"])
    share = held.groupby("month_end")[["rv", "fv", "held"]].sum()
    out = ["NET FLOW BY CLASS, all equities, Rp trillion (Δunits × month-end "
           "price; a name-month is excluded when its balance moved >0.5% or "
           "its individual-type change is mirrored by another local type)",
           f"  {'month':<10}{'retail':>10}{'foreign':>10}"]
    for d, r in m.iterrows():
        out.append(f"  {d:%Y-%m}   {r['retail_rp']:>+9.2f} {r['foreign_rp']:>+9.2f}")
    y = share.iloc[[0, len(share) // 2, -1]]
    out.append("  unit-weighted share of the balance: " + "   ".join(
        f"{d:%Y-%m} retail {r['rv']/r['held']:.1%} foreign {r['fv']/r['held']:.1%}"
        for d, r in y.iterrows()))
    out.append("  The two columns are not mirror images: local institutions, "
               "corporates and the rest are the other side.")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("tickers", nargs="*")
    ap.add_argument("--market", action="store_true")
    a = ap.parse_args()
    if not os.path.exists(ksei.PANEL_PATH):
        print(f"no KSEI panel at {ksei.PANEL_PATH} — run "
              f"`python3 scripts/ksei_collect.py` first")
        return 1
    P = pd.read_parquet(ksei.PANEL_PATH)
    C, F = composition(P), flows(P)
    created = P.groupby("month_end")["created"].first()
    sessions = None
    panel = os.path.join("data", "spine", "price_panel.parquet")
    if os.path.exists(panel):
        sessions = pd.DatetimeIndex(sorted(
            pd.read_parquet(panel, columns=["date"])["date"].unique()))

    print("=" * 78)
    print(" WHO HOLDS IT — KSEI holding composition (descriptive)")
    print("=" * 78)
    for line in h64_caveat():
        print(" " + line)
    print(" A type is a class of holder, not an actor (CLAUDE.md §6.1).\n")
    if a.market:
        for line in market_report(F, C):
            print(" " + line)
        print()
    tickers = [t.upper() for t in a.tickers] or ([] if a.market
                                                 else open_book_tickers())
    if not tickers and not a.market:
        print(" no tickers given and the open book is empty")
    for t in tickers:
        for line in ticker_report(t, C, F, created, sessions):
            print(" " + line)
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
