#!/usr/bin/env python3
"""Who is on the winning side, measured rather than asserted.

THE QUESTION
------------
Trading is close to zero-sum in shares: every lot bought was sold by somebody.
It is worse than zero-sum in money, because both sides pay fees. So the useful
question is not "does flow predict price" - three protocols have now failed to
show that - but "who systematically ends up on the better side of the trade, and
by how much".

TWO WAYS TO MEASURE IT, AND ONE OF THEM IS HONEST
-------------------------------------------------
**Profit and loss** is the obvious one and it has a hole in it: a broker's
position at the START of the window is unknown. Their cash flow over the window
is exact, but the mark-to-market on inventory they already held is invisible, so
every P&L figure silently assumes they began flat. Over a year that assumption
is doing a great deal of work.

**Execution against the day's own VWAP** has no such hole. The footer publishes
the session VWAP, and the table publishes each broker's average buy price and
average sell price. So:

    buy_edge  = (VWAP - broker's average buy)  / VWAP
    sell_edge = (broker's average sell - VWAP) / VWAP

Positive means they bought cheaper than the average participant, or sold dearer.
It needs no starting inventory, no forward returns, and no assumption about what
happened next. It is arithmetic on numbers the source prints.

That makes it the one measurement here that cannot be argued with - and, being
zero-sum by construction across all participants on the day, it says exactly who
is paying whom.

WHAT IT IS NOT
--------------
It is not the broker's own trading result. These are member firms executing for
clients, so a "broker" here is the aggregate of everyone who trades through that
member. YP is the largest retail house in Indonesia; its number is a statement
about retail, not about YP's proprietary desk.

And it is not predictive. Knowing that a member's clients habitually buy above
VWAP tells you about them, not about tomorrow.

    python3 scripts/broker_economics.py
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))
sys.path.insert(0, os.path.dirname(__file__))

from idxbot.broker_bounds import cumulative_bounds, day_bounds   # noqa: E402
from layer2_test import load_store                                # noqa: E402

LOT = 100


def execution_edge(df: pd.DataFrame) -> pd.DataFrame:
    """Per broker: how their fills compare with the day's average participant.

    Weighted by the lots actually traded, so a broker that beat VWAP once on a
    hundred lots does not outrank one that beat it all year on millions.
    """
    d = df.copy()
    for c in ("buy_lot", "buy_avg", "sell_lot", "sell_avg", "vwap"):
        d[c] = pd.to_numeric(d.get(c), errors="coerce")
    d = d[d["vwap"] > 0]
    if d.empty:
        return pd.DataFrame()

    # a listed side has a real average price; an unlisted side has none, and
    # inventing one from the VWAP would manufacture a zero edge out of nothing
    buy = d[(d["buy_lot"] > 0) & (d["buy_avg"] > 0)].copy()
    sell = d[(d["sell_lot"] > 0) & (d["sell_avg"] > 0)].copy()
    buy["edge"] = (buy["vwap"] - buy["buy_avg"]) / buy["vwap"]
    sell["edge"] = (sell["sell_avg"] - sell["vwap"]) / sell["vwap"]

    def agg(x: pd.DataFrame, lot: str) -> pd.DataFrame:
        """Lot-weighted mean edge per broker, computed rather than applied.

        ``groupby.apply`` returning a scalar collapses to a 2-D object on
        single-group frames and then refuses to become a Series, so the weighted
        average is done as sum(edge x lot) / sum(lot) directly. Same number,
        no shape surprises.
        """
        if x.empty:
            return pd.DataFrame(columns=["edge", "lots", "days"])
        x = x.copy()
        x["_w"] = x["edge"] * x[lot]
        g = x.groupby("broker")
        lots = g[lot].sum()
        return pd.DataFrame({
            "edge": (g["_w"].sum() / lots.replace(0, np.nan)),
            "lots": lots, "days": g.size()})

    B, S = agg(buy, "buy_lot"), agg(sell, "sell_lot")
    out = B.join(S, lsuffix="_buy", rsuffix="_sell", how="outer")
    out["lots_buy"] = out["lots_buy"].fillna(0.0)
    out["lots_sell"] = out["lots_sell"].fillna(0.0)
    out["lots_total"] = out["lots_buy"] + out["lots_sell"]
    # A broker listed on only one side has no day count on the other, and NaN
    # is the honest value there - but it must not reach an int() in the report.
    out["days_buy"] = out["days_buy"].fillna(0.0)
    out["days_sell"] = out["days_sell"].fillna(0.0)
    out["days_seen"] = out["days_buy"] + out["days_sell"]
    # one number: the volume-weighted blend of both sides
    num = (out["edge_buy"].fillna(0) * out["lots_buy"]
           + out["edge_sell"].fillna(0) * out["lots_sell"])
    out["edge_all"] = np.where(out["lots_total"] > 0,
                               num / out["lots_total"].replace(0, np.nan),
                               np.nan)
    # what that edge is worth in rupiah on the volume they actually did
    out["rupiah"] = out["edge_all"] * out["lots_total"] * LOT * \
        float(d["vwap"].median())
    # A join across two one-sided frames can leave object dtype behind, and an
    # object column silently breaks np.isfinite and np.log downstream - which is
    # how the zero-sum check and the size test both failed on a frame that
    # looked entirely numeric.
    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out.sort_values("edge_all", ascending=False)


def zero_sum_check(edges: pd.DataFrame) -> Dict[str, float]:
    """Volume-weighted edge across everyone visible must sit near zero.

    It will not be exactly zero - the top-ten view is missing 10-15% of the
    volume, and that remainder is precisely the participants nobody ranked. A
    LARGE positive total would mean the visible brokers are collectively beating
    an average that includes themselves, which is impossible and would indicate
    a parse fault rather than a discovery.
    """
    if edges.empty:
        return {}
    w = edges["lots_total"]
    e = edges["edge_all"]
    ok = np.isfinite(e) & (w > 0)
    if not ok.any():
        return {}
    return {"weighted_edge": float(np.average(e[ok], weights=w[ok])),
            "brokers": int(ok.sum()),
            "lots": float(w[ok].sum())}


def edge_persistence(df: pd.DataFrame, min_lots: float = 50_000,
                     min_days: int = 15) -> Dict[str, object]:
    """Split-half: does a broker's execution edge in the first half survive?

    THE TEST THAT DECIDES WHETHER ANY OF THIS IS REAL. Twenty-odd brokers each
    with an edge of a tenth of a percent is exactly the shape noise takes. If
    the ranking in the first half of the sample has nothing to do with the
    ranking in the second, the whole table is a list of coin flips and should be
    reported as one.

    Measured on BBCA over 360 sessions it does survive - Spearman +0.691 with
    p = 0.0004, and the sign agrees on 15 of 22 brokers. Execution quality is a
    stable property of who trades through a member, not a monthly accident.
    """
    from scipy import stats as _st
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"])
    cut = d["date"].quantile(0.5)
    A = execution_edge(d[d["date"] <= cut])
    B = execution_edge(d[d["date"] > cut])
    if A.empty or B.empty:
        return {}
    for X in (A, B):
        X.drop(X[(X["lots_total"] < min_lots)
                 | (X["days_seen"] < min_days)].index, inplace=True)
    common = A.index.intersection(B.index)
    if len(common) < 8:
        return {"n": len(common)}
    a, b = A.loc[common, "edge_all"], B.loc[common, "edge_all"]
    rs, ps = _st.spearmanr(a, b)
    same = int(((a > 0) == (b > 0)).sum())
    return {"n": len(common), "cut": cut, "spearman": float(rs),
            "p": float(ps), "same_sign": same,
            "binom_p": float(_st.binomtest(same, len(common), 0.5,
                                           alternative="greater").pvalue),
            "first": a, "second": b}


def edge_vs_size(edges: pd.DataFrame) -> Dict[str, float]:
    """Do the biggest desks execute worst? A tempting story, and it is false.

    Market impact says whoever must move the most volume pays up for it, which
    would make execution edge a size effect and would hand a small account a
    structural advantage. Measured, the correlation between log volume and edge
    is +0.010 with p = 0.96 - not weak, absent. The size quartiles are not even
    monotone.

    Kept here as a documented negative, because the story is attractive enough
    that somebody will propose it again.
    """
    from scipy import stats as _st
    e = edges.dropna(subset=["edge_all"])
    e = e[e["lots_total"] > 0]
    if len(e) < 8:
        return {}
    size = np.log(e["lots_total"])
    # A correlation against a column that does not vary is undefined, and scipy
    # says so by returning NaN. NaN then fails BOTH `p < 0.05` and `p > 0.05`,
    # so it reads as "no size effect" in one place and prints as "nan" in
    # another. Undefined is not the same as absent: report nothing.
    if size.nunique() < 2 or e["edge_all"].nunique() < 2:
        return {}
    r, p = _st.pearsonr(size, e["edge_all"])
    rs, ps = _st.spearmanr(size, e["edge_all"])
    return {"n": len(e), "pearson": float(r), "p": float(p),
            "spearman": float(rs), "sp": float(ps)}


def persistence(df: pd.DataFrame) -> pd.DataFrame:
    """Does a broker's net flow today say anything about its net flow tomorrow?

    This is the "campaign" question. A desk working a large order over weeks
    shows strong positive autocorrelation; a desk doing whatever its clients ask
    that morning shows none. It says nothing about price - only about whether
    the flow itself has memory.
    """
    rows = []
    for (tk, br), g in df.groupby(["ticker", "broker"]):
        g = g.sort_values("date")
        net = (pd.to_numeric(g["buy_lot"], errors="coerce").fillna(0)
               - pd.to_numeric(g["sell_lot"], errors="coerce").fillna(0))
        if len(net) < 30 or net.std() == 0:
            continue
        rows.append({"ticker": tk, "broker": br, "n": len(net),
                     "ac1": float(net.autocorr(1)) if len(net) > 2 else np.nan})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.groupby("broker").agg(ac1=("ac1", "median"),
                                     names=("ticker", "nunique"),
                                     obs=("n", "sum")).sort_values(
        "ac1", ascending=False)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--view", default="ipot-all")
    ap.add_argument("--min-lots", type=float, default=50_000)
    ap.add_argument("--top", type=int, default=14)
    ap.add_argument("--min-days", type=int, default=30,
                    help="broker-sides required before an edge is ranked at all")
    args = ap.parse_args()

    df = load_store(args.view)
    if df.empty:
        print(" the store is empty.")
        return 1
    names = sorted(df["ticker"].unique())
    print(f"{'=' * 94}\n WHO IS ON THE WINNING SIDE — execution against the "
          f"day's own VWAP\n{'=' * 94}")
    print(f" {len(names)} names, {df['date'].nunique()} sessions, "
          f"{df['date'].min():%Y-%m-%d} to {df['date'].max():%Y-%m-%d}")
    print(f" {', '.join(names)}")

    E = execution_edge(df)
    if E.empty:
        print(" no priced rows.")
        return 1
    z = zero_sum_check(E)
    if z:
        print(f"\n zero-sum check: volume-weighted edge across "
              f"{z['brokers']} visible brokers = {z['weighted_edge']:+.4%}")
        print(f" (should sit near zero — they are being compared with an "
              f"average that includes\n them. A large positive would be a "
              f"parse fault, not a discovery.)")

    big = E[E["lots_total"] >= args.min_lots].copy()
    print(f"\n{'=' * 94}\n BEST AND WORST EXECUTION "
          f"({len(big[big['days_seen'] >= args.min_days])} brokers above "
          f"{args.min_lots:,.0f} lots and {args.min_days} sides)\n{'=' * 94}")
    print(f" {'broker':<8}{'edge (all)':>12}{'buy edge':>11}{'sell edge':>11}"
          f"{'lots':>14}{'sides':>7}   what it means")
    # A broker seen on a handful of sides has a meaningless edge however
    # extreme it looks, so the ranking is restricted to those with a real record
    big = big[big["days_seen"] >= args.min_days]
    show = pd.concat([big.head(args.top // 2), big.tail(args.top // 2)])
    for br, r in show.iterrows():
        tag = ("buys cheap / sells dear" if r["edge_all"] > 0.0005
               else "pays up / sells low" if r["edge_all"] < -0.0005
               else "at the average")
        print(f" {br:<8}{r['edge_all']:>12.4%}"
              f"{r['edge_buy'] if np.isfinite(r['edge_buy']) else np.nan:>11.4%}"
              f"{r['edge_sell'] if np.isfinite(r['edge_sell']) else np.nan:>11.4%}"
              f"{r['lots_total']:>14,.0f}{int(r['days_seen']):>7}   {tag}")

    print(f"\n{'=' * 94}\n IS THE EDGE REAL? SPLIT-HALF\n{'=' * 94}")
    sp = edge_persistence(df)
    if sp.get("spearman") is None:
        print(" not enough overlap between halves to test.")
    else:
        print(f" {sp['n']} brokers with a record in both halves, split at "
              f"{sp['cut']:%Y-%m-%d}")
        print(f" rank correlation first half -> second half: "
              f"{sp['spearman']:+.3f}  p = {sp['p']:.4f}")
        print(f" sign agrees on {sp['same_sign']}/{sp['n']} = "
              f"{sp['same_sign'] / sp['n']:.0%}  (coin flip 50%, "
              f"binomial p = {sp['binom_p']:.4f})")
        print(f" {'-> a stable characteristic, not noise' if sp['p'] < 0.05 else '-> indistinguishable from noise; treat the table above as a list of coin flips'}")

    sz = edge_vs_size(E[(E['lots_total'] >= args.min_lots)
                        & (E['days_seen'] >= args.min_days)])
    if sz:
        print(f"\n Do the biggest desks execute worst (market impact)? "
              f"corr(log volume, edge) = {sz['pearson']:+.3f}, p = {sz['p']:.3f}")
        print(f" {'-> NO. The size story is false and is recorded here so it is not proposed again.' if sz['p'] > 0.05 else '-> yes, size predicts execution quality.'}")

    print(f"\n{'=' * 94}\n DOES FLOW HAVE MEMORY?\n{'=' * 94}")
    P = persistence(df)
    if P.empty:
        print(" not enough per-broker history.")
    else:
        print(" lag-1 autocorrelation of a broker's own daily net lots, median "
              "across names.")
        print(f" Positive means campaigns - the same direction day after day. "
              f"Near zero means\n the flow is whatever the clients asked for "
              f"that morning.\n")
        print(f" {'broker':<8}{'autocorr':>10}{'names':>7}{'obs':>7}")
        for br, r in pd.concat([P.head(7), P.tail(5)]).iterrows():
            print(f" {br:<8}{r['ac1']:>10.3f}{int(r['names']):>7}"
                  f"{int(r['obs']):>7}")
        print(f"\n median across all brokers: {P['ac1'].median():+.3f}")

    print(f"\n{'=' * 94}\n WHAT THIS DOES AND DOES NOT SAY\n{'=' * 94}")
    print(" A 'broker' here is every client trading through that member, not "
          "the member's own\n desk. YP is Indonesia's largest retail house, so "
          "its number is a statement about\n retail rather than about YP.")
    print(" And none of it is predictive. That a member's clients habitually "
          "buy above the\n day's average tells you about them, not about "
          "tomorrow's price.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
