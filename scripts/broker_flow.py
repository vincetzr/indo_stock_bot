#!/usr/bin/env python3
"""Accurate IDX broker flow from a free source, with the uncertainty attached.

THE BRIEF THIS ANSWERS
----------------------
Get detailed broker summary and running trade for IDX, as accurately as
possible, for nothing. After auditing every route, the answer splits in two and
both halves are worth stating plainly.

RUNNING TRADE, TRADE BY TRADE, IS NOT AVAILABLE FREE. Every print carrying a
buyer AND a seller member code exists in exactly two places: IDX's own ITCH feed
(a signed licence, a limited company, a leased line from an authorised network
provider, and a security deposit of 300% of a Rp 17.9-44m monthly fee) and
licensed redistributors like LSEG, which publish one RIC per broker per
instrument at terminal prices. Every open-source project that appears to have it
turns out to be replaying a Stockbit session token, which is both a credential
this repo will not reuse and a breach of that platform's terms. There is no
clever request left to find - the platforms are not making one either.

END-OF-DAY BROKER SUMMARY IS AVAILABLE FREE, AND MORE COMPLETELY THAN THIS REPO
THOUGHT. IndoPremier, an IDX member, renders the rekap broker on a public
unauthenticated page. Three things about it were being left on the table:

  1. THE FOOTER PUBLISHES THE DAY'S TOTALS. It was parsed and thrown away,
     because ``fetch_day`` never called ``parse_totals``. With the totals the
     top-10 table stops being a biased sample of unknown size and becomes a
     censored sample of known size, which can be bounded instead of guessed.
  2. TOTAL LOTS CAN BE RECOVERED EXACTLY. The printed ``T.Lot`` goes through the
     same abbreviator as everything else - BBCA on a busy day reads ``1.1 M`` -
     but ``T.Val`` carries four figures and ``Avg`` is exact, so value / (100 x
     average) recovers the count to about 0.03%.
  3. THE SAME MODULE SPLITS BY INVESTOR TYPE. An undocumented ``fd`` parameter
     returns the foreign-only and domestic-only rekap, and the two partition the
     whole EXACTLY - verified across 37 broker-sides on three tickers with a
     maximum difference of zero lots. Result 121 of this repo states that the
     investor-type flag "does not exist in a broker summary"; that is wrong and
     is corrected here.

WHAT THE THREE VIEWS BUY
------------------------
    wider    each view ranks its own top ten and they are not the same ten; on
             BBCA the combined view lists 14 brokers and the three together 20
    deeper   the foreign view came back 99-100% covered, so the most-watched
             flow in this market is close to exactly measured, not bounded
    exact    two known sides give the third by subtraction, turning censored
             quantities into exact ones

WHAT IS REPORTED, AND WHAT IS REFUSED
-------------------------------------
Every number carries a bracket, and a broker whose bracket spans zero is
reported as undetermined rather than given a confident sign. That is the whole
difference between this and the naive reading, which treated an unlisted side as
a zero and drove BBCA's market-wide net - which must be exactly zero - to -2.8
million lots.

    python3 scripts/broker_flow.py BBCA --days 20
    python3 scripts/broker_flow.py BBCA --days 20 --single-view
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))
sys.path.insert(0, os.path.dirname(__file__))

from idxbot.config import load_config                     # noqa: E402
from idxbot.data.cache import Cache                       # noqa: E402
from idxbot.data.ipot import IpotBrokerSummary            # noqa: E402
from idxbot.broker_bounds import (                        # noqa: E402
    certain_sign, cumulative_bounds, day_bounds, foreign_net_agreement,
    merge_views, midpoint, naive_error, relative_width, settled_flow_share,
    settled_fraction, visibility, zero_sum_residual)

VIEWS = ("all", "F", "D")


def fetch_views(ticker: str, days: Sequence[pd.Timestamp], cache,
                board: str = "RG", delay: float = 1.1, single: bool = False
                ) -> Tuple[List[pd.DataFrame], List[Dict]]:
    """One frame per session, merged across investor-type views where asked.

    Three requests a session instead of one is a real cost to somebody else's
    server, so it is paced and cached permanently, and ``--single-view`` exists
    for when the extra precision is not needed.
    """
    providers = {v: IpotBrokerSummary(cache=cache, board=board,
                                      session_type=v, delay=delay)
                 for v in (VIEWS if not single else ("all",))}
    frames, stats = [], []
    for d in days:
        got = {}
        for v, p in providers.items():
            try:
                f = p.fetch_day(ticker, d)
            except Exception:                              # noqa: BLE001
                f = None
            if f is not None and not f.empty:
                got[v] = f
        if "all" not in got:
            continue
        vis = visibility(got["all"])
        fv = pd.to_numeric(pd.Series(got["all"].get("foreign_net_val")),
                           errors="coerce").dropna()
        vis["fnval"] = float(fv.iloc[0]) if len(fv) else None
        vis["date"] = d
        vis["views"] = len(got)
        vis["residual"] = zero_sum_residual(got["all"])
        # An INDEPENDENT check, not another way of restating the same rows: the
        # footer's own net-foreign figure is computed upstream, so reproducing
        # it from the foreign view validates the whole reading end to end.
        if "F" in got:
            agree = foreign_net_agreement(got["F"], vis.get("fnval"))
            if "relative_error" in agree:
                vis["fnval_error"] = agree["relative_error"]
        stats.append(vis)
        if len(got) == 3:
            frames.append(merge_views(got["all"], got["F"], got["D"]))
        else:
            frames.append(day_bounds(got["all"]))
    return frames, stats


def report(ticker: str, frames: List[pd.DataFrame], stats: List[Dict],
           top: int = 15) -> None:
    if not frames:
        print(" nothing fetched.")
        return
    S = pd.DataFrame(stats)
    print(f"\n{'=' * 92}\n WHAT THE SOURCE ACTUALLY SHOWED\n{'=' * 92}")
    print(f" {len(frames)} sessions, {S['date'].min():%Y-%m-%d} to "
          f"{S['date'].max():%Y-%m-%d}, {int(S['views'].median())} view(s) a session")
    print(f" visible share of volume   buy {S['cover_buy'].median():.1%} median, "
          f"{S['cover_buy'].min():.1%} worst"
          f"   sell {S['cover_sell'].median():.1%} / {S['cover_sell'].min():.1%}")
    print(f" brokers named per session {int(S['brokers_listed'].median())} median, "
          f"{int(S['brokers_listed'].max())} best")
    worst = float(np.nanmax(np.abs(S["residual"]))) if len(S) else float("nan")
    print(f" zero-sum check            worst residual {worst:,.0f} lots "
          f"(must be 0; this tests the parse, not the data)")
    if "fnval_error" in S and S["fnval_error"].notna().any():
        e = S["fnval_error"].dropna()
        print(f" net-foreign cross-check   the foreign view reproduces the "
              f"source's own published\n                           F.NVal to "
              f"{e.median():.1%} median, {e.max():.1%} worst — an independent "
              f"confirmation\n                           that these three views "
              f"mean what they appear to mean")

    cum = cumulative_bounds(frames)
    if cum.empty:
        print(" no usable sessions.")
        return
    cum["rel"] = relative_width(cum)
    cum["mid"] = midpoint(cum)
    cum["err"] = naive_error(cum)
    print(f"\n{'=' * 92}\n WHAT CAN BE PROVEN OVER THE WINDOW\n{'=' * 92}")
    print(f" {len(cum)} brokers appeared. Direction settled for "
          f"{settled_fraction(cum):.0%} of them, carrying "
          f"{settled_flow_share(cum):.0%} of the net flow.")
    wrong = cum[cum["err"] > 1.0]
    print(f" The plain reading - unlisted side counted as zero - lands OUTSIDE "
          f"the proven\n bracket for {len(wrong)} of {len(cum)} brokers"
          + (f", by up to {wrong['err'].max():,.0f} lots."
             if len(wrong) else "."))
    print(f"\n {'broker':<8}{'net (lots)':>13}{'lower':>13}{'upper':>13}"
          f"{'+/-':>7}{'plain read':>13}{'off by':>11}{'seen':>6}  verdict")
    order = cum.reindex(cum["mid"].abs().sort_values(ascending=False).index)
    for r in order.head(top).itertuples():
        v = ("NET BUYER" if r.net_lo > 0
             else "NET SELLER" if r.net_hi < 0 else "undetermined")
        rel = f"{r.rel:.0%}" if np.isfinite(r.rel) else "-"
        off = f"{r.err:,.0f}" if r.err > 1.0 else "-"
        print(f" {r.broker:<8}{r.mid:>13,.0f}{r.net_lo:>13,.0f}"
              f"{r.net_hi:>13,.0f}{rel:>7}{r.net_naive:>13,.0f}{off:>11}"
              f"{r.days_seen:>6}  {v}")

    settled = certain_sign(cum)
    if not settled.empty:
        buyers = settled[settled["direction"] == "net buyer"]["broker"].tolist()
        sellers = settled[settled["direction"] == "net seller"]["broker"].tolist()
        print(f"\n proven net buyers : {', '.join(buyers) if buyers else 'none'}")
        print(f" proven net sellers: {', '.join(sellers) if sellers else 'none'}")
    exact = cum[np.isclose(cum["width"], 0.0)]
    if not exact.empty:
        print(f" measured EXACTLY (listed both sides every session): "
              f"{', '.join(exact['broker'])}")
    print("\n Everything else is genuinely undetermined by this source and is "
          "left that way.\n A confident number there would be an invention, "
          "not a measurement.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("ticker")
    ap.add_argument("--days", type=int, default=20)
    ap.add_argument("--end", default=None)
    ap.add_argument("--board", default="RG")
    ap.add_argument("--delay", type=float, default=1.1)
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--single-view", action="store_true",
                    help="combined view only: one request a session instead of "
                         "three, at the cost of wider brackets")
    args = ap.parse_args()

    cfg = load_config()
    cache = Cache(cfg.path("data.cache_dir", "data/cache"))
    end = pd.Timestamp(args.end) if args.end else \
        pd.Timestamp.now(tz="Asia/Jakarta").tz_localize(None).normalize()
    days = pd.bdate_range(end=end, periods=args.days)

    print(f"{'=' * 92}\n IDX BROKER FLOW — {args.ticker.upper()}, "
          f"bounded rather than guessed\n{'=' * 92}")
    print(f" source: IndoPremier public rekap module, board {args.board}, "
          f"{'1 view' if args.single_view else '3 views (all / foreign / domestic)'}")
    print(f" {args.delay:.1f}s between requests, everything cached permanently. "
          f"This is someone else's\n server and IDX restricts redistribution of "
          f"its market data — do not bulk harvest it.")

    frames, stats = fetch_views(args.ticker.upper(), days, cache, args.board,
                                args.delay, args.single_view)
    report(args.ticker.upper(), frames, stats, args.top)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
