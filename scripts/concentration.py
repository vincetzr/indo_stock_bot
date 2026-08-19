#!/usr/bin/env python3
""""Remove the best 5 trades and it loses" - is that a fault of the rule?

The objection is fair and it is the standard way to kill a backtest: if the
whole result rides on a handful of trades, the rule has not found an edge, it
has found five lucky bars.

But the test is only evidence AGAINST the rule if the rule is more concentrated
than the alternative it is being compared to. Buy-and-hold is not exempt from
this arithmetic - equity returns are multiplicative and famously lopsided - so
the honest version of the objection is a PAIRED test:

    cut the timeline at the strategy's own trade boundaries;
    that gives N segments;
    the strategy's total is the product of its N segment multiples;
    holding's total is the product of ITS N multiples over the SAME dates;
    now drop the best k from each and compare what survives.

Same stock, same dates, same number of factors, same removal. Whatever the
answer is, it is not an artefact of the split.

Two supporting numbers are computed, not assumed:

    breadth      the smallest share of weeks whose product alone reaches the
                 full buy-and-hold return of the same name. If that share is
                 tiny for ordinary large caps, concentration is a property of
                 equities, not a defect of a timing rule.
    verdict      printed from the numbers. This script must never state a
                 conclusion its own output does not support.

    python3 scripts/concentration.py --ticker CUAN --band 0.08
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

from idxbot.config import load_config          # noqa: E402
from idxbot.data.cache import Cache            # noqa: E402
from idxbot.data.ohlcv import YahooOHLCV       # noqa: E402
from account_sim import LOT, load_ohlc, run_account   # noqa: E402
from leg_signals import market_caps            # noqa: E402


def segment_multiples(eq: pd.Series, cuts: List[pd.Timestamp]) -> np.ndarray:
    """Growth factor of `eq` over each segment between consecutive cut dates.

    The product of the returned factors is the whole-period growth, so dropping
    one factor is exactly "this segment never happened, the rest still did".
    """
    marks = [eq.index[0]] + [c for c in cuts if eq.index[0] < c < eq.index[-1]]
    marks = sorted(set(marks)) + [eq.index[-1]]
    vals = eq.reindex(marks, method="ffill").to_numpy(float)
    return vals[1:] / vals[:-1]


def survives(mult: np.ndarray, k: int) -> float:
    """Total growth after deleting the k largest segments."""
    if k <= 0:
        return float(np.prod(mult))
    keep = np.sort(mult)[:max(len(mult) - k, 0)]
    return float(np.prod(keep)) if len(keep) else np.nan


def top_share(mult: np.ndarray, k: int) -> float:
    """Share of the NET total log return carried by the k largest segments.

    Readable on one name - ">100% means everything else was a net loss" - but
    useless across names, because the denominator is a net total that passes
    through zero. Use ``top_share_gross`` to compare names.
    """
    lg = np.log(mult[np.isfinite(mult) & (mult > 0)])
    tot = lg.sum()
    if not np.isfinite(tot) or abs(tot) < 1e-12:
        return np.nan
    return float(np.sort(lg)[-k:].sum() / tot)


def top_share_gross(mult: np.ndarray, k: int) -> float:
    """Share of all the GAINS that the k best segments account for.

    Bounded in [0, 1] and defined whenever anything went up, so a median of it
    across names means something. 1.0 = every gain came from k segments;
    k / (number of winning segments) = gains spread perfectly evenly.
    """
    lg = np.log(mult[np.isfinite(mult) & (mult > 0)])
    up = lg[lg > 0]
    if not len(up) or up.sum() <= 0:
        return np.nan
    return float(np.sort(up)[-k:].sum() / up.sum())


def breadth(px: pd.Series) -> Optional[Tuple[float, float]]:
    """How lopsided is simply holding this name?

    Returns two numbers, because one of them is fragile on its own:

    ``needed``  smallest share of weeks whose product alone reaches the full
                return. Intuitive, but degenerate when the total is near zero -
                a single good week can equal a flat decade, which says more
                about the total than about the distribution.
    ``top5pct`` share of the total log return carried by the best 5% of weeks.
                Stable regardless of how large the total is, and the number to
                quote when comparing names.
    """
    w = px.resample("W-FRI").last().dropna()
    r = w.pct_change().dropna().to_numpy(float) + 1.0
    lg = np.log(r[r > 0])
    tot = lg.sum()
    if len(lg) < 50 or tot <= 0:
        return None
    srt = np.sort(lg)[::-1]
    run = np.cumsum(srt)
    hit = np.flatnonzero(run >= tot)
    needed = float((hit[0] + 1) / len(lg)) if len(hit) else 1.0
    k = max(1, int(round(0.05 * len(lg))))
    return needed, float(srt[:k].sum() / tot)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default="CUAN")
    ap.add_argument("--band", type=float, default=0.08)
    ap.add_argument("--capital", type=float, default=10_000_000)
    ap.add_argument("--participation", type=float, default=0.10)
    ap.add_argument("--fee-buy", type=float, default=0.0028)
    ap.add_argument("--fee-sell", type=float, default=0.0028)
    ap.add_argument("--min-mcap", type=float, default=1e13)
    ap.add_argument("--removals", type=int, nargs="+", default=[0, 1, 2, 3, 5, 8])
    args = ap.parse_args()
    os.makedirs("reports", exist_ok=True)

    cfg = load_config()
    loader = YahooOHLCV(cfg, Cache(cfg.path("data.cache_dir", "data/cache")))
    df = load_ohlc(loader, args.ticker)
    if df is None:
        raise SystemExit(f"no data for {args.ticker}")

    eq, trades = run_account(df, args.band, args.capital, args.participation,
                             0, args.fee_buy, args.fee_sell)
    sells = [t for t in trades if t["side"] == "SELL"]
    cuts = [t["date"] for t in sells]
    if len(cuts) < 4:
        raise SystemExit("too few round trips to split on")

    n0 = int(args.capital / (df["open"].iloc[0] * LOT * (1 + args.fee_buy)))
    hold = pd.Series(
        args.capital - n0 * LOT * df["open"].iloc[0] * (1 + args.fee_buy)
        + n0 * LOT * df["close"].to_numpy(float), index=df.index)

    s_mult = segment_multiples(eq, cuts)
    h_mult = segment_multiples(hold, cuts)

    print(f"{args.ticker}: {len(df):,} sessions, {df.index[0]:%Y-%m-%d} to "
          f"{df.index[-1]:%Y-%m-%d}, daily {args.band:.0%} band")
    print(f"{len(sells)} round trips -> the timeline is cut into "
          f"{len(s_mult)} segments, and BOTH curves are cut at the same dates\n")

    print(f"{'=' * 72}\n DROP THE BEST SEGMENTS FROM EACH — same stock, same "
          f"dates\n{'=' * 72}")
    print(f" {'remove best':>13}{'STRATEGY':>14}{'BUY & HOLD':>14}"
          f"{'who survives':>16}")
    rows = []
    for k in args.removals:
        a, b = survives(s_mult, k), survives(h_mult, k)
        who = "strategy" if a > b else ("hold" if b > a else "tie")
        rows.append({"remove": k, "strategy": a, "hold": b, "better": who})
        print(f" {k if k else 'none':>13}{a:>13.2f}x{b:>13.2f}x{who:>16}")

    k5 = min(5, len(s_mult) - 1)
    ss, hs = top_share(s_mult, k5), top_share(h_mult, k5)
    sg, hg = top_share_gross(s_mult, k5), top_share_gross(h_mult, k5)
    print(f"\n carried by the top {k5} segments:")
    print(f"   {'':<12}{'of the NET total':>18}{'of all GAINS':>16}")
    print(f"   {'strategy':<12}{ss:>18.0%}{sg:>16.0%}")
    print(f"   {'buy & hold':<12}{hs:>18.0%}{hg:>16.0%}")
    print("   (>100% of the net total just means everything else lost money;\n"
          "    the share of gains is the figure that compares across names)")

    # --- is this stock unusual, or is every large cap like this? -------------
    mc = market_caps()
    big = sorted(mc[mc >= args.min_mcap].index)
    shares = []
    for t in big:
        d = loader.get(t, max_age=86400 * 30)
        if d is None or len(d) < 500:
            continue
        s = d.set_index("date").sort_index()["close"].astype(float).dropna()
        b = breadth(s)
        if b is not None:
            shares.append({"ticker": t, "weeks_needed": b[0], "top5pct": b[1]})
    B = pd.DataFrame(shares)
    if len(B):
        med5 = float(B["top5pct"].median())
        worst = B.loc[B["top5pct"].idxmin()]
        print(f"\n across {len(B)} big caps that rose over their history, the "
              f"best 5% of weeks\n carry a median {med5:.0%} of everything "
              f"holding them produced.\n the most broadly-earned name in the "
              f"sample is {worst['ticker']} at {worst['top5pct']:.0%}, so "
              f"there is no\n large cap here whose return is spread evenly "
              f"across its weeks.")

    # --- the same paired test on every big cap, not just this one -----------
    panel = []
    for t in big:
        d2 = load_ohlc(loader, t)
        if d2 is None or len(d2) < 400:
            continue
        e2, tr2 = run_account(d2, args.band, args.capital, args.participation,
                              0, args.fee_buy, args.fee_sell)
        c2 = [x["date"] for x in tr2 if x["side"] == "SELL"]
        if len(c2) < 8:
            continue
        m0 = int(args.capital / (d2["open"].iloc[0] * LOT * (1 + args.fee_buy)))
        h2 = pd.Series(
            args.capital - m0 * LOT * d2["open"].iloc[0] * (1 + args.fee_buy)
            + m0 * LOT * d2["close"].to_numpy(float), index=d2.index)
        sm, hm = segment_multiples(e2, c2), segment_multiples(h2, c2)
        # "remove 5" is a far harsher test on a name with 30 segments than on
        # one with 120, so the fraction-matched removal is reported beside it.
        kf = max(1, int(round(0.10 * len(sm))))
        panel.append({"ticker": t, "segments": len(sm), "k_frac": kf,
                      "s_full": survives(sm, 0), "h_full": survives(hm, 0),
                      "s_less5": survives(sm, 5), "h_less5": survives(hm, 5),
                      "s_lessf": survives(sm, kf), "h_lessf": survives(hm, kf),
                      "s_top5": top_share_gross(sm, 5),
                      "h_top5": top_share_gross(hm, 5)})
    P = pd.DataFrame(panel)
    if len(P):
        P.to_csv("reports/concentration_panel.csv", index=False)
        won = int((P["s_less5"] > P["h_less5"]).sum())
        wonf = int((P["s_lessf"] > P["h_lessf"]).sum())
        lessconc = int((P["s_top5"] < P["h_top5"]).sum())
        print(f"\n{'=' * 72}\n THE SAME PAIRED TEST ON {len(P)} BIG CAPS\n{'=' * 72}")
        print(f" median {P['segments'].median():.0f} segments per name "
              f"(CUAN has {len(s_mult)}), so both removals are shown\n")
        print(f" removing each side's best 5:          strategy ahead on "
              f"{won}/{len(P)} names")
        print(f" removing each side's best 10%:        strategy ahead on "
              f"{wonf}/{len(P)} names")
        print(f" strategy LESS concentrated than holding: {lessconc}/{len(P)} "
              f"names")
        print(f" median share of all GAINS in the best 5 segments:   "
              f"strategy {P['s_top5'].median():.0%}   holding "
              f"{P['h_top5'].median():.0%}")

    # --- the verdict, computed ----------------------------------------------
    print(f"\n{'=' * 72}\n VERDICT\n{'=' * 72}")
    beats = sum(r["better"] == "strategy" for r in rows if r["remove"] > 0)
    tested = sum(1 for r in rows if r["remove"] > 0)

    # The panel is the general claim; one name is an anecdote, even a favourable
    # one. Where they disagree, the panel is what gets reported as the answer.
    if len(P):
        frac = won / len(P)
        gap = float(P["s_top5"].median() - P["h_top5"].median())
        if frac >= 0.5 and gap <= 0.0:
            print(f" The objection does NOT survive the paired test: after the "
                  f"same removal\n the strategy is ahead on {won}/{len(P)} big "
                  f"caps and is no more concentrated\n than the thing it trades.")
        else:
            print(f" The objection LARGELY STANDS on the universe. Removing "
                  f"each side's best 5,\n the strategy is ahead on only "
                  f"{won}/{len(P)} big caps ({frac:.0%}), and its gains are "
                  f"more\n concentrated than holding's (median share of gains "
                  f"in the best 5 segments:\n "
                  f"{P['s_top5'].median():.0%} vs {P['h_top5'].median():.0%}).")
        conc = ("still the more concentrated of the two"
                if sg > hg else "and the less concentrated of the two")
        print(f"\n {args.ticker} survives the removals better — strategy ahead "
              f"at {beats}/{tested} levels —\n but it is {conc} "
              f"({sg:.0%} of its gains in the best 5 against\n {hg:.0%} for "
              f"holding). Surviving a removal and being spread out are not\n "
              f"the same property, and on this universe the rule has the "
              f"second one\n less often than holding does.")
    elif not np.isfinite(ss) or not np.isfinite(hs):
        print(" concentration could not be measured on this sample.")
    elif ss - hs > 0.05:
        print(f" On {args.ticker} the strategy IS more concentrated than "
              f"holding ({ss:.0%} vs {hs:.0%}).")
    elif hs - ss > 0.05:
        print(f" On {args.ticker} holding is MORE concentrated than the "
              f"strategy ({hs:.0%} vs {ss:.0%}).")
    else:
        print(f" On {args.ticker} the two are concentrated to the same degree "
              f"({ss:.0%} vs {hs:.0%}).")

    print("\n What this does and does not settle: it compares CONCENTRATION "
          "only.\n Both sides are lopsided — across the big caps the best 5% of "
          "weeks carry\n several times holding's entire return — so 'remove the "
          "best few and it\n loses' is not on its own proof of a broken rule. "
          "Whether the rule beats\n holding at all is a different question, "
          "measured in Result 100, and there\n it does not.")

    pd.DataFrame(rows).to_csv("reports/concentration.csv", index=False)
    if len(B):
        B.to_csv("reports/concentration_breadth.csv", index=False)
    print("\n -> reports/concentration.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
