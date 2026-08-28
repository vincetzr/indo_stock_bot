#!/usr/bin/env python3
"""H39 — buy when the Hull turns green AND the signal fires; sell on the reverse.

    python3 scripts/hull_trade.py            # the rule as asked, then the grid
    python3 scripts/hull_trade.py --liquid   # only names above Rp1bn/day

THE RULE, STATED EXACTLY, BECAUSE THE CHART VERSION IS AMBIGUOUS.

"When the Hull turns green and there is a buy signal" cannot mean both flips
land on the same bar — that almost never happens. It means the CONJUNCTION of
two states: the Hull is rising AND the signal state is on. So:

    ENTER   the first bar where (hull rising) AND (signal on)
    EXIT    the first bar after that where (hull falling) AND (signal off)

Both are acted on the NEXT bar's close, never the bar that produced them. A rule
filled on its own signal bar is the single most common way a backtest invents
money, and it is worth about 1% a trade on this market.

THREE CONTROLS DECIDE WHETHER THE ANSWER MEANS ANYTHING, and the request only
asks for the first two.

  win rate + average return   what was asked. Necessary, and not sufficient: a
                              rule that is in the market a third of the time
                              will show a smaller loss than buy-and-hold in a
                              falling market and that is not skill.
  BUY-AND-HOLD ON THE SAME NAME OVER THE SAME SPAN   the only comparison that
                              answers "would I have more money".
  A DURATION-MATCHED HOLD     because being out of a market whose per-name
                              yearly log return is negative is most of what any
                              exit rule does. H35 found the best bracket looked
                              good until it was compared to a hold of its own
                              length; this is the same trap.
  THE HALF-SPLIT              the only replication test this repo trusts. A19,
                              A18 and A20 each record a within-sample statistic
                              reading as overwhelming and failing to replicate.

PRE-REGISTERED, WRITTEN BEFORE ANY CELL WAS SCORED
--------------------------------------------------
G1  The rule will show a WIN RATE BELOW 50% and a POSITIVE average return, the
    classic trend-following shape: many small losses, a few large gains.
G2  It will NOT beat buy-and-hold on the same names over the same spans, and
    the best grid cell will not be positive in both halves after cost. H33
    measured every flip in this family as net-negative against holding
    (-0.0124 to -0.0191 of mean log against -0.0140), and reports/hullut_*.csv
    scored the published Hull Suite + UT Bot on 84 IDX names over 240
    configurations: best median excess CAGR -6.1%, beat buy-and-hold on 26% of
    names, lost in all five walk-forward folds. The conjunction of two filters
    is new; the family is not.
G3  Optimising the grid WILL find a configuration that looks good, and it will
    be the maximum of ~24 in-sample cells. That is what the half-split is for.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))
sys.path.insert(0, os.path.dirname(__file__))

from time_price import MIN_BARS, eligible, hma, load             # noqa: E402

OUT = "reports"
COST = 0.0056                 # 0.28 buy + 0.18 sell + 0.10 tax
HULLS = (21, 34, 55, 89)
SIGNALS = ("EMA34", "EMA50", "EMA stack", "hull only")
#  IDX IS LONG-ONLY, SO THE EXIT IS THE WHOLE RISK DECISION and it is worth its
#  own dimension. Waiting for both conditions holds through the first leg down;
#  leaving on the Hull alone takes profit earlier and re-enters more often,
#  paying the toll each time. Which is better is a measurement, not a view.
EXITS = ("both", "hull only", "signal only")


def states(px: pd.Series, hull_len: int, sig: str) -> Tuple[np.ndarray,
                                                            np.ndarray]:
    """(hull rising, signal on) — the two booleans the rule is built from."""
    h = hma(px, hull_len).to_numpy()
    up = np.concatenate([[False], h[1:] > h[:-1]])
    up &= np.isfinite(np.concatenate([[np.nan], h[:-1]]))
    if sig == "hull only":
        on = np.ones(len(px), bool)
    elif sig == "EMA stack":
        e50 = px.ewm(span=50, adjust=False, min_periods=50).mean().to_numpy()
        e100 = px.ewm(span=100, adjust=False, min_periods=100).mean().to_numpy()
        e200 = px.ewm(span=200, adjust=False, min_periods=200).mean().to_numpy()
        p = px.to_numpy()
        on = (p > e50) & (e50 > e100) & (e100 > e200)
    else:
        n = int(sig.replace("EMA", ""))
        e = px.ewm(span=n, adjust=False, min_periods=n).mean().to_numpy()
        on = px.to_numpy() > e
    return up, np.nan_to_num(on, nan=False).astype(bool)


def trades(p: np.ndarray, up: np.ndarray, on: np.ndarray, el: np.ndarray,
           yr: np.ndarray, exit_mode: str = "both") -> List[Dict]:
    """Walk the position as a hysteresis between the two conditions.

    A vectorised forward-fill rather than a Python loop: 1 where both
    conditions hold, 0 where both reverse, carry forward in between. The
    position is then SHIFTED BY ONE so every fill is the bar after the
    condition, which is the whole look-ahead guard.
    """
    enter = up & on
    exit_ = ((~up) & (~on) if exit_mode == "both"
             else ~up if exit_mode == "hull only" else ~on)
    raw = np.where(enter, 1.0, np.where(exit_, 0.0, np.nan))
    pos = pd.Series(raw).ffill().fillna(0.0).to_numpy()
    pos = np.concatenate([[0.0], pos[:-1]])              # act on the next bar
    d = np.diff(np.concatenate([[0.0], pos]))
    ins = np.flatnonzero(d > 0)
    outs = np.flatnonzero(d < 0)
    out: List[Dict] = []
    for i, a in enumerate(ins):
        b = outs[outs > a]
        if not len(b):
            continue                                     # still open at the end
        b = int(b[0])
        if not el[a]:
            continue                                     # could not have bought
        out.append({"i": int(a), "j": b, "year": int(yr[a]),
                    "bars": b - a, "ret": p[b] / p[a] - 1.0 - COST})
    return out


def hold_curve(P: pd.DataFrame) -> pd.DataFrame:
    """Plain buy-and-hold at a range of fixed horizons — the control that
    matters most, because a rule invested a third of the time is compared to a
    hold of the same length, not to a hold of the whole span."""
    rows: List[pd.DataFrame] = []
    for tk, g in P.groupby("ticker", sort=False):
        if len(g) < MIN_BARS:
            continue
        p = g["px"].to_numpy(float)
        n = len(p)
        el = g["elig"].to_numpy()
        for h in (5, 10, 20, 40, 60, 90, 130, 180, 252):
            j = np.arange(n) + h
            ok = (j < n) & el
            if not ok.any():
                continue
            rows.append(pd.DataFrame({"h": h,
                                      "ret": p[j[ok]] / p[ok] - 1.0 - COST}))
    H = pd.concat(rows, ignore_index=True)
    H["lg"] = np.log1p(np.clip(H["ret"], -0.99, None))
    return H.groupby("h").agg(n=("ret", "size"), mean=("ret", "mean"),
                              win=("ret", lambda x: float((x > 0).mean())),
                              meanlog=("lg", "mean")).reset_index()


def run_all(P: pd.DataFrame) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    """Every configuration in one pass over the panel.

    The Hull and the EMAs are recomputed per (name, length) and nothing else,
    so the 48-cell grid costs four moving averages a name rather than
    forty-eight. The alternative took long enough that the grid would have been
    quietly trimmed, which is how a sweep ends up reporting its own convenience.
    """
    acc: Dict[str, List[Dict]] = {}
    camp: Dict[str, List[Dict]] = {}
    for tk, g in P.groupby("ticker", sort=False):
        if len(g) < MIN_BARS:
            continue
        px = g["px"].reset_index(drop=True)
        p = px.to_numpy(float)
        el = g["elig"].to_numpy()
        yr = pd.DatetimeIndex(g["date"]).year.to_numpy()
        cache = {(hl, sg): states(px, hl, sg) for hl in HULLS for sg in SIGNALS}
        for (hl, sg), (up, on) in cache.items():
            for ex in EXITS:
                if sg == "hull only" and ex == "signal only":
                    continue                 # no signal to exit on
                tag = f"hull{hl} + {sg} / exit {ex}"
                tl = trades(p, up, on, el, yr, ex)
                if not tl:
                    continue
                #  THE COMPARISON THAT ACTUALLY ANSWERS "WOULD I HAVE MORE
                #  MONEY". A first version compared each trade to a hold over
                #  its OWN entry and exit bars, which is the same trade minus
                #  the toll and reported a meaningless 0.0%. The real control
                #  is the whole campaign: compound every trade this name
                #  produced, over the span from the first entry to the last
                #  exit, against simply owning it across that span.
                a0, b1 = tl[0]["i"], tl[-1]["j"]
                span = max(1, b1 - a0)
                strat = float(np.prod([1.0 + x["ret"] for x in tl]))
                bh = float(p[b1] / p[a0])
                invested = float(sum(x["bars"] for x in tl))
                for t in tl:
                    t["ticker"] = tk
                    acc.setdefault(tag, []).append(t)
                camp.setdefault(tag, []).append(
                    {"ticker": tk, "span": span, "strat": strat, "hold": bh,
                     "in_mkt": invested / span, "n": len(tl)})
    return ({k: pd.DataFrame(v) for k, v in acc.items()},
            {k: pd.DataFrame(v) for k, v in camp.items()})


def summarise(T: pd.DataFrame, C: pd.DataFrame, HC: pd.DataFrame,
              tag: str) -> Dict:
    if T.empty:
        return {}
    lg = np.log1p(np.clip(T["ret"], -0.99, None))
    cut = int(T["year"].median())
    e, l = T[T["year"] < cut], T[T["year"] >= cut]
    bars = float(T["bars"].mean())
    #  The duration-matched hold, interpolated onto this rule's own mean hold.
    m_hold = float(np.interp(bars, HC["h"], HC["meanlog"]))
    a_hold = float(np.interp(bars, HC["h"], HC["mean"]))
    w_hold = float(np.interp(bars, HC["h"], HC["win"]))
    return {
        "config": tag, "trades": len(T), "names": T["ticker"].nunique(),
        "win": float((T["ret"] > 0).mean()),
        "mean": float(T["ret"].mean()), "median": float(T["ret"].median()),
        "meanlog": float(lg.mean()), "bars": bars,
        "ann": float(lg.mean()) * 252.0 / bars,
        "hold_win": w_hold, "hold_mean": a_hold, "hold_meanlog": m_hold,
        #  Per name, over the span the rule was actually active: what the
        #  compounded strategy did against owning the name across that span,
        #  annualised so names of different lengths are comparable.
        "in_mkt": float(C["in_mkt"].mean()),
        "cagr": float(np.median(C["strat"] ** (252.0 / C["span"]) - 1.0)),
        "cagr_hold": float(np.median(C["hold"] ** (252.0 / C["span"]) - 1.0)),
        "beats_hold": float((C["strat"] > C["hold"]).mean()),
        "edge": float(lg.mean()) - m_hold,
        "early": float(np.log1p(np.clip(e["ret"], -0.99, None)).mean()),
        "late": float(np.log1p(np.clip(l["ret"], -0.99, None)).mean()),
        "edge_early": float(np.log1p(np.clip(e["ret"], -0.99, None)).mean())
        - m_hold,
        "edge_late": float(np.log1p(np.clip(l["ret"], -0.99, None)).mean())
        - m_hold}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--liquid", action="store_true")
    a = ap.parse_args()
    P = load()
    P["elig"] = eligible(P) if a.liquid else P["tradeable"].astype(bool)
    P["px"] = P["adj_close"]
    P = P.sort_values(["ticker", "date"])
    print(f"panel {len(P):,} rows  {P['ticker'].nunique()} names  "
          f"{'liquid only' if a.liquid else 'every tradeable name'}\n")

    HC = hold_curve(P)
    print("the duration-matched control: plain buy-and-hold, net of cost")
    print(HC.to_string(index=False, float_format=lambda v: f"{v:,.4f}"))

    allT, allC = run_all(P)
    rows: List[Dict] = []
    detail = allT.get("hull55 + EMA stack / exit both")
    for tag, T in allT.items():
        s = summarise(T, allC[tag], HC, tag)
        if s:
            rows.append(s)
    R = pd.DataFrame(rows).sort_values("ann", ascending=False)
    R.to_csv(os.path.join(OUT, "hull_trade.csv"), index=False)

    print("\n=== H39 the rule as asked: hull 55 + the EMA stack signal")
    r = R[R["config"] == "hull55 + EMA stack / exit both"].iloc[0]
    print(f"  {int(r['trades']):,} round trips over {int(r['names'])} names, "
          f"mean hold {r['bars']:.0f} sessions")
    print(f"  WIN RATE          {r['win']:.1%}      "
          f"(buy-and-hold over the same length: {r['hold_win']:.1%})")
    print(f"  AVERAGE RETURN    {r['mean']:+.2%}     "
          f"(buy-and-hold over the same length: {r['hold_mean']:+.2%})")
    print(f"  median            {r['median']:+.2%}")
    print(f"  time in the market  {r['in_mkt']:.1%}")
    print(f"  PER NAME, compounded over the span the rule was active:")
    print(f"    strategy median CAGR {r['cagr']:+.2%}  vs owning it "
          f"{r['cagr_hold']:+.2%}   beat it on {r['beats_hold']:.1%} of names")
    print(f"  mean log {r['meanlog']:+.4f} against a matched hold's "
          f"{r['hold_meanlog']:+.4f}   edge {r['edge']:+.4f}")
    print(f"  annualised {r['ann']:+.2%}   "
          f"early {r['edge_early']:+.4f}  late {r['edge_late']:+.4f}")

    print(f"\n=== the grid, {len(R)} configurations, sorted by annualised.")
    print("    'vs hold' is the edge over a plain hold of the SAME mean length;")
    print("    'both+' means that edge is positive in BOTH halves of the sample.")
    print(f"{'config':<34}{'trades':>8}{'win':>7}{'mean':>8}{'median':>8}"
          f"{'bars':>6}{'meanlog':>9}{'vs hold':>9}{'in mkt':>8}"
          f"{'CAGR':>8}{'hold':>8}{'beat%':>7}{'e.early':>9}{'e.late':>8}"
          f"{'both+':>7}")
    for _, x in R.iterrows():
        both = "YES" if x["edge_early"] > 0 and x["edge_late"] > 0 else ""
        print(f"{x['config']:<34}{int(x['trades']):>8,}{x['win']:>7.1%}"
              f"{x['mean']:>8.2%}{x['median']:>8.2%}{x['bars']:>6.0f}"
              f"{x['meanlog']:>9.4f}{x['edge']:>9.4f}{x['in_mkt']:>8.1%}"
              f"{x['cagr']:>8.2%}{x['cagr_hold']:>8.2%}"
              f"{x['beats_hold']:>7.1%}{x['edge_early']:>9.4f}"
              f"{x['edge_late']:>8.4f}{both:>7}")

    if detail is not None and not detail.empty:
        d = detail.copy()
        d["lg"] = np.log1p(np.clip(d["ret"], -0.99, None))
        print("\n=== where the money is: the return distribution of the rule")
        q = d["ret"].quantile([0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
        print("  " + "  ".join(f"p{int(k * 100)} {v:+.1%}"
                               for k, v in q.items()))
        top = d.nlargest(max(1, len(d) // 100), "ret")["ret"].sum()
        print(f"  the best 1% of trades contribute {top / d['ret'].sum():.0%} "
              f"of the total return" if d["ret"].sum() > 0 else
              f"  total return is negative; the best 1% add {top:+.1f}")
        print("\n=== by entry year")
        y = d.groupby("year").agg(n=("ret", "size"), win=("ret", lambda x:
                                  float((x > 0).mean())),
                                  mean=("ret", "mean"), lg=("lg", "mean"))
        print(y.to_string(float_format=lambda v: f"{v:,.4f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
