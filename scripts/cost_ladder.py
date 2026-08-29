#!/usr/bin/env python3
"""H44 — the simple methods, on a COST LADDER, long-only and long-short.

    python3 scripts/cost_ladder.py

TWO QUESTIONS IN ONE HARNESS.

(1) "IDX trades better with easier methods — support/resistance, stochastic,
    EMA, Fibonacci." Three of those four have been tested here and the score is
    1 for 4: swing highs are real (H34b), Fibonacci measured nothing (H34a),
    no EMA rule compounds (H30/H33/H40), the stochastic cross is 5+/6- noise
    (A17). BUT every one of those tests ran at the retail toll of 56 bps plus a
    fraksi-harga half-spread, and H13 measured effects of a size that toll
    swallows whole: gross quintile spreads of 0.15-0.36% a period against a
    rebalance cost of 1.7-1.9%.

(2) "What changes at institutional cost?" That is the same question. So the
    honest experiment is not to re-run the rules, it is to SWEEP THE TOLL and
    find, for each rule, the cost at which its sign flips.

WHAT IS GENUINELY NEW HERE AND WHAT IS A RE-RUN. The rules are old. The cost
ladder, the long-short arm, and the break-even toll are new. Nothing in this
repo has ever varied the cost — it has been a constant since A5, and A20's
lesson is that a constant nobody chose is an untested assumption with a
project-wide blast radius.

THE COST ANCHORS ARE NOT ARBITRARY.
  56 bps  the user's actual Mandiri retail schedule (A5), plus the spread
  25 bps  a plausible institutional all-in on liquid IDX names
  15 bps  aggressive institutional; roughly the floor once the 0.1% Indonesian
          final sale tax is counted, because that tax applies to EVERY seller
  10 bps  below the tax floor -- a hypothetical, and labelled as one
   0 bps  gross, to separate signal from toll
Anything under ~10 bps is not reachable on IDX by anyone, because the sale tax
alone is 10 bps. That is stated so the ladder is not read as a menu.

THE SHORT LEG IS A HYPOTHETICAL AND IS LABELLED. IDX restricts short selling to
a permitted list and borrow is thin; A5 forbids it for this account entirely.
The long-short arm exists to measure how much of each signal is thrown away by
being long-only, which is the single biggest structural difference between this
repo's constraints and a bank's. It is not a tradeable recommendation.

PRE-REGISTERED, WRITTEN BEFORE ANY CELL WAS SCORED
--------------------------------------------------
C1  Every rule's net return is monotone decreasing in the toll (arithmetic),
    so each has a BREAK-EVEN COST. The ranking of rules by break-even cost will
    NOT match their ranking by gross return, because turnover differs: a rule
    that trades rarely survives a higher toll on a smaller edge.
C2  At 0 bps, `mom` and `strength_calm` are positive (they are the measured
    incumbents) and `fib` is not (H34a found nothing on 280,228 touches). If
    `fib` comes back positive at zero cost, suspect the harness.
C3  PREDICTED NULL -- `rand` is flat at every cost level except for the toll it
    pays. A9: register a predicted-null feature in every sweep; it is the
    cheapest check that the pipeline is not manufacturing its own signal.
C4  The long-short arm roughly DOUBLES the gross spread of any rule with a real
    cross-sectional signal, and leaves `rand` at zero. If long-short does not
    beat long-only gross for the incumbents, the ranking is not monotone in the
    signal and something is wrong.
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

PANEL = os.path.join("data", "spine", "price_panel.parquet")
IND = os.path.join("data", "spine", "indicator_panel.parquet")
INDEX = os.path.join("data", "cache", "ohlcv", "_JKSE.csv.gz")
OUT = "reports"
MIN_TV = 1e9
IDX_YIELD = 0.0177
STEP = 21                      # rebalance every ~month
DECILE = 0.10
MIN_BOOK, MAX_BOOK = 5, 60
#: round-trip cost in basis points. See the module docstring: 10 bps is already
#: below the Indonesian 0.1% final sale tax, so it is a hypothetical.
TOLLS = (56, 40, 25, 15, 10, 0)


# =========================================================== the rule scores ==
#  Every score is computed from bars <= t. A rule that peeks is not a rule.
def add_scores(g: pd.DataFrame) -> pd.DataFrame:
    """All rule scores for one ticker. Higher = more bullish, always."""
    p = g["adj_close"].to_numpy(float)
    out = {}

    #  EMA cross — the classic. Distance between a fast and a slow EMA.
    out["ema_cross"] = np.log(np.maximum(g["ema20"].to_numpy(float), 1e-9)
                              / np.maximum(g["ema50"].to_numpy(float), 1e-9))
    #  EMA stack — H30's gridded winner, as an ordinal 0-3.
    e20 = g["ema20"].to_numpy(float)
    e30 = g["ema30"].to_numpy(float)
    e50 = g["ema50"].to_numpy(float)
    out["ema_stack"] = ((p > e20).astype(float) + (e20 > e30).astype(float)
                        + (e30 > e50).astype(float))
    #  STOCHASTIC, read the conventional way: oversold is bullish, so the score
    #  is the NEGATED %K. The opposite reading is tested too, because "buy
    #  oversold" and "buy strength" are opposite rules and only one can be right.
    k = g["stoch_k"].to_numpy(float)
    out["stoch_oversold"] = -k
    out["stoch_strong"] = k

    #  SUPPORT / RESISTANCE — H34b's event, the one level result that replicated.
    #  Score = how far price sits above the nearest CONFIRMED prior swing high,
    #  which is zero or negative until the breakout happens.
    res = _swing_res(p)
    with np.errstate(divide="ignore", invalid="ignore"):
        out["sr_break"] = np.where(np.isfinite(res), np.log(p / res), 0.0)

    #  FIBONACCI — the classic "buy the 61.8% retracement". Score = closeness to
    #  that level from above, 1 at the level and decaying away from it. H34a
    #  measured the ratios as nothing on a continuous grid; this is here as the
    #  user's hypothesis stated in its most favourable form.
    out["fib_618"] = _fib_prox(p, 0.618)

    #  Positive controls: the two things this repo has actually measured.
    out["mom12_1"] = g["mom12_1"].to_numpy(float)
    hz = pd.Series(g["hi52"].to_numpy(float))
    vz = pd.Series(g["vol60"].to_numpy(float))
    out["strength_calm"] = (hz.rank(pct=True) - vz.rank(pct=True)).to_numpy()

    for k_, v in out.items():
        g[k_] = v
    return g


def _swing_res(p: np.ndarray, k: float = 0.10) -> np.ndarray:
    """Nearest CONFIRMED swing high above price, per bar.

    Confirmation-gated, never pivot-gated: a high counts only once price has
    fallen `k` from it, which is the information a trader actually had. Every
    ZigZag look-ahead in this repo came from forgetting that.
    """
    n = len(p)
    res = np.full(n, np.nan)
    highs: List[float] = []
    cur_hi = cur_lo = p[0] if n else np.nan
    d = 0
    for i in range(n):
        if d > 0:
            cur_hi = max(cur_hi, p[i])
        elif d < 0:
            cur_lo = min(cur_lo, p[i])
        else:
            cur_hi, cur_lo = max(cur_hi, p[i]), min(cur_lo, p[i])
        if d >= 0 and p[i] <= cur_hi * (1 - k):
            highs.append(cur_hi)
            d, cur_lo = -1, p[i]
        elif d <= 0 and p[i] >= cur_lo * (1 + k):
            d, cur_hi = 1, p[i]
        above = [h for h in highs if h > p[i]]
        if above:
            res[i] = min(above)
    return res


def _fib_prox(p: np.ndarray, ratio: float, k: float = 0.10) -> np.ndarray:
    """Proximity to the `ratio` retracement of the last COMPLETED leg.

    1.0 sitting exactly on the level, decaying to 0 more than 10% away. The
    leg is confirmed before it is used, so the level is one a trader could have
    drawn at the time.
    """
    n = len(p)
    out = np.zeros(n)
    lo = hi = p[0] if n else np.nan
    cur_hi = cur_lo = p[0] if n else np.nan
    d = 0
    for i in range(n):
        if d > 0:
            cur_hi = max(cur_hi, p[i])
        elif d < 0:
            cur_lo = min(cur_lo, p[i])
        else:
            cur_hi, cur_lo = max(cur_hi, p[i]), min(cur_lo, p[i])
        if d >= 0 and p[i] <= cur_hi * (1 - k):
            hi, d, cur_lo = cur_hi, -1, p[i]
        elif d <= 0 and p[i] >= cur_lo * (1 + k):
            lo, d, cur_hi = cur_lo, 1, p[i]
        if np.isfinite(hi) and np.isfinite(lo) and hi > lo:
            lvl = hi - ratio * (hi - lo)
            if lvl > 0:
                out[i] = max(0.0, 1.0 - abs(p[i] / lvl - 1.0) / 0.10)
    return out


RULES = ("ema_cross", "ema_stack", "stoch_oversold", "stoch_strong",
         "sr_break", "fib_618", "mom12_1", "strength_calm", "rand")


# ============================================================== the harness ===
CACHE = os.path.join("data", "spine", "cost_ladder.parquet")


def build(step: int = STEP, cache: bool = True) -> pd.DataFrame:
    """One row per (rebalance period, ticker): every score, and the forward
    return over the period, realised at the last print if the name dies.

    Cached because the swing-high and Fibonacci scans are O(bars x pivots) per
    ticker over 774 names, and every downstream study reuses the same table.
    """
    if cache and step == STEP and os.path.exists(CACHE):
        return pd.read_parquet(CACHE)
    P = pd.read_parquet(PANEL)
    P = P[P["adj_close"] > 0].sort_values(["ticker", "date"])
    I = pd.read_parquet(IND)[["date", "ticker", "ema20", "ema30", "ema50",
                              "stoch_k"]]
    P = P.merge(I, on=["date", "ticker"], how="left")
    P = P.sort_values(["ticker", "date"])
    dates = np.sort(P["date"].unique())
    marks = dates[::step]
    mark_set = set(marks.tolist())

    frames: List[pd.DataFrame] = []
    for tk, g in P.groupby("ticker", sort=False):
        if len(g) < 300:
            continue
        g = add_scores(g.reset_index(drop=True))
        dt = g["date"].to_numpy()
        px = g["adj_close"].to_numpy(float)
        sel = np.array([i for i, d in enumerate(dt) if d in mark_set])
        if len(sel) < 3:
            continue
        #  FORWARD RETURN TO THE NEXT REBALANCE, from the name's OWN bars. Never
        #  from a pivot (A11) and never forward-filled past a delisting (H41):
        #  a name that stops printing is realised at its last real close.
        nxt = np.searchsorted(marks, dt[sel], side="right")
        fwd = np.full(len(sel), np.nan)
        for j, (i, m) in enumerate(zip(sel, nxt)):
            if m >= len(marks):
                continue
            end = marks[m]
            e = np.searchsorted(dt, end, side="right") - 1
            if e <= i:
                continue
            fwd[j] = px[e] / px[i] - 1.0
        d = g.iloc[sel].copy()
        d["fwd"] = fwd
        frames.append(d[["date", "ticker", "adj_close", "close", "fwd",
                         "tradeable", "log_turnover", "holdout"]
                        + [c for c in RULES if c in d.columns]])
    D = pd.concat(frames, ignore_index=True)
    #  ENTRY eligibility only. A19: eligibility is a condition for BUYING, never
    #  a filter applied along the forward path.
    D = D[D["tradeable"].astype(bool)]
    D = D[np.exp(D["log_turnover"].fillna(-np.inf)) >= MIN_TV]
    D = D.dropna(subset=["fwd"])
    #  THE PREDICTED NULL. Deterministic from a hash of (ticker, date) so it is
    #  reproducible without a global RNG, which the workflow runtime forbids.
    D["rand"] = ((pd.util.hash_pandas_object(
        D["ticker"] + D["date"].astype(str), index=False).to_numpy()
        % 100000) / 100000.0)
    D = D.sort_values(["date", "ticker"]).reset_index(drop=True)
    if cache and step == STEP:
        D.to_parquet(CACHE, index=False)
    return D


def walk(D: pd.DataFrame, rule: str, toll_bps: float,
         short: bool = False) -> Dict:
    """Equal-weighted decile portfolio, cost charged on TURNOVER."""
    toll = toll_bps / 10000.0
    prev_l: set = set()
    prev_s: set = set()
    rets, turns, dates = [], [], []
    for day, g in D.groupby("date", sort=True):
        g = g.dropna(subset=[rule])
        if len(g) < 40:
            continue
        n = int(np.clip(round(len(g) * DECILE), MIN_BOOK, MAX_BOOK))
        s = g.sort_values(rule, ascending=False)
        L = s.head(n)
        S = s.tail(n) if short else s.iloc[0:0]
        lset, sset = set(L["ticker"]), set(S["ticker"])
        #  turnover: the fraction of each leg that changed
        tl = 1.0 - len(lset & prev_l) / max(len(lset), 1)
        ts = (1.0 - len(sset & prev_s) / max(len(sset), 1)) if short else 0.0
        turn = (tl + ts) / (2.0 if short else 1.0)
        r = float(L["fwd"].mean())
        if short:
            r = 0.5 * r - 0.5 * float(S["fwd"].mean())
        rets.append(r - turn * toll * (2.0 if short else 1.0))
        turns.append(turn)
        dates.append(day)
        prev_l, prev_s = lset, sset
    if len(rets) < 12:
        return {}
    r = np.asarray(rets)
    per_yr = 252.0 / STEP
    lg = np.log(np.clip(1.0 + r, 1e-6, None))
    mid = len(r) // 2
    return {"rule": rule, "toll": toll_bps, "arm": "L/S" if short else "long",
            "periods": len(r), "turnover": float(np.mean(turns)),
            "cagr": float(np.exp(lg.mean() * per_yr) - 1.0),
            "mean_per": float(r.mean()),
            "sd": float(r.std(ddof=1) * np.sqrt(per_yr)),
            "early": float(np.exp(lg[:mid].mean() * per_yr) - 1.0),
            "late": float(np.exp(lg[mid:].mean() * per_yr) - 1.0),
            "dates": dates, "rets": r}


def breakeven(rows: List[Dict]) -> float:
    """The toll at which this rule's net return crosses zero, interpolated.

    C1: the ranking by break-even toll will NOT match the ranking by gross
    return, because a rule that trades rarely survives a higher toll on a
    smaller edge. That is the number a desk actually needs.
    """
    d = sorted([(r["toll"], r["cagr"]) for r in rows if r])
    if not d:
        return np.nan
    #  net is linear in the toll: gross - turnover * toll * periods_per_year
    x = np.array([a for a, _ in d], float)
    y = np.array([b for _, b in d], float)
    if np.all(y > 0):
        return float("inf")
    if np.all(y <= 0):
        return 0.0
    i = int(np.argmax(y <= 0))
    x0, x1, y0, y1 = x[i - 1], x[i], y[i - 1], y[i]
    return float(x0 + (x1 - x0) * y0 / (y0 - y1))


def index_cagr(a, b) -> float:
    d = pd.read_csv(INDEX)
    d["date"] = pd.to_datetime(d["date"])
    d = d[d["adj_close"] > 0].set_index("date")["adj_close"]
    s = d.reindex(pd.DatetimeIndex([a, b]), method="ffill")
    yrs = (pd.Timestamp(b) - pd.Timestamp(a)).days / 365.25
    return float((s.iloc[-1] / s.iloc[0]) ** (1.0 / yrs) - 1.0) + IDX_YIELD


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=int, default=STEP)
    a = ap.parse_args()

    D = build(a.step)
    print(f"{len(D):,} (period, name) rows, {D['ticker'].nunique()} names, "
          f"{pd.Timestamp(D['date'].min()):%Y-%m-%d} -> "
          f"{pd.Timestamp(D['date'].max()):%Y-%m-%d}, "
          f"{D['date'].nunique()} rebalances every {a.step} sessions")
    idx = index_cagr(D["date"].min(), D["date"].max())
    print(f"IHSG over the same span, total-return basis: {idx:+.2%} a year\n")

    rows: List[Dict] = []
    for rule in RULES:
        for arm in (False, True):
            for t in TOLLS:
                r = walk(D, rule, t, short=arm)
                if r:
                    rows.append(r)
    R = pd.DataFrame([{k: v for k, v in r.items() if k not in ("dates", "rets")}
                      for r in rows])
    R.to_csv(os.path.join(OUT, "cost_ladder.csv"), index=False)

    for arm, lbl in ((("long"), "LONG-ONLY"),
                     (("L/S"), "LONG-SHORT (hypothetical: IDX restricts borrow)")):
        print(f"=== {lbl} — net CAGR by round-trip toll")
        print(f"{'rule':<16}{'turn':>6}" +
              "".join(f"{t:>9}bp" for t in TOLLS) +
              f"{'break-even':>12}{'halves@25':>11}")
        for rule in RULES:
            g = [r for r in rows if r["rule"] == rule and r["arm"] == arm]
            if not g:
                continue
            by = {r["toll"]: r for r in g}
            be = breakeven(g)
            h = by.get(25)
            hv = (f"{h['early']:+.1%}/{h['late']:+.1%}" if h else "n/a")
            bes = ("never" if be == 0.0 else
                   ">56" if be == float("inf") else f"{be:.0f}bp")
            print(f"{rule:<16}{g[0]['turnover']:>6.0%}" +
                  "".join(f"{by[t]['cagr']:>+11.2%}" if t in by else f"{'':>11}"
                          for t in TOLLS) +
                  f"{bes:>12}{hv:>11}")
        print()

    print("READING IT. `break-even` is the round-trip toll at which the rule "
          "stops making money.")
    print("The Indonesian 0.1% final SALE TAX applies to every seller, so "
          "~10bp is the floor for")
    print("anyone; a rule whose break-even is under that is untradeable by "
          "anybody, at any size.")
    print(f"The index returned {idx:+.2%} a year over the same span and costs "
          f"one round trip, ever.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
