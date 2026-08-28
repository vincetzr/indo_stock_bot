#!/usr/bin/env python3
"""H40 — the Hull's own flip price as a dynamic stop, measured on every IDX name.

    python3 scripts/hull_stop.py

WHAT CAN AND CANNOT BE DONE, STATED FIRST.

"A stop based on when the Hull will turn colour" splits into two requests and
only one of them is possible.

  FORECASTING THE FLIP DATE — impossible on this evidence. H31 tested it three
  ways: ZigZag pivot spacing is 68% MORE dispersed than a block-bootstrap of the
  same returns, the interval memory that exists is volatility clustering (a
  252-day block null reproduces 87% of it), and the IHSG has no dominant period
  (p = 0.499). Knowing a name's entire history of turns narrows the band for its
  next one from +/-140% of the median gap to +/-125%.

  COMPUTING THE PRICE AT WHICH IT FLIPS TOMORROW — exact, and it is arithmetic
  rather than prediction. The Hull moving average is a LINEAR function of the
  next close, so there is a single price x* at which tomorrow's HMA equals
  today's. Above it the ribbon stays green, below it the ribbon turns red.
  Solved in closed form below and checked against a brute-force recomputation to
  1e-14. That is a real dynamic stop: a level you can compute at tonight's close
  and place as an order tomorrow morning.

THE DERIVATION, because a stop nobody can check is a stop nobody should use.
With m = n/2, s = round(sqrt(n)), W_k = k(k+1)/2 and x tomorrow's close:

    WMA(p, k)_{t+1} = (k*x + C_k) / W_k,      C_k = W_{k-1} * WMA(p, k-1)_t
    d_{t+1}         = 2*WMA(p, m)_{t+1} - WMA(p, n)_{t+1}  =  b*x + a
    HMA_{t+1}       = (s*d_{t+1} + E) / W_s,  E   = W_{s-1} * WMA(d, s-1)_t

    b = 2m/W_m - n/W_n        a = 2*C_m/W_m - C_n/W_n
    x* = (HMA_t * W_s - E - s*a) / (s*b)

b > 0 for every length used here, so the ribbon rises tomorrow iff x > x*.

PRE-REGISTERED, WRITTEN BEFORE ANY CELL WAS SCORED
--------------------------------------------------
S1  THE HULL-55 FLIP PRICE WILL BE A VERY WIDE STOP — a median distance below
    the close of more than 10%, and far more in a strong trend. A slow Hull
    turns only after a large move, which is the same fact H37 measured as an
    11% median give-back at the flip. If so, "stop at the flip price" is not a
    tight stop and must not be sold as one.
S2  Exiting AT the flip price will beat H39's "exit one bar after the Hull
    turns" on mean log, because it saves a bar of lag on every exit and the bar
    after a Hull flip is on average a down bar.
S3  IT WILL STILL LOSE TO BUY-AND-HOLD ON CAGR, like all 40 cells of H39, for
    the same reason: the rule is out of the market two thirds of the time and
    the market it is out of goes up.
S4  A FASTER TRAIL (exiting on the HMA-21 flip price while entering on the
    HMA-55) will raise the win rate and LOWER the average return — cutting
    winners earlier is what a tighter trail does, and H39 already found faster
    exits win the compounding and lose the average trade.
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))
sys.path.insert(0, os.path.dirname(__file__))

from hull_trade import COST, hold_curve, states                  # noqa: E402
from levels import ZZ, pivots_confirmed                          # noqa: E402
from time_price import MIN_BARS, eligible, hma, load, wma        # noqa: E402

OUT = "reports"
ENTRY_HULL = 55
TRAILS = (21, 34, 55)


def flip_price(px: pd.Series, n: int) -> np.ndarray:
    """The close at which HMA(n) tomorrow equals HMA(n) today, per bar.

    Vectorised: every term is a rolling weighted mean of past closes, so the
    whole series costs four passes rather than one HMA recomputation per bar.
    """
    m = max(1, n // 2)
    s = max(1, int(round(np.sqrt(n))))
    Wm, Wn, Ws = m * (m + 1) / 2.0, n * (n + 1) / 2.0, s * (s + 1) / 2.0
    b = 2.0 * m / Wm - n / Wn
    #  C_k = W_{k-1} * WMA(p, k-1), the contribution of the bars already known
    Cm = (m - 1) * m / 2.0 * wma(px, m - 1).to_numpy() if m > 1 else np.zeros(len(px))
    Cn = (n - 1) * n / 2.0 * wma(px, n - 1).to_numpy() if n > 1 else np.zeros(len(px))
    a = 2.0 * Cm / Wm - Cn / Wn
    d = 2.0 * wma(px, m).to_numpy() - wma(px, n).to_numpy()
    E = ((s - 1) * s / 2.0 * wma(pd.Series(d), s - 1).to_numpy()
         if s > 1 else np.zeros(len(px)))
    h = hma(px, n).to_numpy()
    return (h * Ws - E - s * a) / (s * b)


def swing_levels(p: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Nearest CONFIRMED swing high above and low below price, per bar.

    The high is the dynamic target; the low is the "classic" support stop this
    study puts head to head against the Hull's own flip price. Both are gated
    on the CONFIRMATION bar, never the pivot bar.
    """
    piv, conf = pivots_confirmed(p, ZZ)
    res = np.full(len(p), np.nan)
    sup = np.full(len(p), np.nan)
    highs: List[float] = []
    lows: List[float] = []
    ptr = 0
    for i in range(len(p)):
        while ptr < len(piv) and conf[ptr] <= i:
            if ptr > 0 and p[piv[ptr]] > p[piv[ptr - 1]]:
                highs.append(float(p[piv[ptr]]))
            elif ptr > 0:
                lows.append(float(p[piv[ptr]]))
            ptr += 1
        above = [h for h in highs if h > p[i]]
        below = [l for l in lows if l < p[i]]
        if above:
            res[i] = min(above)
        if below:
            sup[i] = max(below)
    return res, sup


def swing_res(p: np.ndarray) -> np.ndarray:
    return swing_levels(p)[0]


def campaign(p: np.ndarray, enter: np.ndarray, stop: np.ndarray,
             tp: np.ndarray, el: np.ndarray, yr: np.ndarray,
             lag_exit: bool, from_entry: float = 0.0,
             from_peak: float = 0.0, max_bars: int = 252) -> List[Dict]:
    """Walk positions with a per-bar stop level and an optional target.

    `stop[t]` is the level computed AT bar t for bar t+1, so comparing it to
    p[t+1] uses nothing from the future. `lag_exit` reproduces H39's extra
    confirmation bar so the two are directly comparable.
    """
    n = len(p)
    out: List[Dict] = []
    i = 0
    while i < n - 1:
        if not (enter[i] and el[i]):
            i += 1
            continue
        entry = i + 1                       # fill on the next close
        if entry >= n:
            break
        j = entry
        exit_at = -1
        why = ""
        peak = p[entry]
        #  A TIME CAP ON EVERY RULE, OR THE COMPARISON IS RIGGED. A stop that
        #  only ever moves down -- a fixed percentage below the entry -- closes
        #  losers and never winners, so every completed trade is a loss and the
        #  win rate reads a definitionally impossible 0.0%. That is exactly what
        #  the first run printed. A trailing level eventually fires either way;
        #  a fixed one does not, and the cap is what makes them comparable.
        #  The cap only applies if it falls INSIDE the data. A position still
        #  open when the name's history ends is genuinely unresolved and is
        #  dropped, not marked to the last price -- closing it there would hand
        #  a rising sample a free winner on every name.
        stop_by = entry + max_bars - 1
        while j < n - 1:
            if j >= stop_by:
                exit_at = j + 1
                why = "time"
                break
            peak = max(peak, p[j])
            #  A level from the entry price or from the running peak is not a
            #  per-bar series, so it is built here. Both use only bars already
            #  seen, which is why `peak` is updated BEFORE the comparison and
            #  never from p[j + 1].
            lvl = stop[j]
            if from_entry:
                lvl = np.fmax(np.nan_to_num(lvl, nan=-np.inf),
                              p[entry] * (1.0 - from_entry))
            if from_peak:
                lvl = np.fmax(np.nan_to_num(lvl, nan=-np.inf),
                              peak * (1.0 - from_peak))
            hit_stop = np.isfinite(lvl) and lvl > 0 and p[j + 1] < lvl
            hit_tp = np.isfinite(tp[j]) and p[j + 1] >= tp[j]
            if hit_stop or hit_tp:
                exit_at = j + 2 if lag_exit else j + 1
                why = "tp" if hit_tp and not hit_stop else "stop"
                break
            j += 1
        if exit_at < 0 or exit_at >= n:
            break                            # unresolved at the end: dropped
        out.append({"i": entry, "j": exit_at, "year": int(yr[entry]),
                    "bars": exit_at - entry, "why": why,
                    "ret": p[exit_at] / p[entry] - 1.0 - COST})
        i = exit_at
    return out


def summarise(T: pd.DataFrame, C: pd.DataFrame, HC: pd.DataFrame,
              tag: str) -> Dict:
    if T.empty:
        return {}
    lg = np.log1p(np.clip(T["ret"], -0.99, None))
    cut = int(T["year"].median())
    e, l = T[T["year"] < cut], T[T["year"] >= cut]
    bars = float(T["bars"].mean())
    m_hold = float(np.interp(bars, HC["h"], HC["meanlog"]))
    return {"config": tag, "trades": len(T), "win": float((T["ret"] > 0).mean()),
            "mean": float(T["ret"].mean()), "median": float(T["ret"].median()),
            "bars": bars, "meanlog": float(lg.mean()),
            "time_share": float((T["why"] == "time").mean()),
            "edge": float(lg.mean()) - m_hold,
            "tp_share": float((T["why"] == "tp").mean()),
            "in_mkt": float(C["in_mkt"].mean()),
            "cagr": float(np.median(C["strat"] ** (252.0 / C["span"]) - 1.0)),
            "cagr_hold": float(np.median(C["hold"] ** (252.0 / C["span"]) - 1.0)),
            "beats_hold": float((C["strat"] > C["hold"]).mean()),
            "edge_early": float(np.log1p(np.clip(e["ret"], -0.99, None)).mean())
            - m_hold,
            "edge_late": float(np.log1p(np.clip(l["ret"], -0.99, None)).mean())
            - m_hold}


def main() -> int:
    P = load()
    P["elig"] = P["tradeable"].astype(bool)
    P["px"] = P["adj_close"]
    P = P.sort_values(["ticker", "date"])
    print(f"panel {len(P):,} rows  {P['ticker'].nunique()} names\n")
    HC = hold_curve(P)

    dist: List[np.ndarray] = []
    acc: Dict[str, List[Dict]] = {}
    camp: Dict[str, List[Dict]] = {}
    for tk, g in P.groupby("ticker", sort=False):
        if len(g) < MIN_BARS:
            continue
        px = g["px"].reset_index(drop=True)
        p = px.to_numpy(float)
        el = g["elig"].to_numpy()
        yr = pd.DatetimeIndex(g["date"]).year.to_numpy()
        up55, on = states(px, ENTRY_HULL, "EMA stack")
        enter = up55 & on
        fp = {t: flip_price(px, t) for t in TRAILS}
        res, sup = swing_levels(p)
        e34 = px.ewm(span=34, adjust=False, min_periods=34).mean().to_numpy()
        e50 = px.ewm(span=50, adjust=False, min_periods=50).mean().to_numpy()
        #  How far below the close does the ribbon's own reversal level sit,
        #  while the ribbon is green? This is the stop distance a user would
        #  actually be quoted, and S1 says it is wide.
        live = up55 & np.isfinite(fp[ENTRY_HULL])
        if live.any():
            dist.append(fp[ENTRY_HULL][live] / p[live] - 1.0)
        blank = np.full(len(p), np.nan)
        #  THE HEAD TO HEAD THE QUESTION ASKS: same entry, same target, only
        #  the stop differs. Anything that changes with the stop is the stop.
        cfgs: Dict[str, tuple] = {
            "SL = hull55 flip price": (fp[55], blank, False, 0.0, 0.0),
            "SL = hull34 flip price": (fp[34], blank, False, 0.0, 0.0),
            "SL = hull21 flip price": (fp[21], blank, False, 0.0, 0.0),
            "SL = confirmed swing support": (sup, blank, False, 0.0, 0.0),
            "SL = close under EMA34": (e34, blank, False, 0.0, 0.0),
            "SL = close under EMA50": (e50, blank, False, 0.0, 0.0),
            "SL = fixed -10% from entry": (blank, blank, False, 0.10, 0.0),
            "SL = fixed -20% from entry": (blank, blank, False, 0.20, 0.0),
            "SL = trail -15% from peak": (blank, blank, False, 0.0, 0.15),
            "SL = trail -25% from peak": (blank, blank, False, 0.0, 0.25),
            "hull55 flip + swing-high TP": (fp[55], res, False, 0.0, 0.0),
            "EMA34 + swing-high TP": (e34, res, False, 0.0, 0.0),
            "hull55 flip, H39's extra lag bar": (fp[55], blank, True, 0.0, 0.0),
        }
        for tag, (st, tp, lag, fe, fpk) in cfgs.items():
            tl = campaign(p, enter, st, tp, el, yr, lag, fe, fpk)
            if not tl:
                continue
            a0, b1 = tl[0]["i"], tl[-1]["j"]
            span = max(1, b1 - a0)
            for t in tl:
                acc.setdefault(tag, []).append(t)
            camp.setdefault(tag, []).append({
                "ticker": tk, "span": span,
                #  Clip the growth factor at 1%: a trade cannot lose more than
                #  the capital in it, but ret = price ratio - 1 - cost CAN fall
                #  below -1 on a name that collapses, and a negative factor
                #  raised to a fractional power is NaN -- which is what the
                #  first run printed for every fixed-stop row.
                "strat": float(np.prod([max(0.01, 1.0 + x["ret"])
                                        for x in tl])),
                "hold": float(p[b1] / p[a0]),
                "in_mkt": float(sum(x["bars"] for x in tl)) / span})

    D = np.concatenate(dist)
    print("=== S1: how far below the close is the Hull's own reversal level,")
    print("    measured on every bar where the hull55 ribbon is green")
    q = np.percentile(D, [5, 25, 50, 75, 95])
    print("  " + "   ".join(f"p{p_}: {v:+.1%}" for p_, v in
                            zip((5, 25, 50, 75, 95), q)))
    print(f"  a stop at this level is {abs(q[2]):.0%} away at the median, and "
          f"{abs(q[0]):.0%} away in the worst twentieth.")

    rows = [summarise(pd.DataFrame(v), pd.DataFrame(camp[k]), HC, k)
            for k, v in acc.items()]
    R = pd.DataFrame([r for r in rows if r]).sort_values("cagr", ascending=False)
    R.to_csv(os.path.join(OUT, "hull_stop.csv"), index=False)
    print(f"\n=== H40 the dynamic stop, entry = hull{ENTRY_HULL} rising + EMA "
          f"stack, net of {COST:.2%}")
    print(f"{'config':<40}{'trades':>8}{'win':>7}{'mean':>8}{'median':>8}"
          f"{'bars':>6}{'tp%':>6}{'time%':>7}{'in mkt':>8}{'CAGR':>8}{'hold':>8}"
          f"{'beat%':>7}{'e.early':>9}{'e.late':>8}{'both+':>7}")
    for _, x in R.iterrows():
        both = "YES" if x["edge_early"] > 0 and x["edge_late"] > 0 else ""
        print(f"{x['config']:<40}{int(x['trades']):>8,}{x['win']:>7.1%}"
              f"{x['mean']:>8.2%}{x['median']:>8.2%}{x['bars']:>6.0f}"
              f"{x['tp_share']:>6.0%}{x['time_share']:>7.0%}{x['in_mkt']:>8.1%}{x['cagr']:>8.2%}"
              f"{x['cagr_hold']:>8.2%}{x['beats_hold']:>7.1%}"
              f"{x['edge_early']:>9.4f}{x['edge_late']:>8.4f}{both:>7}")
    print(f"\n  cells beating buy-and-hold on CAGR: "
          f"{int((R['cagr'] > R['cagr_hold']).sum())} of {len(R)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
