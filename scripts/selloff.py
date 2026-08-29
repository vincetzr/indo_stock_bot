#!/usr/bin/env python3
"""H47 — can a fast sell-off detector cut the give-back without costing return?

    python3 scripts/selloff.py

THE OBSERVATION, WHICH IS CORRECT AND MEASURED.

"The entry is usually great, but it sells a bit late so the profit is not
maximum." Both halves of that are in the data already:

  ENTRY  H39's benchmark column found that spans selected by the EMA-stack
         entry returned +9.9% to +11.2%/yr if you merely OWNED the name,
         against a panel median of +2.56%. The entry carries real information
         about WHAT TO OWN. That is the one unambiguously positive finding
         about this indicator.

  EXIT   H37 measured the give-back — the share of the peak already
         surrendered when the flip fires — at 11.8% for the Hull-55 slope,
         10.9% for EMA34, 12.5% for the EMA stack. And the uncomfortable part:
         a RANDOM detector spending the same number of flips gives back only
         8.5-8.9%. The real detectors give back MORE of the peak than random
         bars do, because a trend flip fires BECAUSE price fell — it is
         conditioned on the drop having already happened.

WHY "SELL AT THE TOP" IS NOT AVAILABLE, AND WHAT IS.
Any rule that fires on a decline is downstream of the decline. Selling at the
top requires firing BEFORE it, which is prediction; H31 tested that directly
and found turning-point spacing is MORE dispersed than a block-bootstrap of the
same returns (CV 2.246 against 1.340). So the reachable question is not "can I
sell at the top" but "for a given amount of give-back saved, what does it
cost" — a frontier, not a point.

WHAT IS ACTUALLY NEW HERE. The 158 exit configurations tested so far are
almost all SLOW: moving-average crossovers, trailing stops, indicator
rollovers. A sell-off has a different signature — one violent day, a
three-day cascade, a volume climax — and an ASYMMETRIC exit (fast when the
decline is sharp, slow when it drifts) has never been cleanly separated from
the symmetric ones. H18's `volume climax` did win its MEAN objective (+18.42%)
before H20's portfolio accounting retracted the family, and that is a genuine
loose end.

PRE-REGISTERED, WRITTEN BEFORE ANY CELL WAS SCORED
--------------------------------------------------
S1  Every sell-off detector reduces give-back relative to waiting for the Hull.
    This is near-mechanical — they fire earlier — and if one does NOT, the
    harness is wrong.
S2  The return cost exceeds the give-back saved, so CAGR falls monotonically
    as give-back falls. This is H40's S4 shape ("a tighter trail lowers BOTH
    the win rate and the mean") arriving from a new direction.
S3  PREDICTED NULL, AND THE ONE THAT DECIDES IT — a RANDOM exit drawn from the
    same holding-period distribution reduces give-back by about as much as a
    real detector. Give-back is mostly a function of WHEN you leave, not WHY,
    so cutting it is not evidence of detection. A real detector has to beat the
    coin flip at the same trading rate or it is doing nothing.
S4  No sell-off exit beats simply holding the name for the same span. 158 have
    failed; a fast one should fail too, and for the same reason — the right
    tail lives past the shakeout.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))
sys.path.insert(0, os.path.dirname(__file__))

from paint_suite import tick_of                                  # noqa: E402

PANEL = os.path.join("data", "spine", "price_panel.parquet")
IND = os.path.join("data", "spine", "indicator_panel.parquet")
OUT = "reports"
MIN_TV = 1e9
MAX_BARS = 252
FEE = 0.0056


def wma(v: np.ndarray, k: int) -> np.ndarray:
    """Vectorised WMA. `rolling().apply()` is ~1000x slower and killed a run."""
    w = np.arange(1, k + 1, dtype=float)
    w /= w.sum()
    out = np.full(len(v), np.nan)
    if len(v) >= k:
        out[k - 1:] = np.convolve(v, w[::-1], mode="valid")
    return out


def hma(v: np.ndarray, n: int) -> np.ndarray:
    m, s = max(1, n // 2), max(1, int(round(np.sqrt(n))))
    return wma(np.nan_to_num(2.0 * wma(v, m) - wma(v, n)), s)


def ema(v: np.ndarray, n: int) -> np.ndarray:
    return pd.Series(v).ewm(span=n, adjust=False).mean().to_numpy()


#: Each detector returns a boolean array: True = get out at this bar's close.
#: Every one is causal — it reads bar t and nothing after it.
def detectors(p: np.ndarray, atr: np.ndarray, tvz: np.ndarray,
              hull: np.ndarray) -> Dict[str, np.ndarray]:
    r1 = np.r_[0.0, p[1:] / p[:-1] - 1.0]
    r3 = np.r_[np.zeros(3), p[3:] / p[:-3] - 1.0]
    red = np.zeros(len(p), bool)
    red[2:] = hull[2:] <= hull[:-2]
    #  H40's S2: confirming ONE BAR after the flip beats exiting at it, because
    #  the bar after a Hull flip is on average an UP bar (H13's `rev1`).
    red_confirmed = np.r_[False, red[:-1]]
    with np.errstate(invalid="ignore"):
        atr_drop = r1 * p < -2.0 * np.nan_to_num(atr, nan=np.inf)
    return {
        "hull55 +1bar": red_confirmed,
        "drop -4%": r1 <= -0.04,
        "drop -6%": r1 <= -0.06,
        "drop -8%": r1 <= -0.08,
        "3day -8%": r3 <= -0.08,
        "3day -12%": r3 <= -0.12,
        "vol climax": (r1 < 0) & (np.nan_to_num(tvz) >= 2.0),
        "2x ATR down": atr_drop,
        "close < EMA20": p < ema(p, 20),
    }


def campaign(p: np.ndarray, enter: np.ndarray, exit_sig: np.ndarray,
             cost: float, tk: str = "", trail: float | None = None,
             rand_bars: np.ndarray | None = None,
             rng: np.random.Generator | None = None,
             reentry: str = "live") -> List[Dict]:
    """Walk every entry to its exit, recording the GIVE-BACK from the peak.

    give_back = (peak while held - exit price) / peak. It is the quantity the
    observation is about: how much of the move you hand back before leaving.
    """
    out: List[Dict] = []
    i = 1
    n = len(p)
    while i < n:
        #  RE-ENTRY MUST BE SYMMETRIC BETWEEN THE ARMS OR THE CONTROL IS RIGGED.
        #  A first version required a fresh RISING EDGE of `enter`. That quietly
        #  handicapped the random arm: a real detector fires on a drop, which
        #  usually breaks the entry condition too, so the real arm gets a fresh
        #  edge and re-enters the next leg — while the random arm exits
        #  mid-trend with `enter` still True and then cannot re-enter until the
        #  whole trend dies and restarts. It was skipping the rest of every
        #  trend it sold into, and the real rules' CAGR advantage over it was
        #  partly that artefact rather than detection.
        #  Both arms now re-enter whenever the setup is live, we are flat, and
        #  the exit signal is not currently firing — which is also what a person
        #  would do: sell the shakeout, buy back when the setup is intact.
        live = (enter[i] and not exit_sig[i]) if reentry == "live" else (
            enter[i] and not enter[i - 1])
        if not live:
            i += 1
            continue
        peak = p[i]
        j = i + 1
        stop_at = None
        if rand_bars is not None and rng is not None:
            #  S3's control: leave after a draw from the REAL rule's own
            #  holding-period distribution, so the trading RATE is matched and
            #  only the reason for leaving differs.
            stop_at = i + int(rng.choice(rand_bars))
        while j < n:
            peak = max(peak, p[j])
            if stop_at is not None:
                if j >= stop_at or j - i >= MAX_BARS:
                    break
            elif trail is not None:
                if p[j] <= peak * (1.0 - trail) or j - i >= MAX_BARS:
                    break
            elif exit_sig[j] or j - i >= MAX_BARS:
                break
            j += 1
        j = min(j, n - 1)
        out.append({"tk": tk, "span": n / 252.0, "i": i, "j": j, "bars": j - i,
                    "ret": p[j] / p[i] - 1.0 - cost,
                    "give_back": (peak - p[j]) / peak if peak > 0 else np.nan,
                    "hold_span": np.log(max(p[-1] / p[0], 0.01))})
        i = j + 1
    return out


def summarise(tr: List[Dict], name: str) -> Dict:
    if len(tr) < 30:
        return {}
    D = pd.DataFrame(tr)
    r = D["ret"].to_numpy()
    g = D["give_back"].to_numpy()
    b = D["bars"].to_numpy()
    D["lg"] = np.log(np.clip(1.0 + r, 0.01, None))
    #  ANNUALISE OVER THE WHOLE SPAN, INCLUDING THE TIME IN CASH.
    #  A first version scaled each rule's mean log by 252/bars_held, which
    #  assumes the trade repeats back-to-back all year. The rule is in the
    #  market ~33% of the time, so that overstated the annual rate roughly
    #  threefold and printed every rule BEATING buy-and-hold — contradicting
    #  H39, H40 and H42 at once. Three studies disagreeing with a fourth is the
    #  signal that the fourth is wrong. The right rate is the ticker's total log
    #  return from its trades divided by the ticker's FULL span, idle time
    #  included, which is exactly what H39 computed.
    per = D.groupby("tk").agg(lg=("lg", "sum"), span=("span", "first"),
                              hs=("hold_span", "first"))
    cagr = float(np.mean(np.exp(per["lg"] / per["span"]) - 1.0))
    hold = float(np.mean(np.exp(per["hs"] / per["span"]) - 1.0))
    return {"rule": name, "trades": len(r), "win": float((r > 0).mean()),
            "give_back": float(np.median(g)), "gb_mean": float(np.mean(g)),
            "bars": float(np.mean(b)), "mean": float(r.mean()),
            #  ANNUALISE THE MEAN LOG, never the mean of per-trade annualised
            #  returns — H42 printed 1.7e28 doing the latter.
            "in_mkt": float((b.sum() / 252.0) / per["span"].sum()),
            "cagr": cagr, "hold": hold, "names": len(per)}


def universe(P: pd.DataFrame, limit: int):
    """Yield (ticker, price, entry mask, cost, atr, tvz, hull) once per name."""
    kept = 0
    for tk, g in P.groupby("ticker", sort=False):
        if len(g) < 400 or kept >= limit:
            continue
        tv = np.exp(g["log_turnover"].to_numpy(float))
        if not np.isfinite(np.nanmedian(tv)) or np.nanmedian(tv) < MIN_TV:
            continue
        kept += 1
        p = g["adj_close"].to_numpy(float)
        med = float(np.nanmedian(g["close"].to_numpy(float)))
        cost = FEE + tick_of(med) / med
        h = hma(p, 55)
        up = np.zeros(len(p), bool)
        up[2:] = h[2:] > h[:-2]
        e50, e100, e200 = ema(p, 50), ema(p, 100), ema(p, 200)
        enter = up & (p > e50) & (e50 > e100) & (e100 > e200)
        yield (tk, p, enter, cost, g["atr22"].to_numpy(float),
               g["tvz20"].to_numpy(float), h)


def run(names, reentry: str, seed: int = 4747) -> pd.DataFrame:
    trades: Dict[str, List[Dict]] = {}
    rng = np.random.default_rng(seed)
    for tk, p, enter, cost, atr, tvz, h in names:
        for nm, sig in detectors(p, atr, tvz, h).items():
            trades.setdefault(nm, []).extend(
                campaign(p, enter, sig, cost, tk, reentry=reentry))
        for t in (0.15, 0.25):
            trades.setdefault(f"trail {t:.0%}", []).extend(
                campaign(p, enter, np.zeros(len(p), bool), cost, tk, trail=t,
                         reentry=reentry))
    #  THE CONTROL (S3): for each real rule, a random exit drawn from THAT
    #  rule's own holding-period distribution — matched trading rate, no signal,
    #  and now the same re-entry policy so the arms play the same game.
    base = {nm: np.clip(np.array([t["bars"] for t in tr]), 1, MAX_BARS)
            for nm, tr in trades.items() if len(tr) >= 30}
    for tk, p, enter, cost, _a, _t, _h in names:
        for nm, bars in base.items():
            trades.setdefault(f"RANDOM ~ {nm}", []).extend(
                campaign(p, enter, np.zeros(len(p), bool), cost, tk,
                         rand_bars=bars, rng=rng, reentry=reentry))
    rows = [summarise(tr, nm) for nm, tr in trades.items()]
    return pd.DataFrame([r for r in rows if r]).sort_values("give_back")


def report(R: pd.DataFrame, label: str, note: str) -> None:
    real = R[~R["rule"].str.startswith("RANDOM")]
    ctrl = R[R["rule"].str.startswith("RANDOM")].set_index("rule")
    print(f"\n=== re-entry policy: {label} — {note}")
    print(f"{'exit rule':<16}{'trades':>8}{'win':>7}{'GIVE-BACK':>11}"
          f"{'bars':>6}{'mean':>9}{'in mkt':>8}{'CAGR':>9}{'HOLD':>9}"
          f"   {'RANDOM of the same speed':<30}")
    for _, r in real.iterrows():
        k = f"RANDOM ~ {r['rule']}"
        c = ctrl.loc[k] if k in ctrl.index else None
        cmp = (f"gb {c['give_back']:>5.1%}  cagr {c['cagr']:>+6.2%}"
               f"  {'REAL WORSE' if r['give_back'] > c['give_back'] else 'real better'}"
               if c is not None else "")
        print(f"{r['rule']:<16}{int(r['trades']):>8,}{r['win']:>7.1%}"
              f"{r['give_back']:>11.1%}{r['bars']:>6.0f}{r['mean']:>+9.2%}"
              f"{r['in_mkt']:>8.0%}{r['cagr']:>+9.2%}{r['hold']:>+9.2%}"
              f"   {cmp:<30}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", type=int, default=400)
    a = ap.parse_args()

    P = pd.read_parquet(PANEL)
    P = P[P["adj_close"] > 0].sort_values(["ticker", "date"])
    I = pd.read_parquet(IND)[["date", "ticker", "atr22", "tvz20"]]
    P = P.merge(I, on=["date", "ticker"], how="left")
    names = list(universe(P, a.names))

    print(f"H47 — sell-off detectors, {len(names)} liquid names, entry = "
          f"hull55 rising AND price>EMA50>100>200, 252-bar cap, net of cost")
    out = []
    for pol, note in (("live", "re-enter whenever the setup is live and the "
                               "exit signal has cleared"),
                      ("edge", "re-enter only on a FRESH rising edge of the "
                               "setup — the natural rule, but it rigs the "
                               "control")):
        R = run(names, pol)
        R["reentry"] = pol
        report(R, pol, note)
        out.append(R)
    pd.concat(out).to_csv(os.path.join(OUT, "selloff.csv"), index=False)
    print("\n  GIVE-BACK = median share of the in-trade peak handed back at the")
    print("  exit. It is the quantity the complaint is about.")
    print("  CAGR and HOLD are both compounded per name over the name's FULL")
    print("  span, so the rule is charged for the time it sits in cash. Scaling")
    print("  a 23-bar trade by 252/23 instead assumes it repeats back-to-back")
    print("  all year and overstates the rate ~3x.")
    print("  The two policies exist because the EDGE policy is the sensible")
    print("  trading rule and the LIVE policy is the fair experiment: a real")
    print("  detector fires on a drop that usually breaks the setup too, so it")
    print("  gets a fresh edge and rejoins the next leg, while the random arm")
    print("  sells mid-trend and is locked out until the trend dies and")
    print("  restarts. Read S3 (give-back) on either; read any CAGR comparison")
    print("  against the control on LIVE only.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
