#!/usr/bin/env python3
"""H37/H38 — does the ribbon turn at the peak, and where should TP and SL sit?

    python3 scripts/turns.py flips    # H37  how close to the top does it flip?
    python3 scripts/turns.py place    # H38  at the level, before it, or after?

TWO QUESTIONS THE CALIBRATION STUDY DOES NOT ANSWER.

H36 asked whether the printed probabilities are honest. This asks the two things
a person actually watches the chart for: does the colour change at the top, and
where should the target and the stop go.

WHY RECALL ALONE IS MEANINGLESS AND THE NULL IS THE WHOLE TEST.
"It caught 80% of the tops" is free: a detector that flips every five bars
catches every top within thirty by construction, and it also flips two hundred
other times. So every detector here is scored on FOUR numbers together —

  recall     of the real tops, how many got a flip within 30 sessions
  precision  of the flips, how many were within 30 sessions after a real top
  lag        median sessions from the top to the flip
  give-back  median share of the peak surrendered before the flip fired

— and against a RANDOM detector with the SAME NUMBER OF FLIPS. A rule that
beats nothing but its own flip count has found nothing.

GROUND TRUTH IS A ZIGZAG PIVOT, WHICH IS HINDSIGHT ON PURPOSE. The peak is the
thing being predicted, so it is allowed to be defined with future information;
the detectors are not, and none of them sees a bar it could not have seen.

PRE-REGISTERED, WRITTEN BEFORE ANY CELL WAS SCORED
--------------------------------------------------
T1  (flips) EVERY DETECTOR WILL TRADE RECALL AGAINST PRECISION ALONG ONE CURVE
    and none will sit above it. A faster line flips nearer the top and flips
    far more often; a slower one is cleaner and later. If that is all there is,
    the choice of moving average is a choice of where to sit on the curve and
    not a choice of how good the indicator is.
T2  (flips) GIVE-BACK WILL BE LARGE — a median above 8% even for the fastest
    line — because a trend-following flip is a confirmation and confirmation
    costs the distance travelled while confirming. That is the number nobody
    quotes and it is the one that decides whether the flip is worth acting on.
T3  (place) PUTTING THE TARGET JUST BELOW RESISTANCE BEATS PUTTING IT AT OR
    ABOVE. H34b measured the false-break rate at a true swing high at 66.4%, so
    two breakouts in three fail; selling into the level should beat waiting for
    it to break.
T4  (place) THE STOP OFFSET WILL MATTER LESS THAN THE TARGET OFFSET, because a
    stop placed near support is hit by the same move that would hit it anywhere
    nearby, whereas a target is a decision about whether to wait for a break.
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))
sys.path.insert(0, os.path.dirname(__file__))

from levels import pivots_confirmed                              # noqa: E402
from time_price import MIN_BARS, eligible, hma, load             # noqa: E402

OUT = "reports"
SEED = 20260828
ZZ = 0.10                # what counts as a real swing
WINDOW = 30              # sessions inside which a flip "caught" the top
HORIZON = 252
COST = 0.0056
OFFSETS = (-0.05, -0.02, 0.0, 0.02, 0.05)
DRAWS = 200


# ============================================== H37 — catching the top ========
def detectors(p: np.ndarray) -> Dict[str, np.ndarray]:
    """Every flip-DOWN a chart in this family would draw. Each is a boolean per
    bar, true on the bar the state turns off, and each uses only closes at or
    before that bar."""
    s = pd.Series(p)
    e = {n: s.ewm(span=n, adjust=False, min_periods=n).mean().to_numpy()
         for n in (34, 50, 100, 200)}
    h55 = hma(s, 55).to_numpy()
    h21 = hma(s, 21).to_numpy()
    states = {
        "hull55 rising": np.concatenate([[False], h55[1:] > h55[:-1]]),
        "hma21 over hma55": h21 > h55,
        "close over EMA34": p > e[34],
        "close over EMA50": p > e[50],
        "price>50>100>200": (p > e[50]) & (e[50] > e[100]) & (e[100] > e[200]),
    }
    out: Dict[str, np.ndarray] = {}
    for k, st in states.items():
        st = np.nan_to_num(st, nan=False).astype(bool)
        prev = np.concatenate([[False], st[:-1]])
        out[k] = prev & ~st                       # the bar the state turns off
    return out


def turn_stats(p: np.ndarray, peaks: np.ndarray,
               flips: np.ndarray) -> Dict[str, float]:
    """Recall, precision, lag and give-back for one detector on one name."""
    fi = np.flatnonzero(flips)
    if not len(fi) or not len(peaks):
        return {}
    #  recall: a real top is "caught" if a flip lands in (peak, peak+WINDOW]
    lag: List[float] = []
    give: List[float] = []
    caught = 0
    for q in peaks:
        nxt = fi[(fi > q) & (fi <= q + WINDOW)]
        if len(nxt):
            caught += 1
            lag.append(float(nxt[0] - q))
            give.append(float(1.0 - p[nxt[0]] / p[q]))
    #  precision: a flip is "on time" if a real top sits in [flip-WINDOW, flip)
    ok = sum(1 for f in fi if np.any((peaks < f) & (peaks >= f - WINDOW)))
    return {"n_peaks": len(peaks), "n_flips": len(fi),
            "caught": caught, "on_time": ok,
            "lag_sum": float(np.sum(lag)) if lag else 0.0,
            "lag_n": len(lag),
            "give": give, "lags": lag}


def random_flips(n: int, k: int, rng) -> np.ndarray:
    """A detector with the same flip count and no information at all."""
    out = np.zeros(n, bool)
    if k:
        out[rng.choice(n, size=min(k, n), replace=False)] = True
    return out


def flips_study(P: pd.DataFrame, draws: int = DRAWS) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    acc: Dict[str, Dict] = {}
    for tk, g in P.groupby("ticker", sort=False):
        if len(g) < MIN_BARS:
            continue
        p = g["adj_close"].to_numpy(float)
        piv, conf = pivots_confirmed(p, ZZ)
        if len(piv) < 3:
            continue
        #  swing HIGHS only: a pivot higher than the pivot before it
        hi = np.array([piv[j] for j in range(1, len(piv))
                       if p[piv[j]] > p[piv[j - 1]]], dtype=np.int64)
        if not len(hi):
            continue
        det = detectors(p)
        for nm, fl in det.items():
            st = turn_stats(p, hi, fl)
            if not st:
                continue
            a = acc.setdefault(nm, {"n_peaks": 0, "n_flips": 0, "caught": 0,
                                    "on_time": 0, "give": [], "lags": []})
            for k in ("n_peaks", "n_flips", "caught", "on_time"):
                a[k] += st[k]
            a["give"].extend(st["give"])
            a["lags"].extend(st["lags"])
            #  THE MATCHED NULL: same number of flips, placed at random.
            nul = acc.setdefault(f"NULL {nm}", {"n_peaks": 0, "n_flips": 0,
                                                "caught": 0, "on_time": 0,
                                                "give": [], "lags": []})
            ns = turn_stats(p, hi, random_flips(len(p), st["n_flips"], rng))
            if ns:
                for k in ("n_peaks", "n_flips", "caught", "on_time"):
                    nul[k] += ns[k]
                nul["give"].extend(ns["give"])
                nul["lags"].extend(ns["lags"])
    rows = []
    for nm, a in acc.items():
        if not a["n_peaks"]:
            continue
        rows.append({
            "detector": nm, "peaks": a["n_peaks"], "flips": a["n_flips"],
            "recall": a["caught"] / a["n_peaks"],
            "precision": a["on_time"] / a["n_flips"] if a["n_flips"] else np.nan,
            "lag_med": float(np.median(a["lags"])) if a["lags"] else np.nan,
            "give_med": float(np.median(a["give"])) if a["give"] else np.nan,
            "give_mean": float(np.mean(a["give"])) if a["give"] else np.nan})
    R = pd.DataFrame(rows)
    R["F1"] = 2 * R["recall"] * R["precision"] / (R["recall"] + R["precision"])
    return R.sort_values("F1", ascending=False)


# ======================================= H38 — where the levels should sit ====
def first_passage_level(p: np.ndarray, lvl: np.ndarray, horizon: int,
                        up: bool) -> np.ndarray:
    """Sessions until the path first crosses a PER-BAR level.

    `first_passage` in time_price.py takes a fixed multiple of the entry price;
    a support or resistance line is a different number on every bar, so the
    level has to travel with the row.
    """
    n = len(p)
    out = np.full(n, -1, dtype=np.int64)
    if n < 2:
        return out
    m = min(horizon, n - 1)
    fill = -np.inf if up else np.inf
    pad = np.concatenate([p[1:], np.full(m, fill)])
    W = np.lib.stride_tricks.sliding_window_view(pad, m)[:n]
    hit = W >= lvl[:, None] if up else W <= lvl[:, None]
    any_hit = hit.any(axis=1)
    out[any_hit] = hit.argmax(axis=1)[any_hit] + 1
    tail = np.arange(n) + m >= n
    out[tail & ~any_hit] = -2
    return out


def swing_levels(p: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """The last CONFIRMED swing high and low as of each bar."""
    piv, conf = pivots_confirmed(p, ZZ)
    n = len(p)
    hi = np.full(n, np.nan)
    lo = np.full(n, np.nan)
    for j in range(len(piv)):
        higher = j > 0 and p[piv[j]] > p[piv[j - 1]]
        lower = j > 0 and p[piv[j]] < p[piv[j - 1]]
        c = int(conf[j])
        if higher:
            hi[c:] = p[piv[j]]
        elif lower:
            lo[c:] = p[piv[j]]
    return hi, lo


def place_study(P: pd.DataFrame) -> pd.DataFrame:
    """Target and stop placed at the swing level, and at offsets around it."""
    P = P.copy()
    P["elig"] = eligible(P)
    rows: List[pd.DataFrame] = []
    for tk, g in P.groupby("ticker", sort=False):
        if len(g) < MIN_BARS:
            continue
        p = g["adj_close"].to_numpy(float)
        n = len(p)
        hi, lo = swing_levels(p)
        el = g["elig"].to_numpy() & np.isfinite(hi) & np.isfinite(lo)
        el &= (hi > p) & (lo < p)
        if not el.any():
            continue
        yr = pd.DatetimeIndex(g["date"]).year.to_numpy()
        ups = {o: first_passage_level(p, hi * (1.0 + o), HORIZON, True)
               for o in OFFSETS}
        dns = {o: first_passage_level(p, lo * (1.0 + o), HORIZON, False)
               for o in OFFSETS}
        for ou, a in ups.items():
            for od, b in dns.items():
                ta = np.where(a > 0, a, np.inf)
                tb = np.where(b > 0, b, np.inf)
                first = np.minimum(ta, tb)
                fin = np.isfinite(first)
                ei = np.clip(np.arange(n) + np.where(fin, first, 0), 0,
                             n - 1).astype(np.int64)
                #  Exit at the CLOSE of the bar that breached, never at the
                #  level. A19/H35: a stop is not filled at the number on it.
                real = p[ei] / p - 1.0
                j = np.arange(n) + HORIZON
                term = np.full(n, np.nan)
                ok = j < n
                term[ok] = p[j[ok]] / p[ok] - 1.0
                res = np.where(fin, real, term)
                sel = el & np.isfinite(res) & (a != -2) & (b != -2)
                if not sel.any():
                    continue
                rows.append(pd.DataFrame({
                    "tp_off": ou, "sl_off": od, "year": yr[sel],
                    "ret": res[sel],
                    "tp_dist": (hi[sel] * (1 + ou)) / p[sel] - 1.0,
                    "sl_dist": 1.0 - (lo[sel] * (1 + od)) / p[sel],
                    "hit": np.where(ta[sel] < tb[sel], "tp",
                                    np.where(tb[sel] < ta[sel], "sl", "time"))}))
    B = pd.concat(rows, ignore_index=True)
    B["net"] = B["ret"] - COST
    B["lg"] = np.log1p(np.clip(B["net"], -0.99, None))
    cut = int(B["year"].median())
    out: List[Dict] = []
    for (ou, od), g in B.groupby(["tp_off", "sl_off"]):
        e, l = g[g["year"] < cut], g[g["year"] >= cut]
        out.append({
            "tp_off": ou, "sl_off": od, "n": len(g),
            "tp_dist": float(g["tp_dist"].median()),
            "sl_dist": float(g["sl_dist"].median()),
            "p_tp": float((g["hit"] == "tp").mean()),
            "p_sl": float((g["hit"] == "sl").mean()),
            "mean": float(g["net"].mean()), "median": float(g["net"].median()),
            "meanlog": float(g["lg"].mean()),
            "early": float(e["lg"].mean()), "late": float(l["lg"].mean())})
    return pd.DataFrame(out).sort_values("meanlog", ascending=False)


# ================================================================== main ======
def main(argv: List[str]) -> int:
    modes = argv or ["flips", "place"]
    P = load()
    print(f"panel {len(P):,} rows  {P['ticker'].nunique()} names\n")

    if "flips" in modes:
        R = flips_study(P)
        R.to_csv(os.path.join(OUT, "turns_flips.csv"), index=False)
        print(f"=== H37 catching the top: a real swing high is a confirmed "
              f"{ZZ:.0%} ZigZag peak,")
        print(f"    'caught' means a flip-down within {WINDOW} sessions after it")
        print(f"{'detector':<22}{'peaks':>8}{'flips':>9}{'recall':>8}"
              f"{'precis':>8}{'F1':>7}{'lag':>6}{'giveback':>10}")
        for _, r in R.iterrows():
            print(f"{r['detector']:<22}{int(r['peaks']):>8,}"
                  f"{int(r['flips']):>9,}{r['recall']:>8.3f}"
                  f"{r['precision']:>8.3f}{r['F1']:>7.3f}"
                  f"{r['lag_med']:>6.0f}{r['give_med']:>10.3f}")
        print()

    if "place" in modes:
        T = place_study(P)
        T.to_csv(os.path.join(OUT, "turns_place.csv"), index=False)
        print("=== H38 where to put the target and the stop, relative to the")
        print("    nearest confirmed swing level. 0.00 = exactly at it,")
        print("    -0.05 = five percent short of it (a more conservative target,")
        print("    a tighter stop). Net of 56 bps, exits at the breaching close.")
        print(f"{'tp off':>7}{'sl off':>7}{'tp dist':>9}{'sl dist':>9}"
              f"{'P(tp)':>8}{'P(sl)':>8}{'mean':>9}{'meanlog':>9}"
              f"{'early':>9}{'late':>9}{'both+':>7}")
        for _, r in T.iterrows():
            both = "YES" if r["early"] > 0 and r["late"] > 0 else ""
            print(f"{r['tp_off']:>7.2f}{r['sl_off']:>7.2f}{r['tp_dist']:>9.3f}"
                  f"{r['sl_dist']:>9.3f}{r['p_tp']:>8.3f}{r['p_sl']:>8.3f}"
                  f"{r['mean']:>9.4f}{r['meanlog']:>9.4f}{r['early']:>9.4f}"
                  f"{r['late']:>9.4f}{both:>7}")
        print("\n  marginal effect of each offset, averaged over the other")
        for col, nm in (("tp_off", "target"), ("sl_off", "stop")):
            m = T.groupby(col)[["mean", "meanlog", "p_tp"]].mean()
            print(f"  {nm}:")
            print("   " + m.to_string(float_format=lambda v: f"{v:,.4f}")
                  .replace("\n", "\n   "))
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
