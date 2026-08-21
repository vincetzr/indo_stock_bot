#!/usr/bin/env python3
"""Run the frozen layer-2 hypotheses against the collected panel.

WHAT THIS IS
------------
``layer2_protocol.py`` wrote down four hypotheses, their horizons, the
correction and the stopping rule BEFORE any data existed, and hashed them. This
runs them. It does not invent new ones, it does not try several definitions of
"accumulation" and report the best, and it prints the protocol hash so a result
can be tied to the experiment it belongs to.

The repo has the cautionary tale already: Result 96 found foreign_net at
d = 0.257, p = 0.047 on a 300-event pilot, and the pre-registered replication on
600 untouched events returned d = 0.002. The pilot was the best of several
things looked at, on a sample small enough for that to happen by luck.

THREE THINGS THAT WOULD OTHERWISE FAKE SIGNIFICANCE HERE
--------------------------------------------------------
1. TREATING TICKER-DAYS AS INDEPENDENT. Ten names on one day are not ten
   observations - they move together. Every test here collapses to a DAY-level
   mean first and tests across days, which is the clustering the protocol
   demands and costs roughly a factor of three in effective sample size.
2. MEASURING RETURN FROM A CLOSE YOU COULD NOT TRADE. The broker summary is
   published in the evening. Entry is therefore the close of t+1 at the earliest
   - the whole of t+1 is available to act in, and the reaction that happens
   during t+1 is NOT credited to the signal.
3. TREATING A CENSORED NUMBER AS EXACT. Nine of ten flow figures are bracketed,
   not measured. Every hypothesis is run three times - at the bottom of the
   bracket, the middle and the top - and a conclusion that flips between them is
   reported as undetermined rather than as whichever version was nicer.

    python3 scripts/layer2_test.py
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))
sys.path.insert(0, os.path.dirname(__file__))

from idxbot.config import load_config                     # noqa: E402
from idxbot.data.cache import Cache                       # noqa: E402
from idxbot.data.ohlcv import YahooOHLCV                  # noqa: E402
from idxbot.broker_bounds import day_bounds               # noqa: E402
from layer2_protocol import (ALPHA, HYPOTHESES, POWER,     # noqa: E402
                             detectable_effect, protocol_hash, required_n)
from factor_study import total_return_series               # noqa: E402

STORE = os.path.join("data", "cache", "broker_daily")


# --------------------------------------------------------------------------
# panel assembly
# --------------------------------------------------------------------------
def load_store(view: str = "ipot-all") -> pd.DataFrame:
    """Every stored ticker-day for one view."""
    files = sorted(glob.glob(os.path.join(STORE, f"*_{view}.csv.gz")))
    out = []
    for f in files:
        try:
            d = pd.read_csv(f)
        except Exception:                                  # noqa: BLE001
            continue
        if not d.empty:
            out.append(d)
    if not out:
        return pd.DataFrame()
    df = pd.concat(out, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    return df


def signals_for_day(g: pd.DataFrame, which: str) -> Dict[str, float]:
    """The four frozen signals for one ticker-day, at three censoring levels.

    ``which`` selects the bracket level - ``lo``, ``mid`` or ``hi`` - so the
    same hypothesis can be re-run at each and a conclusion that depends on the
    censoring can be caught being a conclusion about the censoring.
    """
    b = day_bounds(g)
    if b.empty:
        return {}
    net = {"lo": b["net_lo"], "mid": (b["net_lo"] + b["net_hi"]) / 2.0,
           "hi": b["net_hi"]}[which]
    b = b.assign(net=net).sort_values("net", ascending=False)

    total = float(pd.to_numeric(g.get("total_lot"), errors="coerce").dropna().iloc[0]) \
        if "total_lot" in g and pd.to_numeric(g["total_lot"], errors="coerce").notna().any() \
        else float(g["buy_lot"].sum())
    if not np.isfinite(total) or total <= 0:
        return {}

    top3_net = float(b["net"].head(3).sum()) / total          # H1, scaled
    buy = pd.to_numeric(g["buy_lot"], errors="coerce").fillna(0.0)
    conc = float(buy.nlargest(3).sum() / total) if total > 0 else np.nan  # H2
    fnv = pd.to_numeric(pd.Series(g.get("foreign_net_val")),
                        errors="coerce").dropna()
    tval = pd.to_numeric(pd.Series(g.get("total_val")),
                         errors="coerce").dropna()
    # H3 scaled by the day's value so a big stock does not dominate by size
    fnet = float(fnv.iloc[0] / tval.iloc[0]) if len(fnv) and len(tval) \
        and float(tval.iloc[0]) > 0 else np.nan
    top_buyer = str(b["broker"].iloc[0]) if len(b) else ""
    return {"top3_net": top3_net, "concentration": conc,
            "foreign_net": fnet, "top_buyer": top_buyer}


def build_panel(view: str = "ipot-all", which: str = "mid") -> pd.DataFrame:
    df = load_store(view)
    if df.empty:
        return df
    rows = []
    for (tk, dt), g in df.groupby(["ticker", "date"]):
        s = signals_for_day(g, which)
        if s:
            s.update({"ticker": tk, "date": dt})
            rows.append(s)
    P = pd.DataFrame(rows).sort_values(["ticker", "date"])
    if P.empty:
        return P
    # H4: the same broker top net buyer for a third consecutive session
    P["streak3"] = False
    for tk, g in P.groupby("ticker"):
        tb = g["top_buyer"].to_numpy()
        run = np.zeros(len(tb), dtype=bool)
        for i in range(2, len(tb)):
            if tb[i] and tb[i] == tb[i - 1] == tb[i - 2]:
                run[i] = True
        P.loc[g.index, "streak3"] = run
    return P.reset_index(drop=True)


def attach_returns(P: pd.DataFrame, loader: YahooOHLCV) -> pd.DataFrame:
    """Forward EXCESS returns at each frozen horizon, executable at t+1's close.

    The signal is published in the evening of t. Entry is the close of t+1, so
    the reaction during t+1 is deliberately not credited to the signal - a
    generous convention would enter at t's close, which nobody could have done.
    """
    px = {}
    for tk in sorted(P["ticker"].unique()):
        d = total_return_series(loader, tk, total=True)
        if d is not None:
            px[tk] = d["px"]
    horizons = sorted({h["horizon"] for h in HYPOTHESES})
    for h in horizons:
        P[f"fwd{h}"] = np.nan
    for tk, g in P.groupby("ticker"):
        if tk not in px:
            continue
        s = px[tk]
        pos = {d: i for i, d in enumerate(s.index)}
        for h in horizons:
            vals = []
            for d in g["date"]:
                i = pos.get(pd.Timestamp(d))
                if i is None or i + 1 + h >= len(s):
                    vals.append(np.nan)
                    continue
                entry, exit_ = s.iloc[i + 1], s.iloc[i + h]
                vals.append(exit_ / entry - 1.0 if entry > 0 and h > 1 else np.nan)
            P.loc[g.index, f"fwd{h}"] = vals
    # Excess = minus the equal-weight mean of the panel that day, so a signal
    # cannot win by simply being long on days the whole market rose.
    #
    # With a thin panel that subtraction is degenerate - on ONE name the mean is
    # the name itself and every excess return is exactly zero, which would make
    # a broken test look like a null result. Below five names the IHSG stands in
    # as the market instead.
    n_names = P["ticker"].nunique()
    bench = None
    if n_names < 5:
        try:
            from factor_study import load_index
            idx = load_index(loader, "^JKSE", pd.DatetimeIndex(
                sorted(P["date"].unique())))
            bench = idx.dropna()
        except Exception:                                  # noqa: BLE001
            bench = None
    for h in horizons:
        if n_names >= 5:
            m = P.groupby("date")[f"fwd{h}"].transform("mean")
        elif bench is not None and len(bench) > h + 2:
            b = bench.sort_index()
            pos = {d: i for i, d in enumerate(b.index)}
            mv = []
            for d in P["date"]:
                i = pos.get(pd.Timestamp(d))
                mv.append(float(b.iloc[i + h] / b.iloc[i + 1] - 1.0)
                          if i is not None and i + h < len(b) and i + 1 < len(b)
                          and h > 1 else np.nan)
            m = pd.Series(mv, index=P.index)
        else:
            m = 0.0
        P[f"ex{h}"] = P[f"fwd{h}"] - m
    P.attrs["benchmark"] = ("panel equal-weight" if n_names >= 5
                            else "IHSG (panel too thin)")
    return P


# --------------------------------------------------------------------------
# the test
# --------------------------------------------------------------------------
def cluster_by_day(events: pd.DataFrame, col: str) -> np.ndarray:
    """Collapse ticker-days to one number a day. This is the whole ballgame.

    Ten names on one session are not ten observations. Skipping this step is the
    single easiest way to manufacture significance in a flow study, and it would
    roughly triple the apparent sample.
    """
    if events.empty:
        return np.array([])
    return events.groupby("date")[col].mean().dropna().to_numpy()


def one_sided(x: np.ndarray, direction: str = "positive") -> Dict[str, float]:
    """One-sided t on the day-level series, with Cohen's d."""
    x = np.asarray([v for v in x if np.isfinite(v)])
    n = len(x)
    if n < 8:
        return {"n": n, "mean": np.nan, "d": np.nan, "t": np.nan, "p": np.nan}
    sd = float(np.std(x, ddof=1))
    d = float(np.mean(x) / sd) if sd > 0 else np.nan
    t = float(np.mean(x) / (sd / np.sqrt(n))) if sd > 0 else np.nan
    p = float(1.0 - stats.t.cdf(t, n - 1)) if np.isfinite(t) else np.nan
    if direction == "negative":
        p = float(stats.t.cdf(t, n - 1)) if np.isfinite(t) else np.nan
    return {"n": n, "mean": float(np.mean(x)), "d": d, "t": t, "p": p}


def run_hypothesis(P: pd.DataFrame, h: Dict, top_q: float = 0.8) -> Dict:
    """Events are the top quintile of the signal; the rest is the control."""
    col = {"H1": "top3_net", "H2": "concentration",
           "H3": "foreign_net", "H4": "streak3"}[h["id"]]
    ex = f"ex{h['horizon']}"
    if col not in P or ex not in P:
        return {"id": h["id"], "n": 0}
    d = P.dropna(subset=[ex])
    if d.empty:
        return {"id": h["id"], "n": 0}
    if col == "streak3":
        events = d[d[col]]
    else:
        thr = d[col].quantile(top_q)
        events = d[d[col] >= thr]
    daily = cluster_by_day(events, ex)
    out = one_sided(daily, h["direction"])
    out.update({"id": h["id"], "claim": h["claim"], "horizon": h["horizon"],
                "events": len(events), "signal": col})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--view", default="ipot-all")
    ap.add_argument("--levels", default="lo,mid,hi")
    args = ap.parse_args()

    cfg = load_config()
    loader = YahooOHLCV(cfg, Cache(cfg.path("data.cache_dir", "data/cache")))
    alpha_c = ALPHA / len(HYPOTHESES)

    print(f"{'=' * 94}\n LAYER-2 TEST — protocol {protocol_hash()}\n{'=' * 94}")
    print(" The hypotheses, horizons, correction and stopping rule were fixed "
          "before any\n data existed. This runs them and nothing else.\n")

    base = build_panel(args.view, "mid")
    if base.empty:
        print(" the store is empty for this view.")
        return 1
    base = attach_returns(base, loader)
    days = base["date"].nunique()
    names = base["ticker"].nunique()
    deff = 1.0 + (names - 1) * 0.30
    eff_n = days * names / deff
    mde = detectable_effect(int(eff_n), alpha_c, POWER)

    print(f" panel: {names} names x {days} sessions = {len(base):,} ticker-days")
    print(f"        {base['date'].min():%Y-%m-%d} to {base['date'].max():%Y-%m-%d}")
    print(f" clustering by day at ICC 0.30 gives a design effect of {deff:.1f},")
    print(f" so the effective sample is about {eff_n:,.0f}, and the smallest "
          f"effect this\n could find at 80% power is d = {mde:.3f}.")
    if mde > 0.20:
        print(f"\n ! UNDERPOWERED. The protocol's stopping rule says do not "
              f"report. Anything\n ! below d = {mde:.2f} is invisible here, and "
              f"a real flow edge is nearer 0.10.")

    print(f"\n{'=' * 94}\n RESULTS — Bonferroni alpha {alpha_c:.4f}, one-sided\n"
          f"{'=' * 94}")
    print(f" {'':<4}{'signal':<16}{'h':>3}{'level':>7}{'days':>7}{'events':>8}"
          f"{'mean ex':>10}{'d':>8}{'t':>7}{'p':>9}  verdict")
    verdicts: Dict[str, List[bool]] = {}
    for h in HYPOTHESES:
        for lvl in args.levels.split(","):
            P = attach_returns(build_panel(args.view, lvl), loader) \
                if lvl != "mid" else base
            r = run_hypothesis(P, h)
            if not r.get("n"):
                print(f" {h['id']:<4}{'—':<16}{h['horizon']:>3}{lvl:>7}"
                      f"{'no data':>7}")
                continue
            sig = np.isfinite(r["p"]) and r["p"] < alpha_c
            verdicts.setdefault(h["id"], []).append(bool(sig))
            print(f" {h['id']:<4}{r['signal']:<16}{h['horizon']:>3}{lvl:>7}"
                  f"{r['n']:>7}{r['events']:>8}{r['mean']:>10.4%}"
                  f"{r['d']:>8.3f}{r['t']:>7.2f}{r['p']:>9.4f}  "
                  f"{'SIGNIFICANT' if sig else 'not significant'}")

    print(f"\n{'=' * 94}\n READING\n{'=' * 94}")
    for hid, v in verdicts.items():
        if all(v):
            print(f" {hid}: holds at every censoring level.")
        elif any(v):
            print(f" {hid}: DEPENDS ON THE CENSORING — significant at some "
                  f"bracket levels and not\n     others, which makes it a "
                  f"conclusion about the missing data, not the flow.")
        else:
            print(f" {hid}: not significant at any censoring level.")
    if not any(any(v) for v in verdicts.values()):
        print("\n No frozen hypothesis survives. That is a result, not a "
              "failure to find one -\n and it is the fourth time this repo has "
              "reached it from a different angle.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
