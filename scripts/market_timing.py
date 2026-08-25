#!/usr/bin/env python3
"""Can a weekly rule get you out before an IDX correction and back in after?

NEVER TESTED IN THIS REPO. H9-H21 all ask which NAMES to hold; none asks
whether to be in the market at all. The user asked directly, and answering
"no" without measuring would be the A12 failure this project has recorded six
times. So the rules below are fixed before scoring, and the null is a random
switcher with the SAME number of trades — because a rule that is out of the
market a third of the time will dodge a third of the crashes by construction.

COST. Switching an index position costs the A5 schedule, 0.56% round trip. A
0% column is printed too, because an index mutual fund can be cheaper, and if
the rule fails at zero cost it fails everywhere.
"""

from __future__ import annotations

import os
import sys
from typing import Callable, Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

INDEX = os.path.join("data", "cache", "ohlcv", "_JKSE.csv.gz")
FEE = 0.0056
START = "2004-01-01"


def load() -> pd.DataFrame:
    ix = pd.read_csv(INDEX)
    ix["date"] = pd.to_datetime(ix["date"], utc=True,
                                errors="coerce").dt.tz_localize(None)
    d = ix.set_index("date")["close"].astype(float).sort_index().dropna()
    d = d[(d > 0) & (d.index >= START)]
    F = pd.DataFrame({"close": d})
    F["ret"] = F["close"].pct_change()
    for w in (20, 50, 100, 200):
        F[f"ma{w}"] = F["close"].rolling(w).mean()
    F["hi252"] = F["close"].rolling(252, min_periods=60).max()
    F["dd"] = F["close"] / F["hi252"] - 1.0
    F["mom252"] = F["close"] / F["close"].shift(252) - 1.0
    F["mom20"] = F["close"] / F["close"].shift(20) - 1.0
    F["vol20"] = F["ret"].rolling(20).std() * np.sqrt(252)
    return F


#: Every signal is computed from bars up to and including t and acted on at
#: t+1's close. `.shift(1)` below is what enforces that; without it every rule
#: below reads as brilliant.
RULES: Dict[str, Callable[[pd.DataFrame], pd.Series]] = {
    "always in (buy & hold)":  lambda F: pd.Series(True, index=F.index),
    "above 200d MA":           lambda F: F["close"] > F["ma200"],
    "above 100d MA":           lambda F: F["close"] > F["ma100"],
    "above 50d MA":            lambda F: F["close"] > F["ma50"],
    "50d above 200d (golden)": lambda F: F["ma50"] > F["ma200"],
    "12m momentum > 0":        lambda F: F["mom252"] > 0,
    "1m momentum > 0":         lambda F: F["mom20"] > 0,
    "drawdown < 5%":           lambda F: F["dd"] > -0.05,
    "drawdown < 10%":          lambda F: F["dd"] > -0.10,
    "calm (vol20 < median)":   lambda F: F["vol20"] < F["vol20"].expanding(
        min_periods=250).median(),
}


def run(F: pd.DataFrame, sig: pd.Series, fee: float = FEE) -> Dict:
    """Compound the index while `sig` says in, cash while out, charging fee."""
    pos = sig.shift(1).fillna(False).astype(bool)       # ACT NEXT BAR
    r = F["ret"].fillna(0.0).to_numpy()
    p = pos.to_numpy()
    switch = np.zeros(len(p), bool)
    switch[1:] = p[1:] != p[:-1]
    eq, peak, mdd, out = 1.0, 1.0, 0.0, []
    for i in range(len(p)):
        if switch[i]:
            eq *= (1.0 - fee / 2.0)        # half a round trip per switch
        if p[i]:
            eq *= (1.0 + r[i])
        peak = max(peak, eq)
        mdd = min(mdd, eq / peak - 1.0)
        out.append(eq)
    yrs = (F.index[-1] - F.index[0]).days / 365.25
    return {"cagr": eq ** (1.0 / yrs) - 1.0, "maxdd": mdd, "terminal": eq,
            "switches": int(switch.sum()), "in_frac": float(p.mean()),
            "equity": pd.Series(out, index=F.index)}


def null_matched(F: pd.DataFrame, n_switch: int, in_frac: float,
                 draws: int = 200, seed: int = 7) -> Dict:
    """A RANDOM switcher with the same trade count and the same time in market.

    Without this the table is unreadable: any rule out of the market 30% of the
    time dodges 30% of the crashes by construction, so a shallower drawdown is
    not evidence of anything.
    """
    rng = np.random.default_rng(seed)
    n = len(F)
    cg, dd = [], []
    for _ in range(draws):
        p = np.zeros(n, bool)
        cuts = np.sort(rng.choice(np.arange(1, n), size=max(n_switch, 1),
                                  replace=False))
        state = rng.random() < in_frac
        prev = 0
        for c in cuts:
            p[prev:c] = state
            state = not state
            prev = c
        p[prev:] = state
        s = run(F, pd.Series(p, index=F.index).shift(-1).fillna(False))
        cg.append(s["cagr"])
        dd.append(s["maxdd"])
    return {"cagr_mean": float(np.mean(cg)), "cagr_p95": float(
        np.percentile(cg, 95)), "maxdd_mean": float(np.mean(dd))}


def main() -> int:
    F = load()
    W = 100
    print("=" * W)
    print(" CAN A TIMING RULE BEAT HOLDING THE IHSG?")
    print("=" * W)
    print(f" {len(F):,} sessions {F.index[0].date()} .. {F.index[-1].date()}")
    print(" Signal from bars <= t, acted at t+1 close. 0.56% per round trip,")
    print(" charged as 0.28% per switch.\n")

    base = run(F, RULES["always in (buy & hold)"](F))
    print(f"   {'rule':<26}{'CAGR':>8}{'no-fee':>9}{'maxDD':>8}"
          f"{'switch':>8}{'in %':>7}{'vs hold':>9}")
    res = {}
    for nm, fn in RULES.items():
        s = run(F, fn(F).fillna(False))
        s0 = run(F, fn(F).fillna(False), fee=0.0)
        res[nm] = s
        print(f"   {nm:<26}{s['cagr']:>+8.2%}{s0['cagr']:>+9.2%}"
              f"{s['maxdd']:>+8.1%}{s['switches']:>8}"
              f"{s['in_frac']:>7.0%}{s['cagr'] - base['cagr']:>+9.2%}")

    print("\n THE MATCHED-NULL COLUMN, which decides it. A rule out of the")
    print(" market 30% of the time dodges 30% of the crashes for free.\n")
    print(f"   {'rule':<26}{'CAGR':>8}{'null mean':>11}{'null p95':>10}"
          f"{'beats null?':>13}")
    for nm, s in res.items():
        if nm.startswith("always"):
            continue
        nu = null_matched(F, s["switches"], s["in_frac"])
        ok = "yes" if s["cagr"] > nu["cagr_p95"] else "no"
        print(f"   {nm:<26}{s['cagr']:>+8.2%}{nu['cagr_mean']:>+11.2%}"
              f"{nu['cagr_p95']:>+10.2%}{ok:>13}")

    print("\n" + "=" * W)
    print(" HALF-SPLIT — a rule that works in one era is not a rule")
    print("=" * W)
    mid = F.index[len(F) // 2]
    print(f"   early {F.index[0].date()} .. {mid.date()}, "
          f"late {mid.date()} .. {F.index[-1].date()}\n")
    print(f"   {'rule':<26}{'early vs hold':>15}{'late vs hold':>14}"
          f"{'both?':>8}")
    for nm, fn in RULES.items():
        if nm.startswith("always"):
            continue
        a, b = F[F.index <= mid], F[F.index > mid]
        da = run(a, fn(F).reindex(a.index).fillna(False))["cagr"] - \
            run(a, pd.Series(True, index=a.index))["cagr"]
        db = run(b, fn(F).reindex(b.index).fillna(False))["cagr"] - \
            run(b, pd.Series(True, index=b.index))["cagr"]
        print(f"   {nm:<26}{da:>+15.2%}{db:>+14.2%}"
              f"{('YES' if da > 0 and db > 0 else 'no'):>8}")

    print("\n" + "=" * W)
    print(" THE CONDITIONAL ACTUALLY ASKED FOR: given today's state, what is")
    print(" P(the index falls a further 5% within 20 sessions)?")
    print("=" * W)
    fwd_min = F["close"][::-1].rolling(21, min_periods=2).min()[::-1]
    F = F.assign(fwd_worst=fwd_min / F["close"] - 1.0)
    G = F.dropna(subset=["fwd_worst", "dd", "ma200"])
    base_rate = float((G["fwd_worst"] <= -0.05).mean())
    print(f"\n   unconditional base rate: {base_rate:.1%}"
          f"   (n = {len(G):,} sessions)\n")
    print(f"   {'state':<34}{'n':>7}{'P(-5% in 20d)':>16}{'vs base':>10}")
    states = {
        "above 200d MA": G["close"] > G["ma200"],
        "below 200d MA": G["close"] <= G["ma200"],
        "at/near 52w high (dd > -2%)": G["dd"] > -0.02,
        "dd -2% to -10%": (G["dd"] <= -0.02) & (G["dd"] > -0.10),
        "dd -10% to -20%": (G["dd"] <= -0.10) & (G["dd"] > -0.20),
        "dd worse than -20%": G["dd"] <= -0.20,
        "vol20 top quartile": G["vol20"] > G["vol20"].quantile(0.75),
        "vol20 bottom quartile": G["vol20"] < G["vol20"].quantile(0.25),
    }
    for nm, m in states.items():
        if m.sum() < 200:
            continue
        p = float((G.loc[m, "fwd_worst"] <= -0.05).mean())
        print(f"   {nm:<34}{int(m.sum()):>7,}{p:>16.1%}"
              f"{100 * (p - base_rate):>+7.1f} pp")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
