#!/usr/bin/env python3
"""H20 — what you actually earn, and whether the entry earns any of it.

    python3 scripts/portfolio.py            # build (slow), then report
    python3 scripts/portfolio.py --report   # reuse the cached per-name table

TWO CRITIQUES OF H17-H19, BOTH MINE, BOTH UNTESTED UNTIL NOW
--------------------------------------------------------------
**The cohort MEDIAN is not what an investor receives.** Hold twelve names
equal-weighted and your return is their MEAN. H17 and H18 both selected their
exit rule by maximising the average cohort median, and H18's own objective table
already shows that on the mean NO rule in the catalogue beats buy-and-hold. So
the headline improvements (+4.13%, +6.35%) are improvements in a statistic
nobody is paid.

**The entry may contribute nothing.** H16 measured p = 0.211 for doublers
against random ten-name draws from the same liquid universe — indistinguishable
from a coin. If that is right, every exit result so far is a statement about
trailing stops on volatile Indonesian small caps in general, not about this
strategy, and the correct control has never been run.

PRE-REGISTERED, BEFORE ANY OF THIS WAS SCORED
-----------------------------------------------
  H20a  The exit improvement over buy-and-hold LARGELY SURVIVES on random
        baskets drawn from the same eligible universe, because it is a property
        of fat-tailed names rather than of the selection. Predicted: more than
        half the gap survives. If it does, the exit finding is real but general,
        and the entry is not carrying it.

  H20b  On PORTFOLIO accounting — equal weight within a basket, capital
        redeployed when the exit fires — no rule beats buy-and-hold on terminal
        wealth, but every rule reduces maximum drawdown. This is the
        risk-preference reading of H18's objective table, tested directly.

  H20c  The GROWTH-OPTIMAL rule (maximising mean log wealth relative, the
        criterion for repeated reinvestment) is neither the median-optimal nor
        the mean-optimal rule, and sits between the trail and plain holding.
        Log utility is the principled objective here because it penalises the
        -50% outcomes at the rate compounding actually penalises them, instead
        of me choosing between median and mean by taste.

HOW THE PORTFOLIO IS SIMULATED
--------------------------------
Twelve slots, offset one month apart. A slot buys a basket at a cohort date,
holds until the rule exits, then sits in cash until the next monthly cohort
date and buys again. That rewards an early exit exactly as reality does — the
capital is freed and redeployed — and charges the cost of every extra round
trip, which a per-cohort statistic silently ignores. Slot equity curves are
compounded independently and reported as a distribution, because twelve
overlapping slots are not twelve independent samples.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

from idxbot.spine import exits as X                              # noqa: E402
from idxbot.spine import multiplier as MU                        # noqa: E402

PANEL = os.path.join("data", "spine", "price_panel.parquet")
IND = os.path.join("data", "spine", "indicator_panel.parquet")
CACHE = os.path.join("data", "spine", "portfolio_names.parquet")

#: Rules carried forward. The H17 incumbent, the H18 walk-forward pick, the
#: H19-endorsed one, two references and a floor.
def rules() -> Dict[str, object]:
    P, I = X.catalogue(), X.indicator_catalogue()
    return {
        "hold 252": P["hold 252"],
        "trail 15% armed +50%": P["trail 15% armed +50%"],
        "trail 30% armed +50%": P["trail 30% armed +50%"],
        "chandelier 2x ATR armed +50%": I["chandelier 2x ATR armed +50%"],
        "stoch rollover armed +50%": I["stoch rollover armed +50%"],
        "ema50 break armed +50%": I["ema50 break armed +50%"],
        "stop 25%": P["stop 25%"],
    }


#: How many eligible names to price per cohort. The multiplier basket is always
#: included; the rest form the pool the random control draws from.
POOL = 300


def build(P: pd.DataFrame, I: pd.DataFrame, start="2002-01-01") -> pd.DataFrame:
    """Per-name net return under every rule, for every eligible name and date.

    Priced ONCE per (cohort, name, rule) so that any basket — the multiplier
    pick or a random draw — is a cheap subset average afterwards. Building it
    per basket instead would have made the random control unaffordable, which
    is exactly how a missing control stays missing.
    """
    cells, tab = MU.build_cells(P)
    pre_end = P.loc[~P["holdout"].astype(bool), "date"].max()
    last = pre_end - pd.Timedelta(days=int(X.HORIZON * 1.5))
    R = rules()
    rows: List[Dict] = []
    rng = np.random.default_rng(20260825)

    for d in pd.date_range(start, last, freq="MS"):
        day, M = MU.rank_live(P, d, cells, tab)
        if day is None or len(M) < MU.TOP_N:
            continue
        picked = set(MU.select(M, MU.TOP_N, "all")["ticker"])
        pool = list(M["ticker"])
        if len(pool) > POOL:                      # keep the picks, sample rest
            rest = [t for t in pool if t not in picked]
            keep = list(rng.choice(rest, size=max(POOL - len(picked), 0),
                                   replace=False))
            pool = list(picked) + keep
        pm = MU.path_map(P, day, pool)
        fm = MU.feature_map(I, day, pool)
        for t, (path, cost) in pm.items():
            F = fm.get(t)
            rec = {"as_of": day, "ticker": t, "picked": t in picked,
                   "cost": cost}
            for name, fn in R.items():
                acc, req = X.rule_arity(fn)
                if req and F is None:
                    rec[name] = np.nan
                    rec[name + "|held"] = np.nan
                    continue
                r, held = fn(path, F) if acc else fn(path)
                rec[name] = r - cost
                rec[name + "|held"] = held
            rows.append(rec)
    return pd.DataFrame(rows)


def baskets(D: pd.DataFrame, rule: str, mode: str = "picked",
            size: int = 12, seed: int = 0) -> pd.DataFrame:
    """One row per cohort: equal-weighted basket net return and holding period.

    ``mode="picked"`` is the multiplier-cell basket; ``mode="random"`` draws
    the same number of names from the same eligible pool, which is the control
    that decides whether the entry contributes anything at all.
    """
    rng = np.random.default_rng(seed)
    out = []
    for day, g in D.groupby("as_of", sort=True):
        g = g[np.isfinite(g[rule])]
        if g.empty:
            continue
        if mode == "picked":
            b = g[g["picked"]]
        else:
            n = min(size, len(g))
            b = g.iloc[rng.choice(len(g), size=n, replace=False)]
        if len(b) < 3:
            continue
        out.append({"as_of": day, "n": len(b),
                    "ret": float(b[rule].mean()),            # WHAT YOU EARN
                    "med": float(b[rule].median()),
                    "logret": float(np.log1p(np.maximum(
                        b[rule].mean(), -0.99))),
                    "p2": float((b[rule] >= 1.0).mean()),
                    "pdn": float((b[rule] <= -0.5).mean()),
                    "held": float(b[rule + "|held"].mean())})
    return pd.DataFrame(out)


def slots(B: pd.DataFrame, n_slots: int = 12) -> pd.DataFrame:
    """Compound twelve month-offset slots that redeploy when the exit fires.

    A slot buys at a cohort date, is locked for ``held`` sessions, then buys at
    the next cohort date on or after it is free. Early exits therefore EARN
    something — the capital comes back — and pay for it in extra round trips,
    both of which a per-cohort average hides.
    """
    if B is None or B.empty or "as_of" not in B.columns:
        return pd.DataFrame()          # an unscoreable rule, not a crash
    B = B.sort_values("as_of").reset_index(drop=True)
    dates = pd.DatetimeIndex(B["as_of"])
    out = []
    for s in range(n_slots):
        i, eq, peak, mdd, trades = s, 1.0, 1.0, 0.0, 0
        path = []
        while i < len(B):
            r = B.iloc[i]
            eq *= (1.0 + r["ret"])
            trades += 1
            peak = max(peak, eq)
            mdd = min(mdd, eq / peak - 1.0)
            path.append((r["as_of"], eq))
            # freed after `held` sessions -> next cohort date at or after that
            # LOCK IN COHORT-INDEX SPACE, NOT ON THE CALENDAR. Converting
            # sessions to calendar days and searching the date index skips a
            # cohort whenever the arithmetic lands just past it: a 30-day lock
            # opened on 1 February ends 3 March and misses the 1 March entry
            # entirely. That penalty falls hardest on SHORT-holding rules,
            # which is precisely the comparison this study exists to make.
            # ~21 trading sessions to the month, rounded up.
            i += max(1, int(np.ceil(float(r["held"]) / 21.0)))
        if trades < 5:
            continue
        #  the LAST trade is still running for its own holding period, so the
        #  slot's span ends after its final entry, not on it. Recorded as an
        #  explicit end date because a benchmark compared over [first entry,
        #  last entry] would be short by that final year.
        end = path[-1][0] + pd.Timedelta(days=365.25)
        yrs = (end - path[0][0]).days / 365.25
        out.append({"slot": s, "trades": trades, "terminal": eq,
                    "cagr": eq ** (1.0 / yrs) - 1.0 if eq > 0 else -1.0,
                    "maxdd": mdd, "years": yrs,
                    "start": path[0][0], "end": end})
    return pd.DataFrame(out)


def paired(D: pd.DataFrame, rule: str, base: str = "hold 252",
           mode: str = "picked", draws: int = 1,
           mode_b: str = None) -> Dict:
    """Slot-by-slot CAGR difference against a baseline.

    THE SLOTS ARE PAIRED AND COMPARING THEIR RANGES THROWS THAT AWAY. Slot 3
    under two rules holds overlapping names over identical dates, so most of
    the cross-slot spread is common to both and cancels in the difference.
    Reading two overlapping 10-90% bands as "no difference" is the same error
    as reading two overlapping confidence intervals that way.
    """
    ds = []
    for s in range(draws):
        A = slots(baskets(D, rule, mode, seed=s))
        B = slots(baskets(D, base, mode_b or mode, seed=s))
        if A.empty or B.empty:
            continue
        M = A.merge(B, on="slot", suffixes=("_r", "_b"))
        if M.empty:
            continue
        ds.append(M["cagr_r"] - M["cagr_b"])
    if not ds:
        return {}
    d = pd.concat(ds)
    n = len(d)
    return {"n": n, "mean": float(d.mean()), "sd": float(d.std(ddof=1)),
            "wins": int((d > 0).sum()),
            "t": float(d.mean() / (d.std(ddof=1) / np.sqrt(max(n, 1))))
            if d.std(ddof=1) > 0 else np.nan,
            "lo": float(d.quantile(.1)), "hi": float(d.quantile(.9))}


def summarise(D: pd.DataFrame, rule: str, mode: str, draws: int = 1) -> Dict:
    accs = []
    for s in range(draws):
        B = baskets(D, rule, mode, seed=s)
        if B.empty:
            continue
        S = slots(B)
        if S.empty:
            continue
        accs.append({
            "mean": B["ret"].mean(), "median": B["med"].mean(),
            "log": B["logret"].mean(), "p2": B["p2"].mean(),
            "pdn": B["pdn"].mean(), "held": B["held"].mean(),
            "cagr": S["cagr"].median(), "maxdd": S["maxdd"].median(),
            "terminal": S["terminal"].median(), "trades": S["trades"].mean(),
            "cagr_lo": S["cagr"].quantile(.1),
            "cagr_hi": S["cagr"].quantile(.9)})
    if not accs:
        return {}
    A = pd.DataFrame(accs)
    return {k: float(A[k].mean()) for k in A.columns} | {
        "cagr_sd_across_draws": float(A["cagr"].std(ddof=1))
        if len(A) > 1 else 0.0}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default=PANEL)
    ap.add_argument("--indicators", default=IND)
    ap.add_argument("--cache", default=CACHE)
    ap.add_argument("--report", action="store_true",
                    help="reuse the cached per-name table")
    ap.add_argument("--draws", type=int, default=20,
                    help="random-basket draws for the control")
    a = ap.parse_args()
    a_draws = a.draws

    if a.report and os.path.exists(a.cache):
        D = pd.read_parquet(a.cache)
        D["as_of"] = pd.to_datetime(D["as_of"])
        print(f" [per-name table loaded from {a.cache}]")
    else:
        P = pd.read_parquet(a.panel)
        P["date"] = pd.to_datetime(P["date"])
        P = P.sort_values(["ticker", "date"])
        I = pd.read_parquet(a.indicators)
        I["date"] = pd.to_datetime(I["date"])
        D = build(P, I)
        D.to_parquet(a.cache, index=False)
        print(f" [per-name table cached to {a.cache}]")

    R = rules()
    print("=" * 96)
    print(" H20 — PORTFOLIO ACCOUNTING AND THE RANDOM-ENTRY CONTROL")
    print("=" * 96)
    print(f" {D['as_of'].nunique()} cohorts, {len(D):,} priced name-cohorts, "
          f"{D['ticker'].nunique()} names")
    print(f" {int(D['picked'].sum()):,} of them are multiplier picks; the rest "
          f"are the random pool\n")

    print(" PER-COHORT, multiplier basket — mean is what you earn, median is")
    print(" what H17/H18 optimised. Note where they disagree.\n")
    print(f"   {'rule':<32}{'MEAN':>9}{'median':>9}{'mean log':>10}"
          f"{'P(2x)':>8}{'P(-50%)':>9}{'days':>6}")
    got = {}
    for name in R:
        s = summarise(D, name, "picked")
        if not s:
            continue
        got[name] = s
        print(f"   {name:<32}{s['mean']:>+9.1%}{s['median']:>+9.1%}"
              f"{s['log']:>+10.3f}{s['p2']:>8.1%}{s['pdn']:>9.1%}"
              f"{s['held']:>6.0f}")

    print("\n H20b — PORTFOLIO: 12 month-offset slots, redeployed on exit")
    print(f"   {'rule':<32}{'CAGR':>8}{'10-90% across slots':>22}"
          f"{'max DD':>9}{'x terminal':>12}{'trades':>8}")
    for name, s in got.items():
        print(f"   {name:<32}{s['cagr']:>+8.1%}"
              f"   [{s['cagr_lo']:>+6.1%}, {s['cagr_hi']:>+6.1%}]"
              f"{s['maxdd']:>9.1%}{s['terminal']:>12.1f}{s['trades']:>8.0f}")

    print(f"\n H20a — RANDOM-ENTRY CONTROL ({a.draws} draws from the same pool)")
    print(f"   {'rule':<32}{'MEAN':>9}{'log':>9}{'CAGR':>9}{'sd':>7}"
          f"   vs multiplier basket")
    ctrl = {}
    for name in R:
        s = summarise(D, name, "random", draws=a.draws)
        if not s:
            continue
        ctrl[name] = s
        p = got.get(name, {})
        d = (p.get("cagr", np.nan) - s["cagr"])
        print(f"   {name:<32}{s['mean']:>+9.1%}{s['log']:>+9.3f}"
              f"{s['cagr']:>+9.1%}{s['cagr_sd_across_draws']:>7.1%}"
              f"   CAGR {d:>+6.1%}")

    print("\n PAIRED per slot vs buy-and-hold — slot 3 under two rules holds")
    print(" overlapping names over identical dates, so the pairing is most of")
    print(" the signal and comparing the two 10-90% bands discards it.\n")
    print(f"   {'rule':<32}{'mean dCAGR':>12}{'sd':>8}{'wins':>8}"
          f"{'t (12 slots)':>14}")
    for name in R:
        if name == "hold 252":
            continue
        q = paired(D, name)
        if not q:
            continue
        print(f"   {name:<32}{q['mean']:>+12.2%}{q['sd']:>8.2%}"
              f"{q['wins']:>5}/{q['n']:<2}{q['t']:>14.2f}")
    print("   NOTE: 12 month-offset slots over one 20-year sample are NOT 12")
    print("   independent trials — the t is a within-sample consistency")
    print("   check, not a significance test against the population.")

    # ---- the check that decides whether any of this is solid --------------
    print("\n HALF-SAMPLE SPLIT — a rule that only works in one era is not a")
    print(" rule. Cohorts split at the median date; the paired slot test is")
    print(" re-run inside each half independently.\n")
    mid = D["as_of"].quantile(0.5)
    halves = {"early": D[D["as_of"] <= mid], "late": D[D["as_of"] > mid]}
    print(f"   early {D['as_of'].min().date()} -> {mid.date()}, "
          f"late {mid.date()} -> {D['as_of'].max().date()}")
    print(f"   {'rule':<32}{'early dCAGR':>13}{'wins':>8}"
          f"{'late dCAGR':>13}{'wins':>8}{'both?':>8}")
    for name in R:
        if name == "hold 252":
            continue
        qs = {k: paired(v, name) for k, v in halves.items()}
        if not all(qs.values()):
            continue
        e, l = qs["early"], qs["late"]
        both = "YES" if (e["mean"] > 0 and l["mean"] > 0) else "no"
        print(f"   {name:<32}{e['mean']:>+13.2%}{e['wins']:>5}/{e['n']:<2}"
              f"{l['mean']:>+13.2%}{l['wins']:>5}/{l['n']:<2}{both:>8}")

    # is the ENTRY itself stable, or is it the same one-era artefact?
    print("\n THE ENTRY, SPLIT THE SAME WAY — buy-and-hold on the multiplier")
    print(" picks against buy-and-hold on random draws from the same pool.\n")
    print(f"   {'half':<10}{'picks CAGR':>12}{'random':>10}{'edge':>9}"
          f"{'draw sd':>9}{'sds':>7}")
    for k, v in list(halves.items()) + [("full", D)]:
        a = summarise(v, "hold 252", "picked")
        b = summarise(v, "hold 252", "random", draws=a_draws)
        if not a or not b:
            continue
        e = a["cagr"] - b["cagr"]
        sd = max(b["cagr_sd_across_draws"], 1e-9)
        print(f"   {k:<10}{a['cagr']:>+12.1%}{b['cagr']:>+10.1%}"
              f"{e:>+9.1%}{sd:>9.1%}{e / sd:>7.1f}")

    # The entry is ONE pre-specified comparison, not one of six, so it is the
    # only thing here a half-split can actually certify. Paired per slot.
    print("\n THE ENTRY, PAIRED PER SLOT (picks vs random, same rule, same")
    print(" slots, same dates) — 12 slots x 20 draws.\n")
    print(f"   {'half':<10}{'mean dCAGR':>12}{'sd':>8}{'wins':>12}{'t':>8}")
    for k, v in list(halves.items()) + [("full", D)]:
        q = paired(v, "hold 252", "hold 252", mode="picked",
                   mode_b="random", draws=a_draws)
        if not q:
            continue
        print(f"   {k:<10}{q['mean']:>+12.2%}{q['sd']:>8.2%}"
              f"{q['wins']:>7}/{q['n']:<4}{q['t']:>8.1f}")
    print("   The t assumes 240 independent pairs and they are NOT — 12 slots")
    print("   over one sample, 20 redraws of the same pool. Read the WIN RATE")
    print("   and the two halves agreeing; treat the t as a consistency check.")

    print("\n" + "=" * 96)
    print(" THE THREE PRE-REGISTERED VERDICTS")
    print("=" * 96)
    h = got.get("hold 252", {})
    hc = ctrl.get("hold 252", {})
    if h and hc:
        # H20a: how much of the exit gap survives random entry?
        best = max((k for k in got if k != "hold 252"),
                   key=lambda k: got[k]["cagr"])
        gp = got[best]["cagr"] - h["cagr"]
        gr = ctrl[best]["cagr"] - hc["cagr"]
        frac = gr / gp if abs(gp) > 1e-9 else np.nan
        print(f" H20a  best exit vs hold, multiplier basket: {gp:+.2%} CAGR")
        print(f"       the same, random basket:              {gr:+.2%} CAGR")
        print(f"       -> {frac:.0%} of the gap survives random entry: "
              f"{'SUPPORTED' if frac > 0.5 else 'FAILED'}")
        ent = h["cagr"] - hc["cagr"]
        print(f"       and the ENTRY itself is worth {ent:+.2%} CAGR "
              f"(hold, picks vs random), "
              f"{abs(ent) / max(hc['cagr_sd_across_draws'], 1e-9):.1f} "
              f"draw-sds")
        # H20b
        beats = [k for k, s in got.items()
                 if k != "hold 252" and s["terminal"] > h["terminal"]]
        dds = [k for k, s in got.items()
               if k != "hold 252" and s["maxdd"] > h["maxdd"]]
        print(f"\n H20b  rules beating buy-and-hold on terminal wealth: "
              f"{len(beats)} of {len(got) - 1}"
              + (f" ({', '.join(beats)})" if beats else ""))
        print(f"       rules with a shallower max drawdown: "
              f"{len(dds)} of {len(got) - 1}")
        # H20c
        bl = max(got, key=lambda k: got[k]["log"])
        bm = max(got, key=lambda k: got[k]["mean"])
        bd = max(got, key=lambda k: got[k]["median"])
        print(f"\n H20c  growth-optimal (mean log): {bl}")
        print(f"       mean-optimal:                {bm}")
        print(f"       median-optimal:              {bd}")
        print(f"       -> {'DIFFER' if len({bl, bm, bd}) > 1 else 'AGREE'}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
