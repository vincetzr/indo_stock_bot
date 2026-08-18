#!/usr/bin/env python3
"""The blue-chip algorithm: large caps only, chosen as of the day you traded.

Everything before this part was run on the whole exchange, and the book it
produced today holds JGLE at Rp89 and KOTA at Rp177. Those are not blue chips.
This restricts the same machinery to large caps and asks what changes.

The point-in-time universe
--------------------------
The obvious way to do this is to take the 63 names in ``config/universe.yaml``
under ``bluechip`` and ``lq45``. That would be **wrong in a way that inflates
every number**: that list is today's blue chips. Running it back to 2000 buys
BBCA and BMRI in 2003 because you know they became giants, and never buys the
ones that were giants then and are not now. It is the survivorship bias of
Result 35, applied deliberately.

So the universe here is defined **as of each date**: the top ``--universe-size``
names by trailing 250-day median turnover, recomputed every session from data
available that session. A name enters when it becomes liquid and leaves when it
stops being, exactly as it would have at the time. The fixed-list version is run
too, on the same engine, so the size of the bias is measured rather than
asserted.

What is compared
----------------
    equal weight     own the whole point-in-time large-cap universe, rebalanced
                     - the benchmark a blue-chip investor actually has
    momentum         rank that universe on trailing momentum, hold the top few
    + gates          none / 20-day average / bounded-lag reversal / both

Every gate, count and lookback is scored on five out-of-sample windows, and the
fixed-configuration result is reported beside the nested-selection result,
because Part XVIII showed the selection step loses to leaving it alone.

    python3 scripts/bluechip.py [--universe-size 40] [--quick]
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

from idxbot.config import load_config                       # noqa: E402
from optimize_consistent import CAP, load_wide, score_curve  # noqa: E402
from turn_book import ma_gate, reversal_gate, simulate       # noqa: E402


def pit_universe(W: Dict, size: int, min_turnover: float = 5e9,
                 max_vol_pct: float = 0.5, min_history: int = 750,
                 pool_multiple: int = 3) -> np.ndarray:
    """A point-in-time blue-chip universe built from price and volume alone.

    Ranking on turnover by itself is not a blue-chip screen and today it proves
    it: the top 40 by turnover on the last bar includes BUMI, DEWA, BRMS, ENRG
    and INET - penny stocks in a speculative run, churning more value per day
    than ICBP. They are the opposite of what "blue chip" means, and a universe
    built that way would have quietly become a momentum-junk universe in exactly
    the period the numbers are most likely to be quoted from.

    Market capitalisation would settle it, but the only cap data here is a
    present-day snapshot for 59 names, and using today's share count to build a
    2004 universe is the look-ahead this function exists to avoid.

    So blue chip is defined by three things that ARE knowable at the time:

        liquid      trailing 250-day median turnover in the top ``size *
                    pool_multiple``, and above an absolute floor
        established at least ``min_history`` sessions of trading behind it,
                    which excludes the freshly-listed hot issue
        stable      250-day realised volatility in the calmer half of that
                    liquid pool - the single line that separates a large cap
                    from a penny stock having a year

    Of whatever survives, the top ``size`` by turnover are the universe. All
    three windows look only backwards.
    """
    close = W["close"]
    slow = W["tv"].rolling(250, min_periods=100).median()
    listed = close.notna().cumsum()
    vol = close.pct_change().rolling(250, min_periods=125).std()

    pool = ((slow.rank(axis=1, ascending=False, method="first") <= size * pool_multiple)
            & (slow >= min_turnover) & (listed >= min_history))
    # volatility percentile *within the liquid pool on that date*, so the bar
    # moves with the market rather than being a fixed number that means one
    # thing in 2004 and another in 2021
    vol_in_pool = vol.where(pool)
    calm = vol_in_pool.rank(axis=1, pct=True, ascending=True) <= max_vol_pct

    ok = pool & calm
    final = slow.where(ok).rank(axis=1, ascending=False, method="first") <= size
    return (ok & final).fillna(False).to_numpy().astype(np.int8)


def fixed_universe(W: Dict) -> np.ndarray:
    """The config's blue-chip list, applied to all of history. Deliberately biased."""
    cfg = load_config()
    names = set(cfg.universe("bluechip")) | set(cfg.universe("lq45"))
    cols = list(W["mark"].columns)
    row = np.array([1 if c in names else 0 for c in cols], dtype=np.int8)
    return np.repeat(row[None, :], len(W["mark"]), axis=0)


def equal_weight(W: Dict, mask: np.ndarray, lo: int = 0,
                 hi: Optional[int] = None) -> pd.Series:
    """Own the whole universe, equally, with membership refreshed daily.

    This is the benchmark that matters: not the IHSG, which is capitalisation
    weighted and includes everything, but 'what if I just held the large caps'.
    Returns are capped at the auto-rejection band for the same reason every
    other engine here caps them.
    """
    sl = slice(lo, hi)
    mark = W["mark"].to_numpy()[sl]
    fac = W["fac"].to_numpy()[sl]
    m = mask[sl]
    ret = np.full(mark.shape, np.nan)
    ret[1:] = (mark[1:] / mark[:-1]) * fac[1:] - 1.0
    ret = np.clip(ret, -CAP, CAP)
    held = m[:-1]                       # membership known the previous close
    eq = np.ones(len(mark))
    for i in range(1, len(mark)):
        r = ret[i][(held[i - 1] == 1) & np.isfinite(ret[i])]
        eq[i] = eq[i - 1] * (1.0 + (float(np.mean(r)) if len(r) else 0.0))
    return pd.Series(eq, index=W["mark"].index[sl])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe-size", type=int, default=40)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    os.makedirs("reports", exist_ok=True)

    W = load_wide()
    close = W["close"]
    pit = pit_universe(W, args.universe_size)
    fixed = fixed_universe(W)
    print(f"point-in-time universe: median {pit.sum(axis=1)[pit.sum(axis=1) > 0].mean():.0f} "
          f"names/day, {int(pit.max(axis=0).sum())} distinct names ever a member")
    print(f"fixed list: {int(fixed[0].sum())} names, present for all of history")

    ma20 = ma_gate(close, 20)
    rev15 = reversal_gate(close, 0.15, 0.15)
    base = {"none": None, "ma20": ma20, "rev15": rev15,
            "both": (ma20 & rev15).astype(np.int8)}

    # --- the signal axis --------------------------------------------------- #
    # Momentum is the whole-exchange answer, and the quick run said it is the
    # wrong direction here. Blue chips are the part of the market where "buy the
    # low, sell the peak" is a plausible description of what actually happens,
    # so the ranking rule itself is searched, not only its parameters.
    for lb in (60, 120, 250):
        W["mom"][f"rev{lb}"] = -W["mom"][lb]                 # buy the laggards
    hi250 = close.rolling(250, min_periods=100).max()
    W["mom"]["dd250"] = -(close / hi250 - 1.0)               # buy the most beaten down
    vol120 = close.pct_change().rolling(120, min_periods=60).std()
    W["mom"]["lowvol"] = -vol120                             # buy the calmest
    W["mom"]["rebound"] = (close / close.rolling(250, min_periods=100).min() - 1.0)
    W["mom"]["rebound"] = -W["mom"]["rebound"]               # buy nearest its own low

    def gated(mask: np.ndarray, g: Optional[np.ndarray]) -> np.ndarray:
        return mask if g is None else (mask & g).astype(np.int8)

    n = len(W["mark"])
    edges = np.linspace(int(n * 0.35), n, args.folds + 1).astype(int)
    windows = [(edges[k], edges[k + 1]) for k in range(args.folds)]
    labels = [f"{W['mark'].index[a]:%Y-%m}" for a, _ in windows]

    # ---------------- the benchmark a blue-chip investor has ---------------- #
    print(f"\n{'=' * 108}\n OWNING THE LARGE CAPS — the benchmark, and what the "
          f"fixed list would have told you\n{'=' * 108}")
    print(f" {'universe':<34}" + "".join(f"{f'from {l}':>13}" for l in labels)
          + f"{'mean':>9}{'full':>9}")
    bench = {}
    for name, mask in (("point-in-time top "
                        f"{args.universe_size} by turnover", pit),
                       ("today's blue-chip list (biased)", fixed)):
        cs = [score_curve(equal_weight(W, mask, a, b), 0)["cagr"] for a, b in windows]
        full = score_curve(equal_weight(W, mask), 0)
        bench[name] = (cs, full)
        print(f" {name:<34}" + "".join(f"{v:>+13.1%}" for v in cs)
              + f"{np.mean(cs):>+9.1%}{full['cagr']:>+9.1%}")
    pit_cs, pit_full = bench[f"point-in-time top {args.universe_size} by turnover"]
    fx_cs, fx_full = bench["today's blue-chip list (biased)"]
    print(f"\n the fixed list overstates the benchmark by "
          f"{fx_full['cagr'] - pit_full['cagr']:+.1%} a year over the full record "
          f"and {np.mean(fx_cs) - np.mean(pit_cs):+.1%} across the folds.")
    print(" everything below uses the point-in-time universe.")

    # ------------------------------ the search ------------------------------ #
    lookbacks = (120, "rev120") if args.quick else (
        60, 120, 250, "rev60", "rev120", "rev250", "dd250", "lowvol", "rebound")
    tops = (5,) if args.quick else (3, 5, 8, 12)
    rebals = (10,) if args.quick else (10, 20, 60)
    combos = [(g, lb, tn, rb) for g in base
              for lb, tn, rb in itertools.product(lookbacks, tops, rebals)]
    print(f"\nsearching {len(combos)} configurations on the large-cap universe")

    oos: Dict[Tuple, List[Dict[str, float]]] = {}
    for i, (g, lb, tn, rb) in enumerate(combos, 1):
        gg = gated(pit, base[g])
        oos[(g, lb, tn, rb)] = [score_curve(*simulate(W, gg, lb, tn, rb, lo=a, hi=b))
                                for a, b in windows]
        if i % 12 == 0:
            print(f"  {i}/{len(combos)}")

    rows = []
    for key, sc in oos.items():
        c = [s["cagr"] for s in sc]
        rows.append({"gate": key[0], "lookback": key[1], "top_n": key[2],
                     "rebalance": key[3],
                     **{f"fold{k+1}": c[k] for k in range(args.folds)},
                     "mean": float(np.mean(c)), "worst": float(np.min(c)),
                     "excess": float(np.mean(c) - np.mean(pit_cs)),
                     "mean_dd": float(np.mean([s["max_dd"] for s in sc]))})
    R = pd.DataFrame(rows)
    R.to_csv("reports/bluechip_grid.csv", index=False)

    def show(df, title, k=12):
        print(f"\n{title}")
        print(f" {'gate':<8}{'signal':>9}{'top':>5}{'reb':>5}"
              + "".join(f"{f'f{i+1}':>10}" for i in range(args.folds))
              + f"{'mean':>9}{'worst':>9}{'vs EW':>9}{'meanDD':>9}")
        for _, r in df.head(k).iterrows():
            print(f" {r['gate']:<8}{str(r['lookback']):>9}{r['top_n']:>5.0f}"
                  f"{r['rebalance']:>5.0f}"
                  + "".join(f"{r[f'fold{i+1}']:>+10.1%}" for i in range(args.folds))
                  + f"{r['mean']:>+9.1%}{r['worst']:>+9.1%}"
                  f"{r['excess']:>+9.1%}{r['mean_dd']:>9.0%}")

    print(f"\n{'=' * 108}\n BEST BY OUT-OF-SAMPLE MEAN (a ceiling — chosen on the "
          f"windows it is quoted on)\n{'=' * 108}")
    show(R.sort_values("mean", ascending=False), "")
    print(f"\n{'=' * 108}\n BEST BY WORST FOLD (what you would have had to sit "
          f"through)\n{'=' * 108}")
    show(R.sort_values("worst", ascending=False), "")

    print(f"\n{'=' * 108}\n THE LEVERS, aggregated over everything else\n{'=' * 108}")
    for col, lab in (("lookback", "signal"), ("gate", "gate"),
                     ("top_n", "names held"), ("rebalance", "rebalance")):
        g = R.groupby(col)[["mean", "worst", "mean_dd"]].median()
        print(f"\n by {lab}:")
        print(f"   {'':>10}{'median mean':>14}{'median worst':>15}{'median maxDD':>15}")
        for k, r in g.iterrows():
            print(f"   {str(k):>10}{r['mean']:>+14.1%}{r['worst']:>+15.1%}"
                  f"{r['mean_dd']:>15.0%}")

    # -------- what an honest procedure retrieves, versus the benchmark -------- #
    print(f"\n{'=' * 108}\n NESTED SELECTION versus LEAVING IT ALONE\n{'=' * 108}")
    sel = []
    for mode, trail in (("expanding", None), ("trailing 5y", 5 * 250)):
        chain, ty, picks = 1.0, 0.0, []
        for k, (tr_hi, te_hi) in enumerate(windows):
            lo = 0 if trail is None else max(0, tr_hi - trail)
            best, bv = None, -np.inf
            for key in oos:
                g, lb, tn, rb = key
                s = score_curve(*simulate(W, gated(pit, base[g]), lb, tn, rb,
                                          lo=lo, hi=tr_hi))
                v = s["cagr"] / s["ulcer"] if s["ulcer"] > 0 else -np.inf
                if v > bv:
                    best, bv = key, v
            yrs = (W["mark"].index[te_hi - 1] - W["mark"].index[tr_hi]).days / 365.25
            chain *= (1.0 + oos[best][k]["cagr"]) ** yrs
            ty += yrs
            picks.append(oos[best][k])
            print(f" {mode:<12} fold {k+1} from {W['mark'].index[tr_hi]:%Y-%m}: "
                  f"{best[0]}/{best[1]}d/top{best[2]}/reb{best[3]:<3} -> "
                  f"{oos[best][k]['cagr']:>+7.1%}")
        sel.append({"method": f"nested, {mode}", "growth": chain,
                    "cagr": chain ** (1 / ty) - 1,
                    "mean_fold": float(np.mean([p["cagr"] for p in picks])),
                    "worst_fold": float(np.min([p["cagr"] for p in picks])),
                    "mean_dd": float(np.mean([p["max_dd"] for p in picks]))})

    def chain_of(key) -> Dict[str, float]:
        c, ty = 1.0, 0.0
        for k, (tr_hi, te_hi) in enumerate(windows):
            yrs = (W["mark"].index[te_hi - 1] - W["mark"].index[tr_hi]).days / 365.25
            c *= (1.0 + oos[key][k]["cagr"]) ** yrs
            ty += yrs
        return {"growth": c, "cagr": c ** (1 / ty) - 1,
                "mean_fold": float(np.mean([s["cagr"] for s in oos[key]])),
                "worst_fold": float(np.min([s["cagr"] for s in oos[key]])),
                "mean_dd": float(np.mean([s["max_dd"] for s in oos[key]]))}

    fixed_keys = [("none", 120, 5, 10), ("none", "dd250", 5, 20),
                  ("ma20", "dd250", 5, 20), ("none", "rev250", 5, 20),
                  ("none", "lowvol", 8, 20)]
    for key in fixed_keys:
        if key in oos:
            sel.append({"method": f"fixed {key[0]}/{key[1]}d/top{key[2]}/reb{key[3]}",
                        **chain_of(key)})
    ewc, ty = 1.0, 0.0
    for k, (a, b) in enumerate(windows):
        yrs = (W["mark"].index[b - 1] - W["mark"].index[a]).days / 365.25
        ewc *= (1.0 + pit_cs[k]) ** yrs
        ty += yrs
    sel.append({"method": "own the large caps, equally", "growth": ewc,
                "cagr": ewc ** (1 / ty) - 1, "mean_fold": float(np.mean(pit_cs)),
                "worst_fold": float(np.min(pit_cs)), "mean_dd": np.nan})

    S = pd.DataFrame(sel)
    print(f"\n {'':38}{'growth':>10}{'CAGR':>9}{'mean fold':>12}"
          f"{'worst fold':>12}{'mean maxDD':>12}")
    for _, r in S.iterrows():
        dd = "     n/a" if not np.isfinite(r["mean_dd"]) else f"{r['mean_dd']:>12.0%}"
        print(f" {r['method']:<38}{r['growth']:>9,.1f}x{r['cagr']:>+9.1%}"
              f"{r['mean_fold']:>+12.1%}{r['worst_fold']:>+12.1%}{dd}")
    S.to_csv("reports/bluechip_selection.csv", index=False)
    print("\n -> reports/bluechip_grid.csv, reports/bluechip_selection.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
