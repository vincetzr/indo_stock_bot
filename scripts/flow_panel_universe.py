#!/usr/bin/env python3
"""Pick the tickers the broker-flow panel will be collected for.

WHY A SAMPLE AT ALL
-------------------
CLAUDE.md §4 sizes Phase 1 at ~800 names. The only free full-depth-enough route
is IndoPremier's public module, and it costs ONE REQUEST PER (TICKER, WINDOW).
At a fortnightly window over 2014-2026 that is 304 windows a name, so 800 names
is 243,000 requests. A5 rules out bulk harvesting and that is bulk harvesting.

So the panel is a SAMPLE, and the sample has to be designed rather than taken
off the top of a market-cap list, because §7 requires two things a convenience
sample cannot give:

    the liquidity decile breakdown - "report IC by liquidity decile; if the
    effect lives only in the bottom two deciles it is likely untradeable". That
    needs names IN those deciles. A top-200-by-turnover panel cannot answer the
    question it is meant to answer.

    a survivorship-free cross-section - §5. The recovered delisted names are
    the only reason that is possible at all, so they are IN, at their natural
    weight rather than as a token.

STRATIFY ON POINT-IN-TIME LIQUIDITY, NOT FULL-SAMPLE LIQUIDITY
---------------------------------------------------------------
The obvious thing - rank every name by its median turnover over 2014-2026 and
take 20 from each decile - selects on information from the end of the sample. A
name that was a micro cap in 2015 and a large cap in 2024 lands in a decile it
did not occupy for most of its life, and the decile analysis is then partly a
statement about which names GREW.

Each name is therefore ranked on the median turnover of its **first 250 traded
bars**, which is point-in-time with respect to its own entry into the panel.
That is a weaker sort - liquidity drifts - so the study must re-rank by
trailing turnover when it actually computes the decile IC. This file's job is
only to make sure all ten deciles are represented at entry.

Reproducible from the seed. Writes config/flow_panel.yaml.
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys
from typing import Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

LIVE = os.path.join("data", "cache", "ohlcv")
DEAD = os.path.join("data", "cache", "delisted")
OUT = os.path.join("config", "flow_panel.yaml")

#: Bars a name needs before it can carry a forward-return label at all.
MIN_BARS = 250

#: Requests are the budget, so the panel size is a budget decision. 200 names
#: gives 20 per liquidity decile - thin, and honestly thin, but enough that a
#: decile IC has a standard error worth quoting.
PANEL = 200
DECILES = 10


def candidates(start: str = "2014-01-01") -> pd.DataFrame:
    rows: List[Dict] = []
    for src, d in (("live", LIVE), ("delisted", DEAD)):
        for fp in sorted(glob.glob(os.path.join(d, "*.JK.csv.gz"))):
            t = os.path.basename(fp).split(".")[0]
            if not re.fullmatch(r"[A-Z]{4}", t):
                continue                      # indices, FX, futures
            try:
                x = pd.read_csv(fp, usecols=["date", "close", "volume"])
            except Exception:                                   # noqa: BLE001
                continue
            x["date"] = pd.to_datetime(x["date"])
            x = x[(x["date"] >= start) & (x["close"] > 0)]
            traded = x[x["volume"] > 0]
            if len(x) < MIN_BARS or len(traded) < MIN_BARS // 2:
                continue
            # POINT-IN-TIME liquidity: the first 250 traded bars only.
            head = traded.head(MIN_BARS)
            rows.append({
                "ticker": t, "src": src, "bars": len(x),
                "first": x["date"].min(), "last": x["date"].max(),
                "entry_turnover": float((head["close"] * head["volume"]).median()),
            })
    return pd.DataFrame(rows)


def stratify(U: pd.DataFrame, panel: int = PANEL, seed: int = 20260822
             ) -> pd.DataFrame:
    """Equal draw from each entry-liquidity decile, delisted names forced in."""
    rng = np.random.default_rng(seed)
    U = U.copy()
    U["decile"] = pd.qcut(U["entry_turnover"].rank(method="first"), DECILES,
                          labels=False)
    per = panel // DECILES
    picks = []
    for d, g in U.groupby("decile"):
        dead = g[g["src"] == "delisted"]
        live = g[g["src"] == "live"]
        # Every recovered delisted name in this decile is taken first - there
        # are only 88 of them in the whole spine and they are the only thing
        # standing between this panel and a survivor universe.
        take_dead = dead
        need = max(0, per - len(take_dead))
        take_live = live.sample(min(need, len(live)), random_state=int(
            rng.integers(0, 2**31 - 1)))
        picks.append(pd.concat([take_dead, take_live]))
    out = pd.concat(picks).sort_values(["decile", "ticker"])
    return out.reset_index(drop=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", type=int, default=PANEL)
    ap.add_argument("--seed", type=int, default=20260822)
    ap.add_argument("--start", default="2014-01-01")
    a = ap.parse_args()

    U = candidates(a.start)
    print(f"candidates with >={MIN_BARS} bars since {a.start}: {len(U)} "
          f"(live {int((U.src == 'live').sum())}, "
          f"delisted {int((U.src == 'delisted').sum())})")
    S = stratify(U, a.panel, a.seed)
    print(f"panel: {len(S)} names, {int((S.src == 'delisted').sum())} of them "
          f"delisted\n")
    for d, g in S.groupby("decile"):
        lo, hi = g["entry_turnover"].min(), g["entry_turnover"].max()
        print(f"  decile {int(d)}: {len(g):>3} names, entry turnover "
              f"Rp {lo:>15,.0f} .. {hi:>15,.0f}  "
              f"({int((g.src == 'delisted').sum())} delisted)")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        f.write("# Broker-flow panel universe. Regenerate with\n"
                "#   python3 scripts/flow_panel_universe.py\n"
                "# Stratified on POINT-IN-TIME entry liquidity (first 250\n"
                "# traded bars), not full-sample liquidity - see the module\n"
                "# docstring for why that distinction matters to §7's decile\n"
                "# breakdown.\n")
        f.write(f"seed: {a.seed}\nstart: '{a.start}'\n"
                f"panel_size: {len(S)}\ndeciles: {DECILES}\n")
        f.write("# ticker: [decile, source, entry_turnover_rp]\ntickers:\n")
        for _, r in S.iterrows():
            f.write(f"  {r['ticker']}: [{int(r['decile'])}, {r['src']}, "
                    f"{r['entry_turnover']:.0f}]\n")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
