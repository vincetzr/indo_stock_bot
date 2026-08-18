#!/usr/bin/env python3
"""Does broker flow during a pullback predict whether it recovers?

This is the one candidate left. Result 93 excluded price, volume, the stock's own
trend and the market's trend by measurement - all four leave the false-exit rate
at 70-75%. Order flow is what remains, and it is the premise the repository was
built on: somebody is accumulating into the pullback, or somebody is leaving.

For each pullback event from ``pullback_events.py``, one range request returns
the top-10 buyers and sellers over the window from the peak to the signal. From
that:

    foreign_net     foreign buy lots - foreign sell lots, over total
    bumn_net        the same for state-owned members
    local_net       the same for domestic private
    concentration   top buyer's share of the top-10 buy side
    imbalance       total top-10 buy lots vs sell lots

and the question is whether any of them separates the pullbacks that recovered
from the ones that did not.

Politeness
----------
One request per EVENT, not per session - the range endpoint aggregates, and the
docstring of ``idxbot.data.ipot`` records that directional metrics survive the
display rounding that aggregation introduces. Everything is cached permanently,
so a rerun costs nothing. ``--limit`` bounds the run; the default is a pilot,
deliberately, because there is no case for fetching seventeen hundred windows
before knowing whether the first hundred carry any signal at all.

    python3 scripts/pullback_flow.py --limit 150
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

from idxbot.config import load_config          # noqa: E402
from idxbot.data.cache import Cache            # noqa: E402
from idxbot.data.ipot import (BASE_URL, broker_classes,   # noqa: E402
                              parse_table)

HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"),
    "Referer": "https://www.indopremier.com/",
    "Accept": "text/html,application/xhtml+xml,*/*;q=0.8",
}


def fetch_window(cache: Cache, ticker: str, start: pd.Timestamp,
                 end: pd.Timestamp, delay: float = 1.3,
                 board: str = "RG") -> Optional[pd.DataFrame]:
    """One range request, cached forever. Returns the parsed table or None."""
    import requests

    key = f"{ticker}_{start:%Y%m%d}_{end:%Y%m%d}_{board}_range"
    hit = cache.read("ipot_broker", key)
    if hit is not None:
        return hit
    try:
        time.sleep(delay)
        r = requests.get(BASE_URL, timeout=30, headers=HEADERS,
                         params={"code": ticker, "board": board,
                                 "start": start.strftime("%Y-%m-%d"),
                                 "end": end.strftime("%Y-%m-%d")})
        r.raise_for_status()
    except Exception as exc:
        print(f"   ! {ticker} {start:%Y-%m-%d}..{end:%Y-%m-%d}: {exc}")
        return None
    df = parse_table(r.text, ticker, end)
    if df is not None and not df.empty:
        cache.write("ipot_broker", key, df)
    return df


def flow_features(df: pd.DataFrame, classes: Dict[str, str]) -> Dict[str, float]:
    """Directional summaries. Abbreviation-safe: every one is a ratio."""
    if df is None or df.empty:
        return {}
    b = df["buy_lot"].fillna(0.0)
    s = df["sell_lot"].fillna(0.0)
    tot = float(b.sum() + s.sum())
    if tot <= 0:
        return {}
    out: Dict[str, float] = {
        "imbalance": float((b.sum() - s.sum()) / tot),
        "concentration": float(b.max() / b.sum()) if b.sum() > 0 else np.nan,
        "n_brokers": float(len(df)),
    }
    known = df["broker"].map(lambda x: str(x).upper() in classes)
    out["classified"] = float(known.mean())
    for cls in ("foreign", "bumn", "local"):
        m = df["broker"].map(lambda x: classes.get(str(x).upper()) == cls)
        out[f"{cls}_net"] = float((b[m].sum() - s[m].sum()) / tot)
        out[f"{cls}_share"] = float((b[m].sum() + s[m].sum()) / tot)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=150)
    ap.add_argument("--min-weeks", type=int, default=2,
                    help="skip windows shorter than this; too few sessions to read")
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()
    os.makedirs("reports", exist_ok=True)

    E = pd.read_csv("reports/pullback_events.csv",
                    parse_dates=["peak_date", "signal_date"])
    E = E[E["weeks_from_peak"] >= args.min_weeks].reset_index(drop=True)
    # A spread sample rather than the first N: taking one name's whole history
    # first would confound the flow signal with that name's era.
    E = E.sample(n=min(args.limit, len(E)), random_state=args.seed)
    E = E.sort_values(["ticker", "signal_date"]).reset_index(drop=True)
    print(f"pilot: {len(E)} events across {E['ticker'].nunique()} names, "
          f"{E['recovered'].mean():.0%} recovered")

    cfg = load_config()
    cache = Cache(cfg.path("data.cache_dir", "data/cache"))

    # Broker classification. A single live page only ever shows the top 10 of
    # each side, so scraping classes from one query leaves most codes unlabelled
    # and turns foreign_net into noise. The repository's own registry is the
    # primary source; live flags fill gaps and are recorded where they disagree.
    reg = cfg.brokers
    classes: Dict[str, str] = {}
    for code in reg.codes():
        b = reg.get(code)
        if b is None:
            continue
        classes[code.upper()] = ("bumn" if getattr(b, "state_owned", False)
                                 else "foreign" if getattr(b, "foreign", False)
                                 else "local")
    import requests
    live: Dict[str, str] = {}
    for probe in ("BBCA", "ADRO", "GOTO"):
        try:
            time.sleep(1.3)
            rr = requests.get(BASE_URL, timeout=30, headers=HEADERS,
                              params={"code": probe, "board": "RG",
                                      "start": "2026-08-10", "end": "2026-08-14"})
            live.update(broker_classes(rr.text))
        except Exception as exc:
            print(f"   ! class probe {probe}: {exc}")
    added = {k: v for k, v in live.items() if k not in classes}
    classes.update(added)
    disagree = [k for k, v in live.items()
                if k in classes and classes[k] != v and k not in added]
    print(f"broker classes: {len(classes)} codes "
          f"({sum(v == 'foreign' for v in classes.values())} foreign, "
          f"{sum(v == 'bumn' for v in classes.values())} bumn, "
          f"{sum(v == 'local' for v in classes.values())} local)")
    print(f"  registry {len(reg.codes())}, +{len(added)} from live pages, "
          f"{len(disagree)} disagreements kept as registry: "
          f"{','.join(sorted(disagree)[:8]) if disagree else 'none'}")

    rows: List[Dict] = []
    fetched = 0
    for i, e in E.iterrows():
        df = fetch_window(cache, e["ticker"], e["peak_date"], e["signal_date"])
        fetched += 1
        f = flow_features(df, classes)
        if f:
            rows.append({"ticker": e["ticker"], "signal_date": e["signal_date"],
                         "drawdown": e["drawdown"], "recovered": bool(e["recovered"]),
                         "bounced": bool(e["bounced_5pct"]), **f})
        if fetched % 25 == 0:
            print(f"  {fetched}/{len(E)} ({len(rows)} with data)")

    F = pd.DataFrame(rows)
    F.to_csv("reports/pullback_flow.csv", index=False)
    if F.empty:
        raise SystemExit("no flow data returned")

    print(f"\n{'=' * 96}\n DOES FLOW DURING THE PULLBACK PREDICT THE RECOVERY?"
          f"\n{'=' * 96}")
    print(f" {len(F)} events with data, {F['recovered'].mean():.0%} recovered")
    from scipy import stats
    print(f"\n {'feature':<16}{'recovered':>12}{'did not':>12}{'difference':>13}"
          f"{'t':>8}{'p':>10}")
    print(f" median share of brokers classified: {F['classified'].median():.0%}")
    feats = [c for c in ("foreign_net", "bumn_net", "local_net", "imbalance",
                         "concentration", "foreign_share") if c in F.columns]
    res = []
    for c in feats:
        a = F[F["recovered"]][c].dropna()
        b = F[~F["recovered"]][c].dropna()
        if len(a) < 5 or len(b) < 5:
            continue
        t, p = stats.ttest_ind(a, b, equal_var=False)
        res.append({"feature": c, "recovered": a.mean(), "not": b.mean(),
                    "diff": a.mean() - b.mean(), "t": t, "p": p})
        print(f" {c:<16}{a.mean():>+12.3f}{b.mean():>+12.3f}"
              f"{a.mean() - b.mean():>+13.3f}{t:>+8.2f}{p:>10.3f}")
    R = pd.DataFrame(res)
    R.to_csv("reports/pullback_flow_tests.csv", index=False)

    sig = R[R["p"] < 0.05] if not R.empty else R
    print()
    if sig.empty:
        print(" Nothing separates them at p<0.05. On this pilot, broker flow during")
        print(" the pullback does not tell you whether the pullback recovers - which")
        print(" is the same answer price, volume and trend gave.")
    else:
        print(f" {len(sig)} feature(s) separate the two groups at p<0.05:")
        for _, r in sig.iterrows():
            print(f"   {r['feature']}: {r['diff']:+.3f} difference, p={r['p']:.4f}")
        print("\n Worth expanding the sample. This is a pilot and p-values on six")
        print(" features invite one false positive in twenty by construction.")
    print("\n -> reports/pullback_flow.csv, reports/pullback_flow_tests.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
