#!/usr/bin/env python3
"""H53 — do small and large IDX names move differently, and what does each want?

    python3 scripts/sizeprofile.py

WHY THIS IS WORTH A RUN RATHER THAN AN OPINION. H52 found the strength+calm
screen's edge over a random control collapsing +12.9% -> +0.7% as the universe
narrowed to the largest forty names. That is one signal on one construction. If
the segments genuinely behave differently, it should show up across the WHOLE
feature set, in the return distribution, and in the autocorrelation structure —
and it should come with a per-segment cost, because what is tradeable in a bank
is not tradeable in a Rp600 small cap.

SIZE IS PROXIED BY TRAILING 60-DAY MEDIAN TURNOVER, POINT-IN-TIME. No
point-in-time share count exists in this repo (A25: the only one available is
frozen at 2024-07-10, and using it on a 2010 bar is look-ahead — Indonesian
rights issues are exactly what makes that wrong). Turnover is causal, is what
actually constrains a position, and correlates with size, but it is not size.

THE STALENESS CHECK IS NOT OPTIONAL AND IT IS WHY H44 WAS WITHDRAWN. A thin name
prints the same close for days; its "return" then arrives in a lump, and any
feature that knows the price is stale predicts the catch-up. That is
NON-SYNCHRONOUS PRICING, not signal, and it manufactures beautiful ICs in
exactly the buckets where nothing can be traded. So every bucket reports its
zero-return-day share and its first-order autocorrelation alongside every IC,
and the registered null below exists to catch it.

PRE-REGISTERED, WRITTEN BEFORE ANY BUCKET WAS SCORED
-----------------------------------------------------
S1  SHORT-TERM REVERSAL is stronger in small names than large — `rev1` and
    `rev5` carry more negative IC at the thin end. Predicted: YES, and partly
    NOT TRADEABLE, because bid-ask bounce and stale prints produce mechanical
    reversal that a spread cannot be crossed to capture.
S2  MOMENTUM AND STRENGTH (`mom12_1`, `hi52`) are stronger in small and mid
    names than in the largest. Predicted: YES — this is H52's collapse seen
    feature-by-feature rather than through one portfolio.
S3  LOW VOLATILITY (`lowvol`) works better in LARGE names than small.
    Predicted: YES. Among small caps, volatility is mostly a proxy for
    illiquidity and distress; among large caps it separates steady compounders
    from cyclicals.
S4  PREDICTED NULL — `squeeze` was registered by H13 as a feature that should
    not predict direction (range compression forecasts the SIZE of the next
    move, not its sign). It must show no consistent IC in any bucket. If it
    fires in the thin buckets, those buckets' ICs are non-synchronous pricing
    and not one of them can be believed.
S5  The half-spread rises monotonically as size falls, and the IC-to-cost ratio
    — not the IC — decides which segment is actually tradeable. Predicted: the
    thin end has the largest ICs and the worst ratio.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))
sys.path.insert(0, os.path.dirname(__file__))

from paint_suite import tick_of                                  # noqa: E402

PANEL = os.path.join("data", "spine", "price_panel.parquet")
OUT = "reports"
#: `pastret1` / `pastret5` are computed here from `adj_close` and are the
#: reversal features S1 is scored on. The panel's own `rev1`/`rev5` are kept
#: alongside them but are NOT what their names suggest: `rev1` correlates only
#: **0.209** with the true daily log return and carries AR(1) **+0.925**, so it
#: is a smoothed/normalised construction, not a one-day return. A first version
#: of this script used it as one and reported the LARGEST IDX names at 262%
#: annualised volatility — a number impossible enough to catch itself.
FEATS = ("mom12_1", "hi52", "pastret1", "pastret5", "rev1", "rev5", "lowvol",
         "volz20", "amihud60", "atr_mom20", "squeeze")
HORIZONS = ("fwd5", "fwd20")
NB = 5           # size buckets, thin -> thick


def load() -> pd.DataFrame:
    P = pd.read_parquet(PANEL)
    P = P[(P["adj_close"] > 0) & P["tradeable"].astype(bool)]
    P = P.sort_values(["ticker", "date"])
    import warnings
    warnings.filterwarnings("ignore", message="An input array is constant")
    P["tv"] = np.exp(P["log_turnover"].fillna(-np.inf))
    #  THE TRUE daily log return, from the price. Everything about how a segment
    #  BEHAVES — volatility, skew, staleness, autocorrelation — is measured on
    #  this and never on a panel feature whose construction is unverified.
    P["lr"] = P.groupby("ticker")["adj_close"].transform(
        lambda s: np.log(s).diff())
    P["pastret1"] = P["lr"]
    P["pastret5"] = P.groupby("ticker")["adj_close"].transform(
        lambda s: s.pct_change(5))
    #  TRAILING median turnover, per name, so the bucket is decided by what was
    #  knowable on the bar rather than by the name's eventual liquidity.
    P["tv60"] = P.groupby("ticker")["tv"].transform(
        lambda s: s.rolling(60, min_periods=30).median())
    P = P[P["tv60"] > 0]
    #  BUCKETS ARE FORMED WITHIN EACH DATE. A fixed rupiah cut would put the
    #  whole of 2003 in the "small" bucket and the whole of 2024 in the "large"
    #  one, and the study would be measuring the calendar.
    P["bucket"] = P.groupby("date")["tv60"].transform(
        lambda s: pd.qcut(s.rank(method="first"), NB, labels=False)
        if s.notna().sum() >= NB * 4 else np.nan)
    return P.dropna(subset=["bucket"])


def ic_series(d: pd.DataFrame, f: str, h: str) -> pd.Series:
    """Daily cross-sectional Spearman IC, within one bucket."""
    g = d.dropna(subset=[f, h])
    if len(g) < 200:
        return pd.Series(dtype=float)
    return (g.groupby("date")
            .apply(lambda x: x[f].corr(x[h], method="spearman")
                   if len(x) >= 8 else np.nan, include_groups=False)
            .dropna())


def hac_t(x: np.ndarray, lag: int = 20) -> float:
    """Newey-West t on the mean. Overlapping forward windows make daily ICs
    autocorrelated, and an iid t would be several times too confident."""
    n = len(x)
    if n < 30:
        return float("nan")
    e = x - x.mean()
    s = float(e @ e) / n
    for k in range(1, min(lag, n - 1) + 1):
        c = float(e[k:] @ e[:-k]) / n
        s += 2.0 * (1.0 - k / (lag + 1.0)) * c
    return float(x.mean() / np.sqrt(max(s, 1e-18) / n))


def profile(d: pd.DataFrame) -> Dict:
    """How one size bucket BEHAVES, before any signal is applied."""
    r = d["lr"].to_numpy(float)
    r = r[np.isfinite(r)]
    px = d["close"].to_numpy(float)
    px = px[np.isfinite(px) & (px > 0)]
    spread = np.mean([tick_of(v) / v for v in px[::97]]) if len(px) else np.nan
    #  A stale print shows as an EXACT zero return. In a thin name that is most
    #  days, and it is the mechanism behind H44's withdrawn headline.
    zero = float(np.mean(np.abs(r) < 1e-9)) if len(r) else np.nan
    ar1 = float(pd.Series(r).autocorr(1)) if len(r) > 100 else np.nan
    return {"n": len(d), "names": d["ticker"].nunique(),
            "median_tv": float(d["tv60"].median()),
            "median_px": float(np.median(px)) if len(px) else np.nan,
            "vol_ann": float(np.std(r) * np.sqrt(252)) if len(r) else np.nan,
            "skew": float(pd.Series(r).skew()) if len(r) else np.nan,
            "zero_days": zero, "ar1": ar1,
            "half_spread": float(spread),
            "round_trip": float(0.0056 + spread)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--half", action="store_true",
                    help="also print the early/late split of every IC")
    a = ap.parse_args()
    P = load()
    mid = P["date"].quantile(0.5)

    print(f"H53 — IDX by size, {NB} buckets formed WITHIN each date on trailing "
          f"60-day median turnover.\n{len(P):,} bars, "
          f"{P['ticker'].nunique()} names, "
          f"{pd.Timestamp(P['date'].min()):%Y-%m} → "
          f"{pd.Timestamp(P['date'].max()):%Y-%m}\n")

    rows = []
    print("HOW THE SEGMENTS BEHAVE, before any signal")
    print(f"{'bucket':<10}{'names':>7}{'median Rp/day':>15}{'median px':>11}"
          f"{'ann vol':>9}{'skew':>8}{'ZERO-ret days':>15}{'AR(1)':>9}"
          f"{'half-spread':>13}{'round trip':>12}")
    for b in range(NB):
        d = P[P["bucket"] == b]
        pr = profile(d)
        pr["bucket"] = b
        rows.append(pr)
        lab = ["thinnest", "thin", "mid", "thick", "thickest"][b]
        print(f"{lab:<10}{pr['names']:>7}{pr['median_tv']:>15,.0f}"
              f"{pr['median_px']:>11,.0f}{pr['vol_ann']:>9.1%}"
              f"{pr['skew']:>8.2f}{pr['zero_days']:>15.1%}{pr['ar1']:>9.3f}"
              f"{pr['half_spread']:>13.2%}{pr['round_trip']:>12.2%}")
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "sizeprofile.csv"), index=False)

    ics: List[Dict] = []
    for h in HORIZONS:
        print(f"\nINFORMATION COEFFICIENT by size bucket — {h} "
              f"(Spearman, within date, Newey-West t at 20 lags)")
        print(f"{'feature':<12}" + "".join(
            f"{lab:>16}" for lab in ("thinnest", "thin", "mid", "thick",
                                     "thickest")))
        for f in FEATS:
            cells = []
            for b in range(NB):
                s = ic_series(P[P["bucket"] == b], f, h)
                if len(s) < 60:
                    cells.append((np.nan, np.nan, np.nan, np.nan))
                    continue
                v = s.to_numpy(float)
                e = s[s.index < mid].to_numpy(float)
                l = s[s.index >= mid].to_numpy(float)
                cells.append((float(v.mean()), hac_t(v),
                              float(e.mean()) if len(e) > 30 else np.nan,
                              float(l.mean()) if len(l) > 30 else np.nan))
                ics.append({"feat": f, "h": h, "bucket": b, "ic": v.mean(),
                            "t": hac_t(v), "early": cells[-1][2],
                            "late": cells[-1][3], "n_days": len(v)})
            print(f"{f:<12}" + "".join(
                "             n/a" if not np.isfinite(c[0])
                else f"{c[0]:>+9.4f}{('*' if abs(c[1]) > 3 else ' '):>1}"
                     f"{c[1]:>+6.1f}"
                for c in cells))
    I = pd.DataFrame(ics)
    I.to_csv(os.path.join(OUT, "sizeprofile_ic.csv"), index=False)

    #  --------------------------------------------------------------- verdicts
    def get(f, h, b):
        r = I[(I.feat == f) & (I.h == h) & (I.bucket == b)]
        return float(r["ic"].iloc[0]) if len(r) else np.nan

    print("\n" + "=" * 78)
    for tag, f, h in (("S1 reversal", "rev5", "fwd5"),
                      ("S2 momentum", "mom12_1", "fwd20"),
                      ("S2 strength", "hi52", "fwd20"),
                      ("S3 low vol", "lowvol", "fwd20"),
                      ("S4 NULL squeeze", "squeeze", "fwd20")):
        vals = [get(f, h, b) for b in range(NB)]
        print(f"{tag:<18}{f:<10}{h:<6}" + "".join(
            f"{v:>+9.4f}" if np.isfinite(v) else "      n/a" for v in vals)
            + f"   thin−thick {vals[0] - vals[-1]:>+.4f}")
    sq = I[(I.feat == "squeeze")]
    big = sq[sq["t"].abs() > 3]
    print(f"\n  S4 — `squeeze` (registered predicted-null) clears |t|>3 in "
          f"{len(big)} of {len(sq)} (bucket, horizon) cells."
          f"{'  IT FIRES — read every IC in those buckets as suspect.' if len(big) else ''}")
    if len(big):
        print("     " + ", ".join(f"b{int(r.bucket)}/{r.h} t={r.t:+.1f}"
                                  for _, r in big.iterrows()))
    B = pd.DataFrame(rows)
    print(f"\n  S5 — round-trip cost by bucket: "
          + ", ".join(f"{v:.2%}" for v in B['round_trip'])
          + f"   (thin/thick ratio {B['round_trip'].iloc[0] / B['round_trip'].iloc[-1]:.1f}x)")
    print(f"       stale (zero-return) days: "
          + ", ".join(f"{v:.0%}" for v in B['zero_days']))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# ================================================ WHAT EACH SEGMENT PAYS OUT ==
def segment_spread(P: pd.DataFrame, feat: str, h: str = "fwd20",
                   q: float = 0.8) -> List[Dict]:
    """The MONEY number: a long-only top-quintile tilt inside each size bucket,
    against that bucket's own mean, net of that bucket's own round trip.

    An IC is a rank correlation and A9 records the wedge plainly — "a rank tilt
    is not a return spread". A high IC in a cross-section with little dispersion
    converts to less money than a lower IC in a wide one, and cost differs by
    segment too. So the tuning question is settled here, not in the IC table.
    """
    out = []
    for b in sorted(P["bucket"].unique()):
        d = P[P["bucket"] == b].dropna(subset=[feat, h])
        if len(d) < 5000:
            continue
        r = d.groupby("date")[feat].rank(pct=True)
        top = d[r >= q]
        base = d.groupby("date")[h].mean()
        top_by_date = top.groupby("date")[h].mean()
        j = top_by_date.index.intersection(base.index)
        sp = (top_by_date.loc[j] - base.loc[j]).to_numpy(float)
        if len(sp) < 60:
            continue
        px = d["close"].to_numpy(float)
        px = px[np.isfinite(px) & (px > 0)]
        cost = 0.0056 + float(np.mean([tick_of(v) / v for v in px[::97]]))
        per_yr = 252.0 / 20.0
        out.append({"bucket": int(b), "feat": feat, "n_dates": len(sp),
                    "gross_20d": float(np.mean(sp)),
                    "t": hac_t(sp), "cost": cost,
                    "net_20d": float(np.mean(sp)) - cost,
                    "gross_yr": float(np.mean(sp)) * per_yr,
                    "net_yr": (float(np.mean(sp)) - cost) * per_yr,
                    "disp": float(d.groupby("date")[h].std().mean())})
    return out
