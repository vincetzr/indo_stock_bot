"""Build the 1h panel: every cached hourly file, joined to the daily spine.

The output feeds scripts/mtf_h1.py. It is NOT committed -- it is 3.1m rows
rebuilt in ~2 minutes from data/cache/intraday/, which is gitignored anyway.
"""
import glob, os, sys
sys.path.insert(0, "scripts")
import numpy as np, pandas as pd

SP = os.environ.get("IDX_H1_DIR", os.path.join("data", "cache"))
files = sorted(glob.glob("data/cache/intraday/*_1h_730d.csv.gz"))
P = pd.read_parquet("data/spine/price_panel.parquet")
P = P[P["adj_close"] > 0].copy()
#  THE 1h BARS ARE ON THE RAW (UNADJUSTED) BASIS -- measured, not assumed:
#  the last 1h close of a day matches the panel's `close` exactly on 99.54% of
#  65,415 name-days and its `adj_close` on only 51.75%. So every 1h price must
#  be multiplied by the panel's own factor before any multi-day return is taken,
#  or a split inside the window becomes a fake crash.
P["adjf"] = P["adj_close"] / P["close"]
F = P[["ticker", "date", "adjf", "tv60", "elig"]] if "tv60" in P.columns else None
if F is None:
    P["tv"] = np.exp(P["log_turnover"].fillna(-np.inf))
    P["tv60"] = P.groupby("ticker")["tv"].transform(
        lambda s: s.rolling(60, min_periods=30).median())
    P["elig"] = (P["tradeable"].astype(bool) & (P["tv60"] >= 1e9)
                 & (P["close"] >= 100))
    F = P[["ticker", "date", "adjf", "tv60", "elig"]]

out = []
for i, f in enumerate(files):
    tk = os.path.basename(f).split(".JK")[0]
    try:
        d = pd.read_csv(f, parse_dates=["ts"])
    except Exception:
        continue
    if d.empty or len(d) < 200:
        continue
    d["ticker"] = tk
    d["date"] = d["ts"].dt.normalize()
    out.append(d[["ticker", "ts", "date", "open", "high", "low", "close",
                  "volume"]])
    if i % 100 == 0:
        print(f"  {i}/{len(files)}", flush=True)
H = pd.concat(out, ignore_index=True)
H = H.merge(F, on=["ticker", "date"], how="inner")
for c in ("open", "high", "low", "close"):
    H["a_" + c] = H[c] * H["adjf"]
H = H.sort_values(["ticker", "ts"]).reset_index(drop=True)
H.to_parquet(f"{SP}/h1_panel.parquet")
print(f"\n1h panel: {len(H):,} bars, {H['ticker'].nunique()} names, "
      f"{H['date'].min().date()} -> {H['date'].max().date()}")
print(f"eligible bars: {H['elig'].mean():.3f}")
