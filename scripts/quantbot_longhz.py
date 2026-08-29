"""The horizon gap H50 left open.

H50 swept 21 / 63 / 126 / 252 sessions and stopped at one year. That is the SAME
error A20 names -- a parameter fixed by convenience and inherited -- committed
inside the study whose docstring cites it. On a market with positive drift the
probability a trade ends positive RISES with the horizon, so if the joint target
(>=80% positive AND >=+4% mean) is reachable anywhere, it is out here.

No take-profit, no stop: a pure hold, which is the configuration that maximises
BOTH the mean (right tail intact) and the positive rate (drift accumulates).
If this cannot reach the target, nothing weaker can.
"""
import sys
sys.path.insert(0,'scripts'); sys.path.insert(0,'src')
import numpy as np, pandas as pd
from quantbot import CACHE, MIN_RP, FEE
from paint_suite import tick_of

D = pd.read_parquet(CACHE)
print(f"{'horizon':>9}{'years':>7}{'arm':>10}{'n':>10}{'names':>7}"
      f"{'POSITIVE':>10}{'MEAN':>10}{'median':>10}{'ann':>8}"
      f"{'indep n':>9}{'GOAL?':>7}")
rows=[]
for hor in (252, 504, 1260, 2520):
    parts=[]
    for tk, g in D.groupby("ticker", sort=False):
        p = g["adj"].to_numpy(float)
        if len(p) <= hor + 5:
            continue
        med = float(np.nanmedian(g["close_raw"].to_numpy(float)))
        if not np.isfinite(med) or med <= 0:
            continue
        cost = FEE + tick_of(med)/med
        fwd = np.full(len(p), np.nan)
        fwd[:len(p)-hor] = p[hor:]/np.maximum(p[:len(p)-hor],1e-9) - 1.0 - cost
        parts.append(pd.DataFrame({
            "date": g["date"].to_numpy(), "ticker": tk, "ret": fwd,
            "rp60": g["rp60"].to_numpy(float),
            "close_raw": g["close_raw"].to_numpy(float),
            "stack": g["stack"].to_numpy(float)}))
    L = pd.concat(parts, ignore_index=True).dropna(subset=["ret"])
    L = L[(L.rp60 >= MIN_RP) & (L.close_raw >= 500)]
    for arm, sub in (("all", L), ("trend", L[L["stack"] >= 2])):
        if len(sub) < 500: continue
        r = sub["ret"].to_numpy()
        pos = float((r>0).mean()); mean = float(r.mean())
        ml = float(np.mean(np.log(np.maximum(1+r, .01))))
        ann = float(np.exp(ml*252.0/hor)-1)
        # independent observations: non-overlapping windows per name
        span = sub.groupby("ticker")["date"].agg(lambda s: (s.max()-s.min()).days/1.45/252)
        indep = float((span*252/hor).clip(lower=0).sum())
        goal = pos>=0.80 and mean>=0.04
        print(f"{hor:>9}{hor/252:>7.1f}{arm:>10}{len(sub):>10,}"
              f"{sub.ticker.nunique():>7}{pos:>10.1%}{mean:>+10.2%}"
              f"{np.median(r):>+10.2%}{ann:>+8.1%}{indep:>9.0f}"
              f"{'YES' if goal else 'no':>7}")
        rows.append({"hor":hor,"arm":arm,"n":len(sub),"pos":pos,"mean":mean,
                     "median":float(np.median(r)),"ann":ann,"indep":indep,
                     "goal":goal})
pd.DataFrame(rows).to_csv("reports/quantbot_longhz.csv", index=False)
B=pd.DataFrame(rows)
print(f"\ncells reaching the JOINT goal: {int(B.goal.sum())} of {len(B)}")
print(f"best positive rate anywhere: {B.pos.max():.1%}")
