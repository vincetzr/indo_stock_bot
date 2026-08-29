"""A18: only the half-split decides.

The model book read 30th / 97th / 40th / 100th percentile of its own random
control across 4/8/16/32 slots. Two cells above the band and two below, and
NOT MONOTONE in slot count -- a genuine selection edge cannot be strong at 8,
absent at 16, and strong again at 32. Either the halves agree or it is a draw.
"""
import sys
sys.path.insert(0,'scripts'); sys.path.insert(0,'src')
import numpy as np, pandas as pd
from quantbot import book, index_cagr

W = pd.read_parquet("data/spine/quantbot_wf.parquet")
W["date"] = pd.to_datetime(W["date"])
mid = W["date"].quantile(0.5)
print(f"split at {mid:%Y-%m}\n")
print(f"{'slots':>6}{'half':>7}{'MODEL':>9}{'rand mean':>11}{'rand sd':>9}"
      f"{'pctile':>8}{'index':>9}{'beats rand':>12}")
rows=[]
for slots in (4, 8, 16, 32):
    res={}
    for lab, sub in (("early", W[W.date < mid]), ("late", W[W.date >= mid])):
        ic = index_cagr(sub.date.min(), sub.date.max())
        m = book(sub, slots=slots, rank="p")["cagr"]
        r = np.array([book(sub, slots=slots, rank="rand", seed=s)["cagr"]
                      for s in range(30)])
        pct = float((r < m).mean()); res[lab]=m-r.mean()
        print(f"{slots:>6}{lab:>7}{m:>+9.2%}{r.mean():>+11.2%}"
              f"{r.std(ddof=1):>9.2%}{pct:>8.0%}{ic:>+9.2%}"
              f"{'yes' if m>r.mean() else 'no':>12}")
        rows.append({"slots":slots,"half":lab,"model":m,"rand":float(r.mean()),
                     "rand_sd":float(r.std(ddof=1)),"pctile":pct,"index":ic})
    both = res["early"]>0 and res["late"]>0
    print(f"{'':>6}{'-> both halves positive vs random:':<40}"
          f"{'YES' if both else 'NO'}")
    rows[-1]["both"]=both
pd.DataFrame(rows).to_csv("reports/quantbot_halves.csv", index=False)
B=pd.DataFrame(rows)
n=int(B.groupby('slots').apply(lambda g:(g.model>g['rand']).all()).sum())
print(f"\nslot counts where the model beats random in BOTH halves: {n} of 4")
print(f"slot counts where the model beats the INDEX in both halves: "
      f"{int(B.groupby('slots').apply(lambda g:(g.model>g['index']).all()).sum())} of 4")
