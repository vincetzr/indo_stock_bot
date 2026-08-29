import sys, os
sys.path.insert(0,'scripts'); sys.path.insert(0,'src')
import numpy as np, pandas as pd
from quantbot import (CACHE, FEATS, XS, IDX_YIELD, book, eligible, index_cagr,
                      label, summarise, walk_forward)
D = pd.read_parquet(CACHE)
feats = [f for f in FEATS if f in D.columns] + [f+"_x" for f in XS]
L = eligible(label(D, None, 2.0, 252))
L = L[L["stack"] >= 2].dropna(subset=feats)
print(f"{len(L):,} candidate signals, base positive rate {(L['ret']>0).mean():.1%}")
import os
if os.path.exists("data/spine/quantbot_wf.parquet"):
    W = pd.read_parquet("data/spine/quantbot_wf.parquet")
else:
    W = walk_forward(L, feats, 252)
W["q"] = W.groupby("date")["p"].rank(pct=True)
print(f"{len(W):,} out-of-sample signals "
      f"{pd.Timestamp(W.date.min()):%Y-%m} -> {pd.Timestamp(W.date.max()):%Y-%m}")
for tag, sel in (("all", W), ("top 50%", W[W.q>=.5]), ("top 20%", W[W.q>=.8]),
                 ("top 5%", W[W.q>=.95])):
    r = summarise(sel, tag)
    print(f"  {tag:<9} n {r['n']:>7,}  pos {r['pos']:>6.1%}  mean {r['mean']:>+7.2%}"
          f"  median {r['median']:>+7.2%}  ann {r['ann']:>+6.1%}")
ic = index_cagr(W.date.min(), W.date.max())
print(f"\nTHE BOOK.  index over the same span (total-return basis): {ic:+.2%}/yr")
print(f"{'book':<20}{'slots':>7}{'trades':>8}{'total x':>9}{'CAGR':>9}{'vs index':>10}")
rows=[]
for slots in (1,4,8,16,32):
    for tag, rk in (("model-ranked","p"), ("RANDOM control","rand")):
        b = book(W, slots=slots, rank=rk); b["arm"]=tag
        rows.append(b)
        print(f"{tag:<20}{slots:>7}{b['trades']:>8,}{b['total']:>9.2f}"
              f"{b['cagr']:>+9.2%}{b['cagr']-ic:>+10.2%}")
pd.DataFrame([{k:v for k,v in b.items() if k!='per_slot'} for b in rows]).to_csv(
    "reports/quantbot_book.csv", index=False)
W[["date","ticker","p","q","ret","bars","y"]].to_parquet(
    "data/spine/quantbot_wf.parquet", index=False)
