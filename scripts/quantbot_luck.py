"""How much of the book's CAGR is the rule and how much is the draw?

A 252-session hold gives ~18 SEQUENTIAL decisions in 18 years, so a slot's
equity is a product of ~18 draws from a distribution whose mean is +4.83% and
whose median is -2.70%. The spread across draws has to be measured before any
single book number is quotable — this is A18's lesson (a within-sample
consistency statistic over correlated units reads as overwhelming and says
almost nothing) and A15's (500 random tie-breaks spanned -29% to +37%).
"""
import sys
sys.path.insert(0,'scripts'); sys.path.insert(0,'src')
import numpy as np, pandas as pd
from quantbot import book, index_cagr

W = pd.read_parquet("data/spine/quantbot_wf.parquet")
ic = index_cagr(W.date.min(), W.date.max())
print(f"index over the span, total-return basis: {ic:+.2%}/yr\n")
print(f"{'slots':>6}{'MODEL cagr':>12}{'random: mean':>14}{'sd':>8}"
      f"{'p5':>8}{'p95':>8}{'model pctile':>14}{'reads as':>12}")
rows=[]
for slots in (4, 8, 16, 32):
    m = [book(W, slots=slots, rank="p", seed=s)["cagr"] for s in range(12)]
    r = [book(W, slots=slots, rank="rand", seed=s)["cagr"] for s in range(30)]
    r = np.array(r); mm = float(np.mean(m))
    pct = float((r < mm).mean())
    rows.append({"slots":slots,"model":mm,"model_sd":float(np.std(m,ddof=1)),
                 "rand_mean":float(r.mean()),"rand_sd":float(r.std(ddof=1)),
                 "p5":float(np.percentile(r,5)),"p95":float(np.percentile(r,95)),
                 "pctile":pct,"index":ic})
    print(f"{slots:>6}{mm:>+12.2%}{r.mean():>+14.2%}{r.std(ddof=1):>8.2%}"
          f"{np.percentile(r,5):>+8.2%}{np.percentile(r,95):>+8.2%}"
          f"{pct:>13.0%}{'SIGNAL' if pct>0.95 else 'noise':>12}")
pd.DataFrame(rows).to_csv("reports/quantbot_luck.csv", index=False)
print("\nThe model book is ONE realisation; its own spread across seeds is in")
print("the `model` column's sd. If the model's CAGR sits inside the random")
print("arm's 5-95 band, the earlier 8-slot +12.88% was a draw, not a rule.")
b=pd.DataFrame(rows)
print(f"\nmodel beats the index in {int((b.model>b['index']).sum())} of "
      f"{len(b)} slot counts; random beats it in "
      f"{int((b.rand_mean>b['index']).sum())} of {len(b)}")
