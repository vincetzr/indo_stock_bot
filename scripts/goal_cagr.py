"""What does an ACCOUNT running the goal-clearing rule actually compound at?

The per-trade `ann` column is the single-position answer: one trade at a time,
compounded. A book holding several names at once sits between the mean log and
the arithmetic mean (A18), so where it lands is an empirical question and is
simulated rather than argued.

Cell: buy, target +35%, NO stop, ten-year clock. Selection is irrelevant here --
G4 measured random picking clearing the goal identically -- so the book picks at
random, which is exactly what the rule amounts to.
"""
import sys
sys.path.insert(0,'scripts'); sys.path.insert(0,'src')
import numpy as np, pandas as pd
from quantbot import book, index_cagr

E = pd.read_parquet("data/spine/goalcell.parquet")
E["date"] = pd.to_datetime(E["date"])
r = E["ret"].to_numpy()
bars = float(E["bars"].mean())
ml = float(np.mean(np.log(np.maximum(1+r, .01))))
print(f"cell: target +35%, no stop, 10-year clock. {len(E):,} trades, "
      f"{E.ticker.nunique()} names")
print(f"  positive {(r>0).mean():.1%}   mean per trade {r.mean():+.2%}   "
      f"median {np.median(r):+.2%}   mean hold {bars/252:.2f} yr\n")
print(f"SINGLE POSITION, compounded: {np.exp(ml*252/bars)-1:+.2%}/yr")

_b = book(E, slots=8, rank="rand", seed=0)
ic = index_cagr(_b["start"], _b["end"])
print(f"INDEX over the BOOK's own span ({_b['start']:%Y-%m} -> {_b['end']:%Y-%m}, "
      f"{_b['span']:.1f} yr, total-return basis): {ic:+.2%}/yr")
print("  (measured to the last EXIT, not the last entry: a position opened on")
print("   the final entry date compounds for another 3.4 years, and crediting")
print("   that to the entry window overstated the rate by ~8 points.)\n")
print(f"{'book':<22}{'slots':>7}{'trades':>8}{'total x':>9}{'CAGR':>9}"
      f"{'vs index':>10}")
rows=[]
for slots in (1, 2, 4, 8, 16, 32):
    cg = [book(E, slots=slots, rank="rand", seed=s)["cagr"] for s in range(12)]
    b = book(E, slots=slots, rank="rand", seed=0)
    m = float(np.mean(cg)); sd = float(np.std(cg, ddof=1))
    rows.append({"slots":slots,"cagr":m,"sd":sd,"trades":b["trades"],
                 "total":b["total"],"index":ic})
    print(f"{'equal-weighted':<22}{slots:>7}{b['trades']:>8,}"
          f"{b['total']:>9.2f}{m:>+9.2%}{m-ic:>+10.2%}   (sd across 12 draws "
          f"{sd:.2%})")
pd.DataFrame(rows).to_csv("reports/goal_cagr.csv", index=False)
B = pd.DataFrame(rows)
best = B.loc[B.cagr.idxmax()]
print(f"\nbest book: {best['cagr']:+.2%}/yr at {int(best['slots'])} slots, "
      f"against the index at {ic:+.2%}")
print(f"books beating the index: {int((B.cagr>ic).sum())} of {len(B)}")
print()
print("Rp 100 juta over 10 years:")
for lab, g in (("best book", float(best['cagr'])),
               ("single position", float(np.exp(ml*252/bars)-1)),
               ("the index", ic)):
    print(f"  {lab:<18} Rp {100*(1+g)**10:>7.1f} juta")
