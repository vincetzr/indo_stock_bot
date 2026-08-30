"""Does the cost assumption change the conclusion? Run it rather than argue it."""
import sys; sys.path.insert(0,'scripts'); sys.path.insert(0,'src')
import numpy as np, pandas as pd
import beatindex as B
from rebalance import Prices

P = B.load(); PX = Prices(P); dates = np.sort(P["date"].unique())
ARMS = [
    ("fees only, PERFECT passive fills", 0.0056, 0.0),
    ("fees + HALF a tick (patient)",     0.0056, 0.5),
    ("fees + a FULL tick (my default)",  0.0056, 1.0),
    ("institutional fees + half tick",   0.0015, 0.5),
]
print("H52's best arm (full universe, equal weight, quarterly) at each cost model.")
print("The index is a ZERO-cost, zero-turnover benchmark in every row.\n")
print(f"{'cost model':<36}{'cost/yr':>9}{'CAGR':>9}{'INDEX':>9}{'vs index':>10}"
      f"{'random':>9}{'GROSS edge':>12}")
rows=[]
for lab, fee, sm in ARMS:
    r = B.run(P, PX, dates, 63, 0, None, "equal", fee=fee, spread_mult=sm)
    ctl = [B.run(P, PX, dates, 63, 0, None, "equal", seed=s, fee=fee,
                 spread_mult=sm) for s in range(6)]
    ic = B.index_over(r["start"], r["end"])
    rnd = float(np.mean([c["cagr"] for c in ctl]))
    gross = r["gross"] - float(np.mean([c["gross"] for c in ctl]))
    rows.append({"model":lab,"fee":fee,"sm":sm,"cost_yr":r["cost_yr"],
                 "cagr":r["cagr"],"index":ic,"vs":r["cagr"]-ic,"rand":rnd})
    print(f"{lab:<36}{r['cost_yr']:>9.2%}{r['cagr']:>+9.2%}{ic:>+9.2%}"
          f"{r['cagr']-ic:>+10.2%}{rnd:>+9.2%}{gross:>+12.2%}")
pd.DataFrame(rows).to_csv("reports/costsens.csv", index=False)
d = rows[2]["vs"] - rows[0]["vs"]
print(f"\n  moving from a full tick to zero spread is worth {d:+.2%} a year,")
print(f"  and takes the screen from {rows[2]['vs']:+.2%} to {rows[0]['vs']:+.2%} "
      f"against the index.")
print(f"  arms beating the index: {sum(1 for r in rows if r['vs']>0)} of {len(rows)}")
