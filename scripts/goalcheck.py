"""G4 and the missing benchmark, for the cell that clears the goal.

Two questions decide whether "83.5% positive and +12.52% mean" is worth
anything:

  G4  does a RANDOM name, chosen with no skill at all, clear the same goal at
      the same geometry? If yes, the goal is a property of the BARRIERS.
  A19 what does the INDEX return over the SAME holding period? Comparing a
      per-trade return to nothing is the error A19 records as the one that
      manufactures results -- the alternative the reader would actually take
      has to be priced.
"""
import sys
sys.path.insert(0,'scripts'); sys.path.insert(0,'src')
import numpy as np, pandas as pd
from quantbot import CACHE, FEE, MIN_RP
from paint_suite import tick_of
from goalsearch import touch_times

HOR, TP = 2520, 0.30
D = pd.read_parquet(CACHE)
keep = D.groupby("ticker")["rp60"].max()
names = set(keep[keep >= MIN_RP].index)
rows=[]
for tk, g in D.groupby("ticker", sort=False):
    if tk not in names or len(g) < HOR + 60: continue
    p=g["adj"].to_numpy(float); hi=g["hi_raw"].to_numpy(float)
    lo=g["lo_raw"].to_numpy(float); n=len(p)
    med=float(np.nanmedian(g["close_raw"].to_numpy(float)))
    if not np.isfinite(med) or med<=0: continue
    cost=FEE+tick_of(med)/med
    t_up,_=touch_times(hi,lo,p,(TP,),(None,),HOR)
    cen=(np.arange(n)+HOR)>(n-1)
    ok=(~cen)&(g["rp60"].to_numpy(float)>=MIN_RP)&(g["close_raw"].to_numpy(float)>=500)
    if ok.sum()<30: continue
    idx=np.flatnonzero(ok); tu=t_up[0][idx]
    out=np.where(tu<=HOR,1,0); step=np.minimum(tu,HOR)
    j=np.clip(idx+step,0,n-1)
    fill=np.where(out==1, p[idx]*(1+TP), p[j])
    rows.append(pd.DataFrame({"date":g["date"].to_numpy()[idx],"ticker":tk,
        "ret":fill/np.maximum(p[idx],1e-9)-1.0-cost,"bars":step,
        "exit":g["date"].to_numpy()[j],"stack":g["stack"].to_numpy(float)[idx]}))
E=pd.concat(rows,ignore_index=True)

J=pd.read_csv("data/cache/ohlcv/_JKSE.csv.gz",parse_dates=["date"]).set_index("date")["close"]
J=J.sort_index()
YIELD=0.0177
def idx_ret(d0,d1):
    a=J.asof(pd.Timestamp(d0)); b=J.asof(pd.Timestamp(d1))
    if not np.isfinite(a) or not np.isfinite(b) or a<=0: return np.nan
    yrs=max((pd.Timestamp(d1)-pd.Timestamp(d0)).days/365.25,1e-6)
    return (b/a)*(1+YIELD)**yrs - 1.0
E["idx"]=[idx_ret(a,b) for a,b in zip(E["date"],E["exit"])]
E=E.dropna(subset=["idx"])

def rep(lab,S):
    r=S["ret"].to_numpy(); ml=float(np.mean(np.log(np.maximum(1+r,.01))))
    bars=float(S["bars"].mean()); ann=float(np.exp(ml*252/bars)-1)
    print(f"{lab:<34}{len(S):>9,}{(r>0).mean():>10.1%}{r.mean():>+10.2%}"
          f"{np.median(r):>+10.2%}{bars/252:>8.2f}{ann:>+8.1%}"
          f"{S['idx'].mean():>+11.2%}{(r-S['idx']).mean():>+11.2%}"
          f"{'YES' if (r>0).mean()>=.80 and r.mean()>=.04 else 'no':>6}")

print(f"cell: buy, target +{TP:.0%}, NO stop, {HOR}-session ({HOR/252:.0f}yr) clock\n")
print(f"{'arm':<34}{'n':>9}{'POSITIVE':>10}{'MEAN':>10}{'median':>10}"
      f"{'hold yr':>8}{'ann':>8}{'INDEX same':>11}{'vs index':>11}{'GOAL':>6}")
rep("trend filter", E[E["stack"]>=2])
rep("EVERY eligible bar (no skill)", E)
rng=np.random.default_rng(0)
for s in range(3):
    rng=np.random.default_rng(s)
    rep(f"random 20% subsample seed {s}", E.sample(frac=0.2,random_state=s))
print()
print(f"index beats the trade in {(E['idx']>E['ret']).mean():.1%} of cases")
print(f"share of trades that are total-ish losses (<= -80%): {(E['ret']<=-0.8).mean():.2%}")
print(f"mean of the losing 16.5%: {E.loc[E['ret']<=0,'ret'].mean():+.1%}")
E[["date","ticker","ret","bars","idx","stack"]].to_parquet(
    "data/spine/goalcell.parquet", index=False)
