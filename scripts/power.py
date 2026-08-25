#!/usr/bin/env python3
"""Is the holdout big enough to answer the question? Run this BEFORE spending it.

    python3 scripts/power.py

Before spending the holdout: is it big enough to answer the question?

CLAUDE.md 11 reserves the last 24 months to be touched ONCE. Running an
underpowered test there does not fail honestly - it burns the only clean
sample on a coin flip. So the power comes first, computed entirely on
pre-holdout data.
"""
import sys; sys.path.insert(0,'src')
import numpy as np, pandas as pd
from idxbot.report import brief as B

P = pd.read_parquet('data/spine/price_panel.parquet')
R = B.run_state(P)
D, E = B.conditional_frame(P, R, k=20)
print(f"reference frame: {len(D):,} rows, {D.date.min().date()} -> {D.date.max().date()}")

# THE PRE-REGISTERED FAMILY, fixed before any holdout contact:
# leg == up AND extension tercile == 2 (the most stretched third of advances).
# NOT the argmax cell - that is the selection O1 warns about. This subset is
# the momentum reading, motivated a priori, 9 of the 54 cells.
D['leg_'] = D.bucket.str.split('|').str[0]
D['ext_'] = D.bucket.str.split('|').str[2].astype(int)
sel = (D.leg_ == 'up') & (D.ext_ == 2)
print(f"selected family: {int(sel.sum()):,} rows, {sel.mean():.1%} of the frame")

# per-period excess of an equal-weighted long-only basket over all liquid names
per = []
for d, g in D.groupby('date'):
    s = g[(g.leg_=='up') & (g.ext_==2)]
    if len(s) < 5: continue
    per.append({'date': d, 'n': len(s),
                'exc': float(s.fwd20.mean() - g.fwd20.mean())})
S = pd.DataFrame(per).sort_values('date')
print(f"periods with >=5 names: {len(S):,}  median basket size {S.n.median():.0f}")

# non-overlapping 20-session rebalances, which is how it would be traded
S['blk'] = np.arange(len(S)) // 20
NB = S.groupby('blk')['exc'].mean()
mu, sd = NB.mean(), NB.std(ddof=1)
print(f"\nIN-SAMPLE, non-overlapping 20-session rebalances:")
print(f"  n periods {len(NB)}   mean excess {mu:+.3%}   sd {sd:.3%}"
      f"   t {mu/(sd/np.sqrt(len(NB))):+.2f}")

COST = 0.0090   # 56bps + ~34bps median half-tick, from the watchlist output
print(f"  after a {COST:.2%} round trip per rebalance: {mu-COST:+.3%}")

# how many non-overlapping periods does the holdout actually contain?
H = P[P.holdout.astype(bool)]
hd = pd.to_datetime(H.date)
n_sessions = hd.nunique()
n_periods = n_sessions // 20
print(f"\nHOLDOUT: {n_sessions} sessions -> {n_periods} non-overlapping 20-day periods")

print("\nPOWER, at the in-sample effect size and dispersion:")
for eff in (mu, mu-COST):
    se = sd/np.sqrt(n_periods)
    t = eff/se
    # two-sided power at Bonferroni alpha for 42 trials
    from scipy import stats
    for alpha, lab in ((0.05,'nominal 0.05'), (0.05/42,'Bonferroni 42 trials')):
        crit = stats.t.ppf(1-alpha/2, n_periods-1)
        pw = 1 - stats.t.cdf(crit - t, n_periods-1) + stats.t.cdf(-crit - t, n_periods-1)
        print(f"   effect {eff:+.3%}  se {se:.3%}  t {t:+.2f}   "
              f"alpha {lab:<24} power {pw:.1%}")

print("\nWHAT EFFECT WOULD THE HOLDOUT ACTUALLY DETECT at 80% power?")
from scipy import stats
for alpha, lab in ((0.05,'nominal'), (0.05/42,'Bonferroni')):
    crit = stats.t.ppf(1-alpha/2, n_periods-1)
    need = (crit + stats.t.ppf(0.80, n_periods-1)) * sd/np.sqrt(n_periods)
    print(f"   alpha {lab:<12} minimum detectable excess {need:+.2%} per 20 sessions"
          f"  ({need*12.6:+.1%}/yr)")
