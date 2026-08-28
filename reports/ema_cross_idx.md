# H30 — is there a triple-EMA golden cross that works on IDX?

*744 names, 5,879 pre-holdout sessions, 2000–2024, 60-session forward hold.
Code: `scripts/ema_cross_idx.py`. Raw: `reports/ema_cross_idx.txt`.*

"Trained on IDX" has to mean *measured* on IDX. Porting 50/100/200 because it
is conventional would be the opposite of training — those lengths are US equity
folklore and nothing about them is Indonesian. So all 36 (fast, mid, slow)
combinations were gridded and scored identically.

**Pre-registered:** *no configuration beats buy-and-hold after costs; what I
expect to survive is the alignment STATE as a descriptive conditional, not the
cross as a trigger.*

**Half wrong, half exactly right.**

---

## 1. The benchmark that makes the rest readable

Unconditional, every eligible name-bar, 60 sessions forward:

| | |
|---|---|
| mean | +3.88% |
| median | +0.00% |
| **mean log** | **+0.0021** → **+0.9%/yr** |

**The mean is volatility drag in disguise.** A +3.88% mean with a 0.00% median
and a near-zero mean-log is the signature, and it is why every row below is
scored on mean-log.

## 2. The cross is not a trigger

Best of 36 was `20/21/100`: entering on the bar the stack first aligns and
holding 60 sessions gives mean **+4.25%** net of cost — which does edge
buy-and-hold's +3.88%, so the prediction was wrong on that. But:

| | |
|---|---|
| median | **−0.62%** |
| win rate | **48.5%** |
| mean log | +0.0065 |
| **half-split mean log** | **+0.0274 early, −0.0145 late** |

**It does not compound positively in both halves.** Every one of the 36 rows
has a positive mean, a negative median and a sub-50% win rate — the same
mean/median wedge that withdrew H17 and H18 and left H25 compounding at
−17.5%/yr on a +16.9% arithmetic mean.

Note also that the winning config has **fast = 20, mid = 21**. The fast/mid
distinction contributes nothing; the grid is picking noise on that axis.

## 3. The state works, and a single moving average does not

| state | n | mean log | vs base | early | late | both beat base? |
|---|---|---|---|---|---|---|
| price > EMA100 | 253,136 | +0.0159 | +0.0137 | +0.0308 | +0.0009 | **no** |
| 50>100>200 | 205,790 | +0.0142 | +0.0121 | +0.0270 | +0.0014 | **no** |
| 13>34>100 | 195,393 | +0.0197 | +0.0175 | +0.0331 | +0.0063 | YES |
| **price>50>100>200** | 142,797 | **+0.0218** | +0.0197 | +0.0327 | **+0.0109** | **YES** |

**The full stack is what survives.** `price > EMA100` alone and the bare
`50>100>200` ordering both collapse to the base rate in the recent half —
+0.0009 and +0.0014 against a base of +0.0021. Requiring price to be *above*
the stack **and** the stack to be ordered is what holds up in both eras, at
+0.0218 against +0.0021.

So the conventional 50/100/200 lengths do survive — but only with the price
condition attached, and that is a measured result rather than an inherited one.

## 4. A bug that cost a whole table

The first run reported 174,338 "cross" events and the identical number of
"aligned" bars in every row. That is impossible, and it was the giveaway.

`DataFrame.shift()` on a boolean frame returns **object** dtype with NaN in the
first row. `.fillna(False)` leaves it object, and `~` on a Python bool is
integer negation — `~True` is `-2`, `~False` is `-1`, and **both are truthy**.
So `aligned & ~prev` silently evaluated to `aligned`, and the first table
measured the persistent state while labelling it the trigger. `.astype(bool)`
is the fix and it is load-bearing.

## 5. What ships

`pine/IDX_Suite.pine` plots the stack and **tints by state**, and it
deliberately does **not** fire an entry alert on the crossover — the
measurement says the cross is not a trigger, so turning it into one would
contradict the evidence the indicator is built on.

**Unchanged limits:** in-sample (the holdout was spent at H16), a 60-session
hold, per-name rather than portfolio, and the best row is the best of 36 so its
edge is the maximum of 36 draws and biased upward.
