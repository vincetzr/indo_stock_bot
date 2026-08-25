# H25 — the fast-multiplier screen, and the only result here that clears the bar

*443 name-years (tight tier), 234 names, pre-holdout. Code:
`scripts/fastmover.py`. Tests: `tests/test_fastmover.py` (13). Raw:
`reports/fastmover.txt`.*

H23/H24 answered "7 of 10 multi-baggers" at a **ten-year** horizon, and §7 of
that memo shows the same basket is the *wrong side of the trade* below three
years. This is the opposite instrument: maximise P(a name doubles **within a
year**), which is what an emerging-market position is usually bought for.

---

## 1. What maximises a one-year double

Full eligible universe (≥Rp1bn/day), 31,394 name-years, base rate **12.05%**:

| feature | side | n | touch 2x | lift | median | mean | P(−50%) |
|---|---|---|---|---|---|---|---|
| vol60 | top | 3,303 | **19.89%** | 1.65 | −16.7% | +12.1% | 22.0% |
| lowvol | bot | 3,013 | 19.81% | 1.64 | −18.3% | +10.9% | 22.8% |
| **squeeze** | top | 3,124 | **18.53%** | 1.54 | −3.2% | +20.1% | 12.3% |
| rev1 | top | 3,305 | 17.88% | 1.48 | −5.6% | +15.7% | 13.7% |
| amihud60 | top | 3,303 | 16.77% | 1.39 | −8.9% | +11.5% | 15.3% |

**The screen** — most volatile 5% **and** thinnest-traded 20% — reaches
**21.22%**, lift 1.76×, on 443 name-years.

| | |
|---|---|
| clustered permutation null (5,000 draws) | 12.37% ± 1.56% |
| **z** | **+5.66** |
| **p** | **0.00020** vs a Bonferroni bar of **0.00064** after 78 trials |
| half-split | 1.51× early, 2.06× late — **positive in both** |

**It clears. It is the only result in this project that ever has.**

## 2. And it is variance, not skill — which is exactly why it clears

Rank-correlation of each "signal" with `vol60`, per date: `lowvol` **−1.00**
(it *is* vol60 negated), `amihud60` +0.34, `squeeze` +0.31. **There is one
factor here and it is volatility.**

**H13's PREDICTED-NULL control, `squeeze`, ranks THIRD** at 18.53%. A19 records
that when the negative control fires, significance is not evidence. Here it
does more than fire — it places. A volatile name is *mechanically* likelier to
touch any level:

| | |
|---|---|
| P(touch 2x) | **21.2%** |
| P(end below half) | **18.7%** |

**Per 10 names per year: 2.1 double, 1.9 halve.** That is not a discovery, it
is the definition of variance, and the permutation null cannot see it because
it tests "is this cell different from a random cell" — to which the answer is
trivially yes.

## 3. The price: it does not compound

| | |
|---|---|
| arithmetic mean, net of cost | **+16.9%** ← what a rebalanced basket is paid |
| median | **−19.1%** ← what most single names do |
| **mean log** | **−0.1927** → **−17.5% a year compounded** |
| round-trip cost on these names | 1.38% median (vs 0.90% for the liquid decile) |

A single name in this screen compounds to nothing. Ten names rebalanced
annually recover most of it — an equal-weighted basket is paid closer to the
arithmetic mean — but a full allocation still returns **+5.1% a year against
the index's +14.6%**: **3.1× against 22.8×** over 23 years, with a **14.8%
chance of ending below where you started.**

This is A18/H20's lesson arriving a fifth time. Mean, median and mean-log
disagree, and the one that decides a compounding account is mean-log.

## 4. The resolution is position size, not selection

**The number of names that double does not depend on how much money is in
them.** Hold ten screen names and about two double every year whatever the
sleeve weighs. 23 years, 2001–2024, index on a total-return basis:

| sleeve | median CAGR | terminal | P(<1.0x) | doubles seen per year |
|---|---|---|---|---|
| 0% | +14.6% | 22.8× | 0.0% | — |
| 10% | +14.5% | 22.4× | 0.0% | 2.1 of 10 |
| **20%** | **+14.0%** | 20.2× | 0.0% | 2.1 of 10 |
| **30%** | **+13.7%** | 19.3× | 0.0% | 2.1 of 10 |
| 50% | +12.2% | 14.2× | 0.0% | 2.1 of 10 |
| 75% | +9.5% | 8.1× | 0.4% | 2.1 of 10 |
| 100% | +5.1% | 3.1× | **14.8%** | 2.1 of 10 |

Each 10% of the account moved into the sleeve costs roughly **1 point of
CAGR**. At 20–30% you keep 94–96% of the index's compounding and still watch
two names double a year.

## 5. What is honest about this and what is not

**Honest:** the doubling rate is real, measured, clears the bar, and replicates
across halves. The screen is reproducible from a config and a seed.

**Not a discovery:** it is volatility. Anyone can have it, it requires no
model, and its mirror image — the halving rate — comes with it at almost the
same size.

**The live screen tiers, and the tiers carry their own odds.** The tight screen
intersects two filters and on 2026-08-24 yields only **four** names. Handing
back four when ten were asked for, or quoting 21.2% for a loosened screen, are
both ways of being quietly wrong, so `pick_tier` widens and says which tier it
used. Only the tight tier has the permutation null behind it; the wider tiers
have measured odds and **inherited** significance, which is weaker.

**Unchanged limits:** the holdout is spent (H16), and this is the max of ~25
feature cells plus 3 combinations, so the trial count now stands at 78.

---

## 6. What the cost model does not contain — added after "this is a customer's money"

`brief.cost_bar` is fees plus a point-in-time fraksi-harga half-spread. It has
**no market-impact term, no suspension term and no auto-rejection term**, and on
names this thin all three bite. Measured on the same 1,228 screen name-years:

**Capacity.** Median daily traded value of a screen name is **Rp1.77bn**.

| position | % of one day's volume | days to exit at 10% participation |
|---|---|---|
| Rp 50m | 2.8% | 0.3 |
| Rp 100m | 5.6% | 0.6 |
| **Rp 500m** | **28.2%** | **2.8** |
| Rp 1,000m | 56.5% | 5.6 |

**Suspension.** In the twelve months after entry, **4.54%** of sessions are
untradeable on average. **28.1% of name-years spend more than 5% of sessions
untradeable**; 4.0% spend more than 20%.

**Auto-rejection.** 1.40% of sessions are down ≥9% — near or at the ARB band —
which is roughly **four sessions a year per name where selling may be
impossible at any price.**

All three run against the holder, none is in any number in §1–§4, and the year
in which you most want out is the year the name is suspended or locked down.

## 7. If this is someone else's money

**What the screen is:** a volatility sort. Two lines: percentile rank on 60-day
realised vol, and a turnover filter. It is **not** technical analysis — no
pattern, indicator or trend rule enters the selection. It is **not** narrative —
the news layer is quarantined from every statistic and `tests/test_news.py`
fails the build if `spine/` or `features/` imports it. It is **not** guessing —
it is measured on 443 name-years with a clustered permutation null.

**What it is not:** a forecast. It contains **zero directional information.** It
selects names with high variance, which raises the probability of touching *any*
level. That is why 2.1 double and 1.9 halve per ten names a year, and why H13's
predicted-null control ranks third in the same sweep.

**Three reasons it is unsuitable as a discretionary recommendation for a third
party, none fixable from this dataset:**

1. **The holdout is spent** (H16). Every number here is in-sample. There is no
   untouched period left in this data to validate against.
2. **No impact model**, on names where a Rp500m position is over a quarter of a
   day's volume.
3. **No live track record.** This has never been run forward, not once.

A suitability assessment is a regulated judgement about a specific client's
circumstances, objectives and capacity for loss. This repo cannot supply it, and
nothing in it should be presented to a client as a validated strategy.

**The repo's own framing assumes personal capital.** CLAUDE.md §1 calls this a
research programme; A5 fixes costs to *"the user's actual Mandiri schedule"*.
Managing a third party's money is a different activity with different
obligations, and none of the work here was built for it.
