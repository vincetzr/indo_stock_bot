# The IDX Plan

*What the evidence in this repository supports, what it rules out, and what
it refuses to promise.*

Everything below is reproducible from the repository:

```bash
python3 scripts/base_rates.py      # what the market pays
python3 scripts/factor_study.py    # thirteen factors, tested
python3 scripts/yield_book.py      # the one survivor, attacked
python3 scripts/the_plan.py        # the book, sized in rupiah and lots
python3 scripts/daily_update.py    # twice daily: position report + plan state
```

---

## 1. The brief, and the three words that had to go

The brief was to dominate IDX with client money *so that profit is close to
guarantee*. Every part of that can be delivered except the last three words, and
it is worth being exact about why rather than gesturing at "markets are risky".

Measured on this repository's own data, over 2015–2026:

- **43% of IDX names lost money** across eleven years, dividends included — in a
  universe that contains **zero delistings**, so the real figure is worse.
- **37% of five-year holding periods were negative**, and 83% came in under a
  bank deposit.
- **Every configuration of the best portfolio found here lost between half and
  two thirds of its value** at some point.

A guarantee is not available at any price. What *is* available is a short list of
things that are arithmetic rather than forecast, plus one premium that has been
measured about as carefully as this data allows. The plan is built strictly in
that order — certainty first, then evidence, then nothing.

---

## 2. What is certain, and therefore where the work goes first

These are not predictions. They are properties of the arithmetic, and they are
where most of the achievable improvement actually lives.

### Cost — 0.56% a round trip, paid whether or not the trade works

Buy 0.28%, sell 0.18% plus the 0.1% sales tax. The only line in this entire
project guaranteed to be exactly what it says. **Trading less is the one return
improvement that cannot fail**, and the plan is built around that fact rather
than around a signal.

### Breadth — worth roughly fifteen points a year, for free

Books drawn **at random** from the eligible universe, so nothing here is a
selection effect (`factor_study.py`):

| names held | arithmetic/mo | geometric/mo | variance drag | turnover cost | CAGR | max DD |
|---|---|---|---|---|---|---|
| 1 | 0.547% | −0.412% | 0.959% | 0.551% | **−10.49%** | −84.8% |
| 5 | 0.513% | 0.167% | 0.345% | 0.529% | −4.13% | −70.3% |
| 20 | 0.578% | 0.350% | 0.229% | 0.445% | −1.16% | −58.2% |
| 50 | 0.542% | 0.339% | 0.203% | 0.275% | +0.79% | −52.5% |
| all (~104) | 0.566% | 0.370% | 0.197% | 0.027% | **+4.27%** | −49.8% |

The arithmetic mean is **flat across the entire range** — as it must be, for
random draws. The whole ~15-point gap is variance drag (0.76%/month) plus
turnover (0.52%/month). No forecast appears anywhere in that table. This is the
single largest, most certain effect in the repository, and it is available to
anyone willing to own more names and trade them less.

### Income — 87% of liquid IDX names pay a dividend

Every study in this repository before this work read `close` instead of
`adj_close` and therefore silently excluded every dividend ever paid. Correcting
it lifts the equal-weight liquid book from **2.23%/yr to 4.27%/yr**. The
omission was larger than any edge this repository has ever claimed to find.

### Tax — the exemption is worth more than most signals

Dividends reinvested in Indonesia are exempt under PP 9/2021. Otherwise 10%
final for a domestic individual, 20% or the treaty rate for a foreign holder. On
the book below that exemption is worth about **Rp 5.8m a year per Rp 1bn**
deployed — for filling in a form.

---

## 3. What the market actually pays

The hurdle, before any strategy is judged against it:

| series | final | CAGR | real | max drawdown |
|---|---|---|---|---|
| equal-weight liquid, total return | 1.54× | 4.27% | 1.23% | −49.8% |
| IHSG price index | 1.31× | 2.62% | −0.37% | −34.9% |
| **rupiah time deposit** (assumed 5.5%) | — | **5.50%** | **2.43%** | **0%** |
| equal-weight liquid, **in USD** | 1.13× | 1.15% | — | −58.3% |

The rupiah went 13,158 → 18,030 per USD, −3.08%/yr.

**Over this sample, a bank deposit beat the Indonesian stock market**, with no
drawdown at all. That is the honest starting point, and any plan that does not
begin by acknowledging it is selling something.

### When you started decided almost everything

| hold | worst | median | best | % negative | % below cash |
|---|---|---|---|---|---|
| 1 year | −44.8% | +2.1% | +75.2% | 43% | 60% |
| 3 years | −13.5% | +0.1% | +17.1% | 48% | 79% |
| 5 years | −3.8% | +1.4% | +12.1% | 37% | 83% |
| 10 years | +1.7% | +3.0% | +4.6% | 0% | 100% |

The ten-year row is **one overlapping stretch of history**, not five
observations, and the report says so rather than letting "100% of the time" do
unearned work.

### And where the sample stops moves the answer more than anything else

| measured through | EW book | IHSG | beats cash? |
|---|---|---|---|
| 2019-12 | 4.52% | 6.63% | no |
| 2023-12 | 1.63% | 5.04% | no |
| 2025-12 | 6.00% | 6.05% | **yes** |
| 2026-08 | 4.27% | 2.62% | no |

**4.37 points of annual return depend only on which year you stop counting.**
The IHSG fell from 8,748 in January 2026 to 6,337 by August. Any strategy
comparison that does not survive this table is a comparison of end dates.

---

## 4. What was tested and does not work

None of this is in the plan, and each was excluded by measurement rather than
by taste.

| ruled out | how | where |
|---|---|---|
| Timing at 15m, 1h, 4h, daily | leg structure indistinguishable from a driftless random walk; median gap in leg/M* of −0.001 | Result 110 |
| Every band rule and multi-timeframe combination | loses to **random timing at the same exposure** | Results 111, 113 |
| Fitting the band on a training half | scores *worse* on the holdout than not fitting | Result 115 |
| Picking the best-performing factor | first-half leader placed **11th of 13** out of sample; the average factor did +2.11% while the chosen one did −2.20% | Result 124 |
| Concentration | one random name returns −10.49%/yr | Result 124 |
| Twelve of thirteen price factors | none significant after Bonferroni | Result 124 |
| `trend`, despite the second-best headline | −1.58% in the first half, +13.86% in the second — one stretch, not a factor | Result 124 |

Three separate times this repository has now found that **fitting a choice makes
it worse**. That is the most reliable negative result here and it shapes the
plan: the rule below has no fitted parameters to speak of.

---

## 5. What survived

**Trailing dividend yield.** It is computable point-in-time without any
fundamental history, because `adj_close / close` *is* the accumulated dividend
factor and its growth over the trailing year is knowable at the decision bar.
The implied payouts were checked against reality before anything was believed —
BBCA Rp 54.9 + Rp 282.4, BMRI Rp 98.6 + Rp 376.7, TLKM Rp 226.7. They match.

Then `yield_book.py` tried to kill it:

| attack | result |
|---|---|
| grid: 3 liquidity floors × 3 frequencies × 4 breadths | **36 of 36 cells** beat both the neutral book and the IHSG. Worst +0.79%/yr, median +4.90% |
| lookback 6m / 1y / 2y | +1.37% / +5.68% / +3.58% — not a fitted window |
| skip the top: ranks 1‑30 → 4‑33 → 11‑40 → 21‑50 → 41‑70 | +5.68% → +3.99% → +2.15% → +0.74% → −8.57%. A **smooth decay down the ranking** is what a premium looks like; three lucky names would fall off a cliff after the first row |
| calendar years | ahead in **8 of 11** |
| both halves of the sample | +2.29%, then +9.82% |
| the 2026 crash | **−0.9%** from the index peak, against −16.4% neutral and −27.6% IHSG |
| persistence | 89% of the book carries over month to month; yield rank correlates 0.73 with itself a year later |
| is it one sector? | within-book correlation 0.202 against 0.157 for a random book — no |
| rank IC | 0.079, t = 5.55, p < 0.0001, **significant after Bonferroni for thirteen looks** |

### Three things held against it

1. **It is trailing, not forward.** It ranks what was already paid. A company
   that paid well and then cuts still ranks high on the day the cut is
   announced, and the book will be holding it.
2. **It is a value factor wearing an income name.** Yield is dividend over
   price, so a name whose price halved ranks higher on an unchanged payout. Part
   of what is being measured is simply buying what has fallen.
3. **The drawdown is not smaller.** Every grid cell lost between half and two
   thirds at some point. This is a return improvement, not a risk one.

---

## 6. The plan

### The rule, in full, so it can be followed without me

| | |
|---|---|
| **universe** | IDX equities with median daily turnover ≥ Rp 5bn over the trailing year and at least 250 sessions of history |
| **rank** | dividends paid over the trailing 12 months, divided by price |
| **hold** | the top 30, equal weight |
| **rebalance** | **once a year**, last session of December, executed the next session |
| **never** | no timing overlay, no leverage, no shorting, no concentration |

Quarterly and monthly rebalancing also work. Annual costs **0.19%/yr** against
0.73% monthly and the grid showed no return bought with the difference, so
annual wins on the one axis that is certain.

### What to plan on — deliberately less than what was measured

| | |
|---|---|
| measured in the sample | 9.93%/yr |
| of which the neutral book | 5.11% |
| of which the yield premium | +4.82% |
| **planned on (half the premium)** | **7.52%/yr** |

The haircut is a convention, stated as one. An estimated premium is the true
premium plus whatever the estimate got wrong, and here the error has a known
direction: the universe holds no delistings, and thirteen factors were looked at
before one was chosen. A test forbids that haircut from ever exceeding 1.0, so
the plan cannot be quietly talked upward later.

| tax position | net | real | vs cash |
|---|---|---|---|
| domestic, dividends reinvested in Indonesia | **7.52%** | 4.38% | **+2.02%** |
| domestic, dividends taken as cash | 6.84% | 3.73% | +1.34% |
| foreign holder | 6.16% | 3.07% | +0.66% |

### What it costs to earn that

Worst drawdown of this exact book in the sample: **−55.8%**. On Rp 850m of
equity that is a fall to Rp 375m — **a loss of Rp 475m**, written down before
the money moves rather than explained afterwards.

**Put plainly: you are being asked to accept a halving in order to earn about
two points a year over a bank deposit.** If that trade is not wanted, the
deposit is the better answer and saying so is the job.

37% of five-year holds in this universe were negative, so money that might be
needed inside five years does not belong in this book at all.

### Sizing

- **15% in a rupiah deposit.** Not a market view — it is what lets the annual
  rebalance happen without selling into a fall.
- **Minimum capital: Rp 675m** to hold 30 names within 10% of equal weight.
  Below that, hold *fewer* names rather than uneven ones; the breadth table says
  20 names costs little and 5 costs a great deal.
- IDX trades in lots of 100 shares, so `the_plan.py` prints whole lots, the
  weight error on each, the entry commission, and the trailing income in rupiah.

### When to abandon it — decided now, not later

A rule with no exit is a rule that gets rationalised. Committed before any money
moves, and checked automatically on every run:

1. The book underperforms the equal-weight liquid universe for **three
   consecutive calendar years**. In the sample it was behind in 3 of 11 and
   never twice running.
2. The average trailing yield of the top 30 falls **below the deposit rate**.
   There would then be no income premium left to harvest. It is 6.8% today.
3. The liquidity floor stops clearing 30 names.

**Nothing else.** Not a bad quarter, not a drawdown, not a headline.

---

## 7. Operating it

`scripts/daily_update.py` runs at **07:00 WIB** (two hours before the open) and
**18:00 WIB** (two hours after the close, once the broker summary is out). It
prints two things that are not the same kind of thing:

- **The band section** is a *position report*. Where every large cap stands on
  each timeframe and the exact price that flips it. Those numbers are arithmetic
  and exactly right; they say nothing about where price goes.
- **The plan section** is the part with a measured premium behind it: what the
  book holds, what has drifted out of the top 30, how many days until the
  December rebalance, and whether any abandon condition has gone live.

The drift list is **information, never an instruction**. Acting on it monthly
instead of annually is precisely what turns a 0.19%/yr strategy into a 0.73%/yr
one, for no measured return.

---

## 8. The part that is still open

**Layer 2 — broker flow — has never been tested with enough data to detect an
effect worth having.** That is a different statement from having been tested and
failed, and the difference matters. `scripts/layer2_protocol.py` freezes
hypotheses H1–H4 and hashes them (`6b8e0a2c9d1f4e73`) so they cannot be edited
once the answer is visible; `scripts/broker_collect.py` accumulates the daily
store. When 250 complete days exist on a name, the protocol runs and this plan
is revisited on the result.

Until then the plan assumes nothing about it, which is the entire reason the
hypotheses were frozen in advance.

**Layer 1 — news — remains untested** for want of a point-in-time news set.
Ranking history on today's headlines is the same look-ahead error that rules
`data/cache/fundamentals` out of the factor study.

---

## 9. What this plan does not claim

- It does not claim the next eleven years resemble the last eleven. The sample
  holds one commodity cycle, one pandemic and one crash.
- It does not claim the yield premium is permanent. It is a well-documented
  global effect that has also spent decades at a time not working.
- It does not claim low risk. The drawdown is the same as everything else here.
- It does not claim to beat the deposit rate by much, and under a foreign
  holder's tax it barely does at all.

What it does claim is that of everything tested in this repository, this is the
only construction that survived a Bonferroni correction, a walk-forward, a
grid, a split-half, a skip-the-top, a year-by-year, a crash, and a tax
calculation — and that its two largest components, breadth and low turnover,
are arithmetic rather than forecast and therefore cannot stop working.
