# H55 — multiple timeframes: the data exists, the confluence does not, and the fill is worth 1.1%

*2026-09-03. `scripts/mtf.py` (daily/weekly/monthly, 2000–2026, 919 names),
`scripts/mtf_h1.py` + `scripts/build_h1_panel.py` (hourly, 2023–2026, 766 names,
3,114,456 bars). Pre-registration M0–M5 and H1a–H1c in the two module
docstrings, written before any cell was scored. Cost = 0.56% fee + half a
fraksi-harga tick, quoted separately per A38.*

---

## 0. The answer

**Yes, mechanically.** Two bar frequencies are available and both are now joined
to the point-in-time spine: daily 2000-03-30 → 2026-08-28 across 919 names, and
**hourly 2023-07-25 → 2026-08-07 across 766 names** — 3.1 million bars that had
been sitting in `data/cache/intraday/` since an early task and that **no
hypothesis H1–H54 had ever used**.

**No, usefully, in three separate ways, each measured rather than argued:**

1. **Trading the hourly bar is arithmetically dead.** One round trip costs
   **1.65×** the median hourly move. Only **32.0%** of hourly bars move further
   than a single round trip.
2. **Multi-timeframe confluence carries no information beyond the market's own
   regime.** Confluence sits **+0.40 clustered-null sd**; the pre-registered
   foreign-trend null — confluence with a *randomly chosen other company's*
   weekly trend — sits at **+0.39**. Indistinguishable.
3. **Intraday entry timing has a hard ceiling of ~1.1%, and real rules do not
   reach it.** Buying at the day's single best hourly close with perfect
   hindsight is worth **+0.0115 of log** — **1–2% of one standard deviation** of
   the outcome. The one realistic confirmation rule tested makes the fill
   **worse**.

---

## 1. The hourly data is real, and the first thing to establish is that it joins

Before any of it can be used it has to reconcile with the spine, because a
timeframe that cannot be joined to the daily panel cannot be compared to a
single earlier result in this repo.

| check | result |
|---|---|
| last hourly close vs the panel's **raw** close | **exact on 99.54%** of 65,415 name-days |
| last hourly close vs the panel's **adj_close** | 51.75% |
| session times present | 09:00–11:00, **12:00 absent (0.07%)**, 13:00–16:00 |
| bars per name-day, mode | **7** |
| bars with zero volume | 13.8% |
| bars with no range at all (o=h=l=c) | **30.2%** |
| bars with zero return | **36.9%** |

Two things follow and both are load-bearing. The lunch break is **present**, not
filled, so consecutive "hourly" bars straddle a 90-minute gap once a day. And
the bars are on the **unadjusted** basis — the 51.75% match to `adj_close` is
exactly the subset of names with no corporate action in the window — so every
hourly price is multiplied by the panel's own `adj_close/close` factor before
use. Skipping that turns a split inside the window into a fake crash, which is
the defect A2 records the spine already having had once.

---

## 2. The cost wall, which is most of the answer

| | |
|---|---|
| median absolute **hourly** return | **0.546%** |
| median absolute **daily** return | 0.746% |
| median fraksi-harga tick, as % of price | 0.680% |
| round trip, **fee only** | 0.560% = **1.02× a median hourly move** |
| round trip, **fee + half a tick** | 0.900% = **1.65× hourly**, 1.21× daily |
| round trip, fee + a full tick | 1.240% = 2.27× hourly, 1.66× daily |
| hourly bars that clear one round trip | **32.0%** |

This is subtraction, not a hypothesis. On IDX retail costs the toll on a single
hourly round trip is larger than the bar it is trying to capture, and larger
than a median **daily** bar as well. It is the same wall A9 measured from four
directions and A39 measured again, and at an hourly horizon it is at its
steepest.

**What that does *not* close** is whether a different timeframe helps as a
**filter**, because a filter adds no round trips at all — it changes which day
or which hour you buy, not how often you trade. That is the rest of this memo.

---

## 3. M0 — the positive control failed, and the failure was the control

Nothing below was read until the harness proved it could see a planted effect.
The first version of that check **failed**, at 0.0056 of a planted 0.0640 — and
it was wrong, because it asked two questions at once.

| | |
|---|---|
| **M0a** harness recovers, handed the **true** regime | **+0.1325 of a planted +0.1920 — 69%** |
| **M0b** what a **weekly EMA** transmits of that regime | **+0.0359, a ratio of 0.27** |

The M0a shortfall is arithmetic: a forward window straddles regime flips. The
M0b ratio is not a failure at all — it is a **measured ceiling**, and it is one
of the more useful numbers here:

> **A higher-timeframe EMA state carries about a quarter to two-fifths of
> whatever regime information exists** — 0.22 at a fast-turning regime, 0.40 at
> a slow one — **before any of it has to survive a cost.**

That is what a weekly trend filter *is*, measured on data where the truth is
known. This is A36's Q0 pattern exactly, where an instrument check failed four
times and every failure was the test; the resolution is to decompose, never to
relax the threshold (CLAUDE.md §2).

---

## 4. M1–M4 — confluence, on 2.86m bars, 2000–2026

Weekly state = weekly close above its 10-week EMA, stamped at the first daily
bar **strictly after** that week closed. Daily trigger = close crosses above the
20-day EMA with the EMA rising. Mean log net of cost is the column a sequential
trader compounds at (A36); mean is what an equal-weighted holder receives (A18).

**Horizon 60 sessions:**

| cell | n | mean log | mean | median | win |
|---|---|---|---|---|---|
| eligible (all bars) | 688,745 | −0.0287 | +0.85% | −2.30% | 44.1% |
| HTF only (weekly up, no daily trigger) | 99,874 | −0.0233 | +1.24% | −2.29% | 43.9% |
| LTF only (daily trigger, weekly down) | 44,573 | −0.0293 | +0.72% | −2.83% | 43.1% |
| **CONFLUENCE (both)** | 232,251 | **−0.0053** | **+3.34%** | −0.91% | 47.6% |
| neither | 312,047 | −0.0478 | −1.10% | −3.23% | 41.8% |
| LTF alone, **matched count** | 231,573 | −0.0086 | +2.86% | −1.00% | 47.2% |
| **FOREIGN-TREND null (M3)** | 76,917 | **−0.0082** | **+3.31%** | −1.27% | 46.7% |

**M2 CONFIRMED.** The 2×2 separates, and the higher-timeframe term is the
larger one as predicted: HTF-only −0.0233 against LTF-only −0.0293, with
"neither" at −0.0478 and both at −0.0053.

**M1 fails on the condition that matters.** Confluence does beat the
matched-count daily trigger on mean log (−0.0053 vs −0.0086), and does so in
both halves. But:

**M3 — THE PREDICTED NULL — FIRED, AND IT IS THE FINDING.** Pairing each name
with a **different, randomly chosen company's** weekly trend, holding the
marginal frequency of "weekly up" fixed, reproduces the effect almost exactly:
mean **+3.31% against +3.34%**. Against the clustered permutation null (whole
(ticker, year) blocks reassigned, per A17/A25):

| cell | mean log | null mean | null sd | **z** |
|---|---|---|---|---|
| CONFLUENCE (both) | −0.0053 | −0.0069 | 0.0040 | **+0.40** |
| any daily trigger | −0.0091 | −0.0058 | 0.0036 | −0.93 |
| any weekly up | −0.0107 | −0.0044 | 0.0034 | −1.82 |
| **FOREIGN-TREND null** | −0.0082 | −0.0107 | 0.0063 | **+0.39** |

**+0.40 against +0.39.** The registered bar was 2 null-sd above the foreign
trend; the gap is 0.01. What "multi-timeframe confluence" measures is that
everything trends together in a bull market — the market's regime, not the
agreement of two views of one stock. A22 is the same shape: a screen that
cleared Bonferroni and meant nothing because it selected variance.

**M5 — the benchmark, priced over each entry's own window.** Paired per entry
against the IHSG on a total-return basis:

| horizon | confluence beats the index on | foreign-trend null |
|---|---|---|
| 20 sessions | 42.8% of entries | 42.8% |
| 60 | 43.3% | 43.3% |
| 126 | 42.3% | 42.8% |
| 252 | **40.3%** | 40.2% |

Confluence at one year has a **higher mean** than the index (+13.89% vs +9.72%)
and beats it on **40.3% of individual entries**. The mean is carried by a right
tail; the typical entry loses to simply owning the index. And the foreign-trend
null is identical on the paired statistic.

*(The unpaired "excess" column flatters the foreign-trend arm — +5.78% against
confluence's +4.17% at one year — purely because it fires on different dates and
so faces a different index window. That is A19's error class appearing inside my
own comparison table; the paired beat-rate is the window-safe read and it is the
one quoted.)*

**M4.** Every cell is negative on mean log at every horizon, and **every cell is
positive early and negative late** — confluence at 252 runs early +0.0191, late
−0.0594. Nothing here is positive in both halves in absolute terms.

---

## 5. H55b — what an hour of timing is actually worth

Same trade, same cost, different fill. The **ORACLE** arm buys at the day's
lowest hourly close with perfect hindsight; it is a look-ahead by construction
and exists to **bound** the question, because no real rule can beat it.

| entry arm | h=5 | h=20 | h=60 |
|---|---|---|---|
| daily close (baseline) | −0.0038 | −0.0111 | −0.0365 |
| day's open (first hour) | −0.0062 | −0.0136 | −0.0389 |
| mid-session, 4th hour | −0.0037 | −0.0110 | −0.0364 |
| first hour above the hourly EMA | −0.0034 | −0.0104 | −0.0338 |
| **ORACLE (look-ahead)** | **+0.0077** | **+0.0003** | **−0.0253** |
| **oracle's edge over the close** | **+0.0115** | **+0.0115** | **+0.0112** |
| **as a share of the horizon's return sd** | **0.022** | **0.018** | **0.011** |

> **Perfect intraday hindsight is worth +1.1%, once, and that is 1–2% of one
> standard deviation of the thing it is trying to improve.** It is also flat
> across horizons, which is the signature of a one-off entry improvement rather
> than an edge. No real rule can exceed it, so the intraday-timing route is
> closed by a single measurement rather than by testing a hundred rules.

**And the one plausible rule reads backwards once matched.** The
hourly-EMA-confirmation arm looked like a small gain (−0.0104 against the
close's −0.0111 at h=20). It fires on **24% fewer days** — some days have no
hour above the hourly EMA — so at its natural size it mixes *which hour* with
*which day*. Restricted to exactly the days it fired:

| horizon | FILL edge (which hour) | DAY-SELECTION edge (which day) |
|---|---|---|
| 5 | **−0.0009** | +0.0012 |
| 20 | **−0.0009** | +0.0016 |
| 60 | **−0.0010** | +0.0036 |

**Opposite signs.** Waiting for hourly confirmation costs about a tenth of a
percent in fill at every horizon; all of the apparent benefit was the day.
**H1a's predicted null holds, and the naive reading was wrong in the flattering
direction.** A35's lesson, which is now the third time it has caught something
here: ask of every headline what the thing in between is, and whether it was
priced.

---

## 6. What this leaves standing

**Measured and negative:** hourly trading (cost wall), multi-timeframe
confluence (foreign-trend null), intraday entry timing (oracle bound), hourly
trend confirmation (matched fill edge).

**Measured and useful:** the weekly-EMA transmission ratio of **0.27** — the
first number in this repo that says *how much* a higher-timeframe filter can
carry — and the confirmation that the **higher** timeframe is the more
informative half of a two-timeframe pair, which is at least the opposite of how
retail multi-timeframe advice is usually ordered.

**Untouched by this study:** the cross-sectional results. H54's sticky
strength+calm beat the index in 6 of 6 rebalance calendars by a median +6.50% a
year. That is a *selection* result at a *quarterly* cadence, and nothing here
displaces it — which is itself the finding, because the whole multi-timeframe
programme was an attempt to add timing to it and timing is what does not work.

---

## 6b. "How reliable" — the numbers this repo can honestly quote

A single accuracy percentage would be a lie, and A28 records why: the panel
emits **probabilities, not calls**, so a hit rate is undefined until someone
picks a threshold, and whoever picks the threshold decides the answer. What is
defined is calibration, skill against the base rate, AUC, and band coverage.

**The cone's probability laws** (`reports/calibrate.txt`), where the
walk-forward arm is the only genuinely out-of-sample number in this project —
for test year Y the laws are refitted only on windows that closed before Y began:

| arm | Brier | skill vs base rate | AUC | date-band coverage (claims 0.500) |
|---|---|---|---|---|
| shipped constants | 0.1861 | +0.0374 | 0.591 | **0.497** |
| **purged walk-forward** | 0.1912 | **+0.0130** | **0.580** | **0.500** |
| whole board, 777 names | 0.1917 | +0.0168 | 0.594 | 0.475 |

Calibration is good and the **date band is the best-behaved number in the
project**. Skill is thin, and it is not evenly spread: AUC is 0.648 at −50% and
0.632 at 2x, but **0.502 at +5% and 0.524 at +10% with negative skill** — almost
everything touches +5% within a year, so there is nothing to discriminate. The
race law (which barrier arrives first) has **AUC 0.51 and BSS 0.000**: it prices
a decision you have already made and cannot make one for you.

**The horizon is the condition, and dropping it makes any of these wrong.**
P(a name touches 2x), 725 names, 31,417 monthly cohorts:

| hold | all liquid | liquid decile | per 10 names |
|---|---|---|---|
| 1 year | 9.5% | **4.2%** | 0.4 of 10 |
| 3 years | 27.0% | 24.6% | 2.5 of 10 |
| 10 years | 55.5% | **70.0%** | **7.0 of 10** |

The tilt **inverts**: at one year the most liquid names double *less* often
(4.2%) than the liquid names they exclude (10.2%). Effective n at ten years is
**56**.

**The one setup that clears this repo's own Bonferroni bar** is H26's
strength+calm at a one-year horizon: skew (P(2x) / P(halve)) **2.60** against a
null of 1.20 ± 0.15, **z = +9.44, p = 0.00033** against a bar of 0.00061, halves
2.65 / 2.53, P(halve) **4.1%**. Base skew rises monotonically with horizon —
1.33 / 1.48 / 1.57 / 1.80 / 2.26 at 1/2/3/5/10 years — so the screen buys about
six years of that at a one-year hold.

**And the cadence is not resolvable below a quarter.** H43's rebalance sweep
(2.86m rows, 6 holds × 8 phases) is humped, peaking at a quarter (+11.97%) and
six months (+11.99%), but the spread across frequencies (3.38%) barely exceeds
the spread across *phases within* a frequency (2.93%), a ratio of 1.15 — so
**monthly, quarterly and six-month are mutually indistinguishable**. There is no
evidence here for trading more often than quarterly, and none for less.

---

## 7. Trials

M0–M5 and H1a–H1c is 9 registered tests. **Trials after H55: 332.** Bonferroni
bar α = 0.05/332 = **0.00015**. Every result here is negative or is a
measurement rather than a claim, so nothing is asserted against it. The holdout
was spent at H16. The hourly panel covers 3.03 years and 766 names, which for a
confluence claim is roughly 3 independent macro observations — the same
effective-n problem A20 and A39 record, and no resampling manufactures more.
