# H23 — the horizon, and the first thing in this project that beats the index

*31,417 monthly name-cohorts, 725 names, 2000-05 → 2024-08, pre-holdout only.
Code: `scripts/horizon_sweep.py`. Tests: `tests/test_horizon_sweep.py` (11).
Raw: `reports/horizon.txt`.*

**The question: can 8 of 10 names double, and if not, what is the ceiling?**

Every P(2x) in this repo — H13, H16, H18, H21 — is measured at **252 sessions**,
because that is the horizon H16 happened to choose. A doubling rate is mostly a
function of how long you wait, and A9 named "a horizon long enough that turnover
stops mattering" as untested and then never tested it. So the question had only
ever been asked at one year.

Two definitions of a hit, and the gap between them is the point:

- **touched 2x** — the path reached 2x at some point. What a take-profit order
  captures. H16 measured a mean *peak* of +102.2% against a realised +15.1%.
- **ended 2x** — buy-and-hold. All a terminal-return study can see.

---

## 1. The ceiling, with survivorship handled

Deaths valued at last traded price. `IHSG 2x` is the tide: a hit rate near it is
the index in a costume.

| horizon | n | eff n | touched 2x | ended 2x | profit | P(−50%) | median | IHSG 2x |
|---|---|---|---|---|---|---|---|---|
| 1y | 15,771 | 1,314 | 9.5% | 5.0% | 47.3% | 8.5% | −2.2% | 0.8% |
| 2y | 15,655 | 652 | 18.9% | 10.1% | 46.8% | 14.6% | −3.7% | 6.7% |
| 3y | 14,057 | 390 | 27.0% | 14.4% | 46.8% | 19.1% | −5.2% | 11.1% |
| 5y | 10,934 | 182 | 39.0% | 22.2% | 49.8% | −0.3% | −0.3% | 28.8% |
| 7.5y | 8,689 | 97 | 46.7% | 26.7% | 50.0% | 27.9% | +0.0% | 40.3% |
| **10y** | 6,683 | **56** | **55.5%** | 32.3% | 52.3% | 28.6% | +6.7% | 48.6% |

**80% is not reached at any horizon the data can measure.** At 10 years the
unconditional touch rate is 55.5%, so 8 of 10 would need a **1.44× lift** — and
that is the smallest gap in the table. At one year it needs 8.4×, against the
3.24× that is the largest lift ever measured in this project.

**Two bugs made the first version of this table say 73.8% at 7.5 years.**
`MU.PX` is a list of *cut edges* for the price bucket, not a `[min, max]` pair,
so `close >= PX[0] & close <= PX[1]` restricted the universe to sub-Rp50 names —
the penny board, 336 tickers. And eligibility was applied to *every* bar, so a
name dropping out of the universe mid-hold had its path cut there; eligibility
is a condition for **buying**, and once held the path is whatever the name does.
Requiring a full window on top of that discarded **91% of 7.5-year cohorts** and
measured the doubling rate of the names that lived.

## 2. Does any signal deliver the 1.44×?

The one-year lift is not transferable and has to be measured where it would be
used. Top and bottom decile of every registered feature, `squeeze` kept as
H13's predicted-null control.

**At 3 years** (base 27.0%, 80% needs 2.97×) the best is `mom12_1` top at
**1.31×** — nowhere near, and its median is −15.7%.

**At 10 years** (base 55.5%, 80% needs 1.44×):

| feature | side | n | touched 2x | lift | P(−50%) | median |
|---|---|---|---|---|---|---|
| amihud60 | bottom | 561 | 71.8% | 1.29 | 16.4% | +174.4% |
| **log_turnover** | **top** | 789 | **69.1%** | **1.24** | **13.7%** | **+174.7%** |
| lowvol | top | 787 | 62.6% | 1.13 | 23.3% | +49.0% |
| vol60 | bottom | 562 | 61.7% | 1.11 | 26.5% | +32.0% |

**Nothing reaches 1.44×.** The best is 1.29×, giving ~7.2 of 10.

**And the direction inverts everything the one-year studies found.** At one
year, liquid names were the worst cell in every table (A19: the liquid tercile
trailed the index by 9.5%). At ten years the *most liquid* names are the best on
every axis at once — highest touch rate, **half** the loss rate, and a median
that nearly trebles. The top two rows are the same axis said twice.

## 3. The one candidate, tested properly

`log_turnover` top decile at 10y is the maximum of ~22 cells, so it gets the
treatment rather than the headline. It is preferred over the marginally higher
`amihud60` because they are the same axis and this one has a prior mechanism
already established here — A19 found the cap-weighted index beat equal-weighted
baskets *because a handful of mega-caps carried it*. **This is that finding from
the other side, not an independent discovery.**

The null permutes whole **(ticker, year) blocks**, not rows: a name contributes
~12 near-identical monthly cohorts a year, and a row shuffle leaves the null far
too tight (A17 records that inflating a z to −8.7 before it became a headline).

| | |
|---|---|
| base touch rate | 55.5% |
| liquid decile | **69.1%** (n = 789), lift 1.24× |
| permutation null | 56.6%, sd 4.6%, p95 65.0% |
| **z** | **+2.70**, empirical p = 0.000 on 200 draws — **see §5, it resolves to 0.00140 at 5,000 draws** |

| half | base | liquid decile | lift |
|---|---|---|---|
| early | 70.6% | 76.1% | 1.08 |
| late | 40.3% | 61.3% | 1.52 |

**Positive in both halves.** Note the base rate itself is 70.6% early against
40.3% late — the regime does more work than the selection.

**It does not clear this project's Bonferroni bar.** 200 draws cannot resolve
below 0.005, which is a limitation with a price rather than a fact — §5 pays it
and the answer is unchanged: **p = 0.00140 at 5,000 draws against a bar of
0.00071.** Suggestive, not established.

## 4. And the comparison that has killed everything else

6,332 matched 10-year windows. Index put on a total-return basis with the
measured top-decile yield of 1.77% (H21), which compounds to 1.192× over ten
years.

| | median | mean | P(2x end) | P(2x touch) |
|---|---|---|---|---|
| **liquid decile** | **+188.8%** | **+464.1%** | **62.9%** | **72.8%** |
| all liquid names | +12.8% | +200.8% | 33.7% | 57.9% |
| IHSG price | +75.0% | +172.7% | 36.4% | 45.8% |
| IHSG **total return** | +108.5% | +225.0% | 61.0% | 75.8% |

Paired per window, liquid decile minus index TR: **median +51.8%, mean +239.0%,
winning 57.7% of windows.** Half-split: **+32.9% early, +73.9% late — positive
in both.**

**This is the first construction in the entire project that beats the index.**
Every previous one — flow, broker identity, investor class, price/TA, the
multiplier entry, all 58 exit rules, all 9 timing rules — lost. The reason is
not subtle: at a one-year horizon you pay ~1.3% round trip every year, and at a
ten-year horizon you pay it once, which is 0.13% a year. **A9 flagged exactly
this and it was never tested.**

---

## What this licenses, and what it does not

**8 of 10 is not reachable from any signal tested here at the decile level.**
The ceiling is 72.8% — about **7 of 10** — and it takes a ten-year hold of the
most liquid names to get there. (**§6 finds a narrower cell that does reach
82.8%**, on eight names and an effective n of 1.8. It is reported there with
the four reasons it is not the answer.)

**7 of 10 IS reachable, and the recipe is boring:** buy the ~10 most liquid IDX
names, equal weight, hold ten years. Historically ~7.3 of 10 touched 2x, the
median name returned +188.8%, and P(−50%) was 13.7% against a 28.6% base.

**The honest limits, and they are severe:**

- **Effective n is 56 for the whole sample and ~6 for the decile.** A ten-year
  window over a twenty-four-year panel is about two independent observations per
  name. Overlapping cohorts produce rows, not information, and no permutation
  null can manufacture independent samples the panel does not contain.
- **Only 30 distinct names were ever in the top decile** — TLKM, ASII, BMRI,
  BBRI, BBCA, BUMI, BBNI, PGAS, SMGR, ADRO, UNTR, ANTM and eighteen others. The
  cross-section is a list, not a population.
- It **is** reassuring that BUMI is in it — a top-decile name that fell ~99% —
  so this is not a survivor list.
- **The base rate does more work than the selection**: 70.6% early against 40.3%
  late. A decade beginning in 2026 is not a draw from the same urn as one
  beginning in 2005.
- **The holdout is spent** (H16), so none of this is out of sample.
- **Ten years is the whole mechanism.** Sell at year three and every result in
  §1 says you are back in the regime where costs eat the effect.

---

# H24 — pushing on 7-of-10, and the cell where 8-of-10 appears

*Same panel. Code: `scripts/decade_push.py`, `scripts/decade.py`. Tests:
`tests/test_decade.py` (15). Raw: `reports/decade_push.txt`,
`reports/decade_basket.txt`.*

## 5. Two costs H23 named and did not pay

**"200 draws cannot resolve below 0.005"** was written as a limitation and left
there, when the Bonferroni bar is 0.00071 and more draws is the entire price of
an answer. Paid:

| draws | z | p | verdict |
|---|---|---|---|
| 200 | +3.00 | 0.00995 | does not clear |
| 1,000 | +2.84 | 0.00200 | does not clear |
| **5,000** | **+2.87** | **0.00140** | **does not clear** |

The empirical p uses the +1 correction, because a permutation p can never
honestly be zero. **The answer is no, and it took two minutes to get.**

**The touch/end gap was measured and never acted on.** 69.1% of decile names
touch 2x and 59.6% end there. P1 predicted a 2x take-profit would raise the
realised count and lower wealth:

| | name doubled | I captured it | median | mean |
|---|---|---|---|---|
| hold, sell nothing | 69.1% | 59.6% | +174.7% | **+432.5%** |
| sell 100% at 2x | 69.1% | **69.1%** | +100.0% | +58.1% |

**P1 confirmed, and the cost is brutal:** capturing the last 9.5 points of hit
rate costs 374 points of mean return.

**P4 was pre-registered and FAILED.** I predicted scaling out would dominate
both corners — the hit rate set by the touch, the mean by the remainder. It
does not. The mean falls monotonically with every unit sold: +432.5%, +338.9%,
+245.3%, +151.7%, +58.1% at 0/25/50/75/100% sold. There is no free lunch in the
interior. (`sell 50% at 3x` does lift the median to +186.9% against holding's
+174.7%, at the cost of 160 points of mean — the same mean/median wedge that
decided H18 and H20, arriving a fourth time.)

**And the first draft of that table had a column that conflated two questions.**
"Doubles realised" was computed from the *peak*, which credits a
hold-and-never-sell with captures it did not make: a name that doubled in year
three and ended at 1.4x realised nothing. Split into **name doubled** (a
property of the picking, constant at 69.1% across every selling rule) and **I
captured it** (what the rule decides). **So "7 of 10 multi-baggers" is settled
at ENTRY; the exit only decides how much reaches the account.**

## 6. The cell where 8 of 10 appears — and why it is not the answer

Splitting the decile by how many of the three prior years the name was *already*
in it — knowable at entry, so it passes A5:

| prior years in decile | n | names | touched 2x | median | P(−50%) |
|---|---|---|---|---|---|
| new (0 of 3) | 262 | 35 | 56.9% | +33.6% | 18.3% |
| 1 of 3 | 162 | 14 | 67.9% | +188.7% | 14.2% |
| 2 of 3 | 144 | 9 | 71.5% | +225.5% | 14.6% |
| **3 of 3** | 221 | **8** | **82.8%** | +197.2% | **7.2%** |

**Monotone, and 82.8% is eight of ten.** The eight names: ASII, BBCA, BBNI,
BBRI, BMRI, BUMI, PGAS, TLKM. Clustered permutation over (ticker, year) blocks
gives z = +3.04, p = 0.00100 against a bar of 0.00069 — **closer than the
decile, still short.** And it is remarkably stable: **83.6% early against 82.4%
late, while the base rate collapsed from 70.6% to 40.3%.**

**Four reasons it is not the answer.**

1. **Effective n is 1.8.** Eight names, 221 overlapping windows — about *one*
   independent observation. This is the thinnest cell in the project.
2. **It was found by looking**, after the decile result was in hand. Every
   earlier post-hoc cell here that looked this good failed replication.
3. **It yields four names today** — BBCA, BBRI, BMRI, TLKM — **and three are
   banks.** That is one sector bet, not a basket.
4. **AMMN scores 3 of 3 and is excluded**: it has been listed 3.1 years, so its
   tenure score is its entire life, while the historical eight were all 20-year
   names. `MIN_LISTED_YEARS = 10` enforces that a name whose whole history is
   shorter than the hold cannot be the object the cell measured.

## What the answer to "7 of 10" actually is

**Available, at a price you can read:**

- **10 names at ~69% doubling** — the full liquidity decile, median +174.7%,
  P(−50%) 13.7%, beating the index by a paired median of +51.8% in both halves.
- **4 names at ~83% doubling** — the tenure core, on one effective observation
  and heavily concentrated in banks.

**Ten names at eight-of-ten is not available from this evidence.** The two
cells trade breadth against hit rate and there is no point on that curve that
delivers both.

Today's basket is in `reports/decade_basket.txt`, regenerated by
`python3 scripts/decade.py`. It prints the tenure and listing age beside every
name so the two tiers are visible rather than blended.
