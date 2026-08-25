# The exit layer, and the tie-break that was deciding the answer

*H17. Pre-holdout only. The 24-month holdout was spent at H16 and certifies
nothing now; the 2025 cohort appears here as illustration and is the basis of
no choice. Code: `src/idxbot/spine/exits.py`, `src/idxbot/spine/multiplier.py`,
`scripts/exit_study.py`, `scripts/tie_sensitivity.py`. Tests:
`tests/test_exits.py` (45); suite 1,844.*

---

## The question

H16 ran the multiplier-cell entry on 2025-08-25 and held for a fixed year with
no stop. The attribution was unambiguous: the ten names reached a mean **peak
of +102.2%** and realised **+15.1%**, giving back 43.4 points. Four peaked
above +100% and two ended there. INET was a triple at session 77 and finished
+27.9%.

The entry was not the failure. The absence of an exit was. And it is
structural rather than bad luck: the entry rule selects on P(2x), which selects
for path volatility, which produces large peaks *and* large give-backs by
construction. **A rule that maximises tail width cannot be held passively.**

Two things came out. The second is larger than the first.

---

## 1. THE ENTRY RULE WAS NOT A RULE

Building the exit study meant re-implementing the entry to generate historical
cohorts. Run on 2025-08-25 it returned a *different* basket from H16's — IMPC
where H16 had MERI — and therefore a different buy-and-hold return: **+26.3%
against +15.1%.** Neither implementation had a bug.

The rule scores each liquid name by the historical P(2x) of the
(price band, liquidity quintile, 60-day-vol quintile) cell it occupies. There
are at most 125 cells and roughly 800 live names, so the scores are massively
tied. On 2025-08-25:

| | |
|---|---|
| names sharing the **10th-place score** | **17** |
| distinct scores in the whole top 30 | **4** |

"Take the top ten" was therefore picking ten of seventeen equals by whatever
order the frame happened to be in — decided by `sort_values` stability, not by
the research.

### How much was that worth? Measured, not argued.

**The 2025 cohort, 500 random tie-breaks, one-year net after costs:**

| p5 | p25 | median | p75 | p95 | sd | min | max |
|---|---|---|---|---|---|---|---|
| −16.1% | −2.5% | **+6.9%** | +15.7% | +26.9% | **13.0%** | −29.2% | +36.6% |

H16's basket landed at **+15.8%** — around the 75th percentile of baskets it
had no reason to prefer. The exit-study basket landed at **+26.3%**, near the
95th. **The honest centre of that cohort is +6.9%, not +15.1%.**

**211 pre-holdout monthly cohorts, 40 tie-breaks each:**

| | |
|---|---|
| cohorts where the top-10 cut falls inside a tie | **64%** |
| names sharing the 10th-place score | median 4, p90 12, max 23 |
| distinct scores in the top 30 | median **12 of 30** |
| within-cohort sd across tie-breaks | **4.31%** |
| span of the middle 90% of tie-breaks | **10.92%** |

For scale, the block-bootstrap half-width on the 211-cohort average is ±9.36%.
The tie-break sd is **per cohort** and is additional to that — it appears in no
interval anyone has quoted from this rule, H16's included.

### The fix, and why it is not a better tie-break

`multiplier.select(tie="all")` holds **every name in the group straddling the
cut**, equal-weighted. That is not a smarter tie-break; it is the absence of
one — the only version two implementations are obliged to agree on. Basket size
becomes 12 names on average instead of a fixed 10.

It is also, on this sample, **slightly worse**: −4.42% against tie='first''s
−2.54%, difference −1.47% [−2.48%, −0.56%]. That is the correct trade anyway.
A reproducible rule measuring −4.4% is worth more than an irreproducible one
measuring −2.5%, because the −2.5% was never a property of the rule.

---

## 2. THE EXIT LAYER — it works, and the size is honest

Thirty-two candidate rules (fixed holds, trailing stops with and without an
arming threshold, hard stops, time stops, combinations), scored on **cohorts,
never names**, costs charged once per name at A5's 0.56% plus a point-in-time
fraksi-harga half-spread, entry one bar after the signal.

Selection is **purged** walk-forward: the rule at cohort *t* is chosen only on
cohorts whose forward year had already closed by *t*.

**176 scored cohorts, 2008-09 → 2023-08:**

| | cohort median | 95% CI |
|---|---|---|
| walk-forward selected rule | **+0.94%** | [−8.11%, +10.49%] |
| buy and hold 252 | −3.19% | [−13.94%, +6.38%] |
| **difference** | **+4.13%** | **[+1.87%, +6.33%]** |

The difference CI excludes zero, and the sign test is decisive: the rule
differed from buy-and-hold in 94 cohorts and **won 74 of them, 79%,
p ≈ 1.6 × 10⁻⁸**. It is not a thin-tail artefact — the win rate is the effect.

The magnitudes are symmetric, which is why the median difference is exactly
0.00%: **+14.24% mean when it wins, −16.34% mean when it loses.** Most cohorts
(82 of 176) see no difference at all because the trail never fires.

The walk-forward converged rather than adapted: **153 of 176 cohorts chose
`trail 15% armed +50%`**, 17 chose `trail 40% armed +50%`, 6 chose 30%. Mean
holding period 210 sessions against 250.

### THE CAVEAT THAT MATTERS: it does not fix the downside

`P(−50%)` is **15.0% under the rule against 16.3% buy-and-hold.** Essentially
unchanged, and structurally so: the trail arms at +50%, so a name that goes
straight down never arms and is held to the horizon exactly as before. Every
armed-trail variant in the catalogue reads the identical 15.1%.

That is not fixable for free, and the frontier says so:

| rule | P(−50%) | P(2x) | cohort median | days |
|---|---|---|---|---|
| stop 25% + trail 25% + time 21d | 0.0% | 5.8% | −5.6% | 66 |
| stop 15% | 0.0% | 8.4% | −10.6% | 116 |
| stop 25% | 0.1% | 10.1% | −11.7% | 156 |
| trail 40% | 0.6% | 10.7% | −7.1% | 171 |
| **trail 40% armed +50%** | 15.1% | 11.3% | −2.0% | 235 |
| **hold 252** | 16.3% | 11.6% | −4.3% | 250 |

A hard stop takes P(−50%) to near zero and costs 6–8 points of median return,
because it exits drawdowns that would have recovered. **For a rule selected on
P(2x), cutting the left tail cuts the premise.** There is no rule here that is
better on both axes.

### Illustration — the 2025 cohort, on already-spent holdout data

| rule | mean | median | 2x | −50% | held |
|---|---|---|---|---|---|
| buy and hold 252 | +6.8% | −7.2% | 2 | 4 | 241d |
| trail 15% armed +50% | **+16.0%** | **+24.5%** | 2 | 4 | 147d |

Same direction, same shape: the trail adds return and leaves the four disasters
untouched. This certifies nothing — H16 spent the holdout.

---

## What I now believe, and with what confidence

**Medium confidence, and it is a real improvement.** A trailing stop armed at
+50% adds about **+4% per cohort** to the multiplier-cell entry, out of sample,
purged, on 176 cohorts over 15 years, winning 79% of the cohorts where it does
anything. It halves nothing and cuts no tails; it captures give-back on the
names that ran.

**Low confidence in the entry rule's own numbers, and that is new.** Until the
tie-break is held fixed, the rule's historical record carries ±4.3% of pure
implementation noise per cohort. Every prior figure from it — including H16's
2/10 doublers — sits inside a distribution of equally-valid alternatives that
was never reported.

**What would falsify the exit result:** a walk-forward on cohorts formed weekly
rather than monthly, or on a different liquidity floor, in which the trail's
win rate among differing cohorts falls to chance. The rule choice converged so
hard on one parameter that a genuine effect should survive both.

---

## Three methodological defects found while building this, all mine

**The walk-forward leaked, and the leak was invisible.** The first version
trained on every earlier cohort. Cohorts are monthly and the horizon is a year,
so last month's cohort has eleven months still to run when this month's
decision is made — its outcome is not knowable. `walk_forward_select(purge=True)`
now restricts training to cohorts settled on or before the decision date,
discarding the most recent ~12 at every step. It cost 12 scored cohorts and
moved the difference from +3.91% to +4.13%.

**The cohort bootstrap treated overlapping cohorts as independent.** 176
monthly cohorts each holding a year span about fifteen independent
year-windows; an iid resample says 176 and returns an interval roughly
`sqrt(176/15) ≈ 3.4×` too narrow. This is the *same* error H16 named about its
own twelve cohorts ("effective n is ~1, not 12") — reintroduced one layer up,
in the function written to avoid it. Now a moving-block resample with the block
inferred from cohort spacing. It roughly doubled every interval reported here.

**Two smaller ones fell out of fixing that.** `DatetimeIndex.asi8` is
**microseconds** on a `datetime64[us]` index and nanoseconds on a `[ns]` one, so
a hardcoded divisor returned a block length of 11,783 for monthly cohorts. And
a block that is a large fraction of the sample **degenerates** — measured widths
0.049 at b=1, 0.105 at b=13, 0.047 again at b=63 — so it is capped at a fifth
of the sample.

The generalisation worth keeping: **the unit of resampling and the unit of
independence are different questions, and getting the first right does not
settle the second.**
