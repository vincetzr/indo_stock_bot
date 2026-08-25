# The recovery curve — where the odds actually break, and what the indicators know

*H19. Pre-holdout only. 243,977 armed liquid bars, 560 names, 279 months,
2001-04 → 2024-08. Code: `scripts/recovery.py`. Tests: `tests/test_recovery.py`
(15). Raw: `reports/recovery_curve.csv`, `reports/recovery_conditional.csv`.*

---

## Why this exists: H17 and H18 both dodged the question

Both picked a trailing-stop distance by **grid search** — score 15/20/25/30/40%,
keep whichever won the cohort median. That answers "which of these five did best
on this sample". It does not answer what a holder actually asks:

> given that I am already X% below the peak, what are the odds this comes back,
> and what do I make if I sit through it?

A 15% trail is only right if 15% is roughly where the edge dies. **Nobody had
measured that.** So this measures the conditional directly, on every liquid
pre-holdout bar whose trailing 252-session window contains a run of at least
+50% into its peak — the panel-wide equivalent of an armed position, which is
the only state a trailing stop can fire in.

---

## The curve. Forward horizon 60 sessions.

| give-back from peak | n | **P(new high)** | 95% CI | mean fwd | 95% CI | **median fwd** | in ATRs |
|---|---|---|---|---|---|---|---|
| −5% to 0% | 50,455 | **81.3%** | [79.0, 83.2] | +7.1% | [+4.5, +9.8] | +2.5% | 0.5 |
| −10% to −5% | 33,483 | **56.1%** | [50.6, 60.2] | +5.3% | [+2.2, +7.6] | +0.9% | 2.1 |
| −15% to −10% | 28,830 | **38.8%** | [33.8, 42.7] | +4.5% | [+1.0, +7.1] | **−0.5%** | 3.2 |
| −20% to −15% | 25,105 | **27.1%** | [23.1, 31.1] | +3.6% | [+0.1, +6.3] | −1.0% | 4.3 |
| −25% to −20% | 20,455 | **17.3%** | [14.5, 20.1] | +1.6% | [−1.4, +4.4] | −2.6% | 5.1 |
| **−30% to −25%** | 17,540 | **11.3%** | [8.8, 13.2] | −0.3% | [−3.7, +2.2] | −4.5% | 5.9 |
| −35% to −30% | 15,590 | 8.3% | [5.8, 10.3] | +0.2% | [−3.2, +3.3] | −4.0% | 6.6 |
| −40% to −35% | 12,823 | 5.9% | [3.9, 8.1] | +0.5% | [−2.9, +3.8] | −3.1% | 7.4 |
| −50% to −40% | 17,700 | 2.9% | [1.8, 4.2] | +0.1% | [−4.5, +4.5] | −3.7% | 8.0 |
| −60% to −50% | 10,747 | 2.2% | [0.9, 3.5] | +0.8% | [−4.5, +6.9] | −4.1% | 8.6 |
| −100% to −60% | 11,249 | 0.8% | [0.1, 1.4] | −2.2% | [−8.2, +6.3] | −8.3% | 9.1 |

### Direct answer: at −30%, the chance of rallying back is 11.3%. It is not high.

P(new high) crosses below half between **−5% and −10%**. By −15% it is 38.8%,
by −20% 27.1%, by −25% 17.3%, by −30% **11.3%**, by −40% 5.9%. The decline is
monotone (H19a holds) and the intervals are narrow enough to separate every
adjacent bucket down to −25%.

### H19b was pre-registered and FAILED — in an instructive way

I predicted a depth where the **mean** forward return turns significantly
negative, and that it would be deeper than 15%. **There is no such depth.** The
mean stays between −2.2% and +7.1% and its interval covers zero from −25%
downward — it never turns significantly negative, even 60% off the peak.

What *does* turn negative is the **median**, and it does so at **−10% to −15%**
(−0.5%), deepening steadily thereafter. So:

- **On the typical outcome, the edge dies around −10 to −15%.**
- **On the average outcome, it never dies** — a thin tail of enormous recoveries
  keeps expected value roughly flat all the way down. Deep-drawdown names are a
  lottery ticket with a fair-ish price, not a negative-expectation trap.

This is the H18 mean/median wedge again, from a completely different direction.

### And it independently validates the 15%

The grid search picked `trail 15% armed +50%` by optimising cohort median. The
conditional curve says the median forward return crosses zero at −10 to −15%.
**Two unrelated routes to the same number**, which is the strongest thing I can
say for that parameter — it was never verified before, only selected.

In the name's own volatility, −10% is **2.1 ATR** and −15% is **3.2 ATR**. That
is exactly where `chandelier 2x ATR` sits, which is why H18a came out supported.

**One caveat that cuts against holding deeper**: the mean *further* drawdown
(MAE) over the next 60 sessions widens monotonically — −12.5% at a −10%
give-back, −16.1% at −30%, −22.4% below −60%. Sitting deeper does not just
lower the odds, it raises the additional pain while you wait.

---

## H19c — can the indicators tell? Yes for recovery, no for return.

53 (indicator × depth) cells, each tested against a null that permutes the
label across **whole (ticker, month) blocks** inside the same month and depth.

| outcome | cells clearing Bonferroni (0.0009) | sign split | median effect |
|---|---|---|---|
| **P(new high)** | **14 of 53** | **13 positive / 1 negative** | **12.6 points** |
| forward return | 4 of 53 | 2 positive / 2 negative | 4.2% |

**The indicators move the recovery probability a lot and consistently. They do
not move the forward return in any stable direction.** Four return-cells clear
the bar, but they split evenly in sign, which is what an incoherent effect looks
like — not a usable one.

### Which indicators, specifically

| test | d P(new high), sign consistency | mean shift |
|---|---|---|
| **above EMA50** | **9 positive / 0 negative** | **+7.9 pts** |
| **above EMA20** | 9 / 1 | +3.7 pts |
| **give-back < 2 ATR** | 2 / 0 | +15.4 pts |
| turnover z > 0 | 7 / 4 | +3.8 pts |
| stoch %K > 50 | 8 / 2 | +1.4 pts |
| **stoch %K > %D** | **5 / 6** | **+0.2 pts** |

So of the four things you named: **EMA works, ATR-scaled depth works, volume is
weak, and the stochastic cross is noise** — 5 up against 6 down and a mean shift
of two tenths of a point.

### And the help is concentrated where it is useful

Average d P(new high) across the trend/volatility tests, by depth:

| depth | −10 to −5% | −15 to −10% | −20 to −15% | −25 to −20% | −30 to −25% | deeper |
|---|---|---|---|---|---|---|
| **shift** | **+17.7 pts** | **+11.9 pts** | +3.3 | +5.2 | +4.9 | +2 to +4 |

The indicators are most informative at exactly the depth where you are deciding
whether a small dip is noise or the end, and decay to near-nothing past −20%.
The single largest cell: at a −10 to −5% give-back, **above the EMA50 is worth
+22.4 points of recovery probability**.

---

## What this changes

**Your instinct that 15% might be too tight is right on the mean and wrong on
the median.** If you are playing for the multiplier and can stomach it, holding
past 15% costs you nothing in expected value — the deep buckets have flat mean
returns. If you want the typical trade to work, cut at 10–15%, because that is
where the median outcome turns against you.

**The best-motivated rule the curve endorses is `ema50 break armed +50%`, not
the trail.** Cross-referencing H18's table for the same cohorts:

| rule | cohort median | mean | **P(2x)** | P(−50%) | days |
|---|---|---|---|---|---|
| **ema50 break armed +50%** | +0.8% | **+15.0%** | **7.9%** | 15.1% | 211 |
| trail 15% armed +50% | +0.9% | +11.4% | 7.3% | 15.1% | 208 |
| chandelier 2x ATR armed +50% | +2.9% | +6.7% | 4.0% | 15.1% | 197 |
| stoch rollover armed +50% | +2.6% | +4.8% | 2.3% | 15.1% | 196 |
| hold 252 | −4.3% | +18.8% | 11.6% | 16.3% | 250 |

`ema50 break armed +50%` **matches the trail's median, beats it on mean by 3.6
points, and keeps the highest P(2x) in the whole armed family.** That is what
the recovery curve predicts: the EMA50 is the one indicator that consistently
separates recoverers, so a rule that holds while above it and cuts when it
breaks should keep more of the right tail than a fixed distance. It does.

**Caveat, and it is the same one as always.** That comparison is read off H18's
full-sample table, so it is a post-hoc selection from a table I have already
seen. The mechanism was predicted in advance and confirmed independently here,
which is worth something — but the size is not out-of-sample, and the holdout is
spent. Treat it as the best-motivated choice, not a validated one.

---

## Two method notes

**The null had to permute names, not rows, and the first version got it wrong.**
A name contributes ~20 near-identical bars a month — same indicator state,
overlapping forward windows — so a row shuffle destroys the label while leaving
the null far too tight. Every z came back inflated; one read **−8.7** and became
the headline. Permuting whole (ticker, month) blocks is the fix, and
`test_the_block_null_is_wider_than_a_row_null_on_clustered_data` makes the
difference visible on synthetic data rather than asserting it.

**A cell needs both sides populated.** The first table's largest effect
(**−46.1%**) came from a flag true for 0.2% of rows — a handful of observations
against everything else. The guard only rejected shares of exactly 0 or 1. Now
both sides need 300 observations and a 5% share.
