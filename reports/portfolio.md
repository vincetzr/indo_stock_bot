# H20 — what you actually earn, and the benchmark that was not in the study

*Pre-holdout only. 212 monthly cohorts, 27,451 priced name-cohorts, 658 names,
2004-12 → 2023-08. Code: `scripts/portfolio_sim.py`, `scripts/portfolio_critique.py`.
Tests: `tests/test_portfolio_sim.py` (15), `tests/test_coverage_map.py` (3).
Raw: `reports/portfolio.txt`, `reports/portfolio_critique.txt`.*

> **READ §5 FIRST.** Everything in §1–§4 compares the picks against the picks
> or against a random draw from the same pool. The IHSG is in none of it, and
> when it is added, this memo's own conclusion — "the remaining defensible
> position is buy-and-hold on the picks" — does not survive. That sentence is
> **retracted**; §5 replaces it.

---

## Two critiques of my own H17–H19, neither of which I had tested

**The cohort MEDIAN is not what an investor receives.** Hold twelve names
equal-weighted and you are paid their **mean**. H17 and H18 both selected their
exit rule by maximising the average cohort median — a statistic nobody is paid —
and H18's own objective table already showed that on the mean no rule beats
buy-and-hold. The headline improvements (+4.13%, +6.35%) were improvements in
the wrong number.

**The entry may contribute nothing.** H16 measured p = 0.211 for doublers
against random draws from the same liquid universe. The control had never been
run against the exit layer.

Both are now tested on **portfolio accounting**: twelve month-offset slots, each
buying a basket, holding until the rule exits, then redeploying. That charges
every extra round trip and credits every early exit with the capital it frees.

---

## 1. The mean and the median disagree, and the median was winning

| rule | **MEAN** | median | mean log | P(2x) | P(−50%) | days |
|---|---|---|---|---|---|---|
| hold 252 | **+18.8%** | −4.3% | +0.099 | 11.3% | 17.0% | 250 |
| trail 15% armed +50% | +11.4% | +0.9% | +0.070 | 7.0% | 15.7% | 208 |
| trail 30% armed +50% | +17.2% | −0.6% | +0.102 | 10.4% | 15.7% | 226 |
| chandelier 2x ATR armed +50% | +6.7% | **+2.9%** | +0.036 | 3.6% | 15.7% | 197 |
| stoch rollover armed +50% | +4.8% | +2.6% | +0.021 | 2.2% | 15.8% | 196 |
| ema50 break armed +50% | +15.0% | +0.8% | +0.090 | 7.8% | 15.7% | 211 |
| stop 25% | +17.4% | −11.7% | **+0.103** | 9.8% | **0.2%** | 156 |

The two rules H17 and H18 selected — `trail 15%` and `stoch rollover` — are the
**two worst rules in the table on the mean**, at +11.4% and +4.8% against
buy-and-hold's +18.8%. They won the median by cutting the right tail, and the
right tail is where the return lives.

## 2. Portfolio: CAGR, drawdown, terminal wealth

| rule | CAGR | 10–90% across slots | max DD | × terminal | trades |
|---|---|---|---|---|---|
| hold 252 | +10.5% | [+6.7, +13.8] | −49.8% | 6.4 | 18 |
| **trail 30% armed +50%** | **+13.7%** | [+7.1, +20.8] | **−44.3%** | **12.4** | 19 |
| ema50 break armed +50% | +11.0% | [+1.7, +13.6] | −47.4% | 6.6 | 20 |
| stop 25% | +10.8% | [+9.9, +13.5] | **−68.7%** | 6.9 | 29 |
| chandelier 2x ATR armed +50% | +4.6% | | −43.8% | 2.3 | 21 |
| **trail 15% armed +50%** | **+2.4%** | [+1.7, +3.8] | −48.9% | **1.6** | 20 |
| stoch rollover armed +50% | +1.8% | | −60.6% | 1.4 | 21 |

**H17's rule turns 6.4× into 1.6×.** H18's pick turns it into 1.4×, with a
*deeper* drawdown than holding. Note also that `stop 25%` — which caps every
position at −25% and has a per-name P(−50%) of 0.2% — has the **worst portfolio
drawdown of all**, because it realises losses 29 times instead of 18 and
redeploys straight back into the same bad regime. **Per-position stop-losses can
increase portfolio drawdown.**

## 3. THE HALF-SPLIT, WHICH ENDS THE ARGUMENT

Paired per slot against buy-and-hold, run inside each half independently:

| rule | early ΔCAGR | wins | late ΔCAGR | wins | both? |
|---|---|---|---|---|---|
| trail 15% armed +50% | −9.41% | 1/12 | +0.53% | 7/12 | no |
| **trail 30% armed +50%** | **+3.85%** | 9/12 | **+1.69%** | 6/12 | **YES** |
| chandelier 2x ATR armed +50% | −4.52% | 1/12 | −4.97% | 2/12 | no |
| stoch rollover armed +50% | −8.60% | 0/12 | −4.58% | 2/12 | no |
| ema50 break armed +50% | −3.16% | 5/12 | +6.73% | 10/12 | no |
| stop 25% | +2.83% | 10/12 | −2.26% | 2/12 | no |

**Exactly one of six survives — and one is BELOW what chance predicts.** If each
half were a coin flip, 6 rules would give an expected **1.5** positive in both.
Observing 1 is not evidence for `trail 30%`; it is what a table of noise looks
like. Two rules that looked decisive on the full sample — `stop 25%` (10/12
early) and `ema50 break` (10/12 late) — win in **opposite halves**, which is the
signature of regime-dependent noise.

**So: no exit rule is established. None.** Including the one H19's recovery
curve endorsed on mechanism, which fails the early half.

## 4. The entry — one pre-specified comparison, and it dies in the recent half

The entry is *not* one of six; it is a single comparison, so a half-split can
actually certify it. Paired per slot, buy-and-hold on the picks against
buy-and-hold on random draws from the same eligible pool, 12 slots × 20 draws:

| half | mean ΔCAGR | sd | wins | win rate |
|---|---|---|---|---|
| early 2004-12 → 2017-08 | **+5.86%** | 5.54% | 204/240 | **85%** |
| **late 2017-08 → 2023-08** | **+0.36%** | 9.43% | 122/240 | **51%** |
| full | +3.99% | 4.83% | 197/240 | 82% |

**In the last six years the entry is a coin flip — 51%.** The full-sample
+3.99% is an average of a regime where it worked and a regime where it did not.

---

## 5. THE BENCHMARK THAT WAS NOT IN THE STUDY

Everything above is picks-versus-picks or picks-versus-pool. The thing a retail
account can actually buy instead — the index — appears nowhere. `_JKSE.csv.gz`
has been sitting in `data/cache/ohlcv/` the entire time.

**First, is the benchmark series any good?** A13 found decimal-shift errors in
`IDR=X` from this same unauthenticated endpoint — 888.11 against a true ~8,881,
reversing the next day — so taking `^JKSE` on trust would repeat a mistake this
repo has already made once, and an endpoint-to-endpoint CAGR reads exactly two
bars, which makes a defect at either end maximally damaging rather than averaged
away. Checked before anything was computed from it: **worst landmark error
0.003%** across six published year-end closes, **zero** moves beyond ±20%
(a decimal shift announces itself as a huge move immediately reversed), kurtosis
9.8, and every calendar gap ≤12 days — all Idul Fitri closures. The landmark
half of that is the weaker half, since those values come from knowledge rather
than a fetch of IDX's own publication; it confirms the series is the IHSG but
cannot independently certify a level. The internal checks need no external
reference and are what rule out the `IDR=X` pathology. `validate_index()` runs
it on every invocation and four tests pin it.

**Then make the comparison like for like, and both corrections favour the
picks.** The name returns run on `adj_close`, Yahoo's back-adjusted close, so
they are **total returns**; `^JKSE` is a **price index**. The adjustment
identifies itself: `log(adj_close/close)` steps only at corporate actions, and
back-adjustment makes every dividend step positive going forward. Across 1.75m
steps in this universe there are **3,707 small positive steps and zero small
negative ones**, which is what a dividend series looks like and not what noise
looks like. Steps above 0.10 in log are splits and bonuses, excluded by size.

Yield rises monotonically with liquidity, which is the check that the
measurement is real and not an artefact of the filter:

| liquidity decile | 1 | 3 | 5 | 7 | 9 | 10 |
|---|---|---|---|---|---|---|
| dividend yield | 0.65% | 0.98% | 1.12% | 1.30% | 1.53% | **2.01%** |
| % of ticker-years paying | 21% | 30% | 33% | 38% | 51% | 70% |

The IHSG is cap-weighted, so it yields what the **top deciles** yield — 1.77% —
against the picks' 1.27%. Correcting the index *up* to a total-return basis is
therefore the larger of the two corrections, and it cuts the same way.

### The result

| window | picks TR | index TR | **gap** | picks maxDD | IHSG maxDD |
|---|---|---|---|---|---|
| early 2004-12 → 2017-08 | +13.5% | +16.8% | **−3.3%** | −41.0% | −60.7% |
| late 2017-08 → 2023-08 | +3.2% | +4.7% | **−1.5%** | −46.3% | −41.1% |
| **full** | **+10.5%** | **+12.7%** | **−2.2%** | −49.8% | −60.7% |

**Buy-and-hold on the picks loses to the index in both halves.** It also fails
to buy anything with that shortfall: the drawdown is shallower than the index's
over the full sample but *deeper* in the recent half, so there is no consistent
risk offset either. And the picks pay ~18 round trips at 0.56% to get there,
against one for the index.

**Three things make this worse rather than better, and none is corrected for.**
The picks' figures are already net of costs while the index figure is gross.
1.45% of pick-holds end early because the name stopped trading, and those are
marked at the last traded price rather than at zero — marking them to zero would
cost roughly a further 1.3 points a year. And an index fund is not free either,
which is the one correction running the other way: Indonesian index products
cost perhaps 0.5–1.0% a year in fees and tracking error, so the honest gap is
about **−1.2% to −1.7%** rather than −2.2%.

### And the table above has two window mismatches of its own

Both were in the direction that flatters the picks, and fixing them makes the
result stronger rather than weaker.

The twelve slots **begin in twelve different months and end in twelve
different months**, so benchmarking all of them against one global index window
compares each slot to a period it did not occupy. And a slot's last position is
**still open for its holding period after the final entry**, so its span runs a
year past the last cohort date while the index was measured only to that date.
`slots()` now returns `start` and `end`, and each slot is paired against the
index over its own span — the same pairing §3 and §4 already use:

| window | picks | index TR | **mean Δ** | sd | slots won | 95% CI |
|---|---|---|---|---|---|---|
| early | +14.2% | +16.4% | **−2.18%** | 4.86% | 3/12 | [−4.93%, +0.57%] |
| late | +1.4% | +4.0% | **−2.65%** | 6.56% | 6/12 | [−6.36%, +1.06%] |
| **full** | **+9.8%** | **+12.3%** | **−2.53%** | 3.73% | **3/12** | **[−4.64%, −0.42%]** |

The picks lose to the index in **nine of twelve slots**, by 2.5% a year, and on
the full sample the interval excludes zero. Twelve overlapping slots over one
history are not twelve independent trials, so that interval is too narrow — it
is quoted precisely because it is the reading most favourable to a significance
claim, and the picks still do not win it.

**So the claim does not rest on significance at all.** Even taking the
comparison as a tie — slot dispersion [+6.7%, +13.8%] does contain +12.7% — the
picks cost eighteen round trips, single-name concentration and a −50% drawdown
to arrive at, at best, the same place. **A tie against the cheap alternative is
a loss for the expensive one.**

### Is the shortfall the cost, or the selection?

Only one of those has a fix. If the 2.5% is the toll, a lower-turnover version
of the same rule recovers it; if it is the picking, no amount of patience does.

| | picks | vs index TR |
|---|---|---|
| net of cost | +10.5% | −2.53% |
| **gross, all round trips refunded** | **+12.3%** | **−0.82%** |

The eighteen round trips cost **1.76% a year** — most of the gap — but
**refunding every one of them still leaves the picks behind.** This is a
selection shortfall with a large toll on top, not a toll alone.

### Is it just that the picks are small caps?

The obvious rescue: the picks are equal-weighted mid-caps and the IHSG is
cap-weighted large-caps, so perhaps the rule is fine and the segment is the
handicap. Split the pool by within-cohort liquidity tercile. **The picked
baskets fragment and cannot answer** — the liquid cell scores 54 of 212
cohorts at a median of 4 names, and compounding a four-name basket across
non-contiguous cohorts is not a portfolio. An earlier draft of this section
printed −13.9% for that cell; it is a degenerate-cell artefact of exactly the
kind recorded twice already in this repo, the smallest cell producing the
largest effect, and it is **not reported**.

A *random* draw of twelve is a full basket in every tercile, though, so the
segment handicap itself is readable — and it answers the question **backwards**:

| tercile | random pool CAGR | vs index TR |
|---|---|---|
| thin | +7.9% | −3.54% |
| middle | +4.8% | −6.69% |
| **liquid** | **+3.7%** | **−9.48%** |

The size story predicted the liquid tercile would close the gap. It is the
**worst** of the three. Every tercile trails the index and moving upmarket
makes it worse. So the shortfall is not a small-cap handicap: it is that an
**equal-weighted** basket of IDX names lost to the **cap-weighted** index over
this sample, because a handful of mega-caps carried it, and even the pool's
liquid tercile sits far below those in capitalisation — inheriting none of the
mega-cap return while keeping the equal-weighting penalty. For the record the
rule leans thin (51% of picks against 15% liquid), which given this table is
the better end to lean toward and is not the source of the shortfall.

Answering the picked version properly means re-ranking *inside* the liquid
tercile and taking a full basket there. That needs the cell scores `build()`
does not persist: about twenty minutes of rebuild, not a limit of the data.

## 6. Two more critiques of §3–§4, and they split

**"The entry is a coin flip after 2017" was a power statement written as an
effect statement** — A8's exact distinction. It is now computed. Taking the
slot as the unit (averaging the twenty redraws within each slot first, which
H20 did not do):

| window | slots | mean ΔCAGR | 95% CI | smallest detectable |
|---|---|---|---|---|
| early | 12 | +6.08% | [+3.94%, +8.22%] | 2.14% |
| late | 12 | −0.01% | [−4.15%, +4.14%] | 4.14% |

The late half's interval **excludes** +6.08%, so the break is real and H20's
claim survives — but the interval is wide, and the late half **cannot rule out
an edge as large as +4.1% a year**. "Coin flip" is too strong; "the early
effect is excluded, anything up to +4% is not" is what was measured.

**And 2017 is a decay, not a cliff.** A single cut chosen for convenience finds
a break somewhere, so the same paired statistic was run on rolling six-year
windows stepped a year: **14 of 15 are positive**, from +12.1% (2007–2013) down
to −0.6% (2018–2024), trend −0.45% per year of start date. The edge over the
random pool is persistent and decaying, not switched off in 2017. The halves
cut where the decay bites. (Overlapping windows share five years of six — this
is a shape, not fifteen measurements.)

Neither changes §5. An edge over an equal-weighted draw from the same pool that
still trails the cap-weighted index is a statement about the pool, not a
tradeable edge.

---

## What is actually solid

**Negative, and stable in both halves:** `chandelier 2x ATR armed +50%`
(−4.5%, −5.0%) and `stoch rollover armed +50%` (−8.6%, −4.6%) are reliably
*worse* than holding. The second is H18's walk-forward selection. These are the
only findings here that replicate across eras, and they are both negative.

**Not established:** every positive result. No exit rule survives the split at
better than the chance rate. The entry beats a random draw from its own pool in
14 of 15 rolling windows and decays; it does not beat the index in any window.

**Retracted:** H17's `+4.13%` and H18's `+6.35%` were improvements in the cohort
median. On portfolio CAGR the same rules deliver **+2.4%** and **+1.8%** against
buy-and-hold's **+10.5%**. I reported both as improvements and they are not.

**Retracted:** "the remaining defensible position is buy-and-hold on the picks."
On a total-return basis the picks trail the index by **2.2% a year**, in both
halves. The defensible position is the index.

---

## The bug that changed which rule won

The slot scheduler locked capital by converting held sessions to calendar days
and searching the date index. A 30-day lock opened on 1 February ends on 3 March
and **misses the 1 March cohort entirely**, idling for a month. The penalty
scales with turnover, so it fell hardest on short-holding rules — exactly the
comparison the study exists to make. Before the fix `stop 25%` looked like the
best rule at 12/12 slots and t = +7.05; after it, `stop 25%` fails the late half
and `trail 30%` is the survivor.

`test_a_slot_never_re_enters_before_its_position_is_free` and
`test_an_early_exit_redeploys_and_therefore_trades_more` now pin the behaviour.
Locking is done in cohort-index space (~21 sessions to the month, rounded up),
which cannot skip.

## The fifteen tests this study deleted without failing anything

The H20 tests were written to `tests/test_portfolio.py`. **That file already
existed** — fifteen tests covering `src/idxbot/portfolio.py`, a module the CLI
exposes — and the write replaced them. Both files held exactly fifteen tests,
so the suite total did not move, and nothing in any run said anything had gone.
`git status` reported ` M` rather than `??` and that single character was the
only warning issued.

The general failure is that a module can go from covered to uncovered without
any test failing, because the evidence of coverage is the tests themselves. The
script is now `scripts/portfolio_sim.py` and its tests `tests/test_portfolio_sim.py`;
the originals are restored; and `tests/test_coverage_map.py` asserts every
module under `src/idxbot/` is named by some test, with the two genuinely
uncovered modules listed by name rather than hidden. The suite goes 1,892 →
**1,910**, which is the arithmetic that should have held all along.

**The general lesson, and it is the same one three times now:** a within-sample
consistency statistic over correlated units — 12 overlapping slots, 20 redraws,
188 overlapping cohorts — reads as overwhelming (12/12, t = 7.05) and carries
almost no information about whether the effect replicates. The half-split is
cheap, and it is the only thing in this study that changed my mind.
