# Findings from 25 years of real IDX data

Two results, both on genuine exchange data, both reproducible from this repo:

1. **The accumulation score is inverted.** Buying quiet bases lost to buying
   everything else — strongly before 2017, decaying to noise after.
2. **A momentum score built from that diagnosis survives out-of-sample.** On a
   holdout period never used for selection, the top quintile beat the bottom by
   **+5.16% over 60 days (t = 5.90)**, and a long-only top-10 portfolio beat the
   equal-weight universe by **+4.67% per 60-day period (t = 2.32)** with a
   materially smaller drawdown — but with 4 losing years in 10, and roughly
   8 points of the headline CAGR being survivorship bias (§4).

Result 2 is not a licence to switch on the screener and stop thinking. Read §5.

**Part II** then re-runs everything on the full exchange — 724 tickers instead
of 66 — and adds a walk-forward that chooses the parameters out-of-sample too.
Three results there matter more than anything in Part I:

3. **The median IDX stock loses money.** Under half of all 60-day holding
   windows are profitable, and the top 1% of observations supply *more than
   100%* of total return at short horizons (§8).
4. **One component carries the signal and it is not the composite.** Distance
   from the 52-week high beats the blend it sits inside, and unlike the blend it
   has not decayed — it is at its strongest in 2021-26 (§9).
5. **Training to optimum made things worse.** Re-optimising the weights every
   two years roughly halved the return versus simply fixing a sensible trend
   profile. There is no stable optimum on this data to train toward (§10).

---

## What was tested

| | |
|---|---|
| Observations | **55,699** |
| Tickers | 66 |
| Period | 2001-08-03 → 2026-08-04 (**25 years**) |
| Data | Real Yahoo daily OHLCV + real IHSG. **Nothing simulated.** |
| Mode | `price-only` (`--providers none`) — no broker data anywhere |
| Sampling | Shared exchange calendar, every 5th trading day, 300-bar warm-up, no look-ahead (unit-tested). Median **52 names per cross-section**. |
| Split | Chronological. Train 2001-08 → 2016-12. **Holdout 2016-12 → 2026-08.** |

```bash
idxbot backtest --universe all --providers none --profile momentum \
    --out reports/obs_momentum.csv
idxbot evaluate --observations reports/obs_momentum.csv --split --components
```

### A note on method

The first version of this analysis pooled every (ticker, date) observation into
one bucket comparison. That mixes two different questions — "is now a good time
to be long IDX?" and "which of these names should I buy today?" — and the first
is dominated by market beta.

A screener answers the second. So the metrics below are **cross-sectional**:
ranked *within each date*, so market direction cancels out. The t-statistic is
computed over dates, where each date is one largely independent observation,
instead of over thousands of overlapping pooled returns that inflate it.

---

## Result 1 — the accumulation score is inverted

Cross-sectional rank IC of the composite:

| Period | 20d IC | t | 60d IC | t |
|---|---|---|---|---|
| Train 2001–2016 | **−0.0379** | −5.30 | −0.0379 | −5.07 |
| Holdout 2016–2026 | +0.0001 | +0.02 | −0.0103 | −1.33 |

Holdout quintile spread: **−1.04% at 20d (t = −2.85)** and −2.63% at 60d
(t = −3.38). Still adverse, just weaker than in training.

Strongly negative for the first sixteen years, then decaying to roughly nothing.
Either way it never earns its keep as a *buy* signal.

The pooled bucket test agrees: the lowest-scoring names returned +6.86% over 60
days versus +4.49% for the signal cohort, monotonically across all five buckets.
Wyckoff phase E — which the planner blocks as "a chase" — was the best state at
+10.67%/60d, against +3.38% for the spring setup the engine rates highest.

### Component diagnosis (training half only)

| Component | Family | 20d IC | t | Verdict |
|---|---|---|---|---|
| `trend_persistence` | momentum | **+0.0476** | +5.30 | helps |
| `momentum` (12-1) | momentum | **+0.0436** | +4.26 | helps |
| `relative_strength` | momentum | **+0.0433** | +4.53 | helps |
| `near_high` | momentum | **+0.0313** | +3.18 | helps |
| `wyckoff` | contrarian | −0.0037 | −0.52 | no signal |
| `obv_divergence` | contrarian | −0.0251 | −2.84 | hurts |
| `range_compression` | contrarian | −0.0276 | −3.44 | hurts |
| `volume_dryup` | contrarian | **−0.0478** | −6.35 | hurts most |

The split is perfectly clean along family lines: **every momentum component is
positive, every contrarian component is zero or negative.** The composite wasn't
noise — it was **systematically backwards**, and `volume_dryup` (the single most
Wyckoff-ish component, "supply is drying up") was the most reliably wrong signal
in the whole set.

---

## Result 2 — the momentum profile survives the holdout

Component selection used the **training half only**, then a coarse, deliberately
unfitted weighting (0.30 / 0.25 / 0.25 / 0.20) across 12-1 momentum, relative
strength, trend persistence, and proximity to the 52-week high. The holdout was
not looked at until the profile was frozen.

| Profile | Train 20d IC (t) | **Holdout 20d IC (t)** | **Holdout 60d IC (t)** |
|---|---|---|---|
| `accumulation` | −0.0379 (−5.30) | +0.0001 (+0.02) | −0.0103 (−1.33) |
| `momentum` | +0.0612 (+6.56) | **+0.0313 (+3.08)** | **+0.0462 (+4.92)** |

Holdout quintile spread (top minus bottom, within each date):

| Horizon | Top | Bottom | Spread | t | Dates positive |
|---|---|---|---|---|---|
| 20d | 2.61% | 1.04% | **+1.56%** | 3.69 | 58.3% |
| 60d | 7.49% | 2.33% | **+5.16%** | 5.90 | 62.3% |

At 60 days that is ~+4.76% net of the configured 0.40% round trip.

*(An earlier version of these numbers sampled each ticker every 5 bars from its
own start index, which left tickers on disjoint date grids and only ~8 names per
cross-section. Scoring now runs on a shared exchange calendar — median 52 names
per date — which is both correct and, as it turns out, more favourable.)*

Note the composite beats its own best component: `relative_strength` alone had a
holdout IC of just +0.003 (t = 0.37). The lift comes from combining four
correlated-but-distinct trend measures — an ensemble effect, not one lucky
indicator.

---

## Result 3 — but it has real drawdowns

Holdout, year by year, 60-day quintile spread:

| Year | IC | Spread | | Year | IC | Spread |
|---|---|---|---|---|---|---|
| 2017 | +0.096 | +3.83% | | 2022 | +0.132 | **+9.13%** |
| 2018 | −0.050 | **−3.46%** | | 2023 | +0.048 | −0.20% |
| 2019 | +0.116 | +5.92% | | 2024 | −0.038 | **−6.65%** |
| 2020 | −0.057 | +13.87% | | 2025 | +0.027 | +9.84% |
| 2021 | +0.143 | +6.92% | | 2026¹ | −0.108 | −0.66% |

¹ partial year.

**Six of ten years positive.** 2018 and 2024 lost meaningfully. This is exactly
how momentum behaves everywhere it has been studied: a positive long-run
expectancy punctuated by sharp, multi-quarter crashes, usually at sharp market
reversals. Anyone trading it needs to survive a −6.65% year without abandoning
the strategy at the bottom.

---

## Result 4 — the tradeable version, and how much of it is an illusion

An IC is not money. This simulates what a retail account can actually do: hold
the **top 10 names, equal weight, rebalanced every 60 trading days**, net of a
0.2%-per-side cost on turnover (~51% per rebalance). 42 non-overlapping periods
in the holdout.

| Metric | Strategy | Universe EW | **IHSG** |
|---|---|---|---|
| CAGR | 31.61% | 12.09% | **4.14%** |
| Annual volatility | 38.93% | 24.83% | 15.70% |
| Sharpe | 0.87 | 0.58 | 0.34 |
| Max drawdown | **−16.89%** | −28.19% | −32.23% |
| Hit rate | 64.3% | 61.9% | 52.4% |

Excess vs equal-weight universe: **+4.67% per period, t = 2.32**, positive in
64% of periods. Versus IHSG: +6.83% per period, t = 2.85.

### The survivorship correction — read this before believing the CAGR

**The equal-weight universe beat IHSG by +7.96% CAGR (12.09% vs 4.14%).** That
gap is not skill and not strategy. It is the universe being built from *today's*
LQ45 / deep-history / speculative constituent lists: names that collapsed or were
delisted are simply absent, and names that 10×'d are present for the whole run
*because* they 10×'d.

So of the strategy's headline +27.5% CAGR over IHSG:

- **~8 points are pure survivorship** in the universe construction
- the remaining ~19.5 points are measured against a universe carrying the same
  bias — which is the *fair* comparison, and the one to quote

And even that is generous: a **momentum** strategy benefits from survivorship
more than equal weight does, because the survivors are precisely the names with
persistent momentum. The honest reading of the strategy-vs-IHSG column is
**upper bound, not expectation.**

The one result that is *not* obviously inflated is the drawdown: −16.89% against
−32.23% for IHSG. A momentum screen rotating out of falling names ahead of an
index that must hold them is a mechanism that does not depend on survivorship.

### Sample-size caveat

42 non-overlapping periods is a small sample for a strategy claim. t = 2.32
against the fair benchmark is real but not decisive, and §3's year-by-year table
shows 4 of 10 years losing. This is evidence worth acting on carefully, not a
settled result.

---

## Result 5 — most of the edge was survivorship bias (the decisive test)

Two audits, prompted by an implausible number, materially cut every figure above.

### 5a. Phantom trades: the liquidity cheat

The 2020 backtest showed **+441%** for a 3-name basket. Auditing rather than
accepting it: the basket held ARTO at 418 -> 1078 -> 2239. The prices are real
(Bank Jago genuinely ran ~35x). But ARTO's median turnover at the March 2020
signal was **Rp26 juta/day, with zero-volume sessions**. A Rp10jt position would
have been ~40% of a day's volume. **Unfillable.** The tradeability filter existed
but ran only in the live screener, never in the backtest.

Applying a point-in-time liquidity floor (trailing 20-day median turnover >=
Rp5bn, known at each date):

| | Before | After |
|---|---|---|
| median per 60d | 9.14% | **7.22%** |
| mean | 14.86% | **9.41%** |
| P(>+5%) | 65.8% | **55.3%** |
| 2020 annual | +441% | **+130%** |

### 5b. Survivorship: the universe itself is the problem

The universe is built from **today's** index constituents. Names are present
*because* they later became large enough to be included. Control for it by
restricting to names that were already liquid at the *start* of the holdout —
established in early 2017, before any measured return happened:

| Universe | Names | Median 60d | P(>+5%) | CAGR | maxDD |
|---|---|---|---|---|---|
| all 66 (today's members) | 66 | 7.22% | 55.3% | **35.5%** | −45.9% |
| already >=Rp5bn/day in 2017 | 41 | 2.84% | 44.7% | **8.9%** | −46.1% |
| already >=Rp20bn/day in 2017 | 26 | 0.64% | 39.5% | **1.6%** | −50.8% |

**CAGR collapses from 35.5% to 8.9%.** The mechanism is explicit: 49% of all
picks were names *not* established in 2017, and those returned **+15.7% mean per
60 days versus +4.1%** for the established ones. INKP, BRIS, ENRG, SRTG, ESSA and
similar names are in the ticker list because they were later promoted into an
index. A screener running in 2017 would not have had most of them.

**Honest bottom line: roughly high-single-digit CAGR on a survivorship-controlled
universe, against IHSG's 4.1% over the same period.** A real but modest edge -
not the 35-40% reported earlier in this document, and not enough to support any
claim of ">50% a year".

Part of the drop comes from having fewer names to rank across (41 and 26 are
small pools), so 8.9% is a floor rather than a point estimate. Properly fixing
this needs a point-in-time constituent list including delisted names, which no
free source provides.

## Result 6 — the regime gate is the most robust thing here

Entering with IHSG above its 200-day average: median **+11.7%** per 60 days,
66.7% of periods cleared +5% (n=24). Below it: **+0.2%** and 35.7% (n=14).
Strategy correlation with the index's own next-60-day return is **0.60** - this is
substantially a levered directional bet, which is how momentum behaves everywhere.

Robustness across the parameter, which is what separates a finding from a fit:

| Gate | CAGR | maxDD | % in market |
|---|---|---|---|
| none | 35.5% | −45.9% | 100% |
| IHSG > 50d MA | 27.0% | −43.8% | 66% |
| IHSG > 100d MA | 34.2% | **−10.3%** | 53% |
| IHSG > 150d MA | 36.7% | **−10.3%** | 61% |
| IHSG > 200d MA | 40.4% | **−10.3%** | 63% |
| IHSG > 250d MA | 33.1% | −31.5% | 71% |
| IHSG 60d return > 0 | 38.0% | −17.1% | 53% |

**The drawdown benefit is robust across 100-200 days** (all −10.3%, a ~4x
reduction) and several independent formulations agree. The CAGR peak at exactly
200d is noise - do not read it as optimal. Use ~150d as a middle choice.

## Result 7 — per-name technicals do not discriminate

Comparing winners (>+5% at 60 days) against losers on every score component,
liquid picks only:

| Feature | Winners | Losers | Gap |
|---|---|---|---|
| momentum | 0.526 | 0.530 | −0.004 |
| trend_persistence | 0.481 | 0.478 | +0.002 |
| near_high | 0.352 | 0.365 | −0.013 |
| obv_divergence | 0.500 | 0.482 | +0.018 |
| volume_dryup | 0.423 | 0.440 | −0.017 |

Every gap is under 0.02. At the individual-stock level these indicators carry
**no** discriminating power. Whatever edge exists is cross-sectional ranking plus
regime - not "this chart looks good".

---

# Part II — the full exchange, and an attempt to train to optimum

Everything above ran on 66 index names. That universe is chosen *today*, so it
quietly excludes every company that delisted, collapsed or was suspended — §5b
measured that bias at roughly 8 points of CAGR but could not remove it.

This part re-runs the whole analysis on **every ticker Yahoo will return for
IDX**: 838 symbols, 2.59M daily bars, of which 724 survive the cleaning below.
It is not a true point-in-time universe (a dead company's ticker is gone from
the screener too), but it is a far larger and far less flattering sample.

| | |
|---|---|
| Observations | **133,165** (from 460,519 raw) |
| Tickers | **724** |
| Period | 2001-08-03 → 2026-08-04 |
| Cross-section | median **104 names** per date (was 52) |
| Cleaning | `close ≥ Rp50` (IDX regular-market minimum) and point-in-time trailing-20d **median** turnover `≥ Rp1bn/day`. Trailing-only, so the filter is a decision the screener could have made live. |

```bash
idxbot backtest --universe idx_all --providers none --profile momentum \
    --step 5 --start-index 300 --out reports/obs_full.csv
python3 scripts/backfill_liquidity.py reports/obs_full.csv reports/obs_full_clean.csv
idxbot evaluate    --observations reports/obs_full_clean.csv --split --components
idxbot walkforward --observations reports/obs_full_clean.csv --horizon 60 --top 5
```

The backfill step drops **327,354 of 460,519 rows** — 71% of the raw run is
below Rp50 or under Rp1bn/day of turnover and could not have been traded with
real money. `idxbot backtest` now emits the `vt` column itself; the script
exists to patch observation files produced before that change.

### The cleaning was not cosmetic

The raw run reported a 5-day baseline of **+10.91%**, which is absurd. The cause
was 3,242 observations priced below Rp10 — split-adjustment artifacts, since
IDX's regular market cannot trade below Rp50. Their mean 5-day forward return
was **+1485%**, and one (KOPI at a nominal Rp0.013) showed +4,615,285%. A
handful of impossible rows had moved the average for a quarter of a million
real ones. Anything computed before that filter was wrong.

---

## Result 8 — the median IDX stock loses money

This is the structural fact that governs everything else.

| Horizon | Mean | Median | % positive | Share of total return from the top 1% |
|---|---|---|---|---|
| 5d | +0.25% | +0.00% | 43.7% | **178%** |
| 20d | +0.94% | −0.65% | 45.0% | **111%** |
| 60d | +2.56% | −1.84% | 44.5% | **84%** |

The top 1% of observations supply *more than all* of the total return at 5 and
20 days, which means the other 99% net negative. Fewer than half of all 60-day
holding windows are profitable.

Two consequences follow directly:

- **Diversifying across IDX is not safe, it is the trap.** Buying the average
  name is a losing proposition; the index survives only because a few enormous
  winners drag the mean up. This is why the equal-weight benchmark below is so
  weak, and why "just buy a basket" fails here in a way it does not in the US.
- **The strategy must be selective and it must let winners run.** A rule that
  caps upside — a fixed 5% target, say — cuts off precisely the tail that pays
  for everything else.

## Result 9 — the composite decayed; one component did not

Component IC on the full universe at 20 days, each judged alone:

| Component | mean IC | t |
|---|---|---|
| **near_high** | **+0.0660** | **10.20** |
| trend_persistence | +0.0365 | 6.41 |
| momentum | +0.0257 | 4.18 |
| relative_strength | +0.0221 | 3.54 |
| range_compression | −0.0073 | −1.44 |
| obv_divergence | −0.0116 | −2.03 |
| volume_dryup | −0.0178 | −3.66 |

**The best single component beats the composite it is blended into** (composite
20d IC +0.0470). The three contrarian components are not merely weak, they are
negative — Result 1 replicated on eleven times the data.

The era breakdown is what makes this decisive:

| Era | names/date | composite IC60 | t | **near_high IC60** | t |
|---|---|---|---|---|---|
| 2001-07 | 23 | +0.0578 | 3.26 | +0.0566 | 3.32 |
| 2008-12 | 79 | +0.1138 | 7.09 | +0.1266 | 8.23 |
| 2013-17 | 107 | +0.0497 | 4.15 | +0.0748 | 6.25 |
| 2018-20 | 126 | **−0.0047** | −0.31 | +0.0334 | 2.16 |
| 2021-26 | 221 | +0.0286 | 3.58 | **+0.1034** | **13.85** |

The composite went to **zero** in 2018-20 and only partly recovered. `near_high`
never turned negative and is *strongest in the most recent era*. On the
chronological holdout the same gap appears: composite 60d IC +0.0217 (t=2.78)
against near_high +0.0868 (t=10.67).

Note also that the survivorship correction bit hard, exactly as §5b predicted:
the composite's holdout 60d IC fell from **+0.0462 on the 66-name universe to
+0.0217** here. More than half the apparent edge was the universe, not the signal.

**Why proximity to the 52-week high works and the others fade.** It is the one
component that is not a lookback statistic — it is a *level* relative to a
reference every participant can see. It encodes the absence of trapped sellers:
above the 52-week high nobody is holding a losing position waiting to break
even, so the supply that normally caps a rally is not there. Momentum and trend
persistence measure the same underlying trend but through a rear-view window,
and that window keeps shortening as the market gets faster. This also explains
the era pattern: as the cross-section grew from 23 names to 221, statistics that
need a stable lookback degraded while a positional fact did not.

## Result 10 — training to optimum made it worse

The request was to train until optimum. The honest way to do that is
walk-forward: choose the weighting on an expanding training window, score it on
the next two years it has never seen, refit, repeat. Ten candidate weight sets,
nine folds, non-overlapping 60-day rebalances, top-5 equal weight.

```
 fold  1  train->2009-08  test 2009-08..2011-08  chose [rel_strength only]   oos  +0.76%
 fold  2  train->2011-08  test 2011-08..2013-08  chose [equal all-7]         oos  +5.05%
 fold  3  train->2013-08  test 2013-08..2015-08  chose [equal all-7]         oos  −2.41%
 fold  4  train->2015-08  test 2015-08..2017-08  chose [near_high heavy]     oos  +6.05%
 fold  5  train->2017-08  test 2017-08..2019-08  chose [equal trend-4]       oos +13.64%
 fold  6  train->2019-08  test 2019-08..2021-08  chose [equal trend-4]       oos  +2.81%
 fold  7  train->2021-08  test 2021-08..2023-08  chose [shipped momentum]    oos  +9.51%
 fold  8  train->2023-08  test 2023-08..2025-08  chose [shipped momentum]    oos  +4.82%
 fold  9  train->2025-08  test 2025-08..2026-08  chose [shipped momentum]    oos  +3.93%
```

No candidate won more than 3 of 9 folds. And the comparison that matters:

| Strategy (2009-08 → 2026-08, 60d, top-5) | mean/period | median | win |
|---|---|---|---|
| **walk-forward (re-optimised every 2y)** | **+4.97%** | +3.11% | 59% |
| fixed: near_high only | **+9.51%** | +4.61% | **66%** |
| fixed: momentum only | +9.78% | +1.33% | 55% |
| fixed: equal trend-4 | +8.93% | +3.39% | 59% |
| fixed: shipped momentum | +8.62% | +2.97% | 63% |
| fixed: contrarian | +1.87% | −2.16% | 42% |
| *equal-weight universe (no selection)* | *+2.69%* | *+2.03%* | *58%* |

**Re-optimising roughly halved the return.** Every fixed trend profile beat the
adaptive one. The optimiser was not finding a moving optimum; it was chasing
whichever weighting had just had a good run, and arriving late every time.

This is the answer to "train yourself until you reach optimum": on this data,
*there is no optimum to train to*. The differences between sensible trend
weightings are inside the noise, and the act of choosing between them
destroys value. What survives is much cruder and much more robust — be in the
trend family at all, and stay there.

Compounded over the same 17 years:

| | Total | CAGR | max DD | t |
|---|---|---|---|---|
| near_high only | +11,216% | **32.1%** | −51.1% | 2.87 |
| equal trend-4 | +10,335% | 31.4% | −49.4% | 3.28 |
| shipped momentum | +8,552% | 30.0% | −48.4% | 3.20 |
| contrarian | +9% | 0.5% | −65.6% | 0.80 |
| *equal-weight universe* | *+313%* | *8.7%* | *−40.6%* | |

The contrarian profile turned 17 years into **+9% total**. That is the
accumulation thesis, priced.

### Robustness — 32 cells, not one lucky setting

Mean return per rebalance minus the equal-weight benchmark, every combination
of horizon and position count:

| | 20d top-1 | 20d top-3 | 20d top-5 | 20d top-10 | 60d top-1 | 60d top-3 | 60d top-5 | 60d top-10 |
|---|---|---|---|---|---|---|---|---|
| near_high only | +3.05% | +1.99% | +2.46% | +2.22% | +5.52% | +4.41% | +6.82% | +6.14% |
| shipped momentum | +2.18% | +3.53% | +3.29% | +2.65% | −0.85% | +5.71% | +5.93% | +5.28% |
| equal trend-4 | +2.35% | +3.62% | +3.40% | +2.47% | −1.73% | +5.70% | +6.24% | +4.50% |
| contrarian | −1.54% | −0.77% | −0.56% | −0.76% | −3.44% | −2.98% | −0.81% | −2.34% |

The trend family beats the benchmark in 23 of 24 cells. The contrarian profile
loses in **all eight**. The single failure is 60d top-1, where holding one name
for a quarter is mostly variance — which is itself a useful warning against
concentrating into a single position at long horizons.

### Does it still work now?

| Span | benchmark | near_high | edge | shipped momentum | edge |
|---|---|---|---|---|---|
| 2009-08 → 2015-01 | +4.12% | +6.60% | +2.48% | +8.61% | +4.50% |
| 2015-01 → 2021-01 | +1.96% | +4.85% | +2.89% | +10.08% | +8.12% |
| 2021-01 → 2026-09 | +2.61% | **+16.46%** | **+13.85%** | +10.47% | +7.86% |

Yes — and the edge is largest in the most recent period. That is unusual and
deserves suspicion rather than celebration: the recent cross-section is three
times wider (221 names vs 79), so the top-5 is selected from a much deeper pool,
and a wider pool means a more extreme top. Expect some of this to be a
sample-size effect rather than a genuinely improving market.

### The caveat that limits all of Part II

Non-overlapping 60-day rebalances give only **71 independent observations** over
17 years. A t-stat near 3 on 71 samples is real but not overwhelming, and the
−50% drawdown is the number to plan around, not the 32% CAGR. Overlapping
windows would show t≈9, which is why this repo does not use them.

---

# Part III — the exit rule, and how to actually reach an 80% win rate

Everything above measures returns close-to-close at a fixed horizon: buy today,
look 60 bars later. That is the right way to test whether a *signal* carries
information and the wrong way to test a *trade*, because nobody holds blind for
three months. `idxbot barrier` walks each position forward bar by bar until it
touches a target or a stop.

Three rules keep it honest, all unit-tested:

- **Entry is the next bar's open**, never the signal bar's close — that close is
  not a price you could have traded on.
- **A bar that spans both barriers scores as a stop.** Daily data cannot say
  which came first, and assuming the good one is the single easiest way to
  manufacture a win rate that does not exist.
- **Costs charged every trade** (0.4% round trip). At a 3% target that is an
  eighth of the gross win, which is exactly why small targets flatter the hit
  rate and starve the return.

## Result 11 — the take-profit is what destroys the edge, not the stop

Reconciling the barrier engine against the fixed-horizon baseline, same signals:

| exit rule | expectancy | avg win | days held |
|---|---|---|---|
| hold 60 days flat | **+9.27%** | +28.62% | 59 |
| stop −8%, no target | +5.60% | +28.78% | 34 |
| target +5%, no stop | **+0.02%** | +4.88% | 24 |
| target +5% and stop −8% | +0.11% | +4.88% | 12 |

The flat hold reproduces §10's +9.51%, so the engine is not lying. Everything
follows from the second column: **unconstrained winners average +28.6%**, and a
+5% target amputates them. Selling the whole position at +5% converts a +9.27%
expectancy into +0.02% — the entire edge, gone, while the *stop* costs barely a
third of it.

This is Result 8 restated as a trading instruction. The top 1% of observations
supply more than 100% of total return; a fixed target is a machine for
guaranteeing you are never holding them.

## Result 12 — 88% win rate, and what it costs

A high win rate is trivially purchasable by widening the stop, and that road
ends in ruin, so every figure below is paired with expectancy. The structure
that actually works has three parts:

1. **Sell only a quarter at the target.** The rest still runs.
2. **Lift the stop to entry once the target prints.** This — not the scale-out —
   is what lifts the win rate. A 25% slice banked at +2% cannot rescue the other
   75% falling to the stop; a breakeven floor can.
3. **A wide −15% initial stop**, because a tight one ejects you before the
   target ever prints and arms the floor.

Scaling out alone already transforms the result: `+3%/−8%` goes from **−0.48% to
+3.78%** expectancy by selling 25% at the target instead of 100%, hit rate
unchanged.

**Validated on a chronological holdout.** The config was chosen on 2009–17 and
then run on 2018–26, untouched:

| rule | exit | train win | holdout win | holdout expectancy | PF |
|---|---|---|---|---|---|
| shipped momentum | +2%/−15% x25% BE | 87% | **86%** | +3.38% | 2.61 |
| shipped momentum | +3%/−15% x25% BE | 84% | 83% | +4.11% | 2.57 |
| near_high only | +2%/−15% x25% BE | 82% | 82% | +2.67% | 2.03 |
| shipped momentum | +5%/−10% x25% (no BE) | 44% | 38% | +5.86% | 2.13 |
| shipped momentum | hold 60d flat | 61% | 50% | +8.02% | 1.73 |

Picking purely on the train half selects `+2%/−15% x25% BE`, and it delivers 86%
out-of-sample. It holds across **9 of 9** rule × position-count combinations
(82–87%), so it is not a single lucky cell.

### The cadence correction, and an artifact caught

Barrier trades exit after ~17 days. Rebalancing every 60 leaves capital idle for
43 of them, which is why the naive portfolio CAGR looked like 8.8%. Matching the
cadence to the holding period is what the rule is actually worth — but that is
only legitimate if a tranche closes before the next opens, so:

| exit rule | still open at 20d | implied leverage | verdict |
|---|---|---|---|
| +2%/−15% x25% BE | 27% | **0.86x** | valid (capital idle 14% of the time) |
| +3%/−15% x25% BE | 36% | 1.10x | marginal |
| hold 60d flat | 100% | **2.98x** | **invalid** |

Holding flat at a 20-day cadence appeared to return 122% CAGR. It was three
positions deep at all times — 3x leverage, not a strategy. Discarded. The
honest comparison, each at a cadence it can actually support:

| | win rate | CAGR | max DD | leverage |
|---|---|---|---|---|
| **+2%/−15% x25% BE, 20d cadence** | **88%** | **33.2%** | **−30.5%** | 0.86x |
| hold 60 days flat, 60d cadence | 56% | 27.7% | −50.2% | 1.00x |

Better on all three: higher win rate, higher return, materially smaller
drawdown. The gain is not from better selection — the signal is identical — it
is from recycling capital three times as often at a slightly lower per-trade
return.

### What the 88% is not

- **Not 88% of trades making +5%.** The target is +2%, and the average win is
  +5.05% only because the untouched 75% sometimes runs a long way. Roughly a
  sixth of wins are near-scratch exits at breakeven after costs.
- **Not low-risk.** The 12% of trades that lose average **−12.8%**, and the
  portfolio still draws down 30%. Profit factor 2.23 is the honest summary; the
  win rate on its own is the most misleading number in this document.
- **Not free of selection.** Six exit configs were compared, and while the
  holdout confirms the winner, six is still six. The effect is large and
  monotonic across the grid rather than a single spike, which is the reassuring
  part.
- **Not a day-trade rule.** Average hold is 17 days. Nothing here rehabilitates
  the intraday results in `docs/DAYTRADE.md`.

---

# Part IV — foreign flow, volume, macro

Three axes beyond price technicals. One is still unmeasurable, one is actively
unstable, and one is real but does something other than what it appears to.

```bash
idxbot macro                    # current regime and the foreign-appetite proxy
```

A data-layer bug had to be fixed first: `to_yahoo_symbol` appended `.JK` to
anything without a dot, so `BZ=F`, `USDIDR=X` and `EEM` became `BZ=F.JK` and
friends. Yahoo answers those with an empty result rather than an error, so every
macro series simply came back missing with no indication why. Twelve series now
retrieve, most from 2000–2003.

## Result 13 — foreign flow is still unmeasured, and the proxy is honest about it

Per-broker foreign net buy/sell was never obtainable, so nothing here measures
it. What is measurable is the constraint every foreign buyer faces: **to own an
Indonesian share you must first own rupiah.** Sustained foreign accumulation
should therefore show up as rupiah strength alongside EM risk appetite.

`foreign_appetite` combines those two as trailing percentiles. It is a proxy for
foreign *appetite*, not a measurement of foreign *flow*, and both the module and
the rendered output say so wherever the number appears.

In-sample it looked like the strongest conditioner in the entire project:

| foreign appetite | n | strategy 20d return | win rate |
|---|---|---|---|
| high | 675 | **+6.70%** | 56% |
| low | 675 | **+1.39%** | 51% |

A 5.3-point spread, wider than any technical component. Every other macro cut
agreed and every sign was economically sensible — rupiah weakness bad, strong
dollar bad, copper strength good, IHSG above its 200-day good. That coherence is
exactly what makes it seductive, and it is why §14 matters.

## Result 14 — macro times the market; it does not improve selection

Decomposing each rebalance date into the top-5 return, the equal-weight universe
return, and the difference between them:

| macro state | n | strategy | benchmark | **excess** | t(excess) |
|---|---|---|---|---|---|
| foreign appetite high | 135 | 6.08% | 2.06% | **4.02%** | 2.92 |
| foreign appetite low | 135 | 2.20% | −0.27% | **2.47%** | 2.88 |
| copper 60d up | 137 | 5.93% | 2.07% | 3.86% | 2.88 |
| copper 60d down | 137 | 2.36% | −0.26% | 2.62% | 3.01 |
| dollar 60d down | 137 | 5.65% | 1.62% | 4.04% | 3.06 |
| dollar 60d up | 137 | 2.64% | 0.19% | 2.45% | 2.71 |

The strategy's raw return swings by ~3.9 points across macro states. Its **excess
over the market** swings by ~1.5 points and stays firmly positive in both, with
t-stats bunched around 2.7–3.1. **The macro effect is inherited market beta.**
Good macro lifts everything, including the benchmark; the selection edge is
indifferent to it.

And the part that does not survive contact with a holdout:

| | train | holdout |
|---|---|---|
| selection edge, good macro | 3.84% | 3.90% |
| selection edge, bad macro | 0.63% | **4.56%** |
| difference | +3.21% | **−0.66%** |

The train half says selection works better in good macro. The holdout says the
opposite. That is noise.

The market-timing effect itself is weak and mostly decays:

| feature | train t | holdout t |
|---|---|---|
| copper 60d | 2.66 | **0.44** |
| EM equities 60d | 1.91 | **−0.18** |
| dollar index 60d | 1.91 | **0.15** |
| foreign appetite | 1.30 | 1.83 |
| IHSG vs 200d | 0.61 | 1.92 |

Only the two IDX-proximate ones held up, and neither reaches significance.
Acting on it does not pay either — skipping the worst macro quintile moves CAGR
47.2% → 40.3% and max drawdown −55.9% → −48.3%, which is the same
return-per-drawdown with fewer trades.

**Verdict: macro is context for how much to hold, not a stock picker.** That is a
real use — knowing the benchmark is −0.27% rather than +2.06% is worth having
when sizing — but it is not the edge the in-sample table advertised.

## Result 15 — volume carries no stable cross-sectional signal

Every volume feature tested **reverses sign** between the training and holdout
halves, both directions significant:

| feature | train 60d IC | t | holdout 60d IC | t |
|---|---|---|---|---|
| turnover rank | +0.0327 | 5.00 | **−0.0280** | −5.01 |
| turnover growth | +0.0268 | 4.55 | **−0.0182** | −3.01 |
| volume dry-up | −0.0282 | −4.49 | **+0.0320** | +5.26 |
| OBV divergence | −0.0192 | −2.74 | +0.0037 | 0.60 |

This is not a weak signal, it is an unstable one, and the distinction matters:
a weak feature costs a little, whereas a feature that flips sign actively hurts
in the period after the one it was fitted on. Adding `obv_divergence` at weight
0.15 lifts full-sample 60d IC from +0.0535 to +0.0562 — a pure in-sample
improvement that the split says would reverse.

**Volume stays out of the score.** The only volume-derived quantity that earns
its place is the liquidity *filter* — turnover ≥ Rp1bn/day — which exists to
exclude untradeable names, not to rank tradeable ones.

## Result 16 — what actually survives, all four axes

| axis | status |
|---|---|
| **technical** | the one durable edge: distance from the 52-week high, 20d IC +0.066 (t=10.2), non-decaying |
| **macro** | moves the market, weakly and decaying; does **not** improve selection |
| **volume** | no stable signal — every feature reverses out-of-sample |
| **foreign flow** | **still unmeasured**; the rupiah/EM proxy inherits the macro verdict |

Four axes examined, one edge. That ratio is the honest summary of this project.

---

## What this does and does not establish

**Does:**
- The contrarian accumulation thesis, in its price-only form, is refuted on IDX.
- A simple trend-following composite has a genuine, statistically significant,
  economically meaningful cross-sectional edge at a 60-day horizon out-of-sample.

- On the full 724-name exchange, with sub-Rp50 artifacts and illiquid names
  removed, a fixed trend rule beat the equal-weight universe in 23 of 24
  horizon/position combinations, and the contrarian rule lost in all 8.
- Re-optimising the weighting out-of-sample destroys value relative to fixing it.

**Does not:**
- **That the broker-flow thesis is wrong.** It was never tested — no real broker
  summary was obtainable (`idx.co.id` WAF-blocked, Stockbit auth-gated, GoAPI
  key-gated). The `momentum_plus_flow` profile exists precisely to run that
  experiment when data is connected. That remains the open question. Every
  day-by-day comparison against broker activity that was asked for is blocked
  on exactly this, and no number in this document should be read as saying
  anything about what the foreign desks were doing.
- **That any of this is fundamental analysis.** Only 4 years of annual
  statements are retrievable, and the ratios (PE, PB, ROE, margins) come as a
  *current* snapshot with no history. Point-in-time fundamentals over 25 years
  do not exist in any source reachable from here, so a fundamental backtest is
  not possible — not difficult, not expensive, not possible. Any fundamental
  screen built on this data would be silently comparing today's balance sheet
  against a 2009 price, which is the purest form of look-ahead there is.
  `idxbot fundamentals` therefore ships as a *present-tense* exclusion filter
  and deliberately offers no historical sampling. Two traps it does handle:
  **13 of the 45 LQ45 names report in USD while trading in IDR**, so Yahoo's
  price/book is nonsense as published (ADRO 14,941x, INCO 20,074x — both
  actually under 1.3x once repaired at spot); and a sub-1 current ratio is
  normal for banks, telcos and toll roads, so testing it alone flags TLKM,
  MTEL and UNVR as distressed. Neither is visible unless you look.
- **That the universe is truly point-in-time.** A delisted company's ticker is
  absent from the screener that built `idx_all`, so some survivorship remains.
  The correction from 66 to 724 names cut the holdout IC by more than half; the
  remaining bias points the same way, so treat these numbers as still optimistic.
- **That I discovered something.** 12-1 momentum is among the most-replicated
  factors in the literature. Confirming it works on IDX is a sanity check on the
  machinery, not a finding. Its inclusion was informed by knowledge predating
  this dataset — no leakage from *this* holdout, but not a discovery either.
- **That the reported numbers are achievable.** Still no survivorship adjustment
  (today's constituents), and quintile spreads assume you can trade both legs;
  shorting IDX single names is restricted in practice, so the realistic version
  is long-only top quintile (+6.24%/60d holdout) against a benchmark.

---

## What to do with it

- **Default profile is now `momentum`.** `accumulation` is retained, and
  honestly labelled, because its broker components are the untested half.
- **Do not size a strategy off the 20d edge.** It is +1.56% gross against a
  0.40% round trip, and turnover triples versus the 60-day version. The horizon
  that works is 60 days.
- **Quote the strategy-vs-universe number, not the CAGR.** +4.67% per period
  over an equally-biased universe is defensible; 31.6% CAGR is not.
- **Re-run this with real broker summary.** `--profile momentum_plus_flow`
  combines trend with institutional flow. If broker flow adds information beyond
  price, the holdout IC will rise above +0.041. That is the experiment worth
  paying a data vendor for.

Part II changes two of those instructions:

- **Do not re-optimise the weights.** §10 measured that as a ~4.5 point per
  period cost. Pick one trend profile and leave it alone.
- **Filter before you rank, every time.** `close ≥ Rp50` and a point-in-time
  turnover floor. Without them the backtest reports numbers that are not merely
  optimistic but arithmetically impossible.

```bash
# reproduce everything above
idxbot backtest --universe all --providers none --profile accumulation --out reports/obs_acc.csv
idxbot backtest --universe all --providers none --profile momentum     --out reports/obs_mom.csv
idxbot evaluate  --observations reports/obs_mom.csv --split --components
idxbot portfolio --observations reports/obs_mom.csv --split --top-n 10 --horizon 60
python3 scripts/robustness.py reports/obs_mom.csv

# Part II - full exchange, then out-of-sample parameter selection
idxbot backtest    --universe idx_all --providers none --profile momentum \
                   --step 5 --start-index 300 --out reports/obs_full.csv
idxbot evaluate    --observations reports/obs_full_clean.csv --split --components
idxbot walkforward --observations reports/obs_full_clean.csv --horizon 60 --top 5
```

---

*A backtest that contradicts the strategy is more valuable than one that
flatters it. The first result here cost the original thesis; the second was only
trustworthy because the holdout was left untouched until the end.*


---

# Part V — foreign flow, finally measured

For most of this project the README said per-stock foreign flow could not be
obtained. **That was wrong**, and the error was one of search rather than
availability: only `idx.co.id` and commercial vendors were ever probed, never
public datasets. `wildangunawan/Dataset-Saham-IDX` carries `foreign_buy` and
`foreign_sell` per stock per day for **958 IDX tickers, 2019-07-29 → 2025-02-21**.

Verified directly rather than taken on trust: HTTP 200, 312,457 bytes for BBCA,
and the unit assertion `foreign_buy <= volume` holds on **100.000% of 913,084
usable rows**, which is what confirms the figures are shares. Market-wide yearly
totals match known reality — **−61.7trn** in 2020's COVID outflow, **+44.0trn**
in 2022's inflow, **−28.9trn** in 2024.

## Two different things are called "foreign flow"

| | measure | unit | source |
|---|---|---|---|
| **F-flag** | IDX's per-trade foreign-investor flag, aggregated per stock/day | **shares** | this dataset |
| **F-broker** | sum over members flagged foreign in `config/brokers.yaml` | lots + IDR | `bandarmology.foreign_flow` |

They do not reconcile — a foreign investor can trade through a domestic member,
and a foreign-*owned* member (YP/Mirae) mostly serves domestic retail. They are
never summed, differenced, or shown in one column.

## Result 17 — every validation passed, and the conclusion was still wrong

Cross-sectional rank IC on **322,827 liquid rows, 783 tickers**, median 248 names
per date. 20-day cumulative net foreign over turnover:

| horizon | mean IC | t |
|---|---|---|
| 5d | −0.0060 | −2.67 |
| 20d | −0.0185 | −7.83 |
| **60d** | **−0.0254** | **−10.88** |

Strongly negative, and monotone in *both* horizon and accumulation window. Then
every robustness check passed:

- **Chronological split**: train 60d IC −0.0179 (t=−5.41), **holdout −0.0353
  (t=−11.05)** — *stronger* out of sample.
- **Year by year**: negative in all six years (2019 −0.085, 2020 −0.020,
  2021 −0.003, 2022 −0.012, 2023 −0.030, 2024 −0.041).
- **Not a size proxy**: survives inside every size tercile (smallest −0.0350
  t=−11.46, largest −0.0161 t=−3.98).

The reading that invites itself — "fade foreign buying, t = −10.88" — is wrong,
and would have been the wrong trade.

### The deciles show why

| decile | mean 60d forward |
|---|---|
| **D1 heaviest foreign SELLING** | **+3.34%** |
| D4 | −0.38% |
| D6 | −2.47% |
| D7 | −2.57% |
| D9 | +1.81% |
| **D10 heaviest foreign BUYING** | **+1.25%** |

**U-shaped, not monotone.** Rank IC assumes monotonicity; when the true shape is
a U it reports a large negative number measuring the slope through the middle
and says nothing directional about the tails. Both extremes of foreign activity
outperform a quiet centre — which is closer to a volatility effect than an
information one.

The tradeable spread confirms it. Q1−Q5 is **+0.15%, t = 0.76**. On
non-overlapping 60-day rebalances every variant sits inside its own error bar:

| | mean 60d | t |
|---|---|---|
| long D1 (fade buying) | +4.24% | 1.10 |
| long D10 (follow buying) | +1.63% | 0.54 |
| equal-weight all | −0.32% | −0.11 |

**Net foreign flow, measured this way, is not a tradeable signal in either
direction.** The bandarmology premise — follow the foreign money — is not
supported; neither is its inverse.

### The methodological point

This is the cleanest example in the project of a statistic passing every check
and still not meaning what it appears to. Split-sample, year-by-year, size
controls, a t-statistic of −10.88 — and a long-short spread of zero. **Rank IC
is a monotonicity measure. Always look at the deciles before trading the IC.**

### Live sources, for later

The dataset ends **2025-02-21** — 18 months stale, a research source and never a
live one, and the upstream repo carries IDX's anti-crawling notice and forbids
commercial use. Two live routes were verified and recorded:

- **infovesta.com** — top-5 foreign buy/sell, no auth (verified HTTP 200),
  previous session only, 5 rows per side. A freshness ping, not a universe scan.
- **api.zpi.web.id** — key-gated foreign-flow route (verified 401 without a
  key), documented to carry both shares and IDR back to 2020, free tier 2,000
  requests/month.

```bash
python3 scripts/foreign_flow_study.py     # clones, unit-checks, and re-runs all of the above
```

---

# Part VI — the broker summary itself, finally connected

## Result 18 — the access problem was licensing and geography, not skill

For most of this project's life the README asserted there was no free public
source of IDX broker summary. That claim is now retired. It was wrong the same
way the foreign-flow claim in Part V was wrong: **the search was too narrow, not
the world too closed.**

Three verified facts settle how commercial sites do it:

1. **`idx.co.id` blocks the network, not the endpoint.** Cloudflare returns 403
   for the broker-summary JSON, the stock-summary JSON, the digital-statistics
   API *and the bare homepage*. Nothing is being defended selectively — this
   egress is unwelcome, which is ordinary treatment for a datacentre IP outside
   Indonesia.
2. **The platforms are not scraping.** Stockbit, RTI and the bank terminals are
   IDX-licensed data-feed subscribers redistributing under licence. There is no
   clever request to reverse-engineer because they are not making one.
3. **IDX prohibits scraping explicitly**, so the 403 is policy being enforced.

The route that works was never IDX and never a vendor: **an exchange member
publishes the table**. IndoPremier renders the full rekap broker at a public,
unauthenticated URL. One GET, no key.

### It is the real table, and here is why that is not a guess

Regular-board totals against Yahoo's tape, 2026-08-13:

| ticker | parsed | tape | relative error |
|---|---|---|---|
| BBCA | 832,077 lots | 832,080 | 4e-6 |
| ANTM | 860,776 | 860,777 | 1e-6 |
| ASII | 484,063 | 484,073 | 2e-5 |
| UNVR | 99,947 | 99,958 | 1e-4 |

Independently, the table over-determines itself: value must equal
lots x 100 x average. Pooled over 160 rows and eight stocks the median
disagreement is **0.2%**, the maximum 4.5% — under the 5% ceiling that one
decimal place of display rounding permits. A swapped or mis-scaled column would
miss by orders of magnitude, so nothing real lands in between.

## Result 19 — a top-10 view cannot balance, and the ledger integrates the gap

This is the failure mode worth naming, because it produces confident,
precise-looking numbers that are wrong.

In a complete rekap every lot bought is a lot sold, so summing all members gives
exactly zero net every session. A top-10 view breaks that identity **and not
randomly**: a broker appears only on the side where it was large that day. A
steady accumulator therefore shows up among the top buyers constantly and the
top sellers rarely, its unobserved selling is censored away, and a cumulative
inventory ledger marches upward whether or not it bought anything on net.

Measured on BBCA over 52 sessions to 2026-08-13:

| | |
|---|---|
| DX appears as a top-10 buyer | 21 days |
| DX appears as a top-10 seller | 5 days |
| market-wide cumulative net (must be 0) | **−2,808,171 lots** |
| that drift as a share of observed flow | 2.1% |

**Direction over a window and relative ranking between brokers survive this. An
absolute position, and any cost basis or open P/L derived from one, does not.**
`truncation_bias()` measures it and `idxbot analyze` prints it directly above
the position table rather than in a footnote.

## Result 20 — connecting real data immediately found a silent bug

Twenty broker codes appearing in live IDX data had no entry in
`config/brokers.yaml`, so they defaulted to `foreign: false`. That default is
not neutral — it pushed genuine foreign flow into the domestic bucket and
understated net foreign for every stock those desks touched. They are added
now, carrying the source's own F/D flag, which was stable across 90 fetches
(30 stocks x 3 dates) with no code ever changing.

The flag agrees with the repo on every confident entry, **including YP (Mirae)
as foreign** — the retail-serving, foreign-owned broker that motivated
`foreign_basis` in the first place. It disagrees on BQ, DR and TP, which have
Korean, Malaysian and Singaporean parents yet come back domestic every time;
the exchange-side flag appears to follow the member's registration rather than
its shareholders. Both readings answer different questions, so the disagreement
is recorded rather than resolved by fiat.

The simulator was also removed from the default provider chain. It existed
because no real source was reachable; as a fallback it would let a transient
network failure swap fabricated flow into a real-looking report one ticker at a
time.

## Result 21 — the classification is three-way, and the third bucket is the interesting one

The first version of the parser read the source's broker labels as
foreign-or-domestic. There is a third: **`bumn`** — *Badan Usaha Milik Negara*,
an Indonesian state-owned enterprise. Matching only two buckets silently
dropped every state-owned desk from the classification while they continued to
count toward totals.

Across 18 stocks x 10 dates, with no code ever changing class, exactly four
houses carry it: **CC** (Mandiri Sekuritas), **DX** (Bahana), **NI** (BNI
Sekuritas) and **OD** (BRI Danareksa) — the securities arms of the state banks.

This matters beyond bookkeeping. *"The state is accumulating"* and *"a domestic
institution is accumulating"* are different claims, and only one of them is
interesting; in Indonesia the state-linked desks are frequently the counterparty
absorbing foreign selling. A `local_inst` tier cannot express that, so
`state_owned` is now a third axis on `Broker`, independent of `tier` and
`foreign`.

**DX had no registry entry at all** until real data surfaced it — a state-owned
house among the largest desks on the exchange, invisible to every amount of
prior desk research, found in one afternoon of actual broker summary. That is
the argument for connecting data before trusting a config file, in one line.

## What this does and does not establish

It establishes that the data is obtainable, that what arrives is genuinely the
exchange's rekap broker, and that the top-10 truncation has a measured, bounded
effect on which conclusions are safe.

**It establishes nothing about whether broker flow predicts returns.** No
signal in `bandarmology.py` has been tested against forward returns. Part V is
the standing warning here: net foreign passed a split-sample test, a
year-by-year test, size controls and a t-statistic of −10.88, and still had a
long-short spread of zero. The machinery being connected is the start of that
work, not the end of it.

```bash
idxbot analyze BBCA                  # real broker flow, no configuration
python3 -m pytest tests/test_ipot.py # 49 tests, all offline against real captures
```

---

# Part VII — Hull Suite + UT Bot on IDX large caps

The method as usually described: **buy when UT Bot prints a buy and the Hull
band is green; sell when it prints a sell and the band turns red.** Both halves
are ported faithfully from their published Pine sources rather than
approximated, and each is also tested alone, because a combination that is not
compared against its parts is untestable by construction.

## How it was tested

| | |
|---|---|
| Universe | 84 IDX names with 10+ years of history: LQ45, large caps, and conglomerate-controlled listings |
| History | to 2000, median 5,472 bars per name |
| Execution | signal read at the close of bar *t*, filled at the **open of bar t+1** |
| Costs | 0.15% buy, 0.25% sell (the extra 0.1% is the sale tax), 0.10% slippage each way |
| Auto-rejection | a bar locked limit-up cannot be bought, limit-down cannot be sold |
| Dividends | accrue **only while the position is open** |
| Benchmark | buy-and-hold **the same stock over the same window**, total return |

The last two matter more than they look. A timing rule is out of the market
most of the time and forgoes the dividends paid while it is out; on IDX blue
chips at 3-6% yields, comparing a price-return strategy against a price-return
benchmark would hide most of the gap. And the benchmark has to be the stock
itself — beating IHSG while losing to ASII is a stock-picking result wearing a
timing costume.

Before any of it, the ports were checked against independent naive
implementations (agreement to 1e-14) and, more importantly, **against
truncation**: cut the series at bar *t*, recompute, and every value at or
before *t* must be unchanged. That is the only test that can catch look-ahead,
and this repo has shipped a look-ahead bug before (Part III). Wilder's RMA was
confirmed to be exactly `EMA(2n-1)` asymptotically, which pins the ATR down as
Wilder's rather than a simple mean — the most common way a Pine port silently
drifts.

## Result 22 — at published defaults it loses to owning the stock, everywhere

Hull 55/HMA, UT key 1.0 / ATR 10. Nothing fitted.

| variant | CAGR | buy & hold | excess | names beating B&H |
|---|---|---|---|---|
| UT Bot alone | -5.68% | +11.76% | **-16.44%** | 13% |
| Hull alone | +6.93% | +11.76% | **-3.09%** | 36% |
| **confluence (the method)** | **-1.02%** | +11.76% | **-10.25%** | **18%** |
| confluence, Hull exit only | +4.31% | +11.76% | -5.70% | 27% |

Blue chips and conglomerate-controlled names give the same answer (-11.37% and
-10.25% excess). On blue chips **2 of 54 names** beat buy-and-hold.

The confluence filter does help — adding the Hull green filter to a bare UT Bot
recovers 6.2 points of excess (-16.44% to -10.25%) by blocking the worst
entries. It just does not recover enough to reach zero, and the combination is
**7.2 points worse than the Hull half used alone**. Both readings point the same
way: the UT Bot is the component destroying the result, and bolting it onto the
Hull filter subtracts value rather than confirming it.

## Result 23 — it is not a cost problem

| costs | CAGR | excess vs buy & hold |
|---|---|---|
| frictionless (impossible) | +2.65% | **-6.52%** |
| fees only, no slippage | +0.20% | -9.10% |
| realistic | -1.02% | -10.25% |
| wide spread | -2.57% | -11.92% |

Costs are worth about 3.7 points a year and are the difference between a small
profit and a small loss. But **with zero costs it still loses to buy-and-hold
by 6.5 points a year.** The rule is not a good rule being eaten by friction; it
is a rule that is worse than not trading, made worse by friction.

Per trade on ASII: gross +0.58%, net +0.18%, with 0.40% going to costs.
Compounded over 25.8 years that is 2.24x gross and **0.96x net** — costs
consume the entire edge and the strategy ends below where it started.

## Result 24 — it worked once, in the era that suited it

| era | Hull alone | buy & hold | excess | names beating B&H |
|---|---|---|---|---|
| 2001-2008 | +13.36% | +10.00% | **+4.25%** | **61%** |
| 2009-2014 | +9.43% | +31.61% | -19.52% | 17% |
| 2015-2020 | +0.39% | +0.94% | +0.44% | 51% |
| 2021-2026 | -1.77% | +2.50% | -3.49% | 37% |

The pattern is not random and it is not decay — it is a mechanical property of
trend following. **The Hull filter adds value exactly when buy-and-hold returns
are near zero and destroys value in strong bull markets.** 2001-2008 and
2015-2020 are the two windows where sitting out cost nothing; 2009-2014, when
IDX large caps compounded at 31.6%, is where a rule that is in the market half
the time gave up two thirds of the move.

That is a real and honest description of what the indicator does. It is a
volatility-avoidance tool, not a return-generation tool, and Indonesian blue
chips have spent most of the last 25 years rewarding exposure rather than
timing.

## Result 25 — strip out survivorship and the benchmark still wins

Every cohort above is *today's* index membership, so buy-and-hold is measured
only on the names that survived. Re-running on all 413 IDX names with 10+ years
of history instead of the winners' list:

| | curated 84 | broad 413 |
|---|---|---|
| buy-and-hold median CAGR | +11.76% | **+4.94%** |
| confluence median CAGR | -1.02% | -2.77% |
| excess | -10.25% | **-9.34%** |

Survivorship was worth about 6.8 points a year to the benchmark — a large
effect, and confirmation that the curated result was flattering buy-and-hold.
**The strategy still loses by 9.3 points.**

And here the picture finally turns, in the one place the method is supposed to
earn its keep. Splitting the 413 names by what buy-and-hold actually did:

| what buy-and-hold did | names | strategy | buy & hold | excess | beat B&H |
|---|---|---|---|---|---|
| **fell > 10%/yr** | **20** | **-1.53%** | **-13.92%** | **+12.94%** | **90%** |
| fell 0-10%/yr | 97 | -6.62% | -3.85% | -2.41% | 41% |
| gained 0-15%/yr | 226 | -2.78% | +6.31% | -10.18% | 12% |
| gained > 15%/yr | 70 | +0.13% | +19.11% | **-19.30%** | **0%** |

**The insurance works, and it is expensive.** In the 20 names that genuinely
collapsed, the rule saved 12.9 points a year and beat buy-and-hold in 18 of
them. That is not noise and it is exactly what a trailing stop is for. But
those 20 names are **5% of the exchange**, you cannot know in advance which 5%,
and the premium for that cover is 10 to 19 points a year on the other 95% —
including **zero wins out of 70** among the stocks that compounded above 15%.

A coarser bucketing of this table showed the falling cohort as a coin flip,
which was wrong: merging a -13.9%/yr collapse with a -3.9%/yr drift hid the
one thing the method does well. Worth stating plainly, because the corrected
version is the strongest argument *for* the indicator anywhere in this study.

### The conglomerate cohort makes the mechanism concrete

**27% of conglomerate-controlled names beat buy-and-hold, against 4% of blue
chips** — not because the rule works better on them, but because that cohort
contains more value destroyers for it to protect against.

| helped most | B&H CAGR | excess | | hurt most | B&H CAGR | excess |
|---|---|---|---|---|---|---|
| ACST (Astra) | **-18.8%** | +17.7% | | AMRT (Alfamart) | +23.9% | -30.4% |
| ELTY (Bakrie) | **-6.4%** | +16.0% | | BBCA (Djarum) | +20.9% | -25.0% |
| TKIM (Sinar Mas) | +11.5% | +11.7% | | MDKA (Merdeka) | +18.8% | -24.9% |
| LPPF (Lippo) | **-13.7%** | +11.4% | | UNTR (Astra) | +21.3% | -24.0% |
| INDY (Indika) | +1.8% | +10.0% | | MIDI (Alfamart) | +14.8% | -23.1% |

Four of the five names it helped were falling; all five it hurt were
compounding at 15-24% a year. The rule did not distinguish between a Bakrie
property vehicle and BBCA — it simply cut both, which rescued one and ruined
the other. **The indicator has no view on business quality, and on IDX that is
the only variable that mattered.**

## Result 26 — 240 configurations, none of them beat buy-and-hold

A grid over hull length (21/34/55/89/144), hull mode (HMA/EHMA/THMA), UT key
(0.5/1/2/3) and ATR period (5/10/14/21) — 240 configurations, scored **in
sample over the whole 25 years**, which is the most generous test that can be
constructed. It is hindsight with no holdout, and it still fails:

| | median CAGR | buy & hold | excess |
|---|---|---|---|
| best of 240 (`ehma55, key 3.0, ATR 10`) | +3.99% | +11.76% | **-6.10%** |
| worst of 240 (`hma144, key 0.5, ATR 14`) | -4.74% | +11.76% | -16.32% |

**Configurations beating buy-and-hold at the median: 0 of 240.**

The surface is not noise, and its shape is the most informative thing in this
study. Every one of the top ten runs `ut_key = 3.0`, the widest stop tested.
Every one of the bottom five runs `ut_key = 0.5`, the tightest. Within the grid
the ordering is monotone: **the less the UT Bot is allowed to do, the better
the system performs.**

That invites an obvious extrapolation — widen the stop until it never fires and
you should recover Hull alone — and it is wrong. Pushing the key past the grid:

| UT key | excess | trades |
|---|---|---|
| 1.0 | -9.96% | 132 |
| **3.0** | **-6.10%** | 40 |
| 5.0 | -7.54% | 30 |
| 8.0 | -8.55% | 16 |
| 40.0 | -11.44% | 2 |
| **Hull alone, no UT at all** | **-2.62%** | 114 |

The curve turns around at 3.0. Widening the stop does not gradually remove the
UT Bot, because the damage is not the stop at all — it is the **entry**
requirement that price must *cross* the stop line. Widen the stop far enough
and crossings become vanishingly rare: at key 40 the rule takes two trades in
twenty-five years and simply never participates. So key 3.0 is an interior
compromise between "too many bad UT entries" and "no entries at all", and no
setting of it recovers what Hull alone gets with 114 trades.

The conclusion survives the corrected mechanism and is in fact stronger: **Hull
alone beats every one of the 240 confluence configurations**, and the way to
improve the pair is not to tune the UT Bot but to delete it.

Marginalising the grid over each parameter separates a component that matters
from one that does not:

| UT key (stop width) | median excess | | ATR period | median excess |
|---|---|---|---|---|
| 0.5 (tightest) | **-14.61%** | | 5 | -10.55% |
| 1.0 (published) | -10.99% | | 10 | -10.31% |
| 2.0 | -8.53% | | 14 | -10.41% |
| 3.0 (widest) | **-7.88%** | | 21 | -10.78% |

| Hull length | median excess | | Hull mode | median excess |
|---|---|---|---|---|
| 21 | -12.38% | | EHMA | -9.96% |
| **55 (published)** | **-9.25%** | | HMA | -10.66% |
| 89 | -9.92% | | THMA | -11.36% |
| 144 | -10.01% | | | |

Three of the four parameters are flat — ATR period spans 0.5 points across its
whole range, hull mode 1.4. **Only the stop width has a gradient, and it is
monotone: every widening of the stop improves the result.** Meanwhile the
published Hull length of 55 turns out to be the best of the five tested, so the
Hull half was already well chosen by whoever picked it.

So the optimiser, given complete hindsight over 240 configurations, spends its
freedom doing one thing: turning the UT Bot down as far as the grid permits.
And the whole surface, from -16.32% to -6.10%, sits underwater — there is no
spike to mistake for a discovery and no plateau to believe in.

## Result 27 — walk-forward optimisation helps, replicably, and still never reaches zero

Parameters chosen on an expanding in-sample window, then applied to the
untouched slice that follows, with the fixed published default scored on the
same out-of-sample window so the comparison answers the only question that
matters: *did searching beat leaving it alone?*

| fold | test window | chosen | in-sample | out-of-sample | fixed default | value added |
|---|---|---|---|---|---|---|
| 1 | 2008-2012 | ehma89, key 3.0, ATR 10 | -15.10% | -9.66% | -16.89% | **+7.23%** |
| 2 | 2012-2015 | ehma89, key 3.0, ATR 14 | -11.46% | -2.71% | -6.97% | **+4.26%** |
| 3 | 2015-2019 | hma144, key 3.0, ATR 14 | -8.46% | -7.57% | -10.86% | **+3.29%** |
| 4 | 2019-2022 | ehma55, key 3.0, ATR 10 | -6.92% | -3.90% | -6.66% | **+2.75%** |
| 5 | 2022-2026 | ehma55, key 3.0, ATR 14 | -6.26% | -2.19% | -5.62% | **+3.43%** |

| | |
|---|---|
| mean out-of-sample excess, optimised | **-5.21%** |
| mean out-of-sample excess, fixed default | **-9.40%** |
| **value added by optimising** | **+4.19%** |

**Optimisation genuinely worked, in 5 folds out of 5.** That deserves saying
plainly, and it is the opposite of what Part II found for the composite score,
where walk-forward selection actively destroyed value. It is also not a fluke
of one lucky fold: the value added is positive in every window, ranging from
+2.75% to +7.23%.

But look at what it selected. **Every fold picked `ut_key = 3.0`** — the widest
stop available — and the hull settings it chose drifted freely between ehma89,
hma144 and ehma55 without much affecting the outcome. The optimiser was not
discovering a configuration. It was rediscovering, independently and out of
sample five times running, the single finding of Result 26: turn the UT Bot
down. It found the off switch, not an edge.

And having found it, the best out-of-sample result is still **-5.21%** — better
than the -9.40% you get by leaving the defaults alone, and still four to five
points a year behind simply owning the stock.

One number needs care: out-of-sample scored *better* than in-sample in every
fold (+4.43% on average). That is not the strategy improving. Every expanding
training window contains 2009-2014, the era that punished this rule hardest
(Result 24), while the test slices all sit after it. It is a regime artefact of
expanding windows, and reading it as evidence of robustness would be a mistake.

## Result 28 — why it loses, in three numbers

The results above say *that* it loses. This says *why*, and the arithmetic
closes almost exactly, which means there is no fourth explanation hiding.

**1. The signal barely distinguishes good days from bad ones.** Take each
stock's own daily total return and split it by whether the rule was holding
that day:

| | |
|---|---|
| median daily return **while held** | **+9.32 bp** |
| median daily return **while flat** | **+8.47 bp** |
| median daily return, all days | +9.18 bp |
| names where held-days beat flat-days | **56%** |

The days it chose to own the stock were better than the days it sat out by
**0.85 basis points**, and it got the sign right on 56% of names — a coin flip
with a slight lean. The signal is not inverted, which would at least be
exploitable backwards. It is close to uninformative.

**2. The loss is almost entirely the cost of being absent.** For the median
name, annualised:

| | |
|---|---|
| buy-and-hold | **+11.76%** |
| x time in market (18%) | +2.08% |
| forgone drift from sitting out 82% of the time | **-9.68%** |
| trading costs (6.0 trades/yr x 0.60%) | **-3.60%** |
| **predicted** | **-1.53%** |
| **actual** | **-1.02%** |
| residual, i.e. all the timing skill there is | **+0.51%** |

Two terms explain the entire result. IDX large caps drifted up at 11.76% a
year; a rule that owns them 18% of the time forgoes roughly 9.7 points of that
before it does anything at all, then pays 3.6 more in commissions. Timing skill
contributes **+0.51%/yr** — real, positive, and an order of magnitude too small
to pay for the seat.

**3. The exit fires long before the trend it is filtering for.** Across 10,591
trades:

| | |
|---|---|
| median holding period | **5 bars** |
| trades lasting <= 5 bars | **51%**, averaging **-2.87%** |
| trades lasting > 20 bars | 4%, averaging **+20.59%** |
| win rate / avg win / avg loss | 35% / +9.69% / -4.12% |

A Hull 55 band is looking for a trend measured in months. A 1-ATR(10) trailing
stop exits on an ordinary pullback inside a healthy one. So half the trades are
killed inside a week at a mean loss, and the 4% allowed to run return +20.6%.
The per-trade expectancy is positive (+0.70% after costs) — the trades are
fine. There are simply only six of them a year, and six times 0.70% does not
approach 11.76%.

**The general lesson, which is not about this indicator.** In a market with
strong positive drift, *time out of the market is the dominant cost of any
timing rule*, and it is charged before the signal is even consulted. To beat
buy-and-hold from 18% exposure, a signal must be right by roughly 9.7 points a
year. This one is right by 0.5. That is why Result 24 found it working only in
eras when buy-and-hold returned nothing — those are the windows where being
absent was free.

## What to do with this

The rule is not broken and it is not a scam. It does what a trailing stop plus
a slope filter does: it cuts catastrophic losses and it pays for that by
missing compounding. On Indonesian large caps over 25 years, the second effect
is several times larger than the first.

**If you want to use it, the defensible uses are narrow:**

* **Hull alone, not the pair.** This is the single most useful thing in Part
  VII. The UT Bot subtracts value at every horizon and in every test: it turns
  Hull's -2.6% excess into -10.3%, it is what the in-sample grid spends its
  freedom suppressing, and it is what all five walk-forward folds independently
  chose to turn down. The confluence is worse than half of it.
* **As a risk overlay on a position you already hold**, not as a signal to
  build one. The `fell > 10%/yr` row is the whole case for it, and it is a real
  case: 90% of genuine collapses were improved.
* **Never on the compounders.** Zero of 70 names growing above 15%/yr were
  improved by it, and the damage there runs to 19 points a year.

**Do tune it, and do not expect tuning to save it.** Walk-forward selection
added +4.19%/yr out of sample and did so in 5 folds out of 5 — a real,
replicated improvement, and notably the opposite of what happened to the
composite score in Part II. It still finishes 5.2 points a year behind owning
the stock. Both halves of that sentence are true and neither should be dropped.

**What would change the conclusion:** a filter that identifies in advance which
names are in the 5% that collapse. That is a different research problem, and
nothing in this repo solves it.

```bash
idxbot hullut BBCA                          # one name, against its own buy-and-hold
idxbot hullut --universe bluechip           # the cohort table
python3 scripts/hullut_study.py baseline costs eras grid walk broad
```

---

# Part VIII — What actually works on IDX

Part VII ended with a mechanism, not just a verdict: **in a market that drifts
up, time out of the market is charged before the signal is consulted.** Every
design decision below follows from that, and the results are the strongest in
this repository.

## Result 29 — the core engine: pick stocks, never pick moments

Rank liquid IDX names *within each date*, own the top few, hold, repeat. Always
invested. No take-profit, because Part III measured that a +5% cap turns a
+9.27% hold into +0.02%. Non-overlapping rebalances. Rp5bn/day turnover floor,
computed point-in-time. 0.6% round trip per rebalance.

**Walk-forward, configuration re-selected from scratch in every fold:**

| fold | chosen | in-sample | out-of-sample | equal-weight | excess |
|---|---|---|---|---|---|
| 1 | rel_strength 20d top3 | +36.6% | +25.1% | +9.4% | +15.7% |
| 2 | rel_strength 20d top3 | +25.4% | +11.9% | +5.6% | +6.3% |
| 3 | momentum 20d top5 | +22.4% | +9.6% | **-11.4%** | +21.0% |
| 4 | momentum 20d top3 | +24.9% | +51.6% | +19.3% | +32.3% |
| 5 | momentum 20d top3 | +33.5% | +60.2% | +3.4% | +56.7% |

| | |
|---|---|
| mean out-of-sample CAGR | **+31.67%** |
| mean equal-weight, same windows | +5.27% |
| **mean excess** | **+26.40%** |
| folds beating equal-weight | **5 of 5** |
| in-sample to out-of-sample decay | only -3.12% |

Fold 3 is the one to look at: the universe *lost 11.4%* and the book still made
+9.6%. Every fold independently converged on a 20-day horizon and 3-5 names.

A single split agrees. Selecting on 2001-2021 and applying to the untouched
2021-2026: **+23.5% against an equal-weight universe that returned -2.0%.**
Across all 360 configurations the train-to-holdout CAGR rank correlation is
**+0.545** — the ranking genuinely transfers.

The mirror image confirms the mechanism rather than the fit: the `contrarian`
weighting returns **-35% CAGR**, exactly as Part I's inverted-accumulation
result predicts. This is one signal read forwards and backwards, not a search.

## Result 30 — concentration has a floor, and it is not one or two names

Holdout CAGR by book size, same weights and horizon throughout:

| names held | 1 | 2 | **3** | 4 | 5 | 8 | 10 | 15 | 20 | 30 |
|---|---|---|---|---|---|---|---|---|---|---|
| holdout CAGR | **-30.7%** | **-10.5%** | **+23.5%** | +18.9% | +12.4% | +16.3% | +14.6% | +10.6% | +2.8% | +0.2% |

Below three names idiosyncratic risk swamps the signal and the book blows up;
above fifteen the selection is diluted back to the index. **The usable region is
3 to 10 names**, and it is a plateau rather than a point — which matters,
because top-3 being the exact maximum on both halves is partly luck.

## Result 31 — what preceded an IDX 3-bagger

Every liquid IDX name since 2001, labelled by whether it returned +200% over the
following three years, bucketed by trailing features ranked within each date:

| feature | bottom quintile 3x rate | top quintile | direction that wins |
|---|---|---|---|
| **share price** | **11.2%** | 2.8% | **cheap**, 4.0x lift |
| **turnover (size)** | **9.0%** | 3.1% | **small**, 2.9x lift |
| drawdown from all-time high | **9.3%** | 4.9% | **beaten down**, 1.9x |
| distance below 3-year high | **10.1%** | 5.5% | **far below**, 1.8x |
| 12-month momentum | **8.6%** | 6.6% | **low** momentum, 1.3x |
| realised volatility | 4.8% | **8.0%** | high, 1.7x |
| volume vs 1-year normal | 5.3% | **9.0%** | surging, 1.7x |

**The multibagger profile is the exact inverse of the blue-chip profile.**
Momentum selection wants the winners near their highs; 3-baggers came from the
cheap, small, beaten-down names with *low* momentum. That is why one score over
one universe cannot serve both objectives, and why a two-sleeve book can.

## Result 32 — lifting the odds of a 3x is not the same as raising returns

Volatility and volume-surge were the two features that looked best on 3x
*probability*. Both **cost money** when actually traded:

| factor set | sleeve CAGR | change |
|---|---|---|
| price alone | **+23.4%** | +8.9% |
| all five factors | +14.6% | — |
| without volume-surge | +20.1% | **+5.6%** |
| without volatility | +18.3% | **+3.7%** |
| volume-surge alone | +7.9% | -6.7% |
| volatility alone | **+2.7%** | -11.9% |

Not a contradiction. **Volatility raises the chance of a 3x and raises the
chance of a wipeout by more.** A screen built from a probability table without
checking the return table buys lottery tickets above fair value. Both factors
are excluded, and kept in the code as a named `REJECTED_FACTORS` so the reason
survives.

## Result 33 — the 50/50 book: same return, half the bad years

Blue-chip sleeve: momentum, 60-day hold, 5 names, always invested. Holdout
+23.8% against +5.6% equal-weight, and **97% of 30 configurations had positive
holdout excess** — the family works even though the specific ordering does not
transfer (rank correlation +0.075), so take a robust member, not the argmax.

Multibagger sleeve: cheap, small, far below its old high, held three years,
laddered a third at a time so capital is not committed on one date.

| allocation | CAGR | growth | worst year | max drawdown |
|---|---|---|---|---|
| 100% blue chip | **+18.7%** | 13.2x | **-27%** | -27% |
| **50/50, rebalanced yearly** | **+18.3%** | 12.4x | **-14%** | **-21%** |
| 50/50, never rebalanced | +17.4% | 11.1x | -15% | -21% |
| 100% multibagger | +15.9% | 9.1x | -7% | -19% |

**The 50/50 gives up 0.4 points of CAGR and halves the worst year.** The sleeves
are weakly correlated because they are driven by opposite factors, so the
constant-mix rebalance systematically sells whichever ran. And the allocation is
insensitive — every split from 30/70 to 70/30 lands between +17.6% and +18.7%,
so the choice of exactly 50 is not load-bearing.

## Result 34 — the limit that no amount of work removes

**The multibagger sleeve rests on six independent observations.** Six
non-overlapping three-year windows exist in twenty-one years of IDX history:
2004, 2008, 2011, 2015, 2018, 2022. That is a property of the calendar, not of
the screen, and **no three-year strategy on IDX can be validated on IDX data.**

Two further things must be said about that sleeve:

* Its picks have a median share price of **Rp139**. Mean three-year return
  +94.5% against +50.4% for owning everything — but the **median pick returns
  +7.1%**. The mean is a handful of names; the typical holding does nothing.
* **Survivorship falls almost entirely here.** "Cheap, small, beaten-down" is
  also the exact profile of a company about to delist. The winners are all in
  the panel and an unknown share of the losers are not, so 14% of picks losing
  more than half is a floor, not an estimate.

The blue-chip sleeve carries none of that weakness: 77 rebalances, five
walk-forward folds, large liquid names that rarely vanish. **Treat the two
halves as carrying very different evidential weight, because they do.**

## What to actually run

```bash
python3 scripts/multibagger_study.py     # rebuild the 3-year pattern panel
idxbot book                              # today's picks for both sleeves
```

| | evidence | expectation |
|---|---|---|
| blue-chip sleeve | walk-forward, 5 folds, 5/5 positive | strong |
| concentration 3-10 names | plateau on both halves | strong |
| always invested, no take-profit | Result 28 + Part III | strong |
| multibagger sleeve | 6 windows, survivorship-exposed | **weak** |
| the 50/50 split itself | insensitive across 30-70% | moderate |

**The honest headline: the blue-chip engine is the result. The multibagger
sleeve is a defensible way to spend half the book on a lottery whose ticket
price the data cannot confirm.** Every absolute figure above is inflated by
survivorship; the *excess over equal-weight* is the number that survives it.

---

# Part IX — Timing ADRO, dividends, and state ownership

## Result 35 — the ceiling on ADRO, and why "buy the low, sell the peak" misleads

Owning ADRO from 2008 to 2026 returns **7.77x** (+12.0% a year, dividends in).
With perfect foresight:

| trades per year (perfect hindsight) | total | CAGR |
|---|---|---|
| 1 | **198,000x** | +96.3% |
| 2 | 8.05 million x | +141.0% |
| 4 | 1.17 billion x | +217.3% |
| every up day, flat every down day | 8.4 x 10^19 | — |

**The ceiling is not the constraint.** One perfectly-timed round trip a year is
already a 198,000x. Any number you like can be produced by a rule that sees even
slightly into the future, which is exactly why "it multiplied capital N times by
buying lows and selling peaks" is not a claim that can be evaluated without the
timestamps.

What was actually there, without hindsight, is smaller than it sounds. ADRO's
**largest single trough-to-peak run in eighteen years was 2.05x**, and chaining
all seven of its major runs perfectly gives **29x** — an order of magnitude
below the DP ceiling, because the ceiling is built from many small perfectly
timed moves, not a few obvious ones.

## Result 36 — no causal rule beat holding ADRO

112 configurations across six families (Donchian breakout, MA cross, RSI
reversion, z-score reversion, dip-buying, momentum), train 2008-2019, holdout
2019-2026:

| | |
|---|---|
| buy & hold, train | **+0.5%** |
| best rule, train | **+15.8%** |
| buy & hold, holdout | +31.6% |
| best rule, holdout | +30.5% |
| configs beating buy & hold on holdout | **1% of 112** |

Walk-forward, five folds, re-selected each time: **mean OOS +16.7% against
+23.4% for buy-and-hold, 1 fold of 5 ahead.**

The train column is the interesting one. Over 2008-2019 ADRO went **nowhere**
(+0.5% a year) and timing rules returned +15.8%. Over 2019-2026 it compounded at
31.6% and nothing beat it. This is Result 28 again, on a single name: timing pays
when the asset does not trend, and costs when it does. You cannot know which
regime you are in.

**Shorting makes it worse, not better.** Median holdout CAGR with shorts enabled
is -2.5% against +18.4% long-only, and -48.6% at 3x leverage. **Leverage is not
a fix either**: 2x on the best rule reaches +49.3% on holdout with a **-91%
drawdown in training** - a margin call, not a return.

**Commodity prices do not lead the miner.** Brent, WTI and US coal equities are
strongly correlated with ADRO *contemporaneously* (0.26 to 0.44) and carry
essentially no predictive lead (Brent +0.008, WTI -0.007, Peabody -0.047 against
ADRO's next 20 days). The commodity moves with the stock, so that information is
already in the price.

## Result 37 — dividends are not a strategy on IDX

1,209 ex-dividend events across 54 blue chips, median yield 1.94% per event,
0.86 events per name per year, 69% falling in April-July.

**Dividend capture.** Buying ten days before the ex-date and selling ten after
returns +3.46% gross. Against matched controls in the same stocks and the same
calendar months it returns **+1.37%, t = +1.49** — which does not clear the 0.6%
round trip with any confidence. The headline number was mostly market drift over
twenty days plus the seasonal clustering.

**Dividend yield as a factor.** Rank IC against forward one-year returns is
**+0.041, t = +1.04** over 22 annual cross-sections. Not significant. The
*zero-yield* quintile posted the **highest** mean forward return (+50.6%),
because zero-yield names on IDX are the speculative growth tail.

Neither is in the shipped book.

## Result 38 — state ownership is the best multibagger factor found

The "backed by the government" idea, tested:

| | 3x rate | median 3-year | mean 3-year |
|---|---|---|---|
| private | 6.2% | **-10.5%** | +26.7% |
| state-owned (BUMN) | 7.8% | **+9.7%** | +46.9% |

And it concentrates exactly where the rest of the multibagger screen already
points — the cheap end:

| price tercile | state-owned 3x | private 3x | state mean 3y | private mean 3y |
|---|---|---|---|---|
| **cheapest** | **24.1%** | 9.8% | **+146.9%** | +39.8% |
| middle | 8.1% | 5.0% | +45.6% | +32.1% |
| dearest | 1.5% | 3.4% | +9.2% | +4.8% |

**Cheap AND state-backed is the pattern; state-backed alone is not.** An
expensive BUMN is the worst cell in the table. The mechanism is not mysterious:
a beaten-down state issuer gets recapitalised, a beaten-down private microcap
delists — which also makes this the one corner of the multibagger sleeve where
survivorship bias is mild, because these names do not vanish.

Adding it lifts the sleeve from +22.4% to **+24.9%** CAGR. Caveat that matters:
the sleeve still rests on **six** independent windows, and improving a
six-observation backtest by adding a factor is precisely what overfitting looks
like. The *cross-sectional* evidence above (n=79 state-owned observations in the
cheap tercile against 884 private) is the stronger half of the case.

## Result 39 — the book, and what 100x actually requires

| allocation | CAGR | growth | worst year | max drawdown |
|---|---|---|---|---|
| 100% blue chip | +18.7% | 13.2x | -27% | -27% |
| **50/50 rebalanced yearly** | **+19.5%** | **14.5x** | **-15%** | **-23%** |
| 30/70 | +19.3% | 14.0x | -12% | -24% |
| 100% multibagger | +18.2% | 12.2x | -9% | -25% |

Adding state ownership took the 50/50 from +18.3% to **+19.5%** while cutting the
worst year from -27% to -15%. The split remains insensitive across 30/70 to
70/30, so the exact 50 is still not load-bearing.

**On the 100x question**, compounded over ADRO's own eighteen-year lifetime:

| | CAGR | 18-year multiple |
|---|---|---|
| ADRO buy & hold | +12.0% | 8x |
| ADRO best timing rule, walk-forward | +16.7% | 16x |
| 50/50 book | +19.5% | **25x** |
| blue-chip sleeve alone | +21.2% | 32x |
| **cross-sectional book, walk-forward OOS** | **+31.7%** | **145x** |

The hundredfold is reachable, and **not by timing ADRO**. It comes from the one
thing in this repository validated in five out of five walk-forward folds:
rotating cross-sectionally into whatever is strongest, always invested, never
predicting a turning point. Timing a single name — even a violently cyclical one
with a 198,000x hindsight ceiling — did not beat holding it.

---

# Part X — Solving for the perfect trades, and what killed the answer

This part chased one question as far as the data allows: **can you buy the low
and sell the peak?** The method was to stop guessing rules and instead solve for
the optimal trades, look at what they looked like beforehand, and build from
that. It ends with a large result being deleted, which is the most useful thing
in it.

## Result 40 — the optimal trades are at extremes, and they are held for a year

Dynamic programming over ADRO's whole history, 18 round trips (one a year),
returns **198,161x**. Reconstructing *which* trades those were and measuring the
features at each one:

| feature | at optimal BUYS | at optimal SELLS | separation |
|---|---|---|---|
| RSI(14) | 26.6 — **3rd percentile** | 66.3 — **90th percentile** | 86% |
| z-score(60) | -2.04 — 5th pct | +2.14 — 92nd pct | **87%** |
| drawdown from 1y high | **-49%** — 10th pct | -1% — 95th pct | 85% |
| 12-month momentum | -37% | +36% | 61% |

**The turning points are exactly where the folklore says they are.** Every large
winner was bought at a deep drawdown and held **220 to 550 days**:

| | bought | sold | gain | days |
|---|---|---|---|---|
| best | 2008-11 at 88 | 2010-04 at 430 | **+386%** | 514 |
| 2nd | 2016-01 at 100 | 2017-04 at 459 | **+359%** | 440 |
| 3rd | 2021-04 at 346 | 2022-06 at 1291 | **+273%** | 420 |

This explains why every mean-reversion test in Part IX failed. Those rules
exited at z=0 — **selling the first bounce**, three months into a move that ran
for a year and a half. Re-testing with wide bands and long holds on ADRO gives
**30.0x against 7.8x for buy-and-hold**, and the best rule captures 27.9% of the
theoretical log-ceiling against buy-and-hold's 16.8%.

So the shape of the answer is real: *enter deep, exit at a new high, hold through
the middle*. The next section is what happened when it was tested properly.

## Result 41 — the 195x was data corruption, and finding it is the point

Applied across all 654 liquid IDX names from 2000 as a portfolio — buy anything
more than 60% below its one-year high, sell when it makes a new high — the first
run returned **195.0x (+22.1% CAGR)**, against an equal-weight universe that
appeared to return 34,589,778x.

**A 34-million-times benchmark is not a benchmark, it is a bug.** IDX enforces
auto-rejection limits of roughly 20-35% a session, so no daily return can exceed
that. Checking every daily return in the panel:

| | |
|---|---|
| moves beyond +/-35% | 1,080 of 2,471,418 days (**0.044%**) |
| of those, in illiquid names (<Rp1bn/day) | **91%** |
| largest single-day "return" recorded | **KOPI, +10,915,284%**, on zero turnover |

Capping daily returns at the physically possible +/-35% and applying the same
liquidity filter to the benchmark:

| | before the fix | after |
|---|---|---|
| deep-value rule | **195.0x** | **4.4x** |
| equal-weight universe | 34,589,778x | 5.9x |
| verdict | "3.9x better" | **loses by 1.1%/yr** |

**The entire result was a few dozen corrupt prints compounding in a
daily-rebalanced book.** Roughly one day in 2,300 was impossible, and that was
enough to manufacture a 195x. This is the mechanism behind a very large share of
spectacular backtest claims, and it does not require any dishonesty — just an
uncapped `pct_change()` on an unfiltered universe.

## Result 42 — the transactions, and why stops do not save it

977 transactions, median hold 196 days:

| | |
|---|---|
| win rate | 48% |
| **median trade** | **-0.6%** |
| mean trade | +12.1% |
| trades losing more than half | 13% |
| worst | SLIS -99% held 1,436 days; SULI -95% held **3,039 days** |

The distribution is the whole story: 53% of trades lose money, and the 21% that
gain more than 50% contribute +228 against a total of +118. **The median deep
value trade is dead money.**

Entries cluster in crises exactly as the design implies — 50 in 2008, 83 in
2020, 71 in 2022 — which is the strategy working as intended, not a flaw.

The obvious fix fails. Cutting the losers that sit for years makes it **worse**:

| variant | growth | CAGR |
|---|---|---|
| plain: exit only at a new high | 4.4x | +5.81% |
| + 500-day time stop | 2.8x | +3.99% |
| + 250-day time stop | 0.4x | **-3.87%** |
| + 30% stop-loss | 0.8x | -0.83% |

Stops cut the positions that were going to recover. In a strategy whose entire
return lives in the right tail, anything that truncates the tail destroys it —
the same lesson as Part III's take-profit result, arriving from the other side.

And survivorship sits on top of all of it: a stock 60% below its high either
recovers or delists, and only the survivors are in the panel. Every figure above
is an upper bound.

## Result 43 — the cross-sectional book survives the same audit

Having deleted one result for outlier contamination, the honest next step was to
put the headline result through the identical test. Re-running the five-fold
walk-forward with forward returns capped:

| treatment | mean OOS | equal-weight | excess | folds positive |
|---|---|---|---|---|
| as reported | +31.7% | +5.3% | +26.4% | **5/5** |
| capped at +300% | +31.7% | +5.3% | +26.4% | **5/5** |
| capped at +100% | +27.6% | +4.5% | +23.2% | **5/5** |
| **winsorised at the 99th percentile** | **+19.9%** | +3.2% | **+16.8%** | **4/5** |

**It survives, and the honest number is a range rather than the headline.**
Capping at +300% changes nothing at all. Capping at +100% — far below what a
real 20-day move can be — still leaves +23.2% excess with five folds of five.
Clipping the top 1% of every forward return, which is harsher than anything a
real book would experience, drops it to +16.8% excess with four folds of five.

So roughly **a third of the excess lives in the right tail**, and two thirds do
not. That is worth stating plainly, because it changes what to expect: over
eighteen years the book compounds to **145x at the reported rate and about 26x
under the harshest winsorisation.** Both are far above owning the universe;
neither is the same number, and quoting only the first would be the same
mistake this part just caught elsewhere.

The deep-value result did not survive this test. This one did, and now that has
been checked rather than assumed.

## What this part settles

* **Buying the low and selling the peak is the right shape.** The optimal trades
  really do sit at the 3rd and 90th percentiles of ordinary indicators, and the
  winners are held for a year or more.
* **Identifying those points in advance is the part that does not work.** Deep
  drawdowns are common; deep drawdowns that recover to a new high are not, and
  nothing tested separates them ex ante. Across all of IDX the rule returns 4.4x
  against 5.9x for owning the universe.
* **A spectacular backtest number is a bug until proven otherwise.** The
  distance between 195x and 4.4x was one line capping returns at what the
  exchange physically permits. The same audit applied to the cross-sectional
  book left it standing at +16.8% to +26.4% excess depending on how hard the
  tail is clipped — which is what a real result looks like when you attack it.

---

# Part XI — Exhausting the search on ADRO

Parts IX and X kept finding the same answer from new directions. This part
records the last approaches tried, so that the negative is a *searched* negative
rather than an assumed one.

## Result 44 — the perfect-trade rule fails walk-forward too

Part X derived the rule directly from the optimal solution: enter at a 60%
drawdown, exit at a new high, hold through the middle. In-sample on ADRO that is
30.0x against 7.8x. Walk-forward, with the parameters re-chosen in each fold:

| fold | test to | chosen | OOS | buy & hold | excess |
|---|---|---|---|---|---|
| 1 | 2016-12 | dd250 -60%/0% | +28.3% | +23.3% | +5.0% |
| 2 | 2019-05 | dd250 -60%/0% | 0.0% | -5.8% | +5.8% |
| 3 | 2021-10 | dd250 -60%/0% | +56.3% | +28.6% | +27.7% |
| 4 | 2024-03 | dd250 -60%/0% | **0.0%** | +35.3% | **-35.3%** |
| 5 | 2026-08 | dd250 -60%/0% | **0.0%** | +35.8% | **-35.8%** |
| | | | **mean +16.9%** | **+23.4%** | **-6.5%** |

Folds 4 and 5 read 0.0% because **the rule never fired** — ADRO never fell 60%
below its high in those windows, so it sat in cash while the stock returned
+35% a year. That is the defining weakness of waiting for a deep low: the
best years are the ones where the low never comes.

## Result 45 — sector rotation works in exactly one sector, which is how you know it doesn't

If timing one name fails, rotating among related names is the natural next step.
Momentum rotation, top 3 of each sector, liquidity-filtered, returns capped:

| sector | names | rotate top 3 | hold the sector | verdict |
|---|---|---|---|---|
| **coal / energy** | 10 | **168.0x** (+22.6%) | 40.7x (+15.9%) | looks great |
| banks | 12 | 1.9x (+2.6%) | 41.5x (+16.8%) | destroys value |
| metals | 9 | 7.4x (+8.2%) | 62.8x (+17.7%) | destroys value |
| consumer | 12 | 1.7x (+2.0%) | 81.0x (+18.5%) | destroys value |
| property | 10 | **0.0x** (-16.5%) | 19.1x (+12.1%) | total loss |

**One sector out of five is not a strategy, it is the one that happened to
work.** And walk-forwarding the coal case, re-selecting lookback and breadth in
every fold, removes even that: **mean OOS +6.3% against +16.1% for simply
holding the sector, excess -9.8%, 2 folds of 5** — with one fold at -55.0%.

Rotating inside a narrow sector concentrates rather than diversifies: all ten
coal names rise and fall together, so the rotation adds turnover and timing risk
without adding breadth. The cross-sectional book works precisely because it
ranks across the *whole* market, where there is always something in a different
part of its cycle.

## Result 46 — what has now been ruled out

Every approach tried on the "time ADRO" problem, and how each failed:

| approach | result |
|---|---|
| 112 configs, 6 rule families | 1% beat buy & hold on holdout; walk-forward 1/5 |
| wide-band reversion from the DP solution | 30.0x in-sample, walk-forward **-6.5%** excess |
| gradient boosting / random forest / ridge, 64 features | all six variants lost; **OOS IC -0.086 to -0.144** |
| long/short | -2.5% vs +18.4% long-only |
| 2x-3x leverage | +49.3% holdout at a **-91%** training drawdown |
| commodity price as an exogenous signal | Brent +0.008, WTI -0.007 predictive lead |
| deep value across all 654 IDX names | 4.4x vs 5.9x for owning the universe |
| stops and time stops on deep value | every variant worse; 250d stop **-3.87%** |
| coal sector rotation | walk-forward 2/5, excess **-9.8%** |
| dividend capture | +1.37% over controls, **t = 1.49** |
| dividend yield as a factor | IC +0.041, **t = 1.04** |

**One thing survived every test**: ranking the whole liquid market
cross-sectionally and owning the strongest few, always invested. +16.8% to
+26.4% excess over equal-weight depending on how hard the right tail is clipped,
4 to 5 folds positive out of 5.

## The arithmetic of "100x", stated plainly

A **100x CAGR** means +10,000% a year. Compounded over ADRO's eighteen-year
history that is 10^36 — more than the market capitalisation of the planet by
roughly twenty orders of magnitude. No strategy produces it, and any claim of it
is a units error somewhere.

**100x total return** is a different and entirely reasonable target:

| | required CAGR | status |
|---|---|---|
| 100x over 18 years | +29.4% a year | reachable |
| 100x over 10 years | +58.5% a year | not by anything here |
| 100x over 5 years | +151% a year | no |

The validated cross-sectional book compounds to **145x over eighteen years at
its reported walk-forward rate, and about 26x under the harshest winsorising.**
So a hundredfold is on the table — over a long horizon, from broad
cross-sectional selection, and not from timing the peaks and lows of any single
stock. That distinction is the entire finding of Parts IX through XI.

## Result 47 — intraday closes the last door

The dynamic program says frequency raises the ceiling: 1 trade a year on ADRO is
198,161x, 52 a year is 6.9 x 10^19. If timing works anywhere, it should work
where there are more turning points to catch. Yahoo serves 60 days of 5-minute
bars and 730 of hourly, so this is a small window - 3,541 five-minute bars over
0.22 years - but the ceiling behaves exactly as predicted:

| perfect foresight, 5-minute bars | growth over 80 days |
|---|---|
| 10 trades | 1.9x |
| 50 trades | 5.8x |
| **200 trades** | **37.2x** |

And the causal rules find none of it. Across 57 configurations at each frequency:

| | best rule | buy & hold | configs beating hold |
|---|---|---|---|
| hourly | 1.167x | 1.100x | **5%** |
| 5-minute | 1.100x | 1.091x | **11%** |

The tell is the trade count: every rule that "beat" buy-and-hold did so with
**one or two trades** — it degenerated into buy-and-hold and won by a rounding
error. Nothing found structure at any frequency. Eighty days is far too short to
be conclusive on its own; taken with Results 36 and 44 through 46, it is the
last door closing.

**The pattern across every frequency tested is identical.** More available
turning points raise the theoretical ceiling and do nothing for the achievable
result, because the difficulty was never the number of opportunities. It was
always telling, in advance, which local minimum is *the* minimum.

## Result 48 — how a "100x on ADRO" is manufactured

Rather than only proving the number cannot be earned, this reverse-engineers
where it comes from. Taking the best honest rule on ADRO (Donchian 20/100,
next-bar fills, IDX costs, compounded = **30.2x**) and changing one methodology
choice at a time:

| methodology | result | inflation |
|---|---|---|
| **honest**: next-bar fill, costs, compounded | **30.2x** | 1.0x |
| **same-bar fill** (signal and fill on one bar) | **230.9x** | **7.6x** |
| no fees or slippage | 36.7x | 1.2x |
| sum of all trade returns, not compounded | 7.6x | 0.3x |
| sum of *winning* trades only | 9.2x | 0.3x |
| 3x leverage | 17.7x | 0.6x |
| 5x leverage | **0.0x** | wiped out |
| cherry-picked start date | 30.2x | 1.0x |
| **same-bar + no costs + 3x leverage** | **7,333x** | **243x** |

**One line does it.** Filling at the close of the bar that generated the signal,
instead of the next bar's open, takes 30.2x to 230.9x. It is the single most
common error in backtesting, it requires no dishonesty, and on ADRO it alone
lands squarely in the range being claimed.

What does *not* explain it: summing trade returns instead of compounding
actually makes the number *smaller* (7.6x), leverage beyond 3x wipes the account
out entirely, and ADRO's own start date is already the best available so
cherry-picking gains nothing.

So there are exactly four ways to see a hundredfold on this stock: **same-bar
execution, stacked leverage with no margin model, uncapped corrupt prints, or
hindsight.** The first is by far the most likely, and it is testable in one
question: *does the entry price equal a price that existed on the signal bar, or
on the bar after it?*

### The test to apply to any vendor claim

1. **Ask for timestamped entries and exits.** Signal date and fill date must
   differ. If they are the same bar, the number is a one-line artifact.
2. **Ask what the fill price was.** A fill at the signal bar's close or, worse,
   its low, is hindsight.
3. **Ask for the equity curve, not a list of wins.** Summing winners is not a
   return; on ADRO it is 9.2x against a compounded 30.2x.
4. **Ask whether returns were capped.** 0.044% of IDX daily prints are
   physically impossible and they compound into anything you like.
5. **Ask for the benchmark over the identical window.** ADRO returned +31.6% a
   year over 2019-2026; a strategy showing +30% there underperformed.

Every one of those is a question about *methodology*, not about the market, and
each can be answered in a sentence by anyone who actually ran the backtest.

## Result 49 — every added layer of sophistication made it worse

The final optimisation pass combined three techniques not previously stacked:
ensembling four independent trend rules, volatility-targeted position sizing, and
gating on the validated cross-sectional rank. Walk-forward, five folds:

| variant | mean OOS | buy & hold | excess | folds ahead |
|---|---|---|---|---|
| single best rule (Donchian 20/100) | +19.8% | +23.4% | **-3.6%** | 1/5 |
| ensemble of 4 rules | +14.6% | +23.4% | -8.8% | 1/5 |
| ensemble + volatility targeting | +11.2% | +23.4% | -12.2% | 2/5 |
| ensemble + cross-sectional gate | +7.3% | +23.4% | -16.2% | **0/5** |
| ensemble + vol target + gate | +5.7% | +23.4% | **-17.7%** | **0/5** |

**The ordering is monotone and it points the wrong way.** Every layer added
subtracts about five points a year. The mechanism is Result 28 for the last
time: each technique is a form of risk reduction, risk reduction means less
exposure, and less exposure to an asset compounding at 12% a year costs more
than the timing saves. Ensembling four rules means all four must agree, which
means being in the market less. Volatility targeting cuts exposure exactly when
ADRO is moving, which is when it makes its money. The cross-sectional gate is
the strictest filter of all and produces the worst result.

This is the end of the search. Ranked by how much they were expected to help,
against how much they did:

| | full-sample growth | walk-forward excess |
|---|---|---|
| perfect foresight, 18 trades | **198,161x** | — |
| best single rule | 30.2x | -3.6% |
| ADRO buy & hold | 7.8x | 0 by definition |
| ensemble + vol target + gate | 2.1x | -17.7% |

**Nothing tested on this stock beat holding it out of sample.** The honest
maximum for ADRO-specific capital is the buy-and-hold 7.8x, or roughly 16x from
a single trend rule if you accept that its walk-forward record is a coin flip
(1 fold of 5). The 145x - and the hundredfold the whole exercise was aimed at -
lives in cross-sectional selection across the whole market, and nowhere else
that twelve distinct approaches could find.

## Result 50 — a correction: ADRO's buy-and-hold was understated

The AADI spin-off (ex-date 2024-11-29) is **not** adjusted out of Yahoo's
total-return series for ADRO:

| date | close | adj_close | adj change |
|---|---|---|---|
| 2024-11-28 | 2,760 | 2,148.3 | +19.4% |
| **2024-11-29** | 2,080 | **1,619.0** | **-24.6%** |

That -24.6% is a distribution, not a loss. Holders received AADI shares, which
listed at 6,650 on 2024-12-05 and now trade at 9,425 (**+42%**).

So every ADRO buy-and-hold figure in Parts IX to XI is **too low** — the true
multiple is roughly 10.3x rather than 7.77x before counting AADI's subsequent
gain, and the CAGR closer to +13.6% than +12.0%.

**This makes the conclusion stronger, not weaker.** Every timing rule was
measured against a benchmark that was understated, so each one looks *worse*
relative to a correctly-adjusted hold, not better. The single trend rule's
already-marginal walk-forward record (1 fold of 5) gets marginally worse; the
"deep value fails" and "sector rotation fails" results are unaffected because
they use the same series on both sides.

Worth recording as its own result because it is a reminder that the benchmark
deserves the same scrutiny as the strategy. A backtest that beats a
mis-measured hold has beaten nothing, and the error here ran in the direction
that would have flattered any timing rule reported against it.

## Result 51 — the last idea, and the number that ends the argument

Every ADRO rule tested so far parked in **cash** when out, which Result 28 says
is the single largest cost of timing. So the final test removes that entirely:
sell ADRO at the peak, **redeploy into the market**, buy ADRO back at the low.
Capital never idles.

| ADRO rule | parked in | growth | CAGR | in ADRO |
|---|---|---|---|---|
| ma_cross 20/100 | **LQ45** | **30.39x** | +20.8% | 53% |
| donchian 20/100 | LQ45 | 30.24x | +20.8% | 79% |
| donchian 20/100 | IHSG | 20.83x | +18.3% | 79% |
| dd -50% / new high | LQ45 | 13.41x | +15.4% | 97% |

It works — 30.4x against 7.77x for holding ADRO, nearly 4x better. And then the
baseline that matters:

| | growth | CAGR |
|---|---|---|
| hold ADRO | **7.77x** | +12.0% |
| hold IHSG | 2.86x | +6.0% |
| **hold LQ45 equal-weight, no timing at all** | **24.15x** | **+19.3%** |
| best ADRO timing + redeployment | 30.39x | +20.8% |

**Simply owning the LQ45 basket returned 24.15x.** All the ADRO timing
machinery, with capital perfectly redeployed and parameters chosen in hindsight,
adds **+1.5% a year** over doing nothing at all — an edge far too thin to survive
the walk-forward that every other variant failed.

And there is the number that ends the whole exercise: **ADRO returned 7.77x
while the LQ45 basket returned 24.15x.** ADRO was not a good thing to own. Three
parts of research went into timing the peaks and lows of a stock that
underperformed a passive basket of Indonesian large caps by a factor of three.

**The best thing to do with ADRO capital was never to time it. It was not to
concentrate it in ADRO.** That is the same answer as Result 29, Result 39 and
Result 46, arrived at from the opposite direction — and it is why the
hundredfold lives in cross-sectional breadth and not in any single name.

---

# Part XII — Rp50,000,000, run as a real account

Every number so far has been an index of returns. This runs the validated engine
as an actual account with actual constraints, from the start of the data to
today, and logs every fill.

**Constraints enforced**: whole lots only (1 lot = 100 shares); IDX retail fees
of 0.15% to buy and 0.25% to sell; ranking computed on the rebalance date and
filled at the *next* session's open; dividends collected; daily returns capped
at the +/-35% auto-rejection band; and a **capacity limit** - no position may
exceed 10% of that name's 20-day median turnover.

## Result 52 — two bugs that a returns-index backtest cannot expose

**Missing marks valued at zero.** A name that did not trade on a given day has
no price on that row. Left as NaN it silently valued the holding at **zero**,
producing "losses" of -99.95% in a single day while the underlying stocks moved
-12% to -16%. Prices must be carried forward for valuation; fills still require
a day the stock actually traded. This bug is invisible in a returns index
because there are no positions to mis-mark.

**No capacity limit.** Uncapped, the account compounded into **Rp7.8 billion
positions in stocks trading Rp10 billion a day** - fills that would move the
market by more than the edge being harvested. Adding a 10%-of-turnover
participation cap is what makes a compounding account's returns decay with size,
which is the single most important thing a returns index hides.

## Result 53 — the account, and the sweep

| config | final value | growth | CAGR | max DD | trades |
|---|---|---|---|---|---|
| **top 3, 20-day** | **Rp23,127,511,407** | **462.6x** | +26.2% | **-74%** | 1,382 |
| top 5, 20-day | Rp17,990,489,691 | 359.8x | +25.0% | -66% | 2,244 |
| top 8, 20-day | Rp16,983,111,544 | 339.7x | +24.7% | -61% | 3,374 |
| top 8, 60-day | Rp7,894,409,103 | 157.9x | +21.2% | -69% | 1,291 |
| top 15, 20-day | Rp4,327,386,543 | 86.5x | +18.4% | -69% | 5,721 |
| top 5, 60-day | Rp3,784,653,325 | 75.7x | +17.8% | -82% | 866 |
| top 15, 60-day | Rp2,055,860,732 | 41.1x | +15.1% | -64% | 2,211 |
| top 3, 60-day | Rp1,782,032,645 | 35.6x | +14.5% | -81% | 540 |

The 20-day rebalance dominates the 60-day everywhere. But note the instability:
at 60 days, top-3 returns 35.6x while top-8 returns 157.9x - a 4.4x spread from a
parameter that should not matter that much. **Single-path outcomes are
high-variance**, which is exactly why the walk-forward in Part VIII, not this
table, is the evidence.

## Result 54 — what the account is actually made of

| what you did with Rp50m on 2000-03-30 | value today | growth |
|---|---|---|
| top 3, 20-day rebalance | Rp23.1bn | 462.6x |
| **the strategy (top 5, 20-day)** | **Rp18.0bn** | **359.8x** |
| **equal-weight blue chips, never traded** | **Rp14.8bn** | **295.2x** |
| UNTR alone | Rp7.4bn | 148.3x |
| BBCA alone | Rp3.3bn | 66.9x |
| **IHSG index** | **Rp0.54bn** | **10.9x** |
| ADRO alone | Rp0.39bn | 7.8x |
| 6% bank deposit | Rp0.23bn | 4.7x |

Two comparisons matter and they point in opposite directions.

**Against the passive basket**: equal-weighting blue chips and never trading
returns 295.2x against the strategy's 359.8x. Twenty-six years of ranking,
rebalancing and 2,244 transactions add **+0.7% a year**.

**Against the honest index**: IHSG returned **10.9x**. The blue-chip basket
returned 295.2x on the same data. That gap - a factor of **27** - is
survivorship, and it is not a small correction to the headline. It *is* most of
the headline.

So the 359.8x is arithmetically correct on this panel and mostly composed of
Indonesian equity beta plus the fact that the panel contains only companies that
still exist. The defensible claim is the **+0.7% a year over a passive basket**,
not the multiple - and even that is thinner than the walk-forward excess in Part
VIII because a single path is one draw, not a distribution.

And it required sitting through a **-66% drawdown**, with drawdowns worse than
-35% in **20 of the 26 years**.

## Result 55 — the true maximum on ADRO capital, and the last 6%

Combining every legitimate improvement found across Parts IX-XII - the best
timing rule, dividends reinvested, and redeploying into blue chips rather than
sitting in cash when out of ADRO:

| construction | growth | CAGR | max DD |
|---|---|---|---|
| buy & hold ADRO, dividends reinvested | 7.7x | +12.0% | -82% |
| **best rule, unlevered, redeploying** | **37.1x** | **+22.1%** | **-54%** |
| best rule, 1.5x leverage | 70.9x | +26.6% | -73% |
| **best rule, 2x leverage** | **94.2x** | **+28.6%** | **-84%** |

**94.2x is 94% of the way to a hundredfold** - Rp50,000,000 into
Rp4,709,424,972. And it is not attainable, because the leverage that produces
it does not survive a margin account:

| setup | outcome |
|---|---|
| 1x, unlevered | survived, 37.1x |
| 1.5x with 30% maintenance margin | **LIQUIDATED 2008-10** |
| 2x with 30% maintenance margin | **LIQUIDATED 2008-09** |
| 2x with no margin model | "survived", 65.0x |

Both levered versions are wiped out in the global financial crisis, in the
account's first year. The -84% drawdown is not a bad stretch to sit through on
borrowed money; it is a forced sale at the bottom. Only the version with **no
margin model at all** reaches the headline, which is precisely the kind of
missing constraint that Result 41 and Result 52 were about.

**So the true maximum for ADRO-centric capital is 37.1x: Rp50,000,000 into
Rp1,856,853,026 at +22.1% a year, with a -54% drawdown.** Reaching 100x needs
+29.0% a year. The remaining 6.9 points can only be bought with leverage that
2008 liquidates.

For scale, the perfect-foresight ceiling on the same stock over the same window
is **198,161x**. The distance from 37x to 198,161x is not effort - fourteen
distinct approaches were spent on it - it is the difference between knowing
which low is *the* low and guessing.
