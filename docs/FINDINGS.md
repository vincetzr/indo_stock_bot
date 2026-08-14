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
