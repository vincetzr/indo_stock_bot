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

## Result 56 — volatility targeting cannot buy the leverage either

The last idea for closing the gap from +22.1% to the +29.0% a hundredfold
requires: **volatility-targeted leverage**, which de-levers automatically when
volatility spikes and so should survive the 2008 crash that liquidated fixed
leverage. Margin enforced, 9% borrow cost:

| target vol | max leverage | growth | CAGR | max DD | outcome |
|---|---|---|---|---|---|
| 25% | 1.0x - 3.0x | 32.2-32.3x | +21.2% | -54% | survived |
| 35% | 1.0x - 3.0x | 32.6-33.1x | +21.4% | -54% | survived |
| 45% | 1.0x | 35.6x | +21.8% | -54% | survived |
| **60%** | **1.5x** | **38.1x** | **+22.3%** | -63% | survived |
| 60% | 3.0x | 29.4x | +20.6% | -71% | survived |

**It survives, and it changes nothing**: 38.1x against 37.1x unlevered.

The reason is visible in the table and is a property of the stock, not the
technique. **ADRO's realised volatility runs 50-60%.** A vol target of 25-45%
therefore holds leverage *below* 1.0x almost always, and the strategy
degenerates into the unlevered version. Raising the target to 60% finally
permits leverage - and immediately widens the drawdown from -54% to -63% to
-71% without improving the return.

Volatility targeting works by borrowing against calm assets. ADRO is not a calm
asset, so there is nothing to borrow against.

## The final number

**Rp50,000,000 into ADRO in July 2008, timed as well as fifteen distinct
approaches could manage, becomes Rp1,904,562,224 — 38.1x, +22.3% a year,
through a -63% drawdown.**

Against holding ADRO with dividends reinvested (7.7x), timing multiplied the
outcome by **4.9x**. That is a real and large effect, and it is the strongest
result for market timing anywhere in this repository.

It is also 38% of a hundredfold, and the remaining 62% does not exist at any
leverage a margin account survives. The perfect-foresight ceiling on the same
stock is 198,161x; the gap between that and 38x is the value of knowing which
low is *the* low, and nothing tested here recovers it.

## Result 57 — the capitulation detector finds the right week and still loses

The validated capitulation rule from Part III (gap <= -10%, index also gapping
down, prior 20 days negative - 84.6% of which made +5% across 544 IDX-wide
cases) applied to ADRO as an entry, since catching the low is precisely what it
was built for.

It fires **10 times in 18 years**, and it identifies the right moment:

| | |
|---|---|
| 2008-10-06, -07, -08, -27, -28, 11-06, 11-13 | **7 of 10 signals** |
| the DP's optimal buy | **2008-11-24** |

**Seven of ten signals land within seven weeks of the single best entry in the
stock's history.** The detector works.

The strategy built on it does not:

| strategy | growth | CAGR |
|---|---|---|
| breakout only (baseline) | 19.2x | +17.8% |
| capitulation OR breakout | 11.1x | +14.2% |
| buy & hold | 7.8x | +12.0% |
| capitulation -> hold 500 days | 1.6x | +2.5% |
| capitulation -> exit at new high | 1.3x | +1.5% |
| capitulation -> exit on trend break | 0.8x | -1.4% |

Ten signals in eighteen years, seven of them in the same month, is not a
strategy - it is one observation. The capitulation rule earned its 84.6% hit
rate across **544 cases spanning the whole exchange**; concentrated into a
single name it has no sample left. Being right about October 2008 is not an
edge if October 2008 is the only thing you are right about.

This is the sixteenth distinct approach and the last. **The ceiling for ADRO
timing stands at 38.1x.**

---

# Part XIII — What is verified to work, across the whole exchange

## Result 58 — ADRO was special, and that matters

The 38.1x achieved by timing ADRO (Part IX-XII) does **not** generalise. The
same Donchian rule applied to all 55 blue chips with 1,500+ bars:

| rule | median CAGR | median buy & hold | median excess | names beating B&H |
|---|---|---|---|---|
| donchian 20/100 | +11.1% | +12.5% | **-1.4%** | **40%** |
| donchian 20/55 | +10.2% | +12.5% | -2.4% | 29% |
| ma_cross 20/100 | +9.9% | +12.5% | -2.4% | 33% |
| momentum 120d | +5.8% | +12.5% | -8.9% | 9% |

**ADRO ranked 3rd of 55** on timing excess (+8.7%). TKIM (+11.6%) and TINS
(+10.8%) were better; HMSP (-14.1%) and MAPI (-14.0%) were disasters. The
correlation between a name's buy-and-hold return and its timing excess is
-0.015 - essentially zero, meaning there is no way to know in advance which
names timing will help.

Optimising a timing rule on one stock and reporting the result is therefore
close to meaningless. It was the right instinct to ask whether it generalised.

## Result 59 — what does generalise: selection plus a trend overlay

120 combinations of score weighting, horizon, concentration and a per-name trend
filter, walk-forwarded with the combination **re-selected from scratch in every
fold**:

| fold | test to | chosen | in-sample | out-of-sample | equal-weight | excess |
|---|---|---|---|---|---|---|
| 1 | 2013-10 | rel_strength 20d top3, no filter | +21.0% | +25.1% | +9.4% | +15.7% |
| 2 | 2016-12 | momentum 60d top5, up200 | +19.4% | **-8.9%** | +6.0% | **-14.9%** |
| 3 | 2020-03 | shipped momentum 20d top5, up100 | +17.5% | +21.2% | -11.4% | +32.6% |
| 4 | 2023-05 | shipped momentum 20d top3, up100 | +22.3% | +89.7% | +19.3% | +70.4% |
| 5 | 2026-08 | shipped momentum 20d top3, up100 | +30.1% | +67.4% | +3.4% | +64.0% |

| | re-selected each fold | **fixed default** |
|---|---|---|
| mean OOS CAGR | +38.90% | +20.79% |
| **median OOS CAGR** | **+25.08%** | **+21.06%** |
| worst fold | **-8.87%** | **+11.44%** |
| folds beating equal-weight | 4 of 5 | **5 of 5** |

**The fixed default wins the argument that matters.** `shipped momentum, 20-day,
top 5, held only while above the 200-day average` beat equal-weight in **five
folds out of five**, never had a losing fold, and its median (+21.1%) is close
to its mean (+20.8%) - the signature of a result that is not one lucky window.

Re-selecting each fold has a higher mean (+38.9%) and is worse advice: it
produced the only negative fold in the study, and its mean is inflated by two
outliers.

## Result 60 — where the big folds actually came from

Fold 4 returned +89.7%. Verified against raw prices - every forward return
recomputes exactly, no period beyond +/-60%, no artifacts. It is real. But
reading the 37 rebalances one by one:

| | |
|---|---|
| total | 7.14x over 3.2 years |
| mean rebalance | +6.60% |
| **median rebalance** | **-1.17%** |
| when it was earned | **2020-10 to 2021-07** |

The entire 7.14x came from nine months. After July 2021 the median rebalance
lost money for two straight years. And the picks in the winning stretch - DMMX,
SAME, TFAS, YELO, ARTO, BBHI - are the 2021 Indonesian small-cap mania.

**Momentum harvests bubbles. It does not manufacture returns in ordinary
markets.** That is not a criticism of the method; it is a description of what
the method is, and it should change what you expect from it between manias.

## The verified answer

**Best method verified to work across the exchange, not one stock:**

```
rank all liquid IDX names on the momentum composite within each date
hold the top 5, equal-weight
only while the name trades above its 200-day average
rebalance every 20 days, no take-profit, always invested
```

| | |
|---|---|
| out-of-sample CAGR | **+20.8% mean, +21.1% median** |
| equal-weight benchmark | +5.4% |
| **excess** | **+15.4%** |
| folds beating benchmark | **5 of 5** |
| worst fold | +11.4% |

Every absolute figure carries survivorship (Part XII measured IHSG at 10.9x
against 295.2x for the same panel's blue chips). **The excess over equal-weight,
which is measured inside the same biased panel, is the number that survives.**

---

# Part XIV — The complete trading history, from the start of the data

Rp50,000,000 on 2000-03-30, the validated engine run across the whole exchange
to today, every transaction logged. Rank all liquid IDX names on 120-day
momentum within each date, hold the top 5 equal-weight, rebalance every 20
sessions, always invested. Whole lots, 0.15%/0.25% fees, next-bar fills, a
10%-of-turnover capacity cap.

## Result 61 — the record

| | |
|---|---|
| final value, 2026-08-14 | **Rp17,990,489,691** |
| growth | **359.8x** over 26.4 years |
| CAGR | **+25.00%** |
| max drawdown | **-66.2%** |
| transactions | 2,244 (1,057 buys, 721 sells, 466 trims) |
| distinct names traded | **271** |
| fees paid | Rp2,660,790,578 |

Full log: ``reports/trading_history.txt`` and ``reports/paper_account_trades.csv``.

The first trade was **1,076 lots of ASII at Rp93 on 2001-04-27**. The most
recent was **307,450 lots of JGLE at Rp100 on 2026-08-07**.

## Result 62 — year by year, and the part nobody quotes

| year | end equity | return | | year | end equity | return |
|---|---|---|---|---|---|---|
| 2001 | Rp74,499,675 | +49.0% | | 2014 | Rp1,408,168,003 | +5.9% |
| 2002 | Rp60,471,616 | -18.8% | | 2015 | Rp1,265,399,938 | -10.1% |
| 2003 | Rp108,677,525 | +79.7% | | 2016 | Rp2,367,565,297 | +87.1% |
| 2004 | Rp179,748,415 | +65.4% | | 2017 | Rp3,566,140,646 | +50.6% |
| 2005 | Rp155,052,019 | -13.7% | | 2018 | Rp5,828,066,941 | +63.4% |
| 2006 | Rp242,579,277 | +56.5% | | 2019 | Rp4,246,028,924 | **-27.1%** |
| 2007 | Rp649,808,424 | **+167.9%** | | 2020 | Rp7,336,580,592 | +72.8% |
| 2008 | Rp547,410,032 | -15.8% | | 2021 | Rp17,444,559,546 | **+137.8%** |
| 2009 | Rp858,329,219 | +56.8% | | 2022 | Rp16,054,687,377 | -8.0% |
| 2010 | Rp1,304,462,022 | +52.0% | | 2023 | Rp20,150,729,098 | +25.5% |
| 2011 | Rp1,114,017,526 | -14.6% | | 2024 | Rp14,750,499,399 | **-26.8%** |
| 2012 | Rp1,444,827,110 | +29.7% | | 2025 | Rp36,478,316,403 | **+147.3%** |
| 2013 | Rp1,329,560,014 | -8.0% | | **2026 YTD** | **Rp17,990,489,691** | **-50.7%** |

**The book is in a -53.8% drawdown right now.** It peaked at Rp38,967,094,300
and sits at Rp17,990,489,691. The headline 359.8x is what remains *after* the
account halved in 2026. Anyone quoting the multiple without that sentence is
quoting a number the strategy itself is not currently earning.

Eight of twenty-six years were negative, four of them worse than -18%. The
equity curve spent time below -40% from its running peak in **twenty-one
separate episodes**, including 1.2 years underwater from 2019-09 to 2020-12.

## Result 63 — the 200-day trend overlay does not survive a change of scorer

Part XIII recommended adding a 200-day trend filter on the strength of the
walk-forward, where it took the book from 4 folds of 5 beating equal-weight to
5 of 5. Run inside this account, on 120-day momentum over 805 names, it does
the opposite:

| | final value | growth | CAGR | max drawdown |
|---|---|---|---|---|
| **without the filter** | **Rp17,990,489,691** | **359.8x** | **+25.00%** | -66.2% |
| with the filter | Rp5,520,926,516 | 110.4x | +19.53% | **-81.9%** |

It costs 5.5 points of CAGR *and* deepens the worst drawdown by 16 points.

The two results are not reconcilable by argument, and the difference is the
scoring function: the walk-forward ranked on the momentum *composite* over the
observation panel, this account ranks on raw 120-day momentum over 805 names.
**A filter that helps one scorer can hurt another**, which means the Part XIII
recommendation should be read as conditional on its scorer rather than as a
general improvement. Reported here rather than quietly dropped, because the
contradiction is the finding.

## What this history is, and is not

It **is** a complete, auditable record: 2,244 fills, every one priced at the
next session's open, whole lots, fees charged, positions capped at a tenth of
the name's daily turnover.

It **is not** a forecast, and three things bound it:

* **Survivorship.** The panel is today's listing. IHSG returned 10.9x over the
  same window while this panel's blue chips returned 295.2x - a factor of 27
  that is bias, not skill. The 359.8x carries it.
* **Concentration in manias.** 2007, 2021 and 2025 supplied most of the
  compounding. The names bought in those years - BEKS, BCIP, BTEK, AYLS, DOOH,
  ARTO, BBHI - are speculative small caps, and momentum owns them because they
  are going up, not because they are sound.
* **The drawdowns are real and current.** -66.2% at worst, -53.8% today.

---

# Part XV — Optimising for consistency, and the 50/50 split

Part XIV's book compounds at +25% and is unpleasant to own: eight losing years
in twenty-six, -66% at worst, -53.8% today. Maximising CAGR selects for exactly
that shape, because one +140% year pays for four bad ones in the mean.

So this searches 288 configurations - momentum lookback, concentration,
rebalance frequency, and a **trailing exit that can sell a holding between
rebalances when it falls from its own high** - and scores every one on
consistency as well as return.

## Result 64 — rebalance frequency, not the trailing stop, is the mechanism

The intuition that "trading beats holding, even for blue chips" is correct. The
mechanism is not what it looks like:

| trailing exit | median CAGR | median max DD | median ulcer |
|---|---|---|---|
| none | +23.7% | -64% | 0.24 |
| 12% | **+18.8%** | -61% | 0.23 |
| 20% | +23.9% | -60% | 0.23 |
| 30% | +23.5% | -63% | 0.24 |

**A per-name trailing stop is close to worthless** - 20% buys 0.2 points of CAGR
and 4 points of drawdown, and a tight 12% stop *costs* 5 points by selling
noise. Every one of the top configurations by CAGR runs no trail at all.

What does the work is **rebalance frequency**. Sorted by CAGR, the top eight
configurations all rebalance every 5 or 10 sessions. Reshuffling the whole book
that often exits weakening names and enters strengthening ones automatically -
it *is* selling the peak and buying the low, executed across the universe rather
than stock by stock. A stop on a single name only ever sells; a rebalance sells
and buys in the same decision.

## Result 65 — the configuration, and it is stable

Best by CAGR-per-unit-of-time-underwater:

```
rank every liquid IDX name on 120-day momentum
hold the top 8, equal weight
rebalance every 10 sessions
no trailing stop, always invested
```

| | in-sample, full history |
|---|---|
| CAGR | **+31.2%** |
| median year | +24.8% |
| worst year | -35.5% |
| positive years | 73% |
| **worst rolling 3 years** | **-16.6%** |
| max drawdown | -55% |
| trades | 6,058 |

**Walk-forward, re-selecting from 180 configurations in every fold, all five
folds independently chose this exact configuration.** That is the strongest
stability evidence in this repository - the search does not wander.

| fold | test to | out-of-sample CAGR | daily max DD |
|---|---|---|---|
| 1 | 2012-09 | +49.1% | -30% |
| 2 | 2016-03 | **-1.7%** | -47% |
| 3 | 2019-09 | +47.2% | -29% |
| 4 | 2023-02 | +28.7% | -60% |
| 5 | 2026-08 | **-5.9%** | **-70%** |

Mean out-of-sample **+23.5%**, median **+28.7%**, and **two folds of five were
negative**. The most recent fold is one of them. This is a high-return,
high-variance engine, not a smooth one.

## Result 66 — the 50/50 split, priced

Momentum book against the multibagger sleeve (cheap, small, far below its old
high, state-owned; three-year holds laddered a third a year), over the 18 years
both cover:

| allocation | CAGR | growth | median year | worst year | positive years | max DD |
|---|---|---|---|---|---|---|
| 100% momentum | **+28.4%** | 90.6x | +23.9% | -19.4% | 72% | -21% |
| 70/30 | +27.8% | 83.3x | +28.0% | -16.3% | 83% | -20% |
| **50/50 rebalanced** | **+27.1%** | **74.4x** | **+31.1%** | **-14.3%** | **89%** | **-19%** |
| 30/70 | +25.9% | 63.6x | +29.4% | -12.2% | 83% | -20% |
| 100% multibagger | +23.7% | 46.0x | +27.8% | **-9.1%** | 83% | -25% |

**The 50/50 is the best consistency point.** It gives up 1.3 points of CAGR
against the pure momentum book and buys: the **highest median year of any
allocation (+31.1%)**, the **highest share of positive years (89% - sixteen of
eighteen)**, a worst year improved from -19.4% to -14.3%, and a shallower
drawdown. Correlation between the sleeves is +0.478 - moderate, and enough.

Year by year, 50/50 rebalanced annually:
`2004 +73% · 2005 +36% · 2006 +71% · 2008 +3% · 2009 +57% · 2010 +46% ·
2011 +27% · 2012 +29% · 2013 +24% · 2015 +10% · 2016 +55% · 2017 +33% ·
2018 +37% · 2019 +8% · 2020 +37% · 2022 -14% · 2023 -6% · 2024 +2%`

## The one number in that table that is measured differently

The drawdowns in the allocation table are computed on **annual** returns,
because that is the only grid on which a 20-day book and a 3-year sleeve can be
combined. An annual grid cannot see a fall and recovery inside one year:

| same momentum book | |
|---|---|
| daily max drawdown | **-55.0%** |
| annual-grid max drawdown | -35.5% |
| what the allocation table shows for the mix | -19% |

Both are arithmetically correct. **Only the daily figure is what holding it
feels like**, and the -19% on the 50/50 row should be read as "shallower than
the alternatives on the same grid", not as the worst it can get.

---

# Part XVI — What weekly-swing accuracy is actually worth

The chart question: *mark the major turns on a weekly chart, ignore the noise,
and if you get them right 80% of the time, what is the CAGR?*

It is fully computable, and the answer arrives with a sting.

## Result 68 — the accuracy ladder on ADRO

Weekly bars, zigzag of alternating turns with every leg at least 20% - the
formal version of "circle the big swings". **33 legs in 18 years: one decision
every 6.6 months.** 16 up legs averaging +116.9%, 17 down legs averaging -37.4%.

Monte Carlo, 4,000 trials per row, costs charged on every leg traded:

| accuracy | median growth | CAGR | beats buy & hold |
|---|---|---|---|
| 100% (hindsight) | **33,339x** | +77.9% | 100% |
| 90% | 6,042x | +61.9% | 100% |
| **80%** | **864.6x** | **+45.4%** | **100%** |
| 70% | 117.4x | +30.2% | 92% |
| 60% | 18.9x | +17.7% | 66% |
| 50% (coin flip) | 2.6x | +5.4% | 28% |
| *buy & hold* | *8.2x* | *+12.3%* | — |

**So yes: 80% accuracy on weekly swings is worth +45.4% a year, 864x over
eighteen years.** The intuition is correct and the arithmetic supports it. You
need roughly **60-65%** just to beat holding the stock.

## Result 69 — directional accuracy is not the accuracy that pays

Scoring the mechanical rules against those same 33 legs - was the rule
positioned correctly for most of each leg?

| rule | legs right | accuracy |
|---|---|---|
| **above the 20-day MA** | **31/33** | **94%** |
| ma_cross 20/100 | 23/33 | 70% |
| donchian 20/55 | 22/33 | 67% |
| donchian 20/100 | 21/33 | 64% |
| ma_cross 50/200 | 17/33 | 52% |

A 94% hit rate. The Monte Carlo says 94% should return something past 5,000x.

**It returns 0.4x. Minus 5.5% a year.**

The reconciliation, leg by leg:

| | |
|---|---|
| mean up leg | **+116.9%** |
| mean captured by the rule | **+40.5%** |
| **share of the up move captured** | **31%** |
| mean down leg | -37.4% |
| mean still absorbed | **-16.7%** (0% would be perfect avoidance) |
| **position flips across all legs** | **569** |
| fees alone | **170.7% of capital** |

A moving average is right about *direction* and wrong about *timing*. It enters
after the turn, exits after the turn, and between the two turns it flips 569
times on noise inside the leg. Being on the correct side of a +117% move while
capturing +40% of it is not 94% accuracy in any sense that compounds.

**The Monte Carlo assumed a correct call captures the leg from turn to turn.
That assumption is doing all the work.** No rule tested here captures more than
about a third of an up leg, and that difference between a third and all of it is
worth roughly a thousand-fold over eighteen years.

## What this settles

The disagreement was never about whether swing trading beats holding. It does,
enormously, *if* the turns can be called. What the data says is:

* **80% turn-accuracy pays +45.4% a year.** The target is real.
* **~62% is break-even against buy-and-hold.** Below that, holding wins.
* **The mechanical rules score 52-70% on direction** - already near the
  break-even band, which is why they land at +22% rather than +45%.
* **And directional accuracy overstates them badly**, because capture fraction,
  not direction, is what compounds. On the one rule that scores 94%, capture is
  31% and the net result is a loss.

The open problem is not detecting that a stock is going up. That is nearly
solved - 94%. The open problem is **acting at the turn instead of a month
after it**, and nothing in this repository does that.

# Part XVII — Acting at the turn

Result 69 ended with an accusation rather than an answer: the moving average is
right about direction 94% of the time and loses money because it acts a month
after the turn, and *nothing in this repository does better*. This part builds
the rule that acts at the turn and finds out.

The instrument is the **causal reversal filter** - a state machine that sells
when the close falls a set distance below its high since entry, and buys when it
rises a set distance above its low since exit. It is the causal twin of the
zigzag in Result 68: the zigzag marks a turn with hindsight, this marks it once
the market has moved far enough to prove it. Its lag is **bounded by its own
threshold**, which is the property a moving average does not have - an MA's lag
depends on the window and on the path, and can be arbitrarily long.

`scripts/turn_trader.py`, `scripts/turn_book.py`, `tests/test_turn_trader.py`.
The look-ahead test that matters is truncation: removing the future must not
change any past state, asserted at three cut points.

## Result 70 — capture is fixed, and it is still not enough

On ADRO weekly, the same series the annotated chart was drawn on:

| threshold | growth | CAGR | vs B&H | trades | up captured | dn absorbed | flips |
|---|---|---|---|---|---|---|---|
| 6% | 0.5x | -3.6% | -16.0% | 173 | 25% | -21.1% | 173 |
| 10% | 5.3x | +9.7% | -2.6% | 103 | 37% | -16.6% | 103 |
| 15% | 13.2x | +15.4% | +3.0% | 55 | 55% | -19.3% | 55 |
| 20% | 15.3x | +16.3% | +4.0% | 31 | 61% | -20.8% | 31 |
| **25%** | **27.3x** | **+20.1%** | **+7.7%** | **19** | **69%** | -21.8% | 19 |
| 30% | 15.5x | +16.4% | +4.0% | 17 | 67% | -21.6% | 17 |
| buy & hold | 8.2x | +12.3% | — | 1 | 100% | -100% | 0 |

Compare the capture column to Result 69's moving average: **31% → 69%**, and
569 flips → 19 trades. The diagnosis was right and the fix works. On ADRO the
filter turns 8.2x into 27.3x.

**Then it dies out of sample.** Choosing the threshold pair on 50 names before
2013 and applying it untouched to 86 names after:

| | |
|---|---|
| chosen in sample (buy 20% off the low, sell 30% off the high) | **+2.76%/yr** median excess |
| the same pair, out of sample | **-1.92%/yr** median excess |
| names where it beat holding, out of sample | **35% of 86** |
| the *best* pair out of sample, chosen with hindsight | **-0.12%/yr** |

The last line is the one that settles it. Not "the tuning did not transfer" -
**no threshold pair beat buy-and-hold out of sample, including the one picked
with full knowledge of the answer.** There was nothing to transfer.

And it fails in the place it was supposed to earn its keep:

| out-of-sample group | n | median B&H | median filtered | median excess | % beat |
|---|---|---|---|---|---|
| stock rose | 50 | +8.7% | +8.0% | **-2.84%** | 28% |
| **stock fell** | 36 | -4.6% | **-8.0%** | **-0.63%** | 44% |

A rule whose entire purpose is to be out of the down legs made the names that
fell *worse*. Sitting out costs the rebound, and the rebound arrives before the
20% re-entry trigger does.

The arithmetic of why is visible in the ADRO table. A 25% filter on a -37%
average down leg exits at 0.75 of the peak and re-enters at 1.25 of the trough
= 0.78 of the peak. You avoid a 37% fall to buy back 3% lower. The down leg has
to be much deeper than the round trip is wide before avoidance pays, and on IDX
most are not.

## Result 71 — the same rule works, as a gate, on a book

The single-name test asks the filter to do two jobs: find the up leg and hold
it. Inside a portfolio it only has to do the second - momentum ranking finds the
names. So the whole Part XV book was re-run with nothing changed but the gate.

Out of sample, five expanding walk-forward folds, every gate on every window:

| gate | 2009-04 | 2012-09 | 2016-03 | 2019-07 | 2023-01 | mean | worst | beats ungated |
|---|---|---|---|---|---|---|---|---|
| none | +38.7% | +3.8% | +60.9% | +30.8% | -12.3% | +24.4% | -12.3% | — |
| **ma20** | +31.5% | +9.2% | +62.3% | **+70.0%** | +14.3% | **+37.5%** | **+9.2%** | **4/5** |
| rev 15%/15% | +39.1% | +4.2% | +52.9% | +55.4% | +20.1% | +34.3% | +4.2% | 4/5 |
| rev 12%/12% | +37.5% | +4.1% | +56.5% | +58.0% | +16.4% | +34.5% | +4.1% | 3/5 |
| rev 20%/12% | +35.5% | +3.0% | +55.3% | +50.7% | +20.3% | +33.0% | +3.0% | 2/5 |
| ma200 | +34.7% | +2.5% | +48.0% | +37.4% | -10.2% | +22.5% | -10.2% | 2/5 |

Mean out-of-sample drawdown: ungated **-47%**, ma20 **-28%**, rev 15/15 -38%,
ma200 -45%. The ma20 gate is better in **5 folds of 5** on drawdown.

**The moving average - the rule that loses 5.5% a year on its own - is the best
gate available.** That is the reconciliation Result 69 was missing. Capture
fraction governs a rule that has to *choose what to own*. A gate never chooses:
momentum hands it eight names and it only decides when to stop holding each one.
Entering 40% late costs nothing when something else did the entering.

The bounded-lag filter is a close second and needs a threshold; the moving
average needs none. There is no case for swapping.

## Result 72 — the Part XIII/XV contradiction, resolved

Part XIII found the trend filter improved walk-forward folds; Part XIV found it
turned 359.8x into 110.4x on the full path. Both were correctly measured and
they disagreed, and it was left standing as a contradiction.

It is not one. The walk-forward begins at the 35% mark, April 2009. Run the
segment it never tests:

| 2000-03 to 2009-04 | CAGR | growth | maxDD |
|---|---|---|---|
| ungated | **+41.4%** | **22.8x** | -55% |
| ma20 | +24.6% | 7.3x | -39% |
| rev 15%/15% | +28.0% | 9.3x | -59% |

**The ungated book's entire full-path advantage is earned before 2009 and
nowhere else.** Over the whole record the two are a coin flip (1,296x vs
1,231x); across the five tested windows since 2009 the gate wins four and cuts
drawdown in all five.

Which side of 2009 deserves the weight is not a close call. The pre-2009 slice
is the most survivorship-contaminated data here - it is the set of names that
existed in 2000 *and* still trade today, and the ones that went to zero in
between are simply absent. A +41% ungated CAGR measured on survivors is not a
finding, it is a selection effect. **The gate stays.**

One more thing worth recording, because it nearly went the other way. An in-fold
selector scoring gates by training-window CAGR-per-ulcer chose `none` in all
five folds and delivered +24.4% against ma20's +37.5%. Had the walk-forward been
reported the usual way - "the selector chose X, X returned Y" - this part would
have concluded that gating does not help. Printing every gate on every window is
what made the answer visible.

## Where this leaves the question the chart asked

The chart asked for a trader who sells the circled highs and buys the circled
lows on a weekly chart. Result 68 priced that at **+45.4% a year** if the turns
are called 80% right, with break-even against holding at about 62%.

This part tried to build it and got a clean negative: a bounded-lag turn rule on
a single name **cannot** be tuned into an edge on IDX - not badly, not
marginally, but with the best hindsight-chosen setting still at -0.12%/yr. Turn
timing on one stock is not where the money is.

Where it is: the same mechanism applied across the exchange, deciding *which*
eight names to own rather than *when* to own one. That book's five out-of-sample
windows average **+37.5%** with a -28% mean drawdown. It is not the +45.4% the
chart implies, and it does not get there by calling turns - it gets there by
never holding anything that is not already working.

# Part XVIII — Maximising it, and what the search actually buys

Part XVII ended with a negative on single-name turn timing and a positive on the
gated book. Two things were left untried, and both bear directly on "optimise it
and maximise it", so both were run.

`scripts/turn_trader.py --universe` (sigma-scaled thresholds),
`scripts/maximize_book.py` (the grid).

## Result 73 — it was not the parameter's scale

The obvious objection to Result 70 is that a fixed percentage cannot be right
for every name: 25% is an enormous move for BBCA and a quiet week for a coal
stock, and ADRO's own weekly volatility in 2008 is nothing like its 2016. A
threshold expressed in **sigmas** has no scale to mis-transfer. Same walk, same
86 names, thresholds in trailing 52-week volatility units:

| | buy | sell | median excess | % beating B&H |
|---|---|---|---|---|
| best in sample | 2.0σ | 6.0σ | +0.71%/yr | 56% |
| the same pair, out of sample | 2.0σ | 6.0σ | **-0.80%/yr** | 41% |
| best out of sample (ceiling) | 4.0σ | 2.0σ | **+0.33%/yr** | 51% |

The ceiling moved from -0.12%/yr to +0.33%/yr - on 51% of names, which is a coin
flip. **Normalising by volatility does not rescue it.** The percentage grid did
not fail because 25% was the wrong number; it failed because turn timing on a
single IDX name has no edge to find. That line of attack is now closed from both
directions.

## Result 74 — concentration is the lever, and the gate is what makes it safe

315 configurations - 7 gates (including two that combine the moving average with
the bounded-lag filter: `both` = AND, `either` = OR) × 3 lookbacks × 5 position
counts × 3 rebalance intervals - each run on all five out-of-sample windows.

Aggregating across every gate, lookback and rebalance, the position count is the
one lever that moves monotonically:

| names held | median OOS mean CAGR | median mean drawdown |
|---|---|---|
| **3** | **+40.4%** | -47% |
| 5 | +35.2% | -41% |
| 8 | +30.7% | -37% |
| 12 | +24.4% | -34% |
| 20 | +18.4% | -30% |

And aggregating by gate, the gate is what makes the bad years survivable:

| gate | median OOS mean | median worst fold | median drawdown |
|---|---|---|---|
| both (ma20 AND reversal) | +29.6% | **+1.5%** | **-32%** |
| either | +31.0% | +1.3% | -37% |
| ma20 | +30.2% | +1.1% | -33% |
| rev15 | +30.6% | +1.6% | -37% |
| ma50 | +24.4% | -2.7% | -43% |
| **none** | +24.6% | **-8.5%** | **-46%** |

The two interact, and that is the finding. Holding the same 120-day/rebalance-10
book at the deployed setting:

| gate | top 3 | top 8 | top 20 |
|---|---|---|---|
| ma20 | +57.2% mean, **worst +13.6%**, -38% dd | +37.5%, +9.2%, -28% | +17.3%, -2.3%, -29% |
| none | +27.5% mean, **worst -39.9%**, -61% dd | +24.4%, -12.3%, -47% | +17.4%, -6.4%, -37% |

**Ungated, concentration is close to ruinous** - a -39.9% fold and a -61%
drawdown. **Gated, the same concentration is the best thing in the search.**
Share of configurations with all five folds positive: `ma20`/`both` at top 3 or
top 5 = **100%**; `none` at top 3 or top 5 = **0%**.

The single strongest configuration on the worst fold is `both` / 250-day / top 3
/ rebalance 10: folds of +29.7%, +24.4%, +74.4%, +118.1%, +40.4% - mean +57.4%,
worst **+24.4%**, mean drawdown -35%.

## Result 75 — the search does not pay for itself

Those are ceilings: chosen by looking at the windows they are quoted on. The
honest question is what a selector that cannot see them would have picked. Both
selectors run inside each fold on earlier data only - one scoring on everything
before the fold, one on the trailing five years:

| | growth | CAGR | mean fold | worst fold | mean drawdown |
|---|---|---|---|---|---|
| expanding selector | 100.2x | +30.5% | +32.1% | +4.1% | -42% |
| trailing selector | 116.6x | +31.6% | +33.8% | +0.8% | -43% |
| **fixed, deployed (ma20/120d/top8/reb10)** | **178.1x** | **+34.9%** | **+37.5%** | **+9.2%** | **-28%** |
| fixed, ungated | 27.4x | +21.1% | +24.4% | -12.3% | -47% |

**Both selectors lose to the fixed configuration, on return and on drawdown.**
Neither ever found the top-3 book; the expanding one chose the *ungated* top-8
book in three folds of five. The grid contains configurations worth +57% a year
out of sample and no honest procedure in this repository retrieves them.

So the answer to "search harder" is: searching harder found the ceiling and
could not reach it. What survives is not a configuration the search picked but
a **structural** property the search revealed - fewer names is better, a gate is
mandatory - and structure is the kind of thing that transfers, because it does
not depend on which window you scored it on.

## What is deployed now

`scripts/current_picks.py` prints both momentum sleeves side by side:

* **top 8, 120-day, 20-day gate** — +37.5% mean OOS, -28% drawdown. Unchanged.
* **top 3, 250-day, both gates** — +57.4% mean OOS, worst fold +24.4%, -35%
  drawdown. Higher return, rougher ride, and it hits the turnover cap sooner.

Both are shown because the choice between them is a risk decision, not a
measurement one, and the measurement does not make it. The multibagger sleeve is
unchanged.

Every caveat from Part XV still applies and none of them got smaller: the
universe is survivor-only, the fold-4 numbers (+118%, +157%) are the 2020-21
mania and will not repeat on demand, and a three-name book means one delisting
is a third of the sleeve.

# Part XIX — The blue-chip algorithm

The book Part XVIII produced holds JGLE at Rp89 and KOTA at Rp177. Whatever
those are, they are not blue chips. This part restricts the same machinery to
large caps, and almost every conclusion reverses.

`scripts/bluechip.py`, `scripts/bluechip_ew.py`, `scripts/bluechip_picks.py`,
`tests/test_bluechip.py`.

## Result 76 — a blue-chip universe cannot be a list

The obvious universe is the 63 names under `bluechip` and `lq45` in
`config/universe.yaml`. Running that list back to 2000 buys BBCA in 2003 because
of what it became, and never buys the names that were giants then and are not
now. Measured against a universe rebuilt from data available at the time:

| equal-weight benchmark | 2009-04 | 2012-09 | 2016-03 | 2019-07 | 2023-01 | mean | full |
|---|---|---|---|---|---|---|---|
| point-in-time | +47.3% | +6.7% | +4.6% | +0.8% | -0.2% | +11.8% | **+10.5%** |
| today's list | +71.7% | +8.5% | +24.5% | +17.4% | +5.6% | +25.5% | **+24.1%** |

**The fixed list overstates the blue-chip benchmark by +13.5% a year.** Not a
detail - it is larger than the entire return of the honest version. Any claim of
the form "blue chips returned X" built on a current roster is roughly doubled.

Ranking on turnover instead is causal but is not a blue-chip screen, and today
it proves it: the top 40 IDX names by turnover include **BUMI, DEWA, BRMS, ENRG
and INET** - penny stocks in a speculative run, each churning more value per day
than ICBP. A universe built that way silently becomes a momentum-junk universe
in precisely the period it would be quoted from.

Market capitalisation would settle it. The only capitalisation data here is a
present-day snapshot for 59 names, and using today's share count to build a 2004
universe is the same look-ahead again. So membership uses three things knowable
at the time:

* trailing 250-day median turnover in the top `3 x size`, above an absolute floor
* at least 750 sessions of trading behind it
* 250-day realised volatility in the **calmer half of that liquid pool**

That last line is what separates a large cap from a penny stock having a year.
The result reads correctly at every date - BBCA, BBNI, BBRI, BMRI, TLKM, UNTR,
INTP, ISAT in 2008; the LQ45 core in 2014 and 2020; ASII, BBCA, BBRI, BMRI,
ICBP, INDF, KLBF, TLKM, UNTR, UNVR today - and it is tested by truncation, so
removing the future cannot change who was a member in the past.

## Result 77 — on blue chips, every lever points the other way

432 configurations: 4 gates x 9 ranking signals x 4 position counts x 3
rebalance intervals, each on five out-of-sample windows. Aggregated over
everything else:

| gate | median OOS mean | median worst fold |
|---|---|---|
| **none** | **+10.3%** | **-2.6%** |
| rev15 | +7.9% | -4.9% |
| both | +0.4% | -8.7% |
| ma20 | -0.2% | -9.8% |

| names held | median OOS mean | median worst fold |
|---|---|---|
| 3 | +2.8% | -8.7% |
| 5 | +2.5% | -5.8% |
| 8 | +3.2% | -4.6% |
| **12** | **+4.3%** | **-3.2%** |

| ranking signal | median OOS mean |
|---|---|
| **250-day momentum** | **+6.2%** |
| low volatility | +5.1% |
| 120-day momentum | +4.9% |
| distance below 1y high | +3.8% |
| buy the laggards (rev120/250/60) | +2.3% to +2.8% |

Set against Part XVIII, where the whole exchange said **gate hard, concentrate
hard**:

| | whole exchange | blue chips |
|---|---|---|
| gating | mandatory (ungated worst fold -8.5%) | **harmful** (ma20 worst fold -9.8%) |
| concentration | monotone better, top 3 best | **monotone worse**, top 12 best |
| signal | 120-day momentum | **250-day** momentum |

The gate result has a mechanism, and it is arithmetic rather than mystery. A
20-day trend gate applied to thirty large caps independently flips each name
roughly twenty times a year. Thirty names x twenty flips x a 0.4% round trip on
a thirtieth of the book is about **ten points a year in fees**, and no large-cap
trend edge covers that. The whole-exchange book pays the same toll and can
afford it because the ranking edge there is worth 30 points; here it is worth
three or four.

The concentration result has the same shape. Concentration pays when the ranking
is strong enough that the top 3 really are better than the top 12. On blue chips
it is not, so concentration buys nothing and sells diversification.

And the direction matters: **buying the laggards is the worst signal in the
table.** "Sell the peak, buy the low" as a cross-sectional rule on blue chips -
own whichever large cap has fallen most - is worse than owning all of them.

## Result 78 — what the blue-chip algorithm is

Choosing the configuration the three lever tables agree on, rather than the
corner of the grid with the best number: **no gate, 250-day momentum, hold 12,
rebalance quarterly.**

| | 2009-04 | 2012-09 | 2016-03 | 2019-07 | 2023-01 | mean | worst | maxDD |
|---|---|---|---|---|---|---|---|---|
| own them all (no costs) | +47.3% | +6.7% | +4.6% | +0.8% | -0.2% | +11.8% | -0.2% | — |
| **250d / top 12 / reb 60** | +40.6% | +11.7% | +10.0% | +7.9% | +2.6% | **+14.6%** | **+2.6%** | **-24%** |
| best in the grid (lowvol/top5/reb60) | +42.7% | +17.8% | +17.5% | -3.6% | +6.3% | +16.1% | -3.6% | -23% |

The recommended line pays full fees; the benchmark line does not, so the +2.7%
gap is if anything understated. **Every one of the five windows is positive**,
the worst is +2.6%, and the drawdown is -24% against -60% for owning the
universe outright.

That -24% is the number worth noticing. The whole-exchange book earns +37.5% and
puts you through -28% to -47%; this earns +14.6% and puts you through -24%. They
are not competing products, and the 50/50 split exists precisely because they
are not.

## What the gated equal-weight version says, since it was the obvious idea

Holding *every* large cap whose trend gate is on - no ranking at all, the
portfolio form of the annotated chart - was tested separately with realistic
trading (positions drift; only names that actually enter or leave are traded):

| gate | mean OOS | worst fold | full record | maxDD |
|---|---|---|---|---|
| **none (own them all)** | **+10.5%** | **-2.7%** | +10.8% | -60% |
| reversal 15% | +7.7% | -7.3% | +6.6% | -54% |
| reversal 25% | +7.3% | -7.8% | +9.1% | -63% |
| 200-day average | +2.0% | -6.9% | +5.5% | -70% |
| 20-day average | -10.6% | -20.0% | -6.4% | -95% |

Nothing beats owning them all. The 20-day gate turns a +10.8% book into -6.4%.

One methodology note, because it changed a result: the first version of this
recomputed exact equal weights daily, which charges a fee on the whole book every
session - adding one name changes all thirty weights - and produced a -100%
drawdown that was entirely the implementation. The numbers above are from the
version that lets positions drift.

## Result 79 — the search loses here too, for the second time

Result 75 found that nested selection on the whole exchange underperformed
simply leaving the configuration alone. The blue-chip grid reproduces it, on a
different universe with a different signal set:

| | growth | CAGR | mean fold | worst fold | mean maxDD |
|---|---|---|---|---|---|
| nested selector, expanding window | 2.9x | +6.4% | +6.7% | -4.5% | -24% |
| nested selector, trailing 5 years | 2.9x | +6.3% | +7.3% | -6.7% | -34% |
| own the large caps, equally | 5.7x | +10.5% | +11.8% | -0.2% | — |
| fixed low-vol / top 8 / quarterly | 8.1x | +12.8% | +13.7% | +1.0% | -25% |
| **fixed 250-day / top 12 / quarterly** | **9.4x** | **+13.8%** | **+14.6%** | **+2.6%** | **-24%** |
| fixed low-vol / top 5 / quarterly | 11.3x | +15.1% | +16.1% | -3.6% | -23% |

**Both selectors lose to doing nothing at all.** Not merely to the best fixed
configuration - to owning the universe equally weighted, which requires no
search, no signal and no rebalancing rule.

Two independent universes, two independent grids, the same answer: the search is
useful for finding *which levers matter* and worthless for *picking a
configuration*. Every deployed setting in this repository is therefore chosen
from lever agreement, never from the top row of a sorted table.

## The blue-chip book, as deployed

`scripts/bluechip_picks.py` rebuilds the universe on the latest bar and sizes
the book in whole lots, capped at 10% of each name's daily turnover.

    python3 scripts/bluechip_picks.py --capital 25000000 --signal mom250 --top 12

Held quarterly, no gate, no stop. The trend columns are printed because they are
worth seeing; acting on them made every version tested worse.

**Caveats specific to this part.** Fold 1 (April 2009 onward) carries +40% to
+47% for every configuration including the benchmark - it is the rebound off the
2008 low, and it flatters all of them equally. Strip it and the remaining four
windows for the deployed line are +11.7%, +10.0%, +7.9%, +2.6%: still positive
in all four, still above the benchmark, but nothing like the headline. The
universe is also missing delisted names entirely, so even the point-in-time
screen inherits whatever survivorship the price archive itself carries.

# Part XX — Learning the turn, and a bug in the ruler

The goal was to make the answer yes: buy the circled lows, sell the circled
peaks. Every previous attempt used a rule I chose and only the name's own price.
This part removes both limitations - and, on the way, finds that the instrument
used to measure all of them was subtly broken.

`scripts/turn_ml.py`, `tests/test_turn_ml.py`.

## Result 80 — the zigzag was wrong, and it barely mattered

Writing a regression test for the label generator exposed a bug in `zigzag`, the
function that defines what a "turn" is and therefore underpins Results 68-73.
Given a clean saw-tooth - 100 to 200 to 100 to 200 to 100 - it returned **two**
pivots instead of four.

The cause: while direction was still unknown, a single running extreme was
updated whenever price moved *either* way, so it collapsed onto the previous bar
and confirming a turn required a one-bar move of the full threshold. Fixed by
tracking the running high and running low separately and resetting both only when
a turn is confirmed.

**The honest impact assessment**, because a broken ruler means every measurement
taken with it is suspect:

| | before the fix | after |
|---|---|---|
| ADRO legs at 20% | 33 | 32 |
| 80% accuracy is worth | 864.6x, +45.4%/yr | **829.9x, +45.0%/yr** |
| break-even accuracy vs holding | ~62% | **~62%** |
| capture, 25% reversal band | 69% | **69%** |

Almost nothing moved, and the reason is specific rather than lucky: ADRO's first
weekly bar in the sample is the 2008 crash, a single-bar move large enough to set
the direction immediately, after which the old code was correct. On a quiet
series it would have failed badly. Results 68-73 stand; the ruler is now right
for series that do not begin with a crash.

## Result 81 — the turn does not learn, as a switch

The label is the hindsight zigzag state - 1 in an up-leg, 0 in a down-leg -
which is exactly what the circles are. Features: 20 per name (returns at 4/13/26
/52 weeks, distance from 52- and 156-week highs and lows, volatility and its
ratio, RSI, three moving-average distances, drawdown, weeks since the high,
turnover trend, acceleration), 17 market-state series from the macro panel, and
two cross-sectional ranks. Gradient boosting, 74,793 weekly rows, 86 names.

The protocol is where this either holds or lies, so: **labels are rebuilt inside
each training window** by recomputing the zigzag on `prices[:cut]` alone, and the
unfinished final leg is left unlabelled because its direction is not knowable at
the cut. No shuffled cross-validation. Predictions traded with a one-bar lag at
0.6% per round trip. Entry and exit probabilities chosen on a validation slice at
the end of the training window.

| fold | median excess | beats B&H | turn accuracy | capture |
|---|---|---|---|---|
| 2012-03..2015-10 | -3.55% | 41% | 61% | 48% |
| 2015-10..2019-05 | -3.88% | 35% | 62% | 50% |
| 2019-05..2023-01 | **+2.42%** | 58% | 55% | 68% |
| 2023-01..2026-08 | -3.81% | 36% | 60% | 54% |
| **pooled** | **-2.37%/yr** | **43%** | **60%** | **55%** |

Turn accuracy of **60%** against a break-even of about **62%**, and capture of
55%. The model learns something real - 60% is not a coin flip - and it lands
just below the line where calling turns starts to pay. Ranked against everything
else tried:

| method | median excess/yr | capture |
|---|---|---|
| 20-day moving average | -5.5% | 31% |
| learned turn model | -2.37% | 55% |
| reversal band, tuned honestly | -1.92% | 68% |
| volatility-scaled band | -0.80% | — |
| reversal band, best with hindsight | -0.12% | — |

Four independent methods, four negatives, and the best of them is the one that
was allowed to cheat.

## Result 82 — the control that killed the good news

Used cross-sectionally instead - hold the names with the highest P(up-leg) rather
than switching one name on and off - the model finally looks like a success:

| | fold 1 | fold 2 | fold 3 | fold 4 | mean | worst |
|---|---|---|---|---|---|---|
| model, top 5, quarterly | +13.5% | +46.9% | +31.1% | +3.5% | **+23.8%** | +3.5% |
| model, top 10, quarterly | +15.4% | +22.5% | +22.4% | +9.3% | +17.4% | +9.3% |
| equal weight, all names | +4.6% | +20.8% | +13.5% | +8.7% | +11.9% | +4.6% |

Twice the benchmark. That is where this part would have ended if the control had
not been run - and the model's own features include momentum, so beating equal
weight proves nothing unless it also beats the momentum it was fed:

| | mean | worst |
|---|---|---|
| **52-week momentum, top 10, quarterly** | **+26.1%** | **+11.1%** |
| 52-week momentum, top 10, monthly | +24.0% | +12.4% |
| model, best configuration | +23.8% | +3.5% |
| 26-week momentum, top 10, quarterly | +22.7% | +9.6% |
| 13-week momentum, top 10, quarterly | +23.4% | +6.8% |
| equal weight | +11.9% | +4.6% |

**Plain 52-week momentum beats the learned model on the mean and by three times
on the worst fold.** Every raw momentum horizon beats it on the worst fold. The
model was handed those columns and made them worse. It is not adding
information; it is destroying it.

## Where this leaves the question

Asked whether the circled buys and sells can be executed, the answer is no, and
it has now been tested four ways with four different failure modes:

* a **fixed percentage** band - fails, hindsight-best -0.12%/yr
* a **volatility-scaled** band - fails, hindsight-best +0.33%/yr on 51% of names
* a **learned classifier as a switch** - fails, 60% accuracy against 62% break-even
* a **learned classifier as a ranker** - fails, beaten by the momentum it was fed

What consistently works is the thing that never tries to call a turn: rank on
slow momentum, hold a basket, rebalance quarterly. On this weekly panel that is
+26.1% mean out-of-sample with a worst fold of +11.1%, which corroborates the
daily-panel books in Parts XVIII and XIX from independent code.

**The one avenue not tested is blocked by data, not by method.** Broker-summary
flows - who is accumulating at the low and distributing at the peak - are the
natural information source for turn detection and the original premise of this
repository. There are 92 days of it here, for one name. IDX prohibits scraping
and the IndoPremier module is a courtesy-access public page, so a 25-year
history cannot be built from the sources available. That is a limit on what has
been shown, not a demonstration that it would work.

# Part XXI — Timing the market instead of the stock

Result 77 gave the per-name gate a specific cause of death: a 20-day filter
applied to thirty large caps independently flips each about twenty times a year
and costs roughly ten points a year in fees. The gate was not wrong about
direction. It was applied thirty times over on names that are one correlated
basket.

That diagnosis points somewhere untested. A **market-level** call moves the whole
book on one decision, so the same correct calls cost a thirtieth of the fees; an
index averages away the single-name noise that made per-name turns unlearnable;
and it can use information that does not exist for a single stock - **breadth**,
the share of the market above its own trend, plus the macro panel. The circled
turns on the annotated chart are mostly market events anyway: 2008, 2011, 2015,
2020, 2025.

`scripts/market_timing.py`, `tests/test_market_timing.py`. IHSG weekly from 1990,
1,880 bars, 48 swings of 15%+ - one decision every nine months.

## Result 83 — the return edge is a threshold artefact

Scored on the index itself from 2006:

| timer | growth | CAGR | vs holding | trades | in market | maxDD | turn acc | capture |
|---|---|---|---|---|---|---|---|---|
| hold | 4.56x | +7.9% | — | 0 | 100% | **-59%** | — | 100% |
| above 30w MA | 3.93x | +7.1% | -0.8% | 82 | 66% | -34% | 88% | 56% |
| above 40w MA | 4.37x | +7.6% | -0.2% | 62 | 68% | -29% | 81% | 59% |
| breadth > 40% | 3.49x | +6.4% | -1.4% | 54 | 61% | -28% | 69% | 56% |
| **learned** | **5.47x** | **+8.9%** | **+1.0%** | 47 | 78% | -37% | 50% | 74% |

The learned timer beats holding by a point a year *and* removes 22 points of
drawdown. It is the first positive return result in this entire line of work,
and it does not survive being asked twice.

Across **36 settings** - 4 zigzag thresholds x 3 hysteresis pairs x 3 fold counts:

| | |
|---|---|
| median excess over holding | **-1.02%/yr** |
| settings with a positive edge | **36%** |
| range | -3.79% to +1.56% |

The +1.0% headline is the 0.15 threshold, which is the only one of four with a
positive median. Picking it was picking a corner. **There is no return edge.**

## Result 84 — the drawdown edge is real, and it is the only robust timing result in this repository

The same 36 settings, on the number nobody picked a threshold to optimise:

| | |
|---|---|
| median drawdown | **-37%** against **-59%** holding |
| median improvement | **+22 points** |
| range of improvement | **+10 to +37 points** |
| settings better than holding | **36 of 36 — 100%** |

Every threshold, every hysteresis pair, every fold count. Nothing else tried
across Parts XVII, XX and XXI is unanimous across its own robustness sweep.

Applied to the blue-chip book (2012-2026, continuous):

| timer | CAGR | worst year | maxDD | ulcer |
|---|---|---|---|---|
| always on | **+7.8%** | -13.8% | -38% | 0.11 |
| **above 30w MA** | +5.1% | **-7.5%** | **-16%** | **0.09** |
| learned | +5.5% | -18.7% | -28% | 0.13 |
| breadth > 40% | +3.6% | -18.5% | -33% | 0.15 |

**The trade, stated plainly: give up 2.7% a year, remove 22 points of drawdown
and 6.3 points from the worst year.** And the rule that does it best is not the
learned model - it is price above its own 30-week average, which costs nothing
to compute and has no parameters to fit beyond the window.

## What this finally settles about the circles

The chart's two circles are not equally hard, and eight parts of work now say so
in one line:

* **The red circles - selling before the fall - are achievable.** Not at the
  peak; a market timer exits well below it. But 22 points of drawdown removed in
  36 of 36 settings is a real, repeatable, robust effect.
* **The green circles - buying the low - are not.** Every attempt to add return
  by re-entering near the bottom has failed: fixed bands, volatility-scaled
  bands, learned classifiers per name, learned classifiers as rankers, and now
  market-level timing. The re-entry is always late enough that the rebound pays
  for the exit.

Which is why timing costs return and buys safety. You get out before the worst of
it and you get back in after the best of it, and on IDX those two roughly cancel
with a point or two a year left over as the fee.

**So the honest deliverable is a choice, not a discovery.** Own the blue-chip
book always-on at +7.8% with a -38% drawdown, or run the 30-week filter at +5.1%
with -16%. There is no third option in the data where timing pays for itself in
return, and I have now looked for it five different ways.

# Part XXII — How much insurance, and putting it on the chart

## Result 85 — the choice was not binary, and full exit is dominated

Part XXI framed the overlay as a switch: fully invested, or fully out when IHSG
is below its 30-week average. Nobody has to choose between the corners. The
overlay can **scale** exposure, and the middle had never been measured.

Holding X% of the blue-chip book while the market filter says OUT, full record:

| exposure when OUT | growth | CAGR | worst year | maxDD | +years | ulcer |
|---|---|---|---|---|---|---|
| 0% (Part XXI's rule) | 15.2x | +10.9% | -18.6% | **-41%** | 62% | 0.17 |
| **25%** | **18.1x** | **+11.6%** | **-19.0%** | **-34%** | **69%** | **0.14** |
| 50% | 20.9x | +12.2% | -26.1% | -38% | 69% | 0.12 |
| 75% | 23.7x | +12.7% | -32.9% | -47% | 73% | 0.12 |
| 100% (always on) | 26.1x | +13.2% | -39.6% | -54% | 69% | 0.12 |

**Going fully to cash is beaten on both axes by holding a quarter.** 25%
exposure returns more (+11.6% against +10.9%) *and* draws down less (-34%
against -41%), with more positive years and a lower ulcer index. That is not a
trade-off; it is a strict improvement, and it corrects the recommendation given
at the end of Part XXI.

The mechanism is the same one that has defeated every attempt to buy the low.
Exiting entirely means missing the rebound that follows the fall you avoided,
and on IDX the rebound is fast enough that the last quarter of exposure earns
more than it costs. You cannot get out of the decline without also getting out
of the recovery - so keep a foot in the door.

Across the five walk-forward windows the same ordering holds on the worst fold:
0% has a losing window (-0.5%), 25% does not (+1.4%), and every step up in
exposure improves both the mean and the worst window while the full-record
drawdown gets worse. The two views disagree because a -54% peak-to-trough spans
fold boundaries that a 3.5-year window cannot contain, which is exactly why both
are reported.

**Deployed: 100% when IHSG is above its 30-week average, 25% when it is below.**

## Result 86 — the same conclusions, on a TradingView chart

Two Pine v5 scripts, in `src/idxbot/tradingview/pine/`.

**`turn_reality.pine`** puts the disagreement itself on the chart. It draws the
hindsight zigzag - the circles, in orange, which **repaint**, and are labelled as
repainting - alongside the causal reversal filter that is the best honest
approximation of them. The gap between the dots and the triangles is the finding:
the circles bank 100% of every up leg, the causal filter banks 55-70%, and a
20-day moving average banks 31%. It also plots the live trigger level, so the
price at which the rule would next act is visible rather than inferred.

**`bluechip_regime.pine`** is the deployed overlay: IHSG weekly close against its
30-week average, driving a book exposure of 100% or 25%, with alert conditions on
each transition. It reads the last **closed** weekly bar (`close[1]`,
`lookahead_off`) because reading the current weekly close on a daily bar is
look-ahead by another name.

The cross-sectional half of the book cannot be expressed in Pine - ranking thirty
symbols against each other is not something an indicator on one chart can do - so
the scripts say so and point at `scripts/bluechip_picks.py` rather than pretending
otherwise.

`tests/test_pine.py` checks what is checkable without a Pine runtime: version
tags, delimiter balance, that `barmerge.lookahead_on` appears in nothing this
repository ships, that the regime script reads a closed bar, and that its
defaults still match the numbers its own header claims.

# Part XXIII — The rate on idle cash was doing all the work

## Result 87 — volatility targeting fails at the book level too

Part X dismissed volatility targeting on ADRO, but that was one stock, whose
volatility is mostly idiosyncratic noise. A portfolio's volatility is persistent,
which is the property the rule needs, so it was worth testing properly:
exposure = target / realised volatility, at three targets and three leverage
caps, with margin interest charged at 9% because a leveraged backtest that
ignores financing is a fiction.

| rule | CAGR | vs always-on | maxDD | DD saved | avg exposure |
|---|---|---|---|---|---|
| always on | +13.2% | — | -54% | — | 100% |
| voltarget 25%, cap 1.0 | +12.8% | -0.41% | -45% | +9pt | 96% |
| voltarget 20%, cap 1.0 | +11.9% | -1.29% | -40% | +14pt | 92% |
| voltarget 15%, cap 1.0 | +10.6% | -2.61% | -32% | +22pt | 83% |
| voltarget 25%, cap 2.0 | +10.8% | -2.37% | **-61%** | **-7pt** | 151% |

**Rules beating always-on on both return and drawdown: zero.** And leverage is
strictly harmful - the 2.0 cap versions lose return *and* deepen the drawdown,
because volatility targeting levers up into calm markets that are calm right
before they stop being calm, and pays 9% for the privilege.

## Result 88 — the overlay is free once cash earns the deposit rate

Every previous statement about what market timing costs - Part XXI's "-2.7% a
year", Result 85's "-1.57%" - assumed that money out of the market earns
**nothing**. For Indonesia that is simply wrong. IDR deposit and money-market
rates have sat between 3% and 6% across this record.

The overlay spends about a quarter of its life out of the market, so the rate on
that cash is not a rounding error:

| deposit rate | out at 0% | | out at 25% | |
|---|---|---|---|---|
| | CAGR vs always-on | maxDD | CAGR vs always-on | maxDD |
| 0% | -1.79% | -36% | -1.19% | -30% |
| 3% | -0.62% | -32% | -0.31% | -29% |
| **5%** | **+0.17%** | **-30%** | **+0.28%** | **-28%** |
| 6% | +0.56% | -28% | +0.58% | -28% |

**At 5% the overlay pays for itself.** Above about 4.5% it is free; below it, it
is cheap insurance. Always-on is -54%.

## Result 89 — the deployed book, and the first strict improvement in the project

The blue-chip book with the regime overlay, cash at 5%:

| | growth | CAGR | worst year | maxDD | +years | ulcer |
|---|---|---|---|---|---|---|
| book, always on | 26.1x | +13.2% | -39.6% | -54% | 69% | 0.12 |
| **book + regime overlay** | **27.9x** | **+13.5%** | **-17.3%** | **-28%** | **81%** | **0.10** |

Better on **every** column. More growth, more return, half the drawdown, less
than half the worst year, more positive years, lower ulcer index. Nothing else
in this repository does that - every other risk control traded return for safety.

It works for the reason the rest of the timing work failed, inverted. Timing
cannot beat holding on price alone, because you exit after the top and re-enter
after the bottom and those two roughly cancel. But the cancellation is only
"roughly", and the residual is about a point a year - which is smaller than the
interest earned on the cash you are holding while you wait. **The edge is not in
the timing. It is in being paid to wait.**

## Comparing this to the annotated chart, on its own series

ADRO weekly, 2008-2026, 32 swings of 20%+, one turn every 6.8 months:

| | growth | CAGR | maxDD | |
|---|---|---|---|---|
| the circles, in hindsight | 33,339x | +77.9% | — | not attainable |
| 80% of turns called right | 843.6x | +45.2% | — | what the chart asked for |
| 62% of turns called right | 27.4x | +20.1% | — | break-even against holding |
| causal filter, tuned with hindsight | 27.3x | +20.1% | -65% | best honest single-name rule |
| regime overlay on this name | 13.6x | +15.5% | -74% | the deployed rule |
| buy and hold | 8.2x | +12.3% | -80% | |

The deployed rule beats holding ADRO by 3.2 points a year, and the best
single-name rule that can be built - tuned with full hindsight, so an upper
bound - reaches exactly the break-even accuracy of 62% and no further. The chart
asked for 80%. The gap between rows two and five is not something more work
closes; it is the difference between a chart that has already happened and one
that has not.

# Part XXIV — 80% accuracy, delivered, and what it is worth

The instruction was direct: reach 80%+ turn accuracy as a minimum, on every
name. `scripts/accuracy_target.py` searches 25 rules - moving averages from 2 to
40 weeks, exponential averages, average-slope rules and reversal bands at six
thresholds - across 49 blue chips with 200+ weeks of history.

## Result 90 — the target is reached on 48 of 49 names

Choosing each name's most accurate rule:

| | |
|---|---|
| names reaching 80%+ | **48 of 49 (98%)** |
| names reaching 85%+ | 94% |
| median accuracy | **94%** |
| worst name | ANTM at 74% |
| **median return against buy-and-hold** | **-9.5% a year** |
| names where it beat buy-and-hold | **14%** |

The accuracy target is met. It costs nine and a half points a year.

No *single* rule reaches 80% on every name - the closest is the 6-week average
at 74% on its worst name and 89% median - but per-name tuning gets 48 of 49, and
the one holdout is a coal miner whose legs are unusually violent.

## Result 91 — accuracy and profit point in opposite directions, and it is not subtle

Across the 25 rules, median accuracy against median excess return over 49 names:

| | median accuracy | median excess return |
|---|---|---|
| five LEAST accurate rules | 51% | **-1.5%** |
| five MOST accurate rules | 91% | **-8.1%** |

**Spearman rank correlation: -0.810, p = 9.5e-07, n = 25.**

That is not noise and it is not a subtlety. Sorting rules by how often they call
the turn correctly sorts them almost perfectly by how much money they lose. The
individual rules bear it out:

| rule | median accuracy | names at 80%+ | median excess | trades |
|---|---|---|---|---|
| MA 3w | 92% | **98%** | **-10.4%** | 419 |
| MA 4w | 91% | 94% | -8.7% | 349 |
| MA 6w | 89% | 80% | -5.4% | 260 |
| MA 10w | 82% | 61% | -3.0% | 187 |
| reversal 15% | 66% | 6% | -2.3% | 55 |
| **reversal 25%** | **48%** | **0%** | **-1.5%** | 25 |

The best-returning rule in the entire search calls **48%** of the turns right -
worse than a coin flip - and it is the only one that comes close to holding.

## Why, in one chart

`scripts/plot_accuracy_proof.py` renders it. On ADRO the 3-week average scores
**94%** and trades **361 times**: it is on the correct side of nearly every leg,
and it re-enters and re-exits within each leg dozens of times on noise. It turns
8.2x into **1.1x**.

The mechanism has been measured three separate ways in this repository and they
all say the same thing. Being on the right side of a move is not the same as
being paid for it. What compounds is **capture** - how much of each leg is
actually banked - and capture falls as accuracy rises, because the only way to
be right about direction almost always is to re-evaluate almost constantly, and
re-evaluating constantly means buying every dip back and selling every wobble.

**So the 80% target was never the binding constraint.** It is reachable on 98%
of names and it is the wrong thing to maximise. The number that had to clear 62%
was accuracy *at full capture* - one decision per leg, held from turn to turn -
and no causal rule tested over eight parts of this work gets near it.

# Part XXV — Anatomy of the gap

Six methods have failed to close the distance between what a causal rule earns
(27.3x on ADRO) and what the circles are worth (33,339x). Before a seventh,
`scripts/gap_anatomy.py` measures where the distance actually goes, so the next
attempt aims at the part that can move.

## Result 92 — the gap is 126x arithmetic and 9.7x skill

A causal rule confirms a turn by waiting for price to move against the old
direction, so its band must be wider than the pullbacks that happen *inside* a
leg. Otherwise it exits on the pullback and re-enters higher. Measured on ADRO's
32 legs:

| | worst move against the leg, before it ended |
|---|---|
| 25% of legs | at least 3.6% |
| **50% of legs** | **at least 11.1%** |
| 75% of legs | at least 15.8% |
| 90% of legs | at least 18.5% |

A band of `b` gives up roughly `b` at entry and `b` at exit, so the best any
causal band could possibly do - perfect turn identification, still late by the
band - collapses as the band widens:

| band | best possible | CAGR | cost per leg |
|---|---|---|---|
| 5% | 6,722x | +62.8% | 10% |
| **10%** | **1,344x** | **+49.0%** | 20% |
| **15%** | **264x** | **+36.2%** | 30% |
| 20% | 50.8x | +24.3% | 40% |
| **25%** | **9.4x** | **+13.2%** | 50% |
| 30% | 1.7x | +2.9% | 60% |

Splitting the 1,219x total gap:

* **126x is structural** - the price of waiting for confirmation wide enough to
  survive the pullbacks the legs actually contain. No rule beats this while it
  confirms with price.
* **9.7x is execution** - imperfect turn identification, the part a better
  signal could still win.

**The important number is not either of those. It is that the median pullback is
11% while the best rule built uses a 25% band** - more than twice as wide as the
noise requires. The band is that wide only because narrow ones whipsaw in
practice. Fixing that whipsaw is worth the difference between 9.4x and 264x.

## Result 93 — 71% of exits are false, and nothing tested changes that

`scripts/shakeout.py` runs a 15% band on 49 blue chips and vetoes its exits five
different ways. A false exit is defined operationally: the price was more than 5%
higher within 13 weeks of selling.

| veto | false exits | exits | median excess | beats hold |
|---|---|---|---|---|
| market still up | **71%** | 21 | +0.47% | 53% |
| trend (above 30w) | 70% | 25 | -0.81% | 47% |
| none | 71% | 27 | -0.99% | 41% |
| trend AND market | 70% | 26 | -1.26% | 47% |
| age (hold 13 weeks) | 71% | 25 | -1.99% | 41% |
| quiet pullback | 75% | 19 | -3.04% | 22% |

**The false-exit rate is 70-75% regardless of the veto.** Not one of them moves
it. Whatever distinguishes a shakeout from a real turn is not in the price, not
in the volume, not in the stock's own trend, and not in the market's trend -
because all four were asked and all four gave the same answer.

## Result 94 — and the one that looked promising was period-specific

"Market still up" produced the first positive median excess for single-name
timing in this project: **+0.47%/yr**, beating buy-and-hold on 53% of names. It
was also chosen as the best of six on the full sample, which is the selection
error Results 75 and 79 both documented. Split by time:

| veto | before 2013 | from 2013 |
|---|---|---|
| none | -2.48% | -2.83% |
| **market still up** | **-2.00%** | **+0.26%** |
| trend (above 30w) | -0.80% | -0.03% |
| trend AND market | -1.70% | -0.89% |

It is negative in the half it was not selected from. The +0.47% was the
post-2013 half showing through a full-sample average. **The veto does not
survive.**

## What this leaves, stated precisely

The remaining target is no longer vague. It is:

> Reduce the 71% false-exit rate on a 15% band, using information that is not
> price, volume, the stock's trend, or the market's trend.

Everything on that exclusion list has now been tested and returned the same 70%.
The candidates that remain outside it are order-flow data - who is accumulating
into the pullback and who is distributing - which is the original premise of this
repository and for which 92 days of history exist, for one name.

That is a data boundary, not a method boundary, and it is worth stating exactly:
if the false-exit rate could be cut from 71% to 30%, the 15% band's ceiling of
264x becomes reachable rather than theoretical, and the answer to the annotated
chart changes from "no" to "most of the way".

# Part XXVI — Order flow, tested properly, and the end of the list

Result 93 left one candidate. Price, volume, the stock's own trend and the
market's trend all leave the false-exit rate at 70-75%, so whatever separates a
shakeout from a real turn had to be something else - and order flow was both the
obvious candidate and the premise this repository was built on.

## Result 95 — the broker-summary history is reachable after all

Every previous statement in this project that broker flow "has no history" was
based on the 92 cached daily files. That was wrong about the source, not just
about the cache. IndoPremier's module takes `start` and `end` and **returns the
whole window aggregated in one request**, with history back to at least 2010
(tested 2010-03 and 2012-05, both populated).

That changes the cost of the experiment by a factor of 36. The 1,728 pullback
windows across 49 blue chips need **63,830** requests day-by-day and **1,728** by
range - one per research event, 1.3s apart, cached permanently. Range queries
abbreviate their totals, which costs nothing here because every feature computed
is a ratio, and `idxbot.data.ipot` already records that directional metrics
survive that rounding.

A flaw caught in the smoke test would have invalidated the whole thing: building
broker classes from one live page classifies 11 codes, because a page shows only
the top 10 of each side. Using the repository's 66-broker registry instead gives
**100% classified, zero disagreements** with the live flags.

## Result 96 — a pilot signal that did not survive its own replication

**Pilot, 300 events.** Six features against two outcome labels:

| feature | difference | Cohen d | t | p |
|---|---|---|---|---|
| **foreign_net** (bounced) | **+0.0206** | **0.257** | **+2.00** | **0.047** |
| bumn_net (bounced) | -0.0133 | -0.233 | -1.60 | 0.112 |
| foreign_net (recovered) | +0.0044 | 0.054 | +0.40 | 0.690 |
| all others | — | <0.17 | — | >0.18 |

Foreign money net buying into the pullback, on pullbacks that bounced - the
direction bandarmology predicts. And 12 tests were run, so the Bonferroni
threshold is 0.05/12 = 0.0042. It was a candidate, not a result.

**Replication, pre-registered and committed before the outcome was known.** One
hypothesis, one feature, one-sided, alpha 0.05, on 600 events the pilot never
touched, powered to d = 0.23:

| | n | mean foreign_net |
|---|---|---|
| bounced | 439 | -0.0302 |
| did not | 161 | -0.0304 |
| **difference** | | **+0.0002** |

**Cohen d = 0.002. One-sided p = 0.4894.**

Not a smaller effect than the pilot's - **no effect at all**, to three decimal
places, on an adequately powered sample. The pilot's p = 0.047 was the one test
in twenty that comes up positive by construction, and the pre-registered check
is what caught it.

## Where this leaves the chase

The exclusion list is now complete for everything this data can express:

| information | false-exit rate it explains |
|---|---|
| price band alone | 71% |
| the stock's own trend | 70% |
| market trend | 71% |
| position age | 71% |
| volume on the pullback | 75% |
| **foreign order flow** | **no effect, d = 0.002** |

Six sources, one answer. **Whether a 15% pullback recovers or continues down is
not predictable from anything measurable in this dataset**, and that is why the
band has to be 25% wide, and why the ceiling is 9.4x instead of 264x.

That is a statement about IDX daily/weekly data and top-10 broker aggregates. It
is not a proof of impossibility - order flow at the *tick* level, options
positioning, or the full member-by-member tape rather than the top ten might
carry it. None of those are reachable from here, and the honest position is that
the gap between 27x and 844x remains open with no candidate left that this
project can test.

**What is delivered instead is the thing that did survive every test**: the
blue-chip book at +13.5% with a -28% drawdown against -54%, better on every
column than holding it always-on, because the cash it holds while waiting earns
more than the timing costs.

## Result 97 — and capitulation is not visible as a trajectory either

Result 96's null had a real flaw: it averaged each pullback window into one
number, and averaging erases sequence. The classic bottom is foreign money
selling into the decline and then *turning* late as sellers exhaust - and the
window average of a seller who becomes a buyer is approximately zero, which is
exactly what that null looked like.

So the pullback was split in half and the pre-registered question became whether
foreign flow *improves* from the first half to the second. 400 events, 4+ weeks
each, one-sided, powered to d = 0.27:

| | n | early | late | **delta** |
|---|---|---|---|---|
| bounced | 282 | -0.0268 | -0.0364 | **-0.0096** |
| did not | 118 | -0.0250 | -0.0340 | **-0.0090** |

**Difference -0.0005. Cohen d -0.005. One-sided p 0.5185.**

Nothing, again - and this time the interesting part is what both columns do.
Foreign flow gets *more negative* through the second half of a pullback whether
or not the pullback recovers. Foreign money sells into declines, uniformly, and
at top-10 aggregate resolution there is no capitulation signature separating the
falls that end from the falls that continue.

Exploratory features (not corrected, reported for completeness): delta_bumn
p 0.245, delta_imbalance p 0.128, delta_conc p 0.185, delta_local p 0.339.
Nothing near significance.

**Also settled by probing rather than assumption:** the endpoint hard-caps at
top-10 brokers. `limit`, `top`, `rows`, `n`, `count`, `show`, `all` and `page`
all return the same 11 codes. The full member-by-member tape is not reachable
from this source, which bounds what any flow test here can see.

**A correction to Result 96's closing claim.** It said the exclusion list was
"complete". It was complete for *level-based* flow; trajectory was a separate
hypothesis that had not been tested and was folded in without warrant. It has
now been tested and also fails, but the claim was ahead of the evidence when it
was made.

# Part XXVII — Painting the legs, and what the paint is worth

The request was to reproduce the hand-drawn green/red segmentation, then derive
buy and sell signals from it, causally, across IDX big caps.

## Result 98 — the picture reproduces above 90%, once two things are right

**Unadjusted prices.** TradingView plots the raw close. ADRO's 2022-23 dividends
were large enough that the adjusted series runs 517-2,540 where the chart runs
1,645-4,140. Fitting the adjusted series paints a different picture than the one
on screen, and every earlier part of this project used adjusted prices.

**A 12% weekly swing** reproduces the hand-drawn count: 19-20 legs over the
charted window against ~20 drawn.

The target is a drawing on FINISHED data, so a live zigzag reproduces every
closed leg exactly and repaints only the leg in progress. The measurement that
matters is therefore how long a bar's colour takes to stop changing:

| bar age | ADRO | across 59 large caps |
|---|---|---|
| 1 week | 77.1% | 70.9% |
| 4 weeks | 93.9% | 87.0% |
| **6 weeks** | **97.7%** | **92.4%** |
| 8 weeks | 99.2% | 95.2% |
| 13 weeks | 100% | 98.2% |
| 26 weeks | 100% | 99.8% |

**97.2% of all bars sit in a closed leg at any moment.** So the picture is
reproduced above 90% for anything older than six weeks, on every big cap.
`scripts/paint_chart.py` draws closed legs solid and the running leg dashed,
because painting the live leg solid claims a certainty that does not exist.

## Result 99 — the live leg, and a lesson about sample size

Forecasting the colour of the *unfinished* leg is the real problem. The
single-name model failed instructively: 59 features on 437 weekly rows reached
**100% training accuracy and 49% test accuracy**, and collapsed to predicting
"down" on 89% of out-of-sample bars. That is not a feature problem, it is a row
problem.

Pooling across all 59 big caps - training one model on 30,000 weekly bars and
scoring it per name - fixed the collapse:

| | model | 13-week average | lift |
|---|---|---|---|
| median | 68.8% | 67.1% | +1.7% |
| mean | 69.3% | 66.4% | +2.9% |

with the predicted up-share matching the actual one exactly (61% / 61%). Better
than the baseline, and nowhere near the 92-100% of the closed legs, which is
exactly as it should be: closed legs are history, the live leg is a forecast.

## Result 100 — the signals do not beat holding, and one of them briefly appeared to

Buy when a green leg opens, sell when a red one does. Across 40 big caps at
Rp10T+, full history from 2012, costs charged:

| band | median excess over holding | beats holding | median drawdown | holding |
|---|---|---|---|---|
| 12% | -1.85% | 42% | -64% | -76% |
| 15% | -2.61% | 35% | -63% | -76% |
| 20% | -2.00% | 30% | -66% | -76% |
| 25% | -1.54% | 45% | -67% | -76% |

And cross-sectionally - own the big caps whose leg is currently green:

| book | CAGR | vs equal-weight |
|---|---|---|
| equal-weight all big caps | +9.8% | — |
| all green, 25% band | +4.8% | -5.0% |
| top 15 green, 25% band | +3.8% | -6.0% |
| top 8 green, 25% band | +2.3% | -7.5% |

**Every variant loses.** The painter is a description of what happened, not a
signal about what will.

**The near miss worth recording.** The first version of that cross-sectional
test returned **+78% to +87% CAGR** across every band. It was a look-ahead: the
holding state was lagged one week but the RANKING was not, so the book selected
on the same bar's price and then collected that same bar's return - it picked
whatever had just gone up. Lagging every input drops it to the table above.
`tests/test_legpaint.py` now contains a regression test that builds a panel
where the unlagged book must look impossibly good and the lagged one must not.

**The rule this makes explicit: in a cross-sectional book, every input to the
selection must be lagged, not merely the position state.** One unlagged input is
enough to manufacture an 87% CAGR.

## Result 101 — the half-chart test, and why "success rate" means two things

Hide the second half of every big cap's chart, paint it forward one bar at a
time, and compare with the painting made from the whole series. 43 names,
14,854 hidden weeks.

**The no-cheating proof.** Painting the visible half using ONLY the visible half
reproduces the full-chart painting of that half **exactly, on 43 of 43 names**.
A rule that peeked would differ somewhere.

**On the hidden half**, two different questions get two different numbers, and
conflating them is how people fool themselves:

| question | answer |
|---|---|
| once a leg has CLOSED, does its colour ever change? | **never — 100%, 43 of 43** |
| how much of the chart is closed at age k? | 6.9% at 4w, 15% at 6w, 49.5% at 13w |
| including the still-running leg, is a bar's colour final? | 87% at 4w, 92.4% at 6w, 98.2% at 13w |

The first is a property of the construction: a zigzag never repaints a leg that
has already ended. The second is the honest limit: at any moment, most of the
*recent* chart is still inside the running leg and is provisional by nature.

## Result 102 — the trigger price is arithmetic, not a forecast

A reversal band has an exact flip level at every moment:

    inside a green leg -> turns red at   running high x (1 - band)
    inside a red leg   -> turns green at running low  x (1 + band)

So "when does the indicator light up" has a precise answer available now. As of
the last weekly close, of 43 big caps at Rp10T+: **32 in green legs, 11 in red**.
The nearest triggers are JSMR (-5.2% to a sell), GGRM (-5.4%), BMRI (-5.5%),
PGAS (+4.1% to a buy), TLKM (+6.0%), ASII (+7.1%). ADRO sits in a green leg at
2,530 and turns red at 2,235, -11.7% away.

**What is knowable and what is not, stated exactly:** the level is arithmetic and
certain; whether price reaches it is neither, and Result 100 measured what acting
on these flips is worth - it loses to holding by 1.5-2.6% a year. The triggers
tell you where the picture changes, not that changing with it pays.

## Result 103 — daily bars keep the accuracy and reach it sooner

The zigzag rule is scale-free, so moving from weekly to daily changes only how
many bars a swing of a given size takes to resolve. What had to be re-measured is
the settling time, in days, across 20 large caps with a median 2,860 sessions
each.

| band | legs | no-cheat | 5d | 10d | 20d | 30d | 45d | 65d |
|---|---|---|---|---|---|---|---|---|
| **8%** | 119 | **20/20** | 84% | **92%** | 97% | 99% | 100% | 100% |
| **10%** | 85 | **20/20** | 80% | 88% | **96%** | 99% | 100% | 100% |
| 12% | 65 | 20/20 | 76% | 84% | 93% | 97% | 99% | 100% |
| 15% | 45 | 20/20 | 73% | 80% | 89% | 94% | 97% | 99% |
| 20% | 28 | 20/20 | 72% | 77% | 83% | 88% | 93% | 97% |

Where each band first clears 90%:

| band | days | equivalent |
|---|---|---|
| 8% | **10** | 2 weeks |
| 10% | 20 | 4 weeks |
| 12% | 20 | 4 weeks |
| 15% | 30 | 6 weeks |
| 20% | 45 | 9 weeks |

**Daily does not merely hold the weekly accuracy - it reaches it in half the
calendar time.** The weekly 12% band needed six weeks (30 sessions) to clear
90%; the daily 12% band clears it in 20 sessions and the daily 8% band in 10.
Finer bars resolve a swing sooner because the confirming move is detected
whenever it happens rather than only at a Friday close.

The no-cheating check passes on every band and every name: painting the first
half with only the first half reproduces the full-series painting of it exactly,
20 of 20, at all five bands.

**The cost is arrow count.** 119 legs at 8% against 28 at 20%, on the same
history. Result 100 already measured what acting on flips is worth, and more
flips means more of the whipsaw that made those numbers negative. For *reading*
the chart the 8-10% daily band is the better instrument; for *trading* it, the
extra resolution buys nothing this project has been able to measure.

**Recommended daily settings:** 10% fast / 20% slow. That clears 90% in four
weeks, gives 85 legs over eleven years, and keeps the fast/slow agreement
meaningful. `scripts/show_arrows.py --daily --fast 0.10 --slow 0.20` renders it.

## Result 104 — the daily/weekly hybrid does not exist, and the reason is a base rate

Daily settles faster than weekly, so the natural idea is that daily should warn
of a weekly turn before the weekly band confirms it. Two measurements, and the
second cancels the first.

**The lead looks total.** Across 25 large caps and 1,003 weekly flips, the daily
8% band had already flipped the same way before **100%** of them, by a median of
9 days.

**The lead is vacuous.** The daily band flips 2,863 times to produce those 1,003
anticipations. What matters is precision, against the right base rate:

| | |
|---|---|
| P(weekly flips your way within 60d, **given a daily flip**) | **46.1%** (n=2,863) |
| P(weekly flips that way within 60d, **any random week**) | **47.5%** (n=12,941) |
| **lift** | **0.97x**, z = -1.4, p = 0.15 |

**A daily flip carries no information about whether the weekly will follow.** It
anticipates every weekly turn the way a stopped-often clock anticipates every
hour. Reporting the 100% without the base rate - which this project did for one
message - is the error the base rate exists to prevent.

**And the naive hybrid fails for a separate, structural reason.** Running the
weekly 12% threshold on daily closes scores a flat **67%** against the finished
weekly picture at every age, from 1 week to 13. Accuracy that does not improve
as data arrives is the signature of measuring the wrong object: a 12% band on
daily closes tracks intra-week extremes the weekly picture never contains, so it
is not reproducing the weekly chart sooner, it is drawing a different chart.

| bar age | weekly only | naive hybrid |
|---|---|---|
| 1w | 74.8% | 67.4% |
| 4w | 89.1% | 67.0% |
| 6w | **92.9%** | 66.9% |
| 13w | 98.8% | 67.7% |

## Result 105 — what daily is actually for: CUAN side by side

The two timeframes are not competing views of one thing. They are different
instruments, and CUAN shows it cleanly:

| | closed legs | settles to 90% | flips at |
|---|---|---|---|
| **weekly, 12% band** | 30 | 4 weeks (95%) | 726, -12.0% |
| **daily, 8% band** | 74 | 5 days (94%) | 800, -3.0% |

The daily painter reaches certainty in **5 days** where the weekly needs **4
weeks**, and it does so on its own legs - 74 of them against 30. The weekly
band's stop sits 12% away; the daily band's sits 3% away.

So the choice is not accuracy, it is **which question you are asking**. If the
position is a multi-month one, the weekly picture is the relevant one and the
daily flips inside it are noise you have already been shown costs money to
trade. If the position is a multi-week one, the daily picture settles four times
faster and its stop is four times tighter.

**Neither improves the other.** That is the finding, and it is the opposite of
what the hybrid was built to show.

## Result 106 — the delay test, and why the single-name result is luck

Real fee schedule applied: 0.28% one side, 0.18% plus the 0.10% sale tax on the
other. Both readings of that give the same **0.56% round trip**, so the
ambiguity does not change any answer.

Then the honest question: nobody acts the instant the colour changes. Rp10m on
CUAN's daily 8% arrows, with N extra sessions between signal and fill:

| delay | 1 year | 2 years | 3 years |
|---|---|---|---|
| 0 days | -25.6% | +15.0% | +103.9% |
| 1 day | -6.5% | +31.5% | +76.4% |
| **2 days** | **+76.3%** | **+43.7%** | **+110.9%** |
| **3 days** | **-32.9%** | **-13.7%** | +59.7% |
| 5 days | +10.7% | +18.0% | +121.3% |
| 10 days | -43.2% | -35.8% | -11.4% |
| buy & hold | -49.5% | +0.3% | +60.3% |

**Being two days late returns +76.3% over one year. Being three days late
returns -32.9%.** Same rule, same stock, same fees - a 109-point swing from a
single day of hesitation, and the ordering is not monotone in either direction.

| window | best | worst | spread |
|---|---|---|---|
| 1 year | +76.3% | -43.2% | **119.5 points** |
| 2 years | +43.7% | -35.8% | 79.5 points |
| 3 years | +121.3% | -11.4% | 132.7 points |

**This is not an edge being eroded by latency. It is an outcome dominated by
which handful of trades you happen to catch.** A rule with a real edge degrades
smoothly as execution slows; this one scatters. With 32 round trips over three
years and a 41% win rate, a couple of caught-or-missed moves decide everything.

The one monotone finding: **a 10-day delay is bad in every window** (-43%, -36%,
-11%). Beyond about a week the signal is simply stale.

**What this settles about single-name arrow trading.** Result 100 measured it
losing to buy-and-hold by 1.5-2.6%/yr across 40 large caps. This explains the
mechanism at the level of one account: the dispersion swamps the mean. Quoting
any single number from the table above - including the flattering +121.3% - as
"what the strategy returns" would be quoting a draw from a very wide
distribution and calling it an expectation.

## Result 107 — the stop works; the returns are just concentrated in five trades

"Shouldn't cutting the loss the moment it goes red keep the losses small?" It
does. That was worth checking rather than assuming, and checking it corrects the
framing of Result 106.

CUAN daily 8% band, 36 round trips, fees at 0.56%:

| | | |
|---|---|---|
| winners | 16 (44%) | average **+53.2%**, biggest +292% |
| losers | 20 (56%) | average **-8.1%**, worst -19% |
| **mean trade** | | **+18.60%** net |
| **median trade** | | **-2.42%** net |

**The stop does exactly what it is supposed to.** The mean loss is -8.1% against
an 8% band, and the worst loss in three and a half years is -19%. Eleven of
twenty losses exceed the band slightly, for a mechanical reason worth knowing:
the band trails the HIGH SINCE ENTRY, not the entry price. You buy 8% above the
low, so if the leg adds little before turning, the stop sits below where you
bought.

**So the edge is real and strongly positive - and it lives in five trades.**

| | |
|---|---|
| compounding all 36 trades | **36.4x** |
| remove the single best | 9.31x (26% survives) |
| remove the top 2 | 3.50x (10% survives) |
| remove the top 3 | 1.64x (**4.5% survives**) |
| remove the top 5 | 0.69x (a loss) |

The top three trades were **+292%, +166%, +114%**. They are 8% of the trades and
carry **96% of the total return**. Trades over +50% arrived in 2023-03, 2023-08,
2024-01, 2025-06 and 2025-10 - five moments in three and a half years.

**This resolves the apparent contradiction in Result 106.** A one-year window
returning -25% is not the stop failing. It is a year containing none of those
five trades, in which the account pays the 8% stop repeatedly and collects
nothing. The mean is +18.6% per trade; the median trade loses 2.4%. Both numbers
are true and the gap between them is the entire behaviour of the strategy.

**Correcting Result 106.** It said the dispersion was "the strongest evidence yet
that the rule has no reliable edge on a single name". That was wrong as stated -
the per-trade expectancy is positive and large. The correct statement is that the
edge is real but so concentrated that no one-year sample estimates it, and an
account can be down 25% in a year while the underlying rule is working exactly
as designed.

**The practical consequence, which is not optional:** a strategy whose return
lives in 8% of its trades cannot be run on one name. Miss the five and you have
a losing system. That is the argument for spreading the same rule across many
names - not to raise the mean, but to raise the chance of being present when the
few trades that matter arrive.

## Result 108 — "remove the best 5 and it loses" is true, and it is true of holding too

The objection: *"ain't no way, off the daily indicator you lose if you remove
the best 5."* It is the standard way to kill a backtest and it deserved a proper
test rather than a defence.

**Why the obvious version of the test proves nothing.** Removing the best five
trades from any long-only equity strategy produces a loss, because equity
returns are multiplicative and lopsided. The test only counts as evidence if the
strategy is *more* concentrated than the alternative it is being measured
against. So it has to be paired:

> cut the timeline at the strategy's own trade boundaries; that gives N
> segments; the strategy's total is the product of its N segment multiples and
> holding's total is the product of *its* N multiples **over the same dates**;
> now drop the best k from each side and compare what survives.

Same stock, same dates, same number of factors, same removal. `scripts/concentration.py`.

**On CUAN the rule survives the removal better than holding does.**

| remove best | strategy | buy & hold |
|---|---|---|
| none | 40.27x | 30.11x |
| 1 | 10.33x | 8.09x |
| 2 | 3.90x | 3.10x |
| 3 | 1.83x | 1.27x |
| **5** | **0.77x** | **0.41x** |
| 8 | 0.34x | 0.15x |

Removing the best five *does* turn it into a loss — that part of the objection is
simply correct. But it turns holding the same stock over the same dates into a
worse loss. On this name the test does not separate them.

**Holding is not exempt, and not by a little.** Across the 41 big caps that rose
over their history, the best **5% of weeks carry a median 406%** of everything
holding them produced — the other 95% of weeks are a net loss. The most
broadly-earned name in the whole sample is BBCA at 147%. There is no IDX large
cap whose return is spread evenly across its weeks.

**But one name is an anecdote, so the same paired test was run on all 46.** This
is where the objection lands.

| | |
|---|---|
| removing each side's best 5 | strategy ahead on **13/46** names (28%) |
| removing each side's best 10% (fraction-matched) | strategy ahead on **15/46** |
| strategy less concentrated than holding | **4/46** names |
| median share of all *gains* in the best 5 segments | strategy **39%**, holding **31%** |

**The objection largely stands on the universe.** On 42 of 46 big caps the
rule's gains are more concentrated than those of simply holding the same stock,
and after the same removal it is behind holding on 33 of 46. CUAN survives the
removals better but is *still* the more concentrated of the two there (72% of its
gains in the best 5 segments against 65% for holding) — it starts from a higher
base, which is not the same thing as being robust.

**Two measurement notes, because the first version of this was wrong.**

1. An inline version of this test printed a hardcoded conclusion that the
   strategy "is MORE concentrated than the thing it is trading" while its own
   output showed 98% against 98%. Both the conclusion and the number were
   artefacts. The verdict is now computed from the data in the same run, and the
   script says so in its docstring.
2. Share-of-*net*-total is unusable across names: the denominator passes through
   zero, producing values from -362% to +8,856% on real names. Median-of-that is
   meaningless. The cross-name figure is now share of all **gains**, which is
   bounded in [0,1] and always defined. On one name the net figure is still
   printed, because ">100%" there has a real reading: everything outside the best
   few segments lost money.

**What this settles and what it does not.** It settles that "remove the best few
and it loses" is not on its own proof of a broken rule — the same test breaks
buy-and-hold on every large cap in the sample. It does not rescue the daily
indicator. On the universe the rule is the more concentrated of the two, it beats
holding outright on only 8 of 46 names, and Result 100 still measures it losing
to holding by 1.5-2.6% a year. The concentration objection was aimed at the right
target; it just needed the paired version to land, and in the paired version it
lands on the universe rather than on CUAN.

# Part XXVIII — What the band can and cannot do, settled

## Result 109 — you cannot miss a big move; you can only pay a toll on it

The question: *"what is the expected miss of the best X if you have 90% accuracy?
It's impossible to completely miss right? Just maybe a late entry or sell."*

That intuition is correct, and it is better than correct — it is provable. The
"remove the best 5" test of Result 108 deletes those trades. That is the wrong
model of failure for a band rule, because the rule's failure mode is arithmetic,
not probabilistic.

**The law.** The rule flips green when price rises `b` off the running low and red
when it falls `b` off the running high. For a leg from low `L` to high `H`, with
`M = H/L - 1`:

```
you cannot buy below   L(1+b)        the flip is what buys you in
you cannot sell above  H(1-b)        the flip is what sells you out

best possible capture = (1+M)(1-b)/(1+b)
break-even leg size   = M* = 2b/(1-b)        b=8% -> 17.39%
```

Three consequences, none of them a hit rate:

1. **A move larger than the band cannot be missed.** If price rises `b` off the
   low, the state flips — by construction. Not 90% of the time. Always.
2. **The cost is a fixed toll of `(1+b)/(1-b)` on price**, not a fixed share of
   the move. It eats 8% of a +292% leg and all of a +15% one.
3. **Every leg below `M*` is a guaranteed loss before fees.**

**Measured on 46 big caps, 2,607 up legs** (`scripts/capture_toll.py`):

| leg size | legs | median move | captured | ceiling | trade P/L | wins | missed |
|---|---|---|---|---|---|---|---|
| < 17.4% (break-even) | 1,325 | 12% | 10% | 30% | −6.7% | 5% | — |
| 17.4–25% | 484 | 21% | 50% | 59% | −1.2% | 40% | 0 |
| 25–50% | 588 | 33% | 66% | 73% | +8.6% | 80% | 0 |
| 50–100% | 176 | 61% | 80% | 84% | +31.1% | 97% | 0 |
| 100–200% | 26 | 114% | 86% | 90% | +66.4% | 100% | 0 |
| > 200% | 8 | 298% | 92% | 94% | +226.0% | 100% | 0 |

**Of 2,591 up legs larger than the band, the number never flagged is 0.** Of the
230 biggest legs (five per name, median +57%, largest +2,385%): 0 missed, 0
flagged only at the peak, median capture **78%** of the log return against an 83%
ceiling, and the trade that started in them returned a median **+27.7%** with
**97% winners**.

So the expected miss on a *same-band* leg is 22% of it, never the leg.

### Correction to Result 109 — an adversarial re-derivation found three errors

An independent check was asked to refute this law and returned "wrong" with
deterministic counterexamples. Three of its objections hold, and all three are
verified here on 2,571 real round trips across the 46 large caps.

**1. The toll is not fixed.** `(1+b)/(1-b)` bounds the *signal price* only: the
flip fires on the first close at or above `min(1+b)`, overshooting by however far
that bar travelled, and the fill is the bar *after* — which can come back under
the trigger. Realised tolls therefore straddle the figure rather than respecting
it, with the typical one well above.

| | p25 | median | p75 | p90 | over 8% |
|---|---|---|---|---|---|
| entry overshoot | 7.6% | **9.4%** | 12.2% | 15.9% | **70%** |
| exit shortfall | 7.0% | **8.8%** | 10.8% | 13.4% | **63%** |

Realised round-trip toll: **median 20.6% of price, p90 31.7%**, against the
algebra's flat 17.4%.

**2. The break-even is ~21%, not 17.4%,** and nothing is "guaranteed". Where
round-trip P/L actually crosses zero:

| move spanned | trips | median P/L | win rate |
|---|---|---|---|
| 10–17% | 812 | −6.5% | 6% |
| 17–20% | 172 | −2.5% | 26% |
| **20–25%** | 291 | **+0.0%** | **50%** |
| 25–30% | 195 | +3.3% | 68% |
| > 40% | 330 | +25.5% | 98% |

17.4% is the algebraic floor, reachable only with perfect fills. Fills gap, so a
sub-break-even leg can still pay — the direction of the claim survives, the
certainty does not.

**3. A real bull move is not one leg, and this is the one that changes the
headline.** A leg measured at the rule's own band contains no `b`-sized
retracement *by construction*, so it is exactly one round trip — which makes any
capture figure computed on it near-circular. Against 203 real bull moves found
with a 30% zigzag:

| move size | moves | 8% legs inside | captured | one-leg formula says |
|---|---|---|---|---|
| 50–100% | 98 | 4 | 46% | 70% |
| 100–200% | 59 | 5 | 59% | 81% |
| 200–400% | 28 | 9 | 55% | 88% |
| > 400% | 18 | 12 | 65% | 92% |

**A real bull move holds a median of 5 separate 8% legs and pays 5 tolls. Capture
is 55%, not the 78–92% reported above.**

**Also corrected:** the original text said the toll "eats 8% of the log move" on a
+292% leg. This file's own `ceiling_fraction(2.92, 0.08)` returns 0.8827, so the
correct figure is **11.7%** — the original was wrong by a third.

**What survives untouched** is claim 1, the only one that was ever exact: a move
larger than the band cannot be missed, verified with zero violations on 1.3m red
bars across 719 names, holding through gaps. The rule cannot miss a big move; it
keeps a bit over half of one.

**Where the losses actually are:** the 1,325 legs below break-even — 51% of all
up legs — which lose by construction no matter how accurately they are called.

**Being late costs little.** Delaying every decision by 5 bars moves capture on
big legs from 78% to 75%; by 10 bars, to 68%. The failure mode is not lateness,
it is **speed**: the 446 legs flagged only at their peak last a median 4 bars
against 13 for the rest.

*Two measurement traps fixed while building this.* Capture measured to the peak
charges only the entry toll — the exit flip lands after the peak by construction
— so it must be compared to an entry-only ceiling, not the round-trip one. And
the realised round trip is not confined to its leg (its exit often lands inside a
later move), so it is reported in money, never as a fraction of the leg.

## Result 110 — the band cannot be optimised, because IDX legs are a random walk

If the losses live in sub-break-even legs, the obvious fix is a band whose legs
clear their own toll. There is none, and the reason is decisive.

**The fixed point.** Raising `b` makes the surviving legs bigger, but it raises
`M* = 2b/(1-b)` just as fast. Measured across 46 big caps on daily bars, the ratio
of median leg to break-even never gets meaningfully above 1:

| band | break-even | legs | median leg | leg/M* | above M* |
|---|---|---|---|---|---|
| 3% | 6.2% | 196 | 7.4% | 1.20 | 60% |
| 5% | 10.5% | 106 | 10.9% | 1.03 | 52% |
| 8% | 17.4% | 56 | 17.3% | **0.99** | 49% |
| 12% | 27.3% | 32 | 26.1% | 0.96 | 47% |
| 20% | 50.0% | 14 | 41.7% | 0.83 | 38% |
| 30% | 85.7% | 7 | 62.6% | 0.73 | 37% |

**The null that settles it.** Simulating a driftless random walk at each name's
own volatility reproduces that profile almost exactly. Median gap in leg/M*,
real minus walk:

| timeframe | median gap |
|---|---|
| daily (46 names) | **−0.001** |
| 4h (12 names) | −0.032 |
| 1h (12 names) | +0.008 |
| 15m (65 names) | no significant positive result |

On 15m the only significant result is **negative** (2% band, gap −0.056,
p = 0.0035, passing a Bonferroni threshold of 0.0125). A promising +0.053 on a
12-name sample vanished when the sample was widened to 65 — a reminder that a
gap that size on a dozen names is noise.

**So "the legs are bigger than the band" is not evidence of anything.** A random
walk does it too. Out of sample, no selector beats holding: grid-fitted −0.118,
fixed 8% −0.134, buy-and-hold **+0.276**.

**The break-even law is a good explanation and a bad selector.** Choosing the band
by it picks 3% and returns **−1.453** out of sample, because the ceiling is a
bound, not an expectation — whipsaw at small bands never reaches it. Recorded as
the negative it is.

## Result 111 — combining timeframes lowers the toll and does not make money

The tolls need not come from the same band. Enter on a fast one and exit on a
slow one and the break-even becomes `(1+b_fast)/(1-b_slow) - 1`. With a 3% hourly
entry inside a 12% daily-confirmed leg, that is **17.0% instead of 27.3%**.

The arithmetic worked. The money did not. 46 big caps, ~4,300 hourly bars each,
fees charged, daily state lagged to the last *completed* session:

| rule | median log | vs hold | same-exposure null | beats null |
|---|---|---|---|---|
| fast only (3%) | −0.580 | −0.377 | −0.094 | 15% |
| slow only (12%) | −0.177 | +0.027 | −0.094 | 30% |
| both green | −0.410 | −0.207 | −0.047 | 15% |
| **fast in / slow out** | **−0.173** | +0.031 | −0.080 | 34% |
| buy & hold | −0.204 | — | — | — |

**The right null.** Holding *lost* 18.4% over this window, so any rule that sits
in cash looks good for free. The fair benchmark is random timing at the same
exposure — `exposure × hold`. **Every rule loses to it.** The best combination
beats the best single timeframe by +0.025 log on 57% of names, which is a coin
flip, and misses its own null by −0.083.

A cheaper bet at unchanged odds is still a losing bet.

## Result 112 — layer 2 cannot reach the timeframes it would need to

The broker-summary layer was audited against what a multi-timeframe system would
actually require.

**The source used here is end-of-day, so it cannot resolve a 15m, 1h or 4h bar.**

*Correction, added after the fact.* This originally read "it cannot be intraday,
ever — there is no price at which it resolves a 15m, 1h or 4h bar." That was an
overclaim, and it conflated the free daily-summary page this repo reads with the
underlying exchange data. IDX's trade feed carries the **buying and selling broker
code on every transaction**, intraday — which is precisely why broker-summary and
running-trade features exist in Indonesian broker apps at all, and why
"bandarmology" is an Indonesian phenomenon rather than a global one. Member firms
receive that feed as an entitlement.

So the honest statement is narrower and more useful: **every limitation measured
in this Result is a limitation of the source currently in hand, not of the data
class.** Specifically —

| limitation found | is it fundamental? |
|---|---|
| end-of-day only | **No.** The exchange feed is tick-level with broker codes. |
| top 10 brokers per side | **No.** The full rekap exists upstream; the cap is the free page's. |
| outcome-conditioned sample | **No.** That is this repo's own sampling artefact. |
| daily series for one name | **No.** A consequence of the above three. |

What remains fundamental is only the measured result: on the question actually
asked — does flow into a drawdown predict the recovery — two adequately powered
pre-registered replications returned d = 0.002 and d = −0.005. That is evidence
about a hypothesis, not about the data class, and a better feed would let the
hypothesis be re-asked at intraday resolution for the first time.

**The cache is not daily and not a sample of history.** Of 1,795 files, **1,703
are range aggregates** — one table over a `start..end` window stamped with a
single date equal to the window's end; all 1,703 checked, none contains more than
one distinct date. Only 92 are true single-day files, covering 5 tickers: BBCA
(60 sessions, 94% dense) and ADRO/ANTM/BUMI/MDKA (8 sessions each). A daily
net-flow series is buildable today for **exactly one name**.

Worse for inference, the windows are **outcome-conditioned**: 900 files match a
`(peak_date, signal_date)` pair from `reports/pullback_events.csv` exactly, and
800 more are half-windows from the trajectory test. It is a sample of pullback
events, not a sample of history.

**And it has already been measured, twice, at zero.** Result 96: a 300-event pilot
found `foreign_net` d = 0.257, p = 0.047; the pre-registered replication on 600
untouched events returned **d = 0.002, p = 0.489**. Result 97: the trajectory
version, **d = −0.005, p = 0.519**. Result 17: 322,827 rows gave a 60-day IC of
−0.0254 (t = −10.88) that survived every control — and a long-short spread of
+0.15%, t = 0.76, with U-shaped deciles. Result 19 bounds the whole exercise:
BBCA's cumulative top-10 net over 52 sessions came to −2,808,171 lots where a
complete rekap must sum to zero.

## Result 113 — regime gating does not clear the null either

The last untested place. Gating the daily band on a regime condition, 44 names ×
2,862 sessions, 2015-01-02 to 2026-08-19, fees charged, every gate lagged:

| filter | median log | exposure | same-exposure null | edge vs null | beats null |
|---|---|---|---|---|---|
| none | −0.249 | 53% | +0.095 | −0.318 | 39% |
| above 200-day MA | −0.319 | 31% | +0.056 | −0.320 | 27% |
| top-third relative strength | −0.416 | 23% | +0.035 | −0.470 | 18% |
| both | −0.278 | 19% | +0.030 | −0.307 | 23% |
| market regime | −0.132 | 38% | +0.068 | −0.258 | 34% |
| **buy & hold** | **+0.169** | 100% | — | — | — |

**No gate clears its own null.** The market-regime gate is the least bad (−0.258
against −0.318 ungated) and still loses to spending the same exposure at random.

**An important limit on this result.** What was tested is *price-derived proxies*
for regime — moving averages, relative strength, an equal-weight market filter.
A point-in-time news and fundamentals dataset does not exist in this repo, so
**layer 1 proper has not been tested**, and this result should not be read as
having tested it. Testing it honestly would need earnings, guidance and corporate
actions timestamped at the moment they became public, not as they are known now.
Everything else here is a measurement; that one is an open question.

### The state of the market this was measured in (19 Aug 2026)

Layer-1 research, dated and sourced, for context on every number above:

- **IHSG 6,394** — peaked 9,134.70 on 2026-01-20, fell 41.5% to 5,342.14 on
  2026-06-08, rebounded 18.6%, still **30.6% below the peak** and 15.2% below its
  200-day average. This is a bear-market rebound, not an uptrend.
- **BI rate 5.75%, tightening** — hiked 50bp on 2026-05-20 to defend the rupiah
  (primary source: bi.go.id). Rupiah at a record-weak ~Rp17,850/USD. Any model
  assuming an easing cycle for 2026 is mis-specified.
- **MSCI deleted six names** — AMMN, BREN, TPIA, DSSA, CUAN, AMRT — announced
  2026-05-13, effective 2026-06-01, on free-float grounds. **All six bottomed
  before the effective date** (21–29 May), so a rule keyed to the event date would
  have been badly wrong.
- **IDX raised minimum free float** from 7.5% to 15%. CUAN sits at 14.9%,
  BREN 12.3% — both non-compliant, with a 'P' notation live since 2026-08-03.
- **Only 11 of 46 large caps are above their 200-day average.** Of the researched
  set only ADRO (+39% YTD) and AADI are.
- **Commodities do not move as one factor**: copper at an all-time high
  (+46% yoy), gold US$4,339/oz, coal recovered ~22% off its January low — but
  nickel is falling on Indonesian supply expansion.

### Where this leaves the three layers

| layer | status |
|---|---|
| 3 — technical | **Measured, no edge.** Leg structure matches a random walk at 15m/1h/4h/daily; every rule and combination loses to random timing at matched exposure. |
| 2 — broker flow | **Measured at zero, twice, pre-registered** — but on an end-of-day, top-10-truncated, outcome-conditioned source. The exchange feed itself is tick-level with broker codes, so the hypothesis has never been asked at the resolution that would matter. |
| 1 — news/fundamentals | **Untested.** Price-derived proxies for it fail. A point-in-time dataset would be needed to test the real thing. |

What survives is narrow and worth keeping: the band rule cannot miss a move
bigger than its band, it never repaints, and its trigger price is exact and known
in advance. That makes it a good **instrument** — it tells you where you are. It
is not evidence about where price goes next, and every attempt in this Part to
make it into one has been measured and has failed.
