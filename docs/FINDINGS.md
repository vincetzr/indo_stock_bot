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

## What this does and does not establish

**Does:**
- The contrarian accumulation thesis, in its price-only form, is refuted on IDX.
- A simple trend-following composite has a genuine, statistically significant,
  economically meaningful cross-sectional edge at a 60-day horizon out-of-sample.

**Does not:**
- **That the broker-flow thesis is wrong.** It was never tested — no real broker
  summary was obtainable (`idx.co.id` WAF-blocked, Stockbit auth-gated, GoAPI
  key-gated). The `momentum_plus_flow` profile exists precisely to run that
  experiment when data is connected. That remains the open question.
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

```bash
# reproduce everything above
idxbot backtest --universe all --providers none --profile accumulation --out reports/obs_acc.csv
idxbot backtest --universe all --providers none --profile momentum     --out reports/obs_mom.csv
idxbot evaluate  --observations reports/obs_mom.csv --split --components
idxbot portfolio --observations reports/obs_mom.csv --split --top-n 10 --horizon 60
python3 scripts/robustness.py reports/obs_mom.csv
```

---

*A backtest that contradicts the strategy is more valuable than one that
flatters it. The first result here cost the original thesis; the second was only
trustworthy because the holdout was left untouched until the end.*
