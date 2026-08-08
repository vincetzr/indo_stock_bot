# Findings from 25 years of real IDX data

Two results, both on genuine exchange data, both reproducible from this repo:

1. **The accumulation score is inverted.** Buying quiet bases lost to buying
   everything else — strongly before 2017, decaying to noise after.
2. **A momentum score built from that diagnosis survives out-of-sample.** On a
   holdout period never used for selection, the top quintile beat the bottom by
   **+3.83% over 60 days (t = 7.00)** — but with 4 losing years in 10.

Result 2 is not a licence to switch on the screener and stop thinking. Read §5.

---

## What was tested

| | |
|---|---|
| Observations | **56,745** |
| Tickers | 66 |
| Period | 2001-07-31 → 2026-08-07 (**25 years**) |
| Data | Real Yahoo daily OHLCV + real IHSG. **Nothing simulated.** |
| Mode | `price-only` (`--providers none`) — no broker data anywhere |
| Sampling | Every 5th bar, 300-bar warm-up, no look-ahead (unit-tested) |
| Split | Chronological. Train 2001-07 → 2017-01. **Holdout 2017-01 → 2026-08.** |

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
| Train 2001–2017 | **−0.0566** | −7.62 | −0.0457 | −6.28 |
| Holdout 2017–2026 | −0.0017 | −0.25 | −0.0204 | −2.93 |

Strongly negative for the first sixteen years, then decaying to roughly nothing.
Either way it never earns its keep as a *buy* signal.

The pooled bucket test agrees: the lowest-scoring names returned +6.86% over 60
days versus +4.49% for the signal cohort, monotonically across all five buckets.
Wyckoff phase E — which the planner blocks as "a chase" — was the best state at
+10.67%/60d, against +3.38% for the spring setup the engine rates highest.

### Component diagnosis (training half only)

| Component | 20d IC | t | Verdict |
|---|---|---|---|
| `relative_strength` | **+0.0380** | +4.70 | the only one helping |
| `wyckoff` | −0.0202 | −2.64 | hurting |
| `range_compression` | −0.0293 | −3.90 | hurting |
| `obv_divergence` | −0.0398 | −5.14 | hurting |
| `volume_dryup` | −0.0476 | −6.30 | hurting most |

Every contrarian component had a negative IC. The single momentum-flavoured one
was positive. The composite wasn't noise — it was **systematically backwards**.

---

## Result 2 — the momentum profile survives the holdout

Component selection used the **training half only**, then a coarse, deliberately
unfitted weighting (0.30 / 0.25 / 0.25 / 0.20) across 12-1 momentum, relative
strength, trend persistence, and proximity to the 52-week high. The holdout was
not looked at until the profile was frozen.

| Profile | Train 20d IC (t) | **Holdout 20d IC (t)** | **Holdout 60d IC (t)** |
|---|---|---|---|
| `accumulation` | −0.0566 (−7.62) | −0.0017 (−0.25) | −0.0204 (−2.93) |
| `momentum` | +0.0538 (+6.73) | **+0.0261 (+3.47)** | **+0.0411 (+5.52)** |

Holdout quintile spread (top minus bottom, within each date):

| Horizon | Top | Bottom | Spread | t | Dates positive |
|---|---|---|---|---|---|
| 20d | 2.12% | 1.14% | **+0.98%** | 3.16 | 52.2% |
| 60d | 6.24% | 2.40% | **+3.83%** | 7.00 | 57.3% |

At 60 days that is ~+3.43% net of the configured 0.40% round trip.

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
- **Do not size a strategy off the 20d edge.** It is +0.98% gross against a
  0.40% round trip. The horizon that works is 60 days.
- **Re-run this with real broker summary.** `--profile momentum_plus_flow`
  combines trend with institutional flow. If broker flow adds information beyond
  price, the holdout IC will rise above +0.041. That is the experiment worth
  paying a data vendor for.

```bash
# reproduce everything above
idxbot backtest --universe all --providers none --profile accumulation --out reports/obs_acc.csv
idxbot backtest --universe all --providers none --profile momentum     --out reports/obs_mom.csv
idxbot evaluate --observations reports/obs_mom.csv --split --components
python3 scripts/robustness.py reports/obs_mom.csv
```

---

*A backtest that contradicts the strategy is more valuable than one that
flatters it. The first result here cost the original thesis; the second was only
trustworthy because the holdout was left untouched until the end.*
