# Findings from 25 years of real IDX data

**Headline: the price-only accumulation score is inversely related to forward
returns on IDX. Do not trade it as a buy signal.**

This is a negative result on the strategy's own premise, produced by its own
backtester on genuine exchange data. It is reported here rather than buried
because it is the single most decision-relevant thing in the repository.

---

## What was tested

| | |
|---|---|
| Observations | **56,732** |
| Tickers | 66 |
| Period | 2001-07-31 → 2026-08-07 (**25 years**) |
| Data | Real Yahoo daily OHLCV + real IHSG benchmark. **Nothing simulated.** |
| Mode | `price-only` — broker components disabled via `--providers none` |
| Sampling | Every 5th bar, 300-bar warm-up, no look-ahead (unit-tested) |

Reproduce:

```bash
idxbot backtest --universe all --providers none --step 5 \
    --out reports/backtest_observations.csv
python3 scripts/robustness.py reports/backtest_observations.csv
```

---

## Result 1 — the score is monotonically inverted

Mean forward return by score bucket:

| Score bucket | n | 5d | 10d | 20d | 60d |
|---|---|---|---|---|---|
| 0–39 | 24,433 | **+0.76%** | **+1.37%** | **+2.44%** | **+6.86%** |
| 40–54 | 22,039 | +0.27% | +0.51% | +1.18% | +4.07% |
| 55–64 | 7,647 | +0.00% | +0.22% | +0.76% | +4.52% |
| 65–77 | 2,577 | +0.08% | +0.26% | +0.67% | +4.59% |
| 78–100 | 36 | −0.39% | −1.16% | **−3.04%** | — |

The *lowest*-scoring names outperformed the highest at every horizon. The signal
cohort (score ≥ 65) underperformed the unconditional baseline at all four
horizons, with t-statistics from −1.32 to −3.53.

## Result 2 — the Wyckoff phase ranking is backwards too

| Phase | Engine's rating | n | 20d | 60d |
|---|---|---|---|---|
| C — spring | **highest (0.95)** | 14,607 | +0.79% | +3.38% |
| D — markup starting | high (0.85) | 3,591 | +1.14% | +3.31% |
| B — building cause | medium (0.55) | 14,301 | +0.80% | +2.68% |
| A — stopping action | low (0.25) | 12,230 | +2.61% | +8.47% |
| **E — markup extended** | **lowest (0.30)** | 5,716 | **+4.00%** | **+10.67%** |

Phase E is what the planner actively blocks as *"a chase, not an accumulation
entry."* It was the best-performing state in the sample by a wide margin — more
than 3× the 60-day return of the spring setup the engine rates highest.

## Result 3 — it is not a regime artifact

Split-sample, using the high-minus-low 20-day spread:

| Subsample | n | high − low (20d) | t |
|---|---|---|---|
| Full sample | 56,732 | −1.82% | −5.87 |
| First half (2001–2017) | 28,362 | −1.39% | −2.57 |
| Second half (2017–2026) | 28,370 | −1.63% | −4.42 |
| Ex-crisis (no 2008/09/20) | 50,060 | −1.49% | −4.76 |

Same sign, same rough magnitude, in every slice. Bucket monotonicity holds in
each. This is not one crash doing the work.

---

## What it means

**IDX over 2001–2026 rewarded momentum, not mean reversion into bases.** Almost
every price-only component in the score is contrarian by construction — volume
drying up, range compressing, buying weakness, the Wyckoff spring. Those
components were systematically on the wrong side of a market where trend
continuation paid. The phase-E result is independent corroboration from a
different part of the code.

**Three things this does *not* prove:**

1. **That accumulation detection doesn't work.** It shows the *price/volume
   proxy* for it doesn't. The actual thesis — that you can see institutional
   absorption in broker-level flow — was never tested, because no real broker
   summary was obtainable. That remains open.
2. **That inverting the score would make money.** Flipping a failed signal is a
   textbook overfitting trap: the inversion is measured in-sample on the same
   data that produced it, and it ignores costs, borrow, and the fact that a
   momentum strategy has entirely different drawdown behaviour. The phase-E
   corroboration is suggestive, not a strategy.
3. **That the code is wrong.** The ledger, campaign and microstructure maths are
   unit-tested (60 tests), and the no-look-ahead property is explicitly verified.
   The machinery is sound; the hypothesis it encodes is what failed.

---

## What to do with this

**Do not run `idxbot plan` on price-only signals and trade the output.** The
evidence says those levels are, if anything, mildly adverse.

The engine still earns its place:

- **The ledger and campaign analytics are the real product.** They are what
  answer "where is this desk's cost basis, and how do they take profit" — and
  they are untested only because the data is paywalled, not because they are
  wrong.
- **Connect real broker summary** (`docs/LIVE_DATA.md`), then re-run this exact
  backtest *with* broker components enabled. If the broker-flow half carries the
  information the price-only half lacks, the buckets will stop being inverted.
  That is the experiment worth running, and the code is ready for it.
- **Re-examine the weights.** They were set by judgement, not fitted. Given this
  result, `relative_strength` (the one momentum-flavoured component) deserves
  more weight and the contrarian components less — but fit that on a holdout, not
  on this sample.

---

*Reported in full because a backtest that contradicts the strategy is more
valuable than one that flatters it.*
