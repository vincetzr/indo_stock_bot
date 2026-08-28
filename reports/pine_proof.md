# H28 — proving the TradingView rule, and the architecture the proof rejected

*31,394 pre-holdout name-years, 725 names, 1-year forward windows. Code:
`scripts/pine_proof.py`. Raw: `reports/pine_proof.txt`.*

A TradingView indicator sees **one chart**. Every result in this repo is
**cross-sectional** — it ranks names against each other on a given day. So the
Pine port cannot be the rule H26 measured; it can only approximate it, and an
approximation of a selection rule has to be measured on its own terms. Quoting
H26's skew of 2.60 for a rule that picks a different set of names would be the
same error as quoting a ten-year doubling rate as a one-year one (A21).

Three arms, all on the identical pipeline:

| arm | n | P(2x) | P(halve) | **skew** | median | CAGR/name | captures TRUE |
|---|---|---|---|---|---|---|---|
| TRUE — full cross-section | 2,022 | 10.5% | 4.1% | **2.60** | +4.8% | +6.4% | 100% |
| **ABS — fixed thresholds** | 2,593 | 8.3% | 4.1% | **2.01** | +1.9% | +1.9% | **72.1%** |
| PROXY — 36-name basket | 1,828 | 9.7% | 5.2% | 1.86 | +5.0% | +4.6% | 53.4% |
| none — no screen | 31,394 | 12.1% | 9.0% | 1.33 | −3.4% | −5.0% | — |

## The proof rejected the architecture I had chosen

The intended design pulled 36 reference symbols via `request.security` to
approximate the daily cross-section, with fixed thresholds as a degraded
fallback. **The fallback measures better.** ABS reads skew 2.01 against the
proxy's 1.86 and captures 72% of the true rule's picks against 53% — while
needing **zero** security calls, working on any IDX chart, and having no
reference basket to rot.

The reason is not subtle: a percentile ranked against ~24 live names is noisy,
and that noise costs more than the fixed threshold's regime drift.

**And the first basket was worse still.** Picked from *today's* liquidity
leaders, it had a median of **six** live members historically — three in 2005,
four in 2010 — because most of those names listed recently. The proxy then
captured 28.2% of TRUE's picks and the early half had 87 observations, too few
to read. Rebuilding it from names listed before 2008 took liveness to 28 in
2005 and 36 from 2010, and *improved* present-day accuracy too (mean absolute
percentile error 0.067 against 0.095). **A measuring stick has to exist for the
whole period it measures** — and it still lost to no measuring stick at all.

## The shipping rule, tested

```
strength:  close >= 0.9625 × 252-session high     (within 3.75% of the high)
calm:      60-session stdev <= 0.0257 daily       (40.8% annualised)
```

| | |
|---|---|
| P(touch 2x within 252 sessions) | **8.3%** |
| P(end at or below half) | **4.1%** |
| **skew** | **2.01** against an unscreened 1.33 |
| clustered null, 5,000 draws | 1.20 ± 0.13 |
| **z / p** | **+6.06** / **0.00020** against a bar of 0.00057 — **clears** |
| half-split | 2.44 early / 1.39 late, against a base of 1.61 / 1.13 |

**Positive in both halves**, though the late half at 1.39 vs 1.13 is thin — the
edge has narrowed, exactly as the base rate has.

## What this does not prove

- **In sample.** The holdout was spent at H16. There is no untouched period
  left in this data.
- **One year.** Every number above is a 252-session figure. H23's table shows
  the same family of rules inverts below three years and again above five.
- **No magnitude.** 2× is the level the probability was measured *against*, not
  a target. Nothing in this repo forecasts how far a price goes.
- **Not a portfolio result.** These are per-name rates. H20 showed that
  per-name statistics and portfolio outcomes can disagree completely.
- **72%, not 100%.** The chart rule is a lossy version of the measured one.
