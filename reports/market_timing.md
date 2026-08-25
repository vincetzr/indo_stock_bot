# H22 — can a rule tell you to be out of IDX before a correction?

*5,486 IHSG sessions, 2004-01-02 → 2026-08-21. Code: `scripts/market_timing.py`.
Tests: `tests/test_market_timing.py` (12). Raw: `reports/market_timing.txt`.*

Never asked before. H9–H21 all ask which **names** to hold; none asks whether
to be in the market at all. Nine rules fixed before scoring, signal from bars
≤ t acted at t+1's close, 0.56% per round trip charged as 0.28% per switch.

---

## 1. Not one rule beats holding

| rule | CAGR | at zero cost | max DD | switches | % in |
|---|---|---|---|---|---|
| **always in (buy & hold)** | **+10.32%** | +10.33% | −60.7% | 1 | 100% |
| above 200d MA | +6.82% | +8.72% | **−27.2%** | 142 | 68% |
| above 100d MA | +8.59% | +11.54% | −30.3% | 216 | 64% |
| above 50d MA | +6.45% | +11.01% | −37.4% | 339 | 64% |
| 50d above 200d (golden cross) | +8.58% | +8.88% | **−23.9%** | 22 | 68% |
| 12m momentum > 0 | +1.83% | +3.82% | −55.8% | 156 | 72% |
| 1m momentum > 0 | +3.91% | +10.76% | −64.5% | 515 | 61% |
| drawdown < 5% | +1.99% | +5.04% | −46.0% | 238 | 51% |
| drawdown < 10% | +4.75% | +6.61% | −32.0% | 142 | 72% |
| calm (vol20 < median) | +3.12% | +5.45% | −36.8% | 180 | 60% |

**Every one loses, and the best of them loses 1.7% a year.** Two — `above 100d`
and `above 50d` — beat holding *at zero cost* (+11.5%, +11.0%) and lose once
switching is charged. That is the whole finding in miniature: the signal is
real and the toll eats it, which is the same wall H13 and H19 hit from other
directions.

**They do cut drawdown.** The golden cross takes −60.7% to −23.9%. That is a
genuine and reasonable thing to buy, but it is a purchase, not a free lunch:
it costs 1.7% a year.

## 2. The matched null, which is what decides the table

A rule out of the market a third of the time dodges a third of the crashes by
construction, so a shallow drawdown proves nothing on its own. Each rule is
compared to 200 **random** switchers with the *same* trade count and the same
fraction of time invested.

| rule | CAGR | null mean | null p95 | beats null? |
|---|---|---|---|---|
| above 200d MA | +6.82% | +3.47% | +6.90% | no |
| **above 100d MA** | **+8.59%** | +2.82% | +6.67% | **yes** |
| **above 50d MA** | **+6.45%** | +0.68% | +4.14% | **yes** |
| 50d above 200d | +8.58% | +5.05% | +9.50% | no |
| 12m momentum > 0 | +1.83% | +3.13% | +6.84% | no |
| **1m momentum > 0** | **+3.91%** | −1.41% | +2.18% | **yes** |
| drawdown < 5% | +1.99% | +2.21% | +5.84% | no |
| drawdown < 10% | +4.75% | +3.44% | +6.90% | no |
| calm | +3.12% | +2.93% | +6.84% | no |

Three rules carry information — they beat random switching at the same
turnover. **All three still lose to not switching at all.** Being better than
a coin flip about when to trade is not the same as trading being worth it.

## 3. The half-split, and it is unanimous

| rule | early 2004→2015 | late 2015→2026 | both? |
|---|---|---|---|
| above 200d MA | −5.92% | −1.42% | no |
| above 100d MA | −1.34% | −2.04% | no |
| above 50d MA | −4.11% | −3.66% | no |
| 50d above 200d | −3.31% | −0.40% | no |
| 12m momentum > 0 | −12.69% | −4.82% | no |
| 1m momentum > 0 | −2.92% | −9.18% | no |
| drawdown < 5% | −12.77% | −4.46% | no |
| drawdown < 10% | −8.88% | −2.70% | no |
| calm | −10.22% | −4.57% | no |

**Eighteen cells, all eighteen negative.** No era, no rule. This is the
cleanest negative result in the project — everywhere else at least one cell
survived by chance.

## 4. What IS knowable: the conditional correction rate

Timing rules fail; the *conditional* is still real and is what to use.

**P(the index falls a further 5% within 20 sessions)** — unconditional base
rate **20.7%** over 5,286 sessions:

| state | n | P(−5% in 20d) | vs base |
|---|---|---|---|
| vol20 bottom quartile | 1,322 | **11.6%** | −9.0 pp |
| dd −2% to −10% | 2,085 | 17.6% | −3.1 pp |
| at/near 52w high | 1,755 | 18.1% | −2.5 pp |
| above 200d MA | 3,732 | 18.3% | −2.3 pp |
| below 200d MA | 1,554 | 26.3% | +5.6 pp |
| dd −10% to −20% | 963 | 26.6% | +5.9 pp |
| **dd worse than −20%** | 483 | **31.5%** | +10.8 pp |
| **vol20 top quartile** | 1,322 | **32.5%** | +11.9 pp |

The spread runs 11.6% to 32.5% — a factor of three, and real. Note what it
says: **risk is highest when you are already down, not when you are at a
high.** Selling "before" a correction from a calm high means acting on an
11.6% probability. The states that genuinely predict further falls are ones
you can only be in *after* the fall has started.

That is why §1–§3 fail. The conditional is informative and still leaves
roughly two-thirds of the high-risk cases *not* falling further, which is not
enough to overcome 56 bps a switch.

---

## What this licenses

- **Do not time IDX with these rules.** Nine rules, both halves, all negative.
- **A drawdown-limiting rule is a legitimate purchase**, not an edge: the
  golden cross halves the worst drawdown for about 1.7% a year. Buy it if the
  −60% is intolerable; do not buy it expecting more money.
- **The conditional table is a risk gauge, not a trigger.** Use it to size, to
  decide whether to add, and to know what kind of tape you are in.
