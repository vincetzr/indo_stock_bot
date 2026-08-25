# H27 — the cross-sectional model §8 asked for, and the null that beat it twice

*31,394 name-years, 725 names, 15 purged walk-forward folds, 27,561 test-fold
observations. Code: `scripts/xsection_model.py`. Tests:
`tests/test_xsection_model.py` (8). Raw: `reports/xsection_model.txt`.*

Every sweep in this project — H13, H23, H25, H26 — ranked **discrete cells with
hand-tuned percentile cuts**. CLAUDE.md §8 says the opposite in as many words:
*"compute as numeric features feeding a cross-sectional model, not as discrete
buy/sell rules with hand-tuned parameters."* **The model was never built.** A
cell can only express "high vol AND thin"; a model can express an interaction,
a non-monotone response, a conditional no percentile cut reaches.

**Purged walk-forward.** A cohort dated *t* does not settle until *t*+252, so
training on it to predict *t*+30 leaks eleven months of overlapping future. For
each test year, training is restricted to cohorts whose forward window **closed
before that year began** — a year of data discarded at every fold.

---

## 1. The result

| | P(2x) | P(−50%) | **skew** | median | mean log |
|---|---|---|---|---|---|
| **model top decile** | 9.2% | **4.0%** | **2.31** | +0.3% | +0.0132 |
| all names (base) | 10.6% | 8.7% | 1.22 | | |
| H26 hand-cut cell | 10.5% | 4.1% | 2.60 | +0.0% | +0.0494 |
| + sector (11 IDX-IC) | 9.0% | 3.6% | 2.46 | −1.4% | +0.0034 |

**Null: 1.15 ± 0.06 over five runs. Model 2.31 → z = +20.51.**

**R1 — the model does NOT beat the hand-cut cell, and that is the finding.**
2.31 against 2.60. But the two numbers are not the same kind of number: the
model's is **out of sample** through a purged walk-forward, and the cell's is
**in sample**. Two entirely different methods — eleven features through
gradient boosting, and two percentile cuts chosen by hand — land within 0.3 of
each other. **That convergence is the strongest evidence in this project that
the structure is real and not a mining artifact.**

**R2 — SUPPORTED.** Permutation importance: `vol60` 0.073, `log_turnover`
0.021, `amihud60` 0.012, then a long tail. The model is using the same axes
H26 found. `hi52` scores low on the *up* leg — it earns its place by
suppressing the *down* leg, which is why a ratio objective found it and a rate
objective did not.

**Sector adds +0.15 skew and costs 0.0098 of mean log** at 99.6% coverage —
marginal, not a breakthrough. Its `shares` column is deliberately **not** used:
the file is frozen at 2024-07-10, so applying a 2024 share count to a 2010 bar
is look-ahead, and Indonesian rights issues are exactly what makes it wrong.

## 2. The null beat the model twice, and both times it was the null

The first two nulls returned **3.06**, above the fitted model's 2.31. A null
that beats the thing it tests is not a weak model — it is a broken null.

**Bug one: `up` and `down` permuted independently.** That breaks their real
link. A name that can double is the *same* name that can halve, both driven by
its volatility. Shuffling separately invents observations that doubled with no
halving risk, and a flexible model finds them immediately.

**Bug two — subtler and the actual cause: permuting *inside* (ticker, year)
blocks is nearly a no-op.** The ~12 monthly cohorts of one ticker-year hold
near-identical labels, because their forward windows overlap by eleven months.
Shuffling within the block preserves the ticker-year → label mapping almost
exactly, so the null retained the structure it existed to destroy.

**The fix that works:** reassign whole blocks' **labels** to other blocks'
**features**, which breaks the feature-label link while preserving the
clustering. The null fell from 3.06 to **1.15**.

This is the seventh time in this repo that the permutation null was the thing
that decided a result, and the first time it was wrong in the direction of
*understating* a real effect. A17's lesson was that too fine a shuffle leaves
the null too tight; this is the same lesson with the opposite sign.

---

## What this settles

**There IS a pattern and it is not random.** z = +20.51 out of sample against a
correctly-built null, from a purged walk-forward, confirming a cell found by a
completely different method. Roughly **2.3 : 1** asymmetry, driven by
volatility, liquidity and distance from the 52-week high.

**And a flexible model does not find more than two hand-picked filters.** With
eleven collinear price features over one macro history, the interaction space
is empty. That bounds what this data contains: the structure is real, it is
small, and it is already fully captured by "strong and calm".

**What would change this** is a feature family that is not price — the sector
map moved it +0.15, and genuine fundamentals (earnings, book value, debt) are
not in this repo at any horizon. That is the honest next instrument, and
`data/cache/fundamentals` holds 59 names, not 725.
