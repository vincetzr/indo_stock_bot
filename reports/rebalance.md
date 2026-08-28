# H43 — how often to rebalance the surviving screen, and a conflict it exposes

*2,860,181 rows, 919 names, 2000-03-30 → 2026-08-28. Six holding periods ×
8 start phases = 48 portfolio paths per arm, each against a size-matched random
basket from the identical universe and against the IHSG on a total-return basis
over that arm's own window. Code: `scripts/rebalance.py`. Raw:
`reports/rebalance.txt`, `reports/rebalance.csv`.*

H26's strength+calm screen is the only result in this project that cleared the
Bonferroni bar, replicated in both halves, and was independently confirmed out
of sample by H27. Its basket was rebalanced **annually — a horizon H16 picked
for an unrelated reason and twelve studies then inherited.** A20 measured what
that costs: when the horizon was finally varied it inverted the answer.

---

## 1. The table

| hold | held | turnover | cost/yr | **CAGR** | random | index | **vs index** | GROSS edge over random |
|---|---|---|---|---|---|---|---|---|
| 1 month | 9 | **58%** | 5.84% | +10.91% | −5.21% | +11.52% | −0.61% | **+12.78%** |
| **1 quarter** | 9 | 76% | 2.53% | **+11.97%** | −0.13% | +11.64% | **+0.33%** | +11.53% |
| 6 months | 9 | 84% | 1.39% | +11.99% | +2.67% | +12.18% | −0.18% | +9.12% |
| 1 year | 9 | 91% | 0.74% | +10.98% | +5.31% | +13.20% | −2.22% | +5.60% |
| 2 years | 9 | 92% | 0.37% | +7.94% | +5.90% | +13.84% | −5.90% | +2.01% |
| 3 years | 8 | 96% | 0.25% | +6.67% | +3.68% | +14.53% | −7.86% | +2.99% |

**P1 CONFIRMED — the curve is humped**, peaking between a quarter and six
months, and the long end is much worse. Registered as "monthly worst, optimum
between a quarter and a year"; monthly is not worst, but the interior optimum
is where predicted.

**P2 FAILED, and this is the finding.** I registered that the selection edge
would be roughly *invariant* to how often you act on it, on the reasoning that
signal quality is a property of the cross-section. It is not: the **gross** edge
over a random basket decays monotonically from **+12.78% at a one-month hold to
+2.01% at two years.** The screen's information has a half-life well under a
year. **It is a short-horizon signal, not a buy-and-hold-forever one.**

**The turnover measurement is new and useful.** Cost was charged on *turnover*,
not on frequency — two names changing out of ten costs a fifth of a round trip,
not a whole one, and every earlier study here priced the whole book. The screen
turns out to be **sticky: only 58% of the basket changes month to month**, so
monthly rebalancing costs 5.84%/yr rather than the ~12% a full rotation implies.

---

## 2. P3, the predicted null, very nearly fired

The start month is arbitrary. If the spread across *phases* rivals the spread
across *frequencies*, the table is a calendar accident.

| | |
|---|---|
| spread across frequencies (sd of the six means) | **3.38%** |
| spread across phases within a frequency | **2.93%** |
| ratio | **1.15×** |

A first version printed *"frequency dominates, the result is readable"* for that
ratio, which is far too generous — 1.15 is very nearly a tie. What the numbers
actually license: with 8 phases the standard error of a frequency's mean is
**1.04%**, so **differences under ~2.08% are not readable at all.**

Readable pairs: `21v504, 21v756, 63v252, 63v504, 63v756, 126v504, 126v756,
252v504, 252v756`. **Monthly, quarterly and six-month are mutually
indistinguishable.** So:

> **The shape is real — two- and three-year holds are readably worse than
> anything shorter. The precise optimum is not, because an arbitrary choice of
> start month moves any single arm by more than the middle frequencies differ
> from each other.**

The null was measured on the **excess over the index**, not raw CAGR: a
three-year phase offset moves the start date by three years, so raw CAGR would
confound phase with window. A19's lesson — comparing quantities measured over
different windows was the error running through every draft of that section.

---

## 3. THE CONFLICT, which matters more than the frequency question

| | screen | index | difference |
|---|---|---|---|
| **H26's memo** (`reports/asymmetry.md`) | +18.5%/yr | +14.4% | **+4.1 pts** |
| **H43, annual rebalance** (this study) | **+10.98%/yr** | +13.20% | **−2.22 pts** |

**The sign flips on the central claim of the one surviving result.**

Three things are worth stating precisely about that.

**H26's basket number is not reproducible from this repo.** `scripts/asymmetry.py`
carries it as a hardcoded `EVIDENCE` dict — `basket_x: 58.3`, `p_beat: 0.902` —
and the simulation that produced it is not in the repository. CLAUDE.md §15
requires every experiment to be reproducible from a seed and a config. This one
is not, and that is a defect independent of which number is right.

**The two are not measuring the same object.** H26's language — "median 58.3×",
"beats the index in 90.2% of draws", "10th percentile", "90th percentile" — is
the language of *resampling*, not of a historical path. H43 walks one actual
portfolio: buy the cell, hold, rebalance, pay the toll on the rotated fraction,
realise delisted names at their last print. **If H26's draws sampled names
independently from a pooled return distribution, they would understate how
correlated one year's screen actually is** — every name in the cell is a
high-momentum low-volatility liquid name, and they move together — which
inflates the median compounded outcome. That is a hypothesis about the
discrepancy, not a demonstration, because the code is gone.

**Until it is reproduced, believe the reproducible one.** H43 is a costed,
window-matched, phase-averaged portfolio walk whose code is in the repo and
whose controls are the two that have decided every result here. Reproducing
H26's basket would cost perhaps an hour; nothing else in this memo depends on
the answer.

---

## 4. What the reproducible version actually says

**The screen does not beat the index at any rebalance frequency.** The `vs
index` column reads −0.61%, **+0.33%**, −0.18%, −2.22%, −5.90%, −7.86%. The one
positive cell is quarterly, at +0.33% against a phase noise of ±1.18%.

**And beating the random control is necessary and nowhere near sufficient.** The
random arm — an equal-weighted basket of 9 liquid IDX names — returns −5.21% to
+5.90% depending on frequency, against an index at +11.5% to +14.5%. So "beats
random by +12.78%" and "loses to the index by 0.61%" are both true of the same
portfolio, and only the second one is a decision. **A19 recorded the missing
index comparison as the error that manufactured a result; this is the same
comparison catching a second one.**

**The half-split is positive in both halves at every frequency except three
years** — but that is the edge over *random*, which is the low bar. It is not
evidence of an edge over the index.

---

## What this licenses

- **Do not hold this screen for years.** The gross edge over random decays from
  +12.78% at a month to +2.01% at two years. This corrects advice I gave one
  message earlier, which extrapolated H23's ten-year result on the *liquid
  decile* to a screen that behaves nothing like it.
- **If you run it, run it quarterly** — but the choice among monthly, quarterly
  and six-monthly is not readable from this data, and the start month moves the
  answer more than the choice does.
- **The two- and three-year holds are readably bad.** That much is established.
- **The screen's headline advantage over the index does not survive a
  reproducible portfolio walk**, and the number that says otherwise cannot
  currently be re-run.
