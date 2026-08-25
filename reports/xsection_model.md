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

---

## 3. The live output, and what each number is

`scripts/xsection_model.py` ends by fitting on every **settled** cohort — the
same purge, so nothing dated inside the last ~370 days trains the model — and
scoring today's cross-section. It prints, per name: close, the 2× level,
fitted P(2x), fitted P(halve), their ratio, and the round-trip cost.

**Two things about those numbers that must travel with them.**

**The fitted probabilities are optimistic.** Today's top decile reads a basket
P(2x) of 14.2% and P(halve) of 3.4%, ratio 4.17. The **realised**
out-of-sample decile over 15 walk-forward folds ran **9.2% / 4.0%, ratio
2.31**. A model's own predicted probabilities on the names it likes best are
always the flattering end of the estimate; the walk-forward pair is the one to
plan on. Both are printed side by side so the gap is visible rather than
implied.

**The 2× level is not a forecast.** Nothing in this repo predicts magnitude —
not one study, at any horizon. 2× is simply the level the probability was
measured against, and therefore where a take-profit would sit. Quoting it as a
target price would be inventing a number the evidence does not contain.

**And H24's arithmetic still applies:** whether the name doubles is settled at
entry; the exit only decides how much of it reaches the account. Selling
everything at 2× captures the full hit rate and costs most of the mean return
(H24 §5); holding captures less of the hit rate and keeps the tail.

---

## 4. Why no minerals or AI-data-centre names appear — asked directly

**First, the structural answer: the model cannot see narrative at all.** It
reads eleven cross-sectionally ranked price statistics and a sector code.
There is no channel through which "AI data centre" or "nickel cycle" could
reach it, and that is deliberate: there is no point-in-time news archive, so a
backtest conditioned on narrative would be look-ahead by construction.
`tests/test_news.py` walks the AST of `spine/` and `features/` and fails the
build if either imports the news module.

**But they are not merely absent. They rank near the bottom, and on the DOWN
leg.** Of 276 eligible names:

| theme | best rank | worst rank | notable |
|---|---|---|---|
| power for data centres | KEEN 57, POWR 64 | BREN 271 | POWR ratio 2.51 |
| critical minerals / nickel | NCKL 89, INCO 101 | MDKA 195 | none in the top decile |
| AI / digital infra | EXCL 112, MTEL 121 | TLKM 265 | none in the top decile |
| coal / energy | ADRO 115, PTBA 123 | BUMI 267 | none in the top decile |
| **Prajogo / Barito complex** | PTRO 252 | **TPIA 276 of 276** | see below |

The Barito names are the clearest case. **TPIA ranks last of 276 with a fitted
P(halve) of 26.7%**; BREN 16.3%, CDIA 21.7%, CUAN 15.3% — against a top-decile
4.0%. The model is not expressing a view on Chandra Asri's business. It is
reading price: these are the most extended, most volatile large names in the
market, and extension plus volatility is precisely what loads the down leg.

**The sector tilt says the same thing.** Top 28 against the universe: Consumer
Cyclicals **+13**, Industrials +5, Healthcare +4; Energy **−9**,
Infrastructures −6. The model systematically leans away from the commodity and
infrastructure themes and toward duller consumer and industrial names.

**And this is a trade, not a free win.** The model's top decile doubles
**9.2%** of the time against a base of **10.6%** — it gives up upside. Its
entire edge is the denominator: 4.0% halving against 8.7%. So if the single
thing you want is the highest chance of a double, the low-ranked narrative
names are not obviously worse on *that axis alone*; they are far worse on the
ratio, which is the objective H26 fixed and this model inherited.

**The honest limitation.** If the data-centre or nickel thesis resolves upward,
this model misses it entirely — it has no way to know. What it can tell you is
what those names' current price state implies about their risk, and right now
it implies elevated halving risk. That is a genuine blind spot and not a
defence of one.
