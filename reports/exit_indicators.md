# Indicator exits — you can have consistency or multipliers, not both

*H18. Pre-holdout only; the holdout was spent at H16. Code:
`src/idxbot/spine/signals.py`, `src/idxbot/spine/exits.py`,
`scripts/build_indicators.py`, `scripts/exit_indicators.py`,
`src/idxbot/report/monitor.py`, `scripts/positions.py`.
Tests: `tests/test_signals.py` (37), `tests/test_monitor.py` (16).*

---

## First: news cannot be in this table, and that is not a gap I can close

There is no point-in-time news archive. `tests/test_news.py` walks the AST of
`spine/` and `features/` and fails the build if either imports the news module,
precisely so a headline visible today cannot be attached to a 2015 bar. **Any
news-conditioned exit rule that appeared in a backtest here would be look-ahead
by construction**, and its measured edge would be an artefact of knowing the
future. That is a fact about the data, not about effort: fixing it needs a
timestamped archive nobody publishes free.

So news is wired as a **live-only overlay** in `scripts/positions.py` — standing
event tags (suspension, UMA, rights issue, delisting) printed beside the levels,
computed into nothing. A suspension is a fact about whether you can trade at
all, worth seeing whether or not it carries measurable alpha.

Stochastic, EMA, ATR and volume are all causal and were tested properly.

---

## The data question that had to be settled first

The spine panel carries close, adj_close and volume — **no high or low** — so a
first reading says ATR and stochastic are not computable and only close-only
variants are available. That reading was wrong. `data/cache/ohlcv/` holds full
OHLCV for **all 919 panel names**; on a 25-name sample the cached close matches
the panel's close on **100%** of overlapping bars and high/low bracket it on
**99.999%**. High and low are rebased with the panel's own `adj_close/close`
factor so they inherit its repairs — which surfaced SCCO and SINI, two of the
three repaired names, as the only tickers where the vendor's raw close disagrees
with the panel's (0.4% and 0.5% of bars). Those bars get NaN rather than a
mixed basis.

---

## What was pre-registered, and how each one came out

**H18a — a volatility-normalised trail beats a fixed percentage one.**
Mechanism: the entry selects for high realised vol, so a fixed 15% band is a
*different rule* for every name it picks. **SUPPORTED.** Best chandelier
+2.9% cohort median against the best fixed trail's +0.9%, and the walk-forward
chose a chandelier in 56 of 176 cohorts.

**H18b — an indicator stop cuts P(−50%) more cheaply than a hard stop.**
Mechanism: it can tell trend failure from noise; a percentage cannot.
**FAILED, clearly.** The undominated (P(−50%), P(2x)) frontier is *entirely*
price rules. Every armed indicator rule reads P(−50%) = 15.1%, identical to the
armed trails, for the same structural reason: a name that falls from entry never
arms. The unarmed indicator rules do reach P(−50%) ≈ 0.1% — by exiting after
10–17 sessions, with P(2x) collapsing to 0.7–1.1%. The prediction was wrong and
is logged as wrong.

**NULL — a random exit should behave like a matched-length hold.** It did, and
that is the check working:

| | median | P(2x) | days |
|---|---|---|---|
| NULL random exit | −4.6% | 6.0% | 122 |
| `hold 126` | −5.0% | 6.8% | 125 |

**And the null beat every hard stop** (−4.6% against −10.6% to −11.7%). Exiting
on a coin flip does better than a 15–30% hard stop on this entry. That single
comparison is worth more than the rest of the frontier table.

---

## THE RESULT: the objective decides the answer, and it is a genuine trade-off

The same purged walk-forward, over the same 176 cohorts and the same 58 rules,
selecting on three different targets:

| objective | cohort median | mean | **P(2x)** | P(−50%) | days | most-chosen rule |
|---|---|---|---|---|---|---|
| **median** | **+3.16%** | +7.10% | **3.5%** | 14.8% | 195 | stoch rollover armed +50% |
| **mean** | −2.86% | **+18.42%** | 11.0% | 15.8% | 239 | volume climax armed +50% |
| **P(2x)** | −3.36% | +19.97% | **11.6%** | 16.1% | 247 | volume climax z3 armed +50% |
| buy and hold | −4.30% | +18.80% | 11.6% | 16.3% | 250 | — |

Read the P(2x) column. **Optimising the median cuts the doubling rate from
11.6% to 3.5% — it throws away two-thirds of the multipliers.** Optimising for
mean or for P(2x) lands essentially on buy-and-hold: +18.4% and +20.0% mean
against buy-and-hold's +18.8%, holding 239–247 sessions out of 252.

So the honest statement is:

> **No rule in this 58-rule catalogue beats buy-and-hold on mean return or on
> P(2x).** The entire measured improvement is a median effect — being right more
> often, on smaller amounts.

For an entry rule whose whole premise is P(2x), selecting its exit on median
return is optimising against the premise. H17 made that choice and so did the
headline run here; the table above is what forced it into the open.

### The headline number, stated with its objective attached

On `objective="median"`: **+3.16% against buy-and-hold's −3.19%, difference
+6.35% [+3.57%, +9.08%]**, winning 85 of 176 cohorts (86% of the 99 that
differ), sign test p = 1.4 × 10⁻¹³. P(−50%) 14.8% against 16.3%.

**Against H17's incumbent** (`trail 15% armed +50%`) on the same cohorts:
**+1.21% [+0.07%, +2.42%]**, better in 37 of the 55 cohorts that differ, sign
test **p = 0.014**. That does **not** clear the Bonferroni bar — 49 trials gives
α = 0.001 — so the indicator layer's edge over the plain trail is *suggestive,
not established*. The direction is right; the size is inside the noise of the
trial count.

### On the 2025 cohort (already-spent holdout — certifies nothing)

| rule | mean | median | 2x | −50% | held |
|---|---|---|---|---|---|
| buy and hold 252 | +6.8% | −7.2% | 2 | 4 | 241d |
| H17 `trail 15% armed +50%` | **+16.0%** | +24.5% | **2** | 4 | 147d |
| H18 pick `stoch rollover armed +50%` | +8.3% | +23.4% | **0** | 4 | 145d |

The same trade-off on one date: the median-optimal indicator rule matched the
trail's median and produced **zero doublers where the trail produced two.**

---

## What I would actually use, and why

**If you want multipliers — which is what this entry rule is for — hold, or use
`trail 15% armed +50%`.** It is the only rule that improves the median without
gutting P(2x) (7.3% against buy-and-hold's 11.6%, versus 3.5% for the
median-optimal indicator rule).

**If you want fewer bad years, `chandelier 2x ATR armed +50%` or `stoch
rollover armed +50%`** raise the median by 3–4 points. Accept that you are
buying consistency with roughly two-thirds of the doubling rate.

**Do not use a hard stop on this entry.** It underperforms a random exit date
by 6–7 points of median, because it cuts positions out of drawdowns that would
have recovered — and P(−50%) is the one thing it fixes, which for a P(2x)-
selected rule is fixing the wrong end.

`scripts/positions.py` prints all of these as prices for names you hold, and
replays which ones already fired and at what.

---

## Method notes worth carrying

**Causality is tested, not asserted.** Every indicator is recomputed on
truncated prefixes and required to be bit-identical at bar *i*
(`test_indicator_is_causal`), plus a test that the harness can actually fail
against a deliberately peeking series. Reading the code and believing it is a
different and weaker thing.

**A blanket `except TypeError` around the thing you are measuring is a bug
factory.** The first `apply_rule` caught any failure and recorded it as "this
name had no data", which silently dropped every one-argument rule and would have
disguised a genuine crash inside a rule as an inapplicable one. Replaced by
explicit signature inspection.

**`id()` is not a cache key.** CPython reuses the id of a collected object, so a
short-lived lambda's arity was being served for a different rule created later
at the same address. It presented as "takes one argument but two were given".
Now a `WeakKeyDictionary`.

**A label containing another label as a substring will be misread.** The monitor
printed `chandelier 3x ATR (unarmed)` — which contains "armed" — and a filter
written against it silently matched the wrong rows. Renamed to
`(trails from entry)`.

**Registering a null is cheap and it earned its place twice here:** once by
behaving exactly like a matched-length hold (so the pipeline is not
manufacturing signal), and once by beating every hard stop, which is the most
useful single line in the frontier table.
