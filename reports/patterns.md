# H61 — §14 chart patterns, and the base rate a vision layer would need first

*2026-09-07. `scripts/patterns.py`, `tests/test_patterns.py`. Roadmap stage 9.
Registered in the script's docstring before any cell was scored.*

---

## 1. What §14 asks for, and why this is not it

The brief wants a vision model that looks at a chart and names what it sees.
Two reasons this is not that:

- **There is no vision endpoint callable from this process.** Stating that and
  building the prerequisite is the honest move; building a stub that returns
  pattern names would be worse than nothing.
- **It is the wrong thing to build first regardless.** A layer that returns
  *"double bottom, high confidence"* is only usable once someone has measured
  what a double bottom is worth. The assessment's own note on stage 9 says it:
  **every pattern needs a base rate before it is used.** This file is that base
  rate, and it is what would make a vision layer safe if one were ever added.

Six classic patterns, defined **programmatically** and **causally** — each
fires on the bar at which it would have been visible, never at the pivot it is
built from. A32 records why that is the whole game: one non-causal helper
anywhere in the chain turns the study into a look-ahead and nothing in the
output looks wrong. `tests/test_patterns.py` truncates the panel at three past
dates and demands each detector return the same firings, plus a positive
control — a deliberately peeking detector that must FAIL that test, or passing
it means nothing.

## 2. The control is the point

A28 scored every trend detector against a random detector spending the **same
number of flips**; A34 records that a control denied the treatment's own
affordances is a handicap rather than a null. So each pattern here is scored
against random bars drawn from the **same (ticker, year) cells at the same
counts** — matched on everything except the shape. Without that, a pattern that
happens to fire on liquid names in good years beats a panel-wide control while
carrying no information at all.

## 3. The result: every pattern is negative, in both halves

63-session horizon, 2000-03-30 → 2026-09-04, 968 names.

| pattern | fires | mean log | control | edge | null sd | z | raw | **net edge** |
|---|---|---|---|---|---|---|---|---|
| breakout from squeeze | 4,014 | +0.0234 | +0.0413 | −0.0179 | 0.0034 | −6.23 | +6.86% | **−2.46%** |
| golden cross | 1,705 | −0.0125 | +0.0145 | −0.0270 | 0.0055 | −4.63 | +2.20% | −3.30% |
| higher highs and lows | 54,244 | −0.0031 | +0.0562 | **−0.0593** | 0.0011 | −54.31 | +4.26% | **−7.57%** |
| double bottom confirmed | 9,752 | +0.0164 | +0.0474 | −0.0310 | 0.0021 | −14.35 | +6.17% | −3.52% |
| gap up on volume | 5,849 | −0.0419 | −0.0067 | −0.0352 | 0.0035 | −8.66 | +2.68% | −3.86% |
| **digit (PREDICTED NULL)** | 427,592 | −0.0212 | −0.0148 | **−0.0064** | 0.0001 | **−47.65** | +1.51% | −1.27% |

Half-split of the edge, each half scored against a control drawn inside that
half:

| pattern | early | late | same sign |
|---|---|---|---|
| breakout from squeeze | −0.0119 | −0.0378 | yes |
| golden cross | −0.0057 | −0.0358 | yes |
| higher highs and lows | −0.0559 | −0.0683 | yes |
| double bottom confirmed | −0.0252 | −0.0398 | yes |
| gap up on volume | −0.0105 | −0.0271 | yes |
| digit | −0.0036 | −0.0084 | yes |

## 4. P1 failed in direction

I registered that the trend-continuation family — breakout, golden cross,
higher highs — would show a **positive** edge, on the strength of H13's
momentum features clearing t = 10. All six are **negative**, and the same sign
in both halves.

**Buying the confirmation bar is worse than buying a random day in the same
name-year.** That is a short-term reversal effect and it is the same fact H13
measured as `rev1`, arriving from a different direction: the discrete pattern
does not merely add nothing to the continuous feature, it selects the worse
bars within the cells it fires in. CLAUDE.md §8 asks for the continuous form
for exactly this reason.

## 5. P2's predicted null fired, and it decides how the table reads

`digit` — the close ending in 0 or 5 — is read off the price like every other
pattern here and means nothing. It reads **z = −47.65**, clearing the
Bonferroni bar by orders of magnitude.

**So significance in this table is not evidence, and the whole thing must be
read on effect size against cost.** A9 registered `squeeze` the same way and it
fired at t = +3.55 on two million rows; this is the second occurrence and the
more emphatic one. A predicted-null feature costs one column and is the only
cheap check that a pipeline is not manufacturing its own signal.

*What `digit` is probably picking up*, stated as interpretation rather than
measurement: prices sitting exactly on the fraksi-harga grid are
disproportionately the cheap, thin, low-priced end of the board, and that
cohort underperforms. It is a level effect wearing a pattern's clothes — which
is precisely why it belongs in the table.

## 6. P3 confirmed, and the gap between two columns is the reason it is readable

**0 of 6 patterns beat their matched control after the 0.56% fee alone**,
before any spread. Meanwhile the **raw** return looks positive for **6 of 6**.

That gap is the entire argument for the benchmark. A first version of this
table printed the pattern's own return net of the fee and read **+6.30%** for a
pattern whose matched control returned more — A19's error class, and exactly
the number a reader would quote.

## 7. Three bugs, all caught by impossible numbers

**`~` on an object-dtype Series inverts the integer.** `Series.shift(1)` on a
bool column returns object dtype (it must hold the NaN it introduces),
`.fillna(False)` leaves it object, and `~False == -1` — truthy. So
`up & ~up.shift(1).fillna(False)` is just `up`, with the negation silently
doing nothing. The golden cross fired **357,925 times on 690,591 eligible
bars — 52% of the panel**, for a signal that should fire a dozen times per name
in twenty-six years. After the fix: **1,705**. A30's rule, and this repo's
fifth catch of that kind.

**The `net` column had no benchmark in it.** See §6.

**A planted-shape test found an off-by-one.** `p_double_bottom` began its scan
at `2*w + 1`, discarding the first valid bar, and failed to fire on a textbook
double bottom until it was corrected to `2*w`. A detector that cannot find the
shape it is named after proves nothing by not finding it — A26's sine wave and
A27's planted Fibonacci bump, the same discipline a third time.

**And the harness itself was rebuilt once for cost.** The first control
filtered the whole 2.8m-row panel inside a loop over (ticker, year) cells and
never finished. Correct and unusable is its own kind of wrong: a control nobody
can afford to redraw is a control that quietly gets dropped, which is the one
thing this study cannot lose.

## 8. What this does not say

- Every number is **in-sample** — the holdout was spent at H16.
- The cost is A23's small-order model: **no impact, suspension or
  auto-rejection term**, and on the thin names where several of these patterns
  fire, all three bite.
- The control matches on **(ticker, year)**, not on within-year timing. A
  pattern that systematically fires in a particular part of the year is not
  fully controlled for, and matching on the date would make the control
  identical to the treatment.
- **A negative edge is not a short signal.** A5 forbids shorting, and the
  reversal measured here is smaller than the round-trip cost in both
  directions.
