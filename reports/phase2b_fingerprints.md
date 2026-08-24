# H14 memo — §9.4–9.6: broker style, and the null that inverted the answer

**Date:** 2026-08-24
**Verdict: one style dimension genuinely persists. Archetypes do not exist.
No dossiers are written.**
Reproduce with `python3 scripts/fingerprint_run.py --draws 40`.
Pre-registration: `hypotheses.md`, H14.

---

## 1. Why this is not a repeat of H11

H11 found a broker code's *margin rank* does not persist. That says nothing
about whether the broker's **behaviour** is stable, and the two come apart
cleanly: a market maker that crosses 90% of its flow every year of its life has
a completely stable style and, on H11's evidence, no stable edge whatsoever.

§9.4 asks for the description of the business. §9.3 asked for the description
of the result. This module computes no P&L anywhere — there is a test asserting
that — because reporting a stable *style* as a stable *edge* is precisely the
failure §9.6 exists to prevent.

Panel: **89 codes, 13 years, 1,015 broker-years**, 2014–2026.

---

## 2. Q1 — and the null inverted the answer

I registered the prediction that `cross`, `hhi` and `share` would persist above
+0.5, far beyond margin's +0.078. **On the raw numbers that is exactly what
happened. Against the null it is mostly wrong.**

| metric | observed | null | **distance from null** | verdict |
|---|---|---|---|---|
| `censor` | +0.845 | +0.484 ± 0.025 | **+14.6 sd** | real — but it is a *data artefact*, not behaviour |
| `cross` | +0.521 | +0.245 ± 0.038 | **+7.3 sd** | **real style persistence** |
| `edge_buy` | +0.104 | +0.009 ± 0.035 | +2.7 sd | weak |
| `hhi` | +0.603 | +0.575 ± 0.014 | +2.1 sd | weak — mostly artefact |
| `edge_sell` | +0.058 | +0.010 ± 0.031 | +1.6 sd | nothing |
| `ar1` | +0.009 | +0.002 ± 0.032 | +0.2 sd | nothing |
| `share` | **+0.912** | **+0.919 ± 0.002** | **−2.5 sd** | **BELOW its own null** |

### `share` persists at +0.912 and that is not a finding

This is the sharpest example this repo has produced. A year-over-year rank
correlation of **+0.912** looks like overwhelming evidence that broker size is
a stable attribute. Its label-shuffled null is **+0.919** — *higher*.

The mechanism is plain once seen. The shuffle permutes broker labels *within*
each ticker-window, so every code keeps the exact set of windows it appeared in
and only the values change. A broker present in 5,000 windows still draws 5,000
times, so its annual gross is driven by how often it shows up — which the
shuffle preserves perfectly. `share` is therefore measuring **presence, not
size**, and presence is conserved by construction.

`hhi` fails the same way for the same reason (+0.603 against a +0.575 null):
which tickers a code appears in survives the shuffle, and HHI is mostly a
function of that.

**Reading either against zero would have produced a confident wrong answer.
That is now the fourth occasion in this repo** — after H9's broken null, H10's
WAC bug, and H11's off-centre Track A null. The rule has earned its place:
the permutation null is the first statistic, not the last.

### What survives

**`cross`, the crossing ratio, at +7.3 standard deviations.** How much of a
broker's flow is matched on both sides — market-making churn versus directional
conviction — is a genuine, stable property of the firm, well beyond what
presence alone explains. `edge_buy` survives weakly at +2.7 sd.

So the honest answer to Q1 is narrower than my prediction: **the shape of the
book persists; nothing resembling skill does.** Execution edge is marginal,
and horizon (`ar1`, +0.2 sd) is indistinguishable from noise.

---

## 3. Q2 — the predicted degradation is not there

§6.3 argues large players split orders across brokers *because* the summary is
public, and §9.5 says to plot distinctiveness by year and report any decay
prominently. **I predicted a decline. There is no trend.**

```
2014 3.41   2015 3.41   2016 3.41   2017 3.42   2018 3.55   2019 3.46   2020 3.47
2021 3.45   2022 3.53   2023 3.48   2024 3.30   2025 3.29   2026 3.48
```

Slope **−0.0026 a year**; rank correlation with year **+0.132** — the wrong
sign for a decline; last value (3.48) *above* the first (3.41). Full range 0.25,
7.4% of the level. The three indicators disagree, which is what "no trend"
looks like.

**Logged as a failed prediction.** An earlier version of the output called this
"DECLINING" purely because the fitted slope was negative, while printing
endpoints that contradicted it. The verdict now requires slope, rank
correlation and endpoints to agree before it says anything — a negative slope
of 0.003 on a series sitting at 3.4 is not a decline, and calling it one is how
a null result becomes a finding.

**What this does and does not say.** Order-splitting may well be spreading; it
is simply not visible in fortnightly top-10 data over this window. The
dataset's shelf life is not obviously expiring, which is mildly good news and
the opposite of what §6.3 led me to expect.

---

## 4. Q3 — archetypes do not exist, and that is decisive

§9.5's mandatory check: fit on 2014–2019, assign on 2020–2026. 83 codes appear
in both eras.

**HDBSCAN finds zero clusters. 100% of codes are labelled noise.**

That is the whole answer. HDBSCAN declines to partition data that has no
density structure, and it declined completely. The brokers form one continuous
cloud in style space, not separable groups.

Forcing a partition anyway confirms it:

| k | GMM agreement | chance | KMeans agreement | chance |
|---|---|---|---|---|
| 2 | 77% | 63% | 83% | 67% |
| 3 | 57% | 40% | 69% | 38% |
| 4 | 54% | 36% | 54% | 29% |
| 5 | 52% | 33% | 45% | 26% |

A two-way split beats chance by ~15 points and degrades toward chance as k
rises. But k-means and GMM *always* return k clusters — they cannot report that
there are none. What they produce here is a cut through a continuum, and the
cut is only modestly reproducible across eras.

Per-dimension carry-over from early era to late: `share` +0.72, `hhi` +0.53,
`cross` +0.29, `edge_sell` +0.12, `edge_buy` +0.10, `ar1` −0.01 — and the first
two are the ones §2 showed to be largely presence artefacts.

**My prediction here was right, and stronger than I expected:** broad
separation survives modestly, finer splits collapse, the cluster count is not
stable — and HDBSCAN says there was never any natural clustering to find.

---

## 5. §9.6 — no dossiers, and that was decided in advance

The pre-registration said dossiers are written **only** if Q3 shows stable
archetypes. It does not, so none are written.

This matters more than it looks. §9.6 is the section most exposed to
fabrication — a template with headed sections invites filling them in, and a
"behavioural read" of a broker code is very easy to write and very hard to
falsify. The conditional was registered before the answer was known precisely
so that the decision not to write them could not be second-guessed after the
fact. §9.6's own rule governs: below the threshold, write "insufficient data"
and move on.

What *could* honestly be written, for the handful of codes with the most data,
is one line: a crossing ratio with a confidence interval, and the note that it
is the only dimension that persists. That is not a dossier. It is a single
number, and it belongs in a table rather than a narrative.

---

## 6. What I believe now

**High:** a broker's crossing ratio is a real, stable property of the firm
(+7.3 sd beyond its null). The shape of the book persists.

**High:** almost nothing else does. Execution edge is marginal (+2.7 sd on the
buy side, +1.6 on the sell), horizon is noise, and the two headline-looking
metrics — size and concentration — are largely artefacts of which windows a
code appears in.

**High:** there are no archetypes. Not "the archetypes are unstable" — HDBSCAN
finds no density structure at all, so there is nothing to be unstable.

**High, and this is the transferable one:** a statistic can persist *less* than
its own label-shuffled null while looking overwhelming against zero. `share` at
+0.912 versus a +0.919 null is the cleanest demonstration this repo has.

**Medium:** that order-splitting is not eroding the data's usefulness. Q2 finds
no trend, but thirteen fortnightly-resolution years is not a powerful test of a
slow structural change.

---

## 7. Where this leaves §9

§9 asked for cohort P&L (done, H10), profitability persistence (done, H11),
fingerprints (this memo), archetypes (do not exist) and dossiers (not written,
by prior agreement). The section is complete, and its answer is that **broker
codes are stable identities whose stable part carries no information about
returns.**

That closes the last open branch of the research programme. Combined with
H9, H12 and H13, the picture does not change: there is structure in this
market, and none of it survives the cost of acting on it.
