# The daily brief — what it reports, and the one thing in it that is new

**Date:** 2026-08-24
**Built:** `src/idxbot/report/brief.py`, `scripts/brief.py`, 26 tests.
**Run it:** `python3 scripts/brief.py --session post [--ticker BBCA]`

---

## 1. What was asked for, and what is reachable

The request was a twice-daily read on: what the market is doing, the
narratives, potential candidates, and whether a given move is over or just
starting. Three of the four are reachable from the spine. One is not, and the
distinction is worth stating precisely because it is not a matter of effort.

| asked for | status |
|---|---|
| what the market is doing | **built.** Arithmetic on 829 names: breadth on three horizons, advance/decline, cross-sectional dispersion, 250-day extremes, equal- and turnover-weighted index returns and vol percentile, and a count of names closed at the point-in-time auto-rejection band. |
| the narratives | **half built, and the half that is missing is the half most people mean.** Co-movement groups are derived from a factorisation fitted on the 250 sessions ending the day *before* the bar. WHY those names moved is not in this data. |
| potential candidates | **built, with its own refutation attached.** The eight H13-registered features, ranked in their registered direction, each line carrying H13's measured post-cost result. |
| is the run over or just started | **built as a description plus a historical frequency.** Where the move sits in its own history, and what followed bars in the same cell — with the base rate, the effective sample size, a block-bootstrap interval and a permutation null. |

### The news narrative is genuinely absent, not merely unbuilt

There is no news, filings or announcement source anywhere in this repository,
and §3's data table lists none. `narrative_gap()` exists as a function rather
than an omitted section so the absence is printed: a brief that simply had no
narrative heading would read as though there were nothing to say.

What the co-movement section gives instead is real and, for this purpose,
arguably better than a sector map. §7 already uses trailing principal
components as a sector substitute because this repo has no sector data — the
`sectors.app` module is the licensed API and needs a key. A sector label is a
fixed opinion about which names belong together; a component is measured from
how they actually traded. On 2026-08-14 the components separate cleanly
without being told anything:

```
PC2   4.2% of variance    BBNI BBCA BBRI BMRI BBTN BRIS KLBF ICBP
PC3   3.3% of variance    AADI ADRO INDY MEDC MDKA ELSA ADMR ANTM
PC5   2.0% of variance    BREN CUAN BRPT ADRO RAJA PTRO RATU CDIA
```

Banks, coal-and-metals, and the Prajogo complex, none of them named in any
config file. **They are printed as constituent lists and nothing else.**
Whether PC3 is "the coal trade" is an interpretation, and §9.6's rule — state
the regularity, mark the interpretation separately — applies here as much as
it does to a broker dossier.

---

## 2. The one genuinely new result, and why it is a lead rather than a finding

**"Is the run over" is a directional question, which is the thing H13 answered
in the negative.** So the brief does not answer it. It reports where the move
sits and what the historical distribution from there looked like. Building
that reference table produced a number that was not expected.

### The construction

A run is measured from the last 250-session extreme of the opposite sign.
`run_z = log-return / (vol60 × √run_days)` is the extension in standard
deviations *for a move of that length*, so a quiet name up 30% in ten days and
a volatile one up 30% in two hundred are not confused. Bars are bucketed on
four dimensions fixed before any cell was looked at — leg, run age tercile,
extension tercile (cut **per leg**), and index-vol tercile — giving 54 cells
over 1,127,670 liquid pre-holdout bars, 2001 to 2024.

Each cell reports its mean forward 20-session return, the equal-weighted mean
over all liquid names **on the same dates**, and the difference.

### The result

| | |
|---|---|
| largest cell excess over base rate | **+1.67%** per 20 sessions |
| same statistic under shuffled state labels | **+0.37%** mean, +0.53% at the 95th percentile |
| spread across the 54 cells | observed **0.68%**, null **0.13%** |
| p(null ≥ observed), 200 draws | **0.000** |

**The state conditioning carries information well beyond chance**, and it is
coherent rather than scattered: old stretched advances continued (+1.0% to
+1.7% over base, intervals excluding zero in all three vol regimes), while
advances that had run a long way in time but not in price underperformed
(−0.7% to −1.0%). That is a readable, internally consistent picture and it
agrees in sign with H13's momentum features.

### Why it is not reported as tradeable, and must not be

Four reasons, and each on its own is sufficient.

**It is the largest of fifty-four.** The null tests whether *any* cell is
extreme and answers yes. It does not license reading the maximum, which is
biased upward by exactly the selection that found it. Roughly three of the 54
uncorrected intervals clear zero by luck.

**The cells were not pre-registered.** The four dimensions were fixed in
advance; which cells matter was not. §11 requires the trial count to travel
with the result, and this adds cells to it rather than answers.

**It is entirely in-sample.** The 24-month holdout is untouched and stays
untouched. A brief that runs twice a day would spend it in a week, which is
why every reference distribution here is estimated on `holdout == False` rows.

**H13 measured very nearly this and found it net-negative.** `mom12_1`, `hi52`
and `atr_mom20` all encode "old stretched advance", and all were net-negative
at every horizon. The difference in construction — a long-only excess over the
cross-sectional mean here, a control-neutralised quintile spread there — is
exactly the kind of difference that can manufacture an effect, and the burden
is on the new construction, not the old one.

**Logged in `hypotheses.md` as an observation, not a hypothesis tested.** The
pre-registered test that would settle it does not exist yet.

---

## 3. Three defects the build surfaced, all of which produced believable output

Every one of these printed something a reader would have accepted.

**Breadth was being computed on a watchlist.** The panel held bars through
2026-08-21, but only 46 names traded in the last four sessions — a partial
refresh. Nothing said so, and "71.7% of names above the 20-day" was 46 large
caps. `resolve_asof` now falls back to the last session with a representative
cross-section and `coverage_warning` prints what it skipped. The failure mode
generalises: a date column being current is not the same as a cross-section
being current.

**Moving averages on a pivot returned nothing.** A wide date × ticker frame is
indexed by the union of every name's trading days, so a suspended name
acquires NaN rows it never had and `rolling(200, min_periods=200)` then
returns nothing — for every column at once. The first output read "0 of 830
names above the 200-day average", which is not wrong so much as vacant.
Grouping by ticker keeps each window on that ticker's own bars.

**`np.argmax` anchors runs to holes in the data.** NaN compares False against
everything, so the scan never displaces it and returns its index. The spine
carries 2,327 bars with a non-positive adjusted close, and before masking they
produced **915 "advances" with a negative return from their own anchor** — an
impossibility under the definition, sitting quietly in a conditional table.
A further 164,627 bars carry `vol60 <= 0`, which turns a motionless name into
an infinite extension.

And one that was purely statistical: **the block bootstrap selected its drawn
blocks with `np.isin`**, a set-membership test, so a block drawn twice
contributed once. Every resample was smaller and less variable than the sample
it came from and every interval came out too narrow. A bootstrap that
understates uncertainty is worse than no bootstrap, because it looks like
rigour.

---

## 4. Known limitations, stated rather than patched

**`give_back` is not a bucket dimension.** How far price has come back from the
leg's extreme is arguably the single most informative "is it over" variable,
and it is deliberately absent from the 54 cells. The four dimensions were
fixed before any cell was seen; adding a fifth after noticing which cells came
out large is the trap the trial count exists to catch. A name deep in recovery
therefore pools with one still falling. That is a real cost and it is the
correct one to pay.

**The auto-rejection count is an upper bound.** The panel carries no intraday
high or low, so this is `close == band`, not `reference.was_locked`'s full
test. A name that traded away from the band intraday still counts.

**The turnover-weighted index is not IHSG.** This repo has no
shares-outstanding series. Turnover and market cap are correlated and not the
same thing. Both rows are printed so a divergence is visible rather than
averaged into one number.

**A pre-open run knows nothing a post-close run did not.** There is no
overnight or pre-market IDX data here, and no global-macro feed. The two
sessions differ in what has settled, not in what is known. The `--session pre`
banner says so rather than implying freshness the data does not have.

**Survivorship is still one-sided.** A2: the delisted store's snapshot ends
2019-04-07, so names that died after that are missing their final months. The
decline cells are the ones this flatters.

---

## 5. What I believe now

**High:** the brief describes state correctly, and the three arithmetic
sections are exact.

**High:** the co-movement section recovers real economic structure — banks,
coal, the Prajogo complex — from returns alone, with no sector data.

**High:** the run-state conditioning carries information far beyond its
permutation null. p = 0.000 against 200 draws, with the cell spread five times
the null's.

**Low, and deliberately so:** that this is tradeable. It is the maximum of 54
in-sample post-hoc cells, and H13 measured very nearly the same thing and
found it net-negative after costs. The honest status is a lead awaiting a
pre-registered test against a holdout that is still untouched.

**Unchanged:** everything in A9. Nothing here reopens H9, H11, H12 or H13.
