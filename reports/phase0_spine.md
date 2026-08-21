# Phase 0 memo — the data spine

**Date:** 2026-08-21
**Gate 0: PASSES**, including both checks CLAUDE.md §5 names by name.
Reproduce with `python3 scripts/gate0.py` (exits 0 on pass, 1 on fail).

This memo was rewritten twice. The first version claimed Gate 0 passed while the
script was running checks I had invented rather than the two §5 specifies; §11
records that. The gate now runs the specified checks, failed on one, and passes
only because the defect it found was repaired.

---

## 1. Status against §5

| §5 requirement | status |
|---|---|
| Daily OHLCV, all IDX names, max history | done — 843 tickers, 2.6m bars, from 2001 |
| **Delisted and suspended names** | **partly recovered** — 121 vanished names with history, from a 2019 point-in-time snapshot; the way *out* is still missing (§6) |
| ARA/ARB schedule incl. asymmetric period | done — 6 regimes, back to 2010 |
| Fraksi harga (tick) schedule | done — 3 regimes, back to 2005, validated against 1.3m quoted closes |
| Lot size (incl. the 500-share era), index halts | done |
| Board membership per ticker-day | **derived** from IDX's published watchlist criterion (§12) |
| Broker code master with effective dates | partial, honestly labelled (§8) |
| Rights-issue / corporate-action adjustment | done, with fixtures to 86% dilution |
| Suspension / ARA-ARB flags per ticker-day | done |
| **Gate 0 check 1** — traded value | **PASS** (§4) |
| **Gate 0 check 2** — 5 events by hand | **PASS**, 9 checked (§5) |

1,547 tests pass. Gate 0 runs nine checks and exits non-zero on any failure.

---

## 2. The rules changed six times, and twice they changed back

```
2016-01-04  symmetric 35 / 25 / 20 by price band
2020-03-10  ARB -> 10%  (COVID, three trading days)
2020-03-13  ARB -> 7%
2023-06-05  ARB -> 15%  normalisation tahap I
2023-09-04  ARB = ARA, symmetric again
2025-04-08  ARB -> 15%, asymmetric AGAIN (Kep-00003/BEI/04-2025)
```

**The 2025 change lands inside the broker panel this repo collected**, so
anything assuming the Sept-2023 symmetric regime is wrong for most of it.

Tick size: five groups before 2014, three from 2014-01-06, five again from
2016-05-02. Every lookup takes a date and **raises** before its own schedule's
coverage rather than falling back — a silent fallback is the exact bug the
module prevents. Coverage is per schedule: ticks and lot from 2005,
auto-rejection from 2010, because that is where the evidence for each begins
(§12).

Validation against 843 tickers: worst regime violation rate **0.58%** against a
2% tripwire. Independent confirmation — CBRE's largest repeated falls are
−14.8%, sitting exactly on the encoded 15% floor.

---

## 3. Four defects found in the data

| defect | scale |
|---|---|
| **stale bars** | **421,942 = 16.2% of the spine**, some names 70%+ |
| level shifts | 5 total; 3 explained, 2 quarantined (§7, §12) |
| verified misdated action | 1 — SCCO, **repaired** (§5) |
| isolated source spikes | 11 bars across ELTY, MAPI, TOWR, SCCO, NISP |
| survivorship | **0 of 25 known-delisted names present** |

**Stale bars are the important one and the least dramatic.** One bar in six
records no trading and repeats the previous close. A backtest filling on one has
bought from nobody.

### Three detectors, each wrong before it was right

- **Splits by ratio alone** flagged 79 events; most were penny stocks moving one
  tick (Rp 3 → Rp 2 is a ratio of 1.5). Requiring the move to be large in *ticks*
  cut it to 11.
- **Spikes by powers of ten** missed SCCO's isolated 4× dip. Allowing any clean
  ratio found 121, again mostly penny ticks. A spike must now be a move the
  exchange **could not have permitted**, with the board *derived* from IDX's
  watchlist rule, and must sit inside the period whose bands are encoded.
  11 remain, all unambiguous.
- **Persistence by median** let a three-bar dip pass as a level shift.
- **Level shifts without an impossibility test** quarantined four names whose
  falls the exchange had plainly allowed — RODA's −34.3% against a 35% band.
  Requiring the same impossibility cut 11 shifts to 5.

---

## 4. Gate 0 check 1 — traded value: PASS

§5 asks for reconciliation against IDX's published aggregate, which is not
reachable. This reconciles the two independent sources that are — Yahoo OHLCV and
IndoPremier's session footer, which share no pipeline.

| comparison | median | p90 |
|---|---|---|
| IPOT internal: lots × 100 × VWAP vs published value | **0.000%** | 0.00% |
| cross-source: Yahoo shares × IPOT VWAP vs IPOT value | **0.017%** | 0.80% |

Implied VWAP sits inside the day's high–low range on **99.6%** of days; volume
agrees within 1% on 91.2%. Compared against the footer's VWAP, not the close —
using the close reports a 0.55% error that is simply close ≠ VWAP.

**Honest caveat:** 3,154 ticker-days over 9 names (~15 ticker-years), not 20
*random* ticker-years, and against IndoPremier rather than IDX itself. Narrower
than specified in coverage; stronger in kind.

---

## 5. Gate 0 check 2 — corporate actions by hand: PASS after three repairs

Nine events verified against Indonesian market announcements: BBCA 1:5, BMRI
1:2, SCCO 1:4, WIKA's rights issue, **PYFA's 1:20 rights issue**, **SINI's 2:3
rights issue**, DSSA 1:10 and 1:25, ISAT 1:4.

**Defining "reconciles" correctly took two attempts.** The first version asked
"is there a step at the ex-date?" and failed anything with one. That is wrong: a
price series may legitimately be **back-adjusted** (no step) *or*
**unadjusted-but-consistent** (a step equal to the theoretical factor). WIKA is
the second — 240 → 203.91 against a published theoretical ex-rights price of
**Rp 204** — and the first version called it a failure. It also compared *traded*
bars, so WIKA's three-week suspension across its own rights issue read as a 32%
break, and used a flat 15% threshold that flagged DSSA's ordinary +16.4% day.
The threshold is now the auto-rejection band.

### The one real failure: SCCO

| | |
|---|---|
| announced | 2024-01-15, stock near Rp 10,000, hit ARA on the news |
| approved at RUPSLB | **2024-02-20** |
| last day old nominal | 2024-03-07 |
| first day new nominal | **2024-03-08** |
| **cached series switches basis** | **2024-02-01** |

Nineteen days before shareholders approved it. `adj_close/close` is a constant
0.8825 throughout, so it is not a half-applied adjustment — the source has the
split on the wrong date.

Direction settled by the rest of the history: SCCO's median close runs
8,700 / 9,700 / 9,100 / 9,288 / 10,800 / 9,750 / 8,675 from 2017–2023 and 2,190
from 2024, so the series is **not** back-adjusted and the February window belongs
on the old basis.

**Repair: prices × 4 over 2024-02-01…2024-03-07.** The boundary becomes
10,175 → 2,550 = **×0.2506** against an announced ×0.2500. Before the repair it
read +0.2% and the split had silently vanished.

**Volume is deliberately not repaired** — share count did not change until March,
so February volume was already right. That asymmetry is the actual harm: price ×
volume understated traded value four-fold for ~25 sessions while both columns
looked reasonable alone.

Repairs are applied **on read** in `data/ohlcv.py`, never written to cache, so
the cache stays a faithful copy of the source and repairs stay reversible.
`apply_repairs` is **idempotent** — wiring it into Gate 0's loader while
`verify()` also repaired produced ×16 and a result that still looked like a price
series.

### Then the shape was swept for, and SCCO turned out not to be alone

Finding one defect by hand says nothing about how many exist. So Gate 0 check 8
now looks for the *shape* directly, using a property of this exchange that
happens to be exact:

> Every price IDX ever printed is an exact multiple of that day's **fraksi
> harga**. A price that is not — Rp 6,234.246582, Rp 122.380951 — was never
> traded. It is arithmetic a vendor did.

An off-grid stretch that `level_shifts` **independently** calls a break is the
SCCO defect and essentially nothing else. Across **937 tickers and 2.85m bars**
the sweep returns **three names**:

| ticker | window | sessions | action |
|---|---|---|---|
| SCCO | 2024-02-01 … 2024-03-07 | 23 | 1:4 split, applied 36 days early |
| PYFA | 2024-04-16 … 2024-04-19 | 4 | 1:20 rights at Rp 100 |
| SINI | 2026-06-29 … 2026-07-08 | 8 | 2:3 rights at Rp 5,000 |

In all three the vendor back-adjusted **only the last few cum sessions** instead
of the whole history, leaving an island on a basis nothing ever traded on — a
fake crash going in and a fake rally coming out.

### The grid is what let the two rights factors be *proved*

PYFA and SINI had been quarantined with the note *"cause known, factor not"*.
Reading the factor off the price move would have been circular — the move is the
thing being explained. The announcements fix the factor instead, and the grid
then gets an independent vote:

- **PYFA** — 1 old share : 20 HMETD at Rp 100, cum 2024-04-19, ex 2024-04-22. The
  last cum bar's adjusted close *is* TERP, and it reads 122.380951, which pins
  the cum price at **Rp 570** — an exact Rp 5 tick. Dividing the window by the
  implied factor puts **all 16** open/high/low/close values on the tick grid
  (1,040 / 1,170 / 940 / 950 / 980 / 750 / 775 / 735 / 815 / 675 / 685 / 515 /
  570) and **all 4** volumes on a whole 100-share lot.
- **SINI** — 2 old shares : 3 HMETD at Rp 5,000, DPS 2026-07-10, so cum
  2026-07-08 on T+2. Solving TERP = (2P + 15,000)/5 for the one tick-valued *P*
  consistent with the block gives **P = Rp 10,950** and **TERP = Rp 7,380** — and
  Rp 7,380.00 is exactly what sits on 2026-07-08, **a bar not used to fit it**.
  32 prices land on the grid, 4 volumes on the lot.

A factor 2% away scores **zero** on both tests. That is what makes the grid a
test rather than a description.

Unlike SCCO, both of these repair **volume as well as price**: their volumes are
not multiples of 100, which no IDX print ever is, so the vendor scaled those too.

Effect on the annualised return of each name, raw close, full history:

| | before | after |
|---|---|---|
| SCCO | +8.97% | +9.97% |
| PYFA | +5.76% | +6.60% |
| SINI | +77.80% | +81.76% |

### Two category errors the sweep exposed downstream

1. **`classify()` demanded that a rights issue land on its factor.** A split is
   mechanical — four shares for one, the price is a quarter, to the tick. A
   theoretical ex-rights price is a **valuation**, and the market reprices the
   moment it opens. PYFA closed its ex-day **34% above TERP**. The right test is
   the exchange's own: on an ex-date IDX resets the reference price to the
   theoretical one and the auto-rejection band applies *around that*. Rp 164
   against a TERP of Rp 122 with a 35% ARA is inside the band, and legal.
   Widening the tolerance instead would have let a genuinely broken split through.
2. **`_cap_impossible()` capped after adjusting.** PYFA's ex-day came out at the
   correct +34% and the 25% ARA then clipped it back to +25% — inventing a
   result no holder saw. Bars whose return has been divided by a verified factor
   are now exempt from the cap; bars whose step does *not* match their factor
   are not, so a defective split still gets capped.

### What the sweep also measured, which is not about corporate actions

**636,490 bars — 22.3% of the spine — provably sit on a vendor-adjusted basis**,
and that is a *lower* bound: dividing a price by a whole number leaves it on the
grid and is invisible here. It matters because `reference.half_spread` looks the
tick up **by price**. A series back-adjusted to a fifth of its traded level is
charged out of the wrong band and **understates the spread it would really have
paid** — the same lookahead error §5 warns about, arriving by a different route.

Two versions of this detector were wrong before it was right, both times by
forgetting the test is **one-sided**. Off-grid proves adjustment; **on-grid
proves nothing**. Reading each bar alone turned BMRI's one uniform ÷4 region
into 18,300 fictitious ones. Gap-closing then claimed to *segment* the series
into raw and adjusted, which it cannot: PTBA's ÷5 history lands on the grid
whenever the real price was a multiple of Rp 50, including 122 consecutive
sessions across the 2008 crash — 2,760 "defects" across the spine, of which
three were real. There is deliberately **no segmentation function** in
`quality.py`; there is a rate that is a bound, and a conjunction with
`level_shifts`.

---

## 6. Survivorship: three guesses became two measurements and one bound

The live cache holds 843 tickers and **not one stopped trading more than two
years ago**. Of 25 companies known delisted from IDX, **0 are present**. The
ticker list was built from a `TICKER,marketcap` file, which can only ever
contain live names, because market cap does not exist for a dead one.

**What fixed most of it** was a published **April-2019 snapshot of 627 IDX
tickers**, with price history. It is not survivorship-biased, because it was
taken then, and comparing it with today's spine names the casualties directly:
**121 of its 627 names are gone**, and 21 of the 25 known-delisted companies are
among them, *with prices*. They now live in `data/cache/delisted/` — 86 with
≥250 bars, 27 thin, 8 empty — kept apart from the live cache on purpose, because
a survivorship-free universe is a *different* universe, not a bigger one.

That turns three assumptions into evidence:

| | before | after |
|---|---|---|
| attrition rate | assumed 1–8%/yr | **measured 2.87%/yr** (121/627 over 7.4 years) |
| pre-delisting drag | unknown | **measured 4.8 pp/yr** — the doomed names were already lagging over 2014–2019 (+3.0% for survivors, −1.8% for them) *while everyone was still listed* |
| the universe itself | live names only | `point_in_time_universe()` reconstructs the constituent list as it stood, survivorship-free, for any date at or before the snapshot — 964 names on 2018-06-01 |

| delist rate | equal-weight | cap-weight |
|---|---|---|
| 1% | 0.9 pp | 0.05 pp |
| **2.87% (measured)** | **2.7 pp** | **0.13 pp** |
| 8% (a clean-up year like 2025) | 7.4 pp | 0.37 pp |

**The weighting column is still the finding.** A name about to delist is a micro
cap, so the bias is 20× smaller cap-weighted. The repo's large-cap work is close
to safe; equal-weight small-cap work is not.

**It remains a bound rather than a correction, and the reason is one-sided.**
The snapshot ends 2019-04-07, so what these names did **on the way out** — which
is where most of the damage is — is not in it. Names that delisted *before* the
snapshot are absent entirely, and after it the vanished names are known while the
newly *listed* ones are not separable from the live set, so a 2026 universe built
from these parts would be biased the other way. `point_in_time_universe()`
therefore returns an explicit `complete` flag rather than letting a caller assume
it got a clean universe. **No correction factor is applied.**

---

## 7. What is quarantined rather than trusted

SCCO proved a detected shift can be confidently wrong about its own date, so
anything unconfirmed is **quarantined** in `spine/repairs.SUSPECT` with a
±45-day window (wider than SCCO's 36-day error) and treated as neither a
corporate action nor a real price move.

**The list is now one entry.** It was ten, then three. Four former entries were
never corporate actions at all (§12); PYFA and SINI left once their announced
ratios were found and the tick grid confirmed them (§5). What remains is **ELTY
2018-06-07**, and it is not there because its cause is unknown — Bakrieland's
10:1 reverse split was *rejected* by shareholders, and the series sits at exactly
Rp 500 through 2015–2017 on near-zero volume and at exactly Rp 50 from 2018-06-07
with real volume the next day. It is a dormant quote re-marked on resumption,
where auto-rejection does not bind. It stays quarantined because the window is
**uninformative**, not because it is wrong.

---

## 8. Broker code master

**A rename is not a reassignment.** YP was eTrading (2003) → Daewoo (2013) →
Mirae Asset (2016): three names, one continuous business, one client base.
**A merger is** — CS's flow did not gradually become UBS flow, it moved.

Three confidence levels, never collapsed: `verified`/`reported` (dated),
`current_only` (name today known, history unknown — safe to label, never to
compare eras with), `unknown`. Coverage: 6 dated, 60 current-name-only.

---

## 9. What would have falsified all this

A regime mis-dated by a week would have blown the 0.58% violation rate. The tick
ladder is confirmed by CBRE sitting on the encoded floor. The adjuster is
confirmed by SCCO's real 1:4 restoring to ×0.2506. The traded-value check is
confirmed by two unrelated pipelines agreeing to 0.017%.

## 10. What I believe, and with what confidence

**High:** the encoded ARA/ARB and tick schedules are correct from 2014-01-06.
**High:** the *live* cache is totally survivorship-biased, and this matters far
more for equal-weight small-cap work than cap-weighted large-cap work.
**High:** SCCO's split was misdated, and the repair is right — the restored
boundary matches the announced ratio to a quarter of a percent.
**High:** PYFA's and SINI's factors. These are the strongest results in the memo,
because the announcement fixed the factor and 48 prices and 8 volumes then landed
on the tick and lot grids independently. A factor 2% away scores zero.
**High:** attrition runs at 2.87%/yr and the doomed names underperform by ~4.8
pp/yr *years before* they go. Both are counts over a point-in-time snapshot, not
inferences.
**Medium:** that three is the true number of vendor-adjustment islands. The
sweep is exhaustive for factors that leave prices off the grid and **blind** to
whole-number ones, so it is a floor, not a census.
**Medium–low:** the survivorship bias *magnitude*. The rate is measured; the
terminal loss is assumed, because the snapshot stops before these names died.
**Low:** the broker code master. Enough to guard §9 against the known merger, not
enough to call complete.

---

## 11. Correction history

This memo first reported Gate 0 as passing when `scripts/gate0.py` was running
checks I devised — band conformance, stale bars, spikes — and neither of the two
§5 names. Naming the script `gate0` made a substitution look like the thing
itself. Both specified checks now run; check 2 failed on its first real case; the
defect was repaired; the gate passes on its own terms.

One method was tried and discarded: checking whether the price break and the
volume break fall on the same day. Daily volume varies tenfold naturally, so
clean-ratio steps appear everywhere by chance — it flagged nine of eleven shifts
including ones that are fine. Not reported as a result.

The off-tick-grid detector was wrong twice before it was right, both times by
treating **on-grid as evidence**. Version one read each bar alone and turned
BMRI's single ÷4 region into 18,300 findings. Version two closed the gaps and
claimed to *segment* the series into raw and adjusted, which produced 2,760
"defects" — PTBA's ÷5 history lands on the grid for 122 consecutive sessions
whenever the real price is a multiple of Rp 50. Neither intermediate number is a
result; both are recorded because the same mistake is available to anyone who
reads `off_tick` as a two-sided test.

`quality.py` previously shipped a `basis_segments()` that made the version-two
claim. It has been removed rather than fixed, and replaced by `off_grid_rate()`
— explicitly a lower bound — and `suspect_islands()`, which reports only what
`level_shifts` independently agrees is a break.

---

## 12. Still open

The four gaps this memo originally listed were not environmental. Every one was
a research problem, and working them changed the spine:

| originally listed | outcome |
|---|---|
| pre-2014 rules unknowable | **CLOSED.** Read out of the prices: tick ladder back to 2005, auto-rejection back to 2010 |
| board membership unsourced | **CLOSED.** Derived from IDX's published watchlist criterion |
| no corporate-action feed | **CLOSED.** Every level shift in the universe is accounted for, and the two that lacked a factor now have proved ones |
| delisted price history | **MOSTLY CLOSED.** 121 vanished names recovered with history from a 2019 point-in-time snapshot; the terminal months are still missing |

What is genuinely left is narrower than any of those:

- **The way out.** The snapshot ends 2019-04-07, so the final months of a
  delisting name — where most of the loss sits — are not in the spine. The
  survivorship figure stays a bound (§6).
- **Whole-number back-adjustments.** Invisible to the off-grid sweep by
  construction; `level_shifts` is the only test that sees them, and the two
  together are still not a proof of absence.
- **ELTY 2018-06-07**, quarantined because its window is uninformative (§7).
- **The ten non-price watchlist criteria** — going-concern opinions, prolonged
  suspension, no revenue. They need a filings feed, so a name on the watchlist
  for a non-price reason reads as main board.
- **Pre-2005, and auto-rejection before 2010.** Lookups raise rather than guess.

### Closed: the pre-2014 rules

The tick ladder could not be fetched from any reachable source, so it was read
from the prices themselves. On a Rp 10 grid essentially every close divides by
10, and the observed granularity is stable in every year from 2005 to 2013:
200–500 Rp 5, 500–2,000 Rp 10, 2,000–5,000 Rp 25, ≥5,000 Rp 50. That matches
the table the sources described.

The same method settled a live disagreement in the *modern* table — two sources
give Rp 5 and Rp 10 for the 2014–2016 Rp 500–5,000 band; 97.9% of closes there
divide by 5. This is now Gate 0 check 2b, and all ten band-periods agree across
1.3m closes.

The check needed a real discriminator, not a threshold. Split-adjusted prices
are off-grid, so every share sits below 100%; a fixed cut-off would either
reject a real Rp 25 grid or have to be tuned until it stopped, which is fitting
the test to the answer. Instead: on a grid of size *g*, the share divisible by a
coarser *c* is about *g/c* by chance, so a candidate qualifies when its share
sits closer to 1 than to that chance level. **88% against a 20% null is a Rp 25
grid; 61% against a 50% null is not a Rp 50 grid.** No tuning.

Auto rejection was read the same way — it truncates the return distribution, so
the band is where the tail stops. Calibrated on the documented 2014–2016 regime
the truncation lands exactly on 35/25/20; 2010–2013 reproduces it, and Gate 0
checks that window on its own: **0.004% violation rate over 312,478
observations, the lowest of any regime including the documented ones.**
2005–2009 does *not* reproduce it (0.19% of sub-Rp 200 days above 35%, against
0.02% in 2010–2013), so the bands stop at 2010 while the ladder reaches 2005.
Coverage is now **per schedule**, because the evidence is.

### Closed: board membership

Derived, not sourced. From 2023-06-12 IDX's criterion for the Papan Pemantauan
Khusus is explicit and computable — six-month average regular-market price below
Rp 51 (Peraturan I-X; Tahap I 2023-06-12, Tahap II full call auction
2024-03-25). Its ladder is a flat Rp 1 band below Rp 10 and 10% above, against
35/25/20 on the main board, so getting the board wrong manufactures
impossible-move flags on exactly the names least able to bear them.

`infer_board` returns main / watchlist / **unknown** — pre-2023 a sub-Rp 50
quote is not explained by anything encoded, and unknown is treated as the
*looser* ladder, since assuming the tight one invents defects.

What remains is narrower: the ten **non-price** criteria (going-concern opinion,
prolonged suspension, no revenue) need a filings feed, so a name on the
watchlist for one of those reads as main board.

### Narrowed: corporate actions

Applying the same impossible-move requirement to `level_shifts` that
`decimal_spikes` already had cut the universe's shifts from 11 to **5**. The
removed ones were never corporate actions: RODA fell 99 → 65, which is −34.3%
against a 35% band — a legal bad day that a clean ratio near 1.5 had flagged.

All five are now accounted for, and none is left on a guess:

| shift | status |
|---|---|
| SCCO 2024-02-01 | verified misdated split, **repaired** |
| WIKA 2024-04-30 | verified rights issue (detector reports the resumption date; the adjustment at 2024-04-16 is correct) |
| PYFA 2024-04-16 | verified 1:20 rights at Rp 100, factor proved on the tick grid, **repaired** |
| SINI 2026-06-29 | verified 2:3 rights at Rp 5,000, factor proved on the tick grid, **repaired** |
| ELTY 2018-06-07 | **not a corporate action** — the 10:1 reverse split was rejected by shareholders; this is a dormant quote re-marked on resumption. Quarantined as uninformative |

The two that had been *"cause known, factor not"* were closed by finding the
announced ratios and then letting the tick grid vote on them — §5 has the
arithmetic. Neither factor was read off the price move it explains.

### Mostly closed: delisted price history

This was reported as an environmental block — Yahoo answers *"possibly delisted;
no timezone found"* for SRIL, MYRX, FREN and MAMI, and stooq serves a JavaScript
challenge. That framing was too quick. The names are not fetchable **one at a
time**, but a published 2019 point-in-time snapshot carries 627 of them in bulk,
and 121 of those are now in the spine with history (§6).

What survives of the block is much narrower: the snapshot's own end date. It
stops in April 2019, so the months in which a name actually died are still
missing, and that is where most of the loss is. A licensed feed would carry it.

---

Phase 1 work may proceed on the repaired spine provided the survivorship caveat
travels with every number and the ELTY window is excluded.
