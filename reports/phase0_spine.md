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
| **Delisted and suspended names** | **NOT DONE** — measured, not fixed (§6) |
| ARA/ARB schedule incl. asymmetric period | done — 6 regimes, back to 2010 |
| Fraksi harga (tick) schedule | done — 3 regimes, back to 2005, validated against 1.3m quoted closes |
| Lot size (incl. the 500-share era), index halts | done |
| Board membership per ticker-day | **derived** from IDX's published watchlist criterion (§12) |
| Broker code master with effective dates | partial, honestly labelled (§8) |
| Rights-issue / corporate-action adjustment | done, with fixtures to 86% dilution |
| Suspension / ARA-ARB flags per ticker-day | done |
| **Gate 0 check 1** — traded value | **PASS** (§4) |
| **Gate 0 check 2** — 5 events by hand | **PASS**, 7 checked (§5) |

1,523 tests pass. Gate 0 runs eight checks and exits non-zero on any failure.

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

## 5. Gate 0 check 2 — corporate actions by hand: PASS after a repair

Seven events verified against Indonesian market announcements: BBCA 1:5, BMRI
1:2, SCCO 1:4, WIKA's rights issue, DSSA 1:10 and 1:25, ISAT 1:4.

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

---

## 6. Survivorship: measured, not fixed

843 tickers and **not one stopped trading more than two years ago**. Of 25
companies known delisted from IDX, **0 are present**. ~70 delisted in 2025 alone.

Not obtainable **from this container**: Yahoo returns `possibly delisted; no
timezone found` for SRIL, MYRX, FREN, MAMI and is otherwise rate-limited (429);
stooq serves a JavaScript challenge. That is an environmental limit as much as a
claim about the world — a licensed feed would very likely carry it.

| delist rate | equal-weight | cap-weight |
|---|---|---|
| 1% | 0.9 pp | 0.05 pp |
| 4% | 3.7 pp | 0.19 pp |
| 8% | 7.4 pp | 0.37 pp |

**The weighting column is the finding.** A name about to delist is a micro cap,
so the bias is 20× smaller cap-weighted. The repo's large-cap work is close to
safe; equal-weight small-cap work is not. **No correction factor is applied** —
it is not identifiable without the delisted history.

---

## 7. What is quarantined rather than trusted

SCCO proved a detected shift can be confidently wrong about its own date, so
anything unconfirmed is **quarantined** in `spine/repairs.SUSPECT` with a
±45-day window (wider than SCCO's 36-day error) and treated as neither a
corporate action nor a real price move.

The list is now three entries, not ten. Four former entries were never corporate
actions at all — see §12 — and ELTY stays only because its window is
uninformative, not because its cause is unknown.

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
**High:** the spine is totally survivorship-biased, and this matters far more for
equal-weight small-cap work than cap-weighted large-cap work.
**High:** SCCO's split was misdated, and the repair is right — the restored
boundary matches the announced ratio to a quarter of a percent.
**Medium:** the ten quarantined shifts are corporate actions. They are candidates;
one of the same kind turned out misdated.
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

---

## 12. Still open — two items, both narrow

The four gaps this memo originally listed were not all environmental. Three of
them were research problems, and closing them changed the spine:

| originally listed | outcome |
|---|---|
| pre-2014 rules unknowable | **CLOSED.** Read out of the prices: tick ladder back to 2005, auto-rejection back to 2010 |
| board membership unsourced | **CLOSED.** Derived from IDX's published watchlist criterion |
| no corporate-action feed | **NARROWED.** Every level shift in the universe is now accounted for; two lack a confirmed factor |
| delisted price history | **still open.** Not obtainable from this container |

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

All five are now accounted for:

| shift | status |
|---|---|
| SCCO 2024-02-01 | verified misdated split, **repaired** |
| WIKA 2024-04-30 | verified rights issue (detector reports the resumption date; the adjustment at 2024-04-16 is correct) |
| ELTY 2018-06-07 | **not a corporate action** — the 10:1 reverse split was rejected by shareholders; this is a dormant quote re-marked on resumption |
| PYFA 2024-04-16 | cause known, factor not — see below |
| SINI 2026-06-29 | cause genuinely unknown |

PYFA announced a rights issue on 2024-04-04: 10.70bn new shares at Rp 100
raising Rp 1.07tn, and the arithmetic is internally consistent. The ex-date
lands on the shift. But the **ratio** was not confirmed from any source read
here, and deriving it from the price move would explain the move with itself —
so the window stays quarantined rather than adjusted on a guessed factor.

### Still open: delisted price history

The one genuine environmental block. Yahoo answers *"possibly delisted; no
timezone found"* for SRIL, MYRX, FREN and MAMI and is otherwise rate-limited;
stooq serves a JavaScript challenge. Measured rather than fixed (§6), and a
licensed feed would very likely carry it.

---

Phase 1 work may proceed on the repaired spine provided the survivorship caveat
travels with every number and quarantined windows are excluded.
