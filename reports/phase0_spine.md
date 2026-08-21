# Phase 0 memo — the data spine

**Date:** 2026-08-21
**Gate 0: PASSES on the checks it can run.** Two hard checks pass; four defects
are now quantified rather than unknown; three §5 requirements remain open and
are named below rather than glossed.

Run it: `python3 scripts/gate0.py`

---

## 1. What was built

| §5 requirement | status |
|---|---|
| ARA/ARB schedule, incl. asymmetric period | **done** — `spine/reference.py`, 6 regimes |
| Fraksi harga (tick) schedule | **done** — 2 regimes |
| Lot size, session halts | **done** |
| Broker code master with effective dates | **partial** — `spine/brokers.py`, honestly labelled |
| Rights-issue / corporate-action adjustment | **done** — `spine/corporate_actions.py` |
| Delisted and suspended names | **NOT DONE** — measured, not fixed. §5 |
| Suspension / ARA-ARB flags per ticker-day | **done** — `spine/quality.py::locked_bars` |
| Foreign net flow per ticker | pre-existing, in the broker store |
| Gate 0 reconciliation | **done** — `scripts/gate0.py` |

1,459 tests pass. Everything below is reproducible from the script.

---

## 2. The rules changed six times, and twice they changed back

This is the part that would have silently corrupted everything. Auto rejection:

```
2016-01-04  symmetric 35 / 25 / 20 by price band
2020-03-10  ARB -> 10%  (COVID, three trading days)
2020-03-13  ARB -> 7%
2023-06-05  ARB -> 15%  normalisation tahap I
2023-09-04  ARB = ARA, symmetric again, tahap II
2025-04-08  ARB -> 15%, asymmetric AGAIN (Kep-00003/BEI/04-2025)
```

**The 2025 change lands inside the broker panel this repo collected.** Anything
assuming the September 2023 symmetric regime is wrong for every session after
8 April 2025 — which is most of the panel.

Tick size: three groups until 2016-05-02, five after. Applying today's ladder to
2015 understates the half-spread on a Rp 300 stock by half, always flatteringly.

Every lookup takes a date and **raises** for anything before 2014-01-06 rather
than falling back on the nearest entry. A silent fallback is the exact bug the
module exists to prevent.

---

## 3. Rules vs reality — the check that validates the schedule

Encoded bands against 843 tickers and 2.6m bars. If the schedule were mis-dated,
real falls would breach it.

| regime | limit | observations | past floor | rate |
|---|---|---|---|---|
| symmetric 25% | 25% | 356,469 | 24 | 0.007% |
| COVID 10% | 10% | 1,369 | 8 | 0.584% |
| COVID 7% | 7% | 458,687 | 568 | 0.124% |
| normalisation 15% | 15% | 42,131 | 6 | 0.014% |
| symmetric again | 25% | 285,229 | 22 | 0.008% |
| asymmetric 15% | 15% | 261,842 | 25 | 0.010% |

Worst rate 0.58% against a 2% tripwire. **PASS.** Residual breaches are
resumptions after suspension and unadjusted corporate actions, both enumerated.

Independent confirmation: CBRE's largest repeated falls are −14.8%, sitting
exactly on the encoded 15% floor.

---

## 4. Four defects found in the data itself

The band check was meant to validate the *rules*. It also surfaced this:

| defect | scale | consequence |
|---|---|---|
| **stale bars** | **421,942 = 16.2% of the spine** | days the stock did not trade |
| unadjusted corporate actions | 11 across 9 tickers | fake crashes up to −75% |
| decimal source errors | 7 bars, 2 tickers | whole row 10× or 1/10th |
| survivorship | **0 of 25 known-delisted names present** | universe is winners only |

**The stale bars are the important one and the least dramatic.** One bar in six
records no trading and repeats the previous close; some names are over 70%. A
backtest that fills on one has bought from nobody, and a return series that
keeps one reports a real zero where there was no observation.

### The split detector was wrong first time

Ratio alone flagged 79 "unadjusted corporate actions" and most were nonsense:
BTEK going Rp 3 → Rp 2 is a ratio of 1.5 and **one tick**. IDX has hundreds of
names in single rupiah where an ordinary tick is a 33–50% move. Requiring the
move to be large in *ticks* as well cut 79 → 11. That correction is the useful
part of this section.

---

## 5. Survivorship: measured, not fixed

843 tickers and **not one stopped trading more than two years ago.** Of 25
companies known delisted from IDX, 0 are present. ~70 delisted in 2025 alone.
The ticker list came from a `TICKER,marketcap` file, which can only contain live
names.

**The fix §5 asks for is not available.** Yahoo answers *"possibly delisted; no
timezone found"* for SRIL, MYRX, FREN and MAMI, repeatedly. No other free source
reached carries delisted `.JK` history.

So the bias is bounded rather than corrected:

| delist rate | equal-weight | cap-weight |
|---|---|---|
| 1% | 0.9 pp | 0.05 pp |
| 4% | 3.7 pp | 0.19 pp |
| 8% | 7.4 pp | 0.37 pp |

**The weighting column is the finding.** A name about to delist is a micro cap,
so a cap-weighted book barely holds it — the bias is 20× smaller there. The
repo's large-cap work is close to safe; its equal-weight small-cap work is not.
That distinction was previously unstated.

No correction factor is applied. The correction is not identifiable without the
delisted history, and applying one would turn a known gap into a hidden
assumption.

---

## 6. Rights issues — the trap §5 names

A split is a relabelling. A rights issue is a **transfer of value at a chosen
price**, so the ex-date fall depends on three numbers that are not in the price
series:

```
TERP = (held × P_cum + new × subscription) / (held + new)
```

The naive fix — "large drop, clean ratio, must be a split" — is exactly wrong
here: it computes the factor from the observed drop, which is the thing being
explained, and thereby defines every rights issue as already correct. So
`adjustment_factor` **raises** rather than deriving anything from the tape.

Fixtures go to 1-for-1 at a 90% discount and 10-for-1 at a 95% discount — events
removing 45% and 86% of the quote while costing a participating holder nothing.
The acceptance test has two halves: the 86% fake crash adjusts to **exactly
zero**, and a *real* 10% fall on an ex-date **survives**. An adjustment that
flattened everything would pass the first and destroy the data.

---

## 7. Broker code master — the distinction that decides its value

§5 warns that mergers reassign codes. True, but the shape of the risk is not
what it looks like:

- **A rename is not a reassignment.** YP was eTrading (2003) → Daewoo (2013) →
  Mirae Asset (2016): three names, one continuous business, one client base.
  Splitting YP at each rename would destroy a real fifteen-year record.
- **A merger is a discontinuity.** When UBS absorbed Credit Suisse the flow
  behind CS did not gradually become UBS flow; it moved.

Confidence is part of the data, with three answers never collapsed:
`verified`/`reported` (dated record), `current_only` (name today known, history
unknown — safe to label, never safe to compare eras with), `unknown`.

Coverage on the panel: 6 codes dated, 60 current-name-only, rest unknown. Modest
and honestly labelled rather than 78 confident guesses.

The empirical audit is deliberately weakened: on a **top-ten** source a code is
listed only on days it ranked, so absence means smallness. It flags RB, PI, MU
on the panel and none is a new licence. On a full-depth rekap the check becomes
strong.

---

## 8. What would have falsified this

A regime mis-dated by even a week would have shown a violation rate far above
0.58% in the affected window. It did not. The tick ladder is confirmed by CBRE
sitting exactly on the encoded floor. The corporate-action adjustment is
confirmed by SCCO's real 1:4 adjusting to exactly zero.

## 9. What I believe, and with what confidence

**High:** the encoded ARA/ARB and tick schedules are correct for 2014-01-06
onward. Two independent checks agree and 99.4–99.99% of 2.6m bars sit inside the
bands.

**High:** the spine is totally survivorship-biased, and this matters far more
for equal-weight small-cap work than for cap-weighted large-cap work.

**Medium:** the 11 detected level shifts are genuine corporate actions. They are
*candidates* — no announcement feed has confirmed them.

**Low / unresolved:** the broker code master. 6 of 78 panel codes have dated
history. It is enough to guard §9's fingerprints against the known merger and
not enough to be called complete.

---

## 10. Still open before §5 is fully met

1. **Delisted price history.** Measured, not fixed. Needs a paid or licensed
   source. This is the largest remaining gap.
2. **A corporate-action feed.** The module can adjust; nothing supplies terms.
   Detection is not adjustment.
3. **Board membership per ticker-day.** The acceleration/watchlist ladder is
   encoded but nothing says which names were on it when.
4. **Pre-2014 rules.** Lookups raise. The 500-share-lot era is unmodelled.

None of these blocks the §12 cohort-P&L work on the existing panel, provided the
survivorship caveat travels with every number.
