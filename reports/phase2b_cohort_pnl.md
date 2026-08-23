# Phase 2b memo — is the flow behind a broker code profitable?

**Date:** 2026-08-23
**Verdict: no detectable broker-identity signal in cohort P&L.**
Reproduce with `python3 scripts/cohort_pnl_run.py --null`.

---

## 1. The headline, and the number that decides it

| | median margin_bps | n | 95% CI |
|---|---|---|---|
| **round-trip, real labels** | **−25.3** | 147 broker-tickers, 628 episodes | [−395, −29] |
| **round-trip, SHUFFLED labels** | **−32.5** | 162, 883 episodes | [−278, −28] |

**The null is indistinguishable from the signal.** Shuffling which broker did
what leaves the cohort-P&L distribution essentially where it was. On this store,
**broker identity carries no detectable information about profitability.**

That is a legitimate and useful outcome — §9.5's Gate 2b says so explicitly:
*"If the broker-aware model does not improve out-of-sample IC over baseline,
identifying individual brokers is decorative and the project drops back to
aggregate flow."* This is the cohort-P&L half of that verdict, and it points the
same way as Phase 1's.

### The second finding, which is about the market rather than the brokers

Both numbers are **negative, and by a similar amount**. The median cohort round
trip loses ~25–32 bps of the value it traded, whoever did it. That is the spread
being paid: a buyer lifts the offer, a seller hits the bid, and any cohort doing
both inside one episode pays the gap twice.

For scale, A5's cost schedule is **56 bps a round trip**. So the median cohort
round trip is already a losing trade on execution alone, *before* a single
rupiah of commission or tax. That is worth carrying into §12: the counterparty
of a persistently losing cohort has to be beating a bar that starts below zero.

---

## 2. Neither store satisfies §9.3, for opposite reasons

§9.3 specifies a **daily** walk-forward and makes round-trip episodes the primary
estimate. Both requirements cannot be met at once with what exists.

| | Track A — daily | Track B — fortnightly |
|---|---|---|
| names | 9 (+ANTM with 2 sessions) | **176**, 64 delisted |
| span | ~360 sessions, 2025-02 … 2026-08 | 329 windows, 2014 … 2026 |
| resolution | **daily — correct** | fortnightly |
| round trips computable | **yes** | **no** |
| §9.3's 250-day burn-in | leaves **111 usable days** | ~304 windows, fine |

**Track A has the right resolution and not enough length.** Applying §9.3's
250-day burn-in to a 361-session series discards 69% of it and leaves **7
broker-tickers with any round trip at all, across 16 episodes**. That is not a
sample and no number is quoted from it — the burn-in row exists in the output
only so the trade-off is visible, per §9.3's instruction to report with and
without.

**Track B has the length and cannot compute a round trip at all.** A fortnight
gives net flow over ten sessions; the path inside the window is gone, and the
path is what an episode is made of. A cohort that buys 50,000 lots on Monday and
sells them on Friday reads as exactly zero.

**The two tables are never combined.** Averaging an exact 9-name figure with an
approximate 176-name one produces something that is neither and reads as more
authoritative than either.

---

## 3. The estimator was wrong twice, and the null caught it both times

**First version: +3,333 bps out of nothing.** §9.3's formula is
`realized_t = sell_vol_t × (sell_avg_price_t − WAC_{t−1})`, and the walk starts
with WAC at zero. So a cohort that was *already long* when the series began —
which is most of them — booked its opening sell's **entire proceeds as profit**.
On a synthetic case buying 10,000 shares at Rp 1,000 and selling 10,000 at
Rp 1,000, true P&L zero, the estimator returned **+Rp 10,000,000**.

The tell was the null: shuffled broker labels came back at **+6.3 bps with a 95%
CI of [+8.6, +71.4]**, excluding zero. No shuffle can honestly be profitable.

Two fixes, both of which §9.3 already implies:

- **Only the attributable part of a sell gets a cost basis.** Shares sold beyond
  recorded holdings came from inventory whose cost is unknowable; they are
  counted in `unattributable_sh` and booked at nothing. Inventory is still
  allowed to go negative, because that is the direct measurement of §9.2's
  starting-inventory problem.
- **Round-trip P&L is `sell value − buy value` inside the episode.** No weighted
  average cost, no starting inventory, nothing assumed. That is precisely what
  §9.3 means by *"unambiguous and independent of the initial-position problem"*,
  and using the WAC-based `realized` there would have dragged the contamination
  straight back in.

Verified on known cases: the synthetic false profit goes to **0**, and a genuine
round trip buying at 1,000 and selling at 1,100 returns **+476 bps** as it should.

**Second version: −13,000 bps.** The full-path estimate then produced margins of
−130% of gross traded value, which cannot happen. Cause: `unrealized =
inventory × (close − WAC)` on a **negative** final inventory — a position the
data never saw acquired, priced against a meaningless cost basis. It is now NaN
there rather than a number, and the share is reported: **full-path P&L is
computable on only 49% of series.**

---

## 4. §9.2's structural limits, measured rather than asserted

The brief requires these stated in every output. They are now quantified, and
two are worse than the prose suggests.

| limit | measurement |
|---|---|
| **starting inventory unknown** | reconstructed inventory is **negative 46.4% of the time** (median broker-ticker). Nearly half the series implies selling shares never seen bought. |
| **crossing inflates gross volume** | crossing ratio median **0.84** — the typical broker-ticker prints both sides at 84% overlap. Turnover with little directional exposure, and it sits in the denominator of every margin_bps. |
| **foreign nominees are omnibus** | not resolved; flagged low-confidence by construction and never treated as one actor |
| **codes not stable over history** | handled by the broker code master, partial (§8 of the Phase 0 memo) |

The 46.4% figure is the important one. It means the full-path estimate is not
merely noisy — for about half the series it is not defined. §9.3's instruction to
treat round-trip as primary and full-path as "the noisy secondary" understates
it: on this data full-path is often not a secondary estimate at all.

The 0.84 crossing ratio matters for a different reason. A cohort can look
unprofitable **per rupiah traded** simply by crossing a lot, because crossing
adds to gross value without adding exposure. Any per-rupiah comparison across
brokers has to condition on it.

---

## 5. What Track B can say without claiming a round trip

| | value |
|---|---|
| net imbalance | median +0.0000, mean −0.0011, sd 0.1060 |
| foreign net share | median +0.0000, mean −0.0052 |
| domestic net share | median +0.0014, mean +0.0041 |
| two-sided codes | median **7 of 13** |

That last row is a constraint on everything downstream. The source ranks buyers
and sellers **independently**, so a per-broker *net* only exists for codes
appearing in both top tens — about half of them. For the other half one side is
an unknown lower bound, not zero.

---

## 6. What I believe, and with what confidence

**High:** broker identity does not beat a label shuffle on cohort P&L in this
store. The two distributions sit on top of each other.

**High:** the median cohort round trip loses the spread, ~25–32 bps, before any
fee. Both real and shuffled labels agree, which is what makes it a market fact
rather than a broker fact.

**High:** §9.2's starting-inventory problem is severe here — 46.4% negative
inventory — and it is the reason the clean estimate had to be episode-based.

**Medium:** that 628 episodes across 9 names generalises. These are 9 of the most
liquid names on IDX over 18 months. Nothing here speaks to small caps, and the
crossing ratio is likely very different there.

**Low, and deliberately unstated:** any claim about *why* a particular code's
cohort loses. §6 point 4 forbids it and nothing here identifies a mechanism.

---

## 7. What would have falsified this

A real broker-identity effect would have shown the shuffled null collapsing
toward zero while the real labels held a distinct, non-overlapping distribution.
It did not: the null sits within a few bps of the signal with heavily
overlapping intervals.

---

## 8. Where this leaves Gate 2b and §12

Gate 2b asks whether broker identity beats aggregate flow. Two halves now point
the same way: Phase 1 measured the aggregate-flow baseline at essentially
nothing, and this memo finds no identity signal in cohort P&L either.

The remaining §12 question is **not** answered by this memo and should not be
read as answered: *which flow is persistently dumb, is it identifiable in real
time, and is it large enough to trade against after costs.* Persistence is the
part untested here — a cohort losing 25 bps a round trip is not interesting
unless the *same* cohort keeps doing it, and 18 months on 9 names cannot
establish that. §12's own argument is that persistence is what makes the losing
side durable, and testing it needs either a longer daily store or a way to
detect persistence at fortnightly resolution.
