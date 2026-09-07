# H57 — a point-in-time share count, and the conclusion it lets us re-test

*2026-09-07. `scripts/shares_pit.py`, `tests/test_shares_pit.py`,
`data/spine/shares_pit.parquet`. Registered in the module docstring before any
cell was scored; the registration is unaltered and a test pins it there.*

---

## 1. What was blocked, and why it stopped being blocked

A25 records market capitalisation as **unusable** in this project. The only
share-count source found was `wildangunawan/Dataset-Saham-IDX`'s `shares`
column, frozen at 2024-07-10, and applying a 2024 count to a 2010 bar is
look-ahead — Indonesian rights issues are precisely what makes it wrong. So
every study that needed **size** proxied it with trailing turnover. H52 is the
one that matters: its conclusion is that H26's strength-plus-calm alpha **is a
size effect**, because the edge collapses from +12.9%/yr on the whole board to
+0.7% in the top 40. **Every tier in that study was cut on turnover.**

The delisted-recovery dataset brought in at roadmap stage 0b carries
`listed_shares` **per bar**, which is a different column from the frozen
`shares` one and is not look-ahead.

**How much does it move?**

| | |
|---|---|
| names whose count ever changes | **405 of 949 (43%)** |
| largest ratio | **PANI 410,000,000 → 16,883,595,500 (41×)** |
| next | PYFA 21×, BBKP 16× over **26 distinct values** |
| store | 1,078,040 rows, 958 names, 2019-07-29 → 2025-02-21 |
| coverage of eligible panel bars in that window | **100.0%** |

A frozen count misprices PANI's 2019 capitalisation by a factor of forty. This
is not a rounding difference and A25's refusal was correct.

Outside the dataset's own 5.6-year span there is still no point-in-time count.
This **adds a column**; it does not make market cap available to the rest of
the repo.

---

## 2. The join, and the one way it could have been wrong

`attach()` is a backward-only `merge_asof` by ticker. A share count is known
from the day it takes effect and never before, so a **backward fill would put
a post-rights-issue count on pre-issue bars** — pricing a company at its
post-dilution size months early, in exactly the names that later issued equity,
with nothing in the output looking wrong.

That is the one test this module exists for
(`test_attach_never_carries_a_count_backwards`). Bars before a name's first
recorded count get **no cap at all** rather than a guess at the first one, and
capitalisation is priced at the **traded** close, never the back-adjusted one —
A19 measured back-adjustment moving `adj_close` away from `close` at every
corporate action, so `adj_close × shares` would report size in a currency that
never existed.

---

## 3. P1 — the proxy is good and its errors are systematic

**Registered:** Spearman above 0.7, residual concentrated in low-turnover large
caps.

**Measured**, within date, on 278,750 eligible bars and 611 names:

| | |
|---|---|
| Spearman(turnover rank, cap rank) | **+0.650** |
| median absolute rank gap | 0.137 |
| p90 absolute rank gap | 0.414 |
| big-but-thin (cap top 20%, turnover bottom half) | 4,859 bars, **32 names** |
| small-but-busy (cap bottom half, turnover top 20%) | 4,869 bars, 60 names |

Most misranked by turnover: **MYOR, CMRY, BYAN, BNGA, POLL, BELI, TBIG, GEMS**
— large companies with tightly held registers that trade rarely. The correlation
came in below the registered 0.7 and the *shape* of the disagreement is exactly
as predicted, so P1 is confirmed on its substance and its number is recorded as
missed.

---

## 4. C0 — the positive control, and it failed on the obvious statistic

P2 asks whether H52's tier collapse is a property of its **ruler**. That is
only answerable if the turnover ruler — H52's own — reproduces H52's own
collapse **on whatever statistic is used**. If it does not, a difference
between the two arms is a fact about the statistic, and reading it as a ruler
effect is the error A26's sine-wave control and A36's Q0 martingale exist to
stop.

Full panel, 2000-07-13 → 2026-09-04, 690,591 eligible bars, 741 names, H52's
tiers (`None, 150, 100, 60, 40`, ranked within date) and H26's screen
(`hi52 ≥ q90 AND vol60 ≤ q50`, cut within date) verbatim.

**On the arithmetic mean of the 60-session forward return, it does not
reproduce:**

| tier | screen | rest | edge |
|---|---|---|---|
| all | +0.0706 | +0.0752 | **−0.0046** |
| top 150 | +0.0642 | +0.0597 | +0.0045 |
| top 100 | +0.0639 | +0.0547 | **+0.0092** |
| top 60 | +0.0661 | +0.0575 | +0.0085 |
| top 40 | +0.0625 | +0.0601 | **+0.0024** |

Hump-shaped, negative at the whole board, peaking in the middle. **A market-cap
arm differing from this would have said nothing whatever about market cap.**

**On the mean log it is there, and monotone:**

| tier | n | screen | rest | edge | null | sd | z |
|---|---|---|---|---|---|---|---|
| all | 42,432 | +0.0312 | +0.0058 | **+0.0253** | −0.0001 | 0.0101 | +2.50 |
| top 150 | 35,600 | +0.0245 | +0.0027 | +0.0217 | −0.0003 | 0.0101 | +2.19 |
| top 100 | 29,708 | +0.0262 | +0.0043 | +0.0219 | −0.0004 | 0.0105 | +2.12 |
| top 60 | 22,649 | +0.0293 | +0.0090 | +0.0203 | +0.0000 | 0.0121 | +1.67 |
| top 40 | 18,901 | +0.0302 | +0.0139 | **+0.0164** | −0.0003 | 0.0140 | +1.19 |

**A36 records the arithmetic mean and the mean log disagreeing in SIGN on this
repo's data. Here they disagree about whether the effect EXISTS.** The mean log
is the arm P2 is scored on because it is the quantity H52 measured — a
compounded portfolio — not because it is the arm that happened to pass. The
choice is stated before the result rather than inferred from it.

Nulls throughout are clustered: whole `(ticker, year)` blocks have their
**labels reassigned to other blocks' rows**, per A17 (a row shuffle leaves the
null far too tight) and A25 (shuffling *inside* a block is nearly a no-op).
`tests/test_shares_pit.py` asserts the clustered null is more than twice as wide
as an iid one on synthetic clustered data, rather than asserting the code is
correct.

---

## 5. P2 — confirmed, and the answer is that the ruler does not matter

Row-matched: both arms computed on an **identical frame** of 15,691 screen bars
in 2019-07-29 → 2025-02-21. Running the turnover ruler on the same rows (C1) is
what separates "the ruler changed the answer" from "the window changed the
answer" — A19's error class, committed twice in this repo already.

| ruler | all | top 150 | top 100 | top 60 | top 40 |
|---|---|---|---|---|---|
| turnover (C1) | +0.0358 | +0.0250 | +0.0248 | +0.0207 | **+0.0184** |
| point-in-time market cap (P2) | +0.0358 | +0.0307 | +0.0169 | +0.0161 | **+0.0185** |

**H52's size conclusion is not an artefact of its proxy.** The predicted null
held: the same collapse appears when the tiers are cut on real capitalisation,
and the two ladders land within 0.0001 of each other at the tier that matters.

---

## 6. And it is an early-half phenomenon — which a boolean would have hidden

Fall from `all` to `top 40`, split on the return period:

| ruler | early | late |
|---|---|---|
| turnover | **+0.0292** | **+0.0029** |
| market cap | **+0.0278** | **+0.0038** |

Same sign in all four cells and a **factor of ten** apart in size. A verdict of
"declines in both halves" is True four times over and reports none of that.
A33 records exactly this — *a binary verdict on a ratio near one is a way of
not reporting a null* — so the script prints the magnitude of the fall, not a
boolean.

The two rulers also agree only to within **0.0199** across the ten (tier, half)
cells, which is **68% of the largest fall being measured**. The agreement is
directional, not tight.

---

## 7. P3 — a lead, and it is named as one

Where the rulers disagree most:

| cell | screen | rest | edge | null | z | n |
|---|---|---|---|---|---|---|
| cap ≫ turnover (**thin big**) | +0.0355 | −0.0105 | **+0.0460** | −0.0006 ± 0.0218 | **+2.14** | 6,024 |
| cap ≪ turnover (busy small) | +0.0131 | −0.0431 | +0.0562 | +0.0025 ± 0.0633 | +0.85 | 1,422 |

Market cap does mark a cell turnover misses: the screen's largest edge in the
window sits in **large companies that trade rarely**, above the pooled +0.0358.
The mirror cell has a null sd of 0.0633 on 1,422 rows and is unreadable — the
degenerate-cell shape A19 records three times, which is why its larger point
estimate is not the headline.

**+2.14 is nowhere near the Bonferroni bar of 0.00014, and the cell was chosen
after seeing P1.** It is a lead. P3's registered prediction (cap adds nothing)
is therefore neither confirmed nor overturned.

---

## 8. What this does not establish

**A per-bar gross mean log is not a costed, quarterly-rebalanced,
turnover-charged portfolio CAGR.** H52 reported +12.9% → +0.7%/yr, a 95%
collapse. This reproduces roughly a 35% one. The study answers the **ruler**
question and reproduces the **shape**; it says nothing about the **level**, and
it does not independently establish the effect the two rulers agree about.

Three further limits, stated rather than left to be found:

- **The window is 5.6 years of one regime.** The cap ruler exists nowhere else.
- **Costs are absent.** A38's rule applies: fee and spread are separate
  quantities and neither is in any number above.
- **The holdout was spent at H16.** Every number here is in-sample.

---

## 9. What is now available to the rest of the repo

`data/spine/shares_pit.parquet` — `(ticker, date, listed_shares)`, 1,078,040
rows, 958 names, 2019-07-29 → 2025-02-21, rebuildable with
`python scripts/shares_pit.py --rebuild`. `attach()` joins it to any panel
frame backward-only and adds a `mktcap` column.

It is **not** wired into `features/` or the shipped rule card, and should not be
until a source covering the full panel span exists: a size factor available for
5.6 of 26 years would silently restrict every study that used it to that window,
which is the confound this study had to spend a positive control to see around.
