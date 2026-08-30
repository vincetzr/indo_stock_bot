# H53 — do small and large IDX names move differently? Yes, monotonically.

*2026-08-30. `scripts/sizeprofile.py`. Pre-registration S1–S5 in the module
docstring. 2,323,396 bars, 919 names, 2001–2026. Five buckets formed **within
each date** on trailing 60-day median turnover, so the study measures size and
not the calendar. Size is proxied by turnover because no point-in-time share
count exists here (A25).*

---

## 1. They behave differently, and every column is monotone

| bucket | names | median Rp/day | median px | ann vol | **stale days** | **AR(1)** | round trip |
|---|---|---|---|---|---|---|---|
| thinnest | 648 | 745k | 187 | **89.7%** | **42.1%** | **−0.086** | **1.94%** |
| thin | 771 | 18m | 215 | 69.9% | 34.3% | −0.071 | 1.73% |
| mid | 841 | 152m | 218 | 65.6% | 30.6% | −0.034 | 1.58% |
| thick | 813 | 1.43bn | 320 | 58.4% | 24.3% | −0.027 | 1.41% |
| thickest | 558 | 20.2bn | 1,025 | **53.6%** | **15.4%** | **+0.025** | **1.10%** |

**The daily autocorrelation changes SIGN.** Thin names mean-revert day to day
(−0.086: bid-ask bounce and stale-print correction); the largest names show mild
*continuation* (+0.025). That is a textbook microstructure difference and it is
clean and monotone across all five buckets.

**42% of days in the thinnest bucket are exact zero returns.** Those names do
not trade; their return arrives in lumps, and any feature that knows the price
is stale "predicts" the catch-up. This is the mechanism behind H44's withdrawn
headline.

## 2. The signals do not work in the same places

Information coefficient, `fwd20`, Spearman within date:

| feature | thinnest | thin | mid | thick | thickest |
|---|---|---|---|---|---|
| `hi52` strength | **−0.008** | +0.038 | +0.070 | **+0.090** | +0.069 |
| `mom12_1` | +0.002 | +0.034 | +0.043 | +0.039 | +0.025 |
| `lowvol` | +0.042 | +0.062 | **+0.083** | +0.074 | +0.052 |
| `pastret5` reversal | **−0.046** | −0.011 | +0.011 | **+0.025** | +0.017 |
| `squeeze` **(null)** | −0.017 | −0.021 | −0.011 | +0.014 | +0.026 |

**S2 FAILED, and it is the opposite of what I predicted.** I registered momentum
and strength as *stronger in small names*. They are **absent** there and
strongest in the thick bucket: `hi52` runs −0.008 → +0.090.

**S1 CONFIRMED, more sharply than registered.** `pastret5` flips sign: −0.046 in
the thinnest (reversal) to +0.025 in the thick (continuation).

**S3 FAILED.** `lowvol` peaks in the middle and works everywhere; it has almost
no size dependence (thin−thick −0.011).

**S4 — THE PREDICTED NULL FIRED**, |t|>3 in 5 of 10 cells. So **t-statistics are
not usable here** and every effect must be read against `squeeze`'s own
magnitude, not against zero. That is A9's lesson: at two million observations
significance is free.

## 3. The money number, which decides the tuning

Long-only top-quintile tilt inside each bucket, 20-session hold, against that
bucket's own mean, net of that bucket's own round trip:

| feature | bucket | dispersion | **gross/20d** | cost | net/20d |
|---|---|---|---|---|---|
| `hi52` | thinnest | 21.2% | **−0.28%** | 1.95% | −2.23% |
| `hi52` | mid | 16.1% | +0.90% | 1.58% | −0.68% |
| **`hi52`** | **thick** | 15.1% | **+1.26%** | 1.40% | **−0.14%** |
| `hi52` | thickest | 12.5% | +0.50% | 1.10% | −0.60% |
| `squeeze` **(null)** | thick | 15.1% | **+0.68%** | 1.40% | −0.72% |
| `squeeze` **(null)** | thickest | 12.5% | **+0.62%** | 1.10% | −0.47% |

**Two things fall out and they are the whole answer.**

**The `thick` bucket is the only segment where a signal clearly beats its own
null** — `hi52` +1.26% against `squeeze` +0.68%, roughly 2×. In the thickest
bucket `hi52` (+0.50%) is *below* the null (+0.62%): indistinguishable from
noise. In the thinnest it is negative.

**`lowvol` has a strongly positive IC and a NEGATIVE spread in every bucket**
(−1.35% to −0.06%). That reproduces A9 exactly: a rank tilt is not a return
spread. Low volatility is a **filter**, not a **tilt** — it tells you what to
exclude, not what to overweight.

## 4. Reconciling this with H52

H52 concluded the screen's alpha "IS a size effect", strongest in small/mid.
H53's ICs say the opposite — strength peaks in the *thick* bucket. Both are
right and the wedge is dispersion: **cross-sectional dispersion falls
monotonically, 21.2% → 12.5%**, so an identical IC converts to less money at the
large end. H52 measured money in a narrow top-40 universe where dispersion is
lowest; H53 measures rank correlation. **The optimum is neither extreme — it is
the `thick` bucket, where IC is near its peak and dispersion is still 15%.**

## 5. Trials

S1–S5 is 5 registered tests; the IC and spread tables are a map.
**Trials after H53: 315.** Bonferroni bar **0.00016** — and irrelevant here,
because the registered null fires and effect size against `squeeze` is the only
usable discriminator. Every net column is negative at a 20-session rebalance:
the frequency is wrong for every segment, which H43 measured separately
(quarterly/six-month optimum).
