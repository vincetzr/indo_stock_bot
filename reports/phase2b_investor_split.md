# §12 memo — foreign vs domestic: is an investor CLASS persistently on the wrong side?

**Date:** 2026-08-23
**Verdict: no. All three pre-registered conditions fail, for both classes.**
Reproduce with `python3 scripts/investor_split_run.py --draws 200`.
Pre-registration: `hypotheses.md`, H12, written before the data existed.

---

## 1. Why this test, and why it is the last one on this branch

H11 found that a broker code's margin rank does not persist, and closed by
saying that finding was about the **instrument**, not the phenomenon: a broker
code aggregates thousands of accounts of mixed type (§6.1), while the Taiwan
and Finland results §12 cites identify account **types**. IDX tags every trade
with the investor's domicile and IndoPremier serves that split, so §12's
question could finally be asked with an instrument that matches it.

This is therefore not another variation on a failed idea. It is the one
remaining free instrument that the literature's own argument points at.

---

## 2. The answer

18 names (liquidity decile 9), 329 fortnights, **5,993 class-windows**,
2014-01-14 … 2026-08-11. 2,935 ticker-windows carry both views.

| | FOREIGN | DOMESTIC |
|---|---|---|
| share of gross | 47% | 53% |
| **margin** | **−1.70 bps** / fortnight | **+1.02 bps** |
| direction null | +0.55 [−12.72, +11.29] — **p 0.692** | +0.12 [−8.35, +11.86] — **p 0.816** |
| selection null | −6.71 [−32.16, +12.11] — p 0.587 | +4.02 [−7.73, +17.17] — p 0.627 |
| years sharing the pooled sign | **38%** of 13 | **46%** of 13 |
| lag-1 autocorrelation of annual margin | **−0.410** | **−0.331** |
| pooled annual mean → dropping largest year | −2.53 → **+0.67** | +0.18 → **−1.89** |
| \|margin\| vs 56 bps round trip | 1.70 — **fails by 33×** | 1.02 — **fails by 55×** |

**All three conditions fail, both classes.** Nothing is close to any of them.

### The prediction was wrong in sign, and that is recorded rather than reframed

H12 pre-registered **foreign > 0 > domestic**, on the reasoning that foreign
participation in IDX is overwhelmingly institutional and domestic flow carries
essentially all of Indonesian retail. The observed signs are the **opposite**:
foreign −1.70, domestic +1.02. Both are deep inside their nulls, so the
reversal is not a finding either — but it is the pre-registered direction
failing, and it goes in the log that way.

---

## 3. The result is not "we could not tell". A tradeable effect would have shown.

This is the part that makes the negative result worth something.

| | null sd | detectable at p<0.05 | where 56 bps sits |
|---|---|---|---|
| foreign | 6.34 bps | ±12.4 bps | **8.8 null-sds away** |
| domestic | 4.95 bps | ±9.7 bps | **11.3 null-sds away** |

The cost bar that §12's strategy has to clear is **eight to eleven standard
deviations outside** what this test could not distinguish from noise. So the
conclusion is not that the sample is too small — it is that an effect large
enough to trade against, had one existed, would have been found comfortably.
What is there instead is roughly 1–2 bps a fortnight, in a sign that will not
hold still.

---

## 4. Persistence fails in the strongest available way: it reverses

The persistence condition does not merely come out weak. Both classes have
**negative** lag-1 autocorrelation of the annual margin — −0.410 for foreign,
−0.331 for domestic. A good year tends to be followed by a bad one. That is
mean reversion, which is the opposite of the property §12's strategy needs,
and it is the reason the pooled figure is near zero.

```
foreign   2014 +4  2015 -8  2016 -41  2017 +7  2018 +3  2019 -10  2020 +6
          2021 +2  2022 -9  2023 +6   2024 +25 2025 -34 2026 +17
domestic  2014 +9  2015 +4  2016 +25  2017 -12 2018 -6  2019 -3   2020 -8
          2021 +9  2022 +7  2023 -6   2024 -21 2025 +20 2026 -15
```

**And the H11 detector fires again.** Dropping the single largest year flips
the pooled sign for *both* classes: foreign −2.53 → +0.67, domestic +0.18 →
−1.89. A pooled average whose sign depends on one of thirteen years is not
describing a durable property. This check exists in the code because H11's
headline was carried by a single thin year; here it earns its place a second
time.

---

## 5. The measure behaves the way market structure says it must

Two internal checks, both passed, which is what licenses reading the numbers at
all:

- **Zero-sum.** Every rupiah bought is a rupiah sold, so the two classes must
  be near mirror images. Annual margins correlate **−0.896**, and the signs are
  opposite in **10 of 13 years**.
- **Censoring bound, measured not assumed.** F_net and D_net are structurally
  exact mirrors, so their residual *is* the top-10 cut: **2.29% of gross at the
  median, 9.43% at p90**.

That second number needs its caveat stated plainly. Per window the censoring
error is far larger than the margin being measured. It averages down across
2,935 windows *if it is unbiased*, which is plausible but not proven. So **a
positive result of 1–2 bps from this design could not have been trusted** — and
that costs nothing here, because the bar that matters is 56 bps and the test
resolves ±10–12.

---

## 6. What this does and does not say

**Does not say the Taiwan/Finland literature is wrong.** Foreign-versus-
domestic is an imperfect proxy for institution-versus-retail: domestic
institutions are large in IDX and are pooled into "domestic", which dilutes
exactly the contrast the literature draws. A cleaner test needs an account-type
split that IDX does not publish.

**Does not measure realised P&L.** `timing_pnl = net_value × forward_return` is
directional timing. A class earning the spread intraday reads zero here while
being profitable. H11 carries the same limitation; both memos state it.

**Fortnightly, and 18 of the most liquid names.** A class that reliably buys
high and sells low *within* ten sessions is invisible, and nothing here speaks
to small caps — though foreign participation there is a rounding error (3.4% in
decile 5 against 49.4% in decile 9), so there is little to measure.

---

## 7. What would have falsified this

A class margin outside its direction null, holding its sign across most years,
surviving the drop-largest-year check, and exceeding 56 bps. The test had the
power for the last of those by a factor of nine or more. Nothing came close on
any of the four.

---

## 8. Where this leaves §12, and the honest recommendation

Three instruments have now been tried against §12's question and all three
returned nothing:

| | instrument | result |
|---|---|---|
| H9 | aggregate broker flow | faint in-sample tilt, sign-unstable, zero spread pre-cost |
| H10/H11 | individual broker codes | no identity signal; margin rank does not persist |
| **H12** | **foreign/domestic investor class** | **no level, no persistence, 33–55× too small** |

§12's premise — that some flow is persistently dumb and identifiable — may well
be true of Indonesian retail. What this repo can now say with evidence is that
**none of the free instruments available resolves it.** The broker code is too
coarse; the domicile split is too coarse in a different direction; and both are
fortnightly.

**The recommendation is to stop buying instruments for this question.** The
remaining moves are an account-type split IDX does not publish, or a licensed
daily feed — neither of which is free, and neither of which is worth paying for
on the strength of three negative results. §4's ordering was by cost-to-falsify
and that ordering has now been run to its end on this branch.

What has *not* been tested at all is the price/TA and structural feature
families in §8, on a spine that Gate 0 passes. That is where the unexplored
cheap ground actually is.
