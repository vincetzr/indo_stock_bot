# H26 — search on ASYMMETRY, and it retracts H25

*31,394 name-years, 725 names, pre-holdout. Code: `scripts/asymmetry.py`.
Raw: `reports/asymmetry.txt`.*

Asked for "profit far more likely than loss, and still a multi-bagger". That is
a request for a **ratio**, and nothing in this project had ever ranked cells by
one. H25 maximised P(touch 2x) alone and found a volatility sort, because
variance raises **both** tails at once.

**The objective, fixed before any cell was scored:**

```
skew = P(touch 2x) / P(end <= 0.5)
```

with a 300-observation floor and a requirement that both legs be estimable — a
cell with three halvings has an undefined ratio, not an infinite one.

---

## 1. The frontier — you cannot maximise both

| cell | P(2x) | P(−50%) | **skew** | doubles/10 | CAGR per name |
|---|---|---|---|---|---|
| everything (no screen) | 12.1% | 9.0% | 1.33 | 1.2 | −6.3% |
| **STRENGTH + CALM** | 10.5% | **4.1%** | **2.60** | 1.1 | **+5.2%** |
| strength only (hi52 top) | 13.6% | 6.3% | 2.15 | 1.4 | +1.1% |
| strength, very strong | 14.6% | 7.5% | 1.95 | 1.5 | +0.1% |
| strength + some vol | 21.4% | 13.5% | 1.59 | 2.1 | −8.1% |
| **H25 volatility screen** | 21.2% | 18.7% | **1.13** | 2.1 | **−16.1%** |

**Monotone within the strength family** — the four cells that share a
construction. Every step up in doubling rate costs asymmetry *and* compounding,
and there is no cell with both. That is the honest answer to "is that so much
to ask".

*A first draft called the whole table "perfectly monotone" and a test caught
it.* Two rows do not belong on the curve: the unscreened baseline is a
reference point rather than a point on it, and H25's screen has no strength
filter at all. **The comparison those two rows obscure is the useful one:**
H25's screen and `strength + some vol` double at the *same* rate — 21.2% and
21.4% — and adding the strength filter takes the ratio from 1.13 to 1.59, the
halving rate from 18.7% to 13.5%, and per-name compounding from −16.1% to
−8.1%. **At an equal doubling rate, strength is free.**

## 2. The winner: strength AND calm

Within ~2% of the 52-week high **and** below-median 60-day volatility.
n = 2,022.

| | |
|---|---|
| asymmetry (skew) | **2.60** vs null 1.20 ± 0.15 |
| **z** | **+9.44**, p = **0.00033** vs a bar of 0.00061 → **CLEARS** |
| half-split | 2.65 early, 2.53 late — both far above base |
| P(a name doubles in a year) | 10.5% |
| **P(a name halves)** | **4.1%** |

**10-name basket, 24 years, annually rebalanced:** median **58.3×** =
**+18.5% CAGR** against the index's 25.1× = +14.4%. **Beats the index in 90.2%
of draws**, 10th percentile +14.4%, 90th +22.7%. **Even the bad draws match the
index.**

This is not a data-mined curiosity: it is momentum plus low-volatility, two of
the most replicated factors in the global literature, and the prior existed
before the cell was scored.

## 3. It retracts H25

H25's volatility screen, on the asymmetry objective: **skew 1.13 against a null
of 1.18 — z = −0.19, p = 0.54.** It is **indistinguishable from a random cell**
on the thing that matters. Its 10-name basket returns 3.4× against the index's
22.8×, beating it in **5.0%** of draws.

H25 cleared the Bonferroni bar on P(2x) and that told us nothing, exactly as
its own §2 warned. **Optimising the upside alone finds variance. Optimising the
ratio finds something that survives.**

## 4. Q2 pre-registered and FAILED — buy strength, not weakness

The classic retail intuition is that a name already down 70% has less room to
fall. Predicted to fail, and it does, monotonically:

| distance from 52w high | P(2x) | P(−50%) | skew |
|---|---|---|---|
| nearest the high (top 10%) | 13.6% | 6.3% | **2.15** |
| middle | 11.2% | 7.4% | 1.53 |
| **furthest below (bottom 10%)** | 15.1% | **18.9%** | **0.80** |

**Fallen names halve more often than they double.** The intuition is exactly
backwards in IDX, and A17's recovery curve said so from the other direction.

## 5. Q1 pre-registered and SUPPORTED — asymmetry is a function of horizon

Base skew by holding period: **1.33 (1y), 1.48 (2y), 1.57 (3y), 1.80 (5y),
2.26 (10y)** — monotone. Time converts diffusion into drift. The screen buys
about six years' worth of that at a one-year horizon.

---

## What this licenses, and what it does not

**Licensed:** asymmetry of roughly **2.6 : 1** is real, measured, clears the
bar, replicates in both halves, and comes with a basket that beat the index in
90% of draws over 24 years. The mechanism is standard and was not invented after
the fact.

**Not licensed:** "multi-bagger fast". This screen doubles **1.1 names in ten
per year**, not 2.1. Buying the higher doubling rate costs the asymmetry, the
compounding, and — on the H25 version — everything.

**Unchanged limits.** The holdout is spent (H16), so this is in-sample. This is
the best of ~30 cells, so the trial count now stands at 82. And §6–§7 of
`reports/fastmover.md` still apply to any thin name: no impact model, no
suspension term, no auto-rejection term. These names are far more liquid than
H25's — Rp2–69bn a day against Rp1.77bn — which materially reduces that
exposure but does not remove it.
