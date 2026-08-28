# H29 — the Pine screen on the top 50 big caps: works on the ratio, fails on the money

*31,394 pre-holdout name-years; 12,292 inside the point-in-time top 50 by
turnover, 302 distinct names. Code: `scripts/bigcap_backtest.py`. Raw:
`reports/bigcap_backtest.txt`.*

A new universe is a new test. The screen was measured on all 725 eligible IDX
names; "the top 50 big caps" is a different population, and this repo has been
burned by quoting a number measured on one population as if it held on another.

**Pre-registered before scoring:** *the screen will be WORSE on the top 50,
because big caps double far less often (H23 measured the liquid decile at a
4.2% one-year touch rate against a 10.2% base).*

**The prediction was wrong in direction and right in conclusion.**

---

## 1. On the ratio, it works — better here than on the full universe

| cell | n | names | P(2x) | P(halve) | **skew** | median | CAGR/name |
|---|---|---|---|---|---|---|---|
| ALL IDX — no screen | 31,394 | 725 | 12.1% | 9.0% | 1.33 | −3.4% | −5.0% |
| ALL IDX — screen on | 2,593 | 269 | 8.3% | 4.1% | 2.01 | +1.9% | +1.9% |
| TOP 50 — no screen | 12,292 | 302 | 11.4% | 7.4% | 1.54 | +2.0% | −0.5% |
| **TOP 50 — screen on** | 1,390 | 120 | **6.3%** | **2.7%** | **2.35** | +4.5% | +4.2% |

Lift within the top 50 is **1.53×** against 1.51× on the full universe — very
slightly better, not worse.

| | |
|---|---|
| clustered null, 5,000 draws | 1.34 ± 0.20 |
| **z / p** | **+5.01** / **0.00020** vs a bar of 0.00054 — **clears** |
| half-split | 2.85 early / 1.18 late, against a base of 1.93 / 0.86 — **both above** |

P(halve) falls from 7.4% to **2.7%**. As a risk filter on big caps it does
exactly what it does everywhere else.

## 2. On the money, it fails — and this is the part that decides it

Ten names a year, equal weight, rebalanced, costs charged, 400 random draws,
25 years:

| strategy | median × | CAGR | 10th | 90th | beats index |
|---|---|---|---|---|---|
| **screen inside the top 50** | **5.6×** | **+7.1%** | +4.7% | +9.6% | **0.0%** |
| random from the top 50 | 8.4× | +8.9% | +5.1% | +13.5% | 21.8% |
| **IHSG total return** | **15.9×** | **+11.7%** | | | |

**The screen is worse than picking at random from the same 50 names, and it
beats the index in zero of 400 draws.**

The mechanism is visible in the table above: the screen cuts P(2x) from 11.4%
to 6.3%. It buys its ratio by giving up the upside, and in a big-cap universe
where doubling is already rare, giving up doubling is fatal. The ratio improves
and the compounding gets worse. **These are not the same statistic and here
they point in opposite directions.**

Note also that *random* big caps return +8.9% against the index's +11.7% —
A19's finding again, that an equal-weighted basket of IDX names loses to the
cap-weighted index because a handful of mega-caps carry it.

## 3. What I got wrong, and what that changes

I predicted the skew would fall on big caps. It rose. The reasoning — that big
caps double less often, so the numerator has less room — was correct as far as
it went (P(2x) did fall, 11.4% → 6.3%), but I forgot the denominator falls
faster. A ratio has two legs and I only reasoned about one.

The conclusion survives anyway, on a different axis than I expected:
**do not use this screen to pick big caps.** Not because it fails its
significance test — it passes comfortably — but because passing that test is
not the same as making money, which is the lesson H25 recorded and this repeats
with a cleaner separation than any earlier study.

## 4. What this does and does not license about the indicator

**The `[exact]` layer is untouched by this.** Tick size, ARA/ARB levels, max
price step and the round-trip cost floor are arithmetic from published IDX
rules. They are correct on a big cap, a small cap and anything else, and they
are the part of the indicator worth having.

**The `[measured]` conditionals are also untouched** — they are frequencies
attached to a state, not a recommendation to act on it, and they are what they
are on the population they were measured on.

**The optional screen should stay off on a big-cap chart**, and the script now
says so. It ships off by default; this is why.

**Unchanged limits:** in-sample (the holdout was spent at H16), one-year
horizon, per-name rather than portfolio for the first two tables, and the
equity curve is 400 draws over 25 overlapping years, so its dispersion is
understated.
