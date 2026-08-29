# H44 — the simple methods on a cost ladder, and why the headline was an artefact

*36,508 (period, name) rows, 774 names, monthly rebalances 2000–2026. Code:
`scripts/cost_ladder.py`. Raw: `reports/cost_ladder.txt`,
`reports/cost_ladder_null.txt`, `reports/cost_ladder_null_xs.txt`.*

The question was whether IDX rewards "easier methods" — support/resistance,
stochastic, EMA, Fibonacci — and whether the answer changes at institutional
cost. **The first run said yes, spectacularly. It was wrong, and the way it was
wrong is the finding.**

---

## 1. What the first run said

Long-only decile portfolios, monthly, against a 200-draw clustered permutation
null: `ema_cross` +26.5% at 56bp with z = +6.58, `stoch_strong` +21.1% with
z = +5.08, `sr_break` +21.0% with z = +4.66. `fib_618` z = −0.79 and the
predicted-null `rand` z = −1.41, both behaving exactly as registered.

Three of the user's four methods clearing a clustered null at z > 4, surviving
the retail toll, with the predicted null flat. **It would have been the first
positive result in the project.**

---

## 2. Four defects, in ascending order of how much they mattered

**The tie-break (A15).** `ema_stack` is a 0–3 ordinal — **96.4% of its values
are tied within a date** — and `sort_values` resolves ties by the frame's
existing order, which is alphabetical by ticker. Re-run with 25 random
tie-breaks its answer spans **+16.2% to +36.2%, sd 3.98%**. Its number was one
arbitrary draw. `sr_break` is mildly affected (8.5% ties, sd 1.24%).
`ema_cross` and `stoch_strong` are tie-free and bit-identical under reshuffle,
so this was not the main cause — but `ema_stack` is withdrawn outright.

**The omitted half-spread (A23, committed for the second time).** The docstring
promised "56 bps plus the spread" and `walk` charged fees only. Measured on the
harness's own picks the fraksi-harga round trip is **44–54 bps** — almost as
much again as the entire commission — so the ladder's top rung sat below the
real retail floor. Adding it took `stoch_strong` from +21.1% to +15.0%.

**The annualisation.** 252/21 = 12 periods a year, but only 238 of 303 marks
survive the 40-name gate, so the true rate is **10.99/yr** — worth another
~3 points, and applied inconsistently because different rules were scored on
different period counts.

**`sr_break` was never measuring breakouts.** Its score is `log(price /
nearest confirmed high above)`, which is capped at exactly **0.0000 across all
36,508 rows** — a name with nothing confirmed overhead is at new highs. So the
"breakout" rule is a 52-week-high proximity flag with alphabetical tie-breaking.

---

## 3. THE DEFECT THAT ENDS IT: the effect lives where you cannot trade

The mandatory IC-by-liquidity check (CLAUDE.md §7) was never run. Run:

| | rank IC of stoch_k vs forward return |
|---|---|
| bottom turnover tercile | **+0.072** |
| top turnover tercile | **−0.0021 (t = −0.15)** |

And it rises monotonically with **price staleness** — +0.035 for names that
move nearly every day, **+0.099 for names flat more than 30% of the trailing
month.** That is **non-synchronous trading**, not forecasting: a stale price
mechanically "predicts" its own catch-up.

Restricting to a universe a professional could actually build a book in —
close ≥ Rp500 **and** 60-day median turnover ≥ Rp10bn:

| rule | as shipped | **tradeable universe** |
|---|---|---|
| stoch_strong | +21.7% gross / +15.0% net | **−1.7% / −7.4%** |
| ema_cross | +26.1% / +21.2% | **+1.5% / −2.8%** |
| sr_break | +21.4% / +16.5% | +7.8% / +3.5% |
| mom12_1 | +17.9% / +15.6% | **+10.4% / +8.1%** |
| `rand` | −1.1% / −7.0% | −4.0% / −9.7% |
| *equal-weighted pool* | *+11.5%* | *+4.1%* |

**Stochastic and EMA finish BELOW the pool they are drawn from.** Against a
40-draw random-book null the tradeable `stoch_strong` reads **z = −0.17**.

**And the framing was spurious anyway.** In the same harness a plain 21-session
trailing return scores +34.3% and a close-only 14-day range position +37.1%.
There is one factor and it is *recent price change on thin names* — A22's
finding, reached from a new direction.

---

## 4. The H13 contradiction, resolved

H13 measured this family as net-negative; this measured it hugely positive. The
two were never measuring the same thing: **H13 quoted a control-NEUTRALISED
long-short QUINTILE spread on the whole 891-name panel; this quotes a RAW
long-only DECILE-minus-mean on the liquid third.** Once the liquidity check is
run, both agree — and H13's own note that "restricting to the most liquid 5%
collapses the signal's t fivefold" was the answer all along.

---

## 5. The answer to "which method is best and most reliable"

**Of the four asked about: none, at a tradeable liquidity threshold.**

| method | verdict |
|---|---|
| **Fibonacci** | nothing, twice. z = −0.79 here, z ≤ 0.95 on 280,228 touches in H34a |
| **Stochastic (oversold)** | **actively harmful** — z = −5.17, −11.1%/yr net |
| **Stochastic (strength)** | collapses to −1.7% gross on tradeable names, z = −0.17 |
| **EMA** | collapses to +1.5% gross; `ema_stack` withdrawn on ties |
| **support/resistance** | weakly survives (+7.8% gross vs a +4.1% pool) and is not a breakout signal |
| *12-1 momentum (the academic control)* | *the only survivor: +10.4% gross, +8.1% net, lowest turnover at 30%* |

The one thing that survives is the textbook factor, not a chart method — and it
still loses to the index over the same window.

---

## 6. What this study is worth

The negative result is solid and the process is the point. **Five defects, four
of them mine, and the two that mattered most were caught by controls this repo
already mandates and I skipped:** the IC-by-liquidity-decile split (§7) and the
fraksi-harga spread (A23). The permutation null did NOT catch either — it
happily returned z = +6.58 on an artefact, because a null that shuffles labels
still holds the same illiquid names.

**A permutation null tests whether a ranking beats a random ranking. It cannot
tell you the ranking is picking names nobody can trade.** That is a new lesson
for this repo and it is the most important sentence here.
