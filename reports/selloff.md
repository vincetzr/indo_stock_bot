# Selling into the sell-off — H47, and the board-wide Hull number I owed — H48

*2026-08-29. `scripts/selloff.py`, `scripts/hull_colour.py`.
Pre-registration S1–S4 in the `selloff.py` docstring, written before any cell
was scored. 233 liquid names (median turnover ≥ Rp1bn/day), 2000–2026, net of
0.56% fees plus each name's own fraksi-harga half-spread. The 24-month holdout
was spent at H16; everything here is in-sample.*

---

## 0. The question, and why it splits in two

> *"The Hull suite is great but sometimes it sells a bit late so the profit is
> not maximum, but usually the entry is great. I was thinking if I can detect a
> sell-off then I can sell mostly at the top."*

Both halves of the observation are correct and both are already measured.

**The entry does carry information.** H39 added a benchmark column as a control
and it turned out to contain the finding: spans selected by the EMA-stack entry
returned **+9.9% to +11.2% a year if you merely owned the name**, against a
panel median of **+2.56%**. That is the one unambiguously positive result about
this indicator anywhere in the project.

**The exit does give a lot back.** H37 measured the give-back — the share of the
in-trade peak already surrendered when the flip fires — at **11.8%** for the
Hull-55 slope, 10.9% for EMA34, 12.5% for the EMA stack.

So the complaint is right. What is not available is the fix as stated, and the
reason is structural: **any rule that fires on a decline is downstream of the
decline.** Selling at the top requires firing *before* it, which is prediction —
H31 tested that directly and found turning-point spacing is *more* dispersed
than a block bootstrap of the same returns (CV **2.246** against **1.340**). A
cycle would make spacing more regular; the data makes it less.

So the reachable question is not "can I sell at the top" but "for a given amount
of give-back saved, what does it cost". That is a frontier, and this study maps
it with the control that decides it.

---

## 1. The design

**Entry, held fixed at the thing being complained about:** HMA(55) rising *and*
price > EMA50 > EMA100 > EMA200. **Exit:** eleven rules, nine of them genuine
sell-off detectors and two plain trailing stops as the slow reference. Cap of
252 sessions on every trade. Cost charged at every exit.

**The control is the whole design.** For each rule, a **random exit drawn from
that rule's own holding-period distribution** — same entries, same trading rate,
same number of round trips, no information about the sell-off. S3 pre-registered
that the random exit would cut give-back by about as much as a real detector,
because *give-back is mostly a function of when you leave, not why*.

### The control was rigged, and fixing it changed a number I would have quoted

A first version required a **fresh rising edge** of the setup to re-enter. That
looks innocuous and is not. A real detector fires on a drop, and the drop
usually breaks the entry condition too — so the real arm gets a fresh edge and
rejoins the next leg. The random arm sells *mid-trend* with the setup still
live, and is then locked out until the whole trend dies and restarts. It was
skipping the remainder of every trend it sold into.

Both policies are now run and printed. `live` (re-enter whenever the setup is
live and the exit signal has cleared) is the fair experiment; `edge` is the more
natural trading rule. Under the fair policy the random control's CAGR rises by
**1 to 3 points** in every cell — e.g. `trail 25%` control **+3.76% → +4.60%**,
`3day −8%` **+2.01% → +3.72%**. The give-back conclusion is unchanged under
both, which is why it is the conclusion.

---

## 2. The table

Fair (`live`) re-entry policy. GIVE-BACK is the median share of the in-trade
peak handed back at the exit. CAGR and HOLD are both compounded per name over
the name's **full span**, so the rule is charged for the time it sits in cash.

| exit rule | trades | win | **give-back** | bars | in mkt | CAGR | HOLD | random of same speed |
|---|---|---|---|---|---|---|---|---|
| close < EMA20 | 11,533 | 24.3% | **4.7%** | 11 | 20% | +4.71% | +13.06% | gb **1.9%**, cagr −0.37% |
| hull55 +1 bar | 6,734 | 36.2% | **6.9%** | 23 | 23% | +5.01% | +13.06% | gb **4.1%**, cagr +1.51% |
| drop −4% | 9,064 | 38.2% | **7.3%** | 24 | 33% | +6.44% | +13.06% | gb **2.8%**, cagr +1.89% |
| drop −6% | 4,683 | 42.4% | 11.6% | 61 | 43% | +8.22% | +13.06% | gb 5.1%, cagr +2.61% |
| 3-day −8% | 4,777 | 39.9% | 12.5% | 53 | 38% | +8.37% | +13.06% | gb 4.8%, cagr +3.72% |
| volume climax | 2,747 | 46.9% | 12.7% | 123 | 51% | +4.78% | +13.06% | gb 10.7%, cagr +5.61% |
| 2× ATR down | 2,434 | 47.5% | 15.3% | 149 | 54% | +5.87% | +13.06% | gb 12.4%, cagr +4.27% |
| drop −8% | 2,833 | 46.0% | 15.8% | 122 | 51% | +7.94% | +13.06% | gb 8.8%, cagr +4.18% |
| trail 15% | 3,810 | 40.1% | 16.3% | 63 | 36% | +7.45% | +13.06% | gb 6.6%, cagr +4.48% |
| 3-day −12% | 2,781 | 44.7% | 18.5% | 118 | 49% | +7.60% | +13.06% | gb 9.3%, cagr +4.31% |
| trail 25% | 2,340 | 43.1% | 25.8% | 126 | 44% | +7.88% | +13.06% | gb 11.4%, cagr +4.60% |

---

## 3. What was registered, and what happened

**S1 CONFIRMED.** Fast detectors do cut the give-back, and by a lot: from
**25.8%** for a 25% trailing stop down to **4.7%** for an EMA20 break. The
complaint is fixable in its own terms. This was near-mechanical and is not the
finding.

**S2 FAILED.** CAGR is *not* monotone in give-back. The best cells sit in the
**middle** of the table (`3day −8%` at +8.37%, `drop −6%` at +8.22%), not at
either end. Cutting give-back to 4.7% costs 3.7 points of CAGR; cutting it to
12.5% costs nothing relative to the slow rules. There is a genuine interior
optimum, which H40's "tighter is worse on both axes" shape did not predict.

**S3 FIRED, AND IT IS THE ANSWER TO THE QUESTION.**

> **Every single real detector gives back MORE of the peak than a random exit
> spending the same number of bars.** Eleven of eleven, under both re-entry
> policies — twenty-two cells, no exceptions.

`hull55 +1bar` gives back 6.9% where a coin flip of the same speed gives back
**4.1%**. `drop −4%` gives back 7.3% against **2.8%**. `trail 25%` gives back
25.8% against **11.4%**.

The mechanism is not subtle once stated: **a detector fires *because* price
fell, so it is conditioned on the drop having already happened.** A random bar
is not. Leaving sooner is what cuts the give-back, and detection is not what
makes you leave sooner — it is a reason to leave that arrives strictly after the
thing you wanted to avoid. This replicates H37's finding on a completely
different construction: there the detectors were compared to random bars in a
peak-detection framing, here to random exits in a portfolio framing, and the
sign is the same both times.

**S4 CONFIRMED.** No exit rule beats simply holding the name. Best is
`3day −8%` at **+8.37%** against **+13.06%** for owning it. That is now **169** exit
configurations with none beating a hold — H17's 32 rules, H18's 58, H35 and
H38's 55 bracket and placement combinations, H40's 13 stops and H47's 11.

**One thing the detectors DO earn.** Under the fair policy they still beat their
own random control on return in all eleven cells (`hull55 +1bar` +5.01% against
+1.51%; `close < EMA20` +4.71% against −0.37%). So the detectors are not
information-free — they leave at better *times* than chance. They simply do not
leave nearer the *top*, which is what was asked for.

---

## 4. H48 — the board-wide Hull colour number I owed

Separately: I told you the Hull suite loses to buy-and-hold, you asked *"really?
are you high?"*, and it turned out I had measured **EMA34 breaks** and labelled
them the ribbon. On BBCA the actual HMA(55) colour returns **−17.9%** against a
hold of **−29.0%** — it beat holding on the chart being eyeballed. I said I
would re-run the pure colour rule across every liquid name over full history and
report it whichever way it came out. `scripts/hull_colour.py`, 891 names,
2,850,959 name-bars, 2000-03 → 2026-08.

**In when HMA(55) is rising, out when it is not. Nothing else.**

| universe | names | trades | in mkt | cost | RULE | HOLD | edge | names where rule wins |
|---|---|---|---|---|---|---|---|---|
| every name | 891 | 71.2 | 44% | 1.44% | **−1.32%** | +6.42% | **−7.74%** | 31.3% |
| liquid ≥ Rp1bn/day | 233 | 64.0 | 47% | 1.23% | **+7.15%** | +12.62% | **−5.47%** | 36.9% |
| middle Rp0.1–1bn | 239 | 54.2 | 45% | 1.54% | +4.16% | +3.19% | +0.96% | 45.2% |
| thin < Rp0.1bn/day | 419 | 84.9 | 43% | 1.49% | −9.15% | +4.80% | −13.96% | 20.3% |

Half-split at 2013-06-30 — the only replication test this repo trusts:

| universe | rule early | hold early | edge | rule late | hold late | edge | both? |
|---|---|---|---|---|---|---|---|
| every name | −2.70% | +15.71% | −18.41% | −1.61% | +4.79% | −6.40% | **no** |
| liquid | +12.95% | +21.54% | −8.59% | +5.05% | +9.84% | −4.79% | **no** |
| middle | +6.51% | +26.05% | −19.54% | +3.83% | +1.70% | +2.13% | **no** |
| thin | −10.92% | +10.82% | −21.74% | −8.41% | +3.74% | −12.15% | **no** |

**So the claim survives the board test, but I reached it by the wrong route and
that matters.** The ribbon loses to holding on the board by **7.7 points a
year**, on the liquid names by **5.5**, negative in both halves of every
universe. BBCA is one of the **36.9%** of liquid names where it wins — a real
minority, not a fluke, and precisely why one chart could not settle it in either
direction.

The mechanism is the same wall as everywhere else: the rule is in the market
**44%** of the time and pays **~1.4%** per round trip across **71 round trips**
per name. In the top turnover decile the names returned **+20.28%** a year and
the rule captured **+8.10%** of it. The middle bucket's +0.96% full-sample edge
is the only positive cell in the table and it fails the half-split (−19.54%
early, +2.13% late), so it is not evidence.

---

## 5. What I would actually do with this

The Hull with **one-bar confirmation** already sits near the efficient corner of
the give-back frontier: **6.9%** give-back, second-lowest in the table, at
+5.01% CAGR — the same CAGR band as rules giving back twice as much. Confirming
one bar after the flip rather than at it is worth having for the reason H40
found: the bar after a Hull flip is on average an *up* bar.

**Do not add a sell-off detector on top of it.** Every one tested makes the
give-back worse than leaving at random at the same speed, and none of them beats
holding the name. If the goal is genuinely to hand back less of the peak, the
thing that achieves it is **leaving sooner**, full stop — and the price is 3 to
4 points of annual return, paid whether or not a detector is involved.

**And the whole family is dominated by the entry.** H39's benchmark column,
H48's decile 10, and this table all say the same thing from different angles:
the signal carries real information about **what to own**, and flipping in and
out of what it selects destroys most of that value. The 2026-08-29 position is
unchanged — hold what the entry selects, do not trade it.

---

## 6. Trials and standing caveats

S1–S4 plus H48's single confirmatory test take the count to **291**. Bonferroni
bar α = 0.05/291 = **0.00017**. Nothing here is offered as clearing it: S3 and
S4 are *negative* results, which is what they are reported as, and no positive
claim is made.

The holdout was spent at H16, so every number above is in-sample. The board run
is split by liquidity decile because H44's headline was withdrawn when that
check — CLAUDE.md §7, which had never been run — showed the whole effect was
non-synchronous pricing in thin, stale names. Costs are fees plus a
fraksi-harga half-spread and contain **no impact, suspension or auto-rejection
term**; A23 measured all three biting on thin names, and all three run against
the holder.

---

## 7. H49 — where the 31.3% actually comes from, and the control H48 was missing

*Added 2026-08-29 after the number was challenged: "I refuse to believe it only
wins 30-something percent. By the central limit theorem it should be 50%, since
the price can only go up or down." `scripts/winrate.py`, pre-registration W1–W4
in its docstring.*

**Two different statistics were both called "win" and that is partly my fault.**
The **per-trade** win rate is the share of round trips that ended positive. The
**per-name** win rate is the share of tickers where the rule's CAGR beat owning
the ticker over the same span. Only the first has any reason to be 50%; the
second compares a rule that is in the market 44% of the time and pays ~71 tolls
against one that is in 100% of the time and pays one.

### W1 — the harness on a case with a known answer

A26 established that a detector which cannot find a cycle in a pure sine wave
proves nothing by finding none. Same discipline: before believing 31.3% about
IDX, the machinery has to return **50%** on a driftless log random walk with no
cost, where 50% is arithmetically correct.

| synthetic market | per-name win | per-trade win | mean trade | median trade |
|---|---|---|---|---|
| driftless, no cost — **the coin flip** | **47.7%** | **50.2%** | +1.19% | +0.04% |
| driftless, 1.44% toll | **28.3%** | 44.3% | −0.25% | −1.40% |
| IDX's own drift and vol, no cost | **50.3%** | 50.9% | +2.26% | +0.28% |
| IDX's own drift and vol, 1.44% toll | **27.3%** | 46.3% | +0.82% | −1.16% |

**W1 CONFIRMED** — 47.7% and 50.2% against a standard error of 2.9%. The coin
flip is exactly where it should be, so the harness can be believed about the
rest.

### W2 FAILED, and W3 is the whole explanation

I registered **drift** as the largest term. It is not: adding IDX's own measured
drift (+3.39%/yr median log) with no cost moves the per-name rate from 47.7% to
**50.3%** — nothing. At 44% exposure the forgone drift over a span is small.

**The toll is the entire effect.** The same driftless market with nothing added
but 1.44% a round trip falls from 47.7% to **28.3%** — nineteen points, landing
right beside the observed 31.3%. The arithmetic is blunt: **71 round trips ×
1.44% = 102% of the position's value paid in fees and spread over the span.**
The rule pays for its own position, twice over, in tolls.

### W4's predicted null FAILED — in the direction that favours the ribbon

H48 compared rule against hold with **nothing in between**, which the challenge
correctly exposed. The null here reorders the ribbon's own green and red runs:
exposure, trade count and run-length distribution preserved exactly, only the
alignment with price destroyed.

| universe | names | expo | **beats hold** | run-shuffled null | trade win | null |
|---|---|---|---|---|---|---|
| every name | 891 | 44% | **31.3%** | **21.7%** | 26.9% | 35.4% |
| liquid ≥ Rp1bn/day | 233 | 47% | **36.9%** | **20.3%** | 31.7% | 41.3% |
| middle | 239 | 45% | 45.2% | 26.1% | 29.0% | 36.2% |
| thin | 419 | 43% | 20.3% | 20.0% | 23.2% | 31.7% |

**Real minus null: +9.6 points, se 0.8% — about twelve standard errors. On
liquid names, +16.6 points.**

I predicted the two would be about equal. They are not, and the failure is
informative: **the ribbon's timing carries real information.** Applied at random
with the identical trading rate and exposure, the same rule beats holding on
21.7% of names; aligned to the actual Hull, 31.3%. It still loses to holding —
but it loses by far less than chance does, and on the liquid names it is nearly
twice as good as random timing.

This **qualifies H48 without overturning it.** "The ribbon loses to buy-and-hold"
stands, on the board and in both halves. What was missing is why: not because
the indicator is noise — it demonstrably is not — but because the information it
carries is smaller than 102% of capital in tolls. Note also that the thin bucket
is the one place where real and null coincide (20.3% against 20.0%), which is
consistent with H44's withdrawal: there is no timing information to have in
names that barely trade.

### And the per-trade win rate below 50% is the rule working, not failing

The matched-speed random exits from §2 win **40–51%** of the time — the coin
flip again — and make **less** per trade. The real detectors win less and make
more:

| exit | real win | real mean | random win | random mean |
|---|---|---|---|---|
| close < EMA20 | **24.3%** | **+2.05%** | 40.2% | +0.72% |
| hull55 +1 bar | **36.2%** | **+4.27%** | 46.2% | +2.29% |
| trail 15% | **40.1%** | **+10.18%** | 49.4% | +8.12% |

A trend rule cuts losers early and lets winners run, so it converts what would
have been recoveries into small realised losses and pays for that with a fat
right tail. Board-wide the ribbon's mean trade is **+3.54%** while its median is
**−4.16%**. H40 measured the trade-off directly from the other side: adding a
take-profit lifts the win rate to 46.6% and takes the CAGR to **−1.98%**.
**Win rate and profitability are close to orthogonal, and a high win rate is
bought with the right tail.**

**Trials after H49: 295.** W1 is an instrument check, not a market claim; W4's
+9.6 points is the one positive result here and it is reported as qualifying a
negative, not as a strategy.
