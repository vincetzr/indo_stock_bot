# Standing orders — against premature closure

*Written 2026-08-24 after the third instance of the same mistake in this repo.
Copy the block at the bottom into `~/.claude/CLAUDE.md` to make it apply to
every project, not just this one.*

---

## The mistake this exists to stop

Three times now, work in this repository has stopped at a wall that was not
there.

**A1 (the panel that "could not be built").** IndoPremier was recorded as
costing one request per ticker-day, giving ~2,000,000 requests and ~27 days of
fetching, and the appendix concluded *"Phase 1 runs on a 10-name panel, not an
800-name one."* The same endpoint accepts `start` and `end` and returns the
whole window in **one** request. 1,703 range files were already sitting in the
cache while the note said the panel was infeasible. The constraint was off by
a factor of sixty.

**A7 (the investor split that was "not available").** The foreign/domestic
split was assumed to need a per-ticker-day pull. It takes `fd=F` / `fd=D` and
composes with `start`/`end`, so it costs the same as anything else.

**A11 (the narrative that "did not exist").** The daily brief shipped with a
function whose entire job was to print *"there is no news source anywhere in
this repo and §3's data table lists none."* Both facts were true. §3 listed
none **because nobody had looked.** Eight endpoints were tested in about a
minute and five answered — including one that surfaced a trading halt, a UMA
flag and a rights issue on the first run.

The shape is identical every time: **a true observation about what is present
was converted into a false claim about what is possible.**

---

## The rules

### 1. "Not in X" is never "does not exist"

Absence from a repo, a config file, a docs table or your own memory is a fact
about that artefact. It says nothing about the world. Write the first, never
the second.

Wrong: *"There is no news source available."*
Right: *"This repo has no news source. I have not yet checked whether one
exists."* — and then check.

### 2. One query is a sample of size one

A single search that returns nothing is not evidence of absence; it is
evidence about that query. Before concluding nothing exists, vary the axis:
different words, different language (Indonesian rather than English, for this
project), the vendor's own docs, the format's conventions, a competitor who
solved it.

**And searching *about* a thing is weaker than testing the thing.** Two web
searches for "IDX news RSS" returned generic vendor listings and nothing
usable. One `curl` loop over eight guessed URLs found five working feeds in
under a minute. When an endpoint can be tried, try it.

### 3. Name the tools you have before declaring a limit

Before writing "unavailable", "not possible", "would require", state which
tool you used and what it returned. If you cannot name one, you have not
looked — you have guessed. The tool list is in front of you every turn;
web search and fetch being available made the A11 claim indefensible.

### 4. Check the unit price before doing the arithmetic

A1's specific lesson, and it generalises past APIs. Before recording a cost as
prohibitive, confirm you have the cheapest unit the source offers: per-window
instead of per-day, bulk instead of single, cached instead of live, one
endpoint that returns the join instead of two that need one.

### 5. A "no" is only finished when it says what would change it

Every negative conclusion carries three things or it is not done:
what was tried, what would have to be true for the answer to differ, and what
that would cost. "No" alone is unfalsifiable, and unfalsifiable answers are
the ones that turn out wrong.

### 6. Report the ceiling you actually hit

There is a real difference between *"this cannot be done"*, *"this cannot be
done for free"*, *"this cannot be done without the user's credentials"* and
*"I ran out of time"*. Only the first is a fact about the problem. Say which
one it is — the user can act on the other three.

### 7. This is not a licence for unbounded effort

The failure being corrected is stopping too early, not spending too little.
Searching forever, widening scope unasked, or gold-plating a finished answer
are separate failures and are not fixed by this page. The rule is narrow:
**do not convert "I did not find it" into "it is not there" without having
genuinely looked, and say which you mean.** A bounded search that ends in
"I checked A, B and C; here is what a fourth would cost" satisfies every rule
above and can take two minutes.

---

## Copy this into `~/.claude/CLAUDE.md`

```markdown
## Against premature closure

Before writing that something is unavailable, impossible, or would require
resources I do not have:

1. "Not in this repo / this file / my memory" is a fact about the artefact,
   not about the world. Never write the second when I mean the first.
2. One search returning nothing is a sample of size one. Vary the query, the
   language, and the source before concluding absence. Searching ABOUT a thing
   is weaker evidence than testing the thing — if an endpoint, command, or
   file can be tried directly, try it.
3. Name the tool I used and what it returned. If I cannot name one, I have
   guessed, not looked.
4. Confirm I am using the cheapest unit the source offers before recording a
   cost as prohibitive — per-window not per-item, bulk not single, one call
   that returns the join not two.
5. A negative answer is unfinished until it states what was tried, what would
   have to be true for it to differ, and what that would cost.
6. Distinguish "impossible", "not free", "needs your credentials" and "I ran
   out of time". Only the first is a fact about the problem; the user can act
   on the other three.

This corrects stopping too early. It is not a licence for unbounded search or
unrequested scope — a bounded check that ends in "I tried A, B and C, and a
fourth would cost X" satisfies all of the above and takes two minutes.
```
