# Getting live broker flow on IDX

> "My broker only shows broker summary two hours after the close. Any way to get it live?"

Short answer: **yes, but not by finding a live "broker summary" feed.** You get it
by capturing **running trade**, which is already live on your platform, and
aggregating it yourself.

---

## 1. The structural fact that makes this possible

Broker summary is not a separate data product. It is an **aggregation of running
trade**.

Every print on IDX carries a buyer member code and a seller member code:

```
09:41:07   BBCA   6350   150 lot   BK -> YP
09:41:09   BBCA   6350    40 lot   AK -> PD
09:41:11   BBCA   6375   200 lot   BK -> CC
```

Sum those by broker and you have reconstructed the broker summary table. That is
literally all your platform is doing when it shows you "Broker Summary" — it just
does the summing on a batch job that runs after the close.

**Running trade with broker codes is displayed live, during the session, on
essentially every Indonesian platform.** The delay you are hitting applies to the
aggregated *view*, not to the underlying tick stream. So the problem reduces to:
get the prints out of the platform you already have.

That is what `src/idxbot/data/running_trade.py` implements:

```bash
# follow a file your platform (or a userscript) appends to
idxbot live --file ticks.jsonl --follow

# or pipe ticks in
cat ticks.jsonl | idxbot live --stdin

# or aggregate a completed session's prints
idxbot live --file session_2026-08-07.csv --out data/broker_summary/BBCA.csv
```

Expected fields per tick — the parser accepts English or Indonesian headers, and
JSONL or CSV:

| field | meaning |
|---|---|
| `ts` | print timestamp (a bare `09:41:07` works; pass `--date`) |
| `ticker` | bare IDX code |
| `price` | trade price in rupiah |
| `lot` | size in lots |
| `buyer` | buying member code |
| `seller` | selling member code |

---

## 1b. Getting the DAILY broker summary (three routes, easiest first)

Everything above is about intraday. If you just want the end-of-day table:

**Route A — paste it (works today, zero setup).** Select the broker-summary
table in your platform, copy, and run:

```bash
idxbot paste BBCA --date 2026-08-06
```

The parser handles tab/pipe/multi-space/comma separation, Indonesian numbers
(`1.234.567,89`), abbreviated values (`1,2 M`, `450 rb`), and the side-by-side
buyers|sellers layout. It then checks that total buy lots equal total sell lots
— every lot bought is a lot sold, so a mismatch means the columns were misread,
and it says so rather than saving nonsense.

**Route B — fetch it with a real browser (run on your machine).**

```bash
npm install playwright && npx playwright install chromium
node scripts/fetch_broker_summary.mjs BBCA 2026-08-06
```

idx.co.id blocks plain HTTP clients (403) but lets real browsers through after a
Cloudflare JS challenge. Headless Chromium passes that challenge *from an
ordinary connection*. It does not pass from a locked-down build sandbox, which
is why this is a script you run rather than something the engine calls. Use
`--headed` to watch it and fix selectors if IDX redesigns the page.

**Route C — a paid vendor.** GoAPI's `/stock/idx/{SYMBOL}/broker_summary` route
is live and key-gated (it answers 401, not 404). Set `IDXBOT_GOAPI_KEY` and use
`--providers goapi`.

All three land in `data/broker_summary/<TICKER>.csv`, which the CSV provider
reads automatically.

## 2. Where to actually get the stream

Ordered by how much friction each involves.

### A. Your platform's running-trade window (cheapest, most friction)

Most desktop platforms (Mirae HOTS/Neo, IPOT, RTI, BIONS, MOST, Stockbit) show a
live running-trade panel with broker codes. Getting it into a file means either:

- an **export button** if the platform has one (some allow CSV export of running
  trade mid-session), or
- a **browser userscript** that mirrors the running-trade DOM/websocket into a
  local file, for web platforms.

⚠️ **Read your platform's terms of service before automating this.** Scraping a
paid terminal generally violates its ToS even when the data is on your own
screen, and IDX's market-data rules separately restrict *redistribution*.
Personal use of data you are licensed to see is the least problematic case;
sharing or reselling it is not. This is your call to make, not mine — but make it
knowingly.

### B. A commercial data vendor (cleanest, costs money)

| Vendor | Notes |
|---|---|
| **GoAPI.id** | Has a broker-summary endpoint. Probing `https://api.goapi.io/stock/idx/{SYMBOL}/broker_summary` returns `401 invalid API key` rather than 404 — the route is live and gated on a key. Set `IDXBOT_GOAPI_KEY` and use `--providers goapi`. Check with them whether your tier is EOD or intraday. |
| **Sectors.app** | Indonesian market data API (v2; v1 was discontinued 2026-05-11). Coverage of broker-level data varies by plan. |
| **IDX Data Services** | The authoritative source. IDX licenses a real-time datafeed including trade-by-trade with member codes. This is the "correct" answer and it is priced for institutions. |
| **Refinitiv / Bloomberg** | Carry IDX broker-level data via the exchange feed. Institutional pricing. |

`idxbot` ships a GoAPI adapter (`src/idxbot/data/broker_summary.py`) and a
provider interface so adding another vendor is one class.

### C. Your broker's own API

Some Indonesian brokers expose programmatic access to their terminal data. If
yours does, that is the least legally ambiguous live route, because you are an
authenticated customer using a supported interface. Ask your broker's support
whether they offer an API tier — this varies by firm and changes often enough
that any list here would be stale.

---

## 3. The part you may not want to hear: you probably don't need it live

This engine detects **multi-week accumulation campaigns**. Look at what the
campaign segmentation actually finds: a median accumulation leg runs *dozens of
trading days*. A desk quietly absorbing supply over six weeks does not care, and
neither should you, whether you saw today's prints at 15:30 or 17:30.

Concretely, a two-hour delay costs you nothing on this strategy:

| | needs live tick data? |
|---|---|
| Detecting a multi-week accumulation base | **No** — daily granularity is plenty |
| Reconstructing a broker's cost basis | **No** — daily VWAPs are what feed it |
| Knowing a desk flipped to net seller | **No** — that is a multi-day pattern |
| Timing an entry *within* the session | Yes |
| Scalping off a single large print | Yes |

The intended workflow is deliberately end-of-day:

```
17:30-18:00 WIB   broker summary lands, drop it in data/broker_summary/
18:00             idxbot screen --universe lq45
18:05             idxbot plan --tickers <hits> --pool lq45
next session      work the plan's limit orders in the entry band
```

Where live data genuinely helps is **execution**, not detection: you already know
from last night which name you want and at what price, and intraday flow tells
you whether to lean in or wait. For that, the price/volume half of the signal is
enough, and TradingView gives you that live via
`src/idxbot/tradingview/pine/accumulation_score.pine`.

---

## 4. Interpreting intraday broker flow when you do have it

Two traps worth knowing:

**Raw intraday numbers are meaningless without a pace baseline.** Every broker's
volume is "small" at 10:00 because the day is incomplete. What carries
information is whether they are running *ahead of* their normal rate.
`intraday_pace()` computes exactly this — `pace_ratio > 1` means the desk has
absorbed more than its usual share for this point in the session.

**A broker code is not a firm's opinion.** `BK` is every client routing through
J.P. Morgan's membership: a foreign pension fund, a hedge fund, an index tracker
rebalancing, a delta hedge against a warrant. "J.P. Morgan bought 2 million lots"
means flow crossed their membership. It is a strong signal of institutional
intent. It is not a confession of one.

---

## 5. What is blocked, and why the fallback exists

From this build environment:

| Source | Result |
|---|---|
| Yahoo Finance chart API | ✅ works — full daily history, IHSG back to 1990-04-06 |
| `idx.co.id` broker summary | ❌ 403, Cloudflare WAF |
| Stockbit API | ❌ requires an authenticated app session |
| GoAPI broker summary | 🔑 route exists, needs a paid key |

Because no free public broker-summary API exists, the repo ships a **simulator**
(`src/idxbot/data/synthetic.py`) so the pipeline runs end to end today. It is
clearly labelled everywhere it is used, and every report prints a warning banner,
because the simulator *assumes* institutions buy weakness — so analysing its
output and concluding "institutions buy weakness" is circular and proves nothing.

Use it to see the machinery work. Connect a real source before you trade.
