# Getting the FULL broker summary — the answer, and the line I won't cross

You were right that there is a way. Here it is, along with why one version of it
is fine and another is not.

---

## 1. IDX publishes the full broker summary themselves, free

**`https://www.idx.co.id/en/market-data/trading-summary/broker-summary`**

Official, authoritative, complete — every member, not a top ten — and it is the
source every other site is ultimately repeating. It costs nothing.

## 2. It is not my network that is blocked. It is automation

This container gets **403 on every idx.co.id path**, including the homepage.
Earlier in this project that was read as a geographic or network block. It is
not. The open-source scraper that does reach it (`nichsedge/idx-bei`) states
plainly what it uses:

> `curl_cffi` — *"browser impersonation, Cloudflare bypass"*

So idx.co.id sits behind **Cloudflare**, and the 403 is a bot check on the TLS
fingerprint. Defeating it means pretending to be a browser that isn't there.

**You showed me earlier that idx.co.id loads fine for you.** That is the whole
point: the site is not blocking *you*, it is blocking *scripts*.

## 3. Why I am not going to bypass it

Two reasons, and the first is enough:

- **IDX's terms prohibit scraping their market data.** Deliberately defeating a
  bot check on a source that forbids automated collection is not a grey area,
  and it is not a decision I should make on your behalf with your name and your
  client money attached.
- A banned IP or a complaint to a member firm is a real operational risk on a
  project whose entire premise is not losing other people's money.

The distinction that matters: **a person reading a public page in a browser is
the intended use. A script defeating a bot check is the thing being forbidden.**
Those are different acts even though the bytes are identical.

## 4. So: you download, I ingest

This is not a workaround, it is the sanctioned route, and the pipeline for it
already exists.

**One file a day covers the whole market.** Unlike the top-ten source — one
request per stock per day — IDX's broker summary is published per session. If it
downloads as one file covering every stock, then **~4,500 files is the entire
history of the exchange**, at full depth, with no censoring at all.

What that changes, concretely:

| today, from the free top-ten source | with the full rekap |
|---|---|
| 85–90% of volume visible | **100%** |
| positions are *brackets* | positions are **exact** |
| cost basis unavailable | cost basis **exact** |
| 63% of a year's flow has a proven direction | **all of it** |
| `idxbot.broker_bounds` is required | it becomes **unnecessary** |

### How to hand it over

```bash
# drop whatever you downloaded here
cp ~/Downloads/*.csv  data/inbox/
python3 scripts/broker_collect.py --ingest data/inbox
```

`scripts/import_broker_data.py` already reads CSV, Excel and saved HTML, and
already distinguishes a **running-trade** export (buyer and seller code on every
print) from a **broker-summary** table, flagging which it found.

**Send me one file first.** I do not know IDX's exact column layout and I am not
going to guess it — the last time you sent a real sample it exposed three parser
bugs in one go (dropped magnitude suffixes, two brokers merged into one row, and
the column order being Val/Lot/Avg rather than Lot/Val/Avg). One file and I will
have the importer matching it exactly, then you can bulk-drop the rest.

## 5. The other sites — reachable, but not mine to enable

From this container these respond:

| site | reachable | note |
|---|---|---|
| sahamgain.com | yes | |
| new.sahamidx.com | yes | |
| pasardana.id | yes | |
| sabarkaya.com | yes | the bandarmology site you mentioned |
| britama.com | no | proxy error |
| **idx.co.id** | **no** | Cloudflare |

I have **not** wired a fetcher to any of them, and `data.broker_allowed_hosts`
in the config is still empty. That is deliberate and it is the repo's own
standing rule: a host goes on that list only after *you* have checked its
licensing, because every one of these is redistributing IDX market data under
terms I cannot see from here.

If you want one of them enabled, say which, and I will look at its terms with
you before writing a line of fetch code.

## 6. What I am doing in the meantime

The top-ten source is legitimate, works, and is already collecting. It is not a
substitute for the full rekap but it is not nothing either: 85–90% of volume,
with the remainder rigorously bracketed rather than guessed, and a panel sized
from the frozen protocol's own power requirement — 10 names × 400 sessions,
which is what it takes to detect a real flow edge at d = 0.10.

If the IDX file lands, that panel gets replaced by an exact one and every
bracket in this project collapses to a number.
