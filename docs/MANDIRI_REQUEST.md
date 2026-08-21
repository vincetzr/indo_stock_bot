# Asking Mandiri Sekuritas for data access

## What is actually being asked for, and what to expect

You are a client of an IDX member. That entitles you to **look at** broker
summary and running trade inside MOST — it does not, by itself, entitle you or
anything you run to receive it as a feed. The distinction matters because the
constraint sits on Mandiri, not on you: **IDX's market-data rules bind the
member**, and letting a client's software pull the same data in bulk is a
redistribution question they have to answer to the exchange for.

So the realistic ladder of outcomes, best to worst:

| outcome | likelihood | what it gives |
|---|---|---|
| A real API / data feed for own-account use | **low** | everything; would replace the whole free pipeline |
| A CSV/Excel export from MOST | **plausible** | drop into `data/inbox/`, already supported |
| "Use the screen" | **most likely** | already handled by `scripts/ocr_broker.py` |
| A quote for a licensed IDX feed | possible | Rp 17.9–44m/month, needs a PT and a 300% deposit |

Ask anyway. It costs one email, the answer is worth knowing precisely rather
than by inference, and if they *do* have an institutional API the whole
accuracy problem disappears.

**Do not send them credentials, and do not give any automated system your MOST
login.** That is a live trading account. Nothing in this repository asks for it
and nothing should — a broker login that can place orders is not a data source.

---

## Draft, Indonesian

> **Perihal: Permintaan informasi akses data pasar (broker summary & running trade)**
>
> Yth. Tim Layanan Nasabah Mandiri Sekuritas,
>
> Saya nasabah Mandiri Sekuritas dan sedang melakukan riset kuantitatif atas
> data perdagangan IDX untuk keperluan pribadi (analisis portofolio sendiri,
> bukan untuk didistribusikan ulang atau dijual kembali).
>
> Saya ingin menanyakan empat hal:
>
> 1. **Ekspor data.** Apakah MOST menyediakan fasilitas ekspor (CSV/Excel) untuk
>    **broker summary harian** dan **running trade**? Jika ya, apakah tersedia
>    untuk data historis, dan sampai berapa lama ke belakang?
>
> 2. **Akses API.** Apakah Mandiri Sekuritas menyediakan akses API (REST/FIX/
>    websocket) untuk nasabah — baik ritel maupun institusi — untuk data pasar
>    dan/atau eksekusi? Jika tersedia, apa saja syarat dan biayanya?
>
> 3. **Lisensi data IDX.** Untuk kebutuhan data *running trade* yang memuat
>    **kode anggota bursa pembeli dan penjual per transaksi**, apakah hal ini
>    dimungkinkan melalui Mandiri Sekuritas, atau harus melalui lisensi langsung
>    dari IDX (ITCH/Total View)? Jika melalui IDX, apakah Mandiri Sekuritas dapat
>    membantu proses tersebut?
>
> 4. **Batasan penggunaan.** Untuk data yang saya lihat di platform MOST, apa
>    batasan penggunaan yang berlaku bagi nasabah — khususnya untuk analisis
>    pribadi yang tidak didistribusikan ulang?
>
> Terima kasih atas bantuannya.
>
> Hormat saya,
> [nama] — [nomor rekening efek]

Send to `care_center@mandirisek.co.id`, or through the in-app support channel so
it lands attached to your account.

---

## Draft, English

> **Subject: Market data access enquiry — broker summary and running trade**
>
> Dear Mandiri Sekuritas Client Services,
>
> I am a Mandiri Sekuritas client conducting quantitative research on IDX
> trading data for personal use — analysis of my own portfolio, not for
> redistribution or resale.
>
> I would like to ask four things:
>
> 1. **Export.** Does MOST provide a CSV/Excel export for the daily **broker
>    summary** and for **running trade**? If so, is historical data available,
>    and how far back?
>
> 2. **API access.** Does Mandiri Sekuritas offer API access (REST/FIX/
>    websocket) to clients — retail or institutional — for market data and/or
>    execution? If so, what are the requirements and costs?
>
> 3. **IDX licensing.** For running trade carrying the **buyer and seller member
>    codes on each transaction**, is that obtainable through Mandiri Sekuritas,
>    or does it require a direct IDX licence (ITCH / Total View)? If the latter,
>    can Mandiri Sekuritas assist with that process?
>
> 4. **Permitted use.** For data I can already see in MOST, what use restrictions
>    apply to me as a client — specifically for personal analysis that is not
>    redistributed?
>
> Thank you for your help.
>
> [name] — [securities account number]

---

## If they say yes to an export

Drop the file into `data/inbox/` and run:

```bash
python3 scripts/broker_collect.py --ingest data/inbox
```

The importer already handles CSV, Excel and saved HTML, detects whether a file
is a **running trade** export or a **broker summary**, and flags which. A running
trade export is the one that matters: every print carries a buyer and a seller
code, so it rebuilds the **complete** rekap for all ~90 members — no censoring,
no brackets, exact positions and exact cost basis. That is the single change
that would retire `idxbot.broker_bounds` entirely.

## If they say "use the screen"

That is already supported and sends no request to anybody:

```bash
python3 scripts/ocr_broker.py shot.png --ticker BBCA --date 2026-08-20
```

The parser validates every row against `value = lots × 100 × average` and
refuses the table rather than storing a plausible-looking wrong number.
