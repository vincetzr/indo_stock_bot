# Getting IDX broker data through your own Mandiri Sekuritas account

You use **Growin'** (not MOST), so this is written for Growin'. It matters: the
two products have different data surfaces, and Growin' turns out to carry the
things this project needs.

---

## What Growin' actually has

From Mandiri Sekuritas' own user guide (*User Guidelines Growin' — Juli 2025*,
page 45), **Pro View → Stock Details** carries:

| screen | the guide's description | why it matters here |
|---|---|---|
| **Broker Summary** | *"Rangkuman aktivitas jual beli saham oleh para broker"* | **the prize** — see below |
| **Foreign Activity** | *"Ringkasan aktivitas jual beli saham oleh investor asing"* | the investor-type split, independently |
| **Trade Done** | *"Daftar transaksi jual beli saham yang telah terjadi di pasar"* | executed prints |

And separately, **Running Trade** on the Quick Menu — real-time prints across the
whole exchange, filterable by Side (All/Buy/Sell), Stock Code, Price Range,
Change %, and Minimal Lot.

There is also a **web platform at `pro.growin.id`**, which matters because you
are on Linux — screenshots and a full-size table are far easier there than on a
phone.

This corrects something I said earlier. I told you running trade "is not
available free". More precisely: **it is not available as a feed**, but you can
already *see* it, and reading your own screen sends no request to anybody. That
is the one route that is both free and clean, and `scripts/ocr_broker.py` was
built for exactly it.

---

## The one question that decides everything

**How many broker rows does Growin's Broker Summary show?**

| what you see | what it means |
|---|---|
| ~10 buyers and ~10 sellers | same as the free source; the bracketing in `idxbot.broker_bounds` stays necessary |
| **all ~40–90 members** | **the censoring disappears entirely** — exact positions, exact cost basis, and `broker_bounds` becomes unnecessary |

Most Indonesian broker platforms show the full rekap to their own clients,
because the member already receives it. If Growin' does, this project stops
estimating and starts measuring.

**Please check and send me one screenshot** of Pro View → Stock Details → Broker
Summary for any liquid name (BBCA is ideal — I have 228 sessions of it to check
against). Scroll to the bottom first so I can see whether the list ends at ten.

Send the Foreign Activity screen too if it is easy. I can cross-check it against
the foreign/domestic split I recovered from the public source, where the two
partition the whole to **zero lots** across 37 broker-sides.

---

## What to screenshot, and what not to bother with

**Worth it — Broker Summary.** The whole table fits in one or two shots, so a
screenshot captures the entire day. This is the high-value capture.

```bash
python3 scripts/ocr_broker.py shot.png --ticker BBCA --date 2026-08-20
```

Every row is validated against `value = lots × 100 × average`; a misread digit
breaks the identity and the table is refused rather than stored as a
plausible-looking wrong number.

**Not worth it — Running Trade.** A screenful is perhaps thirty prints out of
tens of thousands in a session. As a way to rebuild the rekap it is hopeless —
the end-of-day Broker Summary already covers 85–90% of volume in a single
capture. Running Trade is for watching the market live, not for building
history.

**The exception:** if Growin' can *export* Running Trade to CSV, that is the
single most valuable file in this whole project. Every print carries a buyer and
a seller code, so it rebuilds the **complete** rekap for all ~90 members — no
censoring, no brackets, exact cost basis. `scripts/import_broker_data.py`
already detects and handles that format:

```bash
python3 scripts/broker_collect.py --ingest data/inbox
```

---

## One thing not to do

**Do not give any automated system your Growin' login.** It is a live trading
account that can place orders; it is not a data source. Nothing in this
repository asks for one and nothing should. Screenshots of your own screen are
the clean route and they are already supported.

---

## If you want to ask Mandiri directly

Worth one email — if they have an export or an API, everything above becomes
unnecessary. Send to `care_center@mandirisek.co.id` or through in-app support so
it attaches to your account.

> **Perihal: Permintaan informasi ekspor data & akses API (Growin')**
>
> Yth. Tim Layanan Nasabah Mandiri Sekuritas,
>
> Saya nasabah pengguna aplikasi Growin' dan sedang melakukan analisis
> kuantitatif atas portofolio saya sendiri (untuk keperluan pribadi, tidak
> untuk didistribusikan ulang maupun dijual kembali).
>
> Saya ingin menanyakan empat hal:
>
> 1. **Ekspor data.** Apakah Growin' (aplikasi atau `pro.growin.id`) menyediakan
>    fasilitas ekspor CSV/Excel untuk **Broker Summary**, **Foreign Activity**,
>    atau **Running Trade / Trade Done**? Jika ya, apakah mencakup data historis
>    dan sampai berapa lama ke belakang?
>
> 2. **Kelengkapan Broker Summary.** Apakah Broker Summary di Growin'
>    menampilkan **seluruh anggota bursa** yang bertransaksi pada saham
>    tersebut, atau hanya 10 teratas per sisi?
>
> 3. **Akses API.** Apakah tersedia akses API (REST/FIX/websocket) untuk nasabah
>    — ritel maupun institusi — untuk data pasar dan/atau eksekusi? Jika ada,
>    apa syarat dan biayanya?
>
> 4. **Batasan penggunaan.** Untuk data yang saya lihat di Growin', batasan
>    penggunaan apa yang berlaku bagi nasabah untuk analisis pribadi yang tidak
>    didistribusikan ulang?
>
> Terima kasih atas bantuannya.
>
> Hormat saya,
> [nama] — [nomor rekening efek]

**Expected answer, honestly:** an export is plausible, an API is unlikely, and
"use the app" is most likely. The constraint sits on Mandiri rather than on you
— IDX's market-data rules bind the member, so a client's software receiving a
bulk feed is a redistribution question they answer to the exchange for. That is
also why the screenshot route is the clean one: you are licensed to look at your
own screen.
