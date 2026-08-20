#!/usr/bin/env python3
"""Reverse-engineer how each broker actually trades: builds, exits, and at what P/L.

THE IDEA
--------
A broker summary gives more than volume. It gives ``buy_avg`` and ``sell_avg`` -
the broker's own VWAP on each side, each day. So their book can be reconstructed:

    inventory_t   = inventory_{t-1} + buy_lot - sell_lot
    cost basis    = weighted average of everything they bought
    realised P/L  = sell_lot x (sell_avg - cost_basis) x 100 shares

And that last line is the whole point. Every sale is a decision made at a known
profit or loss against their own average cost, so the DISTRIBUTION of that number
is the broker's exit habit - where they take profit and where they cut. It does
not have to be inferred from price action; it is arithmetic on their own prints.

WHAT THIS CAN AND CANNOT SEE, STATED FIRST
------------------------------------------
The public broker summary shows the TOP TEN per side. A broker below the cut on a
given day traded an unknown amount, so every inventory built from it drifts.
Result 19 measured the damage: BBCA's cumulative top-10 net over 52 sessions came
to -2,808,171 lots, where a complete rekap must sum to exactly zero.

This module does not pretend that away. It:

  * measures each broker's APPEARANCE RATE - the share of days they were visible;
  * BOUNDS the unobserved volume. On a day a broker is not in the top ten they
    traded strictly less than the tenth-ranked broker did, so summing that cutoff
    over their missing days gives a hard upper bound on the error;
  * refuses to report an exit profile for any broker whose bound exceeds their
    observed position, because for those the inventory is noise wearing a number.

A broker who is in the top ten nearly every day is measurable. One who appears
occasionally is not, and saying so is the difference between analysis and
astrology.

    python3 scripts/broker_habits.py --ticker BBCA
    python3 scripts/broker_habits.py --ticker BBCA --min-appearance 0.5
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))
sys.path.insert(0, os.path.dirname(__file__))

from idxbot.config import load_config           # noqa: E402
from idxbot.data.cache import Cache             # noqa: E402
from idxbot.data.ohlcv import YahooOHLCV        # noqa: E402
from account_sim import load_ohlc               # noqa: E402

LOT = 100
DAILY_STORE = os.path.join("data", "cache", "broker_daily")
LEGACY_STORE = os.path.join("data", "cache", "ipot_broker")


def load_daily(ticker: str) -> pd.DataFrame:
    """Every true single-day broker table for one name, newest store first."""
    frames = []
    for pat in (os.path.join(DAILY_STORE, f"{ticker}_*.csv.gz"),
                os.path.join(LEGACY_STORE, f"{ticker}_????????_RG.csv.gz")):
        for f in sorted(glob.glob(pat)):
            try:
                d = pd.read_csv(f)
            except Exception:
                continue
            if {"broker", "buy_lot", "sell_lot"} <= set(d.columns):
                frames.append(d)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "broker"])
    for c in ("buy_lot", "sell_lot", "buy_avg", "sell_avg"):
        if c not in df.columns:
            df[c] = np.nan
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    # one row per broker per day; a repeated import must not double-count
    return (df.groupby(["date", "broker"], as_index=False)
              .agg({"buy_lot": "max", "sell_lot": "max",
                    "buy_avg": "max", "sell_avg": "max"}))


def visibility(df: pd.DataFrame) -> pd.DataFrame:
    """Appearance rate per broker, and the hard bound on what was missed.

    On a day a broker is absent from the top ten they traded strictly less than
    the smallest visible broker that day. Summing that cutoff across their absent
    days is an upper bound on the volume this data cannot see.
    """
    days = sorted(df["date"].unique())
    per_day_min = (df.assign(tot=df["buy_lot"] + df["sell_lot"])
                     .groupby("date")["tot"].min())
    rows = []
    for b, g in df.groupby("broker"):
        seen = set(g["date"])
        missed = [d for d in days if d not in seen]
        rows.append({
            "broker": b,
            "days_seen": len(seen),
            "appearance": len(seen) / max(len(days), 1),
            "observed_lot": float((g["buy_lot"] + g["sell_lot"]).sum()),
            "unseen_bound": float(per_day_min.reindex(missed).fillna(0).sum()),
        })
    V = pd.DataFrame(rows)
    V["bound_ratio"] = V["unseen_bound"] / V["observed_lot"].replace(0, np.nan)
    return V.sort_values("appearance", ascending=False)


def ledger(df: pd.DataFrame, broker: str) -> pd.DataFrame:
    """One broker's book: inventory, weighted-average cost, realised P/L per day.

    Weighted-average cost accounting: buying moves the basis, selling does not.
    A sale realises ``lots x (sell_avg - basis) x 100`` - the broker's own P/L on
    their own prints, at their own average price.
    """
    g = df[df["broker"] == broker].sort_values("date")
    inv = 0.0
    basis = np.nan
    rows = []
    for _, r in g.iterrows():
        buy, sell = float(r["buy_lot"]), float(r["sell_lot"])
        bavg, savg = float(r["buy_avg"]), float(r["sell_avg"])
        realised = np.nan
        pl_pct = np.nan
        if sell > 0 and np.isfinite(basis) and basis > 0 and savg > 0:
            realised = sell * (savg - basis) * LOT
            pl_pct = savg / basis - 1.0
        if buy > 0 and bavg > 0:
            if not np.isfinite(basis) or inv <= 0:
                basis = bavg
            else:
                basis = (basis * inv + bavg * buy) / (inv + buy)
        inv += buy - sell
        rows.append({"date": r["date"], "buy_lot": buy, "sell_lot": sell,
                     "buy_avg": bavg, "sell_avg": savg, "inventory": inv,
                     "basis": basis, "realised": realised, "pl_pct": pl_pct})
    return pd.DataFrame(rows)


def exit_profile(led: pd.DataFrame) -> Dict[str, float]:
    """Where this broker takes profit and where they cut, from their own sales."""
    s = led[led["sell_lot"] > 0].dropna(subset=["pl_pct"])
    if len(s) < 3:
        return {}
    w = s["sell_lot"]
    wins, losses = s[s["pl_pct"] > 0], s[s["pl_pct"] < 0]
    return {
        "sales": len(s),
        "win_rate": float((s["pl_pct"] > 0).mean()),
        "median_exit": float(s["pl_pct"].median()),
        "vw_exit": float((s["pl_pct"] * w).sum() / w.sum()) if w.sum() else np.nan,
        "take_profit": float(wins["pl_pct"].median()) if len(wins) else np.nan,
        "cut_loss": float(losses["pl_pct"].median()) if len(losses) else np.nan,
        "worst_held": float(s["pl_pct"].min()),
        "best_taken": float(s["pl_pct"].max()),
        "realised_total": float(s["realised"].sum()),
    }


def technical_context(df: pd.DataFrame, ticker: str,
                      band: float = 0.08) -> Optional[pd.DataFrame]:
    """Were they buying into strength or into weakness?

    Joins each broker-day to the technical state that was KNOWN THEN - the band
    colour from the previous close, and the distance to the 200-day average - so
    "they bought in a down leg" is a statement about what a person could have
    seen at the time, not about what the chart looks like now.
    """
    sys.path.insert(0, os.path.dirname(__file__))
    from paint_live import band_state
    cfg = load_config()
    loader = YahooOHLCV(cfg, Cache(cfg.path("data.cache_dir", "data/cache")))
    px = load_ohlc(loader, ticker)
    if px is None or px.empty:
        return None
    close = px["close"]
    st, _ = band_state(close.to_numpy(float), band)
    # lag by one bar: the colour that governed a day was set by the prior close
    state = pd.Series(st, index=close.index).shift(1)
    vs200 = (close / close.rolling(200).mean() - 1.0).shift(1)
    ctx = pd.DataFrame({"green": state, "vs200": vs200})
    ctx.index = ctx.index.normalize()

    d = df.copy()
    d["key"] = pd.to_datetime(d["date"]).dt.normalize()
    d = d.join(ctx, on="key")
    d = d.dropna(subset=["green"])
    if d.empty:
        return None
    rows = []
    for b, g in d.groupby("broker"):
        buys, sells = g[g["buy_lot"] > 0], g[g["sell_lot"] > 0]
        if len(buys) < 3:
            continue
        bw = buys["buy_lot"]
        rows.append({
            "broker": b,
            "buy_days": len(buys),
            "buy_in_green": float((buys["green"] > 0).mean()),
            "vw_buy_in_green": float((buys["green"] * bw).sum() / bw.sum())
            if bw.sum() else np.nan,
            "sell_in_green": float((sells["green"] > 0).mean()) if len(sells) else np.nan,
            "buy_vs200": float(buys["vs200"].median()),
        })
    return pd.DataFrame(rows) if rows else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default="BBCA")
    ap.add_argument("--min-appearance", type=float, default=0.5,
                    help="ignore brokers seen on fewer than this share of days")
    args = ap.parse_args()
    os.makedirs("reports", exist_ok=True)

    df = load_daily(args.ticker)
    print(f"{'=' * 96}\n BROKER HABITS — {args.ticker}\n{'=' * 96}")
    if df.empty:
        print(" no daily broker data for this name.\n")
        print(" This needs TRUE SINGLE-DAY tables. Range aggregates cannot build "
              "an inventory:\n a window stamped with one date says what happened "
              "over a month, not on a day.")
        print(" Capture with scripts/ocr_broker.py or scripts/paste_broker.py.")
        return 1

    days = df["date"].nunique()
    print(f" {days} sessions, {df['broker'].nunique()} brokers seen, "
          f"{df['date'].min():%Y-%m-%d} to {df['date'].max():%Y-%m-%d}")

    V = visibility(df)
    V.to_csv(f"reports/broker_visibility_{args.ticker}.csv", index=False)
    print(f"\n{'=' * 96}\n WHO IS MEASURABLE — top-10 truncation decides this"
          f"\n{'=' * 96}")
    print(f" {'broker':<8}{'days seen':>11}{'appearance':>12}"
          f"{'observed lot':>15}{'unseen bound':>15}{'bound/obs':>11}")
    for _, r in V.head(12).iterrows():
        print(f" {r['broker']:<8}{int(r['days_seen']):>11}"
              f"{r['appearance']:>12.0%}{r['observed_lot']:>15,.0f}"
              f"{r['unseen_bound']:>15,.0f}"
              f"{(r['bound_ratio'] if np.isfinite(r['bound_ratio']) else 0):>11.2f}")

    usable = V[(V["appearance"] >= args.min_appearance)
               & (V["bound_ratio"].fillna(9e9) < 1.0)]
    print(f"\n {len(usable)} of {len(V)} brokers clear the bar "
          f"(seen on >= {args.min_appearance:.0%} of days AND unseen bound below "
          f"observed volume).")
    if usable.empty:
        print(" None is measurable on this sample. That is the honest answer, "
              "not a failure of\n method: with the top ten only, a broker who "
              "drops out of view takes their\n inventory with them.")
        return 0

    print(f"\n{'=' * 96}\n HOW THEY EXIT — every sale, priced against their own "
          f"average cost\n{'=' * 96}")
    print(f" {'broker':<8}{'sales':>7}{'win':>7}{'take profit':>13}"
          f"{'cut loss':>11}{'median exit':>13}{'worst held':>12}"
          f"{'realised Rp':>16}")
    rows = []
    for b in usable["broker"]:
        led = ledger(df, b)
        prof = exit_profile(led)
        if not prof:
            continue
        rows.append({"broker": b, **prof})
        print(f" {b:<8}{int(prof['sales']):>7}{prof['win_rate']:>7.0%}"
              f"{prof['take_profit']:>+13.2%}{prof['cut_loss']:>+11.2%}"
              f"{prof['median_exit']:>+13.2%}{prof['worst_held']:>+12.2%}"
              f"{prof['realised_total']:>16,.0f}")
    if not rows:
        print(" no broker had enough sales to profile.")
        return 0
    P = pd.DataFrame(rows)
    P.to_csv(f"reports/broker_habits_{args.ticker}.csv", index=False)

    print(f"\n{'=' * 96}\n WHAT THIS SAMPLE SUPPORTS\n{'=' * 96}")
    print(f" median take-profit across {len(P)} profiled brokers: "
          f"{P['take_profit'].median():+.2%}")
    print(f" median cut-loss:                           "
          f"{P['cut_loss'].median():+.2%}")
    asym = P["take_profit"].median() / abs(P["cut_loss"].median()) \
        if P["cut_loss"].median() else np.nan
    print(f" profit-to-loss ratio at exit:              {asym:.2f}x")
    print(f"\n {days} sessions is a demonstration, not a finding. These exits are "
          f"measured on\n whatever fraction of each broker's book was visible, "
          f"and the bound column above\n says how much was not. Treat the "
          f"MACHINERY as validated and the NUMBERS as\n provisional until the "
          f"panel is long enough for the layer-2 protocol.")
    T = technical_context(df, args.ticker)
    if T is not None:
        T = T[T["broker"].isin(P["broker"])]
        if len(T):
            print(f"\n{'=' * 96}\n WHEN THEY BUY — the technical state that was "
                  f"visible at the time\n{'=' * 96}")
            print(f" {'broker':<8}{'buy days':>10}{'bought in up leg':>19}"
                  f"{'weighted by size':>19}{'sold in up leg':>17}"
                  f"{'vs 200d when buying':>22}")
            for _, r in T.iterrows():
                print(f" {r['broker']:<8}{int(r['buy_days']):>10}"
                      f"{r['buy_in_green']:>19.0%}{r['vw_buy_in_green']:>19.0%}"
                      f"{(r['sell_in_green'] if np.isfinite(r['sell_in_green']) else 0):>17.0%}"
                      f"{r['buy_vs200']:>+22.1%}")
            # The gap between the two columns is the finding, not either alone.
            # Buying on more up-days than down-days while putting MORE SIZE into
            # down-days is a contrarian book hiding behind a trend-following
            # day count, and only the size-weighted figure sees it.
            T["size_tilt"] = T["vw_buy_in_green"] - T["buy_in_green"]
            gap = float(T["size_tilt"].median())
            print(f"\n by day count they buy in up legs "
                  f"{T['buy_in_green'].median():.0%} of the time; weighted by "
                  f"SIZE, {T['vw_buy_in_green'].median():.0%}.")
            print(f" median gap {gap:+.0%} — "
                  + ("their BIGGEST buys go into weakness while their frequent "
                     "small ones\n follow strength. A contrarian book behind a "
                     "trend-following day count."
                     if gap < -0.05 else
                     "size and frequency point the same way; no hidden tilt."))
            biggest = T.nsmallest(3, "size_tilt")[["broker", "size_tilt"]]
            print(" most size-contrarian: "
                  + ", ".join(f"{r.broker} {r.size_tilt:+.0%}"
                              for r in biggest.itertuples()))
            T.to_csv(f"reports/broker_technical_{args.ticker}.csv", index=False)

    print(f"\n{'=' * 96}\n WHAT TO DISTRUST HERE\n{'=' * 96}")
    print(" The PERCENTAGE exits are the robust column: each is one sale priced "
          "against a\n basis built from that broker's own visible buys.")
    print(" The REALISED RUPIAH column is the fragile one. It compounds every "
          "basis error\n across the whole window, so a broker who dropped out "
          "of the top ten for a\n week carries that gap into every later figure. "
          "Read the signs, not the totals.")

    print(f"\n -> reports/broker_habits_{args.ticker}.csv, "
          f"reports/broker_visibility_{args.ticker}.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
