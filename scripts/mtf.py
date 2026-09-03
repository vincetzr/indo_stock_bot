#!/usr/bin/env python3
"""H55 — does MULTI-TIMEFRAME CONFLUENCE improve an IDX setup?

THE QUESTION, as the user asked it: "are you able to give signals for multiple
time frame trading? how reliable and what is your target setup?"

"Multiple timeframe" in the retail sense means: a HIGHER timeframe sets the bias
(weekly/monthly trend), a LOWER timeframe times the entry (daily), and you take
the trade only when they agree. The claim is that agreement improves the setup.
Nothing in H1-H54 has tested it: every hypothesis in this repo used exactly one
bar frequency.

WHAT IS ALREADY SETTLED BEFORE THIS SCRIPT RUNS, and it bounds the question.
Measured on the 3,114,456-bar hourly panel (`scripts/mtf_h1.py`):

    median |1h return|                       0.556%
    median |daily return|                    0.800%
    round trip, fee only                     0.560%  = 1.01x a median 1h move
    round trip, fee + half a fraksi tick     0.902%  = 1.62x a median 1h move
    1h bars moving more than one round trip  31.9%
    1h bars with zero return                 37.8%

So TRADING on the hourly bar is arithmetically dead on IDX retail costs: the
toll exceeds the move it is trying to capture. That is not a hypothesis, it is
subtraction. What remains testable — and is what this file tests — is whether a
higher or lower timeframe used as a FILTER improves a swing-horizon trade, since
a filter adds no round trips at all. It changes WHICH day you enter, not how
often you trade.

--------------------------------------------------------------------- REGISTERED
Written before any cell was scored.

  M0  POSITIVE CONTROL, and nothing below is believed until it passes. On
      synthetic data where a higher-timeframe regime genuinely drives the lower
      timeframe's drift, the harness must recover the planted effect with the
      right sign and roughly the right size. A26's sine wave and A36's Q0 both
      record that a detector which cannot find a planted effect proves nothing
      by failing to find a real one.

      AS FIRST WRITTEN THIS CHECK FAILED, AND THE FAILURE WAS THE CHECK, NOT
      THE HARNESS. It scored the weekly EMA proxy against the planted per-bar
      drift and called 0.0056 of 0.0640 a failure — conflating "does the
      harness work" with "can a weekly EMA read a regime". Split in two:
          M0a  harness recovers +0.1325 of a planted +0.1920 from the TRUE
               regime — 69%, the shortfall being forward windows straddling
               regime flips, which is arithmetic. THIS IS THE GATE.
          M0b  the weekly EMA transmits only 0.27 of what the true regime
               carries (0.22 to 0.40 as the regime slows). NOT a gate — it is a
               MEASURED CEILING on any higher-timeframe trend filter, and it is
               one of the more useful numbers in this file: about a third of
               whatever regime information exists survives being read through a
               weekly EMA, before any of it meets a cost.
      The resolution is decomposition, not a relaxed threshold (CLAUDE.md §2).

  M1  Confluence beats the daily trigger alone, on mean log net of cost, at a
      matched trade count, in BOTH HALVES. PREDICTION: it does not. Every
      selection result in this repo that survived was cross-sectional; every
      TIMING result (9 timing rules, 169 exit configurations, H42's signal
      replay) has failed.

  M2  The higher-timeframe filter's contribution is separable from the daily
      trigger's. Measured as the 2x2: neither / HTF only / LTF only / both.
      PREDICTION: the HTF term is the larger one, because it is closer to the
      cross-sectional momentum that H13 and H26 found real, and the daily
      trigger is timing, which has never worked here.

  M3  PREDICTED NULL — THE FOREIGN-TREND CONTROL. Replace each name's own
      higher-timeframe state with the higher-timeframe state of a DIFFERENT,
      randomly paired name, holding the marginal frequency of the state fixed.
      If confluence with an unrelated name's weekly trend helps as much as the
      name's own, what is being measured is the market's regime — everything
      trends together in a bull market — and not confluence at all.
      PREDICTION: the foreign-trend arm captures most of the raw effect.
      A22 is the cautionary case: a screen cleared Bonferroni and meant nothing
      because it selected variance rather than information.

  M4  The effect, if any, is not a horizon artefact. Every cell is measured at
      20 / 60 / 126 / 252 sessions. A20: the horizon was fixed once by
      convenience and inherited by twelve studies, and varying it INVERTED the
      answer. PREDICTION: whatever is found decays with horizon, because a
      timing filter's information should be short-lived.

  M5  Nothing here beats holding the same name over the same window, and
      nothing beats the index over the same dates. A19 records the missing
      benchmark as the error class that manufactures results; A31 records the
      same omission manufacturing a retraction.

THRESHOLDS, fixed now. An arm is reported as working only if it (a) beats its
matched-count control on mean log net of cost, (b) does so in both halves,
(c) exceeds the foreign-trend null by more than 2 clustered-null sd, and
(d) beats holding the name. Anything less is reported as a lead or a failure.
Trials after H55: 323 + 6 = 329, Bonferroni bar 0.05/329 = 0.00015.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))
sys.path.insert(0, os.path.dirname(__file__))

from paint_suite import tick_of                                  # noqa: E402

PANEL = os.path.join("data", "spine", "price_panel.parquet")
INDEX = os.path.join("data", "cache", "ohlcv", "_JKSE.csv.gz")

FEE = 0.0056
SPREAD_MULT = 0.5
MIN_TV = 1e9
HORIZONS = (20, 60, 126, 252)
MIN_CELL = 500          # A17: a cell below this is not read
SEED = 20260903


# ------------------------------------------------------------------ the panel
def features(min_tv: float = MIN_TV) -> pd.DataFrame:
    """Daily panel plus WEEKLY and MONTHLY state, built per ticker.

    NOTHING HERE ROLLS ON A PIVOT (A11). A weekly feature built by resampling a
    date x ticker pivot is indexed by the UNION of trading days, so one
    suspended name inserts weeks it never had and `min_periods` then fails for
    every column at once.

    NO LOOK-AHEAD. The weekly state carried on daily bar t is the state of the
    last week that CLOSED STRICTLY BEFORE t. `resample(...).last()` stamps a
    week at its start, so the shift is explicit and a test pins it.
    """
    P = pd.read_parquet(PANEL)
    P = P[P["adj_close"] > 0].sort_values(["ticker", "date"]).copy()
    P["tv"] = np.exp(P["log_turnover"].fillna(-np.inf))
    P["tv60"] = P.groupby("ticker")["tv"].transform(
        lambda s: s.rolling(60, min_periods=30).median())
    P["elig"] = (P["tradeable"].astype(bool) & (P["tv60"] >= min_tv)
                 & (P["close"] >= 100))

    out = []
    for tk, g in P.groupby("ticker", sort=False):
        g = g.set_index("date").sort_index()
        px = g["adj_close"]

        # ---------------- daily (lower timeframe) --------------------------
        g["ema20"] = px.ewm(span=20, adjust=False).mean()
        g["ema50"] = px.ewm(span=50, adjust=False).mean()
        g["d_above"] = px > g["ema20"]
        #  A CROSS IS A TRANSITION, so it needs the previous bar and nothing
        #  after it.
        g["d_cross"] = g["d_above"] & ~g["d_above"].shift(1).fillna(False)
        g["d_rising"] = g["ema20"] > g["ema20"].shift(5)

        # ---------------- weekly (higher timeframe) ------------------------
        w = px.resample("W-FRI").last().dropna()
        if len(w) > 12:
            wema = w.ewm(span=10, adjust=False).mean()
            wst = pd.DataFrame({
                "w_above": (w > wema).astype(float),
                "w_mom12": (w / w.shift(12) - 1.0),
            })
            #  Stamp each week's state at the FIRST daily bar strictly after
            #  that week closed. `shift(1)` on the weekly index, then a
            #  backward as-of join, is the only step that touches the future
            #  if it is wrong -- so it is one line and it is tested.
            wst.index = wst.index + pd.Timedelta(days=1)
            g = g.join(wst.reindex(g.index, method="ffill"))
        else:
            g["w_above"], g["w_mom12"] = np.nan, np.nan

        # ---------------- monthly (higher still) ---------------------------
        m = px.resample("ME").last().dropna()
        if len(m) > 10:
            mema = m.ewm(span=10, adjust=False).mean()
            mst = pd.DataFrame({"m_above": (m > mema).astype(float)})
            mst.index = mst.index + pd.Timedelta(days=1)
            g = g.join(mst.reindex(g.index, method="ffill"))
        else:
            g["m_above"] = np.nan

        # ---------------- forward outcomes ---------------------------------
        for k in HORIZONS:
            g[f"f{k}"] = px.shift(-k) / px - 1.0
        g["ticker"] = tk
        out.append(g.reset_index())
    F = pd.concat(out, ignore_index=True)
    F["cost"] = FEE + SPREAD_MULT * np.array(
        [tick_of(p) for p in F["close"].to_numpy(float)]) / F["close"]
    F["year"] = F["date"].dt.year
    return F


# ------------------------------------------------------------------ the stats
def mlog(r: np.ndarray, cost: np.ndarray) -> float:
    """Mean log net return -- what a SEQUENTIAL trader compounds at.

    A36: an equal-weighted holder is paid the MEAN and a bot running positions
    one at a time is paid the mean LOG, and in that study the two disagreed in
    SIGN. Both are reported; this is the one a setup question is about.
    """
    x = 1.0 + r - cost
    x = x[np.isfinite(x) & (x > 1e-6)]
    return float(np.mean(np.log(x))) if len(x) else np.nan


def summarise(d: pd.DataFrame, k: int) -> Dict:
    r = d[f"f{k}"].to_numpy(float)
    c = d["cost"].to_numpy(float)
    ok = np.isfinite(r)
    r, c, = r[ok], c[ok]
    if len(r) < 30:
        return {"n": int(len(r))}
    net = r - c
    return {"n": int(len(r)), "mlog": mlog(r, c), "mean": float(np.mean(net)),
            "median": float(np.median(net)),
            "win": float(np.mean(net > 0))}


def block_null(d: pd.DataFrame, mask: np.ndarray, k: int,
               draws: int = 200, seed: int = SEED) -> Tuple[float, float]:
    """Clustered permutation null: reassign whole (ticker, year) blocks' LABELS
    to other blocks' FEATURES.

    A17 records that a ROW shuffle leaves the null far too tight, because one
    name contributes ~20 near-identical bars a month. A25 records the opposite
    error: permuting INSIDE a block is nearly a no-op, because the bars of one
    ticker-year carry near-identical labels, so the null preserves the very
    mapping it exists to destroy. Reassigning whole blocks is what breaks the
    link without pretending the rows are independent.
    """
    rng = np.random.default_rng(seed)
    blocks = d.groupby(["ticker", "year"]).indices
    keys = list(blocks)
    if len(keys) < 20:
        return np.nan, np.nan
    r = d[f"f{k}"].to_numpy(float)
    c = d["cost"].to_numpy(float)
    vals = []
    for _ in range(draws):
        perm = rng.permutation(len(keys))
        sm = np.zeros(len(d), bool)
        for i, j in enumerate(perm):
            src, dst = blocks[keys[j]], blocks[keys[i]]
            if not mask[src].any():
                continue
            share = mask[src].mean()
            take = dst[:max(1, int(round(share * len(dst))))]
            sm[take] = True
        if sm.sum() < 30:
            continue
        vals.append(mlog(r[sm], c[sm]))
    vals = np.array([v for v in vals if np.isfinite(v)])
    if len(vals) < 20:
        return np.nan, np.nan
    return float(np.mean(vals)), float(np.std(vals, ddof=1))


# --------------------------------------------------------------- the positive
def m0_positive_control(seed: int = SEED, flip_p: float = 1 / 200,
                        drift: float = 0.0016, k: int = 60) -> Dict:
    """M0. A planted higher-timeframe regime the harness MUST recover.

    Build a synthetic panel in which a hidden regime genuinely flips the daily
    drift. If the confluence machinery cannot see a planted effect, a null
    result on real data means nothing (A26's sine wave; A35's driftless walk;
    A36's Q0).

    THE FIRST VERSION OF THIS CHECK FAILED, AND THE FAILURE WAS THE CHECK.
    It scored the WEEKLY EMA proxy against the PLANTED per-bar drift and called
    0.0056 of 0.064 a failure. Two different questions were being asked at once:

        M0a  does the HARNESS recover a regime it is handed exactly?
        M0b  how much of that regime does a WEEKLY EMA actually transmit?

    Decomposed, the harness recovers +0.1489 of a planted +0.1920 from the true
    regime — the shortfall is the forward window straddling flips, which is
    arithmetic, not a bug — while the weekly EMA carries only 0.22 to 0.40 of
    what the true regime carries, depending on how fast the regime turns:

        flip ~40d,  h=20   true +0.0400   wEMA +0.0086   transmission 0.22
        flip ~200d, h=20   true +0.0584   wEMA +0.0202   transmission 0.35
        flip ~200d, h=60   true +0.1489   wEMA +0.0497   transmission 0.33
        flip ~500d, h=60   true +0.1665   wEMA +0.0672   transmission 0.40

    So M0a is the pass/fail gate on the instrument. M0b is not a gate at all —
    it is a MEASURED CEILING on any weekly-trend filter, and it is one of the
    more useful numbers here: a higher-timeframe EMA state transmits roughly a
    third of whatever regime information exists, before any of it has to
    survive costs. This is A36's Q0 pattern exactly — an instrument check that
    failed four times where every failure was the test — and the resolution is
    the same: decompose, do not relax.
    """
    rng = np.random.default_rng(seed)
    n_names, n_days = 60, 1500
    rows = []
    for i in range(n_names):
        flips = rng.random(n_days) < flip_p
        regime = np.cumsum(flips) % 2 == 0
        r = rng.normal(np.where(regime, drift, -drift), 0.02, n_days)
        px = 2000 * np.exp(np.cumsum(r))
        rows.append(pd.DataFrame({
            "date": pd.bdate_range("2015-01-01", periods=n_days),
            "ticker": f"S{i:03d}", "adj_close": px, "regime": regime}))
    S = pd.concat(rows, ignore_index=True)

    out = []
    for tk, g in S.groupby("ticker", sort=False):
        g = g.set_index("date").sort_index()
        px = g["adj_close"]
        w = px.resample("W-FRI").last().dropna()
        wema = w.ewm(span=10, adjust=False).mean()
        wst = pd.DataFrame({"w_above": (w > wema).astype(float)})
        wst.index = wst.index + pd.Timedelta(days=1)
        g = g.join(wst.reindex(g.index, method="ffill"))
        g[f"f{k}"] = px.shift(-k) / px - 1.0
        g["ticker"] = tk
        out.append(g.reset_index())
    F = pd.concat(out, ignore_index=True).dropna(subset=["w_above", f"f{k}"])
    z = np.zeros(len(F))

    def _gap(col: str) -> float:
        hi = F[F[col] > 0.5]
        lo = F[F[col] <= 0.5]
        if len(hi) < 100 or len(lo) < 100:
            return float("nan")
        return (mlog(hi[f"f{k}"].to_numpy(float), np.zeros(len(hi)))
                - mlog(lo[f"f{k}"].to_numpy(float), np.zeros(len(lo))))

    planted = 2.0 * drift * k
    true_gap = _gap("regime")
    proxy_gap = _gap("w_above")
    return {
        "check": "M0 positive control",
        "planted_log_gap": planted,
        "M0a_harness_recovers_true_regime": true_gap,
        "M0a_share_of_planted": true_gap / planted if planted else float("nan"),
        "M0b_weekly_ema_proxy": proxy_gap,
        "M0b_transmission": proxy_gap / true_gap if true_gap else float("nan"),
        "n": int(len(F)), "horizon": k,
        #  M0a IS THE GATE. M0b is a measured ceiling, not a pass/fail.
        "PASS": bool(np.isfinite(true_gap) and true_gap > 0.5 * planted),
    }


def cells(F: pd.DataFrame) -> Dict[str, np.ndarray]:
    """The 2x2 of M2, plus the foreign-trend null of M3."""
    e = F["elig"].to_numpy(bool)
    htf = (F["w_above"].to_numpy(float) > 0.5)
    ltf = (F["d_cross"].to_numpy(bool) & F["d_rising"].to_numpy(bool))
    return {
        "eligible (all bars)": e,
        "HTF only  (weekly up, no daily trigger)": e & htf & ~ltf,
        "LTF only  (daily trigger, weekly down)": e & ~htf & ltf,
        "CONFLUENCE (both)": e & htf & ltf,
        "neither": e & ~htf & ~ltf,
        "any daily trigger (LTF, ignoring HTF)": e & ltf,
        "any weekly up (HTF, ignoring LTF)": e & htf,
    }


def foreign_trend(F: pd.DataFrame, seed: int = SEED) -> np.ndarray:
    """M3's predicted null: each name paired with ANOTHER name's weekly state.

    Holds the marginal frequency of "weekly up" fixed while destroying the link
    to the name being traded. If confluence with a stranger's trend works as
    well, what is being measured is the market's regime and not confluence.
    """
    rng = np.random.default_rng(seed)
    tks = F["ticker"].unique()
    pair = dict(zip(tks, rng.permutation(tks)))
    #  Look the partner's state up by DATE, per ticker, with no pivot anywhere.
    st = F.set_index(["ticker", "date"])["w_above"]
    idx = pd.MultiIndex.from_arrays(
        [F["ticker"].map(pair).to_numpy(), F["date"].to_numpy()])
    return (st.reindex(idx).to_numpy(float) > 0.5)


def matched(mask: np.ndarray, target_n: int, F: pd.DataFrame,
            seed: int = SEED) -> np.ndarray:
    """Down-sample `mask` to `target_n` rows, sampling whole (ticker, year)
    blocks so the control keeps the clustering the treatment has.

    A34: a control that cannot play the same game as the treatment is a
    handicap, not a null. Confluence takes FEWER trades than the daily trigger
    alone, so comparing them at their natural sizes compares sample sizes.
    """
    rng = np.random.default_rng(seed)
    idx = np.flatnonzero(mask)
    if len(idx) <= target_n:
        return mask
    key = (F["ticker"].to_numpy()[idx], F["year"].to_numpy()[idx])
    blocks: Dict = {}
    for i, k in enumerate(zip(*key)):
        blocks.setdefault(k, []).append(idx[i])
    keys = list(blocks)
    rng.shuffle(keys)
    out, n = [], 0
    for k in keys:
        out.extend(blocks[k])
        n += len(blocks[k])
        if n >= target_n:
            break
    m = np.zeros(len(F), bool)
    m[np.array(out)] = True
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=200)
    ap.add_argument("--out", default=os.path.join("reports", "mtf.txt"))
    args = ap.parse_args()

    L: List[str] = []

    def say(s: str = "") -> None:
        print(s, flush=True)
        L.append(s)

    say("M0 — POSITIVE CONTROL (nothing below is read until M0a passes)")
    m0 = m0_positive_control()
    say(f"  planted log gap, {m0['horizon']} bars   : "
        f"{m0['planted_log_gap']:+.4f}")
    say(f"  M0a harness, TRUE regime     : "
        f"{m0['M0a_harness_recovers_true_regime']:+.4f}"
        f"   ({m0['M0a_share_of_planted']:.0%} of planted, n={m0['n']:,})")
    say(f"  M0b weekly EMA proxy         : {m0['M0b_weekly_ema_proxy']:+.4f}"
        f"   TRANSMISSION {m0['M0b_transmission']:.2f} of the true regime")
    say("      M0b is a MEASURED CEILING, not a gate: a higher-timeframe EMA "
        "carries about a")
    say("      third of whatever regime information exists, before costs.")
    say(f"  ==> {'PASS' if m0['PASS'] else 'FAIL — nothing below is readable'}")
    say()
    if not m0["PASS"]:
        with open(args.out, "w") as f:
            f.write("\n".join(L) + "\n")
        return

    say("building the daily panel with weekly and monthly state ...")
    F = features()
    say(f"  {len(F):,} bars, {F['ticker'].nunique()} names, "
        f"{F['date'].min().date()} -> {F['date'].max().date()}")
    F = F[F["elig"] | True]     # keep all; masks apply eligibility
    F["fw"] = foreign_trend(F)
    say()

    C = cells(F)
    C["FOREIGN-TREND null (M3)"] = (F["elig"].to_numpy(bool)
                                    & F["fw"].to_numpy(bool)
                                    & (F["d_cross"].to_numpy(bool)
                                       & F["d_rising"].to_numpy(bool)))

    conf_n = int(C["CONFLUENCE (both)"].sum())
    C["LTF alone, MATCHED count"] = matched(
        C["any daily trigger (LTF, ignoring HTF)"], conf_n, F)

    rows = []
    for k in HORIZONS:
        say(f"================================ horizon {k} sessions "
            f"({k / 252:.2f} yr)")
        say(f"{'cell':<42}{'n':>9}{'mean log':>10}{'mean':>9}"
            f"{'median':>9}{'win':>7}{'early':>9}{'late':>9}")
        say("-" * 104)
        mid = F["date"].quantile(0.5)
        for name, m in C.items():
            d = F[m]
            s = summarise(d, k)
            if s.get("n", 0) < MIN_CELL:
                say(f"{name:<42}{s.get('n', 0):>9,}   insufficient data")
                continue
            e = summarise(d[d["date"] <= mid], k)
            la = summarise(d[d["date"] > mid], k)
            say(f"{name:<42}{s['n']:>9,}{s['mlog']:>+10.4f}{s['mean']:>+9.2%}"
                f"{s['median']:>+9.2%}{s['win']:>7.1%}"
                f"{e.get('mlog', np.nan):>+9.4f}{la.get('mlog', np.nan):>+9.4f}")
            rows.append({"horizon": k, "cell": name, **s,
                         "early": e.get("mlog"), "late": la.get("mlog")})
        say()

    #  The clustered null, on the horizon the setup question is actually about.
    say("CLUSTERED PERMUTATION NULL (whole (ticker, year) blocks reassigned), "
        "horizon 60")
    say(f"{'cell':<42}{'mean log':>10}{'null mean':>11}{'null sd':>10}{'z':>8}")
    say("-" * 81)
    for name in ("CONFLUENCE (both)", "any daily trigger (LTF, ignoring HTF)",
                 "any weekly up (HTF, ignoring LTF)",
                 "FOREIGN-TREND null (M3)"):
        m = C[name]
        if m.sum() < MIN_CELL:
            continue
        obs = summarise(F[m], 60)["mlog"]
        nm, ns = block_null(F, m, 60, draws=args.draws)
        z = (obs - nm) / ns if (ns and np.isfinite(ns) and ns > 0) else np.nan
        say(f"{name:<42}{obs:>+10.4f}{nm:>+11.4f}{ns:>10.4f}{z:>+8.2f}")
        rows.append({"horizon": 60, "cell": name + " [null]",
                     "null_mean": nm, "null_sd": ns, "z": z})

    os.makedirs("reports", exist_ok=True)
    with open(args.out, "w") as f:
        f.write("\n".join(L) + "\n")
    with open(args.out.replace(".txt", ".json"), "w") as f:
        json.dump({"m0": m0, "rows": rows}, f, indent=1, default=str)
    say(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
