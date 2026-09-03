#!/usr/bin/env python3
"""H54 — the buy-and-hold benchmark harness. One definition, impossible to game.

Every earlier goal in this project was satisfiable and meaningless. ">=+4% per
trade at 80%+" was met by drawing two lines on a randomly chosen stock (H51).
"Beat the index" was nearly met by nine quarters holding one stock (H52). The
common failure is that the target did not contain the alternative the holder
would actually take.

So this harness fixes the alternative — three of them — and a strategy must beat
ALL THREE, net of cost, POSITIVE IN BOTH HALVES:

  BH_INDEX     buy and hold the IHSG on a total-return basis over the
               strategy's own window. The zero-effort option.
  BH_UNIVERSE  buy an equal-weighted basket of everything eligible on the
               FIRST bar and never touch it again. This is the "you could have
               thrown a dart and gone to sleep" option, and it is the one that
               isolates whether REBALANCING earns its keep.
  BH_PICKS     buy the strategy's OWN FIRST BASKET and hold it to the end.
               This is the ADRO test: the rule picked those names, so does
               trading them beat simply owning them? A rule can have real
               selection skill and still fail here, and that is the point.

Plus a RANDOM control drawn from the same universe at the same turnover, so a
win has to be selection rather than exposure or luck.

WHAT MAKES THIS HARD TO CHEAT
  * `select` is handed ONLY the rows of one rebalance bar. It cannot see the
    future because the future is not in the dataframe.
  * Delisting is realised at the last real print, never carried forward (H41
    had a position whose ticker vanished held at its last price forever).
  * Cost is `fee` + `spread_mult` * fraksi-harga tick, both explicit, because a
    blended figure hides an execution assumption inside what looks like a fee.
  * Every benchmark is measured over the STRATEGY'S OWN WINDOW, first entry to
    last exit. A19 records comparing quantities over different windows as the
    error class that manufactures results, and it has been committed twice here.
  * The half-split is on the RETURN PERIOD, not the start date.

Usage from a strategy script:

    from bhbench import Bench, load
    B = Bench(load())
    def select(day):           # day = one rebalance bar, all eligible names
        s = day.nlargest(10, "mom12_1")
        return list(zip(s["ticker"], [1/len(s)]*len(s)))
    print(B.evaluate(select, label="top-10 momentum", freq=63))
"""

from __future__ import annotations

import os
import sys
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))
sys.path.insert(0, os.path.dirname(__file__))

from paint_suite import tick_of                                  # noqa: E402

PANEL = os.path.join("data", "spine", "price_panel.parquet")
INDEX = os.path.join("data", "cache", "ohlcv", "_JKSE.csv.gz")
OUT = "reports"

FEE = 0.0056           # A5: 0.28 buy + 0.18 sell + 0.10 sell tax
SPREAD_MULT = 0.5      # half a tick: patient but not perfect execution
MIN_TV = 1e9           # Rp1bn/day trailing, point-in-time
IDX_YIELD = 0.0177     # A19: measured top-decile dividend yield
MIN_UNIV = 40          # H52: below this a "portfolio" is one or two names
MIN_BASKET = 5


def load(min_tv: float = MIN_TV) -> pd.DataFrame:
    P = pd.read_parquet(PANEL)
    P = P[P["adj_close"] > 0].sort_values(["ticker", "date"])
    P["tv"] = np.exp(P["log_turnover"].fillna(-np.inf))
    P["tv60"] = P.groupby("ticker")["tv"].transform(
        lambda s: s.rolling(60, min_periods=30).median())
    P["elig"] = (P["tradeable"].astype(bool) & (P["tv60"] >= min_tv)
                 & (P["close"] >= 100))
    return P


class Prices:
    """Per-ticker arrays. Nothing here rolls on a pivot (A11)."""

    def __init__(self, P: pd.DataFrame):
        self.d: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        for tk, g in P.groupby("ticker", sort=False):
            self.d[tk] = (g["date"].to_numpy(),
                          g["adj_close"].to_numpy(float))

    def at(self, tk: str, day) -> float:
        dt, px = self.d.get(tk, (None, None))
        if dt is None:
            return np.nan
        i = np.searchsorted(dt, day)
        return float(px[i]) if i < len(dt) and dt[i] == day else np.nan

    def exit_price(self, tk: str, sell_day) -> float:
        """Price on `sell_day`, or the last print the name ever made.

        A NAME THAT STOPS PRINTING IS A DELISTING, NOT A HOLD FOREVER.
        """
        dt, px = self.d.get(tk, (None, None))
        if dt is None:
            return np.nan
        j = np.searchsorted(dt, sell_day, side="right") - 1
        return float(px[j]) if j >= 0 else np.nan


def _cagr(mult: float, yrs: float) -> float:
    return float(max(mult, 1e-9) ** (1.0 / max(yrs, 1e-9)) - 1.0)


def half_cagr(curve: Sequence[Tuple]) -> Tuple[float, float]:
    """CAGR in the early and late half of one equity path.

    THE SPLIT IS ON THE RETURN PERIOD, not the start date: every arm here
    starts at the same time, so splitting on `start` would put all of them in
    one bucket and silently report a replication test that never ran.
    """
    if len(curve) < 6:
        return float("nan"), float("nan")
    mid = len(curve) // 2
    out = []
    for seg in (curve[:mid + 1], curve[mid:]):
        d0, e0 = seg[0]
        d1, e1 = seg[-1]
        y = (d1 - d0).astype("timedelta64[D]").astype(float) / 365.25
        out.append(_cagr(e1 / max(e0, 1e-12), y))
    return out[0], out[1]


class Bench:
    def __init__(self, P: pd.DataFrame, fee: float = FEE,
                 spread_mult: float = SPREAD_MULT):
        self.P = P
        self.PX = Prices(P)
        self.dates = np.sort(P["date"].unique())
        self.fee = fee
        self.spread_mult = spread_mult
        J = pd.read_csv(INDEX, parse_dates=["date"]).sort_values("date")
        self.J = J.set_index("date")["close"]

    # ------------------------------------------------------------ benchmarks
    def index_cagr(self, a, b) -> float:
        s = self.J[(self.J.index >= pd.Timestamp(a))
                   & (self.J.index <= pd.Timestamp(b))]
        if len(s) < 50:
            return float("nan")
        y = (s.index[-1] - s.index[0]).days / 365.25
        #  Total-return basis: the names run on adj_close and already are total
        #  return, ^JKSE is a PRICE index. A19 records comparing them raw as the
        #  error that manufactured a result.
        return _cagr(float(s.iloc[-1] / s.iloc[0]), y) + IDX_YIELD

    def hold_basket(self, tks: Sequence[str], a, b) -> float:
        """Equal-weighted, bought on `a`, untouched, valued on `b`."""
        rets, tolls = [], []
        for tk in tks:
            p0 = self.PX.at(tk, a)
            p1 = self.PX.exit_price(tk, b)
            if np.isfinite(p0) and p0 > 0 and np.isfinite(p1) and p1 > 0:
                rets.append(p1 / p0 - 1.0)
                #  The tick is per NAME and per PRICE, not a flat guess: a
                #  Rp200 stock pays 1.00% a round trip and a Rp10,000 one pays
                #  0.25%, and a buy-and-hold benchmark that ignores that is
                #  being handed a discount the strategy does not get.
                tolls.append(self.fee + self.spread_mult * tick_of(p0) / p0)
        if not rets:
            return float("nan")
        y = (pd.Timestamp(b) - pd.Timestamp(a)).days / 365.25
        #  ONE round trip, not zero: even a buy-and-hold is bought and sold.
        return _cagr(float(np.mean(rets)) + 1.0 - float(np.mean(tolls)), y)

    def _carry(self, prev: Dict[str, float], a, b, eq: float, gross: float):
        """A mark the strategy cannot act on: HOLD THE EXISTING BOOK.

        THE BUG THIS FIXES WAS WORTH MORE THAN ANY EFFECT THE HARNESS MEASURED.
        The first version appended the unchanged equity and moved on, so a
        strategy earned ZERO across every skipped mark while all three
        buy-and-hold benchmarks kept compounding. At freq=252 that was 9 of 27
        marks, including windows in which the IHSG ran +90.6%, +45.2% and
        +95.5% — a systematic handicap of 1.8 to 6.7 points a year, applied to
        the strategy and to nothing else.

        The guard was meant to stop a DEGENERATE NEW BASKET (H52's one-name
        book). Refusing to BUY a thin cross-section is right; liquidating a book
        you already hold because the screen went quiet is not something any
        holder would do, and it is not what the benchmark does either.
        """
        if not prev:
            return eq, gross, prev
        tks, rets, ws = [], [], []
        for t, w in prev.items():
            p0 = self.PX.at(t, a)
            if not np.isfinite(p0) or p0 <= 0:
                continue
            p1 = self.PX.exit_price(t, b)
            if not np.isfinite(p1) or p1 <= 0:
                continue
            tks.append(t)
            rets.append(p1 / p0 - 1.0)
            ws.append(w)
        if not rets:
            return eq, gross, prev
        ws = np.array(ws) / np.sum(ws)
        rets = np.array(rets)
        r = float(np.dot(ws, rets))
        #  THE WEIGHTS DRIFT, and `prev` is what the next turnover is measured
        #  against. Returning the pre-drift weights would charge the strategy
        #  for a rebalance it had already been given for free.
        gw = ws * (1.0 + rets)
        gw = gw / gw.sum()
        return eq * (1.0 + r), gross * (1.0 + r), dict(zip(tks, gw.tolist()))

    # -------------------------------------------------------------- the walk
    def walk(self, select: Callable, freq: int = 63, offset: int = 0,
             rng: np.random.Generator | None = None) -> Dict:
        marks = self.dates[offset::freq]
        if len(marks) < 8:
            return {}
        by_date = {d: g for d, g in
                   self.P[self.P["date"].isin(marks)].groupby("date")}
        eq, gross = 1.0, 1.0
        prev: Dict[str, float] = {}
        curve, turns, costs, sizes = [], [], [], []
        first_basket: List[str] = []
        #  THE WINDOW STARTS AT THE FIRST ACTUAL TRADE, NOT THE FIRST MARK.
        #  `elig` needs 30 bars of trailing turnover, so nothing is eligible on
        #  the panel's first date and the strategy sits in cash for years. A
        #  first version measured every benchmark from marks[0] and returned NaN
        #  for both buy-and-hold arms, because the universe was EMPTY there.
        #  That is A19's error class -- comparing quantities measured over
        #  different windows -- and it has now been committed three times in
        #  this repo, so the start date is derived from the trades themselves.
        t0 = None
        for a, b in zip(marks[:-1], marks[1:]):
            day = by_date.get(a)
            if day is None:
                eq, gross, prev = self._carry(prev, a, b, eq, gross)
                curve.append((b, eq))
                continue
            day = day[day["elig"]]
            if len(day) < MIN_UNIV:
                eq, gross, prev = self._carry(prev, a, b, eq, gross)
                curve.append((b, eq))
                continue
            if rng is not None:
                k = min(max(len(first_basket) or 10, MIN_BASKET), len(day))
                idx = rng.choice(len(day), k, replace=False)
                picks = [(t, 1.0 / k) for t in day["ticker"].to_numpy()[idx]]
            else:
                picks = select(day.copy())
            picks = [(t, float(w)) for t, w in (picks or [])
                     if np.isfinite(w) and w > 0]
            if len(picks) < MIN_BASKET:
                eq, gross, prev = self._carry(prev, a, b, eq, gross)
                curve.append((b, eq))
                continue
            tot = sum(w for _, w in picks)
            cur = {t: w / tot for t, w in picks}
            if not first_basket:
                first_basket = list(cur)
                t0 = a
            sizes.append(len(cur))
            keys = set(cur) | set(prev)
            turn = 0.5 * sum(abs(cur.get(k, 0.0) - prev.get(k, 0.0))
                             for k in keys)
            px_a = {t: self.PX.at(t, a) for t in cur}
            toll = [self.fee + self.spread_mult * tick_of(v) / v
                    for v in px_a.values() if np.isfinite(v) and v > 0]
            toll = float(np.mean(toll)) if toll else self.fee
            eq *= (1.0 - turn * toll)
            turns.append(turn)
            costs.append(turn * toll)
            rets, ws = [], []
            for t, w in cur.items():
                p0 = px_a[t]
                if not np.isfinite(p0) or p0 <= 0:
                    continue
                p1 = self.PX.exit_price(t, b)
                if not np.isfinite(p1) or p1 <= 0:
                    continue
                rets.append(p1 / p0 - 1.0)
                ws.append(w)
            if rets:
                ws = np.array(ws) / np.sum(ws)
                r = float(np.dot(ws, rets))
                eq *= (1.0 + r)
                gross *= (1.0 + r)
            prev = cur
            curve.append((b, eq))
        if not first_basket or t0 is None:
            return {}
        curve = [c for c in curve if c[0] > t0]
        if len(curve) < 6:
            return {}
        a0, b1 = t0, curve[-1][0]
        yrs = (b1 - a0).astype("timedelta64[D]").astype(float) / 365.25
        e0, e1 = half_cagr([(a0, 1.0)] + list(curve))
        return {"cagr": _cagr(eq, yrs), "gross": _cagr(gross, yrs),
                "years": yrs, "start": a0, "end": b1,
                "early": e0, "late": e1, "eq": eq,
                "basket": float(np.mean(sizes)) if sizes else np.nan,
                "turnover": float(np.mean(turns)) if turns else np.nan,
                "cost_yr": float(np.sum(costs) / yrs),
                "first_basket": first_basket, "curve": curve}

    # ----------------------------------------------------------- the verdict
    def evaluate(self, select: Callable, label: str, freq: int = 63,
                 draws: int = 6, phases: int = 6) -> Dict:
        """Verdict across `phases` rebalance calendars, not one.

        THE REBALANCE PHASE IS A FREE PARAMETER NOBODY CHOSE. `marks` starts at
        `dates[offset]`, and offset 0 is the panel's first bar for no reason but
        that it is first. Three separate strategies passed this harness at
        offset 0 and failed at every other offset tested — the PASS was a
        property of the START DATE, which is A20's lesson (a parameter fixed by
        convenience and inherited by every study) applied to the harness itself.

        So a PASS now requires a MAJORITY of phases, and the headline CAGR is
        the MEDIAN across them. The spread across phases is reported because it
        is the honest error bar on every number here: if a strategy's CAGR moves
        four points depending on which Tuesday you start, no single run of it
        means anything.
        """
        offs = [int(round(i * freq / max(phases, 1))) for i in range(phases)]
        runs = [self._verdict(select, label, freq, o, draws) for o in offs]
        ok = [v for v in runs if v.get("ok")]
        if not ok:
            return runs[0] if runs else {"label": label, "ok": False,
                                         "why": "no phase produced a path"}
        med = float(np.median([v["cagr"] for v in ok]))
        base = min(ok, key=lambda v: abs(v["cagr"] - med))
        out = dict(base)
        npass = sum(bool(v["PASS"]) for v in ok)
        out.update({
            "phases": len(ok), "phases_pass": npass,
            "cagr_med": med,
            "cagr_lo": float(np.min([v["cagr"] for v in ok])),
            "cagr_hi": float(np.max([v["cagr"] for v in ok])),
            "phase_cagrs": [v["cagr"] for v in ok],
            "pass_offset0": bool(runs[0].get("PASS")) if runs else False,
            "PASS": bool(npass * 2 > len(ok)),
        })
        return out

    def _verdict(self, select: Callable, label: str, freq: int, offset: int,
                 draws: int) -> Dict:
        r = self.walk(select, freq=freq, offset=offset)
        if not r:
            return {"label": label, "ok": False, "offset": offset,
                    "why": "no valid equity path (too few rebalances or "
                           "baskets below the floor)"}
        a0, b1 = r["start"], r["end"]
        uni0 = self.P[(self.P["date"] == a0) & self.P["elig"]]["ticker"].tolist()
        bh_index = self.index_cagr(a0, b1)
        bh_universe = self.hold_basket(uni0, a0, b1)
        bh_picks = self.hold_basket(r["first_basket"], a0, b1)
        #  THE CONTROL MUST RUN ON THE SAME CALENDAR. A first version omitted
        #  `offset` here, so at every phase but zero the random arm was measured
        #  over a different window from the strategy it is the control for --
        #  A19's error class, inside the control written to prevent it.
        ctl = [self.walk(select, freq=freq, offset=offset,
                         rng=np.random.default_rng(s)) for s in range(draws)]
        ctl = [c for c in ctl if c]
        rand = float(np.mean([c["cagr"] for c in ctl])) if ctl else np.nan
        rand_sd = (float(np.std([c["cagr"] for c in ctl], ddof=1))
                   if len(ctl) > 1 else np.nan)
        #  The half-split of each benchmark, so "positive in both halves" is a
        #  real test rather than a comparison of a split path to a whole one.
        mid = r["curve"][len(r["curve"]) // 2][0]
        bench_e = {"index": self.index_cagr(a0, mid),
                   "universe": self.hold_basket(uni0, a0, mid),
                   "picks": self.hold_basket(r["first_basket"], a0, mid)}
        bench_l = {"index": self.index_cagr(mid, b1),
                   "universe": self.hold_basket(uni0, mid, b1),
                   "picks": self.hold_basket(r["first_basket"], mid, b1)}
        beats = {k: r["cagr"] > v for k, v in
                 (("index", bh_index), ("universe", bh_universe),
                  ("picks", bh_picks))}
        both = {k: (r["early"] > bench_e[k] and r["late"] > bench_l[k])
                for k in bench_e}
        return {"label": label, "ok": True, "freq": freq, "offset": offset,
                "cagr": r["cagr"], "gross": r["gross"], "years": r["years"],
                "start": str(pd.Timestamp(a0).date()),
                "end": str(pd.Timestamp(b1).date()),
                "basket": r["basket"], "turnover": r["turnover"],
                "cost_yr": r["cost_yr"], "early": r["early"], "late": r["late"],
                "bh_index": bh_index, "bh_universe": bh_universe,
                "bh_picks": bh_picks, "random": rand, "random_sd": rand_sd,
                "beats_index": beats["index"],
                "beats_universe": beats["universe"],
                "beats_picks": beats["picks"], "beats_random": r["cagr"] > rand,
                "both_halves_index": both["index"],
                "both_halves_universe": both["universe"],
                "both_halves_picks": both["picks"],
                "PASS": bool(all(beats.values()) and all(both.values())
                             and r["cagr"] > rand)}


def report(v: Dict) -> str:
    if not v.get("ok"):
        return f"{v['label']}: FAILED TO RUN — {v.get('why')}"
    L = [f"{v['label']}   [{v['start']} → {v['end']}, {v['years']:.1f}yr, "
         f"basket {v['basket']:.0f}, turnover {v['turnover']:.0%}, "
         f"cost {v['cost_yr']:.2%}/yr]",
         f"  STRATEGY            {v['cagr']:+8.2%}   "
         f"(early {v['early']:+.2%} / late {v['late']:+.2%})"]
    for k, nm in (("index", "BH index"), ("universe", "BH universe"),
                  ("picks", "BH its own picks")):
        L.append(f"  {nm:<18}{v['bh_' + k]:+8.2%}   "
                 f"beats={'YES' if v['beats_' + k] else 'no':<3} "
                 f"both halves={'YES' if v['both_halves_' + k] else 'no'}")
    L.append(f"  {'random control':<18}{v['random']:+8.2%}   "
             f"beats={'YES' if v['beats_random'] else 'no':<3} "
             f"(sd {v['random_sd']:.2%})")
    if "phases" in v:
        #  The phase spread is the honest error bar. A strategy whose CAGR moves
        #  four points depending on which bar the calendar starts on has not
        #  measured anything, however good the median looks.
        L.append(f"  {'phase spread':<18}{v['cagr_lo']:+8.2%} .. "
                 f"{v['cagr_hi']:+.2%}  (median {v['cagr_med']:+.2%}, "
                 f"{v['phases_pass']}/{v['phases']} phases pass"
                 f"{', offset-0 only' if v['pass_offset0'] and v['phases_pass'] == 1 else ''})")
    L.append(f"  ==> {'PASS' if v['PASS'] else 'FAIL'}")
    return "\n".join(L)
