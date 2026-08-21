"""What a top-10 broker summary can and cannot prove, stated as bounds.

THE PROBLEM THIS SOLVES
-----------------------
Every free route to IDX broker flow returns the same shape: the ten largest
buyers and the ten largest sellers, ranked independently. Roughly 85-90% of the
day's volume is inside that view and the rest is not. Until now this repo read
the table the obvious way - broker X bought 100,000 lots and sold nothing - and
that reading is simply false. X sold *some unknown amount below the tenth-ranked
seller*, and calling it zero biases every position, cost basis and P/L in the
same direction, every day, cumulatively. On BBCA over 52 sessions it drove the
market-wide net, which must be exactly zero, to -2.8 million lots.

The fix is not a better source. It is to stop pretending the missing numbers are
zero and start bracketing them.

WHAT MAKES THE BRACKET POSSIBLE
-------------------------------
The table's own footer publishes the day's market-wide totals - total value,
total lots and the VWAP. That single fact converts a biased sample of unknown
size into a CENSORED sample of known size:

    visible       named brokers, measured
    hidden pool   exactly ``total - visible`` lots, spread over unnamed brokers
    each hidden   individually at most the tenth-ranked visible broker, or it
                  would have displaced that broker from the ranking

So for any broker on any day:

    listed on a side      that side is known
    not listed on a side  that side lies in ``[0, min(rank10, hidden_pool)]``

and the net follows by subtraction. Bounds add across days, so a window gives a
bracket on cumulative inventory rather than a number that drifts.

THE ONE THING THAT IS NOT EXTRA INFORMATION
-------------------------------------------
The market-wide zero-sum identity looks like it should tighten these bounds and
does not. Hidden net is ``(T - V_buy) - (T - V_sell) = V_sell - V_buy``, which
is exactly the negative of the visible net, identically, for any table. The
identity is satisfied by construction and therefore constrains nothing. It is
still worth computing as an arithmetic check on the parse - it catches a misread
column instantly - but it is a test, not evidence.

WHAT THIS BUYS
--------------
Claims that survive: a broker's net sign when the whole bracket sits one side of
zero, relative ranking among the visible, and a bounded cumulative position.
Claims that do not: an exact cost basis, and any statement about a broker that
was never listed at all.

    from idxbot.broker_bounds import day_bounds, cumulative_bounds, certain_sign
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

#: A listed broker cannot have traded zero on the side it was listed on, so a
#: zero in either lot column means "not in that ranking", never "traded none".
#: Reading those zeros as observations is the whole bug this module exists for.
UNLISTED = 0.0

#: Below this the source prints exact figures; at or above it they are rendered
#: as ``3.4 M`` and carry one decimal of the magnitude unit.
ABBREVIATION_FLOOR = 1e6


def rounding_tolerance(value: float, floor: float = ABBREVIATION_FLOOR) -> float:
    """Largest error display rounding can put on a printed figure.

    Not a percentage. A lot count of 2,948,267 prints as ``2.9 M``, so the unit
    is a million, one decimal is kept, and the error is at most half of that
    last decimal - 50,000 lots. A flat 5% would have allowed 147,000 and thrown
    away the exactness the source does give on everything under a million, where
    figures are printed in full and the tolerance should be zero.
    """
    v = abs(float(value))
    if not np.isfinite(v) or v < floor:
        return 0.0
    unit = 10.0 ** (3.0 * np.floor(np.log10(v) / 3.0))
    return 0.05 * unit


#: How far visible volume may exceed the published total before the day is
#: called contradictory rather than merely rounded. Abbreviated cells carry one
#: decimal, so a handful of rows can each be out by a few percent in the same
#: direction; an order-of-magnitude excess cannot come from rounding.
TOTALS_SLACK = 0.02


def _rank_floor(lots: pd.Series) -> float:
    """Smallest lot count that still made the visible ranking.

    Anything censored on this side is at most this, because a larger figure
    would have displaced the tenth-ranked broker. When nobody is listed the
    ranking imposes no ceiling at all and the bound falls back to the pool.
    """
    seen = lots[lots > UNLISTED]
    return float(seen.min()) if len(seen) else float("inf")


def day_bounds(rows: pd.DataFrame, total_lot: Optional[float] = None
               ) -> pd.DataFrame:
    """Bracket every listed broker's buy, sell and net for one ticker-day.

    ``rows`` is one day of the canonical schema. ``total_lot`` defaults to the
    ``total_lot`` column when the provider attached it; without it the hidden
    pool is unknown and only the ranking ceiling applies, which is weaker but
    still sound.
    """
    cols = ["broker", "buy_lo", "buy_hi", "sell_lo", "sell_hi",
            "net_lo", "net_hi", "net_naive", "buy_listed", "sell_listed",
            "cap_buy", "cap_sell", "consistent"]
    if rows is None or rows.empty:
        return pd.DataFrame(columns=cols)
    df = rows.copy()
    for c in ("buy_lot", "sell_lot"):
        df[c] = pd.to_numeric(df.get(c), errors="coerce").fillna(0.0)

    if total_lot is None and "total_lot" in df:
        v = pd.to_numeric(df["total_lot"], errors="coerce").dropna()
        total_lot = float(v.iloc[0]) if len(v) else None

    vis_buy = float(df["buy_lot"].sum())
    vis_sell = float(df["sell_lot"].sum())
    # Visible volume exceeding the published total is a CONTRADICTION, not a
    # small pool: the rows and the footer are describing different things - a
    # board mismatch, a misparsed column, a date that slipped. Clamping the pool
    # to zero would quietly turn that into a very confident bound, which is the
    # worst possible response. The day is marked inconsistent instead.
    ok = True
    if total_lot and np.isfinite(total_lot) and total_lot > 0:
        slack = 1.0 + TOTALS_SLACK
        ok = vis_buy <= total_lot * slack and vis_sell <= total_lot * slack
    pool_buy = max(0.0, float(total_lot) - vis_buy) if total_lot else float("inf")
    pool_sell = max(0.0, float(total_lot) - vis_sell) if total_lot else float("inf")

    cap_buy = min(_rank_floor(df["buy_lot"]), pool_buy)
    cap_sell = min(_rank_floor(df["sell_lot"]), pool_sell)

    buy_listed = df["buy_lot"] > UNLISTED
    sell_listed = df["sell_lot"] > UNLISTED
    out = pd.DataFrame({
        "broker": df["broker"].astype(str).values,
        "buy_lo": np.where(buy_listed, df["buy_lot"], 0.0),
        "buy_hi": np.where(buy_listed, df["buy_lot"], cap_buy),
        "sell_lo": np.where(sell_listed, df["sell_lot"], 0.0),
        "sell_hi": np.where(sell_listed, df["sell_lot"], cap_sell),
        "net_naive": df["buy_lot"].values - df["sell_lot"].values,
        "buy_listed": buy_listed.values,
        "sell_listed": sell_listed.values,
        # carried so a broker ABSENT from this day's table entirely can still be
        # bounded when days are added up - see cumulative_bounds
        "cap_buy": cap_buy,
        "cap_sell": cap_sell,
        "consistent": ok,
    })
    out["net_lo"] = out["buy_lo"] - out["sell_hi"]
    out["net_hi"] = out["buy_hi"] - out["sell_lo"]
    return out[cols]


def visibility(rows: pd.DataFrame, total_lot: Optional[float] = None
               ) -> Dict[str, float]:
    """What share of the day the table actually shows, per side.

    This is the number that decides whether bounded inference is worth doing at
    all. At 90% visible the brackets are narrow enough to settle most sign
    questions; at 30% they would be too wide to say anything.
    """
    if rows is None or rows.empty:
        return {}
    if total_lot is None and "total_lot" in rows:
        v = pd.to_numeric(rows["total_lot"], errors="coerce").dropna()
        total_lot = float(v.iloc[0]) if len(v) else None
    if not total_lot or not np.isfinite(total_lot) or total_lot <= 0:
        return {}
    b = float(pd.to_numeric(rows["buy_lot"], errors="coerce").fillna(0).sum())
    s = float(pd.to_numeric(rows["sell_lot"], errors="coerce").fillna(0).sum())
    return {"total_lot": float(total_lot),
            "visible_buy": b, "visible_sell": s,
            "cover_buy": b / total_lot, "cover_sell": s / total_lot,
            "hidden_buy": max(0.0, total_lot - b),
            "hidden_sell": max(0.0, total_lot - s),
            "brokers_listed": int(rows["broker"].nunique())}


def zero_sum_residual(rows: pd.DataFrame,
                      total_lot: Optional[float] = None) -> float:
    """Visible net plus hidden net, which must be exactly zero.

    Not evidence - it holds identically for any table - but a sharp test of the
    parse. A misread column or a lot/value swap breaks it by orders of
    magnitude, while display rounding moves it by a fraction of a percent.
    """
    v = visibility(rows, total_lot)
    if not v:
        return float("nan")
    visible_net = v["visible_buy"] - v["visible_sell"]
    hidden_net = v["hidden_buy"] - v["hidden_sell"]
    return float(visible_net + hidden_net)


def side_intervals(rows: pd.DataFrame, side: str,
                   total_lot: Optional[float] = None) -> Dict[str, Tuple[float, float]]:
    """``{broker: (lo, hi)}`` for one side of one view, listed or censored."""
    b = day_bounds(rows, total_lot)
    if b.empty:
        return {}
    lo, hi = f"{side}_lo", f"{side}_hi"
    return {str(r.broker): (float(getattr(r, lo)), float(getattr(r, hi)))
            for r in b.itertuples()}


def _narrow(cur: Tuple[float, float], cand: Tuple[float, float],
            tol: float) -> Tuple[Tuple[float, float], bool]:
    """Intersect two intervals, but never into an empty one.

    An intersection that comes out empty means the views disagree - almost
    always because lot counts at or above a million are printed to two or three
    significant figures, so ``all`` and ``foreign + domestic`` can miss each
    other by a few thousand lots on a busy day. Letting that invert the interval
    produces a bracket whose lower bound sits ABOVE its upper, which is not a
    weaker claim than the truth, it is a nonsensical one - and the first run of
    this on real BBCA data produced exactly that, a width of -10,571 lots.

    So the candidate is widened by the display-rounding allowance first, and if
    it still cannot be reconciled the ORIGINAL interval is kept. Keeping a
    looser bracket is always sound; inventing an inverted one never is.
    """
    lo = max(cur[0], cand[0] - tol)
    hi = min(cur[1], cand[1] + tol)
    if lo > hi:
        return cur, False
    return (lo, hi), True


def _tighten(a: Tuple[float, float], f: Tuple[float, float],
             d: Tuple[float, float], rounds: int = 4
             ) -> Tuple[Tuple[float, float], Tuple[float, float],
                        Tuple[float, float], bool]:
    """Interval arithmetic on ``all = foreign + domestic``, to a fixed point.

    Three intervals tied by one linear equation. Each narrows the other two, and
    two or three passes settle it. This is where the censoring is actually
    beaten rather than merely bounded: a broker listed in the combined view and
    in the foreign view has its DOMESTIC side pinned exactly by subtraction,
    even though it never appeared in the domestic ranking at all.

    Returns the three intervals and whether every step reconciled.
    """
    ok = True
    for _ in range(rounds):
        # Three printed figures each carry their own rounding and the identity
        # ties all three, so the allowance is what each can be out by, summed.
        # Below a million the source prints in full and this is exactly zero,
        # which is what keeps the exact derivations exact.
        tol = sum(rounding_tolerance(x) for x in (a[1], f[1], d[1])
                  if np.isfinite(x))
        a, k1 = _narrow(a, (f[0] + d[0], f[1] + d[1]), tol)
        f, k2 = _narrow(f, (max(a[0] - d[1], 0.0), a[1] - d[0]), tol)
        d, k3 = _narrow(d, (max(a[0] - f[1], 0.0), a[1] - f[0]), tol)
        a = (max(a[0], 0.0), max(a[1], max(a[0], 0.0)))
        ok = ok and k1 and k2 and k3
    return a, f, d, ok


def merge_views(combined: pd.DataFrame, foreign: pd.DataFrame,
                domestic: pd.DataFrame) -> pd.DataFrame:
    """Fold the all / foreign-only / domestic-only tables into one bracket set.

    THE FREE ACCURACY UPGRADE. The same public module will filter the rekap to
    foreign-investor trades or domestic-investor trades, and the two partition
    the whole exactly - verified per broker to the lot on BBCA, where AK, BK, CC
    and ZP all reconcile with zero difference. Three consequences, all of them
    worth having:

    1. THE UNION IS WIDER. Each view publishes its own top ten, and they are not
       the same ten. On BBCA the combined view lists 14 brokers and the three
       views together list 20.
    2. FOREIGN FLOW IS NEARLY COMPLETE. The foreign view came back 99.0%
       covered - foreign buying is concentrated enough that its top ten is
       essentially all of it - so the most-watched flow in this market is close
       to exactly measurable rather than bounded.
    3. THE IDENTITY DERIVES WHAT THE RANKING HID. Two known sides give the third
       by subtraction, which turns censored quantities into exact ones.

    Returns one row per broker with the tightened combined-view brackets, plus
    the foreign and domestic components and a flag for any broker whose three
    views cannot be reconciled.
    """
    views = {"all": combined, "F": foreign, "D": domestic}
    sides = ("buy", "sell")
    iv = {s: {k: side_intervals(v, s) for k, v in views.items()} for s in sides}
    # A broker missing from a view is CENSORED in that view, not unbounded in
    # it: it failed to make that view's top ten, so it is under that view's own
    # ceiling. Defaulting the absent case to infinity instead leaks an infinite
    # bracket into every broker that appears in only one of the three tables -
    # which is exactly what the first run of this produced.
    caps = {}
    for k, v in views.items():
        b = day_bounds(v)
        caps[k] = {"buy": float(b["cap_buy"].iloc[0]) if not b.empty else np.inf,
                   "sell": float(b["cap_sell"].iloc[0]) if not b.empty else np.inf}
    everyone = sorted({b for s in sides for k in views for b in iv[s][k]})

    rows = []
    for br in everyone:
        rec: Dict[str, object] = {"broker": br, "reconciled": True}
        for s in sides:
            a = iv[s]["all"].get(br, (0.0, caps["all"][s]))
            f = iv[s]["F"].get(br, (0.0, caps["F"][s]))
            d = iv[s]["D"].get(br, (0.0, caps["D"][s]))
            a, f, d, ok = _tighten(a, f, d)
            if not ok:
                rec["reconciled"] = False
            rec[f"{s}_lo"], rec[f"{s}_hi"] = a
            rec[f"{s}_F_lo"], rec[f"{s}_F_hi"] = f
            rec[f"{s}_D_lo"], rec[f"{s}_D_hi"] = d
        rows.append(rec)

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["net_lo"] = out["buy_lo"] - out["sell_hi"]
    out["net_hi"] = out["buy_hi"] - out["sell_lo"]
    out["buy_listed"] = out["buy_lo"] >= out["buy_hi"] - 1e-9
    out["sell_listed"] = out["sell_lo"] >= out["sell_hi"] - 1e-9
    base = day_bounds(combined)
    # net_naive stays the COMBINED view's plain reading so a merged run and a
    # single-view run report the same headline number and differ only in the
    # bracket around it. A midpoint here would silently change what the column
    # means depending on how many views were fetched.
    naive = (base.set_index("broker")["net_naive"] if not base.empty
             else pd.Series(dtype=float))
    out["net_naive"] = out["broker"].map(naive).fillna(0.0)
    out["cap_buy"] = float(base["cap_buy"].iloc[0]) if not base.empty else np.inf
    out["cap_sell"] = float(base["cap_sell"].iloc[0]) if not base.empty else np.inf
    out["consistent"] = bool(base["consistent"].iloc[0]) if not base.empty else True
    return out


def foreign_net(rows: pd.DataFrame) -> float:
    """Net value bought by foreign investors, from the foreign-investor view."""
    if rows is None or rows.empty:
        return float("nan")
    b = pd.to_numeric(rows.get("buy_val"), errors="coerce").fillna(0.0).sum()
    s = pd.to_numeric(rows.get("sell_val"), errors="coerce").fillna(0.0).sum()
    return float(b - s)


def foreign_net_agreement(foreign_rows: pd.DataFrame,
                          published: Optional[float] = None) -> Dict[str, float]:
    """Check the foreign-view net against the source's own published figure.

    THE CROSS-CHECK THAT SETTLES THE INTERPRETATION. The footer prints
    ``F. NVal``, the day's net foreign value, computed upstream and independently
    of anything done here. Summing the foreign-investor view reproduces it to a
    median 0.7% across eight ticker-days - and the residual is the censoring
    itself, since that view comes back 99-100% covered rather than 100%.

    Summing the brokers the source FLAGS as foreign-owned, which is how this repo
    and most Indonesian retail analysis compute "net foreign", reproduces it to
    47%. Those are not the same quantity and should never have been treated as
    one: a foreign-owned member executes for domestic clients all day, and
    YP (Mirae) is the largest RETAIL broker in the country while carrying a
    foreign flag.
    """
    if foreign_rows is None or foreign_rows.empty:
        return {}
    if published is None and "foreign_net_val" in foreign_rows:
        v = pd.to_numeric(foreign_rows["foreign_net_val"],
                          errors="coerce").dropna()
        published = float(v.iloc[0]) if len(v) else None
    got = foreign_net(foreign_rows)
    if published is None or not np.isfinite(published) or published == 0:
        return {"computed": got}
    err = abs(got - published) / abs(published)
    return {"computed": got, "published": float(published),
            "relative_error": float(err),
            "agrees": bool(err < 0.05 or abs(got - published) < 1e9)}


def cumulative_bounds(daily: Sequence[pd.DataFrame],
                      skip_inconsistent: bool = True) -> pd.DataFrame:
    """Add per-day brackets into a bracket on cumulative inventory.

    THE TRAP THIS AVOIDS. Summing only the days a broker appears on treats every
    other day as a net of exactly zero - the same "absent means nothing"
    mistake as reading an unlisted side as zero, one level up. A broker missing
    from the whole table on some day still traded that day, somewhere inside
    ``[-cap_sell, +cap_buy]``, and a sum that skips it is not a bound at all. A
    randomised test against known full rekaps caught this: the recovered lower
    bound came out ABOVE the truth.

    So every broker is bounded across every day in the window, whether it was
    listed that day or not. Bounds widen with the window as a result, which is
    the honest shape - and it is why ``certain_sign`` only ever settles the
    desks that show up consistently.
    """
    frames = [d for d in daily if d is not None and not d.empty]
    if skip_inconsistent:
        frames = [d for d in frames
                  if "consistent" not in d or bool(d["consistent"].iloc[0])]
    cols = ["broker", "net_lo", "net_hi", "net_naive", "days", "days_seen",
            "days_buy", "days_sell", "width"]
    if not frames:
        return pd.DataFrame(columns=cols)

    everyone = sorted({b for d in frames for b in d["broker"]})
    lo = {b: 0.0 for b in everyone}
    hi = {b: 0.0 for b in everyone}
    naive = {b: 0.0 for b in everyone}
    seen = {b: 0 for b in everyone}
    nbuy = {b: 0 for b in everyone}
    nsell = {b: 0 for b in everyone}
    for d in frames:
        idx = d.set_index("broker")
        cap_b = float(d["cap_buy"].iloc[0]) if "cap_buy" in d else float("inf")
        cap_s = float(d["cap_sell"].iloc[0]) if "cap_sell" in d else float("inf")
        for b in everyone:
            if b in idx.index:
                r = idx.loc[b]
                lo[b] += float(r["net_lo"])
                hi[b] += float(r["net_hi"])
                naive[b] += float(r["net_naive"])
                seen[b] += 1
                nbuy[b] += int(bool(r["buy_listed"]))
                nsell[b] += int(bool(r["sell_listed"]))
            else:
                # absent from BOTH rankings: unbounded below by its own selling
                # and above by its own buying, each capped by the same ceilings
                lo[b] -= cap_s
                hi[b] += cap_b
    out = pd.DataFrame({
        "broker": everyone,
        "net_lo": [lo[b] for b in everyone],
        "net_hi": [hi[b] for b in everyone],
        "net_naive": [naive[b] for b in everyone],
        "days": len(frames),
        "days_seen": [seen[b] for b in everyone],
        "days_buy": [nbuy[b] for b in everyone],
        "days_sell": [nsell[b] for b in everyone],
    })
    out["width"] = out["net_hi"] - out["net_lo"]
    return out[cols].sort_values("net_naive", ascending=False).reset_index(drop=True)


def certain_sign(bounds: pd.DataFrame) -> pd.DataFrame:
    """Keep only the brokers whose direction the data actually settles.

    A bracket entirely above zero means the broker was a net buyer over the
    window whatever the censored days hid; entirely below, a net seller. A
    bracket spanning zero means the honest answer is "not determined", and
    saying so is the point - the naive reading would have given every one of
    these a confident number.
    """
    if bounds is None or bounds.empty:
        return pd.DataFrame(columns=list(getattr(bounds, "columns", []))
                            + ["direction"])
    b = bounds.copy()
    b["direction"] = np.where(b["net_lo"] > 0, "net buyer",
                              np.where(b["net_hi"] < 0, "net seller",
                                       "undetermined"))
    return b[b["direction"] != "undetermined"].reset_index(drop=True)


def settled_fraction(bounds: pd.DataFrame) -> float:
    """Share of listed brokers whose direction the bounds settle."""
    if bounds is None or bounds.empty:
        return float("nan")
    lo, hi = bounds["net_lo"], bounds["net_hi"]
    return float(((lo > 0) | (hi < 0)).mean())


def settled_flow_share(bounds: pd.DataFrame) -> float:
    """Share of the total NET FLOW carried by brokers whose direction is settled.

    Counting names is the wrong denominator and flatters the wrong way. A window
    can leave two thirds of the codes undetermined and still settle nearly all of
    the flow, because the undetermined ones are the intermittent minnows whose
    trading sits below the source's resolution - which is precisely why they are
    undetermined. What a reader needs to know is how much of the money moved is
    accounted for, not how many two-letter codes were.
    """
    if bounds is None or bounds.empty:
        return float("nan")
    size = midpoint(bounds).abs()
    total = float(size.sum())
    if total <= 0:
        return float("nan")
    settled = (bounds["net_lo"] > 0) | (bounds["net_hi"] < 0)
    return float(size[settled].sum() / total)


def midpoint(bounds: pd.DataFrame) -> pd.Series:
    """Best single estimate: the centre of the proven bracket.

    Not the same thing as the plain reading, and where the two differ it is the
    plain reading that is wrong - see :func:`naive_error`.
    """
    if bounds is None or bounds.empty:
        return pd.Series(dtype=float)
    return (bounds["net_lo"] + bounds["net_hi"]) / 2.0


def naive_error(bounds: pd.DataFrame) -> pd.Series:
    """How far the plain reading falls OUTSIDE the proven bracket, in lots.

    Zero when the obvious reading happens to be admissible; positive when it is
    provably wrong. This is the number that justifies the whole module: reading
    an unlisted side as a zero does not merely add noise, it produces values the
    data rules out. On BBCA over eighteen sessions it puts AK at -873,447 lots
    when the three views pin it at exactly -885,331.
    """
    if bounds is None or bounds.empty:
        return pd.Series(dtype=float)
    below = bounds["net_lo"] - bounds["net_naive"]
    above = bounds["net_naive"] - bounds["net_hi"]
    return np.maximum(0.0, np.maximum(below, above))


def relative_width(bounds: pd.DataFrame) -> pd.Series:
    """Bracket width as a fraction of the position it brackets.

    The number to quote per broker: 0.0 means the censoring cost nothing at all
    (the broker was listed on both sides every session), 1.0 means the bracket is
    as wide as the position and says nothing useful.
    """
    if bounds is None or bounds.empty:
        return pd.Series(dtype=float)
    size = midpoint(bounds).abs().replace(0.0, np.nan)
    return (bounds["net_hi"] - bounds["net_lo"]) / size


def bracket_frame(rows: pd.DataFrame) -> List[pd.DataFrame]:
    """Split a multi-day frame into per-day brackets, in date order."""
    if rows is None or rows.empty:
        return []
    out = []
    for _, g in rows.groupby(["ticker", "date"], sort=True):
        out.append(day_bounds(g))
    return out
