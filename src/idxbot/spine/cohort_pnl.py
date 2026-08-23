"""§9.3 — is the flow behind a broker code profitable?

WHOSE PROFIT THIS IS, WHICH IS THE WHOLE NAMING DISCIPLINE
-----------------------------------------------------------
§9.1 splits one question into two. *How does the brokerage firm make money?* is
commission on turnover, direction-irrelevant, already in their OJK filings, and
there is nothing here to reverse-engineer. *Is the flow behind broker code X
profitable?* is estimable, and what it estimates is **the aggregate P&L of the
client cohort behind that code** — thousands of accounts, overwhelmingly agency,
with the prop desk a small slice.

So everything in this module is ``cohort_pnl``. Never ``broker_profit``. The
name is the guard: §9.1 asks for it explicitly because the conceptual error
leaks into the dossiers the moment the variable is called the wrong thing.

THE FOUR STRUCTURAL LIMITS, WHICH TRAVEL WITH EVERY OUTPUT
-----------------------------------------------------------
§9.2 requires these stated, not assumed away:

    STARTING INVENTORY IS UNKNOWN. Broker summary is flow, not position. The
    walk-forward starts every series at zero, which is certainly wrong, and the
    error never fully washes out — it is a LEVEL ambiguity in inventory and
    therefore in weighted-average cost. :func:`negative_inventory_share` is the
    honest diagnostic: it counts how often the reconstruction implies a broker
    sold shares it is not recorded as holding, which is the direct measure of
    how badly the assumption bites.

    CROSSING INFLATES GROSS VOLUME. A broker printing both sides of the same
    trade adds turnover with no directional exposure, which flatters the
    denominator of margin_bps and shrinks it toward zero. ``crossing_ratio``
    measures it per broker-ticker so it can be conditioned on.

    FOREIGN NOMINEES ARE OMNIBUS. CS, CLSA, UBS, MS carry many uncorrelated end
    clients under one code. Flagged low-confidence by construction; never
    treated as one actor.

    CODES ARE NOT STABLE OVER HISTORY. Mergers and licence changes reassign
    them — see the broker code master.

WHY THERE ARE TWO ESTIMATES AND THE NOISIER ONE IS SECONDARY
-------------------------------------------------------------
§9.3: round-trip P&L is the primary estimate and full-path is the noisy
secondary. Inside an episode where inventory starts near zero, rises, and
returns near zero, the P&L is unambiguous **and independent of the
starting-inventory problem** — everything bought inside the episode is sold
inside it. That is the only clean number this data can produce, and it is
reported as such.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

#: Shares in a lot. IDX has been 100 since 2014-01-06; the 500-share era
#: predates every broker-summary series available here.
LOT = 100

#: §9.3: "Discard the first 250 trading days of each broker-ticker series."
#: Reported with and without, because on a 361-session store this throws away
#: 69% of the sample and that trade-off has to be visible rather than chosen
#: silently.
BURN_IN_DAYS = 250

#: An episode counts as closed when inventory returns within this fraction of
#: its own peak. Not an absolute share count: a broker whose peak is 10,000
#: lots and one whose peak is 10 million need the same test.
ROUND_TRIP_TOL = 0.05

#: Minimum days in an episode. A one-day spike in and out is a crossing
#: artefact, not an accumulation round trip.
MIN_EPISODE_DAYS = 3


def walk_forward(g: pd.DataFrame, close: Optional[pd.Series] = None
                 ) -> pd.DataFrame:
    """§9.3's inventory / WAC / realised / unrealised walk, one broker-ticker.

    ``g`` must be one (broker, ticker) sorted by date with buy_lot, sell_lot,
    buy_avg, sell_avg.

    Implemented as §9.3 literally specifies::

        inventory_t  = inventory_{t-1} + buy_vol_t - sell_vol_t
        WAC_t        = weighted-average cost, updated on BUYS ONLY
        realized_t   = sell_vol_t x (sell_avg_price_t - WAC_{t-1})
        unrealized_t = inventory_t x (close_t - WAC_t)

    Note what is NOT done: the sell is not capped at inventory on hand. Capping
    would hide the starting-inventory problem instead of measuring it, and
    §9.2 wants it measured. Inventory therefore goes negative on codes that
    were already long when the series began, and ``inventory < 0`` is the
    signal for exactly that.
    """
    d = g.sort_values("date").reset_index(drop=True)
    n = len(d)
    buy_sh = pd.to_numeric(d["buy_lot"], errors="coerce").fillna(0.0).to_numpy() * LOT
    sell_sh = pd.to_numeric(d["sell_lot"], errors="coerce").fillna(0.0).to_numpy() * LOT
    buy_px = pd.to_numeric(d["buy_avg"], errors="coerce").to_numpy(dtype=float)
    sell_px = pd.to_numeric(d["sell_avg"], errors="coerce").to_numpy(dtype=float)

    inv = np.zeros(n)
    wac = np.zeros(n)
    realized = np.zeros(n)
    unattributable = np.zeros(n)
    cur_inv = 0.0
    cur_wac = 0.0
    for i in range(n):
        # ONLY the part of a sell that matches shares this series is recorded
        # as holding can be attributed a cost basis. The rest came from
        # inventory the cohort already had when the series began, whose cost is
        # unknowable, and booking it against WAC is how the first version of
        # this produced +3,333 bps out of nothing: WAC starts at zero, so the
        # opening sell booked its entire proceeds as profit. The null caught
        # it — shuffled broker labels came back at +6.3 bps with a confidence
        # interval excluding zero, which no shuffle can honestly do.
        if sell_sh[i] > 0 and np.isfinite(sell_px[i]):
            known = min(sell_sh[i], max(cur_inv, 0.0))
            realized[i] = known * (sell_px[i] - cur_wac)
            unattributable[i] = sell_sh[i] - known
        if buy_sh[i] > 0 and np.isfinite(buy_px[i]):
            base = max(cur_inv, 0.0)
            tot = base + buy_sh[i]
            if tot > 0:
                cur_wac = (base * cur_wac + buy_sh[i] * buy_px[i]) / tot
        # inventory still follows §9.3 literally and is allowed to go negative,
        # because that is the direct measurement of the problem above.
        cur_inv = cur_inv + buy_sh[i] - sell_sh[i]
        inv[i] = cur_inv
        wac[i] = cur_wac

    out = pd.DataFrame({
        "date": d["date"], "broker": d.get("broker"), "ticker": d.get("ticker"),
        "buy_sh": buy_sh, "sell_sh": sell_sh,
        "buy_px": buy_px, "sell_px": sell_px,
        "inventory": inv, "wac": wac, "realized": realized,
        "unattributable_sh": unattributable,
    })
    out["gross_value"] = (buy_sh * np.nan_to_num(buy_px)
                          + sell_sh * np.nan_to_num(sell_px))
    if close is not None:
        c = pd.to_numeric(close, errors="coerce").to_numpy(dtype=float)[:n]
        out["unrealized"] = inv * (c - wac)
    else:
        out["unrealized"] = np.nan
    return out


def negative_inventory_share(w: pd.DataFrame) -> float:
    """How often the walk implies selling shares it never recorded buying.

    THE headline diagnostic for §9.2's starting-inventory problem. A code that
    was already long when the series began sells down into negative
    reconstructed inventory; the higher this runs, the more the level ambiguity
    is doing to the numbers and the less a full-path estimate is worth.
    """
    if w.empty:
        return float("nan")
    return float((w["inventory"] < 0).mean())


def crossing_ratio(g: pd.DataFrame) -> float:
    """§9.4: min(buy_val, sell_val) / max(buy_val, sell_val), per broker-ticker.

    High implies market-making or churn — turnover with little directional
    exposure. It inflates the denominator of margin_bps, so a broker can look
    unprofitable per rupiah simply by crossing a lot.
    """
    b = float((pd.to_numeric(g["buy_lot"], errors="coerce").fillna(0)
               * LOT * pd.to_numeric(g["buy_avg"], errors="coerce")).sum(skipna=True))
    s = float((pd.to_numeric(g["sell_lot"], errors="coerce").fillna(0)
               * LOT * pd.to_numeric(g["sell_avg"], errors="coerce")).sum(skipna=True))
    hi = max(b, s)
    return float(min(b, s) / hi) if hi > 0 else float("nan")


def round_trips(w: pd.DataFrame, tol: float = ROUND_TRIP_TOL,
                min_days: int = MIN_EPISODE_DAYS) -> pd.DataFrame:
    """Episodes where inventory leaves ~zero, peaks, and returns to ~zero.

    §9.3 calls this the CLEAN estimate, and the reason is worth stating: within
    such an episode everything bought is sold, so the P&L does not depend on
    what the cohort held before the series started. It is the one number here
    that the starting-inventory problem cannot corrupt.

    ``tol`` is a fraction of the episode's OWN peak, so the test scales with
    the broker.
    """
    cols = ["start", "end", "days", "peak_sh", "bought_sh", "sold_sh",
            "buy_value", "sell_value", "pnl", "gross_value"]
    if w.empty or len(w) < min_days:
        return pd.DataFrame(columns=cols)
    inv = w["inventory"].to_numpy(dtype=float)
    peak_all = np.max(np.abs(inv)) if len(inv) else 0.0
    if peak_all <= 0:
        return pd.DataFrame(columns=cols)
    near0 = np.abs(inv) <= tol * peak_all

    rows: List[Dict] = []
    i = 0
    n = len(inv)
    while i < n:
        if not near0[i]:
            i += 1
            continue
        j = i + 1
        while j < n and not near0[j]:
            j += 1
        if j >= n:
            break
        seg = slice(i, j + 1)
        days = j - i + 1
        pk = float(np.max(np.abs(inv[seg])))
        if days >= min_days and pk > tol * peak_all:
            b = w["buy_sh"].to_numpy()[seg]
            s = w["sell_sh"].to_numpy()[seg]
            bp = np.nan_to_num(w["buy_px"].to_numpy()[seg])
            sp = np.nan_to_num(w["sell_px"].to_numpy()[seg])
            bv = float((b * bp).sum())
            sv = float((s * sp).sum())
            # THE CLEAN ESTIMATE, and the reason it is clean: inside an episode
            # that opens and closes near flat, everything bought is sold, so
            # P&L is simply what came in minus what went out. No weighted-
            # average cost, no starting inventory, nothing to assume. §9.3 calls
            # this "unambiguous and independent of the initial-position
            # problem" — using realized[] here instead would drag the WAC
            # contamination straight back in.
            rows.append({
                "start": w["date"].iloc[i], "end": w["date"].iloc[j],
                "days": int(days), "peak_sh": pk,
                "bought_sh": float(b.sum()), "sold_sh": float(s.sum()),
                "buy_value": bv, "sell_value": sv,
                "pnl": sv - bv, "gross_value": bv + sv,
            })
        i = j
    return pd.DataFrame(rows, columns=cols)


def margin_bps(pnl: float, gross_value: float) -> float:
    """§9.3's headline metric: 10000 x cohort_pnl / gross_traded_value.

    Per rupiah traded, not absolute rupiah, so it is comparable across brokers
    of wildly different size and answers "how profitable is this flow" directly.
    """
    return float(10000.0 * pnl / gross_value) if gross_value else float("nan")


def bootstrap_ci(x: Sequence[float], n: int = 2000, alpha: float = 0.05,
                 seed: int = 20260823) -> tuple:
    """Percentile CI on the mean. §9.3 wants a distribution, never a point."""
    a = np.asarray([v for v in x if np.isfinite(v)], dtype=float)
    if len(a) < 3:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    m = rng.choice(a, size=(n, len(a)), replace=True).mean(axis=1)
    return (float(np.percentile(m, 100 * alpha / 2)),
            float(np.percentile(m, 100 * (1 - alpha / 2))))


def shuffle_broker_labels(df: pd.DataFrame, seed: int = 20260823
                          ) -> pd.DataFrame:
    """§9.3's null: the same computation on shuffled broker labels.

    Shuffled WITHIN each ticker-day, so the day's total flow, the number of
    codes and the distribution of their sizes are all preserved — only *which*
    code did what is destroyed. That is precisely the thing a broker-identity
    claim asserts, so it is precisely what the null must break.
    """
    rng = np.random.default_rng(seed)
    d = df.copy()
    d["broker"] = d.groupby(["ticker", "date"])["broker"].transform(
        lambda s: rng.permutation(s.to_numpy()))
    return d
