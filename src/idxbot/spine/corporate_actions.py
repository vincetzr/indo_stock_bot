"""Corporate-action adjustment, and the rights issue that breaks naive versions.

CLAUDE.md §5: "Splits are easy. **Rights issues are the trap.** A heavily
dilutive rights issue needs a proper theoretical ex-rights price adjustment or
the return series shows a fake crash."

WHY A RIGHTS ISSUE IS DIFFERENT FROM A SPLIT
--------------------------------------------
A split is a relabelling. Two-for-one, and every holder has twice as many
shares worth half as much; the adjustment factor is the ratio and there is
nothing else to know.

A rights issue is a *transfer of value at a chosen price*. Holders may buy new
shares below market, so the share price falls on the ex-date by an amount that
depends on three numbers - how many new shares, at what price, against what
market price - and NONE of that is recoverable from the price series alone. A
1-for-1 issue at a 90% discount roughly halves the quoted price on ex-day. A
holder who took up their rights lost nothing whatsoever.

So the naive fix - "large drop, clean ratio, must be a split" - is exactly
wrong here. It would compute a factor from the observed drop, which is the
thing being explained, and thereby define away every rights issue as correct.

THE ONE IDENTITY EVERYTHING RESTS ON
------------------------------------
The theoretical ex-rights price is the volume-weighted average of what a
holder ends up with::

    TERP = (R * P_cum + N * S) / (R + N)

for a ratio of ``N`` new shares per ``R`` held at subscription price ``S``.
The adjustment factor applied to all PRIOR prices is ``TERP / P_cum``, which
is below 1 exactly when the issue is dilutive.

The test that matters is not that the arithmetic runs. It is that a holder who
did nothing wrong shows **no return on the ex-date**: the whole point of the
adjustment is that a fake crash disappears and a real one does not.

WHAT THIS MODULE WILL NOT DO
----------------------------
It will not infer a corporate action from a price drop. Detection lives in
:mod:`idxbot.spine.quality` and produces CANDIDATES; adjustment needs the
actual terms - ratio, subscription price, ex-date - which come from an
announcement, not from the tape. Given no terms, this module raises rather than
guessing, because a guessed factor is indistinguishable from a real one
downstream and silently rewrites history.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

#: The kinds of event this module knows how to adjust for.
KINDS = ("split", "reverse_split", "bonus", "rights", "dividend")


class MissingTerms(ValueError):
    """Raised when an adjustment is asked for without the terms to compute it.

    Deliberately fatal. The alternative - deriving a factor from the observed
    price drop - would make every event look perfectly adjusted while
    explaining nothing, and would hide a genuine crash inside a fake one.
    """


@dataclass(frozen=True)
class Action:
    """One corporate action, with the terms needed to price it.

    ``ex_date`` is the first day the price trades WITHOUT the entitlement. The
    adjustment applies to every bar strictly before it.
    """

    ticker: str
    ex_date: pd.Timestamp
    kind: str
    #: split / reverse_split: shares after per share before (2.0 = two-for-one)
    #: bonus, rights: `new` per `held`
    new: float = 0.0
    held: float = 1.0
    ratio: float = 1.0
    #: rights only: the subscription price
    subscription: float = 0.0
    #: dividend only: cash per share
    amount: float = 0.0
    source: str = ""

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}, got {self.kind!r}")


def terp(cum_price: float, new: float, held: float, subscription: float
         ) -> float:
    """Theoretical ex-rights price.

    ``new`` shares offered per ``held`` existing, at ``subscription`` each. The
    result is the weighted average price of the enlarged holding, which is what
    the market opens at if nothing else changes.
    """
    if held <= 0 or (new + held) <= 0:
        raise ValueError("held must be positive")
    if cum_price <= 0:
        raise ValueError("cum_price must be positive")
    return float((held * cum_price + new * subscription) / (held + new))


def adjustment_factor(action: Action, cum_price: Optional[float] = None
                      ) -> float:
    """Multiply every price BEFORE the ex-date by this.

    Below 1 for anything dilutive, exactly 1 for a no-op, above 1 for a reverse
    split. Rights and dividends need ``cum_price``; splits and bonuses do not,
    because their effect is purely proportional.
    """
    k = action.kind
    if k == "split":
        if action.ratio <= 0:
            raise MissingTerms("split needs a positive ratio")
        return 1.0 / float(action.ratio)
    if k == "reverse_split":
        if action.ratio <= 0:
            raise MissingTerms("reverse_split needs a positive ratio")
        return float(action.ratio)
    if k == "bonus":
        # Free shares: the holding grows, total value does not.
        if action.held <= 0:
            raise MissingTerms("bonus needs a positive `held`")
        return float(action.held / (action.held + action.new))
    if k == "rights":
        if cum_price is None or not np.isfinite(cum_price):
            raise MissingTerms(
                "a rights adjustment needs the cum-rights price. Deriving it "
                "from the observed drop would explain the drop with itself.")
        if action.subscription <= 0 or action.new <= 0:
            raise MissingTerms(
                "a rights adjustment needs the subscription price and the "
                "ratio; these come from the announcement, not the tape.")
        return float(terp(cum_price, action.new, action.held,
                          action.subscription) / cum_price)
    if k == "dividend":
        if cum_price is None or not np.isfinite(cum_price) or cum_price <= 0:
            raise MissingTerms("a dividend adjustment needs the cum price")
        if action.amount < 0:
            raise ValueError("dividend amount cannot be negative")
        return float((cum_price - action.amount) / cum_price)
    raise ValueError(f"unhandled kind {k!r}")


def dilution(action: Action, cum_price: Optional[float] = None) -> float:
    """How much of the quoted price the event removes, as a fraction.

    A convenience for reporting: 0.5 means the quote halves on the ex-date
    while the holder's wealth is unchanged. It is the number that makes a fake
    crash legible as a fake crash.
    """
    return 1.0 - adjustment_factor(action, cum_price)


def adjust(prices: pd.DataFrame, actions: Sequence[Action],
           price_columns: Sequence[str] = ("open", "high", "low", "close"),
           volume_column: str = "volume") -> pd.DataFrame:
    """Back-adjust a price frame for a list of actions.

    Prices strictly BEFORE each ex-date are multiplied by that action's factor,
    and volume is divided by it so that price x volume - traded value - is
    preserved. Getting that second part wrong is a quiet way to corrupt every
    liquidity measure in the project while the price chart looks perfect.

    Actions are applied newest-first so that factors compound correctly: a
    stock that split and then had a rights issue needs both applied to the
    oldest bars and only one to the bars in between.
    """
    d = prices.copy()
    if d.empty:
        return d
    d["date"] = pd.to_datetime(d["date"])
    d = d.sort_values("date").reset_index(drop=True)
    for a in sorted(actions, key=lambda x: pd.Timestamp(x.ex_date),
                    reverse=True):
        ex = pd.Timestamp(a.ex_date).normalize()
        before = d["date"] < ex
        if not before.any():
            continue
        # The cum price is the last CLOSE before the ex-date, which is what the
        # exchange itself uses as the reference.
        cum = float(d.loc[before, "close"].iloc[-1])
        f = adjustment_factor(a, cum)
        for c in price_columns:
            if c in d:
                d.loc[before, c] = pd.to_numeric(d.loc[before, c],
                                                 errors="coerce") * f
        if volume_column in d and f > 0:
            d.loc[before, volume_column] = pd.to_numeric(
                d.loc[before, volume_column], errors="coerce") / f
    return d


def ex_date_return(prices: pd.DataFrame, ex_date) -> float:
    """The close-to-close return across an ex-date.

    On a correctly adjusted series this is the stock's REAL move that day, and
    the fake crash is gone. It is the acceptance test for the whole module.
    """
    d = prices.copy()
    d["date"] = pd.to_datetime(d["date"])
    d = d.sort_values("date").reset_index(drop=True)
    ex = pd.Timestamp(ex_date).normalize()
    before = d[d["date"] < ex]
    on = d[d["date"] >= ex]
    if before.empty or on.empty:
        return float("nan")
    prev = float(before["close"].iloc[-1])
    now = float(on["close"].iloc[0])
    return float(now / prev - 1.0) if prev > 0 else float("nan")


def describe(action: Action, cum_price: Optional[float] = None) -> str:
    """One line a human can check against the announcement."""
    try:
        f = adjustment_factor(action, cum_price)
    except MissingTerms as exc:
        return f"{action.ticker} {action.ex_date:%Y-%m-%d} {action.kind}: {exc}"
    body = {
        "split": f"1:{action.ratio:g}",
        "reverse_split": f"{action.ratio:g}:1",
        "bonus": f"{action.new:g} free per {action.held:g}",
        "rights": (f"{action.new:g} new per {action.held:g} at "
                   f"{action.subscription:,.0f}"),
        "dividend": f"{action.amount:,.2f} per share",
    }[action.kind]
    return (f"{action.ticker} {action.ex_date:%Y-%m-%d} {action.kind} "
            f"({body}): factor {f:.4f}, removes {(1 - f):.1%} of the quote")


def from_records(rows: Iterable[Dict]) -> List[Action]:
    """Build actions from plain dicts, e.g. a parsed announcement feed."""
    out = []
    for r in rows:
        out.append(Action(
            ticker=str(r["ticker"]).upper(),
            ex_date=pd.Timestamp(r["ex_date"]).normalize(),
            kind=str(r["kind"]),
            new=float(r.get("new", 0.0) or 0.0),
            held=float(r.get("held", 1.0) or 1.0),
            ratio=float(r.get("ratio", 1.0) or 1.0),
            subscription=float(r.get("subscription", 0.0) or 0.0),
            amount=float(r.get("amount", 0.0) or 0.0),
            source=str(r.get("source", "")),
        ))
    return out
