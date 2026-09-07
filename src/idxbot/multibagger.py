"""§22-25 — the multibagger scenario layer, built to REFUSE to predict.

WHY THIS IS NOT THE ENGINE THE BRIEF ASKED FOR, AND WHY BUILDING THAT ONE WOULD
BE THE WORST THING IN THIS REPO. §24 asks for a scenario model: revenue growth,
margin expansion, re-rating, and out comes a probability that a name is a
ten-bagger. Two facts make that unbuildable here and both are measured, not
argued:

  * NO FUNDAMENTALS AT PANEL SCALE. H57 established that no filing-date source
    is reachable from this environment: Yahoo `quoteSummary` returns 429 on
    every retry, `yfinance`/`curl_cffi` cannot use the proxy, and defeating a
    fingerprint check is out of bounds regardless. `data/cache/fundamentals`
    holds 59 names of 725. The only point-in-time fundamental-shaped series
    that exists here is `listed_shares`, 2019-2025.

  * EVEN WITH THEM, THE SAMPLE COULD NOT VALIDATE A NAME-LEVEL CLAIM. H23
    measured the effective n behind a ten-year multibagger statistic at
    **56 for the whole panel and ~6 per decile**, with only **30 distinct
    names ever in the top decile**. That is a list, not a population.

So the deliverable is a CALCULATOR, and the discipline is in what it will not
do. It takes the analyst's own assumptions, does the arithmetic exactly, and
puts the answer beside the MEASURED historical base rate for that horizon —
clearly separated, and labelled with which is which. It never multiplies a
scenario by a probability, never scores a thesis, and never returns a number
that could be read as "this name has an X% chance".

THE §55 LADDER IS THE WHOLE API. Every value this module returns carries a
`rung`:

    DATA           a measurement from the panel
    INTERPRETATION arithmetic on data
    THESIS         the analyst's assumption, restated
    SIGNAL         a rule this repo has tested
    PREDICTION     never produced here

A22 records the packaging failure this is guarding against: quoting a tight
screen's 21.2% for a loosened one, and returning four names when ten were
asked for. A21 records the sharper one: a **ten-year** doubling rate written in
a summary without "over ten years" beside it and read, reasonably, as a
one-year number — and the tilt INVERTS below three to five years, so it was not
merely a rescaling. Every rate here is returned with its horizon attached, in
the same object, because the fix belongs in the code that prints it rather than
in the discipline of whoever quotes it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

#: Measured base rates. Source: `scripts/decade.py` BY_HORIZON (H23/H24), which
#: is itself pinned by four tests. (years, whole liquid universe, liquid
#: decile, core 3-of-3 cell).
#:
#: THE TILT INVERTS BELOW ~3-5 YEARS and that is why all three columns are kept
#: rather than the headline one. At one year the liquid decile touches 2x
#: **4.2%** of the time against **9.5%** for the liquid universe it is drawn
#: from, and the core cell reads **1.0%**. Quoting the ten-year 82.8% without
#: its horizon is not an approximation, it is the wrong side of the trade.
BASE_RATES: Tuple[Tuple[float, float, float, float], ...] = (
    (1.0, 0.095, 0.042, 0.010),
    (2.0, 0.189, 0.114, 0.054),
    (3.0, 0.270, 0.246, 0.195),
    (5.0, 0.390, 0.493, 0.494),
    (7.5, 0.467, 0.627, 0.695),
    (10.0, 0.555, 0.700, 0.828),
)

#: H26's measured downside for the one screen that cleared this repo's bar.
#: P(2x) 10.5%, P(end <= 0.5) 4.1%, skew 2.60 against a null of 1.20 +- 0.15,
#: over a ONE-YEAR hold.
SCREEN_ONE_YEAR = {"p_double": 0.105, "p_halve": 0.041, "skew": 2.60,
                   "horizon_years": 1.0}

#: Effective sample size behind the ten-year figures (H23). Printed with every
#: base rate, because 6,683 windows and 56 independent observations are the
#: same table read two ways and only one of them is the truth.
EFFECTIVE_N = {"whole_panel_10y": 56, "per_decile_10y": 6,
               "distinct_names_ever_in_decile": 30}

RUNGS = ("DATA", "INTERPRETATION", "THESIS", "SIGNAL", "PREDICTION")


@dataclass(frozen=True)
class Value:
    """One number, its rung, and the sentence that qualifies it.

    A bare float is what lets a THESIS number get quoted as a DATA number three
    messages later. The rung travels with the value.
    """
    value: float
    rung: str
    label: str
    note: str = ""

    def __post_init__(self):
        if self.rung not in RUNGS:
            raise ValueError(f"unknown rung {self.rung!r}; expected {RUNGS}")
        if self.rung == "PREDICTION":
            raise ValueError(
                "this module does not produce PREDICTION-rung values: no "
                "fundamentals at panel scale (H57) and an effective n of ~6 "
                "per decile (H23) cannot support a name-level claim")

    def __str__(self) -> str:
        n = f"  [{self.note}]" if self.note else ""
        return f"{self.rung:<14} {self.label}: {self.value:,.4g}{n}"


@dataclass
class Scenario:
    """The analyst's assumptions. Every field is a THESIS, none is a fact."""
    ticker: str
    horizon_years: float
    revenue_cagr: float = 0.0
    margin_now: float = 0.0
    margin_then: float = 0.0
    multiple_now: float = 0.0
    multiple_then: float = 0.0
    share_growth: float = 0.0          # dilution: +0.05 = 5% more shares a year
    source: str = "analyst assumption, not measured"

    def implied(self) -> Dict[str, Value]:
        """The arithmetic, exactly, with no probability anywhere.

        multiple of money = revenue growth x margin change x re-rating
                            / dilution
        """
        if self.horizon_years <= 0:
            raise ValueError("horizon_years must be positive")
        rev = (1.0 + self.revenue_cagr) ** self.horizon_years
        marg = ((self.margin_then / self.margin_now)
                if self.margin_now > 0 and self.margin_then > 0 else 1.0)
        rerate = ((self.multiple_then / self.multiple_now)
                  if self.multiple_now > 0 and self.multiple_then > 0 else 1.0)
        dilute = (1.0 + self.share_growth) ** self.horizon_years
        mult = rev * marg * rerate / max(dilute, 1e-9)
        cagr = mult ** (1.0 / self.horizon_years) - 1.0
        note = f"over {self.horizon_years:g} years; {self.source}"
        return {
            "revenue_factor": Value(rev, "THESIS", "revenue growth factor",
                                    note),
            "margin_factor": Value(marg, "THESIS", "margin change factor",
                                   note),
            "rerating_factor": Value(rerate, "THESIS", "re-rating factor",
                                     note),
            "dilution_factor": Value(dilute, "THESIS", "dilution factor", note),
            "implied_multiple": Value(mult, "INTERPRETATION",
                                      "implied multiple of money",
                                      "arithmetic on the assumptions above; "
                                      + note),
            "implied_cagr": Value(cagr, "INTERPRETATION", "implied CAGR", note),
        }

    def sensitivity(self, field_name: str,
                    values: List[float]) -> List[Tuple[float, float]]:
        """Implied multiple across a range of one assumption.

        A single scenario is a point estimate of an opinion. §2 asks for effect
        size and uncertainty; the only uncertainty available here is how much
        the answer moves when the assumption does, so the calculator makes that
        cheap rather than optional.
        """
        if not hasattr(self, field_name):
            raise ValueError(f"no such assumption: {field_name}")
        out = []
        original = getattr(self, field_name)
        try:
            for v in values:
                setattr(self, field_name, v)
                out.append((v, self.implied()["implied_multiple"].value))
        finally:
            setattr(self, field_name, original)
        return out


def base_rate(horizon_years: float, cell: str = "liquid_universe"
              ) -> Value:
    """Measured P(touch 2x) at a horizon. Interpolated only BETWEEN measured
    points, never extrapolated past them.

    `cell` is one of `liquid_universe`, `liquid_decile`, `core_3of3`.
    """
    cols = {"liquid_universe": 1, "liquid_decile": 2, "core_3of3": 3}
    if cell not in cols:
        raise ValueError(f"unknown cell {cell!r}; expected {list(cols)}")
    j = cols[cell]
    xs = [r[0] for r in BASE_RATES]
    ys = [r[j] for r in BASE_RATES]
    if horizon_years < xs[0] or horizon_years > xs[-1]:
        raise ValueError(
            f"horizon {horizon_years:g}y is outside the measured range "
            f"{xs[0]:g}-{xs[-1]:g}y. Extrapolating is not available: the tilt "
            f"INVERTS inside this range (the liquid decile touches 2x 4.2% at "
            f"one year against 70.0% at ten), so a curve fitted through it "
            f"says nothing about either end.")
    for i in range(len(xs) - 1):
        if xs[i] <= horizon_years <= xs[i + 1]:
            w = ((horizon_years - xs[i]) / (xs[i + 1] - xs[i])
                 if xs[i + 1] > xs[i] else 0.0)
            v = ys[i] + w * (ys[i + 1] - ys[i])
            break
    note = (f"over {horizon_years:g} years, {cell}; H23/H24, IN-SAMPLE "
            f"(holdout spent at H16), effective n "
            f"{EFFECTIVE_N['whole_panel_10y']} whole-panel and "
            f"~{EFFECTIVE_N['per_decile_10y']} per decile at ten years")
    return Value(v, "DATA", "P(touch 2x)", note)


def compare(s: Scenario, cell: str = "liquid_universe") -> List[str]:
    """The analyst's scenario beside the measured base rate. Never combined.

    THE TWO NUMBERS ARE NOT MULTIPLIED AND CANNOT BE. The scenario is a
    conditional statement about one company under assumptions; the base rate is
    an unconditional frequency over a cell. Multiplying them would produce
    something with the FORM of a probability and no referent, and it is exactly
    the number a reader would quote. So they are printed side by side with the
    rung on each, and the reader does the joining or does not.
    """
    imp = s.implied()
    br = base_rate(s.horizon_years, cell)
    mult = imp["implied_multiple"].value
    L = [f"{s.ticker} — scenario over {s.horizon_years:g} years", ""]
    L += [f"  {v}" for v in imp.values()]
    L += ["", f"  {br}"]
    L.append("")
    L.append(f"  The scenario implies {mult:.2f}x. Historically, "
             f"{br.value:.1%} of names in")
    L.append(f"  the `{cell}` cell touched 2x over {s.horizon_years:g} years.")
    L.append("  THESE ARE NOT THE SAME KIND OF NUMBER AND ARE NOT COMBINED:")
    L.append("  one is arithmetic on your assumptions, the other a measured")
    L.append("  frequency over a cell. No probability is attached to the")
    L.append("  scenario, because nothing in this repo can estimate one — no")
    L.append("  fundamentals at panel scale (H57), and an effective n of ~6")
    L.append("  per decile at ten years (H23).")
    if mult >= 2.0:
        L.append("")
        L.append(f"  Reaching {mult:.2f}x needs "
                 f"{imp['implied_cagr'].value:.1%} a year sustained for "
                 f"{s.horizon_years:g} years.")
    return L


def what_would_falsify(s: Scenario) -> List[str]:
    """§9.6's FALSIFICATION field, generalised to a scenario.

    A thesis with no stated disconfirming observation is not a thesis. The
    thresholds below are derived from the assumptions themselves rather than
    invented: each is the level at which that leg contributes nothing.
    """
    out = []
    if s.revenue_cagr:
        out.append(f"revenue growth falls below {s.revenue_cagr / 2:.1%} for "
                   f"two consecutive years (half the assumed rate)")
    if s.margin_now > 0 and s.margin_then > s.margin_now:
        out.append(f"margin fails to exceed {s.margin_now:.1%} — the assumed "
                   f"expansion to {s.margin_then:.1%} contributes nothing "
                   f"below its starting point")
    if s.multiple_now > 0 and s.multiple_then > s.multiple_now:
        out.append(f"the multiple stays at or below {s.multiple_now:.1f}x "
                   f"through the horizon")
    if s.share_growth > 0:
        out.append(f"share count grows faster than {s.share_growth:.1%} a year "
                   f"— rights issues are what made a frozen share count "
                   f"look-ahead in H57, and they hit the holder the same way")
    if not out:
        out.append("no assumption was supplied, so nothing can falsify it — "
                   "which is itself the answer")
    return out
