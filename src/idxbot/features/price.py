"""§8 price/TA and structural features, as numeric cross-sectional inputs.

WHY THESE ARE FUNCTIONS OF PAST BARS ONLY, AND NOTHING ELSE
-------------------------------------------------------------
Every feature here is computed from bars up to and including day *t*, and every
label is a return starting at *t+1*. §8 is explicit that these must be numeric
features feeding a cross-sectional model rather than discrete buy/sell rules
with hand-tuned parameters, so there are no thresholds, no crossovers and no
tuned lookbacks: the lookbacks are the conventional ones from the literature
each mechanism comes from, fixed before any of this was run and logged in
`hypotheses.md` under H13.

THE ONE-DAY EXECUTION GAP
--------------------------
A feature known at the close of *t* is labelled with the return from the close
of *t+1*. That throws away the t -> t+1 move, which for a reversal signal is a
real part of the effect, and it is thrown away on purpose: it is the same
convention `flow_panel_build` uses, it removes any chance of same-bar
contamination, and a signal that only works if you can trade the instant you
compute it is not a signal this project can use (§3: execution is manual).

WHAT THE ADJUSTED SERIES COSTS US
-----------------------------------
Returns use ``adj_close`` so splits and dividends do not read as crashes. But
A2 established that **22.3% of the spine provably sits on a vendor-adjusted
basis** — a lower bound, since a whole-number factor is invisible to the
off-grid test. On those names ``close`` is not the traded price, so
``reference.half_spread`` looks the tick up from the wrong band and understates
the spread. That is a known, one-sided error and it makes every cost figure
here optimistic rather than conservative.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: Conventional lookbacks, fixed before testing. Not tuned, and not to be.
REV_SHORT = 5
REV_LONG = 21
MOM_LONG = 252
MOM_SKIP = 21
VOL_WIN = 60
AMIHUD_WIN = 60
VOLZ_SHORT = 20
VOLZ_LONG = 250
HI_WIN = 252
ATR_SHORT = 20
ATR_LONG = 250

#: Minimum bars of history before a name enters the cross-section at all.
MIN_HISTORY = 260


def true_range(high, low, close) -> pd.Series:
    """Wilder's true range: the day's range, widened by any overnight gap."""
    h, l = pd.Series(high).astype(float), pd.Series(low).astype(float)
    pc = pd.Series(close).astype(float).shift(1)
    return pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)


def atr(high, low, close, win: int) -> pd.Series:
    return true_range(high, low, close).rolling(win, min_periods=win // 2).mean()


def compute(df: pd.DataFrame) -> pd.DataFrame:
    """All H13 features for one ticker's bar series, past-only.

    ``df`` must be sorted by date and carry close, adj_close, high, low,
    volume. Returns the same index with the feature columns added; a feature is
    NaN wherever its lookback is not yet satisfied, never filled.
    """
    d = df.copy()
    px = pd.to_numeric(d["adj_close"], errors="coerce").astype(float)
    raw = pd.to_numeric(d["close"], errors="coerce").astype(float)
    vol = pd.to_numeric(d["volume"], errors="coerce").astype(float)
    ret1 = px.pct_change()
    value = raw * vol                      # no VWAP available; close x volume

    # ---- controls -------------------------------------------------------
    d["ret1"] = ret1
    d["rev1"] = px / px.shift(REV_LONG) - 1.0
    d["mom12_1"] = px.shift(MOM_SKIP) / px.shift(MOM_LONG) - 1.0
    d["vol60"] = ret1.rolling(VOL_WIN, min_periods=VOL_WIN // 2).std()
    d["log_turnover"] = np.log1p(
        value.rolling(VOL_WIN, min_periods=VOL_WIN // 2).median())

    # ---- the eight tested features --------------------------------------
    # 1. short-horizon reversal: signed so a POSITIVE value is the prediction
    d["rev5"] = -(px / px.shift(REV_SHORT) - 1.0)
    # 2. momentum is d["mom12_1"], already computed as a control
    # 3. low volatility, signed the same way
    d["lowvol"] = -d["vol60"]
    # 4. Amihud illiquidity: |return| per rupiah traded
    illiq = (ret1.abs() / value.replace(0.0, np.nan))
    d["amihud60"] = illiq.rolling(AMIHUD_WIN, min_periods=AMIHUD_WIN // 2).mean()
    # 5. volume z-score: recent turnover against its own long baseline
    v20 = vol.rolling(VOLZ_SHORT, min_periods=VOLZ_SHORT // 2).mean()
    vmu = vol.rolling(VOLZ_LONG, min_periods=VOLZ_LONG // 2).mean()
    vsd = vol.rolling(VOLZ_LONG, min_periods=VOLZ_LONG // 2).std()
    d["volz20"] = (v20 - vmu) / vsd.replace(0.0, np.nan)
    # 6. distance to the 52-week high, as a ratio so it is scale-free
    d["hi52"] = px / px.rolling(HI_WIN, min_periods=HI_WIN // 2).max()
    # 7. volatility-normalised momentum
    a20 = atr(d["high"], d["low"], d["close"], ATR_SHORT)
    d["atr_mom20"] = (px - px.shift(ATR_SHORT)) / (a20.replace(0.0, np.nan))
    # 8. NEGATIVE CONTROL: range compression predicts the SIZE of the next
    #    move, not its sign, so it should show no signed cross-sectional IC.
    a250 = atr(d["high"], d["low"], d["close"], ATR_LONG)
    d["squeeze"] = a20 / a250.replace(0.0, np.nan)

    # ---- infinities are missing data, not extreme values ----------------
    for c in ("rev1", "mom12_1", "vol60", "log_turnover", "rev5", "lowvol",
              "amihud60", "volz20", "hi52", "atr_mom20", "squeeze"):
        d[c] = d[c].replace([np.inf, -np.inf], np.nan)
    return d


def forward_return(adj_close, k: int, gap: int = 1) -> pd.Series:
    """Return over k bars, entered ``gap`` bars after the decision bar.

    With the default gap of 1: the feature is known at the close of t, the
    position is taken at the close of t+1, and this is the return from there to
    the close of t+1+k. Nothing in it is stamped before the decision.
    """
    px = pd.Series(adj_close).astype(float)
    entry = px.shift(-gap)
    exit_ = px.shift(-(gap + k))
    return exit_ / entry - 1.0


#: The features H13 tests, and the sign each was predicted to take BEFORE the
#: test was run. Kept in code so the report cannot quietly re-sign a feature
#: after seeing the result.
PREDICTED = {
    "rev5": +1,
    "mom12_1": +1,
    "lowvol": +1,
    "amihud60": +1,
    "volz20": -1,
    "hi52": +1,
    "atr_mom20": +1,
    "squeeze": 0,        # negative control: no signed prediction
}

FEATURES = tuple(PREDICTED)

#: The control set. A feature that IS a control is dropped from its own
#: controls at test time rather than being regressed on itself.
CONTROLS = ("mom12_1", "rev1", "log_turnover", "vol60")

#: Features that are literally a control, or a monotone transform of one, so
#: neutralising on that control would remove the feature entirely.
SELF_CONTROL = {"mom12_1": "mom12_1", "lowvol": "vol60"}
