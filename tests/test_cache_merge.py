"""The cache must be ADDITIVE for any source with a rolling window.

THE FAILURE THIS PREVENTS. Yahoo serves intraday bars for only ~730 days, so
the cache is the sole copy of anything older. `Cache.write` replaced, so a
refresh deleted history that cannot be re-fetched — measured 2026-09-06 at
1,092,171 of 3,125,000 hourly bars, 34.94%, already outside the window. A
container rebuild destroyed them before the refresh could, which does not make
this less necessary: the next cache to grow past its source's window dies the
same way unless the store is additive.
"""

from __future__ import annotations

import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

from idxbot.data.cache import Cache                                # noqa: E402


def _bars(start, n, price=100.0, col="date"):
    return pd.DataFrame({
        col: pd.bdate_range(start, periods=n),
        "close": [price + i for i in range(n)],
    })


def test_a_rolling_window_refetch_does_not_delete_older_bars(tmp_path):
    """THE WHOLE POINT. A source that only serves recent data must not be able
    to truncate the archive it is being merged into."""
    c = Cache(str(tmp_path))
    c.write("intraday", "X", _bars("2020-01-01", 300), merge_on="date")
    #  A later fetch whose window starts well after the first one's.
    c.write("intraday", "X", _bars("2021-01-01", 300), merge_on="date")
    got = c.read("intraday", "X")
    assert got["date"].min() == pd.Timestamp("2020-01-01"), \
        "the pre-window history was deleted"
    assert got["date"].max() == pd.bdate_range("2021-01-01", periods=300)[-1]
    assert got["date"].is_monotonic_increasing
    assert not got["date"].duplicated().any()


def test_new_wins_on_overlap_so_a_correction_can_propagate(tmp_path):
    """The opposite failure, and the quieter one: if OLD won, a vendor's
    corrected bar could never reach the cache and it would fossilise its first
    answer forever."""
    c = Cache(str(tmp_path))
    c.write("ohlcv", "X", _bars("2020-01-01", 10, price=100.0), merge_on="date")
    c.write("ohlcv", "X", _bars("2020-01-01", 10, price=999.0), merge_on="date")
    got = c.read("ohlcv", "X").sort_values("date")
    assert got["close"].iloc[0] == 999.0, "the correction did not propagate"
    assert len(got) == 10, "overlap was double-counted"


def test_without_merge_on_the_behaviour_is_unchanged(tmp_path):
    """Merge is OPT-IN. A namespace whose rows are not keyed by a stable
    identifier has no correct merge, and unioning those silently would be worse
    than replacing them."""
    c = Cache(str(tmp_path))
    c.write("other", "X", _bars("2020-01-01", 50))
    c.write("other", "X", _bars("2021-01-01", 10))
    assert len(c.read("other", "X")) == 10


def test_a_string_date_read_back_from_csv_does_not_double_the_rows(tmp_path):
    """The merge round-trips through gzipped CSV, so the key comes back as a
    STRING while the incoming frame holds datetime64. Deduping across those two
    dtypes silently keeps both copies of every overlapping row — the bug would
    show as a cache that grows on every refresh and never as an error."""
    c = Cache(str(tmp_path))
    first = _bars("2020-01-01", 20)
    c.write("intraday", "X", first, merge_on="date")
    #  Same rows, but keyed as plain strings, which is what a caller reading
    #  its own CSV back would hand over.
    again = first.copy()
    again["date"] = again["date"].dt.strftime("%Y-%m-%d")
    c.write("intraday", "X", again, merge_on="date")
    assert len(c.read("intraday", "X")) == 20


def test_the_two_rolling_window_callers_actually_opt_in():
    """A merge nobody calls is decoration. These are the two namespaces whose
    source has a window: intraday (~730d) and ohlcv (full history, but a
    truncated response must not be able to shorten the cache)."""
    import inspect
    from idxbot.data import intraday, ohlcv
    assert 'merge_on="ts"' in inspect.getsource(intraday)
    assert 'merge_on="date"' in inspect.getsource(ohlcv)


def test_an_empty_frame_never_wipes_an_existing_cache(tmp_path):
    """A failed fetch returns nothing. It must not be able to erase the archive
    on its way out."""
    c = Cache(str(tmp_path))
    c.write("intraday", "X", _bars("2020-01-01", 30), merge_on="date")
    c.write("intraday", "X", pd.DataFrame(), merge_on="date")
    assert len(c.read("intraday", "X")) == 30
