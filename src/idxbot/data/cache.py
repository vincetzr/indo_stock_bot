"""On-disk cache for market data.

Gzipped CSV rather than parquet so the package has no pyarrow dependency; at
IDX scale (a few thousand daily bars per ticker) the difference is immaterial
and CSV stays inspectable with ordinary tools.
"""

from __future__ import annotations

import os
import time
import uuid
from typing import Optional

import pandas as pd


class Cache:
    def __init__(self, root: str):
        self.root = root
        os.makedirs(root, exist_ok=True)

    def _path(self, namespace: str, key: str) -> str:
        directory = os.path.join(self.root, namespace)
        os.makedirs(directory, exist_ok=True)
        safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in key)
        return os.path.join(directory, f"{safe}.csv.gz")

    def age_seconds(self, namespace: str, key: str) -> Optional[float]:
        path = self._path(namespace, key)
        if not os.path.exists(path):
            return None
        return time.time() - os.path.getmtime(path)

    def read(
        self,
        namespace: str,
        key: str,
        max_age: Optional[float] = None,
        parse_dates: Optional[list] = None,
    ) -> Optional[pd.DataFrame]:
        path = self._path(namespace, key)
        if not os.path.exists(path):
            return None
        if max_age is not None:
            age = self.age_seconds(namespace, key)
            if age is None or age > max_age:
                return None
        try:
            return pd.read_csv(path, parse_dates=parse_dates or ["date"])
        except Exception:
            # A corrupt or truncated cache entry must never break a run.
            return None

    def write(self, namespace: str, key: str, df: pd.DataFrame,
              merge_on: Optional[str] = None) -> None:
        """Persist ``df``. With ``merge_on``, UNION with what is already there.

        WHY MERGE EXISTS, AND IT IS NOT A NICETY.
        A plain replace turns any cache whose SOURCE HAS A ROLLING WINDOW into a
        lossy one. Yahoo serves intraday bars for only ~730 days, so the cache
        was the sole copy of everything older — and a refresh would have deleted
        it. Measured 2026-09-06 before the loss: 1,092,171 of 3,125,000 hourly
        bars, 34.94%, were already outside Yahoo's window and unre-fetchable. A
        container rebuild destroyed them first, which does not make this less
        necessary: the next cache to grow past its source's window dies the same
        way unless the store is additive.

        THE PRECEDENCE RULE IS "NEW WINS ON OVERLAP, OLD SURVIVES OUTSIDE IT."
        New must win, or a vendor's corrected bar could never propagate and the
        cache would fossilise its first answer — the opposite failure, and a
        quieter one. Old must survive, or the window problem above returns.

        This is deliberately OPT-IN per call site. A namespace whose rows are
        not keyed by a stable identifier has no correct merge, and silently
        unioning those would be worse than replacing them.
        """
        if df is None or df.empty:
            return
        path = self._path(namespace, key)
        if merge_on and merge_on in df.columns and os.path.exists(path):
            old = self.read(namespace, key, parse_dates=[merge_on])
            if old is not None and not old.empty and merge_on in old.columns:
                #  Align dtypes before concat: a datetime key read back from CSV
                #  against an in-memory one that is already datetime64 would
                #  otherwise dedupe as two distinct values and silently double
                #  every overlapping row.
                try:
                    old[merge_on] = pd.to_datetime(old[merge_on])
                    new = df.copy()
                    new[merge_on] = pd.to_datetime(new[merge_on])
                except Exception:
                    old, new = old, df.copy()
                #  `keep="last"` with new concatenated LAST is what makes new win.
                merged = pd.concat([old, new], ignore_index=True)
                merged = (merged.drop_duplicates(subset=[merge_on], keep="last")
                          .sort_values(merge_on).reset_index(drop=True))
                df = merged
        # Unique temp name per writer. A fixed "<path>.tmp" collides when two
        # processes cache the same ticker at once: the first one's rename
        # consumes the file the second is about to rename, and that one dies
        # with FileNotFoundError. Surfaced when scaling to the full 838-name
        # universe with a background fetch already running.
        tmp = f"{path}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp"
        try:
            df.to_csv(tmp, index=False, compression="gzip")
            os.replace(tmp, path)  # atomic; a kill cannot leave a partial file
        except Exception:
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except OSError:
                    pass
            raise

    def clear(self, namespace: Optional[str] = None) -> int:
        target = self.root if namespace is None else os.path.join(self.root, namespace)
        removed = 0
        for dirpath, _dirnames, filenames in os.walk(target):
            for name in filenames:
                if name.endswith(".csv.gz"):
                    os.remove(os.path.join(dirpath, name))
                    removed += 1
        return removed
