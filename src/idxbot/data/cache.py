"""On-disk cache for market data.

Gzipped CSV rather than parquet so the package has no pyarrow dependency; at
IDX scale (a few thousand daily bars per ticker) the difference is immaterial
and CSV stays inspectable with ordinary tools.
"""

from __future__ import annotations

import os
import time
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

    def write(self, namespace: str, key: str, df: pd.DataFrame) -> None:
        if df is None or df.empty:
            return
        path = self._path(namespace, key)
        tmp = path + ".tmp"
        df.to_csv(tmp, index=False, compression="gzip")
        os.replace(tmp, path)  # atomic; a killed process cannot leave a partial file

    def clear(self, namespace: Optional[str] = None) -> int:
        target = self.root if namespace is None else os.path.join(self.root, namespace)
        removed = 0
        for dirpath, _dirnames, filenames in os.walk(target):
            for name in filenames:
                if name.endswith(".csv.gz"):
                    os.remove(os.path.join(dirpath, name))
                    removed += 1
        return removed
