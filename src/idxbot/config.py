"""Configuration loading for idxbot.

All tunable behaviour lives in ``config/*.yaml``. Nothing in the analytics code
hard-codes a broker code, a threshold or a universe member, so the engine can be
retargeted (different exchange members, different risk appetite) by editing YAML
alone.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Optional

import yaml

# Repository root, resolved from this file's location so the CLI works from any
# working directory.
PKG_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(PKG_DIR, os.pardir, os.pardir))
CONFIG_DIR = os.path.join(REPO_ROOT, "config")


def _read_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


@dataclass(frozen=True)
class Broker:
    """An IDX exchange member."""

    code: str
    name: str
    tier: str = "unknown"
    foreign: bool = False
    confidence: str = "unverified"

    @property
    def is_bulge(self) -> bool:
        return self.tier == "bulge"

    @property
    def is_institutional(self) -> bool:
        return self.tier in ("bulge", "foreign", "local_inst")

    @property
    def is_retail(self) -> bool:
        return self.tier == "retail"


class BrokerRegistry:
    """Lookup for exchange members, tolerant of codes not present in config."""

    def __init__(self, brokers: Dict[str, Broker]):
        self._brokers = brokers

    @classmethod
    def from_yaml(cls, path: str) -> "BrokerRegistry":
        raw = _read_yaml(path).get("brokers", {}) or {}
        brokers = {}
        for code, meta in raw.items():
            code = str(code).upper()
            meta = meta or {}
            brokers[code] = Broker(
                code=code,
                name=meta.get("name", code),
                tier=meta.get("tier", "unknown"),
                foreign=bool(meta.get("foreign", False)),
                confidence=meta.get("confidence", "unverified"),
            )
        return cls(brokers)

    def get(self, code: str) -> Broker:
        """Return the broker, synthesising an ``unknown``-tier entry if unseen.

        A vendor feed may contain codes that post-date this config. Those are
        still ingested and still contribute to totals; they simply do not count
        as institutional or retail flow in the tier-based rules.
        """
        code = str(code).upper()
        if code not in self._brokers:
            return Broker(code=code, name=code, tier="unknown", foreign=False)
        return self._brokers[code]

    def codes(self, tier: Optional[str] = None, foreign: Optional[bool] = None) -> List[str]:
        out = []
        for code, b in self._brokers.items():
            if tier is not None and b.tier != tier:
                continue
            if foreign is not None and b.foreign != foreign:
                continue
            out.append(code)
        return sorted(out)

    @property
    def bulge_codes(self) -> List[str]:
        return self.codes(tier="bulge")

    @property
    def retail_codes(self) -> List[str]:
        return self.codes(tier="retail")

    def institutional_codes(self) -> List[str]:
        return sorted(c for c, b in self._brokers.items() if b.is_institutional)

    def __len__(self) -> int:
        return len(self._brokers)

    def __contains__(self, code: object) -> bool:
        return str(code).upper() in self._brokers


def _flatten_tickers(entries: Iterable[Any]) -> List[str]:
    """Accept both one-per-line and comma-packed YAML list entries.

    ``universe.yaml`` packs ten tickers per line for readability; YAML hands
    that back as a single string, so split on commas here.
    """
    out: List[str] = []
    for entry in entries or []:
        if entry is None:
            continue
        for part in str(entry).split(","):
            part = part.strip().upper()
            if part:
                out.append(part)
    # De-duplicate while preserving order.
    seen = set()
    unique = []
    for t in out:
        if t not in seen:
            seen.add(t)
            unique.append(t)
    return unique


@dataclass
class Config:
    raw: Dict[str, Any] = field(default_factory=dict)
    brokers: BrokerRegistry = field(default=None)  # type: ignore[assignment]
    universes: Dict[str, List[str]] = field(default_factory=dict)
    indices: Dict[str, str] = field(default_factory=dict)
    repo_root: str = REPO_ROOT

    # -- dotted access ------------------------------------------------------
    def get(self, dotted: str, default: Any = None) -> Any:
        """``cfg.get("accumulation.weights.stealth")``."""
        node: Any = self.raw
        for key in dotted.split("."):
            if not isinstance(node, dict) or key not in node:
                return default
            node = node[key]
        return node

    def path(self, dotted: str, default: str = "") -> str:
        """Resolve a config value that names a path, relative to the repo root."""
        value = self.get(dotted, default)
        if not value:
            return ""
        if os.path.isabs(value):
            return value
        return os.path.join(self.repo_root, value)

    def universe(self, name: str) -> List[str]:
        if name == "all":
            merged: List[str] = []
            for tickers in self.universes.values():
                merged.extend(tickers)
            seen = set()
            return [t for t in merged if not (t in seen or seen.add(t))]
        if name not in self.universes:
            raise KeyError(
                f"Unknown universe '{name}'. Available: {sorted(self.universes)} or 'all'."
            )
        return list(self.universes[name])

    @property
    def universe_names(self) -> List[str]:
        return sorted(self.universes)


@lru_cache(maxsize=4)
def load_config(config_dir: Optional[str] = None) -> Config:
    """Load and cache the configuration bundle."""
    config_dir = config_dir or CONFIG_DIR
    raw = _read_yaml(os.path.join(config_dir, "config.yaml"))
    brokers = BrokerRegistry.from_yaml(os.path.join(config_dir, "brokers.yaml"))

    uni_raw = _read_yaml(os.path.join(config_dir, "universe.yaml"))
    universes = {
        name: _flatten_tickers(tickers)
        for name, tickers in (uni_raw.get("universes") or {}).items()
    }
    indices = dict(uni_raw.get("indices") or {})

    repo_root = os.path.abspath(os.path.join(config_dir, os.pardir))
    return Config(raw=raw, brokers=brokers, universes=universes, indices=indices,
                  repo_root=repo_root)
