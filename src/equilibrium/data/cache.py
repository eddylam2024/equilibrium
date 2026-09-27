"""On-disk cache for downloaded data (CSV/JSON, human-inspectable, no pickle)."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import pandas as pd

log = logging.getLogger(__name__)


def default_cache_dir() -> Path:
    env = os.environ.get("EQUILIBRIUM_CACHE_DIR")
    if env:
        return Path(env).expanduser()
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "equilibrium"


class DataCache:
    def __init__(self, directory: Path | None = None, ttl_hours: float = 12.0,
                 enabled: bool = True) -> None:
        self.directory = directory or default_cache_dir()
        self.ttl_seconds = ttl_hours * 3600
        self.enabled = enabled

    @staticmethod
    def key(*parts: Any) -> str:
        return hashlib.sha256(json.dumps(parts, default=str).encode()).hexdigest()[:24]

    def _path(self, key: str, ext: str) -> Path:
        return self.directory / f"{key}.{ext}"

    def _fresh(self, path: Path, permanent: bool) -> bool:
        if not path.exists():
            return False
        return permanent or (time.time() - path.stat().st_mtime) < self.ttl_seconds

    def get_frame(self, key: str, permanent: bool = False) -> pd.DataFrame | None:
        path = self._path(key, "csv")
        if not self.enabled or not self._fresh(path, permanent):
            return None
        try:
            return pd.read_csv(path, index_col=0, parse_dates=True)
        except (OSError, ValueError) as exc:
            log.warning("Ignoring unreadable cache file %s: %s", path, exc)
            return None

    def put_frame(self, key: str, frame: pd.DataFrame) -> None:
        if not self.enabled:
            return
        self.directory.mkdir(parents=True, exist_ok=True)
        frame.to_csv(self._path(key, "csv"))

    def get_json(self, key: str) -> Any | None:
        path = self._path(key, "json")
        if not self.enabled or not self._fresh(path, permanent=False):
            return None
        try:
            return json.loads(path.read_text())
        except (OSError, ValueError) as exc:
            log.warning("Ignoring unreadable cache file %s: %s", path, exc)
            return None

    def put_json(self, key: str, value: Any) -> None:
        if not self.enabled:
            return
        self.directory.mkdir(parents=True, exist_ok=True)
        self._path(key, "json").write_text(json.dumps(value))

    def clear(self) -> int:
        if not self.directory.exists():
            return 0
        n = 0
        for p in self.directory.glob("*.*"):
            if p.suffix in (".csv", ".json"):
                p.unlink()
                n += 1
        return n
