"""
TTL cache for news source responses.

Two layers:
    - In-memory `cachetools.TTLCache` for fast hot-path lookups.
    - Optional disk persistence (JSON under NEWS_CACHE_DIR) so rate limits
      survive bot restarts. If a bot crashes mid-cycle and restarts 30 seconds
      later, we don't want to re-hit NewsAPI's 100-req/day quota.

TRADING LOGIC NOTE
------------------
Cache TTL is per-source; the default 5 minutes matches the main bot cycle so
one call per cycle is the worst case. News-reactive mode (Phase 1D) bypasses
this cache using `force_refresh=True` because event-driven trading needs
sub-minute freshness.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any

from cachetools import TTLCache

from .types import NewsItem

logger = logging.getLogger(__name__)

_MAX_MEMORY_ITEMS = 256
_DEFAULT_TTL = 300


class NewsCache:
    """
    Thread-safe TTL cache with optional disk persistence.

    Keys are arbitrary strings (the sources use `source:<name>`). Values are
    `list[NewsItem]`. The disk layer is opt-in via env `NEWS_CACHE_DIR`; if
    unset, the cache is memory-only.
    """

    def __init__(self, cache_dir: str | None = None, max_size: int = _MAX_MEMORY_ITEMS) -> None:
        self._lock = threading.RLock()
        self._memory: TTLCache[str, tuple[float, list[NewsItem]]] = TTLCache(
            maxsize=max_size, ttl=_DEFAULT_TTL
        )
        resolved = cache_dir if cache_dir is not None else os.getenv("NEWS_CACHE_DIR", "")
        self._disk_dir: Path | None = Path(resolved) if resolved else None
        if self._disk_dir is not None:
            try:
                self._disk_dir.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                logger.warning("Could not create news cache dir %s: %s — disk layer disabled.",
                               self._disk_dir, exc)
                self._disk_dir = None

    # -------------------------------------------------------------- memory

    def get(self, key: str) -> list[NewsItem] | None:
        """Return cached items for `key`, or None if miss/expired."""
        with self._lock:
            entry = self._memory.get(key)
            if entry is not None:
                expires_at, items = entry
                if expires_at > time.time():
                    return list(items)  # defensive copy
                # expired — fall through to disk
            disk_items = self._read_disk(key)
            if disk_items is not None:
                # re-warm memory so next call is fast
                self._memory[key] = (time.time() + _DEFAULT_TTL, disk_items)
                return disk_items
            return None

    def set(self, key: str, items: list[NewsItem], ttl: int = _DEFAULT_TTL) -> None:
        """Store `items` under `key` with TTL seconds."""
        with self._lock:
            expires_at = time.time() + ttl
            self._memory[key] = (expires_at, list(items))
            self._write_disk(key, items, expires_at)

    def invalidate(self, key: str) -> None:
        """Remove `key` from both layers. Used by Phase 1D force_refresh."""
        with self._lock:
            self._memory.pop(key, None)
            if self._disk_dir is not None:
                path = self._disk_path(key)
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    pass

    def clear(self) -> None:
        """Wipe everything. Primarily for tests."""
        with self._lock:
            self._memory.clear()
            if self._disk_dir is not None and self._disk_dir.exists():
                for p in self._disk_dir.glob("*.json"):
                    try:
                        p.unlink()
                    except OSError:
                        pass

    # ---------------------------------------------------------------- disk

    def _disk_path(self, key: str) -> Path:
        assert self._disk_dir is not None
        safe = key.replace("/", "_").replace(":", "_")
        return self._disk_dir / f"{safe}.json"

    def _read_disk(self, key: str) -> list[NewsItem] | None:
        if self._disk_dir is None:
            return None
        path = self._disk_path(key)
        if not path.exists():
            return None
        try:
            with path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Could not read news cache file %s: %s", path, exc)
            return None
        expires_at = payload.get("expires_at", 0)
        if expires_at <= time.time():
            return None
        try:
            return [NewsItem.from_dict(d) for d in payload.get("items", [])]
        except (KeyError, ValueError, TypeError) as exc:
            logger.warning("Corrupt news cache file %s: %s", path, exc)
            return None

    def _write_disk(self, key: str, items: list[NewsItem], expires_at: float) -> None:
        if self._disk_dir is None:
            return
        path = self._disk_path(key)
        payload: dict[str, Any] = {
            "expires_at": expires_at,
            "items": [item.to_dict() for item in items],
        }
        try:
            # Atomic write: temp file + rename, so a crash mid-write doesn't
            # leave a corrupted cache file that breaks the next startup.
            tmp = path.with_suffix(".json.tmp")
            with tmp.open("w", encoding="utf-8") as f:
                json.dump(payload, f)
            tmp.replace(path)
        except OSError as exc:
            logger.warning("Could not write news cache file %s: %s", path, exc)
