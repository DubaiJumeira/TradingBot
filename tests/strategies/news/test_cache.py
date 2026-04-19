"""Tests for the TTL cache with disk persistence."""

from __future__ import annotations

import time
from datetime import datetime, timezone

from strategies.news.cache import NewsCache
from strategies.news.types import NewsItem


def _sample_items() -> list[NewsItem]:
    return [
        NewsItem(
            source="test",
            title=f"Item {i}",
            content="body",
            published_at=datetime(2026, 4, 8, 12, i, 0, tzinfo=timezone.utc),
            source_credibility=0.8,
        )
        for i in range(3)
    ]


class TestMemoryCache:
    def test_set_and_get(self, tmp_path):
        cache = NewsCache(cache_dir=str(tmp_path))
        cache.set("k", _sample_items(), ttl=60)
        result = cache.get("k")
        assert result is not None
        assert len(result) == 3

    def test_miss_returns_none(self, tmp_path):
        cache = NewsCache(cache_dir=str(tmp_path))
        assert cache.get("missing") is None

    def test_defensive_copy_on_get(self, tmp_path):
        cache = NewsCache(cache_dir=str(tmp_path))
        cache.set("k", _sample_items(), ttl=60)
        got = cache.get("k")
        assert got is not None
        got.clear()
        # Cache itself is not mutated by caller clearing the returned list.
        again = cache.get("k")
        assert again is not None
        assert len(again) == 3


class TestTtlExpiration:
    def test_memory_entry_expires(self, tmp_path):
        cache = NewsCache(cache_dir=str(tmp_path))
        cache.set("k", _sample_items(), ttl=1)
        time.sleep(1.1)
        # Disk layer still holds it but disk also respects expires_at.
        assert cache.get("k") is None


class TestDiskPersistence:
    def test_disk_survives_new_instance(self, tmp_path):
        cache_a = NewsCache(cache_dir=str(tmp_path))
        cache_a.set("k", _sample_items(), ttl=60)

        cache_b = NewsCache(cache_dir=str(tmp_path))
        result = cache_b.get("k")
        assert result is not None
        assert len(result) == 3

    def test_invalidate_removes_from_both_layers(self, tmp_path):
        cache = NewsCache(cache_dir=str(tmp_path))
        cache.set("k", _sample_items(), ttl=60)
        cache.invalidate("k")
        assert cache.get("k") is None

        # Also confirm disk file is gone.
        remaining = list(tmp_path.glob("*.json"))
        assert remaining == []

    def test_clear_wipes_everything(self, tmp_path):
        cache = NewsCache(cache_dir=str(tmp_path))
        cache.set("a", _sample_items(), ttl=60)
        cache.set("b", _sample_items(), ttl=60)
        cache.clear()
        assert cache.get("a") is None
        assert cache.get("b") is None


class TestMemoryOnlyMode:
    def test_no_disk_dir_means_memory_only(self, tmp_path, monkeypatch):
        # Explicit empty string disables disk layer
        monkeypatch.delenv("NEWS_CACHE_DIR", raising=False)
        cache = NewsCache(cache_dir="")
        cache.set("k", _sample_items(), ttl=60)
        assert cache.get("k") is not None
        # No files should exist on disk
        assert list(tmp_path.iterdir()) == []
