"""
Tests for NewsAggregator — parallel fetch, error isolation, dedup, dry-run logging.

Uses fake NewsSource subclasses so nothing touches the network.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import ClassVar

from strategies.news.aggregator import NewsAggregator
from strategies.news.cache import NewsCache
from strategies.news.sources.base import NewsSource
from strategies.news.types import ImpactLevel, NewsItem, SourceKind


class _FakeSource(NewsSource):
    """Test double that returns a fixed list of items."""

    kind: ClassVar[SourceKind] = SourceKind.RSS

    def __init__(self, name: str, items: list[NewsItem], configured: bool = True,
                 raise_on_fetch: bool = False, **kwargs) -> None:
        super().__init__(**kwargs)
        self._name = name
        self._items = items
        self._configured = configured
        self._raise = raise_on_fetch

    @property
    def name(self) -> str:
        return self._name

    def is_configured(self) -> bool:
        return self._configured

    def _fetch_raw(self) -> list[NewsItem]:
        if self._raise:
            raise RuntimeError(f"intentional failure in {self._name}")
        return list(self._items)


def _item(source: str, title: str, credibility: float = 0.8,
          impact: ImpactLevel = ImpactLevel.LOW, minutes_ago: int = 0) -> NewsItem:
    return NewsItem(
        source=source,
        title=title,
        content=title,
        published_at=datetime(2026, 4, 8, 12, 0 - minutes_ago, 0, tzinfo=timezone.utc),
        source_credibility=credibility,
        impact_level=impact,
    )


class TestParallelFetch:
    def test_collects_from_all_sources(self, tmp_path):
        cache = NewsCache(cache_dir=str(tmp_path))
        s1 = _FakeSource("s1", [_item("s1", "A"), _item("s1", "B")], cache=cache)
        s2 = _FakeSource("s2", [_item("s2", "C")], cache=cache)
        agg = NewsAggregator(sources=[s1, s2], cache=cache, dry_run=False)

        result = agg.fetch_all()
        titles = {item.title for item in result}
        assert titles == {"A", "B", "C"}


class TestErrorIsolation:
    def test_one_broken_source_does_not_kill_others(self, tmp_path):
        cache = NewsCache(cache_dir=str(tmp_path))
        good = _FakeSource("good", [_item("good", "Real news")], cache=cache)
        bad = _FakeSource("bad", [], raise_on_fetch=True, cache=cache)
        agg = NewsAggregator(sources=[good, bad], cache=cache, dry_run=False)

        result = agg.fetch_all()
        titles = [item.title for item in result]
        assert "Real news" in titles

    def test_unconfigured_source_returns_empty_no_crash(self, tmp_path):
        cache = NewsCache(cache_dir=str(tmp_path))
        configured = _FakeSource("yes", [_item("yes", "A")], cache=cache)
        missing = _FakeSource("no", [_item("no", "X")], configured=False, cache=cache)
        agg = NewsAggregator(sources=[configured, missing], cache=cache, dry_run=False)

        result = agg.fetch_all()
        titles = [item.title for item in result]
        assert titles == ["A"]


class TestDedupAcrossSources:
    def test_same_story_from_multiple_sources_merges(self, tmp_path):
        cache = NewsCache(cache_dir=str(tmp_path))
        reuters = _FakeSource("reuters",
                              [_item("rss:reuters", "Fed cuts rates 25bps", credibility=1.0)],
                              cache=cache)
        cnbc = _FakeSource("cnbc",
                           [_item("rss:cnbc", "Fed cuts rates 25bps", credibility=0.9)],
                           cache=cache)
        agg = NewsAggregator(sources=[reuters, cnbc], cache=cache, dry_run=False)

        result = agg.fetch_all()
        assert len(result) == 1
        assert result[0].source.startswith("merged:")
        assert result[0].source_credibility == 1.0


class TestCaching:
    def test_second_call_uses_cache(self, tmp_path):
        cache = NewsCache(cache_dir=str(tmp_path))
        items = [_item("s1", "Cached")]
        source = _FakeSource("s1", items, cache=cache)
        agg = NewsAggregator(sources=[source], cache=cache, dry_run=False)

        agg.fetch_all()
        # Swap in a new item list; if cache works, we still see the old one.
        source._items = [_item("s1", "Different")]
        result = agg.fetch_all()
        assert len(result) == 1
        assert result[0].title == "Cached"

    def test_force_refresh_bypasses_cache(self, tmp_path):
        cache = NewsCache(cache_dir=str(tmp_path))
        items = [_item("s1", "Original")]
        source = _FakeSource("s1", items, cache=cache)
        agg = NewsAggregator(sources=[source], cache=cache, dry_run=False)

        agg.fetch_all()
        source._items = [_item("s1", "Fresh")]
        result = agg.fetch_all(force_refresh=True)
        titles = {i.title for i in result}
        assert "Fresh" in titles


class TestHighImpactFilter:
    def test_filters_below_threshold(self, tmp_path):
        cache = NewsCache(cache_dir=str(tmp_path))
        source = _FakeSource("s1", [
            _item("s1", "Low", impact=ImpactLevel.LOW),
            _item("s1", "High", impact=ImpactLevel.HIGH),
            _item("s1", "Critical", impact=ImpactLevel.CRITICAL),
        ], cache=cache)
        agg = NewsAggregator(sources=[source], cache=cache, dry_run=False)

        agg.fetch_all()
        # Use a very old `since` so the fixed 2026-04-08 timestamps fall within the window.
        ancient = datetime(2000, 1, 1, tzinfo=timezone.utc)
        high = agg.high_impact(minimum=ImpactLevel.HIGH, since=ancient)
        titles = {i.title for i in high}
        assert titles == {"High", "Critical"}


class TestDryRunMode:
    def test_dry_run_logs_items(self, tmp_path, caplog):
        import logging
        cache = NewsCache(cache_dir=str(tmp_path))
        source = _FakeSource("s1", [_item("s1", "Dry run test item")], cache=cache)
        agg = NewsAggregator(sources=[source], cache=cache, dry_run=True)

        with caplog.at_level(logging.INFO, logger="strategies.news.aggregator"):
            agg.fetch_all()

        dry_run_lines = [r for r in caplog.records if "[DRY RUN]" in r.getMessage()]
        assert dry_run_lines
        assert any("Dry run test item" in r.getMessage() for r in dry_run_lines)
