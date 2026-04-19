"""Tests for the ForexLive-specific impact classification override."""

from __future__ import annotations

from datetime import datetime, timezone

from strategies.news.sources.forexlive_rss import ForexLiveSource
from strategies.news.types import ImpactLevel, NewsItem


class TestImpactClassification:
    def test_quiet_headline_is_still_medium(self):
        source = ForexLiveSource()
        item = NewsItem(
            source="forexlive",
            title="EUR/USD drifts sideways in quiet European morning",
            content="...",
            published_at=datetime(2026, 4, 8, 8, 0, 0, tzinfo=timezone.utc),
            source_credibility=source.default_credibility,
        )
        assert source._classify_impact(item) == ImpactLevel.MEDIUM

    def test_urgency_keyword_bumps_to_high(self):
        source = ForexLiveSource()
        item = NewsItem(
            source="forexlive",
            title="BREAKING: Fed cuts rates by 50bps",
            content="...",
            published_at=datetime(2026, 4, 8, 14, 0, 0, tzinfo=timezone.utc),
            source_credibility=source.default_credibility,
        )
        assert source._classify_impact(item) == ImpactLevel.HIGH

    def test_credibility_is_macro_grade(self):
        assert ForexLiveSource.default_credibility >= 0.9
