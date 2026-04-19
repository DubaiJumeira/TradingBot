"""Tests for the core NewsItem dataclass and helpers."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from strategies.news.types import (
    ImpactLevel,
    NewsItem,
    as_utc,
    coarse_impact,
    sort_by_time,
)


class TestNewsItem:
    def test_naive_datetime_is_coerced_to_utc(self):
        naive = datetime(2026, 4, 8, 12, 0, 0)  # no tz
        item = NewsItem(source="x", title="t", content="c", published_at=naive)
        assert item.published_at.tzinfo is timezone.utc

    def test_sentiment_score_is_clamped(self):
        item = NewsItem(source="x", title="t", content="c",
                        published_at=datetime.now(tz=timezone.utc),
                        sentiment_score=5.0)
        assert item.sentiment_score == 1.0
        item2 = NewsItem(source="x", title="t", content="c",
                         published_at=datetime.now(tz=timezone.utc),
                         sentiment_score=-3.0)
        assert item2.sentiment_score == -1.0

    def test_credibility_is_clamped(self):
        item = NewsItem(source="x", title="t", content="c",
                        published_at=datetime.now(tz=timezone.utc),
                        source_credibility=1.5)
        assert item.source_credibility == 1.0

    def test_round_trip_dict(self):
        original = NewsItem(
            source="twitter:@POTUS",
            title="Test headline",
            content="Body",
            published_at=datetime(2026, 4, 8, 12, 0, 0, tzinfo=timezone.utc),
            sentiment_score=0.5,
            impact_level=ImpactLevel.HIGH,
            affected_assets=["XAUUSD", "BTC/USDT"],
            source_credibility=0.9,
            url="https://example.com",
            raw_data={"k": "v"},
        )
        revived = NewsItem.from_dict(original.to_dict())
        assert revived.source == original.source
        assert revived.title == original.title
        assert revived.published_at == original.published_at
        assert revived.impact_level == original.impact_level
        assert revived.affected_assets == original.affected_assets
        assert revived.raw_data == original.raw_data


class TestImpactLevel:
    def test_ordering(self):
        assert ImpactLevel.CRITICAL > ImpactLevel.HIGH
        assert ImpactLevel.HIGH > ImpactLevel.MEDIUM
        assert ImpactLevel.MEDIUM > ImpactLevel.LOW
        assert ImpactLevel.HIGH >= ImpactLevel.HIGH

    def test_rank_values(self):
        assert ImpactLevel.LOW.rank == 0
        assert ImpactLevel.CRITICAL.rank == 3


class TestCoarseImpact:
    def test_urgency_keyword_with_high_credibility_bumps_to_high(self):
        assert coarse_impact("BREAKING: Fed cuts rates", 0.9) == ImpactLevel.HIGH

    def test_urgency_keyword_with_low_credibility_stays_low(self):
        assert coarse_impact("BREAKING: something", 0.3) == ImpactLevel.LOW

    def test_top_credibility_floor_is_medium(self):
        assert coarse_impact("Quiet headline", 1.0) == ImpactLevel.MEDIUM

    def test_default_is_low(self):
        assert coarse_impact("Nothing special", 0.5) == ImpactLevel.LOW


class TestAsUtc:
    def test_accepts_iso_with_z(self):
        result = as_utc("2026-04-08T12:00:00Z")
        assert result == datetime(2026, 4, 8, 12, 0, 0, tzinfo=timezone.utc)

    def test_accepts_epoch(self):
        result = as_utc(1_775_000_000)
        assert result.tzinfo is timezone.utc

    def test_naive_datetime_assumed_utc(self):
        result = as_utc(datetime(2026, 4, 8, 12, 0, 0))
        assert result.tzinfo is timezone.utc

    def test_rejects_unknown_type(self):
        with pytest.raises(TypeError):
            as_utc([1, 2, 3])  # type: ignore[arg-type]


class TestSortByTime:
    def test_newest_first(self):
        t1 = datetime(2026, 4, 8, 10, 0, 0, tzinfo=timezone.utc)
        t2 = datetime(2026, 4, 8, 12, 0, 0, tzinfo=timezone.utc)
        t3 = datetime(2026, 4, 8, 11, 0, 0, tzinfo=timezone.utc)
        items = [
            NewsItem(source="a", title="A", content="", published_at=t1),
            NewsItem(source="b", title="B", content="", published_at=t2),
            NewsItem(source="c", title="C", content="", published_at=t3),
        ]
        ordered = sort_by_time(items)
        assert [i.title for i in ordered] == ["B", "C", "A"]
