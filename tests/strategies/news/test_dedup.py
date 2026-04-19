"""Tests for the dedup module — exact hash + fuzzy title matching + merge rules."""

from __future__ import annotations

from datetime import datetime, timezone

from strategies.news.dedup import deduplicate
from strategies.news.types import ImpactLevel, NewsItem


def _item(source: str, title: str, *, published_at: datetime | None = None,
          credibility: float = 0.5, impact: ImpactLevel = ImpactLevel.LOW,
          url: str = "", content: str = "") -> NewsItem:
    return NewsItem(
        source=source,
        title=title,
        content=content or title,
        published_at=published_at or datetime(2026, 4, 8, 12, 0, 0, tzinfo=timezone.utc),
        source_credibility=credibility,
        impact_level=impact,
        url=url or f"https://{source}.test/{abs(hash(title)) % 10_000}",
    )


class TestExactDedup:
    def test_identical_titles_collapse_to_one(self):
        items = [
            _item("rss:reuters", "Fed cuts rates by 25bps", credibility=1.0),
            _item("rss:cnbc", "Fed cuts rates by 25bps", credibility=0.9),
        ]
        result = deduplicate(items)
        assert len(result) == 1
        assert result[0].source.startswith("merged:")
        # Max credibility wins
        assert result[0].source_credibility == 1.0

    def test_punctuation_and_case_differences_still_dedupe(self):
        items = [
            _item("a", "Fed cuts rates by 25bps!"),
            _item("b", "fed cuts rates by 25bps"),
            _item("c", "FED CUTS RATES BY 25BPS."),
        ]
        result = deduplicate(items)
        assert len(result) == 1


class TestFuzzyDedup:
    def test_rephrased_titles_collapse(self):
        # Minor wording differences should collapse. This pair scores ~91 on
        # rapidfuzz.token_set_ratio, safely above the 90 threshold.
        items = [
            _item("rss:reuters", "Oil jumps 3% on OPEC cut talks", credibility=1.0),
            _item("rss:cnbc", "Oil prices jump 3% on OPEC cut talks", credibility=0.9),
        ]
        result = deduplicate(items)
        assert len(result) == 1

    def test_unrelated_headlines_stay_separate(self):
        items = [
            _item("a", "Fed cuts rates by 25bps"),
            _item("b", "Oil prices surge 3% on OPEC cut talks"),
            _item("c", "Bitcoin ETF sees record inflows"),
        ]
        result = deduplicate(items)
        assert len(result) == 3

    def test_opposite_meaning_headlines_do_not_collapse(self):
        # CRITICAL invariant: "Gold hits record high" and "Gold declines from
        # record high" share most tokens (token_set_ratio ≈ 88) but mean
        # OPPOSITE things. If these were merged, Phase 1B's sentiment pipeline
        # would produce garbage. The threshold of 90 exists to protect against
        # this exact class of false positive.
        items = [
            _item("rss:reuters", "Gold hits record high as yields decline"),
            _item("rss:cnbc", "Gold declines from record high as yields rise"),
        ]
        result = deduplicate(items)
        assert len(result) == 2

    def test_approve_vs_reject_do_not_collapse(self):
        items = [
            _item("rss:reuters", "Bitcoin ETF approved by SEC"),
            _item("rss:cnbc", "Bitcoin ETF rejected by SEC"),
        ]
        result = deduplicate(items)
        assert len(result) == 2


class TestMergeRules:
    def test_earliest_timestamp_wins(self):
        early = datetime(2026, 4, 8, 12, 0, 0, tzinfo=timezone.utc)
        late = datetime(2026, 4, 8, 12, 5, 0, tzinfo=timezone.utc)
        items = [
            _item("rss:cnbc", "Fed cuts rates", published_at=late, credibility=0.9),
            _item("rss:reuters", "Fed cuts rates", published_at=early, credibility=1.0),
        ]
        result = deduplicate(items)
        assert len(result) == 1
        assert result[0].published_at == early

    def test_max_impact_wins(self):
        items = [
            _item("a", "Same title", impact=ImpactLevel.LOW),
            _item("b", "Same title", impact=ImpactLevel.HIGH),
            _item("c", "Same title", impact=ImpactLevel.MEDIUM),
        ]
        result = deduplicate(items)
        assert len(result) == 1
        assert result[0].impact_level == ImpactLevel.HIGH

    def test_merged_raw_data_records_all_sources(self):
        items = [
            _item("rss:reuters", "Fed cuts rates", credibility=1.0,
                  url="https://reuters.com/1"),
            _item("rss:cnbc", "Fed cuts rates", credibility=0.9,
                  url="https://cnbc.com/1"),
        ]
        result = deduplicate(items)
        merged_from = result[0].raw_data.get("merged_from", [])
        assert len(merged_from) == 2
        urls = {m["url"] for m in merged_from}
        assert urls == {"https://reuters.com/1", "https://cnbc.com/1"}


class TestEmptyInput:
    def test_empty_list_returns_empty(self):
        assert deduplicate([]) == []
