"""Offline test for Fear & Greed parser."""

from __future__ import annotations

from strategies.news.sources.fear_greed import FearGreedSource
from strategies.news.types import ImpactLevel


class TestParsing:
    def test_parses_fixture(self, load_fixture):
        payload = load_fixture("fear_greed_sample.json")
        items = FearGreedSource()._parse(payload)

        assert len(items) == 1
        item = items[0]
        assert "72" in item.title
        assert "Greed" in item.title
        assert item.raw_data["value"] == 72
        assert item.impact_level == ImpactLevel.LOW

    def test_empty_data_returns_empty(self):
        items = FearGreedSource()._parse({"data": []})
        assert items == []

    def test_always_low_impact_even_with_urgency_words(self):
        source = FearGreedSource()
        items = source._parse({"data": [{
            "value": "10",
            "value_classification": "Extreme Fear",
            "timestamp": "1775000000",
        }]})
        item = items[0]
        assert source._classify_impact(item) == ImpactLevel.LOW
