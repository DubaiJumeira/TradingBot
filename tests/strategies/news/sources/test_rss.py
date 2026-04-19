"""
Offline test for the GenericRSSSource parser.

Uses feedparser against a local XML fixture (feedparser handles filesystem
paths directly) so no network is touched.
"""

from __future__ import annotations

from datetime import timezone

import feedparser

from strategies.news.sources.rss import FeedSpec, GenericRSSSource


class TestParsing:
    def test_parses_local_rss_fixture(self, fixtures_dir):
        rss_path = fixtures_dir / "rss_sample.xml"
        parsed = feedparser.parse(str(rss_path))
        feed_spec = FeedSpec(name="test_wire", url="file://" + str(rss_path), credibility=0.9)
        source = GenericRSSSource(feeds=(feed_spec,))

        items = source._parse_entries(feed_spec, parsed.entries)

        assert len(items) == 3
        titles = [i.title for i in items]
        assert "Breaking: Trump announces new 25% tariff on Chinese imports" in titles
        assert "Powell: Fed will remain data-dependent on rate decisions" in titles
        assert "Gold hits record high as Treasury yields decline" in titles

    def test_timestamps_are_tz_aware(self, fixtures_dir):
        rss_path = fixtures_dir / "rss_sample.xml"
        parsed = feedparser.parse(str(rss_path))
        feed_spec = FeedSpec(name="test_wire", url=str(rss_path), credibility=0.9)
        items = GenericRSSSource(feeds=(feed_spec,))._parse_entries(feed_spec, parsed.entries)
        for item in items:
            assert item.published_at.tzinfo is not None
            assert item.published_at.utcoffset().total_seconds() == 0

    def test_credibility_taken_from_feed_spec(self, fixtures_dir):
        rss_path = fixtures_dir / "rss_sample.xml"
        parsed = feedparser.parse(str(rss_path))
        feed_spec = FeedSpec(name="test_wire", url=str(rss_path), credibility=0.73)
        items = GenericRSSSource(feeds=(feed_spec,))._parse_entries(feed_spec, parsed.entries)
        for item in items:
            assert item.source_credibility == 0.73

    def test_source_label_includes_feed_name(self, fixtures_dir):
        rss_path = fixtures_dir / "rss_sample.xml"
        parsed = feedparser.parse(str(rss_path))
        feed_spec = FeedSpec(name="wire_42", url=str(rss_path), credibility=0.9)
        items = GenericRSSSource(feeds=(feed_spec,))._parse_entries(feed_spec, parsed.entries)
        assert all(i.source == "rss:wire_42" for i in items)


class TestConfiguration:
    def test_empty_feeds_is_unconfigured(self):
        assert not GenericRSSSource(feeds=()).is_configured()

    def test_nonempty_feeds_is_configured(self):
        spec = FeedSpec(name="a", url="http://example.com/rss", credibility=0.5)
        assert GenericRSSSource(feeds=(spec,)).is_configured()
