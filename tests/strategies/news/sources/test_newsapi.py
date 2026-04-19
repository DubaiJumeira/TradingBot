"""Offline test for NewsAPI parsing using a fixture payload."""

from __future__ import annotations

from strategies.news.sources.newsapi import NewsAPISource


class TestParsing:
    def test_parses_fixture(self, load_fixture):
        payload = load_fixture("newsapi_sample.json")
        source = NewsAPISource(api_key="fake-key-for-test")
        items = source._parse(payload)

        # Four entries in fixture, one is "[Removed]" → filtered.
        assert len(items) == 3
        titles = [i.title for i in items]
        assert "Fed signals rate cut possibility at next meeting, minutes show" in titles
        assert "[Removed]" not in titles

    def test_content_falls_back_through_description(self, load_fixture):
        payload = load_fixture("newsapi_sample.json")
        items = NewsAPISource(api_key="fake-key-for-test")._parse(payload)
        fed_item = next(i for i in items if "Fed signals" in i.title)
        assert "Federal Reserve officials discussed" in fed_item.content

    def test_source_name_prefixed(self, load_fixture):
        payload = load_fixture("newsapi_sample.json")
        items = NewsAPISource(api_key="fake-key-for-test")._parse(payload)
        assert any(i.source == "newsapi:Reuters" for i in items)
        assert any(i.source == "newsapi:Bloomberg" for i in items)


class TestConfiguration:
    def test_requires_api_key(self):
        assert not NewsAPISource(api_key="").is_configured()
        assert NewsAPISource(api_key="something").is_configured()
