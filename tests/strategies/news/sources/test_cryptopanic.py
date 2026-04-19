"""Offline test for CryptoPanic parsing using a fixture payload."""

from __future__ import annotations

from strategies.news.sources.cryptopanic import CryptoPanicSource
from strategies.news.types import ImpactLevel


class TestParsing:
    def test_parses_fixture(self, load_fixture):
        payload = load_fixture("cryptopanic_sample.json")
        source = CryptoPanicSource()
        items = source._parse(payload)

        # Three entries in fixture, one with empty title → filtered.
        assert len(items) == 2
        titles = [i.title for i in items]
        assert "BREAKING: SEC approves spot Ethereum ETF applications" in titles

    def test_credibility_is_set(self, load_fixture):
        payload = load_fixture("cryptopanic_sample.json")
        items = CryptoPanicSource()._parse(payload)
        for item in items:
            assert item.source_credibility == CryptoPanicSource.default_credibility

    def test_raw_data_captures_currencies_and_votes(self, load_fixture):
        payload = load_fixture("cryptopanic_sample.json")
        items = CryptoPanicSource()._parse(payload)
        eth_item = next(i for i in items if "ETF" in i.title)
        assert "ETH" in eth_item.raw_data["currencies"]
        assert eth_item.raw_data["votes"]["positive"] == 42

    def test_source_label_includes_subsource(self, load_fixture):
        payload = load_fixture("cryptopanic_sample.json")
        items = CryptoPanicSource()._parse(payload)
        # Subsource should be in the source string
        assert any("CoinDesk" in i.source for i in items)


class TestConfiguration:
    def test_always_configured(self):
        # CryptoPanic has a free/public mode, so it's always "configured"
        assert CryptoPanicSource().is_configured()
        assert CryptoPanicSource(api_key="").is_configured()
