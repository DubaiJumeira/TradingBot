"""
Tests for Phase 1C — Asset-News Correlation Map.

Verifies keyword matching, source filtering, impact upgrades, affected_assets
population, and the helper that flattens matches by asset.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from strategies.asset_correlations import (
    AssetImpact,
    CorrelationMatch,
    NewsAssetMatcher,
    get_affected_assets,
)
from strategies.news.types import ImpactLevel, NewsItem


def _item(source: str, title: str) -> NewsItem:
    return NewsItem(
        source=source,
        title=title,
        content=title,
        published_at=datetime.now(tz=timezone.utc),
        source_credibility=0.9,
    )


# -----------------------------------------------------------------------
# Keyword matching
# -----------------------------------------------------------------------

class TestKeywordMatching:
    def setup_method(self):
        self.matcher = NewsAssetMatcher()

    def test_tariff_headline_matches_trump_tariff(self):
        item = _item("rss:reuters", "Trump announces new 25% tariff on Chinese imports")
        matches = self.matcher.match_item(item)
        names = [m.pattern_name for m in matches]
        assert "trump_tariff" in names

    def test_war_headline_matches_war_escalation(self):
        item = _item("rss:reuters", "Israel launches military strike against Iran targets")
        matches = self.matcher.match_item(item)
        names = [m.pattern_name for m in matches]
        assert "war_escalation" in names

    def test_ceasefire_matches_war_deescalation(self):
        item = _item("rss:forexlive", "Breaking: ceasefire agreed in Middle East conflict")
        matches = self.matcher.match_item(item)
        names = [m.pattern_name for m in matches]
        assert "war_deescalation" in names

    def test_opec_matches(self):
        item = _item("rss:reuters", "OPEC agrees to cut oil output by 500k barrels per day")
        matches = self.matcher.match_item(item)
        names = [m.pattern_name for m in matches]
        assert "opec_decision" in names

    def test_fed_hawkish_matches(self):
        item = _item("rss:reuters", "Fed signals higher for longer as inflation remains persistent")
        matches = self.matcher.match_item(item)
        names = [m.pattern_name for m in matches]
        assert "fed_hawkish" in names

    def test_fed_dovish_matches(self):
        item = _item("rss:reuters", "Fed pivots to rate cut stance, soft landing in sight")
        matches = self.matcher.match_item(item)
        names = [m.pattern_name for m in matches]
        assert "fed_dovish" in names

    def test_cpi_hot_matches(self):
        item = _item("forexlive", "CPI higher than expected at 3.5%, inflation accelerating")
        matches = self.matcher.match_item(item)
        names = [m.pattern_name for m in matches]
        assert "cpi_hot" in names

    def test_crypto_etf_approved_matches(self):
        item = _item("cryptopanic:CoinDesk", "SEC announces ETF approved for spot Bitcoin")
        matches = self.matcher.match_item(item)
        names = [m.pattern_name for m in matches]
        assert "crypto_regulation_positive" in names

    def test_crypto_ban_matches(self):
        item = _item("rss:reuters", "SEC lawsuit filed against major crypto exchange")
        matches = self.matcher.match_item(item)
        names = [m.pattern_name for m in matches]
        assert "crypto_regulation_negative" in names

    def test_exchange_hack_matches(self):
        item = _item("cryptopanic:TheBlock", "DeFi bridge hack: $200 million stolen")
        matches = self.matcher.match_item(item)
        names = [m.pattern_name for m in matches]
        assert "exchange_hack" in names

    def test_no_match_returns_empty(self):
        item = _item("rss:reuters", "Weather forecast: sunny with a chance of clouds")
        matches = self.matcher.match_item(item)
        assert matches == []


# -----------------------------------------------------------------------
# Source filtering
# -----------------------------------------------------------------------

class TestSourceFilter:
    def setup_method(self):
        self.matcher = NewsAssetMatcher()

    def test_trump_crypto_only_matches_potus_sources(self):
        # From Trump's account → should match.
        item_trump = _item("twitter:@realDonaldTrump",
                           "We are creating a strategic bitcoin reserve for America")
        matches = self.matcher.match_item(item_trump)
        names = [m.pattern_name for m in matches]
        assert "trump_crypto" in names

    def test_trump_crypto_ignores_random_accounts(self):
        # Same keywords but from a random account → should NOT match trump_crypto.
        item_random = _item("twitter:@cryptobro",
                            "bitcoin reserve would be great for digital asset adoption")
        matches = self.matcher.match_item(item_random)
        names = [m.pattern_name for m in matches]
        assert "trump_crypto" not in names

    def test_trump_crypto_matches_whitehouse(self):
        item = _item("twitter:@WhiteHouse",
                      "The President signs executive order on digital asset framework")
        matches = self.matcher.match_item(item)
        names = [m.pattern_name for m in matches]
        assert "trump_crypto" in names


# -----------------------------------------------------------------------
# Item enrichment (affected_assets + impact upgrade)
# -----------------------------------------------------------------------

class TestEnrichItem:
    def setup_method(self):
        self.matcher = NewsAssetMatcher()

    def test_populates_affected_assets(self):
        item = _item("rss:reuters", "Military strike reported, troops deployed to border")
        assert item.affected_assets == []
        self.matcher.enrich_item(item)
        assert "XAUUSD" in item.affected_assets
        assert "XTIUSD" in item.affected_assets
        assert "SPX500" in item.affected_assets

    def test_upgrades_impact_level(self):
        item = _item("rss:reuters", "Military strike reported, invasion underway")
        assert item.impact_level == ImpactLevel.LOW  # default
        self.matcher.enrich_item(item)
        assert item.impact_level == ImpactLevel.CRITICAL  # war_escalation is critical for gold/oil

    def test_preserves_existing_assets(self):
        item = _item("rss:reuters", "OPEC announces oil production cut")
        item.affected_assets = ["BTC/USDT"]
        self.matcher.enrich_item(item)
        assert "BTC/USDT" in item.affected_assets  # preserved
        assert "XTIUSD" in item.affected_assets     # added

    def test_multiple_pattern_matches_accumulate(self):
        # "tariff" + "higher for longer" → matches both trump_tariff and fed_hawkish
        item = _item("rss:reuters",
                      "Tariff escalation alongside hawkish Fed, higher for longer expected")
        matches = self.matcher.enrich_item(item)
        pattern_names = [m.pattern_name for m in matches]
        assert "trump_tariff" in pattern_names
        assert "fed_hawkish" in pattern_names
        # Assets from BOTH patterns should be present.
        assert "XAUUSD" in item.affected_assets
        assert "SPX500" in item.affected_assets
        assert "BTC/USDT" in item.affected_assets

    def test_enrich_items_batch(self):
        items = [
            _item("rss:reuters", "OPEC agrees to cut output"),
            _item("rss:reuters", "Weather is nice today"),
            _item("rss:reuters", "SEC lawsuit against crypto exchange"),
        ]
        result = self.matcher.enrich_items(items)
        # Two of three should match.
        assert len(result) == 2
        assert items[0].affected_assets  # OPEC → XTIUSD
        assert items[1].affected_assets == []  # no match
        assert items[2].affected_assets  # crypto regulation


# -----------------------------------------------------------------------
# Impact details (direction, delay)
# -----------------------------------------------------------------------

class TestImpactDetails:
    def setup_method(self):
        self.matcher = NewsAssetMatcher()

    def test_war_escalation_gold_is_positive(self):
        item = _item("rss:reuters", "Military strike on enemy targets")
        matches = self.matcher.match_item(item)
        war_match = next(m for m in matches if m.pattern_name == "war_escalation")
        gold = next(i for i in war_match.impacts if i.asset == "XAUUSD")
        assert gold.direction == "positive"
        assert gold.magnitude == ImpactLevel.CRITICAL
        assert gold.delay_seconds == 30

    def test_fed_dovish_btc_is_positive_with_delay(self):
        item = _item("rss:reuters", "Fed pivots to rate cut stance")
        matches = self.matcher.match_item(item)
        dovish = next(m for m in matches if m.pattern_name == "fed_dovish")
        btc = next(i for i in dovish.impacts if i.asset == "BTC/USDT")
        assert btc.direction == "positive"
        assert btc.delay_seconds == 120  # crypto reacts slower to macro

    def test_whale_movement_has_long_delay(self):
        item = _item("twitter:@WhaleAlert", "500 BTC transferred to exchange")
        matches = self.matcher.match_item(item)
        whale = next(m for m in matches if m.pattern_name == "whale_movement")
        btc = next(i for i in whale.impacts if i.asset == "BTC/USDT")
        assert btc.delay_seconds == 300


# -----------------------------------------------------------------------
# get_affected_assets helper
# -----------------------------------------------------------------------

class TestGetAffectedAssets:
    def test_highest_magnitude_wins_per_asset(self):
        matches = [
            CorrelationMatch("a", "kw1", [
                AssetImpact("XAUUSD", "positive", ImpactLevel.MEDIUM, 60),
            ]),
            CorrelationMatch("b", "kw2", [
                AssetImpact("XAUUSD", "positive", ImpactLevel.CRITICAL, 30),
            ]),
        ]
        result = get_affected_assets(matches)
        assert result["XAUUSD"].magnitude == ImpactLevel.CRITICAL
        assert result["XAUUSD"].delay_seconds == 30

    def test_multiple_assets_returned(self):
        matches = [
            CorrelationMatch("a", "kw", [
                AssetImpact("XAUUSD", "positive", ImpactLevel.HIGH, 30),
                AssetImpact("SPX500", "negative", ImpactLevel.HIGH, 60),
            ]),
        ]
        result = get_affected_assets(matches)
        assert "XAUUSD" in result
        assert "SPX500" in result

    def test_empty_matches_returns_empty(self):
        assert get_affected_assets([]) == {}
