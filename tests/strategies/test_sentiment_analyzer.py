"""
Tests for Phase 1B — SentimentAnalyzer.

All tests use VADER only (FinBERT disabled via use_finbert=False) so tests
stay fast and don't require torch. FinBERT correctness is validated in the
live `make news-smoke` target, not in the offline test suite.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

# Ensure trading-bot is importable.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from strategies.news.types import ImpactLevel, NewsItem
from strategies.sentiment_analyzer import SentimentAnalyzer


def _item(source: str, title: str, credibility: float = 0.8,
          affected_assets: list[str] | None = None,
          raw_data: dict | None = None) -> NewsItem:
    return NewsItem(
        source=source,
        title=title,
        content=title,
        published_at=datetime.now(tz=timezone.utc),
        source_credibility=credibility,
        affected_assets=affected_assets or [],
        raw_data=raw_data or {},
    )


# -----------------------------------------------------------------------
# VADER scoring
# -----------------------------------------------------------------------

class TestVADERScoring:
    def setup_method(self):
        self.analyzer = SentimentAnalyzer(use_finbert=False)

    def test_positive_headline_scores_positive(self):
        items = [_item("rss:reuters", "Amazing rally, investors thrilled with great results")]
        self.analyzer.analyze_items(items)
        assert items[0].sentiment_score > 0

    def test_negative_headline_scores_negative(self):
        items = [_item("rss:reuters", "Markets crash as recession fears grow")]
        self.analyzer.analyze_items(items)
        assert items[0].sentiment_score < 0

    def test_neutral_headline_scores_near_zero(self):
        items = [_item("rss:reuters", "Federal Reserve to meet next Tuesday")]
        self.analyzer.analyze_items(items)
        assert abs(items[0].sentiment_score) < 0.5

    def test_tweets_always_use_vader(self):
        # Even if FinBERT were available, tweets should use VADER for speed.
        items = [_item("twitter:@POTUS", "The economy is doing AMAZING! Best ever!")]
        self.analyzer.analyze_items(items)
        assert items[0].sentiment_score > 0

    def test_score_is_clamped_to_range(self):
        items = [
            _item("a", "Terrible catastrophic disaster"),
            _item("b", "Absolutely amazing incredible fantastic"),
        ]
        self.analyzer.analyze_items(items)
        for item in items:
            assert -1.0 <= item.sentiment_score <= 1.0


# -----------------------------------------------------------------------
# Credibility-weighted aggregate sentiment
# -----------------------------------------------------------------------

class TestAggregateSentiment:
    def setup_method(self):
        self.analyzer = SentimentAnalyzer(use_finbert=False)

    def test_weighted_by_credibility(self):
        # Use explicitly positive language that VADER handles well (financial
        # jargon like "surges" is VADER's blind spot — FinBERT handles it).
        items = [
            _item("rss:reuters", "Amazing rally in gold, investors thrilled",
                  credibility=1.0, affected_assets=["XAUUSD"]),
            _item("reddit:r/wsb", "Gold prices rise sharply, great performance",
                  credibility=0.2, affected_assets=["XAUUSD"]),
        ]
        self.analyzer.analyze_items(items)
        result = self.analyzer.aggregate_sentiment(items, "XAUUSD")
        assert result["item_count"] == 2
        assert result["score"] > 0  # both bullish
        assert result["strongest_source"] == "rss:reuters"

    def test_no_matching_asset_returns_neutral(self):
        items = [_item("rss:reuters", "BTC rises", affected_assets=["BTC/USDT"])]
        self.analyzer.analyze_items(items)
        result = self.analyzer.aggregate_sentiment(items, "XAUUSD")
        assert result["score"] == 0.0
        assert result["item_count"] == 0
        assert result["direction"] == "neutral"

    def test_direction_label(self):
        items = [
            _item("rss:reuters", "Terrible crash, total collapse",
                  credibility=1.0, affected_assets=["SPX500"]),
        ]
        self.analyzer.analyze_items(items)
        result = self.analyzer.aggregate_sentiment(items, "SPX500")
        assert result["direction"] == "bearish"

    def test_mixed_sentiment_averages_out(self):
        items = [
            _item("rss:reuters", "Amazing rally, investors thrilled and optimistic",
                  credibility=1.0, affected_assets=["XAUUSD"]),
            _item("rss:cnbc", "Terrible crash, total collapse and fear",
                  credibility=0.9, affected_assets=["XAUUSD"]),
        ]
        self.analyzer.analyze_items(items)
        result = self.analyzer.aggregate_sentiment(items, "XAUUSD")
        # Mixed signals → closer to neutral than either headline alone.
        assert abs(result["score"]) < 0.8


# -----------------------------------------------------------------------
# Sentiment velocity
# -----------------------------------------------------------------------

class TestSentimentVelocity:
    def setup_method(self):
        self.analyzer = SentimentAnalyzer(use_finbert=False)

    def test_no_history_returns_zero(self):
        assert self.analyzer.sentiment_velocity("XAUUSD") == 0.0

    def test_velocity_detects_positive_shift(self):
        # Simulate two snapshots with increasing sentiment.
        items_neg = [_item("a", "Terrible news for gold",
                           credibility=1.0, affected_assets=["XAUUSD"])]
        self.analyzer.analyze_items(items_neg)
        self.analyzer.aggregate_sentiment(items_neg, "XAUUSD")

        items_pos = [_item("a", "Gold surges on amazing data",
                           credibility=1.0, affected_assets=["XAUUSD"])]
        self.analyzer.analyze_items(items_pos)
        self.analyzer.aggregate_sentiment(items_pos, "XAUUSD")

        vel = self.analyzer.sentiment_velocity("XAUUSD")
        assert vel > 0  # shifted from negative to positive

    def test_velocity_detects_negative_shift(self):
        items_pos = [_item("a", "Gold surges beautifully",
                           credibility=1.0, affected_assets=["XAUUSD"])]
        self.analyzer.analyze_items(items_pos)
        self.analyzer.aggregate_sentiment(items_pos, "XAUUSD")

        items_neg = [_item("a", "Gold collapses catastrophically",
                           credibility=1.0, affected_assets=["XAUUSD"])]
        self.analyzer.analyze_items(items_neg)
        self.analyzer.aggregate_sentiment(items_neg, "XAUUSD")

        vel = self.analyzer.sentiment_velocity("XAUUSD")
        assert vel < 0


# -----------------------------------------------------------------------
# Spam filter
# -----------------------------------------------------------------------

class TestSpamFilter:
    def setup_method(self):
        self.analyzer = SentimentAnalyzer(use_finbert=False)

    def test_wire_services_never_filtered(self):
        items = [
            _item("rss:reuters", "Short"),  # even short titles pass for wire
            _item("newsapi:Bloomberg", "Hi"),
            _item("forexlive", "X"),
        ]
        filtered = self.analyzer.filter_spam(items)
        assert len(filtered) == 3

    def test_short_tweet_is_spam(self):
        items = [_item("twitter:@rando", "lol", raw_data={"handle": "rando"})]
        filtered = self.analyzer.filter_spam(items)
        assert len(filtered) == 0

    def test_new_account_is_spam(self):
        yesterday = (datetime.now(tz=timezone.utc) - timedelta(days=1)).isoformat()
        items = [_item("twitter:@newbot", "Buy $BTC now, trust me bro",
                        raw_data={"account_created": yesterday})]
        filtered = self.analyzer.filter_spam(items)
        assert len(filtered) == 0

    def test_high_post_rate_is_spam(self):
        items = [_item("twitter:@spammer", "Some normal-looking tweet about BTC",
                        raw_data={"posts_per_day": 100})]
        filtered = self.analyzer.filter_spam(items)
        assert len(filtered) == 0

    def test_downvoted_reddit_is_spam(self):
        items = [_item("reddit:r/crypto", "This is definitely not a scam",
                        raw_data={"score": 0})]
        filtered = self.analyzer.filter_spam(items)
        assert len(filtered) == 0

    def test_legit_social_media_passes(self):
        one_year_ago = (datetime.now(tz=timezone.utc) - timedelta(days=400)).isoformat()
        items = [
            _item("twitter:@realDonaldTrump",
                  "The US economy is stronger than ever before",
                  raw_data={"account_created": one_year_ago, "posts_per_day": 10}),
            _item("reddit:r/wallstreetbets",
                  "SPY is going to moon after this CPI print",
                  raw_data={"score": 500, "num_comments": 120}),
        ]
        filtered = self.analyzer.filter_spam(items)
        assert len(filtered) == 2


# -----------------------------------------------------------------------
# Integration: aggregator pipeline with sentiment
# -----------------------------------------------------------------------

class TestAggregatorIntegration:
    """Verify the aggregator calls sentiment scoring when wired up."""

    def test_aggregator_enriches_sentiment(self, tmp_path):
        from strategies.news.aggregator import NewsAggregator
        from strategies.news.cache import NewsCache

        # Use a fake source that returns items with sentiment_score=0.
        from tests.strategies.news.test_aggregator import _FakeSource, _item as agg_item
        cache = NewsCache(cache_dir=str(tmp_path))
        source = _FakeSource(
            "test",
            [agg_item("test", "Markets crash horribly", impact=ImpactLevel.HIGH)],
            cache=cache,
        )
        analyzer = SentimentAnalyzer(use_finbert=False)
        agg = NewsAggregator(
            sources=[source], cache=cache,
            sentiment_analyzer=analyzer, dry_run=False,
        )

        result = agg.fetch_all()
        assert len(result) == 1
        # VADER should have scored this negative headline.
        assert result[0].sentiment_score < 0
