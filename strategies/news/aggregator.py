"""
NewsAggregator — orchestrates all configured sources, dedups, and returns a
unified stream of NewsItems.

Called once per bot cycle (every 5 minutes in the main loop) and can also be
called on-demand by Phase 1D's reactive trading mode with `force_refresh=True`.

DESIGN GOALS
------------
1. NEVER raise — one broken source cannot take down the bot.
2. Parallel fetch — all sources run concurrently so total wall-clock time is
   bounded by the slowest source, not the sum.
3. Dedup after merge — same story from Reuters + Bloomberg + CoinDesk collapses
   to one item with the best-confirmed credibility.
4. Dry-run aware — when NEWS_DRY_RUN=true, log items with a clear DRY RUN
   marker so Phase 1D integration can be verified before it's wired to
   real trade execution.

PARALLELISM
-----------
Uses `concurrent.futures.ThreadPoolExecutor`. The underlying sources are
I/O-bound (requests/feedparser/praw/tweepy) so threads are the right choice;
no need for async.
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable

from .cache import NewsCache
from .dedup import deduplicate
from .sources.base import NewsSource
from .sources.cryptopanic import CryptoPanicSource
from .sources.fear_greed import FearGreedSource
from .sources.forexlive_rss import ForexLiveSource
from .sources.newsapi import NewsAPISource
from .sources.reddit import RedditSource
from .sources.rss import GenericRSSSource
from .sources.twitter import TwitterSource
from .types import ImpactLevel, NewsItem, sort_by_time

logger = logging.getLogger(__name__)


def default_sources(cache: NewsCache) -> list[NewsSource]:
    """
    Construct the default source set. Each source reads its own env flags
    to decide whether to actually run — missing keys just disable that source.
    Sharing a single `NewsCache` instance across sources keeps disk IO cheap.
    """
    return [
        ForexLiveSource(cache=cache),        # macro speed lane
        GenericRSSSource(cache=cache),       # Reuters, CNBC, Bloomberg, CoinDesk, The Block
        CryptoPanicSource(cache=cache),
        NewsAPISource(cache=cache),
        TwitterSource(cache=cache),
        RedditSource(cache=cache),
        FearGreedSource(cache=cache),
    ]


class NewsAggregator:
    """
    Fans out to all sources, dedups, runs spam filtering and sentiment
    analysis (Phase 1B), and returns the merged stream.

    Typical usage in bot.py (Phase 1D will wire this in):

        aggregator = NewsAggregator()
        items = aggregator.fetch_all()
        for item in items:
            if item.impact_level >= ImpactLevel.HIGH:
                ...
    """

    def __init__(self, sources: Iterable[NewsSource] | None = None,
                 *, cache: NewsCache | None = None,
                 sentiment_analyzer: Any = None,
                 max_workers: int = 8,
                 dry_run: bool | None = None) -> None:
        self.cache = cache or NewsCache()
        self.sources: list[NewsSource] = (
            list(sources) if sources is not None else default_sources(self.cache)
        )
        self.max_workers = max_workers
        if dry_run is None:
            dry_run = os.getenv("NEWS_DRY_RUN", "true").lower() == "true"
        self.dry_run = dry_run
        self._last_fetch_at: datetime | None = None
        self._last_items: list[NewsItem] = []

        # Phase 1C: asset-news correlation map.
        try:
            from strategies.asset_correlations import NewsAssetMatcher
            self._matcher = NewsAssetMatcher()
        except Exception as exc:
            logger.warning("NewsAssetMatcher unavailable (%s) — affected_assets will be empty", exc)
            self._matcher = None

        # Phase 1B: sentiment analysis. Lazy import to avoid hard dep from 1A tests.
        if sentiment_analyzer is not None:
            self._sentiment = sentiment_analyzer
        else:
            try:
                from strategies.sentiment_analyzer import SentimentAnalyzer
                self._sentiment = SentimentAnalyzer()
            except Exception as exc:
                logger.warning("SentimentAnalyzer unavailable (%s) — scores will be 0.0", exc)
                self._sentiment = None

        # Store last correlation matches for Phase 1D reactive mode.
        self._last_matches: dict[str, Any] = {}

        logger.info(
            "NewsAggregator initialized with %d sources (dry_run=%s, correlations=%s, sentiment=%s): %s",
            len(self.sources), self.dry_run,
            "enabled" if self._matcher else "disabled",
            "enabled" if self._sentiment else "disabled",
            [s.name for s in self.sources],
        )

    # ------------------------------------------------------------- fetch

    def fetch_all(self, *, force_refresh: bool = False) -> list[NewsItem]:
        """
        Fetch from every source in parallel, dedup, and return newest-first.

        Args:
            force_refresh: bypass per-source caches. Used by Phase 1D's
                reactive mode when a critical event requires sub-minute
                freshness. Do NOT set this on the regular 5-minute cycle —
                it will burn rate-limit quota.
        """
        if force_refresh:
            for source in self.sources:
                self.cache.invalidate(f"source:{source.name}")

        all_items: list[NewsItem] = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {pool.submit(self._safe_fetch, s): s for s in self.sources}
            for future in as_completed(futures):
                source = futures[future]
                try:
                    items = future.result()
                except Exception as exc:  # noqa: BLE001
                    # _safe_fetch already catches everything; this is belt-and-braces.
                    logger.error("Unexpected error from source '%s': %s", source.name, exc)
                    items = []
                all_items.extend(items)

        merged = deduplicate(all_items)

        # Phase 1C: populate affected_assets and upgrade impact_level.
        if self._matcher is not None:
            self._last_matches = self._matcher.enrich_items(merged)

        # Phase 1B: spam filter → sentiment scoring.
        if self._sentiment is not None:
            merged = self._sentiment.filter_spam(merged)
            merged = self._sentiment.analyze_items(merged)

        self._last_fetch_at = datetime.now(tz=timezone.utc)
        self._last_items = merged

        if self.dry_run:
            self._log_dry_run(merged)

        logger.info(
            "NewsAggregator cycle complete: %d raw → %d final (dry_run=%s)",
            len(all_items), len(merged), self.dry_run,
        )
        return merged

    def _safe_fetch(self, source: NewsSource) -> list[NewsItem]:
        """Exception boundary around a single source's fetch."""
        try:
            return source.fetch()
        except Exception as exc:  # noqa: BLE001
            logger.error("Source '%s' raised from fetch(): %s", source.name, exc)
            return []

    # ------------------------------------------------------- convenience

    def high_impact(self, minimum: ImpactLevel = ImpactLevel.HIGH,
                    since: datetime | None = None) -> list[NewsItem]:
        """
        Filter the last fetch's items to those at or above `minimum` impact.
        Phase 1D will call this to decide whether to trigger reactive mode.
        """
        cutoff = since or (datetime.now(tz=timezone.utc) - timedelta(hours=1))
        return [
            item for item in self._last_items
            if item.impact_level.rank >= minimum.rank and item.published_at >= cutoff
        ]

    def last_fetch_at(self) -> datetime | None:
        return self._last_fetch_at

    def last_correlation_matches(self) -> dict:
        """Last cycle's correlation matches — keyed by headline title."""
        return self._last_matches

    def aggregate_sentiment(self, asset: str) -> dict:
        """
        Credibility-weighted sentiment for `asset` using the last fetched items.
        Delegates to SentimentAnalyzer.aggregate_sentiment.
        """
        if self._sentiment is None:
            return {"asset": asset, "score": 0.0, "item_count": 0,
                    "strongest_source": "", "direction": "neutral"}
        return self._sentiment.aggregate_sentiment(self._last_items, asset)

    def sentiment_velocity(self, asset: str) -> float:
        """Rate of sentiment change. See SentimentAnalyzer.sentiment_velocity."""
        if self._sentiment is None:
            return 0.0
        return self._sentiment.sentiment_velocity(asset)

    # ------------------------------------------------------------- dryrun

    def _log_dry_run(self, items: list[NewsItem]) -> None:
        """
        In dry-run mode, log every item with a clear [DRY RUN] marker.
        This is how you validate Phase 1A end-to-end before wiring to the
        signal generator: run the bot, watch the logs, confirm relevant
        headlines are being captured with sensible impact levels.
        """
        if not items:
            logger.info("[DRY RUN] NewsAggregator returned no items this cycle")
            return
        for item in items[:20]:  # cap log spam
            logger.info(
                "[DRY RUN] %-6s | cred=%.2f | sent=%+.2f | %s | %s",
                item.impact_level.value.upper(),
                item.source_credibility,
                item.sentiment_score,
                item.source,
                item.title[:120],
            )
        if len(items) > 20:
            logger.info("[DRY RUN] ... and %d more items", len(items) - 20)
