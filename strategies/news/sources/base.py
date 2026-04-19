"""
NewsSource abstract base class.

Every concrete source adapter (CryptoPanic, RSS, Twitter, Reddit, etc.)
subclasses `NewsSource` and implements `_fetch_raw`. The public `fetch()`
method is a template method that handles:

    1. Enabled/disabled gate (missing API key → return [] silently after warning)
    2. TTL caching (so retries and 5-minute cycles don't burn rate limits)
    3. Exponential backoff on transient errors
    4. Exception isolation (one failing source NEVER kills the aggregator)
    5. Coarse impact classification on the NewsItems returned

This matches the master-prompt rule: "the bot runs 24/7 on a VPS, it must
NEVER crash." The aggregator calls `fetch()` — never `_fetch_raw` directly —
so the error boundary is enforced.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from typing import ClassVar

from ..cache import NewsCache
from ..types import ImpactLevel, NewsItem, SourceKind, coarse_impact

logger = logging.getLogger(__name__)


class NewsSource(ABC):
    """
    Abstract base class for all news sources.

    Subclasses must set:
        kind:               SourceKind enum value
        default_credibility: float in [0, 1] — per-source base weight
        cache_ttl_seconds:   how long to cache fetch results

    Subclasses must implement:
        is_configured():   returns True iff required API keys/config present
        _fetch_raw():      returns list[NewsItem] — may raise; wrapped by fetch()
    """

    kind: ClassVar[SourceKind]
    default_credibility: ClassVar[float] = 0.5
    cache_ttl_seconds: ClassVar[int] = 300  # 5 minutes — matches bot cycle
    max_retries: ClassVar[int] = 3
    retry_base_delay: ClassVar[float] = 1.0  # seconds; exponential: 1, 2, 4

    def __init__(self, cache: NewsCache | None = None) -> None:
        self.cache = cache or NewsCache()
        self._warned_not_configured = False

    @property
    def name(self) -> str:
        """Human-readable source name used in logs and NewsItem.source."""
        return self.kind.value

    @abstractmethod
    def is_configured(self) -> bool:
        """
        Returns True iff this source has everything it needs to run
        (API keys, credentials, etc.). If False, `fetch()` returns [] after
        logging a one-time warning — the bot keeps running.
        """

    @abstractmethod
    def _fetch_raw(self) -> list[NewsItem]:
        """
        Subclass-specific fetch logic. May raise on network or parse errors;
        `fetch()` wraps this in retries + an exception boundary.

        Must return a list of NewsItem instances with `source_credibility`
        already populated. Impact classification is applied by `fetch()`.
        """

    def fetch(self) -> list[NewsItem]:
        """
        Public template-method entry point. Never raises. Never returns None.

        Order of operations:
            1. Configuration gate (missing keys → [])
            2. Cache lookup (hit → return cached list)
            3. Retry loop with exponential backoff
            4. Impact classification
            5. Cache store
        """
        if not self.is_configured():
            if not self._warned_not_configured:
                logger.warning(
                    "News source '%s' is not configured (missing API key?) — disabled.",
                    self.name,
                )
                self._warned_not_configured = True
            return []

        cache_key = f"source:{self.name}"
        cached = self.cache.get(cache_key)
        if cached is not None:
            logger.debug("Cache hit for source '%s' (%d items)", self.name, len(cached))
            return cached

        items: list[NewsItem] = []
        last_exc: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                items = self._fetch_raw()
                break
            except Exception as exc:  # noqa: BLE001 — error isolation is the whole point
                last_exc = exc
                delay = self.retry_base_delay * (2 ** attempt)
                logger.warning(
                    "Source '%s' fetch attempt %d/%d failed: %s (retrying in %.1fs)",
                    self.name, attempt + 1, self.max_retries, exc, delay,
                )
                time.sleep(delay)
        else:
            logger.error(
                "Source '%s' failed after %d attempts: %s. Returning empty list.",
                self.name, self.max_retries, last_exc,
            )
            return []

        # Apply coarse impact classification. Individual sources can override
        # `_classify_impact` if they have domain knowledge (e.g., ForexLive
        # tags macro headlines as at least MEDIUM even without urgency keywords).
        for item in items:
            if item.impact_level == ImpactLevel.LOW:
                item.impact_level = self._classify_impact(item)

        self.cache.set(cache_key, items, ttl=self.cache_ttl_seconds)
        logger.info("Source '%s' fetched %d items", self.name, len(items))
        return items

    def _classify_impact(self, item: NewsItem) -> ImpactLevel:
        """
        Default impact classifier. Override in subclasses for source-specific
        rules (e.g., ForexLive defaults to MEDIUM, WhaleAlert tweets start at MEDIUM).
        """
        return coarse_impact(item.title, item.source_credibility)
