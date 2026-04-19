"""
Generic RSS source — covers Reuters, CNBC, CoinDesk, The Block, Bloomberg Markets.

This is the workhorse of the news engine. RSS is free, unpaywalled, fast,
and rate-limit-free. Each configured feed is polled every cycle and its
entries are merged into one stream.

NOTE ON PAYWALLED OUTLETS
-------------------------
FT (Financial Times) and full Bloomberg newsroom RSS are paywalled or
geoblocked and not included. The Bloomberg Markets feed is publicly
reachable but returns shorter summaries. If you later obtain an FT key,
add a dedicated adapter rather than shoehorning it in here.

FEED LIST
---------
The default feeds were selected for reliability and macro-relevance:
    - Reuters Business News — wire-grade, fast on earnings/macro
    - CNBC Top News — broad US-market coverage
    - Bloomberg Markets — shorter summaries but the brand matters for dedup
    - CoinDesk — crypto-specific
    - The Block — crypto-specific, often breaks regulatory stories

ForexLive gets its own adapter in `forexlive_rss.py` because it needs
different credibility weighting and a MEDIUM impact floor (see docstring there).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, ClassVar, Iterable

import feedparser

from ..types import NewsItem, SourceKind, as_utc
from .base import NewsSource

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FeedSpec:
    """One RSS feed entry: URL + per-feed credibility + display name."""
    name: str
    url: str
    credibility: float


# Default feeds. URLs verified publicly-reachable as of Phase 1A build.
# Keep this list in sync with the docstring above.
DEFAULT_FEEDS: tuple[FeedSpec, ...] = (
    FeedSpec("reuters_business", "https://feeds.reuters.com/reuters/businessNews", 1.0),
    FeedSpec(
        "cnbc_top_news",
        "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=100003114",
        0.9,
    ),
    FeedSpec("bloomberg_markets", "https://feeds.bloomberg.com/markets/news.rss", 1.0),
    FeedSpec("coindesk", "https://www.coindesk.com/arc/outboundfeeds/rss/", 0.75),
    FeedSpec("theblock", "https://www.theblock.co/rss.xml", 0.75),
)


class GenericRSSSource(NewsSource):
    """
    Polls multiple RSS feeds and returns their merged entries.

    Each feed is fetched independently — if one feed 404s or times out, the
    others still return data. This is a second layer of error isolation on
    top of the `NewsSource.fetch()` retry boundary.
    """

    kind: ClassVar[SourceKind] = SourceKind.RSS
    default_credibility: ClassVar[float] = 0.9  # weighted avg of default feeds
    cache_ttl_seconds: ClassVar[int] = 300

    def __init__(self, feeds: Iterable[FeedSpec] | None = None,
                 *, entries_per_feed: int = 20, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.feeds = tuple(feeds) if feeds is not None else DEFAULT_FEEDS
        self.entries_per_feed = entries_per_feed

    def is_configured(self) -> bool:
        return len(self.feeds) > 0

    def _fetch_raw(self) -> list[NewsItem]:
        all_items: list[NewsItem] = []
        for feed in self.feeds:
            try:
                all_items.extend(self._fetch_one(feed))
            except Exception as exc:  # noqa: BLE001
                # Per-feed isolation so one bad feed doesn't blank the whole source.
                logger.warning("RSS feed '%s' (%s) failed: %s", feed.name, feed.url, exc)
        return all_items

    def _fetch_one(self, feed: FeedSpec) -> list[NewsItem]:
        parsed = feedparser.parse(feed.url)
        # feedparser sets `bozo` on parse errors but often still returns entries.
        # Only bail if there are zero entries AND bozo is set with an exception.
        if getattr(parsed, "bozo", False) and not parsed.entries:
            raise RuntimeError(f"feedparser error: {getattr(parsed, 'bozo_exception', 'unknown')}")
        return self._parse_entries(feed, parsed.entries[: self.entries_per_feed])

    def _parse_entries(self, feed: FeedSpec, entries: list[Any]) -> list[NewsItem]:
        items: list[NewsItem] = []
        for entry in entries:
            title = (entry.get("title") or "").strip()
            if not title:
                continue
            published = (
                entry.get("published")
                or entry.get("updated")
                or entry.get("pubDate")
                or ""
            )
            try:
                published_at = as_utc(published) if published else as_utc("1970-01-01T00:00:00Z")
            except Exception:
                published_at = as_utc("1970-01-01T00:00:00Z")

            summary = (entry.get("summary") or entry.get("description") or title).strip()
            items.append(NewsItem(
                source=f"rss:{feed.name}",
                title=title,
                content=summary,
                published_at=published_at,
                source_credibility=feed.credibility,
                url=entry.get("link", ""),
                raw_data={"feed_name": feed.name, "feed_url": feed.url},
            ))
        return items
