"""
News engine package (Phase 1A).

Public API:
    - NewsAggregator:     top-level orchestrator
    - NewsItem:           normalized item shape
    - ImpactLevel:        enum for coarse impact classification
    - SourceKind:         enum for source identification
    - NewsSource:         ABC for implementing new sources

Phase 1A scope: multi-source ingestion + dedup + caching. No sentiment
analysis yet (Phase 1B) and no asset correlation (Phase 1C). The existing
`strategies/news_events.py` module still runs — Phase 1D will replace it.
"""

from .aggregator import NewsAggregator, default_sources
from .cache import NewsCache
from .dedup import deduplicate
from .sources.base import NewsSource
from .types import ImpactLevel, NewsItem, SourceKind

__all__ = [
    "NewsAggregator",
    "NewsCache",
    "NewsItem",
    "NewsSource",
    "ImpactLevel",
    "SourceKind",
    "default_sources",
    "deduplicate",
]
