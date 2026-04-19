"""
NewsAPI.org adapter.

Docs: https://newsapi.org/docs
Free tier: 100 requests/day, articles delayed 24h. The delay is a real
problem for event-driven trading — this source is most useful for
backtesting (Phase 7) and background context, NOT for reactive entries.
For real-time macro headlines, prefer ForexLive + Twitter.

TRADING LOGIC NOTE
------------------
We query the /v2/top-headlines endpoint with `category=business` — broader
than keyword search and cheaper on the quota. A single request pulls up to
100 headlines.
"""

from __future__ import annotations

import os
from typing import Any, ClassVar

import requests

from ..types import NewsItem, SourceKind, as_utc
from .base import NewsSource

_API_URL = "https://newsapi.org/v2/top-headlines"
_TIMEOUT = 10


class NewsAPISource(NewsSource):
    kind: ClassVar[SourceKind] = SourceKind.NEWSAPI
    default_credibility: ClassVar[float] = 0.8
    cache_ttl_seconds: ClassVar[int] = 900  # 15 min — matches 24h delay on free tier

    def __init__(self, api_key: str | None = None, *, country: str = "us",
                 category: str = "business", page_size: int = 50, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.api_key = api_key if api_key is not None else os.getenv("NEWSAPI_KEY", "")
        self.country = country
        self.category = category
        self.page_size = page_size

    def is_configured(self) -> bool:
        return bool(self.api_key)

    def _fetch_raw(self) -> list[NewsItem]:
        params = {
            "country": self.country,
            "category": self.category,
            "pageSize": self.page_size,
            "apiKey": self.api_key,
        }
        resp = requests.get(_API_URL, params=params, timeout=_TIMEOUT)
        resp.raise_for_status()
        return self._parse(resp.json())

    def _parse(self, payload: dict[str, Any]) -> list[NewsItem]:
        items: list[NewsItem] = []
        for art in payload.get("articles", []):
            title = (art.get("title") or "").strip()
            if not title or title == "[Removed]":
                continue
            source_name = (art.get("source") or {}).get("name", "newsapi")
            items.append(NewsItem(
                source=f"newsapi:{source_name}",
                title=title,
                content=(art.get("description") or art.get("content") or title).strip(),
                published_at=as_utc(art.get("publishedAt") or ""),
                source_credibility=self.default_credibility,
                url=art.get("url", ""),
                raw_data={
                    "author": art.get("author"),
                    "source_id": (art.get("source") or {}).get("id"),
                },
            ))
        return items
