"""
CryptoPanic API adapter.

Docs: https://cryptopanic.com/developers/api/
Free tier: 200 requests/day. The public endpoint works without a key but
returns less data. Set CRYPTOPANIC_KEY in .env for full access.

TRADING LOGIC NOTE
------------------
CryptoPanic aggregates from many sub-sources with varying quality, which is
why its base credibility is 0.6 — solidly mid-tier. The community vote field
(`votes.positive` / `votes.negative`) is NOT used for sentiment here; Phase 1B
runs FinBERT on the title itself, which is a more reliable signal than
self-selected upvotes.
"""

from __future__ import annotations

import os
from typing import Any, ClassVar

import requests

from ..types import NewsItem, SourceKind, as_utc
from .base import NewsSource

_API_URL = "https://cryptopanic.com/api/free/v1/posts/"
_TIMEOUT = 10


class CryptoPanicSource(NewsSource):
    kind: ClassVar[SourceKind] = SourceKind.CRYPTOPANIC
    default_credibility: ClassVar[float] = 0.6
    cache_ttl_seconds: ClassVar[int] = 300

    def __init__(self, api_key: str | None = None, *, limit: int = 20, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.api_key = api_key if api_key is not None else os.getenv("CRYPTOPANIC_KEY", "free")
        self.limit = limit

    def is_configured(self) -> bool:
        # CryptoPanic has a public/free mode that works without a real key.
        # It's always "configured", just with reduced quota.
        return True

    def _fetch_raw(self) -> list[NewsItem]:
        params = {
            "auth_token": self.api_key or "free",
            "kind": "news",
            "filter": "important",
            "public": "true",
        }
        resp = requests.get(_API_URL, params=params, timeout=_TIMEOUT)
        resp.raise_for_status()
        return self._parse(resp.json())

    def _parse(self, payload: dict[str, Any]) -> list[NewsItem]:
        items: list[NewsItem] = []
        for post in payload.get("results", [])[: self.limit]:
            title = post.get("title", "").strip()
            if not title:
                continue
            source_title = (post.get("source") or {}).get("title", "cryptopanic")
            items.append(NewsItem(
                source=f"cryptopanic:{source_title}",
                title=title,
                content=title,  # CryptoPanic doesn't return body in free tier
                published_at=as_utc(post.get("published_at") or post.get("created_at") or ""),
                source_credibility=self.default_credibility,
                url=post.get("url", ""),
                raw_data={
                    "currencies": [c.get("code", "") for c in post.get("currencies") or []],
                    "votes": post.get("votes", {}),
                    "domain": post.get("domain", ""),
                },
            ))
        return items
