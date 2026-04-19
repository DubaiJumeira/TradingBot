"""
Fear & Greed Index adapter.

Docs: https://alternative.me/crypto/fear-and-greed-index/
Endpoint: https://api.alternative.me/fng/?limit=1
No auth required. One reading per day (index is daily).

TRADING LOGIC NOTE
------------------
This isn't a news feed — it's a sentiment gauge. We emit it as a NewsItem
anyway so Phase 1B can fold it into the crypto sentiment aggregate without
a special code path. The `content` field carries the numeric index value
so a later analyzer can parse it back out.

The index is one datapoint per day, so cache TTL is 6 hours. The impact
level is always LOW because the number moves slowly — regime shifts matter
(Phase 9's regime detector reads it) but individual updates don't move price.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, ClassVar

import requests

from ..types import ImpactLevel, NewsItem, SourceKind, as_utc
from .base import NewsSource

_API_URL = "https://api.alternative.me/fng/"
_TIMEOUT = 10


class FearGreedSource(NewsSource):
    kind: ClassVar[SourceKind] = SourceKind.FEAR_GREED
    default_credibility: ClassVar[float] = 0.7
    cache_ttl_seconds: ClassVar[int] = 21_600  # 6 hours

    def is_configured(self) -> bool:
        return True  # public endpoint, no auth

    def _fetch_raw(self) -> list[NewsItem]:
        resp = requests.get(_API_URL, params={"limit": 1}, timeout=_TIMEOUT)
        resp.raise_for_status()
        return self._parse(resp.json())

    def _parse(self, payload: dict[str, Any]) -> list[NewsItem]:
        data = payload.get("data") or []
        if not data:
            return []
        entry = data[0]
        try:
            value = int(entry.get("value", 0))
        except (TypeError, ValueError):
            return []
        classification = entry.get("value_classification", "Neutral")
        ts = entry.get("timestamp")
        try:
            published_at = as_utc(int(ts)) if ts is not None else datetime.now(tz=timezone.utc)
        except (TypeError, ValueError):
            published_at = datetime.now(tz=timezone.utc)

        title = f"Crypto Fear & Greed Index: {value} ({classification})"
        return [NewsItem(
            source="fear_greed",
            title=title,
            content=str(value),
            published_at=published_at,
            source_credibility=self.default_credibility,
            impact_level=ImpactLevel.LOW,
            url="https://alternative.me/crypto/fear-and-greed-index/",
            raw_data={"value": value, "classification": classification},
        )]

    def _classify_impact(self, item: NewsItem) -> ImpactLevel:
        # Never upgrade past LOW — this is a background gauge, not an event.
        return ImpactLevel.LOW
