"""
Core types for the news engine.

A NewsItem is the unified shape every source adapter must produce. Downstream
phases (1B sentiment, 1C asset correlation, 1D reactive trading) read these
fields and never touch source-specific raw payloads except via `raw_data` for
debugging/audit.

TRADING LOGIC NOTE
------------------
Phase 1A is responsible for *ingestion and normalization only*. It leaves
`sentiment_score` at 0.0 and `affected_assets` empty — those are filled by
later phases. `impact_level` is set by the aggregator using a crude heuristic
(source credibility + urgency keywords) and is intended as a coarse filter for
Phase 1D's reactive mode, which then re-scores using sentiment velocity and
the asset correlation map.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Iterable


class SourceKind(str, Enum):
    """
    What kind of source this item came from. Used for credibility weighting
    and downstream filtering (e.g., Phase 1C may only match 'trump_crypto'
    keywords against items whose source is TWITTER from the POTUS account).
    """

    TWITTER = "twitter"
    CRYPTOPANIC = "cryptopanic"
    NEWSAPI = "newsapi"
    RSS = "rss"
    FOREXLIVE = "forexlive"
    REDDIT = "reddit"
    FEAR_GREED = "fear_greed"


class ImpactLevel(str, Enum):
    """
    Coarse impact classification. Ordering matters — `>=` comparisons are used
    in the reactive-mode trigger (Phase 1D).
    """

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

    @property
    def rank(self) -> int:
        return {"low": 0, "medium": 1, "high": 2, "critical": 3}[self.value]

    def __ge__(self, other: "ImpactLevel") -> bool:  # type: ignore[override]
        if not isinstance(other, ImpactLevel):
            return NotImplemented
        return self.rank >= other.rank

    def __gt__(self, other: "ImpactLevel") -> bool:  # type: ignore[override]
        if not isinstance(other, ImpactLevel):
            return NotImplemented
        return self.rank > other.rank


@dataclass
class NewsItem:
    """
    Normalized news/social item. Every source adapter MUST emit this shape.

    Fields
    ------
    source:
        The concrete source identifier — either a SourceKind value or a more
        specific string like "twitter:@realDonaldTrump" or "rss:reuters". The
        aggregator's dedup step groups by canonical title, not by source.
    title:
        Short headline / tweet text. Used as the dedup key basis.
    content:
        Full body where available. For tweets, this duplicates `title`.
    published_at:
        Timezone-aware UTC timestamp. Naive datetimes are rejected at __post_init__.
    sentiment_score:
        Range [-1.0, +1.0]. Populated by Phase 1B. Phase 1A always sets 0.0.
    impact_level:
        Coarse severity. Phase 1A assigns this from credibility + urgency keyword
        heuristics; Phase 1C/1D may upgrade it when the asset correlation map
        matches a "critical" pattern (e.g., "OPEC output cut").
    affected_assets:
        List of instrument symbols like "XTIUSD", "BTC/USDT". Empty in Phase 1A.
    source_credibility:
        [0.0, 1.0]. Per-source base weight (e.g. Reuters=1.0, Reddit=0.2).
        Phase 1B uses this to weight sentiment in the aggregate.
    url:
        Canonical link back to the item.
    raw_data:
        Verbatim source payload for debugging and offline replay in tests.
    """

    source: str
    title: str
    content: str
    published_at: datetime
    sentiment_score: float = 0.0
    impact_level: ImpactLevel = ImpactLevel.LOW
    affected_assets: list[str] = field(default_factory=list)
    source_credibility: float = 0.5
    url: str = ""
    raw_data: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Reject naive datetimes — downstream code does UTC math and silent
        # tz coercion has caused bugs in the legacy news_events module.
        if self.published_at.tzinfo is None:
            self.published_at = self.published_at.replace(tzinfo=timezone.utc)
        # Clamp score and credibility so downstream code can trust the range.
        self.sentiment_score = max(-1.0, min(1.0, float(self.sentiment_score)))
        self.source_credibility = max(0.0, min(1.0, float(self.source_credibility)))

    def to_dict(self) -> dict[str, Any]:
        """Serializable form — for logging, fixtures, DB storage (Phase 10)."""
        d = asdict(self)
        d["published_at"] = self.published_at.isoformat()
        d["impact_level"] = self.impact_level.value
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "NewsItem":
        """Inverse of `to_dict` — used by the fixture loader in tests."""
        return cls(
            source=d["source"],
            title=d["title"],
            content=d["content"],
            published_at=datetime.fromisoformat(d["published_at"]),
            sentiment_score=d.get("sentiment_score", 0.0),
            impact_level=ImpactLevel(d.get("impact_level", "low")),
            affected_assets=list(d.get("affected_assets", [])),
            source_credibility=d.get("source_credibility", 0.5),
            url=d.get("url", ""),
            raw_data=dict(d.get("raw_data", {})),
        )


# Urgency keywords used by the aggregator's coarse impact heuristic. This is
# deliberately small and source-agnostic — Phase 1C's asset correlation map
# does the real work of classifying impact per asset.
URGENCY_KEYWORDS: tuple[str, ...] = (
    "breaking",
    "urgent",
    "just in",
    "alert",
    "flash",
    "live:",
    "developing",
)


def coarse_impact(title: str, credibility: float) -> ImpactLevel:
    """
    First-pass impact classification used by Phase 1A only.

    Rules:
        - Title contains any urgency keyword AND source credibility >= 0.7 → HIGH
        - Source credibility == 1.0 (top-tier wires) → MEDIUM floor
        - Otherwise → LOW

    Phase 1C's correlation map is responsible for upgrading to CRITICAL when
    a pattern like "OPEC cut" or "Trump tariff" hits; keeping that logic out
    of 1A avoids tangling ingestion with classification.
    """
    lowered = title.lower()
    has_urgency = any(kw in lowered for kw in URGENCY_KEYWORDS)
    if has_urgency and credibility >= 0.7:
        return ImpactLevel.HIGH
    if credibility >= 1.0:
        return ImpactLevel.MEDIUM
    return ImpactLevel.LOW


def as_utc(dt: datetime | str | int | float) -> datetime:
    """
    Normalize various timestamp shapes to tz-aware UTC datetime.

    Accepts:
        - datetime (naive → assumed UTC; aware → converted to UTC)
        - ISO-8601 string (with or without 'Z' suffix)
        - epoch seconds (int/float)
    """
    if isinstance(dt, datetime):
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    if isinstance(dt, (int, float)):
        return datetime.fromtimestamp(dt, tz=timezone.utc)
    if isinstance(dt, str):
        # feedparser hands us RFC 822, APIs hand us ISO 8601. dateutil handles both.
        try:
            from dateutil import parser as _p  # lazy import — optional dep path
            parsed = _p.parse(dt)
        except Exception:
            # Minimal fallback for the "2024-01-01T00:00:00Z" case.
            parsed = datetime.fromisoformat(dt.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    raise TypeError(f"Unsupported timestamp type: {type(dt).__name__}")


def sort_by_time(items: Iterable[NewsItem]) -> list[NewsItem]:
    """Newest-first ordering used by the aggregator before returning."""
    return sorted(items, key=lambda i: i.published_at, reverse=True)
