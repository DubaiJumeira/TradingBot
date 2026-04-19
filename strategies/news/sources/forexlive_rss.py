"""
ForexLive RSS adapter — dedicated because this feed is the single most
valuable free source for your oil/gold/S&P use case.

WHY A DEDICATED FILE
--------------------
ForexLive is the fastest free publisher of macro headlines: FOMC statements,
CPI surprises, Trump tariff announcements, OPEC decisions, Powell speeches.
Their reporters literally retype wire copy within seconds of a release
crossing the tape. For a bot trading XAU/XTI/SPX, this feed will move the
P&L needle more than Bloomberg and FT combined — those are paywalled and
often behind ForexLive by minutes.

DIFFERENCES FROM THE GENERIC RSS ADAPTER
----------------------------------------
    - Higher base credibility (0.95) — macro-desk grade
    - Impact floor of MEDIUM instead of LOW — even a quiet ForexLive headline
      is worth more than a loud CoinDesk headline for macro instruments
    - Separate cache TTL: 120s — we want fresher data here because this is
      the macro speed-lane that feeds Phase 1D's reactive mode

FEED SELECTION
--------------
ForexLive publishes multiple RSS feeds. The "news" feed covers all desks
(FX, commodities, central banks, equities) and is the right default for
a multi-instrument bot.
"""

from __future__ import annotations

from typing import ClassVar

from ..types import ImpactLevel, NewsItem, SourceKind, coarse_impact
from .rss import FeedSpec, GenericRSSSource

_FOREXLIVE_FEED = FeedSpec(
    name="forexlive",
    url="https://www.forexlive.com/feed/news",
    credibility=0.95,
)


class ForexLiveSource(GenericRSSSource):
    """
    Reuses `GenericRSSSource`'s parsing and per-feed error isolation but
    with ForexLive-specific credibility, caching, and impact classification.
    """

    kind: ClassVar[SourceKind] = SourceKind.FOREXLIVE
    default_credibility: ClassVar[float] = 0.95
    cache_ttl_seconds: ClassVar[int] = 120  # fresher than general RSS — macro speed lane

    def __init__(self, **kwargs) -> None:  # type: ignore[no-untyped-def]
        super().__init__(feeds=(_FOREXLIVE_FEED,), entries_per_feed=30, **kwargs)

    @property
    def name(self) -> str:
        return "forexlive"

    def _classify_impact(self, item: NewsItem) -> ImpactLevel:
        """
        Override: every ForexLive headline starts at MEDIUM. Urgency keywords
        or top-credibility confirmation bump it to HIGH. CRITICAL is reserved
        for Phase 1C's correlation map (e.g., 'OPEC production cut' → CRITICAL).
        """
        base = coarse_impact(item.title, item.source_credibility)
        if base.rank < ImpactLevel.MEDIUM.rank:
            return ImpactLevel.MEDIUM
        return base
