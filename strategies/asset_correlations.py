"""
Phase 1C — Asset-News Correlation Map

Maps incoming news items to affected instruments and expected directional
impact. This is the "knowledge base" that transforms a generic headline into
an actionable trading signal.

HOW IT WORKS
------------
1. Each incoming NewsItem's title+content is scanned against keyword patterns.
2. On a match, the item's `affected_assets` list is populated with the
   instruments from the matching pattern, and `impact_level` is upgraded
   to the pattern's magnitude (may go from LOW→CRITICAL).
3. A `CorrelationMatch` dataclass is returned per match, carrying:
   - expected direction (positive/negative/variable)
   - magnitude → maps to ImpactLevel
   - delay_seconds → how quickly the market typically reacts
   - the matched pattern name (for audit logging)
4. Phase 1D's reactive mode uses these matches to decide WHEN to look for
   ICT setups (after the delay) and WHICH direction to favor.

SOURCE FILTERING
----------------
Some patterns include a `source_filter` — e.g., "trump_crypto" should only
trigger on tweets from @realDonaldTrump or @POTUS, not on random accounts
quoting the same words. The matcher checks `item.source` against this filter.

ADDING NEW PATTERNS
-------------------
Add entries to NEWS_ASSET_MAP below. Each entry needs:
    keywords:       list of substrings to match (case-insensitive, OR logic)
    affects:        dict of instrument → {direction, magnitude, delay_seconds}
    source_filter:  (optional) list of source substrings that must match

The map is intentionally hardcoded, not learned. Market reactions to event
categories are well-understood and stable. ML-based correlation discovery is
a Phase 7 (backtest) concern, not a real-time concern.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from strategies.news.types import ImpactLevel, NewsItem

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Magnitude → ImpactLevel mapping
# ---------------------------------------------------------------------------

_MAGNITUDE_MAP: dict[str, ImpactLevel] = {
    "low": ImpactLevel.LOW,
    "medium": ImpactLevel.MEDIUM,
    "high": ImpactLevel.HIGH,
    "critical": ImpactLevel.CRITICAL,
}


# ---------------------------------------------------------------------------
# The correlation map — the core knowledge base
# ---------------------------------------------------------------------------

NEWS_ASSET_MAP: dict[str, dict[str, Any]] = {
    # ============================= GEOPOLITICAL =============================
    "trump_tariff": {
        "keywords": ["tariff", "trade war", "import duty", "trade deal", "customs"],
        "affects": {
            "SPX500":    {"direction": "negative", "magnitude": "high",     "delay_seconds": 30},
            "US30":      {"direction": "negative", "magnitude": "high",     "delay_seconds": 30},
            "XAUUSD":   {"direction": "positive", "magnitude": "high",     "delay_seconds": 60},
            "BTC/USDT": {"direction": "negative", "magnitude": "medium",   "delay_seconds": 120},
            "XTIUSD":   {"direction": "variable", "magnitude": "high",     "delay_seconds": 60},
        },
    },
    "war_escalation": {
        "keywords": ["military strike", "invasion", "missile", "troops deployed",
                     "war", "conflict escalat"],
        "affects": {
            "XAUUSD":   {"direction": "positive", "magnitude": "critical", "delay_seconds": 30},
            "XTIUSD":   {"direction": "positive", "magnitude": "critical", "delay_seconds": 30},
            "SPX500":   {"direction": "negative", "magnitude": "high",     "delay_seconds": 60},
            "US30":     {"direction": "negative", "magnitude": "high",     "delay_seconds": 60},
        },
    },
    "war_deescalation": {
        "keywords": ["ceasefire", "peace deal", "peace talks", "truce",
                     "de-escalat", "withdrawal"],
        "affects": {
            "XAUUSD":   {"direction": "negative", "magnitude": "high",     "delay_seconds": 60},
            "XTIUSD":   {"direction": "negative", "magnitude": "medium",   "delay_seconds": 60},
            "SPX500":   {"direction": "positive", "magnitude": "medium",   "delay_seconds": 120},
        },
    },
    "opec_decision": {
        "keywords": ["OPEC", "oil production cut", "oil output",
                     "barrel per day", "oil supply"],
        "affects": {
            "XTIUSD":   {"direction": "variable", "magnitude": "critical", "delay_seconds": 30},
        },
    },
    # ================================= FED ==================================
    "fed_hawkish": {
        "keywords": ["rate hike", "hawkish", "tightening",
                     "inflation persistent", "higher for longer"],
        "affects": {
            "XAUUSD":   {"direction": "negative", "magnitude": "high",     "delay_seconds": 60},
            "SPX500":   {"direction": "negative", "magnitude": "high",     "delay_seconds": 60},
            "US30":     {"direction": "negative", "magnitude": "high",     "delay_seconds": 60},
            "BTC/USDT": {"direction": "negative", "magnitude": "medium",   "delay_seconds": 120},
        },
    },
    "fed_dovish": {
        "keywords": ["rate cut", "dovish", "easing", "pivot",
                     "soft landing", "disinflation"],
        "affects": {
            "XAUUSD":   {"direction": "positive", "magnitude": "medium",   "delay_seconds": 60},
            "SPX500":   {"direction": "positive", "magnitude": "high",     "delay_seconds": 60},
            "US30":     {"direction": "positive", "magnitude": "high",     "delay_seconds": 60},
            "BTC/USDT": {"direction": "positive", "magnitude": "high",     "delay_seconds": 120},
        },
    },
    # ================================ MACRO =================================
    "cpi_hot": {
        "keywords": ["CPI above", "CPI higher than expected",
                     "inflation rose", "inflation accelerat"],
        "affects": {
            "XAUUSD":   {"direction": "positive", "magnitude": "high",     "delay_seconds": 30},
            "SPX500":   {"direction": "negative", "magnitude": "high",     "delay_seconds": 30},
            "US30":     {"direction": "negative", "magnitude": "high",     "delay_seconds": 30},
            "BTC/USDT": {"direction": "negative", "magnitude": "medium",   "delay_seconds": 120},
        },
    },
    "cpi_cool": {
        "keywords": ["CPI below", "CPI lower than expected",
                     "inflation cooled", "inflation fell"],
        "affects": {
            "SPX500":   {"direction": "positive", "magnitude": "high",     "delay_seconds": 30},
            "US30":     {"direction": "positive", "magnitude": "high",     "delay_seconds": 30},
            "BTC/USDT": {"direction": "positive", "magnitude": "medium",   "delay_seconds": 120},
            "XAUUSD":   {"direction": "negative", "magnitude": "medium",   "delay_seconds": 60},
        },
    },
    # ============================== CRYPTO ==================================
    "crypto_regulation_positive": {
        "keywords": ["ETF approved", "crypto bill passed", "pro-crypto",
                     "strategic reserve", "stablecoin bill"],
        "affects": {
            "BTC/USDT": {"direction": "positive", "magnitude": "critical", "delay_seconds": 60},
            "ETH/USDT": {"direction": "positive", "magnitude": "high",     "delay_seconds": 60},
            "SOL/USDT": {"direction": "positive", "magnitude": "high",     "delay_seconds": 90},
        },
    },
    "crypto_regulation_negative": {
        "keywords": ["crypto ban", "SEC lawsuit", "exchange shut down",
                     "crypto crackdown"],
        "affects": {
            "BTC/USDT": {"direction": "negative", "magnitude": "critical", "delay_seconds": 30},
            "ETH/USDT": {"direction": "negative", "magnitude": "critical", "delay_seconds": 30},
            "SOL/USDT": {"direction": "negative", "magnitude": "high",     "delay_seconds": 60},
        },
    },
    "trump_crypto": {
        "keywords": ["bitcoin reserve", "crypto friendly", "digital asset"],
        "source_filter": ["@realDonaldTrump", "@POTUS", "whitehouse"],
        "affects": {
            "BTC/USDT": {"direction": "positive", "magnitude": "critical", "delay_seconds": 30},
            "ETH/USDT": {"direction": "positive", "magnitude": "high",     "delay_seconds": 60},
        },
    },
    "exchange_hack": {
        "keywords": ["exchange hack", "funds stolen", "exploit",
                     "bridge hack", "million stolen"],
        "affects": {
            "BTC/USDT": {"direction": "negative", "magnitude": "high",     "delay_seconds": 60},
            "ETH/USDT": {"direction": "negative", "magnitude": "high",     "delay_seconds": 60},
        },
    },
    "whale_movement": {
        "keywords": ["whale alert", "transferred to exchange", "large deposit"],
        "affects": {
            "BTC/USDT": {"direction": "negative", "magnitude": "medium",   "delay_seconds": 300},
        },
    },
}


# ---------------------------------------------------------------------------
# Match result
# ---------------------------------------------------------------------------

@dataclass
class AssetImpact:
    """Expected impact on a single instrument from a matched news pattern."""
    asset: str
    direction: str          # "positive", "negative", "variable"
    magnitude: ImpactLevel
    delay_seconds: int


@dataclass
class CorrelationMatch:
    """
    Result of matching a NewsItem against the correlation map.

    One NewsItem can match multiple patterns (e.g., a headline about "Trump
    tariff on Chinese tech" could hit both trump_tariff and crypto_regulation_negative).
    The caller receives ALL matches and picks the highest magnitude per asset.
    """
    pattern_name: str
    matched_keyword: str
    impacts: list[AssetImpact] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Matcher
# ---------------------------------------------------------------------------

class NewsAssetMatcher:
    """
    Scans NewsItems against NEWS_ASSET_MAP and populates `affected_assets`
    and upgrades `impact_level`.

    Thread-safe (stateless — all state is on the NewsItem itself).
    """

    def __init__(self, asset_map: dict[str, dict[str, Any]] | None = None) -> None:
        self._map = asset_map if asset_map is not None else NEWS_ASSET_MAP
        # Pre-compile lowercased keywords for faster matching.
        self._compiled: list[tuple[str, dict[str, Any], list[str]]] = []
        for name, pattern in self._map.items():
            kws = [kw.lower() for kw in pattern.get("keywords", [])]
            self._compiled.append((name, pattern, kws))

    def match_item(self, item: NewsItem) -> list[CorrelationMatch]:
        """
        Match one NewsItem against all patterns. Returns list of matches
        (may be empty if no keywords hit).
        """
        text = f"{item.title} {item.content}".lower()
        matches: list[CorrelationMatch] = []

        for name, pattern, keywords in self._compiled:
            # Check source filter first (cheap gate).
            source_filter = pattern.get("source_filter")
            if source_filter and not self._source_matches(item.source, source_filter):
                continue

            # Check keywords (OR logic — any hit is a match).
            hit_kw = self._first_keyword_hit(text, keywords)
            if hit_kw is None:
                continue

            # Build impacts.
            impacts: list[AssetImpact] = []
            for asset, spec in pattern.get("affects", {}).items():
                impacts.append(AssetImpact(
                    asset=asset,
                    direction=spec["direction"],
                    magnitude=_MAGNITUDE_MAP.get(spec["magnitude"], ImpactLevel.LOW),
                    delay_seconds=spec.get("delay_seconds", 60),
                ))

            matches.append(CorrelationMatch(
                pattern_name=name,
                matched_keyword=hit_kw,
                impacts=impacts,
            ))

        return matches

    def enrich_item(self, item: NewsItem) -> list[CorrelationMatch]:
        """
        Match AND mutate: populate `item.affected_assets` and upgrade
        `item.impact_level` based on all matching patterns.

        Returns the matches for logging/audit. The item is modified in place.
        """
        matches = self.match_item(item)
        if not matches:
            return []

        # Collect all affected assets and the highest magnitude across all matches.
        assets: set[str] = set(item.affected_assets)
        max_impact = item.impact_level

        for m in matches:
            for impact in m.impacts:
                assets.add(impact.asset)
                if impact.magnitude.rank > max_impact.rank:
                    max_impact = impact.magnitude

        item.affected_assets = sorted(assets)
        item.impact_level = max_impact

        logger.debug(
            "Correlation match: '%s' → patterns=%s assets=%s impact=%s",
            item.title[:80],
            [m.pattern_name for m in matches],
            item.affected_assets,
            item.impact_level.value,
        )
        return matches

    def enrich_items(self, items: list[NewsItem]) -> dict[str, list[CorrelationMatch]]:
        """
        Batch-enrich a list of items. Returns a dict mapping item titles to
        their matches (for downstream logging / Phase 1D reactive decisions).
        """
        all_matches: dict[str, list[CorrelationMatch]] = {}
        enriched_count = 0
        for item in items:
            matches = self.enrich_item(item)
            if matches:
                all_matches[item.title] = matches
                enriched_count += 1
        if enriched_count:
            logger.info(
                "Correlation map enriched %d/%d items with asset mappings",
                enriched_count, len(items),
            )
        return all_matches

    # ---------------------------------------------------------------- helpers

    @staticmethod
    def _source_matches(source: str, filters: list[str]) -> bool:
        """Check if the item's source string contains any of the filter substrings."""
        src_lower = source.lower()
        return any(f.lower() in src_lower for f in filters)

    @staticmethod
    def _first_keyword_hit(text: str, keywords: list[str]) -> str | None:
        """Return the first keyword found in text, or None."""
        for kw in keywords:
            if kw in text:
                return kw
        return None


# ---------------------------------------------------------------------------
# Convenience: get all assets affected by a set of matches
# ---------------------------------------------------------------------------

def get_affected_assets(matches: list[CorrelationMatch]) -> dict[str, AssetImpact]:
    """
    Flatten a list of matches into a dict of asset → highest-magnitude impact.
    Used by Phase 1D to decide which instruments to run reactive analysis on.
    """
    best: dict[str, AssetImpact] = {}
    for m in matches:
        for impact in m.impacts:
            existing = best.get(impact.asset)
            if existing is None or impact.magnitude.rank > existing.magnitude.rank:
                best[impact.asset] = impact
    return best
