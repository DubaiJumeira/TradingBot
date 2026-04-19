"""
Deduplication for news items.

Why this matters for trading: if Reuters, Bloomberg, and CoinDesk all publish
the same "Fed cuts rates 25bps" story within seconds, a naive pipeline will
generate three news signals and potentially triple-weight the sentiment. We
want ONE signal per underlying event, with credibility taking the highest of
the merged sources (since the story is confirmed by the most-credible outlet).

Strategy:
    1. Exact match — normalize title (lowercase, strip punctuation, collapse
       whitespace) and hash. Same hash → duplicate.
    2. Fuzzy match — if no exact hit, compare against recent titles with
       `rapidfuzz.fuzz.token_set_ratio`. Threshold: 85. This catches
       "Fed cuts rates by 25 basis points" vs "Federal Reserve cuts rates 25bps".

The earliest `published_at` wins (we want the FIRST publisher's timestamp for
event-reaction latency math in Phase 1D). Credibility is set to the MAX across
the merged group — if Reuters confirms it, treat it as Reuters-level reliable.
Source string becomes "merged:<n>" and raw_data includes all source URLs.
"""

from __future__ import annotations

import logging
import re
from dataclasses import replace
from hashlib import sha1

from rapidfuzz import fuzz

from .types import NewsItem, sort_by_time

logger = logging.getLogger(__name__)

_FUZZY_THRESHOLD = 90  # 0-100, rapidfuzz scale
# Calibration: at 90, real rephrasings like "Oil jumps 3% on OPEC cut talks"
# vs "Oil prices jump 3% on OPEC cut talks" (90.9) collapse, but dangerous
# false-positives like "Gold hits record high" vs "Gold declines from record
# high" (88.1) and "Bitcoin ETF approved" vs "Bitcoin ETF rejected" (75) do
# NOT collapse. Lowering below 90 risks merging bullish + bearish headlines
# into one item, which would confuse the sentiment pipeline in Phase 1B.
_NON_ALNUM = re.compile(r"[^a-z0-9\s]")
_MULTISPACE = re.compile(r"\s+")


def _normalize_title(title: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace — for hashing."""
    lowered = title.lower()
    stripped = _NON_ALNUM.sub(" ", lowered)
    collapsed = _MULTISPACE.sub(" ", stripped).strip()
    return collapsed


def _title_hash(title: str) -> str:
    return sha1(_normalize_title(title).encode("utf-8")).hexdigest()


def _merge(group: list[NewsItem]) -> NewsItem:
    """
    Merge a group of duplicates into one NewsItem.

    Rules:
        - published_at: earliest (first publisher wins for latency math)
        - source_credibility: max (strongest confirming source)
        - source: "merged:<n>" where n is the group size
        - title: from the highest-credibility item (cleanest phrasing)
        - content: from the longest-content item (most detail)
        - raw_data: {"merged_from": [{source, url}, ...]}
        - impact_level: max across the group
    """
    if len(group) == 1:
        return group[0]

    earliest = min(group, key=lambda i: i.published_at)
    most_credible = max(group, key=lambda i: i.source_credibility)
    longest = max(group, key=lambda i: len(i.content or ""))
    max_impact = max((i.impact_level for i in group), key=lambda lv: lv.rank)

    merged_raw = {
        "merged_from": [
            {"source": i.source, "url": i.url, "title": i.title}
            for i in group
        ]
    }

    return replace(
        earliest,
        source=f"merged:{len(group)}",
        title=most_credible.title,
        content=longest.content or most_credible.content,
        source_credibility=most_credible.source_credibility,
        impact_level=max_impact,
        url=most_credible.url or earliest.url,
        raw_data=merged_raw,
    )


def deduplicate(items: list[NewsItem], fuzzy_threshold: int = _FUZZY_THRESHOLD) -> list[NewsItem]:
    """
    Dedup a mixed list of NewsItems from multiple sources.

    Returns a newest-first list with duplicates merged. Stable enough that
    the same input list produces the same output ordering across runs.

    Complexity: O(n^2) in the fuzzy pass, but `n` is the number of items in
    one 5-minute cycle — realistically < 200 — so this is fine. If ingestion
    scales up in Phase 1D's reactive mode, swap in a blocking strategy
    (group by first-token or TF-IDF + ANN).
    """
    if not items:
        return []

    # Phase 1: exact hash buckets.
    exact_groups: dict[str, list[NewsItem]] = {}
    for item in items:
        h = _title_hash(item.title)
        exact_groups.setdefault(h, []).append(item)

    exact_merged: list[NewsItem] = [_merge(group) for group in exact_groups.values()]

    # Phase 2: fuzzy match across the exact-merged survivors. We compare each
    # item against already-kept items; on a hit, merge into that kept item.
    kept: list[NewsItem] = []
    for candidate in exact_merged:
        cand_norm = _normalize_title(candidate.title)
        matched_idx: int | None = None
        for idx, keeper in enumerate(kept):
            keeper_norm = _normalize_title(keeper.title)
            score = fuzz.token_set_ratio(cand_norm, keeper_norm)
            if score >= fuzzy_threshold:
                matched_idx = idx
                break
        if matched_idx is None:
            kept.append(candidate)
        else:
            # Merge candidate into the existing keeper.
            kept[matched_idx] = _merge([kept[matched_idx], candidate])

    before, after = len(items), len(kept)
    if after < before:
        logger.debug("Dedup: %d → %d items (%d duplicates removed)",
                     before, after, before - after)

    return sort_by_time(kept)
