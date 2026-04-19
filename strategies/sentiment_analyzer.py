"""
Phase 1B — AI-Powered Sentiment Analysis

This module enriches NewsItems from Phase 1A with sentiment scores and
provides per-asset aggregate sentiment, velocity tracking, and spam filtering.

ARCHITECTURE
------------
Two scoring backends behind one interface:

    PRIMARY:  FinBERT (ProsusAI/finbert)
        Financial-domain transformer fine-tuned for earnings calls, analyst
        notes, and financial headlines. Understands that "cuts rates" is
        positive for equities and that "inflation accelerated" is negative.
        Requires torch + ~500MB model download + ~2GB runtime RAM.

    FALLBACK: VADER (Valence Aware Dictionary and sEntiment Reasoner)
        Lexicon + rule-based, ships with the vaderSentiment package, runs in
        microseconds. Good for tweet-length text with explicit sentiment words
        ("amazing", "crash"), weak for domain-specific language ("dovish pivot").
        Used when FinBERT is unavailable AND always used for tweets (speed).

TRADING LOGIC
-------------
Individual item scores are NOT trade signals. The actionable outputs are:

    1. `aggregate_sentiment(items, asset)` — credibility-weighted mean score
       for a single asset. Phase 1C maps news items to assets; this function
       then tells the signal generator the NET directional pressure.

    2. `sentiment_velocity(asset)` — rate of change of aggregate sentiment.
       A sudden spike from neutral to strongly-bullish is a much stronger
       signal than steady bullishness (which the market has already priced).
       Phase 1D's reactive mode triggers on velocity, not level.

    3. `filter_spam(items)` — removes low-quality social media items before
       they pollute the aggregate. Without this, a botnet shilling a token
       can push aggregate sentiment artificially positive.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from strategies.news.types import ImpactLevel, NewsItem

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Spam / bot filtering thresholds
# ---------------------------------------------------------------------------
_MIN_ACCOUNT_AGE_DAYS = 30
_MAX_POSTS_PER_DAY = 50


# ---------------------------------------------------------------------------
# FinBERT wrapper (lazy-loaded, graceful fallback)
# ---------------------------------------------------------------------------

class _FinBERTScorer:
    """
    Thin wrapper around the HuggingFace FinBERT pipeline.

    Loads the model lazily on first call. If torch or the model aren't
    available, `score()` returns None and the caller falls back to VADER.

    FinBERT output: {"label": "positive"|"negative"|"neutral", "score": 0-1}
    We convert to [-1.0, +1.0]: positive → +score, negative → −score,
    neutral → score * 0 (clamped toward zero but weighted by confidence).
    """

    def __init__(self) -> None:
        self._pipeline: Any = None
        self._available: bool | None = None  # None = not yet checked

    @property
    def available(self) -> bool:
        if self._available is not None:
            return self._available
        try:
            import torch  # noqa: F401
            from transformers import pipeline  # noqa: F401
            self._available = True
        except ImportError:
            logger.info(
                "FinBERT unavailable (torch or transformers not installed). "
                "Falling back to VADER for all sentiment scoring."
            )
            self._available = False
        return self._available

    def _load(self) -> None:
        if self._pipeline is not None:
            return
        from transformers import pipeline
        logger.info("Loading FinBERT model (ProsusAI/finbert)...")
        t0 = time.time()
        self._pipeline = pipeline(
            "sentiment-analysis",
            model="ProsusAI/finbert",
            tokenizer="ProsusAI/finbert",
            device=-1,  # CPU — no GPU assumed on a VPS
            top_k=None,  # return all labels with scores
        )
        logger.info("FinBERT loaded in %.1fs", time.time() - t0)

    def score(self, text: str) -> float | None:
        """
        Score a single text. Returns [-1.0, +1.0] or None if unavailable.

        Truncates to 512 tokens (FinBERT's context window). For long articles,
        only the headline matters for trading — the market reacts to headlines,
        not paragraph 7.
        """
        if not self.available:
            return None
        try:
            self._load()
            results = self._pipeline(text[:512], truncation=True)
            # `results` is a list of lists (one per input). We sent one input.
            scores_by_label = {}
            for entry in results[0] if isinstance(results[0], list) else results:
                scores_by_label[entry["label"].lower()] = entry["score"]
            pos = scores_by_label.get("positive", 0.0)
            neg = scores_by_label.get("negative", 0.0)
            # Net score: positive component minus negative component.
            # Neutral confidence pushes toward zero naturally.
            return round(pos - neg, 4)
        except Exception as exc:
            logger.warning("FinBERT scoring failed: %s — falling back to VADER", exc)
            return None


class _VADERScorer:
    """
    VADER wrapper. Always available (vaderSentiment is a pure-Python package).

    Uses the `compound` score which is already normalized to [-1, +1].
    """

    def __init__(self) -> None:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        self._analyzer = SentimentIntensityAnalyzer()

    def score(self, text: str) -> float:
        result = self._analyzer.polarity_scores(text)
        return round(result["compound"], 4)


# ---------------------------------------------------------------------------
# Sentiment history tracking (for velocity calculation)
# ---------------------------------------------------------------------------

@dataclass
class _SentimentSnapshot:
    """One point on the sentiment timeline for an asset."""
    timestamp: datetime
    score: float
    item_count: int


@dataclass
class _AssetSentimentHistory:
    """Rolling window of sentiment snapshots for velocity calculation."""
    snapshots: list[_SentimentSnapshot] = field(default_factory=list)
    max_window: timedelta = field(default_factory=lambda: timedelta(hours=6))

    def add(self, score: float, item_count: int) -> None:
        now = datetime.now(tz=timezone.utc)
        self.snapshots.append(_SentimentSnapshot(now, score, item_count))
        self._prune()

    def velocity(self) -> float:
        """
        Rate of sentiment change over the last 30 minutes.

        Returns a value in [-2.0, +2.0] (theoretical max: a swing from -1.0
        to +1.0 in one window). Positive = getting MORE bullish. Negative =
        getting MORE bearish.

        TRADING LOGIC: A velocity of ±0.3+ in 30 minutes is a significant
        sentiment shift — Phase 1D should check for ICT setups aligned with
        the sentiment direction.
        """
        if len(self.snapshots) < 2:
            return 0.0
        window = timedelta(minutes=30)
        now = datetime.now(tz=timezone.utc)
        cutoff = now - window
        recent = [s for s in self.snapshots if s.timestamp >= cutoff]
        if len(recent) < 2:
            # Not enough data in the window; use oldest vs newest.
            return self.snapshots[-1].score - self.snapshots[0].score
        return recent[-1].score - recent[0].score

    def _prune(self) -> None:
        cutoff = datetime.now(tz=timezone.utc) - self.max_window
        self.snapshots = [s for s in self.snapshots if s.timestamp >= cutoff]


# ---------------------------------------------------------------------------
# Main analyzer class
# ---------------------------------------------------------------------------

class SentimentAnalyzer:
    """
    Enriches NewsItems with sentiment scores and provides per-asset aggregates.

    Typical usage (called from the bot's main cycle or Phase 1D reactive mode):

        analyzer = SentimentAnalyzer()
        items = aggregator.fetch_all()
        enriched = analyzer.analyze_items(items)
        agg = analyzer.aggregate_sentiment(enriched, "XAUUSD")
        velocity = analyzer.sentiment_velocity("XAUUSD")
    """

    def __init__(self, use_finbert: bool = True) -> None:
        self._finbert = _FinBERTScorer() if use_finbert else None
        self._vader = _VADERScorer()
        self._history: dict[str, _AssetSentimentHistory] = defaultdict(_AssetSentimentHistory)
        logger.info(
            "SentimentAnalyzer initialized (FinBERT %s, VADER ready)",
            "available" if self._finbert and self._finbert.available else "unavailable — using VADER only",
        )

    # ------------------------------------------------------------------ scoring

    def analyze_items(self, items: list[NewsItem]) -> list[NewsItem]:
        """
        Enrich each item's `sentiment_score` in place and return the list.

        Routing:
            - Tweets → always VADER (speed matters; FinBERT is overkill for 280 chars)
            - Everything else → FinBERT if available, else VADER
        """
        for item in items:
            if self._is_tweet(item):
                item.sentiment_score = self._vader.score(item.title)
            else:
                finbert_score = self._finbert.score(item.title) if self._finbert else None
                item.sentiment_score = finbert_score if finbert_score is not None else self._vader.score(item.title)
        return items

    # ----------------------------------------------------------- aggregation

    def aggregate_sentiment(self, items: list[NewsItem], asset: str) -> dict[str, Any]:
        """
        Credibility-weighted mean sentiment for a single asset.

        Formula from the spec:
            final_sentiment = sum(score * credibility) / sum(credibility)

        Only considers items whose `affected_assets` includes `asset`.
        Phase 1C populates that field; until then, this returns neutral for
        everything (which is the safe default — no phantom signals).

        Returns:
            {
                "asset": str,
                "score": float,           # [-1, +1] weighted mean
                "item_count": int,
                "strongest_source": str,   # highest-credibility contributor
                "direction": str,          # "bullish" / "bearish" / "neutral"
            }
        """
        relevant = [i for i in items if asset in i.affected_assets]
        if not relevant:
            return {
                "asset": asset,
                "score": 0.0,
                "item_count": 0,
                "strongest_source": "",
                "direction": "neutral",
            }

        weighted_sum = sum(i.sentiment_score * i.source_credibility for i in relevant)
        weight_total = sum(i.source_credibility for i in relevant)
        final = weighted_sum / weight_total if weight_total > 0 else 0.0
        final = max(-1.0, min(1.0, final))

        strongest = max(relevant, key=lambda i: i.source_credibility)

        # Update history for velocity tracking.
        self._history[asset].add(final, len(relevant))

        if final > 0.15:
            direction = "bullish"
        elif final < -0.15:
            direction = "bearish"
        else:
            direction = "neutral"

        return {
            "asset": asset,
            "score": round(final, 4),
            "item_count": len(relevant),
            "strongest_source": strongest.source,
            "direction": direction,
        }

    # ----------------------------------------------------------- velocity

    def sentiment_velocity(self, asset: str) -> float:
        """
        Rate of sentiment change for `asset` over the last 30 minutes.

        Positive = getting more bullish. Negative = getting more bearish.
        Magnitude > 0.3 is a significant shift worth reacting to.
        """
        return self._history[asset].velocity()

    # --------------------------------------------------------- spam filter

    def filter_spam(self, items: list[NewsItem]) -> list[NewsItem]:
        """
        Remove low-quality social-media items that would pollute the aggregate.

        Checks (applied only to Twitter and Reddit sources):
            1. Account age < 30 days → spam
            2. Post rate > 50/day → bot
            3. No meaningful text (< 10 chars) → noise

        Wire services (RSS, NewsAPI, CryptoPanic, ForexLive) are never filtered.

        TRADING LOGIC: Without this filter, a coordinated shill campaign can
        push a token's aggregate sentiment from neutral to strongly-bullish,
        which Phase 1D would interpret as a real signal. The credibility
        weighting (Reddit=0.2) dampens this, but doesn't eliminate it when
        there are hundreds of bot posts.
        """
        kept: list[NewsItem] = []
        filtered_count = 0
        for item in items:
            if not self._is_social_media(item):
                kept.append(item)
                continue
            if self._is_spam(item):
                filtered_count += 1
                continue
            kept.append(item)
        if filtered_count:
            logger.info("Spam filter removed %d/%d social media items", filtered_count, len(items))
        return kept

    # --------------------------------------------------------- internals

    @staticmethod
    def _is_tweet(item: NewsItem) -> bool:
        return item.source.startswith("twitter:")

    @staticmethod
    def _is_social_media(item: NewsItem) -> bool:
        return item.source.startswith("twitter:") or item.source.startswith("reddit:")

    @staticmethod
    def _is_spam(item: NewsItem) -> bool:
        """
        Check raw_data for spam indicators.

        Twitter adapter stores `handle` and `metrics` in raw_data.
        Reddit adapter stores `score` and `num_comments`.
        If metadata is absent, we give the item the benefit of the doubt.
        """
        raw = item.raw_data

        # Text too short to carry real information.
        if len(item.title.strip()) < 10:
            return True

        # Account age check (Twitter — Apify path may include account_created).
        account_created = raw.get("account_created")
        if account_created is not None:
            try:
                from strategies.news.types import as_utc
                created = as_utc(account_created)
                age = datetime.now(tz=timezone.utc) - created
                if age < timedelta(days=_MIN_ACCOUNT_AGE_DAYS):
                    return True
            except Exception:
                pass  # Can't parse → don't filter

        # Post rate check (if the source provides it).
        posts_per_day = raw.get("posts_per_day")
        if posts_per_day is not None:
            try:
                if float(posts_per_day) > _MAX_POSTS_PER_DAY:
                    return True
            except (TypeError, ValueError):
                pass

        # Reddit: very low score is likely noise (downvoted or ignored).
        if item.source.startswith("reddit:"):
            score = raw.get("score", 1)
            try:
                if int(score) <= 0:
                    return True
            except (TypeError, ValueError):
                pass

        return False
