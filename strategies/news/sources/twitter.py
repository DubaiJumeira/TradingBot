"""
X/Twitter adapter with two backends behind one NewsSource interface.

PRIMARY: tweepy + X API v2 Bearer token
----------------------------------------
The clean path. Set TWITTER_BEARER in .env. tweepy handles pagination,
rate-limit headers, and error types for us. Cost: X API Basic tier.

FALLBACK: Apify scraper
-----------------------
When TWITTER_BEARER is empty, we fall back to an Apify actor that scrapes
profile timelines. Apify has a free tier that can comfortably poll a small
list of accounts every few minutes. Set APIFY_TOKEN (and optionally
APIFY_TWITTER_ACTOR) to enable this path.

If BOTH are unset, the source is disabled (returns [] silently).

CREDIBILITY MODEL
-----------------
Per-account credibility is the whole game here — a random crypto Twitter
account has ~0.3 credibility, but @realDonaldTrump gets 0.95 because his
tweets literally move markets whether they're true or not. The account list
lives in `HIGH_IMPACT_ACCOUNTS` below; Phase 1C's correlation map reads the
`source` string (which includes the handle) to filter patterns like
`trump_tariff` to only trigger on POTUS tweets.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, ClassVar

import requests

from ..types import NewsItem, SourceKind, as_utc
from .base import NewsSource

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class HighImpactAccount:
    """One monitored account + its credibility weight + what it moves."""
    handle: str          # without the '@'
    credibility: float
    description: str     # why it's on the list — for audit/logs


# Tuned per the master-prompt's account list. Credibility calibrated to the
# spec: verified wires > track-record financial accounts > general crypto > rest.
HIGH_IMPACT_ACCOUNTS: tuple[HighImpactAccount, ...] = (
    HighImpactAccount("realDonaldTrump", 0.95, "POTUS — tariffs, wars, oil, crypto reserves"),
    HighImpactAccount("POTUS",           0.95, "Official US presidency account"),
    HighImpactAccount("WhiteHouse",      0.90, "White House comms"),
    HighImpactAccount("DeItaone",        0.90, "Fastest financial news aggregator on X"),
    HighImpactAccount("unusual_whales",  0.85, "Options flow + breaking financial news"),
    HighImpactAccount("federalreserve",  1.00, "FOMC / Fed statements"),
    HighImpactAccount("WhaleAlert",      0.70, "Large crypto transfers"),
    HighImpactAccount("elikinosworld",   0.60, "Trump Truth Social reposts to X"),
)


class TwitterSource(NewsSource):
    kind: ClassVar[SourceKind] = SourceKind.TWITTER
    default_credibility: ClassVar[float] = 0.7  # placeholder; per-account overrides
    cache_ttl_seconds: ClassVar[int] = 60  # tweets matter in seconds, not minutes

    def __init__(self, accounts: tuple[HighImpactAccount, ...] | None = None,
                 *, tweets_per_account: int = 5, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.accounts = accounts or HIGH_IMPACT_ACCOUNTS
        self.tweets_per_account = tweets_per_account
        self.bearer = os.getenv("TWITTER_BEARER", "")
        self.apify_token = os.getenv("APIFY_TOKEN", "")
        self.apify_actor = os.getenv("APIFY_TWITTER_ACTOR", "apidojo/twitter-scraper-lite")

    def is_configured(self) -> bool:
        return bool(self.bearer) or bool(self.apify_token)

    def _fetch_raw(self) -> list[NewsItem]:
        if self.bearer:
            return self._fetch_via_tweepy()
        return self._fetch_via_apify()

    # ------------------------------------------------------------- tweepy

    def _fetch_via_tweepy(self) -> list[NewsItem]:
        """
        Primary path: X API v2 via tweepy.

        One API call per account per cycle. With ~8 accounts and the 60s
        cache TTL, that's ~480 calls/hour worst case — well under the Basic
        tier's ~10k/month budget if you also use Phase 1D's reactive refresh
        sparingly.
        """
        try:
            import tweepy  # lazy — optional dep
        except ImportError:
            logger.warning("tweepy not installed; Twitter source disabled until pip install tweepy")
            return []

        client = tweepy.Client(bearer_token=self.bearer, wait_on_rate_limit=False)
        items: list[NewsItem] = []
        for account in self.accounts:
            try:
                user = client.get_user(username=account.handle)
                if not user or not getattr(user, "data", None):
                    continue
                user_id = user.data.id
                tweets = client.get_users_tweets(
                    id=user_id,
                    max_results=max(5, self.tweets_per_account),
                    tweet_fields=["created_at", "text", "public_metrics"],
                    exclude=["retweets", "replies"],
                )
                for tweet in (getattr(tweets, "data", None) or [])[: self.tweets_per_account]:
                    items.append(self._make_item(account, {
                        "id": str(tweet.id),
                        "text": tweet.text,
                        "created_at": tweet.created_at,
                        "metrics": getattr(tweet, "public_metrics", None) or {},
                    }))
            except Exception as exc:  # noqa: BLE001
                # Per-account isolation. Rate-limit errors bubble up as tweepy.TooManyRequests;
                # we swallow and continue so one account's limit doesn't blank the whole batch.
                logger.warning("Twitter fetch for @%s failed: %s", account.handle, exc)
        return items

    # -------------------------------------------------------------- apify

    def _fetch_via_apify(self) -> list[NewsItem]:
        """
        Fallback path: run a synchronous Apify actor and read its dataset.

        This is intentionally minimal — Apify's sync-run endpoint supports
        passing an input JSON and returning the dataset items in one call.
        The exact input shape depends on the actor; the default
        `apidojo/twitter-scraper-lite` accepts `handles` and `tweetsPerQuery`.
        """
        handles = [a.handle for a in self.accounts]
        url = f"https://api.apify.com/v2/acts/{self.apify_actor.replace('/', '~')}/run-sync-get-dataset-items"
        try:
            resp = requests.post(
                url,
                params={"token": self.apify_token},
                json={
                    "handles": handles,
                    "tweetsPerQuery": self.tweets_per_account,
                    "mode": "user",
                },
                timeout=45,
            )
            resp.raise_for_status()
            records = resp.json()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Apify Twitter scrape failed: %s", exc)
            return []

        by_handle: dict[str, HighImpactAccount] = {a.handle.lower(): a for a in self.accounts}
        items: list[NewsItem] = []
        for record in records:
            handle = (record.get("author") or record.get("username") or "").lstrip("@").lower()
            account = by_handle.get(handle)
            if account is None:
                continue
            items.append(self._make_item(account, {
                "id": str(record.get("id") or record.get("tweetId") or ""),
                "text": record.get("text") or record.get("fullText") or "",
                "created_at": record.get("createdAt") or record.get("created_at") or "",
                "metrics": {
                    "like_count": record.get("likeCount"),
                    "retweet_count": record.get("retweetCount"),
                },
            }))
        return items

    # ------------------------------------------------------------- shared

    def _make_item(self, account: HighImpactAccount, payload: dict[str, Any]) -> NewsItem:
        text = (payload.get("text") or "").strip()
        return NewsItem(
            source=f"twitter:@{account.handle}",
            title=text[:280],
            content=text,
            published_at=as_utc(payload.get("created_at") or ""),
            source_credibility=account.credibility,
            url=f"https://x.com/{account.handle}/status/{payload.get('id', '')}",
            raw_data={
                "handle": account.handle,
                "description": account.description,
                "metrics": payload.get("metrics", {}),
                "tweet_id": payload.get("id"),
            },
        )
