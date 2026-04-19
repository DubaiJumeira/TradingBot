"""
Reddit adapter via PRAW.

Docs: https://praw.readthedocs.io/
Auth: script-type OAuth app. Set REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET,
REDDIT_USER_AGENT in .env. If any are missing, the source is disabled.

TRADING LOGIC NOTE
------------------
Reddit sentiment is noisy and lagging — by the time WSB is piling into a
ticker, smart money has already positioned. We weight it at 0.2 credibility
for that reason. The value of this source is NOT the individual post
sentiment; it's the AGGREGATE mood shift that Phase 1B's sentiment velocity
tracker will eventually pick up. Think of it as a contrarian indicator
bucket, not a signal.

We pull "hot" from each configured subreddit rather than "new" to filter
out noise automatically (Reddit's vote system does some of the spam work
for us before FinBERT even sees the text).
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Any, ClassVar

from ..types import NewsItem, SourceKind
from .base import NewsSource

logger = logging.getLogger(__name__)


class RedditSource(NewsSource):
    kind: ClassVar[SourceKind] = SourceKind.REDDIT
    default_credibility: ClassVar[float] = 0.2
    cache_ttl_seconds: ClassVar[int] = 300

    def __init__(self, subreddits: list[str] | None = None,
                 *, posts_per_sub: int = 15, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        env_subs = os.getenv("REDDIT_SUBREDDITS", "wallstreetbets,cryptocurrency,stocks")
        self.subreddits = subreddits or [s.strip() for s in env_subs.split(",") if s.strip()]
        self.posts_per_sub = posts_per_sub
        self.client_id = os.getenv("REDDIT_CLIENT_ID", "")
        self.client_secret = os.getenv("REDDIT_CLIENT_SECRET", "")
        self.user_agent = os.getenv("REDDIT_USER_AGENT", "trading-bot/0.1")

    def is_configured(self) -> bool:
        return bool(self.client_id and self.client_secret and self.user_agent)

    def _fetch_raw(self) -> list[NewsItem]:
        try:
            import praw  # lazy — optional dep
        except ImportError:
            logger.warning("praw not installed; Reddit source disabled until pip install praw")
            return []

        reddit = praw.Reddit(
            client_id=self.client_id,
            client_secret=self.client_secret,
            user_agent=self.user_agent,
            check_for_async=False,
        )
        reddit.read_only = True

        items: list[NewsItem] = []
        for sub_name in self.subreddits:
            try:
                subreddit = reddit.subreddit(sub_name)
                for submission in subreddit.hot(limit=self.posts_per_sub):
                    items.append(self._make_item(sub_name, submission))
            except Exception as exc:  # noqa: BLE001
                logger.warning("Reddit fetch for r/%s failed: %s", sub_name, exc)
        return items

    def _make_item(self, sub_name: str, submission: Any) -> NewsItem:
        title = (getattr(submission, "title", "") or "").strip()
        selftext = (getattr(submission, "selftext", "") or "").strip()
        created = getattr(submission, "created_utc", None)
        if created is not None:
            published_at = datetime.fromtimestamp(float(created), tz=timezone.utc)
        else:
            published_at = datetime.now(tz=timezone.utc)

        score = int(getattr(submission, "score", 0) or 0)
        num_comments = int(getattr(submission, "num_comments", 0) or 0)

        # Boost credibility slightly for highly-upvoted posts — broad agreement
        # is a weak but real signal the post isn't pure noise. Cap at 0.35 so
        # Reddit can never out-credibility an actual wire.
        boosted = self.default_credibility + min(0.15, score / 10_000)

        permalink = getattr(submission, "permalink", "")
        return NewsItem(
            source=f"reddit:r/{sub_name}",
            title=title,
            content=selftext or title,
            published_at=published_at,
            source_credibility=boosted,
            url=f"https://reddit.com{permalink}" if permalink else "",
            raw_data={
                "subreddit": sub_name,
                "score": score,
                "num_comments": num_comments,
                "id": getattr(submission, "id", ""),
            },
        )
