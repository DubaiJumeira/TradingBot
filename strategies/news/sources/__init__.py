"""Source adapters. Each is independently togglable via env flags."""

from .base import NewsSource
from .cryptopanic import CryptoPanicSource
from .fear_greed import FearGreedSource
from .forexlive_rss import ForexLiveSource
from .newsapi import NewsAPISource
from .reddit import RedditSource
from .rss import GenericRSSSource
from .twitter import TwitterSource

__all__ = [
    "NewsSource",
    "CryptoPanicSource",
    "NewsAPISource",
    "GenericRSSSource",
    "ForexLiveSource",
    "FearGreedSource",
    "TwitterSource",
    "RedditSource",
]
