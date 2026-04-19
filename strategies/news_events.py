"""
News & Events Module
Checks for high-impact economic events that could cause volatility.
Uses free CryptoPanic API or manual event schedule.
"""

import requests
import logging
from datetime import datetime, timezone, timedelta

logger = logging.getLogger(__name__)

# Major recurring events (day of week, rough UTC time)
# These cause high volatility — bot should either:
# 1. Avoid trading 15min before/after, OR
# 2. Wait for the move and trade the reaction
KNOWN_EVENTS = {
    "CPI": {"impact": "high", "pause_minutes": 15},
    "FOMC": {"impact": "high", "pause_minutes": 30},
    "NFP": {"impact": "high", "pause_minutes": 15},
    "PPI": {"impact": "medium", "pause_minutes": 10},
    "Unemployment": {"impact": "medium", "pause_minutes": 10},
    "GDP": {"impact": "medium", "pause_minutes": 10},
    "PCE": {"impact": "high", "pause_minutes": 15},
}


def fetch_crypto_news(limit: int = 10):
    """
    Fetch latest crypto news from CryptoPanic (free tier).
    Returns list of news items with sentiment.
    """
    try:
        url = "https://cryptopanic.com/api/free/v1/posts/"
        params = {
            "auth_token": "free",  # Replace with your API key for better results
            "kind": "news",
            "filter": "important",
            "public": "true",
        }
        resp = requests.get(url, params=params, timeout=10)
        if resp.ok:
            data = resp.json()
            results = []
            for post in data.get("results", [])[:limit]:
                results.append({
                    "title": post.get("title", ""),
                    "source": post.get("source", {}).get("title", ""),
                    "url": post.get("url", ""),
                    "published": post.get("published_at", ""),
                    "currencies": [c.get("code", "") for c in post.get("currencies", [])],
                    "votes": post.get("votes", {}),
                })
            return results
    except Exception as e:
        logger.warning(f"Could not fetch crypto news: {e}")

    return []


def check_high_impact_events():
    """
    Check if a high-impact event is happening soon.
    Returns True if bot should pause or be cautious.

    In production, integrate with:
    - ForexFactory API / scraper
    - Investing.com economic calendar
    - Custom calendar you maintain
    """
    # Placeholder — in production, check against a real calendar
    # For now, return a structure you can populate
    return {
        "event_near": False,
        "event_name": None,
        "pause_until": None,
        "impact": None,
    }


def analyze_news_sentiment(news_items: list, symbol: str):
    """
    Simple sentiment analysis based on news volume and votes.
    """
    ticker = symbol.split("/")[0]  # BTC from BTC/USDT

    relevant = [n for n in news_items if ticker in n.get("currencies", [])]

    if not relevant:
        return {"sentiment": "neutral", "news_count": 0, "items": []}

    # Count positive vs negative votes
    positive = sum(n.get("votes", {}).get("positive", 0) for n in relevant)
    negative = sum(n.get("votes", {}).get("negative", 0) for n in relevant)

    if positive > negative * 1.5:
        sentiment = "bullish"
    elif negative > positive * 1.5:
        sentiment = "bearish"
    else:
        sentiment = "neutral"

    return {
        "sentiment": sentiment,
        "news_count": len(relevant),
        "positive_votes": positive,
        "negative_votes": negative,
        "items": relevant[:3],
    }
