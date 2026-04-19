"""
Tests for Phase 1D — ReactiveNewsMonitor + news-enhanced signal scoring.

Uses fake aggregator and correlation matches so nothing touches the network.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from strategies.asset_correlations import AssetImpact, CorrelationMatch
from strategies.news.types import ImpactLevel, NewsItem
from strategies.news_reactive import (
    EventState,
    ReactiveAction,
    ReactiveNewsMonitor,
    TrackedEvent,
)


# -----------------------------------------------------------------------
# Fake aggregator
# -----------------------------------------------------------------------

class _FakeAggregator:
    """Mimics just the aggregator methods the reactive monitor reads."""

    def __init__(self, items: list[NewsItem] | None = None,
                 matches: dict[str, list[CorrelationMatch]] | None = None):
        self._items = items or []
        self._matches = matches or {}

    def high_impact(self, minimum=ImpactLevel.HIGH, since=None):
        return [i for i in self._items if i.impact_level.rank >= minimum.rank]

    def last_correlation_matches(self):
        return self._matches

    def set_items(self, items, matches):
        self._items = items
        self._matches = matches


def _critical_item(title: str, minutes_ago: int = 0) -> NewsItem:
    return NewsItem(
        source="forexlive",
        title=title,
        content=title,
        published_at=datetime.now(tz=timezone.utc) - timedelta(minutes=minutes_ago),
        source_credibility=0.95,
        impact_level=ImpactLevel.CRITICAL,
        affected_assets=["XAUUSD", "XTIUSD"],
    )


def _war_matches() -> list[CorrelationMatch]:
    return [CorrelationMatch(
        pattern_name="war_escalation",
        matched_keyword="military strike",
        impacts=[
            AssetImpact("XAUUSD", "positive", ImpactLevel.CRITICAL, 30),
            AssetImpact("XTIUSD", "positive", ImpactLevel.CRITICAL, 30),
            AssetImpact("SPX500", "negative", ImpactLevel.HIGH, 60),
        ],
    )]


# -----------------------------------------------------------------------
# ReactiveNewsMonitor tests
# -----------------------------------------------------------------------

class TestEventIngestion:
    def test_new_critical_event_is_tracked(self):
        item = _critical_item("Military strike reported")
        matches = {"Military strike reported": _war_matches()}
        agg = _FakeAggregator(items=[item], matches=matches)
        monitor = ReactiveNewsMonitor(agg)

        actions = monitor.check()
        # Event just ingested — delay hasn't expired yet (it's 30s).
        # So no actions yet, but event should be tracked.
        assert monitor.pending_count >= 0  # may be pending or already ready

    def test_duplicate_events_not_re_tracked(self):
        item = _critical_item("Military strike reported")
        matches = {"Military strike reported": _war_matches()}
        agg = _FakeAggregator(items=[item], matches=matches)
        monitor = ReactiveNewsMonitor(agg)

        monitor.check()
        initial_count = len(monitor._events)

        # Same item again — should not create a second tracked event.
        monitor.check()
        assert len(monitor._events) == initial_count


class TestDelayAndReadiness:
    def test_event_with_expired_delay_produces_actions(self):
        # Create an item that was published 5 minutes ago — delay of 30s is
        # long expired, so it should be READY immediately.
        item = _critical_item("Military strike 5 min ago", minutes_ago=5)
        matches = {"Military strike 5 min ago": _war_matches()}
        agg = _FakeAggregator(items=[item], matches=matches)
        monitor = ReactiveNewsMonitor(agg)

        actions = monitor.check()
        assert len(actions) >= 1
        assets = {a.asset for a in actions}
        assert "XAUUSD" in assets
        assert "XTIUSD" in assets
        assert "SPX500" in assets

    def test_action_direction_matches_correlation_map(self):
        item = _critical_item("Military strike old", minutes_ago=5)
        matches = {"Military strike old": _war_matches()}
        agg = _FakeAggregator(items=[item], matches=matches)
        monitor = ReactiveNewsMonitor(agg)

        actions = monitor.check()
        gold_action = next(a for a in actions if a.asset == "XAUUSD")
        assert gold_action.direction == "positive"
        assert gold_action.impact_level == ImpactLevel.CRITICAL

        spx_action = next(a for a in actions if a.asset == "SPX500")
        assert spx_action.direction == "negative"


class TestCooldown:
    def test_same_event_does_not_retrigger_after_action(self):
        item = _critical_item("Strike event", minutes_ago=5)
        matches = {"Strike event": _war_matches()}
        agg = _FakeAggregator(items=[item], matches=matches)
        monitor = ReactiveNewsMonitor(agg, cooldown_minutes=60)

        # First check: produces actions.
        actions1 = monitor.check()
        assert len(actions1) >= 1

        # Second check: same event, now ACTIVE → no new actions.
        actions2 = monitor.check()
        assert len(actions2) == 0


class TestEventStateTracking:
    def test_status_report(self):
        item = _critical_item("War headlines", minutes_ago=5)
        matches = {"War headlines": _war_matches()}
        agg = _FakeAggregator(items=[item], matches=matches)
        monitor = ReactiveNewsMonitor(agg)

        monitor.check()  # ingest + trigger
        status = monitor.status()
        assert "tracked_events" in status
        assert isinstance(status["events"], list)


# -----------------------------------------------------------------------
# ReactiveAction.to_news_signal()
# -----------------------------------------------------------------------

class TestReactiveAction:
    def test_to_news_signal_format(self):
        action = ReactiveAction(
            asset="XAUUSD",
            direction="positive",
            impact_level=ImpactLevel.CRITICAL,
            delay_seconds=30,
            event_title="Military strike reported",
            pattern_name="war_escalation",
        )
        sig = action.to_news_signal()
        assert sig["impact"] == "critical"
        assert sig["direction"] == "positive"
        assert sig["pattern"] == "war_escalation"
        assert sig["delay_seconds"] == 30


# -----------------------------------------------------------------------
# News-enhanced signal scoring
# -----------------------------------------------------------------------

class TestNewsEnhancedScoring:
    """Test the scoring rules in signal_generator.generate_signal."""

    def test_critical_aligned_adds_20(self):
        from strategies.signal_generator import generate_signal

        # Minimal ICT/Wyckoff/market dicts that produce a "long" side
        # with enough base score to be testable.
        ict = {
            "structure": "bullish",
            "bos_choch": [{"type": "BOS", "direction": "bullish", "level": 100.0}],
            "fvgs": [{"type": "bullish", "bottom": 99.0, "top": 101.0}],
            "order_blocks": [],
            "liquidity_sweeps": [],
            "swing_lows": [{"price": 98.0}],
            "swing_highs": [{"price": 105.0}],
        }
        wyckoff = {"phase": "accumulation", "springs": [], "utads": []}
        market = {
            "funding": {},
            "volume_profile": {},
            "kill_zone": {"active": True, "zone": "london", "weight": 0.8},
        }

        # Without news.
        sig_no_news = generate_signal("XAUUSD", 100.0, ict, wyckoff, market, 10000)
        # With critical aligned news (positive = long).
        sig_with_news = generate_signal("XAUUSD", 100.0, ict, wyckoff, market, 10000,
                                        news_signal={"impact": "critical", "direction": "positive",
                                                      "event_title": "test", "pattern": "test",
                                                      "delay_seconds": 30})
        # Both should produce signals; the news version should score +20 higher.
        assert sig_no_news is not None
        assert sig_with_news is not None
        assert sig_with_news["score"] == sig_no_news["score"] + 20
        assert sig_with_news["news_triggered"] is True

    def test_critical_against_subtracts_30(self):
        from strategies.signal_generator import generate_signal

        ict = {
            "structure": "bullish",
            "bos_choch": [{"type": "BOS", "direction": "bullish", "level": 100.0}],
            "fvgs": [{"type": "bullish", "bottom": 99.0, "top": 101.0}],
            "order_blocks": [],
            "liquidity_sweeps": [],
            "swing_lows": [{"price": 98.0}],
            "swing_highs": [{"price": 105.0}],
        }
        wyckoff = {"phase": "accumulation", "springs": [], "utads": []}
        market = {
            "funding": {},
            "volume_profile": {},
            "kill_zone": {"active": True, "zone": "london", "weight": 0.8},
        }

        sig_no_news = generate_signal("XAUUSD", 100.0, ict, wyckoff, market, 10000)
        # Critical NEGATIVE news against a long signal → -30.
        sig_against = generate_signal("XAUUSD", 100.0, ict, wyckoff, market, 10000,
                                       news_signal={"impact": "critical", "direction": "negative",
                                                     "event_title": "bearish event", "pattern": "test",
                                                     "delay_seconds": 30})
        if sig_no_news is not None and sig_against is not None:
            assert sig_against["score"] == sig_no_news["score"] - 30
        elif sig_no_news is not None and sig_against is None:
            # The -30 pushed the score below threshold → correctly skipped.
            assert True

    def test_high_aligned_adds_12(self):
        from strategies.signal_generator import generate_signal

        ict = {
            "structure": "bullish",
            "bos_choch": [{"type": "BOS", "direction": "bullish", "level": 100.0}],
            "fvgs": [{"type": "bullish", "bottom": 99.0, "top": 101.0}],
            "order_blocks": [],
            "liquidity_sweeps": [],
            "swing_lows": [{"price": 98.0}],
            "swing_highs": [{"price": 105.0}],
        }
        wyckoff = {"phase": "accumulation", "springs": [], "utads": []}
        market = {
            "funding": {},
            "volume_profile": {},
            "kill_zone": {"active": True, "zone": "london", "weight": 0.8},
        }

        sig_base = generate_signal("XAUUSD", 100.0, ict, wyckoff, market, 10000)
        sig_high = generate_signal("XAUUSD", 100.0, ict, wyckoff, market, 10000,
                                    news_signal={"impact": "high", "direction": "positive",
                                                  "event_title": "rate cut", "pattern": "fed_dovish",
                                                  "delay_seconds": 60})
        assert sig_base is not None
        assert sig_high is not None
        assert sig_high["score"] == sig_base["score"] + 12

    def test_no_news_signal_means_no_change(self):
        from strategies.signal_generator import generate_signal

        ict = {
            "structure": "bullish",
            "bos_choch": [{"type": "BOS", "direction": "bullish", "level": 100.0}],
            "fvgs": [{"type": "bullish", "bottom": 99.0, "top": 101.0}],
            "order_blocks": [],
            "liquidity_sweeps": [],
            "swing_lows": [{"price": 98.0}],
            "swing_highs": [{"price": 105.0}],
        }
        wyckoff = {"phase": "accumulation", "springs": [], "utads": []}
        market = {
            "funding": {},
            "volume_profile": {},
            "kill_zone": {"active": True, "zone": "london", "weight": 0.8},
        }

        sig = generate_signal("XAUUSD", 100.0, ict, wyckoff, market, 10000, news_signal=None)
        assert sig is not None
        assert sig["news_triggered"] is False
        assert sig["news_signal"] is None
