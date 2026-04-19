"""
Phase 1D — Reactive News-Triggered Trading Mode

Runs alongside the regular 5-minute analysis cycle. When the news aggregator
surfaces a HIGH or CRITICAL event, this module:

    1. Records the event with its correlation-map delay timer
    2. Waits for the initial spike (the delay_seconds window)
    3. THEN triggers an ICT analysis on the affected assets — looking for the
       RETRACEMENT entry, not the spike itself

THE KEY INSIGHT (from the master prompt):
    "Don't chase the news spike. Wait for the spike to create a liquidity
    sweep, THEN enter on the ICT retracement."

This means: a "BREAKING: war escalation" headline fires at T+0. Gold spikes
instantly. At T+30s (the delay), the spike has likely created:
    - A liquidity sweep above recent highs
    - A displacement candle (large body)
    - A Fair Value Gap on the pullback

The reactive mode tells signal_generator to look for those ICT setups on
the affected instruments, with a strong directional bias from the news.

LIFECYCLE
---------
    monitor = ReactiveNewsMonitor(aggregator)

    # Called every ~30 seconds from bot.py's tight loop:
    actions = monitor.check()
    for action in actions:
        # action.asset, action.direction, action.news_signal → feed to analyze_symbol

EVENT STATES
------------
    PENDING  → event detected, waiting for delay_seconds to expire
    READY    → delay expired, trigger analysis on affected assets
    ACTIVE   → analysis triggered, waiting for cooldown
    EXPIRED  → cooldown elapsed, event is done

COOLDOWN
--------
After triggering a reactive analysis, the same event can't re-trigger for
`cooldown_minutes` (default: 15). This prevents the bot from repeatedly
entering the same trade on stale news that's still in the aggregator's cache.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any

from strategies.asset_correlations import (
    AssetImpact,
    CorrelationMatch,
    get_affected_assets,
)
from strategies.news.types import ImpactLevel, NewsItem

logger = logging.getLogger(__name__)


class EventState(str, Enum):
    PENDING = "pending"     # waiting for delay_seconds
    READY = "ready"         # delay expired → trigger analysis
    ACTIVE = "active"       # analysis triggered → cooling down
    EXPIRED = "expired"     # cooldown done → garbage collect


@dataclass
class TrackedEvent:
    """One high-impact news event being tracked through its lifecycle."""
    event_id: str               # hash of title + timestamp for dedup
    title: str
    detected_at: datetime
    matches: list[CorrelationMatch]
    affected_assets: dict[str, AssetImpact]  # asset → best impact
    state: EventState = EventState.PENDING
    triggered_at: datetime | None = None

    @property
    def max_delay(self) -> int:
        """Longest delay_seconds across all affected assets."""
        if not self.affected_assets:
            return 60
        return max(imp.delay_seconds for imp in self.affected_assets.values())

    @property
    def ready_at(self) -> datetime:
        """When the delay window expires and we should look for entries."""
        return self.detected_at + timedelta(seconds=self.max_delay)

    def is_delay_expired(self, now: datetime | None = None) -> bool:
        now = now or datetime.now(tz=timezone.utc)
        return now >= self.ready_at


@dataclass
class ReactiveAction:
    """
    Instruction to the bot: "run analysis on this asset with this news bias."

    The bot's main loop receives a list of these and calls analyze_symbol()
    for each, passing the news_signal dict to the signal generator.
    """
    asset: str
    direction: str          # "positive", "negative", "variable"
    impact_level: ImpactLevel
    delay_seconds: int
    event_title: str
    pattern_name: str

    def to_news_signal(self) -> dict[str, Any]:
        """
        Format consumed by signal_generator.generate_signal()'s news_signal param.

        Maps correlation-map fields to the scoring rules:
            critical + aligned → +20
            high + aligned → +12
            critical + against → -30
        """
        return {
            "impact": self.impact_level.value,
            "direction": self.direction,
            "event_title": self.event_title,
            "pattern": self.pattern_name,
            "delay_seconds": self.delay_seconds,
        }


class ReactiveNewsMonitor:
    """
    Tracks high-impact news events and decides when to trigger reactive analysis.

    Usage from bot.py:
        monitor = ReactiveNewsMonitor(aggregator)

        # In the tight loop (every 30s):
        actions = monitor.check()
        for action in actions:
            signal = analyze_symbol_reactive(action.asset, action.to_news_signal())
    """

    def __init__(
        self,
        aggregator: Any,
        *,
        min_impact: ImpactLevel = ImpactLevel.HIGH,
        cooldown_minutes: int = 15,
        max_tracked: int = 50,
    ) -> None:
        self._aggregator = aggregator
        self._min_impact = min_impact
        self._cooldown = timedelta(minutes=cooldown_minutes)
        self._max_tracked = max_tracked
        self._events: dict[str, TrackedEvent] = {}
        self._seen_titles: set[str] = set()  # prevent re-tracking the same headline

    # ---------------------------------------------------------------- check

    def check(self) -> list[ReactiveAction]:
        """
        Main entry point. Called every ~30 seconds.

        1. Scan aggregator's last fetch for new high-impact items
        2. Register any new events
        3. Advance state machine on all tracked events
        4. Return actions for events that just became READY
        """
        self._ingest_new_events()
        actions = self._advance_states()
        self._gc_expired()
        return actions

    # -------------------------------------------------------------- ingest

    def _ingest_new_events(self) -> None:
        """Check the aggregator's last items for new trackable events."""
        items = self._aggregator.high_impact(minimum=self._min_impact)
        matches_by_title = self._aggregator.last_correlation_matches()

        for item in items:
            eid = self._event_id(item)
            if eid in self._events or item.title in self._seen_titles:
                continue

            item_matches = matches_by_title.get(item.title, [])
            if not item_matches:
                # High impact but no correlation match — no actionable assets.
                continue

            affected = get_affected_assets(item_matches)
            if not affected:
                continue

            event = TrackedEvent(
                event_id=eid,
                title=item.title,
                detected_at=item.published_at,
                matches=item_matches,
                affected_assets=affected,
            )
            self._events[eid] = event
            self._seen_titles.add(item.title)
            logger.info(
                "REACTIVE: new event tracked — '%s' | assets=%s | delay=%ds | state=PENDING",
                item.title[:80],
                list(affected.keys()),
                event.max_delay,
            )

    # -------------------------------------------------------- state machine

    def _advance_states(self) -> list[ReactiveAction]:
        """Advance all tracked events and collect actions from newly READY ones."""
        now = datetime.now(tz=timezone.utc)
        actions: list[ReactiveAction] = []

        for event in self._events.values():
            if event.state == EventState.PENDING and event.is_delay_expired(now):
                event.state = EventState.READY
                logger.info(
                    "REACTIVE: event READY — '%s' (delay expired after %ds)",
                    event.title[:80], event.max_delay,
                )

            if event.state == EventState.READY:
                actions.extend(self._generate_actions(event))
                event.state = EventState.ACTIVE
                event.triggered_at = now
                logger.info(
                    "REACTIVE: triggered %d actions for '%s' → state=ACTIVE (cooldown=%s)",
                    len(actions), event.title[:80], self._cooldown,
                )

            if event.state == EventState.ACTIVE:
                if event.triggered_at and now >= event.triggered_at + self._cooldown:
                    event.state = EventState.EXPIRED

        return actions

    def _generate_actions(self, event: TrackedEvent) -> list[ReactiveAction]:
        """Create one ReactiveAction per affected asset."""
        actions: list[ReactiveAction] = []
        for asset, impact in event.affected_assets.items():
            pattern_name = ""
            for m in event.matches:
                for imp in m.impacts:
                    if imp.asset == asset:
                        pattern_name = m.pattern_name
                        break
                if pattern_name:
                    break
            actions.append(ReactiveAction(
                asset=asset,
                direction=impact.direction,
                impact_level=impact.magnitude,
                delay_seconds=impact.delay_seconds,
                event_title=event.title,
                pattern_name=pattern_name,
            ))
        return actions

    # --------------------------------------------------------------- gc

    def _gc_expired(self) -> None:
        """Remove expired events to bound memory."""
        expired = [eid for eid, ev in self._events.items() if ev.state == EventState.EXPIRED]
        for eid in expired:
            del self._events[eid]
        # Also cap total tracked events.
        if len(self._events) > self._max_tracked:
            oldest = sorted(self._events.values(), key=lambda e: e.detected_at)
            for ev in oldest[: len(self._events) - self._max_tracked]:
                self._events.pop(ev.event_id, None)

    # --------------------------------------------------------------- util

    @staticmethod
    def _event_id(item: NewsItem) -> str:
        """Stable ID for dedup — same title at same time = same event."""
        from hashlib import sha1
        raw = f"{item.title}:{item.published_at.isoformat()}"
        return sha1(raw.encode()).hexdigest()[:16]

    # -------------------------------------------------------------- status

    @property
    def pending_count(self) -> int:
        return sum(1 for e in self._events.values() if e.state == EventState.PENDING)

    @property
    def active_count(self) -> int:
        return sum(1 for e in self._events.values() if e.state == EventState.ACTIVE)

    def status(self) -> dict[str, Any]:
        """Snapshot for Telegram /status command and logging."""
        return {
            "tracked_events": len(self._events),
            "pending": self.pending_count,
            "active": self.active_count,
            "events": [
                {
                    "title": ev.title[:80],
                    "state": ev.state.value,
                    "assets": list(ev.affected_assets.keys()),
                    "detected_at": ev.detected_at.isoformat(),
                }
                for ev in self._events.values()
            ],
        }
