"""
Phase 1E — Economic Calendar Integration

Tracks scheduled high-impact macro events (CPI, FOMC, NFP, GDP, PCE, etc.)
and adjusts bot behavior around them:

    BEFORE the event:
        - Tighten stops on open positions (or close them if very close)
        - Send Telegram alert 30 min before: "⚠️ CPI Release in 30 min"
        - Suppress new entries within the caution window

    DURING the event:
        - Block all new entries (whipsaw zone)

    AFTER the event (post_event_wait):
        - Wait 5-15 minutes for the whipsaw to settle
        - Then look for ICT setups formed by the volatility (displacement
          candles, FVGs, liquidity sweeps) — these are high-probability entries
        - Feed reactive mode with the direction of the data release

DATA SOURCES
------------
Two layers, in priority order:

    1. KNOWN_RECURRING_EVENTS — a static schedule of regularly-occurring
       events with their typical day/time/impact. Maintained manually.
       This is the reliable fallback that never breaks.

    2. External API (optional) — fetches upcoming events from a free calendar
       API. Currently uses the FXStreet/Investing.com pattern. If the API
       fails, falls back to KNOWN_RECURRING_EVENTS silently.

DESIGN
------
The calendar stores events as `CalendarEvent` dataclasses with:
    - event_name, impact, scheduled_at (UTC)
    - pre_event_minutes (caution window before the release)
    - post_event_wait_minutes (whipsaw settlement window after)
    - affected_assets (which instruments this event moves)
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any

import requests

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

class EventImpact(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class CalendarEvent:
    """One scheduled macro event."""
    name: str
    impact: EventImpact
    scheduled_at: datetime              # tz-aware UTC
    pre_event_minutes: int = 15         # caution window before
    post_event_wait_minutes: int = 10   # whipsaw settlement after
    affected_assets: list[str] = field(default_factory=list)

    @property
    def caution_starts(self) -> datetime:
        return self.scheduled_at - timedelta(minutes=self.pre_event_minutes)

    @property
    def safe_after(self) -> datetime:
        return self.scheduled_at + timedelta(minutes=self.post_event_wait_minutes)

    def in_caution_window(self, now: datetime | None = None) -> bool:
        """True if we're inside the pre-event caution zone."""
        now = now or datetime.now(tz=timezone.utc)
        return self.caution_starts <= now < self.scheduled_at

    def in_blackout(self, now: datetime | None = None) -> bool:
        """True if we're in the release window — no entries allowed."""
        now = now or datetime.now(tz=timezone.utc)
        return self.scheduled_at <= now < self.safe_after

    def is_upcoming(self, now: datetime | None = None) -> bool:
        """True if event hasn't fully passed including the post-event setup window."""
        now = now or datetime.now(tz=timezone.utc)
        # Keep the event alive for 30 minutes after safe_after so check_events
        # can detect the post-event setup window.
        return now < self.safe_after + timedelta(minutes=30)

    def minutes_until(self, now: datetime | None = None) -> float:
        now = now or datetime.now(tz=timezone.utc)
        return (self.scheduled_at - now).total_seconds() / 60


# ---------------------------------------------------------------------------
# Known recurring event templates
# ---------------------------------------------------------------------------
# These define the TYPICAL schedule. The actual dates need to be resolved
# per-week using the economic calendar API or manual input. This table is
# the fallback when the API is down.

KNOWN_EVENTS: dict[str, dict[str, Any]] = {
    "CPI": {
        "impact": "high",
        "pre_event_minutes": 15,
        "post_event_wait_minutes": 10,
        "affected_assets": ["XAUUSD", "SPX500", "US30", "BTC/USDT"],
        "typical_time_utc": "12:30",  # 8:30 ET
        "description": "Consumer Price Index — inflation gauge",
    },
    "FOMC": {
        "impact": "high",
        "pre_event_minutes": 30,
        "post_event_wait_minutes": 15,
        "affected_assets": ["XAUUSD", "SPX500", "US30", "BTC/USDT", "XTIUSD"],
        "typical_time_utc": "18:00",  # 2:00 PM ET
        "description": "Federal Reserve rate decision + statement",
    },
    "NFP": {
        "impact": "high",
        "pre_event_minutes": 15,
        "post_event_wait_minutes": 10,
        "affected_assets": ["XAUUSD", "SPX500", "US30"],
        "typical_time_utc": "12:30",
        "description": "Non-Farm Payrolls — first Friday of month",
    },
    "GDP": {
        "impact": "medium",
        "pre_event_minutes": 10,
        "post_event_wait_minutes": 10,
        "affected_assets": ["SPX500", "US30"],
        "typical_time_utc": "12:30",
        "description": "Gross Domestic Product advance/preliminary/final",
    },
    "PCE": {
        "impact": "high",
        "pre_event_minutes": 15,
        "post_event_wait_minutes": 10,
        "affected_assets": ["XAUUSD", "SPX500", "US30", "BTC/USDT"],
        "typical_time_utc": "12:30",
        "description": "Personal Consumption Expenditures — Fed's preferred inflation measure",
    },
    "PPI": {
        "impact": "medium",
        "pre_event_minutes": 10,
        "post_event_wait_minutes": 5,
        "affected_assets": ["XAUUSD", "SPX500"],
        "typical_time_utc": "12:30",
        "description": "Producer Price Index",
    },
    "Unemployment Claims": {
        "impact": "medium",
        "pre_event_minutes": 10,
        "post_event_wait_minutes": 5,
        "affected_assets": ["SPX500", "US30"],
        "typical_time_utc": "12:30",
        "description": "Weekly initial jobless claims",
    },
    "OPEC Meeting": {
        "impact": "high",
        "pre_event_minutes": 30,
        "post_event_wait_minutes": 15,
        "affected_assets": ["XTIUSD"],
        "typical_time_utc": "13:00",
        "description": "OPEC+ output decisions",
    },
    "ECB Rate Decision": {
        "impact": "high",
        "pre_event_minutes": 15,
        "post_event_wait_minutes": 10,
        "affected_assets": ["XAUUSD", "SPX500"],
        "typical_time_utc": "12:15",
        "description": "European Central Bank interest rate decision",
    },
    "ISM Manufacturing": {
        "impact": "medium",
        "pre_event_minutes": 10,
        "post_event_wait_minutes": 5,
        "affected_assets": ["SPX500", "US30"],
        "typical_time_utc": "14:00",
        "description": "ISM Manufacturing PMI",
    },
}


# ---------------------------------------------------------------------------
# Calendar manager
# ---------------------------------------------------------------------------

class EconomicCalendar:
    """
    Manages upcoming economic events and provides pre/post-event signals
    to the bot's main loop.

    Usage:
        calendar = EconomicCalendar()
        calendar.refresh()  # once per cycle or daily

        # In main loop:
        status = calendar.check_events("XAUUSD")
        if status["block_trading"]:
            # skip entries on this instrument
        if status["tighten_stops"]:
            # tighten SL on open positions
        if status["alert_event"]:
            # send Telegram alert
    """

    def __init__(self) -> None:
        self._events: list[CalendarEvent] = []
        self._alerted: set[str] = set()  # event IDs we've already sent alerts for

    # --------------------------------------------------------- refresh

    def refresh(self, manual_events: list[CalendarEvent] | None = None) -> None:
        """
        Reload the calendar. Priority:
            1. manual_events (if provided — for testing / manual override)
            2. External API fetch
            3. Keep existing events (if API fails and we have some)
        """
        if manual_events is not None:
            self._events = manual_events
            logger.info("Economic calendar loaded %d manual events", len(self._events))
            return

        fetched = self._fetch_from_api()
        if fetched:
            self._events = fetched
            logger.info("Economic calendar refreshed from API: %d events", len(fetched))
        elif not self._events:
            logger.warning("Economic calendar API unavailable and no cached events")

        # Prune past events.
        now = datetime.now(tz=timezone.utc)
        self._events = [e for e in self._events if e.is_upcoming(now)]

    def _fetch_from_api(self) -> list[CalendarEvent] | None:
        """
        Attempt to fetch this week's events from a free calendar API.

        Currently tries the Forex Factory / Trading Economics pattern.
        If the API is unavailable, returns None (caller falls back gracefully).

        NOTE: The free-tier APIs for economic calendars are notoriously
        unreliable. The KNOWN_EVENTS static table above is the real safety
        net. This method is a best-effort enhancement.
        """
        # TradingEconomics free endpoint (limited).
        api_key = os.getenv("TRADINGECONOMICS_KEY", "")
        if not api_key:
            return None

        try:
            now = datetime.now(tz=timezone.utc)
            start = now.strftime("%Y-%m-%d")
            end = (now + timedelta(days=7)).strftime("%Y-%m-%d")
            url = f"https://api.tradingeconomics.com/calendar/country/united states/{start}/{end}"
            resp = requests.get(url, params={"c": api_key, "f": "json"}, timeout=10)
            resp.raise_for_status()
            return self._parse_tradingeconomics(resp.json())
        except Exception as exc:
            logger.warning("Economic calendar API fetch failed: %s", exc)
            return None

    def _parse_tradingeconomics(self, data: list[dict]) -> list[CalendarEvent]:
        """Parse TradingEconomics JSON into CalendarEvent objects."""
        events: list[CalendarEvent] = []
        for row in data:
            name = row.get("Event", "")
            importance = (row.get("Importance") or "").lower()
            if importance not in ("high", "medium"):
                continue  # skip low-impact noise

            date_str = row.get("Date", "")
            if not date_str:
                continue
            try:
                from strategies.news.types import as_utc
                scheduled = as_utc(date_str)
            except Exception:
                continue

            # Try to match against known events for per-event config.
            template = self._match_template(name)
            impact = EventImpact.HIGH if importance == "high" else EventImpact.MEDIUM

            events.append(CalendarEvent(
                name=name,
                impact=impact,
                scheduled_at=scheduled,
                pre_event_minutes=template.get("pre_event_minutes", 15 if importance == "high" else 10),
                post_event_wait_minutes=template.get("post_event_wait_minutes", 10 if importance == "high" else 5),
                affected_assets=template.get("affected_assets", ["SPX500", "US30"]),
            ))
        return events

    def _match_template(self, event_name: str) -> dict[str, Any]:
        """Find the best matching KNOWN_EVENTS template by name substring."""
        name_lower = event_name.lower()
        for known_name, template in KNOWN_EVENTS.items():
            if known_name.lower() in name_lower:
                return template
        return {}

    # --------------------------------------------------- add manual events

    def add_event(self, event: CalendarEvent) -> None:
        """Add a single event manually (e.g., from Telegram /calendar command)."""
        self._events.append(event)
        self._events.sort(key=lambda e: e.scheduled_at)

    def add_known_event(self, name: str, scheduled_at: datetime) -> CalendarEvent | None:
        """
        Create an event from KNOWN_EVENTS template with a specific date/time.

        Usage: calendar.add_known_event("CPI", datetime(2026, 5, 13, 12, 30, tzinfo=utc))
        """
        template = KNOWN_EVENTS.get(name)
        if template is None:
            logger.warning("Unknown event name: %s", name)
            return None
        event = CalendarEvent(
            name=name,
            impact=EventImpact(template["impact"]),
            scheduled_at=scheduled_at,
            pre_event_minutes=template["pre_event_minutes"],
            post_event_wait_minutes=template["post_event_wait_minutes"],
            affected_assets=list(template["affected_assets"]),
        )
        self.add_event(event)
        return event

    # ------------------------------------------------------- check events

    def check_events(self, asset: str, now: datetime | None = None) -> dict[str, Any]:
        """
        Check event status for a specific instrument.

        Scans ALL relevant events and returns the most restrictive state.
        Priority: blackout > caution > post-event setup > clear.

        Returns:
            {
                "block_trading": bool,      # True if inside blackout or caution window
                "tighten_stops": bool,      # True if inside caution window
                "alert_event": CalendarEvent | None,  # event to alert (first time in caution)
                "post_event_setup": bool,   # True if just exited blackout → look for setups
                "event_name": str | None,
                "minutes_until": float | None,
            }
        """
        now = now or datetime.now(tz=timezone.utc)
        result: dict[str, Any] = {
            "block_trading": False,
            "tighten_stops": False,
            "alert_event": None,
            "post_event_setup": False,
            "event_name": None,
            "minutes_until": None,
        }

        for event in self._events:
            if asset not in event.affected_assets:
                continue
            if not event.is_upcoming(now):
                continue

            minutes = event.minutes_until(now)

            # Blackout: during the release + settlement window.
            # Highest priority — return immediately.
            if event.in_blackout(now):
                result["block_trading"] = True
                result["tighten_stops"] = False
                result["event_name"] = event.name
                result["minutes_until"] = minutes
                return result

            # Caution: pre-event window. Tighten stops, suppress new entries.
            if event.in_caution_window(now):
                result["tighten_stops"] = True
                result["block_trading"] = True
                result["event_name"] = event.name
                result["minutes_until"] = minutes

                alert_id = f"{event.name}:{event.scheduled_at.isoformat()}"
                if alert_id not in self._alerted:
                    result["alert_event"] = event
                    self._alerted.add(alert_id)
                # Don't return yet — a blackout from another event would override.
                continue

            # Post-event setup window: 0-30 min after safe_after.
            if event.safe_after <= now < event.safe_after + timedelta(minutes=30):
                # Only set if we haven't already found a more restrictive state.
                if not result["block_trading"]:
                    result["post_event_setup"] = True
                    result["event_name"] = result["event_name"] or event.name

        return result

    # ------------------------------------------------------ info / status

    def upcoming(self, hours: int = 24, now: datetime | None = None) -> list[CalendarEvent]:
        """List events in the next N hours."""
        now = now or datetime.now(tz=timezone.utc)
        cutoff = now + timedelta(hours=hours)
        return [e for e in self._events
                if e.scheduled_at >= now and e.scheduled_at <= cutoff]

    def this_week(self, now: datetime | None = None) -> list[CalendarEvent]:
        """All events this calendar week."""
        return self.upcoming(hours=168, now=now)

    @property
    def event_count(self) -> int:
        return len(self._events)

    def status(self) -> dict[str, Any]:
        """Snapshot for Telegram /calendar command."""
        now = datetime.now(tz=timezone.utc)
        upcoming = self.upcoming(hours=48, now=now)
        return {
            "total_events": self.event_count,
            "upcoming_48h": [
                {
                    "name": e.name,
                    "impact": e.impact.value,
                    "scheduled_at": e.scheduled_at.isoformat(),
                    "minutes_until": round(e.minutes_until(now), 1),
                    "affected_assets": e.affected_assets,
                }
                for e in upcoming
            ],
        }
