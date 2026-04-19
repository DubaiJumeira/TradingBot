"""
Tests for Phase 1E — Economic Calendar.

All tests use manually-injected events — no network calls.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from strategies.economic_calendar import (
    CalendarEvent,
    EconomicCalendar,
    EventImpact,
    KNOWN_EVENTS,
)


def _event(name: str = "CPI", minutes_from_now: int = 0,
           impact: EventImpact = EventImpact.HIGH,
           assets: list[str] | None = None,
           pre: int = 15, post: int = 10) -> CalendarEvent:
    """Helper to create events relative to 'now' for deterministic tests."""
    now = datetime.now(tz=timezone.utc)
    return CalendarEvent(
        name=name,
        impact=impact,
        scheduled_at=now + timedelta(minutes=minutes_from_now),
        pre_event_minutes=pre,
        post_event_wait_minutes=post,
        affected_assets=assets or ["XAUUSD", "SPX500"],
    )


# -----------------------------------------------------------------------
# CalendarEvent properties
# -----------------------------------------------------------------------

class TestCalendarEvent:
    def test_caution_window(self):
        evt = _event("CPI", minutes_from_now=10, pre=15)
        now = datetime.now(tz=timezone.utc)
        # 10 min from now, caution starts 15 min before → caution started 5 min ago.
        assert evt.in_caution_window(now)

    def test_not_in_caution_before_window(self):
        evt = _event("CPI", minutes_from_now=60, pre=15)
        now = datetime.now(tz=timezone.utc)
        # 60 min away, caution at -15 → not yet.
        assert not evt.in_caution_window(now)

    def test_blackout_during_release(self):
        evt = _event("CPI", minutes_from_now=-2, post=10)
        now = datetime.now(tz=timezone.utc)
        # Event was 2 min ago, post_wait=10 → in blackout.
        assert evt.in_blackout(now)

    def test_not_in_blackout_after_settlement(self):
        evt = _event("CPI", minutes_from_now=-20, post=10)
        now = datetime.now(tz=timezone.utc)
        # Event was 20 min ago, post_wait=10 → past blackout.
        assert not evt.in_blackout(now)

    def test_is_upcoming(self):
        future = _event("CPI", minutes_from_now=30)
        past = _event("CPI", minutes_from_now=-60, post=10)
        now = datetime.now(tz=timezone.utc)
        assert future.is_upcoming(now)
        assert not past.is_upcoming(now)

    def test_minutes_until(self):
        evt = _event("CPI", minutes_from_now=30)
        now = datetime.now(tz=timezone.utc)
        m = evt.minutes_until(now)
        assert 29 < m < 31


# -----------------------------------------------------------------------
# EconomicCalendar.check_events
# -----------------------------------------------------------------------

class TestCheckEvents:
    def test_blocks_trading_in_caution_window(self):
        cal = EconomicCalendar()
        cal.refresh(manual_events=[_event("CPI", minutes_from_now=10, pre=15)])
        result = cal.check_events("XAUUSD")
        assert result["block_trading"] is True
        assert result["tighten_stops"] is True
        assert result["event_name"] == "CPI"

    def test_blocks_trading_in_blackout(self):
        cal = EconomicCalendar()
        cal.refresh(manual_events=[_event("FOMC", minutes_from_now=-3, post=15)])
        result = cal.check_events("SPX500")
        assert result["block_trading"] is True

    def test_does_not_block_unaffected_asset(self):
        cal = EconomicCalendar()
        cal.refresh(manual_events=[
            _event("CPI", minutes_from_now=10, pre=15, assets=["XAUUSD"]),
        ])
        result = cal.check_events("BTC/USDT")
        assert result["block_trading"] is False

    def test_post_event_setup_detected(self):
        cal = EconomicCalendar()
        # Event was 12 min ago, post_wait=10 → safe_after was 2 min ago.
        # Within the 30-min post-event setup window.
        cal.refresh(manual_events=[_event("CPI", minutes_from_now=-12, post=10)])
        result = cal.check_events("XAUUSD")
        assert result["post_event_setup"] is True

    def test_no_events_returns_clear(self):
        cal = EconomicCalendar()
        cal.refresh(manual_events=[])
        result = cal.check_events("XAUUSD")
        assert result["block_trading"] is False
        assert result["tighten_stops"] is False
        assert result["post_event_setup"] is False

    def test_far_future_event_no_block(self):
        cal = EconomicCalendar()
        cal.refresh(manual_events=[_event("NFP", minutes_from_now=120, pre=15)])
        result = cal.check_events("XAUUSD")
        assert result["block_trading"] is False


# -----------------------------------------------------------------------
# Alert dedup
# -----------------------------------------------------------------------

class TestAlertDedup:
    def test_alert_fires_once(self):
        cal = EconomicCalendar()
        cal.refresh(manual_events=[_event("CPI", minutes_from_now=10, pre=15)])

        r1 = cal.check_events("XAUUSD")
        assert r1["alert_event"] is not None

        r2 = cal.check_events("XAUUSD")
        assert r2["alert_event"] is None  # already alerted


# -----------------------------------------------------------------------
# add_known_event
# -----------------------------------------------------------------------

class TestAddKnownEvent:
    def test_creates_from_template(self):
        cal = EconomicCalendar()
        scheduled = datetime(2026, 5, 13, 12, 30, tzinfo=timezone.utc)
        evt = cal.add_known_event("CPI", scheduled)
        assert evt is not None
        assert evt.name == "CPI"
        assert evt.impact == EventImpact.HIGH
        assert "XAUUSD" in evt.affected_assets
        assert evt.pre_event_minutes == KNOWN_EVENTS["CPI"]["pre_event_minutes"]

    def test_unknown_name_returns_none(self):
        cal = EconomicCalendar()
        result = cal.add_known_event("MadeUpEvent", datetime.now(tz=timezone.utc))
        assert result is None


# -----------------------------------------------------------------------
# upcoming / this_week
# -----------------------------------------------------------------------

class TestUpcoming:
    def test_upcoming_filters_by_hours(self):
        cal = EconomicCalendar()
        cal.refresh(manual_events=[
            _event("CPI", minutes_from_now=30),
            _event("NFP", minutes_from_now=60 * 25),  # 25 hours away
        ])
        up = cal.upcoming(hours=24)
        names = [e.name for e in up]
        assert "CPI" in names
        assert "NFP" not in names


# -----------------------------------------------------------------------
# status
# -----------------------------------------------------------------------

class TestStatus:
    def test_status_format(self):
        cal = EconomicCalendar()
        cal.refresh(manual_events=[_event("CPI", minutes_from_now=30)])
        s = cal.status()
        assert "total_events" in s
        assert len(s["upcoming_48h"]) == 1
        assert s["upcoming_48h"][0]["name"] == "CPI"


# -----------------------------------------------------------------------
# KNOWN_EVENTS coverage
# -----------------------------------------------------------------------

class TestKnownEvents:
    def test_all_entries_have_required_fields(self):
        for name, template in KNOWN_EVENTS.items():
            assert "impact" in template, f"{name} missing impact"
            assert "pre_event_minutes" in template, f"{name} missing pre_event_minutes"
            assert "post_event_wait_minutes" in template, f"{name} missing post_event_wait_minutes"
            assert "affected_assets" in template, f"{name} missing affected_assets"
            assert len(template["affected_assets"]) > 0, f"{name} has empty affected_assets"

    def test_fomc_has_longest_windows(self):
        fomc = KNOWN_EVENTS["FOMC"]
        assert fomc["pre_event_minutes"] >= 30
        assert fomc["post_event_wait_minutes"] >= 15
