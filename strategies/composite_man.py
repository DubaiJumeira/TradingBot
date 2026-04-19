"""
Phase 8 — Composite Man detector.

The "composite man" is Wyckoff's shorthand for the aggregate intent of
large/institutional participants. We can't see their orders directly,
but their footprint shows up as a signature in the existing VSA /
effort-vs-result / event data: stealth accumulation looks like
absorption in a quiet range while ICT marks bullish OBs near the
lows; stealth distribution mirrors it at the highs; a shakeout is a
sharp sweep-and-reclaim that paints a Spring.

This module takes the already-computed wyckoff and ict dicts and
returns a single "intent" label the scorer can act on. It is
deliberately stateless — no new data fetches, no new math, just
interpretation of existing tensors.
"""

from __future__ import annotations

from typing import Any


def detect_composite_man(wyckoff: dict, ict: dict) -> dict[str, Any] | None:
    """Return {'intent': 'accumulating'|'distributing'|'shakeout',
    'confidence': float, 'reason': str} or None.
    """
    phase = wyckoff.get("phase") or ""
    events = wyckoff.get("wyckoff_events", []) or []
    vsa = wyckoff.get("vsa_signals", []) or []
    effort = wyckoff.get("effort_vs_result", []) or []
    order_blocks = ict.get("order_blocks", []) or []
    sweeps = ict.get("liquidity_sweeps", []) or []
    structure = ict.get("structure", "ranging")
    price_zone = ict.get("price_zone", "equilibrium")

    recent_absorption = any(
        e.get("type") == "absorption" for e in effort[-3:]
    ) or any(v.get("type") == "absorption" for v in vsa[-3:])

    # ---- Shakeout: sweep of swing low followed by a quick reclaim +
    # any absorption nearby. Classic spring footprint.
    bull_sweep = any(s.get("type") == "bullish_sweep" for s in sweeps[-3:])
    bear_sweep = any(s.get("type") == "bearish_sweep" for s in sweeps[-3:])
    has_spring = any(e.get("event") == "Spring" for e in events[-5:])

    if (bull_sweep or has_spring) and recent_absorption:
        return {
            "intent": "shakeout",
            "confidence": 0.75,
            "reason": "liquidity swept below range + absorption on reclaim",
        }

    # ---- Stealth accumulation: accumulation phase + discount zone +
    # bullish OB stack + absorption present. Composite man quietly
    # filling longs.
    bullish_obs = [ob for ob in order_blocks if ob.get("type") == "bullish"]
    bearish_obs = [ob for ob in order_blocks if ob.get("type") == "bearish"]

    if (
        phase in ("accumulation", "accumulation_to_markup")
        and price_zone == "discount"
        and len(bullish_obs) >= 2
        and recent_absorption
    ):
        return {
            "intent": "accumulating",
            "confidence": 0.8,
            "reason": f"{len(bullish_obs)} bullish OBs in discount + absorption, phase={phase}",
        }

    # ---- Stealth distribution: distribution phase + premium zone +
    # bearish OB stack + absorption present. Mirror of accumulation.
    if (
        phase in ("distribution", "distribution_to_markdown")
        and price_zone == "premium"
        and len(bearish_obs) >= 2
        and recent_absorption
    ):
        return {
            "intent": "distributing",
            "confidence": 0.8,
            "reason": f"{len(bearish_obs)} bearish OBs in premium + absorption, phase={phase}",
        }

    # ---- Weaker signal: bear sweep in premium + bearish structure.
    if bear_sweep and price_zone == "premium" and structure == "bearish":
        return {
            "intent": "distributing",
            "confidence": 0.6,
            "reason": "liquidity swept above range + bearish structure in premium",
        }

    return None
