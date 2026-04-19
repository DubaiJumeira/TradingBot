"""
Market Data Module
Analyzes: Funding Rate, Open Interest, Volume Profile, Kill Zones

Phase 2 changes:
    - analyze_market_data() accepts an optional instrument config dict so it
      can skip funding/OI for non-crypto instruments and apply per-instrument
      kill zone weights.
    - get_current_kill_zone() accepts optional per-instrument weight overrides
      and respects session restrictions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from datetime import datetime, timezone
from typing import Any
import logging

logger = logging.getLogger(__name__)


def analyze_funding_rate(funding_data: dict):
    """
    Extreme funding = overleveraged market = reversal likely.
    Positive funding > 0.03% = longs paying shorts (bearish signal)
    Negative funding < -0.03% = shorts paying longs (bullish signal)
    """
    if not funding_data:
        return {"signal": "neutral", "rate": 0}

    rate = funding_data.get("fundingRate", 0)
    if rate is None:
        return {"signal": "neutral", "rate": 0}

    signal = "neutral"
    if rate > 0.0005:  # 0.05%+ very high
        signal = "extreme_long"  # fade longs
    elif rate > 0.0003:
        signal = "elevated_long"
    elif rate < -0.0005:
        signal = "extreme_short"  # fade shorts
    elif rate < -0.0003:
        signal = "elevated_short"

    return {
        "signal": signal,
        "rate": round(rate * 100, 4),  # as percentage
        "interpretation": {
            "extreme_long": "Market overleveraged long — expect liquidation cascade down",
            "elevated_long": "Longs paying premium — slight bearish bias",
            "neutral": "Balanced funding — no directional bias from funding",
            "elevated_short": "Shorts paying premium — slight bullish bias",
            "extreme_short": "Market overleveraged short — expect short squeeze up",
        }.get(signal, ""),
    }


def analyze_open_interest(oi_data: dict, price_change_pct: float):
    """
    OI rising + price rising = strong trend (new money entering)
    OI rising + price falling = aggressive shorting (potential squeeze)
    OI falling + price rising = short covering (weak rally)
    OI falling + price falling = long liquidation (capitulation)
    """
    if not oi_data:
        return {"signal": "unknown", "oi": 0}

    oi = oi_data.get("openInterestValue", 0) or oi_data.get("openInterest", 0)

    # We'd need historical OI to compare — simplified version
    return {
        "oi_value": oi,
        "note": "Track OI changes over time for divergence signals",
    }


_DEFAULT_KZ_WEIGHTS: dict[str, float] = {
    "asian": 0.5,
    "london": 0.8,
    "new_york": 1.0,
    "london_ny_overlap": 0.9,
}


def get_current_kill_zone(
    kill_zones: dict,
    *,
    instrument: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Determine if we're currently in a kill zone.

    Parameters
    ----------
    kill_zones : dict
        Mapping of zone_name → (start_utc, end_utc) from Config.KILL_ZONES.
    instrument : dict, optional
        Per-instrument config from INSTRUMENTS. When provided:
        - Only zones listed in instrument["sessions"] are considered active.
        - Kill zone weights come from instrument["kill_zone_weights"].
    """
    now = datetime.now(timezone.utc)
    current_time = now.strftime("%H:%M")

    allowed_sessions = instrument.get("sessions") if instrument else None
    kz_weights = (
        instrument.get("kill_zone_weights", _DEFAULT_KZ_WEIGHTS)
        if instrument else _DEFAULT_KZ_WEIGHTS
    )

    # Check zones in priority order: overlap first, then major sessions.
    # We return the *highest-weight* active zone so the overlap beats a
    # plain london or new_york match when both are active at the same time.
    best: dict[str, Any] | None = None

    for zone_name, (start, end) in kill_zones.items():
        if start <= current_time <= end:
            # If the instrument restricts sessions, skip zones it doesn't trade.
            if allowed_sessions and zone_name not in allowed_sessions:
                continue

            weight = kz_weights.get(zone_name, _DEFAULT_KZ_WEIGHTS.get(zone_name, 0.5))
            candidate = {"active": True, "zone": zone_name, "weight": weight}
            if best is None or weight > best["weight"]:
                best = candidate

    return best or {"active": False, "zone": None, "weight": 0.3}


def calculate_volume_profile(df: pd.DataFrame, num_levels: int = 20):
    """
    Simple volume profile: find price levels with most volume traded.
    High volume nodes = support/resistance.
    """
    price_min = df["low"].min()
    price_max = df["high"].max()
    step = (price_max - price_min) / num_levels

    levels = []
    for i in range(num_levels):
        level_low = price_min + i * step
        level_high = level_low + step
        # Sum volume for candles that traded through this level
        mask = (df["low"] <= level_high) & (df["high"] >= level_low)
        vol = df.loc[mask, "volume"].sum()
        levels.append({
            "price_low": round(level_low, 2),
            "price_high": round(level_high, 2),
            "midpoint": round((level_low + level_high) / 2, 2),
            "volume": vol,
        })

    # Sort by volume to find high volume nodes (HVN) and low volume nodes (LVN)
    sorted_levels = sorted(levels, key=lambda x: x["volume"], reverse=True)
    poc = sorted_levels[0]  # Point of Control = highest volume level

    return {
        "poc": poc["midpoint"],
        "hvn": [l["midpoint"] for l in sorted_levels[:3]],  # top 3 high volume
        "lvn": [l["midpoint"] for l in sorted_levels[-3:]],  # bottom 3 low volume
        "levels": levels,
    }


def _build_liquidation_block(
    symbol: str,
    current_price: float,
    oi_data: dict | None,
) -> dict[str, Any]:
    """Fetch (or estimate) liquidation clusters + derived magnets.

    Best-effort: any failure returns an empty block so downstream
    scoring is a no-op rather than a crash.
    """
    try:
        from strategies.liquidation import fetch_liquidation_clusters
        from strategies.liquidity_magnets import detect_magnets, compute_asymmetry

        oi_value = 0.0
        if oi_data:
            oi_value = float(
                oi_data.get("openInterestValue")
                or oi_data.get("openInterest")
                or 0
            )
        clusters, source = fetch_liquidation_clusters(symbol, current_price, oi_value)
        magnets = detect_magnets(clusters, current_price)
        asymmetry = compute_asymmetry(magnets)
        return {
            "source": source,
            "clusters": clusters,
            "magnets": magnets,
            "asymmetry": asymmetry,
        }
    except Exception as exc:
        logger.warning("liquidation block failed for %s: %s", symbol, exc)
        return {"source": "unavailable", "clusters": [], "magnets": [], "asymmetry": {}}


def _build_manipulation_block(
    exchange,
    symbol: str,
    df: pd.DataFrame,
    current_price: float,
    tracker: Any | None,
) -> dict[str, Any]:
    """Run all manipulation detectors and return the tracker's current state.

    Best-effort: any failure returns an empty block.
    """
    if tracker is None:
        return {"events": [], "cluster": None, "tracker": None}
    try:
        from strategies.manipulation import detect_stop_hunt, detect_absorption

        # Stateless OHLCV detectors run every cycle.
        ohlcv_events = []
        ohlcv_events.extend(detect_stop_hunt(df))
        ohlcv_events.extend(detect_absorption(df))
        if ohlcv_events:
            tracker.ingest_ohlcv_events(ohlcv_events)

        # Stateful order-book detectors: one snapshot per cycle.
        try:
            order_book = exchange.fetch_order_book(symbol, limit=50)
            tracker.ingest_order_book(order_book, current_price)
        except Exception as exc:
            logger.debug("order book fetch failed for %s: %s", symbol, exc)

        recent = tracker.recent_events()
        return {
            "events": [e.as_dict() for e in recent],
            "cluster": tracker.detect_cluster(),
            "tracker": tracker,
        }
    except Exception as exc:
        logger.warning("manipulation block failed for %s: %s", symbol, exc)
        return {"events": [], "cluster": None, "tracker": None}


def analyze_market_data(
    exchange,
    symbol: str,
    df: pd.DataFrame,
    kill_zones: dict,
    instrument: dict[str, Any] | None = None,
    manipulation_tracker: Any | None = None,
):
    """
    Run all market data analysis.

    Parameters
    ----------
    instrument : dict, optional
        Per-instrument config from INSTRUMENTS. Controls:
        - Whether funding rate / OI are fetched (only for crypto).
        - Kill zone weight overrides and session filtering.
    """
    # Funding rate / OI — crypto only.
    has_funding = instrument.get("funding", True) if instrument else True
    if has_funding:
        funding = exchange.fetch_funding_rate(symbol)
        oi = exchange.fetch_open_interest(symbol)
    else:
        funding = None
        oi = None

    price_change = 0
    if len(df) > 1:
        price_change = (df.iloc[-1]["close"] - df.iloc[-2]["close"]) / df.iloc[-2]["close"] * 100

    current_price = float(df.iloc[-1]["close"]) if len(df) else 0.0
    liquidation_block = (
        _build_liquidation_block(symbol, current_price, oi)
        if has_funding and current_price > 0
        else {"source": "disabled", "clusters": [], "magnets": [], "asymmetry": {}}
    )

    manipulation_block = _build_manipulation_block(
        exchange, symbol, df, current_price, manipulation_tracker
    )

    return {
        "funding": analyze_funding_rate(funding),
        "open_interest": analyze_open_interest(oi, price_change),
        "kill_zone": get_current_kill_zone(kill_zones, instrument=instrument),
        "volume_profile": calculate_volume_profile(df),
        "liquidation": liquidation_block,
        "manipulation": manipulation_block,
    }
