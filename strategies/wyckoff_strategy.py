"""
Wyckoff Strategy Module
Detects: Accumulation / Distribution phases, Spring / UTAD,
         Sign of Strength (SOS), Last Point of Support (LPS)

Phase 4 additions:
    - Volume Spread Analysis (VSA): candle spread vs volume anomalies
    - Effort vs Result: volume divergence from price movement
    - Detailed event labeling: PS, SC, AR, ST, Spring, SOS, LPS
      (and distribution: PSY, BC, AR, ST, UTAD, SOW, LPSY)
    - Phase transition detection: accumulation→markup, distribution→markdown
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import logging
from typing import Any

logger = logging.getLogger(__name__)


def detect_volume_profile(df: pd.DataFrame, window: int = 20):
    """Classify volume as above or below average."""
    df = df.copy()
    df["vol_sma"] = df["volume"].rolling(window).mean()
    df["vol_ratio"] = df["volume"] / df["vol_sma"]
    df["high_volume"] = df["vol_ratio"] > 1.5
    df["low_volume"] = df["vol_ratio"] < 0.7
    return df


def detect_trading_range(df: pd.DataFrame, window: int = 50, threshold_pct: float = 8.0):
    """
    Detect if price is in a trading range (consolidation).
    Returns range boundaries if found.
    """
    recent = df.tail(window)
    range_high = recent["high"].max()
    range_low = recent["low"].min()
    range_pct = (range_high - range_low) / range_low * 100

    # If range is relatively tight, it's consolidation
    in_range = range_pct <= threshold_pct

    return {
        "in_range": in_range,
        "range_high": range_high,
        "range_low": range_low,
        "range_pct": round(range_pct, 2),
        "midpoint": (range_high + range_low) / 2,
    }


def detect_spring(df: pd.DataFrame, range_low: float, lookback: int = 10):
    """
    Spring: price briefly dips below range support on volume,
    then closes back inside the range. Bullish signal.
    """
    springs = []
    for i in range(len(df) - lookback, len(df)):
        if i < 0:
            continue
        row = df.iloc[i]
        if row["low"] < range_low and row["close"] > range_low:
            springs.append({
                "type": "spring",
                "low": row["low"],
                "close": row["close"],
                "range_low": range_low,
                "index": i,
                "time": df.index[i],
            })
    return springs


def detect_utad(df: pd.DataFrame, range_high: float, lookback: int = 10):
    """
    Upthrust After Distribution (UTAD): price spikes above range resistance
    then closes back inside. Bearish signal.
    """
    utads = []
    for i in range(len(df) - lookback, len(df)):
        if i < 0:
            continue
        row = df.iloc[i]
        if row["high"] > range_high and row["close"] < range_high:
            utads.append({
                "type": "utad",
                "high": row["high"],
                "close": row["close"],
                "range_high": range_high,
                "index": i,
                "time": df.index[i],
            })
    return utads


def classify_wyckoff_phase(df: pd.DataFrame, trading_range: dict):
    """
    Classify current Wyckoff phase based on price action and volume.

    Phases:
    - accumulation: price at range lows, declining volume, springs
    - markup: price breaking above range with volume
    - distribution: price at range highs, declining volume, UTADs
    - markdown: price breaking below range with volume
    """
    if not trading_range["in_range"]:
        # Check if we're in markup or markdown
        recent_close = df.iloc[-1]["close"]
        prev_close = df.iloc[-20]["close"] if len(df) > 20 else df.iloc[0]["close"]

        if recent_close > trading_range["range_high"]:
            return "markup"
        elif recent_close < trading_range["range_low"]:
            return "markdown"

    # In range — check volume behavior and price position
    vdf = detect_volume_profile(df)
    recent = vdf.tail(10)

    price_in_lower_half = df.iloc[-1]["close"] < trading_range["midpoint"]
    declining_volume = recent["vol_ratio"].mean() < 1.0

    springs = detect_spring(df, trading_range["range_low"])
    utads = detect_utad(df, trading_range["range_high"])

    if price_in_lower_half and declining_volume:
        return "accumulation"
    elif not price_in_lower_half and declining_volume:
        return "distribution"
    elif price_in_lower_half:
        return "accumulation"
    else:
        return "distribution"


# ---------------------------------------------------------------------------
# Phase 4: Volume Spread Analysis (VSA)
# ---------------------------------------------------------------------------

def analyze_vsa(df: pd.DataFrame, window: int = 20) -> list[dict[str, Any]]:
    """
    Volume Spread Analysis: detect anomalies between candle spread and volume.

    Key patterns:
    - High volume + narrow spread = absorption (smart money absorbing supply/demand)
    - Low volume + wide spread = no follow-through (likely to reverse)
    - Very high volume + close near low = selling climax
    - Very high volume + close near high = buying climax
    """
    signals: list[dict] = []
    if len(df) < window + 1:
        return signals

    vdf = df.copy()
    vdf["spread"] = vdf["high"] - vdf["low"]
    vdf["body"] = abs(vdf["close"] - vdf["open"])
    vdf["vol_sma"] = vdf["volume"].rolling(window).mean()
    vdf["spread_sma"] = vdf["spread"].rolling(window).mean()

    for i in range(window, len(vdf)):
        row = vdf.iloc[i]
        if row["vol_sma"] == 0 or row["spread_sma"] == 0:
            continue

        vol_ratio = row["volume"] / row["vol_sma"]
        spread_ratio = row["spread"] / row["spread_sma"]
        close_position = (
            (row["close"] - row["low"]) / row["spread"]
            if row["spread"] > 0 else 0.5
        )

        # High volume + narrow spread = absorption.
        if vol_ratio > 1.5 and spread_ratio < 0.7:
            signals.append({
                "type": "absorption",
                "index": i,
                "vol_ratio": round(vol_ratio, 2),
                "spread_ratio": round(spread_ratio, 2),
                "close_position": round(close_position, 2),
            })

        # Low volume + wide spread = no follow-through.
        elif vol_ratio < 0.5 and spread_ratio > 1.5:
            signals.append({
                "type": "no_follow_through",
                "index": i,
                "vol_ratio": round(vol_ratio, 2),
                "spread_ratio": round(spread_ratio, 2),
            })

        # Very high volume + close near low = selling climax.
        elif vol_ratio > 2.0 and close_position < 0.25:
            signals.append({
                "type": "selling_climax",
                "index": i,
                "vol_ratio": round(vol_ratio, 2),
                "close_position": round(close_position, 2),
            })

        # Very high volume + close near high = buying climax.
        elif vol_ratio > 2.0 and close_position > 0.75:
            signals.append({
                "type": "buying_climax",
                "index": i,
                "vol_ratio": round(vol_ratio, 2),
                "close_position": round(close_position, 2),
            })

    return signals


# ---------------------------------------------------------------------------
# Phase 4: Effort vs Result
# ---------------------------------------------------------------------------

def analyze_effort_vs_result(df: pd.DataFrame, window: int = 5) -> list[dict[str, Any]]:
    """
    Compare volume (effort) to price movement (result) over rolling windows.
    Divergence = smart money absorbing.

    - High effort + low result = absorption (accumulation or distribution)
    - Low effort + high result = mark-up/down on thin volume (vulnerable)
    """
    signals: list[dict] = []
    if len(df) < window + 20:
        return signals

    vol_sma = df["volume"].rolling(20).mean()

    for i in range(window + 20, len(df)):
        # Effort: sum of volume over window.
        effort = df["volume"].iloc[i - window:i].sum()
        avg_effort = vol_sma.iloc[i] * window if vol_sma.iloc[i] > 0 else 1

        # Result: price change over window.
        result = abs(df["close"].iloc[i] - df["close"].iloc[i - window])
        avg_result = df["close"].iloc[i] * 0.01  # 1% as baseline

        effort_ratio = effort / avg_effort if avg_effort > 0 else 1
        result_ratio = result / avg_result if avg_result > 0 else 0

        # High effort, low result → absorption.
        if effort_ratio > 1.5 and result_ratio < 0.5:
            signals.append({
                "type": "absorption",
                "index": i,
                "effort_ratio": round(effort_ratio, 2),
                "result_ratio": round(result_ratio, 2),
            })

        # Low effort, high result → vulnerable move.
        elif effort_ratio < 0.5 and result_ratio > 1.5:
            signals.append({
                "type": "vulnerable_move",
                "index": i,
                "effort_ratio": round(effort_ratio, 2),
                "result_ratio": round(result_ratio, 2),
            })

    return signals


# ---------------------------------------------------------------------------
# Phase 4: Detailed Wyckoff event labeling
# ---------------------------------------------------------------------------

def label_wyckoff_events(
    df: pd.DataFrame,
    trading_range: dict,
    phase: str,
) -> list[dict[str, Any]]:
    """
    Label specific Wyckoff events within a trading range.

    Accumulation sequence: PS → SC → AR → ST → Spring → SOS → LPS
    Distribution sequence: PSY → BC → AR → ST → UTAD → SOW → LPSY
    """
    events: list[dict] = []
    if not trading_range["in_range"] or len(df) < 30:
        return events

    range_high = trading_range["range_high"]
    range_low = trading_range["range_low"]
    midpoint = trading_range["midpoint"]
    vdf = detect_volume_profile(df)

    recent = df.tail(50)
    for i in range(1, len(recent)):
        idx = len(df) - len(recent) + i
        row = recent.iloc[i]
        prev = recent.iloc[i - 1]
        vol_ratio = vdf.iloc[idx]["vol_ratio"] if idx < len(vdf) else 1.0

        if phase in ("accumulation",):
            # Selling Climax (SC): sharp drop to range low on very high volume.
            if row["low"] <= range_low * 1.005 and vol_ratio > 2.0 and row["close"] > row["low"]:
                events.append({"event": "SC", "index": idx, "price": row["low"]})

            # Automatic Rally (AR): bounce from SC area on declining volume.
            if (row["close"] > prev["close"] and row["close"] > midpoint
                    and vol_ratio < 1.0 and prev["close"] < midpoint):
                events.append({"event": "AR", "index": idx, "price": row["close"]})

            # Sign of Strength (SOS): price breaks above range on volume.
            if row["close"] > range_high and vol_ratio > 1.3:
                events.append({"event": "SOS", "index": idx, "price": row["close"]})

            # Last Point of Support (LPS): pullback to range top on low volume.
            if (prev["close"] > range_high and row["close"] >= range_high * 0.99
                    and row["close"] < prev["close"] and vol_ratio < 0.8):
                events.append({"event": "LPS", "index": idx, "price": row["close"]})

        elif phase in ("distribution",):
            # Buying Climax (BC): push to range high on very high volume.
            if row["high"] >= range_high * 0.995 and vol_ratio > 2.0 and row["close"] < row["high"]:
                events.append({"event": "BC", "index": idx, "price": row["high"]})

            # Sign of Weakness (SOW): price breaks below range on volume.
            if row["close"] < range_low and vol_ratio > 1.3:
                events.append({"event": "SOW", "index": idx, "price": row["close"]})

            # Last Point of Supply (LPSY): rally to range bottom on low volume.
            if (prev["close"] < range_low and row["close"] <= range_low * 1.01
                    and row["close"] > prev["close"] and vol_ratio < 0.8):
                events.append({"event": "LPSY", "index": idx, "price": row["close"]})

    return events


# ---------------------------------------------------------------------------
# Phase 4: Phase transition detection
# ---------------------------------------------------------------------------

def detect_phase_transition(
    df: pd.DataFrame,
    trading_range: dict,
    phase: str,
    lookback: int = 10,
) -> dict[str, Any] | None:
    """
    Detect when a Wyckoff phase transitions — these are high-probability moments.

    - accumulation → markup: price breaks above range with volume confirmation
    - distribution → markdown: price breaks below range with volume confirmation
    """
    if len(df) < lookback + 1:
        return None

    vdf = detect_volume_profile(df)
    recent = vdf.tail(lookback)
    current_close = df.iloc[-1]["close"]
    avg_vol = recent["vol_ratio"].mean()

    if phase == "accumulation" and current_close > trading_range["range_high"]:
        if avg_vol > 1.2:
            return {
                "transition": "accumulation_to_markup",
                "breakout_price": current_close,
                "range_high": trading_range["range_high"],
                "volume_confirmation": round(avg_vol, 2),
            }

    if phase == "distribution" and current_close < trading_range["range_low"]:
        if avg_vol > 1.2:
            return {
                "transition": "distribution_to_markdown",
                "breakdown_price": current_close,
                "range_low": trading_range["range_low"],
                "volume_confirmation": round(avg_vol, 2),
            }

    return None


# ---------------------------------------------------------------------------
# Enhanced analyze_wyckoff
# ---------------------------------------------------------------------------

def analyze_wyckoff(df: pd.DataFrame):
    """
    Run full Wyckoff analysis.

    Phase 4 additions: VSA, effort vs result, event labeling,
    phase transition detection.
    """
    trading_range = detect_trading_range(df)
    phase = classify_wyckoff_phase(df, trading_range)
    springs = detect_spring(df, trading_range["range_low"]) if trading_range["in_range"] else []
    utads = detect_utad(df, trading_range["range_high"]) if trading_range["in_range"] else []
    vdf = detect_volume_profile(df)

    # Phase 4.
    vsa = analyze_vsa(df)
    effort_result = analyze_effort_vs_result(df)
    events = label_wyckoff_events(df, trading_range, phase)
    transition = detect_phase_transition(df, trading_range, phase)

    return {
        "phase": phase,
        "trading_range": trading_range,
        "springs": springs,
        "utads": utads,
        "volume_declining": vdf.tail(10)["vol_ratio"].mean() < 1.0,
        "avg_vol_ratio": round(vdf.tail(10)["vol_ratio"].mean(), 2),
        # Phase 4
        "vsa_signals": vsa[-5:],
        "effort_vs_result": effort_result[-3:],
        "wyckoff_events": events[-5:],
        "phase_transition": transition,
    }
