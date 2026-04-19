"""
Phase 5 — Multi-Timeframe Confluence

Analyzes four timeframes and only allows trades where 3+ agree on direction.

    Daily  (1D)  — overall trend bias via 200 EMA + structure  (weight 30%)
    4-Hour (4H)  — Wyckoff phase + intermediate structure     (weight 30%)
    1-Hour (1H)  — confirmation layer                         (weight 25%)
    15-Min (15M) — ICT entry triggers (FVG, OB, sweep)        (weight 15%)

The MTFState class caches analysis per timeframe and only refreshes
when a new candle closes on that timeframe.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Timeframe weights — sum to 1.0.
TF_WEIGHTS: dict[str, float] = {
    "1D": 0.30,
    "4h": 0.30,
    "1h": 0.25,
    "15m": 0.15,
}


# ---------------------------------------------------------------------------
# Per-timeframe bias detection
# ---------------------------------------------------------------------------

def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def detect_tf_bias(df: pd.DataFrame, timeframe: str) -> dict[str, Any]:
    """
    Determine directional bias for a single timeframe.

    Returns:
        {
            "timeframe": str,
            "bias": "bullish" | "bearish" | "neutral",
            "confidence": float 0-1,
            "details": dict with supporting evidence,
        }
    """
    if len(df) < 50:
        return {"timeframe": timeframe, "bias": "neutral", "confidence": 0.0, "details": {}}

    close = df["close"]
    current = close.iloc[-1]

    # EMA structure.
    ema_50 = _ema(close, 50).iloc[-1]
    ema_200 = _ema(close, 200).iloc[-1] if len(df) >= 200 else ema_50

    # Simple HH/HL or LH/LL from recent closes.
    recent_highs = df["high"].tail(20)
    recent_lows = df["low"].tail(20)

    mid_high = recent_highs.iloc[:10].max()
    late_high = recent_highs.iloc[10:].max()
    mid_low = recent_lows.iloc[:10].min()
    late_low = recent_lows.iloc[10:].min()

    hh = late_high > mid_high
    hl = late_low > mid_low
    lh = late_high < mid_high
    ll = late_low < mid_low

    # Bias determination.
    bullish_signals = 0
    bearish_signals = 0

    if current > ema_50:
        bullish_signals += 1
    else:
        bearish_signals += 1

    if current > ema_200:
        bullish_signals += 1
    else:
        bearish_signals += 1

    if ema_50 > ema_200:
        bullish_signals += 1
    else:
        bearish_signals += 1

    if hh and hl:
        bullish_signals += 2
    elif lh and ll:
        bearish_signals += 2

    total = bullish_signals + bearish_signals
    if total == 0:
        return {"timeframe": timeframe, "bias": "neutral", "confidence": 0.0, "details": {}}

    if bullish_signals > bearish_signals:
        bias = "bullish"
        confidence = bullish_signals / total
    elif bearish_signals > bullish_signals:
        bias = "bearish"
        confidence = bearish_signals / total
    else:
        bias = "neutral"
        confidence = 0.5

    return {
        "timeframe": timeframe,
        "bias": bias,
        "confidence": round(confidence, 2),
        "details": {
            "price_vs_ema50": "above" if current > ema_50 else "below",
            "price_vs_ema200": "above" if current > ema_200 else "below",
            "ema_50": round(ema_50, 2),
            "ema_200": round(ema_200, 2),
            "structure": "HH/HL" if (hh and hl) else "LH/LL" if (lh and ll) else "mixed",
        },
    }


# ---------------------------------------------------------------------------
# MTFState — caches analysis per timeframe
# ---------------------------------------------------------------------------

class MTFState:
    """
    Caches per-timeframe analysis and only refreshes when a new candle closes.

    Usage:
        state = MTFState()
        state.update("1D", daily_df)
        state.update("4h", four_hour_df)
        state.update("1h", hourly_df)
        state.update("15m", fifteen_min_df)

        result = state.confluence()
    """

    def __init__(self) -> None:
        self._cache: dict[str, dict[str, Any]] = {}
        self._last_bar: dict[str, Any] = {}  # last bar timestamp per TF

    def update(self, timeframe: str, df: pd.DataFrame) -> dict[str, Any]:
        """
        Update the analysis for a timeframe, but only if the latest bar
        is newer than what we cached.
        """
        if len(df) == 0:
            return self._cache.get(timeframe, {})

        last_ts = df.index[-1] if hasattr(df.index, '__len__') else None

        if timeframe in self._last_bar and self._last_bar[timeframe] == last_ts:
            return self._cache[timeframe]

        bias = detect_tf_bias(df, timeframe)
        self._cache[timeframe] = bias
        self._last_bar[timeframe] = last_ts
        return bias

    def get(self, timeframe: str) -> dict[str, Any]:
        return self._cache.get(timeframe, {"timeframe": timeframe, "bias": "neutral", "confidence": 0.0})

    def confluence(self) -> dict[str, Any]:
        """
        Compute multi-timeframe confluence score.

        Returns:
            {
                "direction": "bullish" | "bearish" | "neutral",
                "score": float (-1.0 to +1.0, weighted),
                "aligned_count": int (how many TFs agree),
                "total_count": int,
                "per_tf": dict[str, dict],
                "trade_allowed": bool (True if 3+ TFs agree),
            }
        """
        weighted_score = 0.0
        bullish_count = 0
        bearish_count = 0
        per_tf = {}

        for tf, weight in TF_WEIGHTS.items():
            state = self._cache.get(tf, {"bias": "neutral", "confidence": 0.0})
            bias = state.get("bias", "neutral")
            conf = state.get("confidence", 0.0)

            if bias == "bullish":
                weighted_score += weight * conf
                bullish_count += 1
            elif bias == "bearish":
                weighted_score -= weight * conf
                bearish_count += 1

            per_tf[tf] = {"bias": bias, "confidence": conf, "weight": weight}

        total = len(TF_WEIGHTS)
        if weighted_score > 0.1:
            direction = "bullish"
            aligned = bullish_count
        elif weighted_score < -0.1:
            direction = "bearish"
            aligned = bearish_count
        else:
            direction = "neutral"
            aligned = 0

        return {
            "direction": direction,
            "score": round(weighted_score, 3),
            "aligned_count": aligned,
            "total_count": total,
            "per_tf": per_tf,
            "trade_allowed": aligned >= 3,
        }

    def status(self) -> dict[str, Any]:
        return {tf: self.get(tf) for tf in TF_WEIGHTS}
