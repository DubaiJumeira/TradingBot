"""
Phase 9 — Smart Session & Regime Management

Detects the current market regime per instrument and adapts strategy parameters.

Regimes:
    - trending:  ADX > 25, clear HH/HL or LH/LL → widen TPs, trail stops
    - ranging:   ADX < 20, price in defined range → mean reversion, tight stops
    - choppy:    High ATR but no direction, lots of wicks → reduce size 50% or skip
    - event:     Critical news just hit → wider stops, delayed entry

Each regime modifies the signal score and position sizing.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ADX calculation
# ---------------------------------------------------------------------------

def calculate_adx(df: pd.DataFrame, period: int = 14) -> float:
    """
    Average Directional Index — measures trend strength (0-100).
    ADX > 25 = trending, ADX < 20 = ranging, between = transition.
    """
    if len(df) < period * 2:
        return 0.0

    high = df["high"].values
    low = df["low"].values
    close = df["close"].values

    # True Range.
    tr = np.zeros(len(df))
    for i in range(1, len(df)):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))

    # Directional movement.
    plus_dm = np.zeros(len(df))
    minus_dm = np.zeros(len(df))
    for i in range(1, len(df)):
        up = high[i] - high[i - 1]
        down = low[i - 1] - low[i]
        if up > down and up > 0:
            plus_dm[i] = up
        if down > up and down > 0:
            minus_dm[i] = down

    # Smoothed TR, +DM, -DM.
    atr = np.zeros(len(df))
    s_plus = np.zeros(len(df))
    s_minus = np.zeros(len(df))

    atr[period] = np.sum(tr[1:period + 1])
    s_plus[period] = np.sum(plus_dm[1:period + 1])
    s_minus[period] = np.sum(minus_dm[1:period + 1])

    for i in range(period + 1, len(df)):
        atr[i] = atr[i - 1] - atr[i - 1] / period + tr[i]
        s_plus[i] = s_plus[i - 1] - s_plus[i - 1] / period + plus_dm[i]
        s_minus[i] = s_minus[i - 1] - s_minus[i - 1] / period + minus_dm[i]

    # +DI, -DI.
    plus_di = np.where(atr > 0, 100 * s_plus / atr, 0)
    minus_di = np.where(atr > 0, 100 * s_minus / atr, 0)

    # DX and ADX.
    dx = np.where(
        (plus_di + minus_di) > 0,
        100 * np.abs(plus_di - minus_di) / (plus_di + minus_di),
        0,
    )

    if len(dx) < period * 2:
        return 0.0

    adx = np.mean(dx[-period:])
    return float(adx)


# ---------------------------------------------------------------------------
# Wick ratio (choppiness indicator)
# ---------------------------------------------------------------------------

def _wick_ratio(df: pd.DataFrame, window: int = 20) -> float:
    """
    Average ratio of wicks to total candle range over last N bars.
    High wick ratio = choppy, lots of rejection. Low = clean moves.
    """
    recent = df.tail(window)
    total_wick = 0.0
    total_range = 0.0
    for _, row in recent.iterrows():
        rng = row["high"] - row["low"]
        if rng == 0:
            continue
        body = abs(row["close"] - row["open"])
        wick = rng - body
        total_wick += wick
        total_range += rng

    return total_wick / total_range if total_range > 0 else 0.5


# ---------------------------------------------------------------------------
# Regime detector
# ---------------------------------------------------------------------------

class RegimeDetector:
    """
    Detects market regime for a given instrument's price data.

    Usage:
        detector = RegimeDetector()
        regime = detector.detect(df)
        # regime = {"regime": "trending", "adx": 30.5, ...}
    """

    def detect(
        self,
        df: pd.DataFrame,
        news_event_active: bool = False,
    ) -> dict[str, Any]:
        """
        Detect current regime.

        Parameters
        ----------
        df : OHLCV DataFrame.
        news_event_active : True if a critical news event is in progress
                           (overrides to "event" regime).
        """
        if len(df) < 30:
            return self._regime("unknown", 0, 0, 0)

        # News-driven regime override.
        if news_event_active:
            adx = calculate_adx(df)
            wick = _wick_ratio(df)
            return self._regime("event", adx, wick, 0)

        adx = calculate_adx(df)
        wick = _wick_ratio(df)

        # ATR relative to price (normalized volatility).
        atr_vals = []
        for i in range(1, min(15, len(df))):
            row = df.iloc[-i]
            prev = df.iloc[-i - 1]
            tr = max(row["high"] - row["low"],
                     abs(row["high"] - prev["close"]),
                     abs(row["low"] - prev["close"]))
            atr_vals.append(tr)
        avg_atr = np.mean(atr_vals) if atr_vals else 0
        price = df.iloc[-1]["close"]
        volatility_pct = (avg_atr / price * 100) if price > 0 else 0

        # Classification.
        if adx > 25:
            return self._regime("trending", adx, wick, volatility_pct)
        elif adx < 20 and wick < 0.5:
            return self._regime("ranging", adx, wick, volatility_pct)
        elif wick > 0.6 or (adx < 20 and volatility_pct > 2):
            return self._regime("choppy", adx, wick, volatility_pct)
        else:
            return self._regime("ranging", adx, wick, volatility_pct)

    def _regime(self, regime: str, adx: float, wick: float, vol: float) -> dict[str, Any]:
        return {
            "regime": regime,
            "adx": round(adx, 1),
            "wick_ratio": round(wick, 3),
            "volatility_pct": round(vol, 2),
            "adjustments": self._get_adjustments(regime),
        }

    def _get_adjustments(self, regime: str) -> dict[str, Any]:
        """
        Return strategy adjustments for the detected regime.

        These modifiers are applied in the signal generator or bot loop.
        """
        if regime == "trending":
            return {
                "tp_multiplier": 1.5,     # widen take-profit
                "sl_multiplier": 1.0,     # normal stops
                "size_multiplier": 1.0,   # full size
                "trailing": True,         # use trailing stops
                "min_score_adjust": 0,    # no score adjustment
            }
        elif regime == "ranging":
            return {
                "tp_multiplier": 0.8,     # tighter TP (mean reversion)
                "sl_multiplier": 0.8,     # tighter SL
                "size_multiplier": 1.0,
                "trailing": False,
                "min_score_adjust": 0,
            }
        elif regime == "choppy":
            return {
                "tp_multiplier": 1.0,
                "sl_multiplier": 1.2,     # wider stops to avoid whipsaw
                "size_multiplier": 0.5,   # half size
                "trailing": False,
                "min_score_adjust": 10,   # need higher score to trade
            }
        elif regime == "event":
            return {
                "tp_multiplier": 1.5,
                "sl_multiplier": 1.5,     # wider stops
                "size_multiplier": 0.75,  # reduced size
                "trailing": True,
                "min_score_adjust": 5,
            }
        else:
            return {
                "tp_multiplier": 1.0,
                "sl_multiplier": 1.0,
                "size_multiplier": 1.0,
                "trailing": True,
                "min_score_adjust": 0,
            }
