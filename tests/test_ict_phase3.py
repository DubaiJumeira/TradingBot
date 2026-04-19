"""
Tests for Phase 3 — ICT Module Enhancements.

Covers: OTE, displacement, breaker blocks, premium/discount,
        inducement, liquidity voids.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from strategies.ict_strategy import (
    _calculate_atr_series,
    is_displacement_candle,
    calculate_premium_discount,
    get_price_zone,
    detect_ote,
    detect_breaker_blocks,
    detect_inducements,
    detect_liquidity_voids,
    analyze_ict,
)


def _make_df(rows):
    """Build DataFrame from list of (open, high, low, close, volume) tuples."""
    return pd.DataFrame(rows, columns=["open", "high", "low", "close", "volume"])


class TestATRSeries:
    def test_length_matches(self):
        df = _make_df([(100, 102, 98, 101, 1000)] * 20)
        atr = _calculate_atr_series(df, period=14)
        assert len(atr) == len(df)

    def test_early_values_are_nan(self):
        df = _make_df([(100, 102, 98, 101, 1000)] * 20)
        atr = _calculate_atr_series(df, period=14)
        assert np.isnan(atr[0])
        assert not np.isnan(atr[13])


class TestDisplacement:
    def test_large_body_is_displacement(self):
        # ATR ~4, body = 10 > 2*4 = 8 → displacement.
        rows = [(100, 102, 98, 101, 1000)] * 15 + [(100, 111, 99, 110, 2000)]
        df = _make_df(rows)
        atr = _calculate_atr_series(df, period=14)
        assert is_displacement_candle(df, 15, atr, multiplier=2.0)

    def test_small_body_is_not_displacement(self):
        rows = [(100, 102, 98, 101, 1000)] * 16
        df = _make_df(rows)
        atr = _calculate_atr_series(df, period=14)
        assert not is_displacement_candle(df, 15, atr, multiplier=2.0)


class TestPremiumDiscount:
    def test_zones_calculated(self):
        highs = [{"price": 110, "index": 0}, {"price": 120, "index": 5}]
        lows = [{"price": 90, "index": 1}, {"price": 95, "index": 6}]
        pd_zones = calculate_premium_discount(highs, lows)
        assert pd_zones["equilibrium"] == (120 + 90) / 2  # 105

    def test_price_in_premium(self):
        pd_zones = {"equilibrium": 105, "range_high": 120, "range_low": 90}
        assert get_price_zone(115, pd_zones) == "premium"

    def test_price_in_discount(self):
        pd_zones = {"equilibrium": 105, "range_high": 120, "range_low": 90}
        assert get_price_zone(95, pd_zones) == "discount"

    def test_empty_swings(self):
        pd_zones = calculate_premium_discount([], [])
        assert pd_zones["equilibrium"] == 0


class TestBreakerBlocks:
    def test_bullish_ob_becomes_bearish_breaker(self):
        # Bullish OB at index 2 (bottom=98, top=100). Price closes below 98 at index 5.
        obs = [{"type": "bullish", "top": 100, "bottom": 98, "index": 2}]
        rows = [
            (100, 102, 98, 101, 1000),  # 0
            (101, 103, 99, 102, 1000),  # 1
            (102, 103, 98, 99, 1000),   # 2 - the OB
            (99, 101, 98, 100, 1000),   # 3
            (100, 101, 97, 99, 1000),   # 4
            (99, 100, 95, 96, 1000),    # 5 - close below OB bottom (98)
        ]
        df = _make_df(rows)
        breakers = detect_breaker_blocks(obs, df)
        assert len(breakers) == 1
        assert breakers[0]["type"] == "bearish_breaker"

    def test_valid_ob_not_broken(self):
        obs = [{"type": "bullish", "top": 100, "bottom": 98, "index": 2}]
        rows = [
            (100, 102, 98, 101, 1000),
            (101, 103, 99, 102, 1000),
            (102, 103, 98, 99, 1000),
            (99, 101, 98, 100, 1000),
            (100, 105, 99, 104, 1000),  # bounced above top → valid OB
        ]
        df = _make_df(rows)
        breakers = detect_breaker_blocks(obs, df)
        assert len(breakers) == 0


class TestInducements:
    def test_bullish_inducement_detected(self):
        # Three swing lows: 90 (major), 95, 93. Price sweeps 93 but stays above 90.
        swing_lows = [
            {"price": 90, "index": 0},
            {"price": 95, "index": 5},
            {"price": 93, "index": 10},
        ]
        swing_highs = [
            {"price": 100, "index": 2},
            {"price": 102, "index": 7},
            {"price": 101, "index": 12},
        ]
        # At index 11, wick below 93 but close above.
        rows = [(100, 102, 98, 101, 1000)] * 11 + [(94, 95, 91, 94, 1000)] + [(94, 96, 93, 95, 1000)] * 3
        df = _make_df(rows)
        inds = detect_inducements(swing_highs, swing_lows, df)
        bullish = [i for i in inds if i["type"] == "bullish_inducement"]
        assert len(bullish) >= 1


class TestLiquidityVoids:
    def test_large_body_no_wicks_detected(self):
        # 15 normal candles + 1 large body, no wicks.
        rows = [(100, 102, 98, 101, 1000)] * 15 + [(100, 112, 100, 112, 2000)]
        df = _make_df(rows)
        atr = _calculate_atr_series(df, period=14)
        voids = detect_liquidity_voids(df, atr, min_body_atr=2.0)
        assert len(voids) >= 1
        assert voids[-1]["type"] == "bullish"

    def test_wicky_candle_not_void(self):
        # Large body but big wicks (>15%).
        rows = [(100, 102, 98, 101, 1000)] * 15 + [(100, 115, 95, 112, 2000)]
        df = _make_df(rows)
        atr = _calculate_atr_series(df, period=14)
        voids = detect_liquidity_voids(df, atr, min_body_atr=2.0)
        # The wick ratio: upper=115-112=3, lower=100-95=5, body=12. 5/12=0.42>0.15 → filtered.
        assert all(v["index"] != 15 for v in voids)


class TestAnalyzeICTPhase3:
    def test_output_has_phase3_keys(self):
        rows = [(100, 102, 98, 101, 1000)] * 50
        df = _make_df(rows)
        result = analyze_ict(df, 101.0)
        assert "price_zone" in result
        assert "premium_discount" in result
        assert "ote" in result
        assert "breaker_blocks" in result
        assert "inducements" in result
        assert "liquidity_voids" in result

    def test_order_blocks_have_displacement_flag(self):
        # Build data with a clear impulse candle.
        rows = [(100, 102, 98, 101, 1000)] * 15
        rows.append((101, 102, 99, 99, 1000))    # bearish candle = bullish OB candidate
        rows.append((99, 120, 98, 119, 5000))     # huge bullish impulse
        rows += [(119, 121, 117, 120, 1000)] * 5
        df = _make_df(rows)
        result = analyze_ict(df, 120.0)
        for ob in result["order_blocks"]:
            assert "displacement" in ob
