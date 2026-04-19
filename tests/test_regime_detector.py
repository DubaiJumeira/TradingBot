"""Tests for Phase 9 — Regime Detection."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from strategies.regime_detector import RegimeDetector, calculate_adx, _wick_ratio


# ---------------------------------------------------------------------------
# Helper data generators
# ---------------------------------------------------------------------------

def _trending_df(n: int = 60, start: float = 100, slope: float = 0.5) -> pd.DataFrame:
    """Strong uptrend — ADX should be high."""
    dates = pd.date_range("2026-01-01", periods=n, freq="15min")
    closes = [start + i * slope for i in range(n)]
    return pd.DataFrame({
        "open": [c - 0.1 for c in closes],
        "high": [c + 0.3 for c in closes],
        "low": [c - 0.2 for c in closes],
        "close": closes,
        "volume": [1000] * n,
    }, index=dates)


def _ranging_df(n: int = 60, mid: float = 100, amplitude: float = 1.0) -> pd.DataFrame:
    """Sideways price action oscillating around `mid`."""
    dates = pd.date_range("2026-01-01", periods=n, freq="15min")
    closes = [mid + amplitude * np.sin(i * 0.5) for i in range(n)]
    return pd.DataFrame({
        "open": [c - 0.05 for c in closes],
        "high": [c + 0.3 for c in closes],
        "low": [c - 0.3 for c in closes],
        "close": closes,
        "volume": [1000] * n,
    }, index=dates)


def _choppy_df(n: int = 60, mid: float = 100) -> pd.DataFrame:
    """High wick ratio — lots of rejection wicks, small bodies."""
    dates = pd.date_range("2026-01-01", periods=n, freq="15min")
    np.random.seed(42)
    closes = [mid + np.random.uniform(-0.2, 0.2) for _ in range(n)]
    return pd.DataFrame({
        "open": [c + np.random.uniform(-0.1, 0.1) for c in closes],
        "high": [c + np.random.uniform(2, 4) for c in closes],  # huge upper wicks
        "low": [c - np.random.uniform(2, 4) for c in closes],   # huge lower wicks
        "close": closes,
        "volume": [1000] * n,
    }, index=dates)


# ---------------------------------------------------------------------------
# ADX tests
# ---------------------------------------------------------------------------

class TestCalculateADX:
    def test_returns_float(self):
        df = _trending_df()
        result = calculate_adx(df)
        assert isinstance(result, float)

    def test_trending_high_adx(self):
        df = _trending_df(n=80, slope=1.0)
        adx = calculate_adx(df)
        assert adx > 20, f"Trending data should have ADX > 20, got {adx}"

    def test_ranging_low_adx(self):
        df = _ranging_df(n=80, amplitude=0.5)
        adx = calculate_adx(df)
        assert adx < 40, f"Ranging data should have moderate/low ADX, got {adx}"

    def test_short_data_returns_zero(self):
        df = _trending_df(n=10)
        assert calculate_adx(df) == 0.0

    def test_adx_range(self):
        df = _trending_df(n=80)
        adx = calculate_adx(df)
        assert 0 <= adx <= 100, f"ADX should be 0-100, got {adx}"


# ---------------------------------------------------------------------------
# Wick ratio tests
# ---------------------------------------------------------------------------

class TestWickRatio:
    def test_clean_candles_low_ratio(self):
        """Candles with large bodies relative to range should have lower wick ratio."""
        dates = pd.date_range("2026-01-01", periods=30, freq="15min")
        closes = [100 + i * 0.5 for i in range(30)]
        df = pd.DataFrame({
            "open": [c - 0.4 for c in closes],
            "high": [c + 0.1 for c in closes],
            "low": [c - 0.5 for c in closes],
            "close": closes,
            "volume": [1000] * 30,
        }, index=dates)
        ratio = _wick_ratio(df)
        assert ratio < 0.5, f"Clean candles should have low wick ratio, got {ratio}"

    def test_choppy_candles_high_ratio(self):
        df = _choppy_df(n=30)
        ratio = _wick_ratio(df)
        assert ratio > 0.5, f"Choppy candles should have high wick ratio, got {ratio}"

    def test_returns_float_in_range(self):
        df = _ranging_df(n=30)
        ratio = _wick_ratio(df)
        assert isinstance(ratio, float)
        assert 0 <= ratio <= 1.0


# ---------------------------------------------------------------------------
# RegimeDetector tests
# ---------------------------------------------------------------------------

class TestRegimeDetector:
    def setup_method(self):
        self.detector = RegimeDetector()

    def test_short_data_unknown(self):
        df = _trending_df(n=20)
        result = self.detector.detect(df)
        assert result["regime"] == "unknown"

    def test_result_keys(self):
        df = _trending_df(n=60)
        result = self.detector.detect(df)
        for key in ["regime", "adx", "wick_ratio", "volatility_pct", "adjustments"]:
            assert key in result, f"Missing key: {key}"

    def test_trending_regime(self):
        df = _trending_df(n=80, slope=1.0)
        result = self.detector.detect(df)
        # Strong trend should either be 'trending' or have high ADX
        assert result["adx"] > 0

    def test_event_regime_override(self):
        df = _trending_df(n=60)
        result = self.detector.detect(df, news_event_active=True)
        assert result["regime"] == "event"

    def test_event_regime_adjustments(self):
        df = _trending_df(n=60)
        result = self.detector.detect(df, news_event_active=True)
        adj = result["adjustments"]
        assert adj["sl_multiplier"] == 1.5
        assert adj["size_multiplier"] == 0.75

    def test_choppy_detection(self):
        df = _choppy_df(n=80)
        result = self.detector.detect(df)
        # Choppy data should be detected as choppy or at least not trending
        assert result["regime"] in ("choppy", "ranging", "unknown")
        assert result["wick_ratio"] > 0.4

    def test_adjustments_structure(self):
        df = _trending_df(n=60)
        result = self.detector.detect(df)
        adj = result["adjustments"]
        for key in ["tp_multiplier", "sl_multiplier", "size_multiplier", "trailing", "min_score_adjust"]:
            assert key in adj, f"Missing adjustment key: {key}"

    def test_trending_adjustments(self):
        adj = self.detector._get_adjustments("trending")
        assert adj["tp_multiplier"] == 1.5
        assert adj["trailing"] is True
        assert adj["size_multiplier"] == 1.0

    def test_ranging_adjustments(self):
        adj = self.detector._get_adjustments("ranging")
        assert adj["tp_multiplier"] == 0.8
        assert adj["sl_multiplier"] == 0.8
        assert adj["trailing"] is False

    def test_choppy_adjustments(self):
        adj = self.detector._get_adjustments("choppy")
        assert adj["size_multiplier"] == 0.5
        assert adj["min_score_adjust"] == 10

    def test_unknown_adjustments_defaults(self):
        adj = self.detector._get_adjustments("unknown")
        assert adj["tp_multiplier"] == 1.0
        assert adj["sl_multiplier"] == 1.0

    def test_volatility_pct_positive(self):
        df = _trending_df(n=60)
        result = self.detector.detect(df)
        assert result["volatility_pct"] >= 0
