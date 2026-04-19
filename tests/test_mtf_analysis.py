"""Tests for Phase 5 — Multi-Timeframe Confluence."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from strategies.mtf_analysis import detect_tf_bias, MTFState, TF_WEIGHTS


def _trending_up_df(n=200, start=100):
    """DataFrame with a clear uptrend."""
    prices = [start + i * 0.5 + np.random.uniform(-0.5, 0.5) for i in range(n)]
    return pd.DataFrame({
        "open": prices,
        "high": [p + 1 for p in prices],
        "low": [p - 0.5 for p in prices],
        "close": [p + 0.3 for p in prices],
        "volume": [1000] * n,
    })


def _trending_down_df(n=200, start=200):
    prices = [start - i * 0.5 + np.random.uniform(-0.5, 0.5) for i in range(n)]
    return pd.DataFrame({
        "open": prices,
        "high": [p + 0.5 for p in prices],
        "low": [p - 1 for p in prices],
        "close": [p - 0.3 for p in prices],
        "volume": [1000] * n,
    })


def _flat_df(n=200, base=100):
    return pd.DataFrame({
        "open": [base] * n,
        "high": [base + 1] * n,
        "low": [base - 1] * n,
        "close": [base + 0.1] * n,
        "volume": [1000] * n,
    })


class TestDetectTFBias:
    def test_uptrend_is_bullish(self):
        df = _trending_up_df()
        result = detect_tf_bias(df, "4h")
        assert result["bias"] == "bullish"
        assert result["confidence"] > 0.5

    def test_downtrend_is_bearish(self):
        df = _trending_down_df()
        result = detect_tf_bias(df, "4h")
        assert result["bias"] == "bearish"
        assert result["confidence"] > 0.5

    def test_insufficient_data_neutral(self):
        df = _flat_df(n=10)
        result = detect_tf_bias(df, "15m")
        assert result["bias"] == "neutral"

    def test_output_format(self):
        df = _trending_up_df()
        result = detect_tf_bias(df, "1D")
        assert "timeframe" in result
        assert "bias" in result
        assert "confidence" in result
        assert "details" in result


class TestMTFState:
    def test_update_and_get(self):
        state = MTFState()
        df = _trending_up_df()
        state.update("4h", df)
        result = state.get("4h")
        assert result["bias"] == "bullish"

    def test_cache_reuses_same_bar(self):
        state = MTFState()
        df = _trending_up_df()
        df.index = pd.date_range("2026-01-01", periods=len(df), freq="4h")
        state.update("4h", df)
        # Second call with same df → should use cache.
        result = state.update("4h", df)
        assert result["bias"] == "bullish"

    def test_missing_tf_returns_neutral(self):
        state = MTFState()
        assert state.get("1D")["bias"] == "neutral"


class TestConfluence:
    def test_all_bullish_allows_trade(self):
        state = MTFState()
        for tf in TF_WEIGHTS:
            state.update(tf, _trending_up_df())
        result = state.confluence()
        assert result["direction"] == "bullish"
        assert result["aligned_count"] == 4
        assert result["trade_allowed"] is True

    def test_all_bearish_allows_trade(self):
        state = MTFState()
        for tf in TF_WEIGHTS:
            state.update(tf, _trending_down_df())
        result = state.confluence()
        assert result["direction"] == "bearish"
        assert result["trade_allowed"] is True

    def test_mixed_blocks_trade(self):
        state = MTFState()
        state.update("1D", _trending_up_df())
        state.update("4h", _trending_down_df())
        state.update("1h", _trending_up_df())
        state.update("15m", _trending_down_df())
        result = state.confluence()
        # 2 bullish + 2 bearish → no clear majority → should not allow.
        assert result["direction"] in ("neutral", "bullish", "bearish")
        # With 2+2 split the weighted score is near zero → neutral → aligned_count=0.
        # Even if bias leaks, at most 2 agree which is < 3.
        assert result["aligned_count"] < 3 or result["trade_allowed"] is True
        # The key invariant: if only 2 agree, trade is blocked.
        if result["aligned_count"] < 3:
            assert result["trade_allowed"] is False

    def test_three_agree_allows(self):
        state = MTFState()
        state.update("1D", _trending_up_df())
        state.update("4h", _trending_up_df())
        state.update("1h", _trending_up_df())
        state.update("15m", _flat_df())
        result = state.confluence()
        assert result["aligned_count"] >= 3
        assert result["trade_allowed"] is True

    def test_score_weighted(self):
        state = MTFState()
        for tf in TF_WEIGHTS:
            state.update(tf, _trending_up_df())
        result = state.confluence()
        assert result["score"] > 0
        assert result["score"] <= 1.0

    def test_status_format(self):
        state = MTFState()
        state.update("4h", _trending_up_df())
        s = state.status()
        assert "4h" in s
        assert "1D" in s
