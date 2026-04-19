"""
Tests for Phase 6 — Risk Management.

Covers:
    - ATR calculation and ATR-based position sizing
    - Drawdown circuit breaker (trigger, reset, edge cases)
    - Correlation-aware exposure limits
    - Trailing stop management (longs, shorts, activation, step)
    - RiskManager composite pre-trade checks
    - PaperTrader.update_sl()
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

from config import Config, CORRELATION_GROUPS, get_correlation_group
from strategies.risk_manager import (
    RiskManager,
    DrawdownMonitor,
    calculate_atr,
    atr_position_size,
    calculate_trailing_stop,
    check_exposure,
)


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def _make_df(prices: list[tuple[float, float, float, float]], volume: float = 1000) -> pd.DataFrame:
    """Create a DataFrame from (open, high, low, close) tuples."""
    return pd.DataFrame(
        [{"open": o, "high": h, "low": l, "close": c, "volume": volume}
         for o, h, l, c in prices]
    )


# -----------------------------------------------------------------------
# ATR calculation
# -----------------------------------------------------------------------

class TestCalculateATR:
    def test_basic_atr(self):
        # 16 candles → ATR(14) uses last 14 true ranges.
        prices = [(100, 102, 99, 101)] * 16
        df = _make_df(prices)
        atr = calculate_atr(df, period=14)
        # Each candle: high-low=3, |high-prev_close|=1, |low-prev_close|=2
        # TR = max(3, 1, 2) = 3 for each. ATR = 3.0.
        assert atr == pytest.approx(3.0, abs=0.01)

    def test_not_enough_data(self):
        df = _make_df([(100, 102, 99, 101)] * 5)
        assert calculate_atr(df, period=14) == 0.0

    def test_increasing_volatility(self):
        # First 8 candles: narrow range. Next 8: wide range.
        narrow = [(100, 101, 99.5, 100.5)] * 8
        wide = [(100, 110, 90, 100)] * 8
        df = _make_df(narrow + wide)
        atr = calculate_atr(df, period=7)
        # ATR(7) over last 7 TRs should reflect the wide candles.
        assert atr > 10  # wide candle TR ≈ 20 (high-low)

    def test_single_candle_tr(self):
        # 2 candles needed for 1 TR.
        prices = [(100, 105, 95, 102), (102, 108, 97, 104)]
        df = _make_df(prices)
        atr = calculate_atr(df, period=1)
        # TR = max(108-97, |108-102|, |97-102|) = max(11, 6, 5) = 11
        assert atr == pytest.approx(11.0, abs=0.01)


# -----------------------------------------------------------------------
# ATR-based position sizing
# -----------------------------------------------------------------------

class TestATRPositionSize:
    def test_normal_conditions(self):
        # When atr_scale * sl_distance >= atr, factor = 1.0 → same as basic sizing.
        size = atr_position_size(
            balance=10000, risk_pct=1.0, entry=100, sl=98,
            atr=2.0, atr_scale=1.5,
        )
        # sl_dist=2, atr_scale*sl_dist=3.0 >= atr=2.0 → factor=1.0
        # base_risk=100, qty=100/2=50, size=50*100=5000
        assert size == 5000.0

    def test_high_volatility_reduces_size(self):
        # ATR much larger than SL distance → factor < 1.0.
        size = atr_position_size(
            balance=10000, risk_pct=1.0, entry=100, sl=98,
            atr=6.0, atr_scale=1.5,
        )
        # sl_dist=2, atr_scale*sl_dist=3.0, atr=6.0 → factor=3/6=0.5
        # base_risk=100, qty=(100*0.5)/2=25, size=25*100=2500
        assert size == 2500.0

    def test_zero_sl_distance(self):
        assert atr_position_size(10000, 1.0, 100, 100, 3.0) == 0.0

    def test_zero_atr(self):
        assert atr_position_size(10000, 1.0, 100, 98, 0.0) == 0.0

    def test_low_volatility_caps_at_full_size(self):
        # ATR very small → factor caps at 1.0.
        size = atr_position_size(
            balance=10000, risk_pct=1.0, entry=100, sl=98,
            atr=0.5, atr_scale=1.5,
        )
        # factor = min(1.0, 3.0/0.5) = min(1.0, 6.0) = 1.0
        assert size == 5000.0


# -----------------------------------------------------------------------
# Drawdown circuit breaker
# -----------------------------------------------------------------------

class TestDrawdownMonitor:
    def test_no_drawdown_initially(self):
        dm = DrawdownMonitor(10000, max_drawdown_pct=10.0)
        assert dm.drawdown_pct == 0.0
        assert dm.is_breaker_active is False

    def test_equity_growth_updates_peak(self):
        dm = DrawdownMonitor(10000, max_drawdown_pct=10.0)
        dm.update(11000)
        assert dm.peak_equity == 11000
        assert dm.drawdown_pct == 0.0

    def test_breaker_triggers_at_threshold(self):
        dm = DrawdownMonitor(10000, max_drawdown_pct=10.0)
        dm.update(10000)
        assert dm.is_breaker_active is False

        # Drop to 9000 → 10% drawdown → triggers.
        dm.update(9000)
        assert dm.drawdown_pct == pytest.approx(10.0)
        assert dm.is_breaker_active is True

    def test_breaker_triggers_beyond_threshold(self):
        dm = DrawdownMonitor(10000, max_drawdown_pct=10.0)
        dm.update(8500)  # 15% drawdown
        assert dm.is_breaker_active is True

    def test_breaker_resets_when_recovered(self):
        dm = DrawdownMonitor(10000, max_drawdown_pct=10.0)
        dm.update(8500)
        assert dm.is_breaker_active is True

        # Recover above threshold (peak stays 10000).
        dm.update(9100)  # 9% drawdown → below 10%.
        assert dm.is_breaker_active is False

    def test_peak_updates_after_recovery(self):
        dm = DrawdownMonitor(10000, max_drawdown_pct=10.0)
        dm.update(12000)  # new peak
        dm.update(11000)  # 8.3% dd from 12000
        assert dm.peak_equity == 12000
        assert dm.is_breaker_active is False

    def test_exactly_at_threshold(self):
        dm = DrawdownMonitor(10000, max_drawdown_pct=10.0)
        dm.update(9000)  # exactly 10%
        assert dm.is_breaker_active is True

    def test_status_dict(self):
        dm = DrawdownMonitor(10000, max_drawdown_pct=10.0)
        dm.update(9500)
        s = dm.status()
        assert s["peak_equity"] == 10000
        assert s["current_equity"] == 9500
        assert s["drawdown_pct"] == 5.0
        assert s["breaker_active"] is False

    def test_zero_peak(self):
        dm = DrawdownMonitor(0, max_drawdown_pct=10.0)
        dm.update(0)
        assert dm.drawdown_pct == 0.0


# -----------------------------------------------------------------------
# Correlation-aware exposure limits
# -----------------------------------------------------------------------

class TestCorrelationGroups:
    def test_get_correlation_group_known(self):
        assert get_correlation_group("BTC/USDT") == "crypto"
        assert get_correlation_group("XAUUSD") == "safe_haven"
        assert get_correlation_group("SPX500") == "us_indices"

    def test_get_correlation_group_unknown(self):
        assert get_correlation_group("FAKE/COIN") is None

    def test_groups_have_required_keys(self):
        for name, group in CORRELATION_GROUPS.items():
            assert "symbols" in group, f"{name} missing symbols"
            assert "max_positions" in group, f"{name} missing max_positions"
            assert len(group["symbols"]) > 0


class TestCheckExposure:
    def test_no_positions_allowed(self):
        result = check_exposure("BTC/USDT", {})
        assert result["allowed"] is True
        assert result["group"] == "crypto"
        assert result["current_count"] == 0

    def test_under_limit_allowed(self):
        positions = {"1": {"symbol": "BTC/USDT", "side": "long"}}
        result = check_exposure("ETH/USDT", positions)
        # Crypto group max=2, 1 open → 1 more allowed.
        assert result["allowed"] is True
        assert result["current_count"] == 1

    def test_at_limit_blocked(self):
        positions = {
            "1": {"symbol": "BTC/USDT", "side": "long"},
            "2": {"symbol": "ETH/USDT", "side": "long"},
        }
        result = check_exposure("SOL/USDT", positions)
        # Crypto group max=2, 2 open → blocked.
        assert result["allowed"] is False
        assert result["current_count"] == 2
        assert "crypto" in result["reason"]

    def test_different_group_not_affected(self):
        positions = {
            "1": {"symbol": "BTC/USDT", "side": "long"},
            "2": {"symbol": "ETH/USDT", "side": "long"},
        }
        # Gold is in safe_haven group → not blocked by crypto being full.
        result = check_exposure("XAUUSD", positions)
        assert result["allowed"] is True
        assert result["group"] == "safe_haven"

    def test_single_max_position_group(self):
        positions = {"1": {"symbol": "XAUUSD", "side": "long"}}
        # safe_haven max=1, 1 open → blocked.
        result = check_exposure("XAUUSD", positions)
        assert result["allowed"] is False

    def test_unknown_symbol_always_allowed(self):
        positions = {"1": {"symbol": "BTC/USDT"}, "2": {"symbol": "ETH/USDT"}}
        result = check_exposure("UNKNOWN/PAIR", positions)
        assert result["allowed"] is True
        assert result["group"] is None

    def test_indices_limit(self):
        positions = {
            "1": {"symbol": "SPX500", "side": "short"},
            "2": {"symbol": "US30", "side": "short"},
        }
        result = check_exposure("NAS100", positions)
        assert result["allowed"] is False
        assert result["group"] == "us_indices"


# -----------------------------------------------------------------------
# Trailing stop management
# -----------------------------------------------------------------------

class TestTrailingStop:
    def test_long_not_activated_below_threshold(self):
        # Entry=100, SL=98 → risk=2. Need unrealized RR >= 1.0 → price >= 102.
        result = calculate_trailing_stop(
            side="long", entry=100, current_sl=98,
            current_price=101.5, tp=106,
            activation_rr=1.0, step_pct=0.5,
        )
        assert result is None  # unrealized RR = 1.5/2 = 0.75 < 1.0

    def test_long_activated_moves_sl_up(self):
        # Entry=100, SL=98 → risk=2. Price=103 → RR=3/2=1.5 ≥ 1.0.
        result = calculate_trailing_stop(
            side="long", entry=100, current_sl=98,
            current_price=103, tp=106,
            activation_rr=1.0, step_pct=0.5,
        )
        # new_sl = 103 - 100*0.5% = 103 - 0.5 = 102.5
        assert result == 102.5

    def test_long_sl_never_moves_backward(self):
        # SL already at 102, price=103 → new_sl=102.5 → not better than 102?
        # Actually 102.5 > 102 → should still move forward.
        result = calculate_trailing_stop(
            side="long", entry=100, current_sl=102,
            current_price=103, tp=106,
            activation_rr=1.0, step_pct=0.5,
        )
        assert result == 102.5

    def test_long_sl_does_not_move_backward(self):
        # SL already at 103, price=103 → new_sl=102.5 → worse → None.
        result = calculate_trailing_stop(
            side="long", entry=100, current_sl=103,
            current_price=103.2, tp=106,
            activation_rr=1.0, step_pct=0.5,
        )
        # new_sl = 103.2 - 0.5 = 102.7, but 102.7 < 103 → None
        assert result is None

    def test_short_not_activated_below_threshold(self):
        # Entry=100, SL=102 → risk=2. Need price <= 98 for RR=1.0.
        result = calculate_trailing_stop(
            side="short", entry=100, current_sl=102,
            current_price=99, tp=94,
            activation_rr=1.0, step_pct=0.5,
        )
        # RR = (100-99)/2 = 0.5 < 1.0.
        assert result is None

    def test_short_activated_moves_sl_down(self):
        # Entry=100, SL=102 → risk=2. Price=97 → RR=3/2=1.5 ≥ 1.0.
        result = calculate_trailing_stop(
            side="short", entry=100, current_sl=102,
            current_price=97, tp=94,
            activation_rr=1.0, step_pct=0.5,
        )
        # new_sl = 97 + 100*0.5% = 97.5, which is < 102 → good.
        assert result == 97.5

    def test_short_sl_does_not_move_backward(self):
        # SL already at 97, price=97.3 → new_sl=97.8 → worse (higher) → None.
        result = calculate_trailing_stop(
            side="short", entry=100, current_sl=97,
            current_price=97.3, tp=94,
            activation_rr=1.0, step_pct=0.5,
        )
        # new_sl = 97.3 + 0.5 = 97.8 > 97 → worse for short → None
        assert result is None

    def test_zero_risk_returns_none(self):
        result = calculate_trailing_stop(
            side="long", entry=100, current_sl=100,
            current_price=105, tp=110,
        )
        assert result is None

    def test_custom_activation_rr(self):
        # Higher activation threshold: need RR >= 2.0.
        result = calculate_trailing_stop(
            side="long", entry=100, current_sl=98,
            current_price=103, tp=106,
            activation_rr=2.0, step_pct=0.5,
        )
        # RR = 3/2 = 1.5 < 2.0 → not activated.
        assert result is None

        # Price=105 → RR = 5/2 = 2.5 ≥ 2.0.
        result = calculate_trailing_stop(
            side="long", entry=100, current_sl=98,
            current_price=105, tp=106,
            activation_rr=2.0, step_pct=0.5,
        )
        assert result == 104.5


# -----------------------------------------------------------------------
# RiskManager composite
# -----------------------------------------------------------------------

class TestRiskManager:
    def test_allows_trade_by_default(self):
        rm = RiskManager(starting_balance=10000, max_drawdown_pct=10.0)
        result = rm.pre_trade_check("BTC/USDT", {})
        assert result["allowed"] is True
        assert result["reasons"] == []

    def test_blocks_when_drawdown_breaker_active(self):
        rm = RiskManager(starting_balance=10000, max_drawdown_pct=10.0)
        rm.update_equity(8500)  # 15% drawdown
        result = rm.pre_trade_check("BTC/USDT", {})
        assert result["allowed"] is False
        assert any("Circuit breaker" in r for r in result["reasons"])

    def test_blocks_when_exposure_exceeded(self):
        rm = RiskManager(starting_balance=10000, max_drawdown_pct=10.0)
        positions = {
            "1": {"symbol": "BTC/USDT"},
            "2": {"symbol": "ETH/USDT"},
        }
        result = rm.pre_trade_check("SOL/USDT", positions)
        assert result["allowed"] is False
        assert any("Correlation" in r for r in result["reasons"])

    def test_blocks_with_both_reasons(self):
        rm = RiskManager(starting_balance=10000, max_drawdown_pct=10.0)
        rm.update_equity(8500)
        positions = {
            "1": {"symbol": "BTC/USDT"},
            "2": {"symbol": "ETH/USDT"},
        }
        result = rm.pre_trade_check("SOL/USDT", positions)
        assert result["allowed"] is False
        assert len(result["reasons"]) == 2

    def test_status_includes_drawdown(self):
        rm = RiskManager(starting_balance=10000)
        s = rm.status()
        assert "drawdown" in s
        assert s["drawdown"]["breaker_active"] is False


# -----------------------------------------------------------------------
# PaperTrader.update_sl
# -----------------------------------------------------------------------

class TestPaperTraderUpdateSL:
    def test_update_sl_on_open_position(self, tmp_path, monkeypatch):
        from exchange_handler import PaperTrader

        # Redirect state file to tmp_path.
        monkeypatch.setattr(
            PaperTrader, "_state_file",
            lambda self: str(tmp_path / "paper_trades.json"),
        )
        pt = PaperTrader(10000)
        pt.open_trade("BTC/USDT", "long", 100, 95, 110, 1000)

        tid = str(pt.trade_id)
        assert pt.positions[tid]["sl_price"] == 95

        result = pt.update_sl(tid, 98)
        assert result is True
        assert pt.positions[tid]["sl_price"] == 98

    def test_update_sl_missing_position(self, tmp_path, monkeypatch):
        from exchange_handler import PaperTrader

        monkeypatch.setattr(
            PaperTrader, "_state_file",
            lambda self: str(tmp_path / "paper_trades.json"),
        )
        pt = PaperTrader(10000)
        assert pt.update_sl("999", 50) is False


# -----------------------------------------------------------------------
# Config values
# -----------------------------------------------------------------------

class TestRiskConfig:
    def test_max_drawdown_default(self):
        assert Config.MAX_DRAWDOWN_PCT == 10.0

    def test_trailing_stop_defaults(self):
        assert Config.TRAILING_STOP_ACTIVATION_RR == 1.0
        assert Config.TRAILING_STOP_STEP_PCT == 0.5

    def test_atr_scale_default(self):
        assert Config.ATR_POSITION_SCALE == 1.5
