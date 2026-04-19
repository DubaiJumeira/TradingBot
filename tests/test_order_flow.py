"""Tests for Phase 11 — Order Flow Analysis."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from strategies.order_flow import (
    Trade, OrderFlowBar, OrderFlowTracker, OrderFlowStreamer, score_order_flow,
)


def _mk_trade(price: float, size: float, side: str, ts: float) -> Trade:
    return Trade(price=price, size=size, side=side, timestamp=ts)


class TestTrade:
    def test_delta_buy(self):
        t = _mk_trade(100, 1.5, "buy", 1000)
        assert t.delta == 1.5

    def test_delta_sell(self):
        t = _mk_trade(100, 1.5, "sell", 1000)
        assert t.delta == -1.5


class TestOrderFlowBar:
    def test_delta_and_imbalance(self):
        bar = OrderFlowBar(
            start_time=0, end_time=60,
            open_price=100, high_price=101, low_price=99, close_price=100.5,
            volume=10, buy_volume=8, sell_volume=2,
        )
        assert bar.delta == 6
        assert bar.imbalance == 0.6
        assert bar.dominant_side == "buy"

    def test_zero_volume(self):
        bar = OrderFlowBar(
            start_time=0, end_time=60,
            open_price=100, high_price=100, low_price=100, close_price=100,
        )
        assert bar.imbalance == 0.0
        assert bar.dominant_side == "neutral"

    def test_to_dict(self):
        bar = OrderFlowBar(
            start_time=0, end_time=60,
            open_price=100, high_price=101, low_price=99, close_price=100,
            volume=5, buy_volume=3, sell_volume=2,
        )
        d = bar.to_dict()
        assert d["delta"] == 1
        assert d["dominant_side"] == "buy"


class TestOrderFlowTracker:
    def test_single_trade_creates_bar(self):
        tracker = OrderFlowTracker(bar_seconds=60)
        tracker.add_trade(_mk_trade(100, 1.0, "buy", 1000))
        bars = tracker.get_bars()
        assert len(bars) == 1
        assert bars[0].buy_volume == 1.0

    def test_bar_rollover(self):
        tracker = OrderFlowTracker(bar_seconds=60)
        tracker.add_trade(_mk_trade(100, 1.0, "buy", 1000))
        tracker.add_trade(_mk_trade(101, 2.0, "sell", 1061))  # new bar
        bars = tracker.get_bars()
        assert len(bars) == 2
        assert bars[0].buy_volume == 1.0
        assert bars[1].sell_volume == 2.0

    def test_multiple_trades_same_bar(self):
        tracker = OrderFlowTracker(bar_seconds=60)
        # All timestamps within the same 60s bar window starting at 960.
        tracker.add_trade(_mk_trade(100, 1.0, "buy", 961))
        tracker.add_trade(_mk_trade(101, 2.0, "buy", 980))
        tracker.add_trade(_mk_trade(99.5, 1.5, "sell", 1000))
        bars = tracker.get_bars()
        assert len(bars) == 1
        bar = bars[0]
        assert bar.buy_volume == 3.0
        assert bar.sell_volume == 1.5
        assert bar.delta == 1.5
        assert bar.high_price == 101
        assert bar.low_price == 99.5

    def test_cvd_cumulative(self):
        tracker = OrderFlowTracker(bar_seconds=60)
        tracker.add_trade(_mk_trade(100, 5.0, "buy", 1000))
        tracker.add_trade(_mk_trade(100, 2.0, "sell", 1061))
        tracker.add_trade(_mk_trade(100, 3.0, "buy", 1121))
        assert tracker.cvd() == 6.0  # 5 - 2 + 3

    def test_cvd_series(self):
        tracker = OrderFlowTracker(bar_seconds=60)
        tracker.add_trade(_mk_trade(100, 5.0, "buy", 1000))
        tracker.add_trade(_mk_trade(100, 2.0, "sell", 1061))
        series = tracker.cvd_series()
        assert series == [5.0, 3.0]

    def test_detect_absorption(self):
        tracker = OrderFlowTracker(bar_seconds=60)
        # First establish a baseline range with a wide bar.
        base = 1000
        for i in range(4):
            ts = base + i * 60
            tracker.add_trade(_mk_trade(100, 5, "buy", ts))
            tracker.add_trade(_mk_trade(104, 3, "buy", ts + 10))
            tracker.add_trade(_mk_trade(96, 2, "sell", ts + 20))
            tracker.add_trade(_mk_trade(100, 4, "sell", ts + 30))
        # Absorption bar: heavy buy delta, tiny body, low range.
        ts = base + 4 * 60
        tracker.add_trade(_mk_trade(100.0, 8, "buy", ts))
        tracker.add_trade(_mk_trade(100.05, 2, "buy", ts + 30))  # closes near open

        signals = tracker.detect_absorption()
        # At least one absorption signal should fire on the narrow buy-dominant bar.
        assert any(s["type"] == "absorption" for s in signals)

    def test_detect_bearish_divergence(self):
        tracker = OrderFlowTracker(bar_seconds=60)
        # Price rising, but CVD declining (more sells absorbed).
        prices = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]
        ts_start = 1000
        for i, p in enumerate(prices):
            ts = ts_start + i * 60
            # Each bar: more sell volume than buy.
            tracker.add_trade(_mk_trade(p, 1, "buy", ts))
            tracker.add_trade(_mk_trade(p, 3, "sell", ts + 30))
        div = tracker.detect_divergence(lookback=10)
        assert div is not None
        assert div["type"] == "bearish_divergence"

    def test_detect_bullish_divergence(self):
        tracker = OrderFlowTracker(bar_seconds=60)
        prices = [100, 99, 98, 97, 96, 95, 94, 93, 92, 91]
        ts_start = 1000
        for i, p in enumerate(prices):
            ts = ts_start + i * 60
            tracker.add_trade(_mk_trade(p, 3, "buy", ts))
            tracker.add_trade(_mk_trade(p, 1, "sell", ts + 30))
        div = tracker.detect_divergence(lookback=10)
        assert div is not None
        assert div["type"] == "bullish_divergence"

    def test_no_divergence_when_aligned(self):
        tracker = OrderFlowTracker(bar_seconds=60)
        # Price up and CVD up — normal trend, no divergence.
        for i in range(10):
            p = 100 + i
            ts = 1000 + i * 60
            tracker.add_trade(_mk_trade(p, 5, "buy", ts))
            tracker.add_trade(_mk_trade(p, 1, "sell", ts + 30))
        div = tracker.detect_divergence(lookback=10)
        assert div is None

    def test_analyze_empty(self):
        tracker = OrderFlowTracker(bar_seconds=60)
        a = tracker.analyze()
        assert a["bars"] == 0
        assert a["cvd"] == 0.0

    def test_analyze_full(self):
        tracker = OrderFlowTracker(bar_seconds=60)
        tracker.add_trade(_mk_trade(100, 5, "buy", 961))
        tracker.add_trade(_mk_trade(100, 2, "sell", 990))
        a = tracker.analyze()
        assert a["bars"] == 1
        assert a["cvd"] == 3.0
        assert a["dominant_side"] == "buy"

    def test_reset(self):
        tracker = OrderFlowTracker(bar_seconds=60)
        tracker.add_trade(_mk_trade(100, 1, "buy", 1000))
        tracker.reset()
        assert tracker.get_bars() == []

    def test_max_bars_limit(self):
        tracker = OrderFlowTracker(bar_seconds=1, max_bars=3)
        for i in range(10):
            tracker.add_trade(_mk_trade(100, 1, "buy", 1000 + i))
        bars = tracker.get_bars(include_current=False)
        assert len(bars) <= 3


class TestOrderFlowStreamer:
    def test_starts_and_stops(self):
        tracker = OrderFlowTracker(bar_seconds=60)
        calls = []

        def fetcher(symbol):
            calls.append(symbol)
            return [_mk_trade(100, 1, "buy", time.time())]

        streamer = OrderFlowStreamer("BTC/USDT", tracker, fetcher, poll_interval=0.05)
        streamer.start()
        time.sleep(0.2)
        streamer.stop()
        assert len(calls) >= 2
        assert tracker.get_bars()[0].volume >= 1


class TestScoreOrderFlow:
    def test_aligned_long(self):
        analysis = {
            "bars": 5, "cvd": 10, "delta_last": 5, "imbalance_last": 0.7,
            "dominant_side": "buy", "absorption": [], "divergence": None,
        }
        score, reasons = score_order_flow(analysis, "long")
        assert score >= 10
        assert any("buy-dominant" in r for r in reasons)

    def test_against_trade(self):
        analysis = {
            "bars": 5, "cvd": 10, "delta_last": 5, "imbalance_last": 0.8,
            "dominant_side": "buy", "absorption": [], "divergence": None,
        }
        score, _ = score_order_flow(analysis, "short")
        assert score < 0

    def test_bullish_divergence_boost(self):
        analysis = {
            "bars": 10, "cvd": -5, "delta_last": -1, "imbalance_last": 0.3,
            "dominant_side": "sell", "absorption": [],
            "divergence": {"type": "bullish_divergence", "price_change": -5, "cvd_change": 3},
        }
        score, reasons = score_order_flow(analysis, "long")
        assert score >= 10
        assert any("bullish CVD divergence" in r for r in reasons)

    def test_absorption_reversal_long(self):
        analysis = {
            "bars": 5, "cvd": 0, "delta_last": 0, "imbalance_last": 0.3,
            "dominant_side": "neutral",
            "absorption": [{"type": "absorption", "side": "sell", "price": 100, "delta": -5}],
            "divergence": None,
        }
        score, reasons = score_order_flow(analysis, "long")
        assert score >= 7
        assert any("absorption of sell" in r for r in reasons)

    def test_empty_analysis(self):
        score, reasons = score_order_flow({"bars": 0}, "long")
        assert score == 0
        assert reasons == []
