"""
Phase 11 — Order Flow Analysis

Tracks bid/ask aggressive fills to compute delta (buy volume - sell volume),
cumulative volume delta (CVD), absorption, and imbalance signals for
ICT/Wyckoff confluence.

Concepts:
    - Delta:   volume_buy - volume_sell per bar (from trade aggressor side)
    - CVD:     running sum of deltas — trend of aggression
    - Absorption: price barely moves despite heavy one-sided delta →
                  institutions soaking liquidity at key level
    - Imbalance: abs(delta) / volume > 0.65 → aggressive one-sided
    - Divergence: price makes HH but CVD makes LH (or vice versa) → reversal signal

The OrderFlowTracker accepts trades (either from websocket or synthetic)
and produces OrderFlowBar aggregations on a time or tick basis.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass
class Trade:
    """One aggressive trade print."""
    price: float
    size: float
    side: str          # "buy" or "sell" (aggressor side)
    timestamp: float   # unix seconds

    @property
    def delta(self) -> float:
        return self.size if self.side == "buy" else -self.size


@dataclass
class OrderFlowBar:
    """One aggregated bar of order flow data."""
    start_time: float
    end_time: float
    open_price: float
    high_price: float
    low_price: float
    close_price: float
    volume: float = 0.0
    buy_volume: float = 0.0
    sell_volume: float = 0.0
    trade_count: int = 0

    @property
    def delta(self) -> float:
        return self.buy_volume - self.sell_volume

    @property
    def imbalance(self) -> float:
        """abs(delta) / volume — 0=balanced, 1=fully one-sided."""
        if self.volume == 0:
            return 0.0
        return abs(self.delta) / self.volume

    @property
    def dominant_side(self) -> str:
        return "buy" if self.delta > 0 else ("sell" if self.delta < 0 else "neutral")

    def to_dict(self) -> dict[str, Any]:
        return {
            "start_time": self.start_time,
            "end_time": self.end_time,
            "open": self.open_price,
            "high": self.high_price,
            "low": self.low_price,
            "close": self.close_price,
            "volume": round(self.volume, 4),
            "buy_volume": round(self.buy_volume, 4),
            "sell_volume": round(self.sell_volume, 4),
            "delta": round(self.delta, 4),
            "imbalance": round(self.imbalance, 3),
            "dominant_side": self.dominant_side,
            "trade_count": self.trade_count,
        }


class OrderFlowTracker:
    """
    Aggregates raw trades into fixed-duration bars and computes order flow
    metrics. Thread-safe for concurrent websocket callback + reader access.

    Usage:
        tracker = OrderFlowTracker(bar_seconds=60)
        tracker.add_trade(Trade(price=50000, size=0.1, side="buy", timestamp=...))
        bars = tracker.get_bars()
        analysis = tracker.analyze()
    """

    def __init__(self, bar_seconds: int = 60, max_bars: int = 500) -> None:
        self.bar_seconds = bar_seconds
        self.max_bars = max_bars
        self._bars: deque[OrderFlowBar] = deque(maxlen=max_bars)
        self._current: OrderFlowBar | None = None
        self._lock = threading.Lock()

    def add_trade(self, trade: Trade) -> None:
        """Incorporate one trade into the current bar. Rolls bar if needed."""
        with self._lock:
            if self._current is None or trade.timestamp >= self._current.end_time:
                if self._current is not None:
                    self._bars.append(self._current)
                start = int(trade.timestamp // self.bar_seconds) * self.bar_seconds
                self._current = OrderFlowBar(
                    start_time=start,
                    end_time=start + self.bar_seconds,
                    open_price=trade.price,
                    high_price=trade.price,
                    low_price=trade.price,
                    close_price=trade.price,
                )

            bar = self._current
            bar.close_price = trade.price
            bar.high_price = max(bar.high_price, trade.price)
            bar.low_price = min(bar.low_price, trade.price)
            bar.volume += trade.size
            bar.trade_count += 1
            if trade.side == "buy":
                bar.buy_volume += trade.size
            else:
                bar.sell_volume += trade.size

    def get_bars(self, include_current: bool = True) -> list[OrderFlowBar]:
        """Snapshot of completed bars, optionally including the in-progress bar."""
        with self._lock:
            bars = list(self._bars)
            if include_current and self._current is not None:
                bars.append(self._current)
        return bars

    def cvd(self) -> float:
        """Cumulative volume delta across all stored bars."""
        return sum(b.delta for b in self.get_bars())

    def cvd_series(self) -> list[float]:
        """Running CVD at each bar close."""
        cum = 0.0
        out = []
        for b in self.get_bars():
            cum += b.delta
            out.append(cum)
        return out

    def detect_absorption(self, window: int = 5, imbalance_threshold: float = 0.65,
                          price_move_atr_frac: float = 0.3) -> list[dict[str, Any]]:
        """
        Detect absorption: large one-sided delta but minimal price movement.

        Looks at the last `window` bars. Computes typical ATR-style range
        across them and flags bars where imbalance exceeds the threshold yet
        the bar's body is less than `price_move_atr_frac` × typical range.
        """
        bars = self.get_bars()
        if len(bars) < window:
            return []

        recent = bars[-window:]
        ranges = [b.high_price - b.low_price for b in recent if b.high_price > b.low_price]
        if not ranges:
            return []
        avg_range = sum(ranges) / len(ranges)
        threshold = avg_range * price_move_atr_frac

        signals = []
        for bar in recent:
            body = abs(bar.close_price - bar.open_price)
            if bar.imbalance >= imbalance_threshold and body < threshold and bar.volume > 0:
                signals.append({
                    "type": "absorption",
                    "side": bar.dominant_side,
                    "price": bar.close_price,
                    "delta": round(bar.delta, 4),
                    "imbalance": round(bar.imbalance, 3),
                    "body": round(body, 4),
                    "avg_range": round(avg_range, 4),
                    "timestamp": bar.end_time,
                })
        return signals

    def detect_divergence(self, lookback: int = 10) -> dict[str, Any] | None:
        """
        Check for price/CVD divergence across the last `lookback` bars.

        bullish_divergence: price makes LL but CVD makes HL → sellers exhausted
        bearish_divergence: price makes HH but CVD makes LH → buyers exhausted
        """
        bars = self.get_bars()
        if len(bars) < lookback:
            return None

        recent = bars[-lookback:]
        first, last = recent[0], recent[-1]
        cvd_start = sum(b.delta for b in bars[:-lookback])
        cvd_end = cvd_start + sum(b.delta for b in recent)

        price_high = max(b.high_price for b in recent)
        price_low = min(b.low_price for b in recent)

        # Rough structure check.
        if last.close_price > first.close_price and cvd_end < cvd_start:
            return {
                "type": "bearish_divergence",
                "price_change": round(last.close_price - first.close_price, 2),
                "cvd_change": round(cvd_end - cvd_start, 4),
                "message": "Price up but CVD down — buyers losing momentum",
            }
        elif last.close_price < first.close_price and cvd_end > cvd_start:
            return {
                "type": "bullish_divergence",
                "price_change": round(last.close_price - first.close_price, 2),
                "cvd_change": round(cvd_end - cvd_start, 4),
                "message": "Price down but CVD up — sellers losing momentum",
            }
        return None

    def analyze(self) -> dict[str, Any]:
        """Complete order-flow snapshot used by the signal generator."""
        bars = self.get_bars()
        if not bars:
            return {
                "bars": 0, "cvd": 0.0, "delta_last": 0.0,
                "imbalance_last": 0.0, "dominant_side": "neutral",
                "absorption": [], "divergence": None,
            }

        last = bars[-1]
        return {
            "bars": len(bars),
            "cvd": round(self.cvd(), 4),
            "delta_last": round(last.delta, 4),
            "imbalance_last": round(last.imbalance, 3),
            "dominant_side": last.dominant_side,
            "absorption": self.detect_absorption(),
            "divergence": self.detect_divergence(),
        }

    def reset(self) -> None:
        with self._lock:
            self._bars.clear()
            self._current = None


# ---------------------------------------------------------------------------
# Websocket streamer — wraps ccxt.pro or raw websocket feed
# ---------------------------------------------------------------------------

class OrderFlowStreamer:
    """
    Background thread that feeds an OrderFlowTracker from a websocket source.

    The caller provides `trade_fetcher`, a callable returning a list of raw
    trades each call. This keeps the module independent of any specific
    websocket library — you can back it with ccxt.pro, python-binance, or a
    mocked feed for testing.
    """

    def __init__(
        self,
        symbol: str,
        tracker: OrderFlowTracker,
        trade_fetcher: Callable[[str], list[Trade]],
        poll_interval: float = 1.0,
    ) -> None:
        self.symbol = symbol
        self.tracker = tracker
        self.trade_fetcher = trade_fetcher
        self.poll_interval = poll_interval
        self._running = False
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        logger.info("OrderFlowStreamer started for %s", self.symbol)

    def stop(self) -> None:
        self._running = False

    def _loop(self) -> None:
        while self._running:
            try:
                trades = self.trade_fetcher(self.symbol)
                for t in trades:
                    self.tracker.add_trade(t)
            except Exception as exc:
                logger.warning("OrderFlow fetcher error for %s: %s", self.symbol, exc)
            time.sleep(self.poll_interval)


# ---------------------------------------------------------------------------
# Order flow score — used by signal generator for confluence
# ---------------------------------------------------------------------------

def score_order_flow(analysis: dict[str, Any], side: str) -> tuple[int, list[str]]:
    """
    Compute additive score contribution from order flow analysis for a
    proposed trade side. Returns (score_delta, reasons).

    Scoring:
        + Dominant side aligned with trade:       +5
        + Strong imbalance (>0.65) aligned:       +5
        + Absorption matching side:               +7
        + Bullish/bearish divergence aligned:     +10
        - Strong imbalance AGAINST trade:         -8
    """
    score = 0
    reasons = []

    if not analysis or analysis.get("bars", 0) == 0:
        return 0, []

    dominant = analysis.get("dominant_side", "neutral")
    imbalance = analysis.get("imbalance_last", 0)

    if side == "long" and dominant == "buy":
        score += 5
        reasons.append(f"Order flow: buy-dominant (Δ={analysis['delta_last']})")
        if imbalance > 0.65:
            score += 5
            reasons.append(f"Order flow: strong buy imbalance ({imbalance})")
    elif side == "short" and dominant == "sell":
        score += 5
        reasons.append(f"Order flow: sell-dominant (Δ={analysis['delta_last']})")
        if imbalance > 0.65:
            score += 5
            reasons.append(f"Order flow: strong sell imbalance ({imbalance})")
    elif imbalance > 0.65:
        score -= 8
        reasons.append(f"Order flow: strong imbalance AGAINST trade ({dominant})")

    for absorp in analysis.get("absorption", []):
        if (side == "long" and absorp["side"] == "sell") or \
           (side == "short" and absorp["side"] == "buy"):
            # Absorption against the dominant side = reversal setup aligned with us.
            score += 7
            reasons.append(f"Order flow: absorption of {absorp['side']} pressure @ {absorp['price']}")

    divergence = analysis.get("divergence")
    if divergence:
        if divergence["type"] == "bullish_divergence" and side == "long":
            score += 10
            reasons.append("Order flow: bullish CVD divergence")
        elif divergence["type"] == "bearish_divergence" and side == "short":
            score += 10
            reasons.append("Order flow: bearish CVD divergence")

    return score, reasons
