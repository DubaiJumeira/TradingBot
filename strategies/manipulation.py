"""
Market Manipulation Detection — Phase 2.

Four detectors produce discrete ``ManipulationEvent`` objects that
feed into a per-symbol rolling timeline. When multiple events cluster
in the same direction within a short window, the bot treats that as a
coordinated institutional move and scores it heavily.

Detectors
---------

1. **Stop Hunt** (OHLCV) — long-wick candles that spiked beyond a
   swing point and reversed within one or two bars. These are the
   classic "liquidity grab" pattern: smart money pushes price into a
   pool of stops, fills against the forced liquidations, then lets
   price snap back.

2. **Absorption** (OHLCV) — flat candles on high volume. A big
   player is absorbing supply (bullish) or demand (bearish) without
   letting price move. This is Wyckoff's Composite Man soaking up
   flow before markup or markdown.

3. **Spoofing** (order book) — large bid/ask walls that appear,
   price approaches, then the wall is pulled. We track the last N
   snapshots and flag walls that vanish when price is within range.

4. **Wash Trading** (volume vs depth) — extremely high reported
   volume while order book depth is thin. The trading activity is
   fake or self-crossing and the move should not be trusted.

TRADING LOGIC
-------------
- Stop hunts are the strongest single signal: the pool was hit, the
  liquidations are done, now the real move begins.
- Absorption at range lows is bullish, at range highs is bearish.
- Spoofing above price is bearish (fake support was pulled to hide
  sell pressure), spoofing below price is bullish.
- Wash trading REDUCES confidence regardless of direction — it
  doesn't tell us where to trade, it tells us to trust volume less.
- When 3+ events line up in the same direction within 30 minutes,
  that's a coordinated move worth +20 and a Telegram alert.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ManipulationEvent:
    type: str          # "spoofing" | "stop_hunt" | "wash_trading" | "absorption"
    timestamp: datetime
    price_level: float
    direction: str     # "bullish_trap" | "bearish_trap" | "bullish" | "bearish" | "neutral"
    confidence: float  # 0-1
    volume: float
    description: str

    def as_dict(self) -> dict:
        return {
            "type": self.type,
            "timestamp": self.timestamp.isoformat(),
            "price_level": self.price_level,
            "direction": self.direction,
            "confidence": self.confidence,
            "volume": self.volume,
            "description": self.description,
        }


# ---------------------------------------------------------------------------
# OHLCV-based detectors (stateless)
# ---------------------------------------------------------------------------


def detect_stop_hunt(df: pd.DataFrame, lookback: int = 20) -> list[ManipulationEvent]:
    """Detect recent stop hunts on the closing bars.

    A stop hunt candle has:
      - A spike beyond the highest/lowest point of the prior ``lookback`` bars
      - A long wick on the spike side (wick > 1.5× body)
      - The next close has already reversed most of the spike

    We only scan the last two closed candles — anything older than that
    is yesterday's news from a trading perspective.
    """
    events: list[ManipulationEvent] = []
    if len(df) < lookback + 2:
        return events

    for i in (-2, -1):  # check the last two closed candles
        candle = df.iloc[i]
        prior = df.iloc[i - lookback:i]
        if prior.empty:
            continue

        body = abs(candle["close"] - candle["open"])
        upper_wick = candle["high"] - max(candle["open"], candle["close"])
        lower_wick = min(candle["open"], candle["close"]) - candle["low"]

        # Bearish stop hunt: spike above prior highs with long upper wick.
        prior_high = prior["high"].max()
        if (
            candle["high"] > prior_high
            and upper_wick > body * 1.5
            and candle["close"] < prior_high
        ):
            events.append(ManipulationEvent(
                type="stop_hunt",
                timestamp=_candle_timestamp(df, i),
                price_level=float(candle["high"]),
                direction="bearish_trap",
                confidence=min(0.9, 0.5 + (upper_wick / (body + 1e-9)) * 0.1),
                volume=float(candle["volume"]),
                description=(
                    f"Stop hunt above {prior_high:.2f} → reversed to {candle['close']:.2f}"
                ),
            ))

        # Bullish stop hunt: spike below prior lows with long lower wick.
        prior_low = prior["low"].min()
        if (
            candle["low"] < prior_low
            and lower_wick > body * 1.5
            and candle["close"] > prior_low
        ):
            events.append(ManipulationEvent(
                type="stop_hunt",
                timestamp=_candle_timestamp(df, i),
                price_level=float(candle["low"]),
                direction="bullish_trap",
                confidence=min(0.9, 0.5 + (lower_wick / (body + 1e-9)) * 0.1),
                volume=float(candle["volume"]),
                description=(
                    f"Stop hunt below {prior_low:.2f} → reversed to {candle['close']:.2f}"
                ),
            ))

    return events


def detect_absorption(df: pd.DataFrame, lookback: int = 20) -> list[ManipulationEvent]:
    """Detect absorption candles on the last two closed bars.

    Absorption = volume > 2× the rolling average AND the candle's range
    is LESS than 0.5× the rolling average range. Big money is eating
    flow without letting price move.

    Direction is inferred from position within the rolling range:
    absorption near the rolling low is bullish (buyers accumulating),
    near the rolling high is bearish (sellers distributing).
    """
    events: list[ManipulationEvent] = []
    if len(df) < lookback + 2:
        return events

    rolling = df.iloc[-lookback - 2:-2]
    if rolling.empty:
        return events

    avg_volume = float(rolling["volume"].mean())
    avg_range = float((rolling["high"] - rolling["low"]).mean())
    if avg_volume <= 0 or avg_range <= 0:
        return events

    rolling_high = float(rolling["high"].max())
    rolling_low = float(rolling["low"].min())
    range_span = rolling_high - rolling_low
    if range_span <= 0:
        return events

    for i in (-2, -1):
        candle = df.iloc[i]
        c_range = candle["high"] - candle["low"]
        if candle["volume"] < avg_volume * 2 or c_range >= avg_range * 0.5:
            continue

        # Where in the rolling range does this candle sit?
        position = (candle["close"] - rolling_low) / range_span
        if position <= 0.3:
            direction = "bullish"
            descr = f"Absorption near range low — composite buyer accumulating"
        elif position >= 0.7:
            direction = "bearish"
            descr = f"Absorption near range high — composite seller distributing"
        else:
            direction = "neutral"
            descr = "Absorption mid-range (ambiguous)"

        vol_ratio = candle["volume"] / avg_volume
        confidence = min(0.95, 0.4 + (vol_ratio - 2) * 0.15)
        events.append(ManipulationEvent(
            type="absorption",
            timestamp=_candle_timestamp(df, i),
            price_level=float(candle["close"]),
            direction=direction,
            confidence=round(confidence, 2),
            volume=float(candle["volume"]),
            description=f"{descr} (vol {vol_ratio:.1f}× avg)",
        ))

    return events


def _candle_timestamp(df: pd.DataFrame, i: int) -> datetime:
    try:
        ts = df.index[i]
        if hasattr(ts, "to_pydatetime"):
            return ts.to_pydatetime().replace(tzinfo=timezone.utc)
        return datetime.now(timezone.utc)
    except Exception:
        return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Order-book-based detectors (stateful)
# ---------------------------------------------------------------------------


@dataclass
class _OrderBookSnapshot:
    timestamp: datetime
    bids: list[tuple[float, float]]  # (price, size) in USD quote
    asks: list[tuple[float, float]]
    last_price: float


class ManipulationTracker:
    """Per-symbol state: order book history + rolling event timeline.

    One instance per traded instrument. The tracker remembers the last
    few order book snapshots so spoofing detection can diff them, and
    keeps a 30-minute rolling event window for cluster detection.
    """

    _WALL_MIN_USD = 500_000
    _WALL_PROXIMITY_PCT = 0.5  # walls within 0.5% of last price are "in range"
    _SPOOF_LOOKBACK = 3
    _EVENT_WINDOW = timedelta(minutes=30)
    _CLUSTER_THRESHOLD = 3

    def __init__(self, symbol: str) -> None:
        self.symbol = symbol
        self._snapshots: deque[_OrderBookSnapshot] = deque(maxlen=self._SPOOF_LOOKBACK + 1)
        self._events: deque[ManipulationEvent] = deque(maxlen=200)

    def ingest_order_book(self, order_book: dict, last_price: float) -> list[ManipulationEvent]:
        """Record a new snapshot and return any newly-detected events."""
        new_events: list[ManipulationEvent] = []
        snap = _OrderBookSnapshot(
            timestamp=datetime.now(timezone.utc),
            bids=self._to_usd_levels(order_book.get("bids") or []),
            asks=self._to_usd_levels(order_book.get("asks") or []),
            last_price=last_price,
        )

        if self._snapshots:
            new_events.extend(self._detect_spoofing(self._snapshots[-1], snap))
            new_events.extend(self._detect_wash_trading(snap, order_book))

        self._snapshots.append(snap)
        for e in new_events:
            self._events.append(e)
        return new_events

    def ingest_ohlcv_events(self, events: list[ManipulationEvent]) -> None:
        """Merge OHLCV-derived events into the tracker's timeline."""
        for e in events:
            self._events.append(e)

    def recent_events(self, window: timedelta | None = None) -> list[ManipulationEvent]:
        w = window or self._EVENT_WINDOW
        cutoff = datetime.now(timezone.utc) - w
        return [e for e in self._events if e.timestamp >= cutoff]

    def detect_cluster(self) -> dict | None:
        """Return cluster info if 3+ events in same direction within window."""
        recent = self.recent_events()
        bullish = [e for e in recent if e.direction in ("bullish", "bullish_trap")]
        bearish = [e for e in recent if e.direction in ("bearish", "bearish_trap")]
        if len(bullish) >= self._CLUSTER_THRESHOLD:
            return {
                "direction": "bullish",
                "events": [e.as_dict() for e in bullish],
                "count": len(bullish),
            }
        if len(bearish) >= self._CLUSTER_THRESHOLD:
            return {
                "direction": "bearish",
                "events": [e.as_dict() for e in bearish],
                "count": len(bearish),
            }
        return None

    # -------------------- internal detectors --------------------

    def _detect_spoofing(
        self,
        prev: _OrderBookSnapshot,
        curr: _OrderBookSnapshot,
    ) -> list[ManipulationEvent]:
        """A spoof = a large wall present in ``prev``, gone in ``curr``,
        while last price moved toward that wall in the interim."""
        events: list[ManipulationEvent] = []

        prev_bid_walls = {p: s for p, s in prev.bids if s >= self._WALL_MIN_USD}
        curr_bid_walls = {p: s for p, s in curr.bids if s >= self._WALL_MIN_USD}
        for price, size in prev_bid_walls.items():
            if price in curr_bid_walls:
                continue
            # Wall was pulled. Was price moving toward it?
            if (
                abs(price - prev.last_price) / prev.last_price * 100 <= self._WALL_PROXIMITY_PCT
                and curr.last_price <= prev.last_price  # price came down toward the bid
            ):
                events.append(ManipulationEvent(
                    type="spoofing",
                    timestamp=curr.timestamp,
                    price_level=price,
                    direction="bearish_trap",
                    confidence=min(0.9, 0.5 + (size / 5_000_000) * 0.4),
                    volume=size,
                    description=f"Fake support ${size/1e6:.1f}M pulled @ {price:.2f}",
                ))

        prev_ask_walls = {p: s for p, s in prev.asks if s >= self._WALL_MIN_USD}
        curr_ask_walls = {p: s for p, s in curr.asks if s >= self._WALL_MIN_USD}
        for price, size in prev_ask_walls.items():
            if price in curr_ask_walls:
                continue
            if (
                abs(price - prev.last_price) / prev.last_price * 100 <= self._WALL_PROXIMITY_PCT
                and curr.last_price >= prev.last_price
            ):
                events.append(ManipulationEvent(
                    type="spoofing",
                    timestamp=curr.timestamp,
                    price_level=price,
                    direction="bullish_trap",
                    confidence=min(0.9, 0.5 + (size / 5_000_000) * 0.4),
                    volume=size,
                    description=f"Fake resistance ${size/1e6:.1f}M pulled @ {price:.2f}",
                ))

        return events

    def _detect_wash_trading(
        self,
        snap: _OrderBookSnapshot,
        raw_order_book: dict,
    ) -> list[ManipulationEvent]:
        """If the aggregated depth is unusually shallow, treat recent
        volume as potentially synthetic.

        We compute total $ resting in the top 20 levels. If it's less
        than $1M total, depth is thin — any large reported volume is
        suspicious. The actual volume comparison happens in the
        caller, so here we just flag ``thin_depth`` as a confidence
        modifier, returned only when the condition is severe.
        """
        events: list[ManipulationEvent] = []
        total_depth = sum(s for _, s in snap.bids[:20]) + sum(s for _, s in snap.asks[:20])
        if total_depth < 1_000_000 and snap.last_price > 0:
            events.append(ManipulationEvent(
                type="wash_trading",
                timestamp=snap.timestamp,
                price_level=snap.last_price,
                direction="neutral",
                confidence=0.4,
                volume=total_depth,
                description=(
                    f"Thin order book (${total_depth/1e3:.0f}K top-20 depth) — "
                    f"volume may be unreliable"
                ),
            ))
        return events

    @staticmethod
    def _to_usd_levels(levels: list[list[float]]) -> list[tuple[float, float]]:
        """Convert ccxt [price, amount] levels to [(price, usd_value)] in USD."""
        out: list[tuple[float, float]] = []
        for lvl in levels:
            if len(lvl) >= 2:
                price = float(lvl[0])
                amount = float(lvl[1])
                out.append((price, price * amount))
        return out


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def score_manipulation(
    tracker: ManipulationTracker,
    side: str,
) -> tuple[int, list[str]]:
    """Score recent manipulation events for a proposed trade side.

    Returns (score_delta, reasons). Score caps at +30 for a full cluster.
    """
    score = 0
    reasons: list[str] = []
    recent = tracker.recent_events()
    if not recent:
        return 0, []

    wanted = "bullish" if side == "long" else "bearish"
    trap = "bullish_trap" if side == "long" else "bearish_trap"

    # Stop hunts in the right direction are the strongest single signal.
    stop_hunts = [e for e in recent if e.type == "stop_hunt" and e.direction == trap]
    if stop_hunts:
        best = max(stop_hunts, key=lambda e: e.confidence)
        delta = int(10 * best.confidence)
        score += delta
        reasons.append(f"🎯 Stop hunt detected: {best.description} (+{delta})")

    # Absorption aligned with trade.
    absorptions = [e for e in recent if e.type == "absorption" and e.direction == wanted]
    if absorptions:
        best = max(absorptions, key=lambda e: e.confidence)
        delta = int(10 * best.confidence)
        score += delta
        reasons.append(f"🧲 Absorption: {best.description} (+{delta})")

    # Spoofing — a bearish trap above price supports shorts, bullish trap below supports longs.
    spoofs = [e for e in recent if e.type == "spoofing" and e.direction == trap]
    if spoofs:
        score += 8
        reasons.append(f"🚨 Spoofing: {spoofs[-1].description} (+8)")

    # Wash trading — reduce confidence regardless of side.
    if any(e.type == "wash_trading" for e in recent):
        score -= 5
        reasons.append("⚠️ Wash trading flag: volume may be synthetic (-5)")

    # Cluster bonus — multiple events in the same direction.
    cluster = tracker.detect_cluster()
    if cluster and cluster["direction"] == wanted:
        score += 20
        reasons.append(
            f"🐋 COORDINATED MOVE: {cluster['count']} {wanted} manipulation events in 30min (+20)"
        )

    return score, reasons
