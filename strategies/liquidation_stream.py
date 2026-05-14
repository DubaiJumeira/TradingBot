"""
Real-time liquidation stream ingestion — Phase 1 (precise heatmap source).

The synthetic estimator in ``liquidation.py`` produces a structurally
correct but *modeled* heatmap — it assumes leverage tiers and uniform
direction distribution. It cannot tell you that an actual $40M wall
of longs got wiped at $83,800 two hours ago, or that shorts above
$3,420 are stacking.

This module fills that gap by subscribing to the free public
liquidation streams exposed by the two largest perps venues:

    Binance Futures: wss://fstream.binance.com/ws/!forceOrder@arr
    Bybit v5:        wss://stream.bybit.com/v5/public/linear
                     topic: allLiquidation.{SYMBOL}

Every forced order is an actual liquidation that happened. We keep a
24h rolling buffer per symbol and bucket the events by price. The
output conforms to the same ``LiquidationCluster`` shape as the
estimator, so the downstream magnet detector doesn't care where the
data came from.

LIFECYCLE
---------
Call ``ensure_stream_started()`` once at boot. A single daemon thread
hosts an asyncio loop that runs two auto-reconnecting tasks, one per
exchange. Reconnects back off with a short fixed delay — the Binance
stream is reliable enough that exponential backoff buys nothing.

DATA PRIORITY
-------------
``fetch_liquidation_clusters`` in liquidation.py consults this module
first. If the buffer has enough observed events (default 20) for the
symbol, those are returned as the authoritative heatmap. Otherwise the
caller falls back to CoinGlass REST or, finally, the synthetic
estimator — so early minutes after a restart are never a blind spot.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass

from strategies.liquidation import LiquidationCluster

logger = logging.getLogger(__name__)


_BINANCE_WS = "wss://fstream.binance.com/ws/!forceOrder@arr"
_BYBIT_WS = "wss://stream.bybit.com/v5/public/linear"

# 24h rolling window — matches the typical CoinGlass lookback so magnet
# strength between sources is comparable.
_WINDOW_SECONDS = 24 * 3600

# Hard ceiling on events per symbol. In practice BTC sees ~1-3k
# liquidations in a calm day, 10-20k during a flush. We cap at 20k so
# a genuinely wild session doesn't starve the buffer of recency but
# the process never balloons past ~5 MB of resident liquidation data.
_MAX_EVENTS_PER_SYMBOL = 20_000

# Bybit requires explicit per-symbol subscriptions (no all-symbols
# topic). Keep this aligned with config.INSTRUMENTS crypto entries.
_BYBIT_SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT")


@dataclass(slots=True)
class _LiqEvent:
    ts: float        # unix seconds
    symbol: str      # normalised, e.g. BTCUSDT
    side: str        # "long" (long was liquidated) / "short"
    price: float
    qty_usd: float
    exchange: str    # "binance" / "bybit"


def _normalise_symbol(symbol: str) -> str:
    """Fold any of the bot's symbol dialects into the exchange form.

    ``BTC/USDT`` / ``BTC/USDT:USDT`` / ``BTCUSDT`` all collapse to
    ``BTCUSDT``. Lowercase input tolerated.
    """
    s = symbol.upper().replace(":USDT", "").replace("/", "")
    return s


class LiquidationStreamManager:
    """Thread-safe store of observed liquidation events + WS runner."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._events: dict[str, deque[_LiqEvent]] = defaultdict(
            lambda: deque(maxlen=_MAX_EVENTS_PER_SYMBOL)
        )
        self._started = False
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stats: dict[str, float] = {
            "binance_connects": 0,
            "bybit_connects": 0,
            "binance_events": 0,
            "bybit_events": 0,
            "binance_last_event_ts": 0.0,
            "bybit_last_event_ts": 0.0,
        }

    # --- public API ---------------------------------------------------

    def start(self) -> None:
        """Start the WS thread. Idempotent, non-blocking."""
        with self._lock:
            if self._started:
                return
            self._started = True
        self._thread = threading.Thread(
            target=self._run_loop, name="liq-stream", daemon=True
        )
        self._thread.start()
        logger.info("liquidation stream manager: background thread started")

    def add_event(self, ev: _LiqEvent) -> None:
        with self._lock:
            self._events[ev.symbol].append(ev)
            self._stats[f"{ev.exchange}_events"] = (
                self._stats.get(f"{ev.exchange}_events", 0) + 1
            )
            self._stats[f"{ev.exchange}_last_event_ts"] = ev.ts

    def get_events(
        self, symbol: str, window_seconds: float = _WINDOW_SECONDS
    ) -> list[_LiqEvent]:
        sym = _normalise_symbol(symbol)
        cutoff = time.time() - window_seconds
        with self._lock:
            bucket = self._events.get(sym)
            if not bucket:
                return []
            return [e for e in bucket if e.ts >= cutoff]

    def stats(self) -> dict:
        with self._lock:
            counts = {s: len(b) for s, b in self._events.items()}
            return {"counters": dict(self._stats), "buffered": counts}

    # --- event loop ---------------------------------------------------

    def _run_loop(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(
                asyncio.gather(self._binance_loop(), self._bybit_loop())
            )
        except Exception:
            logger.exception("liquidation stream loop crashed")

    async def _binance_loop(self) -> None:
        import websockets

        while True:
            try:
                async with websockets.connect(
                    _BINANCE_WS,
                    ping_interval=20,
                    ping_timeout=10,
                    max_size=2**20,
                ) as ws:
                    with self._lock:
                        self._stats["binance_connects"] += 1
                    logger.info("liquidation stream: Binance connected")
                    async for raw in ws:
                        self._handle_binance(raw)
            except Exception as exc:
                logger.warning(
                    "Binance liq ws disconnected (%s); reconnecting in 5s",
                    exc,
                )
                await asyncio.sleep(5)

    def _handle_binance(self, raw: str | bytes) -> None:
        try:
            msg = json.loads(raw)
        except Exception:
            return
        o = msg.get("o") if isinstance(msg, dict) else None
        if not o:
            # combined-stream wrapping (just in case)
            data = msg.get("data") if isinstance(msg, dict) else None
            o = data.get("o") if isinstance(data, dict) else None
        if not o:
            return
        try:
            symbol = _normalise_symbol(o["s"])
            # Binance forceOrder "S": SELL = a LONG was liquidated,
            # BUY = a SHORT was liquidated.
            side = "long" if o["S"] == "SELL" else "short"
            price = float(o.get("ap") or o["p"])
            qty = float(o["q"])
            qty_usd = price * qty
            ts = float(o.get("T", time.time() * 1000)) / 1000.0
        except (KeyError, TypeError, ValueError) as exc:
            logger.debug("binance liq parse skipped: %s", exc)
            return
        self.add_event(_LiqEvent(ts, symbol, side, price, qty_usd, "binance"))

    async def _bybit_loop(self) -> None:
        import websockets

        subs = [f"allLiquidation.{s}" for s in _BYBIT_SYMBOLS]
        while True:
            try:
                async with websockets.connect(
                    _BYBIT_WS,
                    ping_interval=20,
                    ping_timeout=10,
                ) as ws:
                    await ws.send(json.dumps({"op": "subscribe", "args": subs}))
                    with self._lock:
                        self._stats["bybit_connects"] += 1
                    logger.info(
                        "liquidation stream: Bybit connected (%d symbols)",
                        len(subs),
                    )
                    async for raw in ws:
                        self._handle_bybit(raw)
            except Exception as exc:
                logger.warning(
                    "Bybit liq ws disconnected (%s); reconnecting in 5s",
                    exc,
                )
                await asyncio.sleep(5)

    def _handle_bybit(self, raw: str | bytes) -> None:
        try:
            msg = json.loads(raw)
        except Exception:
            return
        if not isinstance(msg, dict):
            return
        topic = msg.get("topic", "")
        if not topic.startswith("allLiquidation."):
            return
        for d in msg.get("data") or []:
            try:
                symbol = _normalise_symbol(d.get("s", topic.split(".")[-1]))
                # Bybit allLiquidation "S": "Sell" = long liquidated,
                # "Buy" = short liquidated.
                s_field = d.get("S") or d.get("side") or ""
                side = "long" if s_field.lower() == "sell" else "short"
                price = float(d["p"])
                qty = float(d["v"])
                qty_usd = price * qty
                ts = float(d.get("T", time.time() * 1000)) / 1000.0
            except (KeyError, TypeError, ValueError) as exc:
                logger.debug("bybit liq parse skipped: %s", exc)
                continue
            self.add_event(_LiqEvent(ts, symbol, side, price, qty_usd, "bybit"))


_manager: LiquidationStreamManager | None = None


def get_manager() -> LiquidationStreamManager:
    global _manager
    if _manager is None:
        _manager = LiquidationStreamManager()
    return _manager


def ensure_stream_started() -> None:
    """Start the background stream if it hasn't been already."""
    get_manager().start()


def get_real_liquidation_clusters(
    symbol: str,
    current_price: float,
    *,
    window_seconds: float = _WINDOW_SECONDS,
    min_events: int = 20,
    bucket_width_pct: float = 0.1,
) -> list[LiquidationCluster]:
    """Return observed-liquidation clusters or [] if buffer is thin.

    Parameters
    ----------
    symbol : str
        Any of the bot's symbol dialects; normalised internally.
    current_price : float
        Needed to compute ``distance_pct`` and pick the bucket width.
    window_seconds : float
        Lookback window over the rolling buffer. Default 24h.
    min_events : int
        Minimum observed events before we consider the buffer reliable.
        Below this we return [] and let the caller fall back.
    bucket_width_pct : float
        Price-bucket width as a percentage of current price. 0.1% is
        tight enough to resolve individual walls and wide enough that a
        typical day's worth of liquidations produces ~30-80 clusters.
    """
    if current_price <= 0:
        return []

    events = get_manager().get_events(symbol, window_seconds=window_seconds)
    if len(events) < min_events:
        return []

    bucket_width = max(current_price * bucket_width_pct / 100.0, 1e-8)
    buckets: dict[tuple[int, str], dict] = {}
    for e in events:
        bidx = int(e.price / bucket_width)
        key = (bidx, e.side)
        entry = buckets.setdefault(key, {"price_wsum": 0.0, "vol": 0.0})
        entry["price_wsum"] += e.price * e.qty_usd
        entry["vol"] += e.qty_usd

    clusters: list[LiquidationCluster] = []
    for (_bidx, side), b in buckets.items():
        if b["vol"] <= 0:
            continue
        avg_price = b["price_wsum"] / b["vol"]
        clusters.append(
            LiquidationCluster(
                price=round(avg_price, 4),
                leverage=0,  # unknown from observed events
                side=side,
                volume_usd=round(b["vol"], 2),
                distance_pct=round(
                    (avg_price - current_price) / current_price * 100, 3
                ),
            )
        )
    clusters.sort(key=lambda c: abs(c.distance_pct))
    return clusters


def get_stats() -> dict:
    """Diagnostic counters — intended for /liquidation-stream telemetry."""
    return get_manager().stats()


def dump_recent_events(
    path: str = "/root/trading-bot/data/observed_liquidations.json",
    window_seconds: float = _WINDOW_SECONDS,
) -> int:
    """Dump recent observed liquidations to JSON for cross-process sharing.

    The dashboard runs in its own process and can't see the bot's
    in-memory buffer, so the bot periodically writes a snapshot that the
    dashboard reads. Returns the number of events written.
    """
    import os
    mgr = get_manager()
    cutoff = time.time() - window_seconds
    all_events: list[dict] = []
    with mgr._lock:
        for sym, bucket in mgr._events.items():
            for e in bucket:
                if e.ts < cutoff:
                    continue
                all_events.append({
                    "ts": e.ts,
                    "symbol": sym,
                    "side": e.side,
                    "price": e.price,
                    "qty_usd": e.qty_usd,
                    "exchange": e.exchange,
                })
    tmp = path + ".tmp"
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(tmp, "w") as f:
            json.dump({"generated_at": time.time(), "events": all_events}, f)
        os.replace(tmp, path)
    except Exception as exc:
        logger.debug("liquidation dump failed: %s", exc)
        return 0
    return len(all_events)
