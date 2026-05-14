"""
Volume Profile — real wide-scale liquidity footprint from historical candles.

The order book is dense but tight (~±0.2% from price). Observed
liquidation events are real but sparse until the buffer fills over
days. To give a wide-range view of where real liquidity *has actually
traded*, we build a volume profile: for each price bucket, sum the
volume that traded through it over the lookback window.

Every bucket is backed by real executed trades — no synthesis. The
profile peaks sit at price zones that have absorbed the most flow, and
those are exactly the zones price magnets toward on big moves (high
volume nodes in Volume Profile / Market Profile theory).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class VolumeNode:
    price: float           # bucket midpoint
    volume_usd: float      # USD notional traded inside the bucket
    distance_pct: float    # signed: negative = below current price


_cache: dict[str, tuple[float, list[VolumeNode]]] = {}
_CACHE_TTL = 600.0  # 10 minutes — candles don't change that fast


def fetch_volume_profile(
    exchange,
    symbol: str,
    current_price: float,
    *,
    timeframe: str = "1h",
    lookback_candles: int = 500,
    num_buckets: int = 80,
) -> list[VolumeNode]:
    """Build a volume profile from recent OHLCV candles.

    Approximates the $ volume traded at each price by using the candle's
    typical price (H+L+C)/3 times its volume, then bins into buckets.
    For 1h candles × 500 bars that's ~3 weeks of BTC trading — enough
    to span meaningful price swings.

    Parameters
    ----------
    exchange : ExchangeHandler or ccxt instance
        Must expose ``fetch_ohlcv(symbol, timeframe, limit=N)``.
    symbol : str
    current_price : float
    timeframe : str
        OHLCV granularity. Default "1h".
    lookback_candles : int
        How many candles back. 500 × 1h ≈ 20 days.
    num_buckets : int
        Number of price buckets. More = finer resolution.

    Returns
    -------
    list[VolumeNode]
        Sorted by price ascending. Includes ALL buckets, even if 0 volume,
        so the caller can render a continuous profile.
    """
    if current_price <= 0:
        return []

    now = time.time()
    cached = _cache.get(symbol)
    if cached and now - cached[0] < _CACHE_TTL:
        return cached[1]

    try:
        candles = exchange.fetch_ohlcv(symbol, timeframe, limit=lookback_candles)
    except Exception as exc:
        logger.debug("volume profile candle fetch failed for %s: %s", symbol, exc)
        return []

    if not candles:
        return []

    prices_hi = [float(c[2]) for c in candles if c and len(c) >= 5]
    prices_lo = [float(c[3]) for c in candles if c and len(c) >= 5]
    if not prices_hi or not prices_lo:
        return []

    lo = min(prices_lo)
    hi = max(prices_hi)
    if hi <= lo:
        return []

    bucket_width = (hi - lo) / num_buckets
    if bucket_width <= 0:
        return []

    buckets = [0.0] * num_buckets
    for c in candles:
        if not c or len(c) < 6:
            continue
        try:
            h, l, cl, v = float(c[2]), float(c[3]), float(c[4]), float(c[5])
        except (TypeError, ValueError):
            continue
        typical = (h + l + cl) / 3
        notional = typical * v
        bidx = int((typical - lo) / bucket_width)
        bidx = max(0, min(num_buckets - 1, bidx))
        buckets[bidx] += notional

    nodes: list[VolumeNode] = []
    for i, vol in enumerate(buckets):
        price = lo + (i + 0.5) * bucket_width
        nodes.append(VolumeNode(
            price=round(price, 4),
            volume_usd=round(vol, 2),
            distance_pct=round((price - current_price) / current_price * 100, 3),
        ))

    _cache[symbol] = (now, nodes)
    return nodes
