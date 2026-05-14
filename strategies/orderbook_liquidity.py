"""
Real order book liquidity heatmap.

Pulls the full depth order book from the exchange (Bybit supports up
to 500 levels on linear perps) and returns real resting limit orders
bucketed into price bins. Unlike the synthetic leverage-tier estimator,
every dollar shown here is an actual live bid or ask — real liquidity
a trader can see and interact with.

Combined with observed liquidation events (from the force-order WS
stream) this gives a full real-data heatmap:

  - orderbook walls  = where resting limit liquidity is stacked now
  - liquidation buffer = where leverage actually got wiped in the last 24h
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class OrderBookWall:
    price: float
    side: str           # "bid" (support) or "ask" (resistance)
    volume_usd: float   # notional $ at that price level (bucketed)
    distance_pct: float # signed: negative = below current price


_cache: dict[str, tuple[float, list[OrderBookWall]]] = {}
_CACHE_TTL = 15.0  # seconds — orderbook changes fast but 15s is cheap


def fetch_orderbook_walls(
    exchange,
    symbol: str,
    current_price: float,
    *,
    depth: int = 500,
    bucket_pct: float = 0.1,
    top_n_per_side: int = 40,
) -> list[OrderBookWall]:
    """Fetch real resting orderbook walls bucketed by price.

    Parameters
    ----------
    exchange : ExchangeHandler
        Must expose ``fetch_order_book(symbol, limit=N)``.
    symbol : str
        Exchange symbol (BTCUSDT form).
    current_price : float
        Used to compute ``distance_pct`` and bucket width.
    depth : int
        Number of order book levels to request. Bybit linear max = 500.
    bucket_pct : float
        Bucket width as % of current price. 0.1% groups very tight
        levels; 0.2% gives fewer, thicker bars.
    top_n_per_side : int
        Cap on walls returned per side (bids vs asks). Keeps the payload
        small without dropping the big walls, since we sort by volume.

    Returns
    -------
    list[OrderBookWall]
        Sorted by distance_pct (nearest first). Empty list on failure.
    """
    if current_price <= 0:
        return []

    now = time.time()
    cached = _cache.get(symbol)
    if cached and now - cached[0] < _CACHE_TTL:
        return cached[1]

    try:
        book = exchange.fetch_order_book(symbol, limit=depth)
    except Exception as exc:
        logger.debug("orderbook fetch failed for %s: %s", symbol, exc)
        return []

    bids = book.get("bids") or []
    asks = book.get("asks") or []
    if not bids or not asks:
        return []

    bucket_width = max(current_price * bucket_pct / 100.0, 1e-8)

    def _bucket(levels: list, side: str) -> list[OrderBookWall]:
        agg: dict[int, dict] = {}
        for entry in levels:
            if len(entry) < 2:
                continue
            try:
                price = float(entry[0])
                size = float(entry[1])
            except (TypeError, ValueError):
                continue
            if price <= 0 or size <= 0:
                continue
            bidx = int(price / bucket_width)
            a = agg.setdefault(bidx, {"pw": 0.0, "vol": 0.0})
            notional = price * size
            a["pw"] += price * notional
            a["vol"] += notional

        out: list[OrderBookWall] = []
        for a in agg.values():
            if a["vol"] <= 0:
                continue
            avg_price = a["pw"] / a["vol"]
            out.append(OrderBookWall(
                price=round(avg_price, 4),
                side=side,
                volume_usd=round(a["vol"], 2),
                distance_pct=round(
                    (avg_price - current_price) / current_price * 100, 3
                ),
            ))
        out.sort(key=lambda w: w.volume_usd, reverse=True)
        return out[:top_n_per_side]

    walls = _bucket(bids, "bid") + _bucket(asks, "ask")
    walls.sort(key=lambda w: abs(w.distance_pct))

    _cache[symbol] = (now, walls)
    return walls


def walls_to_cluster_format(walls: list[OrderBookWall]) -> list:
    """Convert OrderBookWall objects into LiquidationCluster-shaped data.

    The downstream magnet detector expects the cluster shape, so this
    lets us feed real orderbook liquidity into the same pipeline.
    Resting BIDS behave like "long liquidation" magnets (price may drop
    into them to harvest fills), ASKS like "short liquidation" magnets.
    """
    from strategies.liquidation import LiquidationCluster

    out: list[LiquidationCluster] = []
    for w in walls:
        out.append(LiquidationCluster(
            price=w.price,
            leverage=0,
            side="long" if w.side == "bid" else "short",
            volume_usd=w.volume_usd,
            distance_pct=w.distance_pct,
        ))
    return out
