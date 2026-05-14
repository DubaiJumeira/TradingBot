"""
Liquidation Heatmap & Level Estimator — Phase 1 (A + B).

Two responsibilities:

1. ``CoinGlassClient`` — thin REST client for CoinGlass's liquidation
   heatmap and aggregated funding/OI endpoints. Gated entirely on the
   ``COINGLASS_API_KEY`` env var. If the key is not set the client is
   never instantiated and the caller automatically falls back to (2).

2. ``estimate_liquidation_levels`` — an exchange-agnostic estimator that
   produces a liquidation cluster map from a symbol's current price and
   its open interest, by simulating common leverage tiers. This is our
   fallback for when CoinGlass is unavailable (rate limited, free-tier
   exhausted, key missing, network blip).

TRADING LOGIC
-------------
Liquidation clusters act as liquidity magnets: market makers hunt
price toward dense clusters because triggering those liquidations
provides the opposing liquidity they need to fill large orders. The
bigger and closer the cluster, the stronger the pull.

For a long at entry_price with leverage L, the liquidation price is
approximately:

    liq = entry * (1 - 1/L + maintenance_margin)

(maintenance margin is ~0.5% on major pairs for Bybit/Binance, small
enough that we fold it into a single 0.005 constant.)

We don't know WHERE longs entered. We assume most leverage is opened
near the current price (within the last few days of trading), so we
treat ``current_price`` as the effective entry for the average
leveraged position at each tier. That's a deliberate approximation —
the goal is to identify WHERE liquidation clusters sit relative to
current price, not to predict individual positions.

Tier volume weights are calibrated to reflect what's commonly seen on
perps exchanges: retail clusters at high leverage, whales at low.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import requests

logger = logging.getLogger(__name__)

# Common leverage tiers on crypto perps, paired with the fraction of
# total OI we assume sits at that tier. Calibrated from publicly
# reported Bybit/Binance leverage distributions.
_LEVERAGE_TIERS: list[tuple[int, float]] = [
    (100, 0.05),   # whale/retail degen tail
    (50,  0.15),
    (25,  0.30),
    (10,  0.35),
    (5,   0.15),
]

_MAINTENANCE_MARGIN = 0.005  # 0.5% — typical for major pairs


@dataclass
class LiquidationCluster:
    """A single leverage-tier liquidation level on one side of price."""
    price: float
    leverage: int
    side: str          # "long" → cluster of longs that get liquidated here
    volume_usd: float  # estimated $ of positions that liquidate at this level
    distance_pct: float  # signed: negative = below current price


def estimate_liquidation_levels(
    current_price: float,
    open_interest_usd: float,
) -> list[LiquidationCluster]:
    """Produce a synthetic liquidation cluster map.

    Parameters
    ----------
    current_price : float
        Latest market price of the instrument.
    open_interest_usd : float
        Total open interest notional in USD. Pass 0 if unknown — the
        function still returns the relative level structure, just with
        zero volumes (callers can still detect asymmetry from the
        levels' existence, although scoring will be a no-op).

    Returns
    -------
    list[LiquidationCluster]
        Two clusters per leverage tier (one long, one short), sorted by
        ``abs(distance_pct)`` ascending — nearest magnets first.
    """
    if current_price <= 0:
        return []

    clusters: list[LiquidationCluster] = []
    for leverage, weight in _LEVERAGE_TIERS:
        tier_volume = max(open_interest_usd * weight, 0.0)
        # Assume ~half of each tier's OI is long, half is short.
        # Real markets skew, but we don't have per-tier direction data
        # from the free endpoints — asymmetry is detected downstream
        # via the funding rate and OI change deltas.
        half = tier_volume / 2

        # Long liquidations: price has to fall to trigger them.
        long_liq = current_price * (1 - 1 / leverage + _MAINTENANCE_MARGIN)
        # Short liquidations: price has to rise.
        short_liq = current_price * (1 + 1 / leverage - _MAINTENANCE_MARGIN)

        clusters.append(LiquidationCluster(
            price=round(long_liq, 4),
            leverage=leverage,
            side="long",
            volume_usd=round(half, 2),
            distance_pct=round((long_liq - current_price) / current_price * 100, 3),
        ))
        clusters.append(LiquidationCluster(
            price=round(short_liq, 4),
            leverage=leverage,
            side="short",
            volume_usd=round(half, 2),
            distance_pct=round((short_liq - current_price) / current_price * 100, 3),
        ))

    clusters.sort(key=lambda c: abs(c.distance_pct))
    return clusters


# ---------------------------------------------------------------------------
# CoinGlass integration (optional, gated on env key)
# ---------------------------------------------------------------------------


class CoinGlassClient:
    """Minimal CoinGlass REST client — only the endpoints we actually use.

    The free tier is heavily rate-limited (~30 req/min). We cache
    responses for 60 seconds in memory and let the caller decide what
    to do on failure (typically: fall back to the estimator).
    """

    BASE_URL = "https://open-api-v3.coinglass.com/api"

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key
        self._headers = {"CG-API-KEY": api_key, "accept": "application/json"}
        self._cache: dict[str, tuple[float, dict]] = {}

    def _get(self, path: str, params: dict | None = None) -> dict | None:
        import time
        key = f"{path}?{sorted((params or {}).items())}"
        now = time.time()
        cached = self._cache.get(key)
        if cached and now - cached[0] < 60:
            return cached[1]
        try:
            resp = requests.get(
                f"{self.BASE_URL}{path}",
                headers=self._headers,
                params=params or {},
                timeout=10,
            )
            if not resp.ok:
                logger.debug("coinglass %s -> %s: %s", path, resp.status_code, resp.text[:120])
                return None
            data = resp.json()
            if data.get("code") not in (0, "0"):
                logger.debug("coinglass %s non-zero code: %s", path, data.get("code"))
                return None
            self._cache[key] = (now, data)
            return data
        except Exception as exc:
            logger.debug("coinglass %s request failed: %s", path, exc)
            return None

    def liquidation_heatmap(self, symbol: str) -> dict | None:
        """Return the raw liquidation heatmap payload for a symbol.

        Symbol format is the CoinGlass shorthand (e.g. ``BTC``, ``ETH``),
        not the ccxt slash form. The caller normalizes.
        """
        cg_symbol = symbol.split("/")[0].replace("USDT", "")
        return self._get(
            "/futures/liquidation/heatmap/model1",
            {"symbol": cg_symbol, "interval": "1h"},
        )


_client: CoinGlassClient | None = None


def get_coinglass_client() -> CoinGlassClient | None:
    """Return a cached CoinGlass client, or None if the key isn't set."""
    global _client
    if _client is not None:
        return _client
    key = os.environ.get("COINGLASS_API_KEY")
    if not key:
        return None
    _client = CoinGlassClient(key)
    return _client


def fetch_liquidation_clusters(
    symbol: str,
    current_price: float,
    open_interest_usd: float,
    exchange=None,
) -> tuple[list[LiquidationCluster], str]:
    """Unified entry point for real liquidity clusters.

    Only real data sources — no synthetic estimation. Priority:

      1. ``coinglass`` — paid REST liquidation heatmap (if key set).
      2. ``observed+orderbook`` — 24h observed liquidations (WS) merged
         with current deep order book walls (resting limit orders).
      3. ``orderbook`` — deep order book only, if observed buffer is thin.
      4. ``observed`` — observed liquidations only, if exchange unavailable.

    ``open_interest_usd`` is accepted for backwards compatibility but no
    longer used — the synthetic estimator is gone.

    Returns
    -------
    (clusters, source)
    """
    client = get_coinglass_client()
    if client is not None:
        payload = client.liquidation_heatmap(symbol)
        if payload and "data" in payload:
            parsed = _parse_coinglass_heatmap(payload["data"], current_price)
            if parsed:
                return parsed, "coinglass"

    observed: list[LiquidationCluster] = []
    try:
        from strategies.liquidation_stream import (
            ensure_stream_started,
            get_real_liquidation_clusters,
        )
        ensure_stream_started()
        observed = get_real_liquidation_clusters(symbol, current_price)
    except Exception as exc:
        logger.debug("observed liquidation path failed for %s: %s", symbol, exc)

    orderbook: list[LiquidationCluster] = []
    if exchange is not None:
        try:
            from strategies.orderbook_liquidity import (
                fetch_orderbook_walls,
                walls_to_cluster_format,
            )
            walls = fetch_orderbook_walls(exchange, symbol, current_price)
            orderbook = walls_to_cluster_format(walls)
        except Exception as exc:
            logger.debug("orderbook fetch failed for %s: %s", symbol, exc)

    if observed and orderbook:
        merged = list(observed) + list(orderbook)
        merged.sort(key=lambda c: abs(c.distance_pct))
        return merged, "observed+orderbook"
    if orderbook:
        return orderbook, "orderbook"
    if observed:
        return observed, "observed"
    return [], "unavailable"


def _parse_coinglass_heatmap(data: dict, current_price: float) -> list[LiquidationCluster]:
    """Convert a CoinGlass heatmap payload into our LiquidationCluster format.

    CoinGlass returns a 2D grid keyed by (time, price) with liquidation
    $ volumes. We collapse across time to get per-price-level totals,
    then emit one cluster per level, tagged long/short based on whether
    the level is above or below current price.
    """
    try:
        prices = data.get("y") or []
        grid = data.get("liq") or []
        if not prices or not grid:
            return []

        # Sum liquidation $ per price level across all time buckets.
        totals: dict[float, float] = {}
        for cell in grid:
            if len(cell) < 3:
                continue
            _, y_idx, vol = cell[0], cell[1], cell[2]
            if 0 <= y_idx < len(prices):
                level = float(prices[y_idx])
                totals[level] = totals.get(level, 0.0) + float(vol or 0)

        clusters: list[LiquidationCluster] = []
        for level, vol in totals.items():
            if vol <= 0:
                continue
            side = "long" if level < current_price else "short"
            distance_pct = (level - current_price) / current_price * 100
            clusters.append(LiquidationCluster(
                price=round(level, 4),
                leverage=0,  # unknown from heatmap; kept for schema parity
                side=side,
                volume_usd=round(vol, 2),
                distance_pct=round(distance_pct, 3),
            ))

        clusters.sort(key=lambda c: abs(c.distance_pct))
        return clusters
    except Exception as exc:
        logger.warning("coinglass heatmap parse failed: %s", exc)
        return []
