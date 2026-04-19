"""
Liquidity Magnet Detection — Phase 1C.

Turns a raw list of ``LiquidationCluster`` objects into a smaller set
of actionable magnets. Clusters that sit within a small price band of
each other are merged — the market doesn't care whether a cluster is
at $84,100 or $84,105, they're functionally the same liquidity pool.

Magnetic strength is a 0-1 score combining:

  - absolute volume (heavier cluster = stronger pull)
  - proximity (closer to price = more likely to be hunted first)

TRADING LOGIC
-------------
Price is drawn to liquidity. The bot uses magnets two ways:

  1. Scoring: a trade aligned with a dense nearby magnet gets bonus
     confidence. Trading TOWARD liquidity is higher-probability than
     trading away from it.

  2. Take-profit targeting: if a dense magnet sits between the normal
     TP (from ICT structure) and entry, the magnet becomes TP1 — price
     tends to accelerate through liquidity zones as liquidations
     cascade, so partial-exiting there captures the cleanest move.

We also compute an *asymmetry ratio* — how much more liquidity sits on
one side vs the other. Strong asymmetry (ratio > 2) is a directional
tell: the dense side gets hunted first.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from strategies.liquidation import LiquidationCluster

logger = logging.getLogger(__name__)

# Clusters within this percentage of each other are merged into one magnet.
_MERGE_BAND_PCT = 0.3
# Distance at which proximity contribution is halved (in percent).
_PROXIMITY_HALF_LIFE = 2.0


@dataclass
class LiquidityMagnet:
    price_level: float
    direction: str              # "above" (short liqs) or "below" (long liqs)
    estimated_volume_usd: float
    distance_pct: float         # unsigned: distance from current price
    magnetic_strength: float    # 0-1
    leverage_tiers: dict[int, float] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "price_level": self.price_level,
            "direction": self.direction,
            "estimated_volume_usd": self.estimated_volume_usd,
            "distance_pct": self.distance_pct,
            "magnetic_strength": self.magnetic_strength,
            "leverage_tiers": self.leverage_tiers,
        }


def detect_magnets(
    clusters: list[LiquidationCluster],
    current_price: float,
) -> list[LiquidityMagnet]:
    """Merge nearby clusters into magnets and score them.

    Returns magnets sorted by magnetic_strength descending.
    """
    if not clusters or current_price <= 0:
        return []

    # Split clusters by side then merge adjacent ones within the band.
    below: list[LiquidationCluster] = [c for c in clusters if c.price < current_price]
    above: list[LiquidationCluster] = [c for c in clusters if c.price > current_price]
    below.sort(key=lambda c: c.price, reverse=True)  # nearest first
    above.sort(key=lambda c: c.price)

    magnets: list[LiquidityMagnet] = []
    magnets.extend(_merge_side(below, current_price, "below"))
    magnets.extend(_merge_side(above, current_price, "above"))

    # Compute magnetic strength across the combined set so it's a
    # relative score — the strongest magnet always scores 1.0.
    if not magnets:
        return []

    max_volume = max((m.estimated_volume_usd for m in magnets), default=0.0)
    for m in magnets:
        volume_component = (m.estimated_volume_usd / max_volume) if max_volume > 0 else 0.0
        proximity_component = _PROXIMITY_HALF_LIFE / (_PROXIMITY_HALF_LIFE + m.distance_pct)
        m.magnetic_strength = round(0.6 * volume_component + 0.4 * proximity_component, 3)

    magnets.sort(key=lambda m: m.magnetic_strength, reverse=True)
    return magnets


def _merge_side(
    clusters: list[LiquidationCluster],
    current_price: float,
    direction: str,
) -> list[LiquidityMagnet]:
    if not clusters:
        return []

    band = _MERGE_BAND_PCT / 100
    merged: list[LiquidityMagnet] = []
    # Walk clusters in nearest-to-farthest order, bundling any that sit
    # within the merge band of the currently-open bundle.
    bundle_prices: list[float] = []
    bundle_volume = 0.0
    bundle_tiers: dict[int, float] = {}

    def flush() -> None:
        if not bundle_prices:
            return
        vol_weighted_price = (
            sum(p * bundle_tiers.get(l, 0) for p, l in zip(bundle_prices, bundle_tiers))
            or sum(bundle_prices) / len(bundle_prices)
        )
        level = sum(bundle_prices) / len(bundle_prices)
        distance = abs(level - current_price) / current_price * 100
        merged.append(LiquidityMagnet(
            price_level=round(level, 4),
            direction=direction,
            estimated_volume_usd=round(bundle_volume, 2),
            distance_pct=round(distance, 3),
            magnetic_strength=0.0,  # filled in by the caller
            leverage_tiers=dict(bundle_tiers),
        ))
        _ = vol_weighted_price  # reserved for future volume-weighted centroid

    for c in clusters:
        if bundle_prices and abs(c.price - bundle_prices[0]) / bundle_prices[0] > band:
            flush()
            bundle_prices = []
            bundle_volume = 0.0
            bundle_tiers = {}
        bundle_prices.append(c.price)
        bundle_volume += c.volume_usd
        if c.leverage:
            bundle_tiers[c.leverage] = bundle_tiers.get(c.leverage, 0.0) + c.volume_usd

    flush()
    return merged


def compute_asymmetry(magnets: list[LiquidityMagnet]) -> dict:
    """Compute the volume ratio between the two sides.

    Returns a dict with:

        total_above : total $ of liquidations above price
        total_below : total $ of liquidations below price
        ratio       : max(above, below) / max(min, 1) — always >= 1
        dominant    : "above" | "below" | "balanced"
    """
    total_above = sum(m.estimated_volume_usd for m in magnets if m.direction == "above")
    total_below = sum(m.estimated_volume_usd for m in magnets if m.direction == "below")
    if total_above == 0 and total_below == 0:
        return {"total_above": 0.0, "total_below": 0.0, "ratio": 1.0, "dominant": "balanced"}

    dominant = "above" if total_above > total_below else "below"
    larger = max(total_above, total_below)
    smaller = max(min(total_above, total_below), 1.0)
    ratio = round(larger / smaller, 2)
    if ratio < 1.5:
        dominant = "balanced"
    return {
        "total_above": round(total_above, 2),
        "total_below": round(total_below, 2),
        "ratio": ratio,
        "dominant": dominant,
    }


def score_liquidation(
    magnets: list[LiquidityMagnet],
    asymmetry: dict,
    side: str,
) -> tuple[int, list[str], float | None]:
    """Score a trade based on liquidation magnets.

    Returns
    -------
    (score_delta, reasons, tp_override)

    tp_override is the nearest aligned magnet price if one exists
    within a reasonable distance — callers may use it as a TP target.

    Scoring (max +25):

        +15  nearest aligned magnet is within 3% (trading toward liquidity)
        +5   nearest aligned magnet has magnetic_strength > 0.8
        +5   asymmetry ratio > 2 AND dominant side aligns with trade
    """
    if not magnets:
        return 0, [], None

    # "Aligned" means the magnet is in the direction the trade is
    # expecting price to move: longs want magnets ABOVE (short
    # liquidations get hunted upward), shorts want magnets BELOW.
    wanted_direction = "above" if side == "long" else "below"
    aligned = [m for m in magnets if m.direction == wanted_direction]
    if not aligned:
        return 0, ["No aligned liquidation magnet — trading away from liquidity"], None

    nearest = min(aligned, key=lambda m: m.distance_pct)
    score = 0
    reasons: list[str] = []
    tp_override: float | None = None

    if nearest.distance_pct <= 3.0:
        score += 15
        reasons.append(
            f"🔥 Trading toward liquidity: ${nearest.estimated_volume_usd/1e6:.1f}M "
            f"cluster @ {nearest.price_level:.2f} ({nearest.distance_pct:.2f}% away)"
        )
        tp_override = nearest.price_level
    if nearest.magnetic_strength > 0.8:
        score += 5
        reasons.append(
            f"Magnet strength {nearest.magnetic_strength:.2f} — very dense cluster"
        )

    if asymmetry.get("ratio", 1.0) > 2.0 and asymmetry.get("dominant") == wanted_direction:
        score += 5
        reasons.append(
            f"Liquidation asymmetry {asymmetry['ratio']}× "
            f"toward {wanted_direction} — one-sided market"
        )

    return score, reasons, tp_override
