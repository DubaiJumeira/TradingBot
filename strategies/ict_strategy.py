"""
ICT Strategy Module
Detects: Fair Value Gaps (FVG), Order Blocks, Liquidity Sweeps,
         Break of Structure (BOS), Change of Character (ChoCH)

Phase 3 additions:
    - Optimal Trade Entry (OTE): 62-79% Fibonacci retracement into OB/FVG
    - Displacement detection: OBs/FVGs only from displacement candles (body > 2x ATR)
    - Breaker Blocks: failed OBs flip polarity (support → resistance)
    - Premium/Discount zones: longs only in discount, shorts only in premium
    - Inducement detection: minor sweep before major liquidity grab
    - Liquidity Voids: large body candles with no wicks → price will return
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import logging
from typing import Any

logger = logging.getLogger(__name__)


def detect_swing_points(df: pd.DataFrame, lookback: int = 5):
    """Identify swing highs and swing lows."""
    highs = df["high"].values
    lows = df["low"].values
    swing_highs = []
    swing_lows = []

    for i in range(lookback, len(df) - lookback):
        # Swing high: highest high in window
        if highs[i] == max(highs[i - lookback:i + lookback + 1]):
            swing_highs.append({"index": i, "price": highs[i], "time": df.index[i]})
        # Swing low: lowest low in window
        if lows[i] == min(lows[i - lookback:i + lookback + 1]):
            swing_lows.append({"index": i, "price": lows[i], "time": df.index[i]})

    return swing_highs, swing_lows


def detect_market_structure(swing_highs: list, swing_lows: list):
    """
    Determine trend via higher highs/higher lows (bullish)
    or lower highs/lower lows (bearish).
    Returns: 'bullish', 'bearish', or 'ranging'
    """
    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return "ranging"

    hh = swing_highs[-1]["price"] > swing_highs[-2]["price"]
    hl = swing_lows[-1]["price"] > swing_lows[-2]["price"]
    lh = swing_highs[-1]["price"] < swing_highs[-2]["price"]
    ll = swing_lows[-1]["price"] < swing_lows[-2]["price"]

    if hh and hl:
        return "bullish"
    elif lh and ll:
        return "bearish"
    return "ranging"


def detect_bos_choch(swing_highs: list, swing_lows: list, current_price: float):
    """
    Break of Structure (BOS): price breaks a swing point in trend direction.
    Change of Character (ChoCH): price breaks against trend direction.
    """
    signals = []
    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return signals

    structure = detect_market_structure(swing_highs, swing_lows)

    last_high = swing_highs[-1]["price"]
    last_low = swing_lows[-1]["price"]
    prev_high = swing_highs[-2]["price"]
    prev_low = swing_lows[-2]["price"]

    if structure == "bullish":
        # BOS: price breaks above last swing high
        if current_price > last_high:
            signals.append({"type": "BOS", "direction": "bullish", "level": last_high})
        # ChoCH: price breaks below last swing low (trend reversal signal)
        if current_price < last_low:
            signals.append({"type": "ChoCH", "direction": "bearish", "level": last_low})

    elif structure == "bearish":
        if current_price < last_low:
            signals.append({"type": "BOS", "direction": "bearish", "level": last_low})
        if current_price > last_high:
            signals.append({"type": "ChoCH", "direction": "bullish", "level": last_high})

    return signals


def detect_fair_value_gaps(df: pd.DataFrame, min_gap_pct: float = 0.1):
    """
    FVG: 3-candle pattern where candle 1 high < candle 3 low (bullish)
    or candle 1 low > candle 3 high (bearish).
    """
    fvgs = []
    for i in range(2, len(df)):
        c1 = df.iloc[i - 2]
        c3 = df.iloc[i]

        # Bullish FVG: gap between candle 1 high and candle 3 low
        if c3["low"] > c1["high"]:
            gap_size = (c3["low"] - c1["high"]) / c1["high"] * 100
            if gap_size >= min_gap_pct:
                fvgs.append({
                    "type": "bullish",
                    "top": c3["low"],
                    "bottom": c1["high"],
                    "midpoint": (c3["low"] + c1["high"]) / 2,
                    "gap_pct": round(gap_size, 3),
                    "index": i,
                    "time": df.index[i],
                    "filled": False,
                })

        # Bearish FVG: gap between candle 3 high and candle 1 low
        if c1["low"] > c3["high"]:
            gap_size = (c1["low"] - c3["high"]) / c3["high"] * 100
            if gap_size >= min_gap_pct:
                fvgs.append({
                    "type": "bearish",
                    "top": c1["low"],
                    "bottom": c3["high"],
                    "midpoint": (c1["low"] + c3["high"]) / 2,
                    "gap_pct": round(gap_size, 3),
                    "index": i,
                    "time": df.index[i],
                    "filled": False,
                })

    return fvgs


def detect_order_blocks(df: pd.DataFrame, lookback: int = 20):
    """
    Order Block: last opposing candle before a strong impulsive move.
    Bullish OB: last bearish candle before a big bullish move.
    Bearish OB: last bullish candle before a big bearish move.
    """
    obs = []
    closes = df["close"].values
    opens = df["open"].values

    # Calculate average candle body size for threshold
    avg_body = np.mean(np.abs(closes - opens))

    for i in range(1, len(df) - 1):
        body = abs(closes[i] - opens[i])

        # Need an impulsive candle (> 2x average body)
        if body < avg_body * 2:
            continue

        # Bullish impulse: look for last bearish candle before it
        if closes[i] > opens[i]:  # bullish impulse
            for j in range(i - 1, max(i - lookback, 0) - 1, -1):
                if closes[j] < opens[j]:  # bearish candle = bullish OB
                    obs.append({
                        "type": "bullish",
                        "top": opens[j],
                        "bottom": closes[j],
                        "index": j,
                        "time": df.index[j],
                        "strength": round(body / avg_body, 1),
                    })
                    break

        # Bearish impulse
        elif closes[i] < opens[i]:
            for j in range(i - 1, max(i - lookback, 0) - 1, -1):
                if closes[j] > opens[j]:  # bullish candle = bearish OB
                    obs.append({
                        "type": "bearish",
                        "top": closes[j],
                        "bottom": opens[j],
                        "index": j,
                        "time": df.index[j],
                        "strength": round(body / avg_body, 1),
                    })
                    break

    return obs


def detect_liquidity_sweeps(df: pd.DataFrame, swing_highs: list, swing_lows: list, lookback: int = 3):
    """
    Liquidity sweep: price spikes beyond a swing point then reverses.
    (Stop hunt / grab liquidity above highs or below lows)
    """
    sweeps = []

    for sh in swing_highs:
        idx = sh["index"]
        if idx + lookback >= len(df):
            continue
        # Check if price went above swing high then closed below it
        for k in range(idx + 1, min(idx + lookback + 1, len(df))):
            if df.iloc[k]["high"] > sh["price"] and df.iloc[k]["close"] < sh["price"]:
                sweeps.append({
                    "type": "bearish_sweep",
                    "level": sh["price"],
                    "sweep_high": df.iloc[k]["high"],
                    "index": k,
                    "time": df.index[k],
                })
                break

    for sl in swing_lows:
        idx = sl["index"]
        if idx + lookback >= len(df):
            continue
        for k in range(idx + 1, min(idx + lookback + 1, len(df))):
            if df.iloc[k]["low"] < sl["price"] and df.iloc[k]["close"] > sl["price"]:
                sweeps.append({
                    "type": "bullish_sweep",
                    "level": sl["price"],
                    "sweep_low": df.iloc[k]["low"],
                    "index": k,
                    "time": df.index[k],
                })
                break

    return sweeps


def get_unfilled_fvgs(fvgs: list, current_price: float, max_age: int = 50):
    """Get FVGs that haven't been filled yet and are recent enough."""
    unfilled = []
    for fvg in fvgs[-max_age:]:
        if fvg["type"] == "bullish" and current_price > fvg["top"]:
            continue  # already passed through
        if fvg["type"] == "bearish" and current_price < fvg["bottom"]:
            continue
        unfilled.append(fvg)
    return unfilled


# ---------------------------------------------------------------------------
# Phase 3: Displacement detection
# ---------------------------------------------------------------------------

def _calculate_atr_series(df: pd.DataFrame, period: int = 14) -> np.ndarray:
    """Return per-bar ATR as a numpy array (same length as df, NaN-padded)."""
    high = df["high"].values
    low = df["low"].values
    close = df["close"].values

    tr = np.empty(len(df))
    tr[0] = high[0] - low[0]
    for i in range(1, len(df)):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))

    atr = np.full(len(df), np.nan)
    if len(df) >= period:
        atr[period - 1] = np.mean(tr[:period])
        for i in range(period, len(df)):
            atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    return atr


def is_displacement_candle(df: pd.DataFrame, idx: int, atr: np.ndarray, multiplier: float = 2.0) -> bool:
    """
    A displacement candle has a body larger than `multiplier` * ATR.
    These are the only candles that should form valid OBs and FVGs.
    """
    if np.isnan(atr[idx]):
        return False
    body = abs(df.iloc[idx]["close"] - df.iloc[idx]["open"])
    return body > multiplier * atr[idx]


# ---------------------------------------------------------------------------
# Phase 3: Premium / Discount zones
# ---------------------------------------------------------------------------

def calculate_premium_discount(swing_highs: list, swing_lows: list) -> dict[str, Any]:
    """
    Divide the current range into premium (above 50%) and discount (below 50%).
    Only allow longs in discount, shorts in premium — this is an ICT core concept.

    Returns dict with equilibrium, premium_start, discount_end, and zone classifier.
    """
    if not swing_highs or not swing_lows:
        return {"equilibrium": 0, "premium_start": 0, "discount_end": 0}

    range_high = max(sh["price"] for sh in swing_highs[-5:])
    range_low = min(sl["price"] for sl in swing_lows[-5:])
    eq = (range_high + range_low) / 2

    return {
        "range_high": range_high,
        "range_low": range_low,
        "equilibrium": eq,
        "premium_start": eq,   # above EQ = premium
        "discount_end": eq,    # below EQ = discount
    }


def get_price_zone(price: float, pd_zones: dict[str, Any]) -> str:
    """Return 'premium', 'discount', or 'equilibrium'."""
    if not pd_zones.get("equilibrium"):
        return "equilibrium"
    eq = pd_zones["equilibrium"]
    rng = pd_zones.get("range_high", eq) - pd_zones.get("range_low", eq)
    if rng == 0:
        return "equilibrium"
    # Small buffer around equilibrium (±2% of range)
    buffer = rng * 0.02
    if price > eq + buffer:
        return "premium"
    elif price < eq - buffer:
        return "discount"
    return "equilibrium"


# ---------------------------------------------------------------------------
# Phase 3: Optimal Trade Entry (OTE)
# ---------------------------------------------------------------------------

def detect_ote(
    df: pd.DataFrame,
    swing_highs: list,
    swing_lows: list,
    current_price: float,
    fvgs: list,
    order_blocks: list,
) -> list[dict[str, Any]]:
    """
    Optimal Trade Entry: price retraces into the 62-79% Fibonacci zone of
    the last impulse move AND sits inside an OB or FVG.

    This is a high-probability ICT entry model.
    """
    otes: list[dict] = []
    if len(swing_highs) < 1 or len(swing_lows) < 1:
        return otes

    # Check bullish OTE: last impulse was up (swing low → swing high).
    last_high = swing_highs[-1]
    last_low = swing_lows[-1]

    if last_low["index"] < last_high["index"]:
        # Bullish impulse: low then high.
        move = last_high["price"] - last_low["price"]
        if move > 0:
            fib_62 = last_high["price"] - move * 0.62
            fib_79 = last_high["price"] - move * 0.79
            if fib_79 <= current_price <= fib_62:
                # Check if price is also inside a bullish OB or FVG.
                in_zone = False
                for ob in order_blocks:
                    if ob["type"] == "bullish" and ob["bottom"] <= current_price <= ob["top"]:
                        in_zone = True
                        break
                if not in_zone:
                    for fvg in fvgs:
                        if fvg["type"] == "bullish" and fvg["bottom"] <= current_price <= fvg["top"]:
                            in_zone = True
                            break
                if in_zone:
                    otes.append({
                        "type": "bullish",
                        "fib_62": round(fib_62, 2),
                        "fib_79": round(fib_79, 2),
                        "price": current_price,
                    })

    if last_high["index"] < last_low["index"]:
        # Bearish impulse: high then low.
        move = last_high["price"] - last_low["price"]
        if move > 0:
            fib_62 = last_low["price"] + move * 0.62
            fib_79 = last_low["price"] + move * 0.79
            if fib_62 <= current_price <= fib_79:
                in_zone = False
                for ob in order_blocks:
                    if ob["type"] == "bearish" and ob["bottom"] <= current_price <= ob["top"]:
                        in_zone = True
                        break
                if not in_zone:
                    for fvg in fvgs:
                        if fvg["type"] == "bearish" and fvg["bottom"] <= current_price <= fvg["top"]:
                            in_zone = True
                            break
                if in_zone:
                    otes.append({
                        "type": "bearish",
                        "fib_62": round(fib_62, 2),
                        "fib_79": round(fib_79, 2),
                        "price": current_price,
                    })

    return otes


# ---------------------------------------------------------------------------
# Phase 3: Breaker Blocks
# ---------------------------------------------------------------------------

def detect_breaker_blocks(
    order_blocks: list[dict],
    df: pd.DataFrame,
) -> list[dict[str, Any]]:
    """
    When an Order Block gets traded through (fails), it becomes a Breaker Block.
    A bullish OB that fails → bearish breaker (support becomes resistance).
    A bearish OB that fails → bullish breaker (resistance becomes support).
    """
    breakers: list[dict] = []
    for ob in order_blocks:
        ob_idx = ob["index"]
        # Look at candles after the OB was formed.
        for i in range(ob_idx + 1, len(df)):
            if ob["type"] == "bullish":
                # Bullish OB fails if price closes below its bottom.
                if df.iloc[i]["close"] < ob["bottom"]:
                    breakers.append({
                        "type": "bearish_breaker",
                        "top": ob["top"],
                        "bottom": ob["bottom"],
                        "original_type": "bullish",
                        "index": ob_idx,
                        "broken_at": i,
                    })
                    break
                # If price bounced off it, the OB is still valid — stop checking.
                if df.iloc[i]["close"] > ob["top"]:
                    break
            elif ob["type"] == "bearish":
                if df.iloc[i]["close"] > ob["top"]:
                    breakers.append({
                        "type": "bullish_breaker",
                        "top": ob["top"],
                        "bottom": ob["bottom"],
                        "original_type": "bearish",
                        "index": ob_idx,
                        "broken_at": i,
                    })
                    break
                if df.iloc[i]["close"] < ob["bottom"]:
                    break
    return breakers


# ---------------------------------------------------------------------------
# Phase 3: Inducement detection
# ---------------------------------------------------------------------------

def detect_inducements(
    swing_highs: list,
    swing_lows: list,
    df: pd.DataFrame,
) -> list[dict[str, Any]]:
    """
    Inducement: price sweeps a minor swing point but NOT the major one.
    This is a fake breakout designed to grab liquidity before the real move.
    High-probability setup when combined with OB entries.
    """
    inducements: list[dict] = []
    if len(swing_highs) < 3 or len(swing_lows) < 3:
        return inducements

    # Check for bullish inducement: minor low swept, major low intact.
    major_low = min(sl["price"] for sl in swing_lows[-3:])
    minor_lows = [sl for sl in swing_lows[-3:] if sl["price"] > major_low]

    for ml in minor_lows:
        idx = ml["index"]
        for i in range(idx + 1, min(idx + 10, len(df))):
            if df.iloc[i]["low"] < ml["price"] and df.iloc[i]["close"] > ml["price"]:
                # Minor low swept with a wick — inducement.
                if df.iloc[i]["low"] > major_low:
                    inducements.append({
                        "type": "bullish_inducement",
                        "minor_level": ml["price"],
                        "major_level": major_low,
                        "sweep_low": df.iloc[i]["low"],
                        "index": i,
                    })
                break

    # Check for bearish inducement: minor high swept, major high intact.
    major_high = max(sh["price"] for sh in swing_highs[-3:])
    minor_highs = [sh for sh in swing_highs[-3:] if sh["price"] < major_high]

    for mh in minor_highs:
        idx = mh["index"]
        for i in range(idx + 1, min(idx + 10, len(df))):
            if df.iloc[i]["high"] > mh["price"] and df.iloc[i]["close"] < mh["price"]:
                if df.iloc[i]["high"] < major_high:
                    inducements.append({
                        "type": "bearish_inducement",
                        "minor_level": mh["price"],
                        "major_level": major_high,
                        "sweep_high": df.iloc[i]["high"],
                        "index": i,
                    })
                break

    return inducements


# ---------------------------------------------------------------------------
# Phase 3: Liquidity Voids
# ---------------------------------------------------------------------------

def detect_liquidity_voids(df: pd.DataFrame, atr: np.ndarray, min_body_atr: float = 2.5) -> list[dict[str, Any]]:
    """
    Liquidity void: large-body candle with minimal wicks (body > min_body_atr * ATR,
    upper/lower wick each < 15% of body). Price tends to return to fill these voids.
    """
    voids: list[dict] = []
    for i in range(len(df)):
        if np.isnan(atr[i]) or atr[i] == 0:
            continue
        row = df.iloc[i]
        body = abs(row["close"] - row["open"])
        if body < min_body_atr * atr[i]:
            continue

        upper_wick = row["high"] - max(row["open"], row["close"])
        lower_wick = min(row["open"], row["close"]) - row["low"]
        if body == 0:
            continue
        if upper_wick / body > 0.15 or lower_wick / body > 0.15:
            continue

        direction = "bullish" if row["close"] > row["open"] else "bearish"
        voids.append({
            "type": direction,
            "top": max(row["open"], row["close"]),
            "bottom": min(row["open"], row["close"]),
            "body_atr_ratio": round(body / atr[i], 1),
            "index": i,
        })
    return voids


# ---------------------------------------------------------------------------
# Enhanced analyze_ict
# ---------------------------------------------------------------------------

def analyze_ict(df: pd.DataFrame, current_price: float):
    """
    Run full ICT analysis and return structured results.

    Phase 3 additions: OTE, displacement-filtered OBs, breaker blocks,
    premium/discount zones, inducements, liquidity voids.
    """
    atr = _calculate_atr_series(df, period=14)

    swing_highs, swing_lows = detect_swing_points(df)
    structure = detect_market_structure(swing_highs, swing_lows)
    bos_choch = detect_bos_choch(swing_highs, swing_lows, current_price)
    fvgs = detect_fair_value_gaps(df)
    unfilled_fvgs = get_unfilled_fvgs(fvgs, current_price)
    order_blocks = detect_order_blocks(df)
    liquidity_sweeps = detect_liquidity_sweeps(df, swing_highs, swing_lows)

    # Phase 3 enhancements.
    pd_zones = calculate_premium_discount(swing_highs, swing_lows)
    price_zone = get_price_zone(current_price, pd_zones)
    ote = detect_ote(df, swing_highs, swing_lows, current_price, unfilled_fvgs, order_blocks)
    breaker_blocks = detect_breaker_blocks(order_blocks, df)
    inducements = detect_inducements(swing_highs, swing_lows, df)
    liquidity_voids = detect_liquidity_voids(df, atr)

    # Tag displacement-quality OBs.
    for ob in order_blocks:
        # The impulse candle after the OB is what matters.
        impulse_idx = ob["index"] + 1
        if impulse_idx < len(df):
            ob["displacement"] = is_displacement_candle(df, impulse_idx, atr)
        else:
            ob["displacement"] = False

    return {
        "structure": structure,
        "swing_highs": swing_highs[-5:],
        "swing_lows": swing_lows[-5:],
        "bos_choch": bos_choch,
        "fvgs": unfilled_fvgs[-5:],
        "order_blocks": order_blocks[-5:],
        "liquidity_sweeps": liquidity_sweeps[-3:],
        # Phase 3
        "price_zone": price_zone,
        "premium_discount": pd_zones,
        "ote": ote,
        "breaker_blocks": breaker_blocks[-3:],
        "inducements": inducements[-3:],
        "liquidity_voids": liquidity_voids[-5:],
    }
