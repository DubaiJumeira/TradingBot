"""
Phase 6 — Risk Management

Four components that sit between signal generation and order execution:

    1. ATR-based position sizing — scales position size inversely with
       current volatility. High ATR → wider stops → smaller position so
       the dollar risk stays constant.

    2. Drawdown circuit breaker — tracks peak equity vs. current equity.
       If drawdown exceeds MAX_DRAWDOWN_PCT, blocks all new entries and
       fires a Telegram alert. Resets once equity recovers above the
       threshold.

    3. Correlation-aware exposure limits — prevents piling into
       correlated instruments (e.g., 3 crypto longs at once). Uses
       CORRELATION_GROUPS from config.

    4. Trailing stop management — once a position reaches a favorable
       R:R threshold (e.g., 1:1), the stop-loss is trailed upward
       (for longs) or downward (for shorts) to lock in profit.

All four are stateless functions or lightweight classes so they're
easy to test in isolation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from config import Config, CORRELATION_GROUPS, get_correlation_group

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. ATR-based position sizing
# ---------------------------------------------------------------------------

def calculate_atr(df: pd.DataFrame, period: int = 14) -> float:
    """
    Calculate the Average True Range over the last `period` candles.

    Parameters
    ----------
    df : DataFrame with columns 'high', 'low', 'close'.
    period : lookback window (default 14).

    Returns
    -------
    ATR value as a float.  Returns 0.0 if not enough data.
    """
    if len(df) < period + 1:
        return 0.0

    high = df["high"].values
    low = df["low"].values
    close = df["close"].values

    tr = np.empty(len(df) - 1)
    for i in range(1, len(df)):
        tr[i - 1] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1]),
        )

    # Simple moving average of the most recent `period` TR values.
    return float(np.mean(tr[-period:]))


def atr_position_size(
    balance: float,
    risk_pct: float,
    entry: float,
    sl: float,
    atr: float,
    atr_scale: float = Config.ATR_POSITION_SCALE,
) -> float:
    """
    Calculate position size using ATR to normalize volatility.

    The idea: if ATR is large relative to the SL distance, the instrument
    is very volatile and we should size down.  If ATR is small, size up.

    Formula:
        base_risk   = balance * risk_pct / 100
        sl_distance = |entry - sl|
        atr_factor  = min(1.0, (atr_scale * sl_distance) / atr)  if atr > 0
        qty         = (base_risk * atr_factor) / sl_distance
        size_usd    = qty * entry

    When atr_scale * sl_distance >= atr, atr_factor = 1.0 (normal size).
    When atr is much larger than sl_distance, factor shrinks → smaller pos.

    Parameters
    ----------
    balance : account equity in USD.
    risk_pct : risk per trade as percentage (e.g. 1.0 = 1%).
    entry : entry price.
    sl : stop-loss price.
    atr : current ATR value (same price units as entry/sl).
    atr_scale : multiplier that controls how aggressively ATR compresses
                position size (default from Config).
    """
    sl_distance = abs(entry - sl)
    if sl_distance == 0 or atr <= 0:
        return 0.0

    base_risk = balance * (risk_pct / 100)
    atr_factor = min(1.0, (atr_scale * sl_distance) / atr)
    qty = (base_risk * atr_factor) / sl_distance
    return round(qty * entry, 2)


# ---------------------------------------------------------------------------
# 2. Drawdown circuit breaker
# ---------------------------------------------------------------------------

class DrawdownMonitor:
    """
    Tracks equity peaks and triggers a circuit breaker when drawdown
    exceeds the configured threshold.

    Usage:
        monitor = DrawdownMonitor(starting_balance=10000)
        monitor.update(current_equity)
        if monitor.is_breaker_active:
            # block new entries
    """

    def __init__(
        self,
        starting_balance: float,
        max_drawdown_pct: float = Config.MAX_DRAWDOWN_PCT,
    ) -> None:
        self.peak_equity = starting_balance
        self.current_equity = starting_balance
        self.max_drawdown_pct = max_drawdown_pct
        self._breaker_active = False

    def update(self, equity: float) -> None:
        """Update with the latest equity value."""
        self.current_equity = equity
        if equity > self.peak_equity:
            self.peak_equity = equity

        dd_pct = self.drawdown_pct
        if dd_pct >= self.max_drawdown_pct:
            if not self._breaker_active:
                logger.warning(
                    "CIRCUIT BREAKER TRIGGERED: drawdown %.1f%% >= %.1f%% threshold "
                    "(peak=%.2f, current=%.2f)",
                    dd_pct, self.max_drawdown_pct,
                    self.peak_equity, self.current_equity,
                )
            self._breaker_active = True
        else:
            if self._breaker_active:
                logger.info(
                    "Circuit breaker reset: drawdown %.1f%% < %.1f%% "
                    "(peak=%.2f, current=%.2f)",
                    dd_pct, self.max_drawdown_pct,
                    self.peak_equity, self.current_equity,
                )
            self._breaker_active = False

    @property
    def drawdown_pct(self) -> float:
        """Current drawdown as a percentage (0.0 = no drawdown)."""
        if self.peak_equity <= 0:
            return 0.0
        return ((self.peak_equity - self.current_equity) / self.peak_equity) * 100

    @property
    def is_breaker_active(self) -> bool:
        return self._breaker_active

    def status(self) -> dict[str, Any]:
        return {
            "peak_equity": round(self.peak_equity, 2),
            "current_equity": round(self.current_equity, 2),
            "drawdown_pct": round(self.drawdown_pct, 2),
            "breaker_active": self._breaker_active,
        }


# ---------------------------------------------------------------------------
# 3. Correlation-aware exposure limits
# ---------------------------------------------------------------------------

def check_exposure(
    symbol: str,
    open_positions: dict[str, dict],
    side: str | None = None,
) -> dict[str, Any]:
    """
    Check if opening a new position on `symbol` would violate
    correlation group exposure limits.

    Checks both total positions in the group AND same-direction
    positions (e.g. two crypto shorts = same directional bet).

    Parameters
    ----------
    symbol : the instrument we want to trade.
    open_positions : dict of trade_id → position dict (must have 'symbol' key).
    side : "long" or "short" for directional check (optional).

    Returns
    -------
    {
        "allowed": bool,
        "group": str | None,
        "current_count": int,
        "max_allowed": int,
        "reason": str | None,
    }
    """
    group_name = get_correlation_group(symbol)
    if group_name is None:
        return {"allowed": True, "group": None, "current_count": 0,
                "max_allowed": 0, "reason": None}

    group = CORRELATION_GROUPS[group_name]
    group_symbols = set(group["symbols"])
    max_pos = group["max_positions"]
    max_same_dir = group.get("max_same_direction", max_pos)

    group_positions = [
        pos for pos in open_positions.values()
        if pos.get("symbol") in group_symbols
    ]
    count = len(group_positions)

    if count >= max_pos:
        reason = (
            f"Correlation limit: {count}/{max_pos} positions already open "
            f"in '{group_name}' group ({', '.join(sorted(group_symbols))})"
        )
        return {"allowed": False, "group": group_name, "current_count": count,
                "max_allowed": max_pos, "reason": reason}

    if side is not None and max_same_dir < max_pos:
        same_dir = sum(1 for p in group_positions if p.get("side") == side)
        if same_dir >= max_same_dir:
            reason = (
                f"Directional limit: {same_dir} {side}(s) already open "
                f"in '{group_name}' group (max {max_same_dir} same direction)"
            )
            return {"allowed": False, "group": group_name, "current_count": count,
                    "max_allowed": max_pos, "reason": reason}

    return {"allowed": True, "group": group_name, "current_count": count,
            "max_allowed": max_pos, "reason": None}


# ---------------------------------------------------------------------------
# 4. Trailing stop management
# ---------------------------------------------------------------------------

def calculate_trailing_stop(
    side: str,
    entry: float,
    current_sl: float,
    current_price: float,
    tp: float,
    activation_rr: float = Config.TRAILING_STOP_ACTIVATION_RR,
    step_pct: float = Config.TRAILING_STOP_STEP_PCT,
    df=None,
) -> float | None:
    """Structure-based trailing stop with ATR buffer.

    When a DataFrame is provided, the stop trails below the most recent
    swing low (longs) or above the most recent swing high (shorts),
    offset by 0.5× ATR to avoid stop hunts. Falls back to the
    mechanical step-based trail when no structure is available.

    The trailing stop only activates once the unrealised R:R reaches
    ``activation_rr``. It never moves backwards.
    """
    risk = abs(entry - current_sl)
    if risk == 0:
        return None

    if side == "long":
        unrealized_rr = (current_price - entry) / risk
    else:
        unrealized_rr = (entry - current_price) / risk
    if unrealized_rr < activation_rr:
        return None

    # --- Structure-based trail (preferred) ---
    if df is not None and len(df) >= 20:
        new_sl = _structure_trail(side, current_sl, current_price, df)
        if new_sl is not None:
            return new_sl

    # --- Fallback: mechanical step trail ---
    step = entry * (step_pct / 100)
    if side == "long":
        new_sl = current_price - step
        if new_sl > current_sl:
            return round(new_sl, 2)
    else:
        new_sl = current_price + step
        if new_sl < current_sl:
            return round(new_sl, 2)
    return None


def _structure_trail(
    side: str,
    current_sl: float,
    current_price: float,
    df,
    swing_lookback: int = 3,
    atr_period: int = 14,
    atr_buffer_mult: float = 0.5,
) -> float | None:
    """Find the best structural trailing stop from swing points + ATR.

    For longs: SL = most recent swing low below current price − ATR buffer.
    For shorts: SL = most recent swing high above current price + ATR buffer.

    Uses a shorter swing_lookback (3) on 15m data to detect minor
    structure (newly formed higher lows / lower highs), not just major
    pivots.
    """
    import numpy as np

    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values

    # ATR for the buffer.
    if len(df) < atr_period + 1:
        return None
    tr = np.maximum(
        highs[1:] - lows[1:],
        np.maximum(
            np.abs(highs[1:] - closes[:-1]),
            np.abs(lows[1:] - closes[:-1]),
        ),
    )
    atr = float(np.mean(tr[-atr_period:]))
    buffer = atr * atr_buffer_mult

    # Detect swing points with shorter lookback for recent structure.
    swing_points: list[float] = []
    end = len(df) - swing_lookback
    start = max(swing_lookback, len(df) - 60)

    if side == "long":
        for i in range(start, end):
            window = lows[i - swing_lookback : i + swing_lookback + 1]
            if lows[i] == min(window) and lows[i] < current_price:
                swing_points.append(float(lows[i]))
        if not swing_points:
            return None
        # Use the highest (most recent structurally) swing low.
        struct_level = max(swing_points)
        new_sl = round(struct_level - buffer, 2)
        if new_sl > current_sl and new_sl < current_price:
            return new_sl
    else:
        for i in range(start, end):
            window = highs[i - swing_lookback : i + swing_lookback + 1]
            if highs[i] == max(window) and highs[i] > current_price:
                swing_points.append(float(highs[i]))
        if not swing_points:
            return None
        struct_level = min(swing_points)
        new_sl = round(struct_level + buffer, 2)
        if new_sl < current_sl and new_sl > current_price:
            return new_sl

    return None


# ---------------------------------------------------------------------------
# Composite pre-trade check
# ---------------------------------------------------------------------------

class RiskManager:
    """
    Facade that combines all risk checks into a single interface
    for the bot's main loop.
    """

    def __init__(
        self,
        starting_balance: float = Config.STARTING_BALANCE,
        max_drawdown_pct: float = Config.MAX_DRAWDOWN_PCT,
        max_daily_loss_pct: float = 3.0,
        max_loss_streak: int = 3,
    ) -> None:
        self.drawdown = DrawdownMonitor(starting_balance, max_drawdown_pct)
        self.daily_loss = DailyLossTracker(max_daily_loss_pct)
        self.streak = ConsecutiveLossTracker(max_loss_streak)

    def update_equity(self, equity: float) -> None:
        """Call each cycle with current account equity."""
        self.drawdown.update(equity)

    def record_trade_close(self, pnl: float, equity: float) -> None:
        """Hook called from the exits loop whenever a trade closes.
        Updates daily-loss and streak trackers so the next entry check
        can veto if needed.
        """
        self.daily_loss.record_close(pnl, equity)
        self.streak.record_close(pnl)

    def pre_trade_check(
        self,
        symbol: str,
        open_positions: dict[str, dict],
        side: str | None = None,
    ) -> dict[str, Any]:
        """
        Run all pre-trade risk checks before opening a new position.

        Returns:
            {
                "allowed": bool,
                "reasons": list[str],   # why blocked (empty if allowed)
                "drawdown": dict,       # drawdown monitor status
                "exposure": dict,       # exposure check result
            }
        """
        reasons: list[str] = []

        # 1. Drawdown circuit breaker.
        if self.drawdown.is_breaker_active:
            reasons.append(
                f"Circuit breaker active: drawdown {self.drawdown.drawdown_pct:.1f}% "
                f">= {self.drawdown.max_drawdown_pct:.1f}% threshold"
            )

        # 2. Correlation exposure (total + same-direction).
        exposure = check_exposure(symbol, open_positions, side=side)
        if not exposure["allowed"]:
            reasons.append(exposure["reason"])

        # 3. Daily loss limit.
        if self.daily_loss.is_blocked(self.drawdown.current_equity):
            s = self.daily_loss.status()
            reasons.append(
                f"Daily loss limit: -{s['loss_pct']:.1f}% "
                f">= -{s['limit_pct']:.1f}%"
            )

        # 4. Consecutive loss streak.
        if self.streak.is_blocked:
            reasons.append(
                f"Consecutive loss streak: {self.streak._streak} losses "
                f"(max {self.streak.max_streak}) — stepping aside"
            )

        return {
            "allowed": len(reasons) == 0,
            "reasons": reasons,
            "drawdown": self.drawdown.status(),
            "exposure": exposure,
            "daily_loss": self.daily_loss.status(),
            "streak": self.streak.status(),
        }

    def status(self) -> dict[str, Any]:
        return {
            "drawdown": self.drawdown.status(),
            "daily_loss": self.daily_loss.status(),
            "streak": self.streak.status(),
        }


# ---------------------------------------------------------------------------
# 5. Partial take-profit plans (Phase 6 extension)
# ---------------------------------------------------------------------------


@dataclass
class PartialTPLevel:
    """One rung of a scaled-exit plan.

    Each rung closes a percentage of the remaining qty when price
    reaches ``rr`` multiples of the original risk. ``post_action`` is
    what should happen to the STOP after this TP fills — typically
    "breakeven" for TP1, "trail" for TP2, and None for the final runner.
    """
    rr: float
    close_pct: float  # 0-1, fraction of the ORIGINAL position to close
    price: float
    post_action: str | None = None  # "breakeven" | "trail" | None
    filled: bool = False


@dataclass
class PartialTPPlan:
    """Three-rung plan: TP1 = 2R (50%, SL→BE), TP2 = 3R (30%, trail),
    TP3 = user TP (20%, full exit).

    The ``levels`` list is always length 3 so downstream code can index
    it. ``compute_from_signal`` handles the side math — for a long, TP
    prices are above entry; for a short they're below.
    """
    levels: list[PartialTPLevel] = field(default_factory=list)

    @classmethod
    def compute_from_signal(
        cls,
        side: str,
        entry: float,
        sl: float,
        tp: float,
    ) -> "PartialTPPlan":
        risk = abs(entry - sl)
        if risk <= 0 or entry <= 0:
            return cls(levels=[])

        # TP3 is the planned ICT TP. TP1 / TP2 are derived from R multiples.
        def px(r_mult: float) -> float:
            if side == "long":
                return round(entry + risk * r_mult, 2)
            return round(entry - risk * r_mult, 2)

        tp1_price = px(2.0)  # TP1 always at 2R
        tp2_price = px(3.0)  # TP2 at 3R
        tp3_price = round(tp, 2)  # respect the caller's TP exactly

        # If tp3 falls below the 2R mark (tight ICT target), collapse
        # to a 2-rung plan so we don't have overlapping fills.
        if side == "long" and tp3_price <= tp1_price:
            levels = [
                PartialTPLevel(rr=2.0, close_pct=0.5, price=tp1_price, post_action="breakeven"),
                PartialTPLevel(rr=abs(tp3_price - entry) / risk, close_pct=0.5, price=tp3_price),
            ]
        elif side == "short" and tp3_price >= tp1_price:
            levels = [
                PartialTPLevel(rr=2.0, close_pct=0.5, price=tp1_price, post_action="breakeven"),
                PartialTPLevel(rr=abs(entry - tp3_price) / risk, close_pct=0.5, price=tp3_price),
            ]
        else:
            levels = [
                PartialTPLevel(rr=2.0, close_pct=0.5, price=tp1_price, post_action="breakeven"),
                PartialTPLevel(rr=3.0, close_pct=0.3, price=tp2_price, post_action="trail"),
                PartialTPLevel(
                    rr=abs(tp3_price - entry) / risk,
                    close_pct=0.2,
                    price=tp3_price,
                ),
            ]
        return cls(levels=levels)

    def to_dict(self) -> dict:
        return {
            "levels": [
                {
                    "rr": lvl.rr,
                    "close_pct": lvl.close_pct,
                    "price": lvl.price,
                    "post_action": lvl.post_action,
                    "filled": lvl.filled,
                }
                for lvl in self.levels
            ],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PartialTPPlan":
        levels = [
            PartialTPLevel(
                rr=lvl["rr"],
                close_pct=lvl["close_pct"],
                price=lvl["price"],
                post_action=lvl.get("post_action"),
                filled=lvl.get("filled", False),
            )
            for lvl in data.get("levels", [])
        ]
        return cls(levels=levels)


# ---------------------------------------------------------------------------
# 6. Daily loss limit
# ---------------------------------------------------------------------------


class DailyLossTracker:
    """Blocks new entries once the day's realised PnL falls below
    ``-max_daily_loss_pct`` of starting-of-day equity.

    Resets automatically on UTC midnight. This is a separate safeguard
    from the drawdown circuit breaker: the drawdown monitor watches
    peak-to-trough over any timeframe, this one just catches bad days
    before they compound.
    """

    def __init__(self, max_daily_loss_pct: float = 3.0) -> None:
        self.max_daily_loss_pct = max_daily_loss_pct
        self._day: date | None = None
        self._start_equity: float = 0.0
        self._realised_today: float = 0.0

    def _roll_day_if_needed(self, equity: float) -> None:
        today = datetime.now(tz=timezone.utc).date()
        if self._day != today:
            self._day = today
            self._start_equity = equity
            self._realised_today = 0.0

    def record_close(self, pnl: float, equity: float) -> None:
        self._roll_day_if_needed(equity)
        self._realised_today += pnl

    def is_blocked(self, equity: float) -> bool:
        self._roll_day_if_needed(equity)
        if self._start_equity <= 0:
            return False
        loss_pct = (-self._realised_today / self._start_equity) * 100
        return loss_pct >= self.max_daily_loss_pct

    def status(self) -> dict:
        return {
            "date": self._day.isoformat() if self._day else None,
            "start_equity": round(self._start_equity, 2),
            "realised_pnl": round(self._realised_today, 2),
            "loss_pct": round(
                (-self._realised_today / self._start_equity * 100)
                if self._start_equity > 0 else 0.0,
                2,
            ),
            "limit_pct": self.max_daily_loss_pct,
        }


# ---------------------------------------------------------------------------
# 7. Consecutive loss circuit
# ---------------------------------------------------------------------------


class ConsecutiveLossTracker:
    """Pauses new entries after N consecutive losing trades.

    Streaks are a classic sign the market regime has shifted against
    the strategy. Stepping aside for a cycle avoids compounding the
    mistake. Resets on the first winning trade.
    """

    def __init__(self, max_streak: int = 3) -> None:
        self.max_streak = max_streak
        self._streak = 0

    def record_close(self, pnl: float) -> None:
        if pnl > 0:
            self._streak = 0
        else:
            self._streak += 1

    @property
    def is_blocked(self) -> bool:
        return self._streak >= self.max_streak

    def status(self) -> dict:
        return {
            "streak": self._streak,
            "max_streak": self.max_streak,
            "blocked": self.is_blocked,
        }
