"""
Momentum breakout strategy — 4H Donchian-20 breakout, regime-filtered by the
200 SMA and volatility-filtered by ATR(14) vs its 50-period median. Long-only.

One rule in (all four entry conditions must hold at the close of a 4H candle),
one rule out (whichever fires first of 3xATR trailing stop or close below the
50 SMA). No fixed take-profit. Pure functions, no I/O.

Lookahead discipline:
    - Donchian high uses `shift(1).rolling(20).max()` so the bar's own high is
      not part of its lookback. A breakout requires the close to exceed the
      highest high of the *previous* 20 bars.
    - SMA / ATR can use the current bar's close (which is known at bar close).
    - ATR median is computed from already-completed ATR values, no shift needed.

Position sizing follows the textbook risk-parity formula from the spec:
    qty       = (equity * risk_pct) / |entry - stop|
    notional  = qty * entry
This is intentionally simpler than risk_manager.atr_position_size, which adds
an ATR-scaling factor not in this strategy spec.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from config import Config
from strategies.risk_manager import calculate_atr


SMA_LONG = Config.MOMENTUM_SMA_LONG
SMA_SHORT = Config.MOMENTUM_SMA_SHORT
DONCHIAN_LOOKBACK = Config.MOMENTUM_DONCHIAN
ATR_PERIOD = Config.MOMENTUM_ATR_PERIOD
ATR_MEDIAN_PERIOD = Config.MOMENTUM_ATR_MEDIAN_PERIOD
ATR_STOP_MULT = Config.MOMENTUM_ATR_STOP_MULT
RISK_PCT = Config.MOMENTUM_RISK_PCT / 100.0  # 1.0 → 0.01


def _rolling_atr(df: pd.DataFrame, period: int) -> pd.Series:
    """Vectorized ATR series. Same definition as risk_manager.calculate_atr but
    returns a Series so we can compute its rolling median for the volatility
    filter. The bar's own close → TR contribution is fine; ATR uses the
    previous close, so this is non-lookahead by construction."""
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    prev_close = close.shift(1)

    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(window=period, min_periods=period).mean()


class MomentumBreakoutStrategy:
    """4H momentum breakout, long-only.

    The class holds only configuration; all methods are pure with respect to
    their inputs so they can be unit-tested without instantiating
    infrastructure.
    """

    def __init__(
        self,
        sma_long: int = SMA_LONG,
        sma_short: int = SMA_SHORT,
        donchian_lookback: int = DONCHIAN_LOOKBACK,
        atr_period: int = ATR_PERIOD,
        atr_median_period: int = ATR_MEDIAN_PERIOD,
        atr_stop_mult: float = ATR_STOP_MULT,
        risk_pct: float = RISK_PCT,
    ) -> None:
        self.sma_long = sma_long
        self.sma_short = sma_short
        self.donchian_lookback = donchian_lookback
        self.atr_period = atr_period
        self.atr_median_period = atr_median_period
        self.atr_stop_mult = atr_stop_mult
        self.risk_pct = risk_pct

    def compute_indicators(self, df_4h: pd.DataFrame) -> pd.DataFrame:
        """Return a copy of df_4h with indicator columns added.

        Columns added:
            sma_200, sma_50, donchian_high_20, atr_14, atr_median_50
        Column names track the *defaults*; if you override the constants the
        suffixes are still 200/50/20/14/50 for stability of tests.
        """
        out = df_4h.copy()
        close = out["close"].astype(float)
        high = out["high"].astype(float)

        out["sma_200"] = close.rolling(window=self.sma_long, min_periods=self.sma_long).mean()
        out["sma_50"] = close.rolling(window=self.sma_short, min_periods=self.sma_short).mean()

        # Shift by 1 so the bar's own high is excluded from its Donchian
        # lookback. Without this a "breakout" bar would trivially break out of
        # itself.
        out["donchian_high_20"] = (
            high.shift(1).rolling(window=self.donchian_lookback, min_periods=self.donchian_lookback).max()
        )

        out["atr_14"] = _rolling_atr(out, self.atr_period)
        out["atr_median_50"] = (
            out["atr_14"].rolling(window=self.atr_median_period, min_periods=self.atr_median_period).median()
        )
        return out

    def check_entry(self, df_4h: pd.DataFrame, current_bar_index: int) -> bool:
        """All four entry conditions at the close of bar `current_bar_index`.

        Conditions:
            1. close > sma_200      (uptrend regime)
            2. close > donchian_high_20 of previous 20 bars (breakout)
            3. atr_14 > atr_median_50 (volatility filter)
            4. (no existing open position — checked by the caller)

        Returns False if any required indicator is NaN (warmup) or the bar
        index is out of range.
        """
        if current_bar_index < 0 or current_bar_index >= len(df_4h):
            return False

        required = ("sma_200", "donchian_high_20", "atr_14", "atr_median_50")
        if not all(col in df_4h.columns for col in required):
            df_4h = self.compute_indicators(df_4h)

        row = df_4h.iloc[current_bar_index]
        for col in required:
            v = row[col]
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return False

        close = float(row["close"])
        if close <= float(row["sma_200"]):
            return False
        if close <= float(row["donchian_high_20"]):
            return False
        if float(row["atr_14"]) <= float(row["atr_median_50"]):
            return False
        return True

    def entry_diagnostics(
        self, df_4h: pd.DataFrame, current_bar_index: int
    ) -> dict[str, Any]:
        """Return per-condition booleans plus the values used. For logging in
        bot.py so the acceptance criterion ("logs whether entry conditions
        are met") is satisfied with detail."""
        if current_bar_index < 0 or current_bar_index >= len(df_4h):
            return {"ok": False, "reason": "bar_index_out_of_range"}

        required = ("sma_200", "donchian_high_20", "atr_14", "atr_median_50")
        if not all(col in df_4h.columns for col in required):
            df_4h = self.compute_indicators(df_4h)

        row = df_4h.iloc[current_bar_index]
        close = float(row["close"])

        def _val(col: str) -> float | None:
            v = row[col]
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return None
            return float(v)

        sma_200 = _val("sma_200")
        donch = _val("donchian_high_20")
        atr = _val("atr_14")
        atr_med = _val("atr_median_50")

        cond_uptrend = sma_200 is not None and close > sma_200
        cond_breakout = donch is not None and close > donch
        cond_volatility = atr is not None and atr_med is not None and atr > atr_med
        all_ok = bool(cond_uptrend and cond_breakout and cond_volatility)

        return {
            "ok": all_ok,
            "close": close,
            "sma_200": sma_200,
            "donchian_high_20": donch,
            "atr_14": atr,
            "atr_median_50": atr_med,
            "cond_uptrend": cond_uptrend,
            "cond_breakout": cond_breakout,
            "cond_volatility": cond_volatility,
        }

    def check_exit(
        self,
        df_4h: pd.DataFrame,
        position: dict,
        current_bar_index: int,
        highest_since_entry: float,
    ) -> str | None:
        """Return exit reason at the close of bar `current_bar_index`, or None.

        `position` must include `atr_at_entry` (frozen ATR snapshot from the
        entry bar — the trailing stop uses this, NOT a moving ATR, to avoid
        the stop whipping with volatility).

        Exit priority:
            1. trailing stop — bar's low <= highest_since_entry - 3*ATR_at_entry
            2. sma_50 break — bar's close < sma_50

        Returns "trailing_stop", "sma_50_break", or None.
        """
        if current_bar_index < 0 or current_bar_index >= len(df_4h):
            return None

        if "sma_50" not in df_4h.columns:
            df_4h = self.compute_indicators(df_4h)

        row = df_4h.iloc[current_bar_index]
        atr_at_entry = float(position.get("atr_at_entry", 0.0))
        if atr_at_entry <= 0:
            return None

        trail_stop = highest_since_entry - self.atr_stop_mult * atr_at_entry
        if float(row["low"]) <= trail_stop:
            return "trailing_stop"

        sma_50 = row["sma_50"]
        if sma_50 is not None and not (isinstance(sma_50, float) and np.isnan(sma_50)):
            if float(row["close"]) < float(sma_50):
                return "sma_50_break"

        return None

    def position_size(self, equity: float, entry: float, stop: float) -> float:
        """USD notional size.

        Formula (spec):
            qty       = (equity * risk_pct) / |entry - stop|
            notional  = qty * entry

        Returns 0.0 on degenerate input (entry == stop, non-positive equity).
        """
        distance = abs(float(entry) - float(stop))
        if distance == 0 or equity <= 0 or entry <= 0:
            return 0.0
        qty = (equity * self.risk_pct) / distance
        return round(qty * entry, 2)


# Module-level singleton for callers that don't need to override params.
default_strategy = MomentumBreakoutStrategy()


__all__ = ["MomentumBreakoutStrategy", "default_strategy", "calculate_atr"]
