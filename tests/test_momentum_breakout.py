"""Unit tests for the momentum breakout strategy.

No fixtures, no I/O — every test builds its own synthetic 4H DataFrame inline
so the math is obvious from the test body.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from strategies.momentum_breakout import MomentumBreakoutStrategy


def _df(highs, lows, closes, opens=None, volumes=None) -> pd.DataFrame:
    n = len(highs)
    if opens is None:
        opens = closes
    if volumes is None:
        volumes = [1000.0] * n
    idx = pd.date_range("2024-01-01", periods=n, freq="4h", tz="UTC")
    return pd.DataFrame(
        {
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes,
        },
        index=idx,
    )


def _trend_df(n=260, start=100.0, slope=0.5, hl_spread=1.0, seed=0) -> pd.DataFrame:
    """Uptrending series with small noise — large enough to clear the 200 SMA
    warmup."""
    rng = np.random.default_rng(seed)
    closes = np.array([start + i * slope for i in range(n)], dtype=float)
    closes += rng.normal(0.0, 0.05, size=n)  # tiny noise
    highs = closes + hl_spread
    lows = closes - hl_spread
    return _df(highs.tolist(), lows.tolist(), closes.tolist())


def _flat_df(n=260, level=100.0, noise=0.3, seed=1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    closes = level + rng.normal(0.0, noise, size=n)
    highs = closes + 0.2
    lows = closes - 0.2
    return _df(highs.tolist(), lows.tolist(), closes.tolist())


# ---------------------------------------------------------------------------
# compute_indicators
# ---------------------------------------------------------------------------

class TestComputeIndicators:
    def test_columns_present(self):
        df = _trend_df()
        s = MomentumBreakoutStrategy()
        out = s.compute_indicators(df)
        for col in ("sma_200", "sma_50", "donchian_high_20", "atr_14", "atr_median_50"):
            assert col in out.columns

    def test_warmup_nans_then_finite(self):
        df = _trend_df(n=260)
        s = MomentumBreakoutStrategy()
        out = s.compute_indicators(df)
        # SMA-200 needs 200 prior bars; the last bar should be finite.
        assert np.isnan(out["sma_200"].iloc[100])
        assert np.isfinite(out["sma_200"].iloc[-1])
        assert np.isfinite(out["sma_50"].iloc[-1])
        assert np.isfinite(out["donchian_high_20"].iloc[-1])
        assert np.isfinite(out["atr_14"].iloc[-1])
        assert np.isfinite(out["atr_median_50"].iloc[-1])

    def test_donchian_no_lookahead(self):
        """donchian_high_20[N] must equal max(high[N-20:N]), NOT including N."""
        # Build a series where bar 50 has a giant high that should NOT be in
        # its own Donchian, but should appear in bar 51's Donchian.
        n = 80
        highs = [10.0] * n
        lows = [9.0] * n
        closes = [9.5] * n
        highs[50] = 100.0  # huge spike on bar 50
        df = _df(highs, lows, closes)
        out = MomentumBreakoutStrategy().compute_indicators(df)

        # At bar 50, donchian = max of bars 30..49 (which are all 10.0) → 10.0,
        # NOT 100.0.
        assert out["donchian_high_20"].iloc[50] == pytest.approx(10.0)
        # At bar 51, donchian = max of bars 31..50 → 100.0 (spike included).
        assert out["donchian_high_20"].iloc[51] == pytest.approx(100.0)

    def test_atr_matches_risk_manager(self):
        """Strategy ATR series tail should match risk_manager.calculate_atr."""
        from strategies.risk_manager import calculate_atr
        df = _trend_df(n=80)
        out = MomentumBreakoutStrategy().compute_indicators(df)
        # risk_manager's ATR is a scalar over the last `period` bars; compare
        # with the last value of our rolling series.
        rm_atr = calculate_atr(df, period=14)
        assert out["atr_14"].iloc[-1] == pytest.approx(rm_atr, rel=1e-9)


# ---------------------------------------------------------------------------
# check_entry
# ---------------------------------------------------------------------------

class TestCheckEntry:
    def test_bullish_breakout_triggers(self):
        """Strong uptrend with a Donchian breakout on the last bar."""
        n = 250
        # Long uptrend so close > SMA200 and ATR is healthy.
        closes = np.linspace(100.0, 200.0, n).tolist()
        highs = [c + 1.0 for c in closes]
        lows = [c - 1.0 for c in closes]
        # Make the last bar a clean breakout above the prior 20 highs.
        highs[-1] = max(highs[-21:-1]) + 5.0
        closes[-1] = highs[-1] - 0.5  # close just under the high
        df = _df(highs, lows, closes)
        s = MomentumBreakoutStrategy()
        out = s.compute_indicators(df)
        assert s.check_entry(out, len(out) - 1) is True

    def test_chop_no_signal(self):
        # Strictly flat: close exactly at level, high = level + 0.2, low =
        # level - 0.2 on every bar. No noise, so no bar can ever exceed the
        # prior 20 bars' highs (Donchian condition fails on every bar).
        n = 260
        df = _df(
            highs=[100.2] * n,
            lows=[99.8] * n,
            closes=[100.0] * n,
        )
        s = MomentumBreakoutStrategy()
        out = s.compute_indicators(df)
        results = [s.check_entry(out, i) for i in range(220, len(out))]
        assert not any(results)

    def test_below_sma200_no_signal(self):
        """Even with a Donchian breakout, close below SMA-200 must veto."""
        n = 250
        # Downtrend then sharp rip on last bar — close still under SMA200.
        closes = np.linspace(300.0, 100.0, n).tolist()
        highs = [c + 1.0 for c in closes]
        lows = [c - 1.0 for c in closes]
        # Last-bar breakout above the prior 20 highs, but well below SMA200.
        highs[-1] = max(highs[-21:-1]) + 5.0
        closes[-1] = highs[-1] - 0.5
        df = _df(highs, lows, closes)
        s = MomentumBreakoutStrategy()
        out = s.compute_indicators(df)
        # Sanity: SMA200 should sit well above the current close.
        assert out["sma_200"].iloc[-1] > out["close"].iloc[-1]
        assert s.check_entry(out, len(out) - 1) is False

    def test_low_volatility_no_signal(self):
        """High ATR median but current ATR compressed → veto."""
        n = 260
        # Up-trending base.
        closes = np.linspace(100.0, 200.0, n).tolist()
        # Wide highs/lows for the first half (high ATR) then compress.
        highs = [c + 5.0 for c in closes]
        lows = [c - 5.0 for c in closes]
        for i in range(n - 20, n):
            highs[i] = closes[i] + 0.05
            lows[i] = closes[i] - 0.05
        # Make a small "breakout" on the last bar (still up but tiny range).
        highs[-1] = max(highs[-21:-1]) + 0.01
        closes[-1] = highs[-1] - 0.005
        df = _df(highs, lows, closes)
        s = MomentumBreakoutStrategy()
        out = s.compute_indicators(df)
        # Sanity: ATR should now be below its median.
        assert out["atr_14"].iloc[-1] < out["atr_median_50"].iloc[-1]
        assert s.check_entry(out, len(out) - 1) is False

    def test_out_of_range_returns_false(self):
        df = _trend_df()
        s = MomentumBreakoutStrategy()
        out = s.compute_indicators(df)
        assert s.check_entry(out, -1) is False
        assert s.check_entry(out, 10_000) is False


# ---------------------------------------------------------------------------
# check_exit
# ---------------------------------------------------------------------------

class TestCheckExit:
    def _setup(self, n=260):
        df = _trend_df(n=n)
        s = MomentumBreakoutStrategy()
        return s, s.compute_indicators(df)

    def test_trailing_stop_fires(self):
        s, out = self._setup()
        # Force the last bar's low to plunge below highest - 3*ATR.
        idx = len(out) - 1
        atr = float(out["atr_14"].iloc[idx])
        highest = float(out["high"].iloc[idx - 5])
        out = out.copy()
        out.iloc[idx, out.columns.get_loc("low")] = highest - 4 * atr
        position = {"atr_at_entry": atr}
        assert s.check_exit(out, position, idx, highest) == "trailing_stop"

    def test_sma50_break_fires(self):
        s, out = self._setup()
        idx = len(out) - 1
        atr = float(out["atr_14"].iloc[idx])
        sma50 = float(out["sma_50"].iloc[idx])
        out = out.copy()
        # Drop close just below the 50 SMA; lift the low to match so the
        # trailing stop CANNOT trigger first (highest = current close, so
        # trail = close - 3*ATR, well below low).
        new_close = sma50 - 0.5
        out.iloc[idx, out.columns.get_loc("close")] = new_close
        out.iloc[idx, out.columns.get_loc("low")] = new_close
        # highest_since_entry equals the current close — i.e. price has not
        # moved up at all since entry, so the trailing stop is far below.
        position = {"atr_at_entry": atr}
        assert s.check_exit(out, position, idx, new_close) == "sma_50_break"

    def test_no_exit(self):
        s, out = self._setup()
        idx = len(out) - 1
        atr = float(out["atr_14"].iloc[idx])
        # Highest = current close (just entered); low is current low; close
        # is well above SMA-50 by virtue of uptrend.
        highest = float(out["close"].iloc[idx])
        position = {"atr_at_entry": atr}
        assert s.check_exit(out, position, idx, highest) is None

    def test_missing_atr_at_entry_returns_none(self):
        s, out = self._setup()
        idx = len(out) - 1
        assert s.check_exit(out, {}, idx, float(out["close"].iloc[idx])) is None


# ---------------------------------------------------------------------------
# position_size
# ---------------------------------------------------------------------------

class TestPositionSize:
    def test_textbook_formula(self):
        s = MomentumBreakoutStrategy()
        # equity=10_000, entry=100, stop=97 → distance=3, qty = 100/3 ≈ 33.333
        # notional = 33.333 * 100 = 3333.33
        size = s.position_size(10_000.0, 100.0, 97.0)
        assert size == pytest.approx(3333.33, abs=0.01)

    def test_zero_distance_returns_zero(self):
        s = MomentumBreakoutStrategy()
        assert s.position_size(10_000.0, 100.0, 100.0) == 0.0

    def test_non_positive_equity_returns_zero(self):
        s = MomentumBreakoutStrategy()
        assert s.position_size(0.0, 100.0, 97.0) == 0.0
        assert s.position_size(-100.0, 100.0, 97.0) == 0.0

    def test_risk_scales_linearly(self):
        s = MomentumBreakoutStrategy()
        # Doubling equity doubles size for fixed stop distance.
        # Use a clean distance (entry-stop=2) so rounded results are exact.
        a = s.position_size(10_000.0, 100.0, 98.0)
        b = s.position_size(20_000.0, 100.0, 98.0)
        assert a == pytest.approx(5000.0, abs=0.01)
        assert b == pytest.approx(10_000.0, abs=0.01)
