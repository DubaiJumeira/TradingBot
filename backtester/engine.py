"""
Backtesting engine — replays historical data bar-by-bar through the
signal generator and simulates trades with proper SL/TP/trailing logic.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from strategies.ict_strategy import analyze_ict
from strategies.wyckoff_strategy import analyze_wyckoff
from strategies.market_data import calculate_volume_profile, get_current_kill_zone
from strategies.signal_generator import generate_signal
from strategies.risk_manager import calculate_trailing_stop
from strategies.regime_detector import RegimeDetector
from config import Config, get_instrument

logger = logging.getLogger(__name__)


@dataclass
class BacktestTrade:
    """One simulated trade."""
    symbol: str
    side: str
    entry: float
    sl: float
    tp: float
    size_usd: float
    score: int
    open_bar: int
    close_bar: int | None = None
    exit_price: float | None = None
    pnl: float = 0.0
    result: str = "open"  # "TP", "SL", "trailing"
    reasons: list[str] = field(default_factory=list)
    news_triggered: bool = False


class BacktestEngine:
    """
    Bar-by-bar backtester.

    Usage:
        engine = BacktestEngine(symbol="BTC/USDT", balance=10000)
        report = engine.run(df_15m, df_4h)
    """

    def __init__(
        self,
        symbol: str = "BTC/USDT",
        balance: float = 10000,
        max_open: int = 1,
        trailing: bool = True,
    ) -> None:
        self.symbol = symbol
        self.starting_balance = balance
        self.balance = balance
        self.max_open = max_open
        self.trailing = trailing

        self.trades: list[BacktestTrade] = []
        self.open_trades: list[BacktestTrade] = []
        self.equity_curve: list[float] = []
        self._regime_detector = RegimeDetector()

    def run(
        self,
        df_ltf: pd.DataFrame,
        df_htf: pd.DataFrame | None = None,
        min_bars: int = 100,
    ) -> dict[str, Any]:
        """
        Run backtest over df_ltf (lower timeframe, e.g. 15m).

        df_htf is optional higher timeframe for Wyckoff. If not provided,
        a resampled version of df_ltf is used.
        """
        if df_htf is None:
            df_htf = df_ltf.resample("4h").agg({
                "open": "first", "high": "max", "low": "min",
                "close": "last", "volume": "sum",
            }).dropna()

        inst = get_instrument(self.symbol) or {}

        for bar_idx in range(min_bars, len(df_ltf)):
            # Slice data up to current bar (no lookahead).
            ltf_slice = df_ltf.iloc[:bar_idx + 1]
            current_price = ltf_slice.iloc[-1]["close"]
            current_high = ltf_slice.iloc[-1]["high"]
            current_low = ltf_slice.iloc[-1]["low"]

            # Check exits on open trades.
            self._check_exits(current_high, current_low, current_price, bar_idx)

            # Record equity.
            unrealized = sum(self._unrealized_pnl(t, current_price) for t in self.open_trades)
            self.equity_curve.append(self.balance + unrealized)

            # Skip if max open trades reached.
            if len(self.open_trades) >= self.max_open:
                continue

            # Run analysis (every Nth bar to simulate 5-min cycle efficiency).
            if bar_idx % 4 != 0:
                continue

            try:
                ict = analyze_ict(ltf_slice, current_price)

                htf_cutoff = ltf_slice.index[-1]
                htf_slice = df_htf[df_htf.index <= htf_cutoff]
                if len(htf_slice) < 20:
                    continue
                wyckoff = analyze_wyckoff(htf_slice)

                # Simplified market data for backtest (no live exchange).
                market = {
                    "funding": {"signal": "neutral", "rate": 0},
                    "open_interest": {},
                    "kill_zone": {"active": True, "zone": "new_york", "weight": 1.0},
                    "volume_profile": calculate_volume_profile(ltf_slice.tail(50)),
                }

                regime = self._regime_detector.detect(ltf_slice)

                signal = generate_signal(
                    self.symbol, current_price, ict, wyckoff, market,
                    self.balance, instrument=inst, regime=regime,
                )

                if signal:
                    trade = BacktestTrade(
                        symbol=self.symbol,
                        side=signal["side"],
                        entry=signal["entry"],
                        sl=signal["sl"],
                        tp=signal["tp"],
                        size_usd=signal["size_usd"],
                        score=signal["score"],
                        open_bar=bar_idx,
                        reasons=signal.get("reasons", []),
                        news_triggered=signal.get("news_triggered", False),
                    )
                    self.open_trades.append(trade)
                    self.balance -= trade.size_usd

            except Exception as exc:
                logger.debug("Backtest bar %d error: %s", bar_idx, exc)

        # Close any remaining open trades at last price.
        if len(df_ltf) > 0:
            last_price = df_ltf.iloc[-1]["close"]
            for trade in list(self.open_trades):
                self._close_trade(trade, last_price, len(df_ltf) - 1, "end_of_data")

        return self.results()

    def _check_exits(self, high: float, low: float, close: float, bar_idx: int) -> None:
        for trade in list(self.open_trades):
            # Trailing stop.
            if self.trailing:
                new_sl = calculate_trailing_stop(
                    side=trade.side, entry=trade.entry,
                    current_sl=trade.sl, current_price=close, tp=trade.tp,
                )
                if new_sl is not None:
                    trade.sl = new_sl

            if trade.side == "long":
                if low <= trade.sl:
                    self._close_trade(trade, trade.sl, bar_idx, "SL")
                elif high >= trade.tp:
                    self._close_trade(trade, trade.tp, bar_idx, "TP")
            else:
                if high >= trade.sl:
                    self._close_trade(trade, trade.sl, bar_idx, "SL")
                elif low <= trade.tp:
                    self._close_trade(trade, trade.tp, bar_idx, "TP")

    def _close_trade(self, trade: BacktestTrade, exit_price: float, bar_idx: int, result: str) -> None:
        trade.exit_price = exit_price
        trade.close_bar = bar_idx
        trade.result = result

        if trade.side == "long":
            trade.pnl = (exit_price - trade.entry) * (trade.size_usd / trade.entry)
        else:
            trade.pnl = (trade.entry - exit_price) * (trade.size_usd / trade.entry)

        self.balance += trade.size_usd + trade.pnl
        self.trades.append(trade)
        self.open_trades.remove(trade)

    def _unrealized_pnl(self, trade: BacktestTrade, current_price: float) -> float:
        if trade.side == "long":
            return (current_price - trade.entry) * (trade.size_usd / trade.entry)
        return (trade.entry - current_price) * (trade.size_usd / trade.entry)

    def results(self) -> dict[str, Any]:
        """Compute backtest statistics."""
        if not self.trades:
            return {
                "total_trades": 0, "wins": 0, "losses": 0,
                "win_rate": 0, "profit_factor": 0,
                "total_pnl": 0, "avg_pnl": 0,
                "max_drawdown_pct": 0, "sharpe_ratio": 0,
                "final_balance": self.balance, "trades": [],
                "equity_curve": self.equity_curve,
            }

        wins = [t for t in self.trades if t.pnl > 0]
        losses = [t for t in self.trades if t.pnl <= 0]
        total_pnl = sum(t.pnl for t in self.trades)
        gross_profit = sum(t.pnl for t in wins) if wins else 0
        gross_loss = abs(sum(t.pnl for t in losses)) if losses else 1

        # Max drawdown from equity curve.
        max_dd = 0.0
        peak = self.starting_balance
        for eq in self.equity_curve:
            if eq > peak:
                peak = eq
            dd = (peak - eq) / peak * 100
            if dd > max_dd:
                max_dd = dd

        # Simplified Sharpe (annualized assuming 15m bars).
        import numpy as np
        returns = pd.Series(self.equity_curve).pct_change().dropna()
        sharpe = 0.0
        if len(returns) > 1 and returns.std() > 0:
            sharpe = float(returns.mean() / returns.std() * np.sqrt(252 * 24 * 4))

        return {
            "total_trades": len(self.trades),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": round(len(wins) / len(self.trades) * 100, 1),
            "profit_factor": round(gross_profit / gross_loss, 2) if gross_loss > 0 else 0,
            "total_pnl": round(total_pnl, 2),
            "avg_pnl": round(total_pnl / len(self.trades), 2),
            "max_drawdown_pct": round(max_dd, 2),
            "sharpe_ratio": round(sharpe, 2),
            "final_balance": round(self.balance, 2),
            "equity_curve": self.equity_curve,
            "trades": [
                {
                    "side": t.side, "entry": t.entry, "exit": t.exit_price,
                    "sl": t.sl, "tp": t.tp, "pnl": round(t.pnl, 2),
                    "result": t.result, "score": t.score,
                    "bars_held": (t.close_bar or 0) - t.open_bar,
                    "news_triggered": t.news_triggered,
                }
                for t in self.trades
            ],
        }
