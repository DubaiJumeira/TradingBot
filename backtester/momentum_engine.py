"""Momentum breakout backtest engine.

Walks 4H bars; signal fires at the close of bar N → fill at the open of bar
N+1. Models taker fees, adverse slippage, and funding cost on held bars.

This is a separate class from the legacy `BacktestEngine` because the legacy
engine is hard-wired to the ICT/Wyckoff confluence pipeline, assumes 15m
base TF, and has no cost model. Mutating it to accept a strategy object would
risk the legacy mode the user still needs accessible.

Lookahead discipline:
    * Entry signal at close of bar N is evaluated using indicator values at
      bar N. Indicators themselves are non-lookahead (donchian uses shift(1)).
    * Entry fill price is the OPEN of bar N+1, then adjusted for slippage and
      fee. The strategy never sees bar N+1's data when deciding to enter.
    * Exit checks on bar M: trailing stop is checked against the bar's low
      (intrabar), SMA-50 break is checked against the bar's close. Both fill
      on bar M+1's open with slippage + fee (next-bar lag), matching the
      entry convention.

Funding: 0.01% (configurable) of notional is deducted every 8 hours. With 4H
bars that's every other bar. Charged regardless of P&L direction.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from config import Config
from strategies.momentum_breakout import MomentumBreakoutStrategy


BARS_PER_DAY_4H = 6  # 24 / 4
BARS_PER_YEAR_4H = BARS_PER_DAY_4H * 365


@dataclass
class MomentumTrade:
    symbol: str
    entry_time: pd.Timestamp
    entry_price: float        # incl. slippage
    raw_entry_price: float    # bar N+1 open, before slippage
    qty: float
    atr_at_entry: float
    initial_stop: float
    exit_time: pd.Timestamp | None = None
    exit_price: float | None = None
    raw_exit_price: float | None = None
    exit_reason: str | None = None
    bars_held: int = 0
    gross_pnl: float = 0.0
    fees_paid: float = 0.0
    funding_paid: float = 0.0
    net_pnl: float = 0.0
    highest_since_entry: float = 0.0


class MomentumBacktestEngine:
    """Bar-by-bar 4H backtest of MomentumBreakoutStrategy with realistic costs."""

    def __init__(
        self,
        symbol: str = "BTC/USDT",
        balance: float = 10_000.0,
        strategy: MomentumBreakoutStrategy | None = None,
        fee_pct: float = Config.MOMENTUM_FEE_PCT,
        slippage_pct: float = Config.MOMENTUM_SLIPPAGE_PCT,
        funding_pct_8h: float = Config.MOMENTUM_FUNDING_PCT_8H,
        random_seed: int = 0,
    ) -> None:
        self.symbol = symbol
        self.starting_balance = float(balance)
        self.balance = float(balance)
        self.strategy = strategy or MomentumBreakoutStrategy()
        self.fee_pct = float(fee_pct) / 100.0
        self.slippage_pct = float(slippage_pct) / 100.0
        self.funding_pct_8h = float(funding_pct_8h) / 100.0
        self._rng = np.random.default_rng(int(random_seed))

        self.trades: list[MomentumTrade] = []
        self.open_trade: MomentumTrade | None = None
        self.equity_curve: list[tuple[pd.Timestamp, float]] = []
        self._bars_in_market = 0
        self._total_bars = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, df_4h: pd.DataFrame) -> dict[str, Any]:
        """Run the backtest over the provided 4H OHLCV DataFrame.

        df_4h must be indexed by UTC timestamps and have columns
        open/high/low/close/volume.
        """
        if df_4h.empty:
            return self._empty_results()

        df = self.strategy.compute_indicators(df_4h)
        n = len(df)
        warmup = max(
            self.strategy.sma_long,
            self.strategy.atr_median_period + self.strategy.atr_period,
        )

        # The engine processes bars index by index. On bar i we:
        #   1. (If a trade is open) update highest_since_entry; check exit
        #      conditions using bar i data — but mark exit pending so the
        #      actual fill is on bar i+1's open. Apply funding for this bar.
        #   2. (If flat) check entry condition at close of bar i; mark entry
        #      pending so fill is on bar i+1's open.
        # On bar i+1:
        #   3. Fill pending exit at open.
        #   4. Fill pending entry at open.
        pending_entry = False
        pending_exit_reason: str | None = None

        self._total_bars = max(0, n - warmup)

        for i in range(n):
            ts = df.index[i]
            bar = df.iloc[i]

            # --- 1. Execute pending fills on this bar's OPEN -----------
            if pending_exit_reason is not None and self.open_trade is not None:
                self._fill_exit(self.open_trade, bar, ts, pending_exit_reason)
                pending_exit_reason = None

            if pending_entry and self.open_trade is None:
                # Entry uses this bar's open. Get ATR / stop from the *previous*
                # bar (i-1), which is where the signal was generated.
                signal_bar = df.iloc[i - 1]
                self._fill_entry(bar, ts, signal_bar)
            pending_entry = False

            # --- 2. Carry / mark-to-market the open trade --------------
            if self.open_trade is not None:
                t = self.open_trade
                t.bars_held += 1
                if float(bar["high"]) > t.highest_since_entry:
                    t.highest_since_entry = float(bar["high"])

                # Funding: every 2 4H bars (8 hours). Charge per bar's notional.
                # Use bar-index parity on bars_held to apply at the 2nd, 4th,
                # ... bar of the hold.
                if t.bars_held > 0 and t.bars_held % 2 == 0:
                    notional = t.qty * float(bar["close"])
                    fund = self.funding_pct_8h * notional
                    t.funding_paid += fund
                    self.balance -= fund

                # Exit checks (signal at close of bar i → fill at open of i+1).
                if i >= warmup and i < n - 1:
                    position = {"atr_at_entry": t.atr_at_entry}
                    reason = self.strategy.check_exit(
                        df, position, i, t.highest_since_entry
                    )
                    if reason is not None:
                        pending_exit_reason = reason

            # --- 3. Entry signal (only if flat) ------------------------
            if self.open_trade is None and i >= warmup and i < n - 1:
                if self.strategy.check_entry(df, i):
                    pending_entry = True

            # --- 4. Record equity ---------------------------------------
            self._bars_in_market += 1 if self.open_trade is not None else 0
            equity = self._mark_to_market(float(bar["close"]))
            self.equity_curve.append((ts, equity))

        # Force-close any trade left open at the end of data.
        if self.open_trade is not None:
            last_bar = df.iloc[-1]
            last_ts = df.index[-1]
            self._fill_exit(self.open_trade, last_bar, last_ts, "end_of_data",
                            override_price=float(last_bar["close"]))

        return self.results()

    # ------------------------------------------------------------------
    # Trade lifecycle
    # ------------------------------------------------------------------

    def _fill_entry(self, bar: pd.Series, ts: pd.Timestamp, signal_bar: pd.Series) -> None:
        raw_open = float(bar["open"])
        # Adverse slippage on a long entry: pay more.
        entry_price = raw_open * (1.0 + self.slippage_pct)
        atr = float(signal_bar["atr_14"])
        if not math.isfinite(atr) or atr <= 0:
            return
        initial_stop = entry_price - self.strategy.atr_stop_mult * atr
        if initial_stop >= entry_price:
            return

        qty = self.strategy.position_size(self.balance, entry_price, initial_stop)
        # `qty` from the strategy is a USD notional. Convert to a base-unit
        # quantity so PnL math works in price units.
        if qty <= 0:
            return
        base_qty = qty / entry_price

        fee = self.fee_pct * (base_qty * entry_price)
        self.balance -= fee

        self.open_trade = MomentumTrade(
            symbol=self.symbol,
            entry_time=ts,
            entry_price=entry_price,
            raw_entry_price=raw_open,
            qty=base_qty,
            atr_at_entry=atr,
            initial_stop=initial_stop,
            highest_since_entry=float(bar["high"]),
            fees_paid=fee,
        )

    def _fill_exit(
        self,
        trade: MomentumTrade,
        bar: pd.Series,
        ts: pd.Timestamp,
        reason: str,
        override_price: float | None = None,
    ) -> None:
        if override_price is not None:
            raw_exit = override_price
        else:
            raw_exit = float(bar["open"])
        # Adverse slippage on a long exit: receive less.
        exit_price = raw_exit * (1.0 - self.slippage_pct)
        gross = (exit_price - trade.entry_price) * trade.qty
        fee = self.fee_pct * (trade.qty * exit_price)

        trade.exit_time = ts
        trade.exit_price = exit_price
        trade.raw_exit_price = raw_exit
        trade.exit_reason = reason
        trade.fees_paid += fee  # entry fee + exit fee
        trade.gross_pnl = gross
        trade.net_pnl = gross - trade.fees_paid - trade.funding_paid

        self.balance += gross - fee
        self.trades.append(trade)
        self.open_trade = None

    def _mark_to_market(self, current_price: float) -> float:
        if self.open_trade is None:
            return self.balance
        t = self.open_trade
        unrealized = (current_price - t.entry_price) * t.qty
        return self.balance + unrealized

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def results(self) -> dict[str, Any]:
        if not self.trades:
            return self._empty_results()

        pnls = np.array([t.net_pnl for t in self.trades])
        wins = pnls[pnls > 0]
        losses = pnls[pnls <= 0]
        gross_win = float(wins.sum()) if wins.size else 0.0
        gross_loss = float(-losses.sum()) if losses.size else 0.0

        equity = np.array([e for _, e in self.equity_curve], dtype=float)
        equity_times = [t for t, _ in self.equity_curve]

        total_return = (equity[-1] / self.starting_balance) - 1.0 if len(equity) else 0.0

        # CAGR over the actual elapsed time of the backtest.
        if len(equity_times) >= 2:
            elapsed_seconds = (equity_times[-1] - equity_times[0]).total_seconds()
            years = elapsed_seconds / (365.25 * 24 * 3600)
        else:
            years = 0.0
        cagr = ((equity[-1] / self.starting_balance) ** (1.0 / years) - 1.0) if years > 0 else 0.0

        # Sharpe — 4H bar returns, annualized via sqrt(6*365).
        rets = pd.Series(equity).pct_change().dropna()
        if len(rets) > 1 and rets.std() > 0:
            sharpe = float(rets.mean() / rets.std() * math.sqrt(BARS_PER_YEAR_4H))
        else:
            sharpe = 0.0

        peak = self.starting_balance
        max_dd = 0.0
        for e in equity:
            if e > peak:
                peak = e
            dd = (peak - e) / peak * 100 if peak > 0 else 0.0
            if dd > max_dd:
                max_dd = dd

        n_trades = len(self.trades)
        win_rate = (len(wins) / n_trades * 100) if n_trades else 0.0
        profit_factor = (gross_win / gross_loss) if gross_loss > 0 else float("inf") if gross_win > 0 else 0.0
        avg_win = float(wins.mean()) if wins.size else 0.0
        avg_loss = float(losses.mean()) if losses.size else 0.0
        exposure_pct = (self._bars_in_market / self._total_bars * 100) if self._total_bars else 0.0

        return {
            "symbol": self.symbol,
            "starting_balance": self.starting_balance,
            "final_balance": round(float(equity[-1]), 2),
            "total_return_pct": round(total_return * 100, 2),
            "cagr_pct": round(cagr * 100, 2),
            "sharpe": round(sharpe, 2),
            "max_drawdown_pct": round(max_dd, 2),
            "win_rate_pct": round(win_rate, 2),
            "profit_factor": round(profit_factor, 2) if math.isfinite(profit_factor) else profit_factor,
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "num_trades": n_trades,
            "exposure_pct": round(exposure_pct, 2),
            "total_fees": round(sum(t.fees_paid for t in self.trades), 2),
            "total_funding": round(sum(t.funding_paid for t in self.trades), 2),
            "trades": [self._trade_dict(t) for t in self.trades],
            "equity_curve": [(t.isoformat(), round(e, 2)) for t, e in self.equity_curve],
        }

    def _trade_dict(self, t: MomentumTrade) -> dict[str, Any]:
        d = asdict(t)
        for k in ("entry_time", "exit_time"):
            v = d.get(k)
            if isinstance(v, pd.Timestamp):
                d[k] = v.isoformat()
        return d

    def _empty_results(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "starting_balance": self.starting_balance,
            "final_balance": self.starting_balance,
            "total_return_pct": 0.0,
            "cagr_pct": 0.0,
            "sharpe": 0.0,
            "max_drawdown_pct": 0.0,
            "win_rate_pct": 0.0,
            "profit_factor": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "num_trades": 0,
            "exposure_pct": 0.0,
            "total_fees": 0.0,
            "total_funding": 0.0,
            "trades": [],
            "equity_curve": [],
        }


__all__ = ["MomentumBacktestEngine", "MomentumTrade"]
