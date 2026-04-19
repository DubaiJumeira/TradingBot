"""
Phase 13 — historical replay backtester.

Walks forward bar-by-bar through real exchange data, running the full
signal pipeline (ICT / Wyckoff / regime / MTF / Phase 7 advanced ICT /
Phase 12 confluence) at each step and simulating entries, partial TPs,
and SL exits against subsequent candles.

Usage:
    ./venv/bin/python -m tools.backtest BTCUSDT 30
    ./venv/bin/python -m tools.backtest SOLUSDT 10 --tf 15m

The backtester deliberately stubs out data sources that don't have a
historical analogue (funding rate, OI, order flow, news, manipulation
tracker) so scoring is deterministic against candle data alone. This
gives a pessimistic estimate of live-bot performance — anything that
works in backtest should only get stronger with those extra inputs.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from typing import Any

import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)
np.seterr(divide="ignore", invalid="ignore")

# Let this run as both a module and a script.
if __package__ is None or __package__ == "":
    sys.path.insert(0, "/root/trading-bot")

from config import Config, get_instrument
from exchange_handler import ExchangeHandler
from strategies.ict_strategy import analyze_ict
from strategies.mtf_analysis import MTFState
from strategies.regime_detector import RegimeDetector
from strategies.signal_generator import generate_signal
from strategies.wyckoff_strategy import analyze_wyckoff
from strategies.leverage import apply_leverage_to_signal
from strategies.market_data import calculate_volume_profile

logger = logging.getLogger("backtest")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def fetch_ohlcv_paginated(exchange, symbol: str, timeframe: str, total_bars: int) -> pd.DataFrame:
    """Fetch ``total_bars`` candles paginating backwards from now."""
    per_page = 1000
    bar_ms = exchange.parse_timeframe(timeframe) * 1000
    all_raw: list[list] = []
    end_ms = exchange.milliseconds()
    while len(all_raw) < total_bars:
        since = end_ms - per_page * bar_ms
        batch = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=per_page)
        if not batch:
            break
        all_raw = batch + all_raw
        end_ms = batch[0][0] - 1
        if len(batch) < per_page:
            break

    df = pd.DataFrame(all_raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["ts"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df.set_index("ts", inplace=True)
    df = df.drop(columns=["timestamp"])
    df = df[~df.index.duplicated(keep="first")].sort_index()
    return df.tail(total_bars)


def resample(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    out = df.resample(rule).agg({
        "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum",
    }).dropna()
    return out


# ---------------------------------------------------------------------------
# Historical kill zone (timestamp-aware replica of market_data.get_current_kill_zone)
# ---------------------------------------------------------------------------

def kill_zone_at(ts: pd.Timestamp, kill_zones: dict, instrument: dict | None) -> dict:
    hhmm = ts.strftime("%H:%M")
    allowed = instrument.get("sessions") if instrument else None
    weights = (instrument or {}).get("kill_zone_weights", {
        "london": 1.0, "new_york": 1.0, "london_ny_overlap": 1.0, "asia": 0.5,
    })
    best = None
    for name, (start, end) in kill_zones.items():
        if start <= hhmm <= end:
            if allowed and name not in allowed:
                continue
            w = weights.get(name, 0.5)
            if best is None or w > best["weight"]:
                best = {"active": True, "zone": name, "weight": w}
    return best or {"active": False, "zone": None, "weight": 0.3}


# ---------------------------------------------------------------------------
# Replay engine
# ---------------------------------------------------------------------------

class Position:
    __slots__ = ("side", "entry", "sl", "tp", "qty", "leverage", "margin",
                 "opened_at", "tp_plan", "realised_pnl", "original_qty")

    def __init__(self, signal: dict, opened_at: pd.Timestamp):
        self.side = signal["side"]
        self.entry = float(signal["entry"])
        self.sl = float(signal["sl"])
        self.tp = float(signal["tp"])
        self.leverage = int(signal.get("leverage", 1))
        self.margin = float(signal.get("margin_usd", signal["size_usd"]))
        size = float(signal["size_usd"])
        self.qty = size / self.entry
        self.original_qty = self.qty
        self.tp_plan = (signal.get("tp_plan") or {}).get("levels", [])
        self.realised_pnl = 0.0
        self.opened_at = opened_at

    def pnl_at(self, price: float, qty: float) -> float:
        if self.side == "long":
            return (price - self.entry) * qty
        return (self.entry - price) * qty

    def step(self, bar: pd.Series, ts: pd.Timestamp) -> dict | None:
        """Walk one bar forward. Returns a close event when the position
        fully exits, else None. Partial TPs are applied in place.
        """
        low, high = float(bar["low"]), float(bar["high"])

        # SL first (stops dominate) — use pessimistic touch.
        if self.side == "long" and low <= self.sl:
            return self._close(self.sl, "SL", ts)
        if self.side == "short" and high >= self.sl:
            return self._close(self.sl, "SL", ts)

        # Partial TP walk.
        for idx, lvl in enumerate(self.tp_plan):
            if lvl.get("filled"):
                continue
            lvl_price = float(lvl["price"])
            hit = (self.side == "long" and high >= lvl_price) or \
                  (self.side == "short" and low <= lvl_price)
            if not hit:
                break
            lvl["filled"] = True
            is_last = idx == len(self.tp_plan) - 1
            if is_last:
                return self._close(lvl_price, f"TP{idx+1}", ts)
            # Partial fill of original qty.
            partial_qty = self.original_qty * float(lvl["close_pct"])
            leg_pnl = self.pnl_at(lvl_price, partial_qty)
            self.qty = max(0.0, self.qty - partial_qty)
            self.realised_pnl += leg_pnl
            if lvl.get("post_action") == "breakeven":
                self.sl = self.entry

        # Legacy fallback when there's no plan.
        if not self.tp_plan:
            if (self.side == "long" and high >= self.tp) or \
               (self.side == "short" and low <= self.tp):
                return self._close(self.tp, "TP", ts)
        return None

    def _close(self, price: float, result: str, ts: pd.Timestamp) -> dict:
        leg = self.pnl_at(price, self.qty)
        total = self.realised_pnl + leg
        return {
            "side": self.side, "entry": self.entry, "exit": price,
            "sl": self.sl, "tp": self.tp, "result": result,
            "pnl": round(total, 2), "leverage": self.leverage,
            "margin": self.margin, "opened_at": self.opened_at,
            "closed_at": ts,
        }


def run_backtest(
    symbol: str,
    days: int,
    timeframe: str = "15m",
    starting_balance: float = 10_000.0,
) -> dict:
    bars_per_day = {"15m": 96, "5m": 288, "1h": 24}.get(timeframe, 96)
    # We need ~30 1D bars for MTF daily bias = ~2880 15m bars of warmup,
    # on top of the actual ``days`` of test window.
    warmup_bars = 2900
    total = days * bars_per_day + warmup_bars

    exchange = ExchangeHandler()
    print(f"Fetching {total} bars of {symbol} {timeframe}…")
    ltf_df = fetch_ohlcv_paginated(exchange.exchange, symbol, timeframe, total)
    print(f"Got {len(ltf_df)} bars spanning {ltf_df.index[0]} → {ltf_df.index[-1]}")
    if len(ltf_df) < 500:
        raise RuntimeError(f"Only got {len(ltf_df)} bars — not enough history")

    # Build higher-TF series via resample so backtest doesn't depend on
    # the exchange's historical TF availability.
    h1_full = resample(ltf_df, "1h")
    h4_full = resample(ltf_df, "4h")
    d1_full = resample(ltf_df, "1D")

    instrument = get_instrument(symbol) or {}
    mtf = MTFState()
    regime_detector = RegimeDetector()

    warmup = min(warmup_bars, len(ltf_df) - days * bars_per_day)
    warmup = max(warmup, 400)
    print(f"Warmup = {warmup} bars, test window = {len(ltf_df) - warmup} bars")
    balance = starting_balance
    equity_curve: list[tuple[pd.Timestamp, float]] = []
    trades: list[dict] = []
    position: Position | None = None
    signals_generated = 0
    signals_skipped_regime = 0

    for i in range(warmup, len(ltf_df)):
        ts = ltf_df.index[i]
        window = ltf_df.iloc[: i + 1]
        current_bar = ltf_df.iloc[i]
        current_price = float(current_bar["close"])

        # Position management first: step any open trade through THIS bar.
        if position is not None:
            event = position.step(current_bar, ts)
            if event is not None:
                balance += position.margin + event["pnl"]
                event["exit_balance"] = balance
                trades.append(event)
                position = None

        equity_curve.append((ts, balance))

        if position is not None:
            continue  # only one trade at a time in the backtester

        # Snapshot HTF dfs as of this timestamp (exclude bars from the future).
        htf_df = h4_full[h4_full.index <= ts]
        h1_df = h1_full[h1_full.index <= ts]
        d1_df = d1_full[d1_full.index <= ts]
        if len(htf_df) < 50 or len(h1_df) < 50 or len(d1_df) < 10:
            continue

        try:
            ict = analyze_ict(window, current_price)
            wyckoff = analyze_wyckoff(htf_df)
        except Exception as exc:
            logger.debug("ICT/Wyckoff failed at %s: %s", ts, exc)
            continue

        # Build a stubbed market dict — no live funding/OI/news.
        market = {
            "kill_zone": kill_zone_at(ts, Config.KILL_ZONES, instrument),
            "funding": {"rate": 0.0, "signal": "neutral"},
            "open_interest": {},
            "volume_profile": calculate_volume_profile(window.tail(200)),
            "liquidation": {"magnets": [], "asymmetry": {}},
            "manipulation": {"tracker": None},
        }

        try:
            regime = regime_detector.detect(window, news_event_active=False)
        except Exception:
            regime = None

        mtf.update("1D", d1_df)
        mtf.update("4h", htf_df)
        mtf.update("1h", h1_df)
        mtf.update("15m", window)
        confluence = mtf.confluence()

        try:
            signal = generate_signal(
                symbol, current_price, ict, wyckoff, market, balance,
                news_signal=None,
                instrument=instrument,
                regime=regime,
                order_flow=None,
                ltf_df=window,
                mtf_confluence=confluence,
            )
        except Exception as exc:
            logger.debug("generate_signal failed at %s: %s", ts, exc)
            signal = None

        if signal is None:
            signals_skipped_regime += 1
            continue

        try:
            apply_leverage_to_signal(
                signal,
                volatility_pct=(regime or {}).get("volatility_pct", 2.0),
                instrument=instrument,
                regime_name=(regime or {}).get("regime"),
                news_active=False,
                drawdown_pct=0.0,
                balance=balance,
            )
        except Exception as exc:
            logger.debug("leverage failed at %s: %s", ts, exc)

        balance -= float(signal.get("margin_usd", signal["size_usd"]))
        position = Position(signal, ts)
        signals_generated += 1

    # Force-close any still-open trade at final price.
    if position is not None:
        last_bar = ltf_df.iloc[-1]
        event = position._close(float(last_bar["close"]), "EOD", ltf_df.index[-1])
        balance += position.margin + event["pnl"]
        event["exit_balance"] = balance
        trades.append(event)

    return _summarize(symbol, days, trades, equity_curve, starting_balance, balance,
                       signals_generated, signals_skipped_regime)


def _summarize(symbol, days, trades, equity_curve, start_bal, end_bal, generated, skipped):
    wins = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] <= 0]
    total_pnl = sum(t["pnl"] for t in trades)
    win_rate = (len(wins) / len(trades) * 100) if trades else 0.0
    avg_win = (sum(t["pnl"] for t in wins) / len(wins)) if wins else 0.0
    avg_loss = (sum(t["pnl"] for t in losses) / len(losses)) if losses else 0.0
    profit_factor = (
        sum(t["pnl"] for t in wins) / abs(sum(t["pnl"] for t in losses))
        if losses and sum(t["pnl"] for t in losses) < 0 else float("inf")
    )

    # Max drawdown on equity curve.
    peak = start_bal
    max_dd = 0.0
    for _, eq in equity_curve:
        peak = max(peak, eq)
        dd = (peak - eq) / peak * 100
        max_dd = max(max_dd, dd)

    return {
        "symbol": symbol,
        "days": days,
        "starting_balance": start_bal,
        "ending_balance": round(end_bal, 2),
        "return_pct": round((end_bal - start_bal) / start_bal * 100, 2),
        "total_pnl": round(total_pnl, 2),
        "trades": len(trades),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": round(win_rate, 1),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "profit_factor": round(profit_factor, 2) if profit_factor != float("inf") else "∞",
        "max_drawdown_pct": round(max_dd, 2),
        "signals_generated": generated,
        "signals_skipped": skipped,
        "trade_log": trades,
    }


def _print_report(report: dict) -> None:
    print("=" * 60)
    print(f"BACKTEST REPORT — {report['symbol']} ({report['days']} days)")
    print("=" * 60)
    print(f"Starting balance:  ${report['starting_balance']:>10,.2f}")
    print(f"Ending balance:    ${report['ending_balance']:>10,.2f}")
    print(f"Return:             {report['return_pct']:>+10.2f}%")
    print(f"Total PnL:         ${report['total_pnl']:>+10,.2f}")
    print("-" * 60)
    print(f"Trades:            {report['trades']:>10}")
    print(f"Wins / Losses:     {report['wins']:>5} / {report['losses']:<5}")
    print(f"Win rate:          {report['win_rate']:>10}%")
    print(f"Avg win:           ${report['avg_win']:>+10,.2f}")
    print(f"Avg loss:          ${report['avg_loss']:>+10,.2f}")
    print(f"Profit factor:     {str(report['profit_factor']):>10}")
    print(f"Max drawdown:      {report['max_drawdown_pct']:>10}%")
    print("-" * 60)
    print(f"Signals generated: {report['signals_generated']}")
    print(f"Cycles skipped:    {report['signals_skipped']}")
    print("=" * 60)


def main():
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("symbol")
    parser.add_argument("days", type=int)
    parser.add_argument("--tf", default="15m")
    parser.add_argument("--balance", type=float, default=10_000.0)
    args = parser.parse_args()

    report = run_backtest(args.symbol, args.days, args.tf, args.balance)
    _print_report(report)


if __name__ == "__main__":
    main()
