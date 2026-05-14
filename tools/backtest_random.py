"""
Random-window backtest driver — samples non-overlapping windows at random
starting points across many months of history, feeds each window through
the production signal pipeline, and aggregates trades until the target
count is reached.

Usage:
    venv/bin/python -m tools.backtest_random --target-trades 1000
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)
np.seterr(divide="ignore", invalid="ignore")

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
from tools.backtest import (
    Position, kill_zone_at, fetch_ohlcv_paginated, resample,
)

logger = logging.getLogger("backtest_random")


def run_window(
    symbol: str,
    ltf_df: pd.DataFrame,
    h1_full: pd.DataFrame,
    h4_full: pd.DataFrame,
    d1_full: pd.DataFrame,
    start_idx: int,
    end_idx: int,
    starting_balance: float,
    instrument: dict,
) -> tuple[list[dict], float]:
    """Replay one window through the engine. Returns (trade events, end balance)."""
    mtf = MTFState()
    regime_detector = RegimeDetector()
    balance = starting_balance
    trades: list[dict] = []
    position: Position | None = None

    for i in range(start_idx, end_idx):
        ts = ltf_df.index[i]
        window = ltf_df.iloc[: i + 1]
        current_bar = ltf_df.iloc[i]
        current_price = float(current_bar["close"])

        if position is not None:
            event = position.step(current_bar, ts)
            if event is not None:
                balance += position.margin + event["pnl"]
                event["exit_balance"] = balance
                event["symbol"] = symbol
                trades.append(event)
                position = None

        if position is not None:
            continue

        htf_df = h4_full[h4_full.index <= ts]
        h1_df = h1_full[h1_full.index <= ts]
        d1_df = d1_full[d1_full.index <= ts]
        if len(htf_df) < 50 or len(h1_df) < 50 or len(d1_df) < 10:
            continue

        try:
            ict = analyze_ict(window, current_price)
            wyckoff = analyze_wyckoff(htf_df)
        except Exception:
            continue

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
                news_signal=None, instrument=instrument, regime=regime,
                order_flow=None, ltf_df=window, mtf_confluence=confluence,
            )
        except Exception:
            signal = None

        if signal is None:
            continue

        try:
            apply_leverage_to_signal(
                signal,
                volatility_pct=(regime or {}).get("volatility_pct", 2.0),
                instrument=instrument, regime_name=(regime or {}).get("regime"),
                news_active=False, drawdown_pct=0.0, balance=balance,
            )
        except Exception:
            pass

        balance -= float(signal.get("margin_usd", signal["size_usd"]))
        position = Position(signal, ts)

    if position is not None:
        last_bar = ltf_df.iloc[end_idx - 1]
        event = position._close(float(last_bar["close"]), "EOD", ltf_df.index[end_idx - 1])
        balance += position.margin + event["pnl"]
        event["exit_balance"] = balance
        event["symbol"] = symbol
        trades.append(event)

    return trades, balance


def main():
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", default="BTCUSDT,ETHUSDT,SOLUSDT")
    parser.add_argument("--target-trades", type=int, default=1000)
    parser.add_argument("--history-days", type=int, default=300)
    parser.add_argument("--window-days", type=int, default=20)
    parser.add_argument("--tf", default="15m")
    parser.add_argument("--balance", type=float, default=10_000.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    bars_per_day = {"15m": 96, "5m": 288, "1h": 24}.get(args.tf, 96)
    warmup_bars = 2900
    history_bars = args.history_days * bars_per_day + warmup_bars
    window_bars = args.window_days * bars_per_day

    exchange = ExchangeHandler()
    all_trades: list[dict] = []
    per_symbol: dict[str, list[dict]] = {s: [] for s in symbols}
    starting_balance = args.balance

    # Fetch history per symbol once, then randomly sample windows.
    symbol_data: dict[str, dict] = {}
    for sym in symbols:
        print(f"Fetching {history_bars} bars of {sym} {args.tf} (~{args.history_days}d)…")
        ltf_df = fetch_ohlcv_paginated(exchange.exchange, sym, args.tf, history_bars)
        print(f"  got {len(ltf_df)} bars: {ltf_df.index[0]} → {ltf_df.index[-1]}")
        if len(ltf_df) < warmup_bars + window_bars:
            print(f"  insufficient history for {sym}, skipping")
            continue
        symbol_data[sym] = {
            "ltf": ltf_df,
            "h1": resample(ltf_df, "1h"),
            "h4": resample(ltf_df, "4h"),
            "d1": resample(ltf_df, "1D"),
            "instrument": get_instrument(sym) or {},
        }

    # Round-robin random windows across symbols until target is hit.
    valid_symbols = [s for s in symbols if s in symbol_data]
    used_windows: dict[str, set[int]] = {s: set() for s in valid_symbols}
    window_count = 0
    while len(all_trades) < args.target_trades:
        progress = False
        for sym in valid_symbols:
            if len(all_trades) >= args.target_trades:
                break
            data = symbol_data[sym]
            ltf_df = data["ltf"]
            max_start = len(ltf_df) - window_bars
            min_start = warmup_bars
            if max_start <= min_start:
                continue
            # Pick a random start that doesn't overlap a previous window.
            for _ in range(30):
                candidate = random.randint(min_start, max_start)
                if any(abs(candidate - u) < window_bars for u in used_windows[sym]):
                    continue
                used_windows[sym].add(candidate)
                break
            else:
                continue

            start = candidate
            end = start + window_bars
            ts_start = ltf_df.index[start]
            ts_end = ltf_df.index[end - 1]
            print(f"[{sym}] window #{window_count + 1} {ts_start} → {ts_end} "
                  f"(cumulative trades: {len(all_trades)})")
            trades, _end_bal = run_window(
                sym, ltf_df, data["h1"], data["h4"], data["d1"],
                start, end, starting_balance, data["instrument"],
            )
            per_symbol[sym].extend(trades)
            all_trades.extend(trades)
            window_count += 1
            progress = True
        if not progress:
            print("No more non-overlapping windows available; stopping early.")
            break

    _report(all_trades, per_symbol, starting_balance, args)


def _report(all_trades, per_symbol, starting_balance, args):
    print()
    print("=" * 70)
    print(f"AGGREGATE RESULT — {len(all_trades)} trades across random windows")
    print(f"Seed: {args.seed} | history: {args.history_days}d | window: {args.window_days}d")
    print("=" * 70)
    if not all_trades:
        print("No trades generated.")
        return

    def _summarize(trades, label: str):
        wins = [t for t in trades if t["pnl"] > 0]
        losses = [t for t in trades if t["pnl"] <= 0]
        total_pnl = sum(t["pnl"] for t in trades)
        wr = len(wins) / len(trades) * 100
        avg_w = (sum(t["pnl"] for t in wins) / len(wins)) if wins else 0.0
        avg_l = (sum(t["pnl"] for t in losses) / len(losses)) if losses else 0.0
        gw = sum(t["pnl"] for t in wins)
        gl = abs(sum(t["pnl"] for t in losses))
        pf = gw / gl if gl > 0 else float("inf")
        # Group by result label
        by_result: dict[str, int] = {}
        for t in trades:
            by_result[t["result"]] = by_result.get(t["result"], 0) + 1
        print(f"\n--- {label} ({len(trades)} trades) ---")
        print(f"Total PnL:      ${total_pnl:+,.2f}")
        print(f"Win rate:       {wr:.1f}%   (W {len(wins)} / L {len(losses)})")
        print(f"Avg win:        ${avg_w:+,.2f}")
        print(f"Avg loss:       ${avg_l:+,.2f}")
        print(f"Profit factor:  {pf:.2f}" if pf != float("inf") else "Profit factor:  ∞")
        print(f"Exit labels:    {dict(sorted(by_result.items(), key=lambda kv: -kv[1]))}")
        # Side split
        longs = [t for t in trades if t["side"] == "long"]
        shorts = [t for t in trades if t["side"] == "short"]
        if longs and shorts:
            lwr = sum(1 for t in longs if t["pnl"] > 0) / len(longs) * 100
            swr = sum(1 for t in shorts if t["pnl"] > 0) / len(shorts) * 100
            print(f"Long  {len(longs)} trades, WR {lwr:.1f}%, PnL ${sum(t['pnl'] for t in longs):+,.2f}")
            print(f"Short {len(shorts)} trades, WR {swr:.1f}%, PnL ${sum(t['pnl'] for t in shorts):+,.2f}")

    _summarize(all_trades, "TOTAL")
    for sym, trades in per_symbol.items():
        if trades:
            _summarize(trades, sym)


if __name__ == "__main__":
    main()
