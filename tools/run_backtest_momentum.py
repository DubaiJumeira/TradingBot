"""CLI: run the momentum breakout backtest on a given symbol over a date range.

Usage:
    python tools/run_backtest_momentum.py --symbol BTC/USDT \\
        --start 2022-01-01 --end 2025-01-01

Output:
    - Console summary table.
    - Markdown report at data/reports/momentum_{SYMBOL}_{start}_{end}.md.

Data:
    - First tries the parquet cache at data/historical/{SYMBOL}_4h.parquet.
    - If missing or coverage is insufficient, paginates ccxt to fetch the full
      range, then saves to the cache for next time.

All randomness in the engine is seeded; outputs are deterministic.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import pandas as pd

# Allow running as a script from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backtester.data import DATA_DIR, _parquet_path, load_ohlcv, save_ohlcv  # noqa: E402
from backtester.momentum_engine import MomentumBacktestEngine  # noqa: E402
from config import Config  # noqa: E402

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data loading (with pagination)
# ---------------------------------------------------------------------------

def _fetch_paginated_4h(
    symbol: str,
    start_ms: int,
    end_ms: int,
    exchange_id: str = "bybit",
    page_limit: int = 1000,
) -> pd.DataFrame:
    """Paginate ccxt fetch_ohlcv until we cover [start_ms, end_ms]."""
    import ccxt  # heavy import; do it lazily

    ex_cls = getattr(ccxt, exchange_id)
    exchange = ex_cls({"enableRateLimit": True})
    timeframe = "4h"
    bar_ms = 4 * 60 * 60 * 1000

    out: list[list] = []
    since = start_ms
    while since < end_ms:
        try:
            chunk = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=page_limit)
        except Exception as exc:
            logger.warning("fetch_ohlcv error at since=%s: %s; retrying after 2s", since, exc)
            time.sleep(2.0)
            continue
        if not chunk:
            break
        out.extend(chunk)
        last_ts = chunk[-1][0]
        if last_ts <= since:
            break
        since = last_ts + bar_ms
        # ccxt rate-limit (exchange-specific). enableRateLimit handles this.

    if not out:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    df = pd.DataFrame(out, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.drop_duplicates(subset=["timestamp"]).set_index("timestamp").sort_index()
    return df


def _ensure_data(
    symbol: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    use_cache: bool,
    exchange_id: str,
) -> pd.DataFrame:
    """Return a 4H df spanning [start, end] (inclusive), reading cache when
    possible and paginating fresh fetches when needed."""
    cache_path = _parquet_path(symbol, "4h")
    cached: pd.DataFrame | None = None
    if use_cache and cache_path.exists():
        try:
            cached = load_ohlcv(symbol, "4h")
            # Normalize index to UTC if not already tz-aware.
            if cached.index.tz is None:
                cached.index = cached.index.tz_localize("UTC")
        except Exception as exc:
            logger.warning("Cache load failed: %s", exc)
            cached = None

    have_start = cached.index.min() if cached is not None and not cached.empty else None
    have_end = cached.index.max() if cached is not None and not cached.empty else None
    needs_fetch = (
        cached is None
        or have_start is None
        or have_start > start
        or have_end < end
    )

    if needs_fetch:
        logger.info("Cache miss / insufficient — paginating %s 4h from %s to %s",
                    symbol, start.isoformat(), end.isoformat())
        # Fetch from earliest needed; add 250 bars warmup before `start`.
        warmup_ms = 250 * 4 * 60 * 60 * 1000
        fetch_start_ms = int(start.timestamp() * 1000) - warmup_ms
        fetch_end_ms = int(end.timestamp() * 1000) + 4 * 60 * 60 * 1000
        fresh = _fetch_paginated_4h(symbol, fetch_start_ms, fetch_end_ms,
                                     exchange_id=exchange_id)
        if cached is not None and not cached.empty:
            df = pd.concat([cached, fresh])
            df = df[~df.index.duplicated(keep="last")].sort_index()
        else:
            df = fresh
        if use_cache and not df.empty:
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            save_ohlcv(df, symbol, "4h")
    else:
        df = cached  # type: ignore[assignment]

    if df.empty:
        return df

    # Include 250 bars of warmup BEFORE start so the SMA-200 / ATR-median-50
    # have data to settle.
    warmup_start = start - pd.Timedelta(hours=4 * 250)
    return df[(df.index >= warmup_start) & (df.index <= end)].copy()


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _format_console(r: dict) -> str:
    lines = [
        "=" * 60,
        f"Momentum Breakout Backtest — {r['symbol']}",
        "=" * 60,
        f"  Starting balance     ${r['starting_balance']:,.2f}",
        f"  Final balance        ${r['final_balance']:,.2f}",
        f"  Total return         {r['total_return_pct']:.2f}%",
        f"  CAGR                 {r['cagr_pct']:.2f}%",
        f"  Sharpe (annualised)  {r['sharpe']:.2f}",
        f"  Max drawdown         {r['max_drawdown_pct']:.2f}%",
        f"  Win rate             {r['win_rate_pct']:.2f}%",
        f"  Profit factor        {r['profit_factor']}",
        f"  Avg win              ${r['avg_win']:.2f}",
        f"  Avg loss             ${r['avg_loss']:.2f}",
        f"  Num trades           {r['num_trades']}",
        f"  Exposure             {r['exposure_pct']:.2f}%",
        f"  Total fees           ${r['total_fees']:.2f}",
        f"  Total funding        ${r['total_funding']:.2f}",
        "=" * 60,
    ]
    if r["sharpe"] > 3 or r["win_rate_pct"] > 70:
        lines.append(
            "WARNING: Sharpe > 3 or win rate > 70% — investigate lookahead, "
            "missing costs, or other bias before trusting these numbers."
        )
    return "\n".join(lines)


def _write_markdown(r: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append(f"# Momentum Breakout Backtest — {r['symbol']}")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---|")
    lines.append(f"| Starting balance | ${r['starting_balance']:,.2f} |")
    lines.append(f"| Final balance | ${r['final_balance']:,.2f} |")
    lines.append(f"| Total return | {r['total_return_pct']:.2f}% |")
    lines.append(f"| CAGR | {r['cagr_pct']:.2f}% |")
    lines.append(f"| Sharpe (annualised, 4H) | {r['sharpe']:.2f} |")
    lines.append(f"| Max drawdown | {r['max_drawdown_pct']:.2f}% |")
    lines.append(f"| Win rate | {r['win_rate_pct']:.2f}% |")
    lines.append(f"| Profit factor | {r['profit_factor']} |")
    lines.append(f"| Avg win | ${r['avg_win']:.2f} |")
    lines.append(f"| Avg loss | ${r['avg_loss']:.2f} |")
    lines.append(f"| Num trades | {r['num_trades']} |")
    lines.append(f"| Exposure | {r['exposure_pct']:.2f}% |")
    lines.append(f"| Total fees paid | ${r['total_fees']:.2f} |")
    lines.append(f"| Total funding paid | ${r['total_funding']:.2f} |")
    lines.append("")
    if r["sharpe"] > 3 or r["win_rate_pct"] > 70:
        lines.append("> ⚠ **Sanity check failed.** Trend-following on crypto "
                     "typically shows Sharpe 0.8–1.5 and win rate 35–45%. "
                     "Sharpe > 3 or win rate > 70% usually means lookahead bias, "
                     "missing costs, or a stale indicator alignment. Investigate "
                     "before trusting these numbers.")
        lines.append("")

    lines.append("## Cost model")
    lines.append("")
    lines.append(f"- Taker fee per side: {Config.MOMENTUM_FEE_PCT:.3f}%")
    lines.append(f"- Slippage per side: {Config.MOMENTUM_SLIPPAGE_PCT:.3f}%")
    lines.append(f"- Funding cost per 8h: {Config.MOMENTUM_FUNDING_PCT_8H:.3f}% of notional")
    lines.append("")

    lines.append("## Trades")
    lines.append("")
    if not r["trades"]:
        lines.append("_No trades._")
    else:
        lines.append("| # | Entry time | Exit time | Side | Entry | Exit | Qty | Net PnL | Fees | Funding | Reason | Bars |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for i, t in enumerate(r["trades"], 1):
            lines.append(
                f"| {i} | {t['entry_time']} | {t['exit_time']} | long | "
                f"${t['entry_price']:.2f} | ${(t['exit_price'] or 0):.2f} | "
                f"{t['qty']:.6f} | ${t['net_pnl']:.2f} | "
                f"${t['fees_paid']:.2f} | ${t['funding_paid']:.2f} | "
                f"{t['exit_reason']} | {t['bars_held']} |"
            )
    path.write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Momentum breakout backtest")
    parser.add_argument("--symbol", required=True, help="e.g. BTC/USDT")
    parser.add_argument("--start", required=True, help="UTC start date, YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="UTC end date, YYYY-MM-DD")
    parser.add_argument("--balance", type=float, default=10_000.0)
    parser.add_argument("--exchange", default=Config.EXCHANGE)
    parser.add_argument("--output", default=None, help="Path to markdown report")
    parser.add_argument("--no-cache", action="store_true", help="Bypass parquet cache")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    start = pd.Timestamp(args.start, tz="UTC")
    end = pd.Timestamp(args.end, tz="UTC")
    if start >= end:
        print("ERROR: --start must be before --end", file=sys.stderr)
        return 2

    df = _ensure_data(args.symbol, start, end, use_cache=not args.no_cache,
                      exchange_id=args.exchange)
    if df.empty:
        print(f"ERROR: no data for {args.symbol} 4h in [{args.start}, {args.end}]",
              file=sys.stderr)
        return 3
    logger.info("Loaded %d 4h bars for %s (%s → %s)",
                len(df), args.symbol, df.index.min(), df.index.max())

    engine = MomentumBacktestEngine(
        symbol=args.symbol,
        balance=args.balance,
        random_seed=args.seed,
    )
    results = engine.run(df)
    print(_format_console(results))

    out_path = Path(args.output) if args.output else (
        Path("data/reports") /
        f"momentum_{args.symbol.replace('/', '_')}_{args.start}_{args.end}.md"
    )
    _write_markdown(results, out_path)
    print(f"\nReport written to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
