"""
Backtester data management — download, store, and load historical OHLCV data.

Storage format: Parquet files in data/historical/{symbol}_{timeframe}.parquet
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

DATA_DIR = Path("data/historical")


def _parquet_path(symbol: str, timeframe: str) -> Path:
    safe_symbol = symbol.replace("/", "_")
    return DATA_DIR / f"{safe_symbol}_{timeframe}.parquet"


def download_ohlcv(
    exchange,
    symbol: str,
    timeframe: str = "15m",
    since: str | None = None,
    limit: int = 5000,
) -> pd.DataFrame:
    """
    Download OHLCV data from exchange and save as parquet.

    Parameters
    ----------
    exchange : ccxt exchange instance.
    symbol : e.g. "BTC/USDT".
    timeframe : e.g. "15m", "1h", "4h", "1D".
    since : ISO date string for start, or None for latest.
    limit : max candles to fetch.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    since_ts = None
    if since:
        since_ts = int(pd.Timestamp(since).timestamp() * 1000)

    logger.info("Downloading %s %s (since=%s, limit=%d)", symbol, timeframe, since, limit)
    raw = exchange.fetch_ohlcv(symbol, timeframe, since=since_ts, limit=limit)

    df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)

    path = _parquet_path(symbol, timeframe)
    df.to_parquet(path)
    logger.info("Saved %d bars to %s", len(df), path)
    return df


def load_ohlcv(symbol: str, timeframe: str) -> pd.DataFrame:
    """Load previously downloaded OHLCV data from parquet."""
    path = _parquet_path(symbol, timeframe)
    if not path.exists():
        raise FileNotFoundError(f"No data for {symbol} {timeframe} at {path}")
    return pd.read_parquet(path)


def save_ohlcv(df: pd.DataFrame, symbol: str, timeframe: str) -> Path:
    """Save a DataFrame as parquet (for testing / manual data injection)."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = _parquet_path(symbol, timeframe)
    df.to_parquet(path)
    return path
