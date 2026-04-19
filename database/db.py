"""
Phase 10 — SQLite Database Layer

Stores trades, signals, equity snapshots, and regime history for the
dashboard and post-trade analysis.

Thread-safe: uses check_same_thread=False and serialises writes through
a single connection (SQLite WAL mode for concurrent reads).
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path("data/trading_bot.db")


class TradeDB:
    """
    SQLite storage for the trading bot.

    Tables:
        trades       — every opened/closed trade
        signals      — every generated signal (including skipped ones)
        equity       — periodic equity snapshots for drawdown charting
        regimes      — regime detection history per symbol
    """

    def __init__(self, db_path: str | Path = DEFAULT_DB_PATH) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._create_tables()
        logger.info("TradeDB initialized at %s", self.db_path)

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _create_tables(self) -> None:
        with self._conn:
            self._conn.executescript("""
                CREATE TABLE IF NOT EXISTS trades (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    trade_id    TEXT UNIQUE,
                    symbol      TEXT NOT NULL,
                    side        TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    sl_price    REAL,
                    tp_price    REAL,
                    size_usd    REAL,
                    score       INTEGER,
                    regime      TEXT,
                    opened_at   TEXT NOT NULL,
                    closed_at   TEXT,
                    exit_price  REAL,
                    pnl         REAL,
                    result      TEXT,
                    reasons     TEXT,
                    news_triggered INTEGER DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS signals (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol      TEXT NOT NULL,
                    side        TEXT NOT NULL,
                    score       INTEGER,
                    rr          REAL,
                    regime      TEXT,
                    executed    INTEGER DEFAULT 0,
                    skipped_reason TEXT,
                    created_at  TEXT NOT NULL,
                    details     TEXT
                );

                CREATE TABLE IF NOT EXISTS equity (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    balance     REAL NOT NULL,
                    unrealized  REAL DEFAULT 0,
                    drawdown_pct REAL DEFAULT 0,
                    recorded_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS regimes (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol      TEXT NOT NULL,
                    regime      TEXT NOT NULL,
                    adx         REAL,
                    wick_ratio  REAL,
                    volatility_pct REAL,
                    recorded_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol);
                CREATE INDEX IF NOT EXISTS idx_trades_opened ON trades(opened_at);
                CREATE INDEX IF NOT EXISTS idx_signals_created ON signals(created_at);
                CREATE INDEX IF NOT EXISTS idx_equity_recorded ON equity(recorded_at);
                CREATE INDEX IF NOT EXISTS idx_regimes_symbol ON regimes(symbol);
            """)

    # ------------------------------------------------------------------
    # Trades
    # ------------------------------------------------------------------

    def insert_trade(self, trade: dict[str, Any]) -> int:
        """Insert a new trade (on open). Returns the row ID."""
        now = datetime.now(tz=timezone.utc).isoformat()
        with self._conn:
            cur = self._conn.execute(
                """INSERT INTO trades
                   (trade_id, symbol, side, entry_price, sl_price, tp_price,
                    size_usd, score, regime, opened_at, reasons, news_triggered)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    trade.get("trade_id"),
                    trade["symbol"],
                    trade["side"],
                    trade["entry"],
                    trade.get("sl"),
                    trade.get("tp"),
                    trade.get("size_usd"),
                    trade.get("score"),
                    trade.get("regime"),
                    trade.get("opened_at", now),
                    json.dumps(trade.get("reasons", [])),
                    1 if trade.get("news_triggered") else 0,
                ),
            )
        return cur.lastrowid

    def close_trade(self, trade_id: str, exit_price: float, pnl: float, result: str) -> None:
        """Update a trade record when it closes."""
        now = datetime.now(tz=timezone.utc).isoformat()
        with self._conn:
            self._conn.execute(
                """UPDATE trades SET closed_at=?, exit_price=?, pnl=?, result=?
                   WHERE trade_id=?""",
                (now, exit_price, pnl, result, trade_id),
            )

    def get_trades(self, symbol: str | None = None, limit: int = 100) -> list[dict]:
        """Fetch recent trades, optionally filtered by symbol."""
        if symbol:
            rows = self._conn.execute(
                "SELECT * FROM trades WHERE symbol=? ORDER BY opened_at DESC LIMIT ?",
                (symbol, limit),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM trades ORDER BY opened_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

    def get_open_trades(self) -> list[dict]:
        """Get trades that haven't been closed yet."""
        rows = self._conn.execute(
            "SELECT * FROM trades WHERE closed_at IS NULL ORDER BY opened_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]

    def trade_stats(self) -> dict[str, Any]:
        """Aggregate stats from closed trades."""
        rows = self._conn.execute(
            "SELECT * FROM trades WHERE closed_at IS NOT NULL"
        ).fetchall()
        if not rows:
            return {
                "total": 0, "wins": 0, "losses": 0, "win_rate": 0,
                "total_pnl": 0, "avg_pnl": 0, "best": 0, "worst": 0,
            }
        trades = [dict(r) for r in rows]
        pnls = [t["pnl"] for t in trades if t["pnl"] is not None]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        total_pnl = sum(pnls) if pnls else 0
        return {
            "total": len(trades),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": round(len(wins) / len(trades) * 100, 1) if trades else 0,
            "total_pnl": round(total_pnl, 2),
            "avg_pnl": round(total_pnl / len(pnls), 2) if pnls else 0,
            "best": round(max(pnls), 2) if pnls else 0,
            "worst": round(min(pnls), 2) if pnls else 0,
        }

    # ------------------------------------------------------------------
    # Signals
    # ------------------------------------------------------------------

    def insert_signal(self, signal: dict[str, Any], executed: bool = True,
                      skipped_reason: str | None = None) -> int:
        now = datetime.now(tz=timezone.utc).isoformat()
        with self._conn:
            cur = self._conn.execute(
                """INSERT INTO signals
                   (symbol, side, score, rr, regime, executed, skipped_reason,
                    created_at, details)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    signal["symbol"],
                    signal["side"],
                    signal.get("score"),
                    signal.get("rr"),
                    signal.get("regime"),
                    1 if executed else 0,
                    skipped_reason,
                    now,
                    json.dumps(signal, default=str),
                ),
            )
        return cur.lastrowid

    def get_signals(self, limit: int = 50) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM signals ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Equity snapshots
    # ------------------------------------------------------------------

    def record_equity(self, balance: float, unrealized: float = 0,
                      drawdown_pct: float = 0) -> None:
        now = datetime.now(tz=timezone.utc).isoformat()
        with self._conn:
            self._conn.execute(
                "INSERT INTO equity (balance, unrealized, drawdown_pct, recorded_at) VALUES (?,?,?,?)",
                (balance, unrealized, drawdown_pct, now),
            )

    def get_equity_curve(self, limit: int = 500) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM equity ORDER BY recorded_at DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in reversed(rows)]

    # ------------------------------------------------------------------
    # Regime history
    # ------------------------------------------------------------------

    def record_regime(self, symbol: str, regime: dict[str, Any]) -> None:
        now = datetime.now(tz=timezone.utc).isoformat()
        with self._conn:
            self._conn.execute(
                """INSERT INTO regimes
                   (symbol, regime, adx, wick_ratio, volatility_pct, recorded_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    symbol,
                    regime["regime"],
                    regime.get("adx"),
                    regime.get("wick_ratio"),
                    regime.get("volatility_pct"),
                    now,
                ),
            )

    def get_regime_history(self, symbol: str, limit: int = 100) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM regimes WHERE symbol=? ORDER BY recorded_at DESC LIMIT ?",
            (symbol, limit),
        ).fetchall()
        return [dict(r) for r in reversed(rows)]

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def close(self) -> None:
        self._conn.close()
