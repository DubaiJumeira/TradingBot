"""Tests for Phase 10 — Database & Dashboard."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from database.db import TradeDB
from database.dashboard import create_app


@pytest.fixture
def db(tmp_path):
    return TradeDB(db_path=tmp_path / "test.db")


@pytest.fixture
def sample_signal():
    return {
        "symbol": "BTC/USDT",
        "side": "long",
        "entry": 50000.0,
        "sl": 49000.0,
        "tp": 52000.0,
        "rr": 2.0,
        "score": 75,
        "size_usd": 1000.0,
        "regime": "trending",
        "reasons": ["BOS bullish", "FVG tap"],
        "news_triggered": False,
    }


class TestTradeDB:
    def test_init_creates_tables(self, db):
        tables = {r[0] for r in db._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}
        assert "trades" in tables
        assert "signals" in tables
        assert "equity" in tables
        assert "regimes" in tables

    def test_insert_and_fetch_trade(self, db, sample_signal):
        trade = {**sample_signal, "trade_id": "t1"}
        row_id = db.insert_trade(trade)
        assert row_id > 0

        trades = db.get_trades()
        assert len(trades) == 1
        assert trades[0]["symbol"] == "BTC/USDT"
        assert trades[0]["side"] == "long"
        assert trades[0]["score"] == 75
        assert trades[0]["regime"] == "trending"

    def test_close_trade(self, db, sample_signal):
        trade = {**sample_signal, "trade_id": "t2"}
        db.insert_trade(trade)
        db.close_trade("t2", exit_price=51500, pnl=150.0, result="TP")

        trades = db.get_trades()
        assert trades[0]["pnl"] == 150.0
        assert trades[0]["result"] == "TP"
        assert trades[0]["exit_price"] == 51500

    def test_get_open_trades(self, db, sample_signal):
        db.insert_trade({**sample_signal, "trade_id": "open1"})
        db.insert_trade({**sample_signal, "trade_id": "open2"})
        db.insert_trade({**sample_signal, "trade_id": "closed1"})
        db.close_trade("closed1", 51000, 100, "TP")

        open_trades = db.get_open_trades()
        assert len(open_trades) == 2
        ids = {t["trade_id"] for t in open_trades}
        assert ids == {"open1", "open2"}

    def test_trade_stats_empty(self, db):
        stats = db.trade_stats()
        assert stats["total"] == 0
        assert stats["win_rate"] == 0

    def test_trade_stats_with_trades(self, db, sample_signal):
        for i in range(5):
            tid = f"t{i}"
            db.insert_trade({**sample_signal, "trade_id": tid})
            pnl = 100 if i % 2 == 0 else -50  # 3 wins, 2 losses
            db.close_trade(tid, 51000, pnl, "TP" if pnl > 0 else "SL")

        stats = db.trade_stats()
        assert stats["total"] == 5
        assert stats["wins"] == 3
        assert stats["losses"] == 2
        assert stats["win_rate"] == 60.0
        assert stats["total_pnl"] == 200.0  # 3*100 - 2*50

    def test_insert_signal(self, db, sample_signal):
        db.insert_signal(sample_signal, executed=True)
        db.insert_signal(sample_signal, executed=False, skipped_reason="low score")

        signals = db.get_signals()
        assert len(signals) == 2
        assert signals[0]["skipped_reason"] == "low score"
        assert signals[1]["executed"] == 1

    def test_record_equity(self, db):
        db.record_equity(balance=10000, drawdown_pct=0)
        db.record_equity(balance=10150, drawdown_pct=0)
        db.record_equity(balance=9800, drawdown_pct=2.3)

        curve = db.get_equity_curve()
        assert len(curve) == 3
        balances = [r["balance"] for r in curve]
        assert 10000 in balances
        assert 9800 in balances

    def test_record_regime(self, db):
        db.record_regime("BTC/USDT", {
            "regime": "trending",
            "adx": 30.5,
            "wick_ratio": 0.3,
            "volatility_pct": 1.5,
        })
        db.record_regime("BTC/USDT", {
            "regime": "choppy",
            "adx": 15.0,
            "wick_ratio": 0.7,
            "volatility_pct": 2.5,
        })

        history = db.get_regime_history("BTC/USDT")
        assert len(history) == 2
        regimes = [r["regime"] for r in history]
        assert "trending" in regimes
        assert "choppy" in regimes

    def test_filter_trades_by_symbol(self, db, sample_signal):
        db.insert_trade({**sample_signal, "trade_id": "a1", "symbol": "BTC/USDT"})
        db.insert_trade({**sample_signal, "trade_id": "a2", "symbol": "ETH/USDT"})
        db.insert_trade({**sample_signal, "trade_id": "a3", "symbol": "BTC/USDT"})

        btc_trades = db.get_trades(symbol="BTC/USDT")
        assert len(btc_trades) == 2


class TestDashboard:
    def test_index_route(self, db, sample_signal):
        db.insert_trade({**sample_signal, "trade_id": "t1"})
        db.record_equity(10000)
        app = create_app(db)
        client = app.test_client()
        resp = client.get("/")
        assert resp.status_code == 200
        assert b"Trading Bot Dashboard" in resp.data

    def test_api_stats(self, db):
        app = create_app(db)
        client = app.test_client()
        resp = client.get("/api/stats")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "total" in data

    def test_api_trades(self, db, sample_signal):
        db.insert_trade({**sample_signal, "trade_id": "t1"})
        app = create_app(db)
        client = app.test_client()
        resp = client.get("/api/trades")
        assert resp.status_code == 200
        data = resp.get_json()
        assert len(data) == 1

    def test_api_equity(self, db):
        db.record_equity(10000)
        db.record_equity(10150)
        app = create_app(db)
        client = app.test_client()
        resp = client.get("/api/equity")
        assert resp.status_code == 200
        data = resp.get_json()
        assert len(data) == 2

    def test_api_regime(self, db):
        db.record_regime("BTC/USDT", {
            "regime": "trending", "adx": 30, "wick_ratio": 0.3, "volatility_pct": 1.5,
        })
        app = create_app(db)
        client = app.test_client()
        resp = client.get("/api/regime/BTC/USDT")
        assert resp.status_code == 200
        data = resp.get_json()
        assert len(data) == 1
        assert data[0]["regime"] == "trending"
