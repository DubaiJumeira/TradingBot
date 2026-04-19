"""Tests for Phase 7 — Backtesting Engine."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backtester.engine import BacktestEngine, BacktestTrade
from backtester.report import BacktestReport
from backtester.data import save_ohlcv, load_ohlcv


def _trending_df(n=300, start=100, trend=0.1):
    """Create trending OHLCV data."""
    dates = pd.date_range("2026-01-01", periods=n, freq="15min")
    prices = [start + i * trend + np.random.uniform(-0.5, 0.5) for i in range(n)]
    return pd.DataFrame({
        "open": prices,
        "high": [p + abs(np.random.normal(1)) for p in prices],
        "low": [p - abs(np.random.normal(1)) for p in prices],
        "close": [p + np.random.uniform(-0.3, 0.3) for p in prices],
        "volume": [1000 + np.random.randint(0, 500) for _ in range(n)],
    }, index=dates)


class TestBacktestEngine:
    def test_runs_without_error(self):
        df = _trending_df(n=200)
        engine = BacktestEngine(symbol="BTC/USDT", balance=10000)
        results = engine.run(df, min_bars=100)
        assert "total_trades" in results
        assert "equity_curve" in results
        assert len(results["equity_curve"]) > 0

    def test_results_format(self):
        df = _trending_df(n=200)
        engine = BacktestEngine(symbol="BTC/USDT", balance=10000)
        results = engine.run(df, min_bars=100)
        for key in ["total_trades", "win_rate", "profit_factor", "total_pnl",
                     "max_drawdown_pct", "sharpe_ratio", "final_balance", "trades"]:
            assert key in results, f"Missing key: {key}"

    def test_no_trades_handled(self):
        # Very short data → no signals.
        df = _trending_df(n=110, trend=0.0)
        engine = BacktestEngine(symbol="BTC/USDT", balance=10000)
        results = engine.run(df, min_bars=105)
        assert results["total_trades"] == 0
        assert results["final_balance"] == 10000

    def test_equity_curve_length(self):
        df = _trending_df(n=200)
        engine = BacktestEngine(symbol="BTC/USDT", balance=10000)
        engine.run(df, min_bars=100)
        # Equity curve should have one entry per bar after min_bars.
        assert len(engine.equity_curve) == 200 - 100


class TestBacktestTrade:
    def test_trade_dataclass(self):
        t = BacktestTrade(
            symbol="BTC/USDT", side="long", entry=100, sl=95, tp=110,
            size_usd=1000, score=70, open_bar=50,
        )
        assert t.result == "open"
        assert t.pnl == 0.0


class TestBacktestReport:
    def test_summary(self):
        results = {
            "total_trades": 10, "win_rate": 60.0, "profit_factor": 1.5,
            "total_pnl": 500.0, "max_drawdown_pct": 5.0, "sharpe_ratio": 1.2,
            "final_balance": 10500.0, "trades": [], "equity_curve": [],
        }
        report = BacktestReport(results, symbol="BTC/USDT")
        s = report.summary()
        assert "BTC/USDT" in s
        assert "60.0%" in s

    def test_json_output(self, tmp_path):
        results = {
            "total_trades": 5, "win_rate": 40.0, "profit_factor": 0.8,
            "total_pnl": -100.0, "max_drawdown_pct": 8.0, "sharpe_ratio": -0.3,
            "final_balance": 9900.0, "trades": [], "equity_curve": [10000, 9950, 9900],
        }
        report = BacktestReport(results, symbol="TEST")
        path = tmp_path / "report.json"
        report.to_json(path)
        assert path.exists()

    def test_html_output(self, tmp_path):
        results = {
            "total_trades": 2, "win_rate": 50.0, "profit_factor": 1.0,
            "total_pnl": 0.0, "max_drawdown_pct": 3.0, "sharpe_ratio": 0.0,
            "final_balance": 10000.0,
            "trades": [
                {"side": "long", "entry": 100, "exit": 105, "result": "TP",
                 "pnl": 50.0, "score": 65, "bars_held": 20, "news_triggered": False},
            ],
            "equity_curve": [10000, 10025, 10050],
        }
        report = BacktestReport(results, symbol="TEST")
        path = tmp_path / "report.html"
        report.to_html(path)
        assert path.exists()
        html = path.read_text()
        assert "Plotly" in html
        assert "Equity Curve" in html


class TestDataIO:
    def test_save_and_load(self, tmp_path, monkeypatch):
        import backtester.data as data_mod
        monkeypatch.setattr(data_mod, "DATA_DIR", tmp_path)

        df = _trending_df(n=50)
        save_ohlcv(df, "BTC/USDT", "15m")
        loaded = load_ohlcv("BTC/USDT", "15m")
        assert len(loaded) == 50

    def test_load_missing_raises(self, tmp_path, monkeypatch):
        import backtester.data as data_mod
        monkeypatch.setattr(data_mod, "DATA_DIR", tmp_path)

        with pytest.raises(FileNotFoundError):
            load_ohlcv("FAKE/PAIR", "1h")
