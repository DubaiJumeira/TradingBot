"""Tests for Phase 8 — Telegram Command Bot."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.telegram_bot import TelegramCommandBot


class TestCommandRegistration:
    def test_register_and_lookup(self):
        bot = TelegramCommandBot()
        handler = MagicMock(return_value="ok")
        bot.register("/test", handler)
        assert "/test" in bot._handlers

    def test_help_registered_by_default(self):
        bot = TelegramCommandBot()
        assert "/help" in bot._handlers

    def test_help_lists_commands(self):
        bot = TelegramCommandBot()
        bot.register("/stats", lambda c, a: "stats")
        result = bot._cmd_help(123, "")
        assert "/stats" in result
        assert "/help" in result


class TestPauseResume:
    def test_pause_sets_flag(self):
        bot = TelegramCommandBot()
        assert bot.is_paused is False
        bot._paused = True
        assert bot.is_paused is True

    def test_resume_clears_flag(self):
        bot = TelegramCommandBot()
        bot._paused = True
        bot._paused = False
        assert bot.is_paused is False


class TestHandlerBuilders:
    def test_balance_handler(self):
        from utils.telegram_bot import build_balance_handler

        mock_bot = MagicMock()
        mock_bot.exchange.paper.balance = 9500.50
        handler = build_balance_handler(mock_bot)
        result = handler(123, "")
        assert "9,500.50" in result

    def test_positions_handler_empty(self):
        from utils.telegram_bot import build_positions_handler

        mock_bot = MagicMock()
        mock_bot.exchange.paper.positions = {}
        handler = build_positions_handler(mock_bot)
        result = handler(123, "")
        assert "No open" in result

    def test_positions_handler_with_positions(self):
        from utils.telegram_bot import build_positions_handler

        mock_bot = MagicMock()
        mock_bot.exchange.paper.positions = {
            "1": {"side": "long", "symbol": "BTC/USDT", "entry_price": 100,
                   "sl_price": 95, "tp_price": 110},
        }
        handler = build_positions_handler(mock_bot)
        result = handler(123, "")
        assert "BTC/USDT" in result
        assert "LONG" in result

    def test_risk_handler(self):
        from utils.telegram_bot import build_risk_handler

        mock_bot = MagicMock()
        mock_bot.risk_manager.status.return_value = {
            "drawdown": {
                "peak_equity": 10000, "current_equity": 9500,
                "drawdown_pct": 5.0, "breaker_active": False,
            }
        }
        handler = build_risk_handler(mock_bot)
        result = handler(123, "")
        assert "5.0%" in result
        assert "OK" in result
