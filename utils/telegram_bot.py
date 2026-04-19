"""
Interactive Telegram Command Bot — Phase 4A

Polls the Telegram Bot API for commands and dispatches to registered handlers.
Runs in a background thread. Does not use python-telegram-bot v21 — stays on
plain HTTP polling so it drops into the existing live bot without changing
the async runtime.

Commands wired via ``register_all(bot_instance)``:

    /start          — welcome + quick status
    /status         — running/paused, mode, balance, open trades, today's PnL
    /positions      — open positions with live PnL
    /history        — last 20 closed trades
    /stats          — overall performance summary (optionally per symbol)
    /balance        — current balance
    /analyze <sym>  — run full analysis on demand
    /news           — latest high-impact news
    /calendar       — upcoming economic events
    /sentiment <sym>— sentiment breakdown per source
    /liquidation <sym> — liquidation clusters (stub — Phase 1 pending)
    /whales         — whale / manipulation events (stub — Phase 2 pending)
    /regime         — current market regime per symbol
    /risk           — drawdown + circuit-breaker state
    /pause          — pause bot (run_cycle is gated on is_paused)
    /resume         — resume bot
    /close <id>     — close a position (two-step confirmation)
    /closeall       — close all positions (two-step confirmation)
    /set <param> <v>— change a runtime parameter
    /help           — list all commands

Destructive commands (/close, /closeall) use a text confirmation flow instead
of inline keyboards — user re-sends the command with CONFIRM to execute.
Phase 4B will replace this with inline keyboard callbacks.
"""

from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timezone, timedelta
from typing import Any, Callable

import requests

from config import Config

logger = logging.getLogger(__name__)

BASE_URL = f"https://api.telegram.org/bot{Config.TELEGRAM_BOT_TOKEN}"


def _kb(rows: list[list[tuple[str, str]]]) -> dict:
    """Build an inline-keyboard payload — rows of (label, callback_data)."""
    return {
        "inline_keyboard": [
            [{"text": label, "callback_data": data} for label, data in row]
            for row in rows
        ]
    }


STATUS_MENU = {
    "inline_keyboard": [
        [{"text": "💎 Positions", "callback_data": "status:positions"},
         {"text": "📊 Stats", "callback_data": "status:stats"}],
        [{"text": "📜 History", "callback_data": "status:history"},
         {"text": "📰 News", "callback_data": "status:news"}],
        [{"text": "🔥 Liquidation", "callback_data": "status:liq"},
         {"text": "🌊 Regime", "callback_data": "status:regime"}],
        [{"text": "⏸ Pause", "callback_data": "status:pause"},
         {"text": "▶️ Resume", "callback_data": "status:resume"}],
        [{"text": "🖥 Dashboard", "url": "http://212.227.88.122"}],
    ]
}

# Runtime-mutable settings keyed by /set param names → Config attribute names.
# All values are cast to float then re-cast based on the target type.
_SETTABLE_PARAMS: dict[str, tuple[str, type]] = {
    "minrr": ("MIN_RR_RATIO", float),
    "risk": ("DEFAULT_RISK_PERCENT", float),
    "maxtrades": ("MAX_OPEN_TRADES", int),
    "maxdd": ("MAX_DRAWDOWN_PCT", float),
}


def _row(label: str, value: str, width: int = 13) -> str:
    return f"  {label:<{width}}{value}"


def _divider(char: str = "─", length: int = 28) -> str:
    return char * length


class TelegramCommandBot:
    """Polls Telegram for commands and dispatches to handlers."""

    def __init__(self, poll_interval: float = 2.0) -> None:
        self._handlers: dict[str, Callable] = {}
        # Callback-query handlers keyed by prefix (the part before ":"), e.g. "close".
        self._callbacks: dict[str, Callable] = {}
        self._offset: int = 0
        self._poll_interval = poll_interval
        self._running = False
        self._thread: threading.Thread | None = None
        self._paused = False
        # Pending destructive actions keyed by chat_id: (action, args, expires_at)
        self._pending: dict[int, tuple[str, str, datetime]] = {}

        self.register("/help", self._cmd_help)

    @property
    def is_paused(self) -> bool:
        return self._paused

    def register(self, command: str, handler: Callable) -> None:
        self._handlers[command] = handler

    def register_callback(self, prefix: str, handler: Callable) -> None:
        """Register a handler for inline-button callbacks with the given prefix.

        Handler signature: (chat_id: int, data: str, message_id: int) -> str | None
        The returned string, if any, is sent as a reply message.
        """
        self._callbacks[prefix] = handler

    def start(self) -> None:
        if not Config.TELEGRAM_BOT_TOKEN or not Config.TELEGRAM_CHAT_ID:
            logger.warning("Telegram not configured — command bot disabled")
            return
        self._running = True
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()
        logger.info("Telegram command bot started (poll_interval=%.1fs)", self._poll_interval)

    def stop(self) -> None:
        self._running = False

    def _poll_loop(self) -> None:
        while self._running:
            try:
                self._process_updates()
            except Exception as exc:
                logger.warning("Telegram poll error: %s", exc)
            time.sleep(self._poll_interval)

    def _process_updates(self) -> None:
        resp = requests.get(
            f"{BASE_URL}/getUpdates",
            params={
                "offset": self._offset,
                "timeout": 10,
                "allowed_updates": (
                    '["message","edited_message","channel_post",'
                    '"edited_channel_post","callback_query"]'
                ),
            },
            timeout=15,
        )
        if not resp.ok:
            return

        data = resp.json()
        for update in data.get("result", []):
            self._offset = update["update_id"] + 1

            if "callback_query" in update:
                self._handle_callback(update["callback_query"])
                continue

            msg = (
                update.get("message")
                or update.get("channel_post")
                or update.get("edited_message")
                or update.get("edited_channel_post")
                or {}
            )
            text = msg.get("text", "")
            chat_id = msg.get("chat", {}).get("id")
            if not text or not chat_id:
                continue

            if str(chat_id) != str(Config.TELEGRAM_CHAT_ID):
                continue

            parts = text.strip().split(maxsplit=1)
            command = parts[0].lower()
            args = parts[1] if len(parts) > 1 else ""

            if command == "/pause":
                self._paused = True
                self._reply(chat_id, (
                    "⏸️ <b>Bot paused</b>\n\n"
                    "<i>New entries blocked. Active positions still monitored.</i>\n"
                    "Use /resume to continue trading."
                ))
                continue
            elif command == "/resume":
                self._paused = False
                self._reply(chat_id, (
                    "▶️ <b>Bot resumed</b>\n\n"
                    "<i>Scanning for opportunities...</i>"
                ))
                continue

            handler = self._handlers.get(command)
            if handler:
                try:
                    result = handler(chat_id, args)
                    self._send_result(chat_id, result)
                except Exception as exc:
                    logger.error("Command %s failed: %s", command, exc, exc_info=True)
                    self._reply(chat_id, f"⚠️ <b>Error</b> running <code>{command}</code>\n<pre>{exc}</pre>")
            elif text.startswith("/"):
                self._reply(chat_id, f"❓ Unknown command: <code>{command}</code>\nUse /help to see all commands.")

    def _send_result(self, chat_id: int, result: Any) -> None:
        """Handlers may return a plain string or (text, reply_markup) tuple."""
        if result is None:
            return
        if isinstance(result, tuple):
            text, markup = result
            if text:
                self._reply(chat_id, text, reply_markup=markup)
        elif isinstance(result, str):
            self._reply(chat_id, result)

    def _reply(self, chat_id: int, text: str, reply_markup: dict | None = None) -> None:
        try:
            payload: dict[str, Any] = {
                "chat_id": chat_id,
                "text": text,
                "parse_mode": "HTML",
                "disable_web_page_preview": True,
            }
            if reply_markup is not None:
                payload["reply_markup"] = reply_markup
            requests.post(f"{BASE_URL}/sendMessage", json=payload, timeout=10)
        except Exception as exc:
            logger.error("Telegram reply failed: %s", exc)

    def _answer_callback(self, callback_id: str, text: str = "") -> None:
        """Acknowledge a callback_query so the button's loading spinner stops."""
        try:
            requests.post(
                f"{BASE_URL}/answerCallbackQuery",
                json={"callback_query_id": callback_id, "text": text},
                timeout=10,
            )
        except Exception as exc:
            logger.error("Telegram answerCallback failed: %s", exc)

    def _handle_callback(self, query: dict) -> None:
        callback_id = query.get("id", "")
        data = query.get("data", "") or ""
        msg = query.get("message") or {}
        chat_id = msg.get("chat", {}).get("id")
        message_id = msg.get("message_id")
        from_id = query.get("from", {}).get("id")

        # Authorize: only the configured chat or its owner may press buttons.
        if (
            str(chat_id) != str(Config.TELEGRAM_CHAT_ID)
            and str(from_id) != str(Config.TELEGRAM_CHAT_ID)
        ):
            self._answer_callback(callback_id, "Unauthorized")
            return

        prefix = data.split(":", 1)[0]
        payload = data.split(":", 1)[1] if ":" in data else ""
        handler = self._callbacks.get(prefix)
        if handler is None:
            self._answer_callback(callback_id, "Unknown action")
            return

        try:
            result = handler(chat_id, payload, message_id)
            self._answer_callback(callback_id)
            self._send_result(chat_id, result)
        except Exception as exc:
            logger.error("Callback %s failed: %s", data, exc, exc_info=True)
            self._answer_callback(callback_id, "Error")
            self._reply(chat_id, f"⚠️ Error handling {data}: {exc}")

    # ---- Pending-confirmation bookkeeping for destructive commands ----

    def set_pending(self, chat_id: int, action: str, args: str, ttl_seconds: int = 60) -> None:
        self._pending[chat_id] = (action, args, datetime.now(timezone.utc) + timedelta(seconds=ttl_seconds))

    def take_pending(self, chat_id: int, action: str) -> str | None:
        """Consume a pending confirmation if it matches the action and hasn't expired."""
        entry = self._pending.get(chat_id)
        if entry is None:
            return None
        pending_action, args, expires_at = entry
        if pending_action != action or datetime.now(timezone.utc) > expires_at:
            self._pending.pop(chat_id, None)
            return None
        self._pending.pop(chat_id, None)
        return args

    def _cmd_help(self, chat_id: int, args: str):
        text = (
            "🔮 <b><u>Command Reference</u></b>\n"
            f"<pre>{_divider()}</pre>\n"
            "💜 <b>Status</b>\n"
            "<pre>"
            "  /start        Quick overview\n"
            "  /status       Bot state + PnL\n"
            "  /balance      Balance details\n"
            "  /positions    Open positions\n"
            "  /history      Closed trades\n"
            "  /stats [sym]  Performance\n"
            "</pre>\n"
            "🔍 <b>Analysis</b>\n"
            "<pre>"
            "  /analyze sym  Full analysis\n"
            "  /regime       Market regime\n"
            "  /news         High-impact news\n"
            "  /calendar     Economic events\n"
            "  /sentiment    Sentiment data\n"
            "  /liquidation  Liq clusters\n"
            "  /whales       Whale activity\n"
            "  /risk         Drawdown status\n"
            "  /performance  Optimizer report\n"
            "</pre>\n"
            "⚡ <b>Controls</b>\n"
            "<pre>"
            "  /pause        Stop new entries\n"
            "  /resume       Resume trading\n"
            "  /close id     Close position\n"
            "  /closeall     Close everything\n"
            "  /set p v      Change setting\n"
            "</pre>\n"
            "<i>Destructive commands require CONFIRM within 60s</i>"
        )
        return text, STATUS_MENU


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _fetch_prices_for_positions(bot_instance) -> dict[str, float]:
    """Fetch current last-price for every symbol we hold, best-effort."""
    prices: dict[str, float] = {}
    paper = getattr(bot_instance.exchange, "paper", None)
    if not paper:
        return prices
    for pos in paper.positions.values():
        sym = pos["symbol"]
        if sym in prices:
            continue
        try:
            ticker = bot_instance.exchange.fetch_ticker(sym)
            prices[sym] = float(ticker["last"])
        except Exception as exc:
            logger.debug("ticker fetch failed for %s: %s", sym, exc)
    return prices


def _compute_pnl(pos: dict, current_price: float) -> tuple[float, float]:
    """Return (pnl_usd, pnl_pct) for an open position at current_price."""
    entry = pos["entry_price"]
    qty = pos["qty"]
    if pos["side"] == "long":
        pnl = (current_price - entry) * qty
        pct = (current_price / entry - 1) * 100
    else:
        pnl = (entry - current_price) * qty
        pct = (entry / current_price - 1) * 100
    return pnl, pct


def _today_pnl(paper) -> float:
    """Sum of PnL for trades closed today (UTC)."""
    if not paper or not paper.trade_history:
        return 0.0
    today = datetime.now(timezone.utc).date().isoformat()
    return round(sum(
        t.get("pnl", 0) for t in paper.trade_history
        if t.get("closed_at", "").startswith(today)
    ), 2)


def _fmt_price(v: float) -> str:
    if v >= 1000:
        return f"${v:,.2f}"
    if v >= 1:
        return f"${v:,.4f}"
    return f"${v:,.6f}"


# ---------------------------------------------------------------------------
# Command handler builders
# ---------------------------------------------------------------------------

def build_start_handler(bot_instance) -> Callable:
    def handler(chat_id, args):
        cb = bot_instance.command_bot
        paper = bot_instance.exchange.paper
        balance = f"${paper.balance:,.2f}" if paper else "n/a"
        open_count = len(paper.positions) if paper else 0
        mode = "PAPER" if Config.PAPER_TRADING else "LIVE"
        status = "PAUSED ⏸" if cb and cb.is_paused else "RUNNING"
        status_icon = "🟡" if cb and cb.is_paused else "🟢"
        mode_icon = "📝" if Config.PAPER_TRADING else "🔴"
        symbols = " · ".join(Config.SYMBOLS)

        text = (
            f"🔮 <b><u>Trading Bot</u></b>\n"
            f"<pre>{_divider()}</pre>\n"
            f"<pre>"
            f"{_row('Status', f'{status_icon} {status}')}\n"
            f"{_row('Mode', f'{mode_icon} {mode}')}\n"
            f"{_row('Balance', balance)}\n"
            f"{_row('Open', f'{open_count} / {Config.MAX_OPEN_TRADES}')}\n"
            f"{_row('Risk', f'{Config.DEFAULT_RISK_PERCENT}% per trade')}\n"
            f"{_row('Min R:R', f'{Config.MIN_RR_RATIO}')}\n"
            f"</pre>\n"
            f"💜 <b>Symbols</b>\n"
            f"<pre>  {symbols}</pre>\n\n"
            f"<i>Use the buttons below or /help for commands</i>"
        )
        return text, STATUS_MENU
    return handler


def build_status_handler(bot_instance) -> Callable:
    def handler(chat_id, args):
        paper = bot_instance.exchange.paper
        cb = bot_instance.command_bot
        mode = "PAPER" if Config.PAPER_TRADING else "LIVE"
        is_paused = cb and cb.is_paused

        if not paper:
            return f"🔮 <b>Bot Status</b>\n\n<i>No paper trader active.</i>"

        stats = paper.get_stats()
        open_count = len(paper.positions)
        today = _today_pnl(paper)
        wr = stats.get("win_rate", 0)
        total = stats.get("total", 0)
        wins = stats.get("wins", 0)
        losses = stats.get("losses", 0)
        total_pnl = stats.get("total_pnl", 0)

        status_line = "🟡 PAUSED" if is_paused else "🟢 RUNNING"
        today_icon = "📈" if today >= 0 else "📉"
        pnl_icon = "💰" if total_pnl >= 0 else "💸"

        dd_line = ""
        breaker_line = ""
        if bot_instance.risk_manager is not None:
            dd = bot_instance.risk_manager.drawdown
            dd_line = f"{_row('Drawdown', f'{dd.drawdown_pct:.1f}%')}\n"
            if dd.is_breaker_active:
                breaker_line = "\n🚨 <b>Circuit breaker ACTIVE</b>"

        text = (
            f"🔮 <b><u>Bot Status</u></b>  ·  <i>{mode}</i>\n"
            f"<pre>{_divider()}</pre>\n"
            f"<pre>"
            f"{_row('State', status_line)}\n"
            f"{_row('Balance', f'${paper.balance:,.2f}')}\n"
            f"{_row('Open', f'{open_count} / {Config.MAX_OPEN_TRADES}')}\n"
            f"{_row(f'{today_icon} Today', f'${today:+,.2f}')}\n"
            f"{_row(f'{pnl_icon} Total', f'${total_pnl:+,.2f}')}\n"
            f"{dd_line}"
            f"{_row('Win Rate', f'{wr}%  ({wins}W · {losses}L · {total} total)')}\n"
            f"</pre>"
            f"{breaker_line}"
        )
        return text, STATUS_MENU
    return handler


def build_balance_handler(bot_instance) -> Callable:
    def handler(chat_id, args):
        paper = bot_instance.exchange.paper
        if not paper:
            return "🔮 <i>No paper trader active.</i>"
        start = Config.STARTING_BALANCE
        pnl = paper.balance - start
        pct = (pnl / start * 100) if start else 0
        pnl_icon = "📈" if pnl >= 0 else "📉"
        bar_filled = min(int(abs(pct) / 2), 14)
        bar = "█" * bar_filled + "░" * (14 - bar_filled)

        text = (
            f"💜 <b><u>Balance</u></b>\n"
            f"<pre>{_divider()}</pre>\n"
            f"<pre>"
            f"{_row('Current', f'${paper.balance:,.2f}')}\n"
            f"{_row('Starting', f'${start:,.2f}')}\n"
            f"{_row('PnL', f'${pnl:+,.2f}  ({pct:+.2f}%)')}\n"
            f"\n"
            f"  {pnl_icon} [{bar}] {pct:+.2f}%\n"
            f"</pre>"
        )
        return text
    return handler


def build_positions_handler(bot_instance) -> Callable:
    def handler(chat_id, args):
        paper = bot_instance.exchange.paper
        if not paper:
            return "🔮 <i>No paper trader active.</i>"
        if not paper.positions:
            text = (
                "💎 <b><u>Positions</u></b>\n"
                f"<pre>{_divider()}</pre>\n\n"
                "  <i>No open positions</i>\n\n"
                "  The bot is scanning for setups..."
            )
            return text, STATUS_MENU

        prices = _fetch_prices_for_positions(bot_instance)
        lines = [
            f"💎 <b><u>Open Positions</u></b>  ·  <i>{len(paper.positions)} active</i>\n"
            f"<pre>{_divider()}</pre>"
        ]
        close_buttons: list[tuple[str, str]] = []
        total_pnl = 0.0

        for tid, pos in paper.positions.items():
            close_buttons.append((f"✕ #{pos.get('id', tid)}", f"close:{pos.get('id', tid)}"))
            price = prices.get(pos["symbol"])
            side = pos["side"]
            side_icon = "🟢" if side == "long" else "🔴"
            lev = pos.get("leverage", 1)
            margin = pos.get("margin_usd", pos.get("size_usd", 0))
            liq = pos.get("liq_price", 0)
            entry = pos["entry_price"]
            qty = pos.get("qty", 0)

            if price is not None and "qty" in pos:
                pnl, pct = _compute_pnl(pos, price)
                total_pnl += pnl
                pnl_icon = "✅" if pnl > 0 else ("❌" if pnl < 0 else "⚪")
                liq_line = f"  ⚡ Liq   {_fmt_price(liq)}\n" if liq else ""

                lines.append(
                    f"\n{side_icon} <b>#{pos.get('id', tid)}  {side.upper()}  {pos['symbol']}</b>\n"
                    f"<pre>"
                    f"{_row('Entry', _fmt_price(entry))}\n"
                    f"{_row('Now', _fmt_price(price))}\n"
                    f"{_row('SL', _fmt_price(pos['sl_price']))}\n"
                    f"{_row('TP', _fmt_price(pos['tp_price']))}\n"
                    f"{_row('Leverage', f'{lev}×')}\n"
                    f"{_row('Margin', f'${margin:,.0f}')}\n"
                    f"{liq_line}"
                    f"  {pnl_icon} PnL      <b>${pnl:+,.2f}</b>  ({pct:+.2f}%)\n"
                    f"</pre>"
                )

                # Add P&L breakdown for SL and each TP level
                lines.append(f"<pre>  <b>📊 Exit P&L Breakdown</b>")

                # SL calculation
                if side == "long":
                    sl_pnl = (pos['sl_price'] - entry) * qty
                else:
                    sl_pnl = (entry - pos['sl_price']) * qty
                sl_margin_pct = (sl_pnl / margin * 100) if margin > 0 else 0
                lines.append(f"  🔴 SL    {_fmt_price(pos['sl_price']):>9}  {sl_pnl:>+8.2f}  ({sl_margin_pct:>+6.1f}%)")

                # TP levels from tp_plan
                tp_plan = pos.get("tp_plan", {})
                tp_levels = tp_plan.get("levels", [])
                for i, level in enumerate(tp_levels, 1):
                    tp_price = level.get("price", 0)
                    close_pct = level.get("close_pct", 0)
                    tp_qty = qty * close_pct

                    if side == "long":
                        tp_pnl = (tp_price - entry) * tp_qty
                    else:
                        tp_pnl = (entry - tp_price) * tp_qty

                    tp_margin_pct = (tp_pnl / margin * 100) if margin > 0 else 0
                    rr = level.get("rr", 0)
                    lines.append(f"  ✅ TP{i}   {_fmt_price(tp_price):>9}  {tp_pnl:>+8.2f}  ({tp_margin_pct:>+6.1f}%)  RR={rr:.2f}")

                lines.append("</pre>")
            else:
                lines.append(
                    f"\n{side_icon} <b>#{pos.get('id', tid)}  {side.upper()}  {pos['symbol']}</b>\n"
                    f"<pre>"
                    f"{_row('Entry', _fmt_price(pos['entry_price']))}\n"
                    f"{_row('SL', _fmt_price(pos['sl_price']))}\n"
                    f"{_row('TP', _fmt_price(pos['tp_price']))}\n"
                    f"{_row('Leverage', f'{lev}×')}\n"
                    f"{_row('Margin', f'${margin:,.0f}')}\n"
                    f"</pre>"
                )

        total_icon = "💰" if total_pnl >= 0 else "💸"
        lines.append(f"\n{total_icon} <b>Total PnL: ${total_pnl:+,.2f}</b>")

        text = "\n".join(lines)
        rows = [close_buttons[i:i + 2] for i in range(0, len(close_buttons), 2)]
        rows.append([("🔄 Refresh", "status:positions"), ("🔮 Menu", "status:menu")])
        return text, _kb(rows)
    return handler


def build_history_handler(bot_instance) -> Callable:
    def handler(chat_id, args):
        paper = bot_instance.exchange.paper
        if not paper or not paper.trade_history:
            return (
                "📜 <b><u>Trade History</u></b>\n"
                f"<pre>{_divider()}</pre>\n\n"
                "  <i>No closed trades yet</i>"
            )
        trades = paper.trade_history[-20:][::-1]
        total_pnl = sum(t.get("pnl", 0) for t in trades)
        wins = sum(1 for t in trades if t.get("pnl", 0) > 0)

        lines = [
            f"📜 <b><u>Trade History</u></b>  ·  <i>Last {len(trades)}</i>\n"
            f"<pre>{_divider()}</pre>",
            "",
        ]
        for t in trades:
            pnl = t.get("pnl", 0)
            icon = "✅" if pnl > 0 else "❌"
            side = t.get("side", "?").upper()
            side_icon = "🟢" if side == "LONG" else "🔴"
            result = t.get("result", "—")
            lines.append(
                f"  {icon} <b>#{t.get('id')}</b>  {side_icon}{side}  {t['symbol']}\n"
                f"       {result}  ·  <b>${pnl:+,.2f}</b>"
            )

        summary_icon = "💰" if total_pnl >= 0 else "💸"
        lines.append(f"\n{summary_icon} <b>Shown PnL: ${total_pnl:+,.2f}</b>  ·  {wins}W / {len(trades) - wins}L")
        return "\n".join(lines)
    return handler


def build_stats_handler(bot_instance) -> Callable:
    def handler(chat_id, args):
        paper = bot_instance.exchange.paper
        if not paper:
            return "🔮 <i>No paper trader active.</i>"
        symbol = args.strip().upper() if args else None
        history = paper.trade_history
        if symbol:
            history = [t for t in history if t.get("symbol", "").upper() == symbol]
            if not history:
                return f"📊 No closed trades for <b>{symbol}</b>."
        if not history:
            return (
                "📊 <b><u>Statistics</u></b>\n"
                f"<pre>{_divider()}</pre>\n\n"
                "  <i>No closed trades yet</i>"
            )

        wins = [t for t in history if t.get("pnl", 0) > 0]
        losses = [t for t in history if t.get("pnl", 0) <= 0]
        total_pnl = sum(t.get("pnl", 0) for t in history)
        avg_pnl = total_pnl / len(history)
        wr = len(wins) / len(history) * 100
        best = max(history, key=lambda t: t.get("pnl", 0))
        worst = min(history, key=lambda t: t.get("pnl", 0))

        title = f"📊 <b><u>Stats — {symbol}</u></b>" if symbol else "📊 <b><u>Statistics</u></b>"
        wr_bar_filled = min(int(wr / 5), 20)
        wr_bar = "█" * wr_bar_filled + "░" * (20 - wr_bar_filled)

        text = (
            f"{title}\n"
            f"<pre>{_divider()}</pre>\n"
            f"<pre>"
            f"{_row('Trades', f'{len(history)}')}\n"
            f"{_row('Wins', f'{len(wins)}  ✅')}\n"
            f"{_row('Losses', f'{len(losses)}  ❌')}\n"
            f"{_row('Win Rate', f'{wr:.1f}%')}\n"
            f"\n"
            f"  [{wr_bar}]\n"
            f"\n"
            f"{_row('Total PnL', f'${total_pnl:+,.2f}')}\n"
            f"{_row('Avg PnL', f'${avg_pnl:+,.2f}')}\n"
            f"</pre>\n"
            f"🏆 <b>Best</b>   #{best.get('id')} {best['symbol']}  <b>${best.get('pnl', 0):+,.2f}</b>\n"
            f"💀 <b>Worst</b>  #{worst.get('id')} {worst['symbol']}  <b>${worst.get('pnl', 0):+,.2f}</b>"
        )
        return text
    return handler


def build_analyze_handler(bot_instance) -> Callable:
    def handler(chat_id, args):
        symbol = args.strip().upper() if args else ""
        if not symbol:
            return "🔍 Usage: <code>/analyze BTCUSDT</code>"
        if symbol not in Config.SYMBOLS:
            return f"🔍 <b>{symbol}</b> not in watchlist.\n\nAvailable: {', '.join(Config.SYMBOLS)}"

        signal = bot_instance.analyze_symbol(symbol)
        if not signal:
            return (
                f"🔍 <b><u>Analysis — {symbol}</u></b>\n"
                f"<pre>{_divider()}</pre>\n\n"
                f"<i>No signal — score below threshold or filters failed.</i>"
            )

        side = signal['side']
        side_icon = "🟢" if side == "long" else "🔴"
        entry = float(signal['entry'])
        sl = float(signal['sl'])
        tp = float(signal['tp'])
        size_usd = signal['size_usd']

        text = (
            f"🔍 <b><u>Analysis — {symbol}</u></b>\n"
            f"<pre>{_divider()}</pre>\n"
            f"\n{side_icon} <b>{side.upper()}</b>  ·  R:R <b>{signal['rr']}</b>  ·  Score <b>{signal['score']}/100</b>\n"
            f"<pre>"
            f"{_row('Entry', _fmt_price(entry))}\n"
            f"{_row('Stop', _fmt_price(sl))}\n"
            f"{_row('Target', _fmt_price(tp))}\n"
            f"{_row('Size', f'${size_usd:,.2f}')}\n"
            f"</pre>\n"
            f"💜 <b>Context</b>\n"
            f"<pre>"
            f"{_row('ICT', signal.get('ict_structure', '—'))}\n"
            f"{_row('Wyckoff', signal.get('wyckoff_phase', '—'))}\n"
            f"{_row('Kill Zone', signal.get('kill_zone', '—'))}\n"
            f"</pre>"
        )

        chart = None
        try:
            chart = bot_instance._build_signal_chart(signal)
        except Exception as exc:
            logger.warning("/analyze chart build failed: %s", exc)

        if chart is not None:
            from utils.telegram_alerts import send_photo
            sent = send_photo(chart, caption=text)
            if sent is not None:
                return None
        return text
    return handler


def build_regime_handler(bot_instance) -> Callable:
    def handler(chat_id, args):
        detector = bot_instance.regime_detector
        if detector is None:
            return "🌊 <i>Regime detector not available.</i>"
        target = [args.strip().upper()] if args else Config.SYMBOLS
        lines = [
            f"🌊 <b><u>Market Regime</u></b>\n"
            f"<pre>{_divider()}</pre>",
            "",
        ]
        for sym in target:
            try:
                ohlcv = bot_instance.exchange.fetch_ohlcv(sym, "15m", limit=200)
                df = bot_instance.ohlcv_to_df(ohlcv)
                r = detector.detect(df)
                regime = r['regime']
                adx = r['adx']
                vol_pct = r['volatility_pct']
                regime_icon = {
                    "trending_up": "🟢", "trending_down": "🔴",
                    "ranging": "🟡", "volatile": "⚡",
                }.get(regime, "⚪")
                lines.append(
                    f"  {regime_icon} <b>{sym}</b>\n"
                    f"<pre>"
                    f"{_row('Regime', regime)}\n"
                    f"{_row('ADX', f'{adx:.1f}')}\n"
                    f"{_row('Volatility', f'{vol_pct:.2f}%')}\n"
                    f"</pre>"
                )
            except Exception as exc:
                lines.append(f"  ❌ <b>{sym}</b>: {exc}")
        return "\n".join(lines)
    return handler


def build_news_handler(bot_instance) -> Callable:
    def handler(chat_id, args):
        if bot_instance.aggregator is None:
            return "📰 <i>News engine not available.</i>"
        items = bot_instance.aggregator.high_impact(
            since=datetime.now(tz=timezone.utc) - timedelta(hours=6)
        )
        if not items:
            return (
                "📰 <b><u>News</u></b>\n"
                f"<pre>{_divider()}</pre>\n\n"
                "<i>No high-impact news in the last 6 hours.</i>\n\n"
                "Markets are quiet 🤫"
            )
        lines = [
            f"📰 <b><u>High-Impact News</u></b>  ·  <i>Last 6h</i>\n"
            f"<pre>{_divider()}</pre>",
            "",
        ]
        impact_icons = {"high": "🔴", "medium": "🟡", "low": "🟢"}
        for item in items[:10]:
            icon = impact_icons.get(item.impact_level.value.lower(), "⚪")
            lines.append(f"  {icon} {item.title[:85]}")
        return "\n".join(lines)
    return handler


def build_calendar_handler(bot_instance) -> Callable:
    def handler(chat_id, args):
        if bot_instance.calendar is None:
            return "📅 <i>Economic calendar not available.</i>"
        status = bot_instance.calendar.status()
        if not status["upcoming_48h"]:
            return (
                "📅 <b><u>Economic Calendar</u></b>\n"
                f"<pre>{_divider()}</pre>\n\n"
                "<i>No events in the next 48 hours.</i>"
            )
        lines = [
            f"📅 <b><u>Economic Calendar</u></b>  ·  <i>{status['total_events']} events</i>\n"
            f"<pre>{_divider()}</pre>",
            "",
        ]
        impact_icons = {"high": "🔴", "medium": "🟡", "low": "🟢"}
        for evt in status["upcoming_48h"]:
            mins = int(evt["minutes_until"])
            hours = mins // 60
            remaining_mins = mins % 60
            icon = impact_icons.get(evt['impact'].lower(), "⚪")
            time_str = f"{hours}h {remaining_mins}m" if hours else f"{remaining_mins}m"
            lines.append(
                f"  {icon} <b>{evt['name']}</b>\n"
                f"       <i>in {time_str}</i>"
            )
        return "\n".join(lines)
    return handler


def build_sentiment_handler(bot_instance) -> Callable:
    def handler(chat_id, args):
        if bot_instance.aggregator is None:
            return "💬 <i>News engine not available.</i>"
        symbol = args.strip().upper() if args else ""
        if not symbol:
            return "💬 Usage: <code>/sentiment BTCUSDT</code>"
        try:
            from strategies.sentiment_analyzer import SentimentAnalyzer
        except ImportError:
            return "💬 <i>Sentiment analyzer not installed.</i>"

        items = bot_instance.aggregator.recent(
            since=datetime.now(tz=timezone.utc) - timedelta(hours=12)
        )
        if not items:
            return f"💬 No recent news to score for <b>{symbol}</b>."
        analyzer = SentimentAnalyzer()
        analyzer.analyze_items(items)
        agg = analyzer.aggregate_sentiment(items, symbol)
        if not agg or agg.get("item_count", 0) == 0:
            return f"💬 No news scored for <b>{symbol}</b> in the last 12 hours."

        score = agg["weighted_score"]
        if score > 0.1:
            label, label_icon = "Bullish", "🟢"
        elif score < -0.1:
            label, label_icon = "Bearish", "🔴"
        else:
            label, label_icon = "Neutral", "⚪"

        item_count = agg["item_count"]
        velocity = agg.get("velocity", 0)
        gauge_pos = min(max(int((score + 1) * 7), 0), 14)
        gauge = "░" * gauge_pos + "█" + "░" * (14 - gauge_pos)

        text = (
            f"💬 <b><u>Sentiment — {symbol}</u></b>\n"
            f"<pre>{_divider()}</pre>\n\n"
            f"  {label_icon} <b>{label}</b>\n\n"
            f"<pre>"
            f"  🔴 [{gauge}] 🟢\n"
            f"\n"
            f"{_row('Score', f'{score:+.2f}')}\n"
            f"{_row('Items', f'{item_count}')}\n"
            f"{_row('Velocity', f'{velocity:+.2f}')}\n"
            f"</pre>"
        )
        return text
    return handler


def build_liquidation_handler(bot_instance) -> Callable:
    def handler(chat_id, args):
        symbol = args.strip().upper() if args else (Config.SYMBOLS[0] if Config.SYMBOLS else "")
        if not symbol:
            return "🔥 Usage: <code>/liquidation BTCUSDT</code>"
        if Config.SYMBOLS and symbol not in Config.SYMBOLS:
            return f"🔥 <b>{symbol}</b> not in watchlist.\n\nAvailable: {', '.join(Config.SYMBOLS)}"

        try:
            ticker = bot_instance.exchange.fetch_ticker(symbol)
            price = float(ticker["last"])
            oi = bot_instance.exchange.fetch_open_interest(symbol) or {}
            oi_value = float(oi.get("openInterestValue") or oi.get("openInterest") or 0)
        except Exception as exc:
            return f"🔥 Could not fetch data for <b>{symbol}</b>: {exc}"

        from strategies.liquidation import fetch_liquidation_clusters
        from strategies.liquidity_magnets import detect_magnets, compute_asymmetry

        clusters, source = fetch_liquidation_clusters(symbol, price, oi_value)
        magnets = detect_magnets(clusters, price)
        asym = compute_asymmetry(magnets)

        if not magnets:
            return f"🔥 No liquidation data for <b>{symbol}</b> <i>(source={source})</i>"

        above = [m for m in magnets if m.direction == "above"][:5]
        below = [m for m in magnets if m.direction == "below"][:5]
        above.sort(key=lambda m: m.distance_pct)
        below.sort(key=lambda m: m.distance_pct)

        dominant = asym.get('dominant', 'balanced')
        dom_icon = "⬆️" if dominant == "above" else ("⬇️" if dominant == "below" else "⚖️")

        lines = [
            f"🔥 <b><u>Liquidation Map — {symbol}</u></b>\n"
            f"<pre>{_divider()}</pre>",
            f"<pre>"
            f"{_row('Price', _fmt_price(price))}\n"
            f"{_row('Open Int', f'${oi_value/1e6:,.0f}M')}\n"
            f"{_row('Source', source)}\n"
            f"</pre>",
            "",
            "⬆️ <b>Shorts above</b>  <i>(upside magnets)</i>",
            "<pre>",
        ]
        for m in above:
            lines.append(
                f"  +{m.distance_pct:.2f}%  {_fmt_price(m.price_level):>12}  "
                f"${m.estimated_volume_usd/1e6:.1f}M  str={m.magnetic_strength}"
            )
        lines.append("</pre>")
        lines.append("")
        lines.append("⬇️ <b>Longs below</b>  <i>(downside magnets)</i>")
        lines.append("<pre>")
        for m in below:
            lines.append(
                f"  -{m.distance_pct:.2f}%  {_fmt_price(m.price_level):>12}  "
                f"${m.estimated_volume_usd/1e6:.1f}M  str={m.magnetic_strength}"
            )
        lines.append("</pre>")
        lines.append(
            f"\n{dom_icon} <b>Asymmetry:</b> {asym.get('ratio', 1.0):.1f}× {dominant}"
        )
        return "\n".join(lines)
    return handler


def build_whales_handler(bot_instance) -> Callable:
    def handler(chat_id, args):
        trackers = getattr(bot_instance, "manipulation_trackers", {}) or {}
        if not trackers:
            return (
                "🐋 <b><u>Whale Activity</u></b>\n"
                f"<pre>{_divider()}</pre>\n\n"
                "<i>Manipulation trackers not initialized.</i>"
            )

        lines = [
            f"🐋 <b><u>Whale Activity</u></b>  ·  <i>Last 30 min</i>\n"
            f"<pre>{_divider()}</pre>",
            "",
        ]
        total = 0
        for symbol, tracker in trackers.items():
            events = tracker.recent_events()
            cluster = tracker.detect_cluster()
            if not events:
                continue
            total += len(events)
            lines.append(f"  💜 <b>{symbol}</b>  ({len(events)} events)")
            if cluster:
                lines.append(
                    f"     🚨 <b>CLUSTER:</b> {cluster['count']} <b>{cluster['direction']}</b> events"
                )
            for e in events[-5:]:
                icon = {
                    "stop_hunt": "🎯",
                    "absorption": "🧲",
                    "spoofing": "🚨",
                    "wash_trading": "⚠️",
                }.get(e.type, "•")
                ts = e.timestamp.strftime("%H:%M")
                lines.append(f"     {icon} <code>{ts}</code> {e.description[:75]}")
            lines.append("")

        if total == 0:
            return (
                "🐋 <b><u>Whale Activity</u></b>\n"
                f"<pre>{_divider()}</pre>\n\n"
                "<i>No manipulation events detected in the last 30 minutes.</i>\n\n"
                "Markets moving on organic flow 🌊"
            )
        return "\n".join(lines)
    return handler


def build_risk_handler(bot_instance) -> Callable:
    def handler(chat_id, args):
        if bot_instance.risk_manager is None:
            return "🛡 <i>Risk manager not available.</i>"
        s = bot_instance.risk_manager.status()
        dd = s["drawdown"]
        dd_pct = dd['drawdown_pct']
        breaker = dd['breaker_active']
        peak_equity = dd['peak_equity']
        current_equity = dd['current_equity']

        health_icon = "🟢" if dd_pct < 3 else ("🟡" if dd_pct < 7 else "🔴")
        breaker_text = "🚨 ACTIVE" if breaker else "✅ OK"

        dd_bar_filled = min(int(dd_pct), 20)
        dd_bar = "█" * dd_bar_filled + "░" * (20 - dd_bar_filled)

        text = (
            f"🛡 <b><u>Risk Status</u></b>\n"
            f"<pre>{_divider()}</pre>\n"
            f"<pre>"
            f"{_row('Peak', f'${peak_equity:,.2f}')}\n"
            f"{_row('Current', f'${current_equity:,.2f}')}\n"
            f"{_row('Drawdown', f'{dd_pct:.1f}%  {health_icon}')}\n"
            f"\n"
            f"  [{dd_bar}] {dd_pct:.1f}%\n"
            f"\n"
            f"{_row('Breaker', breaker_text)}\n"
            f"{_row('Max DD', f'{Config.MAX_DRAWDOWN_PCT}%')}\n"
            f"</pre>"
        )
        return text
    return handler


def build_performance_handler(bot_instance) -> Callable:
    def handler(chat_id, args):
        if bot_instance.db is None:
            return "🧠 <i>Database not available — no performance history.</i>"
        try:
            from strategies.self_optimizer import PerformanceAnalyzer
            pa = PerformanceAnalyzer(bot_instance.db)
            report = pa.full_report(lookback_days=None)  # All historical trades
        except Exception as exc:
            return f"🧠 Performance report failed: {exc}"

        overall = report["overall"]
        if overall["n"] == 0:
            return (
                "🧠 <b><u>Self-Optimizer</u></b>\n"
                f"<pre>{_divider()}</pre>\n\n"
                "<i>No closed trades yet — optimizer in learn-mode.</i>\n"
                "Weights adjust after 10+ closed trades."
            )

        wr = overall['win_rate']
        n_trades = overall['n']
        o_total_pnl = overall['total_pnl']
        o_expectancy = overall['expectancy']
        lookback = report.get('lookback_days')
        lookback_label = f"{lookback} days" if lookback else "All history"
        lines = [
            f"🧠 <b><u>Performance Report</u></b>  ·  <i>{lookback_label}</i>\n"
            f"<pre>{_divider()}</pre>",
            "",
            f"<pre>"
            f"{_row('Trades', f'{n_trades}')}\n"
            f"{_row('Win Rate', f'{wr}%')}\n"
            f"{_row('Total PnL', f'${o_total_pnl:+,.2f}')}\n"
            f"{_row('Expectancy', f'${o_expectancy:+,.2f} / trade')}\n"
            f"</pre>",
            "",
        ]

        tags = [
            (k, v) for k, v in report["by_tag"].items() if v["n"] >= 3
        ]
        tags.sort(key=lambda kv: kv[1]["expectancy"], reverse=True)
        if tags:
            lines.append("🏷 <b>By Signal Tag</b>")
            lines.append("<pre>")
            for tag, s in tags[:8]:
                emoji = "✅" if s["expectancy"] > 0 else "❌"
                lines.append(
                    f"  {emoji} {tag:<12} {s['n']:>3}× "
                    f"WR {s['win_rate']:>5}%  ${s['expectancy']:+.1f}"
                )
            lines.append("</pre>")
            lines.append("")

        regimes = [(k, v) for k, v in report["by_regime"].items() if v["n"] >= 3]
        if regimes:
            lines.append("🌡 <b>By Regime</b>")
            lines.append("<pre>")
            for reg, s in sorted(regimes, key=lambda kv: kv[1]["expectancy"], reverse=True):
                emoji = "✅" if s["expectancy"] > 0 else "❌"
                lines.append(
                    f"  {emoji} {reg:<12} {s['n']:>3}× "
                    f"WR {s['win_rate']:>5}%  ${s['expectancy']:+.1f}"
                )
            lines.append("</pre>")
            lines.append("")

        symbols = [(k, v) for k, v in report["by_symbol"].items() if v["n"] >= 3]
        if symbols:
            lines.append("💜 <b>By Symbol</b>")
            lines.append("<pre>")
            for sym, s in sorted(symbols, key=lambda kv: kv[1]["expectancy"], reverse=True):
                emoji = "✅" if s["expectancy"] > 0 else "❌"
                lines.append(
                    f"  {emoji} {sym:<12} {s['n']:>3}× "
                    f"WR {s['win_rate']:>5}%  ${s['expectancy']:+.1f}"
                )
            lines.append("</pre>")
            lines.append("")

        try:
            from strategies.self_optimizer import load_weights, WEIGHTS_PATH
            w = load_weights()
            if w.get("sample_size", 0) >= 10:
                lines.append(
                    f"<i>⚡ Active weights: n={w['sample_size']}, "
                    f"updated {w.get('updated_at', 'never')[:19]}</i>"
                )
            else:
                lines.append(
                    f"<i>Optimizer neutral — need ≥10 trades (have {overall['n']})</i>"
                )
        except Exception:
            pass

        return "\n".join(lines)
    return handler


def build_close_handler(bot_instance) -> Callable:
    def handler(chat_id, args):
        cb = bot_instance.command_bot
        paper = bot_instance.exchange.paper
        if not paper:
            return "🔮 <i>No paper trader active.</i>"
        parts = args.strip().split()
        if not parts:
            return "✕ Usage: <code>/close &lt;id&gt;</code>"
        trade_id = parts[0]
        confirm = len(parts) > 1 and parts[1].upper() == "CONFIRM"

        pos = paper.positions.get(str(trade_id))
        if pos is None:
            return f"✕ No open position with id <b>#{trade_id}</b>."

        if not confirm:
            cb.set_pending(chat_id, "close", trade_id)
            side_icon = "🟢" if pos['side'] == "long" else "🔴"
            return (
                f"⚠️ <b><u>Confirm Close</u></b>\n"
                f"<pre>{_divider()}</pre>\n\n"
                f"  {side_icon} <b>#{trade_id}</b>  {pos['side'].upper()}  {pos['symbol']}\n"
                f"  Entry: {_fmt_price(pos['entry_price'])}\n\n"
                f"<i>Reply</i> <code>/close {trade_id} CONFIRM</code> <i>within 60s</i>"
            )

        if cb.take_pending(chat_id, "close") is None:
            return "⏰ <i>Confirmation expired. Re-send /close to restart.</i>"

        try:
            ticker = bot_instance.exchange.fetch_ticker(pos["symbol"])
            price = float(ticker["last"])
        except Exception as exc:
            return f"❌ Could not fetch price: {exc}"

        closed = paper.close_manual(trade_id, price)
        if closed is None:
            return "❌ <i>Close failed (position vanished).</i>"
        try:
            from utils.telegram_alerts import alert_close
            alert_close(closed)
        except Exception:
            pass
        pnl = closed['pnl']
        pnl_icon = "💰" if pnl > 0 else "💸"
        return (
            f"{pnl_icon} <b>Closed #{trade_id}</b> at {_fmt_price(price)}\n"
            f"PnL: <b>${pnl:+,.2f}</b>"
        )
    return handler


def build_closeall_handler(bot_instance) -> Callable:
    def handler(chat_id, args):
        cb = bot_instance.command_bot
        paper = bot_instance.exchange.paper
        if not paper:
            return "🔮 <i>No paper trader active.</i>"
        if not paper.positions:
            return "💎 <i>No open positions.</i>"

        confirm = args.strip().upper() == "CONFIRM CONFIRM" or args.strip().upper() == "CONFIRM"
        if not confirm:
            cb.set_pending(chat_id, "closeall", "")
            return (
                f"⚠️ <b><u>Close All Positions</u></b>\n"
                f"<pre>{_divider()}</pre>\n\n"
                f"  This will close <b>{len(paper.positions)}</b> position(s).\n\n"
                f"<i>Reply</i> <code>/closeall CONFIRM</code> <i>within 60s</i>"
            )
        if cb.take_pending(chat_id, "closeall") is None:
            return "⏰ <i>Confirmation expired. Re-send /closeall to restart.</i>"

        results = []
        total_pnl = 0.0
        from utils.telegram_alerts import alert_close
        for tid in list(paper.positions.keys()):
            pos = paper.positions[tid]
            try:
                price = float(bot_instance.exchange.fetch_ticker(pos["symbol"])["last"])
            except Exception as exc:
                results.append(f"  ❌ #{tid}: price fetch failed")
                continue
            closed = paper.close_manual(tid, price)
            if closed:
                try:
                    alert_close(closed)
                except Exception:
                    pass
                pnl = closed['pnl']
                total_pnl += pnl
                icon = "✅" if pnl > 0 else "❌"
                results.append(f"  {icon} <b>#{tid}</b> {closed['symbol']}  <b>${pnl:+,.2f}</b>")

        total_icon = "💰" if total_pnl >= 0 else "💸"
        text = (
            f"🧹 <b><u>Close All — Complete</u></b>\n"
            f"<pre>{_divider()}</pre>\n\n"
            + "\n".join(results)
            + f"\n\n{total_icon} <b>Total PnL: ${total_pnl:+,.2f}</b>"
        )
        return text
    return handler


def build_set_handler(bot_instance) -> Callable:
    def handler(chat_id, args):
        parts = args.strip().split()
        if len(parts) < 2:
            lines = [
                f"⚙️ <b><u>Settings</u></b>\n"
                f"<pre>{_divider()}</pre>\n"
                f"<pre>",
            ]
            for key, (attr, _) in _SETTABLE_PARAMS.items():
                lines.append(f"{_row(key, str(getattr(Config, attr)))}")
            lines.append("</pre>")
            lines.append(f"\n<i>Usage:</i> <code>/set &lt;param&gt; &lt;value&gt;</code>")
            return "\n".join(lines)
        key, value = parts[0].lower(), parts[1]
        if key not in _SETTABLE_PARAMS:
            return f"⚙️ Unknown param '<b>{key}</b>'.\n\nValid: {', '.join(_SETTABLE_PARAMS)}"
        attr, caster = _SETTABLE_PARAMS[key]
        try:
            new_val = caster(value)
        except ValueError:
            return f"⚙️ Invalid value '<b>{value}</b>' for {key}."
        old_val = getattr(Config, attr)
        setattr(Config, attr, new_val)
        logger.info("RUNTIME SET: %s %s → %s", attr, old_val, new_val)
        return (
            f"⚙️ <b>Setting Updated</b>\n\n"
            f"  <b>{key}</b>: {old_val} → <b>{new_val}</b> ✅"
        )
    return handler


# ---------------------------------------------------------------------------
# Bulk registration
# ---------------------------------------------------------------------------

def register_all(bot_instance) -> None:
    """Register every Phase 4A command and Phase 4B callbacks."""
    cb: TelegramCommandBot = bot_instance.command_bot
    cb.register("/start", build_start_handler(bot_instance))
    cb.register("/status", build_status_handler(bot_instance))
    cb.register("/balance", build_balance_handler(bot_instance))
    cb.register("/positions", build_positions_handler(bot_instance))
    cb.register("/history", build_history_handler(bot_instance))
    cb.register("/stats", build_stats_handler(bot_instance))
    cb.register("/analyze", build_analyze_handler(bot_instance))
    cb.register("/regime", build_regime_handler(bot_instance))
    cb.register("/news", build_news_handler(bot_instance))
    cb.register("/calendar", build_calendar_handler(bot_instance))
    cb.register("/sentiment", build_sentiment_handler(bot_instance))
    cb.register("/liquidation", build_liquidation_handler(bot_instance))
    cb.register("/whales", build_whales_handler(bot_instance))
    cb.register("/risk", build_risk_handler(bot_instance))
    cb.register("/performance", build_performance_handler(bot_instance))
    cb.register("/close", build_close_handler(bot_instance))
    cb.register("/closeall", build_closeall_handler(bot_instance))
    cb.register("/set", build_set_handler(bot_instance))

    # ---- Phase 4B: inline-button callback routes ----

    def _invoke(command: str, chat_id: int, args: str = ""):
        """Dispatch a callback to an existing command handler."""
        handler = cb._handlers.get(command)
        if handler is None:
            return f"Command {command} not available."
        return handler(chat_id, args)

    STATUS_ROUTES = {
        "menu": "/status",
        "refresh": "/status",
        "positions": "/positions",
        "stats": "/stats",
        "history": "/history",
        "news": "/news",
        "liq": "/liquidation",
        "regime": "/regime",
        "balance": "/balance",
    }

    def _status_cb(chat_id: int, payload: str, message_id: int):
        if payload == "pause":
            cb._paused = True
            return (
                "⏸️ <b>Bot paused</b>\n\n"
                "<i>New entries blocked. Active positions still monitored.</i>\n"
                "Use /resume to continue trading."
            )
        if payload == "resume":
            cb._paused = False
            return (
                "▶️ <b>Bot resumed</b>\n\n"
                "<i>Scanning for opportunities...</i>"
            )
        command = STATUS_ROUTES.get(payload)
        if command is None:
            return f"❓ Unknown action: {payload}"
        return _invoke(command, chat_id)

    def _close_cb(chat_id: int, payload: str, message_id: int):
        """Button press for /close — first press arms confirmation, a
        second press (via the confirm button) actually closes."""
        trade_id = payload.strip()
        if not trade_id:
            return "Missing trade id."
        if trade_id.startswith("confirm-"):
            real_id = trade_id.removeprefix("confirm-")
            return _invoke("/close", chat_id, f"{real_id} CONFIRM")
        paper = bot_instance.exchange.paper
        if not paper or str(trade_id) not in paper.positions:
            return f"✕ No open position with id <b>#{trade_id}</b>."
        pos = paper.positions[str(trade_id)]
        cb.set_pending(chat_id, "close", trade_id)
        side_icon = "🟢" if pos['side'] == "long" else "🔴"
        text = (
            f"⚠️ <b><u>Confirm Close</u></b>\n"
            f"<pre>{_divider()}</pre>\n\n"
            f"  {side_icon} <b>#{trade_id}</b>  {pos['side'].upper()}  {pos['symbol']}\n\n"
            f"<i>Confirm within 60s</i>"
        )
        markup = _kb([[
            ("✅ Confirm", f"close:confirm-{trade_id}"),
            ("↩️ Cancel", "status:menu"),
        ]])
        return text, markup

    def _analyze_cb(chat_id: int, payload: str, message_id: int):
        return _invoke("/analyze", chat_id, payload)

    def _be_cb(chat_id: int, payload: str, message_id: int):
        paper = bot_instance.exchange.paper
        if not paper:
            return "🔮 <i>No paper trader active.</i>"
        trade_id = payload.strip()
        pos = paper.positions.get(str(trade_id))
        if pos is None:
            return f"✕ No open position <b>#{trade_id}</b>."
        pos["sl_price"] = pos["entry_price"]
        return (
            f"🎯 <b>Break-Even Set</b>\n\n"
            f"  #{trade_id} SL → {_fmt_price(pos['entry_price'])}"
        )

    cb.register_callback("status", _status_cb)
    cb.register_callback("close", _close_cb)
    cb.register_callback("analyze", _analyze_cb)
    cb.register_callback("be", _be_cb)
