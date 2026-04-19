"""Telegram alerts — purple theme with structured monospace blocks.

All rich messages use HTML <pre> blocks for column alignment with a
consistent design language: bold+underline headers, divider lines,
_row() helpers, and tasteful emoji for message-type tagging.
"""

import json
import logging
from typing import Any

import requests

from config import Config

logger = logging.getLogger(__name__)

BASE_URL = f"https://api.telegram.org/bot{Config.TELEGRAM_BOT_TOKEN}"


def _fmt_price(v: float) -> str:
    if v >= 1000:
        return f"${v:,.2f}"
    if v >= 1:
        return f"${v:,.4f}"
    return f"${v:,.6f}"


def _row(label: str, value: str, width: int = 13) -> str:
    return f"  {label:<{width}}{value}"


def _divider(char: str = "─", length: int = 28) -> str:
    return char * length


def build_keyboard(rows: list[list[tuple[str, str]]]) -> dict:
    return {
        "inline_keyboard": [
            [{"text": label, "callback_data": data} for label, data in row]
            for row in rows
        ]
    }


def send_message(
    text: str,
    parse_mode: str = "HTML",
    reply_markup: dict | None = None,
) -> dict | None:
    if not Config.TELEGRAM_BOT_TOKEN or not Config.TELEGRAM_CHAT_ID:
        logger.warning("Telegram not configured — skipping alert")
        return None

    payload: dict[str, Any] = {
        "chat_id": Config.TELEGRAM_CHAT_ID,
        "text": text,
        "parse_mode": parse_mode,
        "disable_web_page_preview": True,
    }
    if reply_markup is not None:
        payload["reply_markup"] = reply_markup

    try:
        resp = requests.post(f"{BASE_URL}/sendMessage", json=payload, timeout=10)
        if not resp.ok:
            logger.error(f"Telegram error: {resp.text}")
            return None
        return resp.json().get("result")
    except Exception as e:
        logger.error(f"Telegram send failed: {e}")
        return None


def send_photo(
    photo_bytes: bytes,
    caption: str = "",
    parse_mode: str = "HTML",
    reply_markup: dict | None = None,
) -> dict | None:
    if not Config.TELEGRAM_BOT_TOKEN or not Config.TELEGRAM_CHAT_ID:
        logger.warning("Telegram not configured — skipping photo")
        return None

    data: dict[str, Any] = {
        "chat_id": Config.TELEGRAM_CHAT_ID,
        "caption": caption[:1024],
        "parse_mode": parse_mode,
    }
    if reply_markup is not None:
        data["reply_markup"] = json.dumps(reply_markup)

    try:
        resp = requests.post(
            f"{BASE_URL}/sendPhoto",
            data=data,
            files={"photo": ("chart.png", photo_bytes, "image/png")},
            timeout=20,
        )
        if not resp.ok:
            logger.error(f"Telegram sendPhoto error: {resp.text}")
            return None
        return resp.json().get("result")
    except Exception as e:
        logger.error(f"Telegram sendPhoto failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

def alert_startup():
    mode = "PAPER" if Config.PAPER_TRADING else "LIVE"
    mode_icon = "📝" if Config.PAPER_TRADING else "🔴"
    symbols = " · ".join(Config.SYMBOLS)

    body = (
        f"🔮 <b><u>Trading Bot Online</u></b>  ·  <i>{mode}</i>\n"
        f"<pre>{_divider()}</pre>\n"
        f"<pre>"
        f"{_row('Mode', f'{mode_icon} {mode}')}\n"
        f"{_row('Symbols', symbols)}\n"
        f"{_row('Min R:R', f'{Config.MIN_RR_RATIO}')}\n"
        f"{_row('Risk', f'{Config.DEFAULT_RISK_PERCENT}% per trade')}\n"
        f"{_row('Max Open', f'{Config.MAX_OPEN_TRADES}')}\n"
        f"</pre>\n"
        f"<i>Scanning market every 5 minutes</i>"
    )
    markup = {
        "inline_keyboard": [
            [{"text": "🖥 Dashboard", "url": "http://212.227.88.122"}],
        ]
    }
    send_message(body, reply_markup=markup)


# ---------------------------------------------------------------------------
# Signal
# ---------------------------------------------------------------------------

def format_signal_alert(signal: dict) -> str:
    side = signal["side"]
    side_icon = "🟢" if side == "long" else "🔴"
    side_label = "LONG" if side == "long" else "SHORT"

    entry = float(signal["entry"])
    sl = float(signal["sl"])
    tp = float(signal["tp"])
    sl_dist = abs(entry - sl)
    tp_dist = abs(tp - entry)

    lev = signal.get("leverage", 1)
    size_usd = signal["size_usd"]
    margin = signal.get("margin_usd", size_usd)
    liq = signal.get("liq_price", 0.0)
    risk_pct = signal.get("risk_pct", Config.DEFAULT_RISK_PERCENT)
    rr = signal["rr"]
    score = signal["score"]

    liq_line = f"{_row('Liq', _fmt_price(liq))}\n" if liq else ""

    pricing = (
        f"{_row('Entry', _fmt_price(entry))}\n"
        f"{_row('Stop', f'{_fmt_price(sl)}  ({sl_dist:,.2f})')}\n"
        f"{_row('Target', f'{_fmt_price(tp)}  ({tp_dist:,.2f})')}\n"
        f"{_row('R:R', f'{rr}  ·  Score {score}/100')}"
    )

    sizing = (
        f"{_row('Notional', f'${size_usd:,.2f}')}\n"
        f"{_row('Risk', f'{risk_pct}% of balance')}\n"
        f"{_row('Leverage', f'{lev}×')}\n"
        f"{_row('Margin', f'${margin:,.2f}')}\n"
        f"{liq_line}"
    )

    plan = signal.get("tp_plan") or {}
    exits_block = ""
    if plan.get("levels"):
        lines = []
        for idx, lvl in enumerate(plan["levels"]):
            tag = f"TP{idx+1}"
            pct = int(lvl["close_pct"] * 100)
            level_rr = lvl["rr"]
            post = lvl.get("post_action") or ""
            arrow = ""
            if post == "breakeven":
                arrow = "  → BE"
            elif post == "trail":
                arrow = "  → trail"
            price_str = _fmt_price(lvl['price'])
            lines.append(f"  {tag}  {price_str:>12}  {pct:>3}%  {level_rr:>4.1f}R{arrow}")
        exits_block = (
            f"\n\n🎯 <b>Scaled Exits</b>\n"
            f"<pre>" + "\n".join(lines) + "</pre>"
        )

    funding = signal.get("funding_rate", 0)
    context_block = (
        f"{_row('Structure', signal.get('ict_structure', '—'))}\n"
        f"{_row('Phase', signal.get('wyckoff_phase', '—'))}\n"
        f"{_row('Kill Zone', signal.get('kill_zone', '—'))}\n"
        f"{_row('Funding', f'{funding}%')}"
    )

    reasons = signal.get("reasons", [])[:8]
    reasons_text = "\n".join(f"  • {r}" for r in reasons) if reasons else "  <i>—</i>"

    return (
        f"{side_icon} <b><u>{side_label} · {signal['symbol']}</u></b>\n"
        f"<pre>{_divider()}</pre>\n"
        f"<pre>{pricing}</pre>\n"
        f"💜 <b>Sizing</b>\n<pre>{sizing}</pre>"
        f"{exits_block}\n\n"
        f"🔍 <b>Context</b>\n<pre>{context_block}</pre>\n"
        f"📋 <b>Reasons</b>\n{reasons_text}"
    )


def _signal_caption(signal: dict) -> str:
    side = signal["side"]
    side_icon = "🟢" if side == "long" else "🔴"
    side_label = "LONG" if side == "long" else "SHORT"
    lev = signal.get("leverage", 1)
    size_usd = signal["size_usd"]
    margin = signal.get("margin_usd", size_usd)
    rr = signal["rr"]
    score = signal["score"]
    return (
        f"{side_icon} <b><u>{side_label} · {signal['symbol']}</u></b>\n"
        f"<pre>"
        f"{_row('Entry', _fmt_price(float(signal['entry'])))}\n"
        f"{_row('Stop', _fmt_price(float(signal['sl'])))}\n"
        f"{_row('Target', _fmt_price(float(signal['tp'])))}\n"
        f"{_row('R:R', f'{rr}  ·  Score {score}/100')}\n"
        f"{_row('Size', f'${size_usd:,.0f}  @  {lev}×')}\n"
        f"{_row('Margin', f'${margin:,.0f}')}"
        f"</pre>"
    )


def alert_signal(signal: dict, chart_bytes: bytes | None = None):
    trade_id = signal.get("id") or signal.get("trade_id")
    symbol = signal.get("symbol", "")
    rows: list[list[tuple[str, str]]] = []
    if trade_id is not None:
        rows.append([
            ("✕ Close", f"close:{trade_id}"),
            ("🎯 BE", f"be:{trade_id}"),
        ])
    if symbol:
        rows.append([
            ("🔍 Analysis", f"analyze:{symbol}"),
            ("💎 Positions", "status:positions"),
        ])
    markup = build_keyboard(rows) if rows else None

    full_text = format_signal_alert(signal)

    if chart_bytes is None:
        send_message(full_text, reply_markup=markup)
        return

    sent = send_photo(chart_bytes, caption=_signal_caption(signal), reply_markup=markup)
    if sent is None:
        send_message(full_text, reply_markup=markup)
        return
    send_message(full_text)


# ---------------------------------------------------------------------------
# Trade status
# ---------------------------------------------------------------------------

def alert_trade_status(
    positions: list[dict],
    *,
    balance: float,
    drawdown_pct: float | None = None,
    note: str | None = None,
):
    if not positions:
        return

    lines = []
    for p in positions:
        side = (p.get("side") or "").upper()
        side_icon = "🟢" if side == "LONG" else "🔴"
        pnl = p.get("unrealised_pnl", p.get("pnl", 0)) or 0
        entry = float(p.get("entry_price", 0))
        sl = float(p.get("sl_price", 0))
        tp = float(p.get("tp_price", 0))
        lev = p.get("leverage", 1)
        margin = p.get("margin_usd", 0)
        pnl_icon = "✅" if pnl >= 0 else "❌"
        lines.append(
            f"  {side_icon} <b>#{p.get('id')}</b>  {p.get('symbol')}  {side}  {lev}×\n"
            f"     entry {_fmt_price(entry)}  sl {_fmt_price(sl)}\n"
            f"     tp {_fmt_price(tp)}  margin ${margin:,.2f}\n"
            f"     {pnl_icon} pnl <b>${pnl:+,.2f}</b>"
        )

    dd_line = ""
    if drawdown_pct is not None:
        dd_line = f"{_row('Drawdown', f'{drawdown_pct:.1f}%')}\n"
    note_line = ""
    if note:
        note_line = f"{_row('Note', note)}\n"

    header = (
        f"💎 <b><u>Trade Status</u></b>  ·  <i>{len(positions)} open</i>\n"
        f"<pre>{_divider()}</pre>\n"
        f"<pre>"
        f"{_row('Balance', f'${balance:,.2f}')}\n"
        f"{dd_line}"
        f"{note_line}"
        f"</pre>"
    )

    body_pre = "\n\n".join(lines)
    send_message(f"{header}\n{body_pre}")


# ---------------------------------------------------------------------------
# Close / partial fill
# ---------------------------------------------------------------------------

def format_close_alert(trade: dict) -> str:
    pnl = trade.get("pnl", 0) or 0
    tag_icon = "💰" if pnl > 0 else "💸"
    tag_label = "PROFIT" if pnl > 0 else "LOSS"
    result = trade.get("result", "—")
    side = (trade.get("side") or "").upper()
    side_icon = "🟢" if side == "LONG" else "🔴"

    body = (
        f"{_row('Pair', trade['symbol'])}\n"
        f"{_row('Side', f'{side_icon} {side}')}\n"
        f"{_row('Entry', _fmt_price(float(trade['entry_price'])))}\n"
        f"{_row('Exit', _fmt_price(float(trade['exit_price'])))}\n"
        f"{_row('PnL', f'${pnl:+,.2f}')}\n"
        f"{_row('Reason', result)}"
    )
    return (
        f"{tag_icon} <b><u>{tag_label} · Trade Closed</u></b>\n"
        f"<pre>{_divider()}</pre>\n"
        f"<pre>{body}</pre>"
    )


def alert_partial_fill(event: dict):
    label = event.get("label", "TP")
    side_icon = "🟢" if event.get("side") == "long" else "🔴"
    pct = int(event.get("close_pct", 0) * 100)
    pnl = event.get("pnl", 0)
    remaining = event.get("remaining_qty", 0)
    pnl_icon = "✅" if pnl >= 0 else "❌"
    body = (
        f"{_row('Pair', event.get('symbol', '—'))}\n"
        f"{_row('Exit', _fmt_price(float(event.get('exit_price', 0))))}\n"
        f"{_row('Closed', f'{pct}% of position')}\n"
        f"{_row('PnL', f'${pnl:+,.2f}  {pnl_icon}')}\n"
        f"{_row('Remaining', f'{remaining:.6f}')}"
    )
    send_message(
        f"🎯 <b><u>{label} Hit</u></b>  {side_icon}  #{event.get('id')}\n"
        f"<pre>{_divider()}</pre>\n"
        f"<pre>{body}</pre>"
    )


def alert_close(trade: dict):
    markup = build_keyboard([
        [("📊 Stats", "status:stats"), ("📜 History", "status:history")],
    ])
    send_message(format_close_alert(trade), reply_markup=markup)


# ---------------------------------------------------------------------------
# Stats / error
# ---------------------------------------------------------------------------

def format_stats(stats: dict) -> str:
    total = stats["total"]
    wins = stats.get("wins", 0)
    losses = stats.get("losses", 0)
    win_rate = stats["win_rate"]
    total_pnl = stats["total_pnl"]
    avg_pnl = stats.get("avg_pnl", 0)
    balance = stats["balance"]

    wr_bar_filled = min(int(float(win_rate) / 5), 20)
    wr_bar = "█" * wr_bar_filled + "░" * (20 - wr_bar_filled)
    pnl_icon = "📈" if total_pnl >= 0 else "📉"

    body = (
        f"{_row('Trades', f'{total}')}\n"
        f"{_row('Wins', f'{wins}  ✅')}\n"
        f"{_row('Losses', f'{losses}  ❌')}\n"
        f"{_row('Win Rate', f'{win_rate}%')}\n"
        f"\n"
        f"  [{wr_bar}]\n"
        f"\n"
        f"{_row('Total PnL', f'${total_pnl:+,.2f}  {pnl_icon}')}\n"
        f"{_row('Avg PnL', f'${avg_pnl:+,.2f}')}\n"
        f"{_row('Balance', f'${balance:,.2f}')}"
    )
    return (
        f"📊 <b><u>Performance</u></b>\n"
        f"<pre>{_divider()}</pre>\n"
        f"<pre>{body}</pre>"
    )


def alert_stats(stats: dict):
    send_message(format_stats(stats))


def alert_error(error_msg: str):
    send_message(
        f"🚨 <b><u>Bot Error</u></b>\n"
        f"<pre>{_divider()}</pre>\n"
        f"<pre>{error_msg}</pre>\n"
        f"<i>Check the logs for details.</i>"
    )
