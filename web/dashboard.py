"""
Premium Trading Terminal — Complete analytics, responsive design.
All bot data: signals, regime, sentiment, liquidations, risk metrics.
Works on desktop, tablet, mobile.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, "/root/trading-bot")

from flask import Flask, jsonify, render_template_string
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ROOT = Path("/root/trading-bot")
PAPER_STATE = ROOT / "data" / "paper_trades.json"
SESSION_START = datetime(2026, 4, 18)


def _load_paper_state() -> dict:
    if not PAPER_STATE.exists():
        return {"balance": 0.0, "positions": {}, "trade_history": [], "trade_id": 0}
    try:
        with open(PAPER_STATE) as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load paper state: {e}")
        return {"balance": 0.0, "positions": {}, "trade_history": [], "trade_id": 0}


def _load_db():
    try:
        from database.db import TradeDB
        return TradeDB()
    except Exception as e:
        logger.warning(f"Could not load TradeDB: {e}")
        return None


@app.route("/api/stats")
def api_stats():
    state = _load_paper_state()

    try:
        from config import Config
        start_balance = float(Config.STARTING_BALANCE)
    except Exception:
        start_balance = 1000.0

    balance = state.get("balance", start_balance)
    history = state.get("trade_history", [])

    closed = [t for t in history if t.get("status") == "closed" or t.get("pnl") is not None]
    session_closed = [t for t in closed if t.get("closed_at") and
                      datetime.fromisoformat(t["closed_at"].replace("+00:00", "")) >= SESSION_START]

    open_count = len(state.get("positions", {}))
    wins = [t for t in session_closed if (t.get("pnl") or 0) > 0]
    losses = [t for t in session_closed if (t.get("pnl") or 0) < 0]
    flats = [t for t in session_closed if t.get("pnl") == 0]

    gross_win = sum(t.get("pnl", 0) for t in wins) if wins else 0
    gross_loss = abs(sum(t.get("pnl", 0) for t in losses)) if losses else 0
    session_pnl = gross_win - gross_loss

    profit_factor = round(gross_win / gross_loss, 2) if gross_loss > 0 else 0
    avg_win = round(gross_win / len(wins), 2) if wins else 0
    avg_loss = round(gross_loss / len(losses), 2) if losses else 0

    # Expectancy per trade
    expectancy = 0
    if session_closed:
        expectancy = round((avg_win * len(wins) - avg_loss * len(losses)) / len(session_closed), 2)

    # Sharpe ratio (simplified)
    pnls = [t.get("pnl", 0) for t in session_closed]
    if len(pnls) > 1 and pnls:
        import statistics
        std_dev = statistics.stdev(pnls) if len(pnls) > 1 else 0
        sharpe = round((session_pnl / max(std_dev, 0.01)) * (252**0.5) / 100, 2) if std_dev > 0 else 0
    else:
        sharpe = 0

    # Win/loss streaks
    max_win_streak = 0
    max_loss_streak = 0
    current_win_streak = 0
    current_loss_streak = 0
    for t in reversed(session_closed):
        if (t.get("pnl") or 0) > 0:
            current_win_streak += 1
            current_loss_streak = 0
            max_win_streak = max(max_win_streak, current_win_streak)
        else:
            current_loss_streak += 1
            current_win_streak = 0
            max_loss_streak = max(max_loss_streak, current_loss_streak)

    # Trade durations
    durations = []
    for t in session_closed:
        if t.get("opened_at") and t.get("closed_at"):
            try:
                opened = datetime.fromisoformat(t["opened_at"].replace("+00:00", ""))
                closed = datetime.fromisoformat(t["closed_at"].replace("+00:00", ""))
                duration_mins = int((closed - opened).total_seconds() / 60)
                durations.append(duration_mins)
            except:
                pass

    avg_duration = round(sum(durations) / len(durations), 0) if durations else 0
    max_duration = max(durations) if durations else 0
    min_duration = min(durations) if durations else 0

    # Best/worst trades
    best_trade = max(session_closed, key=lambda t: t.get("pnl", 0)) if session_closed else None
    worst_trade = min(session_closed, key=lambda t: t.get("pnl", 0)) if session_closed else None

    # Symbol breakdown
    symbol_stats = {}
    for t in session_closed:
        sym = t.get("symbol", "?")
        if sym not in symbol_stats:
            symbol_stats[sym] = {"trades": 0, "wins": 0, "pnl": 0}
        symbol_stats[sym]["trades"] += 1
        symbol_stats[sym]["pnl"] += t.get("pnl", 0)
        if (t.get("pnl") or 0) > 0:
            symbol_stats[sym]["wins"] += 1

    # Leverage stats
    avg_leverage = 0
    if session_closed:
        leverages = [t.get("leverage", 1) for t in session_closed]
        avg_leverage = round(sum(leverages) / len(leverages), 1)

    # Risk analysis
    max_single_loss = min([t.get("pnl", 0) for t in session_closed], default=0)
    max_single_win = max([t.get("pnl", 0) for t in session_closed], default=0)

    return jsonify({
        "balance": round(balance, 2),
        "start_balance": round(start_balance, 2),
        "session_pnl": round(session_pnl, 2),
        "return_pct": round((balance - start_balance) / start_balance * 100, 2) if start_balance > 0 else 0,
        "session_trades": len(session_closed),
        "open_trades": open_count,
        "wins": len(wins),
        "losses": len(losses),
        "flats": len(flats),
        "win_rate": round(len(wins) / len(session_closed) * 100, 1) if session_closed else 0,
        "profit_factor": profit_factor,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "expectancy": expectancy,
        "sharpe_ratio": sharpe,
        "max_win_streak": max_win_streak,
        "max_loss_streak": max_loss_streak,
        "best_trade_pnl": round(best_trade.get("pnl", 0), 2) if best_trade else 0,
        "worst_trade_pnl": round(worst_trade.get("pnl", 0), 2) if worst_trade else 0,
        "max_single_win": round(max_single_win, 2),
        "max_single_loss": round(max_single_loss, 2),
        "avg_duration_mins": int(avg_duration),
        "max_duration_mins": max_duration,
        "min_duration_mins": min_duration,
        "avg_leverage": avg_leverage,
        "symbol_stats": symbol_stats,
        "last_updated": datetime.utcnow().isoformat(),
    })


@app.route("/api/equity-curve")
def api_equity_curve():
    state = _load_paper_state()
    try:
        from config import Config
        start = float(Config.STARTING_BALANCE)
    except Exception:
        start = 1000.0

    history = state.get("trade_history", [])
    session_history = [t for t in history if t.get("closed_at") and
                       datetime.fromisoformat(t["closed_at"].replace("+00:00", "")) >= SESSION_START]

    data = [{"time": "Start", "balance": start, "dd": 0}]
    peak = start
    running = start

    for t in session_history:
        running += t.get("pnl", 0)
        peak = max(peak, running)
        dd = (peak - running) / peak * 100 if peak > 0 else 0
        time_str = t.get("closed_at", "")[:16] if t.get("closed_at") else "trade"
        data.append({"time": time_str, "balance": round(running, 2), "dd": round(dd, 2)})

    return jsonify(data)


@app.route("/api/trades-session")
def api_trades_session():
    state = _load_paper_state()
    history = state.get("trade_history", [])

    session_trades = [t for t in history if t.get("closed_at") and
                      datetime.fromisoformat(t["closed_at"].replace("+00:00", "")) >= SESSION_START]

    data = []
    for t in session_trades:
        try:
            opened = datetime.fromisoformat(t["opened_at"].replace("+00:00", ""))
            closed = datetime.fromisoformat(t["closed_at"].replace("+00:00", ""))
            duration_mins = int((closed - opened).total_seconds() / 60)
        except:
            duration_mins = 0

        rr = 0
        try:
            entry = t.get("entry_price", 0)
            sl = t.get("sl_price", 0)
            tp = t.get("tp_price", 0)
            if entry and sl and tp:
                risk = abs(entry - sl)
                reward = abs(tp - entry)
                rr = round(reward / risk, 2) if risk > 0 else 0
        except:
            pass

        data.append({
            "id": t.get("id"),
            "symbol": t.get("symbol"),
            "side": t.get("side"),
            "entry_price": t.get("entry_price"),
            "exit_price": t.get("exit_price"),
            "sl_price": t.get("sl_price"),
            "tp_price": t.get("tp_price"),
            "pnl": t.get("pnl"),
            "result": t.get("result"),
            "score": t.get("score", 0),
            "leverage": t.get("leverage", 1),
            "margin": t.get("margin_usd", 0),
            "rr": rr,
            "duration_mins": duration_mins,
            "opened_at": t.get("opened_at"),
            "closed_at": t.get("closed_at"),
        })

    return jsonify(data)


@app.route("/api/positions")
def api_positions():
    state = _load_paper_state()
    positions = state.get("positions", {})
    data = []

    for tid, pos in positions.items():
        current_price = pos.get("entry_price", 0)
        qty = pos.get("qty", 0)
        entry = pos.get("entry_price", 0)
        side = pos.get("side", "long")

        if side == "long":
            unrealized = (current_price - entry) * qty
        else:
            unrealized = (entry - current_price) * qty

        unrealized = round(unrealized + pos.get("realised_pnl", 0), 2)
        unrealized_pct = (unrealized / pos.get("margin_usd", 1) * 100) if pos.get("margin_usd") else 0

        sl_dist = abs(entry - pos.get("sl_price", entry))
        tp_dist = abs(pos.get("tp_price", entry) - entry)
        sl_dist_pct = (sl_dist / entry * 100) if entry > 0 else 0
        tp_dist_pct = (tp_dist / entry * 100) if entry > 0 else 0

        liq = pos.get("liq_price", 0)
        liq_distance_pct = 0
        liq_warning = False
        if liq > 0:
            if side == "long":
                liq_distance_pct = ((current_price - liq) / current_price * 100)
                liq_warning = liq_distance_pct < 20
            else:
                liq_distance_pct = ((liq - current_price) / liq * 100)
                liq_warning = liq_distance_pct < 20

        data.append({
            "id": pos.get("id"),
            "symbol": pos.get("symbol"),
            "side": side,
            "leverage": pos.get("leverage"),
            "entry_price": round(entry, 2),
            "current_price": round(current_price, 2),
            "sl_price": round(pos.get("sl_price", 0), 2),
            "tp_price": round(pos.get("tp_price", 0), 2),
            "liq_price": round(liq, 2),
            "liq_distance_pct": round(liq_distance_pct, 2),
            "margin_usd": round(pos.get("margin_usd", 0), 2),
            "size_usd": round(pos.get("size_usd", 0), 2),
            "unrealized_pnl": unrealized,
            "unrealized_pct": round(unrealized_pct, 2),
            "sl_distance_pct": round(sl_dist_pct, 2),
            "tp_distance_pct": round(tp_dist_pct, 2),
            "liq_warning": liq_warning,
            "opened_at": pos.get("opened_at"),
        })

    return jsonify(data)


@app.route("/api/signals")
def api_signals():
    db = _load_db()
    if not db:
        return jsonify([])
    try:
        signals = db.get_signals(limit=100)
        data = []
        for sig in signals:
            try:
                created = datetime.fromisoformat(sig.get("created_at", "").replace("+00:00", ""))
                if created < SESSION_START:
                    continue
            except:
                pass
            data.append({
                "id": sig.get("id"),
                "symbol": sig.get("symbol"),
                "side": sig.get("side"),
                "score": sig.get("score"),
                "rr": sig.get("rr"),
                "regime": sig.get("regime"),
                "executed": bool(sig.get("executed")),
                "skipped_reason": sig.get("skipped_reason"),
                "created_at": sig.get("created_at"),
                "details": sig.get("details"),
            })
        db.close()
        return jsonify(data)
    except Exception as e:
        logger.warning(f"Could not fetch signals: {e}")
        return jsonify([])


@app.route("/api/regimes")
def api_regimes():
    db = _load_db()
    if not db:
        return jsonify({})
    try:
        from config import get_symbols
        symbols = get_symbols()
        regimes = {}
        for sym in symbols:
            history = db.get_regime_history(sym, limit=1)
            if history:
                r = history[0]
                regimes[sym] = {
                    "regime": r.get("regime"),
                    "adx": round(r.get("adx", 0), 2),
                    "wick_ratio": round(r.get("wick_ratio", 0), 2),
                    "volatility_pct": round(r.get("volatility_pct", 0), 2),
                    "recorded_at": r.get("recorded_at"),
                }
        db.close()
        return jsonify(regimes)
    except Exception as e:
        logger.warning(f"Could not fetch regimes: {e}")
        return jsonify({})


@app.route("/api/kill-zones")
def api_kill_zones():
    try:
        from config import Config
        import pytz
        utc = pytz.utc
        now = datetime.now(utc)
        hour_min = now.strftime("%H:%M")

        active = []
        for zone_name, (start, end) in Config.KILL_ZONES.items():
            is_active = start <= hour_min < end
            active.append({
                "zone": zone_name,
                "start": start,
                "end": end,
                "active": is_active,
            })
        return jsonify(active)
    except Exception as e:
        logger.warning(f"Could not fetch kill zones: {e}")
        return jsonify([])


@app.route("/api/config-instruments")
def api_config_instruments():
    try:
        from config import INSTRUMENTS, get_symbols
        data = {}
        for sym in get_symbols():
            cfg = INSTRUMENTS.get(sym, {})
            data[sym] = {
                "type": cfg.get("type"),
                "min_rr": cfg.get("min_rr"),
                "risk_pct": cfg.get("risk_pct"),
                "fvg_gap": cfg.get("fvg_gap"),
                "funding": cfg.get("funding", False),
            }
        return jsonify(data)
    except Exception as e:
        logger.warning(f"Could not fetch instrument config: {e}")
        return jsonify({})


@app.route("/api/correlation-groups")
def api_correlation_groups():
    state = _load_paper_state()
    try:
        from config import CORRELATION_GROUPS, get_correlation_group
        positions = state.get("positions", {})

        groups = {}
        for group_name, group_cfg in CORRELATION_GROUPS.items():
            symbols = group_cfg.get("symbols", [])
            open_in_group = sum(1 for p in positions.values() if p.get("symbol") in symbols)
            groups[group_name] = {
                "symbols": symbols,
                "max_positions": group_cfg.get("max_positions"),
                "open_count": open_in_group,
                "at_limit": open_in_group >= group_cfg.get("max_positions", 1),
            }
        return jsonify(groups)
    except Exception as e:
        logger.warning(f"Could not fetch correlation groups: {e}")
        return jsonify({})


@app.route("/api/news-impact")
def api_news_impact():
    state = _load_paper_state()
    db = _load_db()
    if not db:
        return jsonify({"news_triggered_count": 0, "trades": []})
    try:
        all_trades = db.get_trades(limit=200)
        news_trades = [t for t in all_trades if t.get("news_triggered")]
        data = {
            "news_triggered_count": len(news_trades),
            "trades": [
                {
                    "id": t.get("id"),
                    "symbol": t.get("symbol"),
                    "side": t.get("side"),
                    "pnl": t.get("pnl"),
                    "opened_at": t.get("opened_at"),
                    "closed_at": t.get("closed_at"),
                }
                for t in news_trades[:20]
            ]
        }
        db.close()
        return jsonify(data)
    except Exception as e:
        logger.warning(f"Could not fetch news impact: {e}")
        return jsonify({"news_triggered_count": 0, "trades": []})


@app.route("/")
def index():
    state = _load_paper_state()
    try:
        from config import Config
        start_balance = float(Config.STARTING_BALANCE)
        mode = "PAPER" if Config.PAPER_TRADING else "LIVE"
    except Exception:
        start_balance = 1000.0
        mode = "UNKNOWN"

    balance = state.get("balance", start_balance)
    return render_template_string(TEMPLATE, balance=round(balance, 2), start_balance=round(start_balance, 2), mode=mode)


TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=5.0, user-scalable=yes" />
<title>Trading Terminal</title>
<link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><rect fill='%23000000' width='100' height='100'/><text x='50' y='70' font-size='80' fill='%2339ff14' text-anchor='middle' font-weight='bold'>⚡</text></svg>" />
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Outfit:wght@400;500;600;700&display=swap" rel="stylesheet" />
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
html, body { height: 100%; }

body {
  font-family: 'Outfit', sans-serif;
  background: #000000;
  color: #e6edf3;
  font-size: 16px;
}

.container {
  max-width: 1800px;
  margin: 0 auto;
  padding: 32px 24px;
}

.header {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  margin-bottom: 40px;
  padding-bottom: 20px;
  border-bottom: 1px solid #1a1a1a;
  flex-wrap: wrap;
  gap: 20px;
}

.header h1 {
  font-size: 40px;
  font-weight: 700;
  letter-spacing: -1.5px;
}

.header-right { text-align: right; }
.status { font-size: 13px; color: #8b949e; margin-bottom: 6px; }
.dot { display: inline-block; width: 8px; height: 8px; background: #39ff14; border-radius: 50%; margin-right: 6px; box-shadow: 0 0 8px #39ff14; }
.time { font-size: 12px; color: #8b949e; font-family: 'JetBrains Mono', monospace; }

.grid {
  display: grid;
  gap: 16px;
  margin-bottom: 20px;
}

.grid-6 { grid-template-columns: repeat(6, 1fr); }
.grid-4 { grid-template-columns: repeat(4, 1fr); }
.grid-3 { grid-template-columns: repeat(3, 1fr); }
.grid-2 { grid-template-columns: 1fr 1fr; }
.wide { grid-column: 1 / -1; }

.card {
  background: #0a0a0a;
  border: 1px solid #1a1a1a;
  border-radius: 10px;
  padding: 20px;
  transition: border-color 0.2s ease, box-shadow 0.2s ease;
  min-width: 0;
  overflow: hidden;
}

.card:hover {
  border-color: #39ff14;
  box-shadow: 0 0 20px rgba(57,255,20,.1);
}

.label {
  font-size: 10px;
  text-transform: uppercase;
  letter-spacing: 1px;
  color: #64748b;
  font-weight: 600;
  margin-bottom: 12px;
}

.value {
  font-size: 44px;
  font-weight: 700;
  font-family: 'JetBrains Mono', monospace;
  letter-spacing: -0.5px;
  margin-bottom: 8px;
  word-break: break-word;
  line-height: 1;
}

.value.gain { color: #39ff14; }
.value.loss { color: #ff4444; }
.value.neutral { color: #e2e8f0; }

.sub {
  font-size: 11px;
  color: #8b949e;
  font-family: 'JetBrains Mono', monospace;
}

.row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin: 10px 0;
  font-size: 12px;
  gap: 12px;
  padding: 6px 0;
}

.row-label {
  color: #8b949e;
  flex-shrink: 0;
  max-width: 60%;
}

.row-value {
  font-family: 'JetBrains Mono', monospace;
  font-weight: 600;
  color: #e2e8f0;
  text-align: right;
  flex-shrink: 0;
}

canvas { max-height: 350px; }

table {
  width: 100%;
  border-collapse: collapse;
  font-size: 12px;
  font-family: 'JetBrains Mono', monospace;
  margin-top: 16px;
  overflow-x: auto;
}

th {
  text-align: left;
  padding: 12px;
  color: #8b949e;
  font-weight: 600;
  border-bottom: 1px solid #1a1a1a;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  font-size: 10px;
  white-space: nowrap;
}

td {
  padding: 10px 12px;
  border-bottom: 1px solid #1a1a1a;
  white-space: nowrap;
}

tr:hover td { background: rgba(57,255,20,.05); }

.chip {
  display: inline-block;
  padding: 4px 10px;
  border-radius: 4px;
  font-size: 10px;
  font-weight: 600;
  letter-spacing: 0.5px;
}

.chip.long { background: rgba(57,255,20,.15); color: #39ff14; }
.chip.short { background: rgba(255,68,68,.15); color: #ff4444; }

.pos-card {
  background: rgba(57,255,20,.06);
  border: 1px solid rgba(57,255,20,.2);
  border-radius: 8px;
  padding: 14px;
  margin-bottom: 10px;
  min-width: 0;
}

.pos-card.danger {
  border-color: rgba(255,68,68,.3);
  background: rgba(255,68,68,.06);
}

.pos-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 10px;
  font-weight: 600;
  font-size: 13px;
  gap: 8px;
  flex-wrap: wrap;
}

.pos-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 8px;
  font-size: 11px;
}

.pos-row {
  display: flex;
  justify-content: space-between;
  color: #8b949e;
  gap: 8px;
}

.pos-val {
  font-family: 'JetBrains Mono', monospace;
  font-weight: 600;
  color: #c9d1d9;
  text-align: right;
}

.gain { color: #39ff14; }
.loss { color: #ff4444; }
.empty { color: #8b949e; font-size: 12px; }

.footer {
  text-align: center;
  color: #8b949e;
  font-size: 11px;
  margin-top: 48px;
  padding-top: 24px;
  border-top: 1px solid #1a1a1a;
}

/* Responsive Design */
@media (max-width: 1600px) {
  .grid-6 { grid-template-columns: repeat(3, 1fr); }
  .grid-4 { grid-template-columns: repeat(2, 1fr); }
}

@media (max-width: 1200px) {
  .container { padding: 24px 20px; }
  .grid-6 { grid-template-columns: repeat(2, 1fr); }
  .grid-4 { grid-template-columns: repeat(2, 1fr); }
  .grid-3 { grid-template-columns: repeat(2, 1fr); }
  .value { font-size: 38px; }
}

@media (max-width: 768px) {
  .container { padding: 20px 16px; }
  .header {
    flex-direction: column;
    align-items: flex-start;
    margin-bottom: 28px;
  }
  .header h1 { font-size: 28px; margin-bottom: 8px; }
  .header-right { width: 100%; text-align: left; }
  .value { font-size: 32px; }
  .grid { gap: 14px; margin-bottom: 16px; }
  .grid-6, .grid-4, .grid-3, .grid-2 { grid-template-columns: 1fr; }
  .card { padding: 16px; }
  .label { font-size: 9px; margin-bottom: 10px; }
  .row { font-size: 11px; margin: 8px 0; }
  th, td { padding: 10px 8px; font-size: 11px; }
  canvas { max-height: 250px; }
  .pos-grid { grid-template-columns: 1fr; }
}

@media (max-width: 480px) {
  .container { padding: 16px 12px; }
  .header h1 { font-size: 24px; }
  .value { font-size: 28px; }
  .grid { gap: 12px; margin-bottom: 12px; }
  .card { padding: 14px; border-radius: 8px; }
  .label { font-size: 9px; margin-bottom: 8px; }
  .row { font-size: 11px; margin: 6px 0; padding: 4px 0; }
  table { font-size: 10px; margin-top: 12px; }
  th, td { padding: 8px 6px; }
  .chip { padding: 3px 8px; font-size: 9px; }
  .pos-card { padding: 12px; margin-bottom: 8px; }
}
</style>
</head>
<body>

<div class="container">

<div class="header">
  <h1>⚡ TERMINAL</h1>
  <div class="header-right">
    <div class="status"><span class="dot"></span>{{ mode }}</div>
    <div class="time"><span id="time">—</span></div>
  </div>
</div>

<!-- Core KPIs -->
<div class="grid grid-6">
  <div class="card">
    <div class="label">Balance</div>
    <div class="value neutral" id="bal">${{ balance }}</div>
    <div class="sub" id="bal-pct">+0.00%</div>
  </div>

  <div class="card">
    <div class="label">Session P&L</div>
    <div class="value" id="pnl">+$0</div>
    <div class="sub" id="ret">+0%</div>
  </div>

  <div class="card">
    <div class="label">Win Rate</div>
    <div class="value" id="wr">0%</div>
    <div class="sub" id="wr-sub">0W / 0L</div>
  </div>

  <div class="card">
    <div class="label">Profit Factor</div>
    <div class="value" id="pf">0.00</div>
    <div class="sub">Win / Loss</div>
  </div>

  <div class="card">
    <div class="label">Expectancy</div>
    <div class="value" id="exp">$0</div>
    <div class="sub">Per Trade</div>
  </div>

  <div class="card">
    <div class="label">Sharpe Ratio</div>
    <div class="value" id="sharpe">0.00</div>
    <div class="sub">Risk-Adjusted</div>
  </div>
</div>

<!-- Risk Metrics Row -->
<div class="grid grid-6">
  <div class="card">
    <div class="label">Trades</div>
    <div class="value" id="trades">0</div>
    <div class="sub"><span id="open">0</span> open</div>
  </div>

  <div class="card">
    <div class="label">Avg Leverage</div>
    <div class="value" id="avg-lev">0.0×</div>
    <div class="sub">Per Trade</div>
  </div>

  <div class="card">
    <div class="label">Avg Duration</div>
    <div class="value" id="avg-dur">0h</div>
    <div class="sub" id="dur-sub">Min-Max</div>
  </div>

  <div class="card">
    <div class="label">Best Trade</div>
    <div class="value gain" id="best">+$0</div>
    <div class="sub" id="best-sym">—</div>
  </div>

  <div class="card">
    <div class="label">Worst Trade</div>
    <div class="value loss" id="worst">-$0</div>
    <div class="sub" id="worst-sym">—</div>
  </div>

  <div class="card">
    <div class="label">Streak</div>
    <div class="value gain" id="streak">0W</div>
    <div class="sub" id="streak-sub">Max Streak</div>
  </div>
</div>

<!-- Charts & Breakdown -->
<div class="grid grid-2">
  <div class="card">
    <div class="label">Equity Curve</div>
    <canvas id="equity"></canvas>
  </div>

  <div class="card">
    <div class="label">Trade Statistics</div>
    <div class="row">
      <span class="row-label">Avg Win</span>
      <span class="row-value gain" id="avg-win">+$0</span>
    </div>
    <div class="row">
      <span class="row-label">Avg Loss</span>
      <span class="row-value loss" id="avg-loss">-$0</span>
    </div>
    <div class="row">
      <span class="row-label">Max Single Win</span>
      <span class="row-value gain" id="max-win">+$0</span>
    </div>
    <div class="row">
      <span class="row-label">Max Single Loss</span>
      <span class="row-value loss" id="max-loss">-$0</span>
    </div>
    <div class="row">
      <span class="row-label">Win Streak</span>
      <span class="row-value" id="win-str">0</span>
    </div>
    <div class="row">
      <span class="row-label">Loss Streak</span>
      <span class="row-value" id="loss-str">0</span>
    </div>
    <canvas id="pnl-dist" height="200"></canvas>
  </div>
</div>

<!-- Analytics -->
<div class="grid grid-3">
  <div class="card">
    <div class="label">Symbol Performance</div>
    <div id="symbols">
      <div class="empty">—</div>
    </div>
  </div>

  <div class="card">
    <div class="label">Win/Loss Distribution</div>
    <canvas id="win-loss"></canvas>
  </div>

  <div class="card">
    <div class="label">Leverage Profile</div>
    <canvas id="leverage"></canvas>
  </div>
</div>


<!-- Positions & Trades -->
<div class="grid grid-2">
  <div class="card">
    <div class="label">Active Positions</div>
    <div id="positions">
      <div class="empty">No active positions</div>
    </div>
  </div>

  <div class="card">
    <div class="label">Risk Analysis</div>
    <div id="risk-analysis">
      <div class="empty">—</div>
    </div>
  </div>
</div>


<!-- Full Trade Table -->
<div class="card wide">
  <div class="label">Session Trades (Detail)</div>
  <div style="overflow-x: auto;">
    <table>
      <thead><tr><th>#</th><th>Pair</th><th>Side</th><th>Entry</th><th>Exit</th><th>SL</th><th>TP</th><th>P&L</th><th>R:R</th><th>Lev</th><th>Time</th><th>Duration</th><th>Score</th></tr></thead>
      <tbody id="trades-table">
        <tr><td colspan="13" class="empty">No trades yet</td></tr>
      </tbody>
    </table>
  </div>
</div>


<!-- Market Intelligence -->
<div class="grid grid-3">
  <div class="card">
    <div class="label">Market Regimes (Live)</div>
    <div id="regimes-display">
      <div class="empty">—</div>
    </div>
  </div>

  <div class="card">
    <div class="label">Kill Zones (UTC)</div>
    <div id="kill-zones">
      <div class="empty">—</div>
    </div>
  </div>

  <div class="card">
    <div class="label">Correlation Groups</div>
    <div id="corr-groups">
      <div class="empty">—</div>
    </div>
  </div>
</div>


<!-- Signal Analytics -->
<div class="grid grid-2">
  <div class="card">
    <div class="label">Recent Signals (Last 15)</div>
    <div id="signals-display">
      <div class="empty">No signals</div>
    </div>
  </div>

  <div class="card">
    <div class="label">Instrument Configuration</div>
    <div id="config-display">
      <div class="empty">—</div>
    </div>
  </div>
</div>


<!-- News & External Events -->
<div class="grid grid-2">
  <div class="card">
    <div class="label">News-Triggered Trades</div>
    <div class="row">
      <span class="row-label">Total News Trades</span>
      <span class="row-value" id="news-count">0</span>
    </div>
    <div id="news-trades" style="margin-top: 12px;">
      <div class="empty">No news trades</div>
    </div>
  </div>

  <div class="card">
    <div class="label">Session Summary</div>
    <div class="row">
      <span class="row-label">Trading Time</span>
      <span class="row-value" id="trading-duration">—</span>
    </div>
    <div class="row">
      <span class="row-label">Total Margin Used</span>
      <span class="row-value" id="total-margin">—</span>
    </div>
    <div class="row">
      <span class="row-label">Avg R:R Achieved</span>
      <span class="row-value" id="avg-rr">—</span>
    </div>
    <div class="row">
      <span class="row-label">Risk Per Trade</span>
      <span class="row-value" id="risk-per-trade">—</span>
    </div>
  </div>
</div>

<div class="footer">Real-time • Full analytics • Session data • Responsive • © 2026</div>

</div>

<script>
let charts = {};

function fmt(v) { return '$' + v.toLocaleString('en-US', {minimumFractionDigits: 2, maximumFractionDigits: 2}); }
function time(iso) { return iso ? new Date(iso).toLocaleTimeString('en-US', {hour: '2-digit', minute: '2-digit'}) : '—'; }
function duration(mins) { return mins < 60 ? mins + 'm' : (mins / 60).toFixed(1) + 'h'; }

function initCharts() {
  if (charts.equity) return;

  const ctxEq = document.getElementById('equity').getContext('2d');
  charts.equity = new Chart(ctxEq, {
    type: 'line',
    data: { labels: [], datasets: [{
      label: 'Balance',
      data: [],
      borderColor: '#39ff14',
      backgroundColor: 'rgba(57,255,20,0.1)',
      fill: true,
      tension: 0.3,
      pointRadius: 0,
      borderWidth: 2,
    }]},
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { display: false },
        y: { grid: { color: '#1a1a1a' }, ticks: { color: '#8b949e' } },
      }
    }
  });

  const ctxPnl = document.getElementById('pnl-dist').getContext('2d');
  charts.pnlDist = new Chart(ctxPnl, {
    type: 'bar',
    data: { labels: [], datasets: [{ label: 'P&L', data: [], backgroundColor: [] }]},
    options: {
      indexAxis: 'y',
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: { x: { grid: { color: '#1a1a1a' }, ticks: { color: '#8b949e' } } }
    }
  });

  const ctxWL = document.getElementById('win-loss').getContext('2d');
  charts.winLoss = new Chart(ctxWL, {
    type: 'doughnut',
    data: {
      labels: ['Wins', 'Losses', 'BE'],
      datasets: [{ data: [0,0,0], backgroundColor: ['#39ff14', '#ff4444', '#ffa500'] }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } }
    }
  });

  const ctxLev = document.getElementById('leverage').getContext('2d');
  charts.leverage = new Chart(ctxLev, {
    type: 'bar',
    data: { labels: [], datasets: [{ label: 'Leverage', data: [], backgroundColor: '#39ff14' }]},
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        y: { grid: { color: '#1a1a1a' }, ticks: { color: '#8b949e' } },
        x: { grid: { color: '#1a1a1a' }, ticks: { color: '#8b949e' } }
      }
    }
  });
}

async function refresh() {
  const [stats, eq, trades, pos, signals, regimes, kz, cfg, corr, news] = await Promise.all([
    fetch('/api/stats').then(r => r.json()),
    fetch('/api/equity-curve').then(r => r.json()),
    fetch('/api/trades-session').then(r => r.json()),
    fetch('/api/positions').then(r => r.json()),
    fetch('/api/signals').then(r => r.json()),
    fetch('/api/regimes').then(r => r.json()),
    fetch('/api/kill-zones').then(r => r.json()),
    fetch('/api/config-instruments').then(r => r.json()),
    fetch('/api/correlation-groups').then(r => r.json()),
    fetch('/api/news-impact').then(r => r.json()),
  ]);

  if (!Object.keys(charts).length) initCharts();

  document.getElementById('time').textContent = new Date().toLocaleTimeString();

  document.getElementById('bal').textContent = fmt(stats.balance);
  document.getElementById('bal').className = 'value ' + (stats.balance >= stats.start_balance ? 'gain' : 'loss');
  document.getElementById('bal-pct').textContent = (stats.return_pct >= 0 ? '+' : '') + stats.return_pct.toFixed(2) + '%';

  document.getElementById('pnl').className = 'value ' + (stats.session_pnl >= 0 ? 'gain' : 'loss');
  document.getElementById('pnl').textContent = (stats.session_pnl >= 0 ? '+' : '') + fmt(stats.session_pnl);
  document.getElementById('ret').textContent = (stats.return_pct >= 0 ? '+' : '') + stats.return_pct.toFixed(2) + '%';

  document.getElementById('wr').className = 'value ' + (stats.win_rate > 50 ? 'gain' : 'neutral');
  document.getElementById('wr').textContent = stats.win_rate.toFixed(0) + '%';
  document.getElementById('wr-sub').textContent = stats.wins + 'W / ' + stats.losses + 'L';

  document.getElementById('pf').textContent = stats.profit_factor.toFixed(2);
  document.getElementById('exp').className = 'value ' + (stats.expectancy >= 0 ? 'gain' : 'loss');
  document.getElementById('exp').textContent = (stats.expectancy >= 0 ? '+' : '') + fmt(stats.expectancy);
  document.getElementById('sharpe').textContent = stats.sharpe_ratio.toFixed(2);

  document.getElementById('trades').textContent = stats.session_trades;
  document.getElementById('open').textContent = stats.open_trades;
  document.getElementById('avg-lev').textContent = stats.avg_leverage.toFixed(1) + '×';

  const durStr = stats.avg_duration_mins < 60 ? stats.avg_duration_mins + 'm' : (stats.avg_duration_mins / 60).toFixed(1) + 'h';
  document.getElementById('avg-dur').textContent = durStr;
  document.getElementById('dur-sub').textContent = stats.min_duration_mins + 'm – ' + stats.max_duration_mins + 'm';

  document.getElementById('best').textContent = '+' + fmt(stats.best_trade_pnl);
  document.getElementById('worst').textContent = fmt(stats.worst_trade_pnl);

  document.getElementById('avg-win').textContent = '+' + fmt(stats.avg_win);
  document.getElementById('avg-loss').textContent = fmt(stats.avg_loss);
  document.getElementById('max-win').textContent = '+' + fmt(stats.max_single_win);
  document.getElementById('max-loss').textContent = fmt(stats.max_single_loss);
  document.getElementById('win-str').textContent = stats.max_win_streak;
  document.getElementById('loss-str').textContent = stats.max_loss_streak;

  document.getElementById('streak').textContent = stats.max_win_streak + 'W';
  document.getElementById('streak-sub').textContent = stats.max_loss_streak + 'L';

  // Equity chart
  if (charts.equity && eq.length > 0) {
    charts.equity.data.labels = eq.map(d => d.time);
    charts.equity.data.datasets[0].data = eq.map(d => d.balance);
    charts.equity.update();
  }

  // P&L distribution
  if (charts.pnlDist && trades.length > 0) {
    charts.pnlDist.data.labels = trades.map(t => '#' + t.id);
    charts.pnlDist.data.datasets[0].data = trades.map(t => t.pnl);
    charts.pnlDist.data.datasets[0].backgroundColor = trades.map(t => t.pnl >= 0 ? '#39ff14' : '#ff4444');
    charts.pnlDist.update();
  }

  // Win/Loss
  if (charts.winLoss) {
    charts.winLoss.data.datasets[0].data = [stats.wins, stats.losses, stats.flats];
    charts.winLoss.update();
  }

  // Leverage
  if (charts.leverage && trades.length > 0) {
    charts.leverage.data.labels = trades.map(t => '#' + t.id);
    charts.leverage.data.datasets[0].data = trades.map(t => t.leverage);
    charts.leverage.update();
  }

  // Symbol stats
  const symHtml = Object.entries(stats.symbol_stats).map(([sym, st]) => {
    const wr = st.trades > 0 ? (st.wins / st.trades * 100).toFixed(0) : 0;
    const cls = st.pnl >= 0 ? 'gain' : 'loss';
    return `<div class="row"><span class="row-label">${sym}</span><span class="row-value">${st.trades}T ${wr}% <span class="${cls}">${st.pnl >= 0 ? '+' : ''}${fmt(st.pnl)}</span></span></div>`;
  }).join('');
  document.getElementById('symbols').innerHTML = symHtml || '<div class="empty">—</div>';

  // Risk analysis
  const riskHtml = `
    <div class="row"><span class="row-label">Max DD from Peak</span><span class="row-value">~${(eq[eq.length - 1]?.dd || 0).toFixed(1)}%</span></div>
    <div class="row"><span class="row-label">Risk Per Trade</span><span class="row-value">${fmt((stats.session_pnl / Math.max(stats.session_trades, 1)).toFixed(2))}</span></div>
    <div class="row"><span class="row-label">Total Margin Used</span><span class="row-value">${(trades.reduce((s, t) => s + (t.margin || 0), 0) / Math.max(trades.length, 1)).toFixed(0)}%</span></div>
    <div class="row"><span class="row-label">Avg R:R</span><span class="row-value">${(trades.reduce((s, t) => s + (t.rr || 0), 0) / Math.max(trades.length, 1)).toFixed(2)}R</span></div>
  `;
  document.getElementById('risk-analysis').innerHTML = riskHtml;

  // Positions
  document.getElementById('positions').innerHTML = pos.length === 0
    ? '<div class="empty">No active positions</div>'
    : pos.map(p => {
      const pnlClass = p.unrealized_pnl >= 0 ? 'gain' : 'loss';
      const warnClass = p.liq_warning ? ' danger' : '';
      return `<div class="pos-card${warnClass}">
        <div class="pos-header">
          <div>${p.symbol} <span class="chip ${p.side}">${p.side.toUpperCase()}</span> ${p.leverage}×</div>
          <div class="${pnlClass}">${p.unrealized_pnl >= 0 ? '+' : ''}${fmt(p.unrealized_pnl)}</div>
        </div>
        <div class="pos-grid">
          <div class="pos-row"><span>Entry</span><span class="pos-val">${fmt(p.entry_price)}</span></div>
          <div class="pos-row"><span>SL ${p.sl_distance_pct.toFixed(1)}% away</span><span class="pos-val">${fmt(p.sl_price)}</span></div>
          <div class="pos-row"><span>TP ${p.tp_distance_pct.toFixed(1)}% away</span><span class="pos-val">${fmt(p.tp_price)}</span></div>
          <div class="pos-row"><span>Liq ${p.liq_distance_pct.toFixed(1)}% ${p.liq_warning ? '⚠️' : '✓'}</span><span class="pos-val">${fmt(p.liq_price)}</span></div>
        </div>
      </div>`;
    }).join('');

  // Trades table
  document.getElementById('trades-table').innerHTML = trades.length === 0
    ? '<tr><td colspan="13" class="empty">No trades yet</td></tr>'
    : trades.map(t => {
      const pnlClass = t.pnl >= 0 ? 'gain' : 'loss';
      return `<tr>
        <td>#${t.id}</td>
        <td><b>${t.symbol}</b></td>
        <td><span class="chip ${t.side}">${t.side.toUpperCase()}</span></td>
        <td>${fmt(t.entry_price)}</td>
        <td>${fmt(t.exit_price)}</td>
        <td>${fmt(t.sl_price)}</td>
        <td>${fmt(t.tp_price)}</td>
        <td class="${pnlClass}"><b>${t.pnl >= 0 ? '+' : ''}${fmt(t.pnl)}</b></td>
        <td>${t.rr.toFixed(2)}R</td>
        <td>${t.leverage}×</td>
        <td>${time(t.closed_at)}</td>
        <td>${duration(t.duration_mins)}</td>
        <td>${t.score}</td>
      </tr>`;
    }).join('');

  // Kill Zones
  const kzHtml = kz.map(z => {
    const cls = z.active ? 'gain' : 'neutral';
    return `<div class="row"><span class="row-label">${z.zone}</span><span class="row-value ${cls}">${z.start}—${z.end} ${z.active ? '●' : '○'}</span></div>`;
  }).join('');
  document.getElementById('kill-zones').innerHTML = kzHtml || '<div class="empty">—</div>';

  // Regimes
  const regHtml = Object.entries(regimes).map(([sym, r]) => {
    const regimeColor = r.regime === 'trending' ? 'gain' : r.regime === 'choppy' ? 'loss' : 'neutral';
    return `<div class="row"><span class="row-label">${sym}</span><span class="row-value"><span class="${regimeColor}">${r.regime}</span> ADX:${r.adx} Vol:${r.volatility_pct}%</span></div>`;
  }).join('');
  document.getElementById('regimes-display').innerHTML = regHtml || '<div class="empty">—</div>';

  // Correlation Groups
  const corrHtml = Object.entries(corr).map(([group, g]) => {
    const cls = g.at_limit ? 'loss' : 'neutral';
    return `<div class="row"><span class="row-label">${group}</span><span class="row-value ${cls}">${g.open_count}/${g.max_positions} open</span></div>`;
  }).join('');
  document.getElementById('corr-groups').innerHTML = corrHtml || '<div class="empty">—</div>';

  // Instrument Config
  const cfgHtml = Object.entries(cfg).map(([sym, c]) => {
    return `<div class="row"><span class="row-label">${sym}</span><span class="row-value">RR:${c.min_rr} Risk:${c.risk_pct}% FVG:${c.fvg_gap}</span></div>`;
  }).join('');
  document.getElementById('config-display').innerHTML = cfgHtml || '<div class="empty">—</div>';

  // Recent Signals
  const sigHtml = signals.slice(0, 15).map(s => {
    const scoreColor = s.score >= 75 ? 'gain' : s.score >= 60 ? 'neutral' : 'loss';
    const status = s.executed ? '✓' : '✗ ' + (s.skipped_reason || 'skip');
    return `<div class="row"><span class="row-label">${time(s.created_at)} ${s.symbol} <span class="chip ${s.side}">${s.side[0]}</span></span><span class="row-value"><span class="${scoreColor}">${s.score}</span> ${s.rr ? s.rr.toFixed(1)+'R' : '—'} ${status}</span></div>`;
  }).join('');
  document.getElementById('signals-display').innerHTML = sigHtml || '<div class="empty">No signals</div>';

  // News Impact
  document.getElementById('news-count').textContent = news.news_triggered_count || 0;
  const newsHtml = news.trades.slice(0, 10).map(t => {
    const pnlClass = t.pnl >= 0 ? 'gain' : 'loss';
    return `<div class="row"><span class="row-label">${t.symbol} <span class="chip ${t.side}">${t.side[0]}</span></span><span class="row-value ${pnlClass}">${t.pnl >= 0 ? '+' : ''}${fmt(t.pnl)}</span></div>`;
  }).join('');
  document.getElementById('news-trades').innerHTML = newsHtml || '<div class="empty">No news trades</div>';

  // Session Summary
  if (trades.length > 0) {
    const firstTrade = trades[0];
    const lastTrade = trades[trades.length - 1];
    try {
      const start = new Date(firstTrade.opened_at);
      const end = new Date(lastTrade.closed_at);
      const durationHours = ((end - start) / (1000 * 60 * 60)).toFixed(1);
      document.getElementById('trading-duration').textContent = durationHours + 'h';
    } catch (e) {
      document.getElementById('trading-duration').textContent = '—';
    }

    const totalMargin = trades.reduce((s, t) => s + (t.margin || 0), 0);
    const avgMargin = (totalMargin / trades.length).toFixed(0);
    document.getElementById('total-margin').textContent = avgMargin + '%';

    const avgRR = trades.reduce((s, t) => s + (t.rr || 0), 0) / trades.length;
    document.getElementById('avg-rr').textContent = avgRR.toFixed(2) + 'R';

    const riskPerTrade = (stats.session_pnl / Math.max(stats.session_trades, 1)).toFixed(2);
    document.getElementById('risk-per-trade').textContent = fmt(riskPerTrade);
  }
}

refresh();
setInterval(refresh, 10000);
</script>

</body>
</html>"""


def main():
    port = int(os.environ.get("DASHBOARD_PORT", "8080"))
    host = os.environ.get("DASHBOARD_HOST", "0.0.0.0")
    logger.info(f"Starting dashboard on {host}:{port}")
    app.run(host=host, port=port, debug=False)


if __name__ == "__main__":
    main()
