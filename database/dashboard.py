"""
Phase 10 — Flask Dashboard

Lightweight web UI showing equity curve, open positions, recent trades,
signal history, and per-symbol regime. Reads from TradeDB.

Usage:
    from database import TradeDB, create_app
    db = TradeDB("data/trading_bot.db")
    app = create_app(db)
    app.run(host="0.0.0.0", port=5000)
"""

from __future__ import annotations

import logging
from typing import Any

from flask import Flask, jsonify, render_template_string

from database.db import TradeDB

logger = logging.getLogger(__name__)


_DASHBOARD_HTML = """
<!DOCTYPE html>
<html>
<head>
  <title>Trading Bot Dashboard</title>
  <meta http-equiv="refresh" content="30">
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    body { font-family: -apple-system, sans-serif; margin: 0; padding: 20px;
           background: #0d1117; color: #c9d1d9; }
    h1, h2 { color: #58a6ff; }
    .grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 20px; margin: 20px 0; }
    .card { background: #161b22; border: 1px solid #30363d; border-radius: 8px;
            padding: 16px; }
    .metric { font-size: 28px; font-weight: bold; }
    .metric.green { color: #3fb950; }
    .metric.red { color: #f85149; }
    .label { font-size: 12px; text-transform: uppercase; color: #8b949e; }
    table { width: 100%; border-collapse: collapse; margin: 10px 0; }
    th, td { padding: 8px 12px; text-align: left; border-bottom: 1px solid #30363d; }
    th { background: #161b22; color: #8b949e; font-size: 12px; text-transform: uppercase; }
    .long { color: #3fb950; }
    .short { color: #f85149; }
    .pnl-pos { color: #3fb950; }
    .pnl-neg { color: #f85149; }
    #equity-chart { width: 100%; height: 400px; background: #161b22;
                    border: 1px solid #30363d; border-radius: 8px; padding: 10px; }
  </style>
</head>
<body>
  <h1>📊 Trading Bot Dashboard</h1>

  <div class="grid">
    <div class="card">
      <div class="label">Balance</div>
      <div class="metric">${{ "%.2f"|format(stats.balance) }}</div>
    </div>
    <div class="card">
      <div class="label">Total PnL</div>
      <div class="metric {{ 'green' if stats.total_pnl >= 0 else 'red' }}">
        ${{ "%.2f"|format(stats.total_pnl) }}
      </div>
    </div>
    <div class="card">
      <div class="label">Win Rate</div>
      <div class="metric">{{ stats.win_rate }}%</div>
    </div>
    <div class="card">
      <div class="label">Total Trades</div>
      <div class="metric">{{ stats.total }}</div>
    </div>
  </div>

  <h2>Equity Curve</h2>
  <div id="equity-chart"></div>
  <script>
    var trace = {
      x: {{ equity_times|tojson }},
      y: {{ equity_values|tojson }},
      type: 'scatter',
      mode: 'lines',
      line: { color: '#58a6ff', width: 2 },
      fill: 'tozeroy',
      fillcolor: 'rgba(88, 166, 255, 0.1)'
    };
    var layout = {
      paper_bgcolor: '#161b22',
      plot_bgcolor: '#161b22',
      font: { color: '#c9d1d9' },
      xaxis: { gridcolor: '#30363d' },
      yaxis: { gridcolor: '#30363d', title: 'Balance ($)' },
      margin: { l: 60, r: 20, t: 20, b: 40 }
    };
    Plotly.newPlot('equity-chart', [trace], layout, {displayModeBar: false});
  </script>

  <h2>Open Positions ({{ open_trades|length }})</h2>
  <table>
    <tr><th>Symbol</th><th>Side</th><th>Entry</th><th>SL</th><th>TP</th>
        <th>Size $</th><th>Score</th><th>Regime</th><th>Opened</th></tr>
    {% for t in open_trades %}
    <tr>
      <td>{{ t.symbol }}</td>
      <td class="{{ t.side }}">{{ t.side|upper }}</td>
      <td>${{ "%.2f"|format(t.entry_price) }}</td>
      <td>${{ "%.2f"|format(t.sl_price or 0) }}</td>
      <td>${{ "%.2f"|format(t.tp_price or 0) }}</td>
      <td>${{ "%.2f"|format(t.size_usd or 0) }}</td>
      <td>{{ t.score or '-' }}</td>
      <td>{{ t.regime or '-' }}</td>
      <td>{{ t.opened_at[:16] }}</td>
    </tr>
    {% else %}
    <tr><td colspan="9" style="text-align:center;color:#8b949e">No open positions</td></tr>
    {% endfor %}
  </table>

  <h2>Recent Closed Trades</h2>
  <table>
    <tr><th>Symbol</th><th>Side</th><th>Entry</th><th>Exit</th><th>PnL</th>
        <th>Result</th><th>Score</th><th>Regime</th><th>Closed</th></tr>
    {% for t in closed_trades %}
    <tr>
      <td>{{ t.symbol }}</td>
      <td class="{{ t.side }}">{{ t.side|upper }}</td>
      <td>${{ "%.2f"|format(t.entry_price) }}</td>
      <td>${{ "%.2f"|format(t.exit_price or 0) }}</td>
      <td class="{{ 'pnl-pos' if (t.pnl or 0) >= 0 else 'pnl-neg' }}">
        ${{ "%.2f"|format(t.pnl or 0) }}
      </td>
      <td>{{ t.result or '-' }}</td>
      <td>{{ t.score or '-' }}</td>
      <td>{{ t.regime or '-' }}</td>
      <td>{{ (t.closed_at or '')[:16] }}</td>
    </tr>
    {% else %}
    <tr><td colspan="9" style="text-align:center;color:#8b949e">No closed trades yet</td></tr>
    {% endfor %}
  </table>

  <h2>Recent Signals</h2>
  <table>
    <tr><th>Time</th><th>Symbol</th><th>Side</th><th>Score</th><th>RR</th>
        <th>Regime</th><th>Executed</th><th>Skip Reason</th></tr>
    {% for s in signals %}
    <tr>
      <td>{{ s.created_at[:16] }}</td>
      <td>{{ s.symbol }}</td>
      <td class="{{ s.side }}">{{ s.side|upper }}</td>
      <td>{{ s.score or '-' }}</td>
      <td>{{ s.rr or '-' }}</td>
      <td>{{ s.regime or '-' }}</td>
      <td>{{ '✅' if s.executed else '⏭️' }}</td>
      <td>{{ s.skipped_reason or '' }}</td>
    </tr>
    {% endfor %}
  </table>

  <p style="color:#8b949e;font-size:11px;margin-top:30px;">
    Auto-refreshes every 30s • {{ now }}
  </p>
</body>
</html>
"""


def create_app(db: TradeDB) -> Flask:
    """Create the Flask app bound to a TradeDB instance."""
    app = Flask(__name__)

    @app.route("/")
    def index():
        from datetime import datetime, timezone

        stats = db.trade_stats()
        equity_rows = db.get_equity_curve(limit=500)

        # Current balance = latest equity or starting balance fallback.
        current_balance = equity_rows[-1]["balance"] if equity_rows else 10000
        stats["balance"] = current_balance

        open_trades = db.get_open_trades()
        all_closed = [t for t in db.get_trades(limit=30) if t["closed_at"]]

        return render_template_string(
            _DASHBOARD_HTML,
            stats=stats,
            open_trades=open_trades,
            closed_trades=all_closed,
            signals=db.get_signals(limit=20),
            equity_times=[r["recorded_at"] for r in equity_rows],
            equity_values=[r["balance"] for r in equity_rows],
            now=datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        )

    @app.route("/api/stats")
    def api_stats():
        return jsonify(db.trade_stats())

    @app.route("/api/trades")
    def api_trades():
        return jsonify(db.get_trades(limit=100))

    @app.route("/api/signals")
    def api_signals():
        return jsonify(db.get_signals(limit=50))

    @app.route("/api/equity")
    def api_equity():
        return jsonify(db.get_equity_curve(limit=500))

    @app.route("/api/open")
    def api_open():
        return jsonify(db.get_open_trades())

    @app.route("/api/regime/<path:symbol>")
    def api_regime(symbol: str):
        return jsonify(db.get_regime_history(symbol, limit=100))

    return app
