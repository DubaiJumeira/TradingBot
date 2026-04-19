"""
Backtester report generation — JSON + HTML with Plotly charts.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class BacktestReport:
    """Generate reports from backtest results."""

    def __init__(self, results: dict[str, Any], symbol: str = "") -> None:
        self.results = results
        self.symbol = symbol

    def to_json(self, path: str | Path) -> None:
        """Save results as JSON (excluding equity curve for size)."""
        out = {k: v for k, v in self.results.items() if k != "equity_curve"}
        Path(path).write_text(json.dumps(out, indent=2, default=str))
        logger.info("Report saved to %s", path)

    def to_html(self, path: str | Path) -> None:
        """Generate an HTML report with Plotly equity curve chart."""
        r = self.results
        eq = r.get("equity_curve", [])

        # Build equity curve chart as inline Plotly.
        eq_json = json.dumps(eq)
        trades_json = json.dumps(r.get("trades", []), default=str)

        html = f"""<!DOCTYPE html>
<html><head>
<title>Backtest Report — {self.symbol}</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
  body {{ font-family: system-ui, sans-serif; max-width: 1000px; margin: 40px auto; padding: 0 20px; }}
  .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin: 20px 0; }}
  .stat {{ background: #f5f5f5; padding: 12px; border-radius: 8px; }}
  .stat .label {{ font-size: 12px; color: #666; text-transform: uppercase; }}
  .stat .value {{ font-size: 24px; font-weight: 600; margin-top: 4px; }}
  .positive {{ color: #22c55e; }}
  .negative {{ color: #ef4444; }}
  table {{ width: 100%; border-collapse: collapse; margin: 20px 0; font-size: 14px; }}
  th, td {{ padding: 8px 12px; border-bottom: 1px solid #e5e5e5; text-align: left; }}
  th {{ background: #f5f5f5; font-weight: 600; }}
</style>
</head><body>
<h1>Backtest Report — {self.symbol}</h1>

<div class="stats">
  <div class="stat"><div class="label">Total Trades</div><div class="value">{r.get('total_trades', 0)}</div></div>
  <div class="stat"><div class="label">Win Rate</div><div class="value">{r.get('win_rate', 0)}%</div></div>
  <div class="stat"><div class="label">Profit Factor</div><div class="value">{r.get('profit_factor', 0)}</div></div>
  <div class="stat"><div class="label">Total PnL</div><div class="value {'positive' if r.get('total_pnl',0) >= 0 else 'negative'}">${r.get('total_pnl', 0):,.2f}</div></div>
  <div class="stat"><div class="label">Max Drawdown</div><div class="value negative">{r.get('max_drawdown_pct', 0):.1f}%</div></div>
  <div class="stat"><div class="label">Sharpe Ratio</div><div class="value">{r.get('sharpe_ratio', 0)}</div></div>
  <div class="stat"><div class="label">Final Balance</div><div class="value">${r.get('final_balance', 0):,.2f}</div></div>
</div>

<h2>Equity Curve</h2>
<div id="equity-chart"></div>

<h2>Trade Log</h2>
<table>
<tr><th>#</th><th>Side</th><th>Entry</th><th>Exit</th><th>Result</th><th>PnL</th><th>Score</th><th>Bars</th><th>News</th></tr>
"""
        for i, t in enumerate(r.get("trades", []), 1):
            pnl_class = "positive" if t.get("pnl", 0) >= 0 else "negative"
            html += (
                f'<tr><td>{i}</td><td>{t.get("side","")}</td>'
                f'<td>{t.get("entry","")}</td><td>{t.get("exit","")}</td>'
                f'<td>{t.get("result","")}</td>'
                f'<td class="{pnl_class}">${t.get("pnl",0):,.2f}</td>'
                f'<td>{t.get("score","")}</td><td>{t.get("bars_held","")}</td>'
                f'<td>{"Y" if t.get("news_triggered") else ""}</td></tr>\\n'
            )

        html += f"""</table>

<script>
var eq = {eq_json};
Plotly.newPlot('equity-chart', [{{
  y: eq, mode: 'lines', name: 'Equity',
  line: {{color: '#3b82f6', width: 2}}
}}], {{
  title: 'Equity Curve',
  xaxis: {{title: 'Bar'}},
  yaxis: {{title: 'Equity ($)'}},
  template: 'plotly_white',
}});
</script>
</body></html>"""

        Path(path).write_text(html)
        logger.info("HTML report saved to %s", path)

    def summary(self) -> str:
        """One-line text summary."""
        r = self.results
        return (
            f"{self.symbol}: {r.get('total_trades',0)} trades, "
            f"{r.get('win_rate',0)}% WR, PF={r.get('profit_factor',0)}, "
            f"PnL=${r.get('total_pnl',0):,.2f}, MaxDD={r.get('max_drawdown_pct',0):.1f}%"
        )
