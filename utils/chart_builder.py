"""
Chart rendering for Telegram signal alerts — Phase 4C.

Produces a compact candlestick PNG annotated with entry, stop-loss, and
take-profit levels plus a direction marker at the entry candle. Kept
deliberately minimal: 100 candles, one pane of volume, no zone shading.
Advanced overlays (FVG, OB, liquidation bands) belong to later phases.

The render is wrapped in its own try/except by callers — a chart failure
must never block the Telegram alert.
"""

from __future__ import annotations

import io
import logging
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless: no display needed on the VPS
import mplfinance as mpf
import pandas as pd

logger = logging.getLogger(__name__)

_STYLE = mpf.make_mpf_style(
    base_mpf_style="nightclouds",
    marketcolors=mpf.make_marketcolors(
        up="#26a69a", down="#ef5350",
        edge={"up": "#26a69a", "down": "#ef5350"},
        wick={"up": "#26a69a", "down": "#ef5350"},
        volume={"up": "#26a69a", "down": "#ef5350"},
    ),
    gridstyle=":",
    facecolor="#0e1117",
    edgecolor="#0e1117",
    figcolor="#0e1117",
    gridcolor="#1f2530",
    y_on_right=True,
)


def render_signal_chart(df: pd.DataFrame, signal: dict[str, Any]) -> bytes | None:
    """Render a signal chart to PNG bytes.

    Parameters
    ----------
    df : DataFrame with a DatetimeIndex and open/high/low/close/volume columns.
    signal : dict carrying symbol, side, entry, sl, tp, score.

    Returns
    -------
    PNG bytes, or None if rendering failed (caller falls back to text-only).
    """
    try:
        plot_df = df.tail(100).copy()
        plot_df.columns = [c.capitalize() for c in plot_df.columns]

        hlines = dict(
            hlines=[signal["entry"], signal["sl"], signal["tp"]],
            colors=["#ffd54f", "#ef5350", "#26a69a"],
            linestyle=["-", "--", "--"],
            linewidths=[1.2, 1.0, 1.0],
        )

        title = (
            f"{signal['symbol']}  {signal['side'].upper()}  "
            f"@ {signal['entry']:.2f}   score {signal.get('score', '?')}/100"
        )

        buf = io.BytesIO()
        mpf.plot(
            plot_df,
            type="candle",
            style=_STYLE,
            volume=True,
            hlines=hlines,
            figsize=(9, 6),
            tight_layout=True,
            title=title,
            ylabel="",
            ylabel_lower="",
            savefig=dict(fname=buf, format="png", dpi=110, bbox_inches="tight"),
        )
        buf.seek(0)
        return buf.getvalue()
    except Exception as exc:
        logger.warning("chart render failed for %s: %s", signal.get("symbol"), exc)
        return None
