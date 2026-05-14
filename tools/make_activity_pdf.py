"""
Generate a PDF snapshot of bot activity, history, and recent changes.
Output: data/bot_activity_report.pdf
"""
from __future__ import annotations

import json
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, "/root/trading-bot")

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak,
)

DB = "/root/trading-bot/data/trading_bot.db"
PAPER = "/root/trading-bot/data/paper_trades.json"
WEIGHTS = "/root/trading-bot/data/optimizer_weights.json"
OUT_PDF = "/root/trading-bot/data/bot_activity_report.pdf"
EQUITY_PNG = "/tmp/bot_equity.png"
PNL_PNG = "/tmp/bot_pnl.png"


def load_equity() -> list[tuple[str, float, float]]:
    with sqlite3.connect(DB) as c:
        cur = c.cursor()
        cur.execute("SELECT recorded_at, balance, drawdown_pct FROM equity ORDER BY id ASC")
        return cur.fetchall()


def load_signals() -> list[dict]:
    with sqlite3.connect(DB) as c:
        c.row_factory = sqlite3.Row
        rows = c.execute(
            "SELECT created_at, symbol, side, score, rr, regime, executed, details "
            "FROM signals ORDER BY id ASC"
        ).fetchall()
    return [dict(r) for r in rows]


def load_trade_history() -> tuple[list[dict], float, dict]:
    with open(PAPER) as f:
        d = json.load(f)
    return d.get("trade_history", []), d.get("balance", 0.0), d.get("positions", {})


def load_weights() -> dict:
    try:
        with open(WEIGHTS) as f:
            return json.load(f)
    except Exception:
        return {}


def render_equity_png(equity: list[tuple[str, float, float]], path: str) -> None:
    if not equity:
        return
    ts = [datetime.fromisoformat(r[0].replace("Z", "+00:00")) for r in equity]
    bal = [r[1] for r in equity]
    dd = [r[2] for r in equity]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 5), gridspec_kw={"height_ratios": [3, 1]})
    ax1.plot(ts, bal, color="#1f77b4", linewidth=1.4)
    ax1.fill_between(ts, bal, min(bal), alpha=0.12, color="#1f77b4")
    ax1.set_ylabel("Balance ($)")
    ax1.grid(alpha=0.3)
    ax1.set_title("Paper-trading equity curve")
    ax2.fill_between(ts, dd, 0, color="#d62728", alpha=0.5)
    ax2.set_ylabel("Drawdown %")
    ax2.grid(alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M"))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close(fig)


def render_pnl_png(trades: list[dict], path: str) -> None:
    if not trades:
        return
    pnls = [t.get("pnl", 0) for t in trades]
    labels = [f"#{t['id']} {t['symbol'][:3]}" for t in trades]
    cs = ["#22c55e" if p > 0 else "#ef4444" for p in pnls]
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.bar(range(len(pnls)), pnls, color=cs)
    ax.set_xticks(range(len(pnls)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.axhline(0, color="#333", linewidth=0.6)
    ax.set_ylabel("PnL ($)")
    ax.set_title("Per-trade PnL (paper)")
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close(fig)


def _p(style, text):
    return Paragraph(text.replace("\n", "<br/>"), style)


def build_pdf():
    equity = load_equity()
    signals = load_signals()
    trades, balance, open_positions = load_trade_history()
    weights = load_weights()

    render_equity_png(equity, EQUITY_PNG)
    render_pnl_png(trades, PNL_PNG)

    doc = SimpleDocTemplate(OUT_PDF, pagesize=A4,
                            leftMargin=1.6 * cm, rightMargin=1.6 * cm,
                            topMargin=1.6 * cm, bottomMargin=1.6 * cm)
    styles = getSampleStyleSheet()
    H1 = ParagraphStyle("H1", parent=styles["Heading1"], fontSize=20, spaceAfter=10)
    H2 = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=14, spaceBefore=10, spaceAfter=6)
    H3 = ParagraphStyle("H3", parent=styles["Heading3"], fontSize=11, spaceBefore=6)
    body = styles["BodyText"]
    small = ParagraphStyle("small", parent=body, fontSize=8, leading=10)
    mono = ParagraphStyle("mono", parent=body, fontName="Courier", fontSize=8, leading=10)

    elements = []

    # ---------- Cover ----------
    elements.append(_p(H1, "ICT / Wyckoff Crypto Trading Bot"))
    elements.append(_p(body, f"<b>Activity &amp; Change Report</b> — generated {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}"))
    elements.append(Spacer(1, 0.3 * cm))

    first_sig = signals[0]["created_at"][:10] if signals else "n/a"
    last_sig = signals[-1]["created_at"][:10] if signals else "n/a"
    summary_rows = [
        ["Data span (signals)", f"{first_sig} → {last_sig}"],
        ["Signals logged", str(len(signals))],
        ["Paper trades closed", str(len(trades))],
        ["Paper trades open", str(len(open_positions))],
        ["Current balance", f"${balance:,.2f}"],
        ["Equity samples", str(len(equity))],
        ["Mode", "PAPER"],
        ["Running symbols", "BTCUSDT, ETHUSDT, SOLUSDT"],
    ]
    t = Table(summary_rows, colWidths=[5 * cm, 10 * cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#0d1421")),
        ("TEXTCOLOR", (0, 0), (0, -1), colors.white),
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
    ]))
    elements.append(t)
    elements.append(Spacer(1, 0.4 * cm))

    # ---------- Architecture / Phases ----------
    elements.append(_p(H2, "1. Architecture &amp; development phases"))
    elements.append(_p(body, (
        "The bot was built incrementally in numbered phases. Each phase adds a "
        "strategy layer or risk feature. Current state corresponds to <b>Phase 12+</b> "
        "plus the live-data / self-optimization work added in this session."
    )))
    phases = [
        ("Phase 1-2", "ICT core (order blocks, FVGs, breakers, inducements, BOS/ChoCH); per-instrument risk config; MIN_RR gate."),
        ("Phase 3",   "ICT enrichments: OTE fib zone, premium/discount, liquidity voids, displacement."),
        ("Phase 4",   "Wyckoff phase engine (accumulation, distribution, markup, markdown, spring/UTAD); VSA effort-vs-result."),
        ("Phase 5",   "Self-optimizer: nightly learner that re-weights signal components from closed-trade PnL."),
        ("Phase 6",   "Partial-TP plan: TP1 at 2R (50%), TP2 at 3R (30%), runner at extended target; breakeven+ after TP1."),
        ("Phase 7",   "Advanced ICT: composite-man detection, stop-hunt, structural trail, coordinated-move clusters."),
        ("Phase 8",   "Manipulation tracker: spoofing detection (fake support/resistance), wash-trading flag."),
        ("Phase 9",   "Regime detector: trending / ranging / volatile / squeeze with TP-mult, SL-mult, min-score adjust."),
        ("Phase 10",  "Persistent DB (signals, trades, equity, regimes); daily stats; equity snapshots every cycle."),
        ("Phase 11",  "Order-flow analysis (cvd, absorption on orderbook)."),
        ("Phase 12",  "MTF confluence (1D / 4h / 1h / 15m cached bias); MTF veto scoring."),
        ("Phase 13",  "Historical-replay backtester (tools/backtest.py) with warmup + signal pipeline at every bar."),
        ("+ session", "Live-data layer: orderbook walls, observed liquidations (Binance+Bybit WS), volume profile HVN. HVN scoring bonus, HVN-aware TP extension, wall-aware SL tightening. Counter-zone hard veto. ATR SL floor raised. Runner-SL labeling. Floor re-enforced after regime mult."),
    ]
    tbl = Table(phases, colWidths=[3.2 * cm, 13 * cm])
    tbl.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#1a2535")),
        ("TEXTCOLOR", (0, 0), (0, -1), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
    ]))
    elements.append(tbl)
    elements.append(PageBreak())

    # ---------- Changes this session ----------
    elements.append(_p(H2, "2. Changes made in this session"))
    changes = [
        ("orderbook_liquidity.py (new)",
         "Real order-book depth fetcher + bucketing. 500-level depth from Binance/Bybit. 15s cache. Surfaces walls thick enough to act as magnets/absorbers."),
        ("volume_profile.py (new)",
         "Wide-scale volume profile from historical candles. Bins 20d of 1h data into 80 price buckets. Identifies HVN, LVN, POC. 10m cache."),
        ("liquidation_stream.py",
         "Added dump_recent_events() — atomic JSON dump every 30s for cross-process sharing (bot → dashboard heatmap)."),
        ("liquidation.py",
         "Priority reorder: coinglass → observed+orderbook → orderbook → observed → unavailable. Removed synthetic estimator from live path (pure real data)."),
        ("market_data.py",
         "Added walls[] + volume_profile_deep{hvn,lvn,poc,nodes} to market analysis block. Now feeds downstream scoring."),
        ("signal_generator.py",
         "(a) HVN scoring bonus: +10 within 2%, +6 within 5%, aligned direction only. "
         "(b) TP-extension picker: farthest-aligned of liq-magnet vs HVN, never pulls TP in. "
         "(c) Wall-aware SL tightening: snap SL past thick wall between entry and old SL. "
         "(d) Counter-zone HARD veto: long-in-premium / short-in-discount rejected outright. "
         "(e) ATR SL floor raised 0.5×ATR/0.5% → 1.0×ATR/0.8%. "
         "(f) ATR floor re-enforced AFTER regime multiplier (fixes bug where ranging ×0.8 shrank SL back below floor)."),
        ("exchange_handler.py",
         "Runner-SL labeling: distinguishes <i>SL_RUNNER_TP{n}</i> (partials filled before final stop) from pure <i>SL</i>, so +R runner outcomes stop being misreported."),
        ("web/dashboard.py",
         "Added /api/liquidity-heatmap/&lt;symbol&gt; endpoint + multi-layer Chart.js heatmap UI (volume profile background, orderbook walls bold, liquidation markers)."),
        ("utils/telegram_bot.py",
         "/liquidation command now shows top 10 walls per side + wide-scale HVN data."),
        ("bot.py",
         "Scheduled 30s liquidation dump for cross-process sharing."),
        ("tools/backtest_random.py (new)",
         "Random-window driver for multi-symbol sampled backtests (not used due to backtester per-bar cost)."),
        ("tools/make_activity_pdf.py (this file)",
         "Generates this report."),
    ]
    tbl2 = Table(changes, colWidths=[4.5 * cm, 11.7 * cm])
    tbl2.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#1a2535")),
        ("TEXTCOLOR", (0, 0), (0, -1), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
    ]))
    elements.append(tbl2)
    elements.append(PageBreak())

    # ---------- Equity curve ----------
    elements.append(_p(H2, "3. Equity curve"))
    if Path(EQUITY_PNG).exists():
        elements.append(Image(EQUITY_PNG, width=17 * cm, height=9.5 * cm))
    else:
        elements.append(_p(body, "No equity data."))
    elements.append(Spacer(1, 0.3 * cm))

    # Quick stats
    if equity:
        balances = [r[1] for r in equity]
        peak = max(balances)
        trough = min(balances)
        start = balances[0]
        end = balances[-1]
        stats = [
            ["Start balance", f"${start:,.2f}"],
            ["End balance", f"${end:,.2f}"],
            ["Return", f"{(end - start) / start * 100:+.2f}%"],
            ["Peak balance", f"${peak:,.2f}"],
            ["Trough", f"${trough:,.2f}"],
            ["Max drawdown from peak", f"{(peak - trough) / peak * 100:.2f}%"],
        ]
        t = Table(stats, colWidths=[5 * cm, 4 * cm])
        t.setStyle(TableStyle([
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
            ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#0d1421")),
            ("TEXTCOLOR", (0, 0), (0, -1), colors.white),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
        ]))
        elements.append(t)
    elements.append(PageBreak())

    # ---------- Per-trade PnL chart + table ----------
    elements.append(_p(H2, "4. Paper trades — per-trade PnL"))
    if Path(PNL_PNG).exists():
        elements.append(Image(PNL_PNG, width=17 * cm, height=6 * cm))
    elements.append(Spacer(1, 0.2 * cm))

    # Trade table
    header = ["#", "Symbol", "Side", "Entry", "SL", "TP", "Exit", "PnL $", "Result", "Opened", "Closed"]
    rows = [header]
    for t in trades:
        rows.append([
            str(t["id"]),
            t["symbol"],
            t["side"],
            f"{t['entry_price']:.2f}",
            f"{t['sl_price']:.2f}",
            f"{t['tp_price']:.2f}",
            f"{t.get('exit_price', 0) or 0:.2f}",
            f"{t['pnl']:+.2f}",
            str(t.get("result", ""))[:14],
            str(t.get("opened_at", ""))[:16],
            str(t.get("closed_at", ""))[:16],
        ])
    tbl = Table(rows, colWidths=[0.7, 1.8, 1.0, 1.6, 1.6, 1.6, 1.6, 1.2, 1.7, 2.4, 2.4], repeatRows=1)
    tbl._argW = [c * cm for c in tbl._argW]
    ts = [
        ("FONTSIZE", (0, 0), (-1, -1), 7),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a2535")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("ALIGN", (3, 1), (7, -1), "RIGHT"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
    ]
    for i, t in enumerate(trades, start=1):
        ts.append(("BACKGROUND", (7, i), (7, i),
                   colors.HexColor("#d1fae5") if t["pnl"] > 0 else colors.HexColor("#fee2e2")))
    tbl.setStyle(TableStyle(ts))
    elements.append(tbl)

    # Aggregate stats
    if trades:
        wins = [t for t in trades if t["pnl"] > 0]
        losses = [t for t in trades if t["pnl"] <= 0]
        gw = sum(t["pnl"] for t in wins)
        gl = abs(sum(t["pnl"] for t in losses)) or 1e-9
        agg = [
            ["Total trades", str(len(trades))],
            ["Wins / losses", f"{len(wins)} / {len(losses)}"],
            ["Win rate", f"{len(wins) / len(trades) * 100:.1f}%"],
            ["Gross win $", f"${gw:+.2f}"],
            ["Gross loss $", f"${-gl:+.2f}"],
            ["Profit factor", f"{gw / gl:.2f}"],
            ["Avg win", f"${(gw / len(wins)) if wins else 0:+.2f}"],
            ["Avg loss", f"${-(gl / len(losses)) if losses else 0:+.2f}"],
        ]
        elements.append(Spacer(1, 0.25 * cm))
        t = Table(agg, colWidths=[4 * cm, 3.5 * cm])
        t.setStyle(TableStyle([
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
            ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#0d1421")),
            ("TEXTCOLOR", (0, 0), (0, -1), colors.white),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
        ]))
        elements.append(t)
    elements.append(PageBreak())

    # ---------- Signals summary ----------
    elements.append(_p(H2, "5. Signals — last 30"))
    rows = [["Created", "Symbol", "Side", "Score", "RR", "Regime", "Exec"]]
    for s in signals[-30:]:
        rows.append([
            s["created_at"][:16],
            s["symbol"],
            s["side"],
            str(s["score"]),
            f"{s['rr']:.2f}" if s["rr"] else "-",
            s.get("regime", "") or "",
            "Y" if s["executed"] else "n",
        ])
    tbl = Table(rows, colWidths=[3.8, 2.1, 1.0, 1.0, 1.0, 2.2, 0.8], repeatRows=1)
    tbl._argW = [c * cm for c in tbl._argW]
    tbl.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 7),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a2535")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
    ]))
    elements.append(tbl)
    elements.append(PageBreak())

    # ---------- Self-optimizer ----------
    elements.append(_p(H2, "6. Self-optimizer status"))
    elements.append(_p(body, (
        "The self-optimizer (Phase 5, <i>strategies/self_optimizer.py</i>) aggregates every "
        "closed trade's reasons + PnL, groups by tag / regime / symbol / score-bucket, and emits "
        "a weight multiplier (clipped 0.5-1.5). Buckets with N &lt; 10 trades get neutral weight 1.0. "
        "Runs nightly at 00:30 UTC via <i>bot._recompute_optimizer_weights</i>. "
        "<i>signal_generator.apply_weights_to_score</i> loads the file on mtime change."
    )))
    if weights:
        info = [
            ["Last updated", weights.get("updated_at", "")[:19]],
            ["Sample size", str(weights.get("sample_size", 0))],
            ["Baseline expectancy", f"${weights.get('baseline_expectancy', 0):+.2f}"],
            ["Tag weights", f"{len(weights.get('tag_weights', {}))} buckets"],
            ["Regime weights", f"{len(weights.get('regime_weights', {}))} buckets"],
            ["Symbol weights", f"{len(weights.get('symbol_weights', {}))} buckets"],
        ]
        t = Table(info, colWidths=[5 * cm, 6 * cm])
        t.setStyle(TableStyle([
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
            ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#0d1421")),
            ("TEXTCOLOR", (0, 0), (0, -1), colors.white),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
        ]))
        elements.append(t)

        sym_stats = weights.get("symbol_stats", {})
        if sym_stats:
            elements.append(Spacer(1, 0.3 * cm))
            elements.append(_p(H3, "Per-symbol stats (from optimizer)"))
            rows = [["Symbol", "N", "WR %", "Expectancy", "Total PnL"]]
            for sym, st in sym_stats.items():
                rows.append([
                    sym, str(st["n"]), f"{st['win_rate']:.1f}%",
                    f"${st['expectancy']:+.2f}", f"${st['total_pnl']:+.2f}",
                ])
            t = Table(rows, colWidths=[3, 1.2, 1.5, 2.2, 2.2])
            t._argW = [c * cm for c in t._argW]
            t.setStyle(TableStyle([
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a2535")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ]))
            elements.append(t)
    elements.append(PageBreak())

    # ---------- Known issues / next steps ----------
    elements.append(_p(H2, "7. Known findings &amp; next steps"))
    findings = [
        ("Finding", "Detail"),
        ("Directional thesis often correct, SL hit first",
         "On 2026-04-21 trades #13 (ETH short) and #14 (BTC short): price moved in intended direction, but SL fired on noise spike before TP1 was reached."),
        ("Regime-mult-before-ATR-floor bug (FIXED)",
         "Ranging regime SL×0.8 was shrinking SL back below the ATR floor (trade #14: floor 609.63 → actual SL 487.70 → hit on same cycle). Floor now re-enforced AFTER regime mult."),
        ("SOL still dominant loser",
         "Backtest 90d 1h: SOL 2W/8L, PF 0.25. Paper: consistent SL hits. Candidate fix: raise SOL min_score to 65+ or min_rr to 2.5, or disable until per-symbol optimizer accumulates enough samples."),
        ("Self-optimizer waiting for samples",
         "Needs 10 trades per bucket (tag × regime × symbol × score). Currently 12 total closed trades → only global-level stats meaningful; tag-level weights all neutral."),
        ("Backtester is pessimistic (by design)",
         "Stubs out liquidation, HVN, walls, news, order flow, manipulation. Real-world signal rate is 5-10× higher than backtest. Treat backtest PnL as worst-case floor."),
        ("LLM self-improvement loop (not built)",
         "Discussed: nightly analyst → improver that writes PR-style patches to data/proposed_patches → human /apply gate via Telegram. Deferred pending more paper trade data."),
    ]
    tbl = Table(findings, colWidths=[4.5 * cm, 11.7 * cm])
    tbl.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a2535")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
    ]))
    elements.append(tbl)

    elements.append(Spacer(1, 0.6 * cm))
    elements.append(_p(small, (
        "Generated by tools/make_activity_pdf.py. Data sources: "
        "data/trading_bot.db (signals, trades, equity), data/paper_trades.json (live paper state), "
        "data/optimizer_weights.json (self-optimizer output)."
    )))

    doc.build(elements)
    print(f"Wrote {OUT_PDF}")


if __name__ == "__main__":
    build_pdf()
