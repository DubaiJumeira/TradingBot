"""
ICT/Wyckoff Crypto Trading Bot
Main entry point — runs the analysis loop.

Two operating modes run in parallel:

    REGULAR MODE: every 5 minutes, analyze all symbols using the full
    ICT + Wyckoff + Market Data scoring pipeline.

    REACTIVE MODE (Phase 1D): every 30 seconds, poll the news aggregator
    for HIGH/CRITICAL events. When one fires, wait for the correlation-map
    delay (spike window), then run ICT analysis on the affected instruments
    with a strong directional bias from the news. This catches the
    POST-SPIKE retracement entry, not the spike itself.
"""

import os
import time
import logging
import pandas as pd
import schedule
from datetime import datetime, timezone

from config import Config, get_instrument
from exchange_handler import ExchangeHandler
from strategies.ict_strategy import analyze_ict
from strategies.wyckoff_strategy import analyze_wyckoff
from strategies.market_data import analyze_market_data
from strategies.signal_generator import generate_signal
from strategies.news_events import check_high_impact_events  # legacy fallback
from strategies.momentum_breakout import MomentumBreakoutStrategy
from utils.telegram_alerts import (
    alert_signal, alert_close, alert_stats,
    alert_error, alert_startup, send_message,
    alert_trade_status,
)

# Phase 4C: candlestick chart rendering for signal alerts (best-effort).
try:
    from utils.chart_builder import render_signal_chart
    _CHART_AVAILABLE = True
except Exception as _chart_exc:
    render_signal_chart = None  # type: ignore
    _CHART_AVAILABLE = False
    logger_init_chart_warn = _chart_exc

# Phase 4A interactive Telegram command bot — graceful fallback.
try:
    from utils.telegram_bot import TelegramCommandBot, register_all as register_telegram_commands
    _TELEGRAM_CMD_AVAILABLE = True
except ImportError as _exc:
    _TELEGRAM_CMD_AVAILABLE = False
    logging.getLogger("TradingBot").warning("Telegram command bot not available: %s", _exc)

# Phase 6 risk management — graceful fallback.
try:
    from strategies.risk_manager import (
        RiskManager, calculate_atr, atr_position_size, calculate_trailing_stop,
    )
    _RISK_AVAILABLE = True
except ImportError as _exc:
    _RISK_AVAILABLE = False
    logging.getLogger("TradingBot").warning("Risk manager not available: %s", _exc)

# Phase 11 order flow — graceful fallback.
try:
    from strategies.order_flow import OrderFlowTracker
    _ORDER_FLOW_AVAILABLE = True
except ImportError as _exc:
    _ORDER_FLOW_AVAILABLE = False
    logging.getLogger("TradingBot").warning("Order flow not available: %s", _exc)

# Phase 10 database — graceful fallback.
try:
    from database import TradeDB
    _DB_AVAILABLE = True
except ImportError as _exc:
    _DB_AVAILABLE = False
    logging.getLogger("TradingBot").warning("Database not available: %s", _exc)

# Phase 9 regime detection — graceful fallback.
try:
    from strategies.regime_detector import RegimeDetector
    _REGIME_AVAILABLE = True
except ImportError as _exc:
    _REGIME_AVAILABLE = False
    logging.getLogger("TradingBot").warning("Regime detector not available: %s", _exc)

# Phase 1 news engine — graceful fallback if unavailable.
try:
    from strategies.news import NewsAggregator
    from strategies.news_reactive import ReactiveNewsMonitor
    _NEWS_AVAILABLE = True
except ImportError as _exc:
    _NEWS_AVAILABLE = False
    logging.getLogger("TradingBot").warning("News engine not available: %s", _exc)

# Phase 1E economic calendar — graceful fallback.
try:
    from strategies.economic_calendar import EconomicCalendar
    _CALENDAR_AVAILABLE = True
except ImportError as _exc:
    _CALENDAR_AVAILABLE = False
    logging.getLogger("TradingBot").warning("Economic calendar not available: %s", _exc)

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler("data/bot.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("TradingBot")

# How often the reactive news check runs (seconds).
_REACTIVE_POLL_INTERVAL = 30


def _fmt_opt(v) -> str:
    """Format an optional float for log lines; 'NA' for None/NaN."""
    if v is None:
        return "NA"
    try:
        if pd.isna(v):
            return "NA"
    except (TypeError, ValueError):
        pass
    return f"{float(v):.2f}"


class TradingBot:
    def __init__(self):
        self.exchange = ExchangeHandler()
        self.last_signals: dict[str, datetime] = {}
        self.dry_run_news = os.getenv("NEWS_DRY_RUN", "true").lower() == "true"
        self._breaker_was_active = False

        # Phase 1 news engine.
        self.aggregator = None
        self.reactive_monitor = None
        if _NEWS_AVAILABLE:
            try:
                self.aggregator = NewsAggregator()
                self.reactive_monitor = ReactiveNewsMonitor(self.aggregator)
                logger.info("News engine + reactive monitor initialized")
            except Exception as exc:
                logger.warning("News engine init failed: %s — running without news", exc)

        # Phase 1E economic calendar.
        self.calendar = None
        if _CALENDAR_AVAILABLE:
            try:
                self.calendar = EconomicCalendar()
                self.calendar.refresh()
                logger.info("Economic calendar initialized (%d events)", self.calendar.event_count)
            except Exception as exc:
                logger.warning("Economic calendar init failed: %s", exc)

        # Phase 6 risk manager.
        self.risk_manager = None
        if _RISK_AVAILABLE:
            try:
                balance = (
                    self.exchange.paper.balance
                    if self.exchange.paper else Config.STARTING_BALANCE
                )
                self.risk_manager = RiskManager(starting_balance=balance)
                logger.info("Risk manager initialized (max_dd=%.1f%%)", Config.MAX_DRAWDOWN_PCT)
            except Exception as exc:
                logger.warning("Risk manager init failed: %s", exc)

        # Phase 9 regime detector.
        self.regime_detector = RegimeDetector() if _REGIME_AVAILABLE else None

        # Phase 10 database.
        self.db = None
        if _DB_AVAILABLE:
            try:
                self.db = TradeDB()
            except Exception as exc:
                logger.warning("TradeDB init failed: %s", exc)

        # Phase 11 order flow trackers (one per symbol).
        self.order_flow_trackers: dict[str, Any] = {}
        if _ORDER_FLOW_AVAILABLE:
            for sym in Config.SYMBOLS:
                self.order_flow_trackers[sym] = OrderFlowTracker(bar_seconds=60)

        # Phase 12: per-symbol MTF state (caches per-timeframe bias).
        from strategies.mtf_analysis import MTFState
        self.mtf_states: dict[str, MTFState] = {sym: MTFState() for sym in Config.SYMBOLS}

        # Phase 2: market manipulation trackers (one per symbol).
        self.manipulation_trackers: dict[str, Any] = {}
        try:
            from strategies.manipulation import ManipulationTracker
            for sym in Config.SYMBOLS:
                self.manipulation_trackers[sym] = ManipulationTracker(sym)
            logger.info("Manipulation trackers ready (%d symbols)", len(self.manipulation_trackers))
        except Exception as exc:
            logger.warning("Manipulation tracker init failed: %s", exc)

        # Phase 4A: interactive Telegram command bot.
        self.command_bot = None
        if _TELEGRAM_CMD_AVAILABLE:
            try:
                self.command_bot = TelegramCommandBot()
                register_telegram_commands(self)
                logger.info("Telegram command bot ready (%d commands)", len(self.command_bot._handlers))
            except Exception as exc:
                logger.warning("Telegram command bot init failed: %s", exc)

        self._last_reactive_check = datetime.min.replace(tzinfo=timezone.utc)

        # Strategy mode dispatch.
        self._momentum_strategy: MomentumBreakoutStrategy | None = None
        self._last_4h_signal_time: dict[str, pd.Timestamp] = {}
        # trade_id -> {"atr_at_entry": float, "highest_since_entry": float}
        self._momentum_position_state: dict[str, dict] = {}
        self._momentum_test_signal_fired = False
        if Config.STRATEGY_MODE == "momentum_breakout":
            self._momentum_strategy = MomentumBreakoutStrategy()
            logger.info("STRATEGY_MODE=momentum_breakout (legacy confluence bypassed)")
        else:
            logger.info("STRATEGY_MODE=%s", Config.STRATEGY_MODE)

        logger.info(
            "Bot initialized (news=%s, calendar=%s, risk=%s, regime=%s)",
            "enabled" if self.aggregator else "disabled",
            "enabled" if self.calendar else "disabled",
            "enabled" if self.risk_manager else "disabled",
            "enabled" if self.regime_detector else "disabled",
        )

    def ohlcv_to_df(self, ohlcv):
        """Convert OHLCV data to DataFrame."""
        df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)
        return df

    def _build_signal_chart(self, signal: dict) -> bytes | None:
        """Fetch fresh OHLCV and render a signal chart. Best-effort — any
        failure returns None and the alert falls back to text-only."""
        if not _CHART_AVAILABLE or render_signal_chart is None:
            return None
        try:
            ohlcv = self.exchange.fetch_ohlcv(signal["symbol"], "15m", limit=100)
            df = self.ohlcv_to_df(ohlcv)
            return render_signal_chart(df, signal)
        except Exception as exc:
            logger.warning("chart build failed for %s: %s", signal.get("symbol"), exc)
            return None

    # ==================================================================
    # REGULAR MODE — every 5 minutes, all symbols
    # ==================================================================

    def analyze_symbol(self, symbol: str, news_signal: dict | None = None):
        """
        Run full analysis on a single symbol.

        Parameters
        ----------
        news_signal : dict, optional
            From ReactiveAction.to_news_signal(). Passed through to
            signal_generator for news-enhanced scoring.
        """
        # Branch on strategy mode. Momentum breakout runs the new 4H pipeline;
        # legacy_confluence (default) runs the ICT/Wyckoff stack below.
        if Config.STRATEGY_MODE == "momentum_breakout":
            return self._analyze_momentum(symbol)

        try:
            # Look up per-instrument config (Phase 2).
            inst = get_instrument(symbol) or {}

            htf_ohlcv = self.exchange.fetch_ohlcv(symbol, "4h", limit=200)
            ltf_ohlcv = self.exchange.fetch_ohlcv(symbol, "15m", limit=300)

            htf_df = self.ohlcv_to_df(htf_ohlcv)
            ltf_df = self.ohlcv_to_df(ltf_ohlcv)

            # Phase 12: MTF confluence — pull 1D/1h and refresh cached bias
            # for the full 4-TF stack. MTFState only re-scores on new bars
            # so these calls are cheap after the first cycle.
            mtf_confluence: dict[str, Any] | None = None
            try:
                d1_df = self.ohlcv_to_df(self.exchange.fetch_ohlcv(symbol, "1d", limit=220))
                h1_df = self.ohlcv_to_df(self.exchange.fetch_ohlcv(symbol, "1h", limit=220))
                state = self.mtf_states[symbol]
                state.update("1D", d1_df)
                state.update("4h", htf_df)
                state.update("1h", h1_df)
                state.update("15m", ltf_df)
                mtf_confluence = state.confluence()
                logger.info(
                    "%s MTF: %s score=%+.2f aligned=%d/%d",
                    symbol, mtf_confluence["direction"],
                    mtf_confluence["score"], mtf_confluence["aligned_count"],
                    mtf_confluence["total_count"],
                )
            except Exception as exc:
                logger.warning("MTF fetch/confluence failed for %s: %s", symbol, exc)

            current_price = ltf_df.iloc[-1]["close"]

            # 1. ICT Analysis (lower TF for entries)
            ict = analyze_ict(ltf_df, current_price)

            # 2. Wyckoff Analysis (higher TF for bias)
            wyckoff = analyze_wyckoff(htf_df)

            # 3. Market Data (funding, OI, volume profile, kill zones)
            #    Instrument config controls funding/OI fetch and kill zone weights.
            market = analyze_market_data(
                self.exchange, symbol, ltf_df, Config.KILL_ZONES,
                instrument=inst,
                manipulation_tracker=self.manipulation_trackers.get(symbol),
            )

            # 4. Economic calendar check (Phase 1E — replaces legacy pause).
            if news_signal is None and self.calendar is not None:
                cal = self.calendar.check_events(symbol)
                if cal["block_trading"]:
                    logger.info(
                        "Economic event '%s' in %.0f min — blocking %s",
                        cal["event_name"], cal["minutes_until"] or 0, symbol,
                    )
                    # Send Telegram alert if this is the first time we're flagging it.
                    if cal["alert_event"] is not None:
                        evt = cal["alert_event"]
                        send_message(
                            f"⚠️ <b>{evt.name}</b> in {int(evt.minutes_until())} min — "
                            f"tightening stops, blocking new entries on "
                            f"{', '.join(evt.affected_assets)}"
                        )
                    return None

            # 5. Phase 9: regime detection.
            regime = None
            if self.regime_detector is not None:
                news_active = news_signal is not None and news_signal.get("impact") in ("high", "critical")
                regime = self.regime_detector.detect(ltf_df, news_event_active=news_active)
                logger.info(
                    "%s regime: %s (ADX=%.1f, wick=%.3f, vol=%.2f%%)",
                    symbol, regime["regime"], regime["adx"],
                    regime["wick_ratio"], regime["volatility_pct"],
                )

            # 6. Phase 11: order flow analysis (if available).
            order_flow = None
            tracker = self.order_flow_trackers.get(symbol)
            if tracker is not None:
                try:
                    order_flow = tracker.analyze()
                except Exception as exc:
                    logger.debug("Order flow analyze failed for %s: %s", symbol, exc)

            # 7. Generate signal (with optional news scoring from Phase 1D)
            #    Instrument config provides per-symbol min_rr and risk_pct.
            balance = self.exchange.paper.balance if self.exchange.paper else Config.STARTING_BALANCE
            signal = generate_signal(
                symbol, current_price, ict, wyckoff, market, balance,
                news_signal=news_signal,
                instrument=inst,
                regime=regime,
                order_flow=order_flow,
                ltf_df=ltf_df,
                mtf_confluence=mtf_confluence,
            )

            # 8. Phase 3: attach dynamic leverage, margin, liquidation price.
            if signal is not None:
                try:
                    from strategies.leverage import apply_leverage_to_signal
                    vol_pct = (regime or {}).get("volatility_pct", 2.0)
                    regime_name = (regime or {}).get("regime")
                    news_active = news_signal is not None and news_signal.get("impact") in ("high", "critical")
                    dd_pct = 0.0
                    if self.risk_manager is not None:
                        dd_pct = self.risk_manager.drawdown.drawdown_pct
                    apply_leverage_to_signal(
                        signal,
                        volatility_pct=vol_pct,
                        instrument=inst,
                        regime_name=regime_name,
                        news_active=news_active,
                        drawdown_pct=dd_pct,
                        balance=balance,
                    )
                except Exception as exc:
                    logger.warning("Leverage attach failed for %s: %s", symbol, exc)
                    signal.setdefault("leverage", 1)
                    signal.setdefault("margin_usd", signal.get("size_usd", 0))
                    signal.setdefault("liq_price", 0.0)

            return signal

        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {e}", exc_info=True)
            return None

    # ==================================================================
    # MOMENTUM BREAKOUT MODE
    # ==================================================================

    def _analyze_momentum(self, symbol: str) -> dict | None:
        """4H Donchian breakout pipeline. Returns a signal dict matching the
        legacy shape on entry, else None.

        Logs the per-condition diagnostics every cycle so it's visible whether
        the strategy is silent because of warmup, no breakout, or filter
        veto."""
        if self._momentum_strategy is None:
            return None
        try:
            # Need enough bars for SMA-200 plus ATR-median-50 warmup.
            ohlcv = self.exchange.fetch_ohlcv(symbol, "4h", limit=300)
            df = self.ohlcv_to_df(ohlcv)
            if df.index.tz is None:
                df.index = df.index.tz_localize("UTC")
            ind = self._momentum_strategy.compute_indicators(df)

            last_idx = len(ind) - 1
            last_ts = ind.index[last_idx]
            diag = self._momentum_strategy.entry_diagnostics(ind, last_idx)
            logger.info(
                "%s 4H entry_conditions: ok=%s uptrend=%s breakout=%s "
                "volatility=%s close=%.2f sma200=%s donch20=%s atr14=%s atr_med50=%s",
                symbol, diag.get("ok"),
                diag.get("cond_uptrend"), diag.get("cond_breakout"),
                diag.get("cond_volatility"), diag.get("close", 0.0),
                _fmt_opt(diag.get("sma_200")),
                _fmt_opt(diag.get("donchian_high_20")),
                _fmt_opt(diag.get("atr_14")),
                _fmt_opt(diag.get("atr_median_50")),
            )

            # Test-signal override: force a synthetic long once so we can
            # verify the Telegram + DB plumbing end-to-end.
            if Config.MOMENTUM_FORCE_TEST_SIGNAL and not self._momentum_test_signal_fired:
                self._momentum_test_signal_fired = True
                close = float(ind["close"].iloc[-1])
                atr = float(ind["atr_14"].iloc[-1]) if pd.notna(ind["atr_14"].iloc[-1]) else close * 0.01
                stop = close - self._momentum_strategy.atr_stop_mult * atr
                balance = self.exchange.paper.balance if self.exchange.paper else Config.STARTING_BALANCE
                size = self._momentum_strategy.position_size(balance, close, stop)
                logger.warning("MOMENTUM_FORCE_TEST_SIGNAL=true — emitting forced test signal for %s", symbol)
                return self._momentum_signal_dict(symbol, close, stop, atr, size, forced=True)

            # Same-bar dedupe: don't re-emit on the same 4H candle.
            last_acted = self._last_4h_signal_time.get(symbol)
            if last_acted is not None and last_ts <= last_acted:
                return None

            if not diag.get("ok"):
                return None

            close = float(ind["close"].iloc[-1])
            atr = float(ind["atr_14"].iloc[-1])
            stop = close - self._momentum_strategy.atr_stop_mult * atr
            balance = self.exchange.paper.balance if self.exchange.paper else Config.STARTING_BALANCE
            size = self._momentum_strategy.position_size(balance, close, stop)
            if size <= 0:
                logger.info("%s momentum: position_size <= 0 (balance=%.2f, atr=%.2f) — skipping", symbol, balance, atr)
                return None

            self._last_4h_signal_time[symbol] = last_ts
            return self._momentum_signal_dict(symbol, close, stop, atr, size)

        except Exception as exc:
            logger.error("momentum analyze failed for %s: %s", symbol, exc, exc_info=True)
            return None

    def _momentum_signal_dict(
        self,
        symbol: str,
        entry: float,
        stop: float,
        atr: float,
        size_usd: float,
        forced: bool = False,
    ) -> dict:
        # No fixed TP — set a sentinel far above so existing TP-hit logic
        # never triggers. Exit handled by trailing stop + SMA-50 break in
        # `_check_momentum_exits`.
        tp_sentinel = entry + 1000.0 * atr if atr > 0 else entry * 100.0
        return {
            "symbol": symbol,
            "side": "long",
            "entry": round(entry, 2),
            "sl": round(stop, 2),
            "tp": round(tp_sentinel, 2),
            "size_usd": size_usd,
            "score": 100,
            "leverage": 1,
            "margin_usd": size_usd,
            "liq_price": 0.0,
            "tp_plan": None,
            "reasons": ["momentum_breakout"] + (["forced_test"] if forced else []),
            "strategy_mode": "momentum_breakout",
            "atr_at_entry": atr,
        }

    def check_exits(self):
        """Check trailing stops, then check if any SL or TP has been hit."""
        if not self.exchange.paper:
            return

        current_prices = {}
        for pos in self.exchange.paper.positions.values():
            sym = pos["symbol"]
            if sym not in current_prices:
                try:
                    ticker = self.exchange.fetch_ticker(sym)
                    current_prices[sym] = ticker["last"]
                except Exception as e:
                    logger.error(f"Could not fetch price for {sym}: {e}")

        # Momentum mode: replace the structure-based trailing stop with the
        # strategy's 3*ATR trail, and force-close on close-below-50-SMA.
        if Config.STRATEGY_MODE == "momentum_breakout" and self._momentum_strategy is not None:
            self._check_momentum_exits(current_prices)
            return

        # Phase 6: structure-based trailing stop management.
        # Fetch 15m candles per symbol (once) for swing-point detection.
        trail_dfs: dict[str, Any] = {}
        if self.risk_manager is not None:
            for tid, pos in self.exchange.paper.positions.items():
                sym = pos["symbol"]
                price = current_prices.get(sym)
                if price is None:
                    continue
                if sym not in trail_dfs:
                    try:
                        ohlcv = self.exchange.fetch_ohlcv(sym, "15m", limit=100)
                        trail_dfs[sym] = self.ohlcv_to_df(ohlcv)
                    except Exception:
                        trail_dfs[sym] = None
                new_sl = calculate_trailing_stop(
                    side=pos["side"],
                    entry=pos["entry_price"],
                    current_sl=pos["sl_price"],
                    current_price=price,
                    tp=pos["tp_price"],
                    df=trail_dfs.get(sym),
                )
                if new_sl is not None:
                    self.exchange.paper.update_sl(tid, new_sl)

        events = self.exchange.paper.check_positions(current_prices)
        for event in events:
            if event.get("partial"):
                # Partial TP fill — alert it separately, no trackers update.
                try:
                    from utils.telegram_alerts import alert_partial_fill
                    alert_partial_fill(event)
                except Exception as exc:
                    logger.debug("Partial alert failed: %s", exc)
                continue

            # Full close.
            alert_close(event)
            if self.risk_manager is not None and self.exchange.paper:
                self.risk_manager.record_trade_close(
                    pnl=event.get("pnl", 0) or 0,
                    equity=self.exchange.paper.balance,
                )
            # Phase 10: DB update.
            if self.db is not None:
                try:
                    self.db.close_trade(
                        trade_id=str(event.get("id")),
                        exit_price=event.get("exit_price", 0),
                        pnl=event.get("pnl", 0) or 0,
                        result=event.get("result", "?"),
                    )
                except Exception as exc:
                    logger.debug("DB close update failed: %s", exc)

    def _check_momentum_exits(self, current_prices: dict[str, float]) -> None:
        """Momentum-mode exit handler.

        For each open position:
          1. Refresh `highest_since_entry` from current price.
          2. Set the position's SL to `highest_since_entry - 3 * ATR_at_entry`
             (only moves up, never down) — this is the 3xATR trailing stop.
          3. Fetch 4H candles and force-close on close < SMA-50.
        Then run the standard paper.check_positions() to fire the SL if hit.
        """
        # Refresh trailing stops first.
        per_symbol_4h: dict[str, Any] = {}
        for tid, pos in list(self.exchange.paper.positions.items()):
            sym = pos["symbol"]
            price = current_prices.get(sym)
            if price is None:
                continue

            state = self._momentum_position_state.get(str(tid))
            if state is None:
                # Position opened without our state tracking — bootstrap from
                # what we can recover. Use entry as the initial high and a
                # rough ATR estimate (entry - sl) / 3.
                atr_at_entry = max(1e-9, (pos["entry_price"] - pos["sl_price"]) / 3.0)
                state = {
                    "atr_at_entry": atr_at_entry,
                    "highest_since_entry": max(pos["entry_price"], price),
                }
                self._momentum_position_state[str(tid)] = state

            if price > state["highest_since_entry"]:
                state["highest_since_entry"] = price

            new_sl = state["highest_since_entry"] - self._momentum_strategy.atr_stop_mult * state["atr_at_entry"]
            if new_sl > pos["sl_price"]:
                self.exchange.paper.update_sl(tid, round(new_sl, 2))

            # 4H SMA-50 break check — only once per symbol per cycle.
            if sym not in per_symbol_4h:
                try:
                    ohlcv = self.exchange.fetch_ohlcv(sym, "4h", limit=120)
                    df = self.ohlcv_to_df(ohlcv)
                    if df.index.tz is None:
                        df.index = df.index.tz_localize("UTC")
                    per_symbol_4h[sym] = self._momentum_strategy.compute_indicators(df)
                except Exception as exc:
                    logger.debug("4h fetch for SMA-50 check failed (%s): %s", sym, exc)
                    per_symbol_4h[sym] = None

            ind = per_symbol_4h.get(sym)
            if ind is not None and not ind.empty:
                last_close = float(ind["close"].iloc[-1])
                last_sma50 = ind["sma_50"].iloc[-1]
                if pd.notna(last_sma50) and last_close < float(last_sma50):
                    # Force-close at current market.
                    logger.info("%s momentum exit: close < SMA-50 (%.2f < %.2f) — closing",
                                sym, last_close, float(last_sma50))
                    closed = self.exchange.paper.close_manual(tid, price)
                    if closed is not None:
                        closed.setdefault("result", "sma_50_break")
                        alert_close(closed)
                        if self.risk_manager is not None:
                            self.risk_manager.record_trade_close(
                                pnl=closed.get("pnl", 0) or 0,
                                equity=self.exchange.paper.balance,
                            )
                        if self.db is not None:
                            try:
                                self.db.close_trade(
                                    trade_id=str(tid),
                                    exit_price=closed.get("exit_price", price),
                                    pnl=closed.get("pnl", 0) or 0,
                                    result="sma_50_break",
                                )
                            except Exception as exc:
                                logger.debug("DB close update failed: %s", exc)
                        self._momentum_position_state.pop(str(tid), None)

        # Standard SL/TP scan (catches the trailing SL we just updated).
        events = self.exchange.paper.check_positions(current_prices)
        for event in events:
            if event.get("partial"):
                continue
            alert_close(event)
            if self.risk_manager is not None:
                self.risk_manager.record_trade_close(
                    pnl=event.get("pnl", 0) or 0,
                    equity=self.exchange.paper.balance,
                )
            if self.db is not None:
                try:
                    self.db.close_trade(
                        trade_id=str(event.get("id")),
                        exit_price=event.get("exit_price", 0),
                        pnl=event.get("pnl", 0) or 0,
                        result=event.get("result", "?"),
                    )
                except Exception as exc:
                    logger.debug("DB close update failed: %s", exc)
            self._momentum_position_state.pop(str(event.get("id")), None)

    def run_cycle(self):
        """Run one full REGULAR analysis cycle across all symbols."""
        logger.info("=" * 50)
        logger.info(f"Starting analysis cycle at {datetime.now(timezone.utc).isoformat()}")

        # Exits still run while paused — we don't want stuck positions if /pause is left on.
        self.check_exits()

        if self.command_bot is not None and self.command_bot.is_paused:
            logger.info("Bot paused via Telegram — skipping new entries this cycle")
            return

        # Phase 6: update equity for drawdown tracking.
        if self.risk_manager is not None and self.exchange.paper:
            self.risk_manager.update_equity(self.exchange.paper.balance)
            breaker_active = self.risk_manager.drawdown.is_breaker_active
            if breaker_active:
                dd = self.risk_manager.drawdown
                logger.warning(
                    "CIRCUIT BREAKER ACTIVE — drawdown %.1f%% (peak=%.2f, now=%.2f). "
                    "Skipping all entries.",
                    dd.drawdown_pct, dd.peak_equity, dd.current_equity,
                )
                # Edge-trigger: alert only on the first cycle the breaker
                # engages. If there are open trades, send a one-shot trade
                # status snapshot instead of the breaker notification so
                # we don't spam the chat each cycle.
                if not self._breaker_was_active:
                    open_positions = list(self.exchange.paper.positions.values())
                    if open_positions:
                        alert_trade_status(
                            open_positions,
                            balance=self.exchange.paper.balance,
                            drawdown_pct=dd.drawdown_pct,
                            note="Circuit breaker engaged — no new entries",
                        )
                    else:
                        send_message(
                            f"<b>🔴 Circuit breaker active</b>\n"
                            f"<pre>  Drawdown   {dd.drawdown_pct:.1f}%\n"
                            f"  Peak       ${dd.peak_equity:,.0f}\n"
                            f"  Current    ${dd.current_equity:,.0f}</pre>"
                        )
                self._breaker_was_active = True
                return
            # Reset-edge: breaker cleared — notify once.
            if self._breaker_was_active and not breaker_active:
                send_message("<b>🟢 Circuit breaker cleared</b> — trading resumed.")
            self._breaker_was_active = breaker_active

        # Refresh news aggregator once per regular cycle.
        if self.aggregator is not None:
            try:
                self.aggregator.fetch_all()
            except Exception as exc:
                logger.warning("News aggregator fetch failed: %s", exc)

        open_count = len(self.exchange.paper.positions) if self.exchange.paper else 0
        if open_count >= Config.MAX_OPEN_TRADES:
            logger.info(f"Max open trades reached ({open_count}/{Config.MAX_OPEN_TRADES}). Skipping.")
            return

        for symbol in Config.SYMBOLS:
            signal = self.analyze_symbol(symbol)

            if signal:
                last = self.last_signals.get(symbol)
                if last and (datetime.now(timezone.utc) - last).seconds < 3600:
                    logger.info(f"Signal for {symbol} too recent — cooling off")
                    continue

                # Phase 6: pre-trade risk checks (correlation exposure).
                if self.risk_manager is not None and self.exchange.paper:
                    risk_check = self.risk_manager.pre_trade_check(
                        symbol, self.exchange.paper.positions,
                        side=signal["side"],
                    )
                    if not risk_check["allowed"]:
                        for reason in risk_check["reasons"]:
                            logger.info("RISK BLOCKED %s: %s", symbol, reason)
                        continue

                trade = self.exchange.place_order(
                    symbol=signal["symbol"],
                    side=signal["side"],
                    entry=signal["entry"],
                    sl=signal["sl"],
                    tp=signal["tp"],
                    size_usd=signal["size_usd"],
                    leverage=signal.get("leverage", 1),
                    margin_usd=signal.get("margin_usd"),
                    liq_price=signal.get("liq_price", 0.0),
                    tp_plan=signal.get("tp_plan"),
                )

                # Momentum mode: stash ATR-at-entry + initial highest for the
                # trailing-stop trail to use later.
                if signal.get("strategy_mode") == "momentum_breakout":
                    tid = trade.get("id") if isinstance(trade, dict) else trade
                    self._momentum_position_state[str(tid)] = {
                        "atr_at_entry": float(signal.get("atr_at_entry", 0.0)),
                        "highest_since_entry": float(signal["entry"]),
                    }

                # Phase 10: persist to DB.
                if self.db is not None:
                    try:
                        self.db.insert_signal(signal, executed=True)
                        self.db.insert_trade({**signal, "trade_id": str(trade.get("id") if isinstance(trade, dict) else trade)})
                    except Exception as exc:
                        logger.warning("DB persist failed: %s", exc)

                alert_signal(signal, chart_bytes=self._build_signal_chart(signal))
                self.last_signals[symbol] = datetime.now(timezone.utc)

                open_count += 1
                if open_count >= Config.MAX_OPEN_TRADES:
                    break

        # Phase 10: record equity snapshot at end of cycle.
        if self.db is not None and self.exchange.paper:
            try:
                dd_pct = 0
                if self.risk_manager is not None:
                    dd_pct = self.risk_manager.drawdown.drawdown_pct
                self.db.record_equity(
                    balance=self.exchange.paper.balance,
                    drawdown_pct=dd_pct,
                )
            except Exception as exc:
                logger.warning("DB equity record failed: %s", exc)

    # ==================================================================
    # REACTIVE MODE (Phase 1D) — every 30 seconds, news-triggered only
    # ==================================================================

    def run_reactive_check(self):
        """
        Poll the reactive monitor for news events whose delay has expired.

        For each ready event, run analyze_symbol on the affected instruments
        with the news_signal bias. This finds post-spike ICT setups (FVG fill,
        liquidity sweep retracement) aligned with the news direction.
        """
        if self.reactive_monitor is None:
            return
        if self.command_bot is not None and self.command_bot.is_paused:
            return

        now = datetime.now(tz=timezone.utc)
        if (now - self._last_reactive_check).total_seconds() < _REACTIVE_POLL_INTERVAL:
            return
        self._last_reactive_check = now

        try:
            actions = self.reactive_monitor.check()
        except Exception as exc:
            logger.warning("Reactive monitor check failed: %s", exc)
            return

        if not actions:
            return

        if self.dry_run_news:
            for action in actions:
                logger.info(
                    "[DRY RUN] REACTIVE would analyze %s (%s, %s) — %s",
                    action.asset, action.direction, action.impact_level.value,
                    action.event_title[:60],
                )
            return

        logger.info("REACTIVE: %d actions from news events", len(actions))

        open_count = len(self.exchange.paper.positions) if self.exchange.paper else 0

        for action in actions:
            if open_count >= Config.MAX_OPEN_TRADES:
                logger.info("Max open trades reached — skipping remaining reactive actions")
                break

            symbol = action.asset
            news_sig = action.to_news_signal()

            logger.info(
                "REACTIVE: analyzing %s (dir=%s, impact=%s) — %s",
                symbol, action.direction, action.impact_level.value,
                action.event_title[:60],
            )

            signal = self.analyze_symbol(symbol, news_signal=news_sig)

            if signal:
                last = self.last_signals.get(symbol)
                if last and (datetime.now(timezone.utc) - last).seconds < 1800:
                    logger.info(f"REACTIVE: {symbol} signal too recent — cooling off")
                    continue

                # Phase 6: pre-trade risk checks (correlation + drawdown).
                if self.risk_manager is not None and self.exchange.paper:
                    risk_check = self.risk_manager.pre_trade_check(
                        symbol, self.exchange.paper.positions,
                        side=signal["side"],
                    )
                    if not risk_check["allowed"]:
                        for reason in risk_check["reasons"]:
                            logger.info("RISK BLOCKED REACTIVE %s: %s", symbol, reason)
                        continue

                trade = self.exchange.place_order(
                    symbol=signal["symbol"],
                    side=signal["side"],
                    entry=signal["entry"],
                    sl=signal["sl"],
                    tp=signal["tp"],
                    size_usd=signal["size_usd"],
                    leverage=signal.get("leverage", 1),
                    margin_usd=signal.get("margin_usd"),
                    liq_price=signal.get("liq_price", 0.0),
                    tp_plan=signal.get("tp_plan"),
                )

                # Tag Telegram alert as news-triggered.
                signal["_reactive"] = True
                alert_signal(signal, chart_bytes=self._build_signal_chart(signal))
                self.last_signals[symbol] = datetime.now(timezone.utc)
                open_count += 1

    # ==================================================================
    # ECONOMIC CALENDAR (Phase 1E)
    # ==================================================================

    def _recompute_optimizer_weights(self):
        """Phase 5: nightly recompute of self-optimizer weights.

        Writes data/optimizer_weights.json. The signal generator reloads
        on mtime change, so the next cycle picks up fresh weights for
        free. No-op until ≥ 10 closed trades exist.
        """
        if self.db is None:
            return
        try:
            from strategies.self_optimizer import PerformanceAnalyzer
            pa = PerformanceAnalyzer(self.db)
            weights = pa.persist_weights(lookback_days=None)  # All historical trades
            n = weights.get("sample_size", 0)
            if n >= 10:
                send_message(
                    f"🧠 <b>Self-optimizer updated</b>\n"
                    f"  Sample size: {n} trades (all history)\n"
                    f"  Baseline expectancy: ${weights.get('baseline_expectancy', 0):+.2f}\n"
                    f"  Use /performance to see the breakdown."
                )
        except Exception as exc:
            logger.warning("Optimizer recompute failed: %s", exc)

    def _refresh_calendar(self):
        """Daily refresh of the economic calendar (scheduled at 05:00 UTC)."""
        if self.calendar is None:
            return
        try:
            self.calendar.refresh()
            upcoming = self.calendar.upcoming(hours=24)
            if upcoming:
                lines = [f"📅 <b>Today's economic events:</b>"]
                for evt in upcoming:
                    mins = int(evt.minutes_until())
                    lines.append(
                        f"  • <b>{evt.name}</b> ({evt.impact.value}) — "
                        f"in {mins // 60}h {mins % 60}m — "
                        f"affects {', '.join(evt.affected_assets)}"
                    )
                send_message("\n".join(lines))
            logger.info("Economic calendar refreshed: %d events", self.calendar.event_count)
        except Exception as exc:
            logger.warning("Calendar refresh failed: %s", exc)

    # ==================================================================
    # DAILY STATS
    # ==================================================================

    def send_daily_stats(self):
        """Send daily performance summary."""
        if self.exchange.paper:
            stats = self.exchange.paper.get_stats()
            alert_stats(stats)
            logger.info(f"Daily stats: {stats}")

    # ==================================================================
    # MAIN LOOP
    # ==================================================================

    def run(self):
        """Main bot loop — regular cycle + reactive news polling."""
        os.makedirs("data", exist_ok=True)

        mode = "PAPER" if Config.PAPER_TRADING else "LIVE"
        logger.info(f"Starting bot in {mode} mode")
        logger.info(f"Symbols: {Config.SYMBOLS}")
        logger.info(f"Min R:R: {Config.MIN_RR_RATIO}")
        logger.info(f"News engine: {'enabled' if self.aggregator else 'disabled'}")
        logger.info(f"News dry run: {self.dry_run_news}")
        logger.info(f"Economic calendar: {'enabled' if self.calendar else 'disabled'}")
        logger.info(f"Risk manager: {'enabled' if self.risk_manager else 'disabled'}")
        logger.info(f"Regime detector: {'enabled' if self.regime_detector else 'disabled'}")

        alert_startup()

        # Start real-time liquidation stream ingestion (Binance + Bybit).
        # Runs in a daemon thread and populates the heatmap buffer so that
        # the first 5-min cycle already has some observed data — without
        # this, early cycles fall back to the synthetic estimator.
        try:
            from strategies.liquidation_stream import ensure_stream_started
            ensure_stream_started()
        except Exception as exc:
            logger.warning("Liquidation stream unavailable: %s", exc)

        # Phase 4A: start interactive command polling in the background.
        if self.command_bot is not None:
            self.command_bot.start()

        # Schedule regular analysis every 5 minutes.
        schedule.every(5).minutes.do(self.run_cycle)

        # Daily stats at midnight UTC.
        schedule.every().day.at("00:00").do(self.send_daily_stats)

        # Refresh economic calendar daily at 05:00 UTC (before London open).
        if self.calendar is not None:
            schedule.every().day.at("05:00").do(self._refresh_calendar)

        # Phase 5: recompute self-optimizer weights daily at 00:30 UTC.
        if self.db is not None:
            schedule.every().day.at("00:30").do(self._recompute_optimizer_weights)

        # Dump 24h observed liquidation events every 30s so the dashboard
        # (separate process) can overlay them on the heatmap.
        try:
            from strategies.liquidation_stream import dump_recent_events
            schedule.every(30).seconds.do(dump_recent_events)
        except Exception as exc:
            logger.debug("Could not schedule liquidation dump: %s", exc)

        # Run first cycle immediately.
        self.run_cycle()

        while True:
            try:
                schedule.run_pending()
                # Reactive news check runs every ~30s (gated inside the method).
                self.run_reactive_check()
                time.sleep(10)
            except KeyboardInterrupt:
                logger.info("Bot stopped by user")
                send_message("🛑 <b>Bot stopped</b>")
                break
            except Exception as e:
                logger.error(f"Loop error: {e}", exc_info=True)
                alert_error(str(e))
                time.sleep(60)


if __name__ == "__main__":
    bot = TradingBot()
    bot.run()
