import ccxt
import logging
import json
import os
from datetime import datetime
from config import Config

logger = logging.getLogger(__name__)


class PaperTrader:
    """Simulates trades with virtual money."""

    def __init__(self, starting_balance: float):
        self.balance = starting_balance
        self.positions = {}
        self.trade_history = []
        self.trade_id = 0
        self._load_state()

    def _state_file(self):
        return "data/paper_trades.json"

    def _load_state(self):
        path = self._state_file()
        if os.path.exists(path):
            with open(path) as f:
                state = json.load(f)
            self.balance = state.get("balance", self.balance)
            self.positions = state.get("positions", {})
            self.trade_history = state.get("trade_history", [])
            self.trade_id = state.get("trade_id", 0)
            logger.info(f"Loaded paper state: balance={self.balance:.2f}, open={len(self.positions)}")

    def _save_state(self):
        os.makedirs("data", exist_ok=True)
        with open(self._state_file(), "w") as f:
            json.dump({
                "balance": self.balance,
                "positions": self.positions,
                "trade_history": self.trade_history,
                "trade_id": self.trade_id,
            }, f, indent=2, default=str)

    def open_trade(
        self,
        symbol,
        side,
        entry_price,
        sl_price,
        tp_price,
        size_usd,
        leverage: int = 1,
        margin_usd: float | None = None,
        liq_price: float = 0.0,
        tp_plan: dict | None = None,
    ):
        self.trade_id += 1
        # With leverage, only the margin is locked up. size_usd stays as
        # the notional exposure so PnL math on close remains consistent.
        if margin_usd is None:
            margin_usd = round(size_usd / max(leverage, 1), 2)
        trade = {
            "id": self.trade_id,
            "symbol": symbol,
            "side": side,
            "entry_price": entry_price,
            "sl_price": sl_price,
            "tp_price": tp_price,
            "size_usd": size_usd,
            "original_size_usd": size_usd,  # for partial-exit bookkeeping
            "margin_usd": margin_usd,
            "original_margin_usd": margin_usd,
            "leverage": leverage,
            "liq_price": liq_price,
            "qty": size_usd / entry_price,
            "original_qty": size_usd / entry_price,
            "tp_plan": tp_plan,  # dict form of PartialTPPlan (or None → single TP)
            "realised_pnl": 0.0,  # accumulated from partial fills
            "opened_at": datetime.utcnow().isoformat(),
            "status": "open",
        }
        self.positions[str(self.trade_id)] = trade
        # NOTE: On real exchanges (Bybit, Binance), margin is borrowed and does NOT
        # deduct from balance. Only unrealized PnL affects equity. The balance stays
        # unchanged when opening a position. This mirrors actual exchange behavior.
        self._save_state()
        logger.info(
            f"PAPER OPEN #{self.trade_id}: {side} {symbol} @ {entry_price} "
            f"| {leverage}× | margin=${margin_usd:.2f} | SL={sl_price} TP={tp_price}"
        )
        return trade

    def update_sl(self, trade_id: str | int, new_sl: float) -> bool:
        """
        Move the stop-loss on an open position (used by trailing stop).

        Returns True if the SL was updated, False if the position wasn't found.
        """
        tid = str(trade_id)
        pos = self.positions.get(tid)
        if pos is None:
            return False

        old_sl = pos["sl_price"]
        pos["sl_price"] = new_sl
        self._save_state()
        logger.info(
            "PAPER TRAIL #{}: SL moved %.2f → %.2f".replace("%.2f → %.2f", f"{old_sl:.2f} → {new_sl:.2f}"),
            pos["id"],
        )
        return True

    def close_manual(self, trade_id: str | int, current_price: float) -> dict | None:
        """
        Force-close a position at the given market price (used by /close and /closeall).
        Returns the closed trade dict, or None if the id wasn't found.
        """
        tid = str(trade_id)
        pos = self.positions.get(tid)
        if pos is None:
            return None

        if pos["side"] == "long":
            pnl = (current_price - pos["entry_price"]) * pos["qty"]
        else:
            pnl = (pos["entry_price"] - current_price) * pos["qty"]

        pos["exit_price"] = current_price
        pos["pnl"] = round(pnl, 2)
        pos["result"] = "MANUAL"
        pos["closed_at"] = datetime.utcnow().isoformat()
        pos["status"] = "closed"

        # Only credit PnL to balance. Margin is borrowed, not deducted.
        self.balance += pnl
        self.trade_history.append(pos)
        del self.positions[tid]
        self._save_state()
        logger.info(f"PAPER MANUAL CLOSE #{pos['id']}: PnL={pnl:+.2f} @ {current_price}")
        return pos

    def check_positions(self, current_prices: dict):
        """Check SLs, partial TPs, and full exits.

        Returns a list of events, each being either a fully-closed
        position dict (legacy shape) or a partial-fill event dict with
        a ``partial`` flag. Callers that only care about closed trades
        can filter with ``[e for e in events if not e.get("partial")]``.
        """
        events: list[dict] = []
        for tid, pos in list(self.positions.items()):
            price = current_prices.get(pos["symbol"])
            if price is None:
                continue

            side = pos["side"]
            entry = pos["entry_price"]

            # ---- SL check first — stops ALWAYS take precedence ----
            if (side == "long" and price <= pos["sl_price"]) or \
               (side == "short" and price >= pos["sl_price"]):
                # Distinguish a pure SL from a runner-SL (partials filled
                # first, then the remainder hit the trailed/BE stop).
                # Previously both were labelled "SL", hiding +R outcomes.
                plan = pos.get("tp_plan") or {}
                tps_filled = sum(1 for lvl in plan.get("levels", []) if lvl.get("filled"))
                result_label = f"SL_RUNNER_TP{tps_filled}" if tps_filled > 0 else "SL"
                self._close_full(pos, tid, pos["sl_price"], result_label)
                events.append(pos)
                continue

            # ---- Partial TP plan ----
            plan = pos.get("tp_plan")
            if plan and plan.get("levels"):
                fired = False
                for idx, lvl in enumerate(plan["levels"]):
                    if lvl.get("filled"):
                        continue
                    lvl_price = lvl["price"]
                    hit = (side == "long" and price >= lvl_price) or \
                          (side == "short" and price <= lvl_price)
                    if not hit:
                        break  # levels are ordered — don't skip ahead
                    fired = True
                    is_last = idx == len(plan["levels"]) - 1
                    if is_last:
                        # Final rung closes the remainder in one go.
                        self._close_full(pos, tid, lvl_price, f"TP{idx+1}")
                        events.append(pos)
                        lvl["filled"] = True
                        break
                    # Partial exit of close_pct of the ORIGINAL position.
                    partial_event = self._close_partial(
                        pos, lvl_price, lvl["close_pct"], f"TP{idx+1}",
                    )
                    lvl["filled"] = True
                    events.append(partial_event)
                    # Post-action: breakeven+ or trail.
                    if lvl.get("post_action") == "breakeven":
                        # Lock in 25% of the TP1 distance rather than flat BE.
                        tp1_dist = abs(lvl_price - entry)
                        lock_in = tp1_dist * 0.25
                        if side == "long":
                            new_sl = round(entry + lock_in, 2)
                        else:
                            new_sl = round(entry - lock_in, 2)
                        pos["sl_price"] = new_sl
                        logger.info(
                            "PAPER BE+ #%s: SL moved to %.2f (entry + 25%% of TP1 dist) after TP%d",
                            pos["id"], new_sl, idx + 1,
                        )
                if fired:
                    continue

            # ---- Legacy single-TP fallback (plan is None) ----
            if not plan:
                hit_tp = (side == "long" and price >= pos["tp_price"]) or \
                         (side == "short" and price <= pos["tp_price"])
                if hit_tp:
                    self._close_full(pos, tid, pos["tp_price"], "TP")
                    events.append(pos)

        if events:
            self._save_state()
        return events

    # ------------------------------------------------------------------
    # Partial-exit helpers (Phase 6)
    # ------------------------------------------------------------------

    def _close_full(self, pos: dict, tid: str, exit_price: float, result: str) -> None:
        """Fully close a position, credit balance + margin + pnl."""
        side = pos["side"]
        entry = pos["entry_price"]
        # PnL on the REMAINING qty only. Partial fills have already
        # credited their share into realised_pnl + balance.
        remaining_qty = pos["qty"]
        if side == "long":
            pnl = (exit_price - entry) * remaining_qty
        else:
            pnl = (entry - exit_price) * remaining_qty

        pos["exit_price"] = exit_price
        pos["pnl"] = round(pos.get("realised_pnl", 0.0) + pnl, 2)
        pos["result"] = result
        pos["closed_at"] = datetime.utcnow().isoformat()
        pos["status"] = "closed"

        # Only credit PnL to balance. Margin is borrowed and not deducted
        # from balance on real exchanges — it's just a utilization metric.
        self.balance += pnl
        self.trade_history.append(pos)
        self.positions.pop(tid, None)
        logger.info(
            "PAPER CLOSE #%s: %s | leg PnL=%+.2f | total PnL=%+.2f @ %.2f",
            pos["id"], result, pnl, pos["pnl"], exit_price,
        )

    def _close_partial(
        self,
        pos: dict,
        exit_price: float,
        close_pct: float,
        label: str,
    ) -> dict:
        """Close ``close_pct`` of the ORIGINAL position at ``exit_price``.

        Updates size_usd / qty / margin_usd in place; accumulates pnl
        into ``realised_pnl``; credits margin + pnl to the balance.
        Returns a partial-fill event dict for alerting.
        """
        side = pos["side"]
        entry = pos["entry_price"]
        original_qty = pos.get("original_qty", pos["qty"])
        partial_qty = original_qty * close_pct

        if side == "long":
            pnl = (exit_price - entry) * partial_qty
        else:
            pnl = (entry - exit_price) * partial_qty
        pnl = round(pnl, 2)

        # Proportionally return margin for the closed slice.
        original_margin = pos.get("original_margin_usd", pos.get("margin_usd", 0))
        margin_released = round(original_margin * close_pct, 2)

        pos["qty"] = max(0.0, pos["qty"] - partial_qty)
        pos["size_usd"] = round(pos["qty"] * entry, 2)
        pos["margin_usd"] = max(0.0, round(pos.get("margin_usd", 0) - margin_released, 2))
        pos["realised_pnl"] = round(pos.get("realised_pnl", 0.0) + pnl, 2)

        # Only credit PnL to balance. Margin is borrowed, not deducted from balance.
        self.balance += pnl
        logger.info(
            "PAPER PARTIAL #%s: %s %.0f%% @ %.2f | PnL=%+.2f | remaining qty=%.6f",
            pos["id"], label, close_pct * 100, exit_price, pnl, pos["qty"],
        )
        return {
            "partial": True,
            "id": pos["id"],
            "symbol": pos["symbol"],
            "side": side,
            "label": label,
            "exit_price": exit_price,
            "pnl": pnl,
            "close_pct": close_pct,
            "remaining_qty": pos["qty"],
        }

    def get_stats(self):
        if not self.trade_history:
            return {"total": 0, "win_rate": 0, "total_pnl": 0, "balance": self.balance}
        wins = sum(1 for t in self.trade_history if t["pnl"] > 0)
        total_pnl = sum(t["pnl"] for t in self.trade_history)
        return {
            "total": len(self.trade_history),
            "wins": wins,
            "losses": len(self.trade_history) - wins,
            "win_rate": round(wins / len(self.trade_history) * 100, 1),
            "total_pnl": round(total_pnl, 2),
            "avg_pnl": round(total_pnl / len(self.trade_history), 2),
            "balance": round(self.balance, 2),
        }


class ExchangeHandler:
    """Handles exchange data fetching and order execution."""

    def __init__(self):
        exchange_class = getattr(ccxt, Config.EXCHANGE)
        self.exchange = exchange_class({
            "apiKey": Config.EXCHANGE_API_KEY,
            "secret": Config.EXCHANGE_SECRET,
            "options": {"defaultType": "swap"},
        })
        if Config.EXCHANGE_TESTNET:
            self.exchange.set_sandbox_mode(True)

        self.paper = PaperTrader(Config.STARTING_BALANCE) if Config.PAPER_TRADING else None

    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 500):
        """Fetch candle data."""
        return self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)

    def fetch_ticker(self, symbol: str):
        """Get current price."""
        return self.exchange.fetch_ticker(symbol)

    def fetch_order_book(self, symbol: str, limit: int = 50):
        """Get order book for volume analysis."""
        return self.exchange.fetch_order_book(symbol, limit=limit)

    def fetch_funding_rate(self, symbol: str):
        """Get funding rate (crypto-specific)."""
        try:
            return self.exchange.fetch_funding_rate(symbol)
        except Exception as e:
            logger.warning(f"Could not fetch funding rate for {symbol}: {e}")
            return None

    def fetch_open_interest(self, symbol: str):
        """Get open interest."""
        try:
            return self.exchange.fetch_open_interest(symbol)
        except Exception as e:
            logger.warning(f"Could not fetch OI for {symbol}: {e}")
            return None

    def place_order(
        self,
        symbol,
        side,
        entry,
        sl,
        tp,
        size_usd,
        leverage: int = 1,
        margin_usd: float | None = None,
        liq_price: float = 0.0,
        tp_plan: dict | None = None,
    ):
        """Place order — paper or live."""
        if self.paper:
            return self.paper.open_trade(
                symbol, side, entry, sl, tp, size_usd,
                leverage=leverage, margin_usd=margin_usd, liq_price=liq_price,
                tp_plan=tp_plan,
            )

        # Live execution — ask exchange to set leverage on the symbol first.
        if leverage > 1:
            try:
                self.exchange.set_leverage(leverage, symbol)
            except Exception as exc:
                logger.warning("set_leverage failed for %s: %s", symbol, exc)

        qty = size_usd / entry
        order = self.exchange.create_order(
            symbol=symbol,
            type="limit",
            side="buy" if side == "long" else "sell",
            amount=qty,
            price=entry,
            params={
                "stopLoss": {"triggerPrice": sl},
                "takeProfit": {"triggerPrice": tp},
            },
        )
        logger.info(
            f"LIVE ORDER: {side} {symbol} @ {entry} | {leverage}× | "
            f"SL={sl} TP={tp}"
        )
        return order
