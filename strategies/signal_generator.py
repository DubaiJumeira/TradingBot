"""
Signal Generator
Combines ICT, Wyckoff, and Market Data into actionable trade signals.
Enforces minimum Risk:Reward ratio.

Phase 2: accepts optional instrument config for per-instrument min_rr and risk_pct.
Phase 9: regime-aware scoring — regime adjustments modify TP/SL/size and min score.
"""

from __future__ import annotations

import logging
from typing import Any

from config import Config

logger = logging.getLogger(__name__)


def calculate_position_size(balance: float, risk_pct: float, entry: float, sl: float):
    """Calculate position size based on risk percentage."""
    risk_amount = balance * (risk_pct / 100)
    sl_distance = abs(entry - sl)
    if sl_distance == 0:
        return 0
    qty = risk_amount / sl_distance
    size_usd = qty * entry
    return round(size_usd, 2)


def find_entry_sl_tp(ict: dict, wyckoff: dict, market: dict, current_price: float, side: str):
    """
    Determine entry, stop-loss, and take-profit levels from analysis.
    """
    entry = current_price

    if side == "long":
        # SL below nearest order block or swing low
        sl_candidates = []
        for ob in ict.get("order_blocks", []):
            if ob["type"] == "bullish" and ob["bottom"] < current_price:
                sl_candidates.append(ob["bottom"])
        for sl_pt in ict.get("swing_lows", []):
            if sl_pt["price"] < current_price:
                sl_candidates.append(sl_pt["price"])

        if not sl_candidates:
            sl = current_price * 0.985  # fallback 1.5%
        else:
            sl = max(sl_candidates)  # nearest below
            # Add a small buffer
            sl = sl * 0.998

        # TP at nearest bearish OB, swing high, or unfilled FVG above
        tp_candidates = []
        for ob in ict.get("order_blocks", []):
            if ob["type"] == "bearish" and ob["bottom"] > current_price:
                tp_candidates.append(ob["bottom"])
        for sh in ict.get("swing_highs", []):
            if sh["price"] > current_price:
                tp_candidates.append(sh["price"])

        if not tp_candidates:
            tp = current_price * 1.03  # fallback 3%
        else:
            tp = min(tp_candidates)  # nearest above

    else:  # short
        sl_candidates = []
        for ob in ict.get("order_blocks", []):
            if ob["type"] == "bearish" and ob["top"] > current_price:
                sl_candidates.append(ob["top"])
        for sh in ict.get("swing_highs", []):
            if sh["price"] > current_price:
                sl_candidates.append(sh["price"])

        if not sl_candidates:
            sl = current_price * 1.015
        else:
            sl = min(sl_candidates)
            sl = sl * 1.002

        tp_candidates = []
        for ob in ict.get("order_blocks", []):
            if ob["type"] == "bullish" and ob["top"] < current_price:
                tp_candidates.append(ob["top"])
        for sl_pt in ict.get("swing_lows", []):
            if sl_pt["price"] < current_price:
                tp_candidates.append(sl_pt["price"])

        if not tp_candidates:
            tp = current_price * 0.97
        else:
            tp = max(tp_candidates)

    return entry, sl, tp


def calculate_rr(entry, sl, tp):
    """Calculate risk:reward ratio."""
    risk = abs(entry - sl)
    reward = abs(tp - entry)
    if risk == 0:
        return 0
    return round(reward / risk, 2)


def generate_signal(symbol: str, current_price: float, ict: dict, wyckoff: dict,
                    market: dict, balance: float, news_signal: dict | None = None,
                    instrument: dict[str, Any] | None = None,
                    regime: dict[str, Any] | None = None,
                    order_flow: dict[str, Any] | None = None,
                    ltf_df: Any = None,
                    mtf_confluence: dict[str, Any] | None = None):
    """
    Generate trade signal by combining all strategy modules.

    Scoring system (0-100+):
    - ICT alignment: 0-40 points
    - Wyckoff phase: 0-25 points
    - Market data: 0-20 points
    - Kill zone: 0-15 points
    - News confluence (Phase 1D): -30 to +20 points

    Minimum score to trade: 55

    Parameters
    ----------
    news_signal : dict, optional
        From ReactiveAction.to_news_signal(). Keys:
            impact:     "low" | "medium" | "high" | "critical"
            direction:  "positive" | "negative" | "variable"
            event_title: headline text (for logging)
            pattern:    correlation map pattern name
            delay_seconds: how long the market typically takes to react
        If None, news scoring is skipped (regular 5-minute cycle with no
        active news event).
    """
    score = 0
    side = None
    reasons = []

    # =========================================
    # 1. ICT ANALYSIS (max 40 pts)
    # =========================================
    structure = ict.get("structure", "ranging")

    # Market structure direction
    if structure == "bullish":
        score += 10
        side = "long"
        reasons.append(f"Market structure: bullish (HH/HL)")
    elif structure == "bearish":
        score += 10
        side = "short"
        reasons.append(f"Market structure: bearish (LH/LL)")
    else:
        reasons.append("Market structure: ranging — reduced confidence")

    # BOS / ChoCH
    for signal in ict.get("bos_choch", []):
        if signal["type"] == "BOS":
            score += 8
            reasons.append(f"BOS confirmed {signal['direction']} @ {signal['level']:.2f}")
        elif signal["type"] == "ChoCH":
            score += 10
            side = "long" if signal["direction"] == "bullish" else "short"
            reasons.append(f"ChoCH detected → {signal['direction']} @ {signal['level']:.2f}")

    # Price near FVG (potential entry zone)
    for fvg in ict.get("fvgs", []):
        if fvg["type"] == "bullish" and fvg["bottom"] <= current_price <= fvg["top"]:
            score += 10
            side = side or "long"
            reasons.append(f"Price inside bullish FVG ({fvg['bottom']:.2f} - {fvg['top']:.2f})")
            break
        elif fvg["type"] == "bearish" and fvg["bottom"] <= current_price <= fvg["top"]:
            score += 10
            side = side or "short"
            reasons.append(f"Price inside bearish FVG ({fvg['bottom']:.2f} - {fvg['top']:.2f})")
            break

    # Price near order block
    for ob in ict.get("order_blocks", []):
        if ob["type"] == "bullish" and ob["bottom"] <= current_price <= ob["top"]:
            score += 10
            side = side or "long"
            reasons.append(f"Price at bullish OB ({ob['bottom']:.2f} - {ob['top']:.2f}), strength={ob['strength']}")
            break
        elif ob["type"] == "bearish" and ob["bottom"] <= current_price <= ob["top"]:
            score += 10
            side = side or "short"
            reasons.append(f"Price at bearish OB ({ob['bottom']:.2f} - {ob['top']:.2f}), strength={ob['strength']}")
            break

    # Recent liquidity sweep
    for sweep in ict.get("liquidity_sweeps", []):
        if sweep["type"] == "bullish_sweep":
            score += 7
            side = side or "long"
            reasons.append(f"Bullish liquidity sweep @ {sweep['level']:.2f}")
        elif sweep["type"] == "bearish_sweep":
            score += 7
            side = side or "short"
            reasons.append(f"Bearish liquidity sweep @ {sweep['level']:.2f}")

    # Phase 3: OTE (Optimal Trade Entry)
    for ote in ict.get("ote", []):
        if ote["type"] == "bullish" and side == "long":
            score += 10
            reasons.append(f"OTE bullish: price in 62-79% Fib zone ({ote['fib_79']:.2f}-{ote['fib_62']:.2f})")
        elif ote["type"] == "bearish" and side == "short":
            score += 10
            reasons.append(f"OTE bearish: price in 62-79% Fib zone ({ote['fib_62']:.2f}-{ote['fib_79']:.2f})")

    # Phase 3: Breaker blocks
    for bb in ict.get("breaker_blocks", []):
        if bb["type"] == "bullish_breaker" and side == "long":
            score += 5
            reasons.append(f"Bullish breaker block ({bb['bottom']:.2f}-{bb['top']:.2f})")
        elif bb["type"] == "bearish_breaker" and side == "short":
            score += 5
            reasons.append(f"Bearish breaker block ({bb['bottom']:.2f}-{bb['top']:.2f})")

    # Phase 3: Inducement
    for ind in ict.get("inducements", []):
        if ind["type"] == "bullish_inducement" and side == "long":
            score += 7
            reasons.append(f"Bullish inducement: minor low swept @ {ind['minor_level']:.2f}")
        elif ind["type"] == "bearish_inducement" and side == "short":
            score += 7
            reasons.append(f"Bearish inducement: minor high swept @ {ind['minor_level']:.2f}")

    # Phase 3: Premium / Discount zone filter
    # Counter-zone entries (long-in-premium / short-in-discount) were a
    # majority of the 2026-04-19/20 losing trades — marking them with a
    # hard-veto flag so the signal is rejected before emission.
    price_zone = ict.get("price_zone", "equilibrium")
    counter_zone_veto = False
    if side == "long" and price_zone == "discount":
        score += 5
        reasons.append("Price in discount zone (below equilibrium)")
    elif side == "short" and price_zone == "premium":
        score += 5
        reasons.append("Price in premium zone (above equilibrium)")
    elif side == "long" and price_zone == "premium":
        counter_zone_veto = True
        reasons.append("VETO: long entry in premium zone (counter-trend setup)")
    elif side == "short" and price_zone == "discount":
        counter_zone_veto = True
        reasons.append("VETO: short entry in discount zone (counter-trend setup)")

    # =========================================
    # 2. WYCKOFF PHASE (max 25 pts)
    # =========================================
    phase = wyckoff.get("phase", "unknown")

    if phase == "accumulation" and side in ("long", None):
        score += 15
        side = side or "long"
        reasons.append("Wyckoff: Accumulation phase (bullish)")
    elif phase == "distribution" and side in ("short", None):
        score += 15
        side = side or "short"
        reasons.append("Wyckoff: Distribution phase (bearish)")
    elif phase == "markup" and side == "long":
        score += 10
        reasons.append("Wyckoff: Markup phase — trend continuation")
    elif phase == "markdown" and side == "short":
        score += 10
        reasons.append("Wyckoff: Markdown phase — trend continuation")
    elif phase in ("accumulation", "distribution"):
        score -= 5
        reasons.append(f"Wyckoff: {phase} conflicts with ICT direction")

    # Springs / UTADs
    for spring in wyckoff.get("springs", []):
        score += 10
        side = "long"
        reasons.append(f"Wyckoff Spring detected @ {spring['low']:.2f}")

    for utad in wyckoff.get("utads", []):
        score += 10
        side = "short"
        reasons.append(f"Wyckoff UTAD detected @ {utad['high']:.2f}")

    # Phase 4: Phase transition detection (highest-probability moments).
    transition = wyckoff.get("phase_transition")
    if transition:
        if transition.get("transition") == "accumulation_to_markup" and side == "long":
            score += 15
            reasons.append(f"Wyckoff: accumulation→markup transition (vol={transition['volume_confirmation']})")
        elif transition.get("transition") == "distribution_to_markdown" and side == "short":
            score += 15
            reasons.append(f"Wyckoff: distribution→markdown transition (vol={transition['volume_confirmation']})")

    # Phase 4: VSA absorption signals.
    for vsa in wyckoff.get("vsa_signals", [])[-2:]:
        if vsa["type"] == "selling_climax" and side == "long":
            score += 5
            reasons.append("VSA: selling climax detected")
        elif vsa["type"] == "buying_climax" and side == "short":
            score += 5
            reasons.append("VSA: buying climax detected")
        elif vsa["type"] == "absorption":
            score += 3
            reasons.append(f"VSA: absorption (vol_ratio={vsa['vol_ratio']})")

    # Phase 8: effort vs result — absorption on recent bars in the
    # chosen direction = smart money filling orders. Vulnerable moves
    # in the opposite direction = a retracement likely to unwind.
    for ev in wyckoff.get("effort_vs_result", [])[-2:]:
        if ev["type"] == "absorption":
            score += 4
            reasons.append(
                f"Effort/Result: absorption (E={ev['effort_ratio']}, R={ev['result_ratio']})"
            )
        elif ev["type"] == "vulnerable_move":
            score -= 3
            reasons.append("Effort/Result: move on thin volume — vulnerable")

    # Phase 8: Wyckoff event sequence — SOS/LPS confirm markup,
    # SOW/LPSY confirm markdown. These are the schematic's "go" signals.
    for ev in wyckoff.get("wyckoff_events", [])[-3:]:
        label = ev.get("event")
        if label in ("SOS", "LPS", "Spring") and side == "long":
            score += 6
            reasons.append(f"Wyckoff event: {label} confirms long bias")
        elif label in ("SOW", "LPSY", "UTAD") and side == "short":
            score += 6
            reasons.append(f"Wyckoff event: {label} confirms short bias")

    # Phase 8: Composite-man intent (stealth accumulation / distribution).
    try:
        from strategies.composite_man import detect_composite_man
        cm = detect_composite_man(wyckoff, ict)
        if cm and side:
            if cm["intent"] == "accumulating" and side == "long":
                score += 8
                reasons.append(f"🐋 Composite Man accumulating: {cm['reason']}")
            elif cm["intent"] == "distributing" and side == "short":
                score += 8
                reasons.append(f"🐋 Composite Man distributing: {cm['reason']}")
            elif cm["intent"] == "shakeout":
                # Shakeout is a long-side signal (spring-like).
                if side == "long":
                    score += 10
                    reasons.append(f"🐋 Composite Man shakeout: {cm['reason']}")
    except Exception as exc:
        logger.debug("Composite man detection failed: %s", exc)

    # =========================================
    # 3. MARKET DATA (max 20 pts)
    # =========================================
    funding = market.get("funding", {})
    if funding.get("signal") == "extreme_long" and side == "short":
        score += 12
        reasons.append(f"Funding extreme positive ({funding['rate']}%) — fade longs")
    elif funding.get("signal") == "extreme_short" and side == "long":
        score += 12
        reasons.append(f"Funding extreme negative ({funding['rate']}%) — fade shorts")
    elif funding.get("signal") in ("elevated_long", "elevated_short"):
        score += 5
        reasons.append(f"Funding elevated ({funding['rate']}%)")

    # Volume Profile — price near POC = strong S/R
    vp = market.get("volume_profile", {})
    poc = vp.get("poc", 0)
    if poc and abs(current_price - poc) / current_price < 0.005:
        score += 5
        reasons.append(f"Price near Volume POC ({poc:.2f})")

    # Deep Volume Profile (20d real traded volume) — HVN targeting.
    # Trading toward a nearby high-volume node is high probability: HVNs
    # are where markets have historically accepted size, so price
    # tends to mean-revert / decelerate into them. Within 5% counts as
    # "reachable" on intraday timeframes.
    hvn_tp_candidate: float | None = None
    vp_deep = market.get("volume_profile_deep", {})
    hvns = vp_deep.get("hvn", []) or []
    if side and hvns:
        if side == "long":
            aligned_hvns = [h for h in hvns if h["price"] > current_price]
        else:
            aligned_hvns = [h for h in hvns if h["price"] < current_price]
        if aligned_hvns:
            nearest = min(aligned_hvns, key=lambda h: abs(h["distance_pct"]))
            dist_pct = abs(nearest["distance_pct"])
            if dist_pct <= 5.0:
                bonus = 10 if dist_pct <= 2.0 else 6
                score += bonus
                reasons.append(
                    f"📊 Trading toward HVN: ${nearest['volume_usd']/1e9:.1f}B "
                    f"@ {nearest['price']:.2f} ({dist_pct:.2f}% away)"
                )
                hvn_tp_candidate = nearest["price"]

    # =========================================
    # 4. KILL ZONE (max 15 pts)
    # =========================================
    kz = market.get("kill_zone", {})
    if kz.get("active"):
        kz_score = int(15 * kz["weight"])
        score += kz_score
        reasons.append(f"Active Kill Zone: {kz['zone']} (weight={kz['weight']})")
    else:
        reasons.append("Outside kill zone — reduced confidence")

    # =========================================
    # 5. NEWS CONFLUENCE (Phase 1D, -30 to +20 pts)
    # =========================================
    # News scoring applies when the reactive monitor passes a news_signal.
    # The direction field from the correlation map is compared against the
    # ICT/Wyckoff side determined above.
    #
    # TRADING LOGIC:
    #   - Critical news ALIGNED with technical side = very strong confluence (+20)
    #   - High news aligned = good confluence (+12)
    #   - Critical news AGAINST technical side = hard veto (-30, will likely
    #     push score below threshold → skip the trade)
    #   - "variable" direction (e.g., OPEC decision) = mild boost (+5) for
    #     having a catalyst, but no directional bias
    news_triggered = False
    if news_signal is not None:
        news_impact = news_signal.get("impact", "low")
        news_dir = news_signal.get("direction", "variable")
        event_title = news_signal.get("event_title", "")

        # Map news direction to expected trade side.
        # "positive" → expect price to go up → aligns with "long"
        # "negative" → expect price to go down → aligns with "short"
        news_side = {"positive": "long", "negative": "short"}.get(news_dir)

        if news_impact == "critical" and news_side and news_side == side:
            score += 20
            news_triggered = True
            reasons.append(f"📰 CRITICAL news aligned: {event_title[:60]}")
        elif news_impact == "high" and news_side and news_side == side:
            score += 12
            news_triggered = True
            reasons.append(f"📰 High-impact news aligned: {event_title[:60]}")
        elif news_impact == "critical" and news_side and news_side != side:
            score -= 30
            reasons.append(f"⚠️ CRITICAL news AGAINST trade direction: {event_title[:60]}")
        elif news_dir == "variable":
            score += 5
            news_triggered = True
            reasons.append(f"📰 News catalyst (variable dir): {event_title[:60]}")
        elif news_impact in ("high", "critical"):
            # News present but direction is None or doesn't map cleanly.
            score += 3
            reasons.append(f"📰 News context: {event_title[:60]}")

    # =========================================
    # 6. ORDER FLOW (Phase 11, -8 to +17 pts)
    # =========================================
    if order_flow and side:
        try:
            from strategies.order_flow import score_order_flow
            of_score, of_reasons = score_order_flow(order_flow, side)
            score += of_score
            reasons.extend(of_reasons)
        except Exception as exc:
            logger.debug("Order flow scoring failed: %s", exc)

    # =========================================
    # 8. MARKET MANIPULATION (Phase 2, -5 to +30 pts)
    # =========================================
    # Stop hunts, absorption, spoofing, wash trading. A cluster of
    # events in the same direction over a 30-minute window indicates
    # a coordinated institutional move and scores heavily.
    if side:
        manip_block = market.get("manipulation") or {}
        tracker = manip_block.get("tracker")
        if tracker is not None:
            try:
                from strategies.manipulation import score_manipulation
                mp_score, mp_reasons = score_manipulation(tracker, side)
                score += mp_score
                reasons.extend(mp_reasons)
            except Exception as exc:
                logger.debug("Manipulation scoring failed: %s", exc)

    # =========================================
    # 7. LIQUIDATION MAGNETS (Phase 1, max +25 pts)
    # =========================================
    # Price is drawn to liquidity. Trading toward a dense liquidation
    # cluster is higher probability than trading away from one —
    # market makers hunt those pools to find the opposing side for
    # their large orders. The nearest aligned magnet may also be used
    # as an alternative TP if it's FURTHER out than the ICT TP.
    liq_magnet_tp: float | None = None
    if side:
        liq_block = market.get("liquidation") or {}
        magnets = liq_block.get("magnets") or []
        asymmetry = liq_block.get("asymmetry") or {}
        if magnets:
            try:
                from strategies.liquidity_magnets import score_liquidation
                liq_score, liq_reasons, liq_magnet_tp = score_liquidation(
                    magnets, asymmetry, side
                )
                score += liq_score
                reasons.extend(liq_reasons)
            except Exception as exc:
                logger.debug("Liquidation scoring failed: %s", exc)

    # =========================================
    # 9. SELF-OPTIMIZER (Phase 5)
    # =========================================
    # Apply learned weights from historical trade outcomes. This is a
    # no-op until enough closed trades exist — below _MIN_SAMPLES the
    # optimizer returns a neutral 1.0× multiplier.
    if side:
        try:
            from strategies.self_optimizer import apply_weights_to_score
            regime_name_for_opt = (regime or {}).get("regime")
            adjusted, opt_reasons = apply_weights_to_score(
                score, reasons, regime_name_for_opt, symbol,
            )
            if adjusted != score:
                reasons.append(f"🧠 Self-optimizer: {score} → {adjusted}")
                reasons.extend(opt_reasons)
                score = adjusted
        except Exception as exc:
            logger.debug("Self-optimizer failed: %s", exc)

    # =========================================
    # 11. MTF CONFLUENCE (Phase 12, -25..+20 pts)
    # =========================================
    # Four-timeframe bias stack from MTFState (1D/4h/1h/15m). Trading
    # *with* a weighted stack is the whole point of MTF; trading
    # *against* a fully-aligned stack is a near-guaranteed lose, so
    # counter-trend setups get heavily punished instead of vetoed
    # outright (a ChoCH/BOS genuinely reversing HTF bias should still
    # be able to score through).
    if side and mtf_confluence is not None:
        mtf_dir = mtf_confluence.get("direction", "neutral")
        mtf_score_raw = abs(mtf_confluence.get("score", 0.0))  # 0..1
        aligned = mtf_confluence.get("aligned_count", 0)
        trade_side_bias = "bullish" if side == "long" else "bearish"

        # Hard veto: when MTF is strongly aligned against our side
        # (≥3/4 TFs + score ≥0.75), block entirely. A real ChoCH reversal
        # still scores on structure; everything else is just fading the
        # dominant trend, which burned us all day 2026-04-17.
        if (mtf_dir != "neutral"
                and mtf_dir != trade_side_bias
                and aligned >= 3
                and mtf_score_raw >= 0.75):
            logger.info(
                "%s: MTF veto — %s trade against %s (%d/4 TFs, score=%.2f)",
                symbol, side, mtf_dir, aligned, mtf_confluence["score"],
            )
            return None

        if mtf_dir == trade_side_bias:
            bonus = int(round(20 * mtf_score_raw))
            score += bonus
            reasons.append(
                f"📊 MTF aligned {mtf_dir} ({aligned}/4 TFs, score={mtf_confluence['score']:+.2f}) → +{bonus}"
            )
        elif mtf_dir != "neutral":
            penalty = int(round(25 * mtf_score_raw))
            score -= penalty
            reasons.append(
                f"⚠️ MTF opposes {side} ({mtf_dir}, {aligned}/4 TFs, score={mtf_confluence['score']:+.2f}) → -{penalty}"
            )
        else:
            reasons.append(f"MTF neutral ({aligned}/4 TFs aligned) — no adjustment")

    # =========================================
    # 10. ADVANCED ICT (Phase 7, +0..+22 pts)
    # =========================================
    # Silver Bullet windows, Judas swing reversal, Turtle Soup failed
    # breakout. Requires the 15m dataframe — skipped silently if caller
    # didn't pass one.
    if side and ltf_df is not None:
        try:
            from strategies.ict_advanced import score_advanced_ict
            adv_score, adv_reasons = score_advanced_ict(ltf_df, side)
            if adv_score:
                score += adv_score
                reasons.extend(adv_reasons)
        except Exception as exc:
            logger.debug("Advanced ICT scoring failed: %s", exc)

    # =========================================
    # CHASE / OVEREXTENSION FILTER
    # =========================================
    # Penalise entries after a move has already played out.
    # Primary signal: directional move % over last 12 bars (3h on 15m).
    # Secondary: compound of range position + move — catches entries near
    # extremes only when a significant move is also present (avoids
    # flagging legitimate breakouts with small moves).
    # Amplifier: candle-body acceleration (parabolic late entries).
    if side and ltf_df is not None and len(ltf_df) >= 48:
        lookback = min(48, len(ltf_df))
        recent = ltf_df.iloc[-lookback:]
        range_high = float(recent["high"].max())
        range_low = float(recent["low"].min())
        range_size = range_high - range_low if range_high != range_low else 1e-9
        range_pct = (current_price - range_low) / range_size

        lb_move = min(12, len(ltf_df))
        move_start = float(ltf_df.iloc[-lb_move]["open"])
        move_pct = ((current_price - move_start) / move_start * 100) if move_start > 0 else 0.0

        bodies_recent = ltf_df.iloc[-3:].apply(
            lambda r: abs(r["close"] - r["open"]), axis=1
        ).mean()
        bodies_prior = ltf_df.iloc[-12:-3].apply(
            lambda r: abs(r["close"] - r["open"]), axis=1
        ).mean()
        accel_ratio = bodies_recent / bodies_prior if bodies_prior > 0 else 1.0

        chase_penalty = 0

        if side == "long":
            if move_pct > 3.0:
                chase_penalty += 20
                reasons.append(f"Chase: +{move_pct:.1f}% move in 3h (overextended)")
            elif move_pct > 2.0:
                chase_penalty += 12
                reasons.append(f"Chase: +{move_pct:.1f}% move in 3h")
            if range_pct > 0.90 and move_pct > 1.0:
                chase_penalty += 8
                reasons.append(f"Chase: near 48-bar high ({range_pct:.0%}) with +{move_pct:.1f}% move")
        else:
            if move_pct < -3.0:
                chase_penalty += 20
                reasons.append(f"Chase: {move_pct:.1f}% move in 3h (overextended)")
            elif move_pct < -2.0:
                chase_penalty += 12
                reasons.append(f"Chase: {move_pct:.1f}% move in 3h")
            if range_pct < 0.10 and move_pct < -1.0:
                chase_penalty += 8
                reasons.append(f"Chase: near 48-bar low ({range_pct:.0%}) with {move_pct:.1f}% move")

        if accel_ratio > 2.5:
            chase_penalty += 8
            reasons.append(f"Chase: parabolic candles ({accel_ratio:.1f}×)")
        elif accel_ratio > 2.0:
            chase_penalty += 5
            reasons.append(f"Chase: accelerating candles ({accel_ratio:.1f}×)")

        if chase_penalty > 0:
            score -= chase_penalty
            logger.info(
                "%s: Chase filter -%d pts (range=%.2f, move=%+.1f%%, accel=%.1fx)",
                symbol, chase_penalty, range_pct, move_pct, accel_ratio,
            )

    # =========================================
    # SESSION EXHAUSTION FILTER
    # =========================================
    # If the current kill-zone session has already moved significantly
    # in the signal direction, the entry is late — most of the session's
    # move is used up. Measure from session open candle to current price.
    if side and ltf_df is not None and kz.get("active") and len(ltf_df) >= 12:
        from datetime import datetime as _dt, timezone as _tz
        now_utc = _dt.now(_tz.utc)
        kz_name = kz.get("zone", "")
        kz_map = {
            "asian": (0, 0), "london": (7, 0), "new_york": (13, 0),
            "london_new_york_overlap": (13, 0),
        }
        if kz_name in kz_map:
            sh, sm = kz_map[kz_name]
            session_open_ts = now_utc.replace(hour=sh, minute=sm, second=0, microsecond=0)
            if session_open_ts > now_utc:
                session_open_ts = session_open_ts.replace(day=session_open_ts.day - 1)
            # Find the candle closest to session open.
            if "timestamp" in ltf_df.columns:
                ts_col = ltf_df["timestamp"]
                so_ms = int(session_open_ts.timestamp() * 1000)
                idx = (ts_col - so_ms).abs().idxmin()
                session_open_price = float(ltf_df.loc[idx, "open"])
            else:
                session_open_price = float(ltf_df.iloc[-12]["open"])

            if session_open_price > 0:
                session_move = (current_price - session_open_price) / session_open_price * 100
                exhaustion_penalty = 0
                if side == "long" and session_move > 1.5:
                    exhaustion_penalty = 10
                    reasons.append(f"Session exhaustion: {kz_name} already +{session_move:.1f}%")
                elif side == "short" and session_move < -1.5:
                    exhaustion_penalty = 10
                    reasons.append(f"Session exhaustion: {kz_name} already {session_move:.1f}%")
                if exhaustion_penalty > 0:
                    score -= exhaustion_penalty
                    logger.info(
                        "%s: Session exhaustion -%d pts (%s move=%+.1f%%)",
                        symbol, exhaustion_penalty, kz_name, session_move,
                    )

    # =========================================
    # FINAL DECISION
    # =========================================
    if not side:
        return None

    # Calculate entry, SL, TP
    entry, sl, tp = find_entry_sl_tp(ict, wyckoff, market, current_price, side)

    # Enforce minimum SL distance to avoid noise-level stops.
    # Observed losers (2026-04-19 → 2026-04-20 paper run): every single
    # SL-hit loss closed with 0/3 TPs filled, meaning price moved against
    # the position and tagged SL before reaching even TP1. Raising the
    # floor from 0.5×ATR/0.5% → 1.0×ATR/0.8% gives normal chop more room.
    # This reduces RR so some marginal signals will fail the RR gate.
    if ltf_df is not None and len(ltf_df) >= 15:
        try:
            from strategies.risk_manager import calculate_atr
            atr = calculate_atr(ltf_df, period=14)
        except Exception:
            atr = 0.0
        atr_floor = atr * 1.0
        pct_floor = entry * 0.008
        min_sl_dist = max(atr_floor, pct_floor)
        current_sl_dist = abs(entry - sl)
        if current_sl_dist < min_sl_dist and min_sl_dist > 0:
            if side == "long":
                sl = entry - min_sl_dist
            else:
                sl = entry + min_sl_dist
            reasons.append(
                f"SL widened to {min_sl_dist:.2f} floor "
                f"(was {current_sl_dist:.2f}, ATR={atr:.2f})"
            )

    # Phase 1D: pick the best aligned TP magnet. Candidates are:
    #   - liq_magnet_tp (observed/orderbook liquidation cluster)
    #   - hvn_tp_candidate (20d high-volume node — historical magnet)
    # We only EXTEND the TP (never pull it in), because tightening TP
    # below ICT structure would force early exits before the planned
    # move completes. Between the two candidates pick the FARTHER one
    # still in the trade's direction — it gives the larger reward.
    best_magnet_tp: float | None = None
    for candidate in (liq_magnet_tp, hvn_tp_candidate):
        if candidate is None:
            continue
        if side == "long" and candidate > tp:
            if best_magnet_tp is None or candidate > best_magnet_tp:
                best_magnet_tp = candidate
        elif side == "short" and candidate < tp:
            if best_magnet_tp is None or candidate < best_magnet_tp:
                best_magnet_tp = candidate

    if best_magnet_tp is not None:
        kind = "HVN" if best_magnet_tp == hvn_tp_candidate else "liquidation magnet"
        reasons.append(
            f"TP extended to {kind} @ {best_magnet_tp:.2f} "
            f"(ICT TP was {tp:.2f})"
        )
        tp = best_magnet_tp

    # Wall-aware SL tightening: if a thick orderbook wall sits between
    # entry and the current SL, move SL to just past the wall. Walls are
    # real resting liquidity — price breaking through a $5M+ wall is
    # strong evidence the thesis is wrong, so that's a better stop than
    # waiting for the full ICT swing distance. SL can only get TIGHTER
    # (closer to entry), never looser, and must still respect the ATR
    # floor computed above.
    liq_block = market.get("liquidation") or {}
    walls = liq_block.get("walls") or []
    if walls and side:
        def _wall_volume_threshold(walls: list[dict]) -> float:
            vols = sorted((w["volume_usd"] for w in walls), reverse=True)
            if not vols:
                return 0.0
            # Median of the top 10 walls — "thick enough to matter"
            top = vols[: min(10, len(vols))]
            return top[len(top) // 2]

        thresh = _wall_volume_threshold(walls)
        if side == "long":
            # Look for thick BIDS sitting strictly between SL and entry
            between = [
                w for w in walls
                if w["side"] == "bid" and sl < w["price"] < entry
                and w["volume_usd"] >= thresh
            ]
            if between:
                thickest = max(between, key=lambda w: w["volume_usd"])
                buffer = (entry - sl) * 0.08 or entry * 0.0005
                new_sl = thickest["price"] - buffer
                if new_sl > sl:
                    # Respect the ATR floor we just enforced
                    if abs(entry - new_sl) >= (min_sl_dist if 'min_sl_dist' in locals() else 0):
                        reasons.append(
                            f"SL tightened to bid wall @ {thickest['price']:.2f} "
                            f"(${thickest['volume_usd']/1e6:.1f}M, was {sl:.2f})"
                        )
                        sl = new_sl
        else:
            between = [
                w for w in walls
                if w["side"] == "ask" and entry < w["price"] < sl
                and w["volume_usd"] >= thresh
            ]
            if between:
                thickest = max(between, key=lambda w: w["volume_usd"])
                buffer = (sl - entry) * 0.08 or entry * 0.0005
                new_sl = thickest["price"] + buffer
                if new_sl < sl:
                    if abs(new_sl - entry) >= (min_sl_dist if 'min_sl_dist' in locals() else 0):
                        reasons.append(
                            f"SL tightened to ask wall @ {thickest['price']:.2f} "
                            f"(${thickest['volume_usd']/1e6:.1f}M, was {sl:.2f})"
                        )
                        sl = new_sl

    # Phase 9: apply regime adjustments to TP/SL.
    regime_adj = (regime or {}).get("adjustments", {})
    regime_name = (regime or {}).get("regime", "unknown")
    tp_mult = regime_adj.get("tp_multiplier", 1.0)
    sl_mult = regime_adj.get("sl_multiplier", 1.0)
    size_mult = regime_adj.get("size_multiplier", 1.0)
    min_score_adj = regime_adj.get("min_score_adjust", 0)

    if tp_mult != 1.0 or sl_mult != 1.0:
        tp_dist = abs(tp - entry)
        sl_dist = abs(entry - sl)
        if side == "long":
            tp = entry + tp_dist * tp_mult
            sl = entry - sl_dist * sl_mult
        else:
            tp = entry - tp_dist * tp_mult
            sl = entry + sl_dist * sl_mult
        reasons.append(f"Regime '{regime_name}': TP×{tp_mult}, SL×{sl_mult}")

    # Re-enforce the ATR floor AFTER regime mult. Previously the floor ran
    # first and a regime SL<1.0 multiplier silently shrank the stop back
    # below the floor (trade #14 2026-04-21: floor set 609.63, ranging
    # ×0.8 shrank to 487.70 → hit next cycle).
    if 'min_sl_dist' in locals() and min_sl_dist > 0:
        post_regime_dist = abs(entry - sl)
        if post_regime_dist < min_sl_dist:
            if side == "long":
                sl = entry - min_sl_dist
            else:
                sl = entry + min_sl_dist
            reasons.append(
                f"SL floor re-enforced after regime mult: "
                f"{post_regime_dist:.2f} → {min_sl_dist:.2f}"
            )

    rr = calculate_rr(entry, sl, tp)

    # Counter-zone hard veto: long-in-premium / short-in-discount. These
    # setups showed poor outcomes on 2026-04-19/20 paper trades and are
    # now rejected outright rather than merely downscored.
    if counter_zone_veto:
        logger.info(f"{symbol}: VETO counter-zone {side} entry in {price_zone} zone. Skipping.")
        return None

    # Enforce minimum RR — per-instrument override or global default.
    min_rr = (instrument or {}).get("min_rr", Config.MIN_RR_RATIO)
    if rr < min_rr:
        logger.info(f"{symbol}: Signal found (score={score}) but RR={rr} < min {min_rr}. Skipping.")
        return None

    # Minimum score threshold — raised by regime adjustment.
    min_score = 55 + min_score_adj
    if score < min_score:
        logger.info(f"{symbol}: Score {score}/100 below threshold ({min_score}, regime={regime_name}). Skipping.")
        return None

    # Position sizing — per-instrument risk_pct or global default.
    # Fixed 1% (or per-instrument) risk per trade: no regime multiplier,
    # so every trade risks exactly risk_pct of current balance.
    risk_pct = (instrument or {}).get("risk_pct", Config.DEFAULT_RISK_PERCENT)
    size_usd = calculate_position_size(balance, risk_pct, entry, sl)

    # Hard cap: position size cannot exceed MAX_POSITION_PCT of balance.
    # This prevents oversizing even if SL is very tight.
    max_size_usd = balance * Config.MAX_POSITION_PCT
    if size_usd > max_size_usd:
        logger.info(
            f"{symbol}: Position size ${size_usd:.2f} exceeds max "
            f"(${max_size_usd:.2f}), capped."
        )
        size_usd = max_size_usd

    # Phase 6: compute a partial-TP plan (TP1 50% @ 1R → BE, TP2 30% @ 2R
    # → trail, TP3 20% at the planned ICT TP). The PaperTrader steps
    # through this plan on each exit check.
    tp_plan_dict = None
    try:
        from strategies.risk_manager import PartialTPPlan
        plan = PartialTPPlan.compute_from_signal(side, entry, sl, tp)
        if plan.levels:
            tp_plan_dict = plan.to_dict()
    except Exception as exc:
        logger.debug("Partial TP plan generation failed: %s", exc)

    signal = {
        "symbol": symbol,
        "side": side,
        "entry": round(entry, 2),
        "sl": round(sl, 2),
        "tp": round(tp, 2),
        "rr": rr,
        "score": score,
        "size_usd": size_usd,
        "risk_pct": risk_pct,
        "reasons": reasons,
        "ict_structure": structure,
        "wyckoff_phase": phase,
        "kill_zone": kz.get("zone", "none"),
        "funding_rate": funding.get("rate", 0),
        "news_triggered": news_triggered,
        "news_signal": news_signal,
        "regime": regime_name,
        "tp_plan": tp_plan_dict,
        "mtf": mtf_confluence,
    }

    tag = "📰 NEWS-SIGNAL" if news_triggered else "SIGNAL"
    logger.info(f"{tag}: {side.upper()} {symbol} | Score={score} | RR={rr} | Entry={entry}")
    return signal
