"""
Dynamic Leverage — Phase 3.

Leverage is not a constant. The optimal amount of leverage for a trade
depends on confidence (score), volatility, instrument, market regime,
and current account drawdown. This module encapsulates the decision
so the rest of the pipeline can stay leverage-agnostic.

DESIGN GOALS
------------
- Higher-confidence signals earn more leverage (score 55→3×, 85+→15×).
- Volatile markets get LESS leverage regardless of score — an 80 score
  in a 4% daily-range environment is not the same risk as in a 1% one.
- Choppy / news / drawdown regimes are hard caps, not soft penalties.
  These are environments where bad things happen fast and reducing
  exposure is the only defensive play.
- The risk per trade ($) doesn't change — only margin utilization does.
  More leverage means the same SL hit costs the same $ but consumes
  less margin, freeing capital for additional concurrent trades.

RETURN SHAPE
------------
calculate_optimal_leverage returns a dict:
    {
        "leverage": int,            # the chosen multiplier
        "base": int,                # starting leverage by score
        "caps_applied": list[str],  # which caps reduced it, for logging
        "reasons": list[str],       # human-readable trail for Telegram
    }

Callers then pass leverage into compute_margin / compute_liquidation_price.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


# Maintenance margin assumed by the liquidation estimator. Bybit USDT-perp
# uses a tiered MM that starts near 0.5% for standard-size positions.
_MAINTENANCE_MARGIN_PCT = 0.005

# Hard ceilings per instrument type — never exceed these even if all
# multipliers would push higher.
_MAX_LEVERAGE_BY_TYPE = {
    "crypto": 10,
    "cfd": 10,
}

# Fallback when the instrument config doesn't specify a cap.
_DEFAULT_MAX_LEVERAGE = 10


def _base_leverage_for_score(score: int) -> int:
    """Score → base leverage before any caps.

    The curve is deliberately gentle in the 55-65 band (where most
    signals live) and only rewards the upper tail aggressively.
    """
    if score >= 85:
        return 15
    if score >= 80:
        return 12
    if score >= 75:
        return 10
    if score >= 70:
        return 8
    if score >= 65:
        return 6
    if score >= 60:
        return 5
    return 3  # 55-59


def calculate_optimal_leverage(
    *,
    signal_score: int,
    volatility_pct: float,
    instrument: dict | None = None,
    regime_name: str | None = None,
    news_active: bool = False,
    drawdown_pct: float = 0.0,
) -> dict:
    """Decide how much leverage to use for a given signal.

    Parameters
    ----------
    signal_score : int
        Final confidence score (0-100+).
    volatility_pct : float
        Recent volatility as a percentage (e.g. 2.5 = 2.5% daily range).
        Usually sourced from the regime detector's ``volatility_pct``.
    instrument : dict, optional
        Per-instrument config from ``INSTRUMENTS``. The ``type`` field
        ("crypto" / "cfd") picks the hard cap; an explicit
        ``max_leverage`` field overrides it.
    regime_name : str, optional
        From the regime detector — "choppy", "trending", "volatile",
        "squeeze", etc. Choppy markets get a hard 5× cap.
    news_active : bool
        True if a high/critical news event is in flight. News trades
        get a 10× cap regardless of score.
    drawdown_pct : float
        Current account drawdown from peak equity. Above 2% we cut
        leverage aggressively; above 5% we ramp it down to 2×.
    """
    reasons: list[str] = []
    caps: list[str] = []

    base = _base_leverage_for_score(signal_score)
    leverage = base
    reasons.append(f"Base {base}× for score {signal_score}")

    # --- Volatility dampener ------------------------------------------------
    # A 3%+ daily range means stops get run more often — halve leverage.
    # 5%+ is a full regime shift (think BTC during CPI), so drop further.
    if volatility_pct >= 5.0:
        leverage = max(2, leverage // 3)
        caps.append("vol>5%")
        reasons.append(f"High volatility {volatility_pct:.1f}% → /3")
    elif volatility_pct >= 3.0:
        leverage = max(2, leverage // 2)
        caps.append("vol>3%")
        reasons.append(f"Elevated volatility {volatility_pct:.1f}% → /2")

    # --- Regime caps --------------------------------------------------------
    # Choppy ranges are where leverage goes to die. News is a slightly
    # softer cap because aligned news is actually high-edge — we just
    # don't want to be overexposed when the next headline hits.
    if regime_name == "choppy":
        if leverage > 5:
            caps.append("choppy→5")
            leverage = 5
            reasons.append("Choppy regime → cap 5×")
    if regime_name == "volatile" and leverage > 8:
        caps.append("volatile→8")
        leverage = 8
        reasons.append("Volatile regime → cap 8×")

    if news_active and leverage > 10:
        caps.append("news→10")
        leverage = 10
        reasons.append("News event active → cap 10×")

    # --- Drawdown protection ------------------------------------------------
    # As the account bleeds, shrink exposure. This is a capital-preservation
    # reflex — losing streaks are when stops get clustered and the worst
    # thing you can do is be leveraged into them.
    if drawdown_pct >= 5.0 and leverage > 2:
        caps.append("dd>5%→2")
        leverage = 2
        reasons.append(f"Drawdown {drawdown_pct:.1f}% → cap 2×")
    elif drawdown_pct >= 2.0 and leverage > 3:
        caps.append("dd>2%→3")
        leverage = 3
        reasons.append(f"Drawdown {drawdown_pct:.1f}% → cap 3×")

    # --- Instrument ceiling -------------------------------------------------
    inst_type = (instrument or {}).get("type", "crypto")
    type_max = _MAX_LEVERAGE_BY_TYPE.get(inst_type, _DEFAULT_MAX_LEVERAGE)
    inst_max = (instrument or {}).get("max_leverage", type_max)
    if leverage > inst_max:
        caps.append(f"inst_max={inst_max}")
        leverage = inst_max
        reasons.append(f"{inst_type} cap → {inst_max}×")

    # Never below 1× (would mean no position).
    leverage = max(1, int(leverage))

    return {
        "leverage": leverage,
        "base": base,
        "caps_applied": caps,
        "reasons": reasons,
    }


def compute_margin_required(size_usd: float, leverage: int) -> float:
    """Margin consumed for a position of the given notional size."""
    if leverage <= 0:
        return size_usd
    return round(size_usd / leverage, 2)


def compute_liquidation_price(
    entry: float,
    leverage: int,
    side: str,
    maintenance_margin_pct: float = _MAINTENANCE_MARGIN_PCT,
) -> float:
    """Estimated liquidation price for an isolated-margin position.

    Formula (isolated margin, ignoring fees):
        long:  entry * (1 - 1/L + MM)
        short: entry * (1 + 1/L - MM)

    At leverage 1× the long liquidation is below zero (position can't be
    liquidated), so we clamp to 0 for longs.
    """
    if leverage <= 1:
        return 0.0 if side == "long" else entry * 2
    inv_l = 1.0 / leverage
    if side == "long":
        return round(entry * (1 - inv_l + maintenance_margin_pct), 2)
    return round(entry * (1 + inv_l - maintenance_margin_pct), 2)


def apply_leverage_to_signal(
    signal: dict,
    *,
    volatility_pct: float,
    instrument: dict | None,
    regime_name: str | None,
    news_active: bool,
    drawdown_pct: float,
    balance: float | None = None,
    max_margin_pct: float = 0.10,
) -> dict:
    """Attach leverage / margin / liquidation-price fields to a signal dict.

    Mutates and returns the signal for convenience. The stop-loss is
    also sanity-checked against the liquidation price: if SL sits beyond
    liq the leverage is reduced until it's safely inside (the SL is
    always the intended exit, never the liquidation).
    """
    decision = calculate_optimal_leverage(
        signal_score=signal.get("score", 0),
        volatility_pct=volatility_pct,
        instrument=instrument,
        regime_name=regime_name,
        news_active=news_active,
        drawdown_pct=drawdown_pct,
    )
    leverage = decision["leverage"]
    entry = float(signal["entry"])
    sl = float(signal["sl"])
    side = signal["side"]

    # Sanity check: the SL must be closer to entry than the liquidation
    # price, otherwise we'd get liquidated before the stop triggers.
    # Reduce leverage until SL distance < liq distance * 0.8 (20% buffer).
    while leverage > 1:
        liq = compute_liquidation_price(entry, leverage, side)
        liq_distance = abs(entry - liq)
        sl_distance = abs(entry - sl)
        if sl_distance < liq_distance * 0.8:
            break
        leverage -= 1
        decision["caps_applied"].append("sl_vs_liq")

    liq_price = compute_liquidation_price(entry, leverage, side)
    margin = compute_margin_required(signal["size_usd"], leverage)

    # Hard cap: margin must not exceed ``max_margin_pct`` of current
    # balance. When the cap fires, bump leverage instead of shrinking
    # size — that keeps risk at the intended 1% of balance. Only fall
    # back to shrinking size if even max leverage (constrained by the
    # SL-vs-liq safety check) cannot fit the notional within the cap.
    if balance is not None and balance > 0:
        max_margin = balance * max_margin_pct
        if margin > max_margin:
            inst_type = (instrument or {}).get("type", "crypto")
            max_lev = _MAX_LEVERAGE_BY_TYPE.get(inst_type, 20)
            import math
            needed_lev = min(max_lev, math.ceil(signal["size_usd"] / max_margin))
            # Re-enforce SL-vs-liq safety at the bumped leverage.
            while needed_lev > leverage:
                liq = compute_liquidation_price(entry, needed_lev, side)
                if abs(entry - sl) < abs(entry - liq) * 0.8:
                    break
                needed_lev -= 1
            if needed_lev > leverage:
                leverage = needed_lev
                liq_price = compute_liquidation_price(entry, leverage, side)
                margin = compute_margin_required(signal["size_usd"], leverage)
                decision["caps_applied"].append("margin_cap_leverage_bumped")
                decision["reasons"].append(
                    f"Leverage bumped to {leverage}× to keep margin "
                    f"under {max_margin_pct*100:.0f}% of balance (risk preserved)"
                )
            if margin > max_margin:
                shrink = max_margin / margin
                signal["size_usd"] = round(signal["size_usd"] * shrink, 2)
                margin = round(margin * shrink, 2)
                decision["caps_applied"].append("margin_vs_balance")
                decision["reasons"].append(
                    f"Margin still over cap at max leverage — shrunk size "
                    f"by {(1-shrink)*100:.0f}% (risk reduced)"
                )

    signal["leverage"] = leverage
    signal["margin_usd"] = margin
    signal["liq_price"] = liq_price
    signal["leverage_reasons"] = decision["reasons"]

    return signal
