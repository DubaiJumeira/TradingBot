"""
Phase 7 — Advanced ICT concepts.

- Silver Bullet windows: three fixed 1h windows per session where ICT
  entries have historically concentrated. Hitting one adds score.
- Judas Swing: the opening 1h of the NY session often prints a fake
  move that reverses — a setup that aligns with that reversal counts.
- Turtle Soup: a failed breakout of a recent range extreme that closes
  back inside. Essentially a swept-liquidity reversal at the most
  obvious stop cluster in the tape.

All detectors are pure-pandas and take the 15m/1h dataframe the rest
of the pipeline already computes, so there's no extra data fetch.
"""

from __future__ import annotations

from datetime import datetime, time, timezone
from typing import Any

import pandas as pd


# Silver Bullet windows in UTC (approximate NY-time anchors, no DST
# correction — close enough for a 1h bucket scoring boost).
SILVER_BULLET_WINDOWS: list[tuple[str, time, time]] = [
    ("london_sb", time(3, 0), time(4, 0)),
    ("ny_am_sb", time(14, 0), time(15, 0)),
    ("ny_pm_sb", time(19, 0), time(20, 0)),
]


def in_silver_bullet_window(now: datetime | None = None) -> dict[str, Any]:
    """Return {'active': bool, 'window': name|None, 'weight': float}.

    Weight is 1.0 inside a window and 0.0 otherwise. The scorer uses
    this as an independent ICT-time filter on top of the broader
    kill-zone bonus.
    """
    now = now or datetime.now(timezone.utc)
    t = now.time()
    for name, start, end in SILVER_BULLET_WINDOWS:
        if start <= t < end:
            return {"active": True, "window": name, "weight": 1.0}
    return {"active": False, "window": None, "weight": 0.0}


def detect_judas_swing(
    df: pd.DataFrame,
    session_open_utc_hour: int = 13,
    lookahead_minutes: int = 90,
) -> dict[str, Any] | None:
    """Detect a NY-session Judas swing on the most recent session.

    Definition used here: at the NY open (13:30 UTC), price makes an
    initial move in one direction within the first ~30–60min, then
    reverses through the session-open price within ``lookahead_minutes``.
    A long-side Judas sets up shorts (fake pump), a short-side Judas
    sets up longs (fake dump).

    Expects a datetime-indexed OHLC dataframe at <=15m resolution.
    Returns None if the pattern isn't present.
    """
    if not isinstance(df.index, pd.DatetimeIndex) or len(df) < 10:
        return None

    last_ts = df.index[-1]
    if last_ts.tzinfo is None:
        last_ts = last_ts.tz_localize("UTC")

    session_start = last_ts.replace(hour=session_open_utc_hour, minute=30, second=0, microsecond=0)
    if last_ts < session_start:
        session_start -= pd.Timedelta(days=1)

    window_end = session_start + pd.Timedelta(minutes=lookahead_minutes)

    try:
        session_df = df.loc[session_start:window_end]
    except KeyError:
        return None
    if len(session_df) < 3:
        return None

    open_price = float(session_df.iloc[0]["open"])
    initial_high = float(session_df.iloc[: max(2, len(session_df) // 3)]["high"].max())
    initial_low = float(session_df.iloc[: max(2, len(session_df) // 3)]["low"].min())
    last_price = float(session_df.iloc[-1]["close"])

    up_move = (initial_high - open_price) / open_price
    down_move = (open_price - initial_low) / open_price
    reversal = (last_price - open_price) / open_price

    # Fake pump → short setup: initial up ≥ 0.25%, price now back below open.
    if up_move >= 0.0025 and reversal <= -0.001:
        return {
            "type": "bearish",
            "trade_side": "short",
            "fake_high": initial_high,
            "session_open": open_price,
            "current": last_price,
        }
    # Fake dump → long setup.
    if down_move >= 0.0025 and reversal >= 0.001:
        return {
            "type": "bullish",
            "trade_side": "long",
            "fake_low": initial_low,
            "session_open": open_price,
            "current": last_price,
        }
    return None


def detect_turtle_soup(
    df: pd.DataFrame,
    lookback: int = 20,
) -> dict[str, Any] | None:
    """Detect a failed breakout of the ``lookback``-bar range that closes
    back inside. This is the classic "turtle soup" reversal setup.

    Returns a trade-side dict or None.
    """
    if len(df) < lookback + 2:
        return None

    window = df.iloc[-(lookback + 2) : -1]  # exclude current in-progress bar
    prior_high = float(window.iloc[:-1]["high"].max())
    prior_low = float(window.iloc[:-1]["low"].min())
    last = df.iloc[-1]

    # False high breakout → short.
    if float(last["high"]) > prior_high and float(last["close"]) < prior_high:
        return {
            "type": "bearish",
            "trade_side": "short",
            "swept_level": prior_high,
            "sweep_high": float(last["high"]),
        }
    # False low breakout → long.
    if float(last["low"]) < prior_low and float(last["close"]) > prior_low:
        return {
            "type": "bullish",
            "trade_side": "long",
            "swept_level": prior_low,
            "sweep_low": float(last["low"]),
        }
    return None


def score_advanced_ict(
    df: pd.DataFrame,
    side: str,
    now: datetime | None = None,
) -> tuple[int, list[str]]:
    """Combine the three advanced-ICT detectors into a (score, reasons)
    tuple the main scorer can fold in. Max +22 points when all three
    align with ``side``; -0 when nothing fires.
    """
    score = 0
    reasons: list[str] = []

    sb = in_silver_bullet_window(now)
    if sb["active"]:
        score += 8
        reasons.append(f"🥈 Silver Bullet window: {sb['window']}")

    judas = detect_judas_swing(df)
    if judas and judas["trade_side"] == side:
        score += 8
        reasons.append(
            f"Judas swing {judas['type']} — faked move then reversed through open"
        )

    soup = detect_turtle_soup(df)
    if soup and soup["trade_side"] == side:
        score += 6
        reasons.append(
            f"Turtle soup: failed breakout of {soup['swept_level']:.2f}"
        )

    return score, reasons
