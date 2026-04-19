"""
Self-Optimization — Phase 5.

The bot learns from its own track record. Every closed trade in the
database carries the reasons that produced the signal, the regime, the
symbol, and the PnL. By aggregating across those dimensions we can
discover which signal components actually make money and which ones
are noise, then feed that back into the scoring pipeline as weight
multipliers.

APPROACH
--------
The analyzer groups trades by:
    - reason_tag    (e.g. "OB", "FVG", "ChoCH", "Wyckoff", "Liquidation",
                     "Stop hunt", "Absorption", "OTE", "News")
    - regime        (choppy / trending / ranging / volatile / squeeze)
    - symbol        (BTCUSDT, ETHUSDT, ...)
    - score_bucket  (55-64, 65-74, 75-84, 85+)

For each bucket we compute win_rate and expectancy (avg_pnl). When a
bucket has enough trades to be statistically meaningful (N ≥ 10) we
derive a weight multiplier that the signal generator can apply:

    weight = clip(expectancy / baseline_expectancy, 0.5, 1.5)

Buckets with too few samples get a neutral weight of 1.0 (no effect).
This guarantees the optimizer does nothing dangerous on day one — it
gradually influences scoring as real evidence accumulates.

OUTPUT
------
Recommended weights are persisted to data/optimizer_weights.json so
the signal generator can load them on each call without touching the
database. The weights file carries a ``updated_at`` timestamp and a
``sample_size`` per bucket so stale recommendations can be ignored.
"""

from __future__ import annotations

import json
import logging
import os
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


WEIGHTS_PATH = Path("data/optimizer_weights.json")

# Minimum sample count before a bucket gets a non-neutral weight. Below
# this, we refuse to let randomness drive the scoring.
_MIN_SAMPLES = 10

# Weight multipliers are clipped into this range so the optimizer can
# nudge scoring but never nuke or double it based on a streak.
_WEIGHT_MIN = 0.5
_WEIGHT_MAX = 1.5

# Tag extractors — maps a regex pattern → canonical tag. The first
# match wins so order matters when patterns overlap (e.g. put
# "Liquidation" before generic "order block").
_TAG_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"ChoCH", re.I), "choch"),
    (re.compile(r"\bBOS\b", re.I), "bos"),
    (re.compile(r"order block|\bOB\b", re.I), "order_block"),
    (re.compile(r"FVG|fair value gap", re.I), "fvg"),
    (re.compile(r"liquidity sweep", re.I), "sweep"),
    (re.compile(r"\bOTE\b|fib zone", re.I), "ote"),
    (re.compile(r"breaker block", re.I), "breaker"),
    (re.compile(r"inducement", re.I), "inducement"),
    (re.compile(r"Wyckoff|accumulation|distribution|markup|markdown|spring|utad", re.I), "wyckoff"),
    (re.compile(r"VSA|climax|absorption.*range", re.I), "vsa"),
    (re.compile(r"stop hunt", re.I), "stop_hunt"),
    (re.compile(r"spoofing|fake support|fake resistance", re.I), "spoofing"),
    (re.compile(r"wash trading", re.I), "wash_trading"),
    (re.compile(r"COORDINATED MOVE", re.I), "manipulation_cluster"),
    (re.compile(r"liquidity|liquidation magnet|Trading toward", re.I), "liquidation"),
    (re.compile(r"News|📰", re.I), "news"),
    (re.compile(r"Kill Zone", re.I), "kill_zone"),
    (re.compile(r"order flow|CVD|imbalance|aggressor", re.I), "order_flow"),
    (re.compile(r"funding", re.I), "funding"),
]


def extract_tags(reasons: list[str] | str) -> set[str]:
    """Parse a trade's reasons list into a set of canonical tags.

    Reasons is normally a list[str]; when loaded from the DB it may
    still be a JSON-encoded string, so we handle both.
    """
    if isinstance(reasons, str):
        try:
            reasons = json.loads(reasons)
        except Exception:
            reasons = [reasons]
    tags: set[str] = set()
    for r in reasons or []:
        for pattern, tag in _TAG_PATTERNS:
            if pattern.search(r):
                tags.add(tag)
    return tags


def _score_bucket(score: int | None) -> str:
    if score is None:
        return "unknown"
    if score >= 85:
        return "85+"
    if score >= 75:
        return "75-84"
    if score >= 65:
        return "65-74"
    return "55-64"


class PerformanceAnalyzer:
    """Computes per-bucket win rate and expectancy from closed trades."""

    def __init__(self, db: Any) -> None:
        self.db = db

    # ---------- data loading ----------

    def _closed_trades(self, lookback_days: int | None = 90) -> list[dict]:
        """All trades with a pnl and closed_at, newest first.

        lookback_days=None → no cutoff.
        Reads from paper_trades.json (primary source for paper trading mode).
        """
        # Try paper_trades.json first (paper trading mode)
        try:
            paper_file = Path("data/paper_trades.json")
            if paper_file.exists():
                with open(paper_file) as f:
                    state = json.load(f)
                    trades = state.get("trade_history", [])
                    closed = [t for t in trades if t.get("closed_at") and t.get("pnl") is not None]
                    if lookback_days is None:
                        return closed
                    cutoff = datetime.now(tz=timezone.utc).timestamp() - lookback_days * 86_400
                    kept = []
                    for t in closed:
                        try:
                            ts = datetime.fromisoformat(t["closed_at"].replace("+00:00", "")).timestamp()
                        except Exception:
                            continue
                        if ts >= cutoff:
                            kept.append(t)
                    return kept
        except Exception as e:
            logger.debug(f"Could not load paper_trades.json: {e}")

        # Fallback to database
        trades = self.db.get_trades(limit=10_000)
        closed = [t for t in trades if t.get("closed_at") and t.get("pnl") is not None]
        if lookback_days is None:
            return closed
        cutoff = datetime.now(tz=timezone.utc).timestamp() - lookback_days * 86_400
        kept = []
        for t in closed:
            try:
                ts = datetime.fromisoformat(t["closed_at"]).timestamp()
            except Exception:
                continue
            if ts >= cutoff:
                kept.append(t)
        return kept

    # ---------- aggregations ----------

    @staticmethod
    def _stats(trades: list[dict]) -> dict:
        n = len(trades)
        if n == 0:
            return {"n": 0, "win_rate": 0.0, "expectancy": 0.0, "total_pnl": 0.0}
        wins = sum(1 for t in trades if (t.get("pnl") or 0) > 0)
        total = sum((t.get("pnl") or 0) for t in trades)
        return {
            "n": n,
            "win_rate": round(wins / n * 100, 1),
            "expectancy": round(total / n, 2),
            "total_pnl": round(total, 2),
        }

    def by_tag(self, lookback_days: int | None = 90) -> dict[str, dict]:
        """Group trades by reason tag. A trade appears in every tag it
        carries — a single FVG+OTE trade contributes to both buckets.
        """
        trades = self._closed_trades(lookback_days)
        buckets: dict[str, list[dict]] = defaultdict(list)
        for t in trades:
            for tag in extract_tags(t.get("reasons", "[]")):
                buckets[tag].append(t)
        return {tag: self._stats(lst) for tag, lst in buckets.items()}

    def by_regime(self, lookback_days: int | None = 90) -> dict[str, dict]:
        trades = self._closed_trades(lookback_days)
        buckets: dict[str, list[dict]] = defaultdict(list)
        for t in trades:
            buckets[t.get("regime") or "unknown"].append(t)
        return {k: self._stats(v) for k, v in buckets.items()}

    def by_symbol(self, lookback_days: int | None = 90) -> dict[str, dict]:
        trades = self._closed_trades(lookback_days)
        buckets: dict[str, list[dict]] = defaultdict(list)
        for t in trades:
            buckets[t["symbol"]].append(t)
        return {k: self._stats(v) for k, v in buckets.items()}

    def by_score_bucket(self, lookback_days: int | None = 90) -> dict[str, dict]:
        trades = self._closed_trades(lookback_days)
        buckets: dict[str, list[dict]] = defaultdict(list)
        for t in trades:
            buckets[_score_bucket(t.get("score"))].append(t)
        return {k: self._stats(v) for k, v in buckets.items()}

    def full_report(self, lookback_days: int | None = 90) -> dict[str, Any]:
        """Everything the Telegram /performance command needs."""
        trades = self._closed_trades(lookback_days)
        return {
            "overall": self._stats(trades),
            "lookback_days": lookback_days,
            "by_tag": self.by_tag(lookback_days),
            "by_regime": self.by_regime(lookback_days),
            "by_symbol": self.by_symbol(lookback_days),
            "by_score_bucket": self.by_score_bucket(lookback_days),
            "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        }

    # ---------- weight derivation ----------

    def recommend_weights(self, lookback_days: int | None = 90) -> dict[str, Any]:
        """Derive score multipliers from historical performance.

        Baseline expectancy is the average expectancy across all closed
        trades. Each bucket's weight = clip(bucket_exp / baseline, 0.5, 1.5),
        but only if the bucket has ≥ _MIN_SAMPLES trades. Otherwise it
        defaults to 1.0.
        """
        trades = self._closed_trades(lookback_days)
        if len(trades) < _MIN_SAMPLES:
            logger.info(
                "Optimizer: only %d closed trades (< %d) — using neutral weights",
                len(trades), _MIN_SAMPLES,
            )
            return {
                "updated_at": datetime.now(tz=timezone.utc).isoformat(),
                "sample_size": len(trades),
                "tag_weights": {},
                "regime_weights": {},
                "symbol_weights": {},
                "baseline_expectancy": 0.0,
            }

        baseline = sum(t.get("pnl") or 0 for t in trades) / len(trades)
        # Guard against zero baseline — if overall expectancy is near
        # zero, derive weights from win rate relative to 50%.
        use_winrate = abs(baseline) < 0.01

        def _weight(bucket: dict) -> float:
            if bucket["n"] < _MIN_SAMPLES:
                return 1.0
            if use_winrate:
                # Map win rate 30% → 0.5, 50% → 1.0, 70% → 1.5
                w = 0.5 + (bucket["win_rate"] - 30) / 40
            else:
                w = bucket["expectancy"] / baseline if baseline else 1.0
            return round(max(_WEIGHT_MIN, min(_WEIGHT_MAX, w)), 3)

        tag_stats = self.by_tag(lookback_days)
        regime_stats = self.by_regime(lookback_days)
        symbol_stats = self.by_symbol(lookback_days)

        return {
            "updated_at": datetime.now(tz=timezone.utc).isoformat(),
            "sample_size": len(trades),
            "baseline_expectancy": round(baseline, 2),
            "tag_weights": {k: _weight(v) for k, v in tag_stats.items()},
            "regime_weights": {k: _weight(v) for k, v in regime_stats.items()},
            "symbol_weights": {k: _weight(v) for k, v in symbol_stats.items()},
            "tag_stats": tag_stats,
            "regime_stats": regime_stats,
            "symbol_stats": symbol_stats,
        }

    def persist_weights(self, lookback_days: int | None = 90) -> dict[str, Any]:
        """Compute weights and write them to WEIGHTS_PATH."""
        weights = self.recommend_weights(lookback_days)
        os.makedirs(WEIGHTS_PATH.parent, exist_ok=True)
        WEIGHTS_PATH.write_text(json.dumps(weights, indent=2))
        logger.info(
            "Optimizer weights saved (n=%d, baseline=%.2f)",
            weights["sample_size"], weights.get("baseline_expectancy", 0),
        )
        return weights


# ---------------------------------------------------------------------------
# Load-side helper — the signal generator uses this on every call.
# ---------------------------------------------------------------------------


_CACHED_WEIGHTS: dict[str, Any] | None = None
_CACHE_MTIME: float = 0.0


def load_weights() -> dict[str, Any]:
    """Return the latest persisted weights, or neutral defaults.

    Cached by file mtime so we only re-read when the file changes.
    Signal generator calls this every cycle, so it must be cheap.
    """
    global _CACHED_WEIGHTS, _CACHE_MTIME
    if not WEIGHTS_PATH.exists():
        return _neutral_weights()
    try:
        mtime = WEIGHTS_PATH.stat().st_mtime
    except OSError:
        return _neutral_weights()
    if _CACHED_WEIGHTS is not None and mtime == _CACHE_MTIME:
        return _CACHED_WEIGHTS
    try:
        _CACHED_WEIGHTS = json.loads(WEIGHTS_PATH.read_text())
        _CACHE_MTIME = mtime
        return _CACHED_WEIGHTS
    except Exception as exc:
        logger.warning("Failed to load optimizer weights: %s", exc)
        return _neutral_weights()


def _neutral_weights() -> dict[str, Any]:
    return {
        "updated_at": None,
        "sample_size": 0,
        "tag_weights": {},
        "regime_weights": {},
        "symbol_weights": {},
    }


def apply_weights_to_score(
    score: int,
    reasons: list[str],
    regime_name: str | None,
    symbol: str,
    weights: dict[str, Any] | None = None,
) -> tuple[int, list[str]]:
    """Apply the learned multipliers to a raw score.

    Returns (adjusted_score, adjustment_reasons). Adjustments are
    gentle — the combined multiplier is averaged across tag/regime/
    symbol so a single strong bucket can't dominate.
    """
    weights = weights or load_weights()
    if weights.get("sample_size", 0) < _MIN_SAMPLES:
        return score, []

    tag_w = weights.get("tag_weights", {})
    regime_w = weights.get("regime_weights", {})
    symbol_w = weights.get("symbol_weights", {})

    multipliers: list[float] = []
    notes: list[str] = []

    tags = extract_tags(reasons)
    tag_mults = [tag_w[t] for t in tags if t in tag_w]
    if tag_mults:
        avg = sum(tag_mults) / len(tag_mults)
        multipliers.append(avg)
        if abs(avg - 1.0) >= 0.05:
            notes.append(f"🧠 tag-fit {avg:.2f}×")

    if regime_name and regime_name in regime_w:
        multipliers.append(regime_w[regime_name])
        if abs(regime_w[regime_name] - 1.0) >= 0.05:
            notes.append(f"🧠 regime '{regime_name}' {regime_w[regime_name]:.2f}×")

    if symbol in symbol_w:
        multipliers.append(symbol_w[symbol])
        if abs(symbol_w[symbol] - 1.0) >= 0.05:
            notes.append(f"🧠 symbol {symbol_w[symbol]:.2f}×")

    if not multipliers:
        return score, []

    combined = sum(multipliers) / len(multipliers)
    adjusted = int(round(score * combined))
    return adjusted, notes
