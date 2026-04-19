"""
Tests for Phase 4 — Wyckoff Module Enhancements.

Covers: VSA, effort vs result, event labeling, phase transitions.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from strategies.wyckoff_strategy import (
    analyze_vsa,
    analyze_effort_vs_result,
    label_wyckoff_events,
    detect_phase_transition,
    detect_trading_range,
    detect_volume_profile,
    analyze_wyckoff,
)


def _make_df(rows):
    return pd.DataFrame(rows, columns=["open", "high", "low", "close", "volume"])


def _range_df(n=60, base=100, range_pct=5, volume=1000):
    """Create a DataFrame that forms a trading range."""
    rows = []
    for i in range(n):
        mid = base + (range_pct / 2) * np.sin(i * 0.2)
        o = mid - 0.5
        h = mid + 1.5
        l = mid - 1.5
        c = mid + 0.5
        rows.append((o, h, l, c, volume))
    return _make_df(rows)


class TestVSA:
    def test_absorption_detected(self):
        # Normal candles, then high volume + narrow spread.
        rows = [(100, 102, 98, 101, 1000)] * 25
        # Absorption candle: high volume, narrow spread.
        rows.append((100, 100.5, 99.8, 100.2, 5000))
        df = _make_df(rows)
        signals = analyze_vsa(df, window=20)
        absorption = [s for s in signals if s["type"] == "absorption"]
        assert len(absorption) >= 1

    def test_selling_climax_detected(self):
        rows = [(100, 102, 98, 101, 1000)] * 25
        # Very high volume + close near low.
        rows.append((100, 102, 94, 94.5, 5000))
        df = _make_df(rows)
        signals = analyze_vsa(df, window=20)
        climax = [s for s in signals if s["type"] == "selling_climax"]
        assert len(climax) >= 1

    def test_insufficient_data(self):
        df = _make_df([(100, 102, 98, 101, 1000)] * 5)
        assert analyze_vsa(df, window=20) == []


class TestEffortVsResult:
    def test_absorption_pattern(self):
        # Lots of volume but price barely moves.
        rows = [(100, 102, 98, 101, 1000)] * 30
        # High volume window but price stays flat.
        for _ in range(10):
            rows.append((100, 101, 99, 100.1, 5000))
        df = _make_df(rows)
        signals = analyze_effort_vs_result(df, window=5)
        absorption = [s for s in signals if s["type"] == "absorption"]
        assert len(absorption) >= 1

    def test_insufficient_data(self):
        df = _make_df([(100, 102, 98, 101, 1000)] * 10)
        assert analyze_effort_vs_result(df) == []


class TestPhaseTransition:
    def test_accumulation_to_markup(self):
        rows = [(100, 103, 97, 101, 1000)] * 55
        # Breakout above range with volume.
        rows += [(101, 110, 100, 109, 3000)] * 5
        df = _make_df(rows)
        # Manually define the range that the first 55 bars form.
        tr = {"in_range": True, "range_high": 103, "range_low": 97, "midpoint": 100, "range_pct": 6.19}
        transition = detect_phase_transition(df, tr, "accumulation")
        assert transition is not None
        assert transition["transition"] == "accumulation_to_markup"

    def test_no_transition_in_range(self):
        df = _range_df(n=60)
        tr = detect_trading_range(df, window=50)
        transition = detect_phase_transition(df, tr, "accumulation")
        assert transition is None


class TestAnalyzeWyckoffPhase4:
    def test_output_has_phase4_keys(self):
        df = _range_df(n=80)
        result = analyze_wyckoff(df)
        assert "vsa_signals" in result
        assert "effort_vs_result" in result
        assert "wyckoff_events" in result
        assert "phase_transition" in result

    def test_phase_still_works(self):
        df = _range_df(n=80)
        result = analyze_wyckoff(df)
        assert result["phase"] in ("accumulation", "distribution", "markup", "markdown")
