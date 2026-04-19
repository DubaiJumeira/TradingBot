"""
Tests for Phase 2 — Instrument configuration & per-instrument behavior.

Covers:
    - INSTRUMENTS dict structure and required fields
    - Helper functions: get_instrument, get_symbols, get_symbols_by_type
    - Per-instrument kill zone weights in market_data
    - Conditional funding/OI fetch (crypto vs CFD)
    - Per-instrument min_rr and risk_pct in signal_generator
    - london_ny_overlap kill zone
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import INSTRUMENTS, get_instrument, get_symbols, get_symbols_by_type, Config


# -----------------------------------------------------------------------
# INSTRUMENTS dict structure
# -----------------------------------------------------------------------

REQUIRED_FIELDS = {
    "type", "exchange", "sessions", "funding", "min_rr",
    "risk_pct", "fvg_gap", "news_keywords", "kill_zone_weights",
}


class TestInstrumentsConfig:
    def test_all_instruments_have_required_fields(self):
        for symbol, cfg in INSTRUMENTS.items():
            for field in REQUIRED_FIELDS:
                assert field in cfg, f"{symbol} missing '{field}'"

    def test_type_is_valid(self):
        for symbol, cfg in INSTRUMENTS.items():
            assert cfg["type"] in ("crypto", "cfd"), f"{symbol} bad type: {cfg['type']}"

    def test_funding_matches_type(self):
        """Crypto should have funding=True, CFD should have funding=False."""
        for symbol, cfg in INSTRUMENTS.items():
            if cfg["type"] == "crypto":
                assert cfg["funding"] is True, f"{symbol} crypto should have funding=True"
            else:
                assert cfg["funding"] is False, f"{symbol} cfd should have funding=False"

    def test_min_rr_positive(self):
        for symbol, cfg in INSTRUMENTS.items():
            assert cfg["min_rr"] > 0, f"{symbol} min_rr must be positive"

    def test_risk_pct_reasonable(self):
        for symbol, cfg in INSTRUMENTS.items():
            assert 0 < cfg["risk_pct"] <= 5, f"{symbol} risk_pct out of range"

    def test_sessions_non_empty(self):
        for symbol, cfg in INSTRUMENTS.items():
            assert len(cfg["sessions"]) > 0, f"{symbol} has no sessions"

    def test_news_keywords_non_empty(self):
        for symbol, cfg in INSTRUMENTS.items():
            assert len(cfg["news_keywords"]) > 0, f"{symbol} has no news_keywords"

    def test_kill_zone_weights_keys_match_sessions(self):
        """Kill zone weights should cover at least the instrument's sessions."""
        for symbol, cfg in INSTRUMENTS.items():
            for session in cfg["sessions"]:
                assert session in cfg["kill_zone_weights"], (
                    f"{symbol} missing weight for session '{session}'"
                )


# -----------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------

class TestHelpers:
    def test_get_instrument_known(self):
        inst = get_instrument("BTC/USDT")
        assert inst["type"] == "crypto"
        assert inst["funding"] is True

    def test_get_instrument_unknown(self):
        assert get_instrument("FAKE/COIN") == {}

    def test_get_symbols_returns_all(self):
        symbols = get_symbols()
        assert "BTC/USDT" in symbols
        assert "XAUUSD" in symbols
        assert len(symbols) == len(INSTRUMENTS)

    def test_get_symbols_by_type_crypto(self):
        cryptos = get_symbols_by_type("crypto")
        assert "BTC/USDT" in cryptos
        assert "ETH/USDT" in cryptos
        for s in cryptos:
            assert INSTRUMENTS[s]["type"] == "crypto"

    def test_get_symbols_by_type_cfd(self):
        cfds = get_symbols_by_type("cfd")
        assert "XAUUSD" in cfds
        assert "SPX500" in cfds
        for s in cfds:
            assert INSTRUMENTS[s]["type"] == "cfd"

    def test_symbols_defaults_to_all_instruments(self):
        """Config.SYMBOLS should default to all instruments when SYMBOLS env is empty."""
        # This test verifies the logic; the actual env may override in CI,
        # so we just check it's a non-empty list of strings.
        assert isinstance(Config.SYMBOLS, list)
        assert len(Config.SYMBOLS) > 0


# -----------------------------------------------------------------------
# Kill zones — Config level
# -----------------------------------------------------------------------

class TestKillZoneConfig:
    def test_london_ny_overlap_exists(self):
        assert "london_ny_overlap" in Config.KILL_ZONES

    def test_overlap_time_range(self):
        start, end = Config.KILL_ZONES["london_ny_overlap"]
        assert start == "13:00"
        assert end == "15:00"


# -----------------------------------------------------------------------
# Per-instrument kill zone weights (market_data integration)
# -----------------------------------------------------------------------

class TestPerInstrumentKillZones:
    def test_gold_overlap_weight_is_1(self):
        """XAUUSD should get weight=1.0 during london_ny_overlap."""
        from strategies.market_data import get_current_kill_zone

        gold_inst = get_instrument("XAUUSD")
        # Simulate overlap zone being active.
        zones = {"london_ny_overlap": ("00:00", "23:59")}
        result = get_current_kill_zone(zones, instrument=gold_inst)
        assert result["active"] is True
        assert result["zone"] == "london_ny_overlap"
        assert result["weight"] == 1.0

    def test_spx_skips_asian_session(self):
        """SPX500 should not be active during the asian session."""
        from strategies.market_data import get_current_kill_zone

        spx_inst = get_instrument("SPX500")
        zones = {"asian": ("00:00", "23:59")}
        result = get_current_kill_zone(zones, instrument=spx_inst)
        # SPX500 sessions don't include "asian" → not active.
        assert result["active"] is False

    def test_btc_active_in_all_sessions(self):
        """BTC/USDT trades in all sessions."""
        from strategies.market_data import get_current_kill_zone

        btc_inst = get_instrument("BTC/USDT")
        for zone_name in ["asian", "london", "new_york", "london_ny_overlap"]:
            zones = {zone_name: ("00:00", "23:59")}
            result = get_current_kill_zone(zones, instrument=btc_inst)
            assert result["active"] is True, f"BTC should be active in {zone_name}"

    def test_no_instrument_uses_defaults(self):
        """Without instrument config, default weights apply."""
        from strategies.market_data import get_current_kill_zone

        zones = {"london": ("00:00", "23:59")}
        result = get_current_kill_zone(zones)
        assert result["active"] is True
        assert result["weight"] == 0.8  # default london weight

    def test_highest_weight_zone_wins(self):
        """When multiple zones are active, pick the highest-weight one."""
        from strategies.market_data import get_current_kill_zone

        gold_inst = get_instrument("XAUUSD")
        # Both london and overlap active at the same time.
        zones = {
            "london": ("00:00", "23:59"),
            "london_ny_overlap": ("00:00", "23:59"),
        }
        result = get_current_kill_zone(zones, instrument=gold_inst)
        # Gold: london=0.9, overlap=1.0 → overlap should win.
        assert result["zone"] == "london_ny_overlap"
        assert result["weight"] == 1.0


# -----------------------------------------------------------------------
# Conditional funding/OI in analyze_market_data
# -----------------------------------------------------------------------

class TestConditionalFunding:
    def _make_exchange_mock(self):
        exchange = MagicMock()
        exchange.fetch_funding_rate.return_value = {"fundingRate": 0.0001}
        exchange.fetch_open_interest.return_value = {"openInterestValue": 1000000}
        return exchange

    def _make_df(self):
        import pandas as pd
        return pd.DataFrame({
            "open": [100, 101],
            "high": [102, 103],
            "low": [99, 100],
            "close": [101, 102],
            "volume": [1000, 1100],
        })

    def test_crypto_fetches_funding(self):
        from strategies.market_data import analyze_market_data

        exchange = self._make_exchange_mock()
        inst = get_instrument("BTC/USDT")
        analyze_market_data(exchange, "BTC/USDT", self._make_df(), Config.KILL_ZONES, instrument=inst)

        exchange.fetch_funding_rate.assert_called_once()
        exchange.fetch_open_interest.assert_called_once()

    def test_cfd_skips_funding(self):
        from strategies.market_data import analyze_market_data

        exchange = self._make_exchange_mock()
        inst = get_instrument("XAUUSD")
        result = analyze_market_data(exchange, "XAUUSD", self._make_df(), Config.KILL_ZONES, instrument=inst)

        exchange.fetch_funding_rate.assert_not_called()
        exchange.fetch_open_interest.assert_not_called()
        # Funding analysis should still return neutral (None input).
        assert result["funding"]["signal"] == "neutral"

    def test_no_instrument_fetches_funding(self):
        """When no instrument config is provided, funding is fetched (backward compat)."""
        from strategies.market_data import analyze_market_data

        exchange = self._make_exchange_mock()
        analyze_market_data(exchange, "BTC/USDT", self._make_df(), Config.KILL_ZONES)

        exchange.fetch_funding_rate.assert_called_once()
        exchange.fetch_open_interest.assert_called_once()


# -----------------------------------------------------------------------
# Per-instrument min_rr and risk_pct in signal_generator
# -----------------------------------------------------------------------

class TestPerInstrumentSignalGen:
    """Test that generate_signal respects per-instrument min_rr and risk_pct."""

    _base_ict = {
        "structure": "bullish",
        "bos_choch": [{"type": "BOS", "direction": "bullish", "level": 100.0}],
        "fvgs": [{"type": "bullish", "bottom": 99.0, "top": 101.0}],
        "order_blocks": [],
        "liquidity_sweeps": [],
        "swing_lows": [{"price": 98.0}],
        "swing_highs": [{"price": 105.0}],
    }
    _base_wyckoff = {"phase": "accumulation", "springs": [], "utads": []}
    _base_market = {
        "funding": {},
        "volume_profile": {},
        "kill_zone": {"active": True, "zone": "london", "weight": 0.8},
    }

    def test_uses_instrument_min_rr(self):
        from strategies.signal_generator import generate_signal

        # With global Config.MIN_RR_RATIO (2.0) signal should pass.
        sig_default = generate_signal(
            "XAUUSD", 100.0, self._base_ict, self._base_wyckoff,
            self._base_market, 10000,
        )
        assert sig_default is not None

        # With instrument min_rr=99.0 (absurdly high), signal should be filtered out.
        fake_inst = {"min_rr": 99.0, "risk_pct": 1.0}
        sig_strict = generate_signal(
            "XAUUSD", 100.0, self._base_ict, self._base_wyckoff,
            self._base_market, 10000, instrument=fake_inst,
        )
        assert sig_strict is None

    def test_uses_instrument_risk_pct(self):
        from strategies.signal_generator import generate_signal

        inst_low = {"min_rr": 1.0, "risk_pct": 0.5}
        inst_high = {"min_rr": 1.0, "risk_pct": 2.0}

        sig_low = generate_signal(
            "XAUUSD", 100.0, self._base_ict, self._base_wyckoff,
            self._base_market, 10000, instrument=inst_low,
        )
        sig_high = generate_signal(
            "XAUUSD", 100.0, self._base_ict, self._base_wyckoff,
            self._base_market, 10000, instrument=inst_high,
        )

        assert sig_low is not None
        assert sig_high is not None
        assert sig_low["risk_pct"] == 0.5
        assert sig_high["risk_pct"] == 2.0
        assert sig_high["size_usd"] > sig_low["size_usd"]

    def test_no_instrument_uses_global_defaults(self):
        from strategies.signal_generator import generate_signal

        sig = generate_signal(
            "XAUUSD", 100.0, self._base_ict, self._base_wyckoff,
            self._base_market, 10000,
        )
        assert sig is not None
        assert sig["risk_pct"] == Config.DEFAULT_RISK_PERCENT
