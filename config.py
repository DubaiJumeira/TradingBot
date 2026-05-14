import os
from dotenv import load_dotenv

load_dotenv()


# ---------------------------------------------------------------------------
# Instrument definitions
# ---------------------------------------------------------------------------
# Each entry fully describes how the bot should treat one tradeable symbol.
#
#   type            "crypto" | "cfd"
#   exchange        ccxt exchange id (only crypto instruments use this today)
#   sessions        which kill zones this instrument trades in
#   funding         True → fetch funding rate / OI (crypto only)
#   min_rr          per-instrument minimum Risk:Reward ratio
#   risk_pct        per-instrument risk % of account per trade
#   fvg_gap         minimum FVG gap % to qualify (wider for volatile assets)
#   news_keywords   keywords that link news to this instrument
#   kill_zone_weights  per-instrument override for kill zone importance
# ---------------------------------------------------------------------------

INSTRUMENTS: dict[str, dict] = {
    # --- Crypto -----------------------------------------------------------
    "BTC/USDT": {
        "type": "crypto",
        "exchange": "bybit",
        "sessions": ["asian", "london", "new_york", "london_ny_overlap"],
        "funding": True,
        "min_rr": 2.0,
        "risk_pct": 1.0,
        "fvg_gap": 0.10,
        "news_keywords": ["bitcoin", "BTC", "crypto"],
        "kill_zone_weights": {
            "asian": 0.6,
            "london": 0.8,
            "new_york": 1.0,
            "london_ny_overlap": 0.9,
        },
    },
    "ETH/USDT": {
        "type": "crypto",
        "exchange": "bybit",
        "sessions": ["asian", "london", "new_york", "london_ny_overlap"],
        "funding": True,
        "min_rr": 2.0,
        "risk_pct": 1.0,
        "fvg_gap": 0.12,
        "news_keywords": ["ethereum", "ETH", "crypto"],
        "kill_zone_weights": {
            "asian": 0.6,
            "london": 0.8,
            "new_york": 1.0,
            "london_ny_overlap": 0.9,
        },
    },
    "SOL/USDT": {
        "type": "crypto",
        "exchange": "bybit",
        "sessions": ["asian", "london", "new_york", "london_ny_overlap"],
        "funding": True,
        "min_rr": 2.0,
        "risk_pct": 0.75,
        "fvg_gap": 0.15,
        "news_keywords": ["solana", "SOL", "crypto"],
        "kill_zone_weights": {
            "asian": 0.5,
            "london": 0.7,
            "new_york": 1.0,
            "london_ny_overlap": 0.85,
        },
    },

    # --- Gold (XAU/USD CFD) -----------------------------------------------
    "XAUUSD": {
        "type": "cfd",
        "exchange": "bybit",
        "sessions": ["london", "new_york", "london_ny_overlap"],
        "funding": False,
        "min_rr": 1.8,
        "risk_pct": 1.0,
        "fvg_gap": 0.05,
        "news_keywords": ["gold", "XAUUSD", "safe haven", "precious metals"],
        "kill_zone_weights": {
            "asian": 0.3,
            "london": 0.9,
            "new_york": 1.0,
            "london_ny_overlap": 1.0,  # Gold loves the overlap
        },
    },

    # --- Oil (WTI CFD) ----------------------------------------------------
    "XTIUSD": {
        "type": "cfd",
        "exchange": "bybit",
        "sessions": ["london", "new_york", "london_ny_overlap"],
        "funding": False,
        "min_rr": 1.8,
        "risk_pct": 0.75,
        "fvg_gap": 0.08,
        "news_keywords": ["oil", "WTI", "crude", "OPEC", "energy"],
        "kill_zone_weights": {
            "asian": 0.2,
            "london": 0.7,
            "new_york": 1.0,
            "london_ny_overlap": 0.9,
        },
    },

    # --- US Indices (CFD) -------------------------------------------------
    "SPX500": {
        "type": "cfd",
        "exchange": "bybit",
        "sessions": ["new_york", "london_ny_overlap"],
        "funding": False,
        "min_rr": 1.5,
        "risk_pct": 1.0,
        "fvg_gap": 0.05,
        "news_keywords": ["S&P 500", "SPX", "stocks", "equities"],
        "kill_zone_weights": {
            "asian": 0.2,
            "london": 0.4,
            "new_york": 1.0,
            "london_ny_overlap": 0.8,
        },
    },
    "US30": {
        "type": "cfd",
        "exchange": "bybit",
        "sessions": ["new_york", "london_ny_overlap"],
        "funding": False,
        "min_rr": 1.5,
        "risk_pct": 1.0,
        "fvg_gap": 0.05,
        "news_keywords": ["Dow Jones", "US30", "DJIA", "stocks"],
        "kill_zone_weights": {
            "asian": 0.2,
            "london": 0.4,
            "new_york": 1.0,
            "london_ny_overlap": 0.8,
        },
    },
    "NAS100": {
        "type": "cfd",
        "exchange": "bybit",
        "sessions": ["new_york", "london_ny_overlap"],
        "funding": False,
        "min_rr": 1.5,
        "risk_pct": 0.75,
        "fvg_gap": 0.06,
        "news_keywords": ["Nasdaq", "NAS100", "tech stocks", "QQQ"],
        "kill_zone_weights": {
            "asian": 0.2,
            "london": 0.4,
            "new_york": 1.0,
            "london_ny_overlap": 0.8,
        },
    },
}


# ---------------------------------------------------------------------------
# Correlation groups — instruments that move together
# ---------------------------------------------------------------------------
# Used by the risk manager to prevent overexposure to correlated assets.
# Each group has a max_positions limit.

CORRELATION_GROUPS: dict[str, dict] = {
    "crypto": {
        "symbols": ["BTCUSDT", "ETHUSDT", "SOLUSDT"],
        "max_positions": 2,
        "max_same_direction": 1,
    },
    "safe_haven": {
        "symbols": ["XAUUSD"],
        "max_positions": 1,
        "max_same_direction": 1,
    },
    "energy": {
        "symbols": ["XTIUSD"],
        "max_positions": 1,
        "max_same_direction": 1,
    },
    "us_indices": {
        "symbols": ["SPX500", "US30", "NAS100"],
        "max_positions": 2,
        "max_same_direction": 1,
    },
}


def get_correlation_group(symbol: str) -> str | None:
    """Return the correlation group name for a symbol, or None."""
    for group_name, group in CORRELATION_GROUPS.items():
        if symbol in group["symbols"]:
            return group_name
    return None


def get_instrument(symbol: str) -> dict:
    """Look up instrument config by symbol. Returns empty dict if unknown."""
    return INSTRUMENTS.get(symbol, {})


def get_symbols() -> list[str]:
    """Return all configured instrument symbols (replaces Config.SYMBOLS)."""
    return list(INSTRUMENTS.keys())


def get_symbols_by_type(inst_type: str) -> list[str]:
    """Return symbols filtered by type ('crypto' or 'cfd')."""
    return [s for s, cfg in INSTRUMENTS.items() if cfg["type"] == inst_type]


class Config:
    # Exchange
    EXCHANGE = os.getenv("EXCHANGE", "bybit")
    EXCHANGE_API_KEY = os.getenv("EXCHANGE_API_KEY", "")
    EXCHANGE_SECRET = os.getenv("EXCHANGE_SECRET", "")
    EXCHANGE_TESTNET = os.getenv("EXCHANGE_TESTNET", "true").lower() == "true"

    # Telegram
    TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
    TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

    # Trading
    # SYMBOLS is now derived from INSTRUMENTS. Env override still works for
    # restricting which instruments to trade (comma-separated subset).
    _env_symbols = os.getenv("SYMBOLS", "")
    SYMBOLS: list[str] = (
        _env_symbols.split(",") if _env_symbols
        else get_symbols()
    )
    TIMEFRAMES = os.getenv("TIMEFRAMES", "5m,15m,1h,4h,1D").split(",")
    DEFAULT_RISK_PERCENT = float(os.getenv("DEFAULT_RISK_PERCENT", "1.0"))
    MIN_RR_RATIO = float(os.getenv("MIN_RR_RATIO", "2.0"))
    MAX_OPEN_TRADES = int(os.getenv("MAX_OPEN_TRADES", "3"))
    PAPER_TRADING = os.getenv("PAPER_TRADING", "true").lower() == "true"
    STARTING_BALANCE = float(os.getenv("STARTING_BALANCE", "10000"))

    # Position sizing: max notional size per trade as % of balance.
    # Combined with 1% risk and leverage capping, this prevents oversizing.
    MAX_POSITION_PCT = float(os.getenv("MAX_POSITION_PCT", "0.40"))  # 40% of balance max

    # Risk Management (Phase 6)
    MAX_DRAWDOWN_PCT = float(os.getenv("MAX_DRAWDOWN_PCT", "10.0"))
    TRAILING_STOP_ACTIVATION_RR = float(os.getenv("TRAILING_ACTIVATION_RR", "1.0"))
    TRAILING_STOP_STEP_PCT = float(os.getenv("TRAILING_STEP_PCT", "0.5"))
    ATR_POSITION_SCALE = float(os.getenv("ATR_POSITION_SCALE", "1.5"))

    # Strategy selection — "momentum_breakout" runs the new 4H Donchian
    # breakout system; "legacy_confluence" keeps the ICT/Wyckoff pipeline.
    STRATEGY_MODE = os.getenv("STRATEGY_MODE", "legacy_confluence")

    # Momentum breakout parameters
    MOMENTUM_SMA_LONG = int(os.getenv("MOMENTUM_SMA_LONG", "200"))
    MOMENTUM_SMA_SHORT = int(os.getenv("MOMENTUM_SMA_SHORT", "50"))
    MOMENTUM_DONCHIAN = int(os.getenv("MOMENTUM_DONCHIAN", "20"))
    MOMENTUM_ATR_PERIOD = int(os.getenv("MOMENTUM_ATR_PERIOD", "14"))
    MOMENTUM_ATR_MEDIAN_PERIOD = int(os.getenv("MOMENTUM_ATR_MEDIAN_PERIOD", "50"))
    MOMENTUM_ATR_STOP_MULT = float(os.getenv("MOMENTUM_ATR_STOP_MULT", "3.0"))
    MOMENTUM_RISK_PCT = float(os.getenv("MOMENTUM_RISK_PCT", "1.0"))  # 1% of equity per trade
    MOMENTUM_FEE_PCT = float(os.getenv("MOMENTUM_FEE_PCT", "0.05"))  # taker fee per side
    MOMENTUM_SLIPPAGE_PCT = float(os.getenv("MOMENTUM_SLIPPAGE_PCT", "0.05"))  # per side
    MOMENTUM_FUNDING_PCT_8H = float(os.getenv("MOMENTUM_FUNDING_PCT_8H", "0.01"))  # per 8h
    MOMENTUM_FORCE_TEST_SIGNAL = (
        os.getenv("MOMENTUM_FORCE_TEST_SIGNAL", "false").lower() == "true"
    )

    # Kill Zones (UTC) — includes the new london_ny_overlap zone.
    KILL_ZONES = {
        "asian": (os.getenv("ASIAN_KZ_START", "00:00"), os.getenv("ASIAN_KZ_END", "03:00")),
        "london": (os.getenv("LONDON_KZ_START", "07:00"), os.getenv("LONDON_KZ_END", "10:00")),
        "new_york": (os.getenv("NY_KZ_START", "13:00"), os.getenv("NY_KZ_END", "16:00")),
        "london_ny_overlap": (
            os.getenv("OVERLAP_KZ_START", "13:00"),
            os.getenv("OVERLAP_KZ_END", "15:00"),
        ),
    }
