# ICT/Wyckoff Crypto Trading Bot

A modular crypto trading bot combining ICT (Inner Circle Trader), Wyckoff, and market microstructure analysis. Sends alerts to Telegram and supports paper trading for testing.

## Architecture

```
bot.py                          ← Main loop (runs every 5 min)
├── exchange_handler.py         ← Exchange API + Paper Trader
├── strategies/
│   ├── ict_strategy.py         ← FVG, Order Blocks, Liquidity Sweeps, BOS/ChoCH
│   ├── wyckoff_strategy.py     ← Accumulation/Distribution, Springs, UTADs
│   ├── market_data.py          ← Funding Rate, OI, Volume Profile, Kill Zones
│   ├── signal_generator.py     ← Combines all → scored signal with R:R check
│   └── news_events.py          ← News sentiment + high-impact event filter
├── utils/
│   └── telegram_alerts.py      ← Formatted Telegram notifications
├── config.py                   ← Settings from .env
└── data/
    ├── paper_trades.json       ← Paper trading state (persists across restarts)
    └── bot.log                 ← Log file
```

## Signal Scoring System (0-100)

| Component       | Max Points | What it checks                              |
|-----------------|-----------|----------------------------------------------|
| ICT Analysis    | 40        | Structure, BOS/ChoCH, FVG, OB, Liq Sweeps   |
| Wyckoff Phase   | 25        | Accumulation/Distribution, Springs, UTADs    |
| Market Data     | 20        | Funding rate extremes, Volume POC            |
| Kill Zone       | 15        | Session timing (London/NY = highest weight)  |

**Minimum score to trade: 55**
**Minimum R:R enforced: 2.0** (configurable)

## Setup

### 1. Clone & Install

```bash
git clone <your-repo>
cd trading-bot
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
nano .env
```

Fill in:
- **Exchange API keys** (Bybit recommended — use testnet first!)
- **Telegram Bot Token** (create via @BotFather on Telegram)
- **Telegram Chat ID** (send a message to your bot, then visit `https://api.telegram.org/bot<TOKEN>/getUpdates`)
- **Trading parameters** (symbols, risk %, min R:R, etc.)

### 3. Test Locally

```bash
# Paper trading mode (default)
python bot.py
```

### 4. Deploy to VPS

```bash
# Copy files to your VPS
scp -r trading-bot/ user@your-vps:/home/user/

# SSH into VPS
ssh user@your-vps
cd trading-bot

# Install
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
nano .env  # fill in your keys

# Create systemd service for auto-restart
sudo nano /etc/systemd/system/trading-bot.service
```

Paste this into the service file:

```ini
[Unit]
Description=ICT/Wyckoff Crypto Trading Bot
After=network.target

[Service]
Type=simple
User=your_username
WorkingDirectory=/home/your_username/trading-bot
ExecStart=/home/your_username/trading-bot/venv/bin/python bot.py
Restart=always
RestartSec=30
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
```

Then enable and start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable trading-bot
sudo systemctl start trading-bot

# Check status
sudo systemctl status trading-bot

# View logs
journalctl -u trading-bot -f
```

## How It Works

Every 5 minutes the bot:

1. **Checks exits** — scans open positions for SL/TP hits
2. **Fetches data** — 4H candles (Wyckoff bias) + 15M candles (ICT entries)
3. **Runs ICT analysis** — detects FVGs, order blocks, liquidity sweeps, market structure
4. **Runs Wyckoff analysis** — identifies accumulation/distribution phase
5. **Checks market data** — funding rate, volume profile, kill zone timing
6. **Checks news** — pauses near high-impact events
7. **Generates signal** — scores confluence, enforces min R:R
8. **Executes** — paper or live order + Telegram alert

## Customization

### Adjust Sensitivity
- Edit `MIN_RR_RATIO` in `.env` (higher = fewer but better trades)
- Edit score thresholds in `signal_generator.py` (line ~165)
- Edit FVG minimum gap in `ict_strategy.py` (`min_gap_pct` parameter)

### Add New Strategies
Create a new file in `strategies/`, return a dict of findings, then add scoring logic in `signal_generator.py`.

### Switch to Live Trading
1. Set `PAPER_TRADING=false` in `.env`
2. Set `EXCHANGE_TESTNET=false`
3. Use real API keys with **trading permissions only** (no withdrawal)
4. Start with minimum position sizes

## Disclaimer

This bot is for educational purposes. Crypto trading carries significant risk. Always test extensively with paper trading before risking real money. Past performance does not guarantee future results.
