# Trading Bot Configuration Guide

Complete guide for configuring the trading bot for Upstox paper trading.

## Quick Start

```bash
# 1. Run setup validator
python setup.py

# 2. If validation passes, start the bot
python -m src.main
```

---

## Required Configuration

### 1. Upstox API Credentials

Get these from [Upstox Developer Console](https://upstox.com/developer/api-documentation/):

| Variable | Description | Example |
|----------|-------------|---------|
| `UPSTOX_CLIENT_ID` | Your API Key | `a1b2c3d4e5f6...` |
| `UPSTOX_CLIENT_SECRET` | Your API Secret | `g7h8i9j0k1l2...` |
| `UPSTOX_REDIRECT_URI` | OAuth Callback URL | `https://127.0.0.1/callback` |

**⚠️ NEVER commit these to git! Store only in `.env` file.**

---

### 2. Trading Mode (CRITICAL!)

| Variable | Values | Description |
|----------|--------|-------------|
| `TRADING_MODE` | `paper` or `live` | `paper` = simulated orders, `live` = real orders |

```bash
# For testing - SAFE
TRADING_MODE=paper

# ⚠️ DANGER - Only when ready for real trading
TRADING_MODE=live
```

**The bot will refuse to start in live mode until all validations pass.**

---

### 3. Capital & Risk Settings

All monetary values are in **PAISA** (1 Rupee = 100 paisa):

| Variable | Default | Description | Formula |
|----------|---------|-------------|---------|
| `CAPITAL` | `100000000` | Total capital (₹10L) | `₹10,00,000 × 100 = 100000000` |
| `MAX_DAILY_LOSS` | `300000` | Daily loss limit (₹3k = 3%) | `CAPITAL × 0.03` |
| `MAX_OPEN_POSITIONS` | `5` | Max concurrent trades | - |
| `MAX_POSITION_SIZE_PCT` | `20` | Max % per position | - |
| `MAX_CAPITAL_DEPLOYMENT_PCT` | `80` | Max % of capital in use | - |
| `CONSECUTIVE_LOSS_PAUSE` | `3` | Pause after N losses | - |
| `PAUSE_MINUTES` | `30` | Minutes to pause | - |
| `SLIPPAGE_PCT` | `0.05` | Simulated slippage | 0.05% for paper trading |

**Example for ₹5 Lakhs capital:**
```bash
CAPITAL=500000000              # ₹5L in paisa
MAX_DAILY_LOSS=150000          # 3% = ₹1500
MAX_OPEN_POSITIONS=5
MAX_POSITION_SIZE_PCT=20       # Max ₹1L per position
```

---

## Optional Configuration

### 4. Automated Login (Optional)

For fully automated login without browser interaction:

| Variable | Description | Format |
|----------|-------------|--------|
| `UPSTOX_USERNAME` | Your Upstox username | - |
| `UPSTOX_PASSWORD` | Your Upstox password | - |
| `UPSTOX_PIN_CODE` | 6-digit PIN | `123456` |
| `UPSTOX_TOTP_SECRET` | TOTP secret from authenticator | Base32 string |

**To get TOTP_SECRET:**
1. Enable 2FA in Upstox
2. When scanning QR code, also copy the secret key
3. Paste that key as `UPSTOX_TOTP_SECRET`

**Without these:** You'll need to manually log in via browser on first run.

---

### 5. Telegram Notifications (Optional)

Get notifications for trades, errors, and daily P&L:

| Variable | Description | How to Get |
|----------|-------------|------------|
| `TELEGRAM_BOT_TOKEN` | Bot authentication | Message @BotFather on Telegram |
| `TELEGRAM_CHAT_ID` | Your user/chat ID | Message @userinfobot on Telegram |

**Setup:**
1. Message @BotFather, create new bot, copy token
2. Message @userinfobot, get your Chat ID
3. Start a chat with your new bot

---

### 6. Database & Logging

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | `sqlite:///data/trading_bot.db` | SQLite database path |
| `LOG_LEVEL` | `INFO` | Logging verbosity (DEBUG, INFO, WARNING, ERROR) |
| `LOG_FILE` | `logs/trading_bot.log` | Main log file path |

**For PostgreSQL (production):**
```bash
DATABASE_URL=postgresql://user:password@localhost/trading_bot
```

---

### 7. Research Module Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `RESEARCH_ENABLED` | `true` | Enable AI research module |
| `RESEARCH_MIN_SCORE` | `65` | Minimum score to trade (0-100) |
| `RESEARCH_SCORE_FOR_FULL_SIZE` | `75` | Score for full position size |
| `RESEARCH_MAX_ANALYSIS_TIME_SECONDS` | `2.0` | Max time for analysis |
| `RESEARCH_LOG_ALL` | `true` | Log all research reports |

**Score Thresholds:**
- **≥ 75:** Full position size, high confidence
- **65-74:** Reduced position size (70%)
- **< 65:** Skip trade

---

## Complete `.env` Example

```bash
# ============================================================================
# UPSTOX API CREDENTIALS (Required)
# Get from: https://upstox.com/developer/api-documentation/
# ============================================================================
UPSTOX_CLIENT_ID=your_upstox_api_key_here
UPSTOX_CLIENT_SECRET=your_upstox_api_secret_here
UPSTOX_REDIRECT_URI=https://127.0.0.1/callback

# ============================================================================
# TRADING MODE (CRITICAL!)
# ============================================================================
TRADING_MODE=paper              # 'paper' for testing, 'live' for real trading

# ============================================================================
# CAPITAL & RISK MANAGEMENT (values in PAISA - 1 Rupee = 100 paisa)
# ============================================================================
CAPITAL=100000000               # ₹10 Lakhs
MAX_DAILY_LOSS=300000           # 3% daily loss limit = ₹3000
MAX_OPEN_POSITIONS=5            # Max 5 concurrent positions
MAX_POSITION_SIZE_PCT=20        # Max 20% of capital per position
MAX_CAPITAL_DEPLOYMENT_PCT=80   # Max 80% capital in use
CONSECUTIVE_LOSS_PAUSE=3        # Pause after 3 consecutive losses
PAUSE_MINUTES=30                # Pause for 30 minutes
SLIPPAGE_PCT=0.05               # 0.05% slippage simulation

# ============================================================================
# OPTIONAL: Automated Login (manual login works without these)
# ============================================================================
# UPSTOX_USERNAME=your_username
# UPSTOX_PASSWORD=your_password
# UPSTOX_PIN_CODE=123456
# UPSTOX_TOTP_SECRET=your_totp_secret

# ============================================================================
# OPTIONAL: Telegram Notifications
# ============================================================================
# TELEGRAM_BOT_TOKEN=your_bot_token
# TELEGRAM_CHAT_ID=your_chat_id

# ============================================================================
# INFRASTRUCTURE
# ============================================================================
DATABASE_URL=sqlite:///data/trading_bot.db
LOG_LEVEL=INFO
LOG_FILE=logs/trading_bot.log

# ============================================================================
# RESEARCH MODULE (AI Signal Validation)
# ============================================================================
RESEARCH_ENABLED=true
RESEARCH_MIN_SCORE=65
RESEARCH_SCORE_FOR_FULL_SIZE=75
RESEARCH_MAX_ANALYSIS_TIME_SECONDS=2.0
RESEARCH_LOG_ALL=true
```

---

## Strategy Configuration

Edit `config/strategies/macd_crossover.json`:

```json
{
  "name": "MACD_Crossover_BankNifty",
  "enabled": true,

  "underlying": {
    "instrument_key": "NSE_INDEX|Nifty Bank",
    "symbol": "BANKNIFTY",
    "segment": "NSE_FO"
  },

  "instrument_selection": {
    "type": "options",
    "expiry_type": "weekly_current",
    "strike_selection": "atm",
    "option_types": ["CE", "PE"]
  },

  "entry_sets": [
    {
      "name": "MACD_Bullish_Cross",
      "signal": "CE",
      "conditions": [
        {
          "indicator": "MACD",
          "comparison": "crosses_above",
          "against": "MACD_Signal",
          "timeframe": "5min",
          "parameters": { "fast": 12, "slow": 26, "signal": 9 }
        },
        {
          "indicator": "RSI",
          "comparison": ">",
          "value": 50,
          "timeframe": "5min",
          "parameters": { "length": 14 }
        }
      ]
    }
  ],

  "exit_rules": {
    "stop_loss_pct": 30,
    "target_pct": 60,
    "time_based_exit": "15:10:00"
  },

  "position_sizing": {
    "method": "fixed_quantity",
    "quantity": 15
  }
}
```

### Key Strategy Settings

| Setting | Description | Common Values |
|---------|-------------|---------------|
| `underlying.symbol` | Index/stock to trade | `BANKNIFTY`, `NIFTY`, `RELIANCE` |
| `strike_selection` | Which strike to pick | `atm`, `itm_1`, `otm_1` |
| `expiry_type` | Which expiry | `weekly_current`, `monthly` |
| `stop_loss_pct` | Stop loss % | 20-40% for options |
| `target_pct` | Target % | 40-80% for options |
| `quantity` | Lots to trade | 1-25 (check your margin) |

---

## First Run Checklist

Before starting the bot for the first time:

- [ ] Created `.env` file from `.env.example`
- [ ] Added Upstox `CLIENT_ID` and `CLIENT_SECRET`
- [ ] Set `TRADING_MODE=paper` (not live!)
- [ ] Configured capital and risk settings
- [ ] Created required directories: `data/`, `logs/`
- [ ] Have at least one strategy JSON in `config/strategies/`
- [ ] Run `python setup.py` - all checks pass
- [ ] Understand that paper trading simulates orders (no real money)

---

## Common Issues

### "UPSTOX_CLIENT_ID is missing"
```bash
# Solution: Add to .env
UPSTOX_CLIENT_ID=your_actual_api_key
```

### "TRADING_MODE is set to LIVE"
```bash
# Solution: Edit .env
TRADING_MODE=paper
```

### "No strategy files found"
```bash
# Solution: Create strategy
mkdir -p config/strategies
cp config/strategies/example.json config/strategies/my_strategy.json
# Edit the file with your settings
```

### "Database initialization failed"
```bash
# Solution: Create data directory
mkdir -p data
# Check write permissions
chmod 755 data
```

---

## Safety Features

The bot has multiple safety checks:

1. **Validation Script:** `setup.py` validates all config before startup
2. **Trading Mode Check:** Refuses to run live without explicit confirmation
3. **Daily Loss Limit:** Stops trading if daily loss exceeded
4. **Position Limits:** Enforces max positions and size limits
5. **Research Module:** AI validation before each trade (score ≥ 65)
6. **Strategy Quarantine:** Auto-disables losing strategies
7. **Input Validation:** Validates all prices, quantities, symbols

---

## Next Steps

1. **Run validation:** `python setup.py`
2. **Fix any errors** reported by the validator
3. **Start in paper mode:** `python -m src.main`
4. **Monitor logs:** `tail -f logs/trading_bot.log`
5. **Check trades:** View `data/trading_bot.db` with SQLite browser
6. **After 1 week of profitable paper trading:** Consider live mode

---

## Support

- Upstox API Docs: https://upstox.com/developer/api-documentation/
- Issues: Check logs in `logs/trading_bot.log`
- Database: View with any SQLite browser
