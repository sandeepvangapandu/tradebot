# Claude Project Instructions — Autonomous Trading Bot (Indian Markets / Upstox)

---

## PROJECT OVERVIEW

You are helping me build an **autonomous trading bot** for the **Indian stock and derivatives markets** using the **Upstox broker platform**. The bot will eventually trade live, but **Phase 1 is paper trading only** — all orders are simulated using Upstox's sandbox/paper trading environment.

The tech stack is **Python** unless I specify otherwise. The codebase should be modular, production-grade, well-tested, and designed for eventual live deployment.

---

## CORE ARCHITECTURE & REQUIREMENTS

### 1. Broker Integration — Upstox API v2
- Use the **Upstox Python SDK** (`upstox-python-sdk`) or direct REST API calls to Upstox API v2.
- API docs reference: https://upstox.com/developer/api-documentation/
- Implement **OAuth 2.0 authentication flow** (authorization code grant) for Upstox.
- Store tokens securely (environment variables or encrypted config — never hardcode).
- Handle token refresh automatically before expiry.
- Implement a **broker adapter/interface layer** so the bot isn't tightly coupled to Upstox — this makes it easier to add other brokers later.

### 2. Market Data
- Use Upstox **WebSocket feed** for real-time market data (LTP, OHLC, depth).
- Implement **historical data fetching** via Upstox REST API for backtesting and indicator calculation.
- Support instruments: **NSE Equity, NSE F&O (Futures & Options), BSE Equity**.
- Cache instrument master files daily (Upstox provides instrument dumps).
- Handle Indian market hours: Pre-open (9:00–9:15 AM IST), Regular (9:15 AM–3:30 PM IST), Post-close.

### 3. Paper Trading Engine
- Build a **simulated order management system (OMS)** that mimics real order placement.
- Simulate: Market orders, Limit orders, Stop-Loss orders, Stop-Loss Market orders, Bracket orders, Cover orders.
- Track: Positions, Holdings, P&L (realized + unrealized), Order book, Trade book.
- Apply realistic **slippage** (configurable, default 0.05%), **brokerage** (Upstox fee structure), **STT, GST, stamp duty, exchange charges** for accurate P&L.
- Log every simulated trade with timestamps, entry/exit prices, quantity, and P&L.
- Paper trading should use the **same code paths** as live trading — only the execution layer differs.

### 4. Strategy Framework
- Design a **pluggable strategy architecture**: each strategy is a Python class inheriting from a `BaseStrategy` interface.
- BaseStrategy interface should define: `on_tick()`, `on_candle()`, `generate_signals()`, `on_order_update()`, `on_position_update()`.
- Support **multiple strategies running concurrently** on different instruments.
- Include these **starter strategies** (implement from scratch):
  - **Moving Average Crossover** (EMA 9/21)
  - **RSI Mean Reversion** (RSI 14, oversold < 30, overbought > 70)
  - **VWAP Breakout** (for intraday)
  - **Supertrend** (ATR-based trend following)
- All strategies must define: entry rules, exit rules, stop-loss, target, position sizing, max positions.

### 5. Risk Management (CRITICAL)
- **Per-trade risk**: Max loss per trade (configurable, default 1% of capital).
- **Daily loss limit**: Stop trading if daily loss exceeds X% (default 3%).
- **Max open positions**: Configurable limit (default 5).
- **Max capital deployment**: Never exceed X% of total capital (default 80%).
- **Per-instrument exposure limit**: No single stock > X% of capital (default 20%).
- **Circuit breaker**: If 3 consecutive losses, pause for N minutes.
- **Kill switch**: Manual emergency stop that exits all positions.
- All risk parameters in a YAML/JSON config file.

### 6. Technical Indicators
- Use **pandas-ta** or **ta-lib** for indicator calculations.
- Pre-build these: EMA, SMA, RSI, MACD, Bollinger Bands, ATR, Supertrend, VWAP, OBV, ADX.
- Indicators should be computed on DataFrames and cached efficiently.

### 7. Data Storage & Logging
- Use **SQLite** for paper trading (upgradeable to PostgreSQL for production).
- Tables: `orders`, `trades`, `positions`, `daily_pnl`, `strategy_signals`, `market_data_cache`.
- Structured logging with Python's `logging` module — log levels: DEBUG, INFO, WARNING, ERROR.
- Separate log files: `trading.log`, `orders.log`, `errors.log`.
- Log rotation (daily, max 30 days).

### 8. Dashboard & Monitoring
- Build a simple **Streamlit** or **FastAPI + HTML** dashboard showing:
  - Current positions and P&L
  - Today's trades
  - Strategy performance metrics
  - Equity curve chart
  - Active strategies and their signals
- Optional: Telegram bot notifications for trades, daily P&L summary, and alerts.

### 9. Backtesting Engine
- Ability to run strategies against **historical data**.
- Generate reports: Total P&L, Win rate, Sharpe ratio, Max drawdown, Profit factor, Average trade duration.
- Plot equity curve, drawdown chart, trade distribution.

### 10. Configuration Management
- All parameters in `config.yaml` or `.env`:
  ```
  UPSTOX_API_KEY=
  UPSTOX_API_SECRET=
  UPSTOX_REDIRECT_URI=
  TRADING_MODE=paper  # paper | live
  CAPITAL=1000000     # INR
  MAX_RISK_PER_TRADE=0.01
  DAILY_LOSS_LIMIT=0.03
  MAX_POSITIONS=5
  STRATEGIES=ema_crossover,rsi_reversal
  INSTRUMENTS=NSE_EQ:RELIANCE,NSE_EQ:TCS,NSE_EQ:INFY
  LOG_LEVEL=INFO
  ```

---

## PROJECT STRUCTURE

```
trading-bot/
├── config/
│   ├── config.yaml
│   ├── instruments.yaml
│   └── strategies.yaml
├── src/
│   ├── __init__.py
│   ├── main.py                  # Entry point
│   ├── broker/
│   │   ├── __init__.py
│   │   ├── base_broker.py       # Abstract broker interface
│   │   ├── upstox_broker.py     # Upstox implementation
│   │   └── paper_broker.py      # Paper trading broker
│   ├── data/
│   │   ├── __init__.py
│   │   ├── market_feed.py       # WebSocket handler
│   │   ├── historical.py        # Historical data fetcher
│   │   └── instruments.py       # Instrument master management
│   ├── strategy/
│   │   ├── __init__.py
│   │   ├── base_strategy.py     # Strategy interface
│   │   ├── ema_crossover.py
│   │   ├── rsi_reversal.py
│   │   ├── vwap_breakout.py
│   │   └── supertrend.py
│   ├── risk/
│   │   ├── __init__.py
│   │   └── risk_manager.py
│   ├── orders/
│   │   ├── __init__.py
│   │   └── order_manager.py
│   ├── indicators/
│   │   ├── __init__.py
│   │   └── technical.py
│   ├── backtest/
│   │   ├── __init__.py
│   │   └── backtester.py
│   ├── dashboard/
│   │   ├── __init__.py
│   │   └── app.py               # Streamlit/FastAPI dashboard
│   ├── notifications/
│   │   ├── __init__.py
│   │   └── telegram_bot.py
│   └── utils/
│       ├── __init__.py
│       ├── database.py
│       ├── logger.py
│       └── helpers.py
├── tests/
│   ├── test_broker.py
│   ├── test_strategies.py
│   ├── test_risk.py
│   └── test_backtest.py
├── data/
│   └── instruments/             # Cached instrument files
├── logs/
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## CODING STANDARDS & CONVENTIONS

- **Python 3.11+**
- Type hints on all functions.
- Docstrings (Google style) on all classes and public methods.
- Use `async/await` for WebSocket and I/O-bound operations.
- Use `dataclasses` or `Pydantic` models for data structures (Orders, Positions, Trades).
- Follow PEP 8. Use `black` for formatting, `ruff` for linting.
- Write unit tests with `pytest`. Target 80%+ coverage on core logic.
- Error handling: never let the bot crash silently — catch, log, and recover gracefully.
- All monetary values in **Paisa (integer)** internally to avoid floating-point issues, convert to Rupees only for display.

---

## INDIAN MARKET SPECIFIC RULES

- **Trading holidays**: Load NSE holiday calendar. Do not attempt to trade on holidays.
- **Circuit limits**: Respect upper/lower circuit limits on stocks.
- **Lot sizes**: F&O instruments have specific lot sizes — fetch from instrument master.
- **T+1 settlement**: Equity delivery is T+1 in India. Track this in the holdings logic.
- **Intraday square-off**: All intraday (MIS) positions must be squared off before 3:15 PM IST.
- **Tick size**: NSE equity tick size is ₹0.05.
- **Market segments**: Clearly distinguish between NSE, BSE, NFO (F&O), CDS, MCX.
- **Timezone**: All timestamps must be in **IST (Asia/Kolkata)**.

---

## PHASE-WISE DEVELOPMENT PLAN

### Phase 1 — Foundation (Current Phase)
- [ ] Project setup, folder structure, config management
- [ ] Upstox OAuth2 authentication
- [ ] Instrument master download and caching
- [ ] Market data feed (WebSocket + historical)
- [ ] Paper trading OMS
- [ ] One working strategy (EMA Crossover)
- [ ] Basic risk management
- [ ] SQLite database and logging
- [ ] Unit tests for core modules

### Phase 2 — Strategy Expansion
- [ ] Add RSI, VWAP, Supertrend strategies
- [ ] Backtesting engine
- [ ] Performance reporting
- [ ] Strategy parameter optimization

### Phase 3 — Monitoring & Alerts
- [ ] Streamlit dashboard
- [ ] Telegram notifications
- [ ] Daily P&L reports

### Phase 4 — Live Trading Preparation
- [ ] Switch paper broker to live broker (same interface)
- [ ] Add comprehensive error handling for network/API failures
- [ ] Add order reconciliation
- [ ] Stress testing
- [ ] Go-live with minimal capital

---

## IMPORTANT CONSTRAINTS

1. **NEVER place real orders in Phase 1.** All execution must go through `PaperBroker`.
2. **NEVER hardcode API keys** — always use environment variables.
3. **Always respect risk limits** — the risk manager should be checked BEFORE any order.
4. **Assume unreliable network** — implement retries, reconnection logic for WebSocket.
5. **Indian regulations**: Be aware that fully automated trading by retail traders in India has regulatory considerations with SEBI. This bot is for personal use and paper trading/learning.

---

## HOW TO INTERACT WITH ME

- Build **one module at a time**, test it, then move to the next.
- After writing code, always explain **what it does, how it fits the architecture, and what to test**.
- If I ask to implement a strategy, provide the **full strategy code + a backtest example**.
- When suggesting improvements, explain **tradeoffs** (speed vs accuracy, complexity vs maintainability).
- Proactively flag **potential bugs, edge cases, or risks** you notice.
- If something is unclear, ask before assuming.

---

## REFERENCE LINKS (for context, not for fetching)
- Upstox API v2 Docs: https://upstox.com/developer/api-documentation/
- Upstox Python SDK: https://github.com/upstox/upstox-python
- NSE Holidays: https://www.nseindia.com/resources/exchange-communication-holidays
- pandas-ta: https://github.com/twopirllc/pandas-ta
