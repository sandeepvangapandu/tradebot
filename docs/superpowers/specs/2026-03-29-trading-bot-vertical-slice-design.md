# Upstox Autonomous Trading Bot — Vertical Slice (Phase 1) Design

**Date:** 2026-03-29
**Status:** Draft
**Scope:** End-to-end paper trading bot with one strategy (MACD Crossover on Bank Nifty options)

---

## 1. Overview

Build a runnable paper-trading bot for Indian equity/F&O markets using Upstox API v2/v3. The bot authenticates daily via TOTP, streams real-time market data, evaluates a JSON-driven MACD crossover strategy, simulates order execution, enforces risk limits, and persists all state to SQLite.

### Success Criteria

- Bot starts unattended, authenticates via TOTP, and connects to Upstox WebSocket
- Instrument master downloaded and searchable (equities + F&O)
- MACD crossover strategy evaluates conditions against live data and generates signals
- Paper trading engine simulates order fills using LTP from WebSocket
- Risk manager blocks trades that violate limits (daily loss, position count, exposure)
- All orders, trades, positions, and P&L persisted to SQLite
- Graceful shutdown saves state; restart recovers open positions
- Tests cover indicators, risk checks, strategy conditions, paper execution

### What's OUT of Scope

- Telegram notifications
- Backtesting engine
- Docker containerization
- Additional strategies (RSI, VWAP, SuperTrend)
- Options Greeks computation module
- Streamlit/FastAPI dashboard

---

## 2. Architecture

### 2.1 Directory Structure

```
upstox-trading-bot/
├── pyproject.toml
├── requirements.txt
├── .env.example
├── .gitignore
├── config/
│   ├── settings.py              # Pydantic settings from .env
│   ├── constants.py             # Market hours, segments, fees, tick sizes
│   └── strategies/
│       └── macd_crossover.json  # Example strategy config
├── src/
│   ├── __init__.py
│   ├── main.py                  # Entry point / orchestrator
│   ├── auth/
│   │   ├── __init__.py
│   │   ├── auto_login.py        # TOTP-based auto-login (upstox-totp)
│   │   └── token_manager.py     # Token caching, validation, daily refresh
│   ├── data/
│   │   ├── __init__.py
│   │   ├── websocket_feed.py    # MarketDataStreamerV3 wrapper
│   │   ├── portfolio_feed.py    # PortfolioDataStreamer wrapper
│   │   ├── historical.py        # V3 historical candle fetcher
│   │   └── instruments.py       # Instrument file download, search, option chain
│   ├── strategy/
│   │   ├── __init__.py
│   │   ├── engine.py            # Multi-threaded strategy evaluator
│   │   ├── builder.py           # JSON strategy parser/validator
│   │   ├── conditions.py        # Condition evaluation logic
│   │   └── indicators.py        # Technical indicators via pandas-ta
│   ├── execution/
│   │   ├── __init__.py
│   │   ├── order_manager.py     # Order placement (live V3 API / paper sim)
│   │   ├── position_manager.py  # Position tracking, P&L
│   │   └── order_tracker.py     # Order status, fill tracking
│   ├── risk/
│   │   ├── __init__.py
│   │   ├── risk_manager.py      # Pre-trade checks, daily loss, limits
│   │   └── circuit_breaker.py   # Emergency stop, kill switch
│   ├── persistence/
│   │   ├── __init__.py
│   │   ├── database.py          # SQLAlchemy engine/session setup
│   │   ├── models.py            # ORM models
│   │   └── trade_log.py         # Trade audit logging
│   └── utils/
│       ├── __init__.py
│       ├── logger.py            # Loguru setup (file + console + rotation)
│       ├── rate_limiter.py      # Token bucket (40 req/sec)
│       ├── scheduler.py         # APScheduler for daily tasks
│       ├── helpers.py           # IST time, data formatting
│       └── exceptions.py        # Custom exception hierarchy
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_indicators.py
│   ├── test_strategy_engine.py
│   ├── test_risk_manager.py
│   ├── test_order_manager.py
│   └── test_instruments.py
└── data/
    └── instruments/             # Cached instrument JSON files
```

### 2.2 Data Flow

```
[Upstox WebSocket] ──ticks──> [WebSocket Feed]
                                    │
                                    ├──> [OHLCV Bar Builder] (1m, 5m, 15m in-memory)
                                    │         │
                                    │         └──> [Strategy Engine] (per-set threads)
                                    │                   │
                                    │              signal generated
                                    │                   │
                                    │              [Risk Manager] ── block ──> log + skip
                                    │                   │ approved
                                    │                   │
                                    │              [Order Manager]
                                    │              (paper: sim fill from LTP)
                                    │              (live: V3 API call)
                                    │                   │
                                    │              [Position Manager]
                                    │                   │
                                    └──> [Position Manager] (tick-level SL/target monitoring)
                                                        │
                                                   [SQLite DB]

[Portfolio WebSocket] ──updates──> [Order Tracker] ──> [Position Manager] ──> [DB]

[APScheduler]
  ├── 06:15 IST: Download instruments
  ├── 08:45 IST: Auto-login (TOTP)
  ├── 09:00 IST: Reconcile positions with Upstox API
  ├── 15:15 IST: Force exit all intraday positions
  └── 15:30 IST: Log daily P&L summary
```

### 2.3 Threading Model

| Thread | Purpose |
|--------|---------|
| Main | Scheduler, signal handlers, lifecycle management |
| WebSocket-Market | MarketDataStreamerV3 — receives ticks, dispatches to subscribers |
| WebSocket-Portfolio | PortfolioDataStreamer — order/position update events |
| Strategy-Set-N | One thread per strategy set evaluating conditions in a loop |
| Bar-Builder | Consumes ticks from queue, builds OHLCV candles |

Communication between threads uses `queue.Queue` (thread-safe). Strategy threads read from shared candle data (read-only after bar close) and write signals to an order queue consumed by the main thread's order manager.

---

## 3. Module Specifications

### 3.1 Config (`config/`)

**settings.py** — Pydantic `BaseSettings` loading from `.env`:
- `upstox_username`, `upstox_password`, `upstox_pin_code`, `upstox_totp_secret`
- `upstox_client_id`, `upstox_client_secret`, `upstox_redirect_uri`
- `trading_mode`: Literal["paper", "live"] (default: "paper")
- `default_exchange`: str (default: "NSE")
- `max_daily_loss`: int (paisa, default: 500000 = Rs 5000)
- `max_open_positions`: int (default: 5)
- `max_position_size_pct`: int (default: 20)
- `capital`: int (paisa, default: 100000000 = Rs 10 lakh)
- `database_url`, `log_level`, `log_file`

**constants.py** — Static values:
- `MARKET_OPEN = time(9, 15)`, `MARKET_CLOSE = time(15, 30)`, `PRE_OPEN_START = time(9, 0)`
- `INTRADAY_SQUARE_OFF = time(15, 15)`
- `IST = ZoneInfo("Asia/Kolkata")`
- `TICK_SIZE_NSE = 5` (paisa = Rs 0.05)
- Brokerage/fee structure: STT, GST, stamp duty, exchange charges, Upstox brokerage
- Instrument file URLs per exchange

### 3.2 Utils (`src/utils/`)

**logger.py** — Loguru configuration:
- Console: INFO+ with colored formatting
- File: `logs/trading_bot.log`, rotation 10MB, retention 30 days
- Separate error file: `logs/errors.log`, ERROR+ only
- Orders file: `logs/orders.log`, filtered by "ORDER" tag

**exceptions.py** — Custom exception hierarchy:
```python
class TradingBotError(Exception): ...
class AuthenticationError(TradingBotError): ...
class TokenExpiredError(AuthenticationError): ...
class OrderError(TradingBotError): ...
class InsufficientMarginError(OrderError): ...
class RiskLimitBreachedError(TradingBotError): ...
class RateLimitError(TradingBotError): ...
class WebSocketError(TradingBotError): ...
class InstrumentError(TradingBotError): ...
```

**rate_limiter.py** — Token bucket algorithm:
- Capacity: 40 tokens, refill rate: 40/sec
- `acquire()` blocks until a token is available
- Used as a decorator or context manager on API call functions

**helpers.py** — Utility functions:
- `now_ist() -> datetime` — current time in IST
- `is_market_open() -> bool` — checks against market hours and holiday calendar
- `rupees_to_paisa(r: float) -> int` and `paisa_to_rupees(p: int) -> float`
- `format_instrument_key(segment: str, identifier: str) -> str`

**scheduler.py** — APScheduler wrapper:
- `BackgroundScheduler` with IST timezone
- Methods to add daily jobs (login, instrument download, forced exit, etc.)
- Graceful shutdown on bot stop

### 3.3 Auth (`src/auth/`)

**auto_login.py**:
- Uses `upstox_totp.UpstoxTOTP` which reads credentials from env vars
- Calls `upx.app_token.get_access_token()` to get token without browser
- Returns access token string or raises `AuthenticationError`
- Handles sandbox mode: uses `upstox_client.Configuration(sandbox=True)` when `trading_mode=paper`

**token_manager.py**:
- Caches token to `data/token_cache.json` with fields: `access_token`, `obtained_at` (ISO timestamp), `expires_at` (3:30 AM IST next day)
- `get_valid_token() -> str` — returns cached token if still valid, otherwise triggers auto-login
- `is_token_valid() -> bool` — checks current time against expiry
- `refresh_token()` — forces re-authentication
- Scheduled to run at 08:45 IST daily

### 3.4 Data — Instruments (`src/data/instruments.py`)

**InstrumentManager** class:
- `download_instruments(exchanges: list[str])` — fetches gzipped JSON from Upstox CDN, saves to `data/instruments/`
- `load_instruments() -> pd.DataFrame` — reads cached JSON into DataFrame with columns: segment, name, instrument_key, trading_symbol, lot_size, tick_size, expiry, strike_price, option_type, underlying_key, etc.
- `search(segment, symbol) -> list[dict]` — filter instruments
- `get_option_chain(underlying_key, expiry_date) -> pd.DataFrame` — returns CE/PE pairs for all strikes
- `get_atm_strike(underlying_key, spot_price, expiry_date) -> dict` — finds nearest strike to spot
- `get_weekly_expiry(underlying_key) -> date` — returns current/next weekly expiry
- `get_instrument_key(segment, symbol) -> str` — lookup instrument key by trading symbol

Scheduled to refresh at 06:15 IST daily.

### 3.5 Data — WebSocket Feed (`src/data/websocket_feed.py`)

**MarketDataFeed** class:
- Wraps `upstox_client.MarketDataStreamerV3`
- Constructor takes access token and list of instrument keys
- `start()` — connects with `auto_reconnect(enable=True, interval=5, retry_count=50)`
- `subscribe(instrument_keys, mode="full")` / `unsubscribe(instrument_keys)`
- `on_message` callback parses tick data, pushes to:
  - `tick_queue: Queue` — consumed by bar builder
  - Registered callbacks (strategy engines, position manager for SL monitoring)
- `stop()` — graceful disconnect
- Reconnection: on disconnect, log warning, SDK handles reconnect; after 50 failures, raise `WebSocketError`

**BarBuilder** class:
- Consumes ticks from `tick_queue`
- Maintains dict of `{instrument_key: {timeframe: current_bar_data}}`
- On bar close (e.g., minute boundary), appends to a deque of recent bars (max 500 per timeframe) and notifies strategy threads via `threading.Event`
- Timeframes: 1min, 5min, 15min
- Each bar: `[timestamp, open, high, low, close, volume]` stored as list of ints (prices in paisa)

### 3.6 Data — Portfolio Feed (`src/data/portfolio_feed.py`)

**PortfolioFeed** class:
- Wraps `upstox_client.PortfolioDataStreamer`
- Subscribes to `order_update=True, position_update=True`
- On order update: dispatches to `OrderTracker`
- On position update: dispatches to `PositionManager`

### 3.7 Data — Historical (`src/data/historical.py`)

**HistoricalDataFetcher** class:
- Uses `upstox_client.HistoryV3Api`
- `fetch_candles(instrument_key, unit, interval, from_date, to_date) -> pd.DataFrame`
- Returns DataFrame with columns: timestamp, open, high, low, close, volume, oi
- Prices stored in paisa (multiplied on fetch)
- Used for indicator warmup on bot start (fetch last N candles to seed MACD, RSI, etc.)
- Respects rate limiter

### 3.8 Strategy — Indicators (`src/strategy/indicators.py`)

**IndicatorEngine** class:
- Takes a DataFrame of OHLCV bars, computes indicators and returns enriched DataFrame
- Supported indicators (all via pandas-ta):
  - `ema(period)` — Exponential Moving Average
  - `sma(period)` — Simple Moving Average
  - `rsi(period=14)` — Relative Strength Index
  - `macd(fast=12, slow=26, signal=9)` — returns MACD line, Signal line, Histogram
  - `bbands(period=20, std=2)` — Bollinger Bands (upper, middle, lower)
  - `atr(period=14)` — Average True Range
  - `supertrend(period=7, multiplier=3)` — SuperTrend
  - `vwap()` — Volume Weighted Average Price (intraday, resets daily)
- Indicators computed incrementally: on each new bar, recompute on the tail of the DataFrame (not full recalculation)

### 3.9 Strategy — Builder (`src/strategy/builder.py`)

**StrategyBuilder** class:
- `load_strategy(path: str) -> StrategyConfig` — reads JSON, validates against Pydantic model
- **StrategyConfig** (Pydantic model):
  - `name`, `version`, `underlying`, `active`, `trading_hours` (start/end)
  - `instrument_selection`: type (options/equity), expiry, strike_selection, option_type_from_signal
  - `entry`: dict with `sets` list — each set has `name`, `signal` (CE/PE/BUY/SELL), `conditions` list
  - `exit`: stop_loss_pct, target_pct, trailing_sl config, time_based_exit, max_holding_minutes
  - `position_sizing`: method, quantity, max_risk_per_trade_pct

### 3.10 Strategy — Conditions (`src/strategy/conditions.py`)

**ConditionEvaluator** class:
- `evaluate(condition: dict, bars: dict[str, pd.DataFrame]) -> bool`
- Condition types:
  - Indicator vs value: `{"indicator": "RSI", "comparison": ">", "value": 50, "timeframe": "5min"}`
  - Indicator vs indicator: `{"indicator": "MACD", "comparison": ">", "against": "Signal", "timeframe": "5min"}`
  - With lookback: `{"indicator": "MACD", "comparison": "<", "against": "Signal", "lookback": 1}` — checks previous bar
  - Price-based: `{"indicator": "Spot_price", "comparison": ">", "against": "VWAP"}`
  - Time-based: `{"indicator": "Time", "comparison": ">", "value": "09:20:00"}`
- Comparisons: `>`, `<`, `>=`, `<=`, `==`, `crosses_above`, `crosses_below`
- `crosses_above`: current bar indicator > against AND previous bar indicator <= against

### 3.11 Strategy — Engine (`src/strategy/engine.py`)

**StrategyEngine** class:
- Manages all active strategies
- `load_strategies(directory: str)` — loads all JSON files, creates per-set evaluation threads
- Each **SetEvaluator** (runs in its own thread):
  - Waits for bar close event from BarBuilder
  - Computes indicators on latest bars
  - Evaluates all conditions in the set sequentially (all must be True)
  - If all conditions met, generates a **Signal** (BUY CE / BUY PE / BUY / SELL)
  - Signal includes: strategy name, set name, instrument selection criteria, quantity, SL/target
  - Pushes signal to `signal_queue: Queue` consumed by order manager
- Respects `trading_hours` — only evaluates during configured window
- Thread-safe access to shared bar data (bars are immutable after close)

### 3.12 Execution — Order Manager (`src/execution/order_manager.py`)

**OrderManager** class:
- Consumes signals from `signal_queue`
- Resolves instrument: uses InstrumentManager to find exact instrument key (e.g., ATM CE option for Bank Nifty)
- **Paper mode**: Creates simulated order, fills immediately at current LTP + slippage (configurable, default 0.05%)
- **Live mode**: Calls `OrderApiV3.place_order()` with `slice=True` for F&O
- Order model: id, strategy, instrument_key, transaction_type, order_type, quantity, price, trigger_price, status, timestamps
- All orders logged to DB and `orders.log`

**PaperBroker** (implements BaseBroker interface):
- `place_order(order) -> OrderResponse` — simulates fill at LTP +/- slippage
- `modify_order(order_id, changes)` — updates simulated order
- `cancel_order(order_id)` — marks as cancelled
- `get_positions() -> list[Position]` — returns tracked positions
- `get_order_book() -> list[Order]` — returns all orders for the day

**BaseBroker** (abstract):
- Defines interface: `place_order`, `modify_order`, `cancel_order`, `get_positions`, `get_order_book`, `get_funds`
- `UpstoxBroker` (live) and `PaperBroker` both implement this

### 3.13 Execution — Position Manager (`src/execution/position_manager.py`)

**PositionManager** class:
- Tracks all open positions with: instrument_key, strategy, entry_price (paisa), quantity, side, entry_time, unrealized_pnl
- On each tick (from WebSocket): updates unrealized P&L, checks SL/target/trailing SL
- Stop-loss: if unrealized loss >= stop_loss_pct of entry premium, generates exit signal
- Target: if unrealized profit >= target_pct of entry premium, generates exit signal
- Trailing SL: after activation_pct profit reached, trail at trail_pct below peak
- Time-based exit: at configured time (e.g., 15:10), generate exit for all positions in that strategy
- Forced exit: at INTRADAY_SQUARE_OFF (15:15), exit ALL intraday positions regardless of strategy
- On position close: compute realized P&L (including fees), log trade to DB

### 3.14 Execution — Order Tracker (`src/execution/order_tracker.py`)

**OrderTracker** class:
- Receives order updates from Portfolio WebSocket
- Updates order status in DB: PLACED → OPEN → COMPLETE/REJECTED/CANCELLED
- Handles partial fills: updates filled quantity, average price
- On REJECTED: logs reason, notifies strategy engine
- Retry logic (via tenacity): transient failures retried with exponential backoff (max 3 attempts)

### 3.15 Risk (`src/risk/`)

**RiskManager** class:
- Pre-trade checks (called before every order):
  - `check_daily_loss_limit()` — sum of realized + unrealized P&L vs `max_daily_loss`
  - `check_position_count()` — open positions vs `max_open_positions`
  - `check_position_size(instrument, quantity)` — order value vs `max_position_size_pct * capital`
  - `check_capital_deployment()` — total deployed vs 80% of capital
- Returns `RiskCheckResult(approved: bool, reason: str | None)`
- If any check fails, order is blocked and logged

**CircuitBreaker** class:
- `consecutive_losses: int` — if >= 3, pause trading for N minutes (configurable, default 30)
- `kill_switch()` — cancels all pending orders, exits all positions, halts all strategy threads
- `is_halted() -> bool` — checked by strategy engine before generating signals
- Kill switch can be triggered manually (via signal/API) or automatically by risk manager

### 3.16 Persistence (`src/persistence/`)

**database.py**:
- SQLAlchemy engine from `database_url` setting
- Session factory with context manager
- `init_db()` — creates all tables

**models.py** — ORM models:
- `OrderRecord`: id, strategy, instrument_key, transaction_type, order_type, quantity, price, trigger_price, status, filled_qty, avg_fill_price, placed_at, updated_at, tag, broker_order_id
- `TradeRecord`: id, strategy, instrument_key, side, entry_price, exit_price, quantity, realized_pnl, fees, entry_time, exit_time, holding_duration
- `PositionRecord`: id, strategy, instrument_key, side, entry_price, quantity, status (open/closed), opened_at, closed_at
- `DailyPnL`: date, realized_pnl, unrealized_pnl, total_pnl, trades_count, win_count
- `StrategyState`: strategy_name, set_name, last_evaluated_bar, state_json, updated_at

**trade_log.py**:
- `log_order(order)`, `log_trade(trade)`, `log_daily_summary()`
- Writes to both DB and structured log file

### 3.17 Main Orchestrator (`src/main.py`)

Startup sequence:
1. Load settings from `.env`
2. Initialize loguru logger
3. Authenticate (TokenManager.get_valid_token())
4. Download/load instruments
5. Initialize database, create tables, load saved state
6. Start MarketDataFeed WebSocket (subscribe to strategy underlyings)
7. Start PortfolioFeed WebSocket
8. Start BarBuilder thread
9. Load strategies from `config/strategies/`, start evaluation threads
10. Initialize RiskManager with account funds
11. Start OrderManager (consuming signal queue)
12. Configure APScheduler (daily login, instrument refresh, forced exit, daily summary)
13. Register SIGINT/SIGTERM handlers for graceful shutdown

Shutdown sequence:
1. Stop strategy evaluation threads
2. If configured, exit all open positions
3. Save strategy state to DB
4. Stop WebSocket connections
5. Stop scheduler
6. Log final daily P&L summary
7. Close database connections

---

## 4. Fee Structure (Paper Trading Accuracy)

For realistic P&L in paper mode, apply these fees on each trade:

| Fee | Rate |
|-----|------|
| Brokerage (Upstox) | Rs 20 per executed order (F&O), 0 for equity delivery |
| STT | 0.0125% on sell side (options), 0.02% buy+sell (futures) |
| Exchange charges (NSE) | 0.053% (options), 0.002% (futures) |
| GST | 18% on (brokerage + exchange charges) |
| SEBI charges | Rs 10 per crore |
| Stamp duty | 0.003% on buy side |

All fees computed in paisa and deducted from realized P&L.

---

## 5. Example Strategy JSON (MACD Crossover on Bank Nifty)

```json
{
  "name": "MACD_Crossover_BankNifty",
  "version": "1.0",
  "underlying": "NSE_INDEX|Nifty Bank",
  "active": true,
  "trading_hours": {"start": "09:20:00", "end": "15:10:00"},
  "instrument_selection": {
    "type": "options",
    "expiry": "weekly_current",
    "strike_selection": "ATM",
    "option_type_from_signal": true
  },
  "entry": {
    "sets": [
      {
        "name": "MACD_Bullish_Cross",
        "signal": "CE",
        "conditions": [
          {"indicator": "MACD", "comparison": "crosses_above", "against": "Signal", "timeframe": "5min"},
          {"indicator": "RSI", "comparison": ">", "value": 50, "timeframe": "5min"},
          {"indicator": "Spot_price", "comparison": ">", "against": "VWAP", "timeframe": "1min"}
        ]
      },
      {
        "name": "MACD_Bearish_Cross",
        "signal": "PE",
        "conditions": [
          {"indicator": "MACD", "comparison": "crosses_below", "against": "Signal", "timeframe": "5min"},
          {"indicator": "RSI", "comparison": "<", "value": 50, "timeframe": "5min"},
          {"indicator": "Spot_price", "comparison": "<", "against": "VWAP", "timeframe": "1min"}
        ]
      }
    ]
  },
  "exit": {
    "stop_loss_pct": 30,
    "target_pct": 60,
    "trailing_sl": {"enable": true, "activation_pct": 30, "trail_pct": 15},
    "time_based_exit": "15:10:00",
    "max_holding_minutes": 120
  },
  "position_sizing": {
    "method": "fixed_quantity",
    "quantity": 15,
    "max_risk_per_trade_pct": 2
  }
}
```

---

## 6. Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Paisa (int) for all monetary values | Avoids floating-point precision errors; convert to rupees only for display |
| BaseBroker interface with Paper/Live implementations | Same code paths for paper and live; swap by config |
| Thread per strategy set (not asyncio) | Strategy evaluation is CPU-bound (indicator calc); threads with GIL release during I/O are simpler than async |
| Queue-based inter-thread communication | Thread-safe, decoupled; signal_queue, tick_queue, order_queue |
| SQLite for Phase 1 | Zero-config, single-file; SQLAlchemy makes PostgreSQL upgrade trivial |
| Loguru over stdlib logging | Simpler API, structured output, rotation built-in |
| pandas-ta for indicators | Comprehensive library, pandas-native, well-maintained |
| APScheduler for scheduling | Mature, supports cron-like triggers, IST timezone aware |
| Token cached to JSON file | Simple, survives restarts; no external dependency |
| Instrument files cached locally | Avoids repeated downloads; refreshed daily at 06:15 |
