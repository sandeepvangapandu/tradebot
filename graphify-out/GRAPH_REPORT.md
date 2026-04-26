# Graph Report - .  (2026-04-13)

## Corpus Check
- 164 files · ~204,889 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 4599 nodes · 16065 edges · 102 communities detected
- Extraction: 33% EXTRACTED · 67% INFERRED · 0% AMBIGUOUS · INFERRED: 10770 edges (avg confidence: 0.5)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Market Data & Bar Builder|Market Data & Bar Builder]]
- [[_COMMUNITY_Backtester Engine & Dashboard|Backtester Engine & Dashboard]]
- [[_COMMUNITY_Strategy Condition Framework|Strategy Condition Framework]]
- [[_COMMUNITY_Broker Interface & Orders|Broker Interface & Orders]]
- [[_COMMUNITY_Dashboard & Monitoring|Dashboard & Monitoring]]
- [[_COMMUNITY_AI Agent Pipeline|AI Agent Pipeline]]
- [[_COMMUNITY_Upstox Auth & Login|Upstox Auth & Login]]
- [[_COMMUNITY_Pattern Recognition|Pattern Recognition]]
- [[_COMMUNITY_Data Loading & CSV|Data Loading & CSV]]
- [[_COMMUNITY_Research Analysis Models|Research Analysis Models]]
- [[_COMMUNITY_Volume Analyzer & Tests|Volume Analyzer & Tests]]
- [[_COMMUNITY_Health Monitor|Health Monitor]]
- [[_COMMUNITY_Paper Broker & Fees|Paper Broker & Fees]]
- [[_COMMUNITY_Condition Evaluator|Condition Evaluator]]
- [[_COMMUNITY_Technical Indicators Tests|Technical Indicators Tests]]
- [[_COMMUNITY_Straddle Strategy|Straddle Strategy]]
- [[_COMMUNITY_Setup & Entry Point|Setup & Entry Point]]
- [[_COMMUNITY_Broker Factory|Broker Factory]]
- [[_COMMUNITY_Architecture Design Docs|Architecture Design Docs]]
- [[_COMMUNITY_Tick Processing|Tick Processing]]
- [[_COMMUNITY_RSI Divergence Detection|RSI Divergence Detection]]
- [[_COMMUNITY_Utility Functions|Utility Functions]]
- [[_COMMUNITY_Instrument Manager|Instrument Manager]]
- [[_COMMUNITY_Kill Switch & Circuit Breaker|Kill Switch & Circuit Breaker]]
- [[_COMMUNITY_Agent Pipeline Docs|Agent Pipeline Docs]]
- [[_COMMUNITY_Module 25|Module 25]]
- [[_COMMUNITY_Module 26|Module 26]]
- [[_COMMUNITY_Module 27|Module 27]]
- [[_COMMUNITY_Module 28|Module 28]]
- [[_COMMUNITY_Module 29|Module 29]]
- [[_COMMUNITY_Module 30|Module 30]]
- [[_COMMUNITY_Module 31|Module 31]]
- [[_COMMUNITY_Module 32|Module 32]]
- [[_COMMUNITY_Module 33|Module 33]]
- [[_COMMUNITY_Module 34|Module 34]]
- [[_COMMUNITY_Module 35|Module 35]]
- [[_COMMUNITY_Module 36|Module 36]]
- [[_COMMUNITY_Module 37|Module 37]]
- [[_COMMUNITY_Module 38|Module 38]]
- [[_COMMUNITY_Module 39|Module 39]]
- [[_COMMUNITY_Module 40|Module 40]]
- [[_COMMUNITY_Module 41|Module 41]]
- [[_COMMUNITY_Module 42|Module 42]]
- [[_COMMUNITY_Module 43|Module 43]]
- [[_COMMUNITY_Module 44|Module 44]]
- [[_COMMUNITY_Module 45|Module 45]]
- [[_COMMUNITY_Module 46|Module 46]]
- [[_COMMUNITY_Module 47|Module 47]]
- [[_COMMUNITY_Module 48|Module 48]]
- [[_COMMUNITY_Module 49|Module 49]]
- [[_COMMUNITY_Module 50|Module 50]]
- [[_COMMUNITY_Module 51|Module 51]]
- [[_COMMUNITY_Module 52|Module 52]]
- [[_COMMUNITY_Module 53|Module 53]]
- [[_COMMUNITY_Module 54|Module 54]]
- [[_COMMUNITY_Module 55|Module 55]]
- [[_COMMUNITY_Module 56|Module 56]]
- [[_COMMUNITY_Module 57|Module 57]]
- [[_COMMUNITY_Module 58|Module 58]]
- [[_COMMUNITY_Module 59|Module 59]]
- [[_COMMUNITY_Module 60|Module 60]]
- [[_COMMUNITY_Module 61|Module 61]]
- [[_COMMUNITY_Module 62|Module 62]]
- [[_COMMUNITY_Module 63|Module 63]]
- [[_COMMUNITY_Module 64|Module 64]]
- [[_COMMUNITY_Module 65|Module 65]]
- [[_COMMUNITY_Module 66|Module 66]]
- [[_COMMUNITY_Module 67|Module 67]]
- [[_COMMUNITY_Module 68|Module 68]]
- [[_COMMUNITY_Module 69|Module 69]]
- [[_COMMUNITY_Module 70|Module 70]]
- [[_COMMUNITY_Module 71|Module 71]]
- [[_COMMUNITY_Module 72|Module 72]]
- [[_COMMUNITY_Module 73|Module 73]]
- [[_COMMUNITY_Module 74|Module 74]]
- [[_COMMUNITY_Module 75|Module 75]]
- [[_COMMUNITY_Module 76|Module 76]]
- [[_COMMUNITY_Module 77|Module 77]]
- [[_COMMUNITY_Module 78|Module 78]]
- [[_COMMUNITY_Module 79|Module 79]]
- [[_COMMUNITY_Module 80|Module 80]]
- [[_COMMUNITY_Module 81|Module 81]]
- [[_COMMUNITY_Module 82|Module 82]]
- [[_COMMUNITY_Module 83|Module 83]]
- [[_COMMUNITY_Module 84|Module 84]]
- [[_COMMUNITY_Module 85|Module 85]]
- [[_COMMUNITY_Module 86|Module 86]]
- [[_COMMUNITY_Module 87|Module 87]]
- [[_COMMUNITY_Module 88|Module 88]]
- [[_COMMUNITY_Module 89|Module 89]]
- [[_COMMUNITY_Module 90|Module 90]]
- [[_COMMUNITY_Module 91|Module 91]]
- [[_COMMUNITY_Module 92|Module 92]]
- [[_COMMUNITY_Module 93|Module 93]]
- [[_COMMUNITY_Module 94|Module 94]]
- [[_COMMUNITY_Module 95|Module 95]]
- [[_COMMUNITY_Module 96|Module 96]]
- [[_COMMUNITY_Module 97|Module 97]]
- [[_COMMUNITY_Module 98|Module 98]]
- [[_COMMUNITY_Module 99|Module 99]]
- [[_COMMUNITY_Module 100|Module 100]]
- [[_COMMUNITY_Module 101|Module 101]]

## God Nodes (most connected - your core abstractions)
1. `AnalysisComponent` - 400 edges
2. `IndicatorEngine` - 310 edges
3. `MarketRegime` - 295 edges
4. `VolatilityRegime` - 295 edges
5. `RiskManager` - 254 edges
6. `OrderSide` - 232 edges
7. `ProductType` - 232 edges
8. `SignalStrength` - 200 edges
9. `TradeVerdict` - 200 edges
10. `OptimalEntryType` - 200 edges

## Surprising Connections (you probably didn't know these)
- `Integration tests for BacktestHarness using real CSV data.  If nifty50_1min_30da` --uses--> `BacktestHarness`  [INFERRED]
  tests/test_backtest_harness.py → src/backtest/harness.py
- `Full backtest run with real data completes without exceptions.` --uses--> `BacktestHarness`  [INFERRED]
  tests/test_backtest_harness.py → src/backtest/harness.py
- `All closed trades must have positive entry_price, quantity, and a side.` --uses--> `BacktestHarness`  [INFERRED]
  tests/test_backtest_harness.py → src/backtest/harness.py
- `Equity curve must never go negative (capital floor of 0).` --uses--> `BacktestHarness`  [INFERRED]
  tests/test_backtest_harness.py → src/backtest/harness.py
- `HarnessResults.metrics must contain all required keys.` --uses--> `BacktestHarness`  [INFERRED]
  tests/test_backtest_harness.py → src/backtest/harness.py

## Hyperedges (group relationships)
- **Multi-Layer Risk Protection (RiskManager + CircuitBreaker + Safety Features + Kill Switch)** — vertical_slice_risk_manager, vertical_slice_circuit_breaker, configuration_safety_features, pro_trader_circuit_breaker, claude_md_risk_management [INFERRED 0.85]
- **AI Agent Pipeline (Regime -> Signal Validation -> Risk Check via LangGraph)** — ai_pipeline_langgraph_state_machine, ai_pipeline_regime_agent, ai_pipeline_signal_validator, ai_pipeline_news_agent, ai_pipeline_llm_client, ai_pipeline_models [EXTRACTED 0.95]
- **Live-Parity Backtesting System (Harness reuses all live components)** — backtest_harness_design, backtest_harness_bar_provider, backtest_harness_evaluate_bar_sync, vertical_slice_order_manager, vertical_slice_risk_manager, vertical_slice_position_manager [EXTRACTED 0.95]

## Communities

### Community 0 - "Market Data & Bar Builder"
Cohesion: 0.01
Nodes (454): BarBuilder, Stop the bar builder thread., Get all instrument keys with bar data., Clear bar history.          Args:             instrument_key: Specific instrumen, Builds OHLCV bars from ticks and notifies on bar close.      Runs in a separate, Start the bar builder thread., analyze(), analyze_index_correlation() (+446 more)

### Community 1 - "Backtester Engine & Dashboard"
Cohesion: 0.01
Nodes (411): Backtester dashboard page — full-fidelity backtesting using live components.  Up, Render the backtester page., render(), BacktestBarProvider, Historical bar provider implementing BarBuilder + HistoricalDataFetcher interfac, Return daily bars from historical data up to current index.          Args:, Drop-in replacement for BarBuilder and HistoricalDataFetcher.      Holds full pr, Advance the time window to bar ``index`` (0-based, inclusive).          Must be (+403 more)

### Community 2 - "Strategy Condition Framework"
Cohesion: 0.01
Nodes (345): BaseModel, ComparisonOperator, Condition, create_sample_strategy(), EntrySet, ExitRules, IndicatorRef, IndicatorType (+337 more)

### Community 3 - "Broker Interface & Orders"
Cohesion: 0.02
Nodes (356): BaseBroker, BrokerError, Config, Funds, Order, OrderResponse, OrderSide, OrderStatus (+348 more)

### Community 4 - "Dashboard & Monitoring"
Cohesion: 0.01
Nodes (237): Base, Main Streamlit dashboard entry point.  Run with: streamlit run src/dashboard/das, DashboardDataService, _paisa_to_rupees(), PositionView, Dashboard data service - queries database and formats for display., Get today's completed trades.          Returns:             List of TradeView ob, Get filtered trade history.          Args:             start_date: Filter trades (+229 more)

### Community 5 - "AI Agent Pipeline"
Cohesion: 0.01
Nodes (176): ABC, BaseAgent, _build_prompt(), _build_system_prompt(), _fallback(), _parse_response(), Abstract base class for all LLM-powered agents.  Every agent follows the same pa, Store the most recent decision as an AgentDecision for observability.          A (+168 more)

### Community 6 - "Upstox Auth & Login"
Cohesion: 0.01
Nodes (139): auto_login(), Automated Upstox login using TOTP-based authentication., Authenticate with Upstox using TOTP and return an access token.      Reads crede, Exception, AuthenticationError, InstrumentError, InsufficientMarginError, OrderError (+131 more)

### Community 7 - "Pattern Recognition"
Cohesion: 0.04
Nodes (104): PatternData, Pattern recognition data., detect_pattern(), PatternName, PatternRecognizer, PatternResult, PatternType, Candlestick pattern recognition module.  Detects various candlestick patterns on (+96 more)

### Community 8 - "Data Loading & CSV"
Cohesion: 0.03
Nodes (67): BacktestDataLoader, Load and prepare historical OHLCV data for backtesting., Loads historical OHLCV data from CSV files or the database.      All price colum, Load OHLCV data from a CSV file.          Args:             file_path: Path to t, Resample OHLCV data to a larger timeframe.          Args:             df: Source, BacktestEngine, BacktestResults, calculate_trade_fees() (+59 more)

### Community 9 - "Research Analysis Models"
Cohesion: 0.05
Nodes (85): MomentumData, Support and resistance analysis data., Detailed trend alignment data., Momentum analysis data., SupportResistanceData, TrendAlignmentData, analyze_candle_context(), analyze_momentum() (+77 more)

### Community 10 - "Volume Analyzer & Tests"
Cohesion: 0.04
Nodes (74): falling_volume_data(), high_volume_data(), illiquid_option_data(), insufficient_data(), liquid_option_data(), Tests for the volume analysis module.  All prices are in PAISA (integer) - 1 Rup, Return a DataFrame with insufficient data., Return liquid option data. (+66 more)

### Community 11 - "Health Monitor"
Cohesion: 0.03
Nodes (44): ComponentNotHealthyError, ComponentStatus, get_health_monitor(), HealthMonitor, MemoryMonitor, Self-healing health monitor for the trading bot.  This module provides comprehen, Create or return the singleton instance.          Returns:             The singl, Initialize the health monitor (only runs once due to singleton). (+36 more)

### Community 12 - "Paper Broker & Fees"
Cohesion: 0.03
Nodes (64): broker(), broker_no_slippage(), broker_with_slippage(), calculate_fees(), FeeCalculator, Order, OrderStatus, OrderType (+56 more)

### Community 13 - "Condition Evaluator"
Cohesion: 0.04
Nodes (45): ConditionEvaluator, MockStrategy, MockStrategyBuilder, Tests for the strategy engine components.  Tests StrategyBuilder, ConditionEvalu, Check if col1 crossed above col2 on the last bar., Check if col1 crossed below col2 on the last bar., Evaluates entry sets and generates signals., Evaluate an entry set and return signal if all conditions met.          Args: (+37 more)

### Community 14 - "Technical Indicators Tests"
Cohesion: 0.04
Nodes (29): Tests for the technical indicators engine.  Tests EMA, MACD, RSI, and VWAP calcu, Test detecting MACD crossovers., Test MACD behavior in strongly trending market., Test Exponential Moving Average calculations., Test Relative Strength Index calculations., Test RSI values are always between 0 and 100., Test RSI has NaN values during warmup period., Test RSI correctly identifies overbought/oversold conditions. (+21 more)

### Community 15 - "Straddle Strategy"
Cohesion: 0.06
Nodes (30): calculate_atm_strike(), calculate_straddle_profit_loss(), check_entry_conditions(), estimate_straddle_payoff(), ExitReason, ExpiryStraddleStrategy, is_expiry_day(), is_monthly_expiry() (+22 more)

### Community 16 - "Setup & Entry Point"
Cohesion: 0.13
Nodes (19): main(), Check required environment variables., Validate trading mode is set to paper., Validate capital and risk settings., Check required directories exist., Check strategy configuration files., Validates trading bot configuration before startup., Check database can be initialized. (+11 more)

### Community 17 - "Broker Factory"
Cohesion: 0.12
Nodes (17): BrokerFactory, create(), create_broker(), create_live(), _create_live_broker(), create_paper(), _create_paper_broker(), Tests for broker factory. (+9 more)

### Community 18 - "Architecture Design Docs"
Cohesion: 0.09
Nodes (25): AI-Enhanced Trading Pipeline Implementation Plan, Backtest Harness Design Spec (Full Live-Parity Backtesting), evaluate_bar_sync() Method (Synchronous Strategy Evaluation for Backtest), Guiding Principle: Plumbing vs Business Logic Separation, Backtest Harness Implementation Plan, BacktestEngine Divergence Problem (reimplements execution logic), Backtesting Engine Requirements, Broker Adapter Interface Pattern (BaseBroker) (+17 more)

### Community 19 - "Tick Processing"
Cohesion: 0.11
Nodes (11): CurrentBar, OHLCVBar, OHLCV bar builder from WebSocket ticks.  Consumes ticks from a queue and builds, Main loop consuming ticks from queue., Process a single tick update.          Args:             tick: Tick data with ke, Update a bar for given instrument and timeframe.          Returns:             T, Get the bar start time for a given timestamp and timeframe., Parse timestamp to datetime in IST. (+3 more)

### Community 20 - "RSI Divergence Detection"
Cohesion: 0.17
Nodes (14): calculate_proximity_to_level(), detect_hidden_divergence(), detect_rsi_divergence(), DivergenceResult, find_swing_points(), get_support_resistance_levels(), Divergence detection for RSI and price.  Provides functions to detect bullish an, Detect RSI divergence patterns.      Bullish Divergence (Potential Reversal Up): (+6 more)

### Community 21 - "Utility Functions"
Cohesion: 0.15
Nodes (13): format_instrument_key(), is_market_open(), now_ist(), paisa_to_rupees(), parse_time(), General-purpose utility functions for the trading bot., Return the current datetime in IST (Asia/Kolkata).      Returns:         Current, Check whether the Indian equity market is currently open.      Checks that today (+5 more)

### Community 22 - "Instrument Manager"
Cohesion: 0.17
Nodes (6): Load cached JSON instrument files into a single DataFrame.          If instrumen, Filter loaded instruments by segment, symbol, and/or instrument type.          A, Retrieve the option chain for a given underlying and expiry.          Args:, Find the nearest weekly expiry for an underlying symbol.          Args:, Look up the Upstox instrument key by segment and trading symbol.          Args:, Resolve a trading symbol or instrument key to a full instrument key.          If

### Community 23 - "Kill Switch & Circuit Breaker"
Cohesion: 0.25
Nodes (5): OrderManagerProtocol, Circuit breaker and kill-switch module.  Halts trading automatically after a con, Emergency halt — stops all trading indefinitely.          When *order_manager* i, Minimal interface expected from an order manager by the kill switch., Protocol

### Community 24 - "Agent Pipeline Docs"
Cohesion: 0.22
Nodes (9): LangGraph StateGraph Pipeline Orchestration, Agent Pipeline Data Models (PipelineState, RegimeClassification, etc.), News Sentiment Agent (RSS + LLM Scoring), Regime Detection Agent (LLM-Powered Market Regime Classification), LLM Signal Validator Agent, Research Module Behaviour in Backtest (Graceful Degradation), Research Module AI Signal Validation Config, feedparser for News RSS (+1 more)

### Community 25 - "Module 25"
Cohesion: 0.25
Nodes (7): get_degraded_analyzers(), graceful_degrade(), is_degraded_result(), Graceful degradation utilities for the trading bot.  This module provides decora, Check if a result is the default/degraded value.      Args:         result: The, Extract list of analyzer names that returned degraded results.      Args:, Decorator that catches exceptions and returns a default value.      This decorat

### Community 26 - "Module 26"
Cohesion: 0.32
Nodes (7): get_next_trading_day(), is_nse_holiday(), is_trading_day(), NSE Trading Holiday Calendar Utility., Check if a given date is an NSE trading holiday., Check if a given date is a valid trading day (not weekend or holiday)., Get the next trading day after the given date.

### Community 27 - "Module 27"
Cohesion: 0.33
Nodes (5): get_session(), init_db(), Database engine and session management (SQLAlchemy 2.0 style).  Provides lazy-in, Create the SQLAlchemy engine, build all tables, and wire up the session factory., Yield a transactional database session.      Commits automatically when the bloc

### Community 28 - "Module 28"
Cohesion: 0.4
Nodes (5): Deterministic Fallback Pattern (LLM unavailable -> rule-based), LLM Client with Circuit Breaker and Rate Limiter, Pro Trader Circuit Breaker (8% catastrophic protection), Groq SDK for Llama-3.3-70b Inference, CircuitBreaker (Consecutive Losses, Kill Switch)

### Community 29 - "Module 29"
Cohesion: 0.5
Nodes (3): Logging configuration using loguru., Configure and return the application logger.      Args:         log_level: Minim, setup_logger()

### Community 30 - "Module 30"
Cohesion: 0.5
Nodes (1): CLI entry point for the BacktestHarness.  Usage:     python -m src.backtest.runn

### Community 31 - "Module 31"
Cohesion: 0.67
Nodes (3): 4-Phase Development Plan (Foundation -> Live), Remaining Phases Implementation Plan (Phases 2-4), Streamlit Dashboard Dependency

### Community 32 - "Module 32"
Cohesion: 0.67
Nodes (3): Per-Bar Loop (Advance -> Day Reset -> Exits -> Session Gate -> Signals -> Equity), APScheduler for Daily Tasks, Live Data Flow (WebSocket -> BarBuilder -> Strategy -> Risk -> Order -> DB)

### Community 33 - "Module 33"
Cohesion: 0.67
Nodes (3): Trade Memory System (Outcome Analysis, Mistake Classification, Lessons), Learning System Behaviour in Backtest (Source Tagging, No Cross-Contamination), Rationale: Why Backtest Lessons Are Not Promoted to Live

### Community 34 - "Module 34"
Cohesion: 1.0
Nodes (0): 

### Community 35 - "Module 35"
Cohesion: 1.0
Nodes (1): Static constants for Indian market trading.

### Community 36 - "Module 36"
Cohesion: 1.0
Nodes (2): Pluggable Strategy Architecture (BaseStrategy Interface), pandas-ta Technical Indicators Library

### Community 37 - "Module 37"
Cohesion: 1.0
Nodes (2): Indian Market Specific Rules (IST, T+1, Circuit Limits, Holidays), Pro Trader Trading Sessions (Opening Range + Afternoon Trend)

### Community 38 - "Module 38"
Cohesion: 1.0
Nodes (2): Project Overview: Autonomous Trading Bot for Indian Markets, Trading Bot Configuration Guide

### Community 39 - "Module 39"
Cohesion: 1.0
Nodes (2): Upstox OAuth2 Configuration, Vertical Slice Phase 1 Design Spec

### Community 40 - "Module 40"
Cohesion: 1.0
Nodes (2): BacktestBarProvider (No-Lookahead Historical Data Adapter), BarBuilder (Tick -> OHLCV Candle Aggregation)

### Community 41 - "Module 41"
Cohesion: 1.0
Nodes (2): SQLAlchemy ORM Dependency, SQLite Persistence (OrderRecord, TradeRecord, PositionRecord, DailyPnL)

### Community 42 - "Module 42"
Cohesion: 1.0
Nodes (2): JSON-Driven Strategy Configuration (MACD Example), MACD Crossover Strategy on Bank Nifty (JSON-driven)

### Community 43 - "Module 43"
Cohesion: 1.0
Nodes (0): 

### Community 44 - "Module 44"
Cohesion: 1.0
Nodes (0): 

### Community 45 - "Module 45"
Cohesion: 1.0
Nodes (0): 

### Community 46 - "Module 46"
Cohesion: 1.0
Nodes (0): 

### Community 47 - "Module 47"
Cohesion: 1.0
Nodes (0): 

### Community 48 - "Module 48"
Cohesion: 1.0
Nodes (0): 

### Community 49 - "Module 49"
Cohesion: 1.0
Nodes (0): 

### Community 50 - "Module 50"
Cohesion: 1.0
Nodes (0): 

### Community 51 - "Module 51"
Cohesion: 1.0
Nodes (1): Create sample market data for testing.

### Community 52 - "Module 52"
Cohesion: 1.0
Nodes (1): Create a mock condition evaluator.

### Community 53 - "Module 53"
Cohesion: 1.0
Nodes (0): 

### Community 54 - "Module 54"
Cohesion: 1.0
Nodes (1): Calculate total fees for a trade.          Args:             transaction_type: B

### Community 55 - "Module 55"
Cohesion: 1.0
Nodes (1): Create a paper broker instance.

### Community 56 - "Module 56"
Cohesion: 1.0
Nodes (1): Create broker with no slippage.

### Community 57 - "Module 57"
Cohesion: 1.0
Nodes (1): Create broker with 0.5% slippage.

### Community 58 - "Module 58"
Cohesion: 1.0
Nodes (1): Create a paper broker instance.

### Community 59 - "Module 59"
Cohesion: 1.0
Nodes (1): Create a paper broker instance.

### Community 60 - "Module 60"
Cohesion: 1.0
Nodes (0): 

### Community 61 - "Module 61"
Cohesion: 1.0
Nodes (0): 

### Community 62 - "Module 62"
Cohesion: 1.0
Nodes (1): Signal is approved (includes modified signals).

### Community 63 - "Module 63"
Cohesion: 1.0
Nodes (1): Validate price is provided for LIMIT and SL orders.

### Community 64 - "Module 64"
Cohesion: 1.0
Nodes (1): Validate trigger_price is provided for SL orders.

### Community 65 - "Module 65"
Cohesion: 1.0
Nodes (1): Validate signal type.

### Community 66 - "Module 66"
Cohesion: 1.0
Nodes (1): Validate stop_loss < entry_price < target for BUY signals.

### Community 67 - "Module 67"
Cohesion: 1.0
Nodes (1): Validate per_instrument_limit doesn't exceed max_capital_deployment.

### Community 68 - "Module 68"
Cohesion: 1.0
Nodes (1): Validate a trading symbol.          Args:             symbol: The symbol to vali

### Community 69 - "Module 69"
Cohesion: 1.0
Nodes (1): Validate a price value.          Args:             price: The price to validate

### Community 70 - "Module 70"
Cohesion: 1.0
Nodes (1): Validate a quantity value.          Args:             qty: The quantity to valid

### Community 71 - "Module 71"
Cohesion: 1.0
Nodes (1): Validate order parameters dictionary.          Required fields: symbol, quantity

### Community 72 - "Module 72"
Cohesion: 1.0
Nodes (1): Validate a trading signal dictionary.          Required fields from Signal datac

### Community 73 - "Module 73"
Cohesion: 1.0
Nodes (1): Validate trading configuration dictionary.          Args:             config: Di

### Community 74 - "Module 74"
Cohesion: 1.0
Nodes (1): Place a new order with the broker.          Args:             order: The order t

### Community 75 - "Module 75"
Cohesion: 1.0
Nodes (1): Modify an existing open order.          Args:             order_id: The unique o

### Community 76 - "Module 76"
Cohesion: 1.0
Nodes (1): Cancel an open order.          Args:             order_id: The unique order ID t

### Community 77 - "Module 77"
Cohesion: 1.0
Nodes (1): Get all current positions.          Returns:             List of Position object

### Community 78 - "Module 78"
Cohesion: 1.0
Nodes (1): Get all orders for the day.          Returns:             List of OrderResponse

### Community 79 - "Module 79"
Cohesion: 1.0
Nodes (1): Get available funds for trading.          Returns:             Available capital

### Community 80 - "Module 80"
Cohesion: 1.0
Nodes (1): Get the status of a specific order.          Args:             order_id: The uni

### Community 81 - "Module 81"
Cohesion: 1.0
Nodes (1): Check if broker connection is active.          Returns:             True if conn

### Community 82 - "Module 82"
Cohesion: 1.0
Nodes (1): Return True if this trade was profitable.

### Community 83 - "Module 83"
Cohesion: 1.0
Nodes (1): Return True if this trade was a loss.

### Community 84 - "Module 84"
Cohesion: 1.0
Nodes (1): Return absolute value of P&L.

### Community 85 - "Module 85"
Cohesion: 1.0
Nodes (1): Ensure exactly one of ``against`` or ``value`` is set.          Exception: the `

### Community 86 - "Module 86"
Cohesion: 1.0
Nodes (1): Normalize field names from config files.

### Community 87 - "Module 87"
Cohesion: 1.0
Nodes (1): Validate that required params are present for each method.

### Community 88 - "Module 88"
Cohesion: 1.0
Nodes (1): Parse start_time string to time object.

### Community 89 - "Module 89"
Cohesion: 1.0
Nodes (1): Parse end_time string to time object.

### Community 90 - "Module 90"
Cohesion: 1.0
Nodes (1): Alias for enabled (backward compat).

### Community 91 - "Module 91"
Cohesion: 1.0
Nodes (1): Primary timeframe (backward compat).

### Community 92 - "Module 92"
Cohesion: 1.0
Nodes (1): Max concurrent positions from risk_management.

### Community 93 - "Module 93"
Cohesion: 1.0
Nodes (1): Max trades per day from risk_management.

### Community 94 - "Module 94"
Cohesion: 1.0
Nodes (1): Load a strategy configuration from a file.          Args:             path: Path

### Community 95 - "Module 95"
Cohesion: 1.0
Nodes (1): Load all strategy configurations from a directory.          Args:             di

### Community 96 - "Module 96"
Cohesion: 1.0
Nodes (1): Save a strategy configuration to a file.          Args:             config: Stra

### Community 97 - "Module 97"
Cohesion: 1.0
Nodes (1): Create a sample EMA crossover strategy for reference.

### Community 98 - "Module 98"
Cohesion: 1.0
Nodes (1): BacktestDataLoader (CSV + Resampling)

### Community 99 - "Module 99"
Cohesion: 1.0
Nodes (1): BacktestReportGenerator (Text + DataFrame Reports)

### Community 100 - "Module 100"
Cohesion: 1.0
Nodes (1): InstrumentManager (Download, Search, Option Chain, ATM Strike)

### Community 101 - "Module 101"
Cohesion: 1.0
Nodes (1): Indian Market Fee Structure (Brokerage, STT, GST, Stamp Duty)

## Knowledge Gaps
- **622 isolated node(s):** `Validates trading bot configuration before startup.`, `Print a formatted header.`, `Print a success message.`, `Print an error message.`, `Print a warning message.` (+617 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `Module 34`** (2 nodes): `get_trades_summary()`, `analyze_trades.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Module 35`** (2 nodes): `constants.py`, `Static constants for Indian market trading.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Module 36`** (2 nodes): `Pluggable Strategy Architecture (BaseStrategy Interface)`, `pandas-ta Technical Indicators Library`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Module 37`** (2 nodes): `Indian Market Specific Rules (IST, T+1, Circuit Limits, Holidays)`, `Pro Trader Trading Sessions (Opening Range + Afternoon Trend)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Module 38`** (2 nodes): `Project Overview: Autonomous Trading Bot for Indian Markets`, `Trading Bot Configuration Guide`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Module 39`** (2 nodes): `Upstox OAuth2 Configuration`, `Vertical Slice Phase 1 Design Spec`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Module 40`** (2 nodes): `BacktestBarProvider (No-Lookahead Historical Data Adapter)`, `BarBuilder (Tick -> OHLCV Candle Aggregation)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Module 41`** (2 nodes): `SQLAlchemy ORM Dependency`, `SQLite Persistence (OrderRecord, TradeRecord, PositionRecord, DailyPnL)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Module 42`** (2 nodes): `JSON-Driven Strategy Configuration (MACD Example)`, `MACD Crossover Strategy on Bank Nifty (JSON-driven)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Module 43`** (1 nodes): `parse_trades.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Module 44`** (1 nodes): `analyze_macd.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Module 45`** (1 nodes): `strategy_analysis.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Module 46`** (1 nodes): `analyze_orders.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Module 47`** (1 nodes): `deep_analysis.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Module 48`** (1 nodes): `debug_indicators.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Module 49`** (1 nodes): `data.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Module 50`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Module 51`** (1 nodes): `Create sample market data for testing.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Module 52`** (1 nodes): `Create a mock condition evaluator.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Module 53`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Module 54`** (1 nodes): `Calculate total fees for a trade.          Args:             transaction_type: B`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Module 55`** (1 nodes): `Create a paper broker instance.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Module 56`** (1 nodes): `Create broker with no slippage.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Module 57`** (1 nodes): `Create broker with 0.5% slippage.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Module 58`** (1 nodes): `Create a paper broker instance.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Module 59`** (1 nodes): `Create a paper broker instance.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Module 60`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Module 61`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Module 62`** (1 nodes): `Signal is approved (includes modified signals).`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Module 63`** (1 nodes): `Validate price is provided for LIMIT and SL orders.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Module 64`** (1 nodes): `Validate trigger_price is provided for SL orders.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Module 65`** (1 nodes): `Validate signal type.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Module 66`** (1 nodes): `Validate stop_loss < entry_price < target for BUY signals.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Module 67`** (1 nodes): `Validate per_instrument_limit doesn't exceed max_capital_deployment.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Module 68`** (1 nodes): `Validate a trading symbol.          Args:             symbol: The symbol to vali`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Module 69`** (1 nodes): `Validate a price value.          Args:             price: The price to validate`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Module 70`** (1 nodes): `Validate a quantity value.          Args:             qty: The quantity to valid`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Module 71`** (1 nodes): `Validate order parameters dictionary.          Required fields: symbol, quantity`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Module 72`** (1 nodes): `Validate a trading signal dictionary.          Required fields from Signal datac`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Module 73`** (1 nodes): `Validate trading configuration dictionary.          Args:             config: Di`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Module 74`** (1 nodes): `Place a new order with the broker.          Args:             order: The order t`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Module 75`** (1 nodes): `Modify an existing open order.          Args:             order_id: The unique o`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Module 76`** (1 nodes): `Cancel an open order.          Args:             order_id: The unique order ID t`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Module 77`** (1 nodes): `Get all current positions.          Returns:             List of Position object`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Module 78`** (1 nodes): `Get all orders for the day.          Returns:             List of OrderResponse`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Module 79`** (1 nodes): `Get available funds for trading.          Returns:             Available capital`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Module 80`** (1 nodes): `Get the status of a specific order.          Args:             order_id: The uni`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Module 81`** (1 nodes): `Check if broker connection is active.          Returns:             True if conn`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Module 82`** (1 nodes): `Return True if this trade was profitable.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Module 83`** (1 nodes): `Return True if this trade was a loss.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Module 84`** (1 nodes): `Return absolute value of P&L.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Module 85`** (1 nodes): `Ensure exactly one of ``against`` or ``value`` is set.          Exception: the ``
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Module 86`** (1 nodes): `Normalize field names from config files.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Module 87`** (1 nodes): `Validate that required params are present for each method.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Module 88`** (1 nodes): `Parse start_time string to time object.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Module 89`** (1 nodes): `Parse end_time string to time object.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Module 90`** (1 nodes): `Alias for enabled (backward compat).`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Module 91`** (1 nodes): `Primary timeframe (backward compat).`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Module 92`** (1 nodes): `Max concurrent positions from risk_management.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Module 93`** (1 nodes): `Max trades per day from risk_management.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Module 94`** (1 nodes): `Load a strategy configuration from a file.          Args:             path: Path`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Module 95`** (1 nodes): `Load all strategy configurations from a directory.          Args:             di`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Module 96`** (1 nodes): `Save a strategy configuration to a file.          Args:             config: Stra`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Module 97`** (1 nodes): `Create a sample EMA crossover strategy for reference.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Module 98`** (1 nodes): `BacktestDataLoader (CSV + Resampling)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Module 99`** (1 nodes): `BacktestReportGenerator (Text + DataFrame Reports)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Module 100`** (1 nodes): `InstrumentManager (Download, Search, Option Chain, ATM Strike)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Module 101`** (1 nodes): `Indian Market Fee Structure (Brokerage, STT, GST, Stamp Duty)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Strategy module for the trading bot.  Provides technical indicators, strategy co` connect `Upstox Auth & Login` to `Market Data & Bar Builder`, `Backtester Engine & Dashboard`, `Strategy Condition Framework`, `Broker Interface & Orders`, `Dashboard & Monitoring`, `AI Agent Pipeline`, `Pattern Recognition`, `Data Loading & CSV`, `Research Analysis Models`, `Health Monitor`, `Broker Factory`, `Tick Processing`?**
  _High betweenness centrality (0.464) - this node is a cross-community bridge._
- **Why does `AnalysisComponent` connect `Market Data & Bar Builder` to `Research Analysis Models`, `Strategy Condition Framework`, `Volume Analyzer & Tests`, `Upstox Auth & Login`?**
  _High betweenness centrality (0.078) - this node is a cross-community bridge._
- **Why does `IndicatorEngine` connect `Strategy Condition Framework` to `Market Data & Bar Builder`, `Research Analysis Models`, `Health Monitor`, `Upstox Auth & Login`?**
  _High betweenness centrality (0.070) - this node is a cross-community bridge._
- **Are the 397 inferred relationships involving `AnalysisComponent` (e.g. with `TestTradeScorecardInit` and `TestFinalScoreCalculation`) actually correct?**
  _`AnalysisComponent` has 397 INFERRED edges - model-reasoned connections that need verification._
- **Are the 277 inferred relationships involving `IndicatorEngine` (e.g. with `TestCPRIndicator` and `TestCPRVWAPStrategyConfig`) actually correct?**
  _`IndicatorEngine` has 277 INFERRED edges - model-reasoned connections that need verification._
- **Are the 291 inferred relationships involving `MarketRegime` (e.g. with `TestTradeScorecardInit` and `TestFinalScoreCalculation`) actually correct?**
  _`MarketRegime` has 291 INFERRED edges - model-reasoned connections that need verification._
- **Are the 291 inferred relationships involving `VolatilityRegime` (e.g. with `TestTradeScorecardInit` and `TestFinalScoreCalculation`) actually correct?**
  _`VolatilityRegime` has 291 INFERRED edges - model-reasoned connections that need verification._