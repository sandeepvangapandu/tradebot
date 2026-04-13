# Backtest Harness Design — Full Live-Parity Backtesting

**Date:** 2026-04-07  
**Status:** Approved for implementation  
**Goal:** Replace the standalone `BacktestEngine` with a `BacktestHarness` that drives every live component (PaperBroker, RiskManager, PositionManager, PartialProfitManager, OrderManager, TradeAnalyzer, LearningIntegration) synchronously bar-by-bar from historical OHLCV data — so backtest results faithfully represent what the live system would do.

---

## Problem Statement

The current `BacktestEngine` (`src/backtest/engine.py`) reimplements execution logic inline:
- Its own fee calculation (diverges from `PaperBroker`)
- Its own position tracking (no trailing stops, no partial profit)
- Its own risk checks (flat daily loss %, not the full `RiskManager`)
- No research/AI scoring (live system requires 65+ score to trade)
- No learning system adaptation
- No strategy quarantine

A backtest run therefore does not predict live behaviour. This design replaces it with a harness that uses the real components unchanged.

---

## Architecture

### Guiding Principle

> The live system's WebSocket + BarBuilder + threads are **delivery plumbing**, not business logic. The harness replaces the plumbing with a deterministic bar-by-bar loop. Every component that contains business logic runs unmodified.

### New Files

| File | Purpose |
|------|---------|
| `src/backtest/harness.py` | `BacktestHarness` — synchronous driver |
| `src/backtest/bar_provider.py` | `BacktestBarProvider` — BarBuilder adapter for historical data |
| `src/backtest/runner.py` | CLI entry point |

### Modified Files

| File | Change |
|------|--------|
| `src/strategy/engine.py` | Add `evaluate_bar_sync(bars, bar_time)` public method |
| `src/dashboard/pages/backtester.py` | Swap `BacktestEngine` → `BacktestHarness` |

### Unchanged Files (Zero Modifications)

`PaperBroker`, `RiskManager`, `CircuitBreaker`, `PositionManager`, `PartialProfitManager`,
`OrderManager`, `TradeAnalyzer`, `LearningIntegration`, `StrategyQuarantine`,
`KellyPositionSizer`, `TradeLog`, `ConditionEvaluator`, `StrategyBuilder`, all research analyzers.

---

## Component Wiring

```python
# src/backtest/harness.py — startup sequence (no auth, no WebSocket)

from src.persistence.database import init_db, get_session
from src.execution.paper_broker import PaperBroker
from src.execution.partial_profit import PartialProfitManager, FOUR_TIER_CONFIG
from src.execution.position_manager import PositionManager
from src.execution.order_manager import OrderManager
from src.risk.risk_manager import RiskManager
from src.risk.circuit_breaker import CircuitBreaker
from src.risk.position_sizer import KellyPositionSizer
from src.risk.strategy_quarantine import StrategyQuarantine
from src.research.trade_analyzer import TradeAnalyzer
from src.learning.integration import LearningIntegration
from src.persistence.trade_log import TradeLog
from src.strategy.builder import StrategyBuilder
from src.strategy.engine import StrategyEngine
from src.backtest.bar_provider import BacktestBarProvider
from config.settings import get_settings

settings = get_settings()
db_session = get_session()          # SQLAlchemy session — same DB as live

paper_broker     = PaperBroker(initial_capital=capital, slippage_pct=settings.slippage_pct)
partial_profit   = PartialProfitManager(config=FOUR_TIER_CONFIG)
risk_manager     = RiskManager(
    capital=capital,
    max_daily_loss=settings.max_daily_loss,
    max_open_positions=settings.max_open_positions,
    max_position_size_pct=settings.max_position_size_pct,
    max_capital_deployment_pct=settings.max_capital_deployment_pct,
)
circuit_breaker  = CircuitBreaker(
    consecutive_loss_limit=settings.consecutive_loss_pause,
    pause_minutes=settings.pause_minutes,
)
strategy_quarantine = StrategyQuarantine()
position_sizer   = KellyPositionSizer()

# Bar provider: BarBuilder interface backed by historical data
bar_provider     = BacktestBarProvider(data)          # data: {instrument_key: DataFrame}

# Research module — same TradeAnalyzer, different data source
trade_analyzer   = TradeAnalyzer(
    settings=settings,
    bar_builder=bar_provider,           # ← BacktestBarProvider, not live BarBuilder
    historical_fetcher=bar_provider,    # ← same object, implements both interfaces
    instrument_manager=None,            # gracefully degrades options lookup
)

learning         = LearningIntegration(db_session=db_session, enabled=True)
trade_log        = TradeLog()

position_manager = PositionManager(
    broker=paper_broker,
    risk_manager=risk_manager,
    partial_profit_manager=partial_profit,
    on_position_close=learning.on_position_closed,   # learning adapts on every close
)

order_manager    = OrderManager(
    signal_queue=None,              # not used — we call process_signal() directly
    broker=paper_broker,
    trading_mode="paper",
    db_url=settings.database_url,
    trade_analyzer=trade_analyzer,
    risk_manager=risk_manager,
    position_manager=position_manager,
    position_sizer=position_sizer,
    strategy_quarantine=strategy_quarantine,
)
```

**Notes:**
- `TradeLog` writes to the live SQLite DB. Backtest trades carry `source="backtest"` tag so they don't pollute live P&L views.
- `LearningIntegration` loads its **current persisted lessons** from the DB at startup, then adapts further as trades close during the backtest run — exactly as live.
- `StrategyQuarantine` and `CircuitBreaker` start fresh per run (clean slate).
- No `InstrumentManager` is needed for backtest — `TradeAnalyzer` gracefully degrades when it can't resolve options instruments.

---

## BacktestBarProvider

**File:** `src/backtest/bar_provider.py`

Implements two interfaces with one class:
1. `BarBuilder.get_bars(instrument_key, timeframe, include_current)` — consumed by `TradeAnalyzer`
2. `HistoricalDataFetcher.fetch_candles(instrument_key, interval, days)` — consumed by `TradeAnalyzer._gather_data()`

```python
class BacktestBarProvider:
    """Drop-in replacement for BarBuilder + HistoricalDataFetcher.

    Holds the full pre-loaded OHLCV DataFrame per instrument.
    Serves slices up to current_bar_index — strictly no lookahead.
    Resamples 1-minute source data to 5min/15min/daily on demand.
    """

    def __init__(self, data: dict[str, pd.DataFrame]) -> None:
        # data: instrument_key → full 1-min OHLCV DataFrame (paisa prices, IST index)
        self._data = data
        self._current_index: dict[str, int] = {k: 0 for k in data}

    def advance(self, instrument_key: str, index: int) -> None:
        """Called by harness each bar before any evaluation."""
        self._current_index[instrument_key] = index

    def get_bars(
        self,
        instrument_key: str,
        timeframe: int,          # minutes: 1, 5, 15
        include_current: bool = False,
    ) -> pd.DataFrame:
        idx = self._current_index.get(instrument_key, 0)
        window = self._data[instrument_key].iloc[: idx + 1]
        if timeframe == 1:
            return window
        rule = f"{timeframe}min"
        return window.resample(rule).agg(
            {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
        ).dropna()

    def fetch_candles(
        self,
        instrument_key: str,
        interval: str = "day",
        days: int = 30,
    ) -> pd.DataFrame:
        idx = self._current_index.get(instrument_key, 0)
        window = self._data[instrument_key].iloc[: idx + 1]
        daily = window.resample("1D").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
        ).dropna()
        return daily.tail(days)
```

**No-lookahead guarantee:** `advance(index)` is called before any evaluation. The provider always slices `data[:index+1]`, so bar `i` only sees bars `0..i`.

---

## Per-Bar Loop (BacktestHarness.run)

```
for i, (bar_time, bar) in enumerate(master_df.iterrows()):

    ┌─ Advance bar provider ──────────────────────────────────────────┐
    │  bar_provider.advance(instrument_key, i)                         │
    │  paper_broker.update_ltp(instrument_key, bar.close)              │
    └─────────────────────────────────────────────────────────────────┘

    ┌─ Phase 1: New trading day? ─────────────────────────────────────┐
    │  if bar_time.date() != prev_date:                                │
    │      risk_manager.reset_daily()                                  │
    │      circuit_breaker.reset_daily()                               │
    │      strategy_quarantine.reset_daily_counts()                    │
    │      prev_date = bar_time.date()                                 │
    └─────────────────────────────────────────────────────────────────┘

    ┌─ Phase 2: Position exits ───────────────────────────────────────┐
    │  closed = position_manager.on_tick(instrument_key, bar.close)   │
    │  (PositionManager handles: SL, target, trailing SL,             │
    │   partial profit tiers, intraday square-off at 15:15)            │
    └─────────────────────────────────────────────────────────────────┘

    ┌─ Phase 3: Session gate (no new entries after 15:00) ───────────┐
    │  if bar_time.time() >= NO_NEW_ENTRY_AFTER: continue             │
    │  if circuit_breaker.is_paused(): continue                        │
    └─────────────────────────────────────────────────────────────────┘

    ┌─ Phase 4: Signal generation ────────────────────────────────────┐
    │  signals = strategy_engine.evaluate_bar_sync(                    │
    │      bars={instrument_key: master_df.iloc[:i+1]},               │
    │      bar_time=bar_time,                                          │
    │  )                                                               │
    │  (same ConditionEvaluator, same session filters as live)         │
    └─────────────────────────────────────────────────────────────────┘

    ┌─ Phase 5: Signal processing (per signal) ───────────────────────┐
    │  for signal in signals:                                          │
    │      signal.timestamp = bar_time   # historical time, not now   │
    │      order_manager.process_signal(signal)                        │
    │      # ↑ internally runs: validate → Kelly sizing →             │
    │      #   quarantine check → TradeAnalyzer.analyze() →           │
    │      #   RiskManager.pre_trade_check() → PaperBroker.place()    │
    └─────────────────────────────────────────────────────────────────┘

    ┌─ Phase 6: Equity snapshot ──────────────────────────────────────┐
    │  equity_curve.append(paper_broker.get_capital())                 │
    └─────────────────────────────────────────────────────────────────┘
```

**Order matters:** exits before entries, signal generation after exits. This matches the live event order exactly.

---

## StrategyEngine Change — evaluate_bar_sync

**File:** `src/strategy/engine.py`  
**Change:** Add one public method. The existing thread loop (`SetEvaluator.run`) calls this same method internally, guaranteeing parity.

```python
def evaluate_bar_sync(
    self,
    bars: dict[str, pd.DataFrame],
    bar_time: datetime,
) -> list[Signal]:
    """Evaluate all loaded strategies for a single bar (synchronous).

    Used by the backtest harness. Applies the same session filters,
    signal-per-day limits, and condition evaluation as the live thread path.

    Args:
        bars: Dict mapping instrument_key → OHLCV DataFrame (up to current bar).
        bar_time: Timestamp of the bar being evaluated (IST).

    Returns:
        List of Signal objects generated this bar.
    """
    signals: list[Signal] = []
    for strategy_config, entry_set in self._iter_strategy_sets():
        if not self._passes_session_filter(strategy_config, bar_time):
            continue
        instrument = strategy_config.instrument_key
        if instrument not in bars:
            continue
        window = {instrument: bars[instrument]}
        try:
            all_met = self._evaluator.evaluate_all(
                entry_set.conditions, window, instrument
            )
        except Exception:
            all_met = False
        if all_met:
            signal = self._build_signal(strategy_config, entry_set, bar_time)
            signals.append(signal)
    return signals
```

`_passes_session_filter`, `_iter_strategy_sets`, `_build_signal` are factored out of the existing `SetEvaluator.run()` — same logic, now shared.

---

## CLI Interface

**File:** `src/backtest/runner.py`

```bash
# Run backtest on all strategies in config/strategies/
python -m src.backtest.runner \
    --data nifty50_1min_30days.csv \
    --instrument "NSE_INDEX|Nifty Bank" \
    --capital 1000000 \
    --from 2026-01-01 \
    --to 2026-03-31

# Specify a subset of strategies
python -m src.backtest.runner \
    --data nifty50_1min_30days.csv \
    --instrument "NSE_INDEX|Nifty Bank" \
    --strategies orb_vwap_banknifty supertrend_ema_rsi_banknifty
```

**Output:**
```
┌─────────────────────────────────────────┐
│  BACKTEST RESULTS  2026-01-01 → 2026-03-31
├──────────────────┬──────────────────────┤
│ Total Trades     │ 47                   │
│ Win Rate         │ 57.4%                │
│ Profit Factor    │ 1.84                 │
│ Net P&L          │ ₹ 28,430             │
│ Max Drawdown     │ ₹ 7,200 (7.2%)       │
│ Sharpe Ratio     │ 1.43                 │
│ Total Fees       │ ₹ 2,180              │
│ Return           │ +2.84%               │
└──────────────────┴──────────────────────┘
Trade log saved to: data/trading_bot.db (source=backtest)
```

---

## Dashboard Integration

**File:** `src/dashboard/pages/backtester.py`

Replace the call to `BacktestEngine` with:
```python
from src.backtest.harness import BacktestHarness

harness = BacktestHarness(
    data_file=uploaded_csv_path,
    instrument_key=selected_instrument,
    capital=selected_capital,
    start_date=from_date,
    end_date=to_date,
)
results = harness.run()
# results.equity_curve, results.trades, results.metrics
# → same Plotly charts already in the page, unchanged
```

The dashboard gains a CSV file uploader widget and instrument/date inputs. Results feed the existing equity curve, trade table, and metrics cards.

---

## Data Requirements

The harness accepts **1-minute OHLCV CSV** as the canonical input (same format as `nifty50_1min_30days.csv`):

```
timestamp,open,high,low,close,volume
2026-01-02 09:15:00+05:30,5231500,5240000,5229000,5237000,12400
...
```

- Prices in **paisa** (integers)
- Timestamps **IST-aware**
- The `BacktestDataLoader` (existing, unchanged) handles loading and date filtering
- The `BacktestBarProvider` resamples to 5min/15min/daily on demand

If the user provides rupee-denominated data, a conversion flag `--prices-in-rupees` multiplies all price columns by 100 at load time.

---

## Research Module Behaviour in Backtest

| Analyzer | Data Source | Behaviour |
|---|---|---|
| TechnicalAnalyzer | `BacktestBarProvider` (OHLCV window) | Fully active |
| VolumeAnalyzer | `BacktestBarProvider` (OHLCV window) | Fully active |
| MarketRegimeDetector | `BacktestBarProvider` (15min + daily) | Fully active |
| PatternRecognizer | `BacktestBarProvider` (5min window) | Fully active |
| MLSignalValidator | Features from OHLCV window | Fully active |
| StrategyConfluenceAnalyzer | Signal history from this run | Fully active |
| OptionsAnalyzer | No live options chain | Gracefully degrades → neutral score |
| CorrelationAnalyzer | No live portfolio feed | Gracefully degrades → neutral score |
| EventCalendarAnalyzer | Bar timestamp (date only) | Active for known holiday/expiry checks |

Degraded analyzers contribute 0 to the weighted score rather than failing. The minimum score threshold (65) still applies — a signal with only OHLCV-based analyzers scoring well can still pass.

---

## Learning System Behaviour in Backtest

### During the run
- Loads persisted lessons from SQLite at harness startup — **only `source="live"` and `source="paper"` lessons** are loaded, never previous backtest lessons. This prevents historical backtest runs from skewing position sizing.
- `LearningIntegration.on_position_closed(position)` is called via `PositionManager`'s `on_position_close` callback — identical to live wiring.
- Position sizing adapts as the backtest progresses (trades early in the run influence later sizing), giving a realistic picture of how the learning system would behave in live trading.

### Lesson persistence
- Backtest lessons **are saved to SQLite** but tagged `source="backtest"` — the same tagging used for backtest trades.
- The live `LearningIntegration` filters its lesson load query to `source IN ("live", "paper")` — so backtest lessons never automatically drive live position sizing.
- This keeps backtest lesson data available for analysis and comparison without contaminating the live model.

### Why this is safe
Backtest win rates are optimistic by construction (strategies were designed knowing the historical data). Automatically promoting those lessons to live sizing would cause over-confident Kelly fractions. The tag-based separation lets you inspect whether backtest and live win rates converge over time — if they do, you can promote them.

### Optional promotion
A future CLI command `python -m src.backtest.promote_lessons --weight 0.5` can merge backtest lessons into the live model at a reduced weight (default 50%). This is a deliberate, one-time action — not automatic.

---

## What BacktestEngine (engine.py) Becomes

The existing `BacktestEngine` class is **kept as-is** for backward compatibility with any existing tests or scripts that reference it. It is no longer the primary backtesting path. The dashboard and CLI both use `BacktestHarness`. The old engine's tests remain valid.

---

## Testing Strategy

- `tests/test_backtest_harness.py` — integration test: loads `nifty50_1min_30days.csv`, runs harness for 5 trading days, asserts:
  - At least one trade is produced
  - All trades have valid entry/exit prices, fees > 0
  - Equity curve length equals number of bars processed
  - No trade has zero quantity
- `tests/test_bar_provider.py` — unit tests:
  - `get_bars()` never returns bars beyond `current_index`
  - Resampling 1min → 5min produces correct OHLCV aggregation
  - `fetch_candles()` returns correct daily bars
- Existing `tests/test_backtest_*.py` continue to pass (engine.py unchanged)
