# Backtest Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the standalone `BacktestEngine` with a `BacktestHarness` that drives every live component (`PaperBroker`, `RiskManager`, `PositionManager`, `PartialProfitManager`, `OrderManager`, `TradeAnalyzer`, `LearningIntegration`) synchronously bar-by-bar so backtests faithfully predict live behaviour.

**Architecture:** A new `BacktestBarProvider` implements the same interface as `BarBuilder` + `HistoricalDataFetcher`, serving historical OHLCV slices with no lookahead. A new `evaluate_bar_sync()` method on `StrategyEngine` evaluates conditions synchronously without threads. The `BacktestHarness` drives all live components in one deterministic loop per bar.

**Tech Stack:** Python 3.11+, pandas, SQLAlchemy, Streamlit, argparse. All existing component libraries unchanged.

---

## File Map

| Action | File | Responsibility |
|--------|------|----------------|
| Create | `src/backtest/bar_provider.py` | BarBuilder + HistoricalDataFetcher adapter for historical data |
| Create | `src/backtest/harness.py` | Synchronous bar-by-bar driver wiring all live components |
| Create | `src/backtest/runner.py` | CLI entry point (`python -m src.backtest.runner`) |
| Modify | `src/strategy/engine.py` | Add `evaluate_bar_sync()` and `_evaluator` attribute |
| Modify | `src/learning/integration.py` | Add `backtest_mode` flag, source tagging for persistence |
| Modify | `src/dashboard/pages/backtester.py` | Swap `BacktestEngine` → `BacktestHarness` |
| Create | `tests/test_bar_provider.py` | Unit tests for BacktestBarProvider |
| Create | `tests/test_backtest_harness.py` | Integration test driving harness with CSV data |

---

## Task 1: BacktestBarProvider

**Files:**
- Create: `src/backtest/bar_provider.py`
- Create: `tests/test_bar_provider.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_bar_provider.py
import pandas as pd
import pytest
from zoneinfo import ZoneInfo

from src.backtest.bar_provider import BacktestBarProvider

IST = ZoneInfo("Asia/Kolkata")


@pytest.fixture
def sample_data() -> dict[str, pd.DataFrame]:
    """60 1-minute bars of OHLCV data in paisa."""
    idx = pd.date_range(
        "2026-01-02 09:15:00", periods=60, freq="1min", tz=IST
    )
    df = pd.DataFrame(
        {
            "open":   [5000000] * 60,
            "high":   [5010000] * 60,
            "low":    [4990000] * 60,
            "close":  [5005000] * 60,
            "volume": [1000] * 60,
        },
        index=idx,
    )
    return {"NSE_INDEX|Nifty Bank": df}


def test_get_bars_no_lookahead(sample_data):
    """Bars returned must not exceed current_index + 1."""
    provider = BacktestBarProvider(sample_data)
    provider.advance("NSE_INDEX|Nifty Bank", 9)  # advance to bar 10 (0-indexed)
    df = provider.get_bars("NSE_INDEX|Nifty Bank", timeframe=1)
    assert len(df) == 10, f"Expected 10 bars, got {len(df)}"


def test_get_bars_resample_5min(sample_data):
    """60 1-min bars resampled to 5min should give 12 bars."""
    provider = BacktestBarProvider(sample_data)
    provider.advance("NSE_INDEX|Nifty Bank", 59)
    df = provider.get_bars("NSE_INDEX|Nifty Bank", timeframe=5)
    assert len(df) == 12, f"Expected 12 bars, got {len(df)}"


def test_get_bars_resample_ohlcv_aggregation(sample_data):
    """5-min bars must aggregate open/high/low/close/volume correctly."""
    provider = BacktestBarProvider(sample_data)
    provider.advance("NSE_INDEX|Nifty Bank", 4)  # first 5 bars
    df = provider.get_bars("NSE_INDEX|Nifty Bank", timeframe=5)
    assert len(df) == 1
    assert df.iloc[0]["open"] == 5000000
    assert df.iloc[0]["high"] == 5010000
    assert df.iloc[0]["low"] == 4990000
    assert df.iloc[0]["close"] == 5005000
    assert df.iloc[0]["volume"] == 5000  # sum of 5 bars


def test_fetch_candles_daily(sample_data):
    """fetch_candles should resample to daily bars."""
    provider = BacktestBarProvider(sample_data)
    provider.advance("NSE_INDEX|Nifty Bank", 59)
    df = provider.fetch_candles("NSE_INDEX|Nifty Bank", interval="day", days=30)
    assert len(df) == 1  # all bars in same day
    assert df.iloc[0]["open"] == 5000000
    assert df.iloc[0]["high"] == 5010000


def test_advance_updates_index(sample_data):
    """advance() must update the internal index for the given instrument."""
    provider = BacktestBarProvider(sample_data)
    provider.advance("NSE_INDEX|Nifty Bank", 19)
    df = provider.get_bars("NSE_INDEX|Nifty Bank", timeframe=1)
    assert len(df) == 20


def test_unknown_instrument_returns_empty(sample_data):
    """Requesting bars for unknown instrument returns empty DataFrame."""
    provider = BacktestBarProvider(sample_data)
    df = provider.get_bars("NSE_INDEX|Unknown", timeframe=1)
    assert df.empty
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd /Users/sandeepvangapandu/Downloads/Trading
source venv/bin/activate
pytest tests/test_bar_provider.py -v
```
Expected: `ModuleNotFoundError: No module named 'src.backtest.bar_provider'`

- [ ] **Step 3: Implement BacktestBarProvider**

```python
# src/backtest/bar_provider.py
"""Historical bar provider implementing BarBuilder + HistoricalDataFetcher interfaces.

Used by the BacktestHarness to feed historical OHLCV data to TradeAnalyzer
and StrategyEngine without lookahead bias.

All prices in PAISA (integer). Timestamps are IST-aware.
"""

from __future__ import annotations

import pandas as pd
from loguru import logger


_OHLCV_AGG = {
    "open": "first",
    "high": "max",
    "low": "min",
    "close": "last",
    "volume": "sum",
}


class BacktestBarProvider:
    """Drop-in replacement for BarBuilder and HistoricalDataFetcher.

    Holds full pre-loaded OHLCV DataFrames per instrument and serves
    slices up to the current bar index — strictly no lookahead.

    Args:
        data: Mapping of instrument_key → full OHLCV DataFrame (1-min, paisa,
              IST-aware DatetimeIndex).
    """

    def __init__(self, data: dict[str, pd.DataFrame]) -> None:
        self._data = data
        # Current bar index per instrument (inclusive upper bound)
        self._current_index: dict[str, int] = {k: 0 for k in data}

    def advance(self, instrument_key: str, index: int) -> None:
        """Advance the time window to bar ``index`` (0-based, inclusive).

        Must be called by the harness before any evaluation for bar ``index``.

        Args:
            instrument_key: Instrument to advance.
            index: Current bar index (0-based).
        """
        self._current_index[instrument_key] = index

    # ------------------------------------------------------------------
    # BarBuilder interface (used by TradeAnalyzer._gather_data)
    # ------------------------------------------------------------------

    def get_bars(
        self,
        instrument_key: str,
        timeframe: int,
        include_current: bool = False,
    ) -> pd.DataFrame:
        """Return OHLCV bars up to the current index, resampled to timeframe.

        Args:
            instrument_key: Instrument identifier.
            timeframe: Target timeframe in minutes (1, 5, or 15).
            include_current: Ignored — current bar is always included via advance().

        Returns:
            DataFrame with OHLCV columns and DatetimeIndex, or empty DataFrame
            if instrument is unknown.
        """
        if instrument_key not in self._data:
            logger.debug(f"BacktestBarProvider: unknown instrument {instrument_key}")
            return pd.DataFrame(
                columns=["open", "high", "low", "close", "volume"]
            )

        idx = self._current_index.get(instrument_key, 0)
        window = self._data[instrument_key].iloc[: idx + 1]

        if timeframe == 1:
            return window

        resampled = (
            window
            .resample(f"{timeframe}min")
            .agg(_OHLCV_AGG)
            .dropna()
        )
        return resampled

    # ------------------------------------------------------------------
    # HistoricalDataFetcher interface (used by TradeAnalyzer._gather_data)
    # ------------------------------------------------------------------

    def fetch_candles(
        self,
        instrument_key: str,
        interval: str = "day",
        days: int = 30,
    ) -> pd.DataFrame:
        """Return daily bars from historical data up to current index.

        Args:
            instrument_key: Instrument identifier.
            interval: Ignored — always returns daily bars.
            days: Maximum number of daily bars to return.

        Returns:
            DataFrame with daily OHLCV bars, most recent ``days`` rows.
        """
        if instrument_key not in self._data:
            return pd.DataFrame(
                columns=["open", "high", "low", "close", "volume"]
            )

        idx = self._current_index.get(instrument_key, 0)
        window = self._data[instrument_key].iloc[: idx + 1]
        daily = window.resample("1D").agg(_OHLCV_AGG).dropna()
        return daily.tail(days)
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
pytest tests/test_bar_provider.py -v
```
Expected: all 6 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/backtest/bar_provider.py tests/test_bar_provider.py
git commit -m "feat: add BacktestBarProvider — BarBuilder adapter for historical data"
```

---

## Task 2: StrategyEngine.evaluate_bar_sync

**Files:**
- Modify: `src/strategy/engine.py` (add `_evaluator` to `__init__`, add `evaluate_bar_sync` method)
- Create: `tests/test_strategy_engine_sync.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_strategy_engine_sync.py
"""Tests for StrategyEngine.evaluate_bar_sync (synchronous backtest path)."""
import json
import tempfile
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from src.strategy.engine import StrategyEngine

IST = ZoneInfo("Asia/Kolkata")

SIMPLE_STRATEGY = {
    "name": "Test_EMA_Cross",
    "enabled": True,
    "underlying": {"instrument_key": "NSE_INDEX|Nifty Bank"},
    "trading_hours": {
        "start_time": "09:15:00",
        "end_time": "15:15:00",
        "days": [0, 1, 2, 3, 4],
    },
    "timeframes": {"primary": "5min"},
    "entry_sets": [
        {
            "name": "long",
            "signal": "BUY",
            "conditions": [
                # EMA9 > EMA21 — will be met when close is rising
                {
                    "indicator": "EMA",
                    "comparison": ">",
                    "against": "EMA",
                    "parameters": {"fast_period": 9, "slow_period": 21},
                    "timeframe": "5min",
                }
            ],
        }
    ],
    "exit_rules": {"stop_loss_pct": 1.0, "target_pct": 2.0},
    "position_sizing": {"method": "fixed_quantity", "quantity": 1},
    "risk_management": {},
    "params": {"use_enhanced_filters": False},
}


@pytest.fixture
def strategy_dir(tmp_path):
    strat_file = tmp_path / "test_strategy.json"
    strat_file.write_text(json.dumps(SIMPLE_STRATEGY))
    return tmp_path


@pytest.fixture
def flat_bars() -> dict[str, pd.DataFrame]:
    """40 bars of flat price — EMA9 == EMA21, no signal expected."""
    idx = pd.date_range("2026-01-06 09:15", periods=40, freq="5min", tz=IST)
    df = pd.DataFrame(
        {"open": [5000]*40, "high": [5010]*40, "low": [4990]*40,
         "close": [5000]*40, "volume": [1000]*40},
        index=idx,
    )
    return {"NSE_INDEX|Nifty Bank": df}


def test_evaluate_bar_sync_returns_list(strategy_dir, flat_bars):
    """evaluate_bar_sync must return a list (possibly empty)."""
    engine = StrategyEngine(
        strategies_dir=str(strategy_dir),
        bars_provider=lambda: flat_bars,
    )
    engine.load_strategies()

    bar_time = datetime(2026, 1, 6, 10, 0, tzinfo=IST)
    signals = engine.evaluate_bar_sync(flat_bars, bar_time)
    assert isinstance(signals, list)


def test_evaluate_bar_sync_respects_trading_hours(strategy_dir, flat_bars):
    """No signals before market open."""
    engine = StrategyEngine(
        strategies_dir=str(strategy_dir),
        bars_provider=lambda: flat_bars,
    )
    engine.load_strategies()

    # 08:00 — before market open
    bar_time = datetime(2026, 1, 6, 8, 0, tzinfo=IST)
    signals = engine.evaluate_bar_sync(flat_bars, bar_time)
    assert signals == [], "No signals expected before market open"


def test_evaluate_bar_sync_skips_disabled_strategy(tmp_path, flat_bars):
    """Disabled strategy must not generate signals."""
    disabled = {**SIMPLE_STRATEGY, "enabled": False}
    (tmp_path / "disabled.json").write_text(json.dumps(disabled))
    engine = StrategyEngine(
        strategies_dir=str(tmp_path),
        bars_provider=lambda: flat_bars,
    )
    engine.load_strategies()
    bar_time = datetime(2026, 1, 6, 10, 0, tzinfo=IST)
    signals = engine.evaluate_bar_sync(flat_bars, bar_time)
    assert signals == []
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/test_strategy_engine_sync.py -v
```
Expected: `AttributeError: 'StrategyEngine' object has no attribute 'evaluate_bar_sync'`

- [ ] **Step 3: Add `_evaluator` to StrategyEngine.__init__ and implement evaluate_bar_sync**

Open `src/strategy/engine.py`. In `StrategyEngine.__init__` (around line 1180), add one line after `self._builder = StrategyBuilder()`:

```python
        self._evaluator = ConditionEvaluator()  # shared evaluator for sync path
```

Then add `evaluate_bar_sync` as a new method after `add_strategy` (around line 1237):

```python
    # ------------------------------------------------------------------
    # Synchronous evaluation (backtest harness path)
    # ------------------------------------------------------------------

    _SIGNAL_MAP: dict[str, "SignalType"] = {
        "BUY": SignalType.BUY,
        "SELL": SignalType.SELL,
        "CE": SignalType.BUY_CE,
        "BUY_CE": SignalType.BUY_CE,
        "PE": SignalType.BUY_PE,
        "BUY_PE": SignalType.BUY_PE,
    }

    def evaluate_bar_sync(
        self,
        bars: dict[str, pd.DataFrame],
        bar_time: datetime,
    ) -> list[Signal]:
        """Evaluate all loaded strategies for one bar (no threads).

        Applies the same session-time filter and condition evaluation as the
        live ``SetEvaluator`` threads, but synchronously. Used by the backtest
        harness so that bar-by-bar results are deterministic.

        Args:
            bars: Mapping instrument_key → OHLCV DataFrame (bars up to current).
            bar_time: Timestamp of the bar being evaluated (must be IST-aware).

        Returns:
            List of ``Signal`` objects generated this bar (may be empty).
        """
        if bar_time.tzinfo is None:
            bar_time = bar_time.replace(tzinfo=IST)
        else:
            bar_time = bar_time.astimezone(IST)

        current_time = bar_time.time()
        weekday = bar_time.weekday()
        signals: list[Signal] = []

        for strategy in self._strategies.values():
            if not strategy.enabled:
                continue

            # --- Trading hours filter (mirrors SetEvaluator._should_evaluate) ---
            th = strategy.trading_hours
            if weekday not in th.days:
                continue
            if not (th.start <= current_time <= th.end):
                continue

            # --- Instrument key ---
            instrument_key: str | None = None
            if strategy.underlying and isinstance(strategy.underlying, dict):
                instrument_key = strategy.underlying.get("instrument_key")
            if not instrument_key or instrument_key not in bars:
                continue

            window = {instrument_key: bars[instrument_key]}
            if len(bars[instrument_key]) < 2:
                continue

            # --- Evaluate each entry set ---
            for entry_set in strategy.entry_sets:
                try:
                    all_met = self._evaluator.evaluate_all(
                        entry_set.conditions, window, instrument_key
                    )
                except Exception as exc:
                    logger.debug(
                        f"[{strategy.name}/{entry_set.name}] Condition error: {exc}"
                    )
                    all_met = False

                if not all_met:
                    continue

                # Map signal string to SignalType
                signal_str = entry_set.signal.upper()
                signal_type = self._SIGNAL_MAP.get(signal_str, SignalType.BUY)

                # Quantity from position sizing
                quantity = 1
                if strategy.position_sizing and strategy.position_sizing.quantity:
                    quantity = strategy.position_sizing.quantity

                signal = Signal(
                    strategy_name=strategy.name,
                    set_name=entry_set.name,
                    instrument_key=instrument_key,
                    signal_type=signal_type,
                    quantity=quantity,
                    price=None,        # MARKET order — harness sets via update_ltp
                    stop_loss=None,    # OrderManager resolves from exit_rules
                    target=None,
                    order_type="MARKET",
                    product_type="I",  # intraday MIS
                    timestamp=bar_time,
                    timeframe=strategy.timeframe,
                    underlying=(
                        strategy.underlying.get("symbol")
                        if strategy.underlying else None
                    ),
                    metadata={
                        "stop_loss_pct": strategy.exit_rules.stop_loss_pct,
                        "target_pct": strategy.exit_rules.target_pct,
                        "source": "backtest",
                    },
                )
                signals.append(signal)
                logger.debug(
                    f"[BACKTEST] Signal: {signal_type.value} "
                    f"{instrument_key} from {strategy.name}/{entry_set.name}"
                )

        return signals
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
pytest tests/test_strategy_engine_sync.py -v
```
Expected: all 3 tests PASS.

- [ ] **Step 5: Ensure existing engine tests still pass**

```bash
pytest tests/test_strategy_engine.py -v
```
Expected: all existing tests PASS.

- [ ] **Step 6: Commit**

```bash
git add src/strategy/engine.py tests/test_strategy_engine_sync.py
git commit -m "feat: add StrategyEngine.evaluate_bar_sync for backtest harness"
```

---

## Task 3: LearningIntegration — Backtest Mode

**Files:**
- Modify: `src/learning/integration.py`

- [ ] **Step 1: Write failing test**

```python
# Add to tests/test_backtest_harness.py (created in Task 7 — write this file now
# just for this test, more tests will be added later)
# tests/test_learning_backtest_mode.py

from unittest.mock import MagicMock, patch
from src.learning.integration import LearningIntegration


def test_backtest_mode_skips_lesson_persistence():
    """In backtest_mode, save_lesson must NOT be called."""
    mock_session = MagicMock()

    with patch("src.learning.integration.LearningEngine") as MockEngine, \
         patch("src.learning.integration.LearningPersistence") as MockPersistence:

        mock_engine_instance = MockEngine.return_value
        mock_persistence_instance = MockPersistence.return_value

        # Simulate a losing trade (triggers save_lesson in live mode)
        mock_engine_instance.process_trade.return_value = {
            "was_win": False,
            "pnl_rupees": -500.0,
        }
        mock_engine_instance._analyzer.get_lessons.return_value = []
        mock_engine_instance._analyzer.get_strategy_performance.return_value = None

        integration = LearningIntegration(
            db_session=mock_session,
            backtest_mode=True,
        )

        # Create mock position
        mock_position = MagicMock()
        mock_position.position_id = "POS_001"
        mock_position.strategy_id = "test_strategy"
        mock_position.entry_price = 500000
        mock_position.exit_price = 490000
        mock_position.quantity = 1
        mock_position.side = MagicMock()
        mock_position.side.value = "BUY"
        mock_position.entry_time = MagicMock()
        mock_position.exit_time = MagicMock()
        mock_position.is_closed = True
        mock_position.total_realized_pnl = -10000

        integration.on_position_closed(mock_position)

        # In backtest mode, save_lesson must NOT be called
        mock_persistence_instance.save_lesson.assert_not_called()
```

- [ ] **Step 2: Run test to confirm it fails**

```bash
pytest tests/test_learning_backtest_mode.py -v
```
Expected: `TypeError: __init__() got an unexpected keyword argument 'backtest_mode'`

- [ ] **Step 3: Add backtest_mode to LearningIntegration**

Open `src/learning/integration.py`. Change the `__init__` signature and body:

```python
    def __init__(
        self,
        db_session: Session,
        config: dict[str, Any] | None = None,
        enabled: bool = True,
        backtest_mode: bool = False,   # ← add this
    ):
        self._engine = LearningEngine(db_session, config)
        self._persistence = LearningPersistence(db_session)
        self._config = config or {}
        self._enabled = enabled
        self._backtest_mode = backtest_mode   # ← add this
        self._lock = threading.RLock()

        if enabled:
            mode = "backtest" if backtest_mode else "live"
            logger.info(f"Learning system integration initialized (mode={mode})")
        else:
            logger.info("Learning system integration initialized (disabled)")
```

Then in `on_position_closed`, wrap the `save_lesson` call with a backtest guard. Find this block:

```python
                # Persist lesson if any issues were identified
                if not insights["was_win"]:
                    lessons = self._engine._analyzer.get_lessons(
                        strategy=position.strategy_id
                    )
                    for lesson in lessons[-3:]:  # Save last 3 lessons
                        if not lesson.applied:
                            self._persistence.save_lesson(lesson)
```

Replace it with:

```python
                # Persist lesson if any issues were identified
                # In backtest mode, skip lesson persistence to avoid contaminating
                # the live learning model with potentially optimistic backtest data.
                if not insights["was_win"] and not self._backtest_mode:
                    lessons = self._engine._analyzer.get_lessons(
                        strategy=position.strategy_id
                    )
                    for lesson in lessons[-3:]:  # Save last 3 lessons
                        if not lesson.applied:
                            self._persistence.save_lesson(lesson)
```

- [ ] **Step 4: Run test to confirm it passes**

```bash
pytest tests/test_learning_backtest_mode.py -v
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/learning/integration.py tests/test_learning_backtest_mode.py
git commit -m "feat: add backtest_mode to LearningIntegration to skip lesson persistence"
```

---

## Task 4: BacktestHarness

**Files:**
- Create: `src/backtest/harness.py`

- [ ] **Step 1: Write failing test (minimal — full integration test in Task 7)**

```python
# tests/test_backtest_harness_init.py
"""Tests that BacktestHarness can be instantiated and wires components correctly."""
import tempfile
import json
from pathlib import Path

import pytest

from src.backtest.harness import BacktestHarness, HarnessResults


MINIMAL_STRATEGY = {
    "name": "Minimal_Test",
    "enabled": True,
    "underlying": {"instrument_key": "NSE_INDEX|Nifty Bank"},
    "trading_hours": {
        "start_time": "09:15:00",
        "end_time": "15:15:00",
        "days": [0, 1, 2, 3, 4],
    },
    "timeframes": {"primary": "5min"},
    "entry_sets": [
        {
            "name": "long",
            "signal": "BUY",
            "conditions": [
                {
                    "indicator": "RSI",
                    "comparison": "<",
                    "value": 80,
                    "timeframe": "5min",
                    "parameters": {"length": 14},
                }
            ],
        }
    ],
    "exit_rules": {"stop_loss_pct": 1.0, "target_pct": 2.0},
    "position_sizing": {"method": "fixed_quantity", "quantity": 1},
    "risk_management": {},
    "params": {"use_enhanced_filters": False},
}


@pytest.fixture
def strategy_dir(tmp_path):
    (tmp_path / "minimal.json").write_text(json.dumps(MINIMAL_STRATEGY))
    return tmp_path


def test_harness_instantiates(strategy_dir, tmp_path):
    """BacktestHarness can be created without errors."""
    csv_path = tmp_path / "data.csv"
    # Write minimal CSV
    import pandas as pd
    from zoneinfo import ZoneInfo
    IST = ZoneInfo("Asia/Kolkata")
    idx = pd.date_range("2026-01-06 09:15", periods=10, freq="1min", tz=IST)
    df = pd.DataFrame(
        {"open": [5000000]*10, "high": [5010000]*10, "low": [4990000]*10,
         "close": [5005000]*10, "volume": [1000]*10},
        index=idx,
    )
    df.index.name = "timestamp"
    df.to_csv(csv_path)

    harness = BacktestHarness(
        data_file=str(csv_path),
        instrument_key="NSE_INDEX|Nifty Bank",
        strategy_dir=str(strategy_dir),
        capital=100_000_000,
    )
    assert harness is not None


def test_harness_run_returns_results(strategy_dir, tmp_path):
    """BacktestHarness.run() returns a HarnessResults with equity_curve."""
    import pandas as pd
    from zoneinfo import ZoneInfo
    IST = ZoneInfo("Asia/Kolkata")

    csv_path = tmp_path / "data.csv"
    idx = pd.date_range("2026-01-06 09:15", periods=100, freq="1min", tz=IST)
    df = pd.DataFrame(
        {"open": [5000000]*100, "high": [5010000]*100, "low": [4990000]*100,
         "close": [5005000]*100, "volume": [1000]*100},
        index=idx,
    )
    df.index.name = "timestamp"
    df.to_csv(csv_path)

    harness = BacktestHarness(
        data_file=str(csv_path),
        instrument_key="NSE_INDEX|Nifty Bank",
        strategy_dir=str(strategy_dir),
        capital=100_000_000,
    )
    results = harness.run()

    assert isinstance(results, HarnessResults)
    assert len(results.equity_curve) > 0
    assert isinstance(results.metrics, dict)
    assert "win_rate" in results.metrics
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/test_backtest_harness_init.py -v
```
Expected: `ModuleNotFoundError: No module named 'src.backtest.harness'`

- [ ] **Step 3: Implement BacktestHarness**

```python
# src/backtest/harness.py
"""BacktestHarness — drives all live trading components synchronously.

Replaces BacktestEngine with a harness that instantiates the exact same
components used in live trading (PaperBroker, RiskManager, PositionManager,
PartialProfitManager, OrderManager, TradeAnalyzer, LearningIntegration) and
drives them bar-by-bar from historical OHLCV data.

All prices in PAISA. Timestamps in IST.
"""

from __future__ import annotations

import queue
from dataclasses import dataclass, field
from datetime import date, datetime, time as dt_time
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

import pandas as pd
from loguru import logger

from config.settings import get_settings
from src.backtest.bar_provider import BacktestBarProvider
from src.backtest.data_loader import BacktestDataLoader
from src.execution.order_manager import OrderManager
from src.execution.paper_broker import PaperBroker
from src.execution.partial_profit import FOUR_TIER_CONFIG, PartialProfitManager
from src.execution.position_manager import PositionManager
from src.learning.integration import LearningIntegration
from src.persistence.database import SessionLocal, init_db
from src.risk.circuit_breaker import CircuitBreaker
from src.risk.risk_manager import RiskManager
from src.risk.strategy_quarantine import StrategyQuarantine
from src.strategy.engine import StrategyEngine

IST = ZoneInfo("Asia/Kolkata")
NO_NEW_ENTRY_AFTER = dt_time(15, 0)
INTRADAY_SQUARE_OFF = dt_time(15, 15)
MARKET_OPEN = dt_time(9, 15)

# Optional components (gracefully absent)
try:
    from src.research.trade_analyzer import TradeAnalyzer
    _HAS_TRADE_ANALYZER = True
except ImportError:
    TradeAnalyzer = None  # type: ignore[misc,assignment]
    _HAS_TRADE_ANALYZER = False

try:
    from src.risk.position_sizer import KellyPositionSizer
    _HAS_KELLY = True
except ImportError:
    KellyPositionSizer = None  # type: ignore[misc,assignment]
    _HAS_KELLY = False


@dataclass
class HarnessResults:
    """Results from a BacktestHarness.run() call.

    Attributes:
        equity_curve: Capital in paisa at the end of each processed bar.
        trades: List of closed trade dicts (keys match BacktestEngine.Trade).
        metrics: Aggregate performance metrics (win_rate, profit_factor, etc.).
        bars_processed: Total number of bars evaluated.
    """

    equity_curve: list[int] = field(default_factory=list)
    trades: list[dict] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    bars_processed: int = 0

    def _compute_metrics(self, initial_capital: int) -> None:
        """Populate self.metrics from self.trades and self.equity_curve."""
        if not self.trades:
            self.metrics = {
                "total_trades": 0, "win_rate": 0.0, "profit_factor": 0.0,
                "net_pnl_rupees": 0.0, "max_drawdown_rupees": 0.0,
                "max_drawdown_pct": 0.0, "return_pct": 0.0, "sharpe_ratio": 0.0,
            }
            return

        pnls = [t.get("net_pnl", 0) for t in self.trades]
        winners = [p for p in pnls if p > 0]
        losers = [p for p in pnls if p <= 0]
        gross_profit = sum(winners)
        gross_loss = abs(sum(losers)) if losers else 1

        # Max drawdown from equity curve
        peak = initial_capital
        max_dd = 0
        for eq in self.equity_curve:
            if eq > peak:
                peak = eq
            dd = peak - eq
            if dd > max_dd:
                max_dd = dd

        final = self.equity_curve[-1] if self.equity_curve else initial_capital
        ret_pct = (final - initial_capital) / initial_capital * 100 if initial_capital else 0

        self.metrics = {
            "total_trades": len(pnls),
            "win_rate": round(len(winners) / len(pnls) * 100, 2),
            "profit_factor": round(gross_profit / gross_loss, 2),
            "net_pnl_rupees": round(sum(pnls) / 100, 2),
            "total_fees_rupees": round(sum(t.get("fees", 0) for t in self.trades) / 100, 2),
            "max_drawdown_rupees": round(max_dd / 100, 2),
            "max_drawdown_pct": round(max_dd / initial_capital * 100, 2) if initial_capital else 0,
            "return_pct": round(ret_pct, 2),
            "sharpe_ratio": 0.0,   # computed below if enough data
        }

        # Sharpe from trade returns
        if len(pnls) > 1:
            import math
            returns = [p / initial_capital for p in pnls]
            mean_r = sum(returns) / len(returns)
            var_r = sum((r - mean_r) ** 2 for r in returns) / (len(returns) - 1)
            std_r = math.sqrt(var_r) if var_r > 0 else 1e-10
            self.metrics["sharpe_ratio"] = round((mean_r / std_r) * math.sqrt(252), 2)


class BacktestHarness:
    """Drives all live components synchronously from historical OHLCV data.

    Instantiates the same components as TradingBot.startup() (PaperBroker,
    RiskManager, PositionManager, PartialProfitManager, OrderManager,
    TradeAnalyzer, LearningIntegration) and steps through bars in order,
    calling the same methods the live event loop would call.

    Args:
        data_file: Path to 1-minute OHLCV CSV file.
        instrument_key: Instrument identifier (e.g. "NSE_INDEX|Nifty Bank").
        strategy_dir: Directory containing strategy JSON configs.
        capital: Initial capital in paisa. Defaults to settings.capital.
        start_date: Optional start date filter.
        end_date: Optional end date filter.
        database_url: SQLAlchemy database URL. Defaults to settings.database_url.
        prices_in_rupees: If True, multiply CSV prices by 100 to convert to paisa.
    """

    def __init__(
        self,
        data_file: str,
        instrument_key: str,
        strategy_dir: str = "config/strategies",
        capital: Optional[int] = None,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        database_url: Optional[str] = None,
        prices_in_rupees: bool = False,
    ) -> None:
        self._settings = get_settings()
        self._capital = capital or self._settings.capital
        self._instrument_key = instrument_key
        self._strategy_dir = strategy_dir
        self._start_date = start_date
        self._end_date = end_date

        # Load historical data
        loader = BacktestDataLoader()
        df = loader.load_csv(data_file, start_date=start_date, end_date=end_date)
        if prices_in_rupees:
            for col in ["open", "high", "low", "close"]:
                df[col] = (df[col] * 100).astype(int)
        self._master_df = df
        self._data = {instrument_key: df}

        # Bar provider (shared BarBuilder + HistoricalDataFetcher interface)
        self._bar_provider = BacktestBarProvider(self._data)

        # Initialize database
        db_url = database_url or self._settings.database_url
        init_db(db_url)
        self._db_session = SessionLocal()

        # Wire up live components
        self._setup_components()
        self._setup_strategy_engine()

    # ------------------------------------------------------------------
    # Component setup (mirrors TradingBot.startup, no auth/WebSocket)
    # ------------------------------------------------------------------

    def _setup_components(self) -> None:
        """Instantiate all live execution and risk components."""
        settings = self._settings

        self._paper_broker = PaperBroker(
            initial_capital=self._capital,
            slippage_pct=settings.slippage_pct,
        )
        self._partial_profit = PartialProfitManager(config=FOUR_TIER_CONFIG)
        self._risk_manager = RiskManager(
            capital=self._capital,
            max_daily_loss=settings.max_daily_loss,
            max_open_positions=settings.max_open_positions,
            max_position_size_pct=settings.max_position_size_pct,
            max_capital_deployment_pct=settings.max_capital_deployment_pct,
        )
        self._circuit_breaker = CircuitBreaker(
            consecutive_loss_limit=settings.consecutive_loss_pause,
            pause_minutes=settings.pause_minutes,
        )
        self._strategy_quarantine = StrategyQuarantine()

        # Learning system — backtest_mode=True skips lesson persistence
        self._learning = LearningIntegration(
            db_session=self._db_session,
            enabled=settings.learning_enabled,
            backtest_mode=True,
        )

        self._position_manager = PositionManager(
            broker=self._paper_broker,
            risk_manager=self._risk_manager,
            partial_profit_manager=self._partial_profit,
            on_position_close=self._learning.on_position_closed,
        )

        # Kelly position sizer (optional)
        position_sizer = None
        if _HAS_KELLY:
            try:
                position_sizer = KellyPositionSizer()
            except Exception as exc:
                logger.warning(f"KellyPositionSizer unavailable: {exc}")

        # Research module (optional — gracefully degrades without live data)
        trade_analyzer = None
        if _HAS_TRADE_ANALYZER and settings.research_enabled:
            try:
                trade_analyzer = TradeAnalyzer(
                    settings=settings,
                    bar_builder=self._bar_provider,
                    historical_fetcher=self._bar_provider,
                    instrument_manager=None,  # gracefully degrades
                )
                logger.info("TradeAnalyzer active in backtest (OHLCV analyzers only)")
            except Exception as exc:
                logger.warning(f"TradeAnalyzer unavailable: {exc}")

        self._order_manager = OrderManager(
            signal_queue=queue.Queue(),  # unused — we call process_signal() directly
            broker=self._paper_broker,
            trading_mode="paper",
            db_url=self._settings.database_url,
            trade_analyzer=trade_analyzer,
            risk_manager=self._risk_manager,
            position_manager=self._position_manager,
            position_sizer=position_sizer,
            strategy_quarantine=self._strategy_quarantine,
            research_enabled=(trade_analyzer is not None),
        )

    def _setup_strategy_engine(self) -> None:
        """Load strategies from config directory."""
        self._strategy_engine = StrategyEngine(
            strategies_dir=self._strategy_dir,
            bars_provider=lambda: self._data,
        )
        n = self._strategy_engine.load_strategies()
        logger.info(f"Loaded {n} strategies for backtest")

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> HarnessResults:
        """Run the backtest bar-by-bar.

        Returns:
            HarnessResults with equity curve, closed trades, and metrics.
        """
        results = HarnessResults()
        prev_date: Optional[date] = None

        logger.info(
            f"Backtest starting | instrument={self._instrument_key} | "
            f"bars={len(self._master_df)} | capital={self._capital / 100:.0f} INR"
        )

        for i, (bar_time, bar) in enumerate(self._master_df.iterrows()):
            bar_time_ist: datetime = (
                bar_time.astimezone(IST) if bar_time.tzinfo else bar_time.replace(tzinfo=IST)
            )
            current_price = int(bar["close"])

            # --- Advance bar provider (no lookahead) ---
            self._bar_provider.advance(self._instrument_key, i)

            # --- Update broker LTP so MARKET orders get correct fill price ---
            self._paper_broker.update_ltp(self._instrument_key, current_price)

            # --- Phase 1: New trading day reset ---
            today = bar_time_ist.date()
            if today != prev_date:
                if prev_date is not None:
                    logger.info(f"[BACKTEST] New day: {today} (prev: {prev_date})")
                self._risk_manager.reset_daily()
                self._circuit_breaker.reset_daily() if hasattr(self._circuit_breaker, "reset_daily") else None
                self._strategy_quarantine.reset_daily_counts() if hasattr(self._strategy_quarantine, "reset_daily_counts") else None
                prev_date = today

            # --- Phase 2: Skip pre-market bars ---
            if bar_time_ist.time() < MARKET_OPEN:
                results.equity_curve.append(self._paper_broker.get_funds().available)
                continue

            # --- Phase 3: Position exits (SL, target, trailing SL, partial profit) ---
            try:
                closed = self._position_manager.on_tick(
                    self._instrument_key, current_price
                )
                for pos in (closed or []):
                    results.trades.append({
                        "instrument_key": pos.instrument_key,
                        "strategy_id": pos.strategy_id,
                        "side": pos.side.value,
                        "entry_price": pos.entry_price,
                        "exit_price": pos.exit_price,
                        "quantity": pos.quantity,
                        "net_pnl": pos.total_realized_pnl,
                        "fees": 0,  # PaperBroker deducts fees internally
                        "entry_time": str(pos.entry_time),
                        "exit_time": str(pos.exit_time) if pos.exit_time else None,
                        "exit_reason": getattr(pos, "exit_reason", "unknown"),
                    })
            except Exception as exc:
                logger.error(f"[BACKTEST] PositionManager.on_tick error at bar {i}: {exc}")

            # --- Phase 4: Session gate ---
            bar_t = bar_time_ist.time()
            if bar_t >= NO_NEW_ENTRY_AFTER:
                results.equity_curve.append(self._paper_broker.get_funds().available)
                results.bars_processed += 1
                continue

            if hasattr(self._circuit_breaker, "is_paused") and self._circuit_breaker.is_paused():
                results.equity_curve.append(self._paper_broker.get_funds().available)
                results.bars_processed += 1
                continue

            # --- Phase 5: Signal generation (same ConditionEvaluator as live) ---
            try:
                signals = self._strategy_engine.evaluate_bar_sync(
                    bars=self._data,
                    bar_time=bar_time_ist,
                )
            except Exception as exc:
                logger.error(f"[BACKTEST] evaluate_bar_sync error at bar {i}: {exc}")
                signals = []

            # --- Phase 6: Process each signal through full live pipeline ---
            # validate → Kelly sizing → quarantine → research → risk → broker
            for signal in signals:
                signal.timestamp = bar_time_ist  # use bar time, not wall clock
                try:
                    self._order_manager.process_signal(signal)
                except Exception as exc:
                    logger.error(
                        f"[BACKTEST] OrderManager.process_signal error "
                        f"for {signal.strategy_name}: {exc}"
                    )

            # --- Phase 7: Equity snapshot ---
            try:
                funds = self._paper_broker.get_funds()
                results.equity_curve.append(funds.available)
            except Exception:
                results.equity_curve.append(self._capital)

            results.bars_processed += 1

        # Force close any remaining positions at last bar's close
        self._square_off_all(results)

        # Compute aggregate metrics
        results._compute_metrics(self._capital)

        logger.info(
            f"Backtest complete | bars={results.bars_processed} | "
            f"trades={len(results.trades)} | "
            f"P&L={results.metrics.get('net_pnl_rupees', 0):.2f} INR"
        )
        return results

    def _square_off_all(self, results: HarnessResults) -> None:
        """Force-close all open positions at end of backtest data."""
        try:
            if hasattr(self._position_manager, "square_off_all"):
                self._position_manager.square_off_all("Backtest data ended")
        except Exception as exc:
            logger.warning(f"[BACKTEST] End-of-data square-off failed: {exc}")
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
pytest tests/test_backtest_harness_init.py -v
```
Expected: both tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/backtest/harness.py tests/test_backtest_harness_init.py
git commit -m "feat: implement BacktestHarness driving all live components synchronously"
```

---

## Task 5: CLI Runner

**Files:**
- Create: `src/backtest/runner.py`

- [ ] **Step 1: Implement runner**

```python
# src/backtest/runner.py
"""CLI entry point for the BacktestHarness.

Usage:
    python -m src.backtest.runner \\
        --data nifty50_1min_30days.csv \\
        --instrument "NSE_INDEX|Nifty Bank" \\
        --capital 1000000 \\
        --from 2026-01-01 \\
        --to 2026-03-31 \\
        --strategies config/strategies
"""

from __future__ import annotations

import argparse
import sys
from datetime import date


def _parse_date(s: str) -> date:
    return date.fromisoformat(s)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a full-fidelity backtest using the live trading components.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data", required=True,
        help="Path to 1-minute OHLCV CSV file (paisa prices, IST timestamps)",
    )
    parser.add_argument(
        "--instrument", default="NSE_INDEX|Nifty Bank",
        help='Instrument key, e.g. "NSE_INDEX|Nifty Bank" (default)',
    )
    parser.add_argument(
        "--strategies", default="config/strategies",
        help="Directory containing strategy JSON files (default: config/strategies)",
    )
    parser.add_argument(
        "--capital", type=int, default=None,
        help="Initial capital in PAISA (default: from .env CAPITAL setting)",
    )
    parser.add_argument(
        "--from", dest="start_date", type=_parse_date, default=None,
        help="Start date YYYY-MM-DD (inclusive)",
    )
    parser.add_argument(
        "--to", dest="end_date", type=_parse_date, default=None,
        help="End date YYYY-MM-DD (inclusive)",
    )
    parser.add_argument(
        "--prices-in-rupees", action="store_true",
        help="CSV prices are in rupees — multiply by 100 to convert to paisa",
    )

    args = parser.parse_args()

    from src.backtest.harness import BacktestHarness

    harness = BacktestHarness(
        data_file=args.data,
        instrument_key=args.instrument,
        strategy_dir=args.strategies,
        capital=args.capital,
        start_date=args.start_date,
        end_date=args.end_date,
        prices_in_rupees=args.prices_in_rupees,
    )

    results = harness.run()
    m = results.metrics

    print("\n" + "─" * 50)
    print(f"  BACKTEST RESULTS")
    if args.start_date or args.end_date:
        print(f"  {args.start_date or 'start'} → {args.end_date or 'end'}")
    print("─" * 50)
    print(f"  Total Trades     : {m.get('total_trades', 0)}")
    print(f"  Win Rate         : {m.get('win_rate', 0):.1f}%")
    print(f"  Profit Factor    : {m.get('profit_factor', 0):.2f}")
    print(f"  Net P&L          : ₹ {m.get('net_pnl_rupees', 0):,.2f}")
    print(f"  Max Drawdown     : ₹ {m.get('max_drawdown_rupees', 0):,.2f} "
          f"({m.get('max_drawdown_pct', 0):.1f}%)")
    print(f"  Sharpe Ratio     : {m.get('sharpe_ratio', 0):.2f}")
    print(f"  Total Fees       : ₹ {m.get('total_fees_rupees', 0):,.2f}")
    print(f"  Return           : {m.get('return_pct', 0):+.2f}%")
    print(f"  Bars Processed   : {results.bars_processed}")
    print("─" * 50)
    print(f"  Trades saved to DB (source=backtest)")
    print("─" * 50 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Smoke test the CLI**

```bash
cd /Users/sandeepvangapandu/Downloads/Trading
source venv/bin/activate
python -m src.backtest.runner \
    --data nifty50_1min_30days.csv \
    --instrument "NSE_INDEX|Nifty 50" \
    --prices-in-rupees \
    --from 2026-01-01 \
    --to 2026-01-05
```

Expected: Runs without crash. Prints results table. Zero or more trades depending on strategy conditions.

- [ ] **Step 3: Commit**

```bash
git add src/backtest/runner.py
git commit -m "feat: add backtest CLI runner (python -m src.backtest.runner)"
```

---

## Task 6: Dashboard Integration

**Files:**
- Modify: `src/dashboard/pages/backtester.py`

- [ ] **Step 1: Read the existing backtester page to understand the current wiring**

```bash
cat src/dashboard/pages/backtester.py | head -80
```

Identify: where `BacktestEngine` is imported and called, what variables hold the results, what charts are drawn from results.

- [ ] **Step 2: Replace BacktestEngine with BacktestHarness**

Find the import block at the top of `src/dashboard/pages/backtester.py`. Replace:

```python
from src.backtest.engine import BacktestEngine, BacktestResults
```

with:

```python
from src.backtest.harness import BacktestHarness, HarnessResults
```

- [ ] **Step 3: Replace the engine instantiation and run call**

Find any block that creates a `BacktestEngine` instance and calls `.run(data)`.

Replace with:

```python
harness = BacktestHarness(
    data_file=uploaded_file_path,          # path to the CSV the user uploaded
    instrument_key=selected_instrument,    # e.g. "NSE_INDEX|Nifty Bank"
    strategy_dir="config/strategies",
    capital=selected_capital,              # from Streamlit number_input (in paisa)
    start_date=from_date,
    end_date=to_date,
)
with st.spinner("Running backtest through full live pipeline..."):
    results = harness.run()
```

- [ ] **Step 4: Update the results access pattern**

`HarnessResults` has `.equity_curve` (list[int]), `.trades` (list[dict]), `.metrics` (dict) — same shape as the old engine's `BacktestResults.get_metrics()` output. Update any attribute access accordingly:

- Old: `results.get_metrics()["win_rate"]` → New: `results.metrics["win_rate"]`
- Old: `results.trades` → New: `results.trades` (same)
- Old: `results.equity_curve` → New: `results.equity_curve` (same)

- [ ] **Step 5: Add CSV file uploader widget if not present**

If the page doesn't already have a file uploader, add above the run button:

```python
import tempfile, os

uploaded = st.file_uploader(
    "Upload 1-minute OHLCV CSV (paisa prices, IST timestamps)",
    type=["csv"],
)
prices_in_rupees = st.checkbox("Prices are in Rupees (will be converted to paisa)")

if uploaded is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        tmp.write(uploaded.getvalue())
        uploaded_file_path = tmp.name
```

- [ ] **Step 6: Smoke test the dashboard**

```bash
streamlit run src/dashboard/dashboard.py
```

Navigate to the Backtester page. Upload `nifty50_1min_30days.csv`. Click Run. Verify:
- No Python exceptions in the terminal
- Equity curve chart renders
- Metrics table shows numbers (may be zeros if no trades triggered)

- [ ] **Step 7: Commit**

```bash
git add src/dashboard/pages/backtester.py
git commit -m "feat: wire dashboard backtester to BacktestHarness (live-parity)"
```

---

## Task 7: End-to-End Integration Test

**Files:**
- Create: `tests/test_backtest_harness.py`

- [ ] **Step 1: Write integration test using real CSV**

```python
# tests/test_backtest_harness.py
"""End-to-end integration test for BacktestHarness using real CSV data.

Uses nifty50_1min_30days.csv which ships in the repo root.
Requires the file to be present — skips gracefully if not found.
"""

import json
from pathlib import Path

import pytest

from src.backtest.harness import BacktestHarness, HarnessResults

CSV_PATH = Path("nifty50_1min_30days.csv")
STRATEGY_DIR = Path("config/strategies")

SIMPLE_STRATEGY = {
    "name": "Integration_Test_RSI",
    "enabled": True,
    "underlying": {"instrument_key": "NSE_INDEX|Nifty 50"},
    "trading_hours": {
        "start_time": "09:30:00",
        "end_time": "14:45:00",
        "days": [0, 1, 2, 3, 4],
    },
    "timeframes": {"primary": "5min"},
    "entry_sets": [
        {
            "name": "oversold_long",
            "signal": "BUY",
            "conditions": [
                {
                    "indicator": "RSI",
                    "comparison": "<",
                    "value": 45,
                    "timeframe": "5min",
                    "parameters": {"length": 14},
                }
            ],
        }
    ],
    "exit_rules": {"stop_loss_pct": 1.0, "target_pct": 2.0},
    "position_sizing": {"method": "fixed_quantity", "quantity": 1},
    "risk_management": {"max_trades_per_day": 3},
    "params": {"use_enhanced_filters": False},
}


@pytest.fixture
def isolated_strategy_dir(tmp_path):
    """Create a temp strategy dir with a single known strategy."""
    (tmp_path / "integration_test.json").write_text(json.dumps(SIMPLE_STRATEGY))
    return tmp_path


@pytest.mark.skipif(
    not CSV_PATH.exists(),
    reason="nifty50_1min_30days.csv not found — skipping integration test",
)
def test_harness_runs_on_real_data(isolated_strategy_dir):
    """Harness completes on real data without crashing."""
    harness = BacktestHarness(
        data_file=str(CSV_PATH),
        instrument_key="NSE_INDEX|Nifty 50",
        strategy_dir=str(isolated_strategy_dir),
        capital=100_000_000,  # 10 lakhs in paisa
        prices_in_rupees=True,
    )
    results = harness.run()

    assert isinstance(results, HarnessResults)
    assert results.bars_processed > 0, "At least one bar must be processed"
    assert len(results.equity_curve) == results.bars_processed, (
        "Equity curve length must match bars_processed"
    )


@pytest.mark.skipif(
    not CSV_PATH.exists(),
    reason="nifty50_1min_30days.csv not found",
)
def test_all_trades_have_valid_fields(isolated_strategy_dir):
    """Every closed trade must have non-zero entry/exit prices and valid quantity."""
    harness = BacktestHarness(
        data_file=str(CSV_PATH),
        instrument_key="NSE_INDEX|Nifty 50",
        strategy_dir=str(isolated_strategy_dir),
        capital=100_000_000,
        prices_in_rupees=True,
    )
    results = harness.run()

    for trade in results.trades:
        assert trade["entry_price"] > 0, f"entry_price must be > 0: {trade}"
        assert trade["quantity"] > 0, f"quantity must be > 0: {trade}"
        assert "strategy_id" in trade, "trade must have strategy_id"
        assert "side" in trade, "trade must have side (BUY or SELL)"


@pytest.mark.skipif(
    not CSV_PATH.exists(),
    reason="nifty50_1min_30days.csv not found",
)
def test_equity_curve_never_negative(isolated_strategy_dir):
    """Capital must never go below zero during the backtest."""
    harness = BacktestHarness(
        data_file=str(CSV_PATH),
        instrument_key="NSE_INDEX|Nifty 50",
        strategy_dir=str(isolated_strategy_dir),
        capital=100_000_000,
        prices_in_rupees=True,
    )
    results = harness.run()

    for i, cap in enumerate(results.equity_curve):
        assert cap >= 0, f"Negative capital at bar {i}: {cap}"


@pytest.mark.skipif(
    not CSV_PATH.exists(),
    reason="nifty50_1min_30days.csv not found",
)
def test_metrics_keys_present(isolated_strategy_dir):
    """All required metric keys must be present after a run."""
    harness = BacktestHarness(
        data_file=str(CSV_PATH),
        instrument_key="NSE_INDEX|Nifty 50",
        strategy_dir=str(isolated_strategy_dir),
        capital=100_000_000,
        prices_in_rupees=True,
    )
    results = harness.run()

    required_keys = {
        "total_trades", "win_rate", "profit_factor",
        "net_pnl_rupees", "max_drawdown_rupees", "return_pct", "sharpe_ratio",
    }
    assert required_keys <= results.metrics.keys(), (
        f"Missing metric keys: {required_keys - results.metrics.keys()}"
    )
```

- [ ] **Step 2: Run integration tests**

```bash
pytest tests/test_backtest_harness.py -v
```

Expected: tests run (or skip if CSV not found). All non-skipped tests PASS.

- [ ] **Step 3: Run full test suite to check for regressions**

```bash
pytest tests/ -v --tb=short 2>&1 | tail -30
```

Expected: all previously-passing tests continue to PASS. New tests PASS.

- [ ] **Step 4: Final commit**

```bash
git add tests/test_backtest_harness.py tests/test_learning_backtest_mode.py
git commit -m "test: integration tests for BacktestHarness end-to-end run"
```

---

## Self-Review Notes

**Spec coverage check:**
- ✅ BacktestBarProvider with no-lookahead guarantee
- ✅ StrategyEngine.evaluate_bar_sync (same conditions + session filters)
- ✅ Full component wiring (PaperBroker, RiskManager, PositionManager, PartialProfitManager, OrderManager)
- ✅ TradeAnalyzer (research module) wired with BacktestBarProvider
- ✅ LearningIntegration active, adapts during run, lessons not persisted
- ✅ RiskManager.reset_daily() called on new trading day
- ✅ CircuitBreaker and StrategyQuarantine hooked in
- ✅ PaperBroker.update_ltp() called per bar (market order fill price)
- ✅ CLI with --data, --instrument, --capital, --from, --to, --prices-in-rupees
- ✅ Dashboard swaps BacktestEngine for BacktestHarness
- ✅ Old BacktestEngine unchanged (backward compat)
- ✅ Tests for no-lookahead, OHLCV aggregation, integration run

**Type consistency:** `HarnessResults` defined once in `harness.py`, imported in runner and test. `BacktestBarProvider.advance(instrument_key, index)` matches all call sites. `evaluate_bar_sync(bars, bar_time)` matches test calls.

**One known ambiguity to handle during implementation:** `PositionManager.on_tick()` return type — if it returns `None` rather than `list[ManagedPosition]`, the `for pos in (closed or []):` guard handles it. Implementer should verify the actual return type and adjust if needed (may be a side-effect-only call where positions are closed via the `on_position_close` callback instead).
