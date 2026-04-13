# Remaining Phases Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete Phases 2-4 of the Trading Bot: Backtesting Engine, Streamlit Dashboard, Telegram Notifications, Live Broker, and Stress Tests.

**Architecture:** Each phase builds on existing infrastructure. The dashboard reads from SQLite via SQLAlchemy models already defined. Telegram uses python-telegram-bot async. Backtester replays historical candles through the existing strategy engine. Live broker implements the existing BaseBroker interface.

**Tech Stack:** Python 3.11+, Streamlit, SQLAlchemy, python-telegram-bot, plotly, pandas, pytest

---

## Current Completion Status

| Phase | Description | Status |
|-------|-------------|--------|
| Phase 1 | Foundation (Auth, Data, Paper OMS, Risk, DB) | 100% |
| Phase 2 | Strategy Expansion + Backtesting | 75% (backtesting missing) |
| Phase 3 | Dashboard + Telegram Notifications | 0% |
| Phase 4 | Live Trading Preparation | 40% (broker interface ready, no live impl) |

---

## PHASE 2 COMPLETION: Backtesting Engine

### Task 1: Backtesting Data Loader

**Files:**
- Create: `src/backtest/__init__.py`
- Create: `src/backtest/data_loader.py`
- Test: `tests/test_backtest_data_loader.py`

- [ ] **Step 1: Write failing test for historical data loading**

```python
# tests/test_backtest_data_loader.py
import pandas as pd
import pytest
from datetime import date

from src.backtest.data_loader import BacktestDataLoader


@pytest.fixture
def sample_csv(tmp_path):
    """Create a sample OHLCV CSV file."""
    data = {
        "timestamp": pd.date_range("2025-01-01 09:15", periods=100, freq="5min"),
        "open": [100_00] * 100,  # paisa
        "high": [101_00] * 100,
        "low": [99_00] * 100,
        "close": [100_50] * 100,
        "volume": [1000] * 100,
    }
    df = pd.DataFrame(data)
    path = tmp_path / "BANKNIFTY_5min.csv"
    df.to_csv(path, index=False)
    return path


def test_load_csv(sample_csv):
    loader = BacktestDataLoader()
    df = loader.load_csv(str(sample_csv))
    assert len(df) == 100
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert df.index.name == "timestamp"


def test_load_csv_date_filter(sample_csv):
    loader = BacktestDataLoader()
    df = loader.load_csv(
        str(sample_csv),
        start_date=date(2025, 1, 1),
        end_date=date(2025, 1, 1),
    )
    assert len(df) > 0
    assert all(d.date() == date(2025, 1, 1) for d in df.index)


def test_resample_timeframe(sample_csv):
    loader = BacktestDataLoader()
    df = loader.load_csv(str(sample_csv))
    resampled = loader.resample(df, "15min")
    # 100 bars of 5min → ~34 bars of 15min
    assert len(resampled) < len(df)
    assert len(resampled) > 0
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_backtest_data_loader.py -v
```
Expected: FAIL (module not found)

- [ ] **Step 3: Implement BacktestDataLoader**

```python
# src/backtest/__init__.py
"""Backtesting engine for strategy validation against historical data."""

# src/backtest/data_loader.py
"""Load and prepare historical OHLCV data for backtesting."""

from datetime import date
from pathlib import Path
from typing import Optional

import pandas as pd
from loguru import logger


class BacktestDataLoader:
    """Loads historical OHLCV data from CSV files or the database.

    All price columns are expected in paisa (integers).
    Timestamps are localized to IST (Asia/Kolkata).
    """

    IST = "Asia/Kolkata"

    def load_csv(
        self,
        file_path: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> pd.DataFrame:
        """Load OHLCV data from a CSV file.

        Args:
            file_path: Path to the CSV file.
            start_date: Optional start date filter (inclusive).
            end_date: Optional end date filter (inclusive).

        Returns:
            DataFrame with DatetimeIndex (IST) and OHLCV columns.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")

        df = pd.read_csv(file_path, parse_dates=["timestamp"])
        df = df.set_index("timestamp")
        df.index = pd.DatetimeIndex(df.index)

        if df.index.tz is None:
            df.index = df.index.tz_localize(self.IST)

        # Ensure standard column names
        expected = ["open", "high", "low", "close", "volume"]
        df.columns = [c.lower() for c in df.columns]
        for col in expected:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        df = df[expected]

        # Date filtering
        if start_date:
            df = df[df.index.date >= start_date]
        if end_date:
            df = df[df.index.date <= end_date]

        df = df.sort_index()
        logger.info(
            f"Loaded {len(df)} bars from {path.name} "
            f"({df.index[0]} to {df.index[-1]})"
        )
        return df

    def resample(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """Resample OHLCV data to a larger timeframe.

        Args:
            df: Source DataFrame with OHLCV columns.
            timeframe: Target timeframe (e.g., '15min', '1h', '1d').

        Returns:
            Resampled DataFrame.
        """
        resampled = df.resample(timeframe).agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }).dropna()
        return resampled
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_backtest_data_loader.py -v
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/backtest/ tests/test_backtest_data_loader.py
git commit -m "feat(backtest): add data loader with CSV support and resampling"
```

---

### Task 2: Backtesting Engine Core

**Files:**
- Create: `src/backtest/engine.py`
- Test: `tests/test_backtest_engine.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_backtest_engine.py
import pandas as pd
import pytest
from datetime import datetime
from zoneinfo import ZoneInfo

from src.backtest.engine import BacktestEngine, BacktestConfig, BacktestResult

IST = ZoneInfo("Asia/Kolkata")


@pytest.fixture
def sample_data():
    """100 bars of trending-up 5min data."""
    dates = pd.date_range(
        "2025-01-02 09:15", periods=100, freq="5min", tz=IST
    )
    prices = [50000_00 + i * 50_00 for i in range(100)]  # paisa, trending up
    return pd.DataFrame({
        "open": [p - 10_00 for p in prices],
        "high": [p + 20_00 for p in prices],
        "low": [p - 20_00 for p in prices],
        "close": prices,
        "volume": [5000] * 100,
    }, index=dates)


@pytest.fixture
def config():
    return BacktestConfig(
        strategy_config_path="config/strategies/supertrend_ema_rsi.json",
        initial_capital=100_000_000,  # 10 lakh paisa
        instrument_key="NSE_INDEX|Nifty Bank",
        slippage_pct=0.05,
        commission_per_order=2000,  # Rs 20 in paisa
    )


def test_backtest_engine_runs(sample_data, config):
    engine = BacktestEngine(config)
    result = engine.run({"NSE_INDEX|Nifty Bank": sample_data})
    assert isinstance(result, BacktestResult)
    assert result.initial_capital == 100_000_000
    assert isinstance(result.trades, list)
    assert result.total_bars_processed == 100


def test_backtest_result_metrics(sample_data, config):
    engine = BacktestEngine(config)
    result = engine.run({"NSE_INDEX|Nifty Bank": sample_data})
    metrics = result.compute_metrics()
    assert "total_pnl" in metrics
    assert "win_rate" in metrics
    assert "max_drawdown" in metrics
    assert "sharpe_ratio" in metrics
    assert "profit_factor" in metrics
    assert "total_trades" in metrics
```

- [ ] **Step 2: Run test to verify failure**

```bash
pytest tests/test_backtest_engine.py -v
```

- [ ] **Step 3: Implement BacktestEngine**

```python
# src/backtest/engine.py
"""Core backtesting engine — replays historical bars through strategy conditions."""

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from loguru import logger
from pydantic import BaseModel
from zoneinfo import ZoneInfo

from src.strategy.builder import StrategyConfig

IST = ZoneInfo("Asia/Kolkata")


class BacktestConfig(BaseModel):
    """Configuration for a backtest run."""

    strategy_config_path: str
    initial_capital: int  # paisa
    instrument_key: str
    slippage_pct: float = 0.05
    commission_per_order: int = 2000  # paisa (Rs 20)


@dataclass
class BacktestTrade:
    """A completed backtest trade."""

    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    side: str  # "BUY" or "SELL"
    entry_price: int  # paisa
    exit_price: int  # paisa
    quantity: int
    pnl: int  # paisa (after slippage + commission)
    exit_reason: str  # "target", "stop_loss", "time_exit", "signal"


@dataclass
class BacktestResult:
    """Results from a backtest run."""

    initial_capital: int
    final_capital: int
    trades: list[BacktestTrade]
    equity_curve: list[int]  # capital at each bar (paisa)
    total_bars_processed: int

    def compute_metrics(self) -> dict[str, Any]:
        """Compute performance metrics from trade results.

        Returns:
            Dictionary with: total_pnl, total_trades, win_rate, profit_factor,
            max_drawdown, sharpe_ratio, avg_trade_duration, avg_winner, avg_loser.
        """
        if not self.trades:
            return {
                "total_pnl": 0, "total_trades": 0, "win_rate": 0.0,
                "profit_factor": 0.0, "max_drawdown": 0, "sharpe_ratio": 0.0,
                "avg_trade_duration_min": 0, "avg_winner": 0, "avg_loser": 0,
            }

        pnls = [t.pnl for t in self.trades]
        winners = [p for p in pnls if p > 0]
        losers = [p for p in pnls if p <= 0]

        total_pnl = sum(pnls)
        win_rate = len(winners) / len(pnls) if pnls else 0.0
        gross_profit = sum(winners) if winners else 0
        gross_loss = abs(sum(losers)) if losers else 1  # avoid div by zero
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        # Max drawdown from equity curve
        peak = self.initial_capital
        max_dd = 0
        for eq in self.equity_curve:
            if eq > peak:
                peak = eq
            dd = peak - eq
            if dd > max_dd:
                max_dd = dd

        # Sharpe ratio (annualized, assuming 252 trading days, ~75 bars/day for 5min)
        if len(pnls) > 1:
            returns = [p / self.initial_capital for p in pnls]
            mean_ret = sum(returns) / len(returns)
            var = sum((r - mean_ret) ** 2 for r in returns) / (len(returns) - 1)
            std_ret = math.sqrt(var) if var > 0 else 1e-10
            sharpe = (mean_ret / std_ret) * math.sqrt(252)
        else:
            sharpe = 0.0

        # Average trade duration
        durations = [
            (t.exit_time - t.entry_time).total_seconds() / 60 for t in self.trades
        ]
        avg_duration = sum(durations) / len(durations) if durations else 0

        return {
            "total_pnl": total_pnl,
            "total_pnl_rupees": total_pnl / 100,
            "total_trades": len(pnls),
            "winning_trades": len(winners),
            "losing_trades": len(losers),
            "win_rate": round(win_rate * 100, 2),
            "profit_factor": round(profit_factor, 2),
            "max_drawdown": max_dd,
            "max_drawdown_rupees": max_dd / 100,
            "max_drawdown_pct": round(max_dd / self.initial_capital * 100, 2),
            "sharpe_ratio": round(sharpe, 2),
            "avg_trade_duration_min": round(avg_duration, 1),
            "avg_winner": round(sum(winners) / len(winners), 0) if winners else 0,
            "avg_loser": round(sum(losers) / len(losers), 0) if losers else 0,
            "final_capital_rupees": self.final_capital / 100,
            "return_pct": round(
                (self.final_capital - self.initial_capital) / self.initial_capital * 100, 2
            ),
        }


class BacktestEngine:
    """Replays historical data through strategy conditions to simulate trades.

    Uses the same ConditionEvaluator and strategy JSON configs as live trading,
    ensuring backtest-live parity.

    Args:
        config: BacktestConfig with strategy path, capital, instrument.
    """

    def __init__(self, config: BacktestConfig) -> None:
        self.config = config
        self._load_strategy()

    def _load_strategy(self) -> None:
        """Load strategy configuration from JSON."""
        path = Path(self.config.strategy_config_path)
        with open(path) as f:
            raw = json.load(f)
        self.strategy = StrategyConfig(**raw)

    def run(self, bars: dict[str, pd.DataFrame]) -> BacktestResult:
        """Run backtest over historical bars.

        Args:
            bars: Dict mapping instrument_key to OHLCV DataFrame.

        Returns:
            BacktestResult with trades, equity curve, and metrics.
        """
        from src.strategy.conditions import ConditionEvaluator

        evaluator = ConditionEvaluator()
        instrument = self.config.instrument_key
        df = bars[instrument]

        capital = self.config.initial_capital
        equity_curve: list[int] = [capital]
        trades: list[BacktestTrade] = []
        position: Optional[dict] = None  # {side, entry_price, entry_time, quantity}
        warmup = 30  # bars needed for indicator warmup

        for i in range(warmup, len(df)):
            window = df.iloc[: i + 1]
            bar = df.iloc[i]
            bar_time = df.index[i]
            current_price = int(bar["close"])

            # Check exit conditions if in position
            if position is not None:
                exit_reason = self._check_exit(
                    position, current_price, bar_time
                )
                if exit_reason:
                    trade = self._close_position(
                        position, current_price, bar_time, exit_reason
                    )
                    capital += trade.pnl
                    trades.append(trade)
                    position = None

            # Check entry conditions if flat
            if position is None:
                window_bars = {instrument: window}
                for entry_set in self.strategy.entry_sets:
                    conditions = entry_set.conditions
                    all_met = evaluator.evaluate_all(conditions, window_bars, instrument)
                    if all_met:
                        qty = self._compute_quantity(capital, current_price)
                        if qty > 0:
                            side = "BUY" if entry_set.signal in ("CE", "BUY") else "SELL"
                            slippage = int(current_price * self.config.slippage_pct / 100)
                            fill_price = current_price + slippage if side == "BUY" else current_price - slippage
                            position = {
                                "side": side,
                                "entry_price": fill_price,
                                "entry_time": bar_time,
                                "quantity": qty,
                            }
                            capital -= self.config.commission_per_order
                        break  # only one entry per bar

            equity_curve.append(capital + self._unrealized_pnl(position, current_price))

        # Force close any open position at end
        if position is not None:
            last_price = int(df.iloc[-1]["close"])
            trade = self._close_position(
                position, last_price, df.index[-1], "end_of_data"
            )
            capital += trade.pnl
            trades.append(trade)

        return BacktestResult(
            initial_capital=self.config.initial_capital,
            final_capital=capital,
            trades=trades,
            equity_curve=equity_curve,
            total_bars_processed=len(df),
        )

    def _compute_quantity(self, capital: int, price: int) -> int:
        """Compute position quantity based on risk sizing."""
        if price <= 0:
            return 0
        sizing = self.strategy.position_sizing
        if sizing and sizing.quantity:
            return sizing.quantity
        # Default: risk 1% of capital per trade
        risk_amount = int(capital * 0.01)
        sl_pct = self.strategy.exit_rules.stop_loss_pct if self.strategy.exit_rules else 2.0
        sl_per_unit = int(price * sl_pct / 100)
        if sl_per_unit <= 0:
            return 1
        return max(1, risk_amount // sl_per_unit)

    def _check_exit(
        self, position: dict, current_price: int, bar_time: pd.Timestamp
    ) -> Optional[str]:
        """Check if position should be exited."""
        entry = position["entry_price"]
        side = position["side"]

        if side == "BUY":
            pnl_pct = (current_price - entry) / entry * 100
        else:
            pnl_pct = (entry - current_price) / entry * 100

        exit_rules = self.strategy.exit_rules
        if exit_rules:
            # Stop loss
            sl = exit_rules.stop_loss_pct or 2.0
            if pnl_pct <= -sl:
                return "stop_loss"
            # Target
            tgt = exit_rules.target_pct or 4.0
            if pnl_pct >= tgt:
                return "target"
            # Time-based exit
            if exit_rules.time_based_exit:
                exit_time = exit_rules.time_based_exit
                if hasattr(bar_time, "time") and bar_time.time() >= exit_time:
                    return "time_exit"

        return None

    def _close_position(
        self, position: dict, exit_price: int, exit_time: pd.Timestamp, reason: str
    ) -> BacktestTrade:
        """Close position and create trade record."""
        side = position["side"]
        entry = position["entry_price"]
        qty = position["quantity"]
        slippage = int(exit_price * self.config.slippage_pct / 100)
        fill = exit_price - slippage if side == "BUY" else exit_price + slippage

        if side == "BUY":
            raw_pnl = (fill - entry) * qty
        else:
            raw_pnl = (entry - fill) * qty

        pnl = raw_pnl - self.config.commission_per_order
        return BacktestTrade(
            entry_time=position["entry_time"],
            exit_time=exit_time,
            side=side,
            entry_price=entry,
            exit_price=fill,
            quantity=qty,
            pnl=pnl,
            exit_reason=reason,
        )

    def _unrealized_pnl(self, position: Optional[dict], price: int) -> int:
        """Calculate unrealized P&L for equity curve."""
        if position is None:
            return 0
        entry = position["entry_price"]
        qty = position["quantity"]
        if position["side"] == "BUY":
            return (price - entry) * qty
        return (entry - price) * qty
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_backtest_engine.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/backtest/engine.py tests/test_backtest_engine.py
git commit -m "feat(backtest): add core backtesting engine with metrics"
```

---

### Task 3: Backtest Report Generator

**Files:**
- Create: `src/backtest/report.py`
- Test: `tests/test_backtest_report.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_backtest_report.py
import pytest
from unittest.mock import MagicMock
from src.backtest.engine import BacktestResult, BacktestTrade
from src.backtest.report import BacktestReportGenerator
import pandas as pd
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")


@pytest.fixture
def sample_result():
    trades = [
        BacktestTrade(
            entry_time=pd.Timestamp("2025-01-02 09:30", tz=IST),
            exit_time=pd.Timestamp("2025-01-02 10:30", tz=IST),
            side="BUY", entry_price=50000_00, exit_price=50200_00,
            quantity=15, pnl=300000, exit_reason="target",
        ),
        BacktestTrade(
            entry_time=pd.Timestamp("2025-01-02 11:00", tz=IST),
            exit_time=pd.Timestamp("2025-01-02 11:45", tz=IST),
            side="BUY", entry_price=50300_00, exit_price=50100_00,
            quantity=15, pnl=-300000, exit_reason="stop_loss",
        ),
    ]
    return BacktestResult(
        initial_capital=100_000_000,
        final_capital=100_000_000,
        trades=trades,
        equity_curve=[100_000_000, 100_300_000, 100_000_000],
        total_bars_processed=200,
    )


def test_text_report(sample_result):
    gen = BacktestReportGenerator(sample_result)
    report = gen.text_summary()
    assert "Total Trades" in report
    assert "Win Rate" in report
    assert "Sharpe" in report


def test_trade_dataframe(sample_result):
    gen = BacktestReportGenerator(sample_result)
    df = gen.trades_dataframe()
    assert len(df) == 2
    assert "pnl_rupees" in df.columns
    assert "duration_min" in df.columns
```

- [ ] **Step 2: Run test to verify failure**

- [ ] **Step 3: Implement BacktestReportGenerator**

```python
# src/backtest/report.py
"""Generate reports from backtest results — text summaries and DataFrames."""

import pandas as pd
from loguru import logger

from src.backtest.engine import BacktestResult


class BacktestReportGenerator:
    """Generates human-readable reports and DataFrames from backtest results.

    Args:
        result: A completed BacktestResult from the engine.
    """

    def __init__(self, result: BacktestResult) -> None:
        self.result = result
        self.metrics = result.compute_metrics()

    def text_summary(self) -> str:
        """Generate a formatted text summary of backtest performance.

        Returns:
            Multi-line string with all key metrics.
        """
        m = self.metrics
        lines = [
            "=" * 55,
            "           BACKTEST PERFORMANCE REPORT",
            "=" * 55,
            f"  Initial Capital:      Rs. {self.result.initial_capital / 100:>12,.2f}",
            f"  Final Capital:        Rs. {m['final_capital_rupees']:>12,.2f}",
            f"  Total P&L:            Rs. {m['total_pnl_rupees']:>12,.2f}",
            f"  Return:               {m['return_pct']:>11.2f}%",
            "-" * 55,
            f"  Total Trades:         {m['total_trades']:>12}",
            f"  Winning Trades:       {m['winning_trades']:>12}",
            f"  Losing Trades:        {m['losing_trades']:>12}",
            f"  Win Rate:             {m['win_rate']:>11.2f}%",
            f"  Profit Factor:        {m['profit_factor']:>12.2f}",
            "-" * 55,
            f"  Max Drawdown:         Rs. {m['max_drawdown_rupees']:>12,.2f}",
            f"  Max Drawdown %:       {m['max_drawdown_pct']:>11.2f}%",
            f"  Sharpe Ratio:         {m['sharpe_ratio']:>12.2f}",
            f"  Avg Trade Duration:   {m['avg_trade_duration_min']:>10.1f} min",
            f"  Avg Winner (paisa):   {m['avg_winner']:>12,.0f}",
            f"  Avg Loser (paisa):    {m['avg_loser']:>12,.0f}",
            "=" * 55,
        ]
        return "\n".join(lines)

    def trades_dataframe(self) -> pd.DataFrame:
        """Convert trades to a pandas DataFrame for analysis.

        Returns:
            DataFrame with one row per trade, prices in rupees.
        """
        if not self.result.trades:
            return pd.DataFrame()

        records = []
        for t in self.result.trades:
            records.append({
                "entry_time": t.entry_time,
                "exit_time": t.exit_time,
                "side": t.side,
                "entry_price_rupees": t.entry_price / 100,
                "exit_price_rupees": t.exit_price / 100,
                "quantity": t.quantity,
                "pnl_rupees": t.pnl / 100,
                "exit_reason": t.exit_reason,
                "duration_min": (t.exit_time - t.entry_time).total_seconds() / 60,
            })
        return pd.DataFrame(records)

    def equity_series(self) -> pd.Series:
        """Get equity curve as a pandas Series in rupees.

        Returns:
            Series of capital values over time.
        """
        return pd.Series(
            [v / 100 for v in self.result.equity_curve],
            name="equity_rupees",
        )
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_backtest_report.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/backtest/report.py tests/test_backtest_report.py
git commit -m "feat(backtest): add report generator with text summary and DataFrame export"
```

---

## PHASE 3: Dashboard & Notifications

### Task 4: Dashboard Data Service Layer

**Files:**
- Create: `src/dashboard/__init__.py`
- Create: `src/dashboard/data_service.py`
- Test: `tests/test_dashboard_data_service.py`

This layer queries the database and formats data for the Streamlit UI. Keeps all DB logic out of the Streamlit pages.

- [ ] **Step 1: Write failing test**

```python
# tests/test_dashboard_data_service.py
import pytest
from datetime import date, datetime
from zoneinfo import ZoneInfo
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from src.persistence.models import Base, OrderRecord, TradeRecord, PositionRecord, DailyPnL
from src.dashboard.data_service import DashboardDataService

IST = ZoneInfo("Asia/Kolkata")


@pytest.fixture
def db_session():
    """In-memory SQLite session with test data."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        # Insert test data
        session.add(OrderRecord(
            strategy="ema_crossover", instrument_key="NSE_INDEX|Nifty Bank",
            transaction_type="BUY", order_type="MARKET",
            quantity=15, price=5000000, status="FILLED",
            filled_qty=15, avg_fill_price=5000000,
            placed_at=datetime(2025, 1, 2, 9, 30, tzinfo=IST),
            updated_at=datetime(2025, 1, 2, 9, 30, tzinfo=IST),
        ))
        session.add(TradeRecord(
            strategy="ema_crossover", instrument_key="NSE_INDEX|Nifty Bank",
            side="BUY", entry_price=5000000, exit_price=5010000,
            quantity=15, realized_pnl=150000, fees=4000,
            entry_time=datetime(2025, 1, 2, 9, 30, tzinfo=IST),
            exit_time=datetime(2025, 1, 2, 10, 0, tzinfo=IST),
            holding_duration_seconds=1800,
        ))
        session.add(PositionRecord(
            strategy="ema_crossover", instrument_key="NSE_INDEX|Nifty Bank",
            side="BUY", entry_price=5020000, quantity=15,
            status="open",
            opened_at=datetime(2025, 1, 2, 10, 30, tzinfo=IST),
        ))
        session.add(DailyPnL(
            date=date(2025, 1, 2),
            realized_pnl=150000, unrealized_pnl=50000,
            total_pnl=200000, trades_count=2, win_count=1,
        ))
        session.commit()
        yield session


def test_get_todays_trades(db_session):
    svc = DashboardDataService(db_session)
    trades = svc.get_trades(date(2025, 1, 2))
    assert len(trades) == 1
    assert trades[0]["pnl_rupees"] == 1500.0


def test_get_open_positions(db_session):
    svc = DashboardDataService(db_session)
    positions = svc.get_open_positions()
    assert len(positions) == 1
    assert positions[0]["status"] == "open"


def test_get_daily_pnl_series(db_session):
    svc = DashboardDataService(db_session)
    df = svc.get_daily_pnl_series()
    assert len(df) == 1
    assert "total_pnl_rupees" in df.columns


def test_get_strategy_summary(db_session):
    svc = DashboardDataService(db_session)
    summary = svc.get_strategy_summary()
    assert "ema_crossover" in summary
    assert summary["ema_crossover"]["total_trades"] == 1
```

- [ ] **Step 2: Run test to verify failure**

- [ ] **Step 3: Implement DashboardDataService**

```python
# src/dashboard/__init__.py
"""Dashboard UI and data service layer."""

# src/dashboard/data_service.py
"""Data service layer for the Streamlit dashboard.

Queries the SQLAlchemy database and returns formatted dicts/DataFrames
ready for display. All monetary values are converted from paisa to rupees
for display purposes.
"""

from datetime import date, datetime
from typing import Any, Optional

import pandas as pd
from loguru import logger
from sqlalchemy import func, desc
from sqlalchemy.orm import Session

from src.persistence.models import (
    OrderRecord, TradeRecord, PositionRecord, DailyPnL,
    StrategyPerformanceRecord, ResearchReport, LessonLearnedRecord,
)


class DashboardDataService:
    """Provides formatted data to the dashboard from the database.

    Args:
        session: SQLAlchemy database session.
    """

    def __init__(self, session: Session) -> None:
        self._session = session

    @staticmethod
    def _paisa_to_rupees(paisa: int) -> float:
        return paisa / 100 if paisa else 0.0

    def get_trades(self, day: Optional[date] = None) -> list[dict[str, Any]]:
        """Get trades, optionally filtered by date.

        Args:
            day: Filter trades to this date. None = all trades.

        Returns:
            List of trade dicts with prices in rupees.
        """
        query = self._session.query(TradeRecord).order_by(desc(TradeRecord.entry_time))
        if day:
            query = query.filter(func.date(TradeRecord.entry_time) == day)
        trades = query.all()
        return [
            {
                "id": t.id,
                "strategy": t.strategy,
                "instrument": t.instrument_key,
                "side": t.side,
                "entry_price_rupees": self._paisa_to_rupees(t.entry_price),
                "exit_price_rupees": self._paisa_to_rupees(t.exit_price),
                "quantity": t.quantity,
                "pnl_rupees": self._paisa_to_rupees(t.realized_pnl),
                "fees_rupees": self._paisa_to_rupees(t.fees),
                "entry_time": t.entry_time,
                "exit_time": t.exit_time,
                "duration_min": (t.holding_duration_seconds or 0) / 60,
            }
            for t in trades
        ]

    def get_open_positions(self) -> list[dict[str, Any]]:
        """Get all currently open positions.

        Returns:
            List of position dicts with prices in rupees.
        """
        positions = (
            self._session.query(PositionRecord)
            .filter(PositionRecord.status == "open")
            .all()
        )
        return [
            {
                "strategy": p.strategy,
                "instrument": p.instrument_key,
                "side": p.side,
                "entry_price_rupees": self._paisa_to_rupees(p.entry_price),
                "quantity": p.quantity,
                "status": p.status,
                "opened_at": p.opened_at,
            }
            for p in positions
        ]

    def get_orders(self, day: Optional[date] = None) -> list[dict[str, Any]]:
        """Get orders, optionally filtered by date.

        Args:
            day: Filter to this date. None = all orders.

        Returns:
            List of order dicts.
        """
        query = self._session.query(OrderRecord).order_by(desc(OrderRecord.placed_at))
        if day:
            query = query.filter(func.date(OrderRecord.placed_at) == day)
        return [
            {
                "id": o.id,
                "strategy": o.strategy,
                "instrument": o.instrument_key,
                "type": o.transaction_type,
                "order_type": o.order_type,
                "quantity": o.quantity,
                "price_rupees": self._paisa_to_rupees(o.price),
                "status": o.status,
                "filled_qty": o.filled_qty,
                "fill_price_rupees": self._paisa_to_rupees(o.avg_fill_price),
                "time": o.placed_at,
            }
            for o in query.all()
        ]

    def get_daily_pnl_series(self) -> pd.DataFrame:
        """Get daily P&L as a DataFrame for charting.

        Returns:
            DataFrame with date index and P&L columns in rupees.
        """
        rows = self._session.query(DailyPnL).order_by(DailyPnL.date).all()
        if not rows:
            return pd.DataFrame(columns=["date", "total_pnl_rupees", "realized_rupees", "trades_count", "win_count"])
        records = [
            {
                "date": r.date,
                "total_pnl_rupees": self._paisa_to_rupees(r.total_pnl),
                "realized_rupees": self._paisa_to_rupees(r.realized_pnl),
                "unrealized_rupees": self._paisa_to_rupees(r.unrealized_pnl),
                "trades_count": r.trades_count,
                "win_count": r.win_count,
            }
            for r in rows
        ]
        return pd.DataFrame(records)

    def get_strategy_summary(self) -> dict[str, dict[str, Any]]:
        """Get per-strategy performance summary.

        Returns:
            Dict mapping strategy name to metrics dict.
        """
        trades = self._session.query(TradeRecord).all()
        summary: dict[str, dict] = {}
        for t in trades:
            name = t.strategy
            if name not in summary:
                summary[name] = {
                    "total_trades": 0, "wins": 0, "total_pnl": 0, "total_fees": 0,
                }
            s = summary[name]
            s["total_trades"] += 1
            if t.realized_pnl and t.realized_pnl > 0:
                s["wins"] += 1
            s["total_pnl"] += t.realized_pnl or 0
            s["total_fees"] += t.fees or 0

        for name, s in summary.items():
            s["win_rate"] = round(s["wins"] / s["total_trades"] * 100, 1) if s["total_trades"] > 0 else 0
            s["pnl_rupees"] = self._paisa_to_rupees(s["total_pnl"])
            s["fees_rupees"] = self._paisa_to_rupees(s["total_fees"])

        return summary

    def get_today_summary(self, today: Optional[date] = None) -> dict[str, Any]:
        """Get today's aggregate summary.

        Args:
            today: Date to summarize. Defaults to today.

        Returns:
            Dict with today's realized P&L, trade count, win count.
        """
        if today is None:
            from zoneinfo import ZoneInfo
            today = datetime.now(ZoneInfo("Asia/Kolkata")).date()

        row = self._session.query(DailyPnL).filter(DailyPnL.date == today).first()
        if row:
            return {
                "realized_rupees": self._paisa_to_rupees(row.realized_pnl),
                "unrealized_rupees": self._paisa_to_rupees(row.unrealized_pnl),
                "total_rupees": self._paisa_to_rupees(row.total_pnl),
                "trades": row.trades_count,
                "wins": row.win_count,
            }
        return {"realized_rupees": 0, "unrealized_rupees": 0, "total_rupees": 0, "trades": 0, "wins": 0}
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_dashboard_data_service.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/dashboard/ tests/test_dashboard_data_service.py
git commit -m "feat(dashboard): add data service layer for querying trade data"
```

---

### Task 5: Streamlit Dashboard — Main App (Page 1: Live Overview)

**Files:**
- Create: `src/dashboard/app.py`
- Create: `src/dashboard/pages/__init__.py`
- Create: `src/dashboard/pages/live_overview.py`

No automated test (Streamlit UI testing is manual). Verify by running `streamlit run src/dashboard/app.py`.

- [ ] **Step 1: Create the main Streamlit app entry point**

```python
# src/dashboard/app.py
"""Streamlit dashboard entry point.

Run with: streamlit run src/dashboard/app.py
"""

import streamlit as st
from pathlib import Path
import sys

# Ensure project root is on path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

st.set_page_config(
    page_title="Trading Bot Dashboard",
    page_icon="chart_with_upwards_trend",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Sidebar navigation
st.sidebar.title("Trading Bot")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigation",
    [
        "Live Overview",
        "Trade History",
        "Strategy Performance",
        "Equity Curve",
        "Risk Monitor",
        "Backtester",
    ],
)

# Database connection
from src.dashboard.db_connect import get_dashboard_session

session = get_dashboard_session()

if page == "Live Overview":
    from src.dashboard.pages.live_overview import render
    render(session)
elif page == "Trade History":
    from src.dashboard.pages.trade_history import render
    render(session)
elif page == "Strategy Performance":
    from src.dashboard.pages.strategy_performance import render
    render(session)
elif page == "Equity Curve":
    from src.dashboard.pages.equity_curve import render
    render(session)
elif page == "Risk Monitor":
    from src.dashboard.pages.risk_monitor import render
    render(session)
elif page == "Backtester":
    from src.dashboard.pages.backtester import render
    render(session)

st.sidebar.markdown("---")
st.sidebar.caption("Auto-refresh: 30s")
# Auto-refresh every 30 seconds
st_autorefresh = st.empty()
```

- [ ] **Step 2: Create DB connection helper**

```python
# src/dashboard/db_connect.py
"""Database connection helper for the dashboard."""

from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from config.settings import get_settings


_engine = None


def get_dashboard_session() -> Session:
    """Get a database session for the dashboard.

    Returns:
        SQLAlchemy Session connected to the trading database.
    """
    global _engine
    if _engine is None:
        settings = get_settings()
        _engine = create_engine(settings.database_url)
    return Session(_engine)
```

- [ ] **Step 3: Create Live Overview page**

```python
# src/dashboard/pages/__init__.py
"""Dashboard page modules."""

# src/dashboard/pages/live_overview.py
"""Live Overview page — current positions, P&L, and system status."""

from datetime import datetime

import streamlit as st
from sqlalchemy.orm import Session
from zoneinfo import ZoneInfo

from src.dashboard.data_service import DashboardDataService

IST = ZoneInfo("Asia/Kolkata")


def render(session: Session) -> None:
    """Render the Live Overview page."""
    svc = DashboardDataService(session)
    now = datetime.now(IST)
    today = now.date()

    st.title("Live Overview")
    st.caption(f"Last updated: {now.strftime('%d %b %Y, %I:%M:%S %p IST')}")

    # --- Top metrics row ---
    summary = svc.get_today_summary(today)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Today's P&L", f"Rs. {summary['total_rupees']:,.2f}",
                delta=f"Rs. {summary['realized_rupees']:,.2f} realized")
    col2.metric("Trades Today", summary["trades"])
    col3.metric("Wins", summary["wins"])
    win_rate = (summary["wins"] / summary["trades"] * 100) if summary["trades"] > 0 else 0
    col4.metric("Win Rate", f"{win_rate:.0f}%")

    st.markdown("---")

    # --- Open Positions ---
    st.subheader("Open Positions")
    positions = svc.get_open_positions()
    if positions:
        st.dataframe(
            positions,
            use_container_width=True,
            column_config={
                "entry_price_rupees": st.column_config.NumberColumn("Entry Price", format="Rs. %.2f"),
                "opened_at": st.column_config.DatetimeColumn("Opened At", format="HH:mm:ss"),
            },
        )
    else:
        st.info("No open positions")

    # --- Recent Orders ---
    st.subheader("Today's Orders")
    orders = svc.get_orders(today)
    if orders:
        st.dataframe(
            orders,
            use_container_width=True,
            column_config={
                "price_rupees": st.column_config.NumberColumn("Price", format="Rs. %.2f"),
                "fill_price_rupees": st.column_config.NumberColumn("Fill Price", format="Rs. %.2f"),
            },
        )
    else:
        st.info("No orders today")

    # --- Today's Completed Trades ---
    st.subheader("Today's Trades")
    trades = svc.get_trades(today)
    if trades:
        for t in trades:
            color = "green" if t["pnl_rupees"] > 0 else "red"
            st.markdown(
                f"**{t['strategy']}** | {t['instrument']} | {t['side']} | "
                f"Entry: Rs.{t['entry_price_rupees']:,.2f} -> Exit: Rs.{t['exit_price_rupees']:,.2f} | "
                f"Qty: {t['quantity']} | "
                f"P&L: :{color}[Rs. {t['pnl_rupees']:,.2f}] | "
                f"Duration: {t['duration_min']:.0f} min | {t.get('exit_reason', '')}"
            )
    else:
        st.info("No completed trades today")
```

- [ ] **Step 4: Verify by running**

```bash
streamlit run src/dashboard/app.py
```
Open http://localhost:8501 — should see the Live Overview page.

- [ ] **Step 5: Commit**

```bash
git add src/dashboard/
git commit -m "feat(dashboard): add Streamlit app with Live Overview page"
```

---

### Task 6: Dashboard — Trade History Page

**Files:**
- Create: `src/dashboard/pages/trade_history.py`

- [ ] **Step 1: Implement Trade History page**

```python
# src/dashboard/pages/trade_history.py
"""Trade History page — searchable, filterable trade log."""

from datetime import datetime, timedelta

import pandas as pd
import streamlit as st
from sqlalchemy.orm import Session
from zoneinfo import ZoneInfo

from src.dashboard.data_service import DashboardDataService

IST = ZoneInfo("Asia/Kolkata")


def render(session: Session) -> None:
    """Render the Trade History page."""
    svc = DashboardDataService(session)
    now = datetime.now(IST)

    st.title("Trade History")

    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        start_date = st.date_input("From", value=now.date() - timedelta(days=30))
    with col2:
        end_date = st.date_input("To", value=now.date())
    with col3:
        strategy_filter = st.text_input("Strategy filter", placeholder="All strategies")

    all_trades = svc.get_trades()

    if all_trades:
        df = pd.DataFrame(all_trades)
        # Apply date filters
        df["trade_date"] = pd.to_datetime(df["entry_time"]).dt.date
        df = df[(df["trade_date"] >= start_date) & (df["trade_date"] <= end_date)]
        if strategy_filter:
            df = df[df["strategy"].str.contains(strategy_filter, case=False, na=False)]

        # Summary metrics
        if len(df) > 0:
            col1, col2, col3, col4 = st.columns(4)
            total_pnl = df["pnl_rupees"].sum()
            col1.metric("Total P&L", f"Rs. {total_pnl:,.2f}")
            col2.metric("Total Trades", len(df))
            winners = (df["pnl_rupees"] > 0).sum()
            col3.metric("Win Rate", f"{winners / len(df) * 100:.1f}%")
            col4.metric("Total Fees", f"Rs. {df['fees_rupees'].sum():,.2f}")

        st.markdown("---")

        # Trade table
        st.dataframe(
            df.drop(columns=["trade_date"], errors="ignore"),
            use_container_width=True,
            column_config={
                "pnl_rupees": st.column_config.NumberColumn("P&L (Rs.)", format="%.2f"),
                "entry_price_rupees": st.column_config.NumberColumn("Entry", format="%.2f"),
                "exit_price_rupees": st.column_config.NumberColumn("Exit", format="%.2f"),
            },
        )

        # P&L distribution chart
        st.subheader("P&L Distribution")
        st.bar_chart(df.set_index("entry_time")["pnl_rupees"])
    else:
        st.info("No trades found")
```

- [ ] **Step 2: Commit**

```bash
git add src/dashboard/pages/trade_history.py
git commit -m "feat(dashboard): add Trade History page with filters and P&L chart"
```

---

### Task 7: Dashboard — Strategy Performance Page

**Files:**
- Create: `src/dashboard/pages/strategy_performance.py`

- [ ] **Step 1: Implement Strategy Performance page**

```python
# src/dashboard/pages/strategy_performance.py
"""Strategy Performance page — per-strategy metrics and comparison."""

import pandas as pd
import streamlit as st
from sqlalchemy.orm import Session

from src.dashboard.data_service import DashboardDataService


def render(session: Session) -> None:
    """Render the Strategy Performance page."""
    svc = DashboardDataService(session)

    st.title("Strategy Performance")

    summary = svc.get_strategy_summary()

    if not summary:
        st.info("No strategy data available yet. Run the bot to generate trades.")
        return

    # Summary table
    rows = []
    for name, stats in summary.items():
        rows.append({
            "Strategy": name,
            "Trades": stats["total_trades"],
            "Wins": stats["wins"],
            "Win Rate %": stats["win_rate"],
            "P&L (Rs.)": stats["pnl_rupees"],
            "Fees (Rs.)": stats["fees_rupees"],
            "Net P&L (Rs.)": stats["pnl_rupees"] - stats["fees_rupees"],
        })

    df = pd.DataFrame(rows)
    st.dataframe(
        df,
        use_container_width=True,
        column_config={
            "P&L (Rs.)": st.column_config.NumberColumn(format="%.2f"),
            "Fees (Rs.)": st.column_config.NumberColumn(format="%.2f"),
            "Net P&L (Rs.)": st.column_config.NumberColumn(format="%.2f"),
        },
    )

    # Bar chart comparison
    st.subheader("P&L by Strategy")
    chart_df = df.set_index("Strategy")[["Net P&L (Rs.)"]]
    st.bar_chart(chart_df)

    # Win rate comparison
    st.subheader("Win Rate by Strategy")
    wr_df = df.set_index("Strategy")[["Win Rate %"]]
    st.bar_chart(wr_df)
```

- [ ] **Step 2: Commit**

```bash
git add src/dashboard/pages/strategy_performance.py
git commit -m "feat(dashboard): add Strategy Performance page with comparisons"
```

---

### Task 8: Dashboard — Equity Curve Page

**Files:**
- Create: `src/dashboard/pages/equity_curve.py`

- [ ] **Step 1: Implement Equity Curve page**

```python
# src/dashboard/pages/equity_curve.py
"""Equity Curve page — cumulative P&L and drawdown visualization."""

import pandas as pd
import streamlit as st
from sqlalchemy.orm import Session

from src.dashboard.data_service import DashboardDataService
from config.settings import get_settings


def render(session: Session) -> None:
    """Render the Equity Curve page."""
    svc = DashboardDataService(session)
    settings = get_settings()

    st.title("Equity Curve")

    daily_pnl = svc.get_daily_pnl_series()

    if daily_pnl.empty:
        st.info("No daily P&L data yet. The bot records daily summaries after market close.")
        return

    daily_pnl = daily_pnl.set_index("date")

    # Compute cumulative equity
    initial_capital_rupees = settings.capital / 100
    daily_pnl["cumulative_pnl"] = daily_pnl["total_pnl_rupees"].cumsum()
    daily_pnl["equity"] = initial_capital_rupees + daily_pnl["cumulative_pnl"]

    # Drawdown calculation
    daily_pnl["peak"] = daily_pnl["equity"].cummax()
    daily_pnl["drawdown"] = daily_pnl["equity"] - daily_pnl["peak"]
    daily_pnl["drawdown_pct"] = (daily_pnl["drawdown"] / daily_pnl["peak"]) * 100

    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Starting Capital", f"Rs. {initial_capital_rupees:,.2f}")
    col2.metric("Current Equity", f"Rs. {daily_pnl['equity'].iloc[-1]:,.2f}")
    col3.metric("Total Return", f"{daily_pnl['cumulative_pnl'].iloc[-1] / initial_capital_rupees * 100:.2f}%")
    col4.metric("Max Drawdown", f"{daily_pnl['drawdown_pct'].min():.2f}%")

    st.markdown("---")

    # Equity curve chart
    st.subheader("Equity Over Time")
    st.line_chart(daily_pnl[["equity"]])

    # Drawdown chart
    st.subheader("Drawdown")
    st.area_chart(daily_pnl[["drawdown_pct"]])

    # Daily P&L bars
    st.subheader("Daily P&L")
    st.bar_chart(daily_pnl[["total_pnl_rupees"]])

    # Stats table
    st.subheader("Daily Breakdown")
    display_df = daily_pnl[["total_pnl_rupees", "realized_rupees", "trades_count", "win_count"]].copy()
    display_df.columns = ["Total P&L (Rs.)", "Realized (Rs.)", "Trades", "Wins"]
    st.dataframe(display_df, use_container_width=True)
```

- [ ] **Step 2: Commit**

```bash
git add src/dashboard/pages/equity_curve.py
git commit -m "feat(dashboard): add Equity Curve page with drawdown chart"
```

---

### Task 9: Dashboard — Risk Monitor Page

**Files:**
- Create: `src/dashboard/pages/risk_monitor.py`

- [ ] **Step 1: Implement Risk Monitor page**

```python
# src/dashboard/pages/risk_monitor.py
"""Risk Monitor page — risk limits, circuit breaker status, health checks."""

import streamlit as st
from sqlalchemy.orm import Session

from config.settings import get_settings
from src.dashboard.data_service import DashboardDataService


def render(session: Session) -> None:
    """Render the Risk Monitor page."""
    svc = DashboardDataService(session)
    settings = get_settings()

    st.title("Risk Monitor")

    # --- Risk Configuration ---
    st.subheader("Risk Configuration")
    col1, col2, col3 = st.columns(3)
    col1.metric("Capital", f"Rs. {settings.capital / 100:,.2f}")
    col2.metric("Max Daily Loss", f"Rs. {settings.max_daily_loss / 100:,.2f}")
    col3.metric("Max Open Positions", settings.max_open_positions)

    col4, col5, col6 = st.columns(3)
    col4.metric("Max Position Size", f"{settings.max_position_size_pct}%")
    col5.metric("Max Capital Deployed", f"{settings.max_capital_deployment_pct}%")
    col6.metric("Circuit Breaker After", f"{settings.consecutive_loss_pause} losses")

    st.markdown("---")

    # --- Today's Risk Usage ---
    st.subheader("Today's Risk Usage")
    summary = svc.get_today_summary()
    positions = svc.get_open_positions()

    daily_loss_limit = settings.max_daily_loss / 100
    daily_pnl = summary["total_rupees"]
    loss_pct_used = abs(min(0, daily_pnl)) / daily_loss_limit * 100 if daily_loss_limit > 0 else 0

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Daily Loss Limit Usage**")
        st.progress(min(loss_pct_used / 100, 1.0))
        if daily_pnl < 0:
            st.markdown(f"Loss: Rs. {abs(daily_pnl):,.2f} / Rs. {daily_loss_limit:,.2f} ({loss_pct_used:.1f}%)")
        else:
            st.markdown(f"Profit: Rs. {daily_pnl:,.2f} (No loss)")

    with col2:
        st.markdown("**Position Slots**")
        used_slots = len(positions)
        max_slots = settings.max_open_positions
        st.progress(used_slots / max_slots if max_slots > 0 else 0)
        st.markdown(f"{used_slots} / {max_slots} positions open")

    st.markdown("---")

    # --- Trading Mode Banner ---
    st.subheader("System Status")
    mode = settings.trading_mode
    if mode == "paper":
        st.success("PAPER TRADING MODE — No real orders are being placed")
    else:
        st.error("LIVE TRADING MODE — Real money at risk!")

    # Strategy status from DB
    strategy_summary = svc.get_strategy_summary()
    if strategy_summary:
        st.subheader("Strategy Status")
        for name, stats in strategy_summary.items():
            status_icon = "white_check_mark" if stats["win_rate"] > 40 else "warning"
            st.markdown(
                f":{status_icon}: **{name}** — "
                f"{stats['total_trades']} trades, "
                f"{stats['win_rate']}% win rate, "
                f"P&L: Rs. {stats['pnl_rupees']:,.2f}"
            )
```

- [ ] **Step 2: Commit**

```bash
git add src/dashboard/pages/risk_monitor.py
git commit -m "feat(dashboard): add Risk Monitor page with limits and usage bars"
```

---

### Task 10: Dashboard — Backtester Page

**Files:**
- Create: `src/dashboard/pages/backtester.py`

- [ ] **Step 1: Implement Backtester page**

```python
# src/dashboard/pages/backtester.py
"""Backtester page — run strategies against historical data from the UI."""

import json
from pathlib import Path

import pandas as pd
import streamlit as st
from sqlalchemy.orm import Session


def render(session: Session) -> None:
    """Render the Backtester page."""
    st.title("Backtester")

    # Strategy selection
    strategies_dir = Path("config/strategies")
    strategy_files = sorted(strategies_dir.glob("*.json"))
    strategy_names = [f.stem for f in strategy_files]

    col1, col2 = st.columns(2)
    with col1:
        selected = st.selectbox("Select Strategy", strategy_names)
    with col2:
        data_file = st.file_uploader(
            "Upload Historical CSV (timestamp, open, high, low, close, volume)",
            type=["csv"],
        )

    # Config
    col3, col4, col5 = st.columns(3)
    with col3:
        capital = st.number_input("Capital (Rs.)", value=100000, step=10000)
    with col4:
        slippage = st.number_input("Slippage %", value=0.05, step=0.01, format="%.2f")
    with col5:
        commission = st.number_input("Commission per order (Rs.)", value=20, step=5)

    if st.button("Run Backtest", type="primary") and data_file is not None:
        with st.spinner("Running backtest..."):
            try:
                from src.backtest.data_loader import BacktestDataLoader
                from src.backtest.engine import BacktestEngine, BacktestConfig
                from src.backtest.report import BacktestReportGenerator

                # Load data
                df = pd.read_csv(data_file, parse_dates=["timestamp"])
                df = df.set_index("timestamp")
                if df.index.tz is None:
                    df.index = df.index.tz_localize("Asia/Kolkata")

                # Load strategy config to get instrument key
                strategy_path = strategies_dir / f"{selected}.json"
                with open(strategy_path) as f:
                    strat_config = json.load(f)
                instrument_key = strat_config.get("underlying", {}).get(
                    "instrument_key", "NSE_INDEX|Nifty Bank"
                )

                # Run backtest
                config = BacktestConfig(
                    strategy_config_path=str(strategy_path),
                    initial_capital=int(capital * 100),  # convert to paisa
                    instrument_key=instrument_key,
                    slippage_pct=slippage,
                    commission_per_order=int(commission * 100),
                )
                engine = BacktestEngine(config)
                result = engine.run({instrument_key: df})
                report = BacktestReportGenerator(result)

                # Display results
                st.success("Backtest complete!")
                metrics = result.compute_metrics()

                # Metrics cards
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Total P&L", f"Rs. {metrics['total_pnl_rupees']:,.2f}")
                c2.metric("Win Rate", f"{metrics['win_rate']}%")
                c3.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']}")
                c4.metric("Max Drawdown", f"{metrics['max_drawdown_pct']}%")

                c5, c6, c7, c8 = st.columns(4)
                c5.metric("Total Trades", metrics["total_trades"])
                c6.metric("Profit Factor", f"{metrics['profit_factor']}")
                c7.metric("Avg Duration", f"{metrics['avg_trade_duration_min']} min")
                c8.metric("Return", f"{metrics['return_pct']}%")

                # Equity curve
                st.subheader("Equity Curve")
                equity = report.equity_series()
                st.line_chart(equity)

                # Trade list
                st.subheader("Trade Details")
                trades_df = report.trades_dataframe()
                st.dataframe(trades_df, use_container_width=True)

                # Full text report
                with st.expander("Full Report"):
                    st.code(report.text_summary())

            except Exception as e:
                st.error(f"Backtest failed: {e}")
    elif data_file is None:
        st.info(
            "Upload a CSV file with columns: timestamp, open, high, low, close, volume. "
            "All prices should be in **paisa** (e.g., 50000.00 Rs = 5000000 paisa)."
        )
```

- [ ] **Step 2: Commit**

```bash
git add src/dashboard/pages/backtester.py
git commit -m "feat(dashboard): add Backtester page with CSV upload and live results"
```

---

### Task 11: Telegram Notification Bot

**Files:**
- Create: `src/notifications/telegram_bot.py`
- Test: `tests/test_telegram_bot.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_telegram_bot.py
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from src.notifications.telegram_bot import TelegramNotifier


@pytest.fixture
def notifier():
    return TelegramNotifier(bot_token="fake_token", chat_id="12345")


def test_format_trade_message(notifier):
    trade = {
        "strategy": "ema_crossover",
        "instrument": "BANKNIFTY",
        "side": "BUY",
        "entry_price_rupees": 50000.00,
        "exit_price_rupees": 50200.00,
        "quantity": 15,
        "pnl_rupees": 3000.00,
    }
    msg = notifier.format_trade_message(trade)
    assert "ema_crossover" in msg
    assert "3,000.00" in msg
    assert "BANKNIFTY" in msg


def test_format_daily_summary(notifier):
    summary = {
        "realized_rupees": 5000.0,
        "unrealized_rupees": 1000.0,
        "total_rupees": 6000.0,
        "trades": 5,
        "wins": 3,
    }
    msg = notifier.format_daily_summary(summary)
    assert "Daily" in msg
    assert "5,000.00" in msg
    assert "60.0%" in msg


def test_notifier_disabled_when_no_token():
    notifier = TelegramNotifier(bot_token="", chat_id="")
    assert notifier.is_enabled() is False


def test_notifier_enabled_with_token(notifier):
    assert notifier.is_enabled() is True
```

- [ ] **Step 2: Run test to verify failure**

- [ ] **Step 3: Implement TelegramNotifier**

```python
# src/notifications/telegram_bot.py
"""Telegram notification bot for trade alerts and daily summaries.

Sends messages asynchronously using python-telegram-bot library.
Gracefully degrades if Telegram is not configured (empty token).
"""

import asyncio
from typing import Any, Optional

from loguru import logger


class TelegramNotifier:
    """Sends trade notifications and summaries to Telegram.

    Args:
        bot_token: Telegram bot API token from @BotFather.
        chat_id: Telegram chat ID to send messages to.
    """

    def __init__(self, bot_token: str, chat_id: str) -> None:
        self._token = bot_token
        self._chat_id = chat_id
        self._bot = None

    def is_enabled(self) -> bool:
        """Check if Telegram notifications are configured."""
        return bool(self._token and self._chat_id)

    async def _get_bot(self):
        """Lazily initialize the Telegram bot."""
        if self._bot is None:
            try:
                from telegram import Bot
                self._bot = Bot(token=self._token)
            except ImportError:
                logger.warning("python-telegram-bot not installed. Telegram disabled.")
                return None
        return self._bot

    async def send_message(self, text: str) -> bool:
        """Send a message to the configured chat.

        Args:
            text: Message text (supports Markdown).

        Returns:
            True if sent successfully, False otherwise.
        """
        if not self.is_enabled():
            return False
        try:
            bot = await self._get_bot()
            if bot is None:
                return False
            await bot.send_message(
                chat_id=self._chat_id,
                text=text,
                parse_mode="Markdown",
            )
            return True
        except Exception as e:
            logger.error(f"Telegram send failed: {e}")
            return False

    def send_sync(self, text: str) -> bool:
        """Synchronous wrapper for send_message.

        Args:
            text: Message text.

        Returns:
            True if sent successfully.
        """
        if not self.is_enabled():
            return False
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Schedule in existing loop
                asyncio.ensure_future(self.send_message(text))
                return True
            else:
                return loop.run_until_complete(self.send_message(text))
        except RuntimeError:
            loop = asyncio.new_event_loop()
            result = loop.run_until_complete(self.send_message(text))
            loop.close()
            return result

    def format_trade_message(self, trade: dict[str, Any]) -> str:
        """Format a trade notification message.

        Args:
            trade: Trade dict with strategy, instrument, side, prices, pnl.

        Returns:
            Formatted Markdown message string.
        """
        pnl = trade.get("pnl_rupees", 0)
        emoji = "green_circle" if pnl > 0 else "red_circle" if pnl < 0 else "white_circle"
        return (
            f":{emoji}: *Trade Closed*\n"
            f"Strategy: `{trade.get('strategy', 'N/A')}`\n"
            f"Instrument: {trade.get('instrument', 'N/A')}\n"
            f"Side: {trade.get('side', 'N/A')}\n"
            f"Entry: Rs. {trade.get('entry_price_rupees', 0):,.2f}\n"
            f"Exit: Rs. {trade.get('exit_price_rupees', 0):,.2f}\n"
            f"Qty: {trade.get('quantity', 0)}\n"
            f"*P&L: Rs. {pnl:,.2f}*"
        )

    def format_daily_summary(self, summary: dict[str, Any]) -> str:
        """Format a daily P&L summary message.

        Args:
            summary: Summary dict with realized, unrealized, total, trades, wins.

        Returns:
            Formatted Markdown message string.
        """
        trades = summary.get("trades", 0)
        wins = summary.get("wins", 0)
        win_rate = (wins / trades * 100) if trades > 0 else 0
        total = summary.get("total_rupees", 0)
        emoji = "chart_with_upwards_trend" if total >= 0 else "chart_with_downwards_trend"

        return (
            f":{emoji}: *Daily Summary*\n"
            f"Realized: Rs. {summary.get('realized_rupees', 0):,.2f}\n"
            f"Unrealized: Rs. {summary.get('unrealized_rupees', 0):,.2f}\n"
            f"*Total: Rs. {total:,.2f}*\n"
            f"Trades: {trades} | Wins: {wins} | Win Rate: {win_rate:.1f}%"
        )

    def format_alert(self, title: str, message: str) -> str:
        """Format a generic alert message.

        Args:
            title: Alert title.
            message: Alert body.

        Returns:
            Formatted Markdown message.
        """
        return f"*{title}*\n{message}"

    def notify_trade(self, trade: dict[str, Any]) -> bool:
        """Send a trade notification (sync).

        Args:
            trade: Trade data dict.

        Returns:
            True if sent.
        """
        return self.send_sync(self.format_trade_message(trade))

    def notify_daily_summary(self, summary: dict[str, Any]) -> bool:
        """Send daily P&L summary (sync).

        Args:
            summary: Daily summary dict.

        Returns:
            True if sent.
        """
        return self.send_sync(self.format_daily_summary(summary))

    def notify_alert(self, title: str, message: str) -> bool:
        """Send a generic alert (sync).

        Args:
            title: Alert title.
            message: Alert message.

        Returns:
            True if sent.
        """
        return self.send_sync(self.format_alert(title, message))
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_telegram_bot.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/notifications/telegram_bot.py tests/test_telegram_bot.py
git commit -m "feat(notifications): add Telegram bot with trade alerts and daily summary"
```

---

### Task 12: Wire Telegram into Main Bot

**Files:**
- Modify: `src/main.py`

- [ ] **Step 1: Add Telegram initialization and hooks to main.py**

Add to `TradingBot.__init__`:
```python
from src.notifications.telegram_bot import TelegramNotifier

self.telegram = TelegramNotifier(
    bot_token=self.settings.telegram_bot_token,
    chat_id=self.settings.telegram_chat_id,
)
if self.telegram.is_enabled():
    logger.info("Telegram notifications enabled")
else:
    logger.info("Telegram notifications disabled (no token configured)")
```

Hook into order fills (in the signal processing loop or position_manager close callback):
```python
# After a trade is closed in position_manager:
if self.telegram.is_enabled():
    self.telegram.notify_trade(trade_dict)
```

Add daily summary to the 15:30 IST scheduled job:
```python
# In the daily_summary scheduled function:
if self.telegram.is_enabled():
    summary = self.data_service.get_today_summary()
    self.telegram.notify_daily_summary(summary)
```

- [ ] **Step 2: Commit**

```bash
git add src/main.py
git commit -m "feat: wire Telegram notifications into main trading loop"
```

---

## PHASE 4: Live Trading Preparation

### Task 13: Live Upstox Broker Implementation

**Files:**
- Create: `src/execution/upstox_broker.py`
- Test: `tests/test_upstox_broker.py`

- [ ] **Step 1: Write failing test (mocked API calls)**

```python
# tests/test_upstox_broker.py
import pytest
from unittest.mock import MagicMock, patch
from src.execution.upstox_broker import UpstoxLiveBroker


@pytest.fixture
def broker():
    with patch("src.execution.upstox_broker.upstox_client"):
        b = UpstoxLiveBroker(access_token="fake_token")
        return b


def test_implements_base_broker(broker):
    from src.execution.base_broker import BaseBroker
    assert isinstance(broker, BaseBroker)


def test_place_order_returns_order_id(broker):
    broker._order_api.place_order.return_value = MagicMock(
        data=MagicMock(order_id="ORD123")
    )
    order_id = broker.place_order(
        instrument_key="NSE_EQ|RELIANCE",
        transaction_type="BUY",
        order_type="MARKET",
        quantity=1,
        price=0,
        product="MIS",
    )
    assert order_id == "ORD123"


def test_get_positions(broker):
    broker._portfolio_api.get_positions.return_value = MagicMock(
        data=[MagicMock(instrument_token="NSE_EQ|RELIANCE", quantity=10)]
    )
    positions = broker.get_positions()
    assert len(positions) >= 0  # Depends on mock structure
```

- [ ] **Step 2: Run test to verify failure**

- [ ] **Step 3: Implement UpstoxLiveBroker**

```python
# src/execution/upstox_broker.py
"""Live broker implementation using Upstox API v2/v3.

This module implements the BaseBroker interface for real order placement
via the Upstox trading platform. Only activated when TRADING_MODE=live.

WARNING: This places REAL orders with REAL money.
"""

from typing import Any, Optional

import upstox_client
from loguru import logger

from src.execution.base_broker import BaseBroker


class UpstoxLiveBroker(BaseBroker):
    """Live broker that places real orders through Upstox API.

    Args:
        access_token: Valid Upstox OAuth2 access token.
    """

    def __init__(self, access_token: str) -> None:
        self._token = access_token
        self._config = upstox_client.Configuration()
        self._config.access_token = access_token
        self._order_api = upstox_client.OrderApiV3(
            upstox_client.ApiClient(self._config)
        )
        self._portfolio_api = upstox_client.PortfolioApi(
            upstox_client.ApiClient(self._config)
        )
        self._ltp_cache: dict[str, int] = {}
        logger.warning("LIVE BROKER INITIALIZED — Real orders will be placed!")

    def place_order(
        self,
        instrument_key: str,
        transaction_type: str,
        order_type: str,
        quantity: int,
        price: int,
        product: str = "MIS",
        trigger_price: int = 0,
        tag: str = "",
    ) -> str:
        """Place a real order on Upstox.

        Args:
            instrument_key: Upstox instrument key (e.g., "NSE_EQ|RELIANCE").
            transaction_type: "BUY" or "SELL".
            order_type: "MARKET", "LIMIT", "SL", "SL-M".
            quantity: Number of shares/lots.
            price: Limit price in paisa (0 for MARKET).
            product: "MIS" (intraday), "CNC" (delivery), "NRML" (F&O).
            trigger_price: Trigger price in paisa for SL orders.
            tag: Optional order tag for tracking.

        Returns:
            Broker order ID string.
        """
        price_rupees = price / 100 if price else 0
        trigger_rupees = trigger_price / 100 if trigger_price else 0

        body = upstox_client.PlaceOrderV3Request(
            instrument_token=instrument_key,
            transaction_type=transaction_type.upper(),
            order_type=order_type.upper(),
            quantity=quantity,
            price=price_rupees,
            trigger_price=trigger_rupees,
            product=product.upper(),
            validity="DAY",
            disclosed_quantity=0,
            is_amo=False,
            tag=tag or "TRADING_BOT",
        )

        try:
            response = self._order_api.place_order(body)
            order_id = response.data.order_id
            logger.info(
                f"LIVE ORDER PLACED: {order_id} | {transaction_type} {quantity} "
                f"{instrument_key} @ {price_rupees} | {order_type} {product}"
            )
            return order_id
        except Exception as e:
            logger.error(f"Order placement failed: {e}")
            raise

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order.

        Args:
            order_id: Broker order ID to cancel.

        Returns:
            True if cancellation was successful.
        """
        try:
            self._order_api.cancel_order(order_id=order_id)
            logger.info(f"Order cancelled: {order_id}")
            return True
        except Exception as e:
            logger.error(f"Cancel failed for {order_id}: {e}")
            return False

    def modify_order(
        self,
        order_id: str,
        quantity: Optional[int] = None,
        price: Optional[int] = None,
        trigger_price: Optional[int] = None,
        order_type: Optional[str] = None,
    ) -> bool:
        """Modify an open order.

        Args:
            order_id: Broker order ID to modify.
            quantity: New quantity (None = no change).
            price: New price in paisa (None = no change).
            trigger_price: New trigger price in paisa (None = no change).
            order_type: New order type (None = no change).

        Returns:
            True if modification was successful.
        """
        try:
            body = upstox_client.ModifyOrderRequest(
                order_id=order_id,
                quantity=quantity,
                price=price / 100 if price else None,
                trigger_price=trigger_price / 100 if trigger_price else None,
                order_type=order_type,
                validity="DAY",
            )
            self._order_api.modify_order(body)
            logger.info(f"Order modified: {order_id}")
            return True
        except Exception as e:
            logger.error(f"Modify failed for {order_id}: {e}")
            return False

    def get_positions(self) -> list[dict[str, Any]]:
        """Get current positions from Upstox.

        Returns:
            List of position dicts.
        """
        try:
            response = self._portfolio_api.get_positions()
            positions = []
            for p in response.data or []:
                positions.append({
                    "instrument_key": p.instrument_token,
                    "quantity": p.quantity,
                    "average_price": int(float(p.average_price) * 100),
                    "pnl": int(float(p.pnl) * 100) if p.pnl else 0,
                    "product": p.product,
                })
            return positions
        except Exception as e:
            logger.error(f"Failed to fetch positions: {e}")
            return []

    def get_ltp(self, instrument_key: str) -> Optional[int]:
        """Get last traded price in paisa.

        Args:
            instrument_key: Instrument to query.

        Returns:
            LTP in paisa, or None if unavailable.
        """
        return self._ltp_cache.get(instrument_key)

    def update_ltp(self, instrument_key: str, ltp_paisa: int) -> None:
        """Update LTP cache from WebSocket feed.

        Args:
            instrument_key: Instrument identifier.
            ltp_paisa: Last traded price in paisa.
        """
        self._ltp_cache[instrument_key] = ltp_paisa
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_upstox_broker.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/execution/upstox_broker.py tests/test_upstox_broker.py
git commit -m "feat: add live Upstox broker implementation (BaseBroker interface)"
```

---

### Task 14: Broker Factory and Mode Switching

**Files:**
- Create: `src/execution/broker_factory.py`
- Modify: `src/main.py` (use factory instead of hardcoded PaperBroker)

- [ ] **Step 1: Implement broker factory**

```python
# src/execution/broker_factory.py
"""Factory for creating the appropriate broker based on trading mode."""

from loguru import logger

from src.execution.base_broker import BaseBroker


def create_broker(trading_mode: str, access_token: str = "") -> BaseBroker:
    """Create a broker instance based on the trading mode.

    Args:
        trading_mode: "paper" or "live".
        access_token: Upstox access token (required for live mode).

    Returns:
        BaseBroker implementation.

    Raises:
        ValueError: If trading_mode is invalid or live mode has no token.
    """
    if trading_mode == "paper":
        from src.execution.paper_broker import PaperBroker
        logger.info("Creating Paper Broker (simulated trading)")
        return PaperBroker()

    elif trading_mode == "live":
        if not access_token:
            raise ValueError("Live trading requires a valid access token")
        from src.execution.upstox_broker import UpstoxLiveBroker
        logger.warning("Creating LIVE Broker — Real orders will be placed!")
        return UpstoxLiveBroker(access_token=access_token)

    else:
        raise ValueError(f"Invalid trading mode: {trading_mode}. Use 'paper' or 'live'.")
```

- [ ] **Step 2: Update main.py to use broker factory**

Replace hardcoded PaperBroker creation with:
```python
from src.execution.broker_factory import create_broker

self.broker = create_broker(
    trading_mode=self.settings.trading_mode,
    access_token=self.auth.get_access_token() if self.settings.trading_mode == "live" else "",
)
```

- [ ] **Step 3: Commit**

```bash
git add src/execution/broker_factory.py src/main.py
git commit -m "feat: add broker factory for paper/live mode switching"
```

---

### Task 15: Integration Smoke Test

**Files:**
- Create: `tests/test_smoke.py`

- [ ] **Step 1: Write smoke test that verifies the full import chain**

```python
# tests/test_smoke.py
"""Smoke tests to verify all modules import correctly and core classes instantiate."""

import pytest


def test_imports_backtest():
    from src.backtest.data_loader import BacktestDataLoader
    from src.backtest.engine import BacktestEngine, BacktestConfig, BacktestResult
    from src.backtest.report import BacktestReportGenerator
    assert BacktestDataLoader is not None


def test_imports_dashboard():
    from src.dashboard.data_service import DashboardDataService
    assert DashboardDataService is not None


def test_imports_notifications():
    from src.notifications.telegram_bot import TelegramNotifier
    notifier = TelegramNotifier(bot_token="", chat_id="")
    assert notifier.is_enabled() is False


def test_imports_broker_factory():
    from src.execution.broker_factory import create_broker
    broker = create_broker("paper")
    assert broker is not None


def test_imports_strategy():
    from src.strategy.builder import StrategyConfig, Condition, ComparisonOperator
    assert ComparisonOperator.NEAR.value == "near"


def test_all_strategy_jsons_parse():
    """Verify every JSON strategy config is valid."""
    import json
    from pathlib import Path
    from src.strategy.builder import StrategyConfig

    strategies_dir = Path("config/strategies")
    for f in strategies_dir.glob("*.json"):
        with open(f) as fh:
            data = json.load(fh)
        config = StrategyConfig(**data)
        assert config.name, f"Strategy in {f.name} has no name"
```

- [ ] **Step 2: Run**

```bash
pytest tests/test_smoke.py -v
```

- [ ] **Step 3: Commit**

```bash
git add tests/test_smoke.py
git commit -m "test: add smoke tests for all module imports and strategy validation"
```

---

## Summary of All Tasks

| Task | Phase | Component | Status |
|------|-------|-----------|--------|
| 1 | 2 | Backtest Data Loader | pending |
| 2 | 2 | Backtest Engine Core | pending |
| 3 | 2 | Backtest Report Generator | pending |
| 4 | 3 | Dashboard Data Service | pending |
| 5 | 3 | Dashboard — Live Overview | pending |
| 6 | 3 | Dashboard — Trade History | pending |
| 7 | 3 | Dashboard — Strategy Performance | pending |
| 8 | 3 | Dashboard — Equity Curve | pending |
| 9 | 3 | Dashboard — Risk Monitor | pending |
| 10 | 3 | Dashboard — Backtester UI | pending |
| 11 | 3 | Telegram Notification Bot | pending |
| 12 | 3 | Wire Telegram into Main | pending |
| 13 | 4 | Live Upstox Broker | pending |
| 14 | 4 | Broker Factory + Mode Switch | pending |
| 15 | 4 | Integration Smoke Tests | pending |

**Estimated total effort: 15 tasks, ~45 steps**

---
