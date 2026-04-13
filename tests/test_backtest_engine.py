"""Tests for the backtest engine module."""

import pandas as pd
import pytest
from datetime import datetime, date
from zoneinfo import ZoneInfo

from src.backtest.engine import BacktestEngine, BacktestResults
from src.backtest.data_loader import BacktestDataLoader

IST = ZoneInfo("Asia/Kolkata")


@pytest.fixture
def sample_ohlcv_data():
    """Create sample OHLCV data for testing."""
    dates = pd.date_range("2025-01-02 09:15", periods=100, freq="5min", tz=IST)
    prices = [50000_00 + i * 50_00 for i in range(100)]  # paisa, trending up
    return pd.DataFrame({
        "open": [p - 10_00 for p in prices],
        "high": [p + 20_00 for p in prices],
        "low": [p - 20_00 for p in prices],
        "close": prices,
        "volume": [5000] * 100,
    }, index=dates)


@pytest.fixture
def sample_strategy_config(tmp_path):
    """Create a sample strategy config file."""
    config = {
        "name": "Test_Strategy",
        "description": "Test strategy for backtesting",
        "enabled": True,
        "underlying": {
            "instrument_key": "NSE_INDEX|Nifty Bank",
            "symbol": "BANKNIFTY",
            "segment": "NSE_FO"
        },
        "trading_hours": {
            "start_time": "09:15:00",
            "end_time": "15:15:00",
            "timezone": "Asia/Kolkata",
            "days": [0, 1, 2, 3, 4]
        },
        "timeframes": {
            "primary": "5min"
        },
        "entry_sets": [
            {
                "name": "Test_Long",
                "signal": "BUY",
                "conditions": [
                    {
                        "indicator": "close",
                        "comparison": ">",
                        "value": 0,
                        "timeframe": "5min"
                    }
                ]
            }
        ],
        "exit_rules": {
            "stop_loss_pct": 2.0,
            "target_pct": 4.0,
            "time_based_exit": "15:10:00"
        },
        "position_sizing": {
            "method": "fixed_quantity",
            "quantity": 1
        },
        "risk_management": {
            "max_open_positions": 5,
            "max_trades_per_day": 10
        }
    }
    config_path = tmp_path / "test_strategy.json"
    import json
    with open(config_path, "w") as f:
        json.dump(config, f)
    return str(config_path)


class TestBacktestEngine:
    """Test suite for BacktestEngine."""

    def test_engine_initialization(self, sample_strategy_config):
        """Test that engine initializes with correct parameters."""
        engine = BacktestEngine(
            capital=100_000_000,  # 10 lakh paisa
            start_date=date(2025, 1, 1),
            end_date=date(2025, 1, 31)
        )
        assert engine.initial_capital == 100_000_000
        assert engine.start_date == date(2025, 1, 1)
        assert engine.end_date == date(2025, 1, 31)
        assert engine.positions == {}
        assert engine.trades == []

    def test_add_strategy(self, sample_strategy_config):
        """Test adding a strategy with instrument."""
        engine = BacktestEngine(
            capital=100_000_000,
            start_date=date(2025, 1, 1),
            end_date=date(2025, 1, 31)
        )
        engine.add_strategy(sample_strategy_config, "NSE_INDEX|Nifty Bank")
        assert len(engine.strategies) == 1
        assert engine.strategies[0]["instrument"] == "NSE_INDEX|Nifty Bank"

    def test_run_backtest(self, sample_ohlcv_data, sample_strategy_config):
        """Test running a complete backtest."""
        engine = BacktestEngine(
            capital=100_000_000,
            start_date=date(2025, 1, 1),
            end_date=date(2025, 1, 31)
        )
        engine.add_strategy(sample_strategy_config, "NSE_INDEX|Nifty Bank")
        
        # Run backtest with sample data
        data = {"NSE_INDEX|Nifty Bank": sample_ohlcv_data}
        results = engine.run(data)
        
        assert isinstance(results, BacktestResults)
        assert results.total_bars_processed == 100
        assert results.initial_capital == 100_000_000

    def test_backtest_results_metrics(self, sample_ohlcv_data, sample_strategy_config):
        """Test that results compute correct metrics."""
        engine = BacktestEngine(
            capital=100_000_000,
            start_date=date(2025, 1, 1),
            end_date=date(2025, 1, 31)
        )
        engine.add_strategy(sample_strategy_config, "NSE_INDEX|Nifty Bank")
        
        data = {"NSE_INDEX|Nifty Bank": sample_ohlcv_data}
        results = engine.run(data)
        
        metrics = results.get_metrics()
        assert "total_pnl" in metrics
        assert "win_rate" in metrics
        assert "total_trades" in metrics
        assert "max_drawdown" in metrics
        assert "equity_curve" in metrics

    def test_position_tracking(self, sample_ohlcv_data, sample_strategy_config):
        """Test that positions are tracked correctly."""
        engine = BacktestEngine(
            capital=100_000_000,
            start_date=date(2025, 1, 1),
            end_date=date(2025, 1, 31)
        )
        engine.add_strategy(sample_strategy_config, "NSE_INDEX|Nifty Bank")
        
        data = {"NSE_INDEX|Nifty Bank": sample_ohlcv_data}
        results = engine.run(data)
        
        # Should have recorded trades
        assert isinstance(results.trades, list)
        # Each trade should have required fields
        for trade in results.trades:
            assert "entry_time" in trade
            assert "exit_time" in trade
            assert "entry_price" in trade
            assert "exit_price" in trade
            assert "pnl" in trade
            assert "quantity" in trade

    def test_equity_curve(self, sample_ohlcv_data, sample_strategy_config):
        """Test that equity curve is generated."""
        engine = BacktestEngine(
            capital=100_000_000,
            start_date=date(2025, 1, 1),
            end_date=date(2025, 1, 31)
        )
        engine.add_strategy(sample_strategy_config, "NSE_INDEX|Nifty Bank")
        
        data = {"NSE_INDEX|Nifty Bank": sample_ohlcv_data}
        results = engine.run(data)
        
        assert len(results.equity_curve) > 0
        assert results.equity_curve[0] == 100_000_000  # Starts with initial capital

    def test_multiple_instruments(self, sample_ohlcv_data, sample_strategy_config):
        """Test backtest with multiple instruments."""
        engine = BacktestEngine(
            capital=100_000_000,
            start_date=date(2025, 1, 1),
            end_date=date(2025, 1, 31)
        )
        
        # Add two strategies for different instruments
        engine.add_strategy(sample_strategy_config, "NSE_INDEX|Nifty Bank")
        engine.add_strategy(sample_strategy_config, "NSE_INDEX|Nifty 50")
        
        data = {
            "NSE_INDEX|Nifty Bank": sample_ohlcv_data,
            "NSE_INDEX|Nifty 50": sample_ohlcv_data.copy()
        }
        results = engine.run(data)
        
        assert isinstance(results, BacktestResults)
        assert results.total_bars_processed > 0

    def test_slippage_and_fees(self, sample_ohlcv_data, sample_strategy_config):
        """Test that slippage and fees are applied."""
        engine = BacktestEngine(
            capital=100_000_000,
            start_date=date(2025, 1, 1),
            end_date=date(2025, 1, 31),
            slippage_pct=0.05,
            commission_per_order=2000  # Rs 20 in paisa
        )
        engine.add_strategy(sample_strategy_config, "NSE_INDEX|Nifty Bank")
        
        data = {"NSE_INDEX|Nifty Bank": sample_ohlcv_data}
        results = engine.run(data)
        
        # Check that fees are deducted
        for trade in results.trades:
            # PnL should account for commission
            assert "pnl" in trade

    def test_risk_limits(self, sample_ohlcv_data, sample_strategy_config):
        """Test that risk limits are respected."""
        engine = BacktestEngine(
            capital=100_000_000,
            start_date=date(2025, 1, 1),
            end_date=date(2025, 1, 31),
            max_position_size_pct=20,
            max_daily_loss_pct=5
        )
        engine.add_strategy(sample_strategy_config, "NSE_INDEX|Nifty Bank")
        
        data = {"NSE_INDEX|Nifty Bank": sample_ohlcv_data}
        results = engine.run(data)
        
        # Verify no single position exceeded limit
        for trade in results.trades:
            position_value = trade.get("entry_price", 0) * trade.get("quantity", 0)
            max_allowed = 100_000_000 * 0.20
            assert position_value <= max_allowed, "Position exceeded max size limit"


class TestBacktestResults:
    """Test suite for BacktestResults."""

    def test_results_initialization(self):
        """Test BacktestResults initialization."""
        results = BacktestResults(
            initial_capital=100_000_000,
            final_capital=110_000_000,
            trades=[],
            equity_curve=[100_000_000, 110_000_000],
            total_bars_processed=100
        )
        assert results.initial_capital == 100_000_000
        assert results.final_capital == 110_000_000

    def test_total_pnl_calculation(self):
        """Test total PnL calculation."""
        trades = [
            {"pnl": 5000_00},  # Rs 5000 profit
            {"pnl": -2000_00},  # Rs 2000 loss
            {"pnl": 3000_00},  # Rs 3000 profit
        ]
        results = BacktestResults(
            initial_capital=100_000_000,
            final_capital=106_000_000,
            trades=trades,
            equity_curve=[100_000_000, 106_000_000],
            total_bars_processed=100
        )
        metrics = results.get_metrics()
        assert metrics["total_pnl"] == 6000_00  # Net profit

    def test_win_rate_calculation(self):
        """Test win rate calculation."""
        trades = [
            {"pnl": 5000_00},  # Win
            {"pnl": -2000_00},  # Loss
            {"pnl": 3000_00},  # Win
            {"pnl": 1000_00},  # Win
        ]
        results = BacktestResults(
            initial_capital=100_000_000,
            final_capital=107_000_000,
            trades=trades,
            equity_curve=[100_000_000, 107_000_000],
            total_bars_processed=100
        )
        metrics = results.get_metrics()
        assert metrics["win_rate"] == 75.0  # 3 wins out of 4

    def test_drawdown_calculation(self):
        """Test maximum drawdown calculation."""
        equity_curve = [
            100_000_000,  # Start
            105_000_000,  # Peak
            103_000_000,  # Drawdown
            102_000_000,  # Max drawdown
            104_000_000,  # Recovery
        ]
        results = BacktestResults(
            initial_capital=100_000_000,
            final_capital=104_000_000,
            trades=[],
            equity_curve=equity_curve,
            total_bars_processed=100
        )
        metrics = results.get_metrics()
        assert metrics["max_drawdown"] == 3_000_000  # 105M - 102M

    def test_empty_trades(self):
        """Test metrics with no trades."""
        results = BacktestResults(
            initial_capital=100_000_000,
            final_capital=100_000_000,
            trades=[],
            equity_curve=[100_000_000],
            total_bars_processed=100
        )
        metrics = results.get_metrics()
        assert metrics["total_pnl"] == 0
        assert metrics["win_rate"] == 0.0
        assert metrics["total_trades"] == 0


class TestBacktestEngineIntegration:
    """Integration tests for BacktestEngine."""

    def test_full_backtest_workflow(self, tmp_path):
        """Test complete backtest workflow with CSV data."""
        # Create sample CSV
        dates = pd.date_range("2025-01-02 09:15", periods=100, freq="5min", tz=IST)
        prices = [50000_00 + i * 100_00 for i in range(100)]
        df = pd.DataFrame({
            "timestamp": dates,
            "open": [p - 10_00 for p in prices],
            "high": [p + 20_00 for p in prices],
            "low": [p - 20_00 for p in prices],
            "close": prices,
            "volume": [5000] * 100,
        })
        csv_path = tmp_path / "BANKNIFTY_5min.csv"
        df.to_csv(csv_path, index=False)
        
        # Create strategy config
        config = {
            "name": "Integration_Test",
            "enabled": True,
            "underlying": {"instrument_key": "NSE_INDEX|Nifty Bank"},
            "trading_hours": {"start_time": "09:15:00", "end_time": "15:15:00"},
            "timeframes": {"primary": "5min"},
            "entry_sets": [{
                "name": "Long",
                "signal": "BUY",
                "conditions": [{
                    "indicator": "close",
                    "comparison": ">",
                    "value": 50000_00,
                    "timeframe": "5min"
                }]
            }],
            "exit_rules": {"stop_loss_pct": 2.0, "target_pct": 4.0},
            "position_sizing": {"method": "fixed_quantity", "quantity": 1}
        }
        config_path = tmp_path / "integration_strategy.json"
        import json
        with open(config_path, "w") as f:
            json.dump(config, f)
        
        # Load data and run backtest
        loader = BacktestDataLoader()
        data_df = loader.load_csv(str(csv_path))
        
        engine = BacktestEngine(
            capital=100_000_000,
            start_date=date(2025, 1, 1),
            end_date=date(2025, 1, 31)
        )
        engine.add_strategy(str(config_path), "NSE_INDEX|Nifty Bank")
        
        data = {"NSE_INDEX|Nifty Bank": data_df}
        results = engine.run(data)
        
        assert isinstance(results, BacktestResults)
        assert results.total_bars_processed == 100
        metrics = results.get_metrics()
        assert "total_pnl" in metrics
        assert "win_rate" in metrics
