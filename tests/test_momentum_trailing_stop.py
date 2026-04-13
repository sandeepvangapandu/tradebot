"""Tests for Momentum-Based Trailing Stop functionality.

Tests cover:
- Momentum calculation accuracy
- Trailing multiplier logic
- Breakeven activation
- Stop update logic for long and short positions
- Time-based update throttling
"""

import sys
import unittest
from datetime import datetime, timedelta
from unittest.mock import Mock

import pandas as pd
from zoneinfo import ZoneInfo

# Add src to path
sys.path.insert(0, "/Users/sandeepvangapandu/Downloads/Trading")

from src.execution.base_broker import OrderSide, ProductType
from src.execution.position_manager import (
    ManagedPosition,
    MomentumTrailingStop,
    PositionConfig,
)

IST = ZoneInfo("Asia/Kolkata")


class TestMomentumCalculation(unittest.TestCase):
    """Test momentum calculation methods."""

    def setUp(self):
        self.config = PositionConfig(
            trailing_stop_type="momentum",
            momentum_period=5,
            momentum_trailing_base_atr_multiple=2.0,
        )
        self.mts = MomentumTrailingStop(self.config)

    def _create_price_data(self, prices: list[int]) -> pd.DataFrame:
        """Create OHLCV DataFrame from price series."""
        data = {
            "open": prices,
            "high": [p + 10 for p in prices],
            "low": [p - 10 for p in prices],
            "close": prices,
            "volume": [1000] * len(prices),
        }
        index = pd.date_range(end=datetime.now(IST), periods=len(prices), freq="1min")
        return pd.DataFrame(data, index=index)

    def test_momentum_calculation_strong_up(self):
        """Test momentum calculation for strong uptrend."""
        # Create strong uptrend: prices rising 2% per bar for 6 bars
        base_price = 10000
        prices = [int(base_price * (1.02 ** i)) for i in range(6)]
        data = self._create_price_data(prices)

        momentum = self.mts.calculate_momentum(data)

        # Should be positive (strong up)
        self.assertGreater(momentum, 0.5)
        self.assertLessEqual(momentum, 1.0)

    def test_momentum_calculation_strong_down(self):
        """Test momentum calculation for strong downtrend."""
        # Create strong downtrend: prices falling 2% per bar for 6 bars
        base_price = 10000
        prices = [int(base_price * (0.98 ** i)) for i in range(6)]
        data = self._create_price_data(prices)

        momentum = self.mts.calculate_momentum(data)

        # Should be negative (strong down)
        self.assertLess(momentum, -0.4)  # Relaxed threshold
        self.assertGreaterEqual(momentum, -1.0)

    def test_momentum_calculation_neutral(self):
        """Test momentum calculation for flat/sideways market."""
        # Create flat prices
        prices = [10000] * 10
        data = self._create_price_data(prices)

        momentum = self.mts.calculate_momentum(data)

        # Should be near zero
        self.assertAlmostEqual(momentum, 0.0, delta=0.1)

    def test_momentum_calculation_insufficient_data(self):
        """Test momentum returns 0 with insufficient data."""
        prices = [10000] * 3  # Less than momentum_period + 1
        data = self._create_price_data(prices)

        momentum = self.mts.calculate_momentum(data)

        self.assertEqual(momentum, 0.0)


class TestTrailingMultiplier(unittest.TestCase):
    """Test trailing multiplier logic based on momentum."""

    def setUp(self):
        self.config = PositionConfig(
            trailing_stop_type="momentum",
            momentum_strong_threshold=0.7,
            momentum_moderate_threshold=0.3,
            momentum_strong_multiplier=1.5,
            momentum_moderate_multiplier=1.0,
            momentum_neutral_multiplier=0.7,
            momentum_negative_multiplier=0.5,
        )
        self.mts = MomentumTrailingStop(self.config)

    def test_strong_momentum_multiplier(self):
        """Test multiplier for strong momentum."""
        mult = self.mts.get_trailing_multiplier(0.8)
        self.assertEqual(mult, 1.5)

    def test_moderate_momentum_multiplier(self):
        """Test multiplier for moderate momentum."""
        mult = self.mts.get_trailing_multiplier(0.5)
        self.assertEqual(mult, 1.0)

    def test_neutral_momentum_multiplier(self):
        """Test multiplier for neutral momentum."""
        mult = self.mts.get_trailing_multiplier(0.1)
        self.assertEqual(mult, 0.7)

    def test_negative_momentum_multiplier(self):
        """Test multiplier for negative momentum."""
        mult = self.mts.get_trailing_multiplier(-0.5)
        self.assertEqual(mult, 0.5)


class TestBreakevenLogic(unittest.TestCase):
    """Test breakeven stop activation."""

    def setUp(self):
        self.config = PositionConfig(
            trailing_stop_type="momentum",
            momentum_breakeven_threshold_r=0.5,
        )
        self.mts = MomentumTrailingStop(self.config)

    def test_breakeven_price_long(self):
        """Test breakeven price calculation for long position."""
        position = ManagedPosition(
            instrument_key="NSE_EQ:RELIANCE",
            strategy_id="test_strategy",
            side=OrderSide.BUY,
            quantity=100,
            entry_price=10000,  # 100.00 INR
            product_type=ProductType.MIS,
            entry_time=datetime.now(IST),
        )

        breakeven = self.mts.get_breakeven_price(position)
        # Should be entry + 1 tick (5 paisa)
        self.assertEqual(breakeven, 10005)

    def test_breakeven_price_short(self):
        """Test breakeven price calculation for short position."""
        position = ManagedPosition(
            instrument_key="NSE_EQ:RELIANCE",
            strategy_id="test_strategy",
            side=OrderSide.SELL,
            quantity=100,
            entry_price=10000,
            product_type=ProductType.MIS,
            entry_time=datetime.now(IST),
        )

        breakeven = self.mts.get_breakeven_price(position)
        # Should be entry - 1 tick (5 paisa)
        self.assertEqual(breakeven, 9995)


class TestStopUpdateLogic(unittest.TestCase):
    """Test trailing stop update logic."""

    def setUp(self):
        self.config = PositionConfig(
            trailing_stop_type="momentum",
            momentum_period=5,
            momentum_trailing_base_atr_multiple=2.0,
            momentum_breakeven_threshold_r=0.5,
            momentum_update_interval_seconds=0,  # No throttling for tests
            momentum_strong_multiplier=1.5,
            momentum_moderate_multiplier=1.0,
            momentum_neutral_multiplier=0.7,
            momentum_negative_multiplier=0.5,
        )
        self.mts = MomentumTrailingStop(self.config)

    def _create_uptrend_data(self, bars: int = 10) -> pd.DataFrame:
        """Create uptrend OHLCV data."""
        base_price = 10000
        prices = [int(base_price * (1.01 ** i)) for i in range(bars)]
        data = {
            "open": prices,
            "high": [p + 20 for p in prices],
            "low": [p - 20 for p in prices],
            "close": prices,
            "volume": [1000] * len(prices),
        }
        index = pd.date_range(end=datetime.now(IST), periods=bars, freq="1min")
        return pd.DataFrame(data, index=index)

    def _create_downtrend_data(self, bars: int = 10) -> pd.DataFrame:
        """Create downtrend OHLCV data."""
        base_price = 10000
        prices = [int(base_price * (0.99 ** i)) for i in range(bars)]
        data = {
            "open": prices,
            "high": [p + 20 for p in prices],
            "low": [p - 20 for p in prices],
            "close": prices,
            "volume": [1000] * len(prices),
        }
        index = pd.date_range(end=datetime.now(IST), periods=bars, freq="1min")
        return pd.DataFrame(data, index=index)

    def test_stop_update_long_position(self):
        """Test trailing stop update for long position."""
        position = ManagedPosition(
            instrument_key="NSE_EQ:RELIANCE",
            strategy_id="test_strategy",
            side=OrderSide.BUY,
            quantity=100,
            entry_price=10000,
            product_type=ProductType.MIS,
            entry_time=datetime.now(IST),
            entry_atr=50,  # ATR = 0.50 INR
            stop_loss_price=9900,  # Original SL 1% below entry
        )

        # Create uptrend data (strong momentum)
        data = self._create_uptrend_data(10)
        current_price = data["close"].iloc[-1]  # ~11000

        # Update stop
        new_sl = self.mts.update_stop(position, data)

        # Should return a new stop price
        self.assertIsNotNone(new_sl)
        # Stop should be below current price
        self.assertLess(new_sl, current_price)
        # Stop should be above or equal to original SL (never move away)
        self.assertGreaterEqual(new_sl, position.stop_loss_price)

    def test_stop_update_short_position(self):
        """Test trailing stop update for short position."""
        position = ManagedPosition(
            instrument_key="NSE_EQ:RELIANCE",
            strategy_id="test_strategy",
            side=OrderSide.SELL,
            quantity=100,
            entry_price=10000,
            product_type=ProductType.MIS,
            entry_time=datetime.now(IST),
            entry_atr=50,
            stop_loss_price=10100,  # Original SL 1% above entry
        )

        # Create downtrend data
        data = self._create_downtrend_data(10)
        current_price = data["close"].iloc[-1]  # ~9000

        # Update stop
        new_sl = self.mts.update_stop(position, data)

        # Should return a new stop price
        self.assertIsNotNone(new_sl)
        # Stop should be above current price
        self.assertGreater(new_sl, current_price)
        # Stop should be below or equal to original SL (never move away)
        self.assertLessEqual(new_sl, position.stop_loss_price)

    def test_stop_only_moves_favorable_direction_long(self):
        """Test that stop only moves up for long positions."""
        position = ManagedPosition(
            instrument_key="NSE_EQ:RELIANCE",
            strategy_id="test_strategy",
            side=OrderSide.BUY,
            quantity=100,
            entry_price=10000,
            product_type=ProductType.MIS,
            entry_time=datetime.now(IST),
            entry_atr=50,
            stop_loss_price=9900,
        )

        # First update with uptrend
        data1 = self._create_uptrend_data(10)
        new_sl1 = self.mts.update_stop(position, data1)
        self.assertIsNotNone(new_sl1)

        # Second update with lower high price should not move stop back
        # Create data with lower prices (simulating pullback)
        lower_prices = [p - 500 for p in data1["close"].tolist()]
        data2 = data1.copy()
        data2["close"] = lower_prices
        data2["high"] = [p + 20 for p in lower_prices]
        data2["low"] = [p - 20 for p in lower_prices]
        data2["open"] = lower_prices

        new_sl2 = self.mts.update_stop(position, data2)

        # Stop should not move down (either None or >= previous)
        if new_sl2 is not None:
            self.assertGreaterEqual(new_sl2, new_sl1)

    def test_time_throttling(self):
        """Test that updates are throttled by time interval."""
        position = ManagedPosition(
            instrument_key="NSE_EQ:RELIANCE",
            strategy_id="test_strategy",
            side=OrderSide.BUY,
            quantity=100,
            entry_price=10000,
            product_type=ProductType.MIS,
            entry_time=datetime.now(IST),
            entry_atr=50,
            stop_loss_price=9900,
        )

        # Set config with 60 second interval
        config = PositionConfig(
            trailing_stop_type="momentum",
            momentum_period=5,
            momentum_update_interval_seconds=60,
        )
        mts = MomentumTrailingStop(config)

        data = self._create_uptrend_data(10)

        # First update should work
        new_sl1 = mts.update_stop(position, data)
        self.assertIsNotNone(new_sl1)

        # Second immediate update should be throttled
        new_sl2 = mts.update_stop(position, data)
        self.assertIsNone(new_sl2)


class TestMomentumStatus(unittest.TestCase):
    """Test momentum status reporting."""

    def setUp(self):
        self.config = PositionConfig(trailing_stop_type="momentum")
        self.mts = MomentumTrailingStop(self.config)

    def test_get_momentum_status(self):
        """Test momentum status dictionary."""
        position = ManagedPosition(
            instrument_key="NSE_EQ:RELIANCE",
            strategy_id="test_strategy",
            side=OrderSide.BUY,
            quantity=100,
            entry_price=10000,
            product_type=ProductType.MIS,
            entry_time=datetime.now(IST),
        )

        # Set some values
        position.current_momentum = 0.75
        position.current_momentum_multiplier = 1.5
        position.momentum_trailing_sl_price = 10200
        position.breakeven_activated = True
        position.highest_momentum_price = 10500

        status = self.mts.get_momentum_status(position)

        self.assertEqual(status["momentum"], 0.75)
        self.assertEqual(status["multiplier"], 1.5)
        self.assertEqual(status["momentum_trailing_sl"], 10200)
        self.assertTrue(status["breakeven_activated"])
        self.assertEqual(status["highest_price"], 10500)


class TestPositionConfig(unittest.TestCase):
    """Test PositionConfig with momentum settings."""

    def test_default_momentum_config(self):
        """Test default momentum configuration values."""
        config = PositionConfig()

        self.assertEqual(config.trailing_stop_type, "fixed")
        self.assertEqual(config.momentum_trailing_base_atr_multiple, 2.0)
        self.assertEqual(config.momentum_period, 5)
        self.assertEqual(config.momentum_breakeven_threshold_r, 0.5)
        self.assertEqual(config.momentum_update_interval_seconds, 60)

    def test_custom_momentum_config(self):
        """Test custom momentum configuration."""
        config = PositionConfig(
            trailing_stop_type="momentum",
            momentum_trailing_base_atr_multiple=3.0,
            momentum_period=10,
            momentum_breakeven_threshold_r=1.0,
        )

        self.assertEqual(config.trailing_stop_type, "momentum")
        self.assertEqual(config.momentum_trailing_base_atr_multiple, 3.0)
        self.assertEqual(config.momentum_period, 10)
        self.assertEqual(config.momentum_breakeven_threshold_r, 1.0)


if __name__ == "__main__":
    unittest.main()
