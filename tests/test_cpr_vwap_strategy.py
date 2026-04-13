"""Tests for CPR (Central Pivot Range) indicator and CPR+VWAP Bounce strategy.

Tests CPR calculations, narrow CPR detection, and strategy condition evaluation.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.strategy.builder import StrategyBuilder, StrategyConfig
from src.strategy.conditions import ConditionEvaluator
from src.strategy.indicators import IndicatorEngine


class TestCPRIndicator:
    """Test Central Pivot Range calculations."""

    @pytest.fixture
def sample_daily_data(self) -> pd.DataFrame:
        """Create sample daily OHLC data for CPR calculation."""
        # BankNifty-like prices in PAISA (45000 = 45000.00 INR)
        data = {
            "open": [4480000, 4500000, 4520000, 4510000, 4530000],
            "high": [4520000, 4550000, 4560000, 4540000, 4580000],
            "low": [4470000, 4490000, 4500000, 4500000, 4520000],
            "close": [4505000, 4530000, 4545000, 4520000, 4550000],
            "volume": [1000000, 1200000, 1100000, 900000, 1300000],
        }
        index = pd.date_range("2024-01-01", periods=5, freq="D")
        return pd.DataFrame(data, index=index)

    def test_cpr_calculation_structure(self, sample_daily_data: pd.DataFrame) -> None:
        """Test that CPR calculation returns correct structure."""
        engine = IndicatorEngine()
        cpr = engine.calculate_cpr(sample_daily_data)

        # Should have all expected keys
        expected_keys = [
            "pivot", "bc", "tc", "cpr_width", "cpr_width_pct",
            "r1", "r2", "r3", "s1", "s2", "s3",
        ]
        for key in expected_keys:
            assert key in cpr, f"Missing key: {key}"

    def test_cpr_calculation_values(self, sample_daily_data: pd.DataFrame) -> None:
        """Test CPR calculation with known values."""
        engine = IndicatorEngine()
        cpr = engine.calculate_cpr(sample_daily_data)

        # Last day: high=4580000, low=4520000, close=4550000
        # Pivot = (4580000 + 4520000 + 4550000) / 3 = 4550000
        # BC = (4580000 + 4520000) / 2 = 4550000
        # TC = 2*4550000 - 4550000 = 4550000

        expected_pivot = (4580000 + 4520000 + 4550000) / 3
        expected_bc = (4580000 + 4520000) / 2
        expected_tc = 2 * expected_pivot - expected_bc

        assert cpr["pivot"] == pytest.approx(expected_pivot, rel=1e-5)
        assert cpr["bc"] == pytest.approx(expected_bc, rel=1e-5)
        assert cpr["tc"] == pytest.approx(expected_tc, rel=1e-5)

    def test_cpr_width_calculation(self, sample_daily_data: pd.DataFrame) -> None:
        """Test CPR width calculation."""
        engine = IndicatorEngine()
        cpr = engine.calculate_cpr(sample_daily_data)

        # Width should be absolute difference between TC and BC
        expected_width = abs(cpr["tc"] - cpr["bc"])
        assert cpr["cpr_width"] == pytest.approx(expected_width, rel=1e-5)

        # Width percentage should be width / pivot * 100
        expected_width_pct = (expected_width / cpr["pivot"]) * 100
        assert cpr["cpr_width_pct"] == pytest.approx(expected_width_pct, rel=1e-5)

    def test_support_resistance_levels(self, sample_daily_data: pd.DataFrame) -> None:
        """Test support and resistance level calculations."""
        engine = IndicatorEngine()
        cpr = engine.calculate_cpr(sample_daily_data)

        # R1 = 2*Pivot - Low
        expected_r1 = 2 * cpr["pivot"] - 4520000
        assert cpr["r1"] == pytest.approx(expected_r1, rel=1e-5)

        # R2 = Pivot + (High - Low)
        expected_r2 = cpr["pivot"] + (4580000 - 4520000)
        assert cpr["r2"] == pytest.approx(expected_r2, rel=1e-5)

        # S1 = 2*Pivot - High
        expected_s1 = 2 * cpr["pivot"] - 4580000
        assert cpr["s1"] == pytest.approx(expected_s1, rel=1e-5)

    def test_narrow_cpr_detection(self, sample_daily_data: pd.DataFrame) -> None:
        """Test narrow CPR detection."""
        engine = IndicatorEngine()
        cpr = engine.calculate_cpr(sample_daily_data)

        # With BC = TC = 4550000, width is 0, so it's narrow
        is_narrow = engine.is_narrow_cpr(cpr, threshold_pct=0.4)
        assert is_narrow is True

        # Test with higher threshold
        is_narrow_high = engine.is_narrow_cpr(cpr, threshold_pct=0.0)
        assert is_narrow_high is False

    def test_price_in_value_zone(self, sample_daily_data: pd.DataFrame) -> None:
        """Test price in CPR value zone detection."""
        engine = IndicatorEngine()
        cpr = engine.calculate_cpr(sample_daily_data)

        # Price between BC and TC should be in value zone
        # BC = 4550000, TC = 4550000 (in this case)
        price_in_zone = 4550000
        assert engine.is_price_in_cpr_value_zone(price_in_zone, cpr) is True

        # Price outside should not be in value zone
        price_above = 4600000
        assert engine.is_price_in_cpr_value_zone(price_above, cpr) is False

    def test_cpr_position_detection(self, sample_daily_data: pd.DataFrame) -> None:
        """Test CPR position detection."""
        engine = IndicatorEngine()
        cpr = engine.calculate_cpr(sample_daily_data)

        # Price at pivot should be in value zone
        position = engine.get_cpr_position(cpr["pivot"], cpr)
        assert position == "value_zone"

        # Price above R3
        position = engine.get_cpr_position(cpr["r3"] + 1000, cpr)
        assert position == "above_r3"

        # Price below S3
        position = engine.get_cpr_position(cpr["s3"] - 1000, cpr)
        assert position == "below_s3"

    def test_cpr_empty_data(self) -> None:
        """Test CPR calculation with empty data."""
        engine = IndicatorEngine()
        empty_df = pd.DataFrame()

        with pytest.raises(ValueError, match="cannot be empty"):
            engine.calculate_cpr(empty_df)

    def test_cpr_missing_columns(self) -> None:
        """Test CPR calculation with missing columns."""
        engine = IndicatorEngine()
        df = pd.DataFrame({"open": [100], "close": [110]})

        with pytest.raises(ValueError, match="Missing required columns"):
            engine.calculate_cpr(df)


class TestCPRVWAPStrategyConfig:
    """Test CPR VWAP Bounce strategy configuration."""

    def test_config_file_exists(self) -> None:
        """Test that the strategy config file exists and is valid JSON."""
        config_path = Path("config/strategies/cpr_vwap_bounce_banknifty.json")
        assert config_path.exists(), f"Config file not found: {config_path}"

        with open(config_path) as f:
            config = json.load(f)

        assert config["name"] == "CPR_VWAP_Bounce_BankNifty"
        assert config["enabled"] is True

    def test_config_loads_with_builder(self) -> None:
        """Test that the strategy config loads correctly with StrategyBuilder."""
        config_path = Path("config/strategies/cpr_vwap_bounce_banknifty.json")
        config = StrategyBuilder.load_strategy(config_path)

        assert isinstance(config, StrategyConfig)
        assert config.name == "CPR_VWAP_Bounce_BankNifty"
        assert config.underlying["symbol"] == "BANKNIFTY"

    def test_entry_sets_configuration(self) -> None:
        """Test entry sets are properly configured."""
        config_path = Path("config/strategies/cpr_vwap_bounce_banknifty.json")
        config = StrategyBuilder.load_strategy(config_path)

        assert len(config.entry_sets) == 2

        # Check long bounce set
        long_set = next(s for s in config.entry_sets if s.name == "CPR_Long_Bounce")
        assert long_set.signal == "CE"
        assert len(long_set.conditions) == 5

        # Check short bounce set
        short_set = next(s for s in config.entry_sets if s.name == "CPR_Short_Bounce")
        assert short_set.signal == "PE"
        assert len(short_set.conditions) == 5

    def test_exit_rules_configuration(self) -> None:
        """Test exit rules are properly configured."""
        config_path = Path("config/strategies/cpr_vwap_bounce_banknifty.json")
        config = StrategyBuilder.load_strategy(config_path)

        assert config.exit_rules.stop_loss_pct == 25
        assert config.exit_rules.target_pct == 37
        assert config.exit_rules.time_based_exit == "14:30:00"

    def test_risk_management_configuration(self) -> None:
        """Test risk management parameters."""
        config_path = Path("config/strategies/cpr_vwap_bounce_banknifty.json")
        config = StrategyBuilder.load_strategy(config_path)

        assert config.risk_management["max_open_positions"] == 2
        assert config.risk_management["max_trades_per_day"] == 3
        assert config.risk_management["require_narrow_cpr"] is True

    def test_strategy_params(self) -> None:
        """Test strategy-specific parameters."""
        config_path = Path("config/strategies/cpr_vwap_bounce_banknifty.json")
        config = StrategyBuilder.load_strategy(config_path)

        assert config.params["cpr_narrow_threshold"] == 0.4
        assert config.params["atr_threshold"] == 1500


class TestCPRConditionEvaluation:
    """Test CPR condition evaluation in the strategy engine."""

    @pytest.fixture
def intraday_bars(self) -> dict[str, pd.DataFrame]:
        """Create sample intraday 5-min bars for testing."""
        # Create 5-min bars for a trading day
        times = pd.date_range("2024-01-05 09:15", "2024-01-05 15:30", freq="5min")
        n_bars = len(times)

        # Simulate price around CPR levels (pivot ~4550000)
        base_price = 4550000
        prices = base_price + np.random.randn(n_bars) * 5000

        df = pd.DataFrame({
            "open": prices + np.random.randn(n_bars) * 100,
            "high": prices + abs(np.random.randn(n_bars) * 200) + 50,
            "low": prices - abs(np.random.randn(n_bars) * 200) - 50,
            "close": prices,
            "volume": np.random.randint(100000, 500000, n_bars),
        }, index=times)

        return {"BANKNIFTY": df}

    @pytest.fixture
def cpr_data(self) -> dict[str, float]:
        """Sample CPR data for testing."""
        return {
            "pivot": 4550000,
            "bc": 4548000,
            "tc": 4552000,
            "cpr_width": 4000,
            "cpr_width_pct": 0.088,
            "r1": 4560000,
            "r2": 4570000,
            "r3": 4580000,
            "s1": 4540000,
            "s2": 4530000,
            "s3": 4520000,
        }

    def test_cpr_indicator_resolution(self, intraday_bars: dict[str, pd.DataFrame], cpr_data: dict[str, float]) -> None:
        """Test that CPR indicators can be resolved in conditions."""
        evaluator = ConditionEvaluator()

        # Create condition with CPR data in parameters
        from src.strategy.builder import Condition, ComparisonOperator

        condition = Condition(
            indicator="CPR_PIVOT",
            comparison=ComparisonOperator.GTE,
            value=4500000,
            parameters={"cpr_data": cpr_data},
        )

        # Evaluate the condition
        result = evaluator.evaluate(condition, intraday_bars, "BANKNIFTY")

        # Pivot is 4550000, which is >= 4500000
        assert result is True

    def test_cpr_long_bounce_conditions(self, intraday_bars: dict[str, pd.DataFrame], cpr_data: dict[str, float]) -> None:
        """Test evaluation of CPR long bounce conditions."""
        evaluator = ConditionEvaluator()

        from src.strategy.builder import Condition, ComparisonOperator

        # Condition: Price >= CPR_BC (bottom central)
        condition1 = Condition(
            indicator="close",
            comparison=ComparisonOperator.GTE,
            against="CPR_BC",
            parameters={"cpr_data": cpr_data},
        )

        # Manually set price to be above BC
        intraday_bars["BANKNIFTY"].loc[:, "close"] = 4549000  # Above BC (4548000)

        result = evaluator.evaluate(condition1, intraday_bars, "BANKNIFTY")
        assert result is True

    def test_narrow_cpr_validation(self, cpr_data: dict[str, float]) -> None:
        """Test narrow CPR validation logic."""
        engine = IndicatorEngine()

        # This CPR has width 0.088%, which is less than 0.4%
        is_narrow = engine.is_narrow_cpr(cpr_data, threshold_pct=0.4)
        assert is_narrow is True

        # Test with wide CPR
        wide_cpr = cpr_data.copy()
        wide_cpr["cpr_width_pct"] = 0.5  # 0.5% width
        is_narrow = engine.is_narrow_cpr(wide_cpr, threshold_pct=0.4)
        assert is_narrow is False


class TestCPRStrategyIntegration:
    """Integration tests for CPR+VWAP strategy."""

    def test_strategy_end_to_end(self) -> None:
        """Test the complete strategy flow from config to signal generation."""
        # Load strategy config
        config_path = Path("config/strategies/cpr_vwap_bounce_banknifty.json")
        config = StrategyBuilder.load_strategy(config_path)

        # Verify all components are present
        assert config.name == "CPR_VWAP_Bounce_BankNifty"
        assert len(config.entry_sets) == 2
        assert config.exit_rules.stop_loss_pct is not None
        assert config.exit_rules.target_pct is not None

        # Verify risk management
        assert "max_open_positions" in config.risk_management
        assert "require_narrow_cpr" in config.risk_management

    def test_cpr_with_vwap_confluence(self) -> None:
        """Test that CPR and VWAP work together for mean reversion."""
        # Create sample data
        times = pd.date_range("2024-01-05 09:15", "2024-01-05 10:00", freq="5min")
        n = len(times)

        # Price bouncing between CPR BC and Pivot, above VWAP
        df = pd.DataFrame({
            "open": [4548500] * n,
            "high": [4549500] * n,
            "low": [4547500] * n,
            "close": [4549000] * n,  # Between BC (4548000) and Pivot (4550000)
            "volume": [200000] * n,
        }, index=times)

        # Calculate VWAP
        engine = IndicatorEngine(df)
        df["vwap"] = engine.vwap()

        # Price above VWAP indicates bullish intraday
        assert df["close"].iloc[-1] > df["vwap"].iloc[-1]

        # Price in CPR value zone (between BC and TC)
        cpr_data = {
            "pivot": 4550000,
            "bc": 4548000,
            "tc": 4552000,
        }
        assert engine.is_price_in_cpr_value_zone(df["close"].iloc[-1], cpr_data)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
