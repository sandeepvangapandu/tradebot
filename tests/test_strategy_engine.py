"""Tests for the strategy engine components.

Tests StrategyBuilder, ConditionEvaluator, and SetEvaluator.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


# Mock strategy components for testing
class MockStrategyBuilder:
    """Mock strategy builder that loads JSON config."""

    def __init__(self, config_path: str) -> None:
        self.config_path = Path(config_path)
        self.config: dict[str, Any] | None = None

    def load(self) -> dict[str, Any]:
        """Load strategy configuration from JSON file."""
        with open(self.config_path, "r") as f:
            self.config = json.load(f)
        return self.config

    def validate(self) -> bool:
        """Validate the loaded configuration."""
        if self.config is None:
            raise ValueError("Config not loaded. Call load() first.")

        required_fields = ["name", "entry_sets", "exit_rules", "position_sizing"]
        for field in required_fields:
            if field not in self.config:
                raise ValueError(f"Missing required field: {field}")

        return True

    def build(self) -> "MockStrategy":
        """Build a strategy instance from config."""
        if self.config is None:
            self.load()
        return MockStrategy(self.config)


class MockStrategy:
    """Mock strategy instance."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.name = config["name"]
        self.entry_sets = config.get("entry_sets", [])
        self.exit_rules = config.get("exit_rules", {})


class ConditionEvaluator:
    """Evaluates individual conditions for strategy signals."""

    COMPARISON_OPS = {
        ">": lambda a, b: a > b,
        ">=": lambda a, b: a >= b,
        "<": lambda a, b: a < b,
        "<=": lambda a, b: a <= b,
        "==": lambda a, b: a == b,
        "!=": lambda a, b: a != b,
    }

    def __init__(self, data: dict[str, pd.DataFrame]) -> None:
        """Initialize with market data for different timeframes.

        Args:
            data: Dictionary mapping timeframe strings to DataFrames.
        """
        self.data = data

    def evaluate(self, condition: dict[str, Any]) -> bool:
        """Evaluate a single condition.

        Args:
            condition: Condition dict with indicator, comparison, reference, etc.

        Returns:
            True if condition is met, False otherwise.
        """
        timeframe = condition.get("timeframe", "5min")
        df = self.data.get(timeframe)

        if df is None or df.empty:
            return False

        indicator = condition["indicator"]
        comparison = condition["comparison"]
        reference = condition["reference"]

        # Get indicator value (last value)
        if indicator == "Spot_Price":
            indicator_value = df["close"].iloc[-1]
        elif indicator in df.columns:
            indicator_value = df[indicator].iloc[-1]
        else:
            return False

        # Get reference value
        if reference == "constant":
            reference_value = condition.get("constant_value", 0)
        elif reference in df.columns:
            reference_value = df[reference].iloc[-1]
        else:
            return False

        # Handle crosses_above and crosses_below
        if comparison == "crosses_above":
            return self._crosses_above(df, indicator, reference)
        elif comparison == "crosses_below":
            return self._crosses_below(df, indicator, reference)

        # Standard comparison
        op = self.COMPARISON_OPS.get(comparison)
        if op is None:
            return False

        return op(indicator_value, reference_value)

    def _crosses_above(self, df: pd.DataFrame, col1: str, col2: str) -> bool:
        """Check if col1 crossed above col2 on the last bar."""
        if col1 == "Spot_Price":
            col1 = "close"

        if len(df) < 2:
            return False

        current = df[col1].iloc[-1] > df[col2].iloc[-1]
        previous = df[col1].iloc[-2] <= df[col2].iloc[-2]
        return current and previous

    def _crosses_below(self, df: pd.DataFrame, col1: str, col2: str) -> bool:
        """Check if col1 crossed below col2 on the last bar."""
        if col1 == "Spot_Price":
            col1 = "close"

        if len(df) < 2:
            return False

        current = df[col1].iloc[-1] < df[col2].iloc[-1]
        previous = df[col1].iloc[-2] >= df[col2].iloc[-2]
        return current and previous


class SetEvaluator:
    """Evaluates entry sets and generates signals."""

    def __init__(self, condition_evaluator: ConditionEvaluator) -> None:
        self.condition_evaluator = condition_evaluator

    def evaluate_set(self, entry_set: dict[str, Any]) -> dict[str, Any] | None:
        """Evaluate an entry set and return signal if all conditions met.

        Args:
            entry_set: Entry set configuration with conditions.

        Returns:
            Signal dict if conditions met, None otherwise.
        """
        conditions = entry_set.get("conditions", [])

        for condition in conditions:
            if not self.condition_evaluator.evaluate(condition):
                return None

        # All conditions met - generate signal
        return {
            "set_name": entry_set["name"],
            "signal": entry_set["signal"],
            "timestamp": pd.Timestamp.now(),
        }

    def evaluate_all(self, entry_sets: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Evaluate all entry sets and return list of active signals.

        Args:
            entry_sets: List of entry set configurations.

        Returns:
            List of signal dicts for sets with all conditions met.
        """
        signals = []
        for entry_set in entry_sets:
            signal = self.evaluate_set(entry_set)
            if signal:
                signals.append(signal)
        return signals


class TestStrategyBuilder:
    """Test StrategyBuilder configuration loading."""

    def test_load_json_config(self, temp_strategy_config_file: str) -> None:
        """Test loading strategy configuration from JSON file."""
        builder = MockStrategyBuilder(temp_strategy_config_file)
        config = builder.load()

        assert config is not None
        assert config["name"] == "Test_Strategy"
        assert "entry_sets" in config
        assert "exit_rules" in config

    def test_validate_config_success(self, temp_strategy_config_file: str) -> None:
        """Test validating a valid configuration."""
        builder = MockStrategyBuilder(temp_strategy_config_file)
        builder.load()

        assert builder.validate() is True

    def test_validate_config_missing_fields(self) -> None:
        """Test validation fails with missing required fields."""
        invalid_config = {"name": "Invalid", "enabled": True}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(invalid_config, f)
            temp_path = f.name

        try:
            builder = MockStrategyBuilder(temp_path)
            builder.load()

            with pytest.raises(ValueError, match="Missing required field"):
                builder.validate()
        finally:
            Path(temp_path).unlink()

    def test_build_strategy(self, temp_strategy_config_file: str) -> None:
        """Test building a strategy instance from config."""
        builder = MockStrategyBuilder(temp_strategy_config_file)
        strategy = builder.build()

        assert isinstance(strategy, MockStrategy)
        assert strategy.name == "Test_Strategy"
        assert len(strategy.entry_sets) == 1

    def test_load_macd_crossover_config(self) -> None:
        """Test loading the MACD crossover strategy config."""
        config_path = Path(__file__).parent.parent / "config" / "strategies" / "macd_crossover.json"

        if not config_path.exists():
            pytest.skip("MACD crossover config not found")

        builder = MockStrategyBuilder(str(config_path))
        config = builder.load()

        assert config["name"] == "MACD_Crossover_BankNifty"
        assert config["underlying"]["symbol"] == "BANKNIFTY"
        assert len(config["entry_sets"]) == 2

        # Check entry sets
        bullish_set = config["entry_sets"][0]
        assert bullish_set["name"] == "MACD_Bullish_Cross"
        assert bullish_set["signal"] == "BUY"
        assert len(bullish_set["conditions"]) == 4

        bearish_set = config["entry_sets"][1]
        assert bearish_set["name"] == "MACD_Bearish_Cross"
        assert bearish_set["signal"] == "SELL"


class TestConditionEvaluator:
    """Test ConditionEvaluator comparison logic."""

    @pytest.fixture
    def sample_data(self) -> dict[str, pd.DataFrame]:
        """Create sample market data for testing."""
        dates = pd.date_range(start="2024-01-01", periods=50, freq="5min")

        # Create data with known values
        df_5min = pd.DataFrame({
            "open": [100.0] * 50,
            "high": [105.0] * 50,
            "low": [95.0] * 50,
            "close": [100.0 + i * 0.5 for i in range(50)],  # Rising prices
            "volume": [1000] * 50,
            "RSI": [25.0] * 25 + [75.0] * 25,  # Starts oversold, becomes overbought
            "MACD": [0.0] * 40 + [1.0] * 10,
            "MACD_Signal": [0.0] * 45 + [0.5] * 5,
            "VWAP": [100.0] * 50,
        }, index=dates)

        df_1min = pd.DataFrame({
            "open": [100.0] * 50,
            "high": [105.0] * 50,
            "low": [95.0] * 50,
            "close": [102.0] * 50,  # Above VWAP
            "volume": [200] * 50,
            "VWAP": [100.0] * 50,
        }, index=dates)

        return {"5min": df_5min, "1min": df_1min}

    def test_evaluate_simple_comparison(self, sample_data: dict[str, pd.DataFrame]) -> None:
        """Test evaluating simple comparison conditions."""
        evaluator = ConditionEvaluator(sample_data)

        # Test RSI > 50 (last value is 75)
        condition = {
            "indicator": "RSI",
            "comparison": ">",
            "reference": "constant",
            "constant_value": 50,
            "timeframe": "5min",
        }
        assert evaluator.evaluate(condition) == True

        # Test RSI < 30 (last value is 75)
        condition = {
            "indicator": "RSI",
            "comparison": "<",
            "reference": "constant",
            "constant_value": 30,
            "timeframe": "5min",
        }
        assert evaluator.evaluate(condition) == False

    def test_evaluate_indicator_vs_indicator(self, sample_data: dict[str, pd.DataFrame]) -> None:
        """Test comparing two indicators."""
        evaluator = ConditionEvaluator(sample_data)

        # Test MACD > MACD_Signal (last values: 1.0 > 0.5)
        condition = {
            "indicator": "MACD",
            "comparison": ">",
            "reference": "MACD_Signal",
            "timeframe": "5min",
        }
        assert evaluator.evaluate(condition) == True

    def test_evaluate_crosses_above(self, sample_data: dict[str, pd.DataFrame]) -> None:
        """Test crosses_above detection."""
        # Create data with a crossover at the last two bars (iloc[-2] below, iloc[-1] above)
        dates = pd.date_range(start="2024-01-01", periods=10, freq="5min")
        df = pd.DataFrame({
            "close": [100.0, 100.0, 100.0, 100.0, 100.0, 101.0, 102.0, 103.0, 104.0, 105.0],
            "sma_fast": [99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.5, 100.5],
            "sma_slow": [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
        }, index=dates)

        evaluator = ConditionEvaluator({"5min": df})

        condition = {
            "indicator": "sma_fast",
            "comparison": "crosses_above",
            "reference": "sma_slow",
            "timeframe": "5min",
        }

        # Should detect crossover at last bar
        assert evaluator.evaluate(condition) == True

    def test_evaluate_crosses_below(self, sample_data: dict[str, pd.DataFrame]) -> None:
        """Test crosses_below detection."""
        # Create data with a crossunder at the last two bars (iloc[-2] above, iloc[-1] below)
        dates = pd.date_range(start="2024-01-01", periods=10, freq="5min")
        df = pd.DataFrame({
            "close": [100.0] * 10,
            "sma_fast": [105.0, 105.0, 105.0, 105.0, 105.0, 105.0, 105.0, 105.0, 100.5, 99.5],
            "sma_slow": [100.0] * 10,
        }, index=dates)

        evaluator = ConditionEvaluator({"5min": df})

        condition = {
            "indicator": "sma_fast",
            "comparison": "crosses_below",
            "reference": "sma_slow",
            "timeframe": "5min",
        }

        # Should detect crossunder at last bar
        assert evaluator.evaluate(condition) == True

    def test_evaluate_missing_timeframe(self, sample_data: dict[str, pd.DataFrame]) -> None:
        """Test evaluation fails gracefully with missing timeframe."""
        evaluator = ConditionEvaluator(sample_data)

        condition = {
            "indicator": "RSI",
            "comparison": ">",
            "reference": "constant",
            "constant_value": 50,
            "timeframe": "15min",  # Not in data
        }
        assert evaluator.evaluate(condition) is False

    def test_evaluate_spot_price_vs_vwap(self, sample_data: dict[str, pd.DataFrame]) -> None:
        """Test Spot_Price vs VWAP comparison."""
        evaluator = ConditionEvaluator(sample_data)

        # Test Spot_Price > VWAP on 1min (close=102, VWAP=100)
        condition = {
            "indicator": "Spot_Price",
            "comparison": ">",
            "reference": "VWAP",
            "timeframe": "1min",
        }
        assert evaluator.evaluate(condition) == True

        # Test Spot_Price < VWAP (should be false)
        condition = {
            "indicator": "Spot_Price",
            "comparison": "<",
            "reference": "VWAP",
            "timeframe": "1min",
        }
        assert evaluator.evaluate(condition) == False


class TestSetEvaluator:
    """Test SetEvaluator signal generation."""

    @pytest.fixture
    def mock_evaluator(self) -> ConditionEvaluator:
        """Create a mock condition evaluator."""
        mock = MagicMock(spec=ConditionEvaluator)
        return mock

    def test_evaluate_set_all_conditions_met(self, mock_evaluator: ConditionEvaluator) -> None:
        """Test signal generated when all conditions are met."""
        mock_evaluator.evaluate.return_value = True

        set_evaluator = SetEvaluator(mock_evaluator)

        entry_set = {
            "name": "Test_Bullish",
            "signal": "CE",
            "conditions": [{}, {}, {}],  # Three conditions
        }

        signal = set_evaluator.evaluate_set(entry_set)

        assert signal is not None
        assert signal["set_name"] == "Test_Bullish"
        assert signal["signal"] == "CE"
        assert mock_evaluator.evaluate.call_count == 3

    def test_evaluate_set_condition_not_met(self, mock_evaluator: ConditionEvaluator) -> None:
        """Test no signal when a condition fails."""
        # First two conditions pass, third fails
        mock_evaluator.evaluate.side_effect = [True, True, False]

        set_evaluator = SetEvaluator(mock_evaluator)

        entry_set = {
            "name": "Test_Bullish",
            "signal": "CE",
            "conditions": [{}, {}, {}],
        }

        signal = set_evaluator.evaluate_set(entry_set)

        assert signal is None
        assert mock_evaluator.evaluate.call_count == 3

    def test_evaluate_set_first_condition_fails(self, mock_evaluator: ConditionEvaluator) -> None:
        """Test evaluation stops at first failing condition."""
        mock_evaluator.evaluate.return_value = False

        set_evaluator = SetEvaluator(mock_evaluator)

        entry_set = {
            "name": "Test_Bullish",
            "signal": "CE",
            "conditions": [{}, {}, {}],
        }

        signal = set_evaluator.evaluate_set(entry_set)

        assert signal is None
        # Evaluation short-circuits on first failing condition
        assert mock_evaluator.evaluate.call_count == 1

    def test_evaluate_all_multiple_sets(self, mock_evaluator: ConditionEvaluator) -> None:
        """Test evaluating multiple entry sets."""
        # First set passes, second fails
        mock_evaluator.evaluate.side_effect = [
            True, True,           # First set (2 conditions)
            True, False,          # Second set (2 conditions)
        ]

        set_evaluator = SetEvaluator(mock_evaluator)

        entry_sets = [
            {"name": "Set1", "signal": "CE", "conditions": [{}, {}]},
            {"name": "Set2", "signal": "PE", "conditions": [{}, {}]},
        ]

        signals = set_evaluator.evaluate_all(entry_sets)

        assert len(signals) == 1
        assert signals[0]["set_name"] == "Set1"
        assert signals[0]["signal"] == "CE"

    def test_evaluate_all_no_signals(self, mock_evaluator: ConditionEvaluator) -> None:
        """Test evaluating when no sets generate signals."""
        mock_evaluator.evaluate.return_value = False

        set_evaluator = SetEvaluator(mock_evaluator)

        entry_sets = [
            {"name": "Set1", "signal": "CE", "conditions": [{}]},
            {"name": "Set2", "signal": "PE", "conditions": [{}]},
        ]

        signals = set_evaluator.evaluate_all(entry_sets)

        assert len(signals) == 0

    def test_evaluate_empty_conditions(self, mock_evaluator: ConditionEvaluator) -> None:
        """Test evaluating set with no conditions."""
        mock_evaluator.evaluate.return_value = True

        set_evaluator = SetEvaluator(mock_evaluator)

        entry_set = {
            "name": "Empty_Set",
            "signal": "CE",
            "conditions": [],
        }

        signal = set_evaluator.evaluate_set(entry_set)

        # Empty conditions should pass (vacuously true)
        assert signal is not None
        assert signal["set_name"] == "Empty_Set"


class TestStrategyEngineIntegration:
    """Integration tests for strategy engine components."""

    def test_full_strategy_evaluation(self, sample_strategy_config: dict[str, Any]) -> None:
        """Test complete strategy evaluation flow."""
        # Create sample data that meets conditions
        dates = pd.date_range(start="2024-01-01", periods=30, freq="5min")
        df = pd.DataFrame({
            "open": [100.0] * 30,
            "high": [105.0] * 30,
            "low": [95.0] * 30,
            "close": [100.0] * 30,
            "volume": [1000] * 30,
            "RSI": [35.0] * 14 + [25.0] * 16,  # RSI drops below 30
        }, index=dates)

        data = {"5min": df}

        # Evaluate conditions
        evaluator = ConditionEvaluator(data)
        set_evaluator = SetEvaluator(evaluator)

        signals = set_evaluator.evaluate_all(sample_strategy_config["entry_sets"])

        # Should generate signal since RSI < 30
        assert len(signals) == 1
        assert signals[0]["signal"] == "CE"

    def test_macd_strategy_conditions(self) -> None:
        """Test MACD strategy entry set conditions."""
        # Create data with MACD crossover
        dates = pd.date_range(start="2024-01-01", periods=50, freq="5min")

        # MACD starts below signal, then crosses above at last two bars
        macd_values = [0.0] * 50
        signal_values = [2.0] * 50

        # Make crossover happen at the last bar (index 48 below, index 49 above)
        macd_values[48] = 1.5   # Second-to-last: MACD below signal
        signal_values[48] = 2.0
        macd_values[49] = 2.5   # Last bar: MACD above signal
        signal_values[49] = 2.0

        df_5min = pd.DataFrame({
            "open": [100.0] * 50,
            "high": [105.0] * 50,
            "low": [95.0] * 50,
            "close": [100.0] * 50,
            "volume": [1000] * 50,
            "RSI": [55.0] * 50,  # RSI > 50
            "MACD": macd_values,
            "MACD_Signal": signal_values,
        }, index=dates)

        df_1min = pd.DataFrame({
            "open": [101.0] * 50,
            "high": [106.0] * 50,
            "low": [96.0] * 50,
            "close": [101.0] * 50,
            "volume": [200] * 50,
            "VWAP": [100.0] * 50,
        }, index=dates)

        data = {"5min": df_5min, "1min": df_1min}

        evaluator = ConditionEvaluator(data)

        # Test MACD crosses above Signal
        condition1 = {
            "indicator": "MACD",
            "comparison": "crosses_above",
            "reference": "MACD_Signal",
            "timeframe": "5min",
        }
        assert evaluator.evaluate(condition1) == True

        # Test RSI > 50
        condition2 = {
            "indicator": "RSI",
            "comparison": ">",
            "reference": "constant",
            "constant_value": 50,
            "timeframe": "5min",
        }
        assert evaluator.evaluate(condition2) == True

        # Test Spot_Price > VWAP
        condition3 = {
            "indicator": "Spot_Price",
            "comparison": ">",
            "reference": "VWAP",
            "timeframe": "1min",
        }
        assert evaluator.evaluate(condition3) == True
