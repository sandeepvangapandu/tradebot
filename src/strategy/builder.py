"""Strategy configuration builder.

Defines Pydantic models for strategy configuration and a builder class
for loading strategies from YAML/JSON files.
"""

from datetime import time
from enum import Enum
from pathlib import Path
from typing import Any, Literal, Optional, Union

import yaml
from loguru import logger
from pydantic import BaseModel, Field, field_validator, model_validator


class ComparisonOperator(str, Enum):
    """Supported comparison operators for conditions."""

    GT = ">"
    LT = "<"
    GTE = ">="
    LTE = "<="
    EQ = "=="
    CROSSES_ABOVE = "crosses_above"
    CROSSES_BELOW = "crosses_below"
    NEAR = "near"


class IndicatorType(str, Enum):
    """Supported indicator types."""

    EMA = "ema"
    SMA = "sma"
    RSI = "rsi"
    MACD = "macd"
    BBANDS = "bbands"
    ATR = "atr"
    SUPERTREND = "supertrend"
    VWAP = "vwap"
    CLOSE = "close"
    OPEN = "open"
    HIGH = "high"
    LOW = "low"
    VOLUME = "volume"


class SignalType(str, Enum):
    """Supported signal types for options and equity."""

    BUY_CE = "BUY CE"  # Buy Call Option
    BUY_PE = "BUY PE"  # Buy Put Option
    BUY = "BUY"  # Buy Equity/Futures
    SELL = "SELL"  # Sell Equity/Futures
    STRADDLE = "STRADDLE"  # Sell Option Straddle


class PositionSizingMethod(str, Enum):
    """Position sizing methods."""

    FIXED_QUANTITY = "fixed_quantity"
    FIXED_CAPITAL = "fixed_capital"
    RISK_BASED = "risk_based"
    PERCENT_OF_EQUITY = "percent_of_equity"


class IndicatorRef(BaseModel):
    """Reference to an indicator with optional parameters.

    Used for defining indicator comparisons in conditions.

    Examples:
        Simple indicator reference:

            {"indicator": "EMA", "period": 20}

        With custom parameters:

            {"indicator": "MACD", "params": {"fast": 12, "slow": 26, "signal": 9}}
    """

    indicator: IndicatorType = Field(
        ..., description="Indicator type (e.g. 'ema', 'rsi', 'macd')"
    )
    period: Optional[int] = Field(
        default=None, description="Indicator period (e.g. 20 for EMA 20)"
    )
    params: dict[str, Any] = Field(
        default_factory=dict, description="Additional indicator parameters"
    )

    @field_validator("indicator", mode="before")
    @classmethod
    def validate_indicator(cls, v: Any) -> IndicatorType:
        if isinstance(v, str):
            return IndicatorType(v.lower())
        return v


class Condition(BaseModel):
    """A single condition for entry/exit evaluation.

    Matches the JSON strategy format. Each condition compares an indicator
    against either another indicator (via ``against``) or a constant (via
    ``value``).

    Examples:
        Indicator vs indicator::

            {"indicator": "MACD", "comparison": "crosses_above",
             "against": "MACD_Signal", "timeframe": "5min",
             "parameters": {"fast": 12, "slow": 26, "signal": 9}}

        Indicator vs constant value::

            {"indicator": "RSI", "comparison": ">", "value": 50,
             "timeframe": "5min", "parameters": {"length": 14}}
    """

    indicator: str = Field(
        ..., description="Primary indicator or price field (e.g. 'MACD', 'RSI', 'Spot_Price')"
    )
    comparison: ComparisonOperator = Field(
        ..., description="Comparison operator (e.g. '>', 'crosses_above')"
    )
    against: Optional[str] = Field(
        default=None,
        description="Reference indicator for indicator-vs-indicator comparison (e.g. 'MACD_Signal', 'VWAP')",
    )
    value: Optional[Union[float, int, str]] = Field(
        default=None,
        description="Constant value for indicator-vs-value comparison (e.g. 50 or 'bullish')",
    )
    timeframe: str = Field(
        default="5min", description="Timeframe for this condition (e.g. '5min', '1min')"
    )
    lookback: int = Field(
        default=0, description="How many bars back to check (0 = current bar)"
    )
    parameters: dict[str, Any] = Field(
        default_factory=dict,
        description="Indicator parameters (e.g. {'fast': 12, 'slow': 26, 'signal': 9})",
    )

    @field_validator("comparison", mode="before")
    @classmethod
    def validate_comparison(cls, v: Any) -> ComparisonOperator:
        if isinstance(v, str):
            mapping = {
                ">": ComparisonOperator.GT,
                "<": ComparisonOperator.LT,
                ">=": ComparisonOperator.GTE,
                "<=": ComparisonOperator.LTE,
                "==": ComparisonOperator.EQ,
                "gt": ComparisonOperator.GT,
                "lt": ComparisonOperator.LT,
                "gte": ComparisonOperator.GTE,
                "lte": ComparisonOperator.LTE,
                "eq": ComparisonOperator.EQ,
                "crosses_above": ComparisonOperator.CROSSES_ABOVE,
                "crosses_below": ComparisonOperator.CROSSES_BELOW,
            }
            return mapping.get(v.lower(), ComparisonOperator(v))
        return v

    @model_validator(mode="after")
    def validate_operand(self) -> "Condition":
        """Ensure exactly one of ``against`` or ``value`` is set.

        Exception: the ``near`` operator requires both ``against`` (reference
        indicator) and ``value`` (tolerance as a fraction, e.g. 0.002 = 0.2%).
        """
        if self.against is None and self.value is None:
            raise ValueError(
                "Condition must specify either 'against' (indicator) or 'value' (constant)"
            )
        # 'near' needs both against (reference) and value (tolerance)
        if self.comparison == ComparisonOperator.NEAR:
            if self.against is None or self.value is None:
                raise ValueError(
                    "'near' comparison requires both 'against' (indicator) and 'value' (tolerance)"
                )
            return self
        if self.against is not None and self.value is not None:
            raise ValueError(
                "Condition must specify only one of 'against' or 'value', not both"
            )
        return self


class EntrySet(BaseModel):
    """A set of conditions that must all be true for entry."""

    name: str = Field(..., description="Name of this entry set")
    signal: str = Field(
        ..., description="Signal to generate (e.g. 'CE', 'PE', 'BUY', 'SELL')"
    )
    conditions: list[Condition] = Field(
        ..., description="All conditions must be true for entry"
    )
    priority: int = Field(
        default=0, description="Higher priority sets are evaluated first"
    )


class ExitRules(BaseModel):
    """Exit rules for a strategy."""

    stop_loss_pct: Optional[Union[int, float]] = Field(
        default=None,
        description="Stop loss as percentage of entry price",
    )
    target_pct: Optional[Union[int, float]] = Field(
        default=None,
        description="Target as percentage of entry price",
    )
    trailing_stop_loss: Optional[dict[str, Any]] = Field(
        default=None,
        description="Trailing SL config: {enabled, activation_pct, trail_pct}",
    )
    time_based_exit: Optional[str] = Field(
        default=None,
        description="Exit all positions at this time (e.g., '15:10:00')",
    )
    max_holding_minutes: Optional[int] = Field(
        default=None,
        description="Maximum holding time in minutes",
    )
    exit_conditions: list[Condition] = Field(
        default_factory=list,
        description="Additional conditions that trigger exit",
    )


class PositionSizing(BaseModel):
    """Position sizing configuration."""

    method: PositionSizingMethod = PositionSizingMethod.FIXED_QUANTITY
    quantity: Optional[int] = None
    capital_per_trade: Optional[int] = None  # In PAISA
    risk_per_trade_percent: Optional[float] = None  # e.g., 1.0 for 1%
    percent_of_equity: Optional[float] = None  # e.g., 10.0 for 10%
    max_quantity: Optional[int] = None
    min_quantity: Optional[int] = 1

    @model_validator(mode="before")
    @classmethod
    def normalize_field_names(cls, data: Any) -> Any:
        """Normalize field names from config files."""
        if isinstance(data, dict):
            # Handle max_risk_per_trade_pct -> risk_per_trade_percent
            if "max_risk_per_trade_pct" in data and "risk_per_trade_percent" not in data:
                data["risk_per_trade_percent"] = data.pop("max_risk_per_trade_pct")
        return data

    @model_validator(mode="after")
    def validate_params(self) -> "PositionSizing":
        """Validate that required params are present for each method."""
        if self.method == PositionSizingMethod.FIXED_QUANTITY:
            if self.quantity is None:
                raise ValueError("quantity is required for fixed_quantity method")
        elif self.method == PositionSizingMethod.FIXED_CAPITAL:
            if self.capital_per_trade is None:
                raise ValueError(
                    "capital_per_trade is required for fixed_capital method"
                )
        elif self.method == PositionSizingMethod.RISK_BASED:
            if self.risk_per_trade_percent is None:
                raise ValueError(
                    "risk_per_trade_percent is required for risk_based method"
                )
        elif self.method == PositionSizingMethod.PERCENT_OF_EQUITY:
            if self.percent_of_equity is None:
                raise ValueError(
                    "percent_of_equity is required for percent_of_equity method"
                )
        return self


class InstrumentSelection(BaseModel):
    """Instrument selection criteria."""

    # For equity/F&O
    symbols: list[str] = Field(
        default_factory=list, description="List of symbols to trade"
    )
    # For options
    underlying: Optional[str] = Field(
        default=None, description="Underlying symbol for options"
    )
    option_type: Optional[Literal["CE", "PE"]] = None
    strike_selection: Optional[dict[str, Any]] = Field(
        default=None,
        description="ATM, ITM, OTM selection with offset",
    )
    expiry_selection: Optional[str] = Field(
        default=None, description="WEEKLY, MONTHLY, or specific date"
    )
    # Filters
    min_price: Optional[int] = None  # In PAISA
    max_price: Optional[int] = None
    min_volume: Optional[int] = None


class TradingHours(BaseModel):
    """Trading hours for the strategy."""

    start_time: str = Field(default="09:15:00", description="Market open time (HH:MM:SS)")
    end_time: str = Field(default="15:15:00", description="Market close/square-off time (HH:MM:SS)")
    timezone: str = Field(default="Asia/Kolkata", description="Timezone for trading hours")
    days: list[int] = Field(
        default_factory=lambda: [0, 1, 2, 3, 4],
        description="Days of week: 0=Monday, 6=Sunday",
    )

    @field_validator("days")
    @classmethod
    def validate_days(cls, v: list[int]) -> list[int]:
        for day in v:
            if not 0 <= day <= 6:
                raise ValueError(f"Day must be 0-6, got {day}")
        return v

    @property
    def start(self) -> time:
        """Parse start_time string to time object."""
        parts = self.start_time.split(":")
        return time(int(parts[0]), int(parts[1]), int(parts[2]) if len(parts) > 2 else 0)

    @property
    def end(self) -> time:
        """Parse end_time string to time object."""
        parts = self.end_time.split(":")
        return time(int(parts[0]), int(parts[1]), int(parts[2]) if len(parts) > 2 else 0)


class StrategyConfig(BaseModel):
    """Complete strategy configuration.

    This model defines all parameters for a trading strategy including
    entry conditions, exit rules, position sizing, and instrument selection.
    Designed to match the JSON strategy file format.
    """

    name: str = Field(..., description="Unique strategy name")
    version: str = Field(default="1.0.0", description="Strategy version")
    description: Optional[str] = None
    enabled: bool = Field(default=True, description="Whether strategy is enabled")

    # Underlying instrument
    underlying: Optional[dict[str, Any]] = Field(
        default=None,
        description="Underlying instrument info: {instrument_key, symbol, segment}",
    )

    # Trading parameters
    trading_hours: TradingHours = Field(default_factory=TradingHours)
    instrument_selection: Optional[Union[InstrumentSelection, dict[str, Any]]] = Field(
        default=None, description="Instrument selection criteria"
    )

    # Timeframes
    timeframes: dict[str, str] = Field(
        default_factory=lambda: {"primary": "5min"},
        description="Timeframe definitions: {primary: '5min', secondary: '1min'}",
    )

    # Entry and exit
    entry_sets: list[EntrySet] = Field(
        ..., description="Entry condition sets"
    )
    exit_rules: ExitRules = Field(default_factory=ExitRules)

    # Position management
    position_sizing: PositionSizing = Field(default_factory=PositionSizing)

    # Risk management
    risk_management: dict[str, Any] = Field(
        default_factory=dict,
        description="Risk management overrides: {max_open_positions, avoid_overnight, ...}",
    )

    # Additional params
    params: dict[str, Any] = Field(default_factory=dict)

    @property
    def active(self) -> bool:
        """Alias for enabled (backward compat)."""
        return self.enabled

    @property
    def timeframe(self) -> str:
        """Primary timeframe (backward compat)."""
        return self.timeframes.get("primary", "5min")

    @property
    def max_positions(self) -> int:
        """Max concurrent positions from risk_management."""
        return self.risk_management.get("max_open_positions", 5)

    @property
    def max_trades_per_day(self) -> Optional[int]:
        """Max trades per day from risk_management."""
        return self.risk_management.get("max_trades_per_day")


class StrategyBuilder:
    """Builder class for loading and validating strategy configurations.

    Supports YAML and JSON formats.
    """

    @staticmethod
    def load_strategy(path: Union[str, Path]) -> StrategyConfig:
        """Load a strategy configuration from a file.

        Args:
            path: Path to YAML or JSON file.

        Returns:
            Validated StrategyConfig instance.

        Raises:
            FileNotFoundError: If file doesn't exist.
            ValueError: If file format is invalid.
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Strategy file not found: {path}")

        logger.info(f"Loading strategy from {path}")

        content = path.read_text(encoding="utf-8")

        if path.suffix in (".yaml", ".yml"):
            data = yaml.safe_load(content)
        elif path.suffix == ".json":
            import json

            data = json.loads(content)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")

        config = StrategyConfig.model_validate(data)
        logger.info(f"Loaded strategy '{config.name}' v{config.version}")

        return config

    @staticmethod
    def load_strategies_from_directory(directory: Union[str, Path]) -> list[StrategyConfig]:
        """Load all strategy configurations from a directory.

        Args:
            directory: Path to directory containing strategy files.

        Returns:
            List of validated StrategyConfig instances.
        """
        directory = Path(directory)

        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")

        configs = []
        for ext in ("*.yaml", "*.yml", "*.json"):
            for file_path in directory.glob(ext):
                try:
                    config = StrategyBuilder.load_strategy(file_path)
                    configs.append(config)
                except Exception as e:
                    logger.error(f"Failed to load strategy from {file_path}: {e}")

        logger.info(f"Loaded {len(configs)} strategies from {directory}")
        return configs

    @staticmethod
    def save_strategy(config: StrategyConfig, path: Union[str, Path]) -> None:
        """Save a strategy configuration to a file.

        Args:
            config: StrategyConfig to save.
            path: Path to save to (YAML or JSON based on extension).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = config.model_dump(mode="json")

        if path.suffix in (".yaml", ".yml"):
            content = yaml.dump(data, default_flow_style=False, sort_keys=False)
        elif path.suffix == ".json":
            import json

            content = json.dumps(data, indent=2)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")

        path.write_text(content, encoding="utf-8")
        logger.info(f"Saved strategy '{config.name}' to {path}")

    @staticmethod
    def create_sample_strategy() -> StrategyConfig:
        """Create a sample EMA crossover strategy for reference."""
        return StrategyConfig(
            name="ema_crossover",
            description="EMA 9/21 crossover strategy for NSE equity",
            enabled=True,
            underlying={
                "instrument_key": "NSE_EQ|RELIANCE",
                "symbol": "RELIANCE",
                "segment": "NSE_EQ",
            },
            trading_hours=TradingHours(
                start_time="09:15:00",
                end_time="15:15:00",
                timezone="Asia/Kolkata",
                days=[0, 1, 2, 3, 4],
            ),
            instrument_selection=InstrumentSelection(
                symbols=["RELIANCE", "TCS", "INFY"],
                min_volume=100000,
            ),
            timeframes={"primary": "5min", "secondary": "1min"},
            entry_sets=[
                EntrySet(
                    name="bullish_crossover",
                    signal="BUY",
                    conditions=[
                        Condition(
                            indicator="EMA_9",
                            comparison=ComparisonOperator.CROSSES_ABOVE,
                            against="EMA_21",
                            timeframe="5min",
                            parameters={"period": 9},
                        ),
                        Condition(
                            indicator="Spot_Price",
                            comparison=ComparisonOperator.GT,
                            against="VWAP",
                            timeframe="5min",
                        ),
                    ],
                ),
                EntrySet(
                    name="bearish_crossover",
                    signal="SELL",
                    conditions=[
                        Condition(
                            indicator="EMA_9",
                            comparison=ComparisonOperator.CROSSES_BELOW,
                            against="EMA_21",
                            timeframe="5min",
                            parameters={"period": 9},
                        ),
                        Condition(
                            indicator="Spot_Price",
                            comparison=ComparisonOperator.LT,
                            against="VWAP",
                            timeframe="5min",
                        ),
                    ],
                ),
            ],
            exit_rules=ExitRules(
                stop_loss_pct=2.0,
                target_pct=4.0,
                time_based_exit="15:15:00",
            ),
            position_sizing=PositionSizing(
                method=PositionSizingMethod.RISK_BASED,
                risk_per_trade_percent=1.0,
                max_quantity=100,
            ),
            risk_management={"max_open_positions": 3},
        )
