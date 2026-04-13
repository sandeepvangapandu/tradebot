"""Strategy module for the trading bot.

Provides technical indicators, strategy configuration, condition evaluation,
and the strategy execution engine.
"""

from src.strategy.builder import (
    ComparisonOperator,
    Condition,
    EntrySet,
    ExitRules,
    IndicatorRef,
    IndicatorType,
    InstrumentSelection,
    PositionSizing,
    PositionSizingMethod,
    SignalType,
    StrategyBuilder,
    StrategyConfig,
    TradingHours,
)
from src.strategy.conditions import ConditionEvaluator
from src.strategy.engine import SetEvaluator, Signal, StrategyEngine
from src.strategy.indicators import IndicatorEngine, IndicatorResult

__all__ = [
    # Indicators
    "IndicatorEngine",
    "IndicatorResult",
    # Builder
    "StrategyBuilder",
    "StrategyConfig",
    "EntrySet",
    "ExitRules",
    "Condition",
    "IndicatorRef",
    "IndicatorType",
    "ComparisonOperator",
    "SignalType",
    "InstrumentSelection",
    "PositionSizing",
    "PositionSizingMethod",
    "TradingHours",
    # Conditions
    "ConditionEvaluator",
    # Engine
    "StrategyEngine",
    "SetEvaluator",
    "Signal",
]
