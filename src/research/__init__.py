"""AI Trade Research & Decision Engine.

This module provides comprehensive multi-factor analysis for trading signals,
assigning conviction scores (0-100) and making execution decisions.

Usage:
    from src.research import TradeAnalyzer, TradeResearchReport

    analyzer = TradeAnalyzer(settings, bar_builder, historical_fetcher, instrument_manager)
    report = analyzer.analyze(signal)

    if report.verdict == TradeVerdict.EXECUTE:
        # Proceed with order
        pass
"""

from src.research.models import (
    AnalysisComponent,
    CorrelationData,
    EventData,
    MarketRegime,
    MomentumData,
    OptimalEntryType,
    OptionsData,
    PatternData,
    ResearchConfig,
    SignalStrength,
    SupportResistanceData,
    TradeResearchReport,
    TradeVerdict,
    TrendAlignmentData,
    VolatilityRegime,
    VolumeData,
)
from src.research.multi_strategy import (
    ConfluenceBasedOrderFilter,
    ConfluenceLevel,
    ConfluenceResult,
    StrategyCategory,
    StrategyConfluenceAnalyzer,
    StrategyInfo,
    StrategySignal,
)
from src.research.trade_analyzer import TradeAnalyzer

# Optional imports - don't fail if dependencies missing
try:
    from src.research.ml_validator import MLSignalValidator
except ImportError:
    MLSignalValidator = None  # type: ignore

__all__ = [
    # Main orchestrator
    "TradeAnalyzer",
    # Models
    "TradeResearchReport",
    "AnalysisComponent",
    "TradeVerdict",
    "SignalStrength",
    "MarketRegime",
    "VolatilityRegime",
    "OptimalEntryType",
    "ResearchConfig",
    # Analysis data models
    "TrendAlignmentData",
    "MomentumData",
    "SupportResistanceData",
    "VolumeData",
    "OptionsData",
    "CorrelationData",
    "EventData",
    "PatternData",
    # ML Validator
    "MLSignalValidator",
    # Multi-strategy confluence
    "StrategyConfluenceAnalyzer",
    "StrategySignal",
    "ConfluenceLevel",
    "ConfluenceResult",
    "StrategyCategory",
    "StrategyInfo",
    "ConfluenceBasedOrderFilter",
]
