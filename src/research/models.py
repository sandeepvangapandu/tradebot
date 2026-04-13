"""Pydantic models for the AI Trade Research & Decision Engine.

All monetary values are in PAISA (integer) - 1 Rupee = 100 paisa.
All timestamps are in IST (Asia/Kolkata).
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class MarketRegime(str, Enum):
    """Market regime classification."""

    STRONG_UPTREND = "strong_uptrend"
    WEAK_UPTREND = "weak_uptrend"
    RANGING = "ranging"
    WEAK_DOWNTREND = "weak_downtrend"
    STRONG_DOWNTREND = "strong_downtrend"
    VOLATILE = "volatile"  # high ATR, no clear direction
    QUIET = "quiet"  # low ATR, compressed range


class VolatilityRegime(str, Enum):
    """Volatility regime classification."""

    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    EXTREME = "extreme"


class SignalStrength(str, Enum):
    """Signal strength classification."""

    STRONG_BUY = "strong_buy"
    BUY = "buy"
    WEAK_BUY = "weak_buy"
    NEUTRAL = "neutral"
    WEAK_SELL = "weak_sell"
    SELL = "sell"
    STRONG_SELL = "strong_sell"


class TradeVerdict(str, Enum):
    """Final verdict on whether to execute a trade."""

    EXECUTE = "EXECUTE"  # Full size
    REDUCE_SIZE = "REDUCE_SIZE"  # Reduced position size
    SKIP = "SKIP"  # Do not trade


class OptimalEntryType(str, Enum):
    """Optimal entry order type recommendation."""

    MARKET = "MARKET"
    LIMIT_AT_SUPPORT = "LIMIT_AT_SUPPORT"
    WAIT_FOR_PULLBACK = "WAIT_FOR_PULLBACK"


class AnalysisComponent(BaseModel):
    """Individual analysis result from a specific analyzer."""

    name: str = Field(..., description="Name of the analyzer")
    score: float = Field(..., ge=0, le=100, description="Score 0-100")
    weight: float = Field(..., ge=0, le=1, description="Weight in final score 0.0-1.0")
    confidence: float = Field(
        default=100.0, ge=0, le=100, description="Confidence in this analysis 0-100"
    )
    reasoning: str = Field(..., description="Human-readable explanation")
    data: dict[str, Any] = Field(default_factory=dict, description="Raw data backing analysis")


class TrendAlignmentData(BaseModel):
    """Detailed trend alignment data."""

    trend_1min: str = Field(default="neutral", description="1min trend direction")
    trend_5min: str = Field(default="neutral", description="5min trend direction")
    trend_15min: str = Field(default="neutral", description="15min trend direction")
    trend_daily: str = Field(default="neutral", description="Daily trend direction")
    adx_1min: float = Field(default=0, description="1min ADX value")
    adx_5min: float = Field(default=0, description="5min ADX value")
    adx_15min: float = Field(default=0, description="15min ADX value")
    adx_daily: float = Field(default=0, description="Daily ADX value")


class MomentumData(BaseModel):
    """Momentum analysis data."""

    rsi_current: float = Field(default=50, description="Current RSI value")
    rsi_trend: str = Field(default="flat", description="RSI trend direction")
    macd_histogram: float = Field(default=0, description="MACD histogram value")
    macd_histogram_trend: str = Field(default="flat", description="MACD histogram trend")
    stochastic_k: float = Field(default=50, description="Stochastic %K")
    stochastic_d: float = Field(default=50, description="Stochastic %D")
    roc_10: float = Field(default=0, description="10-period rate of change")
    divergence_detected: bool = Field(default=False, description="Bullish/bearish divergence")
    divergence_type: str | None = Field(default=None, description="Type of divergence")


class SupportResistanceData(BaseModel):
    """Support and resistance analysis data."""

    current_price: int = Field(default=0, description="Current price in paisa")
    pdh: int | None = Field(default=None, description="Previous day high")
    pdl: int | None = Field(default=None, description="Previous day low")
    pdc: int | None = Field(default=None, description="Previous day close")
    vwap: int | None = Field(default=None, description="VWAP")
    vwap_upper_1std: int | None = Field(default=None, description="VWAP + 1 std")
    vwap_lower_1std: int | None = Field(default=None, description="VWAP - 1 std")
    pivot_point: int | None = Field(default=None, description="Pivot point")
    r1: int | None = Field(default=None, description="Resistance 1")
    r2: int | None = Field(default=None, description="Resistance 2")
    s1: int | None = Field(default=None, description="Support 1")
    s2: int | None = Field(default=None, description="Support 2")
    nearest_support_distance_pct: float = Field(default=0, description="Distance to nearest support")
    nearest_resistance_distance_pct: float = Field(default=0, description="Distance to nearest resistance")


class VolumeData(BaseModel):
    """Volume analysis data."""

    current_volume: int = Field(default=0, description="Current candle volume")
    avg_volume_20: int = Field(default=0, description="20-period average volume")
    volume_ratio: float = Field(default=1.0, description="Current/avg volume ratio")
    obv_trend: str = Field(default="flat", description="OBV trend direction")
    volume_trend: str = Field(default="flat", description="Volume trend direction")
    bid_ask_spread_pct: float | None = Field(default=None, description="Bid-ask spread %")
    oi_current: int | None = Field(default=None, description="Open interest")
    oi_change_pct: float | None = Field(default=None, description="OI change %")


class OptionsData(BaseModel):
    """Options-specific analysis data."""

    iv_current: float | None = Field(default=None, description="Current implied volatility")
    iv_percentile: float | None = Field(default=None, description="IV percentile 0-100")
    iv_rank: float | None = Field(default=None, description="IV rank 0-100")
    delta: float | None = Field(default=None, description="Option delta")
    theta: float | None = Field(default=None, description="Option theta")
    gamma: float | None = Field(default=None, description="Option gamma")
    vega: float | None = Field(default=None, description="Option vega")
    theta_pct_of_premium: float | None = Field(default=None, description="Theta as % of premium")
    pcr_ratio: float | None = Field(default=None, description="Put-call ratio")
    max_pain: int | None = Field(default=None, description="Max pain price")
    days_to_expiry: int | None = Field(default=None, description="Days to expiry")
    is_weekly_expiry: bool = Field(default=False, description="Is weekly expiry")
    is_monthly_expiry: bool = Field(default=False, description="Is monthly expiry")


class CorrelationData(BaseModel):
    """Correlation analysis data."""

    nifty_correlation_5d: float = Field(default=0, description="5-day correlation with Nifty")
    nifty_correlation_20d: float = Field(default=0, description="20-day correlation with Nifty")
    sector_correlation_5d: float = Field(default=0, description="5-day correlation with sector")
    relative_strength_vs_nifty: float = Field(default=0, description="RS vs Nifty")
    relative_strength_vs_sector: float = Field(default=0, description="RS vs sector")
    sector_performance_5d: float = Field(default=0, description="Sector 5-day return %")
    market_breadth: str = Field(default="neutral", description="Market breadth")


class EventData(BaseModel):
    """Event calendar data."""

    is_weekly_expiry: bool = Field(default=False, description="Is weekly expiry day")
    is_monthly_expiry: bool = Field(default=False, description="Is monthly expiry")
    days_to_monthly_expiry: int | None = Field(default=None, description="Days to monthly expiry")
    is_rbi_policy_day: bool = Field(default=False, description="Is RBI policy day")
    is_fed_day: bool = Field(default=False, description="Is Fed meeting day")
    is_budget_day: bool = Field(default=False, description="Is Budget day")
    is_earnings_season: bool = Field(default=False, description="Is earnings season")
    time_of_day_score: float = Field(default=50, description="Time of day quality score")
    event_risk_level: str = Field(default="low", description="low/medium/high/extreme")


class PatternData(BaseModel):
    """Pattern recognition data."""

    pattern_detected: str | None = Field(default=None, description="Detected pattern name")
    pattern_type: str | None = Field(default=None, description="reversal/continuation")
    pattern_strength: float = Field(default=0, description="Pattern strength 0-100")
    pattern_timeframe: str = Field(default="5m", description="Timeframe of pattern")
    confirms_signal: bool = Field(default=False, description="Pattern confirms signal direction")
    contradicts_signal: bool = Field(default=False, description="Pattern contradicts signal")


class TradeResearchReport(BaseModel):
    """Complete research output for a trading signal.

    This is the main output of the research engine containing all analysis
    and the final verdict on whether to execute the trade.
    """

    signal_id: str = Field(..., description="Unique signal identifier")
    timestamp: datetime = Field(..., description="Analysis timestamp")
    instrument_key: str = Field(..., description="Instrument to trade")
    direction: str = Field(..., description="BUY or SELL")
    strategy_name: str = Field(..., description="Originating strategy")

    # Scores
    final_score: float = Field(..., ge=0, le=100, description="Weighted composite score 0-100")
    components: list[AnalysisComponent] = Field(
        default_factory=list, description="Individual analysis scores"
    )

    # Detailed analysis data
    trend_data: TrendAlignmentData = Field(default_factory=TrendAlignmentData)
    momentum_data: MomentumData = Field(default_factory=MomentumData)
    sr_data: SupportResistanceData | None = Field(default=None, description="S/R analysis")
    volume_data: VolumeData = Field(default_factory=VolumeData)
    options_data: OptionsData | None = Field(default=None, description="Options analysis")
    correlation_data: CorrelationData = Field(default_factory=CorrelationData)
    event_data: EventData = Field(default_factory=EventData)
    pattern_data: PatternData = Field(default_factory=PatternData)

    # Verdicts
    verdict: TradeVerdict = Field(..., description="EXECUTE, REDUCE_SIZE, or SKIP")
    signal_strength: SignalStrength = Field(..., description="Signal strength classification")
    market_regime: MarketRegime = Field(..., description="Detected market regime")
    volatility_regime: VolatilityRegime = Field(..., description="Volatility regime")

    # Adjustments
    suggested_quantity_multiplier: float = Field(
        default=1.0, ge=0, le=2.0, description="Position size multiplier"
    )
    suggested_sl_adjustment: float = Field(
        default=1.0, ge=0.5, le=2.0, description="SL distance multiplier"
    )
    suggested_target_adjustment: float = Field(
        default=1.0, ge=0.5, le=2.0, description="Target distance multiplier"
    )
    optimal_entry_type: OptimalEntryType = Field(
        default=OptimalEntryType.MARKET, description="Recommended entry type"
    )

    # Context
    key_risks: list[str] = Field(default_factory=list, description="Top risk factors")
    key_supports: list[str] = Field(default_factory=list, description="Top supporting factors")
    summary: str = Field(default="", description="2-3 sentence summary")

    # Execution tracking
    was_executed: bool | None = Field(default=None, description="Whether trade was executed")
    execution_result: dict[str, Any] | None = Field(
        default=None, description="Execution outcome data"
    )

    # Degradation tracking
    degraded_analyzers: list[str] = Field(
        default_factory=list,
        description="List of analyzer names that returned default/degraded values"
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        return self.model_dump(mode="json")


class ResearchConfig(BaseModel):
    """Configuration for the research module."""

    enabled: bool = Field(default=True, description="Whether research is enabled")
    min_score: int = Field(default=65, description="Minimum score to execute")
    score_for_full_size: int = Field(default=75, description="Score for full position size")
    max_analysis_time_seconds: float = Field(
        default=2.0, description="Maximum time allowed for analysis"
    )
    log_all: bool = Field(default=True, description="Log even SKIP decisions")
    weights: dict[str, float] = Field(default_factory=dict, description="Custom weights")

    # Cache settings
    regime_cache_minutes: int = Field(default=5, description="Market regime cache duration")
    sr_cache_minutes: int = Field(default=180, description="S/R levels cache duration")
    iv_cache_minutes: int = Field(default=15, description="IV percentile cache duration")
    correlation_cache_minutes: int = Field(default=30, description="Correlation cache duration")
