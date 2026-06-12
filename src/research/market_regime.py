"""Market regime detection module for Indian markets.

Detects market regimes using ADX, ATR percentiles, Bollinger Band width,
and price position within range. Includes India-specific context like
weekly/monthly expiry detection.

All timestamps are in IST (Asia/Kolkata).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import TYPE_CHECKING, Any
from zoneinfo import ZoneInfo

import pandas as pd
from loguru import logger

try:
    import pandas_ta  # noqa: F401 — registers df.ta accessor
except Exception as exc:
    pandas_ta = None  # type: ignore[assignment]
    logger.warning("pandas_ta unavailable; using fallback indicators where possible: {}", exc)

# Add parent directory to path for imports
sys.path.insert(0, "/Users/sandeepvangapandu/Downloads/Trading")

from src.research.models import (
    AnalysisComponent,
    EventData,
    MarketRegime,
    VolatilityRegime,
)
from src.strategy.indicators import IndicatorEngine
from src.utils.degradation import graceful_degrade

IST = ZoneInfo("Asia/Kolkata")


class StrategyType(str, Enum):
    """Strategy classification for regime compatibility."""

    TREND_FOLLOWING = "trend_following"
    MEAN_REVERSION = "mean_reversion"
    BREAKOUT = "breakout"
    MOMENTUM = "momentum"
    SCALPING = "scalping"


class SignalDirection(str, Enum):
    """Signal direction."""

    LONG = "long"
    SHORT = "short"
    NEUTRAL = "neutral"


@dataclass
class RegimeData:
    """Container for regime detection results."""

    market_regime: MarketRegime
    volatility_regime: VolatilityRegime
    adx_value: float
    adx_trend: str  # "rising", "falling", "flat"
    atr_percentile: float
    atr_current: float
    atr_20d_average: float
    bb_width: float
    bb_width_percentile: float
    price_position_in_range: float  # 0-100, where 50 is middle
    range_20d_high: float
    range_20d_low: float
    is_squeeze: bool
    timestamp: datetime


class MarketRegimeDetector:
    """Detects market regime using multiple timeframe analysis.

    Uses ADX for trend strength, ATR percentiles for volatility regime,
    Bollinger Band width for squeeze detection, and price position
    within range for directional bias.

    Includes India-specific context: weekly/monthly expiry detection,
    event day flagging.

    Args:
        cache_minutes: How often to recompute regime (default 5 min).
    """

    def __init__(self, cache_minutes: int = 5):
        """Initialize the regime detector.

        Args:
            cache_minutes: Cache duration in minutes.
        """
        self._cache_minutes = cache_minutes
        self._cache: dict[str, tuple[datetime, RegimeData]] = {}
        self._event_cache: dict[str, EventData] = {}

    def _is_cache_valid(self, key: str) -> bool:
        """Check if cached regime data is still valid.

        Args:
            key: Cache key (typically instrument_key).

        Returns:
            True if cache is valid and not expired.
        """
        if key not in self._cache:
            return False

        cached_time, _ = self._cache[key]
        now = datetime.now(IST)
        age = now - cached_time

        return age < timedelta(minutes=self._cache_minutes)

    def _get_cache_key(self, instrument_key: str) -> str:
        """Generate cache key for an instrument.

        Args:
            instrument_key: The instrument identifier.

        Returns:
            Cache key string.
        """
        return f"{instrument_key}_regime"

    @graceful_degrade(
        default_return=(
            MarketRegime.RANGING,
            VolatilityRegime.NORMAL,
            AnalysisComponent(
                name="market_regime",
                score=50.0,
                weight=0.20,
                confidence=30.0,
                reasoning="Market regime detection failed - using default neutral values",
                data={"regime": "unknown", "confidence": 0.5},
            )
        )
    )
    def detect_regime(
        self,
        df_15min: pd.DataFrame,
        df_daily: pd.DataFrame,
        instrument_key: str | None = None,
    ) -> tuple[MarketRegime, VolatilityRegime, AnalysisComponent]:
        """Detect market regime from 15min and daily data.

        Uses ADX(14) for trend strength:
        - < 20: Ranging
        - 20-40: Trending
        - > 40: Strong trend

        Uses ATR(14) percentile vs 20 days:
        - < 25th: Quiet
        - 25-75: Normal
        - 75-95: Volatile
        - > 95: Extreme

        Uses Bollinger Band width for squeeze detection.
        Uses price position within 20-day range for directional bias.

        Args:
            df_15min: 15-minute OHLCV DataFrame (minimum 50 bars).
            df_daily: Daily OHLCV DataFrame (minimum 20 bars).
            instrument_key: Optional instrument key for caching.

        Returns:
            Tuple of (MarketRegime, VolatilityRegime, AnalysisComponent).
        """
        # Check cache first
        if instrument_key:
            cache_key = self._get_cache_key(instrument_key)
            if self._is_cache_valid(cache_key):
                cached_time, regime_data = self._cache[cache_key]
                component = self._create_analysis_component(regime_data)
                return regime_data.market_regime, regime_data.volatility_regime, component

        # Validate data
        if len(df_15min) < 10:
            raise ValueError(f"Need at least 10 bars of 15min data, got {len(df_15min)}")
        if len(df_daily) < 20:
            raise ValueError(f"Need at least 20 bars of daily data, got {len(df_daily)}")

        # Calculate indicators on 15min data
        engine_15min = IndicatorEngine(df_15min)

        # Calculate ADX(14) on 15min
        adx_series = self._calculate_adx(engine_15min, period=14)
        adx_current = adx_series.iloc[-1]
        adx_prev = adx_series.iloc[-5] if len(adx_series) >= 5 else adx_series.iloc[-2]

        # Determine ADX trend
        if adx_current > adx_prev * 1.05:
            adx_trend = "rising"
        elif adx_current < adx_prev * 0.95:
            adx_trend = "falling"
        else:
            adx_trend = "flat"

        # Calculate ATR(14) on 15min
        atr_series = engine_15min.atr(period=14)
        atr_current = atr_series.iloc[-1]

        # Calculate ATR percentile vs 20 days (approx 480 15min bars)
        atr_lookback = min(480, len(atr_series) - 14)
        atr_history = atr_series.dropna().iloc[-atr_lookback:]
        atr_percentile = self._calculate_percentile(atr_current, atr_history)
        atr_20d_avg = atr_history.mean()

        # Calculate Bollinger Band width on 15min
        bb_result = engine_15min.bbands(period=20, std=2.0)
        bb_upper = bb_result["upper"].iloc[-1]
        bb_lower = bb_result["lower"].iloc[-1]
        bb_middle = bb_result["middle"].iloc[-1]

        if bb_middle > 0:
            bb_width = (bb_upper - bb_lower) / bb_middle * 100
        else:
            bb_width = 0.0

        # Calculate BB width percentile (lookback ~100 bars)
        bb_width_series = (bb_result["upper"] - bb_result["lower"]) / bb_result["middle"] * 100
        bb_width_series = bb_width_series.dropna()
        bb_width_percentile = self._calculate_percentile(
            bb_width, bb_width_series.iloc[-100:] if len(bb_width_series) >= 100 else bb_width_series
        )

        # Squeeze detection: BB width in bottom 20% of recent history
        is_squeeze = bb_width_percentile < 20

        # Calculate daily indicators for longer-term context
        engine_daily = IndicatorEngine(df_daily)

        # Price position within 20-day range
        recent_20d = df_daily.iloc[-20:]
        range_20d_high = recent_20d["high"].max()
        range_20d_low = recent_20d["low"].min()
        current_close = df_daily["close"].iloc[-1]

        if range_20d_high > range_20d_low:
            price_position = (current_close - range_20d_low) / (range_20d_high - range_20d_low) * 100
        else:
            price_position = 50.0

        # Determine market regime based on ADX and price action
        market_regime = self._classify_market_regime(
            adx_current=adx_current,
            adx_trend=adx_trend,
            price_position=price_position,
            atr_percentile=atr_percentile,
            is_squeeze=is_squeeze,
        )

        # Determine volatility regime based on ATR percentile
        volatility_regime = self._classify_volatility_regime(atr_percentile)

        # Create regime data
        regime_data = RegimeData(
            market_regime=market_regime,
            volatility_regime=volatility_regime,
            adx_value=adx_current,
            adx_trend=adx_trend,
            atr_percentile=atr_percentile,
            atr_current=atr_current,
            atr_20d_average=atr_20d_avg,
            bb_width=bb_width,
            bb_width_percentile=bb_width_percentile,
            price_position_in_range=price_position,
            range_20d_high=range_20d_high,
            range_20d_low=range_20d_low,
            is_squeeze=is_squeeze,
            timestamp=datetime.now(IST),
        )

        # Cache the result
        if instrument_key:
            cache_key = self._get_cache_key(instrument_key)
            self._cache[cache_key] = (datetime.now(IST), regime_data)

        # Create analysis component
        component = self._create_analysis_component(regime_data)

        return market_regime, volatility_regime, component

    def _calculate_adx(self, engine: IndicatorEngine, period: int = 14) -> pd.Series:
        """Calculate ADX using pandas-ta.

        Args:
            engine: IndicatorEngine with OHLCV data.
            period: ADX period.

        Returns:
            ADX series.
        """
        df = engine.get_data()
        adx_result = df.ta.adx(length=period)
        if adx_result is None:
            raise ValueError("ADX calculation failed")
        # pandas-ta returns DataFrame with ADX, DMP, DMN columns
        adx_col = f"ADX_{period}"
        return adx_result[adx_col]

    def _calculate_percentile(self, value: float, series: pd.Series) -> float:
        """Calculate percentile of value within series.

        Args:
            value: The value to rank.
            series: Historical series.

        Returns:
            Percentile (0-100).
        """
        if len(series) == 0:
            return 50.0

        # Count values less than current
        count_below = (series < value).sum()
        percentile = (count_below / len(series)) * 100

        return float(percentile)

    def _classify_market_regime(
        self,
        adx_current: float,
        adx_trend: str,
        price_position: float,
        atr_percentile: float,
        is_squeeze: bool,
    ) -> MarketRegime:
        """Classify market regime based on indicators.

        Args:
            adx_current: Current ADX value.
            adx_trend: ADX trend direction.
            price_position: Price position in 20-day range (0-100).
            atr_percentile: ATR percentile.
            is_squeeze: Whether BB squeeze is detected.

        Returns:
            MarketRegime classification.
        """
        # Handle extreme volatility first
        if atr_percentile > 95:
            return MarketRegime.VOLATILE

        # Handle quiet/squeeze conditions
        if atr_percentile < 25 or is_squeeze:
            return MarketRegime.QUIET

        # Classify based on ADX
        if adx_current < 20:
            return MarketRegime.RANGING
        elif adx_current < 40:
            # Weak trend - check direction
            if price_position > 60:
                return MarketRegime.WEAK_UPTREND
            elif price_position < 40:
                return MarketRegime.WEAK_DOWNTREND
            else:
                return MarketRegime.RANGING
        else:
            # Strong trend
            if price_position > 60:
                return MarketRegime.STRONG_UPTREND
            elif price_position < 40:
                return MarketRegime.STRONG_DOWNTREND
            else:
                # ADX high but price in middle - possible consolidation
                return MarketRegime.RANGING

    def _classify_volatility_regime(self, atr_percentile: float) -> VolatilityRegime:
        """Classify volatility regime based on ATR percentile.

        Args:
            atr_percentile: ATR percentile (0-100).

        Returns:
            VolatilityRegime classification.
        """
        if atr_percentile < 25:
            return VolatilityRegime.LOW
        elif atr_percentile < 75:
            return VolatilityRegime.NORMAL
        elif atr_percentile < 95:
            return VolatilityRegime.HIGH
        else:
            return VolatilityRegime.EXTREME

    def _create_analysis_component(self, regime_data: RegimeData) -> AnalysisComponent:
        """Create AnalysisComponent from regime data.

        Args:
            regime_data: Regime detection results.

        Returns:
            AnalysisComponent for research report.
        """
        # Score based on regime clarity
        if regime_data.market_regime in [
            MarketRegime.STRONG_UPTREND,
            MarketRegime.STRONG_DOWNTREND,
        ]:
            score = 80.0
        elif regime_data.market_regime in [
            MarketRegime.WEAK_UPTREND,
            MarketRegime.WEAK_DOWNTREND,
        ]:
            score = 65.0
        elif regime_data.market_regime == MarketRegime.RANGING:
            score = 50.0
        elif regime_data.market_regime == MarketRegime.QUIET:
            score = 55.0
        else:  # VOLATILE
            score = 40.0

        # Adjust for volatility
        if regime_data.volatility_regime == VolatilityRegime.EXTREME:
            score -= 15
        elif regime_data.volatility_regime == VolatilityRegime.HIGH:
            score -= 5

        # Clamp score
        score = max(0, min(100, score))

        # Build reasoning
        reasoning_parts = [
            f"Market regime: {regime_data.market_regime.value}",
            f"Volatility regime: {regime_data.volatility_regime.value}",
            f"ADX: {regime_data.adx_value:.1f} ({regime_data.adx_trend})",
            f"ATR percentile: {regime_data.atr_percentile:.1f}%",
        ]

        if regime_data.is_squeeze:
            reasoning_parts.append("BB squeeze detected - potential breakout imminent")

        reasoning = "; ".join(reasoning_parts)

        return AnalysisComponent(
            name="market_regime",
            score=score,
            weight=0.20,  # 20% weight in final score
            confidence=85.0 if regime_data.adx_value > 20 else 70.0,
            reasoning=reasoning,
            data={
                "market_regime": regime_data.market_regime.value,
                "volatility_regime": regime_data.volatility_regime.value,
                "adx_value": regime_data.adx_value,
                "adx_trend": regime_data.adx_trend,
                "atr_percentile": regime_data.atr_percentile,
                "atr_current": regime_data.atr_current,
                "bb_width": regime_data.bb_width,
                "bb_width_percentile": regime_data.bb_width_percentile,
                "price_position_in_range": regime_data.price_position_in_range,
                "is_squeeze": regime_data.is_squeeze,
                "range_20d_high": regime_data.range_20d_high,
                "range_20d_low": regime_data.range_20d_low,
            },
        )

    def calculate_regime_signal_compatibility(
        self,
        market_regime: MarketRegime,
        signal_direction: SignalDirection,
        strategy_type: StrategyType,
    ) -> float:
        """Calculate compatibility multiplier between regime and signal.

        Scoring logic:
        - Trending regime + Trend-following = HIGH (1.2x)
        - Trending regime + Mean-reversion = LOW (0.6x)
        - Ranging regime + Mean-reversion = HIGH (1.2x)
        - Ranging regime + Trend-following = LOW (0.6x)
        - Volatile regime = REDUCE by 20% (0.8x)
        - Quiet before breakout = INCREASE breakout signals (1.3x)

        Args:
            market_regime: Detected market regime.
            signal_direction: Direction of the signal (long/short).
            strategy_type: Type of strategy generating the signal.

        Returns:
            Compatibility multiplier (0.0 to 1.5).
        """
        base_multiplier = 1.0

        # Trend-following strategies
        if strategy_type == StrategyType.TREND_FOLLOWING:
            if market_regime in [
                MarketRegime.STRONG_UPTREND,
                MarketRegime.STRONG_DOWNTREND,
            ]:
                base_multiplier = 1.2
            elif market_regime in [
                MarketRegime.WEAK_UPTREND,
                MarketRegime.WEAK_DOWNTREND,
            ]:
                base_multiplier = 1.1
            elif market_regime == MarketRegime.RANGING:
                base_multiplier = 0.6
            elif market_regime == MarketRegime.QUIET:
                base_multiplier = 0.8
            else:  # VOLATILE
                base_multiplier = 0.7

        # Mean-reversion strategies
        elif strategy_type == StrategyType.MEAN_REVERSION:
            if market_regime == MarketRegime.RANGING:
                base_multiplier = 1.2
            elif market_regime == MarketRegime.QUIET:
                base_multiplier = 1.0
            elif market_regime in [
                MarketRegime.STRONG_UPTREND,
                MarketRegime.STRONG_DOWNTREND,
            ]:
                base_multiplier = 0.6
            elif market_regime in [
                MarketRegime.WEAK_UPTREND,
                MarketRegime.WEAK_DOWNTREND,
            ]:
                base_multiplier = 0.8
            else:  # VOLATILE
                base_multiplier = 0.9

        # Breakout strategies
        elif strategy_type == StrategyType.BREAKOUT:
            if market_regime == MarketRegime.QUIET:
                base_multiplier = 1.3  # Quiet before breakout
            elif market_regime in [
                MarketRegime.STRONG_UPTREND,
                MarketRegime.STRONG_DOWNTREND,
            ]:
                base_multiplier = 1.1
            elif market_regime == MarketRegime.VOLATILE:
                base_multiplier = 0.8
            else:
                base_multiplier = 1.0

        # Momentum strategies
        elif strategy_type == StrategyType.MOMENTUM:
            if market_regime in [
                MarketRegime.STRONG_UPTREND,
                MarketRegime.STRONG_DOWNTREND,
            ]:
                base_multiplier = 1.2
            elif market_regime == MarketRegime.VOLATILE:
                base_multiplier = 0.9
            else:
                base_multiplier = 1.0

        # Scalping strategies
        elif strategy_type == StrategyType.SCALPING:
            if market_regime == MarketRegime.VOLATILE:
                base_multiplier = 1.1  # Scalpers like volatility
            elif market_regime in [
                MarketRegime.STRONG_UPTREND,
                MarketRegime.STRONG_DOWNTREND,
            ]:
                base_multiplier = 1.0
            else:
                base_multiplier = 0.9

        # Direction alignment check
        if signal_direction != SignalDirection.NEUTRAL:
            if market_regime == MarketRegime.STRONG_UPTREND and signal_direction == SignalDirection.SHORT:
                base_multiplier *= 0.7  # Penalty for counter-trend short
            elif market_regime == MarketRegime.STRONG_DOWNTREND and signal_direction == SignalDirection.LONG:
                base_multiplier *= 0.7  # Penalty for counter-trend long

        # Volatile regime reduction
        if market_regime == MarketRegime.VOLATILE:
            base_multiplier *= 0.8

        return round(base_multiplier, 2)

    @graceful_degrade(
        default_return=EventData(
            is_weekly_expiry=False,
            is_monthly_expiry=False,
            days_to_monthly_expiry=None,
            is_rbi_policy_day=False,
            is_fed_day=False,
            is_budget_day=False,
            is_earnings_season=False,
            time_of_day_score=50.0,
            event_risk_level="low",
        )
    )
    def detect_indian_market_context(self, timestamp: datetime | None = None) -> EventData:
        """Detect India-specific market context.

        Detects:
        - Weekly expiry day (Thursday for Nifty/BankNifty)
        - Monthly expiry week (last week of month)
        - Event day flagging

        Args:
            timestamp: Timestamp to analyze (default: now).

        Returns:
            EventData with Indian market context.
        """
        if timestamp is None:
            timestamp = datetime.now(IST)

        # Check cache
        date_key = timestamp.strftime("%Y-%m-%d")
        if date_key in self._event_cache:
            return self._event_cache[date_key]

        # Weekly expiry: Thursday (weekday 3)
        is_weekly_expiry = timestamp.weekday() == 3

        # Monthly expiry: Last Thursday of month
        # Check if this is the last Thursday
        next_week = timestamp + timedelta(days=7)
        is_monthly_expiry = is_weekly_expiry and next_week.month != timestamp.month

        # Days to monthly expiry
        days_to_monthly_expiry = None
        if not is_monthly_expiry:
            # Find last Thursday of month
            last_day = timestamp.replace(day=28)  # Start from 28th
            while last_day.month == timestamp.month:
                last_day += timedelta(days=1)
            last_day -= timedelta(days=1)  # Last day of month

            # Find last Thursday
            while last_day.weekday() != 3:
                last_day -= timedelta(days=1)

            days_to_monthly_expiry = (last_day - timestamp).days

        # Time of day score (Indian market hours)
        # 9:15 AM - 3:30 PM IST
        hour = timestamp.hour
        minute = timestamp.minute
        time_score = 50.0  # Neutral

        if hour == 9 and minute >= 15:
            # Opening - higher volatility, avoid first 15 min
            if minute < 30:
                time_score = 40.0
            else:
                time_score = 60.0
        elif hour == 10:
            time_score = 70.0  # Good trading hour
        elif hour == 11:
            time_score = 65.0
        elif hour == 12:
            time_score = 55.0  # Lunch time - lower volume
        elif hour == 13:
            time_score = 60.0
        elif hour == 14:
            time_score = 70.0  # Afternoon session
        elif hour == 15:
            if minute < 15:
                time_score = 65.0
            else:
                time_score = 45.0  # Close approaching

        # Event risk level
        event_risk = "low"
        if is_monthly_expiry:
            event_risk = "high"
        elif is_weekly_expiry:
            event_risk = "medium"

        event_data = EventData(
            is_weekly_expiry=is_weekly_expiry,
            is_monthly_expiry=is_monthly_expiry,
            days_to_monthly_expiry=days_to_monthly_expiry,
            is_rbi_policy_day=False,  # Would need external calendar
            is_fed_day=False,  # Would need external calendar
            is_budget_day=False,  # Would need external calendar
            is_earnings_season=False,  # Would need external calendar
            time_of_day_score=time_score,
            event_risk_level=event_risk,
        )

        # Cache result
        self._event_cache[date_key] = event_data

        return event_data

    def get_cached_regime(self, instrument_key: str) -> RegimeData | None:
        """Get cached regime data if available.

        Args:
            instrument_key: The instrument identifier.

        Returns:
            RegimeData if cached and valid, None otherwise.
        """
        cache_key = self._get_cache_key(instrument_key)
        if self._is_cache_valid(cache_key):
            _, regime_data = self._cache[cache_key]
            return regime_data
        return None

    def clear_cache(self, instrument_key: str | None = None) -> None:
        """Clear regime cache.

        Args:
            instrument_key: Specific instrument to clear, or None for all.
        """
        if instrument_key:
            cache_key = self._get_cache_key(instrument_key)
            self._cache.pop(cache_key, None)
        else:
            self._cache.clear()
            self._event_cache.clear()

    def get_regime_summary(self, instrument_key: str) -> dict[str, Any] | None:
        """Get human-readable regime summary.

        Args:
            instrument_key: The instrument identifier.

        Returns:
            Dictionary with regime summary or None if not cached.
        """
        regime_data = self.get_cached_regime(instrument_key)
        if not regime_data:
            return None

        return {
            "market_regime": regime_data.market_regime.value,
            "volatility_regime": regime_data.volatility_regime.value,
            "adx": f"{regime_data.adx_value:.1f} ({regime_data.adx_trend})",
            "atr_percentile": f"{regime_data.atr_percentile:.1f}%",
            "bb_squeeze": regime_data.is_squeeze,
            "price_position": f"{regime_data.price_position_in_range:.1f}%",
            "timestamp": regime_data.timestamp.isoformat(),
        }


def create_default_detector() -> MarketRegimeDetector:
    """Create a MarketRegimeDetector with default settings.

    Returns:
        Configured MarketRegimeDetector.
    """
    return MarketRegimeDetector(cache_minutes=5)


# Example usage and testing
if __name__ == "__main__":
    # Create sample data for testing
    import numpy as np

    # Generate sample OHLCV data
    np.random.seed(42)
    n_bars = 500

    # Create 15min data
    dates_15min = pd.date_range(
        start="2024-01-01 09:15",
        periods=n_bars,
        freq="15min",
        tz=IST,
    )

    # Generate trending price series
    trend = np.cumsum(np.random.randn(n_bars) * 10)
    close_15min = 50000 + trend
    high_15min = close_15min + np.abs(np.random.randn(n_bars) * 50)
    low_15min = close_15min - np.abs(np.random.randn(n_bars) * 50)
    open_15min = close_15min + np.random.randn(n_bars) * 30
    volume_15min = np.random.randint(1000, 10000, n_bars)

    df_15min = pd.DataFrame({
        "open": open_15min,
        "high": high_15min,
        "low": low_15min,
        "close": close_15min,
        "volume": volume_15min,
    }, index=dates_15min)

    # Create daily data
    dates_daily = pd.date_range(
        start="2024-01-01",
        periods=30,
        freq="D",
        tz=IST,
    )

    close_daily = 50000 + np.cumsum(np.random.randn(30) * 100)
    high_daily = close_daily + np.abs(np.random.randn(30) * 200)
    low_daily = close_daily - np.abs(np.random.randn(30) * 200)
    open_daily = close_daily + np.random.randn(30) * 100
    volume_daily = np.random.randint(100000, 1000000, 30)

    df_daily = pd.DataFrame({
        "open": open_daily,
        "high": high_daily,
        "low": low_daily,
        "close": close_daily,
        "volume": volume_daily,
    }, index=dates_daily)

    # Test regime detection
    detector = create_default_detector()

    market_regime, vol_regime, component = detector.detect_regime(
        df_15min=df_15min,
        df_daily=df_daily,
        instrument_key="NSE_EQ:RELIANCE",
    )

    print("Market Regime:", market_regime.value)
    print("Volatility Regime:", vol_regime.value)
    print("Analysis Score:", component.score)
    print("Reasoning:", component.reasoning)

    # Test compatibility calculation
    compatibility = detector.calculate_regime_signal_compatibility(
        market_regime=market_regime,
        signal_direction=SignalDirection.LONG,
        strategy_type=StrategyType.TREND_FOLLOWING,
    )
    print(f"\nCompatibility (Trend-following Long): {compatibility}x")

    compatibility_mr = detector.calculate_regime_signal_compatibility(
        market_regime=market_regime,
        signal_direction=SignalDirection.SHORT,
        strategy_type=StrategyType.MEAN_REVERSION,
    )
    print(f"Compatibility (Mean-reversion Short): {compatibility_mr}x")

    # Test Indian market context
    event_data = detector.detect_indian_market_context()
    print(f"\nWeekly Expiry: {event_data.is_weekly_expiry}")
    print(f"Monthly Expiry: {event_data.is_monthly_expiry}")
    print(f"Time Score: {event_data.time_of_day_score}")
    print(f"Event Risk: {event_data.event_risk_level}")

    # Test caching
    print("\n--- Testing Cache ---")
    cached = detector.get_cached_regime("NSE_EQ:RELIANCE")
    print(f"Cache hit: {cached is not None}")

    summary = detector.get_regime_summary("NSE_EQ:RELIANCE")
    print(f"Regime Summary: {summary}")
