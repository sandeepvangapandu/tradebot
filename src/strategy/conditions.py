"""Condition evaluator for strategy entry/exit conditions.

Evaluates conditions against market data bars, supporting comparisons,
indicator vs indicator, indicator vs value, price-based, and time-based conditions.

The Condition model uses the JSON-native format:
    - indicator: str  (e.g. "MACD", "RSI", "Spot_Price")
    - comparison: str (e.g. ">", "crosses_above")
    - against: str | None  (reference indicator for indicator-vs-indicator)
    - value: float | None  (constant for indicator-vs-value)
    - parameters: dict  (indicator params like {"fast": 12, "slow": 26})
"""

from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
from enum import Enum
from threading import Lock
from typing import Any, Union
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from loguru import logger

from src.strategy.builder import (
    ComparisonOperator,
    Condition,
)
from src.strategy.indicators import IndicatorEngine

IST = ZoneInfo("Asia/Kolkata")


class TrendStrength(Enum):
    """ADX-based trend strength classification."""

    RANGING = "ranging"  # ADX < 20
    TRENDING = "trending"  # ADX >= 25
    STRONG_TREND = "strong_trend"  # ADX >= 40


class VolatilityRegime(Enum):
    """Volatility regime classification based on ATR percentiles.

    Used for volatility-adjusted entry thresholds, stop distances,
    and position sizing.
    """

    EXTREMELY_LOW = "extremely_low"  # ATR < 5th percentile
    LOW = "low"                      # ATR 5-25th percentile
    NORMAL = "normal"                # ATR 25-75th percentile
    HIGH = "high"                    # ATR 75-95th percentile
    EXTREME = "extreme"              # ATR > 95th percentile


@dataclass
class FilterResult:
    """Result of an enhanced filter evaluation.

    Attributes:
        filter_name: Name of the filter (e.g., "adx_filter", "volume_confirmation")
        passed: Whether the filter passed
        score: Numerical score (0-1) representing filter strength
        details: Additional details for debugging
    """

    filter_name: str
    passed: bool
    score: float = 0.0
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class MultiTimeframeConfluence:
    """Multi-timeframe confluence analysis result.

    Attributes:
        timeframes: List of timeframes analyzed (e.g., ["1m", "5m", "15m"])
        alignment_score: Percentage of timeframes aligning (0-1)
        aligned_count: Number of timeframes in alignment
        total_count: Total number of timeframes checked
        timeframe_signals: Dict mapping timeframe to signal direction
    """

    timeframes: list[str] = field(default_factory=list)
    alignment_score: float = 0.0
    aligned_count: int = 0
    total_count: int = 0
    timeframe_signals: dict[str, str] = field(default_factory=dict)

    def has_confluence(self, min_aligned: int = 2) -> bool:
        """Check if minimum number of timeframes align.

        Args:
            min_aligned: Minimum number of timeframes required for confluence.

        Returns:
            True if confluence requirement is met.
        """
        return self.aligned_count >= min_aligned

# Price-like column names that map directly to DataFrame columns
_PRICE_COLUMNS = {"spot_price", "price", "close", "open", "high", "low", "volume"}

# Default ADX thresholds for trend strength
ADX_THRESHOLD_RANGING = 20.0
ADX_THRESHOLD_TRENDING = 25.0
ADX_THRESHOLD_STRONG = 40.0

# Default volume confirmation multiplier
VOLUME_CONFIRMATION_MULTIPLIER = 1.2

# Default OBV lookback for trend detection
OBV_TREND_LOOKBACK = 5

# Default volatility regime ATR percentile thresholds
ATR_PERCENTILE_EXTREMELY_LOW = 5.0
ATR_PERCENTILE_LOW = 25.0
ATR_PERCENTILE_HIGH = 75.0
ATR_PERCENTILE_EXTREME = 95.0


@dataclass
class VolatilityAdjustedEntry:
    """Volatility-adjusted entry conditions for dynamic threshold management.

    Adjusts entry thresholds, stop distances, and position sizes based on
    current volatility regime. Uses ATR percentile ranking over a lookback
    period to classify volatility.

    Regime-specific multipliers:
        - EXTREMELY_LOW: Tighter entries (0.5x), tighter stops (0.7x), larger size (1.3x)
        - LOW: Slightly tighter entries (0.7x), tighter stops (0.8x), larger size (1.2x)
        - NORMAL: Baseline multipliers (1.0x all)
        - HIGH: Wider entries (1.3x), wider stops (1.5x), reduced size (0.7x)
        - EXTREME: Very wide entries (1.5x), very wide stops (2.0x), small size (0.5x)

    Attributes:
        atr_period: Period for ATR calculation (default 14)
        lookback: Lookback period for percentile calculation (default 100)
        _atr_cache: Thread-safe cache for ATR calculations by symbol
        _last_regime: Cache for last detected regime by symbol
    """

    atr_period: int = 14
    lookback: int = 100
    _atr_cache: dict[str, pd.Series] = field(default_factory=dict)
    _last_regime: dict[str, VolatilityRegime] = field(default_factory=dict)
    _lock: Lock = field(default_factory=Lock)

    # Regime-specific multipliers
    _ENTRY_MULTIPLIERS: dict[VolatilityRegime, float] = field(
        default_factory=lambda: {
            VolatilityRegime.EXTREMELY_LOW: 0.5,
            VolatilityRegime.LOW: 0.7,
            VolatilityRegime.NORMAL: 1.0,
            VolatilityRegime.HIGH: 1.3,
            VolatilityRegime.EXTREME: 1.5,
        }
    )

    _STOP_MULTIPLIERS: dict[VolatilityRegime, float] = field(
        default_factory=lambda: {
            VolatilityRegime.EXTREMELY_LOW: 0.7,
            VolatilityRegime.LOW: 0.8,
            VolatilityRegime.NORMAL: 1.0,
            VolatilityRegime.HIGH: 1.5,
            VolatilityRegime.EXTREME: 2.0,
        }
    )

    _SIZE_FACTORS: dict[VolatilityRegime, float] = field(
        default_factory=lambda: {
            VolatilityRegime.EXTREMELY_LOW: 1.3,
            VolatilityRegime.LOW: 1.2,
            VolatilityRegime.NORMAL: 1.0,
            VolatilityRegime.HIGH: 0.7,
            VolatilityRegime.EXTREME: 0.5,
        }
    )

    def detect_regime(
        self,
        data: pd.DataFrame,
        symbol: str | None = None,
    ) -> VolatilityRegime:
        """Detect current volatility regime based on ATR percentile ranking.

        Calculates ATR for the lookback period and compares current ATR
        to the historical distribution to determine percentile rank.

        Args:
            data: OHLCV DataFrame with at least 'high', 'low', 'close' columns
            symbol: Optional symbol identifier for caching

        Returns:
            VolatilityRegime enum value representing current regime

        Raises:
            ValueError: If data is insufficient for calculation
        """
        if data is None or len(data) == 0:
            raise ValueError("Data cannot be empty")

        required_cols = {'high', 'low', 'close'}
        if not required_cols.issubset(data.columns):
            raise ValueError(f"Data must contain columns: {required_cols}")

        # Need at least lookback + atr_period bars
        min_bars = self.lookback + self.atr_period
        if len(data) < min_bars:
            logger.warning(
                f"Insufficient data for volatility regime detection. "
                f"Have {len(data)} bars, need {min_bars}. Returning NORMAL."
            )
            return VolatilityRegime.NORMAL

        cache_key = symbol or "default"

        # Calculate ATR using pandas-ta style calculation (vectorized)
        atr_series = self._calculate_atr(data)

        with self._lock:
            self._atr_cache[cache_key] = atr_series

        # Get current ATR and historical ATR values
        current_atr = float(atr_series.iloc[-1])
        historical_atr = atr_series.iloc[-self.lookback:-1].values

        # Calculate percentile rank (vectorized)
        percentile = self._calculate_percentile(current_atr, historical_atr)

        # Map percentile to regime
        regime = self._percentile_to_regime(percentile)

        with self._lock:
            self._last_regime[cache_key] = regime

        logger.debug(
            f"Volatility regime for {cache_key}: {regime.value} "
            f"(ATR: {current_atr:.4f}, percentile: {percentile:.1f}%)"
        )

        return regime

    def _calculate_atr(self, data: pd.DataFrame) -> pd.Series:
        """Calculate Average True Range using vectorized operations.

        Uses pandas-ta style calculation for consistency with existing codebase.

        Args:
            data: OHLCV DataFrame

        Returns:
            Series with ATR values
        """
        high = data['high']
        low = data['low']
        close = data['close']

        # True Range calculation (vectorized)
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()

        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # ATR is EMA of true range (RMA - Wilder's smoothing)
        # Using ewm with alpha=1/period for Wilder's smoothing
        atr = true_range.ewm(alpha=1.0/self.atr_period, min_periods=self.atr_period).mean()

        return atr

    def _calculate_percentile(self, value: float, distribution: pd.Series | np.ndarray) -> float:
        """Calculate percentile rank of value within distribution.

        Args:
            value: Current value to rank
            distribution: Historical values for comparison

        Returns:
            Percentile rank (0-100)
        """
        if isinstance(distribution, pd.Series):
            distribution = distribution.values

        # Remove NaN values
        distribution = distribution[~np.isnan(distribution)]

        if len(distribution) == 0:
            return 50.0  # Default to middle if no valid data

        # Calculate percentile rank (vectorized)
        below_count = np.sum(distribution < value)
        percentile = (below_count / len(distribution)) * 100.0

        return float(percentile)

    def _percentile_to_regime(self, percentile: float) -> VolatilityRegime:
        """Map percentile rank to volatility regime.

        Args:
            percentile: ATR percentile rank (0-100)

        Returns:
            Corresponding VolatilityRegime
        """
        if percentile < ATR_PERCENTILE_EXTREMELY_LOW:
            return VolatilityRegime.EXTREMELY_LOW
        elif percentile < ATR_PERCENTILE_LOW:
            return VolatilityRegime.LOW
        elif percentile < ATR_PERCENTILE_HIGH:
            return VolatilityRegime.NORMAL
        elif percentile < ATR_PERCENTILE_EXTREME:
            return VolatilityRegime.HIGH
        else:
            return VolatilityRegime.EXTREME

    def get_entry_multiplier(self, regime: VolatilityRegime) -> float:
        """Get entry threshold multiplier for given regime.

        Entry multiplier determines how far price must move for entry.
        Lower values = tighter entries in low volatility.
        Higher values = wider entries in high volatility.

        Args:
            regime: Volatility regime

        Returns:
            Multiplier value
        """
        return self._ENTRY_MULTIPLIERS.get(regime, 1.0)

    def get_stop_multiplier(self, regime: VolatilityRegime) -> float:
        """Get stop-loss distance multiplier for given regime.

        Stop multiplier adjusts ATR-based stop distances.
        Lower values = tighter stops in low volatility.
        Higher values = wider stops in high volatility.

        Args:
            regime: Volatility regime

        Returns:
            Multiplier value
        """
        return self._STOP_MULTIPLIERS.get(regime, 1.0)

    def get_position_size_factor(self, regime: VolatilityRegime) -> float:
        """Get position size adjustment factor for given regime.

        Size factor reduces position size in high volatility to manage risk.
        Increases size in low volatility to capture more profit.

        Args:
            regime: Volatility regime

        Returns:
            Size factor (multiply base quantity by this)
        """
        return self._SIZE_FACTORS.get(regime, 1.0)

    def adjust_entry_conditions(
        self,
        base_conditions: dict[str, Any],
        regime: VolatilityRegime,
    ) -> dict[str, Any]:
        """Adjust entry conditions based on volatility regime.

        Applies regime-specific multipliers to:
        - stop_loss: Adjusted by stop multiplier
        - target: Adjusted by entry multiplier (for R:R consistency)
        - quantity: Adjusted by size factor

        Args:
            base_conditions: Dictionary with 'stop_loss', 'target', 'quantity'
            regime: Current volatility regime

        Returns:
            Adjusted conditions dictionary

        Example:
            >>> adjuster = VolatilityAdjustedEntry()
            >>> base = {"stop_loss": 100, "target": 200, "quantity": 100}
            >>> adjusted = adjuster.adjust_entry_conditions(base, VolatilityRegime.HIGH)
            >>> # Result: {"stop_loss": 150, "target": 260, "quantity": 70}
        """
        adjusted = base_conditions.copy()

        entry_mult = self.get_entry_multiplier(regime)
        stop_mult = self.get_stop_multiplier(regime)
        size_factor = self.get_position_size_factor(regime)

        # Adjust stop loss
        if "stop_loss" in adjusted:
            adjusted["stop_loss"] = adjusted["stop_loss"] * stop_mult

        # Adjust target (maintain R:R ratio relative to adjusted stop)
        if "target" in adjusted:
            # Target scales with entry multiplier to capture larger moves
            # in high volatility while maintaining reasonable R:R
            adjusted["target"] = adjusted["target"] * entry_mult

        # Adjust position size
        if "quantity" in adjusted:
            adjusted["quantity"] = int(adjusted["quantity"] * size_factor)

        # Add metadata for logging
        adjusted["_volatility_regime"] = regime.value
        adjusted["_entry_multiplier"] = entry_mult
        adjusted["_stop_multiplier"] = stop_mult
        adjusted["_size_factor"] = size_factor

        logger.info(
            f"Adjusted entry conditions for {regime.value}: "
            f"stop_mult={stop_mult:.2f}, entry_mult={entry_mult:.2f}, "
            f"size_factor={size_factor:.2f}"
        )

        return adjusted

    def get_cached_atr(self, symbol: str | None = None) -> pd.Series | None:
        """Get cached ATR series for a symbol.

        Args:
            symbol: Symbol identifier

        Returns:
            Cached ATR series or None
        """
        cache_key = symbol or "default"
        with self._lock:
            return self._atr_cache.get(cache_key)

    def get_last_regime(self, symbol: str | None = None) -> VolatilityRegime | None:
        """Get last detected regime for a symbol.

        Args:
            symbol: Symbol identifier

        Returns:
            Last VolatilityRegime or None
        """
        cache_key = symbol or "default"
        with self._lock:
            return self._last_regime.get(cache_key)

    def clear_cache(self, symbol: str | None = None) -> None:
        """Clear ATR and regime cache.

        Args:
            symbol: Symbol to clear (None = clear all)
        """
        with self._lock:
            if symbol is None:
                self._atr_cache.clear()
                self._last_regime.clear()
            else:
                self._atr_cache.pop(symbol, None)
                self._last_regime.pop(symbol, None)


class ConditionEvaluator:
    """Evaluates strategy conditions against market data.

    Supports:
    - Comparison operators: >, <, >=, <=, ==, crosses_above, crosses_below
    - Indicator vs value comparisons
    - Indicator vs indicator comparisons
    - Price-based conditions
    - Time-based conditions
    - Lookback for checking previous bars
    """

    def __init__(self):
        """Initialize the condition evaluator."""
        self._indicator_engines: dict[str, IndicatorEngine] = {}
        self._last_values: dict[str, Any] = {}
        # CPR cache: {(symbol, date_iso): cpr_dict}. Keyed on the date we
        # are evaluating; CPR computed from the *prior* trading day's OHLC.
        self._cpr_cache: dict[tuple[str, str], dict[str, float]] = {}

    def _auto_cpr(
        self,
        engine: "IndicatorEngine",
        symbol: str | None,
    ) -> dict[str, float]:
        """Compute CPR levels from prior trading day's OHLC.

        Resamples the engine's bar history to daily, takes the second-to-last
        completed day (today is in-progress), feeds calculate_cpr.

        Returns:
            CPR dict (pivot, bc, tc, r1-r3, s1-s3). Empty dict if not enough
            data (e.g. first day of bars, no prior day available).
        """
        try:
            df = engine.get_data()
        except Exception:
            return {}
        if df is None or df.empty:
            return {}

        # Cache key: prior-trading-day's date. Pin cache to the latest bar's
        # date so multiple condition checks within the same session reuse it.
        latest_date = df.index[-1].date()
        cache_key = (symbol or "_", latest_date.isoformat())
        cached = self._cpr_cache.get(cache_key)
        if cached is not None:
            return cached

        # Resample 1-minute bars to daily OHLC
        try:
            daily = df.resample("1D").agg(
                {"open": "first", "high": "max", "low": "min",
                 "close": "last", "volume": "sum"}
            ).dropna()
        except Exception:
            return {}

        # Need at least one fully-completed prior day
        if len(daily) < 2:
            self._cpr_cache[cache_key] = {}
            return {}

        # Use the last fully-closed day = second-to-last row when today's
        # session is in progress; or last row when latest_date is not today.
        prev_day_df = daily.iloc[[-2]]
        if daily.index[-1].date() != latest_date:
            # Latest bar is on a different (older) date; treat last as prev.
            prev_day_df = daily.iloc[[-1]]

        try:
            cpr = engine.calculate_cpr(prev_day_df)
        except Exception as exc:
            logger.debug(f"calculate_cpr failed: {exc}")
            cpr = {}

        self._cpr_cache[cache_key] = cpr
        return cpr

    def register_instrument(
        self, symbol: str, engine: IndicatorEngine
    ) -> None:
        """Register an indicator engine for a symbol.

        Args:
            symbol: Trading symbol.
            engine: IndicatorEngine instance for this symbol.
        """
        self._indicator_engines[symbol] = engine
        logger.debug(f"Registered indicator engine for {symbol}")

    def get_engine(self, symbol: str) -> IndicatorEngine | None:
        """Get the cached IndicatorEngine for a symbol.

        Args:
            symbol: Trading symbol.

        Returns:
            Cached IndicatorEngine or None if not registered.
        """
        return self._indicator_engines.get(symbol)

    def evaluate(
        self,
        condition: Condition,
        bars: dict[str, pd.DataFrame],
        symbol: str | None = None,
    ) -> bool:
        """Evaluate a single condition against market data.

        Args:
            condition: The Condition object (JSON-native format).
            bars: Dictionary mapping symbol to OHLCV DataFrame.
            symbol: Optional specific symbol to evaluate for.

        Returns:
            True if condition is met, False otherwise.
        """
        # Resolve the primary indicator series
        indicator_series = self._resolve_indicator_name(
            condition.indicator, condition.parameters, bars, symbol
        )
        if indicator_series is None:
            logger.debug(f"Could not resolve indicator '{condition.indicator}'")
            return False

        # Resolve the right-hand side: another indicator or a constant value
        if condition.against is not None:
            right_series = self._resolve_indicator_name(
                condition.against, condition.parameters, bars, symbol
            )
            if right_series is None:
                logger.debug(
                    f"Could not resolve 'against' indicator '{condition.against}'"
                )
                return False
        else:
            right_series = None

        lookback = condition.lookback

        # For crosses_above / crosses_below we need the full series
        if condition.comparison in (
            ComparisonOperator.CROSSES_ABOVE,
            ComparisonOperator.CROSSES_BELOW,
        ):
            if right_series is None:
                # Allow crossing a constant level (e.g. RSI crosses_above 30)
                if condition.value is not None:
                    right_series = pd.Series(
                        float(condition.value), index=indicator_series.index
                    )
                else:
                    logger.warning(
                        "crosses_above/crosses_below requires 'against' indicator "
                        "or a constant 'value'"
                    )
                    return False
            return self._check_cross(
                condition.comparison, indicator_series, right_series, lookback
            )

        # "near" comparison: check if indicator is within tolerance% of reference
        # Uses both 'against' (reference indicator) and 'value' (tolerance fraction)
        if condition.comparison == ComparisonOperator.NEAR:
            if right_series is None:
                logger.warning("'near' comparison requires 'against' indicator")
                return False
            left_val = self._get_value_at_lookback(indicator_series, lookback)
            ref_val = self._get_value_at_lookback(right_series, lookback)
            if left_val is None or ref_val is None or ref_val == 0:
                return False
            tolerance = float(condition.value) if condition.value is not None else 0.002
            result = abs(left_val - ref_val) / abs(ref_val) <= tolerance
            logger.debug(
                f"Condition: {condition.indicator} ({left_val:.4f}) "
                f"near {condition.against} ({ref_val:.4f}) "
                f"tolerance={tolerance} = {result}"
            )
            return result

        # Scalar comparison: resolve to single values at the lookback bar
        left_val = self._get_value_at_lookback(indicator_series, lookback)
        if right_series is not None:
            right_val = self._get_value_at_lookback(right_series, lookback)
        else:
            # Handle string value comparisons (e.g., RSI_Divergence == "bullish")
            if condition.value is not None:
                if isinstance(condition.value, str):
                    right_val = condition.value
                else:
                    right_val = float(condition.value)
            else:
                right_val = None

        if left_val is None or right_val is None:
            return False

        # Handle string comparisons (e.g., RSI_Divergence == "bullish")
        if isinstance(right_val, str):
            result = self._apply_string_operator(condition.comparison, left_val, right_val, condition.indicator)
        else:
            result = self._apply_operator(condition.comparison, left_val, right_val)

        logger.debug(
            f"Condition: {condition.indicator} ({left_val:.4f}) "
            f"{condition.comparison.value} "
            f"{condition.against or condition.value} ({right_val:.4f}) = {result}"
        )
        return result

    def evaluate_all(
        self,
        conditions: list[Condition],
        bars: dict[str, pd.DataFrame],
        symbol: str | None = None,
    ) -> bool:
        """Evaluate all conditions (AND logic).

        Args:
            conditions: List of conditions to evaluate.
            bars: Dictionary mapping symbol to OHLCV DataFrame.
            symbol: Optional specific symbol to evaluate for.

        Returns:
            True if ALL conditions are met, False otherwise.
        """
        if not conditions:
            return True

        for cond in conditions:
            passed = self.evaluate(cond, bars, symbol)
            if not passed:
                logger.debug(
                    f"[{symbol or 'Strategy'}] Condition FAILED: "
                    f"{cond.indicator} {cond.comparison.value} "
                    f"{cond.against or cond.value}"
                )
                return False
                
        logger.debug(f"[{symbol or 'Strategy'}] All basic conditions MET.")
        return True

    # ------------------------------------------------------------------
    # Indicator resolution
    # ------------------------------------------------------------------

    def _resolve_indicator_name(
        self,
        name: str,
        parameters: dict[str, Any],
        bars: dict[str, pd.DataFrame],
        symbol: str | None,
    ) -> pd.Series | None:
        """Resolve an indicator name string to a pandas Series.

        Handles:
        - Price columns: Spot_Price, close, open, high, low, volume
        - Standard indicators: MACD, MACD_Signal, RSI, EMA_N, SMA_N,
          VWAP, ATR, Supertrend, Bollinger Bands, etc.

        Args:
            name: Indicator name from the condition JSON.
            parameters: Indicator parameters dict.
            bars: Market data keyed by symbol.
            symbol: Target symbol.

        Returns:
            Series of indicator values, or None.
        """
        target_symbol = symbol
        if target_symbol is None:
            if not bars:
                return None
            target_symbol = list(bars.keys())[0]

        if target_symbol not in bars:
            logger.warning(f"No data for symbol: {target_symbol}")
            return None

        df = bars[target_symbol]
        name_lower = name.lower()

        # Volume with ratio mode — when parameters request relative comparison
        if name_lower == "volume" and parameters and (
            parameters.get("relative_to") or parameters.get("type") == "average_multiplier"
        ):
            if "volume" not in df.columns:
                return None
            period = parameters.get("period", 20)
            avg_vol = df["volume"].rolling(window=period).mean()
            ratio = df["volume"] / avg_vol.replace(0, float("nan"))
            return ratio.fillna(0)

        # Raw price / volume columns
        if name_lower in _PRICE_COLUMNS:
            col = "close" if name_lower in ("spot_price", "price") else name_lower
            if col in df.columns:
                return df[col]
            return None

        # Get or create indicator engine
        if target_symbol not in self._indicator_engines:
            self._indicator_engines[target_symbol] = IndicatorEngine(df)
        else:
            self._indicator_engines[target_symbol].update(df)

        engine = self._indicator_engines[target_symbol]

        try:
            return self._compute_indicator(engine, name, parameters, target_symbol)
        except Exception as e:
            # These failures are almost always "insufficient warmup bars"
            # (pandas-ta returns None until enough data is available). Log at
            # DEBUG so warmup doesn't flood the backtest console. Real
            # indicator bugs still surface via the returned None path.
            msg = str(e)
            if (
                "calculation failed" in msg
                or "No data loaded" in msg
                or "insufficient" in msg.lower()
            ):
                logger.debug(f"Indicator '{name}' unavailable (warmup): {e}")
            else:
                logger.error(f"Failed to compute indicator '{name}': {e}")
            return None

    def _compute_indicator(
        self,
        engine: IndicatorEngine,
        name: str,
        parameters: dict[str, Any],
        target_symbol: str | None = None,
    ) -> pd.Series | None:
        """Compute an indicator by name using the engine.

        Args:
            engine: IndicatorEngine instance.
            name: Indicator name string from the JSON condition.
            parameters: Indicator parameters.

        Returns:
            Series with indicator values.
        """
        name_lower = name.lower()

        # MACD line
        if name_lower == "macd":
            result = engine.macd(
                fast=parameters.get("fast", 12),
                slow=parameters.get("slow", 26),
                signal=parameters.get("signal", 9),
            )
            return result["macd"]

        # MACD Signal line
        if name_lower == "macd_signal":
            result = engine.macd(
                fast=parameters.get("fast", 12),
                slow=parameters.get("slow", 26),
                signal=parameters.get("signal", 9),
            )
            return result["signal"]

        # MACD Histogram
        if name_lower == "macd_histogram":
            result = engine.macd(
                fast=parameters.get("fast", 12),
                slow=parameters.get("slow", 26),
                signal=parameters.get("signal", 9),
            )
            return result["histogram"]

        # RSI
        if name_lower == "rsi":
            return engine.rsi(
                parameters.get("length", parameters.get("period", 14))
            )

        # EMA with optional period suffix (e.g. EMA_9, EMA_21)
        if name_lower.startswith("ema"):
            period = parameters.get("period")
            if period is None:
                parts = name.split("_")
                period = (
                    int(parts[1])
                    if len(parts) > 1 and parts[1].isdigit()
                    else 20
                )
            return engine.ema(period)

        # SMA with optional period suffix
        if name_lower.startswith("sma"):
            period = parameters.get("period")
            if period is None:
                parts = name.split("_")
                period = (
                    int(parts[1])
                    if len(parts) > 1 and parts[1].isdigit()
                    else 20
                )
            return engine.sma(period)

        # VWAP
        if name_lower == "vwap":
            return engine.vwap()

        # ATR
        if name_lower == "atr":
            return engine.atr(parameters.get("period", 14))

        # Supertrend
        if name_lower == "supertrend":
            result = engine.supertrend(
                period=parameters.get("period", 7),
                multiplier=parameters.get("multiplier", 3.0),
            )
            return result["supertrend"]

        # Bollinger Bands
        if name_lower.startswith("bb") or name_lower.startswith("bollinger"):
            result = engine.bbands(
                period=parameters.get("period", 20),
                std=parameters.get("std", 2.0),
            )
            band = parameters.get("band", "middle")
            return result[band]

        # ADX (Average Directional Index)
        if name_lower == "adx":
            return engine.adx(parameters.get("period", 14))

        # OBV (On-Balance Volume)
        if name_lower == "obv":
            return engine.obv()

        # Volume SMA
        if name_lower == "volume_sma" or name_lower == "vol_sma":
            return engine.volume_sma(parameters.get("period", 20))

        # CPR (Central Pivot Range) levels
        if name_lower.startswith("cpr_"):
            # CPR levels are passed via parameters from daily data calculation
            cpr_level = name_lower.replace("cpr_", "")
            cpr_data = parameters.get("cpr_data", {})

            # Auto-compute CPR from prior-day bars when caller didn't supply
            # cpr_data. Avoids requiring every strategy config to inject
            # daily OHLC. Cached per (symbol, date) on the evaluator.
            if not cpr_data:
                cpr_data = self._auto_cpr(engine, target_symbol)

            if not cpr_data:
                logger.debug(f"CPR data unavailable for {name}")
                return None

            # Map CPR level names to keys in cpr_data
            level_map = {
                "pivot": "pivot",
                "p": "pivot",
                "bc": "bc",
                "bottom_central": "bc",
                "tc": "tc",
                "top_central": "tc",
                "r1": "r1",
                "r2": "r2",
                "r3": "r3",
                "s1": "s1",
                "s2": "s2",
                "s3": "s3",
            }

            key = level_map.get(cpr_level)
            if key and key in cpr_data:
                # Return a constant series with the CPR value
                df = engine.get_data()
                value = cpr_data[key]
                return pd.Series([value] * len(df), index=df.index)

            logger.warning(f"Unknown CPR level: {cpr_level}")
            return None

        # RSI Divergence detection
        if name_lower == "rsi_divergence":
            # Import here to avoid circular imports
            try:
                from src.indicators.divergence import detect_rsi_divergence

                df = engine.get_data()
                result = detect_rsi_divergence(
                    df,
                    lookback=parameters.get("lookback", 10),
                    rsi_length=parameters.get("rsi_length", 14),
                    min_swing_pct=parameters.get("min_swing_pct", 0.001),
                )
                # Return a series with the divergence type encoded as numeric
                # 1 = bullish, -1 = bearish, 0 = none
                divergence_value = 0
                if result.type == "bullish":
                    divergence_value = 1
                elif result.type == "bearish":
                    divergence_value = -1

                return pd.Series([divergence_value] * len(df), index=df.index)
            except Exception as e:
                logger.error(f"RSI divergence calculation failed: {e}")
                return None

        # Support/Resistance Proximity
        if name_lower in ("support_proximity", "resistance_proximity"):
            try:
                from src.indicators.divergence import (
                    get_support_resistance_levels,
                    calculate_proximity_to_level,
                )

                df = engine.get_data()
                levels_data = get_support_resistance_levels(
                    df,
                    lookback=parameters.get("lookback", 20),
                    zone_width_pct=parameters.get("zone_width_pct", 0.2),
                )

                proximity_pct = parameters.get("proximity_pct", 0.3)
                current_price = float(df["close"].iloc[-1])

                if name_lower == "support_proximity":
                    levels = levels_data.get("support", [])
                    is_near, distance, _ = calculate_proximity_to_level(
                        current_price, levels, proximity_pct
                    )
                else:  # resistance_proximity
                    levels = levels_data.get("resistance", [])
                    is_near, distance, _ = calculate_proximity_to_level(
                        current_price, levels, proximity_pct
                    )

                # Return series with proximity distance as percentage
                return pd.Series([distance] * len(df), index=df.index)
            except Exception as e:
                logger.error(f"Support/Resistance proximity calculation failed: {e}")
                return None

        # Previous Day High (PDH)
        if name_lower == "pdh":
            df = engine.get_data()
            if len(df) < 2:
                return None
            today = df.index[-1].date() if hasattr(df.index[-1], 'date') else df.index[-1]
            prev_day_mask = df.index.date < today if hasattr(df.index[0], 'date') else pd.Series([False] * len(df))
            prev_day_data = df[prev_day_mask]
            if len(prev_day_data) == 0:
                return None
            last_prev_day = prev_day_data.index[-1].date() if hasattr(prev_day_data.index[-1], 'date') else None
            if last_prev_day is None:
                return None
            prev_day_only = prev_day_data[prev_day_data.index.date == last_prev_day]
            pdh_value = float(prev_day_only["high"].max())
            return pd.Series([pdh_value] * len(df), index=df.index)

        # Previous Day Low (PDL)
        if name_lower == "pdl":
            df = engine.get_data()
            if len(df) < 2:
                return None
            today = df.index[-1].date() if hasattr(df.index[-1], 'date') else df.index[-1]
            prev_day_mask = df.index.date < today if hasattr(df.index[0], 'date') else pd.Series([False] * len(df))
            prev_day_data = df[prev_day_mask]
            if len(prev_day_data) == 0:
                return None
            last_prev_day = prev_day_data.index[-1].date() if hasattr(prev_day_data.index[-1], 'date') else None
            if last_prev_day is None:
                return None
            prev_day_only = prev_day_data[prev_day_data.index.date == last_prev_day]
            pdl_value = float(prev_day_only["low"].min())
            return pd.Series([pdl_value] * len(df), index=df.index)

        # Previous Day Close (PDC)
        if name_lower == "pdc":
            df = engine.get_data()
            if len(df) < 2:
                return None
            today = df.index[-1].date() if hasattr(df.index[-1], 'date') else df.index[-1]
            prev_day_mask = df.index.date < today
            prev_day_data = df[prev_day_mask]
            if len(prev_day_data) == 0:
                return None
            return pd.Series([float(prev_day_data["close"].iloc[-1])] * len(df), index=df.index)

        # Gap percentage: (today_open - prev_close) / prev_close * 100
        if name_lower in ("gap", "gap_pct"):
            df = engine.get_data()
            if len(df) < 2:
                return None
            today = df.index[-1].date() if hasattr(df.index[-1], 'date') else df.index[-1]
            today_mask = df.index.date == today
            today_data = df[today_mask]
            prev_day_mask = df.index.date < today
            prev_day_data = df[prev_day_mask]
            if len(today_data) == 0 or len(prev_day_data) == 0:
                return None
            today_open = float(today_data["open"].iloc[0])
            prev_close = float(prev_day_data["close"].iloc[-1])
            if prev_close == 0:
                return None
            gap_pct = (today_open - prev_close) / prev_close * 100
            return pd.Series([gap_pct] * len(df), index=df.index)

        # Open Type: detects Open=High (bearish) or Open=Low (bullish)
        # Returns 1 for Open=High, -1 for Open=Low, 0 for neither
        if name_lower == "open_type":
            df = engine.get_data()
            if len(df) < 1:
                return None
            today = df.index[-1].date() if hasattr(df.index[-1], 'date') else df.index[-1]
            today_mask = df.index.date == today
            today_data = df[today_mask]
            if len(today_data) == 0:
                return None
            today_open = float(today_data["open"].iloc[0])
            today_high = float(today_data["high"].max())
            today_low = float(today_data["low"].min())
            tick = 5  # 5 paisa tick size
            if abs(today_open - today_high) <= tick:
                val = 1  # Open = High (bearish signal)
            elif abs(today_open - today_low) <= tick:
                val = -1  # Open = Low (bullish signal)
            else:
                val = 0
            return pd.Series([val] * len(df), index=df.index)

        # Day Type: 1 = expiry day, 0 = non-expiry day
        if name_lower == "day_type":
            df = engine.get_data()
            if len(df) < 1:
                return None
            today = df.index[-1].date() if hasattr(df.index[-1], 'date') else df.index[-1]
            weekday = today.weekday()  # 0=Mon, 1=Tue, 2=Wed, 3=Thu, 4=Fri
            expiry_day_str = parameters.get("expiry_day", "wednesday").lower()
            expiry_day_map = {"monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3, "friday": 4}
            is_expiry = 1 if weekday == expiry_day_map.get(expiry_day_str, 2) else 0
            return pd.Series([is_expiry] * len(df), index=df.index)

        # India VIX (placeholder — returns from parameters if provided)
        if name_lower == "india_vix":
            df = engine.get_data()
            vix_value = parameters.get("current_vix", 15.0)
            return pd.Series([vix_value] * len(df), index=df.index)

        # Event Filter (placeholder — returns 1 for no event, 0 for event day)
        if name_lower == "event_filter":
            df = engine.get_data()
            # Default: no event (safe to trade)
            return pd.Series([1] * len(df), index=df.index)

        # Days to expiry — calculates days until next weekly expiry
        if name_lower == "days_to_expiry":
            from datetime import date, timedelta

            df = engine.get_data()
            today = df.index[-1].date() if hasattr(df.index[-1], "date") else date.today()
            expiry_day_str = parameters.get("expiry_day", "wednesday").lower()
            expiry_day_map = {"monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3, "friday": 4}
            target_weekday = expiry_day_map.get(expiry_day_str, 2)
            current_weekday = today.weekday()
            days_ahead = (target_weekday - current_weekday) % 7
            if days_ahead == 0:
                # Today is expiry day — 0 days to expiry
                days_ahead = 0
            dte = float(days_ahead)
            return pd.Series([dte] * len(df), index=df.index)

        # Synthetic ATM IV (Virtual Indicator for Backtesting)
        # Uses ATR-based synthesis when real IV data is unavailable.
        # If ATR is higher than its 50-bar MA, IV > 20; if lower, IV < 20.
        # Falls back to parameter-supplied value when insufficient data.
        if name_lower == "atm_iv":
            df = engine.get_data()
            # Try BSM-based IV if options pricing module is available
            try:
                from src.research.options_pricing import implied_volatility as _iv_calc
                iv_from_params = parameters.get("current_iv")
                if iv_from_params is not None:
                    return pd.Series([float(iv_from_params)] * len(df), index=df.index)
            except ImportError:
                pass

            # Synthesize IV from ATR ratio (works in backtests without real IV)
            try:
                atr = engine.atr(14)
                atr_ma = atr.rolling(50).mean()
                # Scale: if ATR == its MA → IV = 20; 2× MA → IV ≈ 40
                synth_iv = 20 * (atr / atr_ma.replace(0, float("nan")))
                synth_iv = synth_iv.clip(lower=5.0, upper=80.0)  # Bound to realistic range
                return synth_iv.fillna(20.0)
            except Exception as e:
                logger.debug(f"ATM_IV ATR-based synthesis failed: {e}")

            # Final fallback: static parameter value
            iv_value = parameters.get("current_iv", 20.0)
            return pd.Series([float(iv_value)] * len(df), index=df.index)

        # ORB (Opening Range Breakout) indicators
        if name_lower in ("orb_breakout", "orb_breakdown", "orb_high", "orb_low"):
            df = engine.get_data()
            if len(df) < 1:
                return None
            today = df.index[-1].date() if hasattr(df.index[-1], "date") else df.index[-1]
            today_mask = df.index.date == today if hasattr(df.index[0], "date") else pd.Series([True] * len(df))
            today_data = df[today_mask]
            if len(today_data) == 0:
                return None
            orb_minutes = parameters.get("orb_minutes", 15)
            # Use time-based opening range window: market opens at 09:15 IST
            market_open = pd.Timestamp(today, tz=IST).replace(hour=9, minute=15, second=0, microsecond=0)
            orb_end = market_open + pd.Timedelta(minutes=orb_minutes)
            orb_slice = today_data[(today_data.index >= market_open) & (today_data.index < orb_end)]
            if len(orb_slice) == 0:
                # Fallback: use first N bars if time filtering fails (e.g., missing times)
                orb_bars = parameters.get("orb_bars", max(1, orb_minutes // 5))
                orb_slice = today_data.iloc[:orb_bars]
                if len(orb_slice) == 0:
                    return None
            orb_high_val = float(orb_slice["high"].max())
            orb_low_val = float(orb_slice["low"].min())

            if name_lower == "orb_high":
                return pd.Series([orb_high_val] * len(df), index=df.index)
            elif name_lower == "orb_low":
                return pd.Series([orb_low_val] * len(df), index=df.index)
            elif name_lower == "orb_breakout":
                # 1 if current close > ORB high, 0 otherwise
                current_close = float(df["close"].iloc[-1])
                val = 1.0 if current_close > orb_high_val else 0.0
                return pd.Series([val] * len(df), index=df.index)
            elif name_lower == "orb_breakdown":
                # 1 if current close < ORB low, 0 otherwise
                current_close = float(df["close"].iloc[-1])
                val = 1.0 if current_close < orb_low_val else 0.0
                return pd.Series([val] * len(df), index=df.index)

        # First 5-minute candle close
        if name_lower == "first_5min_close":
            df = engine.get_data()
            if len(df) < 1:
                return None
            today = df.index[-1].date() if hasattr(df.index[-1], "date") else df.index[-1]
            today_mask = df.index.date == today if hasattr(df.index[0], "date") else pd.Series([True] * len(df))
            today_data = df[today_mask]
            if len(today_data) == 0:
                return None
            first_close = float(today_data["close"].iloc[0])
            return pd.Series([first_close] * len(df), index=df.index)

        # Volume ratio (current volume / N-period average)
        if name_lower == "volume_ratio":
            df = engine.get_data()
            if "volume" not in df.columns or len(df) < 2:
                return None
            period = parameters.get("period", 20)
            avg_vol = df["volume"].rolling(window=period).mean()
            ratio = df["volume"] / avg_vol.replace(0, float("nan"))
            return ratio.fillna(0)

        # Days To Expiry (Virtual Indicator)
        if name_lower == "days_to_expiry":
            df = engine.get_data()
            try:
                # Assuming Indian markets: BankNifty typically expires on Wednesday (2)
                # Nifty typically expires on Thursday (3)
                expiry_target = parameters.get("expiry_day", 2)  # Default BankNifty (Wednesday)
                current_day = df.index.dayofweek
                # Calculate days to expiry
                dte = (expiry_target - current_day) % 7
                return pd.Series(dte, index=df.index)
            except Exception as e:
                logger.error(f"Days_To_Expiry calculation failed: {e}")
                return None



        logger.warning(f"Unknown indicator to compute: {name}")
        return None

    # ------------------------------------------------------------------
    # Value extraction helpers
    # ------------------------------------------------------------------

    def _get_value_at_lookback(
        self, series: pd.Series, lookback: int
    ) -> float | None:
        """Get value at specified lookback from the end.

        Args:
            series: Data series.
            lookback: Number of bars back (0 = current).

        Returns:
            Value at the specified position, or None if insufficient data.
        """
        if series is None or len(series) == 0:
            return None
        if len(series) <= lookback:
            return float(series.iloc[0])
        return float(series.iloc[-(lookback + 1)])

    # ------------------------------------------------------------------
    # Comparison operators
    # ------------------------------------------------------------------

    def _apply_operator(
        self,
        operator: ComparisonOperator,
        left: float,
        right: float,
    ) -> bool:
        """Apply a simple comparison operator.

        Args:
            operator: The comparison operator.
            left: Left operand value.
            right: Right operand value.

        Returns:
            True if condition is met.
        """
        if operator == ComparisonOperator.GT:
            return left > right
        elif operator == ComparisonOperator.LT:
            return left < right
        elif operator == ComparisonOperator.GTE:
            return left >= right
        elif operator == ComparisonOperator.LTE:
            return left <= right
        elif operator == ComparisonOperator.EQ:
            return abs(left - right) < 1e-10
        else:
            logger.warning(f"Unexpected operator for scalar comparison: {operator}")
            return False

    def _apply_string_operator(
        self,
        operator: ComparisonOperator,
        left: float,
        right: str,
        indicator_name: str,
    ) -> bool:
        """Apply comparison operator for string values (e.g., divergence types).

        Handles special indicators that encode categorical values as numbers:
        - RSI_Divergence: 1=bullish, -1=bearish, 0=none

        Args:
            operator: The comparison operator.
            left: Left operand value (numeric encoding).
            right: Right operand value (string like "bullish", "bearish").
            indicator_name: Name of the indicator for context.

        Returns:
            True if condition is met.
        """
        if operator != ComparisonOperator.EQ:
            logger.warning(f"String comparison only supports '==' operator, got {operator}")
            return False

        indicator_lower = indicator_name.lower()

        # RSI_Divergence encoding: 1=bullish, -1=bearish, 0=none
        if indicator_lower == "rsi_divergence":
            if right.lower() == "bullish":
                return left == 1
            elif right.lower() == "bearish":
                return left == -1
            elif right.lower() == "none":
                return left == 0
            else:
                logger.warning(f"Unknown RSI_Divergence value: {right}")
                return False

        # Generic string comparison - try to match as equality
        logger.debug(f"String comparison for {indicator_name}: {left} == {right}")
        return str(left).lower() == right.lower()

    # ------------------------------------------------------------------
    # Cross detection
    # ------------------------------------------------------------------

    def _check_cross(
        self,
        operator: ComparisonOperator,
        left_series: pd.Series,
        right_series: pd.Series,
        lookback: int,
    ) -> bool:
        """Check if left crosses above/below right at lookback.

        Args:
            operator: CROSSES_ABOVE or CROSSES_BELOW.
            left_series: Left indicator series.
            right_series: Right indicator series.
            lookback: Number of bars back.

        Returns:
            True if a crossover occurred at the specified bar.
        """
        if len(left_series) < lookback + 2 or len(right_series) < lookback + 2:
            return False

        idx = -(lookback + 1)
        prev_idx = idx - 1

        current_left = float(left_series.iloc[idx])
        current_right = float(right_series.iloc[idx])
        prev_left = float(left_series.iloc[prev_idx])
        prev_right = float(right_series.iloc[prev_idx])

        if operator == ComparisonOperator.CROSSES_ABOVE:
            return current_left > current_right and prev_left <= prev_right
        elif operator == ComparisonOperator.CROSSES_BELOW:
            return current_left < current_right and prev_left >= prev_right

        return False

    # ------------------------------------------------------------------
    # Time / day conditions
    # ------------------------------------------------------------------

    def evaluate_time_condition(
        self,
        current_time: time,
        start_time: time | None = None,
        end_time: time | None = None,
    ) -> bool:
        """Evaluate if current time is within trading hours.

        Args:
            current_time: Current time to check.
            start_time: Trading start time (inclusive).
            end_time: Trading end time (inclusive).

        Returns:
            True if within trading hours.
        """
        if start_time and current_time < start_time:
            return False
        if end_time and current_time > end_time:
            return False
        return True

    def evaluate_day_condition(
        self,
        current_date: datetime,
        allowed_days: list[int] | None = None,
    ) -> bool:
        """Evaluate if current day is a trading day.

        Args:
            current_date: Current datetime (timezone-aware).
            allowed_days: List of allowed weekdays (0=Monday, 6=Sunday).

        Returns:
            True if trading is allowed on this day.
        """
        if allowed_days is None:
            allowed_days = [0, 1, 2, 3, 4]  # Mon-Fri

        # Ensure timezone-aware
        if current_date.tzinfo is None:
            current_date = current_date.replace(tzinfo=IST)
        else:
            current_date = current_date.astimezone(IST)

        weekday = current_date.weekday()
        return weekday in allowed_days

    # ------------------------------------------------------------------
    # Enhanced Entry Filters (C1 Implementation)
    # ------------------------------------------------------------------

    def evaluate_adx_filter(
        self,
        bars: dict[str, pd.DataFrame],
        symbol: str | None = None,
        min_adx: float = ADX_THRESHOLD_TRENDING,
        max_adx: float | None = None,
        adx_period: int = 14,
    ) -> FilterResult:
        """Evaluate ADX trend strength filter.

        ADX > 25 = trending (good for trend-following)
        ADX > 40 = strong trend (high conviction)
        ADX < 20 = ranging (avoid trend-following)

        Args:
            bars: Dictionary mapping symbol to OHLCV DataFrame.
            symbol: Optional specific symbol to evaluate.
            min_adx: Minimum ADX value required (default 25 for trending).
            max_adx: Optional maximum ADX value (None = no upper limit).
            adx_period: Period for ADX calculation. Default 14.

        Returns:
            FilterResult with pass/fail status and trend strength details.
        """
        target_symbol = symbol
        if target_symbol is None:
            if bars is None or (isinstance(bars, dict) and not bars):
                return FilterResult(
                    filter_name="adx_filter",
                    passed=False,
                    score=0.0,
                    details={"error": "No bars provided"},
                )
            target_symbol = list(bars.keys())[0] if isinstance(bars, dict) else "default"

        # Handle both dict and DataFrame inputs
        if isinstance(bars, dict):
            if target_symbol not in bars:
                return FilterResult(
                    filter_name="adx_filter",
                    passed=False,
                    score=0.0,
                    details={"error": f"No data for symbol: {target_symbol}"},
                )
            df = bars[target_symbol]
        else:
            # bars is a DataFrame directly
            df = bars

        # Get or create indicator engine
        if target_symbol not in self._indicator_engines:
            self._indicator_engines[target_symbol] = IndicatorEngine(df)
        else:
            self._indicator_engines[target_symbol].update(df)

        engine = self._indicator_engines[target_symbol]

        try:
            adx_series = engine.adx(period=adx_period)
            current_adx = float(adx_series.iloc[-1])

            # Determine trend strength classification
            if current_adx >= ADX_THRESHOLD_STRONG:
                trend_strength = TrendStrength.STRONG_TREND
                strength_score = min(1.0, current_adx / 50.0)  # Cap at 50
            elif current_adx >= ADX_THRESHOLD_TRENDING:
                trend_strength = TrendStrength.TRENDING
                strength_score = current_adx / 40.0
            elif current_adx >= ADX_THRESHOLD_RANGING:
                trend_strength = TrendStrength.RANGING
                strength_score = current_adx / 25.0
            else:
                trend_strength = TrendStrength.RANGING
                strength_score = current_adx / 20.0

            # Check if ADX meets criteria
            passed = current_adx >= min_adx
            if max_adx is not None:
                passed = passed and (current_adx <= max_adx)

            return FilterResult(
                filter_name="adx_filter",
                passed=passed,
                score=strength_score,
                details={
                    "adx_value": current_adx,
                    "trend_strength": trend_strength.value,
                    "min_required": min_adx,
                    "max_allowed": max_adx,
                    "adx_period": adx_period,
                },
            )

        except Exception as e:
            logger.error(f"Failed to evaluate ADX filter for {target_symbol}: {e}")
            return FilterResult(
                filter_name="adx_filter",
                passed=False,
                score=0.0,
                details={"error": str(e)},
            )

    def evaluate_volume_confirmation(
        self,
        bars: dict[str, pd.DataFrame],
        symbol: str | None = None,
        volume_multiplier: float = VOLUME_CONFIRMATION_MULTIPLIER,
        volume_sma_period: int = 20,
        check_obv_trend: bool = True,
        obv_lookback: int = OBV_TREND_LOOKBACK,
        signal_direction: str = "long",  # "long" or "short"
    ) -> FilterResult:
        """Evaluate volume confirmation filter.

        Checks:
        1. Current volume > X * average volume (default 1.2x)
        2. OBV trending in signal direction (optional)

        Args:
            bars: Dictionary mapping symbol to OHLCV DataFrame.
            symbol: Optional specific symbol to evaluate.
            volume_multiplier: Current volume must exceed this multiple of avg volume.
            volume_sma_period: Period for volume SMA calculation.
            check_obv_trend: Whether to check OBV trend direction.
            obv_lookback: Lookback period for OBV trend detection.
            signal_direction: Expected OBV direction ("long" or "short").

        Returns:
            FilterResult with pass/fail status and volume analysis details.
        """
        target_symbol = symbol
        if target_symbol is None:
            if bars is None or (isinstance(bars, dict) and not bars):
                return FilterResult(
                    filter_name="volume_confirmation",
                    passed=False,
                    score=0.0,
                    details={"error": "No bars provided"},
                )
            target_symbol = list(bars.keys())[0] if isinstance(bars, dict) else "default"

        # Handle both dict and DataFrame inputs
        if isinstance(bars, dict):
            if target_symbol not in bars:
                return FilterResult(
                    filter_name="volume_confirmation",
                    passed=False,
                    score=0.0,
                    details={"error": f"No data for symbol: {target_symbol}"},
                )
            df = bars[target_symbol]
        else:
            df = bars

        # Get or create indicator engine
        if target_symbol not in self._indicator_engines:
            self._indicator_engines[target_symbol] = IndicatorEngine(df)
        else:
            self._indicator_engines[target_symbol].update(df)

        engine = self._indicator_engines[target_symbol]

        try:
            details = {}
            score = 0.0
            volume_passed = False
            obv_passed = True  # Default to True if not checking

            # Check volume vs average
            current_volume = float(df["volume"].iloc[-1])
            volume_sma = engine.volume_sma(period=volume_sma_period)
            avg_volume = float(volume_sma.iloc[-1])

            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 0
            volume_passed = volume_ratio >= volume_multiplier

            details["current_volume"] = current_volume
            details["avg_volume"] = avg_volume
            details["volume_ratio"] = round(volume_ratio, 2)
            details["volume_threshold"] = volume_multiplier

            # Volume score: 0.5 base + up to 0.5 based on how much it exceeds
            if volume_passed:
                score += 0.5 + min(0.5, (volume_ratio - volume_multiplier) * 0.25)
            else:
                score += max(0, volume_ratio / (volume_multiplier * 2))

            # Check OBV trend if requested
            if check_obv_trend:
                obv_series = engine.obv()

                if len(obv_series) >= obv_lookback + 1:
                    current_obv = float(obv_series.iloc[-1])
                    past_obv = float(obv_series.iloc[-(obv_lookback + 1)])

                    obv_trending_up = current_obv > past_obv
                    obv_trending_down = current_obv < past_obv

                    details["obv_current"] = current_obv
                    details["obv_past"] = past_obv
                    details["obv_lookback"] = obv_lookback
                    details["signal_direction"] = signal_direction

                    if signal_direction == "long":
                        obv_passed = obv_trending_up
                        details["obv_trend"] = "up" if obv_trending_up else "down"
                    else:  # short
                        obv_passed = obv_trending_down
                        details["obv_trend"] = "down" if obv_trending_down else "up"

                    if obv_passed:
                        score += 0.3
                    details["obv_passed"] = obv_passed
                else:
                    details["obv_error"] = "Insufficient data for OBV trend"
                    obv_passed = False

            passed = volume_passed and obv_passed

            return FilterResult(
                filter_name="volume_confirmation",
                passed=passed,
                score=min(1.0, score),
                details=details,
            )

        except Exception as e:
            logger.error(
                f"Failed to evaluate volume confirmation for {target_symbol}: {e}"
            )
            return FilterResult(
                filter_name="volume_confirmation",
                passed=False,
                score=0.0,
                details={"error": str(e)},
            )

    def evaluate_multi_timeframe_confluence(
        self,
        bars_by_timeframe: dict[str, dict[str, pd.DataFrame]],
        symbol: str | None = None,
        timeframes: list[str] | None = None,
        min_aligned: int = 2,
        indicator_name: str = "ema",
        indicator_params: dict[str, Any] | None = None,
    ) -> MultiTimeframeConfluence:
        """Evaluate multi-timeframe confluence for entry confirmation.

        Checks alignment across multiple timeframes (e.g., 1min, 5min, 15min).
        Scores based on percentage of timeframes aligning.
        Requires at least min_aligned timeframes for entry.

        Args:
            bars_by_timeframe: Dict mapping timeframe to bars dict.
                Example: {"1m": {"RELIANCE": df1}, "5m": {"RELIANCE": df2}}
            symbol: Optional specific symbol to evaluate.
            timeframes: List of timeframes to check (default: ["1m", "5m", "15m"]).
            min_aligned: Minimum number of timeframes required for confluence.
            indicator_name: Indicator to check for alignment (e.g., "ema", "supertrend").
            indicator_params: Parameters for the indicator.

        Returns:
            MultiTimeframeConfluence with alignment analysis.
        """
        if timeframes is None:
            timeframes = ["1m", "5m", "15m"]

        if indicator_params is None:
            indicator_params = {}

        target_symbol = symbol
        if target_symbol is None:
            # Try to get symbol from first timeframe's bars
            for tf in timeframes:
                if tf in bars_by_timeframe and bars_by_timeframe[tf]:
                    target_symbol = list(bars_by_timeframe[tf].keys())[0]
                    break

        if target_symbol is None:
            return MultiTimeframeConfluence(
                timeframes=timeframes,
                alignment_score=0.0,
                aligned_count=0,
                total_count=len(timeframes),
                timeframe_signals={"error": "No symbol found"},
            )

        timeframe_signals: dict[str, str] = {}
        bullish_count = 0
        bearish_count = 0
        neutral_count = 0

        for tf in timeframes:
            if tf not in bars_by_timeframe:
                timeframe_signals[tf] = "no_data"
                neutral_count += 1
                continue

            tf_bars = bars_by_timeframe[tf]
            if target_symbol not in tf_bars:
                timeframe_signals[tf] = "no_data"
                neutral_count += 1
                continue

            df = tf_bars[target_symbol]

            # Get or create indicator engine for this timeframe
            engine_key = f"{target_symbol}_{tf}"
            if engine_key not in self._indicator_engines:
                self._indicator_engines[engine_key] = IndicatorEngine(df)
            else:
                self._indicator_engines[engine_key].update(df)

            engine = self._indicator_engines[engine_key]

            try:
                signal = self._get_timeframe_signal(
                    engine, indicator_name, indicator_params, df
                )
                timeframe_signals[tf] = signal

                if signal == "bullish":
                    bullish_count += 1
                elif signal == "bearish":
                    bearish_count += 1
                else:
                    neutral_count += 1

            except Exception as e:
                logger.warning(f"Failed to get signal for {tf}: {e}")
                timeframe_signals[tf] = "error"
                neutral_count += 1

        # Determine dominant direction
        if bullish_count >= bearish_count and bullish_count >= neutral_count:
            dominant_signal = "bullish"
            aligned_count = bullish_count
        elif bearish_count >= bullish_count and bearish_count >= neutral_count:
            dominant_signal = "bearish"
            aligned_count = bearish_count
        else:
            dominant_signal = "neutral"
            aligned_count = neutral_count

        total_count = len(timeframes)
        alignment_score = aligned_count / total_count if total_count > 0 else 0.0

        # Add dominant signal to timeframe_signals for reference
        timeframe_signals["_dominant"] = dominant_signal
        timeframe_signals["_aligned_count"] = str(aligned_count)

        return MultiTimeframeConfluence(
            timeframes=timeframes,
            alignment_score=alignment_score,
            aligned_count=aligned_count,
            total_count=total_count,
            timeframe_signals=timeframe_signals,
        )

    def _get_timeframe_signal(
        self,
        engine: IndicatorEngine,
        indicator_name: str,
        indicator_params: dict[str, Any],
        df: pd.DataFrame,
    ) -> str:
        """Get bullish/bearish signal from an indicator on a single timeframe.

        Args:
            engine: IndicatorEngine for this timeframe.
            indicator_name: Name of indicator to evaluate.
            indicator_params: Parameters for the indicator.
            df: OHLCV DataFrame.

        Returns:
            "bullish", "bearish", or "neutral".
        """
        name_lower = indicator_name.lower()

        if name_lower == "ema":
            # Default to EMA 9 vs EMA 21 crossover
            fast = indicator_params.get("fast", 9)
            slow = indicator_params.get("slow", 21)
            ema_fast = engine.ema(fast)
            ema_slow = engine.ema(slow)

            if ema_fast.iloc[-1] > ema_slow.iloc[-1]:
                return "bullish"
            elif ema_fast.iloc[-1] < ema_slow.iloc[-1]:
                return "bearish"
            return "neutral"

        elif name_lower == "supertrend":
            period = indicator_params.get("period", 7)
            multiplier = indicator_params.get("multiplier", 3.0)
            result = engine.supertrend(period=period, multiplier=multiplier)

            direction = result["direction"].iloc[-1]
            if direction == 1:
                return "bullish"
            elif direction == -1:
                return "bearish"
            return "neutral"

        elif name_lower == "rsi":
            period = indicator_params.get("period", 14)
            oversold = indicator_params.get("oversold", 30)
            overbought = indicator_params.get("overbought", 70)
            rsi_series = engine.rsi(period)
            current_rsi = rsi_series.iloc[-1]

            if current_rsi < oversold:
                return "bullish"  # Oversold = potential bounce up
            elif current_rsi > overbought:
                return "bearish"  # Overbought = potential pullback
            return "neutral"

        elif name_lower == "macd":
            fast = indicator_params.get("fast", 12)
            slow = indicator_params.get("slow", 26)
            signal = indicator_params.get("signal", 9)
            result = engine.macd(fast=fast, slow=slow, signal=signal)

            if result["macd"].iloc[-1] > result["signal"].iloc[-1]:
                return "bullish"
            elif result["macd"].iloc[-1] < result["signal"].iloc[-1]:
                return "bearish"
            return "neutral"

        elif name_lower == "vwap":
            vwap_series = engine.vwap()
            close = df["close"].iloc[-1]

            if close > vwap_series.iloc[-1]:
                return "bullish"
            elif close < vwap_series.iloc[-1]:
                return "bearish"
            return "neutral"

        else:
            # Default: compare close to SMA
            period = indicator_params.get("period", 20)
            sma_series = engine.sma(period)
            close = df["close"].iloc[-1]

            if close > sma_series.iloc[-1]:
                return "bullish"
            elif close < sma_series.iloc[-1]:
                return "bearish"
            return "neutral"

    def evaluate_all_enhanced_filters(
        self,
        bars: dict[str, pd.DataFrame],
        symbol: str | None = None,
        signal_direction: str = "long",
        bars_by_timeframe: dict[str, dict[str, pd.DataFrame]] | None = None,
        strict_mode: bool = False,
        adx_min: float = ADX_THRESHOLD_TRENDING,
        adx_max: float | None = None,
        volume_multiplier: float = VOLUME_CONFIRMATION_MULTIPLIER,
        require_confluence: bool = True,
        min_timeframes_aligned: int = 2,
    ) -> tuple[bool, list[FilterResult], MultiTimeframeConfluence | None]:
        """Evaluate all enhanced filters and return combined result.

        Args:
            bars: Primary timeframe bars dict.
            symbol: Symbol to evaluate.
            signal_direction: "long" or "short" for OBV check.
            bars_by_timeframe: Optional multi-timeframe bars for confluence check.
            strict_mode: If True, ALL filters must pass. If False, majority must pass.
            adx_min: Minimum ADX value required.
            adx_max: Optional maximum ADX value.
            volume_multiplier: Volume confirmation multiplier.
            require_confluence: Whether to require multi-timeframe confluence.
            min_timeframes_aligned: Minimum timeframes needed for confluence.

        Returns:
            Tuple of (overall_passed, list of FilterResults, confluence_result).
        """
        results: list[FilterResult] = []

        # 1. ADX Filter
        adx_result = self.evaluate_adx_filter(
            bars=bars,
            symbol=symbol,
            min_adx=adx_min,
            max_adx=adx_max,
        )
        results.append(adx_result)

        # 2. Volume Confirmation
        volume_result = self.evaluate_volume_confirmation(
            bars=bars,
            symbol=symbol,
            volume_multiplier=volume_multiplier,
            signal_direction=signal_direction,
        )
        results.append(volume_result)

        # 3. Multi-Timeframe Confluence (if data provided)
        confluence_result: MultiTimeframeConfluence | None = None
        if bars_by_timeframe and require_confluence:
            confluence_result = self.evaluate_multi_timeframe_confluence(
                bars_by_timeframe=bars_by_timeframe,
                symbol=symbol,
                min_aligned=min_timeframes_aligned,
            )

            # Create a FilterResult for confluence
            confluence_passed = confluence_result.has_confluence(min_timeframes_aligned)
            confluence_filter_result = FilterResult(
                filter_name="multi_timeframe_confluence",
                passed=confluence_passed,
                score=confluence_result.alignment_score,
                details={
                    "aligned_count": confluence_result.aligned_count,
                    "total_count": confluence_result.total_count,
                    "alignment_score": confluence_result.alignment_score,
                    "timeframe_signals": confluence_result.timeframe_signals,
                },
            )
            results.append(confluence_filter_result)

        # Log all filter results
        for result in results:
            status = "PASS" if result.passed else "FAIL"
            logger.debug(
                f"Filter '{result.filter_name}': {status} "
                f"(score: {result.score:.2f}, details: {result.details})"
            )

        # Determine overall pass/fail
        if strict_mode:
            # All filters must pass
            overall_passed = all(r.passed for r in results)
        else:
            # Majority of filters must pass (at least 2/3 or 1/2)
            min_passed = max(1, len(results) // 2)
            passed_count = sum(1 for r in results if r.passed)
            overall_passed = passed_count >= min_passed

        logger.info(
            f"Enhanced filters evaluation: {overall_passed} "
            f"({sum(1 for r in results if r.passed)}/{len(results)} passed, "
            f"strict_mode={strict_mode})"
        )

        return overall_passed, results, confluence_result

    # ------------------------------------------------------------------
    # ORB (Opening Range Breakout) Condition Evaluator
    # ------------------------------------------------------------------

    def evaluate_orb_condition(
        self,
        bars: dict[str, pd.DataFrame],
        direction: str,
        symbol: str | None = None,
        orb_minutes: int = 15,
    ) -> FilterResult:
        """Evaluate Opening Range Breakout condition.

        Defines ORB high/low from first N minutes of trading after market open (9:15 AM IST).
        Returns True when price breaks above high (breakout) or below low (breakdown).

        Args:
            bars: Dictionary mapping symbol to OHLCV DataFrame with datetime index.
            direction: "breakout" for long signal, "breakdown" for short signal.
            symbol: Optional specific symbol to evaluate.
            orb_minutes: Number of minutes after open to define ORB range (default 15).

        Returns:
            FilterResult with pass/fail status and ORB details including
            suggested stop loss at ORB midpoint.
        """
        target_symbol = symbol
        if target_symbol is None:
            if bars is None or (isinstance(bars, dict) and not bars):
                return FilterResult(
                    filter_name="orb_condition",
                    passed=False,
                    score=0.0,
                    details={"error": "No bars provided"},
                )
            target_symbol = list(bars.keys())[0] if isinstance(bars, dict) else "default"

        # Handle both dict and DataFrame inputs
        if isinstance(bars, dict):
            if target_symbol not in bars:
                return FilterResult(
                    filter_name="orb_condition",
                    passed=False,
                    score=0.0,
                    details={"error": f"No data for symbol: {target_symbol}"},
                )
            df = bars[target_symbol]
        else:
            df = bars

        if df is None or len(df) == 0:
            return FilterResult(
                filter_name="orb_condition",
                passed=False,
                score=0.0,
                details={"error": "Empty DataFrame"},
            )

        try:
            # Ensure index is datetime
            if not isinstance(df.index, pd.DatetimeIndex):
                return FilterResult(
                    filter_name="orb_condition",
                    passed=False,
                    score=0.0,
                    details={"error": "DataFrame index must be DatetimeIndex"},
                )

            # Market open time is 9:15 AM IST
            market_open_time = time(9, 15)

            # Get today's date from the last bar
            last_bar_time = df.index[-1]
            if last_bar_time.tzinfo is None:
                last_bar_time = last_bar_time.replace(tzinfo=IST)
            today = last_bar_time.date()

            # Calculate ORB window end time
            orb_start = datetime.combine(today, market_open_time).replace(tzinfo=IST)
            orb_end = orb_start + timedelta(minutes=orb_minutes)

            # Filter bars within ORB window (9:15 to 9:30 for 15min ORB)
            orb_mask = (df.index >= orb_start) & (df.index < orb_end)
            orb_bars = df[orb_mask]

            if len(orb_bars) == 0:
                return FilterResult(
                    filter_name="orb_condition",
                    passed=False,
                    score=0.0,
                    details={
                        "error": f"No bars in ORB window ({orb_start.time()} - {orb_end.time()})",
                        "orb_minutes": orb_minutes,
                    },
                )

            # Calculate ORB high and low
            orb_high = float(orb_bars["high"].max())
            orb_low = float(orb_bars["low"].min())
            orb_midpoint = (orb_high + orb_low) / 2

            # Get current price (last close)
            current_price = float(df["close"].iloc[-1])

            # Check if we have enough bars after ORB window
            current_time = df.index[-1]
            if current_time < orb_end:
                return FilterResult(
                    filter_name="orb_condition",
                    passed=False,
                    score=0.0,
                    details={
                        "error": "Still in ORB formation period",
                        "current_time": current_time.isoformat(),
                        "orb_end": orb_end.isoformat(),
                    },
                )

            # Evaluate breakout/breakdown
            details = {
                "orb_high": orb_high,
                "orb_low": orb_low,
                "orb_midpoint": orb_midpoint,
                "orb_range": orb_high - orb_low,
                "current_price": current_price,
                "orb_minutes": orb_minutes,
                "direction": direction,
            }

            if direction == "breakout":
                # For breakout: current price must be above ORB high
                passed = current_price > orb_high
                score = 1.0 if passed else max(0.0, (current_price - orb_low) / (orb_high - orb_low))
                details["breakout_triggered"] = passed
                details["distance_to_high"] = current_price - orb_high

            elif direction == "breakdown":
                # For breakdown: current price must be below ORB low
                passed = current_price < orb_low
                score = 1.0 if passed else max(0.0, (orb_high - current_price) / (orb_high - orb_low))
                details["breakdown_triggered"] = passed
                details["distance_to_low"] = orb_low - current_price

            else:
                return FilterResult(
                    filter_name="orb_condition",
                    passed=False,
                    score=0.0,
                    details={"error": f"Invalid direction: {direction}. Use 'breakout' or 'breakdown'"},
                )

            # Suggested stop loss at ORB midpoint for breakout/breakdown
            if direction == "breakout":
                suggested_sl = orb_midpoint
            else:  # breakdown
                suggested_sl = orb_midpoint

            details["suggested_stop_loss"] = suggested_sl
            details["sl_distance"] = abs(current_price - suggested_sl)

            return FilterResult(
                filter_name="orb_condition",
                passed=passed,
                score=score,
                details=details,
            )

        except Exception as e:
            logger.error(f"Failed to evaluate ORB condition for {target_symbol}: {e}")
            return FilterResult(
                filter_name="orb_condition",
                passed=False,
                score=0.0,
                details={"error": str(e)},
            )

    def get_orb_levels(
        self,
        bars: dict[str, pd.DataFrame],
        symbol: str | None = None,
        orb_minutes: int = 15,
    ) -> dict[str, float] | None:
        """Get ORB levels (high, low, midpoint) for a symbol.

        Args:
            bars: Dictionary mapping symbol to OHLCV DataFrame.
            symbol: Optional specific symbol.
            orb_minutes: Number of minutes for ORB calculation.

        Returns:
            Dictionary with orb_high, orb_low, orb_midpoint, orb_range or None if error.
        """
        target_symbol = symbol
        if target_symbol is None:
            if bars is None or (isinstance(bars, dict) and not bars):
                return None
            target_symbol = list(bars.keys())[0] if isinstance(bars, dict) else "default"

        if isinstance(bars, dict):
            if target_symbol not in bars:
                return None
            df = bars[target_symbol]
        else:
            df = bars

        if df is None or len(df) == 0:
            return None

        try:
            if not isinstance(df.index, pd.DatetimeIndex):
                return None

            market_open_time = time(9, 15)
            last_bar_time = df.index[-1]
            if last_bar_time.tzinfo is None:
                last_bar_time = last_bar_time.replace(tzinfo=IST)
            today = last_bar_time.date()

            orb_start = datetime.combine(today, market_open_time).replace(tzinfo=IST)
            orb_end = orb_start + timedelta(minutes=orb_minutes)

            orb_mask = (df.index >= orb_start) & (df.index < orb_end)
            orb_bars = df[orb_mask]

            if len(orb_bars) == 0:
                return None

            orb_high = float(orb_bars["high"].max())
            orb_low = float(orb_bars["low"].min())
            orb_midpoint = (orb_high + orb_low) / 2
            orb_range = orb_high - orb_low

            return {
                "orb_high": orb_high,
                "orb_low": orb_low,
                "orb_midpoint": orb_midpoint,
                "orb_range": orb_range,
                "orb_start": orb_start.isoformat(),
                "orb_end": orb_end.isoformat(),
            }

        except Exception as e:
            logger.error(f"Failed to get ORB levels for {target_symbol}: {e}")
            return None
