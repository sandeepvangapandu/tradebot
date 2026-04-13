"""Divergence detection for RSI and price.

Provides functions to detect bullish and bearish divergence patterns
between price action and RSI momentum indicator. Divergence signals
potential trend reversals when price and momentum disagree.

All prices are in PAISA (integer) - 1 Rupee = 100 paisa.
All timestamps in IST (Asia/Kolkata).
"""

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import pandas as pd
import pandas_ta as ta
from loguru import logger


@dataclass
class DivergenceResult:
    """Result container for divergence detection.

    Attributes:
        type: Divergence type - "bullish", "bearish", or None
        strength: Divergence strength score (0.0 - 1.0)
        price_swing_low: Price at the lower swing low (for bullish)
        price_swing_high: Price at the higher swing high (for bearish)
        rsi_swing_low: RSI value at the lower RSI low (for bullish)
        rsi_swing_high: RSI value at the higher RSI high (for bearish)
        swing_low_idx: Index of the price swing low
        swing_high_idx: Index of the price swing high
        confirmation: Whether divergence is confirmed by recent price action
    """

    type: Optional[Literal["bullish", "bearish"]] = None
    strength: float = 0.0
    price_swing_low: Optional[float] = None
    price_swing_high: Optional[float] = None
    rsi_swing_low: Optional[float] = None
    rsi_swing_high: Optional[float] = None
    swing_low_idx: Optional[int] = None
    swing_high_idx: Optional[int] = None
    confirmation: bool = False

    def is_valid(self, min_strength: float = 0.5) -> bool:
        """Check if divergence is valid and meets strength threshold.

        Args:
            min_strength: Minimum strength required (0.0 - 1.0)

        Returns:
            True if divergence exists and meets strength threshold
        """
        return self.type is not None and self.strength >= min_strength


def find_swing_points(
    series: pd.Series,
    order: int = 3,
    min_swing_pct: float = 0.001,
) -> tuple[list[int], list[int]]:
    """Find swing highs and lows in a price series.

    Uses a local extrema detection algorithm that looks for points
    where the series changes direction by examining 'order' bars
    on each side.

    Args:
        series: Price series to analyze
        order: Number of bars to look on each side for local extrema
        min_swing_pct: Minimum swing size as percentage (filters noise)

    Returns:
        Tuple of (swing_lows_indices, swing_highs_indices)
    """
    if len(series) < 2 * order + 1:
        return [], []

    swing_lows: list[int] = []
    swing_highs: list[int] = []

    # Calculate rolling min/max
    rolling_min = series.rolling(window=2 * order + 1, center=True).min()
    rolling_max = series.rolling(window=2 * order + 1, center=True).max()

    # Find local extrema
    for i in range(order, len(series) - order):
        current = series.iloc[i]

        # Check for swing low
        if current == rolling_min.iloc[i]:
            # Verify it's actually a local minimum
            window = series.iloc[i - order : i + order + 1]
            if current == window.min():
                # Check minimum swing size
                if i > 0:
                    prev_val = series.iloc[i - 1]
                    swing_size = abs(current - prev_val) / prev_val if prev_val != 0 else 0
                    if swing_size >= min_swing_pct:
                        swing_lows.append(i)

        # Check for swing high
        if current == rolling_max.iloc[i]:
            # Verify it's actually a local maximum
            window = series.iloc[i - order : i + order + 1]
            if current == window.max():
                # Check minimum swing size
                if i > 0:
                    prev_val = series.iloc[i - 1]
                    swing_size = abs(current - prev_val) / prev_val if prev_val != 0 else 0
                    if swing_size >= min_swing_pct:
                        swing_highs.append(i)

    return swing_lows, swing_highs


def detect_rsi_divergence(
    df: pd.DataFrame,
    lookback: int = 10,
    rsi_length: int = 14,
    min_swing_pct: float = 0.001,
    order: int = 2,
    require_confirmation: bool = True,
) -> DivergenceResult:
    """Detect RSI divergence patterns.

    Bullish Divergence (Potential Reversal Up):
    - Price makes lower low (LL)
    - RSI makes higher low (HL)
    - Indicates weakening selling pressure

    Bearish Divergence (Potential Reversal Down):
    - Price makes higher high (HH)
    - RSI makes lower high (LH)
    - Indicates weakening buying pressure

    Args:
        df: DataFrame with OHLCV data (must have 'close' column)
        lookback: Number of bars to look back for swing points
        rsi_length: RSI calculation period
        min_swing_pct: Minimum swing size as percentage
        order: Order for swing point detection
        require_confirmation: Whether to require price confirmation

    Returns:
        DivergenceResult with detection details

    Example:
        >>> df = pd.DataFrame({
        ...     'open': [100, 101, 99, 98, 97],
        ...     'high': [102, 103, 101, 100, 99],
        ...     'low': [99, 100, 98, 97, 96],
        ...     'close': [101, 99, 98, 97, 96],
        ...     'volume': [1000, 1100, 1050, 1200, 1150]
        ... })
        >>> result = detect_rsi_divergence(df, lookback=10, rsi_length=14)
        >>> if result.type == "bullish":
        ...     print(f"Bullish divergence detected with strength {result.strength}")
    """
    if df is None or len(df) < lookback + rsi_length:
        logger.debug(f"Insufficient data for divergence detection: {len(df) if df is not None else 0} bars")
        return DivergenceResult()

    if "close" not in df.columns:
        logger.error("DataFrame must have 'close' column")
        return DivergenceResult()

    # Calculate RSI
    try:
        rsi = ta.rsi(df["close"], length=rsi_length)
        if rsi is None or rsi.isna().all():
            logger.warning("RSI calculation failed or returned all NaN")
            return DivergenceResult()
    except Exception as e:
        logger.error(f"RSI calculation error: {e}")
        return DivergenceResult()

    # Get recent data within lookback window
    recent_df = df.iloc[-lookback:].copy()
    recent_rsi = rsi.iloc[-lookback:]

    if len(recent_df) < 5:  # Need at least 5 bars for meaningful divergence
        return DivergenceResult()

    # Find swing points in price and RSI
    price_lows, price_highs = find_swing_points(
        recent_df["close"], order=order, min_swing_pct=min_swing_pct
    )
    rsi_lows, rsi_highs = find_swing_points(
        recent_rsi, order=order, min_swing_pct=min_swing_pct / 2  # RSI swings can be smaller
    )

    result = DivergenceResult()

    # Check for bullish divergence (price LL, RSI HL)
    if len(price_lows) >= 2 and len(rsi_lows) >= 2:
        # Get the two most recent swing lows
        price_ll_idx = price_lows[-1]  # Most recent lower low
        price_ll_idx_prev = price_lows[-2]  # Previous low

        price_ll = recent_df["close"].iloc[price_ll_idx]
        price_ll_prev = recent_df["close"].iloc[price_ll_idx_prev]

        # Find corresponding RSI lows
        rsi_at_price_ll = recent_rsi.iloc[price_ll_idx]

        # Look for RSI higher low near the price low
        for rsi_low_idx in reversed(rsi_lows):
            if rsi_low_idx <= price_ll_idx + 2:  # Within 2 bars
                rsi_hl = recent_rsi.iloc[rsi_low_idx]

                # Check if price made lower low but RSI made higher low
                price_lower = price_ll < price_ll_prev * 0.999  # Allow small tolerance
                rsi_higher = rsi_hl > recent_rsi.iloc[max(0, rsi_low_idx - 3)] * 1.01

                if price_lower and rsi_higher:
                    # Calculate strength based on divergence magnitude
                    price_drop_pct = (price_ll_prev - price_ll) / price_ll_prev
                    rsi_rise = rsi_hl - recent_rsi.iloc[max(0, rsi_low_idx - 3)]

                    strength = min(1.0, (price_drop_pct * 100) * (rsi_rise / 10))
                    strength = max(0.3, min(1.0, strength))  # Clamp between 0.3 and 1.0

                    # Check for confirmation (recent price action)
                    confirmation = False
                    if require_confirmation and price_ll_idx < len(recent_df) - 1:
                        # Price should show some reversal signs
                        next_close = recent_df["close"].iloc[price_ll_idx + 1]
                        confirmation = next_close > price_ll * 1.001

                    result = DivergenceResult(
                        type="bullish",
                        strength=strength,
                        price_swing_low=float(price_ll),
                        price_swing_high=None,
                        rsi_swing_low=float(rsi_hl),
                        rsi_swing_high=None,
                        swing_low_idx=price_ll_idx,
                        swing_high_idx=None,
                        confirmation=confirmation if require_confirmation else True,
                    )
                    break

    # Check for bearish divergence (price HH, RSI LH) if no bullish found
    if result.type is None and len(price_highs) >= 2 and len(rsi_highs) >= 2:
        # Get the two most recent swing highs
        price_hh_idx = price_highs[-1]  # Most recent higher high
        price_hh_idx_prev = price_highs[-2]  # Previous high

        price_hh = recent_df["close"].iloc[price_hh_idx]
        price_hh_prev = recent_df["close"].iloc[price_hh_idx_prev]

        # Find corresponding RSI highs
        rsi_at_price_hh = recent_rsi.iloc[price_hh_idx]

        # Look for RSI lower high near the price high
        for rsi_high_idx in reversed(rsi_highs):
            if rsi_high_idx <= price_hh_idx + 2:  # Within 2 bars
                rsi_lh = recent_rsi.iloc[rsi_high_idx]

                # Check if price made higher high but RSI made lower high
                price_higher = price_hh > price_hh_prev * 1.001  # Allow small tolerance
                rsi_lower = rsi_lh < recent_rsi.iloc[max(0, rsi_high_idx - 3)] * 0.99

                if price_higher and rsi_lower:
                    # Calculate strength based on divergence magnitude
                    price_rise_pct = (price_hh - price_hh_prev) / price_hh_prev
                    rsi_drop = recent_rsi.iloc[max(0, rsi_high_idx - 3)] - rsi_lh

                    strength = min(1.0, (price_rise_pct * 100) * (rsi_drop / 10))
                    strength = max(0.3, min(1.0, strength))  # Clamp between 0.3 and 1.0

                    # Check for confirmation (recent price action)
                    confirmation = False
                    if require_confirmation and price_hh_idx < len(recent_df) - 1:
                        # Price should show some reversal signs
                        next_close = recent_df["close"].iloc[price_hh_idx + 1]
                        confirmation = next_close < price_hh * 0.999

                    result = DivergenceResult(
                        type="bearish",
                        strength=strength,
                        price_swing_low=None,
                        price_swing_high=float(price_hh),
                        rsi_swing_low=None,
                        rsi_swing_high=float(rsi_lh),
                        swing_low_idx=None,
                        swing_high_idx=price_hh_idx,
                        confirmation=confirmation if require_confirmation else True,
                    )
                    break

    return result


def detect_hidden_divergence(
    df: pd.DataFrame,
    lookback: int = 15,
    rsi_length: int = 14,
    min_swing_pct: float = 0.001,
    order: int = 2,
) -> DivergenceResult:
    """Detect hidden divergence patterns (trend continuation).

    Hidden Bullish Divergence (Trend Continuation Up):
    - Price makes higher low (HL)
    - RSI makes lower low (LL)
    - Indicates trend continuation after pullback

    Hidden Bearish Divergence (Trend Continuation Down):
    - Price makes lower high (LH)
    - RSI makes higher high (HH)
    - Indicates trend continuation after pullback

    Args:
        df: DataFrame with OHLCV data
        lookback: Number of bars to look back
        rsi_length: RSI calculation period
        min_swing_pct: Minimum swing size as percentage
        order: Order for swing point detection

    Returns:
        DivergenceResult with detection details
    """
    if df is None or len(df) < lookback + rsi_length:
        return DivergenceResult()

    if "close" not in df.columns:
        return DivergenceResult()

    # Calculate RSI
    try:
        rsi = ta.rsi(df["close"], length=rsi_length)
        if rsi is None or rsi.isna().all():
            return DivergenceResult()
    except Exception:
        return DivergenceResult()

    # Get recent data
    recent_df = df.iloc[-lookback:].copy()
    recent_rsi = rsi.iloc[-lookback:]

    if len(recent_df) < 5:
        return DivergenceResult()

    # Find swing points
    price_lows, price_highs = find_swing_points(
        recent_df["close"], order=order, min_swing_pct=min_swing_pct
    )
    rsi_lows, rsi_highs = find_swing_points(
        recent_rsi, order=order, min_swing_pct=min_swing_pct / 2
    )

    result = DivergenceResult()

    # Hidden Bullish: Price HL, RSI LL
    if len(price_lows) >= 2 and len(rsi_lows) >= 2:
        price_hl_idx = price_lows[-1]
        price_hl_idx_prev = price_lows[-2]

        price_hl = recent_df["close"].iloc[price_hl_idx]
        price_hl_prev = recent_df["close"].iloc[price_hl_idx_prev]

        for rsi_low_idx in reversed(rsi_lows):
            if rsi_low_idx <= price_hl_idx + 2:
                rsi_ll = recent_rsi.iloc[rsi_low_idx]
                rsi_ll_prev = recent_rsi.iloc[max(0, rsi_low_idx - 3)]

                price_higher_low = price_hl > price_hl_prev * 1.001
                rsi_lower_low = rsi_ll < rsi_ll_prev * 0.99

                if price_higher_low and rsi_lower_low:
                    strength = 0.5 + min(0.5, abs(rsi_ll - rsi_ll_prev) / 20)
                    result = DivergenceResult(
                        type="bullish",
                        strength=strength,
                        price_swing_low=float(price_hl),
                        rsi_swing_low=float(rsi_ll),
                        swing_low_idx=price_hl_idx,
                        confirmation=True,
                    )
                    break

    # Hidden Bearish: Price LH, RSI HH
    if result.type is None and len(price_highs) >= 2 and len(rsi_highs) >= 2:
        price_lh_idx = price_highs[-1]
        price_lh_idx_prev = price_highs[-2]

        price_lh = recent_df["close"].iloc[price_lh_idx]
        price_lh_prev = recent_df["close"].iloc[price_lh_idx_prev]

        for rsi_high_idx in reversed(rsi_highs):
            if rsi_high_idx <= price_lh_idx + 2:
                rsi_hh = recent_rsi.iloc[rsi_high_idx]
                rsi_hh_prev = recent_rsi.iloc[max(0, rsi_high_idx - 3)]

                price_lower_high = price_lh < price_lh_prev * 0.999
                rsi_higher_high = rsi_hh > rsi_hh_prev * 1.01

                if price_lower_high and rsi_higher_high:
                    strength = 0.5 + min(0.5, abs(rsi_hh - rsi_hh_prev) / 20)
                    result = DivergenceResult(
                        type="bearish",
                        strength=strength,
                        price_swing_high=float(price_lh),
                        rsi_swing_high=float(rsi_hh),
                        swing_high_idx=price_lh_idx,
                        confirmation=True,
                    )
                    break

    return result


def get_support_resistance_levels(
    df: pd.DataFrame,
    lookback: int = 20,
    zone_width_pct: float = 0.2,
) -> dict[str, list[float]]:
    """Calculate support and resistance levels from recent swing points.

    Args:
        df: DataFrame with OHLCV data
        lookback: Number of bars to analyze
        zone_width_pct: Width of support/resistance zone as percentage

    Returns:
        Dictionary with 'support' and 'resistance' levels
    """
    if df is None or len(df) < lookback:
        return {"support": [], "resistance": []}

    recent_df = df.iloc[-lookback:].copy()

    # Find swing points
    swing_lows, swing_highs = find_swing_points(recent_df["close"], order=2)

    support_levels = []
    resistance_levels = []

    # Calculate support from swing lows
    for idx in swing_lows:
        level = float(recent_df["close"].iloc[idx])
        # Check if this level is near other levels (cluster)
        is_cluster = False
        for existing in support_levels:
            if abs(level - existing) / existing < zone_width_pct / 100:
                is_cluster = True
                break
        if not is_cluster:
            support_levels.append(level)

    # Calculate resistance from swing highs
    for idx in swing_highs:
        level = float(recent_df["close"].iloc[idx])
        is_cluster = False
        for existing in resistance_levels:
            if abs(level - existing) / existing < zone_width_pct / 100:
                is_cluster = True
                break
        if not is_cluster:
            resistance_levels.append(level)

    # Sort levels
    support_levels.sort(reverse=True)  # Highest support first
    resistance_levels.sort()  # Lowest resistance first

    return {
        "support": support_levels[:3],  # Top 3 support levels
        "resistance": resistance_levels[:3],  # Top 3 resistance levels
    }


def calculate_proximity_to_level(
    price: float,
    levels: list[float],
    threshold_pct: float = 0.3,
) -> tuple[bool, float, Optional[float]]:
    """Calculate if price is near a support/resistance level.

    Args:
        price: Current price
        levels: List of support or resistance levels
        threshold_pct: Proximity threshold as percentage

    Returns:
        Tuple of (is_near, proximity_pct, nearest_level)
    """
    if not levels or price <= 0:
        return False, 0.0, None

    nearest_level = min(levels, key=lambda x: abs(x - price))
    distance_pct = abs(price - nearest_level) / price * 100

    is_near = distance_pct <= threshold_pct

    return is_near, distance_pct, nearest_level
