"""Candlestick pattern recognition module.

Detects various candlestick patterns on OHLCV DataFrames.
All prices are in PAISA (integer) - 1 Rupee = 100 paisa.

Patterns detected:
- Reversal patterns: Hammer, Inverted Hammer, Bullish/Bearish Engulfing,
  Morning Star, Evening Star, Piercing Line, Dark Cloud Cover
- Continuation patterns: Three White Soldiers, Three Black Crows,
  Rising Three Methods, Falling Three Methods
- Indecision: Doji
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

import pandas as pd

from src.research.models import PatternData
from src.utils.degradation import graceful_degrade

if TYPE_CHECKING:
    from pandas import DataFrame


class PatternType(str, Enum):
    """Type of pattern detected."""

    REVERSAL = "reversal"
    CONTINUATION = "continuation"
    INDECISION = "indecision"


class PatternName(str, Enum):
    """Available candlestick patterns."""

    HAMMER = "hammer"
    INVERTED_HAMMER = "inverted_hammer"
    BULLISH_ENGULFING = "bullish_engulfing"
    BEARISH_ENGULFING = "bearish_engulfing"
    MORNING_STAR = "morning_star"
    EVENING_STAR = "evening_star"
    DOJI = "doji"
    PIERCING_LINE = "piercing_line"
    DARK_CLOUD_COVER = "dark_cloud_cover"
    THREE_WHITE_SOLDIERS = "three_white_soldiers"
    THREE_BLACK_CROWS = "three_black_crows"
    RISING_THREE_METHODS = "rising_three_methods"
    FALLING_THREE_METHODS = "falling_three_methods"


@dataclass
class PatternResult:
    """Result of pattern detection."""

    detected: bool
    pattern_name: str | None
    pattern_type: PatternType | None
    strength: float  # 0-100
    score: int  # Scoring value for research engine
    description: str


class PatternRecognizer:
    """Recognizes candlestick patterns in OHLCV data.

    All calculations work with prices in PAISA (integer) to avoid
    floating-point precision issues.
    """

    # Pattern scoring values
    SCORE_CONFIRMING_BULLISH = 20
    SCORE_CONFIRMING_BEARISH = 20
    SCORE_CONTRADICTING_BULLISH = -15
    SCORE_CONTRADICTING_BEARISH = -15

    # Pattern strength thresholds
    MIN_BODY_RATIO = 0.1  # Minimum body/range ratio for significant candles
    MAX_DOJI_RATIO = 0.05  # Maximum body/range ratio for doji

    def __init__(self) -> None:
        """Initialize the pattern recognizer."""
        self._pattern_methods = {
            PatternName.HAMMER: self._detect_hammer,
            PatternName.INVERTED_HAMMER: self._detect_inverted_hammer,
            PatternName.BULLISH_ENGULFING: self._detect_bullish_engulfing,
            PatternName.BEARISH_ENGULFING: self._detect_bearish_engulfing,
            PatternName.MORNING_STAR: self._detect_morning_star,
            PatternName.EVENING_STAR: self._detect_evening_star,
            PatternName.DOJI: self._detect_doji,
            PatternName.PIERCING_LINE: self._detect_piercing_line,
            PatternName.DARK_CLOUD_COVER: self._detect_dark_cloud_cover,
            PatternName.THREE_WHITE_SOLDIERS: self._detect_three_white_soldiers,
            PatternName.THREE_BLACK_CROWS: self._detect_three_black_crows,
            PatternName.RISING_THREE_METHODS: self._detect_rising_three_methods,
            PatternName.FALLING_THREE_METHODS: self._detect_falling_three_methods,
        }

    @graceful_degrade(
        default_return={
            "detected": False,
            "strength": 0.0,
            "type": None,
            "score": 0,
            "description": "Pattern detection failed - using default values",
        }
    )
    def detect_pattern(
        self,
        df: DataFrame,
        pattern_name: PatternName | str,
    ) -> dict:
        """Detect a specific pattern in the DataFrame.

        Args:
            df: OHLCV DataFrame with columns ['open', 'high', 'low', 'close', 'volume']
            pattern_name: Name of the pattern to detect

        Returns:
            Dictionary with detection results:
            - detected: bool
            - strength: float (0-100)
            - type: str (reversal/continuation/indecision)
            - score: int (for research engine scoring)
            - description: str
        """
        if isinstance(pattern_name, str):
            pattern_name = PatternName(pattern_name.lower())

        if pattern_name not in self._pattern_methods:
            return {
                "detected": False,
                "strength": 0.0,
                "type": None,
                "score": 0,
                "description": f"Unknown pattern: {pattern_name}",
            }

        result = self._pattern_methods[pattern_name](df)

        return {
            "detected": result.detected,
            "strength": result.strength,
            "type": result.pattern_type.value if result.pattern_type else None,
            "score": result.score,
            "description": result.description,
        }

    @graceful_degrade(
        default_return=PatternData(
            pattern_detected=None,
            pattern_type=None,
            pattern_strength=0,
            pattern_timeframe="5m",
            confirms_signal=False,
            contradicts_signal=False,
        )
    )
    def analyze(
        self,
        df: DataFrame,
        signal_direction: str,  # "BUY" or "SELL"
        timeframe: str = "5m",
    ) -> PatternData:
        """Analyze DataFrame for all patterns and return PatternData.

        Args:
            df: OHLCV DataFrame
            signal_direction: "BUY" or "SELL" from the originating signal
            timeframe: Candle timeframe (e.g., "5m", "15m", "1h")

        Returns:
            PatternData with detected pattern and signal alignment
        """
        if df.empty or len(df) < 3:
            return PatternData(
                pattern_detected=None,
                pattern_type=None,
                pattern_strength=0,
                pattern_timeframe=timeframe,
                confirms_signal=False,
                contradicts_signal=False,
            )

        # Check all patterns and find the strongest one
        best_pattern: PatternResult | None = None
        best_strength = 0.0

        for pattern_name in PatternName:
            result = self._pattern_methods[pattern_name](df)
            if result.detected and result.strength > best_strength:
                best_pattern = result
                best_strength = result.strength

        if best_pattern is None:
            return PatternData(
                pattern_detected=None,
                pattern_type=None,
                pattern_strength=0,
                pattern_timeframe=timeframe,
                confirms_signal=False,
                contradicts_signal=False,
            )

        # Determine if pattern confirms or contradicts signal
        confirms = False
        contradicts = False

        is_bullish_pattern = best_pattern.pattern_name in {
            PatternName.HAMMER.value,
            PatternName.INVERTED_HAMMER.value,
            PatternName.BULLISH_ENGULFING.value,
            PatternName.MORNING_STAR.value,
            PatternName.PIERCING_LINE.value,
            PatternName.THREE_WHITE_SOLDIERS.value,
            PatternName.RISING_THREE_METHODS.value,
        }

        is_bearish_pattern = best_pattern.pattern_name in {
            PatternName.BEARISH_ENGULFING.value,
            PatternName.EVENING_STAR.value,
            PatternName.DARK_CLOUD_COVER.value,
            PatternName.THREE_BLACK_CROWS.value,
            PatternName.FALLING_THREE_METHODS.value,
        }

        if signal_direction.upper() == "BUY":
            if is_bullish_pattern:
                confirms = True
            elif is_bearish_pattern:
                contradicts = True
        elif signal_direction.upper() == "SELL":
            if is_bearish_pattern:
                confirms = True
            elif is_bullish_pattern:
                contradicts = True

        return PatternData(
            pattern_detected=best_pattern.pattern_name,
            pattern_type=best_pattern.pattern_type.value if best_pattern.pattern_type else None,
            pattern_strength=best_pattern.strength,
            pattern_timeframe=timeframe,
            confirms_signal=confirms,
            contradicts_signal=contradicts,
        )

    def _get_candle_metrics(self, df: DataFrame, idx: int = -1) -> dict:
        """Calculate candle metrics for pattern detection.

        Args:
            df: OHLCV DataFrame
            idx: Index of candle to analyze (default: last candle)

        Returns:
            Dictionary with candle metrics
        """
        if idx < 0:
            idx = len(df) + idx

        if idx < 0 or idx >= len(df):
            return {}

        open_p = int(df["open"].iloc[idx])
        high_p = int(df["high"].iloc[idx])
        low_p = int(df["low"].iloc[idx])
        close_p = int(df["close"].iloc[idx])

        body = abs(close_p - open_p)
        range_total = high_p - low_p
        upper_shadow = high_p - max(open_p, close_p)
        lower_shadow = min(open_p, close_p) - low_p

        is_bullish = close_p > open_p
        is_bearish = close_p < open_p

        # Avoid division by zero
        body_ratio = body / range_total if range_total > 0 else 0
        upper_ratio = upper_shadow / range_total if range_total > 0 else 0
        lower_ratio = lower_shadow / range_total if range_total > 0 else 0

        return {
            "open": open_p,
            "high": high_p,
            "low": low_p,
            "close": close_p,
            "body": body,
            "range": range_total,
            "upper_shadow": upper_shadow,
            "lower_shadow": lower_shadow,
            "body_ratio": body_ratio,
            "upper_ratio": upper_ratio,
            "lower_ratio": lower_ratio,
            "is_bullish": is_bullish,
            "is_bearish": is_bearish,
        }

    def _detect_hammer(self, df: DataFrame) -> PatternResult:
        """Detect Hammer pattern (bullish reversal).

        Characteristics:
        - Small body at the upper end of the trading range
        - Long lower shadow (at least 2x the body)
        - Little to no upper shadow
        - Appears in a downtrend
        """
        if len(df) < 2:
            return PatternResult(False, None, None, 0, 0, "Insufficient data")

        curr = self._get_candle_metrics(df, -1)
        prev = self._get_candle_metrics(df, -2)

        if not curr or not prev:
            return PatternResult(False, None, None, 0, 0, "Invalid data")

        # Hammer criteria
        has_small_body = curr["body_ratio"] < 0.3
        has_long_lower = curr["lower_ratio"] > 0.6
        has_small_upper = curr["upper_ratio"] < 0.1
        is_downtrend = prev["close"] > curr["close"]  # Price declining

        detected = has_small_body and has_long_lower and has_small_upper and is_downtrend

        if not detected:
            return PatternResult(False, None, None, 0, 0, "Hammer not detected")

        # Calculate strength based on lower shadow length
        strength = min(100, curr["lower_ratio"] * 100 + 20)

        return PatternResult(
            detected=True,
            pattern_name=PatternName.HAMMER.value,
            pattern_type=PatternType.REVERSAL,
            strength=strength,
            score=self.SCORE_CONFIRMING_BULLISH if detected else 0,
            description=f"Hammer detected at bottom with {strength:.1f}% strength",
        )

    def _detect_inverted_hammer(self, df: DataFrame) -> PatternResult:
        """Detect Inverted Hammer pattern (bullish reversal).

        Characteristics:
        - Small body at the lower end of the trading range
        - Long upper shadow (at least 2x the body)
        - Little to no lower shadow
        - Appears in a downtrend
        """
        if len(df) < 2:
            return PatternResult(False, None, None, 0, 0, "Insufficient data")

        curr = self._get_candle_metrics(df, -1)
        prev = self._get_candle_metrics(df, -2)

        if not curr or not prev:
            return PatternResult(False, None, None, 0, 0, "Invalid data")

        # Inverted hammer criteria
        has_small_body = curr["body_ratio"] < 0.3
        has_long_upper = curr["upper_ratio"] > 0.6
        has_small_lower = curr["lower_ratio"] < 0.1
        is_downtrend = prev["close"] > curr["close"]

        detected = has_small_body and has_long_upper and has_small_lower and is_downtrend

        if not detected:
            return PatternResult(False, None, None, 0, 0, "Inverted hammer not detected")

        strength = min(100, curr["upper_ratio"] * 100 + 20)

        return PatternResult(
            detected=True,
            pattern_name=PatternName.INVERTED_HAMMER.value,
            pattern_type=PatternType.REVERSAL,
            strength=strength,
            score=self.SCORE_CONFIRMING_BULLISH if detected else 0,
            description=f"Inverted hammer detected at bottom with {strength:.1f}% strength",
        )

    def _detect_bullish_engulfing(self, df: DataFrame) -> PatternResult:
        """Detect Bullish Engulfing pattern (bullish reversal).

        Characteristics:
        - First candle is bearish (close < open)
        - Second candle is bullish (close > open)
        - Second candle's body completely engulfs first candle's body
        - Appears in a downtrend
        """
        if len(df) < 2:
            return PatternResult(False, None, None, 0, 0, "Insufficient data")

        curr = self._get_candle_metrics(df, -1)
        prev = self._get_candle_metrics(df, -2)

        if not curr or not prev:
            return PatternResult(False, None, None, 0, 0, "Invalid data")

        # Bullish engulfing criteria
        prev_bearish = prev["is_bearish"]
        curr_bullish = curr["is_bullish"]
        engulfs_body = (curr["close"] > prev["open"]) and (curr["open"] < prev["close"])
        is_downtrend = prev["close"] < prev["open"]  # Previous was bearish

        detected = prev_bearish and curr_bullish and engulfs_body and is_downtrend

        if not detected:
            return PatternResult(False, None, None, 0, 0, "Bullish engulfing not detected")

        # Strength based on engulfing ratio
        engulf_ratio = curr["body"] / max(prev["body"], 1)
        strength = min(100, 50 + engulf_ratio * 25)

        return PatternResult(
            detected=True,
            pattern_name=PatternName.BULLISH_ENGULFING.value,
            pattern_type=PatternType.REVERSAL,
            strength=strength,
            score=self.SCORE_CONFIRMING_BULLISH if detected else 0,
            description=f"Bullish engulfing detected with {strength:.1f}% strength",
        )

    def _detect_bearish_engulfing(self, df: DataFrame) -> PatternResult:
        """Detect Bearish Engulfing pattern (bearish reversal).

        Characteristics:
        - First candle is bullish (close > open)
        - Second candle is bearish (close < open)
        - Second candle's body completely engulfs first candle's body
        - Appears in an uptrend
        """
        if len(df) < 2:
            return PatternResult(False, None, None, 0, 0, "Insufficient data")

        curr = self._get_candle_metrics(df, -1)
        prev = self._get_candle_metrics(df, -2)

        if not curr or not prev:
            return PatternResult(False, None, None, 0, 0, "Invalid data")

        # Bearish engulfing criteria
        prev_bullish = prev["is_bullish"]
        curr_bearish = curr["is_bearish"]
        engulfs_body = (curr["close"] < prev["open"]) and (curr["open"] > prev["close"])
        is_uptrend = prev["close"] > prev["open"]

        detected = prev_bullish and curr_bearish and engulfs_body and is_uptrend

        if not detected:
            return PatternResult(False, None, None, 0, 0, "Bearish engulfing not detected")

        engulf_ratio = curr["body"] / max(prev["body"], 1)
        strength = min(100, 50 + engulf_ratio * 25)

        return PatternResult(
            detected=True,
            pattern_name=PatternName.BEARISH_ENGULFING.value,
            pattern_type=PatternType.REVERSAL,
            strength=strength,
            score=self.SCORE_CONFIRMING_BEARISH if detected else 0,
            description=f"Bearish engulfing detected with {strength:.1f}% strength",
        )

    def _detect_morning_star(self, df: DataFrame) -> PatternResult:
        """Detect Morning Star pattern (bullish reversal).

        Characteristics:
        - First candle: Large bearish
        - Second candle: Small body (doji or spinning top), gaps down
        - Third candle: Large bullish, closes well into first candle's body
        """
        if len(df) < 3:
            return PatternResult(False, None, None, 0, 0, "Insufficient data")

        c1 = self._get_candle_metrics(df, -3)  # First candle
        c2 = self._get_candle_metrics(df, -2)  # Star
        c3 = self._get_candle_metrics(df, -1)  # Confirmation

        if not all([c1, c2, c3]):
            return PatternResult(False, None, None, 0, 0, "Invalid data")

        # Morning star criteria
        first_bearish = c1["is_bearish"] and c1["body_ratio"] > 0.5
        star_small = c2["body_ratio"] < 0.3
        star_gaps_down = c2["high"] < c1["low"]
        third_bullish = c3["is_bullish"] and c3["body_ratio"] > 0.5
        closes_into_first = c3["close"] > (c1["open"] + c1["close"]) / 2

        detected = first_bearish and star_small and third_bullish and closes_into_first

        if not detected:
            return PatternResult(False, None, None, 0, 0, "Morning star not detected")

        # Strength based on how far third candle penetrates first
        penetration = (c3["close"] - c1["close"]) / max(c1["body"], 1)
        strength = min(100, 60 + penetration * 30)

        return PatternResult(
            detected=True,
            pattern_name=PatternName.MORNING_STAR.value,
            pattern_type=PatternType.REVERSAL,
            strength=strength,
            score=self.SCORE_CONFIRMING_BULLISH if detected else 0,
            description=f"Morning star detected with {strength:.1f}% strength",
        )

    def _detect_evening_star(self, df: DataFrame) -> PatternResult:
        """Detect Evening Star pattern (bearish reversal).

        Characteristics:
        - First candle: Large bullish
        - Second candle: Small body, gaps up
        - Third candle: Large bearish, closes well into first candle's body
        """
        if len(df) < 3:
            return PatternResult(False, None, None, 0, 0, "Insufficient data")

        c1 = self._get_candle_metrics(df, -3)
        c2 = self._get_candle_metrics(df, -2)
        c3 = self._get_candle_metrics(df, -1)

        if not all([c1, c2, c3]):
            return PatternResult(False, None, None, 0, 0, "Invalid data")

        # Evening star criteria
        first_bullish = c1["is_bullish"] and c1["body_ratio"] > 0.5
        star_small = c2["body_ratio"] < 0.3
        star_gaps_up = c2["low"] > c1["high"]
        third_bearish = c3["is_bearish"] and c3["body_ratio"] > 0.5
        closes_into_first = c3["close"] < (c1["open"] + c1["close"]) / 2

        detected = first_bullish and star_small and third_bearish and closes_into_first

        if not detected:
            return PatternResult(False, None, None, 0, 0, "Evening star not detected")

        penetration = (c1["close"] - c3["close"]) / max(c1["body"], 1)
        strength = min(100, 60 + penetration * 30)

        return PatternResult(
            detected=True,
            pattern_name=PatternName.EVENING_STAR.value,
            pattern_type=PatternType.REVERSAL,
            strength=strength,
            score=self.SCORE_CONFIRMING_BEARISH if detected else 0,
            description=f"Evening star detected with {strength:.1f}% strength",
        )

    def _detect_doji(self, df: DataFrame) -> PatternResult:
        """Detect Doji pattern (indecision).

        Characteristics:
        - Open and close are virtually equal (very small body)
        - Shadows can be long or short
        - Indicates market indecision
        """
        if len(df) < 1:
            return PatternResult(False, None, None, 0, 0, "Insufficient data")

        curr = self._get_candle_metrics(df, -1)

        if not curr:
            return PatternResult(False, None, None, 0, 0, "Invalid data")

        # Doji criteria
        detected = curr["body_ratio"] < self.MAX_DOJI_RATIO

        if not detected:
            return PatternResult(False, None, None, 0, 0, "Doji not detected")

        # Strength based on how perfect the doji is
        strength = max(0, 100 - curr["body_ratio"] * 1000)

        return PatternResult(
            detected=True,
            pattern_name=PatternName.DOJI.value,
            pattern_type=PatternType.INDECISION,
            strength=strength,
            score=0,  # Neutral score for indecision
            description=f"Doji detected indicating market indecision ({strength:.1f}% purity)",
        )

    def _detect_piercing_line(self, df: DataFrame) -> PatternResult:
        """Detect Piercing Line pattern (bullish reversal).

        Characteristics:
        - First candle: Bearish
        - Second candle: Bullish, opens below previous low
        - Second candle closes above midpoint of first candle's body
        """
        if len(df) < 2:
            return PatternResult(False, None, None, 0, 0, "Insufficient data")

        prev = self._get_candle_metrics(df, -2)
        curr = self._get_candle_metrics(df, -1)

        if not prev or not curr:
            return PatternResult(False, None, None, 0, 0, "Invalid data")

        # Piercing line criteria
        prev_bearish = prev["is_bearish"]
        curr_bullish = curr["is_bullish"]
        opens_below = curr["open"] < prev["low"]
        midpoint = (prev["open"] + prev["close"]) / 2
        closes_above_mid = curr["close"] > midpoint

        detected = prev_bearish and curr_bullish and opens_below and closes_above_mid

        if not detected:
            return PatternResult(False, None, None, 0, 0, "Piercing line not detected")

        # Strength based on penetration depth
        penetration = (curr["close"] - midpoint) / max(prev["body"] / 2, 1)
        strength = min(100, 50 + penetration * 40)

        return PatternResult(
            detected=True,
            pattern_name=PatternName.PIERCING_LINE.value,
            pattern_type=PatternType.REVERSAL,
            strength=strength,
            score=self.SCORE_CONFIRMING_BULLISH if detected else 0,
            description=f"Piercing line detected with {strength:.1f}% strength",
        )

    def _detect_dark_cloud_cover(self, df: DataFrame) -> PatternResult:
        """Detect Dark Cloud Cover pattern (bearish reversal).

        Characteristics:
        - First candle: Bullish
        - Second candle: Bearish, opens above previous high
        - Second candle closes below midpoint of first candle's body
        """
        if len(df) < 2:
            return PatternResult(False, None, None, 0, 0, "Insufficient data")

        prev = self._get_candle_metrics(df, -2)
        curr = self._get_candle_metrics(df, -1)

        if not prev or not curr:
            return PatternResult(False, None, None, 0, 0, "Invalid data")

        # Dark cloud cover criteria
        prev_bullish = prev["is_bullish"]
        curr_bearish = curr["is_bearish"]
        opens_above = curr["open"] > prev["high"]
        midpoint = (prev["open"] + prev["close"]) / 2
        closes_below_mid = curr["close"] < midpoint

        detected = prev_bullish and curr_bearish and opens_above and closes_below_mid

        if not detected:
            return PatternResult(False, None, None, 0, 0, "Dark cloud cover not detected")

        penetration = (midpoint - curr["close"]) / max(prev["body"] / 2, 1)
        strength = min(100, 50 + penetration * 40)

        return PatternResult(
            detected=True,
            pattern_name=PatternName.DARK_CLOUD_COVER.value,
            pattern_type=PatternType.REVERSAL,
            strength=strength,
            score=self.SCORE_CONFIRMING_BEARISH if detected else 0,
            description=f"Dark cloud cover detected with {strength:.1f}% strength",
        )

    def _detect_three_white_soldiers(self, df: DataFrame) -> PatternResult:
        """Detect Three White Soldiers pattern (bullish continuation).

        Characteristics:
        - Three consecutive bullish candles
        - Each opens within previous candle's body
        - Each closes higher than previous
        - Relatively small upper shadows
        """
        if len(df) < 3:
            return PatternResult(False, None, None, 0, 0, "Insufficient data")

        c1 = self._get_candle_metrics(df, -3)
        c2 = self._get_candle_metrics(df, -2)
        c3 = self._get_candle_metrics(df, -1)

        if not all([c1, c2, c3]):
            return PatternResult(False, None, None, 0, 0, "Invalid data")

        # Three white soldiers criteria
        all_bullish = c1["is_bullish"] and c2["is_bullish"] and c3["is_bullish"]
        c2_opens_in_c1 = c1["open"] < c2["open"] < c1["close"]
        c3_opens_in_c2 = c2["open"] < c3["open"] < c2["close"]
        higher_closes = c1["close"] < c2["close"] < c3["close"]
        small_upper_shadows = (
            c1["upper_ratio"] < 0.2
            and c2["upper_ratio"] < 0.2
            and c3["upper_ratio"] < 0.2
        )

        detected = all_bullish and c2_opens_in_c1 and c3_opens_in_c2 and higher_closes and small_upper_shadows

        if not detected:
            return PatternResult(False, None, None, 0, 0, "Three white soldiers not detected")

        # Strength based on consistency
        body_consistency = 1 - abs(c1["body"] - c3["body"]) / max(c1["body"], 1)
        strength = min(100, 70 + body_consistency * 30)

        return PatternResult(
            detected=True,
            pattern_name=PatternName.THREE_WHITE_SOLDIERS.value,
            pattern_type=PatternType.CONTINUATION,
            strength=strength,
            score=self.SCORE_CONFIRMING_BULLISH if detected else 0,
            description=f"Three white soldiers detected with {strength:.1f}% strength",
        )

    def _detect_three_black_crows(self, df: DataFrame) -> PatternResult:
        """Detect Three Black Crows pattern (bearish continuation).

        Characteristics:
        - Three consecutive bearish candles
        - Each opens within previous candle's body
        - Each closes lower than previous
        - Relatively small lower shadows
        """
        if len(df) < 3:
            return PatternResult(False, None, None, 0, 0, "Insufficient data")

        c1 = self._get_candle_metrics(df, -3)
        c2 = self._get_candle_metrics(df, -2)
        c3 = self._get_candle_metrics(df, -1)

        if not all([c1, c2, c3]):
            return PatternResult(False, None, None, 0, 0, "Invalid data")

        # Three black crows criteria
        all_bearish = c1["is_bearish"] and c2["is_bearish"] and c3["is_bearish"]
        c2_opens_in_c1 = c1["close"] < c2["open"] < c1["open"]
        c3_opens_in_c2 = c2["close"] < c3["open"] < c2["open"]
        lower_closes = c1["close"] > c2["close"] > c3["close"]
        small_lower_shadows = (
            c1["lower_ratio"] < 0.2
            and c2["lower_ratio"] < 0.2
            and c3["lower_ratio"] < 0.2
        )

        detected = all_bearish and c2_opens_in_c1 and c3_opens_in_c2 and lower_closes and small_lower_shadows

        if not detected:
            return PatternResult(False, None, None, 0, 0, "Three black crows not detected")

        body_consistency = 1 - abs(c1["body"] - c3["body"]) / max(c1["body"], 1)
        strength = min(100, 70 + body_consistency * 30)

        return PatternResult(
            detected=True,
            pattern_name=PatternName.THREE_BLACK_CROWS.value,
            pattern_type=PatternType.CONTINUATION,
            strength=strength,
            score=self.SCORE_CONFIRMING_BEARISH if detected else 0,
            description=f"Three black crows detected with {strength:.1f}% strength",
        )

    def _detect_rising_three_methods(self, df: DataFrame) -> PatternResult:
        """Detect Rising Three Methods pattern (bullish continuation).

        Characteristics:
        - First candle: Long bullish
        - Middle 3 candles: Small bearish candles within first candle's range
        - Last candle: Long bullish, closes above first candle's close
        """
        if len(df) < 5:
            return PatternResult(False, None, None, 0, 0, "Insufficient data")

        c1 = self._get_candle_metrics(df, -5)
        c2 = self._get_candle_metrics(df, -4)
        c3 = self._get_candle_metrics(df, -3)
        c4 = self._get_candle_metrics(df, -2)
        c5 = self._get_candle_metrics(df, -1)

        if not all([c1, c2, c3, c4, c5]):
            return PatternResult(False, None, None, 0, 0, "Invalid data")

        # Rising three methods criteria
        first_bullish = c1["is_bullish"] and c1["body_ratio"] > 0.6
        last_bullish = c5["is_bullish"] and c5["close"] > c1["close"]

        # Middle candles should be within first candle's range
        middle_in_range = (
            c1["low"] < c2["low"]
            and c2["high"] < c1["high"]
            and c1["low"] < c3["low"]
            and c3["high"] < c1["high"]
            and c1["low"] < c4["low"]
            and c4["high"] < c1["high"]
        )

        # Middle candles are bearish or small
        middle_bearish = (
            (c2["is_bearish"] or c2["body_ratio"] < 0.3)
            and (c3["is_bearish"] or c3["body_ratio"] < 0.3)
            and (c4["is_bearish"] or c4["body_ratio"] < 0.3)
        )

        detected = first_bullish and last_bullish and middle_in_range and middle_bearish

        if not detected:
            return PatternResult(False, None, None, 0, 0, "Rising three methods not detected")

        strength = min(100, 65 + (c5["close"] - c1["close"]) / max(c1["body"], 1) * 20)

        return PatternResult(
            detected=True,
            pattern_name=PatternName.RISING_THREE_METHODS.value,
            pattern_type=PatternType.CONTINUATION,
            strength=strength,
            score=self.SCORE_CONFIRMING_BULLISH if detected else 0,
            description=f"Rising three methods detected with {strength:.1f}% strength",
        )

    def _detect_falling_three_methods(self, df: DataFrame) -> PatternResult:
        """Detect Falling Three Methods pattern (bearish continuation).

        Characteristics:
        - First candle: Long bearish
        - Middle 3 candles: Small bullish candles within first candle's range
        - Last candle: Long bearish, closes below first candle's close
        """
        if len(df) < 5:
            return PatternResult(False, None, None, 0, 0, "Insufficient data")

        c1 = self._get_candle_metrics(df, -5)
        c2 = self._get_candle_metrics(df, -4)
        c3 = self._get_candle_metrics(df, -3)
        c4 = self._get_candle_metrics(df, -2)
        c5 = self._get_candle_metrics(df, -1)

        if not all([c1, c2, c3, c4, c5]):
            return PatternResult(False, None, None, 0, 0, "Invalid data")

        # Falling three methods criteria
        first_bearish = c1["is_bearish"] and c1["body_ratio"] > 0.6
        last_bearish = c5["is_bearish"] and c5["close"] < c1["close"]

        # Middle candles should be within first candle's range
        middle_in_range = (
            c1["low"] < c2["low"]
            and c2["high"] < c1["high"]
            and c1["low"] < c3["low"]
            and c3["high"] < c1["high"]
            and c1["low"] < c4["low"]
            and c4["high"] < c1["high"]
        )

        # Middle candles are bullish or small
        middle_bullish = (
            (c2["is_bullish"] or c2["body_ratio"] < 0.3)
            and (c3["is_bullish"] or c3["body_ratio"] < 0.3)
            and (c4["is_bullish"] or c4["body_ratio"] < 0.3)
        )

        detected = first_bearish and last_bearish and middle_in_range and middle_bullish

        if not detected:
            return PatternResult(False, None, None, 0, 0, "Falling three methods not detected")

        strength = min(100, 65 + (c1["close"] - c5["close"]) / max(c1["body"], 1) * 20)

        return PatternResult(
            detected=True,
            pattern_name=PatternName.FALLING_THREE_METHODS.value,
            pattern_type=PatternType.CONTINUATION,
            strength=strength,
            score=self.SCORE_CONFIRMING_BEARISH if detected else 0,
            description=f"Falling three methods detected with {strength:.1f}% strength",
        )

    def get_all_patterns(self, df: DataFrame) -> list[dict]:
        """Detect all patterns in the DataFrame.

        Args:
            df: OHLCV DataFrame

        Returns:
            List of detected pattern dictionaries
        """
        patterns = []

        for pattern_name in PatternName:
            result = self.detect_pattern(df, pattern_name)
            if result["detected"]:
                patterns.append({
                    "pattern": pattern_name.value,
                    **result,
                })

        # Sort by strength descending
        patterns.sort(key=lambda x: x["strength"], reverse=True)

        return patterns
