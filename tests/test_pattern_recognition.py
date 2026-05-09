"""Tests for the pattern recognition module.

All prices are in PAISA (integer) - 1 Rupee = 100 paisa.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.research.pattern_recognition import (
    PatternName,
    PatternRecognizer,
    PatternResult,
    PatternType,
)


@pytest.fixture
def pattern_recognizer() -> PatternRecognizer:
    """Return a PatternRecognizer instance."""
    return PatternRecognizer()


@pytest.fixture
def hammer_data() -> pd.DataFrame:
    """Return OHLCV data with a hammer pattern at the end.

    Hammer characteristics:
    - Small body at upper end
    - Long lower shadow (at least 2x body)
    - Little to no upper shadow
    - Appears in downtrend
    """
    # Create 5 candles, last one is a hammer in a downtrend
    data = {
        "open": [50000, 49900, 49800, 49700, 49600],  # Downtrend
        "high": [50100, 50000, 49900, 49800, 49700],
        "low": [49900, 49800, 49700, 49600, 49200],   # Long lower wick on last candle
        "close": [49950, 49850, 49750, 49650, 49650],  # Small body, near high
        "volume": [1000, 1000, 1000, 1000, 1500],
    }
    return pd.DataFrame(data)


@pytest.fixture
def inverted_hammer_data() -> pd.DataFrame:
    """Return OHLCV data with an inverted hammer pattern at the end."""
    data = {
        "open": [50000, 49900, 49800, 49700, 49600],  # Downtrend
        "high": [50100, 50000, 49900, 49800, 50100],  # Long upper wick
        "low": [49900, 49800, 49700, 49600, 49550],   # Small lower wick
        "close": [49950, 49850, 49750, 49650, 49650],  # Small body, near low
        "volume": [1000, 1000, 1000, 1000, 1500],
    }
    return pd.DataFrame(data)


@pytest.fixture
def bullish_engulfing_data() -> pd.DataFrame:
    """Return OHLCV data with a bullish engulfing pattern.

    - First candle bearish (close < open)
    - Second candle bullish (close > open)
    - Second engulfs first's body
    """
    data = {
        "open": [50000, 49900, 49800, 49900, 49700],  # Last candle bullish
        "high": [50100, 50000, 49900, 49950, 50000],
        "low": [49850, 49750, 49650, 49800, 49650],
        "close": [49900, 49800, 49700, 49850, 49950],  # Bullish, engulfs prev
        "volume": [1000, 1000, 1000, 1000, 2000],
    }
    return pd.DataFrame(data)


@pytest.fixture
def bearish_engulfing_data() -> pd.DataFrame:
    """Return OHLCV data with a bearish engulfing pattern."""
    data = {
        "open": [50000, 50100, 50200, 50300, 50400],  # Uptrend
        "high": [50100, 50200, 50300, 50400, 50450],
        "low": [49900, 50000, 50100, 50200, 50250],
        "close": [50050, 50150, 50250, 50350, 50250],  # Bearish, engulfs prev
        "volume": [1000, 1000, 1000, 1000, 2000],
    }
    return pd.DataFrame(data)


@pytest.fixture
def morning_star_data() -> pd.DataFrame:
    """Return OHLCV data with a morning star pattern (3 candles).

    Criteria met:
    - C1: Large bearish (body_ratio > 0.5)
    - C2: Small body (body_ratio < 0.3) — the star
    - C3: Large bullish (body_ratio > 0.5), close > midpoint of C1
    """
    data = {
        # c1: large bearish — body=1000, range=1200, ratio~0.83
        # c2: star — body=50, range=200, ratio=0.25
        # c3: large bullish — body=700, range=800, ratio=0.875; close 50200 > midpoint 50000
        "open":  [50500, 49400, 49500],
        "high":  [50600, 49500, 50250],
        "low":   [49400, 49300, 49450],
        "close": [49500, 49450, 50200],
        "volume": [2000, 800, 2000],
    }
    return pd.DataFrame(data)


@pytest.fixture
def evening_star_data() -> pd.DataFrame:
    """Return OHLCV data with an evening star pattern (3 candles).

    Criteria met:
    - C1: Large bullish (body_ratio > 0.5)
    - C2: Small body (body_ratio < 0.3) — the star
    - C3: Large bearish (body_ratio > 0.5), close < midpoint of C1
    """
    data = {
        # c1: large bullish — body=1000, range=1200, ratio~0.83
        # c2: star — body=50, range=200, ratio=0.25
        # c3: large bearish — body=700, range=800, ratio=0.875; close 49900 < midpoint 50000
        "open":  [49500, 50600, 50600],
        "high":  [50600, 50700, 50650],
        "low":   [49400, 50500, 49850],
        "close": [50500, 50650, 49900],
        "volume": [2000, 800, 2000],
    }
    return pd.DataFrame(data)


@pytest.fixture
def doji_data() -> pd.DataFrame:
    """Return OHLCV data with a doji pattern."""
    data = {
        "open": [50000, 50100, 50050],
        "high": [50200, 50300, 50200],
        "low": [49800, 49900, 49900],
        "close": [50100, 50200, 50055],  # Very close to open (doji)
        "volume": [1000, 1000, 1500],
    }
    return pd.DataFrame(data)


@pytest.fixture
def empty_dataframe() -> pd.DataFrame:
    """Return an empty DataFrame."""
    return pd.DataFrame()


@pytest.fixture
def insufficient_data() -> pd.DataFrame:
    """Return a DataFrame with insufficient data."""
    data = {
        "open": [50000],
        "high": [50100],
        "low": [49900],
        "close": [50050],
        "volume": [1000],
    }
    return pd.DataFrame(data)


class TestPatternRecognizerInit:
    """Test PatternRecognizer initialization."""

    def test_initialization(self, pattern_recognizer: PatternRecognizer) -> None:
        """Test that PatternRecognizer initializes correctly."""
        assert pattern_recognizer is not None
        assert len(pattern_recognizer._pattern_methods) == 13
        assert PatternName.HAMMER in pattern_recognizer._pattern_methods
        assert PatternName.DOJI in pattern_recognizer._pattern_methods

    def test_scoring_constants(self, pattern_recognizer: PatternRecognizer) -> None:
        """Test that scoring constants are set correctly."""
        assert pattern_recognizer.SCORE_CONFIRMING_BULLISH == 20
        assert pattern_recognizer.SCORE_CONFIRMING_BEARISH == 20
        assert pattern_recognizer.SCORE_CONTRADICTING_BULLISH == -15
        assert pattern_recognizer.SCORE_CONTRADICTING_BEARISH == -15


class TestHammerDetection:
    """Test hammer pattern detection."""

    def test_detect_hammer(self, pattern_recognizer: PatternRecognizer, hammer_data: pd.DataFrame) -> None:
        """Test hammer pattern detection."""
        result = pattern_recognizer._detect_hammer(hammer_data)

        assert isinstance(result, PatternResult)
        assert result.detected is True
        assert result.pattern_name == PatternName.HAMMER.value
        assert result.pattern_type == PatternType.REVERSAL
        assert result.strength > 0
        assert result.score == pattern_recognizer.SCORE_CONFIRMING_BULLISH
        assert "Hammer detected" in result.description

    def test_detect_hammer_insufficient_data(self, pattern_recognizer: PatternRecognizer, insufficient_data: pd.DataFrame) -> None:
        """Test hammer detection with insufficient data."""
        result = pattern_recognizer._detect_hammer(insufficient_data)

        assert result.detected is False
        assert "Insufficient data" in result.description

    def test_detect_hammer_not_in_downtrend(self, pattern_recognizer: PatternRecognizer) -> None:
        """Test that hammer is not detected without downtrend."""
        # Uptrend data (no hammer)
        data = {
            "open": [50000, 50100, 50200, 50300, 50400],
            "high": [50100, 50200, 50300, 50400, 50500],
            "low": [49900, 50000, 50100, 50200, 50300],
            "close": [50050, 50150, 50250, 50350, 50450],
            "volume": [1000, 1000, 1000, 1000, 1000],
        }
        df = pd.DataFrame(data)

        result = pattern_recognizer._detect_hammer(df)
        assert result.detected is False

    def test_detect_hammer_via_detect_pattern(self, pattern_recognizer: PatternRecognizer, hammer_data: pd.DataFrame) -> None:
        """Test hammer detection via detect_pattern method."""
        result = pattern_recognizer.detect_pattern(hammer_data, PatternName.HAMMER)

        assert result["detected"] is True
        assert result["type"] == PatternType.REVERSAL.value
        assert result["score"] == pattern_recognizer.SCORE_CONFIRMING_BULLISH


class TestInvertedHammerDetection:
    """Test inverted hammer pattern detection."""

    def test_detect_inverted_hammer(self, pattern_recognizer: PatternRecognizer, inverted_hammer_data: pd.DataFrame) -> None:
        """Test inverted hammer pattern detection."""
        result = pattern_recognizer._detect_inverted_hammer(inverted_hammer_data)

        assert isinstance(result, PatternResult)
        assert result.detected is True
        assert result.pattern_name == PatternName.INVERTED_HAMMER.value
        assert result.pattern_type == PatternType.REVERSAL
        assert result.strength > 0
        assert result.score == pattern_recognizer.SCORE_CONFIRMING_BULLISH

    def test_detect_inverted_hammer_insufficient_data(self, pattern_recognizer: PatternRecognizer, insufficient_data: pd.DataFrame) -> None:
        """Test inverted hammer detection with insufficient data."""
        result = pattern_recognizer._detect_inverted_hammer(insufficient_data)

        assert result.detected is False
        assert "Insufficient data" in result.description


class TestEngulfingPatterns:
    """Test engulfing pattern detection."""

    def test_detect_bullish_engulfing(self, pattern_recognizer: PatternRecognizer, bullish_engulfing_data: pd.DataFrame) -> None:
        """Test bullish engulfing pattern detection."""
        result = pattern_recognizer._detect_bullish_engulfing(bullish_engulfing_data)

        assert isinstance(result, PatternResult)
        assert result.detected is True
        assert result.pattern_name == PatternName.BULLISH_ENGULFING.value
        assert result.pattern_type == PatternType.REVERSAL
        assert result.score == pattern_recognizer.SCORE_CONFIRMING_BULLISH

    def test_detect_bearish_engulfing(self, pattern_recognizer: PatternRecognizer, bearish_engulfing_data: pd.DataFrame) -> None:
        """Test bearish engulfing pattern detection."""
        result = pattern_recognizer._detect_bearish_engulfing(bearish_engulfing_data)

        assert isinstance(result, PatternResult)
        assert result.detected is True
        assert result.pattern_name == PatternName.BEARISH_ENGULFING.value
        assert result.pattern_type == PatternType.REVERSAL
        assert result.score == pattern_recognizer.SCORE_CONFIRMING_BEARISH

    def test_bullish_engulfing_not_detected_without_downtrend(self, pattern_recognizer: PatternRecognizer) -> None:
        """Test that bullish engulfing requires preceding bearish candle."""
        data = {
            "open": [50000, 50100],  # Both bullish
            "high": [50200, 50300],
            "low": [49900, 50000],
            "close": [50100, 50200],  # Both bullish
            "volume": [1000, 1000],
        }
        df = pd.DataFrame(data)

        result = pattern_recognizer._detect_bullish_engulfing(df)
        assert result.detected is False


class TestStarPatterns:
    """Test morning and evening star pattern detection."""

    def test_detect_morning_star(self, pattern_recognizer: PatternRecognizer, morning_star_data: pd.DataFrame) -> None:
        """Test morning star pattern detection."""
        result = pattern_recognizer._detect_morning_star(morning_star_data)

        assert isinstance(result, PatternResult)
        assert result.detected is True
        assert result.pattern_name == PatternName.MORNING_STAR.value
        assert result.pattern_type == PatternType.REVERSAL
        assert result.score == pattern_recognizer.SCORE_CONFIRMING_BULLISH

    def test_detect_evening_star(self, pattern_recognizer: PatternRecognizer, evening_star_data: pd.DataFrame) -> None:
        """Test evening star pattern detection."""
        result = pattern_recognizer._detect_evening_star(evening_star_data)

        assert isinstance(result, PatternResult)
        assert result.detected is True
        assert result.pattern_name == PatternName.EVENING_STAR.value
        assert result.pattern_type == PatternType.REVERSAL
        assert result.score == pattern_recognizer.SCORE_CONFIRMING_BEARISH

    def test_morning_star_insufficient_data(self, pattern_recognizer: PatternRecognizer, insufficient_data: pd.DataFrame) -> None:
        """Test morning star detection with insufficient data."""
        result = pattern_recognizer._detect_morning_star(insufficient_data)

        assert result.detected is False
        assert "Insufficient data" in result.description


class TestDojiDetection:
    """Test doji pattern detection."""

    def test_detect_doji(self, pattern_recognizer: PatternRecognizer, doji_data: pd.DataFrame) -> None:
        """Test doji pattern detection."""
        result = pattern_recognizer._detect_doji(doji_data)

        assert isinstance(result, PatternResult)
        assert result.detected is True
        assert result.pattern_name == PatternName.DOJI.value
        assert result.pattern_type == PatternType.INDECISION
        assert result.score == 0  # Neutral score for indecision
        assert "indecision" in result.description.lower()

    def test_detect_doji_empty_data(self, pattern_recognizer: PatternRecognizer, empty_dataframe: pd.DataFrame) -> None:
        """Test doji detection with empty data."""
        result = pattern_recognizer._detect_doji(empty_dataframe)

        assert result.detected is False
        assert "Insufficient data" in result.description

    def test_no_doji_with_large_body(self, pattern_recognizer: PatternRecognizer) -> None:
        """Test that large body candles are not detected as doji."""
        data = {
            "open": [50000],
            "high": [50500],
            "low": [49500],
            "close": [50200],  # Large body, not a doji
            "volume": [1000],
        }
        df = pd.DataFrame(data)

        result = pattern_recognizer._detect_doji(df)
        assert result.detected is False


class TestPatternScoring:
    """Test pattern scoring with signal confirmation/contradiction."""

    def test_pattern_confirms_buy_signal(self, pattern_recognizer: PatternRecognizer, hammer_data: pd.DataFrame) -> None:
        """Test that bullish patterns confirm BUY signals."""
        result = pattern_recognizer.analyze(hammer_data, signal_direction="BUY", timeframe="5m")

        assert result.pattern_detected == PatternName.HAMMER.value
        assert result.confirms_signal is True
        assert result.contradicts_signal is False

    def test_pattern_confirms_sell_signal(self, pattern_recognizer: PatternRecognizer, bearish_engulfing_data: pd.DataFrame) -> None:
        """Test that bearish patterns confirm SELL signals."""
        result = pattern_recognizer.analyze(bearish_engulfing_data, signal_direction="SELL", timeframe="5m")

        assert result.pattern_detected == PatternName.BEARISH_ENGULFING.value
        assert result.confirms_signal is True
        assert result.contradicts_signal is False

    def test_pattern_contradicts_buy_signal(self, pattern_recognizer: PatternRecognizer, bearish_engulfing_data: pd.DataFrame) -> None:
        """Test that bearish patterns contradict BUY signals."""
        result = pattern_recognizer.analyze(bearish_engulfing_data, signal_direction="BUY", timeframe="5m")

        assert result.pattern_detected == PatternName.BEARISH_ENGULFING.value
        assert result.confirms_signal is False
        assert result.contradicts_signal is True

    def test_pattern_contradicts_sell_signal(self, pattern_recognizer: PatternRecognizer, hammer_data: pd.DataFrame) -> None:
        """Test that bullish patterns contradict SELL signals."""
        result = pattern_recognizer.analyze(hammer_data, signal_direction="SELL", timeframe="5m")

        assert result.pattern_detected == PatternName.HAMMER.value
        assert result.confirms_signal is False
        assert result.contradicts_signal is True

    def test_doji_neutral_for_signals(self, pattern_recognizer: PatternRecognizer, doji_data: pd.DataFrame) -> None:
        """Test that doji is neutral for signals."""
        result = pattern_recognizer.analyze(doji_data, signal_direction="BUY", timeframe="5m")

        assert result.pattern_detected == PatternName.DOJI.value
        assert result.confirms_signal is False
        assert result.contradicts_signal is False


class TestGetAllPatterns:
    """Test getting all detected patterns."""

    def test_get_all_patterns(self, pattern_recognizer: PatternRecognizer, hammer_data: pd.DataFrame) -> None:
        """Test getting all detected patterns."""
        patterns = pattern_recognizer.get_all_patterns(hammer_data)

        assert isinstance(patterns, list)
        # Should find hammer at minimum
        pattern_names = [p["pattern"] for p in patterns]
        assert PatternName.HAMMER.value in pattern_names

        # Should be sorted by strength descending
        if len(patterns) > 1:
            for i in range(len(patterns) - 1):
                assert patterns[i]["strength"] >= patterns[i + 1]["strength"]


class TestCandleMetrics:
    """Test candle metrics calculation."""

    def test_get_candle_metrics(self, pattern_recognizer: PatternRecognizer, hammer_data: pd.DataFrame) -> None:
        """Test candle metrics calculation."""
        metrics = pattern_recognizer._get_candle_metrics(hammer_data, idx=-1)

        assert "open" in metrics
        assert "high" in metrics
        assert "low" in metrics
        assert "close" in metrics
        assert "body" in metrics
        assert "range" in metrics
        assert "upper_shadow" in metrics
        assert "lower_shadow" in metrics
        assert "body_ratio" in metrics
        assert "is_bullish" in metrics

        # For hammer, lower shadow should be significant
        assert metrics["lower_ratio"] > 0.5

    def test_get_candle_metrics_invalid_index(self, pattern_recognizer: PatternRecognizer, hammer_data: pd.DataFrame) -> None:
        """Test candle metrics with invalid index."""
        metrics = pattern_recognizer._get_candle_metrics(hammer_data, idx=100)
        assert metrics == {}

    def test_get_candle_metrics_negative_index(self, pattern_recognizer: PatternRecognizer, hammer_data: pd.DataFrame) -> None:
        """Test candle metrics with negative index."""
        metrics = pattern_recognizer._get_candle_metrics(hammer_data, idx=-2)

        assert metrics["open"] == 49700
        assert metrics["close"] == 49650


class TestDetectPatternMethod:
    """Test the detect_pattern method."""

    def test_detect_pattern_with_enum(self, pattern_recognizer: PatternRecognizer, hammer_data: pd.DataFrame) -> None:
        """Test detect_pattern with PatternName enum."""
        result = pattern_recognizer.detect_pattern(hammer_data, PatternName.HAMMER)

        assert result["detected"] is True
        assert result["type"] == PatternType.REVERSAL.value

    def test_detect_pattern_with_string(self, pattern_recognizer: PatternRecognizer, hammer_data: pd.DataFrame) -> None:
        """Test detect_pattern with string pattern name."""
        result = pattern_recognizer.detect_pattern(hammer_data, "hammer")

        assert result["detected"] is True
        assert result["type"] == PatternType.REVERSAL.value

    def test_detect_unknown_pattern(self, pattern_recognizer: PatternRecognizer, hammer_data: pd.DataFrame) -> None:
        """Test detect_pattern with unknown pattern."""
        result = pattern_recognizer.detect_pattern(hammer_data, "unknown_pattern")

        assert result["detected"] is False
        assert "Unknown pattern" in result["description"]


class TestAnalyzeEmptyData:
    """Test analyze method with empty or insufficient data."""

    def test_analyze_empty_dataframe(self, pattern_recognizer: PatternRecognizer) -> None:
        """Test analyze with empty DataFrame."""
        result = pattern_recognizer.analyze(pd.DataFrame(), signal_direction="BUY")

        assert result.pattern_detected is None
        assert result.pattern_strength == 0
        assert result.confirms_signal is False
        assert result.contradicts_signal is False

    def test_analyze_insufficient_data(self, pattern_recognizer: PatternRecognizer) -> None:
        """Test analyze with insufficient data."""
        data = {
            "open": [50000, 50100],
            "high": [50200, 50300],
            "low": [49900, 50000],
            "close": [50100, 50200],
            "volume": [1000, 1000],
        }
        df = pd.DataFrame(data)

        result = pattern_recognizer.analyze(df, signal_direction="BUY")

        assert result.pattern_detected is None
        assert result.pattern_strength == 0
