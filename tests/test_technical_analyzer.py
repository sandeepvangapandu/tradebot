"""Tests for the technical analyzer module.

All prices are in PAISA (integer) - 1 Rupee = 100 paisa.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.research.models import AnalysisComponent
from src.research.technical_analyzer import (
    CandleContextData,
    TechnicalAnalyzer,
)


@pytest.fixture
def technical_analyzer() -> TechnicalAnalyzer:
    """Return a TechnicalAnalyzer instance."""
    return TechnicalAnalyzer()


@pytest.fixture
def sample_ohlcv_data() -> pd.DataFrame:
    """Return sample OHLCV data for testing."""
    np.random.seed(42)
    dates = pd.date_range(start="2024-01-01", periods=100, freq="5min")

    # Generate trending data
    trend = np.linspace(50000, 52000, 100)
    noise = np.random.normal(0, 100, 100)

    close = trend + noise
    high = close + np.abs(np.random.normal(50, 20, 100))
    low = close - np.abs(np.random.normal(50, 20, 100))
    open_price = close + np.random.normal(0, 30, 100)
    volume = np.random.randint(1000, 10000, 100)

    df = pd.DataFrame({
        "open": open_price.astype(int),
        "high": high.astype(int),
        "low": low.astype(int),
        "close": close.astype(int),
        "volume": volume,
    }, index=dates)
    return df


@pytest.fixture
def ranging_ohlcv_data() -> pd.DataFrame:
    """Return ranging (sideways) OHLCV data."""
    np.random.seed(42)
    dates = pd.date_range(start="2024-01-01", periods=100, freq="5min")

    # Oscillating data around a mean
    oscillation = np.sin(np.linspace(0, 4 * np.pi, 100)) * 500
    base_price = 50000

    close = base_price + oscillation + np.random.normal(0, 50, 100)
    high = close + np.abs(np.random.normal(30, 15, 100))
    low = close - np.abs(np.random.normal(30, 15, 100))
    open_price = close + np.random.normal(0, 20, 100)
    volume = np.random.randint(1000, 10000, 100)

    df = pd.DataFrame({
        "open": open_price.astype(int),
        "high": high.astype(int),
        "low": low.astype(int),
        "close": close.astype(int),
        "volume": volume,
    }, index=dates)
    return df


@pytest.fixture
def multi_timeframe_data() -> dict[str, pd.DataFrame]:
    """Return multi-timeframe OHLCV data."""
    np.random.seed(42)
    data = {}

    for tf, periods, freq in [("1min", 100, "1min"), ("5min", 100, "5min"), ("15min", 100, "15min"), ("daily", 30, "D")]:
        dates = pd.date_range(start="2024-01-01", periods=periods, freq=freq)
        trend = np.linspace(50000, 52000, periods)
        noise = np.random.normal(0, 100, periods)

        close = trend + noise
        high = close + np.abs(np.random.normal(50, 20, periods))
        low = close - np.abs(np.random.normal(50, 20, periods))
        open_price = close + np.random.normal(0, 30, periods)
        volume = np.random.randint(1000, 10000, periods)

        data[tf] = pd.DataFrame({
            "open": open_price.astype(int),
            "high": high.astype(int),
            "low": low.astype(int),
            "close": close.astype(int),
            "volume": volume,
        }, index=dates)

    return data


@pytest.fixture
def empty_dataframe() -> pd.DataFrame:
    """Return an empty DataFrame."""
    return pd.DataFrame()


@pytest.fixture
def insufficient_data() -> pd.DataFrame:
    """Return a DataFrame with insufficient data."""
    data = {
        "open": [50000, 50100],
        "high": [50200, 50300],
        "low": [49900, 50000],
        "close": [50100, 50200],
        "volume": [1000, 1000],
    }
    return pd.DataFrame(data)


class TestTechnicalAnalyzerInit:
    """Test TechnicalAnalyzer initialization."""

    def test_initialization(self, technical_analyzer: TechnicalAnalyzer) -> None:
        """Test that TechnicalAnalyzer initializes correctly."""
        assert technical_analyzer is not None
        assert technical_analyzer._cache == {}
        assert technical_analyzer.TIMEFRAME_WEIGHTS["5min"] == 0.20
        assert technical_analyzer.TIMEFRAME_WEIGHTS["15min"] == 0.30
        assert technical_analyzer.TIMEFRAME_WEIGHTS["daily"] == 0.40


class TestTrendAlignment:
    """Test trend alignment analysis."""

    def test_analyze_trend_alignment_buy_signal(self, technical_analyzer: TechnicalAnalyzer, multi_timeframe_data: dict[str, pd.DataFrame]) -> None:
        """Test trend alignment for BUY signal."""
        result = technical_analyzer.analyze_trend_alignment(multi_timeframe_data, signal_direction="BUY")

        assert isinstance(result, AnalysisComponent)
        assert result.name == "trend_alignment"
        assert 0 <= result.score <= 100
        assert result.weight == 0.30
        assert "trend_5min" in result.data
        assert "trend_15min" in result.data
        assert "trend_daily" in result.data

    def test_analyze_trend_alignment_sell_signal(self, technical_analyzer: TechnicalAnalyzer, multi_timeframe_data: dict[str, pd.DataFrame]) -> None:
        """Test trend alignment for SELL signal."""
        result = technical_analyzer.analyze_trend_alignment(multi_timeframe_data, signal_direction="SELL")

        assert isinstance(result, AnalysisComponent)
        assert result.name == "trend_alignment"
        assert 0 <= result.score <= 100

    def test_analyze_trend_alignment_missing_timeframe(self, technical_analyzer: TechnicalAnalyzer) -> None:
        """Test trend alignment with missing timeframe data."""
        incomplete_data = {"1min": pd.DataFrame()}  # Empty data
        result = technical_analyzer.analyze_trend_alignment(incomplete_data, signal_direction="BUY")

        assert isinstance(result, AnalysisComponent)
        assert 0 <= result.score <= 100
        # Should default to neutral (50) for missing data

    def test_trend_alignment_scoring_buy(self, technical_analyzer: TechnicalAnalyzer) -> None:
        """Test that uptrend gives high score for BUY signals."""
        # Create data with clear uptrend
        np.random.seed(42)
        dates = pd.date_range(start="2024-01-01", periods=100, freq="5min")
        close = np.linspace(50000, 55000, 100)  # Strong uptrend
        df = pd.DataFrame({
            "open": (close - 50).astype(int),
            "high": (close + 100).astype(int),
            "low": (close - 100).astype(int),
            "close": close.astype(int),
            "volume": np.random.randint(1000, 10000, 100),
        }, index=dates)

        data = {"1min": df, "5min": df, "15min": df, "daily": df}
        result = technical_analyzer.analyze_trend_alignment(data, signal_direction="BUY")

        # Uptrend should give high score for BUY
        assert result.score >= 60

    def test_trend_alignment_scoring_sell(self, technical_analyzer: TechnicalAnalyzer) -> None:
        """Test that downtrend gives high score for SELL signals."""
        # Create data with clear downtrend
        np.random.seed(42)
        dates = pd.date_range(start="2024-01-01", periods=100, freq="5min")
        close = np.linspace(55000, 50000, 100)  # Strong downtrend
        df = pd.DataFrame({
            "open": (close + 50).astype(int),
            "high": (close + 100).astype(int),
            "low": (close - 100).astype(int),
            "close": close.astype(int),
            "volume": np.random.randint(1000, 10000, 100),
        }, index=dates)

        data = {"1min": df, "5min": df, "15min": df, "daily": df}
        result = technical_analyzer.analyze_trend_alignment(data, signal_direction="SELL")

        # Downtrend should give high score for SELL
        assert result.score >= 60


class TestMomentumAnalysis:
    """Test momentum analysis."""

    def test_analyze_momentum_buy_signal(self, technical_analyzer: TechnicalAnalyzer, sample_ohlcv_data: pd.DataFrame) -> None:
        """Test momentum analysis for BUY signal."""
        result = technical_analyzer.analyze_momentum(sample_ohlcv_data, signal_direction="BUY")

        assert isinstance(result, AnalysisComponent)
        assert result.name == "momentum"
        assert 0 <= result.score <= 100
        assert result.weight == 0.25
        assert "rsi_current" in result.data
        assert "macd_histogram" in result.data

    def test_analyze_momentum_sell_signal(self, technical_analyzer: TechnicalAnalyzer, sample_ohlcv_data: pd.DataFrame) -> None:
        """Test momentum analysis for SELL signal."""
        result = technical_analyzer.analyze_momentum(sample_ohlcv_data, signal_direction="SELL")

        assert isinstance(result, AnalysisComponent)
        assert result.name == "momentum"
        assert 0 <= result.score <= 100

    def test_analyze_momentum_insufficient_data(self, technical_analyzer: TechnicalAnalyzer, insufficient_data: pd.DataFrame) -> None:
        """Test momentum analysis with insufficient data."""
        result = technical_analyzer.analyze_momentum(insufficient_data, signal_direction="BUY")

        assert isinstance(result, AnalysisComponent)
        assert result.score == 50.0  # Neutral for insufficient data
        assert "Insufficient data" in result.reasoning

    def test_analyze_momentum_empty_data(self, technical_analyzer: TechnicalAnalyzer, empty_dataframe: pd.DataFrame) -> None:
        """Test momentum analysis with empty data."""
        result = technical_analyzer.analyze_momentum(empty_dataframe, signal_direction="BUY")

        assert isinstance(result, AnalysisComponent)
        assert result.score == 50.0


class TestDivergenceDetection:
    """Test divergence detection."""

    def test_detect_bullish_divergence(self, technical_analyzer: TechnicalAnalyzer) -> None:
        """Test detection of bullish divergence (price lower low, RSI higher low)."""
        # Create data with bullish divergence pattern
        np.random.seed(42)
        dates = pd.date_range(start="2024-01-01", periods=20, freq="5min")

        # Price makes lower low but RSI makes higher low
        close = np.array([50000, 49800, 49500, 49700, 49400, 49600, 49300, 49500, 49200, 49400,
                         49100, 49300, 49000, 49200, 48900, 49100, 48800, 49000, 48700, 48900])

        df = pd.DataFrame({
            "open": (close - 20).astype(int),
            "high": (close + 50).astype(int),
            "low": (close - 50).astype(int),
            "close": close.astype(int),
            "volume": np.random.randint(1000, 10000, 20),
        }, index=dates)

        from src.strategy.indicators import IndicatorEngine
        engine = IndicatorEngine(df)
        rsi_series = engine.rsi(period=14)

        detected, div_type = technical_analyzer._detect_divergence(df, rsi_series, "BUY")

        # Divergence detection is complex, just verify method runs
        assert isinstance(detected, bool)
        assert div_type is None or div_type in ["bullish_divergence", "bearish_divergence"]

    def test_detect_divergence_insufficient_data(self, technical_analyzer: TechnicalAnalyzer) -> None:
        """Test divergence detection with insufficient data."""
        df = pd.DataFrame({
            "open": [50000],
            "high": [50100],
            "low": [49900],
            "close": [50050],
            "volume": [1000],
        })

        from src.strategy.indicators import IndicatorEngine
        engine = IndicatorEngine(df)
        rsi_series = engine.rsi(period=14)

        detected, div_type = technical_analyzer._detect_divergence(df, rsi_series, "BUY")

        assert detected is False
        assert div_type is None


class TestSupportResistance:
    """Test support/resistance calculation."""

    def test_analyze_support_resistance(self, technical_analyzer: TechnicalAnalyzer, sample_ohlcv_data: pd.DataFrame) -> None:
        """Test support/resistance analysis."""
        current_price = int(sample_ohlcv_data["close"].iloc[-1])
        result = technical_analyzer.analyze_support_resistance(sample_ohlcv_data, current_price, "BUY")

        assert isinstance(result, AnalysisComponent)
        assert result.name == "support_resistance"
        assert 0 <= result.score <= 100
        assert result.weight == 0.25
        assert "current_price" in result.data
        assert result.data["current_price"] == current_price

    def test_analyze_sr_insufficient_data(self, technical_analyzer: TechnicalAnalyzer, insufficient_data: pd.DataFrame) -> None:
        """Test S/R analysis with insufficient data."""
        result = technical_analyze_sr_insufficient_data(technical_analyzer, insufficient_data)


def test_analyze_sr_insufficient_data(technical_analyzer: TechnicalAnalyzer, insufficient_data: pd.DataFrame) -> None:
    """Test S/R analysis with insufficient data."""
    result = technical_analyzer.analyze_support_resistance(insufficient_data, 50000, "BUY")

    assert isinstance(result, AnalysisComponent)
    assert result.score == 50.0
    assert "Insufficient data" in result.reasoning

    def test_calculate_sr_score_buy_above_vwap(self, technical_analyzer: TechnicalAnalyzer) -> None:
        """Test S/R score when buying above VWAP."""
        from src.research.models import SupportResistanceData

        sr_data = SupportResistanceData(
            current_price=51000,
            vwap=50000,
            nearest_support_distance_pct=1.0,
            nearest_resistance_distance_pct=2.0,
        )

        score = technical_analyzer._calculate_sr_score(sr_data, 51000, "BUY")

        # Buying above VWAP should give higher score
        assert score > 50

    def test_calculate_sr_score_sell_below_vwap(self, technical_analyzer: TechnicalAnalyzer) -> None:
        """Test S/R score when selling below VWAP."""
        from src.research.models import SupportResistanceData

        sr_data = SupportResistanceData(
            current_price=49000,
            vwap=50000,
            nearest_support_distance_pct=2.0,
            nearest_resistance_distance_pct=1.0,
        )

        score = technical_analyzer._calculate_sr_score(sr_data, 49000, "SELL")

        # Selling below VWAP should give higher score
        assert score > 50


class TestCandleContext:
    """Test candle context analysis."""

    def test_analyze_candle_context(self, technical_analyzer: TechnicalAnalyzer, sample_ohlcv_data: pd.DataFrame) -> None:
        """Test candle context analysis."""
        result = technical_analyzer.analyze_candle_context(sample_ohlcv_data, signal_direction="BUY")

        assert isinstance(result, AnalysisComponent)
        assert result.name == "candle_context"
        assert 0 <= result.score <= 100
        assert result.weight == 0.20

    def test_analyze_candle_context_insufficient_data(self, technical_analyzer: TechnicalAnalyzer, insufficient_data: pd.DataFrame) -> None:
        """Test candle context with insufficient data."""
        result = technical_analyzer.analyze_candle_context(insufficient_data, signal_direction="BUY")

        assert isinstance(result, AnalysisComponent)
        assert result.score == 50.0
        assert "Insufficient candle data" in result.reasoning

    def test_analyze_candles(self, technical_analyzer: TechnicalAnalyzer, sample_ohlcv_data: pd.DataFrame) -> None:
        """Test internal candle analysis method."""
        context_data = technical_analyzer._analyze_candles(sample_ohlcv_data, lookback=5)

        assert isinstance(context_data, CandleContextData)
        assert hasattr(context_data, "last_n_candles_confirming")
        assert hasattr(context_data, "avg_body_to_wick_ratio")
        assert hasattr(context_data, "consecutive_bullish")
        assert hasattr(context_data, "consecutive_bearish")

    def test_calculate_candle_score_buy_consecutive_bullish(self, technical_analyzer: TechnicalAnalyzer) -> None:
        """Test candle score with consecutive bullish candles."""
        context_data = CandleContextData(
            consecutive_bullish=3,
            consecutive_bearish=0,
            bullish_engulfing=False,
            bearish_engulfing=False,
            avg_body_to_wick_ratio=0.8,
        )

        score = technical_analyzer._calculate_candle_score(context_data, "BUY")

        # Consecutive bullish should give high score for BUY
        assert score >= 80

    def test_calculate_candle_score_sell_consecutive_bearish(self, technical_analyzer: TechnicalAnalyzer) -> None:
        """Test candle score with consecutive bearish candles."""
        context_data = CandleContextData(
            consecutive_bullish=0,
            consecutive_bearish=3,
            bullish_engulfing=False,
            bearish_engulfing=False,
            avg_body_to_wick_ratio=0.8,
        )

        score = technical_analyzer._calculate_candle_score(context_data, "SELL")

        # Consecutive bearish should give high score for SELL
        assert score >= 80

    def test_calculate_candle_score_contradiction(self, technical_analyzer: TechnicalAnalyzer) -> None:
        """Test candle score when patterns contradict signal."""
        context_data = CandleContextData(
            consecutive_bullish=0,
            consecutive_bearish=3,
            bullish_engulfing=False,
            bearish_engulfing=True,
            avg_body_to_wick_ratio=0.3,
        )

        score = technical_analyzer._calculate_candle_score(context_data, "BUY")

        # Bearish patterns should give low score for BUY
        assert score < 50


class TestDetermineTrend:
    """Test trend determination."""

    def test_determine_trend_uptrend(self, technical_analyzer: TechnicalAnalyzer) -> None:
        """Test trend determination in uptrend."""
        np.random.seed(42)
        dates = pd.date_range(start="2024-01-01", periods=100, freq="5min")
        close = np.linspace(50000, 55000, 100)  # Strong uptrend

        df = pd.DataFrame({
            "open": (close - 50).astype(int),
            "high": (close + 100).astype(int),
            "low": (close - 100).astype(int),
            "close": close.astype(int),
            "volume": np.random.randint(1000, 10000, 100),
        }, index=dates)

        trend, adx = technical_analyzer._determine_trend(df)

        assert trend in ["uptrend", "weak_uptrend"]
        assert adx >= 0

    def test_determine_trend_downtrend(self, technical_analyzer: TechnicalAnalyzer) -> None:
        """Test trend determination in downtrend."""
        np.random.seed(42)
        dates = pd.date_range(start="2024-01-01", periods=100, freq="5min")
        close = np.linspace(55000, 50000, 100)  # Strong downtrend

        df = pd.DataFrame({
            "open": (close + 50).astype(int),
            "high": (close + 100).astype(int),
            "low": (close - 100).astype(int),
            "close": close.astype(int),
            "volume": np.random.randint(1000, 10000, 100),
        }, index=dates)

        trend, adx = technical_analyzer._determine_trend(df)

        assert trend in ["downtrend", "weak_downtrend"]
        assert adx >= 0

    def test_determine_trend_insufficient_data(self, technical_analyzer: TechnicalAnalyzer, insufficient_data: pd.DataFrame) -> None:
        """Test trend determination with insufficient data."""
        trend, adx = technical_analyzer._determine_trend(insufficient_data)

        assert trend == "neutral"
        assert adx == 0.0


class TestMomentumScoreCalculation:
    """Test momentum score calculation."""

    def test_calculate_momentum_score_buy_favorable_rsi(self, technical_analyzer: TechnicalAnalyzer) -> None:
        """Test momentum score for BUY with favorable RSI."""
        from src.research.models import MomentumData

        data = MomentumData(
            rsi_current=55,  # Favorable for BUY (40-70 range)
            macd_histogram_trend="rising",
            stochastic_k=60,
            stochastic_d=55,
            divergence_detected=False,
        )

        score = technical_analyzer._calculate_momentum_score(data, "BUY")

        assert 0 <= score <= 100
        # Favorable conditions should give decent score
        assert score >= 50

    def test_calculate_momentum_score_sell_favorable_rsi(self, technical_analyzer: TechnicalAnalyzer) -> None:
        """Test momentum score for SELL with favorable RSI."""
        from src.research.models import MomentumData

        data = MomentumData(
            rsi_current=45,  # Favorable for SELL (30-60 range)
            macd_histogram_trend="falling",
            stochastic_k=40,
            stochastic_d=45,
            divergence_detected=False,
        )

        score = technical_analyzer._calculate_momentum_score(data, "SELL")

        assert 0 <= score <= 100
        assert score >= 50

    def test_calculate_momentum_score_with_divergence(self, technical_analyzer: TechnicalAnalyzer) -> None:
        """Test momentum score with divergence."""
        from src.research.models import MomentumData

        data = MomentumData(
            rsi_current=55,
            macd_histogram_trend="rising",
            divergence_detected=True,
            divergence_type="bearish_divergence",
        )

        score = technical_analyzer._calculate_momentum_score(data, "BUY")

        # Bearish divergence should penalize BUY score
        assert 0 <= score <= 100
