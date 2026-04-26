"""Tests for the trade analyzer module (main orchestrator).

All prices are in PAISA (integer) - 1 Rupee = 100 paisa.
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from zoneinfo import ZoneInfo

from src.research.models import (
    MarketRegime,
    OptimalEntryType,
    SignalStrength,
    TradeResearchReport,
    TradeVerdict,
    VolatilityRegime,
)
from src.research.trade_analyzer import (
    CorrelationAnalyzer,
    EventCalendarAnalyzer,
    TradeAnalyzer,
    TradeScorecard,
)
from src.research.market_regime import SignalDirection, StrategyType

IST = ZoneInfo("Asia/Kolkata")


@pytest.fixture
def mock_settings():
    """Return mock settings."""
    return MagicMock()


@pytest.fixture
def mock_bar_builder():
    """Return mock BarBuilder."""
    mock = MagicMock()

    # Create sample bars for different timeframes
    np.random.seed(42)

    def create_bars(n=100):
        dates = pd.date_range(start="2024-01-01 09:15", periods=n, freq="5min", tz=IST)
        close = np.linspace(50000, 52000, n)
        return pd.DataFrame({
            "open": (close - 50).astype(int),
            "high": (close + 100).astype(int),
            "low": (close - 100).astype(int),
            "close": close.astype(int),
            "volume": np.random.randint(1000, 10000, n),
        }, index=dates)

    mock.get_bars.return_value = create_bars()
    return mock


@pytest.fixture
def mock_historical_fetcher():
    """Return mock HistoricalDataFetcher."""
    mock = MagicMock()

    np.random.seed(42)
    dates = pd.date_range(start="2024-01-01", periods=30, freq="D", tz=IST)
    close = np.linspace(50000, 52000, 30)

    mock.fetch_candles.return_value = pd.DataFrame({
        "open": (close - 100).astype(int),
        "high": (close + 200).astype(int),
        "low": (close - 200).astype(int),
        "close": close.astype(int),
        "volume": np.random.randint(100000, 1000000, 30),
    }, index=dates)

    return mock


@pytest.fixture
def mock_instrument_manager():
    """Return mock InstrumentManager."""
    mock = MagicMock()
    mock.get_weekly_expiry.return_value = "2024-03-28"
    mock.get_option_chain.return_value = pd.DataFrame({
        "strike_price": [46000, 46100, 46200],
        "expiry": ["2024-03-28"] * 3,
        "ce_oi": [100000, 150000, 80000],
        "ce_volume": [50000, 75000, 40000],
        "ce_iv": [15.0, 16.0, 17.0],
        "ce_ltp": [150.0, 100.0, 60.0],
        "pe_oi": [80000, 120000, 90000],
        "pe_volume": [40000, 60000, 45000],
        "pe_iv": [16.0, 15.5, 16.5],
        "pe_ltp": [80.0, 120.0, 180.0],
    })
    return mock


@pytest.fixture
def mock_signal():
    """Return a mock trading signal."""
    mock = MagicMock()
    mock.symbol = "NSE_EQ:RELIANCE"
    mock.signal_type = MagicMock()
    mock.signal_type.value = "BUY"
    mock.entry_price = 50000
    mock.strategy = "EMA_Crossover"
    mock.timestamp = datetime.now(IST)
    mock.underlying = None
    return mock


@pytest.fixture
def mock_option_signal():
    """Return a mock option trading signal."""
    mock = MagicMock()
    mock.symbol = "NSE_FO|BANKNIFTY24MAR46000CE"
    mock.signal_type = MagicMock()
    mock.signal_type.value = "BUY_CE"
    mock.entry_price = 5000
    mock.strategy = "Options_Strategy"
    mock.timestamp = datetime.now(IST)
    mock.underlying = "NSE_INDEX|Nifty Bank"
    return mock


@pytest.fixture
def trade_analyzer(mock_settings, mock_bar_builder, mock_historical_fetcher, mock_instrument_manager):
    """Return a TradeAnalyzer instance with mocked dependencies."""
    return TradeAnalyzer(
        settings=mock_settings,
        bar_builder=mock_bar_builder,
        historical_fetcher=mock_historical_fetcher,
        instrument_manager=mock_instrument_manager,
    )


@pytest.fixture
def correlation_analyzer():
    """Return a CorrelationAnalyzer instance."""
    return CorrelationAnalyzer()


@pytest.fixture
def event_analyzer():
    """Return an EventCalendarAnalyzer instance."""
    return EventCalendarAnalyzer()


class TestTradeAnalyzerInit:
    """Test TradeAnalyzer initialization."""

    def test_initialization(self, trade_analyzer: TradeAnalyzer) -> None:
        """Test that TradeAnalyzer initializes correctly."""
        assert trade_analyzer is not None
        assert trade_analyzer._technical_analyzer is not None
        assert trade_analyzer._regime_detector is not None
        assert trade_analyzer._volume_analyzer is not None
        assert trade_analyzer._options_analyzer is not None
        assert trade_analyzer._correlation_analyzer is not None
        assert trade_analyzer._event_analyzer is not None
        assert trade_analyzer._pattern_recognizer is not None


class TestTradeScorecard:
    """Test TradeScorecard inner class."""

    def test_scorecard_initialization(self) -> None:
        """Test TradeScorecard initialization."""
        scorecard = TradeScorecard()

        assert scorecard.components == []
        assert scorecard.min_execute_score == 65.0
        assert scorecard.score_for_full_size == 75.0

    def test_add_component(self) -> None:
        """Test adding components to scorecard."""
        scorecard = TradeScorecard()
        component = MagicMock()
        component.name = "test"
        component.score = 80.0
        component.weight = 0.2
        component.confidence = 90.0

        scorecard.add_component(component)

        assert len(scorecard.components) == 1
        assert scorecard.components[0] == component

    def test_calculate_final_score(self) -> None:
        """Test final score calculation."""
        scorecard = TradeScorecard()

        # Add test components
        for name, score, weight in [("comp1", 80.0, 0.5), ("comp2", 60.0, 0.5)]:
            comp = MagicMock()
            comp.name = name
            comp.score = score
            comp.weight = weight
            comp.confidence = 100.0
            scorecard.add_component(comp)

        final_score = scorecard.calculate_final_score()

        # (80 * 0.5 + 60 * 0.5) = 70
        assert final_score == 70.0

    def test_get_verdict_execute(self) -> None:
        """Test EXECUTE verdict."""
        scorecard = TradeScorecard()
        verdict = scorecard.get_verdict(80.0)

        assert verdict == TradeVerdict.EXECUTE

    def test_get_verdict_reduce_size(self) -> None:
        """Test REDUCE_SIZE verdict."""
        scorecard = TradeScorecard()
        verdict = scorecard.get_verdict(70.0)

        assert verdict == TradeVerdict.REDUCE_SIZE

    def test_get_verdict_skip(self) -> None:
        """Test SKIP verdict."""
        scorecard = TradeScorecard()
        verdict = scorecard.get_verdict(50.0)

        assert verdict == TradeVerdict.SKIP

    def test_get_signal_strength(self) -> None:
        """Test signal strength classification."""
        scorecard = TradeScorecard()

        assert scorecard.get_signal_strength(90.0) == SignalStrength.STRONG_BUY
        assert scorecard.get_signal_strength(80.0) == SignalStrength.BUY
        assert scorecard.get_signal_strength(70.0) == SignalStrength.WEAK_BUY
        assert scorecard.get_signal_strength(50.0) == SignalStrength.NEUTRAL

    def test_get_adjustments(self) -> None:
        """Test getting trade adjustments."""
        scorecard = TradeScorecard()

        adjustments = scorecard.get_adjustments(80.0)

        assert "quantity_multiplier" in adjustments
        assert "sl_adjustment" in adjustments
        assert "target_adjustment" in adjustments
        assert "entry_type" in adjustments

        # High score should give full multiplier
        assert adjustments["quantity_multiplier"] == 1.0

    def test_get_key_risks(self) -> None:
        """Test extracting key risks."""
        scorecard = TradeScorecard()

        # Add low score component (risk)
        comp = MagicMock()
        comp.name = "risky_component"
        comp.score = 30.0
        comp.reasoning = "This is a risk factor"
        scorecard.add_component(comp)

        risks = scorecard.get_key_risks()

        assert len(risks) > 0
        assert "risky_component" in risks[0]

    def test_get_key_supports(self) -> None:
        """Test extracting key supports."""
        scorecard = TradeScorecard()

        # Add high score component (support)
        comp = MagicMock()
        comp.name = "strong_component"
        comp.score = 90.0
        comp.reasoning = "This is a supporting factor"
        scorecard.add_component(comp)

        supports = scorecard.get_key_supports()

        assert len(supports) > 0
        assert "strong_component" in supports[0]


class TestCorrelationAnalyzer:
    """Test CorrelationAnalyzer."""

    def test_correlation_analyzer_initialization(self, correlation_analyzer: CorrelationAnalyzer) -> None:
        """Test CorrelationAnalyzer initialization."""
        assert correlation_analyzer is not None
        assert correlation_analyzer._cache == {}

    def test_analyze_with_market_data(self, correlation_analyzer: CorrelationAnalyzer) -> None:
        """Test correlation analysis with market data."""
        np.random.seed(42)

        # Create symbol data
        dates = pd.date_range(start="2024-01-01", periods=30, freq="D")
        symbol_df = pd.DataFrame({
            "close": np.linspace(50000, 52000, 30),
        }, index=dates)

        # Create Nifty data (correlated)
        nifty_df = pd.DataFrame({
            "close": np.linspace(22000, 22500, 30),
        }, index=dates)

        result = correlation_analyzer.analyze("RELIANCE", symbol_df, nifty_df=nifty_df)

        assert result.name == "correlation"
        assert 0 <= result.score <= 100
        assert "nifty_correlation_5d" in str(result.data) or "Nifty" in result.reasoning

    def test_analyze_without_market_data(self, correlation_analyzer: CorrelationAnalyzer) -> None:
        """Test correlation analysis without market data."""
        dates = pd.date_range(start="2024-01-01", periods=30, freq="D")
        symbol_df = pd.DataFrame({
            "close": np.linspace(50000, 52000, 30),
        }, index=dates)

        result = correlation_analyzer.analyze("RELIANCE", symbol_df)

        assert result.name == "correlation"
        assert result.score == 50.0  # Neutral base


class TestEventCalendarAnalyzer:
    """Test EventCalendarAnalyzer."""

    def test_event_analyzer_initialization(self, event_analyzer: EventCalendarAnalyzer) -> None:
        """Test EventCalendarAnalyzer initialization."""
        assert event_analyzer is not None

    def test_analyze_weekday(self, event_analyzer: EventCalendarAnalyzer) -> None:
        """Test event analysis on regular weekday."""
        # Monday at 10 AM
        timestamp = datetime(2024, 3, 25, 10, 0, 0, tzinfo=IST)

        result = event_analyzer.analyze(timestamp)

        assert result.name == "events"
        assert 0 <= result.score <= 100
        assert result.data["is_weekly_expiry"] is False

    def test_analyze_expiry_day(self, event_analyzer: EventCalendarAnalyzer) -> None:
        """Test event analysis on expiry day."""
        # Thursday (expiry day) at 10 AM
        timestamp = datetime(2024, 3, 28, 10, 0, 0, tzinfo=IST)

        result = event_analyzer.analyze(timestamp)

        assert result.name == "events"
        assert result.data["is_weekly_expiry"] is True


class TestAnalyzeMethod:
    """Test the main analyze method."""

    def test_analyze_equity_signal(
        self,
        trade_analyzer: TradeAnalyzer,
        mock_signal: MagicMock,
    ) -> None:
        """Test analyzing an equity signal."""
        report = trade_analyzer.analyze(mock_signal)

        assert isinstance(report, TradeResearchReport)
        assert report.instrument_key == "NSE_EQ:RELIANCE"
        assert report.direction == "BUY"
        assert report.strategy_name == "EMA_Crossover"
        assert 0 <= report.final_score <= 100
        assert report.verdict in [TradeVerdict.EXECUTE, TradeVerdict.REDUCE_SIZE, TradeVerdict.SKIP]

    def test_analyze_option_signal(
        self,
        trade_analyzer: TradeAnalyzer,
        mock_option_signal: MagicMock,
    ) -> None:
        """Test analyzing an option signal."""
        report = trade_analyzer.analyze(mock_option_signal)

        assert isinstance(report, TradeResearchReport)
        assert report.instrument_key == "NSE_FO|BANKNIFTY24MAR46000CE"
        assert report.direction == "BUY"
        assert report.options_data is not None

    def test_analyze_report_components(
        self,
        trade_analyzer: TradeAnalyzer,
        mock_signal: MagicMock,
    ) -> None:
        """Test that report contains all expected components."""
        report = trade_analyzer.analyze(mock_signal)

        assert report.signal_id is not None
        assert report.timestamp is not None
        assert report.components is not None
        assert len(report.components) > 0

        # Check for expected component names
        component_names = [c.name for c in report.components]
        assert "trend_alignment" in component_names or "momentum" in component_names


class TestGatherData:
    """Test data gathering."""

    def test_gather_data_equity(
        self,
        trade_analyzer: TradeAnalyzer,
        mock_signal: MagicMock,
    ) -> None:
        """Test data gathering for equity signal."""
        data = trade_analyzer._gather_data(mock_signal)

        assert "bars_all_tf" in data
        assert "bars_5min" in data
        assert "bars_daily" in data
        assert "option_data" not in data  # Not an option signal

    def test_gather_data_option(
        self,
        trade_analyzer: TradeAnalyzer,
        mock_option_signal: MagicMock,
    ) -> None:
        """Test data gathering for option signal."""
        data = trade_analyzer._gather_data(mock_option_signal)

        assert "bars_all_tf" in data
        assert "option_data" in data
        assert data["option_data"]["instrument_key"] == mock_option_signal.symbol


class TestIsOptionSignal:
    """Test option signal detection."""

    def test_is_option_signal_ce(self, trade_analyzer: TradeAnalyzer) -> None:
        """Test detection of CE option signal."""
        signal = MagicMock()
        signal.signal_type.value = "BUY_CE"
        signal.underlying = None

        assert trade_analyzer._is_option_signal(signal) is True

    def test_is_option_signal_pe(self, trade_analyzer: TradeAnalyzer) -> None:
        """Test detection of PE option signal."""
        signal = MagicMock()
        signal.signal_type.value = "BUY_PE"
        signal.underlying = None

        assert trade_analyzer._is_option_signal(signal) is True

    def test_is_option_signal_equity(self, trade_analyzer: TradeAnalyzer) -> None:
        """Test detection of equity signal."""
        signal = MagicMock()
        signal.signal_type.value = "BUY"
        signal.underlying = None

        assert trade_analyzer._is_option_signal(signal) is False

    def test_is_option_signal_with_underlying(self, trade_analyzer: TradeAnalyzer) -> None:
        """Test detection when underlying differs from symbol."""
        signal = MagicMock()
        signal.signal_type.value = "BUY"
        signal.underlying = "NSE_INDEX|Nifty Bank"
        signal.symbol = "NSE_FO|BANKNIFTY24MAR46000CE"

        assert trade_analyzer._is_option_signal(signal) is True


class TestTimeframeConversion:
    """Test timeframe conversion."""

    def test_tf_to_string_1min(self, trade_analyzer: TradeAnalyzer) -> None:
        """Test 1min conversion."""
        assert trade_analyzer._tf_to_string(1) == "1min"

    def test_tf_to_string_5min(self, trade_analyzer: TradeAnalyzer) -> None:
        """Test 5min conversion."""
        assert trade_analyzer._tf_to_string(5) == "5min"

    def test_tf_to_string_15min(self, trade_analyzer: TradeAnalyzer) -> None:
        """Test 15min conversion."""
        assert trade_analyzer._tf_to_string(15) == "15min"

    def test_tf_to_string_1h(self, trade_analyzer: TradeAnalyzer) -> None:
        """Test 1h conversion."""
        assert trade_analyzer._tf_to_string(60) == "1h"

    def test_tf_to_string_daily(self, trade_analyzer: TradeAnalyzer) -> None:
        """Test daily conversion."""
        assert trade_analyzer._tf_to_string(1440) == "daily"

    def test_tf_to_string_custom(self, trade_analyzer: TradeAnalyzer) -> None:
        """Test custom timeframe conversion."""
        assert trade_analyzer._tf_to_string(30) == "30min"


class TestGenerateSummary:
    """Test summary generation."""

    def test_generate_summary(
        self,
        trade_analyzer: TradeAnalyzer,
        mock_signal: MagicMock,
    ) -> None:
        """Test summary generation."""
        scorecard = TradeScorecard()

        # Add a component
        comp = MagicMock()
        comp.name = "trend_alignment"
        comp.score = 85.0
        scorecard.add_component(comp)

        summary = trade_analyzer._generate_summary(
            mock_signal,
            final_score=80.0,
            verdict=TradeVerdict.EXECUTE,
            scorecard=scorecard,
        )

        assert "BUY" in summary or "buy" in summary.lower()
        assert "RELIANCE" in summary
        assert "80" in summary or "EXECUTE" in summary


class TestLogReport:
    """Test report logging."""

    def test_log_report(self, trade_analyzer: TradeAnalyzer) -> None:
        """Test report logging."""
        report = TradeResearchReport(
            signal_id="test_001",
            timestamp=datetime.now(IST),
            instrument_key="NSE_EQ:RELIANCE",
            direction="BUY",
            strategy_name="Test",
            final_score=80.0,
            components=[],
            verdict=TradeVerdict.EXECUTE,
            signal_strength=SignalStrength.BUY,
            market_regime=MarketRegime.STRONG_UPTREND,
            volatility_regime=VolatilityRegime.NORMAL,
        )

        # Should not raise
        trade_analyzer._log_report(report)


class TestShouldExecute:
    """Test execution decision."""

    def test_should_execute_execute_verdict(self, trade_analyzer: TradeAnalyzer) -> None:
        """Test should_execute for EXECUTE verdict."""
        report = TradeResearchReport(
            signal_id="test_001",
            timestamp=datetime.now(IST),
            instrument_key="NSE_EQ:RELIANCE",
            direction="BUY",
            strategy_name="Test",
            final_score=80.0,
            components=[],
            verdict=TradeVerdict.EXECUTE,
            signal_strength=SignalStrength.BUY,
            market_regime=MarketRegime.STRONG_UPTREND,
            volatility_regime=VolatilityRegime.NORMAL,
            suggested_quantity_multiplier=1.0,
            suggested_sl_adjustment=1.0,
            suggested_target_adjustment=1.0,
            optimal_entry_type=OptimalEntryType.MARKET,
        )

        should_execute, params = trade_analyzer.should_execute(report)

        assert should_execute is True
        assert params["quantity_multiplier"] == 1.0
        assert params["verdict"] == "EXECUTE"

    def test_should_execute_skip_verdict(self, trade_analyzer: TradeAnalyzer) -> None:
        """Test should_execute for SKIP verdict."""
        report = TradeResearchReport(
            signal_id="test_001",
            timestamp=datetime.now(IST),
            instrument_key="NSE_EQ:RELIANCE",
            direction="BUY",
            strategy_name="Test",
            final_score=40.0,
            components=[],
            verdict=TradeVerdict.SKIP,
            signal_strength=SignalStrength.WEAK_SELL,
            market_regime=MarketRegime.RANGING,
            volatility_regime=VolatilityRegime.NORMAL,
            suggested_quantity_multiplier=0.0,
            suggested_sl_adjustment=1.0,
            suggested_target_adjustment=1.0,
            optimal_entry_type=OptimalEntryType.MARKET,
        )

        should_execute, params = trade_analyzer.should_execute(report)

        assert should_execute is False
        assert params["quantity_multiplier"] == 0.0

    def test_should_execute_volatile_warning(self, trade_analyzer: TradeAnalyzer) -> None:
        """Test should_execute adds volatility warning."""
        report = TradeResearchReport(
            signal_id="test_001",
            timestamp=datetime.now(IST),
            instrument_key="NSE_EQ:RELIANCE",
            direction="BUY",
            strategy_name="Test",
            final_score=70.0,
            components=[],
            verdict=TradeVerdict.REDUCE_SIZE,
            signal_strength=SignalStrength.WEAK_BUY,
            market_regime=MarketRegime.VOLATILE,
            volatility_regime=VolatilityRegime.HIGH,
            suggested_quantity_multiplier=0.7,
            suggested_sl_adjustment=1.0,
            suggested_target_adjustment=1.0,
            optimal_entry_type=OptimalEntryType.MARKET,
        )

        should_execute, params = trade_analyzer.should_execute(report)

        assert should_execute is True
        assert params.get("volatility_warning") is True


class TestAsyncAnalyze:
    """Test async analyze wrapper."""

    @pytest.mark.asyncio
    async def test_analyze_async(
        self,
        trade_analyzer: TradeAnalyzer,
        mock_signal: MagicMock,
    ) -> None:
        """Test async analyze method."""
        report = await trade_analyzer.analyze_async(mock_signal)

        assert isinstance(report, TradeResearchReport)
        assert report.instrument_key == "NSE_EQ:RELIANCE"


class TestClose:
    """Test cleanup."""

    def test_close(self, trade_analyzer: TradeAnalyzer) -> None:
        """Test cleanup method."""
        # Should not raise
        trade_analyzer.close()
