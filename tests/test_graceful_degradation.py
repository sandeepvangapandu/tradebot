"""Tests for graceful degradation functionality.

This module tests the graceful degradation decorator and its integration
with the research analyzers.
"""

import pytest
import pandas as pd
from datetime import datetime
from zoneinfo import ZoneInfo

from src.utils.degradation import (
    graceful_degrade,
    get_degraded_analyzers,
    is_degraded_result,
    DEFAULT_TREND_RETURN,
)
from src.research.models import AnalysisComponent, TradeResearchReport
from src.research.technical_analyzer import TechnicalAnalyzer
from src.research.volume_analyzer import VolumeAnalyzer
from src.research.trade_scorecard import TradeScorecard


class TestGracefulDegradeDecorator:
    """Test cases for the @graceful_degrade decorator."""

    def test_decorator_returns_default_on_exception(self):
        """Test that decorator returns default value when function raises."""

        @graceful_degrade(default_return={"score": 50, "error": True})
        def failing_function():
            raise ValueError("Intentional failure")

        result = failing_function()
        assert result == {"score": 50, "error": True}

    def test_decorator_returns_normal_result_on_success(self):
        """Test that decorator returns actual result when function succeeds."""

        @graceful_degrade(default_return={"score": 50, "error": True})
        def successful_function():
            return {"score": 85, "error": False}

        result = successful_function()
        assert result == {"score": 85, "error": False}

    def test_decorator_preserves_function_metadata(self):
        """Test that decorator preserves function name and docstring."""

        @graceful_degrade(default_return=None)
        def my_function():
            """My docstring."""
            return "success"

        assert my_function.__name__ == "my_function"
        assert my_function.__doc__ == "My docstring."

    def test_decorator_logs_failure(self):
        """Test that decorator captures the exception message in its log output.

        loguru bypasses both caplog and capsys — we verify logging behaviour by
        intercepting loguru messages via a custom sink instead.
        """
        from loguru import logger

        log_messages: list[str] = []

        sink_id = logger.add(lambda msg: log_messages.append(msg), level="WARNING")
        try:
            @graceful_degrade(default_return="fallback", log_level="warning")
            def failing_function():
                raise RuntimeError("Test error")

            result = failing_function()
            assert result == "fallback"
        finally:
            logger.remove(sink_id)

        combined = "".join(log_messages)
        assert "failing_function failed" in combined
        assert "RuntimeError" in combined


class TestDegradedAnalyzersDetection:
    """Test cases for detecting degraded analyzer results."""

    def test_is_degraded_result_with_dict(self):
        """Test detection of degraded dict results."""
        default = {"score": 50, "confidence": 0.3}

        # Exact match should be degraded
        assert is_degraded_result({"score": 50, "confidence": 0.3}, default)

        # Different values should not be degraded
        assert not is_degraded_result({"score": 75, "confidence": 0.3}, default)

    def test_get_degraded_analyzers_from_components(self):
        """Test extraction of degraded analyzer names from components."""
        components = [
            AnalysisComponent(
                name="trend",
                score=50,
                weight=0.3,
                confidence=25,  # Low confidence indicates degraded
                reasoning="Default",
                data={},
            ),
            AnalysisComponent(
                name="momentum",
                score=75,
                weight=0.25,
                confidence=80,  # Normal confidence
                reasoning="Good",
                data={},
            ),
            AnalysisComponent(
                name="volume",
                score=50,
                weight=0.2,
                confidence=20,  # Low confidence indicates degraded
                reasoning="Default",
                data={},
            ),
        ]

        degraded = get_degraded_analyzers(components)
        assert "trend" in degraded
        assert "volume" in degraded
        assert "momentum" not in degraded


class TestTechnicalAnalyzerDegradation:
    """Test graceful degradation in TechnicalAnalyzer."""

    def test_analyze_trend_alignment_with_empty_data(self):
        """Test that trend alignment returns default/neutral result on empty data."""
        analyzer = TechnicalAnalyzer()

        # Empty data should trigger neutral/degraded result
        result = analyzer.analyze_trend_alignment({}, "BUY")

        assert result.score == 50.0
        # Confidence may be 0.0 (actual neutral result) or 30.0 (graceful_degrade default);
        # both indicate degraded state
        assert result.confidence <= 30.0

    def test_analyze_momentum_with_empty_data(self):
        """Test that momentum returns default/neutral result on empty data."""
        analyzer = TechnicalAnalyzer()

        # Empty DataFrame should trigger neutral/degraded result
        empty_df = pd.DataFrame()
        result = analyzer.analyze_momentum(empty_df, "BUY")

        assert result.score == 50.0
        # Confidence may be 0.0 (actual neutral result) or 30.0 (graceful_degrade default)
        assert result.confidence <= 30.0


class TestVolumeAnalyzerDegradation:
    """Test graceful degradation in VolumeAnalyzer."""

    def test_analyze_volume_confirmation_with_insufficient_data(self):
        """Test that volume analysis returns default on failure."""
        analyzer = VolumeAnalyzer()

        # Small DataFrame should trigger graceful degradation
        small_df = pd.DataFrame({"close": [100], "volume": [1000]})
        result = analyzer.analyze_volume_confirmation(small_df, "buy")

        # Should return default degraded result
        assert result.score == 50


class TestTradeScorecardDegradationHandling:
    """Test degraded analyzer handling in TradeScorecard."""

    def test_score_capped_when_majority_degraded(self):
        """Test that final score is capped when >50% analyzers degraded."""
        scorecard = TradeScorecard()

        # Create components where majority are degraded (low confidence)
        components = [
            AnalysisComponent(
                name="trend",
                score=85,  # High score but low confidence
                weight=0.3,
                confidence=25,  # Degraded
                reasoning="Degraded",
                data={},
            ),
            AnalysisComponent(
                name="momentum",
                score=80,
                weight=0.25,
                confidence=20,  # Degraded
                reasoning="Degraded",
                data={},
            ),
            AnalysisComponent(
                name="volume",
                score=75,
                weight=0.2,
                confidence=85,  # Normal
                reasoning="Good",
                data={},
            ),
        ]

        # 2/3 degraded (>50%), so score should be capped at 60
        final_score = scorecard.calculate_final_score(components)
        assert final_score <= 60.0

    def test_degraded_analyzers_list_in_report(self):
        """Test that degraded analyzers are tracked in the report."""
        from src.research.models import MarketRegime, VolatilityRegime
        scorecard = TradeScorecard()

        signal_info = {
            "signal_id": "test-123",
            "instrument_key": "NSE_EQ:RELIANCE",
            "direction": "BUY",
            "strategy_name": "TestStrategy",
            "market_regime": MarketRegime.RANGING,
            "volatility_regime": VolatilityRegime.NORMAL,
            "analysis_results": {},
        }

        components = [
            AnalysisComponent(
                name="trend",
                score=50,
                weight=0.3,
                confidence=25,  # Degraded
                reasoning="Default",
                data={},
            ),
            AnalysisComponent(
                name="momentum",
                score=75,
                weight=0.25,
                confidence=80,  # Normal
                reasoning="Good",
                data={},
            ),
        ]

        report = scorecard.score(components, signal_info)

        # Should have degraded_analyzers field
        assert hasattr(report, "degraded_analyzers")
        assert "trend" in report.degraded_analyzers

    def test_degradation_warning_in_key_risks(self):
        """Test that degradation warning appears in key risks when severe."""
        from src.research.models import MarketRegime, VolatilityRegime
        scorecard = TradeScorecard()

        signal_info = {
            "signal_id": "test-123",
            "instrument_key": "NSE_EQ:RELIANCE",
            "direction": "BUY",
            "strategy_name": "TestStrategy",
            "market_regime": MarketRegime.RANGING,
            "volatility_regime": VolatilityRegime.NORMAL,
            "analysis_results": {},
        }

        # Create components where >50% are degraded
        components = [
            AnalysisComponent(
                name="trend",
                score=50,
                weight=0.2,
                confidence=25,  # Degraded
                reasoning="Default",
                data={},
            ),
            AnalysisComponent(
                name="momentum",
                score=50,
                weight=0.2,
                confidence=20,  # Degraded
                reasoning="Default",
                data={},
            ),
            AnalysisComponent(
                name="volume",
                score=50,
                weight=0.2,
                confidence=15,  # Degraded
                reasoning="Default",
                data={},
            ),
            AnalysisComponent(
                name="regime",
                score=75,
                weight=0.2,
                confidence=80,  # Normal
                reasoning="Good",
                data={},
            ),
        ]

        report = scorecard.score(components, signal_info)

        # Should have DEGRADED MODE warning in key risks
        assert any("DEGRADED MODE" in risk for risk in report.key_risks)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
