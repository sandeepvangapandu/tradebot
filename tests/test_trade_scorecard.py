"""Tests for the trade scorecard module.

All prices are in PAISA (integer) - 1 Rupee = 100 paisa.
"""

from __future__ import annotations

import pytest

from src.research.models import (
    AnalysisComponent,
    MarketRegime,
    OptimalEntryType,
    ResearchConfig,
    SignalStrength,
    TradeVerdict,
    VolatilityRegime,
)
from src.research.trade_scorecard import (
    DEFAULT_WEIGHTS,
    TradeScorecard,
)


@pytest.fixture
def trade_scorecard() -> TradeScorecard:
    """Return a TradeScorecard instance with default weights."""
    return TradeScorecard()


@pytest.fixture
def custom_scorecard() -> TradeScorecard:
    """Return a TradeScorecard with custom weights."""
    custom_weights = {
        "trend_alignment": 0.30,
        "momentum": 0.20,
        "support_resistance": 0.15,
        "market_regime": 0.15,
        "volume": 0.10,
        "options": 0.05,
        "correlation": 0.05,
    }
    return TradeScorecard(weights=custom_weights)


@pytest.fixture
def sample_components() -> list[AnalysisComponent]:
    """Return sample analysis components."""
    return [
        AnalysisComponent(
            name="trend_alignment",
            score=80.0,
            weight=0.20,
            confidence=90.0,
            reasoning="Strong uptrend across timeframes",
            data={},
        ),
        AnalysisComponent(
            name="momentum",
            score=75.0,
            weight=0.15,
            confidence=85.0,
            reasoning="RSI in favorable zone",
            data={},
        ),
        AnalysisComponent(
            name="support_resistance",
            score=70.0,
            weight=0.10,
            confidence=80.0,
            reasoning="Price above VWAP",
            data={},
        ),
        AnalysisComponent(
            name="market_regime",
            score=85.0,
            weight=0.15,
            confidence=90.0,
            reasoning="Strong uptrend regime",
            data={},
        ),
        AnalysisComponent(
            name="volume",
            score=65.0,
            weight=0.10,
            confidence=75.0,
            reasoning="Moderate volume confirmation",
            data={},
        ),
    ]


@pytest.fixture
def low_score_components() -> list[AnalysisComponent]:
    """Return components with low scores."""
    return [
        AnalysisComponent(
            name="trend_alignment",
            score=40.0,
            weight=0.20,
            confidence=70.0,
            reasoning="Mixed trend signals",
            data={},
        ),
        AnalysisComponent(
            name="momentum",
            score=35.0,
            weight=0.15,
            confidence=65.0,
            reasoning="RSI overbought",
            data={},
        ),
        AnalysisComponent(
            name="support_resistance",
            score=45.0,
            weight=0.10,
            confidence=60.0,
            reasoning="Near resistance",
            data={},
        ),
    ]


@pytest.fixture
def high_score_components() -> list[AnalysisComponent]:
    """Return components with high scores."""
    return [
        AnalysisComponent(
            name="trend_alignment",
            score=95.0,
            weight=0.20,
            confidence=95.0,
            reasoning="Strong uptrend all timeframes",
            data={},
        ),
        AnalysisComponent(
            name="momentum",
            score=90.0,
            weight=0.15,
            confidence=90.0,
            reasoning="Strong momentum",
            data={},
        ),
        AnalysisComponent(
            name="support_resistance",
            score=88.0,
            weight=0.10,
            confidence=85.0,
            reasoning="Well above support",
            data={},
        ),
    ]


class TestTradeScorecardInit:
    """Test TradeScorecard initialization."""

    def test_initialization_default_weights(self, trade_scorecard: TradeScorecard) -> None:
        """Test initialization with default weights."""
        assert trade_scorecard is not None
        assert trade_scorecard.weights == DEFAULT_WEIGHTS
        assert isinstance(trade_scorecard.config, ResearchConfig)

    def test_initialization_custom_weights(self, custom_scorecard: TradeScorecard) -> None:
        """Test initialization with custom weights."""
        assert custom_scorecard.weights["trend_alignment"] == 0.30
        assert custom_scorecard.weights["momentum"] == 0.20

    def test_initialization_custom_config(self) -> None:
        """Test initialization with custom config."""
        config = ResearchConfig(min_score=70, score_for_full_size=85)
        scorecard = TradeScorecard(config=config)
        assert scorecard.config.min_score == 70
        assert scorecard.config.score_for_full_size == 85

    def test_invalid_weights_sum(self) -> None:
        """Test that invalid weights raise ValueError."""
        invalid_weights = {
            "trend_alignment": 0.5,
            "momentum": 0.3,
            # Sum is 0.8, not 1.0
        }
        with pytest.raises(ValueError, match="Weights must sum to 1.0"):
            TradeScorecard(weights=invalid_weights)


class TestFinalScoreCalculation:
    """Test final score calculation."""

    def test_calculate_final_score(self, trade_scorecard: TradeScorecard, sample_components: list[AnalysisComponent]) -> None:
        """Test final score calculation."""
        final_score = trade_scorecard.calculate_final_score(sample_components)

        assert 0 <= final_score <= 100
        # With these component scores, should be in reasonable range
        assert 60 <= final_score <= 90

    def test_calculate_final_score_empty_components(self, trade_scorecard: TradeScorecard) -> None:
        """Test final score with no components."""
        final_score = trade_scorecard.calculate_final_score([])

        assert final_score == 0.0

    def test_calculate_final_score_with_confidence(
        self,
        trade_scorecard: TradeScorecard,
    ) -> None:
        """Test that confidence affects final score when multiple components are present.

        With a single component, normalization cancels confidence's effect.
        With multiple components, reducing confidence on the high-score component
        pulls the weighted average toward the other (lower-score) component.
        """
        # Case 1: trend_alignment (high score) has HIGH confidence
        components_high_conf = [
            AnalysisComponent(
                name="trend_alignment",
                score=80.0,
                weight=0.18,
                confidence=100.0,
                reasoning="Test high conf",
                data={},
            ),
            AnalysisComponent(
                name="momentum",
                score=30.0,
                weight=0.13,
                confidence=100.0,
                reasoning="Test low score",
                data={},
            ),
        ]
        # Case 2: trend_alignment (high score) has LOW confidence — pulls result toward momentum's 30
        components_low_conf = [
            AnalysisComponent(
                name="trend_alignment",
                score=80.0,
                weight=0.18,
                confidence=50.0,
                reasoning="Test low conf",
                data={},
            ),
            AnalysisComponent(
                name="momentum",
                score=30.0,
                weight=0.13,
                confidence=100.0,
                reasoning="Test low score",
                data={},
            ),
        ]

        score_high = trade_scorecard.calculate_final_score(components_high_conf)
        score_low = trade_scorecard.calculate_final_score(components_low_conf)

        # Reducing confidence on the high-score component should lower the effective final score
        assert score_high > score_low

    def test_calculate_final_score_weight_normalization(
        self,
        trade_scorecard: TradeScorecard,
    ) -> None:
        """Test weight normalization when not all components present."""
        # Only one component with weight 0.2
        components = [
            AnalysisComponent(
                name="trend_alignment",
                score=80.0,
                weight=0.20,
                confidence=100.0,
                reasoning="Test",
                data={},
            ),
        ]

        final_score = trade_scorecard.calculate_final_score(components)
        # Should normalize to use available weight
        assert final_score == 80.0


class TestVerdictDetermination:
    """Test verdict determination."""

    def test_determine_verdict_execute(self, trade_scorecard: TradeScorecard) -> None:
        """Test EXECUTE verdict for high score."""
        verdict, multiplier = trade_scorecard.determine_verdict(80.0)

        assert verdict == TradeVerdict.EXECUTE
        assert multiplier == 1.0

    def test_determine_verdict_reduce_size(self, trade_scorecard: TradeScorecard) -> None:
        """Test REDUCE_SIZE verdict for medium score."""
        verdict, multiplier = trade_scorecard.determine_verdict(70.0)

        assert verdict == TradeVerdict.REDUCE_SIZE
        assert multiplier == 0.7

    def test_determine_verdict_skip(self, trade_scorecard: TradeScorecard) -> None:
        """Test SKIP verdict for low score."""
        verdict, multiplier = trade_scorecard.determine_verdict(45.0)

        assert verdict == TradeVerdict.SKIP
        assert multiplier == 0.0

    def test_determine_verdict_custom_thresholds(self) -> None:
        """Test verdict with custom thresholds."""
        config = ResearchConfig(min_score=60, score_for_full_size=80)
        scorecard = TradeScorecard(config=config)

        verdict, _ = scorecard.determine_verdict(75.0)
        assert verdict == TradeVerdict.REDUCE_SIZE

        verdict, _ = scorecard.determine_verdict(85.0)
        assert verdict == TradeVerdict.EXECUTE

    def test_determine_verdict_boundary_execute(self, trade_scorecard: TradeScorecard) -> None:
        """Test boundary condition for EXECUTE."""
        verdict, _ = trade_scorecard.determine_verdict(75.0)
        assert verdict == TradeVerdict.EXECUTE

    def test_determine_verdict_boundary_reduce(self, trade_scorecard: TradeScorecard) -> None:
        """Test boundary condition for REDUCE_SIZE."""
        verdict, _ = trade_scorecard.determine_verdict(65.0)
        assert verdict == TradeVerdict.REDUCE_SIZE

    def test_determine_verdict_boundary_skip(self, trade_scorecard: TradeScorecard) -> None:
        """Test boundary condition for SKIP."""
        verdict, _ = trade_scorecard.determine_verdict(64.9)
        assert verdict == TradeVerdict.SKIP


class TestSignalStrength:
    """Test signal strength determination."""

    def test_determine_signal_strong_buy(self, trade_scorecard: TradeScorecard) -> None:
        """Test STRONG_BUY signal strength."""
        strength = trade_scorecard.determine_signal_strength(90.0, "BUY")
        assert strength == SignalStrength.STRONG_BUY

    def test_determine_signal_buy(self, trade_scorecard: TradeScorecard) -> None:
        """Test BUY signal strength."""
        strength = trade_scorecard.determine_signal_strength(75.0, "BUY")
        assert strength == SignalStrength.BUY

    def test_determine_signal_weak_buy(self, trade_scorecard: TradeScorecard) -> None:
        """Test WEAK_BUY signal strength."""
        strength = trade_scorecard.determine_signal_strength(60.0, "BUY")
        assert strength == SignalStrength.WEAK_BUY

    def test_determine_signal_neutral(self, trade_scorecard: TradeScorecard) -> None:
        """Test NEUTRAL signal strength."""
        strength = trade_scorecard.determine_signal_strength(50.0, "BUY")
        assert strength == SignalStrength.NEUTRAL

    def test_determine_signal_strong_sell(self, trade_scorecard: TradeScorecard) -> None:
        """Test STRONG_SELL signal strength."""
        strength = trade_scorecard.determine_signal_strength(90.0, "SELL")
        assert strength == SignalStrength.STRONG_SELL

    def test_determine_signal_sell(self, trade_scorecard: TradeScorecard) -> None:
        """Test SELL signal strength."""
        strength = trade_scorecard.determine_signal_strength(75.0, "SELL")
        assert strength == SignalStrength.SELL

    def test_determine_signal_weak_sell(self, trade_scorecard: TradeScorecard) -> None:
        """Test WEAK_SELL signal strength."""
        strength = trade_scorecard.determine_signal_strength(60.0, "SELL")
        assert strength == SignalStrength.WEAK_SELL

    def test_determine_signal_contrarian_buy(self, trade_scorecard: TradeScorecard) -> None:
        """Test contrarian signals for BUY direction with very low scores (< 15)."""
        # Score < 15 maps to STRONG_SELL for BUY direction (contrarian)
        strength = trade_scorecard.determine_signal_strength(10.0, "BUY")
        assert strength == SignalStrength.STRONG_SELL  # Contrarian

    def test_determine_signal_contrarian_sell(self, trade_scorecard: TradeScorecard) -> None:
        """Test contrarian signals for SELL direction with very low scores (< 15)."""
        # Score < 15 maps to STRONG_BUY for SELL direction (contrarian)
        strength = trade_scorecard.determine_signal_strength(10.0, "SELL")
        assert strength == SignalStrength.STRONG_BUY  # Contrarian


class TestPositionSizingAdjustments:
    """Test position sizing adjustments."""

    def test_calculate_adjustments_high_volatility(self, trade_scorecard: TradeScorecard) -> None:
        """Test adjustments for high volatility."""
        analysis_results = {
            "volatility_regime": VolatilityRegime.HIGH,
            "market_regime": MarketRegime.STRONG_UPTREND,
            "trend_strength": 75,
            "volume_ratio": 1.2,
            "nearest_support_distance_pct": 1.5,
        }

        adjustments = trade_scorecard.calculate_adjustments(analysis_results, 80.0)

        assert adjustments["sl_adjustment"] == 1.25  # Widened for high volatility
        assert adjustments["target_adjustment"] == 1.3  # Extended for strong trend
        assert adjustments["entry_type"] == OptimalEntryType.MARKET

    def test_calculate_adjustments_extreme_volatility(self, trade_scorecard: TradeScorecard) -> None:
        """Test adjustments for extreme volatility."""
        analysis_results = {
            "volatility_regime": VolatilityRegime.EXTREME,
            "market_regime": MarketRegime.RANGING,
            "trend_strength": 50,
            "volume_ratio": 1.0,
        }

        adjustments = trade_scorecard.calculate_adjustments(analysis_results, 70.0)

        assert adjustments["sl_adjustment"] == 1.5  # Widened significantly

    def test_calculate_adjustments_low_volatility(self, trade_scorecard: TradeScorecard) -> None:
        """Test adjustments for low volatility."""
        analysis_results = {
            "volatility_regime": VolatilityRegime.LOW,
            "market_regime": MarketRegime.RANGING,
            "trend_strength": 50,
            "volume_ratio": 1.0,
        }

        adjustments = trade_scorecard.calculate_adjustments(analysis_results, 70.0)

        assert adjustments["sl_adjustment"] == 0.75  # Tightened for low volatility

    def test_calculate_adjustments_near_support(self, trade_scorecard: TradeScorecard) -> None:
        """Test adjustments when price is near support."""
        analysis_results = {
            "volatility_regime": VolatilityRegime.NORMAL,
            "market_regime": MarketRegime.STRONG_UPTREND,
            "trend_strength": 75,
            "volume_ratio": 1.2,
            "nearest_support_distance_pct": 0.5,  # Very close to support
        }

        adjustments = trade_scorecard.calculate_adjustments(analysis_results, 80.0)

        assert adjustments["entry_type"] == OptimalEntryType.LIMIT_AT_SUPPORT

    def test_calculate_adjustments_low_volume(self, trade_scorecard: TradeScorecard) -> None:
        """Test adjustments for low volume."""
        analysis_results = {
            "volatility_regime": VolatilityRegime.NORMAL,
            "market_regime": MarketRegime.STRONG_UPTREND,
            "trend_strength": 75,
            "volume_ratio": 0.5,  # Low volume
        }

        adjustments = trade_scorecard.calculate_adjustments(analysis_results, 80.0)

        assert adjustments["entry_type"] == OptimalEntryType.WAIT_FOR_PULLBACK

    def test_calculate_adjustments_ranging_market(self, trade_scorecard: TradeScorecard) -> None:
        """Test adjustments for ranging market."""
        analysis_results = {
            "volatility_regime": VolatilityRegime.NORMAL,
            "market_regime": MarketRegime.RANGING,
            "trend_strength": 40,
            "volume_ratio": 1.0,
        }

        adjustments = trade_scorecard.calculate_adjustments(analysis_results, 60.0)

        assert adjustments["target_adjustment"] == 0.8  # Reduced target


class TestSummaryGeneration:
    """Test summary generation."""

    def test_generate_summary_execute(self, trade_scorecard: TradeScorecard, high_score_components: list[AnalysisComponent]) -> None:
        """Test summary for EXECUTE verdict."""
        summary = trade_scorecard.generate_summary(
            components=high_score_components,
            verdict=TradeVerdict.EXECUTE,
            key_risks=[],
            key_supports=["Strong trend", "Good momentum"],
        )

        assert "meets all criteria" in summary.lower()
        assert "strong trend" in summary.lower() or "good momentum" in summary.lower()

    def test_generate_summary_reduce_size(self, trade_scorecard: TradeScorecard, sample_components: list[AnalysisComponent]) -> None:
        """Test summary for REDUCE_SIZE verdict."""
        summary = trade_scorecard.generate_summary(
            components=sample_components,
            verdict=TradeVerdict.REDUCE_SIZE,
            key_risks=["Near resistance"],
            key_supports=["Good volume"],
        )

        assert "reduced position size" in summary.lower()

    def test_generate_summary_skip(self, trade_scorecard: TradeScorecard, low_score_components: list[AnalysisComponent]) -> None:
        """Test summary for SKIP verdict."""
        summary = trade_scorecard.generate_summary(
            components=low_score_components,
            verdict=TradeVerdict.SKIP,
            key_risks=["Weak trend", "Overbought"],
            key_supports=[],
        )

        assert "does not meet minimum criteria" in summary.lower()
        assert "weak trend" in summary.lower() or "overbought" in summary.lower()

    def test_generate_summary_empty_supports(self, trade_scorecard: TradeScorecard, high_score_components: list[AnalysisComponent]) -> None:
        """Test summary with no explicit supports."""
        summary = trade_scorecard.generate_summary(
            components=high_score_components,
            verdict=TradeVerdict.EXECUTE,
            key_risks=[],
            key_supports=[],
        )

        assert "meets all criteria" in summary.lower()


class TestWeightCustomization:
    """Test weight customization."""

    def test_update_weights_valid(self, trade_scorecard: TradeScorecard) -> None:
        """Test updating weights with valid values."""
        new_weights = {
            "trend_alignment": 0.40,
            "momentum": 0.30,
            "support_resistance": 0.30,
        }

        trade_scorecard.update_weights(new_weights)

        assert trade_scorecard.weights == new_weights

    def test_update_weights_invalid_sum(self, trade_scorecard: TradeScorecard) -> None:
        """Test that invalid weight sum raises error."""
        invalid_weights = {
            "trend_alignment": 0.5,
            "momentum": 0.3,
        }

        with pytest.raises(ValueError, match="Weights must sum to 1.0"):
            trade_scorecard.update_weights(invalid_weights)

    def test_get_component_contributions(self, trade_scorecard: TradeScorecard, sample_components: list[AnalysisComponent]) -> None:
        """Test getting component contributions."""
        contributions = trade_scorecard.get_component_contributions(sample_components)

        assert isinstance(contributions, dict)
        assert "trend_alignment" in contributions
        assert "momentum" in contributions

        # Each contribution should be score * scorecard_weight * confidence
        # Note: the scorecard uses its own weights dict (DEFAULT_WEIGHTS), not comp.weight
        for comp in sample_components:
            scorecard_weight = trade_scorecard.weights.get(comp.name, 0.0)
            expected = comp.score * scorecard_weight * (comp.confidence / 100.0)
            assert contributions[comp.name] == pytest.approx(expected, rel=1e-5)


class TestScoreMethod:
    """Test the main score method."""

    def test_score_complete_report(
        self,
        trade_scorecard: TradeScorecard,
        sample_components: list[AnalysisComponent],
    ) -> None:
        """Test generating complete research report."""
        signal_info = {
            "signal_id": "test_signal_001",
            "instrument_key": "NSE_EQ:RELIANCE",
            "direction": "BUY",
            "strategy_name": "EMA_Crossover",
            "market_regime": MarketRegime.STRONG_UPTREND,
            "volatility_regime": VolatilityRegime.NORMAL,
            "analysis_results": {
                "volatility_regime": VolatilityRegime.NORMAL,
                "market_regime": MarketRegime.STRONG_UPTREND,
                "trend_strength": 80,
                "volume_ratio": 1.2,
            },
        }

        report = trade_scorecard.score(sample_components, signal_info)

        assert report.signal_id == "test_signal_001"
        assert report.instrument_key == "NSE_EQ:RELIANCE"
        assert report.direction == "BUY"
        assert report.strategy_name == "EMA_Crossover"
        assert 0 <= report.final_score <= 100
        assert report.verdict in [TradeVerdict.EXECUTE, TradeVerdict.REDUCE_SIZE, TradeVerdict.SKIP]
        assert report.market_regime == MarketRegime.STRONG_UPTREND

    def test_score_sell_direction(self, trade_scorecard: TradeScorecard, sample_components: list[AnalysisComponent]) -> None:
        """Test score method with SELL direction."""
        signal_info = {
            "signal_id": "test_signal_002",
            "instrument_key": "NSE_EQ:TCS",
            "direction": "SELL",
            "strategy_name": "RSI_Reversal",
            "market_regime": MarketRegime.STRONG_DOWNTREND,
            "volatility_regime": VolatilityRegime.NORMAL,
            "analysis_results": {},
        }

        report = trade_scorecard.score(sample_components, signal_info)

        assert report.direction == "SELL"
        assert "SELL" in report.signal_strength.value or "sell" in report.signal_strength.value

    def test_score_extracts_risks_and_supports(
        self,
        trade_scorecard: TradeScorecard,
    ) -> None:
        """Test that score method extracts risks and supports."""
        components = [
            AnalysisComponent(
                name="high_score",
                score=85.0,
                weight=0.3,
                confidence=90.0,
                reasoning="Excellent condition",
                data={},
            ),
            AnalysisComponent(
                name="low_score",
                score=30.0,
                weight=0.3,
                confidence=70.0,
                reasoning="Poor condition warning",
                data={},
            ),
        ]

        signal_info = {
            "signal_id": "test_signal_003",
            "instrument_key": "NSE_EQ:INFY",
            "direction": "BUY",
            "strategy_name": "Test",
            "market_regime": MarketRegime.RANGING,
            "volatility_regime": VolatilityRegime.NORMAL,
            "analysis_results": {},
        }

        report = trade_scorecard.score(components, signal_info)

        # Should extract low score as risk
        assert len(report.key_risks) > 0
        assert "low_score" in report.key_risks[0]

        # Should extract high score as support
        assert len(report.key_supports) > 0
        assert "high_score" in report.key_supports[0]


class TestDefaultWeights:
    """Test default weights constant."""

    def test_default_weights_sum(self) -> None:
        """Test that default weights sum to 1.0."""
        total = sum(DEFAULT_WEIGHTS.values())
        assert total == pytest.approx(1.0, abs=0.01)

    def test_default_weights_keys(self) -> None:
        """Test that default weights has expected keys."""
        expected_keys = [
            "trend_alignment",
            "momentum",
            "support_resistance",
            "market_regime",
            "volume",
            "options",
            "correlation",
            "event_calendar",
            "time_of_day",
            "candle_patterns",
        ]

        for key in expected_keys:
            assert key in DEFAULT_WEIGHTS
