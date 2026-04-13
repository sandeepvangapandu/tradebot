"""Tests for the composite Market Mood Index."""

import pytest
from src.agents.sentiment import MarketMoodCalculator, classify_mood


class TestClassifyMood:
    def test_extreme_fear(self):
        assert classify_mood(10.0) == "extreme_fear"

    def test_fear(self):
        assert classify_mood(30.0) == "fear"

    def test_neutral(self):
        assert classify_mood(50.0) == "neutral"

    def test_greed(self):
        assert classify_mood(70.0) == "greed"

    def test_extreme_greed(self):
        assert classify_mood(90.0) == "extreme_greed"


class TestMarketMoodCalculator:
    def test_default_weights(self):
        calc = MarketMoodCalculator()
        score = calc.calculate(
            news_sentiment=0.5,
            volatility_score=60.0,
            breadth_score=70.0,
        )
        assert 0 <= score.score <= 100
        assert score.classification in {
            "extreme_fear", "fear", "neutral", "greed", "extreme_greed"
        }

    def test_all_positive(self):
        calc = MarketMoodCalculator()
        score = calc.calculate(
            news_sentiment=1.0,
            volatility_score=90.0,
            breadth_score=90.0,
        )
        assert score.score > 70
        assert score.classification in {"greed", "extreme_greed"}

    def test_all_negative(self):
        calc = MarketMoodCalculator()
        score = calc.calculate(
            news_sentiment=-1.0,
            volatility_score=10.0,
            breadth_score=10.0,
        )
        assert score.score < 30
        assert score.classification in {"extreme_fear", "fear"}

    def test_custom_weights(self):
        calc = MarketMoodCalculator(
            news_weight=0.5,
            volatility_weight=0.3,
            breadth_weight=0.2,
        )
        score = calc.calculate(
            news_sentiment=0.0,
            volatility_score=50.0,
            breadth_score=50.0,
        )
        assert 40 <= score.score <= 60
