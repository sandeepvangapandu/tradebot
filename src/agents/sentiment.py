"""Composite Market Mood Index calculator.

Combines news sentiment, volatility, and market breadth into a
single 0-100 score classified as extreme_fear to extreme_greed.
"""

from __future__ import annotations

from src.agents.models import SentimentScore


def classify_mood(score: float) -> str:
    """Classify a 0-100 mood score.

    Args:
        score: Market Mood Index (0-100).

    Returns:
        Classification string.
    """
    if score < 20:
        return "extreme_fear"
    if score < 40:
        return "fear"
    if score < 60:
        return "neutral"
    if score < 80:
        return "greed"
    return "extreme_greed"


class MarketMoodCalculator:
    """Calculates composite Market Mood Index.

    Combines three inputs into a 0-100 score:
    - News sentiment (-1 to 1, normalized to 0-100)
    - Volatility score (0-100, high vol = low score)
    - Market breadth (0-100, advance/decline ratio)

    Args:
        news_weight: Weight for news sentiment component.
        volatility_weight: Weight for volatility component.
        breadth_weight: Weight for market breadth component.
    """

    def __init__(
        self,
        news_weight: float = 0.35,
        volatility_weight: float = 0.35,
        breadth_weight: float = 0.30,
    ) -> None:
        total = news_weight + volatility_weight + breadth_weight
        self._news_w = news_weight / total
        self._vol_w = volatility_weight / total
        self._breadth_w = breadth_weight / total

    def calculate(
        self,
        news_sentiment: float,
        volatility_score: float,
        breadth_score: float,
    ) -> SentimentScore:
        """Calculate the composite Market Mood Index.

        Args:
            news_sentiment: -1.0 to 1.0 from news agent.
            volatility_score: 0-100 (inverted: high vol = low score).
            breadth_score: 0-100 from advance/decline ratio.

        Returns:
            SentimentScore with composite score and classification.
        """
        # Normalize news sentiment from [-1, 1] to [0, 100]
        news_normalized = (news_sentiment + 1.0) * 50.0

        # Clamp all components to [0, 100]
        news_normalized = max(0.0, min(100.0, news_normalized))
        volatility_score = max(0.0, min(100.0, volatility_score))
        breadth_score = max(0.0, min(100.0, breadth_score))

        # Weighted composite
        score = (
            news_normalized * self._news_w
            + volatility_score * self._vol_w
            + breadth_score * self._breadth_w
        )

        score = max(0.0, min(100.0, score))
        classification = classify_mood(score)

        return SentimentScore(
            score=score,
            classification=classification,
            news_score=news_sentiment,
            volatility_score=volatility_score,
            breadth_score=breadth_score,
        )
