"""Tests for src/strategy/conditions_news.py — news-derived condition helpers."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.strategy import conditions_news


class TestHasNegativeNewsRecent:
    def test_returns_false_without_engine(self):
        assert conditions_news.has_negative_news_recent("RELIANCE") is False

    def test_true_when_negative_dominant(self):
        with patch("src.strategy.conditions_news.NewsQuery") as mock_cls:
            inst = MagicMock()
            inst.has_negative_news.return_value = True
            mock_cls.return_value = inst
            assert conditions_news.has_negative_news_recent("RELIANCE", db_engine=MagicMock()) is True


class TestHasPositiveNewsRecent:
    def test_true_when_positive_dominant(self):
        with patch("src.strategy.conditions_news.NewsQuery") as mock_cls:
            inst = MagicMock()
            inst.has_positive_news.return_value = True
            mock_cls.return_value = inst
            assert conditions_news.has_positive_news_recent("TCS", db_engine=MagicMock()) is True


class TestAggregateSentimentAbove:
    def test_true_when_avg_above_threshold(self):
        with patch("src.strategy.conditions_news.NewsQuery") as mock_cls:
            inst = MagicMock()
            inst.aggregate_sentiment.return_value = {"avg_sentiment": 0.4, "total_articles": 10}
            mock_cls.return_value = inst
            assert conditions_news.aggregate_sentiment_above("INFY", 0.2, db_engine=MagicMock()) is True

    def test_false_when_avg_below_threshold(self):
        with patch("src.strategy.conditions_news.NewsQuery") as mock_cls:
            inst = MagicMock()
            inst.aggregate_sentiment.return_value = {"avg_sentiment": 0.1, "total_articles": 10}
            mock_cls.return_value = inst
            assert conditions_news.aggregate_sentiment_above("INFY", 0.2, db_engine=MagicMock()) is False

    def test_false_without_engine(self):
        assert conditions_news.aggregate_sentiment_above("INFY", 0.2) is False


class TestAggregateSentimentBelow:
    def test_true_when_avg_below_threshold(self):
        with patch("src.strategy.conditions_news.NewsQuery") as mock_cls:
            inst = MagicMock()
            inst.aggregate_sentiment.return_value = {"avg_sentiment": -0.4, "total_articles": 8}
            mock_cls.return_value = inst
            assert conditions_news.aggregate_sentiment_below("ITC", -0.2, db_engine=MagicMock()) is True


class TestNewsVolumeSpike:
    def test_false_without_engine(self):
        assert conditions_news.news_volume_spike("RELIANCE") is False
