"""Tests for src.research.news_query — NewsQuery API.

All tests use inline mocks.  No live DB connection required.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from src.research.news_query import NewsQuery


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_query(db_engine=None) -> NewsQuery:
    return NewsQuery(db_engine=db_engine)


def _mock_conn_engine(rows=None, scalar_value=None):
    """Build a mock engine whose execute() returns the given rows/scalar."""
    mock_result = MagicMock()
    if rows is not None:
        mock_result.fetchall.return_value = rows
        mock_result.fetchone.return_value = rows[0] if rows else None
    if scalar_value is not None:
        mock_result.scalar.return_value = scalar_value

    mock_conn = MagicMock()
    mock_conn.__enter__ = MagicMock(return_value=mock_conn)
    mock_conn.__exit__ = MagicMock(return_value=False)
    mock_conn.execute.return_value = mock_result

    mock_engine = MagicMock()
    mock_engine.connect.return_value = mock_conn
    return mock_engine


def _article_row(
    art_id: int = 1,
    source: str = "moneycontrol_markets",
    title: str = "RELIANCE profit surge beats estimates",
    link: str = "https://example.com/reliance-1",
    published_at=None,
    sentiment: float = 0.5,
    classification: str = "POSITIVE",
    confidence: float = 0.6,
):
    """Build a tuple mimicking a DB row from recent_articles_for_symbol query."""
    pub = published_at or datetime(2026, 4, 28, 9, 15, tzinfo=timezone.utc)
    return (art_id, source, title, link, pub, sentiment, classification, confidence)


def _agg_row(
    avg_sentiment: float = 0.3,
    total: int = 5,
    pos: int = 3,
    neg: int = 1,
    neu: int = 1,
):
    """Build a tuple mimicking a DB row from aggregate_sentiment query."""
    return (avg_sentiment, total, pos, neg, neu)


# ---------------------------------------------------------------------------
# test_recent_articles_filters_by_hours
# ---------------------------------------------------------------------------

class TestRecentArticlesFiltersbyHours:
    """recent_articles_for_symbol should return matching articles."""

    def test_recent_articles_filters_by_hours(self):
        rows = [
            _article_row(art_id=1, sentiment=0.5, classification="POSITIVE"),
            _article_row(art_id=2, sentiment=-0.4, classification="NEGATIVE"),
        ]
        engine = _mock_conn_engine(rows=rows)
        q = _make_query(db_engine=engine)

        results = q.recent_articles_for_symbol("RELIANCE", hours=24)

        assert len(results) == 2
        assert results[0]["id"] == 1
        assert results[0]["classification"] == "POSITIVE"
        assert results[1]["classification"] == "NEGATIVE"

    def test_recent_articles_returns_empty_when_no_engine(self):
        q = _make_query(db_engine=None)
        results = q.recent_articles_for_symbol("RELIANCE", hours=24)
        assert results == []

    def test_recent_articles_returns_empty_when_no_rows(self):
        engine = _mock_conn_engine(rows=[])
        q = _make_query(db_engine=engine)
        results = q.recent_articles_for_symbol("RELIANCE", hours=24)
        assert results == []

    def test_recent_articles_includes_all_fields(self):
        rows = [_article_row(art_id=10, source="et_markets", confidence=0.8)]
        engine = _mock_conn_engine(rows=rows)
        q = _make_query(db_engine=engine)

        results = q.recent_articles_for_symbol("RELIANCE", hours=48)

        art = results[0]
        assert art["id"] == 10
        assert art["source"] == "et_markets"
        assert art["confidence"] == pytest.approx(0.8)
        assert art["published_at"] is not None


# ---------------------------------------------------------------------------
# test_aggregate_sentiment_counts
# ---------------------------------------------------------------------------

class TestAggregateSentimentCounts:
    """aggregate_sentiment should return correct counts and dominant classification."""

    def test_aggregate_sentiment_counts(self):
        engine = _mock_conn_engine(rows=[_agg_row(avg_sentiment=0.3, total=5, pos=3, neg=1, neu=1)])
        q = _make_query(db_engine=engine)

        agg = q.aggregate_sentiment("RELIANCE", hours=24)

        assert agg["total_articles"] == 5
        assert agg["positive_count"] == 3
        assert agg["negative_count"] == 1
        assert agg["neutral_count"] == 1
        assert agg["avg_sentiment"] == pytest.approx(0.3)
        assert agg["dominant_classification"] == "POSITIVE"

    def test_aggregate_sentiment_dominant_negative(self):
        engine = _mock_conn_engine(rows=[_agg_row(avg_sentiment=-0.4, total=6, pos=1, neg=4, neu=1)])
        q = _make_query(db_engine=engine)

        agg = q.aggregate_sentiment("SBIN", hours=24)
        assert agg["dominant_classification"] == "NEGATIVE"

    def test_aggregate_sentiment_dominant_neutral_when_tied(self):
        # Equal pos and neg → dominant is NEUTRAL (tie-break rule)
        engine = _mock_conn_engine(rows=[_agg_row(avg_sentiment=0.0, total=4, pos=2, neg=2, neu=0)])
        q = _make_query(db_engine=engine)

        agg = q.aggregate_sentiment("TCS", hours=24)
        assert agg["dominant_classification"] == "NEUTRAL"

    def test_aggregate_sentiment_returns_empty_when_no_data(self):
        # Row with total=0 signals no data
        engine = _mock_conn_engine(rows=[(None, 0, None, None, None)])
        q = _make_query(db_engine=engine)

        agg = q.aggregate_sentiment("INFY", hours=24)
        assert agg["total_articles"] == 0
        assert agg["avg_sentiment"] == 0.0
        assert agg["dominant_classification"] == "NEUTRAL"

    def test_aggregate_sentiment_returns_empty_when_no_engine(self):
        q = _make_query(db_engine=None)
        agg = q.aggregate_sentiment("RELIANCE", hours=24)
        assert agg == {
            "avg_sentiment": 0.0,
            "total_articles": 0,
            "positive_count": 0,
            "negative_count": 0,
            "neutral_count": 0,
            "dominant_classification": "NEUTRAL",
        }


# ---------------------------------------------------------------------------
# test_has_negative_news_true_when_dominant
# ---------------------------------------------------------------------------

class TestHasNegativeNewsTrueWhenDominant:
    """has_negative_news should return True when high-confidence negative articles exist."""

    def test_has_negative_news_true_when_dominant(self):
        # scalar() returns count > 0
        engine = _mock_conn_engine(scalar_value=2)
        q = _make_query(db_engine=engine)

        result = q.has_negative_news("INFY", hours=24, min_confidence=0.3)
        assert result is True

    def test_has_negative_news_false_when_count_zero(self):
        engine = _mock_conn_engine(scalar_value=0)
        q = _make_query(db_engine=engine)

        result = q.has_negative_news("TCS", hours=24, min_confidence=0.3)
        assert result is False

    def test_has_negative_news_false_when_no_engine(self):
        q = _make_query(db_engine=None)
        assert q.has_negative_news("RELIANCE") is False

    def test_has_positive_news_true_when_count_positive(self):
        engine = _mock_conn_engine(scalar_value=3)
        q = _make_query(db_engine=engine)

        result = q.has_positive_news("RELIANCE", hours=24, min_confidence=0.3)
        assert result is True

    def test_has_positive_news_false_when_no_engine(self):
        q = _make_query(db_engine=None)
        assert q.has_positive_news("SBIN") is False

    def test_has_negative_news_false_on_db_error(self):
        """DB error should degrade gracefully to False."""
        mock_conn = MagicMock()
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)
        mock_conn.execute.side_effect = Exception("DB connection lost")

        mock_engine = MagicMock()
        mock_engine.connect.return_value = mock_conn

        q = _make_query(db_engine=mock_engine)
        result = q.has_negative_news("AXISBANK", hours=24)
        assert result is False
