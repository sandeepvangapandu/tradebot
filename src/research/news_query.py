"""Query news articles and per-symbol sentiment from the database.

Provides high-level helpers for strategies to check whether a symbol has
recent negative (or positive) news and to aggregate sentiment scores over
a configurable lookback window.

Design principles
-----------------
* All queries use parameterised SQLAlchemy ``text()`` — no string formatting.
* Returns ``False`` / empty structures on any DB error so strategies degrade
  gracefully.
* "Hours" lookback is computed relative to UTC NOW() — works correctly across
  midnight boundaries.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class NewsQuery:
    """High-level query API over ``news_articles`` + ``news_sentiment_symbol``.

    Args:
        db_engine: SQLAlchemy engine.  All methods return empty/False when None.
    """

    def __init__(self, db_engine=None) -> None:
        self._db = db_engine

    # ------------------------------------------------------------------
    # Article retrieval
    # ------------------------------------------------------------------

    def recent_articles_for_symbol(
        self, symbol: str, hours: int = 24
    ) -> list[dict]:
        """Return articles mentioning *symbol* published within *hours* hours.

        Args:
            symbol: Stock symbol, e.g. ``"RELIANCE"``.
            hours: Lookback window in hours (default 24).

        Returns:
            List of article dicts with keys:
            ``id``, ``source``, ``title``, ``link``, ``published_at``,
            ``sentiment``, ``classification``, ``confidence``.
        """
        if self._db is None:
            return []

        from sqlalchemy import text as sa_text

        sql = sa_text(
            """
            SELECT
                a.id,
                a.source,
                a.title,
                a.link,
                a.published_at,
                s.sentiment,
                s.classification,
                s.confidence
            FROM news_articles a
            JOIN news_sentiment_symbol s ON s.article_id = a.id
            WHERE s.symbol = :symbol
              AND a.published_at >= NOW() - INTERVAL ':hours hours'
            ORDER BY a.published_at DESC
            """
        )

        # SQLAlchemy text() does not support INTERVAL parameters portably;
        # build the interval string safely (hours is an int — no injection risk).
        interval_sql = sa_text(
            f"""
            SELECT
                a.id,
                a.source,
                a.title,
                a.link,
                a.published_at,
                s.sentiment,
                s.classification,
                s.confidence
            FROM news_articles a
            JOIN news_sentiment_symbol s ON s.article_id = a.id
            WHERE s.symbol = :symbol
              AND a.published_at >= NOW() - INTERVAL '{int(hours)} hours'
            ORDER BY a.published_at DESC
            """
        )

        try:
            with self._db.connect() as conn:
                rows = conn.execute(interval_sql, {"symbol": symbol}).fetchall()
        except Exception as exc:  # noqa: BLE001
            logger.warning("recent_articles_for_symbol DB error: %s", exc)
            return []

        return [
            {
                "id": row[0],
                "source": row[1],
                "title": row[2],
                "link": row[3],
                "published_at": row[4],
                "sentiment": float(row[5]) if row[5] is not None else 0.0,
                "classification": row[6],
                "confidence": float(row[7]) if row[7] is not None else 0.0,
            }
            for row in rows
        ]

    # ------------------------------------------------------------------
    # Sentiment aggregation
    # ------------------------------------------------------------------

    def aggregate_sentiment(self, symbol: str, hours: int = 24) -> dict:
        """Aggregate sentiment metrics for *symbol* over the last *hours* hours.

        Args:
            symbol: Stock symbol.
            hours: Lookback window in hours.

        Returns:
            Dict with keys:
            ``avg_sentiment``, ``total_articles``, ``positive_count``,
            ``negative_count``, ``neutral_count``, ``dominant_classification``.
            Returns all-zero dict when no data or DB error.
        """
        if self._db is None:
            return self._empty_aggregate()

        from sqlalchemy import text as sa_text

        agg_sql = sa_text(
            f"""
            SELECT
                AVG(s.sentiment)                                         AS avg_sentiment,
                COUNT(*)                                                 AS total_articles,
                SUM(CASE WHEN s.classification = 'POSITIVE' THEN 1 ELSE 0 END) AS pos_count,
                SUM(CASE WHEN s.classification = 'NEGATIVE' THEN 1 ELSE 0 END) AS neg_count,
                SUM(CASE WHEN s.classification = 'NEUTRAL'  THEN 1 ELSE 0 END) AS neu_count
            FROM news_articles a
            JOIN news_sentiment_symbol s ON s.article_id = a.id
            WHERE s.symbol = :symbol
              AND a.published_at >= NOW() - INTERVAL '{int(hours)} hours'
            """
        )

        try:
            with self._db.connect() as conn:
                row = conn.execute(agg_sql, {"symbol": symbol}).fetchone()
        except Exception as exc:  # noqa: BLE001
            logger.warning("aggregate_sentiment DB error: %s", exc)
            return self._empty_aggregate()

        if row is None or row[1] == 0:
            return self._empty_aggregate()

        avg_sentiment = float(row[0]) if row[0] is not None else 0.0
        total = int(row[1])
        pos = int(row[2] or 0)
        neg = int(row[3] or 0)
        neu = int(row[4] or 0)

        # Dominant = whichever bucket has most; tie-break: NEUTRAL
        dominant = "NEUTRAL"
        if pos > neg and pos > neu:
            dominant = "POSITIVE"
        elif neg > pos and neg > neu:
            dominant = "NEGATIVE"

        return {
            "avg_sentiment": round(avg_sentiment, 6),
            "total_articles": total,
            "positive_count": pos,
            "negative_count": neg,
            "neutral_count": neu,
            "dominant_classification": dominant,
        }

    # ------------------------------------------------------------------
    # Boolean helpers
    # ------------------------------------------------------------------

    def has_negative_news(
        self,
        symbol: str,
        hours: int = 24,
        min_confidence: float = 0.3,
    ) -> bool:
        """Return True when there is at least one recent high-confidence negative article.

        Args:
            symbol: Stock symbol.
            hours: Lookback window in hours.
            min_confidence: Minimum confidence threshold (0–1).

        Returns:
            True if any NEGATIVE article meets the confidence bar.
        """
        if self._db is None:
            return False

        from sqlalchemy import text as sa_text

        sql = sa_text(
            f"""
            SELECT COUNT(*) FROM news_articles a
            JOIN news_sentiment_symbol s ON s.article_id = a.id
            WHERE s.symbol = :symbol
              AND s.classification = 'NEGATIVE'
              AND s.confidence >= :min_conf
              AND a.published_at >= NOW() - INTERVAL '{int(hours)} hours'
            """
        )

        try:
            with self._db.connect() as conn:
                count = conn.execute(
                    sql, {"symbol": symbol, "min_conf": min_confidence}
                ).scalar()
            return int(count or 0) > 0
        except Exception as exc:  # noqa: BLE001
            logger.warning("has_negative_news DB error: %s", exc)
            return False

    def has_positive_news(
        self,
        symbol: str,
        hours: int = 24,
        min_confidence: float = 0.3,
    ) -> bool:
        """Return True when there is at least one recent high-confidence positive article.

        Args:
            symbol: Stock symbol.
            hours: Lookback window in hours.
            min_confidence: Minimum confidence threshold (0–1).

        Returns:
            True if any POSITIVE article meets the confidence bar.
        """
        if self._db is None:
            return False

        from sqlalchemy import text as sa_text

        sql = sa_text(
            f"""
            SELECT COUNT(*) FROM news_articles a
            JOIN news_sentiment_symbol s ON s.article_id = a.id
            WHERE s.symbol = :symbol
              AND s.classification = 'POSITIVE'
              AND s.confidence >= :min_conf
              AND a.published_at >= NOW() - INTERVAL '{int(hours)} hours'
            """
        )

        try:
            with self._db.connect() as conn:
                count = conn.execute(
                    sql, {"symbol": symbol, "min_conf": min_confidence}
                ).scalar()
            return int(count or 0) > 0
        except Exception as exc:  # noqa: BLE001
            logger.warning("has_positive_news DB error: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _empty_aggregate() -> dict:
        return {
            "avg_sentiment": 0.0,
            "total_articles": 0,
            "positive_count": 0,
            "negative_count": 0,
            "neutral_count": 0,
            "dominant_classification": "NEUTRAL",
        }
