"""Strategy condition helpers — news sentiment filter.

Thin boolean-returning wrappers over :class:`~src.research.news_query.NewsQuery`.
Strategies can gate trades on fresh adverse (or favourable) news events.

Design principles
-----------------
* Each function accepts an optional ``db_engine`` so it is trivially
  unit-testable by injecting a mock.
* Returns ``False`` when no engine / no data — strategies degrade gracefully.
* All thresholds are function parameters; callers override in strategy config.
* No state — pure functions.
"""

from __future__ import annotations

import logging

from src.research.news_query import NewsQuery

logger = logging.getLogger(__name__)


def has_negative_news_recent(
    symbol: str,
    hours: int = 24,
    db_engine=None,
) -> bool:
    """Return True when *symbol* has at least one recent negative-sentiment article.

    Uses the default ``min_confidence=0.3`` threshold from
    :meth:`~src.research.news_query.NewsQuery.has_negative_news`.

    Args:
        symbol: Stock symbol, e.g. ``"RELIANCE"``.
        hours: Lookback window in hours.
        db_engine: SQLAlchemy engine.

    Returns:
        True if adverse news found within the window.
    """
    if db_engine is None:
        return False
    try:

        return NewsQuery(db_engine=db_engine).has_negative_news(symbol, hours=hours)
    except Exception as exc:  # noqa: BLE001
        logger.warning("has_negative_news_recent error for %s: %s", symbol, exc)
        return False


def has_positive_news_recent(
    symbol: str,
    hours: int = 24,
    db_engine=None,
) -> bool:
    """Return True when *symbol* has at least one recent positive-sentiment article.

    Args:
        symbol: Stock symbol.
        hours: Lookback window in hours.
        db_engine: SQLAlchemy engine.

    Returns:
        True if positive news found within the window.
    """
    if db_engine is None:
        return False
    try:

        return NewsQuery(db_engine=db_engine).has_positive_news(symbol, hours=hours)
    except Exception as exc:  # noqa: BLE001
        logger.warning("has_positive_news_recent error for %s: %s", symbol, exc)
        return False


def aggregate_sentiment_above(
    symbol: str,
    threshold: float,
    hours: int = 24,
    db_engine=None,
) -> bool:
    """Return True when the average news sentiment for *symbol* exceeds *threshold*.

    Useful for confirming bullish bias before entering a long trade.

    Args:
        symbol: Stock symbol.
        threshold: Minimum average sentiment score (e.g. ``0.2``).
        hours: Lookback window in hours.
        db_engine: SQLAlchemy engine.

    Returns:
        True when avg_sentiment > threshold.
    """
    if db_engine is None:
        return False
    try:

        agg = NewsQuery(db_engine=db_engine).aggregate_sentiment(symbol, hours=hours)
        return agg["avg_sentiment"] > threshold
    except Exception as exc:  # noqa: BLE001
        logger.warning("aggregate_sentiment_above error for %s: %s", symbol, exc)
        return False


def aggregate_sentiment_below(
    symbol: str,
    threshold: float,
    hours: int = 24,
    db_engine=None,
) -> bool:
    """Return True when the average news sentiment for *symbol* is below *threshold*.

    Useful for filtering out trades when sentiment is bearish.

    Args:
        symbol: Stock symbol.
        threshold: Maximum average sentiment score (e.g. ``-0.2``).
        hours: Lookback window in hours.
        db_engine: SQLAlchemy engine.

    Returns:
        True when avg_sentiment < threshold.
    """
    if db_engine is None:
        return False
    try:

        agg = NewsQuery(db_engine=db_engine).aggregate_sentiment(symbol, hours=hours)
        return agg["avg_sentiment"] < threshold
    except Exception as exc:  # noqa: BLE001
        logger.warning("aggregate_sentiment_below error for %s: %s", symbol, exc)
        return False


def news_volume_spike(
    symbol: str,
    lookback_hours: int = 4,
    baseline_hours: int = 168,
    multiplier: float = 3.0,
    db_engine=None,
) -> bool:
    """Return True when recent news volume for *symbol* is a spike vs baseline.

    A spike is detected when the article count in the last *lookback_hours*
    (annualised to an hourly rate) is at least *multiplier* × the baseline
    hourly rate over the last *baseline_hours*.

    Args:
        symbol: Stock symbol.
        lookback_hours: Recent window for spike detection (default 4 h).
        baseline_hours: Baseline window for normal rate (default 168 h = 1 week).
        multiplier: Spike threshold multiplier (default 3×).
        db_engine: SQLAlchemy engine.

    Returns:
        True when recent hourly rate >= multiplier × baseline hourly rate.
    """
    if db_engine is None:
        return False

    from sqlalchemy import text as sa_text

    recent_sql = sa_text(
        f"""
        SELECT COUNT(*) FROM news_articles a
        JOIN news_sentiment_symbol s ON s.article_id = a.id
        WHERE s.symbol = :symbol
          AND a.published_at >= NOW() - INTERVAL '{int(lookback_hours)} hours'
        """
    )
    baseline_sql = sa_text(
        f"""
        SELECT COUNT(*) FROM news_articles a
        JOIN news_sentiment_symbol s ON s.article_id = a.id
        WHERE s.symbol = :symbol
          AND a.published_at >= NOW() - INTERVAL '{int(baseline_hours)} hours'
        """
    )

    try:
        with db_engine.connect() as conn:
            recent_count = int(
                conn.execute(recent_sql, {"symbol": symbol}).scalar() or 0
            )
            baseline_count = int(
                conn.execute(baseline_sql, {"symbol": symbol}).scalar() or 0
            )
    except Exception as exc:  # noqa: BLE001
        logger.warning("news_volume_spike DB error for %s: %s", symbol, exc)
        return False

    if baseline_count == 0:
        # No baseline data — cannot assert a spike
        return False

    recent_rate = recent_count / max(lookback_hours, 1)
    baseline_rate = baseline_count / max(baseline_hours, 1)

    return recent_rate >= multiplier * baseline_rate
