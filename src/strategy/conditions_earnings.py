"""Strategy condition helpers — earnings calendar blackout filter.

These are thin, boolean-returning wrappers over the ``EarningsCalendar``
query interface.  They follow the same design contract as all other
``conditions_*.py`` modules:

* Each function accepts an optional ``db_engine`` for easy unit-testing.
* When no engine is provided or no data is found the functions return
  ``False`` (safe default — fail open, not closed).
* All thresholds are function parameters with sensible defaults.
* IST-aware: "today" is always resolved in IST (Asia/Kolkata).

Typical strategy usage::

    from src.strategy.conditions_earnings import in_earnings_blackout, earnings_within

    if in_earnings_blackout(symbol, db_engine=engine):
        logger.info("Skipping %s — earnings blackout window", symbol)
        return SignalAction.HOLD
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public condition helpers
# ---------------------------------------------------------------------------

def in_earnings_blackout(
    symbol: str,
    days_before: int = 2,
    days_after: int = 1,
    db_engine=None,
) -> bool:
    """Return True if today falls within the earnings blackout window for symbol.

    The blackout window spans [earnings_date - days_before, earnings_date +
    days_after] (inclusive, calendar days).

    Args:
        symbol: NSE trading symbol (e.g. "RELIANCE").
        days_before: Calendar days before earnings to begin blackout.
        days_after: Calendar days after earnings to end blackout.
        db_engine: Optional SQLAlchemy engine.  Returns False when None.

    Returns:
        True if today is within the blackout window, False otherwise.
    """
    if db_engine is None:
        return False

    from src.research.earnings_calendar import EarningsCalendar

    try:
        cal = EarningsCalendar(db_engine=db_engine)
        blocked, reason = cal.is_blackout(
            symbol=symbol,
            trade_date=None,  # defaults to IST today
            days_before=days_before,
            days_after=days_after,
        )
        if blocked:
            logger.debug("Blackout for %s: %s", symbol, reason)
        return blocked
    except Exception as exc:  # noqa: BLE001
        logger.warning("in_earnings_blackout error for %s: %s", symbol, exc)
        return False


def earnings_within(
    symbol: str,
    days: int = 3,
    db_engine=None,
) -> bool:
    """Return True if the symbol has upcoming earnings within ``days`` calendar days.

    Args:
        symbol: NSE trading symbol.
        days: Number of calendar days to look ahead.
        db_engine: Optional SQLAlchemy engine.  Returns False when None.

    Returns:
        True if there is a known upcoming earnings date within the window.
    """
    if db_engine is None:
        return False

    from src.research.earnings_calendar import EarningsCalendar

    try:
        cal = EarningsCalendar(db_engine=db_engine)
        result = cal.upcoming_earnings(symbol=symbol, days_ahead=days)
        return result is not None
    except Exception as exc:  # noqa: BLE001
        logger.warning("earnings_within error for %s: %s", symbol, exc)
        return False


def positive_earnings_surprise(
    symbol: str,
    threshold_pct: float = 5.0,
    days: int = 30,
    db_engine=None,
) -> bool:
    """Return True if the symbol had a positive earnings surprise recently.

    A "positive surprise" means actual EPS exceeded expected EPS by at least
    ``threshold_pct`` percent in the past ``days`` calendar days.

    Args:
        symbol: NSE trading symbol.
        threshold_pct: Minimum surprise percentage (e.g. 5.0 = 5%).
        days: Look-back window in calendar days.
        db_engine: Optional SQLAlchemy engine.  Returns False when None.

    Returns:
        True if a qualifying positive surprise is found, False otherwise.
    """
    if db_engine is None:
        return False

    from src.research.earnings_calendar import EarningsCalendar

    try:
        cal = EarningsCalendar(db_engine=db_engine)
        return cal.positive_surprise_recent(
            symbol=symbol,
            threshold_pct=threshold_pct,
            days=days,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("positive_earnings_surprise error for %s: %s", symbol, exc)
        return False


def negative_earnings_surprise(
    symbol: str,
    threshold_pct: float = -5.0,
    days: int = 30,
    db_engine=None,
) -> bool:
    """Return True if the symbol had a negative earnings surprise recently.

    A "negative surprise" means actual EPS missed expected EPS by at least
    abs(``threshold_pct``) percent in the past ``days`` calendar days.

    Args:
        symbol: NSE trading symbol.
        threshold_pct: Maximum surprise percentage — should be negative
            (e.g. -5.0 = miss by 5%).
        days: Look-back window in calendar days.
        db_engine: Optional SQLAlchemy engine.  Returns False when None.

    Returns:
        True if a qualifying negative surprise is found, False otherwise.
    """
    if db_engine is None:
        return False

    from src.research.earnings_calendar import EarningsCalendar

    try:
        cal = EarningsCalendar(db_engine=db_engine)
        today = _ist_today()
        from datetime import timedelta
        from_date = today - timedelta(days=days)

        from sqlalchemy import text as sa_text

        sql = sa_text(
            """
            SELECT 1
            FROM earnings_calendar
            WHERE symbol = :symbol
              AND earnings_date BETWEEN :from_date AND :today
              AND surprise_pct <= :threshold
            LIMIT 1
            """
        )

        try:
            with db_engine.connect() as conn:
                row = conn.execute(
                    sql,
                    {
                        "symbol": symbol,
                        "from_date": from_date,
                        "today": today,
                        "threshold": threshold_pct,
                    },
                ).fetchone()
            return row is not None
        except Exception as inner_exc:  # noqa: BLE001
            logger.warning(
                "negative_earnings_surprise DB error for %s: %s", symbol, inner_exc
            )
            return False
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "negative_earnings_surprise error for %s: %s", symbol, exc
        )
        return False


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ist_today():
    """Return today's date in IST."""
    from datetime import datetime, timezone, timedelta
    ist = timezone(timedelta(hours=5, minutes=30))
    return datetime.now(ist).date()
