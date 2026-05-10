"""Query earnings calendar and apply blackout filter.

This module provides the ``EarningsCalendar`` class which wraps all
DB queries against the ``earnings_calendar`` table and exposes
clean helper methods consumed by strategy condition helpers.

Blackout window
---------------
By default the blackout window is T-2 to T+1 (2 trading days before and 1
trading day after the earnings date).  The trade date is compared against
the earnings_date column using simple calendar days (not trading days) to
keep the logic dependency-free.

Design notes
------------
- Returns safe defaults (False / None) when the DB is unavailable.
- All IST date comparisons use ``datetime.date`` objects directly.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta, timezone
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ist_today() -> date:
    """Return today's date in IST (UTC+5:30)."""
    ist = timezone(timedelta(hours=5, minutes=30))
    return datetime.now(ist).date()


def _to_date(d: Any) -> date | None:
    """Coerce a date/str/datetime to ``date`` or return None."""
    if d is None:
        return None
    if isinstance(d, datetime):
        return d.date()
    if isinstance(d, date):
        return d
    for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y"):
        try:
            return datetime.strptime(str(d), fmt).date()
        except ValueError:
            continue
    return None


# ---------------------------------------------------------------------------
# EarningsCalendar
# ---------------------------------------------------------------------------

class EarningsCalendar:
    """Query interface for the earnings_calendar table.

    Args:
        db_engine: Optional SQLAlchemy engine.  When ``None`` all methods
            return empty/safe defaults (useful in unit tests without a DB).
    """

    def __init__(self, db_engine=None) -> None:
        self._engine = db_engine

    # ------------------------------------------------------------------
    # Core query
    # ------------------------------------------------------------------

    def get_earnings(
        self,
        symbol: str,
        from_date: date | str | None = None,
        to_date: date | str | None = None,
    ) -> list[dict]:
        """Return earnings records for a symbol optionally filtered by date.

        Args:
            symbol: NSE trading symbol.
            from_date: Start of date range (inclusive).  Defaults to no lower
                bound.
            to_date: End of date range (inclusive).  Defaults to no upper bound.

        Returns:
            List of dicts with keys: symbol, earnings_date, fiscal_quarter,
            reporting_time, expected_eps, actual_eps, surprise_pct, source.
        """
        if self._engine is None:
            return []

        from sqlalchemy import text as sa_text

        conditions = ["symbol = :symbol"]
        params: dict = {"symbol": symbol}

        fd = _to_date(from_date)
        td = _to_date(to_date)

        if fd is not None:
            conditions.append("earnings_date >= :from_date")
            params["from_date"] = fd
        if td is not None:
            conditions.append("earnings_date <= :to_date")
            params["to_date"] = td

        where = " AND ".join(conditions)
        sql = sa_text(
            f"""
            SELECT symbol, earnings_date, fiscal_quarter, reporting_time,
                   expected_eps, actual_eps, surprise_pct, source
            FROM earnings_calendar
            WHERE {where}
            ORDER BY earnings_date DESC
            """
        )

        try:
            with self._engine.connect() as conn:
                rows = conn.execute(sql, params).fetchall()
        except Exception as exc:  # noqa: BLE001
            logger.warning("get_earnings DB error for %s: %s", symbol, exc)
            return []

        return [
            {
                "symbol": row[0],
                "earnings_date": row[1],
                "fiscal_quarter": row[2],
                "reporting_time": row[3],
                "expected_eps": float(row[4]) if row[4] is not None else None,
                "actual_eps": float(row[5]) if row[5] is not None else None,
                "surprise_pct": float(row[6]) if row[6] is not None else None,
                "source": row[7],
            }
            for row in rows
        ]

    # ------------------------------------------------------------------
    # Blackout filter
    # ------------------------------------------------------------------

    def is_blackout(
        self,
        symbol: str,
        trade_date: date | str | None = None,
        days_before: int = 2,
        days_after: int = 1,
    ) -> tuple[bool, str | None]:
        """Check whether a trade date falls within the earnings blackout window.

        The blackout window covers [earnings_date - days_before, earnings_date
        + days_after] (inclusive, calendar days).

        Args:
            symbol: NSE trading symbol.
            trade_date: The date to check.  Defaults to IST today.
            days_before: Number of calendar days before earnings to block.
            days_after: Number of calendar days after earnings to block.

        Returns:
            A 2-tuple ``(is_blocked, reason_str)``.  ``reason_str`` is a
            human-readable message when blocked, or None when not blocked.
        """
        if self._engine is None:
            return (False, None)

        check_date = _to_date(trade_date) or _ist_today()

        window_start = check_date - timedelta(days=days_before)
        window_end = check_date + timedelta(days=days_after)

        from sqlalchemy import text as sa_text

        sql = sa_text(
            """
            SELECT earnings_date, fiscal_quarter
            FROM earnings_calendar
            WHERE symbol = :symbol
              AND earnings_date BETWEEN :window_start AND :window_end
            ORDER BY earnings_date
            LIMIT 1
            """
        )

        try:
            with self._engine.connect() as conn:
                row = conn.execute(
                    sql,
                    {
                        "symbol": symbol,
                        "window_start": window_start,
                        "window_end": window_end,
                    },
                ).fetchone()
        except Exception as exc:  # noqa: BLE001
            logger.warning("is_blackout DB error for %s: %s", symbol, exc)
            return (False, None)

        if row is None:
            return (False, None)

        earnings_date = row[0]
        fiscal_quarter = row[1]
        qualifier = f" ({fiscal_quarter})" if fiscal_quarter else ""
        reason = f"Earnings on {earnings_date}{qualifier}"
        return (True, reason)

    # ------------------------------------------------------------------
    # Upcoming earnings
    # ------------------------------------------------------------------

    def upcoming_earnings(
        self, symbol: str, days_ahead: int = 7
    ) -> dict | None:
        """Return the nearest upcoming earnings record within ``days_ahead`` days.

        Args:
            symbol: NSE trading symbol.
            days_ahead: How many calendar days forward to look.

        Returns:
            Earnings dict or None if no upcoming earnings found.
        """
        if self._engine is None:
            return None

        today = _ist_today()
        to_date = today + timedelta(days=days_ahead)

        from sqlalchemy import text as sa_text

        sql = sa_text(
            """
            SELECT symbol, earnings_date, fiscal_quarter, reporting_time,
                   expected_eps, actual_eps, surprise_pct, source
            FROM earnings_calendar
            WHERE symbol = :symbol
              AND earnings_date BETWEEN :today AND :to_date
            ORDER BY earnings_date ASC
            LIMIT 1
            """
        )

        try:
            with self._engine.connect() as conn:
                row = conn.execute(
                    sql, {"symbol": symbol, "today": today, "to_date": to_date}
                ).fetchone()
        except Exception as exc:  # noqa: BLE001
            logger.warning("upcoming_earnings DB error for %s: %s", symbol, exc)
            return None

        if row is None:
            return None

        return {
            "symbol": row[0],
            "earnings_date": row[1],
            "fiscal_quarter": row[2],
            "reporting_time": row[3],
            "expected_eps": float(row[4]) if row[4] is not None else None,
            "actual_eps": float(row[5]) if row[5] is not None else None,
            "surprise_pct": float(row[6]) if row[6] is not None else None,
            "source": row[7],
        }

    # ------------------------------------------------------------------
    # Last surprise
    # ------------------------------------------------------------------

    def last_surprise(self, symbol: str) -> dict | None:
        """Return the most recent earnings surprise record for a symbol.

        Only considers past earnings where actual_eps is known.

        Args:
            symbol: NSE trading symbol.

        Returns:
            Dict with keys: date, surprise_pct, expected_eps, actual_eps.
            Returns None if no past results with known actuals are found.
        """
        if self._engine is None:
            return None

        today = _ist_today()

        from sqlalchemy import text as sa_text

        sql = sa_text(
            """
            SELECT earnings_date, surprise_pct, expected_eps, actual_eps
            FROM earnings_calendar
            WHERE symbol = :symbol
              AND earnings_date <= :today
              AND actual_eps IS NOT NULL
            ORDER BY earnings_date DESC
            LIMIT 1
            """
        )

        try:
            with self._engine.connect() as conn:
                row = conn.execute(
                    sql, {"symbol": symbol, "today": today}
                ).fetchone()
        except Exception as exc:  # noqa: BLE001
            logger.warning("last_surprise DB error for %s: %s", symbol, exc)
            return None

        if row is None:
            return None

        return {
            "date": row[0],
            "surprise_pct": float(row[1]) if row[1] is not None else None,
            "expected_eps": float(row[2]) if row[2] is not None else None,
            "actual_eps": float(row[3]) if row[3] is not None else None,
        }

    # ------------------------------------------------------------------
    # Positive surprise recent
    # ------------------------------------------------------------------

    def positive_surprise_recent(
        self,
        symbol: str,
        threshold_pct: float = 5.0,
        days: int = 30,
    ) -> bool:
        """Return True if the symbol had a positive earnings surprise recently.

        Args:
            symbol: NSE trading symbol.
            threshold_pct: Minimum surprise percentage to qualify as positive.
            days: Look-back window in calendar days.

        Returns:
            True if there is a recent earnings row with surprise_pct >=
            threshold_pct, False otherwise.
        """
        if self._engine is None:
            return False

        today = _ist_today()
        from_date = today - timedelta(days=days)

        from sqlalchemy import text as sa_text

        sql = sa_text(
            """
            SELECT 1
            FROM earnings_calendar
            WHERE symbol = :symbol
              AND earnings_date BETWEEN :from_date AND :today
              AND surprise_pct >= :threshold
            LIMIT 1
            """
        )

        try:
            with self._engine.connect() as conn:
                row = conn.execute(
                    sql,
                    {
                        "symbol": symbol,
                        "from_date": from_date,
                        "today": today,
                        "threshold": threshold_pct,
                    },
                ).fetchone()
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "positive_surprise_recent DB error for %s: %s", symbol, exc
            )
            return False

        return row is not None
