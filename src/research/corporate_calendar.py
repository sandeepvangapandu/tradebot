"""Corporate calendar — query corporate actions and apply blackout filter.

Provides a read-only query layer over the ``corporate_actions`` table.
Used by strategy condition helpers to suppress trading around ex-dates.

Typical usage::

    cal = CorporateCalendar(db_engine=engine)
    blackout, reason = cal.is_blackout("RELIANCE", date.today())
    if blackout:
        logger.info("Skip trade: {}", reason)
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

from loguru import logger


class CorporateCalendar:
    """Query corporate-action events and provide blackout windows.

    Args:
        db_engine: SQLAlchemy engine connected to the local DB.
            If ``None`` all methods return empty / False safely.
    """

    def __init__(self, db_engine: Any = None) -> None:
        self.db_engine = db_engine

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_actions_for_symbol(
        self,
        symbol: str,
        from_date: date | None = None,
        to_date: date | None = None,
    ) -> list[dict]:
        """Return corporate actions for *symbol* filtered by date range.

        Args:
            symbol: NSE trading symbol.
            from_date: Inclusive start date (``None`` = no lower bound).
            to_date: Inclusive end date (``None`` = no upper bound).

        Returns:
            List of dicts with keys:
            ``id``, ``symbol``, ``action_type``, ``ex_date``,
            ``record_date``, ``details``, ``ratio``, ``source``.
        """
        if self.db_engine is None:
            return []

        try:
            from sqlalchemy import text
        except ImportError:  # pragma: no cover
            return []

        clauses = ["symbol = :symbol"]
        params: dict[str, Any] = {"symbol": symbol.upper()}

        if from_date is not None:
            clauses.append("ex_date >= :from_date")
            params["from_date"] = from_date.isoformat()

        if to_date is not None:
            clauses.append("ex_date <= :to_date")
            params["to_date"] = to_date.isoformat()

        where = " AND ".join(clauses)
        sql = text(
            f"SELECT id, symbol, action_type, ex_date, record_date, details, ratio, source "
            f"FROM corporate_actions WHERE {where} ORDER BY ex_date"
        )

        try:
            with self.db_engine.connect() as conn:
                rows = conn.execute(sql, params).fetchall()
                columns = ["id", "symbol", "action_type", "ex_date",
                           "record_date", "details", "ratio", "source"]
                return [dict(zip(columns, row)) for row in rows]
        except Exception as exc:
            logger.warning("CorporateCalendar.get_actions_for_symbol error: {}", exc)
            return []

    def is_blackout(
        self,
        symbol: str,
        trade_date: date,
        days_before: int = 1,
        days_after: int = 1,
    ) -> tuple[bool, str | None]:
        """Check whether *trade_date* falls within a blackout window.

        A blackout window is defined as the period from
        ``ex_date - days_before`` to ``ex_date + days_after`` (inclusive)
        around any upcoming corporate action.

        Args:
            symbol: NSE trading symbol.
            trade_date: The date to check.
            days_before: Days before ex_date to include in blackout.
            days_after: Days after ex_date to include in blackout.

        Returns:
            Tuple ``(is_blackout, reason)`` where *reason* is a human-readable
            string like ``"DIVIDEND ex-date 2026-05-10"`` or ``None``.
        """
        if self.db_engine is None:
            return (False, None)

        window_start = trade_date - timedelta(days=days_after)
        window_end = trade_date + timedelta(days=days_before)

        actions = self.get_actions_for_symbol(
            symbol,
            from_date=window_start,
            to_date=window_end,
        )

        for action in actions:
            ex_date = action["ex_date"]
            if isinstance(ex_date, str):
                from datetime import datetime
                ex_date = datetime.strptime(ex_date, "%Y-%m-%d").date()

            window_low = ex_date - timedelta(days=days_before)
            window_high = ex_date + timedelta(days=days_after)

            if window_low <= trade_date <= window_high:
                reason = f"{action['action_type']} ex-date {ex_date}"
                return (True, reason)

        return (False, None)

    def upcoming_actions(
        self,
        symbol: str,
        days_ahead: int = 7,
    ) -> list[dict]:
        """Return actions for *symbol* whose ex_date is within *days_ahead*.

        Args:
            symbol: NSE trading symbol.
            days_ahead: Look-ahead window in calendar days.

        Returns:
            List of action dicts ordered by ex_date ascending.
        """
        today = date.today()
        return self.get_actions_for_symbol(
            symbol,
            from_date=today,
            to_date=today + timedelta(days=days_ahead),
        )

    def has_imminent_action(
        self,
        symbol: str,
        trade_date: date,
        action_types: list[str] | None = None,
        days: int = 3,
    ) -> bool:
        """Return True if an action of a given type is imminent.

        Args:
            symbol: NSE trading symbol.
            trade_date: Reference date (typically today).
            action_types: Action types to check e.g. ``["DIVIDEND", "SPLIT"]``.
                ``None`` means check all types.
            days: Number of days ahead to look.

        Returns:
            ``True`` if any matching action has ex_date within *days* of
            *trade_date*, ``False`` otherwise.
        """
        if self.db_engine is None:
            return False

        actions = self.get_actions_for_symbol(
            symbol,
            from_date=trade_date,
            to_date=trade_date + timedelta(days=days),
        )

        for action in actions:
            if action_types is None or action["action_type"] in action_types:
                return True

        return False
