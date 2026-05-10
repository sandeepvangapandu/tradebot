"""Corporate-action condition helpers for strategy entry/exit filters.

These thin helpers wrap :class:`~src.research.corporate_calendar.CorporateCalendar`
and are designed to be used as boolean guards inside strategy ``on_candle``
or condition evaluation pipelines.

Usage example::

    from src.strategy.conditions_corporate import in_corporate_blackout

    if in_corporate_blackout("RELIANCE", db_engine=engine):
        return  # skip entry
"""

from __future__ import annotations

from datetime import date
from typing import Any

from loguru import logger

from src.research.corporate_calendar import CorporateCalendar


def in_corporate_blackout(
    symbol: str,
    days_before: int = 1,
    days_after: int = 1,
    db_engine: Any = None,
) -> bool:
    """Return ``True`` if today is within a corporate-action blackout window.

    A blackout is triggered when any corporate action has an ex_date within
    ``days_before`` days before or ``days_after`` days after today.

    Args:
        symbol: NSE trading symbol (e.g. ``"RELIANCE"``).
        days_before: Days before ex_date to block.
        days_after: Days after ex_date to block.
        db_engine: SQLAlchemy engine. If ``None`` returns ``False`` (safe default).

    Returns:
        ``True`` if the symbol is in blackout, ``False`` otherwise.
    """
    cal = CorporateCalendar(db_engine=db_engine)
    blackout, reason = cal.is_blackout(
        symbol,
        trade_date=date.today(),
        days_before=days_before,
        days_after=days_after,
    )
    if blackout:
        logger.info(
            "conditions_corporate.in_corporate_blackout: {} blocked — {}", symbol, reason
        )
    return blackout


def has_dividend_in(
    symbol: str,
    days: int = 7,
    db_engine: Any = None,
) -> bool:
    """Return ``True`` if a dividend ex-date is due within *days* days.

    Args:
        symbol: NSE trading symbol.
        days: Look-ahead window in calendar days.
        db_engine: SQLAlchemy engine.

    Returns:
        ``True`` if a DIVIDEND action is imminent, ``False`` otherwise.
    """
    cal = CorporateCalendar(db_engine=db_engine)
    return cal.has_imminent_action(
        symbol,
        trade_date=date.today(),
        action_types=["DIVIDEND"],
        days=days,
    )


def has_split_in(
    symbol: str,
    days: int = 30,
    db_engine: Any = None,
) -> bool:
    """Return ``True`` if a stock split ex-date is due within *days* days.

    Args:
        symbol: NSE trading symbol.
        days: Look-ahead window in calendar days.
        db_engine: SQLAlchemy engine.

    Returns:
        ``True`` if a SPLIT action is imminent, ``False`` otherwise.
    """
    cal = CorporateCalendar(db_engine=db_engine)
    return cal.has_imminent_action(
        symbol,
        trade_date=date.today(),
        action_types=["SPLIT"],
        days=days,
    )


def has_bonus_in(
    symbol: str,
    days: int = 30,
    db_engine: Any = None,
) -> bool:
    """Return ``True`` if a bonus issue ex-date is due within *days* days.

    Args:
        symbol: NSE trading symbol.
        days: Look-ahead window in calendar days.
        db_engine: SQLAlchemy engine.

    Returns:
        ``True`` if a BONUS action is imminent, ``False`` otherwise.
    """
    cal = CorporateCalendar(db_engine=db_engine)
    return cal.has_imminent_action(
        symbol,
        trade_date=date.today(),
        action_types=["BONUS"],
        days=days,
    )
