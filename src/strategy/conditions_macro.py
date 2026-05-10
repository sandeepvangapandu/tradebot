"""Macro overlay condition helpers for strategy entry/exit rules.

Exposes simple boolean condition functions that wrap :class:`~src.research.macro_overlay.MacroOverlay`
to let strategy configs reference macro regime state without direct DB coupling.

All functions default to ``False`` (condition not met) when the DB is unavailable,
data is missing, or the regime has not yet been computed.

Supported macro symbols:  ``"USDINR"``, ``"CRUDE"``, ``"GOLD"``, ``"SILVER"``

Usage example::

    from src.strategy.conditions_macro import macro_trend_up, cross_asset_bullish
    from src.storage.db import get_sync_engine

    engine = get_sync_engine()

    if macro_trend_up("USDINR", db_engine=engine):
        print("USDINR is in an uptrend — IT exporters may benefit")

    if cross_asset_bullish("TCS", db_engine=engine):
        print("Cross-asset signal: TCS bullish given USDINR uptrend")
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal helper: fetch latest trend for a macro symbol
# ---------------------------------------------------------------------------

def _get_macro_trend(symbol: str, db_engine: Any = None) -> str:
    """Return the latest trend classification for a macro symbol.

    Queries ``macro_regime_daily`` for the most recent row for *symbol*.

    Args:
        symbol: One of ``'USDINR'``, ``'CRUDE'``, ``'GOLD'``, ``'SILVER'``.
        db_engine: SQLAlchemy engine.

    Returns:
        One of ``'UP'``, ``'DOWN'``, ``'RANGE'``, or ``'UNKNOWN'`` when data
        is unavailable.
    """
    if db_engine is None:
        return "UNKNOWN"

    try:
        from sqlalchemy import text

        sql = text("""
            SELECT trend
            FROM macro_regime_daily
            WHERE symbol = :symbol
            ORDER BY trade_date DESC
            LIMIT 1
        """)
        with db_engine.connect() as conn:
            row = conn.execute(sql, {"symbol": symbol.upper()}).fetchone()
        if row is None:
            return "UNKNOWN"
        return str(row[0]) if row[0] else "UNKNOWN"
    except Exception as exc:
        logger.error("_get_macro_trend failed for %s: %s", symbol, exc)
        return "UNKNOWN"


def _get_macro_zscore(symbol: str, db_engine: Any = None) -> float | None:
    """Return the latest 20-day z-score for a macro symbol.

    Args:
        symbol: Macro symbol (e.g. ``'USDINR'``).
        db_engine: SQLAlchemy engine.

    Returns:
        Z-score float, or ``None`` if unavailable.
    """
    if db_engine is None:
        return None

    try:
        from sqlalchemy import text

        sql = text("""
            SELECT zscore_20d
            FROM macro_regime_daily
            WHERE symbol = :symbol
            ORDER BY trade_date DESC
            LIMIT 1
        """)
        with db_engine.connect() as conn:
            row = conn.execute(sql, {"symbol": symbol.upper()}).fetchone()
        if row is None or row[0] is None:
            return None
        return float(row[0])
    except Exception as exc:
        logger.error("_get_macro_zscore failed for %s: %s", symbol, exc)
        return None


# ---------------------------------------------------------------------------
# Public condition functions
# ---------------------------------------------------------------------------

def macro_trend_up(symbol: str, db_engine: Any = None) -> bool:
    """Return True when the latest macro regime for *symbol* is UP.

    Args:
        symbol: Macro symbol, e.g. ``'USDINR'``, ``'CRUDE'``, ``'GOLD'``, ``'SILVER'``.
        db_engine: SQLAlchemy engine (optional).  Returns ``False`` when ``None``.

    Returns:
        ``True`` if the most-recent trend is ``'UP'``, ``False`` otherwise.
    """
    return _get_macro_trend(symbol, db_engine) == "UP"


def macro_trend_down(symbol: str, db_engine: Any = None) -> bool:
    """Return True when the latest macro regime for *symbol* is DOWN.

    Args:
        symbol: Macro symbol.
        db_engine: SQLAlchemy engine (optional).

    Returns:
        ``True`` if the most-recent trend is ``'DOWN'``, ``False`` otherwise.
    """
    return _get_macro_trend(symbol, db_engine) == "DOWN"


def macro_zscore_above(
    symbol: str,
    threshold: float,
    db_engine: Any = None,
) -> bool:
    """Return True when the latest 20d z-score for *symbol* exceeds *threshold*.

    Useful for detecting strongly trending macro instruments.

    Args:
        symbol: Macro symbol (e.g. ``'CRUDE'``).
        threshold: Z-score level to compare against (e.g. 1.5 for +1.5 std devs).
        db_engine: SQLAlchemy engine (optional).

    Returns:
        ``True`` when z-score > threshold, ``False`` otherwise (including when
        z-score data is unavailable).
    """
    zscore = _get_macro_zscore(symbol, db_engine)
    if zscore is None:
        return False
    return zscore > threshold


def cross_asset_bullish(stock_symbol: str, db_engine: Any = None) -> bool:
    """Return True when the macro cross-asset signal for a stock is bullish.

    Checks the hard-coded impact mapping in :class:`~src.research.macro_overlay.MacroOverlay`
    against live regime data from the DB.

    Args:
        stock_symbol: NSE equity symbol (e.g. ``'TCS'``, ``'RELIANCE'``).
        db_engine: SQLAlchemy engine (optional).

    Returns:
        ``True`` when cross-asset direction is ``'bullish'``, ``False`` otherwise.
    """
    from src.research.macro_overlay import MacroOverlay

    overlay = MacroOverlay(db_engine=db_engine)
    signal = overlay.cross_asset_signal_for_stock(stock_symbol, trade_date=date.today())
    return signal.get("direction") == "bullish"


def cross_asset_bearish(stock_symbol: str, db_engine: Any = None) -> bool:
    """Return True when the macro cross-asset signal for a stock is bearish.

    Args:
        stock_symbol: NSE equity symbol (e.g. ``'HDFCBANK'``).
        db_engine: SQLAlchemy engine (optional).

    Returns:
        ``True`` when cross-asset direction is ``'bearish'``, ``False`` otherwise.
    """
    from src.research.macro_overlay import MacroOverlay

    overlay = MacroOverlay(db_engine=db_engine)
    signal = overlay.cross_asset_signal_for_stock(stock_symbol, trade_date=date.today())
    return signal.get("direction") == "bearish"
