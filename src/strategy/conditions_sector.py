"""Sector rotation condition evaluators for strategy entry/exit rules.

Wraps :class:`~src.research.sector_rotation.SectorRotationAnalyzer` into
callable boolean conditions that strategy configs can reference.

All functions query the ``sector_rank_daily`` table for the most recent
available trade date so they can be called without an explicit date argument.

Usage example::

    from src.strategy.conditions_sector import sector_in_top_quartile
    from src.storage.db import get_sync_engine

    engine = get_sync_engine()
    if sector_in_top_quartile("HDFCBANK", db_engine=engine):
        print("HDFCBANK's sector (NIFTY_BANK) is a top-quartile sector today")
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Any

from sqlalchemy import text

logger = logging.getLogger(__name__)

# Mapping re-exported from sector_rotation so callers can import here.
from src.research.sector_rotation import _STOCK_TO_SECTOR


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _latest_rank_row(
    sector_symbol: str,
    db_engine,
) -> dict[str, Any] | None:
    """Fetch the most recent rank row for *sector_symbol*.

    Args:
        sector_symbol: Sector index symbol (e.g. ``'NIFTY_BANK'``).
        db_engine: SQLAlchemy sync engine.

    Returns:
        Dict with keys trade_date, rs_score, rank, rank_change or ``None``
        when no data is found.
    """
    if db_engine is None:
        return None
    sql = text(
        """
        SELECT trade_date, rs_score, rank, rank_change
        FROM sector_rank_daily
        WHERE sector_symbol = :sym
        ORDER BY trade_date DESC
        LIMIT 1
        """
    )
    try:
        with db_engine.connect() as conn:
            row = conn.execute(sql, {"sym": sector_symbol}).fetchone()
        if row is None:
            return None
        return {
            "trade_date": row[0],
            "rs_score": float(row[1]),
            "rank": int(row[2]),
            "rank_change": int(row[3]) if row[3] is not None else None,
        }
    except Exception as exc:
        logger.warning("conditions_sector: DB query failed for %s: %s", sector_symbol, exc)
        return None


def _count_active_sectors(db_engine) -> int:
    """Return the count of active sectors in the sector_indices table.

    Args:
        db_engine: SQLAlchemy sync engine.

    Returns:
        Number of active sectors (int). Returns 9 as fallback on error.
    """
    if db_engine is None:
        return 9
    sql = text("SELECT COUNT(*) FROM sector_indices WHERE active = TRUE")
    try:
        with db_engine.connect() as conn:
            result = conn.execute(sql).scalar()
        return int(result) if result else 9
    except Exception:
        return 9


def _resolve_sector(stock_or_sector: str) -> str:
    """Resolve a stock symbol to its sector or return as-is if already a sector.

    Args:
        stock_or_sector: Either a stock symbol (``'HDFC'``) or sector symbol
            (``'NIFTY_BANK'``).

    Returns:
        Sector symbol string (possibly unchanged if already a sector key).
    """
    upper = stock_or_sector.upper()
    # If it's a known stock, map it to sector
    if upper in _STOCK_TO_SECTOR:
        return _STOCK_TO_SECTOR[upper]
    # Otherwise assume it's already a sector symbol
    return upper


# ---------------------------------------------------------------------------
# Public condition helpers
# ---------------------------------------------------------------------------

def sector_in_top_quartile(
    stock_symbol: str,
    db_engine=None,
) -> bool:
    """Return True when the stock's sector is in the top quartile by RS rank.

    Resolves *stock_symbol* to its sector via the hard-coded mapping, then
    checks whether the sector's most recent rank is in the top 25% (i.e.
    rank <= ceil(n_sectors / 4)).

    Args:
        stock_symbol: NSE trading symbol (e.g. ``'HDFCBANK'``).
        db_engine: SQLAlchemy sync engine. When ``None`` the function fetches
            the default engine from ``src.storage.db``.

    Returns:
        ``True`` when the sector is top quartile. ``False`` on any error or
        if no ranking data is available.
    """
    if db_engine is None:
        try:
            from src.storage.db import get_sync_engine
            db_engine = get_sync_engine()
        except Exception:
            return False

    sector_symbol = _resolve_sector(stock_symbol)
    row = _latest_rank_row(sector_symbol, db_engine)
    if row is None:
        return False

    n = _count_active_sectors(db_engine)
    cutoff = max(1, -(-n // 4))  # ceiling division: ceil(n/4)
    return row["rank"] <= cutoff


def sector_in_bottom_quartile(
    stock_symbol: str,
    db_engine=None,
) -> bool:
    """Return True when the stock's sector is in the bottom quartile by RS rank.

    Bottom quartile means rank >= n - floor(n/4) + 1, i.e. the weakest 25%
    of sectors.

    Args:
        stock_symbol: NSE trading symbol (e.g. ``'RELIANCE'``).
        db_engine: SQLAlchemy sync engine.

    Returns:
        ``True`` when the sector is bottom quartile.
    """
    if db_engine is None:
        try:
            from src.storage.db import get_sync_engine
            db_engine = get_sync_engine()
        except Exception:
            return False

    sector_symbol = _resolve_sector(stock_symbol)
    row = _latest_rank_row(sector_symbol, db_engine)
    if row is None:
        return False

    n = _count_active_sectors(db_engine)
    # Bottom quartile: worst ceil(n/4) ranks
    bottom_cutoff = n - max(1, -(-n // 4)) + 1
    return row["rank"] >= bottom_cutoff


def sector_rs_above(
    sector_symbol: str,
    threshold: float,
    db_engine=None,
) -> bool:
    """Return True when the sector's RS score is above *threshold*.

    A positive RS score means the sector outperformed NIFTY 50 over the
    lookback period.

    Args:
        sector_symbol: Sector index symbol (e.g. ``'NIFTY_BANK'``).
        threshold: RS score threshold (e.g. ``0.5`` for 0.5 pp outperformance).
        db_engine: SQLAlchemy sync engine.

    Returns:
        ``True`` when RS score > threshold.
    """
    if db_engine is None:
        try:
            from src.storage.db import get_sync_engine
            db_engine = get_sync_engine()
        except Exception:
            return False

    row = _latest_rank_row(sector_symbol, db_engine)
    if row is None:
        return False
    return row["rs_score"] > threshold


def sector_rank_change_positive(
    sector_symbol: str,
    db_engine=None,
) -> bool:
    """Return True when the sector's rank improved (moved up) versus yesterday.

    A positive rank_change value means the sector rose in rank (e.g. moved
    from rank 5 to rank 3, so rank_change = +2).

    Args:
        sector_symbol: Sector index symbol (e.g. ``'NIFTY_IT'``).
        db_engine: SQLAlchemy sync engine.

    Returns:
        ``True`` when rank_change > 0.  ``False`` when unchanged, worse, or
        no prior-day data exists.
    """
    if db_engine is None:
        try:
            from src.storage.db import get_sync_engine
            db_engine = get_sync_engine()
        except Exception:
            return False

    row = _latest_rank_row(sector_symbol, db_engine)
    if row is None or row["rank_change"] is None:
        return False
    return row["rank_change"] > 0
