"""VIX-based condition helpers for strategy gates.

These helpers expose India VIX regime state as boolean/float conditions that
strategy configs and signal pipelines can reference.  Each function reads the
latest persisted regime data from ``vix_regime_daily`` (or ``vix_regime_intraday``
for intraday spike detection).

All functions accept an optional ``db_engine`` parameter.  When ``None``, the
function returns a safe default (``False`` for booleans, ``0.0`` for floats)
so that strategies do not crash in environments without a DB connection.

Example usage in a strategy gate::

    from src.strategy.conditions_vix import vix_in_regime, vix_below

    if not vix_in_regime("NORMAL") and not vix_in_regime("HIGH"):
        return  # unfavourable volatility environment
    if vix_below(20.0):
        size_up_position()
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timezone

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_today_regime_row(db_engine) -> dict | None:
    """Fetch today's vix_regime_daily row.  Returns None on any failure."""
    if db_engine is None:
        return None

    from sqlalchemy import text

    sql = text(
        """
        SELECT trade_date, vix_close, regime, percentile_60d
        FROM vix_regime_daily
        WHERE trade_date = CURRENT_DATE
        """
    )
    try:
        with db_engine.connect() as conn:
            row = conn.execute(sql).fetchone()
        if row is None:
            return None
        return {
            "trade_date": row[0],
            "vix_close": float(row[1]),
            "regime": str(row[2]),
            "percentile_60d": float(row[3]) if row[3] is not None else None,
        }
    except Exception:  # noqa: BLE001
        logger.warning("conditions_vix: could not fetch today's regime row")
        return None


def _has_intraday_spike_today(db_engine) -> bool:
    """Return True if any intraday row for today has spike_detected=True."""
    if db_engine is None:
        return False

    from sqlalchemy import text

    sql = text(
        """
        SELECT 1
        FROM vix_regime_intraday
        WHERE ts::date = CURRENT_DATE
          AND spike_detected = TRUE
        LIMIT 1
        """
    )
    try:
        with db_engine.connect() as conn:
            row = conn.execute(sql).fetchone()
        return row is not None
    except Exception:  # noqa: BLE001
        logger.warning("conditions_vix: could not query vix_regime_intraday")
        return False


# ---------------------------------------------------------------------------
# Public condition functions
# ---------------------------------------------------------------------------

def vix_in_regime(regime: str, db_engine=None) -> bool:
    """Return True when today's VIX regime matches *regime*.

    Args:
        regime: One of ``'LOW'``, ``'NORMAL'``, ``'HIGH'``, ``'SPIKE'``.
        db_engine: Optional SQLAlchemy sync engine.

    Returns:
        ``True`` when the latest daily regime equals *regime*.
        ``False`` when no data is available or regime differs.
    """
    row = _get_today_regime_row(db_engine)
    if row is None:
        return False
    return row["regime"] == regime


def vix_below(threshold: float, db_engine=None) -> bool:
    """Return True when today's VIX close is strictly below *threshold*.

    Args:
        threshold: VIX level to compare against.
        db_engine: Optional SQLAlchemy sync engine.

    Returns:
        ``True`` when ``vix_close < threshold``.
        ``False`` when no data is available.
    """
    row = _get_today_regime_row(db_engine)
    if row is None:
        return False
    return row["vix_close"] < threshold


def vix_above(threshold: float, db_engine=None) -> bool:
    """Return True when today's VIX close is strictly above *threshold*.

    Args:
        threshold: VIX level to compare against.
        db_engine: Optional SQLAlchemy sync engine.

    Returns:
        ``True`` when ``vix_close > threshold``.
        ``False`` when no data is available.
    """
    row = _get_today_regime_row(db_engine)
    if row is None:
        return False
    return row["vix_close"] > threshold


def vix_spike_detected_today(db_engine=None) -> bool:
    """Return True if an intraday VIX spike (>30% move) was detected today.

    Reads from ``vix_regime_intraday`` where ``spike_detected = TRUE``.

    Args:
        db_engine: Optional SQLAlchemy sync engine.

    Returns:
        ``True`` when at least one spike row exists for today.
    """
    return _has_intraday_spike_today(db_engine)


def vix_percentile_above(percentile: float, db_engine=None) -> bool:
    """Return True when today's VIX 60d-percentile exceeds *percentile*.

    Args:
        percentile: Percentile threshold in ``[0, 100]``.
        db_engine: Optional SQLAlchemy sync engine.

    Returns:
        ``True`` when ``percentile_60d > percentile``.
        ``False`` when no data is available or percentile is not stored.
    """
    row = _get_today_regime_row(db_engine)
    if row is None:
        return False
    p60d = row.get("percentile_60d")
    if p60d is None:
        return False
    return p60d > percentile
