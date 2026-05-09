"""Strategy condition helpers — FII/DII flows and bond yield.

These are thin, boolean-returning wrappers over the data in
``fii_dii_flows_daily``, ``bond_yield_daily``, and ``flow_regime_daily``.

Design principles
-----------------
* Each function accepts an optional ``db_engine`` so it can be unit-tested
  by injecting a mock engine (or a real SQLite/Postgres engine).
* When no engine is provided (or no data is found for today) the functions
  return ``False`` rather than raising — strategies should degrade gracefully.
* All thresholds are function parameters with sensible defaults; callers can
  override them in strategy config without touching this module.
* IST-aware: "today" is always resolved in IST (Asia/Kolkata).
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
    """Return today's date in IST."""
    ist = timezone(timedelta(hours=5, minutes=30))
    return datetime.now(ist).date()


def _fetch_latest_flow(db_engine) -> dict | None:
    """Fetch the most recent row from fii_dii_flows_daily.

    Args:
        db_engine: SQLAlchemy engine.

    Returns:
        Row as a dict or None.
    """
    from sqlalchemy import text as sa_text

    try:
        with db_engine.connect() as conn:
            row = conn.execute(
                sa_text(
                    "SELECT trade_date, fii_net_value_crore, dii_net_value_crore "
                    "FROM fii_dii_flows_daily ORDER BY trade_date DESC LIMIT 1"
                )
            ).fetchone()
    except Exception as exc:  # noqa: BLE001
        logger.warning("_fetch_latest_flow DB error: %s", exc)
        return None

    if row is None:
        return None
    return {
        "trade_date": row[0],
        "fii_net": float(row[1]) if row[1] is not None else None,
        "dii_net": float(row[2]) if row[2] is not None else None,
    }


def _fetch_latest_regime(db_engine) -> dict | None:
    """Fetch the most recent row from flow_regime_daily.

    Args:
        db_engine: SQLAlchemy engine.

    Returns:
        Row as a dict or None.
    """
    from sqlalchemy import text as sa_text

    try:
        with db_engine.connect() as conn:
            row = conn.execute(
                sa_text(
                    "SELECT trade_date, fii_streak, dii_streak, "
                    "fii_regime, dii_regime, combined_signal "
                    "FROM flow_regime_daily ORDER BY trade_date DESC LIMIT 1"
                )
            ).fetchone()
    except Exception as exc:  # noqa: BLE001
        logger.warning("_fetch_latest_regime DB error: %s", exc)
        return None

    if row is None:
        return None
    return {
        "trade_date": row[0],
        "fii_streak": row[1],
        "dii_streak": row[2],
        "fii_regime": row[3],
        "dii_regime": row[4],
        "combined_signal": row[5],
    }


def _fetch_latest_yield(db_engine) -> dict | None:
    """Fetch the most recent row from bond_yield_daily.

    Args:
        db_engine: SQLAlchemy engine.

    Returns:
        Row as a dict or None.
    """
    from sqlalchemy import text as sa_text

    try:
        with db_engine.connect() as conn:
            row = conn.execute(
                sa_text(
                    "SELECT trade_date, yield_10y_pct, trend_5d "
                    "FROM bond_yield_daily ORDER BY trade_date DESC LIMIT 1"
                )
            ).fetchone()
    except Exception as exc:  # noqa: BLE001
        logger.warning("_fetch_latest_yield DB error: %s", exc)
        return None

    if row is None:
        return None
    return {
        "trade_date": row[0],
        "yield_10y_pct": float(row[1]) if row[1] is not None else None,
        "trend_5d": row[2],
    }


# ---------------------------------------------------------------------------
# FII conditions
# ---------------------------------------------------------------------------

def fii_buying(min_net_crore: float = 0, db_engine=None) -> bool:
    """Return True when FII net flow >= *min_net_crore*.

    Args:
        min_net_crore: Minimum FII net inflow threshold in ₹ crore.
            Default 0 means any positive net qualifies.
        db_engine: SQLAlchemy engine.  Returns False when None.

    Returns:
        True if the latest FII net flow meets the threshold.
    """
    if db_engine is None:
        return False

    flow = _fetch_latest_flow(db_engine)
    if flow is None or flow.get("fii_net") is None:
        return False

    return flow["fii_net"] >= min_net_crore


def fii_selling_streak(min_days: int = 3, db_engine=None) -> bool:
    """Return True when FII has been net sellers for at least *min_days* in a row.

    Uses the ``fii_streak`` column from ``flow_regime_daily`` (negative value
    means consecutive selling days).

    Args:
        min_days: Minimum consecutive selling days required.
        db_engine: SQLAlchemy engine.

    Returns:
        True when FII selling streak >= min_days.
    """
    if db_engine is None:
        return False

    regime = _fetch_latest_regime(db_engine)
    if regime is None or regime.get("fii_streak") is None:
        return False

    # Streak is negative for selling days; abs value is the count.
    streak = regime["fii_streak"]
    return streak <= -min_days


# ---------------------------------------------------------------------------
# DII conditions
# ---------------------------------------------------------------------------

def dii_buying(min_net_crore: float = 0, db_engine=None) -> bool:
    """Return True when DII net flow >= *min_net_crore*.

    Args:
        min_net_crore: Minimum DII net inflow threshold in ₹ crore.
        db_engine: SQLAlchemy engine.

    Returns:
        True if the latest DII net flow meets the threshold.
    """
    if db_engine is None:
        return False

    flow = _fetch_latest_flow(db_engine)
    if flow is None or flow.get("dii_net") is None:
        return False

    return flow["dii_net"] >= min_net_crore


# ---------------------------------------------------------------------------
# Combined regime conditions
# ---------------------------------------------------------------------------

def flow_regime_tailwind(db_engine=None) -> bool:
    """Return True when the combined flow signal is TAILWIND.

    TAILWIND means both FII and DII are net buyers (BUY or STRONG_BUY).

    Args:
        db_engine: SQLAlchemy engine.

    Returns:
        True when combined_signal == 'TAILWIND'.
    """
    if db_engine is None:
        return False

    regime = _fetch_latest_regime(db_engine)
    if regime is None:
        return False

    return regime.get("combined_signal") == "TAILWIND"


def flow_regime_headwind(db_engine=None) -> bool:
    """Return True when the combined flow signal is HEADWIND.

    HEADWIND means both FII and DII are net sellers (SELL or STRONG_SELL).

    Args:
        db_engine: SQLAlchemy engine.

    Returns:
        True when combined_signal == 'HEADWIND'.
    """
    if db_engine is None:
        return False

    regime = _fetch_latest_regime(db_engine)
    if regime is None:
        return False

    return regime.get("combined_signal") == "HEADWIND"


# ---------------------------------------------------------------------------
# Bond yield conditions
# ---------------------------------------------------------------------------

def yield_rising(db_engine=None) -> bool:
    """Return True when the 5-day bond yield trend is RISING.

    A rising 10Y G-Sec yield is generally a headwind for equities (higher
    discount rate) and options premium sellers.

    Args:
        db_engine: SQLAlchemy engine.

    Returns:
        True when trend_5d == 'RISING'.
    """
    if db_engine is None:
        return False

    yield_data = _fetch_latest_yield(db_engine)
    if yield_data is None:
        return False

    return yield_data.get("trend_5d") == "RISING"


def yield_falling(db_engine=None) -> bool:
    """Return True when the 5-day bond yield trend is FALLING.

    A falling 10Y G-Sec yield is generally a tailwind for equities.

    Args:
        db_engine: SQLAlchemy engine.

    Returns:
        True when trend_5d == 'FALLING'.
    """
    if db_engine is None:
        return False

    yield_data = _fetch_latest_yield(db_engine)
    if yield_data is None:
        return False

    return yield_data.get("trend_5d") == "FALLING"
