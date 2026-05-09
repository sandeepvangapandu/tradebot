"""Volume profile condition evaluators for strategy entry/exit rules.

All condition functions query the ``volume_profile_daily`` table (or an
in-memory profile dict) and return simple booleans or categorical strings
that strategy engines can use directly.

All price inputs must be in **paisa** (int).  Dates should be
``datetime.date`` objects or ISO-8601 strings.

Usage example::

    from src.strategy.conditions_volume_profile import spot_above_poc, spot_in_value_area
    from src.storage.db import get_sync_engine

    engine = get_sync_engine()
    trade_date = datetime.date.today()

    if spot_above_poc("NSE_EQ|INE002A01018", spot_paisa=2_450_00, trade_date=trade_date, db_engine=engine):
        print("Spot is above POC — bullish bias")
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _fetch_profile(
    instrument_key: str,
    trade_date: date | str,
    db_engine=None,
) -> dict[str, Any] | None:
    """Fetch the volume profile for a given instrument and date.

    Args:
        instrument_key: Upstox instrument identifier.
        trade_date: Trading date (date object or ISO string).
        db_engine: SQLAlchemy engine.  If ``None`` the function returns
            ``None`` (callers should treat a missing profile as condition
            not met).

    Returns:
        Profile dict or ``None``.
    """
    if db_engine is None:
        logger.debug(
            "_fetch_profile: no db_engine provided for %s %s",
            instrument_key,
            trade_date,
        )
        return None

    from src.indicators.volume_profile import VolumeProfileBuilder

    builder = VolumeProfileBuilder(db_engine=db_engine)
    return builder.get_profile(instrument_key, trade_date)


# ---------------------------------------------------------------------------
# Public condition functions
# ---------------------------------------------------------------------------

def spot_above_poc(
    instrument_key: str,
    spot_paisa: int,
    trade_date: date | str,
    db_engine=None,
) -> bool:
    """Return True if spot price is strictly above the Point of Control.

    Args:
        instrument_key: Upstox instrument identifier.
        spot_paisa: Current spot/LTP price in paisa.
        trade_date: Trading date for profile lookup.
        db_engine: SQLAlchemy engine for DB access.

    Returns:
        ``True`` when spot > POC.  ``False`` on missing profile.
    """
    profile = _fetch_profile(instrument_key, trade_date, db_engine)
    if profile is None:
        return False
    result = spot_paisa > profile["poc_paisa"]
    logger.debug(
        "spot_above_poc %s: spot=%d poc=%d → %s",
        instrument_key,
        spot_paisa,
        profile["poc_paisa"],
        result,
    )
    return result


def spot_below_poc(
    instrument_key: str,
    spot_paisa: int,
    trade_date: date | str,
    db_engine=None,
) -> bool:
    """Return True if spot price is strictly below the Point of Control.

    Args:
        instrument_key: Upstox instrument identifier.
        spot_paisa: Current spot/LTP price in paisa.
        trade_date: Trading date for profile lookup.
        db_engine: SQLAlchemy engine for DB access.

    Returns:
        ``True`` when spot < POC.  ``False`` on missing profile.
    """
    profile = _fetch_profile(instrument_key, trade_date, db_engine)
    if profile is None:
        return False
    result = spot_paisa < profile["poc_paisa"]
    logger.debug(
        "spot_below_poc %s: spot=%d poc=%d → %s",
        instrument_key,
        spot_paisa,
        profile["poc_paisa"],
        result,
    )
    return result


def spot_above_vah(
    instrument_key: str,
    spot_paisa: int,
    trade_date: date | str,
    db_engine=None,
) -> bool:
    """Return True if spot price is strictly above the Value Area High.

    A spot above VAH indicates a breakout above the established value area,
    often used as a bullish breakout signal.

    Args:
        instrument_key: Upstox instrument identifier.
        spot_paisa: Current spot/LTP price in paisa.
        trade_date: Trading date for profile lookup.
        db_engine: SQLAlchemy engine for DB access.

    Returns:
        ``True`` when spot > VAH.  ``False`` on missing profile.
    """
    profile = _fetch_profile(instrument_key, trade_date, db_engine)
    if profile is None:
        return False
    result = spot_paisa > profile["vah_paisa"]
    logger.debug(
        "spot_above_vah %s: spot=%d vah=%d → %s",
        instrument_key,
        spot_paisa,
        profile["vah_paisa"],
        result,
    )
    return result


def spot_below_val(
    instrument_key: str,
    spot_paisa: int,
    trade_date: date | str,
    db_engine=None,
) -> bool:
    """Return True if spot price is strictly below the Value Area Low.

    A spot below VAL indicates a breakdown below the value area, often used
    as a bearish signal or mean-reversion opportunity.

    Args:
        instrument_key: Upstox instrument identifier.
        spot_paisa: Current spot/LTP price in paisa.
        trade_date: Trading date for profile lookup.
        db_engine: SQLAlchemy engine for DB access.

    Returns:
        ``True`` when spot < VAL.  ``False`` on missing profile.
    """
    profile = _fetch_profile(instrument_key, trade_date, db_engine)
    if profile is None:
        return False
    result = spot_paisa < profile["val_paisa"]
    logger.debug(
        "spot_below_val %s: spot=%d val=%d → %s",
        instrument_key,
        spot_paisa,
        profile["val_paisa"],
        result,
    )
    return result


def spot_in_value_area(
    instrument_key: str,
    spot_paisa: int,
    trade_date: date | str,
    db_engine=None,
) -> bool:
    """Return True if spot is within the value area [VAL, VAH] (inclusive).

    Price inside the value area suggests mean-reversion tendencies rather
    than directional breakouts.

    Args:
        instrument_key: Upstox instrument identifier.
        spot_paisa: Current spot/LTP price in paisa.
        trade_date: Trading date for profile lookup.
        db_engine: SQLAlchemy engine for DB access.

    Returns:
        ``True`` when VAL <= spot <= VAH.  ``False`` on missing profile.
    """
    profile = _fetch_profile(instrument_key, trade_date, db_engine)
    if profile is None:
        return False
    result = profile["val_paisa"] <= spot_paisa <= profile["vah_paisa"]
    logger.debug(
        "spot_in_value_area %s: spot=%d val=%d vah=%d → %s",
        instrument_key,
        spot_paisa,
        profile["val_paisa"],
        profile["vah_paisa"],
        result,
    )
    return result


def spot_at_va_extreme(
    instrument_key: str,
    spot_paisa: int,
    trade_date: date | str,
    tolerance_paisa: int = 500,
    db_engine=None,
) -> str | None:
    """Detect if spot is near a Value Area extreme (VAH or VAL).

    Useful for mean-reversion strategies that want to fade price when it
    reaches the edge of the established value area.

    Args:
        instrument_key: Upstox instrument identifier.
        spot_paisa: Current spot/LTP price in paisa.
        trade_date: Trading date for profile lookup.
        tolerance_paisa: Distance in paisa within which spot is considered
            "at" the extreme.  Default 500 paisa (₹5).
        db_engine: SQLAlchemy engine for DB access.

    Returns:
        ``'VAH'`` if spot is within ``tolerance_paisa`` of VAH,
        ``'VAL'`` if within tolerance of VAL,
        ``None`` otherwise (including missing profile).
    """
    profile = _fetch_profile(instrument_key, trade_date, db_engine)
    if profile is None:
        return None

    vah = profile["vah_paisa"]
    val = profile["val_paisa"]

    near_vah = abs(spot_paisa - vah) <= tolerance_paisa
    near_val = abs(spot_paisa - val) <= tolerance_paisa

    if near_vah and near_val:
        # Extremely narrow value area — prefer VAH if spot is >= midpoint
        mid = (vah + val) // 2
        result: str | None = "VAH" if spot_paisa >= mid else "VAL"
    elif near_vah:
        result = "VAH"
    elif near_val:
        result = "VAL"
    else:
        result = None

    logger.debug(
        "spot_at_va_extreme %s: spot=%d vah=%d val=%d tol=%d → %s",
        instrument_key,
        spot_paisa,
        vah,
        val,
        tolerance_paisa,
        result,
    )
    return result
