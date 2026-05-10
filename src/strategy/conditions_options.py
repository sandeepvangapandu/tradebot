"""Options-derived condition evaluators for strategy entry/exit logic.

These functions query ``options_chain_snapshots`` and
``options_strike_oi_history`` in Postgres to derive boolean conditions
that strategies can use alongside price/indicator conditions.

All monetary thresholds use **paisa** (int) to stay consistent with the
rest of the codebase.  Timestamps are in IST.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

from loguru import logger
from sqlalchemy import text

IST = ZoneInfo("Asia/Kolkata")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _engine(db_engine=None):
    """Return caller-supplied engine or fall back to the singleton."""
    if db_engine is not None:
        return db_engine
    from src.storage.db import get_sync_engine  # lazy import to avoid circular deps
    return get_sync_engine()


def _latest_snapshot(underlying_key: str, engine) -> dict[str, Any] | None:
    """Return the most recent snapshot row for *underlying_key*.

    Args:
        underlying_key: Upstox instrument key.
        engine: SQLAlchemy sync engine.

    Returns:
        Dict of column values or ``None`` if no rows exist.
    """
    sql = text(
        """
        SELECT pcr, total_call_oi, total_put_oi, atm_iv_call, atm_iv_put,
               iv_skew, spot_paisa, max_pain_strike, ts
        FROM options_chain_snapshots
        WHERE underlying_key = :key
        ORDER BY ts DESC
        LIMIT 1
        """
    )
    with engine.connect() as conn:
        row = conn.execute(sql, {"key": underlying_key}).fetchone()
    if row is None:
        return None
    return dict(zip(["pcr", "total_call_oi", "total_put_oi", "atm_iv_call", "atm_iv_put",
                     "iv_skew", "spot_paisa", "max_pain_strike", "ts"], row))


# ---------------------------------------------------------------------------
# PCR conditions
# ---------------------------------------------------------------------------


def pcr_above(
    underlying_key: str,
    threshold: float,
    db_engine=None,
) -> bool:
    """Return True if the most recent PCR exceeds *threshold*.

    A PCR above ~1.2 is generally considered bearish (more puts than calls
    are open), but can also signal capitulation.

    Args:
        underlying_key: Upstox instrument key for the underlying.
        threshold: PCR value to compare against (e.g. ``1.2``).
        db_engine: Optional SQLAlchemy engine; defaults to singleton.

    Returns:
        ``True`` if current PCR > threshold, ``False`` otherwise.
        Returns ``False`` (safe default) if data is unavailable.
    """
    engine = _engine(db_engine)
    try:
        snap = _latest_snapshot(underlying_key, engine)
    except Exception as exc:
        logger.warning("[conditions_options] pcr_above DB error: {}", exc)
        return False

    if snap is None or snap["pcr"] is None:
        return False

    pcr = float(snap["pcr"])
    if math.isnan(pcr) or math.isinf(pcr):
        return False

    return pcr > threshold


def pcr_below(
    underlying_key: str,
    threshold: float,
    db_engine=None,
) -> bool:
    """Return True if the most recent PCR is below *threshold*.

    A PCR below ~0.7 can indicate call-heavy positioning (bullish sentiment
    or potential for short covering on the put side).

    Args:
        underlying_key: Upstox instrument key.
        threshold: PCR value to compare against.
        db_engine: Optional SQLAlchemy engine; defaults to singleton.

    Returns:
        ``True`` if current PCR < threshold, ``False`` if unavailable.
    """
    engine = _engine(db_engine)
    try:
        snap = _latest_snapshot(underlying_key, engine)
    except Exception as exc:
        logger.warning("[conditions_options] pcr_below DB error: {}", exc)
        return False

    if snap is None or snap["pcr"] is None:
        return False

    pcr = float(snap["pcr"])
    if math.isnan(pcr) or math.isinf(pcr):
        return False

    return pcr < threshold


# ---------------------------------------------------------------------------
# IV percentile
# ---------------------------------------------------------------------------


def iv_percentile(
    underlying_key: str,
    lookback_days: int = 60,
    db_engine=None,
) -> float:
    """Compute the IV percentile of current ATM IV over the lookback window.

    Uses ``atm_iv_call`` from ``options_chain_snapshots`` as the IV proxy.

    Args:
        underlying_key: Upstox instrument key.
        lookback_days: Number of calendar days to look back (default 60).
        db_engine: Optional SQLAlchemy engine; defaults to singleton.

    Returns:
        IV percentile in [0, 100].  Returns NaN if insufficient data.
    """
    engine = _engine(db_engine)
    cutoff = datetime.now(IST) - timedelta(days=lookback_days)

    sql = text(
        """
        SELECT atm_iv_call, ts
        FROM options_chain_snapshots
        WHERE underlying_key = :key
          AND ts >= :cutoff
          AND atm_iv_call IS NOT NULL
        ORDER BY ts ASC
        """
    )
    try:
        with engine.connect() as conn:
            rows = conn.execute(sql, {"key": underlying_key, "cutoff": cutoff}).fetchall()
    except Exception as exc:
        logger.warning("[conditions_options] iv_percentile DB error: {}", exc)
        return float("nan")

    if len(rows) < 2:
        return float("nan")

    ivs = [float(r[0]) for r in rows]
    current_iv = ivs[-1]
    min_iv = min(ivs)
    max_iv = max(ivs)

    if max_iv == min_iv:
        return 50.0

    pct = (current_iv - min_iv) / (max_iv - min_iv) * 100.0
    return round(pct, 2)


# ---------------------------------------------------------------------------
# Max OI strike proximity
# ---------------------------------------------------------------------------


def max_oi_strike_near_spot(
    underlying_key: str,
    spot_paisa: int,
    tolerance_pct: float = 0.5,
    db_engine=None,
) -> bool:
    """Return True if the max-pain strike is within *tolerance_pct* of spot.

    When max-pain is very close to spot, option writers' pin risk is high
    and price tends to gravitate toward that level into expiry.

    Args:
        underlying_key: Upstox instrument key.
        spot_paisa: Current spot price in paisa.
        tolerance_pct: Allowed deviation (%) between max-pain and spot.
        db_engine: Optional SQLAlchemy engine; defaults to singleton.

    Returns:
        ``True`` if abs(max_pain_strike - spot_rupees) / spot_rupees * 100
        <= tolerance_pct.
    """
    engine = _engine(db_engine)
    try:
        snap = _latest_snapshot(underlying_key, engine)
    except Exception as exc:
        logger.warning("[conditions_options] max_oi_strike_near_spot DB error: {}", exc)
        return False

    if snap is None or snap["max_pain_strike"] is None:
        return False

    spot_rupees = spot_paisa / 100.0
    max_pain = float(snap["max_pain_strike"])

    if spot_rupees == 0:
        return False

    deviation_pct = abs(max_pain - spot_rupees) / spot_rupees * 100.0
    return deviation_pct <= tolerance_pct


# ---------------------------------------------------------------------------
# Call writers dominance
# ---------------------------------------------------------------------------


def call_writers_dominant(
    underlying_key: str,
    expiry: str,
    lookback_minutes: int = 30,
    db_engine=None,
) -> bool:
    """Return True if net call OI build exceeds net put OI build recently.

    Compares the sum of ``oi_change`` for CE strikes vs PE strikes over the
    last *lookback_minutes* in ``options_strike_oi_history``.  A positive
    net call OI build (more calls being written than puts) is bearish for
    the underlying.

    Args:
        underlying_key: Upstox instrument key.
        expiry: Expiry date string ``YYYY-MM-DD``.
        lookback_minutes: How many minutes of OI-change history to aggregate.
        db_engine: Optional SQLAlchemy engine; defaults to singleton.

    Returns:
        ``True`` if net call OI build > net put OI build.
    """
    engine = _engine(db_engine)
    cutoff = datetime.now(IST) - timedelta(minutes=lookback_minutes)

    sql = text(
        """
        SELECT option_type, COALESCE(SUM(oi_change), 0) AS net_oi_build
        FROM options_strike_oi_history
        WHERE underlying_key = :key
          AND expiry = :expiry
          AND ts >= :cutoff
          AND oi_change IS NOT NULL
        GROUP BY option_type
        """
    )
    try:
        with engine.connect() as conn:
            rows = conn.execute(
                sql, {"key": underlying_key, "expiry": expiry, "cutoff": cutoff}
            ).fetchall()
    except Exception as exc:
        logger.warning("[conditions_options] call_writers_dominant DB error: {}", exc)
        return False

    net_by_type: dict[str, float] = {r[0]: float(r[1]) for r in rows}
    net_call_build = net_by_type.get("CE", 0.0)
    net_put_build = net_by_type.get("PE", 0.0)

    return net_call_build > net_put_build
