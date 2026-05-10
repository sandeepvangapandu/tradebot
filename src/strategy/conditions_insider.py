"""Strategy condition helpers — block/bulk deals + insider/promoter trading.

These are thin, boolean-returning wrappers over
:class:`~src.research.insider_signals.InsiderSignals`.

Design principles
-----------------
* Each function accepts an optional ``db_engine`` so it can be unit-tested
  by injecting a mock engine or a real database engine.
* When no engine is provided (or no data is found) the functions return
  ``False`` rather than raising — strategies should degrade gracefully.
* All thresholds and day windows are function parameters with sensible
  defaults; callers can override them in strategy config without touching
  this module.

Signal rationale
----------------
* ``block_buy_recent``       — institutional accumulation (multi-week bullish)
* ``block_sell_recent``      — institutional distribution (bearish)
* ``promoter_buying``        — promoter skin-in-the-game (strong bullish)
* ``promoter_selling``       — promoter exit (bearish warning)
* ``institutional_flow_bullish`` — combined block + bulk net buy > threshold
"""

from __future__ import annotations

import logging
from typing import Any

from src.research.insider_signals import InsiderSignals

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Block deal condition helpers
# ---------------------------------------------------------------------------

def block_buy_recent(
    symbol: str,
    threshold_crore: float = 50.0,
    days: int = 3,
    db_engine: Any = None,
) -> bool:
    """Return True if recent net block/bulk BUY exceeds the threshold.

    Uses ``InsiderSignals.has_significant_block_buy`` under the hood.

    Args:
        symbol: NSE equity symbol (e.g. 'RELIANCE').
        threshold_crore: Minimum net BUY value in ₹ crore to be considered
            significant.  Default 50 crore.
        days: Lookback window in calendar days.  Default 3.
        db_engine: Optional SQLAlchemy engine.

    Returns:
        True when net block/bulk buy value >= ``threshold_crore``.
    """
    try:
        signals = InsiderSignals(db_engine=db_engine)
        return signals.has_significant_block_buy(
            symbol, threshold_crore=threshold_crore, days=days
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("block_buy_recent error for %s: %s", symbol, exc)
        return False


def block_sell_recent(
    symbol: str,
    threshold_crore: float = 50.0,
    days: int = 3,
    db_engine: Any = None,
) -> bool:
    """Return True if recent net block/bulk SELL exceeds the threshold.

    Mirrors ``block_buy_recent`` but checks for net selling pressure.

    Args:
        symbol: NSE equity symbol.
        threshold_crore: Minimum net SELL value in ₹ crore.  Default 50 crore.
        days: Lookback window in calendar days.  Default 3.
        db_engine: Optional SQLAlchemy engine.

    Returns:
        True when net block/bulk sell value >= ``threshold_crore``.
    """
    try:
        signals = InsiderSignals(db_engine=db_engine)
        flow = signals.block_flow_for_symbol(symbol, days=days)
        return flow["sell_value_crore"] - flow["buy_value_crore"] >= threshold_crore
    except Exception as exc:  # noqa: BLE001
        logger.warning("block_sell_recent error for %s: %s", symbol, exc)
        return False


# ---------------------------------------------------------------------------
# Insider / promoter condition helpers
# ---------------------------------------------------------------------------

def promoter_buying(
    symbol: str,
    days: int = 30,
    db_engine: Any = None,
) -> bool:
    """Return True if a promoter BUY was disclosed in the past ``days``.

    Uses ``InsiderSignals.promoter_buying_recent``.

    Args:
        symbol: NSE equity symbol.
        days: Lookback window in calendar days.  Default 30.
        db_engine: Optional SQLAlchemy engine.

    Returns:
        True if any promoter or promoter group BUY is found.
    """
    try:
        signals = InsiderSignals(db_engine=db_engine)
        return signals.promoter_buying_recent(symbol, days=days)
    except Exception as exc:  # noqa: BLE001
        logger.warning("promoter_buying error for %s: %s", symbol, exc)
        return False


def promoter_selling(
    symbol: str,
    days: int = 30,
    db_engine: Any = None,
) -> bool:
    """Return True if a promoter SELL was disclosed in the past ``days``.

    Args:
        symbol: NSE equity symbol.
        days: Lookback window in calendar days.  Default 30.
        db_engine: Optional SQLAlchemy engine.

    Returns:
        True if any promoter or promoter group SELL is found.
    """
    try:
        signals = InsiderSignals(db_engine=db_engine)
        return signals.promoter_selling_recent(symbol, days=days)
    except Exception as exc:  # noqa: BLE001
        logger.warning("promoter_selling error for %s: %s", symbol, exc)
        return False


# ---------------------------------------------------------------------------
# Combined flow condition helper
# ---------------------------------------------------------------------------

def institutional_flow_bullish(
    symbol: str,
    days: int = 5,
    threshold_crore: float = 20.0,
    db_engine: Any = None,
) -> bool:
    """Return True when net block/bulk BUY flow is positive and significant.

    Combines block deals + bulk deals net flow check to confirm
    institutional accumulation.

    Args:
        symbol: NSE equity symbol.
        days: Lookback window in calendar days.  Default 5.
        threshold_crore: Minimum net BUY value in ₹ crore.  Default 20 crore.
        db_engine: Optional SQLAlchemy engine.

    Returns:
        True when net_value_crore >= ``threshold_crore``.
    """
    try:
        signals = InsiderSignals(db_engine=db_engine)
        flow = signals.block_flow_for_symbol(symbol, days=days)
        return flow["net_value_crore"] >= threshold_crore
    except Exception as exc:  # noqa: BLE001
        logger.warning("institutional_flow_bullish error for %s: %s", symbol, exc)
        return False
