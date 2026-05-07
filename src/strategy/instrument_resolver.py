"""Resolve option instrument_keys from a strategy config + current spot price."""
from __future__ import annotations
from typing import Any
from loguru import logger

from src.strategy.builder import StrategyConfig
from src.data.instruments import InstrumentManager

# Strike intervals (rupees) per Indian index
STRIKE_INTERVALS = {
    "BANKNIFTY": 100,
    "NIFTY": 50,
    "FINNIFTY": 50,
    "MIDCPNIFTY": 25,
    "SENSEX": 100,
    "BANKEX": 100,
}


def _get_attr(obj: Any, key: str, default: Any = None) -> Any:
    """Read attribute from either dict or pydantic model."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def resolve_options_instruments(
    config: StrategyConfig,
    spot_price_paisa: int,
    instrument_manager: InstrumentManager,
) -> list[str]:
    """Resolve list of option instrument_keys for an options-type strategy.

    Returns empty list if not options-type, missing data, or resolution fails.
    Logs at INFO on success, WARNING on each failure cause.
    """
    sel = config.instrument_selection
    sel_type = _get_attr(sel, "type")
    if sel_type != "options":
        return []

    underlying = config.underlying or {}
    underlying_symbol = _get_attr(underlying, "symbol")
    underlying_key = _get_attr(underlying, "instrument_key")
    if not underlying_symbol or not underlying_key:
        logger.warning(
            "Strategy {} missing underlying.symbol or underlying.instrument_key — skipping",
            config.name,
        )
        return []

    expiry_type = _get_attr(sel, "expiry_type", "weekly_current")
    if expiry_type != "weekly_current":
        logger.warning(
            "Strategy {} expiry_type={} not yet supported (only weekly_current) — using nearest weekly",
            config.name, expiry_type,
        )

    expiry = instrument_manager.get_weekly_expiry(underlying_symbol)
    if not expiry:
        logger.warning(
            "No weekly expiry found for {} — option chain unavailable",
            underlying_symbol,
        )
        return []

    chain = instrument_manager.get_option_chain(underlying_key, expiry)
    if chain.empty:
        logger.warning(
            "Empty option chain for {} @ {} — verify instrument master",
            underlying_key, expiry,
        )
        return []

    interval = STRIKE_INTERVALS.get(underlying_symbol.upper(), 100)
    spot_rupees = spot_price_paisa / 100.0
    atm_rupees = round(spot_rupees / interval) * interval

    strike_sel = (_get_attr(sel, "strike_selection") or "atm")
    target_strike_rupees = atm_rupees
    if isinstance(strike_sel, str):
        s = strike_sel.lower().strip()
        if s.startswith("atm+"):
            try:
                target_strike_rupees = atm_rupees + int(s[4:]) * interval
            except ValueError:
                logger.warning("Bad strike_selection '{}' — defaulting to ATM", strike_sel)
        elif s.startswith("atm-"):
            try:
                target_strike_rupees = atm_rupees - int(s[4:]) * interval
            except ValueError:
                logger.warning("Bad strike_selection '{}' — defaulting to ATM", strike_sel)
        elif s != "atm":
            logger.warning("Unknown strike_selection '{}' — defaulting to ATM", strike_sel)

    option_types = _get_attr(sel, "option_types") or ["CE", "PE"]

    chain_filtered = chain[
        (chain["strike_price"] == target_strike_rupees)
        & (chain["option_type"].isin(option_types))
    ]

    if chain_filtered.empty:
        logger.warning(
            "No options matched strike={} types={} for {} @ {} — closest strikes: {}",
            target_strike_rupees, option_types, underlying_key, expiry,
            sorted(chain["strike_price"].unique())[:10],
        )
        return []

    keys = chain_filtered["instrument_key"].tolist()
    logger.info(
        "Resolved {} option contracts for strategy {} (underlying={}, spot={:.2f}, ATM={}, strike={}, expiry={}): {}",
        len(keys), config.name, underlying_symbol, spot_rupees, atm_rupees,
        target_strike_rupees, expiry, keys,
    )
    return keys
