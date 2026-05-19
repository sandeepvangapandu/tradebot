"""Resolve strategy instrument_selection config into Upstox instrument keys.

Given an active strategy config and a spot price, this module resolves
the correct ATM CE/PE option instrument keys to subscribe to.

Used at startup (after first index tick provides spot price) to dynamically
subscribe to tradeable option contracts rather than the static index-only
subscription that previously existed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from src.data.instruments import InstrumentManager


class OptionsResolver:
    """Resolve strategy option instrument requirements into Upstox keys.

    Args:
        instrument_manager: Loaded InstrumentManager instance.
    """

    def __init__(self, instrument_manager: "InstrumentManager") -> None:
        self._im = instrument_manager

    def resolve_strategy_instruments(
        self,
        strategy_config: object,
        spot_price: int,
    ) -> list[str]:
        """Resolve the option instrument keys required by a strategy.

        Reads the strategy's ``instrument_selection`` block and uses the
        current spot price to find the ATM strike on the nearest weekly
        expiry, then returns the CE and/or PE instrument keys.

        Args:
            strategy_config: A StrategyConfig (or dict-like) with attributes
                ``instrument_selection``, ``underlying``, and ``name``.
            spot_price: Current underlying spot price in **paisa**.

        Returns:
            List of fully-qualified Upstox instrument keys (may be empty if
            the strategy doesn't trade options or resolution fails).
        """
        # Support both object-style (StrategyConfig) and dict-style configs
        if hasattr(strategy_config, "instrument_selection"):
            selection = strategy_config.instrument_selection
            underlying_cfg = getattr(strategy_config, "underlying", None) or {}
            name = getattr(strategy_config, "name", "?")
        elif isinstance(strategy_config, dict):
            selection = strategy_config.get("instrument_selection")
            underlying_cfg = strategy_config.get("underlying") or {}
            name = strategy_config.get("name", "?")
        else:
            return []

        if not selection:
            return []

        # InstrumentSelection may be a Pydantic model or a plain dict.
        # Use a helper to access attributes safely in both cases.
        def _sel_get(attr: str, default=None):
            if isinstance(selection, dict):
                return selection.get(attr, default)
            return getattr(selection, attr, default)

        # Only handle options strategies
        sel_type = _sel_get("type")
        if sel_type != "options":
            logger.debug(f"[OptionsResolver] {name}: not an options strategy (type={sel_type!r}), skipping")
            return []

        # --- Underlying ---
        if isinstance(underlying_cfg, dict):
            underlying_key = underlying_cfg.get("instrument_key", "")
            underlying_symbol = underlying_cfg.get("symbol", "")
        else:
            underlying_key = getattr(underlying_cfg, "instrument_key", "")
            underlying_symbol = getattr(underlying_cfg, "symbol", "")

        if not underlying_key or not underlying_symbol:
            logger.warning(f"[OptionsResolver] {name}: missing underlying config")
            return []

        # --- Expiry ---
        expiry_type = _sel_get("expiry_type", "weekly_current")
        if expiry_type in ("weekly_current", "weekly"):
            expiry = self._im.get_weekly_expiry(underlying_symbol)
        else:
            expiry = self._im.get_weekly_expiry(underlying_symbol)

        if not expiry:
            logger.warning(
                f"[OptionsResolver] {name}: no expiry found for {underlying_symbol}"
            )
            return []

        # --- Option chain ---
        chain = self._im.get_option_chain(underlying_key, expiry)
        if chain.empty:
            logger.warning(
                f"[OptionsResolver] {name}: empty option chain for "
                f"{underlying_symbol} expiry={expiry}"
            )
            return []

        # --- ATM strike ---
        try:
            atm_strike_paisa = self._im.get_atm_strike(spot_price, chain)
        except ValueError as exc:
            logger.error(f"[OptionsResolver] {name}: ATM resolution failed — {exc}")
            return []

        atm_strike_rupees = atm_strike_paisa / 100

        # --- Filter to ATM CE/PE ---
        option_types = _sel_get("option_types") or []
        if not option_types:
            # Fall back to singular option_type
            single = _sel_get("option_type")
            option_types = [single] if single else ["CE", "PE"]

        atm_rows = chain[
            (chain["strike_price"] == atm_strike_rupees)
            & (chain["option_type"].isin(option_types))
        ]

        keys = atm_rows["instrument_key"].dropna().tolist()

        logger.info(
            "[OptionsResolver] {} → expiry={} ATM={:.0f} types={} → {} instruments: {}",
            name,
            expiry,
            atm_strike_rupees,
            option_types,
            len(keys),
            keys,
        )
        return keys

    def resolve_all_strategies(
        self,
        strategies: list[object],
        spot_price: int,
    ) -> list[str]:
        """Resolve instruments for all strategies, deduplicating.

        Args:
            strategies: List of StrategyConfig objects.
            spot_price: Current underlying spot price in paisa.

        Returns:
            Deduplicated list of instrument keys to subscribe to.
        """
        seen: set[str] = set()
        result: list[str] = []
        for strategy in strategies:
            keys = self.resolve_strategy_instruments(strategy, spot_price)
            for k in keys:
                if k not in seen:
                    seen.add(k)
                    result.append(k)
        return result
