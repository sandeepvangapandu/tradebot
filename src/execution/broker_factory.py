"""Broker factory for creating paper or live broker instances.

Supports three broker backends:
- PaperBroker: Simulated order execution (paper trading, default).
- DhanBroker: Live trading via Dhan brokerage (recommended).
- UpstoxLiveBroker: Live trading via Upstox brokerage (legacy).

Select the backend via ACTIVE_BROKER env var ("dhan" | "upstox").
"""

from __future__ import annotations

import os
from typing import Any

from loguru import logger

from src.execution.base_broker import BaseBroker
from src.execution.paper_broker import PaperBroker


def create_broker(
    mode: str | None = None,
    config: dict[str, Any] | None = None,
    require_confirmation: bool = True,
) -> BaseBroker:
    """Factory function to create appropriate broker instance.

    Args:
        mode: 'paper' or 'live'. If None, reads from TRADING_MODE env var.
        config: Optional configuration dictionary.
        require_confirmation: If True, requires explicit confirmation for live mode.

    Returns:
        BaseBroker instance (PaperBroker, DhanBroker, or UpstoxLiveBroker).

    Raises:
        ValueError: If invalid mode specified.
        RuntimeError: If live mode confirmation not provided.
    """
    if mode is None:
        mode = os.getenv("TRADING_MODE", "paper").lower()

    config = config or {}

    if mode == "paper":
        logger.info("Creating PaperBroker for paper trading")
        return _create_paper_broker(config)

    elif mode == "live":
        return _create_live_broker(config, require_confirmation)

    else:
        raise ValueError(f"Invalid trading mode: {mode}. Use 'paper' or 'live'.")


def _create_paper_broker(config: dict[str, Any]) -> PaperBroker:
    """Create a paper trading broker instance."""
    initial_capital = config.get("initial_capital", 1_000_000_00)  # 10L paisa
    slippage_pct = config.get("slippage_pct")
    if slippage_pct is None:
        return PaperBroker(initial_capital=initial_capital)
    return PaperBroker(initial_capital=initial_capital, slippage_pct=float(slippage_pct))


def _create_live_broker(
    config: dict[str, Any],
    require_confirmation: bool,
) -> BaseBroker:
    """Create a live trading broker instance.

    Reads ACTIVE_BROKER env var to select between 'upstox' (default) and 'dhan'.
    """
    # Safety check — MUST be first.
    # Tier 3.10 patch (2026-05-15) — three gates required:
    #   1. TRADING_MODE=live
    #   2. LIVE_TRADING_CONFIRMED=true
    #   3. LIVE_MAX_CAPITAL_PAISA set AND <= configured capital
    # Gate 3 ensures a typo on settings.capital can't accidentally route
    # the full ₹1cr through live broker — operator must explicitly opt-in
    # to a deployment ceiling.
    if require_confirmation:
        live_confirmed = os.getenv("LIVE_TRADING_CONFIRMED", "false").lower()
        if live_confirmed != "true":
            raise RuntimeError(
                "LIVE TRADING SAFETY CHECK FAILED — missing LIVE_TRADING_CONFIRMED.\n\n"
                "To enable live trading, you must:\n"
                "1. Set TRADING_MODE=live in your environment\n"
                "2. Set LIVE_TRADING_CONFIRMED=true\n"
                "3. Set LIVE_MAX_CAPITAL_PAISA to an explicit deployment cap\n"
                "4. Verify your API credentials are correct\n\n"
                "Paper trading is strongly recommended for testing."
            )

        live_max = os.getenv("LIVE_MAX_CAPITAL_PAISA", "").strip()
        if not live_max:
            raise RuntimeError(
                "LIVE TRADING SAFETY CHECK FAILED — missing LIVE_MAX_CAPITAL_PAISA.\n"
                "Set LIVE_MAX_CAPITAL_PAISA=<int> to an explicit per-session "
                "deployment cap (in paisa). Required even if TRADING_MODE=live "
                "and LIVE_TRADING_CONFIRMED=true. Prevents accidental full-capital "
                "exposure from a typo on settings.capital."
            )
        try:
            live_max_int = int(live_max)
        except ValueError:
            raise RuntimeError(f"LIVE_MAX_CAPITAL_PAISA must be an integer paisa value, got: {live_max}")

        configured_cap = int(config.get("capital", 0) or 0)
        if configured_cap > 0 and live_max_int > configured_cap:
            raise RuntimeError(
                f"LIVE TRADING SAFETY CHECK FAILED — LIVE_MAX_CAPITAL_PAISA ({live_max_int}) "
                f"exceeds settings.capital ({configured_cap}). The deployment cap must "
                f"be <= configured capital."
            )
        logger.critical(
            "LIVE MODE confirmed — TRADING_MODE=live, LIVE_TRADING_CONFIRMED=true, "
            "LIVE_MAX_CAPITAL_PAISA=%d (cap <= settings.capital)",
            live_max_int,
        )

    logger.warning("🚨 CREATING LIVE BROKER - REAL ORDERS WILL BE PLACED!")

    active_broker = (config.get("active_broker") or os.getenv("ACTIVE_BROKER", "upstox")).lower()

    if active_broker == "dhan":
        return _create_dhan_broker(config)
    else:
        return _create_upstox_broker(config)


def _create_dhan_broker(config: dict[str, Any]) -> BaseBroker:
    """Create a DhanBroker live instance using env credentials."""
    from src.execution.dhan_broker import DhanBroker

    client_id = config.get("dhan_client_id") or os.getenv("DHAN_CLIENT_ID", "")
    access_token = config.get("dhan_access_token") or os.getenv("DHAN_ACCESS_TOKEN", "")

    if not client_id or not access_token:
        raise ValueError(
            "Dhan credentials missing. Set DHAN_CLIENT_ID and DHAN_ACCESS_TOKEN in .env"
        )

    logger.info("Creating DhanBroker | client_id={}", client_id)
    return DhanBroker(client_id=client_id, access_token=access_token)


def _create_upstox_broker(config: dict[str, Any]) -> BaseBroker:
    """Create an UpstoxLiveBroker instance (legacy)."""
    from src.execution.upstox_live import UpstoxLiveBroker
    from src.auth.token_manager import TokenManager

    token_manager = config.get("token_manager") or TokenManager()
    return UpstoxLiveBroker(token_manager=token_manager)


class BrokerFactory:
    """Factory class for creating broker instances."""

    @staticmethod
    def create(
        mode: str | None = None,
        config: dict[str, Any] | None = None,
    ) -> BaseBroker:
        """Create broker instance (alias for create_broker)."""
        return create_broker(mode, config)

    @staticmethod
    def create_paper(config: dict[str, Any] | None = None) -> PaperBroker:
        """Create paper broker."""
        return create_broker("paper", config, require_confirmation=False)

    @staticmethod
    def create_live(config: dict[str, Any] | None = None) -> BaseBroker:
        """Create live broker with confirmation."""
        return create_broker("live", config, require_confirmation=True)

    @staticmethod
    def create_dhan(config: dict[str, Any] | None = None) -> BaseBroker:
        """Create Dhan live broker through the standard live safety gates."""
        config = config or {}
        config["active_broker"] = "dhan"
        return create_broker("live", config, require_confirmation=True)
