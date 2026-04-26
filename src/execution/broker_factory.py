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
    return PaperBroker(initial_capital=initial_capital)


def _create_live_broker(
    config: dict[str, Any],
    require_confirmation: bool,
) -> BaseBroker:
    """Create a live trading broker instance.

    Reads ACTIVE_BROKER env var to select between 'dhan' (default) and 'upstox'.
    """
    # Safety check — MUST be first
    if require_confirmation:
        live_confirmed = os.getenv("LIVE_TRADING_CONFIRMED", "false").lower()
        if live_confirmed != "true":
            raise RuntimeError(
                "LIVE TRADING SAFETY CHECK FAILED!\n\n"
                "To enable live trading, you must:\n"
                "1. Set TRADING_MODE=live in your environment\n"
                "2. Set LIVE_TRADING_CONFIRMED=true\n"
                "3. Verify your API credentials are correct\n\n"
                "Paper trading is strongly recommended for testing."
            )

    logger.warning("🚨 CREATING LIVE BROKER - REAL ORDERS WILL BE PLACED!")

    active_broker = os.getenv("ACTIVE_BROKER", "dhan").lower()

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
        """Create Dhan live broker directly (bypasses mode env var)."""
        config = config or {}
        return _create_dhan_broker(config)
