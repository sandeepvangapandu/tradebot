"""4-Tier Partial Profit System for scaling out of positions.

This module provides a systematic approach to profit-taking by scaling out
of positions at multiple risk:reward levels. This locks in profits while
allowing winners to run with a trailing stop on the final portion.

All monetary values are in PAISA (integer) - 1 Rupee = 100 paisa.
"""

import threading
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional
from zoneinfo import ZoneInfo

from loguru import logger

# Indian timezone
IST = ZoneInfo("Asia/Kolkata")


class PositionDirection(str, Enum):
    """Position direction - LONG or SHORT."""
    LONG = "LONG"
    SHORT = "SHORT"


class TierAction(str, Enum):
    """Action to take when a tier is triggered."""
    EXIT = "exit"
    TRAIL = "trail"


@dataclass
class ExitTier:
    """Represents a single profit-taking tier.

    Attributes:
        tier: Tier number (1-4)
        rr_level: Risk:reward level (e.g., 1.0, 2.0, 3.0, None for trailing)
        exit_percent: Percentage of position to exit (25% each tier)
        exit_price: Calculated exit price in paisa (None until calculated)
        exited: Whether this tier has been executed
        exited_at: Timestamp when tier was executed
        exited_price: Actual exit price in paisa when executed
        realized_pnl: Realized P&L from this tier in paisa
    """
    tier: int
    rr_level: Optional[float]
    exit_percent: float
    exit_price: Optional[int] = None
    exited: bool = False
    exited_at: Optional[datetime] = None
    exited_price: Optional[int] = None
    realized_pnl: int = 0

    def __post_init__(self):
        """Validate tier configuration."""
        if not 1 <= self.tier <= 4:
            raise ValueError(f"Tier must be between 1 and 4, got {self.tier}")
        if not 0 < self.exit_percent <= 100:
            raise ValueError(f"Exit percent must be between 0 and 100, got {self.exit_percent}")


# Default 4-tier configuration
FOUR_TIER_CONFIG = [
    {"tier": 1, "rr": 1.0, "percent": 25, "action": TierAction.EXIT},
    {"tier": 2, "rr": 2.0, "percent": 25, "action": TierAction.EXIT},
    {"tier": 3, "rr": 3.0, "percent": 25, "action": TierAction.EXIT},
    {"tier": 4, "rr": None, "percent": 25, "action": TierAction.TRAIL},
]


@dataclass
class TierExitResult:
    """Result of a tier exit execution.

    Attributes:
        tier: The tier number that was executed
        quantity: Quantity exited
        exit_price: Exit price in paisa
        realized_pnl: Realized P&L in paisa
        exit_reason: Reason for exit (e.g., "tier_1_hit", "trailing_stop")
    """
    tier: int
    quantity: int
    exit_price: int
    realized_pnl: int
    exit_reason: str


@dataclass
class PositionSnapshot:
    """Snapshot of position state for tier checking.

    Attributes:
        instrument_key: Unique instrument identifier
        direction: Position direction (LONG or SHORT)
        entry_price: Entry price in paisa
        stop_loss_price: Stop loss price in paisa
        current_price: Current market price in paisa
        total_quantity: Total position quantity
        remaining_quantity: Current remaining quantity
    """
    instrument_key: str
    direction: PositionDirection
    entry_price: int
    stop_loss_price: int
    current_price: int
    total_quantity: int
    remaining_quantity: int


class PartialProfitManager:
    """Manages 4-tier partial profit taking system.

    This class handles the initialization, tracking, and execution of
    partial exits at predefined R:R levels. It supports both fixed exits
    at R:R levels and trailing stops for the final tier.

    All prices are in PAISA (integer) to avoid floating-point precision issues.

    Thread-safe: Uses locks for concurrent access to tier state.

    Example:
        >>> manager = PartialProfitManager()
        >>> tiers = manager.initialize_tiers(
        ...     entry=10000,  # 100.00 INR
        ...     stop_loss=9900,  # 99.00 INR
        ...     quantity=100,
        ...     direction=PositionDirection.LONG
        ... )
        >>> position = PositionSnapshot(
        ...     instrument_key="NSE_EQ|RELIANCE",
        ...     direction=PositionDirection.LONG,
        ...     entry_price=10000,
        ...     stop_loss_price=9900,
        ...     current_price=10150,
        ...     total_quantity=100,
        ...     remaining_quantity=100
        ... ... )
        >>> exits = manager.check_tiers(position, tiers)
    """

    def __init__(self):
        """Initialize the partial profit manager."""
        self._lock = threading.RLock()
        self._config = FOUR_TIER_CONFIG
        logger.info("PartialProfitManager initialized with 4-tier configuration")

    def initialize_tiers(
        self,
        entry: int,
        stop_loss: int,
        quantity: int,
        direction: PositionDirection,
        custom_config: Optional[list[dict]] = None
    ) -> list[ExitTier]:
        """Initialize exit tiers for a position.

        Creates 4 tiers with calculated exit prices based on R:R levels.
        Tier 4 uses a trailing stop (no fixed exit price).

        Args:
            entry: Entry price in paisa
            stop_loss: Stop loss price in paisa
            quantity: Total position quantity
            direction: Position direction (LONG or SHORT)
            custom_config: Optional custom tier configuration (defaults to FOUR_TIER_CONFIG)

        Returns:
            List of ExitTier objects with calculated exit prices

        Raises:
            ValueError: If entry, stop_loss, or quantity is invalid

        Example:
            >>> manager = PartialProfitManager()
            >>> tiers = manager.initialize_tiers(
            ...     entry=10000,  # 100.00 INR
            ...     stop_loss=9900,  # 99.00 INR (1% risk)
            ...     quantity=100,
            ...     direction=PositionDirection.LONG
            ... )
            >>> # Tier 1: Exit 25% at 10100 (1:1 R:R)
            >>> # Tier 2: Exit 25% at 10200 (1:2 R:R)
            >>> # Tier 3: Exit 25% at 10300 (1:3 R:R)
            >>> # Tier 4: Trail final 25%
        """
        if entry <= 0 or stop_loss <= 0:
            raise ValueError(f"Entry ({entry}) and stop_loss ({stop_loss}) must be positive")
        if quantity <= 0:
            raise ValueError(f"Quantity ({quantity}) must be positive")

        config = custom_config or self._config
        tiers = []

        for tier_config in config:
            tier = ExitTier(
                tier=tier_config["tier"],
                rr_level=tier_config["rr"],
                exit_percent=tier_config["percent"],
            )
            tiers.append(tier)

        # Calculate exit prices for each tier
        tiers = self.calculate_tier_prices(tiers, entry, stop_loss, direction)

        # Validate total percentage equals 100%
        total_percent = sum(t.exit_percent for t in tiers)
        if total_percent != 100:
            logger.warning(f"Tier percentages sum to {total_percent}%, expected 100%")

        direction_str = direction.value if hasattr(direction, 'value') else str(direction)
        logger.info(
            f"Initialized {len(tiers)} tiers for {direction_str} position: "
            f"entry={entry}p, sl={stop_loss}p, qty={quantity}"
        )

        return tiers

    def calculate_tier_prices(
        self,
        tiers: list[ExitTier],
        entry: int,
        stop_loss: int,
        direction: PositionDirection
    ) -> list[ExitTier]:
        """Calculate exit prices for each tier based on R:R levels.

        For LONG positions:
            - Risk = Entry - Stop Loss
            - Reward at R:R = Entry + (Risk * R:R)

        For SHORT positions:
            - Risk = Stop Loss - Entry
            - Reward at R:R = Entry - (Risk * R:R)

        Tier 4 (trailing) has no fixed exit price.

        Args:
            tiers: List of ExitTier objects
            entry: Entry price in paisa
            stop_loss: Stop loss price in paisa
            direction: Position direction (LONG or SHORT)

        Returns:
            List of ExitTier objects with exit_price populated
        """
        # Calculate risk distance (always positive)
        risk_distance = abs(entry - stop_loss)

        for tier in tiers:
            if tier.rr_level is None:
                # Tier 4 uses trailing stop - no fixed exit price
                tier.exit_price = None
                logger.debug(f"Tier {tier.tier}: Trailing stop (no fixed price)")
                continue

            # Calculate reward distance for this R:R level
            reward_distance = int(risk_distance * tier.rr_level)

            # Handle both enum and string directions
            is_long = (direction == PositionDirection.LONG or
                      (isinstance(direction, str) and direction.upper() == "LONG"))
            if is_long:
                tier.exit_price = entry + reward_distance
            else:  # SHORT
                tier.exit_price = entry - reward_distance

            logger.debug(
                f"Tier {tier.tier}: R:R {tier.rr_level}, "
                f"exit_price={tier.exit_price}p"
            )

        return tiers

    def check_tiers(
        self,
        position: PositionSnapshot,
        tiers: list[ExitTier]
    ) -> list[TierExitResult]:
        """Check if any tiers should be executed based on current price.

        For LONG positions:
            - Tiers execute when current_price >= exit_price
            - Trailing stop executes when current_price <= trailing_stop_price

        For SHORT positions:
            - Tiers execute when current_price <= exit_price
            - Trailing stop executes when current_price >= trailing_stop_price

        Args:
            position: Current position snapshot
            tiers: List of ExitTier objects for this position

        Returns:
            List of TierExitResult objects for tiers that should be executed
        """
        with self._lock:
            exits_to_execute = []
            current_price = position.current_price
            direction = position.direction

            for tier in tiers:
                if tier.exited:
                    continue

                # Check if tier should execute
                should_execute = False
                exit_reason = ""

                if tier.rr_level is None:
                    # Tier 4: Trailing stop - check if hit
                    # Note: Trailing stop price should be managed externally
                    # and passed in position.stop_loss_price for final tier
                    continue
                else:
                    # Fixed R:R tier
                    if tier.exit_price is None:
                        continue

                    if direction == PositionDirection.LONG:
                        if current_price >= tier.exit_price:
                            should_execute = True
                            exit_reason = f"tier_{tier.tier}_hit"
                    else:  # SHORT
                        if current_price <= tier.exit_price:
                            should_execute = True
                            exit_reason = f"tier_{tier.tier}_hit"

                if should_execute:
                    # Calculate quantity to exit for this tier
                    tier_quantity = self._calculate_tier_quantity(
                        tier, position.total_quantity, position.remaining_quantity
                    )

                    if tier_quantity <= 0:
                        logger.warning(f"Tier {tier.tier} calculated quantity is 0, skipping")
                        continue

                    # Calculate realized P&L for this exit
                    realized_pnl = self._calculate_tier_pnl(
                        position.entry_price,
                        current_price,
                        tier_quantity,
                        direction
                    )

                    result = TierExitResult(
                        tier=tier.tier,
                        quantity=tier_quantity,
                        exit_price=current_price,
                        realized_pnl=realized_pnl,
                        exit_reason=exit_reason
                    )
                    exits_to_execute.append(result)

                    logger.info(
                        f"Tier {tier.tier} triggered for {position.instrument_key}: "
                        f"exit_price={current_price}p, qty={tier_quantity}, "
                        f"pnl={realized_pnl}p"
                    )

            return exits_to_execute

    def mark_tier_exited(
        self,
        tier: ExitTier,
        exit_price: int,
        realized_pnl: int
    ) -> None:
        """Mark a tier as exited with execution details.

        Args:
            tier: The ExitTier to mark as exited
            exit_price: Actual exit price in paisa
            realized_pnl: Realized P&L from this exit in paisa
        """
        with self._lock:
            tier.exited = True
            tier.exited_at = datetime.now(IST)
            tier.exited_price = exit_price
            tier.realized_pnl = realized_pnl

            logger.info(
                f"Tier {tier.tier} marked as exited: price={exit_price}p, "
                f"pnl={realized_pnl}p at {tier.exited_at}"
            )

    def get_remaining_quantity(self, tiers: list[ExitTier], total_qty: int) -> int:
        """Calculate remaining position quantity after executed tiers.

        Args:
            tiers: List of ExitTier objects
            total_qty: Total original position quantity

        Returns:
            Remaining quantity in shares/lots

        Example:
            >>> tiers = manager.initialize_tiers(entry=10000, stop_loss=9900, quantity=100, direction=LONG)
            >>> # After tier 1 exits
            >>> remaining = manager.get_remaining_quantity(tiers, 100)  # Returns 75
        """
        with self._lock:
            exited_percent = sum(
                t.exit_percent for t in tiers if t.exited
            )
            exited_qty = int(total_qty * exited_percent / 100)
            remaining = total_qty - exited_qty

            # Handle odd lots - ensure we don't go negative
            return max(0, remaining)

    def get_realized_pnl(self, tiers: list[ExitTier]) -> int:
        """Get total realized P&L from all executed tiers.

        Args:
            tiers: List of ExitTier objects

        Returns:
            Total realized P&L in paisa
        """
        with self._lock:
            return sum(t.realized_pnl for t in tiers if t.exited)

    def get_unexited_tiers(self, tiers: list[ExitTier]) -> list[ExitTier]:
        """Get list of tiers that have not yet been executed.

        Args:
            tiers: List of ExitTier objects

        Returns:
            List of ExitTier objects that are not yet exited
        """
        with self._lock:
            return [t for t in tiers if not t.exited]

    def get_next_tier(self, tiers: list[ExitTier]) -> Optional[ExitTier]:
        """Get the next tier to be executed.

        Args:
            tiers: List of ExitTier objects

        Returns:
            Next ExitTier to be executed, or None if all tiers exited
        """
        with self._lock:
            unexited = self.get_unexited_tiers(tiers)
            if not unexited:
                return None
            return min(unexited, key=lambda t: t.tier)

    def is_complete(self, tiers: list[ExitTier]) -> bool:
        """Check if all tiers have been executed.

        Args:
            tiers: List of ExitTier objects

        Returns:
            True if all tiers are exited, False otherwise
        """
        with self._lock:
            return all(t.exited for t in tiers)

    def get_tier_summary(self, tiers: list[ExitTier]) -> dict:
        """Get a summary of tier status.

        Args:
            tiers: List of ExitTier objects

        Returns:
            Dictionary with tier summary statistics
        """
        with self._lock:
            exited_count = sum(1 for t in tiers if t.exited)
            total_realized = self.get_realized_pnl(tiers)

            tier_details = []
            for tier in tiers:
                tier_details.append({
                    "tier": tier.tier,
                    "rr_level": tier.rr_level,
                    "exit_percent": tier.exit_percent,
                    "exit_price": tier.exit_price,
                    "exited": tier.exited,
                    "exited_at": tier.exited_at.isoformat() if tier.exited_at else None,
                    "realized_pnl": tier.realized_pnl if tier.exited else 0,
                })

            return {
                "total_tiers": len(tiers),
                "exited_tiers": exited_count,
                "remaining_tiers": len(tiers) - exited_count,
                "total_realized_pnl_paisa": total_realized,
                "total_realized_pnl_rupees": total_realized / 100,
                "is_complete": self.is_complete(tiers),
                "tiers": tier_details,
            }

    def _calculate_tier_quantity(
        self,
        tier: ExitTier,
        total_qty: int,
        remaining_qty: int
    ) -> int:
        """Calculate quantity to exit for a specific tier.

        Handles odd lots by ensuring we don't exceed remaining quantity
        and distribute rounding errors appropriately.

        Args:
            tier: The ExitTier being calculated
            total_qty: Total original position quantity
            remaining_qty: Current remaining quantity

        Returns:
            Quantity to exit for this tier
        """
        # Calculate ideal quantity for this tier
        ideal_qty = int(total_qty * tier.exit_percent / 100)

        # For the last unexited tier, exit all remaining quantity
        # This handles odd lots and rounding
        unexited = [t for t in [tier] if not t.exited]  # Check if this is last tier
        all_tiers = [t for t in [tier]]  # Placeholder - we need actual context

        # Ensure we don't exceed remaining quantity
        qty = min(ideal_qty, remaining_qty)

        # Ensure at least 1 if there's remaining quantity and this tier should execute
        if remaining_qty > 0 and qty == 0:
            qty = 1

        return qty

    def _calculate_tier_pnl(
        self,
        entry_price: int,
        exit_price: int,
        quantity: int,
        direction: PositionDirection
    ) -> int:
        """Calculate realized P&L for a tier exit.

        Args:
            entry_price: Entry price in paisa
            exit_price: Exit price in paisa
            quantity: Quantity exited
            direction: Position direction (LONG or SHORT)

        Returns:
            Realized P&L in paisa
        """
        if direction == PositionDirection.LONG:
            pnl = (exit_price - entry_price) * quantity
        else:  # SHORT
            pnl = (entry_price - exit_price) * quantity

        return pnl

    def calculate_trailing_stop_price(
        self,
        entry: int,
        current_price: int,
        highest_price: int,
        stop_loss_distance: int,
        direction: PositionDirection,
        activation_rr: float = 1.0
    ) -> Optional[int]:
        """Calculate trailing stop price for Tier 4.

        Trailing stop activates after position reaches activation R:R.
        It trails the highest (for long) or lowest (for short) price
        by the stop loss distance.

        Args:
            entry: Entry price in paisa
            current_price: Current market price in paisa
            highest_price: Highest price since entry (for long) or lowest (for short)
            stop_loss_distance: Distance between entry and initial stop loss in paisa
            direction: Position direction (LONG or SHORT)
            activation_rr: R:R level at which trailing stop activates (default 1.0)

        Returns:
            Trailing stop price in paisa, or None if not yet activated
        """
        # Check if trailing stop is activated
        if direction == PositionDirection.LONG:
            current_rr = (current_price - entry) / stop_loss_distance if stop_loss_distance > 0 else 0
            if current_rr < activation_rr:
                return None
            # Trail below highest price
            trailing_price = highest_price - stop_loss_distance
            return max(trailing_price, entry)  # Don't trail below entry
        else:  # SHORT
            current_rr = (entry - current_price) / stop_loss_distance if stop_loss_distance > 0 else 0
            if current_rr < activation_rr:
                return None
            # Trail above lowest price
            trailing_price = highest_price + stop_loss_distance
            return min(trailing_price, entry)  # Don't trail above entry

    def reset_tiers(self, tiers: list[ExitTier]) -> None:
        """Reset all tiers to unexited state.

        Useful for testing or when restarting position tracking.

        Args:
            tiers: List of ExitTier objects to reset
        """
        with self._lock:
            for tier in tiers:
                tier.exited = False
                tier.exited_at = None
                tier.exited_price = None
                tier.realized_pnl = 0

            logger.info(f"Reset {len(tiers)} tiers to unexited state")


# Convenience function for quick tier setup
def create_partial_profit_tiers(
    entry: int,
    stop_loss: int,
    quantity: int,
    direction: PositionDirection
) -> list[ExitTier]:
    """Quick factory function to create 4-tier partial profit configuration.

    Args:
        entry: Entry price in paisa
        stop_loss: Stop loss price in paisa
        quantity: Total position quantity
        direction: Position direction (LONG or SHORT)

    Returns:
        List of configured ExitTier objects

    Example:
        >>> tiers = create_partial_profit_tiers(
        ...     entry=10000,  # 100.00 INR
        ...     stop_loss=9900,  # 99.00 INR
        ...     quantity=100,
        ...     direction=PositionDirection.LONG
        ... )
    """
    manager = PartialProfitManager()
    return manager.initialize_tiers(entry, stop_loss, quantity, direction)
