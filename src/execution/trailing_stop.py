"""Momentum-Based Trailing Stop system for dynamic stop-loss management.

This module provides dynamic trailing stops that adapt based on price momentum.
Trail faster when momentum fades, slower when momentum is strong.

All monetary values are in PAISA (1/100 INR) as integers to avoid floating-point issues.
"""

import math
import threading
from dataclasses import dataclass
from datetime import datetime
from typing import Optional
from zoneinfo import ZoneInfo

import pandas as pd
from loguru import logger

# Indian timezone
IST = ZoneInfo("Asia/Kolkata")

# NSE tick size in paisa (0.05 INR = 5 paisa)
NSE_TICK_SIZE_PAISA = 5


@dataclass
class TrailingStopPosition:
    """Minimal position data required for trailing stop calculations.

    This dataclass provides the interface between the trailing stop system
    and the actual position management system.
    """

    position_id: str
    side: str  # "BUY" for long, "SELL" for short
    entry_price: int  # in paisa
    stop_loss_price: Optional[int] = None  # in paisa (original SL)
    entry_atr: Optional[int] = None  # ATR at entry in paisa

    # Trailing stop tracking
    momentum_trailing_sl_price: Optional[int] = None
    last_momentum_update: Optional[datetime] = None
    breakeven_activated: bool = False

    # Extreme prices since entry/momentum tracking started
    highest_price_since_entry: Optional[int] = None
    lowest_price_since_entry: Optional[int] = None

    # Current momentum state
    current_momentum: float = 0.0
    current_momentum_multiplier: float = 1.0


class MomentumTrailingStop:
    """Momentum-based adaptive trailing stop-loss manager.

    Dynamically adjusts trailing stop distance based on price momentum.
    Trail slower when momentum is strong (let winners run), trail faster
    when momentum fades (protect profits).

    Momentum Calculation:
        - Uses Rate of Change (ROC) over N bars normalized to -1 to 1 scale
        - ROC = (current_price - past_price) / past_price
        - Normalized using arctan for bounded output

    Trailing Multiplier Logic:
        - Strong momentum (>0.7): 1.5x base ATR (trail slower)
        - Moderate momentum (0.3-0.7): 1.0x base ATR (normal)
        - Neutral momentum (-0.3-0.3): 0.7x base ATR (trail faster)
        - Negative momentum (<-0.3): 0.5x base ATR (trail aggressively)

    Breakeven Logic:
        - Move stop to entry + 1 tick once profit >= 0.5R
        - Never move stop further from entry than original SL

    Trailing Rules:
        - For LONG: trail_stop = highest_high - (atr * multiplier)
        - For SHORT: trail_stop = lowest_low + (atr * multiplier)
        - Only move stop in favorable direction
        - Update max every 1 minute (configurable)

    Thread Safety:
        - All methods are thread-safe using internal locks
        - Safe to call from multiple threads simultaneously

    Args:
        base_atr_multiple: Base ATR multiplier for trailing distance (default 2.0)
        momentum_period: Period for momentum calculation in bars (default 5)
        update_interval_seconds: Minimum seconds between updates (default 60)
        breakeven_threshold_r: R multiple to trigger breakeven (default 0.5)
        tick_size_paisa: Tick size in paisa (default 5 for NSE)
    """

    def __init__(
        self,
        base_atr_multiple: float = 2.0,
        momentum_period: int = 5,
        update_interval_seconds: int = 60,
        breakeven_threshold_r: float = 0.5,
        tick_size_paisa: int = NSE_TICK_SIZE_PAISA,
    ):
        """Initialize the momentum trailing stop manager.

        Args:
            base_atr_multiple: Base ATR multiplier for trailing distance.
            momentum_period: Period for momentum calculation (bars).
            update_interval_seconds: Minimum seconds between trailing stop updates.
            breakeven_threshold_r: R multiple to trigger breakeven stop.
            tick_size_paisa: Tick size in paisa (default 5 for NSE).
        """
        self._base_atr_multiple = base_atr_multiple
        self._momentum_period = momentum_period
        self._update_interval_seconds = update_interval_seconds
        self._breakeven_threshold_r = breakeven_threshold_r
        self._tick_size = tick_size_paisa

        # Thread safety
        self._lock = threading.RLock()

        # Momentum thresholds
        self._strong_threshold = 0.7
        self._moderate_threshold = 0.3

        # Multipliers for each momentum regime
        self._strong_multiplier = 1.5
        self._moderate_multiplier = 1.0
        self._neutral_multiplier = 0.7
        self._negative_multiplier = 0.5

        logger.info(
            f"MomentumTrailingStop initialized: "
            f"base_atr={base_atr_multiple}, momentum_period={momentum_period}, "
            f"update_interval={update_interval_seconds}s"
        )

    def calculate_momentum(self, data: pd.DataFrame) -> float:
        """Calculate price momentum from recent price action.

        Uses Rate of Change (ROC) over the momentum period, normalized
        to a -1 to 1 scale using arctan for bounded output.

        Args:
            data: DataFrame with OHLCV data (at least 'close' column).
                  Should have at least momentum_period + 1 bars.

        Returns:
            Momentum value between -1 (strong down) and 1 (strong up).
            Returns 0.0 if insufficient data.
        """
        with self._lock:
            if len(data) < self._momentum_period + 1:
                return 0.0

            # Method: Rate of Change (ROC) normalized
            current_price = data["close"].iloc[-1]
            past_price = data["close"].iloc[-(self._momentum_period + 1)]

            if past_price == 0:
                return 0.0

            # Calculate ROC as percentage change
            roc = (current_price - past_price) / past_price

            # Normalize to -1 to 1 scale using arctan for bounded output
            # Scale factor of 10 for sensitivity
            normalized = (2 / math.pi) * math.atan(roc * 10)

            # Clamp to [-1, 1] for safety
            return max(-1.0, min(1.0, normalized))

    def get_trailing_multiplier(self, momentum: float) -> float:
        """Get the ATR multiplier based on momentum value.

        Maps momentum to trailing stop distance multiplier:
            - momentum > 0.7: 1.5x (trail slower, let winners run)
            - 0.3 < momentum <= 0.7: 1.0x (normal trailing)
            - -0.3 <= momentum <= 0.3: 0.7x (trail faster, protect profits)
            - momentum < -0.3: 0.5x (trail aggressively)

        Args:
            momentum: Momentum value from -1 to 1.

        Returns:
            Multiplier to apply to base ATR for trailing stop distance.
        """
        with self._lock:
            if momentum > self._strong_threshold:
                # Strong momentum - trail slower, let winners run
                return self._strong_multiplier
            elif momentum > self._moderate_threshold:
                # Moderate momentum - normal trailing
                return self._moderate_multiplier
            elif momentum >= -self._moderate_threshold:
                # Neutral momentum - trail faster to protect profits
                return self._neutral_multiplier
            else:
                # Negative momentum - trail aggressively
                return self._negative_multiplier

    def _should_update_stop(self, position: TrailingStopPosition) -> bool:
        """Check if trailing stop should be updated based on time interval.

        Args:
            position: The position to check.

        Returns:
            True if enough time has passed since last update.
        """
        if position.last_momentum_update is None:
            return True

        elapsed = (datetime.now(IST) - position.last_momentum_update).total_seconds()
        return elapsed >= self._update_interval_seconds

    def _get_breakeven_price(self, position: TrailingStopPosition) -> int:
        """Calculate breakeven price (entry + 1 tick in favorable direction).

        Args:
            position: The position to calculate for.

        Returns:
            Breakeven stop price in paisa.
        """
        if position.side == "BUY":
            # For long: entry + 1 tick
            return position.entry_price + self._tick_size
        else:
            # For short: entry - 1 tick
            return position.entry_price - self._tick_size

    def _get_current_rr_ratio(self, position: TrailingStopPosition, current_price: int) -> float:
        """Calculate current risk-reward ratio.

        Args:
            position: The position to calculate for.
            current_price: Current market price in paisa.

        Returns:
            Current R:R ratio (e.g., 0.5 = 0.5R profit).
        """
        if position.stop_loss_price is None:
            return 0.0

        risk = abs(position.entry_price - position.stop_loss_price)
        if risk == 0:
            return 0.0

        if position.side == "BUY":
            reward = current_price - position.entry_price
        else:
            reward = position.entry_price - current_price

        return reward / risk

    def _estimate_atr_from_data(self, data: pd.DataFrame) -> Optional[int]:
        """Estimate ATR from recent data if entry ATR not available.

        Uses simple average of high-low ranges as ATR estimation.

        Args:
            data: DataFrame with OHLCV data (high, low columns).

        Returns:
            Estimated ATR in paisa, or None if cannot calculate.
        """
        if len(data) < 2:
            return None

        try:
            # Simple estimation using recent high-low range
            recent_data = data.tail(14)
            ranges = recent_data["high"] - recent_data["low"]
            avg_range = int(ranges.mean())
            return avg_range if avg_range > 0 else None
        except Exception:
            return None

    def update_stop(
        self,
        position: TrailingStopPosition,
        current_data: pd.DataFrame,
    ) -> Optional[int]:
        """Update the trailing stop based on momentum and price action.

        This is the main method to call for updating trailing stops. It:
        1. Checks if enough time has passed since last update
        2. Calculates current momentum from price data
        3. Determines trailing multiplier based on momentum
        4. Calculates new stop price based on extreme prices and ATR
        5. Applies breakeven logic if profit >= 0.5R
        6. Ensures stop only moves in favorable direction

        Args:
            position: The position to update (modified in-place).
            current_data: DataFrame with recent OHLCV data for momentum calc.
                          Must have 'close', 'high', 'low' columns.

        Returns:
            New stop-loss price in paisa if updated, None if no update needed.

        Thread Safety:
            This method is thread-safe.
        """
        with self._lock:
            # Check update interval
            if not self._should_update_stop(position):
                return None

            if len(current_data) < self._momentum_period:
                return None

            current_price = int(current_data["close"].iloc[-1])
            current_atr = position.entry_atr or self._estimate_atr_from_data(current_data)

            if current_atr is None or current_atr <= 0:
                return None

            # Calculate current momentum
            momentum = self.calculate_momentum(current_data)
            position.current_momentum = momentum

            # Get multiplier based on momentum
            multiplier = self.get_trailing_multiplier(momentum)
            position.current_momentum_multiplier = multiplier

            # Calculate effective ATR distance
            effective_atr_distance = int(
                current_atr * self._base_atr_multiple * multiplier
            )

            # Update extreme prices for trailing calculation
            if position.side == "BUY":
                return self._update_long_stop(
                    position, current_price, effective_atr_distance
                )
            else:
                return self._update_short_stop(
                    position, current_price, effective_atr_distance
                )

    def _update_long_stop(
        self,
        position: TrailingStopPosition,
        current_price: int,
        effective_atr_distance: int,
    ) -> Optional[int]:
        """Update trailing stop for a long position.

        Args:
            position: The long position to update.
            current_price: Current market price in paisa.
            effective_atr_distance: Calculated ATR distance for trailing.

        Returns:
            New stop price if updated, None otherwise.
        """
        # Update highest price seen
        if (
            position.highest_price_since_entry is None
            or current_price > position.highest_price_since_entry
        ):
            position.highest_price_since_entry = current_price

        highest = position.highest_price_since_entry

        # Calculate new trailing stop
        new_stop = highest - effective_atr_distance

        # Check breakeven logic
        breakeven_price = self._get_breakeven_price(position)
        current_rr = self._get_current_rr_ratio(position, current_price)

        if current_rr >= self._breakeven_threshold_r and not position.breakeven_activated:
            # Move to breakeven
            if breakeven_price > new_stop:
                new_stop = breakeven_price
                position.breakeven_activated = True
                logger.info(
                    f"Breakeven activated for {position.position_id}: "
                    f"SL moved to {breakeven_price / 100:.2f}"
                )
        elif position.breakeven_activated:
            # Ensure we don't go below breakeven once activated
            if new_stop < breakeven_price:
                new_stop = breakeven_price

        # Ensure we never move stop further from entry than original SL
        if position.stop_loss_price and new_stop < position.stop_loss_price:
            new_stop = position.stop_loss_price

        # Only move stop in favorable direction (up for long)
        if (
            position.momentum_trailing_sl_price is None
            or new_stop > position.momentum_trailing_sl_price
        ):
            position.momentum_trailing_sl_price = new_stop
            position.last_momentum_update = datetime.now(IST)
            logger.debug(
                f"Momentum trailing SL updated for {position.position_id}: "
                f"{new_stop / 100:.2f} (momentum={position.current_momentum:.2f}, "
                f"mult={position.current_momentum_multiplier:.2f})"
            )
            return new_stop

        return None

    def _update_short_stop(
        self,
        position: TrailingStopPosition,
        current_price: int,
        effective_atr_distance: int,
    ) -> Optional[int]:
        """Update trailing stop for a short position.

        Args:
            position: The short position to update.
            current_price: Current market price in paisa.
            effective_atr_distance: Calculated ATR distance for trailing.

        Returns:
            New stop price if updated, None otherwise.
        """
        # Update lowest price seen
        if (
            position.lowest_price_since_entry is None
            or current_price < position.lowest_price_since_entry
        ):
            position.lowest_price_since_entry = current_price

        lowest = position.lowest_price_since_entry

        # Calculate new trailing stop
        new_stop = lowest + effective_atr_distance

        # Check breakeven logic
        breakeven_price = self._get_breakeven_price(position)
        current_rr = self._get_current_rr_ratio(position, current_price)

        if current_rr >= self._breakeven_threshold_r and not position.breakeven_activated:
            # Move to breakeven
            if breakeven_price < new_stop:
                new_stop = breakeven_price
                position.breakeven_activated = True
                logger.info(
                    f"Breakeven activated for {position.position_id}: "
                    f"SL moved to {breakeven_price / 100:.2f}"
                )
        elif position.breakeven_activated:
            # Ensure we don't go above breakeven once activated
            if new_stop > breakeven_price:
                new_stop = breakeven_price

        # Ensure we never move stop further from entry than original SL
        if position.stop_loss_price and new_stop > position.stop_loss_price:
            new_stop = position.stop_loss_price

        # Only move stop in favorable direction (down for short)
        if (
            position.momentum_trailing_sl_price is None
            or new_stop < position.momentum_trailing_sl_price
        ):
            position.momentum_trailing_sl_price = new_stop
            position.last_momentum_update = datetime.now(IST)
            logger.debug(
                f"Momentum trailing SL updated for {position.position_id}: "
                f"{new_stop / 100:.2f} (momentum={position.current_momentum:.2f}, "
                f"mult={position.current_momentum_multiplier:.2f})"
            )
            return new_stop

        return None

    def get_momentum_status(self, position: TrailingStopPosition) -> dict:
        """Get current momentum trailing stop status for reporting.

        Args:
            position: The position to get status for.

        Returns:
            Dictionary with momentum status information:
                - momentum: Current momentum value (-1 to 1)
                - multiplier: Current ATR multiplier
                - momentum_trailing_sl: Current trailing stop price
                - breakeven_activated: Whether breakeven is active
                - last_update: ISO timestamp of last update
                - highest_price: Highest price seen (long positions)
                - lowest_price: Lowest price seen (short positions)
        """
        with self._lock:
            return {
                "momentum": position.current_momentum,
                "multiplier": position.current_momentum_multiplier,
                "momentum_trailing_sl": position.momentum_trailing_sl_price,
                "breakeven_activated": position.breakeven_activated,
                "last_update": (
                    position.last_momentum_update.isoformat()
                    if position.last_momentum_update
                    else None
                ),
                "highest_price": position.highest_price_since_entry,
                "lowest_price": position.lowest_price_since_entry,
            }

    def reset_position(self, position: TrailingStopPosition) -> None:
        """Reset trailing stop state for a position.

        Useful when reusing position objects or restarting tracking.

        Args:
            position: The position to reset.
        """
        with self._lock:
            position.momentum_trailing_sl_price = None
            position.last_momentum_update = None
            position.breakeven_activated = False
            position.highest_price_since_entry = position.entry_price
            position.lowest_price_since_entry = position.entry_price
            position.current_momentum = 0.0
            position.current_momentum_multiplier = 1.0


class FixedTrailingStop:
    """Fixed percentage-based trailing stop for simpler use cases.

    Provides traditional trailing stop functionality that trails by a
    fixed percentage/distance from the extreme price.

    Args:
        trail_pct: Percentage to trail behind extreme price (e.g., 0.01 = 1%)
        activate_after_pct: Profit % required to activate trailing (default 0)
    """

    def __init__(
        self,
        trail_pct: float = 0.01,
        activate_after_pct: float = 0.0,
    ):
        """Initialize fixed trailing stop.

        Args:
            trail_pct: Percentage to trail (e.g., 0.01 = 1%).
            activate_after_pct: Profit % needed to activate trailing.
        """
        self._trail_pct = trail_pct
        self._activate_after_pct = activate_after_pct
        self._lock = threading.RLock()

    def update_stop(
        self,
        position: TrailingStopPosition,
        current_price: int,
    ) -> Optional[int]:
        """Update trailing stop based on fixed percentage.

        Args:
            position: The position to update.
            current_price: Current market price in paisa.

        Returns:
            New stop price if updated, None otherwise.
        """
        with self._lock:
            # Check if trailing should be activated
            if position.side == "BUY":
                profit_pct = (current_price - position.entry_price) / position.entry_price
            else:
                profit_pct = (position.entry_price - current_price) / position.entry_price

            if profit_pct < self._activate_after_pct:
                return None

            # Calculate trail distance
            trail_distance = int(position.entry_price * self._trail_pct)

            if position.side == "BUY":
                # Update highest price
                if (
                    position.highest_price_since_entry is None
                    or current_price > position.highest_price_since_entry
                ):
                    position.highest_price_since_entry = current_price

                highest = position.highest_price_since_entry
                new_stop = highest - trail_distance

                # Only move up
                if (
                    position.momentum_trailing_sl_price is None
                    or new_stop > position.momentum_trailing_sl_price
                ):
                    position.momentum_trailing_sl_price = new_stop
                    return new_stop

            else:  # SHORT
                # Update lowest price
                if (
                    position.lowest_price_since_entry is None
                    or current_price < position.lowest_price_since_entry
                ):
                    position.lowest_price_since_entry = current_price

                lowest = position.lowest_price_since_entry
                new_stop = lowest + trail_distance

                # Only move down
                if (
                    position.momentum_trailing_sl_price is None
                    or new_stop < position.momentum_trailing_sl_price
                ):
                    position.momentum_trailing_sl_price = new_stop
                    return new_stop

            return None


def create_trailing_stop(
    trailing_type: str = "momentum",
    **kwargs,
) -> MomentumTrailingStop | FixedTrailingStop:
    """Factory function to create trailing stop instances.

    Args:
        trailing_type: Type of trailing stop ('momentum' or 'fixed').
        **kwargs: Arguments passed to the specific trailing stop class.

    Returns:
        Configured trailing stop instance.

    Raises:
        ValueError: If trailing_type is not recognized.
    """
    if trailing_type == "momentum":
        return MomentumTrailingStop(**kwargs)
    elif trailing_type == "fixed":
        return FixedTrailingStop(**kwargs)
    else:
        raise ValueError(f"Unknown trailing stop type: {trailing_type}")
