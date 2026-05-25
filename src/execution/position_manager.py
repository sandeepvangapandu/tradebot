"""Position Manager for tracking open positions and managing exits.

This module provides the PositionManager class that tracks open positions,
updates unrealized P&L on each tick, checks stop-loss/target/trailing SL
conditions, and handles time-based exits including forced intraday square-off.

Smart Exit Logic (C2):
- Partial profit booking at 1:1 R:R (50% position)
- ATR-based adaptive stop loss
- Volatility-based target adjustment
- Time-decay exit for options
- Multiple exit price tracking for blended P&L

Momentum-Based Trailing Stop (C2):
- Dynamic trailing stops that adapt based on price momentum
- Trail faster when momentum fades, slower when momentum is strong
- Breakeven stop after 0.5R profit
- Captures more of trending moves while protecting profits
"""

import threading
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
from typing import TYPE_CHECKING, Any, Callable, Optional
from zoneinfo import ZoneInfo

import pandas as pd
from loguru import logger
from pydantic import BaseModel, Field

from src.execution.base_broker import BaseBroker, Order, OrderSide, OrderType, ProductType
from src.execution.paper_broker import PaperBroker
from src.execution.partial_profit import (
    ExitTier,
    PartialProfitManager,
    PositionDirection,
    PositionSnapshot,
    TierExitResult,
)

if TYPE_CHECKING:
    from src.risk.risk_manager import RiskManager

# Indian timezone
IST = ZoneInfo("Asia/Kolkata")

# Market timings
INTRADAY_SQUARE_OFF_TIME = time(15, 15)  # 3:15 PM IST


class PositionConfig(BaseModel):
    """Configuration for position management."""

    stop_loss_pct: float = Field(default=0.02, description="Stop loss percentage (e.g., 0.02 = 2%)")
    target_pct: float = Field(default=0.04, description="Target profit percentage (e.g., 0.04 = 4%)")
    trailing_sl_activation_pct: float = Field(
        default=0.02, description="Profit % to activate trailing SL"
    )
    trailing_sl_pct: float = Field(default=0.01, description="Trailing SL percentage")
    time_based_exit_time: Optional[time] = Field(
        default=None, description="Time-based exit (None for no time exit)"
    )
    enable_trailing_sl: bool = Field(default=True, description="Enable trailing stop-loss")

    # Smart exit configuration
    enable_partial_profit_booking: bool = Field(
        default=True, description="Enable partial profit booking at 1:1 R:R"
    )
    partial_booking_ratio: float = Field(
        default=0.5, description="Fraction of position to book at 1:1 R:R (0.5 = 50%)"
    )
    partial_booking_rr_ratio: float = Field(
        default=1.0, description="R:R ratio at which to book partial profits (1.0 = 1:1)"
    )
    trailing_sl_after_partial_multiplier: float = Field(
        default=1.5, description="Wider SL multiplier after partial booking (1.5x original SL distance)"
    )

    # ATR-based SL configuration
    use_atr_based_sl: bool = Field(default=True, description="Use ATR-based adaptive stop loss")
    atr_period: int = Field(default=14, description="ATR period for SL calculation")
    atr_multiplier_low_volatility: float = Field(default=1.5, description="ATR multiplier in low volatility")
    atr_multiplier_normal_volatility: float = Field(default=2.0, description="ATR multiplier in normal volatility")
    atr_multiplier_high_volatility: float = Field(default=2.5, description="ATR multiplier in high volatility")

    # Volatility-based target adjustment
    enable_volatility_target_adjustment: bool = Field(
        default=True, description="Adjust targets based on volatility regime"
    )
    high_volatility_target_reduction: float = Field(
        default=0.20, description="Reduce target by this % in high volatility (0.20 = 20%)"
    )
    low_volatility_target_extension: float = Field(
        default=0.10, description="Extend target by this % in low volatility (0.10 = 10%)"
    )
    high_volatility_atr_threshold: float = Field(
        default=1.5, description="ATR > 1.5x average = high volatility"
    )
    low_volatility_atr_threshold: float = Field(
        default=0.7, description="ATR < 0.7x average = low volatility"
    )

    # Options time-decay configuration
    enable_options_time_decay_exit: bool = Field(
        default=True, description="Enable time-decay based exits for options"
    )
    options_dte_exit_threshold: int = Field(
        default=2, description="Exit options with DTE < this if not profitable"
    )
    options_theta_threshold_pct: float = Field(
        default=0.05, description="Exit if theta > 5% of premium per day"
    )

    # Momentum-based trailing stop configuration
    trailing_stop_type: str = Field(
        default="fixed", description="Trailing stop type: 'fixed' or 'momentum'"
    )
    momentum_trailing_base_atr_multiple: float = Field(
        default=2.0, description="Base ATR multiplier for momentum trailing stop"
    )
    momentum_period: int = Field(
        default=5, description="Period for momentum calculation (bars)"
    )
    momentum_breakeven_threshold_r: float = Field(
        default=0.5, description="Move to breakeven after this R multiple profit"
    )
    momentum_update_interval_seconds: int = Field(
        default=60, description="Minimum seconds between trailing stop updates"
    )
    momentum_strong_threshold: float = Field(
        default=0.7, description="Momentum > this = strong (0-1 scale)"
    )
    momentum_moderate_threshold: float = Field(
        default=0.3, description="Momentum > this = moderate (0-1 scale)"
    )
    momentum_strong_multiplier: float = Field(
        default=1.5, description="Multiplier for strong momentum (trails slower)"
    )
    momentum_moderate_multiplier: float = Field(
        default=1.0, description="Multiplier for moderate momentum"
    )
    momentum_neutral_multiplier: float = Field(
        default=0.7, description="Multiplier for neutral momentum (trails faster)"
    )
    momentum_negative_multiplier: float = Field(
        default=0.5, description="Multiplier for negative momentum (trails aggressively)"
    )


@dataclass
class PartialExitRecord:
    """Record of a partial exit."""

    exit_price: int  # in paisa
    quantity: int
    exit_time: datetime
    exit_reason: str
    realized_pnl: int  # in paisa


@dataclass
class ManagedPosition:
    """Extended position with management metadata."""

    instrument_key: str
    strategy_id: str
    side: OrderSide  # BUY for long, SELL for short
    quantity: int
    entry_price: int  # in paisa
    entry_time: datetime
    product_type: ProductType
    position_id: str = field(default_factory=lambda: f"POS_{datetime.now(IST).strftime('%Y%m%d%H%M%S')}_{__import__('uuid').uuid4().hex[:8].upper()}")

    # Same-bar exit guard — the bar timestamp at which this position was opened.
    # PositionManager will skip SL/target/trailing checks for this position
    # while the current bar time equals entry_bar_time, preventing lookahead
    # where a position is stopped out by a price that printed before entry.
    entry_bar_time: Optional[datetime] = None

    # Exit configuration
    stop_loss_price: Optional[int] = None  # in paisa
    target_price: Optional[int] = None  # in paisa
    trailing_sl_activated: bool = False
    highest_price_since_entry: Optional[int] = None  # for trailing SL (long)
    lowest_price_since_entry: Optional[int] = None  # for trailing SL (short)
    trailing_sl_price: Optional[int] = None  # current trailing SL price

    # ATR-based SL configuration
    entry_atr: Optional[int] = None  # ATR at entry time (in paisa)
    volatility_regime: str = "normal"  # "low", "normal", "high"
    atr_multiplier_used: float = 2.0  # Multiplier used for SL calculation

    # Options-specific data
    is_option: bool = False
    option_dte: Optional[int] = None  # Days to expiry
    option_theta: Optional[float] = None  # Daily theta decay
    option_premium_at_entry: Optional[int] = None  # Premium at entry (paisa)

    # Underlying-move emergency exit (wired 2026-05-15)
    # When the *underlying index* moves more than emergency_exit_move_pct from
    # underlying_entry_price, PositionManager closes this option position
    # regardless of the option's own SL/target. Used by short straddles to
    # cap tail risk on directional moves that would otherwise blow through
    # the per-leg SL.
    underlying_instrument_key: Optional[str] = None
    underlying_entry_price: Optional[int] = None  # in paisa
    emergency_exit_move_pct: Optional[float] = None  # e.g. 2.0 = 2%

    # Partial profit booking tracking
    partial_exits: list[PartialExitRecord] = field(default_factory=list)
    partial_booking_done: bool = False
    original_quantity: int = 0  # Original position size before partial exits
    remaining_quantity: int = 0  # Current remaining quantity

    # P&L tracking
    unrealized_pnl: int = 0  # in paisa
    realized_pnl: int = 0  # in paisa (includes partial exits)
    current_price: Optional[int] = None  # in paisa

    # Momentum-based trailing stop tracking
    momentum_trailing_sl_price: Optional[int] = None  # Current momentum-based trailing SL
    last_momentum_update: Optional[datetime] = None  # Last time momentum SL was updated
    breakeven_activated: bool = False  # Whether breakeven SL is active
    highest_momentum_price: Optional[int] = None  # Highest price since momentum SL active (long)
    lowest_momentum_price: Optional[int] = None  # Lowest price since momentum SL active (short)
    current_momentum: float = 0.0  # Current momentum value (-1 to 1)
    current_momentum_multiplier: float = 1.0  # Current ATR multiplier based on momentum

    # 4-Tier Partial Profit System
    exit_tiers: list[ExitTier] = field(default_factory=list)
    tier_4_trailing_sl_price: Optional[int] = None  # Trailing SL for Tier 4 (final 25%)
    tier_4_highest_price: Optional[int] = None  # Highest price since Tier 4 active (long)
    tier_4_lowest_price: Optional[int] = None  # Lowest price since Tier 4 active (short)

    # Exit tracking
    exit_reason: Optional[str] = None
    exit_time: Optional[datetime] = None
    exit_price: Optional[int] = None  # Final exit price (blended if partial exits)
    is_closed: bool = False

    def __post_init__(self):
        """Initialize tracking variables."""
        if self.highest_price_since_entry is None:
            self.highest_price_since_entry = self.entry_price
        if self.lowest_price_since_entry is None:
            self.lowest_price_since_entry = self.entry_price
        if self.original_quantity == 0:
            self.original_quantity = self.quantity
        if self.remaining_quantity == 0:
            self.remaining_quantity = self.quantity
        if self.highest_momentum_price is None:
            self.highest_momentum_price = self.entry_price
        if self.lowest_momentum_price is None:
            self.lowest_momentum_price = self.entry_price
        if self.tier_4_highest_price is None:
            self.tier_4_highest_price = self.entry_price
        if self.tier_4_lowest_price is None:
            self.tier_4_lowest_price = self.entry_price

    @property
    def blended_exit_price(self) -> Optional[int]:
        """Calculate blended exit price across all partial exits and final exit.

        Returns:
            Blended exit price in paisa, or None if no exits.
        """
        total_qty = 0
        total_value = 0

        for partial in self.partial_exits:
            total_qty += partial.quantity
            total_value += partial.exit_price * partial.quantity

        if self.exit_price and self.is_closed:
            final_qty = self.remaining_quantity
            total_qty += final_qty
            total_value += self.exit_price * final_qty

        if total_qty == 0:
            return None

        return int(total_value / total_qty)

    @property
    def total_realized_pnl(self) -> int:
        """Get total realized P&L including partial exits.

        Returns:
            Total realized P&L in paisa.
        """
        pnl = self.realized_pnl
        for partial in self.partial_exits:
            pnl += partial.realized_pnl
        return pnl

    def get_current_rr_ratio(self, current_price: int) -> float:
        """Calculate current risk-reward ratio.

        Args:
            current_price: Current market price in paisa.

        Returns:
            Current R:R ratio (e.g., 1.5 = 1.5:1).
        """
        if self.stop_loss_price is None:
            return 0.0

        risk = abs(self.entry_price - self.stop_loss_price)
        if risk == 0:
            return 0.0

        if self.side == OrderSide.BUY:
            reward = current_price - self.entry_price
        else:
            reward = self.entry_price - current_price

        return reward / risk

    def get_unrealized_pnl_for_quantity(self, price: int, qty: int) -> int:
        """Calculate unrealized P&L for a specific quantity.

        Args:
            price: Current price in paisa.
            qty: Quantity to calculate P&L for.

        Returns:
            Unrealized P&L in paisa.
        """
        if self.side == OrderSide.BUY:
            return (price - self.entry_price) * qty
        else:
            return (self.entry_price - price) * qty


class MomentumTrailingStop:
    """Momentum-based adaptive trailing stop-loss manager.

    Dynamically adjusts trailing stop distance based on price momentum.
    Trail slower when momentum is strong (let winners run), trail faster
    when momentum fades (protect profits).

    Momentum Calculation:
    - Uses Rate of Change (ROC) over N bars normalized to -1 to 1 scale
    - Alternative: Price distance from EMA

    Trailing Multiplier Logic:
    - Strong momentum (>0.7): 1.5x base ATR (trail slower)
    - Moderate momentum (0.3-0.7): 1.0x base ATR (normal)
    - Neutral momentum (-0.3-0.3): 0.7x base ATR (trail faster)
    - Negative momentum (<-0.3): 0.5x base ATR (trail aggressively)

    Breakeven Logic:
    - Move stop to entry + 1 tick once profit >= 0.5R
    - Never move stop further from entry than original SL

    Args:
        config: PositionConfig with momentum trailing stop settings
    """

    def __init__(self, config: PositionConfig):
        """Initialize the momentum trailing stop manager.

        Args:
            config: Configuration with momentum trailing stop parameters.
        """
        self._config = config
        self._tick_size = 5  # NSE tick size in paisa (0.05 INR)
        self._backtest_time: Optional[datetime] = None

    def set_backtest_time(self, t: datetime) -> None:
        """Set simulated time for backtest mode (overrides datetime.now)."""
        self._backtest_time = t

    def _now(self) -> datetime:
        """Return current time — backtest bar time if set, else real time."""
        if self._backtest_time is not None:
            return self._backtest_time
        return datetime.now(IST)

    def calculate_momentum(self, data: pd.DataFrame) -> float:
        """Calculate price momentum from recent price action.

        Uses Rate of Change (ROC) over the momentum period, normalized
        to a -1 to 1 scale. Alternative methods include MACD histogram
        slope or price vs EMA distance.

        Args:
            data: DataFrame with OHLCV data (at least 'close' column).
                  Should have at least momentum_period + 1 bars.

        Returns:
            Momentum value between -1 (strong down) and 1 (strong up).
            Returns 0.0 if insufficient data.
        """
        if len(data) < self._config.momentum_period + 1:
            return 0.0

        # Method 1: Rate of Change (ROC) normalized
        current_price = data["close"].iloc[-1]
        past_price = data["close"].iloc[-(self._config.momentum_period + 1)]

        if past_price == 0:
            return 0.0

        # Calculate ROC as percentage change
        roc = (current_price - past_price) / past_price

        # Normalize to -1 to 1 scale using arctan for bounded output
        # This prevents extreme values from distorting the signal
        import math

        normalized = (2 / math.pi) * math.atan(roc * 10)  # Scale factor of 10 for sensitivity

        # Clamp to [-1, 1] for safety
        return max(-1.0, min(1.0, normalized))

    def calculate_momentum_from_ema(self, data: pd.DataFrame, ema_period: int = 20) -> float:
        """Calculate momentum using price distance from EMA.

        Alternative momentum calculation that measures how far price
        is from its EMA, normalized by ATR for volatility adjustment.

        Args:
            data: DataFrame with OHLCV data.
            ema_period: EMA period for baseline. Default 20.

        Returns:
            Momentum value between -1 (far below EMA) and 1 (far above EMA).
        """
        if len(data) < ema_period:
            return 0.0

        try:
            import ta as ta_lib

            ema = ta_lib.trend.EMAIndicator(data["close"], window=ema_period).ema_indicator()
            atr = ta_lib.volatility.AverageTrueRange(data["high"], data["low"], data["close"], window=14).average_true_range()

            if ema is None or atr is None or atr.iloc[-1] == 0:
                return 0.0

            current_price = data["close"].iloc[-1]
            current_ema = ema.iloc[-1]
            current_atr = atr.iloc[-1]

            # Distance from EMA in ATR units
            distance = (current_price - current_ema) / current_atr

            # Normalize to -1 to 1 using tanh
            import math

            normalized = math.tanh(distance * 0.5)  # Scale factor for sensitivity

            return normalized

        except Exception:
            return 0.0

    def get_trailing_multiplier(self, momentum: float) -> float:
        """Get the ATR multiplier based on momentum value.

        Args:
            momentum: Momentum value from -1 to 1.

        Returns:
            Multiplier to apply to base ATR for trailing stop distance.
        """
        if momentum > self._config.momentum_strong_threshold:
            # Strong momentum - trail slower, let winners run
            return self._config.momentum_strong_multiplier
        elif momentum > self._config.momentum_moderate_threshold:
            # Moderate momentum - normal trailing
            return self._config.momentum_moderate_multiplier
        elif momentum >= -self._config.momentum_moderate_threshold:
            # Neutral momentum - trail faster to protect profits
            return self._config.momentum_neutral_multiplier
        else:
            # Negative momentum - trail aggressively
            return self._config.momentum_negative_multiplier

    def should_update_stop(self, position: ManagedPosition) -> bool:
        """Check if trailing stop should be updated based on time interval.

        Args:
            position: The position to check.

        Returns:
            True if enough time has passed since last update.
        """
        if position.last_momentum_update is None:
            return True

        elapsed = (self._now() - position.last_momentum_update).total_seconds()
        return elapsed >= self._config.momentum_update_interval_seconds

    def get_breakeven_price(self, position: ManagedPosition) -> int:
        """Calculate breakeven price (entry + 1 tick in favorable direction).

        Args:
            position: The position to calculate for.

        Returns:
            Breakeven stop price in paisa.
        """
        if position.side == OrderSide.BUY:
            # For long: entry + 1 tick
            return position.entry_price + self._tick_size
        else:
            # For short: entry - 1 tick
            return position.entry_price - self._tick_size

    def update_stop(
        self, position: ManagedPosition, current_data: pd.DataFrame
    ) -> Optional[int]:
        """Update the trailing stop based on momentum and price action.

        Args:
            position: The position to update.
            current_data: DataFrame with recent OHLCV data for momentum calc.

        Returns:
            New stop-loss price if updated, None if no update needed.
        """
        if not self.should_update_stop(position):
            return None

        if len(current_data) < self._config.momentum_period:
            return None

        current_price = current_data["close"].iloc[-1]
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
        effective_atr_distance = int(current_atr * self._config.momentum_trailing_base_atr_multiple * multiplier)

        # Update extreme prices for trailing calculation
        if position.side == OrderSide.BUY:
            if position.highest_momentum_price is None or current_price > position.highest_momentum_price:
                position.highest_momentum_price = current_price

            # Calculate new trailing stop
            new_stop = position.highest_momentum_price - effective_atr_distance

            # Check breakeven logic
            breakeven_price = self.get_breakeven_price(position)
            current_rr = position.get_current_rr_ratio(current_price)

            if current_rr >= self._config.momentum_breakeven_threshold_r and not position.breakeven_activated:
                # Move to breakeven
                if breakeven_price > new_stop:
                    new_stop = breakeven_price
                    position.breakeven_activated = True
                    logger.info(
                        f"Breakeven activated for {position.position_id}: "
                        f"SL moved to {breakeven_price/100:.2f}"
                    )
            elif position.breakeven_activated:
                # Ensure we don't go below breakeven once activated
                if new_stop < breakeven_price:
                    new_stop = breakeven_price

            # Ensure we never move stop further from entry than original SL
            if position.stop_loss_price and new_stop < position.stop_loss_price:
                new_stop = position.stop_loss_price

            # Only move stop in favorable direction (up for long)
            if position.momentum_trailing_sl_price is None or new_stop > position.momentum_trailing_sl_price:
                position.momentum_trailing_sl_price = new_stop
                position.last_momentum_update = self._now()
                logger.debug(
                    f"Momentum trailing SL updated for {position.position_id}: "
                    f"{new_stop/100:.2f} (momentum={momentum:.2f}, mult={multiplier:.2f})"
                )
                return new_stop

        else:  # SELL (short)
            if position.lowest_momentum_price is None or current_price < position.lowest_momentum_price:
                position.lowest_momentum_price = current_price

            # Calculate new trailing stop
            new_stop = position.lowest_momentum_price + effective_atr_distance

            # Check breakeven logic
            breakeven_price = self.get_breakeven_price(position)
            current_rr = position.get_current_rr_ratio(current_price)

            if current_rr >= self._config.momentum_breakeven_threshold_r and not position.breakeven_activated:
                # Move to breakeven
                if breakeven_price < new_stop:
                    new_stop = breakeven_price
                    position.breakeven_activated = True
                    logger.info(
                        f"Breakeven activated for {position.position_id}: "
                        f"SL moved to {breakeven_price/100:.2f}"
                    )
            elif position.breakeven_activated:
                # Ensure we don't go above breakeven once activated
                if new_stop > breakeven_price:
                    new_stop = breakeven_price

            # Ensure we never move stop further from entry than original SL
            if position.stop_loss_price and new_stop > position.stop_loss_price:
                new_stop = position.stop_loss_price

            # Only move stop in favorable direction (down for short)
            if position.momentum_trailing_sl_price is None or new_stop < position.momentum_trailing_sl_price:
                position.momentum_trailing_sl_price = new_stop
                position.last_momentum_update = self._now()
                logger.debug(
                    f"Momentum trailing SL updated for {position.position_id}: "
                    f"{new_stop/100:.2f} (momentum={momentum:.2f}, mult={multiplier:.2f})"
                )
                return new_stop

        return None

    def _estimate_atr_from_data(self, data: pd.DataFrame) -> Optional[int]:
        """Estimate ATR from recent data if entry ATR not available.

        Args:
            data: DataFrame with OHLCV data.

        Returns:
            Estimated ATR in paisa, or None if cannot calculate.
        """
        if len(data) < 2:
            return None

        try:
            # Simple estimation using recent high-low range
            recent_data = data.tail(14)  # Use last 14 bars
            ranges = recent_data["high"] - recent_data["low"]
            avg_range = int(ranges.mean())
            return avg_range if avg_range > 0 else None
        except Exception:
            return None

    def get_momentum_status(self, position: ManagedPosition) -> dict:
        """Get current momentum trailing stop status for reporting.

        Args:
            position: The position to get status for.

        Returns:
            Dictionary with momentum status information.
        """
        return {
            "momentum": position.current_momentum,
            "multiplier": position.current_momentum_multiplier,
            "momentum_trailing_sl": position.momentum_trailing_sl_price,
            "breakeven_activated": position.breakeven_activated,
            "last_update": position.last_momentum_update.isoformat() if position.last_momentum_update else None,
            "highest_price": position.highest_momentum_price,
            "lowest_price": position.lowest_momentum_price,
        }


class PositionManager:
    """Manages open positions, P&L tracking, and exit conditions.

    The PositionManager tracks all open positions, updates unrealized P&L
    on each price tick, checks stop-loss/target/trailing SL conditions,
    and handles time-based exits including the mandatory intraday square-off.

    Smart Exit Features:
    - Partial profit booking at 1:1 R:R
    - ATR-based adaptive stop loss
    - Volatility-based target adjustment
    - Options time-decay exits
    - Blended P&L tracking for partial exits

    Attributes:
        broker: The broker for executing exit orders
        risk_manager: Optional risk manager for P&L and position sync
        config: Position management configuration
        positions: Dictionary of managed positions by position_id
        on_position_close: Optional callback when position closes
    """

    def __init__(
        self,
        broker: BaseBroker,
        risk_manager: Optional["RiskManager"] = None,
        config: Optional[PositionConfig] = None,
        on_position_close: Optional[Callable[[ManagedPosition], None]] = None,
        partial_profit_manager: Optional[PartialProfitManager] = None,
        trade_logger: Optional[Any] = None,
        rl_exit_agent: Optional[Any] = None,
    ):
        """Initialize the PositionManager.

        Args:
            broker: Broker for executing exit orders
            risk_manager: Optional risk manager for P&L and position tracking sync
            config: Position management configuration
            on_position_close: Optional callback when position closes
            partial_profit_manager: Optional 4-tier partial profit manager
            trade_logger: Optional TradeLogger for database persistence
            rl_exit_agent: Optional RLExitAgent for learned exit decisions.
        """
        self._broker = broker
        self._risk_manager = risk_manager
        self._config = config or PositionConfig()
        self._on_position_close = on_position_close
        self._partial_profit_manager = partial_profit_manager
        self._trade_logger = trade_logger
        self._rl_exit_agent = rl_exit_agent
        self._backtest_time: Optional[datetime] = None  # Set by harness for bar-accurate timestamps
        self._lock = threading.Lock()
        self._positions: dict[str, ManagedPosition] = {}
        self._instrument_to_position: dict[str, str] = {}  # instrument_key -> position_id

        # ATR history for volatility regime detection
        self._atr_history: dict[str, list[int]] = {}  # instrument_key -> list of ATR values
        self._atr_history_max_size = 20  # Keep last 20 ATR values

        # Momentum-based trailing stop manager
        self._momentum_trailing_stop: Optional[MomentumTrailingStop] = None
        if self._config.trailing_stop_type == "momentum":
            self._momentum_trailing_stop = MomentumTrailingStop(self._config)
            logger.info("PositionManager initialized with momentum-based trailing stop")

        # Price history for momentum calculation
        self._price_history: dict[str, pd.DataFrame] = {}  # instrument_key -> OHLCV DataFrame
        self._price_history_max_bars = 50  # Keep last 50 bars for momentum calc

        logger.info("PositionManager initialized with smart exit logic")

    def set_rl_exit_agent(self, agent: Any) -> None:
        """Attach a trained RLExitAgent after construction."""
        self._rl_exit_agent = agent

    def set_backtest_time(self, t: datetime) -> None:
        """Set simulated time for backtest mode (overrides datetime.now)."""
        self._backtest_time = t

    def _now(self) -> datetime:
        """Return current time — backtest bar time if set, else real time."""
        if self._backtest_time is not None:
            return self._backtest_time
        return datetime.now(IST)

    def add_position(
        self,
        instrument_key: str,
        strategy_id: str,
        side: OrderSide,
        quantity: int,
        entry_price: int,
        product_type: ProductType = ProductType.MIS,
        stop_loss_pct: Optional[float] = None,
        target_pct: Optional[float] = None,
        stop_loss_price: Optional[int] = None,
        target_price: Optional[int] = None,
        entry_atr: Optional[int] = None,
        volatility_regime: str = "normal",
        is_option: bool = False,
        option_dte: Optional[int] = None,
        option_theta: Optional[float] = None,
        option_premium_at_entry: Optional[int] = None,
        underlying_instrument_key: Optional[str] = None,
        underlying_entry_price: Optional[int] = None,
        emergency_exit_move_pct: Optional[float] = None,
    ) -> ManagedPosition:
        """Add a new position to track.

        Args:
            instrument_key: The instrument identifier
            strategy_id: Strategy that opened this position
            side: BUY for long, SELL for short
            quantity: Position size
            entry_price: Entry price in paisa
            product_type: MIS, CNC, NRML, etc.
            stop_loss_pct: Override default SL percentage (used if stop_loss_price not provided)
            target_pct: Override default target percentage (used if target_price not provided)
            stop_loss_price: Absolute stop loss price in paisa (takes precedence over pct)
            target_price: Absolute target price in paisa (takes precedence over pct)
            entry_atr: ATR(14) value at entry time (for ATR-based SL)
            volatility_regime: "low", "normal", or "high" volatility
            is_option: Whether this is an options position
            option_dte: Days to expiry for options
            option_theta: Daily theta decay for options
            option_premium_at_entry: Premium at entry for options (paisa)

        Returns:
            The created ManagedPosition
        """
        with self._lock:
            # Check if position already exists
            if instrument_key in self._instrument_to_position:
                existing_id = self._instrument_to_position[instrument_key]
                logger.warning(f"Position already exists for {instrument_key}: {existing_id}")
                return self._positions[existing_id]

            # Determine ATR multiplier based on volatility regime
            atr_multiplier = self._get_atr_multiplier(volatility_regime)

            # Calculate stop-loss using ATR if enabled and ATR provided
            if stop_loss_price is None and self._config.use_atr_based_sl and entry_atr is not None:
                if side == OrderSide.BUY:
                    stop_loss_price = int(entry_price - (entry_atr * atr_multiplier))
                else:  # SELL (short)
                    stop_loss_price = int(entry_price + (entry_atr * atr_multiplier))
                logger.info(
                    f"ATR-based SL calculated for {instrument_key}: "
                    f"Entry={entry_price}, ATR={entry_atr}, Multiplier={atr_multiplier}, "
                    f"SL={stop_loss_price}"
                )
            elif stop_loss_price is None:
                # Use percentage-based SL
                sl_pct = stop_loss_pct if stop_loss_pct is not None else self._config.stop_loss_pct
                if side == OrderSide.BUY:
                    stop_loss_price = int(entry_price * (1 - sl_pct))
                else:  # SELL (short)
                    stop_loss_price = int(entry_price * (1 + sl_pct))

            # Calculate target with volatility adjustment
            if target_price is None:
                tgt_pct = target_pct if target_pct is not None else self._config.target_pct

                # Adjust target based on volatility regime
                if self._config.enable_volatility_target_adjustment:
                    if volatility_regime == "high":
                        tgt_pct *= (1 - self._config.high_volatility_target_reduction)
                        logger.info(f"High volatility: Target reduced by {self._config.high_volatility_target_reduction*100}%")
                    elif volatility_regime == "low":
                        tgt_pct *= (1 + self._config.low_volatility_target_extension)
                        logger.info(f"Low volatility: Target extended by {self._config.low_volatility_target_extension*100}%")

                if side == OrderSide.BUY:
                    target_price = int(entry_price * (1 + tgt_pct))
                else:  # SELL (short)
                    target_price = int(entry_price * (1 - tgt_pct))

            position = ManagedPosition(
                instrument_key=instrument_key,
                strategy_id=strategy_id,
                side=side,
                quantity=quantity,
                entry_price=entry_price,
                entry_time=self._now(),
                entry_bar_time=self._backtest_time,  # set in backtest; None in live
                product_type=product_type,
                stop_loss_price=stop_loss_price,
                target_price=target_price,
                current_price=entry_price,
                entry_atr=entry_atr,
                volatility_regime=volatility_regime,
                atr_multiplier_used=atr_multiplier,
                is_option=is_option,
                option_dte=option_dte,
                option_theta=option_theta,
                option_premium_at_entry=option_premium_at_entry,
                underlying_instrument_key=underlying_instrument_key,
                underlying_entry_price=underlying_entry_price,
                emergency_exit_move_pct=emergency_exit_move_pct,
                original_quantity=quantity,
                remaining_quantity=quantity,
            )

            self._positions[position.position_id] = position
            # Support multiple open positions per instrument (e.g. two legs on same strike)
            existing = self._instrument_to_position.get(instrument_key)
            if existing and not isinstance(existing, list):
                self._instrument_to_position[instrument_key] = [existing, position.position_id]
            elif isinstance(existing, list):
                existing.append(position.position_id)
            else:
                self._instrument_to_position[instrument_key] = position.position_id

            # Initialize 4-tier partial profit system
            if self._partial_profit_manager and stop_loss_price is not None:
                direction = PositionDirection.LONG if side == OrderSide.BUY else PositionDirection.SHORT
                tiers = self._partial_profit_manager.initialize_tiers(
                    entry=entry_price,
                    stop_loss=stop_loss_price,
                    quantity=quantity,
                    direction=direction,
                )
                position.exit_tiers = tiers
                logger.info(
                    f"Initialized {len(tiers)} exit tiers for {instrument_key}: "
                    f"T1@1R, T2@2R, T3@3R, T4=Trailing"
                )

            logger.info(
                f"Position added: {position.position_id} | {strategy_id} | "
                f"{side.value} {quantity} {instrument_key} @ {entry_price/100:.2f} | "
                f"SL: {stop_loss_price/100:.2f} | Target: {target_price/100:.2f} | "
                f"ATR: {entry_atr} | VolRegime: {volatility_regime}"
            )

            # Persist to database
            if self._trade_logger:
                try:
                    self._trade_logger.log_position({
                        "strategy": position.strategy_id,
                        "instrument_key": position.instrument_key,
                        "side": position.side.value,
                        "entry_price": position.entry_price,
                        "quantity": position.quantity,
                        "status": "open",
                        "opened_at": position.entry_time,
                    })
                except Exception as e:
                    logger.error(f"Failed to log position to database: {e}")

        # Sync with risk manager after releasing lock
        self._sync_risk_manager()

        return position

    def _get_atr_multiplier(self, volatility_regime: str) -> float:
        """Get ATR multiplier based on volatility regime.

        Args:
            volatility_regime: "low", "normal", or "high".

        Returns:
            ATR multiplier for stop loss calculation.
        """
        if volatility_regime == "low":
            return self._config.atr_multiplier_low_volatility
        elif volatility_regime == "high":
            return self._config.atr_multiplier_high_volatility
        else:
            return self._config.atr_multiplier_normal_volatility

    def update_atr_history(self, instrument_key: str, atr_value: int) -> None:
        """Update ATR history for volatility regime detection.

        Args:
            instrument_key: The instrument identifier.
            atr_value: Current ATR(14) value in paisa.
        """
        if instrument_key not in self._atr_history:
            self._atr_history[instrument_key] = []

        self._atr_history[instrument_key].append(atr_value)

        # Keep only recent history
        if len(self._atr_history[instrument_key]) > self._atr_history_max_size:
            self._atr_history[instrument_key].pop(0)

    def get_volatility_regime(self, instrument_key: str, current_atr: int) -> str:
        """Determine volatility regime based on ATR history.

        Args:
            instrument_key: The instrument identifier.
            current_atr: Current ATR(14) value in paisa.

        Returns:
            "low", "normal", or "high" volatility regime.
        """
        history = self._atr_history.get(instrument_key, [])
        if len(history) < 10:
            return "normal"  # Not enough data

        avg_atr = sum(history) / len(history)
        if avg_atr == 0:
            return "normal"

        ratio = current_atr / avg_atr

        if ratio > self._config.high_volatility_atr_threshold:
            return "high"
        elif ratio < self._config.low_volatility_atr_threshold:
            return "low"
        else:
            return "normal"

    def update_price_bar(
        self,
        instrument_key: str,
        open_price: int,
        high_price: int,
        low_price: int,
        close_price: int,
        volume: int,
        timestamp: Optional[datetime] = None,
    ) -> None:
        """Update price history with a new OHLCV bar for momentum calculation.

        This method should be called with bar data (e.g., 1-minute bars)
        to provide sufficient data for momentum-based trailing stops.

        Args:
            instrument_key: The instrument identifier.
            open_price: Open price in paisa.
            high_price: High price in paisa.
            low_price: Low price in paisa.
            close_price: Close price in paisa.
            volume: Volume for the bar.
            timestamp: Bar timestamp (defaults to now).
        """
        if timestamp is None:
            timestamp = self._now()

        # Create bar data
        bar_data = {
            "open": open_price,
            "high": high_price,
            "low": low_price,
            "close": close_price,
            "volume": volume,
        }

        with self._lock:
            if instrument_key not in self._price_history:
                # Initialize DataFrame for this instrument
                self._price_history[instrument_key] = pd.DataFrame(
                    [bar_data], index=[timestamp]
                )
            else:
                # Append new bar
                df = self._price_history[instrument_key]
                new_row = pd.DataFrame([bar_data], index=[timestamp])
                self._price_history[instrument_key] = pd.concat([df, new_row])

                # Keep only recent history
                if len(self._price_history[instrument_key]) > self._price_history_max_bars:
                    self._price_history[instrument_key] = self._price_history[
                        instrument_key
                    ].tail(self._price_history_max_bars)

    def _update_momentum_trailing_stop(self, position: ManagedPosition, price: int) -> None:
        """Update momentum-based trailing stop if enabled.

        Args:
            position: The position to update.
            price: Current price in paisa.
        """
        if self._config.trailing_stop_type != "momentum":
            return

        if self._momentum_trailing_stop is None:
            return

        instrument_key = position.instrument_key
        if instrument_key not in self._price_history:
            return

        price_data = self._price_history[instrument_key]
        if len(price_data) < self._config.momentum_period + 1:
            return  # Not enough data for momentum calculation

        # Update momentum trailing stop
        new_sl = self._momentum_trailing_stop.update_stop(position, price_data)

        if new_sl is not None:
            # Update the position's stop-loss if momentum trailing stop is better
            if position.momentum_trailing_sl_price is not None:
                if position.side == OrderSide.BUY:
                    # For long: use higher of the two stops
                    if position.momentum_trailing_sl_price > position.stop_loss_price:
                        old_sl = position.stop_loss_price
                        position.stop_loss_price = position.momentum_trailing_sl_price
                        logger.info(
                            f"Momentum trailing SL updated for {position.position_id}: "
                            f"{old_sl/100:.2f} -> {position.stop_loss_price/100:.2f} "
                            f"(momentum={position.current_momentum:.2f})"
                        )
                else:
                    # For short: use lower of the two stops
                    if position.momentum_trailing_sl_price < position.stop_loss_price:
                        old_sl = position.stop_loss_price
                        position.stop_loss_price = position.momentum_trailing_sl_price
                        logger.info(
                            f"Momentum trailing SL updated for {position.position_id}: "
                            f"{old_sl/100:.2f} -> {position.stop_loss_price/100:.2f} "
                            f"(momentum={position.current_momentum:.2f})"
                        )

    def on_underlying_tick(self, underlying_key: str, underlying_price: int) -> list[ManagedPosition]:
        """Check emergency-exit on all option positions tracking this underlying.

        Each option position may carry an `emergency_exit_move_pct` and the
        underlying price at entry. When the live underlying price moves
        more than that pct away (in either direction), the option leg is
        force-closed regardless of its own SL/target. Used by short
        straddles to cap tail risk on directional underlying moves.

        Args:
            underlying_key: Index instrument_key that just ticked (e.g.
                "NSE_INDEX|Nifty Bank").
            underlying_price: Current underlying price in paisa.

        Returns:
            List of option positions that were force-closed.
        """
        closed: list[ManagedPosition] = []
        to_close: list[tuple[ManagedPosition, int, str]] = []
        with self._lock:
            for position in list(self._positions.values()):
                if position.is_closed:
                    continue
                if not position.is_option:
                    continue
                if position.underlying_instrument_key != underlying_key:
                    continue
                if not position.underlying_entry_price or not position.emergency_exit_move_pct:
                    continue
                move_pct = abs(underlying_price - position.underlying_entry_price) / position.underlying_entry_price * 100.0
                if move_pct >= position.emergency_exit_move_pct:
                    logger.warning(
                        "EMERGENCY_EXIT | {} | {} moved {:.2f}% (>= {:.2f}%) — force closing option {}",
                        position.position_id,
                        underlying_key,
                        move_pct,
                        position.emergency_exit_move_pct,
                        position.instrument_key,
                    )
                    # Use the option's last known price as exit reference
                    exit_px = position.current_price or position.entry_price
                    to_close.append((position, exit_px, "EMERGENCY_UNDERLYING_MOVE"))

        for position, exit_px, reason in to_close:
            if self._close_position(position, exit_px, reason):
                closed.append(position)
        return closed

    def on_tick(self, instrument_key: str, price: int) -> list[ManagedPosition]:
        """Process a price tick and check exit conditions.

        Args:
            instrument_key: The instrument that ticked
            price: Current price in paisa

        Returns:
            List of positions that were closed
        """
        closed_positions = []

        with self._lock:
            if instrument_key not in self._instrument_to_position:
                return closed_positions

            position_id = self._instrument_to_position[instrument_key]
            position = self._positions[position_id]

            if position.is_closed:
                return closed_positions

            # --- SAME BAR EXIT GUARD ---
            # Prevent exiting on the exact same minute-bar as entry
            if self._backtest_time and position.entry_time:
                current_min = self._backtest_time.replace(second=0, microsecond=0)
                entry_min = position.entry_time.replace(second=0, microsecond=0)
                if current_min == entry_min:
                    # Still update price and P&L, but skip exit checks to enforce a 1-bar cooldown
                    position.current_price = price
                    position.unrealized_pnl = position.get_unrealized_pnl_for_quantity(
                        price, position.remaining_quantity
                    )
                    return closed_positions

            # Update current price
            position.current_price = price

            # Calculate unrealized P&L for remaining quantity
            position.unrealized_pnl = position.get_unrealized_pnl_for_quantity(
                price, position.remaining_quantity
            )

            # Update trailing SL tracking
            self._update_trailing_sl(position, price)

            # Update momentum-based trailing stop if enabled
            if self._config.trailing_stop_type == "momentum":
                position_for_momentum = position
            else:
                position_for_momentum = None

            # Check for 4-tier partial profit exits
            tier_exits_to_execute: list[TierExitResult] = []
            if position.exit_tiers and self._partial_profit_manager:
                # Create position snapshot for tier checking
                direction = PositionDirection.LONG if position.side == OrderSide.BUY else PositionDirection.SHORT
                snapshot = PositionSnapshot(
                    instrument_key=position.instrument_key,
                    direction=direction,
                    entry_price=position.entry_price,
                    stop_loss_price=position.stop_loss_price or position.entry_price,
                    current_price=price,
                    total_quantity=position.original_quantity,
                    remaining_quantity=position.remaining_quantity,
                )
                tier_exits_to_execute = self._partial_profit_manager.check_tiers(snapshot, position.exit_tiers)

            # Check for legacy partial profit booking (if 4-tier not enabled)
            position_to_partial = None
            current_price_for_partial = None
            if not position.exit_tiers and self._should_book_partial_profit(position, price):
                position_to_partial = position
                current_price_for_partial = price

            # Check for Tier 4 trailing stop
            tier_4_exit = False
            if position.exit_tiers and self._partial_profit_manager:
                tier_4_exit = self._check_tier_4_trailing_stop(position, price)

        # Update momentum trailing stop outside of lock (may access price history)
        if position_for_momentum is not None:
            self._update_momentum_trailing_stop(position_for_momentum, price)

        # Execute tier exits outside of lock
        for tier_exit in tier_exits_to_execute:
            self._execute_tier_exit(position, tier_exit, price)

        # Execute legacy partial exit outside of lock
        if position_to_partial is not None:
            self._execute_partial_exit(position_to_partial, current_price_for_partial, "PARTIAL_PROFIT_1R")

        # Re-acquire lock to check other exit conditions
        with self._lock:
            if instrument_key not in self._instrument_to_position:
                return closed_positions

            position_id = self._instrument_to_position[instrument_key]
            position = self._positions[position_id]

            if position.is_closed:
                return closed_positions

            # Check if Tier 4 trailing stop was hit
            if tier_4_exit:
                position_to_close = position
                reason = "TIER_4_TRAILING_STOP"
            else:
                # RL exit agent: consult if trained; overrides rule-based hold.
                rl_exit_triggered = False
                if self._rl_exit_agent is not None and self._rl_exit_agent.is_trained:
                    try:
                        entry_px = position.entry_price or 1
                        pnl_pct = (position.unrealized_pnl or 0) / entry_px
                        sl_dist = abs((position.stop_loss_price or entry_px) - price) / max(entry_px, 1)
                        tgt_dist = abs((position.target_price or entry_px) - price) / max(entry_px, 1)
                        trailing_on = 1.0 if position.trailing_sl_activated else 0.0
                        peak_gain = max(0.0, pnl_pct)
                        rl_state = {
                            "pnl_pct": float(pnl_pct),
                            "bars_held_norm": 0.5,
                            "momentum": 0.0,
                            "vol_regime": 1.0,
                            "dist_to_sl": float(sl_dist),
                            "dist_to_target": float(tgt_dist),
                            "trailing_active": trailing_on,
                            "peak_gain_norm": float(peak_gain),
                        }
                        rl_action = self._rl_exit_agent.get_action(rl_state)
                        if rl_action == "EXIT":
                            rl_exit_triggered = True
                            position_to_close = position
                            reason = "RL_EXIT"
                        elif rl_action == "TIGHTEN_SL" and position.stop_loss_price is not None:
                            # Tighten SL by 30% toward current price
                            if position.side.value == "BUY":
                                new_sl = position.stop_loss_price + int((price - position.stop_loss_price) * 0.3)
                                position.stop_loss_price = max(position.stop_loss_price, new_sl)
                            else:
                                new_sl = position.stop_loss_price - int((position.stop_loss_price - price) * 0.3)
                                position.stop_loss_price = min(position.stop_loss_price, new_sl)
                    except Exception as _rl_exc:
                        logger.debug("rl_exit_agent.get_action failed: {}", _rl_exc)

                if not rl_exit_triggered:
                    # Check other exit conditions
                    exit_reason = self._check_exit_conditions(position, price)
                    if exit_reason:
                        position_to_close = position
                        reason = exit_reason
                    else:
                        return closed_positions

        # Close position outside of lock
        if self._close_position(position_to_close, price, reason):
            closed_positions.append(position_to_close)

        return closed_positions

    def _should_book_partial_profit(self, position: ManagedPosition, price: int) -> bool:
        """Check if partial profit should be booked.

        Args:
            position: The position to check.
            price: Current price.

        Returns:
            True if partial profit should be booked.
        """
        if not self._config.enable_partial_profit_booking:
            return False

        if position.partial_booking_done:
            return False

        if position.remaining_quantity <= 1:
            return False  # Can't split further

        # Check if 1:1 R:R reached
        current_rr = position.get_current_rr_ratio(price)
        return current_rr >= self._config.partial_booking_rr_ratio

    def _execute_partial_exit(
        self, position: ManagedPosition, price: int, reason: str
    ) -> bool:
        """Execute a partial exit (profit booking).

        Args:
            position: The position to partially exit.
            price: Current price.
            reason: Exit reason.

        Returns:
            True if partial exit was successful.
        """
        try:
            # Calculate quantity to exit
            exit_qty = int(position.remaining_quantity * self._config.partial_booking_ratio)
            if exit_qty < 1:
                exit_qty = 1

            # Ensure we don't exit more than remaining
            if exit_qty >= position.remaining_quantity:
                exit_qty = position.remaining_quantity - 1
                if exit_qty < 1:
                    return False

            # Determine exit side
            exit_side = OrderSide.SELL if position.side == OrderSide.BUY else OrderSide.BUY

            # Place exit order
            exit_order = Order(
                instrument_key=position.instrument_key,
                side=exit_side,
                order_type=OrderType.MARKET,
                product_type=position.product_type,
                quantity=exit_qty,
                strategy_id=position.strategy_id,
                tag=f"EXIT_{reason}",
            )

            response = self._broker.place_order(exit_order)

            if (getattr(response.status, 'value', response.status)) in ("COMPLETE", "OPEN", "PARTIAL_FILL"):
                # Calculate realized P&L for this partial exit
                realized_pnl = position.get_unrealized_pnl_for_quantity(price, exit_qty)

                # Record partial exit
                partial_record = PartialExitRecord(
                    exit_price=price,
                    quantity=exit_qty,
                    exit_time=self._now(),
                    exit_reason=reason,
                    realized_pnl=realized_pnl,
                )

                with self._lock:
                    position.partial_exits.append(partial_record)
                    position.remaining_quantity -= exit_qty
                    position.partial_booking_done = True

                    # Widen stop-loss for remaining position
                    self._widen_stop_loss_after_partial(position)

                logger.info(
                    f"Partial exit executed: {position.position_id} | "
                    f"Qty: {exit_qty} @ {price/100:.2f} | P&L: {realized_pnl/100:.2f} | "
                    f"Remaining: {position.remaining_quantity}"
                )

                # Sync with risk manager
                self._sync_risk_manager()

                return True
            else:
                logger.error(f"Failed to execute partial exit for {position.position_id}: {response.message}")
                return False

        except Exception as e:
            logger.exception(f"Error executing partial exit for {position.position_id}: {e}")
            return False

    def _execute_tier_exit(
        self,
        position: ManagedPosition,
        tier_exit: TierExitResult,
        price: int,
    ) -> bool:
        """Execute a partial exit for a specific tier.

        Args:
            position: The position to exit from.
            tier_exit: The tier exit result with quantity and tier info.
            price: Current market price in paisa.

        Returns:
            True if tier exit was successful.
        """
        try:
            exit_qty = tier_exit.quantity
            if exit_qty <= 0:
                logger.warning(f"Tier {tier_exit.tier} exit quantity is 0, skipping")
                return False

            # Ensure we don't exit more than remaining
            if exit_qty > position.remaining_quantity:
                exit_qty = position.remaining_quantity

            if exit_qty <= 0:
                return False

            # Determine exit side
            exit_side = OrderSide.SELL if position.side == OrderSide.BUY else OrderSide.BUY

            # Place exit order
            exit_order = Order(
                instrument_key=position.instrument_key,
                side=exit_side,
                order_type=OrderType.MARKET,
                product_type=position.product_type,
                quantity=exit_qty,
                strategy_id=position.strategy_id,
                tag=f"EXIT_TIER_{tier_exit.tier}",
            )

            response = self._broker.place_order(exit_order)

            if (getattr(response.status, 'value', response.status)) in ("COMPLETE", "OPEN", "PARTIAL_FILL"):
                # Calculate realized P&L for this tier exit
                realized_pnl = position.get_unrealized_pnl_for_quantity(price, exit_qty)

                # Mark tier as exited
                tier = next((t for t in position.exit_tiers if t.tier == tier_exit.tier), None)
                if tier and self._partial_profit_manager:
                    self._partial_profit_manager.mark_tier_exited(tier, price, realized_pnl)

                # Record partial exit
                partial_record = PartialExitRecord(
                    exit_price=price,
                    quantity=exit_qty,
                    exit_time=self._now(),
                    exit_reason=f"TIER_{tier_exit.tier}_{tier_exit.exit_reason}",
                    realized_pnl=realized_pnl,
                )

                with self._lock:
                    position.partial_exits.append(partial_record)
                    position.remaining_quantity -= exit_qty

                logger.info(
                    f"PARTIAL_EXIT | Tier {tier_exit.tier} | {position.position_id} | "
                    f"{position.instrument_key} | Qty: {exit_qty} @ ₹{price/100:.2f} | "
                    f"P&L: ₹{realized_pnl/100:.2f} | Remaining: {position.remaining_quantity}"
                )

                # Check if all tiers are exited
                if self._partial_profit_manager and self._partial_profit_manager.is_complete(position.exit_tiers):
                    with self._lock:
                        position.is_closed = True
                        position.exit_time = self._now()
                        position.exit_price = price
                        position.exit_reason = "ALL_TIERS_EXITED"
                        if position.instrument_key in self._instrument_to_position:
                            del self._instrument_to_position[position.instrument_key]
                    logger.info(f"Position fully closed via tier exits: {position.position_id}")

                # Sync with risk manager
                self._sync_risk_manager()

                return True
            else:
                logger.error(f"Failed to execute tier {tier_exit.tier} exit for {position.position_id}: {response.message}")
                return False

        except Exception as e:
            logger.exception(f"Error executing tier exit for {position.position_id}: {e}")
            return False

    def _check_tier_4_trailing_stop(self, position: ManagedPosition, price: int) -> bool:
        """Check if Tier 4 trailing stop has been hit.

        Tier 4 uses a momentum-based trailing stop that activates after 1:1 R:R.
        It trails the highest (for long) or lowest (for short) price by the
        original stop loss distance.

        Args:
            position: The position to check.
            price: Current market price in paisa.

        Returns:
            True if Tier 4 trailing stop was hit.
        """
        if not position.exit_tiers or not self._partial_profit_manager:
            return False

        # Check if Tier 4 is the only remaining tier
        unexited_tiers = self._partial_profit_manager.get_unexited_tiers(position.exit_tiers)
        if len(unexited_tiers) != 1 or unexited_tiers[0].tier != 4:
            return False  # Tier 4 is not the last remaining tier

        tier_4 = unexited_tiers[0]

        # Calculate stop loss distance (risk)
        if position.stop_loss_price is None:
            return False

        sl_distance = abs(position.entry_price - position.stop_loss_price)
        if sl_distance <= 0:
            return False

        # Update extreme prices for Tier 4
        with self._lock:
            if position.side == OrderSide.BUY:
                # Update highest price
                if position.tier_4_highest_price is None or price > position.tier_4_highest_price:
                    position.tier_4_highest_price = price

                # Calculate current R:R
                current_rr = (price - position.entry_price) / sl_distance if sl_distance > 0 else 0

                # Trailing stop activates after 1:1 R:R
                if current_rr >= 1.0:
                    # Calculate trailing stop price (trail below highest by SL distance)
                    trailing_sl = position.tier_4_highest_price - sl_distance

                    # Don't trail below entry
                    trailing_sl = max(trailing_sl, position.entry_price)

                    # Update stored trailing SL
                    if position.tier_4_trailing_sl_price is None or trailing_sl > position.tier_4_trailing_sl_price:
                        old_sl = position.tier_4_trailing_sl_price
                        position.tier_4_trailing_sl_price = trailing_sl
                        if old_sl != trailing_sl:
                            logger.debug(
                                f"Tier 4 trailing SL updated for {position.position_id}: "
                                f"{trailing_sl/100:.2f} (highest={position.tier_4_highest_price/100:.2f})"
                            )

                    # Check if trailing stop is hit
                    if position.tier_4_trailing_sl_price and price <= position.tier_4_trailing_sl_price:
                        logger.info(
                            f"Tier 4 trailing stop hit for {position.position_id}: "
                            f"price={price/100:.2f}, trailing_sl={position.tier_4_trailing_sl_price/100:.2f}"
                        )
                        return True
            else:  # SELL (short)
                # Update lowest price
                if position.tier_4_lowest_price is None or price < position.tier_4_lowest_price:
                    position.tier_4_lowest_price = price

                # Calculate current R:R
                current_rr = (position.entry_price - price) / sl_distance if sl_distance > 0 else 0

                # Trailing stop activates after 1:1 R:R
                if current_rr >= 1.0:
                    # Calculate trailing stop price (trail above lowest by SL distance)
                    trailing_sl = position.tier_4_lowest_price + sl_distance

                    # Don't trail above entry
                    trailing_sl = min(trailing_sl, position.entry_price)

                    # Update stored trailing SL
                    if position.tier_4_trailing_sl_price is None or trailing_sl < position.tier_4_trailing_sl_price:
                        old_sl = position.tier_4_trailing_sl_price
                        position.tier_4_trailing_sl_price = trailing_sl
                        if old_sl != trailing_sl:
                            logger.debug(
                                f"Tier 4 trailing SL updated for {position.position_id}: "
                                f"{trailing_sl/100:.2f} (lowest={position.tier_4_lowest_price/100:.2f})"
                            )

                    # Check if trailing stop is hit
                    if position.tier_4_trailing_sl_price and price >= position.tier_4_trailing_sl_price:
                        logger.info(
                            f"Tier 4 trailing stop hit for {position.position_id}: "
                            f"price={price/100:.2f}, trailing_sl={position.tier_4_trailing_sl_price/100:.2f}"
                        )
                        return True

        return False

    def _widen_stop_loss_after_partial(self, position: ManagedPosition) -> None:
        """Widen stop-loss after partial profit booking.

        Args:
            position: The position to adjust.
        """
        if position.stop_loss_price is None:
            return

        # Calculate original SL distance
        if position.side == OrderSide.BUY:
            original_sl_distance = position.entry_price - position.stop_loss_price
            new_sl_distance = int(original_sl_distance * self._config.trailing_sl_after_partial_multiplier)
            new_sl_price = position.entry_price - new_sl_distance

            # For a LONG: wider SL = lower price (further below entry)
            if new_sl_price < position.stop_loss_price:
                logger.info(
                    f"Widening SL for {position.position_id} after partial: "
                    f"{position.stop_loss_price/100:.2f} -> {new_sl_price/100:.2f}"
                )
                position.stop_loss_price = new_sl_price
        else:
            original_sl_distance = position.stop_loss_price - position.entry_price
            new_sl_distance = int(original_sl_distance * self._config.trailing_sl_after_partial_multiplier)
            new_sl_price = position.entry_price + new_sl_distance

            # For a SHORT: wider SL = higher price (further above entry)
            if new_sl_price > position.stop_loss_price:
                logger.info(
                    f"Widening SL for {position.position_id} after partial: "
                    f"{position.stop_loss_price/100:.2f} -> {new_sl_price/100:.2f}"
                )
                position.stop_loss_price = new_sl_price

    def _update_trailing_sl(self, position: ManagedPosition, price: int) -> None:
        """Update trailing stop-loss tracking.

        Args:
            position: The position to update
            price: Current price
        """
        if not self._config.enable_trailing_sl:
            return

        entry_value = position.entry_price * position.remaining_quantity
        current_value = price * position.remaining_quantity

        if position.side == OrderSide.BUY:
            # Update highest price
            if price > position.highest_price_since_entry:
                position.highest_price_since_entry = price

            # Calculate profit percentage
            profit_pct = (current_value - entry_value) / entry_value if entry_value > 0 else 0

            # Activate trailing SL if profit threshold reached
            if profit_pct >= self._config.trailing_sl_activation_pct:
                if not position.trailing_sl_activated:
                    position.trailing_sl_activated = True
                    logger.info(f"Trailing SL activated for {position.position_id}")

                # Update trailing SL price
                new_trailing_sl = int(position.highest_price_since_entry * (1 - self._config.trailing_sl_pct))
                if position.trailing_sl_price is None or new_trailing_sl > position.trailing_sl_price:
                    position.trailing_sl_price = new_trailing_sl
                    logger.debug(f"Trailing SL updated for {position.position_id}: {new_trailing_sl/100:.2f}")

        else:  # SELL (short)
            # Update lowest price
            if price < position.lowest_price_since_entry:
                position.lowest_price_since_entry = price

            # Calculate profit percentage
            profit_pct = (entry_value - current_value) / entry_value if entry_value > 0 else 0

            # Activate trailing SL if profit threshold reached
            if profit_pct >= self._config.trailing_sl_activation_pct:
                if not position.trailing_sl_activated:
                    position.trailing_sl_activated = True
                    logger.info(f"Trailing SL activated for {position.position_id}")

                # Update trailing SL price (for short, SL is above entry)
                new_trailing_sl = int(position.lowest_price_since_entry * (1 + self._config.trailing_sl_pct))
                if position.trailing_sl_price is None or new_trailing_sl < position.trailing_sl_price:
                    position.trailing_sl_price = new_trailing_sl
                    logger.debug(f"Trailing SL updated for {position.position_id}: {new_trailing_sl/100:.2f}")

    def _check_exit_conditions(self, position: ManagedPosition, price: int) -> Optional[str]:
        """Check if any exit condition is triggered.

        Args:
            position: The position to check
            price: Current price

        Returns:
            Exit reason if triggered, None otherwise
        """
        # Check options time-decay exit
        if position.is_option and self._config.enable_options_time_decay_exit:
            time_decay_reason = self._check_options_time_decay_exit(position)
            if time_decay_reason:
                return time_decay_reason

        # Check stop-loss
        if position.stop_loss_price is not None:
            if position.side == OrderSide.BUY:
                if price <= position.stop_loss_price:
                    return "STOP_LOSS"
            else:  # SELL (short)
                if price >= position.stop_loss_price:
                    return "STOP_LOSS"

        # Check target
        if position.target_price is not None:
            if position.side == OrderSide.BUY:
                if price >= position.target_price:
                    return "TARGET"
            else:  # SELL (short)
                if price <= position.target_price:
                    return "TARGET"

        # Check trailing SL
        if position.trailing_sl_activated and position.trailing_sl_price:
            if position.side == OrderSide.BUY:
                if price <= position.trailing_sl_price:
                    return "TRAILING_SL"
            else:  # SELL (short)
                if price >= position.trailing_sl_price:
                    return "TRAILING_SL"

        # Check time-based exit
        if self._config.time_based_exit_time:
            current_time = self._now().time()
            if current_time >= self._config.time_based_exit_time:
                return "TIME_EXIT"

        # Check intraday square-off time for MIS positions
        if position.product_type == ProductType.MIS:
            current_time = self._now().time()
            if current_time >= INTRADAY_SQUARE_OFF_TIME:
                return "INTRADAY_SQUARE_OFF"

        return None

    def _check_options_time_decay_exit(self, position: ManagedPosition) -> Optional[str]:
        """Check options-specific time decay exit conditions.

        Args:
            position: The options position to check.

        Returns:
            Exit reason if triggered, None otherwise.
        """
        if not position.is_option:
            return None

        # Check DTE (Days to Expiry) exit
        if position.option_dte is not None and position.option_dte < self._config.options_dte_exit_threshold:
            # Check if position is profitable
            if position.unrealized_pnl <= 0:
                logger.info(
                    f"Options time decay exit: {position.position_id} | "
                    f"DTE={position.option_dte} < {self._config.options_dte_exit_threshold}, "
                    f"Not profitable (PnL={position.unrealized_pnl/100:.2f})"
                )
                return "OPTIONS_DTE_EXIT"

        # Check theta threshold exit
        if (position.option_theta is not None and
            position.option_premium_at_entry is not None and
            position.option_premium_at_entry > 0):

            theta_pct = abs(position.option_theta) / (position.option_premium_at_entry / 100)
            if theta_pct > self._config.options_theta_threshold_pct:
                # Tighten stop-loss instead of immediate exit
                self._tighten_stop_loss_for_high_theta(position, theta_pct)

        return None

    def _tighten_stop_loss_for_high_theta(self, position: ManagedPosition, theta_pct: float) -> None:
        """Tighten stop-loss when theta decay is high.

        Args:
            position: The options position.
            theta_pct: Theta as percentage of premium.
        """
        if position.stop_loss_price is None:
            return

        # Calculate new tighter SL (move SL closer to entry)
        if position.side == OrderSide.BUY:
            # For long options, tighten SL by moving it up
            sl_distance = position.entry_price - position.stop_loss_price
            new_sl_distance = int(sl_distance * 0.5)  # Reduce SL distance by 50%
            new_sl = position.entry_price - new_sl_distance

            if new_sl > position.stop_loss_price:
                logger.info(
                    f"Tightening SL for {position.position_id} due to high theta ({theta_pct*100:.1f}%): "
                    f"{position.stop_loss_price/100:.2f} -> {new_sl/100:.2f}"
                )
                position.stop_loss_price = new_sl
        else:
            # For short options, tighten SL by moving it down
            sl_distance = position.stop_loss_price - position.entry_price
            new_sl_distance = int(sl_distance * 0.5)
            new_sl = position.entry_price + new_sl_distance

            if new_sl < position.stop_loss_price:
                logger.info(
                    f"Tightening SL for {position.position_id} due to high theta ({theta_pct*100:.1f}%): "
                    f"{position.stop_loss_price/100:.2f} -> {new_sl/100:.2f}"
                )
                position.stop_loss_price = new_sl

    def _close_position(self, position: ManagedPosition, exit_price: int, reason: str) -> bool:
        """Close a position by placing exit order.

        Args:
            position: The position to close
            exit_price: Exit price
            reason: Reason for closing

        Returns:
            True if close was successful
        """
        try:
            # Determine exit side
            exit_side = OrderSide.SELL if position.side == OrderSide.BUY else OrderSide.BUY

            # If partial profit exits already consumed the full position, just mark closed.
            if position.remaining_quantity <= 0:
                with self._lock:
                    position.is_closed = True
                    position.exit_time = self._now()
                    position.exit_price = exit_price
                    position.exit_reason = reason
                    existing = self._instrument_to_position.get(position.instrument_key)
                    if isinstance(existing, list):
                        try:
                            existing.remove(position.position_id)
                        except ValueError:
                            pass
                        if not existing:
                            del self._instrument_to_position[position.instrument_key]
                    elif position.instrument_key in self._instrument_to_position:
                        del self._instrument_to_position[position.instrument_key]
                return True

            # Place exit order for remaining quantity
            exit_order = Order(
                instrument_key=position.instrument_key,
                side=exit_side,
                order_type=OrderType.MARKET,
                product_type=position.product_type,
                quantity=position.remaining_quantity,
                strategy_id=position.strategy_id,
                tag=f"EXIT_{reason}",
            )

            response = self._broker.place_order(exit_order)

            if (getattr(response.status, 'value', response.status)) in ("COMPLETE", "OPEN", "PARTIAL_FILL"):
                # Calculate realized P&L for remaining quantity
                remaining_pnl = position.get_unrealized_pnl_for_quantity(
                    exit_price, position.remaining_quantity
                )

                # Update position
                with self._lock:
                    position.is_closed = True
                    position.exit_time = self._now()
                    position.exit_price = exit_price
                    position.exit_reason = reason
                    position.realized_pnl = remaining_pnl

                    # Clean up tracking (handle list when multiple positions per instrument)
                    existing = self._instrument_to_position.get(position.instrument_key)
                    if isinstance(existing, list):
                        try:
                            existing.remove(position.position_id)
                        except ValueError:
                            pass
                        if not existing:
                            del self._instrument_to_position[position.instrument_key]
                    elif position.instrument_key in self._instrument_to_position:
                        del self._instrument_to_position[position.instrument_key]

                # Calculate blended exit price if partial exits exist
                blended_price = position.blended_exit_price

                # Calculate total realized P&L including partials
                total_pnl = position.total_realized_pnl

                logger.info(
                    f"Position closed: {position.position_id} | Reason: {reason} | "
                    f"Exit: {exit_price/100:.2f} | Blended: {f'{blended_price/100:.2f}' if blended_price else 'N/A'} | "
                    f"Final P&L: {remaining_pnl/100:.2f} | Total P&L: {total_pnl/100:.2f} | "
                    f"Partial exits: {len(position.partial_exits)}"
                )

                # Persist to database
                if self._trade_logger:
                    try:
                        self._trade_logger.close_position(position.instrument_key, "closed")

                        # Attribute fees (brokerage + STT + GST + exchange + SEBI)
                        # by summing charges from the broker's per-fill history
                        # for this instrument since position entry.
                        fees_paisa = 0
                        try:
                            if hasattr(self._broker, "sum_fees_for_instrument"):
                                fees_paisa = int(self._broker.sum_fees_for_instrument(
                                    position.instrument_key,
                                    since=position.entry_time,
                                ))
                        except Exception as fe:
                            logger.warning(f"Fee attribution failed for {position.instrument_key}: {fe}")

                        # Add a TradeRecord
                        self._trade_logger.log_trade({
                            "strategy": position.strategy_id,
                            "instrument_key": position.instrument_key,
                            "side": position.side.value,
                            "entry_price": position.entry_price,
                            "exit_price": blended_price or exit_price,
                            "quantity": position.original_quantity,
                            "realized_pnl": total_pnl,
                            "fees": fees_paisa,
                            "entry_time": position.entry_time,
                            "exit_time": position.exit_time,
                            "holding_duration_seconds": int((position.exit_time - position.entry_time).total_seconds()),
                        })
                    except Exception as e:
                        logger.error(f"Failed to log trade to database: {e}")

                # Sync with risk manager after position close
                self._sync_risk_manager()

                # Wire CircuitBreaker (Scheme 4)
                if getattr(self, "_risk_manager", None) and getattr(self._risk_manager, "circuit_breaker", None):
                    # Per-strategy daily-loss cap (Tier 2.6)
                    try:
                        if hasattr(self._risk_manager.circuit_breaker, "record_strategy_pnl"):
                            self._risk_manager.circuit_breaker.record_strategy_pnl(
                                position.strategy_id, int(total_pnl)
                            )
                    except Exception as exc:
                        logger.debug(f"record_strategy_pnl failed: {exc}")
                    if total_pnl < 0:
                        trade_details = {
                            "direction": "BUY" if position.side.value == "BUY" else "SELL",
                            "strategy": position.strategy_id,
                            "sl_distance_pct": abs((position.entry_price - (position.stop_loss_price or 0)) / position.entry_price * 100) if position.entry_price else 0,
                            "holding_time_min": (position.exit_time - position.entry_time).total_seconds() / 60 if position.exit_time and position.entry_time else 0,
                            "hour": position.entry_time.hour if position.entry_time else -1,
                        }
                        self._risk_manager.circuit_breaker.record_loss(trade_details)
                    elif total_pnl > 0:
                        self._risk_manager.circuit_breaker.record_win()

                # Call callback if set
                if self._on_position_close:
                    try:
                        self._on_position_close(position)
                    except Exception as e:
                        logger.error(f"Error in position close callback: {e}")

                return True
            else:
                logger.error(f"Failed to close position {position.position_id}: {response.message}")
                return False

        except Exception as e:
            logger.exception(f"Error closing position {position.position_id}: {e}")
            return False

    def close_position_manual(self, position_id: str, reason: str = "MANUAL") -> bool:
        """Manually close a position.

        Args:
            position_id: The position ID to close
            reason: Reason for closing

        Returns:
            True if close was successful
        """
        with self._lock:
            if position_id not in self._positions:
                logger.error(f"Position not found: {position_id}")
                return False

            position = self._positions[position_id]
            if position.is_closed:
                logger.warning(f"Position already closed: {position_id}")
                return False

            current_price = position.current_price or position.entry_price

        return self._close_position(position, current_price, reason)

    def close_all_positions(self, reason: str = "MANUAL") -> list[ManagedPosition]:
        """Close all open positions.

        Args:
            reason: Reason for closing

        Returns:
            List of closed positions
        """
        closed_positions = []

        with self._lock:
            open_positions = [
                pos for pos in self._positions.values()
                if not pos.is_closed
            ]

        for position in open_positions:
            current_price = position.current_price or position.entry_price
            if self._close_position(position, current_price, reason):
                closed_positions.append(position)

        return closed_positions

    def get_position(self, position_id: str) -> Optional[ManagedPosition]:
        """Get a position by ID.

        Args:
            position_id: The position ID

        Returns:
            ManagedPosition if found, None otherwise
        """
        with self._lock:
            return self._positions.get(position_id)

    def get_position_by_instrument(self, instrument_key: str) -> Optional[ManagedPosition]:
        """Get position by instrument key.

        Args:
            instrument_key: The instrument key

        Returns:
            ManagedPosition if found, None otherwise
        """
        with self._lock:
            position_id = self._instrument_to_position.get(instrument_key)
            if position_id:
                return self._positions.get(position_id)
            return None

    def get_open_positions(self) -> list[ManagedPosition]:
        """Get all open positions.

        Returns:
            List of open positions
        """
        with self._lock:
            return [pos for pos in self._positions.values() if not pos.is_closed]

    def get_positions_by_strategy(self, strategy_id: str) -> list[ManagedPosition]:
        """Get all positions for a strategy.

        Args:
            strategy_id: The strategy ID

        Returns:
            List of positions
        """
        with self._lock:
            return [
                pos for pos in self._positions.values()
                if pos.strategy_id == strategy_id and not pos.is_closed
            ]

    def get_total_unrealized_pnl(self) -> int:
        """Get total unrealized P&L across all positions.

        Returns:
            Total unrealized P&L in paisa
        """
        with self._lock:
            return sum(pos.unrealized_pnl for pos in self._positions.values() if not pos.is_closed)

    def get_total_realized_pnl(self) -> int:
        """Get total realized P&L from closed positions.

        Returns:
            Total realized P&L in paisa
        """
        with self._lock:
            return sum(pos.total_realized_pnl for pos in self._positions.values() if pos.is_closed)

    def check_intraday_square_off(self) -> list[ManagedPosition]:
        """Check and execute intraday square-off for MIS positions.

        Returns:
            List of positions that were squared off
        """
        current_time = self._now().time()

        if current_time < INTRADAY_SQUARE_OFF_TIME:
            return []

        closed_positions = []

        with self._lock:
            mis_positions = [
                pos for pos in self._positions.values()
                if pos.product_type == ProductType.MIS and not pos.is_closed
            ]

        for position in mis_positions:
            current_price = position.current_price or position.entry_price
            if self._close_position(position, current_price, "INTRADAY_SQUARE_OFF"):
                closed_positions.append(position)

        if closed_positions:
            logger.info(f"Squared off {len(closed_positions)} MIS positions for intraday")

        return closed_positions

    def _sync_risk_manager(self) -> None:
        """Sync P&L and position counts with RiskManager.

        This method updates the risk manager with current realized P&L,
        unrealized P&L, open position count, and deployed capital.
        """
        if self._risk_manager is None:
            return

        try:
            realized_pnl = self.get_total_realized_pnl()
            unrealized_pnl = self.get_total_unrealized_pnl()
            open_positions = self.get_open_positions()

            position_count = len(open_positions)
            deployed_capital = sum(
                pos.entry_price * pos.remaining_quantity for pos in open_positions
            )

            self._risk_manager.update_pnl(realized=realized_pnl, unrealized=unrealized_pnl)
            self._risk_manager.update_positions(count=position_count, deployed=deployed_capital)

            logger.debug(
                f"RiskManager synced | realized={realized_pnl} paisa | "
                f"unrealized={unrealized_pnl} paisa | positions={position_count} | "
                f"deployed={deployed_capital} paisa"
            )
        except Exception as e:
            logger.error(f"Failed to sync with RiskManager: {e}")

    def sync_pnl_with_risk_manager(self) -> None:
        """Public method to force P&L sync with RiskManager.

        Can be called periodically to ensure risk manager has latest data.
        """
        self._sync_risk_manager()

    def update_position_sl_target(
        self,
        position_id: str,
        new_sl: Optional[int] = None,
        new_target: Optional[int] = None,
    ) -> bool:
        """Update stop-loss and/or target for a position.

        Args:
            position_id: The position ID to update.
            new_sl: New stop-loss price in paisa (None to keep current).
            new_target: New target price in paisa (None to keep current).

        Returns:
            True if update was successful.
        """
        with self._lock:
            if position_id not in self._positions:
                logger.error(f"Position not found: {position_id}")
                return False

            position = self._positions[position_id]
            if position.is_closed:
                logger.warning(f"Cannot update closed position: {position_id}")
                return False

            if new_sl is not None:
                old_sl = position.stop_loss_price
                position.stop_loss_price = new_sl
                logger.info(f"Updated SL for {position_id}: {old_sl/100:.2f} -> {new_sl/100:.2f}")

            if new_target is not None:
                old_target = position.target_price
                position.target_price = new_target
                logger.info(f"Updated target for {position_id}: {old_target/100:.2f} -> {new_target/100:.2f}")

            return True

    def get_position_summary(self, position_id: str) -> Optional[dict]:
        """Get a summary of position status including partial exit details.

        Args:
            position_id: The position ID.

        Returns:
            Dictionary with position summary, or None if not found.
        """
        with self._lock:
            position = self._positions.get(position_id)
            if not position:
                return None

            return {
                "position_id": position.position_id,
                "instrument_key": position.instrument_key,
                "strategy_id": position.strategy_id,
                "side": getattr(position.side, 'value', position.side),
                "original_quantity": position.original_quantity,
                "remaining_quantity": position.remaining_quantity,
                "entry_price": position.entry_price,
                "current_price": position.current_price,
                "stop_loss_price": position.stop_loss_price,
                "target_price": position.target_price,
                "unrealized_pnl": position.unrealized_pnl,
                "realized_pnl": position.realized_pnl,
                "total_realized_pnl": position.total_realized_pnl,
                "partial_exits": [
                    {
                        "exit_price": p.exit_price,
                        "quantity": p.quantity,
                        "exit_time": p.exit_time.isoformat(),
                        "exit_reason": p.exit_reason,
                        "realized_pnl": p.realized_pnl,
                    }
                    for p in position.partial_exits
                ],
                "partial_booking_done": position.partial_booking_done,
                "trailing_sl_activated": position.trailing_sl_activated,
                "trailing_sl_price": position.trailing_sl_price,
                "is_closed": position.is_closed,
                "exit_reason": position.exit_reason,
                "exit_price": position.exit_price,
                "blended_exit_price": position.blended_exit_price,
                "entry_atr": position.entry_atr,
                "volatility_regime": position.volatility_regime,
                "is_option": position.is_option,
                "option_dte": position.option_dte,
                # Momentum trailing stop fields
                "momentum_trailing_sl_price": position.momentum_trailing_sl_price,
                "current_momentum": position.current_momentum,
                "current_momentum_multiplier": position.current_momentum_multiplier,
                "breakeven_activated": position.breakeven_activated,
                "highest_momentum_price": position.highest_momentum_price,
                "lowest_momentum_price": position.lowest_momentum_price,
                # 4-Tier partial profit fields
                "exit_tiers": [
                    {
                        "tier": t.tier,
                        "rr_level": t.rr_level,
                        "exit_percent": t.exit_percent,
                        "exit_price": t.exit_price,
                        "exited": t.exited,
                        "exited_at": t.exited_at.isoformat() if t.exited_at else None,
                        "exited_price": t.exited_price,
                        "realized_pnl": t.realized_pnl,
                    }
                    for t in position.exit_tiers
                ] if position.exit_tiers else [],
                "tier_4_trailing_sl_price": position.tier_4_trailing_sl_price,
                "tier_4_highest_price": position.tier_4_highest_price,
                "tier_4_lowest_price": position.tier_4_lowest_price,
            }

    def get_momentum_trailing_status(self, position_id: str) -> Optional[dict]:
        """Get momentum trailing stop status for a position.

        Args:
            position_id: The position ID.

        Returns:
            Dictionary with momentum status, or None if position not found
                     or momentum trailing stop not enabled.
        """
        with self._lock:
            position = self._positions.get(position_id)
            if not position:
                return None

            if self._config.trailing_stop_type != "momentum":
                return {"error": "Momentum trailing stop not enabled"}

            if self._momentum_trailing_stop is None:
                return {"error": "Momentum trailing stop not initialized"}

            return self._momentum_trailing_stop.get_momentum_status(position)

    def on_bar(
        self,
        instrument_key: str,
        open_price: int,
        high_price: int,
        low_price: int,
        close_price: int,
        volume: int,
        timestamp: Optional[datetime] = None,
    ) -> list[ManagedPosition]:
        """Process a new OHLCV bar and check exit conditions.

        This is an alternative to on_tick() that receives full bar data,
        which is required for momentum-based trailing stops.

        Args:
            instrument_key: The instrument identifier.
            open_price: Open price in paisa.
            high_price: High price in paisa.
            low_price: Low price in paisa.
            close_price: Close price in paisa.
            volume: Volume for the bar.
            timestamp: Bar timestamp (defaults to now).

        Returns:
            List of positions that were closed.
        """
        # Update price history first
        self.update_price_bar(
            instrument_key, open_price, high_price, low_price, close_price, volume, timestamp
        )

        # Process the close price as a tick
        return self.on_tick(instrument_key, close_price)
