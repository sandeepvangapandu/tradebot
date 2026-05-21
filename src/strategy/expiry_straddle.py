"""Expiry day straddle sell strategy for theta decay.

This module implements a 0DTE (Zero Days to Expiry) straddle selling strategy
that capitalizes on rapid theta decay on expiry days. The strategy sells
ATM Call and Put options with hedges for protection.

All monetary values are in PAISA (integer) - 1 Rupee = 100 paisa.
All timestamps are in IST (Asia/Kolkata).

Example:
    Basic usage::

        from datetime import datetime
        from src.strategy.expiry_straddle import (
            StraddlePosition, is_expiry_day, should_exit_straddle
        )

        # Check if today is Bank Nifty expiry (Wednesday)
        today = datetime.now()
        if is_expiry_day("BANKNIFTY", today):
            print("Today is expiry day - strategy active")

        # Create a straddle position
        position = StraddlePosition(
            ce_strike=48000,
            pe_strike=48000,
            ce_entry_price=15000,  # Rs 150
            pe_entry_price=12000,  # Rs 120
            ce_hedge_strike=48500,
            pe_hedge_strike=47500,
            entry_time=datetime.now(),
            total_premium_collected=27000  # Rs 270
        )

        # Check exit conditions
        should_exit, reason = should_exit_straddle(
            position, current_spot=48100, entry_spot=48000,
            current_time=datetime.now()
        )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, time
from enum import Enum
from typing import Any, Optional

from loguru import logger


class ExitReason(str, Enum):
    """Reasons for exiting a straddle position."""

    TARGET_PROFIT = "Target profit reached"
    STOP_LOSS = "Stop loss hit"
    TIME_EXIT = "Time exit"
    LARGE_MOVE = "Large move detected"
    HOLD = "Hold"


@dataclass
class StraddlePosition:
    """Represents a straddle position (short CE + short PE).

    A straddle involves selling both a call and put option at the same
    strike price, betting on low volatility and time decay.

    Attributes:
        ce_strike: Call option strike price in paisa
        pe_strike: Put option strike price in paisa
        ce_entry_price: Call option entry price in paisa (premium received)
        pe_entry_price: Put option entry price in paisa (premium received)
        ce_hedge_strike: Optional OTM call hedge strike in paisa
        pe_hedge_strike: Optional OTM put hedge strike in paisa
        entry_time: When the position was entered
        total_premium_collected: Total premium received in paisa
        ce_quantity: Number of call lots (default: 1)
        pe_quantity: Number of put lots (default: 1)
        lot_size: Lot size for the underlying (default: 15 for Bank Nifty)
        entry_spot: Spot price at entry in paisa

    Example:
        >>> position = StraddlePosition(
        ...     ce_strike=48000_00,  # Rs 48,000
        ...     pe_strike=48000_00,
        ...     ce_entry_price=150_00,  # Rs 150 premium
        ...     pe_entry_price=120_00,  # Rs 120 premium
        ...     ce_hedge_strike=48500_00,
        ...     pe_hedge_strike=47500_00,
        ...     entry_time=datetime.now(),
        ...     total_premium_collected=270_00,
        ...     entry_spot=48050_00
        ... )
    """

    ce_strike: int
    pe_strike: int
    ce_entry_price: int
    pe_entry_price: int
    ce_hedge_strike: Optional[int] = None
    pe_hedge_strike: Optional[int] = None
    entry_time: datetime = field(default_factory=datetime.now)
    total_premium_collected: int = 0
    ce_quantity: int = 1
    pe_quantity: int = 1
    lot_size: int = 15  # Bank Nifty lot size
    entry_spot: int = 0

    def __post_init__(self) -> None:
        """Calculate total premium if not provided."""
        if self.total_premium_collected == 0:
            self.total_premium_collected = (
                self.ce_entry_price * self.ce_quantity +
                self.pe_entry_price * self.pe_quantity
            ) * self.lot_size


def is_expiry_day(symbol: str, today: Optional[datetime] = None) -> bool:
    """Check if today is expiry day for the given symbol.

    Bank Nifty: Wednesday (weekday 2)
    Nifty 50: Thursday (weekday 3)

    Note: This checks for weekly expiry days. Monthly expiry (last Thursday
    of the month) may differ and requires additional validation.

    Args:
        symbol: The underlying symbol ("BANKNIFTY" or "NIFTY")
        today: Date to check (defaults to current date)

    Returns:
        True if today is the regular expiry day for the symbol

    Example:
        >>> from datetime import datetime
        >>> is_expiry_day("BANKNIFTY", datetime(2026, 4, 1))  # Wednesday
        True
        >>> is_expiry_day("NIFTY", datetime(2026, 4, 2))  # Thursday
        True
    """
    if today is None:
        today = datetime.now()

    weekday = today.weekday()
    symbol_upper = symbol.upper()

    if symbol_upper == "BANKNIFTY":
        return weekday == 2  # Wednesday
    elif symbol_upper == "NIFTY":
        return weekday == 3  # Thursday

    logger.warning(f"Unknown symbol for expiry check: {symbol}")
    return False


def is_monthly_expiry(today: Optional[datetime] = None) -> bool:
    """Check if today is the last Thursday of the month (monthly expiry).

    Monthly expiry is significant as it typically has higher open interest
    and can show different price action compared to weekly expiry.

    Args:
        today: Date to check (defaults to current date)

    Returns:
        True if today is the last Thursday of the month
    """
    if today is None:
        today = datetime.now()

    # Check if it's a Thursday
    if today.weekday() != 3:
        return False

    # Check if it's the last Thursday of the month
    import calendar

    _, last_day = calendar.monthrange(today.year, today.month)
    remaining_days = last_day - today.day

    # If less than 7 days remain, this is the last Thursday
    return remaining_days < 7


def calculate_straddle_profit_loss(
    position: StraddlePosition,
    current_ce_price: int,
    current_pe_price: int,
    current_ce_hedge_price: Optional[int] = None,
    current_pe_hedge_price: Optional[int] = None,
) -> dict[str, Any]:
    """Calculate P&L of straddle position.

    For a short straddle:
    - Profit when options lose value (premium collected > current premium)
    - Loss when options gain value (current premium > premium collected)

    The calculation accounts for:
    - Short call P&L: (entry_price - current_price) * quantity * lot_size
    - Short put P&L: (entry_price - current_price) * quantity * lot_size
    - Hedge P&L: (current_hedge_price - hedge_entry_price) for protection

    Args:
        position: The straddle position
        current_ce_price: Current call option price in paisa
        current_pe_price: Current put option price in paisa
        current_ce_hedge_price: Current call hedge price (if hedged)
        current_pe_hedge_price: Current put hedge price (if hedged)

    Returns:
        Dictionary containing:
            - ce_pnl: Call leg P&L in paisa
            - pe_pnl: Put leg P&L in paisa
            - hedge_ce_pnl: Call hedge P&L in paisa
            - hedge_pe_pnl: Put hedge P&L in paisa
            - total_pnl: Total P&L in paisa
            - pnl_pct: P&L as percentage of premium collected
            - max_profit: Maximum possible profit (premium collected)
            - max_loss: Theoretical maximum loss (unlimited without hedges)

    Example:
        >>> position = StraddlePosition(
        ...     ce_strike=48000_00, pe_strike=48000_00,
        ...     ce_entry_price=150_00, pe_entry_price=120_00,
        ...     entry_time=datetime.now(), total_premium_collected=270_00
        ... )
        >>> pnl = calculate_straddle_profit_loss(
        ...     position, current_ce_price=80_00, current_pe_price=60_00
        ... )
        >>> print(f"P&L: Rs {pnl['total_pnl']/100:.2f} ({pnl['pnl_pct']:.1f}%)")
    """
    # Calculate short leg P&L (profit when price decreases)
    ce_pnl = (position.ce_entry_price - current_ce_price) * position.ce_quantity * position.lot_size
    pe_pnl = (position.pe_entry_price - current_pe_price) * position.pe_quantity * position.lot_size

    # Calculate hedge P&L (long options, profit when price increases)
    hedge_ce_pnl = 0
    hedge_pe_pnl = 0

    if position.ce_hedge_strike and current_ce_hedge_price is not None:
        # Assume hedge was bought at ~20-30% of short premium
        hedge_ce_entry = int(position.ce_entry_price * 0.25)  # Estimated
        hedge_ce_pnl = (current_ce_hedge_price - hedge_ce_entry) * position.ce_quantity * position.lot_size

    if position.pe_hedge_strike and current_pe_hedge_price is not None:
        hedge_pe_entry = int(position.pe_entry_price * 0.25)  # Estimated
        hedge_pe_pnl = (current_pe_hedge_price - hedge_pe_entry) * position.pe_quantity * position.lot_size

    # Total P&L
    total_pnl = ce_pnl + pe_pnl + hedge_ce_pnl + hedge_pe_pnl

    # Calculate percentage
    pnl_pct = 0.0
    if position.total_premium_collected > 0:
        pnl_pct = (total_pnl / position.total_premium_collected) * 100

    return {
        "ce_pnl": ce_pnl,
        "pe_pnl": pe_pnl,
        "hedge_ce_pnl": hedge_ce_pnl,
        "hedge_pe_pnl": hedge_pe_pnl,
        "total_pnl": total_pnl,
        "pnl_pct": pnl_pct,
        "max_profit": position.total_premium_collected,
        "max_loss": float("inf"),  # Theoretically unlimited for naked straddle
    }


def should_exit_straddle(
    position: StraddlePosition,
    current_ce_price: int,
    current_pe_price: int,
    current_spot: int,
    current_time: datetime,
    target_pct: float = 70.0,
    stop_loss_pct: float = 50.0,
    time_exit: time = time(14, 0),  # 2:00 PM
    emergency_move_pct: float = 1.5,
    current_ce_hedge_price: Optional[int] = None,
    current_pe_hedge_price: Optional[int] = None,
) -> tuple[bool, str]:
    """Check if straddle should be exited based on exit rules.

    Exit scenarios:
    1. Target profit: 70% of premium collected (default)
    2. Stop loss: 50% of premium collected (default)
    3. Time-based: Exit at 2:00 PM to avoid settlement risk
    4. Emergency: Market moves >1.5% from entry (default)

    Args:
        position: The straddle position
        current_ce_price: Current call option price in paisa
        current_pe_price: Current put option price in paisa
        current_spot: Current spot price in paisa
        current_time: Current timestamp
        target_pct: Profit target percentage (default: 70%)
        stop_loss_pct: Stop loss percentage (default: 50%)
        time_exit: Time-based exit time (default: 14:00)
        emergency_move_pct: Emergency exit if spot moves more than this %
        current_ce_hedge_price: Current call hedge price (optional)
        current_pe_hedge_price: Current put hedge price (optional)

    Returns:
        Tuple of (should_exit: bool, reason: str)

    Example:
        >>> position = StraddlePosition(
        ...     ce_strike=48000_00, pe_strike=48000_00,
        ...     ce_entry_price=150_00, pe_entry_price=120_00,
        ...     entry_time=datetime.now(), total_premium_collected=270_00,
        ...     entry_spot=48000_00
        ... )
        >>> should_exit, reason = should_exit_straddle(
        ...     position, current_ce_price=50_00, current_pe_price=40_00,
        ...     current_spot=48100_00, current_time=datetime.now()
        ... )
        >>> if should_exit:
        ...     print(f"Exit: {reason}")
    """
    # Calculate current P&L
    pnl_data = calculate_straddle_profit_loss(
        position, current_ce_price, current_pe_price,
        current_ce_hedge_price, current_pe_hedge_price
    )
    pnl_pct = pnl_data["pnl_pct"]

    # Check target profit (e.g., 70% of premium collected)
    if pnl_pct >= target_pct:
        return True, f"{ExitReason.TARGET_PROFIT}: {pnl_pct:.1f}%"

    # Check stop loss (e.g., 50% loss of premium collected)
    if pnl_pct <= -stop_loss_pct:
        return True, f"{ExitReason.STOP_LOSS}: {pnl_pct:.1f}%"

    # Check time exit (2:00 PM default)
    current_t = current_time.time() if isinstance(current_time, datetime) else current_time
    if current_t >= time_exit:
        return True, f"{ExitReason.TIME_EXIT}: {current_t.strftime('%H:%M')}"

    # Check large move (>1.5% default)
    if position.entry_spot > 0:
        move_pct = abs(current_spot - position.entry_spot) / position.entry_spot * 100
        if move_pct > emergency_move_pct:
            return True, f"{ExitReason.LARGE_MOVE}: {move_pct:.2f}%"

    return False, ExitReason.HOLD


def calculate_atm_strike(spot_price: int, strike_interval: int = 100) -> int:
    """Calculate the ATM (At-The-Money) strike price.

    Args:
        spot_price: Current spot price in paisa
        strike_interval: Strike price interval (default: 100 for Bank Nifty)

    Returns:
        ATM strike price in paisa

    Example:
        >>> calculate_atm_strike(48150_00)  # Rs 48,150
        48200_00  # Rounded to nearest 100
    """
    # Convert to strike units
    spot_in_units = spot_price / 100  # Convert paisa to rupees
    atm_strike = round(spot_in_units / strike_interval) * strike_interval
    return int(atm_strike * 100)  # Convert back to paisa


def select_hedge_strikes(
    atm_strike: int,
    spot_price: int,
    hedge_offset: int = 500,
    strike_interval: int = 100,
) -> tuple[int, int]:
    """Select OTM hedge strikes for straddle protection.

    For a short straddle, we buy OTM options as insurance:
    - Buy OTM Call: ATM + offset (e.g., ATM + 500 points)
    - Buy OTM Put: ATM - offset (e.g., ATM - 500 points)

    Args:
        atm_strike: ATM strike price in paisa
        spot_price: Current spot price in paisa
        hedge_offset: Points away from ATM for hedges (default: 500)
        strike_interval: Strike price interval (default: 100)

    Returns:
        Tuple of (ce_hedge_strike, pe_hedge_strike) in paisa

    Example:
        >>> select_hedge_strikes(48000_00, 48050_00, hedge_offset=500)
        (48500_00, 47500_00)
    """
    # Convert to rupees for easier calculation
    atm_rupees = atm_strike / 100
    offset_rupees = hedge_offset  # Already in points

    # Calculate hedge strikes
    ce_hedge = round((atm_rupees + offset_rupees) / strike_interval) * strike_interval
    pe_hedge = round((atm_rupees - offset_rupees) / strike_interval) * strike_interval

    return int(ce_hedge * 100), int(pe_hedge * 100)


def check_entry_conditions(
    days_to_expiry: int,
    atm_iv: float,
    adx: float,
    atr: int,
    current_time: datetime,
    min_iv: float = 15.0,
    max_iv: float = 35.0,
    max_adx: float = 20.0,
    max_atr: int = 2000,
    entry_window_start: time = time(9, 30),
    entry_window_end: time = time(10, 0),
    open_iv: float | None = None,
    iv_spike_threshold_pct: float = 15.0,
) -> tuple[bool, list[str]]:
    """Check if all entry conditions are met for straddle sell.

    Entry conditions:
    1. Must be expiry day (0 DTE)
    2. ATM IV between min and max (default: 15-35%)
    3. ADX < max_adx (default: 20) - range-bound expectation
    4. ATR < max_atr (default: 2000) - low volatility
    5. Time between entry_window_start and entry_window_end

    Args:
        days_to_expiry: Days to expiry (should be 0)
        atm_iv: Current ATM implied volatility (%)
        adx: ADX value (trend strength)
        atr: ATR value in points
        current_time: Current timestamp
        min_iv: Minimum IV threshold (default: 15%)
        max_iv: Maximum IV threshold (default: 35%)
        max_adx: Maximum ADX for range-bound (default: 20)
        max_atr: Maximum ATR for low volatility (default: 2000)
        entry_window_start: Start of entry window (default: 09:30)
        entry_window_end: End of entry window (default: 10:00)

    Returns:
        Tuple of (all_conditions_met: bool, failure_reasons: list)

    Example:
        >>> check_entry_conditions(
        ...     days_to_expiry=0,
        ...     atm_iv=22.5,
        ...     adx=15.0,
        ...     atr=1500,
        ...     current_time=datetime.now()
        ... )
        (True, [])
    """
    failures = []

    # Check expiry day
    if days_to_expiry != 0:
        failures.append(f"Not expiry day: {days_to_expiry} DTE")

    # Check IV range
    if atm_iv < min_iv:
        failures.append(f"IV too low: {atm_iv:.1f}% < {min_iv}%")
    if atm_iv > max_iv:
        failures.append(f"IV too high: {atm_iv:.1f}% > {max_iv}%")

    # IV spike gate: if IV jumped >15% from open, market is gapping — skip.
    if open_iv is not None and open_iv > 0:
        iv_change_pct = ((atm_iv - open_iv) / open_iv) * 100
        if iv_change_pct > iv_spike_threshold_pct:
            failures.append(
                f"IV spike detected: {open_iv:.1f}% → {atm_iv:.1f}% "
                f"(+{iv_change_pct:.1f}% > {iv_spike_threshold_pct}% threshold)"
            )

    # Check ADX (trend strength)
    if adx > max_adx:
        failures.append(f"Trending market: ADX {adx:.1f} > {max_adx}")

    # Check ATR (volatility)
    if atr > max_atr:
        failures.append(f"High volatility: ATR {atr} > {max_atr}")

    # Check time window
    current_t = current_time.time() if isinstance(current_time, datetime) else current_time
    if current_t < entry_window_start:
        failures.append(f"Too early: {current_t.strftime('%H:%M')} < {entry_window_start.strftime('%H:%M')}")
    if current_t > entry_window_end:
        failures.append(f"Too late: {current_t.strftime('%H:%M')} > {entry_window_end.strftime('%H:%M')}")

    return len(failures) == 0, failures


def estimate_straddle_payoff(
    ce_strike: int,
    pe_strike: int,
    ce_premium: int,
    pe_premium: int,
    spot_range: tuple[int, int],
    steps: int = 20,
) -> list[dict[str, Any]]:
    """Generate payoff diagram data for straddle position.

    Args:
        ce_strike: Call strike price in paisa
        pe_strike: Put strike price in paisa
        ce_premium: Call premium received in paisa
        pe_premium: Put premium received in paisa
        spot_range: Tuple of (min_spot, max_spot) in paisa
        steps: Number of price points to calculate

    Returns:
        List of payoff data points with spot price and P&L

    Example:
        >>> payoffs = estimate_straddle_payoff(
        ...     ce_strike=48000_00, pe_strike=48000_00,
        ...     ce_premium=150_00, pe_premium=120_00,
        ...     spot_range=(47500_00, 48500_00)
        ... )
        >>> for p in payoffs:
        ...     print(f"Spot: {p['spot']/100:.0f}, P&L: {p['pnl']/100:.0f}")
    """
    min_spot, max_spot = spot_range
    step_size = (max_spot - min_spot) / steps
    total_premium = ce_premium + pe_premium

    payoffs = []
    for i in range(steps + 1):
        spot = int(min_spot + (step_size * i))

        # Short call payoff: max(0, spot - strike) - premium
        call_payoff = -max(0, spot - ce_strike) + ce_premium

        # Short put payoff: max(0, strike - spot) - premium
        put_payoff = -max(0, pe_strike - spot) + pe_premium

        total_pnl = call_payoff + put_payoff

        payoffs.append({
            "spot": spot,
            "call_pnl": call_payoff,
            "put_pnl": put_payoff,
            "total_pnl": total_pnl,
            "spot_rupees": spot / 100,
            "pnl_rupees": total_pnl / 100,
        })

    return payoffs


class ExpiryStraddleStrategy:
    """Expiry day straddle sell strategy implementation.

    This class provides a complete implementation of the 0DTE straddle
    selling strategy with entry/exit logic and risk management.

    Attributes:
        symbol: Underlying symbol ("BANKNIFTY" or "NIFTY")
        config: Strategy configuration dictionary
        position: Current straddle position (if any)
        logger: Loguru logger instance

    Example:
        >>> strategy = ExpiryStraddleStrategy("BANKNIFTY", {
        ...     "min_iv": 15, "max_iv": 35,
        ...     "hedge_offset": 500
        ... })
        >>> if strategy.can_trade_today():
        ...     print("Strategy active today")
    """

    def __init__(self, symbol: str, config: Optional[dict[str, Any]] = None):
        """Initialize the strategy.

        Args:
            symbol: Underlying symbol ("BANKNIFTY" or "NIFTY")
            config: Optional configuration overrides
        """
        self.symbol = symbol.upper()
        self.config = config or {}
        self.position: Optional[StraddlePosition] = None
        self.logger = logger.bind(strategy="ExpiryStraddle", symbol=symbol)

        # Default configuration
        self.min_iv = self.config.get("min_iv", 15.0)
        self.max_iv = self.config.get("max_iv", 35.0)
        self.max_adx = self.config.get("max_adx", 20.0)
        self.max_atr = self.config.get("max_atr", 2000)
        self.hedge_offset = self.config.get("hedge_offset", 500)
        self.target_pct = self.config.get("target_pct", 70.0)
        self.stop_loss_pct = self.config.get("stop_loss_pct", 50.0)
        self.emergency_move_pct = self.config.get("emergency_move_pct", 1.5)
        self.lot_size = self.config.get("lot_size", 15)  # Bank Nifty

    def can_trade_today(self, today: Optional[datetime] = None) -> bool:
        """Check if today is a valid trading day for this strategy.

        Args:
            today: Date to check (defaults to today)

        Returns:
            True if today is expiry day for the symbol
        """
        return is_expiry_day(self.symbol, today)

    def check_entry(
        self,
        days_to_expiry: int,
        atm_iv: float,
        adx: float,
        atr: int,
        current_time: datetime,
    ) -> tuple[bool, list[str]]:
        """Check if entry conditions are met.

        Args:
            days_to_expiry: Days to expiry
            atm_iv: Current ATM IV
            adx: ADX value
            atr: ATR value
            current_time: Current timestamp

        Returns:
            Tuple of (can_enter, failure_reasons)
        """
        return check_entry_conditions(
            days_to_expiry=days_to_expiry,
            atm_iv=atm_iv,
            adx=adx,
            atr=atr,
            current_time=current_time,
            min_iv=self.min_iv,
            max_iv=self.max_iv,
            max_adx=self.max_adx,
            max_atr=self.max_atr,
        )

    def enter_position(
        self,
        spot_price: int,
        ce_premium: int,
        pe_premium: int,
        entry_time: datetime,
        with_hedges: bool = True,
    ) -> StraddlePosition:
        """Create and return a new straddle position.

        Args:
            spot_price: Current spot price in paisa
            ce_premium: Call premium in paisa
            pe_premium: Put premium in paisa
            entry_time: Entry timestamp
            with_hedges: Whether to include hedge strikes

        Returns:
            New StraddlePosition instance
        """
        atm_strike = calculate_atm_strike(spot_price)

        ce_hedge = None
        pe_hedge = None
        if with_hedges:
            ce_hedge, pe_hedge = select_hedge_strikes(
                atm_strike, spot_price, self.hedge_offset
            )

        self.position = StraddlePosition(
            ce_strike=atm_strike,
            pe_strike=atm_strike,
            ce_entry_price=ce_premium,
            pe_entry_price=pe_premium,
            ce_hedge_strike=ce_hedge,
            pe_hedge_strike=pe_hedge,
            entry_time=entry_time,
            entry_spot=spot_price,
            lot_size=self.lot_size,
        )

        self.logger.info(
            f"Entered straddle: ATM={atm_strike/100:.0f}, "
            f"Premium={self.position.total_premium_collected/100:.0f}, "
            f"Hedges: CE@{ce_hedge/100 if ce_hedge else None}, "
            f"PE@{pe_hedge/100 if pe_hedge else None}"
        )

        return self.position

    def check_exit(
        self,
        current_ce_price: int,
        current_pe_price: int,
        current_spot: int,
        current_time: datetime,
    ) -> tuple[bool, str]:
        """Check if current position should be exited.

        Args:
            current_ce_price: Current call price
            current_pe_price: Current put price
            current_spot: Current spot price
            current_time: Current timestamp

        Returns:
            Tuple of (should_exit, reason)
        """
        if self.position is None:
            return False, "No position"

        return should_exit_straddle(
            position=self.position,
            current_ce_price=current_ce_price,
            current_pe_price=current_pe_price,
            current_spot=current_spot,
            current_time=current_time,
            target_pct=self.target_pct,
            stop_loss_pct=self.stop_loss_pct,
            emergency_move_pct=self.emergency_move_pct,
        )

    def get_payoff_diagram(
        self,
        spot_range: Optional[tuple[int, int]] = None,
    ) -> list[dict[str, Any]]:
        """Generate payoff diagram for current position.

        Args:
            spot_range: Optional (min, max) spot range

        Returns:
            List of payoff data points
        """
        if self.position is None:
            return []

        if spot_range is None:
            # Default range: +/- 2% from entry
            entry = self.position.entry_spot
            spot_range = (int(entry * 0.98), int(entry * 1.02))

        return estimate_straddle_payoff(
            ce_strike=self.position.ce_strike,
            pe_strike=self.position.pe_strike,
            ce_premium=self.position.ce_entry_price,
            pe_premium=self.position.pe_entry_price,
            spot_range=spot_range,
        )
