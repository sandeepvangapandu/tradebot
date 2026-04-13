"""Smart Entry Optimizer - Market microstructure-based entry optimization.

Optimizes exact entry price using market microstructure analysis.
Instead of market orders, uses limit orders at optimal prices based on
order book flow, spread analysis, and VWAP proximity.

All prices are in PAISA (integer) - 1 Rupee = 100 paisa.
All timestamps are in IST (Asia/Kolkata).
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional
from zoneinfo import ZoneInfo

from loguru import logger

from src.strategy.engine import Signal
from src.strategy.builder import SignalType

IST = ZoneInfo("Asia/Kolkata")


class EntryType(Enum):
    """Type of entry optimization to apply."""

    IMMEDIATE = "immediate"  # Market order
    PULLBACK = "pullback"  # Wait for retracement
    BREAKOUT = "breakout"  # Wait for confirmation
    VWAP = "vwap"  # Entry at VWAP


@dataclass
class MicrostructureData:
    """Market microstructure analysis data.

    Attributes:
        spread_paisa: Bid-ask spread in paisa
        bid_ask_ratio: Bid volume / Ask volume (>1 = bullish pressure)
        vwap_distance: Percentage distance from VWAP
        imbalance: (bids - asks) / (bids + asks), range [-1, 1]
        pressure: Market pressure direction ("buy", "sell", "neutral")
        bid_qty: Total quantity at best bid
        ask_qty: Total quantity at best ask
        ltp: Last traded price in paisa
        vwap: VWAP price in paisa
        timestamp: When the data was captured
    """

    spread_paisa: int = 0
    bid_ask_ratio: float = 1.0
    vwap_distance: float = 0.0
    imbalance: float = 0.0
    pressure: str = "neutral"
    bid_qty: int = 0
    ask_qty: int = 0
    ltp: int = 0
    vwap: int = 0
    timestamp: datetime = field(default_factory=lambda: datetime.now(IST))

    def __post_init__(self):
        """Ensure timestamp is timezone-aware."""
        if self.timestamp.tzinfo is None:
            self.timestamp = self.timestamp.replace(tzinfo=IST)
        else:
            self.timestamp = self.timestamp.astimezone(IST)


@dataclass
class EntryOptimizationResult:
    """Result of entry optimization analysis.

    Attributes:
        entry_type: Recommended entry type
        optimal_price: Suggested entry price in paisa
        timeout: Timeout in seconds before fallback
        reason: Human-readable explanation
        fallback_price: Price to use if optimal not filled (market order)
        microstructure: The microstructure data used for analysis
    """

    entry_type: EntryType
    optimal_price: int
    timeout: int
    reason: str
    fallback_price: int
    microstructure: MicrostructureData


class EntryOptimizer:
    """Optimizes entry prices using market microstructure analysis.

    Analyzes order book data (spread, bid/ask ratio, VWAP distance) to
    determine the optimal entry strategy: immediate market order, limit
    at bid/ask for price improvement, or wait for pullback to VWAP.

    Thread-safe for concurrent signal processing.

    Example:
        >>> optimizer = EntryOptimizer(default_timeout=30)
        >>> result = optimizer.optimize_entry_price(
        ...     signal=signal,
        ...     orderbook={
        ...         "bid": 22450.25,
        ...         "ask": 22451.00,
        ...         "bid_qty": 1500,
        ...         "ask_qty": 800,
        ...         "vwap": 22448.50,
        ...         "ltp": 22450.50
        ...     }
        ... )
        >>> print(result.entry_type)  # EntryType.PULLBACK
    """

    # Threshold constants (as percentages)
    SPREAD_THRESHOLD_PCT = 0.1  # 0.1% max spread for limit orders
    VWAP_THRESHOLD_PCT = 0.3  # 0.3% distance from VWAP for pullback
    BID_ASK_RATIO_BULLISH = 1.2  # >1.2 = bullish pressure
    BID_ASK_RATIO_BEARISH = 0.8  # <0.8 = bearish pressure

    # Time-based fallback stages (seconds)
    LIMIT_TIMEOUT = 10
    MID_PRICE_TIMEOUT = 10

    def __init__(self, default_timeout: int = 30):
        """Initialize the entry optimizer.

        Args:
            default_timeout: Default timeout in seconds before market order fallback
        """
        self.default_timeout = default_timeout
        self._lock = threading.RLock()
        self._stats: dict[str, dict[str, Any]] = {
            "total_signals": 0,
            "by_type": {
                EntryType.IMMEDIATE.value: 0,
                EntryType.PULLBACK.value: 0,
                EntryType.BREAKOUT.value: 0,
                EntryType.VWAP.value: 0,
            },
            "fill_rates": {
                EntryType.IMMEDIATE.value: {"attempted": 0, "filled": 0},
                EntryType.PULLBACK.value: {"attempted": 0, "filled": 0},
                EntryType.BREAKOUT.value: {"attempted": 0, "filled": 0},
                EntryType.VWAP.value: {"attempted": 0, "filled": 0},
            },
        }

    def optimize_entry_price(
        self, signal: Signal, orderbook: dict[str, Any]
    ) -> EntryOptimizationResult:
        """Determine optimal entry price and strategy for a signal.

        Analyzes the order book microstructure and returns a recommendation
        for entry type, optimal price, and timeout settings.

        Args:
            signal: The trading signal to optimize entry for
            orderbook: Order book snapshot with keys:
                - bid: Best bid price (float, in rupees)
                - ask: Best ask price (float, in rupees)
                - bid_qty: Quantity at best bid (int)
                - ask_qty: Quantity at best ask (int)
                - vwap: VWAP price (float, in rupees)
                - ltp: Last traded price (float, in rupees)

        Returns:
            EntryOptimizationResult with entry type, price, and reasoning

        Raises:
            ValueError: If orderbook data is invalid or missing required fields
        """
        with self._lock:
            self._stats["total_signals"] += 1

        # Validate and parse orderbook
        micro = self.analyze_microstructure(orderbook)

        # Determine entry strategy based on signal direction
        is_buy = signal.signal_type in (
            SignalType.BUY,
            SignalType.BUY_CE,
            SignalType.BUY_PE,
        )

        # Calculate optimal entry
        optimal_price = self.calculate_optimal_entry(signal, micro)

        # Determine entry type and reasoning
        if is_buy:
            entry_type, reason = self._analyze_buy_entry(signal, micro)
        else:
            entry_type, reason = self._analyze_sell_entry(signal, micro)

        # Calculate fallback price (market order)
        fallback_price = micro.ltp if micro.ltp > 0 else (signal.price or 0)

        # Update statistics
        with self._lock:
            self._stats["by_type"][entry_type.value] += 1
            self._stats["fill_rates"][entry_type.value]["attempted"] += 1

        logger.info(
            f"Entry optimization for {signal.instrument_key}: "
            f"{entry_type.value} at {optimal_price} paisa - {reason}"
        )

        return EntryOptimizationResult(
            entry_type=entry_type,
            optimal_price=optimal_price,
            timeout=self.default_timeout,
            reason=reason,
            fallback_price=fallback_price,
            microstructure=micro,
        )

    def analyze_microstructure(self, orderbook: dict[str, Any]) -> MicrostructureData:
        """Analyze order book microstructure.

        Extracts and computes microstructure metrics from order book data.

        Args:
            orderbook: Order book snapshot with bid/ask/vwap/ltp data

        Returns:
            MicrostructureData with computed metrics

        Raises:
            ValueError: If required fields are missing
        """
        # Extract raw values with defaults
        bid = float(orderbook.get("bid", 0))
        ask = float(orderbook.get("ask", 0))
        bid_qty = int(orderbook.get("bid_qty", 0))
        ask_qty = int(orderbook.get("ask_qty", 0))
        vwap = float(orderbook.get("vwap", 0))
        ltp = float(orderbook.get("ltp", 0))

        # Validate minimum required data
        if bid <= 0 or ask <= 0:
            logger.warning("Invalid orderbook: missing bid/ask prices")
            # Return degraded microstructure with LTP if available
            ltp_int = int(round(ltp * 100)) if ltp > 0 else 0
            return MicrostructureData(
                spread_paisa=0,
                bid_ask_ratio=1.0,
                vwap_distance=0.0,
                imbalance=0.0,
                pressure="neutral",
                bid_qty=bid_qty,
                ask_qty=ask_qty,
                ltp=ltp_int,
                vwap=int(round(vwap * 100)) if vwap > 0 else ltp_int,
            )

        # Convert to paisa (integer)
        bid_paisa = int(round(bid * 100))
        ask_paisa = int(round(ask * 100))
        vwap_paisa = int(round(vwap * 100)) if vwap > 0 else 0
        ltp_paisa = int(round(ltp * 100)) if ltp > 0 else bid_paisa

        # Calculate spread
        spread_paisa = ask_paisa - bid_paisa

        # Calculate bid-ask ratio (handle division by zero)
        if ask_qty > 0:
            bid_ask_ratio = bid_qty / ask_qty
        else:
            bid_ask_ratio = float("inf") if bid_qty > 0 else 1.0

        # Calculate VWAP distance
        if vwap_paisa > 0 and ltp_paisa > 0:
            vwap_distance = (ltp_paisa - vwap_paisa) / vwap_paisa * 100
        else:
            vwap_distance = 0.0

        # Calculate order book imbalance
        total_qty = bid_qty + ask_qty
        if total_qty > 0:
            imbalance = (bid_qty - ask_qty) / total_qty
        else:
            imbalance = 0.0

        # Determine pressure
        if bid_ask_ratio > self.BID_ASK_RATIO_BULLISH and imbalance > 0.2:
            pressure = "buy"
        elif bid_ask_ratio < self.BID_ASK_RATIO_BEARISH and imbalance < -0.2:
            pressure = "sell"
        else:
            pressure = "neutral"

        return MicrostructureData(
            spread_paisa=spread_paisa,
            bid_ask_ratio=bid_ask_ratio,
            vwap_distance=vwap_distance,
            imbalance=imbalance,
            pressure=pressure,
            bid_qty=bid_qty,
            ask_qty=ask_qty,
            ltp=ltp_paisa,
            vwap=vwap_paisa,
        )

    def calculate_optimal_entry(
        self, signal: Signal, micro: MicrostructureData
    ) -> int:
        """Calculate the optimal entry price in paisa.

        Args:
            signal: The trading signal
            micro: Market microstructure data

        Returns:
            Optimal entry price in paisa
        """
        is_buy = signal.signal_type in (
            SignalType.BUY,
            SignalType.BUY_CE,
            SignalType.BUY_PE,
        )

        # Calculate spread percentage
        if micro.ltp > 0:
            spread_pct = (micro.spread_paisa / micro.ltp) * 100
        else:
            spread_pct = float("inf")

        # Determine entry strategy
        if is_buy:
            return self._calculate_buy_entry_price(signal, micro, spread_pct)
        else:
            return self._calculate_sell_entry_price(signal, micro, spread_pct)

    def _calculate_buy_entry_price(
        self, signal: Signal, micro: MicrostructureData, spread_pct: float
    ) -> int:
        """Calculate optimal buy entry price."""
        # If tight spread and bullish pressure, place limit at bid for price improvement
        if spread_pct < self.SPREAD_THRESHOLD_PCT:
            if micro.bid_ask_ratio > self.BID_ASK_RATIO_BULLISH:
                # Place limit at best bid
                bid_price = micro.ltp - micro.spread_paisa
                return max(bid_price, 0)

        # If price is above VWAP, wait for pullback
        if micro.vwap > 0 and micro.vwap_distance > self.VWAP_THRESHOLD_PCT:
            return micro.vwap

        # Default: use LTP or signal price
        return signal.price or micro.ltp

    def _calculate_sell_entry_price(
        self, signal: Signal, micro: MicrostructureData, spread_pct: float
    ) -> int:
        """Calculate optimal sell entry price."""
        # If tight spread and bearish pressure, place limit at ask for price improvement
        if spread_pct < self.SPREAD_THRESHOLD_PCT:
            if micro.bid_ask_ratio < self.BID_ASK_RATIO_BEARISH:
                # Place limit at best ask
                ask_price = micro.ltp + micro.spread_paisa
                return ask_price

        # If price is below VWAP, wait for pullback up to VWAP
        if micro.vwap > 0 and micro.vwap_distance < -self.VWAP_THRESHOLD_PCT:
            return micro.vwap

        # Default: use LTP or signal price
        return signal.price or micro.ltp

    def _analyze_buy_entry(
        self, signal: Signal, micro: MicrostructureData
    ) -> tuple[EntryType, str]:
        """Analyze buy signal and determine entry type with reasoning."""
        # Calculate spread percentage
        if micro.ltp > 0:
            spread_pct = (micro.spread_paisa / micro.ltp) * 100
        else:
            spread_pct = float("inf")

        # Check for VWAP pullback opportunity
        if micro.vwap > 0 and micro.vwap_distance > self.VWAP_THRESHOLD_PCT:
            distance_pct = abs(micro.vwap_distance)
            return (
                EntryType.VWAP,
                f"Price {distance_pct:.2f}% above VWAP, waiting for pullback to VWAP at {micro.vwap}",
            )

        # Check for tight spread with bullish pressure (limit at bid)
        if spread_pct < self.SPREAD_THRESHOLD_PCT:
            if micro.bid_ask_ratio > self.BID_ASK_RATIO_BULLISH:
                return (
                    EntryType.IMMEDIATE,
                    f"Tight spread ({spread_pct:.3f}%) with bullish pressure (ratio: {micro.bid_ask_ratio:.2f}), "
                    f"placing limit at bid for price improvement",
                )

        # Check for breakout scenario (price near resistance with volume)
        if micro.pressure == "buy" and micro.bid_qty > micro.ask_qty * 2:
            return (
                EntryType.BREAKOUT,
                f"Breakout conditions: strong buy pressure, waiting for confirmation",
            )

        # Default to immediate market order
        return (
            EntryType.IMMEDIATE,
            f"Standard market conditions, using immediate entry (spread: {spread_pct:.3f}%)",
        )

    def _analyze_sell_entry(
        self, signal: Signal, micro: MicrostructureData
    ) -> tuple[EntryType, str]:
        """Analyze sell signal and determine entry type with reasoning."""
        # Calculate spread percentage
        if micro.ltp > 0:
            spread_pct = (micro.spread_paisa / micro.ltp) * 100
        else:
            spread_pct = float("inf")

        # Check for VWAP pullback opportunity (price below VWAP, wait for retrace up)
        if micro.vwap > 0 and micro.vwap_distance < -self.VWAP_THRESHOLD_PCT:
            distance_pct = abs(micro.vwap_distance)
            return (
                EntryType.VWAP,
                f"Price {distance_pct:.2f}% below VWAP, waiting for pullback to VWAP at {micro.vwap}",
            )

        # Check for tight spread with bearish pressure (limit at ask)
        if spread_pct < self.SPREAD_THRESHOLD_PCT:
            if micro.bid_ask_ratio < self.BID_ASK_RATIO_BEARISH:
                return (
                    EntryType.IMMEDIATE,
                    f"Tight spread ({spread_pct:.3f}%) with bearish pressure (ratio: {micro.bid_ask_ratio:.2f}), "
                    f"placing limit at ask for price improvement",
                )

        # Check for breakdown scenario
        if micro.pressure == "sell" and micro.ask_qty > micro.bid_qty * 2:
            return (
                EntryType.BREAKOUT,
                f"Breakdown conditions: strong sell pressure, waiting for confirmation",
            )

        # Default to immediate market order
        return (
            EntryType.IMMEDIATE,
            f"Standard market conditions, using immediate entry (spread: {spread_pct:.3f}%)",
        )

    def should_wait_for_pullback(
        self, signal: Signal, micro: MicrostructureData
    ) -> bool:
        """Determine if we should wait for a price pullback before entry.

        Args:
            signal: The trading signal
            micro: Market microstructure data

        Returns:
            True if waiting for pullback is recommended
        """
        is_buy = signal.signal_type in (
            SignalType.BUY,
            SignalType.BUY_CE,
            SignalType.BUY_PE,
        )

        if is_buy:
            # For buys: wait if price is extended above VWAP
            return micro.vwap > 0 and micro.vwap_distance > self.VWAP_THRESHOLD_PCT
        else:
            # For sells: wait if price is extended below VWAP
            return micro.vwap > 0 and micro.vwap_distance < -self.VWAP_THRESHOLD_PCT

    def should_wait_for_breakout(
        self, signal: Signal, micro: MicrostructureData
    ) -> bool:
        """Determine if we should wait for breakout confirmation.

        Args:
            signal: The trading signal
            micro: Market microstructure data

        Returns:
            True if waiting for breakout confirmation is recommended
        """
        # Breakout conditions: strong pressure in signal direction
        is_buy = signal.signal_type in (
            SignalType.BUY,
            SignalType.BUY_CE,
            SignalType.BUY_PE,
        )

        if is_buy:
            return micro.pressure == "buy" and micro.bid_qty > micro.ask_qty * 2
        else:
            return micro.pressure == "sell" and micro.ask_qty > micro.bid_qty * 2

    def get_time_based_fallback_stages(self) -> list[dict[str, Any]]:
        """Get the time-based fallback stages for entry execution.

        Returns a list of stages to attempt in order:
        1. Try limit order at optimal price for LIMIT_TIMEOUT seconds
        2. Try at mid-price for MID_PRICE_TIMEOUT seconds
        3. Use market order (never miss entry)

        Returns:
            List of stage configurations with timeout and price adjustment
        """
        return [
            {
                "stage": 1,
                "name": "limit_optimal",
                "timeout": self.LIMIT_TIMEOUT,
                "price_adjustment": 0,  # No adjustment
                "description": "Try limit order at optimal price",
            },
            {
                "stage": 2,
                "name": "limit_mid",
                "timeout": self.MID_PRICE_TIMEOUT,
                "price_adjustment": 0.5,  # Move 50% toward market price
                "description": "Try at mid-price (compromise)",
            },
            {
                "stage": 3,
                "name": "market_fallback",
                "timeout": 0,
                "price_adjustment": 1.0,  # Full market price
                "description": "Fallback to market order - never miss entry",
            },
        ]

    def record_fill_result(self, entry_type: EntryType, filled: bool) -> None:
        """Record the fill result for statistics tracking.

        Args:
            entry_type: The type of entry that was attempted
            filled: Whether the order was filled
        """
        with self._lock:
            if filled:
                self._stats["fill_rates"][entry_type.value]["filled"] += 1

    def get_statistics(self) -> dict[str, Any]:
        """Get current optimization statistics.

        Returns:
            Dictionary with fill rates, counts by type, and totals
        """
        with self._lock:
            stats = self._stats.copy()

            # Calculate fill rates
            for entry_type in EntryType:
                attempted = stats["fill_rates"][entry_type.value]["attempted"]
                filled = stats["fill_rates"][entry_type.value]["filled"]
                stats["fill_rates"][entry_type.value]["rate"] = (
                    (filled / attempted * 100) if attempted > 0 else 0.0
                )

            return stats

    def reset_statistics(self) -> None:
        """Reset all statistics counters."""
        with self._lock:
            self._stats = {
                "total_signals": 0,
                "by_type": {
                    EntryType.IMMEDIATE.value: 0,
                    EntryType.PULLBACK.value: 0,
                    EntryType.BREAKOUT.value: 0,
                    EntryType.VWAP.value: 0,
                },
                "fill_rates": {
                    EntryType.IMMEDIATE.value: {"attempted": 0, "filled": 0},
                    EntryType.PULLBACK.value: {"attempted": 0, "filled": 0},
                    EntryType.BREAKOUT.value: {"attempted": 0, "filled": 0},
                    EntryType.VWAP.value: {"attempted": 0, "filled": 0},
                },
            }

    def calculate_mid_price(self, bid: int, ask: int) -> int:
        """Calculate mid price between bid and ask.

        Args:
            bid: Bid price in paisa
            ask: Ask price in paisa

        Returns:
            Mid price in paisa (rounded to nearest tick)
        """
        # NSE tick size is 0.05 rupees = 5 paisa
        TICK_SIZE_PAISA = 5
        mid = (bid + ask) // 2
        # Round to nearest tick
        return (mid // TICK_SIZE_PAISA) * TICK_SIZE_PAISA
