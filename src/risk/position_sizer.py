"""Kelly Criterion position sizing module for PRO TRADER MODE.

This module implements the Kelly Criterion formula for optimal position sizing
that MAXIMIZES PROFITS. In PRO TRADER MODE, we use Full Kelly (not Half-Kelly)
to maximize returns since we have an edge.

All monetary values are in **paisa** (1 INR = 100 paisa) to avoid
floating-point precision issues.

PHILOSOPHY:
- Trade as much as opportunity allows (no arbitrary limits)
- Size positions based on edge (Kelly Criterion)
- Compound winners aggressively
- Cut losers quickly but don't limit opportunity count
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Optional

from loguru import logger


@dataclass
class TradeResult:
    """Represents a completed trade for Kelly calculation.

    Attributes:
        pnl: Realized profit/loss in paisa (positive for win, negative for loss).
        entry_price: Entry price in paisa.
        exit_price: Exit price in paisa.
        quantity: Number of units traded.
    """

    pnl: int
    entry_price: int
    exit_price: int
    quantity: int

    @property
    def is_win(self) -> bool:
        """Return True if this trade was profitable."""
        return self.pnl > 0

    @property
    def is_loss(self) -> bool:
        """Return True if this trade was a loss."""
        return self.pnl < 0

    @property
    def absolute_pnl(self) -> int:
        """Return absolute value of P&L."""
        return abs(self.pnl)


@dataclass
class KellyStats:
    """Statistics used for Kelly Criterion calculation.

    Attributes:
        win_rate: Probability of winning (0.0 to 1.0).
        loss_rate: Probability of losing (0.0 to 1.0).
        avg_win: Average winning trade amount in paisa.
        avg_loss: Average losing trade amount in paisa.
        win_loss_ratio: Ratio of average win to average loss (b in Kelly formula).
        kelly_fraction: Raw Kelly fraction (can be negative).
        sample_size: Number of trades used in calculation.
    """

    win_rate: float = 0.0
    loss_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    win_loss_ratio: float = 0.0
    kelly_fraction: float = 0.0
    sample_size: int = 0


class KellyPositionSizer:
    """Optimal position sizing using Kelly Criterion for maximum long-term growth.

    The Kelly Criterion formula: K% = (p*b - q) / b
    Where:
        p = win rate (probability of winning)
        q = loss rate = 1 - p
        b = average win / average loss (win/loss ratio)

    PRO TRADER MODE: Defaults to Full Kelly for maximum profit. Can optionally
    use Half-Kelly for reduced volatility, but aggressive sizing is the default.

    All monetary values are handled in **paisa** (integer).

    Args:
        half_kelly: If True, use Half-Kelly for safety (default False for max profit).
        max_kelly_pct: Maximum Kelly percentage cap (default 0.40 = 40%).
        min_trades_for_kelly: Minimum trades required before Kelly is valid (default 10).
        fallback_pct: Fallback position size when insufficient history (default 0.05 = 5%).
    """

    def __init__(
        self,
        half_kelly: bool = False,
        max_kelly_pct: float = 0.40,
        min_trades_for_kelly: int = 10,
        fallback_pct: float = 0.05,
    ) -> None:
        self._lock = threading.Lock()

        # Configuration
        self.half_kelly: bool = half_kelly
        self.max_kelly_pct: float = max(max_kelly_pct, 0.01)  # Min 1%
        self.min_trades_for_kelly: int = max(min_trades_for_kelly, 5)
        self.fallback_pct: float = fallback_pct

        # Trade history - thread-safe list
        self._trade_history: list[TradeResult] = []

        # Cached statistics (invalidated when new trades added)
        self._cached_stats: Optional[KellyStats] = None
        self._stats_dirty: bool = True

        logger.info(
            "KellyPositionSizer initialized | half_kelly={} | max_kelly_pct={:.1%} | "
            "min_trades={} | fallback_pct={:.1%}",
            half_kelly,
            max_kelly_pct,
            min_trades_for_kelly,
            fallback_pct,
        )

    # ------------------------------------------------------------------
    # Trade History Management
    # ------------------------------------------------------------------

    def add_trade(self, trade: TradeResult) -> None:
        """Add a completed trade to the history.

        Args:
            trade: The completed trade result.
        """
        with self._lock:
            self._trade_history.append(trade)
            self._stats_dirty = True

        logger.debug(
            "Added trade to Kelly history | pnl={} paisa | total_trades={}",
            trade.pnl,
            len(self._trade_history),
        )

    def add_trades(self, trades: list[TradeResult]) -> None:
        """Add multiple trades to the history.

        Args:
            trades: List of completed trade results.
        """
        with self._lock:
            self._trade_history.extend(trades)
            self._stats_dirty = True

        logger.debug(
            "Added {} trades to Kelly history | total_trades={}",
            len(trades),
            len(self._trade_history),
        )

    def clear_history(self) -> None:
        """Clear all trade history."""
        with self._lock:
            self._trade_history.clear()
            self._stats_dirty = True
            self._cached_stats = None

        logger.info("Kelly trade history cleared")

    def get_trade_count(self) -> int:
        """Get the number of trades in history.

        Returns:
            Number of completed trades.
        """
        with self._lock:
            return len(self._trade_history)

    def has_sufficient_history(self) -> bool:
        """Check if there are enough trades for reliable Kelly calculation.

        Returns:
            True if trade count >= min_trades_for_kelly.
        """
        with self._lock:
            return len(self._trade_history) >= self.min_trades_for_kelly

    # ------------------------------------------------------------------
    # Kelly Calculation
    # ------------------------------------------------------------------

    def calculate_kelly_fraction(self, trade_history: Optional[list[TradeResult]] = None) -> float:
        """Calculate the Kelly Criterion fraction from trade history.

        Formula: K = (p*b - q) / b
        Where:
            p = win rate
            q = 1 - p (loss rate)
            b = average win / average loss

        Args:
            trade_history: Optional list of trades to use. If None, uses internal history.

        Returns:
            Kelly fraction (0.0 to 1.0, can be negative if edge is negative).
            Returns 0.0 if insufficient data or calculation fails.
        """
        trades = trade_history or self._get_trade_history()

        if len(trades) < self.min_trades_for_kelly:
            logger.debug(
                "Insufficient trades for Kelly calculation | trades={} | min_required={}",
                len(trades),
                self.min_trades_for_kelly,
            )
            return 0.0

        # Separate wins and losses
        wins = [t for t in trades if t.is_win]
        losses = [t for t in trades if t.is_loss]

        if not wins or not losses:
            logger.warning(
                "Cannot calculate Kelly: no {} trades in history | wins={} | losses={}",
                "losing" if not losses else "winning",
                len(wins),
                len(losses),
            )
            return 0.0

        # Calculate statistics
        total_trades = len(trades)
        win_count = len(wins)
        loss_count = len(losses)

        p = win_count / total_trades  # Win rate
        q = loss_count / total_trades  # Loss rate

        avg_win = sum(t.pnl for t in wins) / win_count
        avg_loss = sum(abs(t.pnl) for t in losses) / loss_count

        # Edge case: zero average loss
        if avg_loss <= 0:
            logger.warning("Average loss is zero or negative, cannot calculate Kelly")
            return 0.0

        b = avg_win / avg_loss  # Win/loss ratio

        # Kelly formula: K = (p*b - q) / b
        kelly = (p * b - q) / b

        logger.debug(
            "Kelly calculation | trades={} | win_rate={:.2%} | avg_win={} | "
            "avg_loss={} | W/L_ratio={:.2f} | kelly={:.4f}",
            total_trades,
            p,
            avg_win,
            avg_loss,
            b,
            kelly,
        )

        return kelly

    def get_safe_kelly(self, kelly: float) -> float:
        """Apply safety constraints to raw Kelly fraction.

        Constraints applied:
        1. If Kelly is negative, return 0 (don't trade without edge).
        2. Apply Half-Kelly if configured (divide by 2).
        3. Cap at max_kelly_pct to prevent excessive sizing.

        Args:
            kelly: Raw Kelly fraction from calculate_kelly_fraction().

        Returns:
            Safe Kelly fraction ready for position sizing.
        """
        # Don't trade if no edge
        if kelly <= 0:
            logger.debug("Kelly fraction is non-positive ({}), no position", kelly)
            return 0.0

        # Apply Half-Kelly for safety
        if self.half_kelly:
            kelly = kelly / 2.0
            logger.debug("Applied Half-Kelly: {} -> {}", kelly * 2, kelly)

        # Cap at maximum percentage
        if kelly > self.max_kelly_pct:
            logger.debug(
                "Capped Kelly at max: {:.2%} -> {:.2%}",
                kelly,
                self.max_kelly_pct,
            )
            kelly = self.max_kelly_pct

        return kelly

    def get_position_size(
        self,
        capital: int,
        current_price: int,
        max_risk_per_trade_pct: Optional[float] = None,
        trade_history: Optional[list[TradeResult]] = None,
    ) -> int:
        """Calculate position size using Kelly Criterion with safety constraints.

        The position fraction is determined by Kelly, already capped by
        ``self.max_kelly_pct`` inside :py:meth:`get_safe_kelly`.  An
        *additional* cap can be applied via ``max_risk_per_trade_pct`` —
        but only when the caller explicitly passes a value tighter than
        Kelly.  The previous default of 0.01 silently truncated every
        Kelly-sized position to 1 % of capital regardless of edge, which
        defeated the purpose of using Kelly in the first place.

        Args:
            capital: Available capital in paisa.
            current_price: Current market price in paisa.
            max_risk_per_trade_pct: Optional hard cap on position size as a
                fraction of capital. If ``None`` (default), Kelly's own
                ``max_kelly_pct`` cap is the only ceiling.
            trade_history: Optional external trade history. If None, uses
                internal history.

        Returns:
            Quantity to trade (in units, rounded down).
            Returns 0 if Kelly suggests no position or constraints block it.
        """
        if capital <= 0:
            logger.warning("Cannot calculate position size: capital is zero or negative")
            return 0

        if current_price <= 0:
            logger.warning("Cannot calculate position size: invalid price")
            return 0

        # Calculate Kelly fraction
        raw_kelly = self.calculate_kelly_fraction(trade_history)
        safe_kelly = self.get_safe_kelly(raw_kelly)

        # If no Kelly edge, use fallback percentage
        if safe_kelly <= 0:
            safe_kelly = self.fallback_pct
            logger.debug(
                "Using fallback position size: {:.1%} of capital",
                safe_kelly,
            )

        # Calculate position value based on Kelly (already capped at max_kelly_pct)
        position_value = int(capital * safe_kelly)

        # Optional caller-supplied cap. None = trust Kelly's own ceiling.
        if max_risk_per_trade_pct is not None:
            max_risk_value = int(capital * max_risk_per_trade_pct)
            if position_value > max_risk_value:
                logger.debug(
                    "Position capped by caller-supplied risk limit: {} -> {} paisa",
                    position_value,
                    max_risk_value,
                )
                position_value = max_risk_value

        # Convert to quantity
        quantity = position_value // current_price

        logger.info(
            "Kelly position sizing | capital={} | kelly={:.4f} | safe_kelly={:.4f} | "
            "position_value={} | quantity={}",
            capital,
            raw_kelly,
            safe_kelly,
            position_value,
            quantity,
        )

        return max(quantity, 0)

    def get_kelly_stats(self) -> KellyStats:
        """Get detailed Kelly statistics from trade history.

        Returns:
            KellyStats object with win rate, ratios, and calculated fraction.
        """
        with self._lock:
            if not self._stats_dirty and self._cached_stats:
                return self._cached_stats

            trades = list(self._trade_history)

        if len(trades) < self.min_trades_for_kelly:
            return KellyStats(sample_size=len(trades))

        wins = [t for t in trades if t.is_win]
        losses = [t for t in trades if t.is_loss]

        if not wins or not losses:
            return KellyStats(sample_size=len(trades))

        total_trades = len(trades)
        win_count = len(wins)
        loss_count = len(losses)

        p = win_count / total_trades
        q = loss_count / total_trades

        avg_win = sum(t.pnl for t in wins) / win_count
        avg_loss = sum(abs(t.pnl) for t in losses) / loss_count

        b = avg_win / avg_loss if avg_loss > 0 else 0.0
        kelly = (p * b - q) / b if b > 0 else 0.0

        stats = KellyStats(
            win_rate=p,
            loss_rate=q,
            avg_win=avg_win,
            avg_loss=avg_loss,
            win_loss_ratio=b,
            kelly_fraction=kelly,
            sample_size=total_trades,
        )

        with self._lock:
            self._cached_stats = stats
            self._stats_dirty = False

        return stats

    def get_recommended_leverage(self) -> float:
        """Calculate recommended leverage based on Kelly statistics.

        This is useful for strategies that use leverage (F&O).

        Returns:
            Recommended leverage multiplier (1.0 = no leverage).
        """
        stats = self.get_kelly_stats()

        if stats.sample_size < self.min_trades_for_kelly:
            return 1.0

        # Conservative leverage: Half-Kelly * 2 (to get back to full Kelly)
        # But cap at reasonable maximum
        leverage = stats.kelly_fraction * 2.0  # Undo Half-Kelly

        if self.half_kelly:
            # Apply Half-Kelly logic to leverage
            leverage = leverage / 2.0

        # Cap leverage at 3x for safety
        leverage = min(leverage, 3.0)

        # Don't recommend leverage if edge is negative
        return max(leverage, 1.0)

    # ------------------------------------------------------------------
    # Private Helpers
    # ------------------------------------------------------------------

    def _get_trade_history(self) -> list[TradeResult]:
        """Get a copy of trade history (thread-safe).

        Returns:
            Copy of internal trade history list.
        """
        with self._lock:
            return list(self._trade_history)


def create_kelly_sizer_from_trades(
    trades: list[tuple[int, int, int, int]],
    half_kelly: bool = True,
    max_kelly_pct: float = 0.25,
) -> KellyPositionSizer:
    """Convenience factory to create a Kelly sizer from raw trade data.

    Args:
        trades: List of tuples (pnl, entry_price, exit_price, quantity).
        half_kelly: Use Half-Kelly safety (default True).
        max_kelly_pct: Maximum Kelly percentage cap.

    Returns:
        Configured KellyPositionSizer with trades loaded.
    """
    sizer = KellyPositionSizer(half_kelly=half_kelly, max_kelly_pct=max_kelly_pct)

    trade_results = [
        TradeResult(pnl=p, entry_price=e, exit_price=x, quantity=q)
        for p, e, x, q in trades
    ]

    sizer.add_trades(trade_results)
    return sizer


# Example usage and calculation verification
def example_kelly_calculation() -> None:
    """Example demonstrating Kelly Criterion calculation.

    Example:
        Win rate 60%, avg win 200, avg loss 100
        b = 200/100 = 2
        Kelly = (0.6*2 - 0.4) / 2 = 0.4 (40%)
        Half-Kelly = 20%
        For 100,000 capital = 20,000 position
    """
    # Create sample trade history
    trades: list[TradeResult] = []

    # 60% win rate: 12 wins of +200 each
    for _ in range(12):
        trades.append(TradeResult(pnl=20000, entry_price=10000, exit_price=10200, quantity=100))

    # 40% loss rate: 8 losses of -100 each
    for _ in range(8):
        trades.append(TradeResult(pnl=-10000, entry_price=10000, exit_price=9900, quantity=100))

    # Create sizer
    sizer = KellyPositionSizer(half_kelly=True, max_kelly_pct=0.25)
    sizer.add_trades(trades)

    # Calculate
    stats = sizer.get_kelly_stats()
    kelly = sizer.calculate_kelly_fraction()
    safe_kelly = sizer.get_safe_kelly(kelly)

    print(f"Win rate: {stats.win_rate:.1%}")
    print(f"Avg win: {stats.avg_win} paisa")
    print(f"Avg loss: {stats.avg_loss} paisa")
    print(f"W/L ratio: {stats.win_loss_ratio:.2f}")
    print(f"Raw Kelly: {kelly:.2%}")
    print(f"Safe Kelly (Half-Kelly): {safe_kelly:.2%}")

    # Position size for 100,000 INR capital, 100 INR stock
    capital = 10000000  # 100,000 INR in paisa
    price = 10000  # 100 INR in paisa
    quantity = sizer.get_position_size(capital, price)

    print(f"\nFor capital={capital} paisa, price={price} paisa:")
    print(f"Quantity: {quantity}")
    print(f"Position value: {quantity * price} paisa ({quantity * price / capital:.1%} of capital)")


if __name__ == "__main__":
    example_kelly_calculation()
