"""Risk management module for the trading bot.

Enforces per-trade, per-position, and daily risk limits before any order
is submitted.  All monetary values are in **paisa** (1 INR = 100 paisa)
to avoid floating-point precision issues.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from datetime import datetime, time
from typing import Optional
from zoneinfo import ZoneInfo

from loguru import logger

IST = ZoneInfo("Asia/Kolkata")


@dataclass(frozen=True, slots=True)
class RiskCheckResult:
    """Result of a single risk-management check.

    Attributes:
        approved: Whether the check passed.
        reason: Human-readable explanation when the check fails.  ``None``
            when approved is ``True``.
    """

    approved: bool
    reason: str | None = None


class RiskManager:
    """Centralised risk-management gate.

    Every order must pass :meth:`pre_trade_check` before it reaches the
    broker layer.  The manager tracks realised/unrealised P&L, open
    position count, and deployed capital so that limits are enforced in
    real time.

    All monetary parameters and internal counters are in **paisa**.

    Args:
        capital: Total trading capital in paisa.
        max_daily_loss: Maximum allowable daily loss in paisa (positive
            value — e.g. ``300_000_00`` for INR 3,00,000).
        max_open_positions: Maximum number of simultaneous open positions.
        max_position_size_pct: Maximum single-position size as a
            percentage of capital (e.g. ``20`` for 20 %).
        max_capital_deployment_pct: Maximum total capital that may be
            deployed at any time, as a percentage (default ``80``).
    """

    def __init__(
        self,
        capital: int,
        max_daily_loss: int,
        max_open_positions: int,
        max_position_size_pct: int,
        max_capital_deployment_pct: int = 80,
        max_trades_per_day: int = 3,
    ) -> None:
        self._lock = threading.Lock()

        # Configuration (immutable after init)
        self.capital: int = capital
        self.max_daily_loss: int = max_daily_loss
        self.max_open_positions: int = max_open_positions
        self.max_position_size_pct: int = max_position_size_pct
        self.max_capital_deployment_pct: int = max_capital_deployment_pct
        self.max_trades_per_day: int = max_trades_per_day

        # Mutable state — protected by ``_lock``
        self.realized_pnl: int = 0
        self.unrealized_pnl: int = 0
        self.open_position_count: int = 0
        self.deployed_capital: int = 0

        # Daily trade tracking: symbol -> list of trade timestamps
        self._daily_trades: dict[str, list[datetime]] = {}

        logger.info(
            "RiskManager initialised | capital={} paisa | max_daily_loss={} paisa | "
            "max_positions={} | max_position_size={}% | max_deployment={}% | max_trades_per_day={}",
            capital,
            max_daily_loss,
            max_open_positions,
            max_position_size_pct,
            max_capital_deployment_pct,
            max_trades_per_day,
        )

    # ------------------------------------------------------------------
    # Public check API
    # ------------------------------------------------------------------

    def pre_trade_check(self, order_value: int, instrument_key: str) -> RiskCheckResult:
        """Run **all** risk checks for a proposed order.

        Checks are executed in a fixed order; the first failure is
        returned immediately so the caller knows the exact reason.

        Args:
            order_value: Notional value of the proposed order in paisa.
            instrument_key: Exchange-qualified instrument identifier
                (e.g. ``"NSE_EQ:RELIANCE"``).

        Returns:
            :class:`RiskCheckResult` — approved if every check passes,
            otherwise the first failing check result.
        """
        checks: list[RiskCheckResult] = [
            self.check_daily_loss_limit(),
            self.check_position_count(),
            self.check_position_size(order_value),
            self.check_capital_deployment(order_value),
        ]

        for result in checks:
            if not result.approved:
                logger.warning(
                    "Pre-trade check FAILED for {} (order_value={} paisa): {}",
                    instrument_key,
                    order_value,
                    result.reason,
                )
                return result

        logger.debug(
            "Pre-trade check PASSED for {} (order_value={} paisa)",
            instrument_key,
            order_value,
        )
        return RiskCheckResult(approved=True)

    def check_daily_loss_limit(self) -> RiskCheckResult:
        """Check whether the daily loss limit has been breached.

        The total loss is ``realized_pnl + unrealized_pnl``.  Because
        losses are negative, the check triggers when the absolute loss
        exceeds :attr:`max_daily_loss`.

        Returns:
            :class:`RiskCheckResult`
        """
        with self._lock:
            total_pnl = self.realized_pnl + self.unrealized_pnl

        if total_pnl <= -self.max_daily_loss:
            reason = (
                f"Daily loss limit breached: current P&L {total_pnl} paisa "
                f"exceeds max allowed loss of -{self.max_daily_loss} paisa"
            )
            logger.warning(reason)
            return RiskCheckResult(approved=False, reason=reason)

        return RiskCheckResult(approved=True)

    def check_position_count(self) -> RiskCheckResult:
        """Check whether the maximum open-position count is reached.

        Returns:
            :class:`RiskCheckResult`
        """
        with self._lock:
            count = self.open_position_count
            if count >= self.max_open_positions:
                reason = f"Max open positions reached: {count}/{self.max_open_positions}"
                logger.warning(reason)
                return RiskCheckResult(approved=False, reason=reason)

        return RiskCheckResult(approved=True)

    def check_position_size(self, order_value: int) -> RiskCheckResult:
        """Check that a single position does not exceed max_position_size_pct.

        Args:
            order_value: Notional value of the proposed order in paisa.

        Returns:
            :class:`RiskCheckResult`
        """
        max_allowed = int(self.capital * self.max_position_size_pct / 100)
        if order_value > max_allowed:
            reason = (
                f"Position size {order_value} paisa exceeds "
                f"{self.max_position_size_pct}% of capital ({max_allowed} paisa)"
            )
            logger.warning(reason)
            return RiskCheckResult(approved=False, reason=reason)
        return RiskCheckResult(approved=True)

    def check_circuit_limits(
        self, price: int, stop_loss: int | None, target: int | None
    ) -> RiskCheckResult:
        """Check if stop loss and target are within circuit limits.

        Args:
            price: Current price in paisa.
            stop_loss: Stop loss price in paisa.
            target: Target price in paisa.

        Returns:
            RiskCheckResult indicating if within limits.
        """
        # For now, just check that SL and target are on the correct side of price
        # More sophisticated checks could include:
        # - Maximum allowed move (circuit limits)
        # - Tick size validation
        # - Price band validation

        if stop_loss is not None and price is not None and price > 0:
            if price > stop_loss:  # BUY position
                if price - stop_loss > 0.90 * price:  # 90% away (allows option selling wide SL)
                    return RiskCheckResult(
                        approved=False,
                        reason=f"STOP LOSS too far from entry: {stop_loss} vs {price}",
                    )
            else:  # SELL position
                if stop_loss - price > 0.90 * price:  # 90% away (allows option selling wide SL)
                    return RiskCheckResult(
                        approved=False,
                        reason=f"STOP LOSS too far from entry: {stop_loss} vs {price}",
                    )

        if target is not None and price is not None and price > 0:
            if price < target:  # BUY position
                if target - price > 0.90 * price:  # 90% away
                    return RiskCheckResult(
                        approved=False, reason=f"TARGET too far from entry: {target} vs {price}"
                    )
            else:  # SELL position
                if price - target > 0.90 * price:  # 90% away
                    return RiskCheckResult(
                        approved=False, reason=f"TARGET too far from entry: {target} vs {price}"
                    )

        return RiskCheckResult(approved=True)

    def check_capital_deployment(self, order_value: int) -> RiskCheckResult:
        """Check that total deployed capital stays within the cap.

        Args:
            order_value: Notional value of the proposed order in paisa.

        Returns:
            :class:`RiskCheckResult`
        """
        max_deployment = self.max_capital_deployment_pct * self.capital // 100

        with self._lock:
            projected = self.deployed_capital + order_value

        if projected > max_deployment:
            reason = (
                f"Capital deployment limit breached: deployed={self.deployed_capital} paisa "
                f"+ order={order_value} paisa = {projected} paisa "
                f"exceeds {self.max_capital_deployment_pct}% of capital "
                f"({max_deployment} paisa)"
            )
            logger.warning(reason)
            return RiskCheckResult(approved=False, reason=reason)

        return RiskCheckResult(approved=True)

    # ------------------------------------------------------------------
    # State update helpers
    # ------------------------------------------------------------------

    def update_pnl(self, realized: int, unrealized: int) -> None:
        """Update tracked P&L counters.

        Args:
            realized: New cumulative realised P&L in paisa.
            unrealized: New cumulative unrealised P&L in paisa.
        """
        with self._lock:
            self.realized_pnl = realized
            self.unrealized_pnl = unrealized

        logger.debug(
            "P&L updated | realized={} paisa | unrealized={} paisa",
            realized,
            unrealized,
        )

    def update_positions(self, count: int, deployed: int) -> None:
        """Update position-tracking counters.

        Args:
            count: Current number of open positions.
            deployed: Total capital currently deployed in paisa.
        """
        with self._lock:
            self.open_position_count = count
            self.deployed_capital = deployed

        logger.debug(
            "Positions updated | count={} | deployed={} paisa",
            count,
            deployed,
        )

    def reset_daily(self) -> None:
        """Reset daily counters at the start of a new trading day."""
        with self._lock:
            self.realized_pnl = 0
            self.unrealized_pnl = 0
            self._daily_trades.clear()

        logger.info("Daily risk counters reset")

    # ------------------------------------------------------------------
    # Daily trade limit tracking
    # ------------------------------------------------------------------

    def can_take_new_trade(
        self, symbol: str, timestamp: Optional[datetime] = None
    ) -> tuple[bool, str]:
        """Check if we can take a new trade for the symbol.

        PRO TRADER MODE: No arbitrary trade limits. Trade as much as opportunity allows.
        Only block if we've hit catastrophic daily loss or have too many correlated positions.

        Args:
            symbol: The instrument symbol to check.
            timestamp: Optional timestamp (defaults to now in IST).

        Returns:
            Tuple of (allowed, reason). Allowed is True if trade can proceed,
            False otherwise with reason explaining why.
        """
        timestamp = timestamp or datetime.now(IST)

        # Get today's trades for this symbol (for tracking, not limiting)
        today_trades = self._get_todays_trades(symbol, timestamp)
        trade_count = len(today_trades)

        # Enforce max daily trade limit (even in PRO MODE — straddles enter once/day)
        if trade_count >= self.max_trades_per_day:
            return (
                False,
                f"Daily trade limit reached: {trade_count}/{self.max_trades_per_day} trades for {symbol}",
            )

        # Log trade opportunity
        if trade_count > 0:
            logger.info(
                f"PRO TRADER MODE: Trade {trade_count + 1} for {symbol} - Opportunity accepted"
            )

        return True, "OK"

    def record_trade(self, symbol: str, timestamp: Optional[datetime] = None) -> None:
        """Record a trade for daily limit tracking.

        Args:
            symbol: The instrument symbol that was traded.
            timestamp: Optional timestamp (defaults to now in IST).
        """
        timestamp = timestamp or datetime.now(IST)

        with self._lock:
            if symbol not in self._daily_trades:
                self._daily_trades[symbol] = []

            self._daily_trades[symbol].append(timestamp)
            trade_count = len(self._daily_trades[symbol])

        logger.info(f"Trade recorded for {symbol}. Today: {trade_count}/{self.max_trades_per_day}")

    def _get_todays_trades(self, symbol: str, timestamp: datetime) -> list[datetime]:
        """Get today's trades for symbol, cleaning up old entries.

        Args:
            symbol: The instrument symbol.
            timestamp: The reference timestamp.

        Returns:
            List of timestamps for today's trades.
        """
        with self._lock:
            if symbol not in self._daily_trades:
                return []

            today = timestamp.date()
            todays_trades = [t for t in self._daily_trades[symbol] if t.date() == today]
            self._daily_trades[symbol] = todays_trades
            return todays_trades

    def get_daily_trade_count(self, symbol: str, as_of: Optional[datetime] = None) -> int:
        """Get number of trades taken today for symbol.

        Args:
            symbol: The instrument symbol.
            as_of: Reference timestamp (defaults to now in IST). Use for backtests
                so the count matches the date of the last recorded trade.

        Returns:
            Number of trades taken on as_of's date.
        """
        if as_of is None:
            with self._lock:
                trades = self._daily_trades.get(symbol, [])
                if trades:
                    as_of = max(trades)
                else:
                    as_of = datetime.now(IST)
        return len(self._get_todays_trades(symbol, as_of))
