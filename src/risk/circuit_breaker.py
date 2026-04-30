"""Circuit breaker and kill-switch module.

Halts trading automatically after a configurable number of consecutive
losses and provides a manual kill switch for emergency shutdowns.
All timestamps use IST (``Asia/Kolkata``).
"""

from __future__ import annotations

import threading
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Protocol
from zoneinfo import ZoneInfo

from loguru import logger

if TYPE_CHECKING:
    pass

IST = ZoneInfo("Asia/Kolkata")


class OrderManagerProtocol(Protocol):
    """Minimal interface expected from an order manager by the kill switch."""

    def cancel_all_orders(self) -> None: ...
    def exit_all_positions(self) -> None: ...


class CircuitBreaker:
    """Automatic and manual trading halt mechanism.

    The circuit breaker tracks consecutive losing trades.  When the
    streak reaches *max_consecutive_losses*, trading is paused for
    *pause_minutes*.  After the pause elapses, :meth:`is_halted`
    automatically resumes normal operation.

    A manual :meth:`kill_switch` is also available for emergencies — it
    halts trading indefinitely and (optionally) liquidates all positions.

    Args:
        max_consecutive_losses: Number of consecutive losses that
            triggers an automatic pause.
        pause_minutes: Duration of the automatic pause in minutes.
    """

    def __init__(
        self,
        max_consecutive_losses: int = 3,
        pause_minutes: int = 30,
    ) -> None:
        self._lock = threading.Lock()

        # Configuration
        self.max_consecutive_losses: int = max_consecutive_losses
        self.pause_minutes: int = pause_minutes

        # Mutable state — protected by ``_lock``
        self.consecutive_losses: int = 0
        self.halted: bool = False
        self.halt_until: datetime | None = None

        logger.info(
            "CircuitBreaker initialised | max_consecutive_losses={} | pause_minutes={}",
            max_consecutive_losses,
            pause_minutes,
        )

        # Backtest mode: inject bar time to replace wall-clock in halt logic
        self._backtest_time: datetime | None = None

    def set_backtest_time(self, dt: datetime) -> None:
        """Inject the current backtest bar timestamp.

        Call this before each bar in the backtest loop so the circuit
        breaker pause/resume checks use bar time rather than wall-clock time.

        Args:
            dt: Current bar timestamp (should be IST-aware).
        """
        self._backtest_time = dt

    def _now(self) -> datetime:
        """Return current time — backtest bar time if injected, else wall-clock."""
        if self._backtest_time is not None:
            return self._backtest_time
        return datetime.now(tz=IST)

    # ------------------------------------------------------------------
    # Trade outcome recording
    # ------------------------------------------------------------------

    def record_loss(self) -> None:
        """Record a losing trade.

        Increments the consecutive-loss counter.  If the counter reaches
        :attr:`max_consecutive_losses`, an automatic pause is activated.
        """
        with self._lock:
            self.consecutive_losses += 1
            count = self.consecutive_losses

        logger.debug(
            "Loss recorded | consecutive_losses={}/{}",
            count,
            self.max_consecutive_losses,
        )

        if count >= self.max_consecutive_losses:
            self._activate_pause()

    def record_win(self) -> None:
        """Record a winning trade, resetting the consecutive-loss counter."""
        with self._lock:
            self.consecutive_losses = 0

        logger.debug("Win recorded | consecutive_losses reset to 0")

    # ------------------------------------------------------------------
    # Halt management
    # ------------------------------------------------------------------

    def _activate_pause(self) -> None:
        """Activate an automatic time-limited pause.

        Sets :attr:`halted` to ``True`` and schedules an automatic
        resume after :attr:`pause_minutes`.
        """
        now = self._now()
        resume_at = now + timedelta(minutes=self.pause_minutes)

        with self._lock:
            self.halted = True
            self.halt_until = resume_at

        logger.warning(
            "Circuit breaker ACTIVATED after {} consecutive losses | trading paused until {} IST",
            self.max_consecutive_losses,
            resume_at.strftime("%Y-%m-%d %H:%M:%S"),
        )

    def is_halted(self) -> bool:
        """Check whether trading is currently halted.

        If the halt was time-limited and the pause period has elapsed,
        the breaker auto-resumes before returning ``False``.

        Returns:
            ``True`` if trading should be paused, ``False`` otherwise.
        """
        with self._lock:
            if not self.halted:
                return False

            # Permanent halt (kill switch) — halt_until is None
            if self.halt_until is None:
                return True

            # Time-limited pause — check expiry
            now = self._now()
            if now >= self.halt_until:
                self.halted = False
                self.halt_until = None
                self.consecutive_losses = 0
                logger.info(
                    "Circuit breaker auto-resumed after {} minute pause",
                    self.pause_minutes,
                )
                return False

            return True

    def is_allowed(self) -> bool:
        """Check if trading is allowed (not halted).

        Returns:
            True if trading is allowed, False if halted.
        """
        return not self.is_halted()

    def kill_switch(self, order_manager: OrderManagerProtocol | None = None) -> None:
        """Emergency halt — stops all trading indefinitely.

        When *order_manager* is provided its ``cancel_all_orders`` and
        ``exit_all_positions`` methods are called to liquidate the book.
        The halt persists until :meth:`resume` is called manually.

        Args:
            order_manager: Optional order manager instance used to
                cancel open orders and flatten positions.
        """
        with self._lock:
            self.halted = True
            self.halt_until = None  # permanent — manual resume only

        logger.critical("KILL SWITCH activated — all trading halted until manual resume")

        if order_manager is not None:
            try:
                order_manager.cancel_all_orders()
                logger.critical("All open orders cancelled via kill switch")
            except Exception:
                logger.exception("Failed to cancel orders during kill switch")

            try:
                order_manager.exit_all_positions()
                logger.critical("All positions exited via kill switch")
            except Exception:
                logger.exception("Failed to exit positions during kill switch")

    def resume(self) -> None:
        """Manually resume trading after a halt or kill switch."""
        with self._lock:
            self.halted = False
            self.halt_until = None
            self.consecutive_losses = 0

        logger.info("Circuit breaker manually resumed | consecutive_losses reset")
