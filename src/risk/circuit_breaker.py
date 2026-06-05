"""Circuit breaker and kill-switch module.

Halts trading automatically after a configurable number of consecutive
losses and provides a manual kill switch for emergency shutdowns.

Scheme 4 — Consecutive Loss Detox:
Before auto-resuming, the breaker stores the details of the losing
trades so the AI can analyze WHY the losses happened and adjust
behavior before resuming (e.g., switch bias, widen SL).

All timestamps use IST (``Asia/Kolkata``).
"""

from __future__ import annotations

import json
import os
import threading
from datetime import date, datetime, timedelta
from pathlib import Path
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
        per_strategy_daily_loss_cap_pct: float = 0.02,
        capital_paisa: int = 0,
        state_file: str | None = None,
    ) -> None:
        self._lock = threading.Lock()

        # Configuration
        self.max_consecutive_losses: int = max_consecutive_losses
        self.pause_minutes: int = pause_minutes
        self.per_strategy_daily_loss_cap_pct: float = per_strategy_daily_loss_cap_pct
        self.capital_paisa: int = capital_paisa
        self._state_file: Path | None = Path(state_file) if state_file else None

        # Mutable state — protected by ``_lock``
        self.consecutive_losses: int = 0
        self.halted: bool = False
        self.halt_until: datetime | None = None

        # Per-strategy P&L tracking (resets daily): {strategy_id: total_pnl_paisa}
        self._strategy_pnl_today: dict[str, int] = {}
        self._strategy_quarantined: set[str] = set()
        self._pnl_date: date | None = None

        logger.info(
            "CircuitBreaker initialised | max_consecutive_losses={} | pause_minutes={}",
            max_consecutive_losses,
            pause_minutes,
        )

        # Backtest mode: inject bar time to replace wall-clock in halt logic
        self._backtest_time: datetime | None = None
        
        # Scheme 4: Detox — store details of recent losses for AI analysis
        self._recent_losses: list[dict] = []
        self._detox_analysis: dict | None = None
        
        self._on_halt_callback = None

        # Restore state from disk (survives restarts)
        self._load_state()

    # ------------------------------------------------------------------
    # State persistence
    # ------------------------------------------------------------------

    def _save_state(self) -> None:
        """Persist mutable state to JSON so restarts preserve halt status."""
        if self._state_file is None:
            return
        try:
            self._state_file.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "date": datetime.now(IST).date().isoformat(),
                "consecutive_losses": self.consecutive_losses,
                "halted": self.halted,
                "halt_until": self.halt_until.isoformat() if self.halt_until else None,
            }
            tmp = self._state_file.with_suffix(".tmp")
            tmp.write_text(json.dumps(payload))
            os.replace(tmp, self._state_file)
        except Exception as exc:
            logger.warning("CircuitBreaker: failed to save state: {}", exc)

    def _load_state(self) -> None:
        """Restore previously saved state only if the file is from today (IST)."""
        if self._state_file is None or not self._state_file.exists():
            return
        try:
            payload = json.loads(self._state_file.read_text())
            saved_date = payload.get("date")
            today_str = datetime.now(IST).date().isoformat()
            if saved_date != today_str:
                logger.info(
                    "CircuitBreaker: state file is from {} (today is {}) — ignoring",
                    saved_date, today_str,
                )
                return
            self.consecutive_losses = int(payload.get("consecutive_losses", 0))
            self.halted = bool(payload.get("halted", False))
            halt_until_str = payload.get("halt_until")
            if halt_until_str:
                self.halt_until = datetime.fromisoformat(halt_until_str)
            logger.info(
                "CircuitBreaker: restored state — halted={} consecutive_losses={} halt_until={}",
                self.halted, self.consecutive_losses, self.halt_until,
            )
        except Exception as exc:
            logger.warning("CircuitBreaker: failed to load state: {}", exc)

    def set_on_halt_callback(self, callback) -> None:
        """Set a callback to execute when circuit breaker halts."""
        with self._lock:
            self._on_halt_callback = callback

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

    def record_loss(self, trade_details: dict | None = None) -> None:
        """Record a losing trade.

        Increments the consecutive-loss counter.  If the counter reaches
        :attr:`max_consecutive_losses`, an automatic pause is activated.
        
        Args:
            trade_details: Optional dict with trade info for detox analysis
                (e.g., direction, strategy, sl_distance_pct, holding_time_min).
        """
        with self._lock:
            self.consecutive_losses += 1
            count = self.consecutive_losses

            # Store loss details for detox analysis (keep last N)
            if trade_details:
                self._recent_losses.append(trade_details)
                # Keep only last max_consecutive_losses * 2 entries
                max_stored = self.max_consecutive_losses * 2
                if len(self._recent_losses) > max_stored:
                    self._recent_losses = self._recent_losses[-max_stored:]

        self._save_state()
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
            self._recent_losses.clear()

        self._save_state()
        logger.debug("Win recorded | consecutive_losses reset to 0")

    # ------------------------------------------------------------------
    # Per-strategy daily loss cap (Tier 2.6 — 2026-05-15)
    # ------------------------------------------------------------------

    def _roll_daily_pnl_if_new_day(self) -> None:
        """Reset per-strategy P&L tracking at the start of each trading day."""
        today = self._now().date()
        if self._pnl_date != today:
            if self._pnl_date is not None:
                logger.info(
                    "Per-strategy daily P&L roll-over | yesterday={} | reset {} strategies",
                    self._pnl_date,
                    len(self._strategy_pnl_today),
                )
            self._strategy_pnl_today.clear()
            self._strategy_quarantined.clear()
            self._pnl_date = today

    def record_strategy_pnl(self, strategy_id: str, pnl_paisa: int) -> None:
        """Accumulate P&L for a strategy and quarantine it if it breaches the daily cap.

        Called by PositionManager._close_position on every trade close. When
        a strategy's cumulative loss for the day exceeds
        ``capital_paisa * per_strategy_daily_loss_cap_pct``, the strategy is
        quarantined for the rest of the session — its signals will be
        rejected by OrderManager.

        Args:
            strategy_id: The strategy identifier (Signal.strategy_name).
            pnl_paisa: Realized P&L of the closed trade in paisa (negative for loss).
        """
        if not strategy_id:
            return
        with self._lock:
            self._roll_daily_pnl_if_new_day()
            prev = self._strategy_pnl_today.get(strategy_id, 0)
            new = prev + int(pnl_paisa)
            self._strategy_pnl_today[strategy_id] = new

            if strategy_id in self._strategy_quarantined:
                return  # Already quarantined; just keep the running sum

            if self.capital_paisa <= 0:
                return  # No cap configured

            cap = int(self.capital_paisa * self.per_strategy_daily_loss_cap_pct)
            if new <= -cap:
                self._strategy_quarantined.add(strategy_id)
                logger.critical(
                    "STRATEGY QUARANTINED | {} | day_pnl={} paisa (-{:.2f}%) >= cap -{}",
                    strategy_id,
                    new,
                    abs(new) / max(1, self.capital_paisa) * 100,
                    cap,
                )

    def is_strategy_quarantined(self, strategy_id: str) -> bool:
        """Check whether a strategy has been disabled for the day.

        Args:
            strategy_id: The strategy identifier.

        Returns:
            True if the strategy hit its per-strategy daily loss cap.
        """
        with self._lock:
            self._roll_daily_pnl_if_new_day()
            return strategy_id in self._strategy_quarantined

    # ------------------------------------------------------------------
    # Halt management
    # ------------------------------------------------------------------

    def _activate_pause(self) -> None:
        """Activate an automatic time-limited pause.

        Sets :attr:`halted` to ``True`` and schedules an automatic
        resume after :attr:`pause_minutes`.
        """
        with self._lock:
            now = self._now()
            resume_at = now + timedelta(minutes=self.pause_minutes)
            self.halted = True
            self.halt_until = resume_at

        self._save_state()
        logger.warning(
            "Circuit breaker ACTIVATED after {} consecutive losses | trading paused until {} IST",
            self.max_consecutive_losses,
            resume_at.strftime("%Y-%m-%d %H:%M:%S"),
        )
        
        if getattr(self, "_on_halt_callback", None):
            def _run_callback():
                try:
                    self._on_halt_callback()
                except Exception as e:
                    logger.error(f"Error in circuit breaker halt callback: {e}")
            threading.Thread(target=_run_callback, daemon=True).start()

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
                self._save_state()
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

        self._save_state()
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

        self._save_state()
        logger.info("Circuit breaker manually resumed | consecutive_losses reset")

    # ------------------------------------------------------------------
    # Scheme 4: Detox Analysis
    # ------------------------------------------------------------------

    def get_loss_summary_for_detox(self) -> dict:
        """Get a summary of recent losses for AI detox analysis.

        The AI will review this data to decide:
        - Were losses all same direction? (trend against us)
        - Were losses whipsaws? (SL too tight)
        - Were losses during lunch hours? (low volume chop)

        Returns:
            Dict with loss patterns for AI analysis.
        """
        with self._lock:
            losses = list(self._recent_losses)

        if not losses:
            return {"total_losses": 0, "patterns": {}}

        # Analyze patterns
        directions = [l.get("direction", "unknown") for l in losses]
        strategies = [l.get("strategy", "unknown") for l in losses]
        hours = [l.get("hour", -1) for l in losses]
        sl_hit_pcts = [l.get("sl_distance_pct", 0) for l in losses]
        holding_times = [l.get("holding_time_min", 0) for l in losses]

        # All same direction?
        all_same_dir = len(set(directions)) == 1 and len(directions) > 1

        # Whipsaw detection: small losses + quick exits
        avg_sl_pct = sum(sl_hit_pcts) / len(sl_hit_pcts) if sl_hit_pcts else 0
        avg_holding = sum(holding_times) / len(holding_times) if holding_times else 0
        likely_whipsaw = avg_sl_pct < 5.0 and avg_holding < 5.0

        # Lunch hour losses
        lunch_losses = sum(1 for h in hours if 11 <= h <= 13)
        lunch_heavy = lunch_losses > len(hours) / 2

        return {
            "total_losses": len(losses),
            "consecutive_losses": self.consecutive_losses,
            "directions": directions,
            "strategies": strategies,
            "all_same_direction": all_same_dir,
            "dominant_direction": max(set(directions), key=directions.count) if directions else "unknown",
            "avg_sl_distance_pct": avg_sl_pct,
            "avg_holding_time_min": avg_holding,
            "likely_whipsaw": likely_whipsaw,
            "lunch_hour_heavy": lunch_heavy,
            "raw_losses": losses,
        }

    def set_detox_analysis(self, analysis: dict) -> None:
        """Store AI detox analysis results.

        Called by the AI after reviewing the loss summary. The detox
        analysis contains recommendations like 'switch bias' or 'widen SL'.

        Args:
            analysis: AI-generated detox recommendations.
        """
        with self._lock:
            self._detox_analysis = analysis
        logger.info("CircuitBreaker detox analysis stored: {}", analysis)

    def get_detox_analysis(self) -> dict | None:
        """Retrieve the latest detox analysis (if any)."""
        with self._lock:
            return self._detox_analysis
