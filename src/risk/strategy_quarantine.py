"""Strategy quarantine system for auto-disabling underperforming strategies.

Automatically quarantines strategies that exhibit poor performance metrics
to prevent further losses. Strategies can be auto-released after a cooling-off
period with reduced position sizing.

All monetary values are in paisa (1 INR = 100 paisa) to avoid floating-point
precision issues. All timestamps use IST (Asia/Kolkata).
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import TYPE_CHECKING, Optional
from zoneinfo import ZoneInfo

from loguru import logger
from sqlalchemy import (
    DateTime,
    Float,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column

from src.persistence.database import get_session
from src.persistence.models import Base, _utcnow

if TYPE_CHECKING:
    pass

IST = ZoneInfo("Asia/Kolkata")


class QuarantineReason(Enum):
    """Reasons for quarantining a strategy.

    Attributes:
        LOW_WIN_RATE: Win rate below 30% after minimum 20 trades.
        CONSECUTIVE_LOSSES: 5 or more consecutive losing trades.
        HIGH_DRAWDOWN: Drawdown exceeds 10% from peak P&L.
        MANUAL: Manually quarantined by operator.
    """

    LOW_WIN_RATE = "low_win_rate"
    CONSECUTIVE_LOSSES = "consecutive_losses"
    HIGH_DRAWDOWN = "high_drawdown"
    MANUAL = "manual"


@dataclass(frozen=True, slots=True)
class StrategyPerformance:
    """Performance metrics for a strategy.

    Attributes:
        strategy_name: Name of the strategy.
        total_trades: Total number of completed trades.
        winning_trades: Number of profitable trades.
        losing_trades: Number of losing trades.
        consecutive_losses: Current streak of consecutive losses.
        max_drawdown: Maximum drawdown from peak (as decimal, e.g., 0.10 for 10%).
        win_rate: Win rate as percentage (0-100).
        profit_factor: Gross profit / gross loss ratio.
    """

    strategy_name: str
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    consecutive_losses: int = 0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0


class StrategyQuarantineRecord(Base):
    """Database record for quarantined strategies.

    Attributes:
        id: Auto-incrementing primary key.
        strategy_name: Name of the quarantined strategy.
        quarantined_at: Timestamp when quarantine was applied.
        reason: Quarantine reason enum value.
        reason_detail: Additional details about the quarantine.
        released_at: Timestamp when released (null if still quarantined).
        auto_release: Whether this was an auto-release after cooling period.
        released_by: User/system that released the strategy.
        performance_snapshot: JSON snapshot of performance at quarantine time.
    """

    __tablename__ = "strategy_quarantine"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    strategy_name: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    quarantined_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )
    reason: Mapped[str] = mapped_column(String(32), nullable=False)
    reason_detail: Mapped[str | None] = mapped_column(Text, nullable=True)
    released_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    auto_release: Mapped[bool] = mapped_column(default=False, nullable=False)
    released_by: Mapped[str | None] = mapped_column(String(64), nullable=True)
    performance_snapshot: Mapped[str | None] = mapped_column(Text, nullable=True)

    def __repr__(self) -> str:
        status = "released" if self.released_at else "active"
        return (
            f"<StrategyQuarantineRecord(strategy={self.strategy_name!r}, "
            f"reason={self.reason!r}, status={status!r})>"
        )


class StrategyPerformanceState(Base):
    """Persistent state tracking for strategy performance metrics.

    Attributes:
        id: Auto-incrementing primary key.
        strategy_name: Name of the strategy (unique).
        total_trades: Total completed trades.
        winning_trades: Number of winning trades.
        losing_trades: Number of losing trades.
        consecutive_losses: Current consecutive loss streak.
        peak_pnl: Highest P&L achieved in paisa.
        current_pnl: Current realized P&L in paisa.
        gross_profit: Total gross profit in paisa.
        gross_loss: Total gross loss in paisa (positive value).
        updated_at: Last update timestamp.
    """

    __tablename__ = "strategy_performance_state"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    strategy_name: Mapped[str] = mapped_column(
        String(128), nullable=False, unique=True, index=True
    )
    total_trades: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    winning_trades: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    losing_trades: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    consecutive_losses: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    peak_pnl: Mapped[int] = mapped_column(Integer, default=0, nullable=False)  # paisa
    current_pnl: Mapped[int] = mapped_column(Integer, default=0, nullable=False)  # paisa
    gross_profit: Mapped[int] = mapped_column(Integer, default=0, nullable=False)  # paisa
    gross_loss: Mapped[int] = mapped_column(Integer, default=0, nullable=False)  # paisa
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow, onupdate=_utcnow
    )

    def __repr__(self) -> str:
        return (
            f"<StrategyPerformanceState(strategy={self.strategy_name!r}, "
            f"trades={self.total_trades}, win_rate={self.winning_trades / max(self.total_trades, 1):.1%})>"
        )


class StrategyQuarantine:
    """Auto-disable underperforming strategies to prevent further losses.

    Monitors strategy performance metrics and automatically quarantines
    strategies that breach risk thresholds. Quarantined strategies are
    blocked from generating new signals until manually released or
    auto-released after a cooling-off period.

    Thread-safe: All public methods acquire an internal lock.

    Args:
        min_trades_for_win_rate: Minimum trades before win rate check applies.
        max_consecutive_losses: Consecutive losses that trigger quarantine.
        max_drawdown_pct: Maximum allowed drawdown percentage (e.g., 10.0 for 10%).
        auto_release_hours: Hours before auto-release after quarantine.
        reduced_size_pct: Position size reduction after release (e.g., 50.0 for 50%).
    """

    def __init__(
        self,
        min_trades_for_win_rate: int = 20,
        max_consecutive_losses: int = 5,
        max_drawdown_pct: float = 10.0,
        auto_release_hours: int = 24,
        reduced_size_pct: float = 50.0,
    ) -> None:
        self._lock = threading.Lock()

        # Configuration
        self.min_trades_for_win_rate: int = min_trades_for_win_rate
        self.max_consecutive_losses: int = max_consecutive_losses
        self.max_drawdown_pct: float = max_drawdown_pct
        self.auto_release_hours: int = auto_release_hours
        self.reduced_size_pct: float = reduced_size_pct

        # In-memory cache of quarantined strategies (strategy_name -> reason)
        self._quarantined: dict[str, str] = {}
        # Track reduced size status after release (strategy_name -> bool)
        self._reduced_size: dict[str, bool] = {}

        # Load existing quarantines from database
        self._load_quarantine_state()

        logger.info(
            "StrategyQuarantine initialized | min_trades={} | max_losses={} | "
            "max_drawdown={}% | auto_release={}h | reduced_size={}%",
            min_trades_for_win_rate,
            max_consecutive_losses,
            max_drawdown_pct,
            auto_release_hours,
            reduced_size_pct,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_quarantine_state(self) -> None:
        """Load active quarantines from database into memory cache."""
        try:
            with get_session() as session:
                active_quarantines = (
                    session.query(StrategyQuarantineRecord)
                    .filter(StrategyQuarantineRecord.released_at.is_(None))
                    .all()
                )
                for record in active_quarantines:
                    self._quarantined[record.strategy_name] = record.reason
        except Exception:
            logger.exception("Failed to load quarantine state from database")

    def _get_or_create_state(
        self, session, strategy_name: str
    ) -> StrategyPerformanceState:
        """Get existing performance state or create new one."""
        state = (
            session.query(StrategyPerformanceState)
            .filter_by(strategy_name=strategy_name)
            .first()
        )
        if state is None:
            state = StrategyPerformanceState(strategy_name=strategy_name)
            session.add(state)
            session.flush()
        return state

    def _calculate_performance(self, state: StrategyPerformanceState) -> StrategyPerformance:
        """Calculate performance metrics from state."""
        total = state.total_trades
        win_rate = (state.winning_trades / total * 100) if total > 0 else 0.0

        # Calculate max drawdown from peak
        if state.peak_pnl > 0:
            drawdown = (state.peak_pnl - state.current_pnl) / state.peak_pnl
        else:
            drawdown = 0.0

        # Calculate profit factor
        if state.gross_loss > 0:
            profit_factor = state.gross_profit / state.gross_loss
        elif state.gross_profit > 0:
            profit_factor = float("inf")
        else:
            profit_factor = 0.0

        return StrategyPerformance(
            strategy_name=state.strategy_name,
            total_trades=total,
            winning_trades=state.winning_trades,
            losing_trades=state.losing_trades,
            consecutive_losses=state.consecutive_losses,
            max_drawdown=drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
        )

    def _check_quarantine_rules(
        self, performance: StrategyPerformance
    ) -> Optional[QuarantineReason]:
        """Check if strategy should be quarantined based on performance.

        Returns:
            QuarantineReason if strategy should be quarantined, None otherwise.
        """
        # Check consecutive losses
        if performance.consecutive_losses >= self.max_consecutive_losses:
            return QuarantineReason.CONSECUTIVE_LOSSES

        # Check win rate (only after minimum trades)
        if (
            performance.total_trades >= self.min_trades_for_win_rate
            and performance.win_rate < 30.0
        ):
            return QuarantineReason.LOW_WIN_RATE

        # Check drawdown
        if performance.max_drawdown > (self.max_drawdown_pct / 100.0):
            return QuarantineReason.HIGH_DRAWDOWN

        return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_trade_result(self, strategy_name: str, pnl: int) -> None:
        """Record a trade result and update performance metrics.

        Call this method after each completed trade. Updates performance
        statistics and checks if strategy should be quarantined.

        Args:
            strategy_name: Name of the strategy that executed the trade.
            pnl: Realized P&L in paisa (positive for profit, negative for loss).
        """
        with self._lock:
            try:
                with get_session() as session:
                    state = self._get_or_create_state(session, strategy_name)

                    # Update trade counts
                    state.total_trades += 1
                    if pnl > 0:
                        state.winning_trades += 1
                        state.consecutive_losses = 0
                        state.gross_profit += pnl
                    elif pnl < 0:
                        state.losing_trades += 1
                        state.consecutive_losses += 1
                        state.gross_loss += abs(pnl)
                    else:
                        # Breakeven - reset consecutive losses
                        state.consecutive_losses = 0

                    # Update P&L tracking
                    state.current_pnl += pnl
                    if state.current_pnl > state.peak_pnl:
                        state.peak_pnl = state.current_pnl

                    # Calculate performance and check rules
                    performance = self._calculate_performance(state)
                    quarantine_reason = self._check_quarantine_rules(performance)

                    if quarantine_reason and strategy_name not in self._quarantined:
                        self._quarantine_strategy_internal(
                            session, strategy_name, quarantine_reason, performance
                        )

                    logger.debug(
                        "Trade recorded for {} | pnl={}p | total_trades={} | "
                        "consecutive_losses={} | current_pnl={}p",
                        strategy_name,
                        pnl,
                        state.total_trades,
                        state.consecutive_losses,
                        state.current_pnl,
                    )
            except Exception:
                logger.exception("Failed to record trade result for {}", strategy_name)

    def _quarantine_strategy_internal(
        self,
        session,
        strategy_name: str,
        reason: QuarantineReason,
        performance: StrategyPerformance,
    ) -> None:
        """Internal method to quarantine a strategy (must hold lock)."""
        # Create quarantine record
        snapshot = {
            "total_trades": performance.total_trades,
            "winning_trades": performance.winning_trades,
            "losing_trades": performance.losing_trades,
            "win_rate": performance.win_rate,
            "consecutive_losses": performance.consecutive_losses,
            "max_drawdown": performance.max_drawdown,
            "profit_factor": performance.profit_factor,
        }

        record = StrategyQuarantineRecord(
            strategy_name=strategy_name,
            reason=reason.value,
            reason_detail=f"Auto-quarantined: {reason.value}",
            performance_snapshot=json.dumps(snapshot),
        )
        session.add(record)

        # Update in-memory cache
        self._quarantined[strategy_name] = reason.value

        logger.warning(
            "Strategy QUARANTINED | strategy={} | reason={} | "
            "trades={} | win_rate={:.1f}% | consecutive_losses={} | drawdown={:.1f}%",
            strategy_name,
            reason.value,
            performance.total_trades,
            performance.win_rate,
            performance.consecutive_losses,
            performance.max_drawdown * 100,
        )

    def quarantine_strategy(
        self, strategy_name: str, reason: QuarantineReason
    ) -> None:
        """Manually quarantine a strategy.

        Args:
            strategy_name: Name of the strategy to quarantine.
            reason: Reason for quarantine (use QuarantineReason.MANUAL for manual).
        """
        with self._lock:
            if strategy_name in self._quarantined:
                logger.info(
                    "Strategy {} is already quarantined (reason: {})",
                    strategy_name,
                    self._quarantined[strategy_name],
                )
                return

            try:
                with get_session() as session:
                    # Get current performance or create empty
                    state = self._get_or_create_state(session, strategy_name)
                    performance = self._calculate_performance(state)

                    self._quarantine_strategy_internal(
                        session, strategy_name, reason, performance
                    )
            except Exception:
                logger.exception("Failed to manually quarantine strategy {}", strategy_name)

    def check_strategy(self, strategy_name: str) -> tuple[bool, Optional[str]]:
        """Check if a strategy is allowed to trade.

        Args:
            strategy_name: Name of the strategy to check.

        Returns:
            Tuple of (allowed: bool, reason: Optional[str]).
            If allowed is False, reason contains the quarantine reason.
        """
        with self._lock:
            # First check auto-release
            self._auto_release_check_internal()

            if strategy_name in self._quarantined:
                return False, self._quarantined[strategy_name]
            return True, None

    def release_strategy(self, strategy_name: str, released_by: str = "manual") -> bool:
        """Manually release a strategy from quarantine.

        Args:
            strategy_name: Name of the strategy to release.
            released_by: Identifier of who/what released the strategy.

        Returns:
            True if strategy was released, False if not found in quarantine.
        """
        with self._lock:
            if strategy_name not in self._quarantined:
                logger.debug(
                    "Cannot release {}: not currently quarantined", strategy_name
                )
                return False

            try:
                with get_session() as session:
                    record = (
                        session.query(StrategyQuarantineRecord)
                        .filter_by(strategy_name=strategy_name, released_at=None)
                        .first()
                    )
                    if record:
                        record.released_at = datetime.now(tz=IST)
                        record.auto_release = False
                        record.released_by = released_by

                    del self._quarantined[strategy_name]
                    self._reduced_size[strategy_name] = True

                logger.info(
                    "Strategy RELEASED from quarantine | strategy={} | by={}",
                    strategy_name,
                    released_by,
                )
                return True
            except Exception:
                logger.exception("Failed to release strategy {}", strategy_name)
                return False

    def _auto_release_check_internal(self) -> None:
        """Internal auto-release check (must hold lock)."""
        try:
            with get_session() as session:
                cutoff = datetime.now(tz=IST) - timedelta(hours=self.auto_release_hours)

                eligible = (
                    session.query(StrategyQuarantineRecord)
                    .filter(StrategyQuarantineRecord.released_at.is_(None))
                    .filter(StrategyQuarantineRecord.quarantined_at <= cutoff)
                    .all()
                )

                for record in eligible:
                    record.released_at = datetime.now(tz=IST)
                    record.auto_release = True
                    record.released_by = "system_auto_release"

                    if record.strategy_name in self._quarantined:
                        del self._quarantined[record.strategy_name]
                    self._reduced_size[record.strategy_name] = True

                    logger.info(
                        "Strategy AUTO-RELEASED after {}h | strategy={} | "
                        "original_reason={}",
                        self.auto_release_hours,
                        record.strategy_name,
                        record.reason,
                    )
        except Exception:
            logger.exception("Auto-release check failed")

    def auto_release_check(self) -> list[str]:
        """Check and auto-release strategies after cooling-off period.

        Returns:
            List of strategy names that were auto-released.
        """
        with self._lock:
            released_before = set(self._quarantined.keys())
            self._auto_release_check_internal()
            released_after = set(self._quarantined.keys())
            return list(released_before - released_after)

    def get_quarantined_strategies(self) -> list[str]:
        """Get list of currently quarantined strategy names.

        Returns:
            List of quarantined strategy names.
        """
        with self._lock:
            self._auto_release_check_internal()
            return list(self._quarantined.keys())

    def get_performance(self, strategy_name: str) -> Optional[StrategyPerformance]:
        """Get performance metrics for a strategy.

        Args:
            strategy_name: Name of the strategy.

        Returns:
            StrategyPerformance if found, None otherwise.
        """
        try:
            with get_session() as session:
                state = (
                    session.query(StrategyPerformanceState)
                    .filter_by(strategy_name=strategy_name)
                    .first()
                )
                if state:
                    return self._calculate_performance(state)
                return None
        except Exception:
            logger.exception("Failed to get performance for {}", strategy_name)
            return None

    def is_reduced_size(self, strategy_name: str) -> bool:
        """Check if strategy should trade at reduced position size.

        Strategies that have been released from quarantine (either manually
        or auto-released) should trade at reduced size for a period.

        Args:
            strategy_name: Name of the strategy.

        Returns:
            True if position size should be reduced.
        """
        with self._lock:
            return self._reduced_size.get(strategy_name, False)

    def clear_reduced_size_flag(self, strategy_name: str) -> None:
        """Clear the reduced size flag for a strategy.

        Call this after the strategy has traded at reduced size for
        sufficient time and performance has stabilized.

        Args:
            strategy_name: Name of the strategy.
        """
        with self._lock:
            if strategy_name in self._reduced_size:
                del self._reduced_size[strategy_name]
                logger.info(
                    "Reduced size flag cleared for strategy {}", strategy_name
                )

    def get_position_size_multiplier(self, strategy_name: str) -> float:
        """Get the position size multiplier for a strategy.

        Args:
            strategy_name: Name of the strategy.

        Returns:
            0.0 if quarantined (no trading allowed),
            reduced_size_pct/100 if recently released,
            1.0 otherwise (normal sizing).
        """
        with self._lock:
            if strategy_name in self._quarantined:
                return 0.0
            if strategy_name in self._reduced_size:
                return self.reduced_size_pct / 100.0
            return 1.0

    def reset_strategy_performance(self, strategy_name: str) -> bool:
        """Reset all performance metrics for a strategy.

        Use with caution - this clears all historical performance data.

        Args:
            strategy_name: Name of the strategy to reset.

        Returns:
            True if reset was successful.
        """
        with self._lock:
            try:
                with get_session() as session:
                    state = (
                        session.query(StrategyPerformanceState)
                        .filter_by(strategy_name=strategy_name)
                        .first()
                    )
                    if state:
                        state.total_trades = 0
                        state.winning_trades = 0
                        state.losing_trades = 0
                        state.consecutive_losses = 0
                        state.peak_pnl = 0
                        state.current_pnl = 0
                        state.gross_profit = 0
                        state.gross_loss = 0

                    # Also clear from memory caches
                    if strategy_name in self._reduced_size:
                        del self._reduced_size[strategy_name]

                logger.warning(
                    "Strategy performance RESET | strategy={}", strategy_name
                )
                return True
            except Exception:
                logger.exception("Failed to reset performance for {}", strategy_name)
                return False

    def get_quarantine_history(
        self, strategy_name: Optional[str] = None, limit: int = 100
    ) -> list[dict]:
        """Get quarantine history for analysis.

        Args:
            strategy_name: Optional filter by strategy name.
            limit: Maximum number of records to return.

        Returns:
            List of quarantine record dictionaries.
        """
        try:
            with get_session() as session:
                query = session.query(StrategyQuarantineRecord)
                if strategy_name:
                    query = query.filter_by(strategy_name=strategy_name)
                records = query.order_by(
                    StrategyQuarantineRecord.quarantined_at.desc()
                ).limit(limit).all()

                return [
                    {
                        "strategy_name": r.strategy_name,
                        "quarantined_at": r.quarantined_at,
                        "reason": r.reason,
                        "reason_detail": r.reason_detail,
                        "released_at": r.released_at,
                        "auto_release": r.auto_release,
                        "released_by": r.released_by,
                        "performance_snapshot": (
                            json.loads(r.performance_snapshot)
                            if r.performance_snapshot
                            else None
                        ),
                    }
                    for r in records
                ]
        except Exception:
            logger.exception("Failed to get quarantine history")
            return []
