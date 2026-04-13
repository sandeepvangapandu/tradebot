"""Integration module for the learning system.

This module provides integration points between the learning system
and other components of the trading bot (position manager, order manager,
notifications, etc.).

Example:
    from src.learning.integration import LearningIntegration

    # Initialize integration
    learning_integration = LearningIntegration(db_session, config)

    # Connect to position manager
    position_manager.set_on_position_close_callback(
        learning_integration.on_position_closed
    )

    # Use for position sizing in order manager
    adjusted_size = learning_integration.get_adjusted_position_size(
        strategy_name, base_size
    )
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any

from loguru import logger

from src.learning.persistence import LearningPersistence
from src.learning.trade_analyzer import LearningEngine, LessonLearned

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

    from src.execution.position_manager import ManagedPosition
    from src.persistence.models import TradeRecord


class LearningIntegration:
    """Integration layer for the learning system.

    This class provides a simplified interface for other components
    to interact with the learning system without knowing its internals.

    Attributes:
        _engine: LearningEngine instance
        _persistence: LearningPersistence instance
        _config: Configuration dictionary
        _enabled: Whether learning system is enabled
        _lock: Thread lock for thread-safe operations
    """

    def __init__(
        self,
        db_session: Session,
        config: dict[str, Any] | None = None,
        enabled: bool = True,
        backtest_mode: bool = False,
    ):
        """Initialize the learning integration.

        Args:
            db_session: SQLAlchemy session for database operations
            config: Configuration dictionary (uses defaults if None)
            enabled: Whether the learning system is enabled
            backtest_mode: If True, skip lesson persistence to protect the live
                model from potentially optimistic backtest data.
        """
        self._engine = LearningEngine(db_session, config)
        self._persistence = LearningPersistence(db_session)
        self._config = config or {}
        self._enabled = enabled
        self._backtest_mode = backtest_mode
        self._lock = threading.RLock()

        if enabled:
            mode = "backtest" if backtest_mode else "live"
            logger.info(f"Learning system integration initialized (mode={mode})")
        else:
            logger.info("Learning system integration initialized (disabled)")

    def on_position_closed(self, position: ManagedPosition) -> None:
        """Callback for when a position is closed.

        This method should be registered as a callback with the position manager.
        It processes the closed position for learning and persists the trade data.

        Args:
            position: The closed managed position
        """
        if not self._enabled:
            return

        try:
            with self._lock:
                # Create trade record from position
                trade_record = self._create_trade_record(position)

                # Process trade for learning
                insights = self._engine.process_trade(trade_record)

                # Persist lesson if any issues were identified.
                # In backtest_mode, skip lesson persistence to avoid contaminating
                # the live learning model with potentially optimistic backtest data.
                if not insights["was_win"] and not self._backtest_mode:
                    lessons = self._engine._analyzer.get_lessons(
                        strategy=position.strategy_id
                    )
                    for lesson in lessons[-3:]:  # Save last 3 lessons
                        if not lesson.applied:
                            self._persistence.save_lesson(lesson)

                # Persist strategy performance
                perf = self._engine._analyzer.get_strategy_performance(
                    position.strategy_id
                )
                if perf:
                    self._persistence.save_strategy_performance(perf)

                logger.debug(
                    f"Processed closed position {position.position_id} for learning: "
                    f"P&L={insights['pnl_rupees']:.2f}"
                )

        except Exception as e:
            logger.error(f"Error processing closed position for learning: {e}")

    def _create_trade_record(self, position: ManagedPosition) -> TradeRecord:
        """Create a TradeRecord from a ManagedPosition.

        Args:
            position: Closed managed position

        Returns:
            TradeRecord for database
        """
        from src.persistence.models import TradeRecord

        # Calculate holding duration
        if position.entry_time and position.exit_time:
            holding_duration = int(
                (position.exit_time - position.entry_time).total_seconds()
            )
        else:
            holding_duration = 0

        # Calculate total P&L including partial exits
        total_pnl = position.total_realized_pnl

        return TradeRecord(
            strategy=position.strategy_id,
            instrument_key=position.instrument_key,
            side="BUY" if position.side.value == "BUY" else "SELL",
            entry_price=position.entry_price,
            exit_price=position.exit_price or position.entry_price,
            quantity=position.original_quantity or position.quantity,
            realized_pnl=total_pnl,
            fees=0,  # Fees would need to be calculated from partial exits
            entry_time=position.entry_time,
            exit_time=position.exit_time or position.entry_time,
            holding_duration_seconds=holding_duration,
        )

    def get_adjusted_position_size(
        self, strategy_name: str, base_size: int
    ) -> int:
        """Get position size adjusted based on recent performance.

        Args:
            strategy_name: Name of the strategy
            base_size: Base position size

        Returns:
            Adjusted position size
        """
        if not self._enabled:
            return base_size

        if not self._config.get("learning_auto_adjust_size", True):
            return base_size

        with self._lock:
            return self._engine.adjust_position_size(strategy_name, base_size)

    def get_recommendations(self) -> list[dict[str, Any]]:
        """Get current recommendations for strategy improvements.

        Returns:
            List of recommendation dictionaries
        """
        if not self._enabled:
            return []

        with self._lock:
            return self._engine.generate_recommendations()

    def get_lessons_report(self, max_lessons: int = 10) -> str:
        """Get a human-readable lessons learned report.

        Args:
            max_lessons: Maximum number of lessons to include

        Returns:
            Formatted report string
        """
        if not self._enabled:
            return "Learning system is disabled."

        with self._lock:
            return self._engine.get_lessons_report(max_lessons)

    def get_strategy_stats(self, strategy_name: str) -> dict[str, Any] | None:
        """Get statistics for a specific strategy.

        Args:
            strategy_name: Name of the strategy

        Returns:
            Dictionary with strategy statistics or None
        """
        if not self._enabled:
            return None

        with self._lock:
            return self._engine.get_strategy_stats(strategy_name)

    def should_trade_strategy(self, strategy_name: str) -> tuple[bool, str]:
        """Check if a strategy should be allowed to trade.

        Args:
            strategy_name: Name of the strategy

        Returns:
            Tuple of (should_trade, reason)
        """
        if not self._enabled:
            return True, "Learning system disabled"

        with self._lock:
            perf = self._engine._analyzer.get_strategy_performance(strategy_name)

            if perf is None:
                return True, "No performance data yet"

            min_trades = self._engine._config.get("min_trades_for_analysis", 5)

            if perf.total_trades < min_trades:
                return True, f"Insufficient data ({perf.total_trades}/{min_trades} trades)"

            # Check for excessive consecutive losses
            max_cl = self._engine._config.get("max_consecutive_losses_threshold", 4)
            if perf.current_consecutive_losses >= max_cl:
                return (
                    False,
                    f"Strategy quarantined: {perf.current_consecutive_losses} consecutive losses"
                )

            # Check win rate
            low_threshold = self._engine._config.get("low_win_rate_threshold", 0.35)
            if perf.win_rate < low_threshold and perf.total_trades >= min_trades * 2:
                return (
                    False,
                    f"Strategy underperforming: {perf.win_rate:.1%} win rate"
                )

            return True, f"Win rate: {perf.win_rate:.1%}, PF: {perf.profit_factor:.2f}"

    def export_lessons(self, filepath: str | None = None) -> str:
        """Export all lessons to a JSON file.

        Args:
            filepath: Path to output file (uses config default if None)

        Returns:
            Path to exported file
        """
        if not self._enabled:
            return ""

        if filepath is None:
            filepath = self._config.get(
                "learning_export_lessons_path", "data/lessons.json"
            )

        with self._lock:
            self._engine.export_lessons(filepath)

        return filepath

    def import_lessons(self, filepath: str) -> int:
        """Import lessons from a JSON file.

        Args:
            filepath: Path to input JSON file

        Returns:
            Number of lessons imported
        """
        if not self._enabled:
            return 0

        with self._lock:
            count = self._engine.import_lessons(filepath)

            # Persist imported lessons to database
            for lesson in self._engine._analyzer.get_lessons():
                self._persistence.save_lesson(lesson)

            return count

    def get_hourly_performance(self, strategy_name: str) -> dict[int, dict[str, Any]]:
        """Get hourly performance breakdown for a strategy.

        Args:
            strategy_name: Name of the strategy

        Returns:
            Dictionary mapping hour to performance stats
        """
        if not self._enabled:
            return {}

        perf = self._engine._analyzer.get_strategy_performance(strategy_name)
        if perf:
            return perf.hourly_performance
        return {}

    def get_best_trading_hours(self, strategy_name: str) -> list[int]:
        """Get the best trading hours for a strategy.

        Args:
            strategy_name: Name of the strategy

        Returns:
            List of hours (0-23) sorted by performance
        """
        hourly = self.get_hourly_performance(strategy_name)

        if not hourly:
            return []

        # Calculate win rate for each hour
        hour_scores = []
        for hour, stats in hourly.items():
            count = stats.get("count", 0)
            if count >= 3:  # Minimum sample size
                wins = stats.get("wins", 0)
                pnl = stats.get("pnl", 0)
                win_rate = wins / count if count > 0 else 0
                # Score combines win rate and P&L
                score = win_rate * 100 + (pnl / 1000)  # Normalize P&L impact
                hour_scores.append((hour, score))

        # Sort by score descending
        hour_scores.sort(key=lambda x: x[1], reverse=True)

        return [h[0] for h in hour_scores]

    def generate_daily_report(self) -> str:
        """Generate a daily learning report.

        Returns:
            Formatted report string
        """
        report = self.get_lessons_report(max_lessons=20)

        # Add strategy-specific recommendations
        recommendations = self.get_recommendations()

        if recommendations:
            report += "\n\nPRIORITY ACTIONS:\n"
            report += "-" * 70 + "\n"

            for rec in recommendations:
                if rec["priority"] == "high":
                    report += f"\n! {rec['strategy']}: {rec['action']}\n"

        return report


def create_learning_integration_from_config(
    db_session: Session,
) -> LearningIntegration | None:
    """Create a LearningIntegration instance from application config.

    Args:
        db_session: SQLAlchemy session

    Returns:
        LearningIntegration instance or None if disabled
    """
    from config.settings import get_settings

    settings = get_settings()

    if not settings.learning_enabled:
        logger.info("Learning system is disabled in configuration")
        return None

    config = {
        "min_trades_for_analysis": settings.learning_min_trades_for_analysis,
        "low_win_rate_threshold": settings.learning_low_win_rate_threshold,
        "high_win_rate_threshold": settings.learning_high_win_rate_threshold,
        "min_profit_factor": settings.learning_min_profit_factor,
        "target_profit_factor": settings.learning_target_profit_factor,
        "max_consecutive_losses_threshold": settings.learning_max_consecutive_losses_threshold,
        "position_size_reduction_factor": settings.learning_position_size_reduction_factor,
        "position_size_increase_factor": settings.learning_position_size_increase_factor,
        "learning_auto_adjust_size": settings.learning_auto_adjust_size,
        "learning_export_lessons_path": settings.learning_export_lessons_path,
    }

    return LearningIntegration(
        db_session=db_session,
        config=config,
        enabled=True,
    )
