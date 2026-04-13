"""Persistence layer for the learning system.

This module provides database operations for storing and retrieving
lessons learned and strategy performance metrics.
"""

from __future__ import annotations

import json
from datetime import date, datetime
from typing import TYPE_CHECKING, Any

from loguru import logger
from sqlalchemy import and_

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

    from src.learning.trade_analyzer import LessonLearned, StrategyPerformance

from src.persistence.models import LessonLearnedRecord, StrategyPerformanceRecord


class LearningPersistence:
    """Handles database operations for the learning system.

    This class provides methods to save and load lessons learned
    and strategy performance metrics from the database.

    Attributes:
        _db: SQLAlchemy session for database operations
    """

    def __init__(self, db_session: Session):
        """Initialize the persistence layer.

        Args:
            db_session: SQLAlchemy session for database operations
        """
        self._db = db_session

    def save_lesson(self, lesson: LessonLearned) -> None:
        """Save a lesson to the database.

        Args:
            lesson: LessonLearned object to save
        """
        try:
            # Check if lesson already exists
            existing = (
                self._db.query(LessonLearnedRecord)
                .filter(LessonLearnedRecord.lesson_id == lesson.id)
                .first()
            )

            if existing:
                # Update existing lesson
                existing.applied = 1 if lesson.applied else 0
                existing.applied_at = datetime.now() if lesson.applied else None
                existing.confidence = lesson.confidence
            else:
                # Create new lesson record
                record = LessonLearnedRecord(
                    lesson_id=lesson.id,
                    timestamp=lesson.timestamp,
                    strategy_name=lesson.strategy_name,
                    category=lesson.category,
                    description=lesson.description,
                    trigger_pattern_json=json.dumps(lesson.trigger_pattern),
                    impact_paisa=lesson.impact,
                    confidence=lesson.confidence,
                    applied=1 if lesson.applied else 0,
                )
                self._db.add(record)

            self._db.commit()
            logger.debug(f"Saved lesson {lesson.id} for {lesson.strategy_name}")

        except Exception as e:
            self._db.rollback()
            logger.error(f"Failed to save lesson: {e}")
            raise

    def load_lessons(
        self,
        strategy: str | None = None,
        category: str | None = None,
        since: datetime | None = None,
    ) -> list[LessonLearned]:
        """Load lessons from the database.

        Args:
            strategy: Filter by strategy name (optional)
            category: Filter by category (optional)
            since: Load lessons since this date (optional)

        Returns:
            List of LessonLearned objects
        """
        from src.learning.trade_analyzer import LessonLearned

        try:
            query = self._db.query(LessonLearnedRecord)

            if strategy:
                query = query.filter(LessonLearnedRecord.strategy_name == strategy)

            if category:
                query = query.filter(LessonLearnedRecord.category == category)

            if since:
                query = query.filter(LessonLearnedRecord.timestamp >= since)

            records = query.order_by(LessonLearnedRecord.timestamp.desc()).all()

            lessons = []
            for record in records:
                lesson = LessonLearned(
                    id=record.lesson_id,
                    timestamp=record.timestamp,
                    category=record.category,
                    description=record.description,
                    trigger_pattern=json.loads(record.trigger_pattern_json or "{}"),
                    impact=record.impact_paisa,
                    confidence=record.confidence,
                    applied=bool(record.applied),
                    strategy_name=record.strategy_name,
                )
                lessons.append(lesson)

            logger.debug(f"Loaded {len(lessons)} lessons from database")
            return lessons

        except Exception as e:
            logger.error(f"Failed to load lessons: {e}")
            return []

    def save_strategy_performance(
        self, performance: StrategyPerformance, perf_date: date | None = None
    ) -> None:
        """Save strategy performance to the database.

        Args:
            performance: StrategyPerformance object to save
            perf_date: Date for the performance record (defaults to today)
        """
        if perf_date is None:
            perf_date = date.today()

        try:
            # Check for existing record
            existing = (
                self._db.query(StrategyPerformanceRecord)
                .filter(
                    and_(
                        StrategyPerformanceRecord.strategy_name == performance.strategy_name,
                        StrategyPerformanceRecord.date == perf_date,
                    )
                )
                .first()
            )

            if existing:
                # Update existing record
                existing.total_trades = performance.total_trades
                existing.winning_trades = performance.winning_trades
                existing.losing_trades = performance.losing_trades
                existing.total_pnl_paisa = performance.total_pnl
                existing.win_rate = performance.win_rate
                existing.profit_factor = performance.profit_factor
                existing.avg_win_pct = performance.avg_win
                existing.avg_loss_pct = performance.avg_loss
                existing.max_consecutive_losses = performance.max_consecutive_losses
                existing.hourly_performance_json = json.dumps(performance.hourly_performance)
            else:
                # Create new record
                record = StrategyPerformanceRecord(
                    strategy_name=performance.strategy_name,
                    date=perf_date,
                    total_trades=performance.total_trades,
                    winning_trades=performance.winning_trades,
                    losing_trades=performance.losing_trades,
                    total_pnl_paisa=performance.total_pnl,
                    win_rate=performance.win_rate,
                    profit_factor=performance.profit_factor,
                    avg_win_pct=performance.avg_win,
                    avg_loss_pct=performance.avg_loss,
                    max_consecutive_losses=performance.max_consecutive_losses,
                    hourly_performance_json=json.dumps(performance.hourly_performance),
                )
                self._db.add(record)

            self._db.commit()
            logger.debug(
                f"Saved performance for {performance.strategy_name} on {perf_date}"
            )

        except Exception as e:
            self._db.rollback()
            logger.error(f"Failed to save strategy performance: {e}")
            raise

    def load_strategy_performance(
        self, strategy_name: str, days: int = 30
    ) -> list[dict[str, Any]]:
        """Load historical performance for a strategy.

        Args:
            strategy_name: Name of the strategy
            days: Number of days of history to load

        Returns:
            List of performance dictionaries
        """
        try:
            cutoff_date = date.today() - __import__("datetime").timedelta(days=days)

            records = (
                self._db.query(StrategyPerformanceRecord)
                .filter(
                    and_(
                        StrategyPerformanceRecord.strategy_name == strategy_name,
                        StrategyPerformanceRecord.date >= cutoff_date,
                    )
                )
                .order_by(StrategyPerformanceRecord.date.desc())
                .all()
            )

            results = []
            for record in records:
                results.append({
                    "date": record.date.isoformat(),
                    "total_trades": record.total_trades,
                    "winning_trades": record.winning_trades,
                    "losing_trades": record.losing_trades,
                    "total_pnl_paisa": record.total_pnl_paisa,
                    "win_rate": record.win_rate,
                    "profit_factor": record.profit_factor,
                    "avg_win_pct": record.avg_win_pct,
                    "avg_loss_pct": record.avg_loss_pct,
                    "max_consecutive_losses": record.max_consecutive_losses,
                    "hourly_performance": json.loads(record.hourly_performance_json or "{}"),
                })

            return results

        except Exception as e:
            logger.error(f"Failed to load strategy performance: {e}")
            return []

    def get_performance_trend(
        self, strategy_name: str, days: int = 14
    ) -> dict[str, Any]:
        """Analyze performance trend over time.

        Args:
            strategy_name: Name of the strategy
            days: Number of days to analyze

        Returns:
            Dictionary with trend analysis
        """
        history = self.load_strategy_performance(strategy_name, days)

        if not history:
            return {"status": "no_data", "message": "No historical data available"}

        # Calculate trend
        recent_pnl = sum(h["total_pnl_paisa"] for h in history[:7])  # Last 7 days
        older_pnl = sum(h["total_pnl_paisa"] for h in history[7:14])  # Previous 7 days

        recent_trades = sum(h["total_trades"] for h in history[:7])
        older_trades = sum(h["total_trades"] for h in history[7:14])

        recent_wins = sum(h["winning_trades"] for h in history[:7])

        trend = {
            "status": "analyzing",
            "recent_pnl_rupees": recent_pnl / 100,
            "older_pnl_rupees": older_pnl / 100,
            "recent_trades": recent_trades,
            "older_trades": older_trades,
            "recent_win_rate": recent_wins / recent_trades if recent_trades > 0 else 0,
        }

        # Determine trend direction
        if recent_trades >= 5:
            if recent_pnl > older_pnl * 1.2:
                trend["status"] = "improving"
                trend["message"] = "Strategy performance is improving"
            elif recent_pnl < older_pnl * 0.8:
                trend["status"] = "declining"
                trend["message"] = "Strategy performance is declining"
            else:
                trend["status"] = "stable"
                trend["message"] = "Strategy performance is stable"
        else:
            trend["message"] = "Insufficient recent data for trend analysis"

        return trend

    def mark_lesson_applied(self, lesson_id: str) -> bool:
        """Mark a lesson as applied.

        Args:
            lesson_id: ID of the lesson to mark

        Returns:
            True if successful, False otherwise
        """
        try:
            record = (
                self._db.query(LessonLearnedRecord)
                .filter(LessonLearnedRecord.lesson_id == lesson_id)
                .first()
            )

            if record:
                record.applied = 1
                record.applied_at = datetime.now()
                self._db.commit()
                logger.debug(f"Marked lesson {lesson_id} as applied")
                return True

            return False

        except Exception as e:
            self._db.rollback()
            logger.error(f"Failed to mark lesson as applied: {e}")
            return False
