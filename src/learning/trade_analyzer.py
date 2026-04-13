"""Self-learning system for trade analysis and improvement.

This module provides the TradeAnalyzer and LearningEngine classes that analyze
trade outcomes, identify patterns in losing trades, and generate actionable
recommendations for strategy improvement.

Key Features:
    - Automatic pattern recognition in trade outcomes
    - Strategy performance tracking with detailed metrics
    - Dynamic position sizing based on recent performance
    - Lessons learned database for continuous improvement
    - Human-readable reports with recommendations

Example:
    from src.learning import LearningEngine

    engine = LearningEngine(db_session)

    # Process a completed trade
    engine.process_trade(trade_record)

    # Get recommendations
    recommendations = engine.generate_recommendations()

    # Adjust position size based on performance
    adjusted_size = engine.adjust_position_size("ema_crossover", base_size=100)
"""

from __future__ import annotations

import json
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any

from loguru import logger

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

    from src.persistence.models import TradeRecord


@dataclass
class LessonLearned:
    """A lesson extracted from trade analysis.

    Attributes:
        id: Unique identifier for the lesson
        timestamp: When the lesson was learned
        category: Type of lesson (strategy, timing, market_condition, risk)
        description: Human-readable description of the lesson
        trigger_pattern: Dictionary of conditions that led to this lesson
        impact: P&L impact in paisa
        confidence: Confidence level 0-1 based on sample size
        applied: Whether this lesson has been applied to strategy
        strategy_name: Name of the strategy this lesson applies to
    """

    id: str
    timestamp: datetime
    category: str  # "strategy", "timing", "market_condition", "risk"
    description: str
    trigger_pattern: dict[str, Any]
    impact: int  # paisa
    confidence: float  # 0-1, based on sample size
    applied: bool = False
    strategy_name: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert lesson to dictionary for serialization."""
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "category": self.category,
            "description": self.description,
            "trigger_pattern": self.trigger_pattern,
            "impact": self.impact,
            "confidence": self.confidence,
            "applied": self.applied,
            "strategy_name": self.strategy_name,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LessonLearned:
        """Create lesson from dictionary."""
        return cls(
            id=data["id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            category=data["category"],
            description=data["description"],
            trigger_pattern=data["trigger_pattern"],
            impact=data["impact"],
            confidence=data["confidence"],
            applied=data.get("applied", False),
            strategy_name=data.get("strategy_name", ""),
        )


@dataclass
class StrategyPerformance:
    """Performance metrics for a strategy.

    Attributes:
        strategy_name: Name of the strategy
        total_trades: Total number of trades
        winning_trades: Number of winning trades
        losing_trades: Number of losing trades
        total_pnl: Total P&L in paisa
        avg_win: Average win percentage
        avg_loss: Average loss percentage
        win_rate: Win rate as decimal (0-1)
        profit_factor: Profit factor (gross profit / gross loss)
        max_consecutive_losses: Maximum consecutive losses seen
        avg_holding_time: Average holding time in minutes
        best_hour: Best performing hour of entry (-1 if unknown)
        worst_hour: Worst performing hour of entry (-1 if unknown)
        lessons: List of lessons learned for this strategy
        hourly_performance: Dict mapping hour to (wins, losses, pnl)
        recent_trades: List of recent trade outcomes for pattern detection
    """

    strategy_name: str
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: int = 0  # paisa
    avg_win: float = 0.0
    avg_loss: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    max_consecutive_losses: int = 0
    current_consecutive_losses: int = 0
    avg_holding_time: float = 0.0  # minutes
    best_hour: int = -1
    worst_hour: int = -1
    lessons: list[LessonLearned] = field(default_factory=list)
    hourly_performance: dict[int, dict[str, Any]] = field(default_factory=dict)
    recent_trades: list[dict[str, Any]] = field(default_factory=list)

    def update_hourly_performance(self, hour: int, was_win: bool, pnl: int) -> None:
        """Update performance statistics for a specific hour."""
        if hour not in self.hourly_performance:
            self.hourly_performance[hour] = {"wins": 0, "losses": 0, "pnl": 0, "count": 0}

        self.hourly_performance[hour]["count"] += 1
        self.hourly_performance[hour]["pnl"] += pnl
        if was_win:
            self.hourly_performance[hour]["wins"] += 1
        else:
            self.hourly_performance[hour]["losses"] += 1

        # Update best/worst hour
        self._update_best_worst_hour()

    def _update_best_worst_hour(self) -> None:
        """Recalculate best and worst hours based on win rate."""
        best_rate = -1.0
        worst_rate = 2.0

        for hour, stats in self.hourly_performance.items():
            count = stats["count"]
            if count < 3:  # Need minimum sample size
                continue

            wins = stats["wins"]
            rate = wins / count

            if rate > best_rate:
                best_rate = rate
                self.best_hour = hour

            if rate < worst_rate:
                worst_rate = rate
                self.worst_hour = hour

    def add_recent_trade(self, trade_data: dict[str, Any]) -> None:
        """Add trade to recent trades list for pattern detection."""
        self.recent_trades.append(trade_data)
        # Keep only last 50 trades for memory efficiency
        if len(self.recent_trades) > 50:
            self.recent_trades = self.recent_trades[-50:]

    def get_summary(self) -> dict[str, Any]:
        """Get performance summary as dictionary."""
        return {
            "strategy_name": self.strategy_name,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "total_pnl_paisa": self.total_pnl,
            "total_pnl_rupees": self.total_pnl / 100,
            "max_consecutive_losses": self.max_consecutive_losses,
            "avg_holding_time_min": self.avg_holding_time,
            "best_hour": self.best_hour,
            "worst_hour": self.worst_hour,
            "lessons_count": len(self.lessons),
        }


class TradeAnalyzer:
    """Analyzes trades to extract insights and lessons.

    This class processes individual trades and maintains statistics
    about strategy performance, identifying patterns that lead to
    losses and extracting actionable lessons.

    Attributes:
        _db: Database session for persistence
        _lessons: List of lessons learned
        _strategy_performance: Dict mapping strategy names to performance metrics
        _lock: Thread lock for thread-safe operations
    """

    def __init__(self, db_session: Session | None = None):
        """Initialize the TradeAnalyzer.

        Args:
            db_session: SQLAlchemy session for database operations
        """
        self._db = db_session
        self._lessons: list[LessonLearned] = []
        self._strategy_performance: dict[str, StrategyPerformance] = {}
        self._lock = threading.RLock()

    def analyze_trade(self, trade: TradeRecord) -> dict[str, Any]:
        """Analyze a single trade and extract insights.

        Args:
            trade: Completed trade record

        Returns:
            Dictionary containing trade insights
        """
        with self._lock:
            insights = {
                "trade_id": trade.id,
                "strategy": trade.strategy,
                "instrument": trade.instrument_key,
                "was_win": trade.realized_pnl > 0,
                "pnl_paisa": trade.realized_pnl,
                "pnl_rupees": trade.realized_pnl / 100,
                "pnl_pct": self._calculate_pnl_pct(trade),
                "holding_time_min": trade.holding_duration_seconds / 60,
                "hour_of_entry": trade.entry_time.hour,
                "day_of_week": trade.entry_time.strftime("%A"),
                "entry_price": trade.entry_price,
                "exit_price": trade.exit_price,
            }

            # Add to strategy performance
            self._update_strategy_performance(trade, insights)

            logger.debug(
                f"Analyzed trade {trade.id} for {trade.strategy}: "
                f"P&L={insights['pnl_rupees']:.2f}, Win={insights['was_win']}"
            )

            return insights

    def _calculate_pnl_pct(self, trade: TradeRecord) -> float:
        """Calculate P&L percentage relative to capital at risk.

        Args:
            trade: Trade record

        Returns:
            P&L percentage
        """
        # Use entry price as basis for percentage calculation
        entry_value = trade.entry_price * trade.quantity
        if entry_value == 0:
            return 0.0

        # Calculate percentage return
        return (trade.realized_pnl / entry_value) * 100

    def _update_strategy_performance(
        self, trade: TradeRecord, insights: dict[str, Any]
    ) -> None:
        """Update performance metrics for the strategy.

        Args:
            trade: Trade record
            insights: Trade insights from analyze_trade
        """
        strategy = trade.strategy

        if strategy not in self._strategy_performance:
            self._strategy_performance[strategy] = StrategyPerformance(
                strategy_name=strategy
            )

        perf = self._strategy_performance[strategy]
        perf.total_trades += 1
        perf.total_pnl += trade.realized_pnl

        # Track consecutive losses
        if insights["was_win"]:
            perf.winning_trades += 1
            perf.current_consecutive_losses = 0
            perf.avg_win = (
                (perf.avg_win * (perf.winning_trades - 1) + insights["pnl_pct"])
                / perf.winning_trades
            )
        else:
            perf.losing_trades += 1
            perf.current_consecutive_losses += 1
            perf.avg_loss = (
                (perf.avg_loss * (perf.losing_trades - 1) + abs(insights["pnl_pct"]))
                / perf.losing_trades
            )

            # Update max consecutive losses
            if perf.current_consecutive_losses > perf.max_consecutive_losses:
                perf.max_consecutive_losses = perf.current_consecutive_losses

        # Update average holding time
        total_holding = perf.avg_holding_time * (perf.total_trades - 1)
        total_holding += insights["holding_time_min"]
        perf.avg_holding_time = total_holding / perf.total_trades

        # Recalculate metrics
        if perf.total_trades > 0:
            perf.win_rate = perf.winning_trades / perf.total_trades

        gross_profit = perf.avg_win * perf.winning_trades
        gross_loss = perf.avg_loss * perf.losing_trades
        if gross_loss > 0:
            perf.profit_factor = gross_profit / gross_loss
        else:
            perf.profit_factor = float("inf") if gross_profit > 0 else 0.0

        # Update hourly performance
        perf.update_hourly_performance(
            insights["hour_of_entry"], insights["was_win"], trade.realized_pnl
        )

        # Add to recent trades for pattern detection
        perf.add_recent_trade(insights)

    def get_strategy_performance(self, strategy_name: str) -> StrategyPerformance | None:
        """Get performance metrics for a specific strategy.

        Args:
            strategy_name: Name of the strategy

        Returns:
            StrategyPerformance object or None if not found
        """
        with self._lock:
            return self._strategy_performance.get(strategy_name)

    def get_all_performance(self) -> dict[str, StrategyPerformance]:
        """Get performance metrics for all strategies.

        Returns:
            Dictionary mapping strategy names to performance metrics
        """
        with self._lock:
            return self._strategy_performance.copy()

    def add_lesson(self, lesson: LessonLearned) -> None:
        """Add a lesson learned.

        Args:
            lesson: LessonLearned object to add
        """
        with self._lock:
            self._lessons.append(lesson)

            # Also add to strategy's lessons list
            if lesson.strategy_name and lesson.strategy_name in self._strategy_performance:
                self._strategy_performance[lesson.strategy_name].lessons.append(lesson)

    def get_lessons(
        self, category: str | None = None, strategy: str | None = None
    ) -> list[LessonLearned]:
        """Get lessons learned, optionally filtered.

        Args:
            category: Filter by category (optional)
            strategy: Filter by strategy name (optional)

        Returns:
            List of LessonLearned objects
        """
        with self._lock:
            lessons = self._lessons

            if category:
                lessons = [l for l in lessons if l.category == category]

            if strategy:
                lessons = [l for l in lessons if l.strategy_name == strategy]

            return lessons


class LearningEngine:
    """Main learning engine that drives continuous improvement.

    This class orchestrates the learning process by processing trades,
    analyzing losses, generating recommendations, and adjusting position
    sizes based on recent performance.

    Attributes:
        _analyzer: TradeAnalyzer instance for trade analysis
        _recommendations: List of current recommendations
        _lock: Thread lock for thread-safe operations
        _config: Configuration dictionary for learning parameters
    """

    def __init__(
        self, db_session: Session | None = None, config: dict[str, Any] | None = None
    ):
        """Initialize the LearningEngine.

        Args:
            db_session: SQLAlchemy session for database operations
            config: Configuration dictionary with learning parameters
        """
        self._analyzer = TradeAnalyzer(db_session)
        self._recommendations: list[dict[str, Any]] = []
        self._lock = threading.RLock()
        self._config = config or self._default_config()

    def _default_config(self) -> dict[str, Any]:
        """Return default configuration."""
        return {
            "min_trades_for_analysis": 5,
            "low_win_rate_threshold": 0.35,
            "high_win_rate_threshold": 0.6,
            "min_profit_factor": 1.0,
            "target_profit_factor": 1.5,
            "max_consecutive_losses_threshold": 4,
            "position_size_reduction_factor": 0.5,
            "position_size_increase_factor": 1.25,
            "lunch_hours": [11, 12, 13],
            "quick_exit_threshold_min": 5,
            "tight_sl_threshold_pct": -10,
        }

    def process_trade(self, trade: TradeRecord) -> dict[str, Any]:
        """Process a completed trade for learning.

        Args:
            trade: Completed trade record

        Returns:
            Trade insights dictionary
        """
        with self._lock:
            insights = self._analyzer.analyze_trade(trade)

            # Check for patterns that led to losses
            if not insights["was_win"]:
                self._analyze_loss(trade, insights)

            # Update recommendations periodically
            if self._analyzer._strategy_performance.get(trade.strategy, StrategyPerformance(strategy_name=trade.strategy)).total_trades % 5 == 0:
                self._recommendations = self._generate_recommendations()

            return insights

    def _analyze_loss(self, trade: TradeRecord, insights: dict[str, Any]) -> None:
        """Analyze a losing trade to extract lessons.

        Args:
            trade: Trade record
            insights: Trade insights from analyze_trade
        """
        issues: list[dict[str, Any]] = []

        # 1. Time-based issues
        if insights["hour_of_entry"] in self._config["lunch_hours"]:
            issues.append({
                "category": "timing",
                "description": "Trade during lunch hours (11AM-1:30PM) often underperforms due to low volume/volatility",
                "confidence": 0.7,
            })

        # 2. Holding time issues
        if insights["holding_time_min"] < self._config["quick_exit_threshold_min"]:
            issues.append({
                "category": "timing",
                "description": f"Very quick exits (<{self._config['quick_exit_threshold_min']} min) suggest premature stop hits or noise trading",
                "confidence": 0.6,
            })

        # 3. P&L pattern issues - tight stop loss
        pnl_pct = insights["pnl_pct"]
        if self._config["tight_sl_threshold_pct"] < pnl_pct < 0:
            issues.append({
                "category": "risk",
                "description": f"Small losses ({self._config['tight_sl_threshold_pct']}% to 0%) suggest SL too tight or whipsaw",
                "confidence": 0.65,
            })

        # 4. Day of week analysis
        if insights["day_of_week"] in ["Monday", "Friday"]:
            issues.append({
                "category": "market_condition",
                "description": f"Loss on {insights['day_of_week']} - check for overnight gap risk or weekend volatility",
                "confidence": 0.5,
            })

        # Create lessons from issues
        for issue in issues:
            lesson = LessonLearned(
                id=str(uuid.uuid4()),
                timestamp=datetime.now(),
                category=issue["category"],
                description=issue["description"],
                trigger_pattern={
                    "hour": insights["hour_of_entry"],
                    "day": insights["day_of_week"],
                    "instrument": trade.instrument_key,
                },
                impact=trade.realized_pnl,
                confidence=issue["confidence"],
                strategy_name=trade.strategy,
            )
            self._analyzer.add_lesson(lesson)

            logger.info(
                f"Lesson learned for {trade.strategy}: {issue['description']} "
                f"(confidence: {issue['confidence']:.0%})"
            )

    def generate_recommendations(self) -> list[dict[str, Any]]:
        """Generate actionable recommendations based on analysis.

        Returns:
            List of recommendation dictionaries
        """
        with self._lock:
            self._recommendations = self._generate_recommendations()
            return self._recommendations

    def _generate_recommendations(self) -> list[dict[str, Any]]:
        """Internal method to generate recommendations."""
        recommendations: list[dict[str, Any]] = []

        for strategy, perf in self._analyzer._strategy_performance.items():
            min_trades = self._config["min_trades_for_analysis"]

            # Check if strategy needs adjustment - low win rate
            if perf.total_trades >= min_trades and perf.win_rate < self._config["low_win_rate_threshold"]:
                recommendations.append({
                    "priority": "high",
                    "strategy": strategy,
                    "issue": "Low win rate",
                    "action": "Review entry conditions or reduce position size",
                    "current_win_rate": f"{perf.win_rate:.1%}",
                    "target_win_rate": f"{self._config['low_win_rate_threshold']:.0%}",
                    "metrics": perf.get_summary(),
                })

            # Check profit factor
            if perf.total_trades >= min_trades and perf.profit_factor < self._config["min_profit_factor"]:
                recommendations.append({
                    "priority": "high",
                    "strategy": strategy,
                    "issue": f"Unprofitable (PF < {self._config['min_profit_factor']})",
                    "action": "Consider quarantining strategy or reducing size significantly",
                    "current_pf": f"{perf.profit_factor:.2f}",
                    "target_pf": f"{self._config['target_profit_factor']}",
                    "metrics": perf.get_summary(),
                })

            # Check consecutive losses
            if perf.max_consecutive_losses >= self._config["max_consecutive_losses_threshold"]:
                recommendations.append({
                    "priority": "medium",
                    "strategy": strategy,
                    "issue": f"{perf.max_consecutive_losses} consecutive losses",
                    "action": "Check for market regime change or strategy degradation",
                    "suggestion": "Pause strategy for 24 hours or reduce size by 50%",
                    "metrics": perf.get_summary(),
                })

            # Check best/worst hour patterns
            if perf.worst_hour != -1 and perf.total_trades >= min_trades:
                worst_hour_stats = perf.hourly_performance.get(perf.worst_hour, {})
                if worst_hour_stats.get("count", 0) >= 3:
                    recommendations.append({
                        "priority": "low",
                        "strategy": strategy,
                        "issue": f"Poor performance at hour {perf.worst_hour}",
                        "action": "Avoid trading during this hour or tighten risk controls",
                        "metrics": {"hour_stats": worst_hour_stats},
                    })

        # Sort by priority
        priority_order = {"high": 0, "medium": 1, "low": 2}
        recommendations.sort(key=lambda x: priority_order.get(x["priority"], 3))

        return recommendations

    def adjust_position_size(self, strategy_name: str, base_size: int) -> int:
        """Dynamically adjust position size based on recent performance.

        Args:
            strategy_name: Name of the strategy
            base_size: Base position size

        Returns:
            Adjusted position size
        """
        with self._lock:
            perf = self._analyzer.get_strategy_performance(strategy_name)

            if perf is None:
                return base_size

            min_trades = self._config["min_trades_for_analysis"]

            # Not enough data
            if perf.total_trades < min_trades:
                return base_size

            # Reduce size if recent performance poor
            if perf.win_rate < self._config["low_win_rate_threshold"]:
                new_size = max(1, int(base_size * self._config["position_size_reduction_factor"]))
                logger.warning(
                    f"Reducing position size for {strategy_name} from {base_size} to {new_size} "
                    f"due to low win rate ({perf.win_rate:.1%})"
                )
                return new_size

            # Reduce size if unprofitable
            if perf.profit_factor < self._config["min_profit_factor"]:
                new_size = max(1, int(base_size * self._config["position_size_reduction_factor"]))
                logger.warning(
                    f"Reducing position size for {strategy_name} from {base_size} to {new_size} "
                    f"due to poor profit factor ({perf.profit_factor:.2f})"
                )
                return new_size

            # Reduce size during consecutive losses
            if perf.current_consecutive_losses >= 3:
                reduction = self._config["position_size_reduction_factor"] ** (perf.current_consecutive_losses - 2)
                new_size = max(1, int(base_size * reduction))
                logger.warning(
                    f"Reducing position size for {strategy_name} from {base_size} to {new_size} "
                    f"due to {perf.current_consecutive_losses} consecutive losses"
                )
                return new_size

            # Increase size if performing well
            if perf.win_rate > self._config["high_win_rate_threshold"] and perf.profit_factor > 2.0:
                new_size = int(base_size * self._config["position_size_increase_factor"])
                logger.info(
                    f"Increasing position size for {strategy_name} from {base_size} to {new_size} "
                    f"due to strong performance (WR: {perf.win_rate:.1%}, PF: {perf.profit_factor:.2f})"
                )
                return new_size

            return base_size

    def get_lessons_report(self, max_lessons: int = 10) -> str:
        """Generate a human-readable lessons learned report.

        Args:
            max_lessons: Maximum number of lessons to include

        Returns:
            Formatted report string
        """
        with self._lock:
            report: list[str] = []
            report.append("=" * 70)
            report.append("TRADE LEARNING REPORT")
            report.append("=" * 70)
            report.append("")

            # Strategy performance summary
            report.append("STRATEGY PERFORMANCE SUMMARY")
            report.append("-" * 70)

            if not self._analyzer._strategy_performance:
                report.append("No trades analyzed yet.")
            else:
                for strategy, perf in self._analyzer._strategy_performance.items():
                    report.append(f"\n  {strategy}:")
                    report.append(f"    Total Trades: {perf.total_trades}")
                    report.append(f"    Win Rate: {perf.win_rate:.1%} ({perf.winning_trades}/{perf.total_trades})")
                    report.append(f"    P&L: {perf.total_pnl / 100:,.2f}")
                    report.append(f"    Profit Factor: {perf.profit_factor:.2f}")
                    report.append(f"    Avg Win: {perf.avg_win:.2f}%")
                    report.append(f"    Avg Loss: {perf.avg_loss:.2f}%")
                    report.append(f"    Max Consecutive Losses: {perf.max_consecutive_losses}")
                    if perf.best_hour != -1:
                        report.append(f"    Best Hour: {perf.best_hour}:00")
                    if perf.worst_hour != -1:
                        report.append(f"    Worst Hour: {perf.worst_hour}:00")

            # Lessons learned
            lessons = self._analyzer.get_lessons()
            if lessons:
                report.append("\n")
                report.append("LESSONS LEARNED")
                report.append("-" * 70)

                for lesson in lessons[-max_lessons:]:
                    status = "[APPLIED]" if lesson.applied else "[NEW]"
                    report.append(f"\n  [{lesson.category.upper()}] {status}")
                    report.append(f"    Strategy: {lesson.strategy_name}")
                    report.append(f"    Description: {lesson.description}")
                    report.append(f"    Impact: {lesson.impact / 100:,.2f}")
                    report.append(f"    Confidence: {lesson.confidence:.0%}")
                    report.append(f"    Learned: {lesson.timestamp.strftime('%Y-%m-%d %H:%M')}")

            # Recommendations
            recommendations = self._recommendations or self._generate_recommendations()
            if recommendations:
                report.append("\n")
                report.append("RECOMMENDATIONS")
                report.append("-" * 70)

                for rec in recommendations:
                    report.append(f"\n  [{rec['priority'].upper()}] {rec['strategy']}")
                    report.append(f"    Issue: {rec['issue']}")
                    report.append(f"    Action: {rec['action']}")
                    if "suggestion" in rec:
                        report.append(f"    Suggestion: {rec['suggestion']}")

            report.append("\n")
            report.append("=" * 70)
            report.append(f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            report.append("=" * 70)

            return "\n".join(report)

    def get_strategy_stats(self, strategy_name: str) -> dict[str, Any] | None:
        """Get statistics for a specific strategy.

        Args:
            strategy_name: Name of the strategy

        Returns:
            Dictionary with strategy statistics or None
        """
        perf = self._analyzer.get_strategy_performance(strategy_name)
        if perf:
            return perf.get_summary()
        return None

    def export_lessons(self, filepath: str) -> None:
        """Export all lessons to a JSON file.

        Args:
            filepath: Path to output JSON file
        """
        with self._lock:
            data = {
                "exported_at": datetime.now().isoformat(),
                "lessons": [lesson.to_dict() for lesson in self._analyzer._lessons],
                "strategies": {
                    name: perf.get_summary()
                    for name, perf in self._analyzer._strategy_performance.items()
                },
            }

            with open(filepath, "w") as f:
                json.dump(data, f, indent=2)

            logger.info(f"Exported {len(self._analyzer._lessons)} lessons to {filepath}")

    def import_lessons(self, filepath: str) -> int:
        """Import lessons from a JSON file.

        Args:
            filepath: Path to input JSON file

        Returns:
            Number of lessons imported
        """
        with self._lock:
            with open(filepath, "r") as f:
                data = json.load(f)

            imported = 0
            for lesson_data in data.get("lessons", []):
                lesson = LessonLearned.from_dict(lesson_data)
                self._analyzer.add_lesson(lesson)
                imported += 1

            logger.info(f"Imported {imported} lessons from {filepath}")
            return imported


def create_learning_report(
    db_path: str,
    days: int = 7,
    output_path: str | None = None,
) -> str:
    """Create a learning report from recent trades in the database.

    This is a convenience function for generating reports from historical
    trade data without running the full trading system.

    Args:
        db_path: Path to SQLite database file
        days: Number of days of history to analyze
        output_path: Optional path to save the report

    Returns:
        Formatted report string
    """
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from src.persistence.models import TradeRecord

    # Create database session
    engine = create_engine(f"sqlite:///{db_path}")
    SessionLocal = sessionmaker(bind=engine)
    db = SessionLocal()

    try:
        # Create learning engine
        engine_learn = LearningEngine(db)

        # Load recent trades
        cutoff_date = datetime.now() - timedelta(days=days)
        trades = (
            db.query(TradeRecord)
            .filter(TradeRecord.exit_time >= cutoff_date)
            .order_by(TradeRecord.exit_time)
            .all()
        )

        logger.info(f"Loaded {len(trades)} trades from last {days} days")

        # Process each trade
        for trade in trades:
            engine_learn.process_trade(trade)

        # Generate report
        report = engine_learn.get_lessons_report()

        # Save to file if requested
        if output_path:
            with open(output_path, "w") as f:
                f.write(report)
            logger.info(f"Report saved to {output_path}")

        return report

    finally:
        db.close()
