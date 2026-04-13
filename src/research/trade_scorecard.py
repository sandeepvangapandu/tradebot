"""Trade Scorecard - Calculates weighted composite scores for trade signals.

All monetary values are in PAISA (integer) - 1 Rupee = 100 paisa.
All timestamps are in IST (Asia/Kolkata).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from zoneinfo import ZoneInfo

from src.research.models import (
    AnalysisComponent,
    MarketRegime,
    OptimalEntryType,
    ResearchConfig,
    SignalStrength,
    TradeResearchReport,
    TradeVerdict,
    VolatilityRegime,
)
from src.utils.degradation import get_degraded_analyzers

logger = logging.getLogger(__name__)

# Default weights for each analysis component
DEFAULT_WEIGHTS = {
    "trend_alignment": 0.18,
    "momentum": 0.13,
    "support_resistance": 0.09,
    "market_regime": 0.13,
    "volume": 0.09,
    "options": 0.09,
    "correlation": 0.04,
    "event_calendar": 0.04,
    "time_of_day": 0.04,
    "candle_patterns": 0.04,
    "ml_validator": 0.13,
}

# Score thresholds for verdict determination
EXECUTE_THRESHOLD = 75
REDUCE_SIZE_THRESHOLD = 65
SKIP_THRESHOLD = 50

# Position size multipliers
FULL_SIZE_MULTIPLIER = 1.0
REDUCED_SIZE_MULTIPLIER = 0.7
SKIP_SIZE_MULTIPLIER = 0.0


@dataclass
class AnalyzerMetrics:
    """Performance metrics for a single analyzer.

    Attributes:
        analyzer_name: Name of the analyzer.
        total_predictions: Total number of predictions tracked.
        correct_predictions: Number of correct directional predictions.
        total_pnl: Sum of all P&L values (for correlation calculation).
        correlation: Correlation between prediction scores and P&L.
        accuracy: Directional accuracy (0.0-1.0).
        avg_score_when_win: Average prediction score for winning trades.
        avg_score_when_loss: Average prediction score for losing trades.
    """

    analyzer_name: str
    total_predictions: int = 0
    correct_predictions: int = 0
    total_pnl: int = 0
    correlation: float = 0.0
    accuracy: float = 0.0
    avg_score_when_win: float = 0.0
    avg_score_when_loss: float = 0.0


class WeightOptimizer:
    """Optimizes analyzer weights based on actual trade performance.

    This class tracks analyzer predictions vs actual trade outcomes and
    automatically adjusts weights to give higher weight to analyzers that
    better predict profitable trades.

    Attributes:
        lookback_days: Number of days to look back for performance calculation.
        min_samples: Minimum number of samples before optimization kicks in.
        ema_alpha: Smoothing factor for weight transitions (0.0-1.0).
        min_weight: Minimum allowed weight per analyzer.
        max_weight: Maximum allowed weight per analyzer.
        weights_file: Path to JSON file for storing optimized weights.
    """

    def __init__(
        self,
        lookback_days: int = 30,
        min_samples: int = 20,
        ema_alpha: float = 0.3,
        min_weight: float = 0.02,
        max_weight: float = 0.30,
        weights_file: Path | None = None,
    ) -> None:
        """Initialize the WeightOptimizer.

        Args:
            lookback_days: Days of history to consider for optimization.
            min_samples: Minimum trades needed before optimizing.
            ema_alpha: Smoothing factor for weight transitions (lower = smoother).
            min_weight: Minimum weight bound per analyzer.
            max_weight: Maximum weight bound per analyzer.
            weights_file: Path to store/load optimized weights JSON.
        """
        self.lookback_days = lookback_days
        self.min_samples = min_samples
        self.ema_alpha = ema_alpha
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.weights_file = weights_file or Path("optimized_weights.json")

        # In-memory cache of predictions for current session
        self._prediction_cache: list[dict[str, Any]] = []

        # Loaded weights by regime
        self._optimized_weights: dict[str, dict[str, float]] = {}

        # Performance metrics cache
        self._metrics_cache: dict[str, AnalyzerMetrics] = {}
        self._cache_timestamp: datetime | None = None

        self._load_weights_from_file()

    def _load_weights_from_file(self) -> None:
        """Load previously optimized weights from JSON file."""
        if self.weights_file.exists():
            try:
                with open(self.weights_file, "r") as f:
                    data = json.load(f)
                    self._optimized_weights = data.get("weights_by_regime", {})
                    logger.info(
                        f"Loaded optimized weights for {len(self._optimized_weights)} regimes"
                    )
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Could not load weights file: {e}")
                self._optimized_weights = {}

    def _save_weights_to_file(self) -> None:
        """Save optimized weights to JSON file."""
        try:
            data = {
                "weights_by_regime": self._optimized_weights,
                "last_updated": datetime.now(ZoneInfo("Asia/Kolkata")).isoformat(),
            }
            with open(self.weights_file, "w") as f:
                json.dump(data, f, indent=2)
            logger.info("Saved optimized weights to file")
        except IOError as e:
            logger.error(f"Failed to save weights file: {e}")

    def _get_regime_key(
        self, market_regime: MarketRegime | str, volatility_regime: VolatilityRegime | str
    ) -> str:
        """Convert market/volatility regimes to a regime key.

        Args:
            market_regime: The market regime classification.
            volatility_regime: The volatility regime classification.

        Returns:
            A regime key string for weight lookup.
        """
        # Map detailed regimes to broader categories
        regime_str = str(market_regime).lower()
        vol_str = str(volatility_regime).lower()

        if "trend" in regime_str:
            market_category = "trending"
        elif "range" in regime_str or "quiet" in regime_str:
            market_category = "ranging"
        elif "volatile" in regime_str:
            market_category = "volatile"
        else:
            market_category = "mixed"

        if vol_str in ("high", "extreme"):
            vol_category = "high_volatility"
        else:
            vol_category = "low_volatility"

        return f"{market_category}_{vol_category}"

    def track_outcome(
        self,
        analyzer_name: str,
        prediction: float,
        actual_outcome: float,
        signal_id: str,
        market_regime: MarketRegime | str = MarketRegime.RANGING,
        volatility_regime: VolatilityRegime | str = VolatilityRegime.NORMAL,
        was_executed: bool = True,
    ) -> None:
        """Record a prediction and its actual outcome.

        Args:
            analyzer_name: Name of the analyzer.
            prediction: Predicted score (0-100).
            actual_outcome: Actual P&L in paisa.
            signal_id: Unique identifier for the trade signal.
            market_regime: Market regime at time of prediction.
            volatility_regime: Volatility regime at time of prediction.
            was_executed: Whether the trade was actually executed.
        """
        record = {
            "analyzer_name": analyzer_name,
            "prediction": prediction,
            "actual_outcome": actual_outcome,
            "signal_id": signal_id,
            "market_regime": str(market_regime),
            "volatility_regime": str(volatility_regime),
            "was_executed": was_executed,
            "timestamp": datetime.now(ZoneInfo("Asia/Kolkata")),
        }
        self._prediction_cache.append(record)

        # Persist to database if available
        try:
            self._persist_to_database(record)
        except Exception as e:
            logger.debug(f"Could not persist to database: {e}")

    def _persist_to_database(self, record: dict[str, Any]) -> None:
        """Persist performance record to database.

        Args:
            record: The performance record to persist.
        """
        try:
            from src.persistence.database import get_session
            from src.persistence.models import AnalyzerPerformance

            with get_session() as session:
                perf = AnalyzerPerformance(
                    signal_id=record["signal_id"],
                    analyzer_name=record["analyzer_name"],
                    prediction_score=record["prediction"],
                    prediction_contribution=0.0,  # Will be updated later
                    market_regime=record["market_regime"],
                    volatility_regime=record["volatility_regime"],
                    was_executed=record["was_executed"],
                )
                session.add(perf)
        except Exception as e:
            logger.debug(f"Database persistence skipped: {e}")

    def update_trade_outcome(
        self,
        signal_id: str,
        actual_pnl: int,
        trade_duration_seconds: int | None = None,
    ) -> None:
        """Update the actual outcome for a tracked trade.

        Args:
            signal_id: The signal identifier.
            actual_pnl: Actual P&L in paisa.
            trade_duration_seconds: How long the trade was held.
        """
        # Determine trade result
        if actual_pnl > 0:
            trade_result = "win"
        elif actual_pnl < 0:
            trade_result = "loss"
        else:
            trade_result = "breakeven"

        # Update in-memory cache
        for record in self._prediction_cache:
            if record["signal_id"] == signal_id:
                record["actual_outcome"] = actual_pnl
                record["trade_result"] = trade_result

        # Update database
        try:
            from src.persistence.database import get_session
            from src.persistence.models import AnalyzerPerformance

            with get_session() as session:
                records = session.query(AnalyzerPerformance).filter_by(
                    signal_id=signal_id
                ).all()
                for record in records:
                    record.actual_pnl = actual_pnl
                    record.trade_result = trade_result
                    record.trade_duration_seconds = trade_duration_seconds
                    record.outcome_recorded_at = datetime.now(ZoneInfo("Asia/Kolkata"))
        except Exception as e:
            logger.debug(f"Database update skipped: {e}")

        # Invalidate metrics cache
        self._metrics_cache = {}
        self._cache_timestamp = None

    def calculate_accuracy(self, analyzer_name: str) -> float:
        """Calculate directional accuracy for an analyzer.

        Args:
            analyzer_name: Name of the analyzer.

        Returns:
            Directional accuracy (0.0-1.0), or 0.0 if insufficient data.
        """
        metrics = self._get_analyzer_metrics(analyzer_name)
        return metrics.accuracy if metrics else 0.0

    def _get_analyzer_metrics(self, analyzer_name: str) -> AnalyzerMetrics | None:
        """Get performance metrics for an analyzer.

        Args:
            analyzer_name: Name of the analyzer.

        Returns:
            AnalyzerMetrics or None if insufficient data.
        """
        # Check cache
        if (
            analyzer_name in self._metrics_cache
            and self._cache_timestamp
            and (datetime.now() - self._cache_timestamp).seconds < 300
        ):
            return self._metrics_cache[analyzer_name]

        # Get recent records from cache and database
        records = self._get_recent_records(analyzer_name)

        if len(records) < self.min_samples:
            return None

        # Calculate metrics
        wins = sum(1 for r in records if r.get("actual_outcome", 0) > 0)
        losses = sum(1 for r in records if r.get("actual_outcome", 0) < 0)
        total_trades = wins + losses

        if total_trades == 0:
            return None

        accuracy = wins / total_trades if total_trades > 0 else 0.0

        # Calculate correlation between prediction and outcome
        predictions = [r["prediction"] for r in records if "actual_outcome" in r]
        outcomes = [r["actual_outcome"] for r in records if "actual_outcome" in r]

        correlation = 0.0
        if len(predictions) >= self.min_samples:
            correlation = self._calculate_correlation(predictions, outcomes)

        # Average scores for wins vs losses
        win_scores = [
            r["prediction"]
            for r in records
            if r.get("actual_outcome", 0) > 0
        ]
        loss_scores = [
            r["prediction"]
            for r in records
            if r.get("actual_outcome", 0) < 0
        ]

        metrics = AnalyzerMetrics(
            analyzer_name=analyzer_name,
            total_predictions=len(records),
            correct_predictions=wins,
            total_pnl=sum(outcomes) if outcomes else 0,
            correlation=correlation,
            accuracy=accuracy,
            avg_score_when_win=sum(win_scores) / len(win_scores) if win_scores else 0.0,
            avg_score_when_loss=sum(loss_scores) / len(loss_scores) if loss_scores else 0.0,
        )

        self._metrics_cache[analyzer_name] = metrics
        self._cache_timestamp = datetime.now()

        return metrics

    def _get_recent_records(self, analyzer_name: str) -> list[dict[str, Any]]:
        """Get recent performance records for an analyzer.

        Args:
            analyzer_name: Name of the analyzer.

        Returns:
            List of performance records.
        """
        cutoff = datetime.now(ZoneInfo("Asia/Kolkata")) - timedelta(
            days=self.lookback_days
        )

        # Get from in-memory cache
        records = [
            r
            for r in self._prediction_cache
            if r["analyzer_name"] == analyzer_name and r["timestamp"] >= cutoff
        ]

        # Try to get from database
        try:
            from src.persistence.database import get_session
            from src.persistence.models import AnalyzerPerformance

            with get_session() as session:
                db_records = (
                    session.query(AnalyzerPerformance)
                    .filter(
                        AnalyzerPerformance.analyzer_name == analyzer_name,
                        AnalyzerPerformance.timestamp >= cutoff,
                    )
                    .all()
                )

                for db_record in db_records:
                    records.append({
                        "analyzer_name": db_record.analyzer_name,
                        "prediction": db_record.prediction_score,
                        "actual_outcome": db_record.actual_pnl or 0,
                        "signal_id": db_record.signal_id,
                        "market_regime": db_record.market_regime,
                        "volatility_regime": db_record.volatility_regime,
                        "was_executed": db_record.was_executed,
                        "timestamp": db_record.timestamp,
                    })
        except Exception as e:
            logger.debug(f"Database query skipped: {e}")

        return records

    def _calculate_correlation(self, x: list[float], y: list[int]) -> float:
        """Calculate Pearson correlation coefficient.

        Args:
            x: First variable (predictions).
            y: Second variable (outcomes).

        Returns:
            Correlation coefficient (-1.0 to 1.0).
        """
        n = len(x)
        if n < 2:
            return 0.0

        mean_x = sum(x) / n
        mean_y = sum(y) / n

        numerator = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
        denom_x = sum((xi - mean_x) ** 2 for xi in x) ** 0.5
        denom_y = sum((yi - mean_y) ** 2 for yi in y) ** 0.5

        if denom_x == 0 or denom_y == 0:
            return 0.0

        return numerator / (denom_x * denom_y)

    def optimize_weights(
        self,
        current_weights: dict[str, float],
        regime: str = "global",
    ) -> dict[str, float]:
        """Optimize weights based on analyzer performance.

        Args:
            current_weights: Current weight dictionary.
            regime: Market regime key for regime-specific optimization.

        Returns:
            Optimized weight dictionary (or current weights if insufficient data).
        """
        # Check if we have enough data
        total_samples = 0
        analyzer_metrics: dict[str, AnalyzerMetrics] = {}

        for analyzer_name in current_weights.keys():
            metrics = self._get_analyzer_metrics(analyzer_name)
            if metrics:
                analyzer_metrics[analyzer_name] = metrics
                total_samples += metrics.total_predictions

        if total_samples < self.min_samples * len(current_weights):
            logger.info(
                f"Insufficient data for optimization: {total_samples} samples, "
                f"need {self.min_samples * len(current_weights)}"
            )
            return current_weights.copy()

        # Calculate new weights based on correlation and accuracy
        raw_scores: dict[str, float] = {}

        for analyzer_name, weight in current_weights.items():
            if analyzer_name in analyzer_metrics:
                metrics = analyzer_metrics[analyzer_name]
                # Combine correlation and accuracy into a score
                # High correlation means predictions align with P&L
                # High accuracy means predictions correctly identify winning trades
                correlation_score = max(0.0, metrics.correlation)  # Only positive correlation
                accuracy_score = metrics.accuracy

                # Weighted combination: correlation is more important
                performance_score = 0.6 * correlation_score + 0.4 * accuracy_score

                # Scale by current weight as baseline
                raw_scores[analyzer_name] = weight * (1.0 + performance_score)
            else:
                # No data, keep current weight
                raw_scores[analyzer_name] = weight

        # Apply bounds and normalize
        optimized = self._apply_bounds_and_normalize(raw_scores)

        # Apply EMA smoothing with current weights
        smoothed = self._apply_smoothing(current_weights, optimized)

        # Store optimized weights
        self._optimized_weights[regime] = smoothed.copy()
        self._save_weights_to_file()

        # Log the optimization
        self._log_optimization(current_weights, smoothed, analyzer_metrics)

        return smoothed

    def _apply_bounds_and_normalize(self, weights: dict[str, float]) -> dict[str, float]:
        """Apply min/max bounds and normalize weights to sum to 1.0.

        Args:
            weights: Raw weight dictionary.

        Returns:
            Bounded and normalized weights.
        """
        # Apply bounds
        bounded = {
            k: max(self.min_weight, min(self.max_weight, v))
            for k, v in weights.items()
        }

        # Normalize to sum to 1.0
        total = sum(bounded.values())
        if total > 0:
            return {k: v / total for k, v in bounded.items()}
        return weights

    def _apply_smoothing(
        self,
        current_weights: dict[str, float],
        new_weights: dict[str, float],
    ) -> dict[str, float]:
        """Apply EMA smoothing to prevent rapid weight changes.

        Args:
            current_weights: Current weight dictionary.
            new_weights: New weight dictionary.

        Returns:
            Smoothed weights.
        """
        smoothed = {}
        for key in current_weights.keys():
            old = current_weights.get(key, 0.0)
            new = new_weights.get(key, old)
            smoothed[key] = self.ema_alpha * new + (1 - self.ema_alpha) * old

        # Re-normalize after smoothing
        total = sum(smoothed.values())
        if total > 0:
            smoothed = {k: v / total for k, v in smoothed.items()}

        return smoothed

    def _log_optimization(
        self,
        old_weights: dict[str, float],
        new_weights: dict[str, float],
        metrics: dict[str, AnalyzerMetrics],
    ) -> None:
        """Log weight optimization details.

        Args:
            old_weights: Previous weights.
            new_weights: Optimized weights.
            metrics: Performance metrics per analyzer.
        """
        logger.info("=" * 60)
        logger.info("WEIGHT OPTIMIZATION COMPLETED")
        logger.info("=" * 60)

        for analyzer_name in old_weights.keys():
            old_w = old_weights.get(analyzer_name, 0.0)
            new_w = new_weights.get(analyzer_name, 0.0)
            change = new_w - old_w
            change_pct = (change / old_w * 100) if old_w > 0 else 0

            if analyzer_name in metrics:
                m = metrics[analyzer_name]
                logger.info(
                    f"{analyzer_name:20s}: {old_w:.3f} -> {new_w:.3f} "
                    f"({change:+.3f}, {change_pct:+.1f}%) | "
                    f"accuracy={m.accuracy:.2f}, correlation={m.correlation:.2f}"
                )
            else:
                logger.info(
                    f"{analyzer_name:20s}: {old_w:.3f} -> {new_w:.3f} "
                    f"({change:+.3f}, {change_pct:+.1f}%) | no data"
                )

        logger.info("=" * 60)

    def get_analyzer_performance(self) -> dict[str, dict[str, Any]]:
        """Get performance metrics for all analyzers.

        Returns:
            Dictionary mapping analyzer names to their performance metrics.
        """
        performance = {}

        # Get unique analyzer names from cache and database
        analyzer_names = set()
        for record in self._prediction_cache:
            analyzer_names.add(record["analyzer_name"])

        try:
            from src.persistence.database import get_session
            from src.persistence.models import AnalyzerPerformance

            with get_session() as session:
                db_names = session.query(AnalyzerPerformance.analyzer_name).distinct().all()
                analyzer_names.update(name[0] for name in db_names)
        except Exception as e:
            logger.debug(f"Database query skipped: {e}")

        for name in analyzer_names:
            metrics = self._get_analyzer_metrics(name)
            if metrics:
                performance[name] = {
                    "total_predictions": metrics.total_predictions,
                    "accuracy": metrics.accuracy,
                    "correlation": metrics.correlation,
                    "avg_score_when_win": metrics.avg_score_when_win,
                    "avg_score_when_loss": metrics.avg_score_when_loss,
                }

        return performance

    def get_weights_for_regime(
        self,
        market_regime: MarketRegime | str,
        volatility_regime: VolatilityRegime | str,
        default_weights: dict[str, float],
    ) -> dict[str, float]:
        """Get optimized weights for a specific market regime.

        Args:
            market_regime: Current market regime.
            volatility_regime: Current volatility regime.
            default_weights: Fallback weights if no optimized weights exist.

        Returns:
            Weight dictionary for the regime.
        """
        regime_key = self._get_regime_key(market_regime, volatility_regime)

        # Try specific regime first
        if regime_key in self._optimized_weights:
            return self._optimized_weights[regime_key].copy()

        # Try broader categories
        for key in self._optimized_weights:
            if key in regime_key or regime_key in key:
                return self._optimized_weights[key].copy()

        # Fall back to global weights
        if "global" in self._optimized_weights:
            return self._optimized_weights["global"].copy()

        return default_weights.copy()

    def generate_weekly_report(self) -> str:
        """Generate a weekly weight optimization report.

        Returns:
            Formatted report string.
        """
        lines = ["=" * 70, "WEEKLY WEIGHT OPTIMIZATION REPORT", "=" * 70]

        performance = self.get_analyzer_performance()

        if not performance:
            lines.append("No performance data available yet.")
            return "\n".join(lines)

        lines.append(f"\nLookback Period: {self.lookback_days} days")
        lines.append(f"Minimum Samples Required: {self.min_samples}")
        lines.append(f"EMA Smoothing Alpha: {self.ema_alpha}")
        lines.append("")

        # Sort by accuracy
        sorted_analyzers = sorted(
            performance.items(),
            key=lambda x: x[1].get("accuracy", 0),
            reverse=True,
        )

        lines.append(f"{'Analyzer':<25} {'Accuracy':>10} {'Correlation':>12} {'Samples':>8}")
        lines.append("-" * 70)

        for name, metrics in sorted_analyzers:
            acc = metrics.get("accuracy", 0)
            corr = metrics.get("correlation", 0)
            samples = metrics.get("total_predictions", 0)
            lines.append(f"{name:<25} {acc:>10.2%} {corr:>12.3f} {samples:>8d}")

        lines.append("")
        lines.append("Optimized Weights by Regime:")
        lines.append("-" * 70)

        for regime, weights in self._optimized_weights.items():
            lines.append(f"\n{regime}:")
            for name, weight in sorted(weights.items()):
                lines.append(f"  {name:<25} {weight:>8.3f}")

        lines.append("=" * 70)

        return "\n".join(lines)


class TradeScorecard:
    """Calculates weighted trade scores and generates research reports.

    This class combines multiple analysis components into a final composite score,
    determines the trade verdict, and generates actionable recommendations.

    Attributes:
        weights: Dictionary mapping component names to their weights.
        config: ResearchConfig instance with thresholds and settings.
        weight_optimizer: Optional WeightOptimizer for dynamic weight adjustment.
    """

    def __init__(
        self,
        weights: dict[str, float] | None = None,
        config: ResearchConfig | None = None,
        weight_optimizer: WeightOptimizer | None = None,
    ) -> None:
        """Initialize the TradeScorecard.

        Args:
            weights: Custom weights for components. Uses DEFAULT_WEIGHTS if None.
            config: Research configuration. Uses default if None.
            weight_optimizer: Optional WeightOptimizer for dynamic weight adjustment.
        """
        self._default_weights = weights or DEFAULT_WEIGHTS.copy()
        self.weights = self._default_weights.copy()
        self.config = config or ResearchConfig()
        self.weight_optimizer = weight_optimizer

        # Validate weights sum to approximately 1.0
        total_weight = sum(self.weights.values())
        if not (0.99 <= total_weight <= 1.01):
            raise ValueError(f"Weights must sum to 1.0, got {total_weight}")

    def calculate_final_score(
        self,
        components: list[AnalysisComponent],
        degraded_analyzers: list[str] | None = None,
    ) -> float:
        """Calculate the weighted final score from analysis components.

        Formula: final_score = sum(component_score * component_weight * component_confidence/100)

        If more than 50% of analyzers are degraded, the final score is capped at 60
        to indicate reduced confidence in the analysis.

        Args:
            components: List of AnalysisComponent results from various analyzers.
            degraded_analyzers: Optional list of analyzer names that returned
                default/degraded values.

        Returns:
            Final composite score between 0 and 100.
        """
        if not components:
            return 0.0

        total_score = 0.0
        total_weight_used = 0.0

        for component in components:
            # Get weight for this component, default to 0 if not found
            weight = self.weights.get(component.name, 0.0)

            # Normalize confidence to 0.0-1.0 range
            confidence_factor = component.confidence / 100.0

            # Calculate weighted contribution
            contribution = component.score * weight * confidence_factor
            total_score += contribution
            total_weight_used += weight * confidence_factor

        # Normalize if total weight used is less than 1.0 (some components missing)
        if total_weight_used > 0 and total_weight_used < 0.99:
            total_score = total_score / total_weight_used

        # Check for degraded analyzers and cap score if >50% degraded
        degraded = degraded_analyzers or get_degraded_analyzers(components)
        if components and degraded:
            degraded_ratio = len(degraded) / len(components)
            if degraded_ratio > 0.5:
                # Cap score at 60 when majority of analyzers are degraded
                total_score = min(total_score, 60.0)

        # Clamp to valid range
        return max(0.0, min(100.0, total_score))

    def determine_verdict(
        self,
        final_score: float,
        research_config: ResearchConfig | None = None,
    ) -> tuple[TradeVerdict, float]:
        """Determine the trade verdict based on final score.

        Args:
            final_score: The calculated final composite score (0-100).
            research_config: Optional config with custom thresholds.

        Returns:
            Tuple of (TradeVerdict, position_size_multiplier).
            - Score >= 75: EXECUTE (1.0x size)
            - Score 65-74: REDUCE_SIZE (0.7x size)
            - Score 50-64: SKIP (0x size)
            - Score < 50: SKIP
        """
        config = research_config or self.config

        # Use config thresholds if available, otherwise use defaults
        execute_threshold = getattr(config, "score_for_full_size", EXECUTE_THRESHOLD)
        reduce_threshold = getattr(config, "min_score", REDUCE_SIZE_THRESHOLD)

        if final_score >= execute_threshold:
            return TradeVerdict.EXECUTE, FULL_SIZE_MULTIPLIER
        elif final_score >= reduce_threshold:
            return TradeVerdict.REDUCE_SIZE, REDUCED_SIZE_MULTIPLIER
        else:
            return TradeVerdict.SKIP, SKIP_SIZE_MULTIPLIER

    def determine_signal_strength(
        self,
        final_score: float,
        direction: str,
    ) -> SignalStrength:
        """Map final score to SignalStrength enum based on direction.

        Args:
            final_score: The calculated final composite score (0-100).
            direction: Trade direction - "BUY" or "SELL".

        Returns:
            SignalStrength enum value appropriate for the score and direction.
        """
        is_buy = direction.upper() == "BUY"

        if final_score >= 85:
            return SignalStrength.STRONG_BUY if is_buy else SignalStrength.STRONG_SELL
        elif final_score >= 70:
            return SignalStrength.BUY if is_buy else SignalStrength.SELL
        elif final_score >= 55:
            return SignalStrength.WEAK_BUY if is_buy else SignalStrength.WEAK_SELL
        elif final_score >= 45:
            return SignalStrength.NEUTRAL
        elif final_score >= 30:
            return SignalStrength.WEAK_SELL if is_buy else SignalStrength.WEAK_BUY
        elif final_score >= 15:
            return SignalStrength.SELL if is_buy else SignalStrength.BUY
        else:
            return SignalStrength.STRONG_SELL if is_buy else SignalStrength.STRONG_BUY

    def calculate_adjustments(
        self,
        analysis_results: dict[str, Any],
        final_score: float,
    ) -> dict[str, Any]:
        """Calculate trade adjustments based on analysis results.

        Args:
            analysis_results: Dictionary containing analysis data from various components.
            final_score: The calculated final composite score.

        Returns:
            Dictionary with adjustment recommendations:
            - sl_adjustment: Stop-loss distance multiplier (0.5-2.0)
            - target_adjustment: Target distance multiplier (0.5-2.0)
            - entry_type: Recommended entry order type
            - reasoning: Explanation for the adjustments
        """
        adjustments = {
            "sl_adjustment": 1.0,
            "target_adjustment": 1.0,
            "entry_type": OptimalEntryType.MARKET,
            "reasoning": [],
        }

        # Extract relevant data
        volatility_regime = analysis_results.get("volatility_regime", VolatilityRegime.NORMAL)
        market_regime = analysis_results.get("market_regime", MarketRegime.RANGING)
        trend_strength = analysis_results.get("trend_strength", 50)
        volume_ratio = analysis_results.get("volume_ratio", 1.0)
        nearest_support_distance = analysis_results.get("nearest_support_distance_pct", 0)

        # SL adjustment based on volatility
        if volatility_regime == VolatilityRegime.EXTREME:
            adjustments["sl_adjustment"] = 1.5
            adjustments["reasoning"].append("High volatility: widening SL by 50%")
        elif volatility_regime == VolatilityRegime.HIGH:
            adjustments["sl_adjustment"] = 1.25
            adjustments["reasoning"].append("Elevated volatility: widening SL by 25%")
        elif volatility_regime == VolatilityRegime.LOW:
            adjustments["sl_adjustment"] = 0.75
            adjustments["reasoning"].append("Low volatility: tightening SL by 25%")

        # Target adjustment based on trend strength and market regime
        if market_regime in (MarketRegime.STRONG_UPTREND, MarketRegime.STRONG_DOWNTREND):
            adjustments["target_adjustment"] = 1.3
            adjustments["reasoning"].append("Strong trend: extending target by 30%")
        elif trend_strength >= 70:
            adjustments["target_adjustment"] = 1.2
            adjustments["reasoning"].append("High trend strength: extending target by 20%")
        elif market_regime == MarketRegime.RANGING:
            adjustments["target_adjustment"] = 0.8
            adjustments["reasoning"].append("Ranging market: reducing target by 20%")

        # Entry type recommendation
        if nearest_support_distance > 0 and nearest_support_distance < 1.0:
            adjustments["entry_type"] = OptimalEntryType.LIMIT_AT_SUPPORT
            adjustments["reasoning"].append("Near support: use limit order for better entry")
        elif volume_ratio < 0.7:
            adjustments["entry_type"] = OptimalEntryType.WAIT_FOR_PULLBACK
            adjustments["reasoning"].append("Low volume: wait for pullback entry")

        return adjustments

    def generate_summary(
        self,
        components: list[AnalysisComponent],
        verdict: TradeVerdict,
        key_risks: list[str],
        key_supports: list[str],
    ) -> str:
        """Generate a human-readable 2-3 sentence summary of the analysis.

        Args:
            components: List of analysis components with scores.
            verdict: The final trade verdict.
            key_risks: List of top risk factors.
            key_supports: List of top supporting factors.

        Returns:
            A concise 2-3 sentence summary string.
        """
        # Sort components by weighted contribution (score * weight)
        sorted_components = sorted(
            components,
            key=lambda c: c.score * self.weights.get(c.name, 0),
            reverse=True,
        )

        # Get top positive and negative factors
        top_positive = [c for c in sorted_components if c.score >= 60][:2]
        top_negative = [c for c in sorted_components if c.score < 50][:2]

        sentences = []

        # Verdict sentence
        if verdict == TradeVerdict.EXECUTE:
            sentences.append("Trade signal meets all criteria for execution.")
        elif verdict == TradeVerdict.REDUCE_SIZE:
            sentences.append("Trade signal is acceptable but with reduced position size.")
        else:
            sentences.append("Trade signal does not meet minimum criteria.")

        # Supporting factors sentence
        if key_supports:
            support_text = "; ".join(key_supports[:2])
            sentences.append(f"Key supports: {support_text}.")

        # Risk factors sentence (only if significant)
        if key_risks and verdict != TradeVerdict.EXECUTE:
            risk_text = "; ".join(key_risks[:2])
            sentences.append(f"Key risks: {risk_text}.")

        # Component summary if needed
        if not key_supports and top_positive:
            positive_names = ", ".join([c.name.replace("_", " ") for c in top_positive[:2]])
            sentences.append(f"Strongest factors: {positive_names}.")

        summary = " ".join(sentences)
        return summary[:500]  # Limit length

    def score(
        self,
        components: list[AnalysisComponent],
        signal_info: dict[str, Any],
    ) -> TradeResearchReport:
        """Calculate complete trade research report from analysis components.

        This is the main entry point that orchestrates all scoring calculations
        and generates the final TradeResearchReport.

        Args:
            components: List of AnalysisComponent results from various analyzers.
            signal_info: Dictionary containing signal metadata:
                - signal_id: Unique identifier
                - instrument_key: Instrument to trade
                - direction: "BUY" or "SELL"
                - strategy_name: Originating strategy name
                - market_regime: Current MarketRegime
                - volatility_regime: Current VolatilityRegime
                - analysis_results: Additional analysis data for adjustments

        Returns:
            Complete TradeResearchReport with scores, verdict, and recommendations.
        """
        # Get current timestamp in IST
        timestamp = datetime.now(ZoneInfo("Asia/Kolkata"))

        # Identify degraded analyzers
        degraded_analyzers = get_degraded_analyzers(components)

        # Calculate final composite score (with degradation handling)
        final_score = self.calculate_final_score(components, degraded_analyzers)

        # Determine verdict and position size
        verdict, size_multiplier = self.determine_verdict(final_score, self.config)

        # Determine signal strength
        direction = signal_info.get("direction", "BUY")
        signal_strength = self.determine_signal_strength(final_score, direction)

        # Calculate adjustments
        analysis_results = signal_info.get("analysis_results", {})
        adjustments = self.calculate_adjustments(analysis_results, final_score)

        # Extract key risks and supports from components
        key_risks = []
        key_supports = []
        for comp in components:
            if comp.score < 40:
                key_risks.append(f"{comp.name}: {comp.reasoning}")
            elif comp.score > 75:
                key_supports.append(f"{comp.name}: {comp.reasoning}")

        # Add degradation warning if applicable
        degraded_ratio = len(degraded_analyzers) / len(components) if components else 0
        if degraded_analyzers:
            if degraded_ratio > 0.5:
                key_risks.insert(0, f"DEGRADED MODE: {len(degraded_analyzers)}/{len(components)} analyzers failed ({', '.join(degraded_analyzers[:3])})")
            else:
                key_risks.append(f"Partial degradation: {', '.join(degraded_analyzers)} analysis unavailable")

        # Generate summary
        summary = self.generate_summary(components, verdict, key_risks, key_supports)

        # Add degradation notice to summary if significant
        if degraded_ratio > 0.5:
            summary = f"WARNING: Analysis running in degraded mode. {summary}"

        # Build the report
        report = TradeResearchReport(
            signal_id=signal_info.get("signal_id", ""),
            timestamp=timestamp,
            instrument_key=signal_info.get("instrument_key", ""),
            direction=direction,
            strategy_name=signal_info.get("strategy_name", ""),
            final_score=final_score,
            components=components,
            verdict=verdict,
            signal_strength=signal_strength,
            market_regime=signal_info.get("market_regime", MarketRegime.RANGING),
            volatility_regime=signal_info.get(
                "volatility_regime", VolatilityRegime.NORMAL
            ),
            suggested_quantity_multiplier=size_multiplier,
            suggested_sl_adjustment=adjustments["sl_adjustment"],
            suggested_target_adjustment=adjustments["target_adjustment"],
            optimal_entry_type=adjustments["entry_type"],
            key_risks=key_risks[:5],  # Top 5 risks
            key_supports=key_supports[:5],  # Top 5 supports
            summary=summary,
            degraded_analyzers=degraded_analyzers,
        )

        # Track predictions for weight optimization
        self._track_component_predictions(
            components, signal_info, timestamp, final_score
        )

        return report

    def _track_component_predictions(
        self,
        components: list[AnalysisComponent],
        signal_info: dict[str, Any],
        timestamp: datetime,
        final_score: float,
    ) -> None:
        """Track component predictions for weight optimization.

        Args:
            components: List of analysis components.
            signal_info: Signal metadata.
            timestamp: Analysis timestamp.
            final_score: Final composite score.
        """
        if not self.weight_optimizer:
            return

        signal_id = signal_info.get("signal_id", "")
        market_regime = signal_info.get("market_regime", MarketRegime.RANGING)
        volatility_regime = signal_info.get(
            "volatility_regime", VolatilityRegime.NORMAL
        )

        for component in components:
            # Calculate contribution (score * weight * confidence)
            weight = self.weights.get(component.name, 0.0)
            confidence_factor = component.confidence / 100.0
            contribution = component.score * weight * confidence_factor

            self.weight_optimizer.track_outcome(
                analyzer_name=component.name,
                prediction=component.score,
                actual_outcome=0,  # Will be updated when trade closes
                signal_id=signal_id,
                market_regime=market_regime,
                volatility_regime=volatility_regime,
                was_executed=False,  # Will be updated when trade executes
            )

    def update_weights_from_outcome(
        self,
        signal_id: str,
        actual_pnl: int,
        trade_duration_seconds: int | None = None,
    ) -> None:
        """Update weights based on actual trade outcome.

        This method should be called after a trade closes to update
        the weight optimizer with the actual P&L.

        Args:
            signal_id: The signal identifier.
            actual_pnl: Actual P&L in paisa.
            trade_duration_seconds: How long the trade was held.
        """
        if not self.weight_optimizer:
            return

        self.weight_optimizer.update_trade_outcome(
            signal_id=signal_id,
            actual_pnl=actual_pnl,
            trade_duration_seconds=trade_duration_seconds,
        )

        logger.info(
            f"Updated trade outcome for signal {signal_id}: "
            f"P&L={actual_pnl} paisa"
        )

    def refresh_weights(
        self,
        market_regime: MarketRegime | str = MarketRegime.RANGING,
        volatility_regime: VolatilityRegime | str = VolatilityRegime.NORMAL,
    ) -> dict[str, float]:
        """Refresh weights using the optimizer for current market regime.

        Args:
            market_regime: Current market regime.
            volatility_regime: Current volatility regime.

        Returns:
            Updated weight dictionary.
        """
        if not self.weight_optimizer:
            return self.weights.copy()

        # Get regime-specific weights if available
        regime_weights = self.weight_optimizer.get_weights_for_regime(
            market_regime, volatility_regime, self._default_weights
        )

        # Optimize weights based on recent performance
        optimized = self.weight_optimizer.optimize_weights(
            current_weights=regime_weights,
            regime=self.weight_optimizer._get_regime_key(
                market_regime, volatility_regime
            ),
        )

        # Validate optimized weights
        total = sum(optimized.values())
        if 0.99 <= total <= 1.01:
            self.weights = optimized
            logger.info(f"Refreshed weights for regime {market_regime}/{volatility_regime}")
        else:
            logger.warning(f"Optimized weights sum to {total}, keeping current weights")

        return self.weights.copy()

    def update_weights(self, new_weights: dict[str, float]) -> None:
        """Update the component weights.

        Args:
            new_weights: Dictionary mapping component names to new weights.

        Raises:
            ValueError: If weights do not sum to approximately 1.0.
        """
        total_weight = sum(new_weights.values())
        if not (0.99 <= total_weight <= 1.01):
            raise ValueError(f"Weights must sum to 1.0, got {total_weight}")
        self.weights = new_weights.copy()

    def get_component_contributions(
        self,
        components: list[AnalysisComponent],
    ) -> dict[str, float]:
        """Calculate each component's contribution to the final score.

        Args:
            components: List of analysis components.

        Returns:
            Dictionary mapping component names to their weighted contributions.
        """
        contributions = {}
        for component in components:
            weight = self.weights.get(component.name, 0.0)
            confidence_factor = component.confidence / 100.0
            contribution = component.score * weight * confidence_factor
            contributions[component.name] = contribution
        return contributions
