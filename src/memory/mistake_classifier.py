"""Rule-based trade mistake classifier.

Analyzes trade outcome metrics and classifies mistakes into
categories with severity levels. These feed the memory system
so agents can learn from past errors.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from loguru import logger


class MistakeCategory(str, Enum):
    """Types of trading mistakes."""

    REGIME_MISMATCH = "regime_mismatch"
    STOP_TOO_TIGHT = "stop_too_tight"
    STOP_TOO_LOOSE = "stop_too_loose"
    PREMATURE_EXIT = "premature_exit"
    LATE_EXIT = "late_exit"
    POOR_TIMING = "poor_timing"
    OVERTRADING = "overtrading"
    POSITION_SIZING = "position_sizing"
    CHASING = "chasing"
    SIGNAL_QUALITY = "signal_quality"


SEVERITY_SCORES = {
    "critical": 2.0,
    "high": 1.5,
    "medium": 1.0,
    "low": 0.5,
}


@dataclass
class Mistake:
    """A classified trading mistake.

    Attributes:
        category: Type of mistake.
        severity: critical, high, medium, or low.
        description: Human-readable description.
        lesson: Actionable lesson for future trades.
        score: Severity score for weighting.
    """

    category: MistakeCategory
    severity: str
    description: str
    lesson: str

    @property
    def score(self) -> float:
        return SEVERITY_SCORES.get(self.severity, 1.0)


class MistakeClassifier:
    """Classifies trade mistakes from outcome analysis.

    Uses deterministic rules to identify common trading errors.
    Each rule checks a specific condition and produces a Mistake
    with category, severity, and actionable lesson.
    """

    def classify(
        self, analysis: dict[str, Any], regime: str = "unknown"
    ) -> list[Mistake]:
        """Classify mistakes in a trade outcome.

        Args:
            analysis: Output from OutcomeAnalyzer.analyze().
            regime: Current market regime when the trade was taken.

        Returns:
            List of identified Mistake objects (may be empty for good trades).
        """
        mistakes: list[Mistake] = []
        direction = analysis.get("direction", "").upper()
        was_win = analysis.get("was_win", True)

        # --- Regime mismatch ---
        counter_trend = (
            (direction == "BUY" and regime == "trending_down")
            or (direction == "SELL" and regime == "trending_up")
        )
        if counter_trend:
            mistakes.append(Mistake(
                category=MistakeCategory.REGIME_MISMATCH,
                severity="high",
                description=f"{direction} signal taken in {regime} regime",
                lesson=f"Avoid {direction} trades when market is {regime}",
            ))

        # --- Stop too tight ---
        if analysis.get("quick_stop") and analysis.get("stop_loss_hit"):
            mistakes.append(Mistake(
                category=MistakeCategory.STOP_TOO_TIGHT,
                severity="medium",
                description="Stop loss hit within 10 minutes of entry",
                lesson="Widen stop loss or wait for better entry in this volatility",
            ))

        # --- Stop too loose ---
        if not was_win and not analysis.get("stop_loss_hit") and analysis.get("mae_pct", 0) > 3.0:
            mistakes.append(Mistake(
                category=MistakeCategory.STOP_TOO_LOOSE,
                severity="medium",
                description=f"Loss of {analysis.get('mae_pct', 0):.1f}% without hitting stop loss",
                lesson="Tighten stop loss or use trailing stop for this strategy",
            ))

        # --- Premature exit ---
        if analysis.get("premature_exit"):
            mistakes.append(Mistake(
                category=MistakeCategory.PREMATURE_EXIT,
                severity="low",
                description="Exited winning trade capturing less than 50% of MFE",
                lesson="Let winners run longer or use trailing stop instead of fixed target",
            ))

        # --- Late exit ---
        if analysis.get("late_exit"):
            mistakes.append(Mistake(
                category=MistakeCategory.LATE_EXIT,
                severity="low",
                description="Gave back more than 40% of maximum favorable excursion",
                lesson="Use trailing stop to protect profits once in the money",
            ))

        if mistakes:
            logger.info(
                "Classified {} mistakes for trade {} | categories: {}",
                len(mistakes),
                analysis.get("trade_id", "?"),
                [m.category.value for m in mistakes],
            )

        return mistakes
