"""In-memory lesson database with time-decay scoring.

Stores trade lessons with severity-based scores that decay over
time. Lessons can be boosted when they prove useful. Retrieval
is filtered by regime and strategy for contextual injection.

For persistence, lessons are also stored via the existing
LearningPersistence layer. This in-memory store is the
fast-path for pipeline injection.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from datetime import datetime

from loguru import logger


@dataclass
class MemoryLesson:
    """A lesson stored in the memory system.

    Attributes:
        lesson_id: Unique identifier.
        category: Mistake category (e.g., regime_mismatch).
        strategy: Strategy this lesson applies to.
        regime: Market regime when the mistake occurred.
        description: Human-readable lesson text.
        severity: critical, high, medium, low.
        base_score: Severity score (can be boosted).
        created_at: When the lesson was created.
        times_injected: How many times injected into agent context.
        times_useful: How many times the lesson proved useful.
        effective_score: Score after time-decay (computed on retrieval).
    """

    lesson_id: str
    category: str
    strategy: str
    regime: str
    description: str
    severity: str
    base_score: float
    created_at: datetime
    times_injected: int = 0
    times_useful: int = 0
    effective_score: float = 0.0


class MemoryDB:
    """In-memory lesson store with time-decay and boost.

    Args:
        decay_rate: Exponential decay rate per week (default 0.05 = 5%).
        decay_start_days: Days before decay begins (default 30).
        max_base_score_multiplier: Maximum boost cap as multiplier of
            original score (default 2.0).
    """

    def __init__(
        self,
        decay_rate: float = 0.05,
        decay_start_days: int = 30,
        max_base_score_multiplier: float = 2.0,
    ) -> None:
        self._lock = threading.Lock()
        self._lessons: dict[str, MemoryLesson] = {}
        self._decay_rate = decay_rate
        self._decay_start_days = decay_start_days
        self._max_multiplier = max_base_score_multiplier
        # Track original scores for cap calculation
        self._original_scores: dict[str, float] = {}

    def store(self, lesson: MemoryLesson) -> None:
        """Store a lesson in the database.

        Args:
            lesson: The lesson to store.
        """
        with self._lock:
            self._lessons[lesson.lesson_id] = lesson
            if lesson.lesson_id not in self._original_scores:
                self._original_scores[lesson.lesson_id] = lesson.base_score
        logger.debug("Stored lesson {} | category={}", lesson.lesson_id, lesson.category)

    def get_relevant_lessons(
        self,
        regime: str = "",
        strategy: str = "",
        max_lessons: int = 10,
    ) -> list[MemoryLesson]:
        """Retrieve lessons relevant to current context.

        Filters by regime and/or strategy, applies time-decay,
        and returns top N by effective score.

        Args:
            regime: Current market regime to filter by.
            strategy: Current strategy to filter by.
            max_lessons: Maximum lessons to return.

        Returns:
            List of MemoryLesson sorted by effective_score descending.
        """
        now = datetime.now()

        with self._lock:
            candidates = []
            for lesson in self._lessons.values():
                # Filter by regime if specified
                if regime and lesson.regime != regime:
                    continue
                # Filter by strategy if specified
                if strategy and lesson.strategy != strategy:
                    continue

                # Calculate effective score with time decay
                effective = self._apply_decay(lesson.base_score, lesson.created_at, now)
                lesson.effective_score = effective
                candidates.append(lesson)

        # Sort by effective score descending
        candidates.sort(key=lambda l: l.effective_score, reverse=True)
        return candidates[:max_lessons]

    def boost_lesson(self, lesson_id: str, factor: float = 1.1) -> None:
        """Boost a lesson's score when it proves useful.

        Args:
            lesson_id: ID of the lesson to boost.
            factor: Multiplicative boost factor.
        """
        with self._lock:
            lesson = self._lessons.get(lesson_id)
            if not lesson:
                return

            original = self._original_scores.get(lesson_id, lesson.base_score)
            max_score = original * self._max_multiplier
            lesson.base_score = min(lesson.base_score * factor, max_score)
            lesson.times_useful += 1

        logger.debug(
            "Boosted lesson {} | new_score={:.2f}",
            lesson_id,
            lesson.base_score,
        )

    def record_injection(self, lesson_id: str) -> None:
        """Record that a lesson was injected into agent context.

        Args:
            lesson_id: ID of the lesson that was injected.
        """
        with self._lock:
            lesson = self._lessons.get(lesson_id)
            if lesson:
                lesson.times_injected += 1

    def _apply_decay(
        self, base_score: float, created_at: datetime, now: datetime
    ) -> float:
        """Apply exponential time-decay to a score.

        Args:
            base_score: Original score.
            created_at: When the lesson was created.
            now: Current time.

        Returns:
            Decayed score.
        """
        age = now - created_at
        age_days = age.total_seconds() / 86400

        if age_days <= self._decay_start_days:
            return base_score

        # Weeks past the decay start
        decay_weeks = (age_days - self._decay_start_days) / 7.0
        decay_factor = (1 - self._decay_rate) ** decay_weeks
        return base_score * decay_factor

    def get_all_lessons(self) -> list[MemoryLesson]:
        """Get all stored lessons."""
        with self._lock:
            return list(self._lessons.values())

    def remove_lesson(self, lesson_id: str) -> None:
        """Remove a lesson from the database."""
        with self._lock:
            self._lessons.pop(lesson_id, None)
            self._original_scores.pop(lesson_id, None)
