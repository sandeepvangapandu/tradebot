"""Memory injector that retrieves and formats lessons for agent context.

Selects the most relevant lessons from the MemoryDB based on current
regime and strategy, formats them as human-readable strings, and
tracks injection counts.
"""

from __future__ import annotations

from loguru import logger

from src.memory.memory_db import MemoryDB


class MemoryInjector:
    """Retrieves and formats lessons for injection into agent prompts.

    Args:
        memory_db: The memory lesson database.
        max_lessons: Maximum lessons to inject per agent call.
    """

    def __init__(self, memory_db: MemoryDB, max_lessons: int = 5) -> None:
        self._db = memory_db
        self._max_lessons = max_lessons

    def get_lessons_for_agent(
        self,
        agent_name: str,
        regime: str = "",
        strategy: str = "",
    ) -> list[str]:
        """Get formatted lesson strings for an agent.

        Retrieves lessons relevant to the current context,
        formats them as human-readable strings, and records
        the injection.

        Args:
            agent_name: Name of the agent requesting lessons.
            regime: Current market regime.
            strategy: Current strategy name.

        Returns:
            List of formatted lesson strings.
        """
        lessons = self._db.get_relevant_lessons(
            regime=regime,
            strategy=strategy,
            max_lessons=self._max_lessons,
        )

        if not lessons:
            return []

        formatted = []
        for lesson in lessons:
            text = f"[{lesson.category}] {lesson.description}"
            formatted.append(text)
            self._db.record_injection(lesson.lesson_id)

        logger.debug(
            "Injected {} lessons into {} | regime={} | strategy={}",
            len(formatted),
            agent_name,
            regime,
            strategy,
        )

        return formatted
