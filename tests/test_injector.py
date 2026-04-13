"""Tests for memory injector that formats lessons for agent context."""

import pytest
from datetime import datetime
from src.memory.injector import MemoryInjector
from src.memory.memory_db import MemoryDB, MemoryLesson


class TestMemoryInjector:
    def _populate_db(self, db: MemoryDB) -> None:
        db.store(MemoryLesson(
            lesson_id="L001", category="regime_mismatch", strategy="ema_crossover",
            regime="trending_down", description="Avoid BUY in strong downtrend",
            severity="high", base_score=1.5, created_at=datetime.now(),
        ))
        db.store(MemoryLesson(
            lesson_id="L002", category="stop_too_tight", strategy="ema_crossover",
            regime="volatile", description="Widen SL in volatile markets by 1.5x ATR",
            severity="medium", base_score=1.0, created_at=datetime.now(),
        ))
        db.store(MemoryLesson(
            lesson_id="L003", category="premature_exit", strategy="vwap_breakout",
            regime="trending_up", description="Use trailing stop instead of fixed target",
            severity="low", base_score=0.5, created_at=datetime.now(),
        ))

    def test_get_lessons_for_regime(self):
        db = MemoryDB()
        self._populate_db(db)
        injector = MemoryInjector(memory_db=db, max_lessons=5)
        lessons = injector.get_lessons_for_agent(
            agent_name="regime_agent",
            regime="trending_down",
        )
        assert len(lessons) >= 1
        assert any("downtrend" in l.lower() for l in lessons)

    def test_get_lessons_for_strategy(self):
        db = MemoryDB()
        self._populate_db(db)
        injector = MemoryInjector(memory_db=db, max_lessons=5)
        lessons = injector.get_lessons_for_agent(
            agent_name="signal_validator",
            regime="volatile",
            strategy="ema_crossover",
        )
        assert len(lessons) >= 1
        assert any("volatile" in l.lower() or "widen" in l.lower() for l in lessons)

    def test_max_lessons_respected(self):
        db = MemoryDB()
        for i in range(20):
            db.store(MemoryLesson(
                lesson_id=f"L{i:03d}", category="test", strategy="test",
                regime="test", description=f"Lesson {i}",
                severity="medium", base_score=1.0, created_at=datetime.now(),
            ))
        injector = MemoryInjector(memory_db=db, max_lessons=3)
        lessons = injector.get_lessons_for_agent(agent_name="test", regime="test")
        assert len(lessons) <= 3

    def test_empty_db_returns_empty(self):
        db = MemoryDB()
        injector = MemoryInjector(memory_db=db, max_lessons=5)
        lessons = injector.get_lessons_for_agent(agent_name="test", regime="unknown")
        assert lessons == []

    def test_injection_recorded(self):
        db = MemoryDB()
        self._populate_db(db)
        injector = MemoryInjector(memory_db=db, max_lessons=5)
        injector.get_lessons_for_agent(agent_name="regime_agent", regime="trending_down")
        lesson = db.get_relevant_lessons(regime="trending_down")[0]
        assert lesson.times_injected == 1
