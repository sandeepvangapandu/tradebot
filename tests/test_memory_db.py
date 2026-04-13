"""Tests for the memory lesson database with time-decay."""

import pytest
from datetime import datetime, timedelta
from src.memory.memory_db import MemoryDB, MemoryLesson


class TestMemoryDB:
    def test_store_and_retrieve_lesson(self):
        db = MemoryDB()
        lesson = MemoryLesson(
            lesson_id="L001",
            category="regime_mismatch",
            strategy="ema_crossover",
            regime="trending_down",
            description="Avoid BUY in strong downtrend",
            severity="high",
            base_score=1.5,
            created_at=datetime.now(),
        )
        db.store(lesson)
        retrieved = db.get_relevant_lessons(regime="trending_down", strategy="ema_crossover")
        assert len(retrieved) == 1
        assert retrieved[0].lesson_id == "L001"

    def test_lessons_filtered_by_regime(self):
        db = MemoryDB()
        db.store(MemoryLesson(
            lesson_id="L001", category="regime_mismatch", strategy="ema_crossover",
            regime="trending_down", description="test1", severity="high",
            base_score=1.5, created_at=datetime.now(),
        ))
        db.store(MemoryLesson(
            lesson_id="L002", category="stop_too_tight", strategy="ema_crossover",
            regime="volatile", description="test2", severity="medium",
            base_score=1.0, created_at=datetime.now(),
        ))
        results = db.get_relevant_lessons(regime="trending_down")
        assert len(results) == 1
        assert results[0].lesson_id == "L001"

    def test_time_decay_reduces_score(self):
        db = MemoryDB(decay_rate=0.05, decay_start_days=0)
        old_lesson = MemoryLesson(
            lesson_id="L001", category="regime_mismatch", strategy="ema_crossover",
            regime="trending_down", description="old lesson", severity="high",
            base_score=1.5, created_at=datetime.now() - timedelta(weeks=4),
        )
        db.store(old_lesson)
        retrieved = db.get_relevant_lessons(regime="trending_down")
        assert len(retrieved) == 1
        # Score should be decayed: 1.5 * (1-0.05)^4 = ~1.22
        assert retrieved[0].effective_score < 1.5

    def test_boost_lesson(self):
        db = MemoryDB()
        lesson = MemoryLesson(
            lesson_id="L001", category="regime_mismatch", strategy="ema_crossover",
            regime="trending_down", description="useful lesson", severity="high",
            base_score=1.5, created_at=datetime.now(),
        )
        db.store(lesson)
        db.boost_lesson("L001", factor=1.1)
        retrieved = db.get_relevant_lessons(regime="trending_down")
        assert retrieved[0].base_score == pytest.approx(1.65, rel=0.01)

    def test_boost_capped_at_2x(self):
        db = MemoryDB()
        lesson = MemoryLesson(
            lesson_id="L001", category="test", strategy="test",
            regime="test", description="test", severity="high",
            base_score=1.5, created_at=datetime.now(),
        )
        db.store(lesson)
        # Boost many times
        for _ in range(20):
            db.boost_lesson("L001", factor=1.1)
        retrieved = db.get_relevant_lessons(regime="test")
        assert retrieved[0].base_score <= 3.0  # 1.5 * 2.0 cap

    def test_max_lessons_returned(self):
        db = MemoryDB()
        for i in range(10):
            db.store(MemoryLesson(
                lesson_id=f"L{i:03d}", category="test", strategy="test",
                regime="test", description=f"lesson {i}", severity="high",
                base_score=float(i), created_at=datetime.now(),
            ))
        results = db.get_relevant_lessons(regime="test", max_lessons=5)
        assert len(results) == 5
        # Should return highest-scored first
        assert results[0].base_score >= results[-1].base_score
