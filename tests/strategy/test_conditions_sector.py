"""Tests for src/strategy/conditions_sector.py.

All tests use inline mocks — no external fixtures or DB connections required.
The DB engine is mocked to return controlled rank data.
"""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_engine_with_rows(
    rank_rows: list[tuple],
    active_count: int = 9,
) -> MagicMock:
    """Return a mock SQLAlchemy engine that returns *rank_rows* for any query.

    Args:
        rank_rows: Rows returned by the rank query. Each row is a tuple of
            (trade_date, rs_score, rank, rank_change).
        active_count: COUNT(*) for sector_indices (used for quartile logic).

    Returns:
        Mock engine with a `.connect()` context manager.
    """
    engine = MagicMock()

    def _connect_cm():
        cm = MagicMock()
        conn = MagicMock()

        def _execute(sql, params=None):
            result = MagicMock()
            sql_str = str(sql) if not isinstance(sql, str) else sql
            # Distinguish COUNT query from rank query
            if "COUNT" in sql_str.upper():
                result.scalar.return_value = active_count
                return result
            # Rank rows
            result.fetchone.return_value = rank_rows[0] if rank_rows else None
            return result

        conn.execute.side_effect = _execute
        cm.__enter__ = MagicMock(return_value=conn)
        cm.__exit__ = MagicMock(return_value=False)
        return cm

    engine.connect.side_effect = _connect_cm
    return engine


# ---------------------------------------------------------------------------
# sector_in_top_quartile
# ---------------------------------------------------------------------------

class TestSectorInTopQuartile:
    """Tests for sector_in_top_quartile()."""

    def test_sector_in_top_quartile_true_for_leader(self):
        """Returns True when sector has rank 1 out of 9 (clearly top quartile)."""
        from src.strategy.conditions_sector import sector_in_top_quartile

        # rank=1, rs_score=8.5, rank_change=2
        engine = _make_engine_with_rows(
            [(date(2026, 5, 8), 8.5, 1, 2)],
            active_count=9,
        )
        result = sector_in_top_quartile("HDFCBANK", db_engine=engine)
        assert result is True

    def test_sector_in_top_quartile_true_for_rank_3_of_9(self):
        """Returns True when rank=3 out of 9 (cutoff = ceil(9/4) = 3)."""
        from src.strategy.conditions_sector import sector_in_top_quartile

        engine = _make_engine_with_rows(
            [(date(2026, 5, 8), 2.1, 3, 1)],
            active_count=9,
        )
        result = sector_in_top_quartile("TCS", db_engine=engine)
        assert result is True

    def test_sector_in_top_quartile_false_for_rank_4_of_9(self):
        """Returns False when rank=4 out of 9 (just outside top quartile)."""
        from src.strategy.conditions_sector import sector_in_top_quartile

        engine = _make_engine_with_rows(
            [(date(2026, 5, 8), 1.5, 4, 0)],
            active_count=9,
        )
        result = sector_in_top_quartile("RELIANCE", db_engine=engine)
        assert result is False

    def test_sector_in_top_quartile_false_when_no_data(self):
        """Returns False when no DB row found for sector."""
        from src.strategy.conditions_sector import sector_in_top_quartile

        engine = _make_engine_with_rows([], active_count=9)
        result = sector_in_top_quartile("HDFCBANK", db_engine=engine)
        assert result is False

    def test_sector_in_top_quartile_resolves_stock_symbol(self):
        """Resolves known stock symbols to their sector before querying."""
        from src.strategy.conditions_sector import sector_in_top_quartile

        engine = _make_engine_with_rows(
            [(date(2026, 5, 8), 5.0, 1, 1)],
            active_count=9,
        )
        # SBIN maps to NIFTY_BANK
        result = sector_in_top_quartile("SBIN", db_engine=engine)
        assert result is True


# ---------------------------------------------------------------------------
# sector_in_bottom_quartile
# ---------------------------------------------------------------------------

class TestSectorInBottomQuartile:
    """Tests for sector_in_bottom_quartile()."""

    def test_sector_in_bottom_quartile_true_for_laggard(self):
        """Returns True when sector has rank 9 out of 9 (clearly bottom quartile)."""
        from src.strategy.conditions_sector import sector_in_bottom_quartile

        # With 9 sectors: bottom_cutoff = 9 - ceil(9/4) + 1 = 9 - 3 + 1 = 7
        # So ranks 7, 8, 9 are bottom quartile
        engine = _make_engine_with_rows(
            [(date(2026, 5, 8), -3.5, 9, -2)],
            active_count=9,
        )
        result = sector_in_bottom_quartile("RELIANCE", db_engine=engine)
        assert result is True

    def test_sector_in_bottom_quartile_true_for_rank_7_of_9(self):
        """Returns True when rank=7 out of 9 (first bottom-quartile rank)."""
        from src.strategy.conditions_sector import sector_in_bottom_quartile

        engine = _make_engine_with_rows(
            [(date(2026, 5, 8), -1.2, 7, -1)],
            active_count=9,
        )
        result = sector_in_bottom_quartile("ITC", db_engine=engine)
        assert result is True

    def test_sector_in_bottom_quartile_false_for_middle_rank(self):
        """Returns False when sector is in the middle (rank 5 of 9)."""
        from src.strategy.conditions_sector import sector_in_bottom_quartile

        engine = _make_engine_with_rows(
            [(date(2026, 5, 8), 0.2, 5, 0)],
            active_count=9,
        )
        result = sector_in_bottom_quartile("INFY", db_engine=engine)
        assert result is False

    def test_sector_in_bottom_quartile_false_when_no_data(self):
        """Returns False when no DB row found."""
        from src.strategy.conditions_sector import sector_in_bottom_quartile

        engine = _make_engine_with_rows([], active_count=9)
        result = sector_in_bottom_quartile("TCS", db_engine=engine)
        assert result is False


# ---------------------------------------------------------------------------
# sector_rs_above
# ---------------------------------------------------------------------------

class TestSectorRsAbove:
    """Tests for sector_rs_above()."""

    def test_sector_rs_above_threshold_true(self):
        """Returns True when RS score exceeds the threshold."""
        from src.strategy.conditions_sector import sector_rs_above

        engine = _make_engine_with_rows(
            [(date(2026, 5, 8), 4.5, 2, 1)],
            active_count=9,
        )
        result = sector_rs_above("NIFTY_BANK", threshold=3.0, db_engine=engine)
        assert result is True

    def test_sector_rs_above_threshold_false_when_below(self):
        """Returns False when RS score is below the threshold."""
        from src.strategy.conditions_sector import sector_rs_above

        engine = _make_engine_with_rows(
            [(date(2026, 5, 8), 1.0, 5, 0)],
            active_count=9,
        )
        result = sector_rs_above("NIFTY_IT", threshold=2.0, db_engine=engine)
        assert result is False

    def test_sector_rs_above_threshold_false_when_equal(self):
        """Returns False when RS score equals the threshold (strict >)."""
        from src.strategy.conditions_sector import sector_rs_above

        engine = _make_engine_with_rows(
            [(date(2026, 5, 8), 2.0, 4, 0)],
            active_count=9,
        )
        result = sector_rs_above("NIFTY_FMCG", threshold=2.0, db_engine=engine)
        assert result is False  # strict greater-than

    def test_sector_rs_above_false_when_no_data(self):
        """Returns False when no DB row found."""
        from src.strategy.conditions_sector import sector_rs_above

        engine = _make_engine_with_rows([], active_count=9)
        result = sector_rs_above("NIFTY_METAL", threshold=0.0, db_engine=engine)
        assert result is False


# ---------------------------------------------------------------------------
# sector_rank_change_positive
# ---------------------------------------------------------------------------

class TestSectorRankChangePositive:
    """Tests for sector_rank_change_positive()."""

    def test_rank_change_positive_true_when_improved(self):
        """Returns True when rank_change > 0 (sector moved up in ranking)."""
        from src.strategy.conditions_sector import sector_rank_change_positive

        engine = _make_engine_with_rows(
            [(date(2026, 5, 8), 3.0, 2, 3)],  # rank_change=+3 → improved
            active_count=9,
        )
        result = sector_rank_change_positive("NIFTY_AUTO", db_engine=engine)
        assert result is True

    def test_rank_change_positive_false_when_worsened(self):
        """Returns False when rank_change < 0 (sector moved down)."""
        from src.strategy.conditions_sector import sector_rank_change_positive

        engine = _make_engine_with_rows(
            [(date(2026, 5, 8), -1.0, 7, -2)],  # rank_change=-2 → worsened
            active_count=9,
        )
        result = sector_rank_change_positive("NIFTY_REALTY", db_engine=engine)
        assert result is False

    def test_rank_change_positive_false_when_unchanged(self):
        """Returns False when rank_change == 0."""
        from src.strategy.conditions_sector import sector_rank_change_positive

        engine = _make_engine_with_rows(
            [(date(2026, 5, 8), 0.5, 5, 0)],
            active_count=9,
        )
        result = sector_rank_change_positive("NIFTY_PHARMA", db_engine=engine)
        assert result is False

    def test_rank_change_positive_false_when_none(self):
        """Returns False when rank_change is NULL (first day of data)."""
        from src.strategy.conditions_sector import sector_rank_change_positive

        engine = _make_engine_with_rows(
            [(date(2026, 5, 8), 2.0, 3, None)],  # rank_change=NULL
            active_count=9,
        )
        result = sector_rank_change_positive("NIFTY_ENERGY", db_engine=engine)
        assert result is False
