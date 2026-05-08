"""Tests for ColdStorageArchiver — tick and depth archiving to Parquet."""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import psycopg
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DB_URL = os.environ.get("LOCAL_DB_URL", "postgresql://sandeepvangapandu@localhost:5432/tradebot")


def _psycopg_url(url: str) -> str:
    for prefix in ("postgresql+psycopg://", "postgres+psycopg://"):
        if url.startswith(prefix):
            return "postgresql://" + url[len(prefix):]
    return url


def _conn() -> psycopg.Connection:
    return psycopg.connect(_psycopg_url(DB_URL), autocommit=True)


SYMBOL = "NSE_INDEX|Nifty50_ArchTest"


# ---------------------------------------------------------------------------
# Module fixture: ensure schema exists, clean up test rows
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module", autouse=True)
def ensure_schema():
    """Ensure phase0 tables exist (run migrate up if needed)."""
    import subprocess

    result = subprocess.run(
        ["python3", "-m", "src.storage.migrate", "up"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"migrate up failed: {result.stderr}"

    yield

    # Cleanup test rows
    with _conn() as conn:
        conn.execute(
            "DELETE FROM ticks WHERE instrument_key = %s", (SYMBOL,)
        )
        conn.execute(
            "DELETE FROM depth_snapshots WHERE instrument_key = %s", (SYMBOL,)
        )


# ---------------------------------------------------------------------------
# Tick archiver tests
# ---------------------------------------------------------------------------

class TestTickArchiver:
    """ColdStorageArchiver.archive_ticks_older_than() tests."""

    def _insert_mixed_ticks(self, conn: psycopg.Connection) -> tuple[list[int], list[int]]:
        """Insert 50 old ticks + 25 fresh ticks. Returns (old_ids, new_ids)."""
        now = datetime.now(timezone.utc)
        old_ids: list[int] = []
        new_ids: list[int] = []

        for i in range(50):
            ts = now - timedelta(days=10, seconds=i)
            row = conn.execute(
                """
                INSERT INTO ticks (instrument_key, ts, ltp, volume)
                VALUES (%s, %s, %s, %s)
                RETURNING id
                """,
                (SYMBOL, ts, 4800000 + i, i * 10),
            ).fetchone()
            old_ids.append(row[0])

        for i in range(25):
            ts = now - timedelta(hours=1, seconds=i)  # fresh — within 7d
            row = conn.execute(
                """
                INSERT INTO ticks (instrument_key, ts, ltp, volume)
                VALUES (%s, %s, %s, %s)
                RETURNING id
                """,
                (SYMBOL, ts, 5000000 + i, i * 5),
            ).fetchone()
            new_ids.append(row[0])

        return old_ids, new_ids

    def test_archive_ticks_creates_parquet_and_removes_old_rows(
        self, tmp_path: Path
    ) -> None:
        from src.storage.archiver import ColdStorageArchiver

        # Clear engine cache before test
        from src.storage.db import get_async_engine, get_sync_engine
        get_sync_engine.cache_clear()
        get_async_engine.cache_clear()

        # Clean slate for this symbol
        with _conn() as conn:
            conn.execute("DELETE FROM ticks WHERE instrument_key = %s", (SYMBOL,))
            old_ids, new_ids = self._insert_mixed_ticks(conn)

        archiver = ColdStorageArchiver(cold_dir=str(tmp_path))
        archived_count = archiver.archive_ticks_older_than(days=7)

        # Should have archived all old rows
        assert archived_count == 50

        # Parquet file should exist
        parquet_files = list((tmp_path / "ticks").rglob("*.parquet"))
        assert len(parquet_files) >= 1

        # Verify file contains rows
        import pyarrow.parquet as _pq

        for pf in parquet_files:
            table = _pq.read_table(pf)
            assert table.num_rows > 0

        # Old rows removed from DB
        with _conn() as conn:
            remaining_old = conn.execute(
                "SELECT COUNT(*) FROM ticks WHERE id = ANY(%s)",
                (old_ids,),
            ).fetchone()[0]
            remaining_new = conn.execute(
                "SELECT COUNT(*) FROM ticks WHERE id = ANY(%s)",
                (new_ids,),
            ).fetchone()[0]

        assert remaining_old == 0
        assert remaining_new == 25

    def test_archive_is_idempotent(self, tmp_path: Path) -> None:
        """Running archiver twice should not error or double-delete."""
        from src.storage.archiver import ColdStorageArchiver
        from src.storage.db import get_async_engine, get_sync_engine
        get_sync_engine.cache_clear()
        get_async_engine.cache_clear()

        with _conn() as conn:
            conn.execute("DELETE FROM ticks WHERE instrument_key = %s", (SYMBOL,))
            old_ids, _ = self._insert_mixed_ticks(conn)

        archiver = ColdStorageArchiver(cold_dir=str(tmp_path))
        first = archiver.archive_ticks_older_than(days=7)
        second = archiver.archive_ticks_older_than(days=7)  # should not fail

        assert first == 50
        # Second run: parquet already exists, deletes residual rows (0 remaining)
        assert second == 0


# ---------------------------------------------------------------------------
# Depth archiver tests
# ---------------------------------------------------------------------------

class TestDepthArchiver:
    """ColdStorageArchiver.archive_depth_older_than() tests."""

    def _insert_mixed_depth(self, conn: psycopg.Connection) -> tuple[list[int], list[int]]:
        """Insert 30 old + 10 fresh depth snapshots."""
        now = datetime.now(timezone.utc)
        bids = json.dumps([{"price": 4800000, "qty": 100}])
        asks = json.dumps([{"price": 4801000, "qty": 50}])
        old_ids: list[int] = []
        new_ids: list[int] = []

        for i in range(30):
            ts = now - timedelta(days=8, seconds=i)
            row = conn.execute(
                """
                INSERT INTO depth_snapshots
                    (instrument_key, ts, bids, asks)
                VALUES (%s, %s, %s::jsonb, %s::jsonb)
                RETURNING id
                """,
                (SYMBOL, ts, bids, asks),
            ).fetchone()
            old_ids.append(row[0])

        for i in range(10):
            ts = now - timedelta(hours=2, seconds=i)
            row = conn.execute(
                """
                INSERT INTO depth_snapshots
                    (instrument_key, ts, bids, asks)
                VALUES (%s, %s, %s::jsonb, %s::jsonb)
                RETURNING id
                """,
                (SYMBOL, ts, bids, asks),
            ).fetchone()
            new_ids.append(row[0])

        return old_ids, new_ids

    def test_archive_depth_creates_parquet_and_removes_old_rows(
        self, tmp_path: Path
    ) -> None:
        from src.storage.archiver import ColdStorageArchiver
        from src.storage.db import get_async_engine, get_sync_engine
        get_sync_engine.cache_clear()
        get_async_engine.cache_clear()

        with _conn() as conn:
            conn.execute(
                "DELETE FROM depth_snapshots WHERE instrument_key = %s", (SYMBOL,)
            )
            old_ids, new_ids = self._insert_mixed_depth(conn)

        archiver = ColdStorageArchiver(cold_dir=str(tmp_path))
        archived_count = archiver.archive_depth_older_than(days=7)

        assert archived_count == 30

        parquet_files = list((tmp_path / "depth").rglob("*.parquet"))
        assert len(parquet_files) >= 1

        import pyarrow.parquet as _pq2

        for pf in parquet_files:
            table = _pq2.read_table(pf)
            assert table.num_rows > 0

        with _conn() as conn:
            remaining_old = conn.execute(
                "SELECT COUNT(*) FROM depth_snapshots WHERE id = ANY(%s)",
                (old_ids,),
            ).fetchone()[0]
            remaining_new = conn.execute(
                "SELECT COUNT(*) FROM depth_snapshots WHERE id = ANY(%s)",
                (new_ids,),
            ).fetchone()[0]

        assert remaining_old == 0
        assert remaining_new == 10
