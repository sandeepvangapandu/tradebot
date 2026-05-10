"""Tests verifying the Phase 0 database schema is correctly applied.

Uses a module-scoped fixture that:
1. Drops all phase0 tables.
2. Runs `python3 -m src.storage.migrate up`.
3. Verifies tables exist.
4. Tears down after the module completes.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
from datetime import datetime, timezone
from typing import Any

import psycopg
import pytest
from sqlalchemy import text


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DB_URL = os.environ.get("LOCAL_DB_URL", "postgresql://sandeepvangapandu@localhost:5432/tradebot")

PHASE0_TABLES = [
    "bot_config",
    "schema_version",
    "instruments",
    "bars",
    "ticks",
    "depth_snapshots",
    "orders",
    "fills",
    "positions",
    "daily_pnl",
    "signals",
    "bot_runs",
]


def _psycopg_url(url: str) -> str:
    """Strip SQLAlchemy driver prefix for psycopg."""
    for prefix in ("postgresql+psycopg://", "postgres+psycopg://"):
        if url.startswith(prefix):
            return "postgresql://" + url[len(prefix):]
    return url


def _conn() -> psycopg.Connection:
    return psycopg.connect(_psycopg_url(DB_URL), autocommit=True)


# ---------------------------------------------------------------------------
# Module-scoped fixture: setup + teardown
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module", autouse=True)
def setup_phase0_schema():
    """Drop and recreate phase0 tables, yield for tests, then drop again."""
    # Teardown helper
    def _drop_tables(conn: psycopg.Connection) -> None:
        # Drop in reverse dependency order
        drops = [
            "DROP TABLE IF EXISTS fills CASCADE",
            "DROP TABLE IF EXISTS orders CASCADE",
            "DROP TABLE IF EXISTS positions CASCADE",
            "DROP TABLE IF EXISTS daily_pnl CASCADE",
            "DROP TABLE IF EXISTS signals CASCADE",
            "DROP TABLE IF EXISTS bot_runs CASCADE",
            "DROP TABLE IF EXISTS ticks CASCADE",
            "DROP TABLE IF EXISTS depth_snapshots CASCADE",
            "DROP TABLE IF EXISTS bars CASCADE",
            "DROP TABLE IF EXISTS instruments CASCADE",
            "DROP TABLE IF EXISTS bot_config CASCADE",
            "DROP TABLE IF EXISTS schema_version CASCADE",
        ]
        for stmt in drops:
            conn.execute(stmt)

    # --- Setup ---
    with _conn() as conn:
        _drop_tables(conn)

    # Clear lru_cache so engines recreate with fresh env
    from src.storage.db import get_async_engine, get_sync_engine
    get_sync_engine.cache_clear()
    get_async_engine.cache_clear()

    result = subprocess.run(
        ["python3", "-m", "src.storage.migrate", "up"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"Migration failed: {result.stderr}"
    assert "Applied" in result.stdout or "already applied" in result.stdout.lower()

    yield

    # --- Teardown ---
    with _conn() as conn:
        _drop_tables(conn)

    # Clear caches again after teardown
    from src.storage.db import get_async_engine, get_sync_engine
    get_sync_engine.cache_clear()
    get_async_engine.cache_clear()


# ---------------------------------------------------------------------------
# Schema existence tests
# ---------------------------------------------------------------------------

class TestTablesExist:
    """All phase0 tables should exist after migration."""

    def test_all_tables_present(self) -> None:
        with _conn() as conn:
            rows = conn.execute(
                """
                SELECT tablename FROM pg_tables
                WHERE schemaname = 'public'
                """
            ).fetchall()
            existing = {r[0] for r in rows}

        for table in PHASE0_TABLES:
            assert table in existing, f"Table '{table}' not found after migration"


# ---------------------------------------------------------------------------
# Instruments
# ---------------------------------------------------------------------------

class TestInstruments:
    def test_insert_and_query_by_underlying(self) -> None:
        with _conn() as conn:
            conn.execute(
                """
                INSERT INTO instruments
                    (instrument_key, segment, trading_symbol, underlying_key, expiry, option_type)
                VALUES
                    ('NSE_FO|123', 'NFO', 'BANKNIFTY26MAY50000CE',
                     'NSE_INDEX|Nifty Bank', '2026-05-26', 'CE')
                ON CONFLICT DO NOTHING
                """
            )
            rows = conn.execute(
                """
                SELECT instrument_key FROM instruments
                WHERE underlying_key = 'NSE_INDEX|Nifty Bank'
                  AND option_type = 'CE'
                """
            ).fetchall()
        assert len(rows) >= 1
        assert rows[0][0] == "NSE_FO|123"


# ---------------------------------------------------------------------------
# Bars
# ---------------------------------------------------------------------------

class TestBars:
    def test_primary_key_uniqueness(self) -> None:
        """Inserting a duplicate (instrument_key, timeframe, ts) should fail."""
        ts = "2026-05-08 09:15:00+05:30"
        with _conn() as conn:
            conn.execute(
                """
                INSERT INTO bars (instrument_key, timeframe, ts, open, high, low, close)
                VALUES ('NSE_INDEX|Nifty Bank', '1m', %s, 4800000, 4810000, 4790000, 4805000)
                ON CONFLICT DO NOTHING
                """,
                (ts,),
            )
            # Second insert should silently ignore due to ON CONFLICT
            conn.execute(
                """
                INSERT INTO bars (instrument_key, timeframe, ts, open, high, low, close)
                VALUES ('NSE_INDEX|Nifty Bank', '1m', %s, 9999999, 9999999, 9999999, 9999999)
                ON CONFLICT DO NOTHING
                """,
                (ts,),
            )
            row = conn.execute(
                """
                SELECT open FROM bars
                WHERE instrument_key = 'NSE_INDEX|Nifty Bank'
                  AND timeframe = '1m' AND ts = %s
                """,
                (ts,),
            ).fetchone()
        # Original value preserved
        assert row is not None
        assert row[0] == 4800000


# ---------------------------------------------------------------------------
# Ticks
# ---------------------------------------------------------------------------

class TestTicks:
    def test_bulk_insert_100_rows(self) -> None:
        """bulk_insert_ticks should insert 100 rows and return correct count."""

        async def _run() -> int:
            from src.storage.db import bulk_insert_ticks, get_async_engine

            get_async_engine.cache_clear()
            ts_base = datetime(2026, 5, 8, 9, 15, 0, tzinfo=timezone.utc)
            rows = [
                {
                    "instrument_key": "NSE_INDEX|Nifty Bank",
                    "ts": ts_base.replace(second=i % 60, microsecond=i * 100),
                    "ltp": 4800000 + i,
                    "volume": i * 10,
                }
                for i in range(100)
            ]
            count = await bulk_insert_ticks(rows)
            return count

        count = asyncio.run(_run())
        assert count == 100


# ---------------------------------------------------------------------------
# Depth snapshots
# ---------------------------------------------------------------------------

class TestDepthSnapshots:
    def test_jsonb_bids_asks_roundtrip(self) -> None:
        """JSONB bids/asks should survive a round-trip insert+select."""
        bids = [{"price": 4800000, "qty": 100}, {"price": 4799500, "qty": 200}]
        asks = [{"price": 4800500, "qty": 150}, {"price": 4801000, "qty": 50}]

        with _conn() as conn:
            conn.execute(
                """
                INSERT INTO depth_snapshots
                    (instrument_key, ts, bids, asks)
                VALUES
                    ('NSE_INDEX|Nifty Bank', NOW(), %s::jsonb, %s::jsonb)
                """,
                (json.dumps(bids), json.dumps(asks)),
            )
            row = conn.execute(
                """
                SELECT bids, asks FROM depth_snapshots
                WHERE instrument_key = 'NSE_INDEX|Nifty Bank'
                ORDER BY id DESC LIMIT 1
                """
            ).fetchone()

        assert row is not None
        fetched_bids = row[0]
        fetched_asks = row[1]
        assert len(fetched_bids) == 2
        assert len(fetched_asks) == 2
        assert fetched_bids[0]["price"] == 4800000


# ---------------------------------------------------------------------------
# Orders → Fills FK
# ---------------------------------------------------------------------------

class TestOrdersFillsFK:
    def test_fill_references_order(self) -> None:
        """Fill with valid order_id should insert successfully."""
        with _conn() as conn:
            conn.execute(
                """
                INSERT INTO orders
                    (order_id, instrument_key, side, order_type, quantity, status)
                VALUES
                    ('TEST-ORD-001', 'NSE_INDEX|Nifty Bank', 'BUY', 'MARKET', 10, 'COMPLETE')
                ON CONFLICT DO NOTHING
                """
            )
            conn.execute(
                """
                INSERT INTO fills
                    (fill_id, order_id, instrument_key, side, quantity, price, filled_at)
                VALUES
                    ('TEST-FILL-001', 'TEST-ORD-001', 'NSE_INDEX|Nifty Bank',
                     'BUY', 10, 4800000, NOW())
                ON CONFLICT DO NOTHING
                """
            )
            row = conn.execute(
                "SELECT fill_id FROM fills WHERE order_id = 'TEST-ORD-001'"
            ).fetchone()
        assert row is not None
        assert row[0] == "TEST-FILL-001"

    def test_fill_with_invalid_order_id_raises(self) -> None:
        """Fill referencing non-existent order_id should raise FK violation."""
        with pytest.raises(Exception):
            with psycopg.connect(_psycopg_url(DB_URL), autocommit=True) as conn:
                conn.execute(
                    """
                    INSERT INTO fills
                        (fill_id, order_id, instrument_key, side, quantity, price, filled_at)
                    VALUES
                        ('BAD-FILL-999', 'DOES-NOT-EXIST', 'NSE_INDEX|Nifty Bank',
                         'BUY', 1, 100, NOW())
                    """
                )


# ---------------------------------------------------------------------------
# Signals
# ---------------------------------------------------------------------------

class TestSignals:
    def test_insert_signal_with_conditions_json(self) -> None:
        from src.storage.db import record_signal, get_sync_engine

        get_sync_engine.cache_clear()
        signal_id = record_signal(
            strategy_name="orb_vwap",
            instrument_key="NSE_FO|123",
            signal_type="BUY",
            conditions_met={"vwap_above": True, "rsi_ok": True},
            conditions_failed={"volume_ok": False},
        )
        assert isinstance(signal_id, int)
        assert signal_id > 0

        with _conn() as conn:
            row = conn.execute(
                "SELECT strategy_name, signal_type FROM signals WHERE id = %s",
                (signal_id,),
            ).fetchone()
        assert row is not None
        assert row[0] == "orb_vwap"
        assert row[1] == "BUY"


# ---------------------------------------------------------------------------
# Bot runs
# ---------------------------------------------------------------------------

class TestBotRuns:
    def test_start_end_roundtrip(self) -> None:
        from src.storage.db import get_sync_engine, record_bot_run_end, record_bot_run_start

        get_sync_engine.cache_clear()
        run_id = record_bot_run_start(
            branch="feature/bloomberg-data-execution",
            git_sha="abc123",
            mode="paper",
            capital_paisa=100_000_000,
        )
        assert isinstance(run_id, int)
        assert run_id > 0

        record_bot_run_end(run_id, exit_reason="test_complete")

        with _conn() as conn:
            row = conn.execute(
                "SELECT exit_reason, ended_at FROM bot_runs WHERE run_id = %s",
                (run_id,),
            ).fetchone()
        assert row is not None
        assert row[0] == "test_complete"
        assert row[1] is not None  # ended_at set


# ---------------------------------------------------------------------------
# Schema version
# ---------------------------------------------------------------------------

class TestSchemaVersion:
    def test_phase0_foundation_recorded(self) -> None:
        with _conn() as conn:
            row = conn.execute(
                "SELECT version FROM schema_version WHERE version = 'phase0_foundation'"
            ).fetchone()
        assert row is not None
        assert row[0] == "phase0_foundation"
