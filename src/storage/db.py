"""Database connection management for the trading bot storage layer.

Provides sync and async SQLAlchemy engines selected by ENV environment variable.
ENV=dev → LOCAL_DB_URL (default)
ENV=prod or anything else → SUPABASE_DB_URL
"""

from __future__ import annotations

import functools
import os
from datetime import datetime, timezone
from typing import Any

import psycopg
from sqlalchemy import create_engine, text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
from sqlalchemy.pool import QueuePool

# ---------------------------------------------------------------------------
# Connection string selection
# ---------------------------------------------------------------------------

def _get_db_url(async_driver: bool = False) -> str:
    """Return the appropriate DB URL based on ENV.

    Args:
        async_driver: If True, rewrite scheme to use asyncpg driver.

    Returns:
        Connection URL string.
    """
    env = os.environ.get("ENV", "dev")
    if env == "dev":
        url = os.environ.get("LOCAL_DB_URL", "postgresql://localhost/tradebot")
    else:
        url = os.environ.get("SUPABASE_DB_URL", "")
        if not url:
            raise RuntimeError("SUPABASE_DB_URL env var is required when ENV != dev")

    if async_driver:
        # Replace postgresql:// or postgres:// with postgresql+asyncpg://
        url = url.replace("postgresql://", "postgresql+asyncpg://", 1)
        url = url.replace("postgres://", "postgresql+asyncpg://", 1)
    else:
        # Ensure sync driver uses psycopg (v3)
        if url.startswith("postgresql://") or url.startswith("postgres://"):
            url = url.replace("postgresql://", "postgresql+psycopg://", 1)
            url = url.replace("postgres://", "postgresql+psycopg://", 1)

    return url


# ---------------------------------------------------------------------------
# Engine factories (singletons via lru_cache)
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=1)
def get_sync_engine():
    """Return a singleton sync SQLAlchemy engine.

    Pool config: pool_size=10, max_overflow=5.
    Suitable for low-frequency operations (orders, positions, signals).

    Returns:
        SQLAlchemy Engine instance.
    """
    url = _get_db_url(async_driver=False)
    return create_engine(
        url,
        poolclass=QueuePool,
        pool_size=10,
        max_overflow=5,
        pool_pre_ping=True,
        echo=False,
    )


@functools.lru_cache(maxsize=1)
def get_async_engine() -> AsyncEngine:
    """Return a singleton async SQLAlchemy engine using asyncpg.

    Pool config: min_size=5, max_size=20.
    Suitable for high-frequency operations (ticks, depth snapshots).

    Returns:
        SQLAlchemy AsyncEngine instance.
    """
    url = _get_db_url(async_driver=True)
    return create_async_engine(
        url,
        pool_size=5,
        max_overflow=15,  # min=5, max=20 total
        pool_pre_ping=True,
        echo=False,
    )


# ---------------------------------------------------------------------------
# Bulk insert helpers
# ---------------------------------------------------------------------------

async def bulk_insert_ticks(rows: list[dict[str, Any]]) -> int:
    """Insert tick rows using ON CONFLICT DO NOTHING.

    Uses RETURNING id to accurately count inserted rows because asyncpg
    returns rowcount=-1 for executemany INSERT ... ON CONFLICT DO NOTHING.

    Args:
        rows: List of dicts with keys matching `ticks` table columns.
              Required: instrument_key, ts, ltp.

    Returns:
        Number of rows actually inserted.
    """
    if not rows:
        return 0

    engine = get_async_engine()
    insert_sql = text(
        """
        INSERT INTO ticks
            (instrument_key, ts, ltp, volume, bid, ask, bid_qty, ask_qty,
             total_buy_qty, total_sell_qty)
        VALUES
            (:instrument_key, :ts, :ltp, :volume, :bid, :ask, :bid_qty, :ask_qty,
             :total_buy_qty, :total_sell_qty)
        ON CONFLICT DO NOTHING
        """
    )
    count_sql = text("SELECT COUNT(*) FROM ticks WHERE id > :max_id_before")

    # Normalise rows — fill missing optional fields with None
    defaults: dict[str, Any] = {
        "volume": 0,
        "bid": None,
        "ask": None,
        "bid_qty": None,
        "ask_qty": None,
        "total_buy_qty": None,
        "total_sell_qty": None,
    }
    normalised = [{**defaults, **r} for r in rows]

    async with engine.begin() as conn:
        # Get max id before insert to count inserted rows accurately
        max_id_row = await conn.execute(text("SELECT COALESCE(MAX(id), 0) FROM ticks"))
        max_id_before = max_id_row.scalar()
        await conn.execute(insert_sql, normalised)
        count_row = await conn.execute(count_sql, {"max_id_before": max_id_before})
        return count_row.scalar() or 0


async def bulk_insert_depth(rows: list[dict[str, Any]]) -> int:
    """Insert depth_snapshots rows using ON CONFLICT DO NOTHING.

    Args:
        rows: List of dicts with keys matching `depth_snapshots` table columns.
              Required: instrument_key, ts, bids (JSONB), asks (JSONB).

    Returns:
        Number of rows actually inserted.
    """
    if not rows:
        return 0

    engine = get_async_engine()
    sql = text(
        """
        INSERT INTO depth_snapshots
            (instrument_key, ts, bids, asks, spread_bps, bid_wall_top5, ask_wall_top5)
        VALUES
            (:instrument_key, :ts, CAST(:bids AS jsonb), CAST(:asks AS jsonb),
             :spread_bps, :bid_wall_top5, :ask_wall_top5)
        ON CONFLICT DO NOTHING
        RETURNING id
        """
    )
    import json

    defaults: dict[str, Any] = {
        "spread_bps": None,
        "bid_wall_top5": None,
        "ask_wall_top5": None,
    }
    normalised = []
    for r in rows:
        row = {**defaults, **r}
        # Ensure JSONB fields are JSON strings if passed as dicts
        if isinstance(row.get("bids"), (dict, list)):
            row["bids"] = json.dumps(row["bids"])
        if isinstance(row.get("asks"), (dict, list)):
            row["asks"] = json.dumps(row["asks"])
        normalised.append(row)

    async with engine.begin() as conn:
        result = await conn.execute(sql, normalised)
        return len(result.fetchall())


# ---------------------------------------------------------------------------
# Bot run helpers (sync — low frequency)
# ---------------------------------------------------------------------------

def record_bot_run_start(
    branch: str | None,
    git_sha: str | None,
    mode: str,
    capital_paisa: int,
) -> int:
    """Insert a bot_runs row and return the generated run_id.

    Args:
        branch: Git branch name.
        git_sha: Full git commit SHA.
        mode: Trading mode (paper/live).
        capital_paisa: Starting capital in paisa.

    Returns:
        run_id (BIGSERIAL primary key).
    """
    engine = get_sync_engine()
    sql = text(
        """
        INSERT INTO bot_runs (branch, git_sha, mode, capital_paisa)
        VALUES (:branch, :git_sha, :mode, :capital_paisa)
        RETURNING run_id
        """
    )
    with engine.begin() as conn:
        result = conn.execute(
            sql,
            {
                "branch": branch,
                "git_sha": git_sha,
                "mode": mode,
                "capital_paisa": capital_paisa,
            },
        )
        row = result.fetchone()
        return int(row[0])


def record_bot_run_end(run_id: int, exit_reason: str) -> None:
    """Update bot_runs row with ended_at and exit_reason.

    Args:
        run_id: The run_id returned by record_bot_run_start.
        exit_reason: Human-readable reason for shutdown.
    """
    engine = get_sync_engine()
    sql = text(
        """
        UPDATE bot_runs
        SET ended_at = NOW(), exit_reason = :exit_reason
        WHERE run_id = :run_id
        """
    )
    with engine.begin() as conn:
        conn.execute(sql, {"run_id": run_id, "exit_reason": exit_reason})


# ---------------------------------------------------------------------------
# Signal helper (sync)
# ---------------------------------------------------------------------------

def record_signal(
    strategy_name: str,
    instrument_key: str,
    signal_type: str,
    conditions_met: dict[str, Any] | None = None,
    conditions_failed: dict[str, Any] | None = None,
    rejection_reason: str | None = None,
) -> int:
    """Insert a signal row and return generated id.

    Args:
        strategy_name: Name of the strategy generating the signal.
        instrument_key: Upstox instrument key.
        signal_type: e.g. 'BUY', 'SELL', 'HOLD'.
        conditions_met: JSON-serialisable dict of passing conditions.
        conditions_failed: JSON-serialisable dict of failing conditions.
        rejection_reason: Optional rejection message.

    Returns:
        Signal id (BIGSERIAL primary key).
    """
    import json

    engine = get_sync_engine()
    sql = text(
        """
        INSERT INTO signals
            (strategy_name, instrument_key, signal_type,
             conditions_met, conditions_failed, rejection_reason)
        VALUES
            (:strategy_name, :instrument_key, :signal_type,
             CAST(:conditions_met AS jsonb), CAST(:conditions_failed AS jsonb),
             :rejection_reason)
        RETURNING id
        """
    )
    with engine.begin() as conn:
        result = conn.execute(
            sql,
            {
                "strategy_name": strategy_name,
                "instrument_key": instrument_key,
                "signal_type": signal_type,
                "conditions_met": json.dumps(conditions_met) if conditions_met is not None else None,
                "conditions_failed": json.dumps(conditions_failed) if conditions_failed is not None else None,
                "rejection_reason": rejection_reason,
            },
        )
        row = result.fetchone()
        return int(row[0])
