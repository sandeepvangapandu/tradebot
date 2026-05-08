"""Cold storage archiver — skeleton for Phase A full implementation.

Moves ticks and depth_snapshots older than N days from Postgres into
Parquet files under data/cold/. Idempotent: skips days already archived.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import psycopg
import psycopg.rows
import pyarrow as pa
import pyarrow.parquet as pq


def _get_local_db_url() -> str:
    """Return LOCAL_DB_URL from environment (libpq format)."""
    url = os.environ.get("LOCAL_DB_URL", "postgresql://localhost/tradebot")
    for prefix in ("postgresql+psycopg://", "postgres+psycopg://"):
        if url.startswith(prefix):
            url = "postgresql://" + url[len(prefix):]
    return url


class ColdStorageArchiver:
    """Archive hot Postgres data to Parquet files after retention window.

    Args:
        cold_dir: Base directory for cold storage parquet files.
    """

    def __init__(self, cold_dir: str = "data/cold") -> None:
        self.cold_dir = Path(cold_dir)
        self._db_url = _get_local_db_url()

    # ------------------------------------------------------------------
    # Ticks
    # ------------------------------------------------------------------

    def archive_ticks_older_than(self, days: int = 7) -> int:
        """Archive tick rows older than `days` days to Parquet, then delete.

        Parquet files are written to:
            {cold_dir}/ticks/<instrument_key>/<YYYY-MM-DD>.parquet

        Args:
            days: Retention window in days. Rows with ts < NOW()-days are archived.

        Returns:
            Total count of rows archived.
        """
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        total_archived = 0

        with psycopg.connect(self._db_url, autocommit=False, row_factory=psycopg.rows.dict_row) as conn:
            # Fetch distinct (instrument_key, date) pairs older than cutoff
            pairs = conn.execute(
                """
                SELECT DISTINCT instrument_key,
                       DATE(ts AT TIME ZONE 'UTC') AS tick_date
                FROM ticks
                WHERE ts < %s
                ORDER BY instrument_key, tick_date
                """,
                (cutoff,),
            ).fetchall()

            for pair in pairs:
                instrument_key: str = pair["instrument_key"]
                tick_date: Any = pair["tick_date"]

                parquet_path = (
                    self.cold_dir
                    / "ticks"
                    / instrument_key.replace("|", "_").replace("/", "_")
                    / f"{tick_date}.parquet"
                )

                if parquet_path.exists():
                    # Already archived — just delete residual rows from DB
                    deleted = conn.execute(
                        """
                        DELETE FROM ticks
                        WHERE instrument_key = %s
                          AND DATE(ts AT TIME ZONE 'UTC') = %s
                          AND ts < %s
                        """,
                        (instrument_key, tick_date, cutoff),
                    ).rowcount
                    conn.commit()
                    total_archived += deleted
                    continue

                # Fetch rows for this (symbol, date)
                rows = conn.execute(
                    """
                    SELECT id, instrument_key, ts, ltp, volume,
                           bid, ask, bid_qty, ask_qty,
                           total_buy_qty, total_sell_qty
                    FROM ticks
                    WHERE instrument_key = %s
                      AND DATE(ts AT TIME ZONE 'UTC') = %s
                      AND ts < %s
                    ORDER BY ts
                    """,
                    (instrument_key, tick_date, cutoff),
                ).fetchall()

                if not rows:
                    continue

                # Write to Parquet
                parquet_path.parent.mkdir(parents=True, exist_ok=True)
                table = pa.Table.from_pylist([dict(r) for r in rows])
                pq.write_table(table, parquet_path, compression="snappy")

                # Delete from DB
                ids = [r["id"] for r in rows]
                conn.execute(
                    "DELETE FROM ticks WHERE id = ANY(%s)",
                    (ids,),
                )
                conn.commit()
                total_archived += len(rows)

        return total_archived

    # ------------------------------------------------------------------
    # Depth snapshots
    # ------------------------------------------------------------------

    def archive_depth_older_than(self, days: int = 7) -> int:
        """Archive depth_snapshots rows older than `days` days to Parquet.

        Parquet files written to:
            {cold_dir}/depth/<instrument_key>/<YYYY-MM-DD>.parquet

        Args:
            days: Retention window in days.

        Returns:
            Total count of rows archived.
        """
        import json

        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        total_archived = 0

        with psycopg.connect(self._db_url, autocommit=False, row_factory=psycopg.rows.dict_row) as conn:
            pairs = conn.execute(
                """
                SELECT DISTINCT instrument_key,
                       DATE(ts AT TIME ZONE 'UTC') AS snap_date
                FROM depth_snapshots
                WHERE ts < %s
                ORDER BY instrument_key, snap_date
                """,
                (cutoff,),
            ).fetchall()

            for pair in pairs:
                instrument_key: str = pair["instrument_key"]
                snap_date: Any = pair["snap_date"]

                parquet_path = (
                    self.cold_dir
                    / "depth"
                    / instrument_key.replace("|", "_").replace("/", "_")
                    / f"{snap_date}.parquet"
                )

                if parquet_path.exists():
                    deleted = conn.execute(
                        """
                        DELETE FROM depth_snapshots
                        WHERE instrument_key = %s
                          AND DATE(ts AT TIME ZONE 'UTC') = %s
                          AND ts < %s
                        """,
                        (instrument_key, snap_date, cutoff),
                    ).rowcount
                    conn.commit()
                    total_archived += deleted
                    continue

                rows = conn.execute(
                    """
                    SELECT id, instrument_key, ts, bids, asks,
                           spread_bps, bid_wall_top5, ask_wall_top5
                    FROM depth_snapshots
                    WHERE instrument_key = %s
                      AND DATE(ts AT TIME ZONE 'UTC') = %s
                      AND ts < %s
                    ORDER BY ts
                    """,
                    (instrument_key, snap_date, cutoff),
                ).fetchall()

                if not rows:
                    continue

                # Serialise JSONB fields to strings for Parquet
                serialised = []
                for r in rows:
                    row_dict = dict(r)
                    row_dict["bids"] = json.dumps(row_dict["bids"])
                    row_dict["asks"] = json.dumps(row_dict["asks"])
                    serialised.append(row_dict)

                parquet_path.parent.mkdir(parents=True, exist_ok=True)
                table = pa.Table.from_pylist(serialised)
                pq.write_table(table, parquet_path, compression="snappy")

                ids = [r["id"] for r in rows]
                conn.execute(
                    "DELETE FROM depth_snapshots WHERE id = ANY(%s)",
                    (ids,),
                )
                conn.commit()
                total_archived += len(rows)

        return total_archived
