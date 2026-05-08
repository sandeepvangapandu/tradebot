"""Plain-SQL migration runner (no Alembic).

Usage:
    python -m src.storage.migrate up      # apply all pending migrations
    python -m src.storage.migrate status  # list migrations and applied state
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import psycopg


def _get_local_db_url() -> str:
    """Return LOCAL_DB_URL from environment.

    Returns:
        PostgreSQL DSN string (libpq format, not SQLAlchemy URL).
    """
    url = os.environ.get("LOCAL_DB_URL", "postgresql://localhost/tradebot")
    # psycopg uses libpq format — strip the driver prefix if present
    for prefix in ("postgresql+psycopg://", "postgres+psycopg://"):
        if url.startswith(prefix):
            url = "postgresql://" + url[len(prefix):]
    return url


def _migrations_dir() -> Path:
    """Return path to supabase/migrations/ relative to repo root."""
    repo_root = Path(__file__).resolve().parents[2]
    return repo_root / "supabase" / "migrations"


def _ensure_schema_version_table(conn: psycopg.Connection) -> None:
    """Create schema_version table if it doesn't exist."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_version (
            version TEXT PRIMARY KEY,
            applied_at TIMESTAMPTZ DEFAULT NOW(),
            description TEXT
        )
        """
    )
    conn.commit()


def _get_applied_versions(conn: psycopg.Connection) -> set[str]:
    """Return set of version strings already recorded in schema_version."""
    rows = conn.execute("SELECT version FROM schema_version").fetchall()
    return {r[0] for r in rows}


def _apply_migration(conn: psycopg.Connection, sql_file: Path) -> None:
    """Execute a migration file and record it in schema_version.

    The version key is the filename without extension.
    Recording uses ON CONFLICT DO NOTHING so it is idempotent.

    Args:
        conn: Open psycopg connection.
        sql_file: Path to the .sql migration file.
    """
    version = sql_file.stem
    sql = sql_file.read_text(encoding="utf-8")
    conn.execute(sql)
    conn.execute(
        "INSERT INTO schema_version (version) VALUES (%s) ON CONFLICT (version) DO NOTHING",
        (version,),
    )
    conn.commit()
    print(f"  Applied: {sql_file.name}")


def cmd_up() -> None:
    """Apply all pending migrations in alphabetical order."""
    mig_dir = _migrations_dir()
    if not mig_dir.exists():
        print(f"ERROR: migrations directory not found: {mig_dir}")
        sys.exit(1)

    sql_files = sorted(mig_dir.glob("*.sql"))
    if not sql_files:
        print("No migration files found.")
        return

    url = _get_local_db_url()
    with psycopg.connect(url, autocommit=False) as conn:
        _ensure_schema_version_table(conn)
        applied = _get_applied_versions(conn)

        pending = [f for f in sql_files if f.stem not in applied]
        if not pending:
            print("All migrations already applied.")
            return

        for sql_file in pending:
            _apply_migration(conn, sql_file)

    print(f"Done. Applied {len(pending)} migration(s).")


def cmd_status() -> None:
    """List all migration files and whether each has been applied."""
    mig_dir = _migrations_dir()
    if not mig_dir.exists():
        print(f"ERROR: migrations directory not found: {mig_dir}")
        sys.exit(1)

    sql_files = sorted(mig_dir.glob("*.sql"))
    if not sql_files:
        print("No migration files found.")
        return

    url = _get_local_db_url()
    with psycopg.connect(url, autocommit=False) as conn:
        _ensure_schema_version_table(conn)
        applied = _get_applied_versions(conn)

    print(f"{'FILE':<55} {'STATUS'}")
    print("-" * 65)
    for sql_file in sql_files:
        status = "applied" if sql_file.stem in applied else "pending"
        print(f"{sql_file.name:<55} {status}")


def main() -> None:
    """CLI entry point."""
    if len(sys.argv) < 2 or sys.argv[1] not in ("up", "status"):
        print("Usage: python -m src.storage.migrate [up|status]")
        sys.exit(1)

    command = sys.argv[1]
    if command == "up":
        cmd_up()
    elif command == "status":
        cmd_status()


if __name__ == "__main__":
    main()
