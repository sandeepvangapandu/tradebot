"""Database engine and session management (SQLAlchemy 2.0 style).

Provides lazy-initialised engine creation, a session factory, and a
context-manager helper that commits on success and rolls back on error.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Generator

from loguru import logger
from sqlalchemy import Engine, create_engine, inspect, text
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from src.persistence.models import Base

# ---------------------------------------------------------------------------
# Module-level singletons (lazy-initialised via ``init_db``)
# ---------------------------------------------------------------------------
engine: Engine | None = None
SessionLocal: sessionmaker[Session] | None = None


def init_db(database_url: str) -> Engine:
    """Create the SQLAlchemy engine, build all tables, and wire up the session factory.

    This function is idempotent: calling it a second time with the same URL
    is a no-op and returns the existing engine.

    Args:
        database_url: SQLAlchemy connection string, e.g.
            ``sqlite:///data/trading.db`` or
            ``postgresql+psycopg2://user:pass@host/db``.

    Returns:
        The newly created (or existing) ``Engine`` instance.
    """
    global engine, SessionLocal  # noqa: PLW0603

    if engine is not None:
        logger.debug("Database engine already initialised — returning existing engine.")
        return engine

    logger.info("Initialising database engine: {}", database_url)

    if database_url.startswith("sqlite"):
        engine = create_engine(
            database_url,
            echo=False,
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
    else:
        engine = create_engine(
            database_url,
            echo=False,
            pool_pre_ping=True,
        )

    Base.metadata.create_all(bind=engine)
    _run_sqlite_compat_migrations(engine)
    logger.info("All tables created / verified.")

    SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)

    return engine


def _run_sqlite_compat_migrations(db_engine: Engine) -> None:
    """Apply small SQLite migrations that ``create_all`` cannot perform.

    SQLite does not alter existing tables when ORM models gain columns. Keep
    this deliberately narrow and additive so live paper-trading data is not
    dropped during startup.
    """
    if db_engine.dialect.name != "sqlite":
        return

    inspector = inspect(db_engine)
    if not inspector.has_table("daily_pnl"):
        return

    columns = {col["name"] for col in inspector.get_columns("daily_pnl")}
    if "strategy" in columns:
        return

    with db_engine.begin() as conn:
        conn.execute(text("ALTER TABLE daily_pnl RENAME TO daily_pnl_legacy"))
        conn.execute(
            text(
                "CREATE TABLE daily_pnl ("
                "id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT, "
                "date DATE NOT NULL, "
                "strategy VARCHAR(128) NOT NULL DEFAULT '__total__', "
                "realized_pnl BIGINT NOT NULL, "
                "unrealized_pnl BIGINT NOT NULL, "
                "total_pnl BIGINT NOT NULL, "
                "trades_count INTEGER NOT NULL, "
                "win_count INTEGER NOT NULL, "
                "CONSTRAINT uq_daily_pnl_date_strategy UNIQUE (date, strategy)"
                ")"
            )
        )
        conn.execute(
            text(
                "INSERT INTO daily_pnl "
                "(id, date, strategy, realized_pnl, unrealized_pnl, total_pnl, trades_count, win_count) "
                "SELECT id, date, '__total__', realized_pnl, unrealized_pnl, total_pnl, trades_count, win_count "
                "FROM daily_pnl_legacy"
            )
        )
        conn.execute(text("DROP TABLE daily_pnl_legacy"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS ix_daily_pnl_strategy ON daily_pnl(strategy)"))
    logger.info("SQLite migration applied: daily_pnl rebuilt with per-strategy uniqueness.")


@contextmanager
def get_session() -> Generator[Session, None, None]:
    """Yield a transactional database session.

    Commits automatically when the block exits cleanly.  Rolls back and
    re-raises if an exception occurs.

    Yields:
        An active ``Session`` bound to the module-level engine.

    Raises:
        RuntimeError: If ``init_db`` has not been called yet.
    """
    if SessionLocal is None:
        raise RuntimeError(
            "Database not initialised. Call init_db(database_url) first."
        )

    session: Session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        logger.exception("Session rolled back due to exception.")
        raise
    finally:
        session.close()
