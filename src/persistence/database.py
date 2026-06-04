"""Database engine and session management (SQLAlchemy 2.0 style).

Provides lazy-initialised engine creation, a session factory, and a
context-manager helper that commits on success and rolls back on error.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Generator

from loguru import logger
from sqlalchemy import Engine, create_engine
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
    logger.info("All tables created / verified.")

    SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)

    return engine


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
