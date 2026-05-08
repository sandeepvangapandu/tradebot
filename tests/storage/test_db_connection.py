"""Tests for database connection routing and engine creation."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest
import pytest_asyncio


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _reset_engine_cache() -> None:
    """Clear lru_cache on engine factories between tests."""
    from src.storage import db as storage_db

    storage_db.get_sync_engine.cache_clear()
    storage_db.get_async_engine.cache_clear()


# ---------------------------------------------------------------------------
# Sync engine
# ---------------------------------------------------------------------------

class TestSyncEngine:
    """Sync engine connectivity tests."""

    def test_sync_engine_connects_and_pings(self) -> None:
        """Sync engine should connect to LOCAL_DB_URL and execute SELECT 1."""
        _reset_engine_cache()
        from sqlalchemy import text

        from src.storage.db import get_sync_engine

        engine = get_sync_engine()
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1")).scalar()
        assert result == 1

    def test_sync_engine_is_singleton(self) -> None:
        """get_sync_engine() should return the same object on repeated calls."""
        _reset_engine_cache()
        from src.storage.db import get_sync_engine

        e1 = get_sync_engine()
        e2 = get_sync_engine()
        assert e1 is e2


# ---------------------------------------------------------------------------
# Async engine
# ---------------------------------------------------------------------------

class TestAsyncEngine:
    """Async engine connectivity tests."""

    @pytest.mark.asyncio
    async def test_async_engine_connects_and_pings(self) -> None:
        """Async engine should connect to LOCAL_DB_URL and execute SELECT 1."""
        _reset_engine_cache()
        from sqlalchemy import text

        from src.storage.db import get_async_engine

        engine = get_async_engine()
        async with engine.connect() as conn:
            result = await conn.execute(text("SELECT 1"))
            value = result.scalar()
        assert value == 1

    @pytest.mark.asyncio
    async def test_async_engine_is_singleton(self) -> None:
        """get_async_engine() should return the same object on repeated calls."""
        _reset_engine_cache()
        from src.storage.db import get_async_engine

        e1 = get_async_engine()
        e2 = get_async_engine()
        assert e1 is e2


# ---------------------------------------------------------------------------
# ENV routing
# ---------------------------------------------------------------------------

class TestEnvRouting:
    """ENV variable should control which DB URL is selected."""

    def test_env_dev_routes_to_local_db_url(self) -> None:
        """ENV=dev should cause _get_db_url() to return LOCAL_DB_URL (sync variant)."""
        _reset_engine_cache()
        local_url = "postgresql://sandeepvangapandu@localhost:5432/tradebot"

        with patch.dict(os.environ, {"ENV": "dev", "LOCAL_DB_URL": local_url}):
            from src.storage.db import _get_db_url

            result = _get_db_url(async_driver=False)

        # The URL should include the local host and database name
        assert "localhost" in result
        assert "tradebot" in result

    def test_env_dev_async_driver_uses_asyncpg_scheme(self) -> None:
        """ENV=dev async_driver=True should rewrite scheme to postgresql+asyncpg."""
        _reset_engine_cache()
        local_url = "postgresql://sandeepvangapandu@localhost:5432/tradebot"

        with patch.dict(os.environ, {"ENV": "dev", "LOCAL_DB_URL": local_url}):
            from src.storage.db import _get_db_url

            result = _get_db_url(async_driver=True)

        assert result.startswith("postgresql+asyncpg://")

    def test_env_prod_raises_without_supabase_url(self) -> None:
        """ENV=prod without SUPABASE_DB_URL set should raise RuntimeError."""
        _reset_engine_cache()

        # Build a minimal env with ENV=prod and no SUPABASE_DB_URL
        minimal_env = {k: v for k, v in os.environ.items() if k != "SUPABASE_DB_URL"}
        minimal_env["ENV"] = "prod"

        with patch.dict(os.environ, minimal_env, clear=True):
            from src.storage.db import _get_db_url

            with pytest.raises(RuntimeError, match="SUPABASE_DB_URL"):
                _get_db_url(async_driver=False)
