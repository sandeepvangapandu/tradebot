"""Tests for src/research/sector_rotation.py.

Strategy
--------
- Uses the real Postgres test DB (LOCAL_DB_URL, defaults to tradebot).
- The ``module_db`` fixture runs migrations (idempotent) and cleans up test
  rows after the module completes.
- ``historical_provider`` is mocked with deterministic synthetic bar DataFrames
  so tests do not require Upstox API credentials.
- ``InstrumentManager`` is mocked; the module does not rely on it for any
  critical computation at this phase.

All tests use ``_TRADE_DATE = date(2026, 5, 8)`` for reproducibility.
"""

from __future__ import annotations

import os
import subprocess
from datetime import date, datetime, timedelta
from typing import Callable
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import psycopg
import pytest
from sqlalchemy import create_engine, text

# ---------------------------------------------------------------------------
# DB connection helpers
# ---------------------------------------------------------------------------

_DB_URL_ENV = os.environ.get(
    "LOCAL_DB_URL", "postgresql://sandeepvangapandu@localhost:5432/tradebot"
)


def _raw_psycopg_url(url: str) -> str:
    """Strip SQLAlchemy driver prefix so psycopg can use it directly."""
    for prefix in ("postgresql+psycopg://", "postgres+psycopg://"):
        if url.startswith(prefix):
            return "postgresql://" + url[len(prefix):]
    return url


def _sqla_url(url: str) -> str:
    """Ensure URL has +psycopg driver suffix for SQLAlchemy."""
    if "+psycopg" in url:
        return url
    return url.replace("postgresql://", "postgresql+psycopg://", 1).replace(
        "postgres://", "postgresql+psycopg://", 1
    )


_PSYCOPG_URL = _raw_psycopg_url(_DB_URL_ENV)
_SQLA_URL = _sqla_url(_DB_URL_ENV)

_TRADE_DATE = date(2026, 5, 8)

_SECTOR_TABLES = ["sector_rank_daily"]


# ---------------------------------------------------------------------------
# Module-scope fixture: run migrations + teardown test rows
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module", autouse=True)
def module_db():
    """Ensure Phase-B1 migration tables exist; clean up test rows after module."""
    result = subprocess.run(
        ["python3", "-m", "src.storage.migrate", "up"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"Migration failed:\n{result.stderr}"

    yield

    with psycopg.connect(_PSYCOPG_URL, autocommit=True) as conn:
        for tbl in _SECTOR_TABLES:
            conn.execute(
                f"DELETE FROM {tbl} WHERE trade_date = %s",
                (_TRADE_DATE,),
            )


# ---------------------------------------------------------------------------
# Synthetic bar DataFrame factory
# ---------------------------------------------------------------------------

def _make_bar_df(
    n: int = 25,
    base_price: int = 2_450_000,
    return_pct: float = 0.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Return a synthetic daily-bar DataFrame (prices in paisa).

    Args:
        n: Number of bars.
        base_price: Starting close price in paisa.
        return_pct: Cumulative return over n bars (percentage).
        seed: RNG seed.

    Returns:
        DataFrame with DatetimeIndex and columns: open, high, low, close, volume.
    """
    rng = np.random.default_rng(seed)
    sigma = 0.01
    per_bar_drift = (return_pct / 100.0) / max(n - 1, 1) if n > 1 else 0.0
    log_ret = per_bar_drift + sigma * rng.standard_normal(n)
    closes = base_price * np.exp(np.cumsum(log_ret))
    closes = np.maximum(closes, 1).astype(int)

    bar_range = 0.005
    highs = (closes * (1 + np.abs(rng.normal(0, bar_range, n)))).astype(int)
    lows = (closes * (1 - np.abs(rng.normal(0, bar_range, n)))).astype(int)
    opens = np.roll(closes, 1)
    opens[0] = closes[0]
    opens = opens.astype(int)
    highs = np.maximum(highs, np.maximum(opens, closes))
    lows = np.minimum(lows, np.minimum(opens, closes))
    volumes = (1_000_000 * rng.lognormal(0, 0.5, n)).astype(int)

    idx = pd.date_range(end="2026-05-08", periods=n, freq="B", tz="Asia/Kolkata")
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": volumes},
        index=idx,
    )


def _make_provider(
    benchmark_return_pct: float = 2.0,
) -> Callable[[str, str, int], pd.DataFrame]:
    """Return a mock historical_provider that gives each sector a unique return.

    Assigns each sector a distinct return (ranging from +15% to -6%) so that
    ranks are all unique and deterministic. The benchmark always returns
    *benchmark_return_pct*.

    Args:
        benchmark_return_pct: Cumulative return for the NIFTY 50 benchmark.

    Returns:
        Callable matching the historical_provider contract.
    """
    benchmark_key = "NSE_INDEX|Nifty 50"

    # Fixed per-sector returns: cover a spread from +15% to -6% so all 9 ranks differ.
    _SECTOR_RETURNS: dict[str, float] = {
        "NSE_INDEX|Nifty Bank":         15.0,
        "NSE_INDEX|Nifty IT":           12.0,
        "NSE_INDEX|Nifty Auto":         10.0,
        "NSE_INDEX|Nifty Pharma":        8.0,
        "NSE_INDEX|Nifty Metal":         5.0,
        "NSE_INDEX|Nifty FMCG":          3.0,
        "NSE_INDEX|Nifty Energy":        1.0,
        "NSE_INDEX|Nifty Realty":       -2.0,
        "NSE_INDEX|Nifty Fin Service":  -6.0,
    }

    def provider(instrument_key: str, interval: str, days: int) -> pd.DataFrame:
        n = max(days, 5)
        if instrument_key == benchmark_key:
            return _make_bar_df(n, base_price=2_450_000, return_pct=benchmark_return_pct, seed=99)
        ret = _SECTOR_RETURNS.get(instrument_key, 0.0)
        seed = abs(hash(instrument_key)) % 1000
        return _make_bar_df(n, base_price=5_200_000, return_pct=ret, seed=seed)

    return provider


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def db_engine():
    """SQLAlchemy sync engine for the tradebot Postgres DB."""
    engine = create_engine(_SQLA_URL, pool_pre_ping=True)
    yield engine
    engine.dispose()


@pytest.fixture(scope="module")
def analyzer_factory(db_engine):
    """Return a factory that creates SectorRotationAnalyzer with a given provider."""
    from src.research.sector_rotation import SectorRotationAnalyzer

    def _factory(provider: Callable | None = None) -> "SectorRotationAnalyzer":
        im = MagicMock()
        if provider is None:
            provider = _make_provider()
        return SectorRotationAnalyzer(
            instrument_manager=im,
            historical_provider=provider,
            db_engine=db_engine,
        )

    return _factory


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestGetActiveSectors:
    """Tests for SectorRotationAnalyzer.get_active_sectors()."""

    def test_get_active_sectors_returns_seeded_9(self, analyzer_factory):
        """get_active_sectors() returns exactly 9 rows (all seeded by migration)."""
        analyzer = analyzer_factory()
        sectors = analyzer.get_active_sectors()
        assert len(sectors) == 9, (
            f"Expected 9 active sectors, got {len(sectors)}: "
            f"{[s['symbol'] for s in sectors]}"
        )

    def test_get_active_sectors_contains_expected_symbols(self, analyzer_factory):
        """All 9 expected sector symbols are present."""
        expected = {
            "NIFTY_BANK", "NIFTY_IT", "NIFTY_AUTO", "NIFTY_PHARMA",
            "NIFTY_METAL", "NIFTY_FMCG", "NIFTY_ENERGY", "NIFTY_REALTY",
            "NIFTY_FIN_SERVICE",
        }
        analyzer = analyzer_factory()
        symbols = {s["symbol"] for s in analyzer.get_active_sectors()}
        assert symbols == expected

    def test_get_sector_instrument_keys_returns_9(self, analyzer_factory):
        """get_sector_instrument_keys() returns 9 instrument keys."""
        analyzer = analyzer_factory()
        keys = analyzer.get_sector_instrument_keys()
        assert len(keys) == 9
        for k in keys:
            assert k.startswith("NSE_INDEX|"), f"Expected NSE_INDEX prefix, got: {k}"


class TestComputeReturns:
    """Tests for SectorRotationAnalyzer.compute_returns()."""

    def test_compute_returns_positive_when_price_up(self):
        """compute_returns() is positive when the closing price trended up.

        Uses strictly monotonically increasing prices to guarantee a positive
        return independent of GBM noise.
        """
        from src.research.sector_rotation import SectorRotationAnalyzer

        key = "NSE_INDEX|Nifty Bank"

        def provider(instrument_key: str, interval: str, days: int) -> pd.DataFrame:
            n = max(days, 5)
            closes = np.linspace(5_200_000, 5_720_000, n).astype(int)  # +10%
            idx = pd.date_range(end="2026-05-08", periods=n, freq="B", tz="Asia/Kolkata")
            return pd.DataFrame(
                {
                    "open": closes,
                    "high": closes + 100,
                    "low": closes - 100,
                    "close": closes,
                    "volume": np.ones(n, dtype=int) * 1_000_000,
                },
                index=idx,
            )

        analyzer = SectorRotationAnalyzer(
            instrument_manager=MagicMock(),
            historical_provider=provider,
            db_engine=MagicMock(),
        )
        ret = analyzer.compute_returns(key, lookback_days=20)
        assert ret > 0, f"Expected positive return, got {ret}"

    def test_compute_returns_zero_on_empty_df(self):
        """compute_returns() returns 0.0 when provider returns empty DataFrame."""
        from src.research.sector_rotation import SectorRotationAnalyzer

        def provider(instrument_key: str, interval: str, days: int) -> pd.DataFrame:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        analyzer = SectorRotationAnalyzer(
            instrument_manager=MagicMock(),
            historical_provider=provider,
            db_engine=MagicMock(),
        )
        ret = analyzer.compute_returns("NSE_INDEX|Nifty Bank", lookback_days=20)
        assert ret == 0.0


class TestComputeRsScore:
    """Tests for SectorRotationAnalyzer.compute_rs_score()."""

    def test_compute_rs_score_positive_when_outperforms(self):
        """RS score is positive when the sector outperforms the benchmark.

        Uses monotonically increasing prices to guarantee the sign of the return
        without relying on GBM noise.
        """
        from src.research.sector_rotation import SectorRotationAnalyzer

        sector_key = "NSE_INDEX|Nifty Bank"
        benchmark_key = "NSE_INDEX|Nifty 50"

        def _monotone_df(n: int, start: int, end: int) -> pd.DataFrame:
            """Return a DataFrame with linearly rising close prices."""
            closes = np.linspace(start, end, n).astype(int)
            idx = pd.date_range(end="2026-05-08", periods=n, freq="B", tz="Asia/Kolkata")
            return pd.DataFrame(
                {
                    "open": closes,
                    "high": closes + 100,
                    "low": closes - 100,
                    "close": closes,
                    "volume": np.ones(n, dtype=int) * 1_000_000,
                },
                index=idx,
            )

        def provider(instrument_key: str, interval: str, days: int) -> pd.DataFrame:
            n = max(days, 5)
            if instrument_key == benchmark_key:
                # Benchmark: +2% (e.g. 2_450_000 → 2_499_000)
                return _monotone_df(n, 2_450_000, 2_499_000)
            # Sector: +10% (clearly outperforms benchmark's +2%)
            return _monotone_df(n, 5_200_000, 5_720_000)

        analyzer = SectorRotationAnalyzer(
            instrument_manager=MagicMock(),
            historical_provider=provider,
            db_engine=MagicMock(),
        )
        rs = analyzer.compute_rs_score(sector_key, benchmark_key=benchmark_key, lookback=20)
        assert rs > 0, f"Expected positive RS score, got {rs}"

    def test_compute_rs_score_negative_when_underperforms(self):
        """RS score is negative when the sector underperforms the benchmark.

        Uses monotonically priced DataFrames to guarantee sign without GBM noise.
        """
        from src.research.sector_rotation import SectorRotationAnalyzer

        sector_key = "NSE_INDEX|Nifty Realty"
        benchmark_key = "NSE_INDEX|Nifty 50"

        def _monotone_df(n: int, start: int, end: int) -> pd.DataFrame:
            closes = np.linspace(start, end, n).astype(int)
            idx = pd.date_range(end="2026-05-08", periods=n, freq="B", tz="Asia/Kolkata")
            return pd.DataFrame(
                {
                    "open": closes,
                    "high": closes + 100,
                    "low": closes - 100,
                    "close": closes,
                    "volume": np.ones(n, dtype=int) * 1_000_000,
                },
                index=idx,
            )

        def provider(instrument_key: str, interval: str, days: int) -> pd.DataFrame:
            n = max(days, 5)
            if instrument_key == benchmark_key:
                # Benchmark: +8%
                return _monotone_df(n, 2_450_000, 2_646_000)
            # Sector: +1% — clearly underperforms
            return _monotone_df(n, 500_000, 505_000)

        analyzer = SectorRotationAnalyzer(
            instrument_manager=MagicMock(),
            historical_provider=provider,
            db_engine=MagicMock(),
        )
        rs = analyzer.compute_rs_score(sector_key, benchmark_key=benchmark_key, lookback=20)
        assert rs < 0, f"Expected negative RS score, got {rs}"


class TestRankAll:
    """Tests for SectorRotationAnalyzer.rank_all() — requires DB."""

    def test_rank_all_persists_to_db(self, analyzer_factory, db_engine):
        """rank_all() writes one row per sector to sector_rank_daily."""
        analyzer = analyzer_factory(provider=_make_provider())
        df = analyzer.rank_all(_TRADE_DATE)

        # Returns a DataFrame
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 9

        # Verify DB rows
        with db_engine.connect() as conn:
            count = conn.execute(
                text("SELECT COUNT(*) FROM sector_rank_daily WHERE trade_date = :td"),
                {"td": _TRADE_DATE},
            ).scalar()
        assert count == 9, f"Expected 9 DB rows, got {count}"

    def test_rank_all_ranks_are_1_through_n(self, analyzer_factory):
        """rank_all() produces consecutive integer ranks 1..n."""
        analyzer = analyzer_factory(provider=_make_provider())
        df = analyzer.rank_all(_TRADE_DATE)
        ranks = sorted(df["rank"].tolist())
        assert ranks == list(range(1, len(df) + 1)), f"Non-consecutive ranks: {ranks}"

    def test_get_today_ranking_returns_same_data(self, analyzer_factory, db_engine):
        """get_today_ranking() reads back the rows persisted by rank_all()."""
        analyzer = analyzer_factory(provider=_make_provider())
        # Ensure rows are written (may duplicate from previous test — upsert handles it)
        analyzer.rank_all(_TRADE_DATE)
        df = analyzer.get_today_ranking(_TRADE_DATE)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 9
        # Should be sorted by rank ascending
        ranks = df["rank"].tolist()
        assert ranks == sorted(ranks), "get_today_ranking() should return rows sorted by rank ASC"


class TestGetSectorForStock:
    """Tests for SectorRotationAnalyzer.get_sector_for_stock()."""

    def test_get_sector_for_stock_returns_correct_mapping(self):
        """get_sector_for_stock() maps known stocks to correct sectors."""
        from src.research.sector_rotation import SectorRotationAnalyzer

        analyzer = SectorRotationAnalyzer(
            instrument_manager=MagicMock(),
            historical_provider=lambda *a: pd.DataFrame(),
            db_engine=MagicMock(),
        )
        assert analyzer.get_sector_for_stock("HDFCBANK") == "NIFTY_BANK"
        assert analyzer.get_sector_for_stock("ICICIBANK") == "NIFTY_BANK"
        assert analyzer.get_sector_for_stock("AXISBANK") == "NIFTY_BANK"
        assert analyzer.get_sector_for_stock("KOTAKBANK") == "NIFTY_BANK"
        assert analyzer.get_sector_for_stock("SBIN") == "NIFTY_BANK"
        assert analyzer.get_sector_for_stock("TCS") == "NIFTY_IT"
        assert analyzer.get_sector_for_stock("INFY") == "NIFTY_IT"
        assert analyzer.get_sector_for_stock("ITC") == "NIFTY_FMCG"
        assert analyzer.get_sector_for_stock("HUL") == "NIFTY_FMCG"
        assert analyzer.get_sector_for_stock("RELIANCE") == "NIFTY_ENERGY"

    def test_get_sector_for_stock_returns_none_for_unknown(self):
        """get_sector_for_stock() returns None for unknown symbols."""
        from src.research.sector_rotation import SectorRotationAnalyzer

        analyzer = SectorRotationAnalyzer(
            instrument_manager=MagicMock(),
            historical_provider=lambda *a: pd.DataFrame(),
            db_engine=MagicMock(),
        )
        result = analyzer.get_sector_for_stock("UNKNOWN_STOCK_XYZ")
        assert result is None


class TestIsTopQuartile:
    """Tests for SectorRotationAnalyzer.is_top_quartile()."""

    def test_is_top_quartile_true_for_rank_1_or_2(self, analyzer_factory, db_engine):
        """is_top_quartile() returns True when the sector has rank 1 or 2."""
        # First establish ranking data
        analyzer = analyzer_factory(provider=_make_provider())
        df = analyzer.rank_all(_TRADE_DATE)

        # Find the symbol with rank 1
        rank1_sym = df[df["rank"] == 1].iloc[0]["sector_symbol"]
        rank2_sym = df[df["rank"] == 2].iloc[0]["sector_symbol"]

        # With 9 sectors, ceil(9/4) = 3, so ranks 1, 2, 3 are top quartile
        assert analyzer.is_top_quartile(rank1_sym, _TRADE_DATE) is True, (
            f"Rank-1 sector {rank1_sym} should be top quartile"
        )
        assert analyzer.is_top_quartile(rank2_sym, _TRADE_DATE) is True, (
            f"Rank-2 sector {rank2_sym} should be top quartile"
        )

    def test_is_top_quartile_false_for_last_rank(self, analyzer_factory, db_engine):
        """is_top_quartile() returns False for the lowest-ranked sector."""
        analyzer = analyzer_factory(provider=_make_provider())
        df = analyzer.rank_all(_TRADE_DATE)
        last_sym = df[df["rank"] == df["rank"].max()].iloc[0]["sector_symbol"]

        # With 9 sectors cutoff is 3 — rank 9 is definitely not top quartile
        assert analyzer.is_top_quartile(last_sym, _TRADE_DATE) is False, (
            f"Last-rank sector {last_sym} should NOT be top quartile"
        )

    def test_is_top_quartile_false_for_empty_db(self, analyzer_factory, db_engine):
        """is_top_quartile() returns False when no ranking data exists for date."""
        analyzer = analyzer_factory()
        future_date = date(2030, 1, 1)
        result = analyzer.is_top_quartile("NIFTY_BANK", future_date)
        assert result is False
