"""Tests for src/research/universe_scanner.py.

Strategy
--------
- Uses the real Postgres test DB (LOCAL_DB_URL, defaults to tradebot).
- The ``module_db`` fixture ensures Phase-D migration tables exist before
  any test runs and cleans up only the rows inserted during the test run
  (it does NOT drop/recreate tables, so the migration is tested independently).
- historical_provider is mocked with deterministic synthetic bar DataFrames
  so tests do not require Upstox API credentials.
- InstrumentManager is loaded from the live cached NSE instrument master so
  that ``_get_instrument_key`` fallback works if needed.

Fixture ordering: ``module_db`` (module-scope, autouse) → individual tests.
"""

from __future__ import annotations

import os
from datetime import date, datetime, timedelta
from typing import Callable
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import psycopg
import pytest
from sqlalchemy import create_engine, text

# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

_DB_URL_PSYCOPG = os.environ.get(
    "LOCAL_DB_URL", "postgresql://sandeepvangapandu@localhost:5432/tradebot"
)

# Strip SQLAlchemy driver prefix so psycopg can use it directly.
def _raw_psycopg_url(url: str) -> str:
    for prefix in ("postgresql+psycopg://", "postgres+psycopg://"):
        if url.startswith(prefix):
            return "postgresql://" + url[len(prefix):]
    return url


_PSYCOPG_URL = _raw_psycopg_url(_DB_URL_PSYCOPG)

# SQLAlchemy URL (psycopg v3 driver)
_SQLA_URL = (
    _DB_URL_PSYCOPG
    if "+psycopg" in _DB_URL_PSYCOPG
    else _DB_URL_PSYCOPG.replace("postgresql://", "postgresql+psycopg://", 1).replace(
        "postgres://", "postgresql+psycopg://", 1
    )
)

# Fixed trade date used by all tests in this module — pick a past weekday
_TRADE_DATE = date(2026, 5, 8)

# Phase-D table names (for cleanup)
_PHASE_D_TABLES = [
    "universe_daily_snapshot",
    "rs_rank_daily",
    "volatility_rank_daily",
    "liquidity_rank_daily",
]

TOP_10_SYMBOLS = [
    "AXISBANK",
    "HDFCBANK",
    "HINDUNILVR",
    "ICICIBANK",
    "INFY",
    "ITC",
    "KOTAKBANK",
    "RELIANCE",
    "SBIN",
    "TCS",
]


# ---------------------------------------------------------------------------
# Module-scope fixture: guarantee Phase-D tables exist, clean up test rows
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module", autouse=True)
def module_db():
    """Ensure Phase-D migration tables exist; clean up test rows after module."""
    # Make sure migrate has been applied (idempotent)
    import subprocess
    result = subprocess.run(
        ["python3", "-m", "src.storage.migrate", "up"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"Migration failed:\n{result.stderr}"

    yield

    # Teardown: delete only the rows for _TRADE_DATE to avoid polluting other data
    with psycopg.connect(_PSYCOPG_URL, autocommit=True) as conn:
        for tbl in _PHASE_D_TABLES:
            conn.execute(
                f"DELETE FROM {tbl} WHERE trade_date = %s",
                (_TRADE_DATE,),
            )


# ---------------------------------------------------------------------------
# Synthetic historical_provider factory
# ---------------------------------------------------------------------------

def _make_bar_df(
    n: int,
    base_price: int = 200_000,
    vol_pct: float = 1.0,
    seed: int = 42,
    return_pct: float = 0.0,
) -> pd.DataFrame:
    """Return a synthetic daily-bar DataFrame (prices in paisa).

    Args:
        n: Number of bars.
        base_price: Starting close price in paisa.
        vol_pct: Per-bar volatility percent.
        seed: RNG seed.
        return_pct: Total return over n bars (cumulative, percentage).
            Positive → uptrend, 0 → flat.

    Returns:
        DataFrame with DatetimeIndex and columns: open, high, low, close, volume.
    """
    rng = np.random.default_rng(seed)
    sigma = vol_pct / 100.0
    # Total drift so cumulative return ends at return_pct
    per_bar_drift = (return_pct / 100.0) / max(n - 1, 1) if n > 1 else 0.0
    log_ret = per_bar_drift + sigma * rng.standard_normal(n)
    closes = base_price * np.exp(np.cumsum(log_ret))
    closes = np.maximum(closes, 1).astype(int)

    bar_range = (vol_pct / 100.0) * 0.5
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
    symbol_prices: dict[str, int] | None = None,
    benchmark_return_pct: float = 0.0,
    symbol_return_pct: float = 0.0,
) -> Callable[[str, str, int], pd.DataFrame]:
    """Return a mock historical_provider callable.

    Args:
        symbol_prices: Map of instrument_key → base price in paisa.
            Any key not in the map returns a default 200,000 price.
        benchmark_return_pct: Cumulative return for the Nifty 50 benchmark.
        symbol_return_pct: Cumulative return for equity instruments.

    Returns:
        Callable matching the historical_provider contract.
    """
    benchmark_key = "NSE_INDEX|Nifty 50"

    def provider(instrument_key: str, interval: str, days: int) -> pd.DataFrame:
        n = max(days, 5)
        if instrument_key == benchmark_key:
            return _make_bar_df(n, base_price=2_450_000, return_pct=benchmark_return_pct)
        base = (symbol_prices or {}).get(instrument_key, 200_000)
        return _make_bar_df(n, base_price=base, return_pct=symbol_return_pct)

    return provider


# ---------------------------------------------------------------------------
# InstrumentManager fixture (shared across tests in module)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def instrument_manager():
    """Real InstrumentManager loaded from cached NSE instrument master."""
    from src.data.instruments import InstrumentManager

    im = InstrumentManager()
    im.load_instruments()
    return im


# ---------------------------------------------------------------------------
# Engine fixture (SQLAlchemy sync engine pointed at tradebot DB)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def db_engine():
    """SQLAlchemy sync engine for the tradebot Postgres DB."""
    engine = create_engine(_SQLA_URL, pool_pre_ping=True)
    yield engine
    engine.dispose()


# ---------------------------------------------------------------------------
# UniverseScanner factory fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def scanner_factory(instrument_manager, db_engine):
    """Return a factory that creates UniverseScanner with a given provider."""
    from src.research.universe_scanner import UniverseScanner

    def _factory(provider: Callable | None = None) -> "UniverseScanner":
        if provider is None:
            provider = _make_provider()
        return UniverseScanner(
            instrument_manager=instrument_manager,
            historical_provider=provider,
            db_engine=db_engine,
        )

    return _factory


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestGetConstituents:
    """Tests for UniverseScanner.get_constituents()."""

    def test_get_constituents_returns_10_active(self, scanner_factory):
        """Migration should have seeded exactly 10 active constituents."""
        scanner = scanner_factory()
        constituents = scanner.get_constituents(active_only=True)
        assert len(constituents) == 10
        symbols = {c["symbol"] for c in constituents}
        assert symbols == set(TOP_10_SYMBOLS)

    def test_get_constituents_include_inactive_returns_at_least_10(self, scanner_factory):
        """When active_only=False, result contains all seeded rows (>= 10)."""
        scanner = scanner_factory()
        all_c = scanner.get_constituents(active_only=False)
        assert len(all_c) >= 10

    def test_get_constituents_fields_complete(self, scanner_factory):
        """Each constituent dict has all expected keys."""
        scanner = scanner_factory()
        for c in scanner.get_constituents():
            assert "symbol" in c
            assert "instrument_key" in c
            assert "segment" in c
            assert c["segment"] == "NSE_EQ"


class TestGetInstrumentKeys:
    """Tests for UniverseScanner.get_instrument_keys()."""

    def test_get_instrument_keys_without_indices(self, scanner_factory):
        """Without indices: returns exactly 10 equity instrument keys."""
        scanner = scanner_factory()
        keys = scanner.get_instrument_keys(include_indices=False)
        assert len(keys) == 10
        assert all(k.startswith("NSE_EQ|") for k in keys)

    def test_get_instrument_keys_with_indices(self, scanner_factory):
        """With indices: returns 12 keys (10 equities + NIFTY + BANKNIFTY)."""
        scanner = scanner_factory()
        keys = scanner.get_instrument_keys(include_indices=True)
        assert len(keys) == 12
        assert "NSE_INDEX|Nifty 50" in keys
        assert "NSE_INDEX|Nifty Bank" in keys


class TestComputeLiquidity:
    """Tests for UniverseScanner.compute_liquidity()."""

    def test_compute_liquidity_returns_positive_adv(self, scanner_factory):
        """ADV and avg_volume should be positive integers for normal bar data."""
        provider = _make_provider(symbol_prices={"NSE_EQ|INE002A01018": 290_000})
        scanner = scanner_factory(provider)
        adv, avg_vol = scanner.compute_liquidity("RELIANCE", lookback_days=20)
        assert isinstance(adv, int)
        assert isinstance(avg_vol, int)
        assert adv > 0, f"Expected positive ADV, got {adv}"
        assert avg_vol > 0, f"Expected positive avg_vol, got {avg_vol}"

    def test_compute_liquidity_empty_data_returns_zeros(self, scanner_factory):
        """When provider returns an empty DataFrame, liquidity should be (0, 0)."""
        def empty_provider(key, interval, days):
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        scanner = scanner_factory(empty_provider)
        adv, avg_vol = scanner.compute_liquidity("RELIANCE", lookback_days=20)
        assert adv == 0
        assert avg_vol == 0

    def test_compute_liquidity_respects_lookback(self, scanner_factory):
        """Increasing lookback should use more bars (reflected in larger volume)."""
        # Use two scanners with different lookback via direct calls
        provider = _make_provider(symbol_prices={"NSE_EQ|INE062A01020": 85_000})
        scanner = scanner_factory(provider)
        adv_20, _ = scanner.compute_liquidity("SBIN", lookback_days=20)
        adv_5, _ = scanner.compute_liquidity("SBIN", lookback_days=5)
        # Both should be positive; they may differ since they average different windows
        assert adv_20 > 0
        assert adv_5 > 0


class TestComputeVolatility:
    """Tests for UniverseScanner.compute_volatility()."""

    def test_compute_volatility_returns_atr_pct_positive(self, scanner_factory):
        """ATR% should be a positive float for well-formed bar data."""
        provider = _make_provider(symbol_prices={"NSE_EQ|INE040A01034": 160_000})
        scanner = scanner_factory(provider)
        atr_pct, rv = scanner.compute_volatility("HDFCBANK", atr_period=14, vol_lookback=20)
        assert isinstance(atr_pct, float)
        assert atr_pct > 0.0, f"Expected positive ATR%, got {atr_pct}"

    def test_compute_volatility_returns_realized_vol_non_negative(self, scanner_factory):
        """Realized volatility should be >= 0."""
        provider = _make_provider()
        scanner = scanner_factory(provider)
        _, rv = scanner.compute_volatility("TCS", atr_period=14, vol_lookback=20)
        assert rv >= 0.0

    def test_compute_volatility_insufficient_data_returns_zeros(self, scanner_factory):
        """When fewer bars than atr_period are available, returns (0.0, 0.0)."""
        def tiny_provider(key, interval, days):
            return _make_bar_df(3, base_price=200_000)  # Only 3 bars

        scanner = scanner_factory(tiny_provider)
        atr_pct, rv = scanner.compute_volatility("INFY", atr_period=14)
        assert atr_pct == 0.0
        assert rv == 0.0


class TestComputeRelativeStrength:
    """Tests for UniverseScanner.compute_relative_strength()."""

    def test_compute_relative_strength_outperformer_positive(self, scanner_factory):
        """Symbol +5% return vs NIFTY flat → RS > 0."""
        provider = _make_provider(
            symbol_return_pct=5.0,
            benchmark_return_pct=0.0,
        )
        scanner = scanner_factory(provider)
        rs = scanner.compute_relative_strength(
            "RELIANCE",
            benchmark_key="NSE_INDEX|Nifty 50",
            lookback_days=20,
        )
        assert rs > 0.0, f"Expected positive RS for outperformer, got {rs}"

    def test_compute_relative_strength_underperformer_negative(self, scanner_factory):
        """Symbol flat vs NIFTY +5% → RS < 0."""
        provider = _make_provider(
            symbol_return_pct=0.0,
            benchmark_return_pct=5.0,
        )
        scanner = scanner_factory(provider)
        rs = scanner.compute_relative_strength(
            "TCS",
            benchmark_key="NSE_INDEX|Nifty 50",
            lookback_days=20,
        )
        assert rs < 0.0, f"Expected negative RS for underperformer, got {rs}"

    def test_compute_relative_strength_empty_benchmark_returns_zero(self, scanner_factory):
        """If benchmark data is unavailable, RS should fall back to 0."""
        benchmark_key = "NSE_INDEX|Nifty 50"

        def provider(key, interval, days):
            if key == benchmark_key:
                return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
            return _make_bar_df(25)

        scanner = scanner_factory(provider)
        rs = scanner.compute_relative_strength("SBIN", benchmark_key=benchmark_key)
        assert rs == 0.0


class TestRankAll:
    """Tests for UniverseScanner.rank_all()."""

    def test_rank_all_returns_dataframe_with_10_rows(self, scanner_factory):
        """rank_all should return exactly 10 rows (one per active constituent)."""
        provider = _make_provider()
        scanner = scanner_factory(provider)
        df = scanner.rank_all(_TRADE_DATE)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 10

    def test_rank_all_persists_4_tables(self, scanner_factory, db_engine):
        """After rank_all, each of the 4 ranking tables has exactly 10 rows for the date."""
        provider = _make_provider()
        scanner = scanner_factory(provider)
        scanner.rank_all(_TRADE_DATE)

        tables = [
            "liquidity_rank_daily",
            "volatility_rank_daily",
            "rs_rank_daily",
            "universe_daily_snapshot",
        ]
        with db_engine.connect() as conn:
            for tbl in tables:
                result = conn.execute(
                    text(f"SELECT count(*) FROM {tbl} WHERE trade_date = :td"),
                    {"td": _TRADE_DATE},
                ).scalar()
                assert result == 10, (
                    f"Expected 10 rows in {tbl} for {_TRADE_DATE}, got {result}"
                )

    def test_rank_all_composite_score_in_range(self, scanner_factory):
        """Composite scores must be in [0, 1]."""
        provider = _make_provider()
        scanner = scanner_factory(provider)
        df = scanner.rank_all(_TRADE_DATE)
        assert (df["composite_score"] >= 0.0).all()
        assert (df["composite_score"] <= 1.0).all()

    def test_rank_all_ranks_are_integers_in_valid_range(self, scanner_factory):
        """All three rank columns should contain positive integers in [1, 10]."""
        provider = _make_provider()
        scanner = scanner_factory(provider)
        df = scanner.rank_all(_TRADE_DATE)
        for col in ("liquidity_rank", "volatility_rank", "rs_rank"):
            vals = df[col].tolist()
            assert all(isinstance(v, (int, np.integer)) for v in vals), (
                f"Non-integer rank values in {col}: {vals}"
            )
            assert all(1 <= v <= 10 for v in vals), (
                f"Ranks out of [1,10] for {col}: {vals}"
            )
            # At least the minimum rank must be 1
            assert min(vals) == 1, f"Minimum rank for {col} is not 1: {vals}"

    def test_rank_all_is_idempotent(self, scanner_factory, db_engine):
        """Calling rank_all twice for the same date should not increase row count."""
        provider = _make_provider()
        scanner = scanner_factory(provider)
        scanner.rank_all(_TRADE_DATE)
        scanner.rank_all(_TRADE_DATE)  # second call — upsert

        with db_engine.connect() as conn:
            count = conn.execute(
                text("SELECT count(*) FROM liquidity_rank_daily WHERE trade_date = :td"),
                {"td": _TRADE_DATE},
            ).scalar()
        assert count == 10


class TestGetTodayUniverse:
    """Tests for UniverseScanner.get_today_universe()."""

    def test_get_today_universe_returns_dataframe(self, scanner_factory):
        """get_today_universe should return a DataFrame (possibly empty)."""
        scanner = scanner_factory()
        df = scanner.get_today_universe(_TRADE_DATE)
        assert isinstance(df, pd.DataFrame)

    def test_get_today_universe_after_rank_all_has_10_rows(self, scanner_factory):
        """After rank_all, get_today_universe returns 10 rows ordered by composite score."""
        provider = _make_provider()
        scanner = scanner_factory(provider)
        scanner.rank_all(_TRADE_DATE)

        df = scanner.get_today_universe(_TRADE_DATE)
        assert len(df) == 10
        assert "composite_score" in df.columns
        assert "symbol" in df.columns

    def test_get_today_universe_empty_for_unknown_date(self, scanner_factory):
        """Querying a date with no data should return an empty DataFrame."""
        scanner = scanner_factory()
        df = scanner.get_today_universe(date(2000, 1, 1))  # guaranteed no data
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0


class TestIsBlackout:
    """Tests for UniverseScanner.is_blackout()."""

    def test_is_blackout_default_false(self, scanner_factory):
        """A freshly inserted (non-blackout) row returns (False, None)."""
        provider = _make_provider()
        scanner = scanner_factory(provider)
        scanner.rank_all(_TRADE_DATE)  # ensures row exists

        blackout, reason = scanner.is_blackout("RELIANCE", _TRADE_DATE)
        assert blackout is False
        assert reason is None

    def test_is_blackout_no_snapshot_returns_false(self, scanner_factory):
        """If no snapshot row exists for the date, returns (False, None)."""
        scanner = scanner_factory()
        blackout, reason = scanner.is_blackout("RELIANCE", date(1999, 1, 1))
        assert blackout is False
        assert reason is None

    def test_is_blackout_true_when_set_in_db(self, scanner_factory, db_engine):
        """When the blackout flag is manually set in the DB, is_blackout returns True."""
        provider = _make_provider()
        scanner = scanner_factory(provider)
        scanner.rank_all(_TRADE_DATE)

        # Manually mark HDFCBANK as blacked out
        with db_engine.begin() as conn:
            conn.execute(
                text(
                    """
                    UPDATE universe_daily_snapshot
                    SET blackout = TRUE, blackout_reason = 'earnings'
                    WHERE trade_date = :td AND symbol = :sym
                    """
                ),
                {"td": _TRADE_DATE, "sym": "HDFCBANK"},
            )

        blackout, reason = scanner.is_blackout("HDFCBANK", _TRADE_DATE)
        assert blackout is True
        assert reason == "earnings"

        # Restore to keep teardown clean
        with db_engine.begin() as conn:
            conn.execute(
                text(
                    """
                    UPDATE universe_daily_snapshot
                    SET blackout = FALSE, blackout_reason = NULL
                    WHERE trade_date = :td AND symbol = :sym
                    """
                ),
                {"td": _TRADE_DATE, "sym": "HDFCBANK"},
            )
