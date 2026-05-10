"""Tests for src.indicators.volume_profile.

All tests use synthetic OHLCV DataFrames with controlled volume distributions
so that POC/VAH/VAL locations are deterministic and easy to assert.

No external DB or network required; DB persistence tests use SQLite in-memory.
"""

from __future__ import annotations

import json
from datetime import date
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.indicators.volume_profile import (
    VolumeProfileBuilder,
    compute_volume_profile,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BIN = 50  # 50 paisa bin (default: 5 * 10)


def _bar(high_p: int, low_p: int, volume: int) -> dict:
    """Return a single bar dict in paisa."""
    return {"open": low_p, "high": high_p, "low": low_p, "close": high_p, "volume": volume}


def _bars(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# compute_volume_profile — basic shape tests
# ---------------------------------------------------------------------------


class TestComputeVolumeProfileUniformBars:
    """Uniform volume across evenly spaced bars — POC should be near the middle."""

    def test_poc_at_median(self):
        """With equal volume in every price bucket, POC lands at one of the
        middle bins (tie broken by dict ordering — first maximum found)."""
        rows = [
            _bar(high_p=10_050, low_p=10_000, volume=100),
            _bar(high_p=10_100, low_p=10_050, volume=100),
            _bar(high_p=10_150, low_p=10_100, volume=100),
            _bar(high_p=10_200, low_p=10_150, volume=100),
            _bar(high_p=10_250, low_p=10_200, volume=100),
        ]
        df = _bars(rows)
        profile = compute_volume_profile(df, bin_size_paisa=50)
        assert profile is not None
        # POC must be one of the known price bins
        known_bins = {10_000, 10_050, 10_100, 10_150, 10_200}
        assert profile["poc_paisa"] in known_bins

    def test_vah_above_poc_val_below(self):
        """VAH must always be >= POC; VAL must always be <= POC."""
        rows = [
            _bar(high_p=10_050, low_p=10_000, volume=100),
            _bar(high_p=10_100, low_p=10_050, volume=200),
            _bar(high_p=10_150, low_p=10_100, volume=100),
        ]
        df = _bars(rows)
        profile = compute_volume_profile(df, bin_size_paisa=50)
        assert profile is not None
        assert profile["vah_paisa"] >= profile["poc_paisa"]
        assert profile["val_paisa"] <= profile["poc_paisa"]

    def test_bins_sorted_ascending(self):
        """Bins list must be sorted by price (ascending)."""
        rows = [
            _bar(high_p=10_050, low_p=10_000, volume=100),
            _bar(high_p=10_150, low_p=10_100, volume=100),
            _bar(high_p=10_250, low_p=10_200, volume=100),
        ]
        df = _bars(rows)
        profile = compute_volume_profile(df, bin_size_paisa=50)
        prices = [b["price"] for b in profile["bins"]]
        assert prices == sorted(prices)

    def test_total_volume_matches_input(self):
        """total_volume must equal the sum of input bar volumes."""
        rows = [
            _bar(10_050, 10_000, 300),
            _bar(10_100, 10_050, 500),
            _bar(10_150, 10_100, 200),
        ]
        df = _bars(rows)
        profile = compute_volume_profile(df, bin_size_paisa=50)
        assert profile["total_volume"] == 1000


class TestComputeVolumeProfileConcentratedVolume:
    """One bin dominates — POC must land exactly at that bin."""

    def test_poc_at_concentration(self):
        """90% of volume concentrated in a single 50-paisa bucket."""
        rows = [
            _bar(10_050, 10_000, 100),   # small
            _bar(10_100, 10_050, 9_000),  # dominant
            _bar(10_150, 10_100, 100),   # small
        ]
        df = _bars(rows)
        profile = compute_volume_profile(df, bin_size_paisa=50)
        assert profile is not None
        # Dominant bar covers 10_050; its bin lower edge is 10_050
        assert profile["poc_paisa"] == 10_050

    def test_vah_above_poc_val_below_concentrated(self):
        rows = [
            _bar(10_050, 10_000, 100),
            _bar(10_100, 10_050, 9_000),
            _bar(10_150, 10_100, 100),
        ]
        df = _bars(rows)
        profile = compute_volume_profile(df, bin_size_paisa=50)
        assert profile["vah_paisa"] >= profile["poc_paisa"]
        assert profile["val_paisa"] <= profile["poc_paisa"]


class TestComputeVolumeProfileValueArea:
    """Value area coverage tests."""

    def test_value_area_contains_70_pct_volume(self):
        """The volume of all bins whose lower edge is in [VAL, POC ... VAH-bin_size]
        should collectively be >= 70% of total volume.

        We verify by summing the bins that fall within [VAL, VAH).
        """
        rows = [
            _bar(10_050, 10_000, 500),
            _bar(10_100, 10_050, 3_000),  # POC
            _bar(10_150, 10_100, 500),
            _bar(10_200, 10_150, 300),
            _bar(10_250, 10_200, 200),
        ]
        df = _bars(rows)
        profile = compute_volume_profile(df, bin_size_paisa=50, value_area_pct=0.70)
        assert profile is not None

        val = profile["val_paisa"]
        vah = profile["vah_paisa"]
        total = profile["total_volume"]

        # Sum volumes of bins whose lower price edge falls in [val, vah]
        va_vol = sum(
            b["volume"] for b in profile["bins"]
            if val <= b["price"] <= vah
        )
        assert va_vol / total >= 0.70

    def test_single_bin_covers_100_pct(self):
        """All volume in exactly one bin → value area spans only that bin.

        A bar where high == low falls entirely within one price bucket, so
        both POC, VAH, and VAL derive from that single bin.
        """
        # Use high == low so the bar maps to exactly one bin boundary
        rows = [{"open": 10_050, "high": 10_050, "low": 10_050, "close": 10_050, "volume": 5_000}]
        df = _bars(rows)
        profile = compute_volume_profile(df, bin_size_paisa=50)
        assert profile is not None
        # Single bin: POC == VAL (lower edge)
        assert profile["poc_paisa"] == profile["val_paisa"]
        # VAH == lower edge + bin_size - 1 (upper edge of the single bin)
        assert profile["vah_paisa"] == profile["poc_paisa"] + 50 - 1
        assert len(profile["bins"]) == 1


class TestComputeVolumeProfileEdgeCases:
    """Edge case handling."""

    def test_handles_empty_bars(self):
        """Empty DataFrame must return None without raising."""
        df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        assert compute_volume_profile(df) is None

    def test_handles_none_bars(self):
        """None input must return None without raising."""
        assert compute_volume_profile(None) is None  # type: ignore[arg-type]

    def test_handles_single_bar(self):
        """Single bar must produce a valid profile."""
        df = _bars([_bar(10_100, 10_000, 1_000)])
        profile = compute_volume_profile(df, bin_size_paisa=50)
        assert profile is not None
        assert profile["poc_paisa"] is not None
        assert profile["vah_paisa"] >= profile["poc_paisa"]
        assert profile["val_paisa"] <= profile["poc_paisa"]
        assert profile["total_volume"] == 1_000

    def test_handles_zero_volume_bar(self):
        """Bars with zero total volume must return None."""
        df = _bars([
            _bar(10_100, 10_000, 0),
            _bar(10_200, 10_100, 0),
        ])
        assert compute_volume_profile(df) is None

    def test_rupee_prices_auto_converted(self):
        """Prices expressed as rupees (< 10 000) should be auto-converted to paisa."""
        # Prices in rupees: high=101.00, low=100.00 → paisa: 10_100 / 10_000
        rows = [{"open": 100.0, "high": 101.0, "low": 100.0, "close": 101.0, "volume": 1_000}]
        df = pd.DataFrame(rows)
        profile = compute_volume_profile(df, bin_size_paisa=50)
        assert profile is not None
        # POC must be in paisa scale (>= 10_000 paisa)
        assert profile["poc_paisa"] >= 10_000

    def test_returns_bin_size_in_profile(self):
        """Returned profile must echo back the bin_size_paisa used."""
        df = _bars([_bar(10_100, 10_000, 500)])
        profile = compute_volume_profile(df, bin_size_paisa=100)
        assert profile["bin_size_paisa"] == 100

    def test_missing_required_columns_raises(self):
        """Missing 'volume' column must raise ValueError."""
        df = pd.DataFrame({"open": [100], "high": [101], "low": [100], "close": [101]})
        with pytest.raises(ValueError, match="volume"):
            compute_volume_profile(df)


# ---------------------------------------------------------------------------
# VolumeProfileBuilder — session persistence (SQLite in-memory)
# ---------------------------------------------------------------------------


def _make_sqlite_engine():
    """Create a fresh SQLite in-memory engine with the required table."""
    from sqlalchemy import create_engine, text

    engine = create_engine("sqlite:///:memory:", future=True)
    with engine.begin() as conn:
        conn.execute(text(
            """
            CREATE TABLE volume_profile_daily (
                instrument_key TEXT NOT NULL,
                trade_date TEXT NOT NULL,
                poc_paisa INTEGER NOT NULL,
                vah_paisa INTEGER NOT NULL,
                val_paisa INTEGER NOT NULL,
                total_volume INTEGER NOT NULL,
                bin_size_paisa INTEGER NOT NULL,
                profile_bins TEXT,
                PRIMARY KEY (instrument_key, trade_date)
            )
            """
        ))
    return engine


class TestVolumeProfileBuilderPersistence:
    """Tests for VolumeProfileBuilder.compute_session and get_profile."""

    def _sample_bars(self) -> pd.DataFrame:
        return _bars([
            _bar(10_050, 10_000, 500),
            _bar(10_100, 10_050, 3_000),
            _bar(10_150, 10_100, 400),
        ])

    def test_compute_session_persists_to_db(self):
        """compute_session should write a row to volume_profile_daily."""
        from sqlalchemy import text

        engine = _make_sqlite_engine()
        builder = VolumeProfileBuilder(db_engine=engine)
        td = date(2026, 5, 8)

        profile = builder.compute_session("NSE_EQ|TEST", self._sample_bars(), td)
        assert profile is not None

        with engine.connect() as conn:
            row = conn.execute(
                text("SELECT poc_paisa, vah_paisa, val_paisa FROM volume_profile_daily "
                     "WHERE instrument_key = :k AND trade_date = :d"),
                {"k": "NSE_EQ|TEST", "d": "2026-05-08"},
            ).fetchone()

        assert row is not None
        assert row[0] == profile["poc_paisa"]
        assert row[1] == profile["vah_paisa"]
        assert row[2] == profile["val_paisa"]

    def test_get_profile_roundtrip(self):
        """Profile stored by compute_session must be retrievable via get_profile."""
        engine = _make_sqlite_engine()
        builder = VolumeProfileBuilder(db_engine=engine)
        td = date(2026, 5, 8)

        stored = builder.compute_session("NSE_EQ|RTTRIP", self._sample_bars(), td)
        fetched = builder.get_profile("NSE_EQ|RTTRIP", td)

        assert fetched is not None
        assert fetched["poc_paisa"] == stored["poc_paisa"]
        assert fetched["vah_paisa"] == stored["vah_paisa"]
        assert fetched["val_paisa"] == stored["val_paisa"]
        assert fetched["total_volume"] == stored["total_volume"]
        assert fetched["bin_size_paisa"] == stored["bin_size_paisa"]

    def test_get_profile_returns_none_for_missing(self):
        """get_profile must return None when no row exists for that date."""
        engine = _make_sqlite_engine()
        builder = VolumeProfileBuilder(db_engine=engine)
        result = builder.get_profile("NSE_EQ|MISSING", date(2026, 1, 1))
        assert result is None

    def test_compute_session_string_date(self):
        """compute_session must accept ISO date strings as well as date objects."""
        engine = _make_sqlite_engine()
        builder = VolumeProfileBuilder(db_engine=engine)
        profile = builder.compute_session("NSE_EQ|STRDATE", self._sample_bars(), "2026-05-08")
        assert profile is not None
        fetched = builder.get_profile("NSE_EQ|STRDATE", "2026-05-08")
        assert fetched is not None
        assert fetched["poc_paisa"] == profile["poc_paisa"]

    def test_compute_session_no_engine_does_not_raise(self):
        """compute_session without a db_engine returns profile without error."""
        builder = VolumeProfileBuilder(db_engine=None)
        profile = builder.compute_session("NSE_EQ|NODB", self._sample_bars(), date(2026, 5, 8))
        assert profile is not None
        assert "poc_paisa" in profile

    def test_compute_today_intraday_returns_profile(self):
        """compute_today_intraday must return profile without persisting."""
        engine = _make_sqlite_engine()
        builder = VolumeProfileBuilder(db_engine=engine)
        profile = builder.compute_today_intraday("NSE_EQ|LIVE", self._sample_bars())
        assert profile is not None
        # Nothing should be in DB since compute_today_intraday does not persist
        from sqlalchemy import text
        with engine.connect() as conn:
            count = conn.execute(
                text("SELECT COUNT(*) FROM volume_profile_daily")
            ).scalar()
        assert count == 0

    def test_upsert_on_conflict_updates_existing(self):
        """A second compute_session for same key+date should overwrite, not duplicate."""
        from sqlalchemy import text

        engine = _make_sqlite_engine()
        builder = VolumeProfileBuilder(db_engine=engine)
        td = date(2026, 5, 8)

        builder.compute_session("NSE_EQ|UPSERT", self._sample_bars(), td)
        # Re-compute with different bars
        new_bars = _bars([_bar(10_300, 10_250, 8_000)])
        builder.compute_session("NSE_EQ|UPSERT", new_bars, td)

        with engine.connect() as conn:
            count = conn.execute(
                text("SELECT COUNT(*) FROM volume_profile_daily WHERE instrument_key='NSE_EQ|UPSERT'")
            ).scalar()
        assert count == 1
