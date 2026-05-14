"""Tests for src.research.kronos_validator.KronosValidator.

All tests use mock DB engines — no real Postgres or SQLite schema required.
"""
from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any
from unittest.mock import MagicMock, patch, call

import pytest

from src.research.kronos_validator import (
    AccuracyMetrics,
    KronosValidator,
    THRESHOLD_DROP,
    THRESHOLD_KEEP,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_row(
    instrument_key: str = "NSE_INDEX|Nifty 50",
    horizon_bars: int = 12,
    predicted_direction: str = "UP",
    actual_direction: str = "UP",
    predicted_close_paisa: int = 2_200_000,
    actual_close: float = 2_200_000.0,
    predicted_high_paisa: int = 2_210_000,
    predicted_low_paisa: int = 2_190_000,
    actual_high: float = 2_210_000.0,
    actual_low: float = 2_190_000.0,
) -> dict:
    """Return a minimal prediction-outcome row dict."""
    return {
        "forecast_ts": datetime(2026, 5, 13, 9, 30, 0, tzinfo=timezone.utc),
        "instrument_key": instrument_key,
        "horizon_bars": horizon_bars,
        "timeframe": "5m",
        "predicted_close_paisa": predicted_close_paisa,
        "predicted_high_paisa": predicted_high_paisa,
        "predicted_low_paisa": predicted_low_paisa,
        "predicted_direction": predicted_direction,
        "predicted_change_pct": 0.5,
        "actual_close": actual_close,
        "actual_high": actual_high,
        "actual_low": actual_low,
        "actual_direction": actual_direction,
    }


def _make_validator(engine=None) -> KronosValidator:
    return KronosValidator(db_engine=engine)


# ---------------------------------------------------------------------------
# compute_metrics_for_instrument
# ---------------------------------------------------------------------------


class TestComputeMetricsPerfect:
    """100% direction accuracy when every prediction matches actual direction."""

    def test_compute_metrics_perfect_predictions_returns_100_pct(self):
        rows = [
            _make_row(predicted_direction="UP", actual_direction="UP"),
            _make_row(predicted_direction="DOWN", actual_direction="DOWN"),
            _make_row(predicted_direction="UP", actual_direction="UP"),
            _make_row(predicted_direction="DOWN", actual_direction="DOWN"),
        ]
        v = _make_validator()
        m = v.compute_metrics_for_instrument(rows, "NSE_INDEX|Nifty 50", 12)

        assert m.prediction_count == 4
        assert m.direction_correct == 4
        assert m.direction_accuracy_pct == pytest.approx(100.0)
        assert m.above_baseline is True


class TestComputeMetricsRandom:
    """~50% accuracy when predictions are evenly split correct/wrong."""

    def test_compute_metrics_random_predictions_returns_around_50_pct(self):
        rows = [
            _make_row(predicted_direction="UP", actual_direction="UP"),
            _make_row(predicted_direction="UP", actual_direction="DOWN"),
            _make_row(predicted_direction="DOWN", actual_direction="DOWN"),
            _make_row(predicted_direction="DOWN", actual_direction="UP"),
        ]
        v = _make_validator()
        m = v.compute_metrics_for_instrument(rows, "NSE_INDEX|Nifty 50", 12)

        assert m.prediction_count == 4
        assert m.direction_correct == 2
        assert m.direction_accuracy_pct == pytest.approx(50.0)
        assert m.above_baseline is False  # not strictly > 50


class TestComputeMetricsEmptyList:
    """Empty input returns zero metrics — no crash."""

    def test_compute_metrics_handles_empty_list(self):
        v = _make_validator()
        m = v.compute_metrics_for_instrument([], "NSE_EQ|RELIANCE", 6)

        assert isinstance(m, AccuracyMetrics)
        assert m.prediction_count == 0
        assert m.direction_correct == 0
        assert m.direction_accuracy_pct == 0.0
        assert m.close_mae_pct == 0.0
        assert m.range_mae_pct == 0.0
        assert m.above_baseline is False


class TestComputeMetricsCloseMAE:
    """MAE is correctly computed as mean abs % error."""

    def test_compute_metrics_close_mae_pct_correct(self):
        # predicted=2_200_000 paisa, actual=2_100_000 → error = 100_000/2_100_000 ≈ 4.762%
        row1 = _make_row(
            predicted_close_paisa=2_200_000,
            actual_close=2_100_000.0,
        )
        # predicted=2_000_000 paisa, actual=2_000_000 → error = 0%
        row2 = _make_row(
            predicted_close_paisa=2_000_000,
            actual_close=2_000_000.0,
        )
        rows = [row1, row2]

        v = _make_validator()
        m = v.compute_metrics_for_instrument(rows, "NSE_INDEX|Nifty 50", 12)

        expected_mae = (100_000 / 2_100_000 * 100.0 + 0.0) / 2.0
        assert m.close_mae_pct == pytest.approx(expected_mae, rel=1e-4)


# ---------------------------------------------------------------------------
# compute_daily
# ---------------------------------------------------------------------------


class TestComputeDailyPersists:
    """compute_daily persists exactly one row per (instrument, horizon) combo."""

    def test_compute_daily_persists_one_row_per_combo(self):
        # Two instruments × two horizons = 4 combos
        rows_from_db = [
            _make_row("NSE_INDEX|Nifty 50", horizon_bars=6),
            _make_row("NSE_INDEX|Nifty 50", horizon_bars=12),
            _make_row("NSE_EQ|RELIANCE", horizon_bars=6),
            _make_row("NSE_EQ|RELIANCE", horizon_bars=12),
        ]

        engine_mock = MagicMock()
        v = _make_validator(engine=engine_mock)

        # Patch fetch_predictions_with_outcomes to return our pre-built rows
        with patch.object(
            v, "fetch_predictions_with_outcomes", return_value=rows_from_db
        ):
            # Patch _persist_daily_accuracy to track calls without touching DB
            with patch.object(v, "_persist_daily_accuracy") as mock_persist:
                results = v.compute_daily(date(2026, 5, 13))

        assert len(results) == 4
        assert mock_persist.call_count == 4

        # Each call should receive the correct trade_date
        for call_args in mock_persist.call_args_list:
            assert call_args[0][0] == date(2026, 5, 13)


# ---------------------------------------------------------------------------
# get_accuracy_summary
# ---------------------------------------------------------------------------


def _build_mock_engine_with_rows(rows: list[tuple]) -> MagicMock:
    """Return a mock engine whose .connect() yields a result with *rows*."""
    mock_result = MagicMock()
    mock_result.fetchall.return_value = rows

    mock_conn = MagicMock()
    mock_conn.execute.return_value = mock_result
    mock_conn.__enter__ = MagicMock(return_value=mock_conn)
    mock_conn.__exit__ = MagicMock(return_value=False)

    mock_engine = MagicMock()
    mock_engine.connect.return_value = mock_conn
    return mock_engine


class TestGetAccuracySummaryAggregates:
    """get_accuracy_summary aggregates raw rows correctly by instrument."""

    def test_get_accuracy_summary_aggregates_by_instrument(self):
        # Simulate 2 rows from DB: (instrument_key, horizon, count, correct, mae, range_mae)
        db_rows = [
            ("NSE_INDEX|Nifty 50", 12, 100, 62, 1.5, 0.5),
            ("NSE_EQ|RELIANCE", 6, 50, 20, 2.0, 0.8),
        ]
        engine = _build_mock_engine_with_rows(db_rows)
        v = _make_validator(engine=engine)

        summary = v.get_accuracy_summary(days=30)

        assert summary["total_predictions"] == 150
        assert summary["by_instrument"]["NSE_INDEX|Nifty 50"]["accuracy_pct"] == pytest.approx(62.0)
        assert summary["by_instrument"]["NSE_EQ|RELIANCE"]["accuracy_pct"] == pytest.approx(40.0)
        assert summary["best_instrument"] == "NSE_INDEX|Nifty 50"
        assert summary["worst_instrument"] == "NSE_EQ|RELIANCE"


# ---------------------------------------------------------------------------
# Recommendation thresholds
# ---------------------------------------------------------------------------


class TestRecommendDrop:
    """accuracy < 45% → DROP."""

    def test_get_accuracy_summary_recommends_drop_below_45_pct(self):
        # 44 correct out of 100 → 44%
        db_rows = [("NSE_INDEX|Nifty 50", 12, 100, 44, 1.5, 0.5)]
        engine = _build_mock_engine_with_rows(db_rows)
        v = _make_validator(engine=engine)
        summary = v.get_accuracy_summary(days=30)

        assert summary["recommendation"] == "DROP"
        assert summary["overall_accuracy_pct"] == pytest.approx(44.0)


class TestRecommendKeep:
    """accuracy > 55% → KEEP."""

    def test_get_accuracy_summary_recommends_keep_above_55_pct(self):
        # 56 correct out of 100 → 56%
        db_rows = [("NSE_INDEX|Nifty 50", 12, 100, 56, 1.0, 0.4)]
        engine = _build_mock_engine_with_rows(db_rows)
        v = _make_validator(engine=engine)
        summary = v.get_accuracy_summary(days=30)

        assert summary["recommendation"] == "KEEP"
        assert summary["overall_accuracy_pct"] == pytest.approx(56.0)


class TestRecommendReduceWeight:
    """45 ≤ accuracy ≤ 55 → REDUCE_WEIGHT."""

    def test_get_accuracy_summary_recommends_reduce_in_band(self):
        # 50 correct out of 100 → 50.0% — in the middle band
        db_rows = [("NSE_INDEX|Nifty 50", 12, 100, 50, 1.2, 0.6)]
        engine = _build_mock_engine_with_rows(db_rows)
        v = _make_validator(engine=engine)
        summary = v.get_accuracy_summary(days=30)

        assert summary["recommendation"] == "REDUCE_WEIGHT"
        assert summary["overall_accuracy_pct"] == pytest.approx(50.0)

    def test_get_accuracy_summary_boundary_45_pct_is_reduce_not_drop(self):
        # Use _recommendation directly to avoid float division artefacts
        # 45.0 is not strictly < 45.0 so should be REDUCE_WEIGHT
        assert KronosValidator._recommendation(45.0) == "REDUCE_WEIGHT"
        assert KronosValidator._recommendation(45.01) == "REDUCE_WEIGHT"

    def test_get_accuracy_summary_boundary_55_pct_is_reduce_not_keep(self):
        # Use _recommendation directly to avoid float division artefacts
        # 55.0 is not strictly > 55.0 so should be REDUCE_WEIGHT
        assert KronosValidator._recommendation(55.0) == "REDUCE_WEIGHT"
        assert KronosValidator._recommendation(54.99) == "REDUCE_WEIGHT"


# ---------------------------------------------------------------------------
# generate_report
# ---------------------------------------------------------------------------


class TestGenerateReport:
    """generate_report returns a Markdown string with per-instrument breakdown."""

    def test_generate_report_includes_per_instrument_breakdown(self):
        db_rows = [
            ("NSE_INDEX|Nifty 50", 12, 80, 50, 1.1, 0.4),
            ("NSE_EQ|INFY", 6, 40, 22, 2.0, 0.9),
        ]
        engine = _build_mock_engine_with_rows(db_rows)
        v = _make_validator(engine=engine)

        report = v.generate_report(days=30)

        # Must contain instrument names
        assert "NSE_INDEX|Nifty 50" in report
        assert "NSE_EQ|INFY" in report

        # Must contain the section header
        assert "Per-Instrument Breakdown" in report

        # Must be a non-empty string
        assert len(report) > 100

    def test_generate_report_no_predictions_returns_no_predictions_message(self):
        """When no forecasts exist, report says 'No predictions found'."""
        engine = _build_mock_engine_with_rows([])
        v = _make_validator(engine=engine)
        report = v.generate_report(days=7)

        assert "No predictions found" in report

    def test_generate_report_no_db_returns_no_predictions_message(self):
        """With no DB engine, report says 'No predictions found'."""
        v = _make_validator(engine=None)
        report = v.generate_report(days=7)

        assert "No predictions found" in report

    def test_generate_report_includes_recommendation(self):
        # 60 correct / 100 → 60% → KEEP
        db_rows = [("NSE_INDEX|Nifty 50", 12, 100, 60, 0.8, 0.3)]
        engine = _build_mock_engine_with_rows(db_rows)
        v = _make_validator(engine=engine)
        report = v.generate_report(days=30)

        assert "KEEP" in report


# ---------------------------------------------------------------------------
# No-engine / no-data edge cases
# ---------------------------------------------------------------------------


class TestNoEngineEdgeCases:
    """All public methods degrade gracefully when no DB engine is provided."""

    def test_fetch_predictions_without_engine_returns_empty_list(self):
        v = _make_validator()
        result = v.fetch_predictions_with_outcomes(date.today())
        assert result == []

    def test_compute_daily_without_engine_returns_empty_list(self):
        v = _make_validator()
        result = v.compute_daily(date.today())
        assert result == []

    def test_get_accuracy_summary_without_engine_returns_empty_summary(self):
        v = _make_validator()
        summary = v.get_accuracy_summary(days=30)

        assert summary["total_predictions"] == 0
        assert summary["overall_accuracy_pct"] == 0.0
        assert summary["by_instrument"] == {}
        assert summary["by_horizon"] == {}
        assert summary["recommendation"] == "REDUCE_WEIGHT"
