"""Tests for src.strategy.confluence_engine.

All tests use inline mocks — no external DB or network required.
"""

from __future__ import annotations

from unittest.mock import MagicMock, call, patch

import pytest

from src.strategy.confluence_engine import (
    ConfluenceDimension,
    ConfluenceEngine,
    DEFAULT_WEIGHTS,
    DimensionResult,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_engine_that_accepts_insert() -> MagicMock:
    """Return a mock SQLAlchemy sync engine that supports begin() context manager."""
    mock_engine = MagicMock()
    mock_conn = MagicMock()
    mock_conn.__enter__ = MagicMock(return_value=mock_conn)
    mock_conn.__exit__ = MagicMock(return_value=False)
    mock_engine.begin.return_value = mock_conn
    # connect() used by get_recent_scores / get_pass_rate
    mock_engine.connect.return_value = mock_conn
    return mock_engine


def _all_dimensions_at(score: float, skipped: bool = False) -> list[DimensionResult]:
    """Build a list of DimensionResults for every dimension at the same score."""
    return [
        DimensionResult(
            dimension=dim,
            score=score,
            weight=DEFAULT_WEIGHTS[dim],
            matched=["cond_a"] if score > 0 else [],
            failed=[] if score > 0 else ["cond_a"],
            skipped=skipped,
        )
        for dim in ConfluenceDimension
    ]


# ---------------------------------------------------------------------------
# test_evaluate_perfect_score_returns_100
# ---------------------------------------------------------------------------

class TestEvaluatePerfectScore:
    def test_evaluate_perfect_score_returns_100(self):
        """All dimensions at score=1.0 → composite_score == 100."""
        engine = ConfluenceEngine(threshold=70.0, db_engine=None)
        results = _all_dimensions_at(1.0)
        outcome = engine.evaluate(
            strategy_name="TestStrategy",
            instrument_key="NSE_INDEX|Nifty Bank",
            signal_type="BUY_CE",
            dimension_results=results,
        )
        assert outcome["composite_score"] == pytest.approx(100.0, abs=0.01)
        assert outcome["passed"] is True


# ---------------------------------------------------------------------------
# test_evaluate_zero_score_returns_0
# ---------------------------------------------------------------------------

class TestEvaluateZeroScore:
    def test_evaluate_zero_score_returns_0(self):
        """All dimensions at score=0.0 → composite_score == 0."""
        engine = ConfluenceEngine(threshold=70.0, db_engine=None)
        results = _all_dimensions_at(0.0)
        outcome = engine.evaluate(
            strategy_name="TestStrategy",
            instrument_key="NSE_EQ|RELIANCE",
            signal_type="SELL",
            dimension_results=results,
        )
        assert outcome["composite_score"] == pytest.approx(0.0, abs=0.01)
        assert outcome["passed"] is False


# ---------------------------------------------------------------------------
# test_evaluate_partial_score_weighted_average
# ---------------------------------------------------------------------------

class TestEvaluatePartialScore:
    def test_evaluate_partial_score_weighted_average(self):
        """Two equal-weight dimensions at 0.5 and 1.0 → weighted avg."""
        # Only PRICE_ACTION + MICROSTRUCTURE, weight 0.25 and 0.15
        results = [
            DimensionResult(
                dimension=ConfluenceDimension.PRICE_ACTION,
                score=1.0,
                weight=0.25,
                matched=["cpr_breakout"],
            ),
            DimensionResult(
                dimension=ConfluenceDimension.MICROSTRUCTURE,
                score=0.0,
                weight=0.25,  # override to equal weight for simple arithmetic
                failed=["cvd_divergence"],
            ),
        ]
        engine = ConfluenceEngine(threshold=70.0, db_engine=None)
        outcome = engine.evaluate(
            strategy_name="S",
            instrument_key="NSE_EQ|TCS",
            signal_type="BUY",
            dimension_results=results,
        )
        # weighted average: (1.0*0.25 + 0.0*0.25) / (0.25+0.25) * 100 = 50.0
        assert outcome["composite_score"] == pytest.approx(50.0, abs=0.01)


# ---------------------------------------------------------------------------
# test_evaluate_above_threshold_passes
# ---------------------------------------------------------------------------

class TestEvaluateAboveThreshold:
    def test_evaluate_above_threshold_passes(self):
        """Score ≥ threshold → passed is True."""
        engine = ConfluenceEngine(threshold=70.0, db_engine=None)
        results = [
            DimensionResult(
                dimension=ConfluenceDimension.PRICE_ACTION,
                score=1.0,
                weight=1.0,
                matched=["signal"],
            )
        ]
        outcome = engine.evaluate("S", "K", "BUY", results)
        assert outcome["passed"] is True
        assert outcome["composite_score"] >= 70.0


# ---------------------------------------------------------------------------
# test_evaluate_below_threshold_fails
# ---------------------------------------------------------------------------

class TestEvaluateBelowThreshold:
    def test_evaluate_below_threshold_fails(self):
        """Score < threshold → passed is False."""
        engine = ConfluenceEngine(threshold=70.0, db_engine=None)
        results = [
            DimensionResult(
                dimension=ConfluenceDimension.PRICE_ACTION,
                score=0.5,
                weight=1.0,
                matched=["weak_signal"],
            )
        ]
        outcome = engine.evaluate("S", "K", "SELL", results)
        assert outcome["passed"] is False
        assert outcome["composite_score"] == pytest.approx(50.0, abs=0.01)


# ---------------------------------------------------------------------------
# test_skipped_dimensions_renormalize_weights
# ---------------------------------------------------------------------------

class TestSkippedDimensionsRenormalize:
    def test_skipped_dimensions_renormalize_weights(self):
        """Skipped dimensions are excluded; remaining weights re-normalised."""
        # PRICE_ACTION weight=0.25, skipped
        # MICROSTRUCTURE weight=0.15, score=1.0
        results = [
            DimensionResult(
                dimension=ConfluenceDimension.PRICE_ACTION,
                score=0.0,
                weight=0.25,
                skipped=True,
            ),
            DimensionResult(
                dimension=ConfluenceDimension.MICROSTRUCTURE,
                score=1.0,
                weight=0.15,
                matched=["cvd_ok"],
            ),
        ]
        engine = ConfluenceEngine(threshold=70.0, db_engine=None)
        outcome = engine.evaluate("S", "K", "BUY", results)
        # Only MICROSTRUCTURE is active; its weight re-normalises to 1.0
        # score = (1.0 * 0.15) / 0.15 * 100 = 100.0
        assert outcome["composite_score"] == pytest.approx(100.0, abs=0.01)
        assert outcome["passed"] is True

    def test_all_skipped_returns_zero(self):
        """When every dimension is skipped, score is 0 and does not pass."""
        results = _all_dimensions_at(1.0, skipped=True)
        engine = ConfluenceEngine(threshold=70.0, db_engine=None)
        outcome = engine.evaluate("S", "K", "BUY", results)
        assert outcome["composite_score"] == pytest.approx(0.0, abs=0.01)
        assert outcome["passed"] is False


# ---------------------------------------------------------------------------
# test_evaluate_persists_to_db
# ---------------------------------------------------------------------------

class TestEvaluatePersistsToDb:
    def test_evaluate_persists_to_db(self):
        """evaluate() calls engine.begin() once to insert the audit row."""
        mock_engine = _make_engine_that_accepts_insert()
        ce = ConfluenceEngine(threshold=70.0, db_engine=mock_engine)

        results = [
            DimensionResult(
                dimension=ConfluenceDimension.PRICE_ACTION,
                score=0.8,
                weight=1.0,
                matched=["cpr_ok"],
            )
        ]
        ce.evaluate("Strategy", "NSE_EQ|INFY", "BUY", results, signal_id=42)

        # engine.begin() should have been called (for the INSERT)
        mock_engine.begin.assert_called_once()

    def test_evaluate_no_db_does_not_raise(self):
        """When db_engine=None, evaluate() returns a result without raising."""
        ce = ConfluenceEngine(threshold=70.0, db_engine=None)
        results = [
            DimensionResult(
                dimension=ConfluenceDimension.VIX,
                score=0.9,
                weight=1.0,
                matched=["vix_low"],
            )
        ]
        outcome = ce.evaluate("S", "K", "BUY", results)
        assert "composite_score" in outcome
        assert "passed" in outcome


# ---------------------------------------------------------------------------
# test_evaluate_handles_no_dimensions_gracefully
# ---------------------------------------------------------------------------

class TestEvaluateNoDimensions:
    def test_evaluate_handles_no_dimensions_gracefully(self):
        """Empty dimension_results list → score 0, not passed, no crash."""
        engine = ConfluenceEngine(threshold=70.0, db_engine=None)
        outcome = engine.evaluate("S", "K", "BUY", [])
        assert outcome["composite_score"] == pytest.approx(0.0, abs=0.01)
        assert outcome["passed"] is False
        assert outcome["dimension_breakdown"] == []


# ---------------------------------------------------------------------------
# test_get_pass_rate_returns_fraction_in_range
# ---------------------------------------------------------------------------

class TestGetPassRate:
    def test_get_pass_rate_returns_fraction_in_range(self):
        """get_pass_rate() returns a float in [0, 1] when db returns rows."""
        mock_engine = MagicMock()
        mock_conn = MagicMock()
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)
        mock_engine.connect.return_value = mock_conn

        # Simulate 7 passes out of 10 total
        mock_conn.execute.return_value.fetchone.return_value = (7, 10)

        ce = ConfluenceEngine(threshold=70.0, db_engine=mock_engine)
        rate = ce.get_pass_rate("TestStrategy", days=7)

        assert 0.0 <= rate <= 1.0
        assert rate == pytest.approx(0.7, abs=0.001)

    def test_get_pass_rate_no_rows_returns_zero(self):
        """get_pass_rate() returns 0.0 when no rows exist."""
        mock_engine = MagicMock()
        mock_conn = MagicMock()
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)
        mock_engine.connect.return_value = mock_conn
        mock_conn.execute.return_value.fetchone.return_value = (0, 0)

        ce = ConfluenceEngine(threshold=70.0, db_engine=mock_engine)
        assert ce.get_pass_rate("S") == 0.0

    def test_get_pass_rate_no_db_returns_zero(self):
        """get_pass_rate() returns 0.0 when db_engine is None."""
        ce = ConfluenceEngine(threshold=70.0, db_engine=None)
        assert ce.get_pass_rate("S") == 0.0


# ---------------------------------------------------------------------------
# test_default_weights_sum_to_one
# ---------------------------------------------------------------------------

class TestDefaultWeights:
    def test_default_weights_sum_to_one(self):
        """DEFAULT_WEIGHTS must sum to exactly 1.0 (within float tolerance)."""
        total = sum(DEFAULT_WEIGHTS.values())
        assert total == pytest.approx(1.0, abs=0.001)

    def test_all_dimensions_in_default_weights(self):
        """Every ConfluenceDimension has a weight in DEFAULT_WEIGHTS."""
        for dim in ConfluenceDimension:
            assert dim in DEFAULT_WEIGHTS, f"Missing weight for {dim}"


# ---------------------------------------------------------------------------
# test_update_weights_validates_sum_to_one
# ---------------------------------------------------------------------------

class TestUpdateWeights:
    def test_update_weights_accepts_valid_weights(self):
        """update_weights() succeeds when values sum to 1.0."""
        ce = ConfluenceEngine(threshold=70.0, db_engine=None)
        new_weights = {dim: 1.0 / len(ConfluenceDimension) for dim in ConfluenceDimension}
        # Should not raise
        ce.update_weights(new_weights)

    def test_update_weights_validates_sum_to_one(self):
        """update_weights() raises ValueError when weights do not sum to ~1.0."""
        ce = ConfluenceEngine(threshold=70.0, db_engine=None)
        bad_weights = {dim: 0.01 for dim in ConfluenceDimension}  # sum = 0.10, not 1.0
        with pytest.raises(ValueError, match="must sum to 1.0"):
            ce.update_weights(bad_weights)

    def test_update_weights_persists_to_config(self):
        """update_weights() calls engine.begin() to persist to bot_config."""
        mock_engine = _make_engine_that_accepts_insert()
        ce = ConfluenceEngine(threshold=70.0, db_engine=mock_engine)
        new_weights = {dim: 1.0 / len(ConfluenceDimension) for dim in ConfluenceDimension}
        ce.update_weights(new_weights)
        mock_engine.begin.assert_called()

    def test_update_weights_string_keys_accepted(self):
        """update_weights() accepts string dimension keys."""
        ce = ConfluenceEngine(threshold=70.0, db_engine=None)
        new_weights = {dim.value: 1.0 / len(ConfluenceDimension) for dim in ConfluenceDimension}
        ce.update_weights(new_weights)  # should not raise

    def test_constructor_invalid_weights_raise(self):
        """Passing invalid weights to constructor raises ValueError."""
        bad = {dim: 0.01 for dim in ConfluenceDimension}  # sum=0.10
        with pytest.raises(ValueError, match="must sum to 1.0"):
            ConfluenceEngine(weights=bad, threshold=70.0)


# ---------------------------------------------------------------------------
# Additional edge-case tests
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_result_contains_all_expected_keys(self):
        """evaluate() result dict has all required keys."""
        ce = ConfluenceEngine(threshold=70.0, db_engine=None)
        outcome = ce.evaluate("S", "K", "BUY", [])
        required = {
            "composite_score",
            "passed",
            "threshold",
            "dimension_breakdown",
            "matched_conditions",
            "failed_conditions",
        }
        assert required.issubset(outcome.keys())

    def test_threshold_returned_in_result(self):
        """evaluate() result contains the configured threshold."""
        ce = ConfluenceEngine(threshold=65.0, db_engine=None)
        outcome = ce.evaluate("S", "K", "BUY", [])
        assert outcome["threshold"] == pytest.approx(65.0)

    def test_score_clamped_to_100(self):
        """composite_score cannot exceed 100 even with weight > 1 supplied."""
        ce = ConfluenceEngine(threshold=70.0, db_engine=None)
        # score=2.0 (invalid but should clamp)
        results = [
            DimensionResult(
                dimension=ConfluenceDimension.PRICE_ACTION,
                score=2.0,
                weight=1.0,
            )
        ]
        outcome = ce.evaluate("S", "K", "BUY", results)
        assert outcome["composite_score"] <= 100.0

    def test_signal_id_none_accepted(self):
        """signal_id=None is accepted and does not raise."""
        ce = ConfluenceEngine(threshold=70.0, db_engine=None)
        results = [
            DimensionResult(
                dimension=ConfluenceDimension.VIX,
                score=0.5,
                weight=1.0,
            )
        ]
        outcome = ce.evaluate("S", "K", "BUY", results, signal_id=None)
        assert outcome is not None


# ---------------------------------------------------------------------------
# Phase G.3: MODEL_FORECAST dimension tests
# ---------------------------------------------------------------------------

class TestModelForecastDimension:
    def test_default_weights_include_model_forecast(self):
        """MODEL_FORECAST is present in DEFAULT_WEIGHTS and total still sums to ~1.0."""
        assert ConfluenceDimension.MODEL_FORECAST in DEFAULT_WEIGHTS
        total = sum(DEFAULT_WEIGHTS.values())
        assert total == pytest.approx(1.0, abs=0.001)

    def test_evaluate_with_kronos_dimension_passes_through(self):
        """A DimensionResult with MODEL_FORECAST dim appears in dimension_breakdown of result."""
        ce = ConfluenceEngine(threshold=70.0, db_engine=None)
        results = [
            DimensionResult(
                dimension=ConfluenceDimension.MODEL_FORECAST,
                score=0.85,
                weight=0.055,
                matched=["kronos_up_pct=+1.20"],
                skipped=False,
            ),
        ]
        outcome = ce.evaluate(
            strategy_name="TestKronos",
            instrument_key="NSE_INDEX|Nifty Bank",
            signal_type="BUY_CE",
            dimension_results=results,
        )
        breakdown_dims = [entry["dimension"] for entry in outcome["dimension_breakdown"]]
        assert "model_forecast" in breakdown_dims
        # composite should reflect the MODEL_FORECAST score (0.85 * 100 = 85.0)
        assert outcome["composite_score"] == pytest.approx(85.0, abs=0.01)
        assert outcome["passed"] is True
