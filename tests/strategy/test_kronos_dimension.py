"""Tests for src.strategy.kronos_dimension.

All tests are self-contained — no Kronos model is loaded, no DB required.
We only import from src.strategy.kronos_dimension and src.strategy.confluence_engine.
"""

from __future__ import annotations

import pytest

from src.strategy.confluence_engine import ConfluenceDimension
from src.strategy.kronos_dimension import kronos_dimension_result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_summary(direction: str, change_pct: float) -> dict:
    """Build a minimal forecast_summary dict."""
    return {
        "predicted_direction": direction,
        "predicted_change_pct": change_pct,
        "predicted_range_pct": abs(change_pct) * 1.5,
    }


# ---------------------------------------------------------------------------
# test_returns_skipped_when_no_forecast
# ---------------------------------------------------------------------------

class TestNoForecast:
    def test_returns_skipped_when_no_forecast(self):
        """When forecast_summary is None, result is skipped and failed=['no_forecast_available']."""
        result = kronos_dimension_result(signal_type="BUY", forecast_summary=None)
        assert result.dimension == ConfluenceDimension.MODEL_FORECAST
        assert result.skipped is True
        assert "no_forecast_available" in result.failed
        assert result.score == 0.0

    def test_returns_skipped_when_empty_dict(self):
        """When forecast_summary is an empty dict, result is also skipped."""
        result = kronos_dimension_result(signal_type="BUY", forecast_summary={})
        assert result.skipped is True
        assert "no_forecast_available" in result.failed


# ---------------------------------------------------------------------------
# test_returns_neutral_for_flat_prediction
# ---------------------------------------------------------------------------

class TestFlatPrediction:
    def test_returns_neutral_for_flat_prediction(self):
        """A FLAT predicted_direction yields score=0.5 (neutral, no info)."""
        summary = _make_summary("FLAT", 0.2)
        result = kronos_dimension_result(signal_type="BUY", forecast_summary=summary)
        assert result.skipped is False
        assert result.score == pytest.approx(0.5, abs=0.001)
        assert "kronos_flat_pred" in result.matched

    def test_flat_prediction_for_sell_signal(self):
        """FLAT prediction is neutral regardless of signal direction."""
        summary = _make_summary("FLAT", 0.5)
        result = kronos_dimension_result(signal_type="SELL", forecast_summary=summary)
        assert result.score == pytest.approx(0.5, abs=0.001)


# ---------------------------------------------------------------------------
# test_buy_signal_with_up_prediction_returns_high_score
# ---------------------------------------------------------------------------

class TestBuySignalUpPrediction:
    def test_buy_signal_with_up_prediction_returns_high_score(self):
        """BUY signal + UP prediction → score > 0.5 (aligned)."""
        summary = _make_summary("UP", 0.8)
        result = kronos_dimension_result(signal_type="BUY", forecast_summary=summary)
        assert result.skipped is False
        assert result.score > 0.5
        assert result.score <= 1.0
        assert result.matched  # has a match marker

    def test_buy_ce_with_up_prediction_high_score(self):
        """BUY_CE is long-delta (bullish) → aligned with UP."""
        summary = _make_summary("UP", 1.5)  # >1% → capped at 1.0 magnitude
        result = kronos_dimension_result(signal_type="BUY_CE", forecast_summary=summary)
        assert result.score == pytest.approx(1.0, abs=0.001)  # 0.5 + 0.5*1 = 1.0

    def test_strong_alignment_reaches_1_0(self):
        """pred_change_pct >= 1.0% with alignment → score == 1.0."""
        summary = _make_summary("UP", 2.0)  # way above 1%, magnitude capped at 1
        result = kronos_dimension_result(signal_type="BUY", forecast_summary=summary)
        assert result.score == pytest.approx(1.0, abs=0.001)


# ---------------------------------------------------------------------------
# test_buy_signal_with_down_prediction_returns_low_score
# ---------------------------------------------------------------------------

class TestBuySignalDownPrediction:
    def test_buy_signal_with_down_prediction_returns_low_score(self):
        """BUY signal + DOWN prediction → score < 0.5 (opposing)."""
        summary = _make_summary("DOWN", 0.8)
        result = kronos_dimension_result(signal_type="BUY", forecast_summary=summary)
        assert result.skipped is False
        assert result.score < 0.5
        assert result.failed  # has a failure marker

    def test_strong_opposition_reaches_0_0(self):
        """pred_change_pct >= 1.0% with opposition → score == 0.0."""
        summary = _make_summary("DOWN", 1.5)
        result = kronos_dimension_result(signal_type="BUY", forecast_summary=summary)
        assert result.score == pytest.approx(0.0, abs=0.001)


# ---------------------------------------------------------------------------
# test_buy_pe_with_down_prediction_returns_high_score
# ---------------------------------------------------------------------------

class TestBuyPeDownPrediction:
    def test_buy_pe_with_down_prediction_returns_high_score(self):
        """BUY_PE is a bearish bet (put option) → aligned with DOWN prediction."""
        summary = _make_summary("DOWN", 1.0)
        result = kronos_dimension_result(signal_type="BUY_PE", forecast_summary=summary)
        assert result.skipped is False
        assert result.score == pytest.approx(1.0, abs=0.001)  # full alignment at 1% move

    def test_buy_pe_with_up_prediction_returns_low_score(self):
        """BUY_PE vs UP prediction → opposition (put buyer loses if market goes up)."""
        summary = _make_summary("UP", 1.0)
        result = kronos_dimension_result(signal_type="BUY_PE", forecast_summary=summary)
        assert result.score == pytest.approx(0.0, abs=0.001)


# ---------------------------------------------------------------------------
# test_sell_pe_with_up_prediction_returns_high_score
# ---------------------------------------------------------------------------

class TestSellPeUpPrediction:
    def test_sell_pe_with_up_prediction_returns_high_score(self):
        """SELL_PE is short puts (bullish bias) → aligned with UP prediction."""
        summary = _make_summary("UP", 1.0)
        result = kronos_dimension_result(signal_type="SELL_PE", forecast_summary=summary)
        assert result.skipped is False
        assert result.score == pytest.approx(1.0, abs=0.001)

    def test_sell_pe_with_down_prediction_returns_low_score(self):
        """SELL_PE vs DOWN prediction → short puts face losses when market drops."""
        summary = _make_summary("DOWN", 1.0)
        result = kronos_dimension_result(signal_type="SELL_PE", forecast_summary=summary)
        assert result.score == pytest.approx(0.0, abs=0.001)


# ---------------------------------------------------------------------------
# test_straddle_with_flat_prediction_returns_neutral
# ---------------------------------------------------------------------------

class TestStraddleFlatPrediction:
    def test_straddle_with_flat_prediction_returns_neutral(self):
        """STRADDLE expects FLAT. When prediction is also FLAT → neutral 0.5."""
        summary = _make_summary("FLAT", 0.1)
        result = kronos_dimension_result(signal_type="STRADDLE", forecast_summary=summary)
        assert result.skipped is False
        assert result.score == pytest.approx(0.5, abs=0.001)

    def test_straddle_expects_flat_direction(self):
        """STRADDLE maps to expected=FLAT; UP prediction is then 'opposing'."""
        summary = _make_summary("UP", 1.0)
        result = kronos_dimension_result(signal_type="STRADDLE", forecast_summary=summary)
        # UP vs expected FLAT → opposed → low score
        assert result.score == pytest.approx(0.0, abs=0.001)


# ---------------------------------------------------------------------------
# test_score_scales_with_predicted_magnitude
# ---------------------------------------------------------------------------

class TestScaleWithMagnitude:
    def test_score_scales_with_predicted_magnitude_aligned(self):
        """Higher predicted_change_pct → higher score when aligned."""
        summary_small = _make_summary("UP", 0.5)  # 50% of 1% cap
        summary_large = _make_summary("UP", 1.5)  # capped at 100%

        r_small = kronos_dimension_result("BUY", summary_small)
        r_large = kronos_dimension_result("BUY", summary_large)

        # 0.5% change: score = 0.5 + 0.5*0.5 = 0.75
        assert r_small.score == pytest.approx(0.75, abs=0.001)
        # >=1.0% change: score = 0.5 + 0.5*1.0 = 1.0
        assert r_large.score == pytest.approx(1.0, abs=0.001)
        assert r_large.score > r_small.score

    def test_score_scales_with_predicted_magnitude_opposing(self):
        """Higher predicted_change_pct → lower score when opposing."""
        summary_small = _make_summary("DOWN", 0.5)
        summary_large = _make_summary("DOWN", 1.5)

        r_small = kronos_dimension_result("BUY", summary_small)
        r_large = kronos_dimension_result("BUY", summary_large)

        # 0.5% opposing: score = 0.5 - 0.5*0.5 = 0.25
        assert r_small.score == pytest.approx(0.25, abs=0.001)
        # >=1.0% opposing: score = 0.5 - 0.5*1.0 = 0.0
        assert r_large.score == pytest.approx(0.0, abs=0.001)
        assert r_large.score < r_small.score

    def test_zero_magnitude_aligned_returns_0_5(self):
        """0% predicted change with alignment → score == 0.5 (weak agreement)."""
        summary = _make_summary("UP", 0.0)
        result = kronos_dimension_result("BUY", summary)
        assert result.score == pytest.approx(0.5, abs=0.001)


# ---------------------------------------------------------------------------
# test_unknown_signal_type_returns_neutral_with_marker
# ---------------------------------------------------------------------------

class TestUnknownSignalType:
    def test_unknown_signal_type_returns_neutral_with_marker(self):
        """Unrecognised signal_type → score 0.5 with unknown_signal_type marker in matched."""
        summary = _make_summary("UP", 1.0)
        result = kronos_dimension_result(signal_type="EXOTIC_SPREAD", forecast_summary=summary)
        assert result.skipped is False
        assert result.score == pytest.approx(0.5, abs=0.001)
        assert any("unknown_signal_type" in m for m in result.matched)

    def test_case_insensitive_signal_type(self):
        """signal_type is normalised to upper-case before lookup."""
        summary = _make_summary("UP", 1.0)
        r_upper = kronos_dimension_result("BUY", summary)
        r_lower = kronos_dimension_result("buy", summary)
        assert r_upper.score == pytest.approx(r_lower.score, abs=0.001)


# ---------------------------------------------------------------------------
# Additional: dimension field is always MODEL_FORECAST
# ---------------------------------------------------------------------------

class TestDimensionField:
    def test_dimension_is_model_forecast_for_all_paths(self):
        """All code paths return ConfluenceDimension.MODEL_FORECAST."""
        paths = [
            (None, "BUY"),
            (_make_summary("FLAT", 0.5), "BUY"),
            (_make_summary("UP", 1.0), "BUY"),
            (_make_summary("DOWN", 1.0), "BUY"),
            (_make_summary("UP", 1.0), "UNKNOWN_TYPE"),
        ]
        for summary, sig in paths:
            r = kronos_dimension_result(sig, summary)
            assert r.dimension == ConfluenceDimension.MODEL_FORECAST, (
                f"Expected MODEL_FORECAST for signal={sig}, summary={summary}"
            )

    def test_custom_weight_is_respected(self):
        """Caller-supplied weight is passed through to DimensionResult."""
        summary = _make_summary("UP", 0.5)
        result = kronos_dimension_result("BUY", summary, weight=0.10)
        assert result.weight == pytest.approx(0.10, abs=0.0001)
