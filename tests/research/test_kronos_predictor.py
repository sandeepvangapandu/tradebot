"""Phase G — KronosForecaster tests.

All tests mock the heavy KronosPredictor.predict() call so no model weights
are downloaded during test execution.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch, PropertyMock
from zoneinfo import ZoneInfo

import pandas as pd
import numpy as np
import pytest

IST = ZoneInfo("Asia/Kolkata")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_bars(n: int = 50, with_amount: bool = True) -> pd.DataFrame:
    """Return a minimal OHLCV DataFrame with DatetimeIndex."""
    ts = pd.date_range("2026-01-02 09:15", periods=n, freq="5min", tz="Asia/Kolkata")
    rng = np.random.default_rng(42)
    close = 50000.0 + rng.normal(0, 200, n).cumsum()
    df = pd.DataFrame(
        {
            "open": close - rng.uniform(10, 50, n),
            "high": close + rng.uniform(10, 80, n),
            "low": close - rng.uniform(10, 80, n),
            "close": close,
            "volume": rng.integers(1000, 5000, n).astype(float),
        },
        index=ts,
    )
    if with_amount:
        df["amount"] = df["close"] * df["volume"]
    return df


def _make_forecast(horizon: int = 12, base_close: float = 50000.0) -> pd.DataFrame:
    """Return a synthetic forecast DataFrame."""
    ts = pd.date_range("2026-01-02 13:15", periods=horizon, freq="5min", tz="Asia/Kolkata")
    rng = np.random.default_rng(99)
    close = base_close + rng.normal(0, 100, horizon).cumsum()
    return pd.DataFrame(
        {
            "open": close - 20,
            "high": close + 80,
            "low": close - 80,
            "close": close,
            "volume": rng.integers(500, 2000, horizon).astype(float),
            "amount": close * rng.integers(500, 2000, horizon).astype(float),
        },
        index=ts,
    )


# ---------------------------------------------------------------------------
# Fixture: patch the Kronos load chain so tests never download weights
# ---------------------------------------------------------------------------

@pytest.fixture()
def forecaster_no_load():
    """KronosForecaster with _predictor pre-populated via mock (skips load())."""
    from src.research.kronos_predictor import KronosForecaster

    fc = KronosForecaster(model_size="small", device="cpu", max_context=64)
    # Inject a mock predictor so load() is never actually called
    mock_pred = MagicMock()
    fc._predictor = mock_pred
    fc._model = MagicMock()
    fc._tokenizer = MagicMock()
    return fc, mock_pred


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestLoadLazy:
    def test_load_called_twice_is_idempotent(self):
        """load() must be idempotent — second call should not re-initialise."""
        from src.research.kronos_predictor import KronosForecaster

        fc = KronosForecaster(model_size="small", device="cpu")

        # Pre-inject a fake predictor so load() short-circuits
        sentinel = object()
        fc._predictor = sentinel

        fc.load()  # should return immediately without touching HuggingFace

        assert fc._predictor is sentinel, "load() replaced already-loaded predictor"

    def test_predictor_none_before_load(self):
        """Fresh instance has _predictor = None."""
        from src.research.kronos_predictor import KronosForecaster

        fc = KronosForecaster()
        assert fc._predictor is None


class TestSynthesizesAmount:
    def test_amount_synthesised_when_missing(self, forecaster_no_load):
        """predict() synthesises 'amount' = close * volume when column absent."""
        fc, mock_pred = forecaster_no_load
        bars = _make_bars(20, with_amount=False)
        assert "amount" not in bars.columns

        fake_forecast = _make_forecast(horizon=5)
        mock_pred.predict.return_value = fake_forecast

        result = fc.predict(bars, horizon=5)
        assert result is not None
        # The predictor should have been called — which means amount was synthesised
        mock_pred.predict.assert_called_once()

    def test_raises_when_volume_also_missing(self, forecaster_no_load):
        """predict() raises ValueError if both volume and amount are absent."""
        fc, _ = forecaster_no_load
        bars = _make_bars(20, with_amount=False).drop(columns=["volume"])

        with pytest.raises(ValueError, match="missing required columns"):
            fc.predict(bars, horizon=5)


class TestTruncatesToMaxContext:
    def test_bars_truncated_to_max_context(self, forecaster_no_load):
        """predict() trims bars to max_context before calling predictor."""
        fc, mock_pred = forecaster_no_load
        fc.max_context = 20  # tiny context for the test

        bars = _make_bars(n=50)  # 50 bars, but context cap is 20
        fake_forecast = _make_forecast(horizon=5)
        mock_pred.predict.return_value = fake_forecast

        fc.predict(bars, horizon=5)

        # The df argument passed to predictor.predict must be <= max_context rows
        call_args = mock_pred.predict.call_args
        df_kwarg = call_args.kwargs.get("df")
        df_passed = df_kwarg if df_kwarg is not None else call_args.args[0]
        assert len(df_passed) <= 20


class TestPredictSummaryReturnsRequiredKeys:
    REQUIRED_KEYS = {
        "ts",
        "instrument_key",
        "model_size",
        "timeframe",
        "context_bars",
        "horizon_bars",
        "current_close_paisa",
        "predicted_close_paisa",
        "predicted_high_paisa",
        "predicted_low_paisa",
        "predicted_direction",
        "predicted_change_pct",
        "predicted_range_pct",
        "raw_forecast",
        "inference_ms",
        "model_version",
    }

    def test_all_keys_present(self, forecaster_no_load):
        fc, mock_pred = forecaster_no_load
        bars = _make_bars(30)
        fake_forecast = _make_forecast(horizon=12)
        mock_pred.predict.return_value = fake_forecast

        summary = fc.predict_summary(
            instrument_key="NSE_EQ|TEST",
            bars=bars,
            timeframe="5m",
            horizon=12,
            persist=False,
        )

        missing = self.REQUIRED_KEYS - set(summary.keys())
        assert not missing, f"Summary missing keys: {missing}"

    def test_paisa_values_are_integers(self, forecaster_no_load):
        fc, mock_pred = forecaster_no_load
        bars = _make_bars(30)
        fake_forecast = _make_forecast(horizon=12)
        mock_pred.predict.return_value = fake_forecast

        summary = fc.predict_summary(
            instrument_key="NSE_EQ|TEST",
            bars=bars,
            timeframe="5m",
            horizon=12,
            persist=False,
        )

        for key in ("current_close_paisa", "predicted_close_paisa", "predicted_high_paisa", "predicted_low_paisa"):
            assert isinstance(summary[key], int), f"{key} should be int, got {type(summary[key])}"


class TestPredictSummaryPersists:
    def test_persist_called_when_engine_provided(self, forecaster_no_load):
        """_persist() is invoked when persist=True and engine is set."""
        fc, mock_pred = forecaster_no_load
        bars = _make_bars(30)
        fake_forecast = _make_forecast(horizon=12)
        mock_pred.predict.return_value = fake_forecast

        mock_engine = MagicMock()
        mock_conn = MagicMock()
        mock_engine.begin.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_engine.begin.return_value.__exit__ = MagicMock(return_value=False)
        fc._engine = mock_engine

        fc.predict_summary(
            instrument_key="NSE_EQ|TEST",
            bars=bars,
            persist=True,
        )

        mock_engine.begin.assert_called_once()

    def test_no_persist_when_flag_false(self, forecaster_no_load):
        """_persist() must NOT be called when persist=False."""
        fc, mock_pred = forecaster_no_load
        bars = _make_bars(30)
        fake_forecast = _make_forecast(horizon=12)
        mock_pred.predict.return_value = fake_forecast

        mock_engine = MagicMock()
        fc._engine = mock_engine

        fc.predict_summary(
            instrument_key="NSE_EQ|TEST",
            bars=bars,
            persist=False,
        )

        mock_engine.begin.assert_not_called()


class TestPredictedDirectionReturnsOneOfThree:
    @pytest.mark.parametrize(
        "last_close,pred_close,expected",
        [
            (50000.0, 50600.0, "UP"),    # +1.2% -> UP
            (50000.0, 49400.0, "DOWN"),  # -1.2% -> DOWN
            (50000.0, 50040.0, "FLAT"),  # +0.08% change -> FLAT (threshold is < 0.1%)
        ],
    )
    def test_direction_correct(self, forecaster_no_load, last_close, pred_close, expected):
        fc, mock_pred = forecaster_no_load
        bars = _make_bars(30)
        bars["close"] = last_close  # fix last close

        fake_forecast = _make_forecast(horizon=12, base_close=last_close)
        # Override close so the last row matches pred_close
        fake_forecast["close"] = np.linspace(last_close, pred_close, 12)
        mock_pred.predict.return_value = fake_forecast

        direction = fc.predicted_direction("NSE_EQ|TEST", bars, horizon=12)
        assert direction in ("UP", "DOWN", "FLAT")
        assert direction == expected

    def test_direction_always_one_of_three_values(self, forecaster_no_load):
        fc, mock_pred = forecaster_no_load
        bars = _make_bars(30)
        fake_forecast = _make_forecast(horizon=6)
        mock_pred.predict.return_value = fake_forecast

        direction = fc.predicted_direction("NSE_EQ|TEST", bars, horizon=6)
        assert direction in ("UP", "DOWN", "FLAT")


class TestPersistHandlesDbErrorGracefully:
    def test_db_error_does_not_propagate(self, forecaster_no_load):
        """_persist() must swallow DB errors and log a warning, not raise."""
        fc, mock_pred = forecaster_no_load
        bars = _make_bars(30)
        fake_forecast = _make_forecast(horizon=12)
        mock_pred.predict.return_value = fake_forecast

        # Engine that raises on begin()
        mock_engine = MagicMock()
        mock_engine.begin.side_effect = RuntimeError("Connection refused")
        fc._engine = mock_engine

        # Should not raise — errors must be caught internally
        summary = fc.predict_summary(
            instrument_key="NSE_EQ|TEST",
            bars=bars,
            persist=True,
        )
        # Summary still returned despite DB failure
        assert summary.get("predicted_direction") in ("UP", "DOWN", "FLAT")
