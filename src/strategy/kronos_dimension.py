"""Helper: convert a Kronos forecast row into a ConfluenceEngine DimensionResult.

This is a thin adapter so the engine doesn't need to know about Kronos internals.
The Kronos forecast is produced by ``KronosForecaster.predict_summary()`` in
``src.research.kronos_predictor`` (built in Phase G.1, parallel branch).

Usage example::

    from src.strategy.kronos_dimension import kronos_dimension_result

    summary = kronos_forecaster.predict_summary("NSE_INDEX|Nifty Bank")
    dim_result = kronos_dimension_result(signal_type="BUY_CE", forecast_summary=summary)
    # Pass dim_result into ConfluenceEngine.evaluate() alongside other dimensions.
"""
from __future__ import annotations

from typing import Any

from loguru import logger

from src.strategy.confluence_engine import ConfluenceDimension, DimensionResult

# Default weight matches DEFAULT_WEIGHTS[MODEL_FORECAST] in confluence_engine.py
_DEFAULT_WEIGHT: float = 0.055


def kronos_dimension_result(
    signal_type: str,
    forecast_summary: dict[str, Any] | None,
    weight: float = _DEFAULT_WEIGHT,
) -> DimensionResult:
    """Score Kronos forecast vs proposed signal direction.

    Converts the Kronos directional forecast into a 0-1 score that is
    incorporated into the ConfluenceEngine composite score as the
    ``MODEL_FORECAST`` dimension.

    Score logic:
      - signal_dir matches predicted_dir → score = 0.5 + 0.5 * magnitude
        where magnitude = min(abs(pred_change_pct) / 1.0, 1.0).
        A 0% change → 0.5 (weak agreement); ≥1% change → 1.0 (strong).
      - Opposite direction → score = 0.5 - 0.5 * magnitude (symmetric penalty).
      - Flat prediction or unknown signal_type → 0.5 (neutral, no penalty).
      - No forecast available → skipped=True (excluded from composite score).

    Signal-to-direction mapping:
      BUY / BUY_CE  → expects UP   (long delta positions)
      BUY_PE        → expects DOWN (put options = bearish bet)
      SELL / SELL_CE → expects DOWN
      SELL_PE       → expects UP   (short puts = bullish bias)
      STRADDLE      → expects FLAT (low-volatility play)

    Args:
        signal_type: One of BUY / SELL / BUY_CE / BUY_PE / SELL_CE / SELL_PE
                     / STRADDLE (case-insensitive).
        forecast_summary: Output of ``KronosForecaster.predict_summary()``.
            Expected keys:
                - ``predicted_direction`` (str): "UP" | "DOWN" | "FLAT"
                - ``predicted_change_pct`` (float): expected % move magnitude
                - ``predicted_range_pct`` (float, optional): intrabar range
            When ``None`` or empty, the dimension is skipped entirely.
        weight: Contribution weight for this dimension in the composite score.
            Defaults to 0.055 (shadow validation weight).

    Returns:
        DimensionResult for ConfluenceDimension.MODEL_FORECAST.
    """
    # --- no forecast available: skip cleanly so it doesn't penalise score ---
    if not forecast_summary:
        logger.debug("kronos_dimension: no forecast available — skipping MODEL_FORECAST")
        return DimensionResult(
            dimension=ConfluenceDimension.MODEL_FORECAST,
            score=0.0,
            weight=weight,
            skipped=True,
            failed=["no_forecast_available"],
        )

    pred_dir: str = forecast_summary.get("predicted_direction", "FLAT")
    pred_change_pct: float = float(forecast_summary.get("predicted_change_pct", 0.0))

    # --- map signal_type → expected directional bias ---
    _signal_to_expected: dict[str, str] = {
        "BUY":      "UP",
        "BUY_CE":   "UP",
        "BUY_PE":   "DOWN",
        "SELL":     "DOWN",
        "SELL_CE":  "DOWN",
        "SELL_PE":  "UP",
        "STRADDLE": "FLAT",
    }
    normalised_signal = str(signal_type).upper()
    expected = _signal_to_expected.get(normalised_signal)

    if expected is None:
        logger.warning(
            "kronos_dimension: unrecognised signal_type=%r — returning neutral 0.5",
            signal_type,
        )
        return DimensionResult(
            dimension=ConfluenceDimension.MODEL_FORECAST,
            score=0.5,
            weight=weight,
            skipped=False,
            matched=[f"unknown_signal_type:{signal_type}"],
        )

    # --- flat Kronos prediction: no directional info ---
    if pred_dir == "FLAT":
        logger.debug("kronos_dimension: predicted FLAT — returning neutral 0.5")
        return DimensionResult(
            dimension=ConfluenceDimension.MODEL_FORECAST,
            score=0.5,
            weight=weight,
            skipped=False,
            matched=["kronos_flat_pred"],
        )

    # --- score based on alignment ---
    # Magnitude capped at 1.0 (i.e. 1% move = full conviction)
    magnitude = min(abs(pred_change_pct) / 1.0, 1.0)

    if pred_dir == expected:
        # Aligned: weak → 0.5, strong → 1.0
        score = 0.5 + 0.5 * magnitude
        return DimensionResult(
            dimension=ConfluenceDimension.MODEL_FORECAST,
            score=score,
            weight=weight,
            skipped=False,
            matched=[f"kronos_{pred_dir.lower()}_pct={pred_change_pct:+.2f}"],
        )
    else:
        # Opposite direction: weak penalty → 0.5, strong penalty → 0.0
        score = 0.5 - 0.5 * magnitude
        return DimensionResult(
            dimension=ConfluenceDimension.MODEL_FORECAST,
            score=score,
            weight=weight,
            skipped=False,
            failed=[f"kronos_{pred_dir.lower()}_vs_{expected.lower()}_signal"],
        )
