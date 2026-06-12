from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd
from loguru import logger

from src.strategy.builder import Condition
from src.strategy.conditions import ConditionEvaluator


def test_supertrend_short_window_is_warmup_not_error() -> None:
    """Short Supertrend windows should not be logged as indicator failures."""
    now = datetime(2026, 6, 1, 9, 15)
    index = [now + timedelta(minutes=i) for i in range(4)]
    bars = {
        "NSE_INDEX|Nifty Bank": pd.DataFrame(
            {
                "open": [100_00, 101_00, 102_00, 103_00],
                "high": [101_00, 102_00, 103_00, 104_00],
                "low": [99_00, 100_00, 101_00, 102_00],
                "close": [100_50, 101_50, 102_50, 103_50],
                "volume": [0, 0, 0, 0],
            },
            index=pd.DatetimeIndex(index),
        )
    }
    condition = Condition(
        indicator="Supertrend",
        comparison=">",
        value=0,
        parameters={"period": 10, "multiplier": 3.0},
    )

    messages: list[str] = []
    sink_id = logger.add(lambda msg: messages.append(str(msg)), level="DEBUG")
    try:
        assert ConditionEvaluator().evaluate(condition, bars, "NSE_INDEX|Nifty Bank") is False
    finally:
        logger.remove(sink_id)

    assert not any("Failed to compute indicator 'Supertrend'" in message for message in messages)


def test_missing_support_level_does_not_pass_proximity_gate() -> None:
    """No detected support level should not look like zero-distance support."""
    df = pd.DataFrame(
        {
            "open": [100_00] * 30,
            "high": [100_10] * 30,
            "low": [99_90] * 30,
            "close": [100_00] * 30,
            "volume": [0] * 30,
        },
        index=pd.date_range(
            "2026-01-01 09:15",
            periods=30,
            freq="5min",
            tz="Asia/Kolkata",
        ),
    )

    result = ConditionEvaluator().evaluate(
        Condition(
            indicator="Support_Proximity",
            comparison="<",
            value=0.3,
            parameters={"lookback": 20, "proximity_pct": 0.3},
        ),
        {"NSE_INDEX|Nifty Bank": df},
        "NSE_INDEX|Nifty Bank",
    )

    assert result is False
