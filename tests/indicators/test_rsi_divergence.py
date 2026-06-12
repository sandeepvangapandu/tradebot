from __future__ import annotations

import pandas as pd

from src.indicators import divergence


class _FakeRSIIndicator:
    def __init__(self, close: pd.Series, window: int) -> None:
        self._index = close.index

    def rsi(self) -> pd.Series:
        values = [55, 52, 50, 45, 40, 30, 35, 45, 40, 36, 42, 47, 50, 52]
        return pd.Series(values, index=self._index, dtype="float64")


def test_detect_rsi_divergence_uses_rsi_at_price_swings(monkeypatch) -> None:
    """A lower price low with higher RSI at the swing should be bullish."""
    monkeypatch.setattr(
        divergence.ta_lib.momentum,
        "RSIIndicator",
        _FakeRSIIndicator,
    )
    close = [101, 100, 100, 99, 98, 96, 98, 99, 97, 94, 96, 97, 98, 99]
    df = pd.DataFrame(
        {
            "open": close,
            "high": [value + 1 for value in close],
            "low": [value - 1 for value in close],
            "close": close,
            "volume": [100] * len(close),
        }
    )

    result = divergence.detect_rsi_divergence(
        df,
        lookback=12,
        rsi_length=2,
        min_swing_pct=0.001,
        order=2,
    )

    assert result.type == "bullish"
    assert result.rsi_swing_low == 36
    assert result.price_swing_low == 94
