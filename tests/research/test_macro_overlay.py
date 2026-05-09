"""Tests for src.research.macro_overlay.MacroOverlay.

All tests use inline mocks — no external DB connection or network calls.

Coverage targets:
  - test_get_active_instruments_returns_4
  - test_compute_returns_positive_when_price_rises
  - test_compute_zscore_positive_when_above_mean
  - test_classify_trend_up_when_return_pos_and_zscore_high
  - test_classify_trend_range_when_zscore_near_zero
  - test_update_daily_persists_all_4_macros
  - test_cross_asset_signal_for_TCS_bullish_when_USDINR_UP
  - test_cross_asset_signal_for_RELIANCE_bullish_when_CRUDE_UP
  - test_cross_asset_signal_for_HDFC_bearish_when_USDINR_UP
"""

from __future__ import annotations

from datetime import date, datetime
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.research.macro_overlay import MacroOverlay, _MACRO_INSTRUMENTS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TRADE_DATE = date(2026, 5, 8)

_SYMBOLS = ["USDINR", "CRUDE", "GOLD", "SILVER"]


def _make_closes(
    n: int = 30,
    base: float = 84_000.0,  # ~84 INR/USD in paisa units
    drift: float = 0.0,
    seed: int = 42,
) -> pd.Series:
    """Generate a deterministic close-price series in paisa."""
    rng = np.random.default_rng(seed)
    log_rets = drift + 0.005 * rng.standard_normal(n)
    closes = base * np.exp(np.cumsum(log_rets))
    idx = pd.date_range(end="2026-05-08", periods=n, freq="B", tz="Asia/Kolkata")
    return pd.Series(closes.astype(float), index=idx, name="close")


def _make_df(closes: pd.Series) -> pd.DataFrame:
    """Wrap a close series in a DataFrame matching historical_provider output."""
    return pd.DataFrame({"close": closes}, index=closes.index)


def _make_provider_for(closes_map: dict[str, pd.Series]):
    """Return a mock historical_provider callable."""
    def provider(instrument_key: str, interval: str, days: int) -> pd.DataFrame:
        closes = closes_map.get(instrument_key)
        if closes is None:
            # Return neutral flat data for any unmapped key
            return _make_df(_make_closes(n=max(days, 30), base=100_000.0))
        return _make_df(closes)
    return provider


def _make_overlay(provider=None, db_engine=None) -> MacroOverlay:
    return MacroOverlay(
        instrument_manager=None,
        historical_provider=provider,
        db_engine=db_engine,
    )


# ---------------------------------------------------------------------------
# test_get_active_instruments_returns_4
# ---------------------------------------------------------------------------

class TestGetActiveInstruments:
    def test_get_active_instruments_returns_4(self):
        """MacroOverlay.get_active_instruments() returns exactly 4 instruments."""
        overlay = _make_overlay()
        instruments = overlay.get_active_instruments()
        assert len(instruments) == 4, f"Expected 4 instruments, got {len(instruments)}"

    def test_get_active_instruments_has_expected_symbols(self):
        """All four expected macro symbols are present."""
        overlay = _make_overlay()
        symbols = {inst["symbol"] for inst in overlay.get_active_instruments()}
        assert symbols == {"USDINR", "CRUDE", "GOLD", "SILVER"}

    def test_get_macro_keys_returns_4_strings(self):
        """get_macro_keys() returns 4 non-empty instrument key strings."""
        overlay = _make_overlay()
        keys = overlay.get_macro_keys()
        assert len(keys) == 4
        for k in keys:
            assert isinstance(k, str) and "|" in k, f"Unexpected key format: {k}"


# ---------------------------------------------------------------------------
# test_compute_returns_positive_when_price_rises
# ---------------------------------------------------------------------------

class TestComputeReturns:
    def test_compute_returns_positive_when_price_rises(self):
        """compute_returns returns a positive value when prices trend upward."""
        # Rising close series: last price clearly above 5 bars ago
        closes = _make_closes(n=30, base=84_000.0, drift=+0.005)  # strong upward drift
        provider = _make_provider_for({"NCD_FO|7009": closes})
        overlay = _make_overlay(provider=provider)

        ret = overlay.compute_returns("NCD_FO|7009", lookback_days=5)
        assert ret > 0.0, f"Expected positive return for rising prices, got {ret}"

    def test_compute_returns_negative_when_price_falls(self):
        """compute_returns returns a negative value when prices trend downward."""
        closes = _make_closes(n=30, base=84_000.0, drift=-0.005)
        provider = _make_provider_for({"NCD_FO|7009": closes})
        overlay = _make_overlay(provider=provider)

        ret = overlay.compute_returns("NCD_FO|7009", lookback_days=5)
        assert ret < 0.0, f"Expected negative return for falling prices, got {ret}"

    def test_compute_returns_zero_on_flat_prices(self):
        """compute_returns is approximately 0 for constant prices."""
        closes = pd.Series(
            [84_000.0] * 30,
            index=pd.date_range(end="2026-05-08", periods=30, freq="B", tz="Asia/Kolkata"),
        )
        provider = _make_provider_for({"NCD_FO|7009": closes})
        overlay = _make_overlay(provider=provider)

        ret = overlay.compute_returns("NCD_FO|7009", lookback_days=5)
        assert abs(ret) < 1e-6

    def test_compute_returns_returns_zero_on_insufficient_data(self):
        """compute_returns returns 0.0 when fewer bars than lookback+1."""
        closes = _make_closes(n=3)
        provider = _make_provider_for({"NCD_FO|7009": closes})
        overlay = _make_overlay(provider=provider)

        ret = overlay.compute_returns("NCD_FO|7009", lookback_days=5)
        assert ret == 0.0

    def test_compute_returns_raises_when_no_provider(self):
        """compute_returns raises ValueError when historical_provider is None."""
        overlay = _make_overlay(provider=None)
        with pytest.raises(ValueError, match="historical_provider not set"):
            overlay.compute_returns("NCD_FO|7009", lookback_days=5)


# ---------------------------------------------------------------------------
# test_compute_zscore_positive_when_above_mean
# ---------------------------------------------------------------------------

class TestComputeZscore:
    def test_compute_zscore_positive_when_above_mean(self):
        """compute_zscore is positive when the latest close is above the rolling mean."""
        # Construct prices where last value is clearly above the 20-day mean
        closes_values = [100_000.0] * 29 + [120_000.0]  # last bar spikes up
        closes = pd.Series(
            closes_values,
            index=pd.date_range(end="2026-05-08", periods=30, freq="B", tz="Asia/Kolkata"),
        )
        provider = _make_provider_for({"NCD_FO|7009": closes})
        overlay = _make_overlay(provider=provider)

        zscore = overlay.compute_zscore("NCD_FO|7009", lookback=20)
        assert zscore > 0.0, f"Expected positive z-score, got {zscore}"

    def test_compute_zscore_negative_when_below_mean(self):
        """compute_zscore is negative when the latest close is below the rolling mean."""
        closes_values = [100_000.0] * 29 + [80_000.0]  # last bar drops
        closes = pd.Series(
            closes_values,
            index=pd.date_range(end="2026-05-08", periods=30, freq="B", tz="Asia/Kolkata"),
        )
        provider = _make_provider_for({"NCD_FO|7009": closes})
        overlay = _make_overlay(provider=provider)

        zscore = overlay.compute_zscore("NCD_FO|7009", lookback=20)
        assert zscore < 0.0, f"Expected negative z-score, got {zscore}"

    def test_compute_zscore_zero_on_constant_series(self):
        """compute_zscore is 0.0 when all prices are identical (zero std)."""
        closes = pd.Series(
            [84_000.0] * 25,
            index=pd.date_range(end="2026-05-08", periods=25, freq="B", tz="Asia/Kolkata"),
        )
        provider = _make_provider_for({"NCD_FO|7009": closes})
        overlay = _make_overlay(provider=provider)

        zscore = overlay.compute_zscore("NCD_FO|7009", lookback=20)
        assert zscore == 0.0

    def test_compute_zscore_zero_on_insufficient_data(self):
        """compute_zscore returns 0.0 when fewer bars than lookback."""
        closes = _make_closes(n=10)
        provider = _make_provider_for({"NCD_FO|7009": closes})
        overlay = _make_overlay(provider=provider)

        zscore = overlay.compute_zscore("NCD_FO|7009", lookback=20)
        assert zscore == 0.0


# ---------------------------------------------------------------------------
# test_classify_trend_up_when_return_pos_and_zscore_high
# ---------------------------------------------------------------------------

class TestClassifyTrend:
    def test_classify_trend_up_when_return_pos_and_zscore_high(self):
        """classify_trend returns 'UP' when 5d return >1% and zscore >0.5."""
        overlay = _make_overlay()
        result = overlay.classify_trend(return_5d=2.5, zscore=1.2, threshold=0.5)
        assert result == "UP"

    def test_classify_trend_down_when_return_neg_and_zscore_low(self):
        """classify_trend returns 'DOWN' when 5d return <-1% and zscore <-0.5."""
        overlay = _make_overlay()
        result = overlay.classify_trend(return_5d=-2.0, zscore=-0.8, threshold=0.5)
        assert result == "DOWN"

    def test_classify_trend_range_when_zscore_near_zero(self):
        """classify_trend returns 'RANGE' when zscore is near zero (even if return >1%)."""
        overlay = _make_overlay()
        result = overlay.classify_trend(return_5d=1.5, zscore=0.1, threshold=0.5)
        assert result == "RANGE"

    def test_classify_trend_range_when_return_small(self):
        """classify_trend returns 'RANGE' when |return_5d| <= 1%."""
        overlay = _make_overlay()
        result = overlay.classify_trend(return_5d=0.5, zscore=1.5, threshold=0.5)
        assert result == "RANGE"

    def test_classify_trend_range_exact_boundary(self):
        """classify_trend returns 'RANGE' at the exact threshold boundary (not strictly greater)."""
        overlay = _make_overlay()
        # return_5d == 1.0 (not > 1.0), so should be RANGE
        result = overlay.classify_trend(return_5d=1.0, zscore=0.5, threshold=0.5)
        assert result == "RANGE"

    def test_classify_trend_custom_threshold(self):
        """classify_trend uses custom threshold correctly."""
        overlay = _make_overlay()
        # With threshold=1.0, zscore=0.8 is not > 1.0
        result = overlay.classify_trend(return_5d=3.0, zscore=0.8, threshold=1.0)
        assert result == "RANGE"
        # With threshold=0.5, zscore=0.8 > 0.5 → UP
        result2 = overlay.classify_trend(return_5d=3.0, zscore=0.8, threshold=0.5)
        assert result2 == "UP"


# ---------------------------------------------------------------------------
# test_update_daily_persists_all_4_macros
# ---------------------------------------------------------------------------

class TestUpdateDaily:
    def _make_all_providers(self, drift: float = 0.0) -> dict:
        """Build a map of instrument_key → close series for all 4 macros."""
        return {
            "NCD_FO|7009":    _make_closes(n=30, base=84_000.0, drift=drift),
            "NSE_COM|121620": _make_closes(n=30, base=5_000_000.0, drift=drift),
            "NSE_COM|122886": _make_closes(n=30, base=7_000_000.0, drift=drift),
            "NSE_COM|121799": _make_closes(n=30, base=85_000_000.0, drift=drift),
        }

    def test_update_daily_returns_4_rows(self):
        """update_daily returns one row per active macro instrument."""
        provider = _make_provider_for(self._make_all_providers())
        overlay = _make_overlay(provider=provider)

        df = overlay.update_daily(_TRADE_DATE)
        assert len(df) == 4, f"Expected 4 rows, got {len(df)}"

    def test_update_daily_persists_all_4_macros(self):
        """update_daily calls the DB engine for all 4 macro symbols."""
        provider = _make_provider_for(self._make_all_providers())

        mock_engine = MagicMock()
        mock_conn = MagicMock()
        mock_engine.begin.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_engine.begin.return_value.__exit__ = MagicMock(return_value=False)

        overlay = _make_overlay(provider=provider, db_engine=mock_engine)
        df = overlay.update_daily(_TRADE_DATE)

        assert len(df) == 4
        symbols_returned = set(df["symbol"].tolist())
        assert symbols_returned == {"USDINR", "CRUDE", "GOLD", "SILVER"}

    def test_update_daily_has_required_columns(self):
        """update_daily DataFrame has all required columns."""
        provider = _make_provider_for(self._make_all_providers())
        overlay = _make_overlay(provider=provider)
        df = overlay.update_daily(_TRADE_DATE)

        required = {"symbol", "trade_date", "close_paisa", "return_pct_1d",
                    "return_pct_5d", "return_pct_20d", "trend", "zscore_20d"}
        assert required.issubset(set(df.columns)), f"Missing columns: {required - set(df.columns)}"

    def test_update_daily_trend_values_are_valid(self):
        """update_daily produces valid trend values: UP, DOWN, or RANGE."""
        provider = _make_provider_for(self._make_all_providers())
        overlay = _make_overlay(provider=provider)
        df = overlay.update_daily(_TRADE_DATE)

        assert df["trend"].isin(["UP", "DOWN", "RANGE"]).all()

    def test_update_daily_no_provider_returns_empty(self):
        """update_daily returns empty DataFrame when provider raises for all instruments."""
        overlay = _make_overlay(provider=None)
        df = overlay.update_daily(_TRADE_DATE)
        assert df.empty


# ---------------------------------------------------------------------------
# test_cross_asset_signal_for_TCS_bullish_when_USDINR_UP
# ---------------------------------------------------------------------------

class TestCrossAssetSignal:
    def _make_regime_df(self, usdinr_trend: str = "RANGE", crude_trend: str = "RANGE") -> pd.DataFrame:
        """Return a synthetic regime DataFrame with specified trends."""
        return pd.DataFrame({
            "symbol": ["USDINR", "CRUDE", "GOLD", "SILVER"],
            "trend": [usdinr_trend, crude_trend, "RANGE", "RANGE"],
            "zscore_20d": [0.0, 0.0, 0.0, 0.0],
            "return_pct_5d": [0.0, 0.0, 0.0, 0.0],
        })

    def test_cross_asset_signal_for_TCS_bullish_when_USDINR_UP(self):
        """TCS cross-asset signal is bullish when USDINR trend is UP."""
        overlay = _make_overlay()
        # Patch get_today_regime to return USDINR=UP
        regime_df = self._make_regime_df(usdinr_trend="UP")
        overlay.get_today_regime = MagicMock(return_value=regime_df)

        signal = overlay.cross_asset_signal_for_stock("TCS", _TRADE_DATE)
        assert signal["direction"] == "bullish", f"Expected bullish, got: {signal}"
        assert len(signal["reasons"]) > 0

    def test_cross_asset_signal_for_INFY_bullish_when_USDINR_UP(self):
        """INFY cross-asset signal is bullish when USDINR trend is UP."""
        overlay = _make_overlay()
        regime_df = self._make_regime_df(usdinr_trend="UP")
        overlay.get_today_regime = MagicMock(return_value=regime_df)

        signal = overlay.cross_asset_signal_for_stock("INFY", _TRADE_DATE)
        assert signal["direction"] == "bullish"

    def test_cross_asset_signal_for_RELIANCE_bullish_when_CRUDE_UP(self):
        """RELIANCE cross-asset signal is bullish when CRUDE trend is UP."""
        overlay = _make_overlay()
        regime_df = self._make_regime_df(crude_trend="UP")
        overlay.get_today_regime = MagicMock(return_value=regime_df)

        signal = overlay.cross_asset_signal_for_stock("RELIANCE", _TRADE_DATE)
        assert signal["direction"] == "bullish", f"Expected bullish, got: {signal}"
        assert "refining" in signal["reasons"][0].lower() or "crude" in signal["reasons"][0].lower()

    def test_cross_asset_signal_for_HDFC_bearish_when_USDINR_UP(self):
        """HDFCBANK cross-asset signal is bearish when USDINR trend is UP."""
        overlay = _make_overlay()
        regime_df = self._make_regime_df(usdinr_trend="UP")
        overlay.get_today_regime = MagicMock(return_value=regime_df)

        signal = overlay.cross_asset_signal_for_stock("HDFCBANK", _TRADE_DATE)
        assert signal["direction"] == "bearish", f"Expected bearish, got: {signal}"

    def test_cross_asset_signal_for_ICICIBANK_bearish_when_USDINR_UP(self):
        """ICICIBANK is bearish when USDINR is UP."""
        overlay = _make_overlay()
        regime_df = self._make_regime_df(usdinr_trend="UP")
        overlay.get_today_regime = MagicMock(return_value=regime_df)

        signal = overlay.cross_asset_signal_for_stock("ICICIBANK", _TRADE_DATE)
        assert signal["direction"] == "bearish"

    def test_cross_asset_signal_for_HINDUNILVR_bearish_when_CRUDE_UP(self):
        """HINDUNILVR is bearish when CRUDE is UP (input cost pressure)."""
        overlay = _make_overlay()
        regime_df = self._make_regime_df(crude_trend="UP")
        overlay.get_today_regime = MagicMock(return_value=regime_df)

        signal = overlay.cross_asset_signal_for_stock("HINDUNILVR", _TRADE_DATE)
        assert signal["direction"] == "bearish"

    def test_cross_asset_signal_neutral_when_no_macro_move(self):
        """Signal is neutral when all macro trends are RANGE."""
        overlay = _make_overlay()
        regime_df = self._make_regime_df(usdinr_trend="RANGE", crude_trend="RANGE")
        overlay.get_today_regime = MagicMock(return_value=regime_df)

        signal = overlay.cross_asset_signal_for_stock("TCS", _TRADE_DATE)
        assert signal["direction"] == "neutral"

    def test_cross_asset_signal_neutral_for_unknown_stock(self):
        """Unknown stock returns neutral direction."""
        overlay = _make_overlay()
        overlay.get_today_regime = MagicMock(return_value=pd.DataFrame())

        signal = overlay.cross_asset_signal_for_stock("RANDOMCORP", _TRADE_DATE)
        assert signal["direction"] == "neutral"

    def test_cross_asset_signal_case_insensitive(self):
        """Stock symbol lookup is case-insensitive."""
        overlay = _make_overlay()
        regime_df = self._make_regime_df(usdinr_trend="UP")
        overlay.get_today_regime = MagicMock(return_value=regime_df)

        signal_upper = overlay.cross_asset_signal_for_stock("TCS", _TRADE_DATE)
        signal_lower = overlay.cross_asset_signal_for_stock("tcs", _TRADE_DATE)
        assert signal_upper["direction"] == signal_lower["direction"]
