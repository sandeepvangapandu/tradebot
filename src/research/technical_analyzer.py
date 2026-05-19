"""Technical analysis module for the AI Trade Research Engine.

Provides multi-timeframe technical analysis including trend alignment,
momentum, support/resistance, and candlestick pattern analysis.

All prices are in PAISA (integer) - 1 Rupee = 100 paisa.
All timestamps are in IST (Asia/Kolkata).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING
from zoneinfo import ZoneInfo

import pandas as pd
import ta as ta_lib
from loguru import logger

from src.research.models import (
    AnalysisComponent,
    MomentumData,
    SupportResistanceData,
    TrendAlignmentData,
)
from src.strategy.indicators import IndicatorEngine
from src.utils.degradation import graceful_degrade

if TYPE_CHECKING:
    from datetime import datetime

IST = ZoneInfo("Asia/Kolkata")


@dataclass
class CandleContextData:
    """Data container for candlestick context analysis."""

    last_n_candles_confirming: int = 0
    rejection_wicks_present: bool = False
    avg_body_to_wick_ratio: float = 0.0
    gap_up: bool = False
    gap_down: bool = False
    bullish_engulfing: bool = False
    bearish_engulfing: bool = False
    doji_present: bool = False
    consecutive_bullish: int = 0
    consecutive_bearish: int = 0


class TechnicalAnalyzer:
    """Multi-timeframe technical analyzer for trade signal validation.

    Analyzes trend alignment across timeframes, momentum conditions,
    support/resistance levels, and candlestick patterns to provide
    a comprehensive technical score for trade decisions.

    All prices are in PAISA (integer).
    Timestamps are in IST.
    """

    # Timeframe weights for trend alignment
    TIMEFRAME_WEIGHTS = {
        "1min": 0.10,
        "5min": 0.20,
        "15min": 0.30,
        "daily": 0.40,
    }

    def __init__(self):
        """Initialize the technical analyzer."""
        self._cache: dict[str, dict] = {}

    def _get_engine(self, df: pd.DataFrame) -> IndicatorEngine:
        """Create or update an IndicatorEngine from DataFrame.

        Args:
            df: OHLCV DataFrame with prices in paisa.

        Returns:
            IndicatorEngine instance.
        """
        return IndicatorEngine(df)

    def _calculate_adx(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate ADX (Average Directional Index).

        Args:
            df: OHLCV DataFrame.
            period: ADX period. Default 14.

        Returns:
            Series with ADX values.
        """
        try:
            return ta_lib.trend.ADXIndicator(df["high"], df["low"], df["close"], window=period).adx()
        except Exception:
            return pd.Series(index=df.index, data=0.0)

    def _determine_trend(
        self,
        df: pd.DataFrame,
        fast_period: int = 20,
        slow_period: int = 50,
    ) -> tuple[str, float]:
        """Determine trend direction using EMA and Supertrend.

        Args:
            df: OHLCV DataFrame.
            fast_period: Fast EMA period. Default 20.
            slow_period: Slow EMA period. Default 50.

        Returns:
            Tuple of (trend_direction, adx_value).
            Trend direction: "uptrend", "downtrend", or "neutral".
        """
        if len(df) < slow_period:
            return "neutral", 0.0

        engine = self._get_engine(df)

        # EMA analysis
        ema_fast = engine.ema(fast_period)
        ema_slow = engine.sma(slow_period)
        vwap = engine.vwap()

        # Get latest values
        latest_close = df["close"].iloc[-1]
        latest_fast = ema_fast.iloc[-1] if not ema_fast.empty else latest_close
        latest_slow = ema_slow.iloc[-1] if not ema_slow.empty else latest_close
        latest_vwap = vwap.iloc[-1] if not vwap.empty else latest_close

        # Supertrend
        try:
            st_result = engine.supertrend(period=7, multiplier=3.0)
            st_direction = st_result["direction"].iloc[-1]
        except Exception:
            st_direction = 0

        # ADX for trend strength
        adx_series = self._calculate_adx(df)
        adx_value = adx_series.iloc[-1] if not adx_series.empty else 0.0

        # Determine trend
        ema_bullish = latest_fast > latest_slow
        ema_bearish = latest_fast < latest_slow
        above_vwap = latest_close > latest_vwap
        below_vwap = latest_close < latest_vwap

        # Strong uptrend: EMA bullish, above VWAP, supertrend up
        if ema_bullish and above_vwap and st_direction == 1:
            return "uptrend", adx_value

        # Strong downtrend: EMA bearish, below VWAP, supertrend down
        if ema_bearish and below_vwap and st_direction == -1:
            return "downtrend", adx_value

        # Weak uptrend: majority of conditions met
        if (ema_bullish and above_vwap) or (ema_bullish and st_direction == 1) or (above_vwap and st_direction == 1):
            return "weak_uptrend", adx_value

        # Weak downtrend
        if (ema_bearish and below_vwap) or (ema_bearish and st_direction == -1) or (below_vwap and st_direction == -1):
            return "weak_downtrend", adx_value

        return "neutral", adx_value

    @graceful_degrade(
        default_return=AnalysisComponent(
            name="trend_alignment",
            score=50.0,
            weight=0.30,
            confidence=30.0,
            reasoning="Trend alignment analysis failed - using default neutral score",
            data={"score": 50, "confidence": 0.3, "regime": "unknown"},
        )
    )
    def analyze_trend_alignment(
        self,
        bars_dict: dict[str, pd.DataFrame],
        signal_direction: str,
    ) -> AnalysisComponent:
        """Analyze trend alignment across multiple timeframes.

        Args:
            bars_dict: Dictionary mapping timeframe strings to OHLCV DataFrames.
                       Expected keys: "1min", "5min", "15min", "daily".
            signal_direction: "BUY" or "SELL".

        Returns:
            AnalysisComponent with score 0-100 and TrendAlignmentData.
        """
        trend_data = TrendAlignmentData()
        scores: dict[str, float] = {}

        for tf, weight in self.TIMEFRAME_WEIGHTS.items():
            if tf not in bars_dict or bars_dict[tf].empty:
                scores[tf] = 50.0  # Neutral if no data
                continue

            df = bars_dict[tf]
            trend, adx = self._determine_trend(df)

            # Store trend data
            if tf == "1min":
                trend_data.trend_1min = trend
                trend_data.adx_1min = adx
            elif tf == "5min":
                trend_data.trend_5min = trend
                trend_data.adx_5min = adx
            elif tf == "15min":
                trend_data.trend_15min = trend
                trend_data.adx_15min = adx
            elif tf == "daily":
                trend_data.trend_daily = trend
                trend_data.adx_daily = adx

            # Score this timeframe
            if signal_direction == "BUY":
                if trend == "uptrend":
                    tf_score = 100.0
                elif trend == "weak_uptrend":
                    tf_score = 75.0
                elif trend == "neutral":
                    tf_score = 50.0
                elif trend == "weak_downtrend":
                    tf_score = 25.0
                else:  # downtrend
                    tf_score = 0.0
            else:  # SELL
                if trend == "downtrend":
                    tf_score = 100.0
                elif trend == "weak_downtrend":
                    tf_score = 75.0
                elif trend == "neutral":
                    tf_score = 50.0
                elif trend == "weak_uptrend":
                    tf_score = 25.0
                else:  # uptrend
                    tf_score = 0.0

            # Adjust score based on ADX (trend strength)
            if adx > 40:
                tf_score = min(100.0, tf_score * 1.1)  # Boost strong trends
            elif adx < 20:
                tf_score = tf_score * 0.8  # Penalize weak trends

            scores[tf] = tf_score

        # Calculate weighted score
        final_score = sum(
            scores.get(tf, 50.0) * weight
            for tf, weight in self.TIMEFRAME_WEIGHTS.items()
        )

        # Build reasoning
        aligned_timeframes = [
            tf for tf, score in scores.items()
            if (signal_direction == "BUY" and score >= 75) or
               (signal_direction == "SELL" and score >= 75)
        ]

        reasoning = (
            f"Trend alignment for {signal_direction}: "
            f"1min={trend_data.trend_1min}({trend_data.adx_1min:.1f}), "
            f"5min={trend_data.trend_5min}({trend_data.adx_5min:.1f}), "
            f"15min={trend_data.trend_15min}({trend_data.adx_15min:.1f}), "
            f"daily={trend_data.trend_daily}({trend_data.adx_daily:.1f}). "
            f"Aligned timeframes: {', '.join(aligned_timeframes) if aligned_timeframes else 'none'}."
        )

        return AnalysisComponent(
            name="trend_alignment",
            score=final_score,
            weight=0.30,
            confidence=min(100.0, (trend_data.adx_15min + trend_data.adx_daily) / 2),
            reasoning=reasoning,
            data=trend_data.model_dump(),
        )

    @graceful_degrade(
        default_return=AnalysisComponent(
            name="momentum",
            score=50.0,
            weight=0.25,
            confidence=30.0,
            reasoning="Momentum analysis failed - using default neutral score",
            data={"score": 50, "confidence": 0.3, "regime": "neutral"},
        )
    )
    def analyze_momentum(
        self,
        df: pd.DataFrame,
        signal_direction: str,
    ) -> AnalysisComponent:
        """Analyze momentum conditions for the signal.

        Args:
            df: OHLCV DataFrame with prices in paisa.
            signal_direction: "BUY" or "SELL".

        Returns:
            AnalysisComponent with score 0-100 and MomentumData.
        """
        if df.empty or len(df) < 20:
            return AnalysisComponent(
                name="momentum",
                score=50.0,
                weight=0.25,
                confidence=0.0,
                reasoning="Insufficient data for momentum analysis",
                data={},
            )

        engine = self._get_engine(df)
        momentum_data = MomentumData()

        # RSI Analysis
        rsi_series = engine.rsi(period=14)
        if not rsi_series.empty:
            momentum_data.rsi_current = rsi_series.iloc[-1]
            # RSI trend (comparing last 3 values)
            if len(rsi_series) >= 3:
                rsi_recent = rsi_series.iloc[-3:].values
                if rsi_recent[-1] > rsi_recent[0]:
                    momentum_data.rsi_trend = "rising"
                elif rsi_recent[-1] < rsi_recent[0]:
                    momentum_data.rsi_trend = "falling"
                else:
                    momentum_data.rsi_trend = "flat"

        # MACD Analysis
        try:
            macd_result = engine.macd(12, 26, 9)
            hist_series = macd_result["histogram"]
            if not hist_series.empty:
                momentum_data.macd_histogram = hist_series.iloc[-1]
                # Histogram trend
                if len(hist_series) >= 3:
                    hist_recent = hist_series.iloc[-3:].values
                    if hist_recent[-1] > hist_recent[0]:
                        momentum_data.macd_histogram_trend = "rising"
                    elif hist_recent[-1] < hist_recent[0]:
                        momentum_data.macd_histogram_trend = "falling"
                    else:
                        momentum_data.macd_histogram_trend = "flat"
        except Exception:
            pass

        # Stochastic Analysis
        try:
            stoch = ta_lib.momentum.StochasticOscillator(df["high"], df["low"], df["close"], window=14, smooth_window=3)
            momentum_data.stochastic_k = stoch.stoch().iloc[-1]
            momentum_data.stochastic_d = stoch.stoch_signal().iloc[-1]
        except Exception:
            pass

        # Rate of Change
        try:
            roc_series = ta_lib.momentum.ROCIndicator(df["close"], window=10).roc()
            momentum_data.roc_10 = roc_series.iloc[-1]
        except Exception:
            pass

        # Divergence Detection (price vs RSI)
        momentum_data.divergence_detected, momentum_data.divergence_type = (
            self._detect_divergence(df, rsi_series, signal_direction)
        )

        # Calculate momentum score
        score = self._calculate_momentum_score(momentum_data, signal_direction)

        # Build reasoning
        reasoning = (
            f"RSI: {momentum_data.rsi_current:.1f} ({momentum_data.rsi_trend}), "
            f"MACD hist: {momentum_data.macd_histogram:.0f} ({momentum_data.macd_histogram_trend}), "
            f"Stoch: {momentum_data.stochastic_k:.1f}/{momentum_data.stochastic_d:.1f}. "
        )
        if momentum_data.divergence_detected:
            reasoning += f"Divergence: {momentum_data.divergence_type}. "

        return AnalysisComponent(
            name="momentum",
            score=score,
            weight=0.25,
            confidence=80.0 if not momentum_data.divergence_detected else 60.0,
            reasoning=reasoning,
            data=momentum_data.model_dump(),
        )

    def _detect_divergence(
        self,
        df: pd.DataFrame,
        rsi_series: pd.Series,
        signal_direction: str,
    ) -> tuple[bool, str | None]:
        """Detect bullish or bearish divergence between price and RSI.

        Args:
            df: OHLCV DataFrame.
            rsi_series: RSI values.
            signal_direction: "BUY" or "SELL".

        Returns:
            Tuple of (divergence_detected, divergence_type).
        """
        if len(df) < 10 or rsi_series.empty:
            return False, None

        # Get last 10 periods
        closes = df["close"].iloc[-10:].values
        rsis = rsi_series.iloc[-10:].values

        # Find local minima and maxima
        price_low_idx = closes.argmin()
        price_high_idx = closes.argmax()
        rsi_low_idx = rsis.argmin()
        rsi_high_idx = rsis.argmax()

        # Bullish divergence: lower price low, higher RSI low
        if price_low_idx < len(closes) - 3:  # Not the most recent
            price_making_lower_low = closes[-1] < closes[price_low_idx]
            rsi_making_higher_low = rsis[-1] > rsis[rsi_low_idx]
            if price_making_lower_low and rsi_making_higher_low:
                return True, "bullish_divergence"

        # Bearish divergence: higher price high, lower RSI high
        if price_high_idx < len(closes) - 3:
            price_making_higher_high = closes[-1] > closes[price_high_idx]
            rsi_making_lower_high = rsis[-1] < rsis[rsi_high_idx]
            if price_making_higher_high and rsi_making_lower_high:
                return True, "bearish_divergence"

        return False, None

    def _calculate_momentum_score(
        self,
        data: MomentumData,
        signal_direction: str,
    ) -> float:
        """Calculate momentum score based on indicator values.

        Args:
            data: MomentumData with indicator values.
            signal_direction: "BUY" or "SELL".

        Returns:
            Score from 0-100.
        """
        scores = []

        # RSI Score
        if signal_direction == "BUY":
            # Favorable: RSI 40-70 (not oversold, not overbought)
            if 40 <= data.rsi_current <= 70:
                scores.append(100.0)
            elif data.rsi_current < 40:
                scores.append(70.0)  # Oversold, potential reversal
            else:
                scores.append(30.0)  # Overbought, caution
        else:  # SELL
            # Favorable: RSI 30-60
            if 30 <= data.rsi_current <= 60:
                scores.append(100.0)
            elif data.rsi_current > 60:
                scores.append(70.0)  # Overbought, potential reversal
            else:
                scores.append(30.0)  # Oversold, caution

        # MACD Histogram Score
        if data.macd_histogram_trend == "rising":
            scores.append(100.0 if signal_direction == "BUY" else 30.0)
        elif data.macd_histogram_trend == "falling":
            scores.append(30.0 if signal_direction == "BUY" else 100.0)
        else:
            scores.append(50.0)

        # Stochastic Score
        if signal_direction == "BUY":
            if data.stochastic_k > data.stochastic_d and data.stochastic_k < 80:
                scores.append(100.0)
            else:
                scores.append(50.0)
        else:
            if data.stochastic_k < data.stochastic_d and data.stochastic_k > 20:
                scores.append(100.0)
            else:
                scores.append(50.0)

        # Divergence penalty
        if data.divergence_detected:
            if (signal_direction == "BUY" and data.divergence_type == "bearish_divergence") or \
               (signal_direction == "SELL" and data.divergence_type == "bullish_divergence"):
                scores.append(0.0)  # Strong contradiction
            else:
                scores.append(100.0)  # Confirming divergence

        return sum(scores) / len(scores) if scores else 50.0

    @graceful_degrade(
        default_return=AnalysisComponent(
            name="support_resistance",
            score=50.0,
            weight=0.25,
            confidence=30.0,
            reasoning="Support/Resistance analysis failed - using default neutral score",
            data={"near_support": False, "near_resistance": False, "score": 50},
        )
    )
    def analyze_support_resistance(
        self,
        df: pd.DataFrame,
        current_price: int,
        signal_direction: str,
    ) -> AnalysisComponent:
        """Analyze support and resistance levels.

        Args:
            df: OHLCV DataFrame with at least 2 days of data.
            current_price: Current price in paisa.
            signal_direction: "BUY" or "SELL".

        Returns:
            AnalysisComponent with score 0-100 and SupportResistanceData.
        """
        if df.empty or len(df) < 50:
            return AnalysisComponent(
                name="support_resistance",
                score=50.0,
                weight=0.25,
                confidence=0.0,
                reasoning="Insufficient data for S/R analysis",
                data={},
            )

        if current_price == 0:
            current_price = int(df["close"].iloc[-1])

        engine = self._get_engine(df)
        sr_data = SupportResistanceData(current_price=current_price)

        # Group by date to find previous day data
        df_with_date = df.copy()
        df_with_date["date"] = df_with_date.index.date
        dates = sorted(df_with_date["date"].unique())

        if len(dates) >= 2:
            prev_day = dates[-2]
            prev_day_data = df_with_date[df_with_date["date"] == prev_day]

            if not prev_day_data.empty:
                sr_data.pdh = int(prev_day_data["high"].max())
                sr_data.pdl = int(prev_day_data["low"].min())
                sr_data.pdc = int(prev_day_data["close"].iloc[-1])

        # Current day high/low
        if len(dates) >= 1:
            current_day = dates[-1]
            current_day_data = df_with_date[df_with_date["date"] == current_day]

            if not current_day_data.empty:
                current_day_high = int(current_day_data["high"].max())
                current_day_low = int(current_day_data["low"].min())

                # VWAP
                vwap_series = engine.vwap()
                if not vwap_series.empty:
                    sr_data.vwap = int(vwap_series.iloc[-1])

                    # VWAP bands (1 std dev)
                    vwap_std = current_day_data["close"].std()
                    if not pd.isna(vwap_std):
                        sr_data.vwap_upper_1std = int(sr_data.vwap + vwap_std)
                        sr_data.vwap_lower_1std = int(sr_data.vwap - vwap_std)

                # Pivot Points (Standard)
                if sr_data.pdh and sr_data.pdl and sr_data.pdc:
                    sr_data.pivot_point = int((sr_data.pdh + sr_data.pdl + sr_data.pdc) / 3)
                    sr_data.r1 = int(2 * sr_data.pivot_point - sr_data.pdl)
                    sr_data.r2 = int(sr_data.pivot_point + (sr_data.pdh - sr_data.pdl))
                    sr_data.s1 = int(2 * sr_data.pivot_point - sr_data.pdh)
                    sr_data.s2 = int(sr_data.pivot_point - (sr_data.pdh - sr_data.pdl))

        # Calculate distances to nearest S/R
        supports = [s for s in [sr_data.s1, sr_data.s2, sr_data.pdl, sr_data.vwap_lower_1std] if s]
        resistances = [r for r in [sr_data.r1, sr_data.r2, sr_data.pdh, sr_data.vwap_upper_1std] if r]

        if supports and current_price > 0:
            below = [s for s in supports if s < current_price]
            nearest_support = max(below) if below else min(supports)
            sr_data.nearest_support_distance_pct = ((current_price - nearest_support) / current_price) * 100

        if resistances and current_price > 0:
            above = [r for r in resistances if r > current_price]
            nearest_resistance = min(above) if above else max(resistances)
            sr_data.nearest_resistance_distance_pct = ((nearest_resistance - current_price) / current_price) * 100

        # Calculate score
        score = self._calculate_sr_score(sr_data, current_price, signal_direction)

        # Build reasoning
        reasoning = (
            f"Price {current_price} vs VWAP {sr_data.vwap}. "
            f"PDH: {sr_data.pdh}, PDL: {sr_data.pdl}. "
            f"Pivot: {sr_data.pivot_point}, R1: {sr_data.r1}, S1: {sr_data.s1}. "
            f"Dist to S: {sr_data.nearest_support_distance_pct:.2f}%, "
            f"Dist to R: {sr_data.nearest_resistance_distance_pct:.2f}%."
        )

        return AnalysisComponent(
            name="support_resistance",
            score=score,
            weight=0.25,
            confidence=85.0,
            reasoning=reasoning,
            data=sr_data.model_dump(),
        )

    def _calculate_sr_score(
        self,
        sr_data: SupportResistanceData,
        current_price: int,
        signal_direction: str,
    ) -> float:
        """Calculate support/resistance score.

        Args:
            sr_data: SupportResistanceData.
            current_price: Current price in paisa.
            signal_direction: "BUY" or "SELL".

        Returns:
            Score from 0-100.
        """
        if sr_data.vwap is None:
            return 50.0

        scores = []

        # VWAP position
        if signal_direction == "BUY":
            # Buying above VWAP is favorable
            if current_price > sr_data.vwap:
                scores.append(100.0)
            else:
                scores.append(50.0)
        else:
            # Selling below VWAP is favorable
            if current_price < sr_data.vwap:
                scores.append(100.0)
            else:
                scores.append(50.0)

        # Distance to support/resistance
        if signal_direction == "BUY":
            # For buys, want to be near support (low distance to support)
            if sr_data.nearest_support_distance_pct < 0.5:
                scores.append(100.0)  # Very close to support
            elif sr_data.nearest_support_distance_pct < 1.0:
                scores.append(80.0)
            else:
                scores.append(60.0)

            # Room to run to resistance
            if sr_data.nearest_resistance_distance_pct > 2.0:
                scores.append(100.0)
            elif sr_data.nearest_resistance_distance_pct > 1.0:
                scores.append(70.0)
            else:
                scores.append(40.0)
        else:  # SELL
            # For sells, want to be near resistance
            if sr_data.nearest_resistance_distance_pct < 0.5:
                scores.append(100.0)
            elif sr_data.nearest_resistance_distance_pct < 1.0:
                scores.append(80.0)
            else:
                scores.append(60.0)

            # Room to run to support
            if sr_data.nearest_support_distance_pct > 2.0:
                scores.append(100.0)
            elif sr_data.nearest_support_distance_pct > 1.0:
                scores.append(70.0)
            else:
                scores.append(40.0)

        return sum(scores) / len(scores) if scores else 50.0

    @graceful_degrade(
        default_return=AnalysisComponent(
            name="candle_context",
            score=50.0,
            weight=0.20,
            confidence=30.0,
            reasoning="Candle pattern analysis failed - using default neutral score",
            data={"pattern": None, "score": 50},
        )
    )
    def analyze_candle_context(
        self,
        df: pd.DataFrame,
        signal_direction: str,
        lookback: int = 5,
    ) -> AnalysisComponent:
        """Analyze candlestick patterns and context.

        Args:
            df: OHLCV DataFrame.
            signal_direction: "BUY" or "SELL".
            lookback: Number of candles to analyze. Default 5.

        Returns:
            AnalysisComponent with score 0-100.
        """
        if df.empty or len(df) < lookback + 1:
            return AnalysisComponent(
                name="candle_context",
                score=50.0,
                weight=0.20,
                confidence=0.0,
                reasoning="Insufficient candle data",
                data={},
            )

        context_data = self._analyze_candles(df, lookback)
        score = self._calculate_candle_score(context_data, signal_direction)

        # Build reasoning
        reasoning_parts = []
        if context_data.last_n_candles_confirming > 0:
            reasoning_parts.append(f"{context_data.last_n_candles_confirming} confirming candles")
        if context_data.rejection_wicks_present:
            reasoning_parts.append("rejection wicks present")
        if context_data.bullish_engulfing:
            reasoning_parts.append("bullish engulfing")
        if context_data.bearish_engulfing:
            reasoning_parts.append("bearish engulfing")
        if context_data.gap_up:
            reasoning_parts.append("gap up")
        if context_data.gap_down:
            reasoning_parts.append("gap down")

        reasoning = "; ".join(reasoning_parts) if reasoning_parts else "No significant patterns"

        return AnalysisComponent(
            name="candle_context",
            score=score,
            weight=0.20,
            confidence=70.0,
            reasoning=reasoning,
            data={
                "last_n_confirming": context_data.last_n_candles_confirming,
                "rejection_wicks": context_data.rejection_wicks_present,
                "body_wick_ratio": context_data.avg_body_to_wick_ratio,
                "gap_up": context_data.gap_up,
                "gap_down": context_data.gap_down,
                "bullish_engulfing": context_data.bullish_engulfing,
                "bearish_engulfing": context_data.bearish_engulfing,
                "consecutive_bullish": context_data.consecutive_bullish,
                "consecutive_bearish": context_data.consecutive_bearish,
            },
        )

    def _analyze_candles(
        self,
        df: pd.DataFrame,
        lookback: int,
    ) -> CandleContextData:
        """Analyze candlestick patterns.

        Args:
            df: OHLCV DataFrame.
            lookback: Number of candles to analyze.

        Returns:
            CandleContextData with analysis results.
        """
        data = CandleContextData()
        recent = df.iloc[-lookback:].copy()

        # Calculate candle properties
        recent["body"] = (recent["close"] - recent["open"]).abs()
        recent["upper_wick"] = recent["high"] - recent[["open", "close"]].max(axis=1)
        recent["lower_wick"] = recent[["open", "close"]].min(axis=1) - recent["low"]
        recent["total_range"] = recent["high"] - recent["low"]
        recent["is_bullish"] = recent["close"] > recent["open"]
        recent["is_bearish"] = recent["close"] < recent["open"]

        # Body to wick ratio
        recent["body_to_range"] = recent["body"] / recent["total_range"].clip(lower=1)
        data.avg_body_to_wick_ratio = recent["body_to_range"].mean()

        # Count confirming candles (last N candles moving in same direction)
        closes = recent["close"].values
        data.consecutive_bullish = 0
        data.consecutive_bearish = 0

        for i in range(len(closes) - 1, 0, -1):
            if closes[i] > closes[i - 1]:
                data.consecutive_bullish += 1
                data.consecutive_bearish = 0
            elif closes[i] < closes[i - 1]:
                data.consecutive_bearish += 1
                data.consecutive_bullish = 0
            else:
                break

        data.last_n_candles_confirming = max(data.consecutive_bullish, data.consecutive_bearish)

        # Rejection wicks (long wick in opposite direction of close)
        last_candle = recent.iloc[-1]
        if last_candle["is_bullish"] and last_candle["lower_wick"] > last_candle["body"] * 2:
            data.rejection_wicks_present = True
        elif last_candle["is_bearish"] and last_candle["upper_wick"] > last_candle["body"] * 2:
            data.rejection_wicks_present = True

        # Gap analysis
        if len(df) > lookback:
            prev_close = df.iloc[-lookback - 1]["close"]
            curr_open = recent.iloc[0]["open"]
            gap_pct = 0.0
            if prev_close != 0:
                gap_pct = abs(curr_open - prev_close) / prev_close * 100

            if gap_pct > 0.5:  # 0.5% gap threshold
                if curr_open > prev_close:
                    data.gap_up = True
                else:
                    data.gap_down = True

        # Engulfing patterns
        if len(recent) >= 2:
            prev = recent.iloc[-2]
            curr = recent.iloc[-1]

            # Bullish engulfing
            if (prev["is_bearish"] and curr["is_bullish"] and
                curr["open"] < prev["close"] and curr["close"] > prev["open"]):
                data.bullish_engulfing = True

            # Bearish engulfing
            if (prev["is_bullish"] and curr["is_bearish"] and
                curr["open"] > prev["close"] and curr["close"] < prev["open"]):
                data.bearish_engulfing = True

        # Doji detection
        data.doji_present = any(recent["body_to_range"] < 0.1)

        return data

    def _calculate_candle_score(
        self,
        data: CandleContextData,
        signal_direction: str,
    ) -> float:
        """Calculate candle context score.

        Args:
            data: CandleContextData.
            signal_direction: "BUY" or "SELL".

        Returns:
            Score from 0-100.
        """
        scores = []

        # Consecutive candles
        if signal_direction == "BUY":
            if data.consecutive_bullish >= 3:
                scores.append(100.0)
            elif data.consecutive_bullish >= 2:
                scores.append(80.0)
            elif data.consecutive_bullish >= 1:
                scores.append(60.0)
            elif data.consecutive_bearish >= 2:
                scores.append(20.0)  # Contradicting
            else:
                scores.append(50.0)
        else:  # SELL
            if data.consecutive_bearish >= 3:
                scores.append(100.0)
            elif data.consecutive_bearish >= 2:
                scores.append(80.0)
            elif data.consecutive_bearish >= 1:
                scores.append(60.0)
            elif data.consecutive_bullish >= 2:
                scores.append(20.0)  # Contradicting
            else:
                scores.append(50.0)

        # Pattern bonuses
        if signal_direction == "BUY":
            if data.bullish_engulfing:
                scores.append(100.0)
            if data.rejection_wicks_present and not data.bearish_engulfing:
                scores.append(80.0)
            if data.gap_up:
                scores.append(70.0)
            if data.bearish_engulfing:
                scores.append(10.0)  # Strong contradiction
        else:  # SELL
            if data.bearish_engulfing:
                scores.append(100.0)
            if data.rejection_wicks_present and not data.bullish_engulfing:
                scores.append(80.0)
            if data.gap_down:
                scores.append(70.0)
            if data.bullish_engulfing:
                scores.append(10.0)  # Strong contradiction

        # Body to wick ratio (higher is better - decisive candles)
        if data.avg_body_to_wick_ratio > 0.7:
            scores.append(90.0)
        elif data.avg_body_to_wick_ratio > 0.5:
            scores.append(70.0)
        else:
            scores.append(50.0)

        return sum(scores) / len(scores) if scores else 50.0
