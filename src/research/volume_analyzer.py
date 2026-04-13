"""Volume analysis module for trading signal confirmation.

This module provides volume-based analysis to confirm trading signals,
analyze liquidity, and assess market participation.
"""

from dataclasses import asdict, dataclass
from typing import Any, Optional
from enum import Enum

import pandas as pd
import numpy as np

from loguru import logger

from src.research.models import AnalysisComponent
from src.utils.degradation import graceful_degrade


class VolumeTrend(Enum):
    """Volume trend direction."""
    INCREASING = "increasing"
    DECREASING = "decreasing"
    FLAT = "flat"


@dataclass
class VolumeData:
    """Volume analysis data container.

    Attributes:
        current_volume: Current period volume
        avg_volume_20: 20-period average volume
        volume_ratio: Current volume / average volume
        volume_trend: Direction of volume trend
        obv_direction: OBV trend direction
        obv_value: Current OBV value
        bid_ask_spread_pct: Bid-ask spread as percentage (for options)
        open_interest: Current open interest (for derivatives)
        volume_oi_ratio: Volume to open interest ratio
        delivery_pct: Delivery percentage (for equity)
        large_lot_ratio: Ratio of large lots to total volume
    """
    current_volume: int = 0
    avg_volume_20: int = 0
    volume_ratio: float = 0.0
    volume_trend: VolumeTrend = VolumeTrend.FLAT
    obv_direction: VolumeTrend = VolumeTrend.FLAT
    obv_value: int = 0
    bid_ask_spread_pct: float = 0.0
    open_interest: int = 0
    volume_oi_ratio: float = 0.0
    delivery_pct: float = 0.0
    large_lot_ratio: float = 0.0


class VolumeAnalyzer:
    """Analyzer for volume confirmation, liquidity, and participation.

    All price values are handled in PAISA (integer) internally.
    """

    # Score thresholds
    STRONG_VOLUME_MULTIPLIER = 1.5
    MODERATE_VOLUME_MULTIPLIER = 1.2
    MIN_VOLUME_MULTIPLIER = 1.0

    # Liquidity thresholds (for options)
    LIQUID_SPREAD_THRESHOLD = 1.0  # < 1% = liquid
    ACCEPTABLE_SPREAD_THRESHOLD = 3.0  # 1-3% = acceptable
    POOR_SPREAD_THRESHOLD = 5.0  # 3-5% = poor

    # Volume-to-OI ratio thresholds
    HIGH_VOLUME_OI_RATIO = 1.0
    MODERATE_VOLUME_OI_RATIO = 0.5

    # Delivery percentage thresholds (for equity)
    HIGH_DELIVERY_THRESHOLD = 60.0
    MODERATE_DELIVERY_THRESHOLD = 40.0

    # Large lot threshold (institutional proxy)
    LARGE_LOT_THRESHOLD = 100000  # in PAISA value

    def __init__(self):
        """Initialize the volume analyzer."""
        logger.info("VolumeAnalyzer initialized")

    def _calculate_obv(self, df: pd.DataFrame) -> pd.Series:
        """Calculate On Balance Volume (OBV).

        Args:
            df: DataFrame with 'close' and 'volume' columns

        Returns:
            Series with OBV values
        """
        obv = pd.Series(index=df.index, dtype=np.int64)
        obv.iloc[0] = df['volume'].iloc[0]

        for i in range(1, len(df)):
            if df['close'].iloc[i] > df['close'].iloc[i - 1]:
                obv.iloc[i] = obv.iloc[i - 1] + df['volume'].iloc[i]
            elif df['close'].iloc[i] < df['close'].iloc[i - 1]:
                obv.iloc[i] = obv.iloc[i - 1] - df['volume'].iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i - 1]

        return obv

    def _get_volume_trend(self, volumes: pd.Series) -> VolumeTrend:
        """Determine volume trend direction.

        Args:
            volumes: Series of volume values

        Returns:
            VolumeTrend enum value
        """
        if len(volumes) < 5:
            return VolumeTrend.FLAT

        # Compare first half vs second half
        mid = len(volumes) // 2
        first_half_avg = volumes.iloc[:mid].mean()
        second_half_avg = volumes.iloc[mid:].mean()

        if second_half_avg > first_half_avg * 1.1:
            return VolumeTrend.INCREASING
        elif second_half_avg < first_half_avg * 0.9:
            return VolumeTrend.DECREASING
        return VolumeTrend.FLAT

    @graceful_degrade(
        default_return=AnalysisComponent(
            name="volume_confirmation",
            score=50,
            weight=0.40,
            confidence=0.0,
            reasoning="Volume analysis failed - using default neutral score",
            data=asdict(VolumeData()),
        )
    )
    def analyze_volume_confirmation(
        self,
        df: pd.DataFrame,
        signal_direction: str
    ) -> AnalysisComponent:
        """Analyze volume confirmation for a trading signal.

        Args:
            df: DataFrame with OHLCV data (close and volume in PAISA)
            signal_direction: 'buy' or 'sell'

        Returns:
            AnalysisComponent with score 0-100 and VolumeData
        """
        if len(df) < 20:
            logger.warning("Insufficient data for volume analysis (need 20 periods)")
            return AnalysisComponent(
                name="volume_confirmation",
                score=50,
                weight=0.10,
                reasoning="Insufficient data for volume analysis",
                data={}
            )

        # Calculate metrics
        current_volume = int(df['volume'].iloc[-1])
        avg_volume_20 = int(df['volume'].tail(20).mean())
        volume_ratio = current_volume / avg_volume_20 if avg_volume_20 > 0 else 0.0

        # Volume trend
        volume_trend = self._get_volume_trend(df['volume'].tail(10))

        # OBV calculation and direction
        obv = self._calculate_obv(df)
        obv_value = int(obv.iloc[-1])
        obv_recent = obv.tail(10)

        if len(obv_recent) >= 5:
            obv_mid = len(obv_recent) // 2
            if obv_recent.iloc[-1] > obv_recent.iloc[obv_mid] * 1.02:
                obv_direction = VolumeTrend.INCREASING
            elif obv_recent.iloc[-1] < obv_recent.iloc[obv_mid] * 0.98:
                obv_direction = VolumeTrend.DECREASING
            else:
                obv_direction = VolumeTrend.FLAT
        else:
            obv_direction = VolumeTrend.FLAT

        # Calculate score
        score = 0
        signals = []

        # Volume ratio scoring
        if volume_ratio >= self.STRONG_VOLUME_MULTIPLIER:
            score += 40
            signals.append(f"Strong volume ({volume_ratio:.2f}x average)")
        elif volume_ratio >= self.MODERATE_VOLUME_MULTIPLIER:
            score += 30
            signals.append(f"Moderate volume ({volume_ratio:.2f}x average)")
        elif volume_ratio >= self.MIN_VOLUME_MULTIPLIER:
            score += 15
            signals.append(f"Average volume ({volume_ratio:.2f}x)")
        else:
            signals.append(f"Weak volume ({volume_ratio:.2f}x average)")

        # Volume trend scoring
        if volume_trend == VolumeTrend.INCREASING:
            score += 20
            signals.append("Volume trend increasing")
        elif volume_trend == VolumeTrend.DECREASING:
            score += 5
            signals.append("Volume trend decreasing")
        else:
            score += 10
            signals.append("Volume trend flat")

        # OBV confirmation scoring
        if signal_direction.lower() == 'buy':
            if obv_direction == VolumeTrend.INCREASING:
                score += 40
                signals.append("OBV confirming bullish")
            elif obv_direction == VolumeTrend.DECREASING:
                score += 10
                signals.append("OBV diverging bearish")
            else:
                score += 20
                signals.append("OBV neutral")
        else:  # sell
            if obv_direction == VolumeTrend.DECREASING:
                score += 40
                signals.append("OBV confirming bearish")
            elif obv_direction == VolumeTrend.INCREASING:
                score += 10
                signals.append("OBV diverging bullish")
            else:
                score += 20
                signals.append("OBV neutral")

        # Ensure score is within bounds
        score = max(0, min(100, score))

        data = VolumeData(
            current_volume=current_volume,
            avg_volume_20=avg_volume_20,
            volume_ratio=volume_ratio,
            volume_trend=volume_trend,
            obv_direction=obv_direction,
            obv_value=obv_value
        )

        signal_text = "; ".join(signals) if signals else "No clear volume signal"

        logger.debug(
            f"Volume confirmation analysis: score={score}, "
            f"ratio={volume_ratio:.2f}, trend={volume_trend.value}, "
            f"OBV={obv_direction.value}"
        )

        return AnalysisComponent(
            name="volume_confirmation",
            score=score,
            weight=0.10,
            reasoning=signal_text,
            data={"volume_ratio": volume_ratio, "trend": volume_trend.value}
        )

    @graceful_degrade(
        default_return=AnalysisComponent(
            name="liquidity",
            score=50,
            weight=0.05,
            reasoning="Liquidity analysis failed - using default neutral score",
            data={}
        )
    )
    def analyze_liquidity(
        self,
        instrument_key: str,
        option_data: Optional[dict] = None
    ) -> AnalysisComponent:
        """Analyze liquidity for an instrument.

        For options: analyzes bid-ask spread, open interest, and volume-to-OI ratio.
        For equity: analyzes average volume and delivery.

        Args:
            instrument_key: Instrument identifier
            option_data: Optional dict with 'bid', 'ask', 'oi', 'volume' for options

        Returns:
            AnalysisComponent with score 0-100 and VolumeData
        """
        data = VolumeData()
        signals = []
        score = 0

        if option_data:
            # Option liquidity analysis
            bid = option_data.get('bid', 0)
            ask = option_data.get('ask', 0)
            oi = option_data.get('oi', 0)
            volume = option_data.get('volume', 0)

            # Bid-ask spread analysis (prices in PAISA)
            if bid > 0 and ask > 0:
                mid_price = (bid + ask) / 2
                spread = ask - bid
                spread_pct = (spread / mid_price) * 100 if mid_price > 0 else 0
                data.bid_ask_spread_pct = spread_pct

                if spread_pct < self.LIQUID_SPREAD_THRESHOLD:
                    score += 50
                    signals.append(f"Highly liquid: {spread_pct:.2f}% spread")
                elif spread_pct < self.ACCEPTABLE_SPREAD_THRESHOLD:
                    score += 35
                    signals.append(f"Acceptable liquidity: {spread_pct:.2f}% spread")
                elif spread_pct < self.POOR_SPREAD_THRESHOLD:
                    score += 20
                    signals.append(f"Poor liquidity: {spread_pct:.2f}% spread")
                else:
                    score += 5
                    signals.append(f"Illiquid: {spread_pct:.2f}% spread")
            else:
                signals.append("No bid-ask data available")

            # Open Interest check
            data.open_interest = oi
            if oi > 100000:  # High OI
                score += 25
                signals.append(f"High OI: {oi:,}")
            elif oi > 50000:
                score += 15
                signals.append(f"Moderate OI: {oi:,}")
            elif oi > 10000:
                score += 10
                signals.append(f"Low OI: {oi:,}")
            else:
                signals.append(f"Very low OI: {oi:,}")

            # Volume-to-OI ratio
            if oi > 0:
                volume_oi_ratio = volume / oi
                data.volume_oi_ratio = volume_oi_ratio

                if volume_oi_ratio >= self.HIGH_VOLUME_OI_RATIO:
                    score += 25
                    signals.append(f"High turnover: {volume_oi_ratio:.2f} V/OI")
                elif volume_oi_ratio >= self.MODERATE_VOLUME_OI_RATIO:
                    score += 15
                    signals.append(f"Moderate turnover: {volume_oi_ratio:.2f} V/OI")
                else:
                    score += 5
                    signals.append(f"Low turnover: {volume_oi_ratio:.2f} V/OI")
            else:
                signals.append("No OI data for ratio")
        else:
            # Equity liquidity analysis (simplified)
            # In practice, this would fetch from market data
            signals.append("Equity liquidity: using volume metrics")
            score = 70  # Default moderate score for equity

        # Ensure score is within bounds
        score = max(0, min(100, score))

        signal_text = "; ".join(signals) if signals else "Liquidity analysis inconclusive"

        logger.debug(
            f"Liquidity analysis for {instrument_key}: score={score}, "
            f"spread={data.bid_ask_spread_pct:.2f}%, OI={data.open_interest:,}"
        )

        return AnalysisComponent(
            name="liquidity",
            score=score,
            weight=0.05,
            reasoning=signal_text,
            data={"spread_pct": data.bid_ask_spread_pct, "oi": data.open_interest}
        )

    @graceful_degrade(
        default_return=AnalysisComponent(
            name="participation",
            score=50,
            weight=0.05,
            reasoning="Participation analysis failed - using default neutral score",
            data={}
        )
    )
    def analyze_participation(
        self,
        df: pd.DataFrame,
        signal_direction: str
    ) -> AnalysisComponent:
        """Analyze market participation for confirmation.

        Uses institutional participation proxies like large lot sizes
        and delivery percentage for equity.

        Args:
            df: DataFrame with OHLCV data
            signal_direction: 'buy' or 'sell'

        Returns:
            AnalysisComponent with score 0-100 and VolumeData
        """
        data = VolumeData()
        signals = []
        score = 0

        if len(df) < 5:
            logger.warning("Insufficient data for participation analysis")
            return AnalysisComponent(
                name="participation",
                score=50,
                weight=0.05,
                reasoning="Insufficient data for participation analysis",
                data={}
            )

        # Large lot analysis (proxy for institutional participation)
        # Calculate based on volume and price - large value trades
        if 'close' in df.columns and 'volume' in df.columns:
            recent_df = df.tail(5)

            # Estimate large lot ratio based on volume distribution
            # High volume with price stability suggests institutional participation
            avg_volume = recent_df['volume'].mean()
            avg_price = recent_df['close'].mean()

            # Calculate trade value in PAISA
            trade_value = avg_volume * avg_price

            # Estimate large lot ratio (simplified heuristic)
            volume_std = recent_df['volume'].std()
            volume_mean = recent_df['volume'].mean()

            if volume_mean > 0:
                cv = volume_std / volume_mean  # Coefficient of variation
                # Lower CV with high volume suggests institutional participation
                if cv < 0.3 and trade_value > self.LARGE_LOT_THRESHOLD:
                    data.large_lot_ratio = 0.6
                    score += 40
                    signals.append("Strong institutional participation suspected")
                elif cv < 0.5 and trade_value > self.LARGE_LOT_THRESHOLD / 2:
                    data.large_lot_ratio = 0.4
                    score += 25
                    signals.append("Moderate institutional participation")
                else:
                    data.large_lot_ratio = 0.2
                    score += 10
                    signals.append("Retail-dominated volume")
            else:
                signals.append("Unable to calculate participation metrics")

        # Delivery percentage analysis (if available in df)
        if 'delivery_pct' in df.columns:
            avg_delivery = df['delivery_pct'].tail(5).mean()
            data.delivery_pct = avg_delivery

            if avg_delivery >= self.HIGH_DELIVERY_THRESHOLD:
                score += 40
                signals.append(f"High delivery: {avg_delivery:.1f}% (institutional)")
            elif avg_delivery >= self.MODERATE_DELIVERY_THRESHOLD:
                score += 25
                signals.append(f"Moderate delivery: {avg_delivery:.1f}%")
            else:
                score += 10
                signals.append(f"Low delivery: {avg_delivery:.1f}% (speculative)")
        else:
            # Default delivery score based on volume pattern
            score += 20
            signals.append("Delivery data not available, using volume patterns")

        # Direction confirmation
        if signal_direction.lower() == 'buy':
            if data.large_lot_ratio > 0.4 or data.delivery_pct > 50:
                score += 20
                signals.append("Participation supports bullish move")
            else:
                score += 10
                signals.append("Participation neutral for bullish")
        else:
            if data.large_lot_ratio > 0.4 or data.delivery_pct > 50:
                score += 20
                signals.append("Participation supports bearish move")
            else:
                score += 10
                signals.append("Participation neutral for bearish")

        # Ensure score is within bounds
        score = max(0, min(100, score))

        signal_text = "; ".join(signals) if signals else "Participation analysis inconclusive"

        logger.debug(
            f"Participation analysis: score={score}, "
            f"large_lot={data.large_lot_ratio:.2f}, delivery={data.delivery_pct:.1f}%"
        )

        return AnalysisComponent(
            name="participation",
            score=score,
            weight=0.05,
            reasoning=signal_text,
            data={"large_lot_ratio": data.large_lot_ratio, "delivery_pct": data.delivery_pct}
        )

    @graceful_degrade(
        default_return=(
            AnalysisComponent(
                name="volume_analysis",
                score=50,
                weight=0.10,
                reasoning="Volume analysis failed - using default neutral score",
                data={}
            ),
            VolumeData()
        )
    )
    def analyze(
        self,
        df: pd.DataFrame,
        instrument_key: str,
        signal_direction: str,
        is_option: bool = False,
        option_data: Optional[dict] = None
    ) -> tuple[AnalysisComponent, VolumeData]:
        """Run complete volume analysis.

        Combines volume confirmation, liquidity, and participation analysis.

        Args:
            df: DataFrame with OHLCV data
            instrument_key: Instrument identifier
            signal_direction: 'buy' or 'sell'
            is_option: Whether instrument is an option
            option_data: Optional data for option liquidity analysis

        Returns:
            Tuple of (combined AnalysisComponent, VolumeData with all metrics)
        """
        logger.info(
            f"Starting volume analysis for {instrument_key} "
            f"(direction={signal_direction}, is_option={is_option})"
        )

        # Run individual analyses
        volume_confirmation = self.analyze_volume_confirmation(df, signal_direction)
        liquidity = self.analyze_liquidity(
            instrument_key,
            option_data if is_option else None
        )
        participation = self.analyze_participation(df, signal_direction)

        # Combine scores with weights
        # Volume confirmation: 40%, Liquidity: 35%, Participation: 25%
        combined_score = int(
            volume_confirmation.score * 0.40 +
            liquidity.score * 0.35 +
            participation.score * 0.25
        )

        # Merge VolumeData — the sub-analyses store their raw numbers in
        # AnalysisComponent.data as a dict (see analyze_volume_confirmation
        # return at ~line 261). Use .get() so we never crash if a key is
        # missing or the data payload is shaped differently.
        vc_data = volume_confirmation.data if isinstance(volume_confirmation.data, dict) else {}
        lq_data = liquidity.data if isinstance(liquidity.data, dict) else {}
        pp_data = participation.data if isinstance(participation.data, dict) else {}

        combined_data = VolumeData(
            volume_ratio=float(vc_data.get("volume_ratio", 0.0)),
            bid_ask_spread_pct=float(lq_data.get("spread_pct", 0.0)),
            open_interest=int(lq_data.get("oi", 0) or 0),
            delivery_pct=float(pp_data.get("delivery_pct", 0.0)),
            large_lot_ratio=float(pp_data.get("large_lot_ratio", 0.0)),
        )

        # Combined human-readable reasoning — AnalysisComponent uses
        # `reasoning` for this (no `signal` field).
        combined_reasoning = (
            f"Volume: {volume_confirmation.reasoning} | "
            f"Liquidity: {liquidity.reasoning} | "
            f"Participation: {participation.reasoning}"
        )

        logger.info(
            f"Volume analysis complete for {instrument_key}: "
            f"combined_score={combined_score}"
        )

        return (
            AnalysisComponent(
                name="volume_analysis",
                score=combined_score,
                weight=0.10,
                confidence=75.0,
                reasoning=combined_reasoning,
                data={
                    "volume_ratio": combined_data.volume_ratio,
                    "bid_ask_spread_pct": combined_data.bid_ask_spread_pct,
                    "open_interest": combined_data.open_interest,
                    "delivery_pct": combined_data.delivery_pct,
                    "large_lot_ratio": combined_data.large_lot_ratio,
                },
            ),
            combined_data
        )
