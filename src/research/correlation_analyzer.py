"""Correlation analysis module for the AI Trade Research Engine.

Analyzes correlations between symbols and indices, relative strength,
and sector rotation patterns to validate trade signals.

All prices are in PAISA (integer) - 1 Rupee = 100 paisa.
All timestamps are in IST (Asia/Kolkata).
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import TYPE_CHECKING
from zoneinfo import ZoneInfo

import pandas as pd
from loguru import logger

from src.research.models import AnalysisComponent, CorrelationData
from src.utils.degradation import graceful_degrade

if TYPE_CHECKING:
    from src.data.historical import HistoricalDataFetcher

IST = ZoneInfo("Asia/Kolkata")

# Instrument keys for major indices
NIFTY_50_KEY = "NSE_INDEX|Nifty 50"
BANK_NIFTY_KEY = "NSE_INDEX|Nifty Bank"
FINNIFTY_KEY = "NSE_INDEX|Nifty Fin Service"

# Sector index mappings (symbol -> sector index key)
SECTOR_INDEX_MAP: dict[str, str] = {
    # Banking sector
    "HDFCBANK": "NSE_INDEX|Nifty Bank",
    "ICICIBANK": "NSE_INDEX|Nifty Bank",
    "SBIN": "NSE_INDEX|Nifty Bank",
    "KOTAKBANK": "NSE_INDEX|Nifty Bank",
    "AXISBANK": "NSE_INDEX|Nifty Bank",
    "INDUSINDBK": "NSE_INDEX|Nifty Bank",
    "BANKBARODA": "NSE_INDEX|Nifty Bank",
    "PNB": "NSE_INDEX|Nifty Bank",
    "CANBK": "NSE_INDEX|Nifty Bank",
    "UNIONBANK": "NSE_INDEX|Nifty Bank",
    # Financial Services
    "BAJFINANCE": "NSE_INDEX|Nifty Fin Service",
    "BAJAJFINSV": "NSE_INDEX|Nifty Fin Service",
    "HDFCLIFE": "NSE_INDEX|Nifty Fin Service",
    "ICICIPRULI": "NSE_INDEX|Nifty Fin Service",
    "SBILIFE": "NSE_INDEX|Nifty Fin Service",
    "RECLTD": "NSE_INDEX|Nifty Fin Service",
    "PFC": "NSE_INDEX|Nifty Fin Service",
    # IT Sector
    "TCS": "NSE_INDEX|Nifty IT",
    "INFY": "NSE_INDEX|Nifty IT",
    "WIPRO": "NSE_INDEX|Nifty IT",
    "HCLTECH": "NSE_INDEX|Nifty IT",
    "TECHM": "NSE_INDEX|Nifty IT",
    "LTIM": "NSE_INDEX|Nifty IT",
    "PERSISTENT": "NSE_INDEX|Nifty IT",
    "COFORGE": "NSE_INDEX|Nifty IT",
    # Auto Sector
    "MARUTI": "NSE_INDEX|Nifty Auto",
    "TATAMOTORS": "NSE_INDEX|Nifty Auto",
    "M&M": "NSE_INDEX|Nifty Auto",
    "HEROMOTOCO": "NSE_INDEX|Nifty Auto",
    "EICHERMOT": "NSE_INDEX|Nifty Auto",
    "BAJAJ-AUTO": "NSE_INDEX|Nifty Auto",
    "TVSMOTOR": "NSE_INDEX|Nifty Auto",
    # Pharma
    "SUNPHARMA": "NSE_INDEX|Nifty Pharma",
    "DRREDDY": "NSE_INDEX|Nifty Pharma",
    "CIPLA": "NSE_INDEX|Nifty Pharma",
    "DIVISLAB": "NSE_INDEX|Nifty Pharma",
    "LUPIN": "NSE_INDEX|Nifty Pharma",
    "AUROPHARMA": "NSE_INDEX|Nifty Pharma",
    "TORNTPHARM": "NSE_INDEX|Nifty Pharma",
    # FMCG
    "HINDUNILVR": "NSE_INDEX|Nifty FMCG",
    "NESTLEIND": "NSE_INDEX|Nifty FMCG",
    "ITC": "NSE_INDEX|Nifty FMCG",
    "BRITANNIA": "NSE_INDEX|Nifty FMCG",
    "DABUR": "NSE_INDEX|Nifty FMCG",
    "GODREJCP": "NSE_INDEX|Nifty FMCG",
    "MARICO": "NSE_INDEX|Nifty FMCG",
    "VBL": "NSE_INDEX|Nifty FMCG",
    # Energy/Oil
    "RELIANCE": "NSE_INDEX|Nifty Energy",
    "ONGC": "NSE_INDEX|Nifty Energy",
    "NTPC": "NSE_INDEX|Nifty Energy",
    "POWERGRID": "NSE_INDEX|Nifty Energy",
    "ADANIGREEN": "NSE_INDEX|Nifty Energy",
    "TATAPOWER": "NSE_INDEX|Nifty Energy",
    # Metal
    "TATASTEEL": "NSE_INDEX|Nifty Metal",
    "JSWSTEEL": "NSE_INDEX|Nifty Metal",
    "HINDALCO": "NSE_INDEX|Nifty Metal",
    "COALINDIA": "NSE_INDEX|Nifty Metal",
    "VEDL": "NSE_INDEX|Nifty Metal",
    "NMDC": "NSE_INDEX|Nifty Metal",
    # Realty
    "DLF": "NSE_INDEX|Nifty Realty",
    "GODREJPROP": "NSE_INDEX|Nifty Realty",
    "OBEROIRLTY": "NSE_INDEX|Nifty Realty",
    "PHOENIXLTD": "NSE_INDEX|Nifty Realty",
    "PRESTIGE": "NSE_INDEX|Nifty Realty",
    "BRIGADE": "NSE_INDEX|Nifty Realty",
    # Media
    "ZEEL": "NSE_INDEX|Nifty Media",
    "SUNTV": "NSE_INDEX|Nifty Media",
    "PVRINOX": "NSE_INDEX|Nifty Media",
    # Consumer Durables
    "TITAN": "NSE_INDEX|Nifty Consumer Durables",
    "HAVELLS": "NSE_INDEX|Nifty Consumer Durables",
    "CROMPTON": "NSE_INDEX|Nifty Consumer Durables",
    "WHIRLPOOL": "NSE_INDEX|Nifty Consumer Durables",
}


class CorrelationAnalyzer:
    """Correlation analyzer for trade signal validation.

    Analyzes correlations between symbols and their sector indices,
    relative strength vs Nifty, and sector rotation patterns.

    All prices are in PAISA (integer).
    Timestamps are in IST.
    """

    def __init__(self, data_fetcher: HistoricalDataFetcher | None = None) -> None:
        """Initialize the correlation analyzer.

        Args:
            data_fetcher: HistoricalDataFetcher instance for fetching index data.
        """
        self._data_fetcher = data_fetcher
        self._cache: dict[str, dict] = {}
        self._cache_ttl_minutes = 30

    def _get_cache_key(self, symbol: str, analysis_type: str) -> str:
        """Generate cache key for correlation data.

        Args:
            symbol: Symbol being analyzed.
            analysis_type: Type of analysis (e.g., 'correlation', 'rs').

        Returns:
            Cache key string.
        """
        return f"{symbol}_{analysis_type}"

    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached data is still valid.

        Args:
            cache_key: Cache key to check.

        Returns:
            True if cache is valid, False otherwise.
        """
        if cache_key not in self._cache:
            return False

        cached_time = self._cache[cache_key].get("timestamp")
        if cached_time is None:
            return False

        age = datetime.now(IST) - cached_time
        return age < timedelta(minutes=self._cache_ttl_minutes)

    def _get_cached_data(self, cache_key: str) -> dict | None:
        """Get cached correlation data if valid.

        Args:
            cache_key: Cache key to retrieve.

        Returns:
            Cached data dict or None if expired/missing.
        """
        if self._is_cache_valid(cache_key):
            return self._cache[cache_key].get("data")
        return None

    def _set_cached_data(self, cache_key: str, data: dict) -> None:
        """Cache correlation data with timestamp.

        Args:
            cache_key: Cache key to store.
            data: Data to cache.
        """
        self._cache[cache_key] = {
            "timestamp": datetime.now(IST),
            "data": data,
        }

    def _extract_symbol_from_key(self, underlying_key: str) -> str:
        """Extract symbol from instrument key.

        Args:
            underlying_key: Upstox instrument key (e.g., 'NSE_EQ|RELIANCE').

        Returns:
            Symbol string (e.g., 'RELIANCE').
        """
        parts = underlying_key.split("|")
        if len(parts) >= 2:
            return parts[1].split(":")[0]  # Handle both formats
        return underlying_key

    def _get_sector_index(self, symbol: str) -> str | None:
        """Get sector index key for a symbol.

        Args:
            symbol: Stock symbol.

        Returns:
            Sector index instrument key or None if not found.
        """
        return SECTOR_INDEX_MAP.get(symbol.upper())

    def _calculate_correlation(
        self,
        symbol_returns: pd.Series,
        index_returns: pd.Series,
        period: int = 5,
    ) -> float:
        """Calculate rolling correlation between two return series.

        Args:
            symbol_returns: Symbol returns series.
            index_returns: Index returns series.
            period: Rolling period for correlation. Default 5.

        Returns:
            Correlation coefficient (-1 to 1).
        """
        if len(symbol_returns) < period or len(index_returns) < period:
            return 0.0

        # Align the series
        aligned_symbol, aligned_index = symbol_returns.align(index_returns, join="inner")

        if len(aligned_symbol) < period:
            return 0.0

        # Calculate rolling correlation on last 'period' values
        correlation = aligned_symbol.iloc[-period:].corr(aligned_index.iloc[-period:])

        return correlation if not pd.isna(correlation) else 0.0

    def _calculate_returns(self, df: pd.DataFrame) -> pd.Series:
        """Calculate percentage returns from price data.

        Args:
            df: OHLCV DataFrame with prices in paisa.

        Returns:
            Returns series.
        """
        if df.empty or "close" not in df.columns:
            return pd.Series(dtype=float)

        # Calculate percentage returns
        returns = df["close"].pct_change().dropna()
        return returns

    @graceful_degrade(
        default_return=AnalysisComponent(
            name="correlation",
            score=50.0,
            weight=0.15,
            confidence=30.0,
            reasoning="Correlation analysis failed - using default neutral score",
            data={"diversified": True, "score": 50},
        )
    )
    def analyze_index_correlation(
        self,
        underlying_key: str,
        bars_dict: dict[str, pd.DataFrame],
        signal_direction: str,
    ) -> AnalysisComponent:
        """Analyze correlation with relevant index.

        For Bank Nifty: checks correlation with Nifty 50.
        For stocks: checks correlation with sector index.
        Calculates 5-day and 20-day rolling correlations.

        Args:
            underlying_key: Upstox instrument key.
            bars_dict: Dictionary mapping timeframe strings to OHLCV DataFrames.
            signal_direction: "BUY" or "SELL".

        Returns:
            AnalysisComponent with score 0-100 and CorrelationData.
        """
        symbol = self._extract_symbol_from_key(underlying_key)
        cache_key = self._get_cache_key(symbol, "correlation")

        # Check cache
        cached = self._get_cached_data(cache_key)
        if cached:
            logger.debug("Using cached correlation data for {}", symbol)
            corr_data = CorrelationData(**cached)
        else:
            # Calculate correlations
            corr_data = self._calculate_correlations(underlying_key, bars_dict)
            self._set_cached_data(cache_key, corr_data.model_dump())

        # Calculate score based on correlations
        score = self._calculate_correlation_score(corr_data, signal_direction, underlying_key)

        # Build reasoning
        if "BANK_NIFTY" in underlying_key.upper() or symbol.upper() == "BANKNIFTY":
            reasoning = (
                f"Bank Nifty correlation with Nifty 50: "
                f"5d={corr_data.nifty_correlation_5d:.2f}, "
                f"20d={corr_data.nifty_correlation_20d:.2f}. "
            )
        else:
            sector_idx = self._get_sector_index(symbol)
            if sector_idx:
                sector_name = sector_idx.split("|")[-1] if "|" in sector_idx else sector_idx
                reasoning = (
                    f"{symbol} correlation: vs Nifty 5d={corr_data.nifty_correlation_5d:.2f}, "
                    f"vs {sector_name} 5d={corr_data.sector_correlation_5d:.2f}. "
                )
            else:
                reasoning = (
                    f"{symbol} correlation with Nifty: "
                    f"5d={corr_data.nifty_correlation_5d:.2f}, "
                    f"20d={corr_data.nifty_correlation_20d:.2f}. "
                )

        # Add relative strength context
        if corr_data.relative_strength_vs_nifty > 0.02:
            reasoning += f"Outperforming Nifty by {corr_data.relative_strength_vs_nifty*100:.1f}%."
        elif corr_data.relative_strength_vs_nifty < -0.02:
            reasoning += f"Underperforming Nifty by {abs(corr_data.relative_strength_vs_nifty)*100:.1f}%."
        else:
            reasoning += "Tracking Nifty closely."

        return AnalysisComponent(
            name="correlation",
            score=score,
            weight=0.15,
            confidence=75.0,
            reasoning=reasoning,
            data=corr_data.model_dump(),
        )

    def _calculate_correlations(
        self,
        underlying_key: str,
        bars_dict: dict[str, pd.DataFrame],
    ) -> CorrelationData:
        """Calculate correlation metrics for a symbol.

        Args:
            underlying_key: Upstox instrument key.
            bars_dict: Dictionary of timeframe -> DataFrame.

        Returns:
            CorrelationData with all correlation metrics.
        """
        corr_data = CorrelationData()
        symbol = self._extract_symbol_from_key(underlying_key)

        # Get symbol returns from 5min or 15min bars
        symbol_df = bars_dict.get("5min") or bars_dict.get("15min") or bars_dict.get("1min")
        if symbol_df is None or symbol_df.empty:
            return corr_data

        symbol_returns = self._calculate_returns(symbol_df)

        # Check if this is Bank Nifty
        is_bank_nifty = (
            "BANK" in underlying_key.upper()
            or symbol.upper() == "BANKNIFTY"
            or symbol.upper() == "NIFTY BANK"
        )

        if is_bank_nifty:
            # Bank Nifty: correlate with Nifty 50
            if self._data_fetcher:
                nifty_df = self._fetch_index_data(NIFTY_50_KEY, days=25)
                if not nifty_df.empty:
                    nifty_returns = self._calculate_returns(nifty_df)
                    corr_data.nifty_correlation_5d = self._calculate_correlation(
                        symbol_returns, nifty_returns, period=5
                    )
                    corr_data.nifty_correlation_20d = self._calculate_correlation(
                        symbol_returns, nifty_returns, period=20
                    )
        else:
            # Stock: correlate with Nifty 50 and sector index
            if self._data_fetcher:
                # Nifty 50 correlation
                nifty_df = self._fetch_index_data(NIFTY_50_KEY, days=25)
                if not nifty_df.empty:
                    nifty_returns = self._calculate_returns(nifty_df)
                    corr_data.nifty_correlation_5d = self._calculate_correlation(
                        symbol_returns, nifty_returns, period=5
                    )
                    corr_data.nifty_correlation_20d = self._calculate_correlation(
                        symbol_returns, nifty_returns, period=20
                    )

                # Sector index correlation
                sector_key = self._get_sector_index(symbol)
                if sector_key:
                    sector_df = self._fetch_index_data(sector_key, days=25)
                    if not sector_df.empty:
                        sector_returns = self._calculate_returns(sector_df)
                        corr_data.sector_correlation_5d = self._calculate_correlation(
                            symbol_returns, sector_returns, period=5
                        )

        return corr_data

    def _fetch_index_data(self, index_key: str, days: int = 25) -> pd.DataFrame:
        """Fetch historical data for an index.

        Args:
            index_key: Upstox instrument key for index.
            days: Number of days to fetch.

        Returns:
            DataFrame with OHLCV data.
        """
        if self._data_fetcher is None:
            return pd.DataFrame()

        try:
            return self._data_fetcher.fetch_candles(
                instrument_key=index_key,
                interval="5minute",
                days=days,
            )
        except Exception as exc:
            logger.warning("Failed to fetch index data for {}: {}", index_key, exc)
            return pd.DataFrame()

    def _calculate_correlation_score(
        self,
        corr_data: CorrelationData,
        signal_direction: str,
        underlying_key: str,
    ) -> float:
        """Calculate correlation-based score.

        Args:
            corr_data: CorrelationData with metrics.
            signal_direction: "BUY" or "SELL".
            underlying_key: Instrument key for context.

        Returns:
            Score from 0-100.
        """
        scores = []

        # Check if this is Bank Nifty
        symbol = self._extract_symbol_from_key(underlying_key)
        is_bank_nifty = (
            "BANK" in underlying_key.upper()
            or symbol.upper() == "BANKNIFTY"
            or symbol.upper() == "NIFTY BANK"
        )

        if is_bank_nifty:
            # Bank Nifty: check correlation with Nifty 50
            # High correlation with Nifty is expected and good for confirmation
            nifty_corr = abs(corr_data.nifty_correlation_5d)

            if nifty_corr > 0.8:
                scores.append(90.0)  # Strong correlation
            elif nifty_corr > 0.6:
                scores.append(75.0)
            elif nifty_corr > 0.4:
                scores.append(60.0)
            else:
                scores.append(40.0)  # Weak correlation - divergence

            # Check if correlations are consistent (5d vs 20d)
            corr_diff = abs(corr_data.nifty_correlation_5d - corr_data.nifty_correlation_20d)
            if corr_diff < 0.2:
                scores.append(90.0)  # Stable correlation
            elif corr_diff < 0.4:
                scores.append(70.0)
            else:
                scores.append(50.0)  # Changing correlation regime

        else:
            # Stock: check sector correlation
            sector_corr = abs(corr_data.sector_correlation_5d)

            if sector_corr > 0.7:
                scores.append(85.0)  # Moving with sector
            elif sector_corr > 0.5:
                scores.append(70.0)
            elif sector_corr > 0.3:
                scores.append(55.0)
            else:
                scores.append(45.0)  # Diverging from sector

            # Nifty correlation
            nifty_corr = abs(corr_data.nifty_correlation_5d)
            if nifty_corr > 0.6:
                scores.append(80.0)
            elif nifty_corr > 0.4:
                scores.append(65.0)
            else:
                scores.append(50.0)

        # Relative strength bonus
        if corr_data.relative_strength_vs_nifty > 0.01:
            # Outperforming Nifty - positive for both BUY and SELL
            # (shows relative strength in either direction)
            scores.append(80.0)
        elif corr_data.relative_strength_vs_nifty < -0.01:
            scores.append(60.0)
        else:
            scores.append(70.0)

        return sum(scores) / len(scores) if scores else 50.0

    def analyze_relative_strength(
        self,
        symbol: str,
        bars: pd.DataFrame,
        index_bars: pd.DataFrame,
        sector_bars: pd.DataFrame | None = None,
    ) -> dict:
        """Analyze relative strength of symbol vs indices.

        Compares symbol performance vs Nifty and vs sector index.

        Args:
            symbol: Stock symbol.
            bars: Symbol OHLCV DataFrame.
            index_bars: Nifty 50 OHLCV DataFrame.
            sector_bars: Sector index OHLCV DataFrame (optional).

        Returns:
            Dictionary with relative strength metrics.
        """
        result = {
            "symbol": symbol,
            "vs_nifty_1d": 0.0,
            "vs_nifty_5d": 0.0,
            "vs_sector_1d": 0.0,
            "vs_sector_5d": 0.0,
            "relative_strength_score": 50.0,
        }

        if bars.empty or index_bars.empty:
            return result

        # Calculate returns
        symbol_returns = self._calculate_returns(bars)
        nifty_returns = self._calculate_returns(index_bars)

        if len(symbol_returns) < 1 or len(nifty_returns) < 1:
            return result

        # 1-day relative strength
        symbol_1d = symbol_returns.iloc[-1] if len(symbol_returns) >= 1 else 0
        nifty_1d = nifty_returns.iloc[-1] if len(nifty_returns) >= 1 else 0
        result["vs_nifty_1d"] = symbol_1d - nifty_1d

        # 5-day relative strength
        symbol_5d = symbol_returns.iloc[-5:].sum() if len(symbol_returns) >= 5 else symbol_returns.sum() if len(symbol_returns) > 0 else 0
        nifty_5d = nifty_returns.iloc[-5:].sum() if len(nifty_returns) >= 5 else nifty_returns.sum() if len(nifty_returns) > 0 else 0
        result["vs_nifty_5d"] = symbol_5d - nifty_5d

        # Sector comparison
        if sector_bars is not None and not sector_bars.empty:
            sector_returns = self._calculate_returns(sector_bars)

            if len(sector_returns) >= 1:
                sector_1d = sector_returns.iloc[-1]
                result["vs_sector_1d"] = symbol_1d - sector_1d

            if len(sector_returns) >= 5:
                sector_5d = sector_returns.iloc[-5:].sum()
                result["vs_sector_5d"] = symbol_5d - sector_5d

        # Calculate relative strength score (0-100)
        # Higher score = stronger relative performance
        rs_5d = result["vs_nifty_5d"]
        if rs_5d > 0.02:  # Outperforming by >2%
            result["relative_strength_score"] = 85.0
        elif rs_5d > 0.01:
            result["relative_strength_score"] = 70.0
        elif rs_5d > 0:
            result["relative_strength_score"] = 60.0
        elif rs_5d > -0.01:
            result["relative_strength_score"] = 45.0
        else:
            result["relative_strength_score"] = 30.0

        return result

    def analyze_sector_rotation(
        self,
        sector_bars: pd.DataFrame,
        nifty_bars: pd.DataFrame,
    ) -> dict:
        """Analyze sector rotation patterns.

        Compares sector performance vs Nifty over 5 days to detect
        money flow direction.

        Args:
            sector_bars: Sector index OHLCV DataFrame.
            nifty_bars: Nifty 50 OHLCV DataFrame.

        Returns:
            Dictionary with sector rotation metrics.
        """
        result = {
            "sector_performance_5d": 0.0,
            "nifty_performance_5d": 0.0,
            "relative_performance": 0.0,
            "money_flow_direction": "neutral",
            "rotation_strength": 0.0,
            "sector_momentum": "neutral",
        }

        if sector_bars.empty or nifty_bars.empty:
            return result

        # Calculate 5-day performance
        sector_returns = self._calculate_returns(sector_bars)
        nifty_returns = self._calculate_returns(nifty_bars)

        if len(sector_returns) < 5 or len(nifty_returns) < 5:
            return result

        sector_5d = sector_returns.iloc[-5:].sum()
        nifty_5d = nifty_returns.iloc[-5:].sum()

        result["sector_performance_5d"] = sector_5d
        result["nifty_performance_5d"] = nifty_5d
        result["relative_performance"] = sector_5d - nifty_5d

        # Determine money flow direction
        relative_perf = result["relative_performance"]
        if relative_perf > 0.01:
            result["money_flow_direction"] = "into_sector"
            result["rotation_strength"] = min(100.0, relative_perf * 100 * 5)
        elif relative_perf < -0.01:
            result["money_flow_direction"] = "out_of_sector"
            result["rotation_strength"] = min(100.0, abs(relative_perf) * 100 * 5)
        else:
            result["money_flow_direction"] = "neutral"
            result["rotation_strength"] = 0.0

        # Sector momentum
        if len(sector_returns) >= 10:
            recent_5d = sector_returns.iloc[-5:].sum()
            prev_5d = sector_returns.iloc[-10:-5].sum()

            if recent_5d > prev_5d * 1.5:
                result["sector_momentum"] = "accelerating"
            elif recent_5d < prev_5d * 0.5:
                result["sector_momentum"] = "decelerating"
            else:
                result["sector_momentum"] = "stable"

        return result

    def analyze_market_breadth(self) -> str:
        """Analyze overall market breadth.

        Simple placeholder for future implementation.
        Would require fetching advance-decline data for Nifty stocks.

        Returns:
            Market breadth indicator string.
        """
        # Placeholder - in production this would:
        # 1. Fetch advance-decline data for Nifty 50
        # 2. Calculate advancing vs declining issues
        # 3. Return "bullish", "bearish", or "neutral"

        logger.debug("Market breadth analysis - placeholder implementation")
        return "neutral"

    @graceful_degrade(
        default_return=(
            AnalysisComponent(
                name="correlation",
                score=50.0,
                weight=0.15,
                confidence=30.0,
                reasoning="Correlation analysis failed - using default neutral score",
                data={"diversified": True, "score": 50},
            ),
            CorrelationData()
        )
    )
    def analyze(
        self,
        symbol: str,
        underlying_key: str,
        bars_dict: dict[str, pd.DataFrame],
        signal_direction: str,
    ) -> tuple[AnalysisComponent, CorrelationData]:
        """Run full correlation analysis.

        Args:
            symbol: Stock symbol.
            underlying_key: Upstox instrument key.
            bars_dict: Dictionary of timeframe -> DataFrame.
            signal_direction: "BUY" or "SELL".

        Returns:
            Tuple of (AnalysisComponent, CorrelationData).
        """
        # Run index correlation analysis
        component = self.analyze_index_correlation(
            underlying_key, bars_dict, signal_direction
        )

        # Extract CorrelationData from component
        corr_data = CorrelationData(**component.data)

        # Update correlation data with additional analysis if data fetcher available
        if self._data_fetcher:
            # Get symbol bars
            symbol_df = bars_dict.get("5min") or bars_dict.get("15min")

            if symbol_df is not None and not symbol_df.empty:
                # Fetch Nifty data for relative strength
                nifty_df = self._fetch_index_data(NIFTY_50_KEY, days=10)

                if not nifty_df.empty:
                    # Calculate relative strength
                    rs_metrics = self.analyze_relative_strength(
                        symbol, symbol_df, nifty_df, None
                    )
                    corr_data.relative_strength_vs_nifty = rs_metrics.get("vs_nifty_5d", 0.0)

                    # Sector rotation analysis
                    sector_key = self._get_sector_index(symbol)
                    if sector_key:
                        sector_df = self._fetch_index_data(sector_key, days=10)
                        if not sector_df.empty:
                            rotation = self.analyze_sector_rotation(sector_df, nifty_df)
                            corr_data.sector_performance_5d = rotation.get("sector_performance_5d", 0.0)

        # Market breadth
        corr_data.market_breadth = self.analyze_market_breadth()

        return component, corr_data
