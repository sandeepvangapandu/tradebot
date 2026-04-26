"""Technical indicator engine using pandas-ta.

Computes technical indicators on OHLCV data with incremental updates.
All prices are in PAISA (integer) - 1 Rupee = 100 paisa.
All timestamps in IST (Asia/Kolkata).
"""

from dataclasses import dataclass
from typing import Optional
from zoneinfo import ZoneInfo

import pandas as pd
import pandas_ta as ta
from loguru import logger

IST = ZoneInfo("Asia/Kolkata")


@dataclass
class IndicatorResult:
    """Result container for indicator calculations."""

    name: str
    data: pd.Series
    columns: list[str]


class IndicatorEngine:
    """Engine for computing technical indicators on OHLCV data.

    Uses pandas-ta for calculations. Supports incremental computation
    on the tail of DataFrames for performance.

    Args:
        df: DataFrame with columns [open, high, low, close, volume]
            Prices should be in PAISA (integer).
    """

    REQUIRED_COLUMNS = ["open", "high", "low", "close", "volume"]

    def __init__(self, df: Optional[pd.DataFrame] = None):
        """Initialize the indicator engine.

        Args:
            df: Optional initial OHLCV DataFrame.
        """
        self._df = df.copy() if df is not None else None
        self._cache: dict[str, pd.DataFrame] = {}
        self._last_computed_index: int = 0
        # Memoization for indicator results. Keyed on
        # (method_name, params_tuple). Invalidated by self._data_version
        # which bumps whenever the underlying df changes.
        self._result_cache: dict[tuple, tuple[int, object]] = {}
        self._data_version: int = 0

        if self._df is not None:
            self._validate_dataframe(self._df)
            self._ensure_datetime_index()
            self._data_version = 1

    def _cached(self, key: tuple, compute):
        """Return cached result for `key` or compute and store.

        Cache invalidates when self._data_version changes (i.e. df updated).
        """
        hit = self._result_cache.get(key)
        if hit is not None and hit[0] == self._data_version:
            return hit[1]
        value = compute()
        self._result_cache[key] = (self._data_version, value)
        return value

    def _validate_dataframe(self, df: pd.DataFrame) -> None:
        """Validate that DataFrame has required columns."""
        missing = [col for col in self.REQUIRED_COLUMNS if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    def _ensure_datetime_index(self) -> None:
        """Ensure DataFrame index is timezone-aware datetime in IST."""
        if self._df is None:
            return

        if not isinstance(self._df.index, pd.DatetimeIndex):
            self._df.index = pd.to_datetime(self._df.index)

        if self._df.index.tz is None:
            self._df.index = self._df.index.tz_localize(IST)
        elif str(self._df.index.tz) != "Asia/Kolkata":
            self._df.index = self._df.index.tz_convert(IST)

    def update(self, df: pd.DataFrame) -> "IndicatorEngine":
        """Update the engine with new data.

        Fast path: if `df` has the same length and last-index timestamp as
        the cached frame, treat it as identical and skip the concat/sort
        (the caller in conditions.py routinely passes the same window).

        Args:
            df: New OHLCV DataFrame to replace or append.

        Returns:
            Self for method chaining.
        """
        self._validate_dataframe(df)

        if self._df is None:
            self._df = df.copy()
            self._ensure_datetime_index()
            self._last_computed_index = len(self._df)
            self._data_version += 1
            return self

        # Fast path: same window — no-op, keep cache hot.
        if (
            len(df) == len(self._df)
            and len(df) > 0
            and df.index[-1] == self._df.index[-1]
        ):
            return self

        # Incremental append: if `df` extends our existing frame (same last
        # bar present at same position, more rows after), append only the
        # tail. Avoids the O(N) concat+sort+drop_duplicates cost on every
        # bar in the backtest hot path.
        if (
            len(df) > len(self._df)
            and len(self._df) > 0
            and len(df) > 0
        ):
            anchor = self._df.index[-1]
            if anchor in df.index:
                pos = df.index.get_loc(anchor)
                # Confirm the prefix matches before fast-appending.
                if isinstance(pos, int) and pos == len(self._df) - 1:
                    tail = df.iloc[len(self._df):]
                    if len(tail) > 0:
                        self._df = pd.concat([self._df, tail])
                    self._ensure_datetime_index()
                    self._last_computed_index = len(self._df)
                    self._data_version += 1
                    return self

        # Fallback: full concat (handles re-bases / out-of-order updates).
        self._df = pd.concat([self._df, df]).drop_duplicates()
        self._df = self._df.sort_index()

        self._ensure_datetime_index()
        self._last_computed_index = len(self._df)
        self._data_version += 1
        return self

    def get_data(self) -> pd.DataFrame:
        """Get the underlying OHLCV data.

        Returns:
            DataFrame with OHLCV columns.
        """
        if self._df is None:
            raise ValueError("No data loaded in engine")
        return self._df.copy()

    def ema(self, period: int = 20) -> pd.Series:
        """Calculate Exponential Moving Average.

        Args:
            period: EMA period. Default 20.

        Returns:
            Series with EMA values in PAISA.
        """
        if self._df is None:
            raise ValueError("No data loaded")
        return self._cached(
            ("ema", period),
            lambda: ta.ema(self._df["close"], length=period),
        )

    def sma(self, period: int = 20) -> pd.Series:
        """Calculate Simple Moving Average.

        Args:
            period: SMA period. Default 20.

        Returns:
            Series with SMA values in PAISA.
        """
        if self._df is None:
            raise ValueError("No data loaded")
        return self._cached(
            ("sma", period),
            lambda: ta.sma(self._df["close"], length=period),
        )

    def rsi(self, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index.

        Args:
            period: RSI period. Default 14.

        Returns:
            Series with RSI values (0-100).
        """
        if self._df is None:
            raise ValueError("No data loaded")
        return self._cached(
            ("rsi", period),
            lambda: ta.rsi(self._df["close"], length=period),
        )

    def macd(
        self, fast: int = 12, slow: int = 26, signal: int = 9
    ) -> dict[str, pd.Series]:
        """Calculate MACD (Moving Average Convergence Divergence).

        Args:
            fast: Fast EMA period. Default 12.
            slow: Slow EMA period. Default 26.
            signal: Signal line period. Default 9.

        Returns:
            Dictionary with keys: macd, signal, histogram.
            Values are in PAISA for macd/signal, PAISA for histogram.
        """
        if self._df is None:
            raise ValueError("No data loaded")

        def _compute():
            macd_result = ta.macd(
                self._df["close"], fast=fast, slow=slow, signal=signal
            )
            if macd_result is None:
                raise ValueError("MACD calculation failed")
            macd_col = f"MACD_{fast}_{slow}_{signal}"
            signal_col = f"MACDs_{fast}_{slow}_{signal}"
            hist_col = f"MACDh_{fast}_{slow}_{signal}"
            return {
                "macd": macd_result[macd_col],
                "signal": macd_result[signal_col],
                "histogram": macd_result[hist_col],
            }

        return self._cached(("macd", fast, slow, signal), _compute)

    def bbands(
        self, period: int = 20, std: float = 2.0
    ) -> dict[str, pd.Series]:
        """Calculate Bollinger Bands.

        Args:
            period: Moving average period. Default 20.
            std: Standard deviation multiplier. Default 2.0.

        Returns:
            Dictionary with keys: upper, middle, lower.
            Values are in PAISA.
        """
        if self._df is None:
            raise ValueError("No data loaded")

        def _compute():
            bbands_result = ta.bbands(self._df["close"], length=period, std=std)
            if bbands_result is None:
                raise ValueError("Bollinger Bands calculation failed")
            std_str = str(std).replace(".", "_")
            lower_col = f"BBL_{period}_{std_str}"
            middle_col = f"BBM_{period}_{std_str}"
            upper_col = f"BBU_{period}_{std_str}"
            return {
                "lower": bbands_result[lower_col],
                "middle": bbands_result[middle_col],
                "upper": bbands_result[upper_col],
            }

        return self._cached(("bbands", period, std), _compute)

    def atr(self, period: int = 14) -> pd.Series:
        """Calculate Average True Range.

        Args:
            period: ATR period. Default 14.

        Returns:
            Series with ATR values in PAISA.
        """
        if self._df is None:
            raise ValueError("No data loaded")
        return self._cached(
            ("atr", period),
            lambda: ta.atr(
                self._df["high"],
                self._df["low"],
                self._df["close"],
                length=period,
            ),
        )

    def supertrend(
        self, period: int = 7, multiplier: float = 3.0
    ) -> dict[str, pd.Series]:
        """Calculate SuperTrend indicator.

        Args:
            period: ATR period. Default 7.
            multiplier: ATR multiplier. Default 3.0.

        Returns:
            Dictionary with keys: supertrend, direction, long, short.
            - supertrend: SuperTrend line value in PAISA
            - direction: 1 for uptrend, -1 for downtrend
            - long: Long entry signal (boolean)
            - short: Short entry signal (boolean)
        """
        if self._df is None:
            raise ValueError("No data loaded")

        def _compute():
            st_result = ta.supertrend(
                self._df["high"],
                self._df["low"],
                self._df["close"],
                length=period,
                multiplier=multiplier,
            )
            if st_result is None:
                raise ValueError("SuperTrend calculation failed")
            mult_str = str(multiplier)
            supert_col = f"SUPERT_{period}_{mult_str}"
            direction_col = f"SUPERTd_{period}_{mult_str}"
            long_col = f"SUPERTl_{period}_{mult_str}"
            short_col = f"SUPERTs_{period}_{mult_str}"
            return {
                "supertrend": st_result[supert_col],
                "direction": st_result[direction_col],
                "long": st_result[long_col],
                "short": st_result[short_col],
            }

        return self._cached(("supertrend", period, multiplier), _compute)

    def vwap(self) -> pd.Series:
        """Calculate Volume Weighted Average Price (intraday).

        VWAP resets at the start of each trading day.
        For Indian markets, trading day starts at 9:15 AM IST.

        Returns:
            Series with VWAP values in PAISA.
        """
        if self._df is None:
            raise ValueError("No data loaded")

        def _compute():
            # Vectorized intraday VWAP. Groupby with cumsum is O(N), no
            # Python loop over days.
            df = self._df
            typical_price = (df["high"] + df["low"] + df["close"]) / 3
            tpv = typical_price * df["volume"]
            day_key = df.index.date
            cum_tpv = tpv.groupby(day_key).cumsum()
            cum_vol = df["volume"].groupby(day_key).cumsum()
            return cum_tpv / cum_vol.replace(0, float("nan"))

        return self._cached(("vwap",), _compute)

    def compute_all(
        self,
        ema_periods: Optional[list[int]] = None,
        sma_periods: Optional[list[int]] = None,
        rsi_period: int = 14,
        macd_params: Optional[dict] = None,
        bbands_params: Optional[dict] = None,
        atr_period: int = 14,
        supertrend_params: Optional[dict] = None,
        include_vwap: bool = True,
    ) -> pd.DataFrame:
        """Compute all requested indicators and return enriched DataFrame.

        Args:
            ema_periods: List of EMA periods to compute.
            sma_periods: List of SMA periods to compute.
            rsi_period: RSI period.
            macd_params: Dict with keys fast, slow, signal.
            bbands_params: Dict with keys period, std.
            atr_period: ATR period.
            supertrend_params: Dict with keys period, multiplier.
            include_vwap: Whether to include VWAP.

        Returns:
            DataFrame with original OHLCV plus all indicator columns.
        """
        if self._df is None:
            raise ValueError("No data loaded")

        result_df = self._df.copy()

        # EMAs
        if ema_periods:
            for period in ema_periods:
                result_df[f"ema_{period}"] = self.ema(period)

        # SMAs
        if sma_periods:
            for period in sma_periods:
                result_df[f"sma_{period}"] = self.sma(period)

        # RSI
        result_df[f"rsi_{rsi_period}"] = self.rsi(rsi_period)

        # MACD
        if macd_params is None:
            macd_params = {"fast": 12, "slow": 26, "signal": 9}
        macd_result = self.macd(**macd_params)
        result_df["macd"] = macd_result["macd"]
        result_df["macd_signal"] = macd_result["signal"]
        result_df["macd_histogram"] = macd_result["histogram"]

        # Bollinger Bands
        if bbands_params is None:
            bbands_params = {"period": 20, "std": 2.0}
        bb_result = self.bbands(**bbands_params)
        result_df["bb_upper"] = bb_result["upper"]
        result_df["bb_middle"] = bb_result["middle"]
        result_df["bb_lower"] = bb_result["lower"]

        # ATR
        result_df[f"atr_{atr_period}"] = self.atr(atr_period)

        # SuperTrend
        if supertrend_params is None:
            supertrend_params = {"period": 7, "multiplier": 3.0}
        st_result = self.supertrend(**supertrend_params)
        result_df["supertrend"] = st_result["supertrend"]
        result_df["supertrend_direction"] = st_result["direction"]
        result_df["supertrend_long"] = st_result["long"]
        result_df["supertrend_short"] = st_result["short"]

        # VWAP
        if include_vwap:
            result_df["vwap"] = self.vwap()

        return result_df

    def get_latest(self, indicator_name: str, lookback: int = 0) -> float:
        """Get the latest (or lookback) value of an indicator.

        Args:
            indicator_name: Name of the indicator column.
            lookback: Number of bars to look back (0 = current).

        Returns:
            The indicator value.
        """
        if self._df is None:
            raise ValueError("No data loaded")

        if indicator_name not in self._df.columns:
            raise ValueError(f"Indicator '{indicator_name}' not found")

        idx = -(lookback + 1)
        return self._df[indicator_name].iloc[idx]

    def crosses_above(
        self, series1: pd.Series, series2: pd.Series
    ) -> pd.Series:
        """Detect when series1 crosses above series2.

        Args:
            series1: First series (e.g., price).
            series2: Second series (e.g., moving average).

        Returns:
            Boolean series where True indicates cross above.
        """
        return (series1 > series2) & (series1.shift(1) <= series2.shift(1))

    def crosses_below(
        self, series1: pd.Series, series2: pd.Series
    ) -> pd.Series:
        """Detect when series1 crosses below series2.

        Args:
            series1: First series (e.g., price).
            series2: Second series (e.g., moving average).

        Returns:
            Boolean series where True indicates cross below.
        """
        return (series1 < series2) & (series1.shift(1) >= series2.shift(1))

    def adx(self, period: int = 14) -> pd.Series:
        """Calculate Average Directional Index (ADX).

        ADX measures trend strength regardless of direction.
        - ADX > 25: Trending market (good for trend-following)
        - ADX > 40: Strong trend (high conviction)
        - ADX < 20: Ranging/choppy market (avoid trend-following)

        Args:
            period: ADX period. Default 14.

        Returns:
            Series with ADX values (0-100).
        """
        if self._df is None:
            raise ValueError("No data loaded")

        result = ta.adx(
            self._df["high"],
            self._df["low"],
            self._df["close"],
            length=period,
        )

        if result is None:
            raise ValueError("ADX calculation failed")

        # pandas-ta returns DataFrame with ADX, DMP, DMN columns
        adx_col = f"ADX_{period}"
        return result[adx_col]

    def obv(self) -> pd.Series:
        """Calculate On-Balance Volume (OBV).

        OBV measures buying and selling pressure as a cumulative indicator
        that adds volume on up days and subtracts volume on down days.

        Returns:
            Series with OBV values.
        """
        if self._df is None:
            raise ValueError("No data loaded")

        result = ta.obv(self._df["close"], self._df["volume"])
        return result

    def volume_sma(self, period: int = 20) -> pd.Series:
        """Calculate Volume Simple Moving Average.

        Args:
            period: SMA period for volume. Default 20.

        Returns:
            Series with volume SMA values.
        """
        if self._df is None:
            raise ValueError("No data loaded")

        result = ta.sma(self._df["volume"], length=period)
        return result

    def calculate_cpr(self, daily_df: pd.DataFrame) -> dict[str, float]:
        """Calculate Central Pivot Range (CPR) from daily OHLC data.

        CPR consists of:
        - Pivot (P) = (High + Low + Close) / 3
        - BC (Bottom Central) = (High + Low) / 2
        - TC (Top Central) = (Pivot - BC) + Pivot = 2*Pivot - BC

        Also calculates classic support and resistance levels:
        - R1 = 2*Pivot - Low
        - R2 = Pivot + (High - Low)
        - R3 = High + 2*(Pivot - Low)
        - S1 = 2*Pivot - High
        - S2 = Pivot - (High - Low)
        - S3 = Low - 2*(High - Pivot)

        Args:
            daily_df: DataFrame with daily OHLC data. Must contain
                     'high', 'low', 'close' columns. Uses the last row
                     (most recent complete day) for calculations.

        Returns:
            Dictionary with CPR values and support/resistance levels:
            {
                'pivot': float,      # Central pivot
                'bc': float,         # Bottom central
                'tc': float,         # Top central
                'cpr_width': float,  # TC - BC (width of CPR)
                'cpr_width_pct': float,  # Width as % of pivot
                'r1': float,         # Resistance 1
                'r2': float,         # Resistance 2
                'r3': float,         # Resistance 3
                's1': float,         # Support 1
                's2': float,         # Support 2
                's3': float,         # Support 3
            }
            All values are in PAISA (same as input data).

        Raises:
            ValueError: If daily_df is empty or missing required columns.
        """
        if daily_df is None or len(daily_df) == 0:
            raise ValueError("Daily DataFrame cannot be empty")

        required_cols = ["high", "low", "close"]
        missing = [col for col in required_cols if col not in daily_df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Use the last row (most recent complete day)
        prev_day = daily_df.iloc[-1]
        high = float(prev_day["high"])
        low = float(prev_day["low"])
        close = float(prev_day["close"])

        # Calculate CPR components
        pivot = (high + low + close) / 3
        bc = (high + low) / 2
        tc = 2 * pivot - bc

        # Calculate CPR width
        cpr_width = abs(tc - bc)
        cpr_width_pct = (cpr_width / pivot) * 100 if pivot != 0 else 0

        # Calculate support and resistance levels
        r1 = 2 * pivot - low
        r2 = pivot + (high - low)
        r3 = high + 2 * (pivot - low)
        s1 = 2 * pivot - high
        s2 = pivot - (high - low)
        s3 = low - 2 * (high - pivot)

        return {
            "pivot": pivot,
            "bc": bc,
            "tc": tc,
            "cpr_width": cpr_width,
            "cpr_width_pct": cpr_width_pct,
            "r1": r1,
            "r2": r2,
            "r3": r3,
            "s1": s1,
            "s2": s2,
            "s3": s3,
        }

    def is_narrow_cpr(
        self, cpr_data: dict[str, float], threshold_pct: float = 0.4
    ) -> bool:
        """Check if CPR is narrow (width < threshold % of pivot).

        Narrow CPR indicates consolidation and higher probability of
        mean reversion when price is within the CPR value zone.

        Args:
            cpr_data: Dictionary containing CPR calculations from calculate_cpr().
            threshold_pct: Width threshold as percentage of pivot.
                          Default 0.4% (typical for BankNifty).

        Returns:
            True if CPR width is less than threshold percentage.
        """
        if "cpr_width_pct" not in cpr_data:
            raise ValueError("cpr_data must contain 'cpr_width_pct'")

        return cpr_data["cpr_width_pct"] < threshold_pct

    def is_price_in_cpr_value_zone(
        self, price: float, cpr_data: dict[str, float]
    ) -> bool:
        """Check if price is within the CPR value zone (between BC and TC).

        The value zone (between Bottom Central and Top Central) is considered
        the equilibrium area where mean reversion strategies work best.

        Args:
            price: Current price in PAISA.
            cpr_data: Dictionary containing CPR calculations.

        Returns:
            True if price is between BC and TC (inclusive).
        """
        bc = cpr_data.get("bc", 0)
        tc = cpr_data.get("tc", 0)

        # Handle case where TC < BC (rare but possible)
        lower = min(bc, tc)
        upper = max(bc, tc)

        return lower <= price <= upper

    def get_cpr_position(self, price: float, cpr_data: dict[str, float]) -> str:
        """Get the position of price relative to CPR levels.

        Args:
            price: Current price in PAISA.
            cpr_data: Dictionary containing CPR calculations.

        Returns:
            String indicating position: 'above_r3', 'r2_r3', 'r1_r2',
            'tc_r1', 'value_zone', 'bc_s1', 's1_s2', 's2_s3', 'below_s3'
        """
        pivot = cpr_data.get("pivot", 0)
        bc = cpr_data.get("bc", 0)
        tc = cpr_data.get("tc", 0)
        r1 = cpr_data.get("r1", 0)
        r2 = cpr_data.get("r2", 0)
        r3 = cpr_data.get("r3", 0)
        s1 = cpr_data.get("s1", 0)
        s2 = cpr_data.get("s2", 0)
        s3 = cpr_data.get("s3", 0)

        if price > r3:
            return "above_r3"
        elif price > r2:
            return "r2_r3"
        elif price > r1:
            return "r1_r2"
        elif price > tc:
            return "tc_r1"
        elif price >= bc:
            return "value_zone"
        elif price > s1:
            return "bc_s1"
        elif price > s2:
            return "s1_s2"
        elif price > s3:
            return "s2_s3"
        else:
            return "below_s3"

    def get_previous_day_high_low(self, df_daily: pd.DataFrame) -> tuple[float, float]:
        """Get previous day's high and low.

        Args:
            df_daily: Daily OHLCV DataFrame with at least 2 days of data.
                Should have columns: open, high, low, close, volume.

        Returns:
            (pdh, pdl) tuple in paisa. Returns (None, None) if insufficient data.
        """
        if df_daily is None or len(df_daily) < 2:
            return None, None

        # Get the second last row (previous trading day)
        prev_day = df_daily.iloc[-2]
        return float(prev_day['high']), float(prev_day['low'])

    def calculate_pdh_pdl_levels(
        self, df_daily: pd.DataFrame
    ) -> dict[str, float]:
        """Calculate PDH, PDL and extension levels.

        Calculates previous day high/low and extension levels based on
        the previous day's range. Used for breakout trading strategies.

        Args:
            df_daily: Daily OHLCV DataFrame with at least 2 days of data.
                Should have columns: open, high, low, close, volume.

        Returns:
            Dictionary with:
            - pdh: Previous day high (in paisa)
            - pdl: Previous day low (in paisa)
            - pd_range: PDH - PDL (in paisa)
            - pd_mid: Midpoint of PDH and PDL (in paisa)
            - extension_0.5: 0.5x range extension above PDH (in paisa)
            - extension_1.0: 1.0x range extension above PDH (in paisa)
            - extension_down_0.5: 0.5x range extension below PDL (in paisa)
            - extension_down_1.0: 1.0x range extension below PDL (in paisa)
            - range_pct: Previous day range as percentage of close

        Returns empty dict if insufficient data.
        """
        pdh, pdl = self.get_previous_day_high_low(df_daily)

        if pdh is None or pdl is None:
            return {}

        pd_range = pdh - pdl
        pd_mid = (pdh + pdl) / 2

        # Get previous day close for percentage calculation
        prev_close = float(df_daily.iloc[-2]['close'])
        range_pct = (pd_range / prev_close) * 100 if prev_close > 0 else 0

        return {
            "pdh": pdh,
            "pdl": pdl,
            "pd_range": pd_range,
            "pd_mid": pd_mid,
            "extension_0.5": pdh + (pd_range * 0.5),
            "extension_1.0": pdh + pd_range,
            "extension_down_0.5": pdl - (pd_range * 0.5),
            "extension_down_1.0": pdl - pd_range,
            "range_pct": range_pct,
        }

    def is_pdh_pdl_range_valid(
        self, df_daily: pd.DataFrame, min_range_pct: float = 0.5
    ) -> bool:
        """Check if the previous day's high-low range is valid for trading.

        Filters out days with extremely narrow ranges that may lead to
        false breakouts or whipsaws.

        Args:
            df_daily: Daily OHLCV DataFrame with at least 2 days of data.
            min_range_pct: Minimum range as percentage of previous close.
                Default 0.5%.

        Returns:
            True if PDH-PDL range is >= min_range_pct of previous close.
        """
        levels = self.calculate_pdh_pdl_levels(df_daily)

        if not levels:
            return False

        return levels.get("range_pct", 0) >= min_range_pct

    def calculate_gap(self, prev_close: float, current_open: float) -> dict:
        """Calculate gap percentage and direction.

        Used for gap fade strategies to identify small gaps (0.2-0.5%)
        that are likely to mean revert.

        Args:
            prev_close: Previous day's close price in PAISA.
            current_open: Current day's open price in PAISA.

        Returns:
            Dictionary with gap metrics:
            {
                "gap_pct": Gap as percentage of previous close,
                "gap_points": Absolute gap in points (PAISA),
                "direction": "up" | "down" | "flat",
                "is_fadeable": True if gap is between 0.2% and 0.5%
            }
        """
        if prev_close == 0:
            return {
                "gap_pct": 0.0,
                "gap_points": 0.0,
                "direction": "flat",
                "is_fadeable": False,
            }

        gap_points = current_open - prev_close
        gap_pct = (gap_points / prev_close) * 100

        # Determine direction
        if gap_pct > 0.05:
            direction = "up"
        elif gap_pct < -0.05:
            direction = "down"
        else:
            direction = "flat"

        # Gap is fadeable if between 0.2% and 0.5% (small gaps mean revert)
        abs_gap_pct = abs(gap_pct)
        is_fadeable = 0.2 <= abs_gap_pct <= 0.5

        return {
            "gap_pct": gap_pct,
            "gap_points": gap_points,
            "direction": direction,
            "is_fadeable": is_fadeable,
        }

    def get_first_n_minutes_candle(
        self, n: int = 5
    ) -> dict[str, float]:
        """Get aggregated candle for first N minutes of the day.

        Used for gap fade strategies to check if the first 5-min candle
        shows rejection of the gap (close lower than open for gap up,
        close higher than open for gap down).

        Args:
            n: Number of minutes to aggregate. Default 5.

        Returns:
            Dictionary with aggregated OHLCV data:
            {
                "open": First minute open price,
                "high": Highest price in first N minutes,
                "low": Lowest price in first N minutes,
                "close": Last minute close price,
                "volume": Total volume in first N minutes
            }

        Raises:
            ValueError: If no data loaded or insufficient bars.
        """
        if self._df is None:
            raise ValueError("No data loaded")

        if len(self._df) < n:
            raise ValueError(
                f"Insufficient data. Have {len(self._df)} bars, need {n}"
            )

        first_n = self._df.head(n)

        return {
            "open": float(first_n["open"].iloc[0]),
            "high": float(first_n["high"].max()),
            "low": float(first_n["low"].min()),
            "close": float(first_n["close"].iloc[-1]),
            "volume": int(first_n["volume"].sum()),
        }

    def check_gap_rejection(
        self,
        gap_data: dict,
        first_candle: dict[str, float],
    ) -> dict:
        """Check if the first candle shows rejection of the gap.

        For gap up fade: First candle should close lower than open
        For gap down fade: First candle should close higher than open

        Args:
            gap_data: Gap calculation result from calculate_gap().
            first_candle: First N-minute candle from get_first_n_minutes_candle().

        Returns:
            Dictionary with rejection analysis:
            {
                "rejection_type": "bullish" | "bearish" | "none",
                "is_rejected": True if gap is rejected,
                "rejection_strength": Percentage reversal from open to close,
                "body_pct": Body as percentage of range
            }
        """
        open_price = first_candle["open"]
        close_price = first_candle["close"]
        high = first_candle["high"]
        low = first_candle["low"]

        direction = gap_data.get("direction", "flat")

        # Calculate candle metrics
        body = abs(close_price - open_price)
        range_val = high - low if high != low else 1
        body_pct = (body / range_val) * 100 if range_val > 0 else 0

        # Check rejection based on gap direction
        if direction == "up":
            # Gap up: Look for bearish rejection (close < open)
            is_rejected = close_price < open_price
            rejection_type = "bearish" if is_rejected else "none"
            # Strength: how much of the gap was retraced
            rejection_strength = ((open_price - close_price) / range_val) * 100
        elif direction == "down":
            # Gap down: Look for bullish rejection (close > open)
            is_rejected = close_price > open_price
            rejection_type = "bullish" if is_rejected else "none"
            rejection_strength = ((close_price - open_price) / range_val) * 100
        else:
            is_rejected = False
            rejection_type = "none"
            rejection_strength = 0.0

        return {
            "rejection_type": rejection_type,
            "is_rejected": is_rejected,
            "rejection_strength": rejection_strength,
            "body_pct": body_pct,
        }

    def calculate_gap_fill_target(
        self,
        gap_data: dict,
        prev_close: float,
        fill_pct: float = 80.0,
    ) -> dict:
        """Calculate target for gap fill mean reversion.

        Args:
            gap_data: Gap calculation result from calculate_gap().
            prev_close: Previous day's close price in PAISA.
            fill_pct: Target percentage of gap to fill. Default 80%.

        Returns:
            Dictionary with target levels:
            {
                "target_price": Price target for 80% gap fill,
                "full_fill_price": Price for 100% gap fill (prev close),
                "target_points": Points to target from current,
                "risk_reward": Risk/reward ratio based on typical SL
            }
        """
        gap_pct = gap_data.get("gap_pct", 0)
        gap_points = gap_data.get("gap_points", 0)

        # Calculate target price (partial fill of the gap)
        if gap_pct > 0:
            # Gap up: Target is moving down toward prev close
            target_price = prev_close + (gap_points * (1 - fill_pct / 100))
        else:
            # Gap down: Target is moving up toward prev close
            target_price = prev_close - (abs(gap_points) * (1 - fill_pct / 100))

        # Points needed to reach target
        current = prev_close + gap_points  # Current is at gap open
        target_points = abs(current - target_price)

        # Estimate R:R (assuming 20% SL on options premium)
        # This is approximate - actual R:R depends on option pricing
        risk_reward = (fill_pct / 100) / 0.2  # Target % / SL %

        return {
            "target_price": target_price,
            "full_fill_price": prev_close,
            "target_points": target_points,
            "fill_pct": fill_pct,
            "risk_reward": risk_reward,
        }
