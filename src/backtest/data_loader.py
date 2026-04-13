"""Load and prepare historical OHLCV data for backtesting."""

from datetime import date
from pathlib import Path
from typing import Optional

import pandas as pd
from loguru import logger


class BacktestDataLoader:
    """Loads historical OHLCV data from CSV files or the database.

    All price columns are expected in paisa (integers).
    Timestamps are localized to IST (Asia/Kolkata).
    """

    IST = "Asia/Kolkata"

    def load_csv(
        self,
        file_path: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> pd.DataFrame:
        """Load OHLCV data from a CSV file.

        Args:
            file_path: Path to the CSV file.
            start_date: Optional start date filter (inclusive).
            end_date: Optional end date filter (inclusive).

        Returns:
            DataFrame with DatetimeIndex (IST) and OHLCV columns.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")

        # Auto-detect the datetime column (may be named "timestamp" or "datetime")
        peek = pd.read_csv(file_path, nrows=0)
        cols_lower = [c.lower() for c in peek.columns]
        if "timestamp" in cols_lower:
            time_col = peek.columns[cols_lower.index("timestamp")]
        elif "datetime" in cols_lower:
            time_col = peek.columns[cols_lower.index("datetime")]
        else:
            # Fall back: first column
            time_col = peek.columns[0]

        df = pd.read_csv(file_path, parse_dates=[time_col])
        df = df.set_index(time_col)
        df.index = pd.DatetimeIndex(df.index)
        df.index.name = "timestamp"

        if df.index.tz is None:
            df.index = df.index.tz_localize(self.IST)
        else:
            df.index = df.index.tz_convert(self.IST)

        # Ensure standard column names
        expected = ["open", "high", "low", "close", "volume"]
        df.columns = [c.lower() for c in df.columns]
        for col in expected:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        df = df[expected]

        # Date filtering
        if start_date:
            df = df[df.index.date >= start_date]
        if end_date:
            df = df[df.index.date <= end_date]

        df = df.sort_index()
        logger.info(
            f"Loaded {len(df)} bars from {path.name} "
            f"({df.index[0]} to {df.index[-1]})"
        )
        return df

    def resample(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """Resample OHLCV data to a larger timeframe.

        Args:
            df: Source DataFrame with OHLCV columns.
            timeframe: Target timeframe (e.g., '15min', '1h', '1d').

        Returns:
            Resampled DataFrame.
        """
        resampled = df.resample(timeframe).agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }).dropna()
        return resampled
