"""Sector rotation rank: relative strength of each sector vs NIFTY 50.

Computes RS scores and daily ranks for all active NSE sector indices
(Nifty Bank, IT, Auto, Pharma, Metal, FMCG, Energy, Realty, Fin Service)
relative to the NIFTY 50 benchmark.

Results are persisted to ``sector_rank_daily`` (via SQLAlchemy sync engine).

Expected ``historical_provider`` callable signature::

    historical_provider(instrument_key: str, interval: str, days: int) -> pd.DataFrame

The returned DataFrame must have a DatetimeIndex (or integer index) and
columns: open, high, low, close, volume (prices in paisa, integers).

All timestamps are in IST (Asia/Kolkata).
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Callable

import numpy as np
import pandas as pd
from sqlalchemy import text

logger = logging.getLogger(__name__)

# Benchmark used for relative strength computation.
_BENCHMARK_KEY = "NSE_INDEX|Nifty 50"

# Stock → sector symbol hard-coded mapping for top-10 universe.
_STOCK_TO_SECTOR: dict[str, str] = {
    # Banks
    "HDFCBANK": "NIFTY_BANK",
    "ICICIBANK": "NIFTY_BANK",
    "AXISBANK":  "NIFTY_BANK",
    "KOTAKBANK": "NIFTY_BANK",
    "SBIN":      "NIFTY_BANK",
    # IT
    "TCS":       "NIFTY_IT",
    "INFY":      "NIFTY_IT",
    # FMCG
    "HINDUNILVR": "NIFTY_FMCG",
    "HUL":        "NIFTY_FMCG",
    "ITC":        "NIFTY_FMCG",
    # Energy
    "RELIANCE":  "NIFTY_ENERGY",
}


class SectorRotationAnalyzer:
    """Ranks NSE sector indices daily by relative strength vs NIFTY 50.

    Args:
        instrument_manager: InstrumentManager instance (already loaded).
            Used for future extension; currently not required for DB lookups.
        historical_provider: Callable returning a bar DataFrame. See module
            docstring for the expected signature.
        db_engine: Optional SQLAlchemy sync Engine. Defaults to
            ``src.storage.db.get_sync_engine()`` when ``None``.
    """

    def __init__(
        self,
        instrument_manager,
        historical_provider: Callable[[str, str, int], pd.DataFrame],
        db_engine=None,
    ) -> None:
        self._im = instrument_manager
        self._hist = historical_provider
        if db_engine is None:
            from src.storage.db import get_sync_engine
            db_engine = get_sync_engine()
        self._engine = db_engine

    # ------------------------------------------------------------------
    # Sector catalogue helpers
    # ------------------------------------------------------------------

    def get_active_sectors(self) -> list[dict]:
        """Return all active sectors from the ``sector_indices`` table.

        Returns:
            List of dicts with keys: symbol, instrument_key, display_name,
            active, added_at.
        """
        sql = text(
            "SELECT symbol, instrument_key, display_name, active, added_at "
            "FROM sector_indices WHERE active = TRUE ORDER BY symbol"
        )
        with self._engine.connect() as conn:
            rows = conn.execute(sql).fetchall()
        return [
            {
                "symbol": r[0],
                "instrument_key": r[1],
                "display_name": r[2],
                "active": r[3],
                "added_at": r[4],
            }
            for r in rows
        ]

    def get_sector_instrument_keys(self) -> list[str]:
        """Return instrument keys for all active sector indices.

        Returns:
            List of Upstox instrument key strings.
        """
        return [s["instrument_key"] for s in self.get_active_sectors()]

    # ------------------------------------------------------------------
    # Return & RS computation
    # ------------------------------------------------------------------

    def compute_returns(
        self,
        instrument_key: str,
        lookback_days: int,
    ) -> float:
        """Compute total percentage return over *lookback_days* trading days.

        Fetches daily bars for the instrument and computes the return as:
        ``(close[-1] - close[0]) / close[0] * 100``

        Args:
            instrument_key: Upstox instrument identifier.
            lookback_days: Number of trading days for the return window.

        Returns:
            Total return in percentage (float). Returns 0.0 on insufficient data.
        """
        needed = lookback_days + 5
        df = self._hist(instrument_key, "day", needed)
        if df.empty or len(df) < 2:
            logger.warning("Insufficient data for %s to compute returns", instrument_key)
            return 0.0

        df_tail = df.tail(lookback_days + 1)
        if len(df_tail) < 2:
            return 0.0

        first_close = float(df_tail["close"].iloc[0])
        last_close = float(df_tail["close"].iloc[-1])
        if first_close <= 0:
            return 0.0

        return (last_close - first_close) / first_close * 100.0

    def compute_rs_score(
        self,
        sector_key: str,
        benchmark_key: str = _BENCHMARK_KEY,
        lookback: int = 20,
    ) -> float:
        """Compute relative strength of a sector vs the benchmark.

        RS score = sector_return(lookback) - benchmark_return(lookback)
        (in percentage points, so positive means outperformance).

        Args:
            sector_key: Upstox instrument key for the sector index.
            benchmark_key: Instrument key for the benchmark (default: Nifty 50).
            lookback: Number of trading days for return computation.

        Returns:
            RS score as a float (percentage points of outperformance).
        """
        sector_return = self.compute_returns(sector_key, lookback)
        benchmark_return = self.compute_returns(benchmark_key, lookback)
        return sector_return - benchmark_return

    # ------------------------------------------------------------------
    # Ranking
    # ------------------------------------------------------------------

    def rank_all(self, trade_date: date) -> pd.DataFrame:
        """Compute RS-based ranks for all active sectors and persist to DB.

        Computes 5-day return, 20-day return, RS score (20d), and then
        assigns ranks (rank 1 = highest RS = strongest sector).

        Also reads the previous day's rank to populate ``rank_change``.

        Args:
            trade_date: The trade date to tag all rows with.

        Returns:
            DataFrame with columns: sector_symbol, instrument_key,
            rs_score, return_pct_5d, return_pct_20d, rank, rank_change.
        """
        sectors = self.get_active_sectors()
        if not sectors:
            logger.warning("No active sectors in sector_indices table")
            return pd.DataFrame()

        records: list[dict] = []
        for s in sectors:
            sym = s["symbol"]
            key = s["instrument_key"]
            try:
                rs = self.compute_rs_score(key, lookback=20)
                ret5 = self.compute_returns(key, lookback_days=5)
                ret20 = self.compute_returns(key, lookback_days=20)
                records.append(
                    {
                        "sector_symbol": sym,
                        "instrument_key": key,
                        "rs_score": rs,
                        "return_pct_5d": ret5,
                        "return_pct_20d": ret20,
                    }
                )
                logger.debug("%s: rs=%.2f ret5=%.2f ret20=%.2f", sym, rs, ret5, ret20)
            except Exception as exc:
                logger.error("Error computing sector metrics for %s: %s", sym, exc)
                records.append(
                    {
                        "sector_symbol": sym,
                        "instrument_key": key,
                        "rs_score": 0.0,
                        "return_pct_5d": 0.0,
                        "return_pct_20d": 0.0,
                    }
                )

        df = pd.DataFrame(records)

        # Rank by RS score: rank 1 = highest RS = strongest
        df["rank"] = (
            df["rs_score"].rank(ascending=False, method="min").astype(int)
        )

        # Read previous day's ranks for rank_change
        prev_ranks = self._load_previous_ranks(trade_date)
        if prev_ranks:
            df["rank_change"] = df.apply(
                lambda row: (
                    prev_ranks.get(row["sector_symbol"], row["rank"]) - row["rank"]
                ),
                axis=1,
            )
        else:
            df["rank_change"] = None

        self._persist_ranks(df, trade_date)
        return df

    def _load_previous_ranks(self, trade_date: date) -> dict[str, int]:
        """Load rank values from the most recent prior date.

        Args:
            trade_date: Current trade date (rows strictly before this date
                are queried).

        Returns:
            Dict mapping sector_symbol → rank for the prior date.
            Empty dict if no prior data found.
        """
        sql = text(
            """
            SELECT sector_symbol, rank
            FROM sector_rank_daily
            WHERE trade_date = (
                SELECT MAX(trade_date)
                FROM sector_rank_daily
                WHERE trade_date < :td
            )
            """
        )
        try:
            with self._engine.connect() as conn:
                rows = conn.execute(sql, {"td": trade_date}).fetchall()
            return {r[0]: int(r[1]) for r in rows}
        except Exception as exc:
            logger.warning("Could not load previous ranks: %s", exc)
            return {}

    def _persist_ranks(self, df: pd.DataFrame, trade_date: date) -> None:
        """Upsert sector ranks into ``sector_rank_daily``.

        Args:
            df: DataFrame with columns sector_symbol, rs_score,
                return_pct_5d, return_pct_20d, rank, rank_change.
            trade_date: Trade date for all rows.
        """
        upsert = text(
            """
            INSERT INTO sector_rank_daily
              (trade_date, sector_symbol, rs_score, return_pct_5d, return_pct_20d,
               rank, rank_change)
            VALUES
              (:trade_date, :sector_symbol, :rs_score, :return_pct_5d,
               :return_pct_20d, :rank, :rank_change)
            ON CONFLICT (trade_date, sector_symbol) DO UPDATE SET
              rs_score       = EXCLUDED.rs_score,
              return_pct_5d  = EXCLUDED.return_pct_5d,
              return_pct_20d = EXCLUDED.return_pct_20d,
              rank           = EXCLUDED.rank,
              rank_change    = EXCLUDED.rank_change
            """
        )
        rows = []
        for _, row in df.iterrows():
            rank_change = row.get("rank_change")
            rows.append(
                {
                    "trade_date": trade_date,
                    "sector_symbol": row["sector_symbol"],
                    "rs_score": float(row["rs_score"]),
                    "return_pct_5d": float(row["return_pct_5d"]) if row["return_pct_5d"] is not None else None,
                    "return_pct_20d": float(row["return_pct_20d"]) if row["return_pct_20d"] is not None else None,
                    "rank": int(row["rank"]),
                    "rank_change": int(rank_change) if rank_change is not None and not (isinstance(rank_change, float) and np.isnan(rank_change)) else None,
                }
            )
        with self._engine.begin() as conn:
            conn.execute(upsert, rows)
        logger.info("Persisted %d sector ranks for %s", len(rows), trade_date)

    # ------------------------------------------------------------------
    # Read-back helpers
    # ------------------------------------------------------------------

    def get_today_ranking(self, trade_date: date) -> pd.DataFrame:
        """Read the sector ranking for a given trade date from the DB.

        Args:
            trade_date: Date to query.

        Returns:
            DataFrame with columns: sector_symbol, rs_score, return_pct_5d,
            return_pct_20d, rank, rank_change. Sorted by rank ascending.
            Empty DataFrame if no data found.
        """
        sql = text(
            """
            SELECT sector_symbol, rs_score, return_pct_5d, return_pct_20d,
                   rank, rank_change
            FROM sector_rank_daily
            WHERE trade_date = :td
            ORDER BY rank ASC
            """
        )
        with self._engine.connect() as conn:
            rows = conn.execute(sql, {"td": trade_date}).fetchall()
        if not rows:
            return pd.DataFrame(
                columns=[
                    "sector_symbol", "rs_score", "return_pct_5d",
                    "return_pct_20d", "rank", "rank_change",
                ]
            )
        return pd.DataFrame(rows, columns=[
            "sector_symbol", "rs_score", "return_pct_5d",
            "return_pct_20d", "rank", "rank_change",
        ])

    def is_top_quartile(self, sector_symbol: str, trade_date: date) -> bool:
        """Return True when the sector ranks in the top 25% on *trade_date*.

        With 9 sectors the top quartile is the top 2 or 3 by rank (rank <= 2
        for strict top-25%, rank <= ceil(9*0.25) = 3 for floor-rounding).
        We use ``rank <= ceil(n * 0.25)`` so at least 1 sector is always
        eligible.

        Args:
            sector_symbol: Sector symbol (e.g. ``'NIFTY_BANK'``).
            trade_date: Date to evaluate.

        Returns:
            ``True`` when the sector is in the top quartile.
        """
        df = self.get_today_ranking(trade_date)
        if df.empty:
            return False
        n = len(df)
        cutoff = max(1, -(-n // 4))  # ceiling division: ceil(n/4)
        row = df[df["sector_symbol"] == sector_symbol]
        if row.empty:
            return False
        return int(row.iloc[0]["rank"]) <= cutoff

    # ------------------------------------------------------------------
    # Stock → sector mapping
    # ------------------------------------------------------------------

    def get_sector_for_stock(self, stock_symbol: str) -> str | None:
        """Return the sector index symbol for a given stock symbol.

        Hard-coded mapping for the top-10 NSE equity universe:
        - HDFCBANK, ICICIBANK, AXISBANK, KOTAKBANK, SBIN → NIFTY_BANK
        - TCS, INFY → NIFTY_IT
        - HINDUNILVR (HUL), ITC → NIFTY_FMCG
        - RELIANCE → NIFTY_ENERGY

        Args:
            stock_symbol: NSE trading symbol (e.g. ``'RELIANCE'``).

        Returns:
            Sector symbol string or ``None`` if not in the mapping.
        """
        return _STOCK_TO_SECTOR.get(stock_symbol.upper())
