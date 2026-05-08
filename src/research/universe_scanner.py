"""Daily universe scanner: ranks Top 10 NSE equities by liquidity / volatility / RS.

Computes three daily ranking dimensions for the locked Top-10 universe:
  - Liquidity   : average daily value (ADV) and volume over a 20-day lookback.
  - Volatility  : ATR(14) as % of close + annualised realised volatility (20d).
  - Relative Strength: price return of symbol vs NIFTY 50 over 20 trading days.

A composite score is computed as:
  0.4 * (1 - rank_norm(liquidity_rank))
  + 0.3 * (1 - rank_norm(volatility_rank_inverse))
  + 0.3 * (1 - rank_norm(rs_rank))

Higher composite score means more attractive for trading.

Results are persisted to four Postgres tables (via src.storage.db.get_sync_engine):
  liquidity_rank_daily, volatility_rank_daily, rs_rank_daily, universe_daily_snapshot.

Expected historical_provider callable signature::

    historical_provider(instrument_key: str, interval: str, days: int) -> pd.DataFrame

The returned DataFrame must have a DatetimeIndex and columns:
  open, high, low, close, volume  (prices in paisa, integers).
"""

from __future__ import annotations

import math
from datetime import date, timedelta
from typing import Callable

import numpy as np
import pandas as pd
from loguru import logger
from sqlalchemy import text

# Indices always tracked but never ranked.
INDEX_KEYS: dict[str, str] = {
    "NIFTY":     "NSE_INDEX|Nifty 50",
    "BANKNIFTY": "NSE_INDEX|Nifty Bank",
}

_BENCHMARK_KEY = "NSE_INDEX|Nifty 50"


class UniverseScanner:
    """Ranks the Top-10 NSE equity universe daily and persists results.

    Args:
        instrument_manager: InstrumentManager instance (already loaded).
        historical_provider: Callable that returns a bar DataFrame for a given
            instrument_key and lookback. See module docstring for signature.
        db_engine: Optional SQLAlchemy sync Engine; defaults to
            ``src.storage.db.get_sync_engine()`` when not supplied.
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
    # Universe queries
    # ------------------------------------------------------------------

    def get_constituents(self, active_only: bool = True) -> list[dict]:
        """Return all universe constituents from the database.

        Args:
            active_only: When True (default) only return rows where active = TRUE.

        Returns:
            List of dicts with keys: symbol, instrument_key, segment, active, notes.
        """
        where = "WHERE active = TRUE" if active_only else ""
        sql = text(
            f"SELECT symbol, instrument_key, segment, active, notes "
            f"FROM universe_constituents {where} ORDER BY symbol"
        )
        with self._engine.connect() as conn:
            rows = conn.execute(sql).fetchall()
        return [
            {
                "symbol": r[0],
                "instrument_key": r[1],
                "segment": r[2],
                "active": r[3],
                "notes": r[4],
            }
            for r in rows
        ]

    def get_instrument_keys(self, include_indices: bool = True) -> list[str]:
        """Return instrument keys for all active constituents.

        Args:
            include_indices: When True (default) also appends the NIFTY and
                BANKNIFTY index keys at the end of the list.

        Returns:
            List of instrument key strings.
        """
        constituents = self.get_constituents(active_only=True)
        keys = [c["instrument_key"] for c in constituents]
        if include_indices:
            keys.extend(INDEX_KEYS.values())
        return keys

    # ------------------------------------------------------------------
    # Metric computation
    # ------------------------------------------------------------------

    def compute_liquidity(
        self,
        symbol: str,
        lookback_days: int = 20,
    ) -> tuple[int, int]:
        """Compute average daily value (ADV) and average daily volume.

        Fetches daily bars for the instrument and aggregates over the last
        ``lookback_days`` rows (which should correspond to ~20 trading days
        when the interval is 'day').

        Args:
            symbol: Trading symbol (e.g. 'RELIANCE').
            lookback_days: Number of trading days for the rolling average.

        Returns:
            Tuple of (adv_paisa, avg_daily_volume):
            - adv_paisa  : Average daily traded value in paisa (int).
            - avg_daily_volume: Average daily volume in shares (int).
        """
        instrument_key = self._get_instrument_key(symbol)
        df = self._hist(instrument_key, "day", lookback_days + 5)  # small buffer
        if df.empty or len(df) == 0:
            logger.warning("No bar data for {} to compute liquidity", symbol)
            return (0, 0)

        df = df.tail(lookback_days).copy()
        # Daily value = close (paisa) * volume
        df["daily_value"] = df["close"] * df["volume"]
        adv = int(df["daily_value"].mean())
        avg_vol = int(df["volume"].mean())
        return (adv, avg_vol)

    def compute_volatility(
        self,
        symbol: str,
        atr_period: int = 14,
        vol_lookback: int = 20,
    ) -> tuple[float, float]:
        """Compute ATR% and realised volatility.

        ATR is calculated using Wilder's smoothed ATR on daily bars.
        Realised vol is the annualised standard deviation of log returns
        over the last ``vol_lookback`` days.

        Args:
            symbol: Trading symbol (e.g. 'RELIANCE').
            atr_period: ATR period in days (default 14).
            vol_lookback: Lookback for realised vol (default 20).

        Returns:
            Tuple of (atr_pct, realized_vol_20d):
            - atr_pct : ATR(14) / close * 100  (float, percentage)
            - realized_vol_20d: Annualised stdev of log returns (float)
        """
        instrument_key = self._get_instrument_key(symbol)
        needed = max(atr_period, vol_lookback) + 5
        df = self._hist(instrument_key, "day", needed)
        if df.empty or len(df) < atr_period:
            logger.warning("Insufficient data for {} volatility calc", symbol)
            return (0.0, 0.0)

        # True range
        df = df.copy()
        df["prev_close"] = df["close"].shift(1)
        df["tr"] = df[["high", "low", "prev_close"]].apply(
            lambda r: max(
                r["high"] - r["low"],
                abs(r["high"] - (r["prev_close"] if not math.isnan(r["prev_close"]) else r["high"])),
                abs(r["low"] - (r["prev_close"] if not math.isnan(r["prev_close"]) else r["low"])),
            ),
            axis=1,
        )

        # Wilder smoothed ATR (EWM with alpha = 1/atr_period)
        atr_series = df["tr"].ewm(alpha=1.0 / atr_period, adjust=False).mean()
        latest_atr = float(atr_series.iloc[-1])
        latest_close = float(df["close"].iloc[-1])
        atr_pct = (latest_atr / latest_close * 100) if latest_close > 0 else 0.0

        # Realised vol: annualised stdev of log returns
        returns_df = df.tail(vol_lookback + 1)
        if len(returns_df) >= 2:
            log_ret = np.log(returns_df["close"] / returns_df["close"].shift(1)).dropna()
            realized_vol = float(log_ret.std() * math.sqrt(252)) if len(log_ret) > 0 else 0.0
        else:
            realized_vol = 0.0

        return (atr_pct, realized_vol)

    def compute_relative_strength(
        self,
        symbol: str,
        benchmark_key: str = _BENCHMARK_KEY,
        lookback_days: int = 20,
    ) -> float:
        """Compute relative strength of symbol vs benchmark over lookback.

        RS score = (symbol_return_20d - benchmark_return_20d) * 100

        A positive score means the symbol outperformed the benchmark.

        Args:
            symbol: Trading symbol (e.g. 'RELIANCE').
            benchmark_key: Instrument key for the benchmark index.
            lookback_days: Number of trading days for the return window.

        Returns:
            RS score as a float (percentage points of outperformance).
        """
        instrument_key = self._get_instrument_key(symbol)
        needed = lookback_days + 5

        sym_df = self._hist(instrument_key, "day", needed)
        bm_df = self._hist(benchmark_key, "day", needed)

        if sym_df.empty or len(sym_df) < 2:
            logger.warning("Insufficient data for {} RS calc", symbol)
            return 0.0
        if bm_df.empty or len(bm_df) < 2:
            logger.warning("Insufficient benchmark data for RS calc")
            return 0.0

        sym_close = sym_df["close"].tail(lookback_days + 1)
        bm_close = bm_df["close"].tail(lookback_days + 1)

        sym_return = (float(sym_close.iloc[-1]) - float(sym_close.iloc[0])) / float(sym_close.iloc[0])
        bm_return = (float(bm_close.iloc[-1]) - float(bm_close.iloc[0])) / float(bm_close.iloc[0])

        return (sym_return - bm_return) * 100.0

    # ------------------------------------------------------------------
    # Ranking
    # ------------------------------------------------------------------

    def rank_all(self, trade_date: date) -> pd.DataFrame:
        """Compute and persist ranks for all active constituents.

        Fetches metrics for every active symbol, assigns ranks (lower rank =
        better for liquidity/RS; lower rank = worse vol so we invert), then
        writes to all four tables:
          - liquidity_rank_daily
          - volatility_rank_daily
          - rs_rank_daily
          - universe_daily_snapshot

        Args:
            trade_date: The trade date to tag all rows with.

        Returns:
            DataFrame with one row per symbol and columns:
            symbol, adv_paisa, avg_daily_volume, liquidity_rank,
            atr_pct, realized_vol_20d, volatility_rank,
            rs_score, rs_rank, composite_score.
        """
        constituents = self.get_constituents(active_only=True)
        if not constituents:
            logger.warning("No active constituents in universe_constituents table")
            return pd.DataFrame()

        records: list[dict] = []
        for c in constituents:
            sym = c["symbol"]
            try:
                adv, avg_vol = self.compute_liquidity(sym)
                atr_pct, rv = self.compute_volatility(sym)
                rs = self.compute_relative_strength(sym)
                records.append(
                    {
                        "symbol": sym,
                        "adv_paisa": adv,
                        "avg_daily_volume": avg_vol,
                        "atr_pct": atr_pct,
                        "realized_vol_20d": rv,
                        "rs_score": rs,
                    }
                )
                logger.debug(
                    "{}: adv={} avg_vol={} atr_pct={:.2f} rv={:.2f} rs={:.2f}",
                    sym, adv, avg_vol, atr_pct, rv, rs,
                )
            except Exception as exc:
                logger.error("Error computing metrics for {}: {}", sym, exc)
                records.append(
                    {
                        "symbol": sym,
                        "adv_paisa": 0,
                        "avg_daily_volume": 0,
                        "atr_pct": 0.0,
                        "realized_vol_20d": 0.0,
                        "rs_score": 0.0,
                    }
                )

        df = pd.DataFrame(records)
        n = len(df)

        # Liquidity rank: higher ADV = better = lower rank number
        df["liquidity_rank"] = df["adv_paisa"].rank(ascending=False, method="min").astype(int)

        # Volatility rank: lower ATR% = lower rank (calmer stocks preferred by
        # the composite; but vol is a signal dimension — higher vol rank means
        # more volatile, which we down-weight in composite).
        df["volatility_rank"] = df["atr_pct"].rank(ascending=True, method="min").astype(int)

        # RS rank: higher RS score = better = lower rank number
        df["rs_rank"] = df["rs_score"].rank(ascending=False, method="min").astype(int)

        # Composite score (higher = more attractive)
        # Normalise: rank_norm(r) = (r - 1) / (n - 1)  for r in [1, n]
        def rank_norm(series: pd.Series) -> pd.Series:
            if n <= 1:
                return pd.Series([0.0] * len(series), index=series.index)
            return (series - 1) / (n - 1)

        df["composite_score"] = (
            0.4 * (1 - rank_norm(df["liquidity_rank"]))
            + 0.3 * (1 - rank_norm(df["volatility_rank"]))
            + 0.3 * (1 - rank_norm(df["rs_rank"]))
        )

        self._persist_ranks(df, trade_date)
        return df

    def get_today_universe(self, trade_date: date) -> pd.DataFrame:
        """Read the universe snapshot for a given trade date.

        Args:
            trade_date: Date to query.

        Returns:
            DataFrame from universe_daily_snapshot ordered by composite_score DESC.
        """
        sql = text(
            """
            SELECT symbol, in_universe, liquidity_rank, volatility_rank,
                   rs_rank, composite_score, blackout, blackout_reason
            FROM universe_daily_snapshot
            WHERE trade_date = :td
            ORDER BY composite_score DESC NULLS LAST
            """
        )
        with self._engine.connect() as conn:
            rows = conn.execute(sql, {"td": trade_date}).fetchall()
        if not rows:
            return pd.DataFrame(
                columns=[
                    "symbol", "in_universe", "liquidity_rank", "volatility_rank",
                    "rs_rank", "composite_score", "blackout", "blackout_reason",
                ]
            )
        return pd.DataFrame(
            rows,
            columns=[
                "symbol", "in_universe", "liquidity_rank", "volatility_rank",
                "rs_rank", "composite_score", "blackout", "blackout_reason",
            ],
        )

    def is_blackout(self, symbol: str, trade_date: date) -> tuple[bool, str | None]:
        """Check if a symbol is blacked out for a given trade date.

        Reads from universe_daily_snapshot. Returns (False, None) if no
        snapshot row exists yet (Phase C has not run yet).

        Args:
            symbol: Trading symbol to check.
            trade_date: Date to check.

        Returns:
            Tuple (blackout: bool, blackout_reason: str | None).
        """
        sql = text(
            """
            SELECT blackout, blackout_reason
            FROM universe_daily_snapshot
            WHERE trade_date = :td AND symbol = :sym
            """
        )
        with self._engine.connect() as conn:
            row = conn.execute(sql, {"td": trade_date, "sym": symbol}).fetchone()
        if row is None:
            return (False, None)
        return (bool(row[0]), row[1])

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_instrument_key(self, symbol: str) -> str:
        """Look up instrument_key from universe_constituents table.

        Falls back to InstrumentManager search if not found in DB.

        Args:
            symbol: Trading symbol.

        Returns:
            Instrument key string.

        Raises:
            ValueError: If the symbol cannot be resolved.
        """
        sql = text(
            "SELECT instrument_key FROM universe_constituents WHERE symbol = :sym AND active = TRUE"
        )
        with self._engine.connect() as conn:
            row = conn.execute(sql, {"sym": symbol}).fetchone()
        if row is not None:
            return row[0]

        # Fallback: InstrumentManager
        df = self._im.search(segment="NSE_EQ", symbol=symbol)
        exact = df[df["trading_symbol"] == symbol]
        if not exact.empty:
            return str(exact.iloc[0]["instrument_key"])
        raise ValueError(f"Cannot resolve instrument_key for symbol: {symbol}")

    def _persist_ranks(self, df: pd.DataFrame, trade_date: date) -> None:
        """Upsert ranking rows into all four tables.

        Args:
            df: DataFrame produced by rank_all (one row per symbol).
            trade_date: Trade date for all rows.
        """
        with self._engine.begin() as conn:
            for _, row in df.iterrows():
                sym = row["symbol"]
                # liquidity_rank_daily
                conn.execute(
                    text(
                        """
                        INSERT INTO liquidity_rank_daily
                            (trade_date, symbol, adv_paisa, avg_daily_volume, rank)
                        VALUES (:td, :sym, :adv, :avg_vol, :rank)
                        ON CONFLICT (trade_date, symbol) DO UPDATE SET
                            adv_paisa = EXCLUDED.adv_paisa,
                            avg_daily_volume = EXCLUDED.avg_daily_volume,
                            rank = EXCLUDED.rank
                        """
                    ),
                    {
                        "td": trade_date,
                        "sym": sym,
                        "adv": int(row["adv_paisa"]),
                        "avg_vol": int(row["avg_daily_volume"]),
                        "rank": int(row["liquidity_rank"]),
                    },
                )
                # volatility_rank_daily
                conn.execute(
                    text(
                        """
                        INSERT INTO volatility_rank_daily
                            (trade_date, symbol, atr_pct, realized_vol_20d, rank)
                        VALUES (:td, :sym, :atr, :rv, :rank)
                        ON CONFLICT (trade_date, symbol) DO UPDATE SET
                            atr_pct = EXCLUDED.atr_pct,
                            realized_vol_20d = EXCLUDED.realized_vol_20d,
                            rank = EXCLUDED.rank
                        """
                    ),
                    {
                        "td": trade_date,
                        "sym": sym,
                        "atr": float(row["atr_pct"]),
                        "rv": float(row["realized_vol_20d"]),
                        "rank": int(row["volatility_rank"]),
                    },
                )
                # rs_rank_daily
                conn.execute(
                    text(
                        """
                        INSERT INTO rs_rank_daily
                            (trade_date, symbol, rs_score, rs_rank)
                        VALUES (:td, :sym, :rs, :rank)
                        ON CONFLICT (trade_date, symbol) DO UPDATE SET
                            rs_score = EXCLUDED.rs_score,
                            rs_rank = EXCLUDED.rs_rank
                        """
                    ),
                    {
                        "td": trade_date,
                        "sym": sym,
                        "rs": float(row["rs_score"]),
                        "rank": int(row["rs_rank"]),
                    },
                )
                # universe_daily_snapshot
                conn.execute(
                    text(
                        """
                        INSERT INTO universe_daily_snapshot
                            (trade_date, symbol, in_universe,
                             liquidity_rank, volatility_rank, rs_rank,
                             composite_score)
                        VALUES (:td, :sym, TRUE, :liq_r, :vol_r, :rs_r, :comp)
                        ON CONFLICT (trade_date, symbol) DO UPDATE SET
                            in_universe    = TRUE,
                            liquidity_rank = EXCLUDED.liquidity_rank,
                            volatility_rank= EXCLUDED.volatility_rank,
                            rs_rank        = EXCLUDED.rs_rank,
                            composite_score= EXCLUDED.composite_score
                        """
                    ),
                    {
                        "td": trade_date,
                        "sym": sym,
                        "liq_r": int(row["liquidity_rank"]),
                        "vol_r": int(row["volatility_rank"]),
                        "rs_r": int(row["rs_rank"]),
                        "comp": float(row["composite_score"]),
                    },
                )
        logger.info("Persisted {} symbol ranks for {}", len(df), trade_date)
